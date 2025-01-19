from __future__ import absolute_import, division, print_function
from six.moves import range
# -*- Mode: Python; c-basic-offset: 2; indent-tabs-mode: nil; tab-width: 8 -*-
#
# LIBTBX_SET_DISPATCHER_NAME cctbx.xfel.stripe_experiment
#
# Given an LCLS experiment results directory and a trial, group results by
# run group and then distrbute each run group's results into subgroups and run
# dials.combine_experiments (optionally with clustering and selecting clusters).
#
from dials.util import show_mail_on_error
from dxtbx.model import ExperimentList
from libtbx.phil import parse
from libtbx.utils import Sorry
from libtbx import easy_run
from xfel.util.dials_file_matcher import match_dials_files
from xfel.util.mp import mp_phil_str as multiprocessing_str
from xfel.util.mp import get_submit_command_chooser
import sys

import os, math
import six

multiprocessing_override_str = '''
mp {
  method = local
  use_mpi = False
  mpi_command = source
  mpi_option = ""
  local.include_mp_in_command = False
}
'''

striping_str = '''
striping {
  results_dir = None
    .type = path
    .help = "LCLS results directory containint runs starting with r."
  rungroup = None
    .type = int
    .multiple = True
    .help = "Selected rungroups to stripe. If None, all rungroups are accepted."
  run = None
    .type = str
    .multiple = True
    .help = "Selected runs to stripe. If None, all runs are accepted."
  trial = None
    .type = int
    .help = "Trial identifier for an XFEL GUI formatted processing trial."
  stripe = False
    .type = bool
    .help = "Enable to select results evenly spaced across each rungroup"
            "(stripes) as opposed to contiguous chunks."
  chunk_size = 1000
    .type = int(value_min=1)
    .help = "Maximum number of images per chunk or stripe."
  respect_rungroup_barriers = True
    .type = bool
    .help = "Enforce separation by rungroup at time of striping (default)."
            "Turn off to allow multiple rungroups to share a detector model."
  dry_run = False
    .type = bool
    .help = "Only set up jobs but do not execute them"
  output_folder = None
    .type = path
    .help = "Path for output data. If None, use current directory"
}
'''

combining_str = '''
combine_experiments {
  clustering {
    dendrogram = False
      .type = bool
      .help = "Overrides any multiprocessing parameters to allow interactive"
      .help = "run. Clustering dendrograms can only be displayed in this mode."
    }
  keep_integrated = False
    .type = bool
    .help = "Combine refined.expt and integrated.refl files."
    .help = "If False, ignore integrated.refl files in favor of"
    .help = "indexed.refl files in preparation for reintegrating."
  include scope dials.command_line.combine_experiments.phil_scope
}
'''

combining_override_str = '''
combine_experiments {
  output {
    experiments_filename = FILENAME_combined.expt
    reflections_filename = FILENAME_combined.refl
    delete_shoeboxes = False
  }
  reference_from_experiment {
    detector = 0
  }
  clustering {
    use = True
  }
}
'''

# future feature: filter experiments by rmsd after combining/clustering
filtering_str = '''
filtering {
  enable = False
}
'''

refinement_str = '''
refinement {
  include scope dials.command_line.refine.phil_scope
  input {
    experiments = None
    reflections = None
  }
}
'''

refinement_override_str = '''
refinement {
  output {
    experiments = FILENAME_refined_CLUSTER.expt
    reflections = FILENAME_refined_CLUSTER.refl
    include_unused_reflections = False
    log = FILENAME_refine_CLUSTER.log
    debug_log = FILENAME_refine_CLUSTER.debug.log
  }
  refinement {
    parameterisation {
      auto_reduction {
        action = remove
      }
      beam {
        fix = all
      }
    }
    refinery {
      engine = SparseLevMar
    }
    reflections {
      outlier {
        algorithm = sauter_poon
        minimum_number_of_reflections = 3
        separate_experiments = False
        separate_panels = False
      }
    }
  }
  input {
    experiments = FILENAME_combined_CLUSTER.expt
    reflections = FILENAME_combined_CLUSTER.refl
  }
}
'''

recompute_mosaicity_str = '''
recompute_mosaicity {
  include scope xfel.command_line.recompute_mosaicity.phil_scope
  input {
    experiments = None
    reflections = None
  }
}
'''

recompute_mosaicity_override_str = '''
recompute_mosaicity {
  input {
    experiments = FILENAME_refined_CLUSTER.expt
    reflections = FILENAME_refined_CLUSTER.refl
  }
  output {
    experiments = FILENAME_refined_CLUSTER.expt
    reflections = FILENAME_refined_CLUSTER.refl
  }
}
'''


# reintegration after dials refinement
reintegration_str = '''
reintegration {
  enable = True
  include scope xfel.merging.command_line.mpi_integrate.phil_scope
  input {
    experiments = None
    reflections = None
  }
}
'''

reintegration_override_str = '''
reintegration{
  dispatch {
    step_list = input balance integrate
  }
  output {
    prefix = FILENAME_reintegrated_CLUSTER
    save_experiments_and_reflections = True
  }
  input {
    path = .
    experiments_suffix = FILENAME_refined_CLUSTER.expt
    reflections_suffix = FILENAME_refined_CLUSTER.refl
  }
}
'''

# split results and coerce to integration pickle for merging
postprocessing_str = '''
postprocessing {
  enable = False
  include scope xfel.command_line.frame_extractor.phil_scope
}
'''

postprocessing_override_str = """
postprocessing {
  input {
    experiments = FILENAME_reintegrated_CLUSTER*.expt
    reflections = FILENAME_reintegrated_CLUSTER*.refl
  }
  output {
    filename = FILENAME_CLUSTER_ITER_extracted.refl
    dirname = %s
  }
}
"""

master_defaults_str = multiprocessing_str + striping_str + combining_str + filtering_str + \
                        refinement_str + recompute_mosaicity_str + reintegration_str + postprocessing_str

# initialize a master scope from the multiprocessing phil string
master_defaults_scope = parse(master_defaults_str, process_includes=True)
# update master scope with customized and local phil scopes
phil_scope = master_defaults_scope.fetch(parse(postprocessing_override_str, process_includes=True))
phil_scope = phil_scope.fetch(parse(reintegration_override_str, process_includes=True))
phil_scope = phil_scope.fetch(parse(recompute_mosaicity_override_str, process_includes=True))
phil_scope = phil_scope.fetch(parse(refinement_override_str, process_includes=True))
phil_scope = phil_scope.fetch(parse(combining_override_str, process_includes=True))
phil_scope = phil_scope.fetch(parse(multiprocessing_override_str, process_includes=True))

helpstring = """cctbx.xfel.stripe_experiment: parallel processing of an XFEL UI-generated trial.

usage: cctbx.xfel.stripe_experiment striping.results_dir=/path/to/results striping.trial=000

for interactive unit cell clustering, use combine_experiments.clustering.dendrogram=True
"""


def chunk_pairs(expt_paths, refl_paths, max_size=1000):
  """Distribute matching expt-refl pairs into chunks with < `max_size` expts"""
  expt_lengths = [len(ExperimentList.from_file(expt_path, check_format=False))
                  for expt_path in expt_paths]
  chunk_count = math.ceil(sum(expt_lengths) / max_size)
  estimated_fill = sum(expt_lengths) / chunk_count
  chunks_indices = [[] for _ in range(chunk_count)]
  chunk_lengths = [0] * chunk_count
  currently_filled_chunk = 0
  for len_index, len_ in enumerate(expt_lengths):
    if len_ / 2 + chunk_lengths[currently_filled_chunk] > estimated_fill:
      currently_filled_chunk = min(chunk_count - 1, currently_filled_chunk + 1)
    chunks_indices[currently_filled_chunk].append(len_index)
    chunk_lengths[currently_filled_chunk] += len_
  chunked_expts, chunked_refls = [], []
  for chunk_indices in chunks_indices:
    chunked_expts.append([expt_paths[i] for i in chunk_indices])
    chunked_refls.append([refl_paths[i] for i in chunk_indices])
  return chunked_expts, chunked_refls, chunk_lengths


def stripe_pairs(expt_paths, refl_paths, max_size=1000):
  """Distribute matching expt-refl pairs into stripes with <`max_size` expts"""
  expt_lengths = [len(ExperimentList.from_file(expt_path, check_format=False))
                  for expt_path in expt_paths]
  stripe_count = math.ceil(sum(expt_lengths) / max_size)
  stripe_indices = [[] for _ in range(stripe_count)]
  stripe_lengths = [0, ] * stripe_count
  for len_index, len_ in enumerate(expt_lengths):
    currently_filled_stripe = stripe_lengths.index(min(stripe_lengths))
    stripe_indices[currently_filled_stripe].append(len_index)
    stripe_lengths[currently_filled_stripe] += len_
  striped_expts, striped_refls = [], []
  for chunk_indices in stripe_indices:
    striped_expts.append([expt_paths[i] for i in chunk_indices])
    striped_refls.append([refl_paths[i] for i in chunk_indices])
  return striped_expts, striped_refls, stripe_lengths


def allocate_chunks(results_dir,
                    trial_no,
                    rgs_selected=None,
                    respect_rungroup_barriers=True,
                    runs_selected=None,
                    stripe=False,
                    max_size=1000,
                    integrated=False):
  refl_ending = "_integrated" if integrated else "_indexed"
  expt_ending = "_refined.expt"
  trial = "%03d" % trial_no
  print("processing trial %s" % trial)
  if rgs_selected:
    rg_condition = lambda rg: rg in rgs_selected
  else:
    rg_condition = lambda rg: True
  rgs = {} # rungroups and associated runs
  for run in os.listdir(results_dir):
    if runs_selected and run not in runs_selected:
      continue
    trgs = [trg for trg in os.listdir(os.path.join(results_dir, run))
            if (trg[:6] == trial + "_rg") and rg_condition(trg[-5:])]
    if not trgs:
      continue
    rungroups = set([n.split("_")[1] for n in trgs])
    for rg in rungroups:
      if rg not in rgs:
        rgs[rg] = [run]
      else:
        rgs[rg].append(run)
  batch_contents = {}
  if respect_rungroup_barriers:
    batchable = {rg:{rg:runs} for rg, runs in six.iteritems(rgs)}
  else:
    batchable = {"all":rgs}
  # for either grouping, iterate over the top level keys in batchable and
  # distribute the events within those "batches" in stripes or chunks
  extension = None
  for batch, rungroups in six.iteritems(batchable):
    rg_by_run = {}
    for rungroup, runs in six.iteritems(rungroups):
      for run in runs:
        rg_by_run[run] = rungroup
    n_img = 0
    batch_contents[batch] = []
    for run, rg in six.iteritems(rg_by_run):
      try:
        trg = trial + "_" + rg
        contents = sorted(os.listdir(os.path.join(results_dir, run, trg, "out")))
      except OSError:
        print("skipping run %s missing out directory" % run)
        continue
      abs_contents = [os.path.join(results_dir, run, trg, "out", c)
                      for c in contents]
      batch_contents[batch].extend(abs_contents)
      expts = [c for c in contents if c.endswith(expt_ending)]
      n_img += len(expts)
      if extension is None:
        if any(c.endswith(".mpack") for c in contents):
          extension = ".mpack"
        elif any(c.endswith(".refl") for c in contents):
          extension = ".refl"
        else:
         extension = ".pickle"
    if n_img == 0:
      print("no images found for %s" % batch)
      del batch_contents[batch]
      continue
  if len(batch_contents) == 0:
    raise Sorry("no DIALS integration results found.")
  refl_ending += extension
  batch_chunks = {}
  for batch in batchable:
    batch_chunks[batch] = []
    contents = batch_contents[batch]
    expts = sorted([c for c in contents if c.endswith(expt_ending)])
    refls = sorted([c for c in contents if c.endswith(refl_ending)])
    expts, refls = match_dials_files(expts, refls, expt_ending, refl_ending)
    pack_func = stripe_pairs if stripe else chunk_pairs
    expts_packs, refls_packs, pack_lengths = pack_func(expts, refls, max_size)
    for expts_pack, refls_pack in zip(expts_packs, refls_packs):
      batch_chunks[batch].append((expts_pack, refls_pack))
    r = '{} {} experiments from {} files in {} into {} {} with sizes = {}'
    print(r.format("Striped" if stripe else "Chunked", sum(pack_lengths),
                   len(expts), batch, len(pack_lengths),
                   "stripes" if stripe else "chunks", pack_lengths))
    return batch_chunks


def parse_retaining_scope(args, phil_scope=phil_scope):
  if "-c" in args:
    phil_scope.show(attributes_level=2)
    return
  file_phil = []
  cmdl_phil = []
  for arg in args:
    if os.path.isfile(arg):
      try:
        file_phil.append(parse(file_name=arg))
      except Exception as e:
        raise Sorry("Unrecognized file: %s" % arg)
    else:
      try:
        cmdl_phil.append(parse(arg))
      except Exception as e:
        raise Sorry("Unrecognized argument: %s" % arg)

  run_scope, unused1 = phil_scope.fetch(sources=file_phil, track_unused_definitions=True)
  run_scope, unused2 = run_scope.fetch(sources=cmdl_phil, track_unused_definitions=True)
  if any([unused1, unused2]):
    msg = "\n".join([str(loc) for loc in unused1 + unused2])
    raise Sorry("Unrecognized argument(s): " + msg)

  return run_scope

def script_to_expand_over_clusters(clustered_json_name,
                                   phil_template_name, command, location):
  """
  Write a bash script to find results of a clustering step and produce customized
  phils and commands to run with each of them. For example, run the command
  dials.refine ...cluster8.expt ...cluster8.refl ...cluster8.phil followed by
  dials.refine ...cluster9.expt ...cluster9.refl ...cluster9.phil.
  clustered_json_name, clustered_refl_name and phil_template_name must each
  contain an asterisk, and substitution in phil_template itself will occur at
  each instance of CLUSTER.
  """
  clj_part_first, clj_part_last = clustered_json_name.split("CLUSTER")
  clustered_template_name = clj_part_first + "*" + clj_part_last
  ph_part_first, ph_part_last = phil_template_name.split("CLUSTER")

  bash_str = '''
#! /bin/sh

for file in `ls {clname}`
  do export cluster=`echo $file | sed "s:{cljfirst}::; s:{cljlast}::"`
  export philname="{phfirst}${cluster}{phlast}"
  export outname=`echo $philname | sed "s:.phil:.out:"`
  sed "s:CLUSTER:${cluster}:g" {phtempl} > $philname
  {command} $philname > $outname
done
'''.format(clname=clustered_template_name, phtempl=phil_template_name,
           cljfirst=clj_part_first, cljlast=clj_part_last,
           phfirst=ph_part_first, phlast=ph_part_last,
           command=command, cluster="{cluster}")

  bash_name = "generator".join([ph_part_first, ph_part_last]).split(".phil")[0] + ".sh"
  with open(os.path.join(location, bash_name), "w") as script:
    script.write(bash_str)
  return bash_name

class Script(object):

  def __init__(self, args = None):
    '''Initialise the script.'''

    # The script usage
    self.master_defaults_scope = master_defaults_scope
    if args is None: args = sys.argv[1:]
    self.run_scope = parse_retaining_scope(args)
    self.diff_scope = self.master_defaults_scope.fetch_diff(self.run_scope)
    self.params = self.run_scope.extract()

    # Validation
    if self.params.reintegration.enable:
      if self.params.combine_experiments.output.delete_shoeboxes:
        raise Sorry("Keep shoeboxes during combine_experiments and joint refinement when reintegrating."+
          "Set combine_experiments.output.delete_shoeboxes = False when using reintegration.")

    # Setup
    self.clustering = self.params.combine_experiments.clustering.use

  def set_up_section(self, section_tag, dispatcher_name,
    clustering=False, custom_parts=None, lambda_diff_str=None):
    diff_str = self.diff_scope.get(section_tag).as_str().replace("FILENAME", self.filename)
    if lambda_diff_str is not None:
      diff_str = lambda_diff_str(diff_str)
    if not clustering:
      diff_str = diff_str.replace("_CLUSTER", "")
    diff_parts = diff_str.split("\n")[1:-2]
    if custom_parts is not None:
      for part in custom_parts:
        diff_parts.append(part)
    diff_str = "\n".join(diff_parts)
    phil_filename = "%s_%s_CLUSTER.phil" % (self.filename, section_tag) if clustering else \
      "%s_%s.phil" % (self.filename, section_tag)
    phil_path = os.path.join(self.params.striping.output_folder, self.intermediates, phil_filename)
    if os.path.isfile(phil_path):
      os.remove(phil_path)
    with open(phil_path, "w") as phil_outfile:
      phil_outfile.write(diff_str + "\n")
    if clustering:
      script = script_to_expand_over_clusters(
        self.params.refinement.input.experiments[0].replace("FILENAME", self.filename),
        phil_filename,
        dispatcher_name,
        self.intermediates)
      command = ". %s" % os.path.join(self.params.striping.output_folder, self.intermediates, script)
    else:
      command = "%s_phil=%s" % (dispatcher_name, phil_filename)
    self.argument_sequence.append(command)

  def run(self):
    '''Execute the script.'''
    runs = ["r%04d" % int(r) if r.isnumeric() else r for r in self.params.striping.run]
    if self.params.striping.run:
      print("processing runs " + ", ".join(runs))
    if self.params.striping.rungroup:
      print("processing rungroups " + ", ".join(["rg%03d" % rg for rg in self.params.striping.rungroup]))
    batch_chunks = allocate_chunks(self.params.striping.results_dir,
                                   self.params.striping.trial,
                                   rgs_selected=["rg%03d" % rg for rg in self.params.striping.rungroup],
                                   respect_rungroup_barriers=self.params.striping.respect_rungroup_barriers,
                                   runs_selected=runs,
                                   stripe=self.params.striping.stripe,
                                   max_size=self.params.striping.chunk_size,
                                   integrated=self.params.combine_experiments.keep_integrated)
    self.dirname = os.path.join(self.params.striping.output_folder, "combine_experiments_t%03d" % self.params.striping.trial)
    self.intermediates = os.path.join(self.dirname, "intermediates")
    self.extracted = os.path.join(self.dirname, "final_extracted")
    for d in self.dirname, self.intermediates, self.extracted:
      if not os.path.isdir(d):
        os.mkdir(d)
    if self.params.striping.output_folder is None:
      self.params.striping.output_folder = os.getcwd()
    tag = "stripe" if self.params.striping.stripe else "chunk"
    all_commands = []
    for batch, ch_list in six.iteritems(batch_chunks):
      for idx in range(len(ch_list)):
        chunk = ch_list[idx]

        # reset for this chunk/stripe
        self.filename = "t%03d_%s_%s%03d" % (self.params.striping.trial, batch, tag, idx)
        self.argument_sequence = []

        # set up the file containing input expts and refls (logging)
        chunk_path = os.path.join(self.params.striping.output_folder, self.intermediates, self.filename)
        if os.path.isfile(chunk_path):
          os.remove(chunk_path)
        with open(chunk_path, "w") as outfile:
          for i in (0, 1): # expts then refls
            outfile.write("\n".join(chunk[i]) + "\n")

        # set up the params for dials.combine_experiments
        custom_parts = ["  input {"]
        for expt_path in chunk[0]:
          custom_parts.append("    experiments = %s" % expt_path)
        for refl_path in chunk[1]:
          custom_parts.append("    reflections = %s" % refl_path)
        custom_parts.append("  }")
        self.set_up_section("combine_experiments", "combine_experiments",
          clustering=False, custom_parts=custom_parts)

        # refinement of the grouped experiments
        self.set_up_section("refinement", "refine",
          clustering=self.clustering)

        # refinement of the grouped experiments
        self.set_up_section("recompute_mosaicity", "recompute_mosaicity",
          clustering=self.clustering)

        # reintegration
        if self.params.reintegration.enable:
          self.set_up_section("reintegration", "integration", clustering=self.clustering)

        # extract results to integration pickles for merging
        if self.params.postprocessing.enable:
          pass # disabled
          #lambda_diff_str = lambda diff_str: (diff_str % \
          #  (os.path.join("..", "final_extracted"))).replace("ITER", "%04d")
          #self.set_up_section("postprocessing", "cctbx.xfel.frame_extractor",
          #  lambda_diff_str=lambda_diff_str, clustering=self.clustering)

        # submit queued job from appropriate directory
        os.chdir(self.intermediates)
        command = "cctbx.xfel.ensemble_refinement_pipeline " + " ".join(self.argument_sequence)

        if self.params.mp.method != 'shifter' and self.params.mp.mpi_command:
          command = "%s %s"%(self.params.mp.mpi_command, command)

        if self.params.combine_experiments.clustering.dendrogram:
          easy_run.fully_buffered(command).raise_if_errors().show_stdout()
        else:
          submit_folder = os.path.join(self.params.striping.output_folder, self.intermediates)
          submit_path = os.path.join(submit_folder, "combine_%s.sh" % self.filename)
          submit_command = get_submit_command_chooser(command, submit_path, self.intermediates, self.params.mp,
            log_name=os.path.splitext(os.path.basename(submit_path))[0] + ".out",
            err_name=os.path.splitext(os.path.basename(submit_path))[0] + ".err",
            root_dir = submit_folder)
          all_commands.append(submit_command)
          if not self.params.striping.dry_run:
            print("executing command: %s" % submit_command)
            try:
              easy_run.fully_buffered(submit_command).raise_if_errors().show_stdout()
            except Exception as e:
              if not "Warning: job being submitted without an AFS token." in str(e):
                raise e
    return all_commands

if __name__ == "__main__":
  import sys
  if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
    print(helpstring)
    exit()
  if "-c" in sys.argv[1:]:
    expert_level = int(sys.argv[sys.argv.index("-e") + 1]) if "-e" in sys.argv[1:] else 0
    attr_level = int(sys.argv[sys.argv.index("-a") + 1]) if "-a" in sys.argv[1:] else 0
    phil_scope.show(expert_level=expert_level, attributes_level=attr_level)
    with open("striping_defaults.phil", "w") as defaults:
      defaults.write(phil_scope.as_str())
    exit()
  with show_mail_on_error():
    script = Script()
    script.run()
