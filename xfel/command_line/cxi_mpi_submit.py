from __future__ import absolute_import, division, print_function
from six.moves import range
# -*- Mode: Python; c-basic-offset: 2; indent-tabs-mode: nil; tab-width: 8 -*-
#
# LIBTBX_SET_DISPATCHER_NAME cxi.mpi_submit
# LIBTBX_SET_DISPATCHER_NAME cctbx.xfel.mpi_submit
#
# Submit a cctbx.xfel processing job to the cluster using bsub, after first creating
# an appropiate trial directory in the requested location and copying and modifying
# the given psana config files and cctbx.xfel phil files
#
import os, sys, shutil
from libtbx.utils import Sorry
from libtbx.phil import parse
from libtbx import easy_run
from xfel.util.mp import mp_phil_str, get_submit_command_chooser

help_str = """
Use cxi.mpi_submit to submit a cctbx.xfel job to a cluster for analyzing
diffraction data.  cxi.mpi_submit will create a directory for
you in the location specified by output.output_dir using the run and trial
numbers specified.  If no trial number is specified, one will be chosen auto-
matically by examining the output directory.  The output directory will be
created if it doesn't exist.

Examples (LCLS, for non-LCLS see below):

cxi.mpi_submit input.cfg=cxi49812/thermo.cfg input.experiment=cxi49812 \\
  input.run_num=25 output.output_dir=\\
  /reg/d/psdm/cxi/cxi49812/results/username/results mp.nproc=100 mp.queue=psanaq

This will submit run 25 of experiment cxi49812 to the psana queue for
processing using the modules specified by cxi49812/thermo.cfg.

cxi.mpi_submit submit.phil

Use the phil parameters in submit.phil to control job submission.

Before the job is submitted, the following occurs:

Directory /reg/d/psdm/cxi/cxi49812/results/username/results is created if it
doesn't exist.

Directory /reg/d/psdm/cxi/cxi49812/results/username/results/r0025 is created if it
doesn't exist.

The run directory is searched for the next available trial ID, say 003.  That
directory is created.

Under /reg/d/psdm/cxi/cxi49812/results/username/results/r0025/003, the following is
created:

Directory stdout.  The log for the experiment will go here.
File psana_orig.cfg.  This is a verbatim copy of the input config file.
File psana.cfg.  This is a copy of the input config file, re-written to respect
the new paths for this trial.  Any parameters with _dirname are re-written
to include this trial directory and the associated directories are created. Any
phil files found are copied to the trial directory, renamed, and updated.  This
includes phil files included by copied phil files.

The final bsub command is saved in submit.sh for future use.

For processing non-LCLS data, a different dispatcher and any non-default queueing
and multiprocessing parameters should be explicitly requested. For example,
mp.env_script should be specified on any system where the interactive environment
is not propagated to the queued jobs.

Example for processing XFEL stills at SACLA:

cxi.mpi_submit /path/to/h5/runs/directory/266708-0/run*.h5 \\
  input.dispatcher=dials.stills_process output.output_dir=/path/to/results \\
  input.target=/path/to//stills_process.phil input.trial=0 \\
  mp.method=pbs mp.queue=psmall mp.nnodes=10 mp.nproc_per_node=28 \\
  mp.env_script=/path/to/setpaths.sh
"""


phil_str = '''
  dry_run = False
    .type = bool
    .help = If True, the program will create the trial directory but not submit the job, \
            and will show the command that would have been executed.
  input {
    experiment = None
      .type = str
      .help = Experiment identifier, e.g. cxi84914 (mandatory for xtc stream processing)
    run_num = None
      .type = str
      .help = Run number to process. Can be a string instead of a number.
    trial = None
      .type = int
      .help = Trial number for this job.  Leave blank to auto determine.
    rungroup = None
      .type = int
      .help = Optional. If used, will add _rgXXX to the trial path. Useful for organizing \
              runs with similar parameters into logical groupings.
    dispatcher = cctbx.xfel.xtc_process
      .type = str
      .help = Which program to run. cxi.xtc_process is for module only based processing, \
              such as mod_hitfind and LABELIT. cctbx.xfel.xtc_process uses the DIALS back \
              end but converts the raw data to CBF before processing it. Use \
              cctbx.xfel.process to use the XTC streams natively without CBF.
    target = None
      .type = path
      .help = Optional path to phil file with additional parameters to be run by the \
              processing program.
    locator = None
      .type = str
      .help = Locator file needed for cctbx.xfel.process and dials.stills_process to find \
              the XTC streams
    task = None
      .type = int
      .help = Optional task number, meaning the dispatcher is working on previously \
              processed data from this run. In this case, the trial folder is not \
              created and task is used as a folder within the trial folder to save \
              the output.
  }
  output {
    output_dir = "."
      .type = str
      .help = Directory for output files
    add_output_dir_option = True
      .type = bool
      .help = If True, include output.output_dir on the command line.
    split_logs = True
      .type = bool
      .help = Option to split error and log files into separate per process
  }
'''

phil_scope = parse(phil_str + mp_phil_str, process_includes=True)
mp_phil_scope = parse(mp_phil_str, process_includes=True)

def copy_wrapper(src, dst):
  # OsWrapper is a signleton => as long as it is constructed by Script() first,
  # repeated calls to the constructor will return the previously constructed
  # instance
  from xfel.util.sfapi_connector import OsWrapper
  os = OsWrapper()
  with os.open(src, "r") as f_in:
    with os.open(dst, "w") as f_out:
      f_out.write(f_in.read())

def copy_config(config, dest_dir, root_name, params, target_num):
  """ Copy a config file to a directory, and all of its referenced phil files. Recursively
  copies the files and changes the includes statements to match the new file names.
  @param config config file to copy
  @param dest_dir Full path to directory to copy file to
  @param root_name Name to give to copied file. Included phil files will be given
  this name + _N where N is incremented for each file included in the phil file
  @param params phil parameters
  @param target_num for target phil files found in config, latest number found
  """
  # OsWrapper is a signleton => as long as it is constructed by Script() first,
  # repeated calls to the constructor will return the previously constructed
  # instance
  from xfel.util.sfapi_connector import OsWrapper
  os = OsWrapper()
  # make a copy of the original cfg file
  # shutil.copy(config, os.path.join(dest_dir, "%s_orig.cfg"%root_name))
  copy_wrapper(config, os.path.join(dest_dir, "%s_orig.cfg"%root_name))
  config_path = os.path.join(dest_dir, "%s.cfg"%root_name)

  # Re-write the config file, changing paths to be relative to the trial directory.
  # Also copy and re-write included phil files, while updating the config file to
  # the new paths
  # Note, some legacy rewrites done by cxi.lsf are not included here.
  with os.open(config_path, 'w') as f:
    for line in open(config).readlines():
      if "[pyana]" in line:
        raise Sorry("Pyana not supported. Check your config file.")
      if "RUN_NO" in line:
        line = line.replace("RUN_NO", str(params.input.run_num))
      if "RUN_STR" in line:
        line = line.replace("RUN_STR", "r%04d"%(params.input.run_num))
      if "trial_id" in line:
        key, val = line.split("=")
        line = "%s= %d\n"%(key,params.input.trial)
      if "rungroup_id" in line:
        key, val = line.split("=")
        line = "%s= %s\n"%(key,params.input.rungroup) # None ok
      elif "_dirname" in line:
        key, val = line.split("=")
        val = os.path.join(dest_dir, os.path.basename(val.strip()))
        if not os.path.exists(val):
          os.mkdir(val)
        line = "%s= %s\n"%(key,val)
      elif "xtal_target" in line:
        key, val = line.split("=")
        val = val.strip()
        if not os.path.exists(val):
          raise Sorry("One of the xtal_target files in the cfg file doesn't exist: %s"%val)
        new_target = "params_%d"%target_num
        copy_target(val, dest_dir, new_target)
        target_num += 1
        line = "%s= %s.phil\n"%(key,os.path.join(dest_dir, new_target))
      f.write(line)
  # f.close()
  return target_num

def copy_target(target, dest_dir, root_name):
  """ Copy a phil file to a directory, and all of its included phil files. Recursively
  copies the files and changes the includes statements to match the new file names.
  @param target Phil file to copy
  @param dest_dir Full path to directory to copy file to
  @param root_name Name to give to copied file. Included phil files will be given
  this name + _N where N is incremented for each file included in the phil file
  """
  # OsWrapper is a signleton => as long as it is constructed by Script() first,
  # repeated calls to the constructor will return the previously constructed
  # instance
  from xfel.util.sfapi_connector import OsWrapper
  os = OsWrapper()
  # Each included phil file will be named root_name_N.phil where N is this number,
  # incremented for each included phil file
  num_sub_targets = 1
  with os.open(os.path.join(dest_dir, root_name + ".phil"), 'w') as f:
    for line in open(target).readlines():
      if "include" in line:
        inc_str, include_type, sub_target = line.strip().split() # Example: include file cxi-8.2.phil
        if include_type != 'file':
          raise Sorry("Include isn't a file") # FIXME look up what other values are possible here
        sub_target_root_name = "%s_%d"%(root_name, num_sub_targets)
        line = " ".join([inc_str, include_type, sub_target_root_name  + ".phil\n"])
        # recursive call to check for other included files
        copy_target(os.path.join(os.path.dirname(target), sub_target), dest_dir, sub_target_root_name)
        num_sub_targets += 1
      f.write(line)

def get_trialdir(output_dir, run_num, trial = None, rungroup = None, task = None):
    # OsWrapper is a signleton => as long as it is constructed by Script() first,
    # repeated calls to the constructor will return the previously constructed
    # instance
    from xfel.util.sfapi_connector import OsWrapper
    os = OsWrapper()

    try:
      rundir = os.path.join(output_dir, "r%04d"%int(run_num))
    except ValueError:
      rundir = os.path.join(output_dir, run_num)

    if not os.path.exists(output_dir):
       os.makedirs(output_dir)

    if not os.path.exists(rundir):
      os.mkdir(rundir)

    # If a trial number wasn't included, find the next available, up to 999 trials
    if trial is None:
      found_one = False
      for i in range(1000):
        trialdir = os.path.join(rundir, "%03d"%i)
        if rungroup is not None:
          trialdir += "_rg%03d"%rungroup
        if not os.path.exists(trialdir):
          found_one = True
          break
      if found_one:
        trial = i
      else:
        raise Sorry("All trial numbers in use")
    else:
      trialdir = os.path.join(rundir, "%03d"%trial)
      if rungroup is not None:
        trialdir += "_rg%03d"%rungroup
      if os.path.exists(trialdir) and task is None:
        raise Sorry("Trial %d already in use"%trial)

    if not os.path.exists(trialdir):
      os.mkdir(trialdir)

    if task is not None:
      trialdir = os.path.join(trialdir, "task%03d"%task)
      if os.path.exists(trialdir):
        raise Sorry("Task %d already exists"%task)
      os.mkdir(trialdir)
    return trial, trialdir

def get_submission_id(result, method):
  if method == "mpi" or method == "lsf":
    submission_id = None
    for line in result.stdout_lines:
      # example for lsf: 'Job <XXXXXX> is submitted to queue <YYYYYYY>.'
      if len(line.split()) < 2: continue
      s = line.split()[1].lstrip('<').rstrip('>')
      try:
        s = int(s)
      except ValueError:
        pass
      else:
        submission_id = str(s)
    print(submission_id)
    return submission_id
  elif method == 'pbs':
    submission_id = "".join(result.stdout_lines).strip()
    print(submission_id)
    return submission_id
  elif method == 'slurm' or method == "shifter":
    # Assuming that all shifter instances are running on NERSC (slurm) systems
    return result.stdout_lines[0].split()[-1].strip()
  elif method == 'sfapi':
    submission_id = str(result)
    return submission_id
  elif method == 'htcondor':
    return result.stdout_lines[-1].split()[-1].rstrip('.')
  elif method == 'sge':
    #  example for sge at Diamond: 'Your job JOB_ID ("JOB_NAME") has been submitted'
    line = "".join(result.stdout_lines).strip()
    s = line.split()
    submission_id = ''.join(i for i in s if i.isdigit())   # or just simply use s[2], the command used will extract the number from the "line"
    print (line)
    print('Submission', 'ID', 'is', submission_id)
    return submission_id

def do_submit(command, submit_path, stdoutdir, mp_params, log_name="log.out", err_name="log.err", job_name=None, dry_run=False):
  submit_command = get_submit_command_chooser(command,
                                              submit_path,
                                              stdoutdir,
                                              mp_params,
                                              log_name=log_name,
                                              err_name=err_name,
                                              job_name=job_name,
                                              )
  if mp_params.method in ['lsf', 'sge', 'pbs']:
    parts = submit_command.split(" ")
    script = open(parts.pop(-1), "rb")
    run_command = script.read().split(b"\n")[-2]
    command = " ".join(parts + [run_command.decode()])
  else:
    command = submit_command
  submit_command = str(submit_command) # unicode workaround

  if dry_run:
    print("Dry run: job not submitted. Trial directory created here:", os.path.dirname(submit_path))
    print("Execute this command to submit the job:")
    print(submit_command)
  elif mp_params.method == 'local':
    submission_id = os.fork()
    if submission_id > 0:
      return submission_id
    else:
      stdout = os.open(os.path.join(stdoutdir, 'submit.log'), os.O_WRONLY|os.O_CREAT|os.O_TRUNC); os.dup2(stdout, 1)
      stderr = os.open(os.path.join(stdoutdir, 'submit.err'), os.O_WRONLY|os.O_CREAT|os.O_TRUNC); os.dup2(stderr, 2)
      os.execv(command.split()[0], command.split())
  elif mp_params.method == 'sfapi':
    job_stript_path = submit_command # the SFAPI submit command is the location of the job script

    from sfapi_client              import Client
    from sfapi_client.compute      import Machine
    from xfel.util.sfapi_connector import KeyManager, LOGGER

    km = KeyManager()
    with Client(key=km.key) as client:
      perlmutter = client.compute(Machine.perlmutter)
      [job_script_remote] = perlmutter.ls(
          job_stript_path.replace("~/", km.home), directory=False
      )

      print(f"Jobscript is at: {job_script_remote}")
      job = perlmutter.submit_job(job_script_remote)
      print(f"Submitted_job: {job.jobid}")
      result = job.jobid
  else:
    try:
      result = easy_run.fully_buffered(command=submit_command)
      result.raise_if_errors()
    except Exception as e:
      if not "Warning: job being submitted without an AFS token." in str(e):
        raise e
  return get_submission_id(result, mp_params.method)

class Script(object):
  """ Script to submit XFEL data for processing"""
  def __init__(self):
    pass

  def run(self, argv = None):
    """ Set up run folder and submit the job. """
    if argv is None:
      argv = sys.argv[1:]

    if len(argv) == 0 or "-h" in argv or "--help" in argv or "-c" in argv:
      print(help_str)
      print("Showing phil parameters:")
      print(phil_scope.as_str(attributes_level = 2))
      return

    user_phil = []
    dispatcher_args = []
    for arg in argv:
      if arg.endswith(".h5"):
        dispatcher_args.append(arg)
      elif (os.path.isfile(arg)):
        try:
          user_phil.append(parse(file_name=arg))
        except Exception as e:
          if os.path.splitext(arg)[1] == ".phil": raise e
          dispatcher_args.append(arg)
      else:
        try:
          user_phil.append(parse(arg))
        except RuntimeError as e:
          dispatcher_args.append(arg)
    scope, unused = phil_scope.fetch(sources=user_phil, track_unused_definitions=True)
    params = scope.extract()
    dispatcher_args.extend(["%s=%s"%(u.path,u.object.words[0].value) for u in unused])

    # OsWrapper is a signleton => as long as it is constructed by Script()
    # first, repeated calls to the constructor will return the previously
    # constructed instance
    if params.mp.method == "sfapi":
      import logging
      from xfel.util.sfapi_connector import OsWrapper, OsSFAPI, LOGGER
      LOGGER.setLevel(logging.DEBUG)
      self.os = OsWrapper(backend=OsSFAPI())
    else:
      self.os = os

    assert params.input.run_num is not None
    if params.input.dispatcher in ["cxi.xtc_process", "cctbx.xfel.xtc_process"]:
      # processing XTC streams at LCLS -- dispatcher will locate raw data
      assert params.input.experiment is not None or params.input.locator is not None
      print("Submitting run %d of experiment %s"%(int(params.input.run_num), params.input.experiment))
    else:
      print("Submitting run %s"%(params.input.run_num))
    trial, trialdir = get_trialdir(params.output.output_dir, params.input.run_num, params.input.trial, params.input.rungroup, params.input.task)
    params.input.trial = trial
    print("Using trial", params.input.trial)

    # log file will live here
    stdoutdir = self.os.path.join(trialdir, "stdout")
    self.os.mkdir(stdoutdir)
    logging_str = ""
    if params.output.split_logs:# test parameter for split_log then open and close log file and loop over nprocs
      if params.mp.method=='shifter' and params.mp.shifter.staging=='DataWarp':
        bbdirstr = "${DW_JOB_STRIPED}/stdout"
        logging_str = "output.logging_dir=%s"%bbdirstr
      else:
        for i in range(params.mp.nproc):
          error_files = self.os.path.join(stdoutdir,"error_rank%04d.out"%i)
          log_files = self.os.path.join(stdoutdir,"log_rank%04d.out"%i)
          self.os.open(log_files,'a').close()
          self.os.open(error_files,'a').close()
        logging_str = "output.logging_dir=%s"%stdoutdir
    else:
      logging_str = ""

    # Copy any config or phil files specified
    target_num = 1
    has_config = False
    redone_args = []
    for arg in dispatcher_args:
      if not len(arg.split('=')) == 2:
        redone_args.append(arg)
        continue
      name, value = arg.split('=')

      if "cfg" in name and self.os.path.splitext(value)[1].lower() == ".cfg":
        cfg = value
        if not self.os.path.exists(cfg):
          raise Sorry("Config file doesn't exist: %s"%cfg)
        if has_config:
          raise Sorry("Multiple config files found")
        has_config = True
        target_num = copy_config(cfg, trialdir, "psana", params, target_num)
        redone_args.append("%s=%s"%(name, self.os.path.join(trialdir, "psana.cfg")))
      elif "target" in name or self.os.path.splitext(value)[1].lower() == ".phil":
        phil = value
        if not self.os.path.exists(phil):
          raise Sorry("Phil file doesn't exist: %s"%phil)
        copy_target(phil, trialdir, "params_%d"%target_num)
        redone_args.append("%s=%s"%(name, self.os.path.join(trialdir, "params_%d.phil"%target_num)))
        target_num += 1
      else:
        redone_args.append(arg)
    dispatcher_args = redone_args

    # If additional phil params are provided, copy them over too
    if params.input.target is not None:
      # don't use self.os => need to to check host
      if not os.path.exists(params.input.target):
        raise Sorry("Target file doesn't exist: %s"%params.input.target)
      copy_target(params.input.target, trialdir, "params_%d"%target_num)
      params.input.target = self.os.path.join(trialdir, "params_%d.phil"%target_num)
      target_num += 1

    # Some configs files will specify out_dirname. If not, we want to explicitly create one
    # so the dispatcher will have an output directory.
    output_dir = self.os.path.join(trialdir, "out")
    if not self.os.path.exists(output_dir):
      self.os.makedirs(output_dir)

    # Write out a script for submitting this job and submit it
    submit_path = self.os.path.join(trialdir, "submit.sh")

    extra_str = ""
    data_str = ""

    assert [params.input.locator, params.input.experiment].count(None) != 0
    if params.input.locator is not None:
      locator_file = self.os.path.join(trialdir, "data.loc")
      # shutil.copyfile(params.input.locator, locator_file)
      copy_wrapper(params.input.locator, locator_file)
      data_str += locator_file

    from xfel.ui import known_dials_dispatchers
    if params.input.dispatcher in known_dials_dispatchers:
      import importlib
      dispatcher_params = importlib.import_module(known_dials_dispatchers[params.input.dispatcher]).phil_scope.extract()
    else:
      dispatcher_params = None

    if params.input.experiment is None:
      if hasattr(dispatcher_params, 'input') and hasattr(dispatcher_params.input, 'trial'):
        assert hasattr(dispatcher_params.input, 'run_num')
        data_str += " input.trial=%s input.run_num=%s" % ( # pass along for logging
          params.input.trial, params.input.run_num)
    else:
      data_str += " input.trial=%s input.experiment=%s input.run_num=%s" % (
        params.input.trial, params.input.experiment, params.input.run_num)

    for arg in dispatcher_args:
      extra_str += " %s" % arg
    if params.input.target is not None:
      extra_str += " %s" % params.input.target

    if hasattr(dispatcher_params, 'input') and hasattr(dispatcher_params.input, 'rungroup') and params.input.rungroup is not None:
      data_str += " input.rungroup=%d" % params.input.rungroup

    command = f"{params.input.dispatcher} {data_str} {logging_str} {extra_str}"
    if params.output.add_output_dir_option:
      command += f" output.output_dir={output_dir}"

    job_name = "r%s"%params.input.run_num

    submission_id = do_submit(command, submit_path, stdoutdir, params.mp, log_name="log.out", err_name="err.out", job_name=job_name, dry_run=params.dry_run)
    print("Job submitted.  Output in", trialdir)
    return submission_id

if __name__ == "__main__":
  script = Script()
  script.run()
