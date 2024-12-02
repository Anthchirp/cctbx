from __future__ import absolute_import, division, print_function
from six.moves import range
from six.moves import zip
# -*- Mode: Python; c-basic-offset: 2; indent-tabs-mode: nil; tab-width: 8 -*-
#
# LIBTBX_SET_DISPATCHER_NAME cctbx.xfel.xtc_process
#

try:
  import psana
except ImportError:
  pass # for running at home without psdm build

import errno
from xfel.cftbx.detector import cspad_cbf_tbx
from xfel.cxi.cspad_ana import cspad_tbx, rayonix_tbx
import pycbf, os, sys, copy, socket
import libtbx.load_env
from libtbx.utils import Sorry, Usage
from dials.util import show_mail_on_error
from dials.util.options import ArgumentParser
from libtbx.phil import parse
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex
from libtbx import easy_pickle
from dxtbx.model.experiment_list import ExperimentList

import io # fix buffering py2/3

# check version of psana
from xfel.cftbx.detector.cspad_cbf_tbx import PSANA2_VERSION

xtc_phil_str = '''
  dispatch {
    max_events = None
      .type = int
      .help = If not specified, process all events. Otherwise, only process this many
    process_percent = None
      .type = int(value_min=1, value_max=100)
      .help = Percent of events to process
    estimate_gain_only = False
      .type = bool
      .help = Use to print estimated gain parameters for each event, then exit without attempting \
              further processing.
    find_spots = True
      .type = bool
      .help = Whether to do spotfinding. Needed for indexing/integration
    datasource = None
      .type = str
      .expert_level = 2
      .help = This is to specify which datasource should be used for processing data at LCLS \
              Format is exp=<experiment_name>:run=<run_number>:<mode> \
              eg. exp=mfxo1916:run=20:xtc \
              More info at https://confluence.slac.stanford.edu/display/PSDM/Manual#Manual-Datasetspecification
    hit_finder{
      enable = True
        .type = bool
        .help = Whether to do hitfinding. hit_finder=False: process all images
      minimum_number_of_reflections = 16
        .type = int
        .help = If the number of strong reflections on an image is less than this, and \
                 the hitfinder is enabled, discard this image.
      maximum_number_of_reflections = None
       .type = int
       .help = If specified, ignores images with more than this many number of reflections
    }
    index = True
      .type = bool
      .help = Attempt to index images
    refine = False
      .type = bool
      .help = If True, after indexing, refine the experimental models
    integrate = True
      .type = bool
      .help = Integrated indexed images. Ignored if index=False
    coset = False
      .expert_level = 2
      .type = bool
      .help = Within the integrate dispatcher, integrate a sublattice coset intended to represent \
              negative control spots with no Bragg diffraction.
    dump_strong = False
      .type = bool
      .help = Save strongly diffracting images to cbf format
    dump_indexed = True
      .type = bool
      .help = Save indexed images to cbf format
    dump_all = False
      .type = bool
      .help = All frames will be saved to cbf format if set to True
    reindex_strong = False
      .type = bool
      .help = If true, after indexing and refinement, re-index the strong reflections with \
              no outlier rejection
  }
  debug
    .help = Use these flags to track down problematic events that cause unhandled exceptions. \
            Here, a bad event means it caused an unhandled exception, not that the image \
            failed to index. \
            Examples: \
            Process only unprocessed events (excluding bad events): \
              skip_processed_events=True, skip_unprocessed_events=False skip_bad_events=True \
            Process only bad events (for debugging): \
              skip_processed_events=True, skip_unprocessed_events=True skip_bad_events=False \
            Note, due to how MPI works, if an unhandled exception occurrs, some bad events \
            will be marked as bad that were simply in process when the program terminated \
            due to a bad event. Try processing only bad events single process to find the \
            culprit and alert the program authors.
  {
    skip_processed_events = False
      .type = bool
      .help = If True, will look for diagnostic files in the output directory and use \
              them to skip events that had already been processed (succesfully or not)
    skip_unprocessed_events = False
      .type = bool
      .help = If True, will look for diagnostic files in the output directory and use \
              them to skip events that had haven't been processed
    skip_bad_events = False
      .type = bool
      .help = If True, will look for diagnostic files in the output directory and use \
              them to skip events that had caused unhandled exceptions previously
    event_timestamp = None
      .type = str
      .multiple = True
      .help = List of timestamps. If set, will only process the events that match them
  }
  input {
    cfg = None
      .type = str
      .help = Path to psana config file. Genearlly not needed for CBFs. For image pickles, \
              the psana config file should have a mod_image_dict module.
    experiment = None
      .type = str
      .help = Experiment identifier, e.g. cxi84914
    run_num = None
      .type = int
      .help = Run number or run range to process
    address = None
      .type = str
      .help = Detector address, e.g. CxiDs2.0:Cspad.0, or detector alias, e.g. Ds1CsPad
    stream = None
      .type = ints
      .expert_level = 2
      .help = Stream number to read from. Usually not necessary as psana will read the data \
              from all streams by default
    override_spotfinding_trusted_max = None
      .type = int
      .help = During spot finding, override the saturation value for this data. \
              Overloads will not be integrated, but they can assist with indexing.
    override_spotfinding_trusted_min = None
      .type = int
      .help = During spot finding, override the minimum pixel value \
              for this data. This does not affect integration.
    override_integration_trusted_max = None
      .type = int
      .help = During integration, override the saturation value for this data.
    override_integration_trusted_min = None
      .type = int
      .help = During integration, override the minimum pixel value \
              for this data.
    use_ffb = False
      .type = bool
      .help = Run on the ffb if possible. Only for active users!
    xtc_dir = None
      .type = str
      .help = Optional path to data directory if it's non-standard. Only needed if xtc \
              streams are not in the standard location for your PSDM installation.
    calib_dir = None
      .type = str
      .help = Optional path to calib directory if it's non-standard. Only needed if calib \
              data are not in the standard location for your PSDM installation.
    trial = None
      .type = int
      .help = Optional. Trial number for this run.
    rungroup = None
      .type = int
      .help = Optional. Useful for organizing runs with similar parameters into logical \
              groupings.
    known_orientations_folder = None
      .type = str
      .expert_level = 2
      .help = Folder with previous processing results including crystal orientations. \
              If specified, images will not be re-indexed, but instead the known \
              orientations will be used.
    ignore_gain_mismatch = False
      .type = bool
      .expert_level = 3
      .help = Detector gain should be set on the detector models loaded from the images or in the \
              processing parameters, not both. Override the check that this is true with this flag. \
 }
  format {
    file_format = *cbf pickle
      .type = choice
      .help = Output file format, 64 tile segmented CBF or image pickle
    pickle {
      out_key = cctbx.xfel.image_dict
        .type = str
        .help = Key name that mod_image_dict uses to put image data in each psana event
    }
    cbf {
      detz_offset = None
        .type = float
        .help = Distance from back of detector rail to sample interaction region (CXI) \
                or actual detector distance (XPP/MFX)
      override_energy = None
        .type = float
        .help = If not None, use the input energy for every event instead of the energy \
                from the XTC stream
      override_distance = None
        .type = float
        .help = If not None, use the input distance for every event instead of the distance \
                from the XTC stream
      invalid_pixel_mask = None
        .type = str
        .help = Path to invalid pixel mask, in the dials.generate_mask format. If not set, use the \
                psana computed invalid pixel mask. Regardless, pixels outside of the trusted range \
                for each image will also be masked out. See cxi.make_dials_mask.
      mode = *cspad rayonix
        .type = choice
        .help = CBFs output in the designated mode
      cspad {
        mask_nonbonded_pixels = False
          .type = bool
          .help = If true, try to get non-bonded pixels from psana calibrations and apply them. Includes \
                  the 4 pixels on each side of each pixel. Only used if a custom invalid_pixel_mask is \
                  provided (otherwise the psana calibration will mask these out automatically).
        gain_mask_value = None
          .type = float
          .help = If not None, use the gain mask for the run to multiply the low-gain pixels by this number
        per_pixel_gain = False
          .type = bool
          .help = If True, use a per pixel gain from the run's calib folder, if available
        additional_gain_factor = None
          .type = float
          .help = If set, pixels counts are divided by this number after all other corrections.
        common_mode {
          algorithm = default custom
            .type = choice
            .help = Choice of SLAC's common mode correction algorithms. If not specified, use no common \
                    mode correction, only dark pedestal subtraction. Default: use the default common_mode \
                    correction. Custom, see \
                    https://confluence.slac.stanford.edu/display/PSDM/Common+mode+correction+algorithms
          custom_parameterization = None
            .type = ints
            .help = Parameters to control SLAC's common mode correction algorithms. Should be None if \
                    common_mode.algorithm is default or None.  See \
                    https://confluence.slac.stanford.edu/display/PSDM/Common+mode+correction+algorithms
        }
      }
      rayonix {
        bin_size = 2
          .type = int
          .help = Detector binning mode
        override_beam_x = None
          .type = float
          .help = If set, override the beam X position
        override_beam_y = None
          .type = float
          .help = If set, override the beam Y position
      }
    }
    per_pixel_absorption_correction
      .multiple = True {
      apply = False
        .type = bool
      algorithm = *fuller_kapton
        .type = choice
      fuller_kapton {
        xtal_height_above_kapton_mm {
          value = 0.02
            .type = float
            .help = height of the beam (or the irradiated crystal) above the kapton tape
        }
        rotation_angle_deg {
          value = 1.15
            .type = float
            .help = angle of the tape from vertical
        }
        kapton_half_width_mm {
          value = 1.5875
            .type = float
            .help = forward distance from irradiated crystal to edge of tape nearest detector
        }
        kapton_thickness_mm {
          value = 0.05
            .type = float
            .help = tape thickness
        }
      }
    }
  }
  output {
    output_dir = .
      .type = str
      .help = Directory output files will be placed
    composite_output = True
      .type = bool
      .help = If True, save one set of json/pickle files per process, where each is a \
              concatenated list of all the successful events examined by that process. \
              If False, output a separate json/pickle file per image (generates a \
              lot of files).
    delete_integration_shoeboxes = True
      .type = bool
      .help = Delete integration shoeboxes when finished with each image.
    logging_dir = None
      .type = str
      .help = Directory output log files will be placed
    experiments_filename = %s.expt
      .type = str
      .help = The filename for output experiment list
    strong_filename = %s_strong.refl
      .type = str
      .help = The filename for strong reflections from spot finder output.
    indexed_filename = %s_indexed.refl
      .type = str
      .help = The filename for indexed reflections.
    refined_experiments_filename = %s_refined.expt
      .type = str
      .help = The filename for saving refined experimental models
    integrated_filename = %s_integrated.refl
      .type = str
      .help = The filename for final experimental modls
    integrated_experiments_filename = %s_integrated.expt
      .type = str
      .help = The filename for final integrated reflections.
    coset_filename = %s_coset%d.refl
      .type = str
      .help = The filename for final coset reflections.
    coset_experiments_filename = %s_coset%d.expt
      .type = str
      .help = The filename for saving final coset experimental models.
    profile_filename = None
      .type = str
      .help = The filename for output reflection profile parameters
    integration_pickle = int-%d-%s.pickle
      .type = str
      .help = Filename for cctbx.xfel-style integration pickle files
    reindexedstrong_filename = %s_reindexedstrong.refl
      .type = str
      .help = The file name for re-indexed strong reflections
    tmp_output_dir = "(NONE)"
      .type = str
      .help = Directory for CBFlib temporary output files
  }
  mp {
    method = *mpi sge
      .type = choice
      .help = Muliprocessing method
    mpi {
      method = *client_server striping
        .type = choice
        .help = Method of serving data to child processes in MPI. client_server:    \
                use one process as a server that sends timestamps to each process.  \
                All processes will stay busy at all times at the cost of MPI send/  \
                recieve overhead. striping: each process uses its rank to determine \
                which events to process. Some processes will finish early and go    \
                idle, but no MPI overhead is incurred.
    }
    composite_stride = None
      .type = int
      .help = For MPI, if using composite mode, specify how many ranks to    \
              aggregate data from.  For example, if you have 100 processes,  \
              composite mode will output N*100 files, where N is the number  \
              of file types (json, pickle, etc). If you specify stride = 25, \
              then each group of 25 process will send their results to 4     \
              processes and only N*4 files will be created. Ideally, match   \
              stride to the number of processors per node.
  }
'''

from dials.command_line.stills_process import dials_phil_str, program_defaults_phil_str

extra_dials_phil_str = '''
  border_mask {
    include scope dials.util.masking.phil_scope
  }

  joint_reintegration {
    enable = False
      .type = bool
      .help = If enabled, after processing the data, do a joint refinement and \
              re-integration
    minimum_results = 30
      .type = int
      .help = Minimum number of integration results needed for joint reintegration
    maximum_results_per_chunk = 500
      .type = int

    include scope dials.algorithms.refinement.refiner.phil_scope
    include scope dials.algorithms.integration.integrator.phil_scope
  }
'''

def filter(evt):
    return True

def run_psana2(ims, params, comm):
    """" Begins psana2
    This setup a DataSource psana2 style. The parallelization is determined within
    the generation of the DataSource.

    ims: InMemScript (cctbx driver class)
    params: input parameters
    comm: mpi comm for broadcasting per run calibration files"""
    ds = psana.DataSource(exp=params.input.experiment, run=params.input.run_num, \
            dir=params.input.xtc_dir, max_events=params.dispatch.max_events, \
            det_name=params.input.address)

    for run in ds.runs():
      det = run.Detector(ds.det_name)
      # broadcast cctbx per run calibration
      if comm.Get_rank() == 0:
        PS_CALIB_DIR = os.environ.get('PS_CALIB_DIR')
        assert PS_CALIB_DIR
        dials_mask = easy_pickle.load(params.format.cbf.invalid_pixel_mask)
      else:
        dials_mask = None
      dials_mask = comm.bcast(dials_mask, root=0)

      for evt in run.events():
        ims.base_dxtbx = cspad_cbf_tbx.env_dxtbx_from_slac_metrology(run, params.input.address)
        ims.dials_mask = dials_mask
        ims.spotfinder_mask = None
        ims.integration_mask = None
        ims.psana_det = det
        if params.format.file_format == 'cbf':
          if params.format.cbf.cspad.common_mode.algorithm == "custom":
            ims.common_mode = params.format.cbf.cspad.common_mode.custom_parameterization
            assert ims.common_mode is not None
          else:
            ims.common_mode = params.format.cbf.cspad.common_mode.algorithm # could be None or default
        ims.process_event(run, evt)

    ims.finalize()

class EventOffsetSerializer(object):
  """ Pickles python object """
  def __init__(self,psanaOffset):
    self.filenames = psanaOffset.filenames()
    self.offsets = psanaOffset.offsets()
    self.lastBeginCalibCycleDgram = psanaOffset.lastBeginCalibCycleDgram()

from xfel.ui import db_phil_str
from xfel.command_line.xfel_process import radial_average_phil_str

phil_scope = parse(xtc_phil_str + dials_phil_str + extra_dials_phil_str + db_phil_str + radial_average_phil_str, process_includes=True).fetch(parse(program_defaults_phil_str))

from xfel.command_line.xfel_process import Script as DialsProcessScript
from xfel.ui.db.frame_logging import DialsProcessorWithLogging
from xfel.ui.db.dxtbx_db import dxtbx_xfel_db_application

class InMemScript(DialsProcessScript, DialsProcessorWithLogging):
  """ Script to process XFEL data at LCLS """
  def __init__(self):
    """ Set up the option parser. Arguments come from the command line or a phil file """
    self.usage = """
%s input.experiment=experimentname input.run_num=N input.address=address
 format.file_format=cbf format.cbf.detz_offset=N
%s input.experiment=experimentname input.run_num=N input.address=address
 format.file_format=pickle input.cfg=filename
    """%(libtbx.env.dispatcher_name, libtbx.env.dispatcher_name)
    self.parser = ArgumentParser(
      usage = self.usage,
      phil = phil_scope)

    self.debug_file_path = None
    self.debug_str = None
    self.mpi_log_file_path = None

    self.reference_detector = None

    self.composite_tag = None
    self.all_imported_experiments = None
    self.all_strong_reflections = None
    self.all_indexed_experiments = None
    self.all_indexed_reflections = None
    self.all_integrated_experiments = None
    self.all_integrated_reflections = None
    self.all_int_pickle_filenames = []
    self.all_int_pickles = []

    self.cached_ranges = None

    self.tt_low = None
    self.tt_high = None
    self.db_app = None

  def debug_start(self, ts):
    self.debug_str = "%s,%s"%(socket.gethostname(), ts)
    self.debug_str += ",%s,%s,%s\n"
    self.debug_write("start")

  def debug_write(self, string, state = None):
    ts = cspad_tbx.evt_timestamp() # Now
    debug_file_handle = open(self.debug_file_path, 'a')
    if string == "":
      debug_file_handle.write("\n")
    else:
      if state is None:
        state = "    "
      debug_file_handle.write(self.debug_str%(ts, state, string))
    debug_file_handle.close()

  def mpi_log_write(self, string):
    print(string)
    mpi_log_file_handle = open(self.mpi_log_file_path, 'a')
    mpi_log_file_handle.write(string)
    mpi_log_file_handle.close()

  def psana_mask_to_dials_mask(self, psana_mask):
    if psana_mask.dtype == bool:
      psana_mask = flex.bool(psana_mask)
    else:
      psana_mask = flex.bool(psana_mask == 1)
    assert psana_mask.focus() == (32, 185, 388)
    dials_mask = []
    for i in range(32):
      dials_mask.append(psana_mask[i:i+1,:,:194])
      dials_mask[-1].reshape(flex.grid(185,194))
      dials_mask.append(psana_mask[i:i+1,:,194:])
      dials_mask[-1].reshape(flex.grid(185,194))
    return dials_mask

  def run(self):
    """ Process all images assigned to this thread """

    try:
      params, options = self.parser.parse_args(
        show_diff_phil=True, quick_parse=True)
    except Exception as e:
      if "Unknown command line parameter definition" in str(e) or \
          "The following definitions were not recognised" in str(e):
        deprecated_params = ['mask_nonbonded_pixels','gain_mask_value','algorithm','custom_parameterization']
        deprecated_strs = ['%s','%s','common_mode.%s','common_mode.%s']
        for i in range(len(deprecated_params)):
          if deprecated_params[i] in str(e):
            print("format.cbf.%s"%(deprecated_strs[i]%deprecated_params[i]), "has changed to format.cbf.cspad.%s"%(deprecated_strs[i]%deprecated_params[i]))
      raise

    if params.dispatch.coset:
      self.all_coset_experiments = ExperimentList()
      self.all_coset_reflections = flex.reflection_table()

    # Check inputs
    if params.input.experiment is None or \
       params.input.run_num is None or \
       (params.input.address is None and params.format.file_format != 'pickle'):
      raise Usage(self.usage)

    if params.format.file_format == "cbf":
      if params.format.cbf.detz_offset is None:
        raise Usage(self.usage)
    elif params.format.file_format == "pickle":
      if params.input.cfg is None:
        raise Usage(self.usage)
    else:
      raise Usage(self.usage)

    if not os.path.exists(params.output.output_dir):
      raise Sorry("Output path not found:" + params.output.output_dir)

    if params.format.file_format == "cbf":
      if params.output.tmp_output_dir == "(NONE)":
        tmp_dir = params.output.tmp_output_dir
      else:
        #Environment variable redirect for CBFLib temporary CBF_TMP_XYZ file output
        if params.output.tmp_output_dir is None:
          tmp_dir = os.path.join(params.output.output_dir, '.tmp')
        else:
          tmp_dir = os.path.join(params.output.tmp_output_dir, '.tmp')
        if not os.path.exists(tmp_dir):
          with show_mail_on_error():
            try:
              os.makedirs(tmp_dir)
              # Can fail if running multiprocessed - that's OK if the folder was created
            except OSError as e:  # In Python 2, a FileExistsError is just an OSError
              if e.errno != errno.EEXIST:  # If this OSError is not a FileExistsError
                raise
      os.environ['CBF_TMP_DIR'] = tmp_dir

    for abs_params in params.integration.absorption_correction:
      if abs_params.apply and abs_params.algorithm == "fuller_kapton":
        if not (params.integration.debug.output and not params.integration.debug.separate_files):
          raise Sorry('Shoeboxes must be saved to integration intermediates to apply an absorption correction. '\
            +'Set integration.debug.output=True and integration.debug.separate_files=False to save shoeboxes.')

    self.params = params
    self.load_reference_geometry()

    if params.output.composite_output:
      self.all_imported_experiments = ExperimentList()
      self.all_strong_reflections = flex.reflection_table()
      self.all_indexed_experiments = ExperimentList()
      self.all_indexed_reflections = flex.reflection_table()
      self.all_integrated_experiments = ExperimentList()
      self.all_integrated_reflections = flex.reflection_table()
    else:
      # The convention is to put %s in the phil parameter to add a time stamp to
      # each output datafile. Save the initial templates here.
      self.strong_filename_template                 = params.output.strong_filename
      self.indexed_filename_template                = params.output.indexed_filename
      self.refined_experiments_filename_template    = params.output.refined_experiments_filename
      self.integrated_filename_template             = params.output.integrated_filename
      self.integrated_experiments_filename_template = params.output.integrated_experiments_filename
      self.coset_filename_template                  = params.output.coset_filename
      self.coset_experiments_filename_template      = params.output.coset_experiments_filename
      self.reindexedstrong_filename_template        = params.output.reindexedstrong_filename

    # Don't allow the strong reflections to be written unless there are enough to
    # process
    params.output.strong_filename = None

    # Save the paramters
    self.params_cache = copy.deepcopy(params)
    self.options = options

    if params.mp.method == "mpi":
      from libtbx.mpi4py import MPI
      comm = MPI.COMM_WORLD
      rank = comm.Get_rank() # each process in MPI has a unique id, 0-indexed
      size = comm.Get_size() # size: number of processes running in this job
    elif params.mp.method == "sge" and \
        'SGE_TASK_ID'    in os.environ and \
        'SGE_TASK_FIRST' in os.environ and \
        'SGE_TASK_LAST'  in os.environ:
      if 'SGE_STEP_SIZE' in os.environ:
        assert int(os.environ['SGE_STEP_SIZE']) == 1
      if os.environ['SGE_TASK_ID'] == 'undefined' or os.environ['SGE_TASK_ID'] == 'undefined' or os.environ['SGE_TASK_ID'] == 'undefined':
        rank = 0
        size = 1
      else:
        rank = int(os.environ['SGE_TASK_ID']) - int(os.environ['SGE_TASK_FIRST'])
        size = int(os.environ['SGE_TASK_LAST']) - int(os.environ['SGE_TASK_FIRST']) + 1
    else:
      rank = 0
      size = 1
    self.composite_tag = "%04d"%rank

    # Configure the logging
    if params.output.logging_dir is None:
      logfile = ''
    elif params.output.logging_dir == 'DevNull':
      print("Redirecting stdout, stderr and other DIALS logfiles to /dev/null")
      logfile = os.devnull
      try:
        # Python 3
        sys.stdout = io.TextIOWrapper(open(os.devnull,'wb', 0), write_through=True)
        sys.stderr = io.TextIOWrapper(open(os.devnull,'wb', 0), write_through=True)
      except TypeError:
        # Python 2
        sys.stdout = open(os.devnull,'w', buffering=0)
        sys.stderr = open(os.devnull,'w',buffering=0)

      info_path = os.devnull
      debug_path = os.devnull
    else:
      log_path = os.path.join(params.output.logging_dir, "log_rank%04d.out"%rank)
      error_path = os.path.join(params.output.logging_dir, "error_rank%04d.out"%rank)
      print("Redirecting stdout to %s"%log_path)
      print("Redirecting stderr to %s"%error_path)
      try:
        # Python 3
        sys.stdout = io.TextIOWrapper(open(log_path,'ab', 0), write_through=True)
        sys.stderr = io.TextIOWrapper(open(error_path,'ab', 0), write_through=True)
      except TypeError:
        # Python 2
        sys.stdout = open(log_path,'a', buffering=0)
        sys.stderr = open(error_path,'a',buffering=0)

      print("Should be redirected now")

      logfile = os.path.join(params.output.logging_dir, "info_rank%04d.out"%rank)

    from dials.util import log
    log.config(options.verbose, logfile=logfile)

    debug_dir = os.path.join(params.output.output_dir, "debug")
    if not os.path.exists(debug_dir):
      try:
        os.makedirs(debug_dir)
      except OSError as e:
        pass # due to multiprocessing, makedirs can sometimes fail
    assert os.path.exists(debug_dir)

    if params.debug.skip_processed_events or params.debug.skip_unprocessed_events or params.debug.skip_bad_events:
      print("Reading debug files...")
      self.known_events = {}
      for filename in os.listdir(debug_dir):
        # format: hostname,timestamp_event,timestamp_now,status,detail
        for line in open(os.path.join(debug_dir, filename)):
          vals = line.strip().split(',')
          if len(vals) != 5:
            continue
          _, ts, _, status, detail = vals
          if status in ["done", "stop", "fail"]:
            self.known_events[ts] = status
          else:
            self.known_events[ts] = "unknown"

    self.debug_file_path = os.path.join(debug_dir, "debug_%d.txt"%rank)
    write_newline = os.path.exists(self.debug_file_path)
    if write_newline: # needed if the there was a crash
      self.debug_write("")

    if params.mp.method != 'mpi' or params.mp.mpi.method == 'client_server':
      if rank == 0:
        self.mpi_log_file_path = os.path.join(debug_dir, "mpilog.out")
        write_newline = os.path.exists(self.mpi_log_file_path)
        if write_newline: # needed if the there was a crash
          self.mpi_log_write("\n")

    # FIXME MONA: psana 2 has pedestals and geometry hardcoded for cxid9114.
    # We can remove after return code when all interfaces are ready.
    if PSANA2_VERSION:
        print(("PSANA2_VERSION", PSANA2_VERSION))
        run_psana2(self, params, comm)
        return

    # set up psana
    if params.input.cfg is not None:
      psana.setConfigFile(params.input.cfg)
    # all cores in stripe mode and the master in client-server mode read smd
    if params.dispatch.datasource is None:
      datasource = "exp=%s:run=%s:%s"%(params.input.experiment,params.input.run_num,'smd')
      if params.input.xtc_dir is not None:
        if params.input.use_ffb:
          raise Sorry("Cannot specify the xtc_dir and use SLAC's ffb system")
        datasource += ":dir=%s"%params.input.xtc_dir
      elif params.input.use_ffb:
      # as ffb is only at SLAC, ok to hardcode /reg/d here
        datasource += ":dir=/reg/d/ffb/%s/%s/xtc"%(params.input.experiment[0:3],params.input.experiment)
      if params.input.stream is not None and len(params.input.stream) > 0:
        datasource += ":stream=%s"%(",".join(["%d"%stream for stream in params.input.stream]))
      if params.input.calib_dir is not None:
        psana.setOption('psana.calib-dir',params.input.calib_dir)
      if params.mp.method == "mpi" and params.mp.mpi.method == 'client_server' and size > 2:
        dataset_name_client = datasource.replace(":smd",":rax")
      # for client-server, master reads smd - clients read rax
        if rank == 0:
          ds = psana.DataSource(datasource)
        else:
          ds = psana.DataSource(dataset_name_client)

      else:
      # for stripe, all cores read smd
        ds = psana.DataSource(datasource)
    else:
      datasource = params.dispatch.datasource
      ds = psana.DataSource(datasource)

    if params.format.file_format == "cbf":
      self.psana_det = psana.Detector(params.input.address, ds.env())

    # set this to sys.maxint to analyze all events
    if params.dispatch.max_events is None:
      max_events = sys.maxsize
    else:
      max_events = params.dispatch.max_events

    # Set up db connection if being used
    if self.params.experiment_tag is not None:
      self.db_app = dxtbx_xfel_db_application(params)

    for run in ds.runs():
      if params.format.file_format == "cbf":
        if params.format.cbf.mode == "cspad":
          # load a header only cspad cbf from the slac metrology
          try:
            self.base_dxtbx = cspad_cbf_tbx.env_dxtbx_from_slac_metrology(run, params.input.address)
          except Exception as e:
            raise Sorry("Couldn't load calibration file for run %d, %s"%(run.run(), str(e)))
        elif params.format.cbf.mode == "rayonix":
          # load a header only rayonix cbf from the input parameters
          detector_size = rayonix_tbx.get_rayonix_detector_dimensions(ds.env())
          self.base_dxtbx = rayonix_tbx.get_dxtbx_from_params(params.format.cbf.rayonix, detector_size)

        if self.base_dxtbx is None:
          raise Sorry("Couldn't load calibration file for run %d"%run.run())

        if params.format.file_format == 'cbf':
          if params.format.cbf.cspad.common_mode.algorithm == "custom":
            self.common_mode = params.format.cbf.cspad.common_mode.custom_parameterization
            assert self.common_mode is not None
          else:
            self.common_mode = params.format.cbf.cspad.common_mode.algorithm # could be None or default

        if params.format.cbf.invalid_pixel_mask is not None:
          self.dials_mask = easy_pickle.load(params.format.cbf.invalid_pixel_mask)
          if params.format.cbf.mode == "cspad":
            assert len(self.dials_mask) == 64
            if self.params.format.cbf.cspad.mask_nonbonded_pixels:
              psana_mask = self.psana_det.mask(run.run(),calib=False,status=False,edges=False,central=False,unbond=True,unbondnbrs=True)
              dials_mask = self.psana_mask_to_dials_mask(psana_mask)
              self.dials_mask = [self.dials_mask[i] & dials_mask[i] for i in range(len(dials_mask))]
        else:
          if params.format.cbf.mode == "cspad":
            psana_mask = self.psana_det.mask(run.run(),calib=True,status=True,edges=True,central=True,unbond=True,unbondnbrs=True)
            self.dials_mask = self.psana_mask_to_dials_mask(psana_mask)
          else:
            self.dials_mask = None

      if self.params.spotfinder.lookup.mask is not None:
        self.spotfinder_mask = easy_pickle.load(self.params.spotfinder.lookup.mask)
      else:
        self.spotfinder_mask = None
      if self.params.integration.lookup.mask is not None:
        self.integration_mask = easy_pickle.load(self.params.integration.lookup.mask)
      else:
        self.integration_mask = None

      # prepare fractions of process_percent, if given
      process_fractions = None
      if params.dispatch.process_percent:
        import fractions
        percent = params.dispatch.process_percent / 100
        process_fractions = fractions.Fraction(percent).limit_denominator(100)
      # list of all events
      # only cycle through times in client_server mode
      if params.mp.method == "mpi" and params.mp.mpi.method == 'client_server' and size > 2:
        # process fractions only works in idx-striping mode
        if params.dispatch.process_percent:
          raise Sorry("Process percent only works in striping mode.")
        print("Using MPI client server in rax mode")
        # use a client/server approach to be sure every process is busy as much as possible
        # only do this if there are more than 2 processes, as one process will be a server
        try:
          if rank == 0:
            # server process
            self.mpi_log_write("MPI START\n")
            for nevt, evt in enumerate(run.events()):
              if nevt == max_events: break
              self.mpi_log_write("Getting next available process\n")
              offset = evt.get(psana.EventOffset)
              rankreq = comm.recv(source=MPI.ANY_SOURCE)
              t = evt.get(psana.EventId).time()
              ts = cspad_tbx.evt_timestamp((t[0],t[1]/1e6))
              self.mpi_log_write("Process %s is ready, sending ts %s\n"%(rankreq, ts))
              comm.send(EventOffsetSerializer(offset),dest=rankreq)
            # send a stop command to each process
            self.mpi_log_write("MPI DONE, sending stops\n")
            for rankreq in range(size-1):
              self.mpi_log_write("Getting next available process\n")
              rankreq = comm.recv(source=MPI.ANY_SOURCE)
              self.mpi_log_write("Sending stop to %d\n"%rankreq)
              comm.send('endrun',dest=rankreq)
            self.mpi_log_write("All stops sent.")
          else:
            # client process
            while True:
              # inform the server this process is ready for an event
              print("Rank %d getting next task"%rank)
              comm.send(rank,dest=0)
              print("Rank %d waiting for response"%rank)
              offset = comm.recv(source=0)
              if offset == 'endrun':
                print("Rank %d recieved endrun"%rank)
                break
              evt = ds.jump(offset.filenames, offset.offsets, offset.lastBeginCalibCycleDgram)
              print("Rank %d beginning processing"%rank)
              try:
                self.process_event(run, evt)
              except Exception as e:
                print("Rank %d unhandled exception processing event"%rank, str(e))
              print("Rank %d event processed"%rank)
        except Exception as e:
          print("Error caught in main loop")
          print(str(e))
        print("Synchronizing rank %d"%rank)
        comm.Barrier()
        print("Rank %d done with main loop"%rank)
      else:
        import resource
        # chop the list into pieces, depending on rank.  This assigns each process
        # events such that the get every Nth event where N is the number of processes
        print("Striping events")

        nevent = mem = first = last = 0
        if process_fractions:
          def process_this_event(nevent):
            # nevent modulo the denominator gives us which fraction we're in
            n_mod_denom = nevent % process_fractions.denominator
            # compare the 0-indexed modulo against the 1-indexed numerator (intentionally not <=)
            n_accept = n_mod_denom < process_fractions.numerator
            return n_accept
        for nevent, evt in enumerate(run.events()):
          if nevent%size != rank: continue
          if nevent >= max_events: break
          if process_fractions and not process_this_event(nevent): continue

          self.process_event(run, evt)

          mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
          if nevent < 50:
            #print "Mem test rank %03d"%rank, i, mem
            continue
          #print "Mem test rank %03d"%rank, 'Cycle %6d total %7dkB increase %4dkB' % (i, mem, mem - last)
          if not first:
            first = mem
          last = mem
        print('Total memory leaked in %d cycles: %dkB' % (nevent+1-50, mem - first))

    print("Rank %d finalizing"%rank)
    try:
      self.finalize()
    except Exception as e:
      print("Rank %d, exception caught in finalize"%rank)
      print(str(e))

    if params.format.file_format == "cbf" and params.output.tmp_output_dir == "(NONE)":
      try:
        os.rmdir(tmp_dir)
      except Exception as e:
        pass

    if params.joint_reintegration.enable:
      if params.output.composite_output:
        raise NotImplementedError("Joint reintegration not implemented for composite output yet")
      assert self.params.dispatch.dump_indexed, "Cannot do joint reintegration unless indexed files were dumped"
      if rank == 0:
        reint_dir = os.path.join(params.output.output_dir, "reint")
        if not os.path.exists(reint_dir):
          os.makedirs(reint_dir)
        images = []
        experiment_jsons = []
        indexed_tables = []
        for filename in os.listdir(params.output.output_dir):
          if not filename.endswith("_indexed.refl"):
            continue
          experiment_jsons.append(os.path.join(params.output.output_dir, filename.split("_indexed.refl")[0] + "_refined.expt"))
          indexed_tables.append(os.path.join(params.output.output_dir, filename))
          if params.format.file_format == "cbf":
            images.append(os.path.join(params.output.output_dir, filename.split("_indexed.refl")[0] + ".cbf"))
          elif params.format.file_format == "pickle":
            images.append(os.path.join(params.output.output_dir, filename.split("_indexed.refl")[0] + ".pickle"))

        if len(images) < params.joint_reintegration.minimum_results:
          pass # print and return

        # TODO: maximum_results_per_chunk = 500
        combo_input = os.path.join(reint_dir, "input.phil")
        f = open(combo_input, 'w')
        for json, indexed in zip(experiment_jsons, indexed_tables):
          f.write("input {\n")
          f.write("  experiments = %s\n"%json)
          f.write("  reflections = %s\n"%indexed)
          f.write("}\n")
        f.close()

        combined_experiments_file = os.path.join(reint_dir, "combined.expt")
        combined_reflections_file = os.path.join(reint_dir, "combined.refl")
        command = "dials.combine_experiments reference_from_experiment.average_detector=True %s output.reflections=%s output.experiments=%s"% \
          (combo_input, combined_reflections_file, combined_experiments_file)
        print(command)
        from libtbx import easy_run
        easy_run.fully_buffered(command).raise_if_errors().show_stdout()

        from dxtbx.model.experiment_list import ExperimentListFactory

        combined_experiments = ExperimentListFactory.from_json_file(combined_experiments_file, check_format=False)
        combined_reflections = flex.reflection_table.from_file(combined_reflections_file)

        from dials.algorithms.refinement import RefinerFactory

        refiner = RefinerFactory.from_parameters_data_experiments(
          params.joint_reintegration, combined_reflections, combined_experiments)

        refiner.run()
        experiments = refiner.get_experiments()
        reflections = combined_reflections.select(refiner.selection_used_for_refinement())

        experiments.as_file(os.path.join(reint_dir, "refined.expt"))
        reflections.as_pickle(os.path.join(reint_dir, "refined.refl"))

        for expt_id, (expt, img_file) in enumerate(zip(experiments, images)):
          try:
            refls = reflections.select(reflections['id'] == expt_id)
            refls['id'] = flex.int(len(refls), 0)
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            self.params.output.integrated_filename = os.path.join(reint_dir, base_name + "_integrated.refl")

            expts = ExperimentList([expt])
            self.integrate(expts, refls)
            expts.as_file(os.path.join(reint_dir, base_name + "_refined.expt"))
          except Exception as e:
            print("Couldn't reintegrate", img_file, str(e))
    print("Rank %d signing off"%rank)

  def get_run_and_timestamp(self, obj):
    # Used by database logger
    return self.run.run(), self.timestamp

  def process_event(self, run, evt):
    """
    Process a single event from a run
    @param run psana run object
    @param timestamp psana timestamp object
    """
    if PSANA2_VERSION:
      sec  = evt._seconds
      nsec = evt._nanoseconds
    else:
      time = evt.get(psana.EventId).time()
      fid = evt.get(psana.EventId).fiducials()
      sec  = time[0]
      nsec = time[1]

    ts = cspad_tbx.evt_timestamp((sec,nsec/1e6))
    if ts is None:
      print("No timestamp, skipping shot")
      return

    if len(self.params_cache.debug.event_timestamp) > 0 and ts not in self.params_cache.debug.event_timestamp:
      return
    self.run = run

    if self.params_cache.debug.skip_processed_events or self.params_cache.debug.skip_unprocessed_events or self.params_cache.debug.skip_bad_events:
      if ts in self.known_events:
        if self.known_events[ts] not in ["stop", "done", "fail"]:
          if self.params_cache.debug.skip_bad_events:
            print("Skipping event %s: possibly caused an unknown exception previously"%ts)
            return
        elif self.params_cache.debug.skip_processed_events:
          print("Skipping event %s: processed successfully previously"%ts)
          return
      else:
        if self.params_cache.debug.skip_unprocessed_events:
          print("Skipping event %s: not processed previously"%ts)
          return

    self.debug_start(ts)

    # FIXME MONA: below will be replaced with filter() callback
    if not PSANA2_VERSION:
      if evt.get("skip_event") or "skip_event" in [key.key() for key in evt.keys()]:
        print("Skipping event",ts)
        self.debug_write("psana_skip", "skip")
        return

    print("Accepted", ts)
    self.params = copy.deepcopy(self.params_cache)

    # the data needs to have already been processed and put into the event by psana
    if self.params.format.file_format == 'cbf':
      if self.params.format.cbf.mode == "cspad":
        # get numpy array, 32x185x388
        data = cspad_cbf_tbx.get_psana_corrected_data(self.psana_det, evt, use_default=False, dark=True,
                                                    common_mode=self.common_mode,
                                                    apply_gain_mask=self.params.format.cbf.cspad.gain_mask_value is not None,
                                                    gain_mask_value=self.params.format.cbf.cspad.gain_mask_value,
                                                    per_pixel_gain=self.params.format.cbf.cspad.per_pixel_gain,
                                                    additional_gain_factor=self.params.format.cbf.cspad.additional_gain_factor)
      elif self.params.format.cbf.mode == "rayonix":
        data = rayonix_tbx.get_data_from_psana_event(evt, self.params.input.address)
      if data is None:
        print("No data")
        self.debug_write("no_data", "skip")
        return

      if self.params.format.cbf.override_distance is None:
        if self.params.format.cbf.mode == "cspad":
          distance = cspad_tbx.env_distance(self.params.input.address, run.env(), self.params.format.cbf.detz_offset)
        elif self.params.format.cbf.mode == "rayonix":
          distance = self.params.format.cbf.detz_offset
        if distance is None:
          print("No distance, skipping shot")
          self.debug_write("no_distance", "skip")
          return
      else:
        distance = self.params.format.cbf.override_distance

      if self.params.format.cbf.override_energy is None:
        if PSANA2_VERSION:
          wavelength = 12398.4187/self.psana_det.raw.photonEnergy(evt)
        else:
          wavelength = cspad_tbx.evt_wavelength(evt)
        if wavelength is None:
          print("No wavelength, skipping shot")
          self.debug_write("no_wavelength", "skip")
          return
      else:
        wavelength = 12398.4187/self.params.format.cbf.override_energy

    if self.params.format.file_format == 'pickle':
      image_dict = evt.get(self.params.format.pickle.out_key)
      data = image_dict['DATA']

    self.timestamp = timestamp = t = ts
    s = t[0:4] + t[5:7] + t[8:10] + t[11:13] + t[14:16] + t[17:19] + t[20:23]
    print("Processing shot", s)

    def build_dxtbx_image():
      if self.params.format.file_format == 'cbf':
        # stitch together the header, data and metadata into the final dxtbx format object
        if self.params.format.cbf.mode == "cspad":
          dxtbx_img = cspad_cbf_tbx.format_object_from_data(self.base_dxtbx, data, distance, wavelength, timestamp, self.params.input.address, round_to_int=False)
        elif self.params.format.cbf.mode == "rayonix":
          dxtbx_img = rayonix_tbx.format_object_from_data(self.base_dxtbx, data, distance, wavelength, timestamp, self.params.input.address)

        if self.params.input.reference_geometry is not None:
          from dxtbx.model import Detector
          # copy.deep_copy(self.reference_detctor) seems unsafe based on tests. Use from_dict(to_dict()) instead.
          dxtbx_img._detector_instance = Detector.from_dict(self.reference_detector.to_dict())
          if self.params.format.cbf.mode == "cspad":
            dxtbx_img.sync_detector_to_cbf() #FIXME need a rayonix version of this??

      elif self.params.format.file_format == 'pickle':
        from dxtbx.format.FormatPYunspecifiedStill import FormatPYunspecifiedStillInMemory
        dxtbx_img = FormatPYunspecifiedStillInMemory(image_dict)
      return dxtbx_img

    dxtbx_img = build_dxtbx_image()
    for correction in self.params.format.per_pixel_absorption_correction:
      if correction.apply:
        if correction.algorithm == "fuller_kapton":
          from dials.algorithms.integration.kapton_correction import all_pixel_image_data_kapton_correction
          data = all_pixel_image_data_kapton_correction(image_data=dxtbx_img, params=correction.fuller_kapton)()
          dxtbx_img = build_dxtbx_image() # repeat as necessary to update the image pixel data and rebuild the image

    self.tag = s # used when writing integration pickle

    if self.params.dispatch.dump_all:
      self.save_image(dxtbx_img, self.params, os.path.join(self.params.output.output_dir, "shot-" + s))

    self.cache_ranges(dxtbx_img, self.params.input.override_spotfinding_trusted_min, self.params.input.override_spotfinding_trusted_max)

    from dxtbx.imageset import ImageSet, ImageSetData, MemReader
    imgset = ImageSet(ImageSetData(MemReader([dxtbx_img]), None))
    imgset.set_beam(dxtbx_img.get_beam())
    imgset.set_detector(dxtbx_img.get_detector())

    if self.params.dispatch.estimate_gain_only:
      from dials.command_line.estimate_gain import estimate_gain
      estimate_gain(imgset)
      return

    # FIXME MONA: radial avg. is currently disabled
    if not PSANA2_VERSION:
      # Two values from a radial average can be stored by mod_radial_average. If present, retrieve them here
      key_low = 'cctbx.xfel.radial_average.two_theta_low'
      key_high = 'cctbx.xfel.radial_average.two_theta_high'
      tt_low = evt.get(key_low)
      tt_high = evt.get(key_high)

    if self.params.radial_average.enable:
      if tt_low is not None or tt_high is not None:
        print("Warning, mod_radial_average is being used while also using xtc_process radial averaging. mod_radial_averaging results will not be logged to the database.")

    from dxtbx.model.experiment_list import ExperimentListFactory
    experiments = ExperimentListFactory.from_imageset_and_crystal(imgset, None)

    try:
      self.pre_process(experiments)
    except Exception as e:
      self.debug_write("preprocess_exception", "fail")
      return

    if not self.params.dispatch.find_spots:
      self.debug_write("data_loaded", "done")
      return

    # before calling DIALS for processing, set output paths according to the templates
    if not self.params.output.composite_output:
      if self.indexed_filename_template is not None and "%s" in self.indexed_filename_template:
        self.params.output.indexed_filename = os.path.join(self.params.output.output_dir, self.indexed_filename_template%("idx-" + s))
      if "%s" in self.refined_experiments_filename_template:
        self.params.output.refined_experiments_filename = os.path.join(self.params.output.output_dir, self.refined_experiments_filename_template%("idx-" + s))
      if "%s" in self.integrated_filename_template:
        self.params.output.integrated_filename = os.path.join(self.params.output.output_dir, self.integrated_filename_template%("idx-" + s))
      if "%s" in self.integrated_experiments_filename_template:
        self.params.output.integrated_experiments_filename = os.path.join(self.params.output.output_dir, self.integrated_experiments_filename_template%("idx-" + s))
      if "%s" in self.coset_filename_template:
        self.params.output.coset_filename = os.path.join(self.params.output.output_dir, self.coset_filename_template%("idx-" + s,
          self.params.integration.coset.transformation))
      if "%s" in self.coset_experiments_filename_template:
        self.params.output.coset_experiments_filename = os.path.join(self.params.output.output_dir, self.coset_experiments_filename_template%("idx-" + s,
          self.params.integration.coset.transformation))
      if "%s" in self.reindexedstrong_filename_template:
        self.params.output.reindexedstrong_filename = os.path.join(self.params.output.output_dir, self.reindexedstrong_filename_template%("idx-" + s))

    if self.params.input.known_orientations_folder is not None:
      expected_orientation_path = os.path.join(self.params.input.known_orientations_folder, os.path.basename(self.params.output.refined_experiments_filename))
      if os.path.exists(expected_orientation_path):
        print("Known orientation found")
        from dxtbx.model.experiment_list import ExperimentListFactory
        self.known_crystal_models = ExperimentListFactory.from_json_file(expected_orientation_path, check_format=False).crystals()
      else:
        print("Image not previously indexed, skipping.")
        self.debug_write("not_previously_indexed", "stop")
        return

    # Load a dials mask from the trusted range and psana mask
    import dials.util.masking
    mask = dials.util.masking.generate_mask(imgset, self.params.border_mask)
    if self.params.format.file_format == "cbf" and self.dials_mask is not None:
      mask = tuple([a&b for a, b in zip(mask,self.dials_mask)])
    if self.spotfinder_mask is None:
      self.params.spotfinder.lookup.mask = mask
    else:
      self.params.spotfinder.lookup.mask = tuple([a&b for a, b in zip(mask,self.spotfinder_mask)])

    self.debug_write("spotfind_start")
    try:
      observed = self.find_spots(experiments)
    except Exception as e:
      import traceback; traceback.print_exc()
      print(str(e), "event", timestamp)
      self.debug_write("spotfinding_exception", "fail")
      return

    print("Found %d bright spots"%len(observed))

    if self.params.dispatch.hit_finder.enable and len(observed) < self.params.dispatch.hit_finder.minimum_number_of_reflections:
      print("Not enough spots to index")
      self.debug_write("not_enough_spots_%d"%len(observed), "stop")
      return
    if self.params.dispatch.hit_finder.maximum_number_of_reflections is not None:
      if self.params.dispatch.hit_finder.enable and len(observed) > self.params.dispatch.hit_finder.maximum_number_of_reflections:
        print("Too many spots to index - Possibly junk")
        self.debug_write("too_many_spots_%d"%len(observed), "stop")
        return

    self.restore_ranges(dxtbx_img)

    # save cbf file
    if self.params.dispatch.dump_strong:
      self.save_image(dxtbx_img, self.params, os.path.join(self.params.output.output_dir, "hit-" + s))

      # save strong reflections.  self.find_spots() would have done this, but we only
      # want to save data if it is enough to try and index it
      if self.strong_filename_template:
        if "%s" in self.strong_filename_template:
          strong_filename = self.strong_filename_template%("hit-" + s)
        else:
          strong_filename = self.strong_filename_template
        strong_filename = os.path.join(self.params.output.output_dir, strong_filename)

        from dials.util.command_line import Command
        Command.start('Saving {0} reflections to {1}'.format(
            len(observed), os.path.basename(strong_filename)))
        observed.as_pickle(strong_filename)
        Command.end('Saved {0} observed to {1}'.format(
            len(observed), os.path.basename(strong_filename)))

    if not self.params.dispatch.index:
      self.debug_write("strong_shot_%d"%len(observed), "done")
      return

    # index and refine
    self.debug_write("index_start")
    try:
      experiments, indexed = self.index(experiments, observed)
    except Exception as e:
      import traceback; traceback.print_exc()
      print(str(e), "event", timestamp)
      self.debug_write("indexing_failed_%d"%len(observed), "stop")
      return

    if self.params.dispatch.dump_indexed:
      img_path = self.save_image(dxtbx_img, self.params, os.path.join(self.params.output.output_dir, "idx-" + s))
      imgset = ExperimentListFactory.from_filenames([img_path]).imagesets()[0]
      assert len(experiments.detectors()) == 1;   imgset.set_detector(experiments[0].detector)
      assert len(experiments.beams()) == 1;       imgset.set_beam(experiments[0].beam)
      assert len(experiments.scans()) <= 1;       imgset.set_scan(experiments[0].scan)
      assert len(experiments.goniometers()) <= 1; imgset.set_goniometer(experiments[0].goniometer)
      for expt_id, expt in enumerate(experiments):
        expt.imageset = imgset

    self.debug_write("refine_start")

    try:
      experiments, indexed = self.refine(experiments, indexed)
    except Exception as e:
      import traceback; traceback.print_exc()
      print(str(e), "event", timestamp)
      self.debug_write("refine_failed_%d"%len(indexed), "fail")
      return

    if self.params.dispatch.reindex_strong:
      self.debug_write("reindex_start")
      try:
        self.reindex_strong(experiments, observed)
      except Exception as e:
        import traceback; traceback.print_exc()
        print(str(e), "event", timestamp)
        self.debug_write("reindexstrong_failed_%d"%len(indexed), "fail")
        return

    if not self.params.dispatch.integrate:
      self.debug_write("index_ok_%d"%len(indexed), "done")
      return

    # integrate
    self.debug_write("integrate_start")
    self.cache_ranges(dxtbx_img, self.params.input.override_integration_trusted_min, self.params.input.override_integration_trusted_max)

    if self.cached_ranges is not None:
      # Load a dials mask from the trusted range and psana mask
      imgset = ImageSet(ImageSetData(MemReader([dxtbx_img]), None))
      imgset.set_beam(dxtbx_img.get_beam())
      imgset.set_detector(dxtbx_img.get_detector())
      import dials.util.masking
      mask = dials.util.masking.generate_mask(imgset, self.params.border_mask)
      if self.params.format.file_format == "cbf" and self.dials_mask is not None:
        mask = tuple([a&b for a, b in zip(mask,self.dials_mask)])
    if self.integration_mask is None:
      self.params.integration.lookup.mask = mask
    else:
      self.params.integration.lookup.mask = tuple([a&b for a, b in zip(mask,self.integration_mask)])

    try:
      integrated = self.integrate(experiments, indexed)
    except Exception as e:
      import traceback; traceback.print_exc()
      print(str(e), "event", timestamp)
      self.debug_write("integrate_failed_%d"%len(indexed), "fail")
      return
    self.restore_ranges(dxtbx_img)

    self.debug_write("integrate_ok_%d"%len(integrated), "done")

  def save_image(self, image, params, root_path):
    """ Save an image, in either cbf or pickle format.
    @param image dxtbx format object
    @param params phil scope object
    @param root_path output file path without extension
    """

    if params.format.file_format == 'cbf':
      dest_path = root_path + ".cbf"
    elif params.format.file_format == 'pickle':
      dest_path = root_path + ".pickle"

    try:
      if params.format.file_format == 'cbf':
        image._cbf_handle.write_widefile(dest_path.encode(), pycbf.CBF,\
          pycbf.MIME_HEADERS|pycbf.MSG_DIGEST|pycbf.PAD_4K, 0)
      elif params.format.file_format == 'pickle':
        easy_pickle.dump(dest_path, image._image_file)
    except Exception:
      print("Warning, couldn't save image:", dest_path)

    return dest_path

  def cache_ranges(self, dxtbx_img, min_val, max_val):
    """ Save the current trusted ranges, and replace them with the given overrides, if present.
    @param cspad_image dxtbx format object
    """
    if min_val is None and max_val is None:
      return

    detector = dxtbx_img.get_detector()
    self.cached_ranges = []
    for panel in detector:
      new_range = cached_range = panel.get_trusted_range()
      self.cached_ranges.append(cached_range)
      if max_val is not None:
        new_range = new_range[0], max_val
      if min_val is not None:
        new_range = min_val, new_range[1]

      panel.set_trusted_range(new_range)

  def restore_ranges(self, dxtbx_img):
    """ Restore the previously cached trusted ranges, if present.
    @param cspad_image dxtbx format object
    """
    if self.cached_ranges is None:
      return

    detector = dxtbx_img.get_detector()
    for cached_range, panel in zip(self.cached_ranges, detector):
      panel.set_trusted_range(cached_range)

    self.cached_ranges = None

  def reindex_strong(self, experiments, strong):
    print("Reindexing strong reflections using refined experimental models and no outlier rejection...")
    from dials.algorithms.indexing.stills_indexer import StillsIndexerKnownOrientation
    indexer = StillsIndexerKnownOrientation(strong, experiments.imagesets(), self.params, experiments.crystals())
    indexed_reflections = indexer.reflections.select(indexer.indexed_reflections)

    print("Indexed %d strong reflections out of %d"%(len(indexed_reflections), len(strong)))
    self.save_reflections(indexed_reflections, self.params.output.reindexedstrong_filename)

  def finalize(self):
    if self.params.output.composite_output:
      # Each process will write its own set of output files
      s = self.composite_tag
      self.params.output.indexed_filename                = os.path.join(self.params.output.output_dir, self.params.output.indexed_filename%("idx-" + s))
      self.params.output.refined_experiments_filename    = os.path.join(self.params.output.output_dir, self.params.output.refined_experiments_filename%("idx-" + s))
      self.params.output.integrated_filename             = os.path.join(self.params.output.output_dir, self.params.output.integrated_filename%("idx-" + s))
      self.params.output.integrated_experiments_filename = os.path.join(self.params.output.output_dir, self.params.output.integrated_experiments_filename%("idx-" + s))
      self.params.output.coset_filename                  = os.path.join(self.params.output.output_dir, self.params.output.coset_filename%("idx-" + s,
        self.params.integration.coset.transformation))
      self.params.output.coset_experiments_filename      = os.path.join(self.params.output.output_dir, self.params.output.coset_experiments_filename%("idx-" + s,
        self.params.integration.coset.transformation))
      self.params.output.reindexedstrong_filename        = os.path.join(self.params.output.output_dir, self.params.output.reindexedstrong_filename%("idx-" + s))

    super(InMemScript, self).finalize()

if __name__ == "__main__":
  with show_mail_on_error():
    script = InMemScript()
    script.run()
