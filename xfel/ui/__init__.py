from __future__ import absolute_import, division, print_function
import os
from iotbx.phil import parse
from libtbx.utils import Sorry

master_phil_str = """
dispatcher = cctbx.xfel.process
  .type = str
  .help = Which program to run. cctbx.xfel.process should be used in most cases.
dry_run = False
  .type = bool
  .help = If True, the program will create the trial directory but not submit the job, \
          and will show the command that would have been executed.
monitoring_mode = False
  .type = bool
  .help = If True, will hide the submit jobs buttons and remove tabs for configuring \
          Trials and Datasets.
facility {
  name = lcls *standalone
    .type = choice
    .help = Facility for the XFEL gui. LCLS configures the GUI to use LCLS services \
            for data monitoring, job submission, and so forth. Standlone runs the \
            GUI for all other data sources.
  lcls {
    experiment = ""
      .type = str
      .help = Experiment name, eg cxid9114
    web {
      location = "SLAC"
        .type = str
        .help = Where to look for XTC streams. Can be SLAC, SLACFFB (active experiment \
                only, retired), SRCF_FFB (active experiment only), SDF or NERSC \
                (contact LCLS staff to arrange file mover for SDF or NERSC).
      user = ""
        .type = str
        .help = Username for LCLS run database web service
      password = ""
        .type = str
        .help = Web password. Will be cached in plain text!
      enforce80 = False
        .type = bool
        .help = report only on stream 81, FEE spectrometer
      enforce81 = False
        .type = bool
        .help = report only on stream 81, FEE spectrometer
    }
    use_ffb = False
      .type = bool
      .help = Run on the ffb if possible. Only for active users!
    dump_shots = False
      .type = bool
      .help = Write images to disk whether they index or not. \
              Helpful for tuning spotfinding and indexing parameters, and necessary \
              for the "Should have indexed" feature of the Run Stats tab.
    api {
      protocol = http *https
        .type = choice
        .help = Use http or https for API connections
      host = pswww.slac.stanford.edu
        .type = str
        .help = Address of the LCLS API Host
    }
  }
  standalone {
    data_dir = None
      .type = path
      .help = Folder to monitor for new data
    monitor_for = *files folders
      .type = choice
      .help = Whether to monitor for new files or new folders with files in them
    folders {
      method = *status_file n_files
        .type = choice
        .help = How to determine if a run is complete. status_file: look for a \
                status.txt file. n_files: run is complete when at least \
                n_files_needed files are found
      n_files_needed = 0
        .type = int
        .help = If criteria is n_files, this is how many files are needed \
                before a run is complete
    }
    files {
      last_modified = 0
        .type = float
        .help = This is the miniumum number of seconds since a file was last \
                modified before the run is complete
      minimum_file_size = 0
        .type = int
        .help = Minimum file size (in bytes) before the run is complete
    }
    template = None
      .type = str
      .help = File matching pattern for new data. Example: *.h5
    composite_files = True
      .type = bool
      .help = If True, files are submitted as individual runs. Otherwise, groups \
              of files are submitted as single runs
  }
}
output_folder = ""
  .type = path
  .help = Processing results will go in this folder

include scope xfel.command_line.cxi_mpi_submit.mp_phil_scope
"""
db_phil_str = """
experiment_tag = None
  .type = str
  .help = User defined tag to describe the set of trials being performed. All database tables will \
          be pre-pended with this string
db {
  host = psdb-user.slac.stanford.edu
    .type = str
    .help = Host name for mysql databse server
  name = ""
    .type = str
    .help = Database name
  user = ""
    .type=str
    .help = Database user name
  password = ""
    .type = str
    .help = Database password. Will be cached as plain text!
  verbose = False
    .type = bool
    .expert_level = 2
    .help = Print to the terminal all database queries
  port = 3306
    .type = int
    .help = Port number to be used for connection
  logging_batch_size = 3
    .type = int
    .help = Number of images to log at once. Increase if using many (thousands) of processors.
  server {
    basedir = None
      .type = path
      .help = Root folder for mysql database

    prompt_for_root_password = False
      .type = bool
      .help = Whether to always ask for the root password. Note, root password is always \
              needed when the database is initialized.

    root_user = 'root'
      .type = str
      .help = username for root
      .expert_level = 2

    root_password = None
      .type = str
      .help = Password for root user
      .expert_level = 2
  }
}
"""
master_phil_scope = parse(master_phil_str + db_phil_str, process_includes=True)

settings_dir = os.path.join(os.path.expanduser('~'), '.cctbx.xfel')
settings_file = os.path.join(settings_dir, 'settings.phil')

known_dials_dispatchers = {
  'cctbx.xfel.xtc_process': 'xfel.command_line.xtc_process',
  'cctbx.xfel.process': 'xfel.command_line.xfel_process',
  'dials.stills_process': 'dials.command_line.stills_process',
  'cctbx.xfel.small_cell_process': 'xfel.small_cell.command_line.xfel_small_cell_process',
  'cctbx.xfel.stripe_experiment': 'xfel.command_line.striping',
  'cctbx.xfel.merge': 'xfel.merging.application.phil.phil'
}

def load_phil_scope_from_dispatcher(dispatcher):
  import importlib
  module = importlib.import_module(known_dials_dispatchers[dispatcher])
  importlib.reload(module)
  phil_scope = module.phil_scope
  return phil_scope

def load_cached_settings():
  if os.path.exists(settings_file):
    user_phil = parse(file_name = settings_file)
    params, unused = master_phil_scope.fetch(source = user_phil, track_unused_definitions=True)
    params = params.extract()
    if len(unused) > 0:
      from .phil_patch import sync_phil
      sync_phil(params, unused)
    return params
  else:
    return master_phil_scope.extract()

def save_cached_settings(params):
  if not os.path.exists(settings_dir):
    os.makedirs(settings_dir)

  working_phil = master_phil_scope.format(python_object = params)
  diff_phil = master_phil_scope.fetch_diff(source = working_phil)

  try:
    f = open(settings_file.encode('utf8'), 'wb')
    f.write(diff_phil.as_str().encode('utf8'))
    f.close()
  except IOError:
    raise Sorry('Unable to write %s.' % settings_file)
