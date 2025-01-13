from __future__ import absolute_import, division, print_function
#
# Handle multiprocessing with any of the implemented methods so that this step
# is abstracted away from the use case (e.g. cxi_mpi_submit).
#
from libtbx.utils import Sorry
import os
import math

mp_phil_str = '''
  mp {
    method = local *lsf sge pbs slurm shifter sfapi htcondor custom
      .type = choice
      .help = Computing environment
    use_mpi = True
      .type = bool
      .help = Use mpi multiprocessing
    mpi_command = mpirun
      .type = str
      .help = Command to invoke MPI processing. Include extra arguments to \
              this command here.
    mpi_option = "mp.method=mpi"
      .type = str
      .expert_level = 2
      .help = Parameter to turn on MPI in the dispatcher program
    nproc = 1
      .type = int
      .help = Number of processes total (== nnodes x nproc_per_node). \
              If two of the three params (nproc, nnodes, nproc_per_node) are \
              specified, the last will be determined by modular arithmetic. \
              If all three are specified, nnodes is ignored. nproc alone is \
              sufficient for most methods.
    nnodes = 1
      .type = int
      .help = Number of nodes to request
    nnodes_index = None
      .type = int
      .help = If defined, use this many nodes for indexing and integration. \
              Currently only works for mp.method=shifter or slurm.
    nnodes_tder = None
      .type = int
      .help = If defined, use this many nodes for ensemble refinement. \
              Currently only works for mp.method=shifter or slurm.
    nnodes_scale = None
      .type = int
      .help = If defined, use this many nodes for scaling. \
              Currently only works for mp.method=shifter or slurm.
    nnodes_merge = None
      .type = int
      .help = If defined, use this many nodes for merging. \
              Currently only works for mp.method=shifter or slurm.
    nproc_per_node = 1
      .type = int
      .help = Number of processes to allocate per node
    queue = None
      .type = str
      .help = Queue to submit multiprocessing job to (optional for some methods)
    memory = None
      .type = int
      .help = Memory (in MB) to allocate for a job (optional)
    wall_time = None
      .type = int
      .help = Wall time limit (in minutes) to impose (optional)
    max_queued = None
      .type = int
      .help = Maximum number of jobs running or queued
    extra_options = None
      .type = str
      .multiple = True
      .help = Any other options to be included in the job submission command
    extra_args = None
      .type = str
      .multiple = True
      .help = Any other arguments to the main command
    env_script = None
      .type = str
      .multiple = True
      .help = Path to script sourcing a particular environment (optional)
    phenix_script = None
      .type = str
      .multiple = True
      .help = Path to script sourcing a phenix environment (optional)
    local {
      include_mp_in_command = True
        .type = bool
        .help = Whether to decorate command with appropiate multiprocessing \
                arguments. If False, it's assumed the multiprocessing  \
                arguments are provided by the calling program.
    }
    shifter {
      submit_command = "sbatch "
        .type = str
        .help = Command used to run the zero-level script sbatch.sh.
      shifter_image = None
        .type = str
        .help = Name of Shifter image to use for processing, as you would use \
                in an sbatch script. Example: docker:dwpaley/cctbx-xfel:fix18
      sbatch_script_template = None
        .type = path
        .help = Script template to be run with sbatch. The script will be copied \
                to the trial directory as sbatch.sh and modified. Must contain \
                exactly one srun command of the format srun [...] <srun_script> \
                (including the <> brackets). May contain additional srun commands. \
                May also contain substitutables <walltime>, <partition>, <nnodes> \
                and <nproc>.
      srun_script_template = None
        .type = path
        .help = Script template to be run with srun. The script will be copied \
                to the trial directory as srun.sh and modified. Must contain \
                exactly one instance of the string <command> (including the <> \
                brackets) after setting up the necessary environment.
      partition = regular
        .type = str
        .help = Partition to run jobs on, e.g. regular or debug.
      jobname = LCLS_EXP
        .type = str
        .help = Job Name
      project = None
        .type = str
        .help = Name of NERSC project -- formerly "repo"
      reservation = None
        .type = str
        .help = Name of NERSC reservation
      constraint = haswell
        .type = str
        .help = Haswell or KNL
      staging = DataWarp *None
        .type = choice
        .help = Optionally stage logs to the DataWarp burst buffer. Only works \
                when writing to Cori cscratch.
    }
    htcondor {
      executable_path = None
        .type = path
        .help = Path to executable script (should be openmpiscript or mp2script). \
                See examples folder that comes with htcondor.
      filesystemdomain = sdfarm.kr
        .type = str
        .help = Domain of shared filesystem (see htcondor docs)
    }
    custom {
      submit_command_template = None
        .type = str
        .help = Job submission command template. There should be one instance of the \
                string <script> (including the <> brackets) in the command template. \
                Fields <queue>, <nproc>, <memory>, <walltime>, <outfile>, <errfile>, \
                <envscripts>, and <args> will similarly be replaced if present.
      submit_script_template = None
        .type = path
        .help = Submission script template. The script will be copied to the trial \
                directory and modified. There should be one instance of the string \
                <command> (including the <> brackets) in the template script, which \
                will be replaced with the processing command. <queue>, <nproc>, \
                <memory>, <walltime>, <outfile>, <errfile>, <envscripts>, and <args> \
                will similarly be replaced if present.
      wall_time_string = None
        .type = str
        .help = Fully formatted wall time limit (e.g. 00:30:00). For custom computing \
                environments, mp.wall_time is ignored because the format is unknown.
    }
    encapsulate_submit_script = True
      .type = bool
      .help = Encapsulate the submission command itself in another script containing \
              the job submission command (e.g. qsub, bsub, condor_submit, sbatch \
              etc.
  }
'''

class get_submit_command(object):
  def __init__(self, command, submit_path, stdoutdir, params,
               log_name="log.out", err_name="log.err", job_name=None, root_dir=None):
    """ Get a submit command for the various compute environments
    @param command Any command line program and its arguments
    @param submit_path Submit script will be written here
    @param stdoutdir Log file will be created in this directory
    @param params Multiprocessing phil params (see mp_phil_scope)
    @param log_name Filename for stdout (optional).
    @param err_name Filename for stderr (if None, combined with the stdout).
    @param job_name For applicable queueing systems, identifier for the job (optional).
    """
    self.shell_path = "/bin/bash"
    self.source_env_scripts = []
    self.options_inside_submit_script = []
    self.submit_head = "qsub"
    self.submit_path = submit_path
    self.stdoutdir = stdoutdir
    self.log_name = log_name
    self.err_name = err_name
    self.params = params
    self.job_name = job_name
    self.root_dir = root_dir
    self.command = command
    self.options = []
    self.args = []

    # Wrap the `os` module if running in sfapi mode
    if params.method == "sfapi":
      import logging
      from xfel.util.sfapi_connector import OsWrapper, OsSFAPI, LOGGER
      LOGGER.setLevel(logging.DEBUG)
      self.os = OsWrapper(backend=OsSFAPI())
    else:
      self.os = os

  def customize_for_method(self):
    pass

  def eval_params(self):
    pass

  def substitute(self, template, marker, value):
    if marker in template:
      if value is None:
        raise Sorry("No value found for %s" % marker)
      return template.replace(marker, value)
    else:
      return template

  def delete(self, template, marker):
    template_lines = template.split('\n')
    return '\n'.join([l for l in template_lines if marker not in l])

  def make_executable(self, file):
    import stat
    st = self.os.stat(file)
    self.os.chmod(file, st.st_mode | stat.S_IXUSR)

  def write_script(self):
    command_str = " ".join([self.command] + self.args)
    with self.os.open(self.submit_path, 'w') as f:
      f.write("#! %s\n" % self.shell_path)
      for line in self.options_inside_submit_script:
        f.write("%s\n" % line)
      for line in self.source_env_scripts:
        f.write("%s\n" % line)
      f.write("\n")
      f.write("%s\n" % command_str)
    self.make_executable(self.submit_path)

  def generate_submit_command(self):
    return " ".join([self.submit_head] + self.options + [self.submit_path])

  def encapsulate_submit(self):
    path, ext = self.os.path.splitext(self.submit_path)
    encapsulate_path = path + "_submit" + ext
    with self.os.open(encapsulate_path, 'w') as f:
      f.write("#! /bin/%s\n\n" % ext[1:])
      f.write(self.generate_submit_command())
      f.write("\n")

  def __call__(self):
    self.customize_for_method()
    self.eval_params()
    self.write_script()
    if self.params.encapsulate_submit_script:
      self.encapsulate_submit()
    return self.generate_submit_command()

class get_local_submit_command(get_submit_command):

  def customize_for_method(self):
    if self.params.local.include_mp_in_command:
      if self.params.use_mpi:
        self.command = "%s -n %d %s" % (self.params.mpi_command, self.params.nproc, self.command)
        self.command += " %s"%self.params.mpi_option
      elif self.params.nproc > 1:
        self.command += " mp.nproc=%d" % self.params.nproc

  def eval_params(self):
    # <args> (optional, following the command)
    for arg in self.params.extra_args:
      self.args.append(arg)

  def generate_submit_command(self):
    return self.submit_path

class get_lsf_submit_command(get_submit_command):

  def customize_for_method(self):
    self.submit_head = "bsub"
    if self.params.use_mpi:
      self.command = "%s %s" % (self.params.mpi_command, self.command)
      self.command += " %s"%self.params.mpi_option
  def eval_params(self):
    # -n <nproc>
    nproc_str = "-n %d" % self.params.nproc
    self.options.append(nproc_str)

    # -o <outfile>
    out_str = "-o %s" % os.path.join(self.stdoutdir, self.log_name)
    self.options.append(out_str)

    # -e <errfile> (optional)
    if self.err_name is not None:
      err_str = "-e %s" % os.path.join(self.stdoutdir, self.err_name)
      self.options.append(err_str)

    # -q <queue> (optional on LSF)
    if self.params.queue is not None:
      queue_str = "-q %s" % self.params.queue
      self.options.append(queue_str)

    # -W <wall_time_limit> (optional)
    if self.params.wall_time is not None:
      hours = self.params.wall_time // 60
      minutes = self.params.wall_time % 60
      wt_str = "-W %2d:%02d" % (hours, minutes)
      self.options.append(wt_str)

    # -R "rusage[mem=<memory_requested>]" (optional)
    if self.params.memory is not None:
      memory_str = "-R \"rusage[mem=%d]\"" % self.params.memory
      self.options.append(memory_str)

    # <extra_options> (optional, preceding the command)
    for cmd in self.params.extra_options:
      self.options.append(cmd)

    # source </path/to/env.sh> (optional)
    for env in self.params.env_script:
      env_str = "source %s\n" % env
      self.source_env_scripts.append(env_str)

    # <args> (optional, following the command)
    for arg in self.params.extra_args:
      self.args.append(arg)

class get_sge_submit_command(get_submit_command):

  def customize_for_method(self):
    self.shell_path += " -q"
    self.options.append("-cwd")
#    self.options.append("mp.method=sge")
    if self.params.use_mpi:
      self.command = "%s -n ${NSLOTS} %s"%(self.params.mpi_command, self.command) #This command currently (14/10/2020) has problems at Diamond as it will randomly use incorrect number of cores
      self.command += " %s"%self.params.mpi_option
    else:
      self.command = "%s mp.nproc=${NSLOTS}"%(self.command)

  def eval_params(self):
    # -t 1-<nproc>
    if self.params.nproc > 1:
      nproc_str = "-pe smp %d" % self.params.nproc #Change the submission command to smp, as the openmpi currently confilicts with mpi of Dials and cctbx.xfel.merge
      self.options.append(nproc_str)

    # -o <outfile>
    out_str = "-o %s" % os.path.join(self.stdoutdir, self.log_name)
    self.options.append(out_str)

    # -j [y/n] -e <errfile> (optional)
    if self.err_name is not None:
      err_str = "-j n -e %s" % os.path.join(self.stdoutdir, self.err_name)
      self.options.append(err_str)
    else:
      self.options.append("-j y")

    # -q <queue>
    if self.params.queue is None:
      raise Sorry("Queue not specified.")
    queue_str = "-q %s" % self.params.queue
    self.options.append(queue_str)

    # -l h_rt=<wall_time_limit> (optional)
    if self.params.wall_time is not None:
      hours = self.params.wall_time // 60
      minutes = self.params.wall_time % 60
      wt_str = "-l h_rt=%02d:%02d:00" % (hours, minutes)
      self.options.append(wt_str)

    # -l mem_free=<memory_requested> (optional)
    if self.params.memory is not None:
      memory_str = "-l mem_free=%dM" % self.params.memory
      self.options.append(memory_str)

    # -N <job_name>
    if self.job_name is not None:
      name_str = "-N %s" % self.job_name
      self.options.append(name_str)

    # <extra_options> (optional, preceding the command)
    for cmd in self.params.extra_options:
      self.options.append(cmd)

    # source </path/to/env.sh> (optional)
    for env in self.params.env_script:
      env_str = "source %s\n" % env
      self.source_env_scripts.append(env_str)

    # <args> (optional, following the command)
    for arg in self.params.extra_args:
      self.args.append(arg)

class get_pbs_submit_command(get_submit_command):

  def customize_for_method(self):
    if (self.params.nnodes > 1) or (self.params.nproc_per_node > 1):
      self.params.nproc = self.params.nnodes * self.params.nproc_per_node
    if self.params.use_mpi:
      self.command = "mpiexec --hostfile $PBS_NODEFILE %s" % (self.command)
      self.command += " %s"%self.params.mpi_option
  def eval_params(self):

    # # -t 1-<nproc> # deprecated
    # if self.params.nproc > 1:
    #   nproc_str = "#PBS -l mppwidth=%d" % self.params.nproc
    #   self.options_inside_submit_script.append(nproc_str)

    # -l nodes=<nnodes>:ppn=<procs_per_node>
    if max(self.params.nproc, self.params.nproc_per_node, self.params.nnodes) > 1:
      # If specified, nproc overrides procs_per_node and procs_per_node overrides
      # nnodes. One process per node is requested if only nproc is specified.
      if self.params.nproc > 1:
        import math
        if self.params.nproc <= self.params.nproc_per_node:
          procs_per_node = self.params.nproc
          nnodes = 1
        elif self.params.nproc_per_node > 1:
          procs_per_node = self.params.nproc_per_node
          nnodes = int(math.ceil(self.params.nproc/procs_per_node))
        elif self.params.nnodes > 1:
          procs_per_node = int(math.ceil(self.params.nproc/self.params.nnodes))
          nnodes = self.params.nnodes
        else: # insufficient information; allocate 1 proc per node
          procs_per_node = 1
          nnodes = self.params.nproc
      else:
        procs_per_node = self.params.nproc_per_node
        nnodes = self.params.nnodes
      nproc_str = "#PBS -l nodes=%d:ppn=%d" % (nnodes, procs_per_node)
      self.options_inside_submit_script.append(nproc_str)

    # -o <outfile>
    out_str = "#PBS -o %s" % os.path.join(self.stdoutdir, self.log_name)
    self.options_inside_submit_script.append(out_str)

    # [-j oe/-e <errfile>] (optional)
    if self.err_name is not None:
      err_str = "#PBS -e %s" % os.path.join(self.stdoutdir, self.err_name)
      self.options_inside_submit_script.append(err_str)
    else:
      self.options_inside_submit_script.append("#PBS -j oe")

    # -q <queue>
    if self.params.queue is None:
      raise Sorry("Queue not specified.")
    queue_str = "#PBS -q %s" % self.params.queue
    self.options_inside_submit_script.append(queue_str)

    # -l walltime=<wall_time_limit> (optional)
    if self.params.wall_time is not None:
      hours = self.params.wall_time // 60
      minutes = self.params.wall_time % 60
      wt_str = "#PBS -l walltime=%2d:%02d:00" % (hours, minutes)
      self.options_inside_submit_script.append(wt_str)

    # -l mem_free=<memory_requested> (optional)
    if self.params.memory is not None:
      memory_str = "#PBS -l mem=%dmb" % self.params.memory
      self.options_inside_submit_script.append(memory_str)

    # -N <job_name>
    if self.job_name is not None:
      name_str = "#PBS -N %s" % self.job_name
      self.options_inside_submit_script.append(name_str)

    # <extra_options> (optional, preceding the command)
    for cmd in self.params.extra_options:
      cmd_str = "#PBS %s" % cmd
      self.options_inside_submit_script.append(cmd_str)

    if self.root_dir is not None:
      cmd_str = "cd %s"%self.root_dir
      self.options_inside_submit_script.append(cmd_str)

    # source </path/to/env.sh> (optional)
    for env in self.params.env_script:
      env_str = "source %s\n" % env
      self.source_env_scripts.append(env_str)

    if '<output_dir>' in self.command:
      self.command = self.command.replace(
        '<output_dir>',
        os.path.split(self.stdoutdir[0])
      )
    # <args> (optional, following the command)
    image_average_output_dir = os.path.join(os.path.split(self.stdoutdir)[0], 'out')
    for arg in self.params.extra_args:
      if '<output_dir>' in arg:
        arg = arg.replace('<output_dir>', image_average_output_dir)
      self.args.append(arg)

class get_slurm_submit_command(get_submit_command):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def customize_for_method(self):
    self.submit_head = "sbatch"
    if self.params.use_mpi:
      self.command = "%s %s" % (self.params.mpi_command, self.command)
      self.command += " %s"%self.params.mpi_option

  def eval_params(self):
    nproc_str = "#SBATCH --nodes %d" % self.params.nnodes
    if self.params.nproc_per_node:
      nproc_str += "\n#SBATCH --ntasks-per-node=%d" % self.params.nproc_per_node
    self.options_inside_submit_script.append(nproc_str)

    # -o <outfile>
    out_str = "#SBATCH --output=%s" % os.path.join(self.stdoutdir, self.log_name)
    self.options_inside_submit_script.append(out_str)

    # [-j oe/-e <errfile>] (optional)
    if self.err_name is not None:
      err_str = "#SBATCH --error=%s" % os.path.join(self.stdoutdir, self.err_name)
      self.options_inside_submit_script.append(err_str)

    # -q <queue>
    if self.params.queue and self.params.queue.strip():
      queue_str = "#SBATCH --partition %s" % self.params.queue.strip()
      self.options_inside_submit_script.append(queue_str)

    # -l walltime=<wall_time_limit> (optional)
    if self.params.wall_time is not None:
      hours = self.params.wall_time // 60
      minutes = self.params.wall_time % 60
      wt_str = "#SBATCH --time=%02d:%02d:00" % (hours, minutes)
      self.options_inside_submit_script.append(wt_str)

    # -l mem_free=<memory_requested> (optional)
    if self.params.memory is not None:
      memory_str = "#SBATCH --mem=%dmb" % self.params.memory
      self.options_inside_submit_script.append(memory_str)

    # -N <job_name>
    if self.job_name is not None:
      name_str = "#SBATCH --job-name=%s" % self.job_name
      self.options_inside_submit_script.append(name_str)

    # <extra_options> (optional, preceding the command)
    for cmd in self.params.extra_options:
      cmd_str = "#SBATCH %s" % cmd
      self.options_inside_submit_script.append(cmd_str)

    # source </path/to/env.sh> (optional)
    for env in self.params.env_script:
      env_str = "source %s\n" % env
      self.source_env_scripts.append(env_str)

    if 'phenix' in self.command:
      self.source_env_scripts.append("cd %s\n"%os.path.dirname(self.submit_path))

    if '<output_dir>' in self.command:
      self.command = self.command.replace(
        '<output_dir>',
        os.path.split(self.stdoutdir[0])
      )
    # <args> (optional, following the command)
    image_average_output_dir = os.path.join(os.path.split(self.stdoutdir)[0], 'out')
    for arg in self.params.extra_args:
      if '<output_dir>' in arg:
        arg = arg.replace('<output_dir>', image_average_output_dir)
      self.args.append(arg)

class get_sfapi_submit_command(get_slurm_submit_command):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  # No need for constructor -- the interited constructor is just fine for sfapi
  def generate_submit_command(self):
    # For SFAPI, only return the path of the generated jobscript
    return self.submit_path

class get_shifter_submit_command(get_submit_command):

  # No need for constructor -- the interited constructor is just fine for shifter

  def customize_for_method(self):
    # template for sbatch.sh
    self.sbatch_template = self.params.shifter.sbatch_script_template
    self.destination = os.path.dirname(self.submit_path)
    self.prefix = os.path.splitext(os.path.basename(self.submit_path))[0]
    if self.prefix: self.prefix += "_"
    if not self.sbatch_template:
      from xfel.ui.db.cfgs import shifter_templates
      self.sbatch_contents = shifter_templates.sbatch_template
    else:
      with open(self.sbatch_template, "r") as sb:
        self.sbatch_contents = sb.read()
    self.sbatch_path = os.path.join(self.destination, self.prefix + "sbatch.sh")

    # template for srun.sh
    self.srun_template = self.params.shifter.srun_script_template
    if not self.srun_template:
      from xfel.ui.db.cfgs import shifter_templates
      self.srun_contents = shifter_templates.srun_template
    else:
      with open(self.srun_template, "r") as sr:
        self.srun_contents = sr.read()
    if self.params.use_mpi:
      self.command = "%s" % (self.command)
      self.command += " %s"%self.params.mpi_option
    self.srun_path = os.path.join(self.destination, self.prefix + "srun.sh")


  def eval_params(self):

    # --image <shifter_image>
    if self.params.shifter.shifter_image:
      self.sbatch_contents = self.substitute(
          self.sbatch_contents,
          "<shifter_image>",
          self.params.shifter.shifter_image
      )
    else:
      raise Sorry("Must supply a shifter image")

    # -N <nnodes>
    self.sbatch_contents = self.substitute(self.sbatch_contents, "<nnodes>",
      str(self.params.nnodes))

    # --tasks-per-node <nproc_per_node> (optional)
    self.sbatch_contents = self.substitute(self.sbatch_contents, "<nproc_per_node>",
      str(self.params.nproc_per_node))

    # For now use nproc = nnodes*nproc_per_node
    # TODO: find a way for the user to specify _either_ nproc_per_node, or nproc
    nproc = self.params.nnodes * self.params.nproc_per_node

    # -n <nproc> (optional)
    self.sbatch_contents = self.substitute(self.sbatch_contents, "<nproc>",
      str(nproc))

    # -W <walltime> (optional)
    if self.params.wall_time is not None:
      hours = self.params.wall_time // 60
      minutes = self.params.wall_time % 60
      wt_str = "%02d:%02d:00" % (hours, minutes)
      self.sbatch_contents = self.substitute(self.sbatch_contents, "<walltime>",
        wt_str)

    # --qos <queue>
    self.sbatch_contents = self.substitute(self.sbatch_contents, "<queue>",
      self.params.queue)

    # --partition <partition>
    self.sbatch_contents = self.substitute(self.sbatch_contents, "<partition>",
      self.params.shifter.partition)

    # --job-name
    self.sbatch_contents = self.substitute(self.sbatch_contents, "<jobname>",
      self.params.shifter.jobname)

    # -A
    self.sbatch_contents = self.substitute(self.sbatch_contents, "<project>",
      self.params.shifter.project)

    # --reservation
    if self.params.shifter.reservation:
      self.sbatch_contents = self.substitute(
          self.sbatch_contents, "<reservation>", self.params.shifter.reservation
      )
    else:
      self.sbatch_contents = self.delete(self.sbatch_contents, "<reservation>")

    # --constraint
    self.sbatch_contents = self.substitute(self.sbatch_contents, "<constraint>",
      self.params.shifter.constraint)

    self.sbatch_contents = self.substitute(self.sbatch_contents, "<out_log>",
      os.path.join(self.destination , self.prefix + "out.log"))

    self.sbatch_contents = self.substitute(self.sbatch_contents, "<err_log>",
      os.path.join(self.destination , self.prefix + "err.log"))

    self.sbatch_contents = self.substitute(self.sbatch_contents, "<output_dir>",
      self.destination)

    # Delete datawarp instructions if we're not staging logs
    if self.params.shifter.staging != "DataWarp":
      self.sbatch_contents = self.delete(self.sbatch_contents, "#DW")

    # <srun_script>
    self.sbatch_contents = self.substitute(self.sbatch_contents, "<srun_script>",
      self.srun_path)

    # <command> and any extra args
    if len(self.params.extra_args) > 0:
      self.srun_contents = self.substitute(self.srun_contents, "<command>",
        "<command> %s" % " ".join(self.params.extra_args))
    self.srun_contents = self.substitute(self.srun_contents, "<command>",
      self.command)


  def generate_submit_command(self):
    return self.params.shifter.submit_command + " " + self.sbatch_path

  def encapsulate_submit(self):
    pass

  def generate_sbatch_script(self):
    with open(self.sbatch_path, "w") as sb:
      sb.write(self.sbatch_contents)
      sb.write("\n")
    self.make_executable(self.sbatch_path)

  def generate_srun_script(self):
    with open(self.srun_path, "w") as sr:
      sr.write(self.srun_contents)
      sr.write("\n")
    self.make_executable(self.srun_path)

  def write_script(self):
    self.generate_sbatch_script()
    self.generate_srun_script()

class get_htcondor_submit_command(get_submit_command):
  def __init__(self, *args, **kwargs):
    super(get_htcondor_submit_command, self).__init__(*args, **kwargs)
    self.destination = os.path.dirname(self.submit_path)
    self.basename = os.path.splitext(os.path.basename(self.submit_path))[0]

  def customize_for_method(self):
    self.submit_head = "condor_submit"
    if self.params.use_mpi:
      self.command = "%s" % (self.command)
      self.command += " %s"%self.params.mpi_option

  def generate_submit_command(self):
    return "condor_submit " + os.path.join(self.destination, self.basename + "_condorparams")

  def eval_params(self):
    if self.params.use_mpi:
      from libtbx import easy_run
      d = dict(executable_path = self.params.htcondor.executable_path,
               arguments       = self.submit_path,
               nproc           = self.params.nproc,
               working_folder  = self.destination,
               log_path        = os.path.join(self.stdoutdir, self.basename + '_condor.log'),
               output_path     = os.path.join(self.stdoutdir, self.log_name),
               error_path      = os.path.join(self.stdoutdir, self.err_name),
               requirements    = 'target.filesystemdomain == "%s"'% self.params.htcondor.filesystemdomain)

      # Find if there is a continguous set of slots available on one node
      r = easy_run.fully_buffered('condor_status | grep Unclaimed | grep %s'%self.params.htcondor.filesystemdomain)
      machines = {}
      for line in r.stdout_lines:
        try:
          machine = line.split()[0].split('@')[1]
        except IndexError: continue
        if machine not in machines:
          machines[machine] = 0
        machines[machine] += 1

      for machine in machines:
        if machines[machine] >= self.params.nproc:
          d['requirements'] += ' && machine == "%s"'%machine
          break

      condor_params = """
universe = parallel
executable = {executable_path}
arguments = {arguments}
machine_count = {nproc}
initialdir = {working_folder}
when_to_transfer_output = on_exit
log                     = {log_path}
output                  = {output_path}
error                   = {error_path}
requirements = {requirements}
+ParallelShutdownPolicy = "WAIT_FOR_ALL"
RunAsOwner = True
queue
"""
    else:
      assert self.params.htcondor.executable_path is None
      d = dict(executable_path = self.submit_path,
               working_folder  = self.destination,
               log_path        = os.path.join(self.stdoutdir, self.basename + '_condor.log'),
               output_path     = os.path.join(self.stdoutdir, self.log_name),
               error_path      = os.path.join(self.stdoutdir, self.err_name),
               filesystemdomain= self.params.htcondor.filesystemdomain)
      condor_params = """
universe = vanilla
executable = {executable_path}
initialdir = {working_folder}
when_to_transfer_output = on_exit
log                     = {log_path}
output                  = {output_path}
error                   = {error_path}
requirements = target.filesystemdomain == "{filesystemdomain}"
RunAsOwner = True
queue
"""

    with open(os.path.join(self.destination, self.basename + "_condorparams"), 'w') as f:
      f.write(condor_params.format(**d))

    # source </path/to/env.sh> (optional)
    for env in self.params.env_script:
      env_str = "source %s\n" % env
      self.source_env_scripts.append(env_str)

class get_custom_submit_command(get_submit_command):

  def customize_for_method(self):
    # template for the script to be submitted, beginning with #!
    self.script_template = self.params.custom.submit_script_template
    if not os.path.exists(self.template):
      raise Sorry("Custom submission template file not found: %s" % self.template)

    # template for the submission command itself, e.g. qsub -n <nproc> -q <queue> script.sh
    self.command_template = self.params.custom.submit_command_template
    if self.command_template is None:
      raise Sorry("Custom submit command must be specified for custom environments.")

  def eval_params(self):
    # any changes to the script to be submitted
    with open(self.script_template, "r") as script:
      self.script_contents = script.read()

    # <command> and any <args>
    if len(self.params.extra_args) > 0:
      self.script_contents = self.script_contents.replace("<command>",
        "<command> %s" % " ".join(self.params.extra_args))
    self.script_contents = self.script_contents.replace("<command>", self.command)

    # other changes to the contents of the script
    for marker, value in [
      ("<queue>", self.params.queue),
      ("<nproc>", self.params.nproc),
      ("<memory>", self.params.memory),
      ("<walltime>", self.params.custom.wall_time_string),
      ("<outfile>", os.path.join(self.stdoutdir, self.log_name)),
      ("<errfile>", os.path.join(self.stdoutdir, self.err_name)),
      ("<envscripts>", self.params.env_script)]:
      self.script_contents = self.substitute(self.script_contents, marker, value)

    # any changes to the submission command
    # <script> and any extra <options>
    if len(self.params.extra_options) > 0:
      self.submit_command_contents = self.params.custom.submit_command_template.replace("<script>",
        "%s <script>" % " ".join(self.params.extra_options))
    self.submit_command_contents = self.submit_command_contents.replace("<script>",
      self.submit_path)

    # other changes to the submission command
    for marker, value in [
      ("<queue>", self.params.queue),
      ("<nproc>", self.params.nproc),
      ("<memory>", self.params.memory),
      ("<walltime>", self.params.custom.wall_time_string),
      ("<outfile>", os.path.join(self.stdoutdir, self.log_name)),
      ("<errfile>", os.path.join(self.stdoutdir, self.err_name))]:
      self.submit_command_contents = self.substitute(self.submit_command_contents, marker, value)

  def write_script(self):
    with open(self.submit_path, "w") as f:
      f.write(self.script_contents)
      f.write("\n")

  def generate_submit_command(self):
    return self.submit_command_contents

def get_submit_command_chooser(command, submit_path, stdoutdir, params,
                               log_name="log.out", err_name="log.err", job_name=None,
                               root_dir=None):
  if params.method == "local":
    choice = get_local_submit_command
  elif params.method == "lsf":
    choice = get_lsf_submit_command
  elif params.method == "sge":
    choice = get_sge_submit_command
  elif params.method == "pbs":
    choice = get_pbs_submit_command
  elif params.method == "slurm":
    choice = get_slurm_submit_command
  elif params.method == "shifter":
    choice = get_shifter_submit_command
  elif params.method == "sfapi":
    choice = get_sfapi_submit_command
  elif params.method == "htcondor":
    choice = get_htcondor_submit_command
  elif params.method == "custom":
    choice = get_custom_submit_command
  else:
    raise Sorry("Multiprocessing method %s not recognized" % params.method)
  command_generator = choice(command, submit_path, stdoutdir, params,
                             log_name=log_name, err_name=err_name, job_name=job_name, root_dir=root_dir)
  return command_generator()
