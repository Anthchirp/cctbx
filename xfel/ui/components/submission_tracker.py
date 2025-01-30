from __future__ import absolute_import, division, print_function

from libtbx import easy_run

class JobStopper(object):
  def __init__(self, queueing_system):
    self.queueing_system = queueing_system
    if self.queueing_system in ["mpi", "lsf"]:
      self.command = "bkill %s"
    elif self.queueing_system in ["pbs", "sge"]:
      self.command = "qdel %s"
    elif self.queueing_system == 'local':
      pass
    elif self.queueing_system == 'slurm' or self.queueing_system == 'shifter':
      # The current implementation of the shifter mp method assumes that we're
      # running on NERSC's systems => jobs should be tracked using the _slurm_
      # submission tracker.
      self.command = "scancel %s"
    elif self.queueing_system == 'htcondor':
      self.command = "condor_rm %s"
    else:
      raise NotImplementedError("job stopper not implemented for %s queueing system" \
      % self.queueing_system)

  def stop_job(self, submission_id):
    if not submission_id: return
    for sid in submission_id.split(','):
      if self.queueing_system == 'local':
        import psutil
        try:
          process = psutil.Process(int(sid))
        except psutil.NoSuchProcess:
          return
        for child in process.children(recursive=True): child.kill()
        process.kill()
      else:
        result = easy_run.fully_buffered(command=self.command%sid)
        status = "\n".join(result.stdout_lines)
        error = "\n".join(result.stderr_lines)
        print(status)
        print(error)

class QueueInterrogator(object):
  """A queue monitor that returns the status of a given queued job, or ERR if the job cannot
  be found in the queue."""
  def __init__(self, queueing_system):
    self.queueing_system = queueing_system
    if self.queueing_system in ["mpi", "lsf"]:
      self.command = "bjobs %s | grep %s | awk '{ print $3 }'"
    elif self.queueing_system == 'pbs':
      self.command = "qstat -H %s | tail -n 1 | awk '{ print $10 }'"
    elif self.queueing_system == 'sge':
      self.command = "qstat |grep %s | awk '{print $5}'" # previous one "qstat -j %s | awk '/job_state/ {print $3}'"
    elif self.queueing_system == 'local':
      pass
    elif self.queueing_system == 'slurm' or  self.queueing_system == 'shifter':
      # The current implementation of the shifter mp method assumes that we're
      # running on NERSC's systems => jobs should be tracked using the _slurm_
      # submission tracker.
      self.command = "sacct --job %s --format state --noheader"
    elif self.queueing_system == 'sfapi':
      self.command = None
    elif self.queueing_system == 'htcondor':
      self.command1 = "condor_q %s -nobatch | grep -A 1 OWNER | tail -n 1"
      self.command2 = "condor_history %s | grep -A 1 OWNER | tail -n 1"
    else:
      raise NotImplementedError(
      "queue interrogator not implemented for %s queueing system"%self.queueing_system)

  def query(self, submission_id):
    if self.queueing_system in ["mpi", "lsf"]:
      result = easy_run.fully_buffered(command=self.command % \
        (submission_id, submission_id))
    elif self.queueing_system == 'pbs':
      result = easy_run.fully_buffered(command=self.command%submission_id)
    elif self.queueing_system == 'sge':
      result = easy_run.fully_buffered(command=self.command%submission_id)
    elif self.queueing_system == 'local':
      import psutil
      try:
        process = psutil.Process(int(submission_id))
      except psutil.NoSuchProcess:
        return "DONE"
      statuses = [p.status() for p in [process] + process.children(recursive=True)]
      if 'running' in statuses: return "RUN"
      if 'sleeping' in statuses: return "RUN"
      # zombie jobs can be left because the GUI process that forked them is still running
      if len(statuses) == 1 and statuses[0] == 'zombie': return "DONE"
      return ", ".join(statuses)
    elif self.queueing_system in ['slurm', 'shifter', 'sfapi']:
      # The current implementation of the shifter mp method assumes that we're
      # running on NERSC's systems => jobs should be tracked using the _slurm_
      # submission tracker.
      if self.queueing_system in ['slurm', 'shifter']:
        result = easy_run.fully_buffered(command=self.command%submission_id)
        if len(result.stdout_lines) == 0: return 'UNKWN'
        status = result.stdout_lines[0].strip().rstrip('+')

      elif self.queueing_system == 'sfapi':
        from sfapi_client         import Client
        from sfapi_client.compute import Machine
        from sfapi_client.jobs    import JobState
        from xfel.util.sfapi_connector import KeyManager
        km = KeyManager()
        with Client(key=km.key) as client:
          machine = client.compute(Machine.perlmutter) # TODO: this could be a setting
          job = machine.job(submission_id)
          status = job.state.name

      statuses = {'COMPLETED': 'DONE',
                  'COMPLETING': 'RUN',
                  'FAILED': 'EXIT',
                  'PENDING': 'PEND',
                  'PREEMPTED': 'SUSP',
                  'RUNNING': 'RUN',
                  'SUSPENDED': 'SUSP',
                  'STOPPED': 'SUSP',
                  'CANCELLED': 'EXIT',
                  'TIMEOUT': 'TIMEOUT',
                 }
      return statuses[status] if status in statuses else 'UNKWN'

    elif self.queueing_system == 'htcondor':
      # (copied from the man page)
      # H = on hold, R = running, I = idle (waiting for a machine to execute on), C = completed,
      # X = removed, S = suspended (execution of a running job temporarily suspended on execute node),
      # < = transferring input (or queued to do so), and > = transferring output (or queued to do so).
      statuses = {'H': 'HOLD',
                  'R': 'RUN',
                  'I': 'PEND',
                  'C': 'DONE',
                  'X': 'DONE',
                  'S': 'SUSP',
                  '<': 'RUN',
                  '>': 'RUN'}
      for c in [self.command1, self.command2]:
        result = easy_run.fully_buffered(command=c%submission_id)
        if len(result.stdout_lines) != 1 or len(result.stdout_lines[0]) == 0: continue
        status = result.stdout_lines[0].split()[5]
        return statuses[status] if status in statuses else 'UNKWN'
      return 'ERR'
    status = "\n".join(result.stdout_lines)
    error = "\n".join(result.stderr_lines)
    if error != "" and not "Warning: job being submitted without an AFS token." in error:
      if "not found" in error:
        return "ERR"
      else:
        return error
    else:
      return status

  def get_mysql_server_hostname(self, submission_id):
    if self.queueing_system in ["mpi", "lsf"]:
      print("method to obtain hostname running MySQL server not implemented for ", self.queueing_system)
      pass
    elif self.queueing_system == 'pbs':
      print("method to obtain hostname running MySQL server not implemented for ", self.queueing_system)
      pass
    elif self.queueing_system == 'sge':
      print("method to obtain hostname running MySQL server not implemented for ", self.queueing_system)
      pass
    elif self.queueing_system == 'slurm' or self.queueing_system == "shifter":
      hostname_command = "sacct --job %s -o NODELIST --noheader | tail -n 1"
      result = easy_run.fully_buffered(command=hostname_command%submission_id).raise_if_errors()
      if result.stdout_lines:
        return result.stdout_lines[0].strip()
      else:
        return ""
    elif self.queueing_system == 'htcondor':
      print("method to obtain hostname running MySQL server not implemented for ", self.queueing_system)
      pass

class LogReader(object):
  """A log reader that distinguishes between expected results of successful and unsuccessful
  log file termination, and returns an error message if the log file cannot be found or read."""
  def __init__(self, queueing_system):
    self.queueing_system = queueing_system
    if self.queueing_system in ["mpi", "lsf", "pbs", "local", "sge"]:
      self.command = "tail -23 %s | head -1"
    elif self.queueing_system in ["slurm", "shifter", "htcondor", "sfapi"]:
      pass # no log reader used
    else:
      raise NotImplementedError(
      "queue interrogator not implemented for %s queueing system"%self.queueing_system)

  def read_result(self, log_path):
    result = easy_run.fully_buffered(command=self.command % log_path)
    status = "\n".join(result.stdout_lines)
    error = "\n".join(result.stderr_lines)
    if error != "" and not "Warning: job being submitted without an AFS token." in error:
      return "Error reading log file."
    else:
      return status

class SubmissionTracker(object):
  """An object that uses the QueueInterrogator and LogReader to query a queueing system and log
  file to determine the status of a queued job."""
  def __init__(self, params):
    self.queueing_system = params.mp.method
    self.interrogator = QueueInterrogator(self.queueing_system)
    self.reader = LogReader(self.queueing_system)
  def track(self, submission_id, log_path):
    if submission_id is None:
      return "UNKWN"
    all_statuses = [self._track(sid, log_path) for sid in submission_id.split(',')]
    if all_statuses and all([all_statuses[0] == s for s in all_statuses[1:]]):
      return all_statuses[0]
    else:
      return "UNKWN"

  def _track(self, submission_id, log_path):
    raise NotImplementedError("Override me!")

class LSFSubmissionTracker(SubmissionTracker):
  def _track(self, submission_id, log_path):
    from xfel.ui.db.job import known_job_statuses
    status = self.interrogator.query(submission_id)
    if status == "ERR":
      log_status = self.reader.read_result(log_path)
      if log_status == "Successfully completed.":
        return "DONE"
      elif "exit" in log_status.lower():
        return "EXIT"
      else:
        return "ERR" # error querying the queueing system
    elif status in known_job_statuses:
      return status
    else:
      print("Found an unknown status", status)

class PBSSubmissionTracker(SubmissionTracker):
  def _track(self, submission_id, log_path):
    status = self.interrogator.query(submission_id)
    if status == "F":
      return "DONE"
    elif status in ["Q", "H", "S"]:
      return "PEND"
    elif status in ["R", "T", "W", "E"]:
      return "RUN"
    else:
      print("Found an unknown status", status)

class SGESubmissionTracker(SubmissionTracker):
  def _track(self, submission_id, log_path):
    status = self.interrogator.query(submission_id)
    if status == "":
      return "DONE"
    elif status in ["s", "t", "S"]:
      return "PEND"
    elif status == "r":
      return "RUN"
    elif status == "qw":
      return "WAITING"
    else:
      print("Found an unknown status", status)

class SlurmSubmissionTracker(SubmissionTracker):
  def _track(self, submission_id, log_path):
    return self.interrogator.query(submission_id)

class HTCondorSubmissionTracker(SubmissionTracker):
  def _track(self, submission_id, log_path):
    return self.interrogator.query(submission_id)

class LocalSubmissionTracker(SubmissionTracker):
  def _track(self, submission_id, log_path):
    return self.interrogator.query(submission_id)

class TrackerFactory(object):
  @staticmethod
  def from_params(params):
    if params.mp.method in ['mpi', 'lsf']:
      return LSFSubmissionTracker(params)
    elif params.mp.method == 'pbs':
      return PBSSubmissionTracker(params)
    elif params.mp.method == 'sge' :
      return SGESubmissionTracker(params)
    elif params.mp.method == 'local':
      return LocalSubmissionTracker(params)
    elif params.mp.method == 'slurm':
      return SlurmSubmissionTracker(params)
    elif params.mp.method == 'shifter':
      # The current implementation of the shifter mp method assumes that we're
      # running on NERSC's systems => jobs should be tracked using the _slurm_
      # submission tracker.
      return SlurmSubmissionTracker(params)
    elif params.mp.method == 'sfapi':
      # The SFAPI queue plugin pretends to be slurm
      return SlurmSubmissionTracker(params)
    elif params.mp.method == 'htcondor':
      return HTCondorSubmissionTracker(params)
