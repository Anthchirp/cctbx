#!/usr/bin/env python
# -*- mode: python; coding: utf-8; indent-tabs-mode: nil; python-indent: 2 -*-

# Running bootstrap requires a minimum Python version of 2.6.

# To download this file:
# wget https://raw.githubusercontent.com/cctbx/cctbx_project/master/libtbx/auto_build/bootstrap.py
# or
# curl https://raw.githubusercontent.com/cctbx/cctbx_project/master/libtbx/auto_build/bootstrap.py > bootstrap.py

# Environment variables:
#   CCTBX_SKIP_CHEMDATA_CACHE_REBUILD - if this exists, the rotarama and cablam caches are not rebuilt

from __future__ import absolute_import, division, print_function

import ntpath
import os
import os.path
import platform
import posixpath
import re
import shutil
import socket as pysocket
import stat
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
import traceback
try: # Python 3
    from urllib.parse import urlparse
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError, URLError
except ImportError: # Python 2
    from urlparse import urlparse
    from urllib2 import urlopen, Request, HTTPError, URLError
import zipfile

try:
  import argparse
except ImportError:
  raise RuntimeError("""
The argparse module is required. If you are using Python 2.6, you may
need to install it separately. On CentOS 6, you can run

  yum install python-argpase

with administrative privileges.
""")

_BUILD_DIR = "build"  # set by arg parser further on down

windows_remove_list = []

rosetta_version_tar_bundle='rosetta_src_2018.33.60351_bundle'
rosetta_version_directory=rosetta_version_tar_bundle
# LICENSE REQUIRED
afitt_version="AFITT-2.4.0.4-redhat-RHEL7-x64" #binary specific to cci-vm-1
envs = {
  "PHENIX_ROSETTA_PATH" : ["modules", "rosetta"],
  "OE_EXE"              : ["modules", "openeye", "bin"],
  "OE_LICENSE"          : ["oe_license.txt"], # needed for license
}

# Utility function to be executed on slave machine or called directly by standalone bootstrap script
def tar_extract(workdir, archive, modulename=None):
  try:
    # delete tar target folder if it exists
    if modulename and os.path.exists(modulename):
      def remShut(*args):
        func, path, _ = args # onerror returns a tuple containing function, path and exception info
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
        os.remove(path)
      shutil.rmtree(modulename, onerror=remShut)
      # hack to work around possible race condition on Windows where deleted files may briefly
      # exist as phantoms and result in "access denied" error by subsequent IO operations
      cnt=0
      while os.path.exists(modulename):
        time.sleep(1)
        cnt = cnt + 1
        if cnt > 5:
          break
    # using tarfile module rather than unix tar command which is not platform independent
    tar = tarfile.open(os.path.join(workdir, archive), errorlevel=2)
    tar.extractall(path=workdir)
    tarfoldername = os.path.join(workdir, os.path.commonprefix(tar.getnames()).split('/')[0])
    tar.close()
    # take full permissions on all extracted files
    module = os.path.join(workdir, tarfoldername)
    for root, dirs, files in os.walk(module):
      for fname in files:
        full_path = os.path.join(root, fname)
        os.chmod(full_path, stat.S_IREAD | stat.S_IWRITE | stat.S_IRGRP | stat.S_IROTH)
    # rename to expected folder name, e.g. boost_hot -> boost
    # only rename if folder names differ
    if modulename:
      if modulename != tarfoldername:
        os.rename(tarfoldername, modulename)
  except Exception as e:
    raise Exception("Extracting tar archive resulted in error: " + str(e) + "\n" \
      + traceback.format_exc())
    return 1
  return 0

# Mock commands to run standalone, without buildbot.
class ShellCommand(object):
  def __init__(self, **kwargs):
    self.kwargs = kwargs

  def get_command(self):
    return self.kwargs['command']

  def get_description(self):
    if 'description' in self.kwargs:
      return self.kwargs['description']
    return None

  def get_workdir(self):
    return self.kwargs.get('workdir', _BUILD_DIR)

  def get_environment(self):
    # gets environment from kwargs
    env = self.kwargs.get('env', None)
    if env:
      for key, item in env.items():
        if item is None:
          env[key] = ''
        else:
          env[key] = os.path.abspath(item)
      rc = os.environ
      rc.update(env)
      env=rc
    return env

  def run(self):
    t0=time.time()
    command = self.get_command()
    description = self.get_description()
    workdir = self.get_workdir()
    env = self.get_environment()
    if not self.kwargs.get("quiet", False):
      if description:
        print("===== Running in %s:"%workdir, description)
      else:
        print("===== Running in %s:"%workdir, " ".join(command))
    if workdir:
      try:
        os.makedirs(workdir)
      except OSError:
        pass
    if command[0] == 'tar':
      # don't think any builders explicitly calls tar but leave it here just in case
      modname = None
      if len(command) > 3 and command[3]:
        modname = command[3]
      return tar_extract(workdir, command[2], modname)
    if command[0] == 'rm':
      # XXX use shutil rather than rm which is not platform independent
      for directory in command[2:]:
        if os.path.exists(directory):
          print('Deleting directory : %s' % directory)
          try: shutil.rmtree(directory)
          except OSError:
            print("Strangely couldn't delete %s" % directory)
      return 0
    if 0:
      print('command',command)
      print('workdir',workdir)
      print('env',env)
      print(os.environ.get("PATH", None))
    try:
      #if not os.path.isabs(command[0]):
        # executable path isn't located relative to workdir
      #  command[0] = os.path.join(workdir, command[0])
      stderr, stdout = None, None
      if self.kwargs.get("silent", False):
        stderr = stdout = open(os.devnull, 'wb')
      p = subprocess.Popen(
        args=command,
        cwd=workdir,
        stdout=stdout,
        stderr=stderr,
        env=env,
      )
    except Exception as e: # error handling
      if not self.kwargs.get('haltOnFailure'):
        return 1
      if isinstance(e, OSError):
        if e.errno == 2:
          executable = os.path.normpath(os.path.join(workdir, command[0]))
          raise RuntimeError("Could not run %s: File not found" % executable)
      if 'child_traceback' in dir(e):
        print("Calling subprocess resulted in error; ", e.child_traceback)
      raise e

    p.wait()
    if p.returncode != 0 and self.kwargs.get('haltOnFailure'):
      print("Process failed with return code %s"%(p.returncode))
      sys.exit(1)
    if 0:
      if description:
        outl = "%s - %s" % (workdir, description)
      else:
        outl = "%s - %s" % (workdir, " ".join(command))
      print('===== Time to %s : %0.1f' % (outl, time.time()-t0))
    return p.returncode

class Toolbox(object):
  @staticmethod
  def download_to_file(url, file, log=sys.stdout, status=True, cache=True):
    """Downloads a URL to file. Returns the file size.
       Returns -1 if the downloaded file size does not match the expected file
       size
       Returns -2 if the download is skipped due to the file at the URL not
       being newer than the local copy (identified by A. matching timestamp and
       size, or B. matching etag).
    """

    # Create directory structure if necessary
    if os.path.dirname(file):
      try:
        os.makedirs(os.path.dirname(file))
      except Exception:
        pass

    localcopy = os.path.isfile(file)

    # Get existing ETag, if present
    etag = None
    tagfile = '%s/.%s.etag' % os.path.split(os.path.abspath(file))
    if cache and os.path.isfile(tagfile):
      if not localcopy:
        # Having an ETag without a file is pointless
        os.remove(tagfile)
      else:
        tf = open(tagfile, 'r')
        etag = tf.readline()
        tf.close()

    try:
      import ssl
      from ssl import SSLError
    except ImportError:
      ssl = None
      SSLError = None

    # Open connection to remote server
    try:
      if sys.platform == "win32" and 'lbl.gov' in url:
# Downloading from http://cci.lbl.gov/cctbx_dependencies caused
# SSL: CERTIFICATE_VERIFY_FAILED error on Windows only as of today (why?).
# Quick and dirty hack to disable ssl certificate verification.
        try:
          _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
          # Legacy Python that doesn't verify HTTPS certificates by default
          pass
        except NameError:
          # ssl module was not loaded
          pass
        else:
          # Handle target environment that doesn't support HTTPS verification
          ssl._create_default_https_context = _create_unverified_https_context
      url_request = Request(url)
      if etag:
        url_request.add_header("If-None-Match", etag)
      if localcopy:
        # Shorten timeout to 7 seconds if a copy of the file is already present
        socket = urlopen(url_request, None, 7)
      else:
        socket = urlopen(url_request)
    except SSLError as e:
      # This could be a timeout
      if localcopy:
        # Download failed for some reason, but a valid local copy of
        # the file exists, so use that one instead.
        log.write("%s\n" % str(e))
        return -2
      # otherwise pass on the error message
      raise
    except (pysocket.timeout, HTTPError) as e:
      if isinstance(e, HTTPError) and etag and e.code == 304:
        # When using ETag. a 304 error means everything is fine
        log.write("local copy is current (etag)\n")
        return -2
      if localcopy:
        # Download failed for some reason, but a valid local copy of
        # the file exists, so use that one instead.
        log.write("%s\n" % str(e))
        return -2
      # otherwise pass on the error message
      raise
    except URLError as e:
      if localcopy:
        # Download failed for some reason, but a valid local copy of
        # the file exists, so use that one instead.
        log.write("%s\n" % str(e))
        return -2
      # if url fails to open, try using curl
      # temporary fix for old OpenSSL in system Python on macOS
      # https://github.com/cctbx/cctbx_project/issues/33
      command = ['/usr/bin/curl', '--http1.0', '-fLo', file, '--retry', '5', url]
      subprocess.call(command, shell=False)
      socket = None     # prevent later socket code from being run
      try:
        received = os.path.getsize(file)
      except OSError:
        raise RuntimeError("Download failed")

    if (socket is not None):
      try:
        file_size = int(socket.info().get('Content-Length'))
      except Exception:
        file_size = 0

      if os.path.isfile(tagfile):
        # ETag did not match, so delete any existing ETag.
        os.remove(tagfile)

      remote_mtime = 0
      try:
        remote_mtime = time.mktime(socket.info().getdate('last-modified'))
      except Exception:
        pass

      if (file_size > 0):
        if (remote_mtime > 0):
          # check if existing file matches remote size and timestamp
          try:
            (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(file)
            if (size == file_size) and (remote_mtime == mtime):
              log.write("local copy is current\n")
              socket.close()
              return -2
          except Exception:
            # proceed with download if timestamp/size check fails for any reason
            pass

        hr_size = (file_size, "B")
        if (hr_size[0] > 500): hr_size = (hr_size[0] / 1024, "kB")
        if (hr_size[0] > 500): hr_size = (hr_size[0] / 1024, "MB")
        log.write("%.1f %s\n" % hr_size)
        if status:
          log.write("    [0%")
          log.flush()

      received = 0
      block_size = 8192
      progress = 1
      # Allow for writing the file immediately so we can empty the buffer
      tmpfile = file + '.tmp'

      f = open(tmpfile, 'wb')
      while True:
        block = socket.read(block_size)
        received += len(block)
        f.write(block)
        if status and (file_size > 0):
          while (100 * received / file_size) > progress:
            progress += 1
            if (progress % 20) == 0:
              log.write("%d%%" % progress)
            elif (progress % 2) == 0:
              log.write(".")
            log.flush()

        if not block: break
      f.close()
      socket.close()

      if status and (file_size > 0):
        log.write("]\n")
      else:
        log.write("%d kB\n" % (received / 1024))
      log.flush()

      # Do not overwrite file during the download. If a download temporarily fails we
      # may still have a clean, working (yet older) copy of the file.
      shutil.move(tmpfile, file)

      if (file_size > 0) and (file_size != received):
        return -1

      if remote_mtime > 0:
        # set file timestamp if timestamp information is available
        from stat import ST_ATIME
        st = os.stat(file)
        atime = st[ST_ATIME] # current access time
        os.utime(file,(atime,remote_mtime))

      if cache and socket.info().get('ETag'):
        # If the server sent an ETAG, then keep it alongside the file
        open(tagfile, 'w').write(socket.info().get('ETag'))

    return received

  @staticmethod
  def unzip(archive, directory, trim_directory=0, verbose=False):
    '''unzip a file into a directory.'''
    if verbose:
      print("===== Installing %s into %s" % (archive, directory))
    if not zipfile.is_zipfile(archive):
      raise Exception("%s is not a valid .zip file" % archive)
    z = zipfile.ZipFile(archive, 'r')
    for member in z.infolist():
      is_directory = member.filename.endswith('/')
      filename = os.path.join(*member.filename.split('/')[trim_directory:])
      if filename != '':
        filename = os.path.normpath(filename)
        if '../' in filename:
          raise Exception('Archive %s contains invalid filename %s' % (archive, filename))
        filename = os.path.join(directory, filename)
        upperdirs = os.path.dirname(filename)
        try:
          if is_directory and not os.path.exists(filename):
            os.makedirs(filename)
          elif upperdirs and not os.path.exists(upperdirs):
            os.makedirs(upperdirs)
        except Exception: pass
        if not is_directory:
          source = z.open(member)
          target = open(filename, "wb")
          shutil.copyfileobj(source, target)
          target.close()
          source.close()

          # Preserve executable permission, if set
          unix_executable = member.external_attr >> 16 & 0o111
          # rwxrwxrwx => --x--x--x => 0o111
          if unix_executable:
            mode = os.stat(filename).st_mode
            mode |= (mode & 0o444) >> 2 # copy R bits to X
             # r--r--r-- => 0o444
            os.chmod(filename, mode)
    z.close()

  @staticmethod
  def set_git_repository_config_to_rebase(config):
    with open(config, 'r') as fh:
      cfg = fh.readlines()

    branch, remote, rebase = False, False, False
    insertions = []
    for n, line in enumerate(cfg):
      if line.startswith('['):
        if branch and remote and not rebase:
          insertions.insert(0, (n, branch))
        if line.startswith('[branch'):
          branch = line.split('"')[1]
        else:
          branch = False
        remote, rebase = False, False
      if re.match(r'remote\s*=', line.strip()):
        remote = True
      if re.match(r'rebase\s*=', line.strip()):
        rebase = True
    if branch and remote and not rebase:
      insertions.insert(0, (n + 1, branch))
    for n, branch in insertions:
      print("  setting branch %s to rebase" % branch)
      cfg.insert(n, '\trebase = true\n')
    with open(config, 'w') as fh:
      fh.write("".join(cfg))

  @staticmethod
  def git(module, parameters, destination=None, use_ssh=False, verbose=False, reference=None):
    '''Retrieve a git repository, either by running git directly
       or by downloading and unpacking an archive.'''
    git_available = True
    try:
      subprocess.call(['git', '--version'], stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
    except OSError:
      git_available = False

    # put molprobity repository at same level as "modules" to reproduce svn behavior
    if module == 'molprobity':
      destination = os.path.join('.', module)

    if destination is None:
      destination = os.path.join('modules', module)
    destpath, destdir = os.path.split(destination)

    # default to using ssh for private phenix repositories
    if module in ['elbow',
                  'ksdssp',
                  'labelit',
                  'muscle',
                  'opt_resources',
                  'phenix',
                  'phenix_pathwalker',
                  'Plex',
                  'PyQuante',
                  'pulchra',
                  'reel',
                  'solve_resolve',
                  ]:
      use_ssh = True

    if os.path.exists(destination):
      if git_available and os.path.exists(os.path.join(destination, '.git')):
        if not open(os.path.join(destination, '.git', 'HEAD'), 'r').read().startswith('ref:'):
          print("WARNING: Can not update existing git repository! You are not on a branch.")
          print("This may be legitimate when run eg. via Jenkins, but be aware that you cannot commit any changes")
          return

        else:
          # This may fail for unclean trees and merge problems. In this case manual
          # user intervention will be required.
          # For the record, you can clean up the tree and *discard ALL changes* with
          #   git reset --hard origin/master
          #   git clean -dffx
          return ShellCommand(
            command=['git', 'pull', '--rebase'], workdir=destination, silent=False, haltOnFailure=True).run()

      print("Existing non-git directory -- don't know what to do. skipping: %s" % module)
      if ('cctbx_project.git' in parameters[0]):
        print('\n' + '=' * 80 + '\nCCTBX moved to git on November 22, 2016.\n\nTo update cctbx_project to the last available subversion revision please run "svn update" while in the cctbx_project directory.\n' + '*'*80 + '\n')
      return
    if isinstance(parameters, str):
      parameters = [ parameters ]
    git_parameters = []
    for source_candidate in parameters:
      if source_candidate.startswith('-'):
        git_parameters = source_candidate.split(' ')
        continue
      if (not source_candidate.lower().startswith('http') and not use_ssh):
        continue
      if source_candidate.lower().endswith('.git'):
        if not git_available:
          continue
        reference_parameters = []
        if reference is not None:
          if os.path.exists(reference) and os.path.exists(os.path.join(reference, '.git')):
            reference_parameters = [ '--reference', reference ]
        cmd = [ 'git', 'clone', '--recursive' ] + git_parameters + [ source_candidate, destdir ] + reference_parameters
        if verbose:
          cmd = cmd + [ '--progress', '--verbose' ]
        returncode = ShellCommand(
          command=cmd, workdir=destpath, silent=False
        ).run()
        if returncode:
          return returncode # no point trying to continue on error
        if reference_parameters:
          # Sever the link between checked out and reference repository
          cmd = [ 'git', 'repack', '-a', '-d' ]
          returncode = ShellCommand(
            command=cmd, workdir=destination, silent=False
          ).run()
          try:
            os.remove(os.path.join(destination, '.git', 'objects', 'info', 'alternates'))
          except OSError:
            returncode = 1
        Toolbox.set_git_repository_config_to_rebase(os.path.join(destination, '.git', 'config'))
        if returncode:
          return returncode # no point trying to continue on error
        # Show the hash for the checked out commit for debugging purposes, ignore any failures.
        ShellCommand(
          command=[ 'git', 'rev-parse', 'HEAD' ], workdir=destination, silent=False
        ).run()
        return returncode
      filename = "%s-%s" % (module,
                            urlparse(source_candidate)[2].split('/')[-1])
      filename = os.path.join(destpath, filename)
      if verbose:
        print("===== Downloading %s: " % source_candidate, end=' ')
      Toolbox.download_to_file(source_candidate, filename)
      Toolbox.unzip(filename, destination, trim_directory=1, verbose=verbose)
      return

    error = "Cannot satisfy git dependency for module %s: None of the sources are available." % module
    if not git_available:
      print(error)
      error = "A git installation has not been found."
    raise Exception(error)

class cleanup_ext_class(object):
  def __init__(self, filename_ext, workdir=None, walk=True):
    self.filename_ext = filename_ext
    self.workdir = workdir
    self.walk = walk

  def get_command(self):
    return "delete *%s in %s" % (self.filename_ext, self.workdir).split()

  def remove_ext_files(self):
    cwd=os.getcwd()
    if self.workdir is not None:
      if os.path.exists(self.workdir):
        os.chdir(self.workdir)
      else:
        return
    print("\n  removing %s files in %s, walk? %s" % (self.filename_ext,
                                                     os.getcwd(),
                                                     self.walk,
      ))
    i=0
    if self.walk:
      for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
          if name.endswith(self.filename_ext):
            os.remove(os.path.join(root, name))
            i+=1
    else:
      for name in os.listdir(os.getcwd()):
        if name.endswith(self.filename_ext):
          os.remove(os.path.join(name))
          i+=1
    os.chdir(cwd)
    print("  removed %d files" % i)

  def run(self):
    self.remove_ext_files()

class cleanup_dirs(object):
  """Command to remove unwanted subdirectories"""

  def __init__(self, dirs, workdir=None):
    """
    :param dirs:    List of subdirectories to remove from workdir
    :param workdir: The root directory for everything in dirs. If None, then
                    the command will be run in the current working directory.
    """
    self.dirs = dirs
    self.workdir = workdir

  def get_command(self):
    return "cleanup dirs in %s" % (self.workdir).split()

  def remove_dirs(self):
    cwd=os.getcwd()
    try:
      # Move to the workdir
      if self.workdir is not None:
        if os.path.exists(self.workdir):
          os.chdir(self.workdir)
        else:
          return

      # Don't notify the user if we aren't doing anything
      if any(os.path.exists(d) for d in self.dirs):
        print("===== Removing directories in %s" % (os.getcwd()))

        for d in self.dirs:
          if os.path.exists(d):
            print("      removing %s" % (os.path.join(os.getcwd(),d)))
            shutil.rmtree(d)
    finally:
      # Leave the directory untouched even if we failed
      os.chdir(cwd)

  def run(self):
    self.remove_dirs()

##### Modules #####
class SourceModule(object):
  _modules = {}
  module = None
  authenticated = None
  authentarfile = None
  anonymous = None
  def __init__(self):
    if not self._modules:
      self.update_subclasses()

  def items(self):
    return list(self._modules.items())

  @classmethod
  def update_subclasses(cls):
    for i in cls.__subclasses__():
      cls._modules[i.module] = i

  def get_module(self, module):
    if module in self._modules:
      return self._modules[module]
    raise KeyError("Unknown module: %s"%module)

  def get_url(self, auth=None):
    repo = None
    try:
      repo = self.get_authenticated(auth=auth)
    except KeyError as e:
      repo = self.get_anonymous()
      if not repo:
        raise Exception('No anonymous access method defined for module: %s. Try with --%s'%(self.module, e.args[0]))
    repo = repo or self.get_anonymous()
    if not repo:
      raise Exception('No access method defined for module: %s'%self.module)
    return repo

  def get_authenticated(self, auth=None):
    auth = auth or {}
    if not self.authenticated:
      return None
    return [self.authenticated[0], self.authenticated[1]%auth]

  def get_tarauthenticated(self, auth=None):
    auth = auth or {}
    if self.authentarfile:
      return [self.authentarfile[0]%auth, self.authentarfile[1], self.authentarfile[2]]
    return None, None, None

  def get_anonymous(self):
    return self.anonymous

# Core external repositories
# The trailing slashes ARE significant.
# These must all provide anonymous access.
# On Windows due to absence of rsync we use pscp from the Putty programs.
class ccp4io_module(SourceModule):
  module = 'ccp4io'
  anonymous = ['git',
               'git@github.com:cctbx/ccp4io.git',
               'https://github.com/cctbx/ccp4io.git',
               'https://github.com/cctbx/ccp4io/archive/master.zip']

class annlib_module(SourceModule):
  module = 'annlib'
  anonymous = ['git',
               'git@github.com:cctbx/annlib.git',
               'https://github.com/cctbx/annlib.git',
               'https://github.com/cctbx/annlib/archive/master.zip']

class scons_module(SourceModule):
  module = 'scons'
  anonymous = ['git', '-b 3.1.1',
               'https://github.com/SCons/scons/archive/3.1.1.zip']

# external modules
class rosetta_class(SourceModule):
  module = 'rosetta'
  authenticated = [
    'rsync',
    '%(cciuser)s@boa.lbl.gov:/net/cci/auto_build/externals/'+rosetta_version_tar_bundle+'/',
  ]
  authenticated = [
    'scp',
    '%(cciuser)s@boa.lbl.gov:/net/cci-filer2/raid1/auto_build/externals/'+rosetta_version_tar_bundle+'.tgz']

class afitt_class(SourceModule):
  module = 'afitt'
  authenticated = [
    'scp',
    '%(cciuser)s@boa.lbl.gov:/net/cci-filer2/raid1/auto_build/externals/'+afitt_version+'.gz']

# Core CCTBX repositories
# These must all provide anonymous access.
class cctbx_module(SourceModule):
  module = 'cctbx_project'
  anonymous = ['git',
               'git@github.com:cctbx/cctbx_project.git',
               'https://github.com/cctbx/cctbx_project.git',
               'https://github.com/cctbx/cctbx_project/archive/master.zip']

class amber_adaptbx_module(SourceModule):
  module = 'amber_adaptbx'
  anonymous = ['git',
               'git@github.com:phenix-project/amber_adaptbx.git',
               'https://github.com/phenix-project/amber_adaptbx.git',
               ]

class amber_library_module(SourceModule):
  module = 'amber_library'
  anonymous = ['git',
               'git@github.com:phenix-project/amber_library.git',
               'https://github.com/phenix-project/amber_library.git',
               ]

class aimnet2calc_module(SourceModule):
  module = 'aimnet2calc'
  anonymous = ['git',
               'git@github.com:zubatyuk/aimnet2calc.git',
               'https://github.com/zubatyuk/aimnet2calc.git',
               ]

class qrefine_module(SourceModule):
  module = 'qrefine'
  anonymous = ['git',
               'git@github.com:qrefine/qrefine.git',
               'https://github.com/qrefine/qrefine.git',
               ]

class pydiscamb_module(SourceModule):
  module = 'pyDiSCaMB'
  anonymous = ['git',
               'git@github.com:viljarjf/pyDiSCaMB.git',
               'https://github.com/viljarjf/pyDiSCaMB.git',
               ]

class mon_lib_module(SourceModule):
  module = 'mon_lib'
  anonymous = ['curl', 'http://boa.lbl.gov/repositories/mon_lib.gz']
  authentarfile = ['%(cciuser)s@boa.lbl.gov', 'mon_lib.tar.gz', '/net/cci/auto_build/repositories/mon_lib']
  #authenticated = ['rsync', '%(cciuser)s@boa.lbl.gov:/net/cci/auto_build/repositories/annlib/']

class geostd_module(SourceModule):
  module = 'geostd'
  anonymous = ['git',
               'git@github.com:phenix-project/geostd.git',
               'https://github.com/phenix-project/geostd.git'
               ]

class boost_module(SourceModule):
  module = 'boost'
  anonymous = ['git',
               'git@github.com:cctbx/boost.git',
               'https://github.com/cctbx/boost.git',
               'https://github.com/cctbx/boost/archive/master.zip']

class cbflib_module(SourceModule):
  module = 'cbflib'
  anonymous = ['git',
               'git@github.com:dials/cbflib.git',
               'https://github.com/dials/cbflib.git',
               'https://github.com/dials/cbflib/archive/main.zip']

class ccp4io_adaptbx(SourceModule):
  module = 'ccp4io_adaptbx'
  anonymous = ['git',
               'git@github.com:cctbx/ccp4io_adaptbx.git',
               'https://github.com/cctbx/ccp4io_adaptbx.git',
               'https://github.com/cctbx/ccp4io_adaptbx/archive/master.zip']

class annlib_adaptbx(SourceModule):
  module = 'annlib_adaptbx'
  anonymous = ['git',
               'git@github.com:cctbx/annlib_adaptbx.git',
               'https://github.com/cctbx/annlib_adaptbx.git',
               'https://github.com/cctbx/annlib_adaptbx/archive/master.zip']

class tntbx_module(SourceModule):
  module = 'tntbx'
  anonymous = ['git',
               'git@github.com:cctbx/tntbx.git',
               'https://github.com/cctbx/tntbx.git',
               'https://github.com/cctbx/tntbx/archive/master.zip']

class clipper_module(SourceModule):
  module = 'clipper'
  anonymous = ['git',
               'git@github.com:cctbx/clipper.git',
               'https://github.com/cctbx/clipper.git',
               'https://github.com/cctbx/clipper/archive/master.zip']

class gui_resources_module(SourceModule):
  module = 'gui_resources'
  anonymous = ['git',
               'git@github.com:cctbx/gui_resources.git',
               'https://github.com/cctbx/gui_resources.git',
               'https://github.com/cctbx/gui_resources/archive/master.zip']

class opt_resources_module(SourceModule):
  module = 'opt_resources'
  authenticated = ['git', 'git@github.com:phenix-project/opt_resources.git']

class eigen_module(SourceModule):
  module = 'eigen'
  anonymous = ['git', '-b 3.4.0',
               'https://gitlab.com/libeigen/eigen.git']

# Phenix repositories
class phenix_module(SourceModule):
  module = 'phenix'
  anonymous = ['git', 'git@github.com:phenix-project/phenix.git']

class phenix_html(SourceModule):
  module = 'phenix_html'
  anonymous = ['git',
               'git@github.com:phenix-project/phenix_html.git',
               'https://github.com/phenix-project/phenix_html.git']

class phenix_dev_doc(SourceModule):
  module = 'phenix_dev_doc'
  anonymous = ['git',
               'git@github.com:phenix-project/phenix_dev_doc.git',
               'https://github.com/phenix-project/phenix_dev_doc.git']

class phenix_examples(SourceModule):
  module = 'phenix_examples'
  anonymous = ['git',
               'git@gitlab.com:phenix_project/phenix_examples.git',
               'https://gitlab.com/phenix_project/phenix_examples.git']

class phenix_regression(SourceModule):
  module = 'phenix_regression'
  anonymous = ['git',
               'git@gitlab.com:phenix_project/phenix_regression.git',
               'https://gitlab.com/phenix_project/phenix_regression.git']

class phenix_colabs(SourceModule):
  module = 'Colabs'
  anonymous = ['git',
               'git@github.com:phenix-project/Colabs.git',
               'https://github.com/phenix-project/Colabs.git']

class plex_module(SourceModule):
  module = 'Plex'
  authenticated = ['git', 'git@github.com:phenix-project/Plex.git']

class pyquante_module(SourceModule):
  module = 'PyQuante'
  authenticated = ['git', 'git@github.com:phenix-project/PyQuante.git']

class chem_data_module(SourceModule):
  module = 'chem_data'
  anonymous = ['git',
               'git@gitlab.com:phenix_project/chem_data.git',
               'https://gitlab.com/phenix_project/chem_data.git']

class elbow_module(SourceModule):
  module = 'elbow'
  authenticated = ['git', 'git@github.com:phenix-project/elbow.git']

class ksdssp_module(SourceModule):
  module = 'ksdssp'
  authenticated = ['git', 'git@github.com:phenix-project/ksdssp.git']

class pulchra_module(SourceModule):
  module = 'pulchra'
  authenticated = ['git', 'git@github.com:phenix-project/pulchra.git']

class solve_resolve_module(SourceModule):
  module = 'solve_resolve'
  anonymous = ['git', 'git@github.com:phenix-project/solve_resolve.git']

class reel_module(SourceModule):
  module = 'reel'
  authenticated = ['git', 'git@github.com:phenix-project/reel.git']

class muscle_module(SourceModule):
  module = 'muscle'
  authenticated = ['git', 'git@github.com:phenix-project/muscle.git']

class cxi_xdr_xes_module(SourceModule):
  module = 'cxi_xdr_xes'
  authenticated = ['svn', 'svn+ssh://%(cciuser)s@boa.lbl.gov/cxi_xdr_xes/trunk']

class buildbot_module(SourceModule):
  module = 'buildbot'
  authenticated = ['git', 'git@github.com:cci-lbl/buildbot.git']

class phenix_pathwalker_module(SourceModule):
  module = 'phenix_pathwalker'
  anonymous = ['git', 'git@github.com:phenix-project/phenix_pathwalker.git']

class alphafold_module(SourceModule):
  module = 'alphafold'
  anonymous = ['git',
               'git@github.com:google-deepmind/alphafold.git',
               'https://github.com/google-deepmind/alphafold.git']

# Phaser repositories
class phaser_module(SourceModule):
  module = 'phaser'
  anonymous = ['git',
               'git@gitlab.developers.cam.ac.uk:scm/haematology/readgroup/phaser.git',
               'https://gitlab.developers.cam.ac.uk/scm/haematology/readgroup/phaser.git']

class phasertng_module(SourceModule):
  module = 'phasertng'
  anonymous = ['git',
               'git@gitlab.developers.cam.ac.uk:scm/haematology/readgroup/phasertng.git',
               'https://gitlab.developers.cam.ac.uk/scm/haematology/readgroup/phasertng.git']

class phaser_voyager_module(SourceModule):
  module = 'phaser_voyager'
  anonymous = ['git',
               'git@gitlab.developers.cam.ac.uk:scm/haematology/readgroup/phaser_voyager.git',
               'https://gitlab.developers.cam.ac.uk/scm/haematology/readgroup/phaser_voyager.git']

class phaser_regression_module(SourceModule):
  module = 'phaser_regression'
  anonymous = ['git',
               'git@gitlab.developers.cam.ac.uk:scm/haematology/readgroup/phaser_regression.git',
               'https://gitlab.developers.cam.ac.uk/scm/haematology/readgroup/phaser_regression.git']

class voyager_regression_module(SourceModule):
  module = 'voyager_regression'
  anonymous = ['git',
               'git@gitlab.developers.cam.ac.uk:scm/haematology/readgroup/voyager_regression.git',
               'https://gitlab.developers.cam.ac.uk/scm/haematology/readgroup/voyager_regression.git']

# DIALS repositories
class labelit_module(SourceModule):
  module = 'labelit'
  anonymous = ['git', 'git@github.com:phenix-project/labelit.git']

class labelit_regression_module(SourceModule):
  module = 'labelit_regression'
  anonymous = ['git',
               'git@gitlab.com:phenix_project/labelit_regression.git',
               'https://gitlab.com/phenix_project/labelit_regression.git']

class dials_module(SourceModule):
  module = 'dials'
  anonymous = ['git',
               'git@github.com:dials/dials.git',
               'https://github.com/dials/dials.git',
               'https://github.com/dials/dials/archive/main.zip']

class dxtbx_module(SourceModule):
  module = 'dxtbx'
  anonymous = ['git',
               'git@github.com:cctbx/dxtbx.git',
               'https://github.com/cctbx/dxtbx.git',
               'https://github.com/cctbx/dxtbx/archive/main.zip']

class dials_regression_module(SourceModule):
  module = 'dials_regression'
  authenticated = ['svn',
                   'svn+ssh://%(cciuser)s@boa.lbl.gov/dials_regression/trunk']

class iota_module(SourceModule):
  module = 'iota'
  anonymous = ['git',
               'git@github.com:ssrl-px/iota.git',
               'https://github.com/ssrl-px/iota.git',
               'https://github.com/ssrl-px/iota/archive/master.zip']

class msgpack_module(SourceModule):
  module = 'msgpack'
  anonymous = ['curl', [
    "https://github.com/dials/dependencies/raw/dials-1.13/msgpack-3.1.1.tar.gz",
  ]]

class xfel_regression_module(SourceModule):
  module = 'xfel_regression'
  authenticated = ['git',
                   'git@gitlab.com:cctbx/xfel_regression.git',
                   'https://gitlab.com/cctbx/xfel_regression.git']

class xia2_module(SourceModule):
  module = 'xia2'
  anonymous = ['git',
               'git@github.com:xia2/xia2.git',
               'https://github.com/xia2/xia2.git',
               'https://github.com/xia2/xia2/archive/main.zip']

class kokkos_module(SourceModule):
  module = 'kokkos'
  anonymous = ['git', '-b 4.2.00',
               'git@github.com:kokkos/kokkos.git',
               'https://github.com/kokkos/kokkos.git',
               'https://github.com/kokkos/kokkos/archive/refs/tags/4.2.00.zip']

class kokkos_kernels_module(SourceModule):
  module = 'kokkos-kernels'
  anonymous = ['git', '-b 4.2.00',
               'git@github.com:kokkos/kokkos-kernels.git',
               'https://github.com/kokkos/kokkos-kernels.git',
               'https://github.com/kokkos/kokkos-kernels/archive/refs/tags/4.2.00.zip']

# Duke repositories
class probe_module(SourceModule):
  module = 'probe'
  anonymous = ['git', 'https://github.com/rlabduke/probe.git']

class reduce_module(SourceModule):
  # Version 4.14 or later should be used to avoid mmtbx_reduce_ext name conflict.
  module = 'reduce'
  anonymous = ['git', 'https://github.com/rlabduke/reduce.git']

class king_module(SourceModule):
  module = 'king'
  anonymous = ['git',
               'https://github.com/rlabduke/phenix_king_binaries.git']

class molprobity_module(SourceModule):
  module = 'molprobity'
  anonymous = ['git', 'https://github.com/rlabduke/MolProbity.git']

class uc_metrics_module(SourceModule):
  module = 'uc_metrics'
  anonymous = ['git',
               'git@gitlab.com:cctbx/uc_metrics.git',
               'https://gitlab.com/cctbx/uc_metrics.git']

class ncdist_module(SourceModule):
  module = 'ncdist'
  anonymous = ['git',
               'git@github.com:yayahjb/ncdist.git',
               'https://github.com/yayahjb/ncdist.git',
               'https://github.com/yayahjb/ncdist/archive/master.zip']

MODULES = SourceModule()

###################################
##### Base Configuration      #####
###################################

class Builder(object):
  """Create buildbot configurations for CCI and CCTBX-like software."""
  # Base packages
  BASE_PACKAGES = 'all'
  # Checkout these codebases
  CODEBASES = ['cctbx_project']
  CODEBASES_EXTRA = []
  # Copy these sources from cci.lbl.gov
  HOT = []
  HOT_EXTRA = []
  # Configure for these cctbx packages
  LIBTBX = ['cctbx']
  LIBTBX_EXTRA = []

  def __init__(
      self,
      category=None,
      subcategory=None,
      platform=None,
      sep=None,
      python_base=None,
      cleanup=False,
      hot=True,
      update=True,
      revert=None,
      base=True,
      build=True,
      tests=True,
      doc=True,
      distribute=False,
      auth=None,
      with_python=None,
      nproc=1,
      verbose=False,
      download_only=False,
      skip_base="",
      force_base_build=False,
      enable_shared=False,
      mpi_build=False,
      python3=False,
      wxpython4=False,
      config_flags=[],
      use_conda=None,
      python='27',
      no_boost_src=False,
    ):
    if nproc is None:
      self.nproc=1
    else:
      self.nproc=nproc
    """Create and add all the steps."""
    # self.cciuser = cciuser or getpass.getuser()
    self.set_auth(auth)
    self.steps = []
    self.category = category
    self.subcategory = subcategory
    if self.subcategory: self.EXTERNAL_CODEBASES = [self.subcategory]
    self.platform = platform
    if self.isPlatformWindows():
      self.op = ntpath
    else:
      self.op = os.path
    self.name = '%s-%s'%(self.category, self.platform)
    # Platform configuration.
    python_executable = 'python'
    self.python3 = python.startswith('3')
    if python3:
      python_executable = 'python3'
    self.wxpython4 = wxpython4
    if self.platform and ('windows' in self.platform or self.platform == 'win32'):
      python_executable = python_executable + '.exe'
    if self.platform and 'windows' in self.platform:
      self.python_base = self.opjoin(*['..', 'base', 'bin', 'python', python_executable])
    elif sys.platform == "win32": # assuming we run standalone without buildbot
      self.python_base = self.opjoin(*[os.getcwd(), 'base', 'bin', 'python', python_executable])
    else:
      self.python_base = self.opjoin(*['..', 'base', 'bin', python_executable])
    self.with_python = with_python
    if self.with_python:
      self.python_base = with_python
    self.verbose = verbose
    self.download_only = download_only
    self.skip_base = skip_base
    self.force_base_build = force_base_build
    # self.config_flags are only from the command line
    # get_libtbx_configure can still be used to always set flags specific to a
    # builder
    self.config_flags = config_flags
    self.use_conda = use_conda
    self.python = python
    self.no_boost_src = no_boost_src
    self.add_init()

    # Cleanup
    if cleanup:
      self.cleanup(['dist', 'tests', 'doc', 'tmp', 'base', 'base_tmp', _BUILD_DIR,
                    'conda_base'])
    else:
      self.cleanup(['dist', 'tests', 'tmp'])

    if self.platform and 'windows' in self.platform: # only executed by buildbot master
      from buildbot.steps.transfer import FileDownload
      # download us to folder above modules on slave so we can run the utility functions defined above
      self.add_step(FileDownload(mastersrc="bootstrap.py", slavedest="../bootstrap.py"))

    # Add 'hot' sources
    if hot:
      # conda builds do not need eigen (disabled), scons
      hot = self.get_hot()
      if self.use_conda is not None:
        for module in ['scons']:
          # SCons conda package may cause issues with procrunner on Python 2.7
          # https://stackoverflow.com/questions/24453387/scons-attributeerror-builtin-function-or-method-object-has-no-attribute-disp
          if module == 'scons' and self.python == '27':
            continue
          try:
            hot.remove(module)
          except ValueError:
            pass
      list(map(self.add_module, hot))

    # Add svn sources.
    self.revert=revert
    if update:
      # check if boost needs to be downloaded
      codebases = self.get_codebases()
      if self.no_boost_src:
        try:
          codebases.remove('boost')
        except ValueError:
          pass
      list(map(self.add_module, codebases))

    # always remove .pyc files
    self.remove_pyc()

    # Build base packages
    if base:
      extra_opts = ["--nproc=%s" % str(self.nproc)]
      if enable_shared:
        extra_opts.append("--python-shared")
      if mpi_build:
        extra_opts.append("--mpi-build")
      self.add_base(extra_opts=extra_opts)

    # Configure, make, get revision numbers
    if build and not self.download_only:
      self.add_configure()
      self.add_make()
      self.add_install()

    # Tests, tests
    if tests and not self.download_only:
      self.add_tests()

    # docs
    if doc:
      self.rebuild_docs()

    # Distribute
    if distribute and not self.download_only:
      self.add_distribute()

    # Distribute does this but uses correct PHENIX_VERSION
    if build and not self.download_only:
      self.add_dispatchers()
      self.add_refresh()

    if self.platform and 'windows' in self.platform: # only executed by buildbot master
      self.add_rm_bootstrap_on_slave()

  def isPlatformWindows(self):
    if self.platform and 'windows' in self.platform:
        return True
    else:
      if self.platform == "dev" and sys.platform == "win32":
        return True
    return False

  def isPlatformLinux(self):
    if self.platform and 'linux' in self.platform:
        return True
    else:
      if self.platform == "dev" and sys.platform.startswith("linux"):
        return True
    return False

  def isPlatformMacOSX(self):
    if self.platform and 'mac' in self.platform:
        return True
    else:
      if self.platform == "dev" and sys.platform.startswith("darwin"):
        return True
    return False

  def add_auth(self, account, username):
    self.auth[account] = username

  def set_auth(self, auth):
    self.auth = auth or {}

  def get_auth(self):
    return self.auth

  def remove_pyc(self):
    self.add_step(cleanup_ext_class(".pyc", "modules"))

  def shell(self, **kwargs):
    # Convenience for ShellCommand
    kwargs['haltOnFailure'] = kwargs.pop('haltOnFailure', True)
    kwargs['description'] = kwargs.get('description') or kwargs.get('name')
    kwargs['timeout'] = 60*60*2 # 2 hours
    if 'workdir' in kwargs:
      kwargs['workdir'] = self.opjoin(*kwargs['workdir'])
    return ShellCommand(**kwargs)

  def run(self):
    for i in self.steps:
      i.run()

  def opjoin(self, *args):
    return self.op.join(*args)

  def get_codebases(self):
    if self.isPlatformWindows():
      rc = set(self.CODEBASES+self.CODEBASES_EXTRA)
      for r in windows_remove_list: rc = rc - set([r])
      return list(rc)
    rc = self.CODEBASES + self.CODEBASES_EXTRA
    if hasattr(self, "EXTERNAL_CODEBASES"):
      rc = self.EXTERNAL_CODEBASES + rc
    return rc

  def get_hot(self):
    return self.HOT + self.HOT_EXTRA

  def get_libtbx_configure(self): # modified in derived class PhenixBuilder
    return self.LIBTBX + self.LIBTBX_EXTRA

  def add_init(self):
    pass

  def cleanup(self, dirs=None):
    dirs = dirs or []
    if self.isPlatformWindows():
      # Delete folders by copying an empty folder with ROBOCOPY is more reliable on Windows
      # If ROBOCOPY fails i.e. ERRORLEVEL>0 then bail to stop bootstrap. Start cmd.exe with
      cmd = ['cmd', '/C', 'mkdir', 'empty', '&',
         '(FOR', '%b', 'IN', '('] + dirs + [')', 'DO',
              '((ROBOCOPY', 'empty', '%b', '/MIR', '/COPYALL', '>', 'nul)',
                 '&', 'rmdir', '/S', '/Q', '%b\\', # remove directory after robocopy
                 '&', '@IF', 'EXIST', '%b\\', # backslash indicates it's a directory and not a file
                         '(echo.', '&', 'echo', 'Error', 'deleting', '%b',
                          '&', 'echo.', '&', 'exit', '/B', '42', ')))',
          '&', 'rmdir', 'empty'
       ]
      self.add_step(self.shell(
        name='Removing directories ' + ', '.join(dirs),
        command =cmd,
        workdir=['.'],
        description="deleting " + ", ".join(dirs),
      ))
    else:
      self.add_step(cleanup_dirs(dirs, "modules"))

  def add_rm_bootstrap_on_slave(self):
    # if file is not found error flag is set. Mask it with cmd shell
    cmd=['cmd', '/c', 'del', '/Q', "bootstrap.py*", '&', 'set', 'ERRORLEVEL=0']
    self.add_step(self.shell(
      name='removing bootstrap utilities',
      command =cmd,
      workdir=['.'],
      description="remove temporary bootstrap.py*",
    ))

  def add_step(self, step):
    """Add a step."""
    self.steps.append(step)
    if 0:
      print("commands "*8)
      for step in self.steps:
        print(step)
        #try:    print " ".join(step.get_command())
        #except: print '????'
      print("commands "*8)

  def add_module(self, module, workdir=None, module_directory=None):
    action = MODULES.get_module(module)().get_url(auth=self.get_auth())
    method, parameters = action[0], action[1:]
    if len(parameters) == 1: parameters = parameters[0]
    tarurl, arxname, dirpath = None, None, None
    if self.isPlatformWindows() and (method == "authenticated" or method == "rsync"):
      tarurl, arxname, dirpath = MODULES.get_module(module)().get_tarauthenticated(auth=self.get_auth())
    if self.isPlatformWindows():
      if module in windows_remove_list:
        return
    if method == 'rsync' and not self.isPlatformWindows():
      self._add_rsync(module,
                      parameters, # really the url
                      workdir=workdir,
                      module_directory=module_directory)
    elif self.isPlatformWindows() and tarurl:
      # if more bootstraps are running avoid potential race condition on
      # remote server by using unique random filenames
      randarxname = next(tempfile._get_candidate_names()) + "_" + arxname
      self._add_remote_make_tar(module, tarurl, randarxname, dirpath)
      self._add_scp(module, tarurl + ':' + randarxname)
      self._add_remote_rm_tar(module, tarurl, randarxname)
    elif method == 'scp':
      self._add_scp(module, parameters)
    elif method == 'curl':
      self._add_curl(module, parameters)
    elif method == 'svn':
      self._add_svn(module, parameters)
    elif method == 'git':
      self._add_git(module, parameters)
    else:
      raise Exception('Unknown access method: %s %s'%(method, str(parameters)))

  def _add_rsync(self, module, url, workdir=None, module_directory=None):
    """Add packages not in source control."""
    # rsync the hot packages.
    if not workdir: workdir=["modules"]
    if not module_directory: module_directory=module
    self.add_step(self.shell(
      name='hot %s'%module,
      command=[
        'rsync',
        '-rptgoDLK', #'-aL',
        '--delete',
        url,
        module_directory,
      ],
      workdir=workdir,
    ))

  def _add_remote_make_tar(self, module, tarurl, arxname, dirpath):
    """Windows: tar up hot packages for quick file transfer since there's no rsync and pscp is painfully slow"""
    if dirpath[-1] == '/':
      dirpath = dirpath[:-1]
    basename = posixpath.basename(dirpath)
    cmd=[
        'ssh',
        tarurl,
        '"' + 'cd',
        posixpath.split(dirpath)[0],
        '&&',
        'tar',
        'cfzh',
        '~/' + arxname,
        basename + '"'
      ]
    mstr= " ".join(cmd)
    self.add_step(self.shell( # pack directory with tar on remote system
      name='hot %s'%module,
      command=mstr,
      workdir=['modules'],
      description="create remote temporary archive %s:%s" %(tarurl, arxname),
    ))

  def _add_remote_rm_tar(self, module, tarurl, arxname):
    """Windows: Delete tar file on remote system, unpack tar file locally, then delete tar file locally"""
    self.add_step(self.shell( # delete the tarfile on remote system
      name='hot %s'%module,
      command=[
        'ssh',
        tarurl,
        'rm ',
        arxname
      ],
      workdir=['modules'],
      description="delete remote temporary archive of %s" %module,
    ))
    self.add_step(self.shell(command=[
      sys.executable,"-c","import sys; sys.path.append('..'); import bootstrap; \
      bootstrap.tar_extract('','%s', '%s')" %(arxname, module) ],
      workdir=['modules'],
      description="extracting archive files to %s" %module,
    ))
    self.add_step(self.shell( # delete the tarfile locally
      # use 'cmd', '/c' as a substitute for shell=True in the subprocess.Popen call
      command=['cmd', '/c', 'del', arxname],
      workdir=['modules'],
      description="delete local temporary archive of %s" %module,
    ))

  def _add_scp(self, module, url):
    self.add_step(self.shell(
      name='hot %s'%module,
      command=[
        'scp',
        '-r',
        url,
        '.',
      ],
      workdir=['modules'],
      description="getting remote file %s" %url.split("/")[-1],
    ))

  def _add_download(self, url, to_file):
    if not isinstance(url, list):
      url = [url]
    class _download(object):
      def run(self):
        for _url in url:
          for retry in (3,3,0):
            print("===== Downloading %s: " % _url, end=' ')
            try:
              Toolbox().download_to_file(_url, to_file)
              return
            except Exception as e:
              print("Download failed with", e)
              if retry:
                print("Retrying in %d seconds" % retry)
                time.sleep(retry)
        raise RuntimeError("Could not download " + to_file)
    self.add_step(_download())

  def _add_curl(self, module, url):
    if isinstance(url, list):
      filename = urlparse(url[0])[2].split('/')[-1]
    else:
      filename = urlparse(url)[2].split('/')[-1]
    # Google Drive URL does not contain the module name
    if filename == 'uc':
      filename = module + '.gz'
    self._add_download(url, os.path.join('modules', filename))
    self.add_step(self.shell(
      name="extracting files from %s" %filename,
      command=[
       sys.executable,"-c","import sys; sys.path.append('..'); import bootstrap; \
       bootstrap.tar_extract('','%s')" %filename],
      workdir=['modules'],
      description="extracting files from %s" %filename,
    ))

  def _add_unzip(self, archive, directory, trim_directory=0):
    class _indirection(object):
      def run(self):
        print("===== Installing %s into %s" % (archive, directory))
        Toolbox().unzip(archive, directory, trim_directory)
    self.add_step(_indirection())

  def _add_svn(self, module, url):
    update_list = ['update']
    if module in ["reduce", "probe", "king"]:
      pass
    elif self.revert:
      update_list = ['update', '-r', self.revert]
    thisworkdir = 'modules'
    if module == 'molprobity' : thisworkdir = '.'
    # avoid stalling bootstrap with prompts
    # or when encountering unknown server certificates
    svnflags = ['--non-interactive', '--trust-server-cert']
    if module == 'chem_data':  # stop pulling geostd from SourceForge
      svnflags.append('--ignore-externals')
    if os.path.exists(self.opjoin(*[thisworkdir, module, '.svn'])):
      self.add_step(self.shell(
          command=['svn'] + update_list +[module] + svnflags,
          workdir=[thisworkdir]
      ))
      self.add_step(self.shell(
          command=['svn', 'status', module] + svnflags,
          workdir=[thisworkdir],
          quiet=True,
      ))
    elif os.path.exists(self.opjoin(*[thisworkdir, module])):
      print("Existing non-svn directory -- don't know what to do. skipping: %s"%module)
    else:
      # print "fresh checkout..."
      self.add_step(self.shell(
          command=['svn', 'co', url, module] + svnflags,
          workdir=[thisworkdir]
      ))
    # replace geostd (replace this once chem_data is moved)
    if module == 'chem_data':
      if not os.path.exists(self.opjoin(thisworkdir, module)) \
        or os.path.exists(self.opjoin(thisworkdir, module, 'geostd', '.svn')):
          self.add_step(cleanup_dirs(['geostd'], self.opjoin(thisworkdir, module)))
      action = MODULES.get_module('geostd')().get_url(auth=self.get_auth())
      method, parameters = action[0], action[1:]
      self._add_git('geostd', parameters, destination=self.opjoin(thisworkdir, 'chem_data', 'geostd'))

  def _add_git(self, module, parameters, destination=None):
    use_git_ssh = self.auth.get('git_ssh', False)
    reference_repository_path = self.auth.get('git_reference', None)
    if reference_repository_path is None:
      if os.name == 'posix' and 'diamond.ac.uk' in pysocket.gethostname():
        reference_repository_path = '/dls/science/groups/scisoft/DIALS/repositories/git-reference'
    if reference_repository_path:
      reference_repository_path = os.path.expanduser(os.path.join(reference_repository_path, module))
    class _indirection(object):
      def run(self):
        Toolbox().git(module, parameters, destination=destination,
            use_ssh=use_git_ssh, verbose=True, reference=reference_repository_path)
    self.add_step(_indirection())

    # Update version information
    if module == 'cctbx_project':
      workdir = ['modules', module]
      self.add_step(self.shell(command=[sys.executable, os.path.join('libtbx', 'version.py')], workdir=workdir))

    # add geostd to chem_data
    if module == 'chem_data':
      action = MODULES.get_module('geostd')().get_url(auth=self.get_auth())
      method, geostd_parameters = action[0], action[1:]
      self._add_git('geostd', geostd_parameters, destination=self.opjoin('modules', 'chem_data', 'geostd'))

    # Use dials-2.2 branches for Python 2
    if (module == 'dials' or module == 'dxtbx' or module == 'xia2') and not self.python3:
      workdir = ['modules', module]
      if module == 'dxtbx':
        self.add_step(self.shell(command=['git', 'remote', 'set-url', 'origin', 'https://github.com/dials/dxtbx.git'], workdir=workdir))
        self.add_step(self.shell(command=['git', 'fetch', 'origin'], workdir=workdir))
      self.add_step(self.shell(command=['git', 'checkout', 'dials-2.2'], workdir=workdir))
      self.add_step(self.shell(
        command=['git', 'branch', '--set-upstream-to=origin/dials-2.2', 'dials-2.2'],
        workdir=workdir))

  def _check_for_Windows_prerequisites(self):
    if self.isPlatformWindows():
      # platform specific checks cannot run on buildbot master so add to build steps to run on slaves
      self.add_step(self.shell(command=[
         sys.executable,"-c","import sys; sys.path.append('..'); import bootstrap; \
          bootstrap.CheckWindowsPrerequisites()"],
        workdir=['modules'],
        description="Checking Windows prerequisites",
      ))

  def _get_conda_manager(self):
    """
    Helper function for determining the location of the conda environment
    """
    if __package__ is None:
      root_path = os.path.dirname(os.path.abspath(__file__))
      paths = [root_path,
               os.path.join(root_path, 'modules', 'cctbx_project', 'libtbx',
                            'auto_build')]
      for path in paths:
        if os.path.isfile(os.path.join(path, 'install_conda.py')):
          if path not in sys.path:
            sys.path.append(path)
            break
      from install_conda import conda_manager
    else:
      from .install_conda import conda_manager

    # drop output
    log = open(os.devnull, 'w')

    # environment is provided, so do check that it exists
    if self.use_conda is not None and os.path.isdir(self.use_conda):
      check_file = True
      self.use_conda = os.path.abspath(self.use_conda)
    # no path provided or file provided
    else:
      check_file = False
      # base step has not run yet, so do not check if files exist
      self.use_conda = os.path.join('..', 'conda_base')
      if self.isPlatformWindows():
        self.use_conda = os.path.join(os.getcwd(), 'conda_base')
    # basic checks for python and conda
    m = conda_manager(root_dir=os.getcwd(), conda_env=self.use_conda,
                      check_file=check_file, log=log)

    return m

  def _get_conda_python(self):
    """
    Helper function for determining the location of Python for the base
    and build actions.
    """
    try:
      m = self._get_conda_manager()
      return m.get_conda_python()
    except ImportError:  # modules directory is not available

      # -----------------------------------------------------------------------
      # duplicate logic from get_conda_python function in install_conda.py
      # since install_conda.py may not be available
      def m_get_conda_python(self):
        m_conda_python = os.path.join('bin', 'python')
        if self.isPlatformWindows():
          m_conda_python = self.op.join('python.exe')
        elif self.isPlatformMacOSX():
          m_conda_python = os.path.join('python.app', 'Contents',
                                        'MacOS', 'python')
        return m_conda_python
      # -----------------------------------------------------------------------

      conda_python = None

      # (case 1)
      # use default location or file provided to --use-conda
      if self.use_conda == '' or os.path.isfile(self.use_conda):
        conda_python = self.op.join('..', 'conda_base',
                                    m_get_conda_python(self))
        if self.isPlatformWindows():
          conda_python = self.op.join(os.getcwd(), 'conda_base', m_get_conda_python(self))
      # (case 2)
      # use path provided to --use-conda
      elif os.path.isdir(self.use_conda):
        self.use_conda = os.path.abspath(self.use_conda)
        conda_python = os.path.join(self.use_conda, m_get_conda_python(self))
      else:
        raise RuntimeError("""
The --use-conda flag can accept a directory to a conda environment or a
file that defines a conda environment. Please make sure a valid conda
environment exists in or is defined by {conda_env}.
""".format(conda_env=self.use_conda))

      if conda_python is None:
        raise RuntimeError('A conda version of python could not be found.')

    return conda_python

  def add_command(self, command, name=None, workdir=None, args=None, **kwargs):
    if self.isPlatformWindows():
      command = command + '.bat'
    # Relative path to workdir.
    workdir = workdir or [_BUILD_DIR]
    dots = [".."]*len(workdir)
    if workdir[0] == '.':
      dots = []
    if sys.platform == "win32": # assuming we run standalone without buildbot
      dots.extend([os.getcwd(), _BUILD_DIR, 'bin', command])
    else:
      dots.extend([_BUILD_DIR, 'bin', command])
    self.add_step(self.shell(
      name=name or command,
      command=[self.opjoin(*dots)] + (args or []),
      workdir=workdir,
      **kwargs
    ))

  def add_test_command(self,
                       command,
                       name=None,
                       workdir=None,
                       args=None,
                       haltOnFailure=False,
                       **kwargs
                       ):
    if name is None: name='test %s'%command
    self.add_command(
      command,
      name=name,
      workdir=(workdir or ['tests', command]),
      args=args,
      haltOnFailure=haltOnFailure,
      **kwargs
    )

  def add_test_parallel(self, module=None, nproc=None, slow_tests=False, **kwargs):
    if nproc is None:
      nprocstr = 'nproc=auto'
    else:
      nprocstr = 'nproc=%d'%nproc
    args=['module=%s'%module, nprocstr, 'verbosity=1']
    if slow_tests:
      args.append('slow_tests=True')
    self.add_command(
      'libtbx.run_tests_parallel',
      name='test %s'%module,
      workdir=['tests', module],
      args=args,
      haltOnFailure=False,
      **kwargs
    )

  def add_refresh(self):
    self.add_command(
      'libtbx.refresh',
      name='libtbx.refresh',
      workdir=['.'],
    )

  # Override these methods.
  def add_base(self, extra_opts=[]):
    """Build the base dependencies, e.g. Python, HDF5, etc."""
    if self.with_python:
      extra_opts = ['--with-python', self.with_python]
    if self.verbose:
      extra_opts.append('-v')
    if self.download_only:
      extra_opts.append('--download-only')
    if self.auth.get('git_ssh',False):
      extra_opts.append('--git-ssh')
    if self.skip_base:
      extra_opts.append('--skip-base=%s' % self.skip_base)
    if self.python3:
      extra_opts.append('--python3')
    if self.wxpython4:
      extra_opts.append('--wxpython4')
    if not self.force_base_build:
      if "--skip-if-exists" not in extra_opts:
        extra_opts.append("--skip-if-exists")
    command=[
      sys.executable,
      self.opjoin('modules', 'cctbx_project', 'libtbx', 'auto_build', 'install_base_packages.py'),
      '--python-shared',
      '--%s'%self.BASE_PACKAGES
    ] + extra_opts

    # Override base with conda
    #
    # The use of conda is focused on 2 main groups
    #   1) Developers who do not actively use conda
    #      A basic conda installation will be created at the same level as the
    #      "modules" and "build" directories. The default environment for the
    #      builder will be created in the "conda_base" directory at the same
    #      level.
    #   2) Developers who do
    #      A path to a conda environment should be provided. No checks are done
    #      on the environment. The environment files for the build should be
    #      used to construct the starting environment and the developer is
    #      responsible for maintaining it.
    if self.use_conda is not None:  # --use-conda flag is set
      # reset command
      command = []

      # file or no path provided (case 1), case 2 handled in _get_conda_python
      if self.use_conda == '' or os.path.isfile(self.use_conda):
        flags = ['--builder={builder}'.format(builder=self.category)]
        # check if a file was an argument
        if os.path.isfile(self.use_conda):
          filename = os.path.abspath(self.use_conda)
          flags.append('--install_env={filename}'.format(filename=filename))
        # check for existing miniconda3 installation
        if not os.path.isdir('mc3'):
          flags.append('--install_conda')
        flags.append('--python={python}'.format(python=self.python))
        command = [
          sys.executable,
          self.opjoin('modules', 'cctbx_project', 'libtbx', 'auto_build',
                      'install_conda.py',)] + flags

    if len(command) > 0:
      print("Installing base packages using:\n  " + " ".join(command))
      self.add_step(self.shell(name='base', command=command, workdir=['.']))

  def add_dispatchers(self, product_name="phenix"):
    """Write dispatcher_include file."""
    """Generating Phenix environment additions for dispatchers..."""
    envcmd = "export"
    dispatcher = os.path.join("build",
                              "dispatcher_include_%s.sh" %
                              product_name)
    if self.isPlatformWindows():
      envcmd = "set"
      dispatcher = os.path.join("build",
                                "dispatcher_include_%s.bat" %
                                product_name)
    if (os.path.isfile(dispatcher)): os.remove(dispatcher)
    env_prefix = product_name.upper() # e.g. "Phenix" -> "PHENIX"
    prologue = "\n".join([
      "%s %s=\"%s\"" % (envcmd, env_prefix, os.getcwd()),
      "%s %s_VERSION=%s" % (envcmd, env_prefix, "dev-svn"),
      "%s %s_ENVIRONMENT=1" % (envcmd, env_prefix),
      #"%s %s_MTYPE=%s" % (envcmd, env_prefix, "none"),
    ] #+ self.product_specific_dispatcher_prologue())
                           )
    #epilogue = "\n".join(self.product_specific_dispatcher_epilogue())
    dispatcher_opts = [
      "--build_dir=%s" % ".",
      "--suffix=%s"    % "phenix",
    ]
    if self.use_conda is None:
      dispatcher_opts += [
        "--base_dir=%s"  % "../base",
        "--gtk_version=2.10.0", # XXX this can change!
        #"--quiet",
      ]
    else:
      # default
      base_dir = self.op.join('..', 'conda_base')
      # use path from --use-conda flag
      # error-checking done in _get_conda_python function
      if os.path.isdir(self.use_conda):
        base_dir = self.use_conda

      dispatcher_opts += [
      "--base_dir=%s" % base_dir,
      "--use_conda",
      "--ignore_missing_dirs"
      ]
    #if (not self.flag_build_gui):
    #  dispatcher_opts.append("--ignore_missing_dirs")
    # FIXME this will happen regardless of whether the GUI modules are being
    # distributed or not - will this be problematic?
    self.add_step(self.shell(
      name='gui dispatcher',
      command=[
        self.python_base, #'python',
        self.opjoin("..",
                    'modules',
                    'cctbx_project',
                    'libtbx',
                    'auto_build',
                    'write_gui_dispatcher_include.py'),
        '--prologue=%s' % prologue,
        #"--epilogue=%s"
      ] + dispatcher_opts,
      workdir=[_BUILD_DIR]
    ))

  def add_configure(self):

    env = None

    if self.use_conda is not None:
      if '--use_conda' not in self.config_flags:
        self.config_flags.append('--use_conda')
      self.python_base = self._get_conda_python()
      # conda python prefers no environment customizations
      # the get_environment function in ShellCommand updates the environment
      if os.environ.get('CCTBX_CONDA_USE_ENVIRONMENT_VARIABLES', None):
        env = {
          'PYTHONPATH': None,
          'LD_LIBRARY_PATH': None,
          'DYLD_LIBRARY_PATH': None,
          'DYLD_FALLBACK_LIBRARY_PATH': None
        }

    configcmd =[
        self.python_base, # default to using our python rather than system python
        self.opjoin('..', 'modules', 'cctbx_project', 'libtbx', 'configure.py')
        ] + self.get_libtbx_configure() + self.config_flags
    self.add_step(self.shell(command=configcmd, workdir=[_BUILD_DIR],
      description="run configure.py", env=env))
    # Prepare saving configure.py command to file should user want to manually recompile Phenix
    fname = self.opjoin("config_modules.cmd")
    ldlibpath = ''
    if self.isPlatformLinux() and self.use_conda is None:
      ldlibpath = 'export LD_LIBRARY_PATH=../base/lib\n'
      # because that was the environment when python and base components were built during bootstrap
    confstr = ldlibpath + subprocess.list2cmdline(configcmd)
    if not self.isPlatformWindows():
      fname = self.opjoin("config_modules.sh")
      confstr = '#!/bin/sh\n\n' + confstr
    # klonky way of writing file later on, but it works
    self.add_step(self.shell(command=[
         sys.executable,'-c','open(r\"%s\",\"w\").write(r\"\"\"%s\"\"\" + \"\\n\")' %(fname, confstr)
         ],
      workdir=[_BUILD_DIR],
      description="save configure command",
    ))
    if not self.isPlatformWindows():
      self.add_step(self.shell(command=[ 'chmod', '+x', fname ],
        workdir=[_BUILD_DIR],
        description="permit execution of config_modules.sh",
      ))

    # write extra setpaths script for conda
    if self.use_conda is not None:
      self.add_command('libtbx.install_conda', args=['--write_setpaths'],
                       description='Writing additional setup scripts for conda.')

  def add_make(self):
    self.add_command('libtbx.scons', args=['-j',
                                           str(self.nproc),
#                                          #"--skip-version", # for Phaser
                                           ])
    # run build again to make sure everything is built
    self.add_command('libtbx.scons', args=['-j',
                                           str(self.nproc),
#                                          #"--skip-version", # for Phaser
                                           ])

  def add_install(self):
    """Run after compile, before tests."""
    if os.getenv('CCTBX_SKIP_CHEMDATA_CACHE_REBUILD') is None:
      self.add_command('mmtbx.rebuild_rotarama_cache',
                      name="rebuild rotarama",
      )
      self.add_command('mmtbx.rebuild_cablam_cache',
                      name="rebuild cablam",
      )

  def add_tests(self):
    """Run the unit tests."""
    pass

  def rebuild_docs(self):
    self.add_command('phenix_html.rebuild_docs')

  def add_distribute(self):
    pass

##### Specific Configurations ######

class CCIBuilder(Builder):
  """Base class for packages that include CCTBX as a dependency."""
  # Base packages
  BASE_PACKAGES = 'all'
  # Checkout these codebases
  CODEBASES = [
    'boost',
    'cbflib',
    'cctbx_project',
    'dxtbx',
    'gui_resources',
    'ccp4io',
    'ccp4io_adaptbx',
    'annlib',
    'annlib_adaptbx',
    'tntbx',
    'clipper',
    'eigen',
    'reduce',
    'ksdssp',
  ]
  CODEBASES_EXTRA = []
  # Copy these sources from cci.lbl.gov
  HOT = [
    'scons',
  ]
  HOT_EXTRA = []
  # Configure for these cctbx packages
  LIBTBX = [
    'cctbx',
    'cctbx_website',
    'cbflib',
    'dxtbx',
    'scitbx',
    'crys3d',
    'libtbx',
    'iotbx',
    'mmtbx',
    'smtbx',
    'gltbx',
    'wxtbx',
    'ksdssp',
  ]
  LIBTBX_EXTRA = []

##### CCTBX-derived packages #####

class MOLPROBITYBuilder(Builder):
  BASE_PACKAGES = 'molprobity'
  # Checkout these codebases
  CODEBASES = [
    'boost',
    'cbflib',
    'cctbx_project',
    'ccp4io',
    'ccp4io_adaptbx',
    'annlib',
    'annlib_adaptbx',
    'tntbx',
  ]
  CODEBASES_EXTRA = [
    'molprobity',
    'chem_data',
    'reduce',
    'probe'
  ]
  # Copy these sources from cci.lbl.gov
  HOT = [
    'scons',
    #"libsvm",
  ]
  HOT_EXTRA = []
  # Configure for these cctbx packages
  LIBTBX = [
    'mmtbx',
  ]
  LIBTBX_EXTRA = [
  ]

  def add_tests(self):
    pass

# def add_base(self, extra_opts=[]):
#   super(MOLPROBITYBuilder, self).add_base(
#     extra_opts=['--molprobity',
#                ] + extra_opts)

  def add_dispatchers(self):
    pass

  def rebuild_docs(self):
    pass

class PhaserBuilder(CCIBuilder):
  BASE_PACKAGES = 'cctbx'
    # Checkout these codebases
  CODEBASES = [
    'boost',
    'cctbx_project',
    'ccp4io',
    'ccp4io_adaptbx',
    'annlib',
    'annlib_adaptbx',
    'eigen',
    'tntbx',
    'phaser_regression',
    'phaser',
    'reduce',
  ]
  # Configure for these cctbx packages
  LIBTBX = [
    'cctbx',
    'scitbx',
    'crys3d',
    'libtbx',
    'iotbx',
    'mmtbx',
    'smtbx',
    'phaser_regression',
    'phaser',
  ]

  def add_tests(self):
    self.add_test_parallel(module='phaser_regression') # run phaser_regression/run_tests.py file
    """
    self.add_test_command('phaser_regression.regression', # run Gabors tests
                          args=['all',
                                '-o',
                                'terse_failed',
                                '-n',
                                '%s' %self.nproc,
                                ],
    )
    """

  def add_base(self, extra_opts=[]):
    # skip unnecessary base packages when building phaser only
    if self.skip_base is None or len(self.skip_base) == 0:
      self.skip_base = "hdf5,lz4_plugin,py2app,wxpython,docutils,pyopengl,pillow,tiff," + \
        "cairo,fonts,render,fontconfig,pixman,png,sphinx,freetype,gtk,matplotlib," + \
        "cython,h5py,gettext,numpy,pythonextra,pytest,junitxml,libsvm,pyrtf,six,send2trash," + \
         "jinja2,ordered_set,tabulate,scipy,scikit_learn,biopython,expat,glib,mrcfile"
    else:
      self.skip_base = ','.join(self.skip_base.split(',') + ['hdf5','lz4_plugin','py2app',
         'wxpython','docutils','pyopengl','pillow','tiff','cairo','fonts','pyrtf','six','send2trash',
         'fontconfig','render','pixman','png','sphinx','freetype','gtk', 'matplotlib',
         'cython', 'h5py', 'gettext', 'numpy', 'pythonextra', 'pytest', 'junitxml','libsvm',
         'jinja2', 'ordered_set', 'tabulate', 'scipy', 'scikit_learn', 'biopython',
         'expat', 'glib', 'mrcfile'
         ])
    super(PhaserBuilder, self).add_base(
      extra_opts=['--cctbx',
                 ] + extra_opts)

  def add_dispatchers(self):
    pass

  def rebuild_docs(self):
    pass

  def get_libtbx_configure(self):
    configlst = super(PhaserBuilder, self).get_libtbx_configure()
    if not self.isPlatformMacOSX():
      configlst.append("--enable_openmp_if_possible=True")
    return configlst

class PhaserTNGBuilder(PhaserBuilder):
  CODEBASES = PhaserBuilder.CODEBASES + ['phasertng', 'phaser_voyager', 'voyager_regression']
  LIBTBX = PhaserBuilder.LIBTBX + ['phasertng', 'phaser_voyager', 'voyager_regression']

  def add_tests(self):
    self.add_test_parallel(module='voyager_regression') # run voyager_regression/run_tests.py file
    self.add_test_parallel(module='phaser_regression') # run phaser_regression/run_tests.py file

  def get_libtbx_configure(self):
    configlst = super(PhaserTNGBuilder, self).get_libtbx_configure()
    if '--enable_cxx11' in configlst:
      configlst.remove('--enable_cxx11')
    set_std = ['cxxstd' in conf for conf in configlst]
    if set_std.count(True) == 0:
      if platform.mac_ver()[-1] == 'arm64':
        configlst.append('--cxxstd=c++14')
      else:
        configlst.append('--cxxstd=c++11')
    if not self.isPlatformMacOSX():
      configlst.append("--enable_openmp_if_possible=True")
    return configlst

  def get_codebases(self):
    """
    Phaser uses Boost in the conda environment for Python 3 and Windows
    """
    codebases = super(PhaserTNGBuilder, self).get_codebases()
    if (self.python3 and self.use_conda is not None):
      try:
        codebases.remove('boost')
      except ValueError:
        pass
    return codebases

class CCTBXLiteBuilder(CCIBuilder):
  BASE_PACKAGES = 'cctbx'
    # Checkout these codebases
  CODEBASES = [
    'boost',
    'cctbx_project',
    'gui_resources',
    'ccp4io',
    'ccp4io_adaptbx',
    'annlib',
    'annlib_adaptbx',
    'tntbx',
    'clipper',
    'eigen'
  ]
  # Configure for these cctbx packages
  LIBTBX = [
    'cctbx',
    'cctbx_website',
    'scitbx',
    'serialtbx',
    'libtbx',
    'iotbx',
    'mmtbx',
    'smtbx',
    'gltbx',
    'wxtbx',
  ]

  def add_tests(self):
    self.add_test_command('libtbx.import_all_python', workdir=['modules', 'cctbx_project'])
    self.add_test_command('cctbx_regression.test_nightly')

  def add_base(self, extra_opts=[]):
    if self.skip_base is None or len(self.skip_base) == 0:
      self.skip_base = "hdf5,lz4_plugin,h5py"
    else:
      self.skip_base = ','.join(self.skip_base.split(',') + ['hdf5','lz4_plugin','h5py'])
    super(CCTBXLiteBuilder, self).add_base(
      extra_opts=['--cctbx',
                 ] + extra_opts)

  def add_dispatchers(self):
    pass

  def rebuild_docs(self):
    pass

class CCTBXBuilder(CCIBuilder):
  BASE_PACKAGES = 'cctbx'
  def add_tests(self):
    self.add_test_command('libtbx.import_all_python', workdir=['modules', 'cctbx_project'])
    self.add_test_command('cctbx_regression.test_nightly')

  def add_base(self, extra_opts=[]):
    super(CCTBXBuilder, self).add_base(
      extra_opts=['--cctbx',
                 ] + extra_opts)

  def add_dispatchers(self):
    pass

  def rebuild_docs(self):
    pass

  def _add_git(self, module, parameters, destination=None):
    super(CCTBXBuilder, self)._add_git(module, parameters, destination)
    # select dials-3.5 branch
    if (module == 'dials' or module == 'dxtbx' or module == 'xia2') and self.python3:
      workdir = ['modules', module]
      if module == 'dxtbx':
        self.add_step(self.shell(command=['git', 'remote', 'set-url', 'origin', 'https://github.com/dials/dxtbx.git'], workdir=workdir))
        self.add_step(self.shell(command=['git', 'fetch', 'origin'], workdir=workdir))
      self.add_step(self.shell(command=['git', 'checkout', 'dials-3.5'], workdir=workdir))
      self.add_step(self.shell(
        command=['git', 'branch', '--set-upstream-to=origin/dials-3.5', 'dials-3.5'],
        workdir=workdir))
    # switch eigen to 3.3.9 for CentOS 6
    if module == 'eigen':
      if sys.platform.startswith('linux') and '.el6.' in platform.platform():
        workdir = ['modules', module]
        self.add_step(self.shell(command=['git', 'checkout', '3.3.9'], workdir=workdir))

class DIALSBuilder(CCIBuilder):
  CODEBASES_EXTRA = ['dials', 'iota', 'xia2', 'kokkos', 'kokkos-kernels']
  LIBTBX_EXTRA = ['dials', 'xia2', 'prime', 'iota', '--skip_phenix_dispatchers']
  HOT_EXTRA = ['msgpack']
  def add_tests(self):
    self.add_test_command('libtbx.pytest',
                          args=['--regression', '-n', 'auto'],
                          workdir=['modules', 'dxtbx'],
                          haltOnFailure=True)
    self.add_test_command('libtbx.pytest',
                          args=['--regression', '-n', 'auto'],
                          workdir=['modules', 'dials'],
                          haltOnFailure=True)

  def add_base(self, extra_opts=[]):
    super(DIALSBuilder, self).add_base(
      extra_opts=['--dials', '--xia2',
                 ] + extra_opts)

  def add_dispatchers(self):
    pass

  def rebuild_docs(self):
    pass

  def get_libtbx_configure(self):
    configlst = super(DIALSBuilder, self).get_libtbx_configure()
    if self.python != "27":
      # Do not enable C++11 for Python 2.7 builds, cf. https://github.com/cctbx/cctbx_project/pull/497
      configlst.append('--enable_cxx11')
    return configlst

class LABELITBuilder(CCIBuilder):
  CODEBASES_EXTRA = ['labelit', 'dials']
  LIBTBX_EXTRA = ['labelit', 'dials']

  def add_base(self, extra_opts=[]):
    super(LABELITBuilder, self).add_base(
      extra_opts=['--labelit', 'dials'] + extra_opts)

  def add_tests(self):
    self.add_test_parallel('labelit', flunkOnFailure=False, warnOnFailure=True)

  def add_dispatchers(self):
    pass

  def rebuild_docs(self):
    pass

class XFELLegacyBuilder(CCIBuilder):
  CODEBASES_EXTRA = [
    'dials',
    'iota',
    'labelit',
    'cxi_xdr_xes'
  ]
  LIBTBX_EXTRA = [
    'dials',
    'labelit',
    'xfel',
    'cxi_xdr_xes',
    'prime',
    'iota'
  ]
  HOT_EXTRA = ['msgpack']

  def add_base(self, extra_opts=[]):
    super(XFELLegacyBuilder, self).add_base(
      extra_opts=['--labelit', '--dials'] + extra_opts)

  def add_tests(self):
    self.add_test_command('cctbx_regression.test_nightly')

  def add_dispatchers(self):
    pass

  def rebuild_docs(self):
    pass

class XFELBuilder(CCIBuilder):
  CODEBASES_EXTRA = [
    'dials',
    'iota',
    'uc_metrics',
    'ncdist',
    'kokkos',
    'kokkos-kernels',
  ]
  LIBTBX_EXTRA = [
    'dials',
    'xfel',
    'prime',
    'iota',
    'uc_metrics',
  ]
  HOT_EXTRA = ['msgpack']

  def add_base(self, extra_opts=[]):
    super(XFELBuilder, self).add_base(
      extra_opts=['--dials'] + extra_opts)

  def get_libtbx_configure(self):
    configlst = super(XFELBuilder, self).get_libtbx_configure()
    if '--enable_cxx11' in configlst: configlst.remove('--enable_cxx11')
    configlst.append('--cxxstd=c++14')
    return configlst

  def add_tests(self):
    self.add_test_command('cctbx_regression.test_nightly')
    self.add_test_parallel(module='uc_metrics')

  def add_dispatchers(self):
    pass

  def rebuild_docs(self):
    pass

class PhenixBuilder(CCIBuilder):
  HOT = []
  CODEBASES_EXTRA = [
    'chem_data',
    'phenix',
    'phenix_dev_doc',
    'phenix_regression',
    'phenix_html',
    'phenix_examples',
    'phenix_pathwalker',
    'Colabs',
    'labelit',
    'Plex',
    'PyQuante',
    'elbow',
    'amber_adaptbx',
    'amber_library',
    'pulchra',
    'qrefine',
    'solve_resolve',
    'reel',
    'gui_resources',
    'opt_resources',
    'muscle',
    'reduce',
    'probe',
    'king',
    'phaser',
    'phasertng',
    'phaser_regression',
    'voyager_regression',
    'phaser_voyager',
    # 'dials',
    # 'xia2',
    # 'iota',
  ]
  LIBTBX_EXTRA = [
    'chem_data',
    'phenix',
    'phenix_dev_doc',
    'phenix_regression',
    'phenix_examples',
    'phenix_pathwalker',
    'Colabs',
    'qrefine',
    'solve_resolve',
    'reel',
    'phaser',
    'phasertng',
    'phaser_regression',
    'voyager_regression',
    'phaser_voyager',
    'labelit',
    'elbow',
    'amber_adaptbx',
    'reduce',
    'probe',
    'cootbx',
    'qttbx',
    # 'dials',
    # 'xia2',
    # 'prime',
  ]

  # select dials-3.8 branch
  def _add_git(self, module, parameters, destination=None):
    super(PhenixBuilder, self)._add_git(module, parameters, destination)
    if module == 'boost':
      workdir = ['modules', module]
      if self.category == 'phenix_discamb':
        self.add_step(self.shell(command=['git', 'checkout', '1.86'], workdir=workdir))
      else:
        self.add_step(self.shell(command=['git', 'checkout', '1.74'], workdir=workdir))
    elif (module == 'dials' or module == 'dxtbx' or module == 'xia2') and self.python3:
      workdir = ['modules', module]
      if module == 'dxtbx':
        self.add_step(self.shell(command=['git', 'remote', 'set-url', 'origin', 'https://github.com/dials/dxtbx.git'], workdir=workdir))
        self.add_step(self.shell(command=['git', 'fetch', 'origin'], workdir=workdir))
      self.add_step(self.shell(command=['git', 'checkout', 'dials-3.8'], workdir=workdir))
      self.add_step(self.shell(
        command=['git', 'branch', '--set-upstream-to=origin/dials-3.8', 'dials-3.8'],
        workdir=workdir))

  def add_module(self, module, workdir=None, module_directory=None):
    """
    Add git-lfs command for phenix_examples and phenix_regression
    If the dev_env directory already exists, it is assumed that git-lfs
    is available in that directory
    """
    super(PhenixBuilder, self).add_module(module, workdir, module_directory)

    # update phenix_regression and phenix_examples with git-lfs
    if module == 'phenix_examples' or module == 'phenix_regression' or module == 'chem_data':
      # prepend path for check
      dev_env = os.path.join('.', 'dev_env', 'bin')
      if sys.platform == 'win32':
        dev_env = os.path.join('.', 'dev_env', 'Library', 'bin')
        os.environ['PATH'] = os.path.abspath(dev_env) + ';'  + os.environ['PATH']
      else:
        os.environ['PATH'] = os.path.abspath(dev_env) + ':'  + os.environ['PATH']

      svn_is_available = False
      git_lfs_is_available = False

      # check if git-lfs and svn are available
      log = open(os.devnull, 'w')

      try:
        returncode = subprocess.call(['svn', '--version'], stdout=log, stderr=log)
        if returncode == 0:
          svn_is_available = True
      except Exception:
        pass

      try:
        returncode = subprocess.call(['git', 'lfs', '--version'], stdout=log, stderr=log)
        if returncode == 0:
          git_lfs_is_available = True
      except Exception:
        pass

      log.close()

      # set if dev_env will be created in base step
      self.install_dev_env = False
      if not svn_is_available or not git_lfs_is_available:
        self.install_dev_env = True

      # get lfs files
      if self.install_dev_env:
        print('*'*79)
        print("""\
An environment containing git-lfs and/or svn will be installed during the "base"
step. Pleaser re-run the "update" step after "base" completes, so that git-lfs
files for {module} will be downloaded.""".format(module=module))
        print('*'*79)
      else:
        workdir = ['modules', module]
        self.add_step(self.shell(command=['git', 'lfs', 'install', '--local'], workdir=workdir))
        self.add_step(self.shell(command=['git', 'lfs', 'pull'], workdir=workdir))

  def add_base(self, extra_opts=[]):
    super(PhenixBuilder, self).add_base(
      extra_opts=['--phenix',
                  '--labelit',
                  '--dials',
                  '--xia2',
                 ] + extra_opts)

    # install extra development environment if necessary
    if hasattr(self, 'install_dev_env') and self.install_dev_env:
      if self.use_conda is None:
        raise RuntimeError("""
Conda is needed for creating the extra environment with git-lfs. Please add
"--use-conda" to your bootstrap.py command or make sure git-lfs is available
in your path. """)
      self.python_base = self._get_conda_python()
      if self.python_base.startswith('../conda_base'):
        self.python_base = self.python_base[1:]  # keep current directory
      env = {
        'PYTHONPATH': None,
        'LD_LIBRARY_PATH': None,
        'DYLD_LIBRARY_PATH': None
      }
      command = [self.python_base,
                 os.path.join('modules', 'cctbx_project', 'libtbx',
                              'auto_build', 'install_conda.py'),
                 '--install_dev_env', '--verbose']
      self.add_step(self.shell(command=command, workdir=['.']))

  def add_install(self):
    Builder.add_install(self)

  def get_libtbx_configure(self):
    configlst = super(PhenixBuilder, self).get_libtbx_configure()
    if '--enable_cxx11' in configlst:
      configlst.remove('--enable_cxx11')
    set_std = ['cxxstd' in conf for conf in configlst]
    if set_std.count(True) == 0:
      if platform.mac_ver()[-1] == 'arm64':
        configlst.append('--cxxstd=c++14')
      else:
        configlst.append('--cxxstd=c++11')
    if not self.isPlatformMacOSX():
      configlst.append("--enable_openmp_if_possible=True")
    return configlst

  def rebuild_docs(self):
    self.add_command('phenix_html.rebuild_docs')

  def add_tests(self):
    # Include cctbx tests.
    self.add_test_command('libtbx.import_all_ext')
    self.add_test_command('cctbx_regression.test_nightly')
    # Windows convenience hack.
    if self.isPlatformWindows():
      self.add_test_command('phenix_regression.test_nightly_windows')
    else:
      self.add_test_command('phenix_regression.test_nightly')
    # Other Phenix tests.
    self.add_test_parallel(module='elbow')
    self.rebuild_docs()
    self.add_test_command('phenix_regression.run_p9_sad_benchmark',
                          name="test p9 sad",
                         )
    self.add_test_command('phenix_regression.run_hipip_refine_benchmark',
                          name="test hipip",
                         )
    self.add_test_command('phenix_regression.wizards.test_all_parallel',
      args = ['nproc=8'],
      name="test wizards",
                         )
    run_dials_tests=True
    if self.isPlatformWindows():
      if 'dials' in windows_remove_list:
        run_dials_tests=False
    if run_dials_tests:
      self.add_test_parallel('dials', flunkOnFailure=False, warnOnFailure=True)

class PhenixDiscambBuilder(PhenixBuilder):
  CODEBASES_EXTRA = PhenixBuilder.CODEBASES_EXTRA + ['pyDiSCaMB']

  def get_libtbx_configure(self):
    configlst = super(PhenixDiscambBuilder, self).get_libtbx_configure()
    # switch to C++14 for new environments
    if '--cxxstd=c++14' not in configlst:
      configlst.append('--cxxstd=c++14')
    return configlst

  def add_make(self):
    super(PhenixDiscambBuilder, self).add_make()
    # install pyDiSCaMB
    python = os.path.normpath(os.path.join(os.getcwd(), 'build', self.python_base))
    self.add_step(self.shell(
      command=[python, '-m', 'pip', 'install', '.'],
        workdir=['modules', 'pyDiSCaMB'],
        description='pip installing pyDiSCaMB',
      ))

class PhenixExternalRegression(PhenixBuilder):
  EXTERNAL_CODEBASES = [
    "afitt",
    "rosetta",
    ]

  def cleanup(self, dirs=None):
    self.add_step(cleanup_ext_class(".bz2", "modules", walk=False))
    lt = time.localtime()
    cleaning = ['dist', 'tests', 'doc', 'tmp', 'base_tmp']
    if lt.tm_wday==5: # do a completer build on Saturday night
      cleaning += ['base', 'base_tmp', _BUILD_DIR, 'conda_base']
    # Preparation
    # AFITT
    if self.subcategory in [None, "afitt"]:
      self.add_step(cleanup_dirs(['openeye'], 'modules'))
    PhenixBuilder.cleanup(self, cleaning)

  def get_environment(self, add_build_python_to_path=True):
    environment = {}
    for env, dirs in envs.items():
      environment[env] = os.path.join(*dirs)
    if add_build_python_to_path:
      old_path = os.environ.get("PATH", "") # this is just another now
                                            # universal hack to get Amber to
                                            # compile...
      environment["PATH"] = '%s:%s' % (os.path.join(#os.getcwd(),
                                                    "build",
                                                    "bin",
                                                  ),
                                       old_path,
                                       )
    return environment

  def write_environment(self,
                        env,
                        filename="setpaths_externals",
                       ):
    # called by add_make which is called in build
    # this is a little funky as it seems to be very often in the wrong remote dir
    outl = ""
    for key, path in env.items():
      if key in ["PATH"]: continue
      outl += 'setenv %(key)s "%%(PWD)s/../%(path)s"\n' % locals()
    fname="%s.csh" % filename
    self.add_step(self.shell(command=[
      sys.executable,
      '-c',
      'import os; open("%s","w").write("""%s""" %% os.environ)' %(fname, outl)
      ],
      workdir=[_BUILD_DIR],
      description="save csh external paths",
    ))
    outl = ""
    for key, path in env.items():
      if key in ["PATH"]: continue
      outl += 'export %(key)s="%%(PWD)s/../%(path)s"\n' % locals()
    fname="%s.sh" % filename
    self.add_step(self.shell(command=[
      sys.executable,
      '-c',
      'import os; open("%s","w").write("""%s""" %% os.environ)' %(fname, outl)
      ],
      workdir=[_BUILD_DIR],
      description="save sh external paths",
    ))

  def add_make(self):
    # Phenix compile
    PhenixBuilder.add_make(self)
    # need to use the Phenix python for building
    # Rosetta
    # AFITT
    env = self.get_environment()
    self.write_environment(env)
    # not universal but works because only slave running this is same as master
    for name, command, workdir in [
        ['AFITT - untar',
         ['tar', 'xvf', '%s.gz' % afitt_version],
         ['modules']],
        ['Rosetta - untar',
         ['tar', 'xvf', '%s.tgz' % rosetta_version_tar_bundle],
         ['modules']],
        ['Rosetta - rm link',
         # not windows compatible
         ['rm', '-f', "rosetta"],
         ['modules']],
        ['Rosetta - link',
         # not windows compatible
         ['ln', '-sf', '%s' % rosetta_version_directory, "rosetta"],
         ['modules']],
        #['Rosetta compile', # not really needed
        # ["./scons.py",
        #  "-j",
        #  self.nproc,
        #  #"mode=release",
        # ],
        # ["modules", 'rosetta', "main", "source"]],
        ]:
      if self.subcategory:
        if name.lower().find(self.subcategory)==-1: continue
      haltOnFailure=True
      self.add_step(self.shell(
        name       = name,
        command    = command,
        workdir    = workdir,
        description= "",
        env        = env,
        haltOnFailure = haltOnFailure,
        ))

    self.add_refresh()
    # Rosetta
    if self.subcategory in [None, "rosetta"]:
      self.add_command(
        'rosetta.build_phenix_interface',
        args = ["nproc=%s" % self.nproc],
        name='rosetta.build_phenix_interface',
        workdir=['.'],
        env=env,
      )

  def add_tests(self):
    # timings
    if self.subcategory in [None, 'timings']:
      self.add_test_command(
        'mmtbx.python',
        args=[os.path.join('..',
                           '..',
                           'modules',
                           'phenix_regression',
                           'development',
                           'runtime_speed_regression_test.py',
                           )],
        name="timings test",
        )
    # amber
    if self.subcategory in [None, "amber"]:
      self.add_test_command('amber.run_tests',
                            env = self.get_environment(),
                            haltOnFailure=False,
                           )
    # rosetta refine
    if self.subcategory in [None, "rosetta"]:
      self.add_test_command('rosetta.run_tests',
                            env = self.get_environment()
                           )
    # MR rosetta
    if self.subcategory in [None, "rosetta"]:
      self.add_test_command(
        'phenix_regression.wizards.run_wizard_test',
        args=['test_prerefine'],
        name="test MR rosetta quick",
        env = self.get_environment()
      )
    # afitt
    if self.subcategory in [None, "afitt"]:
      self.add_test_command('afitt.run_tests',
                            env = self.get_environment()
                           )
    # GLR
    if self.subcategory in [None, "glr"]:
      self.add_test_command('elbow.run_glr_tests',
                            haltOnFailure=False,
                            )

class QRBuilder(PhenixBuilder):
  #
  # Designed to be run in Phenix build to add Q|R
  # and the entire PhenixBuilder if user is builder
  #
  EXTERNAL_CODEBASES = ["qrefine"]
  user = os.environ.get('USER', None)

  def add_make(self):
    if self.user=='builder': PhenixBuilder.add_make(self)
    #
    # XXX Use older ASE (the new one is only Python3)
    # XXX Do not get JPype1 as it fails. This makes QR work only with
    # XXX fast_interaction=True (=False won't work)
    #
    pip_installs = ['ase==3.22.1',]
    instructions = []
    # versioning
    cmd = [os.path.join('..', self.python_base),
           os.path.join('utils', 'make_version.py'),
           ]
    instructions.append(['Versioning', cmd, ['modules/qrefine']])
    for pi in pip_installs:
      instructions.append(['Q|R pip %s' % pi,
                           [self.python_base,
                            '-m',
                            'pip',
                            'install',
                            pi
                          ],
                           ['modules']])

    for name, command, workdir in instructions:
      self.add_step(self.shell(
        name       = name,
        command    = command,
        workdir    = workdir,
        description= "",
        haltOnFailure = 1, #haltOnFailure,
        ))
    self.add_refresh()

  def get_hot(self):
    if self.user=='builder': return PhenixBuilder.get_hot(self)
    return [] # don't have any HOT downloads and the difference between
              # anonymous and cciuser is making a mess

  def get_libtbx_configure(self): # modified in derived class PhenixBuilder
    return self.LIBTBX + self.LIBTBX_EXTRA + self.EXTERNAL_CODEBASES

  def get_codebases(self):
    if self.isPlatformWindows(): assert 0, 'not supported'
    if self.user=='builder':
      rc = PhenixBuilder.get_codebases(self)
    else:
      rc = self.EXTERNAL_CODEBASES #+ ['cctbx_project']
    return rc

  def add_tests(self):
    self.add_test_command('qr.build_interfaces',
                          haltOnFailure=False,
                          env = self.get_environment()
                          )
    self.add_test_command('qr.test',
                          # args=['--non_mopac_only'],
                          haltOnFailure=True,
                          env = self.get_environment()
                          )

  def add_dispatchers(self):
    pass

  def rebuild_docs(self):
    pass

  def get_environment(self, add_build_python_to_path=True):
    environment = {}
    mopac_envs = {
      "MOPAC_LICENSE"  : "/home/builder/software/mopac",
      "MOPAC_COMMAND"  : "/home/builder/software/mopac/mopac.csh",
    }
    for env, dirs in mopac_envs.items():
      environment[env] = dirs
    return environment

class PhenixReleaseBuilder(PhenixBuilder):
  '''
  Phenix with DIALS
  '''
  extra_codebases = ['dials', 'iota', 'xia2']
  extra_libtbx = extra_codebases + ['prime']
  CODEBASES_EXTRA = PhenixBuilder.CODEBASES_EXTRA + extra_codebases
  LIBTBX_EXTRA = PhenixBuilder.LIBTBX_EXTRA + extra_libtbx

def set_builder_defaults(options):
  '''
  Updates defaults for specific builders
  '''
  if options.builder == 'phenix_voyager':
    if not options.no_boost_src:
      options.no_boost_src = True
      # restore default for CentOS 7
      if sys.platform.startswith('linux') and '.el7.' in platform.platform():
        options.no_boost_src = False
  if options.builder == 'phenix' \
    or options.builder == 'phenix_discamb' \
    or options.builder == 'phenix_voyager' \
    or options.builder == 'molprobity':
    # Apple Silicon uses Boost 1.78 in environment, Python 3.9
    if platform.mac_ver()[-1] == 'arm64':
      options.no_boost_src = True
      options.python = '39'
    if not options.no_boost_src:
      options.no_boost_src = True
      if sys.platform.startswith('linux') and '.el7.' in platform.platform():
        options.no_boost_src = False
    if options.use_conda is None:
      options.use_conda = ''
    if options.builder == 'phenix_discamb':
      options.python = '39'

  return options

def run(root=None):
  builders = {
    'cctbxlite': CCTBXLiteBuilder,
    'cctbx': CCTBXBuilder,
    'phenix': PhenixBuilder,
    'phenix_discamb': PhenixDiscambBuilder,
    'phenix_voyager': PhenixBuilder,
    'phenix_release': PhenixReleaseBuilder,
    'xfellegacy': XFELLegacyBuilder,
    'xfel': XFELBuilder,
    'labelit': LABELITBuilder,
    'dials': DIALSBuilder,
    'external': PhenixExternalRegression,
    'molprobity':MOLPROBITYBuilder,
    'qrefine': QRBuilder,
    'phaser': PhaserBuilder,
    'voyager': PhaserTNGBuilder
  }

  wrapper = textwrap.TextWrapper(width=80, initial_indent='  ',
                                 subsequent_indent='    ')
  builders_text = ', '.join(sorted(builders.keys()))
  builders_text = '\n'.join(wrapper.wrap(builders_text))

  prog = os.environ.get('LIBTBX_DISPATCHER_NAME')
  if prog is None or prog.startswith('python') or prog.endswith('python'):
    prog = os.path.basename(sys.argv[0])

  description = """
  You may specify one or more actions:
    hot - Update static sources (scons, etc.)
    update - Update source repositories (cctbx, cbflib, etc.)
    base - Build base dependencies (python, hdf5, wxWidgets, etc.)
    build - Build
    tests - Run tests
    doc - Build documentation

  The default action is to run: hot, update, base, build

  You can specify which package will be downloaded, configured,
  and built with "--builder". Current builders:
  {builders}

  You can provide your SourceForge username with "--sfuser", and
  your CCI SVN username with "--cciuser". These will checkout
  and update repositories with your credentials. Some builders,
  like phenix, require this argument for access to certain
  repositories.

  You can run the compilation step in parallel by providing a
  the number of processes using "--nproc".
  Complete build output is shown with "-v" or "--verbose".

  Finally, you may specify a specific Python interpreter
  using "--with-python".

  Example:

    python bootstrap.py --builder=cctbx --sfuser=metalheadd hot update build tests
  """.format(builders=builders_text)

  parser = argparse.ArgumentParser(
    prog=prog, description=description,
    formatter_class=argparse.RawDescriptionHelpFormatter)
  # parser.add_argument("--root", help="Root directory; this will contain base, modules, build, etc.")
  parser.add_argument('action', nargs='*', help="Actions for building")
  parser.add_argument(
    "--builder",
    help="Builder: " + ",".join(list(builders.keys())),
    default="cctbx")
  parser.add_argument("--cciuser", help="CCI SVN username.")
  parser.add_argument("--sfuser", help="SourceForge SVN username.")
  parser.add_argument("--revert", help="SVN string to revert all SVN trees")
  parser.add_argument("--sfmethod",
                    help="SourceForge SVN checkout method.",
                    default="svn+ssh")
  parser.add_argument(
    "--git-ssh",
    dest="git_ssh",
    action="store_true",
    help="Use ssh connections for git. This allows you to commit changes without changing remotes and use reference repositories.",
    default=False)
  parser.add_argument(
    "--git-reference",
    dest="git_reference",
    help="Path to a directory containing reference copies of repositories for faster checkouts.")
  parser.add_argument("--with-python",
                    dest="with_python",
                    help="Use specified Python interpreter")
  parser.add_argument("--nproc",
                    help="number of parallel processes in compile step.")
  parser.add_argument("--download-only",
                    dest="download_only",
                    action="store_true",
                    help="Do not build, only download prerequisites",
                    default=False)
  parser.add_argument("-v",
                    "--verbose",
                    dest="verbose",
                    action="store_true",
                    help="Verbose output",
                    default=False)
  parser.add_argument("--skip-base-packages",
                    dest="skip_base",
                    action="store",
                    default="")
  parser.add_argument("--force-base-build",
                    dest="force_base_build",
                    action="store_true",
                    default=False)
  parser.add_argument("--enable-shared",
                    dest="enable_shared",
                    action="store_true",
                    default=False)
  parser.add_argument("--mpi-build",
                    dest="mpi_build",
                    help="Builds software with mpi functionality",
                    action="store_true",
                    default=False)
  python_args = parser.add_mutually_exclusive_group(required=False)
  python_args.add_argument('--python',
                    default='37', type=str, nargs='?', const='37',
                    choices=['27', '37', '38', '39', '310'],
                    help="""When set, a specific Python version of the
conda environment will be used. This only affects environments selected with
the --builder flag. For non-conda dependencies, any Python 3 implies
Python 3.7.""")
  parser.add_argument("--config-flags", "--config_flags",
                    dest="config_flags",
                    help="""Pass flags to the configuration step. Flags should
be passed separately with quotes to avoid confusion (e.g
--config_flags="--build=debug" --config_flags="--enable_cxx11")""",
                    action="append",
                    default=[])
  parser.add_argument("--use-conda", "--use_conda", metavar="ENVIRONMENT",
                    dest="use_conda",
                    help="""Use conda for dependencies. The directory to an
existing conda environment or a file defining a conda environment can be
provided. The build will use that environment instead of creating a default one
for the builder. If the currently active conda environment is to be used for
building, $CONDA_PREFIX should be the argument for this flag. Otherwise, a new
environment will be created. The --python flag will be ignored when there is
an argument for this flag. Specifying an environment is for developers that
maintain their own conda environment.""",
                    default=None, nargs='?', const='')
  parser.add_argument("--no-boost-src", "--no_boost_src",
                      dest="no_boost_src",
                      help="""When set, the reduced Boost source code is not
downloaded into the modules directory. This enables the usage of an existing
installation of Boost in the same directory as the Python for configuration.
For example, this flag should be used if the conda package for Boost is
available. This flag only affects the "update" step.""",
                      action="store_true",
                      default=False)
  parser.add_argument("--build-dir",
                     dest="build_dir",
                     help="directory where the build will be. Should be at the same level as modules! default is 'build'",
                     default="build", type=str)

  options = parser.parse_args()
  args = options.action

  global _BUILD_DIR
  _BUILD_DIR = options.build_dir  # TODO: this is probably ok way to go with globalvar, but check and see

  # process external
  options.specific_external_builder=None
  if options.builder.lower() in ["afitt",
                                 "rosetta",
                                 ]:
    options.specific_external_builder=options.builder.lower()
    options.builder="external"

  # Root dir
  # options.root = options.root or root

  # Check actions
  allowedargs = ['cleanup', 'hot', 'update', 'base', 'build', 'tests', 'doc']
  args = args or ['hot', 'update', 'base', 'build']
  actions = []
  for arg in args:
    if arg not in allowedargs:
      raise ValueError("Unknown action: %s"%arg)
  for arg in allowedargs:
    if arg in args:
      actions.append(arg)

  # Check if an action was an argument to --use-conda
  if options.use_conda in allowedargs:
    if len(options.action) == 0:
      actions = [options.use_conda]
    else:
      actions.append(options.use_conda)
    options.use_conda = ''

  # Check if the argument to --use-conda starts with '~'
  if options.use_conda is not None and options.use_conda.startswith('~'):
    options.use_conda = os.path.expanduser(options.use_conda)

  print("Performing actions:", " ".join(actions))

  # Check builder
  if options.builder not in builders:
    raise ValueError("Unknown builder: %s"%options.builder)

  auth = { 'git_ssh': options.git_ssh, 'git_reference': options.git_reference }
  if options.cciuser:
    auth['cciuser'] = options.cciuser
  if options.sfuser:
    auth['sfuser'] = options.sfuser
  if options.sfmethod:
    auth['sfmethod'] = options.sfmethod

  # Apply defaults for specific builders
  options = set_builder_defaults(options)

  # Build
  builder = builders[options.builder]
  builder(
    category=options.builder,
    subcategory=options.specific_external_builder,
    platform='dev',
    with_python=options.with_python,
    auth=auth,
    hot=('hot' in actions),
    update=('update' in actions),
    revert=options.revert,
    base=('base' in actions),
    build=('build' in actions),
    tests=('tests' in actions),
    doc=('doc' in actions),
    cleanup=("cleanup" in actions),
    nproc=options.nproc,
    verbose=options.verbose,
    download_only=options.download_only,
    skip_base=options.skip_base,
    force_base_build=options.force_base_build,
    enable_shared=options.enable_shared,
    mpi_build=options.mpi_build,
    python3=False,
    wxpython4=False,
    config_flags=options.config_flags,
    use_conda=options.use_conda,
    python=options.python,
    no_boost_src=options.no_boost_src,
  ).run()
  print("\nBootstrap success: %s" % ", ".join(actions))

if __name__ == "__main__":
  run()
