
"""
Automated build of CCTBX dependencies for Linux and Mac platforms.
This script will download and install the current Python distribution provided
by LBL, plus any packages required for various CCTBX-dependent apps to
function.  In the future this will be used as the core of the CCI nightly build
system and the Phenix installer.
"""

from __future__ import absolute_import, division, print_function

import os
import os.path as op
import platform
import sys
import time
import zipfile
import subprocess
from optparse import OptionParser

if __name__ == '__main__' and __package__ is None:
  # Cannot use relative imports when run from bootstrap, so add this
  # directory to the path.
  sys.path.append(os.path.dirname(os.path.abspath(__file__)))
  from installer_utils import *
  from package_defs import *
else:
  from .installer_utils import *
  from .package_defs import *

python_dependencies = {"_ssl" : "Secure Socket Library",
                       "zlib" : "XCode command line tools",
                     }

# Turn off user site-packages directory to avoid conflicts
# https://www.python.org/dev/peps/pep-0370/
os.environ['PYTHONNOUSERSITE'] = '1'


class installer(object):
  def __init__(self, args=None, packages=None, log=sys.stdout):
    #assert (sys.platform in ["linux2", "linux3", "darwin"])
    # Check python version >= 2.7 or >= 3.4
    check_python_version()
    self.log = log
    print("""
  ****************************************************************************
                 Automated CCTBX dependencies build script
                 report problems to cctbx-dev@cci.lbl.gov
  ****************************************************************************
""", file=log)
    dist_dir = op.dirname(op.dirname(op.dirname(__file__)))
    parser = OptionParser()
    # Basic options
    parser.add_option("--build_dir", dest="build_dir", action="store",
      help="Build directory", default=os.getcwd())
    parser.add_option("--tmp_dir", dest="tmp_dir", action="store",
      help="Temporary directory",
      default=op.join(os.getcwd(), "base_tmp"))
    parser.add_option("--pkg_dir", dest="pkg_dirs", action="append",
      help="Directory with source packages", default=None)
    parser.add_option("-p", "--nproc", dest="nproc", action="store",
      type="int", help="Number of processors", default=1)
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true",
      help="Verbose output", default=False)
    parser.add_option("--skip-if-exists", action="store_true",
      help="Exit if build_dir/base exists; used with automated builds.")
    parser.add_option("--no-download", dest="no_download", action="store_true",
      help="Use only local packages (no downloads)", default=False)
    parser.add_option("--download-only", dest="download_only", action="store_true",
      help="Only download missing packages, do not compile", default=False)
    parser.add_option("--skip-base", dest="skip_base", action="store",
      help="Comma-separated list of packages to skip", default="")
    parser.add_option("--continue-from", dest="continue_from", action="store",
      help="Skip all packages preceeding this one", default="")
    parser.add_option("--python-shared", dest="python_shared",
      action="store_true", default=False,
      help="Compile Python as shared library (Linux only)")
    parser.add_option("--with-python", dest="with_python",
      help="Use specified Python interpreter")
    parser.add_option("--with-system-python", dest="with_system_python",
      help="Use the system Python interpreter", action="store_true")
    parser.add_option("--python3", dest="python3", action="store_true", default=False,
      help="Install a Python3 interpreter. This is unsupported and purely for development purposes.")
    parser.add_option("--wxpython4", dest="wxpython4", action="store_true", default=False,
      help="Install wxpython4 instead of wxpython3. This is unsupported and purely for development purposes.")
    parser.add_option("--mpi-build", dest="mpi_build", action="store_true", default=False,
      help="Installs software with MPI functionality")
    parser.add_option("-g", "--debug", dest="debug", action="store_true",
      help="Build in debugging mode", default=False)
    # Package set options.
    parser.add_option("--molprobity", dest="molprobity", action="store_true",
      help="Build mmtbx dependencies - like --cctbx without gui dependencies")
    parser.add_option("--cctbx", dest="cctbx", action="store_true",
      help="Build CCTBX dependencies")
    parser.add_option("--phenix", dest="phenix", action="store_true",
      help="Build PHENIX dependencies")
    parser.add_option("--labelit", dest="labelit", action="store_true",
      help="Build LABELIT dependencies")
    parser.add_option("--xia2", dest="xia2", action="store_true",
      help="Build xia2 dependencies")
    parser.add_option("--dials", dest="dials", action="store_true",
      help="Build DIALS dependencies")
    parser.add_option("--gui", dest="build_gui", action="store_true",
      help="Build GUI dependencies")
    parser.add_option("-a", "--all", dest="build_all", action="store_true",
      help="Build all recommended dependencies", default=False)
    # Specific add-on packages.
    parser.add_option("--scipy", dest="build_scipy", action="store_true",
      help="Build SciPy", default=False)
    parser.add_option("--ipython", dest="build_ipython", action="store_true",
      help="Build IPython", default=False)
    parser.add_option("--git-ssh", dest="git_ssh", action="store_true",
      help="Use ssh connections for git. This allows you to commit changes without changing remotes.", default=False)

    # Basic setup.
    options, args = parser.parse_args(args)
    self.nproc = options.nproc
    self.verbose = options.verbose
    self.options = options
    self.include_dirs = []
    self.lib_dirs = []
    self.flag_is_linux = sys.platform.startswith("linux")
    self.flag_is_mac = (sys.platform == "darwin")
    self.cppflags_start = os.environ.get("CPPFLAGS", "")
    self.ldflags_start = os.environ.get("LDFLAGS", "")

    # set default macOS flags
    if (self.flag_is_mac):
      self.min_macos_version = '10.7'
      self.min_macos_version_flag = '-mmacosx-version-min=%s' %\
                                    self.min_macos_version
      self.base_macos_flags = ' -stdlib=libc++ %s' % self.min_macos_version_flag
      self.cppflags_start += self.base_macos_flags
      self.ldflags_start += self.base_macos_flags

    # Compilation flags for CentOS 5 (32-bit)
    if ( (self.flag_is_linux) and (platform.architecture()[0] == '32bit') ):
      old_cflags = os.environ.get('CFLAGS', '')
      os.environ['CFLAGS'] = old_cflags + ' -march=i686'

    # Directory setup.
    self.tmp_dir = options.tmp_dir
    self.build_dir = options.build_dir
    self.pkg_dirs = options.pkg_dirs
    self.base_dir = op.join(self.build_dir, "base")
    self.prefix = "--prefix=\"%s\""%self.base_dir
    if options.skip_if_exists and os.path.exists(self.base_dir) and os.listdir(self.base_dir):
      print("Base directory already exists and --skip-if-exists set; exiting.", file=log)
      return
    print("Setting up directories...", file=log)
    for dir_name in [self.tmp_dir,self.build_dir,self.base_dir]:
      if (not op.isdir(dir_name)):
        print("  creating %s" % dir_name, file=log)
        os.makedirs(dir_name)
    self.check_python_dependencies()

    # Configure package download
    self.fetch_package = fetch_packages(
      dest_dir=self.tmp_dir,
      log=log,
      pkg_dirs=options.pkg_dirs,
      no_download=options.no_download)
    # Shortcut: Extract HDF5 and python for Windows bundled with all preinstalled modules
    if sys.platform == "win32":
      if platform.architecture()[0] == '64bit':
        winpythonpkg = WIN64PYTHON_PKG
        hdf5pkg = WIN64HDF5_PKG
        vcredist = VCREDIST64
        libtiff = WINLIBTIFF64
      else:
        winpythonpkg = WIN32PYTHON_PKG
        hdf5pkg = WIN32HDF5_PKG
        vcredist = VCREDIST32
        libtiff = WINLIBTIFF32

      self.fetch_package(pkg_name=winpythonpkg, pkg_url=BASE_CCI_PKG_URL)
      winpython = zipfile.ZipFile(os.path.join(self.tmp_dir, winpythonpkg), 'r')
      members = winpython.namelist()
      for zipinfo in members:
        print("extracting", zipinfo, file=self.log)
        winpython.extract(zipinfo, path=os.path.join(self.base_dir,'bin'))
      winpython.close()

      self.fetch_package(pkg_name=hdf5pkg, pkg_url=BASE_CCI_PKG_URL)
      winhdf5 = zipfile.ZipFile(os.path.join(self.tmp_dir, hdf5pkg), 'r')
      members = winhdf5.namelist()
      for zipinfo in members:
        print("extracting", zipinfo, file=self.log)
        winhdf5.extract(zipinfo, path=self.base_dir)
      winhdf5.close()

      self.fetch_package(pkg_name=libtiff, pkg_url=BASE_CCI_PKG_URL)
      winlibtiff = zipfile.ZipFile(os.path.join(self.tmp_dir, libtiff), 'r')
      members = winlibtiff.namelist()
      for zipinfo in members:
        print("extracting", zipinfo, file=self.log)
        winlibtiff.extract(zipinfo, path=self.base_dir)
      winlibtiff.close()

      # Provide the VC++ redistributable libraries in case we are compiling with OpenMP
      self.fetch_package(pkg_name=vcredist, pkg_url=BASE_CCI_PKG_URL)
      # On Windows quit now as all required modules are in the precompiled python package
      # and HDF5 installation
      return

    # Which Python interpreter:
    self.python_exe = None
    python_executable = 'python'
    self.python3 = options.python3
    if self.python3: python_executable = 'python3'
    self.wxpython4 = options.wxpython4 # or self.python3 # Python3 should imply wxpython4, but
                                                         # wait until we can actually build it
    if (not self.wxpython4 and
        self.flag_is_mac and
        get_os_version().startswith('10.') and
        int(get_os_version().split('.')[1]) >= 14):
      print("Setting wxpython4=True as Mac OS X version >= 10.14", file=self.log)
      self.wxpython4 = True
    if os.path.exists(os.path.join(self.build_dir, 'base', 'bin', python_executable)):
      self.python_exe = os.path.join(self.build_dir, 'base', 'bin', python_executable)
    elif options.with_python:
      self.python_exe = options.with_python
    elif options.with_system_python:
      self.python_exe = sys.executable

    if self.python_exe:
      print("Using Python interpreter: %s" % self.python_exe, file=log)

    if not self.python_exe and 'SuSE' in platform.platform():
      if 'CONFIG_SITE' in os.environ:
        print('SuSE detected; clobbering CONFIG_SITE in environ', file=log)
        try:
          del(os.environ['CONFIG_SITE'])
        except: # intentional
          pass

    # Set package config.
    pkg_config_dir = op.join(self.base_dir, "lib", "pkgconfig")
    if (not op.isdir(pkg_config_dir) and not options.download_only):
      os.makedirs(pkg_config_dir)
    pkg_config_paths = [pkg_config_dir] + os.environ.get("PKG_CONFIG_PATH", "").split(":")
    os.environ['PKG_CONFIG_PATH'] = ":".join(pkg_config_paths)

    # Set BLAS/ATLAS/LAPACK
    for env_var in ["BLAS","ATLAS","LAPACK"]:
      os.environ[env_var] = "None"

    # Select packages to build.
    packages = self.configure_packages(options)
    # Override and specified packages if provided.
    if len(args) > 1:
      packages = args[1:]

    # Do the work!
    self.check_dependencies(packages=packages)
    self.build_dependencies(packages=packages)

    # On Mac OS X all of the Python-related executables located in base/bin
    # are actually symlinks to absolute paths inside the Python.framework, so
    # we replace them with symlinks to relative paths.
    if self.flag_is_mac and not self.options.download_only:
      print("Regenerating symlinks with relative paths...", file=log)
      regenerate_relative_symlinks(op.join(self.base_dir, "bin"), log=log)

  def configure_packages(self, options):
    packages = []
    # Package groups.
    if options.molprobity:
      options.build_gui = False
      options.build_all = False
      packages += ['tiff', 'psutil', 'mrcfile']
    if options.cctbx:
      options.build_gui = True
      options.build_all = True
      packages += ['pillow']
    if options.phenix:
      options.build_gui = True
      options.build_all = True
    if options.dials:
      options.build_gui = True
      options.build_all = True
      packages += ['pillow', 'jinja2', 'ordered_set', 'scipy', 'scikit_learn', 'tqdm', 'msgpack']
    if options.xia2:
      options.build_gui = True
      options.build_all = True
      packages += ['pillow', 'jinja2', 'tabulate']
    if options.labelit:
      options.build_gui = True
      options.build_all = True
      packages += ['pillow', 'reportlab']

    # Use a specific Python interpreter if provided.
    if self.python_exe:
      self.set_python(self.python_exe)
    else:
      # for better platform independence, build OpenSSL for Linux and macOS
      # whenever Python is built
      packages += ['python', 'openssl', 'certifi']

    # Python 2-3 compatibility packages
    packages += ['python_compatibility']
    # Always build hdf5 and numpy.
    packages += ['cython', 'hdf5', 'h5py', 'numpy', 'pythonextra', 'docutils']
    packages += ['libsvm', 'lz4_plugin']
    # Development and testing packages.
    packages += ['pytest']
    # GUI packages.
    if options.build_gui or options.build_all or options.download_only:
      packages += [
        'png',
        'tiff',
        'pillow',
        'freetype',
        'matplotlib',
        'msgpack',
        'pyopengl',
        'wxpython',
      ]
      if self.flag_is_mac or options.download_only:
        packages += ['py2app']
      if self.flag_is_linux or options.download_only:
        packages += [
          'gettext',
          'glib',
          'expat',
          'fontconfig',
          'render',
          'pixman',
          'cairo',
          'gtk',
          'fonts',
        ]

    # Additional recommended dependencies.
    if options.build_all:
      packages += ['biopython', 'misc', 'sphinx', 'psutil', 'mrcfile']

    # Non-all packages.
    # Scipy
    if options.build_scipy:
      packages += ['scipy']

    # IPython
    if options.build_ipython:
      packages += ['ipython']

    # Package dependencies.
    if 'pillow' in packages:
      packages += ['freetype']

    # mpi4py installation
    if options.mpi_build:
      packages +=['mpi4py']

    return set(packages)

  def call(self, args, log=None, **kwargs):
    if (log is None) : log = self.log
    return call(args, log=log, verbose=self.verbose, **kwargs)

  def chdir(self, dir_name, log=None):
    if (log is None) : log = self.log
    print("cd \"%s\"" % dir_name, file=log)
    os.chdir(dir_name)

  def touch_file(self, file_name):
    f = open(file_name, "w")
    f.write("")
    f.close()
    assert op.isfile(file_name)

  def print_sep(self, char="-"):
    print("", file=self.log)
    print(char*80, file=self.log)
    print("", file=self.log)

  def start_building_package(self, pkg_name, pkg_info=None, pkg_qualifier=''):
    os.chdir(self.tmp_dir)
    install_log = op.join(self.tmp_dir, pkg_name + "_install_log")
    print("Installing %s%s..." % (pkg_name, pkg_qualifier), file=self.log)
    if pkg_info:
      print(pkg_info, file=self.log)
    print("  log file is %s" % install_log, file=self.log)
    return open(install_log, "w")

  def check_download_only(self, pkg_name=None):
    if pkg_name is not None:
      if self.options.download_only:
        print("  skipping installation of %s (--download-only)" % pkg_name, file=self.log)
      else:
        print("  installing %s..." % pkg_name, file=self.log)
    return self.options.download_only

  @staticmethod
  def patch_src(src_file, target, replace_with, output_file=None):
    from shutil import copymode
    if isinstance(target, str):
      assert isinstance(replace_with, str)
      target = [ target ]
      replace_with = [ replace_with ]
    assert len(target) == len(replace_with)
    in_file = src_file
    if (output_file is None):
      in_file += ".dist"
      os.rename(src_file, in_file)
    src_in = open(in_file)
    if (output_file is None):
      output_file = src_file
    src_out = open(output_file, "w")
    for line in src_in.readlines():
      for target_str, replacement_str in zip(target, replace_with):
        line = line.replace(target_str, replacement_str)
      src_out.write(line)
    src_in.close()
    src_out.close()
    copymode(in_file, output_file)

  def untar_and_chdir(self, pkg, log=None):
    if (log is None) : log = self.log
    pkg_dir = untar(pkg, log=log)
    os.chdir(pkg_dir)

  def check_python_version(self):
    try:
      python_version = check_output([
        self.python_exe,
        '-c',
        'import sys; print("%d:%d:%d:%s" % (sys.version_info[0], sys.version_info[1], sys.hexversion, sys.version))',
        ])
    except (OSError, RuntimeError):
      print("""
Error: Could not determine version of Python installed at:
  %s
Error: Python 2.7 or Python 3.4 or higher required. Python3 only supported for development purposes.
Found Python version:
  %s
""" % self.python_exe, file=self.log)
      sys.exit(1)
    if isinstance(python_version, bytes):
        python_version = python_version.decode("utf-8")
    python_version = python_version.strip().split('\n')[0].split(':', 3)
    if (int(python_version[0]) == 2 and int(python_version[1]) < 7) or \
       (int(python_version[0]) > 2 and int(python_version[2]) < 0x03040000):
      print("Error: Python 2.7 or 3.4+ required.\nFound Python version: %s" % python_version[3], file=self.log)
      sys.exit(1)

  def check_python_dependencies(self):
    for module, desc in python_dependencies.items():
      try:
        self.verify_python_module(module, module)
      except RuntimeError:
        print('\n\n\n%s\n\nThis python does not have %s installed\n\n%s\n\n\n' % (
          "*"*80,
          "%s - %s" % (module, desc),
          "*"*80,
        ))
        raise

  def set_python(self, python_exe):
    print("Using Python: %s"%python_exe, file=self.log)
    self.python_exe = python_exe
    # Just an arbitrary import (with .so)
    self.verify_python_module("Python", "socket")
    # check that certain modules are avaliable in python
    self.check_python_dependencies()
    # Check python version >= 2.7 or >= 3.4
    self.check_python_version()

    # Check that we have write access to site-packages dir
    # by creating a temporary file with write permissions.
    # Open with tempfile to auto-handle unlinking.
    import tempfile
    site_packages = check_output([self.python_exe, '-c', 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'])
    site_packages = site_packages.strip()
    print("Checking for write permissions:", site_packages, file=self.log)
    try:
      f = tempfile.TemporaryFile(dir=site_packages)
    except (OSError, RuntimeError):
      print("""
Error: You don't appear to have write access to
the Python site-packages directory:
  %s
Installation of Python packages may fail.
      """%site_packages, file=self.log)
      raise e
    # Update paths.
    self.update_paths()

  def update_paths(self):
    os.environ["PATH"] = ("%s/bin:" % self.base_dir) + os.environ['PATH']
    lib_paths = [ op.join(self.base_dir, "lib") ]
    if self.flag_is_linux:
      if ("LD_LIBRARY_PATH" in os.environ):
        lib_paths.append(os.environ["LD_LIBRARY_PATH"])
      os.environ['LD_LIBRARY_PATH'] = ":".join(lib_paths)
    inc_dir = op.join(self.base_dir, "include")
    if (not op.isdir(inc_dir)):
      os.mkdir(inc_dir)
    self.include_dirs.append(inc_dir)
    self.lib_dirs.append(lib_paths[0])

  def set_cppflags_ldflags(self):
    # XXX ideally we would like to quote the paths to allow for spaces, but
    # the compiler doesn't like this
    inc_paths = ["-I%s" % p for p in self.include_dirs]
    lib_paths = ["-L%s" % p for p in self.lib_dirs]
    os.environ['CPPFLAGS'] = "%s %s"%(" ".join(inc_paths), self.cppflags_start)
    os.environ['LDFLAGS'] = "%s %s"%(" ".join(lib_paths), self.ldflags_start)

  def verify_python_module(self, pkg_name_label, module_name):
    os.chdir(self.tmp_dir) # very important for import to work!
    if hasattr(self, "python_exe"): python_exe = self.python_exe
    else:
      python_exe = sys.executable
    self.log.write("  verifying %s installation in %s" % (
      pkg_name_label,
      python_exe,
    ))
    if sys.platform == "win32": # Windows is picky about using double quotes rather than single quotes
      self.call('"%s" -c "import %s"' % (python_exe, module_name))
    else:
      self.call("%s -c 'import %s'" % (python_exe, module_name))
    print(" OK", file=self.log)

  def workarounds(self):
    '''Look at the directory I am in at the moment and the platform I am
    running on and (perhaps) mess with things to make the build work.'''

    # Ubuntu & Xrender - libtool is broken - replace with system one if
    # installed...

    if 'xrender' in os.path.split(os.getcwd())[-1] and \
      'Ubuntu' in platform.platform() and os.path.exists('libtool'):
      if os.path.exists(os.path.join('/', 'usr', 'bin', 'libtool')):
        self.log.write('Removing xrender libtool; replace with system\n')
        os.remove('libtool')
        os.symlink(os.path.join('/', 'usr', 'bin', 'libtool'), 'libtool')
      else:
        self.log.write('Cannot removing xrender libtool; not installed\n')

    # patch cairo for Ubuntu 12.04
    # issue: configure step does not find these functions from Xrender
    # https://lists.cairographics.org/archives/cairo/2014-September/025552.html
    # fix: manually set these functions to be present in config.h
    if ( ('cairo' in os.path.split(os.getcwd())[-1]) and
         ('Ubuntu' in platform.platform()) and
         ('precise' in platform.platform()) ):
      self.patch_src(src_file='config.h',
                     target="/* #undef HAVE_XRENDERCREATECONICALGRADIENT */",
                     replace_with="#define HAVE_XRENDERCREATECONICALGRADIENT 1")
      self.patch_src(src_file='config.h',
                     target="/* #undef HAVE_XRENDERCREATELINEARGRADIENT */",
                     replace_with="#define HAVE_XRENDERCREATELINEARGRADIENT 1")
      self.patch_src(src_file='config.h',
                     target="/* #undef HAVE_XRENDERCREATERADIALGRADIENT */",
                     replace_with="#define HAVE_XRENDERCREATERADIALGRADIENT 1")
      self.patch_src(src_file='config.h',
                     target="/* #undef HAVE_XRENDERCREATESOLIDFILL */",
                     replace_with="#define HAVE_XRENDERCREATESOLIDFILL 1")

    return

  def configure_and_build(self, config_args=(), log=None, make_args=(), limit_nproc=None):
    # case sensitive file system workaround
    configure = filter(os.path.exists, ('config', 'configure', 'Configure'))
    working_configure = False
    for c in configure:
      if ( os.path.isfile(c) and os.access(c, os.X_OK) ):
        configure = c
        working_configure = True
        break
    assert working_configure, 'No configure script found'
    self.call("./%s %s" % (configure, " ".join(list(config_args))), log=log)
    self.workarounds()
    nproc = self.nproc
    if limit_nproc is not None:
      nproc = min(nproc, limit_nproc)
    #print os.getcwd()
    #print "make -j %d %s" % (nproc, " ".join(list(make_args)))
    self.call("make -j %d %s" % (nproc, " ".join(list(make_args))), log=log)
    self.call("make install", log=log)

  def build_compiled_package_simple(self, pkg_name,pkg_name_label,
                                    pkg_url=None, extra_config_args=None):
    pkg_log = self.start_building_package(pkg_name_label)
    pkg = self.fetch_package(pkg_name=pkg_name, pkg_url=pkg_url)
    if self.check_download_only(pkg_name): return
    self.untar_and_chdir(pkg=pkg, log=pkg_log)
    config_args = [self.prefix]
    if (isinstance(extra_config_args,list)):
      config_args += extra_config_args
    self.configure_and_build(config_args=config_args, log=pkg_log)

  def build_python_module_simple(self,
      pkg_url,
      pkg_name,
      pkg_name_label,
      pkg_local_file=None,
      callback_before_build=None,
      callback_after_build=None,
      confirm_import_module=None):
    pkg_log = self.start_building_package(pkg_name_label)
    if pkg_local_file is None:
      pkg_local_file, size = self.fetch_package(pkg_name=pkg_name,
                                   pkg_url=pkg_url,
                                   return_file_and_status=True)
    if self.check_download_only(pkg_name): return
    self.untar_and_chdir(pkg=pkg_local_file, log=pkg_log)
    if (callback_before_build is not None):
      assert callback_before_build(pkg_log), pkg_name
    debug_flag = ""
    if (self.options.debug):
      debug_flag = "--debug"
    self.call("%s setup.py build %s" % (self.python_exe, debug_flag),
      log=pkg_log)
    self.call("%s setup.py install" % self.python_exe, log=pkg_log)
    if (callback_after_build is not None):
      assert callback_after_build(pkg_log), pkg_name
    os.chdir(self.tmp_dir)
    if (confirm_import_module is not None):
      self.verify_python_module(pkg_name_label, confirm_import_module)

  def build_python_module_pip(self, package_name, package_version=None, download_only=None,
      callback_before_build=None, callback_after_build=None, confirm_import_module=None,
      extra_options=None):
    '''Download and install a package using pip.'''
    if download_only is None:
      download_only = self.options.download_only

    pkg_info = get_pypi_package_information(package_name, information_only=True)
    pkg_info['package'] = package_name
    pkg_info['python'] = self.python_exe
    pkg_info['cachedir'] = self.fetch_package.dest_dir
    pkg_info['debug'] = ''
    if self.options.debug:
      pkg_info['debug'] = '-v'
    if package_version:
      if '=' in package_version or '>' in package_version or '<' in package_version:
        pkg_info['version'] = package_version
      else:
        pkg_info['version'] = '==' + package_version
    else:
      pkg_info['version'] = ''
    if extra_options:
      assert isinstance(extra_options, list), 'extra pip options must be passed as a list'
    if download_only:
      pip_version_cmd = ['python', '-c', 'import pip; print(pip.__version__)']
      pip_version_result = subprocess.run(pip_version_cmd, stdout=subprocess.PIPE)
      if pip_version_result.returncode != 0:
        print("Skipping download of python package %s %s" % \
              (pkg_info['name'], pkg_info['version']))
        print("Your current python environment does not include 'pip',")
        print("which is required to download prerequisites.")
        print("Please see https://pip.pypa.io/en/stable/installing/ for " \
              "more information.")
        print("*" * 75)
        return
    log = self.start_building_package(package_name,
             pkg_info=pkg_info['summary'],
             pkg_qualifier=' ' + (package_version or ''))
    if download_only:
      pip_cmd = filter(None, [sys.executable, '-m', 'pip', 'download',
                              pkg_info['package'] + pkg_info['version'],
                              '-d', pkg_info['cachedir'], pkg_info['debug']])
      if extra_options:
        pip_cmd.extend(extra_options)
      os.environ['PIP_REQ_TRACKER'] = pkg_info['cachedir']
      print("  Running with pip:", pip_cmd)

      assert subprocess.run(pip_cmd).returncode == 0, 'pip download failed'
      return
    if extra_options:
      extra_options = ' '.join(extra_options)
    else:
      extra_options = ''
    if callback_before_build:
      self.call(pkg_info['python'] + ' -m pip download ' + pkg_info['debug'] + \
                ' "' + pkg_info['package'] + pkg_info['version'] + '" -d "' + \
                pkg_info['cachedir'] + '" ' + extra_options,
                log=log)
      assert callback_before_build(log), package_name
      self.call(pkg_info['python'] + ' -m pip install ' + pkg_info['debug'] + \
                ' "' + pkg_info['package'] + pkg_info['version'] + \
                '" --no-index -f "' + pkg_info['cachedir'] + '" ' + extra_options,
                log=log)
    else:
      self.call(pkg_info['python'] + ' -m pip install ' + pkg_info['debug'] + \
                ' "' + pkg_info['package'] + pkg_info['version'] + '" ' + extra_options,
                log=log)
    if callback_after_build:
      assert callback_after_build(log), package_name
    if confirm_import_module:
      os.chdir(self.tmp_dir)
      self.verify_python_module(pkg_info['name'], confirm_import_module)

  def check_dependencies(self, packages=None):
    packages = packages or []

    # no-op

  def build_dependencies(self, packages=None):
    # Build in the correct dependency order.
    packages = packages or []
    order = [
      'openssl',
      'python',
      'python_compatibility',
      'certifi',
      'numpy',
      'cython',
      'png',
      'libsvm',
      'pytest',
      'pythonextra',
      'hdf5',
      'h5py',
      'biopython',
      'docutils',
      'sphinx',
      'ipython',
      'pyopengl',
      'scipy',
      'scikit_learn',
      'mpi4py',
      'py2app',
      'misc',
      'lz4_plugin',
      'jinja2',
      'ordered_set',
      'tqdm',
      'tabulate',
      'psutil',
      'mrcfile',
      # ...
      'freetype',
      'matplotlib',
      'msgpack',
      'pillow',
      'reportlab',
      # START GUI PACKAGES
      'gettext',
      'glib',
      'expat',
      'fontconfig',
      'render',
      'pixman',
      'tiff',
      'cairo',
      'gtk',
      'fonts',
      'wxpython',
    ]
    self.options.skip_base = self.options.skip_base.split(",")
    packages_order = []
    for i in packages:
      assert i in order, "Installation order unknown for %s" % i
    for i in order:
      if i in packages:
        if i in self.options.skip_base: continue
        packages_order.append(i)
    if self.options.continue_from and self.options.continue_from in packages_order:
      packages_order = packages_order[packages_order.index(self.options.continue_from):]

    if self.options.download_only:
      print("Downloading dependencies: %s"%(" ".join(packages_order)), file=self.log)
      action = "download"
    else:
      print("Building dependencies: %s"%(" ".join(packages_order)), file=self.log)
      action = "install"

    os.chdir(self.tmp_dir)
    for i in packages_order:
      self.set_cppflags_ldflags() # up-to-date LDFLAGS/CPPFLAGS
      self.print_sep()
      t0=time.time()
      getattr(self, 'build_%s'%i)()
      print("  package %s took %0.1fs to %s" % (
        i,
        time.time()-t0,
        action
        ), file=self.log)

    if self.options.download_only:
      print("Dependencies finished downloading.", file=self.log)
    else:
      print("Dependencies finished building.", file=self.log)

  #######################################################
  ##### Build Individual Packages #######################
  #######################################################

  def build_python(self):
    if self.python3:
      return self.build_python3()

    if self.flag_is_mac and not op.exists('/usr/include/zlib.h'):
      print("zlib.h missing -- try running 'xcode-select --install' first", file=self.log)
      if get_os_version().startswith('10.') and int(get_os_version().split('.')[1]) >= 14:
        print("followed by 'sudo installer -pkg /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg -target /", file=self.log)
      sys.exit(1)
    log = self.start_building_package("Python")
    os.chdir(self.tmp_dir)
    python_tarball = self.fetch_package(pkg_name=PYTHON_PKG, pkg_url=DEPENDENCIES_BASE)
    if self.check_download_only(PYTHON_PKG): return
    python_dir = untar(python_tarball)
    self.chdir(python_dir, log=log)

    # configure macOS and linux separately
    configure_args = [] # common flags should go here
    if self.flag_is_mac:
      configure_args += [
        self.prefix,
        'CPPFLAGS="-I%s/include %s"' %
        (self.base_dir, os.environ.get('CPPFLAGS', '')),
        'LDFLAGS="-L%s/lib %s"' %
        (self.base_dir, os.environ.get('LDFLAGS', '')),
        "--enable-framework=\"%s\"" % self.base_dir,
        ]
      self.call("./configure %s" % " ".join(configure_args), log=log)

      # patch Mac/Makefile to avoid cruft in /Applications
      targets = ['PYTHONAPPSDIR=/Applications/$(PYTHONFRAMEWORK) $(VERSION)',
                 'installapps: install_Python install_pythonw install_BuildApplet install_PythonLauncher \\',
                 'install_IDLE checkapplepython install_versionedtools']
      replacements = ['PYTHONAPPSDIR=',
                      'installapps: install_Python install_pythonw \\',
                      'checkapplepython install_versionedtools']
      for target,replacement in zip(targets,replacements):
        self.patch_src(src_file='Mac/Makefile',
                       target=target, replace_with=replacement)
    else:
      # Linux
      # Ian: Setup build to use rpath $ORIGIN to find libpython.so.
      # Also note that I'm using shell=False and passing args as list
      #   ... to minimize opportunities for mucking up the "\$$"
      configure_args += ["--prefix", self.base_dir]
      if (self.options.python_shared):
        configure_args.append("--enable-shared")
        configure_args.append("LDFLAGS=-Wl,-rpath=$ORIGIN/../lib")

      # patch Modules/Setup.dist to find custom OpenSSL
      targets = ['#SSL=/usr/local/ssl',
                 '#_ssl _ssl.c \\',
                 '#\t-DUSE_SSL -I$(SSL)/include -I$(SSL)/include/openssl \\',
                 '#\t-L$(SSL)/lib -lssl -lcrypto']
      replacements = ['SSL=%s' % self.base_dir,
                      '_ssl _ssl.c \\',
                      '  -DUSE_SSL -I$(SSL)/include -I$(SSL)/include/openssl \\',
                      '  -L$(SSL)/lib -lssl -lcrypto']
      for target,replacement in zip(targets,replacements):
        self.patch_src(src_file='Modules/Setup.dist',
                       target=target, replace_with=replacement)
      self.call([os.path.join(python_dir, 'configure')] + configure_args,
                log=log, cwd=python_dir, shell=False)

    # build (serial because of potential for race conditions)
    self.call('make', log=log, cwd=python_dir)
    self.call('make install', log=log, cwd=python_dir)
    python_exe = op.abspath(op.join(self.base_dir, "bin", "python"))

    # Install pip *separately* from the python install - this ensures that
    # we don't accidentally pick up the user-site pythonpath
    self.call([python_exe, "-mensurepip"], log=log, cwd=python_dir)

    # Make python relocatable
    python_sysconfig = check_output([ python_exe, '-c',
      'import sys; import os; print os.path.join(sys.exec_prefix, "lib", "python2.7", "_sysconfigdata.py")'
      ]).rstrip()
    try:
      fh = open(python_sysconfig, 'r')
      python_config = fh.read()
      fh.close()
      if 'relocatable' not in python_config:
        fh = open(python_sysconfig, 'a')
        fh.write("""
#
# Fix to make python installation relocatable
#

def _replace_sysconfig_paths(d):
  from os import environ
  from sys import executable
  path = environ.get('LIBTBX_PYEXE', executable)
  if '/base/' in path:
    path = path[:path.rfind('/base/')+5]
    for k, v in d.iteritems():
      if isinstance(v, basestring):
        d[k] = v.replace('%s', path)
_replace_sysconfig_paths(build_time_vars)
""" % self.base_dir)
        fh.close()
    except Exception as e:
      print("Could not make python relocatable:", file=log)
      print(e, file=log)

    # On macOS, base/Python.framework/Versions/2.7/Python (aka
    # libpython2.7.dylib) may be read-only. This affects the create-installer
    # step that makes Python relocatable. For applications (e.g. Rosetta) that
    # link to this library, this can cause crashes, so make it writeable.
    if (self.flag_is_mac):
      filename = os.path.join(self.base_dir, 'Python.framework', 'Versions',
                              '2.7', 'Python')
      os.chmod(filename, 0o755)  # set permissions to -rwxr-xr-x

    self.set_python(op.abspath(python_exe))
    log.close()

  def build_python3(self):
    if self.flag_is_mac and not op.exists('/usr/include/zlib.h'):
      print("zlib.h missing -- try running 'xcode-select --install' first", file=self.log)
      sys.exit(1)
    log = self.start_building_package("Python3")
    os.chdir(self.tmp_dir)
    python_tarball = self.fetch_package(pkg_name=PYTHON3_PKG, pkg_url=DEPENDENCIES_BASE)
    if self.check_download_only(PYTHON3_PKG): return
    python_dir = untar(python_tarball)
    self.chdir(python_dir, log=log)

    configure_args = ['--with-ensurepip=install', '--prefix=' + self.base_dir]
    if (self.options.python_shared):
      configure_args.append("--enable-shared")
    environment = os.environ.copy()
    environment['LDFLAGS'] = "-L{base}/lib/ -L{base}/lib64/".format(base=self.base_dir)
    environment['LDFLAGS'] += " -Wl,-rpath,{base}/lib".format(base=self.base_dir)
    environment['LD_LIBRARY_PATH'] = "{base}/lib/:{base}/lib64/".format(base=self.base_dir)
    environment['CPPFLAGS'] = "-I{base}/include -I{base}/include/openssl".format(base=self.base_dir)
    self.call([os.path.join(python_dir, 'configure')] + configure_args,
              log=log, cwd=python_dir, shell=False, env=environment)
    self.call(['make', '-j', str(self.nproc)],
              log=log, cwd=python_dir, shell=False, env=environment)
    self.call(['make', 'install', '-j', str(self.nproc)],
              log=log, cwd=python_dir, shell=False, env=environment)
    python_exe = op.abspath(op.join(self.base_dir, "bin", "python3"))
    self.set_python(op.abspath(python_exe))

    # Make python relocatable - unclear if required
#   python_sysconfig = check_output([ python_exe, '-c',
#     'import os; import sys; import sysconfig; print(os.path.join(os.path.dirname(sysconfig.__file__), sysconfig._get_sysconfigdata_name() + ".py"))'
#     ]).rstrip()
    if False: # try:
      with open(python_sysconfig, 'r') as fh:
        python_config = fh.read()
      if 'relocatable' not in python_config:
        with open(python_sysconfig, 'a') as fh:
          fh.write("""
#
# Fix to make python installation relocatable
#

def _replace_sysconfig_paths(d):
  from os import environ
  from sys import executable
  path = environ.get('LIBTBX_PYEXE', executable)
  if '/base/' in path:
    path = path[:path.rfind('/base/')+5]
    for k, v in d.iteritems():
      if isinstance(v, basestring):
        d[k] = v.replace('%s', path)
_replace_sysconfig_paths(build_time_vars)
""" % self.base_dir)
#   except Exception, e:
      print("Could not make python relocatable:", file=log)
      print(e, file=log)
    log.close()

  def build_python_compatibility(self):
    self.build_python_module_pip(
      'six', package_version=SIX_VERSION,
      confirm_import_module='six')
    self.build_python_module_pip(
      'future', package_version=FUTURE_VERSION,
      confirm_import_module='future')

  def build_pythonextra(self):
    '''install all python packages found in base_tmp/python_extra/'''

    python_extra_dir = 'python_extra'
    python_extra_full_path = os.path.join(self.tmp_dir, python_extra_dir)
    if os.path.exists(python_extra_full_path):
      print("Installing further python packages...\n", file=self.log)
    else:
      print("No further python packages to install.", file=self.log)
      return True

    files = [ f for f in os.listdir(python_extra_full_path) if f.endswith(".tar.gz") ]

    python_extra_order = os.path.join(python_extra_full_path, 'install.order')
    if os.path.exists(python_extra_order):
      f = open(python_extra_order)
      reorder = []
      for line in f.readlines():
        line = line.strip()
        if line in files:
          files.remove(line)
          reorder.append(line)
      reorder.extend(files)
      files = reorder
      f.close()

    for pkg in files:
      self.build_python_module_simple(
        pkg_url=None, pkg_local_file=os.path.join(python_extra_full_path, pkg),
        pkg_name=pkg, pkg_name_label=os.path.join(python_extra_dir, pkg[:-7]))

  def simple_log_parse_test(self, log_filename, line):
    if not os.path.exists(log_filename): return False
    f = file(log_filename, "rb")
    log_lines = f.read()
    f.close()
    if log_lines.find(line)>-1:
      return True
    return False

  def build_libsvm(self):
    self.build_python_module_simple(
      pkg_url=BASE_CCI_PKG_URL,
      pkg_name=LIBSVM_PKG,
      pkg_name_label="libsvm")

  def build_numpy(self):
    self.build_python_module_pip(
      package_name='numpy',
      package_version=NUMPY_VERSION,
      confirm_import_module="numpy",
    )

  def build_docutils(self):
    self.build_python_module_pip(
      package_name='docutils',
      package_version=DOCUTILS_VERSION,
      confirm_import_module="docutils",
    )

  def build_pytest(self):
    self.build_python_module_pip(
      'mock', package_version=MOCK_VERSION)
    self.build_python_module_pip(
      'pytest', package_version=PYTEST_VERSION)
    self.build_python_module_pip(
      'pytest-xdist', package_version=PYTEST_XDIST_VERSION)

  def build_biopython(self):
    self.build_python_module_simple(
      pkg_url=BASE_CCI_PKG_URL,
      pkg_name=BIOPYTHON_PKG,
      pkg_name_label="biopython",
      confirm_import_module="Bio")

  def build_lz4_plugin(self, patch_src=True):
    log = self.start_building_package("lz4_plugin")
    repos = ["hdf5_lz4", "bitshuffle"]
    for repo in repos:
      fetch_remote_package(repo, log=log, use_ssh=self.options.git_ssh)
    if self.check_download_only("lz4 plugin"): return
    if (patch_src):
      print("Patching hdf5_lz4/Makefile", file=log)
      self.patch_src(src_file="hdf5_lz4/Makefile",
                     target="HDF5_INSTALL = /home/det/hdf5-1.8.11/hdf5/",
                     replace_with="HDF5_INSTALL = %s"%self.base_dir)
      print("Patching bitshuffle/setup.py", file=log)
      self.patch_src(src_file="bitshuffle/setup.py",
                     target=["COMPILE_FLAGS = ['-O3', '-ffast-math', '-march=native', '-std=c99']",
                             'raise ValueError("pkg-config must be installed")',
                             "FALLBACK_CONFIG['include_dirs'] = [d for d in FALLBACK_CONFIG['include_dirs']", "if path.isdir(d)]",
                             "FALLBACK_CONFIG['library_dirs'] = [d for d in FALLBACK_CONFIG['library_dirs']", "if path.isdir(d)]",
                             "except subprocess.CalledProcessError:",
                            ],
                     replace_with=[
                             "COMPILE_FLAGS = ['-O3', '-std=c99']",
                             "pass",
                             "FALLBACK_CONFIG['include_dirs'] = ['%s/include']"%self.base_dir, "",
                             "FALLBACK_CONFIG['library_dirs'] = ['%s/lib']"%self.base_dir, "",
                             "except (subprocess.CalledProcessError, OSError):",
                             ])
    self.chdir("hdf5_lz4",log=log)
    self.call("make", log=log)
    self.chdir("../bitshuffle",log=log)
    site_file = open("setup.cfg", "w")
    site_file.write("[build_ext]\nomp = 0\n")
    site_file.close()
    self.call("CFLAGS='-std=c99' %s setup.py build"%self.python_exe,log=log)
    self.call("CFLAGS='-std=c99' %s setup.py install --h5plugin --h5plugin-dir=../hdf5_lz4"%(self.python_exe),log=log)
    self.chdir("../hdf5_lz4",log=log)
    print("Copying new libraries to base/lib/plugins folder", file=log)
    hdf5_plugin_dir = os.path.join(self.base_dir, "lib", "plugins")
    if not os.path.exists(hdf5_plugin_dir):
      os.mkdir(hdf5_plugin_dir)
    self.call("cp -v *.so %s"%hdf5_plugin_dir,log=log)

  def build_scipy(self):
    self.build_python_module_pip(
      package_name='scipy',
      package_version=SCIPY_VERSION,
      confirm_import_module="scipy")

  def build_scikit_learn(self):
    self.build_python_module_pip(
      package_name='scikit-learn',
      package_version=SCIKIT_LEARN_VERSION,
      confirm_import_module="sklearn")

  def build_mpi4py(self):
    self.build_python_module_pip(
      package_name='mpi4py',
      package_version=MPI4PY_VERSION,
      confirm_import_module="mpi4py")

  def build_py2app(self):
    self.build_python_module_pip(
      package_name='py2app',
      package_version=PY2APP_VERSION,
      confirm_import_module="py2app")

  def build_reportlab(self):
    self.build_python_module_simple(
      pkg_url=BASE_CCI_PKG_URL,
      pkg_name=REPORTLAB_PKG,
      pkg_name_label="reportlab",
      confirm_import_module="reportlab")

  def build_msgpack(self):
    self.build_python_module_pip(
      package_name='msgpack',
      package_version=MSGPACK_VERSION,
      confirm_import_module="msgpack",
    )

  def build_pillow(self):
    self.build_python_module_pip(
      package_name='Pillow',
      package_version=PILLOW_VERSION,
      confirm_import_module="PIL",
    )

  def build_sphinx(self):
    self.build_python_module_pip(
      package_name="Sphinx",
      package_version=SPHINX_VERSION,
      confirm_import_module="sphinx")

  def build_ipython(self):
    self.build_python_module_simple(
      pkg_url=BASE_CCI_PKG_URL,
      pkg_name=IPYTHON_PKG,
      pkg_name_label="IPython",
      confirm_import_module="IPython")

  def build_pyopengl(self):
      self.build_python_module_simple(
        pkg_url=BASE_CCI_PKG_URL,
        pkg_name=PYOPENGL_PKG,
        pkg_name_label="pyopengl",
        confirm_import_module="OpenGL")

  def build_cython(self):
    self.build_python_module_pip(
      'Cython', package_version=CYTHON_VERSION,
      confirm_import_module='cython')

  def build_jinja2(self):
    self.build_python_module_pip(
      'Jinja2', package_version=JINJA2_VERSION,
      confirm_import_module='jinja2')

  def build_orderedset(self):
    self.build_python_module_pip(
      'ordered_set', package_version=ORDEREDSET_VERSION,
      confirm_import_module='ordered_set')

  def build_tqdm(self):
    self.build_python_module_pip('tqdm', package_version=TQDM_VERSION)

  def build_tabulate(self):
    self.build_python_module_pip(
      'tabulate', package_version=TABULATE_VERSION,
      confirm_import_module='tabulate')

  def build_psutil(self):
    self.build_python_module_pip(
      'psutil', package_version=PSUTIL_VERSION,
      confirm_import_module='psutil')

  def build_mrcfile(self):
    self.build_python_module_pip(
      'mrcfile', package_version=MRCFILE_VERSION,
      confirm_import_module='mrcfile')

  def build_hdf5(self):
    pkg_log = self.start_building_package("HDF5")
    hdf5pkg = self.fetch_package(pkg_name=HDF5_PKG, pkg_url=BASE_HDF5_PKG_URL)
    if self.check_download_only(HDF5_PKG): return
    self.untar_and_chdir(pkg=hdf5pkg, log=pkg_log)
    print("Building base HDF5 library...", file=pkg_log)
    make_args = []
    # XXX the HDF5 library uses '//' for comments, which will break if the
    # compiler doesn't support C99 by default.  for some bizarre reason this
    # includes certain (relatively new) versions of gcc...
    if self.flag_is_linux:
      make_args.append("CFLAGS=\"-std=c99\"")
    self.configure_and_build(
      config_args=[self.prefix, "--enable-build-mode=production",],
      log=pkg_log,
      make_args=make_args)

  def build_h5py(self):
    os.environ['HDF5_DIR'] = self.base_dir
    self.build_python_module_pip(
      'h5py', package_version=H5PY_VERSION,
      confirm_import_module='h5py', extra_options=["--no-binary=h5py"])

  def build_openssl(self):
    # https://wiki.openssl.org/index.php/Compilation_and_Installation#Configure_.26_Config
    # http://stackoverflow.com/a/20740964

    pkg_url=DEPENDENCIES_BASE
    pkg_name=OPENSSL_PKG
    pkg_name_label="OpenSSL"
    pkg_log = self.start_building_package(pkg_name_label)
    pkg = self.fetch_package(pkg_name=pkg_name, pkg_url=pkg_url)
    if self.check_download_only(pkg_name): return
    self.untar_and_chdir(pkg=pkg, log=pkg_log)
    if (self.flag_is_mac):
      # help config select darwin64-x86_64-cc (required for 10.9)
      os.environ['KERNEL_BITS'] = '64'
    self.configure_and_build(
      config_args=[self.prefix, "-fPIC", "no-hw", "--openssldir=share"],
      log=pkg_log, limit_nproc=1) # openssl is not parallel buildable
    self.include_dirs.append(op.join(self.base_dir, "include", "openssl"))
    if (self.flag_is_mac):
      os.environ['KERNEL_BITS'] = ''

  def build_certifi(self):
    # No version specified - always take latest version
    self.build_python_module_pip(
      'certifi', confirm_import_module="certifi")
    if self.check_download_only(): return

    # set environment variable for root certificates
    # this affects future pip commands in the installation process and only
    # seems to be needed for macOS 10.11
    cert_file = check_output([self.python_exe, '-c',
                              'import certifi; print(certifi.where())'])
    cert_file = cert_file.strip()
    try:
      os.environ['SSL_CERT_FILE'] = cert_file
    except TypeError: # str is needed instead of bytes (Python 3)
      os.environ['SSL_CERT_FILE'] = cert_file.decode("utf-8")
    print('SSL_CERT_FILE environment variable set to %s' % \
      cert_file, file=self.log)

  def build_freetype(self):
    self.build_compiled_package_simple(
      pkg_url=BASE_CCI_PKG_URL,
      pkg_name=FREETYPE_PKG,
      pkg_name_label="Freetype")
    if self.check_download_only(): return
    self.include_dirs.append(op.join(self.base_dir, "include", "freetype2"))
    # copy ft2build.h from include/freetype2 to inculde/ (for matplotlib)
    from shutil import copy
    copy(op.join(self.base_dir, 'include', 'freetype2', 'ft2build.h'),
         op.join(self.base_dir, 'include'))

  def build_png(self):
    self.build_compiled_package_simple(
      pkg_url=BASE_CCI_PKG_URL,
      pkg_name=LIBPNG_PKG,
      pkg_name_label="libpng")

  def build_gettext(self):
    # gettext
    pkg_log = self.start_building_package("gettext")
    pkg = self.fetch_package(pkg_name=GETTEXT_PKG)
    if self.check_download_only(GETTEXT_PKG): return
    self.untar_and_chdir(pkg=pkg, log=pkg_log)
    os.chdir("gettext-runtime")
    gettext_conf_args = [self.prefix, "--disable-java", "--disable-csharp",
      "--disable-intl-java", "--disable-gcj"]
    self.configure_and_build(config_args=gettext_conf_args, log=pkg_log)

  def build_glib(self):
    # libffi dependency (for CentOS 5, especially)
    self.build_compiled_package_simple(pkg_url=BASE_CCI_PKG_URL,
                                       pkg_name=LIBFFI_PKG,
                                       pkg_name_label='libffi')

    # glib
    pkg_log = self.start_building_package("glib")
    pkg = self.fetch_package(pkg_name=GLIB_PKG)
    if self.check_download_only(GLIB_PKG): return

    # Mock executables.
    if (not op.isdir(op.join(self.base_dir, "bin"))):
      os.makedirs(op.join(self.base_dir, "bin"))
    msgfmt_bin = op.join(self.base_dir, "bin", "msgfmt")
    gettext_bin = op.join(self.base_dir, "bin", "xgettext")
    self.touch_file(msgfmt_bin)
    self.touch_file(gettext_bin)
    self.call("chmod 744 \"%s\""%msgfmt_bin)
    self.call("chmod 744 \"%s\""%gettext_bin)
    # glib
    self.untar_and_chdir(pkg=pkg, log=pkg_log)
    self.configure_and_build(
      config_args=[self.prefix, "--disable-selinux"],
      log=pkg_log)

  def build_expat(self):
    # expat
    pkg_log = self.start_building_package("expat")
    pkg = self.fetch_package(pkg_name=EXPAT_PKG)
    if self.check_download_only(EXPAT_PKG): return
    self.untar_and_chdir(pkg=pkg, log=pkg_log)
    self.configure_and_build(config_args=[self.prefix], log=pkg_log)
    header_files = ["./lib/expat_external.h", "./lib/expat.h"]
    for header in header_files :
      self.call("./conftools/install-sh -c -m 644 %s \"%s\"" % (header,
        op.join(self.base_dir, "include")), log=pkg_log)

  def build_fontconfig(self):
    # fontconfig
    pkg_log = self.start_building_package("fontconfig")
    pkg = self.fetch_package(pkg_name=FONTCONFIG_PKG)
    if self.check_download_only(FONTCONFIG_PKG): return
    self.untar_and_chdir(pkg=pkg, log=pkg_log)
    # Create font directories.
    if (not op.isdir(op.join(self.base_dir, "share", "fonts"))):
      os.makedirs(op.join(self.base_dir, "share", "fonts"))
    if (not op.isdir(op.join(self.base_dir,"etc","fonts"))):
      os.makedirs(op.join(self.base_dir,"etc","fonts"))
    fc_config_args = [ self.prefix,
      "--disable-docs",
      "--with-expat-includes=\"%s\"" % op.join(self.base_dir, "include"),
      "--with-expat-lib=\"%s\"" % op.join(self.base_dir, "lib"),
      "--with-add-fonts=\"%s\"" % op.join(self.base_dir,"share","fonts"),
      "--with-confdir=\"%s\"" % op.join(self.base_dir,"etc","fonts"),
      "--with-docdir=\"%s\"" % op.join(self.base_dir, "doc"),
      "--with-freetype-config=freetype-config", ]
    self.configure_and_build(config_args=fc_config_args, log=pkg_log)

    # replace symbolic links (full paths) in base/etc/fonts/conf.d
    # with actual files to make installation portable
    link_directory = op.join(self.base_dir, 'etc', 'fonts', 'conf.d')
    link_files = os.listdir(link_directory)
    actual_directory = op.join(self.base_dir, 'share', 'fontconfig',
                               'conf.avail')
    actual_files = os.listdir(actual_directory)
    print('\n  Fixing symbolic links in %s' % link_directory, file=self.log)
    for link in link_files:
      if ('conf' in link):     # ignore README
        link_file = op.join(link_directory, link)
        actual_file = op.join(actual_directory, link)
        print('    ', link, file=self.log)
        self.call('rm -f %s' % link_file)
        if (op.isfile(actual_file)):
          self.call('cp %s %s' % (actual_file, link_file))

    # remove hard-coded cache directory from base/etc/fonts/fonts.conf
    print('\n  Removing hard-coded cache directory from fonts.conf', file=self.log)
    cache_directory = op.join(self.base_dir, 'var', 'cache', 'fontconfig')
    fonts_directory = op.join(self.base_dir, 'etc', 'fonts')
    old_conf = open(op.join(fonts_directory, 'fonts.conf'), 'r')
    new_conf = open(op.join(fonts_directory, 'new.conf'), 'w')
    for line in old_conf.readlines():
      if (cache_directory not in line):
        new_conf.write(line)
    old_conf.close()
    new_conf.close()
    old_conf = op.join(fonts_directory, 'fonts.conf')
    new_conf = op.join(fonts_directory, 'new.conf')
    self.call('mv %s %s' % (new_conf, old_conf))

  def build_render(self):
    # render, xrender, xft
    for pkg, name in zip([RENDER_PKG,XRENDER_PKG,XFT_PKG], ["render","Xrender","Xft"]):
      self.build_compiled_package_simple(pkg_name=pkg, pkg_name_label=name)

  def build_pixman(self):

    # set CFLAGS for CentOS 5 (32-bit)
    if ( (self.flag_is_linux) and (platform.architecture()[0] == '32bit') ):
      old_cflags = os.environ.get('CFLAGS', '')
      os.environ['CFLAGS'] = old_cflags + ' -g -O2'

    # pixman
    pkg_log = self.start_building_package("pixman")
    pkg = self.fetch_package(pkg_name=PIXMAN_PKG)
    if self.check_download_only(PIXMAN_PKG): return
    self.untar_and_chdir(pkg=pkg, log=pkg_log)
    self.configure_and_build(
      config_args=[self.prefix, "--disable-gtk", "--disable-static" ],
      log=pkg_log)

    # reset CFLAGS for CentOS 5 32-bit
    if ( (self.flag_is_linux) and (platform.architecture()[0] == '32bit') ):
      os.environ['CFLAGS'] = old_cflags

  def build_cairo(self):
    #    self.include_dirs.append(op.join(self.base_dir, "include", "harfbuzz"))

    # set CXXFLAGS for CentOS 5 (32-bit), needed for harfbuzz
    if ( (self.flag_is_linux) and (platform.architecture()[0] == '32bit') ):
      old_cflags = os.environ.get('CXXFLAGS', '')
      os.environ['CXXFLAGS'] = old_cflags + ' -march=i686'

    for pkg, name in zip([CAIRO_PKG, HARFBUZZ_PKG],["cairo", "harfbuzz"]):
      self.build_compiled_package_simple(pkg_name=pkg, pkg_name_label=name)
    self.build_compiled_package_simple(
      pkg_name=PANGO_PKG, pkg_name_label="pango",
      extra_config_args=["--enable-introspection=no"])
    self.build_compiled_package_simple(
      pkg_name=ATK_PKG, pkg_name_label="atk",
      extra_config_args=["--enable-introspection=no"])

    # reset CXXFLAGS for CentOS 5 32-bit
    if ( (self.flag_is_linux) and (platform.architecture()[0] == '32bit') ):
      os.environ['CXXFLAGS'] = old_cflags

  def build_tiff(self):
    # tiff
    pkg_log = self.start_building_package("tiff")
    pkg = self.fetch_package(pkg_name=TIFF_PKG)
    if self.check_download_only(TIFF_PKG): return
    self.untar_and_chdir(pkg=pkg, log=pkg_log)
    os.environ['MANSCHEME'] = "bsd-source-cat"
    os.environ['DIR_MAN'] = op.join(self.base_dir, "man")
    # disable external codecs
    config_args = [self.prefix, '--disable-pixarlog', '--disable-jpeg',
                   '--disable-old-jpeg', '--disable-jbig', '--disable-lzma' ]
    self.configure_and_build(
      config_args=config_args,
      log=pkg_log)

  def build_gtk(self):
    # gdk-pixbuf, gtk+, clearlooks
    extra_config_args = ["--without-libjpeg", "--enable-relocations",
                         "--enable-introspection=no"]
    self.build_compiled_package_simple(pkg_name=GDK_PIXBUF_PKG,
                                       pkg_name_label='gdk-pixbuf',
                                       extra_config_args=extra_config_args)
    self.build_compiled_package_simple(
      pkg_name=GTK_PKG, pkg_name_label='gtk+',
      extra_config_args=["--enable-introspection=no"])
    self.build_compiled_package_simple(
      pkg_name=GTK_ENGINE_PKG, pkg_name_label='gtk-engine')

  def build_fonts(self):
    # fonts
    fonts_log = self.start_building_package("fonts")
    pkg = self.fetch_package(pkg_name=FONT_PKG)
    if self.check_download_only("fonts"): return

    share_dir = op.join(self.base_dir, "share")
    if (not op.isdir(share_dir)):
      os.makedirs(share_dir)
    os.chdir(share_dir)
    untar(pkg, log=fonts_log, verbose=True)
    os.chdir(self.tmp_dir)

  def build_wxpython(self):
    if self.wxpython4 or self.python3:
      self.build_python_module_pip(
        'wxPython', package_version="4.0.3",
        confirm_import_module='wx')
      return

    pkg_log = self.start_building_package("wxPython")
    pkg_name = WXPYTHON_PKG
    pkg = self.fetch_package(pkg_name)
    if self.check_download_only(pkg_name): return

    pkg_dir = untar(pkg, log=pkg_log)
    os.chdir(pkg_dir)

    # Unconditionally append Debian i386/x86_64 multilib directories
    # to wxPython's list of library search paths.
    line = "SEARCH_LIB=\"`echo \"$SEARCH_INCLUDE\" | " \
           "sed s@include@$wx_cv_std_libpath@g` /usr/$wx_cv_std_libpath"
    self.patch_src(src_file="configure",
                   target=(line, ),
                   replace_with=(line +
                                 " /usr/lib/i386-linux-gnu" +
                                 " /usr/lib/x86_64-linux-gnu", ))

    if self.flag_is_mac and get_os_version().startswith('10.') and int(get_os_version().split('.')[1]) >= 10:
      # Workaround wxwidgets 3.0.2 compilation error on Yosemite
      # This will be fixed in 3.0.3.
      # See:
      #   http://trac.wxwidgets.org/ticket/16329
      #   http://goharsha.com/blog/compiling-wxwidgets-3-0-2-mac-os-x-yosemite/
      print("  patching src/osx/webview_webkit.mm", file=self.log)
      self.patch_src(src_file="src/osx/webview_webkit.mm",
                     target=("#include <WebKit/WebKit.h>",),
                     replace_with=("#include <WebKit/WebKitLegacy.h>",))

    if self.flag_is_mac and get_os_version().startswith('10.') and int(get_os_version().split('.')[1]) >= 12:
      # Workaround wxwidgets 3.0.2 compilation error on Sierra
      # QuickTime Framework deprecated in OS X v10.9
      # See:
      #   http://trac.wxwidgets.org/ticket/17639
      #   http://trac.wxwidgets.org/changeset/f6a2d1caef5c6d412c84aa900cb0d3990b350938/git-wxWidgets
      #   https://developer.apple.com/library/content/documentation/MacOSX/Conceptual/OSX_Technology_Overview/SystemFrameworks/SystemFrameworks.html
      for src_file in ("src/osx/core/bitmap.cpp", "src/osx/carbon/dataobj.cpp"):
        print("  patching %s" %src_file, file=self.log)
        self.patch_src(src_file=src_file,
                       target=("#include <QuickTime/QuickTime.h>",),
                       replace_with=("",))

    # Stage 1: build wxWidgets libraries
    config_opts = [
      self.prefix,
      "--with-opengl",
      "--enable-unicode",
      "--without-libjbig",
      "--without-liblzma",
      "--with-libjpeg=builtin", # Prevents system version, https://github.com/dials/dials/issues/523
    ]

    if (self.options.debug):
      config_opts.extend(["--disable-optimize",
                          "--enable-debug"])
      if (self.flag_is_linux):
        config_opts.append("--disable-debug_gdb")
    else :
      config_opts.extend(["--enable-optimize",
                          "--disable-debugreport"])

    # if (cocoa):
    if (self.flag_is_mac):
      config_opts.extend([
        "--with-osx_cocoa",
        "--with-macosx-version-min=%s" % self.min_macos_version,
        "--with-mac",
        "--enable-monolithic",
        "--disable-mediactrl"
      ])
      if get_os_version().startswith('10.') and int(get_os_version().split('.')[1]) >= 12:
        # See https://trac.wxwidgets.org/ticket/17929 fixed for wxWidgets 3.0.4
        # Does not affect 10.12, but using macro does not hurt
        # Also, -stdlib flag breaks things on Xcode 9, so leave it out.
        config_opts.append('CPPFLAGS="-D__ASSERT_MACROS_DEFINE_VERSIONS_WITHOUT_UNDERSCORES=1 %s"' % self.min_macos_version_flag)
        config_opts.append('LDFLAGS="%s"' % self.min_macos_version_flag)

    elif (self.flag_is_linux):
      config_opts.extend([
        "--with-gtk",
        "--with-gtk-prefix=\"%s\"" % self.base_dir,
        "--with-gtk-exec-prefix=\"%s\"" % op.join(self.base_dir, "lib"),
        "--enable-graphics_ctx",
        "--disable-mediactrl",
        "--enable-display",
        "--without-libnotify"
      ])

    install_gizmos = False #True
    print("  building wxWidgets with options:", file=self.log)
    for opt in config_opts :
      print("    %s" % opt, file=self.log)
    self.call("./configure %s" % " ".join(config_opts), log=pkg_log)
    self.call("make -j %d" % self.nproc, log=pkg_log)
    self.call("make install", log=pkg_log)

    # Stage 2: build wxPython itself
    wxpy_build_opts = [
      "BUILD_GLCANVAS=1",
      "BUILD_DLLWIDGET=0",
      "UNICODE=1"
    ]
    if install_gizmos:
      wxpy_build_opts.append("BUILD_GIZMOS=1")
    if self.flag_is_mac:
      os.environ['CFLAGS'] = os.environ.get('CFLAGS', '') + " -arch x86_64"
      wxpy_build_opts.extend(["BUILD_STC=1",
                              "WXPORT=osx_cocoa"])
      # Xcode 9 fails with -stdlib flag
      if get_os_version().startswith('10.') and int(get_os_version().split('.')[1]) >= 12:
        os.environ['CPPFLAGS'] = self.min_macos_version_flag
        os.environ['LDFLAGS'] = self.min_macos_version_flag
    else :
      wxpy_build_opts.extend(["BUILD_STC=1", #"BUILD_STC=0",
                              #"BUILD_OGL=0",
                              "WX_CONFIG=%s/bin/wx-config" %self.base_dir])
    self.chdir("wxPython", log=pkg_log)
    debug_flag = ""
    if (self.options.debug):
      debug_flag = "--debug"
    print("  building wxPython with options:", file=self.log)
    for opt in wxpy_build_opts :
      print("    %s" % opt, file=self.log)
    self.call("%s setup.py %s build_ext %s" % (self.python_exe,
      " ".join(wxpy_build_opts), debug_flag), log=pkg_log)
    self.call("%s setup.py %s install" % (self.python_exe,
      " ".join(wxpy_build_opts)), log=pkg_log)
    self.verify_python_module("wxPython", "wx")

  def build_matplotlib(self):
    def patch_matplotlib_src(out):
      print("  patching setup.cfg", file=out)
      self.patch_src(src_file="setup.cfg.template",
                     output_file="setup.cfg",
                     target=("#backend = Agg", "#basedirlist = /usr"),
                     replace_with=("backend = WXAgg",
                                   "basedirlist = /usr, %s" % self.base_dir))
      return True

    # delete old font cache
    #if ( (self.flag_is_linux) or (self.flag_is_mac) ):
    #  home = os.path.expanduser('~')
    #  filename = 'fontList.cache'
    #  directories = ['.matplotlib', '.cache/matplotlib']
    #  for directory in directories:
    #    font_cache = os.path.join(home, directory, filename)
    #    print font_cache, os.path.exists(font_cache)
    #    if (os.path.exists(font_cache)):
    #      os.remove(font_cache)

    for dependency in MATPLOTLIB_DEPS:
      self.build_python_module_pip(dependency[0], package_version=dependency[1])

    self.build_python_module_simple(
      pkg_url=BASE_CCI_PKG_URL,
      pkg_name=MATPLOTLIB_PKG,
      pkg_name_label="Matplotlib",
      callback_before_build=patch_matplotlib_src,
      confirm_import_module="matplotlib")

  def build_misc(self):
    if not self.python3: # This module will never be Python3 compatible
     self.build_python_module_simple(
      pkg_url=BASE_CCI_PKG_URL,
      pkg_name=PYRTF_PKG,
      pkg_name_label="PyRTF",
      confirm_import_module="PyRTF")
      # TODO we are patching the source to force it to use the correct backend.
      # I'm not sure if this is truly necessary or if there's a cleaner way...

    self.build_python_module_pip(
      "send2trash", package_version=SEND2TRASH_VERSION,
    )

  # TODO
  def write_dispatcher_include(self):
    raise NotImplementedError()

def check_wxpython_build_dependencies(log=sys.stderr):
  try :
    call(["pkg-config", "--version"], log=log)
  except RuntimeError :
    return """
ERROR The GUI components require pkg-config to build
Please install pkg-config to compile these components or use the --no-gui
option to disable compilation of GUI components.
"""
  # TODO pkg-config version check
  try :
    call(["bash", "--version"], log=log)
  except RuntimeError :
    return """
ERROR: The GUI requires bash to be available to build
Please install bash to compile these components, or use the --no-gui option
to disable GUI compilation.
"""
  if (not op.exists("/usr/include/X11/X.h") and
      not op.exists("/usr/X11R6/include/X11/X.h")):
    return """
ERROR: The X-windows headers appear to be missing
Please install the X11 development packages to compile the GUI components,
or use the --no-gui option to disable GUI compilation.
"""
  return None

if __name__ == "__main__":
  installer(args=sys.argv, log=sys.stdout)
