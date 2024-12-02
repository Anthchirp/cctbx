
"""
Listing of current dependencies for CCTBX and related applications (including
LABELIT, xia2, DIALS, and Phenix with GUI).  Not all of these can be downloaded
via the web (yet).
"""

from __future__ import absolute_import, division, print_function

import json
import os
import os.path as op
import sys
try: # Python 3
  from urllib.request import urlopen
except ImportError: # Python 2
  from urllib2 import urlopen

try:
  from .bootstrap import Toolbox
  from .installer_utils import *
except (ValueError, ImportError):
  # When run from bootstrap the auto_build directory will be in the path
  from bootstrap import Toolbox
  from installer_utils import *

BASE_CCI_PKG_URL = [
  "http://cci.lbl.gov/cctbx_dependencies",
  "https://gitcdn.link/repo/dials/dependencies/master",
  "https://github.com/dials/dependencies/raw/master",
]

def get_pypi_package_information(package, version=None, information_only=False):
  '''Retrieve information about a PyPi package.'''
  metadata = 'https://pypi.python.org/pypi/' + package + '/json'
  try:
    pypidata = urlopen(metadata).read()
  except Exception: # TLS <1.2, ...
    if information_only:
      return {'name': '', 'version': '', 'summary': ''}
    raise
  pkginfo = json.loads(pypidata)
  if information_only:
    return pkginfo['info']
  if not version:
    version = pkginfo['info']['version']
  if version not in pkginfo['releases']:
    raise RuntimeError("Could not find release '%s' for %s on pypi." % (version, package))
  candidates = filter(lambda c: c.get('python_version') == 'source' and c.get('packagetype') == 'sdist', pkginfo['releases'][version])
  if not candidates:
    raise RuntimeError("Could not find a source release file for %s %s on pypi." % (package, version))
  package = candidates[0]
  for field in ('name', 'version', 'summary'):
    package[field] = pkginfo['info'][field]
  return package

DEPENDENCIES_BASE = [
  "https://gitcdn.link/repo/dials/dependencies/master",
  "https://github.com/dials/dependencies/raw/master",
  "https://gitcdn.xyz/repo/dials/dependencies/master",
]
OPENSSL_PKG = "openssl-1.0.2s.tar.gz"    # OpenSSL
PYTHON3_PKG = "Python-3.7.2.tgz"
PYTHON_PKG = "Python-2.7.18.tgz"

# from CCI
IMAGING_PKG = "Imaging-1.1.7.tar.gz"     # for labelit, gltbx
REPORTLAB_PKG = "reportlab-3.5.12.tar.gz"   # for labelit
ZLIB_PKG = "zlib-1.2.11.tar.gz"
PYRTF_PKG = "PyRTF-0.45.tar.gz"          # for phenix.table_one, etc.
BIOPYTHON_PKG = "biopython-1.73.tar.gz"  # used in iotbx
IPYTHON_PKG = "ipython-5.8.0.tar.gz"     # IPython
LIBSVM_PKG = "libsvm-3.17_cci.tar.gz"

# from PyPi
CYTHON_VERSION = "0.28.6"
DOCUTILS_VERSION = "0.14"
FUTURE_VERSION = "0.17.1"
H5PY_VERSION = "2.10.0"
JINJA2_VERSION = "2.10"
MOCK_VERSION = "3.0.5"
MPI4PY_VERSION = "3.0.0"
MRCFILE_VERSION = "1.1.2"
MSGPACK_VERSION = "0.6.1"
NUMPY_VERSION="1.15.4"
ORDEREDSET_VERSION = "2.0.1"
PILLOW_VERSION = "5.4.1"
PY2APP_VERSION="0.7.3"
PYTEST_VERSION = "4.6.5"
PYTEST_XDIST_VERSION = "1.29.0"
SCIKIT_LEARN_VERSION = "0.20.2"
SCIPY_VERSION = "1.2.1"
SEND2TRASH_VERSION = "1.5.0"
SIX_VERSION = "1.12.0"
SPHINX_VERSION = "1.8.4" # for documentation
TABULATE_VERSION = "0.8.3"
TQDM_VERSION = "4.23.4"
PSUTIL_VERSION = "5.5.1"

# HDF5
BASE_HDF5_PKG_URL = "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src"
HDF5_PKG = "hdf5-1.10.5.tar.bz2"

# GUI dependencies
LIBPNG_PKG = "libpng-1.6.36.tar.gz"
FREETYPE_PKG = "freetype-2.6.3.tar.gz"

# Linux-only
# libraries based on X11R7.7 (http://www.x.org/releases/X11R7.7/src/everything/)
WXPYTHON_PKG = "wxPython-src-3.0.2.0.tar.bz2"
GETTEXT_PKG = "gettext-0.19.7.tar.gz"
LIBFFI_PKG = "libffi-3.2.1.tar.gz"
GLIB_PKG = "glib-2.46.2.tar.gz"
EXPAT_PKG = "expat-2.1.0.tar.gz"
FONTCONFIG_PKG = "fontconfig-2.11.1.tar.gz"
RENDER_PKG = "renderproto-0.11.1.tar.gz"
XRENDER_PKG = "libXrender-0.9.7.tar.gz"
XFT_PKG = "libXft-2.3.2.tar.gz"

CAIRO_PKG = "cairo-1.14.4.tar.gz"
PIXMAN_PKG = "pixman-0.34.0.tar.gz"
HARFBUZZ_PKG = "harfbuzz-1.1.3.tar.gz"
GDK_PIXBUF_PKG = "gdk-pixbuf-2.32.3.tar.gz"
PANGO_PKG = "pango-1.38.1.tar.gz"
ATK_PKG = "atk-2.18.0.tar.gz"
TIFF_PKG = "tiff-4.0.10.tar.gz"
GTK_PKG = "gtk+-2.24.29.tar.gz"
GTK_ENGINE_PKG = "clearlooks-0.6.2.tar.gz"
GTK_THEME_PKG = "gtk_themes.tar.gz"
# end Linux-only
FONT_PKG = "fonts.tar.gz"

MATPLOTLIB_PKG = "matplotlib-2.2.3.tar.gz"
MATPLOTLIB_DEPS = [
  ("subprocess32", "3.5.4"),
  ("backports.functools-lru-cache", "1.6.1"),
  ("kiwisolver", "1.1.0"),
]

PYOPENGL_PKG = "PyOpenGL-3.1.0.tar.gz"

# Windows precompiled compiled base packages
WIN64PYTHON_PKG = "Python2.7.15_x86_64_plus_relocatable.zip"
WIN32PYTHON_PKG = "python2.7.12_x86_32_plus_relocatable.zip"
WIN64HDF5_PKG = "HDF5-1.8.16-win64.zip"
WIN32HDF5_PKG = "HDF5-1.8.16-win32.zip"
VCREDIST64 = "vcredist_x64.exe"
VCREDIST32 = "vcredist_x86.exe"
WINLIBTIFF64 = "libtiff4.0.6x64.zip"
WINLIBTIFF32 = "libtiff4.0.6x32.zip"

# Various dependencies from external repositories, distributed as static
# tarballs (since they are not under active development by us or our
# collaborators)
dependency_tarballs = {
  "scons":  ("http://cci.lbl.gov/hot", "scons_hot.tar.gz"),
}
# External SVN repositories that may be required for certain components of
# CCTBX to work.  This includes forked versions (with minimal changes) of the
# core CCP4 libraries, MUSCLE, and ksDSSP.
subversion_repositories = {
  "ccp4io": "http://cci.lbl.gov/svn/ccp4io/trunk",
  "ccp4io_adaptbx": "http://cci.lbl.gov/svn/ccp4io_adaptbx/trunk",
  "gui_resources": "http://cci.lbl.gov/svn/gui_resources/trunk",
  "tntbx": "http://cci.lbl.gov/svn/tntbx/trunk",
  "ksdssp": "http://cci.lbl.gov/svn/ksdssp/trunk",
  "muscle": "http://cci.lbl.gov/svn/muscle/trunk",
  # adding for amber
  "amber_adaptbx": "http://cci.lbl.gov/svn/amber_adaptbx/trunk",
}

# External GIT repositories that may be required for certain components of
# CCTBX to work. Note that the format for git repositories can be more
# sophisticated than that for SVN, and can include multiple possible sources,
# including .zip archives to fall back on when git is not available, and git
# command line parameters
git_repositories = {
  # lz4 and bitshuffle compressions for HDF5
  # The git repositories are disabled for both on purpose.
  # Reason: Repositories are deprecated and unlikely to change ever again.
  #         We currently patch the local copies of the repositories, which means
  #         that if they are cloned git repositories you cannot run bootstrap on
  #         them again without bootstrap detecting uncommitted changes and
  #         stopping with an error message.
  "hdf5_lz4": [#'git@github.com:dectris/HDF5Plugin.git',
               #'https://github.com/dectris/HDF5Plugin.git',
               'https://github.com/dectris/HDF5Plugin/archive/master.zip'],
  "bitshuffle": [#'git@github.com:kiyo-masui/bitshuffle.git',
                 #'https://github.com/kiyo-masui/bitshuffle.git',
                 'https://github.com/kiyo-masui/bitshuffle/archive/master.zip'],
}

class fetch_packages(object):
  """
  Download manager for the packages defined by this module - this is used by
  install_base_packages.py but also for setting up installer bundles.
  """
  def __init__(self, dest_dir, log, pkg_dirs=None, no_download=False,
      copy_files=False):
    self.dest_dir = dest_dir
    self.log = log
    self.pkg_dirs = pkg_dirs
    self.no_download = no_download
    self.copy_files = copy_files
    self.toolbox = Toolbox()

  def __call__(self,
                pkg_name,
                pkg_url=None,
                output_file=None,
                return_file_and_status=False,
                download_url=None, # If given this is the URL used for downloading, otherwise construct using pkg_url and pkg_name
                ):
    if (pkg_url is None):
      pkg_url = BASE_CCI_PKG_URL
    if (output_file is None):
      output_file = pkg_name
    os.chdir(self.dest_dir)
    print("  getting package %s..." % pkg_name, file=self.log)
    if (self.pkg_dirs is not None) and (len(self.pkg_dirs) > 0):
      for pkg_dir in self.pkg_dirs :
        static_file = op.join(pkg_dir, pkg_name)
        if (op.exists(static_file)):
          print("    using %s" % static_file, file=self.log)
          if self.copy_files :
            copy_file(static_file, op.join(self.dest_dir, output_file))
            if return_file_and_status:
              return op.join(self.dest_dir, output_file), 0
            return op.join(self.dest_dir, output_file)
          else :
            if return_file_and_status:
              return static_file, 0
            return static_file
    if (self.no_download):
      if (op.exists(pkg_name)):
        print("    using ./%s" % pkg_name, file=self.log)
        if return_file_and_status:
          return op.join(self.dest_dir, output_file), 0
        return op.join(self.dest_dir, pkg_name)
      else :
        raise RuntimeError(("Package '%s' not found on local filesystems.  ") %
          pkg_name)

    # Generate list of possible URL candidates
    if download_url:
      if isinstance(download_url, list):
        urls = download_url
      else:
        urls = [download_url]
    else:
      if isinstance(pkg_url, list):
        urls = ["%s/%s" % (p, pkg_name) for p in pkg_url]
      else:
        urls = ["%s/%s" % (pkg_url, pkg_name)]

    for url_attempt in urls:
      self.log.write("    downloading from %s : " % url_attempt)
      for retry in (3,3,0):
        try:
          size = self.toolbox.download_to_file(url_attempt, output_file, log=self.log)
          if (size == -2):
            print("    using ./%s (cached)" % pkg_name, file=self.log)
            if return_file_and_status:
              return op.join(self.dest_dir, output_file), size
            return op.join(self.dest_dir, output_file)
          assert size > 0, "File %s has size %d" % (pkg_name, size)
          if return_file_and_status:
            return op.join(self.dest_dir, output_file), size
          return op.join(self.dest_dir, output_file)
        except Exception as e:
          self.log.write("    download failed with %s" % str(e))
          if retry:
            self.log.write("    retrying in %d seconds" % retry)
            time.sleep(retry)
    raise RuntimeError("Could not download " + pkg_name)

def fetch_all_dependencies(dest_dir,
    log,
    pkg_dirs=None,
    copy_files=True,
    gui_packages=True,
    dials_packages=True):
  """
  Download or copy all dependencies into a local directory (prepping for
  source installer bundling).
  """
  fetch_package = fetch_packages(
    dest_dir=dest_dir,
    log=log,
    pkg_dirs=pkg_dirs,
    copy_files=copy_files)
  for pkg_name in [
      PYTHON_PKG, IMAGING_PKG, REPORTLAB_PKG, ZLIB_PKG,
      PYRTF_PKG, BIOPYTHON_PKG,
      IPYTHON_PKG,
    ] :
    fetch_package(pkg_name)
  if (gui_packages):
    for pkg_name in [
        LIBPNG_PKG, FREETYPE_PKG, GETTEXT_PKG, GLIB_PKG, EXPAT_PKG,
        FONTCONFIG_PKG, RENDER_PKG, XRENDER_PKG, XFT_PKG, PIXMAN_PKG,
        CAIRO_PKG, HARFBUZZ_PKG, PANGO_PKG, ATK_PKG, TIFF_PKG, GTK_PKG,
        GTK_ENGINE_PKG, GTK_THEME_PKG, FONT_PKG, WXPYTHON_PKG,
        MATPLOTLIB_PKG, SEND2TRASH_PKG,
      ] :
      fetch_package(pkg_name)

def fetch_svn_repository(pkg_name, pkg_url=None, working_copy=True,
    delete_if_present=False):
  """
  Download an SVN repository, with or without metadata required for ongoing
  development.
  """
  ## TODO: Merge this with _add_svn in bootstrap.py.
  #        Unnecessary code duplication
  if op.exists(pkg_name):
    if delete_if_present :
      shutil.rmtree(pkg_name)
    else :
      raise OSError("Directory '%s' already exists.")
  if (pkg_url is None):
    pkg_url = optional_repositories[pkg_name]
  if working_copy :
    call("svn co --non-interactive --trust-server-cert %s %s" % (pkg_url, pkg_name), sys.stdout)
  else :
    call("svn export --non-interactive --trust-server-cert %s %s" % (pkg_url, pkg_name), sys.stdout)
  assert op.isdir(pkg_name)

def fetch_git_repository(package, use_ssh):
  """ Download a git repository """
  Toolbox.git(package, git_repositories[package], destination=os.path.join(os.getcwd(), package), use_ssh=use_ssh, verbose=True)
  assert op.isdir(package)

def fetch_remote_package(module_name, log=sys.stdout, working_copy=False, use_ssh=False):
  if (module_name in git_repositories):
    fetch_git_repository(module_name, use_ssh)
  elif (module_name in dependency_tarballs):
    if op.isdir(module_name):
      shutil.rmtree(module_name)
    pkg_url, pkg_name = dependency_tarballs[module_name]
    tarfile = module_name + ".tar.gz"
    fetch_packages(
      dest_dir=os.getcwd(),
      log=log).__call__(
        pkg_name=pkg_name,
        pkg_url=pkg_url,
        output_file=tarfile)
    untar(tarfile, log)
    os.remove(tarfile)
  elif (module_name in subversion_repositories):
    if op.isdir(module_name):
      shutil.rmtree(module_name)
    pkg_url = subversion_repositories[module_name]
    fetch_svn_repository(
      pkg_name=module_name,
      pkg_url=pkg_url,
      working_copy=working_copy)
