from __future__ import absolute_import, division, print_function

from collections import namedtuple
import glob
import os
from six.moves import UserDict

import numpy as np
from ordered_set import OrderedSet


class StemLocator(UserDict):
  """Subclass of dict which raises an error when overwriting existing value"""

  class StemExistsError(FileExistsError):
    MSG = 'Multiple files with stem "{}" detected. Matching expts/refls ' \
          'must EITHER be pairwise located in the same directory ' \
          'OR have unique names across all input directories.'

    @classmethod
    def from_stem(cls, stem, *args, **kwargs):
      return cls(cls.MSG.format(stem), *args, **kwargs)  # noqa

  def __setitem__(self, key, value):
    if key in self.data:
      if value != self.data.__getitem__(key):
        raise self.StemExistsError.from_stem(key)
    else:
      self.data.__setitem__(key, value)


class PathPair(namedtuple('PathPair', ['expt_path', 'refl_path'])):
  """Named 2-el. tuple of expt & refl file paths with convenience methods"""
  __slots__ = ()

  expt_suffix = ''
  refl_suffix = ''

  @classmethod
  def from_dir_expt_name(cls, base_path, expt_filename):
    expt_path = os.path.join(base_path, expt_filename)
    refl_path = expt_path.split(cls.expt_suffix, 1)[0] + cls.refl_suffix
    refl_path = refl_path if os.path.exists(refl_path) else None
    return cls(expt_path, refl_path)

  @classmethod
  def from_dir_refl_name(cls, base_path, refl_filename):
    refl_path = os.path.join(base_path, refl_filename)
    expt_path = refl_path.split(cls.refl_suffix, 1)[0] + cls.expt_suffix
    expt_path = expt_path if os.path.exists(expt_path) else None
    return cls(expt_path, refl_path)

  @property
  def expt_stem(self):
    return os.path.basename(self.expt_path).split(self.expt_suffix, 1)[0]

  @property
  def refl_stem(self):
    return os.path.basename(self.refl_path).split(self.refl_suffix, 1)[0]


def list_input_pairs(params):
  """Create a list of expt/refl `PathPair`s on rank 0 based on `params`."""

  # Load alist of files/directories to be exclusively kept or specifically
  # rejected and define all associated alist functions
  alist = set()
  alist_files = params.input.alist.file
  if alist_files is not None:
    for f in alist_files:
      lines = np.loadtxt(f, str, ndmin=1)
      alist = alist.union(lines)
    if params.input.alist.type == "files":
      assert all(os.path.isabs(f) for f in alist)

  def is_on_alist(expt_path):
    if params.input.alist.type == "tags":
      return any(tag in expt_path for tag in alist)
    else:  # alist.type == "files"
      return os.path.abspath(expt_path) in alist

  if len(alist) == 0:
    def is_accepted_expt(*_): return True
  elif params.input.alist.op == "reject":
    def is_accepted_expt(expt_path): return not is_on_alist(expt_path)
  else:  # if params.input.alist.op == "keep"
    def is_accepted_expt(expt_path): return is_on_alist(expt_path)

  # Load paths to the input experiments and reflections. All expt/refl file
  # pairs must have the same filename stem and EITHER be in the same directory
  # OR have unique filename stems across input; otherwise raise StemExistsError
  path_pairs = []
  PathPair.expt_suffix = params.input.experiments_suffix
  PathPair.refl_suffix = params.input.reflections_suffix

  def load_path_if_expt_or_refl(path_, filename_):
    if filename.endswith(params.input.experiments_suffix):
      path_pairs.append(PathPair.from_dir_expt_name(path_, filename_))
    if filename.endswith(params.input.reflections_suffix):
      path_pairs.append(PathPair.from_dir_refl_name(path_, filename_))

  for pathstring in params.input.path:
    for path in glob.glob(pathstring):
      if os.path.isdir(path):
        for filename in os.listdir(path):
          load_path_if_expt_or_refl(path, filename)
      else:
        dir_name, filename = os.path.split(path)
        load_path_if_expt_or_refl(dir_name, filename)
  accepted_pairs = [pp for pp in OrderedSet(path_pairs)
                    if is_accepted_expt(pp.expt_path) or not pp.expt_path]

  # Merge every matching pair of PathPair(expt, None) + PathPair(None, refl)
  # in path_pairs with common stub name into a single PathPair(expt, refl)
  matched_pairs, expt_singlets, refl_singlets = [], StemLocator(), StemLocator()
  for path_pair in accepted_pairs:
    if path_pair.expt_path and path_pair.refl_path:
      matched_pairs.append(path_pair)
    elif path_pair.expt_path and not path_pair.refl_path:
      expt_singlets[path_pair.expt_stem] = path_pair.expt_path
    elif path_pair.refl_path and not path_pair.expt_path:
      refl_singlets[path_pair.refl_stem] = path_pair.refl_path
  common_stems = OrderedSet(expt_singlets).intersection(OrderedSet(refl_singlets))
  new = [PathPair(expt_singlets[c], refl_singlets[c]) for c in common_stems]
  return matched_pairs + new
