from __future__ import absolute_import, division, print_function
from scitbx.array_family import flex
from scitbx.python_utils import command_line
from scitbx import matrix
from libtbx.test_utils import Exception_expected, approx_equal, \
  not_approx_equal, show_diff
import libtbx.math_utils
from six.moves import cStringIO as StringIO
from libtbx.math_utils import cmp
from six.moves import range
from six.moves import cPickle as pickle
import six
import math
import random
import time
import sys, os

# Boost 1.56 changes normal distribution, but keep old tests for testing
# older versions of Boost
import libtbx.load_env
from six.moves import zip
boost_version = libtbx.env.boost_version

def exercise_flex_grid():
  g = flex.grid()
  assert g.nd() == 0
  assert g.size_1d() == 0
  assert g.is_0_based()
  assert g.origin() == ()
  assert g.all() == ()
  assert g.last() == ()
  assert g.last(True) == ()
  assert g.last(False) == ()
  assert not g.is_padded()
  assert not g.is_trivial_1d()
  g = flex.grid((2,3,5))
  assert g.nd() == 3
  assert g.size_1d() == 30
  assert g.origin() == (0,0,0)
  assert g.all() == (2,3,5)
  assert g.last() == (2,3,5)
  assert g.last(True) == (2,3,5)
  assert g.last(False) == (1,2,4)
  assert g((0,0,0)) == 0
  assert g((1,2,4)) == 29
  assert g.is_0_based()
  assert not g.is_padded()
  assert not g.is_trivial_1d()
  assert flex.grid(1).all() == (1,)
  assert flex.grid(1,2).all() == (1,2,)
  assert flex.grid(1,2,3).all() == (1,2,3)
  assert flex.grid(1,2,3,4).all() == (1,2,3,4)
  assert flex.grid(1,2,3,4,5).all() == (1,2,3,4,5)
  assert flex.grid(1,2,3,4,5,6).all() == (1,2,3,4,5,6)
  assert flex.grid(1).set_focus(1).focus() == (1,)
  assert flex.grid(1,2).set_focus(1,2).focus() == (1,2,)
  assert flex.grid(1,2,3).set_focus(1,2,3).focus() == (1,2,3)
  assert flex.grid(1,2,3,4).set_focus(1,2,3,4).focus() == (1,2,3,4)
  assert flex.grid(1,2,3,4,5).set_focus(1,2,3,4,5).focus() == (1,2,3,4,5)
  assert flex.grid(1,2,3,4,5,6).set_focus(1,2,3,4,5,6).focus() == (1,2,3,4,5,6)
  g = flex.grid((1,2,3), (4,6,8))
  assert g.nd() == 3
  assert g.size_1d() == 60
  assert not g.is_0_based()
  assert g.origin() == (1,2,3)
  assert g.all() == (3,4,5)
  assert g.last() == (4,6,8)
  assert g.last(True) == (4,6,8)
  assert g.last(False) == (3,5,7)
  assert g((1,2,3)) == 0
  assert g((3,5,7)) == 59
  assert not g.is_valid_index((0,0,0))
  assert not g.is_padded()
  assert not g.is_trivial_1d()
  g = flex.grid((1,2,3), (4,6,8), False)
  assert g.nd() == 3
  assert g.size_1d() == 120
  assert g.origin() == (1,2,3)
  assert g.all() == (4,5,6)
  assert g.last() == (5,7,9)
  assert g.last(True) == (5,7,9)
  assert g.last(False) == (4,6,8)
  assert g((1,2,3)) == 0
  assert g((4,6,8)) == 119
  assert not g.is_valid_index((0,0,0))
  assert not g.is_valid_index((5,0,0))
  assert not g.is_0_based()
  assert not g.is_padded()
  assert not g.is_trivial_1d()
  g.set_focus((3,-9,5))
  assert g.is_padded()
  assert g.focus() == (3,-9,5)
  assert g.focus(False) == (2,-10,4)
  g.set_focus((3,-9,5), False)
  assert g.focus() == (4,-8,6)
  assert not g.is_0_based()
  assert g.is_padded()
  assert not g.is_trivial_1d()
  s = pickle.dumps(g)
  l = pickle.loads(s)
  assert g.origin() == l.origin()
  assert g.all() == l.all()
  assert g.focus() == l.focus()
  assert g == l
  assert not g != l
  l = flex.grid((1,2,4), (4,6,8), False).set_focus((3,-9,5))
  assert not g == l
  assert g != l
  l = flex.grid((1,2,3), (4,7,8), False).set_focus((3,-9,5))
  assert not g == l
  assert g != l
  l = flex.grid((1,2,3), (4,6,8), False).set_focus((4,-9,5))
  assert not g == l
  assert g != l
  g = flex.grid(1,2)
  h = flex.grid((0,0),(1,2))
  assert g == g
  assert h == h
  assert g == h
  assert h == g
  h.set_focus((1,2))
  assert not h.is_padded()
  assert g == g
  assert h == h
  assert g == h
  assert h == g
  h.set_focus((1,1))
  assert h.is_padded()
  assert h == h
  assert g != h
  assert h != g
  assert flex.grid(2).set_focus(1).is_padded()
  assert not flex.grid(1).set_focus(1).is_padded()
  assert flex.grid(1) == flex.grid(1).set_focus(1)
  assert flex.grid(1,2).set_focus(1,1).is_padded()
  assert not flex.grid(1,2).set_focus(1,2).is_padded()
  assert flex.grid(1,2) == flex.grid(1,2).set_focus(1,2)
  assert flex.grid(1,2,3).set_focus(1,2,2).is_padded()
  assert not flex.grid(1,2,3).set_focus(1,2,3).is_padded()
  assert flex.grid(1,2,3) == flex.grid(1,2,3).set_focus(1,2,3)
  assert flex.grid(1,2,3,4).set_focus(1,2,3,3).is_padded()
  assert not flex.grid(1,2,3,4).set_focus(1,2,3,4).is_padded()
  assert flex.grid(1,2,3,4) == flex.grid(1,2,3,4).set_focus(1,2,3,4)
  assert flex.grid(1,2,3,4,5).set_focus(1,2,3,4,4).is_padded()
  assert not flex.grid(1,2,3,4,5).set_focus(1,2,3,4,5).is_padded()
  assert flex.grid(1,2,3,4,5) == flex.grid(1,2,3,4,5).set_focus(1,2,3,4,5)
  g = flex.grid((1,2,3))
  assert g.shift_origin() == g
  g = flex.grid((1,2,3), (4,6,8))
  s = g.shift_origin()
  assert s.origin() == (0,0,0)
  assert s.all() == g.all()
  assert s.focus() == (3,4,5)
  assert s.focus_size_1d() == g.size_1d()
  g = flex.grid((1,2,3), (4,6,8)).set_focus((3,5,7))
  assert g.focus_size_1d() == 2*3*4
  s = g.shift_origin()
  assert s.origin() == (0,0,0)
  assert s.all() == g.all()
  assert s.focus() == (2,3,4)

def exercise_flex_constructors():
  f = flex.double()
  assert f.size() == 0
  assert f.capacity() == 0
  assert f.accessor().nd() == 1
  assert tuple(f.accessor().origin()) == (0,)
  assert tuple(f.accessor().all()) == (0,)
  assert tuple(f.accessor().last()) == (0,)
  assert tuple(f.accessor().last(True)) == (0,)
  assert tuple(f.accessor().last(False)) == (-1,)
  assert tuple(f.accessor().focus()) == (0,)
  assert tuple(f.accessor().focus(True)) == (0,)
  assert tuple(f.accessor().focus(False)) == (-1,)
  assert f.accessor().is_0_based()
  assert not f.accessor().is_padded()
  assert f.accessor().is_trivial_1d()
  assert f.nd() == 1
  assert f.is_0_based()
  assert tuple(f.origin()) == (0,)
  assert tuple(f.all()) == (0,)
  assert tuple(f.last()) == (0,)
  assert tuple(f.last(True)) == (0,)
  assert tuple(f.last(False)) == (-1,)
  assert not f.is_padded()
  assert tuple(f.focus()) == (0,)
  assert tuple(f.focus(True)) == (0,)
  assert tuple(f.focus(False)) == (-1,)
  assert f.focus_size_1d() == 0
  assert f.is_trivial_1d()
  assert tuple(f) == ()
  f = flex.double(flex.grid((2,3)))
  assert not f.is_square_matrix()
  f = flex.double(flex.grid((4,4)))
  assert f.is_square_matrix()
  f = flex.double(flex.grid((2,3,5)))
  assert f.size() == 30
  assert f.capacity() == 30
  assert list(f) == [0] * 30
  f = flex.double(flex.grid((2,3,5)), 42)
  assert f.size() == 30
  assert list(f) == [42] * 30
  f = flex.double(2)
  assert f.size() == 2
  assert tuple(f) == (0,0)
  f = flex.double(1, 42)
  assert f.size() == 1
  assert tuple(f) == (42,)
  f = flex.double((1,2,3,4,5))
  assert f.size() == 5
  assert tuple(f) == (1,2,3,4,5)
  f = flex.double([2,1,3,5])
  assert f.size() == 4
  assert tuple(f) == (2,1,3,5)
  f = flex.double(range(10,13))
  assert f.size() == 3
  assert tuple(f) == (10,11,12)
  assert flex.to_list(f) == [10,11,12]
  #
  for row_type in [list, tuple]:
    for column_type in [list, tuple]:
      for n_rows in range(4):
        for n_columns in range(4):
          matrix = list()
          for i_row in range(n_rows):
            matrix.append(column_type(flex.random_double(n_columns)))
          m = flex.double(row_type(matrix))
          if (n_rows == 0):
            assert m.focus() == (0,)
          else:
            assert m.focus() == (n_rows, n_columns)
          try: flex.double(row_type([column_type([]),column_type([1])]))
          except RuntimeError as e:
            assert str(e) == "matrix columns must have identical sizes."
          else: raise Exception_expected
          try: flex.double(row_type([column_type([0]),column_type(["x"])]))
          except TypeError as e:
            assert str(e) in [
              "bad argument type for built-in operation",
              "a float is required",
              "must be real number, not str"]
          else: raise Exception_expected
  for arg in [[[0],""], ([0],"",)]:
    try: flex.double(arg)
    except RuntimeError as e:
      assert str(e) == \
        "argument must be a Python list or tuple of lists or tuples."
    else: raise Exception_expected
  #
  for range_fn in [flex.int_range,
                   flex.long_range,
                   flex.int8_range,
                   flex.int16_range,
                   flex.int32_range,
                   flex.int64_range]:
    assert list(range_fn(stop=3)) == list(range(3))
    assert list(range_fn(start=1, stop=3)) == list(range(1, 3))
    assert list(range_fn(start=1, stop=5, step=2)) == list(range(1, 5, 2))
  for start in range(-5,6):
    for stop in range(-5,6):
      for step in range(-5,6):
        if (step == 0): continue
        args = (start,stop,step)
        assert list(flex.int_range(*args)) == list(range(*args))
        assert list(flex.long_range(*args)) == list(range(*args))
        assert list(flex.int8_range(*args)) == list(range(*args))
        assert list(flex.int16_range(*args)) == list(range(*args))
        assert list(flex.int32_range(*args)) == list(range(*args))
        assert list(flex.int64_range(*args)) == list(range(*args))
        assert approx_equal(flex.double_range(*args), list(range(*args)))
        assert approx_equal(flex.float_range(*args), list(range(*args)))
  for range_fn in [flex.size_t_range,
                   flex.uint8_range,
                   flex.uint16_range,
                   flex.uint32_range,
                   flex.uint64_range]:
    assert list(range_fn(stop=3)) == list(range(3))
    assert list(range_fn(start=8, stop=3, step=-1)) == list(range(8, 3, -1))
    try: range_fn(0, 0, 0)
    except RuntimeError as e:
      assert str(e) == "range step argument must not be zero."
    else: raise Exception_expected
    try: range_fn(-1, 0, 1)
    except RuntimeError as e:
      assert str(e) == "range start argument must not be negative."
    else: raise Exception_expected
    try: range_fn(0, -1, 1)
    except RuntimeError as e:
      assert str(e) == "range stop argument must not be negative."
    else: raise Exception_expected

def exercise_numbers_from_string():
  # skips flex.int8
  for flex_type in [flex.int, flex.long, flex.int16, flex.int32, flex.int64]:
    i = flex_type(flex.std_string(('1','+2','-3')))
    assert tuple(i.as_string()) == ('1', '2', '-3')
    assert tuple(i.as_string("%+3d")) == (' +1', ' +2', ' -3')
    assert tuple(i) == (1,2,-3)
    #
  for flex_type in [flex.int, flex.long, flex.int8, flex.int16, flex.int32, flex.int64]:
    try:
      flex_type(flex.std_string(['']))
    except ValueError as e:
      assert not show_diff(str(e),
        'Empty string (integer value expected).')
    else:
      raise Exception_expected
    try:
      flex_type(flex.std_string(['+-0']))
    except ValueError as e:
      assert not show_diff(str(e),
        'Invalid integer value: "+-0"')
    else:
      raise Exception_expected
  for flex_type in [flex.int, flex.long, flex.int8, flex.int16, flex.int32, flex.int64]:
    s = str(2**1000).replace("L", "")
    try:
      flex_type(flex.std_string([s]))
    except ValueError as e:
      assert not show_diff(str(e),
        'Invalid integer value: "%s"' % s)
    else:
      raise Exception_expected
  #
  f = flex.double(flex.std_string(['1.2','+2e-3','3']))
  assert approx_equal(f, (1.2,.002,3.0))
  f = flex.double(flex.std_string(['1.2(3)','+2e-3(1)','3(16)']))
  assert approx_equal(f, (1.2,.002,3.0))
  assert tuple(f.as_string()) == ('1.2', '0.002', '3')
  s = ('0.7', '0.1')
  f = flex.double(flex.std_string(s))
  assert tuple(f.as_string()) == s
  f = flex.double([0.1, -0.0000234])
  assert tuple(f.as_string("%+.3e")) in [
    ('+1.000e-001', '-2.340e-005'),
    ('+1.000e-01',  '-2.340e-05')]
  f = flex.double(flex.std_string([".34+05", "+6-2", "-7+4(1)"]))
  assert approx_equal(f, [34000.0, 0.06, -70000.0])
  #
  try:
    flex.double(flex.std_string(['']))
  except ValueError as e:
    assert not show_diff(str(e),
      'Empty string (floating-point value expected).')
  else:
    raise Exception_expected
  try:
    flex.double(flex.std_string(['1.2(']))
  except ValueError as e:
    assert not show_diff(str(e),
      'Missing closing parenthesis: "1.2("')
  else:
    raise Exception_expected
  try:
    flex.double(flex.std_string(['1.2(3)4']))
  except ValueError as e:
    assert not show_diff(str(e),
      'Unexpected trailing characters after ")": "1.2(3)4"')
  else:
    raise Exception_expected
  try:
    flex.double(flex.std_string(['(3)']))
  except ValueError as e:
    assert not show_diff(str(e),
      'Empty value part: "(3)"')
  else:
    raise Exception_expected
  try:
    flex.double(flex.std_string(['5()']))
  except ValueError as e:
    assert not show_diff(str(e),
      'Empty esd part: "5()"')
  else:
    raise Exception_expected
  try:
    flex.double(flex.std_string(['6x']))
  except ValueError as e:
    assert not show_diff(str(e),
      'Invalid floating-point value: "6x"')
  else:
    raise Exception_expected
  try:
    flex.double(flex.std_string(['7x(8)']))
  except ValueError as e:
    assert not show_diff(str(e),
      'Invalid value part: "7x(8)"')
  else:
    raise Exception_expected
  try:
    flex.double(flex.std_string(['1.2','+2e-3(x)']))
  except ValueError as e:
    assert not show_diff(str(e), 'Invalid esd part: "+2e-3(x)"')
  else:
    raise Exception_expected

def exercise_std_string():
  fss = flex.std_string
  a = fss([" Abc", "dEF", "ghi ", "   JKL ", "1 23 "])
  assert a.max_element_length() == 7
  assert a.strip().max_element_length() == 4
  assert list(a.strip()) == ['Abc', 'dEF', 'ghi', 'JKL', '1 23']
  assert list(a.strip().upper()) == ["ABC","DEF","GHI","JKL", "1 23"]
  assert list(a.strip().lower()) == ["abc","def","ghi","jkl", "1 23"]
  #
  a = fss()
  assert len(a.i_seqs_by_value()) == 0
  ibv = fss([""]).i_seqs_by_value()
  assert list(ibv.keys()) == [""]
  assert list(ibv[""]) == [0]
  ibv = fss(["", ""]).i_seqs_by_value()
  assert list(ibv.keys()) == [""]
  assert list(ibv[""]) == [0,1]
  ibv = fss(["", "a"]).i_seqs_by_value()
  assert sorted(ibv.keys()) == ["", "a"]
  assert list(ibv[""]) == [0]
  assert list(ibv["a"]) == [1]
  ibv = fss(["", "a", ""]).i_seqs_by_value()
  assert sorted(ibv.keys()) == ["", "a"]
  assert list(ibv[""]) == [0,2]
  assert list(ibv["a"]) == [1]
  ibv = fss(list("hello world")).i_seqs_by_value()
  assert sorted(ibv.keys()) == [" ", "d", "e", "h", "l", "o", "r", "w"]
  assert list(ibv[" "]) == [5]
  assert list(ibv["d"]) == [10]
  assert list(ibv["e"]) == [1]
  assert list(ibv["h"]) == [0]
  assert list(ibv["l"]) == [2,3,9]
  assert list(ibv["o"]) == [4,7]
  assert list(ibv["r"]) == [8]
  assert list(ibv["w"]) == [6]

def exercise_misc():
  assert flex.double.element_size() != 0
  f = flex.double((1,2,3))
  assert f.element_size() == flex.double.element_size()
  assert f[0] == 1
  assert f[2] == 3
  assert f[-1] == 3
  assert f[-3] == 1
  assert f.front() == 1
  assert f.back() == 3
  f.fill(42)
  assert tuple(f) == (42,42,42)
  assert f.capacity() == 3
  f.reserve(10)
  assert f.capacity() == 10
  assert f.size() == 3
  f.reserve(5)
  assert f.capacity() == 10
  assert f.size() == 3
  f = flex.double((1,2,3,4,5,6))
  fc = f.deep_copy()
  assert fc.id() != f.id()
  assert tuple(fc) == (1,2,3,4,5,6)
  fc = f.shallow_copy()
  assert fc.id() == f.id()
  assert tuple(fc) == (1,2,3,4,5,6)
  fc.resize(flex.grid((2,3)))
  assert f.nd() == 1
  assert fc.nd() == 2
  f = flex.double((1,2,3))
  g = flex.double(f)
  assert g.id() == f.id()
  f1 = f.as_1d()
  assert f1.id() == f.id()
  assert tuple(f1) == (1,2,3)
  f = flex.double(flex.grid((2,3)))
  assert f.nd() == 2
  f1 = f.as_1d()
  assert f1.id() == f.id()
  assert f1.nd() == 1
  assert f1.size() == 6
  g = flex.grid((1,2,3), (4,6,8)).set_focus((3,5,7))
  f = flex.double(g)
  assert f.focus_size_1d() == 2*3*4
  assert f.shift_origin().accessor() == g.shift_origin()
  for flex_type in [flex.int, flex.long, flex.int8, flex.int16, flex.int32, flex.int64]:
    b = flex_type([0,0,1,0,1,1,1,0,0,1,1,0,0,0]).as_bool()
    assert b.md5().hexdigest() == "a3a1ff7423c672e6252003c23ad5420f"
    b = flex_type([0,0,1,0,1,0,1,0,0,1,1,0,0,0]).as_bool()
    assert b.md5().hexdigest() == "bc115dabbd6dc87323302b082152be14"
    #
    try: flex_type([0,0,1,0,2,0,1,0,0,1,1,0,0,0]).as_bool(strict=True)
    except ValueError as e:
      if flex_type == flex.int8:  # value= instead of value=2
        assert "all array elements" in str(e)
      else:
        assert str(e) == "scitbx.array_family.flex.int.as_bool(strict=True):" \
          " all array elements must be 0 or 1, but value=2 at array index=4."
    else: raise Exception_expected
    assert flex_type([0,0,1,0,2,0,1,0,0,1,1,0,0,0]) \
      .as_bool(strict=False).count(True) == 5
  #
  for flex_type in (flex.int, flex.int32, flex.long, flex.int64):
    a = flex_type([0,1,2,-1,-2,2**30,2**31-1,-2**30,-2**31, 0]).as_long()
    a.reshape(flex.grid(2,5))
    b = flex.long([0,1,2,-1,-2,2**30,2**31-1,-2**30,-2**31, 0])
    assert a.all_eq(b)
  for flex_type in (flex.int, flex.int32, flex.long, flex.int64):
    a = flex_type([0,1,2,2**30,2**31-1]).as_size_t()
    b = flex.size_t([0,1,2,2**30,2**31-1])
    assert a.all_eq(b)
  for flex_type in (flex.int8, flex.int16):
    a = flex_type([0,1,2,-1,-2,2**6,2**7-1,-2**6,-2**7, 0]).as_long()
    a.reshape(flex.grid(2,5))
    b = flex.long([0,1,2,-1,-2,2**6,2**7-1,-2**6,-2**7, 0])
    assert a.all_eq(b)
  #
  class old_style:
    def __init__(self, elems):
      self.elems = tuple(elems)
    def __len__(self):
      return len(self.elems)
    def __getitem__(self, i):
      return self.elems[i]
  class new_style(object):
    def __init__(self, elems):
      self.elems = tuple(elems)
    def __len__(self):
      return len(self.elems)
    def __getitem__(self, i):
      return self.elems[i]
  assert flex.double(old_style([2,3,4])).all_eq(flex.double([2,3,4]))
  assert flex.double(new_style([3,4,5])).all_eq(flex.double([3,4,5]))
  for s in ["", u""]:
    try: flex.double(s)
    except Exception as e:
      assert str(e).startswith("Python argument types in")
    else: raise Exception_expected
  #
  a = flex.double(12)
  assert a.focus() == (12,)
  a.reshape(flex.grid(3,4))
  assert a.focus() == (3,4)
  a.reshape(flex.grid(2,3,2))
  assert a.focus() == (2,3,2)
  try: a.reshape(flex.grid(5,6))
  except RuntimeError as e:
    assert str(e).find("SCITBX_ASSERT(grid.size_1d() == a.size())") > 0
  else: raise Exception_expected
  #
  for l in [[], [12], [2,3], [4,6,7], list(range(-123,2345))]:
    for flex_type,flex_from_byte_str in [
          (flex.int, flex.int_from_byte_str),
          (flex.long, flex.long_from_byte_str),
          (flex.int16, flex.int16_from_byte_str),
          (flex.int32, flex.int32_from_byte_str),
          (flex.int64, flex.int64_from_byte_str),
          (flex.size_t, flex.size_t_from_byte_str),
          (flex.int8, flex.int8_from_byte_str),
          (flex.uint8, flex.uint8_from_byte_str),
          (flex.uint16, flex.uint16_from_byte_str),
          (flex.uint32, flex.uint32_from_byte_str),
          (flex.uint64, flex.uint64_from_byte_str),
          (flex.double, flex.double_from_byte_str)]:
      if (flex_type in (flex.size_t, flex.uint8, flex.uint16, flex.uint32, flex.uint64)
          and len(l) != 0 and l[0] < 0):
        l = list(range(0, 128))
      a = flex_type(l)
      b = a.copy_to_byte_str()
      if (len(l) == 0):
        assert len(b) == 0
      else:
        assert len(b) != 0
        assert len(b) % len(l) == 0
      f = flex_from_byte_str(byte_str=b)
      assert f.size() == len(l)
      assert list(f) == l

      if (not hasattr(a, "slice_to_byte_str")):
        continue
      start_index = int( 0.25 * a.size() )
      stop_index = min(a.size(), int(0.50 * a.size()) + 1)
      a_slice = a[start_index:stop_index]
      b = a.slice_to_byte_str(start_index, stop_index)
      if (len(l) == 0):
        assert len(b) == 0
      else:
        assert len(b) != 0
        assert len(b) % a_slice.size() == 0
      f = flex_from_byte_str(byte_str=b)
      assert f.size() == a_slice.size()
      assert list(f) == list(a_slice)
  #
  for n in range(10):
    a = flex.size_t(range(n)).as_int()
    assert list(a) == list(range(n))
    s = a.as_rgb_scale_string(
      rgb_scales_low=(1,1,1),
      rgb_scales_high=(0,0,0),
      saturation=max(1,n))
    assert len(s) == 3*n
  a = flex.int([1,3,0,2,-1])
  s = a.as_rgb_scale_string(
    rgb_scales_low=(1,1,1),
    rgb_scales_high=(0,0,0),
    saturation=2)
  if six.PY3:
    ord_ = lambda x: x
  else:
    ord_ = ord
  assert [ord_(c) for c in s] \
      == [128, 128, 128, 0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255]
  s = a.as_rgb_scale_string(
    rgb_scales_low=(1,1,1),
    rgb_scales_high=(1,0,0),
    saturation=2)
  assert [ord_(c) for c in s] \
      == [255, 128, 128, 255, 0, 0, 255, 255, 255, 255, 0, 0, 255, 255, 255]
  #
  a = flex.double([1,2,3,-4,5,6])
  a.reshape(flex.grid(2,3))
  s = a.as_scitbx_matrix()
  assert s.elems == (1,2,3,-4,5,6)
  assert s.n == (2,3)
  #
  a = flex.double([1,5,2,8,7])
  m = flex.median(a)
  assert approx_equal(a, [1,5,2,8,7])
  assert approx_equal(m, 5)
  d = flex.median.dispersion(a)
  assert approx_equal(a, [1,5,2,8,7])
  assert approx_equal(d.median, 5)

def exercise_1d_slicing_core(a):
  if (tuple(a[:]) != ()):
    assert tuple(a[:]) == (1,2,3,4,5)
    assert tuple(a[0:]) == (1,2,3,4,5)
    assert tuple(a[1:]) == (2,3,4,5)
    assert tuple(a[-2:]) == (4,5)
    assert tuple(a[-1:]) == (5,)
  else:
    # Numeric slicing appears to be broken in some versions (24.2)
    assert hasattr(a, "typecode") # assert is Numeric array
    assert tuple(a[0:]) == ()
    assert tuple(a[1:]) == ()
    assert tuple(a[-2:]) == ()
    assert tuple(a[-1:]) == ()
  assert tuple(a[::]) == (1,2,3,4,5)
  assert tuple(a[0::]) == (1,2,3,4,5)
  assert tuple(a[1::]) == (2,3,4,5)
  assert tuple(a[-2::]) == (4,5)
  assert tuple(a[-1::]) == (5,)
  assert tuple(a[-6::]) == (1,2,3,4,5) # Numeric-21.0 a[-6:] is different
  assert tuple(a[-7::]) == (1,2,3,4,5) # Numeric-21.0 a[-7:] is different
  assert tuple(a[:-1]) == (1,2,3,4)
  assert tuple(a[:-2]) == (1,2,3)
  assert tuple(a[:-4]) == (1,)
  assert tuple(a[:-5]) == ()
  assert tuple(a[:-6:]) == () # Numeric-21.0 a[:-6] is different
  assert tuple(a[:-7:]) == () # Numeric-21.0 a[:-7] is different
  assert tuple(a[:0]) == ()
  assert tuple(a[:1]) == (1,)
  assert tuple(a[:2]) == (1,2)
  assert tuple(a[:3]) == (1,2,3)
  assert tuple(a[:4]) == (1,2,3,4)
  assert tuple(a[:5]) == (1,2,3,4,5)
  assert tuple(a[:6]) == (1,2,3,4,5)
  assert tuple(a[::1]) == (1,2,3,4,5)
  assert tuple(a[::2]) == (1,3,5)
  assert tuple(a[1::2]) == (2,4)
  assert tuple(a[1:-1:2]) == (2,4)
  assert tuple(a[1:-2:2]) == (2,)
  assert tuple(a[4:2]) == ()
  assert tuple(a[4:2:-1]) == (5,4)
  assert tuple(a[2:4:-1]) == ()
  assert tuple(a[::-1]) == (5,4,3,2,1)
  assert tuple(a[::-2]) == (5,3,1)
  assert tuple(a[-1::-2]) == (5,3,1)
  assert tuple(a[-1:1:-2]) == (5,3)
  assert tuple(a[-1:2:-2]) == (5,)
  try: tuple(a[3:3:0]) == ()
  except ValueError as e: assert str(e) == "slice step cannot be zero"

def exercise_flex_sum_axis():
  try:
    import numpy
  except ImportError:
    print("Skipping flex.sum numpy compatibility testing (numpy not available)")
  else:
    for nd in (1,2,3,4,5):
      dimensions = [random.randint(1,5) for i in range(nd)]
      fa_size = flex.product(flex.int(dimensions))
      fa_int = flex.random_int_gaussian_distribution(fa_size, mu=0, sigma=3)
      fa_double = flex.random_double(fa_size)
      for fa in (fa_int, fa_double):
        fa.resize(flex.grid(dimensions))
        for dim in range(nd):
          for axis in (dim, -dim):
            fa_sum = flex.sum(fa, axis=axis)
            assert approx_equal(fa_sum, fa.as_numpy_array().sum(axis=axis).flatten())
  fa = flex.int([1]*4+[2]*4+[3]*4)
  fa.resize(flex.grid(3,4))
  assert approx_equal(flex.sum(fa, axis=0), (6,6,6,6))
  assert approx_equal(flex.sum(fa, axis=1), (4,8,12))
  fa = flex.double([1]*12+[2]*12)
  fa.resize(flex.grid(2,3,4))
  assert approx_equal(flex.sum(fa, axis=0), ([3]*12))
  assert approx_equal(flex.sum(fa, axis=1), ([3]*4+[6]*4))
  assert approx_equal(flex.sum(fa, axis=2), ([4]*3+[8]*3))

def exercise_1d_slicing():
  exercise_1d_slicing_core(flex.int((1,2,3,4,5)))
  try:
    import numpy
  except ImportError:
    pass
  else:
    print("Testing compatibility with numpy slicing...", end=' ')
    exercise_1d_slicing_core(numpy.array((1,2,3,4,5)))
    print("OK")
  assert list(flex.slice_indices(5, slice(10))) == [0,1,2,3,4]
  assert list(flex.slice_indices(5, slice(0))) == []
  assert list(flex.slice_indices(5, slice(0,10,2))) == [0,2,4]
  assert list(flex.slice_indices(5, slice(1,10,2))) == [1,3]
  assert list(flex.slice_indices(5, slice(1,3,2))) == [1]
  assert list(flex.slice_indices(5, slice(4,0,-2))) == [4,2]

def exercise_nd_slicing():

  for flex_t in (flex.int, flex.double):
    for n in range(0,3):
      for nd in range(1,11): # flex grid supports up to 10-d
        a = flex_t(range(n**nd))
        a.resize(flex.grid(tuple([n]*nd)))
        slices = [slice(None) for i in range(nd)]
        b = a[slices]
        assert approx_equal(b, a)
        assert a is not b

    a = flex_t(range(60))
    a.resize(flex.grid(3,4,5))
    slices = [slice(1,3),slice(2,4),slice(3,5)]
    assert approx_equal(a[slices], [33,34,38,39,53,54,58,59])
    slices = [slice(1,3),slice(2,-1),slice(3,5)]
    assert approx_equal
    a = flex_t(range(5**3))
    a.resize(flex.grid(5,5,5))
    assert approx_equal(a, a[:,:,:])
    assert a is not a[:,:,:]
    assert a[:-5,:-5,:-5].size() == 0
    assert a[:-4:1,:-4:1,:-4:1].size() == 1

    assert approx_equal(a[:2,:2,:2], (0,1,5,6,25,26,30,31))
    assert approx_equal(
      a[1:4,0:3,0:2], (25,26,30,31,35,36,50,51,55,56,60,61,75,76,80,81,85,86))
    try: a[::2,::2,::2]
    except RuntimeError: pass
    else: raise Exception_expected
    slices = [slice(1,3),slice(1,-2,1),slice(-4,-2)]
    assert approx_equal(a[slices], [31, 32, 36, 37, 56, 57, 61, 62])
    slices = [slice(1,3),slice(1,4),slice(3,5)]
    assert approx_equal(a[slices], [33,34,38,39,43,44,58,59,63,64,68,69])

    a = flex_t(range(5**2))
    a.resize(flex.grid(5,5))
    assert approx_equal(a, a[:,:])
    b = a[:2,:2]
    assert b.all() == (2,2)
    assert approx_equal(b, (0,1,5,6))
    c = a[1:3,0:-2]
    assert c.all() == (2,3)
    assert approx_equal(c, (5,6,7,10,11,12))
    c = a[1:3,1:2]
    assert c.all() == (2,1)
    assert approx_equal(c, (6,11))

def exercise_set_nd_slicing():
  for flex_t in (flex.int, flex.double):
    for n in range(0,3):
      for nd in range(1,11): # flex grid supports up to 10-d
        a = flex_t(n**nd, 0)
        a.resize(flex.grid(tuple([n]*nd)))
        b = flex_t(range(n**nd))
        b.resize(flex.grid(tuple([n]*nd)))
        slices = tuple([slice(None) for i in range(nd)])
        a[slices] = b
        assert approx_equal(b, a)
        assert a is not b

    a = flex_t(flex.grid(5,6,7), 0)
    b = flex_t(range(4*5))
    b.resize(flex.grid(1, 4, 5))
    a[2:3,1:5,1:6] = b
    c = a[2:3,1:5,1:6]
    assert approx_equal(b, c)
    assert not b is c

def exercise_numpy_slicing_compatibility():
  try:
    import numpy
  except ImportError:
    print("Skipping exercise_numpy_slicing_compatibility...")
    return
  for j in range(50):
    for n_dim in (3,4,5,6):
      dim = [random.randint(3,15) for i in range(n_dim)]
      size = flex.product(flex.int(dim))
      a = flex.random_double(size)
      a.resize(flex.grid(dim))
      a_numpy = a.as_numpy_array()
      slices = []
      for i in range(n_dim):
        start = random.randint(0,dim[i]-1)
        stop = random.randint(start+1,dim[i])
        slices.append(slice(start,stop))
      assert approx_equal(flex.double(a_numpy[tuple(slices)].flatten()), a[slices])
      slices = []
      for i in range(n_dim):
        start = random.randint(-dim[i],-2)
        stop = random.randint(start+1,-1)
        slices.append(slice(start,stop))
      assert approx_equal(flex.double(a_numpy[tuple(slices)].flatten()), a[slices])

def exercise_push_back_etc():
  a = flex.double(3)
  assert a.size() == 3
  assert tuple(a) == (0, 0, 0)
  a = flex.double()
  assert a.size() == 0
  a.assign(3, 1)
  assert tuple(a) == (1, 1, 1)
  a.append(2)
  assert tuple(a) == (1, 1, 1, 2)
  a.insert(0, 3)
  assert tuple(a) == (3, 1, 1, 1, 2)
  a.insert(2, 3, 4)
  assert tuple(a) == (3, 1, 4, 4, 4, 1, 1, 2)
  a.pop_back()
  assert tuple(a) == (3, 1, 4, 4, 4, 1, 1)
  del a[2]
  assert tuple(a) == (3, 1, 4, 4, 1, 1)
  a.insert(-2, 8)
  assert tuple(a) == (3, 1, 4, 4, 8, 1, 1)
  del a[-3]
  assert tuple(a) == (3, 1, 4, 4, 1, 1)
  del a[3:5]
  assert tuple(a) == (3, 1, 4, 1)
  a.resize(6)
  assert tuple(a) == (3, 1, 4, 1, 0, 0)
  a.resize(8, -1)
  assert tuple(a) == (3, 1, 4, 1, 0, 0, -1, -1)
  a.clear()
  assert a.size() == 0
  a = flex.double((0, 1, 2, 3))
  b = flex.double((4, 5, 6))
  a.extend(b)
  assert tuple(a) == (0, 1, 2, 3, 4, 5, 6)
  g = flex.grid((2,3,5))
  f = flex.double(g, 11)
  g = flex.grid((2,3,7))
  f.resize(g)
  assert list(f) == [11] * 30 + [0] * 12
  g = flex.grid((2,3,9))
  f.resize(g, 22)
  assert list(f) == [11] * 30 + [0] * 12 + [22] * 12
  g = flex.grid((1,2,3))
  f.resize(g)
  assert list(f) == [11] * 6
  b = flex.double([10,13,17])
  assert list(a.reversed()) == [6,5,4,3,2,1,0]
  assert list(a.reversed().reversed()) == list(range(7))
  assert list(a.concatenate(b)) == [0,1,2,3,4,5,6,10,13,17]
  assert list(b.concatenate(a)) == [10,13,17,0,1,2,3,4,5,6]
  #
  for flex_type in [flex.size_t, flex.uint8, flex.uint16, flex.uint32, flex.uint64]:
    a = flex_type()
    a.insert(0, 1)
    assert list(a) == [1]
    a.insert(1, 2)
    assert list(a) == [1,2]
    a.insert(-1, 3)
    assert list(a) == [1,3,2]
    for i in range(-3,4):
      c = a.deep_copy()
      l = list(a)
      c.insert(i, 5)
      l.insert(i, 5)
      assert list(c) == l
    for i in [-5, -4, 4, 5]:
      try: a.insert(i, 5)
      except IndexError: pass
      else: raise Exception_expected
    a.insert(1, 3, 5)
    assert list(a) == [1,5,5,5,3,2]
    a.insert(-1, 2, 6)
    assert list(a) == [1,5,5,5,3,6,6,2]
    a.insert(-5, 1, 7)
    assert list(a) == [1,5,5,7,5,3,6,6,2]

def exercise_setitem():
  a = flex.double(2)
  a[0] = 11
  a[1] = 12
  assert tuple(a) == (11, 12)
  g = flex.grid((2,3))
  a = flex.double(g)
  for i in range(2):
    for j in range(3):
      a[(i,j)] = i * 3 + j
  assert list(a) == list(range(6))

def exercise_select():
  from scitbx import stl
  import scitbx.stl.vector
  a = flex.double((1,2,3,4,5))
  b = flex.bool([bool(f) for f in [0,1,0,1,1]])
  c = flex.size_t((0,2))
  d = stl.vector.unsigned((1,3))
  assert tuple(a.select(b)) == (2,4,5)
  assert tuple(a.select(indices=c)) == (1,3)
  assert tuple(a.select(indices=c, reverse=False)) == (1,3)
  p = flex.size_t([0,3,1,2,4])
  assert list(a.select(p)) == [1,4,2,3,5]
  assert list(a.select(p).select(p, reverse=True)) == list(a)
  for i_trial in range(10):
    p = flex.random_permutation(size=a.size())
    assert list(a.select(p).select(p, reverse=True)) == list(a)
  assert tuple(a.set_selected(b, flex.double((7,8,9)))) == (1,7,3,8,9)
  assert tuple(a.set_selected(c, flex.double((-1,-2)))) == (-1,7,-2,8,9)
  assert tuple(a.set_selected(d, flex.double((-1,-2)))) == (-1,-1,-2,-2,9)
  assert tuple(a.set_selected(
    flex.bool(5,False), flex.double()))==(-1,-1,-2,-2,9)
  assert tuple(a.set_selected(flex.size_t(), flex.double()))==(-1,-1,-2,-2,9)
  assert tuple(a.set_selected(stl.vector.unsigned(), flex.double())) \
      == (-1,-1,-2,-2,9)
  assert tuple(a.set_selected(b, -4)) == (-1,-4,-2,-4,-4)
  assert tuple(a.set_selected(c, -3)) == (-3,-4,-3,-4,-4)
  assert tuple(a.set_selected(d, -2)) == (-3,-2,-3,-2,-4)
  assert tuple(a.set_selected(flex.bool(5, False), -9)) == (-3,-2,-3,-2,-4)
  assert tuple(a.set_selected(flex.size_t(), -9)) == (-3,-2,-3,-2,-4)
  assert tuple(a.set_selected(stl.vector.unsigned(), -9)) == (-3,-2,-3,-2,-4)
  for i,v in enumerate([1,2,3,4]):
    a = flex.double([1,2,3,4])
    a.resize(flex.grid(2,2))
    b = a.deep_copy()
    b[i] *= 10
    assert list(a.set_selected(a==v, v*10)) == list(b)
  #
  a = flex.double([1,2,3])
  d = flex.double([11,12,13])
  assert list(a.set_selected(flex.bool([True, True, True]), d)) == [11,12,13]
  d = flex.double([21,22,23])
  assert list(a.set_selected(flex.bool([False, True, False]), d)) == [11,22,13]
  assert list(a.set_selected(flex.bool([True, False, False]), d)) == [21,22,13]
  #
  a = flex.double([1,2,3])
  d = flex.double([11,12,13])
  s = flex.size_t([0,2])
  assert list(a.copy_selected(s, d)) == [11,2,13]
  a = flex.double([1,2,3])
  s = stl.vector.unsigned([1,2])
  assert list(a.copy_selected(s, d)) == [1,12,13]
  #
  a = flex.double([1,-2,3])
  i = flex.size_t([0,2])
  f = flex.bool([False, True, True])
  v = flex.double([6,-4])
  assert a.add_selected(indices=i, values=v) is a
  assert approx_equal(a, [7,-2,-1])
  assert a.add_selected(indices=i, value=3) is a
  assert approx_equal(a, [10,-2,2])
  assert a.add_selected(flags=f, values=v) is a
  assert approx_equal(a, [10,4,-2])
  v.append(3)
  assert a.add_selected(flags=f, values=v) is a
  assert approx_equal(a, [10,0,1])
  assert a.add_selected(flags=f, value=-3) is a
  assert approx_equal(a, [10,-3,-2])
  #
  a = flex.double((1,2,3,4,5))
  b = flex.size_t((3,1,0,4,2))
  assert tuple(a.select(b)) == (4,2,1,5,3)
  b = flex.size_t((1,4,2))
  assert tuple(a.select(b)) == (2,5,3)
  b = flex.size_t((2,4,1,2,4))
  assert tuple(a.select(b)) == (3,5,2,3,5)
  a = flex.size_t((1,2,3))
  for i in range(3):
    for expected in ([1, 2, 3],
                     [1, 3, 2],
                     [2, 1, 3],
                     [2, 3, 1],
                     [3, 1, 2],
                     [3, 2, 1]):
      assert list(a) == expected
      assert a.next_permutation() == (expected != [3, 2, 1])
  s = [1,2,3]
  a = flex.size_t(s)
  for i in range(3):
    while True:
      assert list(a) == s
      if (not a.next_permutation()):
        assert not libtbx.math_utils.next_permutation(s)
        break
      assert libtbx.math_utils.next_permutation(s)
  assert [list(a) for a in flex.permutation_generator(size=0)] == [[]]
  assert [list(a) for a in flex.permutation_generator(size=1)] == [[0]]
  assert [list(a) for a in flex.permutation_generator(size=2)] == [[0,1],[1,0]]
  a = flex.bool([bool(f) for f in [0,1,0,1,1]])
  assert tuple(a.as_int()) == (0,1,0,1,1)
  assert tuple(a.as_double()) == (0,1,0,1,1)
  assert tuple(a.iselection()) == (1,3,4)
  assert tuple(a.iselection(test_value=True)) == (1,3,4)
  assert tuple(a.iselection(test_value=False)) == (0,2)
  a = flex.bool([False] * 5)
  assert tuple(a.iselection(False)) == (0,1,2,3,4)
  assert tuple(a.iselection(True)) == ()
  isel = stl.vector.unsigned([1,3])
  a = flex.bool(size=5, iselection=isel)
  assert tuple(a) == (False, True, False, True, False)
  isel2 = stl.vector.unsigned([1,4])
  assert tuple(flex.union(size=5, iselections=[isel,isel2])) \
      == (False, True, False, True, True)
  assert tuple(flex.intersection(size=5, iselections=[isel,isel2])) \
      == (False, True, False, False, False)
  assert list(flex.intersection(size=5,iselections=[isel,isel2]).iselection())\
      == [1]
  isel = flex.size_t([1,4])
  a = flex.bool(size=5, iselection=isel)
  assert tuple(a) == (False, True, False, False, True)
  isel2 = flex.size_t([0,4])
  assert tuple(flex.union(size=5, iselections=[isel,isel2])) \
      == (True, True, False, False, True)
  assert tuple(flex.intersection(size=5, iselections=[isel,isel2])) \
      == (False, False, False, False, True)
  assert list(flex.intersection(size=5,iselections=[isel,isel2]).iselection())\
      == [4]
  #
  def check(a, b, expected):
    a = flex.size_t(a)
    b = flex.size_t(b)
    result = a.intersection(other=b)
    assert list(result) == expected
    a_i_seqs, b_i_seqs = a.intersection_i_seqs(other=b)
    assert list(a.select(a_i_seqs)) == expected
    assert list(b.select(b_i_seqs)) == expected
  check([], [], [])
  check([1,2,3,4], [1,2,3,4], [1,2,3,4])
  check([], [1,2,3,4], [])
  check([1,2,3,4], [], [])
  check([1], [1,2,3,4], [1])
  check([1,2,3,4], [1], [1])
  check([4], [1,2,3,4], [4])
  check([1,2,3,4], [4], [4])
  check([1,4], [1,2,3,4], [1,4])
  check([1,2,3,4], [1,4], [1,4])
  check([1,4,5], [1,2,3,4], [1,4])
  check([1,2,3,4], [1,4,5], [1,4])
  check([1,2,3,4], [2], [2])
  check([2], [1,2,3,4], [2])
  check([1,2,3,4], [2,3], [2,3])
  check([2,3], [1,2,3,4], [2,3])
  check([1,2,3,4], [2,4], [2,4])
  check([2,4], [1,2,3,4], [2,4])
  #
  fs = flex.size_t
  try: fs([1,1]).intersection(fs([1,2]))
  except RuntimeError as e:
    assert str(e).find("first") > 0
    assert str(e).find("duplicate") > 0
  else: raise Exception_expected
  try: fs([2,1]).intersection(fs([1,2]))
  except RuntimeError as e:
    assert str(e).find("first") > 0
    assert str(e).find("sorted") > 0
  else: raise Exception_expected
  try: fs([1,2]).intersection(fs([1,1]))
  except RuntimeError as e:
    assert str(e).find("second") > 0
    assert str(e).find("duplicate") > 0
  else: raise Exception_expected
  try: fs([1,3]).intersection(fs([2,1]))
  except RuntimeError as e:
    assert str(e).find("second") > 0
    assert str(e).find("sorted") > 0
  else: raise Exception_expected
  #
  a = flex.double(range(3,12))
  for stl_iterable in [stl.vector.unsigned, stl.set.unsigned]:
    assert a.select(selection=stl_iterable()).size() == 0
    assert approx_equal(a.select(selection=stl_iterable([2,3,7])), [5,6,10])
  #
  a = flex.int(range(7, 22))
  a.reshape(flex.grid((3,5)).set_focus((3,4)))
  assert a.origin() == (0,0)
  assert a.focus() == (3,4)
  assert a.all() == (3,5)
  b = flex.mersenne_twister(seed=84).random_bool(size=15, threshold=0.5)
  assert list(a.select(b)) == [9, 11, 13, 14, 18, 19, 21]
  i = b.iselection()
  assert list(a.select(i)) == [9, 11, 13, 14, 18, 19, 21]
  p = flex.mersenne_twister(seed=84).random_permutation(size=15)
  assert list(a.select(p, reverse=True)) \
      == [16, 14, 20, 13, 19, 21, 7, 10, 12, 11, 8, 9, 15, 18, 17]
  #
  mt = flex.mersenne_twister(seed=92)
  flex.set_random_seed(92)
  for f in (mt.random_selection, flex.random_selection):
    s = f(population_size=15, sample_size=10)
    assert list(s) == [0, 1, 2, 3, 5, 9, 10, 11, 12, 14]
    for n in range(2):
      s = f(population_size=n, sample_size=0)
      assert s.size() == 0

def exercise_from_stl_vector():
  from scitbx import stl
  import scitbx.stl.vector
  for flex_type in [flex.size_t, flex.uint8, flex.uint16, flex.uint32, flex.uint64]:
    assert list(flex_type(stl.vector.unsigned([2,5,9]))) == [2,5,9]
  assert list(flex.double(stl.vector.double([3,-6,10]))) == [3,-6,10]

def exercise_operators():
  a = flex.int((4, 9))
  b = flex.int((2, 3))
  assert tuple(-a) == (-4, -9)
  assert tuple(a + b) == (6, 12)
  assert tuple(a - b) == (2, 6)
  assert tuple(a * b) == (8, 27)
  assert tuple(a / b) == (2, 3)
  assert tuple(a % b) == (0, 0)
  assert tuple(a + 3) == (7, 12)
  assert tuple(3 + a) == (7, 12)
  assert tuple(a - 4) == (0, 5)
  assert tuple(4 - a) == (0, -5)
  assert tuple(a * 5) == (20, 45)
  assert tuple(5 * a) == (20, 45)
  assert tuple(a / 2) == (2, 4)
  assert tuple(9 / a) == (2, 1)
  assert tuple(a % 2) == (0, 1)
  assert tuple(13 % a) == (1, 4)
  assert flex.sum(a) == 13
  a = flex.int((4, 9))
  b = flex.int((2, 12))
  assert tuple(a == b) == (0, 0)
  assert tuple(a != b) == (1, 1)
  assert tuple(a < b) == (0, 1)
  assert tuple(a > b) == (1, 0)
  assert tuple(a <= b) == (0, 1)
  assert tuple(a >= b) == (1, 0)
  assert tuple(a == 9) == (0, 1)
  assert tuple(a.as_double()) == (4, 9)
  assert a.all_eq(a)
  assert not a.all_eq(b)
  assert not a.all_eq(4)
  assert a.all_ne(b)
  assert not a.all_ne(a)
  assert not a.all_ne(4)
  assert a.all_ne(5)
  assert not a.all_lt(b)
  assert a.all_lt(10)
  assert not a.all_gt(b)
  assert a.all_gt(3)
  assert not a.all_le(b)
  assert a.all_le(9)
  assert not a.all_ge(b)
  assert a.all_ge(2)
  assert flex.order(a, b) == cmp(tuple(a), tuple(b))
  assert flex.order(b, a) == cmp(tuple(b), tuple(a))
  assert flex.order(a, a) == 0
  b = a.deep_copy()
  assert flex.order(a, b) == 0
  #
  a = flex.double([1,2,3])
  assert approx_equal(a*(2+3j), [(2+3j), (4+6j), (6+9j)])
  assert approx_equal((3+2j)*a, [(2j+3), (4j+6), (6j+9)])
  assert approx_equal(a*3, [3,6,9])
  assert approx_equal(4*a, [4,8,12])

def exercise_bitwise_operators():
  a = flex.int((
    1 << 0, # 0001
    1 << 1, # 0010
    1 << 2, # 0100
    1 << 3))# 1000
  b = ~a
  assert(list(b) == [-2, -3, -5, -9])
  b = a | (1 << 0)
  assert(list(b) == [1, 3, 5, 9])
  b = a | (1 << 1)
  assert(list(b) == [3, 2, 6, 10])
  b = a | (1 << 2)
  assert(list(b) == [5, 6, 4, 12])
  b = a | (1 << 3)
  assert(list(b) == [9, 10, 12, 8])
  c = flex.int((
    1 << 3,
    1 << 2,
    1 << 1,
    1 << 0))
  b = a | c
  assert(list(b) == [9, 6, 6, 9])
  b = a & (1 << 0)
  assert(list(b) == [1, 0, 0, 0])
  b = a & (1 << 1)
  assert(list(b) == [0, 2, 0, 0])
  b = a & (1 << 2)
  assert(list(b) == [0, 0, 4, 0])
  b = a & (1 << 3)
  assert(list(b) == [0, 0, 0, 8])
  c = flex.int((
    1 << 3,
    1 << 2,
    1 << 1,
    1 << 0))
  b = a & c
  assert(list(b) == [0, 0, 0, 0])
  b = a ^ (1 << 0)
  assert(list(b) == [0, 3, 5, 9])
  b = a ^ (1 << 1)
  assert(list(b) == [3, 0, 6, 10])
  b = a ^ (1 << 2)
  assert(list(b) == [5, 6, 0, 12])
  b = a ^ (1 << 3)
  assert(list(b) == [9, 10, 12, 0])
  c = flex.int((
    1 << 3,
    1 << 2,
    1 << 1,
    1 << 0))
  b = a ^ c
  assert(list(b) == [9, 6, 6, 9])


def exercise_bool():
  a = flex.bool((False, True, False, True))
  b = flex.bool((False, True, True, False))
  f = flex.bool((False, False))
  t = flex.bool((True, True))
  assert tuple(a == b) == (True, True, False, False)
  assert tuple(a != b) == (False, False, True, True)
  assert tuple(a == True) == (False, True, False, True)
  assert tuple(a == False) == (True, False, True, False)
  assert not a == None
  assert type(a == None) == type(False)
  try: a == flex.int()
  except TypeError as e:
    assert str(e) \
        == "Type of argument must be a Python bool, flex.bool, or None."
  else: raise Exception_expected
  assert tuple(a != True) == (True, False, True, False)
  assert tuple(a != False) == (False, True, False, True)
  assert a != None
  assert type(a != None) == type(False)
  try: a != flex.int()
  except TypeError as e:
    assert str(e) \
        == "Type of argument must be a Python bool, flex.bool, or None."
  else: raise Exception_expected
  assert [a, a].count(None) == 0
  assert [a, None].count(None) == 1
  assert a.all_eq(a)
  assert type(a.all_eq(a)) == type(False)
  assert a.all_eq(a.deep_copy())
  assert not a.all_ne(a)
  assert a.all_ne(~a)
  assert not a.all_ne(b)
  assert f.all_eq(False)
  assert f.all_ne(True)
  assert t.all_eq(True)
  assert t.all_ne(False)
  assert type(f.all_eq(f)) == type(False)
  assert type(f.all_ne(f)) == type(False)
  assert type(f.all_eq(False)) == type(False)
  assert type(f.all_ne(False)) == type(False)
  for mf in [t.all_eq, t.all_ne]:
    try: mf(None)
    except TypeError as e:
      assert str(e) == "Type of argument must be a Python bool or flex.bool."
    else: raise Exception_expected
  assert tuple(~a) == (True, False, True, False)
  assert tuple(a & b) == (False, True, False, False)
  assert tuple(a | b) == (False, True, True, True)
  a &= b
  assert tuple(a) == (False, True, False, False)
  a |= flex.bool((True, False, True, False))
  assert tuple(a) == (True, True, True, False)
  assert a.count(False) == 1
  assert a.count(True) == 3
  a &= True
  assert tuple(a) == (True, True, True, False)
  a &= False
  assert tuple(a) == (False, False, False, False)
  a |= True
  assert tuple(a) == (True, True, True, True)
  a |= False
  assert tuple(a) == (True, True, True, True)
  #
  assert flex.order(a,a) == False
  assert flex.order(a,b) == 1
  assert flex.order(b,a) == -1
  #
  a = flex.bool([False, False, True, True])
  b = flex.bool([False, True, False, True])
  assert list(a.exclusive_or(b)) == [False, True, True, False]
  #
  a = flex.bool([True,False,False,True,True])
  b = flex.size_t([0,1,2,3])
  assert list(a.filter_indices(b)) == [0,3]
  #
  a = flex.bool([False, True, True, True, True])
  b = flex.bool([False, False, True, False, True])
  assert a.is_super_set(b) and not b.is_super_set(a)

def exercise_arith_inplace_operators():
  a = flex.int((4, 9))
  a += 3
  assert tuple(a) == (7, 12)
  a -= 3
  assert tuple(a) == (4, 9)
  a *= 3
  assert tuple(a) == (12, 27)
  a /= 3
  assert tuple(a) == (4, 9)
  a %= 3
  assert tuple(a) == (1, 0)
  a += 5
  a += a
  assert tuple(a) == (12, 10)
  a -= flex.int((4, 3))
  assert tuple(a) == (8, 7)
  a *= a
  assert tuple(a) == (64, 49)
  a /= flex.int((2, 1))
  assert tuple(a) == (32, 49)
  a %= flex.int((15, 14))
  assert tuple(a) == (2, 7)

def exercise_functions():
  a = flex.int((-1, 0, 1))
  assert tuple(flex.abs(a)) == (1, 0, 1)
  assert tuple(flex.pow2(a)) == (1, 0, 1)
  assert a.count(0) == 1
  assert a.count(2) == 0
  a = flex.int((1,1,1,3,3,3,3))
  assert a.count(1) == 3
  assert a.count(3) == 4
  a = flex.double((1, 0, 3, 2))
  b = flex.double((4, 5, 6))
  assert a.count(3) == 1
  assert b.count(3) == 0
  assert flex.min_index(a) == 1
  assert flex.min_index(b) == 0
  assert flex.max_index(a) == 2
  assert flex.max_index(b) == 2
  assert flex.min(a) == 0
  assert flex.min(b) == 4
  assert flex.max(a) == 3
  assert flex.max(b) == 6
  assert flex.sum(a) == 6
  assert flex.sum(b) == 15
  assert approx_equal(a.norm(), 3.74165738677)
  assert approx_equal(b.norm(), 8.77496438739)
  assert flex.product(a) == 0
  assert flex.product(b) == 120
  assert flex.mean(a) == 6. / 4
  assert flex.mean(b) == 15. / 3
  assert flex.mean_sq(a) == (1+3.*3.+2.*2.) / 4
  assert approx_equal(flex.mean_sq(b), (4.*4.+5.*5.+6.*6.) / 3)
  a = flex.double((-2, 0, 3))
  assert tuple(flex.pow(a, 2)) == (4, 0, 9)
  a = flex.double((2, 0, 3))
  assert list(flex.sqrt(a)) == [math.sqrt(x) for x in a]
  b = flex.double((1, 1, 1))
  assert (flex.mean(a) - flex.mean_weighted(a, b)) < 1.e-6
  assert (flex.mean_sq(a) - flex.mean_sq_weighted(a, b)) < 1.e-6
  assert a.all_approx_equal(a)
  assert a.all_approx_equal(other=a, tolerance=1.e-6)
  assert not a.all_approx_equal(other=b)
  assert a.all_approx_equal(b, tolerance=3)
  assert not a.all_approx_equal(1)
  assert b.all_approx_equal(other=1)
  assert a.all_approx_equal(other=1, tolerance=3)
  #
  a = flex.double([-3.67,-0.123,-0.678,0.321,0.765,8.01])
  b = a.as_float()
  assert approx_equal(b, a)
  b = a.round()
  assert approx_equal(b, [-4,0,-1,0,1,8])
  for n_digits in [-2,1,0,1,2]:
    b = a.round(n_digits=n_digits)
    for x,y in zip(a,b):
      assert approx_equal(round(x, n_digits), y)
  assert list(b.iround()) == [-4, 0, -1, 0, 1, 8]
  assert flex.double().iround().size() == 0
  #
  a = flex.std_string(["a"]*3 + ["b"]*4)
  assert a.count("a") == 3
  assert a.count("b") == 4
  assert a.count("c") == 0
  list(flex.split_lines(multi_line_string="")) == []
  for multi_line_string in [
    "",
    "\n",
    "\r",
    "\r\n",
    "\n\r",
    "a\nb\n",
    "a\nb",
    "a\nb",
    "a\r\nb\n",
    "a\r\n\nb\n",
    "\ra\n\r\nb\n",
    "\ra\r\n\nb\n"]:
    for keep_ends in [False, True]:
      for count_lines_first in [True, False]:
        assert list(flex.split_lines(
                      multi_line_string,
                      keep_ends=keep_ends,
                      count_lines_first=count_lines_first)) \
            == multi_line_string.splitlines(keep_ends)
  # 2to3 migration comment: must use list(range); do not wrap items() in list, a.counts() is not dict
  a = flex.int(list(range(-2,2)) + list(range(2,4)) + list(range(1,3)))
  assert a.counts().items() == [(-2,1),(-1,1),(0,1),(1,2),(2,2),(3,1)]
  assert a.counts(max_keys=6).items() == a.counts().items()
  a = flex.long(list(range(-2,2)) + list(range(-1,3)) + list(range(1,3)))
  assert a.counts().items() == [(-2,1),(-1,2),(0,2),(1,3),(2,2)]
  assert a.counts(max_keys=5).items() == a.counts().items()
  a = flex.size_t(list(range(1,2)) + list(range(1,3)) + list(range(1,4)))
  assert a.counts().items() == [(1, 3), (2, 2), (3, 1)]
  assert a.counts(max_keys=3).items() == a.counts().items()
  try: a.counts(max_keys=2)
  except RuntimeError as e:
    assert str(e) == "scitbx::af::counts::limited: max_keys exceeded."
  else: raise Exception_expected
  #
  x = flex.double([-6.3,7.2])
  assert approx_equal(flex.fmod(x, 5), [-1.3, 2.2])
  assert approx_equal(flex.fmod_positive(x, 5), [3.7, 2.2])
  #
  values = flex.double()
  stats = values.min_max_mean()
  assert stats.min is None
  assert stats.max is None
  assert stats.mean is None
  for format in ["%.6g", None]:
    assert values.format_min(format=format) == "None"
    assert values.format_max(format=format) == "None"
    assert values.format_mean(format=format) == "None"
    s = StringIO()
    stats.show(out=s, prefix="values ", format=format)
    assert not show_diff(s.getvalue(), """\
values n: 0
values min:  None
values max:  None
values mean: None
""")
  s = StringIO()
  stats.show(out=s, format="%6.2f")
  assert not show_diff(s.getvalue(), """\
n: 0
min:    None
max:    None
mean:   None
""")
  values = flex.double([4,2,-6,14,-7,5])
  stats = values.min_max_mean()
  assert stats.n == 6
  assert approx_equal(stats.min, -7)
  assert approx_equal(stats.max, 14)
  assert approx_equal(stats.sum, 12)
  assert approx_equal(stats.mean, 2)
  for format,p0 in [("%.6g", ""), (None, ".0")]:
    assert values.format_min(format=format) == "-7"+p0
    assert values.format_max(format=format) == "14"+p0
    assert values.format_mean(format=format) == "2"+p0
    s = StringIO()
    stats.show(out=s, format=format)
    assert not show_diff(s.getvalue(), """\
n: 6
min:  -7%s
max:  14%s
mean: 2%s
""" % (p0,p0,p0))
  s = StringIO()
  stats.show(out=s, format="%6.2f")
  assert not show_diff(s.getvalue(), """\
n: 6
min:   -7.00
max:   14.00
mean:   2.00
""")
  #
  a = flex.complex_double([1+1j, 1-1j, 2+2j, 4+2j])
  assert a.part_names() == ("real", "imag")
  assert approx_equal(a.parts(), [(1,1,2,4), (1,-1,2,2)])
  assert approx_equal(a.parts(), [flex.real(a), flex.imag(a)])
  assert flex.mean(a) == 2+1j
  assert approx_equal(flex.sum_sq(a), 32)
  assert approx_equal(flex.mean_sq(a), 8)
  #
  assert flex.double().standard_deviation_of_the_sample() is None
  assert approx_equal(
    flex.double([1]).standard_deviation_of_the_sample(), 0)
  assert approx_equal(
    flex.double([1,2]).standard_deviation_of_the_sample(), 0.5)
  assert flex.double([1]).sample_standard_deviation() is None
  assert approx_equal(flex.double([3,7]).sample_standard_deviation(), 8**0.5)

def exercise_complex_functions():
  assert (flex.complex_double() == None) is False
  try:
    cd_none = flex.complex_double([None])
  except TypeError as e:
    assert "converter" in str(e)
  else: raise Exception_expected
  c = 1+2j
  x = flex.complex_double((c,))
  y = flex.real(x)
  assert approx_equal(y[0], 1)
  y = flex.imag(x)
  assert approx_equal(y[0], 2)
  y = flex.conj(x)
  d = y[0]
  assert approx_equal(d.real, c.real)
  assert approx_equal(d.imag, -c.imag)
  a = flex.abs(x)
  assert approx_equal(a[0], abs(c))
  p = flex.arg(x)
  y = flex.polar(a, p)
  d = y[0]
  assert approx_equal(d.real, c.real)
  assert approx_equal(d.imag, c.imag)
  p = flex.arg(x, False)
  y = flex.polar(a, p, False)
  d = y[0]
  assert approx_equal(d.real, c.real)
  assert approx_equal(d.imag, c.imag)
  p = flex.arg(x, True)
  y = flex.polar(a, p, True)
  d = y[0]
  assert approx_equal(d.real, c.real)
  assert approx_equal(d.imag, c.imag)
  y = flex.polar(a, p, False)
  d = y[0]
  assert not_approx_equal(d.real, c.real)
  assert not_approx_equal(d.imag, c.imag)
  p = flex.arg(x, False)
  y = flex.polar(x, p)
  d = y[0]
  assert approx_equal(d.real, c.real)
  assert approx_equal(d.imag, c.imag)
  y = flex.polar(x, p, False)
  d = y[0]
  assert approx_equal(d.real, c.real)
  assert approx_equal(d.imag, c.imag)
  y = flex.polar(x, p, True)
  d = y[0]
  assert not_approx_equal(d.real, c.real)
  assert not_approx_equal(d.imag, c.imag)
  y = flex.polar(a, x)
  d = y[0]
  assert approx_equal(d.real, c.real)
  assert approx_equal(d.imag, c.imag)
  y = flex.polar(x, x)
  d = y[0]
  assert approx_equal(d.real, c.real)
  assert approx_equal(d.imag, c.imag)
  y = flex.polar(1, p)
  d = y[0]
  assert approx_equal(abs(d), 1)
  assert approx_equal(flex.arg(y)[0], p[0])
  y = flex.polar(a, math.pi/2)
  d = y[0]
  assert approx_equal(d.real, 0)
  assert approx_equal(d.imag, a[0])
  y = flex.double([2])
  assert approx_equal(x*y, [(2+4j)])
  assert approx_equal(y*x, [(2+4j)])

def exercise_sort():
  for flex_type in (flex.int, flex.size_t, flex.double):
    x = flex_type((3,1,2))
    p = flex.sort_permutation(data=x)
    assert tuple(p) == (1,2,0)
    assert approx_equal(x.select(p), (1,2,3))
    p = flex.sort_permutation(data=x, reverse=False)
    assert tuple(p) == (1,2,0)
    assert approx_equal(x.select(p), (1,2,3))
    p = flex.sort_permutation(x, True)
    assert tuple(p) == (0,2,1)
    assert approx_equal(x.select(p), (3,2,1))
    x = flex_type(
      [7, 1, 1, 5, 1, 3, 3, 7, 1, 7, 3, 3, 5, 7, 5, 5, 5, 1, 1, 1])
    expected = [
      1, 2, 4, 8, 17, 18, 19, 5, 6, 10, 11, 3, 12, 14, 15, 16, 0, 7, 9, 13]
    p = flex.sort_permutation(x, stable=True)
    assert approx_equal(p, expected)
    expected = [
      0, 7, 9, 13, 3, 12, 14, 15, 16, 5, 6, 10, 11, 1, 2, 4, 8, 17, 18, 19]
    p = flex.sort_permutation(x, reverse=True, stable=True)
    assert approx_equal(p, expected)

  for i_trial in range(10):
    a = flex.size_t([0,0,0,1,1,2,2,2,3,4,4])
    if (i_trial):
      a = a.select(flex.sort_permutation(flex.random_double(size=a.size())))
    x = flex.random_double(size=5)
    p = flex.sort_permutation(data=x)
    pp = p.inverse_permutation()
    assert pp.inverse_permutation().all_eq(p)
    ap = pp.select(a)
    xp = x.select(p)
    for i,j in zip(a,ap):
      assert x[i] == xp[j]
  #
  a = flex.size_t()
  s = flex.size_t()
  assert a.increment_and_track_up_from_zero(iselection=s) == 0
  a = flex.size_t(3)
  s = flex.size_t([1])
  assert a.increment_and_track_up_from_zero(iselection=s) == 1
  assert list(a) == [0,1,0]
  assert a.increment_and_track_up_from_zero(iselection=s) == 0
  assert list(a) == [0,2,0]
  s = flex.size_t([0,2])
  assert a.increment_and_track_up_from_zero(iselection=s) == 2
  assert list(a) == [1,2,1]

def exercise_random():
  mt = flex.mersenne_twister()
  assert mt.random_size_t_min() == 0
  assert mt.random_size_t_max() == 4294967295
  assert mt.random_size_t() == 1791095845
  assert approx_equal(mt.random_double(), 0.997184808365)
  for i in range(3):
    for j in range(2):
      if (j == 0): mt = flex.mersenne_twister()
      else: mt.seed()
      assert tuple(mt.random_size_t(3)) \
          == (1791095845, 4282876139, 3093770124)
      assert approx_equal(mt.random_double(3),
       (0.9325573593386588, 0.12812444792935673, 0.99904051532414473))
      if (j == 0): mt = flex.mersenne_twister(seed=4357)
      else: mt.seed(value=4357)
      assert tuple(mt.random_size_t(size=3)) \
          == (2983900864, 1547366158, 1775641839)
      assert approx_equal(mt.random_double(size=3),
        (0.10064729869939604, 0.89184217257908471, 0.20721445761797463))
  assert mt.random_size_t(size=3).size() == 3
  assert mt.random_double(size=3).size() == 3
  for i_trial in range(10):
    a = mt.random_size_t(size=100000, modulus=10)
    assert a.size() == 100000
    assert flex.min(a) == 0
    assert flex.max(a) == 9
    a = a.as_double()
    assert approx_equal(flex.mean(a), 4.5, eps=1.e-1)
    assert approx_equal(
      flex.mean(a*a) - flex.mean(a)*flex.mean(a), 8.25, eps=1.e-1)
  for i_trial in range(10):
    a = mt.random_double(size=100000, factor=10)
    assert a.size() == 100000
    assert flex.min(a) >= 0
    assert flex.max(a) < 10
    assert approx_equal(flex.mean(a), 5, eps=1.e-1)
    assert approx_equal(
      flex.mean(a*a) - flex.mean(a)*flex.mean(a), 8.25, eps=0.2)
  flex.set_random_seed(value=0)
  assert tuple(flex.random_size_t(3)) \
      == (1791095845, 4282876139, 3093770124)
  assert approx_equal(flex.random_double(3),
    (0.9325573593386588, 0.12812444792935673, 0.99904051532414473))
  assert list(flex.random_permutation(size=5)) == [2, 1, 4, 0, 3]
  assert list(flex.random_permutation(size=5)) == [1, 0, 4, 3, 2]
  assert list(flex.random_permutation(size=5)) == [0, 2, 3, 1, 4]
  assert list(flex.random_permutation(size=5)) == [3, 1, 0, 4, 2]
  #
  state = flex.random_generator.getstate()
  r1 = flex.random_size_t(13)
  for i_trial in range(10):
    flex.random_generator.setstate(state=state)
    r2 = flex.random_size_t(13)
    assert r2.all_eq(r1)
  #
  flex.set_random_seed(value=0)
  assert approx_equal(flex.random_double_point_on_sphere(),
    [-0.18280757808732773, -0.96904076208358081, -0.165955990594852])
  assert approx_equal(flex.random_double_point_on_sphere(),
    [-0.0069066582975913487, 0.020242159325144421, -0.99977125036531023])
  assert approx_equal(flex.random_double_point_on_sphere(),
    [0.59191534709710958, 0.38795698109618137, -0.70648821836577391])
  a = matrix.col([0,0,0])
  for i_trial in range(1000):
    p = matrix.col(flex.random_double_point_on_sphere())
    assert approx_equal(p.norm_sq(), 1.0)
    a += p
  a /= 1000
  assert approx_equal(a, [0.0202650, -0.0375843, 0.0009599])
  #
  assert list(flex.random_bool(size=3, threshold=0.5)) == [False, True, False]
  assert list(flex.random_bool(size=3, threshold=0)) == [False, False, False]
  assert list(flex.random_bool(size=3, threshold=1)) == [True, True, True]
  #
  if (boost_version < 105600):
    assert approx_equal(flex.random_double_r3_rotation_matrix(),
      [0.6892647, -0.3238509, 0.6481009,
       0.1310654, -0.8240431, -0.5511577,
       0.712556, 0.4648372, -0.5255381])
  else:
    assert approx_equal(flex.random_double_r3_rotation_matrix(),
      [-0.8415466,  0.1031904, -0.5302366,
       -0.4123652,  0.5113358,  0.7539831,
        0.3489328,  0.8531631, -0.3877608])
  def check_r3r(method):
    for i_trial in range(100):
      r = matrix.sqr(method())
      assert approx_equal(r.determinant(), 1)
      assert r.is_r3_rotation_matrix()
  check_r3r(flex.random_double_r3_rotation_matrix)
  #
  if (boost_version < 105600):
    assert approx_equal(flex.random_double_unit_quaternion(), (
      0.516594, 0.2275523, -0.6385604, 0.5230595))
  else:
    assert approx_equal(flex.random_double_unit_quaternion(), (
      -0.707011, -0.3049360, -0.1297506, -0.6247512))
  for i_trial in range(10):
    assert approx_equal(
      abs(matrix.col(flex.random_double_unit_quaternion())), 1)
  #
  if (boost_version < 105600):
    assert list(flex.random_int_gaussian_distribution(
      size=3,
      mu=-4.56,
      sigma=3.89)) == [-8, -5, -3]
  else:
    assert list(flex.random_int_gaussian_distribution(
      size=3,
      mu=-4.56,
      sigma=3.89)) == [-6, -6, -8]
  #
  if (boost_version < 105600):
    assert approx_equal(flex.random_double_r3_rotation_matrix_arvo_1992(),
      [-0.01872456, -0.8729415, -0.4874654,
       -0.8655858, 0.2581717, -0.4290788,
       0.5004104, 0.4139088, -0.76044])
  else:
    assert approx_equal(flex.random_double_r3_rotation_matrix_arvo_1992(),
      [ 0.6332900, -0.5779739,  0.5146744,
        0.7559849,  0.3196769, -0.5712209,
        0.1656213,  0.7508346,  0.6393878])
  check_r3r(flex.random_double_r3_rotation_matrix_arvo_1992)

def exercise_flex_vec3_double():
  flex.exercise_triple(flex.vec3_double, as_double=True)
  a = flex.vec3_double(((1,2,5), (-2,3,4), (3,4,3)))
  assert a.part_names() == ("x", "y", "z")
  assert approx_equal(a.parts(), [(1,-2,3), (2,3,4), (5,4,3)])
  assert approx_equal(a.min(), (-2.0,2.0,3.0))
  assert approx_equal(a.max(), (3.0,4.0,5.0))
  assert approx_equal(a.sum(), (2.0,9.0,12.0))
  assert approx_equal(a.mean(), (2.0/3,9.0/3,12.0/3))
  weights = flex.double([1,1,1])
  assert approx_equal(a.mean_weighted(weights=weights), (2.0/3,9.0/3,12.0/3))
  weights = flex.double([2,3,5])
  assert approx_equal(a.mean_weighted(weights=weights), (1.1,3.3,3.7))
  a += (10,20,30)
  assert approx_equal(tuple(a), ((11,22,35), (8,23,34), (13,24,33)))
  assert approx_equal(tuple(a+(20,30,10)), ((31,52,45),(28,53,44),(33,54,43)))
  assert approx_equal(tuple(a+a),
    ((2*11,2*22,2*35), (2*8,2*23,2*34), (2*13,2*24,2*33)))
  a -= (10,20,30)
  assert approx_equal(tuple(a), ((1,2,5), (-2,3,4), (3,4,3)))
  b = a.deep_copy()
  b *= 3
  assert approx_equal(tuple(b), ((3,6,15), (-6,9,12), (9,12,9)))
  b += a
  assert approx_equal(tuple(b), ((4,8,20), (-8,12,16), (12,16,12)))
  assert approx_equal(tuple(a-(20,30,10)),
    ((-19,-28,-5),(-22,-27,-6),(-17,-26,-7)))
  assert tuple(a-a) == ((0,0,0),(0,0,0),(0,0,0))
  a += (10,20,30)
  assert approx_equal(tuple(a*2), ((22,44,70), (16,46,68), (26,48,66)))
  assert approx_equal(tuple(-3*a),
    ((-33,-66,-105), (-24,-69,-102), (-39,-72,-99)))
  d = flex.double([3,7,-5])
  assert approx_equal(a*d, [(33,66,105), (56,161,238), (-65,-120,-165)])
  assert approx_equal(d*a, [(33,66,105), (56,161,238), (-65,-120,-165)])
  assert approx_equal(a/flex.double([1,-2,3]),
    [(11,22,35), (-4,-11.5,-17), (4+1/3.,8,11)])
  assert approx_equal(tuple(a*(-1,1,0,1,0,-1,1,-1,1)),
    ((46,-24,13),(49,-26,11),(44,-20,9)))
  assert approx_equal(tuple((-1,1,0,1,0,-1,1,-1,1)*a),
    ((11,-24,24),(15,-26,19),(11,-20,22)))
  x = flex.double([1,2,3])
  y = flex.double([4,5,6])
  z = flex.double([7,8,9])
  a = flex.vec3_double(x,y,z)
  assert approx_equal(a, ((1,4,7), (2,5,8), (3,6,9)))
  assert approx_equal(a.dot(a), (66,93,126))
  assert approx_equal(a.dot((1,1,1)), (12, 15, 18))
  assert approx_equal(a.dot(), (66,93,126))
  assert approx_equal(a.cross(a), [(0,0,0)]*3)
  b = flex.vec3_double(z,x,y)
  assert approx_equal(a.cross(b), [(9, 45, -27), (9, 54, -36), (9, 63, -45)])
  assert approx_equal(a.norms(),flex.sqrt(a.dot()))
  assert approx_equal(a.each_normalize().dot(), [1]*3)
  zoz = flex.vec3_double([(0,0,0),(0,1,0),(0,0,0)])
  try:
    zoz.each_normalize()
  except RuntimeError as e:
    assert str(e) == "flex.vec3_double.each_normalize():" \
      " number of vectors with length zero: 2 of 3"
  else: raise Exception_expected
  assert approx_equal(
    zoz.each_normalize(raise_if_length_zero=False).dot(), [0,1,0])
  assert approx_equal(
    a.max_distance(b)**2,
    max(flex.vec3_double(a.as_double()-b.as_double()).dot()))
  a1 = flex.vec3_double([(0,0,0),(0,1,0),(0,0,2)])
  a2 = flex.vec3_double([(5,0,0),(0,3,2),(2,0,2),(5,5,5)])
  assert approx_equal(a1.min_distance_between_any_pair(a2),2)
  assert approx_equal(a1.min_distance_between_any_pair_with_id(a2),(2.0,2,2))
  assert approx_equal(a1.max_distance_between_any_pair_with_id(a2),(8.66025403784,0,3))
  assert approx_equal(a.sum_sq(), 285)
  assert approx_equal(a.sum_sq(), flex.sum_sq(a.as_double()))
  assert approx_equal(a.norm(), math.sqrt(285))
  assert approx_equal(a.rms_difference(b), math.sqrt(flex.mean((a-b).dot())))
  assert approx_equal(a.rms_difference(b), b.rms_difference(a))
  assert approx_equal(a.rms_difference(a), 0)
  assert approx_equal(b.rms_difference(b), 0)
  assert approx_equal(a.rms_length(), math.sqrt(flex.mean(a.dot())))
  assert approx_equal((a-a).rms_length(), 0)
  for i_trial in range(10):
    for n in [7,10,13]:
      sites_1 = flex.vec3_double(flex.random_double(n*3)*5)
      sites_2 = flex.vec3_double(flex.random_double(n*3)*7)
      m1 = matrix.rec(sites_1.as_double(), (sites_1.size(), 3))
      m2 = matrix.rec(sites_2.as_double(), (sites_2.size(), 3))
      assert approx_equal(
        sites_1.transpose_multiply(sites_2),
        m1.transpose()*m2)
  #
  a = flex.vec3_double([(1,2,5), (-2,3,4), (3,4,3)])
  i = flex.size_t([0,2])
  v = flex.vec3_double([(6,2,-8), (-4,9,2)])
  assert a.add_selected(indices=i, values=v) is a
  assert approx_equal(a, [(7,4,-3), (-2,3,4), (-1,13,5)])
  #
  a = flex.vec3_double([(1,2,3), (-4,-5,6)])
  assert approx_equal(a.round(0), a)
  assert approx_equal(a.round(1), a)
  assert approx_equal(a.iround(), a)
  assert isinstance(a.iround(), flex.vec3_int)
  b = flex.vec3_double([(1.1,2.2,3.3), (-9.9,9.9,10.5), (0.02,0.49999, 0.501)])
  assert approx_equal(b.iround(), [(1, 2, 3), (-10, 10, 11), (0, 0, 1)])
  assert approx_equal(b.round(0), b.iround())
  assert approx_equal(b.round(1),
                      [(1.1, 2.2, 3.3), (-9.9, 9.9, 10.5), (0.0, 0.5, 0.5)],
                      eps=1e-15)
  assert approx_equal(b.round(2),
                      [(1.1, 2.2, 3.3), (-9.9, 9.9, 10.5), (0.02, 0.5, 0.5)],
                      eps=1e-15)
  assert approx_equal(b.round(3),
                      [(1.1, 2.2, 3.3), (-9.9, 9.9, 10.5), (0.02, 0.5, 0.501)],
                      eps=1e-15)
  assert approx_equal(b.round(5), b, eps=1e-15)
  #
  a = flex.vec3_double([(1,2,5), (-2,3,4), (3,4,3)])
  assert approx_equal(a, a.rotate_around_origin((1,1,1), 2*math.pi))
  assert approx_equal(
    a, a.rotate_around_origin((1,1,1), flex.double(3, 2*math.pi)))
  axes = flex.vec3_double([(1,0,0), (0,1,0), (0,0,1)])
  assert approx_equal(
    a, a.rotate_around_origin(axes, flex.double(3, 2*math.pi)))
  import random
  for i in range(100):
    axis = matrix.col([random.randint(-5, 5) for j in range(3)])
    if axis.length() == 0: continue
    angle = random.uniform(0, 2*math.pi)
    expected = [matrix.col(point).rotate_around_origin(axis, angle)
                for point in a]
    assert approx_equal(a.rotate_around_origin(axis, angle), expected)
    angles = flex.double([random.uniform(0, 2*math.pi)
                                 for j in range(100)])
    expected = [matrix.col(a[j]).rotate_around_origin(axis, angles[j])
                for j in range(len(a))]
    assert approx_equal(a.rotate_around_origin(axis, angles), expected)
  axis = matrix.col([0,0,0])
  try: a.rotate_around_origin(axis, angle)
  except RuntimeError: pass
  else: raise Exception_expected
  try: a.rotate_around_origin(axis, angles)
  except RuntimeError: pass
  else: raise Exception_expected

  a = flex.vec3_double([(1,2,5), (-2,3,4), (3,4,3)])
  t = a.angle(a)
  eps = 1e-7
  for tt in t:
    assert(abs(tt - 0) < eps)
  b = matrix.col((4,5,6))
  c = flex.vec3_double([b] * 3)
  t1 = a.angle(b)
  t2 = a.angle(c)
  for tt1, tt2 in zip(t1,t2):
    assert(abs(tt1-tt2) < eps)
  for aa, tt1 in zip(a, t1):
    tt2 = matrix.col(aa).angle(b)
    assert(abs(tt1-tt2) < eps)



def exercise_flex_vec2_double():
  #flex.exercise_triple(flex.vec2_double, as_double=True)
  a = flex.vec2_double(((1,2), (-2,3), (3,4)))
  assert approx_equal(a.parts(), [(1,-2,3), (2,3,4)])
  assert approx_equal(a.min(), (-2.0,2.0))
  assert approx_equal(a.max(), (3.0,4.0))
  assert approx_equal(a.sum(), (2.0,9.0))
  assert approx_equal(a.mean(), (2.0/3,9.0/3))
  weights = flex.double([1,1,1])
  assert approx_equal(a.mean_weighted(weights=weights), (2.0/3,9.0/3))
  weights = flex.double([2,3,5])
  assert approx_equal(a.mean_weighted(weights=weights), (1.1,3.3))
  a += (10,20)
  assert approx_equal(tuple(a), ((11,22), (8,23), (13,24)))
  assert approx_equal(tuple(a+(20,30)), ((31,52),(28,53),(33,54)))
  assert approx_equal(tuple(a+a),
    ((2*11,2*22), (2*8,2*23), (2*13,2*24)))
  a -= (10,20)
  assert approx_equal(tuple(a), ((1,2), (-2,3), (3,4)))
  b = a.deep_copy()
  b *= 3
  assert approx_equal(tuple(b), ((3,6), (-6,9), (9,12)))
  b += a
  assert approx_equal(tuple(b), ((4,8), (-8,12), (12,16)))
  assert approx_equal(tuple(a-(20,30)),
    ((-19,-28),(-22,-27),(-17,-26)))
  assert tuple(a-a) == ((0,0),(0,0),(0,0))
  a += (10,20)
  assert approx_equal(tuple(a*2), ((22,44), (16,46), (26,48)))
  assert approx_equal(tuple(-3*a),
    ((-33,-66), (-24,-69), (-39,-72)))
  assert approx_equal(a/flex.double([1,-2,3]),
    [(11,22), (-4,-11.5), (4+1/3.,8)])
  assert approx_equal(tuple(a*(-1,1,0,-1)),
    ((-11,-11),(-8,-15),(-13,-11)))
  assert approx_equal(tuple((-1,0,1,-1)*a),
    ((-11,-11),(-8,-15),(-13,-11)))
  x = flex.double([1,2,3])
  y = flex.double([4,5,6])
  a = flex.vec2_double(x,y)
  assert approx_equal(tuple(a), ((1,4), (2,5), (3,6)))
  assert approx_equal(tuple(a.dot(a)), (17,29,45))
  assert approx_equal(tuple(a.dot()), (17,29,45))
  b = flex.vec2_double(y,x)
  assert approx_equal(a.each_normalize().dot(), [1]*3)
  zoz = flex.vec2_double([(0,0),(0,1),(1,0)])
  try:
    zoz.each_normalize()
  except RuntimeError as e:
    assert str(e) == "flex.vec2_double.each_normalize():" \
      " number of vectors with length zero: 1 of 3"
  else: raise Exception_expected
  assert approx_equal(
    zoz.each_normalize(raise_if_length_zero=False).dot(), [0,1,1])
  assert approx_equal(
    a.max_distance(b)**2,
    max(flex.vec2_double(a.as_double()-b.as_double()).dot()))
  a1 = flex.vec2_double([(0,0),(3,2),(0,0)])
  a2 = flex.vec2_double([(5,0),(0,3),(2,0),(5,5)])
  assert approx_equal(a1.min_distance_between_any_pair(a2),2)
  assert approx_equal(a1.min_distance_between_any_pair_with_id(a2),(2.0, 0, 2))
  assert approx_equal(a.sum_sq(), 91)
  assert approx_equal(a.sum_sq(), flex.sum_sq(a.as_double()))
  assert approx_equal(a.norm(), math.sqrt(91.))
  assert approx_equal(a.rms_difference(b), math.sqrt(flex.mean((a-b).dot())))
  assert approx_equal(a.rms_difference(b), b.rms_difference(a))
  assert approx_equal(a.rms_difference(a), 0)
  assert approx_equal(b.rms_difference(b), 0)
  assert approx_equal(a.rms_length(), math.sqrt(flex.mean(a.dot())))
  assert approx_equal((a-a).rms_length(), 0)
  for i_trial in range(10):
    for n in [7,10,13]:
      sites_1 = flex.vec2_double(flex.random_double(n*2)*5)
      sites_2 = flex.vec2_double(flex.random_double(n*2)*7)
      m1 = matrix.rec(sites_1.as_double(), (sites_1.size(), 2))
      m2 = matrix.rec(sites_2.as_double(), (sites_2.size(), 2))
      assert approx_equal(
        sites_1.transpose_multiply(sites_2),
        m1.transpose()*m2)
  #
  a = flex.vec3_double([(1,2,5), (-2,3,4), (3,4,3)])
  i = flex.size_t([0,2])
  v = flex.vec3_double([(6,2,-8), (-4,9,2)])
  assert a.add_selected(indices=i, values=v) is a
  assert approx_equal(a, [(7,4,-3), (-2,3,4), (-1,13,5)])

  #excercise constructor from 2D array focus to array indices as double:
  array_focus = (3,4) # 3 slow, 4 fast
  B = flex.vec2_double(array_focus)
  assert list(B) == [
    (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0),
    (1.0, 0.0), (1.0, 1.0), (1.0, 2.0), (1.0, 3.0),
    (2.0, 0.0), (2.0, 1.0), (2.0, 2.0), (2.0, 3.0)]

  #print ("test vec2_double.from_parts(), .distance_matrix()")
  x = flex.random_double(size=25)
  y = flex.random_double(size=25)
  x2 = flex.random_double(size=15)
  y2 = flex.random_double(size=15)
  V = flex.vec2_double(x,y); V2 = flex.vec2_double(x2,y2)
  DIST = V.distance_matrix(V2)
  for i in range(len(x)):
    for j in range(len(x2)):
      pydist = math.sqrt( (x[i]-x2[j])**2 + (y[i]-y2[j])**2 )
      assert approx_equal(pydist, DIST[i,j])

  #
  a = flex.vec2_double([(1,2), (-2,3), (3,4)])
  assert approx_equal(a, a.rotate_around_origin(2*math.pi))
  assert approx_equal(
    a, a.rotate_around_origin(flex.double(3, 2*math.pi)))

def exercise_flex_vec3_int():
  flex.exercise_triple(flex.vec3_int)
  a = flex.vec3_int(((1,2,5), (-2,3,4), (3,4,3)))
  assert approx_equal(a.as_vec3_double(), [(1,2,5), (-2,3,4), (3,4,3)])

def exercise_flex_sym_mat3_double():
  a = flex.sym_mat3_double()
  assert a.size() == 0
  a = flex.sym_mat3_double(132)
  for x in a:
    assert x == (0,0,0,0,0,0)
  a = flex.sym_mat3_double(((1,2,3,4,5,6), (2,3,4,5,6,7)))
  assert a.size() == 2
  assert tuple(a) == ((1,2,3,4,5,6), (2,3,4,5,6,7))
  assert approx_equal(a.norms(), (12.961481396815721, 15.779733838059499))
  p = pickle.dumps(a)
  b = pickle.loads(p)
  assert tuple(a) == tuple(b)
  assert approx_equal(tuple(a.as_double()), (1,2,3,4,5,6,2,3,4,5,6,7))
  b = flex.sym_mat3_double(a.as_double())
  assert tuple(a) == tuple(b)
  b = flex.sym_mat3_double([(7,4,-8,2,4,3), (6,2,0,3,5,1)])
  assert approx_equal(a+b, [(8,6,-5,6,9,9), (8,5,4,8,11,8)])
  assert approx_equal(a-b, [(-6,-2,11,2,1,3), (-4,1,4,2,1,6)])
  b *= 3
  assert approx_equal(b, [(21,12,-24,6,12,9), (18,6,0,9,15,3)])
  a00, a11, a22, a12, a13, a23 = [flex.double((0,1,2)),
                                  flex.double((1,2,3)),
                                  flex.double((2,3,4)),
                                  flex.double((3,4,5)),
                                  flex.double((4,5,6)),
                                  flex.double((5,6,7))]
  c = flex.sym_mat3_double(a00, a11, a22, a12, a13, a23)
  assert c.size() == a00.size()
  assert approx_equal(c, ((0,1,2,3,4,5), (1,2,3,4,5,6), (2,3,4,5,6,7)))

def exercise_flex_mat3_double():
  a = flex.mat3_double()
  assert a.size() == 0
  a = flex.mat3_double(132)
  assert a.size() == 132
  for x in a:
    assert x == (0, 0, 0, 0, 0, 0, 0, 0, 0)
  a = flex.mat3_double([(1, 2, 3, 4, 5, 6, 7, 8, 9)])
  assert a.size() == 1
  assert tuple(a) == ((1, 2, 3, 4, 5, 6, 7, 8, 9),)
  p = pickle.dumps(a)
  b = pickle.loads(p)
  assert(tuple(a) == tuple(b))
  assert approx_equal(tuple(a*2), ((2., 4., 6., 8., 10., 12., 14., 16., 18.),))
  assert approx_equal(tuple(a*flex.double((2,))),
    ((2., 4., 6., 8., 10., 12., 14., 16., 18.),))
  assert approx_equal(a * (9, 8, 7, 6, 5, 4, 3, 2, 1),
    ((30.0, 24.0, 18.0, 84.0, 69.0, 54.0, 138.0, 114.0, 90.0),))
  b = flex.mat3_double([(9, 8, 7, 6, 5, 4, 3, 2, 1)])
  assert approx_equal(a * b,
    ((30.0, 24.0, 18.0, 84.0, 69.0, 54.0, 138.0, 114.0, 90.0),))
  v = (1,2,3)
  assert approx_equal(tuple(a*v), ((14., 32., 50.),))
  v = flex.vec3_double([(1,2,3)])
  assert approx_equal(tuple(a*v), ((14., 32., 50.),))

def exercise_flex_tiny_size_t_2():
  a = flex.tiny_size_t_2()
  assert a.size() == 0
  a = flex.tiny_size_t_2(132)
  for x in a:
    assert x == (0,0)
  a = flex.tiny_size_t_2(((1,2), (2,3), (3,4)))
  assert a.size() == 3
  assert tuple(a) == ((1,2), (2,3), (3,4))
  assert tuple(a.column(0)) == (1,2,3)
  assert tuple(a.column(1)) == (2,3,4)

def exercise_histogram():
  x = flex.double(range(20))
  h = flex.histogram(data=x)
  assert h.slots().size() == 1000
  h = flex.histogram(x, n_slots=5)
  assert approx_equal(h.data_min(), 0)
  assert approx_equal(h.data_max(), 19)
  assert approx_equal(h.slot_width(), 19/5.)
  assert tuple(h.slots()) == (4,4,4,4,4)
  assert approx_equal(h.slot_centers(), (1.9, 5.7, 9.5, 13.3, 17.1))
  assert h.get_i_slot(10) == 2
  assert h.get_i_slot(16.2) == 4
  assert h.n_out_of_slot_range() == 0
  assert approx_equal(h.get_cutoff(max_points=15), 7.60038)
  assert approx_equal(h.get_cutoff(15, relative_tolerance=0.1), 7.98)
  h.update(2)
  assert tuple(h.slots()) == (5,4,4,4,4)
  h.update(-12) # out of range
  h.update(21) # out of range
  assert tuple(h.slots()) == (5,4,4,4,4)
  # empty starting histogram
  h2 = flex.histogram(flex.double(), -2, 4, 4)
  h2.update(3)
  h2.update(4.0000001)
  h2.update(4.01)
  assert tuple(h2.slots()) == (0, 0, 0, 2)
  h2.update(4.01, relative_tolerance=1e-1)
  assert tuple(h2.slots()) == (0, 0, 0, 3)
  h3 = flex.histogram(flex.double((1,2,3,4,5)), -2, 4, 4)
  h2.update(h3)
  assert tuple(h2.slots()) == (0, 0, 2, 5)
  y = flex.double(range(-3,23))
  hy = flex.histogram(other=h, data=y)
  assert approx_equal(hy.data_min(), 0)
  assert approx_equal(hy.data_max(), 19)
  assert approx_equal(hy.slot_width(), 19/5.)
  assert tuple(hy.slots()) == (4,4,4,4,4)
  assert hy.n_out_of_slot_range() == 6
  hy = flex.histogram(other=h, data=y, relative_tolerance=0.5)
  assert tuple(hy.slots()) == (5,4,4,4,5)
  assert hy.n_out_of_slot_range() == 4
  hy = flex.histogram(other=h, data=y, relative_tolerance=1)
  assert tuple(hy.slots()) == (7,4,4,4,7)
  assert hy.n_out_of_slot_range() == 0
  assert hy.as_str(prefix="*") == """\
*0 - 3.8: 7
*3.8 - 7.6: 4
*7.6 - 11.4: 4
*11.4 - 15.2: 4
*15.2 - 19: 7"""
  hy = flex.histogram(
    data=y, data_min=2, data_max=10, n_slots=4, relative_tolerance=1.e-4)
  s = StringIO()
  hy.show(f=s, prefix="&")
  assert s.getvalue() == """\
&2 - 4: 2
&4 - 6: 2
&6 - 8: 2
&8 - 10: 3
"""
  assert hy.n_out_of_slot_range() == 17
  centers = [info.center() for info in hy.slot_infos()]
  assert approx_equal(centers, [3,5,7,9])
  p = pickle.dumps(hy)
  l = pickle.loads(p)
  t = StringIO()
  l.show(f=t, prefix="&")
  assert not show_diff(t.getvalue(), s.getvalue())
  assert l.n_out_of_slot_range() == 17

  # treatment of data with NaN in - cctbx/cctbx_project#478
  d = flex.log10(flex.double(range(-10, 100)) + 0.5)
  h = flex.histogram(d, data_min=-10, data_max=10, n_slots=20)
  assert h.slots()[0] == 0

def exercise_weighted_histogram():
  x = flex.double(range(20))
  w = 0.5 * flex.double(range(20))
  h = flex.weighted_histogram(data=x, weights=w)
  assert h.slots().size() == 1000
  h = flex.weighted_histogram(data=x)
  assert h.slots().size() == 1000
  h = flex.weighted_histogram(x, n_slots=5)
  assert approx_equal(h.data_min(), 0)
  assert approx_equal(h.data_max(), 19)
  assert approx_equal(h.slot_width(), 19/5.)
  assert tuple(h.slots()) == (4,4,4,4,4)
  assert approx_equal(h.slot_centers(), (1.9, 5.7, 9.5, 13.3, 17.1))
  assert h.get_i_slot(10) == 2
  assert h.get_i_slot(16.2) == 4
  assert h.n_out_of_slot_range() == 0
  assert approx_equal(h.get_cutoff(max_points=15), 7.60038)
  assert approx_equal(h.get_cutoff(15, relative_tolerance=0.1), 7.98)
  h.update(2)
  assert tuple(h.slots()) == (5,4,4,4,4)
  h.update(-12) # out of range
  h.update(21) # out of range
  assert tuple(h.slots()) == (5,4,4,4,4)
  h = flex.weighted_histogram(x, w, n_slots=5)
  assert approx_equal(h.data_min(), 0)
  assert approx_equal(h.data_max(), 19)
  assert approx_equal(h.slot_width(), 19/5.)
  assert approx_equal(tuple(h.slots()), (3, 11, 19, 27, 35))
  # empty starting weighted_histogram
  h2 = flex.weighted_histogram(flex.double(), -2, 4, 4)
  h2.update(3)
  h2.update(4.0000001)
  h2.update(4.01)
  assert tuple(h2.slots()) == (0, 0, 0, 2)
  h2.update(4.01, relative_tolerance=1e-1)
  assert tuple(h2.slots()) == (0, 0, 0, 3)
  h3 = flex.weighted_histogram(flex.double((1,2,3,4,5)), -2, 4, 4)
  h2.update(h3)
  assert tuple(h2.slots()) == (0, 0, 2, 5)
  y = flex.double(range(-3,23))
  hy = flex.weighted_histogram(other=h, data=y)
  assert approx_equal(hy.data_min(), 0)
  assert approx_equal(hy.data_max(), 19)
  assert approx_equal(hy.slot_width(), 19/5.)
  assert tuple(hy.slots()) == (4,4,4,4,4)
  assert hy.n_out_of_slot_range() == 6
  hy = flex.weighted_histogram(other=h, data=y, relative_tolerance=0.5)
  assert tuple(hy.slots()) == (5,4,4,4,5)
  assert hy.n_out_of_slot_range() == 4
  hy = flex.weighted_histogram(other=h, data=y, relative_tolerance=1)
  assert tuple(hy.slots()) == (7,4,4,4,7)
  assert hy.n_out_of_slot_range() == 0
  s = StringIO()
  hy.show(f=s, prefix="*")
  assert s.getvalue() == """\
*0 - 3.8: 7
*3.8 - 7.6: 4
*7.6 - 11.4: 4
*11.4 - 15.2: 4
*15.2 - 19: 7
"""
  hy = flex.weighted_histogram(
    data=y, data_min=2, data_max=10, n_slots=4, relative_tolerance=1.e-4)
  s = StringIO()
  hy.show(f=s, prefix="&")
  assert s.getvalue() == """\
&2 - 4: 2
&4 - 6: 2
&6 - 8: 2
&8 - 10: 3
"""
  assert hy.n_out_of_slot_range() == 17
  centers = [info.center() for info in hy.slot_infos()]
  assert approx_equal(centers, [3,5,7,9])
  p = pickle.dumps(hy)
  l = pickle.loads(p)
  t = StringIO()
  l.show(f=t, prefix="&")
  assert not show_diff(t.getvalue(), s.getvalue())
  assert l.n_out_of_slot_range() == 17

def exercise_show_count_stats():
  def check(counts, prefix="", group_size=10, expected=None):
    sio = StringIO()
    flex.show_count_stats(
      counts=flex.size_t(counts),
      group_size=group_size,
      out=sio,
      prefix=prefix)
    assert not show_diff(sio.getvalue(), expected)
  check([0], expected="""\
   None:  1  1.00000
""")
  check([0,0], expected="""\
   None:  2  1.00000
""")
  check([1], expected="""\
>=    1:  1  1.00000
""")
  check([1,1], expected="""\
>=    1:  2  1.00000
""")
  check([1,0], expected="""\
>=    1:  1  0.50000
   None:  1  0.50000
""")
  check([1,1,11,11], expected="""\
>=   10:  2  0.50000
>=    1:  4  1.00000
""")
  check([1,1,11,11,0], expected="""\
>=   10:  2  0.40000
>=    1:  4  0.80000
   None:  1  0.20000
""")
  check([10,20], expected="""\
>=   20:  1  0.50000
>=   10:  2  1.00000
""")
  check([10,20,0], expected="""\
>=   20:  1  0.33333
>=   10:  2  0.66667
   None:  1  0.33333
""")
  check(range(34), prefix=":|", expected="""\
:|>=   30:   4  0.11765
:|>=   20:  14  0.41176
:|>=   10:  24  0.70588
:|>=    1:  33  0.97059
:|   None:   1  0.02941
""")
  check(range(23,74), group_size=12, expected="""\
>=   72:   2  0.03922
>=   60:  14  0.27451
>=   48:  26  0.50980
>=   36:  38  0.74510
>=   24:  50  0.98039
>=   12:  51  1.00000
""")
  check([0]*93+list(range(0,10001,100)), group_size=5000, expected="""\
>= 10000:    1  0.00515
>=  5000:   51  0.26289
>=     1:  100  0.51546
    None:   94  0.48454
""")

def simple_linear_regression(x_obs, y_obs, w_obs):
  assert len(x_obs) == len(y_obs)
  assert len(x_obs) == len(w_obs)
  w = 0
  x = 0
  y = 0
  x2 = 0
  y2 = 0
  xy = 0
  for i in range(len(x_obs)):
    w += w_obs[i]
    x += x_obs[i]*w_obs[i]
    y += y_obs[i]*w_obs[i]
    x2 += x_obs[i]**2*w_obs[i]
    y2 += y_obs[i]**2*w_obs[i]
    xy += x_obs[i]*y_obs[i]*w_obs[i]
  determinant = w*x2 - x**2
  a = x2*y - x*xy
  a /= determinant
  b = w*xy - x*y
  b /= determinant
  return a, b

def exercise_linear_regression():
  x = flex.double((1,2,3))
  r = flex.linear_regression(x=x, y=x, epsilon=1.e-6)
  assert r.is_well_defined()
  assert approx_equal(r.y_intercept(), 0)
  assert approx_equal(r.slope(), 1)
  y = flex.double((-1./2+1,-2./2+1,-3./2+1))
  r = flex.linear_regression(x, y)
  assert r.is_well_defined()
  assert approx_equal(r.y_intercept(), 1)
  assert approx_equal(r.slope(), -1./2)
  y = flex.double((0,0,0))
  r = flex.linear_regression(x, y)
  assert r.is_well_defined()
  assert approx_equal(r.y_intercept(), 0)
  assert approx_equal(r.slope(), 0)
  r = flex.linear_regression(y, y)
  assert not r.is_well_defined()
  s = StringIO()
  r.show_summary(f=s, prefix="/:")
  assert s.getvalue() == """\
/:is_well_defined: %s
/:y_intercept: 0.0
/:slope: 0.0
""" % str(False)
  y = flex.double((-1./2+1,-2./2+1,-3./2+1))
  for weight in [1,math.pi]:
    weights = flex.double(3, weight)
    r = flex.linear_regression(x=x, y=y, epsilon=1.e-6)
    rw = flex.linear_regression(x=x, y=y, weights=weights, epsilon=1.e-6)
    assert rw.is_well_defined()
    assert approx_equal(rw.y_intercept(), r.y_intercept())
    assert approx_equal(rw.slope(), r.slope())
    a, b = simple_linear_regression(x, y, weights)
    assert approx_equal(a, r.y_intercept())
    assert approx_equal(b, r.slope())
  for i_trial in range(100):
    x = flex.random_double(size=10)
    y = x + flex.random_double(size=10) * 0.2 - 0.1 + 3
    weights = flex.random_double(size=10)
    rw = flex.linear_regression(x=x, y=y, weights=weights)
    a, b = simple_linear_regression(x, y, weights)
    assert approx_equal(a, rw.y_intercept())
    assert approx_equal(b, rw.slope())
    assert approx_equal(rw.y_intercept(), 3, 1)

def exercise_linear_correlation():
  x = flex.double((1,2,3))
  c = flex.linear_correlation(x=x, y=x, epsilon=1.e-6)
  assert c.is_well_defined()
  assert c.n() == 3
  assert approx_equal(c.mean_x(), 2)
  assert approx_equal(c.mean_y(), 2)
  assert approx_equal(c.numerator(), 2)
  assert approx_equal(c.sum_denominator_x(), 2)
  assert approx_equal(c.sum_denominator_y(), 2)
  assert approx_equal(c.denominator(), 2)
  assert approx_equal(c.coefficient(), 1)
  y = flex.double((-1./2+1,-2./2+1,-3./2+1))
  c = flex.linear_correlation(x, y)
  assert c.is_well_defined()
  assert approx_equal(c.coefficient(), -1)
  y = flex.double((0,0,0))
  c = flex.linear_correlation(x, y)
  assert c.is_well_defined()
  assert approx_equal(c.coefficient(), 0)
  c = flex.linear_correlation(y, y)
  assert c.is_well_defined()
  c = flex.linear_correlation(flex.double(), flex.double())
  assert not c.is_well_defined()
  s = StringIO()
  c.show_summary(f=s)
  assert s.getvalue() == """\
is_well_defined: %s
mean_x: 0.0
mean_y: 0.0
coefficient: 0.0
""" % str(False)

def exercise_mean_and_variance():
  x = flex.double((1,2,3))
  for w in (None, flex.double((1,1,1))):
    if (w is None):
      mv = flex.mean_and_variance(x)
      assert not mv.have_weights()
    else:
      mv = flex.mean_and_variance(x, w)
      assert mv.have_weights()
    assert approx_equal(mv.mean(), 2)
    assert approx_equal(mv.sum_weights(), 3)
    assert approx_equal(mv.sum_weights_sq(), 3)
    assert approx_equal(mv.sum_weights_values(), 6)
    assert approx_equal(mv.sum_weights_delta_sq(), 2)
    assert approx_equal(mv.gsl_stats_wvariance(), 1)
    assert approx_equal(mv.gsl_stats_wsd(), 1)
    assert approx_equal(
      mv.standard_error_of_mean_calculated_from_sample_weights(),
      1/3**0.5)
    if (not mv.have_weights()):
      assert approx_equal(mv.unweighted_sample_variance(), 1)
      assert approx_equal(mv.unweighted_sample_standard_deviation(), 1)
      assert approx_equal(mv.unweighted_standard_error_of_mean(), 1/3**0.5)
  w = flex.double((1,3,2))
  mv = flex.mean_and_variance(x, w)
  assert approx_equal(mv.mean(), 13/6.)
  assert approx_equal(mv.sum_weights(), 6)
  assert approx_equal(mv.sum_weights_sq(), 14)
  assert approx_equal(mv.sum_weights_values(), 13)
  assert approx_equal(mv.sum_weights_delta_sq(), 17/6.)
  assert approx_equal(mv.gsl_stats_wvariance(),
    (6./(36-14))*((1*(1-13/6.)**2)+(3*(2-13/6.)**2)+(2*(3-13/6.)**2)))
  assert approx_equal(mv.gsl_stats_wsd(), math.sqrt(17/22.))
  assert approx_equal(
    mv.standard_error_of_mean_calculated_from_sample_weights(),
    1/6**0.5)
  x = flex.double((1,1,1))
  mv = flex.mean_and_variance(x)
  assert approx_equal(mv.mean(), 1)
  assert approx_equal(mv.gsl_stats_wvariance(), 0)
  assert approx_equal(
    mv.standard_error_of_mean_calculated_from_sample_weights(),
    1/3**0.5)
  assert approx_equal(mv.unweighted_sample_variance(), 0)
  assert approx_equal(mv.unweighted_sample_standard_deviation(), 0)
  assert approx_equal(mv.unweighted_standard_error_of_mean(), 0)

def exercise_linear_interpolation():
  for flex_type in (flex.float, flex.double):
    tab_x = flex_type([1,2,3,4])
    tab_y = flex_type([10,20,30,40])
    assert approx_equal(flex.linear_interpolation(tab_x, tab_y, 3.5, 1.e-6),35)
    assert approx_equal(flex.linear_interpolation(tab_x, tab_y, 2.1), 21)
    assert approx_equal(flex.linear_interpolation(tab_x, tab_y, 1), 10)
    assert approx_equal(flex.linear_interpolation(tab_x, tab_y, 1-1.e-6), 10,
      eps=1.e-4)
    assert approx_equal(flex.linear_interpolation(tab_x, tab_y, 1+1.e-6), 10,
      eps=1.e-4)
    assert approx_equal(flex.linear_interpolation(tab_x, tab_y, 4), 40)
    assert approx_equal(flex.linear_interpolation(tab_x, tab_y, 4+1.e-6), 40,
      eps=1.e-4)
    assert approx_equal(flex.linear_interpolation(tab_x, tab_y, 4-1.e-6), 40,
      eps=1.e-4)
    for i in range(11):
      x = 1+3*i/10.
      assert approx_equal(flex.linear_interpolation(tab_x, tab_y, x), 10*x)
    for x in [0,5]:
      try: flex.linear_interpolation(tab_x, tab_y, 0)
      except KeyboardInterrupt: raise
      except Exception: pass
      else: raise Exception_expected
    x = flex_type([1.3,2.4,3.6])
    assert approx_equal(flex.linear_interpolation(tab_x, tab_y, x, 1.e-6),
      [13,24,36])
    assert approx_equal(flex.linear_interpolation(tab_x, tab_y, x),
      [13,24,36])
    tab_x = flex_type([2,4,8,16])
    tab_y = flex_type([4,6,8,10])
    x = flex_type([2,3,5,6,8,11,12,16])
    assert approx_equal(flex.linear_interpolation(tab_x, tab_y, x),
      [4, 5, 6.5, 7, 8, 8.75, 9, 10])
    tab_x = flex_type([2,4,8,16])
    tab_y = flex_type([4,-6,8,-10])
    for eps in [1.e-6,-1.e-6]:
      x = flex_type([2,3,5,6,8,11,12,16]) + eps
      assert approx_equal(flex.linear_interpolation(tab_x, tab_y, x),
        [4, -1, -6+1/4.*14, -6+2/4.*14, 8, 8-18*3./8, 8-18*4./8, -10],
      eps=1.e-4)

def exercise_loops():
  points = []
  loop = flex.nested_loop(end=(2,3))
  while (not loop.over()):
    points.append(loop())
    loop.incr()
  assert points == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
  points = []
  for index in flex.nested_loop((2,3)):
    points.append(index)
  assert points == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
  points = []
  for index in flex.nested_loop(end=(2,3), open_range=True):
    points.append(index)
  assert points == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
  points = []
  for index in flex.nested_loop((2,3), False):
    points.append(index)
  assert points == [(0,0),(0,1),(0,2),(0,3),
                    (1,0),(1,1),(1,2),(1,3),
                    (2,0),(2,1),(2,2),(2,3)]
  points = []
  for index in flex.nested_loop(begin=(1,2),end=(3,4)):
    points.append(index)
  assert points == [(1, 2), (1, 3), (2, 2), (2, 3)]
  points = []
  for index in flex.nested_loop(begin=(1,2),end=(3,4),open_range=True):
    points.append(index)
  assert points == [(1, 2), (1, 3), (2, 2), (2, 3)]
  points = []
  for index in flex.nested_loop((1,2),(3,4),False):
    points.append(index)
  assert points == [(1,2),(1,3),(1,4),(2,2),(2,3),(2,4),(3,2),(3,3),(3,4)]
  points = []
  for index in flex.nested_loop((1,2),(1,2),False):
    points.append(index)
  assert points == [(1, 2)]
  points = []
  for index in flex.nested_loop((1,2),(1,2)):
    points.append(index)
  assert points == []
  points = []
  for index in flex.nested_loop([],[]):
    points.append(index)
  assert points == []
  #
  n_trials = 2
  nl = flex.nested_loop
  t0 = time.time()
  for i_trial in range(n_trials):
    c = list(nl(begin=[3,-5,8,10,-7], end=[7,0,13,12,-3], open_range=False))
  if (n_trials > 2):
    print("C++ nested_loop: %.2f s" % (time.time()-t0))
  nl = libtbx.math_utils.nested_loop
  t0 = time.time()
  for i_trial in range(n_trials):
    p = [tuple(i) for i in
      nl(begin=[3,-5,8,10,-7], end=[7,0,13,12,-3], open_range=False)]
  if (n_trials > 2):
    print("Python nested_loop: %.2f s" % (time.time()-t0))
  assert p == c

def exercise_extract_attributes():
  class group(object):
    def __init__(self, a, b):
      self.a = a
      self.b = b
  groups = []
  for i in range(3):
    groups.append(group(i, i+1))
  for array in [groups, tuple(groups)]:
    eas = flex.extract_double_attributes(
      array=array, attribute_name="a", none_substitute=None)
    assert approx_equal(eas, [0,1,2])
    ebs = flex.extract_double_attributes(
      array=array, attribute_name="b", none_substitute=None)
    assert approx_equal(ebs, [1,2,3])
  groups[1].a = None
  groups[2].b = None
  for array in [groups, tuple(groups)]:
    eas = flex.extract_double_attributes(
      array=array, attribute_name="a", none_substitute=3)
    assert approx_equal(eas, [0,3,2])
    ebs = flex.extract_double_attributes(
      array=array, attribute_name="b", none_substitute=-4)
    assert approx_equal(ebs, [1,2,-4])

def exercise_exceptions():
  f = flex.double(flex.grid((2,3)))
  try: f.assign(1, 0)
  except RuntimeError as e:
    assert str(e) == "Array must be 0-based 1-dimensional."
  else:
    raise AssertionError("No exception or wrong exception.")
  try: f.append(0)
  except RuntimeError as e:
    assert str(e) == "Array must be 0-based 1-dimensional."
  else:
    raise AssertionError("No exception or wrong exception.")
  try: f[(2,0)]
  except IndexError as e:
    assert str(e) == "Index out of range."
  else:
    raise AssertionError("No exception or wrong exception.")

def exercise_matrix():
  for ag,bg in [[(0,0),(0,0)],[(0,1),(1,0)],[(1,0),(0,2)]]:
    a = flex.double(flex.grid(ag))
    b = flex.double(flex.grid(bg))
    c = a.matrix_multiply(b)
    assert c.focus() == (ag[0],bg[1])
    assert c.all_eq(0)
    c = a.matrix_transpose().matrix_transpose_multiply(b)
    assert c.focus() == (ag[0],bg[1])
    assert c.all_eq(0)
    c = a.matrix_multiply_transpose(b.matrix_transpose())
    assert c.focus() == (ag[0],bg[1])
    assert c.all_eq(0)
    for m in [a,b]:
      mtm = m.matrix_transpose_multiply_as_packed_u()
      n = m.focus()[1]
      assert mtm.size() == n*(n+1)//2
      assert mtm.all_eq(0)
  a = flex.double(range(1,7))
  a.resize(flex.grid(3,2))
  b = flex.double(range(1,7))
  b.resize(flex.grid(2,3))
  for c in [a.matrix_multiply(b),
            a.matrix_transpose().matrix_transpose_multiply(b),
            a.matrix_multiply_transpose(b.matrix_transpose())]:
    assert c.focus() == (3,3)
    assert list(c) == [9, 12, 15, 19, 26, 33, 29, 40, 51]
  for a_n_rows in range(1,4):
    for a_n_columns in range(1,4):
      a = flex.random_double(size=a_n_rows*a_n_columns)
      b = flex.random_double(size=a_n_rows*a_n_columns)
      c = a.matrix_multiply(b)
      d = matrix.row(a) * matrix.col(b)
      assert d.n == (1,1)
      assert approx_equal(c, d[0])
      assert approx_equal(a.dot(b), matrix.row(a).dot(matrix.col(b)))
      a.reshape(flex.grid(a_n_rows, a_n_columns))
      for b_n_columns in range(1,4):
        b = flex.random_double(size=a_n_columns*b_n_columns)
        b.reshape(flex.grid(a_n_columns, b_n_columns))
        d = matrix.rec(a, a.focus()) * matrix.rec(b, b.focus())
        for c in [a.matrix_multiply(b),
                  a.matrix_transpose().matrix_transpose_multiply(b),
                  a.matrix_multiply_transpose(b.matrix_transpose()),
                  (b.matrix_transpose().matrix_multiply(
                   a.matrix_transpose())).matrix_transpose(),
                  (b.matrix_transpose_multiply(
                   a.matrix_transpose())).matrix_transpose(),
                  (b.matrix_transpose().matrix_multiply_transpose(
                   a)).matrix_transpose()]:
          assert c.focus() == d.n
          assert approx_equal(c, d)
        ata = a.matrix_transpose_multiply_as_packed_u() \
          .matrix_packed_u_as_symmetric()
        assert approx_equal(ata, a.matrix_transpose().matrix_multiply(a))
        assert c.matrix_transpose().focus() == (d.n[1], d.n[0])
        assert approx_equal(c.matrix_transpose(), d.transpose())
        assert approx_equal(c.matrix_transpose().matrix_transpose(), d)
        c.matrix_transpose_in_place()
        assert c.focus() == (d.n[1], d.n[0])
        assert approx_equal(c, d.transpose())
        c.matrix_transpose_in_place()
        assert c.focus() == d.n
        assert approx_equal(c, d)
        b = flex.random_double(size=a_n_columns)
        c = a.matrix_multiply(b)
        d = matrix.rec(a, a.focus()) * matrix.col(b)
        assert d.n[1] == 1
        assert c.focus() == (d.n[0],)
        assert approx_equal(c, d)
        b = flex.random_double(size=a_n_rows)
        c = b.matrix_multiply(a)
        d = matrix.row(b) * matrix.rec(a, a.focus())
        assert d.n[0] == 1
        assert c.focus() == (d.n[1],)
        assert approx_equal(c, d)
  #
  for n_rows in range(10):
    for n_columns in range(10):
      m = flex.random_double(size=n_rows*n_columns)*4-2
      m.reshape(flex.grid(n_rows,n_columns))
      p = flex.random_double(size=n_columns*(n_columns+1)//2)
      nopt = m.matrix_multiply(p.matrix_packed_u_as_symmetric())
      assert nopt.focus() == (n_rows, n_columns)
      opt = m.matrix_multiply_packed_u(p)
      assert approx_equal(opt, nopt)
      nopt = nopt.matrix_multiply(m.matrix_transpose()) \
        .matrix_symmetric_as_packed_u()
      assert nopt.size() == n_rows*(n_rows+1)//2
      opt = m.matrix_multiply_packed_u_multiply_lhs_transpose(packed_u=p)
      assert approx_equal(opt, nopt)
      #
      assert approx_equal(
        p.matrix_packed_u_as_symmetric().matrix_diagonal(),
        p.matrix_packed_u_diagonal())
      #
      p = flex.complex_double(
        reals=flex.random_double(size=n_columns*(n_columns+1)//2),
        imags=flex.random_double(size=n_columns*(n_columns+1)//2))
      ps = p.matrix_packed_u_as_symmetric()
      nopt = m.matrix_multiply(ps)
      assert nopt.focus() == (n_rows, n_columns)
      opt = m.matrix_multiply_packed_u(p)
      assert approx_equal(opt, nopt)
      nopts = nopt.matrix_multiply(m.matrix_transpose())
      nopt = flex.complex_double(
        reals=flex.real(nopts).matrix_symmetric_as_packed_u(),
        imags=flex.imag(nopts).matrix_symmetric_as_packed_u())
      assert nopt.size() == n_rows*(n_rows+1)//2
      opt = m.matrix_multiply_packed_u_multiply_lhs_transpose(packed_u=p)
      assert approx_equal(opt, nopt)
      #
      pd = p.matrix_packed_u_diagonal()
      assert approx_equal(flex.real(ps).matrix_diagonal(), flex.real(pd))
      assert approx_equal(flex.imag(ps).matrix_diagonal(), flex.imag(pd))
  #
  a = flex.polar(flex.double(range(1,6+1)), flex.double(range(2,7+1)))
  a.resize(flex.grid(2,3))
  b = a.deep_copy()
  b.resize(flex.grid(3,2))
  c = a.matrix_multiply(b)
  assert approx_equal(c, matrix.rec(a, a.focus())*matrix.rec(b, b.focus()))
  b = flex.real(b)
  c = a.matrix_multiply(b)
  assert approx_equal(c, matrix.rec(a, a.focus())*matrix.rec(b, b.focus()))
  c = b.matrix_multiply(a)
  assert approx_equal(c, matrix.rec(b, b.focus())*matrix.rec(a, a.focus()))
  #
  assert flex.double().matrix_outer_product(rhs=flex.double()).size() == 0
  assert flex.double().matrix_outer_product(rhs=1, rhs_size=0).size() == 0
  assert flex.double([1]).matrix_outer_product(rhs=flex.double()).size() == 0
  assert flex.double([1]).matrix_outer_product(rhs=1, rhs_size=0).size() == 0
  assert flex.double().matrix_outer_product(rhs=flex.double([1])).size() == 0
  assert flex.double().matrix_outer_product(rhs=1, rhs_size=1).size() == 0
  op = flex.double([1,2,3]).matrix_outer_product(rhs=flex.double([4,5]))
  assert op.focus() == (3,2)
  assert approx_equal(op, [4,5,8,10,12,15])
  op = flex.double([1,2,3]).matrix_outer_product(rhs=-1)
  assert op.focus() == (3,3)
  assert approx_equal(op, [-1,-1,-1,-2,-2,-2,-3,-3,-3])
  op = flex.double([1,2,3]).matrix_outer_product(rhs=-1, rhs_size=2)
  assert op.focus() == (3,2)
  assert approx_equal(op, [-1,-1,-2,-2,-3,-3])
  lu = flex.double([1,0,0,0,1,0,0,0,1])
  lu.resize(flex.grid(3,3))
  pivot_indices = lu.matrix_lu_decomposition_in_place()
  assert approx_equal(lu, [1,0,0,0,1,0,0,0,1])
  assert list(pivot_indices) == [0,1,2,0]
  for rank in list(range(1,6)) + [18]:
    for i in range(10):
      m = flex.random_double(size=rank*rank)*3-1.5
      m.resize(flex.grid(rank,rank))
      assert approx_equal(
        m.matrix_diagonal(), [m[j*rank+j] for j in range(rank)])
      mc = m.deep_copy()
      mc.matrix_diagonal_set_in_place(value=3)
      assert approx_equal(mc.matrix_diagonal(), [3]*rank)
      mc.matrix_diagonal_set_in_place(diagonal=flex.double(range(1,1+rank)))
      assert approx_equal(mc.matrix_diagonal(), list(range(1,1+rank)))
      mc.matrix_diagonal_add_in_place(value=-5)
      assert approx_equal(mc.matrix_diagonal(), list(range(1-5,1-5+rank)))
      assert approx_equal(
        m.matrix_diagonal_sum(), flex.sum(m.matrix_diagonal()))
      assert m.matrix_trace() == m.matrix_diagonal_sum()
      assert approx_equal(
        m.matrix_diagonal_product(), flex.product(m.matrix_diagonal()))
      b = flex.random_double(size=rank)*7-3.5
      lu = m.deep_copy()
      pivot_indices = lu.matrix_lu_decomposition_in_place()
      x = lu.matrix_lu_back_substitution(pivot_indices=pivot_indices, b=b)
      assert approx_equal(m.matrix_multiply(x), b)
  lu = flex.double([0,0,0,0,1,0,0,0,1])
  lu.resize(flex.grid(3,3))
  try: lu.matrix_lu_decomposition_in_place()
  except RuntimeError as e:
    assert str(e) == "lu_decomposition_in_place: singular matrix"
  else: raise Exception_expected
  lu = flex.double([1,0,0,0,1,0,0,0,1])
  lu.resize(flex.grid(3,3))
  b = flex.double([1,2,3])
  pivot_indices = flex.size_t([0,1,4,0])
  try: lu.matrix_lu_back_substitution(pivot_indices, b)
  except RuntimeError as e:
    assert str(e) == "lu_back_substitution: pivot_indices[i] out of range"
  else: raise Exception_expected
  lu = flex.double([1,6,4,32,6,2,-1,63,-4,1,4,6,1,0,-13,5])
  lu.resize(flex.grid(4,4))
  pivot_indices = lu.matrix_lu_decomposition_in_place()
  assert approx_equal(
    lu.matrix_determinant_via_lu(pivot_indices=pivot_indices), -16522)
  m = [1,6,4,32,6,2,-1,63,-4,1,4,6,1,0,-13,5]
  assert approx_equal(matrix.sqr(m).determinant(), -16522)
  m = [1,6,4,32,6,2,-1,-63,-4,1,4,6,1,0,-13,5]
  assert approx_equal(matrix.sqr(m).determinant(), 21908)
  m = [0,0,0,0,6,2,-1,63,-4,1,4,6,1,0,-13,5]
  assert matrix.sqr(m).determinant() == 0
  #
  for size in range(10):
    a = flex.double(size)
    b = flex.double(size)
    assert a.cos_angle(b=b, value_if_undefined=-10) == -10
    assert a.cos_angle(b=b) is None
    assert a.angle(b=b) is None
  for size in range(1,10):
    a = flex.double(range(10))
    b = flex.double(range(10))
    assert approx_equal(a.cos_angle(b=b, value_if_undefined=-10), 1,eps=1.e-10)
    assert approx_equal(a.cos_angle(b=b), 1, eps=1.e-10)
    assert approx_equal(a.angle(b=b), 0, eps=1.e-10)
  a = flex.double([1,0])
  b = flex.double([1,1])
  assert approx_equal(a.angle(b, deg=True), 45)
  b = flex.double([1,2])
  assert approx_equal(a.angle(b, deg=True), 63.4349488229)
  mersenne_twister = flex.mersenne_twister(0)
  for size in range(1,100):
    a = mersenne_twister.random_double(size=size)
    b = mersenne_twister.random_double(size=size)
    assert approx_equal(
      a.cos_angle(b=b), matrix.col(a).cos_angle(matrix.col(b)))
    assert approx_equal(a.angle(b=b, deg=False), a.angle(b))
    assert approx_equal(a.angle(b=b, deg=True), a.angle(b)*180/math.pi)
  # these values lead to a floating-point exception (cos(angle) > 1)
  i=[-0.0002974948153438084, 0.00032319472094305282, -0.00039358675259127106]
  j=[-0.00036378012360459966, 0.00039520626736685348, -0.00048128246316264911]
  assert abs(flex.double(i).angle(flex.double(j), deg=True)) < 1.e-10
  #
  assert approx_equal(
    flex.double([[1,2,3],[4,5,6],[7,8,9]]).matrix_upper_triangle_as_packed_u(),
    [1,2,3,5,6,9])
  assert approx_equal(
    flex.double([1,2,3,5,6,9]).matrix_packed_u_as_upper_triangle(),
    [1,2,3, 0,5,6, 0,0,9])
  assert approx_equal(
    flex.double([[1,2,3],[4,5,6],[7,8,9]]).matrix_lower_triangle_as_packed_l(),
    [1,4,5,7,8,9])
  assert approx_equal(
    flex.double([1,4,5,7,8,9]).matrix_packed_l_as_lower_triangle(),
    [1,0,0, 4,5,0, 7,8,9])
  m = flex.double(range(1,17))
  m.resize(flex.grid(4, 4))
  assert approx_equal(
    m.matrix_upper_triangle_as_packed_u(),
    [1,2,3,4,6,7,8,11,12,16])
  assert approx_equal(
    flex.double([1,2,3,4,6,7,8,11,12,16])
      .matrix_packed_u_as_upper_triangle(),
    [1,2,3,4,0,6,7,8,0,0,11,12,0,0,0,16])
  assert approx_equal(
    m.matrix_lower_triangle_as_packed_l(),
    [1,5,6,9,10,11,13,14,15,16])
  assert approx_equal(
    flex.double([1,5,6,9,10,11,13,14,15,16])
      .matrix_packed_l_as_lower_triangle(),
    [1,0,0,0,5,6,0,0,9,10,11,0,13,14,15,16])
  def exercise_packed(n, s, u, l):
    assert s.focus() == (n,n)
    if (u is not None):
      p = s.matrix_upper_triangle_as_packed_u()
      assert approx_equal(p, u)
      t = p.matrix_packed_u_as_upper_triangle()
      assert approx_equal(t.matrix_upper_triangle_as_packed_u(), u)
      p = s.matrix_symmetric_as_packed_u()
      assert approx_equal(p, u)
      us = p.matrix_packed_u_as_symmetric()
      assert us.focus() == (n,n)
      assert approx_equal(us, s)
    if (l is not None):
      p = s.matrix_lower_triangle_as_packed_l()
      assert approx_equal(p, l)
      t = p.matrix_packed_l_as_lower_triangle()
      assert approx_equal(t.matrix_lower_triangle_as_packed_l(), l)
      p = s.matrix_symmetric_as_packed_l()
      assert approx_equal(p, l)
      ls = p.matrix_packed_l_as_symmetric()
      assert ls.focus() == (n,n)
      assert approx_equal(ls, s)
  exercise_packed(
    n=0,
    s=flex.double(flex.grid(0,0)),
    u=flex.double(),
    l=flex.double())
  exercise_packed(
    n=1,
    s=flex.double([[1]]),
    u=flex.double([1]),
    l=flex.double([1]))
  exercise_packed(
    n=2,
    s=flex.double([[1,2],[2,3]]),
    u=flex.double([1,2,3]),
    l=flex.double([1,2,3]))
  exercise_packed(
    n=3,
    s=flex.double([[1,2,3],[2,4,5],[3,5,6]]),
    u=flex.double([1,2,3,4,5,6]),
    l=flex.double([1,2,4,3,5,6]))
  exercise_packed(
    n=4,
    s=flex.double([[1,-2,3,4], [-2,-5,6,-7], [3,6,8,9], [4,-7,9,-10]]),
    u=flex.double([1,-2,3,4,-5,6,-7,8,9,-10]),
    l=flex.double([1,-2,-5,3,6,8,4,-7,9,-10]))
  for n in range(20):
    p = flex.random_double(size=n*(n+1)//2)
    exercise_packed(
      n=n,
      s=p.matrix_packed_u_as_symmetric(),
      u=p,
      l=None)
    exercise_packed(
      n=n,
      s=p.matrix_packed_l_as_symmetric(),
      u=None,
      l=p)
  e = flex.double([[1,2,3],[2,4,5],[3,9,6]])
  try: e.matrix_symmetric_as_packed_u()
  except RuntimeError as err:
    assert str(err) == "symmetric_as_packed_u(): matrix is not symmetric."
  else: raise Exception_expected
  p = e.matrix_symmetric_as_packed_u(relative_epsilon=1)
  assert approx_equal(p, [1,2,3,4,7,6])
  try: e.matrix_symmetric_as_packed_l()
  except RuntimeError as err:
    assert str(err) == "symmetric_as_packed_l(): matrix is not symmetric."
  else: raise Exception_expected
  p = e.matrix_symmetric_as_packed_l(relative_epsilon=1)
  assert approx_equal(p, [1,2,4,3,7,6])
  assert not e.matrix_is_symmetric(relative_epsilon=1e-12)
  assert e.matrix_is_symmetric(relative_epsilon=1)
  p = flex.double(4)
  for method in [p.matrix_packed_u_as_symmetric,
                 p.matrix_packed_l_as_symmetric]:
    try: method()
    except RuntimeError as e:
      assert str(e).endswith(
        "SCITBX_ASSERT(n*(n+1)/2 == packed_size) failure.")
    else: raise Exception_expected
  #
  n_not_symmetric = 0
  for n in range(10):
    a = mersenne_twister.random_double(size=n*n)*2-1
    a.reshape(flex.grid(n,n))
    d = mersenne_twister.random_double(size=n)*2-1
    atda_u = a.matrix_transpose_multiply_diagonal_multiply_as_packed_u(
      diagonal_elements=d)
    dsq = flex.double(flex.grid(n,n), 0)
    for i in range(n):
      dsq[i*(n+1)] = d[i]
    atda_sym = a.matrix_transpose().matrix_multiply(dsq).matrix_multiply(a)
    assert approx_equal(atda_u.matrix_packed_u_as_symmetric(), atda_sym)
    assert atda_sym.matrix_is_symmetric(relative_epsilon=1e-12)
    if (not atda_sym.matrix_is_symmetric(relative_epsilon=1e-30)):
      n_not_symmetric += 1
  assert n_not_symmetric > 0 # could fail if random number generator is changed

def exercise_matrix_int():
  a = flex.int(range(1,7))
  a.resize(flex.grid(3,2))
  assert list(a)==[1, 2, 3, 4, 5, 6]
  assert a.focus()==(3,2)
  a.matrix_transpose_in_place()
  assert a.focus()==(2,3)
  assert list(a)==[1, 3, 5, 2, 4, 6]
  a.matrix_swap_rows_in_place(1,0)
  assert list(a)==[2, 4, 6, 1, 3, 5]
  a.matrix_swap_rows_in_place(0,1)
  a.matrix_swap_columns_in_place(1,1)
  assert list(a)==[1, 3, 5, 2, 4, 6]
  a.matrix_swap_columns_in_place(0,2)
  assert list(a)==[5, 3, 1, 6, 4, 2]
  #
  b = a.matrix_rot90(4)
  c = a.matrix_rot90(-8)
  assert b.focus()==(2,3) and c.focus()==(2,3) and list(b)==list(c)
  b = b.matrix_rot90(5)
  c = c.matrix_rot90(-7)
  assert b.focus()==(3,2) and c.focus()==(3,2) and list(b)==list(c)
  b = b.matrix_rot90(6)
  c = c.matrix_rot90(-6)
  assert b.focus()==(3,2) and c.focus()==(3,2) and list(b)==list(c)
  b = b.matrix_rot90(7)
  c = c.matrix_rot90(-5)
  assert b.focus()==(2,3) and c.focus()==(2,3) and list(b)==list(c)

def exercise_matrix_norms():
  a = flex.double((1,  2, -3,
                   4, -5,  6,
                   7,  8,  9,
                  -1, -2, -3 ))
  a.resize(flex.grid(4,3))
  assert a.matrix_norm_1() == 21
  assert a.matrix_norm_inf() == 24
  assert approx_equal(a.matrix_norm_frobenius(), math.sqrt(299.))

def exercise_matrix_move():
  a = flex.double(flex.grid(0,0))
  b = a.matrix_copy_block(i_row=0, i_column=0, n_rows=0, n_columns=0)
  assert b.focus() == (0,0)
  a.matrix_paste_block_in_place(block=b, i_row=0, i_column=0)
  for a in [flex.double([[1]]), flex.double([[1,2],[3,4]])]:
    b = a.matrix_copy_block(i_row=0, i_column=0, n_rows=0, n_columns=0)
    assert b.focus() == (0,0)
    a.matrix_paste_block_in_place(block=b, i_row=0, i_column=0)
    b = a.matrix_copy_block(i_row=0, i_column=0, n_rows=1, n_columns=0)
    assert b.focus() == (1,0)
    a.matrix_paste_block_in_place(block=b, i_row=0, i_column=0)
    b = a.matrix_copy_block(i_row=0, i_column=0, n_rows=0, n_columns=1)
    assert b.focus() == (0,1)
    a.matrix_paste_block_in_place(block=b, i_row=0, i_column=0)
    b = a.matrix_copy_block(i_row=0, i_column=0, n_rows=1, n_columns=1)
    assert b.focus() == (1,1)
    assert list(b) == [1]
    b[0] = 5
    a.matrix_paste_block_in_place(block=b, i_row=0, i_column=0)
    assert a[0] == 5
  b = a.matrix_copy_block(0,0,2,2)
  assert b.focus() == (2,2)
  assert b.all_eq(a)
  a = flex.double(range(1,20+1))
  a.resize(flex.grid(4,5))
  b = a.matrix_copy_block(0,0,4,5)
  assert b.focus() == (4,5)
  assert b.all_eq(a)
  for i in range(4):
    for j in range(5):
      b = a.matrix_copy_block(i,j,1,1)
      assert b.focus() == (1,1)
      assert b[0] == a[(i,j)]
      c = a.deep_copy()
      c.matrix_paste_block_in_place(flex.double([[93]]), i, j)
      assert c[(i,j)] == 93
  for i in range(3):
    for j in range(5):
      b = a.matrix_copy_block(i,j,2,1)
      assert b.focus() == (2,1)
      assert b[0] == a[(i,j)]
      assert b[1] == a[(i+1,j)]
      c = a.deep_copy()
      c.matrix_paste_block_in_place(flex.double([[91],[-10]]), i, j)
      assert c[(i,j)] == 91
      assert c[(i+1,j)] == -10
  for i in range(3):
    for j in range(4):
      b = a.matrix_copy_block(i,j,2,2)
      assert b.focus() == (2,2)
      assert b[0] == a[(i,j)]
      assert b[1] == a[(i,j+1)]
      assert b[2] == a[(i+1,j)]
      assert b[3] == a[(i+1,j+1)]
      c = a.deep_copy()
      c.matrix_paste_block_in_place(flex.double([[97,-4],[-13,84]]), i, j)
      assert c[(i,j)] == 97
      assert c[(i,j+1)] == -4
      assert c[(i+1,j)] == -13
      assert c[(i+1,j+1)] == 84
  for i in range(3):
    for j in range(3):
      b = a.matrix_copy_block(i,j,2,3)
      assert b.focus() == (2,3)
      assert b[0] == a[(i,j)]
      assert b[1] == a[(i,j+1)]
      assert b[2] == a[(i,j+2)]
      assert b[3] == a[(i+1,j)]
      assert b[4] == a[(i+1,j+1)]
      assert b[5] == a[(i+1,j+2)]
      c = a.deep_copy()
      c.matrix_paste_block_in_place(flex.double([[79,-3,75],[-31,48,-7]]),i,j)
      assert c[(i,j)] == 79
      assert c[(i,j+1)] == -3
      assert c[(i,j+2)] == 75
      assert c[(i+1,j)] == -31
      assert c[(i+1,j+1)] == 48
      assert c[(i+1,j+2)] == -7
  a = flex.complex_double([1,2j,3j,4])
  a.resize(flex.grid(2,2))
  b = a.matrix_copy_block(i_row=1, i_column=0, n_rows=1, n_columns=2)
  assert approx_equal(b, [3j,(4+0j)])
  b = flex.complex_double([10,20j])
  b.resize(flex.grid(2,1))
  assert b.matrix_transpose().focus() == (1,2)
  a.matrix_paste_block_in_place(block=b, i_row=0, i_column=1)
  assert approx_equal(a, [(1+0j), (10+0j), 3j, 20j])
  #
  a = flex.double(flex.grid(0,0))
  a.matrix_copy_upper_to_lower_triangle_in_place()
  a.matrix_copy_lower_to_upper_triangle_in_place()
  a = flex.double([[1]])
  a.matrix_swap_rows_in_place(i=0, j=0)
  a.matrix_swap_columns_in_place(i=0, j=0)
  a.matrix_symmetric_upper_triangle_swap_rows_and_columns_in_place(i=0, j=0)
  assert list(a) == [1]
  u = flex.double([1])
  u.matrix_packed_u_swap_rows_and_columns_in_place(i=0, j=0)
  assert list(u) == [1]
  a = flex.double([[1,2],[3,4]])
  a.matrix_swap_rows_in_place(i=0, j=0)
  assert list(a) == [1,2,3,4]
  a.matrix_swap_rows_in_place(i=0, j=1)
  assert list(a) == [3,4,1,2]
  a.matrix_swap_rows_in_place(i=1, j=0)
  assert list(a) == [1,2,3,4]
  a.matrix_swap_columns_in_place(i=0, j=0)
  assert list(a) == [1,2,3,4]
  a.matrix_swap_columns_in_place(i=0, j=1)
  assert list(a) == [2,1,4,3]
  a.matrix_swap_columns_in_place(i=1, j=0)
  assert list(a) == [1,2,3,4]
  a.matrix_swap_rows_in_place(i=1, j=0)
  a.matrix_swap_columns_in_place(i=1, j=0)
  assert list(a) == [4,3,2,1]
  a.matrix_symmetric_upper_triangle_swap_rows_and_columns_in_place(i=0, j=1)
  assert list(a) == [1,3,2,4]
  u = a.matrix_upper_triangle_as_packed_u()
  u.matrix_packed_u_swap_rows_and_columns_in_place(i=0, j=1)
  assert list(u) == [4,3,1]
  a.matrix_copy_upper_to_lower_triangle_in_place()
  assert list(a) == [1,3,3,4]
  a = flex.double([[1,2],[3,4]])
  a.matrix_copy_lower_to_upper_triangle_in_place()
  assert list(a) == [1,3,3,4]
  a = flex.double([[1,2,3],[4,5,6]])
  a.matrix_swap_rows_in_place(i=1, j=0)
  assert list(a) == [4,5,6,1,2,3]
  a.matrix_swap_columns_in_place(i=1, j=2)
  assert list(a) == [4,6,5,1,3,2]
  a.matrix_swap_columns_in_place(i=2, j=0)
  assert list(a) == [5,6,4,2,3,1]
  a.resize(flex.grid(3,2))
  a.matrix_swap_rows_in_place(i=1, j=2)
  assert list(a) == [5,6,3,1,4,2]
  a.matrix_swap_rows_in_place(i=2, j=0)
  assert list(a) == [4,2,3,1,5,6]
  for n in range(1,11): # must be at least 8 to reveal all bugs
    a0 = flex.double(range(1,n+1))
    a0.resize(flex.grid(n,n))
    for i in range(n):
      for j in range(n):
        for triangle_flag in ["u","l"]:
          a = a0.deep_copy()
          if (triangle_flag == "u"):
            a.matrix_copy_upper_to_lower_triangle_in_place()
          else:
            a.matrix_copy_lower_to_upper_triangle_in_place()
          a.matrix_swap_rows_in_place(i=i, j=j)
          a.matrix_swap_columns_in_place(i=i, j=j)
          b = a0.deep_copy()
          if (triangle_flag == "l"):
            b.matrix_transpose_in_place()
          b0 = b.deep_copy()
          b.matrix_symmetric_upper_triangle_swap_rows_and_columns_in_place(
            i=i, j=j)
          for ii in range(1,n):
            for jj in range(i):
              assert b[(ii,jj)] == b0[(ii,jj)]
          b.matrix_symmetric_upper_triangle_swap_rows_and_columns_in_place(
            i=i, j=j)
          assert list(b) == list(b0)
          b.matrix_symmetric_upper_triangle_swap_rows_and_columns_in_place(
            i=i, j=j)
          b.matrix_copy_upper_to_lower_triangle_in_place()
          assert list(b) == list(a)
          u = b0.matrix_upper_triangle_as_packed_u()
          u.matrix_packed_u_swap_rows_and_columns_in_place(i=i, j=j)
          assert list(u) == list(b.matrix_upper_triangle_as_packed_u())
          u.matrix_packed_u_swap_rows_and_columns_in_place(i=i, j=j)
          assert list(u) == list(b0.matrix_upper_triangle_as_packed_u())
  #
  for n_columns in range(1,4):
    a = flex.double(flex.grid(0, n_columns))
    b = a.matrix_copy_column(i_column=0)
    assert b.size() == 0
  a = flex.double([[1]])
  b = a.matrix_copy_column(i_column=0)
  assert list(b) == [1]
  a = flex.double([[1,2]])
  b = a.matrix_copy_column(i_column=0)
  assert list(b) == [1]
  b = a.matrix_copy_column(i_column=1)
  assert list(b) == [2]
  a = flex.double([[1],[2]])
  b = a.matrix_copy_column(i_column=0)
  assert list(b) == [1,2]
  a = flex.double([[1,2],[3,4]])
  b = a.matrix_copy_column(i_column=0)
  assert list(b) == [1,3]
  b = a.matrix_copy_column(i_column=1)
  assert list(b) == [2,4]
  #
  a = flex.bool([i%2 == 0 for i in range(1,20+1)])
  a.resize(flex.grid(4,5))
  b = a.matrix_copy_block(i_row=1,i_column=2,n_rows=2,n_columns=3)
  assert b.focus() == (2,3)
  assert list(b) == [True,False,True,False,True,False]
  a.matrix_paste_block_in_place(block=b, i_row=0, i_column=1)
  assert list(a) == [
   False,True,False,True,False,
   True,False,True,False,True,
   False,True,False,True,False,
   True,False,True,False,True]
  a.matrix_paste_block_in_place(block=b, i_row=2, i_column=2)
  assert list(a) == [
   False,True,False,True,False,
   True,False,True,False,True,
   False,True,True,False,True,
   True,False,False,True,False]
  #
  a = flex.int(range(1,20+1))
  a.resize(flex.grid(4,5))
  b = a.matrix_copy_block(i_row=1,i_column=2,n_rows=2,n_columns=3)
  assert b.focus() == (2,3)
  assert list(b) == [8,9,10,13,14,15]
  a.matrix_paste_block_in_place(block=b, i_row=0, i_column=1)
  assert list(a) == [
   1,8,9,10,5,
   6,13,14,15,10,
   11,12,13,14,15,
   16,17,18,19,20]
  a.matrix_paste_block_in_place(block=b, i_row=2, i_column=2)
  assert list(a) == [
   1,8,9,10,5,
   6,13,14,15,10,
   11,12,8,9,10,
   16,17,13,14,15]
  #
  a = flex.long(range(1,20+1))
  a.resize(flex.grid(4,5))
  b = a.matrix_copy_block(i_row=1,i_column=2,n_rows=2,n_columns=3)
  assert b.focus() == (2,3)
  assert list(b) == [8,9,10,13,14,15]
  a.matrix_paste_block_in_place(block=b, i_row=0, i_column=1)
  assert list(a) == [
   1,8,9,10,5,
   6,13,14,15,10,
   11,12,13,14,15,
   16,17,18,19,20]
  a.matrix_paste_block_in_place(block=b, i_row=2, i_column=2)
  assert list(a) == [
   1,8,9,10,5,
   6,13,14,15,10,
   11,12,8,9,10,
   16,17,13,14,15]
  #
  a = flex.int([1,2,3,4])
  a.reshape(flex.grid(2,2))
  assert not a.matrix_is_symmetric()
  a[2] = 2
  assert a.matrix_is_symmetric()

  """
  matrix_paste_column_in_place
  """
  m, n = 3, 5
  a = flex.double_range(m*n)
  a.reshape(flex.grid(m, n))
  u = flex.double((-1, -2, -3))
  try:
    a.matrix_paste_column_in_place(u, n)
    raise Exception_expected
  except RuntimeError:
    pass
  try:
    v = flex.double((-1, -2))
    a.matrix_paste_column_in_place(v, 0)
    raise Exception_expected
  except RuntimeError:
    pass
  try:
    v = flex.double((-1, -2, -3, -4))
    a.matrix_paste_column_in_place(v, 0)
    raise Exception_expected
  except RuntimeError:
    pass
  for j in range(n):
    a.matrix_paste_column_in_place(u, j)
    assert a.matrix_copy_column(j).all_eq(u)

def exercise_copy_upper_or_lower_triangle():
  for m, n in [ (5,3), (5,4), (5,5), (4,5), (3,5),
                (4,2), (2,4),
                (3,2), (2,3), (3,3),
                (2,2),
                (2,1), (1,2) ]:
    a = flex.double(range(m*n))
    a.resize(flex.grid(m,n))
    try:
      u = a.matrix_copy_upper_triangle()
      assert u.focus() == (n,n)
      for i in range(n):
        for j in range(n):
          if i > j: assert u[i,j] == 0
          else: assert u[i,j] == a[i,j]
    except RuntimeError as e:
      assert m < n and str(e).find('SCITBX_ASSERT(m >= n)') > 0
    try:
      l = a.matrix_copy_lower_triangle()
      assert l.focus() == (m,m)
      for i in range(m):
        for j in range(m):
          if i < j: assert l[i,j] == 0
          else: assert l[i,j] == a[i,j]
    except RuntimeError as e:
      assert m > n and str(e).find('SCITBX_ASSERT(m <= n)') > 0

def exercise_matrix_bidiagonal():
  for m, n in [ (5,3), (5,4), (5,5), (4,5), (3,5),
                (4,2), (2,4),
                (3,2), (2,3), (3,3),
                (2,2),
                (2,1), (1,2) ]:
    a = flex.double(range(m*n))
    a.resize(flex.grid(m,n))
    d,f = a.matrix_upper_bidiagonal()
    assert len(d) == min(m,n)
    assert len(f) == len(d) - 1
    assert list(d) == [ a[i,i] for i in range(len(d)) ]
    assert list(f) == [ a[i,i+1] for i in range(len(f)) ]
    d,f = a.matrix_lower_bidiagonal()
    assert len(d) == min(m,n)
    assert len(f) == len(d) - 1
    assert list(d) == [ a[i,i] for i in range(len(d)) ]
    assert list(f) == [ a[i+1,i] for i in range(len(f)) ]

def exercise_quadratic_form():
  for n in range(1,10):
    a = flex.random_double(n*(n+1)//2)
    x = flex.double(n)
    s = a.matrix_symmetric_upper_triangle_quadratic_form(x)
    s1 = matrix.col(x).dot(
      matrix.sqr(a.matrix_packed_u_as_symmetric())*matrix.col(x))
    assert approx_equal(s, s1)

def exercise_matrix_inversion_in_place():
  m = flex.double()
  m.resize(flex.grid(0,0))
  m.matrix_inversion_in_place(m)
  b = flex.double()
  b.resize(flex.grid(0,0))
  m.matrix_inversion_in_place(b=b)
  m = flex.double([2])
  m.resize(flex.grid(1,1))
  m.matrix_inversion_in_place()
  assert approx_equal(m, [1/2.])
  m = flex.double([2,0,0,-3])
  m.resize(flex.grid(2,2))
  m.matrix_inversion_in_place()
  assert approx_equal(m, [1/2.,0,0,-1/3.])
  m = flex.double([1,2,-3,4])
  m.resize(flex.grid(2,2))
  m.matrix_inversion_in_place()
  assert approx_equal(m, [2/5.,-1/5.,3/10.,1/10.])
  m = flex.double([2,0,0,0,-3,0,0,0,4])
  m.resize(flex.grid(3,3))
  m.matrix_inversion_in_place()
  assert approx_equal(m, [1/2.,0,0,0,-1/3.,0,0,0,1/4.])
  m = flex.double([1,2,-3,-2,4,-1,8,0,4])
  m.resize(flex.grid(3,3))
  m.matrix_inversion_in_place()
  assert approx_equal(m, [1/7.,-1/14.,5/56.,0,1/4.,1/16.,-2/7.,1/7.,1/14.])
  from scitbx import matrix
  for n in range(1,12):
    u = flex.double(n*n, 0)
    for i in range(0,n*n,n+1): u[i] = 1
    for diag in [1,2]:
      m = flex.double(n*n, 0)
      for i in range(0,n*n,n+1): m[i] = diag
      m.resize(flex.grid(n,n))
      m_orig = matrix.rec(m, (n,n))
      m.matrix_inversion_in_place()
      m_inv = matrix.rec(m, (n,n))
      assert approx_equal(m_orig*m_inv, u)
      assert approx_equal(m_inv*m_orig, u)
      for n_b in range(0,4):
        m = flex.double(m_orig)
        m.resize(flex.grid(n,n))
        b = flex.double(range(1,n*n_b+1))
        b.resize(flex.grid(n_b,n))
        b_orig = matrix.rec(b, (n,n_b))
        m.matrix_inversion_in_place(b)
        m_inv = matrix.rec(m, (n,n))
        x = matrix.rec(b, (n_b,n))
        assert approx_equal(m_orig*m_inv, u)
        assert approx_equal(m_inv*m_orig, u)
        for i_b in range(n_b):
          b_i = matrix.col(b_orig.elems[i_b*n:(i_b+1)*n])
          x_i = matrix.col(x.elems[i_b*n:(i_b+1)*n])
          assert approx_equal(m_orig*x_i, b_i)
  for n in range(1,12):
    u = flex.double(n*n, 0)
    for i in range(0,n*n,n+1): u[i] = 1
    for i_trial in range(3):
      m = 2*flex.random_double(n*n)-1
      m.resize(flex.grid(n,n))
      m_orig = matrix.rec(m, (n,n))
      try:
        m.matrix_inversion_in_place()
      except RuntimeError as e:
        assert str(e) == "inversion_in_place: singular matrix"
      else:
        m_inv = matrix.rec(m, (n,n))
        assert approx_equal(m_orig*m_inv, u)
        assert approx_equal(m_inv*m_orig, u)
        for n_b in range(0,4):
          m = flex.double(m_orig)
          m.resize(flex.grid(n,n))
          b = flex.random_double(n*n_b)
          b.resize(flex.grid(n_b,n))
          b_orig = matrix.rec(b, (n,n_b))
          m.matrix_inversion_in_place(b)
          m_inv = matrix.rec(m, (n,n))
          x = matrix.rec(b, (n_b,n))
          assert approx_equal(m_orig*m_inv, u)
          assert approx_equal(m_inv*m_orig, u)
          for i_b in range(n_b):
            b_i = matrix.col(b_orig.elems[i_b*n:(i_b+1)*n])
            x_i = matrix.col(x.elems[i_b*n:(i_b+1)*n])
            assert approx_equal(m_orig*x_i, b_i)

def exercise_pickle_single_buffered():
  a = flex.bool((True,False,True))
  p = pickle.dumps(a)
  b = pickle.loads(p)
  assert b.size() == 3
  assert tuple(b) == (True,False,True)
  a = flex.double(())
  p = pickle.dumps(a)
  b = pickle.loads(p)
  assert b.size() == 0
  a = flex.double((1,2,3))
  p = pickle.dumps(a)
  b = pickle.loads(p)
  assert b.size() == 3
  assert tuple(b) == (1,2,3)
  a = flex.int((1,2,3))
  p = pickle.dumps(a)
  b = pickle.loads(p)
  assert b.size() == 3
  assert tuple(b) == (1,2,3)
  a = flex.float((1,2,3))
  p = pickle.dumps(a)
  b = pickle.loads(p)
  assert b.size() == 3
  assert tuple(b) == (1,2,3)
  a = flex.complex_double((1+2j, 2+3j, 4+5j))
  p = pickle.dumps(a)
  b = pickle.loads(p)
  assert b.size() == 3
  assert tuple(b) == (1+2j, 2+3j, 4+5j)
  a = flex.double(flex.grid((-1,2,-3), (7,5,3)).set_focus((1,3,2)), 13)
  a[(1,2,-1)] = -8
  p = pickle.dumps(a)
  b = pickle.loads(p)
  assert b.size() == 8 * 3 * 6
  assert tuple(a) == tuple(b)
  assert a.origin() == b.origin()
  assert a.all() == b.all()
  assert a.focus() == b.focus()
  assert a.accessor() == b.accessor()

def exercise_pickle_double_buffered():
  a = flex.std_string()
  p = pickle.dumps(a)
  b = pickle.loads(p)
  assert b.size() == 0
  assert tuple(b) == ()
  a = flex.std_string(("abc", "bcd", "cde"))
  p = pickle.dumps(a)
  b = pickle.loads(p)
  assert b.size() == 3
  assert tuple(b) == (("abc", "bcd", "cde"))

def pickle_large_arrays(max_exp, verbose):
  try:
    for array_type in (
        flex.bool,
        flex.size_t,
        flex.int,
        flex.long,
        flex.float,
        flex.double,
        flex.complex_double,
        flex.std_string):
      for e in range(max_exp+1):
        n = 2**e
        if (array_type == flex.bool):
          val = True
        elif (array_type == flex.size_t):
          val = 2147483647
        elif (array_type == flex.int):
          val = -2147483647
        elif (array_type == flex.long):
          val = -9223372036854775807
          if (sys.platform == 'win32'):
            val = -2147483647
        elif (array_type == flex.complex_double):
          val = complex(-1.234567809123456e+20, -1.234567809123456e+20)
        elif (array_type in (flex.float, flex.double)):
          val = -1.234567809123456e+20
        elif (array_type == flex.std_string):
          val = "x" * 10
        else:
          raise AssertionError("Unexpected array type.")
        a = array_type(n, val)
        for pickler in (0, pickle):
          if (pickler == 0):
            pickler_name = "g/setstate"
            t0 = time.time()
            s = a.__getstate__()
            td = time.time() - t0
            t0 = time.time()
            b = array_type()
            b.__setstate__(s)
            tl = time.time() - t0
          else:
            pickler_name = pickler.__name__
            f = open("pickle.tmp", "wb")
            t0 = time.time()
            pickler.dump(a, f, 1)
            td = time.time() - t0
            f.close()
            f = open("pickle.tmp", "rb")
            t0 = time.time()
            b = pickle.load(f)
            tl = time.time() - t0
            f.close()
          if (verbose):
            print(array_type.__name__, n, pickler_name, "%.2f %.2f" % (td, tl))
          sys.stdout.flush()
  finally:
    try: os.unlink("pickle.tmp")
    except OSError: pass

def exercise_py_object():
  a = flex.py_object(flex.grid(2,3), value=3)
  assert a.accessor().focus() == (2,3)
  assert a.data() == [3,3,3,3,3,3]
  a = flex.py_object(flex.grid(2,3), value_factory=list)
  assert a.data() == [[],[],[],[],[],[]]
  a = flex.py_object(flex.grid(2,3), values=list(range(6)))
  assert a.data() == list(range(6))
  assert a[(1,2)] == 5
  a[(1,2)] = -5
  assert a[(1,2)] == -5
  a[(0,0)] = -10
  a[(0,1)] = -1
  a[(0,2)] = -2
  a[(1,0)] = -3
  a[(1,1)] = -4
  assert a.data() == [-10,-1,-2,-3,-4,-5]

def exercise_condense_as_ranges():
  a = flex.int([])
  assert flex.condense_as_ranges(integer_array=a) == []
  a = flex.int([3])
  assert flex.condense_as_ranges(integer_array=a) == [(3,)]
  a = flex.int([3,4])
  assert flex.condense_as_ranges(integer_array=a) == [(3,4)]
  a = flex.int([3,4,5])
  assert flex.condense_as_ranges(integer_array=a) == [(3,5)]
  a = flex.int([-1,3,4,5])
  assert flex.condense_as_ranges(integer_array=a) == [(-1,),(3,5)]
  a = flex.int([-3,-2,-1,0,1,3,4,5])
  assert flex.condense_as_ranges(integer_array=a) == [(-3,1),(3,5)]
  a = flex.int([3,4,5,7])
  assert flex.condense_as_ranges(integer_array=a) == [(3,5),(7,)]
  a = flex.int([3,4,5,7,8,9,10])
  assert flex.condense_as_ranges(integer_array=a) == [(3,5),(7,10)]

def exercise_first_index_etc():
  """ first_index, last_index, method index """
  a = flex.int([4, -1, 3, 2, -1, 0, -3, 5, 0, 6, 8, 5])
  assert flex.first_index(a, 4) == 0
  assert flex.first_index(a, -1) == 1
  assert flex.last_index(a, 0) == 8
  assert flex.last_index(a, -1) == 4
  assert flex.first_index(a, 10) is None
  assert flex.last_index(a, 66) is None

  a = flex.bool((True, True, False, True, False, False, True, False))
  assert flex.first_index(a, False) == 2
  assert flex.last_index(a, True) == 6
  assert flex.first_index(a, True) == 0
  assert flex.last_index(a, False) == len(a)-1

  a = flex.std_string(("a", "b", "c", "b", "d", "e", "f"))
  assert flex.first_index(a, "b") == 1
  assert flex.last_index(a, "b") == 3
  assert flex.first_index(a, "z") is None

def exercise_c_grid_flex_conversion():
  a = flex.int(range(24))
  a.resize(flex.grid((2,3,4)))
  assert flex.tst_c_grid_flex_conversion(a, -1,4,-2) == a[1,1,2]

def exercise_versa_packed_u_to_flex():
  assert flex.exercise_versa_packed_u_to_flex() == flex.double(
    (11, 12, 13,
         22, 23,
             33)
  )

def exercise_triangular_systems():
  for n in range(1,5):
    a = flex.random_double(n*(n+1)//2)
    isinstance(a, flex.double)
    b = flex.random_double(n)

    x = a.matrix_forward_substitution(b)
    y = a.matrix_packed_l_as_lower_triangle().matrix_multiply(x)
    assert approx_equal(y, b)

    x = a.matrix_back_substitution(b)
    y = a.matrix_packed_u_as_upper_triangle().matrix_multiply(x)
    assert approx_equal(y, b)

    x = a.matrix_forward_substitution_given_transpose(b)
    lt = a.matrix_packed_u_as_upper_triangle().matrix_transpose()
    y = lt.matrix_multiply(x)
    assert approx_equal(y, b)

    x = a.matrix_back_substitution_given_transpose(b)
    ut = a.matrix_packed_l_as_lower_triangle().matrix_transpose()
    y = ut.matrix_multiply(x)
    assert approx_equal(y, b)

def exercise_approx_equal():
  a = flex.double((2.1, 3.1, 4.1))
  b = flex.double((2, 3, 4))
  assert a.all_approx_equal_relatively(3, relative_error=0.5)
  assert a.all_approx_equal_relatively(b, relative_error=0.05)
  c = flex.complex_double((2 + 2j, 3 + 3j, 4 + 4j))
  u = 0.1 + 0.1j
  d = c + flex.complex_double((u,)*3)
  assert d.all_approx_equal_relatively(3 + 3j, relative_error=0.5)
  assert d.all_approx_equal_relatively(c, relative_error=0.07)
  assert not d.all_approx_equal(c, tolerance=0.1)

def exercise_matrix_packed_u_diagonal():
  a = flex.double((1, 2, 3, 4 ,
                      5, 6, 7 ,
                         8, 9 ,
                            10))
  assert tuple(a.matrix_packed_u_diagonal()) == (1, 5, 8, 10)
  a.matrix_packed_u_diagonal_add_in_place(1)
  assert tuple(a.matrix_packed_u_diagonal()) == (2, 6, 9, 11)

def exercise_python_functions():
  a = flex.int([1,2,3,4,5,6])
  a.reshape(flex.grid(2,3))
  rows = list(flex.rows(a))
  assert len(rows) == 2
  assert list(rows[0]) == [1,2,3]
  assert list(rows[1]) == [4,5,6]
  a.reshape(flex.grid(3,2))
  rows = list(flex.rows(a))
  assert len(rows) == 3
  assert list(rows[0]) == [1,2]
  assert list(rows[1]) == [3,4]
  assert list(rows[2]) == [5,6]

def exercise_vec3_double_as_numpy_array():
  try:
    import numpy as np
  except ImportError:
    print("Skipping exercise_vec3_double_as_numpy_array (numpy not available)")
    return

  test_data = [
    [0, 1, 30.5],
    [-4.0, 2.0, 300000],
  ]
  vec3 = flex.vec3_double(test_data)
  np_vec3 = vec3.as_numpy_array()
  assert np.all(np.isclose(np_vec3, np.array(test_data)))

def exercise_fixed_width_int_types():
  try:
    import numpy as np
  except ImportError:
    print("No numpy, so skip exercise_fixed_width_int_types")
    return

  # test numpy conversion
  for itype, dtype in zip(
    [flex.int8, flex.int16, flex.int32, flex.int64,
     flex.uint8, flex.uint16, flex.uint32, flex.uint64],
    ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']):
    f = itype([1, 2, 3, 4, 5])
    n = f.as_numpy_array()
    assert n.dtype == getattr(np, dtype)

  # verify rollover behaviour
  u8 = flex.uint8([1, 2, 3, 4, 5])
  u8 += 0xff
  assert flex.max(u8) == 4
  assert flex.min(u8) == 0

  # int8
  i8 = flex.int8([1, 2, 3, 4, 5])
  assert i8.as_numpy_array().dtype == np.int8

  # verify rollover behaviour
  i8 += 0x7f
  assert flex.min(i8) == -128

  # test overflow for signed types
  for itype, maxvalue in zip(
    [flex.int8, flex.int16, flex.int32, flex.int64],
    [0x7f, 0x7fff, 0x7fffffff, 0x7fffffffffffffff]):
    a = itype([maxvalue])
    try:
      a = itype([maxvalue + 1])
    except OverflowError as e:
      pass
    else:
      raise RuntimeError("should have OverflowError")

  print("Ok")

def exercise_numpy_conversions():
  try:
    import numpy as np
  except ImportError:
    print("Skipping exercise_numpy_conversions (numpy not available)")
    return

  for npy_type_name in ['int8', 'int16', 'int32', 'int64',
                        'uint8', 'uint16', 'uint32', 'uint64',
                        'single', 'double']:
    npy_type = getattr(np, npy_type_name)
    for flex_type_name in ['int8', 'int16', 'int32', 'int64',
                           'uint8', 'uint16', 'uint32', 'uint64',
                           'int', 'long', 'float', 'double']:
      flex_type = getattr(flex, flex_type_name)
      l = 2**7 - 1
      n = np.arange(l).astype(npy_type)
      f = flex_type(n)
      for i in range(l):
        assert(approx_equal(n[i], f[i]))

def run(iterations):
  i = 0
  while (iterations == 0 or i < iterations):
    exercise_flex_sum_axis()
    exercise_nd_slicing()
    exercise_set_nd_slicing()
    exercise_numpy_slicing_compatibility()
    exercise_matrix_packed_u_diagonal()
    exercise_versa_packed_u_to_flex()
    exercise_quadratic_form()
    exercise_approx_equal()
    exercise_triangular_systems()
    exercise_copy_upper_or_lower_triangle()
    exercise_matrix_bidiagonal()
    exercise_c_grid_flex_conversion()
    exercise_first_index_etc()
    exercise_flex_grid()
    exercise_flex_constructors()
    exercise_numbers_from_string()
    exercise_std_string()
    exercise_misc()
    exercise_1d_slicing()
    exercise_push_back_etc()
    exercise_setitem()
    exercise_select()
    exercise_from_stl_vector()
    exercise_operators()
    exercise_bitwise_operators()
    exercise_bool()
    exercise_arith_inplace_operators()
    exercise_functions()
    exercise_complex_functions()
    exercise_random()
    exercise_sort()
    exercise_flex_vec3_double()
    exercise_flex_vec2_double()
    exercise_flex_vec3_int()
    exercise_flex_sym_mat3_double()
    exercise_flex_mat3_double()
    exercise_flex_tiny_size_t_2()
    exercise_histogram()
    exercise_weighted_histogram()
    exercise_show_count_stats()
    exercise_linear_regression()
    exercise_linear_correlation()
    exercise_mean_and_variance()
    exercise_linear_interpolation()
    exercise_loops()
    exercise_extract_attributes()
    exercise_exceptions()
    exercise_matrix()
    exercise_matrix_int()
    exercise_matrix_norms()
    exercise_matrix_move()
    exercise_matrix_inversion_in_place()
    exercise_pickle_single_buffered()
    exercise_pickle_double_buffered()
    pickle_large_arrays(max_exp=2, verbose=0)
    exercise_py_object()
    exercise_condense_as_ranges()
    exercise_python_functions()
    exercise_vec3_double_as_numpy_array()
    exercise_fixed_width_int_types()
    exercise_numpy_conversions()
    i += 1

if (__name__ == "__main__"):
  Flags = command_line.parse_options(sys.argv[1:], (
    "large",
  ))
  n = 1
  if (len(sys.argv) > 1 + Flags.n):
    n = int(Flags.regular_args[0])
  if (Flags.large):
    pickle_large_arrays(max_exp=n, verbose=1)
  else:
    run(n)
  print("OK")
