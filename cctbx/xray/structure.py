# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from cctbx.xray import ext
from cctbx.xray import structure_factors
from cctbx import miller
from cctbx import crystal
from cctbx import sgtbx
import cctbx.eltbx.xray_scattering
from cctbx import adptbx
from cctbx import eltbx
from cctbx.array_family import flex
import scitbx.math
from scitbx import matrix
import math
from itertools import count
import sys
import random
import six
from libtbx.utils import count_max, Sorry, Keep
from libtbx.test_utils import approx_equal
from libtbx import group_args
from cctbx.eltbx.neutron import neutron_news_1992_table
from cctbx import eltbx
from libtbx.utils import format_float_with_standard_uncertainty \
     as format_float_with_su
from cctbx import covariance
from six.moves import range
from six.moves import zip

class scattering_type_registry_params(object):
  def __init__(self,
               custom_dict = None,
               d_min       = None,
               table       = None,
               types_without_a_scattering_contribution = None):
    self.custom_dict = custom_dict
    self.d_min       = d_min
    self.table       = table
    self.types_without_a_scattering_contribution = \
                                        types_without_a_scattering_contribution

class structure(crystal.special_position_settings):
  """A class to describe and handle information related to a crystal structure.

  It offers various methods to modify the crystallographic information contained.

  Important members are:

  - .special_position_settings (base class)
  - .scatterers
  - .site_symmetry
  - .crystal_symmetry

  """
  def __init__(self,
        special_position_settings=None,
        scatterers=None,
        site_symmetry_table=None,
        non_unit_occupancy_implies_min_distance_sym_equiv_zero=False,
        scattering_type_registry=None,
        crystal_symmetry=None,
        wavelength=None):
    assert [special_position_settings, crystal_symmetry].count(None) == 1
    assert scatterers is not None or site_symmetry_table is None
    if (special_position_settings is None):
      special_position_settings = crystal.special_position_settings(
        crystal_symmetry=crystal_symmetry)
    crystal.special_position_settings._copy_constructor(
      self, special_position_settings)
    self.erase_scatterers()
    self._non_unit_occupancy_implies_min_distance_sym_equiv_zero \
      = non_unit_occupancy_implies_min_distance_sym_equiv_zero
    self._scattering_type_registry = scattering_type_registry
    if (scatterers is not None):
      self.add_scatterers(
        scatterers=scatterers,
        site_symmetry_table=site_symmetry_table,
        non_unit_occupancy_implies_min_distance_sym_equiv_zero=
          self._non_unit_occupancy_implies_min_distance_sym_equiv_zero)
    self.scattering_type_registry_params = None
    self.inelastic_form_factors_source = None
    self.wavelength = wavelength

  def _copy_constructor(self, other):
    crystal.special_position_settings._copy_constructor(
      self, special_position_settings)
    self._scatterers = other._scatterers
    self._site_symmetry_table = other._site_symmetry_table
    self._non_unit_occupancy_implies_min_distance_sym_equiv_zero \
      = other._non_unit_occupancy_implies_min_distance_sym_equiv_zero
    self._scattering_type_registry = other._scattering_type_registry
    self._scattering_type_registry_is_out_of_date \
      = other._scattering_type_registry_is_out_of_date
    self.inelastic_form_factors_source = other.inelastic_form_factors_source

  def crystal_symmetry(self):
    """Get crystal symmetry of the structure

    :returns: a new crystal symmetry object
    :rtype: cctbx.crystal.symmetry
    """
    return crystal.symmetry(
      unit_cell = self.unit_cell(),
      space_group_info = self.space_group_info())

  def set_non_unit_occupancy_implies_min_distance_sym_equiv_zero(self, value):
    self._non_unit_occupancy_implies_min_distance_sym_equiv_zero = value

  def erase_scatterers(self):
    """Remove all scatterers from structure

    :returns: none
    """
    self._scatterers = flex.xray_scatterer()
    self._site_symmetry_table = sgtbx.site_symmetry_table()
    self._scattering_type_registry_is_out_of_date = True
    self.inelastic_form_factors_source = None

  def deep_copy_scatterers(self):
    """Create a deep copy of the structure with all scatterers

    :returns: a new cctbx.xray.structure object
    :rtype: cctbx.xray.structure
    """
    cp = structure(self,
      scattering_type_registry=self._scattering_type_registry,
      non_unit_occupancy_implies_min_distance_sym_equiv_zero
        =self._non_unit_occupancy_implies_min_distance_sym_equiv_zero,
      wavelength=self.wavelength)
    cp._scatterers = self._scatterers.deep_copy()
    cp._site_symmetry_table = self._site_symmetry_table.deep_copy()
    return cp

  def customized_copy(self,
        crystal_symmetry=Keep,
        unit_cell=Keep,
        space_group_info=Keep,
        non_unit_occupancy_implies_min_distance_sym_equiv_zero=Keep,
        wavelength=Keep):
    if (crystal_symmetry is Keep):
      crystal_symmetry = self
    crystal_symmetry = crystal.symmetry.customized_copy(
      crystal_symmetry,
      unit_cell=unit_cell,
      space_group_info=space_group_info)
    if (non_unit_occupancy_implies_min_distance_sym_equiv_zero is Keep):
      non_unit_occupancy_implies_min_distance_sym_equiv_zero \
        = self._non_unit_occupancy_implies_min_distance_sym_equiv_zero
    if (wavelength is Keep):
      wavelength = self.wavelength
    str = structure(
      special_position_settings=crystal.special_position_settings(
        crystal_symmetry=crystal_symmetry,
        min_distance_sym_equiv=self._min_distance_sym_equiv,
        u_star_tolerance=self._u_star_tolerance,
        assert_min_distance_sym_equiv=self._assert_min_distance_sym_equiv),
      scatterers=self._scatterers,
      non_unit_occupancy_implies_min_distance_sym_equiv_zero
        =non_unit_occupancy_implies_min_distance_sym_equiv_zero,
      scattering_type_registry=self._scattering_type_registry,
      wavelength=wavelength)
    str.inelastic_form_factors_source = self.inelastic_form_factors_source
    return str

  def scatterers(self):
    """Get all scatterers of the structure

    :returns: a reference to an array of cctbx.xray.scatterer
    :rtype: cctbx.xray.scatterer[]
    """
    return self._scatterers

  def non_unit_occupancy_implies_min_distance_sym_equiv_zero(self):
    return self._non_unit_occupancy_implies_min_distance_sym_equiv_zero

  def set_u_iso(self, value = None, values = None, selection = None):
    """Set isotropic mean thermic displacements of scatterers

    :param value: a single double value to set all u_iso of selected \
    scatterers to
    :type value: double
    :param values: an array of double values to set all u_iso of selected \
    scatterers to
    :type values: double[]
    :param selection: an array of bools to select scatterers to be updated \
    with new u_iso values
    :type selection: boolean[]

    :returns: the modified base object
    :rtype: cctbx.xray.structure
    """
    assert [value, values].count(None) == 1
    s = self._scatterers
    if(selection is None): selection = flex.bool(s.size(), True)
    else:                  assert selection.size() == s.size()
    if(value is not None):
      s.set_u_iso(flex.double(s.size(), value), selection, self.unit_cell())
    else:
      assert values.size() == s.size()
      s.set_u_iso(values, selection, self.unit_cell())
    return self

  def set_b_iso(self, value = None, values = None, selection = None):
    """Set isotropic Debye-Waller/temperature/B factors with automatic conversion
    to u_iso

    :param value: a single double value to set all b_iso of selected \
    scatterers to
    :type value: double
    :param values: an array of double values to set all b_iso of selected \
    scatterers to
    :type values: double[]
    :param selection: an array of bools to select scatterers to be updated with \
    new b_iso values
    :type selection: boolean[]

    :returns: the modified base object
    :rtype: cctbx.xray.structure
    """
    assert [value, values].count(None) == 1
    s = self._scatterers
    if(value is not None):
      self.set_u_iso(value = adptbx.b_as_u(value), selection = selection)
    else:
      assert values.size() == s.size()
      b_iso = values
      u_iso_values = b_iso*adptbx.b_as_u(1)
      self.set_u_iso(values = u_iso_values, selection = selection)
    return self

  def random_remove_sites_selection(self, fraction):
    scatterers_size = self._scatterers.size()
    if(abs(fraction-0.0) < 1.e-3):
      return flex.bool(scatterers_size, True)
    if(fraction < 0.01 or fraction > 0.99):
      raise RuntimeError("fraction must be between 0.01 and 0.99.")
    tol = 999.
    selection = None
    l = max(fraction - 0.05, 0.0)
    r = min(fraction + 0.05, 1.0)
    for i in range(5):
      while l <= r:
        arr = flex.random_double(scatterers_size)-l
        sel = arr > 0.0
        deleted = float((scatterers_size - sel.count(True))) / scatterers_size
        if abs(fraction - deleted) < tol:
          tol = abs(fraction - deleted)
          selection = sel
        l += 0.0001
    return selection

  def replace_sites_frac(self, new_sites, selection=None):
    if(selection is not None):
      new_sites = self.sites_frac().set_selected(selection, new_sites)
    cp = structure(self,
      non_unit_occupancy_implies_min_distance_sym_equiv_zero
        =self._non_unit_occupancy_implies_min_distance_sym_equiv_zero,
      scattering_type_registry=self._scattering_type_registry,
      wavelength=self.wavelength)
    new_scatterers = self._scatterers.deep_copy()
    new_scatterers.set_sites(new_sites)
    cp._scatterers = new_scatterers
    cp._site_symmetry_table = self._site_symmetry_table.deep_copy()
    return cp

  def replace_sites_cart(self, new_sites, selection=None):
    return self.replace_sites_frac(
      new_sites=self.unit_cell().fractionalize(sites_cart=new_sites),
      selection=selection)

  def adjust_u_iso(self):
    self._scatterers.adjust_u_iso()

  def adjust_occupancy(self, occ_max, occ_min, selection = None):
    """Adjust site occupancy factor for selected sites to be between occ_min and
    occ_max.

    :param occ_max: maximal site occupancy factor
    :type occ_max: float
    :param occ_min: minimal site occupancy factor
    :type occ_min: float
    :param selection: an array of bools to select scatterers to be adjusted
    :type selection: boolean[]

    :returns: none
    """
    if(selection is not None):
      if(("%s"%selection.__class__).count("array_family_flex_ext.size_t") > 0):
        selection = flex.bool(self._scatterers.size(), selection)
    occ = self._scatterers.extract_occupancies()
    sel = (occ >= occ_max)
    occ = occ.set_selected(sel, occ_max)
    sel = (occ <= occ_min)
    occ = occ.set_selected(sel, occ_min)
    if(selection is None):
      self._scatterers.set_occupancies(occ)
    else:
      self._scatterers.set_occupancies(occ, selection)

  def all_selection(self):
    """Get a selector array for all scatterers of the structure.

    :returns: an array to select all scatterers of the structure
    :rtype: boolean[]
    """
    return flex.bool(self._scatterers.size(), True)

  def translate(self, x=0, y=0, z=0):
    """Translates all scatterers of this structure by x,y,z.

    :param x: x component of the translation vector
    :type x: float
    :param y: y component of the translation vector
    :type y: float
    :param z: z component of the translation vector
    :type z: float

    :returns: a new translated copy of the structure
    :rtype: cctbx.xray.structure
    """
    sites_cart = self.sites_cart()
    cp = structure(self,
      non_unit_occupancy_implies_min_distance_sym_equiv_zero
        =self._non_unit_occupancy_implies_min_distance_sym_equiv_zero,
      scattering_type_registry=self._scattering_type_registry,
      wavelength=self.wavelength)
    new_scatterers = self._scatterers.deep_copy()
    new_scatterers.set_sites(
      self.unit_cell().fractionalize(
        sites_cart=sites_cart+flex.vec3_double(sites_cart.size(),[x,y,z])))
    cp._scatterers = new_scatterers
    cp._site_symmetry_table = self._site_symmetry_table.deep_copy()
    return cp

  def distances(self, other, selection = None):
    """Calculates pairwise distances between the atoms of this structure and
    another structure with the same number of scatterers.

    :param other: the other structure
    :type other: cctbx.xray.structure
    :param selection: an array of bools to select scatterers to be taken into \
    calculation
    :type selection: boolean[]

    :returns: an array of distances for the selected scatterers
    :rtype: float[]
    """
    if(selection is None): selection = flex.bool(self._scatterers.size(), True)
    s1 = self.sites_cart().select(selection)
    s2 = other.sites_cart().select(selection)
    if(s1.size() != s2.size()):
      raise RuntimeError("Models must be of equal size.")
    return flex.sqrt((s1 - s2).dot())

  def max_distance(self, other, selection = None):
    """Calculates the maximum pairwise distance between the atoms of this
    structure and another structure with the same number of scatterers.

    :param other: the other structure
    :type other: cctbx.xray.structure
    :param selection: an array of bools to select scatterers to be taken into \
    calculation
    :type selection: boolean[]

    :returns: the maximum distance of two corresponding scatterers out of the \
    selected scatterers
    :rtype: float
    """
    return flex.max( self.distances(other = other, selection = selection) )

  def min_distance(self, other, selection = None):
    """Calculates the minimum pairwise distance between the atoms of this
    structure and another structure with the same number of scatterers.

    :param other: the other structure
    :type other: cctbx.xray.structure
    :param selection: an array of bools to select scatterers to be taken into \
    calculation
    :type selection: boolean[]

    :returns: the minimum distance of two corresponding scatterers out of the \
    selected scatterers
    :rtype: float
    """
    return flex.min( self.distances(other = other, selection = selection) )

  def mean_distance(self, other, selection = None):
    """Calculates the arithmetic mean pairwise distance between the atoms
    of this structure and another structure with the same number of scatterers.

    :param other: the other structure
    :type other: cctbx.xray.structure
    :param selection: an array of bools to select scatterers to be taken into \
    calculation
    :type selection: boolean[]

    :returns: the mean pairwise distance of the selected scatterers
    :rtype: float
    """
    return flex.mean( self.distances(other = other, selection = selection) )

  def scale_adp(self, factor, selection=None):
    """Scale the atomic displacement parameters of the selected scatterers
    of the structure with the specified factor.
    If no selection is given, all scatterers will be handled as if selected.

    :param factor: scale factor to apply to the adps of the selected scatterers
    :type factor: float
    :param selection: an array of bools to select the scatterers to have their \
    adps scaled
    :type selection: boolean[]

    :returns: none
    """
    if(selection is not None):
      assert selection.size() == self._scatterers.size()
    else:
      selection = flex.bool(self._scatterers.size(), True)
    for sc,sel in zip(self._scatterers, selection):
      if(sel and sc.flags.use()):
        if(sc.flags.use_u_iso()):
          sc.u_iso = sc.u_iso * factor
        if(sc.flags.use_u_aniso()):
          result = []
          for i in range(6): result.append(sc.u_star[i] * factor)
          sc.u_star = result

  def shake_adp(self, b_max=None, b_min=None, spread=10.0, aniso_spread=0.1,
             keep_anisotropic=False, random_u_cart_scale=1.0, selection=None):
    assert [b_max, b_min].count(None) in [0,2]
    if([b_max, b_min].count(None) == 0): assert spread == 0.0
    if([b_max, b_min].count(None) == 2):
      u_isos = self.extract_u_iso_or_u_equiv().select(self.use_u_iso())
      if(u_isos.size() > 0):
        b_mean = adptbx.u_as_b(flex.mean(u_isos))
        b_max = int(b_mean + spread)
        b_min = int(max(0.0, b_mean - spread))
    if b_min is not None and b_max is not None:
      assert b_min <= b_max, [b_min,b_max,spread,b_mean]
      b_min = int(b_min)
      b_max = int(b_max)
    if(selection is not None):
      assert selection.size() == self._scatterers.size()
    else:
      selection = flex.bool(self._scatterers.size(), True)
    is_special_position = self.site_symmetry_table().is_special_position
    for i_seq,sc,sel in zip(count(), self._scatterers, selection):
      if(sel and sc.flags.use()):
        if(sc.flags.use_u_iso() and b_min != b_max):
          r = max(0, random.randrange(b_min, b_max, 1) + random.random())
          sc.u_iso=adptbx.b_as_u(r)
        if(sc.flags.use_u_aniso() and not keep_anisotropic):
          result = []
          for i in range(6):
            result.append(sc.u_star[i]+sc.u_star[i]*random.choice(
                                             (-aniso_spread,aniso_spread)))
          if(is_special_position(i_seq=i_seq)):
            result = self.space_group().average_u_star(result)
          sc.u_star = result

  def shake_adp_if_all_equal(self, b_iso_tolerance = 0.1):
    performed = False
    if(self.use_u_aniso().count(True) == 0):
      u_isos = self.extract_u_iso_or_u_equiv()
      b_max  = adptbx.u_as_b(flex.max(u_isos))
      b_min  = adptbx.u_as_b(flex.min(u_isos))
      b_mean = adptbx.u_as_b(flex.mean(u_isos))
      if(abs(b_max - b_mean) <= b_iso_tolerance and
                                      abs(b_min - b_mean) <= b_iso_tolerance):
          self.shake_adp()
          performed = True
    return performed

  def min_u_cart_eigenvalue(self):
    u_carts = self._scatterers.extract_u_cart_plus_u_iso(
      unit_cell=self.unit_cell())
    result = flex.double()
    for i_seq, sc in enumerate(self._scatterers):
      if(sc.flags.use_u_iso() or sc.flags.use_u_aniso()):
        result.append(min(adptbx.eigenvalues(u_carts[i_seq])))
    return flex.min(result)

  def shake_occupancies(self, selection = None):
    s = self._scatterers
    q_new = flex.random_double(s.size())
    if(selection is None):
      s.set_occupancies(q_new)
    else:
      assert selection.size() == s.size()
      s.set_occupancies(q_new, selection)

  def shake_fps(self, selection = None):
    s = self._scatterers
    q_deltas = flex.double([random.gauss(0,1) for _ in range(s.size())])
    q_new = q_deltas + flex.double([sc.fp for sc in s])
    if(selection is None):
      s.set_fps(q_new)
    else:
      assert selection.size() == s.size()
      s.set_fps(q_new, selection)

  def shake_fdps(self, selection = None):
    s = self._scatterers
    q_deltas = flex.double([random.gauss(0,1) for _ in range(s.size())])
    q_new = q_deltas + flex.double([sc.fdp for sc in s])
    if(selection is None):
      s.set_fdps(q_new)
    else:
      assert selection.size() == s.size()
      s.set_fdps(q_new, selection)

  def set_occupancies(self, value, selection = None):
    if(selection is not None and isinstance(selection, flex.size_t)):
      selection = flex.bool(self._scatterers.size(), selection)
    s = self._scatterers
    if(hasattr(value, 'size')):
      values = value
      if(selection is not None):
        assert values.size() == selection.size()
    else:
      values = flex.double(s.size(), value)
    if(selection is None):
      s.set_occupancies(values)
    else:
      assert selection.size() == s.size()
      s.set_occupancies(values, selection)
    return self

  def set_fps(self, value, selection = None):
    if(selection is not None and isinstance(selection, flex.size_t)):
      selection = flex.bool(self._scatterers.size(), selection)
    s = self._scatterers
    if(hasattr(value, 'size')):
      values = value
      if(selection is not None):
        assert values.size() == selection.size()
    else:
      values = flex.double(s.size(), value)
    if(selection is None):
      s.set_fps(values)
    else:
      assert selection.size() == s.size()
      s.set_fps(values, selection)
    return self

  def set_fdps(self, value, selection = None):
    if(selection is not None and isinstance(selection, flex.size_t)):
      selection = flex.bool(self._scatterers.size(), selection)
    s = self._scatterers
    if(hasattr(value, 'size')):
      values = value
      if(selection is not None):
        assert values.size() == selection.size()
    else:
      values = flex.double(s.size(), value)
    if(selection is None):
      s.set_fdps(values)
    else:
      assert selection.size() == s.size()
      s.set_fdps(values, selection)
    return self

  def coordinate_degrees_of_freedom_counts(self, selection=None):
    assert selection is None or selection.size() == self._scatterers.size()
    site_symmetry_table = self._site_symmetry_table
    assert site_symmetry_table.indices().size() == self._scatterers.size()
    result = {
      0: 0,
      1: 0,
      2: 0,
      3: -site_symmetry_table.special_position_indices().size()}
    if (selection is None):
      result[3] += self._scatterers.size()
    else:
      result[3] += selection.count(True)
    for i in site_symmetry_table.special_position_indices():
      if (selection is None or selection[i]):
        result[site_symmetry_table
                 .get(i).site_constraints()
                   .n_independent_params()] += 1
    return result

  def get_scattering_table(self):
    if not self.scattering_type_registry_params:
        return None
    return self.scattering_type_registry_params.table

  def guess_scattering_type_neutron(self):
    ac,bc,cc = 0,0,0
    result = False
    for ugl in self.scattering_type_registry().unique_gaussians_as_list():
      ac += len(ugl.array_of_a())
      bc += len(ugl.array_of_b())
      cc += ugl.c()
    if(ac+bc == 0 and cc != 0): result = True
    return result

  def guess_scattering_type_is_a_mixture_of_xray_and_neutron(self):
    has_xray = False
    has_neutron = False
    for ugl in self._scattering_type_registry.unique_gaussians_as_list():
      if ugl is None:
        continue
      if len(ugl.array_of_a())!=0 or len(ugl.array_of_b())!=0:
        has_xray = True
      elif ugl.c() != 0:
        has_neutron = True
    return has_neutron and has_xray

  def scattering_dictionary_as_string(self):
    result = ""
    if self._scattering_type_registry is None:
      return result
    for tg in self._scattering_type_registry.as_type_gaussian_dict().items():
      stype = tg[0]
      gaussian = tg[1]
      aa = "None"
      ab = "None"
      c = "None"
      if gaussian is not None:
        aa = gaussian.array_of_a()
        ab = gaussian.array_of_b()
        c = gaussian.c()
      result += "\n  Element: "+stype +"  a: "+ str(aa)+ "  b: "+ str(ab)+ "  c: "+str(c)
    return result

  def scattering_types_counts_and_occupancy_sums(self):
    result = []
    reg = self.scattering_type_registry()
    unique_counts = reg.unique_counts
    if (flex.sum(unique_counts) != self._scatterers.size()):
      raise RuntimeError("scattering_type_registry out of date.")
    occupancy_sums = reg.occupancy_sums(self._scatterers)
    unit_cell_occupancy_sums = reg.unit_cell_occupancy_sums(self._scatterers)
    for scattering_type,unique_index in reg.type_index_pairs_as_dict().items():
      result.append(group_args(
        scattering_type=scattering_type,
        count=unique_counts[unique_index],
        occupancy_sum=occupancy_sums[unique_index],
        unit_cell_occupancy_sum=unit_cell_occupancy_sums[unique_index]))
    return result

  def crystal_density(self):
    """Get the value of the diffraction-determined density for the crystal,
    suitable for the CIF item _exptl_crystal_density_diffrn

    Density values are calculated from the crystal cell and contents. The
    units are megagrams per cubic metre (=grams per cubic centimetre).

    Equivalent to:
      1.66042 * _chemical_formula_weight * _cell_formula_units_Z / _cell_volume

    :returns: chemical density in megagrams per cubic metre (=grams per cm^3)
    :rtype: float
    """
    from cctbx.eltbx import tiny_pse
    numerator = sum([
      tiny_pse.table(elt.scattering_type).weight() * elt.unit_cell_occupancy_sum
      for elt in self.scattering_types_counts_and_occupancy_sums()])
    denominator = self.unit_cell().volume()
    return 1.66042 * numerator/denominator

  def f_000(self, include_inelastic_part=False):
    """Get the effective number of electrons in the crystal unit cell
    contributing to F(000), suitable for the CIF item _exptl_crystal_F_000.

    According to the CIF definition, this item **may** contain dispersion
    contributions.

    :param include_inelastic_part: If 'True' contributions due to dispersion \
    are included in F(000).
    :type include_inelastic_part: boolean

    :returns: F(000)
    :rtype: float
    """
    elastic_part = 0
    reg = self.scattering_type_registry()
    unique_counts = reg.unique_counts
    if (flex.sum(unique_counts) != self._scatterers.size()):
      raise RuntimeError("scattering_type_registry out of date.")
    unit_cell_occupancy_sums = reg.unit_cell_occupancy_sums(self._scatterers)
    unique_form_factors_at_origin = reg.unique_form_factors_at_d_star_sq(0)
    for scattering_type,unique_index in reg.type_index_pairs_as_dict().items():
      elastic_part +=   unit_cell_occupancy_sums[unique_index] \
                      * unique_form_factors_at_origin[unique_index]
    if not include_inelastic_part:
      return elastic_part
    inelastic_part_real = 0
    inelastic_part_imag = 0
    for sc in self.scatterers():
      if sc.fp:
        inelastic_part_real += sc.fp * sc.occupancy * sc.multiplicity()
      if sc.fdp:
        inelastic_part_imag += sc.fdp * sc.occupancy * sc.multiplicity()
    return abs(complex(elastic_part+inelastic_part_real, inelastic_part_imag))

  def shake_sites_in_place(self,
        rms_difference=None,
        mean_distance=None,
        selection=None,
        allow_all_fixed=False,
        random_double=None):
    """Shake the coordinates of the selected scatterers in this structure.

    :param rms_difference: radial mean square displacement (>=0) to apply to \
    selected scatterers
    :type rms_difference: float
    :param mean_distance: a mean distance shift (>=0) to apply to selected \
    scatterers
    :type mean_distance: float
    :param selection: an array of bools to select scatterers to be shaken
    :type selection: boolean[]
    :param allow_all_fixed: if set to 'True' shaking a structure with all \
    scatterers on fixed special positions will not cause an error
    :type allow_all_fixed: boolean
    :param random_double: "random" numbers to use for displacements
    :type random_double: float[]

    :returns: 'True' if at least one scatterer was moved, 'False' otherwise
    :rtype: boolean
    """
    assert [rms_difference, mean_distance].count(None) == 1
    if (rms_difference is not None):
      assert rms_difference >= 0
      target_difference = rms_difference
    else:
      assert mean_distance >= 0
      target_difference = mean_distance
    if (target_difference == 0): return
    assert self._scatterers.size() > 0
    site_symmetry_table = self._site_symmetry_table
    assert site_symmetry_table.indices().size() == self._scatterers.size()
    if (selection is not None):
      assert selection.size() == self._scatterers.size()
      n_variable = selection.count(True)
      if (n_variable == 0):
        raise RuntimeError("No scatterers selected.")
      all = " selected"
    else:
      n_variable = self._scatterers.size()
      all = ""
    selection_fixed = flex.size_t()
    for i in site_symmetry_table.special_position_indices():
      if (site_symmetry_table.get(i)
            .site_constraints()
               .n_independent_params() == 0):
        if (selection is None or selection[i]):
          selection_fixed.append(i)
    n_variable -= selection_fixed.size()
    if (n_variable == 0):
      if (allow_all_fixed):
        return False
      raise RuntimeError(
        "All%s scatterers are fixed on special positions." % all)
    if (n_variable == self._scatterers.size()):
      selection = None
    scatterers = self._scatterers
    frac = self.unit_cell().fractionalize
    orth = self.unit_cell().orthogonalize
    if (random_double is None):
      random_double = flex.random_double
    for i in count_max(assert_less_than=10):
      shifts_cart = flex.vec3_double(random_double(
        size=self._scatterers.size()*3, factor=2) - 1)
      if (selection is not None):
        shifts_cart.set_selected(~selection, (0,0,0))
      shifts_cart.set_selected(selection_fixed, (0,0,0))
      for i in site_symmetry_table.special_position_indices():
        site_frac_orig = matrix.col(scatterers[i].site)
        site_frac = site_symmetry_table.get(i).special_op() \
                  * (site_frac_orig + matrix.col(frac(shifts_cart[i])))
        shifts_cart[i] = orth(matrix.col(site_frac) - site_frac_orig)
      if (rms_difference is not None):
        difference = (flex.sum(shifts_cart.dot()) / n_variable) ** 0.5
      else:
        difference = flex.sum(flex.sqrt(shifts_cart.dot())) / n_variable
      if (difference > 1.e-6): break # to avoid numerical problems
    shifts_cart *= (target_difference / difference)
    self.set_sites_frac(
      self.sites_frac() + self.unit_cell().fractionalize(shifts_cart))
    return True

  def shift_sites_in_place(self, shift_length, mersenne_twister=None):
    """Shifts the coordinates of all scatterers in this structure.

    :param shift_length: the distance to shift each scatterer with
    :type shift_length: float
    :param mersenne_twister: a mersenne twister to use as entropy source
    :type mersenne_twister: flex.mersenne_twister

    :returns: none
    """
    if (shift_length == 0): return
    sst = self._site_symmetry_table
    assert sst.indices().size() == self._scatterers.size()
    frac = self.unit_cell().fractionalize
    orth = self.unit_cell().orthogonalize
    if (mersenne_twister is None):
      mersenne_twister = flex.mersenne_twister(seed=0)
    col = matrix.col
    for i_sc,sc in enumerate(self._scatterers):
      site_frac = col(sc.site)
      ss = sst.get(i_sc)
      constr = ss.site_constraints()
      np = constr.n_independent_params()
      if (np == 0):
        continue
      if (np == 3):
        def find_3():
          sl = shift_length
          while (sl != 0):
            for i_trial in range(10):
              shift_frac = col(frac(
                col(mersenne_twister.random_double_point_on_sphere()) * sl))
              site_mod = site_frac + shift_frac
              ss_mod = self.site_symmetry(site=site_mod)
              if (ss_mod.is_point_group_1()):
                sc.site = site_mod
                return
            sl *= 0.5
        find_3()
      elif (np == 2):
        plane_vectors = []
        for s0 in [-1,0,1]:
          for s1 in [-1,0,1]:
            indep = list(constr.independent_params(site_frac))
            indep[0] += s0
            indep[1] += s1
            plane_vectors.append(col(orth(
              col(constr.all_params(indep)) - site_frac)))
        assert len(plane_vectors) == 9
        axis = None
        axis_length = None
        for i in range(8):
          vi = plane_vectors[i]
          for j in range(i+1,9):
            vj = plane_vectors[j]
            cross = vi.cross(vj)
            length = cross.length()
            if (axis is None or length > axis_length):
              axis = cross
              axis_length = length
        assert axis is not None
        assert axis_length != 0
        v_max = None
        l_max = None
        for v in plane_vectors:
          l = v.length()
          if (l_max is None or l > l_max):
            v_max = v
            l_max = l
        assert v_max is not None
        def find_2():
          sl = shift_length
          while (sl != 0):
            for i_trial in count_max(assert_less_than=10):
              r = axis.axis_and_angle_as_r3_rotation_matrix(
                angle = mersenne_twister.random_double() * 2 * math.pi)
              shift_frac = col(frac((r * v).normalize() * sl))
              site_mod = site_frac + shift_frac
              ss_mod = self.site_symmetry(site=site_mod)
              if (ss_mod.special_op() == ss.special_op()):
                sc.site = site_mod
                return
            sl *= 0.5
        find_2()
      else:
        def find_1():
          sl = shift_length
          while (sl != 0):
            if (mersenne_twister.random_double() < 0.5):
              us = [1, -1]
            else:
              us = [-1, 1]
            for u in us:
              indep = list(constr.independent_params(site_frac))
              indep[0] += u
              v = col(orth(col(constr.all_params(indep)) - site_frac))
              assert v.length() != 0
              shift_frac = col(frac(v.normalize() * shift_length))
              site_mod = site_frac + shift_frac
              ss_mod = self.site_symmetry(site=site_mod)
              if (ss_mod.special_op() == ss.special_op()):
                sc.site = site_mod
                return
            sl *= 0.5
        find_1()

  def b_iso_min_max_mean(self):
    """Get the minimal, maximal and mean isotropic Debye-Waller/temperature/B \
    factors of all scatterers in this structure.

    :returns: minimal b_iso, maximal b_iso, mean b_iso
    :rtype: float, float, float
    """
    b_isos = self._scatterers.extract_u_iso()/adptbx.b_as_u(1)
    b_min  = flex.min(b_isos)
    b_max  = flex.max(b_isos)
    b_mean = flex.mean(b_isos)
    return b_min, b_max, b_mean

  def discard_scattering_type_registry(self):
    self._scattering_type_registry_is_out_of_date = True

  def n_undefined_multiplicities(self):
    return ext.n_undefined_multiplicities(self._scatterers)

  def sites_frac(self):
    return self._scatterers.extract_sites()

  def use_u_iso(self):
    return self._scatterers.extract_use_u_iso()

  def use_u_aniso(self):
    return self._scatterers.extract_use_u_aniso()

  def set_sites_frac(self, sites_frac):
    """Set the the fractional coordinates of all sites of the structure to \
    'sites_frac'.

    :param sites_frac: a list of the fractional coordinates for all scatterers
    :type sites_frac: scitbx.array_family.flex.vec3_double

    :returns: none
    """
    assert sites_frac.size() == self._scatterers.size()
    self._scatterers.set_sites(sites_frac)

  def sites_cart(self):
    """Get the the cartesian coordinates of all sites of the structure.

    :returns: a list of the sites of the structure in cartesian coordinates
    :rtype: scitbx.array_family.flex.vec3_double
    """
    return self.unit_cell().orthogonalize(sites_frac=self.sites_frac())

  def set_sites_cart(self, sites_cart):
    """Set the the cartesian coordinates of all sites of the structure to \
    'sites_cart'.

    :param sites_cart: a list of the cartesian coordinates for all scatterers
    :type sites_cart: scitbx.array_family.flex.vec3_double

    :returns: none
    """
    self.set_sites_frac(self.unit_cell().fractionalize(sites_cart=sites_cart))

  def scattering_types(self):
    result = flex.std_string()
    for sct in self._scatterers.extract_scattering_types():
      result.append(sct.strip().upper())
    return result

  def extract_u_cart_plus_u_iso(self):
    return self._scatterers.extract_u_cart_plus_u_iso(
      unit_cell=self.unit_cell())

  def extract_u_iso_or_u_equiv(self):
    return self._scatterers.extract_u_iso_or_u_equiv(
      unit_cell=self.unit_cell())

  def b_iso_or_b_equiv(self):
    return self.extract_u_iso_or_u_equiv()*adptbx.u_as_b(1.)

  def scale_adps(self, scale_factor):
    return self._scatterers.scale_adps(scale_factor)

  def switch_to_neutron_scattering_dictionary(self):
    # XXX First step. In future: better to do bookkeeping and be able to swith
    # XXX back and forth between original scat_dict and neutron.
    # XXX Add regression test.
    return self.scattering_type_registry(table="neutron").as_type_gaussian_dict()

  def hd_selection(self):
    """Get a selector array for all hydrogen and deuterium scatterers of the structure.

    :returns: an array to select all H and D scatterers of the structure
    :rtype: boolean[]
    """
    scattering_types = self._scatterers.extract_scattering_types()
    result = flex.bool()
    for sct in scattering_types:
      if(sct.strip() in ['H','D']): result.append(True)
      else: result.append(False)
    return result

  def element_selection(self, *elements):
    """Get a selector array for scatterers of specified element type(s) of the structure.

    :param elements: tuple of element symbols to select
    :type elements: list(string) or set(string) or tuple(string)

    :returns: an array to select the desired scatterers of the structure
    :rtype: boolean[]
    """
    return flex.bool([ sc.element_symbol().strip() in elements
                       for sc in self.scatterers() ])

  def by_index_selection(self, selected_scatterers):
    """Get a selector array for scatterers with specified index from the
    structure. For example you can select scatterers 1,4 and 5 by passing
    (1,4,5) as argument.

    :param selected_scatterers: list of scatterers to select
    :type selected_scatterers: list(int)

    :returns: an array to select the desired scatterers of the structure
    :rtype: flex.bool[]
    """
    result = flex.bool(self._scatterers.size(), False)
    try:
      result.set_selected(flex.size_t(selected_scatterers), True)
    except(RuntimeError):
      raise IndexError("Tried to select a scatterer by index with index => "+\
            "of scatterers for this structuture.")
    return result

  def apply_rigid_body_shift(self, rot, trans, selection = None,
        recompute_site_symmetries=True):
    if(selection is None):
      selection = flex.bool(self._scatterers.size(), True).iselection()
    rbs_obj = self.apply_rigid_body_shift_obj(
                                        sites_cart     = self.sites_cart(),
                                        sites_frac     = self.sites_frac(),
                                        rot            = rot,
                                        trans          = trans,
                                        selection      = selection,
                                        unit_cell      = self.unit_cell(),
                                        atomic_weights = self.atomic_weights())
    self.set_sites_frac(sites_frac = rbs_obj.sites_frac)
    if(recompute_site_symmetries):
      scatterers = self.scatterers()
      self.erase_scatterers()
      self.add_scatterers(scatterers = scatterers)

  def apply_rigid_body_shift_obj(self,
                                 sites_cart,
                                 sites_frac,
                                 rot,
                                 trans,
                                 selection,
                                 unit_cell,
                                 atomic_weights):
    return ext.apply_rigid_body_shift(sites_cart     = sites_cart,
                                      sites_frac     = sites_frac,
                                      rot            = rot,
                                      trans          = trans,
                                      atomic_weights = atomic_weights,
                                      unit_cell      = unit_cell,
                                      selection      = selection)

  def convert_to_isotropic(self, selection=None):
    # FIXME selection must be size_t!
    if(selection is None):
      self._scatterers.convert_to_isotropic(unit_cell=self.unit_cell())
    else:
      self._scatterers.convert_to_isotropic(unit_cell=self.unit_cell(),
                                            selection=selection)

  def convert_to_anisotropic(self, selection=None):
    if(selection is None):
      self._scatterers.convert_to_anisotropic(unit_cell=self.unit_cell())
    else:
      self._scatterers.convert_to_anisotropic(unit_cell=self.unit_cell(),
                                              selection=selection)

  def show_u_statistics(self, text="", out=None, use_hydrogens=False):
    if(out is None): out = sys.stdout
    size = self._scatterers.size()
    if(use_hydrogens):
      hd_selection = flex.bool(size, True)
    else:
      hd_selection = self.hd_selection()
    epis = 8*math.pi**2
    use_u_aniso = self.use_u_aniso().select(~hd_selection)
    use_u_iso   = self.use_u_iso().select(~hd_selection)
    sel_used = use_u_aniso | use_u_iso
    n_anisotropic = use_u_aniso.count(True)
    n_isotropic   = use_u_iso.count(True)
    ipd = self.is_positive_definite_u().select(~hd_selection)
    npd = ipd.count(True)
    nnpd = ipd.count(False)
    beq = (self.extract_u_iso_or_u_equiv() * epis).select(~hd_selection).select(sel_used)
    bisos = (self.scatterers().extract_u_iso() * epis).select(~hd_selection).select(use_u_iso)
    if(bisos.size() == 0): bisos = beq
    part1 = "|-"+text
    part2 = "-|"
    n = 79 - len(part1+part2)
    print(part1 + "-"*n + part2, file=out)
    n = 79 - len(part1 + "|")
    print("| iso = %-5d   aniso = %-5d   pos. def. = %-5d   "\
          "non-pos. def. = %-5d     |"%(n_isotropic,n_anisotropic,npd,nnpd), file=out)
    print("| Total B(isotropic equivalent): min = %-6.2f   "\
                  "max = %-6.2f   mean = %-6.2f"%(flex.min(beq),flex.max(beq),
                  flex.mean(beq))+" "*2+"|", file=out)
    print("| Isotropic B only:              min = %-6.2f   "\
                  "max = %-6.2f   mean = %-6.2f"%(flex.min(bisos),
                  flex.max(bisos),flex.mean(bisos))+" "*2+"|", file=out)
    print("| "+"- "*38+"|", file=out)
    print("|                     Distribution of isotropic B-factors:"\
                  "                    |", file=out)
    print("|            Isotropic                |             Total "\
                  "                    |", file=out)
    histogram_1 = flex.histogram(data = bisos, n_slots = 10)
    low_cutoff_1 = histogram_1.data_min()
    histogram_2 = flex.histogram(data = beq, n_slots = 10)
    low_cutoff_2 = histogram_2.data_min()
    for (i_1,n_1),(i_2,n_2) in zip(enumerate(histogram_1.slots()),
                                   enumerate(histogram_2.slots())):
      high_cutoff_1 = histogram_1.data_min() + histogram_1.slot_width()*(i_1+1)
      high_cutoff_2 = histogram_2.data_min() + histogram_2.slot_width()*(i_2+1)
      print("|  %9.3f -%9.3f:%8d      |    %9.3f -%9.3f:%8d      |" % \
             (low_cutoff_1,high_cutoff_1,n_1,low_cutoff_2,high_cutoff_2,n_2), file=out)
      low_cutoff_1 = high_cutoff_1
      low_cutoff_2 = high_cutoff_2
    print("|" +"-"*77+"|", file=out)

  def site_symmetry_table(self):
    return self._site_symmetry_table

  def special_position_indices(self):
    return self._site_symmetry_table.special_position_indices()

  def scattering_type_registry(self,
        custom_dict=None,
        d_min=None,
        table=None,
        types_without_a_scattering_contribution=None,
        explicitly_allow_mixing=False):
    assert table in [None, "n_gaussian", "it1992", "wk1995", "xray", "electron",
        "neutron"]
    if (table == "it1992"): assert d_min in [0,None] or d_min >= 1/4.
    if (table == "wk1995"): assert d_min in [0,None] or d_min >= 1/12.
    if (table == "electron"): assert d_min in [0,None] or d_min >= 1/4.
    if (   self._scattering_type_registry_is_out_of_date
        or custom_dict is not None
        or d_min is not None
        or table is not None
        or types_without_a_scattering_contribution is not None):
      self.scattering_type_registry_params = \
          scattering_type_registry_params(
             custom_dict = custom_dict,
             d_min       = d_min,
             table       = table,
             types_without_a_scattering_contribution = \
               types_without_a_scattering_contribution)
      new_dict = {"const": eltbx.xray_scattering.gaussian(1)}
      old_dict = {}
      last_used_scattering_table = None
      if (self._scattering_type_registry is not None):
        last_used_scattering_table = self._scattering_type_registry.last_table()
        ugs = self._scattering_type_registry.unique_gaussians_as_list()
        tip = self._scattering_type_registry.type_index_pairs_as_dict()
        for t,i in tip.items():
          if (ugs[i] is not None): old_dict[t] = ugs[i]
        if (d_min is None and table is None):
          new_dict.update(old_dict)
      if (types_without_a_scattering_contribution is not None):
        for t in types_without_a_scattering_contribution:
          new_dict[t] = eltbx.xray_scattering.gaussian(0)
      if (custom_dict is not None):
        new_dict.update(custom_dict)
      if (d_min is None): d_min = 0
      self._scattering_type_registry = ext.scattering_type_registry()
      self._scattering_type_registry.process(self._scatterers)
      for t_undef in self._scattering_type_registry.unassigned_types():
        val = new_dict.get(t_undef, None)
        if (val is None):
          std_lbl = eltbx.xray_scattering.get_standard_label(
            label=t_undef, exact=True, optional=True)
          if (std_lbl is not None):
            if table is None:
              table = last_used_scattering_table
            else:
              last_used_scattering_table = table
            if (table == "it1992"):
              val = eltbx.xray_scattering.it1992(std_lbl, True).fetch()
            elif (table == "wk1995"):
              val = eltbx.xray_scattering.wk1995(std_lbl, True).fetch()
            elif (table == "electron"):
              from cctbx.eltbx.e_scattering import \
                ito_vol_c_2011_table_4_3_2_2_entry_as_gaussian as _
              val = _(label=std_lbl, exact=True)
            elif (table == "neutron"):
              scattering_info = neutron_news_1992_table(std_lbl, True)
              b = scattering_info.bound_coh_scatt_length()
              # TODO:
              # b.imag is ignored here. It depends on wavelength, so values
              # from neutron_news_1992 table are not very useful.
              # Warning is printed by scattering_registry::show.
              val = eltbx.xray_scattering.gaussian(b.real)
            else:
              # TODO mrt: this may lead to a mix of xray/neutron dictionary
              val = eltbx.xray_scattering.n_gaussian_table_entry(
                std_lbl, d_min, 0).gaussian()
              last_used_scattering_table = "n_gaussian"
        if (val is not None):
          self._scattering_type_registry.assign(t_undef, val)
          if last_used_scattering_table is not None:
            self._scattering_type_registry.set_last_table(last_used_scattering_table)
      self._scattering_type_registry_is_out_of_date = False
    if not explicitly_allow_mixing:
      if self.guess_scattering_type_is_a_mixture_of_xray_and_neutron():
        errmsg = "Mixed xray and neutron scattering table! "
        errmsg += self.scattering_dictionary_as_string()
        raise RuntimeError(errmsg)
    return self._scattering_type_registry

  def set_inelastic_form_factors(self, photon, table, set_use_fp_fdp=True):
    if table == "sasaki":
      set_inelastic_ff = ext.set_inelastic_form_factors_from_sasaki
    elif table == "henke":
      set_inelastic_ff = ext.set_inelastic_form_factors_from_henke
    else:
      raise RuntimeError("Unknown inelastic form factors table: %s" % table)
    self.inelastic_form_factors_source = table
    set_inelastic_ff(self.scatterers(), photon, set_use_fp_fdp)

  def set_custom_inelastic_form_factors(self, table, set_use_fp_fdp=True,
                                        source="custom"):
    """ Expects a dictionary of tuples like 'C' : (fp, fdp). If an element is
    not in the dictionary, the fp and fdp are reset to 0 and the use_fp_fdp is
    set to false.
    """
    for sc in self.scatterers():
      fp_fdp = table.get(sc.scattering_type, None)
      if fp_fdp is None:
        sc.fp = 0
        sc.fdp = 0
        sc.use_fp_fdp = False
      else:
        sc.fp = fp_fdp[0]
        sc.fdp = fp_fdp[1]
        if set_use_fp_fdp:
          sc.flags.set_use_fp_fdp(True)
    self.inelastic_form_factors_source = source

  def mean_scattering_density(self):
    r = self.scattering_type_registry()
    return r.sum_of_scattering_factors_at_diffraction_angle_0() \
         / self.unit_cell().volume()

  def __getitem__(self, slice_object):
    assert type(slice_object) == slice
    assert self.scatterers() is not None
    sel = flex.slice_indices(
      array_size=self._scatterers.size(),
      python_slice=slice_object)
    return structure(
      special_position_settings=self,
      scatterers=self._scatterers.select(sel),
      site_symmetry_table=self._site_symmetry_table.select(sel),
      scattering_type_registry=self._scattering_type_registry,
      wavelength=self.wavelength)

  def select(self, selection, negate=False):
    assert self.scatterers() is not None
    if (negate): selection = ~selection
    strp = self.scattering_type_registry_params
    #table = self._scattering_table
    selected_xrs = structure(
      special_position_settings=self,
      scatterers=self._scatterers.select(selection),
      site_symmetry_table=self._site_symmetry_table.select(selection),
      scattering_type_registry=self._scattering_type_registry,
      wavelength=self.wavelength)
    selected_xrs.scattering_type_registry_params = strp
    return selected_xrs

  def select_inplace(self, selection):
    assert self.scatterers() is not None
    self._scatterers          = self._scatterers.select(selection)
    self._site_symmetry_table = self._site_symmetry_table.select(selection)
    self._scattering_type_registry_is_out_of_date = True

  def add_scatterer(self,
        scatterer,
        site_symmetry_ops=None,
        insert_at_index=None):
    if (insert_at_index is None):
      insert_at_index = self._scatterers.size()
    self._scatterers.insert(insert_at_index, scatterer)
    if (site_symmetry_ops is None):
      site_symmetry_ops = self._scatterers[insert_at_index].apply_symmetry(
        unit_cell=self.unit_cell(),
        space_group=self.space_group(),
        min_distance_sym_equiv=self.min_distance_sym_equiv(),
        u_star_tolerance=self.u_star_tolerance(),
        assert_min_distance_sym_equiv=self.assert_min_distance_sym_equiv())
    else:
      self._scatterers[insert_at_index].apply_symmetry(
        site_symmetry_ops=site_symmetry_ops,
        u_star_tolerance=self.u_star_tolerance())
    self._site_symmetry_table.process(
      insert_at_index=insert_at_index, site_symmetry_ops=site_symmetry_ops)
    self._scattering_type_registry_is_out_of_date = True

  def add_scatterers(self,
        scatterers,
        site_symmetry_table=None,
        non_unit_occupancy_implies_min_distance_sym_equiv_zero=False):
    if (site_symmetry_table is None):
      site_symmetry_table = sgtbx.site_symmetry_table()
    else:
      assert site_symmetry_table.indices().size() == scatterers.size()
      assert not non_unit_occupancy_implies_min_distance_sym_equiv_zero
    self._scatterers.extend(scatterers)
    ext.add_scatterers_ext(
      unit_cell=self.unit_cell(),
      space_group=self.space_group(),
      scatterers=self._scatterers,
      site_symmetry_table=self._site_symmetry_table,
      site_symmetry_table_for_new=site_symmetry_table,
      min_distance_sym_equiv=self.min_distance_sym_equiv(),
      u_star_tolerance=self.u_star_tolerance(),
      assert_min_distance_sym_equiv=self.assert_min_distance_sym_equiv(),
      non_unit_occupancy_implies_min_distance_sym_equiv_zero=
        non_unit_occupancy_implies_min_distance_sym_equiv_zero)
    self._scattering_type_registry_is_out_of_date = True

  def concatenate(self, other):
    result = self.deep_copy_scatterers()
    result.add_scatterers(
      scatterers=other._scatterers,
      site_symmetry_table=other._site_symmetry_table)
    return result

  def concatenate_inplace(self, other):
    d1 = self.scattering_type_registry().as_type_gaussian_dict()
    d2 = other.scattering_type_registry().as_type_gaussian_dict()
    for key1, item1 in six.moves.zip(d1.keys(), six.iteritems(d1)):
      for key2, item2 in six.moves.zip(d2.keys(), six.iteritems(d2)):
        if(key1 == key2):
          i1 = item1[1]
          i2 = item2[1]
          problem_flag = False
          for a1, a2 in zip(i1.array_of_a(), i2.array_of_a()):
            if(not approx_equal(a1, a2)): problem_flag = True
          for b1, b2 in zip(i1.array_of_b(), i2.array_of_b()):
            if(not approx_equal(b1, b2)): problem_flag = True
          if(not approx_equal(i1.c(), i2.c())): problem_flag = True
          if(problem_flag):
            raise RuntimeError("Cannot concatenate: conflicting scatterers")
    self.add_scatterers(scatterers          = other._scatterers,
                        site_symmetry_table = other._site_symmetry_table)
    strp1 = self.scattering_type_registry_params
    strp2 = other.scattering_type_registry_params
    self.scattering_type_registry(
      custom_dict = strp1.custom_dict,
      d_min       = strp1.d_min,
      table       = strp1.table,
      types_without_a_scattering_contribution =
                                strp1.types_without_a_scattering_contribution)
    self.scattering_type_registry(
      custom_dict = strp2.custom_dict,
      d_min       = strp2.d_min,
      table       = strp2.table,
      types_without_a_scattering_contribution =
                                strp2.types_without_a_scattering_contribution)

  def selection_within(self, radius, selection):
    assert radius > 0
    assert self.special_position_settings() is not None
    return crystal.neighbors_fast_pair_generator(
      asu_mappings=self.special_position_settings().asu_mappings(
        buffer_thickness=radius,
        sites_cart=self.sites_cart()),
      distance_cutoff=radius).neighbors_of(
        primary_selection=selection)

  def replace_scatterers(self, scatterers, site_symmetry_table="existing"):
    if (site_symmetry_table == "existing"):
      site_symmetry_table = self._site_symmetry_table
    if (site_symmetry_table is not None):
      assert site_symmetry_table.indices().size() == self.scatterers().size()
    self.erase_scatterers()
    self.add_scatterers(
      scatterers=scatterers,
      site_symmetry_table=site_symmetry_table)

  def structure_factors(self, anomalous_flag=None, d_min=None,
                              algorithm=None,
                              cos_sin_table=False,
                              quality_factor=None,
                              u_base=None,
                              b_base=None,
                              wing_cutoff=None):
    """
    Calculate structure factors for the current scatterers using either direct
    summation or FFT method; by default the appropriate method will be guessed
    automatically.

    :param anomalous_flag: toggles whether the returned structure factors are
      anomalous or not
    :type anomalous_flag: bool
    :param d_min: resolution cutoff (required)
    :type d_min: float
    :param algorithm: specifies 'direct' or 'fft', or if None (default), the
      algorithm will be chosen automatically
    :type algorithm: str or None
    :param cos_sin_table: if True, uses interpolation of values from a
      pre-calculated lookup table for trigonometric function calls in the
      structure factor calculations in preference to calling the system
      libraries
    :type cos_sin_table: bool
    :param quality_factor: determines accuracy of sampled density in fft method
    :type quality_factor: float
    :param u_base: additional smearing B-factor for scatterers which have too
      small Bs so they fall between the grid nodes
    :type u_base: float
    :param b_base: same as u_base (but 8*pi^2 larger)
    :type b_base: float
    :param wing_cutoff: is how far away from atomic center you sample density
      around atom
    :type wing_cutoff: float

    :returns: a custom Python object (exact type depends on method used), from
      which f_calc() may be called
    :rtype: derived from cctbx.xray.structure_factors.manager.managed_calculation_base
    """
    if (anomalous_flag is None):
      if (self.scatterers().count_anomalous() != 0):
        anomalous_flag = True
      else:
        anomalous_flag = False
    elif (not anomalous_flag):
      if (self.scatterers().count_anomalous() != 0):
        raise RuntimeError(
            "xray.structure with anomalous scatterers"
          + " but miller.array is non-anomalous.")
    miller_set = miller.build_set(self, anomalous_flag, d_min)
    return structure_factors.from_scatterers(
      crystal_symmetry=self,
      d_min=d_min,
      cos_sin_table=cos_sin_table,
      quality_factor=quality_factor,
      u_base=u_base,
      b_base=b_base,
      wing_cutoff=wing_cutoff)(
        xray_structure=self,
        miller_set=miller_set,
        algorithm=algorithm)

  def as_py_code(self, indent=""):
    """eval(self.as_py_code()) is similar (but sometimes not identical)
    to self and meant to enable quick formatting of self as Python code
    for unit tests.
    """
    r0 = (
      "%sxray.structure(\n"
      "%s  crystal_symmetry=%s,\n"
      "%s  scatterers=flex.xray_scatterer([\n"
        % (indent,
           indent, self.crystal_symmetry().as_py_code(indent=indent+"  "),
           indent))
    r1 = []
    for i,sc in enumerate(self.scatterers()):
      r1.append(
          indent+"    "
        + sc.as_py_code(indent=indent+"    ", comment=" #%d" % i))
    return r0 + ",\n".join(r1) + "]))"

  def show_summary(self, f=None, prefix=""):
    if (f is None): f = sys.stdout
    print(prefix + "Number of scatterers:", \
      self.scatterers().size(), file=f)
    print(prefix + "At special positions:", \
      self.special_position_indices().size(), file=f)
    crystal.symmetry.show_summary(self, f=f, prefix=prefix)
    return self

  def show_scatterers(self, f=None, special_positions_only=False):
    if (f is None): f = sys.stdout
    print(("Label, Scattering, Multiplicity, Coordinates, Occupancy, "
                 "Uiso, Ustar as Uiso"), file=f)
    scatterers = self.scatterers()
    if (special_positions_only):
      scatterers = scatterers.select(self.special_position_indices())
    for sc in scatterers:
      sc.show(f=f, unit_cell=self.unit_cell())
    return self

  def show_special_position_shifts(self,
        sites_frac_original=None,
        sites_cart_original=None,
        out=None,
        prefix=""):
    self._site_symmetry_table.show_special_position_shifts(
      special_position_settings=self,
      site_labels=self.scatterers().extract_labels(),
      sites_frac_original=sites_frac_original,
      sites_cart_original=sites_cart_original,
      sites_frac_exact=self.scatterers().extract_sites(),
      out=out,
      prefix=prefix)

  def is_similar(self, other, eps = 1.e-6):
    """
    Check if similar. Can be endlessly fortified as needed.
    """
    if(self.scatterers().size() != other.scatterers().size()): return False
    f1 = self.crystal_symmetry().is_similar_symmetry(other.crystal_symmetry())
    f2 = approx_equal(self.sites_frac(), other.sites_frac(), eps)
    f3 = approx_equal(self.extract_u_iso_or_u_equiv(),
                      other.extract_u_iso_or_u_equiv(), eps)
    f4 = approx_equal(self.scatterers().extract_occupancies(),
                      other.scatterers().extract_occupancies(), eps)
    f5 = approx_equal(self.scatterers().extract_scattering_types(),
                      other.scatterers().extract_scattering_types())
    sr1 = self.scattering_type_registry().unique_gaussians_as_list()
    sr2 = other.scattering_type_registry().unique_gaussians_as_list()
    f6 = True
    for s1, s2 in zip(sr1, sr2):
      f6 &= approx_equal(s1.parameters(), s2.parameters(), eps)
    f = list(set([f1,f2,f3,f4,f5,f6]))
    return len(f)==1 and f[0]

  def is_positive_definite_u(self, u_cart_tolerance=None):
    if (u_cart_tolerance is None):
      return ext.is_positive_definite_u(
        scatterers=self._scatterers,
        unit_cell=self.unit_cell())
    else:
      return ext.is_positive_definite_u(
        scatterers=self._scatterers,
        unit_cell=self.unit_cell(),
        u_cart_tolerance=u_cart_tolerance)

  def tidy_us(self, u_min = 1.e-6, u_max = adptbx.b_as_u(999.99),
                    anisotropy_min=0.25):
    """
    Clean up atomic displacements so they fall within a sensible range (this
    is especially important when writing out PDB format).
    """
    assert u_min < u_max
    ext.tidy_us(
      scatterers=self._scatterers,
      unit_cell=self.unit_cell(),
      site_symmetry_table=self._site_symmetry_table,
      u_min=u_min,
      u_max=u_max,
      anisotropy_min=anisotropy_min)

  def shift_us(self, u_shift=None, b_shift=None, selection=None):
    assert [u_shift, b_shift].count(None) == 1
    if (u_shift is None):
      u_shift = adptbx.b_as_u(b_shift)
    if(selection is None):
      ext.shift_us(
        scatterers=self._scatterers,
        unit_cell=self.unit_cell(),
        u_shift=u_shift)
    else:
      ext.shift_us(scatterers = self._scatterers,
                   unit_cell  = self.unit_cell(),
                   u_shift    = u_shift,
                   selection  = selection)

  def shift_occupancies(self, q_shift, selection=None):
    if(selection is not None):
      ext.shift_occupancies(scatterers = self._scatterers,
                            q_shift    = q_shift,
                            selection  = selection)
    else:
      ext.shift_occupancies(scatterers = self._scatterers,
                            q_shift    = q_shift)

  def apply_symmetry_sites(self):
    ext.apply_symmetry_sites(
      site_symmetry_table=self._site_symmetry_table,
      scatterers=self._scatterers)

  def apply_symmetry_u_stars(self):
    ext.apply_symmetry_u_stars(
      site_symmetry_table=self._site_symmetry_table,
      scatterers=self._scatterers,
      u_star_tolerance=self.u_star_tolerance())

  def re_apply_symmetry(self, i_scatterer):
    self._scatterers[i_scatterer].apply_symmetry(
      site_symmetry_ops=self._site_symmetry_table.get(i_scatterer),
      u_star_tolerance=self.u_star_tolerance())

  def apply_special_position_ops_d_target_d_site(self, d_target_d_site):
    for i in self.special_position_indices():
      special_op = self._site_symmetry_table.get(i).special_op()
      r = matrix.sqr(special_op.r().as_double())
      d_target_d_site[i] = (matrix.row(d_target_d_site[i]) * r).elems

  def asymmetric_unit_in_p1(self):
    new_structure = structure(
      crystal.special_position_settings(
        crystal.symmetry.cell_equivalent_p1(self)),
      scattering_type_registry=self._scattering_type_registry,
      wavelength=self.wavelength)
    new_structure._scatterers = self.scatterers().deep_copy()
    new_structure._site_symmetry_table = self.site_symmetry_table().deep_copy()
    return new_structure

  def change_basis(self, cb_op):
    if (isinstance(cb_op, str)):
      cb_op = sgtbx.change_of_basis_op(cb_op)
    return structure(
      special_position_settings
        =crystal.special_position_settings.change_basis(self, cb_op),
      scatterers=ext.change_basis(scatterers=self._scatterers, cb_op=cb_op),
      site_symmetry_table=self._site_symmetry_table.change_basis(cb_op=cb_op),
      scattering_type_registry=self._scattering_type_registry,
      wavelength=self.wavelength)

  def change_hand(self):
    ch_op = self.space_group_info().type().change_of_hand_op()
    return self.change_basis(ch_op)

  def expand_to_p1(self,
      append_number_to_labels=False,
      sites_mod_positive=False):
    """Get the current structure expanded into spacegroup P1.
    This turns all symmetry induced scatterers into independent
    individual scatterers. The expanded structure may have sites
    with negative or > 1.0 coordinates. Use '.sites_mod_positive()'
    or '.sites_mod_short()' on the result, or alternatively
    set sites_mod_positive to 'True' in case you want to change this behaviour.

    :param append_number_to_labels: If set to 'True' scatterers generated from \
    symmetry will be labelled with a numerical suffix
    :type append_number_to_labels: boolean
    :param sites_mod_positive: If set to 'True' xyz coordinates of the \
    scatterers will be kept inside [0,1[
    :type sites_mod_positive: boolean

    :returns: a new instance of the structure expanded into P1
    :rtype: cctbx.xray.structure
    """
    result = structure(
      special_position_settings
        =crystal.special_position_settings(
          crystal.symmetry.cell_equivalent_p1(self)),
      scatterers=ext.expand_to_p1(
        unit_cell=self.unit_cell(),
        space_group=self.space_group(),
        scatterers=self._scatterers,
        site_symmetry_table=self._site_symmetry_table,
        append_number_to_labels=append_number_to_labels),
      scattering_type_registry=self._scattering_type_registry,
      wavelength=self.wavelength)
    if (sites_mod_positive):
      result = result.sites_mod_positive()
    return result

  def sites_mod_positive(self):
    """Get the current structure converted into a structure with x,y,z of all
    scatterers in the interval [0,1[

    :returns: the same instance of the structure with only posive coordinates \
    of its scatterers
    :rtype: cctbx.xray.structure
    """
    return structure(
      special_position_settings=self,
      scatterers=self.scatterers().sites_mod_positive(),
      scattering_type_registry=self._scattering_type_registry,
      wavelength=self.wavelength)

  def sites_mod_short(self):
    """Get the current structure converted into a structure with short
    coordinates vectors of all scatterers

    :returns: the same instance of the structure with only short coordinates \
    vectors of its scatterers
    :rtype: cctbx.xray.structure
    """
    return structure(
      special_position_settings=self,
      scatterers=self.scatterers().sites_mod_short(),
      scattering_type_registry=self._scattering_type_registry,
      wavelength=self.wavelength)

  def apply_shift(self, shift, recompute_site_symmetries=False):
    shifted_scatterers = self.scatterers().deep_copy()
    shifted_scatterers.set_sites(shifted_scatterers.extract_sites() + shift)
    if (recompute_site_symmetries):
      site_symmetry_table = None
    else:
      site_symmetry_table = self._site_symmetry_table
    return structure(
      special_position_settings=self,
      scatterers=shifted_scatterers,
      site_symmetry_table=site_symmetry_table,
      scattering_type_registry=self._scattering_type_registry,
      wavelength=self.wavelength)

  def random_shift_sites(self, max_shift_cart=0.2):
    shifts = flex.vec3_double(
      (flex.random_double(self.scatterers().size()*3)*2-1) * max_shift_cart)
    return self.apply_shift(self.unit_cell().fractionalize(sites_cart=shifts))

  def sort(self, by_value="occupancy", reverse=False):
    assert by_value in ("occupancy",)
    assert reverse in (False, True)
    p = flex.sort_permutation(
      data=self.scatterers().extract_occupancies(),
      reverse=reverse)
    return structure(
      special_position_settings=self,
      scatterers=self._scatterers.select(p),
      site_symmetry_table=self._site_symmetry_table.select(p),
      scattering_type_registry=self._scattering_type_registry,
      wavelength=self.wavelength)

  def as_emma_model(self):
    from cctbx import euclidean_model_matching as emma
    positions = []
    for scatterer in self.scatterers():
      positions.append(emma.position(scatterer.label, scatterer.site))
    return emma.model(self, positions)

  def atomic_weights(self):
    from cctbx.eltbx import tiny_pse
    result = flex.double()
    for scatterer in self.scatterers():
      std_lbl = eltbx.xray_scattering.get_standard_label(
        label=scatterer.scattering_type, exact=True, optional=True)
      if (std_lbl is None):
        raise RuntimeError(
          "Unknown atomic weight: " + scatterer.scattering_type)
      result.append(tiny_pse.table(std_lbl).weight())
    return result

  def center_of_mass(self, atomic_weights=None):
    if (atomic_weights is None):
      atomic_weights = self.atomic_weights()
    return self.sites_cart().mean_weighted(weights=atomic_weights)

  def principal_axes_of_inertia(self, atomic_weights=None):
    if (atomic_weights is None):
      atomic_weights = self.atomic_weights()
    return scitbx.math.principal_axes_of_inertia(
      points=self.sites_cart(),
      weights=atomic_weights)

  def set_u_cart(self, u_cart, selection = None):
    assert self._scatterers.size() == u_cart.size()
    if(selection is not None):
      self._scatterers.set_u_cart(unit_cell = self.unit_cell(),
                                  u_cart    = u_cart,
                                  selection = selection)
    else:
      self._scatterers.set_u_cart(unit_cell = self.unit_cell(),
                                  u_cart    = u_cart)

  def show_scatterer_flags_summary(self, out=None):
    # XXX move to C++
    if (out is None): out = sys.stdout
    n_use = 0
    n_use_u_both = 0
    n_use_u_iso = 0
    n_use_u_aniso = 0
    n_use_u_none = 0
    n_grad_site = 0
    n_grad_u_iso = 0
    n_grad_u_aniso = 0
    n_grad_occupancy = 0
    n_grad_fp = 0
    n_grad_fdp = 0
    for scatterer in self.scatterers():
      flags = scatterer.flags
      if (flags.use()): n_use += 1
      i, a = flags.use_u_iso(), flags.use_u_aniso()
      if (i and a): n_use_u_both += 1
      elif (i):     n_use_u_iso += 1
      elif (a):     n_use_u_aniso += 1
      else:         n_use_u_none += 1
      if (flags.grad_site()):      n_grad_site += 1
      if (flags.grad_u_iso()):     n_grad_u_iso += 1
      if (flags.grad_u_aniso()):   n_grad_u_aniso += 1
      if (flags.grad_occupancy()): n_grad_occupancy += 1
      if (flags.grad_fp()):        n_grad_fp += 1
      if (flags.grad_fdp()):       n_grad_fdp += 1
    print("n_use            = ", n_use, file=out)
    if (n_use_u_none != 0):
      print("n_use_u_none     = ", n_use_u_none, file=out)
    if (n_use_u_both != 0):
      print("n_use_u_both     = ", n_use_u_both, file=out)
    print("n_use_u_iso      = ", n_use_u_iso, file=out)
    print("n_use_u_aniso    = ", n_use_u_aniso, file=out)
    print("n_grad_site      = ", n_grad_site, file=out)
    print("n_grad_u_iso     = ", n_grad_u_iso, file=out)
    print("n_grad_u_aniso   = ", n_grad_u_aniso, file=out)
    print("n_grad_occupancy = ", n_grad_occupancy, file=out)
    print("n_grad_fp        = ", n_grad_fp, file=out)
    print("n_grad_fdp       = ", n_grad_fdp, file=out)
    print("total number of scatterers = ", self.scatterers().size(), file=out)

  def scatterer_flags(self):
    return ext.shared_scatterer_flags(self.scatterers())

  def set_scatterer_flags(self, scatterer_flags):
    scatterer_flags.assign_to(self.scatterers())

  def n_parameters(self, considering_site_symmetry_constraints=False):
    # XXX move to C++
    result = 0
    if (considering_site_symmetry_constraints):
      sstab = self.site_symmetry_table()
    else:
      sstab = None
    for i_sc,sc in enumerate(self.scatterers()):
      flags = sc.flags
      if (sstab is None):
        site_symmetry = None
      else:
        site_symmetry = sstab.get(i_sc)
        if (site_symmetry.is_point_group_1()):
          site_symmetry = None
      if (flags.grad_site()):
        if (site_symmetry is None):
          result += 3
        else:
          result += site_symmetry.site_constraints().n_independent_params()
      if (    flags.grad_u_iso()
          and flags.use_u_iso()): result += 1
      if (    flags.grad_u_aniso()
          and flags.use_u_aniso()):
        if (site_symmetry is None):
          result += 6
        else:
          result += site_symmetry.adp_constraints().n_independent_params()
      if (flags.grad_occupancy()): result += 1
      if (flags.grad_fp()): result += 1
      if (flags.grad_fdp()): result += 1
    return result

  def n_grad_u_iso(self):
    return self.scatterers().n_grad_u_iso()

  def n_grad_u_aniso(self):
    return self.scatterers().n_grad_u_aniso()

  def parameter_map(self):
    return cctbx.xray.parameter_map(self.scatterers())

  def truncate_at_pdb_format_precision(self):
    uc = self.unit_cell()
    fra = uc.fractionalize
    ort = uc.orthogonalize
    for sc in self.scatterers():
      sc.site = fra([float("%8.3f"%i) for i in ort(sc.site)])
      if(sc.u_iso != -1.0):
        sc.u_iso = adptbx.b_as_u(float("%6.2f"%adptbx.u_as_b(sc.u_iso)))
      sc.occupancy = float("%5.2f"%sc.occupancy)
      if(sc.u_star != (-1,-1,-1,-1,-1,-1)):
        u_pdb = [int(i*10000) for i in adptbx.u_star_as_u_cart(uc, sc.u_star)]
        sc.u_star = adptbx.u_cart_as_u_star(uc, [float(i)/10000 for i in u_pdb])

  def grads_and_curvs_target_simple(self, miller_indices, da_db, daa_dbb_dab):
    return ext.structure_factors_curvatures_simple_grads_and_curvs_target(
      unit_cell=self.unit_cell(),
      space_group=self.space_group(),
      scatterers=self.scatterers(),
      scattering_type_registry=self.scattering_type_registry(),
      site_symmetry_table=self.site_symmetry_table(),
      miller_indices=miller_indices,
      da_db=da_db,
      daa_dbb_dab=daa_dbb_dab)

  def pair_sym_table_show(self,
        pair_sym_table,
        is_full_connectivity=False,
        out=None):
    if (is_full_connectivity):
      site_symmetry_table = None
    else:
      site_symmetry_table = self.site_symmetry_table()
    return pair_sym_table.show(
      f=out,
      site_labels=self.scatterers().extract_labels(),
      site_symmetry_table=site_symmetry_table,
      sites_frac=self.sites_frac(),
      unit_cell=self.unit_cell())

  def pair_sym_table_show_distances(self,
        pair_sym_table,
        show_cartesian=False,
        skip_j_seq_less_than_i_seq=False,
        skip_sym_equiv=False,
        out=None):
    return pair_sym_table.show_distances(
      unit_cell=self.unit_cell(),
      site_symmetry_table=self.site_symmetry_table(),
      site_labels=self.scatterers().extract_labels(),
      sites_frac=self.sites_frac(),
      show_cartesian=show_cartesian,
      skip_j_seq_less_than_i_seq=skip_j_seq_less_than_i_seq,
      skip_sym_equiv=skip_sym_equiv,
      out=out)

  def asu_mappings(self, buffer_thickness, asu_is_inside_epsilon=None):
    result = crystal.symmetry.asu_mappings(self,
      buffer_thickness=buffer_thickness,
      asu_is_inside_epsilon=asu_is_inside_epsilon)
    ext.asu_mappings_process(
      asu_mappings=result,
      scatterers=self._scatterers,
      site_symmetry_table=self._site_symmetry_table)
    return result

  def pair_asu_table(self,
        distance_cutoff=None,
        asu_mappings_buffer_thickness=None,
        asu_is_inside_epsilon=None,
        min_cubicle_edge=5):
    assert (distance_cutoff is not None
            or asu_mappings_buffer_thickness is not None)
    if (asu_mappings_buffer_thickness is None):
      asu_mappings_buffer_thickness = distance_cutoff
    asu_mappings = self.asu_mappings(
      buffer_thickness=asu_mappings_buffer_thickness,
      asu_is_inside_epsilon=asu_is_inside_epsilon)
    pair_asu_table = crystal.pair_asu_table(asu_mappings=asu_mappings)
    if (distance_cutoff is not None):
      pair_asu_table.add_all_pairs(
        distance_cutoff=distance_cutoff,
        min_cubicle_edge=min_cubicle_edge)
    return pair_asu_table

  def show_distances(self,
        distance_cutoff=None,
        asu_mappings_buffer_thickness=None,
        asu_is_inside_epsilon=None,
        min_cubicle_edge=5,
        pair_asu_table=None,
        show_cartesian=False,
        keep_pair_asu_table=False,
        out=None):
    assert [distance_cutoff, pair_asu_table].count(None) == 1
    if (pair_asu_table is None):
      pair_asu_table = self.pair_asu_table(
        distance_cutoff=distance_cutoff,
        asu_mappings_buffer_thickness=asu_mappings_buffer_thickness,
        asu_is_inside_epsilon=asu_is_inside_epsilon,
        min_cubicle_edge=min_cubicle_edge)
    return pair_asu_table.show_distances(
      site_labels=self.scatterers().extract_labels(),
      sites_frac=self.sites_frac(),
      show_cartesian=show_cartesian,
      keep_pair_asu_table=keep_pair_asu_table,
      out=out)

  def show_angles(self,
        distance_cutoff=None,
        asu_mappings_buffer_thickness=None,
        asu_is_inside_epsilon=None,
        pair_asu_table=None,
        keep_pair_asu_table=False,
        out=None):
    assert [distance_cutoff, pair_asu_table].count(None) == 1
    if (pair_asu_table is None):
      pair_asu_table = self.pair_asu_table(
        distance_cutoff=distance_cutoff,
        asu_mappings_buffer_thickness=asu_mappings_buffer_thickness,
        asu_is_inside_epsilon=asu_is_inside_epsilon)
    return pair_asu_table.show_angles(
      site_labels=self.scatterers().extract_labels(),
      sites_frac=self.sites_frac(),
      keep_pair_asu_table=keep_pair_asu_table,
      out=out)

  def show_dihedral_angles(self,
        distance_cutoff=None,
        max_d=1.7,
        max_angle=170,
        asu_mappings_buffer_thickness=None,
        asu_is_inside_epsilon=None,
        pair_asu_table=None,
        keep_pair_asu_table=False,
        out=None):
    if (pair_asu_table is None):
      pair_asu_table = self.pair_asu_table(
        distance_cutoff=distance_cutoff,
        asu_mappings_buffer_thickness=asu_mappings_buffer_thickness,
        asu_is_inside_epsilon=asu_is_inside_epsilon)
    return pair_asu_table.show_dihedral_angles(
      site_labels=self.scatterers().extract_labels(),
      sites_frac=self.sites_frac(),
      max_d=max_d,
      max_angle=max_angle,
      out=out)

  def conservative_pair_proxies(self, bond_sym_table, conserve_angles):
    return conservative_pair_proxies(
      structure=self,
      bond_sym_table=bond_sym_table,
      conserve_angles=conserve_angles)

  def difference_vectors_cart(self, other):
    return other.sites_cart() - self.sites_cart()

  def rms_difference(self, other):
    return self.sites_cart().rms_difference(other.sites_cart())

  def closest_distances(self, sites_frac, distance_cutoff, use_selection=None):
    class map_next_to_model_and_find_closest_distances(object):
      def __init__(self, xray_structure, sites_frac, use_selection):
        asu_mappings = xray_structure.asu_mappings(buffer_thickness =
          distance_cutoff)
        asu_mappings.process_sites_frac(sites_frac, min_distance_sym_equiv =
          xray_structure.min_distance_sym_equiv())
        pair_generator = crystal.neighbors_fast_pair_generator(asu_mappings =
          asu_mappings, distance_cutoff = distance_cutoff)
        n_xray = xray_structure.scatterers().size()
        new_sites_frac = sites_frac.deep_copy()
        smallest_distances_sq = flex.double(sites_frac.size(),
          distance_cutoff**2+1)
        i_seqs = flex.int(sites_frac.size(), -1)
        for pair in pair_generator:
          if(pair.i_seq < n_xray):
            if (pair.j_seq < n_xray): continue
            # i_seq = molecule
            # j_seq = site
            rt_mx_i = asu_mappings.get_rt_mx_i(pair)
            rt_mx_j = asu_mappings.get_rt_mx_j(pair)
            rt_mx_ji = rt_mx_i.inverse().multiply(rt_mx_j)
            i_seq_new_site_frac = pair.j_seq - n_xray
            new_site_frac = rt_mx_ji * sites_frac[i_seq_new_site_frac]
            jn = pair.i_seq
          else:
            if(pair.j_seq >= n_xray): continue
            # i_seq = site
            # j_seq = molecule
            rt_mx_i = asu_mappings.get_rt_mx_i(pair)
            rt_mx_j = asu_mappings.get_rt_mx_j(pair)
            rt_mx_ij = rt_mx_j.inverse().multiply(rt_mx_i)
            i_seq_new_site_frac = pair.i_seq - n_xray
            new_site_frac = rt_mx_ij * sites_frac[i_seq_new_site_frac]
            jn = pair.j_seq
          if(use_selection[jn]):
            if(smallest_distances_sq[i_seq_new_site_frac] >= pair.dist_sq):
              smallest_distances_sq[i_seq_new_site_frac] = pair.dist_sq
              new_sites_frac[i_seq_new_site_frac] = new_site_frac
              i_seqs[i_seq_new_site_frac] = jn
        self.remove_selection = smallest_distances_sq > distance_cutoff**2
        self.sites_frac = new_sites_frac
        self.smallest_distances = flex.sqrt(
          smallest_distances_sq).set_selected(self.remove_selection, -1)
        self.smallest_distances_sq = smallest_distances_sq.set_selected(
          self.remove_selection, -1)
        self.i_seqs = i_seqs
    if(use_selection is not None):
      assert use_selection.size() == self._scatterers.size()
    else:
      use_selection = flex.bool(self._scatterers.size(), True)
    result = map_next_to_model_and_find_closest_distances(
      xray_structure = self, sites_frac = sites_frac, use_selection =
      use_selection)
    return result

  def orthorhombic_unit_cell_around_centered_scatterers(self, buffer_size):
    sites_cart = self.sites_cart()
    sites_cart_min = sites_cart.min()
    abc = [2*buffer_size+a-i for i,a in zip(sites_cart_min,sites_cart.max())]
    sites_cart += [buffer_size-i for i in sites_cart_min]
    result = structure(
      crystal_symmetry=crystal.symmetry(
        unit_cell=abc,
        space_group_symbol="P1"),
      scatterers=self.scatterers(),
      wavelength=self.wavelength)
    result.set_sites_cart(sites_cart)
    return result

  def cubic_unit_cell_around_centered_scatterers(self, buffer_size):
    sites_cart = self.sites_cart()
    sites_cart_min = sites_cart.min()
    span = [a-i for i,a in zip(sites_cart_min, sites_cart.max())]
    a = max(span) + 2 * buffer_size
    sites_cart += [(a-s)/2-i for i,s in zip(sites_cart_min, span)]
    result = structure(
      crystal_symmetry=crystal.symmetry(
        unit_cell=[a,a,a],
        space_group_symbol="P1"),
      scatterers=self.scatterers(),
      wavelength=self.wavelength)
    result.set_sites_cart(sites_cart)
    return result

  def as_cif_simple(self, out=None, data_name="global", format="mmCIF"):
    if out is None: out = sys.stdout
    import iotbx.cif
    cif = iotbx.cif.model.cif()
    cif[data_name] = self.as_cif_block(format=format)
    print(cif, file=out)

  def as_cif_block(self, covariance_matrix=None,
    cell_covariance_matrix=None, format="mmCIF"):

    from iotbx.cif import atom_type_cif_loop, model
    cs_cif_block = self.crystal_symmetry().as_cif_block(
        format=format,
        cell_covariance_matrix=cell_covariance_matrix)
    # crystal_symmetry_as_cif_block.__init__(
    #   self, xray_structure.crystal_symmetry(),
    #   cell_covariance_matrix=cell_covariance_matrix)
    scatterers = self.scatterers()
    uc = self.unit_cell()
    if covariance_matrix is not None:
      param_map = self.parameter_map()
      covariance_diagonal = covariance_matrix.matrix_packed_u_diagonal()
      u_star_to_u_cif_linear_map_pow2 = flex.pow2(flex.double(
        uc.u_star_to_u_cif_linear_map()))
      u_star_to_u_iso_linear_form = matrix.row(
        uc.u_star_to_u_iso_linear_form())
    fmt = "%.6f"

    # _atom_site_* loop
    atom_site_loop = model.loop(header=(
      '_atom_site_label', '_atom_site_type_symbol',
      '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z',
      '_atom_site_U_iso_or_equiv', '_atom_site_adp_type',
      '_atom_site_occupancy'))
    for i_seq, sc in enumerate(scatterers):
      site = occu = u_iso_or_equiv = None
      # site
      if covariance_matrix is not None:
        params = param_map[i_seq]
        if sc.flags.grad_site() and params.site >= 0:
          site = []
          for i in range(3):
            site.append(format_float_with_su(sc.site[i],
              math.sqrt(covariance_diagonal[params.site+i])))
        #occupancy
        if sc.flags.grad_occupancy() and params.occupancy >= 0:
          occu = format_float_with_su(sc.occupancy,
            math.sqrt(covariance_diagonal[params.occupancy]))
        #Uiso/eq
        if sc.flags.grad_u_iso() or sc.flags.grad_u_aniso():
          if sc.flags.grad_u_iso():
            u_iso_or_equiv = format_float_with_su(
              sc.u_iso, math.sqrt(covariance.variance_for_u_iso(
                i_seq, covariance_matrix, param_map)))
          else:
            cov = covariance.extract_covariance_matrix_for_u_aniso(
              i_seq, covariance_matrix, param_map).matrix_packed_u_as_symmetric()
            var = (u_star_to_u_iso_linear_form * matrix.sqr(cov)
                   ).dot(u_star_to_u_iso_linear_form)
            u_iso_or_equiv = format_float_with_su(
              sc.u_iso_or_equiv(uc), math.sqrt(var))

      if site is None:
        site = [fmt % sc.site[i] for i in range(3)]
      if occu is None:
        occu = fmt % sc.occupancy
      if u_iso_or_equiv is None:
        u_iso_or_equiv = fmt % sc.u_iso_or_equiv(uc)

      if sc.flags.use_u_aniso():
        adp_type = 'Uani'
      else:
        adp_type = 'Uiso'
      atom_site_loop.add_row((
        sc.label, sc.scattering_type, site[0], site[1], site[2], u_iso_or_equiv,
        adp_type, occu))
    cs_cif_block.add_loop(atom_site_loop)

    # _atom_site_aniso_* loop
    aniso_scatterers = scatterers.select(scatterers.extract_use_u_aniso())
    if aniso_scatterers.size():
      labels = list(scatterers.extract_labels())
      aniso_loop = model.loop(header=('_atom_site_aniso_label',
                                      '_atom_site_aniso_U_11',
                                      '_atom_site_aniso_U_22',
                                      '_atom_site_aniso_U_33',
                                      '_atom_site_aniso_U_12',
                                      '_atom_site_aniso_U_13',
                                      '_atom_site_aniso_U_23'))
      anharmonic_scatterers = []
      for sc in aniso_scatterers:
        if sc.anharmonic_adp:
          anharmonic_scatterers.append(sc)
        u_cif = adptbx.u_star_as_u_cif(uc, sc.u_star)
        if covariance_matrix is not None:
          row = [sc.label]
          idx = param_map[labels.index(sc.label)].u_aniso
          if idx > -1:
            var = covariance_diagonal[idx:idx+6] * u_star_to_u_cif_linear_map_pow2
            for i in range(6):
              if var[i] > 0:
                row.append(
                  format_float_with_su(u_cif[i], math.sqrt(var[i])))
              else:
                row.append(fmt%u_cif[i])
          else:
            row = [sc.label] + [fmt%u_cif[i] for i in range(6)]
        else:
          row = [sc.label] + [fmt%u_cif[i] for i in range(6)]
        aniso_loop.add_row(row)
      cs_cif_block.add_loop(aniso_loop)
      if anharmonic_scatterers:
        from cctbx.sgtbx import rank_3_tensor_constraints as t3c
        from cctbx.sgtbx import rank_4_tensor_constraints as t4c
        C_header= ['_atom_site_anharm_GC_C_label']
        D_header= ['_atom_site_anharm_GC_D_label']
        for idx in flex.rows(t3c.indices()):
          C_header.append("_atom_site_anharm_GC_C_%s%s%s" %(idx[0]+1,idx[1]+1,idx[2]+1))
        for idx in flex.rows(t4c.indices()):
          D_header.append("_atom_site_anharm_GC_D_%s%s%s%s" %(idx[0]+1,idx[1]+1,idx[2]+1,idx[3]+1))
        C_loop = model.loop(header=(C_header))
        D_loop = model.loop(header=(D_header))
        for sc in anharmonic_scatterers:
          C_row = [sc.label]
          D_row = [sc.label]
          for i, d in enumerate(sc.anharmonic_adp.data()):
            if covariance_matrix is not None:
              idx = param_map[labels.index(sc.label)].u_aniso
              if idx > -1:
                var = covariance_diagonal[idx+6:idx+6+25]
                d = format_float_with_su(d, math.sqrt(var[i]))
            if i < 10:
              C_row.append(d)
            else:
              D_row.append(d)
          C_loop.add_row(C_row)
          D_loop.add_row(D_row)
        cs_cif_block.add_loop(C_loop)
        cs_cif_block.add_loop(D_loop)
      cs_cif_block.add_loop(atom_type_cif_loop(self, format=format))
    return cs_cif_block

  def as_pdb_file(self,
        remark=None,
        remarks=[],
        fractional_coordinates=False,
        resname=None,
        connect=None):
    import iotbx.pdb.xray_structure
    return iotbx.pdb.xray_structure.as_pdb_file(
      self=self,
      remark=remark,
      remarks=remarks,
      fractional_coordinates=fractional_coordinates,
      resname=resname,
      connect=connect)

  @classmethod
  def from_shelx(cls, *args, **kwds):
    import iotbx.shelx
    return iotbx.shelx._cctbx_xray_structure_from(*args, **kwds)

  @classmethod
  def from_cif(cls, file_object=None, file_path=None, data_block_name=None):
    import iotbx.cif
    from iotbx.cif import builders
    result = iotbx.cif.cctbx_data_structures_from_cif(
      file_object=file_object, file_path=file_path,
      data_block_name=data_block_name,
      data_structure_builder=builders.crystal_structure_builder)
    if len(result.xray_structures):
      if data_block_name is not None:
        return result.xray_structures[data_block_name]
      else:
        return result.xray_structures
    else:
      raise Sorry("Could not extract an xray.structure from the given input")

  def unit_cell_content(self, omit=None):
    """ The content of the unit cell as a chemical formula """
    return dict([ (r.scattering_type, r.unit_cell_occupancy_sum)
                  for r in self.scattering_types_counts_and_occupancy_sums()
                  if not omit or r.scattering_type not in omit ])

  def make_scatterer_labels_shelx_compatible_in_place(self):
    result = []
    upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digits = "0123456789"
    def is_useful_label(lbl):
      if (len(lbl) == 0): return False
      if (len(lbl) > 4): return False
      if (lbl[0] not in upper): return False
      for c in lbl[1:]:
        if (    c not in upper
            and c not in digits):
          return False
      return True
    lbl_set = set()
    def reset(label):
      if (sc.label != label):
        result.append((sc.label, label))
        sc.label = label
      lbl_set.add(label)
    for sc in self.scatterers():
      lbl = sc.label.strip().replace(" ", "").upper()
      lbl_is_useful = is_useful_label(lbl)
      if (lbl not in lbl_set and lbl_is_useful):
        reset(label=lbl)
      else:
        def find_tail_replacement(fmt, n):
          for i in range(1,n):
            s = fmt % i
            trial = lbl[:4-len(s)]+s
            if (trial not in lbl_set):
              reset(label=trial)
              return True
          return False
        def find_replacement_using_scattering_type():
          from cctbx.eltbx.xray_scattering import get_element_and_charge_symbols
          e, _ = get_element_and_charge_symbols(
            scattering_type=sc.scattering_type,
            exact=False)
          if (len(e) == 0): return False
          assert len(e) <= 2
          if (len(e) == 1): fmt, n = "%03d", 1000
          else:             fmt, n = "%02d", 100
          e = e.upper()
          for i in range(1,n):
            trial = e + fmt % i
            if (trial not in lbl_set):
              reset(label=trial)
              return True
          return False
        def find_complete_replacement():
          for c in upper:
            for i in range(1,1000):
              trial = c + "%03d" % i
              if (trial not in lbl_set):
                reset(label=trial)
                return True
          return False
        if (lbl_is_useful):
          if (find_tail_replacement("%d", 10)): continue
          if (find_tail_replacement("%02d", 100)): continue
          if (find_tail_replacement("%03d", 1000)): continue
        if (find_replacement_using_scattering_type()): continue
        if (find_complete_replacement()): continue
        raise RuntimeError(
          "Unable to find unused SHELX-compatible scatterer label.")
    return result

  def intersection_of_scatterers(self, i_seq, j_seq):
    """
    For a pair of scatterers, calculates their overlap given the coordinates
    and displacement parameters (using adptbx.intersection).
    """
    sc1 = self.scatterers()[i_seq]
    sc2 = self.scatterers()[j_seq]
    if (sc1.flags.use_u_aniso()):
      u1 = sc1.u_star
    else :
      u1 = sc1.u_iso
    if (sc2.flags.use_u_aniso()):
      u2 = sc2.u_star
    else :
      u2 = sc2.u_iso
    return adptbx.intersection(
      u_1=u1,
      u_2=u2,
      site_1=sc1.site,
      site_2=sc2.site,
      unit_cell=self.unit_cell())

class conservative_pair_proxies(object):

  def __init__(self, structure, bond_sym_table, conserve_angles):
    from cctbx import geometry_restraints
    buffer_thickness = flex.max_default(
      values=crystal.get_distances(
        pair_sym_table=bond_sym_table,
        orthogonalization_matrix
          =structure.unit_cell().orthogonalization_matrix(),
        sites_frac=structure.sites_frac()),
      default=1)
    if (conserve_angles): buffer_thickness *= 2
    asu_mappings = structure.asu_mappings(buffer_thickness=buffer_thickness)
    bond_pair_asu_table = crystal.pair_asu_table(asu_mappings=asu_mappings)
    bond_pair_asu_table.add_pair_sym_table(sym_table=bond_sym_table)
    self.bond = geometry_restraints.bond_sorted_asu_proxies(
      pair_asu_table=bond_pair_asu_table)
    if (not conserve_angles):
      self.angle = None
    else:
      angle_pair_asu_table = bond_pair_asu_table.angle_pair_asu_table()
      self.angle = geometry_restraints.bond_sorted_asu_proxies(
        pair_asu_table=angle_pair_asu_table)


class meaningful_site_cart_differences(object):
  """ Differences between the Cartesian coordinates of corresponding sites
  in two structures, cancelling continuous origin shifts if any.

  This is especially useful to compare a refined
  structure to a reference structure as the former may have drifted along
  a continuous shift direction during refinement, therefore spoiling
  a naive comparison of corresponding sites.
  """

  def __init__(self, xs1, xs2):
    self.labels = [ sc.label for sc in xs1.scatterers() ]
    self.delta = canonical_delta = xs1.sites_cart() - xs2.sites_cart()
    if xs1.space_group() == xs2.space_group():
      ssi = sgtbx.structure_seminvariants(xs1.space_group())\
                 .select(discrete=False)
      if ssi.size():
        shifts = [ matrix.col(xs1.unit_cell().orthogonalize(vm.v))
                   for vm in ssi.vectors_and_moduli() ]
        if len(shifts) == 1:
          e0 = shifts[0].normalize()
          e1 = e0.ortho()
          e2 = e0.cross(e1)
        elif len(shifts) == 2:
          e0 = shifts[0].normalize()
          v = shifts[1]
          e1 = (e0 - 1/e0.dot(v)*v).normalize()
          e2 = e0.cross(e1)
        elif len(shifts) == 3:
          e0, e1, e2 = [ (1,0,0), (0,1,0), (0,0,1) ]
        deltas = [ canonical_delta.dot(e) for e in (e0, e1, e2) ]
        means = [ flex.mean(d) for d in deltas ]
        if len(shifts) == 1:
          means_correction = (means[0], 0, 0)
        elif len(shifts) == 2:
          means_correction = (means[0], means[1], 0)
        elif len(shifts) == 3:
          means_correction = tuple(means)
        self.delta = flex.vec3_double(deltas[0], deltas[1], deltas[2])
        self.delta -= means_correction

  def max_absolute(self):
    return flex.max_absolute(self.delta.as_double())

  def show(self):
    for lbl, diff in six.moves.zip(self.labels, self.delta):
      print("%6s: (%.6f, %.6f, %.6f)" % ((lbl,) + diff))
