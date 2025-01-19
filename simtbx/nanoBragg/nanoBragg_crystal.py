"""
organizer for setting the nanoBragg crystal properties
"""
from __future__ import absolute_import, division, print_function
try:
    from collections.abc import Iterable
except ModuleNotFoundError:
    from collections import Iterable
from simtbx.nanoBragg import shapetype
from scitbx.matrix import sqr
from cctbx import sgtbx


class NBcrystal(object):

  def __init__(self, init_defaults=True):
    self.xtal_shape = None  # nanoBragg shapetypes, can be e.g. tophat, gauss, square, round
    self.Ncells_abc = None  # 3-tuple of  floats specifying mosaic domains size along a,b,c crystal axes
    self.isotropic_ncells = None  # whether Na=Nb=Nc is a restraint
    self.Ncells_def = None  # 3-tuple of the offiagonal mosaic domain size terms, e.g. NaNb, NaNc, NcNb
    self.thick_mm = None  # crystal thickness used to determine an approximate scale for the crystal (not used much in diffBragg)
    self.symbol = None  # space group symbol e.g. P6522
    self.miller_array = None  # cctbx miller array for setting structure factor amplitudes
    self.mos_spread_deg = None  # mosaic spread
    self.anisotropic_mos_spread_deg = None  # whether mosaic spread is defined using 1 or 3 parameters
    self.n_mos_domains = None  # how many mosaic domains are used to sample the mosaic rotational spread (spherical or ellipsoidal caps)
    self.umat_maker = None  # instance of nanoBragg.anisotropic_mosaicity.AnisoUmats
    self.dxtbx_crystal = None  # dxtbx crystal model for the crystal
    self.cb_op = None
    if init_defaults:
      self.init_defaults()

  def init_defaults(self):
    self.xtal_shape = "gauss"
    self.Ncells_abc = (10, 10, 10)
    self.isotropic_ncells = True
    self.Ncells_def = None
    self.thick_mm = 0.1
    self.symbol = "P43212"
    ucell = (79.1, 79.1, 38.4, 90, 90, 90)
    self.dxtbx_crystal = NBcrystal.dxtbx_crystal_from_ucell_and_symbol(
      ucell_tuple_Adeg=ucell, symbol=self.symbol)
    self.miller_array = NBcrystal.dummie_Fhkl(ucell, self.symbol)
    self.mos_spread_deg = 0
    self.anisotropic_mos_spread_deg = None
    self.n_mos_domains = 1
    self.umat_maker = None

  @property
  def has_anisotropic_mosaicity(self):
    return self.anisotropic_mos_spread_deg is not None

  @property
  def space_group_info(self):
    if self.symbol is not None:
      info = sgtbx.space_group_info(symbol=self.symbol)
      return info
    else:
      raise AttributeError("Set the space group symbol before calling for space_group_info!")

  @property
  def miller_array_high_symmetry(self):
    if self.symbol is not None and self.miller_array is not None:
      return self.miller_array.customized_copy(space_group_info=self.space_group_info)
    else:
      raise AttributeError("Set the symbol and miller_array properties first!")

  @property
  def symbol(self):
    return self._symbol

  @symbol.setter
  def symbol(self, val):
    self._symbol = val

  @property
  def Omatrix(self):
    """
    Change of basis operator
    """
    if self.dxtbx_crystal is None:
      raise AttributeError("Specify the dxtbx crystal object first!")
    sgi = self.dxtbx_crystal.get_space_group().info()
    to_p1 = sgi.change_of_basis_op_to_primitive_setting()
    return sqr(to_p1.c_inv().r().transpose().as_double())

  @property
  def dxtbx_crystal(self):
    return self._dxtbx_crystal

  @dxtbx_crystal.setter
  def dxtbx_crystal(self, val):
    self._dxtbx_crystal = val

  @property
  def miller_array(self):
    return self._miller_array

  @miller_array.setter
  def miller_array(self, val):
    if val is not None:
      if isinstance(val.data()[0], complex):
        self.miller_is_complex = True
      else:
        self.miller_is_complex = False
        if str(val.observation_type()) == "xray.intensity":
          val = val.as_amplitude_array()
      self.cb_op = val.space_group_info().change_of_basis_op_to_primitive_setting()
      val = val.expand_to_p1()
      val = val.generate_bijvoet_mates()
      dtrm = sqr(self.cb_op.c().r().as_double()).determinant()
      if not dtrm == 1:
        val = val.change_basis(self.cb_op)
    self._miller_array = val

  @property
  def Ncells_abc(self):
    return self._Ncells_abc

  @Ncells_abc.setter
  def Ncells_abc(self, val):
    self._Ncells_abc = val

  @property
  def Ncells_def(self):
    return self._Ncells_def

  @Ncells_def.setter
  def Ncells_def(self, val):
    self._Ncells_def = val

  @property
  def anisotropic_mos_spread_deg(self):
    return self._anisotropic_mos_spread_deg

  @anisotropic_mos_spread_deg.setter
  def anisotropic_mos_spread_deg(self, val):
    if val is not None:
      if not isinstance(val, Iterable):
        raise TypeError("anisotropic_mos_spread_deg needs to be a 3-tuple or 6-tuple")
      elif len(val) not in [3, 6]:
        raise ValueError("Anisotropic mosaicity should be either a 3-tuple or a 6-tuple")
    self._anisotropic_mos_spread_deg = val

  @property
  def mos_spread_deg(self):
    return self._mos_spread_deg

  @mos_spread_deg.setter
  def mos_spread_deg(self, val):
    self._mos_spread_deg = val

  @property
  def n_mos_domains(self):
    return self._n_mos_domains

  @n_mos_domains.setter
  def n_mos_domains(self, val):
    self._n_mos_domains = val

  @property
  def xtal_shape(self):
    if self._xtal_shape == "gauss":
      return shapetype.Gauss
    if self._xtal_shape == "gauss_star":
      return shapetype.Gauss_star
    elif self._xtal_shape == "gauss_argchk":
      return shapetype.Gauss_argchk
    elif self._xtal_shape == "round":
      return shapetype.Round
    elif self._xtal_shape == "square":
      return shapetype.Square
    else:
      return shapetype.Tophat  # default after init

  @xtal_shape.setter
  def xtal_shape(self, val):
    self._xtal_shape = val

  @property
  def thick_mm(self):
    return self._thick_mm

  @thick_mm.setter
  def thick_mm(self, val):
    self._thick_mm = val

  @staticmethod
  def dxtbx_crystal_from_ucell_and_symbol(ucell_tuple_Adeg, symbol):
    """
    :param ucell_tuple_Adeg:  unit cell tuple a,b,c al, be, ga in Angstom and degrees
    :param symbol: lookup symbol for space group, e.g. 'P1'
    :return:a default crystal in conventional orientation, a along x-axis
    """
    from cctbx import crystal
    from dxtbx.model.crystal import CrystalFactory
    symm = crystal.symmetry("%f,%f,%f,%f,%f,%f" % ucell_tuple_Adeg, symbol)

    ucell = symm.unit_cell()
    O = ucell.orthogonalization_matrix()
    real_space_a = O[0], O[3], O[6]
    real_space_b = O[1], O[4], O[7]
    real_space_c = O[2], O[5], O[8]

    hall_symbol = symm.space_group_info().type().hall_symbol()

    return CrystalFactory.from_dict(
      {'__id__': 'crystal',
       'real_space_a': real_space_a,
       'real_space_b': real_space_b,
       'real_space_c': real_space_c,
       'space_group_hall_symbol': hall_symbol})

  @staticmethod
  def dummie_Fhkl(ucell, symbol):
    from simtbx.nanoBragg.utils import fcalc_from_pdb
    Fhkl = fcalc_from_pdb(resolution=2, algorithm="fft", wavelength=1, symbol=symbol, ucell=ucell)
    return Fhkl
