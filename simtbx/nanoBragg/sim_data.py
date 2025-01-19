from __future__ import absolute_import, division, print_function

try:
    from collections.abc import Iterable
except ModuleNotFoundError:
    from collections import Iterable
from simtbx.diffBragg import diffBragg
from scitbx.array_family import flex
import numpy as np
from simtbx.nanoBragg.anisotropic_mosaicity import AnisoUmats
from simtbx.nanoBragg import shapetype, nanoBragg
from simtbx.nanoBragg.nanoBragg_crystal import NBcrystal
from simtbx.nanoBragg.nanoBragg_beam import NBbeam
from copy import deepcopy
from scitbx.matrix import sqr, col
import math


def Amatrix_dials2nanoBragg(crystal):
  """
  returns the A matrix from a cctbx crystal object
  in nanoBragg format
  :param crystal: cctbx crystal
  :return: Amatrix as a tuple
  """
  sgi = crystal.get_space_group().info()
  cb_op = sgi.change_of_basis_op_to_primitive_setting()
  dtrm = sqr(cb_op.c().r().as_double()).determinant()
  if not dtrm == 1:
    raise ValueError('You need to convert your crystal model to its primitive setting first')
  Amatrix = sqr(crystal.get_A()).transpose()
  return Amatrix


def determine_spot_scale(beam_size_mm, crystal_thick_mm, mosaic_vol_mm3):
  """
  :param beam_size_mm:  diameter of beam focus (millimeter)
  :param crystal_thick_mm: thickness of crystal (millimeter)
  :param mosaic_vol_mm3:  volume of a mosaic block in crystal (cubic mm)
  :return: roughly the number of exposed mosaic blocks
  """
  if beam_size_mm <= crystal_thick_mm:
    illum_xtal_vol = crystal_thick_mm * beam_size_mm ** 2
  else:
    illum_xtal_vol = crystal_thick_mm ** 3
  return illum_xtal_vol / mosaic_vol_mm3


class SimData:
  def __init__(self, use_default_crystal=False):
    self.detector = SimData.simple_detector(180, 0.1, (512, 512))  # dxtbx detector model
    self.seed = 1  # nanoBragg seed member
    self.crystal = NBcrystal(use_default_crystal)
    self.add_air = False  # whether to add air in the generate_simulated_image method
    self.Umats_method = 0  # how to generate mosaic rotation umats (can be 0,1,2,3)
    self.add_water = True  # whether to add water in the generate_simulated_image method
    self.water_path_mm = 0.005  # path length through water
    self.air_path_mm = 0  # path length through air
    self._mosaicity_crystal = None  # a dxtbx crystal used for anisotropic mosaic spread model
    self.using_diffBragg_spots = False  # whether to use diffBragg
    nbBeam = NBbeam()
    nbBeam.unit_s0 = (0, 0, -1)
    self.beam = nbBeam  # nanoBragg_beam object
    self.using_cuda = False  # whether to use giles mullen cuda acceleration
    self.using_omp = False  # whether to use add_nanoBragg_spots_nks with open mp accel
    self.rois = None  # a list of rois for simiuting rois only
    self.readout_noise = 3  # detector readout noise
    self.gain = 1  # detector gain
    self.psf_fwhm = 0  # detector point spread width
    self.include_noise = True  # include noise in the simulation if using generate_simulated_image method
    self.background_raw_pixels = None  # background raw pixels, should be a 2D flex double array
    self.backrground_scale = 1  # scale factor to apply to background raw pixels
    self.mosaic_seeds = 777, 777  # two random seeds governing the legacy Umats method
    self.D = None  # nanoBragg or diffBragg instance
    self.panel_id = 0  # detector panel id
    self.umat_maker = None  # an instance of AnisoUmats for generating ensembles of mosaic rotations
    self.ucell_man = None   # place holder for a unit cell manager (used in hopper)
    self.Nabc_params = None   # RangedParameter (refiners/parameters) objects used by hopper
    self.RotXYZ_params = None  # RangedParameter (refiners/parameters) objects used by hopper
    self.ucell_params = None  # RangedParameter (refiners/parameters) objects used by hopper
    self.Scale_params = None  # RangedParameter (refiners/parameters) objects used by hopper
    self.DetZ_params = None  # RangedParameter (refiners/parameters) objects used by hopper
    self.num_xtals = 1  # number of xtals, used in the hopper script
    self.dxtbx_spec = None  # spectrum object from dxtbx
    self.functionals = []  # target functionals  container ?
    self.use_psf = False  # flag to specify whether psf should be applied to the model
    self.PSF = None   # place holder for the point-spread-function returned by diffBragg/psf.py method makeMoffat_integPSF
    self.psf_args = None # place holder for a dictionary containing the 3 PSF arguments (see example in diffBragg/psf.py

  @property
  def background_raw_pixels(self):
    return self._background_raw_pixels

  @background_raw_pixels.setter
  def background_raw_pixels(self, val):
    self._background_raw_pixels = val

  @property
  def gain(self):
    return self._gain

  @gain.setter
  def gain(self, val):
    self._gain = val

  @staticmethod
  def default_panels_fast_slow(detector):
    Npanel = len(detector)
    nfast, nslow = detector[0].get_image_size()
    slows, fasts = np.indices((nslow, nfast))
    fasts = list(map(int, np.ravel(fasts)))
    slows = list(map(int, np.ravel(slows)))
    fasts = fasts*Npanel
    slows = slows*Npanel
    pids = []
    for pid in range(Npanel):
      pids += [pid]*(nfast*nslow)

    npix = nslow*nfast*Npanel
    panels_fasts_slows = np.zeros(npix*3, int)
    panels_fasts_slows[0::3] = pids
    panels_fasts_slows[1::3] = fasts
    panels_fasts_slows[2::3] = slows
    return flex.size_t(panels_fasts_slows)

  @property
  def air_path_mm(self):
    return self._air_path_mm

  @air_path_mm.setter
  def air_path_mm(self, val):
    self._air_path_mm = val

  @property
  def water_path_mm(self):
    return self._water_path_mm

  @water_path_mm.setter
  def water_path_mm(self, val):
    self._water_path_mm = val

  @property
  def crystal(self):
    return self._crystal

  @crystal.setter
  def crystal(self, val):
    self._crystal = val

  @property
  def beam(self):
    return self._beam

  @beam.setter
  def beam(self, val):
    self._beam = val

  @property
  def seed(self):
    return self._seed

  @seed.setter
  def seed(self, val):
    self._seed = val

  @staticmethod
  def Umats(mos_spread_deg, n_mos_doms, isotropic=True, seed=777, norm_dist_seed=777):
    import scitbx
    from scitbx.matrix import col
    import scitbx.math
    UMAT_nm = flex.mat3_double()
    mersenne_twister = flex.mersenne_twister(seed=seed)
    scitbx.random.set_random_seed(norm_dist_seed)
    rand_norm = scitbx.random.normal_distribution(mean=0, sigma=(mos_spread_deg * math.pi / 180.0))
    g = scitbx.random.variate(rand_norm)
    mosaic_rotation = g(n_mos_doms)
    for m in mosaic_rotation:
      site = col(mersenne_twister.random_double_point_on_sphere())
      if mos_spread_deg > 0:
        UMAT_nm.append(col(scitbx.math.r3_rotation_axis_and_angle_as_matrix(site, m)))
      else:
        UMAT_nm.append(col(scitbx.math.r3_rotation_axis_and_angle_as_matrix(site, 0)))
      if isotropic and mos_spread_deg > 0:
        UMAT_nm.append(col(scitbx.math.r3_rotation_axis_and_angle_as_matrix(site, -m)))
    return UMAT_nm

  @staticmethod
  def plot_isotropic_umats(UMAT_nm, mos_spread_deg):

    # analysis of rotation angles
    from simtbx.nanoBragg.tst_gaussian_mosaicity import check_distributions
    angle_deg, rot_ax = check_distributions.get_angle_deg_and_axes(UMAT_nm) # in degrees
    nm_rms_angle = math.sqrt(flex.mean(angle_deg*angle_deg))

    from matplotlib import pyplot as plt
    fig_angles,axis_angles = plt.subplots(1,1)
    axis_angles.set_xlabel("rotation (°)")
    axis_angles.hist(angle_deg,bins=int(math.sqrt(len(UMAT_nm))))
    axis_angles.set_title(
    "Histogram of rotation angles\nrms %.4f° expected %.4f°"%(nm_rms_angle, mos_spread_deg))

    # analysis of rotation axes
    fig, axes = plt.subplots(1, 3,figsize=(10,5))
    axes[0].plot(rot_ax.parts()[0], rot_ax.parts()[1], "b,") # looking down from the pole
    axes[0].set_aspect("equal")
    axes[0].set_title("Rot_ax projected on z")
    axes[0].set_xlim(-1.1,1.1)
    axes[0].set_ylim(-1.1,1.1)

    axes[1].plot(rot_ax.parts()[1], rot_ax.parts()[2], "b,") # looking at equator above prime meridian
    axes[1].set_aspect("equal")
    axes[1].set_title("Rot_ax projected on x")
    axes[1].set_xlim(-1.1,1.1)
    axes[1].set_ylim(-1.1,1.1)

    axes[2].plot(rot_ax.parts()[2], rot_ax.parts()[0], "b,") # looking at equator above 90-deg meridian
    axes[2].set_aspect("equal")
    axes[2].set_title("Rot_ax projected on y")
    axes[2].set_xlim(-1.1,1.1)
    axes[2].set_ylim(-1.1,1.1)

    # action on unit vectors
    cube_diag = math.sqrt(1./3) # 0.57735
    unit_vectors = [(1,0,0), (0,1,0), (0,0,1), (cube_diag, cube_diag, cube_diag)]
    fig3,axes3 = plt.subplots(1,4,sharey=True,figsize=(15,7))
    axes3[0].set_ylabel("unit projection")
    for icol, RLP in enumerate(unit_vectors):
          RLP = col(RLP)
          axis = axes3[icol]
          unit = RLP.normalize()
          seed = col(unit_vectors[(icol+2)%(len(unit_vectors))])
          perm2 = unit.cross(seed)
          perm3 = unit.cross(perm2)
          a2 = flex.double(); a3 = flex.double()
          angles_deg = flex.double()
          for u in UMAT_nm:
            U = sqr(u)
            newvec = U * unit
            angles_deg.append((180./math.pi)*math.acos(newvec.dot(unit)))
            a2.append(newvec.dot(perm2)); a3.append(newvec.dot(perm3))
          rms_angle = math.sqrt(flex.mean(angles_deg*angles_deg))
          axis.plot (a2,a3,'r,')[0]
          axis.set_aspect("equal")
          axis_str = "(%5.3f,%5.3f,%5.3f)"%(RLP.elems) if icol==3 else "(%.0f,%.0f,%.0f)"%(RLP.elems)
          axis.set_title("Transformation of unit vector\n%s\nrms %.4f° expected %.4f°"%(
            axis_str, rms_angle, math.sqrt(mos_spread_deg*mos_spread_deg*(2/3)))) # Pythagorean thm
          axis.set_xlabel("unit projection")
          axis.set_xlim(-0.0005,0.0005)
          axis.set_ylim(-0.0005,0.0005)

    plt.show()

  @property
  def Umats_method(self):
    return self._Umats_method

  @Umats_method.setter
  def Umats_method(self, val):
    permitted = [0, 2, 3, 5]
    # 0 is double_random, used for exafel tests
    # 2 is double_uniform, isotropic method
    # 3 is double_uniform, anisotropic method
    # 5 is the same as 2 and 5 is deprecated
    if val not in permitted:
      raise ValueError("Umats method needs to be in %s (4 not yet supported)"%permitted)
    if val == 5:
      print("WARNING: method 5 is really method 2, method 5 is deprecated!")
      val = 2
    self._Umats_method = val

  @property
  def psf_fwhm(self):
    return self._psf_fwhm

  @psf_fwhm.setter
  def psf_fwhm(self, val):
    self._psf_fwhm = val

  @property
  def readout_noise(self):
    return self._readout_noise

  @readout_noise.setter
  def readout_noise(self, val):
    self._readout_noise = val

  @property
  def add_air(self):
    return self._add_air

  @add_air.setter
  def add_air(self, val):
    self._add_air = val

  @property
  def add_water(self):
    return self._add_water

  @add_water.setter
  def add_water(self, val):
    self._add_water = val

  @property
  def detector(self):
    return self._detector

  @detector.setter
  def detector(self, val):
    self._detector = val

  @property
  def rois(self):
    return self._rois

  @rois.setter
  def rois(self, val):
    self._rois = val

  @property
  def using_omp(self):
    return self._using_omp

  @using_omp.setter
  def using_omp(self, val):
    assert val in (True, False)
    self._using_omp = val

  @property
  def using_diffBragg_spots(self):
    return self._using_diffBragg_spots

  @using_diffBragg_spots.setter
  def using_diffBragg_spots(self, val):
    assert(val in [True, False])
    self._using_diffBragg_spots = val

  @property
  def using_cuda(self):
    return self._using_cuda

  @using_cuda.setter
  def using_cuda(self, val):
    assert(val in [True, False])
    self._using_cuda = val

  @property
  def include_noise(self):
    return self._include_noise

  @include_noise.setter
  def include_noise(self, val):
    self._include_noise = val

  def get_detector_corner_res(self):
    dmin = np.inf
    for p in self.detector:
      panel_corner_res = p.get_max_resolution_at_corners(self.beam.nanoBragg_constructor_beam.get_s0())
      dmin = np.min([dmin, panel_corner_res])
    return dmin

  def update_Fhkl_tuple(self):
    if self.crystal.miller_array is not None:
      if np.all(self.crystal.miller_array.data().as_numpy_array()==0):
        raise ValueError("Seems all miller indices are 0")
      d_max, _ = self.crystal.miller_array.resolution_range()
      d_min = self.get_detector_corner_res()
      ma_on_detector = self.crystal.miller_array.resolution_filter(d_min=d_min, d_max=d_max)
      if self.using_diffBragg_spots and self.crystal.miller_is_complex:
        Freal, Fimag = zip(*[(val.real, val.imag) for val in ma_on_detector.data()])
        Freal = flex.double(Freal)
        Fimag = flex.double(Fimag)
        self.D.Fhkl_tuple = ma_on_detector.indices(), Freal, Fimag
      elif self.using_diffBragg_spots:
        inds = ma_on_detector.indices()
        dat = ma_on_detector.data()
        self.D.Fhkl_tuple = inds, dat, None
      else:
        self.D.Fhkl_tuple = ma_on_detector.indices(), ma_on_detector.data(), None

  def set_dspace_binning(self, nbins, verbose=False):
    dsp = self.D.Fhkl.d_spacings().data().as_numpy_array()
    dsp = np.sort(dsp[dsp >= self.get_detector_corner_res()])
    bins = [vals[0] for vals in np.array_split(dsp, nbins)] + [dsp[-1]]
    bins[0] = bins[0]-1e-4
    bins[-1] = bins[-1]+1e-4
    if verbose:
      for d in bins:
        print(d)
    self.D.dspace_bins = bins

  def _crystal_properties(self):
    if self.crystal is None:
      return

    model_mosaic_spread = self.crystal.n_mos_domains > 1
    if not model_mosaic_spread:
      self.D.mosaic_spread_deg = 0
      self.D.mosaic_domains = 1

    self.D.xtal_shape = self.crystal.xtal_shape
    self.D.hall_symbol = self.crystal.space_group_info.type().hall_symbol()
    if hasattr(self.D, "laue_group_num"):
      self.D.laue_group_num = 1

    # TODO: am I unnecessary?
    self.D.unit_cell_tuple = self.crystal.dxtbx_crystal.get_unit_cell().parameters()
    if self.using_diffBragg_spots:
      self.D.Omatrix = self.crystal.Omatrix
      self.D.Bmatrix = self.crystal.dxtbx_crystal.get_B() #
      self.D.Umatrix = self.crystal.dxtbx_crystal.get_U()

    if self.using_diffBragg_spots:
      shape = str(self.crystal.xtal_shape)
      if shape not in ["Gauss", "Gauss_star", "Square"]:
        raise ValueError("nanoBragg_crystal xtal_shape should be square, gauss or gauss_star")
      self.update_Fhkl_tuple()
      self.D.unit_cell_tuple = self.crystal.dxtbx_crystal.get_unit_cell().parameters()
      # init mosaic domain size:
      if self.crystal.isotropic_ncells:
        self.D.Ncells_abc = self.crystal.Ncells_abc[0]
      else:
        self.D.Ncells_abc_aniso = tuple(self.crystal.Ncells_abc)
      if self.crystal.Ncells_def is not None:
        self.D.Ncells_def = self.crystal.Ncells_def

      # init mosaicity
      self._init_diffBragg_umats()

    else:
      self.D.xtal_shape = self.crystal.xtal_shape
      self.update_Fhkl_tuple()
      self.D.Amatrix = Amatrix_dials2nanoBragg(self.crystal.dxtbx_crystal)
      Nabc = self.crystal.Ncells_abc
      if len(Nabc) == 1:
        Nabc = Nabc[0], Nabc[0], Nabc[0]
      self.D.Ncells_abc = Nabc
      self._init_nanoBragg_umats(model_mosaic_spread)

  def _init_nanoBragg_umats(self, model_mosaic_spread):

    # umat implementation for the exascale api
    if model_mosaic_spread and self.umat_maker is None:
      # initialize the parameterized distribution
      assert self.crystal.n_mos_domains % 2 == 0
      self.umat_maker = AnisoUmats(num_random_samples=2*self.crystal.n_mos_domains)

    if self.crystal.anisotropic_mos_spread_deg is None:
      # isotropic case
      self.D.mosaic_spread_deg = self.crystal.mos_spread_deg
      self.D.mosaic_domains = self.crystal.n_mos_domains

      if self.Umats_method == 0: # double_random legacy method, exafel tests
        Umats = SimData.Umats(self.crystal.mos_spread_deg,
                              self.crystal.n_mos_domains,
                              seed=self.mosaic_seeds[0],
                              norm_dist_seed=self.mosaic_seeds[1])
      elif self.Umats_method == 2: # double_uniform isotropic, NOTE: if Umats_method was set as 5, it was modified to be 2
        Umats, _,_ = self.umat_maker.generate_Umats(
          self.crystal.mos_spread_deg, compute_derivs=False, crystal=None, how=2)
      else: raise ValueError("Invalid method for nanoBragg isotropic case")
      #SimData.plot_isotropic_umats(Umats, self.crystal.mos_spread_deg)

    else:
      # anisotropic case with diagonal elements
      Umats, _,_ = self.umat_maker.generate_Umats(
        self.crystal.anisotropic_mos_spread_deg, compute_derivs=False, crystal=self.crystal.dxtbx_crystal, how=1)

    self.exascale_mos_blocks = flex.mat3_double(Umats)
    self.D.set_mosaic_blocks(self.exascale_mos_blocks)

  def _init_diffBragg_umats(self):
    """
    Initializes the mosaic spread framework.
    If the self.crystal.n_mos_domains >1 then a umat_maker is created
    otherwise, no mosaic spread will be modeled
    """
    if self.crystal.anisotropic_mos_spread_deg is not None:
      if tuple(self.crystal.anisotropic_mos_spread_deg) == (0,0,0) and self.crystal.n_mos_domains != 1:
        raise ValueError("If more than 1 mosaic domain are passed, must set a positive value for anisotropic_mos_spread")
      if len(set(self.crystal.anisotropic_mos_spread_deg))==1:
        self.D.has_anisotropic_mosaic_spread = False
        mosaicity = self.crystal.anisotropic_mos_spread_deg[0]
      else:
        self.D.has_anisotropic_mosaic_spread = True
        mosaicity = self.crystal.anisotropic_mos_spread_deg
    else:
      self.D.has_anisotropic_mosaic_spread = False
      mosaicity = self.crystal.mos_spread_deg

    if self.crystal.n_mos_domains==1:
      # we wont set/compute gradients in this case!
      self.D.mosaic_domains = 1
      Umats = [(1, 0, 0, 0, 1, 0, 0, 0, 1)]
      Umats_prime = [(0, 0, 0, 0, 0, 0, 0, 0, 0)]
      self.D.set_mosaic_blocks(Umats)
      self.D.set_mosaic_blocks_prime(Umats_prime)
      self.D.vectorize_umats()
    else:
      assert self.crystal.n_mos_domains % 2 == 0, "Need an even number of mosaic domains!"
      self.D.mosaic_domains = self.crystal.n_mos_domains
      if isinstance(mosaicity, Iterable):
        self.Umats_method=3
        self.D.mosaic_spread_deg = np.mean(mosaicity)
        self._mosaicity_crystal = deepcopy(self.crystal.dxtbx_crystal)
        assert self._mosaicity_crystal is not None

      else:
        self.Umats_method=2
        self.D.mosaic_spread_deg = mosaicity

      self.umat_maker = AnisoUmats(num_random_samples=self.crystal.n_mos_domains)
      self.update_umats_for_refinement(mosaicity)

  def update_umats_for_refinement(self, mos_spread):
    """
    Given the value of mosaic spread, update the umatrices and their derivatives
    :param mos_spread: float (isotropic spread) or 3-tuple (anisotropic spread)
    """
    assert self.Umats_method in [2,3]
    if not hasattr(self, "D"):
      raise AttributeError("Cannot set umats if diffBragg is not yet instantiated")

    if self.Umats_method == 2:
      # isotropic case, mos_spread shuld be a float!
      Umats, Umats_prime, Umats_dbl_prime = self.umat_maker.generate_Umats(mos_spread, self._mosaicity_crystal, how=2, compute_derivs=True)
    else:
      # anisotropic case, mos spread should be a 3-tuple of floats!
      Umats, Umats_prime, Umats_dbl_prime = self.umat_maker.generate_Umats(mos_spread, self._mosaicity_crystal,how=1, transform_eta=True, compute_derivs=True)
      Umats_prime = Umats_prime[0::3] + Umats_prime[1::3] + Umats_prime[2::3]
      Umats_dbl_prime = Umats_dbl_prime[0::3] + Umats_dbl_prime[1::3] + Umats_dbl_prime[2::3]

    self.D.set_mosaic_blocks(Umats)
    self.D.set_mosaic_blocks_prime(Umats_prime)
    self.D.set_mosaic_blocks_dbl_prime(Umats_dbl_prime)
    # here we move the umats from the flex mat3 into vectors of Eigen:
    self.D.vectorize_umats()

  def _beam_properties(self):
    self.D.xray_beams = self.beam.xray_beams
    self.D.beamsize_mm = self.beam.size_mm

  def _seedlings(self):
    self.D.seed = self.seed
    self.D.calib_seed = self.seed
    self.D.mosaic_seed = self.seed

  def determine_spot_scale(self):
    if self.crystal is None:
      return 1
    if self.beam.size_mm <= self.crystal.thick_mm:
      illum_xtal_vol = self.crystal.thick_mm * self.beam.size_mm**2
    else:
      illum_xtal_vol = self.crystal.thick_mm**3
    mosaic_vol = self.D.xtal_size_mm[0]*self.D.xtal_size_mm[1]*self.D.xtal_size_mm[2]
    return illum_xtal_vol / mosaic_vol

  def update_nanoBragg_instance(self, parameter, value):
    setattr(self.D, parameter, value)

  @property
  def panel_id(self):
    return self._panel_id

  @panel_id.setter
  def panel_id(self, val):
    if val >= len(self.detector):
      raise ValueError("panel id cannot be larger than the number of panels in detector (%d)" % len(self.detector))
    if val <0:
      raise ValueError("panel id cannot be negative!")
    if self.D is None:
      self._panel_id = 0
    else:
      self.D.set_dxtbx_detector_panel(self.detector[int(val)], self.beam.nanoBragg_constructor_beam.get_s0())
      self._panel_id = int(val)

  def instantiate_nanoBragg(self, verbose=0, oversample=0, device_Id=0, adc_offset=0,
                            default_F=1e3, interpolate=0):

    self.instantiate_diffBragg(verbose=verbose, oversample=oversample, device_Id=device_Id,
                               adc_offset=adc_offset, default_F=default_F, interpolate=interpolate,
                               use_diffBragg=False)

  def instantiate_diffBragg(self, verbose=0, oversample=0, device_Id=0,
                            adc_offset=0, default_F=1e3, interpolate=0, use_diffBragg=True,
                            auto_set_spotscale=False):

    if not use_diffBragg:
      self.D = nanoBragg(self.detector, self.beam.nanoBragg_constructor_beam,
                         verbose=verbose, panel_id=int(self.panel_id))
    else:
      self.D = diffBragg(self.detector,
                         self.beam.nanoBragg_constructor_beam,
                         verbose)
    self.using_diffBragg_spots = use_diffBragg
    self._seedlings()
    self.D.interpolate = interpolate
    self._crystal_properties()
    self._beam_properties()
    if auto_set_spotscale:
      self.D.spot_scale = self.determine_spot_scale()
    self.D.adc_offset_adu = adc_offset
    self.D.default_F = default_F

    if oversample > 0:
      self.D.oversample = int(oversample)

    self.D.device_Id = device_Id
    if not self.using_diffBragg_spots:
      self._full_roi = self.D.region_of_interest
    else:
      self.D.vectorize_umats()

  def generate_simulated_image(self, instantiate=False, use_diffBragg=True):
    if instantiate:
      self.instantiate_diffBragg(use_diffBragg=use_diffBragg)
    self._add_nanoBragg_spots()
    self._add_background()
    if self.include_noise:
      self._add_noise()
    return self.D.raw_pixels.as_numpy_array()

  def _add_nanoBragg_spots(self):
    if self.using_diffBragg_spots:
      self.D.add_diffBragg_spots()
    else:
      rois = self.rois
      if rois is None:
        rois = [self._full_roi]
      _rawpix = None  # cuda_add_spots doesnt add spots, it resets each time.. hence we need this
      for roi in rois:
        if len(roi)==4:
          roi = (roi[0], roi[1]), (roi[2],roi[3])
        self.D.region_of_interest = roi
        if self.using_cuda:
          self.D.add_nanoBragg_spots_cuda()
          if _rawpix is None and len(rois) > 1:
            _rawpix = deepcopy(self.D.raw_pixels)
          elif _rawpix is not None:
            _rawpix += self.D.raw_pixels

        elif self.using_omp:
          from boost_adaptbx.boost.python import streambuf  # will deposit printout into dummy StringIO as side effect
          from six.moves import StringIO
          self.D.progress_meter = False
          self.D.add_nanoBragg_spots_nks(streambuf(StringIO()))
        else:
          self.D.add_nanoBragg_spots()

      if self.using_cuda and _rawpix is not None:
        self.D.raw_pixels = _rawpix

  def _add_background(self):
    if self.background_raw_pixels is not None:
      self.D.raw_pixels += self.background_raw_pixels
    else:
      if self.add_water:
        print('add water %f mm' % self.water_path_mm)
        water_scatter = flex.vec2_double([
          (0, 2.57), (0.0365, 2.58), (0.07, 2.8), (0.12, 5), (0.162, 8), (0.18, 7.32), (0.2, 6.75),
          (0.216, 6.75), (0.236, 6.5), (0.28, 4.5), (0.3, 4.3), (0.345, 4.36), (0.436, 3.77), (0.5, 3.17)])
        self.D.Fbg_vs_stol = water_scatter
        self.D.amorphous_sample_thick_mm = self.water_path_mm
        self.D.amorphous_density_gcm3 = 1
        self.D.amorphous_molecular_weight_Da = 18
        self.D.add_background(1, 0)

    if self.add_air:
      print("add air %f mm" % self.air_path_mm)
      air_scatter = flex.vec2_double([(0, 14.1), (0.045, 13.5), (0.174, 8.35), (0.35, 4.78), (0.5, 4.22)])
      self.D.Fbg_vs_stol = air_scatter
      self.D.amorphous_sample_thick_mm = self.air_path_mm
      self.D.amorphous_density_gcm3 = 1.2e-3
      self.D.amorphous_sample_molecular_weight_Da = 28  # nitrogen = N2
      self.D.add_background(1, 0)

  def _add_noise(self):
    self.D.detector_psf_kernel_radius_pixels = 5
    self.D.detector_psf_type = shapetype.Unknown
    self.D.detector_psf_fwhm_mm = self.psf_fwhm
    self.D.readout_noise = self.readout_noise
    self.D.quantum_gain = self.gain
    self.D.add_noise()

  @staticmethod
  def simple_detector(detector_distance_mm, pixelsize_mm, image_shape,
                      fast=(1, 0, 0), slow=(0, -1, 0)):
    from dxtbx.model.detector import DetectorFactory
    import numpy as np
    trusted_range = 0, 2e14
    detsize_s = image_shape[0]*pixelsize_mm
    detsize_f = image_shape[1]*pixelsize_mm
    cent_s = (detsize_s + pixelsize_mm*2)/2.
    cent_f = (detsize_f + pixelsize_mm*2)/2.
    beam_axis = np.cross(fast, slow)
    origin = -np.array(fast)*cent_f - np.array(slow)*cent_s + beam_axis*detector_distance_mm

    return DetectorFactory.make_detector("", fast, slow, origin,
                                         (pixelsize_mm, pixelsize_mm), image_shape, trusted_range)


if __name__ == "__main__":
  S = SimData(use_default_crystal=True)
  img = S.generate_simulated_image(instantiate=True, use_diffBragg=False)
  print ("Maximum pixel value: %.3g" % img.max())
  print ("Minimum pixel value: %.3g" % img.min())
