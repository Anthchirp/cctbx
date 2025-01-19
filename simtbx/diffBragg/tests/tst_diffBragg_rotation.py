from __future__ import absolute_import, division, print_function
from simtbx.diffBragg import utils, hopper_utils
from simtbx.nanoBragg import nanoBragg
from simtbx.nanoBragg.tst_nanoBragg_multipanel import beam as dxtbx_beam
from simtbx.nanoBragg.tst_nanoBragg_multipanel import whole_det as dxtbx_det
from simtbx.nanoBragg import nanoBragg_crystal, nanoBragg_beam, sim_data
from iotbx.crystal_symmetry_from_any import extract_from
from dxtbx_model_ext import Crystal
from scitbx import matrix
import numpy as np
import os
import libtbx.load_env # possibly implicit

pdb_lines = """HEADER TEST
CRYST1   50.000   60.000   70.000  90.00  90.00  90.00 P 1
ATOM      1  O   HOH A   1      56.829   2.920  55.702  1.00 20.00           O
ATOM      2  O   HOH A   2      49.515  35.149  37.665  1.00 20.00           O
ATOM      3  O   HOH A   3      52.667  17.794  69.925  1.00 20.00           O
ATOM      4  O   HOH A   4      40.986  20.409  18.309  1.00 20.00           O
ATOM      5  O   HOH A   5      46.896  37.790  41.629  1.00 20.00           O
ATOM      6 SED  MSE A   6       1.000   2.000   3.000  1.00 20.00          SE
END
"""

FAST, SLOW=228,104  # center of one of the Bragg peaks

def write_test_pdb(fileout="test.pdb"):
  from iotbx import pdb
  pdb_inp = pdb.input(source_info=None,lines = pdb_lines)
  pdb_inp.write_pdb_file(fileout)
  return fileout

def run_sim(testpdb, version="nanoBragg", spindle_axis=(1,0,0), phi_start=0, osc_deg=-1, phisteps=-1, verbose=False,
            manual_mode=False, manual_angle=0.):
  # crystal
  symmetry=extract_from(testpdb)
  sg = str(symmetry.space_group_info())
  fmat = matrix.sqr(symmetry.unit_cell().fractionalization_matrix())
  dxtbx_cryst = Crystal(fmat, sg)
  if manual_mode:
      sa = matrix.col(spindle_axis)
      U = matrix.sqr(dxtbx_cryst.get_U())
      Rphi = sa.axis_and_angle_as_r3_rotation_matrix(manual_angle, deg=True)
      U = Rphi*U
      dxtbx_cryst.set_U(U.elems)

  crystal = nanoBragg_crystal.NBcrystal(init_defaults=True)
  crystal.isotropic_ncells = False
  crystal.dxtbx_crystal = dxtbx_cryst
  crystal.Ncells_abc = 10,10,10
  crystal.n_mos_domains = 1 # TODO: setting this causes discrepancy
  crystal.mos_spread_deg = 0#1
  symbol = dxtbx_cryst.get_space_group().info().type().lookup_symbol()
  ucell_p = dxtbx_cryst.get_unit_cell().parameters()
  miller_data = utils.make_miller_array(symbol, ucell_p, defaultF=1000)
  crystal.symbol = miller_data.crystal_symmetry().space_group_info().type().lookup_symbol()
  crystal.miller_array = miller_data
  # beam
  beam = nanoBragg_beam.NBbeam()
  beam.size_mm = 0.001
  beam.unit_s0 = dxtbx_beam.get_unit_s0()
  spectrum = [(dxtbx_beam.get_wavelength(), 1e12)]
  beam.spectrum = spectrum
  # detector
  fsize, ssize = dxtbx_det[0].get_image_size()
  pfs = hopper_utils.full_img_pfs((1,ssize,fsize))
  # simulator
  SIM = sim_data.SimData()
  SIM.detector = utils.strip_thickness_from_detector(dxtbx_det)
  SIM.detector = dxtbx_det
  SIM.crystal = crystal

  SIM.beam = beam
  SIM.panel_id = 0
  def setup_rotation(SIM):
    if not manual_mode:
      SIM.D.spindle_axis = spindle_axis
      SIM.D.phi_deg = phi_start
      SIM.D.osc_deg = osc_deg
      SIM.D.phisteps = phisteps
  if version == "nanoBragg":
    SIM.instantiate_nanoBragg(oversample=1, device_Id=0, default_F=0, interpolate=0)
    setup_rotation(SIM)
    if verbose:
      SIM.D.printout_pixel_fastslow = FAST,SLOW
      SIM.D.show_params()

    SIM.D.add_nanoBragg_spots()
    pix = SIM.D.raw_pixels.as_numpy_array()
    SIM.D.free_all()
    return pix
  else:
    SIM.instantiate_diffBragg(oversample=1, device_Id=0, default_F=0, interpolate=0)
    SIM.D.xray_beams = SIM.beam.xray_beams
    ucell_man = utils.manager_from_params(ucell_p)
    Bmatrix = ucell_man.B_recipspace
    SIM.D.Bmatrix = Bmatrix

    npix = int(len(pfs)/3)
    setup_rotation(SIM)
    if verbose:
      SIM.D.printout_pixel_fastslow = FAST,SLOW
      SIM.D.show_params()

    SIM.D.add_diffBragg_spots_full()
    pix = SIM.D.raw_pixels_roi.as_numpy_array()
    hopper_utils.free_SIM_mem(SIM)
    return pix

def assert_identical_pixels(pix1, pix2):
  assert np.allclose(pix1,pix2, atol=0.1)

def test_manual_rotation():
    testpdb = write_test_pdb(fileout="test.pdb")
    SIM = nanoBragg()
    SIM.phi_deg = 0
    SIM.osc_deg = 0.5
    SIM.phisteps = 50
    phistep_deg = SIM.osc_deg / SIM.phisteps
    assert phistep_deg == SIM.phistep_deg # Note, this was broken by a bug recently

    pix_sums = []
    for version in ("nanoBragg", "diffBragg"):
        pix = None
        for phi_tic in range(SIM.phisteps):
            manual_angle = SIM.phi_deg + phi_tic*phistep_deg
            print("phiTic=%d, deltaPhi=%f deg." % (phi_tic, manual_angle))
            temp = run_sim(testpdb, version=version, spindle_axis=(1, 0, 0), phi_start=0, osc_deg=-1, phisteps=-1,
                    verbose=False, manual_mode=True, manual_angle=manual_angle)
            if pix is None:
                pix = temp
            else:
                pix += temp
        pix /= SIM.phisteps
        pix_sums.append(pix)
    nb_pix, db_pix = pix_sums
    assert np.allclose(nb_pix.ravel(), db_pix)
    print("tst manual diffBragg vs manual nanoBragg: OK!")

    temp_db = run_sim(testpdb, version="diffBragg", spindle_axis=(1, 0, 0), phi_start=SIM.phi_deg, osc_deg=SIM.osc_deg,
                   phisteps=SIM.phisteps, verbose=False, manual_mode=False)
    assert np.allclose(db_pix, temp_db)
    print("tst manual vs internal diffBragg: OK!")

    temp_nb = run_sim(testpdb, version="nanoBragg", spindle_axis=(1, 0, 0), phi_start=SIM.phi_deg, osc_deg=SIM.osc_deg,
                      phisteps=SIM.phisteps, verbose=False, manual_mode=False)
    # for some reason, nanoBragg phi rotation seems to propagate rounding error maybe, so we must use a larger atol...
    assert np.allclose(nb_pix, temp_nb, atol=0.1)
    print("tst manual vs internal nanoBragg: OK!")
    SIM.free_all()

def test_range_of_rotation_steps():
  # automatic param setting for phisteps and osc_deg when passed values are <0
  # conflicting params are tolerated and should be handled with the same decision tree
  testpdb = write_test_pdb(fileout="test.pdb")
  for phi_start in (0, 30):
    for osc_deg in (0., 0.01, 0.1):
      for phisteps in (1, 10, 15):
        pix1 = run_sim(
                testpdb,
                version="nanoBragg",
                spindle_axis=(1,0,0),
                phi_start=phi_start,
                osc_deg=osc_deg,
                phisteps=phisteps
                )#, fileout="rotimg_nano_%03d.h5")
        pix2 = run_sim(
                testpdb,
                version="diffBragg",
                spindle_axis=(1,0,0),
                phi_start=phi_start,
                osc_deg=osc_deg,
                phisteps=phisteps
                )#, fileout="rotimg_diff_%03d.h5")
        pix2 = pix2.reshape(pix1.shape)
        assert_identical_pixels(pix1, pix2)
        print("tst range: phisteps=%d, osc_deg=%f deg., phistart=%f deg. ... OK!" %(phisteps, osc_deg, phi_start))

def test_sweep():
  testpdb = write_test_pdb(fileout="test.pdb")
  for phi_start in range(10):
    pix1 = run_sim(
            testpdb,
            version="nanoBragg",
            spindle_axis=(1,0,0),
            phi_start=phi_start,
            osc_deg=0.05,
            phisteps=20
            )#, fileout="rotimg_nano_%03d.h5")
    pix2 = run_sim(
            testpdb,
            version="diffBragg",
            spindle_axis=(1,0,0),
            phi_start=phi_start,
            osc_deg=0.05,
            phisteps=20
            )#, fileout="rotimg_diff_%03d.h5")
    pix2 = pix2.reshape(pix1.shape)
    assert_identical_pixels(pix1, pix2)
    print("tst sweep: phisteps=20, osc_deg=0.05 deg., phistart=%f deg. ... OK!" % (phi_start, ))

def tst_all():
  test_manual_rotation()
  test_range_of_rotation_steps()
  test_sweep()

if __name__=="__main__":
    import sys
    if "--kokkos" in sys.argv:
        os.environ["DIFFBRAGG_USE_KOKKOS"] = "1"
    from simtbx.diffBragg.utils import find_diffBragg_instances
    from simtbx.diffBragg.device import DeviceWrapper

    with DeviceWrapper(0) as _:
        tst_all()
        for name in find_diffBragg_instances(globals()): del globals()[name]
    print("OK")
