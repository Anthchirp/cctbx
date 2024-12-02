from __future__ import division
import os
import time
import logging
LOGGER = logging.getLogger("diffBragg.main")
import numpy as np
import pandas
from simtbx.nanoBragg.utils import flexBeam_sim_colors
from cctbx import miller
from cctbx.array_family import flex

try:
    from simtbx.gpu import gpu_energy_channels
except ImportError:
    gpu_energy_channels = None

from simtbx.diffBragg import utils
from dxtbx.model.experiment_list import ExperimentListFactory
multipanel_sim = None
try:
    from LS49.adse13_187.cyto_batch import multipanel_sim
except (ImportError, TypeError):
    pass
from simtbx.nanoBragg import utils as nanoBragg_utils
from simtbx.nanoBragg.nanoBragg_crystal import NBcrystal
from simtbx.nanoBragg.nanoBragg_beam import NBbeam
from simtbx.nanoBragg.sim_data import SimData


def panda_frame_from_exp(exp_name, detz_shift_mm=0, Ncells_abc=(30.,30.,30.), spot_scale=1., beamsize_mm=0.001,
                         total_flux=1e12, oversample=1,spectrum_fname=None, spectrum_stride=1, lam0=0, lam1=1):
    df = pandas.DataFrame({
        "opt_exp_name": [exp_name],
        "detz_shift_mm": [detz_shift_mm],
        "ncells": [Ncells_abc],
        "spot_scales": [spot_scale],
        "beamsize_mm": [beamsize_mm],
        "total_flux": [total_flux],
        "oversample": [oversample],
        "spectrum_filename": [spectrum_fname],
        "spectrum_stride": [spectrum_stride],
        "lam0": [lam0],
        "lam1": [lam1]})

    return df


def model_from_expt(exp_name,  model_spots_from_pandas_kwargs=None, panda_frame_from_exp_kwargs=None):
    if model_spots_from_pandas_kwargs is None:
        model_spots_from_pandas_kwargs = {}
    if panda_frame_from_exp_kwargs is None:
        panda_frame_from_exp_kwargs = {}
    df = panda_frame_from_exp(exp_name, **panda_frame_from_exp_kwargs)
    out = model_spots_from_pandas(df, **model_spots_from_pandas_kwargs)
    return out


# TODO name change
# TODO move all these arguments into the pandas dataFrame (mtz_col, from_pdb etc)
def model_spots_from_pandas(pandas_frame,  rois_per_panel=None,
                          mtz_file=None, mtz_col=None,
                          oversample_override=None,
                          Ncells_abc_override=None,
                          pink_stride_override=None,
                          spectrum_override=None,
                          cuda=False, device_Id=0, time_panels=False,
                          d_max=999, d_min=1.5, defaultF=1e3,
                          omp=False,
                          norm_by_spectrum=False,
                          symbol_override=None, quiet=False, reset_Bmatrix=False, nopolar=False,
                          force_no_detector_thickness=False, printout_pix=None, norm_by_nsource=False,
                          use_exascale_api=False, use_db=False, show_timings=False, perpixel_wavelen=False,
                          det_thicksteps=None, from_pdb=None, mosaic_samples_override=None):
    if perpixel_wavelen and not use_db:
        raise NotImplementedError("to get perpixel wavelengths set use_db=True to use the diffBragg backend")
    if use_exascale_api:
        assert gpu_energy_channels is not None, "cant use exascale api if not in a GPU build"
        assert multipanel_sim is not None, "cant use exascale api if LS49: https://github.com/nksauter/LS49.git  is not configured\n install in the modules folder"

    df = pandas_frame.reset_index(drop=True)

    if not quiet: LOGGER.info("Loading experiment models")
    expt_name = df.exp_name.values[0]
    El = ExperimentListFactory.from_json_file(expt_name, check_format=False)
    exp_idx = 0
    if "exp_idx" in list(df):
        exp_idx = int(df.exp_idx.values[0])  # cast to int because has to be 32-bit
    expt = El[exp_idx]
    crystal = expt.crystal
    crystal.set_A(df.Amats.values[0])
    expt.crystal = crystal
    columns = list(df)
    if "detz_shift_mm" in columns:  # NOTE, this could also be inside expt_name directly
        expt.detector = utils.shift_panelZ(expt.detector, df.detz_shift_mm.values[0])

    if force_no_detector_thickness:
        expt.detector = utils.strip_thickness_from_detector(expt.detector)
    if reset_Bmatrix:
        ucell_params = df[["a", "b", "c", "al", "be", "ga"]].values[0]
        ucell_man = utils.manager_from_params(ucell_params)
        expt.crystal.set_B(ucell_man.B_recipspace)
    assert len(df) == 1
    Ncells_abc = df.ncells.values[0]
    if Ncells_abc_override is not None:
        Ncells_abc = Ncells_abc_override
    Ncells_def = df.ncells_def.values[0]
    spot_scale = df.spot_scales.values[0]
    beamsize_mm = df.beamsize_mm.values[0]
    total_flux = df.total_flux.values[0]
    oversample = df.oversample.values[0]
    if oversample_override is not None:
        oversample = oversample_override

    # get the optimized spectra
    if spectrum_override is None:
        if "spectrum_filename" in list(df) and df.spectrum_filename.values[0] is not None:
            spectrum_file = df.spectrum_filename.values[0]
            pink_stride = df.spectrum_stride.values[0]
            if norm_by_spectrum:
                nspec = utils.load_spectra_file(spectrum_file)[0].shape[0]
                spot_scale = spot_scale / nspec
            if pink_stride_override is not None:
                pink_stride = pink_stride_override
            fluxes, energies = utils.load_spectra_file(spectrum_file, total_flux=total_flux,
                                                       pinkstride=pink_stride)
        else:
            fluxes = np.array([total_flux])
            energies = np.array([utils.ENERGY_CONV/expt.beam.get_wavelength()])
            if not quiet: LOGGER.info("Running MONO sim")

    else:
        wavelens, fluxes = map(np.array, zip(*spectrum_override))
        energies = utils.ENERGY_CONV / wavelens

    lam0 = df.lam0.values[0]
    lam1 = df.lam1.values[0]
    if lam0 == -1 or np.isnan(lam0):
        lam0 = 0
    if lam1 == -1 or np.isnan(lam1):
        lam1 = 1
    wavelens = utils.ENERGY_CONV / energies
    wavelens = lam0 + lam1*wavelens
    energies = utils.ENERGY_CONV / wavelens

    if mtz_file is not None:
        assert mtz_col is not None
        Famp = utils.open_mtz(mtz_file, mtz_col)
        Famp = miller.array(Famp.set(), data=flex.double(len(Famp.data()), np.mean(Famp.data())))
    elif from_pdb is not None and from_pdb.name is not None:
        wavelength=None
        if from_pdb.add_anom:
            wavelength = expt.beam.get_wavelength()
        miller_data = utils.get_complex_fcalc_from_pdb(from_pdb.name,
                                                 dmin=d_min,
                                                 dmax=d_max,
                                                 wavelength=wavelength,
                                                 k_sol=from_pdb.k_sol,
                                                 b_sol=from_pdb.b_sol)
        Famp = miller_data.as_amplitude_array()
    else:
        Famp = utils.make_miller_array_from_crystal(expt.crystal, dmin=d_min, dmax=d_max, defaultF=defaultF, symbol=symbol_override)

    diffuse_params = None
    if "use_diffuse_models" in columns and df.use_diffuse_models.values[0]:
        if not use_db:
            raise RuntimeError("Cant simulate diffuse models unless use_db=True (diffBragg modeler)")
        diffuse_params = {"gamma": tuple(df.diffuse_gamma.values[0]),
                          "sigma": tuple(df.diffuse_sigma.values[0]),
                          "gamma_miller_units": False}
        if "gamma_miller_units" in list(df):
            diffuse_params["gamma_miller_units"] = df.gamma_miller_units.values[0]


    if use_exascale_api:
        #===================
        gpu_channels_singleton = gpu_energy_channels(deviceId=0)
        print(gpu_channels_singleton.get_deviceID(), "device")
        from simtbx.nanoBragg import nanoBragg_crystal
        C = nanoBragg_crystal.NBcrystal(init_defaults=False)
        C.miller_array = Famp
        F_P1 = C.miller_array
        F_P1 = Famp.expand_to_p1()
        gpu_channels_singleton.structure_factors_to_GPU_direct(0, F_P1.indices(), F_P1.data())
        Famp = gpu_channels_singleton
        #===========
        results,_,_ = multipanel_sim(CRYSTAL=expt.crystal,
                                 DETECTOR=expt.detector,
                                 BEAM=expt.beam, Famp=Famp,
                                 energies=energies, fluxes=fluxes, Ncells_abc=Ncells_abc,
                                 beamsize_mm=beamsize_mm, oversample=oversample,
                                 spot_scale_override=spot_scale, default_F=0, interpolate=0,
                                 include_background=False,
                                 profile="gauss", cuda=True, show_params=False)
        return results, expt
    elif use_db:
        mos_dom = 1
        if "num_mosaicity_samples" in list(df):
            mos_dom = int(df.num_mosaicity_samples.values[0])
        if mosaic_samples_override is not None:
            mos_dom = mosaic_samples_override
        eta_abc = df.eta_abc.values[0]
        LOGGER.debug("Num mos samples=%d, eta_abc=%f %f %f" % ((mos_dom,)+ eta_abc )  )
        LOGGER.debug("Num energy channels=%d" % len(energies))
        results = diffBragg_forward(CRYSTAL=expt.crystal, DETECTOR=expt.detector, BEAM=expt.beam, Famp=Famp,
                                    fluxes=fluxes, energies=energies, beamsize_mm=beamsize_mm,
                                    Ncells_abc=Ncells_abc, spot_scale_override=spot_scale,
                                    mos_dom=mos_dom, eta_abc=df.eta_abc.values[0],
                                    default_F=np.mean(Famp.data()),
                                    device_Id=device_Id, oversample=oversample,
                                    show_params=not quiet,
                                    nopolar=nopolar,
                                    printout_pix=printout_pix,
                                    diffuse_params=diffuse_params, cuda=cuda,
                                    show_timings=show_timings,
                                    perpixel_wavelen=perpixel_wavelen,
                                    det_thicksteps=det_thicksteps, Ncells_def=Ncells_def)
        return results, expt

    else:
        pids = None
        if rois_per_panel is not None:
            pids = list(rois_per_panel.keys()),
        results = flexBeam_sim_colors(CRYSTAL=expt.crystal, DETECTOR=expt.detector, BEAM=expt.beam, Famp=Famp,
                                      fluxes=fluxes, energies=energies, beamsize_mm=beamsize_mm,
                                      Ncells_abc=Ncells_abc, spot_scale_override=spot_scale,
                                      cuda=cuda, device_Id=device_Id, oversample=oversample,
                                      time_panels=time_panels and not quiet,
                                      pids=pids,
                                      rois_perpanel=rois_per_panel,
                                      omp=omp, show_params=not quiet,
                                      nopolar=nopolar,
                                      printout_pix=printout_pix)
        if norm_by_nsource:
            return np.array([image/len(energies) for _,image in results]), expt
        else:
            return np.array([image for _,image in results]), expt


def diffBragg_forward(CRYSTAL, DETECTOR, BEAM, Famp, energies, fluxes,
                      oversample=0, Ncells_abc=(50, 50, 50),
                      mos_dom=1, mos_spread=0, beamsize_mm=0.001, device_Id=0,
                      show_params=True, crystal_size_mm=None, printout_pix=None,
                      verbose=0, default_F=0, interpolate=0, profile="gauss",
                      spot_scale_override=None,
                      mosaicity_random_seeds=None,
                      nopolar=False, diffuse_params=None, cuda=False,
                      show_timings=False,perpixel_wavelen=False,
                      det_thicksteps=None, eta_abc=None, Ncells_def=None,
                      num_phi_steps=1, delta_phi=None, div_mrad=0, divsteps=0,
                      spindle_axis=(1,0,0)):

    if cuda:
        os.environ["DIFFBRAGG_USE_CUDA"] = "1"
    CRYSTAL, Famp = nanoBragg_utils.ensure_p1(CRYSTAL, Famp)

    nbBeam = NBbeam()
    nbBeam.divergence_mrad = div_mrad  # / 1e3 * 180 / np.pi
    nbBeam.divsteps = divsteps
    nbBeam.size_mm = beamsize_mm
    nbBeam.unit_s0 = BEAM.get_unit_s0()
    wavelengths = utils.ENERGY_CONV / np.array(energies)
    nbBeam.spectrum = list(zip(wavelengths, fluxes))

    nbCrystal = NBcrystal(init_defaults=False)
    nbCrystal.isotropic_ncells = False
    nbCrystal.dxtbx_crystal = CRYSTAL
    nbCrystal.miller_array = Famp
    nbCrystal.Ncells_abc = Ncells_abc
    nbCrystal.Ncells_def = Ncells_def
    nbCrystal.symbol = CRYSTAL.get_space_group().info().type().lookup_symbol()
    nbCrystal.thick_mm = crystal_size_mm
    nbCrystal.xtal_shape = profile
    nbCrystal.n_mos_domains = mos_dom
    nbCrystal.mos_spread_deg = mos_spread
    if eta_abc is not None:
        nbCrystal.anisotropic_mos_spread_deg = eta_abc

    S = SimData()
    S.detector = DETECTOR
    npan = len(DETECTOR)
    nfast, nslow = DETECTOR[0].get_image_size()
    img_shape = npan, nslow, nfast
    S.beam = nbBeam
    S.crystal = nbCrystal
    if mosaicity_random_seeds is not None:
        S.mosaic_seeds = mosaicity_random_seeds

    S.instantiate_diffBragg(verbose=verbose, oversample=oversample, interpolate=interpolate, device_Id=device_Id,
                            default_F=default_F,
                            auto_set_spotscale=crystal_size_mm is not None and spot_scale_override is None)

    if spot_scale_override is not None:
        S.update_nanoBragg_instance("spot_scale", spot_scale_override)
    S.update_nanoBragg_instance("nopolar", nopolar)
    if det_thicksteps is not None:
        S.update_nanoBragg_instance(
            "detector_thicksteps", det_thicksteps)

    if show_params:
        S.D.show_params()
        print("Spot scale=%f" % S.D.spot_scale)

    if show_timings and verbose < 2:
        S.D.verbose = 2
    S.D.store_ave_wavelength_image = perpixel_wavelen
    S.D.record_time = True
    if diffuse_params is not None:
        S.D.use_diffuse = True
        S.D.gamma_miller_units = diffuse_params["gamma_miller_units"]
        S.D.diffuse_gamma = diffuse_params["gamma"]
        S.D.diffuse_sigma = diffuse_params["sigma"]

    if delta_phi is not None:
        utils.update_SIM_with_gonio(S, delta_phi=delta_phi, num_phi_steps=num_phi_steps, spindle_axis=spindle_axis)
    S.D.add_diffBragg_spots_full()
    if show_timings or LOGGER.level <= 10:
        S.D.show_timings()
    t = time.time()
    data = S.D.raw_pixels_roi.as_numpy_array().reshape(img_shape)
    if perpixel_wavelen:
        wavelen_data = S.D.ave_wavelength_image().as_numpy_array().reshape(img_shape)
        hdata,kdata,ldata = map(lambda x:x.as_numpy_array().reshape(img_shape), S.D.ave_hkl_image())

    t = time.time() - t
    if show_timings:
        print("Took %f sec to recast and reshape" % t)
    if printout_pix is not None:
        S.D.raw_pixels_roi*=0
        p,f,s = printout_pix
        S.D.printout_pixel_fastslow = f,s
        S.D.show_params()
        S.D.add_diffBragg_spots(printout_pix)

    # free up memory
    S.D.free_all()
    S.D.free_Fhkl2()
    if S.D.gpu_free is not None:
        S.D.gpu_free()
    if perpixel_wavelen:
        return data, wavelen_data, hdata, kdata, ldata
    else:
        return data


if __name__ == "__main__":
    import sys
    from simtbx.diffBragg import hopper_utils
    df = pandas.read_pickle(sys.argv[1])
    max_prc = 1  #@int(sys.argv[2])
    exp_names = df.exp_name.unique()
    use_exa = True
    PFS = 40, 222, 192
    oo = 3 #None
    PFS = None
    CUDA= True

    if os.environ.get("DIFFBRAGG_USE_CUDA") is None:
        CUDA = False
    for exp_name in exp_names:
        df_exp = df.query("exp_name=='%s'" % exp_name)
        E = ExperimentListFactory.from_json_file(df_exp.exp_name.values[0])[0]
        spectrum = hopper_utils.spectrum_from_expt(E, 1e12)

        t = time.time()
        imgs,_ = model_spots_from_pandas(df_exp, force_no_detector_thickness=True, use_db=True, quiet=True,
                                     oversample_override=oo, nopolar=True, spectrum_override=spectrum)
        t = time.time() - t
        print("----------------------")

        t2 = time.time()
        imgs2,_ = model_spots_from_pandas(df_exp, force_no_detector_thickness=True, cuda=CUDA, quiet=True,
                                      oversample_override=oo, nopolar=True, spectrum_override=spectrum)#, oversample_override=1, nopolar=True, printout_pix=PFS)
        t2 = time.time()-t2
        print("----------------------")
        if np.allclose(imgs, imgs2):
            print("OK1")
        else:
            print("NOPE1")
        if not CUDA:
            exit()
        t3 = time.time()
        imgs3,_ = model_spots_from_pandas(df_exp, use_exascale_api=True, force_no_detector_thickness=True, quiet=True,
                                          oversample_override=oo, nopolar=True, spectrum_override=spectrum)
        t3 = time.time()-t3
        if np.allclose(imgs, imgs3):
            print("OK2")
        else:
            print("NOPE2")
        print("<><><><><><><><><><><><><><><><>")
        print("--------     RESULTS     -------")
        print("<><><><><><><><><><><><><><><><>")
        print("\t\tdiffBragg: %f" % t)
        print("\t\tlegacy: %f" % t2)
        print("\t\texascale: %f" % t3)

        exit()
