from __future__ import division
from simtbx.diffBragg.utils import find_diffBragg_instances
import numpy as np

""""
CHECK NANOBRAGG, DIFFBRAGG, and EXAFEL+KOKKOS PRODUCE EQUIVALENT RESULTS
NOTE: Cannot --kokkos flag will skip the exafel API
"""

import sys
import libtbx.load_env
HAS_KOKKOS_EXAFEL = False
if libtbx.env.build_options.enable_kokkos and sys.platform.startswith('linux'):
    # the Exafel KOKKOS API:
    from simtbx.kokkos import gpu_energy_channels
    from simtbx.kokkos import exascale_api
    from simtbx.kokkos import gpu_detector as kokkosd
    HAS_KOKKOS_EXAFEL = True


def _test_exafel(S, reference, add_background=False):
    """
    # ONLY RUN THIS IF KOKKOS INSTANCE EXISTS
    :param S:  SimData instance (nanoBragg/sim_data.py)
    :param reference: reference 2D image from different API to test against
    :param add_background: whether or not to include background
    """
    kokkos_channels_singleton = gpu_energy_channels()
    inds, amps = S.D.Fhkl_tuple
    kokkos_channels_singleton.structure_factors_to_GPU_direct(0, inds, amps)
    kokkos_simulation = exascale_api(nanoBragg=S.D)
    kokkos_simulation.allocate()

    kokkos_detector = kokkosd(detector=S.detector, beam=S.D.xray_beams[0])
    kokkos_detector.each_image_allocate()

    # loop over energies
    kokkos_simulation.add_energy_channel_from_gpu_amplitudes(
        0, kokkos_channels_singleton, kokkos_detector)
    if add_background:
        kokkos_simulation.add_background(kokkos_detector)
    kokkos_detector.write_raw_pixels(S.D)
    kokkos_img = S.D.raw_pixels.as_numpy_array()
    if add_background:
        # note, why is there a difference in the images when adding a background?
        assert np.allclose(kokkos_img, reference, atol=1e-3, rtol=0)
    else:
        assert np.allclose(kokkos_img, reference)


def main(shape="gauss_star", background=True, test_exafel=False, nb_cuda=False):
    # TODO: test non-P space groups
    # TODO: test divergence model
    # TODO: test dispersion model

    from simtbx.nanoBragg.sim_data import SimData
    S = SimData(use_default_crystal=True)
    # Important, only add water OR air, but not both, because the exafel API will only look for one background model
    S.add_water = True
    S.add_air = False
    S.crystal.xtal_shape = shape
    S.crystal.mos_spread_deg = 1
    S.crystal.n_mos_domains = 20
    S.instantiate_diffBragg()
    S.D.nopolar = True
    S.D.oversample = 3
    S.D.spot_scale = 10

    S.D.add_diffBragg_spots()
    if background:
        S._add_background()

    diff_img = S.D.raw_pixels.as_numpy_array()

    S.D.raw_pixels *= 0
    if nb_cuda:
        S.D.add_nanoBragg_spots_cuda()
    else:
        S.D.add_nanoBragg_spots()
    if background:
        S._add_background()

    nano_img = S.D.raw_pixels.as_numpy_array()

    assert np.allclose(diff_img, nano_img, atol=1e-9)
    a = nano_img.mean()
    b = nano_img.max()
    c = nano_img.min()
    d = nano_img.std()
    e = np.median(nano_img)
    vals = a,b,c,d,e
    print("Reference image has pixel values Mean,Max,Min,Median,Std=%f,%f,%f,%f,%f" %vals)
    if HAS_KOKKOS_EXAFEL and test_exafel:
        S.D.raw_pixels *= 0
        _test_exafel(S, nano_img, background)
    else:
        print("No Kokkos GPU instance, skipping _test_exafel ... ")
    for name in find_diffBragg_instances(globals()): del globals()[name]


if __name__ == "__main__":
    import sys
    import os
    if "--kokkos" in sys.argv:
        # If we use this flag, then the deviceWrapper will instantiate KOKKOS instance
        os.environ["DIFFBRAGG_USE_KOKKOS"]="1"
    nb_cuda = "--cuda" in sys.argv

    from simtbx.diffBragg.device import DeviceWrapper
    with DeviceWrapper(0) as wrapper:
        for background in [False,True]:
            test_exafel = wrapper.is_using_kokkos_gpu
            main("gauss", background=background, test_exafel=test_exafel, nb_cuda=nb_cuda)
            main("gauss_star", background=background, test_exafel=test_exafel, nb_cuda=nb_cuda)
    print ("OK!")
