from __future__ import absolute_import, division, print_function
import os
import socket
import sys
import re
from io import StringIO
import numpy as np
from scipy import fft
import pickle
from scipy.optimize import minimize
from scipy.ndimage import generate_binary_structure, maximum_filter, binary_erosion
from cctbx.array_family import flex
from cctbx import miller, sgtbx
from cctbx.crystal import symmetry
from dxtbx.model import Detector, Panel
from simtbx.nanoBragg.sim_data import SimData
from simtbx.nanoBragg.nanoBragg_beam import NBbeam
from simtbx.nanoBragg.nanoBragg_crystal import NBcrystal
from simtbx.diffBragg import phil
from simtbx.nanoBragg.utils import ENERGY_CONV
from simtbx.diffBragg.refiners.crystal_systems import OrthorhombicManager, TetragonalManager, MonoclinicManager, HexagonalManager
from dxtbx.imageset import MemReader
from dxtbx.imageset import ImageSet, ImageSetData
from dxtbx.model.experiment_list import ExperimentListFactory
import libtbx
from libtbx.phil import parse
from dials.array_family import flex as dials_flex
import mmtbx.programs.fmodel
import mmtbx.utils
from cctbx.eltbx import henke
from simtbx.diffBragg import psf
from dials.algorithms.shoebox import MaskCode
from xfel.merging.application.utils.memory_usage import get_memory_usage


import logging
MAIN_LOGGER = logging.getLogger("diffBragg.main")


def strong_spot_mask(refl_tbl, detector, as_composite=True):
    """
    Form an image of the strong spot masks (True indicates strong spot)
    :param refl_tbl: dials reflection table with shoeboxes
    :param detector: dxtbx detecto model
    :param as_composite: return a single mask same shape as detector, else return shoebox masks as a lst
    """
    Nrefl = len( refl_tbl)
    masks = [ refl_tbl[i]['shoebox'].mask.as_numpy_array()
              for i in range(Nrefl)]
    pids = refl_tbl['panel']
    nfast, nslow = detector[0].get_image_size()
    npan = len(detector)
    code = MaskCode.Foreground.real

    x1, x2, y1, y2, z1, z2 = zip(*[refl_tbl[i]['shoebox'].bbox
                                   for i in range(Nrefl)])
    if not as_composite:
        spot_masks = []
    spot_mask = np.zeros((npan, nslow, nfast), bool)
    for i_ref, (i1, i2, j1, j2, M) in enumerate(zip(x1, x2, y1, y2, masks)):

        slcX = slice(i1, i2, 1)
        slcY = slice(j1, j2, 1)
        spot_mask[pids[i_ref],slcY, slcX] = M & code == code
        if not as_composite:
            spot_masks.append(spot_mask.copy())
            spot_mask *= False
    if as_composite:
        return spot_mask
    else:
        return spot_masks


def label_background_pixels(roi_img, thresh=3.5, iterations=1, only_high=True):
    """
    roi_img, a two dimensional pixel image numpy array
    thresh, median deviation Z-score; pixels with Z-score below thresh are flagged as background
    iterations, lots of bright pixels can skew the stats, so iteratively label background pixels
              and recompute Z-scores
    """
    img_shape = roi_img.shape
    img1 = roi_img.copy().ravel()   # 1-D version
    background_pixels = None
    while iterations > 0:
        if background_pixels is None:
            outliers = is_outlier(img1, thresh)
            inlier_vals = img1[~outliers]
            if inlier_vals.size:
                m = np.median(inlier_vals)
            else:
                m = np.nan
            if only_high:
                outliers = np.logical_and(outliers, img1 > m)
            background_pixels = ~outliers
        else:
            where_bg = np.where(background_pixels)[0]
            outliers = is_outlier(img1[background_pixels], thresh)
            inlier_vals = img1[background_pixels][~outliers]
            if inlier_vals.size:
                m = np.median(inlier_vals)
            else:
                m = np.nan
            if only_high:
                outliers = np.logical_and(outliers, img1[background_pixels] > m)
            background_pixels[where_bg[outliers]] = False
        iterations = iterations - 1

    return background_pixels.reshape(img_shape)


def is_outlier(points, thresh=3.5):
    """http://stackoverflow.com/a/22357811/2077270"""
    if len(points.shape) == 1:
        points = points[:, None]
    if points.size:
        median = np.median(points, axis=0)
    else:
        median = np.nan
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    if diff.size:
        med_abs_deviation = np.median(diff)
    else:
        med_abs_deviation = np.nan
    if med_abs_deviation == 0:
        return np.zeros(points.shape[0], bool)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def get_diffBragg_instance():
    """
    Simple method to get a diffBRagg instance
    USED IN TESTING
    Returns an instance of diffBragg
    """
    from dxtbx.model.crystal import CrystalFactory
    from dxtbx.model.detector import DetectorFactory
    from dxtbx.model.beam import BeamFactory
    from simtbx.nanoBragg.tst_nanoBragg_basic import fcalc_from_pdb
    from simtbx.nanoBragg import shapetype
    from simtbx.diffBragg import diffBragg

    wavelen = 1.24
    flux = 1e15
    SHAPE = shapetype.Gauss

    NCELLS_ABC = 15  # means (15, 15, 15)

    beam_descr = {'direction': (0.0, 0.0, 1.0),
                  'divergence': 0.0,
                  'flux': 5e11,
                  'polarization_fraction': 1.,
                  'polarization_normal': (0.0, 1.0, 0.0),
                  'sigma_divergence': 0.0,
                  'transmission': 1.0,
                  'wavelength': wavelen}

    cryst_descr = {'__id__': 'crystal',
                   'real_space_a': (50, 0, 0),
                   'real_space_b': (0, 60, 0),
                   'real_space_c': (0, 0, 70),
                   'space_group_hall_symbol': '-P 4 2'}

    det_descr = {'panels':
                     [{'fast_axis': (-1.0, 0.0, 0.0),
                       'gain': 1.0,
                       'identifier': '',
                       'image_size': (196, 196),
                       'mask': [],
                       'material': '',
                       'mu': 0.0,
                       'name': 'Panel',
                       'origin': (19.6, -19.6, -550),
                       'pedestal': 0.0,
                       'pixel_size': (0.1, 0.1),
                       'px_mm_strategy': {'type': 'SimplePxMmStrategy'},
                       'raw_image_offset': (0, 0),
                       'slow_axis': (0.0, 1.0, 0.0),
                       'thickness': 0.0,
                       'trusted_range': (0.0, 65536.0),
                       'type': ''}]}

    DET = DetectorFactory.from_dict(det_descr)
    BEAM = BeamFactory.from_dict(beam_descr)
    crystal = CrystalFactory.from_dict(cryst_descr)

    Fhkl = fcalc_from_pdb(resolution=4, algorithm="fft", wavelength=wavelen)

    D = diffBragg(DET, BEAM, verbose=0)
    D.xtal_shape = SHAPE
    D.Ncells_abc = NCELLS_ABC
    D.wavelength_A = wavelen
    D.flux = flux
    D.mosaic_spread_deg = 0.01
    D.mosaic_domains = 10
    from simtbx.nanoBragg.nanoBragg_crystal import NBcrystal
    braggC = NBcrystal()
    braggC.miller_array = Fhkl
    idx = braggC.miller_array.indices()
    amps = braggC.miller_array.data()
    D.Bmatrix = crystal.get_B()
    D.Umatrix = crystal.get_U()
    D.Fhkl_tuple = idx, amps, None
    D.Bmatrix = crystal.get_B()
    D.Umatrix = crystal.get_U()
    return D


def map_hkl_list(Hi_lst, anomalous_flag=True, symbol="P43212"):
    """
    :param Hi_lst: list of miller indices, presumably in P1
    :param anomalous_flag: whether to map to anomalous ASU
    :param symbol: space group symbol
    :return: list of miller indices, mapped to HKL
    """
    sg_type = sgtbx.space_group_info(symbol=symbol).type()
    # necessary for py3 to type cast the ints
    type_casted_Hi_lst = tuple([(int(x), int(y), int(z)) for x, y, z in Hi_lst])
    Hi_flex = dials_flex.miller_index(type_casted_Hi_lst)
    miller.map_to_asu(sg_type, anomalous_flag, Hi_flex)
    return list(Hi_flex)


def compare_with_ground_truth(a, b, c, dxcryst_models, symbol="C121", verbose=False):
    """
    ported from LS49
    :param a: ground truth a (real space)
    :param b: ground truth b (real space)
    :param c: gournd truth c (real space)
    :param dxcryst_models: P1 models or whatever
    :param symbol: target space group
    :return: list of angles, one for each model
    """
    from cctbx import sgtbx
    from dxtbx.model import MosaicCrystalSauter2014
    from cctbx import crystal_orientation
    from libtbx.test_utils import approx_equal
    from scitbx.array_family import flex
    from scitbx.matrix import sqr

    sgi = sgtbx.space_group_info(symbol=symbol)
    CB_OP_C_P = sgi.change_of_basis_op_to_primitive_setting()

    icount = 0
    angles = []
    icount += 1
    for crystal_model in dxcryst_models:
        if crystal_model.get_space_group().info().type().lookup_symbol() == "P 1":
            crystal_model = crystal_model.change_basis(CB_OP_C_P.inverse())
        mosaic_model = MosaicCrystalSauter2014(crystal_model)

        direct_A = mosaic_model.get_A_inverse_as_sqr()
        sim_compatible = direct_A
        integrated_Ori = crystal_orientation.crystal_orientation(
            sim_compatible,
            crystal_orientation.basis_type.direct)

        header_Ori = crystal_orientation.crystal_orientation(
            tuple(a)+tuple(b)+tuple(c),
            crystal_orientation.basis_type.direct)

        C2_ground_truth = header_Ori.change_basis(CB_OP_C_P.inverse())
        if verbose:
            C2_ground_truth.show(legend="C2_ground_truth")

        # align integrated model with ground truth
        cb_op_align = integrated_Ori.best_similarity_transformation(C2_ground_truth, 50, 1)
        aligned_Ori = integrated_Ori.change_basis(sqr(cb_op_align))
        if verbose:
            aligned_Ori.show(legend="integrated, aligned")
            MAIN_LOGGER.debug("alignment matrix", cb_op_align)

        U_integrated = aligned_Ori.get_U_as_sqr()
        U_ground_truth = C2_ground_truth.get_U_as_sqr()
        missetting_rot = U_integrated * U_ground_truth.inverse()
        assert approx_equal(missetting_rot.determinant(), 1.0)

        # now calculate the angle as mean a_to_a,b_to_b,c_to_c
        aoff = aligned_Ori.a.angle(C2_ground_truth.a, deg=True)
        boff = aligned_Ori.b.angle(C2_ground_truth.b, deg=True)
        coff = aligned_Ori.c.angle(C2_ground_truth.c, deg=True)

        hyp = flex.mean(flex.double((aoff, boff, coff)))
        angles.append(hyp)

    return angles


def fcalc_from_pdb(resolution, algorithm=None, wavelength=0.9, anom=True, ucell=None, symbol=None, as_amplitudes=True):
    # borrowed from tst_nanoBragg_basic
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
    import iotbx.pdb
    pdb_inp = iotbx.pdb.input(source_info=None, lines=pdb_lines)
    xray_structure = pdb_inp.xray_structure_simple()
    if ucell is not None:
        assert symbol is not None
        from cctbx.xray import structure
        from cctbx import crystal
        crystal_sym = crystal.symmetry(unit_cell=ucell, space_group_symbol=symbol)
        xray_structure = structure(scatterers=xray_structure.scatterers(), crystal_symmetry=crystal_sym)
    # take a detour to insist on calculating anomalous contribution of every atom
    scatterers = xray_structure.scatterers()
    if anom:
        from cctbx.eltbx import henke
        for sc in scatterers:
            expected_henke = henke.table(sc.element_symbol()).at_angstrom(wavelength)
            sc.fp = expected_henke.fp()
            sc.fdp = expected_henke.fdp()
    # how do we do bulk solvent?
    primitive_xray_structure = xray_structure.primitive_setting()
    P1_primitive_xray_structure = primitive_xray_structure.expand_to_p1()
    fcalc = P1_primitive_xray_structure.structure_factors(
        d_min=resolution, anomalous_flag=anom, algorithm=algorithm).f_calc()
    if as_amplitudes:
        fcalc = fcalc.amplitudes().set_observation_type_xray_amplitude()
    return fcalc


def get_roi_from_spot(refls, fdim, sdim, shoebox_sz=10, centroid='obs'):
    """

    :param refls: reflection table
    :param fdim: fast axis dimension
    :param sdim: slow axis dimension
    :param shoebox_sz: size of the shoeboxes
    :param centroid: either `obs` or  `cal`. correspinds to refl column `xyzobs.px.value` or `xyzxcal.px`, respectively
    :return:
    """
    if centroid=='obs':
        fs_spot, ss_spot, _ = zip(*refls['xyzobs.px.value'])
    elif centroid=='cal':
        fs_spot, ss_spot, _ = zip(*refls['xyzcal.px'])
    else:
        raise NotImplementedError("No instruction to get centroid position from %s" % centroid)
    rois = []
    is_on_edge = []
    for i_spot, (x_com, y_com) in enumerate(zip(fs_spot, ss_spot)):
        x_com = x_com - 0.5
        y_com = y_com - 0.5
        i1 = int(max(x_com - shoebox_sz / 2., 0))
        i2 = int(min(x_com + shoebox_sz / 2., fdim-1))
        j1 = int(max(y_com - shoebox_sz / 2., 0))
        j2 = int(min(y_com + shoebox_sz / 2., sdim-1))

        if i2-i1 < shoebox_sz-1 or j2-j1 < shoebox_sz-1:
            is_on_edge.append(True)
        else:
            is_on_edge.append(False)
        roi = i1, i2, j1, j2
        rois.append(roi)
    return rois, is_on_edge


def add_rlp_column(refls, experiment):
    """
    add Relps to refl tabl
    :param refls: reflection table
    :param experiment: dxtbx experiment
    :return:
    """
    keys = list(refls[0].keys())
    if "s1" in keys:
        s1 = refls['s1']
        s1_norm = np.array(s1) / np.linalg.norm(s1,axis=1)[:,None]
        wavelen = experiment.beam.get_wavelength()
        s0 = np.array([list(experiment.beam.get_unit_s0())]*len(refls))
        q_vecs = 1/wavelen*(s1_norm-s0)
        refls['rlp'] = flex.vec3_double(tuple(map(tuple, q_vecs)))
    else:
        raise KeyError("Need rlp or s1 column in refl table!")


def get_roi_deltaQ(refls, delta_Q, experiment, centroid='obs'):
    """
    :param refls: reflection table (needs rlp column)
    :param delta_Q:  width of the ROI in inverse Angstromg (e.g. 0.05)
    :param experiment:
    :param centroid: flag, obs, cal, or bbox
    :return:
    """
    nref = len(refls)
    assert nref >0

    keys = list(refls[0].keys())
    if "rlp" not in  keys:
        add_rlp_column(refls, experiment)

    beam = experiment.beam
    detector = experiment.detector
    assert beam is not None and experiment is not None

    rois = []
    is_on_edge = []
    for i_refl in range(nref):
        roi, on_edge = determine_shoebox_ROI(detector, delta_Q, beam.get_wavelength(), refls[i_refl], centroid=centroid)
        rois.append(roi)
        is_on_edge.append( on_edge)
    return rois, is_on_edge


# TODO: pass params object directly to this method
def get_roi_background_and_selection_flags(refls, imgs, shoebox_sz=10, reject_edge_reflections=False,
                                   reject_roi_with_hotpix=True, background_mask=None, hotpix_mask=None,
                                   bg_thresh=3.5, set_negative_bg_to_zero=False,
                                   pad_for_background_estimation=None, use_robust_estimation=True, sigma_rdout=3.,
                                   min_trusted_pix_per_roi=4, deltaQ=None, experiment=None, weighted_fit=True,
                                   ret_cov=False, allow_overlaps=False, skip_roi_with_negative_bg=True,
                                   only_high=True, centroid='obs'):
    """

    :param refls: reflection table
    :param imgs: ndimage array, same shape as detector
    :param shoebox_sz:  size of shoeboxes, deltaQ (see below) overrides this
    :param reject_edge_reflections: reject the refl if its centroid is close to the edge
    :param reject_roi_with_hotpix: reject the refl if ROI contains a hot pixel
    :param background_mask: mask specifying which pixels are most likely background (e.g. strong spot pixels should be False)
    :param hotpix_mask: mask labeling hot pixels as True
    :param bg_thresh: median absolute deviation threshold for background selection (e.g. ROI pixels with MAD above this value are treated as hot or strong)
    :param set_negative_bg_to_zero: after fitting background plane, force negative background signals to be 0
    :param pad_for_background_estimation: expand the ROI for the purposes of background fitting, once background is fit, remove expanded pixels
    :param use_robust_estimation: let ROI median value be the background
    :param sigma_rdout: readout noise of the pixel (in ADU)
    :param min_trusted_pix_per_roi: require at least this many trusted pixels in the ROI, otherwise flag refl as unselected
    :param deltaQ: specify the width of the ROI in inverse angstrom (overrides shoebox_sz), such that higher Q ROIS are larger
    :param experiment: dxtbx experiment
    :param weighted_fit: fit a tilt plane with basis error models as weights
    :param ret_cov: return the tilt plane covariance
    :param allow_overlaps: allow overlapping ROIS, otherwise shrink ROIS until the no longer overlap
    :param skip_roi_with_negative_bg: if an ROI has negative signal, dont include it in refinement
    :param only_high: only filter zingers that are above the mean (default is True)
    :param centroid: obs or cal (get centroids from refl column xyzobs.px.value or xyzcal.px)
    :return:
    """


    # TODO handle divide by 0 warning that happens in is_outlier, when labeling background pix?
    npan, sdim, fdim = imgs.shape

    if hotpix_mask is not None:
        assert hotpix_mask.shape == imgs.shape

    if background_mask is not None:
        assert background_mask.shape == imgs.shape

    if deltaQ is None:  # then use fixed size ROIS determined by shoebox_sz
        rois, is_on_edge = get_roi_from_spot(refls, fdim, sdim, shoebox_sz=shoebox_sz, centroid=centroid)
    else:
        assert experiment is not None
        if len(refls) == 0:
            return
        rois, is_on_edge = get_roi_deltaQ(refls, deltaQ, experiment, centroid=centroid)

    tilt_abc = []
    kept_rois = []
    panel_ids = []
    all_cov =[]
    selection_flags = []
    num_roi_negative_bg = 0
    num_roi_nan_bg = 0
    background = np.full_like(imgs, -1, dtype=float)
    i_roi = 0
    while i_roi < len(rois):
        roi = rois[i_roi]
        i1, i2, j1, j2 = roi
        is_selected = True
        refl_bbox_str = "Reflection %d bounded by x1=%d,x2=%d,y1=%d,y2=%d" % (i_roi, i1,i2,j1,j2)
        if is_on_edge[i_roi] and reject_edge_reflections:
            MAIN_LOGGER.debug("Reflection %d is on edge" % i_roi)
            is_selected = False
        pid = refls[i_roi]['panel']

        if hotpix_mask is not None:
            is_hotpix = hotpix_mask[pid, j1:j2, i1:i2]
            num_hotpix = is_hotpix.sum()
            if num_hotpix > 0 and reject_roi_with_hotpix:
                MAIN_LOGGER.debug("reflection %d has hot pixel" % i_roi)
                is_selected = False
            if num_hotpix > min_trusted_pix_per_roi:
                MAIN_LOGGER.debug("reflection %d has too many (%d) hot pixels (%d allowed)!" % (i_roi, num_hotpix, min_trusted_pix_per_roi))
                is_selected = False

        # Before padding and fitting, test for overlaps and shrink if needed
        is_overlapping = not np.all(background[pid, j1:j2, i1:i2] == -1)
        if not allow_overlaps and is_overlapping:
            MAIN_LOGGER.debug("region of interest already accounted for roi size= %d %d" % (i2-i1, j2-j1))
            rois[i_roi] = (i1 + 1, i2, j1 + 1, j2) if (i1 + i2) % 2 \
                else (i1, i2 - 1, j1, j2 - 1)  # shrink alternately from corners
            continue

        dimY, dimX = imgs[pid].shape
        j1_nopad = j1
        i1_nopad = i1
        j2_nopad = j2
        i2_nopad = i2
        if pad_for_background_estimation is not None:
            j1 = max(0, j1-pad_for_background_estimation)
            i1 = max(0, i1-pad_for_background_estimation)
            j2 = min(dimY, j2+pad_for_background_estimation)
            i2 = min(dimX, i2+pad_for_background_estimation)

        shoebox = imgs[pid, j1:j2, i1:i2]

        if shoebox.size < 4:
            MAIN_LOGGER.debug("reflection %d has negative background" % i_roi)
            is_selected = False

        if not isinstance(sigma_rdout, float) and not isinstance(sigma_rdout, int):
            shoebox_sigma_readout = sigma_rdout[pid, j1:j2, i1:i2]
        else:
            shoebox_sigma_readout = sigma_rdout

        if background_mask is not None:
            is_background = background_mask[pid, j1:j2, i1:i2]
        else:
            if shoebox.shape== (0,0):
                is_background = shoebox.copy().astype(bool)
            else:
                is_background = label_background_pixels(shoebox,thresh=bg_thresh, iterations=2, only_high=only_high)

        Ycoords, Xcoords = np.indices((j2-j1, i2-i1))

        if use_robust_estimation:
            bg_pixels = shoebox[is_background]
            if not bg_pixels.size:
                bg_signal = np.nan
            else:
                bg_signal = np.median(bg_pixels)
            if bg_signal < 0:
                num_roi_negative_bg += 1
                if set_negative_bg_to_zero:
                    bg_signal = 0
                elif skip_roi_with_negative_bg:
                    MAIN_LOGGER.debug("reflection %d has negative background" % i_roi)
                    is_selected = False
                elif np.isnan(bg_signal):
                    is_selected = False
            tilt_a, tilt_b, tilt_c = 0, 0, bg_signal
            covariance = None

            tilt_plane = tilt_a * Xcoords + tilt_b * Ycoords + tilt_c
        else:
            fit_results = fit_plane_equation_to_background_pixels(
                shoebox_img=shoebox, fit_sel=is_background, sigma_rdout=shoebox_sigma_readout,
                weighted=weighted_fit)
            if fit_results is None:
                tilt_a = tilt_b = tilt_c = covariance = 0
                MAIN_LOGGER.debug("Reflection %d has no fit!" % i_roi)
                is_selected = False
                MAIN_LOGGER.debug("tilt fit failed for reflection %d, probably too few pixels" % i_roi)
                tilt_plane = np.zeros_like(Xcoords)
            else:
                #MAIN_LOGGER.debug("successfully fit tilt plane")
                (tilt_a, tilt_b, tilt_c), covariance = fit_results
                tilt_plane = tilt_a * Xcoords + tilt_b * Ycoords + tilt_c
                if np.any(np.isnan(tilt_plane)) and is_selected:
                    MAIN_LOGGER.debug("reflection %d has nan in plane" % i_roi)
                    is_selected = False
                    num_roi_nan_bg += 1
                if skip_roi_with_negative_bg and np.min(tilt_plane) < 0:  # dips below
                    num_roi_negative_bg += 1
                    MAIN_LOGGER.debug("reflection %d has tilt plane that dips below 0" % i_roi)
                    is_selected = False

        # unpadded ROI dimension
        roi_dimY = j2_nopad-j1_nopad
        roi_dimX = i2_nopad-i1_nopad

        if roi_dimY < 2 or roi_dimX < 2:
            MAIN_LOGGER.debug("reflection %d is too small" % (i_roi))
            is_selected = False

        j1 = j1_nopad-j1
        i1 = i1_nopad-i1
        plane_in_unpadded_roi = tilt_plane[j1: j1 + roi_dimY, i1: i1 + roi_dimX]
        background[pid, j1_nopad:j2_nopad, i1_nopad:i2_nopad] = plane_in_unpadded_roi
        tilt_abc.append((tilt_a, tilt_b, tilt_c))
        all_cov.append(covariance)
        kept_rois.append(roi)
        panel_ids.append(pid)
        selection_flags.append(is_selected)
        if not is_selected:
            MAIN_LOGGER.debug("--> %s was not selected for above reasons" % refl_bbox_str)
        i_roi += 1

    MAIN_LOGGER.debug("Number of skipped ROI with negative BGs: %d / %d" % (num_roi_negative_bg, len(rois)))
    MAIN_LOGGER.debug("Number of skipped ROI with NAN in BGs: %d / %d" % (num_roi_nan_bg, len(rois)))
    MAIN_LOGGER.info("Number of ROIS that will proceed to refinement: %d/%d" % (np.sum(selection_flags), len(rois)))
    if ret_cov:
        return kept_rois, panel_ids, tilt_abc, selection_flags, background, all_cov
    else:
        return kept_rois, panel_ids, tilt_abc, selection_flags, background


def determine_shoebox_ROI(detector, delta_Q, wavelength_A, refl, centroid="obs"):
    panel = detector[refl["panel"]]
    width = get_width_of_integration_shoebox(panel, delta_Q, wavelength_A, refl['rlp'])
    wby2 = int(round(width/2.))
    fdim,sdim = panel.get_image_size()
    if centroid=='obs':  #TODO: give this more options
        i_com, j_com,_ = refl['xyzobs.px.value']
    elif centroid=='cal':
        i_com, j_com,_ = refl['xyzcal.px']
    else:
        i1,i2,j1,j2,_,_ = refl['bbox']  # super weird funky spots can skew bbox such that its a bad measure of centroid
        i_com = (i1+i2) * .5
        j_com = (j1+j2) * .5
    i1, i2, j1, j2 = i_com - wby2, i_com + wby2, j_com - wby2, j_com + wby2
    on_edge = False
    if i1 < 0 or j1 < 0:
        on_edge = True
    elif i2 >= fdim  or j2 >= sdim:
        on_edge = True
    i1 = max(int(i1), 0)
    i2 = min(int(i2), fdim)
    j1 = max(int(j1), 0)
    j2 = min(int(j2), sdim)
    return (i1, i2, j1, j2), on_edge


def get_width_of_integration_shoebox(detector_panel, delta_Q, ave_wavelength_A,  rlp):
    Qmag = 2 * np.pi * np.linalg.norm(rlp)  # magnitude momentum transfer of the RLP in physicist convention
    detdist_mm = detector_panel.get_distance()
    pixsize_mm = detector_panel.get_pixel_size()[0]
    rad1 = (detdist_mm / pixsize_mm) * np.tan(2 * np.arcsin((Qmag - delta_Q * .5) * ave_wavelength_A / 4 / np.pi))
    rad2 = (detdist_mm / pixsize_mm) * np.tan(2 * np.arcsin((Qmag + delta_Q * .5) * ave_wavelength_A / 4 / np.pi))
    bbox_extent = (rad2 - rad1) / np.sqrt(2)  # rad2 - rad1 is the diagonal across the bbox
    return bbox_extent


def fit_plane_equation_to_background_pixels(shoebox_img, fit_sel, sigma_rdout=3, weighted=True, relative_XY=None):
    """
    :param shoebox_img: 2D pixel image (usually less than 100x100 pixels)
    :param fit_sel: pixels to be fit to plane (usually background pixels)
    :param sigma_rdout: float or 2D image , readout noise term ( in shoebox_img pixel units)
    :return: coefficients of tilt plane (fast-scan, slow-scan, offset), as well as a covariance estimate matrix
    """
    # fast scan pixels, slow scan pixels, pixel values (corrected for gain)
    Y, X = np.indices(shoebox_img.shape)
    if relative_XY is not None:
        X += relative_XY[0]
        Y += relative_XY[1]
    fast, slow, rho_bg = X[fit_sel], Y[fit_sel], shoebox_img[fit_sel]


    try:
        sigma_rdout = sigma_rdout[fit_sel]
    except TypeError:
        pass

    # do the fit of the background plane
    A = np.array([fast, slow, np.ones_like(fast)]).T
    # weights matrix:
    if weighted:
        W = np.diag(1 / (sigma_rdout ** 2 + rho_bg))
        W[W <0] = 0
        W = np.sqrt(W)
    else:
        W = np.eye(len(rho_bg))
    AWA = np.dot(A.T, np.dot(W, A))
    try:
        AWA_inv = np.linalg.inv(AWA)
    except np.linalg.LinAlgError:
        MAIN_LOGGER.debug("WARNING: Fit did not work.. investigate reflection, number of pixels used in fit=%d" % len(fast))
        return None
    AtW = np.dot(A.T, W)
    t1, t2, t3 = np.dot(np.dot(AWA_inv, AtW), rho_bg)

    # get the covariance of the tilt coefficients:
    # vector of residuals
    r = rho_bg - np.dot(A, (t1, t2, t3))
    Nbg = len(rho_bg)
    with np.errstate(divide='ignore', invalid='ignore'):
        r_fact = np.dot(r.T, np.dot(W, r)) / (Nbg - 3)  # 3 parameters fit
    var_covar = AWA_inv * r_fact  # TODO: check for correlations in the off diagonal elems

    return (t1, t2, t3), var_covar


def strip_thickness_from_detector(detector):
    """return a new dxtbx detector with 0 thickness"""
    return set_detector_thickness(detector,0,0)


def set_detector_thickness(detector, thick=0, mu=0):
    """
    warning: this overrides detector hierarchy

    :param detector: dxtbx detector model
    :param thick: sensor thickness in mm
    :param mu: sensor absorption length in mm
    :return: new dxtbx detector model
    """
    px_type = "SimplePxMmStrategy"
    if thick > 0:
        px_type = "ParallaxCorrectedPxMmStrategy"

    new_detector = Detector()
    for i_pan in range(len(detector)):
        panel = detector[i_pan]
        panel_dict = panel.to_dict()
        # NOTE: necessary to set origin and fast/slow because panel.to_dict doesnt care about hierarchy
        panel_dict["origin"] = panel.get_origin()
        panel_dict["fast_axis"] = panel.get_fast_axis()
        panel_dict["slow_axis"] = panel.get_slow_axis()
        panel_dict["mu"] = mu
        panel_dict["thickness"] = thick
        panel_dict["px_mm_strategy"] = {'type': px_type}
        new_panel = Panel.from_dict(panel_dict)
        new_detector.add_panel(new_panel)
    return new_detector


def image_data_from_expt(expt, as_double=True):
    """

    :param expt: dxtbx experiment
    :param as_double: return data as doubles
    :return:
    """
    iset = expt.imageset
    if len(iset) == 0:
        raise ValueError("imageset should have 1 shot")
    if len(iset) > 1:
        raise ValueError("imageset should have only 1 shot. This expt has imageset with %d shots" % len(iset))
    try:
        flex_data = iset.get_raw_data(0)

    except Exception as err:
        assert str(type(err)) == "<class 'Boost.Python.ArgumentError'>", "something weird going on with imageset data"
        flex_data = iset.get_raw_data()
    if not isinstance(flex_data, tuple):
        flex_data = (flex_data, )
    img_data = np.array([data.as_numpy_array() for data in flex_data])
    if as_double:
        img_data = img_data.astype(np.float64)
    return img_data


def simulator_from_expt(expt, oversample=0, device_id=0, init_scale=1, total_flux=1e12,
                        ncells_abc=(10,10,10), has_isotropic_ncells=True, mosaicity=0,
                        num_mosaicity_samples=1, mtz_name=None,
                        mtz_column=None, default_F=0, dmin=1.5, dmax=30,
                        spectra_file=None, spectra_stride=1,
                        complex_F=None):
    raise NotImplementedError("This will be a convenience method")
    # params = get_params_object()
    # params.simulator.oversample = oversample
    # etc
    #return simulator_from_expt_and_params(expt, params)


def simulator_for_refinement(expt, params):
    # TODO: choose a phil param and remove support for the other: crystal.anositropic_mosaicity, or init.eta_abc
    MAIN_LOGGER.info(
        "Setting initial mosaicity from params.init.eta_abc: %f %f %f" % tuple(params.init.eta_abc))
    if params.simulator.crystal.has_isotropic_mosaicity:
        params.simulator.crystal.anisotropic_mosaicity = None
        params.simulator.crystal.mosaicity = params.init.eta_abc[0]
    else:
        params.simulator.crystal.anisotropic_mosaicity = params.init.eta_abc
    MAIN_LOGGER.info("Number of mosaic domains from params: %d" % params.simulator.crystal.num_mosaicity_samples)

    #  GET SIMULATOR #
    SIM = simulator_from_expt_and_params(expt, params)

    if SIM.D.mosaic_domains > 1:
        MAIN_LOGGER.info("Will use mosaic models: %d domains" % SIM.D.mosaic_domains)
    else:
        MAIN_LOGGER.info("Will not use mosaic models, as simulator.crystal.num_mosaicity_samples=1")

    if not params.fix.eta_abc:
        assert SIM.D.mosaic_domains > 1

    if not params.fix.diffuse_gamma or not params.fix.diffuse_sigma:
        assert params.use_diffuse_models
    SIM.D.use_diffuse = params.use_diffuse_models
    SIM.D.gamma_miller_units = params.gamma_miller_units
    SIM.isotropic_diffuse_gamma = params.isotropic.diffuse_gamma
    SIM.isotropic_diffuse_sigma = params.isotropic.diffuse_sigma

    SIM.D.no_Nabc_scale = params.no_Nabc_scale  # TODO check gradients for this setting
    SIM.D.update_oversample_during_refinement = False

    return SIM


def simulator_from_expt_and_params(expt, params=None):
    """
    :param expt:  dxtbx experiment
    :param params: diffBragg/phil.py phil params
    :return:
    """

    oversample = params.simulator.oversample
    device_id = params.simulator.device_id
    init_scale = params.simulator.init_scale
    total_flux = params.simulator.total_flux

    ncells_abc = params.simulator.crystal.ncells_abc
    ncells_def = params.simulator.crystal.ncells_def
    has_isotropic_ncells = params.simulator.crystal.has_isotropic_ncells
    mosaicity = params.simulator.crystal.mosaicity
    num_mosaicity_samples = params.simulator.crystal.num_mosaicity_samples

    default_F = params.simulator.structure_factors.default_F

    spectra_file = params.simulator.spectrum.filename
    spectra_stride = params.simulator.spectrum.stride
    aniso_mos_spread = params.simulator.crystal.anisotropic_mosaicity

    if has_isotropic_ncells:
        if len(set(ncells_abc)) != 1 :
            raise ValueError("`isotropic_ncells=True`, so `ncells_abc` should all be the same, not %d %d %d" % tuple(ncells_abc))

    # make a simulator instance
    SIM = SimData()
    if params.simulator.detector.force_zero_thickness:
        SIM.detector = strip_thickness_from_detector(expt.detector)
    else:
        atten = params.simulator.detector.atten
        thick = params.simulator.detector.thick
        if atten is not None and thick is not None:
            expt.detector = set_detector_thickness(expt.detector, thick, 1./atten)
        SIM.detector = expt.detector

    # create nanoBragg crystal
    crystal = NBcrystal(init_defaults=False)
    crystal.xtal_shape = "gauss"
    crystal.isotropic_ncells = has_isotropic_ncells
    if params.simulator.crystal.rotXYZ_ucell is not None:
        rotXYZ = params.simulator.crystal.rotXYZ_ucell[:3]
        ucell_p = params.simulator.crystal.rotXYZ_ucell[3:]
        crystal.dxtbx_crystal = modify_crystal(rotXYZ, ucell_p, expt.crystal)
    else:
        crystal.dxtbx_crystal = expt.crystal
    crystal.thick_mm = 0.1  # hard code a thickness, will be over-written by the scale
    # mosaic block size
    crystal.Ncells_abc = tuple(ncells_abc)
    if ncells_def is not None:
        crystal.Ncells_def = tuple(ncells_def)
    crystal.anisotropic_mos_spread_deg = aniso_mos_spread
    crystal.n_mos_domains = num_mosaicity_samples
    crystal.mos_spread_deg = mosaicity

    # load the structure factors
    miller_data = load_Fhkl_model_from_params_and_expt(params, expt)
    crystal.miller_array = miller_data
    if params.refiner.force_symbol is not None:
        crystal.symbol = params.refiner.force_symbol
    else:
        crystal.symbol = miller_data.crystal_symmetry().space_group_info().type().lookup_symbol()
    SIM.crystal = crystal

    # create a nanoBragg beam
    beam = NBbeam()
    beam.size_mm = params.simulator.beam.size_mm
    beam.unit_s0 = expt.beam.get_unit_s0()
    if spectra_file is not None:
        init_spectrum = load_spectra_file(spectra_file, total_flux, spectra_stride, as_spectrum=True)
    else:
        assert total_flux is not None
        init_spectrum = [(expt.beam.get_wavelength(), total_flux)]
    beam.spectrum = init_spectrum
    SIM.beam = beam

    # create the diffbragg object, which is the D attribute of SIM
    SIM.panel_id = 0
    SIM.instantiate_diffBragg(oversample=oversample, device_Id=device_id, default_F=default_F, verbose=params.refiner.verbose)
    if init_scale is not None:
        #TODO phase this parameter out since its redundant?
        SIM.update_nanoBragg_instance("spot_scale", init_scale)

    test_panel = SIM.detector[0]
    if test_panel.get_thickness() > 0:
        SIM.update_nanoBragg_instance(
            "detector_thicksteps", params.simulator.detector.thicksteps)
    MAIN_LOGGER.debug("Detector thicksteps = %d" % SIM.D.detector_thicksteps )
    MAIN_LOGGER.debug("Detector thick = %f mm" % SIM.D.detector_thick_mm )
    MAIN_LOGGER.debug("Detector atten len = %f mm" % SIM.D.detector_attenuation_length_mm )
    if params.simulator.psf.use:
        SIM.use_psf = True
        SIM.psf_args = {'pixel_size': SIM.detector[0].get_pixel_size()[0]*1e3,
                'fwhm': params.simulator.psf.fwhm,
                'psf_radius': params.simulator.psf.radius}
        fwhm_pix = SIM.psf_args["fwhm"] / SIM.psf_args["pixel_size"]
        kern_size = SIM.psf_args["psf_radius"]*2 + 1
        SIM.PSF = psf.makeMoffat_integPSF(fwhm_pix, kern_size, kern_size)

    update_SIM_with_gonio(SIM, params)

    return SIM


def update_SIM_with_gonio(SIM, params=None, delta_phi=None, num_phi_steps=5, spindle_axis=(1,0,0)):
    """

    :param SIM: sim_data instance
    :param params: diffBragg phil parameters instance
    :param delta_phi: how much to rotate gonio during model
    :param num_phi_steps: number of phi steps
    :return:
    """
    if not hasattr(SIM, "D"):
        raise AttributeError("Need to instantiate diffBragg first")
    if params is not None:
        delta_phi = params.simulator.gonio.delta_phi
        num_phi_steps = params.simulator.gonio.phi_steps

    if delta_phi is not None:
        SIM.D.spindle_axis = spindle_axis
        SIM.D.phi_deg = 0
        SIM.D.osc_deg = delta_phi
        SIM.D.phisteps = int(num_phi_steps)


def get_complex_fcalc_from_pdb(
        pdb_file,
        wavelength=None,
        dmin=1,
        dmax=None,
        k_sol=0.435, b_sol=46, show_pdb_summary=False):
    """
    produce a structure factor from PDB coords, see mmtbx/programs/fmodel.py for formulation
    k_sol, b_sol form the solvent component of the Fcalc: Fprotein + k_sol*exp(-b_sol*s^2/4) (I think)
    """
    import iotbx.pdb
    pdb_in = iotbx.pdb.input(pdb_file)
    xray_structure = pdb_in.xray_structure_simple()
    if show_pdb_summary:
        xray_structure.show_summary()
    for sc in xray_structure.scatterers():
        if wavelength is not None:
            expected_henke = henke.table(sc.element_symbol()).at_angstrom(wavelength)
            sc.fp = expected_henke.fp()
            sc.fdp = expected_henke.fdp()
    phil2 = mmtbx.programs.fmodel.master_phil
    params2 = phil2.extract()
    params2.high_resolution = dmin
    params2.low_resolution = dmax
    params2.fmodel.k_sol = k_sol
    params2.fmodel.b_sol = b_sol
    params2.structure_factors_accuracy.algorithm = 'fft'
    f_model = mmtbx.utils.fmodel_from_xray_structure(
        xray_structure=xray_structure,
        f_obs=None,
        add_sigmas=False,
        params=params2).f_model
    f_model = f_model.generate_bijvoet_mates()

    return f_model


def open_mtz(mtzfname, mtzlabel=None, verbose=False):
    """

    :param mtzfname: path to mtz
    :param mtzlabel: column in mtz
    :param verbose:
    :return:
    """
    if mtzlabel is None:
        mtzlabel = "fobs(+)fobs(-)"
    if verbose:
        MAIN_LOGGER.info("Opening mtz file %s , label %s" % (mtzfname, mtzlabel))
    from iotbx.reflection_file_reader import any_reflection_file
    miller_arrays = any_reflection_file(mtzfname).as_miller_arrays()

    possible_labels = []
    foundlabel = False
    for ma in miller_arrays:
        label = ma.info().label_string()
        possible_labels.append(label)
        if label == mtzlabel:
            foundlabel = True
            break

    assert foundlabel, "MTZ Label not found... \npossible choices: %s" % (" ".join(possible_labels))
    if not ma.is_xray_amplitude_array():
        ma = ma.as_amplitude_array()
    return ma


def make_miller_array(symbol, unit_cell, defaultF=1000, d_min=1.5, d_max=999):
    """

    :param symbol: space group e.g. P43212
    :param unit_cell: unit cell tuple (6-tuple, a,b,c,alpha,beta,gamma)
    :param defaultF: structure factor amplitude (constant for all HKL)
    :param d_min: high res
    :param d_max: low res
    :return:  cctbx miller array
    """
    sgi = sgtbx.space_group_info(symbol)
    # TODO: allow override of ucell
    symm = symmetry(unit_cell=unit_cell, space_group_info=sgi)
    miller_set = symm.build_miller_set(anomalous_flag=True, d_min=d_min, d_max=d_max)
    # NOTE does build_miller_set automatically expand to p1 ? Does it obey systematic absences ?
    # Note how to handle sys absences here ?
    Famp = flex.double(np.ones(len(miller_set.indices())) * defaultF)
    mil_ar = miller.array(miller_set=miller_set, data=Famp).set_observation_type_xray_amplitude()
    return mil_ar


def make_miller_array_from_crystal(Crystal, dmin, dmax, defaultF=1000, symbol=None):
    if symbol is None:
        symbol = Crystal.get_space_group().info().type().lookup_symbol()
    Famp = make_miller_array(
        symbol=symbol,
        unit_cell=Crystal.get_unit_cell(), d_min=dmin, d_max=dmax, defaultF=defaultF)
    return Famp


def save_spectra_file(spec_file, wavelengths, weights):
    """
    Create a precognition .lam file
    :param spec_file: name
    :param wavelengths: list of wavelen
    :param weights: list of weights
    """
    data = np.array([wavelengths, weights])
    np.savetxt(spec_file, data.T, delimiter=',', header="wavelengths, weights")


def load_spectra_file(spec_file, total_flux=None, pinkstride=1, as_spectrum=False, delim=","):
    """
    load a precognition .lam file
    :param spec_file: path to file
    :param total_flux: total photons per shot
    :param pinkstride: wavelength stride (e.g. pinkstride=10 is every 10th wavelength)
    :param as_spectrum: as a nanoBragg_beam.NBbeam.spectrum object
    :param delim: column delimiter
    """
    wavelengths, weights = np.loadtxt(spec_file, float, delimiter=delim, skiprows=1).T
    if isinstance(wavelengths, float) and isinstance(weights, float):
        # the file had one entry:
        wavelengths = np.array([wavelengths])
        weights = np.array([weights])
    if pinkstride > len(wavelengths) or pinkstride == 0:
        raise ValueError("Incorrect value for pinkstride")
    wavelengths = wavelengths[::pinkstride]
    weights = weights[::pinkstride]
    energies = ENERGY_CONV/wavelengths
    if total_flux is not None:
        weights = weights / weights.sum() * total_flux
    if as_spectrum:
        return list(zip(list(wavelengths), list(weights)))
    else:
        return weights, energies


def save_numpy_mask_as_flex(numpymask, outfile):
    flexmask = tuple((dials_flex.bool(m) for m in numpymask))
    with open(outfile, "wb") as f:
        pickle.dump(flexmask, f)

def load_mask(maskfile):
    """

    :param maskfile: path to a dials mask file (tuple of flex)
    :return:
    """
    if maskfile is None:
        return None
    with open(maskfile, 'rb') as o:
        mask = pickle.load(o)
    if isinstance(mask, tuple):
        mask = np.array([m.as_numpy_array() for m in mask])
    else:
        mask = mask.as_numpy_array()
    return mask


def unitcell_sigmas(unitcell_manager, unitcell_sigmas):
    """

    :param unitcell_manager: unit cell manager (see diffBragg/refiners/crystal_systems
    :param unitcell_sigmas: refinement sensitivities
    :return:
    """
    name_mapping = {'a_Ang': 0, 'b_Ang': 1, 'c_Ang': 2, 'alpha_rad': 3, 'beta_rad': 4, 'gamma_rad': 5}
    variable_sigmas = []
    for name in unitcell_manager.variable_names:
        sig = unitcell_sigmas[name_mapping[name]]
        variable_sigmas.append(sig)
    return variable_sigmas


def manager_from_crystal(crystal):
    """
    :param crystal:  dxtbx crystal model
    :return:
    """
    params = crystal.get_unit_cell().parameters()
    return manager_from_params(params)


def manager_from_params(ucell_p):
    """
    :param ucell_p: unit cell 6-tuple
    :return: unit cell manager (crystal systems)
    """

    a, b, c, al, be, ga = ucell_p
    if np.isclose(a,b) and not np.isclose(a,c) and np.allclose([al, be, ga], [90]*3):
        manager = TetragonalManager(a=a, c=c)

    elif not np.isclose(a,b) and not np.isclose(b,c) and not np.isclose(a,c) and np.allclose([al, be, ga], [90]*3):
        manager = OrthorhombicManager(a=a, b=b, c=c)

    elif np.isclose(a,b) and not np.isclose(a,c) and np.allclose([al, be], [90]*2) and np.allclose([ga], [120]):
        manager = HexagonalManager(a=a, c=c)

    elif not np.isclose(a,b) and not np.isclose(b,c) and not np.isclose(a,c) and np.allclose([al, ga], [90]*2) and not np.allclose([be], [120]):
        manager = MonoclinicManager(a=a, b=b, c=c, beta=be*np.pi/180.)

    else:
        raise NotImplementedError("Not yet implemented for crystal model")

    return manager


def detect_peaks(image_, threshold=0):
    """
    Detector of peaks in a 2d array
    Borrowed recipe: https://stackoverflow.com/a/3689710/2077270
    :param image: 2d img
    :param threshold: float, cutoff, pixels below this are assumed to background (e.g. not peaks)
    :returns: a binary image thats 1 at the potision of the peaks
    """
    image = image_.copy()
    image[image < threshold] = 0
    neighborhood = generate_binary_structure(2,2)
    local_max = maximum_filter(image, footprint=neighborhood)==image
    background = (image == 0)

    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    detected_peaks = local_max ^ eroded_background

    return detected_peaks


def refls_from_sims(panel_imgs, detector, beam, thresh=0, filter=None, panel_ids=None,
                    max_spot_size=1000, use_detect_peaks=False, **kwargs):
    """
    This is for converting the centroids in the noiseless simtbx images
    to a multi panel reflection table

    :param panel_imgs: list or 3D array of detector panel simulations
    :param detector: dxtbx  detector model of a caspad
    :param beam:  dxtxb beam model
    :param thresh: threshol intensity for labeling centroids
    :param filter: optional filter to apply to images before
        labeling threshold, typically one of scipy.ndimage's filters
    :param pids: panel IDS , else assumes panel_imgs is same length as detector
    :param max_spot_size: maximum number of px in a spot
    :param use_detect_peaks: precisely compute the peak of each simulated spot
        and throw away the profile
    :param kwargs: kwargs to pass along to the optional filter
    :return: a reflection table of spot centroids
    """
    from dials.algorithms.spot_finding.factory import FilterRunner
    from dials.model.data import PixelListLabeller, PixelList
    from dials.algorithms.spot_finding.finder import pixel_list_to_reflection_table

    if panel_ids is None:
        panel_ids = np.arange(len(detector))
    pxlst_labs = []
    for i, pid in enumerate(panel_ids):
        plab = PixelListLabeller()
        img = panel_imgs[i]
        if use_detect_peaks:
            mask = detect_peaks(img, thresh)
        elif filter is not None:
            mask = filter(img, **kwargs) > thresh
        else:
            mask = img > thresh
        img_sz = detector[int(pid)].get_image_size()  # for some reason the int cast is necessary in Py3
        flex_img = flex.double(img)
        flex_img.reshape(flex.grid(img_sz))

        flex_mask = flex.bool(mask)
        flex_mask.resize(flex.grid(img_sz))
        pl = PixelList(0, flex.double(img), flex.bool(mask))
        plab.add(pl)

        pxlst_labs.append(plab)

    El = explist_from_numpyarrays(panel_imgs, detector, beam)
    iset = El.imagesets()[0]
    refls = pixel_list_to_reflection_table(
        iset, pxlst_labs,
        min_spot_size=1,
        max_spot_size=max_spot_size,  # TODO: change this ?
        filter_spots=FilterRunner(),  # must use a dummie filter runner!
        write_hot_pixel_mask=False)[0]

    return refls


class FormatInMemory:
    """
    this class is a special image type
    necessary to create dxtbx imagesets and
    datablocks from numpy array images
    and masks.
    """
    def __init__(self, image, mask=None):
        self.image = image
        if image.dtype != np.float64:
            self.image = self.image.astype(np.float64)
        if mask is None:
            self.mask = np.ones_like(self.image).astype(bool)
        else:
            assert (mask.shape == image.shape)
            assert(mask.dtype == bool)
            self.mask = mask

    def get_raw_data(self):
        if len(self.image.shape)==2:
            return flex.double(self.image)
        else:
            return tuple([flex.double(panel) for panel in self.image])

    def get_mask(self, goniometer=None):
        if len(self.image.shape)==2:
            return flex.bool(self.mask)
        else:
            return tuple([flex.bool(panelmask) for panelmask in self.mask])


def explist_from_numpyarrays(image, detector, beam, mask=None):
    """
    So that one can do e.g.
    >> dblock = datablock_from_numpyarrays( image, detector, beam)
    >> refl = flex.reflection_table.from_observations(dblock, spot_finder_params)
    without having to utilize the harddisk

    :param image:  numpy array image, or list of numpy arrays
    :param mask:  numpy mask, should be same shape format as numpy array
    :param detector: dxtbx detector model
    :param beam: dxtbx beam model
    :return: datablock for the image
    """
    if isinstance( image, list):
        image = np.array( image)
    if mask is not None:
        if isinstance( mask, list):
            mask = np.array(mask).astype(bool)
    I = FormatInMemory(image=image, mask=mask)
    reader = MemReader([I])
    iset_Data = ImageSetData(reader, None) # , masker)
    iset = ImageSet(iset_Data)
    iset.set_beam(beam)
    iset.set_detector(detector)
    explist = ExperimentListFactory.from_imageset_and_crystal(iset, None)
    return explist


def load_panel_group_file(panel_group_file):
    """

    :param panel_group_file: file specifying which group IDs for multi panel detectors
    :return:
    """
    lines = open(panel_group_file, 'r').readlines()
    groups = {}
    for l in lines:
        try:
            panel, group = map(int, l.strip().split())
        except ValueError:
            continue
        groups[panel] = group
    return groups


def load_spectra_from_dataframe(df):
    """
    :param df:pandas dataframe
    :return:
    """
    total_flux = df.total_flux.values[0]
    spectrum_file = df.spectrum_filename.values[0]
    pink_stride = df.spectrum_stride.values[0]
    spec = load_spectra_file(spectrum_file, total_flux=total_flux,
                            pinkstride=pink_stride, as_spectrum=True)
    return spec


def refls_to_q(refls, detector, beam, update_table=False):
    """
    gets the Q-vector at each refl
    :param refls: reflection table
    :param detector: detector
    :param beam: beam
    :param update_table: update the table with rlp col
    :return:
    """

    orig_vecs = {}
    fs_vecs = {}
    ss_vecs = {}
    u_pids = set(refls['panel'])
    for pid in u_pids:
        orig_vecs[pid] = np.array(detector[pid].get_origin())
        fs_vecs[pid] = np.array(detector[pid].get_fast_axis())
        ss_vecs[pid] = np.array(detector[pid].get_slow_axis())

    s1_vecs = []
    q_vecs = []
    panels = refls["panel"]
    n_refls = len(refls)
    for i_r in range(n_refls):
        r = refls[i_r]
        pid = r['panel']
        i_fs, i_ss, _ = r['xyzobs.px.value']
        panel = detector[pid]
        orig = orig_vecs[pid] #panel.get_origin()
        fs = fs_vecs[pid] #panel.get_fast_axis()
        ss = ss_vecs[pid] #panel.get_slow_axis()

        fs_pixsize, ss_pixsize = panel.get_pixel_size()
        s1 = orig + i_fs*fs*fs_pixsize + i_ss*ss*ss_pixsize  # scattering vector
        s1 = s1 / np.linalg.norm(s1) / beam.get_wavelength()
        s1_vecs.append(s1)
        q_vecs.append(s1-beam.get_s0())

    if update_table:
        refls['s1'] = flex.vec3_double(tuple(map(tuple, s1_vecs)))
        refls['rlp'] = flex.vec3_double(tuple(map(tuple, q_vecs)))

    return np.vstack(q_vecs)


def modify_crystal(anglesXYZ, ucell_params, starting_crystal):
    """
    applies a rotation and unit cell adjustment to crystal
    :param anglesXYZ: rotation angles
    :param ucell_params: unit cell param
    :param starting_crystal: dxtbx crystal
    :return:
    """
    from scitbx.matrix import col
    from copy import deepcopy
    x = col((-1, 0, 0))
    y = col((0, -1, 0))
    z = col((0, 0, -1))
    RX = x.axis_and_angle_as_r3_rotation_matrix(anglesXYZ[0], deg=False)
    RY = y.axis_and_angle_as_r3_rotation_matrix(anglesXYZ[1], deg=False)
    RZ = z.axis_and_angle_as_r3_rotation_matrix(anglesXYZ[2], deg=False)
    M = RX * RY * RZ
    q = M.r3_rotation_matrix_as_unit_quaternion()
    rot_ang, rot_ax = q.unit_quaternion_as_axis_and_angle(deg=True)

    UCELL_MAN = manager_from_params(ucell_params)
    B = UCELL_MAN.B_recipspace

    C = deepcopy(starting_crystal)
    C.set_B(B)
    if rot_ang > 0:
        C.rotate_around_origin(rot_ax, rot_ang)
    return C


def parse_reso_string(s):
    """
    :param s:  a strong formated as %f-%f,%f-%f,%f-%f etc.
    :return: two floats as a tuple
    """
    vals =[]
    try:
        for subs in s.strip().split(","):
            a, b = map(float, subs.strip().split("-"))
            assert a < b
            vals.append((a,b))
    except Exception as error:
        MAIN_LOGGER.error("Failed to parse string!", error)
        raise ValueError("Wrong string format, see error above")
    return vals


def refls_to_hkl(refls, detector, beam, crystal,
                 update_table=False, returnQ=False, wavelen=None):
    """
    convert pixel panel reflections to miller index data
    :param refls:  reflecton table for a panel or a tuple of (x,y)
    :param detector:  dxtbx detector model
    :param beam:  dxtbx beam model
    :param crystal: dxtbx crystal model
    :param update_table: whether to update the refltable
    :param returnQ: whether to return intermediately computed q vectors
    :return: if as_numpy two Nx3 numpy arrays are returned
        (one for fractional and one for whole HKL)
        else dictionary of hkl_i (nearest) and hkl (fractional)
    """
    from scitbx.matrix import sqr
    if 'rlp' not in list(refls.keys()):
        q_vecs = refls_to_q(refls, detector, beam, update_table=update_table)
    else:
        q_vecs = np.vstack([refls[i_r]['rlp'] for i_r in range(len(refls))])
    Ai = sqr(crystal.get_A()).inverse()
    Ai = Ai.as_numpy_array()
    HKL = np.dot( Ai, q_vecs.T)
    HKLi = np.ceil(HKL-0.5)
    if update_table:
        refls['miller_index'] = flex.miller_index(list(map(tuple, HKLi.T.astype(np.int32))))
    if returnQ:
        return np.vstack(HKL).T, np.vstack(HKLi).T, q_vecs
    else:
        return np.vstack(HKL).T, np.vstack(HKLi).T


def get_panels_fasts_slows(expt, pids, rois, img_sh=None):
    """
    :param expt: dxtbx experiment
    :param pids: panel ids
    :param rois: regions of interest
    :param img_sh: 3-tuple Npan, Nslow, Nfast
    :return:
    """
    if expt is not None:
        npan = len(expt.detector)
        nfast, nslow = expt.detector[0].get_image_size()
    else:
        assert img_sh is not None
        npan, nslow, nfast = img_sh
    MASK = np.zeros((npan, nslow, nfast), bool)
    ROI_ID = np.zeros((npan, nslow, nfast), 'uint16')
    #ROI_ID = NP_ONES((npan, nslow, nfast), 'uint16') * mx
    nspots = len(rois)
    for i_spot in range(nspots):
        x1, x2, y1, y2 = rois[i_spot]
        if x2-x1 == 0 or y2-y1 == 0:
            continue
        pid = pids[i_spot]
        MASK[pid, y1:y2, x1:x2] = True
        ROI_ID[pid, y1:y2, x1:x2] = i_spot
    p,s,f = np.where(MASK)
    roi_id = ROI_ID[p,s,f]
    pan_fast_slow = np.ascontiguousarray((np.vstack([p,f,s]).T).ravel())
    pan_fast_slow = flex.size_t(pan_fast_slow)
    return pan_fast_slow, roi_id


def f_double_prime(energies, a,b,c,d, deriv=None):
    """
    generate a 4 parameter f double prime curve based on the sigmoid function
    :param energies:
    :param a: offset
    :param b: amplitude
    :param c: center
    :param d: slope
    :param deriv: string flag, c,d or None,
        eg 'c' returns deriv of f_dbl_prime w.r.t. c, None returns function f_dbl_prime
    :return: f double prime as a function of energy
    """
    exp_arg = -d * (energies - c)
    e_term = np.exp(exp_arg)
    if deriv=="c":
        return -b*(1+e_term)**-2 * e_term * d
    elif deriv=="d":
        return -b*(1+e_term)**-2 * e_term * (c-energies)
    else:
        return a + b / (1+e_term)


def f_prime(f_double_prime, S=None, padn=5000):
    """
    generate an f_prime from an f_double_prime curve
    using the kramers kronig relationship
    :param f_double_prime: function or its derivative
    :param S:
    :param padn:
    :return:
    """
    if S is None:
        S = np.sin(np.linspace(-np.pi / 2, np.pi / 2, padn)) * 0.5 + 0.5
    else:
        padn = S.shape[0]
    F = f_double_prime
    Fin = np.hstack((F[0] * S,
                     F,
                     F[-1] * (1 - S)))  # sin padding as used in Sherrell thesis, TODO window function ?
    Ft = fft.fft(Fin)
    iFt = -1 * fft.ifft(1j * np.sign(fft.fftfreq(Ft.shape[0])) * Ft).real
    return iFt[padn:-padn]


def shift_panelZ(D, shift):
    """
    :param D:  dxtbx detector
    :param shift:  shift in mm for origin Z component
    :return: new detector with shift applied to each panel origin
    """
    newD = Detector()
    for pid in range(len(D)):
        panel = D[pid]
        pan_dict = panel.to_dict()
        x,y,z = panel.get_origin()
        pan_dict["origin"] = x,y,z+shift
        pan_dict["fast_axis"] = panel.get_fast_axis()
        pan_dict["slow_axis"] = panel.get_slow_axis()
        newP = Panel.from_dict(pan_dict)
        newD.add_panel(newP)
    return newD

def safe_makedirs(name):
    """
    :param name: dirname to create
    """
    if not os.path.exists(name):
        os.makedirs(name)


def _rfactor_minimizer_target(k, F1, F2):
    """
    :param k: scale factor
    :param F1: miller array
    :param F2: miller array
    :return:
    """
    return F1.r1_factor(F2, scale_factor=k[0])


def compute_scale_to_minmize_r_factor(F1, F2, anom=True):
    """
    F1, F2 : two miller arrays (should be of type amplitude)
    scale factor should be applied to F2 such that the R factor is minmized between the two arrays!"""
    if min(F1.data()) < 0:
        F1 = F1.as_amplitude_array()
    if min(F2.data()) < 0:
        F2 = F2.as_amplitude_array()

    indices_common = set(F1.indices()).intersection(F2.indices())
    indices_common = flex.miller_index(list(indices_common))
    F1 = F1.select_indices(indices_common)
    F2_mset = F1.miller_set(indices_common, anom)
    F2_map = {h:d for h,d in zip(F2.indices(), F2.data())}
    F2_data = flex.double([F2_map[h] for h in indices_common])
    F2 = miller.array(F2_mset, F2_data)
    F1=F1.sort(by_value='packed_indices')
    F2=F2.sort(by_value='packed_indices')

    res = minimize(_rfactor_minimizer_target,
                       x0=[1], args=(F1, F2),
                   method='Nelder-Mead')

    assert res.success
    r1_scale = res.x[0]
    MAIN_LOGGER.debug("Optimization successful!, using scale factor=%f" % r1_scale)
    return r1_scale


def show_diffBragg_state(D, debug_pixel_panelfastslow):
    """
    D, diffBragg instance
    debug_pixel_panelfastslow, 3-tuple of ints, panelId, fast coord, slow coord
    """
    # TODO be careful with zero-ing the pixels - is this really what we want to do ?
    # TODO, rather than print the state, pickle the state
    D.show_params()
    MAIN_LOGGER.info("internal spot scale=%f" % D.spot_scale)
    D.raw_pixels*=0
    p, f, s = debug_pixel_panelfastslow
    D.printout_pixel_fastslow = f, s
    D.add_diffBragg_spots((p, f, s))
    D.raw_pixels*=0


def get_phil(params):
    """
    recursively print the phil param string, given a phil scope extract obj
    :param params: libtbx.phil.scope_extract object
    """
    for p in dir(params):
        if p.startswith("_"):
            continue
        name, val = params.__phil_path_and_value__(p)
        if not isinstance(val, libtbx.phil.scope_extract):
            print("%s =" % name, val)
        else:
            get_phil(val)


class Capturing(list):
    # class for capturing function output in a list:
    # https://stackoverflow.com/a/16571630/2077270
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def recover_diff_phil_from_scope_extract(modeler_file):
    """
    Prints the diff phil used to run stage 1 (diffBragg portion only)
    :param modeler_file: npy file written by hopper_process during stage 1, stored in the
    output.output_dir/modelers folder
    """
    # get the phil scope extract
    params = np.load(modeler_file, allow_pickle=True)[()].params
    with Capturing() as output:
        get_phil(params)
    user_phil = parse("\n".join(output))
    master = parse("diffBragg {\n%s\n}" %  (phil.philz + phil.hopper_phil))
    master.fetch_diff(source=user_phil).show()


def get_extracted_params_from_phil_sources(phil_file=None, cmdline_phil_lst=None):
    """
    get a params obj derived from diffBragg/phil.py thats ready to passed to various methods
    :param phil_file: is the path to a phil configuration file for diffBragg/phil.py
    :param cmdline_phil_lst: list of strings, each a command line phil arg, e.g. ['rank0_level=high', 'outdir=something']

    Just like in normal operation, the cmdline str does not need to specify the absolute scope unless there are conflicting scopes
    """

    phil_scope = parse(phil.philz + phil.hopper_phil)
    arg_interp = phil_scope.command_line_argument_interpreter(home_scope="")

    phil_sources = []

    if phil_file is not None:
        phil_from_file = open(phil_file, "r").read()
        user_phil = parse(phil_from_file)
        phil_sources.append(user_phil)

    if cmdline_phil_lst is not None:
        command_line_phils = [arg_interp.process(phil.strip()) for phil in cmdline_phil_lst]
        phil_sources += command_line_phils

    working_phil, unused = phil_scope.fetch(sources=phil_sources, track_unused_definitions=True)
    for loc in unused:
        print("WARNING: unused phil:", loc)

    params = working_phil.extract()
    return params


def get_laue_group_number(sg_symbol=None):
    """ get laue group number from space group symbol """
    if sg_symbol is None:
        laue_sym = "P-1"
    else:
        g = sgtbx.space_group_info(sg_symbol).group()
        if g.laue_group_type() not in ["2/m", "-3m"]:
            laue_sym = "P{}".format(g.laue_group_type())
        else:
            pg = str(g.build_derived_patterson_group().info().symbol_and_number())
            lc = re.sub(r'\([^)]*\)', '', pg[2:]).replace(" ", "")
            if pg[0] == "R":
                lc = lc.replace(":H", "1")
            laue_sym = "P{}".format(lc)

    hm_symbols = ['P-1', 'P112/m', 'P12/m1', 'P2/m11', 'Pmmm', 'P4/m', 'P4/mmm', 'P-3', 'P-3m1', 'P-31m', 'P6/m',
                  'P6/mmm', 'Pm-3', 'Pm-3m']
    lgs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    return lgs[hm_symbols.index(laue_sym)]


def track_fhkl(Modeler):
    try:
        from stream_redirect import Redirect
    except ImportError:
        print("Cannot use track_fhkl unless module stream_redirect is installed: pip install stream-redirect")
        return
    SIM = Modeler.SIM
    SIM.D.track_Fhkl = True
    npix = int( len(Modeler.pan_fast_slow)/ 3)
    PFS = np.reshape(Modeler.pan_fast_slow, (npix, 3))
    uroi = set(Modeler.roi_id)
    all_good_count_stats = []
    all_bad_count_stats = []
    all_count_stats = {}
    for ii, i_roi in enumerate(uroi):
        output = Redirect(stdout=True)
        i_roi_sel = Modeler.roi_id == i_roi
        with output:
            sel = np.logical_and(i_roi_sel, Modeler.all_trusted)
            pfs_roi = PFS[sel]
            pfs_roi = np.ascontiguousarray(pfs_roi.ravel())
            pfs_roi = flex.size_t(pfs_roi)
            SIM.D.add_diffBragg_spots(pfs_roi)
        #i_fcell = Modeler.all_fcell_global_idx[i_roi_sel][0]
        #shoebox_hkl = SIM.asu_from_i_fcell[i_fcell]
        lines = output.stdout.split("\n")
        count_stats = {}
        for l in lines:
            if l.startswith("Pixel"):
                hkl = l.split()[5]
                hkl = tuple(map(int, hkl.split(",")))
                hkl = map_hkl_list([hkl], True, SIM.crystal.space_group_info.type().lookup_symbol())[0]
                count = int(l.split()[8])
                if hkl in count_stats:
                    count_stats[hkl] += count
                else:
                    count_stats[hkl] = count
        ntot = sum(count_stats.values())
        #assert shoebox_hkl in count_stats
        #print("Shoebox hkl", shoebox_hkl)
        for hkl in count_stats:
            frac = 0 if ntot ==0 else count_stats[hkl] / float(ntot)
            h, k, l = hkl
            print("\tstep hkl %d,%d,%d : frac=%.1f%%" % (h, k, l, frac * 100))
            count_stats[hkl] = frac

        all_count_stats[i_roi] = count_stats
    return all_count_stats


def load_Fhkl_model_from_params_and_expt(params, expt):
    """

    :param params:  diffBragg params instance (diffBragg/phil.py)
    :param expt: dxtbx experiment with crystal
    :return:
    """
    sf = params.simulator.structure_factors
    if sf.mtz_name is None:
        if sf.from_pdb.name is not None:
            wavelength=None
            if sf.from_pdb.add_anom:
                wavelength = expt.beam.get_wavelength()
            miller_data = get_complex_fcalc_from_pdb(sf.from_pdb.name,
                dmin=params.simulator.structure_factors.dmin,
                dmax=params.simulator.structure_factors.dmax,
                wavelength=wavelength,
                k_sol=params.simulator.structure_factors.from_pdb.k_sol,
                b_sol=params.simulator.structure_factors.from_pdb.b_sol)
            miller_data = miller_data.as_amplitude_array()

        else:
            miller_data = make_miller_array(
                symbol=expt.crystal.get_space_group().info().type().lookup_symbol(),
                unit_cell=expt.crystal.get_unit_cell(), d_min=sf.dmin,
                d_max=sf.dmax,
                defaultF=sf.default_F)
    else:
        miller_data = open_mtz(sf.mtz_name, sf.mtz_column)

    return miller_data


def find_diffBragg_instances(globe_objs):
    """find any instances of diffbragg in globals
        globe_objs is a return value of globals() (dict)
    """
    inst_names = []
    for name,obj in globe_objs.items():
        if "simtbx.nanoBragg.sim_data.SimData" in str(obj):
            inst_names.append(name)
        if "simtbx_diffBragg_ext.diffBragg" in str(obj):
            inst_names.append(name)
    return inst_names


def memory_report(prefix='Memory usage'):
    """Return a string documenting memory usage; to be used with LOGGER.info"""
    memory_usage_in_gb = get_memory_usage() / 1024.
    host = socket.gethostname()
    return "%s: %f GB on node %s" % (prefix, memory_usage_in_gb, host)


def smooth(x, beta=10.0, window_size=11):
    """
    https://glowingpython.blogspot.com/2012/02/convolution-with-numpy.html

    Apply a Kaiser window smoothing convolution.

    Parameters
    ----------
    x : ndarray, float
        The array to smooth.

    Optional Parameters
    -------------------
    beta : float
        Parameter controlling the strength of the smoothing -- bigger beta
        results in a smoother function.
    window_size : int
        The size of the Kaiser window to apply, i.e. the number of neighboring
        points used in the smoothing.

    Returns
    -------
    smoothed : ndarray, float
        A smoothed version of `x`.
    """

    # make sure the window size is odd
    if window_size % 2 == 0:
        window_size += 1

    # apply the smoothing function
    s = np.r_[x[window_size - 1:0:-1], x, x[-1:-window_size:-1]]
    w = np.kaiser(window_size, beta)
    y = np.convolve(w / w.sum(), s, mode='valid')

    # remove the extra array length convolve adds
    b = int((window_size - 1) / 2)
    smoothed = y[b:len(y) - b]

    return smoothed
