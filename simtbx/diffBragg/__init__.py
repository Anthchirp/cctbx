from __future__ import absolute_import, division, print_function
from boost_adaptbx import boost
import boost_adaptbx.boost.python as bp
import cctbx.uctbx # possibly implicit
from simtbx import nanoBragg # implicit import
ext = boost.python.import_ext("simtbx_diffBragg_ext")
import numpy as np
from simtbx_diffBragg_ext import *

import os
if os.environ.get("DIFFBRAGG_USE_CUDA") is not None and os.environ.get("DIFFBRAGG_USE_KOKKOS") is not None:
    raise RuntimeError("Only set one of DIFFBRAGG_USE_CUDA, DIFFBRAGG_USE_KOKKOS.")

@bp.inject_into(ext.diffBragg)
class _():
    def get_derivative_pixels(self, refine_id):
        return self.__get_derivative_pixels(refine_id)[:self.__number_of_pixels_modeled_using_diffBragg]

    def update_Fhkl_channels(self, channel_ids):
        """channel_ids is a  numpy int array the same length as the number of energy sources,
        that specifies the mapping of structure factor to energy channel,
        allowing one to refine multiple energy-dependent structure factors
        for example in a two-color experiment
        """
        assert isinstance(channel_ids, np.ndarray)
        channel_ids = self._check_contig(channel_ids)
        if channel_ids.dtype != np.int32:
            channel_ids = channel_ids.astype(np.int32)
        # assert len(channel_ids)== number_of_sources
        self.__update_Fhkl_channels(channel_ids)

    def update_Fhkl_scale_factors(self, scales, num_Fhkl_channels):
        assert isinstance(scales, np.ndarray)
        scales = self._check_contig(scales)
        if scales.dtype != np.float64:
            print("Warning, converting scale factors to double!")
            scales = scales.astype(np.float64)
        # TODO add num_asu property, then do the assertion
        #assert len(scales) == self.num_asu*num_Fhkl_channels
        self.__update_Fhkl_scale_factors(scales, num_Fhkl_channels)

    def _check_contig(self, arr):
        if not arr.flags.c_contiguous:
            print("WARNINGS:Making contiguous")
            arr = np.ascontiguousarray(arr)
        return arr

    def get_rotate_principal_axes(self):
        return self._ext_rotate_principal_axes

    def set_rotate_principal_axes(self, val):
        # do stuff to set val, e.g renormalize
        # val =
        self._ext_rotate_principal_axes = val  # as a 9-tuple or a scitbx matrix sqr

    def add_Fhkl_gradients(self, psf, residuals, variance, trusted, freq, num_Fhkl_channels, spot_scale,
                           track=False, errors=False):
        """

        :param psf:
        :param residuals:
        :param variance:
        :param trusted:
        :param freq:
        :param num_Fhkl_channels:
        :param spot_scale:
        :return:
        """

        assert isinstance(residuals, np.ndarray)
        assert isinstance(variance, np.ndarray)
        assert isinstance(trusted, np.ndarray)
        assert isinstance(freq, np.ndarray)
        residuals = self._check_contig(residuals)
        variance = self._check_contig(variance)
        trusted = self._check_contig(trusted)
        Npix = len(psf)/3
        assert Npix == len(residuals)
        assert Npix == len(variance)
        assert Npix == len(trusted)
        assert Npix == len(freq)
        # TODO check or contiguous arrays..

        assert trusted.dtype==bool
        assert freq.dtype==np.int32
        assert residuals.dtype==np.float64
        assert variance.dtype==np.float64
        return self.__add_Fhkl_gradients(psf, residuals, variance, trusted, freq, num_Fhkl_channels, spot_scale,
                                         track,errors)

    def ave_I_cell(self, use_Fhkl_scale=False, i_channel=0, use_geometric_mean=False):
        """

        :param use_Fhkl_scale:
        :param i_channel:
        :param use_geometric_mean: boolean flag, whether arithmetic mean or geometric mean is computed per res bin
        :return:
        """
        # TODO add sanity checks here,
        if not self.dspace_bins:
            raise ValueError("Set the dspace_bins property first . See sim_data.SimData method set_dspace_binning")
        if use_Fhkl_scale and not self.Fhkl_have_scale_factors:
            raise RuntimeError("Set the Fhkl scale factors first! See method self.update_Fhkl_scale_factors")
        return self._ave_I_cell(use_Fhkl_scale, i_channel, use_geometric_mean)

    def Fhkl_restraint_data(self, i_channel=0, restraint_beta=1e12, use_geometric_mean=False, how="ave"):
        """

        :param i_channel:  Fhkl channel, should be 0 unless refining wavelength-dependent Fhkl (e.g. two color)
        :param restraint_beta: restraing the average Fhkl to the initial average Fhkl(per res bin)
        :param use_geometric_mean: boolean flag, whether arithmetic mean or geometric mean is computed per res bin
        :return: (target, gradient vector) , 2-tuple contribution to target and gradient in hopper_utils
        """
        assert how in ["ave", "Friedel", "init"]
        if how=="ave" or how == "init":
            if not self.dspace_bins:
                raise ValueError("Set the dspace_bins property first . See sim_data.SimData method set_dspace_binning")
            assert self.dspace_bins

        if not self.Fhkl_have_scale_factors:
            raise RuntimeError("Set the Fhkl scale factors first! See method self.update_Fhkl_scale_factors")

        flags = {"ave":0, "Friedel": 1, "init": 2}   # controls which underlying restraint method is called in diffBragg

        restraint_data = self._Fhkl_restraint_data(i_channel, restraint_beta, use_geometric_mean, flags[how])
        # the restraint data is always the gradient terms (1 per ASU) followed by the contribution to the target function
        grad_portion = restraint_data[:-1]
        assert grad_portion.shape[0] == self.Num_ASU
        target = restraint_data[-1]
        return target, grad_portion

    def prep_Friedel_restraints(self):
        """
        set the indices of all the positive and negative Fridel mates, for use within diffBragg
        to compute Friedel mate retraints .
        During refinement of Fhkls, Friedel mates should not drift too far apart, especially in the absence
        of anomalous signals
        """
        asu_map = self.get_ASUid_map()
        asu_map_int = {tuple(map(int, k.split(','))): v for k, v in asu_map.items()}
        pos_h = set([h for h in asu_map_int if np.all(np.array(h) > 0)])
        pos_inds = []
        neg_inds = []
        for h in pos_h:
            h_minus = -h[0],-h[1],-h[2]
            if h_minus not in asu_map_int:
                continue
            pos_inds.append(int(asu_map_int[h]))
            neg_inds.append(int(asu_map_int[h_minus]))

        self._set_Friedel_mate_inds(pos_inds, neg_inds)
