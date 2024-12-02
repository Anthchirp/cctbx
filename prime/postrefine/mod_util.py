from __future__ import absolute_import, division, print_function
from cctbx.uctbx import unit_cell
from cctbx import miller, crystal, statistics
from cctbx.array_family import flex
from iotbx import mtz
from libtbx.utils import Sorry
import math, os
import numpy as np
from copy import deepcopy
from six.moves import cPickle as pickle
from collections import Counter
from .mod_merge_data import merge_data_handler
from .mod_mx import mx_handler
from .mod_leastsqr import good_unit_cell
from six.moves import range

class intensities_scaler(object):
  """
  Author      : Uervirojnangkoorn, M.
  Created     : 7/13/2014
  Merge equivalent reflections and report intensity and refinement statistics.
  """
  def __init__(self):
    """
    Constructor
    """
    self.CONST_SE_MIN_WEIGHT = 0.17
    self.CONST_SE_MAX_WEIGHT = 1.0
    self.CONST_SIG_I_FACTOR = 1.5

  def write_stat_pickle(self, iparams, stat_dict):
    fname = iparams.run_no+'/stats/pickle_'+str(os.getpid())+'.stat'
    if os.path.isfile(fname):
      pickle_stat = pickle.load(open(fname,"rb"))
      for key in stat_dict:
        if key in pickle_stat:
          pickle_stat[key].append(stat_dict[key][0])
        else:
          pickle_stat[key] = stat_dict[key]
      pickle.dump(pickle_stat, open(fname,"wb"))
    else:
      pickle.dump(stat_dict, open(fname,"wb"))

  def calc_avg_I_cpp(self, prep_output, iparams, avg_mode):
    group_no, group_id_list, miller_index, miller_indices_ori, I, sigI, G, B, p_set, rs_set, wavelength_set, sin_theta_over_lambda_sq, SE, uc_mean, wavelength_mean, pickle_filename_set, txt_out = prep_output
    from prime import Average_Mode, averaging_engine
    if avg_mode == 'average': avg_mode_cpp = Average_Mode.Average
    elif avg_mode == 'weighted': avg_mode_cpp = Average_Mode.Weighted
    elif avg_mode == 'final': avg_mode_cpp = Average_Mode.Final
    else: raise Sorry("Bad averaging mode selected: %s"%avg_mode)
    sigma_max = iparams.sigma_rejection
    engine = averaging_engine(group_no, group_id_list, miller_index, miller_indices_ori, I, sigI, G, B,p_set, rs_set, wavelength_set, sin_theta_over_lambda_sq, SE, pickle_filename_set)
    engine.avg_mode = avg_mode_cpp
    engine.sigma_max = sigma_max
    engine.flag_volume_correction = iparams.flag_volume_correction
    engine.n_rejection_cycle = iparams.n_rejection_cycle
    engine.flag_output_verbose = iparams.flag_output_verbose
    results = engine.calc_avg_I()
    mdh = merge_data_handler()
    mdh.extend_data(results.miller_index, results.I_avg, results.sigI_avg, (results.r_meas_top, results.r_meas_btm, results.multiplicity), (results.I_avg_even, results.I_avg_odd, results.I_avg_even_h, results.I_avg_odd_h, results.I_avg_even_k, results.I_avg_odd_k, results.I_avg_even_l, results.I_avg_odd_l), uc_mean, wavelength_mean)
    return mdh, results.txt_obs_out, results.txt_reject_out

  def calc_mean_unit_cell(self, results):
    uc_array = [list(pres.uc_params) for pres in results if pres is not None]
    return np.mean(uc_array,0), np.median(uc_array,0), np.std(uc_array,0)

  def calc_mean_postref_parameters(self, results):
    params_array = [[pres.G, pres.B, pres.ry, pres.rz, pres.re, pres.r0, \
        pres.voigt_nu, pres.rotx, pres.roty, pres.R_final, pres.R_xy_final, pres.SE] \
        for pres in results if (pres is not None and not math.isnan(pres.G) and not math.isnan(pres.B) \
        and not math.isnan(pres.ry) and not math.isnan(pres.rz) and not math.isnan(pres.re) and not math.isnan(pres.r0) \
        and not math.isnan(pres.voigt_nu) and not math.isnan(pres.rotx) and not math.isnan(pres.roty) \
        and not math.isnan(pres.R_final) and not math.isnan(pres.R_xy_final) and not math.isnan(pres.SE))]
    return np.mean(params_array,0), np.median(params_array,0), np.std(params_array,0)

  def prepare_output(self, results, iparams, avg_mode):
    if avg_mode == 'average':
      cc_thres = 0
    else:
      cc_thres = iparams.frame_accept_min_cc
    std_filter = iparams.sigma_rejection
    if iparams.flag_weak_anomalous:
      if avg_mode == 'final':
        target_anomalous_flag = iparams.target_anomalous_flag
      else:
        target_anomalous_flag = False
    else:
      target_anomalous_flag = iparams.target_anomalous_flag
    pr_params_mean, pr_params_med, pr_params_std = self.calc_mean_postref_parameters(results)
    G_mean, B_mean, ry_mean, rz_mean, re_mean, r0_mean, voigt_nu_mean, rotx_mean, roty_mean, R_mean, R_xy_mean, SE_mean = pr_params_mean
    G_med, B_med, ry_med, rz_med, re_med, r0_med, voigt_nu_med, rotx_med, roty_med, R_med, R_xy_med, SE_med = pr_params_med
    G_std, B_std, ry_std, rz_std, re_std, r0_std, voigt_nu_std, rotx_std, roty_std, R_std, R_xy_std, SE_std = pr_params_std
    #prepare data for merging
    miller_indices_all = flex.miller_index()
    miller_indices_ori_all = flex.miller_index()
    I_all = flex.double()
    sigI_all = flex.double()
    G_all = flex.double()
    B_all = flex.double()
    p_all = flex.double()
    rx_all = flex.double()
    rs_all = flex.double()
    rh_all = flex.double()
    SE_all = flex.double()
    sin_sq_all = flex.double()
    wavelength_all = flex.double()
    detector_distance_set = flex.double()
    R_init_all = flex.double()
    R_final_all = flex.double()
    R_xy_init_all = flex.double()
    R_xy_final_all = flex.double()
    pickle_filename_all = flex.std_string()
    filtered_results = []
    cn_good_frame, cn_bad_frame_SE, cn_bad_frame_uc, cn_bad_frame_cc, cn_bad_frame_G, cn_bad_frame_re = (0,0,0,0,0,0)
    crystal_orientation_dict = {}
    for pres in results:
      if pres is not None:
        pickle_filepath = pres.pickle_filename.split('/')
        img_filename = pickle_filepath[len(pickle_filepath)-1]
        flag_pres_ok = True
        #check SE, CC, UC, G, B, gamma_e
        if math.isnan(pres.G):
          flag_pres_ok = False
        if math.isnan(pres.SE) or np.isinf(pres.SE):
          flag_pres_ok = False
        if flag_pres_ok and SE_std > 0:
          if abs(pres.SE-SE_med)/SE_std > std_filter:
            flag_pres_ok = False
            cn_bad_frame_SE += 1
        if flag_pres_ok and pres.CC_final < cc_thres:
          flag_pres_ok = False
          cn_bad_frame_cc += 1
        if flag_pres_ok:
          if G_std > 0:
            if abs(pres.G-G_med)/G_std > std_filter:
              flag_pres_ok = False
              cn_bad_frame_G += 1
        if flag_pres_ok:
          if re_std > 0:
            if abs(pres.re-re_med)/re_std > std_filter:
              flag_pres_ok = False
              cn_bad_frame_re += 1
        if flag_pres_ok and not good_unit_cell(pres.uc_params, iparams, iparams.merge.uc_tolerance):
          flag_pres_ok = False
          cn_bad_frame_uc += 1
        data_size = pres.observations.size()
        if flag_pres_ok:
          cn_good_frame += 1
          filtered_results.append(pres)
          R_init_all.append(pres.R_init)
          R_final_all.append(pres.R_final)
          R_xy_init_all.append(pres.R_xy_init)
          R_xy_final_all.append(pres.R_xy_final)
          miller_indices_all.extend(pres.observations.indices())
          miller_indices_ori_all.extend(pres.observations_original.indices())
          I_all.extend(pres.observations.data())
          sigI_all.extend(pres.observations.sigmas())
          G_all.extend(flex.double([pres.G] * data_size))
          B_all.extend(flex.double([pres.B] * data_size))
          p_all.extend(pres.partiality)
          rs_all.extend(pres.rs_set)
          rh_all.extend(pres.rh_set)
          sin_sq_all.extend(pres.observations.two_theta(wavelength=pres.wavelength).sin_theta_over_lambda_sq().data())
          SE_all.extend(flex.double([pres.SE]*data_size))
          wavelength_all.extend(flex.double([pres.wavelength]*data_size))
          detector_distance_set.append(pres.detector_distance_mm)
          pickle_filename_all.extend(flex.std_string([pres.pickle_filename] * data_size))
          crystal_orientation_dict[pres.pickle_filename] = pres.crystal_orientation
    #plot stats
    self.plot_stats(filtered_results, iparams)
    #write out updated crystal orientation as a pickle file
    if not iparams.flag_hush: pickle.dump(crystal_orientation_dict, open(iparams.run_no+'/'+"crystal.o","wb"),pickle.HIGHEST_PROTOCOL)
    #calculate average unit cell
    uc_mean, uc_med, uc_std = self.calc_mean_unit_cell(filtered_results)
    unit_cell_mean = unit_cell(tuple(uc_mean))
    #recalculate stats for pr parameters
    pr_params_mean, pr_params_med, pr_params_std = self.calc_mean_postref_parameters(filtered_results)
    G_mean, B_mean, ry_mean, rz_mean, re_mean, r0_mean, voigt_nu_mean, rotx_mean, roty_mean, R_mean, R_xy_mean, SE_mean = pr_params_mean
    G_med, B_med, ry_med, rz_med, re_med, r0_med, voigt_nu_med, rotx_med, roty_med, R_med, R_xy_med, SE_med = pr_params_med
    G_std, B_std, ry_std, rz_std, re_std, r0_std, voigt_nu_std, rotx_std, roty_std, R_std, R_xy_std, SE_std = pr_params_std
    #from all observations merge them
    crystal_symmetry = crystal.symmetry(
        unit_cell=tuple(uc_mean),
        space_group_symbol=str(iparams.target_space_group))
    miller_set_all=miller.set(
                crystal_symmetry=crystal_symmetry,
                indices=miller_indices_all,
                anomalous_flag=target_anomalous_flag)
    miller_array_all = miller_set_all.array(
              data=I_all,
              sigmas=sigI_all).set_observation_type_xray_intensity()
    #sort reflections according to asymmetric-unit symmetry hkl
    perm = miller_array_all.sort_permutation(by_value="packed_indices")
    miller_indices_all_sort = miller_array_all.indices().select(perm)
    miller_indices_ori_all_sort = miller_indices_ori_all.select(perm)
    I_obs_all_sort = miller_array_all.data().select(perm)
    sigI_obs_all_sort = miller_array_all.sigmas().select(perm)
    G_all_sort = G_all.select(perm)
    B_all_sort = B_all.select(perm)
    p_all_sort = p_all.select(perm)
    rs_all_sort = rs_all.select(perm)
    wavelength_all_sort = wavelength_all.select(perm)
    sin_sq_all_sort = sin_sq_all.select(perm)
    SE_all_sort = SE_all.select(perm)
    pickle_filename_all_sort = pickle_filename_all.select(perm)
    miller_array_uniq = miller_array_all.merge_equivalents().array().complete_array(d_min=iparams.merge.d_min, d_max=iparams.merge.d_max)
    matches_uniq = miller.match_multi_indices(
                  miller_indices_unique=miller_array_uniq.indices(),
                  miller_indices=miller_indices_all_sort)
    pair_0 = flex.int([pair[0] for pair in matches_uniq.pairs()])
    pair_1 = flex.int([pair[1] for pair in matches_uniq.pairs()])
    group_id_list = flex.int([pair_0[pair_1[i]] for i in range(len(matches_uniq.pairs()))])
    tally = Counter()
    for elem in group_id_list: tally[elem] += 1
    cn_group = len(tally)
    #preparte txt out stat
    txt_out = 'Summary of refinement and merging\n'
    txt_out += ' No. good frames:          %12.0f\n'%(cn_good_frame)
    txt_out += ' No. bad cc frames:        %12.0f\n'%(cn_bad_frame_cc)
    txt_out += ' No. bad G frames) :       %12.0f\n'%(cn_bad_frame_G)
    txt_out += ' No. bad unit cell frames: %12.0f\n'%(cn_bad_frame_uc)
    txt_out += ' No. bad gamma_e frames:   %12.0f\n'%(cn_bad_frame_re)
    txt_out += ' No. bad SE:               %12.0f\n'%(cn_bad_frame_SE)
    txt_out += ' No. observations:         %12.0f\n'%(len(I_obs_all_sort))
    txt_out += 'Mean target value (BEFORE: Mean Median (Std.))\n'
    txt_out += ' post-refinement:          %12.2f %12.2f (%9.2f)\n'%(np.mean(R_init_all), np.median(R_init_all), np.std(R_init_all))
    txt_out += ' (x,y) restraints:         %12.2f %12.2f (%9.2f)\n'%(np.mean(R_xy_init_all), np.median(R_xy_init_all), np.std(R_xy_init_all))
    txt_out += 'Mean target value (AFTER: Mean Median (Std.))\n'
    txt_out += ' post-refinement:          %12.2f %12.2f (%9.2f)\n'%(np.mean(R_final_all), np.median(R_final_all), np.std(R_final_all))
    txt_out += ' (x,y) restraints:         %12.2f %12.2f (%9.2f)\n'%(np.mean(R_xy_final_all), np.median(R_xy_final_all), np.std(R_xy_final_all))
    txt_out += ' SE:                       %12.2f %12.2f (%9.2f)\n'%(SE_mean, SE_med, SE_std)
    txt_out += ' G:                        %12.3e %12.3e (%9.2e)\n'%(G_mean, G_med, G_std)
    txt_out += ' B:                        %12.2f %12.2f (%9.2f)\n'%(B_mean, B_med, B_std)
    txt_out += ' Rot.x:                    %12.2f %12.2f (%9.2f)\n'%(rotx_mean*180/math.pi, rotx_med*180/math.pi, rotx_std*180/math.pi)
    txt_out += ' Rot.y:                    %12.2f %12.2f (%9.2f)\n'%(roty_mean*180/math.pi, roty_med*180/math.pi, roty_std*180/math.pi)
    txt_out += ' gamma_y:                  %12.5f %12.5f (%9.5f)\n'%(ry_mean, ry_med, ry_std)
    txt_out += ' gamma_z:                  %12.5f %12.5f (%9.5f)\n'%(rz_mean, rz_med, rz_std)
    txt_out += ' gamma_0:                  %12.5f %12.5f (%9.5f)\n'%(r0_mean, r0_med, r0_std)
    txt_out += ' gamma_e:                  %12.5f %12.5f (%9.5f)\n'%(re_mean, re_med, re_std)
    txt_out += ' voigt_nu:                 %12.5f %12.5f (%9.5f)\n'%(voigt_nu_mean, voigt_nu_med, voigt_nu_std)
    txt_out += ' unit cell\n'
    txt_out += '   a:                      %12.2f %12.2f (%9.2f)\n'%(uc_mean[0], uc_med[0], uc_std[0])
    txt_out += '   b:                      %12.2f %12.2f (%9.2f)\n'%(uc_mean[1], uc_med[1], uc_std[1])
    txt_out += '   c:                      %12.2f %12.2f (%9.2f)\n'%(uc_mean[2], uc_med[2], uc_std[2])
    txt_out += '   alpha:                  %12.2f %12.2f (%9.2f)\n'%(uc_mean[3], uc_med[3], uc_std[3])
    txt_out += '   beta:                   %12.2f %12.2f (%9.2f)\n'%(uc_mean[4], uc_med[4], uc_std[4])
    txt_out += '   gamma:                  %12.2f %12.2f (%9.2f)\n'%(uc_mean[5], uc_med[5], uc_std[5])
    txt_out += 'Parmeters from integration (not-refined)\n'
    txt_out += '  Wavelength:              %12.5f %12.5f (%9.5f)\n'%(np.mean(wavelength_all), np.median(wavelength_all), np.std(wavelength_all))
    txt_out += '  Detector distance:       %12.5f %12.5f (%9.5f)\n'%(np.mean(detector_distance_set), np.median(detector_distance_set), np.std(detector_distance_set))
    txt_out += '* (standard deviation)\n'
    #write out stat. pickle
    if not iparams.flag_hush:
      stat_dict = {"n_frames_good": [cn_good_frame], \
                   "n_frames_bad_cc": [cn_bad_frame_cc], \
                   "n_frames_bad_G": [cn_bad_frame_G], \
                   "n_frames_bad_uc": [cn_bad_frame_uc], \
                   "n_frames_bad_gamma_e": [cn_bad_frame_re], \
                   "n_frames_bad_SE": [cn_bad_frame_SE], \
                   "n_observations": [len(I_obs_all_sort)], \
                   "R_start": [np.mean(R_init_all)], \
                   "R_end": [np.mean(R_final_all)], \
                   "R_xy_start": [np.mean(R_xy_init_all)], \
                   "R_xy_end": [np.mean(R_xy_final_all)], \
                   "mean_gamma_y": [ry_mean], \
                   "std_gamma_y": [ry_std], \
                   "mean_gamma_z": [rz_mean], \
                   "std_gamma_z": [rz_std], \
                   "mean_gamma_0": [r0_mean], \
                   "std_gamma_0": [r0_std], \
                   "mean_gamma_e": [re_mean], \
                   "std_gamma_e": [re_std], \
                   "mean_voigt_nu": [voigt_nu_mean], \
                   "std_voigt_nu": [voigt_nu_std], \
                   "mean_a": [uc_mean[0]], \
                   "std_a": [uc_std[0]], \
                   "mean_b": [uc_mean[1]], \
                   "std_b": [uc_std[1]], \
                   "mean_c": [uc_mean[2]], \
                   "std_c": [uc_std[2]], \
                   "mean_alpha": [uc_mean[3]], \
                   "std_alpha": [uc_std[3]], \
                   "mean_beta": [uc_mean[4]], \
                   "std_beta": [uc_std[4]], \
                   "mean_gamma": [uc_mean[5]], \
                   "std_gamma": [uc_std[5]]}
      self.write_stat_pickle(iparams, stat_dict)
    return cn_group, group_id_list, miller_indices_all_sort, miller_indices_ori_all_sort, \
           I_obs_all_sort, sigI_obs_all_sort,G_all_sort, B_all_sort, \
           p_all_sort, rs_all_sort, wavelength_all_sort, sin_sq_all_sort, SE_all_sort, uc_mean, \
           np.mean(wavelength_all), pickle_filename_all_sort, txt_out

  def write_output(self, mdh, iparams, output_mtz_file_prefix, avg_mode):
    if iparams.flag_weak_anomalous:
      if avg_mode == 'final':
        target_anomalous_flag = iparams.target_anomalous_flag
      else:
        target_anomalous_flag = False
    else:
      target_anomalous_flag = iparams.target_anomalous_flag
    uc_mean = mdh.uc_mean
    wavelength_mean = mdh.wavelength_mean
    #output mtz file and report binning stat
    miller_set_merge = crystal.symmetry(
          unit_cell=unit_cell(tuple(uc_mean)),
          space_group_symbol=str(iparams.target_space_group)
        ).build_miller_set(
          anomalous_flag=target_anomalous_flag,
          d_min=iparams.merge.d_min)
    mdh.generate_miller_array_from_miller_set(miller_set_merge, target_anomalous_flag)
    miller_array_complete = miller_set_merge.array()
    fake_data = flex.double([1.0]*len(miller_array_complete.indices()))
    miller_array_template_asu = miller_array_complete.customized_copy(data=fake_data, \
              sigmas=fake_data).resolution_filter(d_min=iparams.merge.d_min, \
              d_max=iparams.merge.d_max)
    n_refl_all = mdh.get_size()
    #do another resolution filter here
    i_sel_res = mdh.miller_array_merge.resolution_filter_selection(d_min=iparams.merge.d_min, d_max=iparams.merge.d_max)
    mdh.reduce_by_selection(i_sel_res)
    n_refl_out_resolutions = n_refl_all - mdh.get_size()
    #remove outliers
    sequences = flex.int(range(mdh.get_size()))
    good_sequences = []
    for i_rejection in range(iparams.n_rejection_cycle):
      binner_merge = mdh.miller_array_merge.setup_binner(n_bins=200)
      for i_bin in range(1, 201):
        i_binner = (binner_merge.bin_indices() == i_bin)
        I_obs_bin = mdh.miller_array_merge.data().select(i_binner)
        sequences_bin = sequences.select(i_binner)
        if len(I_obs_bin) > 0:
          I_obs_bin = mdh.miller_array_merge.data().select(i_binner)
          try:
            i_filter = flex.abs((I_obs_bin - np.median(I_obs_bin))/np.std(I_obs_bin)) < 10
          except Exception as e:
            print("Warning: outlier rejection by bins failed because of floating point.")
            print(e)
            i_filter = flex.bool([True]*len(I_obs_bin))
          good_sequences.extend(list(sequences_bin.select(i_filter)))
    mdh.reduce_by_selection(flex.size_t(good_sequences))
    n_refl_outliers = n_refl_all - n_refl_out_resolutions - mdh.get_size()
    #get iso if given.
    mxh = mx_handler()
    flag_hklisoin_found, miller_array_iso = mxh.get_miller_array_from_reflection_file(iparams.hklisoin)
    #write output files
    if output_mtz_file_prefix != '':
      #write as mtz file
      miller_array_merge_unique = mdh.miller_array_merge.merge_equivalents().array()
      info = miller.array_info(wavelength=wavelength_mean)
      miller_array_merge_unique.set_info(info)
      mtz_dataset_merge = miller_array_merge_unique.as_mtz_dataset(column_root_label="IOBS")
      mtz_dataset_merge.mtz_object().write(file_name=output_mtz_file_prefix+'_merge.mtz')
      #write as cns file
      f_cns = open(output_mtz_file_prefix+'_merge.hkl', 'w')
      miller_array_merge_unique.export_as_cns_hkl(file_object=f_cns)
      f_cns.close()
    #calculate merging stat table
    if True:
      #calculate isotropic B-factor
      try:
        mxh = mx_handler()
        asu_contents = mxh.get_asu_contents(iparams.n_residues)
        observations_as_f = mdh.miller_array_merge.as_amplitude_array()
        observations_as_f.setup_binner(auto_binning=True)
        wp = statistics.wilson_plot(observations_as_f, asu_contents, e_statistics=True)
        B_merged = wp.wilson_b
      except Exception as e:
        B_merged = 0
        print("Warning: b-factor calculation in mod_util failed. Reset b-factor to 0")
        print(e)
      #report binning stats
      txt_out = '\n'
      txt_out += 'Isotropic B-factor:  %7.2f\n'%(B_merged)
      txt_out += 'No. of reflections\n'
      txt_out += ' all:                %7.0f\n'%(n_refl_all)
      txt_out += ' outside resolution: %7.0f\n'%(n_refl_out_resolutions)
      txt_out += ' outliers:           %7.0f\n'%(n_refl_outliers)
      txt_out += ' total left:         %7.0f\n'%(mdh.get_size())
      txt_out += 'Summary for '+output_mtz_file_prefix+'_merge.mtz\n'
      txt_out += 'Bin Resolution Range     Completeness      <N_obs> |Rmerge  Rsplit   CC1/2   N_ind |CCiso   N_ind|CCanoma  N_ind| <I/sigI>   <I>    <sigI>    <I**2>\n'
      txt_out += '--------------------------------------------------------------------------------------------------------------------------------------------------\n'
      #for stat pickle
      sp_res,sp_complete,sp_n_obs,sp_cc12,sp_cc12_anom,sp_rmerge,sp_i_o_sigi,sp_isqr = ([],[],[],[],[],[],[],[])
      #binning
      binner_template_asu = miller_array_template_asu.setup_binner(n_bins=iparams.n_bins)
      binner_template_asu_indices = binner_template_asu.bin_indices()
      #for stats on axis cones
      mdh_astar = deepcopy(mdh)
      mdh_bstar = deepcopy(mdh)
      mdh_cstar = deepcopy(mdh)
      mdh_astar.reduce_to_cone_on_axis((1,0,0), iparams.percent_cone_fraction)
      mdh_bstar.reduce_to_cone_on_axis((0,1,0), iparams.percent_cone_fraction)
      mdh_cstar.reduce_to_cone_on_axis((0,0,1), iparams.percent_cone_fraction)
      #prepare text out for axis cones
      txt_out_cone = 'Summary of CC1/2 on three crystal axes\n'
      txt_out_cone += 'Bin Resolution Range           CC1/2                      <I>                          N_refl           \n'
      txt_out_cone += '                        a*      b*      c*  |      a*        b*       c*    |    a*      b*     c*      \n'
      txt_out_cone += '---------------------------------------------------------------------------------------------------------\n'
      for i in range(1,iparams.n_bins+1):
        i_binner = (binner_template_asu_indices == i)
        miller_indices_template_bin = miller_array_template_asu.indices().select(i_binner)
        #for all reflections
        mdh_bin = deepcopy(mdh)
        mdh_bin.reduce_by_miller_index(miller_indices_template_bin)
        cc12, n_refl_cc12 = mdh_bin.get_cc12()
        cciso, n_refl_cciso = mdh_bin.get_cciso(miller_array_iso)
        cc_anom_acentric, n_refl_anom_acentric = mdh_bin.get_cc_anom()
        completeness = (mdh_bin.get_size()/len(miller_indices_template_bin))*100
        multiplicity = mdh_bin.get_multiplicity()
        txt_out += '%02d %7.2f - %7.2f %5.1f %6.0f / %6.0f %7.2f %7.2f %7.2f %7.2f %6.0f %7.2f %6.0f %7.2f %6.0f %8.2f %10.1f %8.1f %6.2f\n' \
            %(i, binner_template_asu.bin_d_range(i)[0], binner_template_asu.bin_d_range(i)[1], \
            completeness, \
            mdh_bin.get_size(), len(miller_indices_template_bin),\
            multiplicity, mdh_bin.get_r_meas()*100, mdh_bin.get_r_split()*100, \
            cc12*100, n_refl_cc12, cciso*100, n_refl_cciso, \
            cc_anom_acentric, n_refl_anom_acentric, \
            mdh_bin.get_mean_IoversigI(), mdh_bin.get_mean_I(), mdh_bin.get_mean_sigI(), mdh_bin.get_second_moment())
        #for reflections on cones
        mdh_astar_bin = deepcopy(mdh_astar)
        mdh_astar_bin.reduce_by_miller_index(miller_indices_template_bin)
        cc12_astar, n_refl_cc12_astar = mdh_astar_bin.get_cc12()
        mdh_bstar_bin = deepcopy(mdh_bstar)
        mdh_bstar_bin.reduce_by_miller_index(miller_indices_template_bin)
        cc12_bstar, n_refl_cc12_bstar = mdh_bstar_bin.get_cc12()
        mdh_cstar_bin = deepcopy(mdh_cstar)
        mdh_cstar_bin.reduce_by_miller_index(miller_indices_template_bin)
        cc12_cstar, n_refl_cc12_cstar = mdh_cstar_bin.get_cc12()
        txt_out_cone += '%02d %7.2f - %7.2f %7.2f %7.2f %7.2f %10.1f %10.1f %10.1f %6.0f %6.0f %6.0f\n' \
            %(i, binner_template_asu.bin_d_range(i)[0], binner_template_asu.bin_d_range(i)[1], \
            cc12_astar*100, cc12_bstar*100, cc12_cstar*100, \
            mdh_astar_bin.get_mean_I(), mdh_bstar_bin.get_mean_I(), mdh_cstar_bin.get_mean_I(), \
            n_refl_cc12_astar, n_refl_cc12_bstar, n_refl_cc12_cstar)
        #for stat pickle
        sp_res.append(binner_template_asu.bin_d_range(i)[1])
        sp_complete.append(completeness)
        sp_n_obs.append(multiplicity)
        sp_cc12.append(cc12)
        sp_cc12_anom.append(cc_anom_acentric)
        sp_rmerge.append(mdh_bin.get_r_meas()*100)
        sp_i_o_sigi.append(mdh_bin.get_mean_IoversigI())
        sp_isqr.append(mdh.get_second_moment())
      #txt out total for all reflections
      cc12, n_refl_cc12 = mdh.get_cc12()
      cciso, n_refl_cciso = mdh.get_cciso(miller_array_iso)
      cc_anom_acentric, n_refl_anom_acentric = mdh.get_cc_anom()
      txt_out += '--------------------------------------------------------------------------------------------------------------------------------------------------\n'
      txt_out += '        TOTAL        %5.1f %6.0f / %6.0f %7.2f %7.2f %7.2f %7.2f %6.0f %7.2f %6.0f %7.2f %6.0f %8.2f %10.1f %8.1f %6.2f\n' \
      %((mdh.get_size()/miller_array_template_asu.size())*100, \
          mdh.get_size(), miller_array_template_asu.size(),\
          mdh.get_multiplicity(), mdh.get_r_meas()*100, mdh.get_r_split()*100, \
          cc12*100, n_refl_cc12, cciso*100, n_refl_cciso, \
          cc_anom_acentric, n_refl_anom_acentric, \
          mdh.get_mean_IoversigI(), mdh.get_mean_I(), mdh.get_mean_sigI(), mdh.get_second_moment())
      txt_out += '--------------------------------------------------------------------------------------------------------------------------------------------------\n'
      txt_out += '\n'
      #txt out total for reflections on cones
      cc12_astar, n_refl_cc12_astar = mdh_astar.get_cc12()
      cc12_bstar, n_refl_cc12_bstar = mdh_bstar.get_cc12()
      cc12_cstar, n_refl_cc12_cstar = mdh_cstar.get_cc12()
      txt_out_cone += '----------------------------------------------------------------------------------------------------------\n'
      txt_out_cone += '       total         %7.2f %7.2f %7.2f %10.1f %10.1f %10.1f %6.0f %6.0f %6.0f\n' \
            %(cc12_astar*100, cc12_bstar*100, cc12_cstar*100, \
            mdh_astar.get_mean_I(), mdh_bstar.get_mean_I(), mdh_cstar.get_mean_I(), \
            n_refl_cc12_astar, n_refl_cc12_bstar, n_refl_cc12_cstar)
      txt_out_cone += '----------------------------------------------------------------------------------------------------------\n'
      txt_out_cone += '\n'
      txt_out_table1 = "Table1 ("+avg_mode+")\n"
      txt_out_table1 += "  Space group: "+str(mdh.miller_array_merge.space_group_info())+"\n"
      txt_out_table1 += "  Cell dimensions: %6.2f, %6.2f, %6.2f, %6.2f, %6.2f, %6.2f\n"%tuple(mdh.uc_mean)
      txt_out_table1 += "  Resolution (A): %6.2f - %6.2f (%6.2f - %6.2f)\n"%(mdh.miller_array_merge.d_max_min()[0], mdh.miller_array_merge.d_max_min()[1], sp_res[-2], sp_res[-1])
      txt_out_table1 += "  Rmerge: %6.2f (%6.2f)\n"%(mdh.get_r_meas()*100, sp_rmerge[-1])
      txt_out_table1 += "  CC1/2: %6.2f (%6.2f)\n"%(mdh.get_cc12()[0]*100, sp_cc12[-1])
      txt_out_table1 += "  I/sigI: %6.2f (%6.2f)\n"%(mdh.get_mean_IoversigI(), sp_i_o_sigi[-1])
      txt_out_table1 += "  Completeness (%%): %6.2f (%6.2f)\n"%((mdh.get_size()/miller_array_template_asu.size())*100, sp_complete[-1])
      txt_out_table1 += "  Redundancy: %6.2f (%6.2f)\n"%(mdh.get_multiplicity(), sp_n_obs[-1])
      #save data for stat. pickle in stat_dict
      if not iparams.flag_hush:
        stat_dict = {"binned_resolution": [sp_res], \
            "binned_completeness": [sp_complete], \
            "binned_n_obs": [sp_n_obs], \
            "binned_cc12": [sp_cc12], \
            "binned_cc12_anom": [sp_cc12_anom], \
            "binned_rmerge": [sp_rmerge], \
            "binned_i_o_sigi": [sp_i_o_sigi], \
            "binned_isqr": [sp_isqr], \
            "total_res_max": [mdh.miller_array_merge.d_max_min()[0]], \
            "total_res_min": [mdh.miller_array_merge.d_max_min()[1]], \
            "total_completeness": [(mdh.get_size()/miller_array_template_asu.size())*100], \
            "total_n_obs": [mdh.get_multiplicity()], \
            "total_cc12": [mdh.get_cc12()[0]*100], \
            "total_rmerge": [mdh.get_r_meas()*100], \
            "total_i_o_sigi": [mdh.get_mean_IoversigI()], \
            "space_group_info": [mdh.miller_array_merge.space_group_info()], \
            }
        self.write_stat_pickle(iparams, stat_dict)
      txt_out += txt_out_cone + txt_out_table1
    return mdh, txt_out

  def plot_stats(self, results, iparams):
    #retrieve stats from results and plot them
    if iparams.flag_plot or iparams.flag_output_verbose:
      #for plotting set n_bins = 5 to avoid empty bin
      n_bins_plot = 5
      #get expected f^2
      try:
        mxh = mx_handler()
        asu_contents = mxh.get_asu_contents(iparams.n_residues)
        observations_as_f = results[0].observations.as_amplitude_array()
        binner_template_asu = observations_as_f.setup_binner(n_bins=n_bins_plot)
        wp = statistics.wilson_plot(observations_as_f, asu_contents, e_statistics=True)
        expected_f_sq = wp.expected_f_sq
        mean_stol_sq = wp.mean_stol_sq
      except Exception:
        expected_f_sq = flex.double([0]*n_bins_plot)
        mean_stol_sq = flex.double(range(n_bins_plot))
        print("Warning: Wilson plot calculation in plot stats failed.")
      #setup list
      params_array = np.array([[pres.R_init, pres.R_final, pres.R_xy_init, pres.R_xy_final, \
          pres.G, pres.B, pres.rotx*180/math.pi, pres.roty*180/math.pi, \
          pres.ry, pres.rz, pres.r0, pres.re, pres.voigt_nu, \
          pres.uc_params[0], pres.uc_params[1], pres.uc_params[2], \
          pres.uc_params[3], pres.uc_params[4], pres.uc_params[5], \
          pres.CC_final, pres.pickle_filename] for pres in results])
      params = ['Rinit','Rfinal','Rxyinit', 'Rxyfinal', \
          'G','B','rot_x','rot_y','gamma_y','gamma_z','gamma_0','gamma_e','voigtnu' , \
          'a','b','c','alpha','beta','gamma','CC','Filename']
    #keep parameter history if verbose is selected
    if iparams.flag_output_verbose:
      fileseq_list = flex.int()
      for file_in in os.listdir(iparams.run_no):
        if file_in.endswith('.paramhist'):
          file_split = file_in.split('.')
          fileseq_list.append(int(file_split[0]))
      if len(fileseq_list) == 0:
        new_fileseq = 0
      else:
        new_fileseq = flex.max(fileseq_list) + 1
      newfile_name = str(new_fileseq) + '.paramhist'
      txt_out_verbose = '\n'.join([' '.join(p) for p in params_array])
      f = open(iparams.run_no+'/'+newfile_name, 'w')
      f.write(txt_out_verbose)
      f.close()
    #plotting
    if iparams.flag_plot:
      try:
        import matplotlib.pyplot as plt
      except Exception as e:
        print("Warning: error importing matplotlib.pyplot")
        print(e)
        return
      n_rows = 3
      n_cols = int(math.ceil(len(params)/n_rows))
      num_bins = 10
      for i in range(len(params)-1):
        tmp_params = params_array[:,i].astype(float)
        plt.subplot(n_rows,n_cols,i+1)
        plt.hist(tmp_params, num_bins, normed=0, facecolor='green', alpha=0.5)
        plt.ylabel('Frequencies')
        plt.title(params[i]+'\nmu %5.1f med %5.1f sigma %5.1f' %(np.mean(tmp_params), np.median(tmp_params), np.std(tmp_params)))
      plt.show()

  def combine_pre_merge(self, result, iparams):
    mi_all = flex.miller_index()
    mio_all = flex.miller_index()
    I_all = flex.double()
    sigI_all = flex.double()
    G_all = flex.double()
    B_all = flex.double()
    p_all = flex.double()
    rs_all = flex.double()
    wavelength_all = flex.double()
    sin_all = flex.double()
    SE_all = flex.double()
    uc_mean_set = []
    wavelength_mean_set = []
    pickle_filename_all = flex.std_string()
    for res in result:
      for prep_output in res:
        _, _, mi, mio, I, sigI, G, B, p, rs, wavelength, sin, SE, uc_mean, wavelength_mean, pickle_filename_set, txt_out = prep_output
        mi_all.extend(mi)
        mio_all.extend(mio)
        I_all.extend(I)
        sigI_all.extend(sigI)
        G_all.extend(G)
        B_all.extend(B)
        p_all.extend(p)
        rs_all.extend(rs)
        wavelength_all.extend(wavelength)
        sin_all.extend(sin)
        SE_all.extend(SE)
        uc_mean_set.extend(uc_mean)
        wavelength_mean_set.append(wavelength_mean)
        pickle_filename_all.extend(pickle_filename_set)
    uc_mean = np.mean(np.array(uc_mean_set).reshape(-1,6), axis=0)
    wavelength_mean = np.mean(wavelength_mean_set)
    ms_template = crystal.symmetry(
        unit_cell=tuple(uc_mean),
        space_group_symbol=str(iparams.target_space_group)
        ).build_miller_set(
        anomalous_flag=iparams.target_anomalous_flag,
        d_min=iparams.merge.d_min)
    ma_all = ms_template.array().customized_copy(indices=mi_all, data=I_all, sigmas=sigI_all)
    #sort reflections according to asymmetric-unit symmetry hkl
    perm = ma_all.sort_permutation(by_value="packed_indices")
    mi_all_sort = mi_all.select(perm)
    mio_all_sort = mio_all.select(perm)
    I_all_sort = I_all.select(perm)
    sigI_all_sort = sigI_all.select(perm)
    G_all_sort = G_all.select(perm)
    B_all_sort = B_all.select(perm)
    p_all_sort = p_all.select(perm)
    rs_all_sort = rs_all.select(perm)
    wavelength_all_sort = wavelength_all.select(perm)
    sin_all_sort = sin_all.select(perm)
    SE_all_sort = SE_all.select(perm)
    pickle_filename_all_sort = pickle_filename_all.select(perm)
    ma_uniq = ma_all.merge_equivalents().array().complete_array(d_min=iparams.merge.d_min, d_max=iparams.merge.d_max)
    matches_uniq = miller.match_multi_indices(
                  miller_indices_unique=ma_uniq.indices(),
                  miller_indices=mi_all_sort)
    pair_0 = flex.int([pair[0] for pair in matches_uniq.pairs()])
    pair_1 = flex.int([pair[1] for pair in matches_uniq.pairs()])
    group_id_list = flex.int([pair_0[pair_1[i]] for i in range(len(matches_uniq.pairs()))])
    tally = Counter()
    for elem in group_id_list: tally[elem] += 1
    cn_group = len(tally)
    return cn_group, group_id_list, mi_all_sort, mio_all_sort, \
           I_all_sort, sigI_all_sort, G_all_sort, B_all_sort, \
           p_all_sort, rs_all_sort, wavelength_all_sort, sin_all_sort, SE_all_sort, uc_mean, \
           wavelength_mean, pickle_filename_all_sort, ""
