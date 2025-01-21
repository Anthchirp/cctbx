from __future__ import absolute_import, division, print_function
from dials.array_family import flex
import math
import numpy as np
import os
import scipy.optimize
from scipy.special import gamma
from scipy.special import polygamma
import scipy.stats
from xfel.merging.application.worker import worker
from xfel.merging.application.reflection_table_utils import reflection_table_utils


class error_modifier_mm24(worker):
  def __init__(self, params, mpi_helper=None, mpi_logger=None):
    super(error_modifier_mm24, self).__init__(params=params, mpi_helper=mpi_helper, mpi_logger=mpi_logger)
    if not self.params.merging.error.mm24.expected_gain is None:
      self.expected_sf = math.sqrt(self.params.merging.error.mm24.expected_gain)
    else:
      self.expected_sf = None
    if self.params.merging.error.mm24.n_max_differences is None:
      self.limit_differences = True
    else:
      self.limit_differences = False

    self.tuning_param = self.params.merging.error.mm24.tuning_param
    self.number_of_intensity_bins = self.params.merging.error.mm24.number_of_intensity_bins
    if self.params.merging.error.mm24.constant_sadd:
      self.cc_key = None
    else:
      if self.params.merging.error.mm24.cc_after_pr:
        self.cc_key = 'correlation_after_post'
      else:
        self.cc_key = 'correlation'

  def __repr__(self):
    return 'Adjust intensity errors -- mm24'

  def run(self, experiments, reflections):
    '''Modify intensity errors according to Mittan-Moreau 202X'''
    assert self.params.merging.error.model == "mm24"
    self.logger.log_step_time("ERROR_MODIFIER_MM24")
    self.logger.log("Modifying intensity errors -- mm24 method (starting with %d reflections)"%(len(reflections)))
    reflections = self.modify_errors(reflections)
    self.logger.log_step_time("ERROR_MODIFIER_MM24", True)
    return experiments, reflections

  def modify_errors(self, reflections):
    # First set up a reflection table to do work downstream.
    reflections = self.setup_work_arrays(reflections)
    # Now moving to intensities, find the bin limits using global min/max of the means of each reflection
    self.calculate_intensity_bin_limits()
    # Once bin limits are determined, assign intensities on each rank to appropriate bin limits
    self.distribute_differences_over_intensity_bins()
    self.initialize_mm24_params()
    # Run LBFGSB minimizer
    #  -- only rank0 does minimization but gradients/functionals are calculated using all rank
    self.run_minimizer()

    if self.params.merging.error.mm24.do_diagnostics:
      self.verify_derivatives()
      self.plot_diagnostics(reflections)
    # Finally update the variances of each reflection as per Eq (10) in Brewster et. al (2019)
    if self.cc_key:
      correlation = reflections[self.cc_key]
    else:
      correlation = None
    reflections['intensity.sum.variance'] = self._get_var_mm24(
      reflections['intensity.sum.variance'],
      reflections['biased_mean'],
      correlation
      )
    del reflections['biased_mean']
    return reflections

  def setup_work_arrays(self, reflections):
    def pairing(k1, k2):
      return int((k1 + k2) * (k1 + k2 + 1) / 2 + k2)

    self.work_table = flex.reflection_table()
    self.refl_biased_means = []
    biased_mean = flex.double() # Go with the pairwise differences in self.work_table
    biased_mean_to_reflections = flex.double() # Put into the original reflection table
    self.biased_mean_count = flex.double() # Used to calculate the number of reflections in each intensity bin
    pairwise_differences = flex.double()
    pairwise_differences_normalized = flex.double()
    counting_stats_var_i = flex.double()
    counting_stats_var_j = flex.double()
    if self.cc_key:
      correlation_i = flex.double()
      correlation_j = flex.double()
    number_of_reflections = 0

    for refls in reflection_table_utils.get_next_hkl_reflection_table(reflections):
      number_of_measurements = refls.size()
      # if the returned "refls" list is empty, it's the end of the input "reflections" list
      if number_of_measurements == 0:
        break
      refls_biased_mean = flex.double(len(refls), flex.mean(refls['intensity.sum.value']))
      biased_mean_to_reflections.extend(refls_biased_mean)
      self.refl_biased_means.append(refls_biased_mean[0])
      if number_of_measurements > self.params.merging.minimum_multiplicity:
        I = refls['intensity.sum.value'].as_numpy_array()
        var_cs = refls['intensity.sum.variance'].as_numpy_array()
        if self.cc_key:
          correlation = refls[self.cc_key].as_numpy_array()
        number_of_reflections += I.size
        self.biased_mean_count.extend(flex.double(I.size, refls_biased_mean[0]))
        indices = np.triu_indices(n=I.size, k=1)
        N = indices[0].size
        if self.limit_differences == False:
          if N > self.params.merging.error.mm24.n_max_differences:
            # random number generation needs to be consistent between symmetry related reflections
            # for reproducibility
            hkl = refls[0]['miller_index_asymmetric']
            # Convert hkl to a hash with Cantor's pairing function.
            # Add 1000 to keep inputs positive.
            hkl_hash = pairing(pairing(hkl[0]+1000, hkl[1]+1000), hkl[2]+1000)
            rng = np.random.default_rng(seed=hkl_hash + self.params.merging.error.mm24.random_seed)
            # Reflections are in different order when run with different numbers of ranks
            sort_indices = np.argsort(I)
            rng.shuffle(sort_indices)
            I = I[sort_indices]
            var_cs = var_cs[sort_indices]
            if self.cc_key:
              correlation = correlation[sort_indices]
            # this option is for performance trade-offs
            if N > 1000:
              subset_indices = rng.choice(
                N,
                size=self.params.merging.error.mm24.n_max_differences,
                replace=False,
                shuffle=False
                )
            else:
              subset_indices = rng.permutation(N)[:self.params.merging.error.mm24.n_max_differences]
            indices = (indices[0][subset_indices], indices[1][subset_indices])
            N = self.params.merging.error.mm24.n_max_differences
        differences = flex.double(np.abs(I[indices[0]] - I[indices[1]]))
        pairwise_differences.extend(differences)
        biased_mean.extend(flex.double(N, refls_biased_mean[0]))
        counting_stats_var_i.extend(flex.double(var_cs[indices[0]]))
        counting_stats_var_j.extend(flex.double(var_cs[indices[1]]))
        if self.cc_key:
          correlation_i.extend(flex.double(correlation[indices[0]]))
          correlation_j.extend(flex.double(correlation[indices[1]]))

    self.work_table['pairwise_differences'] = pairwise_differences
    self.work_table['biased_mean'] = biased_mean
    self.work_table['counting_stats_var_i'] = counting_stats_var_i
    self.work_table['counting_stats_var_j'] = counting_stats_var_j
    if self.cc_key:
      self.work_table['correlation_i'] = correlation_i
      self.work_table['correlation_j'] = correlation_j
    reflections['biased_mean'] = biased_mean_to_reflections

    self.logger.log(f"Number of work reflections selected: {number_of_reflections}")
    return reflections

  def calculate_intensity_bin_limits(self):
    '''Calculate the intensity bins between the 0.5 and 99.5 percentiles'''
    all_biased_means = self.mpi_helper.gather_variable_length_numpy_arrays(
      np.array(self.refl_biased_means, dtype=float), root=0, dtype=float
      )
    if self.mpi_helper.rank == 0:
      all_biased_means = np.sort(all_biased_means)
      lower_percentile = 0.005
      upper_percentile = 0.995
      n = all_biased_means.size
      lower = all_biased_means[int(lower_percentile * n)]
      upper = all_biased_means[int(upper_percentile * n)]
      self.intensity_bin_limits = np.linspace(lower, upper, self.number_of_intensity_bins + 1)
    else:
      self.intensity_bin_limits = np.empty(self.number_of_intensity_bins + 1)
    self.mpi_helper.comm.Bcast(self.intensity_bin_limits, root=0)

  def distribute_differences_over_intensity_bins(self):
    self.intensity_bins = [flex.reflection_table() for i in range(self.number_of_intensity_bins)]
    self.n_differences_in_bin = flex.double(self.number_of_intensity_bins, 0)
    self.n_refls_in_bin = flex.double(self.number_of_intensity_bins, 0)
    self.bin_weighting = flex.double(self.number_of_intensity_bins, 0)
    count = self.work_table.size()
    for bin_index in range(self.number_of_intensity_bins):
      subset_work_table = self.work_table.select(
        (self.work_table['biased_mean'] >= self.intensity_bin_limits[bin_index])
        & (self.work_table['biased_mean'] < self.intensity_bin_limits[bin_index + 1])
        )
      self.intensity_bins[bin_index].extend(subset_work_table)
      self.n_differences_in_bin[bin_index] = self.mpi_helper.comm.allreduce(
        len(subset_work_table), self.mpi_helper.MPI.SUM
        )

      subset_biased_mean = self.biased_mean_count.select(
        (self.biased_mean_count >= self.intensity_bin_limits[bin_index])
        & (self.biased_mean_count < self.intensity_bin_limits[bin_index + 1])
        )
      self.n_refls_in_bin[bin_index] = self.mpi_helper.comm.allreduce(
        len(subset_biased_mean), self.mpi_helper.MPI.SUM
        )
      if self.n_differences_in_bin[bin_index] > 0:
        self.bin_weighting[bin_index] = math.sqrt(self.n_refls_in_bin[bin_index]) / self.n_differences_in_bin[bin_index]

    # for debugging
    number_of_differences_distributed = 0
    for intensity_bin in self.intensity_bins:
      number_of_differences_distributed += intensity_bin.size()
    self.logger.log(
      "Distributed over intensity bins %d out of %d differences"
      % (number_of_differences_distributed, count)
      )

  def initialize_mm24_params(self):
    mean_differences = np.zeros(self.number_of_intensity_bins)
    for bin_index, differences in enumerate(self.intensity_bins):
      summation = self.mpi_helper.comm.reduce(
        flex.sum(differences['pairwise_differences']), op=self.mpi_helper.MPI.SUM, root=0
        )
      counts = self.mpi_helper.comm.reduce(
        len(differences['pairwise_differences']), op=self.mpi_helper.MPI.SUM, root=0
        )
      if self.mpi_helper.rank == 0:
        if counts > 0:
          mean_differences[bin_index] = summation / counts
        else:
          mean_differences[bin_index] = np.nan

    if self.mpi_helper.rank == 0:
      def fitting_equation(params, bin_centers, mean_differences_0, return_jac):
        sf = params[0]
        sadd = params[1]
        prefactor = 2 / np.sqrt(np.pi)
        arg = sf**2 * (bin_centers + sadd**2 * bin_centers**2)
        curve = prefactor * np.sqrt(arg) + mean_differences_0
        if return_jac:
          darg_dsf = 2 * sf * (bin_centers + sadd**2 * bin_centers**2)
          darg_dsadd = 2 * sf**2 * sadd * bin_centers**2
          dcurve_darg = 1/2 * prefactor/np.sqrt(arg)
          dcurve_dsf = dcurve_darg * darg_dsf
          dcurve_dsadd = dcurve_darg * darg_dsadd
          return curve, dcurve_dsf, dcurve_dsadd
        else:
          return curve

      def target_fun_bfgs(params, bin_centers, mean_differences):
        curve, dcurve_dsf, dcurve_dsadd = fitting_equation(params, bin_centers, mean_differences[0], True)
        arg = (curve - mean_differences) / mean_differences
        darg_dcurve = 1 / mean_differences
        loss = 0.5 * np.sum(arg**2)
        dloss_dsf = np.sum(arg * darg_dcurve * dcurve_dsf)
        dloss_dsadd = np.sum(arg * darg_dcurve * dcurve_dsadd)
        return loss, (dloss_dsf, dloss_dsadd)

      def target_fun_scalar(sadd, bin_centers, mean_differences):
        curve = fitting_equation([self.expected_sf, sadd], bin_centers, mean_differences[0], False)
        arg = (curve - mean_differences) / mean_differences
        darg_dcurve = 1 / mean_differences
        loss = 0.5 * np.sum(arg**2)
        return loss

      bin_centers = (self.intensity_bin_limits[1:] + self.intensity_bin_limits[:-1]) / 2
      good_indices = np.invert(np.isnan(mean_differences))
      bin_centers = bin_centers[good_indices]
      mean_differences = mean_differences[good_indices]

      positive_indices = bin_centers > 0
      bin_centers = bin_centers[positive_indices]
      mean_differences = mean_differences[positive_indices]

      if self.cc_key:
        self.sadd = [0, 0.001, 0.001]
        self.sadd[1] = 0.001
        self.sadd[2] = 0.001
      else:
        self.sadd = [0]
      if self.expected_sf is None:
        results = scipy.optimize.minimize(
          target_fun_bfgs,
          x0=(1, 1),
          args=(bin_centers, mean_differences),
          jac=True,
          method='BFGS'
          )
        self.sfac = abs(float(results.x[0]))
        self.sadd[0] = math.sqrt(abs(float(results.x[1])))
        fit_curve = fitting_equation(results.x, bin_centers, mean_differences[0], False)
      else:
        results = scipy.optimize.minimize_scalar(
          target_fun_scalar,
          bounds=(0, 10),
          args=(bin_centers, mean_differences),
          )
        self.sfac = self.expected_sf
        self.sadd[0] = math.sqrt(abs(float(results.x)))

      if self.params.merging.error.mm24.do_diagnostics:
        import matplotlib.pyplot as plt
        fit_curve = fitting_equation([self.sfac, self.sadd[0]], bin_centers, mean_differences[0], False)
        fig, axes = plt.subplots(1, 1, figsize=(5, 3))
        axes.plot(
          bin_centers, mean_differences,
          linestyle='none', marker='.', color=[0, 0, 0], label='Data'
          )
        axes.plot(bin_centers, fit_curve, color=[0, 0.8, 0], label='Initialization')
        axes.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(
          self.params.output.output_dir,
          self.params.output.prefix + '_initial_differences.png'
          ))
        plt.close()

    else:
      self.sfac = 0
      if self.cc_key:
        self.sadd = [0, 0, 0]
      else:
        self.sadd = [0]
    self.sfac = self.mpi_helper.comm.bcast(self.sfac, root=0)
    self.sadd = self.mpi_helper.comm.bcast(self.sadd, root=0)

  def run_minimizer(self):
    from scitbx import lbfgsb

    comm = self.mpi_helper.comm
    MPI = self.mpi_helper.MPI
    size = self.mpi_helper.size
    

    if self.params.merging.error.mm24.tuning_param_opt:
      param_shift = 1
      self.x = flex.double([self.tuning_param, self.sfac, *self.sadd])
    else:
      param_shift = 0
      self.x = flex.double([self.sfac, *self.sadd])
    n_parameters = len(self.x)
    l = flex.double(n_parameters, 1e-8)
    u = flex.double(n_parameters, 0)
    if self.params.merging.error.mm24.tuning_param_opt:
      # normalization for the truncated t-distribution is numerically unstable for nu < 2
      l[0] = 2.5
      if self.x[0] < 2.5:
        self.x[0] = 2.5
    for degree_index in range(param_shift, n_parameters):
      l[degree_index] = -1000
    if self.mpi_helper.rank == 0:
      self.minimizer = lbfgsb.minimizer(
        n = n_parameters,
        l = l,
        u = u,
        nbd = flex.int(n_parameters, 1),
      )
    if self.mpi_helper.rank == 0:
      self.logger.main_log(
        'Initial Parameter Estimates = '
        + f'sfac: {self.sfac} '
        + f'sadd: {self.sadd[0]} '
        + f'nu: {self.tuning_param} '
      )
    while True:
      self.compute_functional_and_gradients()
      status = -1
      if self.mpi_helper.rank == 0:
        if self.minimizer.process(self.x, self.L, self.g):
          if self.params.merging.error.mm24.tuning_param_opt:
            self.tuning_param = self.x[0]
            tuning_param = f'{self.tuning_param:0.3f}'
          self.sfac = self.x[0 + param_shift]
          self.sadd = self.x[1 + param_shift:]
          log_out = 'intermediate minimization results = '\
            + f'loss: {self.L:.2f} '\
            + f'sfac: {self.sfac:0.3f} '\
            + f'sadd: {[f"{value:0.3f}" for value in self.sadd]} '
          if self.params.merging.error.mm24.tuning_param_opt:
            log_out += f'nu: {tuning_param}'
          self.logger.main_log(log_out)
          status = 1
        elif self.minimizer.is_terminated():
          status=0

      comm.Barrier()
      status = comm.bcast(status, root=0)
      if status == 1:
        self.tuning_param = comm.bcast(self.tuning_param, root=0)
        self.sfac = comm.bcast(self.sfac, root=0)
        self.sadd = comm.bcast(self.sadd, root=0)
        pass
      if status==0:
        break

    if self.mpi_helper.rank == 0:
      tuning_param = f'{self.tuning_param:0.3f}'
      log_out = 'FINAL mm24 VALUES = '\
        + f'loss: {self.L:.2f} '\
        + f'sfac: {self.sfac:0.3f} '\
        + f'sadd: {[f"{value:0.3f}" for value in self.sadd]} '
      if self.params.merging.error.mm24.tuning_param_opt:
        log_out += f'nu: {tuning_param}'
      self.logger.main_log(log_out)

  def compute_functional_and_gradients(self):
    self.calculate_functional()
    if self.mpi_helper.rank == 0:
      if self.params.merging.error.mm24.tuning_param_opt:
        self.g = flex.double([self.dL_dnu, self.dL_dsfac, *self.dL_dsadd])
      else:
        self.g = flex.double([self.dL_dsfac, *self.dL_dsadd])

  def verify_derivatives(self):
    shift = 0.000001
    import copy

    self.calculate_functional()
    sfac = copy.copy(self.sfac)
    sadd = copy.copy(self.sadd)
    tuning_param = copy.copy(self.tuning_param)
    if self.mpi_helper.rank == 0:
      TF = copy.copy(self.L)
      der_wrt_sfac = copy.copy(self.dL_dsfac)
      der_wrt_sadd = copy.copy(self.dL_dsadd)
      if self.params.merging.error.mm24.tuning_param_opt:
        der_wrt_nu = copy.copy(self.dL_dnu)

    # Tuning param
    if self.params.merging.error.mm24.tuning_param_opt:
      self.tuning_param = tuning_param * (1 + shift)
      self.calculate_functional()
      if self.mpi_helper.rank == 0:
        TF_p = copy.copy(self.L)
      self.tuning_param = tuning_param * (1 - shift)
      self.calculate_functional()
      if self.mpi_helper.rank == 0:
        TF_m = copy.copy(self.L)
      self.tuning_param = tuning_param
      if self.mpi_helper.rank == 0:
        check_der_wrt_nu = (TF_p - TF_m) / (2 * shift * tuning_param)
        print(f'der_wrt_nu numerical: {check_der_wrt_nu} analytical {der_wrt_nu}')

    # sfac
    self.sfac = sfac * (1 + shift)
    self.calculate_functional()
    if self.mpi_helper.rank == 0:
      TF_p = copy.copy(self.L)
    self.sfac = sfac * (1 - shift)
    self.calculate_functional()
    if self.mpi_helper.rank == 0:
      TF_m = copy.copy(self.L)
    self.sfac = sfac
    if self.mpi_helper.rank == 0:
      check_der_wrt_sfac = (TF_p - TF_m) / (2 * shift * sfac)
      print(f'der_wrt_sfac numerical: {check_der_wrt_sfac} analytical {der_wrt_sfac}')

    # sadd:
    for degree_index in range(len(self.sadd)):
      if sadd[degree_index] == 0:
        self.sadd[degree_index] = shift
      else:
        self.sadd[degree_index] = sadd[degree_index] * (1 + shift)
      self.calculate_functional()
      if self.mpi_helper.rank == 0:
        TF_p = copy.copy(self.L)
      if sadd[degree_index] == 0:
        self.sadd[degree_index] = -shift
      else:
        self.sadd[degree_index] = sadd[degree_index] * (1 - shift)
      self.calculate_functional()
      if self.mpi_helper.rank == 0:
        TF_m = copy.copy(self.L)
      self.sadd[degree_index] = sadd[degree_index]
      if self.mpi_helper.rank == 0:
        if sadd[degree_index] == 0:
          check_der_wrt_sadd = (TF_p - TF_m) / (2 * shift)
        else:
          check_der_wrt_sadd = (TF_p - TF_m) / (2 * shift * sadd[degree_index])
        print(
          f'der_wrt_sadd - degree {degree_index} '
          + f'numerical: {check_der_wrt_sadd} '
          + f'analytical {der_wrt_sadd[degree_index]}'
          )

  def _loss_function_normal(self, differences, var_i, var_j):
    var = var_i + var_j
    z = differences / flex.sqrt(var)
    dz_dvar = -differences / (2 * var**(3/2))
    L1 = 1/2*flex.log(var)
    dL1_dvar = 1/2 * 1/var
    L2 = 1/2 * z**2
    dL2_dz = z
    L = L1 + L2
    dL_dvar_x = dL1_dvar + dL2_dz * dz_dvar
    return L, dL_dvar_x

  def _loss_function_t(self, differences, var_i, var_j):
    v = self.tuning_param
    var = var_i + var_j
    z = differences / flex.sqrt(var)
    dz_dvar = -differences / (2 * var**(3/2))
    arg = 1 + 1/v * z**2
    darg_dz = 2*z/v

    L1 = 1/2 * flex.log(var)
    dL1_dvar = 1/2 * 1/var
    L2 = (v+1)/2 * flex.log(arg)
    dL2_darg = (v+1)/2 * 1/arg
    dL2_dvar = dL2_darg * darg_dz * dz_dvar
    L = L1 + L2
    dL_dvar_x = dL1_dvar + dL2_dvar
    return L, dL_dvar_x

  def _loss_function_t_v_opt(self, differences, var_i, var_j):
    v = self.tuning_param
    var = var_i + var_j
    z = differences / flex.sqrt(var)
    dz_dvar = -differences / (2 * var**(3/2))
    arg = 1 + 1/v * z**2
    darg_dz = 2*z/v
    darg_dv = -z**2 / v**2
    darg_dvar = darg_dz * dz_dvar

    L0 = -math.log(gamma((v+1)/2))
    dL0_dv = -float(polygamma(0, (v+1)/2) * 1/2)

    L1 = 1/2 * math.log(np.pi)

    L2 = 1/2 * math.log(v)
    dL2_dv = 1 / (2*v)

    L3 = math.log(gamma(v/2))
    dL3_dv = float(polygamma(0, v/2) * 1/2)

    L4 = 1/2 * flex.log(var)
    dL4_dvar = 1/2 * 1/var

    L5 = (v+1)/2 * flex.log(arg)
    dL5_dvar = (v+1)/2 * 1/arg * darg_dvar
    dL5_dv = 1/2 * flex.log(arg) + (v+1)/2 * 1/arg * darg_dv

    L = L0 + L1 + L2 + L3 + L4 + L5
    dL_dvar = dL4_dvar + dL5_dvar
    dL_dv = dL0_dv + dL2_dv + dL3_dv + dL5_dv
    return L, dL_dvar, dL_dv

  def _get_sadd2(self, correlation):
    if correlation:
      term1 = flex.exp(-self.sadd[1] * correlation)
      sadd2 = self.sadd[0]**2 * term1 + self.sadd[2]**2
      dsadd2_dsaddi = [
        2 * self.sadd[0] * term1,
        -correlation * self.sadd[0]**2 * term1,
        2 * self.sadd[2] * flex.double(len(correlation), 1)
        ]
    else:
      sadd2 = self.sadd[0]**2
      dsadd2_dsaddi = [2 * self.sadd[0]]
    return sadd2, dsadd2_dsaddi

  def _get_var_mm24(self, counting_err, biased_mean, correlation, return_der=False):
    sadd2, dsadd2_dsaddi = self._get_sadd2(correlation)
    var = self.sfac**2 * (counting_err + sadd2 * biased_mean**2)
    if return_der:
      dvar_dsfac = 2 * self.sfac * (counting_err + sadd2 * biased_mean**2)
      dvar_dsadd2 = self.sfac**2 * biased_mean**2
      return var, dvar_dsfac, dvar_dsadd2, dsadd2_dsaddi
    else:
      return var

  def calculate_functional(self):
    comm = self.mpi_helper.comm
    MPI = self.mpi_helper.MPI
    L_bin_rank = flex.double(self.number_of_intensity_bins, 0)
    dL_dsfac_bin_rank = flex.double(self.number_of_intensity_bins, 0)
    dL_dsadd_bin_rank = [flex.double(self.number_of_intensity_bins, 0) for i in range(len(self.sadd))]
    if self.params.merging.error.mm24.tuning_param_opt:
      dL_dnu_bin_rank = flex.double(self.number_of_intensity_bins, 0)

    for bin_index, differences in enumerate(self.intensity_bins):
      if len(differences) > 0:
        if self.cc_key:
          correlation_i = differences['correlation_i']
          correlation_j = differences['correlation_j']
        else:
          correlation_i = None
          correlation_j = None
        var_i, dvar_i_dsfac, dvar_i_dsadd2, dsadd2_i_dsaddi = self._get_var_mm24(
          differences['counting_stats_var_i'],
          differences['biased_mean'],
          correlation_i,
          return_der=True
          )
        var_j, dvar_j_dsfac, dvar_j_dsadd2, dsadd2_j_dsaddi = self._get_var_mm24(
          differences['counting_stats_var_j'],
          differences['biased_mean'],
          correlation_j,
          return_der=True
          )

        if self.params.merging.error.mm24.likelihood == 'normal':
          L_in_bin, dL_dvar_x = self._loss_function_normal(
            differences['pairwise_differences'], var_i, var_j
            )
        elif self.params.merging.error.mm24.likelihood == 't-dist':
          if self.params.merging.error.mm24.tuning_param_opt:
            L_in_bin, dL_dvar_x, dL_dnu = self._loss_function_t_v_opt(
              differences['pairwise_differences'], var_i, var_j
              )
            dL_dnu_bin_rank[bin_index] = flex.sum(dL_dnu)
          else:
            L_in_bin, dL_dvar_x = self._loss_function_t(
              differences['pairwise_differences'], var_i, var_j
              )

        L_bin_rank[bin_index] = flex.sum(L_in_bin)
        dL_dsfac_bin_rank[bin_index] = flex.sum(dL_dvar_x * (dvar_i_dsfac + dvar_j_dsfac))
        for degree_index in range(len(self.sadd)):
          dL_dsadd_bin_rank[degree_index][bin_index] = flex.sum(dL_dvar_x * (
            dvar_i_dsadd2 * dsadd2_i_dsaddi[degree_index] + dvar_j_dsadd2 * dsadd2_j_dsaddi[degree_index]
            ))

    L_bin = comm.reduce(L_bin_rank, MPI.SUM, root=0)
    dL_dsfac_bin = comm.reduce(dL_dsfac_bin_rank, MPI.SUM, root=0)
    dL_dsadd_bin = [None for i in range(len(self.sadd))]
    for degree_index in range(len(self.sadd)):
      dL_dsadd_bin[degree_index] = comm.reduce(dL_dsadd_bin_rank[degree_index], MPI.SUM, root=0)
    if self.params.merging.error.mm24.tuning_param_opt:
      dL_dnu_bin = comm.reduce(dL_dnu_bin_rank, MPI.SUM, root=0)

    if self.mpi_helper.rank == 0:
      self.L = flex.sum(self.bin_weighting * L_bin)
      self.dL_dsfac = flex.sum(self.bin_weighting * dL_dsfac_bin)
      self.dL_dsadd = [0 for i in range(len(self.sadd))]
      for degree_index in range(len(self.sadd)):
        self.dL_dsadd[degree_index] = flex.sum(self.bin_weighting * dL_dsadd_bin[degree_index])
      if self.params.merging.error.mm24.tuning_param_opt:
        self.dL_dnu = flex.sum(self.bin_weighting * dL_dnu_bin)

  def plot_diagnostics(self, reflections):
    import matplotlib.pyplot as plt
    I_scale = 100000
    def get_rankits(n, down_sample, distribution):
      prob_level = (np.arange(1, n+1) - 0.5) / n
      if distribution == 'half normal':
        return scipy.stats.halfnorm.ppf(prob_level[::down_sample])
      elif distribution == 'half t-dist':
        prob_level = (prob_level + 1) / 2
        return scipy.stats.t.ppf(prob_level[::down_sample], df=self.tuning_param)

    # In intensity bins, the estimated std of I/sigma based on pairwise differences
    # See Absolute pairwise differences: https://en.wikipedia.org/wiki/Robust_measures_of_scale
    # Rousseeuw, Peter J.; Croux, Christophe (December 1993),
    #   "Alternatives to the Median Absolute Deviation",
    #   Journal of the American Statistical Association, 88 (424): 1273–1283
    # This gets the conversion factor from median pairwise differences to standard deviation
    def min_fun_t(c, df):
      term0 = scipy.stats.t.ppf(3/4, df=df)
      term1 = scipy.stats.t.cdf(term0 + 1/c, df=df)
      term2 = scipy.stats.t.cdf(term0 - 1/c, df=df)
      term3 = 1/2
      diff = term1 - term2 - term3
      return diff**2
    if self.mpi_helper.rank == 0:
      if self.params.merging.error.mm24.likelihood == 'normal':
        conversion_factor = 1.1926
      elif self.params.merging.error.mm24.likelihood in ['t-dist']:
        results = scipy.optimize.minimize_scalar(
          min_fun_t,
          bounds=(0.1, 2),
          args=(self.tuning_param)
          )
        conversion_factor = results.x
    else:
      conversion_factor = None
    conversion_factor = self.mpi_helper.comm.bcast(conversion_factor)

    median_differences = [[] for i in range(self.number_of_intensity_bins)]
    for refls in reflection_table_utils.get_next_hkl_reflection_table(reflections):
      number_of_reflections = refls.size()
      if number_of_reflections == 0:
        break
      elif number_of_reflections > 1:
        biased_mean = flex.mean(refls['intensity.sum.value'])
        bin_index = np.searchsorted(self.intensity_bin_limits, biased_mean) - 1
        if biased_mean > self.intensity_bin_limits[0] and biased_mean < self.intensity_bin_limits[-1]:
          I = refls['intensity.sum.value'].as_numpy_array()
          if self.cc_key:
            correlation = refls[self.cc_key]
          else:
            correlation = None
          var_mm24 = self._get_var_mm24(
            refls['intensity.sum.variance'],
            flex.double(len(refls), biased_mean),
            correlation
            ).as_numpy_array()

          # calculate the median difference for the pairwise differences
          differences = np.abs(I[np.newaxis] - I[:, np.newaxis])
          variances = var_mm24[np.newaxis] + var_mm24[:, np.newaxis]
          median_differences[bin_index].append(
            np.median(differences / np.sqrt(variances), axis=1)
            )

    binned_scale = np.zeros(self.number_of_intensity_bins)
    for bin_index in range(self.number_of_intensity_bins):
      if len(median_differences[bin_index]) > 0:
        median_differences[bin_index] = np.concatenate(median_differences[bin_index])
      all_median_differences = self.mpi_helper.gather_variable_length_numpy_arrays(
        np.array(median_differences[bin_index], dtype=float), root=0, dtype=float
        )
      if self.mpi_helper.rank == 0:
        binned_scale[bin_index] = np.median(all_median_differences) * conversion_factor

    # Get all the pairwise differences onto rank 0 for plotting
    pairwise_differences = []
    for bin_index, differences in enumerate(self.intensity_bins):
      if len(differences) > 0:
        if self.cc_key:
          correlation_i = differences['correlation_i']
          correlation_j = differences['correlation_j']
        else:
          correlation_i = None
          correlation_j = None
        var_i = self._get_var_mm24(
          differences['counting_stats_var_i'],
          differences['biased_mean'],
          correlation_i,
          return_der=False
          )
        var_j = self._get_var_mm24(
          differences['counting_stats_var_j'],
          differences['biased_mean'],
          correlation_j,
          return_der=False
          )
        normalized_differences = differences['pairwise_differences'] / flex.sqrt(var_i + var_j)
        pairwise_differences.append(normalized_differences.as_numpy_array())
    all_pairwise_differences = self.mpi_helper.gather_variable_length_numpy_arrays(
        np.concatenate(pairwise_differences), root=0, dtype=float
        )

    if self.mpi_helper.rank == 0:
      sorted_pairwise_differences = np.sort(all_pairwise_differences)
      lim = 5
      downsample = 10000
      grey1 = np.array([99, 102, 106]) / 255
      grey2 = np.array([177, 179, 179]) / 255

      pairwise_differences_bins = np.linspace(0, lim, 101)
      pairwise_differences_db = pairwise_differences_bins[1] - pairwise_differences_bins[0]
      pairwise_differences_centers = (pairwise_differences_bins[1:] + pairwise_differences_bins[:-1]) / 2
      pairwise_differences_hist, _ = np.histogram(
        sorted_pairwise_differences, bins=pairwise_differences_bins, density=True
        )

      fig, axes = plt.subplots(1, 3, figsize=(8, 3))
      axes[0].bar(
        pairwise_differences_centers, pairwise_differences_hist,
        width=pairwise_differences_db, label='$\omega_{hkl}$'
        )
      axes[0].plot(
        pairwise_differences_centers,
        scipy.stats.halfnorm.pdf(pairwise_differences_centers),
        color=grey1, label='Normal'
        )
      if self.params.merging.error.mm24.likelihood in ['t-dist']:
        axes[0].plot(
          pairwise_differences_centers,
          2*scipy.stats.t.pdf(pairwise_differences_centers, df=self.tuning_param),
          color=grey2, label=f't-dist\n$\\nu: ${self.tuning_param:0.1f}'
          )

      axes[0].legend(frameon=False, fontsize=8, handlelength=1)
      axes[0].set_ylabel('Distribution of $\omega_{hbk}$')
      axes[0].set_xlabel('Normalized PD ($\omega_{hbk}$)')
      axes[0].set_xlim([0, 4.5])
      axes[0].set_xticks([0, 1, 2, 3, 4])

      axes[1].plot([0, lim], [0, lim], color=[0, 0, 0], linewidth=1, linestyle=':')
      axes[1].plot(
        sorted_pairwise_differences[::downsample],
        get_rankits(sorted_pairwise_differences.size, downsample, 'half normal'),
        color=grey1
        )
      if self.params.merging.error.mm24.likelihood in ['t-dist']:
        axes[1].plot(
          sorted_pairwise_differences[::downsample],
          get_rankits(sorted_pairwise_differences.size, downsample, 'half t-dist'),
          color=grey2
          )

      axes[1].set_ylim([0, lim])
      axes[1].set_ylabel('Rankits')
      axes[1].set_xlabel('Sorted Normalized PD ($\omega_{hbk}$)')
      axes[1].set_box_aspect(1)
      axes[1].set_xticks([0, 1, 2, 3, 4])
      axes[1].set_yticks([0, 1, 2, 3, 4])
      axes[1].set_xlim([0, 4.5])
      axes[1].set_ylim([0, 4.5])

      intensity_centers = (self.intensity_bin_limits[1:] + self.intensity_bin_limits[:-1]) / 2
      x = intensity_centers / I_scale
      if self.params.merging.error.mm24.likelihood in ['t-dist']:
        v = self.tuning_param
        term0 = v / (v - 2)
        term1 = 4*v / (np.pi * (v - 1)**2)
        term2 = (gamma((v + 1) / 2) / gamma(v / 2))**2
        expected_sigma = np.sqrt(term0 - term1 * term2)
        axes[2].plot([x[0], x[-1]], expected_sigma * np.ones(2), color=grey2)
      axes[2].plot([x[0], x[-1]], np.sqrt(1 - 2/np.pi) * np.ones(2), color=[0, 0, 0])
      axes[2].plot(x, binned_scale, label='STD')

      axes[2].set_ylabel('Standard Deviation of $I_{hbk}/\sigma_{hbk}$')
      axes[2].set_xlabel('Mean Intensity X 100,000')
      fig.tight_layout()
      fig.savefig(os.path.join(
        self.params.output.output_dir,
        self.params.output.prefix + '_PairwiseDifferences.png'
        ))
      plt.close()

    # Get the correlations for later plotting
    if self.cc_key:
      cc_all = self.mpi_helper.gather_variable_length_numpy_arrays(
          np.unique(self.work_table['correlation_i'].as_numpy_array()), root=0, dtype=float
          )
      if self.mpi_helper.rank == 0:
        # CC & sadd plots #
        bins = np.linspace(cc_all.min(), cc_all.max(), 101)
        dbin = bins[1] - bins[0]
        centers = (bins[1:] + bins[:-1]) / 2
        hist_all, _ = np.histogram(cc_all, bins=bins)

        hist_color = np.array([0, 49, 60]) / 256
        line_color = np.array([213, 120, 0]) / 256
        sadd2, _ = self._get_sadd2(flex.double(centers))
        fig, axes_hist = plt.subplots(1, 1, figsize=(5, 3))
        axes_sadd = axes_hist.twinx()
        axes_hist.bar(centers, hist_all / 1000, width=dbin, color=hist_color)
        axes_sadd.plot(centers, self.sfac**2 * sadd2, color=line_color)
        axes_hist.set_xlabel('Correlation Coefficient')
        axes_hist.set_ylabel('Lattices (x1,000)')
        axes_sadd.set_ylabel('$s_{\mathrm{fac}}^2 \\times s_{\mathrm{add}}^2$')
        fig.tight_layout()
        fig.savefig(os.path.join(
          self.params.output.output_dir,
          self.params.output.prefix + '_sadd.png'
          ))
        plt.close()

if __name__ == '__main__':
  from xfel.merging.application.worker import exercise_worker
  exercise_worker(error_modifier)
