""" Tools to solve non-linear L.S. problems formulated with normal-equations.
"""
from __future__ import absolute_import, division, print_function

import libtbx
from scitbx.array_family import flex
from timeit import default_timer as current_time


class journaled_non_linear_ls(object):
  """ A decorator that keeps the history of the objective, gradient,
  step, etc of an underlying normal equations object. An instance of this
  class is a drop-in replacement of that underlying object, with the
  journaling just mentioned automatically happening behind the scene.
  """

  def __init__(self, non_linear_ls, journal, track_gradient, track_step):
    """ Decorate the given normal equations. The history will be accumulated
    in relevant attributes of journal. The flags track_xxx specify whether
    to journal the gradient and/or the step, a potentially memory-hungry
    operation.
    """
    self.actual = non_linear_ls
    self.journal = journal
    self.journal.objective_history = flex.double()
    if track_gradient:
      self.journal.gradient_history = []
    else:
      self.journal.gradient_history = None
    self.journal.gradient_norm_history = flex.double()
    if track_step:
      self.journal.step_history = []
    else:
      self.journal.step_history = None
    self.journal.step_norm_history = flex.double()
    self.journal.parameter_vector_norm_history = flex.double()
    if hasattr(non_linear_ls, "scale_factor"):
      self.journal.scale_factor_history = flex.double()
    else:
      self.journal.scale_factor_history = None
    self.normal_equations_building_time = 0
    self.normal_equations_solving_time = 0

  def __getattr__(self, name):
    return getattr(self.actual, name)

  def build_up(self, objective_only=False):
    t0 = current_time()
    self.actual.build_up(objective_only)
    t1 = current_time()
    self.normal_equations_building_time += t1 - t0
    if objective_only: return
    self.journal.parameter_vector_norm_history.append(
      self.actual.parameter_vector_norm())
    self.journal.objective_history.append(self.actual.objective())
    self.journal.gradient_norm_history.append(
      self.actual.opposite_of_gradient().norm_inf())
    if self.journal.gradient_history is not None:
      self.journal.gradient_history.append(-self.actual.opposite_of_gradient())
    if self.journal.scale_factor_history is not None:
      self.journal.scale_factor_history.append(self.actual.scale_factor())

  def solve(self):
    t0 = current_time()
    self.actual.solve()
    t1 = current_time()
    self.normal_equations_solving_time += t1 - t0
    self.journal.step_norm_history.append(self.actual.step().norm())
    if self.journal.step_history is not None:
      self.journal.step_history.append(self.actual.step().deep_copy())

  def step_forward(self):
    self.actual.step_forward()

  def solve_and_step_forward(self):
    self.solve()
    self.step_forward()


class iterations(object):
  """ Iterations to solve a non-linear L.S. minimisation problem.

  It is assumed that the objective function is properly scaled to be
  between 0 and 1 (or approximately so), so that this class can
  meaningfully thresholds the norm of the gradients to watch for
  convergence.

  The interface expected from the normal equations object passed to __init__
  is that of lstbx.non_linear_normal_equations_mixin

  Use classic stopping criteria: c.f. e.g.
  Methods for non-linear least-squares problems,
  K. Madsen, H.B. Nielsen, O. Tingleff,
  http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3215/pdf/imm3215.pdf

  The do_damping and do_scale_shifts function implements the shelxl damping as
  described here:
  http://shelx.uni-ac.gwdg.de/SHELX/shelx.pdf
  do_scale_shifts aslo compares the maximum shift/esd with
  convergence_as_shift_over_esd and returns boolean to indicate if
  the refinement converged according to this criterion
  These functions do nothing unless called by a derived class
  """

  track_step = False
  track_gradient = False
  track_all = False
  n_max_iterations = 100
  gradient_threshold = None
  step_threshold = None
  verbose_iterations = False

  def __init__(self, non_linear_ls, **kwds):
    """
    """
    libtbx.adopt_optional_init_args(self, kwds)
    if self.track_all: self.track_step = self.track_gradient = True
    self.non_linear_ls = journaled_non_linear_ls(non_linear_ls, self,
                                                 self.track_gradient,
                                                 self.track_step)
    self.do()

  def has_gradient_converged_to_zero(self):
    eps_1 = self.gradient_threshold
    g = self.gradient_norm_history[-1]
    return eps_1 is not None and g <= eps_1

  def had_too_small_a_step(self):
    eps_2 = self.step_threshold
    if eps_2 is None: return False
    x = self.parameter_vector_norm_history[-1]
    h = self.step_norm_history[-1]
    return h <= eps_2*(x + eps_2)

  def do_damping(self, value):
    a = self.non_linear_ls.normal_matrix_packed_u()
    a.matrix_packed_u_diagonal_add_in_place(value*a.matrix_packed_u_diagonal())

  def do_scale_shifts(self, limit_shift_over_su):
    x = self.non_linear_ls.step()
    esd = self.non_linear_ls.covariance_matrix().matrix_packed_u_diagonal()
    ls_shifts_over_su = flex.abs(x/flex.sqrt(esd))
    #max shift for the LS
    self.max_ls_shift_over_su = flex.max(ls_shifts_over_su)
    jac_tr = self.non_linear_ls.actual.\
      reparametrisation.jacobian_transpose_matching_grad_fc()
    self.shifts_over_su = jac_tr.transpose() * ls_shifts_over_su
    self.max_shift_over_su = flex.max(self.shifts_over_su)
    if self.max_shift_over_su < self.convergence_as_shift_over_esd:
      return True
    if self.max_ls_shift_over_su > limit_shift_over_su:
      shift_scale = limit_shift_over_su/self.max_ls_shift_over_su
      x *= shift_scale
    return False

  def do(self):
    raise NotImplementedError


class naive_iterations(iterations):

  def do(self):
    self.n_iterations = 0
    while self.n_iterations < self.n_max_iterations:
      self.non_linear_ls.build_up()
      if self.has_gradient_converged_to_zero(): break
      self.non_linear_ls.solve()
      if self.had_too_small_a_step(): break
      self.non_linear_ls.step_forward()
      self.n_iterations += 1

  def __str__(self):
    return "pure Gauss-Newton"


class naive_iterations_with_damping(iterations):

  damping_value = 0.0007

  def do(self):
    self.n_iterations = 0
    do_last = False
    while self.n_iterations < self.n_max_iterations:
      self.non_linear_ls.build_up()
      if self.has_gradient_converged_to_zero():
        do_last = True
      self.do_damping(self.damping_value)
      self.non_linear_ls.solve()
      step_too_small = self.had_too_small_a_step()
      self.non_linear_ls.step_forward()
      self.n_iterations += 1
      if do_last or step_too_small: break

  def __str__(self):
    return "pure Gauss-Newton with damping"


class naive_iterations_with_damping_and_shift_limit(
  naive_iterations_with_damping):

  max_shift_over_esd = 15
  convergence_as_shift_over_esd = 1e-5

  def do(self):
    self.n_iterations = 0
    do_last = False
    while self.n_iterations < self.n_max_iterations:
      self.non_linear_ls.build_up()
      if self.has_gradient_converged_to_zero():
        do_last = True
      self.do_damping(self.damping_value)
      self.non_linear_ls.solve()
      step_too_small = self.had_too_small_a_step()
      if not step_too_small:
        step_too_small = self.do_scale_shifts(self.max_shift_over_esd)
      self.non_linear_ls.step_forward()
      self.n_iterations += 1
      if do_last or step_too_small: break

  def __str__(self):
    return "pure Gauss-Newton with damping and shift scaling"


class levenberg_marquardt_iterations(iterations):

  tau = 1e-3

  @property
  def mu(self):
    return self._mu

  @mu.setter
  def mu(self, value):
    self.mu_history.append(value)
    self._mu = value

  def do(self):
    self.mu_history = flex.double()
    self.n_iterations = 0
    nu = 2
    self.non_linear_ls.build_up()
    if self.has_gradient_converged_to_zero(): return
    a = self.non_linear_ls.normal_matrix_packed_u()
    self.mu = self.tau*flex.max(a.matrix_packed_u_diagonal())
    while self.n_iterations < self.n_max_iterations:
      a.matrix_packed_u_diagonal_add_in_place(self.mu)
      objective = self.non_linear_ls.objective()
      g = -self.non_linear_ls.opposite_of_gradient()
      self.non_linear_ls.solve()
      if self.had_too_small_a_step(): break
      self.n_iterations += 1
      h = self.non_linear_ls.step()
      expected_decrease = 0.5*h.dot(self.mu*h - g)
      self.non_linear_ls.step_forward()
      self.non_linear_ls.build_up(objective_only=True)
      objective_new = self.non_linear_ls.objective()
      actual_decrease = objective - objective_new
      rho = actual_decrease/expected_decrease
      if rho >= 0:
        if self.has_gradient_converged_to_zero(): break
        self.mu *= max(1/3, 1 - (2*rho - 1)**3)
        nu = 2
      else:
        self.non_linear_ls.step_backward()
        self.mu *= nu
        nu *= 2
      self.non_linear_ls.build_up()

  def __str__(self):
    return "Levenberg-Marquardt"

class levenberg_marquardt_iterations_encapsulated_eqns(
      levenberg_marquardt_iterations):

  objective_decrease_threshold = None

  def __init__(self, non_linear_ls, **kwds):
    iterations.__init__(self, non_linear_ls, **kwds)
    """NKS 7/17/2015 Differs from Luc's original code two ways:
      1) unbreak the encapsulation of the normal matrix; enforce access
         to the normal matrix object through the non_linear_ls interface.
         Luc's original version assumes foreknowledge of the normal matrix
         data structure that will change in future sparse matrix implementation.
      2) avoid a memory leak by deleting the following circular reference to self:
    """
    if self.verbose_iterations:
      print("Iteration      Objective        Mu     ||Gradient||       Step")
    del self.non_linear_ls.journal

  def had_too_small_a_step(self):
    if self.verbose_iterations:
      print("%5d %18.4f"%(self.n_iterations,self.objective_history[-1]), end=' ')
      print("%12.7f"%(self.mu),"%12.3f"%(self.gradient_norm_history[-1]), end=' ')
      x = self.parameter_vector_norm_history[-1]
      h = self.step_norm_history[-1]
      import math
      root = (-x + math.sqrt(x*x+4.*h))/2.
      form_p = int(-math.log10(self.step_threshold))
      format = "%%12.%df"%(form_p+1)
      print(format%(root))
    return iterations.had_too_small_a_step(self)

  def do(self):
    self.mu_history = flex.double()
    self.n_iterations = 0
    nu = 2
    self.non_linear_ls.build_up()
    if self.has_gradient_converged_to_zero(): return
    a_diag = self.non_linear_ls.get_normal_matrix_diagonal()

    self.mu = self.tau*flex.max(a_diag)
    while self.n_iterations < self.n_max_iterations:
      self.non_linear_ls.add_constant_to_diagonal(self.mu)
      objective = self.non_linear_ls.objective()
      g = -self.non_linear_ls.opposite_of_gradient()
      self.non_linear_ls.solve()
      if self.had_too_small_a_step(): break
      self.n_iterations += 1
      h = self.non_linear_ls.step()
      expected_decrease = 0.5*h.dot(self.mu*h - g)
      self.non_linear_ls.step_forward()
      self.non_linear_ls.build_up(objective_only=True)
      objective_new = self.non_linear_ls.objective()
      actual_decrease = objective - objective_new
      rho = actual_decrease/expected_decrease
      if self.objective_decrease_threshold is not None:
        if actual_decrease/objective < self.objective_decrease_threshold: break
      if rho >= 0:
        if self.has_gradient_converged_to_zero(): break
        self.mu *= max(1/3, 1 - (2*rho - 1)**3)
        nu = 2
      else:
        self.non_linear_ls.step_backward()
        self.mu *= nu
        nu *= 2
      self.non_linear_ls.build_up()

  def __str__(self):
    return "Levenberg-Marquardt, with Eigen-based sparse matrix algebra"
