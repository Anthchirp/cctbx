from __future__ import absolute_import, division, print_function
import libtbx.load_env
from scitbx.array_family import flex
from scitbx import sparse
from scitbx.lstbx import normal_eqns, normal_eqns_solving
from libtbx.test_utils import approx_equal, Exception_expected

from scitbx.lstbx.tests import test_problems
from six.moves import range

def exercise_linear_normal_equations():
  py_eqs = [ ( 1, (-1,  0,  0),  1),
             ( 2, ( 2, -1,  0),  3),
             (-1, ( 0,  2,  1),  2),
             (-2, ( 0,  1,  0), -2),
             ]

  eqs_0 = normal_eqns.linear_ls(3)
  for b, a, w in py_eqs:
    eqs_0.add_equation(right_hand_side=b,
                       design_matrix_row=flex.double(a),
                       weight=w)

  eqs_1 = normal_eqns.linear_ls(3)
  b = flex.double()
  w = flex.double()
  a = sparse.matrix(len(py_eqs), 3)
  for i, (b_, a_, w_) in enumerate(py_eqs):
    b.append(b_)
    w.append(w_)
    for j in range(3):
      if a_[j]: a[i, j] = a_[j]
  eqs_1.add_equations(right_hand_side=b, design_matrix=a, weights=w)

  assert approx_equal(
    eqs_0.normal_matrix_packed_u(), eqs_1.normal_matrix_packed_u(), eps=1e-15)
  assert approx_equal(
    eqs_0.right_hand_side(), eqs_1.right_hand_side(), eps=1e-15)
  assert approx_equal(
    list(eqs_0.normal_matrix_packed_u()), [ 13, -6, 0, 9, 4, 2 ], eps=1e-15)
  assert approx_equal(
    list(eqs_0.right_hand_side()), [ 11, -6, -2 ], eps=1e-15)


non_linear_ls_with_separable_scale_factor__impls = (
  normal_eqns.non_linear_ls_with_separable_scale_factor__level_2_blas_impl,
)

try:
  from fast_linalg import env
  if env.initialised:
    non_linear_ls_with_separable_scale_factor__impls += (
      normal_eqns.non_linear_ls_with_separable_scale_factor__level_3_blas_impl,
    )
except ImportError:
  print('Skipping fast_linalg checks')

def exercise_non_linear_ls_with_separable_scale_factor():
  for impl in non_linear_ls_with_separable_scale_factor__impls:
    test = test_problems.polynomial_fit(impl)(normalised=False)
    test.build_up()

    assert test.n_equations == test.n_data;

    # Reference values computed in tst_normal_equations.nb
    eps = 5e-14
    assert approx_equal(test.optimal_scale_factor(), 0.6148971786833856, eps)
    assert approx_equal(test.objective(), 0.039642707534326034, eps)
    assert approx_equal(test.chi_sq(), 0.011326487866950296, eps)


    assert not test.step_equations().solved
    try:
      test.step_equations().cholesky_factor_packed_u()
      raise Exception_expected
    except RuntimeError:
      pass
    try:
      test.step_equations().solution()
      raise Exception_expected
    except RuntimeError:
      pass

    assert approx_equal(
      list(test.step_equations().normal_matrix_packed_u()),
      [ 0.371944193675858, 0.39066546997866547  , 0.10797294655500618,
                             0.41859250354804045, 0.08077629438075473,
                                                  0.19767268057900367 ],
      eps)
    assert approx_equal(
      list(test.step_equations().right_hand_side()),
      [ 0.12149917297914861, 0.13803759252793774, -0.025190641142579157 ],
      eps)

    test.step_equations().solve()

    assert test.step_equations().solved
    try:
      test.step_equations().normal_matrix_packed_u()
      raise Exception_expected
    except RuntimeError:
      pass
    try:
      test.step_equations().right_hand_side()
      raise Exception_expected
    except RuntimeError:
      pass

    assert approx_equal(
      list(test.step_equations().cholesky_factor_packed_u()),
      [ 0.6098722765266986, 0.6405693208478925   ,  0.1770418999366983 ,
                              0.09090351333425013, -0.3589664912436558 ,
                                                    0.19357661121640218 ],
      eps)
    assert approx_equal(
      list(test.step_equations().solution()),
      [ 1.2878697604109028, -0.7727798877778043, -0.5151113342942297 ],
      eps=1e-12)

    test_bis = test_problems.polynomial_fit(impl)(normalised=True)
    test_bis.build_up()
    assert approx_equal(test_bis.objective(),
                        test.objective()/test.sum_w_yo_sq(),
                        eps=1e-15)
    assert approx_equal(test_bis.chi_sq(), test.chi_sq(), eps=1e-15)


def exercise_non_linear_ls_with_separable_scale_factor_plus_penalty():
  for impl in non_linear_ls_with_separable_scale_factor__impls:
    test = test_problems.polynomial_fit_with_penalty(impl)(normalised=False)
    test.build_up()
    assert test.n_equations == test.n_data + 1

    eps = 5e-14
    # reference values from tst_normal_equations.nb again

    assert approx_equal(test.optimal_scale_factor(), 0.6148971786833856, eps)
    redu = test.reduced_problem()
    assert test.objective() == redu.objective()
    assert test.step_equations().right_hand_side()\
           .all_eq(redu.step_equations().right_hand_side())
    assert test.step_equations().normal_matrix_packed_u()\
           .all_eq(redu.step_equations().normal_matrix_packed_u())

    assert approx_equal(test.objective(), 1.3196427075343262, eps)
    assert approx_equal(test.chi_sq(), 0.32991067688358156, eps)
    assert approx_equal(
      test.step_equations().right_hand_side(),
      (1.7214991729791487, -1.4619624074720623, 1.5748093588574208),
      eps)
    assert approx_equal(
      test.step_equations().normal_matrix_packed_u(),
      (1.371944193675858, -0.6093345300213344,  1.107972946555006,
                           1.4185925035480405, -0.9192237056192452,
                                                1.1976726805790037),
      eps)

    test_bis = test_problems.polynomial_fit_with_penalty(impl)(normalised=True)
    test_bis.build_up()
    assert approx_equal(test_bis.chi_sq(), test.chi_sq(), eps=1e-15)

    n_equations = test.n_equations
    test.build_up()
    assert test.n_equations == n_equations

def exercise_levenberg_marquardt(non_linear_ls, plot=False):
  non_linear_ls.restart()
  iterations = normal_eqns_solving.levenberg_marquardt_iterations(
    non_linear_ls,
    track_all=True,
    gradient_threshold=1e-8,
    step_threshold=1e-8,
    tau=1e-4,
    n_max_iterations=200)
  assert non_linear_ls.n_equations == non_linear_ls.n_data
  assert approx_equal(non_linear_ls.x, non_linear_ls.arg_min, eps=5e-4)
  print("L-M: %i iterations" % iterations.n_iterations)
  if plot:
    f = open('plot.nb', 'w')
    print("g=%s;" % iterations.gradient_norm_history.mathematica_form(), file=f)
    print("\\[Mu]=%s;" % iterations.mu_history.mathematica_form(), file=f)
    print("ListLogPlot[{g,\\[Mu]},Joined->True]", file=f)
    f.close()
  non_linear_ls.restart()
  iterations = normal_eqns_solving.levenberg_marquardt_iterations(
    non_linear_ls,
    track_all=True,
    gradient_threshold=0,
    step_threshold=0,
    n_max_iterations=200)
  assert iterations.n_iterations == 200

def run():
  import sys
  plot = '--plot' in sys.argv[1:]
  t = test_problems.exponential_fit()
  exercise_levenberg_marquardt(t, plot)
  exercise_linear_normal_equations()
  exercise_non_linear_ls_with_separable_scale_factor()
  exercise_non_linear_ls_with_separable_scale_factor_plus_penalty()
  print('OK')

if __name__ == '__main__':
  run()
