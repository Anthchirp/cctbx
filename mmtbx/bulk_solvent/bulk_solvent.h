#ifndef MMTBX_BULK_SOLVENT_BULK_SOLVENT_H
#define MMTBX_BULK_SOLVENT_BULK_SOLVENT_H

#include <cfloat>
#include <mmtbx/error.h>
#include <mmtbx/import_scitbx_af.h>
#include <cctbx/sgtbx/space_group.h>
#include <scitbx/array_family/accessors/c_grid.h>
#include <mmtbx/f_model/f_model.h>
#include <scitbx/matrix/outer_product.h>
#include <scitbx/array_family/versa_algebra.h>
#include <scitbx/array_family/shared_algebra.h>
#include <scitbx/array_family/small_algebra.h>
#include <scitbx/matrix/eigensystem.h>
#include <scitbx/matrix/packed.h>
#include <scitbx/array_family/versa_matrix.h>
#include <scitbx/math/cubic_equation.h>

namespace mmtbx { namespace detail {

template <typename FloatType=double,
          typename ComplexType=std::complex<double> >
class d_f_model_d_p_one_h
{
public:
  d_f_model_d_p_one_h(
    f_model::core<FloatType,ComplexType> const& fm, std::size_t i)
  {
    // Note: ss = s**2/4
    grad_p.resize(fm.n_shells(), 0.);
    curv_p.resize(fm.n_shells(), 0.);
    ComplexType f_model      = fm.f_model_no_aniso_scale[i];//fm.f_model[i];
    FloatType   f_model_abs  = std::abs(f_model);
    ComplexType f_model_conj = std::conj(f_model);
    FloatType   f_model_abs_sq = f_model_abs*f_model_abs;
    FloatType   f_model_abs_qb = f_model_abs*f_model_abs_sq;
    FloatType k_aniso = 1.;//fm.k_anisotropic[i];
    FloatType ss = fm.ss[i];
    if(f_model_abs > 0){
      for(unsigned short j=0; j<fm.n_shells(); ++j) {
        ComplexType f_mask        = fm.shell_f_mask(j)[i];
        ComplexType f_mask_conj   = std::conj(f_mask);
        FloatType   f_mask_abs    = std::abs(f_mask);
        FloatType   f_mask_abs_sq = f_mask_abs*f_mask_abs;
        ComplexType zvs_zsv = f_model*f_mask_conj+f_model_conj*f_mask;
        FloatType   theta = std::real(zvs_zsv/(2*f_model_abs));
        grad_p[j] = theta;
        FloatType   omega = std::real(
          (4*f_model_abs_sq*f_mask_abs_sq-zvs_zsv*zvs_zsv)/(4*f_model_abs_qb) );
        curv_p[j] = omega;
      }
    }
  }
  af::shared<FloatType> grad_p;
  af::shared<FloatType> curv_p;
};

//------------------------------------------------------------------------------

template <typename FloatType=double,
          typename ComplexType=std::complex<double> >
class d_f_model_d_k_sol_and_d_b_sol_one_h
{
public:
  d_f_model_d_k_sol_and_d_b_sol_one_h(
    f_model::core<FloatType,ComplexType> const& fm, std::size_t i)
  {
    // Note: ss = s**2/4
    grad_k_sols.resize(fm.n_shells(), 0.);
    grad_b_sols.resize(fm.n_shells(), 0.);
    curv_k_sols.resize(fm.n_shells(), 0.);
    curv_b_sols.resize(fm.n_shells(), 0.);
    ComplexType f_model      = fm.f_model_no_aniso_scale[i];//fm.f_model[i];
    FloatType   f_model_abs  = std::abs(f_model);
    ComplexType f_model_conj = std::conj(f_model);
    FloatType   f_model_abs_sq = f_model_abs*f_model_abs;
    FloatType   f_model_abs_qb = f_model_abs*f_model_abs_sq;
    FloatType k_aniso = 1.;//fm.k_anisotropic[i];
    FloatType ss = fm.ss[i];
    if(f_model_abs > 0){
      for(unsigned short j=0; j<fm.n_shells(); ++j) {
        FloatType k_sol = fm.k_sol(j);
        FloatType b_sol = fm.b_sol(j);
        ComplexType f_mask        = fm.shell_f_mask(j)[i];
        ComplexType f_mask_conj   = std::conj(f_mask);
        FloatType f_mask_abs    = std::abs(f_mask);
        FloatType f_mask_abs_sq = f_mask_abs*f_mask_abs;
        FloatType f_b = f_model::f_b_exp_one_h<FloatType>(ss, b_sol);
        ComplexType zvs_zsv = f_model*f_mask_conj+f_model_conj*f_mask;
        FloatType theta = std::real(zvs_zsv/(2*f_model_abs));
        grad_k_sols[j] =  k_aniso * theta * f_b;
        grad_b_sols[j] = -k_aniso * theta * k_sol * ss * f_b;

        FloatType omega = std::real(
          (4*f_model_abs_sq*f_mask_abs_sq-zvs_zsv*zvs_zsv)/(4*f_model_abs_qb) );

        curv_k_sols[j] = k_aniso * omega * f_b * f_b;
        curv_b_sols[j] = k_aniso * k_sol*ss*ss*f_b*(omega*k_sol*f_b+theta);
      }
    }
  }
  af::shared<FloatType> grad_k_sols;
  af::shared<FloatType> grad_b_sols;
  af::shared<FloatType> curv_k_sols;
  af::shared<FloatType> curv_b_sols;
};

template <typename FloatType>
scitbx::af::tiny<FloatType, 6>
d_f_model_d_u_star_one_h(FloatType const& f_model_abs,
                         cctbx::miller::index<> const& mi)
{
  scitbx::af::tiny<FloatType, 6> result;
  FloatType minus_two_pi = -2.0*scitbx::constants::pi*scitbx::constants::pi;
  FloatType coeff = f_model_abs * minus_two_pi;
  result = scitbx::af::tiny<FloatType, 6> (
    coeff *    mi[0]*mi[0],
    coeff *    mi[1]*mi[1],
    coeff *    mi[2]*mi[2],
    coeff * 2.*mi[0]*mi[1],
    coeff * 2.*mi[0]*mi[2],
    coeff * 2.*mi[1]*mi[2]);
  return result;
};

template <typename FloatType=double>
class one_h_ls_u_star
{

FloatType fo;
FloatType f_model_abs_no_k_total;
FloatType overall_scale;
FloatType k_anisotropic;
cctbx::miller::index<> miller_index;

public:
  one_h_ls_u_star(
    FloatType const& fo_,
    FloatType const& f_model_abs_no_k_total_,
    cctbx::miller::index<> const& miller_index_,
    FloatType const& k_anisotropic_,
    FloatType const& overall_scale_)
  :
  fo(fo_), f_model_abs_no_k_total(f_model_abs_no_k_total_),
  miller_index(miller_index_), overall_scale(overall_scale_),
  k_anisotropic(k_anisotropic_)
  {
    FloatType k_total = overall_scale*k_anisotropic;
    diff = fo - k_total * f_model_abs_no_k_total;
    FloatType mtsd = -2. * k_total * diff;
    scitbx::af::tiny<FloatType, 6>
      usg = d_f_model_d_u_star_one_h(f_model_abs_no_k_total, miller_index);
    for(std::size_t j=0; j<6; j++) grad_u_star[j] = mtsd * usg[j];
  }

  FloatType diff;
  scitbx::af::tiny<FloatType, 6> grad_u_star;

};

template <typename FloatType=double,
          typename ComplexType=std::complex<double> >
class one_h_ls
{

FloatType fo;
f_model::core<FloatType,ComplexType> fm;
std::size_t i;
FloatType scale;
FloatType scale_all;
FloatType mtsd;

public:
  one_h_ls(FloatType const& fo_,
           f_model::core<FloatType,ComplexType> const& fm_,
           std::size_t const& i_,
           FloatType const& scale_)
  :
  fo(fo_), fm(fm_), i(i_), scale(scale_)
  {
    FloatType f_model_abs = std::abs(fm.f_model_no_aniso_scale[i]);
    scale_all = scale * fm.k_anisotropic[i];
    diff = fo - scale_all * f_model_abs;
    mtsd = -2. * scale_all * diff;
  }

  void compute_kbp_grad_curv(bool compute_kb_grad,
                            bool compute_kb_curv,
                            bool compute_p_grad,
                            bool compute_p_curv)
  {
    MMTBX_ASSERT(compute_kb_grad || compute_kb_curv);
    if(compute_kb_grad) {
      grad_k_sols.resize(fm.n_shells(), 0.);
      grad_b_sols.resize(fm.n_shells(), 0.);
    }
    if(compute_kb_curv) {
      curv_k_sols.resize(fm.n_shells(), 0.);
      curv_b_sols.resize(fm.n_shells(), 0.);
    }
    FloatType tss = 2*scale_all*scale_all;
    // kb sol
    if(compute_kb_grad || compute_kb_curv) {
      d_f_model_d_k_sol_and_d_b_sol_one_h<FloatType,ComplexType> r =
        d_f_model_d_k_sol_and_d_b_sol_one_h<FloatType,ComplexType> (fm,i);
      for(std::size_t j=0; j<grad_k_sols.size(); ++j) {
        FloatType gk = r.grad_k_sols[j];
        FloatType gb = r.grad_b_sols[j];
        if(compute_kb_grad) {
          grad_k_sols[j] = mtsd*gk;
          grad_b_sols[j] = mtsd*gb;
        }
        if(compute_kb_curv) {
          curv_k_sols[j] = tss*gk*gk + mtsd*r.curv_k_sols[j];
          curv_b_sols[j] = tss*gb*gb + mtsd*r.curv_b_sols[j];
        }
      }
    }
    // p
    if(compute_p_grad || compute_p_curv) {
      d_f_model_d_p_one_h<FloatType,ComplexType> r =
        d_f_model_d_p_one_h<FloatType,ComplexType> (fm,i);
      for(std::size_t j=0; j<grad_k_sols.size(); ++j) {
        FloatType gp = r.grad_p[j];
        if(compute_p_grad) {
          grad_p_sols[j] = mtsd*gp;
        }
        if(compute_p_curv) {
          curv_p_sols[j] = tss*gp*gp + mtsd*r.curv_p[j];
        }
      }
    }
  }

  void compute_u_star_grad()
  {
    FloatType f_model_abs = std::abs(fm.f_model_no_aniso_scale[i]);
    cctbx::miller::index<> mi = fm.hkl[i];
    scitbx::af::tiny<FloatType, 6> usg = d_f_model_d_u_star_one_h(f_model_abs, mi);
    for(std::size_t j=0; j<6; j++) grad_u_star[j] = mtsd * usg[j];
  }

  FloatType diff;
  scitbx::af::tiny<FloatType, 6> grad_u_star;
  scitbx::af::shared<FloatType> grad_k_sols;
  scitbx::af::shared<FloatType> grad_b_sols;
  scitbx::af::shared<FloatType> grad_p_sols;
  scitbx::af::shared<FloatType> curv_k_sols;
  scitbx::af::shared<FloatType> curv_b_sols;
  scitbx::af::shared<FloatType> curv_p_sols;
};

template <typename FloatType=double,
          typename ComplexType=std::complex<double> >
class overall_scale
{
public:
  overall_scale() {}
  overall_scale(af::const_ref<FloatType> const& fo,
                f_model::core<FloatType,ComplexType> const& fm)
  {
    FloatType num=0.;
    FloatType denum=0.;
    sum_fo_sq = 0.;
    for(std::size_t i=0; i < fo.size(); i++) {
      FloatType f_model_abs = std::abs(fm.f_model[i]);
      num += fo[i] * f_model_abs;
      denum += f_model_abs * f_model_abs;
      sum_fo_sq += fo[i]*fo[i];
    }
    scale = (denum == 0 ? 0 : num/denum);
  }
  overall_scale(af::const_ref<FloatType> const& fo,
                f_model::core<FloatType,ComplexType> const& fm1,
                f_model::core<FloatType,ComplexType> const& fm2,
                FloatType const& twin_fraction)
  {
    FloatType num=0.;
    FloatType denum=0.;
    sum_fo_sq = 0.;
    for(std::size_t i=0; i < fo.size(); i++) {
      FloatType f_model_abs1 = std::abs(fm1.f_model[i]);
      FloatType f_model_abs2 = std::abs(fm2.f_model[i]);
      FloatType f_model_abs = std::sqrt(
        (1-twin_fraction)*f_model_abs1*f_model_abs1+
           twin_fraction *f_model_abs2*f_model_abs2);
      num += fo[i] * f_model_abs;
      denum += f_model_abs * f_model_abs;
      sum_fo_sq += fo[i]*fo[i];
    }
    scale = (denum == 0 ? 0 : num/denum);
  }
  FloatType scale, sum_fo_sq;
};

}}

namespace mmtbx { namespace bulk_solvent {

//------------------------------------------------------------------------------
template <typename FloatType>
FloatType
scale(af::const_ref< std::complex<FloatType> > const& fo,
      af::const_ref< std::complex<FloatType> > const& fc)
{
    MMTBX_ASSERT(fo.size()==fc.size());
    FloatType num=0.0;
    FloatType denum=0.0;
    for(std::size_t i=0; i < fo.size(); i++) {
      FloatType fc_abs = std::abs(fc[i]);
      FloatType fo_abs = std::abs(fo[i]);
      num   += fo_abs * fc_abs;
      denum += fc_abs * fc_abs;
    }
    return (denum == 0 ? 0 : num/denum);
};

template <typename FloatType, typename ComplexType>
FloatType
scale(af::const_ref<FloatType> const& fo,
      af::const_ref<ComplexType> const& fc)
{
    MMTBX_ASSERT(fo.size()==fc.size());
    FloatType num=0.0;
    FloatType denum=0.0;
    for(std::size_t i=0; i < fo.size(); i++) {
      FloatType fc_abs = std::abs(fc[i]);
      num += fo[i] * fc_abs;
      denum += fc_abs * fc_abs;
    }
    return (denum == 0 ? 0 : num/denum);
};

template <typename FloatType, typename ComplexType>
FloatType
scale(af::const_ref<FloatType> const& fo,
      af::const_ref<ComplexType> const& fc,
      af::const_ref<bool> const& selection)
{
    MMTBX_ASSERT(fo.size()==fc.size());
    MMTBX_ASSERT(fo.size()==selection.size());
    FloatType num=0.0;
    FloatType denum=0.0;
    for(std::size_t i=0; i < fo.size(); i++) {
      if(selection[i]) {
        FloatType fc_abs = std::abs(fc[i]);
        num += fo[i] * fc_abs;
        denum += fc_abs * fc_abs;
      }
    }
    return (denum == 0 ? 0 : num/denum);
};

template <typename FloatType>
FloatType
scale(af::const_ref<FloatType> const& fo,
      af::const_ref<FloatType> const& fc)
{
    MMTBX_ASSERT(fo.size()==fc.size());
    FloatType num=0.0;
    FloatType denum=0.0;
    for(std::size_t i=0; i < fo.size(); i++) {
      num += fo[i] * fc[i];
      denum += fc[i] * fc[i];
    }
    return (denum == 0 ? 0 : num/denum);
};

template <typename FloatType, typename ComplexType>
FloatType
scale(
  af::const_ref<FloatType> const& fo,
  af::const_ref< std::complex<ComplexType> > const& fc1,
  af::const_ref< std::complex<ComplexType> > const& fc2,
  FloatType const& twin_fraction)
{
  MMTBX_ASSERT(fo.size()==fc1.size());
  MMTBX_ASSERT(fo.size()==fc2.size());
  af::shared<FloatType> fc_abs(fo.size());
  for(std::size_t i=0; i < fo.size(); i++) {
    FloatType fc_abs1 = std::abs(fc1[i]);
    FloatType fc_abs2 = std::abs(fc2[i]);
    fc_abs[i]=std::sqrt((1-twin_fraction)*fc_abs1*fc_abs1+
                           twin_fraction *fc_abs2*fc_abs2);
  }
  return scale(fo,fc_abs.const_ref());
};

template <typename FloatType>
FloatType
r_factor(
  af::const_ref<FloatType> const& fo,
  af::const_ref<FloatType> const& fc,
  FloatType const& scale)
{
  MMTBX_ASSERT(fo.size()==fc.size());
  FloatType num=0.0;
  FloatType denum=0.0;
  for(std::size_t i=0; i < fo.size(); i++) {
    num += std::abs(fo[i] - fc[i] * scale);
    denum += fo[i];
  }
  if(denum == 0) return 1.e+9;
  return num/denum;
};

template <typename FloatType>
FloatType
r_factor(
  af::const_ref< std::complex<FloatType> > const& fo,
  af::const_ref< std::complex<FloatType> > const& fc,
  FloatType const& scale)
{
  MMTBX_ASSERT(fo.size()==fc.size());
  FloatType num=0.0;
  FloatType denum=0.0;
  for(std::size_t i=0; i < fo.size(); i++) {
    num += std::abs(std::abs(fo[i]) - std::abs(fc[i]) * scale);
    denum += std::abs(fo[i]);
  }
  if(denum == 0) return 1.e+9;
  return num/denum;
};

template <typename FloatType>
FloatType
r_factor(
  af::const_ref<FloatType> const& fo,
  af::const_ref< std::complex<FloatType> > const& fc,
  FloatType const& scale)
{
  MMTBX_ASSERT(fo.size()==fc.size());
  FloatType num=0.0;
  FloatType denum=0.0;
  for(std::size_t i=0; i < fo.size(); i++) {
    num += std::abs(fo[i] - std::abs(fc[i]) * scale);
    denum += fo[i];
  }
  if(denum == 0) return 1.e+9;
  return num/denum;
};

template <typename FloatType>
FloatType
r_factor(
  af::const_ref<FloatType> const& fo,
  af::const_ref< std::complex<FloatType> > const& fc,
  af::const_ref<bool> const& selection,
  FloatType const& scale)
{
  MMTBX_ASSERT(fo.size()==fc.size());
  MMTBX_ASSERT(fo.size()==selection.size());
  FloatType num=0.0;
  FloatType denum=0.0;
  for(std::size_t i=0; i < fo.size(); i++) {
    if(selection[i]) {
      num += std::abs(fo[i] - std::abs(fc[i]) * scale);
      denum += fo[i];
    }
  }
  if(denum == 0) return 1.e+9;
  return num/denum;
};

template <typename FloatType>
FloatType
r_factor(
  af::const_ref<FloatType> const& fo,
  af::const_ref<FloatType> const& fc)
{
  MMTBX_ASSERT(fo.size()==fc.size());
  FloatType sc = scale(fo,fc);
  return r_factor(fo,fc,sc);
};

template <typename FloatType, typename ComplexType>
FloatType
r_factor(
  af::const_ref<FloatType> const& fo,
  af::const_ref<std::complex<ComplexType> > const& fc)
{
  MMTBX_ASSERT(fo.size()==fc.size());
  FloatType sc = scale(fo,fc);
  return r_factor(fo,fc,sc);
};

template <typename FloatType, typename ComplexType>
FloatType
r_factor(
  af::const_ref<FloatType> const& fo,
  af::const_ref<std::complex<ComplexType> > const& fc,
  af::const_ref<bool> const& selection)
{
  MMTBX_ASSERT(fo.size()==fc.size());
  MMTBX_ASSERT(fo.size()==selection.size());
  FloatType sc = scale(fo,fc,selection);
  return r_factor(fo,fc,selection,sc);
};

template <typename FloatType, typename ComplexType>
FloatType
r_factor(
  af::const_ref<FloatType> const& fo,
  af::const_ref< std::complex<ComplexType> > const& fc1,
  af::const_ref< std::complex<ComplexType> > const& fc2,
  FloatType const& twin_fraction)
{
  MMTBX_ASSERT(fo.size()==fc1.size());
  MMTBX_ASSERT(fo.size()==fc2.size());
  af::shared<FloatType> fc_abs(fo.size());
  for(std::size_t i=0; i < fo.size(); i++) {
    FloatType fc_abs1 = std::abs(fc1[i]);
    FloatType fc_abs2 = std::abs(fc2[i]);
    fc_abs[i]=std::sqrt((1-twin_fraction)*fc_abs1*fc_abs1+
                           twin_fraction *fc_abs2*fc_abs2);
  }
  FloatType sc = scale(fo,fc_abs.const_ref());
  return r_factor(fo,fc_abs.const_ref(),sc);
};

template <typename FloatType, typename ComplexType>
FloatType
r_factor(
  af::const_ref<FloatType> const& fo,
  af::const_ref< std::complex<ComplexType> > const& fc1,
  af::const_ref< std::complex<ComplexType> > const& fc2,
  FloatType const& twin_fraction,
  FloatType const& scale)
{
  MMTBX_ASSERT(fo.size()==fc1.size());
  MMTBX_ASSERT(fo.size()==fc2.size());
  af::shared<FloatType> fc_abs(fo.size());
  for(std::size_t i=0; i < fo.size(); i++) {
    FloatType fc_abs1 = std::abs(fc1[i]);
    FloatType fc_abs2 = std::abs(fc2[i]);
    fc_abs[i]=std::sqrt((1-twin_fraction)*fc_abs1*fc_abs1+
                           twin_fraction *fc_abs2*fc_abs2);
  }
  return r_factor(fo,fc_abs.const_ref(),scale);
};

using scitbx::vec3;
using scitbx::mat3;
using scitbx::sym_mat3;

template <typename FloatType=double,
          typename OneHLsType=detail::one_h_ls_u_star<FloatType> >
class ls_u_star
{
public:
  ls_u_star() {}

  ls_u_star(
    af::const_ref<FloatType> const& f_model_abs_no_k_total,
    af::const_ref<FloatType> const& f_obs,
    af::const_ref<cctbx::miller::index<> > const& miller_indices,
    af::const_ref<FloatType> const& k_anisotropic)
  {
    MMTBX_ASSERT(f_obs.size() == f_model_abs_no_k_total.size());
    MMTBX_ASSERT(f_obs.size() == k_anisotropic.size());
    MMTBX_ASSERT(f_obs.size() == miller_indices.size());
    grad_u_star_ = scitbx::af::tiny<FloatType, 6>(0,0,0,0,0,0);
    target_ = 0.;
    sum_f_obs_sq = 0.;
    FloatType overall_scale = scale(f_obs, f_model_abs_no_k_total);
    for(std::size_t i=0; i < f_obs.size(); i++) {
      sum_f_obs_sq += f_obs[i]*f_obs[i];
      OneHLsType one_h = OneHLsType(
        f_obs[i],
        f_model_abs_no_k_total[i],
        miller_indices[i],
        k_anisotropic[i],
        overall_scale);
      target_ += one_h.diff * one_h.diff;
      for(std::size_t j=0; j<6; j++) grad_u_star_[j] += one_h.grad_u_star[j];
    }
    // normalize
    MMTBX_ASSERT(sum_f_obs_sq != 0.);
    target_ /= sum_f_obs_sq;
    for(std::size_t i=0; i < 6; i++) grad_u_star_[i] /= sum_f_obs_sq;
  }

  FloatType target() { return target_; }
  scitbx::af::tiny<FloatType, 6> grad_u_star() { return grad_u_star_; }

private:
  FloatType target_;
  FloatType sum_f_obs_sq;
  scitbx::af::tiny<FloatType, 6> grad_u_star_;
};

template <typename FloatType=double,
          typename ComplexType=std::complex<double>,
          typename OneHLsType=detail::one_h_ls<FloatType, ComplexType> >
class ls_kbp_sol_u_star
{
public:
  ls_kbp_sol_u_star() {}

  ls_kbp_sol_u_star(
    f_model::core<FloatType,ComplexType> const& f_model,
    af::const_ref<FloatType> const& f_obs,
    FloatType scale,
    bool const& kb_sol_grad,
    bool const& p_sol_grad,
    bool const& u_star_grad,
    bool const& kb_sol_curv,
    bool const& p_sol_curv)
  {
    MMTBX_ASSERT(f_obs.size() == f_model.f_calc.size());
    if(kb_sol_grad) {
      grad_k_sols_.resize(f_model.n_shells(),0.);
      grad_b_sols_.resize(f_model.n_shells(),0.);
    }
    if(kb_sol_curv) {
      curv_k_sols_.resize(f_model.n_shells(),0.);
      curv_b_sols_.resize(f_model.n_shells(),0.);
    }
    if(u_star_grad) {
      grad_u_star_ = scitbx::af::tiny<FloatType, 6>(0,0,0,0,0,0);
    }
    target_ = 0.;
    sum_f_obs_sq = 0;
    for(std::size_t i=0; i < f_obs.size(); i++) {
      sum_f_obs_sq += f_obs[i]*f_obs[i];
      OneHLsType one_h = OneHLsType(f_obs[i],f_model,i,scale);
      target_ += one_h.diff * one_h.diff;
      if(u_star_grad) {
        one_h.compute_u_star_grad();
        for(std::size_t j=0; j<6; j++) grad_u_star_[j] += one_h.grad_u_star[j];
      }
      // kb
      if(kb_sol_grad || kb_sol_curv) {
        one_h.compute_kbp_grad_curv(kb_sol_grad, kb_sol_curv, false, false);
        for(std::size_t j=0; j<grad_k_sols_.size(); j++) {
          if(kb_sol_grad) {
            grad_k_sols_[j] += one_h.grad_k_sols[j];
            grad_b_sols_[j] += one_h.grad_b_sols[j];
          }
          if(kb_sol_curv) {
            curv_k_sols_[j] += one_h.curv_k_sols[j];
            curv_b_sols_[j] += one_h.curv_b_sols[j];
          }
        }
      }
      // p
      if(p_sol_grad || p_sol_curv) {
        one_h.compute_kbp_grad_curv(false, false, p_sol_grad, p_sol_curv);
        for(std::size_t j=0; j<grad_k_sols_.size(); j++) {
          if(p_sol_grad) {
            grad_p_sols_[j] += one_h.grad_p_sols[j];
          }
          if(p_sol_curv) {
            curv_p_sols_[j] += one_h.curv_p_sols[j];
          }
        }
      }
    }
    // normalize
    MMTBX_ASSERT(sum_f_obs_sq != 0.);
    target_ /= sum_f_obs_sq;
    if(u_star_grad) {
      for(std::size_t i=0; i < 6; i++) grad_u_star_[i] /= sum_f_obs_sq;
    }
    if(kb_sol_grad) {
      for(std::size_t i=0; i<grad_k_sols_.size(); i++) {
        grad_k_sols_[i] /= sum_f_obs_sq;
        grad_b_sols_[i] /= sum_f_obs_sq;
      }
    }
    if(kb_sol_curv) {
      for(std::size_t i=0; i<curv_k_sols_.size(); i++) {
        curv_k_sols_[i] /= sum_f_obs_sq;
        curv_b_sols_[i] /= sum_f_obs_sq;
      }
    }
    if(p_sol_grad) {
      for(std::size_t i=0; i<grad_p_sols_.size(); i++) {
        grad_p_sols_[i] /= sum_f_obs_sq;
      }
    }
    if(p_sol_curv) {
      for(std::size_t i=0; i<curv_p_sols_.size(); i++) {
        curv_p_sols_[i] /= sum_f_obs_sq;
      }
    }
  }

  FloatType target() { return target_; }
  scitbx::af::tiny<FloatType, 6> grad_u_star() { return grad_u_star_; }
  scitbx::af::shared<FloatType> grad_k_sols() { return grad_k_sols_; }
  scitbx::af::shared<FloatType> grad_b_sols() { return grad_b_sols_; }
  scitbx::af::shared<FloatType> grad_p_sols() { return grad_p_sols_; }
  scitbx::af::shared<FloatType> curv_k_sols() { return curv_k_sols_; }
  scitbx::af::shared<FloatType> curv_b_sols() { return curv_b_sols_; }
  scitbx::af::shared<FloatType> curv_p_sols() { return curv_p_sols_; }

private:
  FloatType target_;
  FloatType sum_f_obs_sq;
  scitbx::af::tiny<FloatType, 6> grad_u_star_;
  scitbx::af::shared<FloatType> grad_k_sols_;
  scitbx::af::shared<FloatType> grad_b_sols_;
  scitbx::af::shared<FloatType> grad_p_sols_;
  scitbx::af::shared<FloatType> curv_k_sols_;
  scitbx::af::shared<FloatType> curv_b_sols_;
  scitbx::af::shared<FloatType> curv_p_sols_;
};

//------------------------------------------------------------------------------
// All scales (overall, overall anisotropic, etc) must be applied.
template <
  typename FloatType=double,
  typename ComplexType=std::complex<double>,
  typename CubicEqType=scitbx::math::cubic_equation::real<double, double> >
class bulk_solvent_scale_coefficients_analytical
{
public:
  af::shared<FloatType> x, r;
  FloatType x_best, r_best;

  bulk_solvent_scale_coefficients_analytical() {}

  bulk_solvent_scale_coefficients_analytical(
    af::const_ref<FloatType> const& f_obs,
    af::const_ref<ComplexType> const& f_calc,
    af::const_ref<ComplexType> const& f_mask,
    af::const_ref<bool> const& selection)
  :
  x_best(0), r_best(0)
  {
    MMTBX_ASSERT(f_obs.size() == f_calc.size());
    MMTBX_ASSERT(f_obs.size() == f_mask.size());
    MMTBX_ASSERT(f_obs.size() == selection.size());
    FloatType a2 = 0.0;
    FloatType d3 = 0.0;
    FloatType a = 0.0;
    FloatType b = 0.0;
    FloatType c = 0.0;
    for(std::size_t i=0; i < f_obs.size(); i++) {
      if(selection[i]) {
        FloatType p = std::real(f_calc[i]);
        FloatType r = std::imag(f_calc[i]);
        FloatType q = std::real(f_mask[i]);
        FloatType t = std::imag(f_mask[i]);
        FloatType I = f_obs[i]*f_obs[i];
        FloatType v = p*q+r*t;
        FloatType w = q*q+t*t;
        FloatType u = p*p+r*r;
        a2 += (u*I);
        d3 += (w*w);
        a  += (3.*v*w);
        b  += (2*v*v + u*w - w*I);
        c  += (u*v - v*I);
      }
    }
    MMTBX_ASSERT(d3 != 0.0);
    // coefficients of x**3 + ax**2 + bx + c = 0
    CubicEqType ceo = CubicEqType(1,a/d3,b/d3,c/d3);
    // we are interested in non-negative roots only
    x.push_back(0);
    for(std::size_t j=0; j < 3; j++) {
      if(ceo.x[j]) {
        FloatType root = *ceo.x[j];
        //if(root>=0) {
          x.push_back(root);
        //}
        // MMTBX_ASSERT(std::abs(*ceo.residual()[j]) < 1.e-4); XXX enable back
      }
    }
    // put together plausible results
    bool zero = false;
    af::shared<ComplexType> f_model(f_obs.size());
    for(std::size_t j=0; j < x.size(); j++) {
      //if((x[j]>0) || (x[j]==0 && zero==false)) {
        for(std::size_t i=0; i < f_obs.size(); i++) {
          if(selection[i]) {
            f_model[i] = f_calc[i] + x[j] * f_mask[i];
          }
        }
        r.push_back(r_factor(f_obs, f_model.const_ref(), selection));
      //}
      //else {
      //  r.push_back(-1);
      //}
      if(x[j]==0) zero = true;
    }
    // select best result
    x_best = x[0];
    r_best=1.e+9;
    for(std::size_t j=0; j < x.size(); j++) {
      if(r[j]<=r_best && r[j]>=0) {
        r_best = r[j];
        x_best = x[j];
      }
    }
  }
};
//------------------------------------------------------------------------------

template <
  typename FloatType=double,
  typename ComplexType=std::complex<double>,
  typename CubicEqType=scitbx::math::cubic_equation::real<double,double> >
class overall_and_bulk_solvent_scale_coefficients_analytical
{
public:
  af::shared<FloatType> x, r;
  FloatType x_best, r_best;

  overall_and_bulk_solvent_scale_coefficients_analytical() {}

  overall_and_bulk_solvent_scale_coefficients_analytical(
    af::const_ref<FloatType> const& f_obs,
    af::const_ref<ComplexType> const& f_calc,
    af::const_ref<ComplexType> const& f_mask,
    af::const_ref<bool> const& selection)
  :
  x_best(0), r_best(0)
  {
    MMTBX_ASSERT(f_obs.size() == f_calc.size());
    MMTBX_ASSERT(f_obs.size() == f_mask.size());
    MMTBX_ASSERT(f_obs.size() == selection.size());
    FloatType a2 = 0.0;
    FloatType b2 = 0.0;
    FloatType c2 = 0.0;
    FloatType y2 = 0.0;
    FloatType a3 = 0.0;
    FloatType b3 = 0.0;
    FloatType c3 = 0.0;
    FloatType d3 = 0.0;
    FloatType y3 = 0.0;
    for(std::size_t i=0; i < f_obs.size(); i++) {
      if(selection[i]) {
        FloatType p = std::real(f_calc[i]);
        FloatType r = std::imag(f_calc[i]);
        FloatType q = std::real(f_mask[i]);
        FloatType t = std::imag(f_mask[i]);
        FloatType I = f_obs[i]*f_obs[i];
        FloatType v = p*q+r*t;
        FloatType w = q*q+t*t;
        FloatType u = p*p+r*r;
        a2 += (u*I);
        b2 += (2.*v*I);
        c2 += (w*I);
        y2 += (I*I);
        a3 += (u*v);
        b3 += (2.*v*v+u*w);
        c3 += (3.*v*w);
        d3 += (w*w);
        y3 += (v*I);
      }
    }
    FloatType den = d3*y2-c2*c2;
    MMTBX_ASSERT(den != 0.0);
    // coefficients of x**3 + ax**2 + bc + c = 0
    FloatType a = (c3*y2-c2*b2-c2*y3)/den;
    FloatType b = (b3*y2-c2*a2-y3*b2)/den;
    FloatType c = (a3*y2-y3*a2)/den;
    CubicEqType ceo = CubicEqType(1,a,b,c);
    // we are interested in non-negative roots only
    x.push_back(0);
    for(std::size_t j=0; j < 3; j++) {
      if(ceo.x[j]) {
        FloatType root = *ceo.x[j];
        // residual = x**3 + ax**2 + bc + c = 0
        FloatType residual = std::abs(*ceo.residual()[j]);
        if(root>=0 && residual<1.e-4) { // to avoid numerical issues
          x.push_back(root);
        }
      }
    }
    // put together plausible results
    bool zero = false;
    af::shared<ComplexType> f_model(f_obs.size());
    for(std::size_t j=0; j < x.size(); j++) {
      if((x[j]>0) || (x[j]==0 && zero==false)) {
        for(std::size_t i=0; i < f_obs.size(); i++) {
          if(selection[i]) {
            f_model[i] = f_calc[i] + x[j] * f_mask[i];
          }
        }
        r.push_back(r_factor(f_obs, f_model.const_ref()));
      }
      else {
        r.push_back(-1);
      }
      if(x[j]==0) zero = true;
    }
    // select best result
    x_best = x[0];
    r_best=1.e+9;
    for(std::size_t j=0; j < x.size(); j++) {
      if(r[j]<=r_best && r[j]>=0) {
        r_best = r[j];
        x_best = x[j];
      }
    }
  }
};

template <typename FloatType=double>
class aniso_u_scaler
{
public:
  std::size_t n_rows;
  af::shared<FloatType> u_star_independent;
  scitbx::sym_mat3<FloatType> u_star;
  af::shared<FloatType> a;

  aniso_u_scaler() {}

  aniso_u_scaler(
    af::const_ref<FloatType> const& f_model_abs,
    af::const_ref<FloatType> const& f_obs,
    af::const_ref<cctbx::miller::index<> > const& miller_indices)
  :
  n_rows(6),
  u_star(scitbx::sym_mat3<FloatType>(0,0,0,0,0,0))
  {
    MMTBX_ASSERT(f_obs.size() == f_model_abs.size());
    MMTBX_ASSERT(f_obs.size() == miller_indices.size());
    FloatType minus_two_pi_sq = -2.*std::pow(scitbx::constants::pi, 2);
    af::versa<FloatType, af::mat_grid> m_(af::mat_grid(n_rows, n_rows), 0);
    af::versa<FloatType, af::mat_grid> m(af::mat_grid(n_rows, n_rows), 0);
    af::small<FloatType, 6> b(n_rows, 0);
    af::small<FloatType, 6> vr(n_rows);
    for(std::size_t i=0; i < f_obs.size(); i++) {
      cctbx::miller::index<> const& miller_index = miller_indices[i];
      int i0=miller_index[0],i1=miller_index[1],i2=miller_index[2];
      FloatType fm_abs = f_model_abs[i];
      FloatType fo_i = f_obs[i];
      if(fm_abs<=0 || fo_i<=0) continue;
      FloatType z = std::log(fo_i/fm_abs)/minus_two_pi_sq;
#define _ static_cast<FloatType>
      FloatType const v[] = {
        _(i0*i0), _(i1*i1), _(i2*i2), _(2*i0*i1), _(2*i0*i2), _(2*i1*i2)};
#undef _
      //scitbx::matrix::multiply(
      //  /*a*/ adp_constraint_matrix.begin(),
      //  /*b*/ v,
      //  /*ar*/ n_rows,
      //  /*ac*/ 6,
      //  /*bc*/ 1,
      //  /*ab*/ vr.begin());
      vr[0] = i0*i0;
      vr[1] = i1*i1;
      vr[2] = i2*i2;
      vr[3] = 2*i0*i1;
      vr[4] = 2*i0*i2;
      vr[5] = 2*i1*i2;
      scitbx::matrix::outer_product(m_.begin(),vr.const_ref(),vr.const_ref());
      m += m_;
      b += z*vr;
    }
    af::versa<FloatType, af::c_grid<2> > m_inv(
       scitbx::matrix::packed_u_as_symmetric(
         scitbx::matrix::eigensystem::real_symmetric<FloatType>(
           m.const_ref(), /*relative_epsilon*/ 1.e-9,/*absolute_epsilon*/ 1.e-9)
             .generalized_inverse_as_packed_u().const_ref()));
    af::shared<FloatType> u_star_ = af::matrix_multiply(
      m_inv.const_ref(), b.const_ref());
    for(std::size_t i=0; i < u_star.size(); i++) u_star[i] = u_star_[i];
  }

  aniso_u_scaler(
    af::const_ref<FloatType> const& f_model_abs,
    af::const_ref<FloatType> const& f_obs,
    af::const_ref<cctbx::miller::index<> > const& miller_indices,
    af::const_ref<FloatType, af::mat_grid> const& adp_constraint_matrix)
  :
  n_rows(adp_constraint_matrix.accessor().n_rows()),
  u_star_independent(n_rows, 0)
  {
    MMTBX_ASSERT(f_obs.size() == f_model_abs.size());
    MMTBX_ASSERT(f_obs.size() == miller_indices.size());
    FloatType minus_two_pi_sq = -2.*std::pow(scitbx::constants::pi, 2);
    af::versa<FloatType, af::mat_grid> m_(af::mat_grid(n_rows, n_rows), 0);
    af::versa<FloatType, af::mat_grid> m(af::mat_grid(n_rows, n_rows), 0);
    af::small<FloatType, 6> b(n_rows, 0);
    af::small<FloatType, 6> vr(n_rows);
    for(std::size_t i=0; i < f_obs.size(); i++) {
      cctbx::miller::index<> const& miller_index = miller_indices[i];
      int i0=miller_index[0],i1=miller_index[1],i2=miller_index[2];
      FloatType fm_abs = f_model_abs[i];
      FloatType fo_i = f_obs[i];
      MMTBX_ASSERT(fm_abs > 0);
      MMTBX_ASSERT(fo_i > 0);
      FloatType z = std::log(fo_i/fm_abs)/minus_two_pi_sq;
#define _ static_cast<FloatType>
      FloatType const v[] = {
        _(i0*i0), _(i1*i1), _(i2*i2), _(2*i0*i1), _(2*i0*i2), _(2*i1*i2)};
#undef _
      scitbx::matrix::multiply(
        /*a*/ adp_constraint_matrix.begin(),
        /*b*/ v,
        /*ar*/ n_rows,
        /*ac*/ 6,
        /*bc*/ 1,
        /*ab*/ vr.begin());
      scitbx::matrix::outer_product(m_.begin(),vr.const_ref(),vr.const_ref());
      m += m_;
      b += z*vr;
    }
    af::versa<FloatType, af::c_grid<2> > m_inv(
       scitbx::matrix::packed_u_as_symmetric(
         scitbx::matrix::eigensystem::real_symmetric<FloatType>(
           m.const_ref(), /*relative_epsilon*/ 1.e-9,/*absolute_epsilon*/ 1.e-9)
             .generalized_inverse_as_packed_u().const_ref()));
    u_star_independent = af::matrix_multiply(m_inv.const_ref(), b.const_ref());
  }

  /* SHELXL-like scale: Acta Cryst. (1999). D55, 1158-1167, page 1163
     Note: in the paper it is applied to Fcalc^2, here it is applied to Fcalc.
           The code computes coefficients a. */
  aniso_u_scaler(
    af::const_ref<FloatType> const& f_model_abs,
    af::const_ref<FloatType> const& f_obs,
    af::const_ref<cctbx::miller::index<> > const& miller_indices,
    cctbx::uctbx::unit_cell const& unit_cell)
  :
  a(12, 0)
  {
    MMTBX_ASSERT(f_obs.size() == f_model_abs.size());
    MMTBX_ASSERT(f_obs.size() == miller_indices.size());
    af::versa<FloatType, af::mat_grid> m_(af::mat_grid(12, 12), 0);
    af::versa<FloatType, af::mat_grid> m(af::mat_grid(12, 12), 0);
    af::tiny<FloatType, 12> b;
    b.fill(0);
    af::double6 p = unit_cell.reciprocal_parameters();
    FloatType as=p[0], bs=p[1], cs=p[2];
    for(std::size_t i=0; i < f_obs.size(); i++) {
      cctbx::miller::index<> const& miller_index = miller_indices[i];
      int h=miller_index[0], k=miller_index[1], l=miller_index[2];
      FloatType fm_i = f_model_abs[i];
      FloatType stol = unit_cell.stol_sq(miller_index);
      FloatType s = 0;
      if(stol != 0) s = 1./stol;
      af::tiny<FloatType, 12> v_;
      FloatType* v = v_.begin();
      *v++ = h*h*as*as*s;
      *v++ = h*h*as*as;
      *v++ = k*k*bs*bs*s;
      *v++ = k*k*bs*bs;
      *v++ = l*l*cs*cs*s;
      *v++ = l*l*cs*cs;
      *v++ = 2*k*l*bs*cs*s;
      *v++ = 2*k*l*bs*cs;
      *v++ = 2*h*l*as*cs*s;
      *v++ = 2*h*l*as*cs;
      *v++ = 2*h*k*as*bs*s;
      *v++ = 2*h*k*as*bs;
      b += (f_obs[i]-fm_i)*fm_i*v_;
      v_ *= fm_i;
      scitbx::matrix::outer_product(m_.begin(),v_.const_ref(),v_.const_ref());
      m += m_;
    }
    af::versa<FloatType, af::c_grid<2> > m_inv(
       scitbx::matrix::packed_u_as_symmetric(
         scitbx::matrix::eigensystem::real_symmetric<FloatType>(
           m.const_ref(), /*relative_epsilon*/ 1.e-9,/*absolute_epsilon*/ 1.e-9)
             .generalized_inverse_as_packed_u().const_ref()));
    a = af::matrix_multiply(m_inv.const_ref(), b.const_ref());
  }
};

// Grid search based fit of k*exp(-B*ss) to k_total
template <typename FloatType>
 af::tiny<FloatType, 3>
 fit_k_exp_b_to_k_total(
   af::const_ref<FloatType> const& data,
   af::const_ref<FloatType> const& ss,
   FloatType k_start,
   FloatType b_start)
 {
   MMTBX_ASSERT(data.size() == ss.size());
   FloatType k_best = 0.0;
   FloatType b_best = 0.0;
   FloatType r_best = DBL_MAX;
   FloatType k_min  = std::max(0., k_start-std::abs(k_start));
   FloatType k_max  = k_start+std::abs(k_start);
   FloatType b_min  = b_start-std::abs(b_start);
   FloatType b_max  = b_start+std::abs(b_start);
   if(k_min==k_max) {
     k_min=0;
     k_max=1;
   }
   if(b_min==b_max) {
     b_min=-1;
     b_max= 1;
   }
   int n_iterations = 5;
   FloatType iteration_increment = 1;
   for(std::size_t iter=1; iter <= n_iterations; iter++) {
     int n_steps = 10+1;
     if(iter > 1) n_steps = 5;
     FloatType k_step = (k_max-k_min)/n_steps;
     FloatType b_step = (b_max-b_min)/n_steps;
     FloatType k = k_min;
     for(std::size_t ik=1; ik <= n_steps; ik++) {
       FloatType b = b_min;
       for(std::size_t ib=1; ib <= n_steps; ib++) {
         FloatType n = 0., d=0.;
         for(std::size_t i=0; i < data.size(); i++) {
           FloatType arg = -b*ss[i];
           FloatType d_ = 0;
           if(arg<700) d_ = k*std::exp(arg); // to avoid overflow problem
           n += std::abs(data[i]-d_);
           d += std::abs(data[i]);
         }
         FloatType r_ = 0.;
         if(d!=0) r_ = n/d;
         else return af::tiny<FloatType, 3> (0, 0, 0);
         if(r_<r_best) {
           r_best = r_;
           k_best = k;
           b_best = b;
         }
         b += b_step;
       }
       k += k_step;
     }

     k_start = k_best;
     b_start = b_best;

     iteration_increment -= 1./n_iterations;
     k_min  = std::max(0., k_start-std::abs(k_start)*iteration_increment);
     k_max  = k_start+std::abs(k_start)*iteration_increment;
     b_min  = b_start-std::abs(b_start)*iteration_increment;
     b_max  = b_start+std::abs(b_start)*iteration_increment;
   }
   MMTBX_ASSERT(k_best>=0);
   return af::tiny<FloatType, 3> (k_best, b_best, r_best);
 };

template <typename FloatType, typename ComplexType>
 af::tiny<FloatType, 2>
 k_mask_and_k_overall_grid_search(
   af::const_ref<FloatType>   const& f_obs,
   af::const_ref<ComplexType> const& f_calc,
   af::const_ref<ComplexType> const& f_mask,
   af::const_ref<FloatType>   const& k_mask_range,
   af::const_ref<bool>        const& selection)
 {
   MMTBX_ASSERT(f_mask.size() == f_obs.size());
   MMTBX_ASSERT(f_obs.size() == f_calc.size());
   MMTBX_ASSERT(f_obs.size() == selection.size());
   FloatType k_mask_best = 0.0;
   FloatType k_overall_best = scale(f_obs, f_calc, selection);
   FloatType r_best = r_factor(f_obs, f_calc, selection, k_overall_best);
   af::shared<ComplexType> f_model(f_obs.size());
   for(std::size_t i=0; i < k_mask_range.size(); i++) {
     FloatType k_mask = k_mask_range[i];
     for(std::size_t j=0; j < f_obs.size(); j++) {
       if(selection[j]) {
         f_model[j] = f_calc[j] + k_mask * f_mask[j];
       }
     }
     FloatType k_overall = scale(f_obs, f_model.const_ref(), selection);
     FloatType r = r_factor(f_obs, f_model.const_ref(), selection, k_overall);
     if(r < r_best) {
       k_mask_best = k_mask;
       k_overall_best = k_overall;
       r_best = r;
     }
   }
   return af::tiny<FloatType, 2> (k_mask_best, k_overall_best);
 };

template <typename FloatType=double,
          typename ComplexType=std::complex<double> >
class k_sol_b_sol_k_anisotropic_scaler_twin
{
public:
  k_sol_b_sol_k_anisotropic_scaler_twin() {}

  k_sol_b_sol_k_anisotropic_scaler_twin(
    af::const_ref<FloatType>               const& f_obs,
    af::const_ref<ComplexType>             const& f_calc_1,
    af::const_ref<ComplexType>             const& f_calc_2,
    af::const_ref<ComplexType>             const& f_mask_1,
    af::const_ref<ComplexType>             const& f_mask_2,
    af::const_ref<FloatType>               const& ss,
    FloatType                              const& twin_fraction,
    af::const_ref<FloatType>               const& k_sol_range,
    af::const_ref<FloatType>               const& b_sol_range,
    af::const_ref<cctbx::miller::index<> > const& miller_indices,
    cctbx::uctbx::unit_cell                const& unit_cell,
    FloatType                              const& r_ref)
  :
  k_best(0), b_best(0), r_best(r_ref), k_mask_best(ss.size()),
  k_anisotropic_best(ss.size()), updated_(false),
  u_star_best(scitbx::sym_mat3<FloatType>(0,0,0,0,0,0))
  {
    MMTBX_ASSERT(f_obs.size() == f_calc_1.size());
    MMTBX_ASSERT(f_obs.size() == f_calc_2.size());
    MMTBX_ASSERT(f_obs.size() == f_mask_1.size());
    MMTBX_ASSERT(f_obs.size() == f_mask_2.size());
    MMTBX_ASSERT(f_obs.size() == miller_indices.size());
    MMTBX_ASSERT(f_obs.size() == ss.size());
    k_mask_best.fill(0);
    k_anisotropic_best.fill(1);
    af::shared<FloatType> f_model(ss.size());
    for(std::size_t i=0; i < k_sol_range.size(); i++) {
      FloatType ks = k_sol_range[i];
      for(std::size_t j=0; j < b_sol_range.size(); j++) {
        FloatType mbs = -b_sol_range[j];
        for(std::size_t k=0; k < f_obs.size(); k++) {
          FloatType km = ks * std::exp(mbs * ss[k]);
          FloatType f1 = std::abs(f_calc_1[k]+km*f_mask_1[k]);
          FloatType f2 = std::abs(f_calc_2[k]+km*f_mask_2[k]);
          FloatType f_model_abs = std::sqrt(
           (1-twin_fraction)*f1*f1+
              twin_fraction *f2*f2);
          f_model[k] = f_model_abs;
        }
        // Using polynomial scale
        af::shared<FloatType> a_scale = aniso_u_scaler<FloatType>(
          f_model.ref(), f_obs, miller_indices, unit_cell).a;
        af::shared<FloatType> k_aniso =
          mmtbx::f_model::k_anisotropic<FloatType>(miller_indices, a_scale,
            unit_cell);
//        scitbx::sym_mat3<FloatType> u_star_ = aniso_u_scaler<FloatType>(
//          f_model.ref(), f_obs, miller_indices).u_star;
//        af::shared<FloatType> k_aniso =
//          mmtbx::f_model::k_anisotropic<FloatType>(miller_indices, u_star_);
        FloatType r = r_factor(f_obs, (f_model*k_aniso).const_ref());
        if(r < r_best) {
          k_best = k_sol_range[i];
          b_best = b_sol_range[j];
          k_anisotropic_best = k_aniso;
          //u_star_best = u_star_;
          r_best = r;
        }
      }
    }
    if(r_best!=r_ref) {
      updated_=true;
      for(std::size_t k=0; k < f_obs.size(); k++) {
        k_mask_best[k] = k_best * std::exp(-b_best * ss[k]);
      }
    }
  }

  k_sol_b_sol_k_anisotropic_scaler_twin(
    af::const_ref<FloatType>   const& f_obs,
    af::const_ref<ComplexType> const& f_calc,
    af::const_ref<ComplexType> const& f_mask,
    af::const_ref<FloatType>   const& k_total,
    af::const_ref<FloatType>   const& ss,
    af::const_ref<FloatType>   const& k_sol_range,
    af::const_ref<FloatType>   const& b_sol_range,
    FloatType                  const& r_ref)
  :
  k_best(0), b_best(0), r_best(r_ref), k_mask_best(ss.size()),
  k_anisotropic_best(ss.size()), updated_(false)
  {
    MMTBX_ASSERT(f_obs.size() == f_calc.size());
    MMTBX_ASSERT(f_obs.size() == f_mask.size());
    MMTBX_ASSERT(f_obs.size() == ss.size());
    MMTBX_ASSERT(f_obs.size() == k_total.size());
    k_mask_best.fill(0);
    k_anisotropic_best.fill(1);
    af::shared<FloatType> f_model(ss.size());
    for(std::size_t i=0; i < k_sol_range.size(); i++) {
      FloatType ks = k_sol_range[i];
      for(std::size_t j=0; j < b_sol_range.size(); j++) {
        FloatType mbs = -b_sol_range[j];
        for(std::size_t k=0; k < f_obs.size(); k++) {
          FloatType km = ks * std::exp(mbs * ss[k]);
          f_model[k] = k_total[k]*std::abs(f_calc[k]+km*f_mask[k]);
        }
        FloatType r = r_factor(f_obs, f_model.const_ref());
        if(r < r_best) {
          k_best = k_sol_range[i];
          b_best = b_sol_range[j];
          r_best = r;
        }
      }
    }
    if(r_best!=r_ref) {
      updated_=true;
      for(std::size_t k=0; k < f_obs.size(); k++) {
        k_mask_best[k] = k_best * std::exp(-b_best * ss[k]);
      }
    }
  }

  k_sol_b_sol_k_anisotropic_scaler_twin(
    af::const_ref<FloatType>               const& f_obs,
    af::const_ref<ComplexType>             const& f_calc,
    af::const_ref<ComplexType>             const& f_mask,
    af::const_ref<FloatType>               const& ss,
    af::const_ref<FloatType>               const& k_sol_range,
    af::const_ref<FloatType>               const& b_sol_range,
    af::const_ref<cctbx::miller::index<> > const& miller_indices,
    FloatType                              const& r_ref)
  :
  k_best(0), b_best(0), r_best(r_ref), k_mask_best(ss.size()),
  k_anisotropic_best(ss.size()), updated_(false),
  u_star_best(scitbx::sym_mat3<FloatType>(0,0,0,0,0,0))
  {
    MMTBX_ASSERT(f_obs.size() == f_calc.size());
    MMTBX_ASSERT(f_obs.size() == f_mask.size());
    MMTBX_ASSERT(f_obs.size() == ss.size());
    MMTBX_ASSERT(f_obs.size() == miller_indices.size());
    k_mask_best.fill(0);
    k_anisotropic_best.fill(1);
    af::shared<FloatType> f_model(ss.size());
    for(std::size_t i=0; i < k_sol_range.size(); i++) {
      FloatType ks = k_sol_range[i];
      for(std::size_t j=0; j < b_sol_range.size(); j++) {
        FloatType mbs = -b_sol_range[j];
        for(std::size_t k=0; k < f_obs.size(); k++) {
          FloatType km = ks * std::exp(mbs * ss[k]);
          f_model[k] = std::abs(f_calc[k]+km*f_mask[k]);
        }
        FloatType sc = scale(f_obs, f_model.ref());
        scitbx::sym_mat3<FloatType> u_star_ = aniso_u_scaler<FloatType>(
          (f_model*sc).ref(), f_obs, miller_indices).u_star;
        af::shared<FloatType> k_aniso =
          mmtbx::f_model::k_anisotropic<FloatType>(miller_indices, u_star_);
        FloatType r = r_factor(f_obs, (f_model*k_aniso).const_ref());
        if(r < r_best) {
          k_best = k_sol_range[i];
          b_best = b_sol_range[j];
          k_anisotropic_best = k_aniso;
          u_star_best = u_star_;
          r_best = r;
        }
      }
    }
    if(r_best!=r_ref) {
      updated_=true;
      for(std::size_t k=0; k < f_obs.size(); k++) {
        k_mask_best[k] = k_best * std::exp(-b_best * ss[k]);
      }
    }
  }

  bool updated()                        { return updated_; }
  FloatType r()                         { return r_best; }
  FloatType k_sol()                     { return k_best; }
  FloatType b_sol()                     { return b_best; }
  af::shared<FloatType> k_mask()        { return k_mask_best; }
  af::shared<FloatType> k_anisotropic() { return k_anisotropic_best; }
  scitbx::sym_mat3<FloatType> u_star()  { return u_star_best; }

private:
  FloatType             r_best;
  FloatType             k_best;
  FloatType             b_best;
  af::shared<FloatType> k_mask_best;
  af::shared<FloatType> k_anisotropic_best;
  scitbx::sym_mat3<FloatType> u_star_best;
  bool updated_;
};

// copy of below
template <typename FloatType=double>
class f_kb_scaled
{
public:
  f_kb_scaled() {}

  f_kb_scaled(
    af::const_ref<FloatType> const& f1,
    af::const_ref<FloatType> const& f2,
    af::const_ref<FloatType> const& b_range,
    af::const_ref<FloatType> const& ss)
  {
    // Compute exp(-B*s**2/4) * F2
    MMTBX_ASSERT(f1.size() == f2.size());
    MMTBX_ASSERT(f1.size() == ss.size());
    b_best = 0.0;
    FloatType r_best = 1.e+10;
    k_best = 1.0;
    result.resize(ss.size(), 0.);
    af::shared<FloatType> f2_scaled(ss.size());
    for(std::size_t j=0; j < b_range.size(); j++) {
      FloatType mbs = -b_range[j];
      for(std::size_t k=0; k < ss.size(); k++) {
        FloatType kbs = std::exp(mbs * ss[k]);
        f2_scaled[k] = kbs*f2[k];
      }
      FloatType sc = scale(f1, f2_scaled.const_ref());
      FloatType r = r_factor(f1, f2_scaled.const_ref(), sc);
      if(r < r_best) {
        r_best = r;
        b_best = b_range[j];
        k_best = sc;
      }
    }
    for(std::size_t k=0; k < ss.size(); k++) {
        FloatType all_scale = k_best*std::exp(-1.*b_best*ss[k]);
        result[k] = all_scale*f2[k];
    }
  }

  af::shared<FloatType> scaled() { return result; }
  FloatType b() { return b_best; }
  FloatType k() { return k_best; }

private:
  af::shared<FloatType> result;
  FloatType b_best;
  FloatType k_best;
};


/*
Find best fit of F1+k*exp(-B*s**2/4)*F2 to F0 w.r.t. k and B.
*/

template <typename FloatType=double, typename ComplexType=std::complex<double> >
class add_complex_f_kb_scaled
{
public:
  add_complex_f_kb_scaled() {}

  add_complex_f_kb_scaled(
    af::const_ref<FloatType>   const& f0,
    af::const_ref<ComplexType> const& f1,
    af::const_ref<ComplexType> const& f2,
    af::const_ref<FloatType>   const& k_range,
    af::const_ref<FloatType>   const& b_range,
    af::const_ref<FloatType>   const& ss)
  {
    MMTBX_ASSERT(f0.size() == f1.size());
    MMTBX_ASSERT(f1.size() == f2.size());
    MMTBX_ASSERT(f1.size() == ss.size());
    b_best = -1;
    r_best = r_factor(f0, f1, 1.0);
    k_best = 1.0;
    result.resize(ss.size(), 0.);
    af::shared<ComplexType> f12(ss.size());
    for(std::size_t j=0; j < b_range.size(); j++) {
      FloatType mb = -b_range[j];
      for(std::size_t i=0; i < k_range.size(); i++) {
        FloatType k = k_range[i];
        for(std::size_t m=0; m < ss.size(); m++) {
          FloatType kbs = k*std::exp(mb * ss[m]);
          f12[m] = f1[m]+kbs*f2[m];
        }
        FloatType r = r_factor(f0, f12.const_ref());
        if(r < r_best) {
          r_best = r;
          b_best = b_range[j];
          k_best = k;
        }
      }
    }
    for(std::size_t i=0; i < ss.size(); i++) {
      FloatType scale = k_best*std::exp(-1.*b_best*ss[i]);
      result[i] = scale*f2[i];
    }
  }

  af::shared<ComplexType> scaled() { return result; }
  FloatType b() { return b_best; }
  FloatType k() { return k_best; }
  FloatType r() { return r_best; }

private:
  af::shared<ComplexType> result;
  FloatType b_best;
  FloatType k_best;
  FloatType r_best;
};















template <typename FloatType=double, typename ComplexType=std::complex<double> >
class complex_f_kb_scaled
{
public:
  complex_f_kb_scaled() {}

  complex_f_kb_scaled(
    af::const_ref<ComplexType> const& f1,
    af::const_ref<ComplexType> const& f2,
    af::const_ref<FloatType>   const& b_range,
    af::const_ref<FloatType>   const& ss)
  {
    // Compute exp(-B*s**2/4) * F2
    MMTBX_ASSERT(f1.size() == f2.size());
    MMTBX_ASSERT(f1.size() == ss.size());
    b_best = 0.0;
    FloatType r_best = 1.e+10;
    k_best = 1.0;
    result.resize(ss.size(), 0.);
    af::shared<ComplexType> f2_scaled(ss.size());
    for(std::size_t j=0; j < b_range.size(); j++) {
      FloatType mbs = -b_range[j];
      for(std::size_t k=0; k < ss.size(); k++) {
        FloatType kbs = std::exp(mbs * ss[k]);
        f2_scaled[k] = kbs*f2[k];
      }
      FloatType sc = scale(f1, f2_scaled.const_ref());
      FloatType r = r_factor(f1, f2_scaled.const_ref(), sc);
      if(r < r_best) {
        r_best = r;
        b_best = b_range[j];
        k_best = sc;
      }
    }
    for(std::size_t k=0; k < ss.size(); k++) {
        FloatType all_scale = k_best*std::exp(-1.*b_best*ss[k]);
        result[k] = all_scale*f2[k];
    }
  }

  af::shared<ComplexType> scaled() { return result; }
  FloatType b() { return b_best; }
  FloatType k() { return k_best; }

private:
  af::shared<ComplexType> result;
  FloatType b_best;
  FloatType k_best;
};

template <typename FloatType, typename ComplexType>
 af::shared<ComplexType>
 complex_f_minus_f_kb_scaled(
   af::const_ref<ComplexType> const& f1,
   af::const_ref<ComplexType> const& f2,
   af::const_ref<FloatType>   const& b_range,
   af::const_ref<FloatType>   const& ss)
 {
   // Compute F1 - scale * exp(-B*s**2/4) * F2
   MMTBX_ASSERT(f1.size() == f2.size());
   MMTBX_ASSERT(f1.size() == ss.size());
   FloatType b_best = 0.0;
   FloatType scale_best = 0.0;
   FloatType r_best = 1.e+10;
   af::shared<ComplexType> result(ss.size());
   af::shared<ComplexType> f2_scaled(ss.size());
   for(std::size_t j=0; j < b_range.size(); j++) {
     FloatType mbs = -b_range[j];
     for(std::size_t k=0; k < ss.size(); k++) {
       FloatType kbs = std::exp(mbs * ss[k]);
       f2_scaled[k] = kbs*f2[k];
     }
     FloatType sc = scale(f1, f2_scaled.const_ref());
     FloatType r = r_factor(f1, f2_scaled.const_ref(), sc);
     if(r < r_best) {
       r_best = r;
       b_best = b_range[j];
       scale_best = sc;
     }
   }
   for(std::size_t k=0; k < ss.size(); k++) {
     if(std::abs(scale_best)>1.e-9) {
       FloatType all_scale = scale_best*std::exp(-1.*b_best*ss[k]);
       result[k] = f1[k]-all_scale*f2[k];
     }
     else {
       result[k] = 0.;
     }
   }
   return result;
 };

template <typename FloatType, typename ComplexType>
 af::shared<FloatType>
 ksol_bsol_grid_search(
   af::const_ref<FloatType>   const& f_obs,
   af::const_ref<ComplexType> const& f_calc,
   af::const_ref<ComplexType> const& f_mask,
   af::const_ref<FloatType>   const& k_sol_range,
   af::const_ref<FloatType>   const& b_sol_range,
   af::const_ref<FloatType>   const& ss,
   FloatType                  const& scalar_scale,
   af::const_ref<FloatType>   const& overall_scale,
   af::const_ref<FloatType>   const& k_anisotropic,
   FloatType                  const& r_ref)
 {
   MMTBX_ASSERT(f_mask.size() == f_obs.size());
   MMTBX_ASSERT(f_obs.size() == f_calc.size());
   MMTBX_ASSERT(ss.size() == f_calc.size());
   MMTBX_ASSERT(overall_scale.size() == f_calc.size());
   MMTBX_ASSERT(k_anisotropic.size() == f_calc.size());
   FloatType k_best = 0.0;
   FloatType b_best = 0.0;
   FloatType r_best = r_ref;
   af::shared<ComplexType> f_model(ss.size());
   af::shared<FloatType> bulk_solvent_scale;//(f_obs.size());
   for(std::size_t i=0; i < k_sol_range.size(); i++) {
     FloatType ks = k_sol_range[i];
     for(std::size_t j=0; j < b_sol_range.size(); j++) {
       FloatType mbs = -b_sol_range[j];
       for(std::size_t k=0; k < f_obs.size(); k++) {
         FloatType kbs = ks * std::exp(mbs * ss[k]);
         f_model[k] = scalar_scale * overall_scale[k] *
           k_anisotropic[k] * (f_calc[k] + kbs * f_mask[k]);
       }
       FloatType r = r_factor(f_obs, f_model.const_ref());
       if(r < r_best) {
         k_best = k_sol_range[i];
         b_best = b_sol_range[j];
         r_best = r;
       }
     }
   }
   bulk_solvent_scale.push_back(k_best);
   bulk_solvent_scale.push_back(b_best);
   bulk_solvent_scale.push_back(r_best);
   //for(std::size_t k=0; k < f_obs.size(); k++) {
   //  bulk_solvent_scale[k] = k_best * std::exp(-b_best * ss[k]);
   //}
   return bulk_solvent_scale;
 };

template <typename FloatType>
 af::shared<FloatType>
 set_to_linear_interpolated(
   af::const_ref<FloatType> const& ss,
   FloatType                const& k,
   FloatType                const& b,
   af::const_ref<bool>      const& selection,
   af::shared<FloatType>           data)
 {
   af::shared<FloatType> k_mask(ss.size());
   for(std::size_t i=0; i < ss.size(); i++) {
     if(selection[i]) {
       FloatType v = k * ss[i] + b;
       if(v<0) v=0;
       data[i] = v;
     }
   }
   return data;
 };

template <typename FloatType>
 af::shared<FloatType>
 set_k_mask_to_cubic_polynom(
   af::const_ref<FloatType> const& ss,
   FloatType                const& ss_cutoff,
   af::tiny<FloatType, 4>   const& coeffs)
 {
   af::shared<FloatType> k_mask(ss.size());
   for(std::size_t i=0; i < ss.size(); i++) {
     FloatType x = ss[i];
     FloatType f = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x + coeffs[3]*x*x*x;
     if(f<0) f = 0;
     if(x<ss_cutoff) k_mask[i] = f;
     else k_mask[i] = 0;
   }
   return k_mask;
 };

//------------------------------------------------------------------------------

template <typename DataType, typename TagType>
void
  symmetrize_mask(
      af::ref<DataType, af::c_grid<3> > const& data,
      af::const_ref<TagType, af::c_grid<3> > const& tags)
  {
    MMTBX_ASSERT(tags.accessor().all_eq(data.accessor()));
    for(std::size_t i=0;i<data.size();i++) {
      if (tags[i] < 0) continue;
      if (data[i] == 0) data[tags[i]] = 0;
    }
    for(std::size_t i=0;i<data.size();i++) {
      if (tags[i] < 0) continue;
      data[i] = data[tags[i]];
    }
  }

af::shared<double> fb_cart(sym_mat3<double> const& b_cart,
                            af::const_ref<cctbx::miller::index<> > const& hkl,
                            cctbx::uctbx::unit_cell const& uc);

}} // namespace mmtbx::bulk_solvent

#endif // MMTBX_BULK_SOLVENT_BULK_SOLVENT_H
