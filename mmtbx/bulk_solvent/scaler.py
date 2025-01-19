from __future__ import absolute_import, division, print_function
from scitbx.array_family import flex
import sys
from mmtbx import bulk_solvent
from cctbx import adptbx
import boost_adaptbx.boost.python as bp
from six.moves import range
ext = bp.import_ext("mmtbx_f_model_ext")
from cctbx import sgtbx
from mmtbx.bulk_solvent import kbu_refinery
import mmtbx.f_model
import math
from libtbx import group_args
import scitbx.math
from cctbx import miller
import mmtbx.arrays
import scitbx.math
from libtbx import group_args
from libtbx.test_utils import approx_equal

def moving_average(x):
  x_ = x_ = [x[0]] + list(x) + [x[len(x)-1]]
  for cycle in range(5):
    result = x_[:]
    selection = flex.bool(len(result), False)
    for i, s in enumerate(selection):
      if(i!=0 and i!=len(result)-1):
        if((result[i-1]<result[i] and result[i+1]<result[i]) or
           (result[i-1]>result[i] and result[i+1]>result[i])):
          selection[i]=True
    for i in range(len(result)):
      if(i!=0 and i!=len(result)-1 and selection[i]):
        result[i] = (x_[i-1]+x_[i]+x_[i+1])/3.
    x_ = result[:]
  return result[1:len(result)-1]

def run_simple(fmodel_kbu, bin_selections, r_free_flags, bulk_solvent,
               aniso_scale):
  if(aniso_scale):
    try_poly    = True
    try_expanal = True
    try_expmin  = False
  else:
    try_poly    = False
    try_expanal = False
    try_expmin  = False
  return run(
    f_obs            = fmodel_kbu.f_obs,
    f_calc           = fmodel_kbu.f_calc,
    f_mask           = fmodel_kbu.f_masks,
    r_free_flags     = r_free_flags,
    bulk_solvent     = bulk_solvent,
    try_poly         = try_poly,
    try_expanal      = try_expanal,
    try_expmin       = try_expmin,
    ss               = fmodel_kbu.ss,
    number_of_cycles = 100,
    bin_selections   = bin_selections)

class run(object):
  def __init__(self,
               f_obs,
               f_calc, # can be a sum: f_calc=f_hydrogens+f_calc+f_part
               f_mask, # only one shell is supported
               r_free_flags,
               ss,
               bin_selections=None,
               scale_method="combo",
               number_of_cycles=20, # termination occures much earlier
               auto_convergence_tolerance = 1.e-4,
               log=None,
               auto=True,
               auto_convergence=True,
               bulk_solvent = True,
               try_poly = True,
               try_expanal = True,
               try_expmin = False,
               verbose=False):
    if(log is None): log = sys.stdout
    self.d_hilo = 6
    assert f_obs.indices().all_eq(r_free_flags.indices())
    self.log = log
    self.scale_method = scale_method
    self.verbose = verbose
    self.r_free_flags = r_free_flags
    self.ss = ss
    self.bulk_solvent = bulk_solvent
    self.try_poly    = try_poly
    self.try_expanal = try_expanal
    self.try_expmin  = try_expmin
    self.d_spacings = f_obs.d_spacings()
    self.r_low = None
    self.poly_approx_cutoff = None
    self.f_obs = f_obs
    self.r_free_flags = r_free_flags
    self.auto_convergence = auto_convergence
    self.auto_convergence_tolerance = auto_convergence_tolerance
    self.scale_matrices = None
    self.auto = auto
    self.bin_selections = bin_selections
    self.k_exp_overall, self.b_exp_overall = None,None
    self.u_star = None
    if(self.bin_selections is None):
      self.bin_selections = self.f_obs.log_binning()
    # If R-free flags are bad - discard them and use all reflections instead.
    ifg = self.is_flags_good()
    if(ifg):
      self.selection_work = miller.array(
        miller_set = self.f_obs,
        data       = ~self.r_free_flags.data())
    else:
      self.selection_work = miller.array(
        miller_set = self.f_obs,
        data       = flex.bool(self.f_obs.data().size(), True))
    #
    self.ss = ss
    def init_result():
      return group_args(
        k_mask_bin_orig   = None,
        k_mask_bin_smooth = None,
        k_mask            = None,
        k_isotropic       = None,
        k_mask_fit_params = None)
    self.bss_result = init_result()
    if(verbose):
      print("-"*80, file=log)
      print("Overall, iso- and anisotropic scaling and bulk-solvent modeling:", file=log)
    point_group = sgtbx.space_group_info(
      symbol=f_obs.space_group().type().lookup_symbol()
      ).group().build_derived_point_group()
    self.adp_constraints = sgtbx.tensor_rank_2_constraints(
      space_group=point_group,
      reciprocal_space=True)
    self.core = mmtbx.arrays.init(f_calc = f_calc, f_masks = f_mask)
    if(abs(self.core.f_mask()).data().all_eq(0)): self.bulk_solvent=False
    self.cores_and_selections = []
    self.low_resolution_selection = self._low_resolution_selection()
    self.high_resolution_selection = self._high_resolution_selection()
    if(verbose):
      print("  Using %d resolution bins"%len(self.bin_selections), file=log)
    self.ss_bin_values=[]
    sel_positive = self.f_obs.data()>0
    self.selection_work = self.selection_work.customized_copy(
      data = self.selection_work.data() & sel_positive)
    for i_sel, sel in enumerate(self.bin_selections):
      core_selected = self.core.select(selection=sel)
      sel_use = self.selection_work.data().select(sel)
      sel_work = sel & self.selection_work.data()
      self.cores_and_selections.append([sel, core_selected, sel_use, sel_work])
      ss = self.ss.select(sel)
      self.ss_bin_values.append([
        flex.min(ss),
        flex.max(ss),
        flex.mean(ss)])
    for cycle in range(number_of_cycles):
      r_start = self.r_factor()
      r_start0 = r_start
      if(verbose):
        print("  cycle %d:"%cycle, file=log)
        print("    r(start): %6.4f"%(r_start), file=log)
      # bulk-solvent and overall isotropic scale
      if(self.bulk_solvent):
        if(cycle==0):
          # NEW
          #use_highres = False
          #if(self.f_obs.d_min() < self.d_hilo - 2):
          #  use_highres = True
          #if(use_highres):
          #  r_start = self.anisotropic_scaling(r_start = r_start, use_highres=True)
          #else:
          #  r_start = self.set_k_isotropic_exp(r_start = r_start, verbose=verbose)
          #
          #r_start = self.k_mask_grid_search(r_start=r_start)
          #
          #if(use_highres):
          #  r_start = self.anisotropic_scaling(r_start = r_start, use_highres=True)
          #else:
          #  r_start = self.set_k_isotropic_exp(r_start = r_start, verbose=verbose)
          # OLD
          for mic in [1,2]:
            r_start = self.set_k_isotropic_exp(r_start = r_start, verbose=verbose)
            r_start = self.k_mask_grid_search(r_start=r_start)
            r_start = self.set_k_isotropic_exp(r_start = r_start,  verbose=verbose)
        else:
          r_start = self.bulk_solvent_scaling(r_start = r_start)
          if(verbose):
            print("    r(bulk_solvent_scaling): %6.4f"%r_start, file=self.log)
      # anisotropic scale
      if([try_poly, try_expanal, try_expmin].count(True)):
        if(verbose): print("    anisotropic scaling:", file=log)
        r_start = self.anisotropic_scaling(r_start = r_start, use_highres=False)
      if(self.auto_convergence and self.is_converged(r_start=r_start0,
         tolerance=self.auto_convergence_tolerance)):
        break
    self.apply_overall_scale()
    if(verbose):
      print("  r(final): %6.4f"%(self.r_factor()), file=log)
      self.show()
    #
    self.r_low = self._r_low()
    self.r_high = self._r_high()
    if(verbose):
      d = self.d_spacings.data().select(self.low_resolution_selection)
      d1 = ("%7.4f"%flex.min(d)).strip()
      d2 = ("%7.4f"%flex.max(d)).strip()
      n = d.size()
      print("r(low-resolution: %s-%s A; %d reflections): %6.4f"%(
        d2,d1,n,self.r_low), file=self.log)
      print("-"*80, file=log)
    self.r_final = self.r_factor()

  def is_flags_good(self):
    """
    This function detects inadequate R-free flags.
    """
    result = True
    fd = self.r_free_flags.data()
    for i_sel, sel in enumerate(self.bin_selections):
      fd_ = fd.select(sel)
      if(fd_.count(True)==0):
        result = False
        break
    return result

  def set_k_isotropic_exp(self, r_start, verbose, b_lower_limit = -100):
    if(self.verbose):
      print("    set_k_isotropic_exp:", file=self.log)
      print("      r_start: %6.4f (r_low: %6.4f)"%(r_start,self._r_low()))
    k_iso   = flex.double(self.core.k_isotropic.size(), 1) # Done at start only!
    k_aniso = flex.double(self.core.k_isotropic.size(), 1) # Done at start only!
    arrays = mmtbx.arrays.init(
      f_calc          = self.core.f_calc,
      f_masks         = self.core.f_mask(),
      k_isotropic     = k_iso,
      k_anisotropic   = k_aniso,
      k_masks         = self.core.k_mask())
    sel = self.selection_work.data()
    #
    # At least in one example this gives more accurate answer but higher R than start!
    #
    rf = scitbx.math.gaussian_fit_1d_analytical(
      x = flex.sqrt(self.ss).select(sel),
      y = self.f_obs.data().select(sel),
      z = abs(arrays.f_model).data().select(sel))
    if(rf.b < b_lower_limit): return r_start
    k1 = rf.a * flex.exp(-self.ss * rf.b)
    r1 = self.try_scale(k_isotropic_exp = k1)
    #
    # At least in one example this gives less accurate answer but lower R than start!
    #
    o = bulk_solvent.f_kb_scaled(
      f1 = self.f_obs.data().select(sel),
      f2 = flex.abs(arrays.f_model.data()).select(sel),
      b_range = flex.double(range(-100,100,1)),
      ss = self.ss.select(sel))
    k2 = o.k() * flex.exp(-self.ss * o.b())
    r2 = self.try_scale(k_isotropic_exp = k2)
    #
    if(r1<r2):
      r = r1
      k = k1
    else:
      r = r2
      k = k2
    if(r<r_start):
      self.core = self.core.update(k_isotropic_exp = k)
    r = self.r_factor()
    if(self.verbose):
      print("      r1: %6.4f"%r1)
      print("      r2: %6.4f"%r2)
      print("      r_final: %6.4f (r_low: %6.4f)"%(r, self._r_low()))
    return r

  def try_scale(self,
                k_isotropic_exp=None,
                k_isotropic=None,
                k_mask=None,
                k_anisotropic=None,
                selection=None):
    if(k_isotropic_exp is None): k_isotropic_exp = self.core.k_isotropic_exp
    if(k_isotropic is None):     k_isotropic     = self.core.k_isotropic
    if(k_mask is None):          k_mask          = self.core.k_mask()
    if(k_anisotropic is None):   k_anisotropic   = self.core.k_anisotropic
    c = mmtbx.arrays.init(
      f_calc          = self.core.f_calc,
      f_masks         = self.core.f_mask(),
      k_isotropic_exp = k_isotropic_exp,
      k_isotropic     = k_isotropic,
      k_anisotropic   = k_anisotropic,
      k_masks         = k_mask)
    sel = self.selection_work.data()
    if(selection is not None): sel = selection & sel
    return bulk_solvent.r_factor(self.f_obs.data(), c.f_model.data(), sel)

  def r_all(self):
    return bulk_solvent.r_factor(self.f_obs.data(),
      self.core.f_model.data())

  def r_factor(self):
    return bulk_solvent.r_factor(self.f_obs.data(),
      self.core.f_model.data(), self.selection_work.data())

  def r_work(self):
    return bulk_solvent.r_factor(self.f_obs.data(),
      self.core.f_model.data(), ~self.r_free_flags.data())

  def _r_high(self):
    return bulk_solvent.r_factor(self.f_obs.data(),
      self.core.f_model.data(), self.high_resolution_selection)

  def _r_low(self):
    return bulk_solvent.r_factor(self.f_obs.data(),
      self.core.f_model.data(), self.low_resolution_selection)

  def _high_resolution_selection(self):
    return self.bin_selections[len(self.bin_selections)-1] & \
      self.selection_work.data()

  def _low_resolution_selection(self):
    return self.bin_selections[0] & self.selection_work.data()

  def populate_bin_to_individual_k_mask_linear_interpolation(self, k_mask_bin):
    assert len(k_mask_bin) == len(self.cores_and_selections)
    def linear_interpolation(x1,x2,y1,y2):
      k=0
      if(x1!=x2): k=(y2-y1)/(x2-x1)
      b = y1-k*x1
      return k,b
    result1 = flex.double(self.f_obs.size(), -1)
    result2 = flex.double(self.f_obs.size(), -1)
    result  = flex.double(self.f_obs.size(), -1)
    for i, cas in enumerate(self.cores_and_selections):
      selection, zzz, zzz, zzz = cas
      x1,x2 = self.ss_bin_values[i][0], self.ss_bin_values[i][1]
      y1 = k_mask_bin[i]
      if(i==len(k_mask_bin)-1):
        y2 = k_mask_bin[i-1]
      else:
        y2 = k_mask_bin[i+1]
      k,b = linear_interpolation(x1=x1,x2=x2,y1=y1,y2=y2)
      bulk_solvent.set_to_linear_interpolated(self.ss,k,b,selection,result1)
      result2.set_selected(selection, y1)
      r1 = self.try_scale(k_mask = result1, selection=selection) # XXX inefficient
      r2 = self.try_scale(k_mask = result2, selection=selection) # XXX inefficient
      if(r1<r2):
        bulk_solvent.set_to_linear_interpolated(self.ss,k,b,selection,result)
      else:
        result.set_selected(selection, y1)
    assert (result < 0).count(True) == 0
    return result

  def get_k_total(self, selection=None):
    if(selection is None):
      selection = flex.bool(self.core.k_isotropic.size(), True)
    scale = self.core.k_anisotropic.select(selection) * \
            self.core.k_isotropic.select(selection) * \
            self.core.k_isotropic_exp.select(selection)
    scale_k1 = bulk_solvent.scale(
      self.f_obs.data(),
      self.core.f_model.data(), selection)
    return scale*scale_k1

  def k_mask_grid_search(self, r_start):
    if(self.verbose):
      print("    k_mask_grid_search:", file=self.log)
      print("      r_start: %6.4f (r_low: %6.4f)"%(r_start,self._r_low()))
    #k_mask_trial_range = flex.double([i/1000. for i in range(0,650,50)])
    k_mask_trial_range = flex.double([i/1000. for i in range(0,1010,10)])
    k_mask             = flex.double(self.f_obs.size(), 0)
    k_mask_bin         = flex.double()
    k_isotropic        = flex.double(self.f_obs.size(), 0)
    k_total = self.get_k_total()
    for i_cas, cas in enumerate(self.cores_and_selections):
      selection, core, selection_use, sel_work = cas
      f_obs = self.f_obs.select(selection)
      k_total_ = k_total.select(selection)
      k_mask_bin_, k_isotropic_bin_ = \
        bulk_solvent.k_mask_and_k_overall_grid_search(
          f_obs.data()/k_total_,
          core.f_calc.data(),
          core.f_mask().data(),
          k_mask_trial_range,
          selection_use)
      k_mask_bin.append(k_mask_bin_)
      k_mask.set_selected(selection, k_mask_bin_)
      k_isotropic.set_selected(selection, k_isotropic_bin_)

    k_mask_bin_smooth = self.smooth(k_mask_bin)
    k_mask = self.populate_bin_to_individual_k_mask_linear_interpolation(
      k_mask_bin = k_mask_bin_smooth)
    k_isotropic = self._k_isotropic_as_scale_k1(r_start=r_start,k_mask = k_mask)

    r_try = self.try_scale(k_mask = k_mask, k_isotropic = k_isotropic)
    if(r_try<r_start):
      self.core = self.core.update(k_masks = k_mask, k_isotropic = k_isotropic)
      # ????
      self.bss_result.k_mask_bin_orig   = k_mask_bin
      self.bss_result.k_mask_bin_smooth = k_mask_bin_smooth
      self.bss_result.k_mask            = k_mask
      self.bss_result.k_isotropic       = k_isotropic
    r = self.r_factor()
    if(self.verbose):
      print("      r_final: %6.4f (r_low: %6.4f)"%(r,self._r_low()))
    return r

  def apply_overall_scale(self):
    scale_k1 = bulk_solvent.scale(self.f_obs.data(),
      self.core.f_model.data(), self.selection_work.data())
    self.core = self.core.update(k_isotropic = self.core.k_isotropic*scale_k1)

  def is_converged(self, r_start, tolerance=1.e-4):
    self.r_final = self.r_factor()
    result = False
    if((r_start<=self.r_final) or
       (r_start>self.r_final and abs(r_start-self.r_final)<tolerance)):
      result = True
    diff = abs(round(r_start,4)-round(self.r_final,4))
    if(diff<tolerance): result = True
    return result

  def show(self):
    b = self.bss_result
    print("  Statistics in resolution bins:", file=self.log)
    fmt="  %7.5f %6.2f -%6.2f %5.1f %5d %-6s %-6s %-6s  %6.3f %6.3f %8.2f %6.4f"
    f_model = self.core.f_model.data()
    print("  s^2      Resolution    Compl Nrefl k_mask                 k_iso  k_ani <Fobs>   R", file=self.log)
    print("                (A)        (%)       orig   smooth average", file=self.log)
    k_mask_bin_orig_   = str(None)
    k_mask_bin_smooth_ = str(None)
    k_mask_bin_approx_ = str(None)
    for i_sel, cas in enumerate(self.cores_and_selections):
      selection, core, selection_use, sel_work = cas
      sel = sel_work
      ss_ = self.ss_bin_values[i_sel][2]
      if(b is not None and self.bss_result.k_mask_bin_orig is not None):
        k_mask_bin_orig_ = "%6.4f"%self.bss_result.k_mask_bin_orig[i_sel]
      if(b is not None and self.bss_result.k_mask_bin_smooth is not None):
        k_mask_bin_smooth_ = "%6.4f"%self.bss_result.k_mask_bin_smooth[i_sel]
      k_mask_bin_averaged_ = "%6.4f"%flex.mean(self.core.k_mask().select(sel))
      d_             = self.d_spacings.data().select(sel)
      d_min_         = flex.min(d_)
      d_max_         = flex.max(d_)
      n_ref_         = d_.size()
      f_obs_         = self.f_obs.select(sel)
      f_obs_mean_    = flex.mean(f_obs_.data())
      k_isotropic_   = flex.mean(self.core.k_isotropic.select(sel))
      k_anisotropic_ = flex.mean(self.core.k_anisotropic.select(sel))
      cmpl_          = f_obs_.completeness(d_max=d_max_)*100.
      r_             = bulk_solvent.r_factor(f_obs_.data(),f_model.select(sel))
      print(fmt%(ss_, d_max_, d_min_, cmpl_, n_ref_,
        k_mask_bin_orig_, k_mask_bin_smooth_,k_mask_bin_averaged_,
        k_isotropic_, k_anisotropic_, f_obs_mean_, r_), file=self.log)

  def _k_isotropic_as_scale_k1(self, r_start, k_mask=None):
    k_isotropic = flex.double(self.ss.size(), -1)
    if(k_mask is None): k_mask = self.core.k_mask()
    core_data = mmtbx.arrays.init(
      f_calc          = self.core.f_calc,
      f_masks         = self.core.f_mask(),
      k_isotropic_exp = self.core.k_isotropic_exp,
      k_anisotropic   = self.core.k_anisotropic,
      k_masks         = k_mask).data
    for i_cas, cas in enumerate(self.cores_and_selections):
      selection, core, selection_use, sel_work = cas
      scale_k1 = bulk_solvent.scale(self.f_obs.data(),
        core_data.f_model, sel_work)
      k_isotropic = k_isotropic.set_selected(selection, scale_k1)
    assert k_isotropic.count(-1.) == 0
    return k_isotropic

  def estimate_scale_k1(self, cutoff=4, width=1, min_reflections=500):
    cutoff = min(cutoff, self.f_obs.d_min()+width)
    sel_high = self.d_spacings.data()<cutoff
    sel_high = sel_high & self.selection_work.data()
    scale_k1 = 1
    if(sel_high.count(True)>min_reflections):
      core = self.core.select(selection = sel_high)
      f_obs = self.f_obs.select(sel_high)
      fm = core.k_isotropic_exp * core.k_anisotropic * (core.f_calc.data() +
        core.k_mask() * core.f_mask().data())
      scale_k1 = bulk_solvent.scale(f_obs.data(), fm)
    return scale_k1

  def bulk_solvent_scaling(self, r_start):
    if(self.verbose):
      print("    bulk_solvent_scaling:", file=self.log)
      print("      r_start: %6.4f (r_low: %6.4f)"%(r_start,self._r_low()))
    k_mask     = flex.double(self.f_obs.size(), -1)
    k_mask_bin = flex.double()
    k_mask_trial_range = flex.double([i/1000. for i in range(0,1000,10)])
    k_total = self.get_k_total()
    def get_k_mask_trial_range(x, shift=0.05):
      result = flex.double([x])
      if(x > 1): x = 1
      inc = max(0,x-shift)
      while inc<=x+shift+1.e-3:
        result.append(inc)
        inc+=0.01
      return result
    for i_cas, cas in enumerate(self.cores_and_selections):
      selection, core, selection_use, sel_work = cas
      f_obs  = self.f_obs.select(selection).data()
      f_calc = core.f_calc.data()  *k_total.select(selection)
      f_mask = core.f_mask().data()*k_total.select(selection)
      if(self.scale_method == "k_iso_k_mask_anal"):
        obj = bulk_solvent.overall_and_bulk_solvent_scale_coefficients_analytical(
          f_obs     = f_obs,
          f_calc    = f_calc,
          f_mask    = f_mask,
          selection = selection_use)
        k_mask_bin.append(obj.x_best)
        k_mask.set_selected(selection, obj.x_best)
      elif(self.scale_method == "k_mask_anal"):
        obj = bulk_solvent.bulk_solvent_scale_coefficients_analytical(
          f_obs     = f_obs,
          f_calc    = f_calc,
          f_mask    = f_mask,
          selection = selection_use)
        k_mask_bin.append(obj.x_best)
        k_mask.set_selected(selection, obj.x_best)
      elif(self.scale_method == "k_mask_r_grid_search"):
        k_mask_bin_, k_isotropic_bin_ = \
          bulk_solvent.k_mask_and_k_overall_grid_search(
            f_obs,
            f_calc,
            f_mask,
            k_mask_trial_range,
            selection_use)
        k_mask_bin.append(k_mask_bin_)
        k_mask.set_selected(selection, k_mask_bin_)
      elif(self.scale_method == "combo"):
        r = flex.double()
        k = flex.double()
        #
        if(self.bss_result.k_mask_bin_orig is not None):
          x0 = self.bss_result.k_mask_bin_orig[i_cas]
          k_mask.set_selected(selection, x0)
          r0 = self.try_scale(k_mask = k_mask, selection=selection)
          r.append(r0)
          k.append(x0)
        #
        fmv = flex.min(flex.abs(f_mask).select(selection_use))
        if(abs(fmv)>1.e-9):
          obj1 = bulk_solvent.overall_and_bulk_solvent_scale_coefficients_analytical(
            f_obs     = f_obs,
            f_calc    = f_calc,
            f_mask    = f_mask,
            selection = selection_use)
          k_mask.set_selected(selection, obj1.x_best)
          k.append(obj1.x_best)
        else:
          k_mask.set_selected(selection, 0)
          k.append(0)
        r.append(self.try_scale(k_mask = k_mask, selection=selection))
        #
        s = flex.sort_permutation(r)
        x = k.select(s)[0]
        # fine-sample k_mask around minimum of LS to fall into minimum of R
        k_mask_bin_, k_isotropic_bin_ = \
          bulk_solvent.k_mask_and_k_overall_grid_search(
            f_obs,
            f_calc,
            f_mask,
            get_k_mask_trial_range(x = x),
            selection_use)
        k_mask_bin.append(k_mask_bin_)
        k_mask.set_selected(selection, k_mask_bin_)

        #k_mask_bin.append(x)
        #k_mask.set_selected(selection, x)
      else: assert 0
    #
    k_mask_bin_smooth = self.smooth(k_mask_bin)
    k_mask = self.populate_bin_to_individual_k_mask_linear_interpolation(
      k_mask_bin = k_mask_bin_smooth)
    k_isotropic = self._k_isotropic_as_scale_k1(r_start=r_start,k_mask = k_mask)

    r_try = self.try_scale(k_mask = k_mask, k_isotropic = k_isotropic)
    if(r_try<r_start):
      self.core = self.core.update(k_isotropic = k_isotropic, k_masks = k_mask)
    r = self.r_factor()
    if(self.verbose):
      print("      r_final: %6.4f (r_low: %6.4f)"%(r,self._r_low()))
    return r

  def smooth(self, x):
    result = moving_average(x = x)
    result_ = flex.double(len(result), 0)
    for i, r in enumerate(result):
      d = 1/math.sqrt(self.ss_bin_values[i][1])/2
      if(r==0 and d<3): break
      result_[i]=r
    return result_

  def format_scale_matrix(self, m=None, log=None):
    sm = m
    if(sm is None): sm = self.scale_matrices
    out = log
    if(sm is None):
      print("  k_anisotropic=1", file=log)
      return
    if(len(sm)<=6):
      print("      b_cart(11,22,33,12,13,23):",\
        ",".join([str("%8.4f"%i).strip() for i in sm]), file=out)
    else:
      v0=[]
      v1=[]
      for i, a in enumerate(sm):
        if(i in [0,2,4,6,8,10]): v1.append(a)
        else: v0.append(a)
      print("      V0:",\
        ",".join([str("%8.4f"%i).strip() for i in v0]), file=out)
      print("      V1:",\
        ",".join([str("%8.4f"%i).strip() for i in v1]), file=out)

  def anisotropic_scaling(self, r_start, use_highres):
    r_expanal, r_poly, r_expmin = None,None,None
    k_anisotropic_expanal, k_anisotropic_poly, \
      k_anisotropic_expmin = None, None, None
    scale_matrix_expanal, scale_matrix_poly, scale_matrix_expmin= None,None,None
    sel         = self.selection_work.data()

    if(use_highres):
      sel_ = self.f_obs.d_spacings().data() < self.d_hilo
      sel = sel & sel_


    f_model_abs = flex.abs(self.core.f_model_no_aniso_scale.data().select(sel))
    f_obs       = self.f_obs.data().select(sel)
    mi          = self.f_obs.indices().select(sel)
    uc          = self.f_obs.unit_cell()
    mi_all      = self.f_obs.indices()
    # try exp_anal
    if(self.try_expanal):
      obj = bulk_solvent.aniso_u_scaler(
        f_model_abs    = f_model_abs,
        f_obs          = f_obs,
        miller_indices = mi,
        adp_constraint_matrix = self.adp_constraints.gradient_sum_matrix())
      u_star = self.adp_constraints.all_params(tuple(obj.u_star_independent))
      scale_matrix_expanal = adptbx.u_as_b(adptbx.u_star_as_u_cart(uc, u_star))
      k_anisotropic_expanal = ext.k_anisotropic(mi_all, u_star)
      r_expanal = self.try_scale(k_anisotropic = k_anisotropic_expanal)
      if(self.verbose):
        print("      r_expanal: %6.4f"%r_expanal, file=self.log)
    # try poly
    if(self.try_poly):
      obj = bulk_solvent.aniso_u_scaler(
        f_model_abs    = f_model_abs,
        f_obs          = f_obs,
        miller_indices = mi,
        unit_cell      = uc)
      scale_matrix_poly = obj.a
      k_anisotropic_poly = ext.k_anisotropic(mi_all, obj.a, uc)
      r_poly = self.try_scale(k_anisotropic = k_anisotropic_poly)
      if(self.verbose):
        print("      r_poly   : %6.4f"%r_poly, file=self.log)
    # pre-analyze
    force_to_use_expmin=False
    if(k_anisotropic_poly is not None and self.auto and r_poly<r_expanal and
       (k_anisotropic_poly<=0).count(True)>0):
      force_to_use_expmin = True
      self.try_expmin = True
    # try expmin
    if(self.try_expmin):
      zero = self.f_obs.select(sel).customized_copy(data =
        flex.complex_double(f_obs.size(), 0))
      if(self.u_star is None): self.u_star = [0,0,0,0,0,0]
      fm = mmtbx.f_model.manager_kbu(
        f_obs         = self.f_obs.select(sel),
        f_calc        = self.core.f_model_no_aniso_scale.select(sel),
        f_masks       = [zero],
        f_part1       = zero,
        f_part2       = zero,
        ss            = self.ss)
      obj = kbu_refinery.tgc(
        f_obs   = self.f_obs.select(sel),
        f_calc  = self.core.f_model_no_aniso_scale.select(sel),
        f_masks = [zero],
        ss      = self.ss,
        k_sols  = [0,],
        b_sols  = [0,],
        u_star  = self.u_star)
      obj.minimize_u()
      u_star = obj.kbu.u_star()
      self.u_star = u_star
      scale_matrix_expmin = adptbx.u_as_b(adptbx.u_star_as_u_cart(uc, u_star))
      k_anisotropic_expmin = ext.k_anisotropic(mi_all, u_star)
      r_expmin = self.try_scale(k_anisotropic = k_anisotropic_expmin)
      if(self.verbose): print("    r_expmin   : %6.4f"%r_expmin, file=self.log)
      if(force_to_use_expmin):
        self.core = self.core.update(k_anisotropic = k_anisotropic_expmin)
        if(self.verbose):
          self.format_scale_matrix(m=scale_matrix_expmin)
        return self.r_factor()
    # select best
    r = [(r_expanal, k_anisotropic_expanal, scale_matrix_expanal),
         (r_poly,    k_anisotropic_poly,    scale_matrix_poly),
         (r_expmin,  k_anisotropic_expmin,  scale_matrix_expmin)]
    r_best = r_start
    k_anisotropic_best = None
    scale_matrix_best = None
    for result in r:
      r_factor, k_anisotropic, scale_matrix = result
      if(r_factor is not None and r_factor < r_best):
        r_best = r_factor
        k_anisotropic_best = k_anisotropic.deep_copy()
        scale_matrix_best = scale_matrix[:]
    if(scale_matrix_best is None):
      if(self.verbose):
        print("      result rejected due to r-factor increase", file=self.log)
    else:
      self.scale_matrices = scale_matrix_best
      self.core = self.core.update(k_anisotropic = k_anisotropic_best)
      r_aniso = self.r_factor()
      if(self.verbose):
        self.format_scale_matrix()
        print("      r_final  : %6.4f"%r_aniso, file=self.log)
    return r_best

  def overall_isotropic_kb_estimate(self):
    k_total = self.core.k_isotropic * self.core.k_anisotropic * \
      self.core.k_isotropic_exp
    r = scitbx.math.gaussian_fit_1d_analytical(x=flex.sqrt(self.ss), y=k_total)
    return r.a, r.b

  def k_masks(self):
    return self.core.k_masks

  def k_isotropic(self):
    return self.core.k_isotropic*self.core.k_isotropic_exp

  def k_anisotropic(self):
    return self.core.k_anisotropic

  def apply_back_trace_of_overall_exp_scale_matrix(self, xray_structure=None):
    k,b=self.overall_isotropic_kb_estimate()
    k_total = self.core.k_isotropic * self.core.k_anisotropic * \
      self.core.k_isotropic_exp
    k,b,r = mmtbx.bulk_solvent.fit_k_exp_b_to_k_total(k_total, self.ss, k, b)
    if(r<0.7): self.k_exp_overall,self.b_exp_overall = k,b
    if(xray_structure is None): return None
    b_adj = 0
    if([self.k_exp_overall,self.b_exp_overall].count(None)==0 and k != 0):
      bs1 = xray_structure.extract_u_iso_or_u_equiv()*adptbx.u_as_b(1.)
      def split(b_trace, xray_structure):
        b_min = xray_structure.min_u_cart_eigenvalue()*adptbx.u_as_b(1.)
        b_res = min(0, b_min + b_trace+1.e-6)
        b_adj = b_trace-b_res
        xray_structure.shift_us(b_shift = b_adj)
        return b_adj, b_res
      b_adj,b_res=split(b_trace=self.b_exp_overall,xray_structure=xray_structure)
      k_new = self.k_exp_overall*flex.exp(-self.ss*b_adj)
      bs2 = xray_structure.extract_u_iso_or_u_equiv()*adptbx.u_as_b(1.)
      diff = bs2-bs1
      assert approx_equal(flex.min(diff), flex.max(diff))
      assert approx_equal(flex.max(diff), b_adj)
      self.core = self.core.update(
        k_isotropic = self.core.k_isotropic,
        k_isotropic_exp = self.core.k_isotropic_exp/k_new,
        k_masks = [m*flex.exp(-self.ss*b_adj) for m in self.core.k_masks])
    return group_args(
      xray_structure = xray_structure,
      k_isotropic    = self.k_isotropic(),
      k_anisotropic  = self.k_anisotropic(),
      k_mask         = self.k_masks(),
      b_adj          = b_adj)

# XXX SEVERE DUPLICATION
# XXX Consolidate with analogous function in bulk_solvnet_and_scaling.py :
# XXX apply_back_trace_of_overall_exp_scale_matrix
class tmp(object):
  def __init__(self, xray_structure, k_anisotropic, k_masks, ss):
    self.xray_structure = xray_structure
    self.k_anisotropic  = k_anisotropic
    self.k_masks        = k_masks
    self.ss             = ss
    #
    k_total = self.k_anisotropic
    r = scitbx.math.gaussian_fit_1d_analytical(x=flex.sqrt(self.ss), y=k_total)
    k,b = r.a, r.b
    #
    k,b,r = mmtbx.bulk_solvent.fit_k_exp_b_to_k_total(k_total, self.ss, k, b)
    k_exp_overall, b_exp_overall = None,None
    if(r<0.7): k_exp_overall, b_exp_overall = k,b
    if(self.xray_structure is None): return None
    b_adj = 0
    if([k_exp_overall, b_exp_overall].count(None)==0 and k != 0):
      bs1 = self.xray_structure.extract_u_iso_or_u_equiv()*adptbx.u_as_b(1.)
      def split(b_trace, xray_structure):
        b_min = xray_structure.min_u_cart_eigenvalue()*adptbx.u_as_b(1.)
        b_res = min(0, b_min + b_trace+1.e-6)
        b_adj = b_trace-b_res
        xray_structure.shift_us(b_shift = b_adj)
        return b_adj, b_res
      b_adj,b_res=split(b_trace=b_exp_overall,xray_structure=self.xray_structure)
      k_new = k_exp_overall*flex.exp(-self.ss*b_adj)
      bs2 = self.xray_structure.extract_u_iso_or_u_equiv()*adptbx.u_as_b(1.)
      diff = bs2-bs1
      assert approx_equal(flex.min(diff), flex.max(diff))
      assert approx_equal(flex.max(diff), b_adj)
      self.k_anisotropic = self.k_anisotropic/k_new
      self.k_masks = [m*flex.exp(-self.ss*b_adj) for m in self.k_masks]
