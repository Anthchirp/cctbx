"""
Extend the previous tests in tst_unified
New tests question if there is a difference between all-pixel vs. shoebox-only calculation
Old style (policy == "large_array"): all-pixel and shoebox-only have identical GPU memory footprint
New style (policy == "small_whitelist"): all-pixel and shoebox-only have very different GPU memory footprint

Explanation of function:
  representation of pixels on GPU:
    simulation_kernels.h:
      kokkosSpotsKernel: vector_float_t floatimage
      debranch_maskall_Kernel: vector_float_t floatimage

  calling pattern for the kernels:
    simulation.cpp:  add_energy_channel_from_gpu_amplitudes calls kokkosSpotsKernel (not a target)
    simulation.cpp:  add_energy_channel_mask_allpanel calls debranch_maskall_Kernel, tested by tst_shoeboxes
    simulation.cpp:  add_energy_multichannel_mask_allpanel calls debranch_maskall_Kernel, tested by tst_unified
    each of these calls takes GPU::m_accumulate_floatimage += GPU::m_floatimage

  python calls in gpu_detector that interact with data:
    .def("scale_in_place", "Multiply by a scale factor on the GPU")
    .def("write_raw_pixels", "Update CPU raw_pixels on host with array from GPU::m_accumulate_floatimage")
    .def("get_raw_pixels", "return multipanel detector pixels from GPU::m_accumulate_floatimage as a flex array")
    .def("get_whitelist_raw_pixels", "return only the raw pixels of the whitelist selection, as a 1D flex array, from first memory positions in GPU::m_accumulate_floatimage")

Brief synopsis of this test suite:
1) simple whole-image simulation using add_energy_channel_from_gpu_amplitudes()
2) comparison whole-image simulation using add_energy_multichannel_mask_allpanel()
   also includes instrumentation to show CPU & GPU memory consumption
3) repeat step 2, but only using a whitelist of a few thousand pixels
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import dxtbx
from scitbx.array_family import flex
from scitbx.matrix import sqr
from simtbx.nanoBragg import nanoBragg, shapetype
from simtbx.nanoBragg.tst_gauss_argchk import basic_crystal, basic_beam, basic_detector, amplitudes
from simtbx import get_exascale
from simtbx.tests.tst_unified import several_wavelength_case as several_wavelength_case_unified

import subprocess as sp
import os

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def parse_input():
  from iotbx.phil import parse
  master_phil="""
    context = *kokkos_gpu
      .type = choice
      .optional = False
      .help = backend for parallel execution
  """
  phil_scope = parse(master_phil)
  # The script usage
  import libtbx.load_env # implicit import
  from dials.util.options import ArgumentParser
  # Create the parser
  parser = ArgumentParser(
        usage="\n libtbx.python tst_policy",
        phil=phil_scope,
        epilog="test GPU memory policies large_array vs small_whitelist")
  # Parse the command line. quick_parse is required for MPI compatibility
  params, options = parser.parse_args(show_diff_phil=True,quick_parse=True)
  return params,options

class several_wavelength_case_policy (several_wavelength_case_unified):

  def modularized_exafel_api_for_GPU(self, params, argchk=False, sources=False, scale=None):
    """similar to parent class method, but no background, no diffuse, use self.*"""
    gpu_channels_type = get_exascale("gpu_energy_channels",params.context)
    gpu_channels_singleton = gpu_channels_type (deviceId = 0)

    self.SIM = nanoBragg(self.DETECTOR, self.BEAM, panel_id=0)
    self.SIM.adc_offset_adu=0
    self.SIM.device_Id = 0

    assert gpu_channels_singleton.get_deviceID()==self.SIM.device_Id # API test
    assert gpu_channels_singleton.get_nchannels() == 0 # uninitialized
    for x in range(len(self.flux)):
          gpu_channels_singleton.structure_factors_to_GPU_direct(
           x, self.sfall_channels[x].indices(), self.sfall_channels[x].data())
    assert gpu_channels_singleton.get_nchannels() == len(self.flux) #API test
    self.SIM.Ncells_abc = (20,20,20)
    self.SIM.Amatrix = sqr(self.CRYSTAL.get_A()).transpose()
    self.SIM.oversample = 2
    self.SIM.xtal_shape = shapetype.Gauss
    self.SIM.interpolate = 0
    # allocate GPU arrays
    self.gpu_simulation = get_exascale("exascale_api",params.context)(nanoBragg = self.SIM)
    self.gpu_simulation.allocate()
    self.gpu_detector = get_exascale("gpu_detector",params.context)(
                 deviceId=self.SIM.device_Id, detector=self.DETECTOR, beam=self.BEAM)
    self.gpu_detector.each_image_allocate()
    # self.gpu_detector.show_summary()

    # two completely independent interfaces to loop over energies
    if sources is False: # original exascale API, explicit energy loop in Python
      for x in range(len(self.flux)):
        self.SIM.wavelength_A = self.wavlen[x]
        print("USE_EXASCALE_API+++++++++++++ Wavelength %d=%.6f, Flux %.6e, Fluence %.6e"%(
            x, self.SIM.wavelength_A, self.SIM.flux, self.SIM.fluence))
        self.gpu_simulation.add_energy_channel_from_gpu_amplitudes(
            x, gpu_channels_singleton, self.gpu_detector,
            weight = self.frac[x])
    else: # loop in C++; precludes using sources for divergence; uses sources from nanoBragg table
      self.set_pythony_beams(self.SIM)
      NN = 0 # compute the number of pixels
      for panel in self.DETECTOR:
          sz = panel.get_image_size()
          NN += sz[0]*sz[1]

      #print("GPU detector summary",self.DETECTOR)
      from simtbx.diffBragg.utils import memory_report; print(memory_report())
      print(get_gpu_memory())
      self.gpu_simulation.add_energy_multichannel_mask_allpanel(
            ichannels = flex.int(range(len(self.flux))),
            gpu_amplitudes = gpu_channels_singleton,
            gpu_detector = self.gpu_detector,
            pixel_active_list_ints = flex.size_t(range(NN)),
            weights = self.frac/len(self.frac)
      )

    per_image_scale_factor = self.domains_per_crystal # 1.0
    self.gpu_detector.scale_in_place(per_image_scale_factor) # apply scale directly on GPU
    if sources is False:
      self.SIM.wavelength_A = self.BEAM.get_wavelength() # return to canonical energy for subsequent background
    else:
      self.reset_pythony_beams(self.SIM)

    self.gpu_detector.write_raw_pixels(self.SIM)  # updates SIM.raw_pixels from GPU
    if self.special == 1: #special test breaks encapsulation
      self.special_test_case_1(self.SIM,self.gpu_detector,scale)
      # testing no garbage collect gpu_detector.each_image_free()
      return
    self.data_array = self.gpu_detector.get_raw_pixels()
    assert self.data_array.focus()[0] == len(self.DETECTOR) # number of panels
    if len(self.DETECTOR) == 1: # if one panel, we can assert data are the same both ways
      single_size = self.data_array.focus()[1:3]
      self.data_array.reshape(flex.grid((single_size[0],single_size[1])))
      assert np.allclose(self.SIM.raw_pixels, self.data_array)
    # deallocate GPU arrays afterward
    # testing no garbage collect gpu_detector.each_image_free()
    return

  def specialized_api_for_whitelist(self, params, whitelist_pixels, argchk=False, sources=True, scale=None):
    gpu_channels_type = get_exascale("gpu_energy_channels",params.context)
    gpu_channels_singleton = gpu_channels_type (deviceId = 0)

    self.SIM = nanoBragg(self.DETECTOR, self.BEAM, panel_id=0)
    self.SIM.adc_offset_adu=0
    self.SIM.device_Id = 0

    assert gpu_channels_singleton.get_deviceID()==self.SIM.device_Id # API test
    assert gpu_channels_singleton.get_nchannels() == 0 # uninitialized
    for x in range(len(self.flux)):
          gpu_channels_singleton.structure_factors_to_GPU_direct(
           x, self.sfall_channels[x].indices(), self.sfall_channels[x].data())
    assert gpu_channels_singleton.get_nchannels() == len(self.flux) #API test
    self.SIM.Ncells_abc = (20,20,20)
    self.SIM.Amatrix = sqr(self.CRYSTAL.get_A()).transpose()
    self.SIM.oversample = 2
    self.SIM.xtal_shape = shapetype.Gauss
    self.SIM.interpolate = 0
    # allocate GPU arrays
    self.gpu_simulation = get_exascale("exascale_api",params.context)(nanoBragg = self.SIM)
    self.gpu_simulation.allocate()
    self.gpu_detector = get_exascale("gpu_detector",params.context)(
                 deviceId=self.SIM.device_Id, detector=self.DETECTOR, beam=self.BEAM)
    self.gpu_detector.each_image_allocate()
    # self.gpu_detector.show_summary()

    assert sources
    self.set_pythony_beams(self.SIM)
    from simtbx.diffBragg.utils import memory_report; print(memory_report())
    print(get_gpu_memory())
    self.gpu_simulation.add_energy_multichannel_mask_allpanel(
            ichannels = flex.int(range(len(self.flux))),
            gpu_amplitudes = gpu_channels_singleton,
            gpu_detector = self.gpu_detector,
            pixel_active_list_ints = whitelist_pixels,
            weights = self.frac/len(self.frac)
    )

    per_image_scale_factor = self.domains_per_crystal # 1.0
    self.gpu_detector.scale_in_place(per_image_scale_factor) # apply scale directly on GPU
    self.reset_pythony_beams(self.SIM)
    self.whitelist_values = self.gpu_detector.get_whitelist_raw_pixels(whitelist_pixels)

  def specialized_api_for_whitelist_low_memory(self, params, whitelist_pixels, argchk=False, sources=True, scale=None):
    gpu_channels_type = get_exascale("gpu_energy_channels",params.context)
    gpu_channels_singleton = gpu_channels_type (deviceId = 0)

    self.SIM = nanoBragg(self.DETECTOR, self.BEAM, panel_id=0)
    self.SIM.adc_offset_adu=0
    self.SIM.device_Id = 0

    assert gpu_channels_singleton.get_deviceID()==self.SIM.device_Id # API test
    assert gpu_channels_singleton.get_nchannels() == 0 # uninitialized
    for x in range(len(self.flux)):
          gpu_channels_singleton.structure_factors_to_GPU_direct(
           x, self.sfall_channels[x].indices(), self.sfall_channels[x].data())
    assert gpu_channels_singleton.get_nchannels() == len(self.flux) #API test
    self.SIM.Ncells_abc = (20,20,20)
    self.SIM.Amatrix = sqr(self.CRYSTAL.get_A()).transpose()
    self.SIM.oversample = 2
    self.SIM.xtal_shape = shapetype.Gauss
    self.SIM.interpolate = 0
    # allocate GPU arrays
    self.gpu_simulation = get_exascale("exascale_api_small_whitelist",params.context)(nanoBragg = self.SIM)
    self.gpu_simulation.allocate()
    self.gpu_detector = get_exascale("gpu_detector_small_whitelist",params.context)(
                 deviceId=self.SIM.device_Id, detector=self.DETECTOR, beam=self.BEAM)

    self.gpu_detector.each_image_allocate(n_pixels = whitelist_pixels.size() )
    # self.gpu_detector.show_summary()

    assert sources
    self.set_pythony_beams(self.SIM)
    from simtbx.diffBragg.utils import memory_report; print(memory_report())
    print(get_gpu_memory())
    self.gpu_simulation.add_energy_multichannel_mask_allpanel(
            ichannels = flex.int(range(len(self.flux))),
            gpu_amplitudes = gpu_channels_singleton,
            gpu_detector = self.gpu_detector,
            pixel_active_list_ints = whitelist_pixels,
            weights = self.frac/len(self.frac)
    )

    per_image_scale_factor = self.domains_per_crystal # 1.0
    self.gpu_detector.scale_in_place(per_image_scale_factor) # apply scale directly on GPU
    self.reset_pythony_beams(self.SIM)
    whitelist_idx = flex.size_t(range(whitelist_pixels.size()))
    self.whitelist_values = self.gpu_detector.get_whitelist_raw_pixels(whitelist_idx)

def get_whitelist_from_refls(prefix,SIM=None):
    #image_size = len(SIM.raw_pixels)
    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory
    experiments = ExperimentListFactory.from_json_file("%s_imported.expt"%prefix, check_format = False)
    reflections = flex.reflection_table.from_file("%s_strong.refl"%prefix)
    assert len(experiments)==1
    assert len(experiments[0].detector)==1
    panel = experiments[0].detector[0]
    jslow = panel.get_image_size()[0]
    shoebox_pixels = flex.size_t()
    for iitem in range(len(reflections)):
      item = reflections[iitem]
      box = item['bbox']
      for islow in range(box[2], box[3]):
        for ifast in range(box[0], box[1]):
          idx = islow * jslow + ifast
          shoebox_pixels.append(idx)
    return shoebox_pixels

def get_refls_from_dials(data,prefix):
  os.system(
  "dials.import output.experiments=%s_imported.expt output.log=%s_imported.log %s"%(prefix,prefix,data))
  os.system(
  "dials.find_spots output.reflections=%s_strong.refl output.log=%s_strong.log %s"%(prefix,prefix,data))

def run_all(params):
  # make the dxtbx objects
  BEAM = basic_beam()
  DETECTOR = basic_detector()
  CRYSTAL = basic_crystal()
  SF_model = amplitudes(CRYSTAL)
  # Famp = SF_model.Famp # simple uniform amplitudes
  SF_model.random_structure(CRYSTAL)
  SF_model.ersatz_correct_to_P1()

  # determine overall scale factor
  print("\n# Use case 1 (%s). Monochromatic source"%params.context)
  SWC1 = several_wavelength_case_unified(BEAM, DETECTOR, CRYSTAL, SF_model, weights=flex.double([1.]))
  SIM1 = SWC1.modularized_exafel_api_for_GPU(params=params, argchk=False, gpu_background=True)
  scale = SIM1.get_intfile_scale() # case 1 is the overall scale factor to write all result files
  SIM1.to_cbf("test_policy_%s_001.cbf"%(params.context), intfile_scale=scale)
  print("Scale is:",scale)

  # compute and save background pixels
  print("\n# Use case 0 (%s). Background only"%params.context)
  SWC = several_wavelength_case_unified(BEAM, DETECTOR, CRYSTAL, SF_model, weights=flex.double([0.]))
  SIM0 = SWC.modularized_exafel_api_for_GPU(params=params, argchk=False, gpu_background=True)
  SIM0.to_cbf("test_policy_%s_000.cbf"%(params.context), intfile_scale=scale)
  save_background = SIM0.raw_pixels
  print(save_background.focus())

  # treat the case of multichannel API
  print("\n# Use case 2 (%s). Monochromatic source using multichannel API"%params.context)
  SIM2=SWC1.modularized_exafel_api_for_GPU(params=params,argchk=False,gpu_background=False,sources=True)
  # combine Bragg + background
  SIM2.raw_pixels = SWC1.data_array + save_background
  SIM2.to_cbf("test_policy_%s_002.cbf"%(params.context), intfile_scale=scale)
  # Bragg spot intensity should scale exactly with weight, using multichannel API
  assert np.allclose(SIM1.raw_pixels, SIM2.raw_pixels) # equate monochromatic, comparing two interfaces

  # Now try to reproduce whole-image sims with persistent and accumulating memory
  NTRIALS = 5
  SWCs=[]
  for x in range(NTRIALS):
    print("All-pixel iteration",x)
    SWCs.append(several_wavelength_case_policy(BEAM,DETECTOR,CRYSTAL,SF_model,weights=flex.double([1.])))
    SWCs[-1].modularized_exafel_api_for_GPU(params=params,argchk=False,sources=True)
    SWCs[-1].SIM.raw_pixels = SWCs[-1].data_array + save_background
    SWCs[-1].SIM.to_cbf("test_policy_trials_%03d.cbf"%(x), intfile_scale=scale)
    # Bragg spot intensity should scale exactly with weight, using multichannel API
    assert np.allclose(SWCs[-1].SIM.raw_pixels, SIM2.raw_pixels)

  free_gpu_before = get_gpu_memory()[0]
  del SWCs
  free_gpu_after = get_gpu_memory()[0]
  print((free_gpu_after - free_gpu_before)/NTRIALS,"free")
  assert (free_gpu_after - free_gpu_before)/NTRIALS >= 72 # old policy uses at least 72 MB per sim., actual value 76.8 MB

  get_refls_from_dials(data="test_policy_%s_002.cbf"%(params.context),
                      prefix="test_policy_%s"%(params.context))
  whitelist_pixels = get_whitelist_from_refls(SIM=SIM2,
                                              prefix="test_policy_%s"%(params.context))
  print(whitelist_pixels.focus())

  # simulate the bright spots only with the whitelist mechanism
  print("\n# Use case 3 (%s). Whitelist mechanism, old style with large arrays"%params.context)
  SWC3 = several_wavelength_case_policy(BEAM,DETECTOR,CRYSTAL,SF_model,weights=flex.double([1.]))
  SWC3.specialized_api_for_whitelist(whitelist_pixels=whitelist_pixels,
                                      params=params,argchk=False,sources=True)
  SIM3 = SWC3.SIM
  # combine Bragg + background
  image_size = len(SIM3.raw_pixels)
  working_raw_pixels = flex.double(image_size) # blank array
  working_raw_pixels.set_selected(whitelist_pixels, SWC3.whitelist_values)
  working_raw_pixels.reshape(flex.grid(SIM3.raw_pixels.focus()))
  SIM3.raw_pixels = working_raw_pixels + save_background
  SIM3.to_cbf("test_policy_%s_003.cbf"%(params.context), intfile_scale=scale)

  # get some quantitative assertion saying that whitelist is a reasonable image
  difference_image = SIM2.raw_pixels - SIM3.raw_pixels
  assert flex.sum(difference_image) > 0.
  print("difference",flex.sum(difference_image))
  large_diffs = (difference_image > 1.).count(True)
  print("number of large pixel differences",large_diffs)
  assert (large_diffs < 15000) # actual value 10471

  # Now reproduce whitelist sims showing accumulation of large persistent memory
  SWCs=[]
  for x in range(NTRIALS):
    print("\nWhitelist-only iteration",x)
    SWCs.append(several_wavelength_case_policy(BEAM,DETECTOR,CRYSTAL,SF_model,weights=flex.double([1.])))
    SWCs[-1].specialized_api_for_whitelist(whitelist_pixels=whitelist_pixels,params=params,argchk=False,sources=True)

  reference_whitelist_values = SWCs[-1].whitelist_values
  free_gpu_before = get_gpu_memory()[0]
  del SWCs
  free_gpu_after = get_gpu_memory()[0]
  old_memory_use = (free_gpu_after - free_gpu_before)/NTRIALS
  print(old_memory_use,"free")
  assert old_memory_use >= 50 # old policy uses at least 50 MB per sim., actual value 57.6 MB

  #figure out the minimum change needed to reduce memory consumption by factor of image_size/whitelist_size
  #accomplish the same with compile-time polymorphism
  #have side-by-side test in same python script
  # Reproduce whitelist sims with small-memory mechanism
  SWCs=[]
  for x in range(NTRIALS):
    print("\nWhitelist-only iteration with small memory",x)
    SWCs.append(several_wavelength_case_policy(BEAM,DETECTOR,CRYSTAL,SF_model,weights=flex.double([1.])))
    SWCs[-1].specialized_api_for_whitelist_low_memory(whitelist_pixels=whitelist_pixels,params=params,argchk=False,sources=True)
  #produce an output image file for intermediate debugging
  working_raw_pixels = flex.double(image_size) # blank array
  working_raw_pixels.set_selected(whitelist_pixels, SWCs[-1].whitelist_values)
  working_raw_pixels.reshape(flex.grid(SWCs[-1].SIM.raw_pixels.focus()))
  SWCs[-1].SIM.raw_pixels = working_raw_pixels + save_background
  SWCs[-1].SIM.to_cbf("test_policy_%s_004.cbf"%(params.context), intfile_scale=scale)

  #assert that the two methods give the same answer (whitelist with large memory vs. whitelist with small memory)
  assert np.allclose(reference_whitelist_values, SWCs[-1].whitelist_values)

  free_gpu_before = get_gpu_memory()[0]
  del SWCs
  free_gpu_after = get_gpu_memory()[0]
  new_memory_use = (free_gpu_after - free_gpu_before)/NTRIALS
  print(new_memory_use,"free")
  assert old_memory_use > 4.*new_memory_use # new policy cuts down at least 4-fold on memory, actual value **

def run_subset_for_NESAP_debug(params):
  # make the dxtbx objects
  BEAM = basic_beam()
  DETECTOR = basic_detector()
  CRYSTAL = basic_crystal()
  SF_model = amplitudes(CRYSTAL)
  # Famp = SF_model.Famp # simple uniform amplitudes
  SF_model.random_structure(CRYSTAL)
  SF_model.ersatz_correct_to_P1()
  whitelist_pixels = flex.size_t((11929,
 351293,351294,351295,352828,352829,352830,352831,354364,354365,
 354366,354367,355900,355901,355902,355903,357436,357437,357438,
 357439,352383,352384,352385,353919,353920,353921,355455,355456,
 355457,356991,356992,356993,354120,354121,354122,355656,355657))

  NTRIALS=5
  #figure out the minimum change needed to reduce memory consumption by factor of image_size/whitelist_size
  #accomplish the same with compile-time polymorphism
  #have side-by-side test in same python script
  # Reproduce whitelist sims with small-memory mechanism
  SWCs=[]
  for x in range(NTRIALS):
    print("\n Whitelist-only iteration with small memory",x)
    SWCs.append(several_wavelength_case_policy(BEAM,DETECTOR,CRYSTAL,SF_model,weights=flex.double([1.])))
    SWCs[-1].specialized_api_for_whitelist_low_memory(whitelist_pixels=whitelist_pixels,params=params,argchk=False,sources=True)

if __name__=="__main__":
  params,options = parse_input()
  # Initialize based on GPU context
  gpu_instance_type = get_exascale("gpu_instance", params.context)
  gpu_instance = gpu_instance_type(deviceId = 0)
  run_all(params)
  #run_subset_for_NESAP_debug(params)
print("OK")
