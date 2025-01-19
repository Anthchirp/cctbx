from __future__ import division
from simtbx import nanoBragg # fix for issue #868

def get_exascale(interface, context):
  if context == "kokkos_gpu":
    from simtbx.kokkos import gpu_instance, gpu_energy_channels, gpu_detector, gpu_detector_small_whitelist
    from simtbx.kokkos import exascale_api, exascale_api_small_whitelist
  elif context == "cuda":
    from simtbx.gpu import gpu_instance, gpu_energy_channels, gpu_detector, exascale_api
  else: raise NotImplementedError(context)

  return dict(gpu_instance = gpu_instance, gpu_energy_channels = gpu_energy_channels,
              gpu_detector = gpu_detector, exascale_api = exascale_api,
              gpu_detector_small_whitelist = locals().get("gpu_detector_small_whitelist"),
              exascale_api_small_whitelist = locals().get("exascale_api_small_whitelist"))[interface]

