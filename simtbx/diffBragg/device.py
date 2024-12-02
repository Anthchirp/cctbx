from __future__ import division
import libtbx.load_env  # implicit import
DIFFBRAGG_HAS_KOKKOS = libtbx.env.build_options.enable_kokkos

if DIFFBRAGG_HAS_KOKKOS:
    # TODO fix simtbx.kokkos import ...
    # from simtbx.kokkos import gpu_instance
    from simtbx.diffBragg import initialize_kokkos, finalize_kokkos

import os
USE_KOKKOS_GPU = DIFFBRAGG_HAS_KOKKOS and ("DIFFBRAGG_USE_KOKKOS" in os.environ)


class DeviceWrapper:

    def __init__(self, gpu_id, force=False):
        """

        :param gpu_id: gpu device id  (int from 0 to Ngpu-1)
        :param force: force initialization of kokkos, otherwise, the os.environ will be searched for DIFFBRAGG_USE_KOKKOS
        """
        self.gpu_id = gpu_id
        self.force_kokkos_init = force
        self.is_using_kokkos_gpu = USE_KOKKOS_GPU

    def __enter__(self):
        if USE_KOKKOS_GPU or self.force_kokkos_init:
            initialize_kokkos(self.gpu_id)
        return self

    def __exit__(self, tp, val, tbk):
        if USE_KOKKOS_GPU or self.force_kokkos_init:
            finalize_kokkos()
