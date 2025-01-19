#include "scitbx/array_family/boost_python/flex_fwd.h"
//#include <cudatbx/cuda_base.cuh>
#include "simtbx/kokkos/simulation.h"
#include "simtbx/kokkos/simulation_kernels.h"
#include "kokkostbx/kokkos_utils.h"
#include "scitbx/array_family/flex_types.h"
#include "simtbx/kokkos/detector.h"

#define THREADS_PER_BLOCK_X 128
#define THREADS_PER_BLOCK_Y 1
#define THREADS_PER_BLOCK_TOTAL (THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y)

namespace simtbx {
namespace Kokkos {

  namespace af = scitbx::af;
  //refactor later into helper file
/*  static cudaError_t cudaMemcpyVectorDoubleToDevice(CUDAREAL *dst, const double *src, size_t vector_items) {
        CUDAREAL * temp = new CUDAREAL[vector_items];
        for (size_t i = 0; i < vector_items; i++) {
                temp[i] = src[i];
        }
        cudaError_t ret = cudaMemcpy(dst, temp, sizeof(*dst) * vector_items, cudaMemcpyHostToDevice);
        delete temp;
        return ret;
  }*/

/*  exascale_api::~exascale_api(){
    cudaSafeCall(cudaSetDevice(SIM.device_Id));

        cudaSafeCall(cudaFree(cu_beam_vector));
        cudaSafeCall(cudaFree(cu_spindle_vector));
        cudaSafeCall(cudaFree(cu_source_X));
        cudaSafeCall(cudaFree(cu_source_Y));
        cudaSafeCall(cudaFree(cu_source_Z));
        cudaSafeCall(cudaFree(cu_source_I));
        cudaSafeCall(cudaFree(cu_source_lambda));
        cudaSafeCall(cudaFree(cu_a0));
        cudaSafeCall(cudaFree(cu_b0));
        cudaSafeCall(cudaFree(cu_c0));
        cudaSafeCall(cudaFree(cu_mosaic_umats));
        cudaSafeCall(cudaFree(cu_polar_vector));
  }
*/

  template<> void
  exascale_api<large_array_policy>::add_energy_multichannel_mask_allpanel(
    af::shared<int> const ichannels,
    simtbx::Kokkos::kokkos_energy_channels & kec,
    simtbx::Kokkos::kokkos_detector<large_array_policy> & kdt,
    af::shared<std::size_t> const active_pixel_list,
    af::shared<double> const weight
  ){
    //SCITBX_CHECK_POINT; // tested by tst_unified.py
    kdt.set_active_pixels_on_GPU(active_pixel_list);

    // transfer source_I, source_lambda
    // the int arguments are for sizes of the arrays

    SCITBX_ASSERT(SIM.sources == ichannels.size()); /* For each nanoBragg source, this value instructs
    the simulation where to look for structure factors.  If -1, skip this source wavelength. */
    SCITBX_ASSERT(SIM.sources == weight.size());

    for (int ictr = 0; ictr < SIM.sources; ++ictr){
      if (ichannels[ictr] < 0) continue; // the ichannel array
      //printf("HA HA %4d channel %4d, %5.1f %5.1f %5.1f %10.5f %10.5g\n",ictr,ichannels[ictr],
      //  SIM.source_X[ictr], SIM.source_Y[ictr], SIM.source_Z[ictr],
      //  SIM.source_I[ictr], SIM.source_lambda[ictr]);

      ::Kokkos::resize(m_crystal_orientation, SIM.phisteps, SIM.mosaic_domains, 3);
      calc_CrystalOrientations(
      SIM.phi0, SIM.phistep, SIM.phisteps, m_spindle_vector, m_a0, m_b0, m_c0, SIM.mosaic_spread,
      SIM.mosaic_domains, m_mosaic_umats, m_crystal_orientation);

      // magic happens here: take pointer from singleton, temporarily use it for add Bragg iteration:
      vector_cudareal_t current_channel_Fhkl = kec.d_channel_Fhkl[ichannels[ictr]];

      vector_cudareal_t c_source_X = vector_cudareal_t("c_source_X", 0);
      vector_cudareal_t c_source_Y = vector_cudareal_t("c_source_Y", 0);
      vector_cudareal_t c_source_Z = vector_cudareal_t("c_source_Z", 0);
      vector_cudareal_t c_source_I = vector_cudareal_t("c_source_I", 0);
      vector_cudareal_t c_source_lambda = vector_cudareal_t("c_source_lambda", 0);
      double weighted_I = SIM.source_I[ictr] * weight[ictr];
      kokkostbx::transfer_double2kokkos(c_source_X, &(SIM.source_X[ictr]), 1);
      kokkostbx::transfer_double2kokkos(c_source_Y, &(SIM.source_Y[ictr]), 1);
      kokkostbx::transfer_double2kokkos(c_source_Z, &(SIM.source_Z[ictr]), 1);
      kokkostbx::transfer_double2kokkos(c_source_I, &(weighted_I), 1);
      kokkostbx::transfer_double2kokkos(c_source_lambda, &(SIM.source_lambda[ictr]), 1);

      // set up cell basis vectors for the diffuse parameters (convert vec4 to vec3)
      diffuse.a0 << SIM.a0[1],SIM.a0[2],SIM.a0[3]; diffuse.b0 << SIM.b0[1],SIM.b0[2],SIM.b0[3]; diffuse.c0 << SIM.c0[1],SIM.c0[2],SIM.c0[3];

      debranch_maskall_Kernel(
      kdt.m_panel_count, kdt.m_slow_dim_size, kdt.m_fast_dim_size, active_pixel_list.size(),
      SIM.oversample, SIM.point_pixel,
      SIM.pixel_size, m_subpixel_size, m_steps,
      SIM.detector_thickstep, SIM.detector_thicksteps, SIM.detector_thick, SIM.detector_attnlen,
      kdt.m_sdet_vector,
      kdt.m_fdet_vector,
      kdt.m_odet_vector,
      kdt.m_pix0_vector,
      kdt.m_distance,
      m_beam_vector,
      SIM.dmin, SIM.phisteps, 1,
      c_source_X, c_source_Y,
      c_source_Z,
      c_source_I, c_source_lambda,
      SIM.mosaic_domains, m_crystal_orientation,
      SIM.Na, SIM.Nb, SIM.Nc, SIM.V_cell,
      m_water_size, m_water_F, m_water_MW, simtbx::nanoBragg::r_e_sqr,
      SIM.fluence, simtbx::nanoBragg::Avogadro, SIM.spot_scale, SIM.integral_form,
      SIM.default_F,
      current_channel_Fhkl,
      kec.m_FhklParams,
      SIM.nopolar,
      m_polar_vector, SIM.polarization, SIM.fudge,
      kdt.m_active_pixel_list,
      diffuse,
      // return arrays:
      kdt.m_floatimage,
      kdt.m_omega_reduction,
      kdt.m_max_I_x_reduction,
      kdt.m_max_I_y_reduction,
      kdt.m_rangemap);
    //don't want to free the kec data when the nanoBragg goes out of scope, so switch the pointer
    // cu_current_channel_Fhkl = NULL;

      add_array(kdt.m_accumulate_floatimage, kdt.m_floatimage);
    }// loop over channels
  }

  template<> void
  exascale_api<small_whitelist_policy>::add_energy_multichannel_mask_allpanel(
    af::shared<int> const ichannels,
    simtbx::Kokkos::kokkos_energy_channels & kec,
    simtbx::Kokkos::kokkos_detector<small_whitelist_policy> & kdt,
    af::shared<std::size_t> const active_pixel_list,
    af::shared<double> const weight
  ){
    //SCITBX_CHECK_POINT; // tested by tst_memory_policy.py
    kdt.set_active_pixels_on_GPU(active_pixel_list);

    // transfer source_I, source_lambda
    // the int arguments are for sizes of the arrays

    SCITBX_ASSERT(SIM.sources == ichannels.size()); /* For each nanoBragg source, this value instructs
    the simulation where to look for structure factors.  If -1, skip this source wavelength. */
    SCITBX_ASSERT(SIM.sources == weight.size());

    for (int ictr = 0; ictr < SIM.sources; ++ictr){
      if (ichannels[ictr] < 0) continue; // the ichannel array
      //printf("HA HA %4d channel %4d, %5.1f %5.1f %5.1f %10.5f %10.5g\n",ictr,ichannels[ictr],
      //  SIM.source_X[ictr], SIM.source_Y[ictr], SIM.source_Z[ictr],
      //  SIM.source_I[ictr], SIM.source_lambda[ictr]);

      ::Kokkos::resize(m_crystal_orientation, SIM.phisteps, SIM.mosaic_domains, 3);
      calc_CrystalOrientations(
      SIM.phi0, SIM.phistep, SIM.phisteps, m_spindle_vector, m_a0, m_b0, m_c0, SIM.mosaic_spread,
      SIM.mosaic_domains, m_mosaic_umats, m_crystal_orientation);

      // magic happens here: take pointer from singleton, temporarily use it for add Bragg iteration:
      vector_cudareal_t current_channel_Fhkl = kec.d_channel_Fhkl[ichannels[ictr]];

      vector_cudareal_t c_source_X = vector_cudareal_t("c_source_X", 0);
      vector_cudareal_t c_source_Y = vector_cudareal_t("c_source_Y", 0);
      vector_cudareal_t c_source_Z = vector_cudareal_t("c_source_Z", 0);
      vector_cudareal_t c_source_I = vector_cudareal_t("c_source_I", 0);
      vector_cudareal_t c_source_lambda = vector_cudareal_t("c_source_lambda", 0);
      double weighted_I = SIM.source_I[ictr] * weight[ictr];
      kokkostbx::transfer_double2kokkos(c_source_X, &(SIM.source_X[ictr]), 1);
      kokkostbx::transfer_double2kokkos(c_source_Y, &(SIM.source_Y[ictr]), 1);
      kokkostbx::transfer_double2kokkos(c_source_Z, &(SIM.source_Z[ictr]), 1);
      kokkostbx::transfer_double2kokkos(c_source_I, &(weighted_I), 1);
      kokkostbx::transfer_double2kokkos(c_source_lambda, &(SIM.source_lambda[ictr]), 1);

      // set up cell basis vectors for the diffuse parameters (convert vec4 to vec3)
      diffuse.a0 << SIM.a0[1],SIM.a0[2],SIM.a0[3]; diffuse.b0 << SIM.b0[1],SIM.b0[2],SIM.b0[3]; diffuse.c0 << SIM.c0[1],SIM.c0[2],SIM.c0[3];

      debranch_maskall_low_memory_Kernel(
      kdt.m_panel_count, kdt.m_slow_dim_size, kdt.m_fast_dim_size, active_pixel_list.size(),
      SIM.oversample, SIM.point_pixel,
      SIM.pixel_size, m_subpixel_size, m_steps,
      SIM.detector_thickstep, SIM.detector_thicksteps, SIM.detector_thick, SIM.detector_attnlen,
      kdt.m_sdet_vector,
      kdt.m_fdet_vector,
      kdt.m_odet_vector,
      kdt.m_pix0_vector,
      kdt.m_distance,
      m_beam_vector,
      SIM.dmin, SIM.phisteps, 1,
      c_source_X, c_source_Y,
      c_source_Z,
      c_source_I, c_source_lambda,
      SIM.mosaic_domains, m_crystal_orientation,
      SIM.Na, SIM.Nb, SIM.Nc, SIM.V_cell,
      m_water_size, m_water_F, m_water_MW, simtbx::nanoBragg::r_e_sqr,
      SIM.fluence, simtbx::nanoBragg::Avogadro, SIM.spot_scale, SIM.integral_form,
      SIM.default_F,
      current_channel_Fhkl,
      kec.m_FhklParams,
      SIM.nopolar,
      m_polar_vector, SIM.polarization, SIM.fudge,
      kdt.m_active_pixel_list,
      diffuse,
      // return arrays:
      kdt.m_floatimage,
      kdt.m_omega_reduction,
      kdt.m_max_I_x_reduction,
      kdt.m_max_I_y_reduction,
      kdt.m_rangemap);
    //don't want to free the kec data when the nanoBragg goes out of scope, so switch the pointer
    // cu_current_channel_Fhkl = NULL;

      //for the small_whitelist specialization, have a special version of add_array() that specifies size
      add_array_limit(kdt.m_accumulate_floatimage, kdt.m_floatimage, kdt.m_floatimage.span());
    }// loop over channels
  }

} // Kokkos
} // simtbx
