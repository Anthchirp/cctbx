#ifndef SIMTBX_KOKKOS_SIMULATION_H
#define SIMTBX_KOKKOS_SIMULATION_H

#include "scitbx/array_family/shared.h"
#include "scitbx/array_family/flex_types.h"
#include "simtbx/nanoBragg/nanoBragg.h"
#include "simtbx/kokkos/structure_factors.h"
#include "simtbx/kokkos/detector.h"
#include "kokkostbx/kokkos_types.h"
#include "kokkostbx/kokkos_vector3.h"
#include "kokkostbx/kokkos_matrix3.h"

using vec3 = kokkostbx::vector3<CUDAREAL>;
using mat3 = kokkostbx::matrix3<CUDAREAL>;
using crystal_orientation_t = Kokkos::View<vec3***, Kokkos::LayoutRight, MemSpace>; // [phisteps, domains, 3]

namespace simtbx { namespace Kokkos {
struct diffuse_api {
  inline diffuse_api() {};
  inline void show() {};
  bool enable = false;
  mat3 anisoG;
  mat3 anisoU;
  int stencil_size = 1;
  bool symmetrize_diffuse = true;
  int laue_group_num = 12;
  mat3 rotate_principal_axes;
  vec3 a0, b0, c0; // cell basis vectors
};
}}

#include "simtbx/kokkos/simulation_kernels.h"

namespace simtbx { namespace Kokkos {

namespace af = scitbx::af;
  // extract subview from [start_index * length; (start_index + 1) * length)
  template <typename T>
  view_1d_t<T> extract_subview(view_1d_t<T> A, int start_index, int length) {
    return ::Kokkos::subview(A, ::Kokkos::pair<int, int>(start_index * length, (start_index + 1) * length ));
  }

  // make a unit vector pointing in same direction and report magnitude (both args can be same vector)
  double cpu_unitize(const double * vector, double * new_unit_vector) {

        double v1 = vector[1];
        double v2 = vector[2];
        double v3 = vector[3];

        double mag = sqrt(v1 * v1 + v2 * v2 + v3 * v3);

        if (mag != 0.0) {
                // normalize it
                new_unit_vector[0] = mag;
                new_unit_vector[1] = v1 / mag;
                new_unit_vector[2] = v2 / mag;
                new_unit_vector[3] = v3 / mag;
        } else {
                // can't normalize, report zero vector
                new_unit_vector[0] = 0.0;
                new_unit_vector[1] = 0.0;
                new_unit_vector[2] = 0.0;
                new_unit_vector[3] = 0.0;
        }
        return mag;
  }

template <typename MemoryPolicy=large_array_policy>
struct exascale_api {
  inline
  exascale_api(const simtbx::nanoBragg::nanoBragg& nB) : SIM(nB) { }

  //void show();
  /*void add_energy_channel_from_gpu_amplitudes(int const&,
    simtbx::Kokkos::kokkos_energy_channels &,
    simtbx::Kokkos::kokkos_detector<MemoryPolicy> &,
    double const&
  ); */
  /*void add_energy_channel_mask_allpanel(int const&,
    simtbx::Kokkos::kokkos_energy_channels &,
    simtbx::Kokkos::kokkos_detector<MemoryPolicy> &,
    af::shared<bool>
  ); */
  /*void add_energy_channel_mask_allpanel(int const&,
    simtbx::Kokkos::kokkos_energy_channels &,
    simtbx::Kokkos::kokkos_detector<MemoryPolicy> &,
    af::shared<std::size_t> const
  ); */
  void add_energy_multichannel_mask_allpanel(af::shared<int> const,
    simtbx::Kokkos::kokkos_energy_channels &,
    simtbx::Kokkos::kokkos_detector<MemoryPolicy> &,
    af::shared<std::size_t> const,
    af::shared<double> const
  );

  //void add_background(simtbx::Kokkos::kokkos_detector<MemoryPolicy> &, int const&);
  //af::flex_double add_noise(simtbx::Kokkos::kokkos_detector<MemoryPolicy> &);
  //void allocate();
  //~exascale_api();

  const simtbx::nanoBragg::nanoBragg& SIM;

  CUDAREAL m_subpixel_size;
  int m_steps;

  const int m_vector_length = 4;
  vector_cudareal_t m_beam_vector = vector_cudareal_t("m_beam_vector", m_vector_length);
  vector_cudareal_t m_spindle_vector = vector_cudareal_t("m_spindle_vector", m_vector_length);
  vector_cudareal_t m_a0 = vector_cudareal_t("m_a0", m_vector_length);
  vector_cudareal_t m_b0 = vector_cudareal_t("m_b0", m_vector_length);
  vector_cudareal_t m_c0 = vector_cudareal_t("m_c0", m_vector_length);
  vector_cudareal_t m_polar_vector = vector_cudareal_t("m_polar_vector", m_vector_length);
  vector_cudareal_t m_source_X = vector_cudareal_t("m_source_X", 0);
  vector_cudareal_t m_source_Y = vector_cudareal_t("m_source_Y", 0);
  vector_cudareal_t m_source_Z = vector_cudareal_t("m_source_Z", 0);
  vector_cudareal_t m_source_I = vector_cudareal_t("m_source_I", 0);
  vector_cudareal_t m_source_lambda = vector_cudareal_t("m_source_lambda", 0);
  vector_cudareal_t m_mosaic_umats = vector_cudareal_t("m_mosaic_umats", 0);
  crystal_orientation_t m_crystal_orientation = crystal_orientation_t("m_crystal_orientation", 0, 0, 3);
  CUDAREAL m_water_size = 0;
  CUDAREAL m_water_F = 0;
  CUDAREAL m_water_MW = 0;
  diffuse_api diffuse;

  inline void
  show(){
    SCITBX_EXAMINE(SIM.roi_xmin);
    SCITBX_EXAMINE(SIM.roi_xmax);
    SCITBX_EXAMINE(SIM.roi_ymin);
    SCITBX_EXAMINE(SIM.roi_ymax);
    SCITBX_EXAMINE(SIM.oversample);
    SCITBX_EXAMINE(SIM.point_pixel);
    SCITBX_EXAMINE(SIM.pixel_size);
    SCITBX_EXAMINE(m_subpixel_size);
    SCITBX_EXAMINE(m_steps);
    SCITBX_EXAMINE(SIM.detector_thickstep);
    SCITBX_EXAMINE(SIM.detector_thicksteps);
    SCITBX_EXAMINE(SIM.detector_thick);
    SCITBX_EXAMINE(SIM.detector_attnlen);
    SCITBX_EXAMINE(SIM.curved_detector);
    SCITBX_EXAMINE(SIM.distance);
    SCITBX_EXAMINE(SIM.close_distance);
    SCITBX_EXAMINE(SIM.dmin);
    SCITBX_EXAMINE(SIM.phi0);
    SCITBX_EXAMINE(SIM.phistep);
    SCITBX_EXAMINE(SIM.phisteps);
    SCITBX_EXAMINE(SIM.sources);
    SCITBX_EXAMINE(SIM.mosaic_spread);
    SCITBX_EXAMINE(SIM.mosaic_domains);
    SCITBX_EXAMINE(SIM.Na);
    SCITBX_EXAMINE(SIM.Nb);
    SCITBX_EXAMINE(SIM.Nc);
    SCITBX_EXAMINE(SIM.fluence);
    SCITBX_EXAMINE(SIM.spot_scale);
    SCITBX_EXAMINE(SIM.integral_form);
    SCITBX_EXAMINE(SIM.default_F);
    SCITBX_EXAMINE(SIM.interpolate);
    SCITBX_EXAMINE(SIM.nopolar);
    SCITBX_EXAMINE(SIM.polarization);
    SCITBX_EXAMINE(SIM.fudge);
  }

  inline void
  add_energy_channel_from_gpu_amplitudes(
    int const& ichannel,
    simtbx::Kokkos::kokkos_energy_channels & kec,
    simtbx::Kokkos::kokkos_detector<MemoryPolicy> & kdt,
    double const& weight
  ){
    // transfer source_I, source_lambda
    // the int arguments are for sizes of the arrays
    int source_count = SIM.sources;
    af::shared<double> weighted_sources_I = af::shared<double>(source_count);
    double* wptr = weighted_sources_I.begin();
    for (std::size_t iwt = 0; iwt < source_count; iwt++){wptr[iwt] = weight*(SIM.source_I[iwt]);}
    kokkostbx::transfer_double2kokkos(m_source_I, wptr, source_count);
    kokkostbx::transfer_double2kokkos(m_source_lambda, SIM.source_lambda, source_count);

    ::Kokkos::resize(m_crystal_orientation, SIM.phisteps, SIM.mosaic_domains, 3);
    calc_CrystalOrientations(
      SIM.phi0, SIM.phistep, SIM.phisteps, m_spindle_vector, m_a0, m_b0, m_c0, SIM.mosaic_spread,
      SIM.mosaic_domains, m_mosaic_umats, m_crystal_orientation);

    // magic happens here(?): take pointer from singleton, temporarily use it for add Bragg iteration:
    vector_cudareal_t current_channel_Fhkl = kec.d_channel_Fhkl[ichannel];

    std::size_t panel_size = kdt.m_slow_dim_size * kdt.m_fast_dim_size;

    // the for loop around panels.  Offsets given.
    for (std::size_t panel_id = 0; panel_id < kdt.m_panel_count; panel_id++){
      // loop thru panels and increment the array ptrs
      kokkosSpotsKernel(
      kdt.m_slow_dim_size, kdt.m_fast_dim_size, SIM.roi_xmin, SIM.roi_xmax,
      SIM.roi_ymin, SIM.roi_ymax, SIM.oversample, SIM.point_pixel,
      SIM.pixel_size, m_subpixel_size, m_steps, SIM.detector_thickstep,
      SIM.detector_thicksteps, SIM.detector_thick, SIM.detector_attnlen,
      extract_subview(kdt.m_sdet_vector, panel_id, 1),
      extract_subview(kdt.m_fdet_vector, panel_id, 1),
      extract_subview(kdt.m_odet_vector, panel_id, 1),
      extract_subview(kdt.m_pix0_vector, panel_id, 1),
      SIM.curved_detector, kdt.metrology.dists[panel_id], kdt.metrology.dists[panel_id],
      m_beam_vector,
      SIM.dmin, SIM.phisteps, SIM.sources,
      m_source_X, m_source_Y, m_source_Z,
      m_source_I, m_source_lambda,
      SIM.xtal_shape,
      SIM.mosaic_domains, m_crystal_orientation,
      SIM.Na, SIM.Nb, SIM.Nc, SIM.V_cell,
      m_water_size, m_water_F, m_water_MW, simtbx::nanoBragg::r_e_sqr,
      SIM.fluence, simtbx::nanoBragg::Avogadro, SIM.spot_scale, SIM.integral_form,
      SIM.default_F,
      current_channel_Fhkl,
      kec.m_FhklParams,
      SIM.nopolar,
      m_polar_vector, SIM.polarization, SIM.fudge,
      // &(kdt.m_maskimage[panel_size * panel_id]),
      nullptr,
      // return arrays:
      extract_subview(kdt.m_floatimage, panel_id, panel_size),
      extract_subview(kdt.m_omega_reduction, panel_id, panel_size),
      extract_subview(kdt.m_max_I_x_reduction, panel_id, panel_size),
      extract_subview(kdt.m_max_I_y_reduction, panel_id, panel_size),
      extract_subview(kdt.m_rangemap, panel_id, panel_size));
      //cudaSafeCall(cudaPeekAtLastError());
      ::Kokkos::fence();
    }
    //cudaSafeCall(cudaDeviceSynchronize());

    //don't want to free the kec data when the nanoBragg goes out of scope, so switch the pointer
    // cu_current_channel_Fhkl = NULL;

    add_array(kdt.m_accumulate_floatimage, kdt.m_floatimage);
  }

  inline void
  add_energy_channel_mask_allpanel(
    int const& ichannel,
    simtbx::Kokkos::kokkos_energy_channels & kec,
    simtbx::Kokkos::kokkos_detector<MemoryPolicy> & kdt,
    af::shared<bool> all_panel_mask
  ){
    // here or there, need to convert the all_panel_mask (3D map) into a 1D list of accepted pixels
    // coordinates for the active pixel list are absolute offsets into the detector array
    af::shared<std::size_t> active_pixel_list;
    const bool* jptr = all_panel_mask.begin();
    for (int j=0; j < all_panel_mask.size(); ++j){
      if (jptr[j]) {
        active_pixel_list.push_back(j);
      }
    }
    add_energy_channel_mask_allpanel(ichannel, kec, kdt, active_pixel_list);
  }

  inline void
  add_energy_channel_mask_allpanel(
    int const& ichannel,
    simtbx::Kokkos::kokkos_energy_channels & kec,
    simtbx::Kokkos::kokkos_detector<MemoryPolicy> & kdt,
    af::shared<std::size_t> const active_pixel_list
  ){
    //SCITBX_CHECK_POINT; //tested by tst_shoeboxes.py
    kdt.set_active_pixels_on_GPU(active_pixel_list);

    // transfer source_I, source_lambda
    // the int arguments are for sizes of the arrays
    int source_count = SIM.sources;
    kokkostbx::transfer_double2kokkos(m_source_I, SIM.source_I, source_count);
    kokkostbx::transfer_double2kokkos(m_source_lambda, SIM.source_lambda, source_count);
    // cudaSafeCall(cudaMemcpyVectorDoubleToDevice(m_source_I, SIM.source_I, SIM.sources));
    // cudaSafeCall(cudaMemcpyVectorDoubleToDevice(m_source_lambda, SIM.source_lambda, SIM.sources));

    ::Kokkos::resize(m_crystal_orientation, SIM.phisteps, SIM.mosaic_domains, 3);
    calc_CrystalOrientations(
      SIM.phi0, SIM.phistep, SIM.phisteps, m_spindle_vector, m_a0, m_b0, m_c0, SIM.mosaic_spread,
      SIM.mosaic_domains, m_mosaic_umats, m_crystal_orientation);

    // magic happens here: take pointer from singleton, temporarily use it for add Bragg iteration:
    vector_cudareal_t current_channel_Fhkl = kec.d_channel_Fhkl[ichannel];

    // cudaDeviceProp deviceProps = { 0 };
    // cudaSafeCall(cudaGetDeviceProperties(&deviceProps, SIM.device_Id));
    // int smCount = deviceProps.multiProcessorCount;
    // dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    // dim3 numBlocks(smCount * 8, 1);

    // for call for all panels at the same time

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
      SIM.dmin, SIM.phisteps, SIM.sources,
      m_source_X, m_source_Y,
      m_source_Z,
      m_source_I, m_source_lambda,
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

    // cudaSafeCall(cudaPeekAtLastError());
    // cudaSafeCall(cudaDeviceSynchronize());

    //don't want to free the kec data when the nanoBragg goes out of scope, so switch the pointer
    // cu_current_channel_Fhkl = NULL;

    add_array(kdt.m_accumulate_floatimage, kdt.m_floatimage);
  }

  inline void
  add_background(simtbx::Kokkos::kokkos_detector<MemoryPolicy> & kdt, int const& override_source) {
        // cudaSafeCall(cudaSetDevice(SIM.device_Id));

        // transfer source_I, source_lambda
        // the int arguments are for sizes of the arrays
        int sources_count = SIM.sources;
        kokkostbx::transfer_double2kokkos(m_source_I, SIM.source_I, sources_count);
        kokkostbx::transfer_double2kokkos(m_source_lambda, SIM.source_lambda, sources_count);
        // cudaSafeCall(cudaMemcpyVectorDoubleToDevice(m_source_I, SIM.source_I, SIM.sources));
        // cudaSafeCall(cudaMemcpyVectorDoubleToDevice(m_source_lambda, SIM.source_lambda, SIM.sources));

        vector_cudareal_t stol_of("stol_of", SIM.stols);
        transfer_X2kokkos(stol_of, SIM.stol_of, SIM.stols);
        // CUDAREAL * cu_stol_of;
        // cudaSafeCall(cudaMalloc((void ** )&cu_stol_of, sizeof(*cu_stol_of) * SIM.stols));
        // cudaSafeCall(cudaMemcpyVectorDoubleToDevice(cu_stol_of, SIM.stol_of, SIM.stols));

        vector_cudareal_t Fbg_of("Fbg_of", SIM.stols);
        transfer_X2kokkos(Fbg_of, SIM.Fbg_of, SIM.stols);
        // CUDAREAL * cu_Fbg_of;
        // cudaSafeCall(cudaMalloc((void ** )&cu_Fbg_of, sizeof(*cu_Fbg_of) * SIM.stols));
        // cudaSafeCall(cudaMemcpyVectorDoubleToDevice(cu_Fbg_of, SIM.Fbg_of, SIM.stols));

        // cudaDeviceProp deviceProps = { 0 };
        // cudaSafeCall(cudaGetDeviceProperties(&deviceProps, SIM.device_Id));
        // int smCount = deviceProps.multiProcessorCount;
        // dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
        // dim3 numBlocks(smCount * 8, 1);

        //  initialize the device memory within a kernel.
        //  modify the arguments to initialize multipanel detector.
        ::Kokkos::parallel_for("kokkosSpotsInit", kdt.m_panel_count * kdt.m_slow_dim_size * kdt.m_fast_dim_size, KOKKOS_LAMBDA (const int& j) {
          kdt.m_floatimage(j) = 0;
          kdt.m_omega_reduction(j) = 0;
          kdt.m_max_I_x_reduction(j) = 0;
          kdt.m_max_I_y_reduction(j) = 0;
          kdt.m_rangemap(j) = false;
        });
        // nanoBraggSpotsInitCUDAKernel<<<numBlocks, threadsPerBlock>>>(
        //   kdt.m_panel_count * kdt.m_slow_dim_size, kdt.m_fast_dim_size,
        //   kdt.cu_floatimage, kdt.cu_omega_reduction,
        //   kdt.cu_max_I_x_reduction, kdt.cu_max_I_y_reduction,
        //   kdt.cu_rangemap);
        // cudaSafeCall(cudaPeekAtLastError());
        // cudaSafeCall(cudaDeviceSynchronize());

        std::size_t panel_size = kdt.m_slow_dim_size * kdt.m_fast_dim_size;

        // the for loop around panels.  Offsets given.
        for (std::size_t panel_id = 0; panel_id < kdt.m_panel_count; panel_id++) {
          add_background_kokkos_kernel(SIM.sources,
          SIM.oversample, override_source,
          SIM.pixel_size, kdt.m_slow_dim_size, kdt.m_fast_dim_size, SIM.detector_thicksteps,
          SIM.detector_thickstep, SIM.detector_attnlen,
          extract_subview(kdt.m_sdet_vector, panel_id, 1),
          extract_subview(kdt.m_fdet_vector, panel_id, 1),
          extract_subview(kdt.m_odet_vector, panel_id, 1),
          extract_subview(kdt.m_pix0_vector, panel_id, 1),
          kdt.metrology.dists[panel_id], SIM.point_pixel, SIM.detector_thick,
          m_source_X, m_source_Y, m_source_Z,
          m_source_lambda, m_source_I,
          SIM.stols, stol_of, Fbg_of,
          SIM.nopolar, SIM.polarization, m_polar_vector,
          simtbx::nanoBragg::r_e_sqr, SIM.fluence, SIM.amorphous_molecules,
          // returns:
          extract_subview(kdt.m_floatimage, panel_id, panel_size));

          // cudaSafeCall(cudaPeekAtLastError());
        }
        // cudaSafeCall(cudaDeviceSynchronize());
        ::Kokkos::fence();
        add_array(kdt.m_accumulate_floatimage, kdt.m_floatimage);

        // cudaSafeCall(cudaFree(cu_stol_of));
        // cudaSafeCall(cudaFree(cu_Fbg_of));
  }

  inline af::flex_double
  add_noise(simtbx::Kokkos::kokkos_detector<MemoryPolicy> & kdt) {
        // put raw pixels into a temporary hold
        af::flex_double tmp_hold_pixels = kdt.get_raw_pixels();
        std::size_t panel_size = kdt.m_slow_dim_size * kdt.m_fast_dim_size;
        af::flex_double panel_pixels = af::flex_double(af::flex_grid<>(kdt.m_slow_dim_size,kdt.m_fast_dim_size), af::init_functor_null<double>());
        for (std::size_t panel_id = 0; panel_id < kdt.m_panel_count; panel_id++) {
          double* dst_beg = panel_pixels.begin();
          double* dst_end = panel_pixels.end();
          double* dstptr = dst_beg;
          for (double *srcptr = &(tmp_hold_pixels[panel_id*panel_size]); dstptr != dst_end; ){
            *dstptr++ = *srcptr++;
          } // get the pixels from one panel on GPU memory into panel_pixels for add_noise
          SIM.add_noise(panel_pixels);
          dstptr = dst_beg;
          for (double *srcptr = &(tmp_hold_pixels[panel_id*panel_size]); dstptr != dst_end; ){
            *srcptr++ = *dstptr++;
          } // return the pixels from SIM.raw_pixels to temporary array
        }
        ::Kokkos::fence();
        return tmp_hold_pixels;
  }

  inline void
  allocate() {
    //cudaSafeCall(cudaSetDevice(SIM.device_Id));

    // water_size not defined in class, CLI argument, defaults to 0
    double water_size = 0.0;
    // missing constants
    double water_F = 2.57;
    double water_MW = 18.0;

    // make sure we are normalizing with the right number of sub-steps
    int nb_steps = SIM.phisteps * SIM.mosaic_domains * SIM.oversample * SIM.oversample;
    double nb_subpixel_size = SIM.pixel_size / SIM.oversample;

    //create transfer arguments to device space
    m_subpixel_size = nb_subpixel_size; //check for conflict?
    m_steps = nb_steps; //check for conflict?

    // presumably thickness and attenuation can be migrated to the gpu detector class XXX FIXME
    //cu_detector_thick = SIM.detector_thick;
    //cu_detector_mu = SIM.detector_attnlen; // synonyms
    //cu_distance = SIM.distance; // distance and close distance, detector properties? XXX FIXME
    //cu_close_distance = SIM.close_distance;

    m_water_size = water_size;
    m_water_F = water_F;
    m_water_MW = water_MW;

    //const int vector_length = 4;

    kokkostbx::transfer_double2kokkos(m_beam_vector, SIM.beam_vector, m_vector_length);
    kokkostbx::transfer_double2kokkos(m_spindle_vector, SIM.spindle_vector, m_vector_length);
    kokkostbx::transfer_double2kokkos(m_a0, SIM.a0, m_vector_length);
    kokkostbx::transfer_double2kokkos(m_b0, SIM.b0, m_vector_length);
    kokkostbx::transfer_double2kokkos(m_c0, SIM.c0, m_vector_length);

    // cudaSafeCall(cudaMalloc((void ** )&cu_beam_vector, sizeof(*cu_beam_vector) * vector_length));
    // cudaSafeCall(cudaMemcpyVectorDoubleToDevice(cu_beam_vector, SIM.beam_vector, vector_length));

    // cudaSafeCall(cudaMalloc((void ** )&cu_spindle_vector, sizeof(*cu_spindle_vector) * vector_length));
    // cudaSafeCall(cudaMemcpyVectorDoubleToDevice(cu_spindle_vector, SIM.spindle_vector, vector_length));

    // cudaSafeCall(cudaMalloc((void ** )&cu_a0, sizeof(*cu_a0) * vector_length));
    // cudaSafeCall(cudaMemcpyVectorDoubleToDevice(cu_a0, SIM.a0, vector_length));

    // cudaSafeCall(cudaMalloc((void ** )&cu_b0, sizeof(*cu_b0) * vector_length));
    // cudaSafeCall(cudaMemcpyVectorDoubleToDevice(cu_b0, SIM.b0, vector_length));

    // cudaSafeCall(cudaMalloc((void ** )&cu_c0, sizeof(*cu_c0) * vector_length));
    // cudaSafeCall(cudaMemcpyVectorDoubleToDevice(cu_c0, SIM.c0, vector_length));

    // Unitize polar vector before sending it to the GPU.
    // Optimization do it only once here rather than multiple time per pixel in the GPU.
    double polar_vector_unitized[4];
    cpu_unitize(SIM.polar_vector, polar_vector_unitized);
    kokkostbx::transfer_double2kokkos(m_polar_vector, polar_vector_unitized, m_vector_length);

    int sources_count = SIM.sources;
    kokkostbx::transfer_double2kokkos(m_source_X, SIM.source_X, sources_count);
    kokkostbx::transfer_double2kokkos(m_source_Y, SIM.source_Y, sources_count);
    kokkostbx::transfer_double2kokkos(m_source_Z, SIM.source_Z, sources_count);
    kokkostbx::transfer_double2kokkos(m_source_I, SIM.source_I, sources_count);
    kokkostbx::transfer_double2kokkos(m_source_lambda, SIM.source_lambda, sources_count);

    int mosaic_domains_count = SIM.mosaic_domains;
    kokkostbx::transfer_double2kokkos(m_mosaic_umats, SIM.mosaic_umats, mosaic_domains_count * 9);

    // cudaSafeCall(cudaMalloc((void ** )&cu_polar_vector, sizeof(*cu_polar_vector) * vector_length));
    // cudaSafeCall(cudaMemcpyVectorDoubleToDevice(cu_polar_vector, polar_vector_unitized, vector_length));

    // cudaSafeCall(cudaMalloc((void ** )&cu_source_X, sizeof(*cu_source_X) * sources_count));
    // cudaSafeCall(cudaMemcpyVectorDoubleToDevice(cu_source_X, SIM.source_X, sources_count));

    // cudaSafeCall(cudaMalloc((void ** )&cu_source_Y, sizeof(*cu_source_Y) * sources_count));
    // cudaSafeCall(cudaMemcpyVectorDoubleToDevice(cu_source_Y, SIM.source_Y, sources_count));

    // cudaSafeCall(cudaMalloc((void ** )&cu_source_Z, sizeof(*cu_source_Z) * sources_count));
    // cudaSafeCall(cudaMemcpyVectorDoubleToDevice(cu_source_Z, SIM.source_Z, sources_count));

    // cudaSafeCall(cudaMalloc((void ** )&cu_source_I, sizeof(*cu_source_I) * sources_count));
    // cudaSafeCall(cudaMemcpyVectorDoubleToDevice(cu_source_I, SIM.source_I, sources_count));

    // cudaSafeCall(cudaMalloc((void ** )&cu_source_lambda, sizeof(*cu_source_lambda) * sources_count));
    // cudaSafeCall(cudaMemcpyVectorDoubleToDevice(cu_source_lambda, SIM.source_lambda, sources_count));

    // cudaSafeCall(cudaMalloc((void ** )&cu_mosaic_umats, sizeof(*cu_mosaic_umats) * mosaic_domains_count * 9));
    // cudaSafeCall(cudaMemcpyVectorDoubleToDevice(cu_mosaic_umats, SIM.mosaic_umats, mosaic_domains_count * 9));
  };

};
} // Kokkos
} // simtbx
#endif // SIMTBX_Kokkos_STRUCTURE_FACTORS_H
