#ifndef SIMTBX_KOKKOS_DETECTOR_H
#define SIMTBX_KOKKOS_DETECTOR_H

// class is intended to capture several concepts:
//   1) a dxtbx multipanel detector
//   2) the associated experimental data
//   3) the associated simulated data, in process of being accumulated by kernel calls
//   4) mask data
//   5) possibly other metadata
#include <iostream>

#include "scitbx/array_family/shared.h"
#include "scitbx/array_family/flex_types.h"
#include "dxtbx/model/detector.h"
#include "dxtbx/model/beam.h"
#include "simtbx/nanoBragg/nanoBragg.h"
#include "kokkostbx/kokkos_types.h"
#include "kokkostbx/kokkos_vector3.h"
#include "kokkostbx/kokkos_matrix3.h"
#include "kokkostbx/kokkos_utils.h"

using vec3 = kokkostbx::vector3<CUDAREAL>;
using mat3 = kokkostbx::matrix3<CUDAREAL>;
using Kokkos::fence;


namespace simtbx { namespace Kokkos {

namespace af = scitbx::af;

struct packed_metrology{
  packed_metrology(){/*printf("NO OPERATION")*/;};
  packed_metrology(dxtbx::model::Detector const &,dxtbx::model::Beam const &);
  packed_metrology(const simtbx::nanoBragg::nanoBragg& nB);
  void show() const;
  af::shared<vec3>sdet;
  af::shared<vec3>fdet;
  af::shared<vec3>odet;
  af::shared<vec3>pix0;
  af::shared<double>dists;
  af::shared<double>Xbeam;
  af::shared<double>Ybeam;
};

struct large_array_policy {};
struct small_whitelist_policy {};

template <typename MemoryPolicy>
struct kokkos_detector{
  inline kokkos_detector(){printf("NO OPERATION, DEVICE NUMBER IS NEEDED");};
  //kokkos_detector(int const&, const simtbx::nanoBragg::nanoBragg& nB);
  //kokkos_detector(int const&, dxtbx::model::Detector const &, dxtbx::model::Beam const &);
  //vector_double_t construct_detail(dxtbx::model::Detector const &);

  inline void show_summary(){
    std::cout << "Detector size: " << m_panel_count << " panel" << ( (m_panel_count>1)? "s" : "" ) << std::endl;
    metrology.show();
  }
  void each_image_allocate(const std::size_t&);
  //void scale_in_place(const double&);
  //void write_raw_pixels(simtbx::nanoBragg::nanoBragg&);
  //af::flex_double get_raw_pixels();
  //void set_active_pixels_on_GPU(af::shared<std::size_t>);
  //af::shared<double> get_whitelist_raw_pixels(af::shared<std::size_t>);
  inline void each_image_free(){} //no op in Kokkos
  int h_deviceID;

  //const dxtbx::model::Detector detector;
  packed_metrology const metrology;

  // detector size and dimensions in pixels
  int m_panel_count = -1;
  int m_slow_dim_size = -1;
  int m_fast_dim_size = -1;
  int m_total_pixel_count = -1;

  // kokkos arrays
  vector_double_t m_accumulate_floatimage = vector_double_t("m_accumulate_floatimage", 0);

  // per image input variables, pointers to GPU memory

  // nanoBragg only manages single-panel, multipanel case must mask elsewhere
  vector_ushort_t m_maskimage = vector_ushort_t("m_maskimage", 0);

  // per kernel-call output variables, pointers to GPU memory
  vector_float_t m_omega_reduction = vector_float_t("m_omega_reduction", 0);
  vector_float_t m_max_I_x_reduction = vector_float_t("m_max_I_x_reduction", 0);
  vector_float_t m_max_I_y_reduction = vector_float_t("m_max_I_y_reduction", 0);

  vector_bool_t m_rangemap = vector_bool_t("m_rangemap", 0);

  //most important output, stores the simulated pixel data
  vector_float_t m_floatimage = vector_float_t("m_floatimage", 0);

  // all-panel packed GPU representation of the multi-panel metrology
  view_1d_t<vec3> m_sdet_vector = view_1d_t<vec3>("m_sdet_vector", 0);
  view_1d_t<vec3> m_fdet_vector = view_1d_t<vec3>("m_fdet_vector", 0);
  view_1d_t<vec3> m_odet_vector = view_1d_t<vec3>("m_odet_vector", 0);
  view_1d_t<vec3> m_pix0_vector = view_1d_t<vec3>("m_pix0_vector", 0);
  vector_cudareal_t m_distance = vector_cudareal_t("m_distance", 0);
  vector_cudareal_t m_Xbeam = vector_cudareal_t("m_Xbeam", 0);
  vector_cudareal_t m_Ybeam = vector_cudareal_t("m_Ybeam", 0);

  // all-panel whitelist of active pixels on GPU
  af::shared<std::size_t> active_pixel_list;
  std::size_t m_active_pixel_size;
  vector_size_t m_active_pixel_list = vector_size_t("m_active_pixel_list", 0);

  inline
  kokkos_detector(int const& arg_device,
                             const simtbx::nanoBragg::nanoBragg& nB):
    h_deviceID(arg_device),
    metrology(nB),
    m_panel_count(1),
    m_slow_dim_size(nB.spixels),
    m_fast_dim_size(nB.fpixels),
    m_total_pixel_count( m_panel_count * m_slow_dim_size * m_fast_dim_size ),
    m_accumulate_floatimage( vector_double_t( "m_accumulate_floatimage", m_total_pixel_count) ) { }

  inline
  kokkos_detector(int const& arg_device,
                             dxtbx::model::Detector const & arg_detector,
                             dxtbx::model::Beam const& arg_beam):
    h_deviceID(arg_device),
    metrology(arg_detector, arg_beam),
    m_panel_count( arg_detector.size() ),
    m_slow_dim_size( arg_detector[0].get_image_size()[1] ),
    m_fast_dim_size( arg_detector[0].get_image_size()[0] ),
    m_total_pixel_count( m_panel_count * m_slow_dim_size * m_fast_dim_size ),
    m_accumulate_floatimage( construct_detail(arg_detector) ) { }
    // Easy mistake: not realizing that the dxtbx detector model stores (fast,slow) sizes

  inline
  vector_double_t
  construct_detail(dxtbx::model::Detector const & arg_detector) {
    //1) confirm the size
    SCITBX_ASSERT( m_panel_count == arg_detector.size() );
    SCITBX_ASSERT( m_panel_count >= 1 );

    //2) confirm that array dimensions are similar for each size
    for (int ipanel=1; ipanel < arg_detector.size(); ++ipanel){
      SCITBX_ASSERT( arg_detector[ipanel].get_image_size()[1] == m_slow_dim_size );
      SCITBX_ASSERT( arg_detector[ipanel].get_image_size()[0] == m_fast_dim_size );
    }
    // printf(" m_total_pixel_count: %d\n", m_total_pixel_count);
    // printf("     m_slow_dim_size: %d\n", m_slow_dim_size);
    // printf("     m_fast_dim_size: %d\n", m_fast_dim_size);
    // printf("       m_panel_count: %d\n", m_panel_count);

    //3) allocate a cuda array with these dimensions
    // separate accumulator image outside the usual nanoBragg data structure.
    //       1. accumulate contributions from a sequence of source energy channels computed separately
    //       2. represent multiple panels, all same rectangular shape; slowest dimension = n_panels
    vector_double_t view_floatimage( "m_accumulate_floatimage", m_total_pixel_count );
    return view_floatimage;
  };

  inline void
  scale_in_place(const double& factor){
    auto local_accumulate_floatimage = m_accumulate_floatimage;
    parallel_for("scale_in_place", range_policy(0,m_total_pixel_count), KOKKOS_LAMBDA (const int i) {
      local_accumulate_floatimage( i ) = local_accumulate_floatimage( i ) * factor;
    });
  }

  inline void
  write_raw_pixels(simtbx::nanoBragg::nanoBragg& nB) {
    //only implement the monolithic detector case, one panel
    SCITBX_ASSERT(nB.spixels == m_slow_dim_size);
    SCITBX_ASSERT(nB.fpixels == m_fast_dim_size);
    SCITBX_ASSERT(m_panel_count == 1);
    // nB.raw_pixels = af::flex_double(af::flex_grid<>(nB.spixels,nB.fpixels));
    // do not reallocate CPU memory for the data write, as it is not needed

    kokkostbx::transfer_kokkos2flex(nB.raw_pixels, m_accumulate_floatimage);
    // vector_double_t::HostMirror host_floatimage = create_mirror_view(m_accumulate_floatimage);
    // deep_copy(host_floatimage, m_accumulate_floatimage);

    // printf(" m_total_pixel_count: %d\n", m_total_pixel_count);

    // double * double_floatimage = nB.raw_pixels.begin();
    // for (int i=0; i<m_total_pixel_count; ++i) {
    //   double_floatimage[i] = host_floatimage( i );
    // }
  }

  inline af::flex_double
  get_raw_pixels(){
    //return the data array for the multipanel detector case
    af::flex_double output_array(af::flex_grid<>(m_panel_count,m_slow_dim_size,m_fast_dim_size), af::init_functor_null<double>());
    kokkostbx::transfer_kokkos2flex(output_array, m_accumulate_floatimage);

    // vector_double_t::HostMirror host_floatimage = create_mirror_view(m_accumulate_floatimage);
    // deep_copy(host_floatimage, m_accumulate_floatimage);

    // for (int i=0; i<m_total_pixel_count; ++i) {
    //   output_array_ptr[ i ] = host_floatimage( i );
    // }
    return output_array;
  }

  inline void
  set_active_pixels_on_GPU(af::shared<std::size_t> active_pixel_list_value) {
    m_active_pixel_size = active_pixel_list_value.size();
    kokkostbx::transfer_shared2kokkos(m_active_pixel_list, active_pixel_list_value);
    active_pixel_list = active_pixel_list_value;
  }

  inline af::shared<double>
  get_whitelist_raw_pixels(af::shared<std::size_t> selection) {
    //printf("algorithm: %20s selection size %10d\n",hello().c_str(), selection.size());
    //return the data array for the multipanel detector case, but only for whitelist pixels
    vector_size_t active_pixel_selection = vector_size_t("active_pixel_selection", selection.size());
    kokkostbx::transfer_shared2kokkos(active_pixel_selection, selection);

    size_t output_pixel_size = selection.size();
    vector_cudareal_t active_pixel_results = vector_cudareal_t("active_pixel_results", output_pixel_size);

    auto temp = m_accumulate_floatimage;

    parallel_for("get_active_pixel_selection",
                  range_policy(0, output_pixel_size),
                  KOKKOS_LAMBDA (const int i) {
      size_t index = active_pixel_selection( i );
      active_pixel_results( i ) = temp( index );
    });

    af::shared<double> output_array(output_pixel_size, af::init_functor_null<double>());
    kokkostbx::transfer_kokkos2shared(output_array, active_pixel_results);

    SCITBX_ASSERT(output_array.size() == output_pixel_size);
    return output_array;
  }

  std::string hello();
};
} // Kokkos
} // simtbx
#endif // SIMTBX_KOKKOS_DETECTOR_H
