#include <boost/python/module.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>
#include <boost/python/tuple.hpp>

#include "simtbx/kokkos/kokkos_instance.h"
#include "simtbx/kokkos/detector.h"
#include "simtbx/kokkos/structure_factors.h"
#include "simtbx/kokkos/simulation.h"

namespace simtbx { namespace Kokkos {

  namespace {
    static void set_anisoG(simtbx::Kokkos::diffuse_api& diffuse, boost::python::list const& values) {
      double g0 = boost::python::extract<double>(values[0]);
      double g1 = boost::python::extract<double>(values[1]);
      double g2 = boost::python::extract<double>(values[2]);
      diffuse.anisoG << g0,0,0,0,g1,0,0,0,g2;
    }

    static boost::python::tuple get_anisoG(simtbx::Kokkos::diffuse_api const& diffuse) {
      return boost::python::make_tuple(diffuse.anisoG[0],diffuse.anisoG[4],diffuse.anisoG[8]);
    }

    static void set_anisoU(simtbx::Kokkos::diffuse_api& diffuse, boost::python::list const& values) {
      double g0 = boost::python::extract<double>(values[0]);
      double g1 = boost::python::extract<double>(values[1]);
      double g2 = boost::python::extract<double>(values[2]);
      diffuse.anisoU << g0,0,0,0,g1,0,0,0,g2;
    }

    static boost::python::tuple get_anisoU(simtbx::Kokkos::diffuse_api const& diffuse) {
      return boost::python::make_tuple(diffuse.anisoU[0],diffuse.anisoU[4],diffuse.anisoU[8]);
    }

    static void set_rotate_principal_axes(simtbx::Kokkos::diffuse_api& diffuse, std::string const& value) {
      if (value==std::string("a,b,c")){
        diffuse.rotate_principal_axes << 1.,0.,0.,0.,1.,0.,0.,0.,1.;
      } else if (value==std::string("a-b,a+b,c")){
        double sqrt2o2 = std::sqrt(2.)/2.;
        diffuse.rotate_principal_axes << sqrt2o2,-sqrt2o2,0.,sqrt2o2,sqrt2o2,0.,0.,0.,1.;
      } else {
        throw std::string("rotation case not implemented");
      }
    }

    static boost::python::tuple get_rotate_principal_axes(simtbx::Kokkos::diffuse_api const& diffuse) {
      return boost::python::make_tuple(diffuse.rotate_principal_axes[0],diffuse.rotate_principal_axes[1],diffuse.rotate_principal_axes[2],
      diffuse.rotate_principal_axes[3],diffuse.rotate_principal_axes[4],diffuse.rotate_principal_axes[5],
      diffuse.rotate_principal_axes[6],diffuse.rotate_principal_axes[7],diffuse.rotate_principal_axes[8]);
    }
  }

  struct kokkos_instance_wrapper
  {
    static void
    wrap()
    {
      using namespace boost::python;
      class_<simtbx::Kokkos::kokkos_instance>("gpu_instance",init<>() )
        .def(init< const int& >(( arg("deviceId"))))
        .def("get_deviceID", &simtbx::Kokkos::kokkos_instance::get_deviceID)
        .def("finalize_kokkos", &simtbx::Kokkos::kokkos_instance::finalize_kokkos)
        ;
    }
  };

  template <typename memory_t>
  struct detector_wrapper
  {
    static void
    wrap(std::string pyname)
    {
      using namespace boost::python;
      class_<simtbx::Kokkos::kokkos_detector<memory_t> >(pyname.c_str(),init<>() )
        .def(init<int const&, const simtbx::nanoBragg::nanoBragg&>(
            ( arg("deviceId")=-1,arg("nanoBragg")),
             "Single panel constructor with data taken from nanoBragg instance\n"
             "deviceId is completely optional and ignored for Kokkos, rather\n"
             "the device is set by the gpu_instance class."))
        .def(init<int const&, dxtbx::model::Detector const &, dxtbx::model::Beam const &>(
            ( arg("deviceId")=-1,arg("detector"),arg("beam")),
             "Multipanel constructor with data taken from dxtbx objects\n"
             "deviceId is completely optional and ignored for Kokkos, rather\n"
             "the device is set by the gpu_instance class."))
        .def("show_summary",&simtbx::Kokkos::kokkos_detector<memory_t>::show_summary)
        .def("each_image_allocate",
              &simtbx::Kokkos::kokkos_detector<memory_t>::each_image_allocate,
              ( arg_("n_pixels")=0 ),
             "Allocate large pixel arrays")
        .def("scale_in_place", &simtbx::Kokkos::kokkos_detector<memory_t>::scale_in_place,
             "Multiply by a scale factor on the GPU")
        .def("write_raw_pixels",&simtbx::Kokkos::kokkos_detector<memory_t>::write_raw_pixels,
             "Update raw_pixels on host with array from GPU")
        .def("get_raw_pixels",&simtbx::Kokkos::kokkos_detector<memory_t>::get_raw_pixels,
             "return multipanel detector raw pixels as a flex array")
        .def("get_whitelist_raw_pixels",
             &simtbx::Kokkos::kokkos_detector<memory_t>::get_whitelist_raw_pixels,
            "return only those raw pixels requested by the whitelist selection, as a 1D flex array")
        .def("each_image_free", &simtbx::Kokkos::kokkos_detector<memory_t>::each_image_free)
        ;
    }
  };

  struct structure_factor_wrapper {
    static void
    wrap() {
      using namespace boost::python;
      class_<simtbx::Kokkos::kokkos_energy_channels>("gpu_energy_channels",init<>() )
        .def(init< const int& >(( arg("deviceId")=-1),
             "deviceId is optional and ignored for Kokkos, rather\n"
             "the device is set by the gpu_instance class."))
        .def("get_deviceID", &simtbx::Kokkos::kokkos_energy_channels::get_deviceID)
        .def("get_nchannels", &simtbx::Kokkos::kokkos_energy_channels::get_nchannels)
        .def("structure_factors_to_GPU_direct",
             &simtbx::Kokkos::kokkos_energy_channels::structure_factors_to_GPU_direct,
             (arg_("dummy_int"), arg_("indices"), arg_("amplitudes"))
            )
        .def("structure_factors_replace_GPU_direct",
             &simtbx::Kokkos::kokkos_energy_channels::structure_factors_replace_GPU_direct,
             (arg_("ichannel"), arg_("indices"), arg_("amplitudes"))
            )
        .def("print_Fhkl", &simtbx::Kokkos::kokkos_energy_channels::print_Fhkl,
             (arg_("channel"), arg_("first_element"), arg_("last_element"))
            )
        ;
    }
  };

  template <typename memory_t>
  struct simulation_wrapper
  {
    static void
    wrap(std::string pyname)
    {
      using namespace boost::python;
      typedef return_value_policy<return_by_value> rbv;
      typedef default_call_policies dcp;
      using namespace simtbx::Kokkos;
      class_<simtbx::Kokkos::exascale_api<memory_t> >(pyname.c_str(),no_init )
        .def(init<const simtbx::nanoBragg::nanoBragg&>(
            ( arg("nanoBragg"))))
        .def("allocate",&simtbx::Kokkos::exascale_api<memory_t>::allocate,
             "Allocate and transfer input data on the GPU")
        .def("add_energy_channel_from_gpu_amplitudes",
             &simtbx::Kokkos::exascale_api<memory_t>::add_energy_channel_from_gpu_amplitudes,
             (arg_("channel_number"), arg_("gpu_amplitudes"), arg_("gpu_detector"), arg_("weight")=1.0),
             "Point to Fhkl at a new energy channel on the GPU, and accumulate Bragg spot contributions to the detector's accumulator array")
        .def("add_energy_channel_mask_allpanel",
             static_cast<void (exascale_api<memory_t>::*)(int const&,kokkos_energy_channels&,kokkos_detector<memory_t>&, af::shared<bool>) >
             (&exascale_api<memory_t>::add_energy_channel_mask_allpanel),
             (arg_("channel_number"), arg_("gpu_amplitudes"), arg_("gpu_detector"), arg_("pixel_active_mask_bools")),
             "Point to Fhkl at a new energy channel on the GPU, and accumulate Bragg spots on mask==True pixels\n"
             "The pixel_active_mask_bools is a large array with one bool per detector pixel")
        .def("add_energy_channel_mask_allpanel",
             static_cast<void (exascale_api<memory_t>::*)(int const&,kokkos_energy_channels&,kokkos_detector<memory_t>&, af::shared<std::size_t> const) >
             (&exascale_api<memory_t>::add_energy_channel_mask_allpanel),
             (arg_("channel_number"), arg_("gpu_amplitudes"), arg_("gpu_detector"), arg_("pixel_active_list_ints")),
             "Point to Fhkl at a new energy channel on the GPU, and accumulate Bragg spots on mask==True pixels\n"
             "The pixel_active_list_ints is a small array with integer-offset addresses for each pixel-of-interest")
        .def("add_energy_multichannel_mask_allpanel",
             static_cast<void (exascale_api<memory_t>::*)(af::shared<int> const,kokkos_energy_channels&,kokkos_detector<memory_t>&, af::shared<std::size_t> const,
             af::shared<double> const) >
             (&exascale_api<memory_t>::add_energy_multichannel_mask_allpanel),
             (arg_("ichannels"), arg_("gpu_amplitudes"), arg_("gpu_detector"), arg_("pixel_active_list_ints"), arg_("weights")),
             "Point to Fhkl at a new energy channel on the GPU, and accumulate Bragg spots on mask==True pixels\n"
             "The pixel_active_list_ints is a small array with integer-offset addresses for each pixel-of-interest"
             "ichannels: for each nanoBragg source, the value instructs the simulation which channel in gpu_structure_factors"
             "to use for structure factor lookup.  If -1, skip this source wavelength."
             )
        .def("add_background", &simtbx::Kokkos::exascale_api<memory_t>::add_background,
             (arg_("detector"), arg_("override_source")=-1),
             "Add a background field directly on the GPU")
        .def("add_noise", &simtbx::Kokkos::exascale_api<memory_t>::add_noise,
             (arg_("detector")),
             "Modify pixels with noise on CPU. Unusual pattern, returns pixels directly instead of saving persistent")
        .def("show",&simtbx::Kokkos::exascale_api<memory_t>::show)
        .add_property("diffuse",
             make_getter(&simtbx::Kokkos::exascale_api<memory_t>::diffuse,rbv()),
             make_setter(&simtbx::Kokkos::exascale_api<memory_t>::diffuse,dcp()),
             "the diffuse parameters for the simulation.")
        ;
    }
  };

  struct diffuse_wrapper
  {
    static void
    wrap()
    {
      using namespace boost::python;
      typedef return_value_policy<return_by_value> rbv;
      typedef default_call_policies dcp;
      class_<simtbx::Kokkos::diffuse_api>("diffuse_api",no_init )
        .def(init<>())
        .add_property("enable",
             make_getter(&simtbx::Kokkos::diffuse_api::enable,rbv()),
             make_setter(&simtbx::Kokkos::diffuse_api::enable,dcp()),
             "whether or not to simulate diffuse.")
        .add_property("stencil_size",
             make_getter(&simtbx::Kokkos::diffuse_api::stencil_size,rbv()),
             make_setter(&simtbx::Kokkos::diffuse_api::stencil_size,dcp()),
             "")
        .add_property("symmetrize_diffuse",
             make_getter(&simtbx::Kokkos::diffuse_api::symmetrize_diffuse,rbv()),
             make_setter(&simtbx::Kokkos::diffuse_api::symmetrize_diffuse,dcp()),
             "")
        .add_property("laue_group_num",
             make_getter(&simtbx::Kokkos::diffuse_api::laue_group_num,rbv()),
             make_setter(&simtbx::Kokkos::diffuse_api::laue_group_num,dcp()),
             "")
        .add_property("anisoG",
             make_function(&get_anisoG,rbv()),
             make_function(&set_anisoG,dcp()),
             "")
        .add_property("anisoU",
             make_function(&get_anisoU,rbv()),
             make_function(&set_anisoU,dcp()),
             "")
        .add_property("rotate_principal_axes",
             make_function(&get_rotate_principal_axes,rbv()),
             make_function(&set_rotate_principal_axes,dcp()),
             "")
        .def("show",&simtbx::Kokkos::diffuse_api::show)
        ;
    }
  };

  } // namespace Kokkos

  BOOST_PYTHON_MODULE(simtbx_kokkos_ext) {
    simtbx::Kokkos::kokkos_instance_wrapper::wrap();
    simtbx::Kokkos::structure_factor_wrapper::wrap();
    simtbx::Kokkos::detector_wrapper<simtbx::Kokkos::large_array_policy>::wrap("gpu_detector");
    simtbx::Kokkos::detector_wrapper<simtbx::Kokkos::small_whitelist_policy>::wrap("gpu_detector_small_whitelist");
    simtbx::Kokkos::simulation_wrapper<simtbx::Kokkos::large_array_policy>::wrap("exascale_api");
    simtbx::Kokkos::simulation_wrapper<simtbx::Kokkos::small_whitelist_policy>::wrap("exascale_api_small_whitelist");
    simtbx::Kokkos::diffuse_wrapper::wrap();
  }
} // namespace simtbx
