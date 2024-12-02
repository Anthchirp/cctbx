#ifndef SIMTBX_DIFFBRAGG_H
#define SIMTBX_DIFFBRAGG_H

#include <simtbx/nanoBragg/nanoBragg.h>
#include <simtbx/diffBragg/src/util.h>
#include <vector>
#include <memory>
#include <boost/ptr_container/ptr_vector.hpp>
#include <Eigen/Dense>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <Eigen/StdVector>
#include <boost/python/numpy.hpp>

#ifdef DIFFBRAGG_HAVE_CUDA
#include "diffBraggCUDA.h"
#endif

#ifdef DIFFBRAGG_HAVE_KOKKOS
#include "diffBraggKOKKOS.h"
#endif

//#include <boost/python/numpy.hpp>
namespace np=boost::python::numpy;

namespace simtbx {
namespace nanoBragg {

class derivative_manager{
  public:
    virtual ~derivative_manager(){}
    derivative_manager();
    void initialize(int Npix_total, bool compute_curvatures);
    af::flex_double raw_pixels;
    af::flex_double raw_pixels2;
    double value; // the value of the parameter
    double dI; // the incremental derivative
    double dI2; // the incremental derivative
    bool refine_me;
    bool second_derivatives;
    void increment_image(int idx, double value, double value2, bool curvatures);

};

class panel_manager: public derivative_manager{
  public:
    panel_manager();
   virtual void foo(){}
    virtual ~panel_manager(){}

    Eigen::Matrix3d dR;
    Eigen::Vector3d dF;
    Eigen::Vector3d dS;
    Eigen::Vector3d dk;
    Eigen::Vector3d F_cross_dS;
    Eigen::Vector3d dF_cross_S;

}; // end of rot_manager


class rot_manager: public derivative_manager{
  public:
    rot_manager();
    virtual ~rot_manager(){}
    virtual void set_R();
    void increment(double value, double value2);
    Eigen::Matrix3d XYZ, XYZ2;
    Eigen::Matrix3d R, dR, dR2;

}; // end of rot_manager

class Ncells_manager: public derivative_manager{
  public:
    Ncells_manager();
    virtual ~Ncells_manager(){}
    void increment(double dI_increment, double dI2_increment);
};


class Fcell_manager: public derivative_manager{
  public:
    Fcell_manager();
    virtual ~Fcell_manager(){}
    void increment(double value, double value2);
};

class eta_manager: public derivative_manager{
  public:
    eta_manager();
    virtual ~eta_manager(){}
    void increment(double value, double value2);
};

class lambda_manager: public derivative_manager{
  public:
    lambda_manager();
    virtual ~lambda_manager(){}
    void increment(double value, double value2);
    double dg_dlambda;
};


class ucell_manager: public derivative_manager{
  public:
    ucell_manager();
    virtual ~ucell_manager(){}
    void increment(double value, double value2);
    Eigen::Matrix3d dB, dB2;
};

class origin_manager: public derivative_manager{
  public:
    origin_manager();
    virtual ~origin_manager(){}
    Eigen::Vector3d dk; /* derivative of the diffracted vector along this origin component */
    double FF, FdF, FdF2, dFdF;
};

class rotX_manager: public rot_manager{
  public:
    rotX_manager();
    void set_R();
};
class rotY_manager: public rot_manager{
  public:
    rotY_manager();
    void set_R();
};
class rotZ_manager: public rot_manager{
  public:
    rotZ_manager();
    void set_R();
};

class diffBragg: public nanoBragg{
  public:
  diffBragg(const dxtbx::model::Detector& detector, const dxtbx::model::Beam& beam,
            int verbose);

  ~diffBragg(){};

  af::shared<mat3> get_mosaic_blocks_prime();

  /// pixels
  double* floatimage_roi;
  af::flex_double raw_pixels_roi;
  af::flex_double ave_wavelength_img();
  boost::python::tuple ave_hkl_img();
  int Npix_total, Npix_to_model;
  void diffBragg_list_steps(step_arrays& db_steps);

  // containers
  images first_deriv_imgs, second_deriv_imgs;
  step_arrays db_steps;
  crystal db_cryst;
  beam db_beam;
  flags db_flags;
  cuda_flags db_cu_flags;
  detector db_det;

#ifdef DIFFBRAGG_HAVE_CUDA
    diffBragg_cudaPointers device_pointers;
    inline void cuda_free(){
        freedom(device_pointers);
    }
#endif

#ifdef DIFFBRAGG_HAVE_KOKKOS
    // diffBragg_cudaPointers cuda_pointers;
    inline void kokkos_free() { diffBragg_runner.reset(); }
    // allocate when needed to avoid problems with kokkos initialization when cuda/kokkos isn't used
    std::shared_ptr<diffBraggKOKKOS> diffBragg_runner{};
#endif

    inline void gpu_free(){
#ifdef DIFFBRAGG_HAVE_CUDA
            cuda_free();
#endif
#ifdef DIFFBRAGG_HAVE_KOKKOS
            kokkos_free();
#endif
    }

  // methods
  void update_xray_beams(scitbx::af::versa<dxtbx::model::Beam, scitbx::af::flex_grid<> > const& value);
  void initialize_managers();
  void diffBragg_rot_mats();
  void vectorize_umats();
  void rotate_fs_ss_vecs(double panel_rot_ang);
  void rotate_fs_ss_vecs_3D(double panel_rot_angO, double panel_rot_angF, double panel_rot_angS);
  void add_diffBragg_spots(const af::shared<size_t>& panels_fasts_slows);
  np::ndarray add_Fhkl_gradients(const af::shared<size_t>& panels_fasts_slows,
           np::ndarray& residual, np::ndarray& variance, np::ndarray& trusted, np::ndarray& freq,
           int num_Fhkl_channels, double Gscale, bool track, bool errors);
  void update_Fhkl_channels(np::ndarray& channels);
  boost::python::list get_Fhkl_channels();
  void update_Fhkl_scale_factors(np::ndarray& scale_factors, int num_Fhkl_channels);
  void add_diffBragg_spots(const af::shared<size_t>& panels_fasts_slows, boost::python::list per_pix_nominal_hkl);
  void add_diffBragg_spots();
  af::shared<double> add_diffBragg_spots_full();
  void init_raw_pixels_roi();
  void zero_raw_pixel_rois();
  void set_ucell_derivative_matrix(int refine_id, af::shared<double> const& value);
  void set_ucell_second_derivative_matrix(int refine_id, af::shared<double> const& value);
  void init_Fhkl2();
  inline void free_Fhkl2(){
      if(Fhkl2 != NULL) {
        for (h0=0; h0<=h_range;h0++) {
          for (k0=0; k0<=k_range;k0++) {
            free(Fhkl2[h0][k0]);
          }
          free(Fhkl2[h0]);
        }
        free(Fhkl2);
      }
  }
  //void reset_derivative_pixels(int refine_id);

  /* methods for interacting with the derivative managers */
  void refine(int refine_id);
  void fix(int refine_id);
  void let_loose(int refine_id);
  int detector_panel_id;
  void shift_originZ(const dxtbx::model::Detector& detector, double shift);
  void update_dxtbx_geoms(const dxtbx::model::Detector& detector, const dxtbx::model::Beam& beam,
        int panel_id, double panel_rot_angO=0,
        double panel_rot_angF=0,  double panel_rot_angS=0, double panel_offsetX=0,
        double panel_offsetY=0, double panel_offsetZ=0, bool force=true);
  void set_value( int refine_id, double value);
  void set_ncells_values( boost::python::tuple const& values);
  void set_ncells_def_values( boost::python::tuple const& values);
  void print_if_refining();
  boost::python::tuple get_ncells_values();
  double get_value( int refine_id);
  af::flex_double get_derivative_pixels(int refine_id);
  af::flex_double get_second_derivative_pixels(int refine_id);
  af::flex_double get_raw_pixels_roi();
  boost::python::tuple get_fp_fdp_derivative_pixels();
  boost::python::tuple get_ncells_derivative_pixels();

  PyObject* PyCapsule_Wrapper(DLManagedTensor* (diffBraggKOKKOS::*func)(), image_type &vec);
  PyObject* get_floatimage();
  PyObject* get_wavelenimage();
  PyObject* get_d_diffuse_gamma_images();
  PyObject* get_d_diffuse_sigma_images();
  PyObject* get_d_Umat_images();
  PyObject* get_d2_Umat_images();
  PyObject* get_d_Bmat_images();
  PyObject* get_d2_Bmat_images();
  PyObject* get_d_Ncells_images();
  PyObject* get_d2_Ncells_images();
  PyObject* get_d_fcell_images();
  PyObject* get_d2_fcell_images();
  PyObject* get_d_eta_images();
  PyObject* get_d2_eta_images();
  PyObject* get_d_lambda_images();
  PyObject* get_d2_lambda_images();
  PyObject* get_d_panel_rot_images();
  PyObject* get_d2_panel_rot_images();
  PyObject* get_d_panel_orig_images();
  PyObject* get_d2_panel_orig_images();
  PyObject* get_d_fp_fdp_images();
  PyObject* get_Fhkl_scale_deriv();
  PyObject* get_Fhkl_hessian();

  boost::python::tuple get_diffuse_gamma_derivative_pixels();
  boost::python::tuple get_diffuse_sigma_derivative_pixels();
  boost::python::tuple get_ncells_def_derivative_pixels();
  boost::python::tuple get_ncells_def_second_derivative_pixels();
  boost::python::tuple get_ncells_second_derivative_pixels();
  boost::python::tuple get_aniso_eta_deriv_pixels();
  boost::python::tuple get_aniso_eta_second_deriv_pixels();
  boost::python::tuple get_lambda_derivative_pixels();

  Eigen::Vector3d O_reference;

  Eigen::Matrix3d EYE;
  mat3 Umatrix;
  mat3 Bmatrix;
  mat3 Omatrix;

  // Panel rotation
  double panel_rot_ang;

  //polarization
  Eigen::Vector3d Ei_vec, Bi_vec;

  // mosaic spread
  double * mosaic_umats_prime;
  int nmats;
  double * mosaic_umats_dbl_prime;
  void set_mosaic_blocks_prime(af::shared<mat3> umat_in);  // set the individual mosaic block orientation derivatives from array
  void set_mosaic_blocks_dbl_prime(af::shared<mat3> umat_in);  // set the individual mosaic block orientation derivatives from array
  //bool vectorized_umats;

  /* derivative managers */
  std::vector<boost::shared_ptr<Fcell_manager> > fp_fdp_managers;
  std::vector<boost::shared_ptr<rot_manager> > rot_managers;
  std::vector<boost::shared_ptr<ucell_manager> > ucell_managers;
  std::vector<boost::shared_ptr<Ncells_manager> > Ncells_managers;
  std::vector<boost::shared_ptr<eta_manager> > eta_managers;
  std::vector<boost::shared_ptr<origin_manager> > origin_managers;
  std::vector<boost::shared_ptr<lambda_manager> > lambda_managers;
  std::vector<boost::shared_ptr<panel_manager> > panels;
  std::vector<boost::shared_ptr<derivative_manager> > fcell_managers;
  //boost::shared_ptr<eta_manager> eta_man;
  boost::shared_ptr<panel_manager> panel_rot_man;
  boost::shared_ptr<panel_manager> panel_rot_manF;
  boost::shared_ptr<panel_manager> panel_rot_manS;

  bool compute_curvatures;
  bool update_oversample_during_refinement;
  bool oversample_omega;
  bool only_save_omega_kahn;
  bool host_transfer;

  // miller array
  void quick_Fcell_update(boost::python::tuple const& value);
  double ***Fhkl2;  // = NULL
  af::shared<double> pythony_amplitudes2;
  bool complex_miller;
  double F_cell2; // for storing the imaginary component
  std::vector<double> fpfdp;
  std::vector<double> fpfdp_derivs;
  std::vector<double> atom_data;
  void show_heavy_atom_data();
  void show_fp_fdp();
  bool track_Fhkl;
  std::vector<int> nominal_hkl;
  void linearize_Fhkl(bool compute_dists);
  void sanity_check_linear_Fhkl();
  void update_linear_Fhkl();

  // mosaic domain
  bool isotropic_ncells;
  bool modeling_anisotropic_mosaic_spread;
  double Nd, Ne, Nf;
  bool refine_Ncells_def;
  bool no_Nabc_scale;  // if true, then absorb the Nabc scale into an overall scale factor
  double prev_shiftZ=0; // keep track of when detector Z was shifted  (helps determine when to set the update_detector flag for GPU devices
  Eigen::Matrix3d NABC;

  bool use_lambda_coefficients;

  void set_close_distances();

  // cuda properties
  bool update_dB_matrices_on_device=false;
  bool update_detector_on_device=false;
  bool update_rotmats_on_device=false;
  bool update_umats_on_device=false;
  bool update_panels_fasts_slows_on_device=false;
  bool update_Fhkl_on_device=false;
  bool update_refine_flags_on_device=false;
  bool update_step_positions_on_device=false;
  bool update_panel_deriv_vecs_on_device=false;
  bool use_gpu=false;
  bool force_cpu=false;
  int Npix_to_allocate=-1; // got GPU allocation, -1 is auto mode

  timer_variables TIMERS;
  bool record_time = true;

  void show_timing_stats(int MPI_RANK);
  bool last_kernel_on_GPU; // reveals whether the GPU kernel was run
  boost::python::tuple get_ave_I_cell(bool use_Fhkl_scale, int i_channel, bool use_geometric_mean);
  np::ndarray Fhkl_restraint_data(int i_channel, double Fhkl_beta, bool use_geometric_mean, int flag);
  void set_Friedel_mate_inds(boost::python::list pos_inds, boost::python::list neg_inds);
}; // end of diffBragg


double diffBragg_cpu_kernel_polarization(
    Eigen::Vector3d incident,
    Eigen::Vector3d diffracted,
    Eigen::Vector3d polarization_axis,
    double kahn_factor);

std::vector<double> I_cell_ave(crystal& db_cryst,bool use_Fhkl_scale, int i_channel, std::vector<double>& Fhkl_scale);
void Ih_grad_terms(crystal& db_cryst, int i_chan, std::vector<double>& Fhkl_scale, std::vector<double>& out);
void Friedel_grad_terms(crystal& db_cryst, int i_chan, std::vector<double>& Fhkl_scale, std::vector<double>& out);
void Finit_grad_terms(crystal& db_cryst, int i_chan, std::vector<double>& Fhkl_scale, std::vector<double>& out);

void diffBragg_sum_over_steps(
      int Npix_to_model, std::vector<unsigned int>& panels_fasts_slows,
      image_type& floatimage,
      images& d_image,
      images& d2_image,
      step_arrays& db_steps,
      detector& db_det,
      beam& db_beam,
      crystal& db_cryst,
      flags& db_flags);

af::flex_double get_panel_increment(double Iincrement, double omega_pixel, const Eigen::Ref<const Eigen::Matrix3d>& M, double pix2,
          const Eigen::Ref<const Eigen::Vector3d>& o, const Eigen::Ref<const Eigen::Vector3d>& k_diffracted,
          double per_k, double per_k3, double per_k5, const Eigen::Ref<const Eigen::Vector3d>& V, const Eigen::Ref<const Eigen::Vector3d>& _dk);

} // end of namespace nanoBragg
} // end of namespace simtbx
#endif
