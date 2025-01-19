
#ifndef SIMTBX_DIFFBRAGG_UTIL
#define SIMTBX_DIFFBRAGG_UTIL

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <sys/time.h>

#ifndef CUDAREAL
    #define CUDAREAL double
#endif

#include "simtbx/nanoBragg/nanotypes.h"
using simtbx::nanoBragg::shapetype;
using simtbx::nanoBragg::SQUARE;
using simtbx::nanoBragg::GAUSS;
using simtbx::nanoBragg::GAUSS_STAR;

using image_type = std::vector<CUDAREAL>;
typedef Eigen::Matrix<double,3,1> VEC3;
typedef Eigen::Matrix<double,3,3> MAT3;
typedef std::vector<MAT3,Eigen::aligned_allocator<MAT3> > eigMat3_vec;
typedef std::vector<VEC3,Eigen::aligned_allocator<VEC3> > eigVec3_vec;

inline void easy_time(double& timer, struct timeval& t, bool recording){
    double before_sec = t.tv_sec;
    double before_usec = t.tv_usec;
    gettimeofday(&t, 0);
    double time = (1000000.0 * (t.tv_sec - before_sec) + t.tv_usec - before_usec) / 1000.0;
    if (recording)
        timer += time;
}

struct timer_variables{
    double add_spots_pre=0; // times the initializations for add spots kernel
    double add_spots_post=0; // times the copies that occur after add spots kernel
    double add_spots_kernel_wrapper=0; // times the add spots kernel overall, either CPU or GPU
    double cuda_alloc=0; // times the allocation of the device
    double cuda_copy_to_dev=0; // times the copying from host to device
    double cuda_copy_from_dev=0; // times the copying back from device to host
    double cuda_kernel=0; // times the GPU kernel
    double copy_sources=0;
    double copy_Fhkl_scale=0;
    double copy_umats=0;
    double copy_amats=0;
    double copy_bmats=0;
    double copy_rotmats=0;
    double copy_det=0;
    double copy_nomhkl=0;
    double copy_flags=0;
    double copy_fhkl=0;
    double copy_detderiv=0;
    double copy_pfs=0;
    int timings=0; // how many times these variables were incremented
    bool recording=true;
  };


// CONTAINERS
struct images{
    image_type Fhkl_hessian;
    image_type Fhkl_scale;
    image_type Fhkl_scale_deriv;
    std::vector<bool> trusted;
    std::vector<int> freq;
    image_type residual;
    image_type variance;
    image_type wavelength; // image for storing mean wavelength of each pixel
    image_type Umat; // umatrix gradients
    image_type Bmat;  // Bmatrix gradients
    image_type Ncells; // mosaic domain size gradients
    image_type fcell; // structure factor gradients
    image_type eta; // mosaic spread gradients
    image_type lambda; // spectrum affine transform gradients
    image_type panel_rot; // panel rotation gradients
    image_type panel_orig; // panel translation gradients
    image_type fp_fdp;  // fprime and fdblprime gradients
    image_type diffuse_gamma; // diffuse gamma gradients
    image_type diffuse_sigma; // diffuse sigma gradients
};


struct step_arrays{
   int* subS_pos; // stepping through the slow-scan detector axis
   int* subF_pos; // ''   fast-scan ''
   int* thick_pos; // stepping through the detector thickness
   int* source_pos; // stepping through the beam wavelengths
   int* phi_pos;  // stepping through the gonio scan
   int* mos_pos;  // stepping through mosaic blocks (for mosaic spread)
   int Nsteps; // total number of steps
};


struct cuda_flags{
    int device_Id=0;  // gpu device id
    int Npix_to_allocate; // how much space to allocate for simulating forward model and gradients
    // these following flags indicate whether to update quantities on the GPU device prior to running the kernel
    // ( of course they are all set prior to running the kernel for the first time)
    bool update_step_positions = false;  // step arrays
    bool update_panels_fasts_slows = false; // pixels to simulatoe (panel id, fast scan, slow scan)
    bool update_sources = false;  // beam sources
    bool update_umats = false; // umatrices for mosaic blocks
    bool update_dB_mats = false; // derivative of the orthogonalization matrix (for unit cell derivatives)
    bool update_rotmats = false; // rotation matrices (for Umat derivatives)
    bool update_Fhkl = false; // structure factors
    bool update_Fhkl_scales = false; // structure factors
    bool update_Fhkl_channels = false; // structure factors
    bool update_detector = false; // detector vectors (origin, slow-axis, fast-axis, orth-axis)
    bool update_refine_flags = false;  // refinement flags (in case one is iteratively freezing parameters)
    bool update_panel_deriv_vecs = false; // if one is refining the detector vectors)
};

struct flags{
    bool Fhkl_errors_mode=false;
    bool track_Fhkl_indices = false;
    bool Fhkl_have_scale_factors = false;
    bool using_trusted_mask=false;
    bool Fhkl_gradient_mode=false;
    bool wavelength_img=false;
    bool track_Fhkl = false; // for CPU kernel only, track the HKLS evaluated in the inner most loop
    bool printout = false; // whether to printout debug info for a pixel
    bool nopolar = false; // disable polarization effects
    bool point_pixel = false; // approximate solid angle effects
    bool only_save_omega_kahn = false; // only save the polarization and solid angle corrections (deprecated)
    bool compute_curvatures = false; // whether to compute the curvatures in addition to gradients
    bool isotropic_ncells = false; // one mosaic domain parameter
    bool complex_miller = false;  // is the miller array complex (such thet Fhkl_linear and Fhkl2_linear are both defined)
    bool no_Nabc_scale = false; // no Nabc prefactor
    bool refine_diffuse = false; // flag for computing diffuse gradients
    std::vector<bool> refine_Bmat;  //  Bmatrix
    std::vector<bool> refine_Ncells; // mosaic domain size
    bool refine_Ncells_def = false; // mosaic domain size off diag
    std::vector<bool> refine_panel_origin; // panel shift
    std::vector<bool> refine_panel_rot; // detector panel rotation
    bool refine_fcell = false; // structure factor
    std::vector<bool> refine_lambda; // spectrum affine correction
    bool refine_eta = false; // mosaic spread
    std::vector<bool> refine_Umat; // missetting angle umatrix
    bool refine_fp_fdp = false; // fprime and fbl prime
    bool use_lambda_coefficients = false; // affine correction lam0 , lam1
    bool oversample_omega = false; // omega is computed separately for each sub-pixel
    int printout_fpixel = 0, printout_spixel = 0; // debug printout pixel (fast scan, slow scan) // TODO add panel id
    int verbose = 0; // nanoBragg verbosity flag
    bool use_diffuse = false; // model  diffuse
    bool only_diffuse = false; // model  diffuse scattering (experimental)
    bool refine_Icell = false; // option to refine the structure factor intensity directly (F_cell^2)
                                // The miller array used by nanoBragg/diffBragg is double precision, and hence
                                // allows for negative values. If refine_Icell=True, then the value stored in the data
                                // component of the miller array is assumed to be an intensity, as opposed to an amplitude.
                                // In such cases, the gradient of the scattering w.r.t. this parameter is modified accordingly
                                // such that one could use those gradients as part of a refinement protocol to optimize I_cell

    bool gamma_miller_units = false; // use Miller index units for diffuse gamma matrix
};

struct crystal{
    shapetype xtal_shape = GAUSS;
    double Friedel_beta = 1e10; // restraint factor for Friedel pairs
    double Finit_beta = 1e10; // restraint factor for Friedel pairs
    std::vector<int> pos_inds; // indices of the positive Friedel mate
    std::vector<int> neg_inds; // indices of the negative Friedel mate
    double Fhkl_beta=1e10;
    bool use_geometric_mean=false;
    std::unordered_set<int> Fhkl_grad_idx_tracker;
    int num_Fhkl_channels=1;
    int laue_group_num=1;
    int stencil_size=0;
    Eigen::Matrix3d anisoG;
    Eigen::Matrix3d rotate_principal_axes;
    std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > dG_dgamma;
    std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > dU_dsigma;
    Eigen::Matrix3d anisoU;
    int mosaic_domains; // number of mosaic domains to model
    CUDAREAL Na, Nb, Nc, Nd, Ne, Nf; // mosaic domain terms
    CUDAREAL phi0; // gonio
    CUDAREAL phistep;
    int phisteps;
    CUDAREAL fudge; // factor for Bragg peak exponential falloff adjustment
    CUDAREAL spot_scale; // factor applied to intensity
    int h_range, k_range, l_range;
    int h_max, h_min, k_max, k_min, l_max, l_min;
    CUDAREAL dmin; //res
    std::vector<double> dspace_bins;
    std::vector<CUDAREAL> FhklLinear, Fhkl2Linear; // structure factor amps magnitude (or real, image of complex)
    std::vector<double> ASU_dspace, ASU_Fcell;
    std::vector<int> FhklLinear_ASUid;
    std::unordered_map<std::string, int> ASUid_map;
    int Num_ASU;
    std::string hall_symbol =" P 4nw 2abw";
    std::vector<CUDAREAL> fpfdp; // fprim fdblprime
    std::vector<CUDAREAL> fpfdp_derivs; // fprime fdblprime deriv
    std::vector<CUDAREAL> atom_data; // heavy atom data
    std::vector<int> nominal_hkl; // h,k,l of the pixel (expected)
    CUDAREAL default_F; // place holder amplitude
    CUDAREAL r_e_sqr; // electron rad

    Eigen::Matrix3d eig_U; // Umatrix
    Eigen::Matrix3d eig_O; // O-matrix
    Eigen::Matrix3d eig_B; // B matrix
    Eigen::Matrix3d RXYZ; // Rx*Ry*Rz misset perturtbation matrix (this is whats refined)
    Eigen::Vector3d spindle_vec; // gonio

    std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > UMATS_RXYZ;
    std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > UMATS_RXYZ_prime;
    std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > UMATS_RXYZ_dbl_prime;

    std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > RotMats;
    std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > dRotMats;
    std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > d2RotMats;
    std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > UMATS;
    std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > UMATS_prime;
    std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > UMATS_dbl_prime;
    std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > dB_Mats;
    std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > dB2_Mats;

};

struct beam{
    Eigen::Vector3d polarization_axis;
    std::vector<int> Fhkl_channels; // if refining scale factors for wavelength dependent structure factor intensities
    CUDAREAL fluence; // total fluence
    CUDAREAL kahn_factor; // polarization factor
    CUDAREAL *source_X, *source_Y, *source_Z; // beam vectors
    CUDAREAL *source_lambda; // wavelengths
    CUDAREAL *source_I; // intensities
    CUDAREAL lambda0,lambda1; // affine correction to spectra
    int number_of_sources; // number of beams
};

struct detector{
    std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > dF_vecs; // derivative of the panel fast direction
    std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > dS_vecs; // derivative of the panel slow direction
    CUDAREAL detector_thickstep;
    int detector_thicksteps;
    CUDAREAL detector_thick;
    CUDAREAL detector_attnlen;
    std::vector<CUDAREAL> close_distances; // offsets to the detector origins (Z direction)
    int oversample; // determines the pixel subsampling rate
    CUDAREAL subpixel_size, pixel_size;
    std::vector<CUDAREAL> fdet_vectors, sdet_vectors, odet_vectors, pix0_vectors; // these define the detector (fast, slow, orth, origin)
};

#endif
