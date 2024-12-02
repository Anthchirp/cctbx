#include <Eigen/Dense>
#include <iostream>
#include <math.h>
#include <vector>
#ifndef CUDAREAL
#define CUDAREAL double
#endif

typedef std::vector<CUDAREAL> image_type;
typedef Eigen::Matrix<CUDAREAL,3,1> VEC3;
typedef Eigen::Matrix<CUDAREAL,3,3> MAT3;
typedef std::vector<MAT3,Eigen::aligned_allocator<MAT3> > eigMat3_vec;
typedef std::vector<VEC3,Eigen::aligned_allocator<VEC3> > eigVec3_vec;

__global__ void gpu_sum_over_steps(
        int Npix_to_model, unsigned int* panels_fasts_slows,
        CUDAREAL* floatimage,
        CUDAREAL* wavelenimage,
        CUDAREAL* d_Umat_images, CUDAREAL* d2_Umat_images,
        CUDAREAL* d_Bmat_images, CUDAREAL* d2_Bmat_images,
        CUDAREAL* d_Ncells_images, CUDAREAL* d2_Ncells_images,
        CUDAREAL* d_fcell_images, CUDAREAL* d2_fcell_images,
        CUDAREAL* d_eta_images, CUDAREAL* d2_eta_images,
        CUDAREAL* d_lambda_images, CUDAREAL* d2_lambda_images,
        CUDAREAL* d_panel_rot_images, CUDAREAL* d2_panel_rot_images,
        CUDAREAL* d_panel_orig_images, CUDAREAL* d2_panel_orig_images,
        CUDAREAL* d_fp_fdp_images,
        const int Nsteps, int _printout_fpixel, int _printout_spixel, bool _printout, CUDAREAL _default_F,
        int oversample, bool _oversample_omega, CUDAREAL subpixel_size, CUDAREAL pixel_size,
        CUDAREAL detector_thickstep, CUDAREAL _detector_thick, const CUDAREAL* __restrict__ close_distances, CUDAREAL detector_attnlen,
        int detector_thicksteps, int sources, int phisteps, int mosaic_domains,
        bool use_lambda_coefficients, CUDAREAL lambda0, CUDAREAL lambda1,
        MAT3 eig_U, MAT3 eig_O, MAT3 eig_B, MAT3 RXYZ,
        VEC3* dF_vecs,
        VEC3* dS_vecs,
        const MAT3* __restrict__  UMATS_RXYZ,
        MAT3* UMATS_RXYZ_prime,
        MAT3* UMATS_RXYZ_dbl_prime,
        MAT3* RotMats,
        MAT3* dRotMats,
        MAT3* d2RotMats,
        MAT3* UMATS,
        MAT3* dB_mats,
        MAT3* dB2_mats,
        MAT3* Amatrices,
        const CUDAREAL* __restrict__ source_X, const CUDAREAL* __restrict__ source_Y,
        const CUDAREAL* __restrict__ source_Z, const CUDAREAL* __restrict__ source_lambda,
        const CUDAREAL* __restrict__ source_I,
        CUDAREAL kahn_factor,
        CUDAREAL Na, CUDAREAL Nb, CUDAREAL Nc,
        CUDAREAL Nd, CUDAREAL Ne, CUDAREAL Nf,
        CUDAREAL phi0, CUDAREAL phistep,
        VEC3 spindle_vec, VEC3 _polarization_axis,
        int h_range, int k_range, int l_range,
        int h_max, int h_min, int k_max, int k_min, int l_max, int l_min, CUDAREAL dmin,
        CUDAREAL fudge, bool complex_miller, int verbose, bool only_save_omega_kahn,
        bool isotropic_ncells, bool compute_curvatures,
        const CUDAREAL* __restrict__ _FhklLinear, const CUDAREAL* __restrict__ _Fhkl2Linear,
        bool* refine_Bmat, bool* refine_Ncells, bool refine_Ncells_def,  bool* refine_panel_origin, bool* refine_panel_rot,
        bool refine_fcell, bool* refine_lambda, bool refine_eta, bool* refine_Umat,
        const CUDAREAL* __restrict__ fdet_vectors, const CUDAREAL* __restrict__ sdet_vectors,
        const CUDAREAL* __restrict__ odet_vectors, const CUDAREAL* __restrict__ pix0_vectors,
        bool _nopolar, bool _point_pixel, CUDAREAL _fluence, CUDAREAL _r_e_sqr, CUDAREAL _spot_scale, int Npanels,
        bool aniso_eta, bool no_Nabc_scale,
        const CUDAREAL* __restrict__ fpfdp,
        const CUDAREAL* __restrict__ fpfdp_derivs,
        const CUDAREAL*__restrict__ atom_data, int num_atoms,
        bool doing_fp_fdp_derivs,
        const int* __restrict__ nominal_hkl, bool use_nominal_hkl, MAT3 anisoU, MAT3 anisoG, MAT3 rotate_principal_axes,
        bool use_diffuse, CUDAREAL* d_diffuse_gamma_images, CUDAREAL* d_diffuse_sigma_images, bool refine_diffuse, bool gamma_miller_units,
        bool refine_Icell, bool save_wavelenimage, int laue_group_num, int stencil_size,
        bool Fhkl_gradient_mode, bool Fhkl_errors_mode, bool using_trusted_mask, bool Fhkl_channels_empty, bool Fhkl_have_scale_factors,
        int Num_ASU,
        const CUDAREAL* __restrict__ data_residual, const CUDAREAL* __restrict__ data_variance,
        const int* __restrict__ data_freq, const bool* __restrict__ data_trusted,
        const int* __restrict__ FhklLinear_ASUid,
        const CUDAREAL* __restrict__ Fhkl_channels,
        const CUDAREAL* __restrict__ Fhkl_scale, CUDAREAL* Fhkl_scale_deriv,
        bool gaussian_star_shape, bool square_shape
        );
