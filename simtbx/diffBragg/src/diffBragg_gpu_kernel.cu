#include "diffBraggCUDA.h"
#include <simtbx/diffBragg/src/diffuse_util.h>
#include <stdio.h>

/* Fourier transform of a grating */
__device__ CUDAREAL sincg(CUDAREAL x, CUDAREAL N) {
	if (x != 0.0)
		return sin(x * N) / sin(x);
	return N;
}

__global__
void gpu_sum_over_steps(
        int Npix_to_model, unsigned int* panels_fasts_slows,
        CUDAREAL* floatimage,
        CUDAREAL* wavelenimage,
        CUDAREAL* d_Umat_images, CUDAREAL* d2_Umat_images,
        CUDAREAL* d_Bmat_images, CUDAREAL* d2_Bmat_images,
        CUDAREAL* d_Ncells_images, CUDAREAL* d2_Ncells_images,
        CUDAREAL* d_fcell_images, CUDAREAL* d2_fcell_images,
        CUDAREAL* d_eta_images,
        CUDAREAL* d2_eta_images,
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
        const MAT3* __restrict__ UMATS_RXYZ,
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
        bool* refine_Bmat, bool* refine_Ncells, bool refine_Ncells_def, bool* refine_panel_origin, bool* refine_panel_rot,
        bool refine_fcell, bool* refine_lambda, bool refine_eta, bool* refine_Umat,
        const CUDAREAL* __restrict__ fdet_vectors, const CUDAREAL* __restrict__ sdet_vectors,
        const CUDAREAL* __restrict__ odet_vectors, const CUDAREAL* __restrict__ pix0_vectors,
        bool _nopolar, bool _point_pixel, CUDAREAL _fluence, CUDAREAL _r_e_sqr, CUDAREAL _spot_scale, int Npanels,
        bool aniso_eta, bool no_Nabc_scale,
        const CUDAREAL* __restrict__ fpfdp,
        const CUDAREAL* __restrict__ fpfdp_derivs,
        const CUDAREAL* __restrict__ atom_data, int num_atoms, bool refine_fp_fdp,
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
        bool gaussian_star_shape, bool square_shape)
{ // BEGIN GPU kernel

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_stride = blockDim.x * gridDim.x;
    __shared__ CUDAREAL s_phi0, s_phistep, gx,gy,gz;
    __shared__ int s_phisteps;
    __shared__ bool s_gaussian_star_shape;
    __shared__ bool s_square_shape;
    __shared__ bool s_Fhkl_channels_empty;
    __shared__ bool s_Fhkl_have_scale_factors;
    __shared__ bool s_Fhkl_gradient_mode;
    __shared__ bool s_Fhkl_errors_mode;
    __shared__ int s_Num_ASU;
    __shared__ bool s_refine_Icell;
    __shared__ bool s_use_diffuse;
    __shared__ bool s_use_nominal_hkl;
    __shared__ bool s_refine_fp_fdp;
    __shared__ bool s_complex_miller;
    __shared__ int s_num_atoms;
    __shared__ bool s_aniso_eta;
    __shared__ bool s_no_Nabc_scale;
    __shared__ bool s_compute_curvatures;
    __shared__ MAT3 s_Ot;
    __shared__ MAT3 Ainv;
    __shared__ bool s_refine_diffuse;
    __shared__ bool s_gamma_miller_units;
    __shared__ MAT3 _NABC;
    __shared__ MAT3 s_dN;
    __shared__ CUDAREAL C;
    __shared__ CUDAREAL two_C;
    __shared__ MAT3 Bmat_realspace;
    __shared__ MAT3 Amat_init;
    __shared__ CUDAREAL s_Na;
    __shared__ CUDAREAL s_Nb;
    __shared__ CUDAREAL s_Nc;
    __shared__ CUDAREAL s_NaNbNc_squared;
    __shared__ int s_h_max, s_k_max, s_l_max, s_h_min, s_k_min, s_l_min, s_h_range, s_k_range, s_l_range;
    __shared__ int s_oversample, s_detector_thicksteps, s_sources, s_mosaic_domains,  s_printout_fpixel,
        s_printout_spixel, s_verbose, s_Nsteps;
    __shared__ CUDAREAL s_detector_thickstep, s_detector_attnlen, s_subpixel_size, s_pixel_size, s_lambda0,
        s_lambda1, sX0, sY0, sZ0, s_detector_thick, s_default_F,  s_overall_scale, s_kahn_factor;
    __shared__ bool s_oversample_omega, s_printout, s_nopolar;
    __shared__ VEC3 s_polarization_axis;

    __shared__ bool s_refine_Umat[3];
    __shared__ bool s_refine_panel_origin[3];
    __shared__ bool s_refine_panel_rot[3];
    __shared__ bool s_refine_Ncells[3];
    __shared__ bool s_refine_eta;
    __shared__ bool s_refine_Ncells_def;
    __shared__ bool s_refine_fcell;
    __shared__ bool s_refine_Bmat[6];
    __shared__ bool s_refine_lambda[2];
    __shared__ double s_NABC_det, s_NABC_det_sq;
    __shared__ MAT3 anisoG_local;
    __shared__ MAT3 anisoU_local;
    __shared__ MAT3 laue_mats[24];
    __shared__ MAT3 dG_dgam[3];
    __shared__ int num_laue_mats;
    __shared__ int dhh, dkk, dll;
    __shared__ VEC3 Hmin, Hmax, dHH, Hrange;
    //extern __shared__ CUDAREAL det_vecs[];
    //__shared__ int det_stride;
    __shared__ CUDAREAL s_xtal_size_sq;

    if (threadIdx.x==0){ // TODO can we get speed gains by dividing up the following definitions over more threads ?
        s_phisteps = phisteps;
        s_phi0 = phi0;
        s_phistep = phistep;
        gx = spindle_vec[0];
        gy = spindle_vec[1];
        gz = spindle_vec[2];
        s_gaussian_star_shape = gaussian_star_shape;
        s_square_shape = square_shape;
        for (int i=0; i<3; i++){
            s_refine_Ncells[i] = refine_Ncells[i];
            s_refine_Umat[i] = refine_Umat[i];
            s_refine_panel_origin[i] = refine_panel_origin[i];
            s_refine_panel_rot[i] = refine_panel_rot[i];
        }
        s_Fhkl_channels_empty = Fhkl_channels_empty;
        s_Fhkl_have_scale_factors = Fhkl_have_scale_factors;
        s_Fhkl_gradient_mode = Fhkl_gradient_mode;
        s_Fhkl_errors_mode = Fhkl_errors_mode;
        s_Num_ASU = Num_ASU;
        s_refine_Icell = refine_Icell;
        s_use_nominal_hkl = use_nominal_hkl;
        s_aniso_eta = aniso_eta;
        s_no_Nabc_scale = no_Nabc_scale;
        s_complex_miller = complex_miller;
        s_refine_lambda[0] = refine_lambda[0];
        s_refine_lambda[1] = refine_lambda[1];
        for(int i=0; i<6; i++){
            s_refine_Bmat[i] = refine_Bmat[i];
        }
        s_use_diffuse = use_diffuse;
        s_num_atoms = num_atoms;
        s_refine_fcell = refine_fcell;
        s_refine_eta = refine_eta;
        s_refine_Ncells_def = refine_Ncells_def;
        s_compute_curvatures = compute_curvatures;
        s_refine_fp_fdp = refine_fp_fdp;
        s_refine_diffuse = refine_diffuse;
        s_gamma_miller_units = gamma_miller_units;

        Bmat_realspace = eig_B*1e10;
        s_Ot = eig_O.transpose();
        Amat_init = eig_U*Bmat_realspace*s_Ot;
        Ainv = eig_U*(Bmat_realspace.transpose().inverse())* (eig_O.inverse());
        _NABC << Na,Nd,Nf,
                Nd,Nb,Ne,
                Nf,Ne,Nc;
        s_NABC_det = _NABC.determinant(); // TODO is this slow ?
        s_NABC_det_sq = s_NABC_det*s_NABC_det;
        C = 2 / 0.63 * fudge;
        two_C = 2*C;
        s_Na = Na;
        s_Nb = Nb;
        s_Nc = Nc;
        s_NaNbNc_squared = (Na*Nb*Nc);
        s_NaNbNc_squared *= s_NaNbNc_squared;
        s_h_max = h_max;
        s_k_max = k_max;
        s_l_max = l_max;
        s_h_min = h_min;
        s_k_min = k_min;
        s_l_min = l_min;
        s_h_range = h_range;
        s_k_range = k_range;
        s_l_range = l_range;

        s_oversample = oversample;
        s_detector_thicksteps = detector_thicksteps;
        s_sources = sources;
        s_mosaic_domains = mosaic_domains;
        s_detector_thickstep = detector_thickstep;
        s_detector_attnlen = detector_attnlen;
        s_subpixel_size = subpixel_size;
        s_pixel_size = pixel_size;
        s_detector_thick = _detector_thick;
        s_lambda0 = lambda0;
        s_lambda1 = lambda1;
        s_oversample_omega = _oversample_omega;
        s_printout = _printout;
        s_printout_fpixel = _printout_fpixel;
        s_printout_spixel = _printout_spixel;
        s_pixel_size = pixel_size;
        s_default_F = _default_F;
        s_verbose = verbose;
        s_polarization_axis = _polarization_axis;
        s_kahn_factor = kahn_factor;
        s_nopolar = _nopolar;
        sX0 = source_X[0];
        sY0 = source_Y[0];
        sZ0 = source_Z[0];
        s_Nsteps = Nsteps;
        s_overall_scale = _r_e_sqr *_spot_scale * _fluence / Nsteps ;

        if (s_use_diffuse){
            anisoG_local = anisoG;
            anisoU_local = anisoU;

            num_laue_mats = gen_laue_mats(laue_group_num, laue_mats, rotate_principal_axes);
            for (int i_gam=0; i_gam<3; i_gam++){
              dG_dgam[i_gam] << 0,0,0,0,0,0,0,0,0;
              dG_dgam[i_gam](i_gam, i_gam) = 1;
            }
            if (s_gamma_miller_units){
              anisoG_local = anisoG_local * Bmat_realspace;
              for (int i_gam=0; i_gam<3; i_gam++){
                dG_dgam[i_gam] = dG_dgam[i_gam] * Bmat_realspace;
              }
            }
            dhh = dkk = dll = stencil_size; // Limits of stencil for diffuse calc
        }
        Hmin << s_h_min,s_k_min,s_l_min;
        Hmax << s_h_max,s_k_max,s_l_max;
        dHH << dhh,dkk,dll;
        Hrange << s_h_range,s_k_range,s_l_range;
        //det_stride = Npanels*3;
        //for(int i=0; i< det_stride; i++){
        //    det_vecs[i] = fdet_vectors[i];
        //    det_vecs[i+det_stride] = sdet_vectors[i];
        //}
        MAT3 Amat0 = Amatrices[0];
        VEC3 A(Amat0(0,0),Amat0(0,1), Amat0(0,2));
        VEC3 B(Amat0(1,0),Amat0(1,1), Amat0(1,2));
        VEC3 C(Amat0(2,0),Amat0(2,1), Amat0(2,2));
        CUDAREAL cell_vol = A.dot(B.cross(C));
        s_xtal_size_sq = pow(s_NABC_det*cell_vol, (CUDAREAL)2/(CUDAREAL)3);
    }

    //extern __shared__ CUDAREAL source_data[];
    //int threads_per_block = blockDim.x;
    //int num_source_blocks = (sources + threads_per_block-1)/threads_per_block;
    //int tx = threadIdx.x;
    //if (tx < threads_per_block){
    //    for (int i_source_block=0; i_source_block < num_source_blocks; i_source_block++){
    //        int idx = i_source_block*threads_per_block + tx;
    //        if (idx< sources){
    //            source_data[idx] = source_X[idx];
    //            source_data[sources+idx] = source_Y[idx];
    //            source_data[sources*2+idx] = source_Z[idx];
    //            source_data[sources*3+idx] = source_lambda[idx];
    //            source_data[sources*4+idx] = source_I[idx];
    //        }
    //    }
    //}

    __syncthreads();

    for (int i_pix=tid; i_pix < Npix_to_model; i_pix+= thread_stride){

        if (using_trusted_mask){
            if (!data_trusted[i_pix])
                continue;
        }

        int _pid = panels_fasts_slows[i_pix*3];
        int _fpixel = panels_fasts_slows[i_pix*3+1];
        int _spixel = panels_fasts_slows[i_pix*3+2];

        CUDAREAL Fhkl_deriv_coef=0;
        CUDAREAL Fhkl_hessian_coef=0;
        if (s_Fhkl_gradient_mode){
            CUDAREAL u = data_residual[i_pix];
            CUDAREAL one_by_v = 1/data_variance[i_pix];
            CUDAREAL Gterm = 1 - 2*u - u*u*one_by_v;
            Fhkl_deriv_coef = 0.5 * Gterm*one_by_v / data_freq[i_pix];
            if (s_Fhkl_errors_mode){
                Fhkl_hessian_coef = -0.5*one_by_v*(one_by_v*Gterm - 2  - 2*u*one_by_v -u*u*one_by_v*one_by_v)/data_freq[i_pix];
            }
        }

        //int fcell_idx=1;
        int nom_h, nom_k, nom_l;
        if (s_use_nominal_hkl){
            nom_h = nominal_hkl[i_pix*3];
            nom_k = nominal_hkl[i_pix*3+1];
            nom_l = nominal_hkl[i_pix*3+2];
        }
        CUDAREAL close_distance = close_distances[_pid];

        // reset photon count for this pixel
        double _I=0;
        double Ilambda=0;
        double Imiller_h=0;
        double Imiller_k=0;
        double Imiller_l=0;

        // reset derivative photon counts for the various parameters
        double rot_manager_dI[3] = {0,0,0};
        double rot_manager_dI2[3] = {0,0,0};
        double ucell_manager_dI[6]= {0,0,0,0,0,0};
        double ucell_manager_dI2[6]= {0,0,0,0,0,0};
        double Ncells_manager_dI[6]= {0,0,0,0,0,0};
        double Ncells_manager_dI2[6]= {0,0,0,0,0,0};
        double pan_orig_manager_dI[3]= {0,0,0};
        double pan_orig_manager_dI2[3]= {0,0,0};
        double pan_rot_manager_dI[3]= {0,0,0};
        double pan_rot_manager_dI2[3]= {0,0,0};
        double fcell_manager_dI = 0;
        double fcell_manager_dI2 = 0;
        double eta_manager_dI[3] = {0,0,0};
        double eta_manager_dI2[3] = {0,0,0};
        double lambda_manager_dI[2] = {0,0};
        double lambda_manager_dI2[2] = {0,0};
        double fp_fdp_manager_dI[2] = {0,0};
        double dI_diffuse[6] = {0,0,0,0,0,0};

        for(int _subS=0;_subS<s_oversample;++_subS){
        for(int _subF=0;_subF<s_oversample;++_subF){

            // absolute mm position on detector (relative to its origin)
            CUDAREAL _Fdet = s_subpixel_size*(_fpixel*s_oversample + _subF ) + s_subpixel_size/2.0;
            CUDAREAL _Sdet = s_subpixel_size*(_spixel*s_oversample + _subS ) + s_subpixel_size/2.0;

            // assume "distance" is to the front of the detector sensor layer
            int pid_x = _pid*3;
            int pid_y = _pid*3+1;
            int pid_z = _pid*3+2;

            CUDAREAL fx = fdet_vectors[pid_x];
            CUDAREAL fy = fdet_vectors[pid_y];
            CUDAREAL fz = fdet_vectors[pid_z];
            CUDAREAL sx = sdet_vectors[pid_x];
            CUDAREAL sy = sdet_vectors[pid_y];
            CUDAREAL sz = sdet_vectors[pid_z];
            CUDAREAL ox = odet_vectors[pid_x];
            CUDAREAL oy = odet_vectors[pid_y];
            CUDAREAL oz = odet_vectors[pid_z];
            CUDAREAL px = pix0_vectors[pid_x];
            CUDAREAL py = pix0_vectors[pid_y];
            CUDAREAL pz = pix0_vectors[pid_z];

            VEC3 _o_vec(ox, oy, oz);

        for(int _thick_tic=0;_thick_tic<s_detector_thicksteps;++_thick_tic){

            CUDAREAL _Odet = _thick_tic*s_detector_thickstep;

            CUDAREAL pixposX = _Fdet*fx + _Sdet*sx + _Odet*ox + px;
            CUDAREAL pixposY = _Fdet*fy + _Sdet*sy + _Odet*oy + py;
            CUDAREAL pixposZ = _Fdet*fz + _Sdet*sz + _Odet*oz + pz;
            VEC3 _pixel_pos(pixposX, pixposY, pixposZ);

            CUDAREAL _airpath = _pixel_pos.norm();
            VEC3 _diffracted = _pixel_pos/_airpath;

            // solid angle subtended by a pixel: (pix/airpath)^2*cos(2theta)
            CUDAREAL _omega_pixel = s_pixel_size*s_pixel_size/_airpath/_airpath*close_distance/_airpath;

            // option to turn off obliquity effect, inverse-square-law only
            if(_point_pixel) _omega_pixel = 1.0/_airpath/_airpath;

            // now calculate detector thickness effects
            CUDAREAL _capture_fraction = 1;

            if(s_detector_thick > 0.0 && s_detector_attnlen > 0.0)
            {
                // inverse of effective thickness increase
                CUDAREAL _parallax = _diffracted.dot(_o_vec) ; //dot_product(diffracted,odet_vector);
                _capture_fraction = exp(-_thick_tic*s_detector_thickstep/s_detector_attnlen/_parallax)
                                  -exp(-(_thick_tic+1)*s_detector_thickstep/s_detector_attnlen/_parallax);
            }
            CUDAREAL cap_frac_times_omega = _capture_fraction * _omega_pixel;

        for(int _source=0;_source<s_sources;++_source){
            //VEC3 _incident(-__ldg(&source_X[_source]),
            //               -__ldg(&source_Y[_source]),
            //               -__ldg(&source_Z[_source]));
            //VEC3 _incident(-source_data[_source],
            //               -source_data[s_sources+_source],
            //               -source_data[2*s_sources+_source]);
            //CUDAREAL _lambda = source_data[3*s_sources+_source];
            //CUDAREAL sI = source_data[4*s_sources+_source];
            VEC3 _incident(-source_X[_source],
                           -source_Y[_source],
                           -source_Z[_source]);
            CUDAREAL _lambda = source_lambda[_source];
            CUDAREAL sI = source_I[_source];
            //CUDAREAL _lambda = __ldg(&source_lambda[_source]);
            CUDAREAL lambda_ang = _lambda*1e10;
            if (use_lambda_coefficients){
                lambda_ang = s_lambda0 + s_lambda1*lambda_ang;
                _lambda = lambda_ang*1e-10;
            }

            // polarization
            CUDAREAL polar_for_Fhkl_grad=1;
            if (!s_nopolar && s_Fhkl_gradient_mode){
                //polar_for_Fhkl_grad = diffBragg_gpu_kernel_polarization(_incident, _diffracted,
                //                    s_polarization_axis, s_kahn_factor);
                // component of diffracted unit vector along incident beam unit vector
                CUDAREAL cos2theta = _incident.dot(_diffracted);
                CUDAREAL cos2theta_sqr = cos2theta*cos2theta;
                CUDAREAL sin2theta_sqr = 1-cos2theta_sqr;

                CUDAREAL _psi=0;
                if(s_kahn_factor != 0.0){
                    // cross product to get "vertical" axis that is orthogonal to the cannonical "polarization"
                    VEC3 B_in = s_polarization_axis.cross(_incident);
                    // cross product with incident beam to get E-vector direction
                    VEC3 E_in = _incident.cross(B_in);
                    // get components of diffracted ray projected onto the E-B plane
                    CUDAREAL _kEi = _diffracted.dot(E_in);
                    CUDAREAL _kBi = _diffracted.dot(B_in);
                    // compute the angle of the diffracted ray projected onto the incident E-B plane
                    _psi = -atan2(_kBi,_kEi);
                }
                // correction for polarized incident beam
                polar_for_Fhkl_grad = 0.5*(1.0 + cos2theta_sqr - s_kahn_factor*cos(2*_psi)*sin2theta_sqr);
            }

            VEC3 _scattering = (_diffracted - _incident) / _lambda;

            VEC3 q_vec(_scattering[0], _scattering[1], _scattering[2]);
            q_vec *= 1e-10;

            // TODO rename
            CUDAREAL texture_scale= 1;
            texture_scale *= cap_frac_times_omega;
            texture_scale *= sI;


        for (int _phi_tic=0; _phi_tic<s_phisteps; ++_phi_tic){
            MAT3 Rphi;
            CUDAREAL phi = s_phi0 + s_phistep*_phi_tic;
            if (phi != 0){
                CUDAREAL c = cos(phi);
                CUDAREAL omc = 1-c;
                CUDAREAL s = sin(phi);
                Rphi << c + gx*gx*omc,    gx*gy*omc-gz*s,   gx*gz*omc+gy*s,
                      gy*gx*omc + gz*s,   c + gy*gy*omc,   gy*gz*omc - gx*s,
                      gz*gx*omc - gy*s,  gz*gy*omc + gx*s, c + gz*gz*omc;
            }

        for(int _mos_tic=0;_mos_tic<s_mosaic_domains;++_mos_tic){
            int amat_idx = _mos_tic;
            MAT3 UBO = Amatrices[amat_idx];
            if (phi != 0){
                MAT3 Um = UMATS_RXYZ[_mos_tic]; // note, this will be slow - check if we can simply allow Um and Rphi to commute ...
                UBO = UBO*Um*Rphi.transpose()*Um.transpose();
            }

            VEC3 H_vec = UBO*q_vec;
            CUDAREAL _h = H_vec[0];
            CUDAREAL _k = H_vec[1];
            CUDAREAL _l = H_vec[2];

            int _h0 = ceil(_h - 0.5);
            int _k0 = ceil(_k - 0.5);
            int _l0 = ceil(_l - 0.5);

            VEC3 H0(_h0, _k0, _l0);

            VEC3 delta_H = H_vec - H0;
            VEC3 V = _NABC*delta_H;
            CUDAREAL _hrad_sqr = V.dot(V);
            CUDAREAL I0=0;
            if (s_square_shape){
                I0 = 1;
                if(s_Na>1)
                    I0 *= sincg(M_PI*_h,s_Na);
                if(s_Nb>1)
                    I0 *= sincg(M_PI*_k,s_Nb);
                if(s_Nc>1)
                    I0 *= sincg(M_PI*_l,s_Nc);
                I0 *= I0;
            }
            else{
                CUDAREAL exparg;
                if (s_gaussian_star_shape){
                    MAT3 Ainv = UBO.inverse();
                    VEC3 delta_Q = Ainv*delta_H;
                    CUDAREAL rad_star_sqr = delta_Q.dot(delta_Q)*s_xtal_size_sq;
                    exparg = rad_star_sqr*1.9*fudge;
                }
                else
                    exparg = _hrad_sqr*C/2;
                if (exparg< 35)
                    if (s_no_Nabc_scale)
                        I0 = exp(-2*exparg);
                    else
                        I0 = (s_NABC_det_sq)*exp(-2*exparg);
            }

            // are we doing diffuse scattering
            CUDAREAL step_diffuse_param[6]  = {0,0,0,0,0,0};
            if (s_use_diffuse){
              calc_diffuse_at_hkl(H_vec,H0,dHH,Hmin,Hmax,Hrange,Ainv,&_FhklLinear[0],num_laue_mats,laue_mats,anisoG_local,anisoU_local,dG_dgam,s_refine_diffuse,&I0,step_diffuse_param);
            } // end s_use_diffuse outer

            CUDAREAL _F_cell = s_default_F;
            CUDAREAL _F_cell2 = 0;
            int i_hklasu=0;

            if ( (_h0<=s_h_max) && (_h0>=s_h_min) && (_k0<=s_k_max) && (_k0>=s_k_min) && (_l0<=s_l_max) && (_l0>=s_l_min)  ) {
                int Fhkl_linear_index = (_h0-s_h_min) * s_k_range * s_l_range + (_k0-s_k_min) * s_l_range + (_l0-s_l_min);
                //_F_cell = __ldg(&_FhklLinear[Fhkl_linear_index]);
                _F_cell = _FhklLinear[Fhkl_linear_index];
                //if (complex_miller) _F_cell2 = __ldg(&_Fhkl2Linear[Fhkl_linear_index]);
                if (s_complex_miller) _F_cell2 = _Fhkl2Linear[Fhkl_linear_index];
                if (s_Fhkl_have_scale_factors) i_hklasu = FhklLinear_ASUid[Fhkl_linear_index];
            }


            CUDAREAL c_deriv_Fcell = 0;
            CUDAREAL d_deriv_Fcell = 0;
            if (s_complex_miller){
                CUDAREAL c_deriv_Fcell_real = 0;
                CUDAREAL c_deriv_Fcell_imag = 0;
                CUDAREAL d_deriv_Fcell_real = 0;
                CUDAREAL d_deriv_Fcell_imag = 0;
                if(s_num_atoms > 0){
                   CUDAREAL S_2 = 1.e-20*(_scattering[0]*_scattering[0]+_scattering[1]*_scattering[1]+_scattering[2]*_scattering[2]);

                    // fp is always followed by the fdp value
                   CUDAREAL val_fp = fpfdp[2*_source];
                   CUDAREAL val_fdp = fpfdp[2*_source+1];

                   CUDAREAL c_deriv_prime=0;
                   CUDAREAL c_deriv_dblprime=0;
                   CUDAREAL d_deriv_prime = 0;
                   CUDAREAL d_deriv_dblprime = 0;
                   if (s_refine_fp_fdp){
                   //   currently only supports two parameter model
                       int d_idx = 2*_source;
                       c_deriv_prime = fpfdp_derivs[d_idx];
                       c_deriv_dblprime = fpfdp_derivs[d_idx+1];
                       d_deriv_prime = fpfdp_derivs[d_idx+2*s_sources];
                       d_deriv_dblprime = fpfdp_derivs[d_idx+1+2*s_sources];
                   }

                   for (int  i_atom=0; i_atom < s_num_atoms; i_atom++){
                        // fractional atomic coordinates
                       CUDAREAL atom_x = atom_data[i_atom*5];
                       CUDAREAL atom_y = atom_data[i_atom*5+1];
                       CUDAREAL atom_z = atom_data[i_atom*5+2];
                       CUDAREAL B = atom_data[i_atom*5+3]; // B factor
                       B = exp(-B*S_2/4.0); // TODO: speed me up?
                       CUDAREAL occ = atom_data[i_atom*5+4]; // occupancy
                       CUDAREAL r_dot_h = _h0*atom_x + _k0*atom_y + _l0*atom_z;
                       CUDAREAL phase = 2*M_PI*r_dot_h;
                       CUDAREAL s_rdoth = sin(phase);
                       CUDAREAL c_rdoth = cos(phase);
                       CUDAREAL Bocc = B*occ;
                       CUDAREAL BC = B*c_rdoth;
                       CUDAREAL BS = B*s_rdoth;
                       CUDAREAL real_part = BC*val_fp - BS*val_fdp;
                       CUDAREAL imag_part = BS*val_fp + BC*val_fdp;
                       _F_cell += real_part;
                       _F_cell2 += imag_part;
                       if (s_refine_fp_fdp){
                            c_deriv_Fcell_real += BC*c_deriv_prime - BS*c_deriv_dblprime;
                            c_deriv_Fcell_imag += BS*c_deriv_prime + BC*c_deriv_dblprime;

                            d_deriv_Fcell_real += BC*d_deriv_prime - BS*d_deriv_dblprime;
                            d_deriv_Fcell_imag += BS*d_deriv_prime + BC*d_deriv_dblprime;
                       }
                   }
               }
               CUDAREAL Freal = _F_cell;
               CUDAREAL Fimag = _F_cell2;

               _F_cell = sqrt(Freal*Freal + Fimag*Fimag);
               if (s_refine_fp_fdp){
                   c_deriv_Fcell = Freal*c_deriv_Fcell_real + Fimag*c_deriv_Fcell_imag;
                   d_deriv_Fcell = Freal*d_deriv_Fcell_real + Fimag*d_deriv_Fcell_imag;
               }

            }
            if (!s_oversample_omega && ! s_Fhkl_gradient_mode)
                _omega_pixel = 1;

            CUDAREAL _I_cell = _F_cell;
            if (! s_refine_Icell)
                _I_cell *= _F_cell;
            CUDAREAL s_hkl=1;
            int Fhkl_channel=0;
            if (! s_Fhkl_channels_empty)
                Fhkl_channel = Fhkl_channels[_source];
            if (s_Fhkl_have_scale_factors)
                s_hkl = Fhkl_scale[i_hklasu + Fhkl_channel*s_Num_ASU];
            if (s_Fhkl_gradient_mode){
                CUDAREAL Fhkl_deriv_scale = s_overall_scale*polar_for_Fhkl_grad;
                CUDAREAL I_noFcell=texture_scale*I0;
                CUDAREAL dfhkl = I_noFcell*_I_cell * Fhkl_deriv_scale;
                CUDAREAL grad_incr = dfhkl*Fhkl_deriv_coef;
                int fhkl_grad_idx=i_hklasu + Fhkl_channel*s_Num_ASU;

                if (s_Fhkl_errors_mode){
                    // here we hi-kack the Fhkl_scale_deriv array, if computing errors, in order to store the hessian terms
                    // if we are getting the hessian terms, we no longer need the  gradients (e.g. by this point we are done refininig)
                    CUDAREAL hessian_incr = Fhkl_hessian_coef*dfhkl*dfhkl;
                    atomicAdd(&Fhkl_scale_deriv[fhkl_grad_idx], hessian_incr);
                }
                else{
                    atomicAdd(&Fhkl_scale_deriv[fhkl_grad_idx], grad_incr);
                }
                continue;
            }

            CUDAREAL _I_total = s_hkl*_I_cell *I0;
            CUDAREAL Iincrement = _I_total*texture_scale;
            _I += Iincrement;
            if (save_wavelenimage){
                Ilambda += Iincrement*lambda_ang;
                Imiller_h += Iincrement*_h;
                Imiller_k += Iincrement*_k;
                Imiller_l += Iincrement*_l;
            }

            if (s_refine_diffuse){
                CUDAREAL step_scale = texture_scale*_F_cell*_F_cell;
                for (int i_diff=0; i_diff <6; i_diff++){
                    dI_diffuse[i_diff] += step_scale*step_diffuse_param[i_diff];
                }
            }


            if (s_refine_fp_fdp){
                CUDAREAL I_noFcell = texture_scale*I0;
                fp_fdp_manager_dI[0] += 2*I_noFcell * (c_deriv_Fcell);
                fp_fdp_manager_dI[1] += 2*I_noFcell * (d_deriv_Fcell);
            }

            if(s_verbose > 3)
                printf("hkl= %f %f %f  hkl1= %d %d %d  Fcell=%f\n", _h,_k,_l,_h0,_k0,_l0, _F_cell);

            MAT3 UBOt;
            if (s_refine_Umat[0] || s_refine_Umat[1] ||s_refine_Umat[2] || s_refine_eta){
                UBOt = Amat_init;
                if (phi != 0)
                    UBOt = Rphi*UBOt;
            }
            if (s_refine_Umat[0]){
                MAT3 RyRzUBOt = RotMats[1]*RotMats[2]*UBOt;
                VEC3 delta_H_prime = (UMATS[_mos_tic]*dRotMats[0]*RyRzUBOt).transpose()*q_vec;
                CUDAREAL V_dot_dV = V.dot(_NABC*delta_H_prime);
                CUDAREAL value = -two_C * V_dot_dV * Iincrement;
                CUDAREAL value2 =0;
                if (s_compute_curvatures) {
                    VEC3 delta_H_dbl_prime = (UMATS[_mos_tic]*d2RotMats[0]*RyRzUBOt).transpose()*q_vec;
                    CUDAREAL dV_dot_dV = (_NABC*delta_H_prime).dot(_NABC*delta_H_prime);
                    CUDAREAL dV2_dot_V = (_NABC*delta_H).dot(_NABC*delta_H_dbl_prime);
                    value2 = two_C*(two_C*V_dot_dV*V_dot_dV - dV2_dot_V - dV_dot_dV)*Iincrement;
                }
                rot_manager_dI[0] += value;
                rot_manager_dI2[0] += value2;
            }
            if (s_refine_Umat[1]){
                MAT3 UmosRx = UMATS[_mos_tic]*RotMats[0];
                MAT3 RzUBOt = RotMats[2]*UBOt;
                VEC3 delta_H_prime =(UmosRx*dRotMats[1]*RzUBOt).transpose()*q_vec;
                CUDAREAL V_dot_dV = V.dot(_NABC*delta_H_prime);
                CUDAREAL value = -two_C * V_dot_dV * Iincrement;

                CUDAREAL value2=0;
                if (s_compute_curvatures){
                    VEC3 delta_H_dbl_prime = (UmosRx*d2RotMats[1]*RzUBOt).transpose()*q_vec;
                    CUDAREAL dV_dot_dV = (_NABC*delta_H_prime).dot(_NABC*delta_H_prime);
                    CUDAREAL dV2_dot_V = (_NABC*delta_H).dot(_NABC*delta_H_dbl_prime);
                    value2 = two_C*(two_C*V_dot_dV*V_dot_dV - dV2_dot_V - dV_dot_dV)*Iincrement;
                }
                rot_manager_dI[1] += value;
                rot_manager_dI2[1] += value2;
            }
            if (s_refine_Umat[2]){
                MAT3 UmosRxRy = UMATS[_mos_tic]*RotMats[0]*RotMats[1];
                VEC3 delta_H_prime = (UmosRxRy*dRotMats[2]*UBOt).transpose()*q_vec;
                CUDAREAL V_dot_dV = V.dot(_NABC*delta_H_prime);
                CUDAREAL value = -two_C * V_dot_dV * Iincrement;

                CUDAREAL value2=0;
                if (s_compute_curvatures){
                    VEC3 delta_H_dbl_prime = (UmosRxRy*d2RotMats[2]*UBOt).transpose()*q_vec;
                    CUDAREAL dV_dot_dV = (_NABC*delta_H_prime).dot(_NABC*delta_H_prime);
                    CUDAREAL dV2_dot_V = (_NABC*delta_H).dot(_NABC*delta_H_dbl_prime);
                    value2 = two_C*(two_C*V_dot_dV*V_dot_dV - dV2_dot_V - dV_dot_dV)*Iincrement;
                }
                rot_manager_dI[2] += value;
                rot_manager_dI2[2] += value2;
            }
            //Checkpoint for unit cell derivatives
            //MAT3 Ot = eig_O.transpose();
            MAT3 UmosRxRyRzU;
            VEC3 delta_H_prime;
            for(int i_uc=0; i_uc < 6; i_uc++ ){
                if (s_refine_Bmat[i_uc]){
                    UmosRxRyRzU = UMATS_RXYZ[_mos_tic]*eig_U;
                    delta_H_prime = ((UmosRxRyRzU*(dB_mats[i_uc])*s_Ot).transpose()*q_vec);
                    CUDAREAL V_dot_dV = V.dot(_NABC*delta_H_prime);
                    CUDAREAL value = -two_C * V_dot_dV * Iincrement;
                    CUDAREAL value2 =0;
                    if (s_compute_curvatures){
                        VEC3 delta_H_dbl_prime = ((UmosRxRyRzU*(dB2_mats[i_uc])*s_Ot).transpose()*q_vec);
                        CUDAREAL dV_dot_dV = (_NABC*delta_H_prime).dot(_NABC*delta_H_prime);
                        CUDAREAL dV2_dot_V = (_NABC*delta_H).dot(_NABC*delta_H_dbl_prime);
                        value2 = two_C*(two_C*V_dot_dV*V_dot_dV - dV2_dot_V - dV_dot_dV)*Iincrement;
                    }
                    ucell_manager_dI[i_uc] += value;
                    ucell_manager_dI2[i_uc] += value2;
                }
            } //end ucell deriv

            // Checkpoint for Ncells manager
            if (s_refine_Ncells[0]){
                int num_ncell_deriv = 1;
                if (! isotropic_ncells)
                    num_ncell_deriv = 3;
                for (int i_nc=0; i_nc < num_ncell_deriv; i_nc++) {
                    MAT3 dN;
                    dN << 0,0,0,0,0,0,0,0,0;
                    dN(i_nc, i_nc) = 1;
                    if (num_ncell_deriv == 1){
                        dN(0,0) = 1;
                        dN(1,1) = 1;
                        dN(2,2) = 1;
                    }
                    CUDAREAL N_i = _NABC(i_nc, i_nc);
                    VEC3 dV_dN = dN*delta_H;
                    CUDAREAL determ_deriv = (_NABC.inverse()*dN).trace(); // TODO speedops: precompute these, store shared var _NABC.inverse
                    CUDAREAL deriv_coef= determ_deriv - C* ( dV_dN.dot(V));
                    CUDAREAL value = 2*Iincrement*deriv_coef;
                    CUDAREAL value2=0;
                    if(s_compute_curvatures){
                        dN(i_nc, i_nc) = 0; // TODO check maths
                        value2 = ( -1/N_i/N_i - C*(dV_dN.dot(dV_dN))) *2*Iincrement;
                        value2 += deriv_coef*2*value;
                    }
                    Ncells_manager_dI[i_nc] += value;
                    Ncells_manager_dI2[i_nc] += value2;
                }
            } // end Ncells manager deriv

            if (s_refine_Ncells_def){
                for (int i_nc =3; i_nc < 6; i_nc++ ){
                    MAT3 dN;
                    if (i_nc ==3)
                        dN << 0,1,0,1,0,0,0,0,0;
                    else if (i_nc == 4)
                        dN << 0,0,0,0,0,1,0,1,0;
                    else
                        dN << 0,0,1,0,0,0,1,0,0;
                    VEC3 dV_dN = dN*delta_H;
                    CUDAREAL determ_deriv = (_NABC.inverse()*dN).trace(); // TODO speedops: precompute these
                    CUDAREAL deriv_coef = determ_deriv - C* (dV_dN.dot(V));
                    CUDAREAL value = 2*Iincrement*deriv_coef;
                    Ncells_manager_dI[i_nc] += value;
                    CUDAREAL value2 = 0;
                    if (s_compute_curvatures){
                        value2 = deriv_coef*value;
                        value2 +=  -2*C*Iincrement*(dV_dN.dot(dV_dN));
                    Ncells_manager_dI2[i_nc] += value2;
                    }
                }
            }

            // Checkpoint for Origin manager
            for (int i_pan_orig=0; i_pan_orig < 3; i_pan_orig++){
                if (s_refine_panel_origin[i_pan_orig]){
                    CUDAREAL per_k = 1/_airpath;
                    CUDAREAL per_k3 = pow(per_k,3);
                    CUDAREAL per_k5 = pow(per_k,5);
                    CUDAREAL lambda_ang = _lambda*1e10;

                    MAT3 M = -two_C*(_NABC*UBO)/lambda_ang;
                    VEC3 dk;
                    if (i_pan_orig == 0)
                        dk << 0,0,1;
                    else if (i_pan_orig == 1)
                        dk << 1,0,0;
                    else
                        dk << 0,1,0;

                    CUDAREAL G = dk.dot(_pixel_pos);
                    CUDAREAL pix2 = subpixel_size*subpixel_size;
                    VEC3 dk_hat = -per_k3*G*_pixel_pos + per_k*dk;
                    CUDAREAL coef = (M*dk_hat).dot(V);
                    CUDAREAL coef2 = -3*pix2*per_k5*G * (_o_vec.dot(_pixel_pos));
                    coef2 += pix2*per_k3*(_o_vec.dot(dk));
                    CUDAREAL value = coef*Iincrement + coef2*Iincrement/_omega_pixel;

                    pan_orig_manager_dI[i_pan_orig] += value;
                    pan_orig_manager_dI2[i_pan_orig] += 0;

                } // end origin manager deriv
            }

            for (int i_pan_rot=0; i_pan_rot < 3; i_pan_rot++){
                if(s_refine_panel_rot[i_pan_rot]){
                    CUDAREAL per_k = 1/_airpath;
                    CUDAREAL per_k3 = pow(per_k,3);
                    CUDAREAL per_k5 = pow(per_k,5);
                    CUDAREAL lambda_ang = _lambda*1e10;
                    MAT3 M = -two_C*(_NABC*UBO)/lambda_ang;
                    VEC3 dk = _Fdet*(dF_vecs[_pid*3 + i_pan_rot]) + _Sdet*(dS_vecs[_pid*3 + i_pan_rot]);
                    CUDAREAL G = dk.dot(_pixel_pos);
                    CUDAREAL pix2 = subpixel_size*subpixel_size;
                    VEC3 dk_hat = -per_k3*G*_pixel_pos + per_k*dk;
                    CUDAREAL coef = (M*dk_hat).dot(V);
                    CUDAREAL coef2 = -3*pix2*per_k5*G * (_o_vec.dot(_pixel_pos));
                    coef2 += pix2*per_k3*(_o_vec.dot(dk));
                    CUDAREAL value = coef*Iincrement + coef2*Iincrement/_omega_pixel;

                    pan_rot_manager_dI[i_pan_rot] += value;
                    pan_rot_manager_dI2[i_pan_rot] += 0;
                }
            }

            // checkpoint for Fcell manager
            if (s_refine_fcell){
                CUDAREAL value;
                if (s_refine_Icell)
                    value = I0* texture_scale;
                else
                    value = 2*I0*_F_cell * texture_scale; //Iincrement/_F_cell ;
                CUDAREAL value2=0;
                if (s_compute_curvatures){
                //    NOTE if _Fcell >0
                    value2 = 2*I0 * texture_scale;
                }
                //if (fcell_idx >=0 && fcell_idx <=2){
                if (s_use_nominal_hkl){
                    if (_h0==nom_h && _k0==nom_k && _l0==nom_l){
                        fcell_manager_dI += value;
                        fcell_manager_dI2 += value2;
                    }
                }
                else{
                    fcell_manager_dI += value;
                    fcell_manager_dI2 += value2;

                }
            } // end of fcell man deriv

            // checkpoint for eta manager
            if (s_refine_eta){
                for (int i_eta=0;i_eta<3; i_eta++){
                    if (i_eta > 0 && ! s_aniso_eta)
                        continue;
                    int mtic2 = _mos_tic + i_eta*s_mosaic_domains;
                    VEC3 DeltaH_deriv = (UMATS_RXYZ_prime[mtic2]*UBOt).transpose()*q_vec;
                    // vector V is _Nabc*Delta_H
                    VEC3 dV = _NABC*DeltaH_deriv;
                    CUDAREAL V_dot_dV = V.dot(dV);
                    CUDAREAL Iprime = -two_C*(V_dot_dV)*Iincrement;
                    eta_manager_dI[i_eta] += Iprime;
                    CUDAREAL Idbl_prime=0;
                    if (s_compute_curvatures){
                        VEC3 DeltaH_second_deriv = (UMATS_RXYZ_dbl_prime[mtic2]*UBOt).transpose()*q_vec;
                        VEC3 dV2 = _NABC*DeltaH_second_deriv;
                        Idbl_prime = -two_C*(dV.dot(dV) + V.dot(dV2))*Iincrement;
                        Idbl_prime += -two_C*(V_dot_dV)*Iprime;
                    }
                    eta_manager_dI2[i_eta] += Idbl_prime;
                }
            } // end of eta man deriv

            // checkpoint for lambda manager
            for(int i_lam=0; i_lam < 2; i_lam++){
                if (s_refine_lambda[i_lam]){
                    CUDAREAL lambda_ang = _lambda*1e10;
                    CUDAREAL NH_dot_V = (_NABC*H_vec).dot(V);
                    CUDAREAL dg_dlambda;
                    if (i_lam==0)
                        dg_dlambda = 1;
                    else // i_lam==1
                        dg_dlambda = lambda_ang;
                    CUDAREAL coef = NH_dot_V*two_C*(dg_dlambda) / lambda_ang;
                    CUDAREAL value = coef*Iincrement;
                    CUDAREAL value2 = 0;
                    lambda_manager_dI[i_lam] += value;
                    lambda_manager_dI2[i_lam] += value2;
                }
            }
            //end of lambda deriv
            if( s_printout){
             if( _subS==0 && _subF==0 && _thick_tic==0 && _source==0 &&  _mos_tic==0 ){
              if((_fpixel==s_printout_fpixel && _spixel==s_printout_spixel) || s_printout_fpixel < 0){
                   printf("%4d %4d :  lambda = %10.9g\n", _fpixel,_spixel, _lambda);
                   printf("at %g %g %g\n", _pixel_pos[0],_pixel_pos[1],_pixel_pos[2]);
                   printf("Fdet= %10.7g; Sdet= %10.7g ; Odet= %10.7g\n", _Fdet, _Sdet, _Odet);
                   printf("PIX0: %10.5g %10.5g %10.5g\n" , pix0_vectors[pid_x], pix0_vectors[pid_y], pix0_vectors[pid_z]);
                   printf("F: %10.5g %10.5g %10.5g\n" , fdet_vectors[pid_x], fdet_vectors[pid_y], fdet_vectors[pid_z]);
                   printf("S: %10.5g %10.5g %10.5g\n" , sdet_vectors[pid_x], sdet_vectors[pid_y], sdet_vectors[pid_z]);
                   printf("O: %10.5g %10.5g %10.5g\n" , odet_vectors[pid_x], odet_vectors[pid_y], odet_vectors[pid_z]);
                   printf("pid_x=%d, pid_y=%d; pid_z=%d\n", pid_x, pid_y, pid_z);

                   printf("QVECTOR: %10.5g %10.5g %10.5g\n" , q_vec[0], q_vec[1], q_vec[2]);
                   MAT3 UU = UMATS_RXYZ[_mos_tic];
                     printf("UMAT_RXYZ :\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n",
                      UU(0,0),  UU(0,1), UU(0,2),
                      UU(1,0),  UU(1,1), UU(1,2),
                      UU(2,0),  UU(2,1), UU(2,2));
                   UU = Bmat_realspace;
                     printf("Bmat_realspace :\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n",
                      UU(0,0),  UU(0,1), UU(0,2),
                      UU(1,0),  UU(1,1), UU(1,2),
                      UU(2,0),  UU(2,1), UU(2,2));
                   UU = UBO;
                     printf("UBO :\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n",
                      UU(0,0),  UU(0,1), UU(0,2),
                      UU(1,0),  UU(1,1), UU(1,2),
                      UU(2,0),  UU(2,1), UU(2,2));

                   UU = UBOt;
                     printf("UBOt :\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n",
                      UU(0,0),  UU(0,1), UU(0,2),
                      UU(1,0),  UU(1,1), UU(1,2),
                      UU(2,0),  UU(2,1), UU(2,2));

                   UU = UmosRxRyRzU;
                    printf("UmosRxRyRzU :\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n",
                        UU(0,0),  UU(0,1), UU(0,2),
                        UU(1,0),  UU(1,1), UU(1,2),
                        UU(2,0),  UU(2,1), UU(2,2));
                   VEC3 AA = delta_H_prime;
                   printf("delta_H_prime :\n%f  %f  %f\n",
                        AA[0],  AA[1], AA[2]);
                   printf("Iincrement: %f\n", Iincrement);
                   printf("hkl= %f %f %f  hkl0= %d %d %d\n", _h,_k,_l,_h0,_k0,_l0);
                   printf(" F_cell=%g  F_cell2=%g I_latt=%g   I = %g\n", _F_cell,_F_cell2,I0,_I);
                   printf("I/steps %15.10g\n", _I/s_Nsteps);
                   //printf("Ilatt diffuse %15.10g\n", I_latt_diffuse);
                   printf("omega   %15.10g\n", _omega_pixel);
                   printf("default_F= %f\n", s_default_F);
                   printf("Incident[0]=%15.10g, Incident[1]=%15.10g, Incident[2]=%15.10g\n", _incident[0], _incident[1], _incident[2]);
                   printf("Diffracted[0]=%15.10g, Diffracted[1]=%15.10g, Diffracted[2]=%15.10g\n", _diffracted[0], _diffracted[1], _diffracted[2]);
                   printf("Scattering[0]=%15.10g, Scattering[1]=%15.10g, Scattering[2]=%15.10g\n", _scattering[0], _scattering[1], _scattering[2]);
                   printf("sourceI=%10.7g\n",  sI);
                  if (s_complex_miller)printf("COMPLEX MILLER!\n");
                  if (s_no_Nabc_scale)printf("No Nabc scale!\n");
                }
              }
            }

            } // end of mos_tic loop
            } // end of phi_tic loop
           } // end of source loop
          } // end of thick step loop
         } // end of fpos loop
        } // end of spos loop
        if (s_Fhkl_gradient_mode)
            continue;

        CUDAREAL _Fdet_ave = s_pixel_size*_fpixel + s_pixel_size/2.0;
        CUDAREAL _Sdet_ave = s_pixel_size*_spixel + s_pixel_size/2.0;
        CUDAREAL _Odet_ave = 0; //Odet; // TODO maybe make this more general for thick detectors?

        VEC3 _pixel_pos_ave(0,0,0);
        int pid_x = _pid*3;
        int pid_y = _pid*3+1;
        int pid_z = _pid*3+2;

        CUDAREAL fx = fdet_vectors[pid_x];
        CUDAREAL fy = fdet_vectors[pid_y];
        CUDAREAL fz = fdet_vectors[pid_z];

        CUDAREAL sx = sdet_vectors[pid_x];
        CUDAREAL sy = sdet_vectors[pid_y];
        CUDAREAL sz = sdet_vectors[pid_z];

        CUDAREAL ox = odet_vectors[pid_x];
        CUDAREAL oy = odet_vectors[pid_y];
        CUDAREAL oz = odet_vectors[pid_z];

        CUDAREAL px =pix0_vectors[pid_x];
        CUDAREAL py =pix0_vectors[pid_y];
        CUDAREAL pz =pix0_vectors[pid_z];

        _pixel_pos_ave[0] = _Fdet_ave * fx+_Sdet_ave*sx+_Odet_ave*ox+px;
        _pixel_pos_ave[1] = _Fdet_ave * fy+_Sdet_ave*sy+_Odet_ave*oy+py;
        _pixel_pos_ave[2] = _Fdet_ave * fz+_Sdet_ave*sz+_Odet_ave*oz+pz;

        CUDAREAL _airpath_ave = _pixel_pos_ave.norm();
        VEC3 _diffracted_ave = _pixel_pos_ave/_airpath_ave;
        CUDAREAL _omega_pixel_ave = s_pixel_size*s_pixel_size/_airpath_ave/_airpath_ave*close_distance/_airpath_ave;

        CUDAREAL _polar = 1;
        if (!s_nopolar){
            VEC3 _incident(-sX0, -sY0, -sZ0);
            _incident = _incident / _incident.norm();
            // component of diffracted unit vector along incident beam unit vector
            CUDAREAL cos2theta = _incident.dot(_diffracted_ave);
            CUDAREAL cos2theta_sqr = cos2theta*cos2theta;
            CUDAREAL sin2theta_sqr = 1-cos2theta_sqr;

            CUDAREAL _psi=0;
            if(kahn_factor != 0.0){
                // cross product to get "vertical" axis that is orthogonal to the cannonical "polarization"
                VEC3 B_in = s_polarization_axis.cross(_incident);
                // cross product with incident beam to get E-vector direction
                VEC3 E_in = _incident.cross(B_in);
                // get components of diffracted ray projected onto the E-B plane
                CUDAREAL _kEi = _diffracted_ave.dot(E_in);
                CUDAREAL _kBi = _diffracted_ave.dot(B_in);
                // compute the angle of the diffracted ray projected onto the incident E-B plane
                _psi = -atan2(_kBi,_kEi);
            }
            // correction for polarized incident beam
            _polar = 0.5*(1.0 + cos2theta_sqr - s_kahn_factor*cos(2*_psi)*sin2theta_sqr);
        }

        CUDAREAL _om = 1;
        if (!s_oversample_omega)
            _om=_omega_pixel_ave;
        // final scale term to being everything to photon number units
        CUDAREAL _scale_term = _polar*_om * s_overall_scale;
        floatimage[i_pix] = _scale_term*_I;
        if (save_wavelenimage){
            wavelenimage[i_pix*4] = Ilambda / _I;
            wavelenimage[i_pix*4+1] = Imiller_h / _I;
            wavelenimage[i_pix*4+2] = Imiller_k / _I;
            wavelenimage[i_pix*4+3] = Imiller_l / _I;
        }

        // udpate the rotation derivative images*
        for (int i_rot =0 ; i_rot < 3 ; i_rot++){
            if (s_refine_Umat[i_rot]){
                CUDAREAL value = _scale_term*rot_manager_dI[i_rot];
                CUDAREAL value2 = _scale_term*rot_manager_dI2[i_rot];
                int idx = i_rot*Npix_to_model + i_pix;
                d_Umat_images[idx] = value;
                d2_Umat_images[idx] = value2;
            }
        } // end rot deriv image increment

        //update the ucell derivative images
        for (int i_uc=0 ; i_uc < 6 ; i_uc++){
            if (s_refine_Bmat[i_uc]){
                CUDAREAL value = _scale_term*ucell_manager_dI[i_uc];
                CUDAREAL value2 = _scale_term*ucell_manager_dI2[i_uc];
                int idx= i_uc*Npix_to_model + i_pix;
                d_Bmat_images[idx] = value;
                d2_Bmat_images[idx] = value2;
            }
        }// end ucell deriv image increment

        //update the Ncells derivative image
        if (s_refine_Ncells[0]){
            CUDAREAL value = _scale_term*Ncells_manager_dI[0];
            CUDAREAL value2 = _scale_term*Ncells_manager_dI2[0];
            int idx = i_pix;
            d_Ncells_images[idx] = value;
            d2_Ncells_images[idx] = value2;

            if (! isotropic_ncells){
                value = _scale_term*Ncells_manager_dI[1];
                value2 = _scale_term*Ncells_manager_dI2[1];
                idx = Npix_to_model + i_pix;
                d_Ncells_images[idx] = value;
                d2_Ncells_images[idx] = value2;

                value = _scale_term*Ncells_manager_dI[2];
                value2 = _scale_term*Ncells_manager_dI2[2];
                idx = Npix_to_model*2 + i_pix;
                d_Ncells_images[idx] = value;
                d2_Ncells_images[idx] = value2;
            }
        }// end Ncells deriv image increment
        if (s_refine_Ncells_def){
            for (int i_nc=3; i_nc<6; i_nc++){
                CUDAREAL value = _scale_term*Ncells_manager_dI[i_nc];
                CUDAREAL value2 = _scale_term*Ncells_manager_dI2[i_nc];
                int idx = i_nc* Npix_to_model + i_pix;
                d_Ncells_images[idx] = value;
                d2_Ncells_images[idx] = value2;
            }
        }

        // update Fcell derivative image
        if(s_refine_fcell){
            CUDAREAL value = _scale_term*fcell_manager_dI;
            CUDAREAL value2 = _scale_term*fcell_manager_dI2;
            d_fcell_images[i_pix] = value;
            d2_fcell_images[i_pix] = value2;
        }// end Fcell deriv image increment

        if (s_refine_fp_fdp){
            // c derivative
            CUDAREAL value = _scale_term*fp_fdp_manager_dI[0];
            d_fp_fdp_images[i_pix] = value;
            // d derivative
            value = _scale_term*fp_fdp_manager_dI[1];
            d_fp_fdp_images[Npix_to_model + i_pix] = value;
        }
        if (s_refine_diffuse){
            for (int i_gam=0; i_gam < 3; i_gam++){
                CUDAREAL val = dI_diffuse[i_gam]*_scale_term;
                int img_idx = Npix_to_model*i_gam + i_pix;
                d_diffuse_gamma_images[img_idx] = val;
            }
            for (int i_sig=0; i_sig < 3; i_sig++){
                CUDAREAL val = dI_diffuse[i_sig+3]*_scale_term;
                int img_idx = Npix_to_model*i_sig + i_pix;
                d_diffuse_sigma_images[img_idx] = val;
            }
        }

        // update eta derivative image
        if(s_refine_eta){
            for (int i_eta=0; i_eta<3; i_eta++){
                if (i_eta > 0 && ! s_aniso_eta)
                    continue;
                int idx = i_pix + Npix_to_model*i_eta;
                CUDAREAL value = _scale_term*eta_manager_dI[i_eta];
                CUDAREAL value2 = _scale_term*eta_manager_dI2[i_eta];
                d_eta_images[idx] = value;
                d2_eta_images[idx] = value2;
            }
        }// end eta deriv image increment

        //update the lambda derivative images
        for (int i_lam=0 ; i_lam < 2 ; i_lam++){
            if (s_refine_lambda[i_lam]){
                CUDAREAL value = _scale_term*lambda_manager_dI[i_lam];
                CUDAREAL value2 = _scale_term*lambda_manager_dI2[i_lam];
                int idx = i_lam*Npix_to_model + i_pix;
                d_lambda_images[idx] = value;
                //d2_lambda_images[idx] = value2;
            }
        }// end lambda deriv image increment

        for (int i_pan_rot=0; i_pan_rot < 3; i_pan_rot++){
            if(s_refine_panel_rot[i_pan_rot]){
                CUDAREAL value = _scale_term*pan_rot_manager_dI[i_pan_rot];
                CUDAREAL value2 = _scale_term*pan_rot_manager_dI2[i_pan_rot];
                int idx = i_pan_rot*Npix_to_model + i_pix;
                d_panel_rot_images[idx] = value;
                //d2_panel_rot_images[idx] = value2;
            }
        }// end panel rot deriv image increment

        for (int i_pan_orig=0; i_pan_orig < 3; i_pan_orig++){
            if(s_refine_panel_origin[i_pan_orig]){
                CUDAREAL value = _scale_term*pan_orig_manager_dI[i_pan_orig];
                CUDAREAL value2 = _scale_term*pan_orig_manager_dI2[i_pan_orig];
                int idx = i_pan_orig*Npix_to_model + i_pix;
                d_panel_orig_images[idx] = value;
                //d2_panel_orig_images[idx] = value2;
            }
        }//end panel orig deriv image increment

    } // end i_pix loop

}  // END of GPU kernel
