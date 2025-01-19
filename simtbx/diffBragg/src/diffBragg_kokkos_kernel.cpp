#include <cstdio>
#include "diffBraggKOKKOS.h"
#include <simtbx/kokkos/kernel_math.h>
#include <simtbx/diffBragg/src/diffuse_util_kokkos.h>

void kokkos_sum_over_steps(
    int Npix_to_model,
    vector_uint_t panels_fasts_slows,
    vector_cudareal_t floatimage,
    vector_cudareal_t wavelenimage,
    vector_cudareal_t d_Umat_images,
    vector_cudareal_t d2_Umat_images,
    vector_cudareal_t d_Bmat_images,
    vector_cudareal_t d2_Bmat_images,
    vector_cudareal_t d_Ncells_images,
    vector_cudareal_t d2_Ncells_images,
    vector_cudareal_t d_fcell_images,
    vector_cudareal_t d2_fcell_images,
    vector_cudareal_t d_eta_images,
    vector_cudareal_t d2_eta_images,
    vector_cudareal_t d_lambda_images,
    vector_cudareal_t d2_lambda_images,
    vector_cudareal_t d_panel_rot_images,
    vector_cudareal_t d2_panel_rot_images,
    vector_cudareal_t d_panel_orig_images,
    vector_cudareal_t d2_panel_orig_images,
    vector_cudareal_t d_fp_fdp_images,
    vector_manager_t manager_dI,
    vector_manager_t manager_dI2,
    const int Nsteps,
    int printout_fpixel,
    int printout_spixel,
    bool printout,
    CUDAREAL default_F,
    int oversample,
    bool oversample_omega,
    CUDAREAL subpixel_size,
    CUDAREAL pixel_size,
    CUDAREAL detector_thickstep,
    CUDAREAL detector_thick,
    const vector_cudareal_t close_distances,
    CUDAREAL detector_attnlen,
    int detector_thicksteps,
    int sources,
    int phisteps,
    int mosaic_domains,
    bool use_lambda_coefficients,
    CUDAREAL lambda0,
    CUDAREAL lambda1,
    KOKKOS_MAT3 eig_U,
    KOKKOS_MAT3 eig_O,
    KOKKOS_MAT3 eig_B,
    KOKKOS_MAT3 RXYZ,
    vector_vec3_t dF_vecs,
    vector_vec3_t dS_vecs,
    const vector_mat3_t UMATS_RXYZ,
    vector_mat3_t UMATS_RXYZ_prime,
    vector_mat3_t UMATS_RXYZ_dbl_prime,
    vector_mat3_t RotMats,
    vector_mat3_t dRotMats,
    vector_mat3_t d2RotMats,
    vector_mat3_t UMATS,
    vector_mat3_t dB_mats,
    vector_mat3_t dB2_mats,
    vector_mat3_t Amatrices,
    const vector_cudareal_t source_X,
    const vector_cudareal_t source_Y,
    const vector_cudareal_t source_Z,
    const vector_cudareal_t source_lambda,
    const vector_cudareal_t source_I,
    CUDAREAL kahn_factor,
    CUDAREAL Na,
    CUDAREAL Nb,
    CUDAREAL Nc,
    CUDAREAL Nd,
    CUDAREAL Ne,
    CUDAREAL Nf,
    CUDAREAL phi0,
    CUDAREAL phistep,
    KOKKOS_VEC3 spindle_vec,
    KOKKOS_VEC3 polarization_axis,
    int h_range,
    int k_range,
    int l_range,
    int h_max,
    int h_min,
    int k_max,
    int k_min,
    int l_max,
    int l_min,
    CUDAREAL dmin,
    CUDAREAL fudge,
    bool complex_miller,
    int verbose,
    bool only_save_omega_kahn,
    bool isotropic_ncells,
    bool compute_curvatures,
    const vector_cudareal_t FhklLinear,
    const vector_cudareal_t Fhkl2Linear,
    const uint32_t refine_flag,
    // vector_bool_t refine_Bmat,
    // vector_bool_t refine_Ncells,
    // bool refine_Ncells_def,
    // vector_bool_t refine_panel_origin,
    // vector_bool_t refine_panel_rot,
    // bool refine_fcell,
    // vector_bool_t refine_lambda,
    // bool refine_eta,
    // vector_bool_t refine_Umat,
    const vector_cudareal_t fdet_vectors,
    const vector_cudareal_t sdet_vectors,
    const vector_cudareal_t odet_vectors,
    const vector_cudareal_t pix0_vectors,
    bool nopolar,
    bool point_pixel,
    CUDAREAL fluence,
    CUDAREAL r_e_sqr,
    CUDAREAL spot_scale,
    int Npanels,
    bool aniso_eta,
    bool no_Nabc_scale,
    const vector_cudareal_t fpfdp,
    const vector_cudareal_t fpfdp_derivs,
    const vector_cudareal_t atom_data,
    int num_atoms,
    // bool refine_fp_fdp,
    const vector_int_t nominal_hkl,
    bool use_nominal_hkl,
    KOKKOS_MAT3 anisoU,
    KOKKOS_MAT3 anisoG,
    KOKKOS_MAT3 rotate_principal_axes,
    bool use_diffuse,
    vector_cudareal_t d_diffuse_gamma_images,
    vector_cudareal_t d_diffuse_sigma_images,
    // bool refine_diffuse,
    bool gamma_miller_units,
    // bool refine_Icell,
    bool save_wavelenimage,
    int laue_group_num,
    int stencil_size,
    bool Fhkl_gradient_mode,
    bool Fhkl_errors_mode,
    bool using_trusted_mask,
    bool Fhkl_channels_empty,
    bool Fhkl_have_scale_factors,
    int Num_ASU,
    const vector_cudareal_t data_residual,
    const vector_cudareal_t data_variance,
    const vector_int_t data_freq,
    const vector_bool_t data_trusted,
    const vector_int_t FhklLinear_ASUid,
    const vector_int_t Fhkl_channels,
    const vector_cudareal_t Fhkl_scale,
    vector_cudareal_t Fhkl_scale_deriv,
    bool gaussian_star_shape, bool square_shape
    ) {  // BEGIN GPU kernel

    const KOKKOS_MAT3 Bmat_realspace = eig_B * 1e10;
    const KOKKOS_MAT3 eig_Otranspose = eig_O.transpose();
    const KOKKOS_MAT3 Amat_init = eig_U * Bmat_realspace * eig_Otranspose;
    const KOKKOS_MAT3 Ainv = eig_U*(Bmat_realspace.transpose().inverse())* (eig_O.inverse());
    const CUDAREAL reciprocal_space_volume = 8*M_PI*M_PI*M_PI*Ainv.determinant();
    const KOKKOS_MAT3 _NABC {Na, Nd, Nf, Nd, Nb, Ne, Nf, Ne, Nc};
    const double NABC_det = _NABC.determinant();  // TODO is this slow ?
    const double NABC_det_sq = NABC_det * NABC_det;
    const CUDAREAL C = 2 / 0.63 * fudge;
    const CUDAREAL two_C = 2 * C;
    KOKKOS_MAT3 anisoG_local;
    CUDAREAL anisoG_determ = 0;
    KOKKOS_MAT3 anisoU_local;
    const CUDAREAL _tmpfac = M_PI * 0.63 / fudge;
    const CUDAREAL diffuse_scale = reciprocal_space_volume * sqrt(_tmpfac*_tmpfac*_tmpfac);
    // ***NEEDS UPDATE: use legacy API for passing diffuse scale as KOKKOS_MAT3
    KOKKOS_MAT3 diffuse_scale_mat3(diffuse_scale,0,0,0,0,0,0,0,0);
    vector_mat3_t laue_mats = vector_mat3_t("laue_mats", 24);
    vector_vec3_t dG_dgam = vector_vec3_t("dG_dgam", 3);
    vector_cudareal_t dG_trace = vector_cudareal_t("dG_trace", 3);
    int num_laue_mats = 0;
    int dhh = 0, dkk = 0, dll = 0;
    CUDAREAL gx = spindle_vec[0];
    CUDAREAL gy = spindle_vec[1];
    CUDAREAL gz = spindle_vec[2];

    Kokkos::View<KOKKOS_MAT3*[3]> UMATS_prime("UMATS_prime", mosaic_domains);
    Kokkos::View<KOKKOS_MAT3*[3]> UMATS_dbl_prime("UMATS_dbl_prime", mosaic_domains);
    Kokkos::View<KOKKOS_MAT3*[6]> BMATS_prime("BMATS_prime", mosaic_domains);
    Kokkos::View<KOKKOS_MAT3*[6]> BMATS_dbl_prime("BMATS_dbl_prime", mosaic_domains);

    Kokkos::parallel_for("prepare_UMATS", mosaic_domains, KOKKOS_LAMBDA(const int& _mos_tic) {
        const KOKKOS_MAT3 UBOt = Amat_init;
        UMATS_prime(_mos_tic, 0) = _NABC * (UMATS(_mos_tic) * dRotMats(0) * RotMats(1) * RotMats(2) * UBOt).transpose();
        UMATS_prime(_mos_tic, 1) = _NABC * (UMATS(_mos_tic) * RotMats(0) * dRotMats(1) * RotMats(2) * UBOt).transpose();
        UMATS_prime(_mos_tic, 2) = _NABC * (UMATS(_mos_tic) * RotMats(0) * RotMats(1) * dRotMats(2) * UBOt).transpose();

        UMATS_dbl_prime(_mos_tic, 0) = _NABC * (UMATS(_mos_tic) * d2RotMats(0) * RotMats(1) * RotMats(2) * UBOt).transpose();
        UMATS_dbl_prime(_mos_tic, 1) = _NABC * (UMATS(_mos_tic) * RotMats(0) * d2RotMats(1) * RotMats(2) * UBOt).transpose();
        UMATS_dbl_prime(_mos_tic, 2) = _NABC * (UMATS(_mos_tic) * RotMats(0) * RotMats(1) * d2RotMats(2) * UBOt).transpose();

        for (int i_uc=0; i_uc<6; i_uc++) {
            BMATS_prime(_mos_tic, i_uc) = _NABC * (UMATS_RXYZ(_mos_tic) * eig_U * dB_mats(i_uc) * eig_O.transpose()).transpose();
            BMATS_dbl_prime(_mos_tic, i_uc) = _NABC * (UMATS_RXYZ(_mos_tic) * eig_U * dB2_mats(i_uc) * eig_O.transpose()).transpose();
        }
    });

    if (use_diffuse){
        anisoG_local = anisoG;
        anisoU_local = anisoU;

        if (laue_group_num < 1 || laue_group_num >14 ){
            throw std::string("Laue group number not in range 1-14");
        }

        if (gamma_miller_units){
            anisoG_local = anisoG_local * Bmat_realspace;
        }
        Kokkos::parallel_reduce("prepare diffuse mats", 1, KOKKOS_LAMBDA (const int& i, int& num_laue_mats_temp){
            num_laue_mats_temp = gen_laue_mats(laue_group_num, laue_mats, rotate_principal_axes);
            // KOKKOS_MAT3 rotate_principal_axes;
            // rotate_principal_axes << 0.70710678,  -0.70710678,  0., 0.70710678,  0.70710678,  0., 0.,  0., 1.;

            for ( int iL = 0; iL < num_laue_mats_temp; iL++ ){
                laue_mats(iL) = Ainv * laue_mats(iL);
            }
            // printf("Bmat =");
            // for (int i=0; i<9; ++i) {
            //     printf(" %g", Bmat_realspace[i]);
            // }
            // printf("\n");
            const KOKKOS_MAT3 Ginv = anisoG_local.inverse();
            // printf("Ginv =");
            // for (int i=0; i<9; ++i) {
            //     printf(" %g", Ginv[i]);
            // }
            // printf("\n");
            const KOKKOS_MAT3 dG = Bmat_realspace * Ginv;
            // printf("dG   =");
            // for (int i=0; i<9; ++i) {
            //     printf(" %g", dG[i]);
            // }
            // printf("\n");
            for (int i_gam=0; i_gam<3; i_gam++){
                if (gamma_miller_units) {
                    dG_dgam(i_gam) = KOKKOS_VEC3(Bmat_realspace(i_gam, 0), Bmat_realspace(i_gam, 1), Bmat_realspace(i_gam, 2));
                } else {
                    dG_dgam(i_gam)[i_gam] = 1;
                }
                KOKKOS_MAT3 temp_dgam;
                temp_dgam(i_gam, 0) = dG_dgam(i_gam)[0];
                temp_dgam(i_gam, 1) = dG_dgam(i_gam)[1];
                temp_dgam(i_gam, 2) = dG_dgam(i_gam)[2];
                dG_trace(i_gam) = (Ginv*temp_dgam).trace();
                // printf("TRACE %g\n", dG_trace(i_gam));
                // printf("dgam =");
                // for (int i=0; i<9; ++i) {
                //     printf(" %g", temp_dgam[i]);
                // }
                // printf("\n");

                // dG(i_gam, i_gam);
            }
        }, num_laue_mats);
        anisoG_determ = anisoG_local.determinant();
        dhh = dkk = dll = stencil_size; // Limits of stencil for diffuse calc
    }
    const KOKKOS_VEC3 dHH (dhh, dkk, dll);

    const CUDAREAL overall_scale = r_e_sqr * spot_scale * fluence / Nsteps;

    const CUDAREAL detector_attnlen_r = (detector_attnlen>0) ? 1 / detector_attnlen : 0;

    Kokkos::parallel_for(
        "sum_over_steps", Npix_to_model, KOKKOS_LAMBDA(const int& pixIdx) {

        if (using_trusted_mask) {
            if (!data_trusted(pixIdx))
                return;
        }
        const int _pid = panels_fasts_slows(pixIdx * 3);
        const int _fpixel = panels_fasts_slows(pixIdx * 3 + 1);
        const int _spixel = panels_fasts_slows(pixIdx * 3 + 2);

        CUDAREAL Fhkl_deriv_coef=0;
        CUDAREAL Fhkl_hessian_coef=0;
        if (Fhkl_gradient_mode) {
            CUDAREAL u = data_residual(pixIdx);
            CUDAREAL one_by_v = 1/data_variance(pixIdx);
            CUDAREAL Gterm = 1 - 2*u - u*u*one_by_v;
            Fhkl_deriv_coef = 0.5 * Gterm*one_by_v / data_freq(pixIdx);
            if (Fhkl_errors_mode) {
                Fhkl_hessian_coef = -0.5*one_by_v*(one_by_v*Gterm - 2  - 2*u*one_by_v -u*u*one_by_v*one_by_v)/data_freq(pixIdx);
            }
        }

        // int fcell_idx=1;
        int nom_h = 0, nom_k = 0, nom_l = 0;
        if (use_nominal_hkl) {
            nom_h = nominal_hkl(pixIdx * 3);
            nom_k = nominal_hkl(pixIdx * 3 + 1);
            nom_l = nominal_hkl(pixIdx * 3 + 2);
        }
        CUDAREAL close_distance = close_distances(_pid);

        // reset photon count for this pixel
        double _I = 0;
        double Ilambda = 0;
        double Imiller_h = 0;
        double Imiller_k = 0;
        double Imiller_l = 0;

        kokkos_manager dI, dI2;
        dI.reset();
        dI2.reset();

        for (int _subS = 0; _subS < oversample; ++_subS) {
            for (int _subF = 0; _subF < oversample; ++_subF) {
                // absolute mm position on detector (relative to its origin)
                CUDAREAL _Fdet =
                    subpixel_size * (_fpixel * oversample + _subF) + subpixel_size / 2.0;
                CUDAREAL _Sdet =
                    subpixel_size * (_spixel * oversample + _subS) + subpixel_size / 2.0;

                // assume "distance" is to the front of the detector sensor layer
                int pid_x = _pid * 3;
                int pid_y = _pid * 3 + 1;
                int pid_z = _pid * 3 + 2;


                CUDAREAL fx = fdet_vectors(pid_x);
                CUDAREAL fy = fdet_vectors(pid_y);
                CUDAREAL fz = fdet_vectors(pid_z);
                CUDAREAL sx = sdet_vectors(pid_x);
                CUDAREAL sy = sdet_vectors(pid_y);
                CUDAREAL sz = sdet_vectors(pid_z);
                CUDAREAL ox = odet_vectors(pid_x);
                CUDAREAL oy = odet_vectors(pid_y);
                CUDAREAL oz = odet_vectors(pid_z);
                CUDAREAL px = pix0_vectors(pid_x);
                CUDAREAL py = pix0_vectors(pid_y);
                CUDAREAL pz = pix0_vectors(pid_z);
                KOKKOS_VEC3 _o_vec(ox, oy, oz);

                for (int _thick_tic = 0; _thick_tic < detector_thicksteps; ++_thick_tic) {

                    CUDAREAL _Odet = _thick_tic * detector_thickstep;

                    CUDAREAL pixposX = _Fdet * fx + _Sdet * sx + _Odet * ox + px;
                    CUDAREAL pixposY = _Fdet * fy + _Sdet * sy + _Odet * oy + py;
                    CUDAREAL pixposZ = _Fdet * fz + _Sdet * sz + _Odet * oz + pz;
                    KOKKOS_VEC3 _pixel_pos(pixposX, pixposY, pixposZ);

                    CUDAREAL _airpath_r = 1 / _pixel_pos.length();
                    KOKKOS_VEC3 _diffracted = _pixel_pos.get_unit_vector();

                    const CUDAREAL close_distance = close_distances(_pid);

                    // solid angle subtended by a pixel: (pix/airpath)^2*cos(2theta)
                    CUDAREAL _omega_pixel = pixel_size * pixel_size * _airpath_r * _airpath_r *
                                            close_distance * _airpath_r;

                    // option to turn off obliquity effect, inverse-square-law only
                    if (point_pixel)
                        _omega_pixel = _airpath_r * _airpath_r;

                    // now calculate detector thickness effects
                    CUDAREAL _capture_fraction = 1;

                    CUDAREAL previous_layer = 1.0;
                    if (detector_thick > 0.0 && detector_attnlen_r > 0.0) {
                        // inverse of effective thickness increase
                        KOKKOS_VEC3 _o_vec(ox, oy, oz);
                        CUDAREAL _parallax = _diffracted.dot(_o_vec);
                        CUDAREAL current_layer = ::Kokkos::exp(
                                                -(_thick_tic + 1) * detector_thickstep *
                                                detector_attnlen_r / _parallax);
                        _capture_fraction = previous_layer - current_layer;
                        previous_layer = current_layer;
                    }

                    for (int _source = 0; _source < sources; ++_source) {

                        KOKKOS_VEC3 _incident(
                            -source_X(_source), -source_Y(_source), -source_Z(_source));
                        CUDAREAL _lambda = source_lambda(_source);
                        CUDAREAL sI = source_I(_source);
                        CUDAREAL lambda_ang = _lambda * 1e10;
                        if (use_lambda_coefficients) {
                            lambda_ang = lambda0 + lambda1 * lambda_ang;
                            _lambda = lambda_ang * 1e-10;
                        }

                        // polarization
                        CUDAREAL polar_for_Fhkl_grad=1;
                        if (!nopolar && Fhkl_gradient_mode){

                            // component of diffracted unit vector along incident beam unit vector
                            CUDAREAL cos2theta = _incident.dot(_diffracted);
                            CUDAREAL cos2theta_sqr = cos2theta*cos2theta;
                            CUDAREAL sin2theta_sqr = 1-cos2theta_sqr;

                            CUDAREAL cos2psi=0;
                            if(kahn_factor != 0.0){
                                // cross product to get "vertical" axis that is orthogonal to the cannonical "polarization"
                                KOKKOS_VEC3 B_in = polarization_axis.cross(_incident);
                                // cross product with incident beam to get E-vector direction
                                KOKKOS_VEC3 E_in = _incident.cross(B_in);
                                // get components of diffracted ray projected onto the E-B plane
                                CUDAREAL _kEi = _diffracted.dot(E_in);
                                CUDAREAL _kBi = _diffracted.dot(B_in);
                                // compute the angle of the diffracted ray projected onto the incident E-B plane
                                // calculate cos(2 * atan2(_kBi, _kEi))
                                if (_kEi!=0) {
                                    CUDAREAL ratio = _kBi / _kEi;
                                    cos2psi = (1 - ratio*ratio) / (1 + ratio*ratio);
                                } else {
                                    cos2psi = -1;
                                }
                            }
                            // correction for polarized incident beam
                            polar_for_Fhkl_grad = 0.5*(1.0 + cos2theta_sqr - kahn_factor*cos2psi*sin2theta_sqr);
                        }
                        KOKKOS_VEC3 _scattering = (_diffracted - _incident) / _lambda;

                        KOKKOS_VEC3 q_vec = _scattering * 1e-10;

                        // TODO rename
                        CUDAREAL texture_scale = _capture_fraction * _omega_pixel * sI;
                        for (int _phi_tic=0; _phi_tic<phisteps; ++_phi_tic){
                            KOKKOS_MAT3 Rphi;
                            CUDAREAL phi = phi0 + phistep*_phi_tic;
                            if (phi != 0){
                                CUDAREAL c = cos(phi);
                                CUDAREAL omc = 1-c;
                                CUDAREAL s = sin(phi);
                                Rphi = KOKKOS_MAT3{c + gx*gx*omc,    gx*gy*omc-gz*s,   gx*gz*omc+gy*s,
                                                 gy*gx*omc + gz*s,   c + gy*gy*omc,   gy*gz*omc - gx*s,
                                                 gz*gx*omc - gy*s,  gz*gy*omc + gx*s, c + gz*gz*omc};
                            }

                        for (int _mos_tic = 0; _mos_tic < mosaic_domains; ++_mos_tic) {
                            KOKKOS_MAT3 UBO = Amatrices(_mos_tic);
                            if (phi != 0){
                                KOKKOS_MAT3 Um = UMATS_RXYZ(_mos_tic);
                                UBO = UBO*Um*Rphi.transpose()*Um.transpose();
                            }

                            KOKKOS_VEC3 H_vec = UBO * q_vec;
                            CUDAREAL _h = H_vec[0];
                            CUDAREAL _k = H_vec[1];
                            CUDAREAL _l = H_vec[2];

                            int _h0 = ceil(_h - 0.5);
                            int _k0 = ceil(_k - 0.5);
                            int _l0 = ceil(_l - 0.5);

                            KOKKOS_VEC3 H0(_h0, _k0, _l0);

                            KOKKOS_VEC3 delta_H = H_vec - H0;
                            KOKKOS_VEC3 V = _NABC * delta_H;
                            CUDAREAL _hrad_sqr = V.length_sqr();
                            CUDAREAL I0;
                            if(square_shape){
                                CUDAREAL F_latt = 1.0;
                               if(Na>1)
                                   F_latt *= sincg(M_PI*_h,Na);
                               if(Nb>1)
                                   F_latt *= sincg(M_PI*_k,Nb);
                               if(Nc>1)
                                   F_latt *= sincg(M_PI*_l,Nc);
                               I0 = F_latt*F_latt;
                            }
                            else {
                                CUDAREAL exparg;
                                if (gaussian_star_shape){
                                    // TODO can precompute xtal_size_sq to save time
                                    KOKKOS_VEC3 A {UBO[0], UBO[1], UBO[2]};
                                    KOKKOS_VEC3 B {UBO[3], UBO[4], UBO[5]};
                                    KOKKOS_VEC3 C {UBO[6], UBO[7], UBO[8]};
                                    CUDAREAL cell_vol = A.dot(B.cross(C));
                                    CUDAREAL xtal_size_sq = pow(NABC_det*cell_vol, CUDAREAL(2)/CUDAREAL(3));
                                    KOKKOS_MAT3 Ainv = UBO.inverse();
                                    KOKKOS_VEC3 delta_Q = Ainv*delta_H;
                                    CUDAREAL rad_star_sqr = delta_Q.dot(delta_Q)*xtal_size_sq;
                                    exparg = rad_star_sqr*1.9*fudge ;
                                }
                                else
                                    exparg = _hrad_sqr * C / 2;
                                I0 = 0;
                                if (exparg < 35)
                                    if (no_Nabc_scale)
                                        I0 = ::Kokkos::exp(-2 * exparg);
                                    else
                                        I0 = (NABC_det_sq) *
                                                ::Kokkos::exp(-2 * exparg);
                            }

                            // are we doing diffuse scattering
                            CUDAREAL step_diffuse_param[6] = {0, 0, 0, 0, 0, 0};
                            if (use_diffuse) {
                                // ***NEEDS UPDATE: use legacy API for passing diffuse scale as KOKKOS_MAT3
                               calc_diffuse_at_hkl(H_vec,H0,dHH,h_min,k_min,l_min,h_max,k_max,l_max,h_range,k_range,l_range,diffuse_scale_mat3,FhklLinear,num_laue_mats,laue_mats,anisoG_local,dG_trace,anisoG_determ,anisoU_local,dG_dgam,(refine_flag & REFINE_DIFFUSE)>0,&I0,step_diffuse_param);
                            } // end s_use_diffuse outer

                            CUDAREAL _F_cell = default_F;
                            CUDAREAL _F_cell2 = 0;
                            int i_hklasu=0;

                            if ((_h0 <= h_max) && (_h0 >= h_min) &&
                                (_k0 <= k_max) && (_k0 >= k_min) &&
                                (_l0 <= l_max) && (_l0 >= l_min)) {
                                int Fhkl_linear_index = (_h0 - h_min) * k_range * l_range +
                                                        (_k0 - k_min) * l_range + (_l0 - l_min);
                                //_F_cell = __ldg(&FhklLinear[Fhkl_linear_index]);
                                _F_cell = FhklLinear(Fhkl_linear_index);
                                // if (complex_miller) _F_cell2 =
                                // __ldg(&Fhkl2Linear[Fhkl_linear_index]);
                                if (complex_miller)
                                    _F_cell2 = Fhkl2Linear(Fhkl_linear_index);
                                if (Fhkl_have_scale_factors)
                                    i_hklasu = FhklLinear_ASUid(Fhkl_linear_index);
                            }

                            CUDAREAL c_deriv_Fcell = 0;
                            CUDAREAL d_deriv_Fcell = 0;
                            if (complex_miller) {
                                CUDAREAL c_deriv_Fcell_real = 0;
                                CUDAREAL c_deriv_Fcell_imag = 0;
                                CUDAREAL d_deriv_Fcell_real = 0;
                                CUDAREAL d_deriv_Fcell_imag = 0;
                                if (num_atoms > 0) {
                                    CUDAREAL S_2 = (q_vec[0] * q_vec[0] +
                                                    q_vec[1] * q_vec[1] +
                                                    q_vec[2] * q_vec[2]);

                                    // fp is always followed by the fdp value
                                    CUDAREAL val_fp = fpfdp(2 * _source);
                                    CUDAREAL val_fdp = fpfdp(2 * _source + 1);

                                    CUDAREAL c_deriv_prime = 0;
                                    CUDAREAL c_deriv_dblprime = 0;
                                    CUDAREAL d_deriv_prime = 0;
                                    CUDAREAL d_deriv_dblprime = 0;
                                    if (refine_flag & REFINE_FP_FDP) {
                                        //   currently only supports two parameter model
                                        int d_idx = 2 * _source;
                                        c_deriv_prime = fpfdp_derivs(d_idx);
                                        c_deriv_dblprime = fpfdp_derivs(d_idx + 1);
                                        d_deriv_prime = fpfdp_derivs(d_idx + 2 * sources);
                                        d_deriv_dblprime =
                                            fpfdp_derivs(d_idx + 1 + 2 * sources);
                                    }

                                    for (int i_atom = 0; i_atom < num_atoms; i_atom++) {
                                        // fractional atomic coordinates
                                        CUDAREAL atom_x = atom_data(i_atom * 5);
                                        CUDAREAL atom_y = atom_data(i_atom * 5 + 1);
                                        CUDAREAL atom_z = atom_data(i_atom * 5 + 2);
                                        CUDAREAL B = atom_data(i_atom * 5 + 3);  // B factor
                                        B = ::Kokkos::exp(
                                            -B * S_2 / 4.0);  // TODO: speed me up?
                                        CUDAREAL occ = atom_data(i_atom * 5 + 4);  // occupancy
                                        CUDAREAL r_dot_h =
                                            _h0 * atom_x + _k0 * atom_y + _l0 * atom_z;
                                        CUDAREAL phase = 2 * M_PI * r_dot_h;
                                        CUDAREAL s_rdoth = ::Kokkos::sin(phase);
                                        CUDAREAL c_rdoth = ::Kokkos::cos(phase);
                                        CUDAREAL Bocc = B * occ;
                                        CUDAREAL BC = B * c_rdoth;
                                        CUDAREAL BS = B * s_rdoth;
                                        CUDAREAL real_part = BC * val_fp - BS * val_fdp;
                                        CUDAREAL imag_part = BS * val_fp + BC * val_fdp;
                                        _F_cell += real_part;
                                        _F_cell2 += imag_part;
                                        if (refine_flag & REFINE_FP_FDP) {
                                            c_deriv_Fcell_real +=
                                                BC * c_deriv_prime - BS * c_deriv_dblprime;
                                            c_deriv_Fcell_imag +=
                                                BS * c_deriv_prime + BC * c_deriv_dblprime;

                                            d_deriv_Fcell_real +=
                                                BC * d_deriv_prime - BS * d_deriv_dblprime;
                                            d_deriv_Fcell_imag +=
                                                BS * d_deriv_prime + BC * d_deriv_dblprime;
                                        }
                                    }
                                }
                                CUDAREAL Freal = _F_cell;
                                CUDAREAL Fimag = _F_cell2;
                                _F_cell =
                                    ::Kokkos::sqrt(Freal * Freal + Fimag * Fimag);
                                if (refine_flag & REFINE_FP_FDP) {
                                    c_deriv_Fcell =
                                        Freal * c_deriv_Fcell_real + Fimag * c_deriv_Fcell_imag;
                                    d_deriv_Fcell =
                                        Freal * d_deriv_Fcell_real + Fimag * d_deriv_Fcell_imag;
                                }
                            }
                            if (!oversample_omega && ! Fhkl_gradient_mode)
                                _omega_pixel = 1;

                            CUDAREAL _I_cell = _F_cell;
                            if (!(refine_flag & REFINE_ICELL))
                                _I_cell *= _F_cell;
                            CUDAREAL hkl=1;
                            int Fhkl_channel=0;
                            if (! Fhkl_channels_empty)
                                Fhkl_channel = Fhkl_channels(_source);
                            if (Fhkl_have_scale_factors)
                                hkl = Fhkl_scale(i_hklasu + Fhkl_channel*Num_ASU);
                            if (Fhkl_gradient_mode){
                                CUDAREAL Fhkl_deriv_scale = overall_scale*polar_for_Fhkl_grad;
                                CUDAREAL I_noFcell=texture_scale*I0;
                                CUDAREAL dfhkl = I_noFcell*_I_cell * Fhkl_deriv_scale;
                                CUDAREAL grad_incr = dfhkl*Fhkl_deriv_coef;
                                int fhkl_grad_idx=i_hklasu + Fhkl_channel*Num_ASU;

                                if (Fhkl_errors_mode){
                                    // here we hi-kack the Fhkl_scale_deriv array, if computing errors, in order to store the hessian terms
                                    // if we are getting the hessian terms, we no longer need the  gradients (e.g. by this point we are done refininig)
                                    CUDAREAL hessian_incr = Fhkl_hessian_coef*dfhkl*dfhkl;
                                    ::Kokkos::atomic_add(&Fhkl_scale_deriv(fhkl_grad_idx), hessian_incr);
                                }
                                else{
                                    ::Kokkos::atomic_add(&Fhkl_scale_deriv(fhkl_grad_idx), grad_incr);
                                }
                                continue;
                            }

                            CUDAREAL _I_total = hkl*_I_cell *I0;
                            CUDAREAL Iincrement = _I_total * texture_scale;
                            _I += Iincrement;
                            if (save_wavelenimage){
                                Ilambda += Iincrement * lambda_ang;
                                Imiller_h += Iincrement*_h;
                                Imiller_k += Iincrement*_k;
                                Imiller_l += Iincrement*_k;
                            }

                            if (refine_flag & REFINE_DIFFUSE) {
                                CUDAREAL step_scale = texture_scale * _F_cell * _F_cell;
                                for (int i_diff = 0; i_diff < 6; i_diff++) {
                                    dI.diffuse[i_diff] +=
                                        step_scale * step_diffuse_param[i_diff];
                                }
                            }

                            //*************************************************
                            // START REFINEMENT

                            if (refine_flag & REFINE_FP_FDP) {
                                CUDAREAL I_noFcell = texture_scale * I0;
                                dI.fp_fdp[0] += 2 * I_noFcell * (c_deriv_Fcell);
                                dI.fp_fdp[1] += 2 * I_noFcell * (d_deriv_Fcell);
                            }

                            if (verbose > 3)
                                printf(
                                    "hkl= %f %f %f  hkl1= %d %d %d  Fcell=%f\n", _h, _k, _l,
                                    _h0, _k0, _l0, _F_cell);

                            KOKKOS_MAT3 UBOt;
                             if (refine_flag & (REFINE_UMAT | REFINE_ETA)) {
                                UBOt = Amat_init;
                                if (phi != 0)
                                    UBOt = Rphi*UBOt;
                            }
                            if (refine_flag & REFINE_UMAT1) {
                                const KOKKOS_VEC3 dV = UMATS_prime(_mos_tic, 0) * q_vec;
                                const CUDAREAL V_dot_dV = V.dot(dV);
                                const CUDAREAL value = -two_C * V_dot_dV * Iincrement;
                                CUDAREAL value2 = 0;
                                if (compute_curvatures) {
                                    const CUDAREAL dV_dot_dV = dV.length_sqr();
                                    const CUDAREAL dV2_dot_V = V.dot(UMATS_dbl_prime(_mos_tic, 0)*q_vec);
                                    value2 = two_C * (two_C * V_dot_dV * V_dot_dV - dV2_dot_V - dV_dot_dV) * Iincrement;
                                }
                                dI.rot[0] += value;
                                dI2.rot[0] += value2;
                            }
                            if (refine_flag & REFINE_UMAT2) {
                                KOKKOS_VEC3 dV = UMATS_prime(_mos_tic, 1) * q_vec;
                                CUDAREAL V_dot_dV = V.dot(dV);
                                CUDAREAL value = -two_C * V_dot_dV * Iincrement;

                                CUDAREAL value2 = 0;
                                if (compute_curvatures) {
                                    const CUDAREAL dV_dot_dV = dV.length_sqr();
                                    CUDAREAL dV2_dot_V = V.dot(UMATS_dbl_prime(_mos_tic, 1)*q_vec);
                                    value2 = two_C * (two_C * V_dot_dV * V_dot_dV - dV2_dot_V - dV_dot_dV) * Iincrement;
                                }
                                dI.rot[1] += value;
                                dI2.rot[1] += value2;
                            }
                            if (refine_flag & REFINE_UMAT3) {
                                KOKKOS_VEC3 dV = UMATS_prime(_mos_tic, 2) * q_vec;
                                CUDAREAL V_dot_dV = V.dot(dV);
                                CUDAREAL value = -two_C * V_dot_dV * Iincrement;

                                CUDAREAL value2 = 0;
                                if (compute_curvatures) {
                                    const CUDAREAL dV_dot_dV = dV.length_sqr();
                                    CUDAREAL dV2_dot_V = V.dot(UMATS_dbl_prime(_mos_tic, 2)*q_vec);
                                    value2 = two_C * (two_C * V_dot_dV * V_dot_dV - dV2_dot_V - dV_dot_dV) * Iincrement;
                                }
                                dI.rot[2] += value;
                                dI2.rot[2] += value2;
                            }
                            // Checkpoint for unit cell derivatives
                            for (int i_uc = 0; i_uc < 6; i_uc++) {
                                if (refine_flag & (REFINE_BMAT1 << i_uc)) {
                                    KOKKOS_VEC3 dV = BMATS_prime(_mos_tic, i_uc) * q_vec;
                                    CUDAREAL V_dot_dV = V.dot(dV);
                                    CUDAREAL value = -two_C * V_dot_dV * Iincrement;
                                    CUDAREAL value2 = 0;
                                    if (compute_curvatures) {
                                        const CUDAREAL dV_dot_dV = dV.length_sqr();
                                        CUDAREAL dV2_dot_V = V.dot(BMATS_dbl_prime(_mos_tic, i_uc)*q_vec);
                                        value2 = two_C * (two_C * V_dot_dV * V_dot_dV - dV2_dot_V - dV_dot_dV) * Iincrement;
                                    }
                                    dI.ucell[i_uc] += value;
                                    dI2.ucell[i_uc] += value2;
                                }
                            }  // end ucell deriv

                            // Checkpoint for Ncells manager
                            if (refine_flag & REFINE_NCELLS1) {
                                int num_ncell_deriv = 1;
                                if (!isotropic_ncells)
                                    num_ncell_deriv = 3;
                                for (int i_nc = 0; i_nc < num_ncell_deriv; i_nc++) {
                                    KOKKOS_MAT3 dN;
                                    dN(i_nc, i_nc) = 1;
                                    if (num_ncell_deriv == 1) {
                                        dN(0, 0) = 1;
                                        dN(1, 1) = 1;
                                        dN(2, 2) = 1;
                                    }
                                    CUDAREAL N_i = _NABC(i_nc, i_nc);
                                    KOKKOS_VEC3 dV_dN = dN.dot(delta_H);
                                    // TODO speedops: precompute these, store shared var
                                    // _NABC.inverse
                                    CUDAREAL determ_deriv = (_NABC.inverse().dot(dN)).trace();
                                    CUDAREAL deriv_coef = determ_deriv - C * (dV_dN.dot(V));
                                    CUDAREAL value = 2 * Iincrement * deriv_coef;
                                    CUDAREAL value2 = 0;
                                    if (compute_curvatures) {
                                        value2 = (-1 / N_i / N_i - C * (dV_dN.dot(dV_dN))) * 2 *
                                                    Iincrement;
                                        value2 += deriv_coef * 2 * value;
                                    }
                                    dI.Ncells[i_nc] += value;
                                    dI2.Ncells[i_nc] += value2;
                                }
                            }  // end Ncells manager deriv

                            if (refine_flag & REFINE_NCELLS_DEF) {
                                for (int i_nc = 3; i_nc < 6; i_nc++) {
                                    KOKKOS_MAT3 dN;
                                    if (i_nc == 3)
                                        dN = KOKKOS_MAT3{0, 1, 0, 1, 0, 0, 0, 0, 0};
                                    else if (i_nc == 4)
                                        dN = KOKKOS_MAT3{0, 0, 0, 0, 0, 1, 0, 1, 0};
                                    else
                                        dN = KOKKOS_MAT3{0, 0, 1, 0, 0, 0, 1, 0, 0};
                                    KOKKOS_VEC3 dV_dN = dN.dot(delta_H);
                                    // TODO speedops: precompute these
                                    CUDAREAL determ_deriv = (_NABC.inverse().dot(dN)).trace();
                                    CUDAREAL deriv_coef = determ_deriv - C * (dV_dN.dot(V));
                                    CUDAREAL value = 2 * Iincrement * deriv_coef;
                                    dI.Ncells[i_nc] += value;
                                    CUDAREAL value2 = 0;
                                    if (compute_curvatures) {
                                        value2 = deriv_coef * value;
                                        value2 += -2 * C * Iincrement * (dV_dN.dot(dV_dN));
                                        dI2.Ncells[i_nc] += value2;
                                    }
                                }
                            }

                            // Checkpoint for Origin manager
                            for (int i_pan_orig = 0; i_pan_orig < 3; i_pan_orig++) {
                                if (refine_flag & (REFINE_PANEL_ORIGIN1 << i_pan_orig)) {
                                    CUDAREAL per_k = _airpath_r;
                                    CUDAREAL per_k3 = pow(per_k, 3.);
                                    CUDAREAL per_k5 = pow(per_k, 5.);

                                    KOKKOS_MAT3 M = -two_C * (_NABC.dot(UBO)) / lambda_ang;
                                    KOKKOS_VEC3 dk;
                                    if (i_pan_orig == 0)
                                        dk = KOKKOS_VEC3{0, 0, 1};
                                    else if (i_pan_orig == 1)
                                        dk = KOKKOS_VEC3{1, 0, 0};
                                    else
                                        dk = KOKKOS_VEC3{0, 1, 0};

                                    CUDAREAL G = dk.dot(_pixel_pos);
                                    CUDAREAL pix2 = subpixel_size * subpixel_size;
                                    KOKKOS_VEC3 dk_hat = -per_k3 * G * _pixel_pos + per_k * dk;
                                    CUDAREAL coef = (M.dot(dk_hat)).dot(V);
                                    CUDAREAL coef2 =
                                        -3 * pix2 * per_k5 * G * (_o_vec.dot(_pixel_pos));
                                    coef2 += pix2 * per_k3 * (_o_vec.dot(dk));
                                    CUDAREAL value =
                                        coef * Iincrement + coef2 * Iincrement / _omega_pixel;

                                    dI.pan_orig[i_pan_orig] += value;
                                    dI2.pan_orig[i_pan_orig] += 0;

                                }  // end origin manager deriv
                            }

                            for (int i_pan_rot = 0; i_pan_rot < 3; i_pan_rot++) {
                                if (refine_flag & (REFINE_PANEL_ROT1 << i_pan_rot)) {
                                    CUDAREAL per_k = _airpath_r;
                                    CUDAREAL per_k3 = pow(per_k, 3.);
                                    CUDAREAL per_k5 = pow(per_k, 5.);
                                    KOKKOS_MAT3 M = -two_C * (_NABC.dot(UBO)) / lambda_ang;
                                    KOKKOS_VEC3 dk = _Fdet * (dF_vecs(_pid * 3 + i_pan_rot)) +
                                                _Sdet * (dS_vecs(_pid * 3 + i_pan_rot));
                                    CUDAREAL G = dk.dot(_pixel_pos);
                                    CUDAREAL pix2 = subpixel_size * subpixel_size;
                                    KOKKOS_VEC3 dk_hat = -per_k3 * G * _pixel_pos + per_k * dk;
                                    CUDAREAL coef = (M.dot(dk_hat)).dot(V);
                                    CUDAREAL coef2 =
                                        -3 * pix2 * per_k5 * G * (_o_vec.dot(_pixel_pos));
                                    coef2 += pix2 * per_k3 * (_o_vec.dot(dk));
                                    CUDAREAL value =
                                        coef * Iincrement + coef2 * Iincrement / _omega_pixel;

                                    dI.pan_rot[i_pan_rot] += value;
                                    dI2.pan_rot[i_pan_rot] += 0;
                                }
                            }

                            // checkpoint for Fcell manager
                            if (refine_flag & REFINE_FCELL) {
                                CUDAREAL value;
                                if (refine_flag & REFINE_ICELL)
                                    value = I0 * texture_scale;
                                else
                                    value = 2 * I0 * _F_cell *
                                            texture_scale;  // Iincrement/_F_cell ;
                                CUDAREAL value2 = 0;
                                if (compute_curvatures) {
                                    //    NOTE if _Fcell >0
                                    value2 = 2 * I0 * texture_scale;
                                }
                                // if (fcell_idx >=0 && fcell_idx <=2){
                                if (use_nominal_hkl) {
                                    if (_h0 == nom_h && _k0 == nom_k && _l0 == nom_l) {
                                        dI.fcell += value;
                                        dI2.fcell += value2;
                                    }
                                } else {
                                    dI.fcell += value;
                                    dI2.fcell += value2;
                                }
                            }  // end of fcell man deriv

                            // checkpoint for eta manager
                            if (refine_flag & REFINE_ETA) {
                                for (int i_eta = 0; i_eta < 3; i_eta++) {
                                    if (i_eta > 0 && !aniso_eta)
                                        continue;
                                    int mtic2 = _mos_tic + i_eta * mosaic_domains;
                                    KOKKOS_VEC3 DeltaH_deriv = (UMATS_RXYZ_prime(mtic2).dot(UBOt))
                                                            .transpose()
                                                            .dot(q_vec);
                                    // vector V is _Nabc*Delta_H
                                    KOKKOS_VEC3 dV = _NABC.dot(DeltaH_deriv);
                                    CUDAREAL V_dot_dV = V.dot(dV);
                                    CUDAREAL Iprime = -two_C * (V_dot_dV)*Iincrement;
                                    dI.eta[i_eta] += Iprime;
                                    CUDAREAL Idbl_prime = 0;
                                    if (compute_curvatures) {
                                        KOKKOS_VEC3 DeltaH_second_deriv =
                                            (UMATS_RXYZ_dbl_prime(mtic2).dot(UBOt))
                                                .transpose()
                                                .dot(q_vec);
                                        KOKKOS_VEC3 dV2 = _NABC.dot(DeltaH_second_deriv);
                                        Idbl_prime =
                                            -two_C * (dV.dot(dV) + V.dot(dV2)) * Iincrement;
                                        Idbl_prime += -two_C * (V_dot_dV)*Iprime;
                                    }
                                    dI2.eta[i_eta] += Idbl_prime;
                                }
                            }  // end of eta man deriv

                            // checkpoint for lambda manager
                            for (int i_lam = 0; i_lam < 2; i_lam++) {
                                if (refine_flag & (REFINE_LAMBDA << i_lam)) {
                                    CUDAREAL NH_dot_V = (_NABC.dot(H_vec)).dot(V);
                                    CUDAREAL dg_dlambda;
                                    if (i_lam == 0)
                                        dg_dlambda = 1;
                                    else  // i_lam==1
                                        dg_dlambda = lambda_ang;
                                    CUDAREAL coef =
                                        NH_dot_V * two_C * (dg_dlambda) / lambda_ang;
                                    CUDAREAL value = coef * Iincrement;
                                    CUDAREAL value2 = 0;
                                    dI.lambda[i_lam] += value;
                                    dI2.lambda[i_lam] += value2;
                                }
                            }
                            // end of lambda deriv
                            if (printout) {
                                if (_subS == 0 && _subF == 0 && _thick_tic == 0 &&
                                    _source == 0 && _mos_tic == 0) {
                                    if ((_fpixel == printout_fpixel &&
                                            _spixel == printout_spixel) ||
                                        printout_fpixel < 0) {
                                        printf("%4d %4d :  lambda = %g\n", _fpixel, _spixel, _lambda);
                                        printf(
                                            "at %g %g %g\n", _pixel_pos[0], _pixel_pos[1],
                                            _pixel_pos[2]);
                                        printf("Fdet= %g; Sdet= %g ; Odet= %g\n", _Fdet, _Sdet, _Odet);
                                        printf(
                                            "PIX0: %f %f %f\n", pix0_vectors(pid_x),
                                            pix0_vectors(pid_y), pix0_vectors(pid_z));
                                        printf(
                                            "F: %f %f %f\n", fdet_vectors(pid_x),
                                            fdet_vectors(pid_y), fdet_vectors(pid_z));
                                        printf(
                                            "S: %f %f %f\n", sdet_vectors(pid_x),
                                            sdet_vectors(pid_y), sdet_vectors(pid_z));
                                        printf(
                                            "O: %f %f %f\n", odet_vectors(pid_x),
                                            odet_vectors(pid_y), odet_vectors(pid_z));
                                        printf("pid_x=%d, pid_y=%d; pid_z=%d\n", pid_x, pid_y, pid_z);
                                        printf(
                                            "QVECTOR: %f %f %f\n", q_vec[0], q_vec[1], q_vec[2]);
                                        printf("omega   %15.10g\n", _omega_pixel);
                                        printf(
                                            "Incident: %g %g %g\n",
                                            _incident[0], _incident[1], _incident[2]);

                                        KOKKOS_MAT3 UU = UMATS_RXYZ(_mos_tic);
                                        printf(
                                            "UMAT_RXYZ :\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n",
                                            UU(0, 0), UU(0, 1), UU(0, 2), UU(1, 0), UU(1, 1),
                                            UU(1, 2), UU(2, 0), UU(2, 1), UU(2, 2));
                                        UU = Bmat_realspace;
                                        printf(
                                            "Bmat_realspace :\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n",
                                            UU(0, 0), UU(0, 1), UU(0, 2), UU(1, 0), UU(1, 1),
                                            UU(1, 2), UU(2, 0), UU(2, 1), UU(2, 2));
                                        UU = UBO;
                                        printf(
                                            "UBO :\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n",
                                            UU(0, 0), UU(0, 1), UU(0, 2), UU(1, 0), UU(1, 1),
                                            UU(1, 2), UU(2, 0), UU(2, 1), UU(2, 2));

                                        UU = UBOt;
                                        printf(
                                            "UBOt :\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n",
                                            UU(0, 0), UU(0, 1), UU(0, 2), UU(1, 0), UU(1, 1),
                                            UU(1, 2), UU(2, 0), UU(2, 1), UU(2, 2));

                                        // UU = UmosRxRyRzU;
                                        // printf(
                                        //     "UmosRxRyRzU :\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n",
                                        //     UU(0, 0), UU(0, 1), UU(0, 2), UU(1, 0), UU(1, 1),
                                        //     UU(1, 2), UU(2, 0), UU(2, 1), UU(2, 2));
                                        // KOKKOS_VEC3 AA = delta_H_prime;
                                        // printf(
                                        //     "delta_H_prime :\n%f  %f  %f\n", AA[0], AA[1],
                                        //     AA[2]);
                                        printf("Iincrement: %f\n", Iincrement);
                                        printf(
                                            "hkl= %f %f %f  hkl0= %d %d %d\n", _h, _k, _l, _h0,
                                            _k0, _l0);
                                        printf(
                                            " F_cell=%g  F_cell2=%g I_latt=%g   I = %g\n",
                                            _F_cell, _F_cell2, I0, _I);
                                        printf("I/steps %15.10g\n", _I / Nsteps);
                                        // printf("Ilatt diffuse %15.10g\n", I_latt_diffuse);
                                        printf("default_F= %f\n", default_F);
                                        if (complex_miller)
                                            printf("COMPLEX MILLER!\n");
                                        if (no_Nabc_scale)
                                            printf("No Nabc scale!\n");
                                    }
                                }
                            } // end of printout if

                        } // end of mos_tic loop
                      } // end of phi_tic loop
                    } // end of source loop
                } // end of thick step loop
            } // end of fpos loop
        } // end of spos loop
        floatimage(pixIdx) = _I;
        if (save_wavelenimage){
            wavelenimage(pixIdx*4) = Ilambda / _I;
            wavelenimage(pixIdx*4+1) = Imiller_h / _I;
            wavelenimage(pixIdx*4+2) = Imiller_k / _I;
            wavelenimage(pixIdx*4+3) = Imiller_l / _I;
        }

        if (refine_flag) {
            manager_dI(pixIdx) = dI;
            manager_dI2(pixIdx) = dI2;
        }
    }); // end pixIdx loop

    if (Fhkl_gradient_mode)
        return;

    Kokkos::parallel_for(
        "deriv_image_increment", Npix_to_model, KOKKOS_LAMBDA(const int& pixIdx) {

        int _pid = panels_fasts_slows(pixIdx * 3);
        int _fpixel = panels_fasts_slows(pixIdx * 3 + 1);
        int _spixel = panels_fasts_slows(pixIdx * 3 + 2);

        CUDAREAL _Fdet_ave = pixel_size * _fpixel + pixel_size / 2.0;
        CUDAREAL _Sdet_ave = pixel_size * _spixel + pixel_size / 2.0;
        CUDAREAL _Odet_ave = 0;  // Odet;
        // TODO maybe make this more general for thick detectors?

        KOKKOS_VEC3 _pixel_pos_ave(0, 0, 0);
        int pid_x = _pid * 3;
        int pid_y = _pid * 3 + 1;
        int pid_z = _pid * 3 + 2;

        CUDAREAL fx = fdet_vectors(pid_x);
        CUDAREAL fy = fdet_vectors(pid_y);
        CUDAREAL fz = fdet_vectors(pid_z);

        CUDAREAL sx = sdet_vectors(pid_x);
        CUDAREAL sy = sdet_vectors(pid_y);
        CUDAREAL sz = sdet_vectors(pid_z);

        CUDAREAL ox = odet_vectors(pid_x);
        CUDAREAL oy = odet_vectors(pid_y);
        CUDAREAL oz = odet_vectors(pid_z);

        CUDAREAL px = pix0_vectors(pid_x);
        CUDAREAL py = pix0_vectors(pid_y);
        CUDAREAL pz = pix0_vectors(pid_z);

        _pixel_pos_ave[0] = _Fdet_ave * fx + _Sdet_ave * sx + _Odet_ave * ox + px;
        _pixel_pos_ave[1] = _Fdet_ave * fy + _Sdet_ave * sy + _Odet_ave * oy + py;
        _pixel_pos_ave[2] = _Fdet_ave * fz + _Sdet_ave * sz + _Odet_ave * oz + pz;

        CUDAREAL close_distance = close_distances(_pid);

        CUDAREAL _airpath_ave_r = 1 / _pixel_pos_ave.length();
        KOKKOS_VEC3 _diffracted_ave = _pixel_pos_ave.get_unit_vector();
        CUDAREAL _omega_pixel_ave = pixel_size * pixel_size * _airpath_ave_r * _airpath_ave_r *
                                    close_distance * _airpath_ave_r;

        CUDAREAL _polar = 1;
        if (!nopolar) {
            KOKKOS_VEC3 _incident(-source_X(0), -source_Y(0), -source_Z(0));
            _incident.normalize();
            // component of diffracted unit vector along _incident beam unit vector
            CUDAREAL cos2theta = _incident.dot(_diffracted_ave);
            CUDAREAL cos2theta_sqr = cos2theta * cos2theta;
            CUDAREAL sin2theta_sqr = 1 - cos2theta_sqr;

            CUDAREAL cos2psi = 0;
            if (kahn_factor != 0.0) {
                // cross product to get "vertical" axis that is orthogonal to the cannonical
                // "polarization"
                KOKKOS_VEC3 B_in = polarization_axis.cross(_incident);
                // cross product with _incident beam to get E-vector direction
                KOKKOS_VEC3 E_in = _incident.cross(B_in);
                // get components of diffracted ray projected onto the E-B plane
                CUDAREAL _kEi = _diffracted_ave.dot(E_in);
                CUDAREAL _kBi = _diffracted_ave.dot(B_in);
                // compute the angle of the diffracted ray projected onto the incident E-B plane
                // calculate cos(2 * atan2(_kBi, _kEi))
                if (_kEi!=0) {
                    CUDAREAL ratio = _kBi / _kEi;
                    cos2psi = (1 - ratio*ratio) / (1 + ratio*ratio);
                } else {
                    cos2psi = -1;
                }
            }
            // correction for polarized _incident beam
            _polar = 0.5 * (1.0 + cos2theta_sqr - kahn_factor * cos2psi * sin2theta_sqr);
        }

        CUDAREAL _om = 1;
        if (!oversample_omega)
            _om = _omega_pixel_ave;
        // final scale term to being everything to photon number units
        CUDAREAL _scale_term = _polar * _om * overall_scale;
        floatimage(pixIdx) *= _scale_term;

        auto& dI = manager_dI(pixIdx);
        auto& dI2 = manager_dI2(pixIdx);

        // udpate the rotation derivative images*
        for (int i_rot = 0; i_rot < 3; i_rot++) {
            if (refine_flag & (REFINE_UMAT1 << i_rot)) {
                CUDAREAL value = _scale_term * dI.rot[i_rot];
                CUDAREAL value2 = _scale_term * dI2.rot[i_rot];
                int idx = i_rot * Npix_to_model + pixIdx;
                d_Umat_images(idx) = value;
                d2_Umat_images(idx) = value2;
            }
        }  // end rot deriv image increment

        // update the ucell derivative images
        for (int i_uc = 0; i_uc < 6; i_uc++) {
            if (refine_flag & (REFINE_BMAT1 << i_uc)) {
                CUDAREAL value = _scale_term * dI.ucell[i_uc];
                CUDAREAL value2 = _scale_term * dI2.ucell[i_uc];
                int idx = i_uc * Npix_to_model + pixIdx;
                d_Bmat_images(idx) = value;
                d2_Bmat_images(idx) = value2;
            }
        }  // end ucell deriv image increment

        // update the Ncells derivative image
        if (refine_flag & REFINE_NCELLS1) {
            CUDAREAL value = _scale_term * dI.Ncells[0];
            CUDAREAL value2 = _scale_term * dI2.Ncells[0];
            int idx = pixIdx;
            d_Ncells_images(idx) = value;
            d2_Ncells_images(idx) = value2;

            if (!isotropic_ncells) {
                value = _scale_term * dI.Ncells[1];
                value2 = _scale_term * dI2.Ncells[1];
                idx = Npix_to_model + pixIdx;
                d_Ncells_images(idx) = value;
                d2_Ncells_images(idx) = value2;

                value = _scale_term * dI.Ncells[2];
                value2 = _scale_term * dI2.Ncells[2];
                idx = Npix_to_model * 2 + pixIdx;
                d_Ncells_images(idx) = value;
                d2_Ncells_images(idx) = value2;
            }
        }  // end Ncells deriv image increment
        if (refine_flag & REFINE_NCELLS_DEF) {
            for (int i_nc = 3; i_nc < 6; i_nc++) {
                CUDAREAL value = _scale_term * dI.Ncells[i_nc];
                CUDAREAL value2 = _scale_term * dI2.Ncells[i_nc];
                int idx = i_nc * Npix_to_model + pixIdx;
                d_Ncells_images(idx) = value;
                d2_Ncells_images(idx) = value2;
            }
        }

        // update Fcell derivative image
        if (refine_flag & REFINE_FCELL) {
            CUDAREAL value = _scale_term * dI.fcell;
            CUDAREAL value2 = _scale_term * dI2.fcell;
            d_fcell_images(pixIdx) = value;
            d2_fcell_images(pixIdx) = value2;
        }  // end Fcell deriv image increment

        if (refine_flag & REFINE_FP_FDP) {
            // c derivative
            CUDAREAL value = _scale_term * dI.fp_fdp[0];
            d_fp_fdp_images(pixIdx) = value;
            // d derivative
            value = _scale_term * dI.fp_fdp[1];
            d_fp_fdp_images(Npix_to_model + pixIdx) = value;
        }
        if (refine_flag & REFINE_DIFFUSE) {
            for (int i_gam = 0; i_gam < 3; i_gam++) {
                CUDAREAL val = dI.diffuse[i_gam] * _scale_term;
                int img_idx = Npix_to_model * i_gam + pixIdx;
                d_diffuse_gamma_images(img_idx) = val;
            }
            for (int i_sig = 0; i_sig < 3; i_sig++) {
                CUDAREAL val = dI.diffuse[i_sig + 3] * _scale_term;
                int img_idx = Npix_to_model * i_sig + pixIdx;
                d_diffuse_sigma_images(img_idx) = val;
            }
        }

        // update eta derivative image
        if (refine_flag & REFINE_ETA) {
            for (int i_eta = 0; i_eta < 3; i_eta++) {
                if (i_eta > 0 && !aniso_eta)
                    continue;
                int idx = pixIdx + Npix_to_model * i_eta;
                CUDAREAL value = _scale_term * dI.eta[i_eta];
                CUDAREAL value2 = _scale_term * dI2.eta[i_eta];
                d_eta_images(idx) = value;
                d2_eta_images(idx) = value2;
            }
        }  // end eta deriv image increment

        // update the lambda derivative images
        for (int i_lam = 0; i_lam < 2; i_lam++) {
            if (refine_flag & (REFINE_LAMBDA1 << i_lam)) {
                CUDAREAL value = _scale_term * dI.lambda[i_lam];
                CUDAREAL value2 = _scale_term * dI2.lambda[i_lam];
                int idx = i_lam * Npix_to_model + pixIdx;
                d_lambda_images(idx) = value;
                // d2_lambda_images(idx) = value2;
            }
        }  // end lambda deriv image increment

        for (int i_pan_rot = 0; i_pan_rot < 3; i_pan_rot++) {
            if (refine_flag & (REFINE_PANEL_ROT1 << i_pan_rot)) {
                CUDAREAL value = _scale_term * dI.pan_rot[i_pan_rot];
                CUDAREAL value2 = _scale_term * dI2.pan_rot[i_pan_rot];
                int idx = i_pan_rot * Npix_to_model + pixIdx;
                d_panel_rot_images(idx) = value;
                // d2_panel_rot_images(idx) = value2;
            }
        }  // end panel rot deriv image increment

        for (int i_pan_orig = 0; i_pan_orig < 3; i_pan_orig++) {
            if (refine_flag & (REFINE_PANEL_ORIGIN1 << i_pan_orig)) {
                CUDAREAL value = _scale_term * dI.pan_orig[i_pan_orig];
                CUDAREAL value2 = _scale_term * dI2.pan_orig[i_pan_orig];
                int idx = i_pan_orig * Npix_to_model + pixIdx;
                d_panel_orig_images(idx) = value;
                // d2_panel_orig_images(idx) = value2;
            }
        }  // end panel orig deriv image increment
    });    // end pixIdx loop

}  // END of GPU kernel

//////////////////////////////////////////////////////////////////////////////////

template <
    bool printout,
    bool complex_miller,
    bool compute_curvatures,
    uint32_t refine_flag,
    bool use_diffuse,
    bool save_wavelenimage,
    bool Fhkl_gradient_mode,
    bool Fhkl_errors_mode,
    bool using_trusted_mask,
    bool Fhkl_channels_empty,
    bool Fhkl_have_scale_factors>
void kokkos_sum_over_steps(
    int Npix_to_model,
    vector_uint_t panels_fasts_slows,
    vector_cudareal_t floatimage,
    vector_cudareal_t wavelenimage,
    vector_cudareal_t d_Umat_images,
    vector_cudareal_t d2_Umat_images,
    vector_cudareal_t d_Bmat_images,
    vector_cudareal_t d2_Bmat_images,
    vector_cudareal_t d_Ncells_images,
    vector_cudareal_t d2_Ncells_images,
    vector_cudareal_t d_fcell_images,
    vector_cudareal_t d2_fcell_images,
    vector_cudareal_t d_eta_images,
    vector_cudareal_t d2_eta_images,
    vector_cudareal_t d_lambda_images,
    vector_cudareal_t d2_lambda_images,
    vector_cudareal_t d_panel_rot_images,
    vector_cudareal_t d2_panel_rot_images,
    vector_cudareal_t d_panel_orig_images,
    vector_cudareal_t d2_panel_orig_images,
    vector_cudareal_t d_fp_fdp_images,
    vector_manager_t manager_dI,
    vector_manager_t manager_dI2,
    const int Nsteps,
    int printout_fpixel,
    int printout_spixel,
    /*bool printout,*/
    CUDAREAL default_F,
    int oversample,
    bool oversample_omega,
    CUDAREAL subpixel_size,
    CUDAREAL pixel_size,
    CUDAREAL detector_thickstep,
    CUDAREAL detector_thick,
    const vector_cudareal_t close_distances,
    CUDAREAL detector_attnlen,
    int detector_thicksteps,
    int sources,
    int phisteps,
    int mosaic_domains,
    bool use_lambda_coefficients,
    CUDAREAL lambda0,
    CUDAREAL lambda1,
    KOKKOS_MAT3 eig_U,
    KOKKOS_MAT3 eig_O,
    KOKKOS_MAT3 eig_B,
    KOKKOS_MAT3 RXYZ,
    vector_vec3_t dF_vecs,
    vector_vec3_t dS_vecs,
    const vector_mat3_t UMATS_RXYZ,
    vector_mat3_t UMATS_RXYZ_prime,
    vector_mat3_t UMATS_RXYZ_dbl_prime,
    vector_mat3_t RotMats,
    vector_mat3_t dRotMats,
    vector_mat3_t d2RotMats,
    vector_mat3_t UMATS,
    vector_mat3_t dB_mats,
    vector_mat3_t dB2_mats,
    vector_mat3_t Amatrices,
    const vector_cudareal_t source_X,
    const vector_cudareal_t source_Y,
    const vector_cudareal_t source_Z,
    const vector_cudareal_t source_lambda,
    const vector_cudareal_t source_I,
    CUDAREAL kahn_factor,
    CUDAREAL Na,
    CUDAREAL Nb,
    CUDAREAL Nc,
    CUDAREAL Nd,
    CUDAREAL Ne,
    CUDAREAL Nf,
    CUDAREAL phi0,
    CUDAREAL phistep,
    KOKKOS_VEC3 spindle_vec,
    KOKKOS_VEC3 polarization_axis,
    int h_range,
    int k_range,
    int l_range,
    int h_max,
    int h_min,
    int k_max,
    int k_min,
    int l_max,
    int l_min,
    CUDAREAL dmin,
    CUDAREAL fudge,
    /*bool complex_miller,*/
    int verbose,
    bool only_save_omega_kahn,
    bool isotropic_ncells,
    /*bool compute_curvatures,*/
    const vector_cudareal_t FhklLinear,
    const vector_cudareal_t Fhkl2Linear,
    /*const uint32_t refine_flag,*/
    // vector_bool_t refine_Bmat,
    // vector_bool_t refine_Ncells,
    // bool refine_Ncells_def,
    // vector_bool_t refine_panel_origin,
    // vector_bool_t refine_panel_rot,
    // bool refine_fcell,
    // vector_bool_t refine_lambda,
    // bool refine_eta,
    // vector_bool_t refine_Umat,
    const vector_cudareal_t fdet_vectors,
    const vector_cudareal_t sdet_vectors,
    const vector_cudareal_t odet_vectors,
    const vector_cudareal_t pix0_vectors,
    bool nopolar,
    bool point_pixel,
    CUDAREAL fluence,
    CUDAREAL r_e_sqr,
    CUDAREAL spot_scale,
    int Npanels,
    bool aniso_eta,
    bool no_Nabc_scale,
    const vector_cudareal_t fpfdp,
    const vector_cudareal_t fpfdp_derivs,
    const vector_cudareal_t atom_data,
    int num_atoms,
    // bool refine_fp_fdp,
    const vector_int_t nominal_hkl,
    bool use_nominal_hkl,
    KOKKOS_MAT3 anisoU,
    KOKKOS_MAT3 anisoG,
    KOKKOS_MAT3 rotate_principal_axes,
    /*bool use_diffuse,*/
    vector_cudareal_t d_diffuse_gamma_images,
    vector_cudareal_t d_diffuse_sigma_images,
    // bool refine_diffuse,
    bool gamma_miller_units,
    // bool refine_Icell,
    /*bool save_wavelenimage,*/
    int laue_group_num,
    int stencil_size,
    /*bool Fhkl_gradient_mode,*/
    /*bool Fhkl_errors_mode,*/
    /*bool using_trusted_mask,*/
    /*bool Fhkl_channels_empty,*/
    /*bool Fhkl_have_scale_factors,*/
    int Num_ASU,
    const vector_cudareal_t data_residual,
    const vector_cudareal_t data_variance,
    const vector_int_t data_freq,
    const vector_bool_t data_trusted,
    const vector_int_t FhklLinear_ASUid,
    const vector_int_t Fhkl_channels,
    const vector_cudareal_t Fhkl_scale,
    vector_cudareal_t Fhkl_scale_deriv,
    bool gaussian_star_shape, bool square_shape) {  // BEGIN GPU kernel

    const KOKKOS_MAT3 Bmat_realspace = eig_B * 1e10;
    const KOKKOS_MAT3 eig_Otranspose = eig_O.transpose();
    const KOKKOS_MAT3 Amat_init = eig_U * Bmat_realspace * eig_Otranspose;
    const KOKKOS_MAT3 Ainv = eig_U*(Bmat_realspace.transpose().inverse())* (eig_O.inverse());
    const CUDAREAL reciprocal_space_volume = 8*M_PI*M_PI*M_PI*Ainv.determinant();
    const KOKKOS_MAT3 _NABC {Na, Nd, Nf, Nd, Nb, Ne, Nf, Ne, Nc};
    const double NABC_det = _NABC.determinant();  // TODO is this slow ?
    const double NABC_det_sq = NABC_det * NABC_det;
    const CUDAREAL C = 2 / 0.63 * fudge;
    const CUDAREAL two_C = 2 * C;
    KOKKOS_MAT3 anisoG_local;
    CUDAREAL anisoG_determ = 0;
    KOKKOS_MAT3 anisoU_local;
    const CUDAREAL _tmpfac = M_PI * 0.63 / fudge;
    const CUDAREAL diffuse_scale = reciprocal_space_volume * sqrt(_tmpfac*_tmpfac*_tmpfac);
    // ***NEEDS UPDATE: use legacy API for passing diffuse scale as KOKKOS_MAT3
    KOKKOS_MAT3 diffuse_scale_mat3(diffuse_scale,0,0,0,0,0,0,0,0);
    vector_mat3_t laue_mats = vector_mat3_t("laue_mats", 24);
    vector_vec3_t dG_dgam = vector_vec3_t("dG_dgam", 3);
    vector_cudareal_t dG_trace = vector_cudareal_t("dG_trace", 3);
    int num_laue_mats = 0;
    int dhh = 0, dkk = 0, dll = 0;
    CUDAREAL gx = spindle_vec[0];
    CUDAREAL gy = spindle_vec[1];
    CUDAREAL gz = spindle_vec[2];

    Kokkos::View<KOKKOS_MAT3*[3]> UMATS_prime("UMATS_prime", mosaic_domains);
    Kokkos::View<KOKKOS_MAT3*[3]> UMATS_dbl_prime("UMATS_dbl_prime", mosaic_domains);
    Kokkos::View<KOKKOS_MAT3*[6]> BMATS_prime("BMATS_prime", mosaic_domains);
    Kokkos::View<KOKKOS_MAT3*[6]> BMATS_dbl_prime("BMATS_dbl_prime", mosaic_domains);

    Kokkos::parallel_for("prepare_UMATS", mosaic_domains, KOKKOS_LAMBDA(const int& _mos_tic) {
        const KOKKOS_MAT3 UBOt = Amat_init;
        UMATS_prime(_mos_tic, 0) = _NABC * (UMATS(_mos_tic) * dRotMats(0) * RotMats(1) * RotMats(2) * UBOt).transpose();
        UMATS_prime(_mos_tic, 1) = _NABC * (UMATS(_mos_tic) * RotMats(0) * dRotMats(1) * RotMats(2) * UBOt).transpose();
        UMATS_prime(_mos_tic, 2) = _NABC * (UMATS(_mos_tic) * RotMats(0) * RotMats(1) * dRotMats(2) * UBOt).transpose();

        UMATS_dbl_prime(_mos_tic, 0) = _NABC * (UMATS(_mos_tic) * d2RotMats(0) * RotMats(1) * RotMats(2) * UBOt).transpose();
        UMATS_dbl_prime(_mos_tic, 1) = _NABC * (UMATS(_mos_tic) * RotMats(0) * d2RotMats(1) * RotMats(2) * UBOt).transpose();
        UMATS_dbl_prime(_mos_tic, 2) = _NABC * (UMATS(_mos_tic) * RotMats(0) * RotMats(1) * d2RotMats(2) * UBOt).transpose();

        for (int i_uc=0; i_uc<6; i_uc++) {
            BMATS_prime(_mos_tic, i_uc) = _NABC * (UMATS_RXYZ(_mos_tic) * eig_U * dB_mats(i_uc) * eig_O.transpose()).transpose();
            BMATS_dbl_prime(_mos_tic, i_uc) = _NABC * (UMATS_RXYZ(_mos_tic) * eig_U * dB2_mats(i_uc) * eig_O.transpose()).transpose();
        }
    });

    if (use_diffuse){
        anisoG_local = anisoG;
        anisoU_local = anisoU;

        if (laue_group_num < 1 || laue_group_num >14 ){
            throw std::string("Laue group number not in range 1-14");
        }

        if (gamma_miller_units){
            anisoG_local = anisoG_local * Bmat_realspace;
        }
        Kokkos::parallel_reduce("prepare diffuse mats", 1, KOKKOS_LAMBDA (const int& i, int& num_laue_mats_temp){
            num_laue_mats_temp = gen_laue_mats(laue_group_num, laue_mats, rotate_principal_axes);
            // KOKKOS_MAT3 rotate_principal_axes;
            // rotate_principal_axes << 0.70710678,  -0.70710678,  0., 0.70710678,  0.70710678,  0., 0.,  0., 1.;

            for ( int iL = 0; iL < num_laue_mats_temp; iL++ ){
                laue_mats(iL) = Ainv * laue_mats(iL) * rotate_principal_axes;
            }
            // printf("Bmat =");
            // for (int i=0; i<9; ++i) {
            //     printf(" %g", Bmat_realspace[i]);
            // }
            // printf("\n");
            const KOKKOS_MAT3 Ginv = anisoG_local.inverse();
            // printf("Ginv =");
            // for (int i=0; i<9; ++i) {
            //     printf(" %g", Ginv[i]);
            // }
            // printf("\n");
            const KOKKOS_MAT3 dG = Bmat_realspace * Ginv;
            // printf("dG   =");
            // for (int i=0; i<9; ++i) {
            //     printf(" %g", dG[i]);
            // }
            // printf("\n");
            for (int i_gam=0; i_gam<3; i_gam++){
                if (gamma_miller_units) {
                    dG_dgam(i_gam) = KOKKOS_VEC3(Bmat_realspace(i_gam, 0), Bmat_realspace(i_gam, 1), Bmat_realspace(i_gam, 2));
                } else {
                    dG_dgam(i_gam)[i_gam] = 1;
                }
                KOKKOS_MAT3 temp_dgam;
                temp_dgam(i_gam, 0) = dG_dgam(i_gam)[0];
                temp_dgam(i_gam, 1) = dG_dgam(i_gam)[1];
                temp_dgam(i_gam, 2) = dG_dgam(i_gam)[2];
                dG_trace(i_gam) = (Ginv*temp_dgam).trace();
                // printf("TRACE %g\n", dG_trace(i_gam));
                // printf("dgam =");
                // for (int i=0; i<9; ++i) {
                //     printf(" %g", temp_dgam[i]);
                // }
                // printf("\n");

                // dG(i_gam, i_gam);
            }
        }, num_laue_mats);
        anisoG_determ = anisoG_local.determinant();
        dhh = dkk = dll = stencil_size; // Limits of stencil for diffuse calc
    }
    const KOKKOS_VEC3 dHH (dhh, dkk, dll);

    const CUDAREAL overall_scale = r_e_sqr * spot_scale * fluence / Nsteps;

    const CUDAREAL detector_attnlen_r = (detector_attnlen>0) ? 1 / detector_attnlen : 0;

    Kokkos::parallel_for(
        "sum_over_steps", Npix_to_model, KOKKOS_LAMBDA(const int& pixIdx) {

        if (using_trusted_mask) {
            if (!data_trusted(pixIdx))
                return;
        }
        const int _pid = panels_fasts_slows(pixIdx * 3);
        const int _fpixel = panels_fasts_slows(pixIdx * 3 + 1);
        const int _spixel = panels_fasts_slows(pixIdx * 3 + 2);

        CUDAREAL Fhkl_deriv_coef=0;
        CUDAREAL Fhkl_hessian_coef=0;
        if (Fhkl_gradient_mode) {
            CUDAREAL u = data_residual(pixIdx);
            CUDAREAL one_by_v = 1/data_variance(pixIdx);
            CUDAREAL Gterm = 1 - 2*u - u*u*one_by_v;
            Fhkl_deriv_coef = 0.5 * Gterm*one_by_v / data_freq(pixIdx);
            if (Fhkl_errors_mode) {
                Fhkl_hessian_coef = -0.5*one_by_v*(one_by_v*Gterm - 2  - 2*u*one_by_v -u*u*one_by_v*one_by_v)/data_freq(pixIdx);
            }
        }

        // int fcell_idx=1;
        int nom_h = 0, nom_k = 0, nom_l = 0;
        if (use_nominal_hkl) {
            nom_h = nominal_hkl(pixIdx * 3);
            nom_k = nominal_hkl(pixIdx * 3 + 1);
            nom_l = nominal_hkl(pixIdx * 3 + 2);
        }
        CUDAREAL close_distance = close_distances(_pid);

        // reset photon count for this pixel
        double _I = 0;
        double Ilambda = 0;

        kokkos_manager dI, dI2;
        dI.reset();
        dI2.reset();

        for (int _subS = 0; _subS < oversample; ++_subS) {
            for (int _subF = 0; _subF < oversample; ++_subF) {
                // absolute mm position on detector (relative to its origin)
                CUDAREAL _Fdet =
                    subpixel_size * (_fpixel * oversample + _subF) + subpixel_size / 2.0;
                CUDAREAL _Sdet =
                    subpixel_size * (_spixel * oversample + _subS) + subpixel_size / 2.0;

                // assume "distance" is to the front of the detector sensor layer
                int pid_x = _pid * 3;
                int pid_y = _pid * 3 + 1;
                int pid_z = _pid * 3 + 2;


                CUDAREAL fx = fdet_vectors(pid_x);
                CUDAREAL fy = fdet_vectors(pid_y);
                CUDAREAL fz = fdet_vectors(pid_z);
                CUDAREAL sx = sdet_vectors(pid_x);
                CUDAREAL sy = sdet_vectors(pid_y);
                CUDAREAL sz = sdet_vectors(pid_z);
                CUDAREAL ox = odet_vectors(pid_x);
                CUDAREAL oy = odet_vectors(pid_y);
                CUDAREAL oz = odet_vectors(pid_z);
                CUDAREAL px = pix0_vectors(pid_x);
                CUDAREAL py = pix0_vectors(pid_y);
                CUDAREAL pz = pix0_vectors(pid_z);
                KOKKOS_VEC3 _o_vec(ox, oy, oz);

                for (int _thick_tic = 0; _thick_tic < detector_thicksteps; ++_thick_tic) {

                    CUDAREAL _Odet = _thick_tic * detector_thickstep;

                    CUDAREAL pixposX = _Fdet * fx + _Sdet * sx + _Odet * ox + px;
                    CUDAREAL pixposY = _Fdet * fy + _Sdet * sy + _Odet * oy + py;
                    CUDAREAL pixposZ = _Fdet * fz + _Sdet * sz + _Odet * oz + pz;
                    KOKKOS_VEC3 _pixel_pos(pixposX, pixposY, pixposZ);

                    CUDAREAL _airpath_r = 1 / _pixel_pos.length();
                    KOKKOS_VEC3 _diffracted = _pixel_pos.get_unit_vector();

                    const CUDAREAL close_distance = close_distances(_pid);

                    // solid angle subtended by a pixel: (pix/airpath)^2*cos(2theta)
                    CUDAREAL _omega_pixel = pixel_size * pixel_size * _airpath_r * _airpath_r *
                                            close_distance * _airpath_r;

                    // option to turn off obliquity effect, inverse-square-law only
                    if (point_pixel)
                        _omega_pixel = _airpath_r * _airpath_r;

                    // now calculate detector thickness effects
                    CUDAREAL _capture_fraction = 1;

                    CUDAREAL previous_layer = 1.0;
                    if (detector_thick > 0.0 && detector_attnlen_r > 0.0) {
                        // inverse of effective thickness increase
                        KOKKOS_VEC3 _o_vec(ox, oy, oz);
                        CUDAREAL _parallax = _diffracted.dot(_o_vec);
                        CUDAREAL current_layer = ::Kokkos::exp(
                                                -(_thick_tic + 1) * detector_thickstep *
                                                detector_attnlen_r / _parallax);
                        _capture_fraction = previous_layer - current_layer;
                        previous_layer = current_layer;
                    }

                    for (int _source = 0; _source < sources; ++_source) {

                        KOKKOS_VEC3 _incident(
                            -source_X(_source), -source_Y(_source), -source_Z(_source));
                        CUDAREAL _lambda = source_lambda(_source);
                        CUDAREAL sI = source_I(_source);
                        CUDAREAL lambda_ang = _lambda * 1e10;
                        if (use_lambda_coefficients) {
                            lambda_ang = lambda0 + lambda1 * lambda_ang;
                            _lambda = lambda_ang * 1e-10;
                        }

                        // polarization
                        CUDAREAL polar_for_Fhkl_grad=1;
                        if (!nopolar && Fhkl_gradient_mode){

                            // component of diffracted unit vector along incident beam unit vector
                            CUDAREAL cos2theta = _incident.dot(_diffracted);
                            CUDAREAL cos2theta_sqr = cos2theta*cos2theta;
                            CUDAREAL sin2theta_sqr = 1-cos2theta_sqr;

                            CUDAREAL cos2psi=1;
                            if(kahn_factor != 0.0){
                                // cross product to get "vertical" axis that is orthogonal to the cannonical "polarization"
                                KOKKOS_VEC3 B_in = polarization_axis.cross(_incident);
                                // cross product with incident beam to get E-vector direction
                                KOKKOS_VEC3 E_in = _incident.cross(B_in);
                                // get components of diffracted ray projected onto the E-B plane
                                CUDAREAL _kEi = _diffracted.dot(E_in);
                                CUDAREAL _kBi = _diffracted.dot(B_in);
                                // compute the angle of the diffracted ray projected onto the incident E-B plane
                                // calculate cos(2 * atan2(_kBi, _kEi))
                                if (_kEi!=0) {
                                    CUDAREAL ratio = _kBi / _kEi;
                                    cos2psi = (1 - ratio*ratio) / (1 + ratio*ratio);
                                } else {
                                    cos2psi = -1;
                                }
                            }
                            // correction for polarized incident beam
                            polar_for_Fhkl_grad = 0.5*(1.0 + cos2theta_sqr - kahn_factor*cos2psi*sin2theta_sqr);
                        }
                        KOKKOS_VEC3 _scattering = (_diffracted - _incident) / _lambda;

                        KOKKOS_VEC3 q_vec = _scattering * 1e-10;

                        // TODO rename
                        CUDAREAL texture_scale = _capture_fraction * _omega_pixel * sI;
                        for (int _phi_tic=0; _phi_tic<phisteps; ++_phi_tic){
                            KOKKOS_MAT3 Rphi;
                            CUDAREAL phi = phi0 + phistep*_phi_tic;
                            if (phi != 0){
                                CUDAREAL c = cos(phi);
                                CUDAREAL omc = 1-c;
                                CUDAREAL s = sin(phi);
                                Rphi = KOKKOS_MAT3{c + gx*gx*omc,    gx*gy*omc-gz*s,   gx*gz*omc+gy*s,
                                                 gy*gx*omc + gz*s,   c + gy*gy*omc,   gy*gz*omc - gx*s,
                                                 gz*gx*omc - gy*s,  gz*gy*omc + gx*s, c + gz*gz*omc};
                            }

                        for (int _mos_tic = 0; _mos_tic < mosaic_domains; ++_mos_tic) {
                            KOKKOS_MAT3 UBO = Amatrices(_mos_tic);
                            if (phi != 0){
                                KOKKOS_MAT3 Um = UMATS_RXYZ(_mos_tic);
                                UBO = UBO*Um*Rphi.transpose()*Um.transpose();
                            }

                            KOKKOS_VEC3 H_vec = UBO * q_vec;
                            CUDAREAL _h = H_vec[0];
                            CUDAREAL _k = H_vec[1];
                            CUDAREAL _l = H_vec[2];

                            int _h0 = ceil(_h - 0.5);
                            int _k0 = ceil(_k - 0.5);
                            int _l0 = ceil(_l - 0.5);

                            KOKKOS_VEC3 H0(_h0, _k0, _l0);

                            KOKKOS_VEC3 delta_H = H_vec - H0;
                            KOKKOS_VEC3 V = _NABC * delta_H;
                            CUDAREAL _hrad_sqr = V.length_sqr();
                            CUDAREAL I0=0;
                            if(square_shape){
                               CUDAREAL F_latt = 1.0;
                               /* xtal is a paralelpiped */
                               if(Na>1)
                                   F_latt *= sincg(M_PI*_h,Na);
                               if(Nb>1)
                                   F_latt *= sincg(M_PI*_k,Nb);
                               if(Nc>1)
                                   F_latt *= sincg(M_PI*_l,Nc);
                               I0 = F_latt*F_latt;
                            }
                            else{ // using a gaussian model
                                CUDAREAL exparg;
                                if (gaussian_star_shape){
                                    // TODO: can precompute xtal_vol to save time ... but prob not necessary as gaussian star model only used for forward calc
                                    KOKKOS_VEC3 A {UBO[0], UBO[1], UBO[2]};
                                    KOKKOS_VEC3 B {UBO[3], UBO[4], UBO[5]};
                                    KOKKOS_VEC3 C {UBO[6], UBO[7], UBO[8]};
                                    CUDAREAL cell_vol = A.dot(B.cross(C));
                                    CUDAREAL xtal_size_sq = pow(NABC_det*cell_vol, CUDAREAL(2)/CUDAREAL(3));
                                    KOKKOS_MAT3 Ainv = UBO.inverse();
                                    KOKKOS_VEC3 delta_Q = Ainv*delta_H;
                                    CUDAREAL rad_star_sqr = delta_Q.dot(delta_Q)*xtal_size_sq;
                                    exparg = rad_star_sqr*1.9*fudge ;
                                }
                                else
                                    exparg = _hrad_sqr * C / 2;

                                if (exparg < 35)
                                    if (no_Nabc_scale)
                                        I0 = ::Kokkos::exp(-2 * exparg);
                                    else
                                        I0 = (NABC_det_sq) *
                                                ::Kokkos::exp(-2 * exparg);
                            }

                            // are we doing diffuse scattering
                            CUDAREAL step_diffuse_param[6] = {0, 0, 0, 0, 0, 0};
                            if (use_diffuse) {
                              // ***NEEDS UPDATE: use legacy API for passing diffuse scale as KOKKOS_MAT3
                                calc_diffuse_at_hkl(H_vec,H0,dHH,h_min,k_min,l_min,h_max,k_max,l_max,h_range,k_range,l_range,diffuse_scale_mat3,FhklLinear,num_laue_mats,laue_mats,anisoG_local,dG_trace,anisoG_determ,anisoU_local,dG_dgam,(refine_flag & REFINE_DIFFUSE)>0,&I0,step_diffuse_param);
                            } // end s_use_diffuse outer

                            CUDAREAL _F_cell = default_F;
                            CUDAREAL _F_cell2 = 0;
                            int i_hklasu=0;

                            if ((_h0 <= h_max) && (_h0 >= h_min) &&
                                (_k0 <= k_max) && (_k0 >= k_min) &&
                                (_l0 <= l_max) && (_l0 >= l_min)) {
                                int Fhkl_linear_index = (_h0 - h_min) * k_range * l_range +
                                                        (_k0 - k_min) * l_range + (_l0 - l_min);
                                //_F_cell = __ldg(&FhklLinear[Fhkl_linear_index]);
                                _F_cell = FhklLinear(Fhkl_linear_index);
                                // if (complex_miller) _F_cell2 =
                                // __ldg(&Fhkl2Linear[Fhkl_linear_index]);
                                if (complex_miller)
                                    _F_cell2 = Fhkl2Linear(Fhkl_linear_index);
                                if (Fhkl_have_scale_factors)
                                    i_hklasu = FhklLinear_ASUid(Fhkl_linear_index);
                            }

                            CUDAREAL c_deriv_Fcell = 0;
                            CUDAREAL d_deriv_Fcell = 0;
                            if (complex_miller) {
                                CUDAREAL c_deriv_Fcell_real = 0;
                                CUDAREAL c_deriv_Fcell_imag = 0;
                                CUDAREAL d_deriv_Fcell_real = 0;
                                CUDAREAL d_deriv_Fcell_imag = 0;
                                if (num_atoms > 0) {
                                    CUDAREAL S_2 = (q_vec[0] * q_vec[0] +
                                                    q_vec[1] * q_vec[1] +
                                                    q_vec[2] * q_vec[2]);

                                    // fp is always followed by the fdp value
                                    CUDAREAL val_fp = fpfdp(2 * _source);
                                    CUDAREAL val_fdp = fpfdp(2 * _source + 1);

                                    CUDAREAL c_deriv_prime = 0;
                                    CUDAREAL c_deriv_dblprime = 0;
                                    CUDAREAL d_deriv_prime = 0;
                                    CUDAREAL d_deriv_dblprime = 0;
                                    if (refine_flag & REFINE_FP_FDP) {
                                        //   currently only supports two parameter model
                                        int d_idx = 2 * _source;
                                        c_deriv_prime = fpfdp_derivs(d_idx);
                                        c_deriv_dblprime = fpfdp_derivs(d_idx + 1);
                                        d_deriv_prime = fpfdp_derivs(d_idx + 2 * sources);
                                        d_deriv_dblprime =
                                            fpfdp_derivs(d_idx + 1 + 2 * sources);
                                    }

                                    for (int i_atom = 0; i_atom < num_atoms; i_atom++) {
                                        // fractional atomic coordinates
                                        CUDAREAL atom_x = atom_data(i_atom * 5);
                                        CUDAREAL atom_y = atom_data(i_atom * 5 + 1);
                                        CUDAREAL atom_z = atom_data(i_atom * 5 + 2);
                                        CUDAREAL B = atom_data(i_atom * 5 + 3);  // B factor
                                        B = ::Kokkos::exp(
                                            -B * S_2 / 4.0);  // TODO: speed me up?
                                        CUDAREAL occ = atom_data(i_atom * 5 + 4);  // occupancy
                                        CUDAREAL r_dot_h =
                                            _h0 * atom_x + _k0 * atom_y + _l0 * atom_z;
                                        CUDAREAL phase = 2 * M_PI * r_dot_h;
                                        CUDAREAL s_rdoth = ::Kokkos::sin(phase);
                                        CUDAREAL c_rdoth = ::Kokkos::cos(phase);
                                        CUDAREAL Bocc = B * occ;
                                        CUDAREAL BC = B * c_rdoth;
                                        CUDAREAL BS = B * s_rdoth;
                                        CUDAREAL real_part = BC * val_fp - BS * val_fdp;
                                        CUDAREAL imag_part = BS * val_fp + BC * val_fdp;
                                        _F_cell += real_part;
                                        _F_cell2 += imag_part;
                                        if (refine_flag & REFINE_FP_FDP) {
                                            c_deriv_Fcell_real +=
                                                BC * c_deriv_prime - BS * c_deriv_dblprime;
                                            c_deriv_Fcell_imag +=
                                                BS * c_deriv_prime + BC * c_deriv_dblprime;

                                            d_deriv_Fcell_real +=
                                                BC * d_deriv_prime - BS * d_deriv_dblprime;
                                            d_deriv_Fcell_imag +=
                                                BS * d_deriv_prime + BC * d_deriv_dblprime;
                                        }
                                    }
                                }
                                CUDAREAL Freal = _F_cell;
                                CUDAREAL Fimag = _F_cell2;
                                _F_cell =
                                    ::Kokkos::sqrt(Freal * Freal + Fimag * Fimag);
                                if (refine_flag & REFINE_FP_FDP) {
                                    c_deriv_Fcell =
                                        Freal * c_deriv_Fcell_real + Fimag * c_deriv_Fcell_imag;
                                    d_deriv_Fcell =
                                        Freal * d_deriv_Fcell_real + Fimag * d_deriv_Fcell_imag;
                                }
                            }
                            if (!oversample_omega && ! Fhkl_gradient_mode)
                                _omega_pixel = 1;

                            CUDAREAL _I_cell = _F_cell;
                            if (!(refine_flag & REFINE_ICELL))
                                _I_cell *= _F_cell;
                            CUDAREAL hkl=1;
                            int Fhkl_channel=0;
                            if (! Fhkl_channels_empty)
                                Fhkl_channel = Fhkl_channels(_source);
                            if (Fhkl_have_scale_factors)
                                hkl = Fhkl_scale(i_hklasu + Fhkl_channel*Num_ASU);
                            if (Fhkl_gradient_mode){
                                CUDAREAL Fhkl_deriv_scale = overall_scale*polar_for_Fhkl_grad;
                                CUDAREAL I_noFcell=texture_scale*I0;
                                CUDAREAL dfhkl = I_noFcell*_I_cell * Fhkl_deriv_scale;
                                CUDAREAL grad_incr = dfhkl*Fhkl_deriv_coef;
                                int fhkl_grad_idx=i_hklasu + Fhkl_channel*Num_ASU;

                                if (Fhkl_errors_mode){
                                    // here we hi-kack the Fhkl_scale_deriv array, if computing errors, in order to store the hessian terms
                                    // if we are getting the hessian terms, we no longer need the  gradients (e.g. by this point we are done refininig)
                                    CUDAREAL hessian_incr = Fhkl_hessian_coef*dfhkl*dfhkl;
                                    ::Kokkos::atomic_add(&Fhkl_scale_deriv(fhkl_grad_idx), hessian_incr);
                                }
                                else{
                                    ::Kokkos::atomic_add(&Fhkl_scale_deriv(fhkl_grad_idx), grad_incr);
                                }
                                continue;
                            }

                            CUDAREAL _I_total = hkl*_I_cell *I0;
                            CUDAREAL Iincrement = _I_total * texture_scale;
                            _I += Iincrement;
                            if (save_wavelenimage)
                                Ilambda += Iincrement * lambda_ang;

                            if (refine_flag & REFINE_DIFFUSE) {
                                CUDAREAL step_scale = texture_scale * _F_cell * _F_cell;
                                for (int i_diff = 0; i_diff < 6; i_diff++) {
                                    dI.diffuse[i_diff] +=
                                        step_scale * step_diffuse_param[i_diff];
                                }
                            }

                            //*************************************************
                            // START REFINEMENT

                            if (refine_flag & REFINE_FP_FDP) {
                                CUDAREAL I_noFcell = texture_scale * I0;
                                dI.fp_fdp[0] += 2 * I_noFcell * (c_deriv_Fcell);
                                dI.fp_fdp[1] += 2 * I_noFcell * (d_deriv_Fcell);
                            }

                            if (verbose > 3)
                                printf(
                                    "hkl= %f %f %f  hkl1= %d %d %d  Fcell=%f\n", _h, _k, _l,
                                    _h0, _k0, _l0, _F_cell);

                            KOKKOS_MAT3 UBOt;
                             if (refine_flag & (REFINE_UMAT | REFINE_ETA)) {
                                UBOt = Amat_init;
                                if (phi != 0)
                                    UBOt = Rphi*UBOt;
                            }
                            if (refine_flag & REFINE_UMAT1) {
                                const KOKKOS_VEC3 dV = UMATS_prime(_mos_tic, 0) * q_vec;
                                const CUDAREAL V_dot_dV = V.dot(dV);
                                const CUDAREAL value = -two_C * V_dot_dV * Iincrement;
                                CUDAREAL value2 = 0;
                                if (compute_curvatures) {
                                    const CUDAREAL dV_dot_dV = dV.length_sqr();
                                    const CUDAREAL dV2_dot_V = V.dot(UMATS_dbl_prime(_mos_tic, 0)*q_vec);
                                    value2 = two_C * (two_C * V_dot_dV * V_dot_dV - dV2_dot_V - dV_dot_dV) * Iincrement;
                                }
                                dI.rot[0] += value;
                                dI2.rot[0] += value2;
                            }
                            if (refine_flag & REFINE_UMAT2) {
                                KOKKOS_VEC3 dV = UMATS_prime(_mos_tic, 1) * q_vec;
                                CUDAREAL V_dot_dV = V.dot(dV);
                                CUDAREAL value = -two_C * V_dot_dV * Iincrement;

                                CUDAREAL value2 = 0;
                                if (compute_curvatures) {
                                    const CUDAREAL dV_dot_dV = dV.length_sqr();
                                    CUDAREAL dV2_dot_V = V.dot(UMATS_dbl_prime(_mos_tic, 1)*q_vec);
                                    value2 = two_C * (two_C * V_dot_dV * V_dot_dV - dV2_dot_V - dV_dot_dV) * Iincrement;
                                }
                                dI.rot[1] += value;
                                dI2.rot[1] += value2;
                            }
                            if (refine_flag & REFINE_UMAT3) {
                                KOKKOS_VEC3 dV = UMATS_prime(_mos_tic, 2) * q_vec;
                                CUDAREAL V_dot_dV = V.dot(dV);
                                CUDAREAL value = -two_C * V_dot_dV * Iincrement;

                                CUDAREAL value2 = 0;
                                if (compute_curvatures) {
                                    const CUDAREAL dV_dot_dV = dV.length_sqr();
                                    CUDAREAL dV2_dot_V = V.dot(UMATS_dbl_prime(_mos_tic, 2)*q_vec);
                                    value2 = two_C * (two_C * V_dot_dV * V_dot_dV - dV2_dot_V - dV_dot_dV) * Iincrement;
                                }
                                dI.rot[2] += value;
                                dI2.rot[2] += value2;
                            }
                            // Checkpoint for unit cell derivatives
                            for (int i_uc = 0; i_uc < 6; i_uc++) {
                                if (refine_flag & (REFINE_BMAT1 << i_uc)) {
                                    KOKKOS_VEC3 dV = BMATS_prime(_mos_tic, i_uc) * q_vec;
                                    CUDAREAL V_dot_dV = V.dot(dV);
                                    CUDAREAL value = -two_C * V_dot_dV * Iincrement;
                                    CUDAREAL value2 = 0;
                                    if (compute_curvatures) {
                                        const CUDAREAL dV_dot_dV = dV.length_sqr();
                                        CUDAREAL dV2_dot_V = V.dot(BMATS_dbl_prime(_mos_tic, i_uc)*q_vec);
                                        value2 = two_C * (two_C * V_dot_dV * V_dot_dV - dV2_dot_V - dV_dot_dV) * Iincrement;
                                    }
                                    dI.ucell[i_uc] += value;
                                    dI2.ucell[i_uc] += value2;
                                }
                            }  // end ucell deriv

                            // Checkpoint for Ncells manager
                            if (refine_flag & REFINE_NCELLS1) {
                                int num_ncell_deriv = 1;
                                if (!isotropic_ncells)
                                    num_ncell_deriv = 3;
                                for (int i_nc = 0; i_nc < num_ncell_deriv; i_nc++) {
                                    KOKKOS_MAT3 dN;
                                    dN(i_nc, i_nc) = 1;
                                    if (num_ncell_deriv == 1) {
                                        dN(0, 0) = 1;
                                        dN(1, 1) = 1;
                                        dN(2, 2) = 1;
                                    }
                                    CUDAREAL N_i = _NABC(i_nc, i_nc);
                                    KOKKOS_VEC3 dV_dN = dN.dot(delta_H);
                                    // TODO speedops: precompute these, store shared var
                                    // _NABC.inverse
                                    CUDAREAL determ_deriv = (_NABC.inverse().dot(dN)).trace();
                                    CUDAREAL deriv_coef = determ_deriv - C * (dV_dN.dot(V));
                                    CUDAREAL value = 2 * Iincrement * deriv_coef;
                                    CUDAREAL value2 = 0;
                                    if (compute_curvatures) {
                                        value2 = (-1 / N_i / N_i - C * (dV_dN.dot(dV_dN))) * 2 *
                                                    Iincrement;
                                        value2 += deriv_coef * 2 * value;
                                    }
                                    dI.Ncells[i_nc] += value;
                                    dI2.Ncells[i_nc] += value2;
                                }
                            }  // end Ncells manager deriv

                            if (refine_flag & REFINE_NCELLS_DEF) {
                                for (int i_nc = 3; i_nc < 6; i_nc++) {
                                    KOKKOS_MAT3 dN;
                                    if (i_nc == 3)
                                        dN = KOKKOS_MAT3{0, 1, 0, 1, 0, 0, 0, 0, 0};
                                    else if (i_nc == 4)
                                        dN = KOKKOS_MAT3{0, 0, 0, 0, 0, 1, 0, 1, 0};
                                    else
                                        dN = KOKKOS_MAT3{0, 0, 1, 0, 0, 0, 1, 0, 0};
                                    KOKKOS_VEC3 dV_dN = dN.dot(delta_H);
                                    // TODO speedops: precompute these
                                    CUDAREAL determ_deriv = (_NABC.inverse().dot(dN)).trace();
                                    CUDAREAL deriv_coef = determ_deriv - C * (dV_dN.dot(V));
                                    CUDAREAL value = 2 * Iincrement * deriv_coef;
                                    dI.Ncells[i_nc] += value;
                                    CUDAREAL value2 = 0;
                                    if (compute_curvatures) {
                                        value2 = deriv_coef * value;
                                        value2 += -2 * C * Iincrement * (dV_dN.dot(dV_dN));
                                        dI2.Ncells[i_nc] += value2;
                                    }
                                }
                            }

                            // Checkpoint for Origin manager
                            for (int i_pan_orig = 0; i_pan_orig < 3; i_pan_orig++) {
                                if (refine_flag & (REFINE_PANEL_ORIGIN1 << i_pan_orig)) {
                                    CUDAREAL per_k = _airpath_r;
                                    CUDAREAL per_k3 = pow(per_k, 3.);
                                    CUDAREAL per_k5 = pow(per_k, 5.);

                                    KOKKOS_MAT3 M = -two_C * (_NABC.dot(UBO)) / lambda_ang;
                                    KOKKOS_VEC3 dk;
                                    if (i_pan_orig == 0)
                                        dk = KOKKOS_VEC3{0, 0, 1};
                                    else if (i_pan_orig == 1)
                                        dk = KOKKOS_VEC3{1, 0, 0};
                                    else
                                        dk = KOKKOS_VEC3{0, 1, 0};

                                    CUDAREAL G = dk.dot(_pixel_pos);
                                    CUDAREAL pix2 = subpixel_size * subpixel_size;
                                    KOKKOS_VEC3 dk_hat = -per_k3 * G * _pixel_pos + per_k * dk;
                                    CUDAREAL coef = (M.dot(dk_hat)).dot(V);
                                    CUDAREAL coef2 =
                                        -3 * pix2 * per_k5 * G * (_o_vec.dot(_pixel_pos));
                                    coef2 += pix2 * per_k3 * (_o_vec.dot(dk));
                                    CUDAREAL value =
                                        coef * Iincrement + coef2 * Iincrement / _omega_pixel;

                                    dI.pan_orig[i_pan_orig] += value;
                                    dI2.pan_orig[i_pan_orig] += 0;

                                }  // end origin manager deriv
                            }

                            for (int i_pan_rot = 0; i_pan_rot < 3; i_pan_rot++) {
                                if (refine_flag & (REFINE_PANEL_ROT1 << i_pan_rot)) {
                                    CUDAREAL per_k = _airpath_r;
                                    CUDAREAL per_k3 = pow(per_k, 3.);
                                    CUDAREAL per_k5 = pow(per_k, 5.);
                                    KOKKOS_MAT3 M = -two_C * (_NABC.dot(UBO)) / lambda_ang;
                                    KOKKOS_VEC3 dk = _Fdet * (dF_vecs(_pid * 3 + i_pan_rot)) +
                                                _Sdet * (dS_vecs(_pid * 3 + i_pan_rot));
                                    CUDAREAL G = dk.dot(_pixel_pos);
                                    CUDAREAL pix2 = subpixel_size * subpixel_size;
                                    KOKKOS_VEC3 dk_hat = -per_k3 * G * _pixel_pos + per_k * dk;
                                    CUDAREAL coef = (M.dot(dk_hat)).dot(V);
                                    CUDAREAL coef2 =
                                        -3 * pix2 * per_k5 * G * (_o_vec.dot(_pixel_pos));
                                    coef2 += pix2 * per_k3 * (_o_vec.dot(dk));
                                    CUDAREAL value =
                                        coef * Iincrement + coef2 * Iincrement / _omega_pixel;

                                    dI.pan_rot[i_pan_rot] += value;
                                    dI2.pan_rot[i_pan_rot] += 0;
                                }
                            }

                            // checkpoint for Fcell manager
                            if (refine_flag & REFINE_FCELL) {
                                CUDAREAL value;
                                if (refine_flag & REFINE_ICELL)
                                    value = I0 * texture_scale;
                                else
                                    value = 2 * I0 * _F_cell *
                                            texture_scale;  // Iincrement/_F_cell ;
                                CUDAREAL value2 = 0;
                                if (compute_curvatures) {
                                    //    NOTE if _Fcell >0
                                    value2 = 2 * I0 * texture_scale;
                                }
                                // if (fcell_idx >=0 && fcell_idx <=2){
                                if (use_nominal_hkl) {
                                    if (_h0 == nom_h && _k0 == nom_k && _l0 == nom_l) {
                                        dI.fcell += value;
                                        dI2.fcell += value2;
                                    }
                                } else {
                                    dI.fcell += value;
                                    dI2.fcell += value2;
                                }
                            }  // end of fcell man deriv

                            // checkpoint for eta manager
                            if (refine_flag & REFINE_ETA) {
                                for (int i_eta = 0; i_eta < 3; i_eta++) {
                                    if (i_eta > 0 && !aniso_eta)
                                        continue;
                                    int mtic2 = _mos_tic + i_eta * mosaic_domains;
                                    KOKKOS_VEC3 DeltaH_deriv = (UMATS_RXYZ_prime(mtic2).dot(UBOt))
                                                            .transpose()
                                                            .dot(q_vec);
                                    // vector V is _Nabc*Delta_H
                                    KOKKOS_VEC3 dV = _NABC.dot(DeltaH_deriv);
                                    CUDAREAL V_dot_dV = V.dot(dV);
                                    CUDAREAL Iprime = -two_C * (V_dot_dV)*Iincrement;
                                    dI.eta[i_eta] += Iprime;
                                    CUDAREAL Idbl_prime = 0;
                                    if (compute_curvatures) {
                                        KOKKOS_VEC3 DeltaH_second_deriv =
                                            (UMATS_RXYZ_dbl_prime(mtic2).dot(UBOt))
                                                .transpose()
                                                .dot(q_vec);
                                        KOKKOS_VEC3 dV2 = _NABC.dot(DeltaH_second_deriv);
                                        Idbl_prime =
                                            -two_C * (dV.dot(dV) + V.dot(dV2)) * Iincrement;
                                        Idbl_prime += -two_C * (V_dot_dV)*Iprime;
                                    }
                                    dI2.eta[i_eta] += Idbl_prime;
                                }
                            }  // end of eta man deriv

                            // checkpoint for lambda manager
                            for (int i_lam = 0; i_lam < 2; i_lam++) {
                                if (refine_flag & (REFINE_LAMBDA << i_lam)) {
                                    CUDAREAL NH_dot_V = (_NABC.dot(H_vec)).dot(V);
                                    CUDAREAL dg_dlambda;
                                    if (i_lam == 0)
                                        dg_dlambda = 1;
                                    else  // i_lam==1
                                        dg_dlambda = lambda_ang;
                                    CUDAREAL coef =
                                        NH_dot_V * two_C * (dg_dlambda) / lambda_ang;
                                    CUDAREAL value = coef * Iincrement;
                                    CUDAREAL value2 = 0;
                                    dI.lambda[i_lam] += value;
                                    dI2.lambda[i_lam] += value2;
                                }
                            }
                            // end of lambda deriv
                            if (printout) {
                                if (_subS == 0 && _subF == 0 && _thick_tic == 0 &&
                                    _source == 0 && _mos_tic == 0) {
                                    if ((_fpixel == printout_fpixel &&
                                            _spixel == printout_spixel) ||
                                        printout_fpixel < 0) {
                                        printf("%4d %4d :  lambda = %g\n", _fpixel, _spixel, _lambda);
                                        printf(
                                            "at %g %g %g\n", _pixel_pos[0], _pixel_pos[1],
                                            _pixel_pos[2]);
                                        printf("Fdet= %g; Sdet= %g ; Odet= %g\n", _Fdet, _Sdet, _Odet);
                                        printf(
                                            "PIX0: %f %f %f\n", pix0_vectors(pid_x),
                                            pix0_vectors(pid_y), pix0_vectors(pid_z));
                                        printf(
                                            "F: %f %f %f\n", fdet_vectors(pid_x),
                                            fdet_vectors(pid_y), fdet_vectors(pid_z));
                                        printf(
                                            "S: %f %f %f\n", sdet_vectors(pid_x),
                                            sdet_vectors(pid_y), sdet_vectors(pid_z));
                                        printf(
                                            "O: %f %f %f\n", odet_vectors(pid_x),
                                            odet_vectors(pid_y), odet_vectors(pid_z));
                                        printf("pid_x=%d, pid_y=%d; pid_z=%d\n", pid_x, pid_y, pid_z);
                                        printf(
                                            "QVECTOR: %f %f %f\n", q_vec[0], q_vec[1], q_vec[2]);
                                        printf("omega   %15.10g\n", _omega_pixel);
                                        printf(
                                            "Incident: %g %g %g\n",
                                            _incident[0], _incident[1], _incident[2]);

                                        KOKKOS_MAT3 UU = UMATS_RXYZ(_mos_tic);
                                        printf(
                                            "UMAT_RXYZ :\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n",
                                            UU(0, 0), UU(0, 1), UU(0, 2), UU(1, 0), UU(1, 1),
                                            UU(1, 2), UU(2, 0), UU(2, 1), UU(2, 2));
                                        UU = Bmat_realspace;
                                        printf(
                                            "Bmat_realspace :\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n",
                                            UU(0, 0), UU(0, 1), UU(0, 2), UU(1, 0), UU(1, 1),
                                            UU(1, 2), UU(2, 0), UU(2, 1), UU(2, 2));
                                        UU = UBO;
                                        printf(
                                            "UBO :\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n",
                                            UU(0, 0), UU(0, 1), UU(0, 2), UU(1, 0), UU(1, 1),
                                            UU(1, 2), UU(2, 0), UU(2, 1), UU(2, 2));

                                        UU = UBOt;
                                        printf(
                                            "UBOt :\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n",
                                            UU(0, 0), UU(0, 1), UU(0, 2), UU(1, 0), UU(1, 1),
                                            UU(1, 2), UU(2, 0), UU(2, 1), UU(2, 2));

                                        // UU = UmosRxRyRzU;
                                        // printf(
                                        //     "UmosRxRyRzU :\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n",
                                        //     UU(0, 0), UU(0, 1), UU(0, 2), UU(1, 0), UU(1, 1),
                                        //     UU(1, 2), UU(2, 0), UU(2, 1), UU(2, 2));
                                        // KOKKOS_VEC3 AA = delta_H_prime;
                                        // printf(
                                        //     "delta_H_prime :\n%f  %f  %f\n", AA[0], AA[1],
                                        //     AA[2]);
                                        printf("Iincrement: %f\n", Iincrement);
                                        printf(
                                            "hkl= %f %f %f  hkl0= %d %d %d\n", _h, _k, _l, _h0,
                                            _k0, _l0);
                                        printf(
                                            " F_cell=%g  F_cell2=%g I_latt=%g   I = %g\n",
                                            _F_cell, _F_cell2, I0, _I);
                                        printf("I/steps %15.10g\n", _I / Nsteps);
                                        // printf("Ilatt diffuse %15.10g\n", I_latt_diffuse);
                                        printf("default_F= %f\n", default_F);
                                        if (complex_miller)
                                            printf("COMPLEX MILLER!\n");
                                        if (no_Nabc_scale)
                                            printf("No Nabc scale!\n");
                                    }
                                }
                            } // end of printout if

                        } // end of mos_tic loop
                      } // end of phi_tic loop
                    } // end of source loop
                } // end of thick step loop
            } // end of fpos loop
        } // end of spos loop
        floatimage(pixIdx) = _I;
        if (save_wavelenimage)
            wavelenimage(pixIdx) = Ilambda / _I;

        if (refine_flag) {
            manager_dI(pixIdx) = dI;
            manager_dI2(pixIdx) = dI2;
        }
    }); // end pixIdx loop

    if (Fhkl_gradient_mode)
        return;

    Kokkos::parallel_for(
        "deriv_image_increment", Npix_to_model, KOKKOS_LAMBDA(const int& pixIdx) {

        int _pid = panels_fasts_slows(pixIdx * 3);
        int _fpixel = panels_fasts_slows(pixIdx * 3 + 1);
        int _spixel = panels_fasts_slows(pixIdx * 3 + 2);

        CUDAREAL _Fdet_ave = pixel_size * _fpixel + pixel_size / 2.0;
        CUDAREAL _Sdet_ave = pixel_size * _spixel + pixel_size / 2.0;
        CUDAREAL _Odet_ave = 0;  // Odet;
        // TODO maybe make this more general for thick detectors?

        KOKKOS_VEC3 _pixel_pos_ave(0, 0, 0);
        int pid_x = _pid * 3;
        int pid_y = _pid * 3 + 1;
        int pid_z = _pid * 3 + 2;

        CUDAREAL fx = fdet_vectors(pid_x);
        CUDAREAL fy = fdet_vectors(pid_y);
        CUDAREAL fz = fdet_vectors(pid_z);

        CUDAREAL sx = sdet_vectors(pid_x);
        CUDAREAL sy = sdet_vectors(pid_y);
        CUDAREAL sz = sdet_vectors(pid_z);

        CUDAREAL ox = odet_vectors(pid_x);
        CUDAREAL oy = odet_vectors(pid_y);
        CUDAREAL oz = odet_vectors(pid_z);

        CUDAREAL px = pix0_vectors(pid_x);
        CUDAREAL py = pix0_vectors(pid_y);
        CUDAREAL pz = pix0_vectors(pid_z);

        _pixel_pos_ave[0] = _Fdet_ave * fx + _Sdet_ave * sx + _Odet_ave * ox + px;
        _pixel_pos_ave[1] = _Fdet_ave * fy + _Sdet_ave * sy + _Odet_ave * oy + py;
        _pixel_pos_ave[2] = _Fdet_ave * fz + _Sdet_ave * sz + _Odet_ave * oz + pz;

        CUDAREAL close_distance = close_distances(_pid);

        CUDAREAL _airpath_ave_r = 1 / _pixel_pos_ave.length();
        KOKKOS_VEC3 _diffracted_ave = _pixel_pos_ave.get_unit_vector();
        CUDAREAL _omega_pixel_ave = pixel_size * pixel_size * _airpath_ave_r * _airpath_ave_r *
                                    close_distance * _airpath_ave_r;

        CUDAREAL _polar = 1;
        if (!nopolar) {
            KOKKOS_VEC3 _incident(-source_X(0), -source_Y(0), -source_Z(0));
            _incident.normalize();
            // component of diffracted unit vector along _incident beam unit vector
            CUDAREAL cos2theta = _incident.dot(_diffracted_ave);
            CUDAREAL cos2theta_sqr = cos2theta * cos2theta;
            CUDAREAL sin2theta_sqr = 1 - cos2theta_sqr;

            CUDAREAL cos2psi = 0;
            if (kahn_factor != 0.0) {
                // cross product to get "vertical" axis that is orthogonal to the cannonical
                // "polarization"
                KOKKOS_VEC3 B_in = polarization_axis.cross(_incident);
                // cross product with _incident beam to get E-vector direction
                KOKKOS_VEC3 E_in = _incident.cross(B_in);
                // get components of diffracted ray projected onto the E-B plane
                CUDAREAL _kEi = _diffracted_ave.dot(E_in);
                CUDAREAL _kBi = _diffracted_ave.dot(B_in);
                // compute the angle of the diffracted ray projected onto the incident E-B plane
                // calculate cos(2 * atan2(_kBi, _kEi))
                if (_kEi!=0) {
                    CUDAREAL ratio = _kBi / _kEi;
                    cos2psi = (1 - ratio*ratio) / (1 + ratio*ratio);
                } else {
                    cos2psi = -1;
                }
            }
            // correction for polarized _incident beam
            _polar = 0.5 * (1.0 + cos2theta_sqr - kahn_factor * cos2psi * sin2theta_sqr);
        }

        CUDAREAL _om = 1;
        if (!oversample_omega)
            _om = _omega_pixel_ave;
        // final scale term to being everything to photon number units
        CUDAREAL _scale_term = _polar * _om * overall_scale;
        floatimage(pixIdx) *= _scale_term;

        auto& dI = manager_dI(pixIdx);
        auto& dI2 = manager_dI2(pixIdx);

        // udpate the rotation derivative images*
        for (int i_rot = 0; i_rot < 3; i_rot++) {
            if (refine_flag & (REFINE_UMAT1 << i_rot)) {
                CUDAREAL value = _scale_term * dI.rot[i_rot];
                CUDAREAL value2 = _scale_term * dI2.rot[i_rot];
                int idx = i_rot * Npix_to_model + pixIdx;
                d_Umat_images(idx) = value;
                d2_Umat_images(idx) = value2;
            }
        }  // end rot deriv image increment

        // update the ucell derivative images
        for (int i_uc = 0; i_uc < 6; i_uc++) {
            if (refine_flag & (REFINE_BMAT1 << i_uc)) {
                CUDAREAL value = _scale_term * dI.ucell[i_uc];
                CUDAREAL value2 = _scale_term * dI2.ucell[i_uc];
                int idx = i_uc * Npix_to_model + pixIdx;
                d_Bmat_images(idx) = value;
                d2_Bmat_images(idx) = value2;
            }
        }  // end ucell deriv image increment

        // update the Ncells derivative image
        if (refine_flag & REFINE_NCELLS1) {
            CUDAREAL value = _scale_term * dI.Ncells[0];
            CUDAREAL value2 = _scale_term * dI2.Ncells[0];
            int idx = pixIdx;
            d_Ncells_images(idx) = value;
            d2_Ncells_images(idx) = value2;

            if (!isotropic_ncells) {
                value = _scale_term * dI.Ncells[1];
                value2 = _scale_term * dI2.Ncells[1];
                idx = Npix_to_model + pixIdx;
                d_Ncells_images(idx) = value;
                d2_Ncells_images(idx) = value2;

                value = _scale_term * dI.Ncells[2];
                value2 = _scale_term * dI2.Ncells[2];
                idx = Npix_to_model * 2 + pixIdx;
                d_Ncells_images(idx) = value;
                d2_Ncells_images(idx) = value2;
            }
        }  // end Ncells deriv image increment
        if (refine_flag & REFINE_NCELLS_DEF) {
            for (int i_nc = 3; i_nc < 6; i_nc++) {
                CUDAREAL value = _scale_term * dI.Ncells[i_nc];
                CUDAREAL value2 = _scale_term * dI2.Ncells[i_nc];
                int idx = i_nc * Npix_to_model + pixIdx;
                d_Ncells_images(idx) = value;
                d2_Ncells_images(idx) = value2;
            }
        }

        // update Fcell derivative image
        if (refine_flag & REFINE_FCELL) {
            CUDAREAL value = _scale_term * dI.fcell;
            CUDAREAL value2 = _scale_term * dI2.fcell;
            d_fcell_images(pixIdx) = value;
            d2_fcell_images(pixIdx) = value2;
        }  // end Fcell deriv image increment

        if (refine_flag & REFINE_FP_FDP) {
            // c derivative
            CUDAREAL value = _scale_term * dI.fp_fdp[0];
            d_fp_fdp_images(pixIdx) = value;
            // d derivative
            value = _scale_term * dI.fp_fdp[1];
            d_fp_fdp_images(Npix_to_model + pixIdx) = value;
        }
        if (refine_flag & REFINE_DIFFUSE) {
            for (int i_gam = 0; i_gam < 3; i_gam++) {
                CUDAREAL val = dI.diffuse[i_gam] * _scale_term;
                int img_idx = Npix_to_model * i_gam + pixIdx;
                d_diffuse_gamma_images(img_idx) = val;
            }
            for (int i_sig = 0; i_sig < 3; i_sig++) {
                CUDAREAL val = dI.diffuse[i_sig + 3] * _scale_term;
                int img_idx = Npix_to_model * i_sig + pixIdx;
                d_diffuse_sigma_images(img_idx) = val;
            }
        }

        // update eta derivative image
        if (refine_flag & REFINE_ETA) {
            for (int i_eta = 0; i_eta < 3; i_eta++) {
                if (i_eta > 0 && !aniso_eta)
                    continue;
                int idx = pixIdx + Npix_to_model * i_eta;
                CUDAREAL value = _scale_term * dI.eta[i_eta];
                CUDAREAL value2 = _scale_term * dI2.eta[i_eta];
                d_eta_images(idx) = value;
                d2_eta_images(idx) = value2;
            }
        }  // end eta deriv image increment

        // update the lambda derivative images
        for (int i_lam = 0; i_lam < 2; i_lam++) {
            if (refine_flag & (REFINE_LAMBDA1 << i_lam)) {
                CUDAREAL value = _scale_term * dI.lambda[i_lam];
                CUDAREAL value2 = _scale_term * dI2.lambda[i_lam];
                int idx = i_lam * Npix_to_model + pixIdx;
                d_lambda_images(idx) = value;
                // d2_lambda_images(idx) = value2;
            }
        }  // end lambda deriv image increment

        for (int i_pan_rot = 0; i_pan_rot < 3; i_pan_rot++) {
            if (refine_flag & (REFINE_PANEL_ROT1 << i_pan_rot)) {
                CUDAREAL value = _scale_term * dI.pan_rot[i_pan_rot];
                CUDAREAL value2 = _scale_term * dI2.pan_rot[i_pan_rot];
                int idx = i_pan_rot * Npix_to_model + pixIdx;
                d_panel_rot_images(idx) = value;
                // d2_panel_rot_images(idx) = value2;
            }
        }  // end panel rot deriv image increment

        for (int i_pan_orig = 0; i_pan_orig < 3; i_pan_orig++) {
            if (refine_flag & (REFINE_PANEL_ORIGIN1 << i_pan_orig)) {
                CUDAREAL value = _scale_term * dI.pan_orig[i_pan_orig];
                CUDAREAL value2 = _scale_term * dI2.pan_orig[i_pan_orig];
                int idx = i_pan_orig * Npix_to_model + pixIdx;
                d_panel_orig_images(idx) = value;
                // d2_panel_orig_images(idx) = value2;
            }
        }  // end panel orig deriv image increment
    });    // end pixIdx loop

}  // END of GPU kernel

template void
kokkos_sum_over_steps<
    false, // printout,
    false, // complex_miller,
    false, // compute_curvatures,
    REFINE_FCELL,  // refine_flag,
    false, // use_diffuse,
    false, // save_wavelenimage
    false, // Fhkl_gradient_mode,
    false, // Fhkl_errors_mode,
    false, // using_trusted_mask,
    true, // Fhkl_channels_empty,
    false> // Fhkl_have_scale_factors
    (
    int Npix_to_model,
    vector_uint_t panels_fasts_slows,
    vector_cudareal_t floatimage,
    vector_cudareal_t wavelenimage,
    vector_cudareal_t d_Umat_images,
    vector_cudareal_t d2_Umat_images,
    vector_cudareal_t d_Bmat_images,
    vector_cudareal_t d2_Bmat_images,
    vector_cudareal_t d_Ncells_images,
    vector_cudareal_t d2_Ncells_images,
    vector_cudareal_t d_fcell_images,
    vector_cudareal_t d2_fcell_images,
    vector_cudareal_t d_eta_images,
    vector_cudareal_t d2_eta_images,
    vector_cudareal_t d_lambda_images,
    vector_cudareal_t d2_lambda_images,
    vector_cudareal_t d_panel_rot_images,
    vector_cudareal_t d2_panel_rot_images,
    vector_cudareal_t d_panel_orig_images,
    vector_cudareal_t d2_panel_orig_images,
    vector_cudareal_t d_fp_fdp_images,
    vector_manager_t manager_dI,
    vector_manager_t manager_dI2,
    const int Nsteps,
    int printout_fpixel,
    int printout_spixel,
    /*bool printout,*/
    CUDAREAL default_F,
    int oversample,
    bool oversample_omega,
    CUDAREAL subpixel_size,
    CUDAREAL pixel_size,
    CUDAREAL detector_thickstep,
    CUDAREAL detector_thick,
    const vector_cudareal_t close_distances,
    CUDAREAL detector_attnlen,
    int detector_thicksteps,
    int sources,
    int phisteps,
    int mosaic_domains,
    bool use_lambda_coefficients,
    CUDAREAL lambda0,
    CUDAREAL lambda1,
    KOKKOS_MAT3 eig_U,
    KOKKOS_MAT3 eig_O,
    KOKKOS_MAT3 eig_B,
    KOKKOS_MAT3 RXYZ,
    vector_vec3_t dF_vecs,
    vector_vec3_t dS_vecs,
    const vector_mat3_t UMATS_RXYZ,
    vector_mat3_t UMATS_RXYZ_prime,
    vector_mat3_t UMATS_RXYZ_dbl_prime,
    vector_mat3_t RotMats,
    vector_mat3_t dRotMats,
    vector_mat3_t d2RotMats,
    vector_mat3_t UMATS,
    vector_mat3_t dB_mats,
    vector_mat3_t dB2_mats,
    vector_mat3_t Amatrices,
    const vector_cudareal_t source_X,
    const vector_cudareal_t source_Y,
    const vector_cudareal_t source_Z,
    const vector_cudareal_t source_lambda,
    const vector_cudareal_t source_I,
    CUDAREAL kahn_factor,
    CUDAREAL Na,
    CUDAREAL Nb,
    CUDAREAL Nc,
    CUDAREAL Nd,
    CUDAREAL Ne,
    CUDAREAL Nf,
    CUDAREAL phi0,
    CUDAREAL phistep,
    KOKKOS_VEC3 spindle_vec,
    KOKKOS_VEC3 polarization_axis,
    int h_range,
    int k_range,
    int l_range,
    int h_max,
    int h_min,
    int k_max,
    int k_min,
    int l_max,
    int l_min,
    CUDAREAL dmin,
    CUDAREAL fudge,
    /*bool complex_miller,*/
    int verbose,
    bool only_save_omega_kahn,
    bool isotropic_ncells,
    /*bool compute_curvatures,*/
    const vector_cudareal_t FhklLinear,
    const vector_cudareal_t Fhkl2Linear,
    /*const uint32_t refine_flag,*/
    // vector_bool_t refine_Bmat,
    // vector_bool_t refine_Ncells,
    // bool refine_Ncells_def,
    // vector_bool_t refine_panel_origin,
    // vector_bool_t refine_panel_rot,
    // bool refine_fcell,
    // vector_bool_t refine_lambda,
    // bool refine_eta,
    // vector_bool_t refine_Umat,
    const vector_cudareal_t fdet_vectors,
    const vector_cudareal_t sdet_vectors,
    const vector_cudareal_t odet_vectors,
    const vector_cudareal_t pix0_vectors,
    bool nopolar,
    bool point_pixel,
    CUDAREAL fluence,
    CUDAREAL r_e_sqr,
    CUDAREAL spot_scale,
    int Npanels,
    bool aniso_eta,
    bool no_Nabc_scale,
    const vector_cudareal_t fpfdp,
    const vector_cudareal_t fpfdp_derivs,
    const vector_cudareal_t atom_data,
    int num_atoms,
    // bool refine_fp_fdp,
    const vector_int_t nominal_hkl,
    bool use_nominal_hkl,
    KOKKOS_MAT3 anisoU,
    KOKKOS_MAT3 anisoG,
    KOKKOS_MAT3 rotate_principal_axes,
    /*bool use_diffuse,*/
    vector_cudareal_t d_diffuse_gamma_images,
    vector_cudareal_t d_diffuse_sigma_images,
    // bool refine_diffuse,
    bool gamma_miller_units,
    // bool refine_Icell,
    /*bool save_wavelenimage,*/
    int laue_group_num,
    int stencil_size,
    /*bool Fhkl_gradient_mode,*/
    /*bool Fhkl_errors_mode,*/
    /*bool using_trusted_mask,*/
    /*bool Fhkl_channels_empty,*/
    /*bool Fhkl_have_scale_factors,*/
    int Num_ASU,
    const vector_cudareal_t data_residual,
    const vector_cudareal_t data_variance,
    const vector_int_t data_freq,
    const vector_bool_t data_trusted,
    const vector_int_t FhklLinear_ASUid,
    const vector_int_t Fhkl_channels,
    const vector_cudareal_t Fhkl_scale,
    vector_cudareal_t Fhkl_scale_deriv,
    bool gaussian_star_shape, bool square_shape);
