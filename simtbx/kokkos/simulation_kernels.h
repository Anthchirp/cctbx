#ifndef SIMTBX_KOKKOS_SIMULATION_KERNELS_H
#define SIMTBX_KOKKOS_SIMULATION_KERNELS_H
#include <kokkostbx/kokkos_types.h>
#include <kokkostbx/kokkos_vector3.h>
#include <kokkostbx/kokkos_matrix3.h>
using vec3 = kokkostbx::vector3<CUDAREAL>;
using mat3 = kokkostbx::matrix3<CUDAREAL>;

#include <simtbx/nanoBragg/nanotypes.h>
#include <simtbx/kokkos/kernel_math.h>
#include <simtbx/diffBragg/src/diffBraggKOKKOS.h>
#include <simtbx/diffBragg/src/diffuse_util_kokkos.h> // test diffuse halo in exascale API context.

using simtbx::nanoBragg::shapetype;
using simtbx::nanoBragg::hklParams;
using simtbx::nanoBragg::SQUARE;
using simtbx::nanoBragg::ROUND;
using simtbx::nanoBragg::GAUSS;
using simtbx::nanoBragg::GAUSS_ARGCHK;
using simtbx::nanoBragg::TOPHAT;

void calc_CrystalOrientations(CUDAREAL phi0,
                              CUDAREAL phistep,
                              int phisteps,
                              const vector_cudareal_t spindle_vector,
                              const vector_cudareal_t a0,
                              const vector_cudareal_t b0,
                              const vector_cudareal_t c0,
                              CUDAREAL mosaic_spread,
                              int mosaic_domains,
                              const vector_cudareal_t mosaic_umats,
                              crystal_orientation_t crystal_orientation) {

        Kokkos::parallel_for("calc_CrystalOrientation", phisteps, KOKKOS_LAMBDA(const int& phi_tic) {
                // sweep over phi angles
                CUDAREAL phi = phistep * phi_tic + phi0;

                vec3 spindle_vector_tmp {spindle_vector(1), spindle_vector(2), spindle_vector(3)};
                vec3 a0_tmp {a0(1), a0(2), a0(3)};
                vec3 b0_tmp {b0(1), b0(2), b0(3)};
                vec3 c0_tmp {c0(1), c0(2), c0(3)};

                // rotate about spindle if necessary
                vec3 ap = a0_tmp.rotate_around_axis(spindle_vector_tmp, phi);
                vec3 bp = b0_tmp.rotate_around_axis(spindle_vector_tmp, phi);
                vec3 cp = c0_tmp.rotate_around_axis(spindle_vector_tmp, phi);

                // enumerate mosaic domains
                for (int mos_tic = 0; mos_tic < mosaic_domains; ++mos_tic) {
                        // apply mosaic rotation after phi rotation
                        vec3 a, b, c;

                        if (mosaic_spread > 0.0) {
                                mat3 umat;
                                for (int i=0; i<9; ++i) {
                                        umat[i] = mosaic_umats(mos_tic * 9 + i);
                                }
                                a = umat.dot(ap);
                                b = umat.dot(bp);
                                c = umat.dot(cp);
                        } else {
                                a = ap;
                                b = bp;
                                c = cp;
                        }
                        crystal_orientation(phi_tic, mos_tic, 0) = a;
                        crystal_orientation(phi_tic, mos_tic, 1) = b;
                        crystal_orientation(phi_tic, mos_tic, 2) = c;
                }
        });
}


void kokkosSpotsKernel(int spixels, int fpixels, int roi_xmin, int roi_xmax,
    int roi_ymin, int roi_ymax, int oversample, int point_pixel,
    CUDAREAL pixel_size, CUDAREAL subpixel_size, int steps, CUDAREAL detector_thickstep,
    int detector_thicksteps, CUDAREAL detector_thick, CUDAREAL detector_mu,
    const view_1d_t<vec3> sdet_vector, const view_1d_t<vec3> fdet_vector,
    const view_1d_t<vec3> odet_vector, const view_1d_t<vec3> pix0_vector,
    int curved_detector, CUDAREAL distance, CUDAREAL close_distance,
    const vector_cudareal_t beam_vector,
    CUDAREAL dmin, int phisteps, int sources,
    const vector_cudareal_t source_X, const vector_cudareal_t source_Y,
    const vector_cudareal_t source_Z,
    const vector_cudareal_t source_I, const vector_cudareal_t source_lambda,
    shapetype xtal_shape,
    int mosaic_domains, crystal_orientation_t crystal_orientation,
    CUDAREAL Na, CUDAREAL Nb, CUDAREAL Nc, CUDAREAL V_cell,
    CUDAREAL water_size, CUDAREAL water_F, CUDAREAL water_MW, CUDAREAL r_e_sqr,
    CUDAREAL fluence, CUDAREAL Avogadro, CUDAREAL spot_scale, int integral_form,
    CUDAREAL default_F,
    const vector_cudareal_t Fhkl,
    const hklParams FhklParams,
    int nopolar,
    const vector_cudareal_t polar_vector, CUDAREAL polarization, CUDAREAL fudge,
    const vector_ushort_t * maskimage,
    vector_float_t floatimage /*out*/,
    vector_float_t omega_reduction /*out*/, vector_float_t max_I_x_reduction /*out*/,
    vector_float_t max_I_y_reduction /*out*/, vector_bool_t rangemap) {

        const int s_h_min = FhklParams.h_min;
        const int s_k_min = FhklParams.k_min;
        const int s_l_min = FhklParams.l_min;
        const int s_h_range = FhklParams.h_range;
        const int s_k_range = FhklParams.k_range;
        const int s_l_range = FhklParams.l_range;
        const int s_h_max = s_h_min + s_h_range - 1;
        const int s_k_max = s_k_min + s_k_range - 1;
        const int s_l_max = s_l_min + s_l_range - 1;

        const int total_pixels = spixels * fpixels;

        const CUDAREAL distance_r = 1 / distance;
        const CUDAREAL dmin_r = (dmin > 0.0) ? 1/dmin : 0.0;

        // add background from something amorphous, precalculate scaling
        const CUDAREAL F_bg = water_F;
        const CUDAREAL I_bg = F_bg * F_bg * r_e_sqr * fluence * water_size * water_size * water_size * 1e6 * Avogadro / water_MW;
        const CUDAREAL I_factor = r_e_sqr * spot_scale * fluence / steps;

       Kokkos::parallel_for("kokkosSpotsKernel", total_pixels, KOKKOS_LAMBDA(const int& pixIdx) {

                vec3 polar_vector_tmp {polar_vector(1), polar_vector(2), polar_vector(3)};

                const int fpixel = pixIdx % fpixels;
                const int spixel = pixIdx / fpixels;
                // allow for just a region of interest (roi) on detector to be rendered
                if (fpixel < roi_xmin || fpixel > roi_xmax || spixel < roi_ymin || spixel > roi_ymax) {
                        return;
                }

                // allow for the use of a mask
                if (maskimage != NULL) {
                        // skip any flagged pixels in the mask
                        if ((*maskimage)(pixIdx) == 0) {
                                return;
                        }
                }

                // reset photon count for this pixel
                CUDAREAL I = I_bg;
                CUDAREAL omega_sub_reduction = 0.0;
                CUDAREAL max_I_x_sub_reduction = 0.0;
                CUDAREAL max_I_y_sub_reduction = 0.0;
                CUDAREAL polar = 0.0;
                if (nopolar) {
                        polar = 1.0;
                }

                // add this now to avoid problems with skipping later
                // move this to the bottom to avoid accessing global device memory. floatimage[j] = I_bg;
                // loop over sub-pixels
                int subS, subF;
                for (subS = 0; subS < oversample; ++subS) { // Y voxel
                        for (subF = 0; subF < oversample; ++subF) { // X voxel
                                // absolute mm position on detector (relative to its origin)
                                CUDAREAL Fdet = subpixel_size * (fpixel * oversample + subF) + subpixel_size / 2.0; // X voxel
                                CUDAREAL Sdet = subpixel_size * (spixel * oversample + subS) + subpixel_size / 2.0; // Y voxel
                                // Fdet = pixel_size*fpixel;
                                // Sdet = pixel_size*spixel;

                                max_I_x_sub_reduction = Fdet;
                                max_I_y_sub_reduction = Sdet;

                                int thick_tic;
                                for (thick_tic = 0; thick_tic < detector_thicksteps; ++thick_tic) {
                                        // assume "distance" is to the front of the detector sensor layer
                                        CUDAREAL Odet = thick_tic * detector_thickstep; // Z Orthagonal voxel.

                                        // construct detector subpixel position in 3D space
                                        //                      pixel_X = distance;
                                        //                      pixel_Y = Sdet-Ybeam;
                                        //                      pixel_Z = Fdet-Xbeam;
                                        vec3 pixel_pos;
                                        pixel_pos += Fdet * fdet_vector(0);
                                        pixel_pos += Sdet * sdet_vector(0);
                                        pixel_pos += Odet * odet_vector(0);
                                        pixel_pos += pix0_vector(0);

                                        if (curved_detector) {
                                                // construct detector pixel that is always "distance" from the sample
                                                vec3 dbvector = distance * vec3{beam_vector(1), beam_vector(2), beam_vector(3)};
                                                // treat detector pixel coordinates as radians
                                                vec3 newvector = dbvector.rotate_around_axis(sdet_vector(0), pixel_pos.y_val() * distance_r );
                                                pixel_pos = newvector.rotate_around_axis(fdet_vector(0), pixel_pos.z_val() * distance_r );
                                        }

                                        // construct the diffracted-beam unit vector to this sub-pixel
                                        CUDAREAL airpath_r = 1 / pixel_pos.length();
                                        vec3 diffracted = pixel_pos.get_unit_vector();

                                        // solid angle subtended by a pixel: (pix/airpath)^2*cos(2theta)
                                        CUDAREAL omega_pixel = pixel_size * pixel_size * airpath_r * airpath_r * close_distance * airpath_r;
                                        // option to turn off obliquity effect, inverse-square-law only
                                        if (point_pixel) {
                                                omega_pixel = airpath_r * airpath_r;
                                        }

                                        // now calculate detector thickness effects
                                        CUDAREAL capture_fraction = 1.0;
                                        if (detector_thick > 0.0 && detector_mu> 0.0) {
                                                // inverse of effective thickness increase
                                                CUDAREAL parallax = odet_vector(0).dot(diffracted);
                                                capture_fraction = exp(-thick_tic * detector_thickstep / detector_mu / parallax)
                                                                - exp(-(thick_tic + 1) * detector_thickstep / detector_mu / parallax);
                                        }

                                        // loop over sources now
                                        int source;
                                        for (source = 0; source < sources; ++source) {

                                                // retrieve stuff from cache
                                                vec3 incident = {-source_X(source), -source_Y(source), -source_Z(source)};
                                                CUDAREAL lambda = source_lambda(source);
                                                CUDAREAL source_fraction = source_I(source);

                                                // construct the incident beam unit vector while recovering source distance
                                                // TODO[Giles]: Optimization! We can unitize the source vectors before passing them in.
                                                incident.normalize();

                                                // construct the scattering vector for this pixel
                                                vec3 scattering = (diffracted - incident) / lambda;
                                                CUDAREAL stol = 0.5 * scattering.length();

                                                // rough cut to speed things up when we aren't using whole detector
                                                if (dmin > 0.0 && stol > 0.0) {
                                                        // use reciprocal of (dmin > 0.5 / stol)
                                                        if (dmin_r <= 2 * stol) {
                                                                continue;
                                                        }
                                                }

                                                // polarization factor
                                                if (!nopolar) {
                                                        // need to compute polarization factor
                                                        polar = polarization_factor(polarization, incident, diffracted, polar_vector_tmp);
                                                } else {
                                                        polar = 1.0;
                                                }

                                                // sweep over phi angles
                                                for (int phi_tic = 0; phi_tic < phisteps; ++phi_tic) {
                                                        // enumerate mosaic domains
                                                        for (int mos_tic = 0; mos_tic < mosaic_domains; ++mos_tic) {
                                                                // apply mosaic rotation after phi rotation
                                                                auto a = crystal_orientation(phi_tic, mos_tic, 0);
                                                                auto b = crystal_orientation(phi_tic, mos_tic, 1);
                                                                auto c = crystal_orientation(phi_tic, mos_tic, 2);

                                                                // construct fractional Miller indicies
                                                                CUDAREAL h = a.dot(scattering);
                                                                CUDAREAL k = b.dot(scattering);
                                                                CUDAREAL l = c.dot(scattering);

                                                                // round off to nearest whole index
                                                                int h0 = ceil(h - 0.5);
                                                                int k0 = ceil(k - 0.5);
                                                                int l0 = ceil(l - 0.5);

                                                                // structure factor of the lattice (paralelpiped crystal)
                                                                // F_latt = sin(M_PI*s_Na*h)*sin(M_PI*s_Nb*k)*sin(M_PI*s_Nc*l)/sin(M_PI*h)/sin(M_PI*k)/sin(M_PI*l);

                                                                CUDAREAL F_latt = 1.0; // Shape transform for the crystal.
                                                                CUDAREAL hrad_sqr = 0.0;

                                                                if (xtal_shape == SQUARE) {
                                                                        // xtal is a paralelpiped
                                                                        if (Na > 1) {
                                                                                // F_latt *= sincgrad(h, s_Na);
                                                                                F_latt *= sincg(M_PI * h, Na);
                                                                        }
                                                                        if (Nb > 1) {
                                                                                // F_latt *= sincgrad(k, s_Nb);
                                                                                F_latt *= sincg(M_PI * k, Nb);
                                                                        }
                                                                        if (Nc > 1) {
                                                                                // F_latt *= sincgrad(l, s_Nc);
                                                                                F_latt *= sincg(M_PI * l, Nc);
                                                                        }
                                                                } else {
                                                                        // handy radius in reciprocal space, squared
                                                                        const CUDAREAL hrad = (h - h0) * Na;
                                                                        const CUDAREAL krad = (k - k0) * Nb;
                                                                        const CUDAREAL lrad = (l - l0) * Nc;
                                                                        hrad_sqr = hrad * hrad + krad * krad + lrad * lrad;
                                                                }
                                                                if (xtal_shape == ROUND) {
                                                                        // use sinc3 for elliptical xtal shape,
                                                                        // correcting for sqrt of volume ratio between cube and sphere
                                                                        F_latt = Na * Nb * Nc * 0.723601254558268 * sinc3(M_PI * sqrt(hrad_sqr * fudge));
                                                                }
                                                                if (xtal_shape == GAUSS) {
                                                                        // fudge the radius so that volume and FWHM are similar to square_xtal spots
                                                                        F_latt = Na * Nb * Nc * exp(-(hrad_sqr / 0.63 * fudge));
                                                                }
                                                                if (xtal_shape == GAUSS_ARGCHK) {
                                                                        // fudge the radius so that volume and FWHM are similar to square_xtal spots
                                                                        double my_arg = hrad_sqr / 0.63 * fudge;
                                                                        if (my_arg<35.){ F_latt = Na * Nb * Nc * exp(-(my_arg));
                                                                        } else { F_latt = 0.; } // warps coalesce when blocks of 32 pixels have no Bragg signal
                                                                }
                                                                if (xtal_shape == TOPHAT) {
                                                                        // make a flat-top spot of same height and volume as square_xtal spots
                                                                        F_latt = Na * Nb * Nc * (hrad_sqr * fudge < 0.3969);
                                                                }
                                                                // no need to go further if result will be zero?
                                                                if (F_latt == 0.0 && water_size == 0.0)
                                                                        continue;

                                                                // structure factor of the unit cell
                                                                CUDAREAL F_cell = default_F;
                                                                //F_cell = quickFcell_ldg(s_hkls, s_h_max, s_h_min, s_k_max, s_k_min, s_l_max, s_l_min, h0, k0, l0, s_h_range, s_k_range, s_l_range, default_F, Fhkl);
                                                                if (
                                                                        h0 < s_h_min ||
                                                                        k0 < s_k_min ||
                                                                        l0 < s_l_min ||
                                                                        h0 > s_h_max ||
                                                                        k0 > s_k_max ||
                                                                        l0 > s_l_max
                                                                        ) {
                                                                        F_cell = 0.;
                                                                } else {
                                                                        const int hkl_index = (h0-s_h_min)*s_k_range*s_l_range + (k0-s_k_min)*s_l_range + (l0-s_l_min);
                                                                        F_cell = Fhkl[hkl_index];
                                                                }

                                                                // now we have the structure factor for this pixel

                                                                // convert amplitudes into intensity (photons per steradian)
                                                                I += F_cell * F_cell * F_latt * F_latt * source_fraction * capture_fraction * omega_pixel;
                                                                omega_sub_reduction += omega_pixel;
                                                        } // end of mosaic loop
                                                } // end of phi loop
                                        } // end of source loop
                                } // end of detector thickness loop
                        } // end of sub-pixel y loop
                } // end of sub-pixel x loop
                const double photons = I_bg + I_factor * polar * I;
                floatimage( pixIdx ) = photons;
                omega_reduction( pixIdx ) = omega_sub_reduction; // shared contention
                max_I_x_reduction( pixIdx ) = max_I_x_sub_reduction;
                max_I_y_reduction( pixIdx ) = max_I_y_sub_reduction;
                rangemap( pixIdx ) = true;
        });
    }

void debranch_maskall_Kernel(int npanels, int spixels, int fpixels, int total_pixels,
    int oversample, int point_pixel,
    CUDAREAL pixel_size, CUDAREAL subpixel_size, int steps,
    CUDAREAL detector_thickstep, int detector_thicksteps, CUDAREAL detector_thick, CUDAREAL detector_mu,
    const view_1d_t<vec3> sdet_vector, const view_1d_t<vec3> fdet_vector,
    const view_1d_t<vec3> odet_vector, const view_1d_t<vec3> pix0_vector,
    const vector_cudareal_t close_distance,
    const vector_cudareal_t beam_vector,
    CUDAREAL dmin, int phisteps, int sources,
    const vector_cudareal_t source_X, const vector_cudareal_t source_Y,
    const vector_cudareal_t source_Z,
    const vector_cudareal_t source_I, const vector_cudareal_t source_lambda,
    int mosaic_domains, crystal_orientation_t crystal_orientation,
    CUDAREAL Na, CUDAREAL Nb, CUDAREAL Nc, CUDAREAL V_cell,
    CUDAREAL water_size, CUDAREAL water_F, CUDAREAL water_MW, CUDAREAL r_e_sqr,
    CUDAREAL fluence, CUDAREAL Avogadro, CUDAREAL spot_scale, int integral_form,
    CUDAREAL default_F,
    const vector_cudareal_t Fhkl,
    const hklParams FhklParams,
    int nopolar,
    const vector_cudareal_t polar_vector, CUDAREAL polarization, CUDAREAL fudge,
    const vector_size_t pixel_lookup,
    simtbx::Kokkos::diffuse_api diffuse,
    vector_float_t floatimage /*out*/,
    vector_float_t omega_reduction /*out*/, vector_float_t max_I_x_reduction /*out*/,
    vector_float_t max_I_y_reduction /*out*/, vector_bool_t rangemap) {


        const int s_h_min = FhklParams.h_min;
        const int s_k_min = FhklParams.k_min;
        const int s_l_min = FhklParams.l_min;
        const int s_h_range = FhklParams.h_range;
        const int s_k_range = FhklParams.k_range;
        const int s_l_range = FhklParams.l_range;
        const int s_h_max = s_h_min + s_h_range - 1;
        const int s_k_max = s_k_min + s_k_range - 1;
        const int s_l_max = s_l_min + s_l_range - 1;

        // set up diffuse scattering if needed
        vector_mat3_t laue_mats = vector_mat3_t("laue_mats",24);
        vector_cudareal_t dG_trace = vector_cudareal_t("dG_trace",3);
        vector_vec3_t dG_dgam = vector_vec3_t("dG_dgam",3);
        int num_laue_mats = 0;
        int dhh = 0, dkk = 0, dll = 0;
        KOKKOS_MAT3 rotate_principal_axes = diffuse.rotate_principal_axes; // (1,0,0,0,1,0,0,0,1);
        int laue_group_num = diffuse.laue_group_num;
        int stencil_size = diffuse.stencil_size;
        KOKKOS_MAT3 anisoG = diffuse.anisoG; // (300.,0,0,0,100.,0,0,0,300.);
        KOKKOS_MAT3 anisoU = diffuse.anisoU; // (0.48,0,0,0,0.16,0,0,0,0.16);
        KOKKOS_MAT3 Bmat_realspace(1.,0,0,0,1.,0,0,0,1.); // Placeholder
        KOKKOS_MAT3 anisoG_local;
        CUDAREAL anisoG_determ = 0;
        KOKKOS_MAT3 anisoU_local;
        bool use_diffuse=diffuse.enable;
        // ***NEEDS UPDATE: use legacy API for passing diffuse scale as KOKKOS_MAT3
        vector_mat3_t diffuse_scale_mat3 = vector_mat3_t("diffuse_scale_mat3",1);

        if (use_diffuse){
          anisoG_local = anisoG;
          anisoU_local = anisoU;

          if (laue_group_num < 1 || laue_group_num >14 ){
            throw std::string("Laue group number not in range 1-14");
          }

          Kokkos::parallel_reduce("prepare diffuse mats", 1, KOKKOS_LAMBDA (const int& i, int& num_laue_mats_temp){
            num_laue_mats_temp = gen_laue_mats(laue_group_num, laue_mats, rotate_principal_axes);
            // KOKKOS_MAT3 rotate_principal_axes;
            // rotate_principal_axes << 0.70710678, -0.70710678, 0., 0.70710678, 0.70710678, 0., 0., 0., 1.;

            KOKKOS_MAT3 Amatrix(diffuse.a0[0],diffuse.a0[1],diffuse.a0[2],diffuse.b0[0],diffuse.b0[1],diffuse.b0[2],
                                diffuse.c0[0],diffuse.c0[1],diffuse.c0[2]);
            KOKKOS_MAT3 Ainv = Amatrix.inverse()*1.e-10;
            CUDAREAL reciprocal_space_volume = 8*M_PI*M_PI*M_PI*Ainv.determinant();
            CUDAREAL _tmpfac = M_PI * 0.63 / fudge;
            CUDAREAL diffuse_scale = reciprocal_space_volume * sqrt(_tmpfac*_tmpfac*_tmpfac);

            // ***NEEDS UPDATE: use legacy API for passing diffuse scale as KOKKOS_MAT3
            diffuse_scale_mat3(0)(0,0) = diffuse_scale;

            for ( int iL = 0; iL < num_laue_mats_temp; iL++ ){
              laue_mats(iL) = Ainv * laue_mats(iL);
            }
            const KOKKOS_MAT3 Ginv = anisoG_local.inverse();
            const KOKKOS_MAT3 dG = Bmat_realspace * Ginv;
            for (int i_gam=0; i_gam<3; i_gam++){
              dG_dgam(i_gam)[i_gam] = 1;
              KOKKOS_MAT3 temp_dgam;
              temp_dgam(i_gam, 0) = dG_dgam(i_gam)[0];
              temp_dgam(i_gam, 1) = dG_dgam(i_gam)[1];
              temp_dgam(i_gam, 2) = dG_dgam(i_gam)[2];
              dG_trace(i_gam) = (Ginv*temp_dgam).trace();
            }
          }, num_laue_mats);
          anisoG_determ = anisoG_local.determinant();
          dhh = dkk = dll = stencil_size; // Limits of stencil for diffuse calc
        }
        KOKKOS_VEC3 dHH(dhh,dkk,dll);
        // end of diffuse setup

// Implementation notes.  This kernel is aggressively debranched, therefore the assumptions are:
// 1) mosaicity non-zero positive
// 2) xtal shape is "Gauss" i.e. 3D spheroid.
// 3) No bounds check for access to the structure factor array.
// 4) No check for Flatt=0.
        const CUDAREAL dmin_r = (dmin > 0.0) ? 1/dmin : 0.0;

        // add background from something amorphous, precalculate scaling
        const CUDAREAL F_bg = water_F;
        const CUDAREAL I_bg = F_bg * F_bg * r_e_sqr * fluence * water_size * water_size * water_size * 1e6 * Avogadro / water_MW;
        const CUDAREAL I_factor = r_e_sqr * spot_scale * fluence / steps;

        Kokkos::parallel_for("debranch_maskall", total_pixels, KOKKOS_LAMBDA(const int& pixIdx) {

                vec3 polar_vector_tmp {polar_vector(1), polar_vector(2), polar_vector(3)};

                // position in pixel array
                const int j = pixel_lookup(pixIdx);//pixIdx: index into pixel subset; j: index into the data.
                const int i_panel = j / (fpixels*spixels); // the panel number
                const int j_panel = j % (fpixels*spixels); // the pixel number within the panel
                const int fpixel = j_panel % fpixels;
                const int spixel = j_panel / fpixels;

                // reset photon count for this pixel
                CUDAREAL I = I_bg;
                CUDAREAL omega_sub_reduction = 0.0;
                CUDAREAL max_I_x_sub_reduction = 0.0;
                CUDAREAL max_I_y_sub_reduction = 0.0;
                CUDAREAL polar = 0.0;
                if (nopolar) {
                        polar = 1.0;
                }

                // add this now to avoid problems with skipping later
                // move this to the bottom to avoid accessing global device memory. floatimage[j] = I_bg;
                // loop over sub-pixels
                int subS, subF;
                for (subS = 0; subS < oversample; ++subS) { // Y voxel
                        for (subF = 0; subF < oversample; ++subF) { // X voxel
                                // absolute mm position on detector (relative to its origin)
                                CUDAREAL Fdet = subpixel_size * (fpixel * oversample + subF) + subpixel_size / 2.0; // X voxel
                                CUDAREAL Sdet = subpixel_size * (spixel * oversample + subS) + subpixel_size / 2.0; // Y voxel
                                // Fdet = pixel_size*fpixel;
                                // Sdet = pixel_size*spixel;

                                max_I_x_sub_reduction = Fdet;
                                max_I_y_sub_reduction = Sdet;

                                int thick_tic;
                                for (thick_tic = 0; thick_tic < detector_thicksteps; ++thick_tic) {
                                        // assume "distance" is to the front of the detector sensor layer
                                        CUDAREAL Odet = thick_tic * detector_thickstep; // Z Orthagonal voxel.

                                        // construct detector subpixel position in 3D space
                                        //                      pixel_X = distance;
                                        //                      pixel_Y = Sdet-Ybeam;
                                        //                      pixel_Z = Fdet-Xbeam;
                                        vec3 pixel_pos;
                                        pixel_pos += Fdet * fdet_vector(i_panel);
                                        pixel_pos += Sdet * sdet_vector(i_panel);
                                        pixel_pos += Odet * odet_vector(i_panel);
                                        pixel_pos += pix0_vector(i_panel);

                                        // construct the diffracted-beam unit vector to this sub-pixel
                                        CUDAREAL airpath_r = 1 / pixel_pos.length();
                                        vec3 diffracted = pixel_pos.get_unit_vector();

                                        // solid angle subtended by a pixel: (pix/airpath)^2*cos(2theta)
                                        CUDAREAL omega_pixel = pixel_size * pixel_size * airpath_r * airpath_r * close_distance(i_panel) * airpath_r;
                                        // option to turn off obliquity effect, inverse-square-law only
                                        if (point_pixel) {
                                                omega_pixel = airpath_r * airpath_r;
                                        }

                                        // now calculate detector thickness effects
                                        CUDAREAL capture_fraction = 1.0;
                                        if (detector_thick > 0.0 && detector_mu> 0.0) {
                                                // inverse of effective thickness increase
                                                CUDAREAL parallax = odet_vector(i_panel).dot(diffracted);
                                                capture_fraction = exp(-thick_tic * detector_thickstep / detector_mu / parallax)
                                                                - exp(-(thick_tic + 1) * detector_thickstep / detector_mu / parallax);
                                        }

                                        // loop over sources now
                                        int source;
                                        for (source = 0; source < sources; ++source) {

                                                // retrieve stuff from cache
                                                vec3 incident = {-source_X(source), -source_Y(source), -source_Z(source)};
                                                CUDAREAL lambda = source_lambda(source);
                                                CUDAREAL source_fraction = source_I(source);

                                                // construct the incident beam unit vector while recovering source distance
                                                // TODO[Giles]: Optimization! We can unitize the source vectors before passing them in.
                                                incident.normalize();

                                                // construct the scattering vector for this pixel
                                                vec3 scattering = (diffracted - incident) / lambda;
                                                CUDAREAL stol = 0.5 * scattering.length();

                                                // rough cut to speed things up when we aren't using whole detector
                                                if (dmin > 0.0 && stol > 0.0) {
                                                        // use reciprocal of (dmin > 0.5 / stol)
                                                        if (dmin_r <= 2 * stol) {
                                                                continue;
                                                        }
                                                }

                                                // polarization factor
                                                if (!nopolar) {
                                                        // need to compute polarization factor
                                                        polar = polarization_factor(polarization, incident, diffracted, polar_vector_tmp);
                                                } else {
                                                        polar = 1.0;
                                                }

                                                // sweep over phi angles
                                                for (int phi_tic = 0; phi_tic < phisteps; ++phi_tic) {
                                                        // enumerate mosaic domains
                                                        for (int mos_tic = 0; mos_tic < mosaic_domains; ++mos_tic) {
                                                                // apply mosaic rotation after phi rotation
                                                                auto a = crystal_orientation(phi_tic, mos_tic, 0);
                                                                auto b = crystal_orientation(phi_tic, mos_tic, 1);
                                                                auto c = crystal_orientation(phi_tic, mos_tic, 2);

                                                                // construct fractional Miller indicies
                                                                CUDAREAL h = a.dot(scattering);
                                                                CUDAREAL k = b.dot(scattering);
                                                                CUDAREAL l = c.dot(scattering);

                                                                // round off to nearest whole index
                                                                int h0 = ceil(h - 0.5);
                                                                int k0 = ceil(k - 0.5);
                                                                int l0 = ceil(l - 0.5);

                                                                // structure factor of the lattice (paralelpiped crystal)
                                                                // F_latt = sin(M_PI*s_Na*h)*sin(M_PI*s_Nb*k)*sin(M_PI*s_Nc*l)/sin(M_PI*h)/sin(M_PI*k)/sin(M_PI*l);

                                                                CUDAREAL I_latt = 1.0; // Shape transform for the crystal.
                                                                CUDAREAL hrad_sqr = 0.0;

                                                                // handy radius in reciprocal space, squared
                                                                hrad_sqr = (h - h0) * (h - h0) * Na * Na + (k - k0) * (k - k0) * Nb * Nb + (l - l0) * (l - l0) * Nc * Nc;
                                                                // fudge the radius so that volume and FWHM are similar to square_xtal spots
                                                                double my_arg = hrad_sqr / 0.63 * fudge;
                                                                {
                                                                CUDAREAL F_latt = Na * Nb * Nc * exp(-(my_arg));
                                                                I_latt = F_latt * F_latt;
                                                                }

                                                                // new code for diffuse.
                                                                if (use_diffuse) {
                                                                  KOKKOS_VEC3 H_vec(h,k,l);
                                                                  KOKKOS_VEC3 H0(h0,k0,l0);
                                                                  CUDAREAL step_diffuse_param[6];
                                                                  // ***NEEDS UPDATE: use legacy API for passing diffuse scale as KOKKOS_MAT3
                                                                  calc_diffuse_at_hkl(H_vec,H0,dHH,s_h_min,s_k_min,s_l_min,s_h_max,s_k_max,
                                                                    s_l_max,s_h_range,s_k_range,s_l_range,diffuse_scale_mat3(0),Fhkl,num_laue_mats,laue_mats,
                                                                    anisoG_local,dG_trace,anisoG_determ,anisoU_local,dG_dgam,false,&I_latt,step_diffuse_param);
                                                                }
                                                                // end s_use_diffuse outer

                                                                // structure factor of the unit cell
                                                                CUDAREAL F_cell = default_F;
                                                                //F_cell = quickFcell_ldg(s_hkls, s_h_max, s_h_min, s_k_max, s_k_min, s_l_max, s_l_min, h0, k0, l0, s_h_range, s_k_range, s_l_range, default_F, Fhkl);
                                                                if (
                                                                        h0 < s_h_min ||
                                                                        k0 < s_k_min ||
                                                                        l0 < s_l_min ||
                                                                        h0 > s_h_max ||
                                                                        k0 > s_k_max ||
                                                                        l0 > s_l_max
                                                                        ) {
                                                                        F_cell = 0.;
                                                                } else {
                                                                        const int hkl_index = (h0-s_h_min)*s_k_range*s_l_range + (k0-s_k_min)*s_l_range + (l0-s_l_min);
                                                                        F_cell = Fhkl[hkl_index];
                                                                }

                                                                // now we have the structure factor for this pixel

                                                                // convert amplitudes into intensity (photons per steradian)
                                                                I += F_cell * F_cell * I_latt * source_fraction * capture_fraction * omega_pixel;
                                                                omega_sub_reduction += omega_pixel;
                                                        } // end of mosaic loop
                                                } // end of phi loop
                                        } // end of source loop
                                } // end of detector thickness loop
                        } // end of sub-pixel y loop
                } // end of sub-pixel x loop
                const double photons = I_bg + I_factor * polar * I;
                floatimage( j ) = photons;
                omega_reduction( j ) = omega_sub_reduction; // shared contention
                max_I_x_reduction( j ) = max_I_x_sub_reduction;
                max_I_y_reduction( j ) = max_I_y_sub_reduction;
                rangemap( j ) = true;
        });
}

void debranch_maskall_low_memory_Kernel(int npanels, int spixels, int fpixels, int total_pixels,
    int oversample, int point_pixel,
    CUDAREAL pixel_size, CUDAREAL subpixel_size, int steps,
    CUDAREAL detector_thickstep, int detector_thicksteps, CUDAREAL detector_thick, CUDAREAL detector_mu,
    const view_1d_t<vec3> sdet_vector, const view_1d_t<vec3> fdet_vector,
    const view_1d_t<vec3> odet_vector, const view_1d_t<vec3> pix0_vector,
    const vector_cudareal_t close_distance,
    const vector_cudareal_t beam_vector,
    CUDAREAL dmin, int phisteps, int sources,
    const vector_cudareal_t source_X, const vector_cudareal_t source_Y,
    const vector_cudareal_t source_Z,
    const vector_cudareal_t source_I, const vector_cudareal_t source_lambda,
    int mosaic_domains, crystal_orientation_t crystal_orientation,
    CUDAREAL Na, CUDAREAL Nb, CUDAREAL Nc, CUDAREAL V_cell,
    CUDAREAL water_size, CUDAREAL water_F, CUDAREAL water_MW, CUDAREAL r_e_sqr,
    CUDAREAL fluence, CUDAREAL Avogadro, CUDAREAL spot_scale, int integral_form,
    CUDAREAL default_F,
    const vector_cudareal_t Fhkl,
    const hklParams FhklParams,
    int nopolar,
    const vector_cudareal_t polar_vector, CUDAREAL polarization, CUDAREAL fudge,
    const vector_size_t pixel_lookup,
    simtbx::Kokkos::diffuse_api diffuse,
    vector_float_t floatimage /*out*/,
    vector_float_t omega_reduction /*out*/, vector_float_t max_I_x_reduction /*out*/,
    vector_float_t max_I_y_reduction /*out*/, vector_bool_t rangemap) {


        const int s_h_min = FhklParams.h_min;
        const int s_k_min = FhklParams.k_min;
        const int s_l_min = FhklParams.l_min;
        const int s_h_range = FhklParams.h_range;
        const int s_k_range = FhklParams.k_range;
        const int s_l_range = FhklParams.l_range;
        const int s_h_max = s_h_min + s_h_range - 1;
        const int s_k_max = s_k_min + s_k_range - 1;
        const int s_l_max = s_l_min + s_l_range - 1;

        // set up diffuse scattering if needed
        vector_mat3_t laue_mats = vector_mat3_t("laue_mats",24);
        vector_cudareal_t dG_trace = vector_cudareal_t("dG_trace",3);
        vector_vec3_t dG_dgam = vector_vec3_t("dG_dgam",3);
        int num_laue_mats = 0;
        int dhh = 0, dkk = 0, dll = 0;
        KOKKOS_MAT3 rotate_principal_axes = diffuse.rotate_principal_axes; // (1,0,0,0,1,0,0,0,1);
        int laue_group_num = diffuse.laue_group_num;
        int stencil_size = diffuse.stencil_size;
        KOKKOS_MAT3 anisoG = diffuse.anisoG; // (300.,0,0,0,100.,0,0,0,300.);
        KOKKOS_MAT3 anisoU = diffuse.anisoU; // (0.48,0,0,0,0.16,0,0,0,0.16);
        KOKKOS_MAT3 Bmat_realspace(1.,0,0,0,1.,0,0,0,1.); // Placeholder
        KOKKOS_MAT3 anisoG_local;
        CUDAREAL anisoG_determ = 0;
        KOKKOS_MAT3 anisoU_local;
        bool use_diffuse=diffuse.enable;
        // ***NEEDS UPDATE: use legacy API for passing diffuse scale as KOKKOS_MAT3
        vector_mat3_t diffuse_scale_mat3 = vector_mat3_t("diffuse_scale_mat3",1);

        if (use_diffuse){
          anisoG_local = anisoG;
          anisoU_local = anisoU;

          if (laue_group_num < 1 || laue_group_num >14 ){
            throw std::string("Laue group number not in range 1-14");
          }

          Kokkos::parallel_reduce("prepare diffuse mats", 1, KOKKOS_LAMBDA (const int& i, int& num_laue_mats_temp){
            num_laue_mats_temp = gen_laue_mats(laue_group_num, laue_mats, rotate_principal_axes);
            // KOKKOS_MAT3 rotate_principal_axes;
            // rotate_principal_axes << 0.70710678, -0.70710678, 0., 0.70710678, 0.70710678, 0., 0., 0., 1.;

            KOKKOS_MAT3 Amatrix(diffuse.a0[0],diffuse.a0[1],diffuse.a0[2],diffuse.b0[0],diffuse.b0[1],diffuse.b0[2],
                                diffuse.c0[0],diffuse.c0[1],diffuse.c0[2]);
            KOKKOS_MAT3 Ainv = Amatrix.inverse()*1.e-10;
            CUDAREAL reciprocal_space_volume = 8*M_PI*M_PI*M_PI*Ainv.determinant();
            CUDAREAL _tmpfac = M_PI * 0.63 / fudge;
            CUDAREAL diffuse_scale = reciprocal_space_volume * sqrt(_tmpfac*_tmpfac*_tmpfac);

            // ***NEEDS UPDATE: use legacy API for passing diffuse scale as KOKKOS_MAT3
            diffuse_scale_mat3(0)(0,0) = diffuse_scale;

            for ( int iL = 0; iL < num_laue_mats_temp; iL++ ){
              laue_mats(iL) = Ainv * laue_mats(iL);
            }
            const KOKKOS_MAT3 Ginv = anisoG_local.inverse();
            const KOKKOS_MAT3 dG = Bmat_realspace * Ginv;
            for (int i_gam=0; i_gam<3; i_gam++){
              dG_dgam(i_gam)[i_gam] = 1;
              KOKKOS_MAT3 temp_dgam;
              temp_dgam(i_gam, 0) = dG_dgam(i_gam)[0];
              temp_dgam(i_gam, 1) = dG_dgam(i_gam)[1];
              temp_dgam(i_gam, 2) = dG_dgam(i_gam)[2];
              dG_trace(i_gam) = (Ginv*temp_dgam).trace();
            }
          }, num_laue_mats);
          anisoG_determ = anisoG_local.determinant();
          dhh = dkk = dll = stencil_size; // Limits of stencil for diffuse calc
        }
        KOKKOS_VEC3 dHH(dhh,dkk,dll);
        // end of diffuse setup

// Implementation notes.  This kernel is aggressively debranched, therefore the assumptions are:
// 1) mosaicity non-zero positive
// 2) xtal shape is "Gauss" i.e. 3D spheroid.
// 3) No bounds check for access to the structure factor array.
// 4) No check for Flatt=0.
        const CUDAREAL dmin_r = (dmin > 0.0) ? 1/dmin : 0.0;

        // add background from something amorphous, precalculate scaling
        const CUDAREAL F_bg = water_F;
        const CUDAREAL I_bg = F_bg * F_bg * r_e_sqr * fluence * water_size * water_size * water_size * 1e6 * Avogadro / water_MW;
        const CUDAREAL I_factor = r_e_sqr * spot_scale * fluence / steps;

        Kokkos::parallel_for("debranch_maskall", total_pixels, KOKKOS_LAMBDA(const int& pixIdx) {

                vec3 polar_vector_tmp {polar_vector(1), polar_vector(2), polar_vector(3)};

                // position in pixel array
                const int j = pixel_lookup(pixIdx);//pixIdx: index into pixel subset; j: index into the data.
                const int i_panel = j / (fpixels*spixels); // the panel number
                const int j_panel = j % (fpixels*spixels); // the pixel number within the panel
                const int fpixel = j_panel % fpixels;
                const int spixel = j_panel / fpixels;
                const int outIdx = pixIdx;

                // reset photon count for this pixel
                CUDAREAL I = I_bg;
                CUDAREAL omega_sub_reduction = 0.0;
                CUDAREAL max_I_x_sub_reduction = 0.0;
                CUDAREAL max_I_y_sub_reduction = 0.0;
                CUDAREAL polar = 0.0;
                if (nopolar) {
                        polar = 1.0;
                }

                // add this now to avoid problems with skipping later
                // move this to the bottom to avoid accessing global device memory. floatimage[j] = I_bg;
                // loop over sub-pixels
                int subS, subF;
                for (subS = 0; subS < oversample; ++subS) { // Y voxel
                        for (subF = 0; subF < oversample; ++subF) { // X voxel
                                // absolute mm position on detector (relative to its origin)
                                CUDAREAL Fdet = subpixel_size * (fpixel * oversample + subF) + subpixel_size / 2.0; // X voxel
                                CUDAREAL Sdet = subpixel_size * (spixel * oversample + subS) + subpixel_size / 2.0; // Y voxel
                                // Fdet = pixel_size*fpixel;
                                // Sdet = pixel_size*spixel;

                                max_I_x_sub_reduction = Fdet;
                                max_I_y_sub_reduction = Sdet;

                                int thick_tic;
                                for (thick_tic = 0; thick_tic < detector_thicksteps; ++thick_tic) {
                                        // assume "distance" is to the front of the detector sensor layer
                                        CUDAREAL Odet = thick_tic * detector_thickstep; // Z Orthagonal voxel.

                                        // construct detector subpixel position in 3D space
                                        //                      pixel_X = distance;
                                        //                      pixel_Y = Sdet-Ybeam;
                                        //                      pixel_Z = Fdet-Xbeam;
                                        vec3 pixel_pos;
                                        pixel_pos += Fdet * fdet_vector(i_panel);
                                        pixel_pos += Sdet * sdet_vector(i_panel);
                                        pixel_pos += Odet * odet_vector(i_panel);
                                        pixel_pos += pix0_vector(i_panel);

                                        // construct the diffracted-beam unit vector to this sub-pixel
                                        CUDAREAL airpath_r = 1 / pixel_pos.length();
                                        vec3 diffracted = pixel_pos.get_unit_vector();

                                        // solid angle subtended by a pixel: (pix/airpath)^2*cos(2theta)
                                        CUDAREAL omega_pixel = pixel_size * pixel_size * airpath_r * airpath_r * close_distance(i_panel) * airpath_r;
                                        // option to turn off obliquity effect, inverse-square-law only
                                        if (point_pixel) {
                                                omega_pixel = airpath_r * airpath_r;
                                        }

                                        // now calculate detector thickness effects
                                        CUDAREAL capture_fraction = 1.0;
                                        if (detector_thick > 0.0 && detector_mu> 0.0) {
                                                // inverse of effective thickness increase
                                                CUDAREAL parallax = odet_vector(i_panel).dot(diffracted);
                                                capture_fraction = exp(-thick_tic * detector_thickstep / detector_mu / parallax)
                                                                - exp(-(thick_tic + 1) * detector_thickstep / detector_mu / parallax);
                                        }

                                        // loop over sources now
                                        int source;
                                        for (source = 0; source < sources; ++source) {

                                                // retrieve stuff from cache
                                                vec3 incident = {-source_X(source), -source_Y(source), -source_Z(source)};
                                                CUDAREAL lambda = source_lambda(source);
                                                CUDAREAL source_fraction = source_I(source);

                                                // construct the incident beam unit vector while recovering source distance
                                                // TODO[Giles]: Optimization! We can unitize the source vectors before passing them in.
                                                incident.normalize();

                                                // construct the scattering vector for this pixel
                                                vec3 scattering = (diffracted - incident) / lambda;
                                                CUDAREAL stol = 0.5 * scattering.length();

                                                // rough cut to speed things up when we aren't using whole detector
                                                if (dmin > 0.0 && stol > 0.0) {
                                                        // use reciprocal of (dmin > 0.5 / stol)
                                                        if (dmin_r <= 2 * stol) {
                                                                continue;
                                                        }
                                                }

                                                // polarization factor
                                                if (!nopolar) {
                                                        // need to compute polarization factor
                                                        polar = polarization_factor(polarization, incident, diffracted, polar_vector_tmp);
                                                } else {
                                                        polar = 1.0;
                                                }

                                                // sweep over phi angles
                                                for (int phi_tic = 0; phi_tic < phisteps; ++phi_tic) {
                                                        // enumerate mosaic domains
                                                        for (int mos_tic = 0; mos_tic < mosaic_domains; ++mos_tic) {
                                                                // apply mosaic rotation after phi rotation
                                                                auto a = crystal_orientation(phi_tic, mos_tic, 0);
                                                                auto b = crystal_orientation(phi_tic, mos_tic, 1);
                                                                auto c = crystal_orientation(phi_tic, mos_tic, 2);

                                                                // construct fractional Miller indicies
                                                                CUDAREAL h = a.dot(scattering);
                                                                CUDAREAL k = b.dot(scattering);
                                                                CUDAREAL l = c.dot(scattering);

                                                                // round off to nearest whole index
                                                                int h0 = ceil(h - 0.5);
                                                                int k0 = ceil(k - 0.5);
                                                                int l0 = ceil(l - 0.5);

                                                                // structure factor of the lattice (paralelpiped crystal)
                                                                // F_latt = sin(M_PI*s_Na*h)*sin(M_PI*s_Nb*k)*sin(M_PI*s_Nc*l)/sin(M_PI*h)/sin(M_PI*k)/sin(M_PI*l);

                                                                CUDAREAL I_latt = 1.0; // Shape transform for the crystal.
                                                                CUDAREAL hrad_sqr = 0.0;

                                                                // handy radius in reciprocal space, squared
                                                                hrad_sqr = (h - h0) * (h - h0) * Na * Na + (k - k0) * (k - k0) * Nb * Nb + (l - l0) * (l - l0) * Nc * Nc;
                                                                // fudge the radius so that volume and FWHM are similar to square_xtal spots
                                                                double my_arg = hrad_sqr / 0.63 * fudge;
                                                                {
                                                                CUDAREAL F_latt = Na * Nb * Nc * exp(-(my_arg));
                                                                I_latt = F_latt * F_latt;
                                                                }

                                                                // new code for diffuse.
                                                                if (use_diffuse) {
                                                                  KOKKOS_VEC3 H_vec(h,k,l);
                                                                  KOKKOS_VEC3 H0(h0,k0,l0);
                                                                  CUDAREAL step_diffuse_param[6];
                                                                  // ***NEEDS UPDATE: use legacy API for passing diffuse scale as KOKKOS_MAT3
                                                                  calc_diffuse_at_hkl(H_vec,H0,dHH,s_h_min,s_k_min,s_l_min,s_h_max,s_k_max,
                                                                    s_l_max,s_h_range,s_k_range,s_l_range,diffuse_scale_mat3(0),Fhkl,num_laue_mats,laue_mats,
                                                                    anisoG_local,dG_trace,anisoG_determ,anisoU_local,dG_dgam,false,&I_latt,step_diffuse_param);
                                                                }
                                                                // end s_use_diffuse outer

                                                                // structure factor of the unit cell
                                                                CUDAREAL F_cell = default_F;
                                                                //F_cell = quickFcell_ldg(s_hkls, s_h_max, s_h_min, s_k_max, s_k_min, s_l_max, s_l_min, h0, k0, l0, s_h_range, s_k_range, s_l_range, default_F, Fhkl);
                                                                if (
                                                                        h0 < s_h_min ||
                                                                        k0 < s_k_min ||
                                                                        l0 < s_l_min ||
                                                                        h0 > s_h_max ||
                                                                        k0 > s_k_max ||
                                                                        l0 > s_l_max
                                                                        ) {
                                                                        F_cell = 0.;
                                                                } else {
                                                                        const int hkl_index = (h0-s_h_min)*s_k_range*s_l_range + (k0-s_k_min)*s_l_range + (l0-s_l_min);
                                                                        F_cell = Fhkl[hkl_index];
                                                                }

                                                                // now we have the structure factor for this pixel

                                                                // convert amplitudes into intensity (photons per steradian)
                                                                I += F_cell * F_cell * I_latt * source_fraction * capture_fraction * omega_pixel;
                                                                omega_sub_reduction += omega_pixel;
                                                        } // end of mosaic loop
                                                } // end of phi loop
                                        } // end of source loop
                                } // end of detector thickness loop
                        } // end of sub-pixel y loop
                } // end of sub-pixel x loop
                const double photons = I_bg + I_factor * polar * I;
                floatimage( outIdx ) = photons;
                //omega_reduction( outIdx ) = omega_sub_reduction; // shared contention
                //max_I_x_reduction( outIdx ) = max_I_x_sub_reduction;
                //max_I_y_reduction( outIdx ) = max_I_y_sub_reduction;
                //rangemap( outIdx ) = true;
        });
}

// __global__ void nanoBraggSpotsInitCUDAKernel(int spixels, int fpixesl, float * floatimage, float * omega_reduction,
//                 float * max_I_x_reduction,
//                 float * max_I_y_reduction, bool * rangemap);

template <typename T, typename U>
void add_array( view_1d_t<T> lhs, const view_1d_t<U> rhs ) {
  Kokkos::parallel_for("add_arrays", lhs.span(), KOKKOS_LAMBDA(const int& i) {
    lhs( i ) = lhs( i ) + (T)rhs( i );
    rhs( i ) = 0;
  });
}

template <typename T, typename U>
void add_array_limit( view_1d_t<T> lhs, const view_1d_t<U> rhs, const std::size_t& limit ) {
  Kokkos::parallel_for("add_arrays", limit, KOKKOS_LAMBDA(const int& i) {
    lhs( i ) = lhs( i ) + (T)rhs( i );
    rhs( i ) = 0;
  });
}

void add_background_kokkos_kernel(int sources, int nanoBragg_oversample, int override_source,
    CUDAREAL pixel_size, int spixels, int fpixels, int detector_thicksteps,
    CUDAREAL detector_thickstep, CUDAREAL detector_attnlen,
    const view_1d_t<vec3> sdet_vector, const view_1d_t<vec3> fdet_vector,
    const view_1d_t<vec3> odet_vector, const view_1d_t<vec3> pix0_vector,
    CUDAREAL close_distance, int point_pixel, CUDAREAL detector_thick,
    const vector_cudareal_t  source_X, const vector_cudareal_t  source_Y,
    const vector_cudareal_t  source_Z,
    const vector_cudareal_t  source_lambda, const vector_cudareal_t  source_I,
    int stols, const vector_cudareal_t stol_of, const vector_cudareal_t Fbg_of,
    int nopolar, CUDAREAL polarization, const vector_cudareal_t  polar_vector,
    CUDAREAL r_e_sqr, CUDAREAL fluence, CUDAREAL amorphous_molecules,
    vector_float_t floatimage)
{

    int oversample=-1;                     //override features that usually slow things down,
                                           //like oversampling pixels & multiple sources
    int source_start = 0;
    CUDAREAL n_source_scale = sources;
    // allow user to override automated oversampling decision at call time with arguments
    if(oversample<=0) oversample = nanoBragg_oversample;
    if(oversample<=0) oversample = 1;
    bool full_spectrum = true;
    if(override_source>=0) {
        // user-specified source in the argument
        source_start = override_source;
        sources = source_start +1;
        full_spectrum = false;
    }
    n_source_scale *= !full_spectrum;
    // make sure we are normalizing with the right number of sub-steps
    int steps = oversample*oversample;
    CUDAREAL subpixel_size = pixel_size/oversample;

    // sweep over detector
    const int total_pixels = spixels * fpixels;
    // const int fstride = gridDim.x * blockDim.x;
    // const int sstride = gridDim.y * blockDim.y;
    // const int stride = fstride * sstride;
    Kokkos::parallel_for("add_background", total_pixels, KOKKOS_LAMBDA(const int& pixIdx) {

        vec3 polar_vector_tmp {polar_vector(1), polar_vector(2), polar_vector(3)};

        const int fpixel = pixIdx % fpixels;
        const int spixel = pixIdx / fpixels;
        // reset background photon count for this pixel
        CUDAREAL Ibg = 0;
        int nearest = 0; // sort-stable alogorithm, instead of holding value over from previous pixel
        // loop over sub-pixels
        for(int subS=0; subS<oversample; ++subS) {
            for(int subF=0; subF<oversample; ++subF) {
                // absolute mm position on detector (relative to its origin)
                CUDAREAL Fdet = subpixel_size*(fpixel*oversample + subF ) + subpixel_size/2.0;
                CUDAREAL Sdet = subpixel_size*(spixel*oversample + subS ) + subpixel_size/2.0;

                for(int thick_tic=0; thick_tic<detector_thicksteps; ++thick_tic) {
                    // assume "distance" is to the front of the detector sensor layer
                    CUDAREAL Odet = thick_tic*detector_thickstep;

                    vec3 pixel_pos = Fdet * fdet_vector(0) + Sdet * sdet_vector(0) + Odet * odet_vector(0) + pix0_vector(0);

                    // no curved detector option (future implementation)
                    // construct the diffracted-beam unit vector to this pixel
                    CUDAREAL airpath = pixel_pos.length();
                    vec3 diffracted = pixel_pos.get_unit_vector();

                    // solid angle subtended by a pixel: (pix/airpath)^2*cos(2theta)
                    CUDAREAL omega_pixel = pixel_size*pixel_size/airpath/airpath*close_distance/airpath;
                    // option to turn off obliquity effect, inverse-square-law only
                    if(point_pixel) omega_pixel = 1.0/airpath/airpath;

                    // now calculate detector thickness effects
                    CUDAREAL capture_fraction = 1.0;
                    if(detector_thick > 0.0){
                        // inverse of effective thickness increase
                        // CUDAREAL parallax = diffracted[1] * odet_vector(1) + diffracted[2] * odet_vector(2) + diffracted[3] * odet_vector(3);
                        CUDAREAL parallax = diffracted.dot(odet_vector(0));
                        capture_fraction = exp(-thick_tic*detector_thickstep/detector_attnlen/parallax)
                                            -exp(-(thick_tic+1)*detector_thickstep/detector_attnlen/parallax);
                    }

                    // loop over sources now
                    for(int source=source_start; source<sources; ++source) {

                        // retrieve stuff from cache
                        vec3 incident {-source_X(source), -source_Y(source), -source_Z(source)};
                        CUDAREAL lambda = source_lambda(source);
                        CUDAREAL source_fraction = full_spectrum * source_I(source) + n_source_scale;
                        // construct the incident beam unit vector while recovering source distance
                        incident.normalize();

                        // construct the scattering vector for this pixel
                        vec3 scattering = (diffracted - incident) / lambda;
                        // sin(theta)/lambda is half the scattering vector length
                        CUDAREAL stol = 0.5*scattering.length();

                        // now we need to find the nearest four "stol file" points
                        while(stol > stol_of(nearest) && nearest <= stols){ ++nearest; };
                        while(stol < stol_of(nearest) && nearest >= 2){ --nearest; };

                        // cubic spline interpolation
                        CUDAREAL Fbg;
                        CUDAREAL stol_points[4], Fbg_points[4];
                        stol_points[0] = stol_of(nearest-1);
                        stol_points[1] = stol_of(nearest-1+1);
                        stol_points[2] = stol_of(nearest-1+2);
                        stol_points[3] = stol_of(nearest-1+3);
                        Fbg_points[0] = Fbg_of(nearest-1);
                        Fbg_points[1] = Fbg_of(nearest-1+1);
                        Fbg_points[2] = Fbg_of(nearest-1+2);
                        Fbg_points[3] = Fbg_of(nearest-1+3);

                        polint(stol_points, Fbg_points, stol, &Fbg);

                        // allow negative F values to yield negative intensities
                        CUDAREAL sign=1.0;
                        if(Fbg<0.0) sign=-1.0;

                        // now we have the structure factor for this pixel

                        // polarization factor
                        CUDAREAL polar = 1.0;
                        if(! nopolar){
                            // need to compute polarization factor
                        //     CUDAREAL axis[] = {polar_vector(0), polar_vector(1), polar_vector(2), polar_vector(3)};
                        //     polar = polarization_factor(polarization, incident, diffracted, polar_vector);
                            polar = polarization_factor(polarization, incident, diffracted, polar_vector_tmp);
                        }

                        // accumulate unscaled pixel intensity from this
                        Ibg += sign*Fbg*Fbg*polar*omega_pixel*capture_fraction*source_fraction;
                    } // end of source loop
                } // end of detector thickness loop
            } // end of sub-pixel y loop
        } // end of sub-pixel x loop
        // save photons/pixel (if fluence specified), or F^2/omega if no fluence given
        floatimage(pixIdx) += Ibg*r_e_sqr*fluence*amorphous_molecules/steps;
    }); // end of pixIdx loop
}
#endif // SIMTBX_KOKKOS_SIMULATION_KERNELS_H
