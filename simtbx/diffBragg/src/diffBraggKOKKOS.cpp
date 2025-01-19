#include <sys/time.h>
#include <cstdio>

#include "diffBraggKOKKOS.h"
#include "diffBragg_kokkos_kernel.h"

#define PRINTOUT(flag, function, ...) \
    if (flag) {                       \
        function<true>(__VA_ARGS__);  \
    } else {                          \
        function<false>(__VA_ARGS__); \
    }                                 \

uint32_t combine_refinement_flags(flags& db_flags) {
    uint32_t refine_flag = 0;
    refine_flag |= db_flags.refine_diffuse * REFINE_DIFFUSE;
    refine_flag |= db_flags.refine_fcell * REFINE_FCELL;
    refine_flag |= db_flags.refine_eta * REFINE_ETA;
    refine_flag |= db_flags.refine_Umat[0] * REFINE_UMAT1;
    refine_flag |= db_flags.refine_Umat[1] * REFINE_UMAT2;
    refine_flag |= db_flags.refine_Umat[2] * REFINE_UMAT3;
    refine_flag |= db_flags.refine_Ncells_def * REFINE_NCELLS_DEF;
    refine_flag |= db_flags.refine_Ncells[0] * REFINE_NCELLS1;
    refine_flag |= db_flags.refine_Ncells[1] * REFINE_NCELLS2;
    refine_flag |= db_flags.refine_Ncells[2] * REFINE_NCELLS3;
    refine_flag |= db_flags.refine_panel_rot[0] * REFINE_PANEL_ROT1;
    refine_flag |= db_flags.refine_panel_rot[1] * REFINE_PANEL_ROT2;
    refine_flag |= db_flags.refine_panel_rot[2] * REFINE_PANEL_ROT3;
    refine_flag |= db_flags.refine_panel_origin[0] * REFINE_PANEL_ORIGIN1;
    refine_flag |= db_flags.refine_panel_origin[1] * REFINE_PANEL_ORIGIN2;
    refine_flag |= db_flags.refine_panel_origin[2] * REFINE_PANEL_ORIGIN3;
    refine_flag |= db_flags.refine_lambda[0] * REFINE_LAMBDA1;
    refine_flag |= db_flags.refine_lambda[1] * REFINE_LAMBDA2;
    refine_flag |= db_flags.refine_Bmat[0] * REFINE_BMAT1;
    refine_flag |= db_flags.refine_Bmat[1] * REFINE_BMAT2;
    refine_flag |= db_flags.refine_Bmat[2] * REFINE_BMAT3;
    refine_flag |= db_flags.refine_Bmat[3] * REFINE_BMAT4;
    refine_flag |= db_flags.refine_Bmat[4] * REFINE_BMAT5;
    refine_flag |= db_flags.refine_Bmat[5] * REFINE_BMAT6;
    refine_flag |= db_flags.refine_fp_fdp * REFINE_FP_FDP;
    refine_flag |= db_flags.refine_Icell * REFINE_ICELL;

    return refine_flag;
}

void diffBraggKOKKOS::diffBragg_sum_over_steps_kokkos(
    int Npix_to_model,
    std::vector<unsigned int>& panels_fasts_slows,
    image_type& floatimage,
    images& d_image,
    images& d2_image,
    step_arrays& db_steps,
    detector& db_det,
    beam& db_beam,
    crystal& db_cryst,
    flags& db_flags,
    cuda_flags& db_cu_flags,
    // diffBragg_kokkosPointers& kp,
    timer_variables& TIMERS) {
    if (db_cryst.phi0 != 0 || db_cryst.phisteps > 1) {
        printf(
            "PHI (goniometer position) not supported in GPU code: phi0=%f phisteps=%d, phistep=%f\n",
            db_cryst.phi0, db_cryst.phisteps, db_cryst.phistep);
        exit(-1);
    }

    Kokkos::Tools::pushRegion("diffBragg_sum_over_steps_kokkos");
    Kokkos::Tools::pushRegion("local detector, beam and crystal");
    kokkos_detector local_det(db_det);
    kokkos_beam local_beam(db_beam);
    // kokkos_crystal local_cryst(db_cryst);
    Kokkos::Tools::popRegion();

    double time;
    struct timeval t1, t;  // t1 times larger blocks of code, and t is used to time shorter blocks of code
    gettimeofday(&t1, 0);

    //  determine if we need to allocate pixels, and how many.
    //  For best usage, one should use the diffBragg property (visible from Python) Npix_to_allocate
    //  in order to just allocate to the GPU - this is useful for ensemble refinement, where each
    //  shot can have a variable number of pixels being modeled, and ony only needs to allocate the
    //  device once (with the largest expected number of pixels for a given shot)
    // TODO clean up this logic a bit
    if (m_device_is_allocated && (Npix_to_model > m_npix_allocated)) {
        printf(
            "Need to re-allocate pixels, currently have %d allocated, but trying to model %d\n",
            m_npix_allocated, Npix_to_model);
        exit(-1);
    } else if (db_cu_flags.Npix_to_allocate == -1) {
        db_cu_flags.Npix_to_allocate = Npix_to_model;
    } else if (Npix_to_model > db_cu_flags.Npix_to_allocate) {
        printf(
            "Npix to model=%d is greater than the number of pixel requested for allocation (%d)!\n",
            Npix_to_model, db_cu_flags.Npix_to_allocate);
        exit(-1);
    }

    //  support dynamic allocation for different numbers of sources
    Kokkos::Tools::pushRegion("resize sources");
    if (m_previous_nsource != 0 && m_previous_nsource != local_beam.number_of_sources) {
        // printf("Resizing for %d sources!:\n", local_beam.number_of_sources);
        resize(m_Fhkl_channels, local_beam.number_of_sources);
        resize(m_source_X, local_beam.number_of_sources);
        resize(m_source_Y, local_beam.number_of_sources);
        resize(m_source_Z, local_beam.number_of_sources);
        resize(m_source_I, local_beam.number_of_sources);
        resize(m_source_lambda, local_beam.number_of_sources);
        m_previous_nsource = local_beam.number_of_sources;
    }
    Kokkos::Tools::popRegion();

    if (m_device_is_allocated) {
        if (db_flags.verbose) {
            printf(
                "Will model %d pixels (GPU has %d pre-allocated pix)\n", Npix_to_model,
                m_npix_allocated);
        }
    } else {
        Kokkos::Tools::pushRegion("Device_allocate");
        if (db_flags.verbose) {
            printf(
                "Will model %d pixels and allocate %d pix\n", Npix_to_model,
                db_cu_flags.Npix_to_allocate);
        }
        // Check the Fhkl geradient arrays
        if (db_flags.Fhkl_have_scale_factors){
            resize(m_data_residual, db_cu_flags.Npix_to_allocate);
            resize(m_data_variance, db_cu_flags.Npix_to_allocate);
            resize(m_data_freq, db_cu_flags.Npix_to_allocate);
            resize(m_data_trusted, db_cu_flags.Npix_to_allocate);
            resize(m_FhklLinear_ASUid, db_cryst.FhklLinear_ASUid.size());
            resize(m_Fhkl_scale, d_image.Fhkl_scale.size());
            // alloc Fhkl_scale_deriv to bs same length as Fhkl_scale.size(), as Fhkl_scale_deriv is only set when Fhkl_gradient_mode=True, typpically not first iteration
            resize(m_Fhkl_scale_deriv, d_image.Fhkl_scale.size());
            m_Fhkl_grad_arrays_allocated = true;
        }

        resize(m_Fhkl_channels, local_beam.number_of_sources);
        resize(m_source_X, local_beam.number_of_sources);
        resize(m_source_Y, local_beam.number_of_sources);
        resize(m_source_Z, local_beam.number_of_sources);
        resize(m_source_I, local_beam.number_of_sources);
        resize(m_source_lambda, local_beam.number_of_sources);
        m_previous_nsource = local_beam.number_of_sources;


        resize(m_UMATS, db_cryst.UMATS.size());
        resize(m_UMATS_RXYZ, db_cryst.UMATS_RXYZ.size());
        resize(m_AMATS, db_cryst.UMATS_RXYZ.size());

        if (db_cryst.UMATS_RXYZ_prime.size() > 0) {
            resize(m_UMATS_RXYZ_prime, db_cryst.UMATS_RXYZ_prime.size());
        }

        if (db_cryst.UMATS_RXYZ_dbl_prime.size() > 0) {
            resize(m_UMATS_RXYZ_dbl_prime, db_cryst.UMATS_RXYZ_dbl_prime.size());
        }

        resize(m_dB_Mats, db_cryst.dB_Mats.size());
        resize(m_dB2_Mats, db_cryst.dB2_Mats.size());

        resize(m_RotMats, db_cryst.RotMats.size());
        resize(m_dRotMats, db_cryst.dRotMats.size());
        resize(m_d2RotMats, db_cryst.d2RotMats.size());

        resize(m_fdet_vectors, local_det.fdet_vectors.size());
        resize(m_sdet_vectors, local_det.sdet_vectors.size());
        resize(m_odet_vectors, local_det.odet_vectors.size());
        resize(m_pix0_vectors, local_det.pix0_vectors.size());
        resize(m_close_distances, local_det.close_distances.size());

        if (db_cryst.fpfdp.size() > 0) {
            resize(m_fpfdp, db_cryst.fpfdp.size());
            resize(m_atom_data, db_cryst.atom_data.size());
        }
        if (db_cryst.fpfdp_derivs.size() > 0) {
            resize(m_fpfdp_derivs, db_cryst.fpfdp_derivs.size());
        }

        resize(m_Fhkl, db_cryst.FhklLinear.size());

        if (db_flags.complex_miller) {
            resize(m_Fhkl2, db_cryst.FhklLinear.size());
        }

        resize(m_dF_vecs, local_det.dF_vecs.size());
        resize(m_dS_vecs, local_det.dF_vecs.size());

        resize(m_floatimage, db_cu_flags.Npix_to_allocate);

        if (db_flags.wavelength_img) {
            resize(m_wavelenimage, 4*db_cu_flags.Npix_to_allocate);
        }
        if (db_flags.refine_diffuse) {
            resize(m_d_diffuse_gamma_images, db_cu_flags.Npix_to_allocate * 3);
            resize(m_d_diffuse_sigma_images, db_cu_flags.Npix_to_allocate * 3);
        }
        if (db_flags.refine_fcell) {
            resize(m_d_fcell_images, db_cu_flags.Npix_to_allocate);
            resize(m_d2_fcell_images, db_cu_flags.Npix_to_allocate);
        }
        if (db_flags.refine_eta) {
            resize(m_d_eta_images, db_cu_flags.Npix_to_allocate * 3);
            resize(m_d2_eta_images, db_cu_flags.Npix_to_allocate * 3);
        }
        if (std::count(db_flags.refine_Umat.begin(), db_flags.refine_Umat.end(), true) > 0) {
            resize(m_d_Umat_images, db_cu_flags.Npix_to_allocate * 3);
            resize(m_d2_Umat_images, db_cu_flags.Npix_to_allocate * 3);
        }
        if (std::count(db_flags.refine_Ncells.begin(), db_flags.refine_Ncells.end(), true) > 0 ||
            db_flags.refine_Ncells_def) {
            resize(m_d_Ncells_images, db_cu_flags.Npix_to_allocate * 6);
            resize(m_d2_Ncells_images, db_cu_flags.Npix_to_allocate * 6);
        }
        if (std::count(db_flags.refine_panel_rot.begin(), db_flags.refine_panel_rot.end(), true) >
            0) {
            resize(m_d_panel_rot_images, db_cu_flags.Npix_to_allocate * 3);
        }

        if (std::count(
                db_flags.refine_panel_origin.begin(), db_flags.refine_panel_origin.end(), true) >
            0) {
            resize(m_d_panel_orig_images, db_cu_flags.Npix_to_allocate * 3);
        }

        if (std::count(db_flags.refine_lambda.begin(), db_flags.refine_lambda.end(), true) > 0) {
            resize(m_d_lambda_images, db_cu_flags.Npix_to_allocate * 2);
        }
        if (std::count(db_flags.refine_Bmat.begin(), db_flags.refine_Bmat.end(), true) > 0) {
            resize(m_d_Bmat_images, db_cu_flags.Npix_to_allocate * 6);
            resize(m_d2_Bmat_images, db_cu_flags.Npix_to_allocate * 6);
        }
        if (db_flags.refine_fp_fdp) {
            resize(m_d_fp_fdp_images, db_cu_flags.Npix_to_allocate * 2);
        }

        if (db_cryst.nominal_hkl.size() > 0) {
            resize(m_nominal_hkl, db_cu_flags.Npix_to_allocate * 3);
        }

        resize(m_panels_fasts_slows, db_cu_flags.Npix_to_allocate * 3);

        m_refine_flag = combine_refinement_flags(db_flags);
        if (m_refine_flag) {
            resize(m_manager_dI, db_cu_flags.Npix_to_allocate);
            resize(m_manager_dI2, db_cu_flags.Npix_to_allocate);
        }

        m_npix_allocated = db_cu_flags.Npix_to_allocate;
        Kokkos::Tools::popRegion();
    }  // END of allocation

    bool ALLOC = !m_device_is_allocated;  // shortcut variable

    easy_time(TIMERS.cuda_alloc, t1, TIMERS.recording); //, db_flags.verbose > 1);

    // ALLOC = false;
    //  BEGIN COPYING DATA
    gettimeofday(&t1, 0);
    bool FORCE_COPY = true;

    //  END step position
    if (db_flags.Fhkl_gradient_mode){
        kokkostbx::transfer_vector2kokkos(m_data_residual, d_image.residual);
        kokkostbx::transfer_vector2kokkos(m_data_variance, d_image.variance);
        kokkostbx::transfer_vector2kokkos(m_data_trusted, d_image.trusted);
        kokkostbx::transfer_vector2kokkos(m_data_freq, d_image.freq);
    }

    if (db_flags.Fhkl_have_scale_factors && ALLOC){
        kokkostbx::transfer_vector2kokkos(m_FhklLinear_ASUid, db_cryst.FhklLinear_ASUid);
    }

    Kokkos::Tools::pushRegion("BEGIN Fhkl have scale factors");
    gettimeofday(&t, 0);
    if (db_flags.Fhkl_have_scale_factors){
        if (db_cu_flags.update_Fhkl_scales || ALLOC){
            kokkostbx::transfer_vector2kokkos(m_Fhkl_scale, d_image.Fhkl_scale);
            db_cu_flags.update_Fhkl_scales = false;
        }
        if (db_cu_flags.update_Fhkl_channels || ALLOC){
            kokkostbx::transfer_vector2kokkos(m_Fhkl_channels, db_beam.Fhkl_channels);
            db_cu_flags.update_Fhkl_channels = false;
        }
        ::Kokkos::deep_copy(m_Fhkl_scale_deriv, 0);
    }
    Kokkos::Tools::popRegion();
    easy_time(TIMERS.copy_Fhkl_scale, t, TIMERS.recording); //, db_flags.verbose > 1);

    //  BEGIN sources
    Kokkos::Tools::pushRegion("BEGIN sources");
    gettimeofday(&t, 0);
    if (db_cu_flags.update_sources || ALLOC) {
        int source_count = local_beam.number_of_sources;
        kokkostbx::transfer_double2kokkos(m_source_X, local_beam.source_X, source_count);
        kokkostbx::transfer_double2kokkos(m_source_Y, local_beam.source_Y, source_count);
        kokkostbx::transfer_double2kokkos(m_source_Z, local_beam.source_Z, source_count);
        kokkostbx::transfer_double2kokkos(m_source_I, local_beam.source_I, source_count);
        kokkostbx::transfer_double2kokkos(m_source_lambda, local_beam.source_lambda, source_count);
        auto tmp_src_X = m_source_X;
        auto tmp_src_Y = m_source_Y;
        auto tmp_src_Z = m_source_Z;
        Kokkos::parallel_for(
            "normalize incident vector", source_count, KOKKOS_LAMBDA(const int& i) {
                KOKKOS_VEC3 incident{tmp_src_X(i), tmp_src_Y(i), tmp_src_Z(i)};
                incident.normalize();
                tmp_src_X(i) = incident.x_val();
                tmp_src_Y(i) = incident.y_val();
                tmp_src_Z(i) = incident.z_val();
            });
        Kokkos::fence();
        if (db_flags.verbose > 1)
            printf("H2D sources\n");
        db_cu_flags.update_sources = false;
    }
    easy_time(TIMERS.copy_sources, t, TIMERS.recording); //, db_flags.verbose > 1);

    Kokkos::Tools::popRegion();
    //  END sources

    //  UMATS
    gettimeofday(&t, 0);
    Kokkos::Tools::pushRegion("UMATS");
    if (db_cu_flags.update_umats || ALLOC || FORCE_COPY) {
        transfer_KOKKOS_MAT3(m_UMATS, db_cryst.UMATS);
        transfer_KOKKOS_MAT3(m_UMATS_RXYZ, db_cryst.UMATS_RXYZ);
        transfer_KOKKOS_MAT3(m_UMATS_RXYZ_prime, db_cryst.UMATS_RXYZ_prime);
        transfer_KOKKOS_MAT3(m_UMATS_RXYZ_dbl_prime, db_cryst.UMATS_RXYZ_dbl_prime);

        if (db_flags.verbose > 1)
            printf("H2D Done copying Umats\n");
    }
    Kokkos::Tools::popRegion();
    easy_time(TIMERS.copy_umats, t, TIMERS.recording); //, db_flags.verbose > 1);
    //  END UMATS

    gettimeofday(&t, 0);
    if (db_cu_flags.update_umats || ALLOC || FORCE_COPY) {
        auto Amat_init = db_cryst.eig_U*db_cryst.eig_B*1e10*(db_cryst.eig_O.transpose());
        auto host_AMATS = Kokkos::create_mirror_view(m_AMATS);
        for (int i=0; i<db_cryst.UMATS_RXYZ.size(); ++i) {
            host_AMATS(i) = to_mat3((db_cryst.UMATS_RXYZ[i]*Amat_init).transpose());
        }
        Kokkos::deep_copy(m_AMATS, host_AMATS);
        if (db_flags.verbose > 1)
            printf("H2D Done copying Amats\n");
    }
    easy_time(TIMERS.copy_amats, t, TIMERS.recording);

    //  BMATS
    gettimeofday(&t, 0);
    Kokkos::Tools::pushRegion("BMATS");
    if (db_cu_flags.update_dB_mats || ALLOC || FORCE_COPY) {
        transfer_KOKKOS_MAT3(m_dB_Mats, db_cryst.dB_Mats);
        transfer_KOKKOS_MAT3(m_dB2_Mats, db_cryst.dB2_Mats);
        if (db_flags.verbose > 1)
            printf("H2D Done copying dB_Mats\n");
    }
    Kokkos::Tools::popRegion();
    easy_time(TIMERS.copy_bmats, t, TIMERS.recording);
    //  END BMATS

    //  ROT MATS
    gettimeofday(&t, 0);
    Kokkos::Tools::pushRegion("ROT MATS");
    if (db_cu_flags.update_rotmats || ALLOC || FORCE_COPY) {
        transfer_KOKKOS_MAT3(m_RotMats, db_cryst.RotMats);
        transfer_KOKKOS_MAT3(m_dRotMats, db_cryst.dRotMats);
        transfer_KOKKOS_MAT3(m_d2RotMats, db_cryst.d2RotMats);
        if (db_flags.verbose > 1)
            printf("H2D Done copying rotmats\n");
    }
    Kokkos::Tools::popRegion();
    easy_time(TIMERS.copy_rotmats, t, TIMERS.recording);
    //  END ROT MATS

    //  DETECTOR VECTORS
    gettimeofday(&t, 0);
    Kokkos::Tools::pushRegion("DETECTOR VECTORS");
    if (db_cu_flags.update_detector || ALLOC) {
        kokkostbx::transfer_vector2kokkos(m_fdet_vectors, local_det.fdet_vectors);
        kokkostbx::transfer_vector2kokkos(m_sdet_vectors, local_det.sdet_vectors);
        kokkostbx::transfer_vector2kokkos(m_odet_vectors, local_det.odet_vectors);
        kokkostbx::transfer_vector2kokkos(m_pix0_vectors, local_det.pix0_vectors);
        kokkostbx::transfer_vector2kokkos(m_close_distances, local_det.close_distances);
        if (db_flags.verbose > 1)
            printf("H2D Done copying detector vectors\n");
        db_cu_flags.update_detector = false;
    }
    Kokkos::Tools::popRegion();
    easy_time(TIMERS.copy_det, t, TIMERS.recording);
    //  END  DETECTOR VECTORS

    gettimeofday(&t, 0);
    if (ALLOC || FORCE_COPY) {
        kokkostbx::transfer_vector2kokkos(m_nominal_hkl, db_cryst.nominal_hkl);
        kokkostbx::transfer_vector2kokkos(m_atom_data, db_cryst.atom_data);
        if (db_flags.verbose > 1)
            printf("H2D Done copying atom data\n");

        kokkostbx::transfer_vector2kokkos(m_fpfdp, db_cryst.fpfdp);
        kokkostbx::transfer_vector2kokkos(m_fpfdp_derivs, db_cryst.fpfdp_derivs);
        if (db_flags.verbose > 1)
            printf("H2D Done copying fprime and fdblprime\n");
    }
    easy_time(TIMERS.copy_nomhkl, t, TIMERS.recording);

    //  BEGIN UPDATE REFINEMENT
    gettimeofday(&t, 0);
    Kokkos::Tools::pushRegion("BEGIN UPDATE REFINMENT");
    if (db_cu_flags.update_refine_flags || ALLOC) {
        kokkostbx::transfer_vector2kokkos(m_refine_Umat, db_flags.refine_Umat);
        kokkostbx::transfer_vector2kokkos(m_refine_Ncells, db_flags.refine_Ncells);
        kokkostbx::transfer_vector2kokkos(m_refine_panel_origin, db_flags.refine_panel_origin);
        kokkostbx::transfer_vector2kokkos(m_refine_panel_rot, db_flags.refine_panel_rot);
        kokkostbx::transfer_vector2kokkos(m_refine_lambda, db_flags.refine_lambda);
        kokkostbx::transfer_vector2kokkos(m_refine_Bmat, db_flags.refine_Bmat);
        if (db_flags.verbose > 1)
            printf("H2D Done copying refinement flags\n");
        db_cu_flags.update_refine_flags=false;
    }
    Kokkos::Tools::popRegion();
    easy_time(TIMERS.copy_flags, t, TIMERS.recording);
    //  END UPDATE REFINEMENT

    //  BEGIN Fhkl
    gettimeofday(&t, 0);
    Kokkos::Tools::pushRegion("Begin Fhkl");
    if (db_cu_flags.update_Fhkl || ALLOC) {
        kokkostbx::transfer_vector2kokkos(m_Fhkl, db_cryst.FhklLinear);
        if (db_flags.complex_miller) {
            kokkostbx::transfer_vector2kokkos(m_Fhkl2, db_cryst.Fhkl2Linear);
        }
        if (db_flags.verbose > 1)
            printf("H2D Done copying step Fhkl\n");
        db_cu_flags.update_Fhkl = false;
    }
    Kokkos::Tools::popRegion();
    easy_time(TIMERS.copy_fhkl, t, TIMERS.recording);
    //  END Fhkl

    //  BEGIN panel derivative vecs
    gettimeofday(&t, 0);
    Kokkos::Tools::pushRegion("BEGIN panel derivative vecs");
    if (db_cu_flags.update_panel_deriv_vecs || ALLOC) {
        kokkostbx::transfer_vector2kokkos(m_dF_vecs, local_det.dF_vecs);
        kokkostbx::transfer_vector2kokkos(m_dS_vecs, local_det.dS_vecs);
        if (db_flags.verbose > 1)
            printf("H2D Done copying step panel derivative vectors\n");
        db_cu_flags.update_panel_deriv_vecs=false;
    }
    Kokkos::Tools::popRegion();
    easy_time(TIMERS.copy_detderiv, t, TIMERS.recording);
    //  END panel derivative vecs

    //  BEGIN panels fasts slows
    gettimeofday(&t, 0);
    Kokkos::Tools::pushRegion("BEGIN panels fasts slows");
    if (db_cu_flags.update_panels_fasts_slows || ALLOC || FORCE_COPY) {
        kokkostbx::transfer_vector2kokkos(m_panels_fasts_slows, panels_fasts_slows);
        if (db_flags.verbose > 1)
            printf("H2D Done copying panels_fasts_slows\n");
    }
    Kokkos::Tools::popRegion();
    easy_time(TIMERS.copy_pfs, t, TIMERS.recording);
    //  END panels fasts slows

    easy_time(TIMERS.cuda_copy_to_dev, t1, TIMERS.recording);

    m_device_is_allocated = true;
    ::Kokkos::fence("after copy to device");

    gettimeofday(&t1, 0);

    int Npanels = local_det.fdet_vectors.size() / 3;
    int num_atoms = db_cryst.atom_data.size() / 5;
    // note cannot use atom data if fpfdp is 0, make this cleaner
    if (db_cryst.fpfdp.size() == 0) {
        num_atoms = 0;
    }

    bool aniso_eta = db_cryst.UMATS_RXYZ.size() != db_cryst.UMATS_RXYZ_prime.size();
    bool use_nominal_hkl = !db_cryst.nominal_hkl.empty();
    if ((db_flags.printout==false) &&
       (db_flags.complex_miller==false) &&
       (db_flags.compute_curvatures==false) &&
       (m_refine_flag==REFINE_FCELL) &&
       (db_flags.use_diffuse==false) &&
       (db_flags.wavelength_img==false) &&
       (db_flags.Fhkl_gradient_mode==false) &&
       (db_flags.Fhkl_errors_mode==false) &&
       (db_flags.using_trusted_mask==false) &&
       (db_beam.Fhkl_channels.empty()==true) &&
       (db_flags.Fhkl_have_scale_factors)==false) {
        kokkos_sum_over_steps<false, false, false, REFINE_FCELL, false, false, false, false, false, true, false>(
            Npix_to_model, m_panels_fasts_slows, m_floatimage, m_wavelenimage, m_d_Umat_images,
            m_d2_Umat_images, m_d_Bmat_images, m_d2_Bmat_images, m_d_Ncells_images, m_d2_Ncells_images,
            m_d_fcell_images, m_d2_fcell_images, m_d_eta_images, m_d2_eta_images, m_d_lambda_images,
            m_d2_lambda_images, m_d_panel_rot_images, m_d2_panel_rot_images, m_d_panel_orig_images,
            m_d2_panel_orig_images, m_d_fp_fdp_images, m_manager_dI, m_manager_dI2, db_steps.Nsteps,
            db_flags.printout_fpixel, db_flags.printout_spixel, /*db_flags.printout,*/ db_cryst.default_F,
            local_det.oversample, db_flags.oversample_omega, local_det.subpixel_size, local_det.pixel_size,
            local_det.detector_thickstep, local_det.detector_thick, m_close_distances,
            local_det.detector_attnlen, local_det.detector_thicksteps, local_beam.number_of_sources,
            db_cryst.phisteps, (int) db_cryst.UMATS.size(), db_flags.use_lambda_coefficients,
            local_beam.lambda0, local_beam.lambda1, to_mat3(db_cryst.eig_U), to_mat3(db_cryst.eig_O),
            to_mat3(db_cryst.eig_B), to_mat3(db_cryst.RXYZ), m_dF_vecs, m_dS_vecs, m_UMATS_RXYZ, m_UMATS_RXYZ_prime,
            m_UMATS_RXYZ_dbl_prime, m_RotMats, m_dRotMats, m_d2RotMats, m_UMATS, m_dB_Mats, m_dB2_Mats,
            m_AMATS, m_source_X, m_source_Y, m_source_Z, m_source_lambda, m_source_I,
            local_beam.kahn_factor, db_cryst.Na, db_cryst.Nb, db_cryst.Nc, db_cryst.Nd,
            db_cryst.Ne, db_cryst.Nf, db_cryst.phi0, db_cryst.phistep,
            to_vec3(db_cryst.spindle_vec), local_beam.polarization_axis, db_cryst.h_range,
            db_cryst.k_range, db_cryst.l_range, db_cryst.h_max, db_cryst.h_min,
            db_cryst.k_max, db_cryst.k_min, db_cryst.l_max, db_cryst.l_min,
            db_cryst.dmin, db_cryst.fudge, /*db_flags.complex_miller,*/ db_flags.verbose,
            db_flags.only_save_omega_kahn, db_flags.isotropic_ncells, /*db_flags.compute_curvatures,*/
            m_Fhkl, m_Fhkl2, /*m_refine_flag,*/
            m_fdet_vectors, m_sdet_vectors, m_odet_vectors,
            m_pix0_vectors, db_flags.nopolar, db_flags.point_pixel, local_beam.fluence,
            db_cryst.r_e_sqr, db_cryst.spot_scale, Npanels, aniso_eta, db_flags.no_Nabc_scale,
            m_fpfdp, m_fpfdp_derivs, m_atom_data, num_atoms, m_nominal_hkl,
            use_nominal_hkl, to_mat3(db_cryst.anisoU), to_mat3(db_cryst.anisoG), to_mat3(db_cryst.rotate_principal_axes),
            /*db_flags.use_diffuse,*/ m_d_diffuse_gamma_images, m_d_diffuse_sigma_images,
            db_flags.gamma_miller_units, /*db_flags.wavelength_img,*/
            db_cryst.laue_group_num, db_cryst.stencil_size, /*db_flags.Fhkl_gradient_mode,*/
            /*db_flags.Fhkl_errors_mode,*/ /*db_flags.using_trusted_mask,*/ /*db_beam.Fhkl_channels.empty(),*/
            /*db_flags.Fhkl_have_scale_factors,*/ db_cryst.Num_ASU,
            m_data_residual, m_data_variance,
            m_data_freq, m_data_trusted,
            m_FhklLinear_ASUid,
            m_Fhkl_channels,
            m_Fhkl_scale, m_Fhkl_scale_deriv);
    } else {
        kokkos_sum_over_steps(
            Npix_to_model, m_panels_fasts_slows, m_floatimage, m_wavelenimage, m_d_Umat_images,
            m_d2_Umat_images, m_d_Bmat_images, m_d2_Bmat_images, m_d_Ncells_images, m_d2_Ncells_images,
            m_d_fcell_images, m_d2_fcell_images, m_d_eta_images, m_d2_eta_images, m_d_lambda_images,
            m_d2_lambda_images, m_d_panel_rot_images, m_d2_panel_rot_images, m_d_panel_orig_images,
            m_d2_panel_orig_images, m_d_fp_fdp_images, m_manager_dI, m_manager_dI2, db_steps.Nsteps,
            db_flags.printout_fpixel, db_flags.printout_spixel, db_flags.printout, db_cryst.default_F,
            local_det.oversample, db_flags.oversample_omega, local_det.subpixel_size, local_det.pixel_size,
            local_det.detector_thickstep, local_det.detector_thick, m_close_distances,
            local_det.detector_attnlen, local_det.detector_thicksteps, local_beam.number_of_sources,
            db_cryst.phisteps, db_cryst.UMATS.size(), db_flags.use_lambda_coefficients,
            local_beam.lambda0, local_beam.lambda1, to_mat3(db_cryst.eig_U), to_mat3(db_cryst.eig_O),
            to_mat3(db_cryst.eig_B), to_mat3(db_cryst.RXYZ), m_dF_vecs, m_dS_vecs, m_UMATS_RXYZ, m_UMATS_RXYZ_prime,
            m_UMATS_RXYZ_dbl_prime, m_RotMats, m_dRotMats, m_d2RotMats, m_UMATS, m_dB_Mats, m_dB2_Mats,
            m_AMATS, m_source_X, m_source_Y, m_source_Z, m_source_lambda, m_source_I,
            local_beam.kahn_factor, db_cryst.Na, db_cryst.Nb, db_cryst.Nc, db_cryst.Nd,
            db_cryst.Ne, db_cryst.Nf, db_cryst.phi0, db_cryst.phistep,
            to_vec3(db_cryst.spindle_vec), local_beam.polarization_axis, db_cryst.h_range,
            db_cryst.k_range, db_cryst.l_range, db_cryst.h_max, db_cryst.h_min,
            db_cryst.k_max, db_cryst.k_min, db_cryst.l_max, db_cryst.l_min,
            db_cryst.dmin, db_cryst.fudge, db_flags.complex_miller, db_flags.verbose,
            db_flags.only_save_omega_kahn, db_flags.isotropic_ncells, db_flags.compute_curvatures,
            m_Fhkl, m_Fhkl2, m_refine_flag,
            m_fdet_vectors, m_sdet_vectors, m_odet_vectors,
            m_pix0_vectors, db_flags.nopolar, db_flags.point_pixel, local_beam.fluence,
            db_cryst.r_e_sqr, db_cryst.spot_scale, Npanels, aniso_eta, db_flags.no_Nabc_scale,
            m_fpfdp, m_fpfdp_derivs, m_atom_data, num_atoms, m_nominal_hkl,
            use_nominal_hkl, to_mat3(db_cryst.anisoU), to_mat3(db_cryst.anisoG), to_mat3(db_cryst.rotate_principal_axes),
            db_flags.use_diffuse, m_d_diffuse_gamma_images, m_d_diffuse_sigma_images,
            db_flags.gamma_miller_units, db_flags.wavelength_img,
            db_cryst.laue_group_num, db_cryst.stencil_size, db_flags.Fhkl_gradient_mode,
            db_flags.Fhkl_errors_mode, db_flags.using_trusted_mask, db_beam.Fhkl_channels.empty(),
            db_flags.Fhkl_have_scale_factors, db_cryst.Num_ASU,
            m_data_residual, m_data_variance,
            m_data_freq, m_data_trusted,
            m_FhklLinear_ASUid,
            m_Fhkl_channels,
            m_Fhkl_scale, m_Fhkl_scale_deriv
            );
    }

    ::Kokkos::fence("after kernel call");

    if (db_flags.verbose > 1)
        printf("KERNEL_COMPLETE gpu_sum_over_steps\n");
    easy_time(TIMERS.cuda_kernel, t1, TIMERS.recording);

    gettimeofday(&t1, 0);
    //  COPY BACK FROM DEVICE
    Kokkos::Tools::pushRegion("COPY BACK FROM DEVICE");
    if (db_flags.host_transfer) {
        kokkostbx::transfer_kokkos2vector(floatimage, m_floatimage);

        if (db_flags.wavelength_img) {
            kokkostbx::transfer_kokkos2vector(d_image.wavelength, m_wavelenimage);
        }
        if (db_flags.refine_fcell) {
            kokkostbx::transfer_kokkos2vector(d_image.fcell, m_d_fcell_images);
            kokkostbx::transfer_kokkos2vector(d2_image.fcell, m_d2_fcell_images);
        }
        if (db_flags.Fhkl_gradient_mode){
            if (db_flags.Fhkl_errors_mode){
                kokkostbx::transfer_kokkos2vector(d_image.Fhkl_hessian, m_Fhkl_scale_deriv);
            }
            else{
                kokkostbx::transfer_kokkos2vector(d_image.Fhkl_scale_deriv, m_Fhkl_scale_deriv);
            }
        }
        if (std::count(db_flags.refine_Umat.begin(), db_flags.refine_Umat.end(), true) > 0) {
            kokkostbx::transfer_kokkos2vector(d_image.Umat, m_d_Umat_images);
            kokkostbx::transfer_kokkos2vector(d2_image.Umat, m_d2_Umat_images);
        }
        if (std::count(db_flags.refine_panel_rot.begin(), db_flags.refine_panel_rot.end(), true) > 0) {
            kokkostbx::transfer_kokkos2vector(d_image.panel_rot, m_d_panel_rot_images);
        }
        if (std::count(db_flags.refine_panel_origin.begin(), db_flags.refine_panel_origin.end(), true) >
            0) {
            kokkostbx::transfer_kokkos2vector(d_image.panel_orig, m_d_panel_orig_images);
        }
        if (db_flags.refine_eta) {
            kokkostbx::transfer_kokkos2vector(d_image.eta, m_d_eta_images);
            kokkostbx::transfer_kokkos2vector(d2_image.eta, m_d2_eta_images);
        }
        if (std::count(db_flags.refine_Ncells.begin(), db_flags.refine_Ncells.end(), true) > 0 ||
            db_flags.refine_Ncells_def) {
            kokkostbx::transfer_kokkos2vector(d_image.Ncells, m_d_Ncells_images);
            kokkostbx::transfer_kokkos2vector(d2_image.Ncells, m_d2_Ncells_images);
        }
        if (db_flags.refine_diffuse) {
            kokkostbx::transfer_kokkos2vector(d_image.diffuse_gamma, m_d_diffuse_gamma_images);
            kokkostbx::transfer_kokkos2vector(d_image.diffuse_sigma, m_d_diffuse_sigma_images);
        }
        if (std::count(db_flags.refine_Bmat.begin(), db_flags.refine_Bmat.end(), true) > 0) {
            kokkostbx::transfer_kokkos2vector(d_image.Bmat, m_d_Bmat_images);
            kokkostbx::transfer_kokkos2vector(d2_image.Bmat, m_d2_Bmat_images);
        }
        if (std::count(db_flags.refine_lambda.begin(), db_flags.refine_lambda.end(), true) > 0) {
            kokkostbx::transfer_kokkos2vector(d_image.lambda, m_d_lambda_images);
        }
        if (db_flags.refine_fp_fdp) {
            kokkostbx::transfer_kokkos2vector(d_image.fp_fdp, m_d_fp_fdp_images);
        }
    }

    Kokkos::Tools::popRegion();
    easy_time(TIMERS.cuda_copy_from_dev, t1, TIMERS.recording);
    ::Kokkos::fence("After copy to host");

    Kokkos::Tools::popRegion();
}

DLManagedTensor* diffBraggKOKKOS::get_floatimage() {
    return kokkostbx::view_to_dlpack(m_floatimage);
}

DLManagedTensor* diffBraggKOKKOS::get_wavelenimage() {
    return kokkostbx::view_to_dlpack(m_wavelenimage);
}

DLManagedTensor* diffBraggKOKKOS::get_d_diffuse_gamma_images() {
    return kokkostbx::view_to_dlpack(m_d_diffuse_gamma_images);
}

DLManagedTensor* diffBraggKOKKOS::get_d_diffuse_sigma_images() {
    return kokkostbx::view_to_dlpack(m_d_diffuse_sigma_images);
}

DLManagedTensor* diffBraggKOKKOS::get_d_Umat_images() {
    return kokkostbx::view_to_dlpack(m_d_Umat_images);
}

DLManagedTensor* diffBraggKOKKOS::get_d2_Umat_images() {
    return kokkostbx::view_to_dlpack(m_d2_Umat_images);
}

DLManagedTensor* diffBraggKOKKOS::get_d_Bmat_images() {
    return kokkostbx::view_to_dlpack(m_d_Bmat_images);
}

DLManagedTensor* diffBraggKOKKOS::get_d2_Bmat_images() {
    return kokkostbx::view_to_dlpack(m_d2_Bmat_images);
}

DLManagedTensor* diffBraggKOKKOS::get_d_Ncells_images() {
    return kokkostbx::view_to_dlpack(m_d_Ncells_images);
}

DLManagedTensor* diffBraggKOKKOS::get_d2_Ncells_images() {
    return kokkostbx::view_to_dlpack(m_d2_Ncells_images);
}

DLManagedTensor* diffBraggKOKKOS::get_d_fcell_images() {
    return kokkostbx::view_to_dlpack(m_d_fcell_images);
}

DLManagedTensor* diffBraggKOKKOS::get_d2_fcell_images() {
    return kokkostbx::view_to_dlpack(m_d2_fcell_images);
}

DLManagedTensor* diffBraggKOKKOS::get_d_eta_images() {
    return kokkostbx::view_to_dlpack(m_d_eta_images);
}

DLManagedTensor* diffBraggKOKKOS::get_d2_eta_images() {
    return kokkostbx::view_to_dlpack(m_d2_eta_images);
}

DLManagedTensor* diffBraggKOKKOS::get_d_lambda_images() {
    return kokkostbx::view_to_dlpack(m_d_lambda_images);
}

DLManagedTensor* diffBraggKOKKOS::get_d2_lambda_images() {
    return kokkostbx::view_to_dlpack(m_d2_lambda_images);
}

DLManagedTensor* diffBraggKOKKOS::get_d_panel_rot_images() {
    return kokkostbx::view_to_dlpack(m_d_panel_rot_images);
}

DLManagedTensor* diffBraggKOKKOS::get_d2_panel_rot_images() {
    return kokkostbx::view_to_dlpack(m_d2_panel_rot_images);
}

DLManagedTensor* diffBraggKOKKOS::get_d_panel_orig_images() {
    return kokkostbx::view_to_dlpack(m_d_panel_orig_images);
}

DLManagedTensor* diffBraggKOKKOS::get_d2_panel_orig_images() {
    return kokkostbx::view_to_dlpack(m_d2_panel_orig_images);
}

DLManagedTensor* diffBraggKOKKOS::get_d_fp_fdp_images() {
    return kokkostbx::view_to_dlpack(m_d_fp_fdp_images);
}

DLManagedTensor* diffBraggKOKKOS::get_Fhkl_scale_deriv() {
    return kokkostbx::view_to_dlpack(m_Fhkl_scale_deriv);
}
