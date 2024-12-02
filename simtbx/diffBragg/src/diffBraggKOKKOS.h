#ifndef SIMTBX_DIFFBRAGG_KOKKOS
#define SIMTBX_DIFFBRAGG_KOKKOS

// #include <cmath>
#include <iostream>
#include <vector>

#include "kokkostbx/kokkos_types.h"
#include "kokkostbx/kokkos_utils.h"
#include "kokkostbx/kokkos_dlpack.h"
#include "simtbx/diffBragg/src/util.h"
#include "simtbx/diffBragg/src/util_kokkos.h"
#include "simtbx/diffBragg/src/diffBragg_refine_flag.h"

using vector_vec3_t = view_1d_t<KOKKOS_VEC3>;
using vector_mat3_t = view_1d_t<KOKKOS_MAT3>;
using vector_manager_t = view_1d_t<kokkos_manager>;

#define INTEGER_VIEW(varname) vector_int_t varname = vector_int_t(#varname, 0)
#define CUDAREAL_VIEW(varname) vector_cudareal_t varname = vector_cudareal_t(#varname, 0)
#define MATRIX3_VIEW(varname) vector_mat3_t varname = vector_mat3_t(#varname, 0)

class diffBraggKOKKOS {
   private:
    bool m_device_is_allocated = false;
    int m_npix_allocated = 0;
    int m_previous_nsource = 0;

    vector_uint_t m_panels_fasts_slows = vector_uint_t("m_panels_fasts_slows", 0);

    CUDAREAL_VIEW(m_floatimage);
    CUDAREAL_VIEW(m_wavelenimage);
    CUDAREAL_VIEW(m_d_diffuse_sigma_images);
    CUDAREAL_VIEW(m_d_diffuse_gamma_images);
    CUDAREAL_VIEW(m_d_Umat_images);
    CUDAREAL_VIEW(m_d_Bmat_images);
    CUDAREAL_VIEW(m_d_Ncells_images);
    CUDAREAL_VIEW(m_d_fcell_images);
    CUDAREAL_VIEW(m_d_eta_images);
    CUDAREAL_VIEW(m_d2_eta_images);
    CUDAREAL_VIEW(m_d_lambda_images);
    CUDAREAL_VIEW(m_d_panel_rot_images);
    CUDAREAL_VIEW(m_d_panel_orig_images);

    CUDAREAL_VIEW(m_d2_Umat_images);
    CUDAREAL_VIEW(m_d2_Bmat_images);
    CUDAREAL_VIEW(m_d2_Ncells_images);
    CUDAREAL_VIEW(m_d2_fcell_images);
    CUDAREAL_VIEW(m_d2_lambda_images);
    CUDAREAL_VIEW(m_d2_panel_rot_images);
    CUDAREAL_VIEW(m_d2_panel_orig_images);

    CUDAREAL_VIEW(m_d_sausage_XYZ_scale_images);
    CUDAREAL_VIEW(m_d_fp_fdp_images);

    INTEGER_VIEW(m_subS_pos);
    INTEGER_VIEW(m_subF_pos);
    INTEGER_VIEW(m_thick_pos);
    INTEGER_VIEW(m_source_pos);
    INTEGER_VIEW(m_mos_pos);
    INTEGER_VIEW(m_phi_pos);
    INTEGER_VIEW(m_sausage_pos);

    CUDAREAL_VIEW(m_Fhkl);
    CUDAREAL_VIEW(m_Fhkl2);

    CUDAREAL_VIEW(m_fdet_vectors);
    CUDAREAL_VIEW(m_sdet_vectors);
    CUDAREAL_VIEW(m_odet_vectors);
    CUDAREAL_VIEW(m_pix0_vectors);
    CUDAREAL_VIEW(m_close_distances);

    INTEGER_VIEW(m_nominal_hkl);
    CUDAREAL_VIEW(m_fpfdp);
    CUDAREAL_VIEW(m_fpfdp_derivs);
    CUDAREAL_VIEW(m_atom_data);

    CUDAREAL_VIEW(m_source_X);
    CUDAREAL_VIEW(m_source_Y);
    CUDAREAL_VIEW(m_source_Z);
    CUDAREAL_VIEW(m_source_I);
    CUDAREAL_VIEW(m_source_lambda);
    int m_sources;
    bool m_sources_are_allocated = false;
    bool m_sources_recopy = false;

    MATRIX3_VIEW(m_UMATS);
    MATRIX3_VIEW(m_dB_Mats);
    MATRIX3_VIEW(m_dB2_Mats);
    MATRIX3_VIEW(m_UMATS_RXYZ);
    MATRIX3_VIEW(m_UMATS_RXYZ_prime);
    MATRIX3_VIEW(m_UMATS_RXYZ_dbl_prime);
    MATRIX3_VIEW(m_RotMats);
    MATRIX3_VIEW(m_dRotMats);
    MATRIX3_VIEW(m_d2RotMats);

    MATRIX3_VIEW(m_AMATS);

    vector_vec3_t m_dF_vecs = vector_vec3_t("m_dF_vecs", 0);
    vector_vec3_t m_dS_vecs = vector_vec3_t("m_dS_vecs", 0);

    MATRIX3_VIEW(m_sausages_RXYZ);
    MATRIX3_VIEW(m_d_sausages_RXYZ);
    MATRIX3_VIEW(m_sausages_U);
    CUDAREAL_VIEW(m_sausages_scale);

    uint32_t m_refine_flag = 0;
    vector_bool_t m_refine_Bmat = vector_bool_t("m_refine_Bmat", 6);
    vector_bool_t m_refine_Umat = vector_bool_t("m_refine_Umat", 3);
    vector_bool_t m_refine_Ncells = vector_bool_t("m_refine_Ncells", 3);
    vector_bool_t m_refine_panel_origin = vector_bool_t("m_refine_panel_origin", 3);
    vector_bool_t m_refine_panel_rot = vector_bool_t("m_refine_panel_rot", 3);
    vector_bool_t m_refine_lambda = vector_bool_t("m_refine_lambda", 2);

    vector_manager_t m_manager_dI = vector_manager_t("m_manager_dI", 0);
    vector_manager_t m_manager_dI2 = vector_manager_t("m_manager_dI2", 0);

    bool m_Fhkl_gradient_mode;
    bool m_using_trusted_mask;
    bool m_Fhkl_channels_empty;
    bool m_Fhkl_have_scale_factors;
    // these are copied once at first iteration
    bool m_Fhkl_grad_arrays_allocated=false;
    CUDAREAL_VIEW(m_data_residual); // length is number of modeled pixels
    CUDAREAL_VIEW(m_data_variance); // length is number of modeled pixels
    INTEGER_VIEW(m_data_freq); // length is number of modeled pixels
    vector_bool_t m_data_trusted = vector_bool_t("m_data_trusted", 0); // length is number of modeled pixels
    INTEGER_VIEW(m_FhklLinear_ASUid); // length is number of ASU in FhklLinear
    INTEGER_VIEW(m_Fhkl_channels);
    // Fhkl_scale is dynamically copied each iteration
    // Fhkl_scale_deriv is set to 0 each iteration
    CUDAREAL_VIEW(m_Fhkl_scale);  // length is (number of ASUin FhklLinear) *times* (number of Fhkl channels)
    CUDAREAL_VIEW(m_Fhkl_scale_deriv); // length is (number of ASUin FhklLinear) *times* (number of Fhkl channels)


   public:
    void diffBragg_sum_over_steps_kokkos(
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
        timer_variables& TIMERS);

    DLManagedTensor* get_floatimage();
    DLManagedTensor* get_wavelenimage();
    DLManagedTensor* get_d_diffuse_gamma_images();
    DLManagedTensor* get_d_diffuse_sigma_images();
    DLManagedTensor* get_d_Umat_images();
    DLManagedTensor* get_d2_Umat_images();
    DLManagedTensor* get_d_Bmat_images();
    DLManagedTensor* get_d2_Bmat_images();
    DLManagedTensor* get_d_Ncells_images();
    DLManagedTensor* get_d2_Ncells_images();
    DLManagedTensor* get_d_fcell_images();
    DLManagedTensor* get_d2_fcell_images();
    DLManagedTensor* get_d_eta_images();
    DLManagedTensor* get_d2_eta_images();
    DLManagedTensor* get_d_lambda_images();
    DLManagedTensor* get_d2_lambda_images();
    DLManagedTensor* get_d_panel_rot_images();
    DLManagedTensor* get_d2_panel_rot_images();
    DLManagedTensor* get_d_panel_orig_images();
    DLManagedTensor* get_d2_panel_orig_images();
    DLManagedTensor* get_d_fp_fdp_images();
    DLManagedTensor* get_Fhkl_scale_deriv();
};

#endif
