#include <simtbx/diffBragg/src/diffBragg.h>
#include <simtbx/diffBragg/src/diffuse_util.h>
#include <assert.h>
#include <stdbool.h>
#include<unordered_map>
#include<unordered_set>
#include<string>

#if defined(_OPENMP)
    #include<omp.h>
#endif

namespace simtbx { namespace nanoBragg { // BEGIN namespace simtbx::nanoBragg


double diffBragg_cpu_kernel_polarization (Eigen::Vector3d incident, Eigen::Vector3d diffracted,
                                        Eigen::Vector3d polarization_axis, double kahn_factor){
    // component of diffracted unit vector along incident beam unit vector
    double cos2theta = incident.dot(diffracted);
    double cos2theta_sqr = cos2theta*cos2theta;
    double sin2theta_sqr = 1-cos2theta_sqr;

    double psi=0;
    if(kahn_factor != 0.0){
        // cross product to get "vertical" axis that is orthogonal to the cannonical "polarization"
        Eigen::Vector3d B_in = polarization_axis.cross(incident);
        // cross product with incident beam to get E-vector direction
        Eigen::Vector3d E_in = incident.cross(B_in);
        // get components of diffracted ray projected onto the E-B plane
        double kEi = diffracted.dot(E_in);
        double kBi = diffracted.dot(B_in);
        // compute the angle of the diffracted ray projected onto the incident E-B plane
        psi = -atan2(kBi,kEi);
    }
    // correction for polarized incident beam
    double polar = 0.5*(1.0 + cos2theta_sqr - kahn_factor*cos(2*psi)*sin2theta_sqr);
    return polar;
}

int get_bin(std::vector<double> dspace_bins,  double dspace_sample){
//  use binary search to find bin corresponding to dspace_sample as dspace_bins are unevenly sized
    int start = 0;
    int end = dspace_bins.size() - 1;
    while (start <= end) {
        int mid = (start + end) / 2;
        if (dspace_bins[mid] == dspace_sample)
            return mid;
        else if (dspace_bins[mid] < dspace_sample)
            start = mid + 1;
        else
            end = mid - 1;
    }
    return end + 1;
}

std::vector<double> I_cell_ave(crystal& db_cryst, bool use_Fhkl_scale, int i_channel,
            std::vector<double>& Fhkl_scale){
    std::vector<double> ave;
    std::vector<double> count;
    for (int i=0; i < db_cryst.dspace_bins.size(); i++){
        ave.push_back(0);
        count.push_back(0);
    }
    #pragma omp parallel for
    for (int i=0; i < db_cryst.ASU_dspace.size(); i++){
        double d = db_cryst.ASU_dspace[i];
        double F_cell = db_cryst.ASU_Fcell[i];
        if (F_cell==0){
            continue;
        }
        int bin = get_bin(db_cryst.dspace_bins, d);
        if (bin == 0 || bin >= db_cryst.dspace_bins.size() )
            continue;
        double scale = 1;
        if (use_Fhkl_scale)
            scale = Fhkl_scale[i + db_cryst.Num_ASU*i_channel];

        double summand=scale*F_cell*F_cell;
        if (db_cryst.use_geometric_mean)
            summand = log(summand);
        #pragma omp atomic
        ave[bin] += summand;
        #pragma omp atomic
        count[bin] ++;
    }
    for (int i=0; i <ave.size(); i++){
        if (count[i]>0){
            if (db_cryst.use_geometric_mean)
                ave[i] = exp(ave[i]/count[i]);
            else
                ave[i] = ave[i]/count[i];
        }
    }
    for (int i=0; i < count.size(); i++)
        ave.push_back(count[i]);
    return ave;
}

void Ih_grad_terms(crystal& db_cryst, int i_chan, std::vector<double>& Fhkl_scale, std::vector<double>& out){
    std::vector<double> ave_and_count = I_cell_ave(db_cryst, true, i_chan, Fhkl_scale);
    std::vector<double> ave, count;
    for (int i=0; i < ave_and_count.size()/2; i++){
        ave.push_back(ave_and_count[i]);
        count.push_back(ave_and_count[i+ave_and_count.size()/2]);
    }

    ave_and_count = I_cell_ave(db_cryst, false, i_chan, Fhkl_scale);
    std::vector<double> ave_init;
    for (int i=0; i < ave_and_count.size()/2; i++){
        ave_init.push_back(ave_and_count[i]);
    }


    #pragma omp parallel for
    for (int i=0; i < db_cryst.Num_ASU; i++){
        double dsp = db_cryst.ASU_dspace[i];
        int bin = get_bin(db_cryst.dspace_bins, dsp);
        if (bin==0 || bin>=db_cryst.dspace_bins.size())
            continue;
        if(count[bin]==0) continue;

        double F_cell = db_cryst.ASU_Fcell[i];
        int idx = i_chan*db_cryst.Num_ASU + i;
        double scale_hkl = Fhkl_scale[idx];
        double I_cell_init = F_cell*F_cell;
        double I_cell = scale_hkl * I_cell_init;
        double U = ave[bin] - ave_init[bin];
        double grad_term;
        if (db_cryst.use_geometric_mean){
            double grad_term_right = 1/count[bin] * ave[bin] / scale_hkl;
            grad_term = U/db_cryst.Fhkl_beta *  grad_term_right;
        }
        else {
            grad_term = U/db_cryst.Fhkl_beta * I_cell_init/count[bin];
        }
        #pragma omp atomic
        out[i] += grad_term;
    }

    double ftarget=0;
    for (int i=0; i < ave.size(); i++){
        double U = (ave[i] - ave_init[i]);
        ftarget += 0.5*( log(2*M_PI*db_cryst.Fhkl_beta) + U*U/db_cryst.Fhkl_beta);
    }

    out[db_cryst.Num_ASU] = ftarget; // this is an addition to the target function for this particular restraint, added on for convenience
}


//void Ih_grad_terms(crystal& db_cryst, int i_chan, std::vector<double>& Fhkl_scale, std::vector<double>& out){
//    std::vector<double> ave_and_count = I_cell_ave(db_cryst, true, i_chan, Fhkl_scale);
//    std::vector<double> ave, count;
//    for (int i=0; i < ave_and_count.size()/2; i++){
//        ave.push_back(ave_and_count[i]);
//        count.push_back(ave_and_count[i+ave_and_count.size()/2]);
//    }
//
//    //ave_and_count = I_cell_ave(db_cryst, false, i_chan, Fhkl_scale);
//    //std::vector<double> ave_init;
//    //for (int i=0; i < ave_and_count.size()/2; i++){
//    //    ave_init.push_back(ave_and_count[i]);
//    //}
//
//    #pragma omp parallel for
//    for (int i=0; i < db_cryst.Num_ASU; i++){
//        double dsp = db_cryst.ASU_dspace[i];
//        int bin = get_bin(db_cryst.dspace_bins, dsp);
//        if (bin==0 || bin>=db_cryst.dspace_bins.size())
//            continue;
//        if(count[bin]==0) continue;
//
//        double F_cell = db_cryst.ASU_Fcell[i];
//        int idx = i_chan*db_cryst.Num_ASU + i;
//        double scale_hkl = Fhkl_scale[idx];
//        double I_cell_init = F_cell*F_cell;
//        double I_cell = scale_hkl * I_cell_init;
//
//        double U;
//        if (bin==1){
//            U = ave[bin] - ave[bin+1];
//        }
//        else if (bin==db_cryst.dspace_bins.size()-1) {
//            U = ave[bin] - ave[bin-1];
//        }
//        else {
//            U = 2*ave[bin] - ave[bin+1] - ave[bin-1];
//        }
//        double grad_term = U/db_cryst.Fhkl_beta * I_cell_init/count[bin];
//
//        //if (db_cryst.use_geometric_mean){
//        //    double grad_term_right = 1/count[bin] * ave[bin] / scale_hkl;
//        //    grad_term = U/db_cryst.Fhkl_beta *  grad_term_right;
//        //}
//        //else {
//        //    grad_term = U/db_cryst.Fhkl_beta * I_cell_init/count[bin];
//        //}
//
//        #pragma omp atomic
//        out[i] += grad_term;
//    }
//
//    double ftarget=0;
//    for (int i=1; i < ave.size()-1; i++){
//        double U = (ave[i] - ave[i+1]);
//        ftarget += 0.5*( log(2*M_PI*db_cryst.Fhkl_beta) + U*U/db_cryst.Fhkl_beta);
//    }
//
//    out[db_cryst.Num_ASU] = ftarget;
//}

void Finit_grad_terms(crystal& db_cryst, int i_chan, std::vector<double>& Fhkl_scale, std::vector<double>& out){

    std::vector<double> ave_and_count = I_cell_ave(db_cryst, true, i_chan, Fhkl_scale);
    std::vector<double> ave, count;
    for (int i=0; i < ave_and_count.size()/2; i++){
        ave.push_back(ave_and_count[i]);
        count.push_back(ave_and_count[i+ave_and_count.size()/2]);
    }

    double ftarget=0;
    #pragma omp parallel for reduction(+:ftarget)
    for (int i=0; i < db_cryst.Num_ASU; i++){
        double dsp = db_cryst.ASU_dspace[i];
        int bin = get_bin(db_cryst.dspace_bins, dsp);
        if (bin==0 || bin>=db_cryst.dspace_bins.size())
            continue;
        double N = count[bin];
        if(N==0) continue;
        double F_cell = db_cryst.ASU_Fcell[i];
        double I_cell_init = F_cell*F_cell;

        int idx = i_chan*db_cryst.Num_ASU + i;
        double scale_hkl = Fhkl_scale[idx];
        double I_cell_current = scale_hkl * I_cell_init;

        double I_ave = ave[bin];
        double U = (I_cell_current - I_cell_init) / I_ave;

        ftarget += U*U/2/db_cryst.Finit_beta;
        double grad_term = U/db_cryst.Finit_beta * I_cell_init/I_ave *(1-U/N);
        #pragma omp atomic
        out[i] += grad_term;
    }
    out[db_cryst.Num_ASU] = ftarget;
}

void Friedel_grad_terms(crystal& db_cryst, int i_chan, std::vector<double>& Fhkl_scale, std::vector<double>& out){

    SCITBX_ASSERT(db_cryst.neg_inds.size() == db_cryst.pos_inds.size()) ;


    double log_beta = log(2*M_PI*db_cryst.Friedel_beta);
    double ftarget = 0;
    #pragma omp parallel for reduction(+:ftarget)
    for (int i_pair=0; i_pair < db_cryst.neg_inds.size(); i_pair++){
        int i_minus = db_cryst.neg_inds[i_pair];
        int i_plus = db_cryst.pos_inds[i_pair] ;

        int channel_offset = db_cryst.Num_ASU*i_chan;
        double s_minus = Fhkl_scale[channel_offset + i_minus];
        double s_plus = Fhkl_scale[channel_offset + i_plus];

        double F_minus =  db_cryst.ASU_Fcell[i_minus];
        double I_minus = F_minus*F_minus;
        double F_plus = db_cryst.ASU_Fcell[i_plus];
        double I_plus = F_plus*F_plus;

        double sI_plus = s_plus*I_plus;
        double sI_minus = s_minus*I_minus;
        double S = sI_plus + sI_minus; // 'S' is for summed intensity
        if (S==0) continue;
        double D = sI_plus - sI_minus; // 'D' is for difference intensity

        double friedel_diff = 0.5*D/S; // normalized difference intensity - we want to restrain this ...

        ftarget += friedel_diff*friedel_diff / db_cryst.Friedel_beta + log_beta;

        double D_by_S = D/S;
        double friedel_diff_by_beta = .5* friedel_diff / db_cryst.Friedel_beta;
        double grad_incr_s_plus =  friedel_diff_by_beta * I_plus/S *(1-D_by_S);
        double grad_incr_s_minus =  -friedel_diff_by_beta * I_minus/S * (1+D_by_S);
        #pragma omp atomic
        out[i_plus] += grad_incr_s_plus;
        #pragma omp atomic
        out[i_minus] += grad_incr_s_minus;
    }

    out[db_cryst.Num_ASU] = ftarget*0.5;
}


void diffBragg_sum_over_steps(
        int Npix_to_model, std::vector<unsigned int>& panels_fasts_slows,
        image_type& floatimage,
        images& d_image,
        images& d2_image,
        step_arrays& db_steps,
        detector& db_det,
        beam& db_beam,
        crystal& db_cryst,
        flags& db_flags){

    MAT3 anisoG_local;
    MAT3 anisoU_local;
    MAT3 laue_mats[24];
    MAT3 dG_dgam[3];
    int laue_group_num = db_cryst.laue_group_num;
    int num_laue_mats = 1;
    int dhh=0, dkk=0, dll=0;
    if (db_flags.use_diffuse){
      anisoG_local = db_cryst.anisoG;
      anisoU_local = db_cryst.anisoU;

      if (laue_group_num < 1 || laue_group_num >14 ){
        throw std::string("Laue group number not in range 1-14");
      }

      num_laue_mats = gen_laue_mats(laue_group_num, laue_mats, db_cryst.rotate_principal_axes);
      for (int i_gam=0; i_gam<3; i_gam++){
        dG_dgam[i_gam] << 0,0,0,0,0,0,0,0,0;
        dG_dgam[i_gam](i_gam, i_gam) = 1;
      }
      dhh = dkk = dll = db_cryst.stencil_size; // Limits of stencil for diffuse calc
    }
    VEC3 Hmin(db_cryst.h_min, db_cryst.k_min, db_cryst.l_min);
    VEC3 Hmax(db_cryst.h_max, db_cryst.k_max, db_cryst.l_max);
    VEC3 dHH(dhh,dkk,dll);
    VEC3 Hrange(db_cryst.h_range, db_cryst.k_range, db_cryst.l_range);
    if (db_flags.track_Fhkl_indices)
        db_cryst.Fhkl_grad_idx_tracker.clear();

#if defined(_OPENMP)
    int dyn_state = omp_get_dynamic();
    int nthread_state = omp_get_max_threads();
    if (db_flags.track_Fhkl_indices){
        omp_set_dynamic(0);
        omp_set_num_threads(1);
    }
#endif
    #pragma omp parallel for
    for (int i_pix=0; i_pix < Npix_to_model; i_pix++){


#if defined(_OPENMP)
        if (db_flags.track_Fhkl_indices)
            SCITBX_ASSERT(omp_get_num_threads()==1);
#endif

        if (db_flags.using_trusted_mask){
            if (! d_image.trusted[i_pix])
                continue;
        }
        int pid = panels_fasts_slows[i_pix*3];
        int fpixel = panels_fasts_slows[i_pix*3+1];
        int spixel = panels_fasts_slows[i_pix*3+2];
        double Fhkl_deriv_coef=0;
        double Fhkl_hessian_coef=0;
        if (db_flags.Fhkl_gradient_mode){
            double u = d_image.residual[i_pix];
            double one_by_v = 1/d_image.variance[i_pix];
            double Gterm = 1 - 2*u - u*u*one_by_v;
            Fhkl_deriv_coef = 0.5 * Gterm*one_by_v / d_image.freq[i_pix];
            if (db_flags.Fhkl_errors_mode){
                Fhkl_hessian_coef = -0.5*one_by_v*(one_by_v*Gterm - 2  - 2*u*one_by_v -u*u*one_by_v*one_by_v)/d_image.freq[i_pix];
            }
        }
        double close_distance = db_det.close_distances[pid];
        //std::unordered_map<int, int> Fhkl_tracker;
        std::unordered_map<std::string, int> Fhkl_tracker;
        bool use_nominal_hkl = false;
        if (!db_cryst.nominal_hkl.empty())
            use_nominal_hkl = true;
        // reset photon count for this pixel
        double I=0;
        double Ilambda = 0;
        double Imiller_h =0;
        double Imiller_k =0;
        double Imiller_l =0;
        double II_max = -1;
        double max_stats[11] = {0,0,0,0,0,
                               0,0,0,0,0,0};
        double max_stats2[11] = {0,0,0,0,0,
                               0,0,0,0,0,0};
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
        double fcell_manager_dI=0;
        double fcell_manager_dI2=0;
        double eta_manager_dI[3] = {0,0,0};
        double eta_manager_dI2[3] = {0,0,0};
        double lambda_manager_dI[2] = {0,0};
        double lambda_manager_dI2[2] = {0,0};
        double fp_fdp_manager_dI[2] = {0,0};
        double dI_latt_diffuse[6] = {0,0,0,0,0,0};

        // compute unit cell volume
        Eigen::Vector3d ap_vec(1e10*db_cryst.eig_B(0,0), 1e10*db_cryst.eig_B(1,0), 1e10*db_cryst.eig_B(2,0));
        Eigen::Vector3d bp_vec(1e10*db_cryst.eig_B(0,1), 1e10*db_cryst.eig_B(1,1), 1e10*db_cryst.eig_B(2,1));
        Eigen::Vector3d cp_vec(1e10*db_cryst.eig_B(0,2), 1e10*db_cryst.eig_B(1,2), 1e10*db_cryst.eig_B(2,2));
        double cell_vol = ap_vec.dot(bp_vec.cross(cp_vec)); // cubic Angstrom
        double xtal_size = pow(db_cryst.Na*db_cryst.Nb*db_cryst.Nc*cell_vol, (double)1/double(3)); // Angstrom

        for (int i_step=0; i_step < db_steps.Nsteps; i_step++){

            int subS = db_steps.subS_pos[i_step];
            int subF = db_steps.subF_pos[i_step];
            int thick_tic = db_steps.thick_pos[i_step];
            int source = db_steps.source_pos[i_step];
            int phi_tic = db_steps.phi_pos[i_step];
            int mos_tic = db_steps.mos_pos[i_step];

            /* absolute mm position on detector (relative to its origin) */
            double Fdet = db_det.subpixel_size*(fpixel*db_det.oversample + subF ) + db_det.subpixel_size/2.0;
            double Sdet = db_det.subpixel_size*(spixel*db_det.oversample + subS ) + db_det.subpixel_size/2.0;

            /* assume "distance" is to the front of the detector sensor layer */
            double Odet = thick_tic*db_det.detector_thickstep;
            int pid_x = pid*3;
            int pid_y = pid*3+1;
            int pid_z = pid*3+2;
            Eigen::Vector3d o_vec(db_det.odet_vectors[pid_x], db_det.odet_vectors[pid_y], db_det.odet_vectors[pid_z]);

            double pixposX = Fdet*db_det.fdet_vectors[pid_x]+Sdet*db_det.sdet_vectors[pid_x]+Odet*db_det.odet_vectors[pid_x]+db_det.pix0_vectors[pid_x];
            double pixposY = Fdet*db_det.fdet_vectors[pid_y]+Sdet*db_det.sdet_vectors[pid_y]+Odet*db_det.odet_vectors[pid_y]+db_det.pix0_vectors[pid_y];
            double pixposZ = Fdet*db_det.fdet_vectors[pid_z]+Sdet*db_det.sdet_vectors[pid_z]+Odet*db_det.odet_vectors[pid_z]+db_det.pix0_vectors[pid_z];
            Eigen::Vector3d pixel_pos(pixposX, pixposY, pixposZ);

            double airpath = pixel_pos.norm();
            Eigen::Vector3d diffracted = pixel_pos/airpath;

            double omega_pixel = db_det.pixel_size*db_det.pixel_size/airpath/airpath*close_distance/airpath;

            if(db_flags.point_pixel) omega_pixel = 1.0/airpath/airpath;

            double capture_fraction = 1;
            if(db_det.detector_thick > 0.0 && db_det.detector_attnlen > 0.0)
            {
                double parallax = diffracted.dot(o_vec) ;
                capture_fraction = exp(-thick_tic*db_det.detector_thickstep/db_det.detector_attnlen/parallax)
                                  -exp(-(thick_tic+1)*db_det.detector_thickstep/db_det.detector_attnlen/parallax);
            }
            Eigen::Vector3d incident(-db_beam.source_X[source], -db_beam.source_Y[source], -db_beam.source_Z[source]);
            double lambda = db_beam.source_lambda[source];
            double lambda_ang = lambda*1e10;
            if (db_flags.use_lambda_coefficients){
                lambda_ang = db_beam.lambda0 + db_beam.lambda1*lambda_ang;
                lambda = lambda_ang*1e-10;
            }

            /* construct the incident beam unit vector while recovering source distance */
            double source_path = incident.norm();
            incident /= source_path;

            /* construct the scattering vector for this pixel */
            //vec3 scattering = (diffracted - incident) / lambda;
            Eigen::Vector3d scattering = (diffracted - incident) / lambda;

            /* sin(theta)/lambda is half the scattering vector length */
            double stol = 0.5*(scattering.norm()); //magnitude(scattering);

            // polarization
            double polar_for_Fhkl_grad=1;
            if (!db_flags.nopolar && db_flags.Fhkl_gradient_mode)
                polar_for_Fhkl_grad = diffBragg_cpu_kernel_polarization(incident, diffracted,
                                    db_beam.polarization_axis, db_beam.kahn_factor);

            /* rough cut to speed things up when we aren't using whole detector */
            if(db_cryst.dmin > 0.0 && stol > 0.0)
            {
                if(db_cryst.dmin > 0.5/stol)
                {
                    continue;
                }
            }

            Eigen::Matrix3d Bmat_realspace = 1e10*db_cryst.eig_B;
            //if( phi != 0.0 )
            //{
            //    double cosphi = cos(phi);
            //    double sinphi = sin(phi);

            //    ap_vec = ap_vec*cosphi + db_cryst.spindle_vec.cross(ap_vec)*sinphi + db_cryst.spindle_vec*(db_cryst.spindle_vec.dot(ap_vec))*(1-cosphi);
            //    bp_vec = bp_vec*cosphi + db_cryst.spindle_vec.cross(bp_vec)*sinphi + db_cryst.spindle_vec*(db_cryst.spindle_vec.dot(bp_vec))*(1-cosphi);
            //    cp_vec = cp_vec*cosphi + db_cryst.spindle_vec.cross(cp_vec)*sinphi + db_cryst.spindle_vec*(db_cryst.spindle_vec.dot(cp_vec))*(1-cosphi);

            //    Bmat_realspace << ap_vec[0], bp_vec[0], cp_vec[0],
            //                        ap_vec[1], bp_vec[1], cp_vec[1],
            //                        ap_vec[2], bp_vec[2], cp_vec[2];
            //}
            //Bmat_realspace *= 1e10;
            if (db_flags.use_diffuse && db_flags.gamma_miller_units){
              anisoG_local = anisoG_local * Bmat_realspace;
              for (int i_gam=0; i_gam<3; i_gam++){
                dG_dgam[i_gam] = dG_dgam[i_gam] * Bmat_realspace;
              }
            }

            Eigen::Matrix3d U = db_cryst.eig_U;

            Eigen::Matrix3d UBOt=U*Bmat_realspace*(db_cryst.eig_O.transpose());
            double phi = db_cryst.phi0 + db_cryst.phistep*phi_tic;
            if (phi != 0){
                if (i_pix==0){
                    printf("phistep=%f, phi0=%f, phi=%f\n", db_cryst.phistep, db_cryst.phi0, phi);
                }

                double c = cos(phi);
                double omc = 1-c;
                double s = sin(phi);
                Eigen::Matrix3d Rphi;
                double gx = db_cryst.spindle_vec[0];
                double gy = db_cryst.spindle_vec[1];
                double gz = db_cryst.spindle_vec[2];
                Rphi << c + gx*gx*omc,    gx*gy*omc-gz*s,   gx*gz*omc+gy*s,
                      gy*gx*omc + gz*s,   c + gy*gy*omc,   gy*gz*omc - gx*s,
                      gz*gx*omc - gy*s,  gz*gy*omc + gx*s, c + gz*gz*omc;
                UBOt = Rphi*UBOt;
            }
            Eigen::Matrix3d UBO = (db_cryst.UMATS_RXYZ[mos_tic]*UBOt).transpose();
            Eigen::Matrix3d Ainv = UBO.inverse();
            Eigen::Vector3d q_vec(scattering[0], scattering[1], scattering[2]);
            q_vec *= 1e-10;
            Eigen::Vector3d H_vec = UBO*q_vec;

            double h = H_vec[0];
            double k = H_vec[1];
            double l = H_vec[2];

            /* round off to nearest whole index */
            int h0 = static_cast<int>(ceil(h - 0.5));
            int k0 = static_cast<int>(ceil(k - 0.5));
            int l0 = static_cast<int>(ceil(l - 0.5));

            Eigen::Vector3d H0(h0, k0, l0);
            Eigen::Matrix3d NABC;
            NABC << db_cryst.Na, db_cryst.Nd, db_cryst.Nf,
                     db_cryst.Nd, db_cryst.Nb, db_cryst.Ne,
                     db_cryst.Nf, db_cryst.Ne, db_cryst.Nc;

            double C = 2 / 0.63 * db_cryst.fudge;
            Eigen::Vector3d delta_H = H_vec - H0;
            Eigen::Vector3d V = NABC*delta_H;
            double hrad_sqr = V.dot(V);
            double NABC_determ = NABC.determinant();
            double F_latt = 0;
            if (db_cryst.xtal_shape==SQUARE){
                F_latt = 1;
                if (db_cryst.Na > 1)
                        F_latt *= sincg(M_PI * h, db_cryst.Na);
                if (db_cryst.Nb > 1)
                        F_latt *= sincg(M_PI * k, db_cryst.Nb);
                if (db_cryst.Nc > 1)
                        F_latt *= sincg(M_PI * l, db_cryst.Nc);
            }
            else { // gaussian relp model
                double exparg;
                if (db_cryst.xtal_shape==GAUSS_STAR){
                    Eigen::Vector3d delta_Q = UBO.inverse()*delta_H;
                    double rad_star_sqr = delta_Q.dot(delta_Q)*xtal_size*xtal_size;
                    exparg = rad_star_sqr *1.9* db_cryst.fudge ;
                }
                else
                    exparg = hrad_sqr / 0.63 * db_cryst.fudge;

                F_latt = exp(-exparg);
                if (! db_flags.no_Nabc_scale)
                    F_latt *= NABC_determ;
            }

            /* convert amplitudes into intensity (photons per steradian) */
            if (!db_flags.oversample_omega && !db_flags.Fhkl_gradient_mode)
                omega_pixel = 1;
            double count_scale = db_beam.source_I[source]*capture_fraction*omega_pixel;

            double I0 = 0;
            double step_diffuse_param[6] = {0,0,0,0,0,0};
            if (db_flags.use_diffuse){
              calc_diffuse_at_hkl(H_vec,H0,dHH,Hmin,Hmax,Hrange,Ainv,&db_cryst.FhklLinear[0],num_laue_mats,laue_mats,anisoG_local,anisoU_local,dG_dgam,db_flags.refine_diffuse,&I0,step_diffuse_param);
            }

            /* increment to intensity */
            double latt_scale = 1;
            if (db_flags.only_diffuse)
                latt_scale = 0;
            double I_noFcell = (F_latt*F_latt*latt_scale + I0) * count_scale;

            /* structure factor of the unit cell */
            double F_cell = db_cryst.default_F;
            double F_cell2 = 0;
            int i_hklasu=0;
            int Fhkl_linear_index=-1;
            if ( (h0<=db_cryst.h_max) && (h0>=db_cryst.h_min) && (k0<=db_cryst.k_max) && (k0>=db_cryst.k_min) && (l0<=db_cryst.l_max) && (l0>=db_cryst.l_min)  ) {
                /* just take nearest-neighbor */
                Fhkl_linear_index = (h0-db_cryst.h_min) * db_cryst.k_range * db_cryst.l_range +
                                (k0- db_cryst.k_min) * db_cryst.l_range +
                                (l0-db_cryst.l_min);
                F_cell = db_cryst.FhklLinear[Fhkl_linear_index];
                if (db_flags.track_Fhkl){
                    std::string hkl_s ;
                    hkl_s = std::to_string(h0) + ","+ std::to_string(k0) + "," + std::to_string(l0);
                    if (Fhkl_tracker.count(hkl_s))
                        Fhkl_tracker[hkl_s] += 1;
                    else
                        Fhkl_tracker[hkl_s] = 0;
                    continue;
                }
                if (db_flags.complex_miller) F_cell2 = db_cryst.Fhkl2Linear[Fhkl_linear_index];
                if (db_flags.Fhkl_have_scale_factors)
                    i_hklasu = db_cryst.FhklLinear_ASUid[Fhkl_linear_index];
            }
            //else{
            // F_cell = default_F;
            //}
            bool do_max = false;
            if(db_flags.verbose > 3){//} && i_step==0){
                double F2 = sqrt(F_cell*F_cell + F_cell2*F_cell2);
                double II = I_noFcell*F2*F2;
                if (I_noFcell > II_max){
                    do_max = true;
                    max_stats[0] = F_cell;
                    max_stats[1] = F_cell2;
                    max_stats[2] = I_noFcell;
                    max_stats[3] = F_latt;
                    max_stats[4] = II;
                    max_stats[5] = omega_pixel;
                    max_stats[6] = db_beam.source_I[source];
                    max_stats[7]= db_beam.source_lambda[source];
                    max_stats[8] = h;
                    max_stats[9] = k;
                    max_stats[10] = l;
                    II_max = I_noFcell;
                }
                else{
                do_max=false;}
                //printf("hkl= %f %f %f  hkl1= %d %d %d  |Fcell| before=%f, Ino=%f, I=%f, Flatt=%f, cap_frac=%f, omega_pix=%f, sourceI=%f\n", h,k,l,h0,k0,l0, F2, I_noFcell, II, F_latt,
                //capture_fraction, omega_pixel, source_I[source] );
            }
            double c_deriv_Fcell_real = 0;
            double c_deriv_Fcell_imag = 0;
            double d_deriv_Fcell_real = 0;
            double d_deriv_Fcell_imag = 0;
            double c_deriv_Fcell = 0;
            double d_deriv_Fcell = 0;
            if (db_flags.complex_miller){
            // TODO shouldnt this be constant for each HKL?
              if (db_cryst.fpfdp.size() > 0){
                   double S_2 = 1.e-20*(scattering[0]*scattering[0]+scattering[1]*scattering[1]+scattering[2]*scattering[2]);

                    // fp is always followed by the fdp value
                   double val_fp = db_cryst.fpfdp[2*source];
                   double val_fdp = db_cryst.fpfdp[2*source+1];

                   double c_deriv_prime=0;
                   double c_deriv_dblprime=0;
                   double d_deriv_prime = 0;
                   double d_deriv_dblprime = 0;
                   if (db_flags.refine_fp_fdp){
                   //   currently only supports two parameter model
                       int nsources_times_two = db_cryst.fpfdp.size();
                       int d_idx = 2*source;
                       c_deriv_prime = db_cryst.fpfdp_derivs[d_idx];
                       c_deriv_dblprime = db_cryst.fpfdp_derivs[d_idx+1];
                       d_deriv_prime = db_cryst.fpfdp_derivs[d_idx+nsources_times_two];
                       d_deriv_dblprime = db_cryst.fpfdp_derivs[d_idx+1+nsources_times_two];
                   }
                   // 5 valeus per atom: x,y,z,B,occupancy
                   int num_atoms = db_cryst.atom_data.size()/5;
                   for (int  i_atom=0; i_atom < num_atoms; i_atom++){
                       if (db_flags.verbose>5)
                         printf("Processing atom %d, F_cell=%10.3f\n", i_atom, F_cell);
                       // fractional atomic coordinates
                       double atom_x = db_cryst.atom_data[i_atom*5];
                       double atom_y = db_cryst.atom_data[i_atom*5+1];
                       double atom_z = db_cryst.atom_data[i_atom*5+2];
                       double B = db_cryst.atom_data[i_atom*5+3]; // B factor
                       B = exp(-B*S_2/4.0); // TODO: speed me up?
                       double occ = db_cryst.atom_data[i_atom*5+4]; // occupancy
                       double r_dot_h = h0*atom_x + k0*atom_y + l0*atom_z;
                       double phase = 2*M_PI*r_dot_h;
                       double s_rdoth = sin(phase);
                       double c_rdoth = cos(phase);
                       double Bocc = B*occ;
                       double BC = Bocc*c_rdoth;
                       double BS = Bocc*s_rdoth;
                       double real_part = BC*val_fp - BS*val_fdp;
                       double imag_part = BS*val_fp + BC*val_fdp;
                       F_cell += real_part;
                       F_cell2 += imag_part;
                       if (db_flags.refine_fp_fdp){
                            c_deriv_Fcell_real += BC*c_deriv_prime - BS*c_deriv_dblprime;
                            c_deriv_Fcell_imag += BS*c_deriv_prime + BC*c_deriv_dblprime;

                            d_deriv_Fcell_real += BC*d_deriv_prime - BS*d_deriv_dblprime;
                            d_deriv_Fcell_imag += BS*d_deriv_prime + BC*d_deriv_dblprime;
                       }
                   }
               }
               double Freal = F_cell;
               double Fimag = F_cell2;
               F_cell = sqrt(pow(Freal,2) + pow(Fimag,2));// TODO work around the sqrt ?
               //if (verbose> 3 && i_step==0){
               //     double II = I_noFcell*F_cell*F_cell;
               //     printf("hkl= %f %f %f  hkl1= %d %d %d  |Fcell| after=%f, Ino=%f, I=%f\n", h,k,l,h0,k0,l0, F_cell, I_noFcell, II);
               // }
               if (db_flags.refine_fp_fdp){
                   c_deriv_Fcell = Freal*c_deriv_Fcell_real + Fimag*c_deriv_Fcell_imag;
                   d_deriv_Fcell = Freal*d_deriv_Fcell_real + Fimag*d_deriv_Fcell_imag;
               }
            }

            if(db_flags.verbose > 3){
                double F2 = sqrt(F_cell*F_cell + F_cell2*F_cell2);
                double II = I_noFcell*F2*F2;
                if (do_max){
                    max_stats2[0] = F2 ;
                    max_stats2[1] = 0 ; //F_cell2;
                    max_stats2[2] = I_noFcell;
                    max_stats2[3] = F_latt;
                    max_stats2[4] = II;
                    max_stats2[5] = omega_pixel;
                    max_stats2[6] = db_beam.source_I[source];
                    max_stats2[7]= db_beam.source_lambda[source];
                    max_stats2[8] = h;
                    max_stats2[9] = k;
                    max_stats2[10] = l;
                }
            }

            double I_cell = F_cell;
            if (! db_flags.refine_Icell)
                I_cell *= F_cell;
            double s_hkl = 1;
            int Fhkl_channel = 0;
            if (! db_beam.Fhkl_channels.empty())
                Fhkl_channel = db_beam.Fhkl_channels[source];
            if (db_flags.Fhkl_have_scale_factors)
                s_hkl = d_image.Fhkl_scale[i_hklasu + Fhkl_channel*db_cryst.Num_ASU];

            if (db_flags.Fhkl_gradient_mode){
                double Fhkl_deriv_scale = db_cryst.r_e_sqr*db_beam.fluence*db_cryst.spot_scale*polar_for_Fhkl_grad/db_steps.Nsteps;
                double dfhkl = I_noFcell*I_cell * Fhkl_deriv_scale;
                double grad_incr = dfhkl*Fhkl_deriv_coef;
                int fhkl_grad_idx=i_hklasu + Fhkl_channel*db_cryst.Num_ASU;
                if (db_flags.track_Fhkl_indices)
                    db_cryst.Fhkl_grad_idx_tracker.insert(fhkl_grad_idx);
                // omega pixel is correctly in count_scale, and spot_scale should have been set to its refined value
                #pragma omp atomic
                d_image.Fhkl_scale_deriv[fhkl_grad_idx] += grad_incr;

                if (db_flags.Fhkl_errors_mode){
                    double hessian_incr = Fhkl_hessian_coef*dfhkl*dfhkl;
                    #pragma omp atomic
                    d_image.Fhkl_hessian[fhkl_grad_idx] += hessian_incr;
                }
                continue; // move on to next diffraction step (note, thos will bypass the pintout_pixel settings)
            }

            double Iincrement = s_hkl*I_cell*I_noFcell;
            I += Iincrement;
            if(db_flags.wavelength_img){
                Ilambda += Iincrement*lambda_ang;
                Imiller_h += Iincrement * h ;
                Imiller_k += Iincrement * k;
                Imiller_l += Iincrement * l;
            }
            if (db_flags.refine_diffuse){
                double step_scale = count_scale*I_cell;
                for (int i_gam=0; i_gam <3; i_gam++){
                    int i_sig = i_gam + 3;
                    dI_latt_diffuse[i_gam] += step_scale*step_diffuse_param[i_gam];
                    dI_latt_diffuse[i_sig] += step_scale*step_diffuse_param[i_sig];
                }
            }

            if (db_flags.refine_fp_fdp){
                fp_fdp_manager_dI[0] += 2*I_noFcell * (c_deriv_Fcell);
                fp_fdp_manager_dI[1] += 2*I_noFcell * (d_deriv_Fcell);
            }

            //if(verbose > 3)
            //    printf("hkl= %f %f %f  hkl1= %d %d %d  Fcell=%f\n", h,k,l,h0,k0,l0, F_cell);

            double two_C = 2*C;
            if (db_flags.refine_Umat[0]){
                Eigen::Matrix3d RyRzUBOt = db_cryst.RotMats[1]*db_cryst.RotMats[2]*UBOt;
                Eigen::Vector3d delta_H_prime = (db_cryst.UMATS[mos_tic]*db_cryst.dRotMats[0]*RyRzUBOt).transpose()*q_vec;
                double V_dot_dV = V.dot(NABC*delta_H_prime);
                double value = -two_C * V_dot_dV * Iincrement;
                double value2 =0;
                if (db_flags.compute_curvatures) {
                    Eigen::Vector3d delta_H_dbl_prime = (db_cryst.UMATS[mos_tic]*db_cryst.d2RotMats[0]*RyRzUBOt).transpose()*q_vec;
                    double dV_dot_dV = (NABC*delta_H_prime).dot(NABC*delta_H_prime);
                    double dV2_dot_V = (NABC*delta_H).dot(NABC*delta_H_dbl_prime);
                    value2 = two_C*(two_C*V_dot_dV*V_dot_dV - dV2_dot_V - dV_dot_dV)*Iincrement;
                }
                rot_manager_dI[0] += value;
                rot_manager_dI2[0] += value2;
            }
            if (db_flags.refine_Umat[1]){
                Eigen::Matrix3d UmosRx = db_cryst.UMATS[mos_tic]*db_cryst.RotMats[0];
                Eigen::Matrix3d RzUBOt = db_cryst.RotMats[2]*UBOt;
                Eigen::Vector3d delta_H_prime =(UmosRx*db_cryst.dRotMats[1]*RzUBOt).transpose()*q_vec;
                double V_dot_dV = V.dot(NABC*delta_H_prime);
                double value = -two_C * V_dot_dV * Iincrement;

                double value2=0;
                if (db_flags.compute_curvatures){
                    Eigen::Vector3d delta_H_dbl_prime = (UmosRx*db_cryst.d2RotMats[1]*RzUBOt).transpose()*q_vec;
                    double dV_dot_dV = (NABC*delta_H_prime).dot(NABC*delta_H_prime);
                    double dV2_dot_V = (NABC*delta_H).dot(NABC*delta_H_dbl_prime);
                    value2 = two_C*(two_C*V_dot_dV*V_dot_dV - dV2_dot_V - dV_dot_dV)*Iincrement;
                }
                rot_manager_dI[1] += value;
                rot_manager_dI2[1] += value2;
            }
            if (db_flags.refine_Umat[2]){
                Eigen::Matrix3d UmosRxRy = db_cryst.UMATS[mos_tic]*db_cryst.RotMats[0]*db_cryst.RotMats[1];
                Eigen::Vector3d delta_H_prime = (UmosRxRy*db_cryst.dRotMats[2]*UBOt).transpose()*q_vec;
                double V_dot_dV = V.dot(NABC*delta_H_prime);
                double value = -two_C * V_dot_dV * Iincrement;

                double value2=0;
                if (db_flags.compute_curvatures){
                    Eigen::Vector3d delta_H_dbl_prime = (UmosRxRy*db_cryst.d2RotMats[2]*UBOt).transpose()*q_vec;
                    double dV_dot_dV = (NABC*delta_H_prime).dot(NABC*delta_H_prime);
                    double dV2_dot_V = (NABC*delta_H).dot(NABC*delta_H_dbl_prime);
                    value2 = two_C*(two_C*V_dot_dV*V_dot_dV - dV2_dot_V - dV_dot_dV)*Iincrement;
                }
                rot_manager_dI[2] += value;
                rot_manager_dI2[2] += value2;
            }
            /*Checkpoint for unit cell derivatives*/
            Eigen::Matrix3d Ot = db_cryst.eig_O.transpose();
            Eigen::Matrix3d UmosRxRyRzU ;
            Eigen::Vector3d delta_H_prime;
            for(int i_uc=0; i_uc < 6; i_uc++ ){
                if (db_flags.refine_Bmat[i_uc]){
                    UmosRxRyRzU = db_cryst.UMATS_RXYZ[mos_tic]*U;
                    delta_H_prime = ((UmosRxRyRzU*db_cryst.dB_Mats[i_uc]*Ot).transpose()*q_vec);
                    double V_dot_dV = V.dot(NABC*delta_H_prime);
                    double value = -two_C * V_dot_dV * Iincrement;
                    double value2 =0;
                    if (db_flags.compute_curvatures){
                        Eigen::Vector3d delta_H_dbl_prime = ((UmosRxRyRzU*db_cryst.dB2_Mats[i_uc]*Ot).transpose()*q_vec);
                        double dV_dot_dV = (NABC*delta_H_prime).dot(NABC*delta_H_prime);
                        double dV2_dot_V = (NABC*delta_H).dot(NABC*delta_H_dbl_prime);
                        value2 = two_C*(two_C*V_dot_dV*V_dot_dV - dV2_dot_V - dV_dot_dV)*Iincrement;
                    }
                    ucell_manager_dI[i_uc] += value;
                    ucell_manager_dI2[i_uc] += value2;
                }
            } /*end ucell deriv */

            /* Checkpoint for Ncells manager */
            if (db_flags.refine_Ncells[0]){
                int num_ncell_deriv = 1;
                if (! db_flags.isotropic_ncells)
                    num_ncell_deriv = 3;
                for (int i_nc=0; i_nc < num_ncell_deriv; i_nc++) {
                    Eigen::Matrix3d dN;
                    if (!db_flags.isotropic_ncells){
                        dN << 0,0,0,0,0,0,0,0,0;
                        dN(i_nc, i_nc) = 1;
                        }
                    else
                        dN << 1,0,0,0,1,0,0,0,1;

                    double N_i = NABC(i_nc, i_nc);
                    Eigen::Vector3d dV_dN = dN*delta_H;
                    double determ_deriv = (NABC.inverse()*dN).trace(); // TODO speedops: precompute these
                    double deriv_coef = determ_deriv - C* ( dV_dN.dot(V));
                    double value = 2*Iincrement*deriv_coef;
                    double value2=0;
                    if(db_flags.compute_curvatures){
                        dN(i_nc, i_nc) = 0; // TODO check maths
                        value2 = ( -1/N_i/N_i - C*(dV_dN.dot(dV_dN))) *2*Iincrement;
                        value2 += deriv_coef*2*value;
                    }
                    Ncells_manager_dI[i_nc] += value;
                    Ncells_manager_dI2[i_nc] += value2;
                }

            } /* end Ncells manager deriv */

            if (db_flags.refine_Ncells_def){
                for (int i_nc =3; i_nc < 6; i_nc++ ){
                    Eigen::Matrix3d dN;
                    if (i_nc ==3)
                        dN << 0,1,0,1,0,0,0,0,0;
                    else if (i_nc == 4)
                        dN << 0,0,0,0,0,1,0,1,0;
                    else
                        dN << 0,0,1,0,0,0,1,0,0;
                    Eigen::Vector3d dV_dN = dN*delta_H;
                    double determ_deriv = (NABC.inverse()*dN).trace(); // TODO speedops: precompute these
                    //double deriv_coef = -C* (2* dV_dN.dot(V));
                    double deriv_coef = determ_deriv - C* (dV_dN.dot(V));
                    double value = 2*Iincrement*deriv_coef;
                    Ncells_manager_dI[i_nc] += value;
                    double value2 =0;
                    if (db_flags.compute_curvatures){
                        value2 = deriv_coef*value;
                        value2 +=  -2*C*Iincrement*(dV_dN.dot(dV_dN));
                    Ncells_manager_dI2[i_nc] += value2;
                    }
                }
            }

            ///* Checkpoint for Origin manager */
            for (int i_pan_orig=0; i_pan_orig < 3; i_pan_orig++){
                if (db_flags.refine_panel_origin[i_pan_orig]){
                    double per_k = 1/airpath;
                    double per_k3 = pow(per_k,3);
                    double per_k5 = pow(per_k,5);
                    double lambda_ang = lambda*1e10;

                    Eigen::Matrix3d M = -two_C*(NABC*UBO)/lambda_ang;
                    Eigen::Vector3d dk;
                    if (i_pan_orig == 0)
                        dk << 0,0,1;
                    else if (i_pan_orig == 1)
                        dk << 1,0,0;
                    else
                        dk << 0,1,0;
                    af::flex_double dI_and_dI2 = get_panel_increment(Iincrement, omega_pixel, M, db_det.subpixel_size*db_det.subpixel_size,
                        o_vec, pixel_pos, per_k,  per_k3, per_k5, V, dk);
                    pan_orig_manager_dI[i_pan_orig] += dI_and_dI2[0];
                    pan_orig_manager_dI2[i_pan_orig] += dI_and_dI2[1];

                } /* end origin manager deriv */
            }

            for (int i_pan_rot=0; i_pan_rot < 3; i_pan_rot++){
                if(db_flags.refine_panel_rot[i_pan_rot]){
                    double per_k = 1/airpath;
                    double per_k3 = pow(per_k,3);
                    double per_k5 = pow(per_k,5);
                    double lambda_ang = lambda*1e10;
                    Eigen::Matrix3d M = -two_C*(NABC*UBO)/lambda_ang;

                    Eigen::Vector3d dk = Fdet*(db_det.dF_vecs[pid*3 + i_pan_rot]) + Sdet*(db_det.dS_vecs[pid*3 + i_pan_rot]);
                    af::flex_double dI_and_dI2 = get_panel_increment(Iincrement, omega_pixel, M, db_det.subpixel_size*db_det.subpixel_size,
                        o_vec, pixel_pos, per_k,  per_k3, per_k5, V, dk);
                    pan_rot_manager_dI[i_pan_rot] += dI_and_dI2[0];
                    pan_rot_manager_dI2[i_pan_rot] += dI_and_dI2[1];
                }
            }

            /* checkpoint for Fcell manager */
            if (db_flags.refine_fcell){
            //    TODO rewrite so no divide by Fcell
                int nom_h=h0;
                int nom_k=k0;
                int nom_l=l0;
                //int f_cell_idx = 1;
                if (use_nominal_hkl){

                    nom_h = db_cryst.nominal_hkl[i_pix*3];
                    nom_k = db_cryst.nominal_hkl[i_pix*3+1];
                    nom_l = db_cryst.nominal_hkl[i_pix*3+2];
                    //f_cell_idx = l0 - nom_l + 1;
                }
                double value;
                if (db_flags.refine_Icell)
                    value = I_noFcell;
                else
                    value = 2*I_noFcell*F_cell; //2*Iincrement/F_cell ;
                double value2=0;
                if (db_flags.compute_curvatures){
                    if (F_cell > 0)
                        value2 =2*I_noFcell;
                }
                //if (f_cell_idx >= 0 && f_cell_idx <= 2){
                // NOTE use nominal hkl to regulate when gradients are compputed, as h,k,l can drift within a shoebox
                if (use_nominal_hkl){
                    if (nom_h==h0 && nom_k==k0 && nom_l==l0 ){
                        fcell_manager_dI += value;
                        fcell_manager_dI2 += value2;
                    }
                }
                else{
                    fcell_manager_dI += value;
                    fcell_manager_dI2 += value2;
                }
            } /* end of fcell man deriv */

            /* checkpoint for eta manager */
            if (db_flags.refine_eta){
                for (int i_eta=0; i_eta < 3; i_eta++){
                    if (i_eta != 0 && db_cryst.UMATS_RXYZ_prime.size()==db_cryst.UMATS_RXYZ.size()){
                        continue;
                    }
                    int mtic2 = mos_tic  + i_eta*db_cryst.mosaic_domains;
                    Eigen::Vector3d DeltaH_deriv = (db_cryst.UMATS_RXYZ_prime[mtic2]*UBOt).transpose()*q_vec;
                    // vector V is Nabc*Delta_H
                    Eigen::Vector3d dV = NABC*DeltaH_deriv;
                    double V_dot_dV = V.dot(dV);
                    double Iprime = -two_C*(V_dot_dV)*Iincrement;
                    eta_manager_dI[i_eta] += Iprime;

                    double Idbl_prime = 0;
                    if (db_flags.compute_curvatures){
                        Eigen::Vector3d DeltaH_second_deriv = (db_cryst.UMATS_RXYZ_dbl_prime[mtic2]*UBOt).transpose()*q_vec;
                        Eigen::Vector3d dV2 = NABC*DeltaH_second_deriv;
                        Idbl_prime = -two_C*(dV.dot(dV) + V.dot(dV2))*Iincrement;
                        Idbl_prime += -two_C*(V_dot_dV)*Iprime;
                    }
                    eta_manager_dI2[i_eta] += Idbl_prime;
                }
            } /* end of eta man deriv */

            /*checkpoint for lambda manager*/
            for(int i_lam=0; i_lam < 2; i_lam++){
                if (db_flags.refine_lambda[i_lam]){
                    double lambda_ang = lambda*1e10;
                    double NH_dot_V = (NABC*H_vec).dot(V);
                    double dg_dlambda;
                    if (i_lam==0)
                        dg_dlambda = 1;
                    else // i_lam==1
                        dg_dlambda = lambda_ang;
                    double coef = NH_dot_V*two_C*(dg_dlambda) / lambda_ang;
                    double value = coef*Iincrement;
                    double value2 = 0;
                    //if (compute_curvatures)
                    lambda_manager_dI[i_lam] += value;
                    lambda_manager_dI2[i_lam] += value2;
                    //lambda_managers[i_lam]->increment(value, value2);
                }
            }
            /*end of lambda deriv*/

            if( db_flags.printout && i_step==0 ){
                if((fpixel==db_flags.printout_fpixel && spixel==db_flags.printout_spixel) || db_flags.printout_fpixel < 0)
                {
                    printf("%4d %4d : stol = %g, lambda = %g\n", fpixel,spixel,stol, lambda);
                    printf("at %g %g %g\n", pixel_pos[0],pixel_pos[1],pixel_pos[2]);
                    printf("hkl= %f %f %f  hkl0= %d %d %d\n", h,k,l,h0,k0,l0);
                    printf(" F_cell=%g  F_cell_2=%g F_latt=%g   I = %g\n", F_cell,F_cell2,F_latt,I);
                    printf("I/steps %15.10g\n", I/db_steps.Nsteps);
                    printf("cap frac   %f\n", capture_fraction);
                    printf("Fdet= %g; Sdet= %g ; Odet= %g\n", Fdet, Sdet, Odet);
                    printf("omega   %15.10g\n", omega_pixel);
                    printf("close_distance    %15.10g\n", close_distance);
                    printf("fluence    %15.10g\n", db_beam.fluence);
                    printf("spot scale   %15.10g\n", db_cryst.spot_scale);
                    printf("sourceI[0]:   %15.10g\n", db_beam.source_I[0]);
                    printf("default_F= %f\n", db_cryst.default_F);
                    printf("PIX0: %f %f %f\n" , db_det.pix0_vectors[pid_x], db_det.pix0_vectors[pid_y], db_det.pix0_vectors[pid_z]);
                    printf("F: %f %f %f\n" , db_det.fdet_vectors[pid_x], db_det.fdet_vectors[pid_y], db_det.fdet_vectors[pid_z]);
                    printf("S: %f %f %f\n" , db_det.sdet_vectors[pid_x], db_det.sdet_vectors[pid_y], db_det.sdet_vectors[pid_z]);
                    printf("O: %f %f %f\n" , db_det.odet_vectors[pid_x], db_det.odet_vectors[pid_y], db_det.odet_vectors[pid_z]);
                    printf("QVECTOR: %f %f %f\n" , q_vec[0], q_vec[1], q_vec[2]);
                    printf("MOSAIC UMAT RXYZ\n");
                    std::cout << db_cryst.UMATS_RXYZ[mos_tic] << std::endl;
                    printf("Bmat_realspace\n");
                    std::cout << Bmat_realspace << std::endl;
                    printf("eig_U\n");
                    std::cout << db_cryst.eig_U << std::endl;
                    printf("eig_O\n");
                    std::cout << db_cryst.eig_O << std::endl;
                    printf("UBO\n");
                    std::cout << UBO << std::endl;
                    printf("UBOt\n");
                    std::cout << UBOt << std::endl;
                    printf("UmosRxRyRzU\n");
                    std::cout << UmosRxRyRzU << std::endl;
                    printf("deltaHprime\n");
                    std::cout << delta_H_prime << std::endl;
                    printf("Iincrement: %f\n", Iincrement);
                    printf("pid_x=%d, pid_y=%d; pid_z=%d\n", pid_x, pid_y, pid_z);
                    SCITBX_EXAMINE(db_det.fdet_vectors[0]);
                    SCITBX_EXAMINE(db_det.fdet_vectors[1]);
                    SCITBX_EXAMINE(db_det.fdet_vectors[2]);
                    SCITBX_EXAMINE(db_det.pix0_vectors[0]);
                    SCITBX_EXAMINE(db_det.pix0_vectors[1]);
                    SCITBX_EXAMINE(db_det.pix0_vectors[2]);
                }
            }
        } /* end of i_steps loop */

        if (db_flags.verbose >3){
            printf("hkl= %f %f %f  |Fcell| before=(%f,%f), Ino=%f, I=%f, Flatt=%f, omega_pix=%2.3e, sourceI=%f, sourceLambda=%f\n",
             max_stats[8],max_stats[9],max_stats[10],
             max_stats[0], max_stats[1], max_stats[2], max_stats[4], max_stats[3],
            max_stats[5], max_stats[6], max_stats[7]*1e10 );

            printf("hkl= %f %f %f  |Fcell| after=(%f,%f), Ino=%f, I=%f, Flatt=%f, omega_pix=%2.3e, sourceI=%f, sourceLambda=%f\n",
             max_stats2[8],max_stats2[9],max_stats2[10],
             max_stats2[0], max_stats2[1], max_stats2[2], max_stats2[4], max_stats2[3],
            max_stats2[5], max_stats2[6], max_stats2[7]*1e10 );

        }

        double scale_term=1;
        if (!db_flags.track_Fhkl){

            double Fdet_ave = db_det.pixel_size*fpixel + db_det.pixel_size/2.0;
            double Sdet_ave = db_det.pixel_size*spixel + db_det.pixel_size/2.0;
            double Odet_ave = 0; //Odet; // TODO maybe make this more general for thick detectors?

            Eigen::Vector3d pixel_pos_ave(0,0,0);
            int pid_x = pid*3;
            int pid_y = pid*3+1;
            int pid_z = pid*3+2;
            pixel_pos_ave[0] = Fdet_ave * db_det.fdet_vectors[pid_x]+Sdet_ave*db_det.sdet_vectors[pid_x]+Odet_ave*db_det.odet_vectors[pid_x]+db_det.pix0_vectors[pid_x];
            pixel_pos_ave[1] = Fdet_ave * db_det.fdet_vectors[pid_y]+Sdet_ave*db_det.sdet_vectors[pid_y]+Odet_ave*db_det.odet_vectors[pid_y]+db_det.pix0_vectors[pid_y];
            pixel_pos_ave[2] = Fdet_ave * db_det.fdet_vectors[pid_z]+Sdet_ave*db_det.sdet_vectors[pid_z]+Odet_ave*db_det.odet_vectors[pid_z]+db_det.pix0_vectors[pid_z];

            double airpath_ave = pixel_pos_ave.norm();
            Eigen::Vector3d diffracted_ave = pixel_pos_ave/airpath_ave;
            double omega_pixel_ave = db_det.pixel_size*db_det.pixel_size/airpath_ave/airpath_ave*close_distance/airpath_ave;

            double polar = 1;
            if (!db_flags.nopolar){
                Eigen::Vector3d incident(-db_beam.source_X[0], -db_beam.source_Y[0], -db_beam.source_Z[0]);
                incident = incident / incident.norm();
                polar = diffBragg_cpu_kernel_polarization(incident, diffracted_ave,
                                    db_beam.polarization_axis, db_beam.kahn_factor);
            }

            double om = 1;
            if (!db_flags.oversample_omega)
                om=omega_pixel_ave;

            // final scale term to being everything to photon number units
            scale_term = db_cryst.r_e_sqr*db_beam.fluence*db_cryst.spot_scale*polar*om/db_steps.Nsteps;

            if (db_flags.Fhkl_gradient_mode)
                continue; // move on to next pixel (i_pix)

            floatimage[i_pix] = scale_term*I;

            if (db_flags.wavelength_img){
                d_image.wavelength[i_pix*4] = Ilambda / I;
                d_image.wavelength[i_pix*4+1] = Imiller_h / I;
                d_image.wavelength[i_pix*4+2] = Imiller_k / I;
                d_image.wavelength[i_pix*4+3] = Imiller_l / I;
            }
        }
        if (db_flags.refine_diffuse){
            for (int i_diff=0; i_diff < 6; i_diff++){
                double val = dI_latt_diffuse[i_diff]*scale_term;
                if (i_diff<3) {
                  int img_idx = Npix_to_model*i_diff + i_pix;
                  d_image.diffuse_gamma[img_idx] = val;
                } else {
                  int img_idx = Npix_to_model*(i_diff-3) + i_pix;
                  d_image.diffuse_sigma[img_idx] = val;
                }
            }
        }
        /* udpate the rotation derivative images*/
        for (int i_rot =0 ; i_rot < 3 ; i_rot++){
            if (db_flags.refine_Umat[i_rot]){
                double value = scale_term*rot_manager_dI[i_rot];
                double value2 = scale_term*rot_manager_dI2[i_rot];
                int idx = i_rot*Npix_to_model + i_pix;
                d_image.Umat[idx] = value;
                d2_image.Umat[idx] = value2;
            }
        } /* end rot deriv image increment */

        /*update the ucell derivative images*/
        for (int i_uc=0 ; i_uc < 6 ; i_uc++){
            if (db_flags.refine_Bmat[i_uc]){
                double value = scale_term*ucell_manager_dI[i_uc];
                double value2 = scale_term*ucell_manager_dI2[i_uc];
                int idx= i_uc*Npix_to_model + i_pix;
                d_image.Bmat[idx] = value;
                d2_image.Bmat[idx] = value2;
            }
        }/* end ucell deriv image increment */

        /*update the Ncells derivative image*/
        if (db_flags.refine_Ncells[0]){
            double value = scale_term*Ncells_manager_dI[0];
            double value2 = scale_term*Ncells_manager_dI2[0];
            double idx = i_pix;
            d_image.Ncells[idx] = value;
            d2_image.Ncells[idx] = value2;
            //d_Ncells_images[idx] = value;
            //d2_Ncells_images[idx] = value2;

            if (! db_flags.isotropic_ncells){
                value = scale_term*Ncells_manager_dI[1];
                value2 = scale_term*Ncells_manager_dI2[1];
                idx = Npix_to_model + i_pix;
                d_image.Ncells[idx] = value;
                d2_image.Ncells[idx] = value2;

                value = scale_term*Ncells_manager_dI[2];
                value2 = scale_term*Ncells_manager_dI2[2];
                idx = Npix_to_model*2 + i_pix;
                d_image.Ncells[idx] = value;
                d2_image.Ncells[idx] = value2;
            }
        }/* end Ncells deriv image increment */
        if (db_flags.refine_Ncells_def){
            for (int i_nc=3; i_nc<6; i_nc++){
                double value = scale_term*Ncells_manager_dI[i_nc];
                double value2 = scale_term*Ncells_manager_dI2[i_nc];
                int idx = i_nc* Npix_to_model + i_pix;
                //d_Ncells_images[idx] = value;
                //d2_Ncells_images[idx] = value2;
                d_image.Ncells[idx] = value;
                d2_image.Ncells[idx] = value2;
            }
        }

        /* update Fcell derivative image */
        if(db_flags.refine_fcell){
            double value = scale_term*fcell_manager_dI;
            double value2 = scale_term*fcell_manager_dI2;
            d_image.fcell[i_pix] = value;
            d2_image.fcell[i_pix] = value2;
        }/* end Fcell deriv image increment */

        if (db_flags.refine_fp_fdp){
            // c derivative
            double value = scale_term*fp_fdp_manager_dI[0];
            d_image.fp_fdp[i_pix] = value;
            // d derivative
            value = scale_term*fp_fdp_manager_dI[1];
            d_image.fp_fdp[Npix_to_model + i_pix] = value;
        }

        /* update eta derivative image */
        if(db_flags.refine_eta){
            //double value = scale_term*eta_manager_dI[0];
            //double value2 = scale_term*eta_manager_dI2[0];
            //d_eta_images[i_pix] = value;
            //d2_eta_images[i_pix] = value2;
            //if (db_cryst.UMATS_RXYZ_prime.size() >= 3*mosaic_domains){
            for(int i_eta=0; i_eta<3; i_eta++){
                if (i_eta != 0 && db_cryst.UMATS_RXYZ.size() == db_cryst.UMATS_RXYZ_prime.size())
                    continue;
                int idx = i_pix + Npix_to_model*i_eta;
                double value = scale_term*eta_manager_dI[i_eta];
                double value2 = scale_term*eta_manager_dI2[i_eta];
                d_image.eta[idx] = value;
                d2_image.eta[idx] = value2;
            }
            //}
        }/* end eta deriv image increment */

        /*update the lambda derivative images*/
        for (int i_lam=0 ; i_lam < 2 ; i_lam++){
            if (db_flags.refine_lambda[i_lam]){
                //double value = scale_term*lambda_managers[i_lam]->dI;
                //double value2 = scale_term*lambda_managers[i_lam]->dI2;
                double value = scale_term*lambda_manager_dI[i_lam];
                double value2 = scale_term*lambda_manager_dI2[i_lam];
                int idx = i_lam*Npix_to_model + i_pix;
                d_image.lambda[idx] = value;
                d2_image.lambda[idx] = value2;
            }
        }/* end lambda deriv image increment */

        // panel rotation
        for (int i_pan_rot=0; i_pan_rot < 3; i_pan_rot++){
            if(db_flags.refine_panel_rot[i_pan_rot]){
                double value = scale_term*pan_rot_manager_dI[i_pan_rot];
                double value2 = scale_term*pan_rot_manager_dI2[i_pan_rot];
                int idx = i_pan_rot*Npix_to_model + i_pix;
                d_image.panel_rot[idx] = value;
                d2_image.panel_rot[idx] = value2;
            }
        }/* end panel rot deriv image increment */

        // panel origin
        for (int i_pan_orig=0; i_pan_orig < 3; i_pan_orig++){
            if(db_flags.refine_panel_origin[i_pan_orig]){
                double value = scale_term*pan_orig_manager_dI[i_pan_orig];
                double value2 = scale_term*pan_orig_manager_dI2[i_pan_orig];
                int idx = i_pan_orig*Npix_to_model + i_pix;
                d_image.panel_orig[idx] = value;
                d2_image.panel_orig[idx] = value2;
            }/* end panel orig deriv image increment */
        }
        if (db_flags.track_Fhkl){
            for(auto &x: Fhkl_tracker)
                printf("Pixel %d: Fhkl linear index %s came up %d times\n", i_pix, x.first.c_str(), x.second);
        }
    } // end i_pix loop

#if defined(_OPENMP)
    if (db_flags.track_Fhkl_indices){
        omp_set_dynamic(dyn_state);
        omp_set_num_threads(nthread_state);
    }
#endif
} // END of CPU kernel

}} // END namespace simtbx::nanoBragg
