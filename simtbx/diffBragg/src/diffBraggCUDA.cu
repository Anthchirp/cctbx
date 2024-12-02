#include <sys/time.h>
#include "diffBraggCUDA.h"
#include "diffBragg_gpu_kernel.h"
#include <stdio.h>
//lkalskdlaksdlkalsd

//#define BLOCKSIZE 128
//#define NUMBLOCKS 128
//https://stackoverflow.com/a/14038590/2077270
#define gpuErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void error_msg(cudaError_t err, const char* msg){
    if (err != cudaSuccess){
        printf("%s: CUDA error message: %s\n", msg, cudaGetErrorString(err));
        exit(err);
    }
}

void diffBragg_sum_over_steps_cuda(
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
        diffBragg_cudaPointers& cp,
        timer_variables& TIMERS){

    int numblocks;
    int blocksize;
    char* diffBragg_blocks = getenv("DIFFBRAGG_NUM_BLOCKS");
    char* diffBragg_threads = getenv("DIFFBRAGG_THREADS_PER_BLOCK");
    if (diffBragg_threads==NULL)
        blocksize=128;
    else
        blocksize=atoi(diffBragg_threads);

    if (diffBragg_blocks==NULL)
        numblocks = (Npix_to_model+blocksize-1)/blocksize;
    else
        numblocks = atoi(diffBragg_blocks);

    int cuda_devices;
    cudaGetDeviceCount(&cuda_devices);

    error_msg(cudaGetLastError(), "after device count");
    if (db_flags.verbose > 1)
        printf("Found %d CUDA-capable devices\n", cuda_devices);

    //if (device_Id <= cuda_devices)
    gpuErr(cudaSetDevice(db_cu_flags.device_Id));

    double time;
    struct timeval t1, t2;//, t3 ,t4;
    gettimeofday(&t1, 0);

//  determine if we need to allocate pixels, and how many.
//  For best usage, one should use the diffBragg property (visible from Python) Npix_to_allocate
//  in order to just allocate to the GPU - this is useful for ensemble refinement, where each shot
//  can have a variable number of pixels being modeled, and ony only needs to allocate the device once
//  (with the largest expected number of pixels for a given shot)
   // TODO clean up this logic a bit
    if (cp.device_is_allocated && (cp.npix_allocated < Npix_to_model)){
        printf("Need to re-allocate pixels, currently have %d allocated, but trying to model %d\n",
            cp.npix_allocated, Npix_to_model);
        exit(-1);
    }
    else if (db_cu_flags.Npix_to_allocate==-1){
        db_cu_flags.Npix_to_allocate = Npix_to_model;
    }
    else if (Npix_to_model > db_cu_flags.Npix_to_allocate){
        printf("Npix to model=%d is greater than the number of pixel requested for allocation (%d)!\n",
            Npix_to_model, db_cu_flags.Npix_to_allocate);
        exit(-1);
    }

//  support dynamic allocation for different numbers of sources
    if ( cp.previous_nsource != 0 && cp.previous_nsource != db_beam.number_of_sources){
        gpuErr(cudaFree(cp.Fhkl_channels));
        gpuErr(cudaFree(cp.cu_source_X));
        gpuErr(cudaFree(cp.cu_source_Y));
        gpuErr(cudaFree(cp.cu_source_Z));
        gpuErr(cudaFree(cp.cu_source_I));
        gpuErr(cudaFree(cp.cu_source_lambda));
        //printf("Reallocating for  %d sources!:\n", db_beam.number_of_sources);
        gpuErr(cudaMallocManaged(&cp.Fhkl_channels, db_beam.number_of_sources*sizeof(CUDAREAL)));
        gpuErr(cudaMallocManaged(&cp.cu_source_X, db_beam.number_of_sources*sizeof(CUDAREAL)));
        gpuErr(cudaMallocManaged(&cp.cu_source_Y, db_beam.number_of_sources*sizeof(CUDAREAL)));
        gpuErr(cudaMallocManaged(&cp.cu_source_Z, db_beam.number_of_sources*sizeof(CUDAREAL)));
        gpuErr(cudaMallocManaged(&cp.cu_source_I, db_beam.number_of_sources*sizeof(CUDAREAL)));
        gpuErr(cudaMallocManaged(&cp.cu_source_lambda, db_beam.number_of_sources*sizeof(CUDAREAL)));
        cp.previous_nsource = db_beam.number_of_sources;
    }

    if(cp.device_is_allocated){
        if (db_flags.verbose){
           printf("Will model %d pixels (GPU has %d pre-allocated pix)\n", Npix_to_model, cp.npix_allocated);
        }
    }
    else{
        if (db_flags.verbose){
           printf("Will model %d pixels and allocate %d pix\n", Npix_to_model, db_cu_flags.Npix_to_allocate);
        }
        // Check the Fhkl geradient arrays
        if (db_flags.Fhkl_have_scale_factors){
            gpuErr(cudaMallocManaged(&cp.data_residual, db_cu_flags.Npix_to_allocate*sizeof(CUDAREAL)));
            gpuErr(cudaMallocManaged(&cp.data_variance, db_cu_flags.Npix_to_allocate*sizeof(CUDAREAL)));
            gpuErr(cudaMallocManaged(&cp.data_freq, db_cu_flags.Npix_to_allocate*sizeof(int)));
            gpuErr(cudaMallocManaged(&cp.data_trusted, db_cu_flags.Npix_to_allocate*sizeof(bool)));
            gpuErr(cudaMallocManaged(&cp.FhklLinear_ASUid, db_cryst.FhklLinear_ASUid.size()*sizeof(int)));
            gpuErr(cudaMallocManaged(&cp.Fhkl_scale, d_image.Fhkl_scale.size()*sizeof(CUDAREAL)));
            // alloc Fhkl_scale_deriv to bs same length as Fhkl_scale.size(), as Fhkl_scale_deriv is only set when Fhkl_gradient_mode=True, typpically not first iteration
            gpuErr(cudaMallocManaged(&cp.Fhkl_scale_deriv, d_image.Fhkl_scale.size()*sizeof(CUDAREAL)));
            cp.Fhkl_grad_arrays_allocated = true;
        }

        gpuErr(cudaMallocManaged(&cp.Fhkl_channels, db_beam.number_of_sources*sizeof(CUDAREAL)));
        gpuErr(cudaMallocManaged(&cp.cu_source_X, db_beam.number_of_sources*sizeof(CUDAREAL)));
        gpuErr(cudaMallocManaged(&cp.cu_source_Y, db_beam.number_of_sources*sizeof(CUDAREAL)));
        gpuErr(cudaMallocManaged(&cp.cu_source_Z, db_beam.number_of_sources*sizeof(CUDAREAL)));
        gpuErr(cudaMallocManaged(&cp.cu_source_I, db_beam.number_of_sources*sizeof(CUDAREAL)));
        gpuErr(cudaMallocManaged(&cp.cu_source_lambda, db_beam.number_of_sources*sizeof(CUDAREAL)));
        cp.previous_nsource = db_beam.number_of_sources;

        gpuErr(cudaMallocManaged((void **)&cp.cu_UMATS, db_cryst.UMATS.size()*sizeof(MAT3)));
        gpuErr(cudaMallocManaged((void **)&cp.cu_UMATS_RXYZ, db_cryst.UMATS_RXYZ.size()*sizeof(MAT3)));
        gpuErr(cudaMallocManaged((void **)&cp.cu_AMATS, db_cryst.UMATS_RXYZ.size()*sizeof(MAT3)));
        if (db_cryst.UMATS_RXYZ_prime.size()>0)
            gpuErr(cudaMallocManaged((void **)&cp.cu_UMATS_RXYZ_prime, db_cryst.UMATS_RXYZ_prime.size()*sizeof(MAT3)));
        if (db_cryst.UMATS_RXYZ_dbl_prime.size()>0)
            gpuErr(cudaMallocManaged((void **)&cp.cu_UMATS_RXYZ_dbl_prime, db_cryst.UMATS_RXYZ_dbl_prime.size()*sizeof(MAT3)));

        gpuErr(cudaMallocManaged((void **)&cp.cu_dB_Mats, db_cryst.dB_Mats.size()*sizeof(MAT3)));
        gpuErr(cudaMallocManaged((void **)&cp.cu_dB2_Mats, db_cryst.dB2_Mats.size()*sizeof(MAT3)));

        gpuErr(cudaMallocManaged((void **)&cp.cu_RotMats, db_cryst.RotMats.size()*sizeof(MAT3)));
        gpuErr(cudaMallocManaged((void **)&cp.cu_dRotMats, db_cryst.dRotMats.size()*sizeof(MAT3)));
        gpuErr(cudaMallocManaged((void **)&cp.cu_d2RotMats, db_cryst.d2RotMats.size()*sizeof(MAT3)));

        gpuErr(cudaMallocManaged(&cp.cu_fdet_vectors, db_det.fdet_vectors.size()*sizeof(CUDAREAL)));
        gpuErr(cudaMallocManaged(&cp.cu_sdet_vectors, db_det.fdet_vectors.size()*sizeof(CUDAREAL)));
        gpuErr(cudaMallocManaged(&cp.cu_odet_vectors, db_det.fdet_vectors.size()*sizeof(CUDAREAL)));
        gpuErr(cudaMallocManaged(&cp.cu_pix0_vectors, db_det.fdet_vectors.size()*sizeof(CUDAREAL)));
        gpuErr(cudaMallocManaged(&cp.cu_close_distances, db_det.close_distances.size()*sizeof(CUDAREAL)));

        if (db_cryst.fpfdp.size() > 0){
            gpuErr(cudaMallocManaged(&cp.cu_fpfdp, db_cryst.fpfdp.size()*sizeof(CUDAREAL)));
            gpuErr(cudaMallocManaged(&cp.cu_atom_data, db_cryst.atom_data.size()*sizeof(CUDAREAL)));
        }
        if(db_cryst.fpfdp_derivs.size() > 0)
            gpuErr(cudaMallocManaged(&cp.cu_fpfdp_derivs, db_cryst.fpfdp_derivs.size()*sizeof(CUDAREAL)));

        gpuErr(cudaMallocManaged(&cp.cu_refine_Bmat, 6*sizeof(bool)));
        gpuErr(cudaMallocManaged(&cp.cu_refine_Umat, 3*sizeof(bool)));
        gpuErr(cudaMallocManaged(&cp.cu_refine_Ncells, 3*sizeof(bool)));
        gpuErr(cudaMallocManaged(&cp.cu_refine_panel_origin, 3*sizeof(bool)));
        gpuErr(cudaMallocManaged(&cp.cu_refine_panel_rot, 3*sizeof(bool)));
        gpuErr(cudaMallocManaged(&cp.cu_refine_lambda, 2*sizeof(bool)));

        gpuErr(cudaMallocManaged(&cp.cu_Fhkl, db_cryst.FhklLinear.size()*sizeof(CUDAREAL)));
        if (db_flags.complex_miller)
            gpuErr(cudaMallocManaged(&cp.cu_Fhkl2, db_cryst.FhklLinear.size()*sizeof(CUDAREAL)));

        gpuErr(cudaMallocManaged((void **)&cp.cu_dF_vecs, db_det.dF_vecs.size()*sizeof(VEC3)));
        gpuErr(cudaMallocManaged((void **)&cp.cu_dS_vecs, db_det.dF_vecs.size()*sizeof(VEC3)));

        //gettimeofday(&t3, 0));
        gpuErr(cudaMallocManaged(&cp.cu_floatimage, db_cu_flags.Npix_to_allocate*sizeof(CUDAREAL) ));
        if (db_flags.wavelength_img){
            gpuErr(cudaMallocManaged(&cp.cu_wavelenimage, 4*db_cu_flags.Npix_to_allocate*sizeof(CUDAREAL) ));
        }
        if (db_flags.refine_diffuse){
            gpuErr(cudaMallocManaged(&cp.cu_d_diffuse_gamma_images, db_cu_flags.Npix_to_allocate*3*sizeof(CUDAREAL)));
            gpuErr(cudaMallocManaged(&cp.cu_d_diffuse_sigma_images, db_cu_flags.Npix_to_allocate*3*sizeof(CUDAREAL)));
        }
        if (db_flags.refine_fcell){
            gpuErr(cudaMallocManaged(&cp.cu_d_fcell_images, db_cu_flags.Npix_to_allocate*1*sizeof(CUDAREAL)));
            gpuErr(cudaMallocManaged(&cp.cu_d2_fcell_images, db_cu_flags.Npix_to_allocate*1*sizeof(CUDAREAL)));
        }
        if (db_flags.refine_eta){
            gpuErr(cudaMallocManaged(&cp.cu_d_eta_images, db_cu_flags.Npix_to_allocate*3*sizeof(CUDAREAL)));
            gpuErr(cudaMallocManaged(&cp.cu_d2_eta_images, db_cu_flags.Npix_to_allocate*3*sizeof(CUDAREAL)));
        }
        if (std::count(db_flags.refine_Umat.begin(), db_flags.refine_Umat.end(), true) > 0){
            gpuErr(cudaMallocManaged(&cp.cu_d_Umat_images, db_cu_flags.Npix_to_allocate*3*sizeof(CUDAREAL) ));
            gpuErr(cudaMallocManaged(&cp.cu_d2_Umat_images, db_cu_flags.Npix_to_allocate*3*sizeof(CUDAREAL) ));
        }
        if (std::count(db_flags.refine_Ncells.begin(), db_flags.refine_Ncells.end(), true) > 0 || db_flags.refine_Ncells_def){
            gpuErr(cudaMallocManaged(&cp.cu_d_Ncells_images, db_cu_flags.Npix_to_allocate*6*sizeof(CUDAREAL)));
            gpuErr(cudaMallocManaged(&cp.cu_d2_Ncells_images, db_cu_flags.Npix_to_allocate*6*sizeof(CUDAREAL)));
        }
        if (std::count(db_flags.refine_panel_rot.begin(), db_flags.refine_panel_rot.end(), true) > 0)
            gpuErr(cudaMallocManaged(&cp.cu_d_panel_rot_images, db_cu_flags.Npix_to_allocate*3*sizeof(CUDAREAL)));
        if (std::count(db_flags.refine_panel_origin.begin(), db_flags.refine_panel_origin.end(), true) > 0)
            gpuErr(cudaMallocManaged(&cp.cu_d_panel_orig_images, db_cu_flags.Npix_to_allocate*3*sizeof(CUDAREAL)));
        if (std::count(db_flags.refine_lambda.begin(), db_flags.refine_lambda.end(), true) > 0)
            gpuErr(cudaMallocManaged(&cp.cu_d_lambda_images, db_cu_flags.Npix_to_allocate*2*sizeof(CUDAREAL)));
        if (std::count(db_flags.refine_Bmat.begin(), db_flags.refine_Bmat.end(), true) > 0){
            gpuErr(cudaMallocManaged(&cp.cu_d_Bmat_images, db_cu_flags.Npix_to_allocate*6*sizeof(CUDAREAL)));
            gpuErr(cudaMallocManaged(&cp.cu_d2_Bmat_images, db_cu_flags.Npix_to_allocate*6*sizeof(CUDAREAL)));
        }
        if (db_flags.refine_fp_fdp)
            gpuErr(cudaMallocManaged(&cp.cu_d_fp_fdp_images, db_cu_flags.Npix_to_allocate*2*sizeof(CUDAREAL)));
        if(db_cryst.nominal_hkl.size() >0)
            gpuErr(cudaMallocManaged(&cp.cu_nominal_hkl, db_cu_flags.Npix_to_allocate*3*sizeof(int)));

        //gettimeofday(&t4, 0);
        //time = (1000000.0*(t4.tv_sec-t3.tv_sec) + t4.tv_usec-t3.tv_usec)/1000.0;
        //printf("TIME SPENT ALLOCATING (IMAGES ONLY):  %3.10f ms \n", time);
        gpuErr(cudaMallocManaged(&cp.cu_panels_fasts_slows, db_cu_flags.Npix_to_allocate*3*sizeof(panels_fasts_slows[0])));
        cp.npix_allocated = db_cu_flags.Npix_to_allocate;
    } // END of allocation

    bool ALLOC = !cp.device_is_allocated; // shortcut variable

    gettimeofday(&t2, 0);
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    if (TIMERS.recording) TIMERS.cuda_alloc+= time;
    if(db_flags.verbose>1)
        printf("TIME SPENT ALLOCATING (TOTAL):  %3.10f ms \n", time);

    //ALLOC = false;
//  BEGIN COPYING DATA
    gettimeofday(&t1, 0);
    bool FORCE_COPY=true;

//  END step position
    if (db_flags.Fhkl_gradient_mode){
        for (int i=0; i < Npix_to_model; i++){
            cp.data_residual[i] = d_image.residual[i];
            cp.data_variance[i] = d_image.variance[i];
            cp.data_trusted[i] = d_image.trusted[i];
            cp.data_freq[i] = d_image.freq[i];
        }
    }

    if (db_flags.Fhkl_have_scale_factors && ALLOC){
        for (int i=0; i < db_cryst.FhklLinear_ASUid.size(); i++){
            cp.FhklLinear_ASUid[i] = db_cryst.FhklLinear_ASUid[i];
        }
    }

    if (db_flags.Fhkl_have_scale_factors){
        //SCITBX_ASSERT(db_beam.number_of_sources == db_beam.Fhkl_channels.size());
        for (int i=0; i < db_beam.number_of_sources; i++)
            cp.Fhkl_channels[i] = db_beam.Fhkl_channels[i];

        for (int i=0; i < d_image.Fhkl_scale.size(); i++){
            cp.Fhkl_scale[i] = d_image.Fhkl_scale[i];
            if (db_flags.Fhkl_gradient_mode){
                cp.Fhkl_scale_deriv[i] = 0;
            }
        }
    }

//  BEGIN sources
    if (db_cu_flags.update_sources || ALLOC || FORCE_COPY){
        for (int i=0; i< db_beam.number_of_sources; i++){
            VEC3 incident(db_beam.source_X[i], db_beam.source_Y[i], db_beam.source_Z[i]);
            incident /= incident.norm();
            cp.cu_source_X[i] = incident[0];
            cp.cu_source_Y[i] = incident[1];
            cp.cu_source_Z[i] = incident[2];
            cp.cu_source_I[i] = db_beam.source_I[i];
            cp.cu_source_lambda[i] = db_beam.source_lambda[i];
        }
        if(db_flags.verbose>1 )
          printf("H2D sources\n");
    }
//  END sources


//  UMATS
    if (db_cu_flags.update_umats || ALLOC||FORCE_COPY){
        for (int i=0; i< db_cryst.UMATS.size(); i++)
            cp.cu_UMATS[i] = db_cryst.UMATS[i];
        for (int i=0; i < db_cryst.UMATS_RXYZ.size(); i++)
            cp.cu_UMATS_RXYZ[i] = db_cryst.UMATS_RXYZ[i];
        for (int i=0; i < db_cryst.UMATS_RXYZ_prime.size(); i++)
            cp.cu_UMATS_RXYZ_prime[i] = db_cryst.UMATS_RXYZ_prime[i];
        for (int i=0; i < db_cryst.UMATS_RXYZ_dbl_prime.size(); i++)
            cp.cu_UMATS_RXYZ_dbl_prime[i] = db_cryst.UMATS_RXYZ_dbl_prime[i];
        if(db_flags.verbose>1)
            printf("H2D Done copying Umats\n") ;
    }
//  END UMATS


    if (db_cu_flags.update_umats || ALLOC||FORCE_COPY){
        MAT3 Amat_init = db_cryst.eig_U*db_cryst.eig_B*1e10*(db_cryst.eig_O.transpose());
        for(int i_mos =0; i_mos< db_cryst.UMATS_RXYZ.size(); i_mos++){
            cp.cu_AMATS[i_mos] = (db_cryst.UMATS_RXYZ[i_mos]*Amat_init).transpose();
            }
        if(db_flags.verbose>1)
            printf("H2D Done copying Amats\n") ;
    }


//  BMATS
    if(db_cu_flags.update_dB_mats || ALLOC || FORCE_COPY){
        for (int i=0; i< db_cryst.dB_Mats.size(); i++)
            cp.cu_dB_Mats[i] = db_cryst.dB_Mats[i];
        for (int i=0; i< db_cryst.dB2_Mats.size(); i++)
            cp.cu_dB2_Mats[i] = db_cryst.dB2_Mats[i];
        if(db_flags.verbose>1)
            printf("H2D Done copying dB_Mats\n") ;
    }
//  END BMATS


//  ROT MATS
    if(db_cu_flags.update_rotmats || ALLOC || FORCE_COPY){
        for (int i=0; i<db_cryst.RotMats.size(); i++)
            cp.cu_RotMats[i] = db_cryst.RotMats[i];
        for (int i=0; i<db_cryst.dRotMats.size(); i++)
            cp.cu_dRotMats[i] = db_cryst.dRotMats[i];
        for (int i=0; i<db_cryst.d2RotMats.size(); i++)
            cp.cu_d2RotMats[i] = db_cryst.d2RotMats[i];
        if (db_flags.verbose>1)
          printf("H2D Done copying rotmats\n");
    }
//  END ROT MATS

//  DETECTOR VECTORS
    if (db_cu_flags.update_detector || ALLOC || FORCE_COPY){
        for (int i=0; i<db_det.fdet_vectors.size(); i++){
            cp.cu_fdet_vectors[i] = db_det.fdet_vectors[i];
            cp.cu_sdet_vectors[i] = db_det.sdet_vectors[i];
            cp.cu_odet_vectors[i] = db_det.odet_vectors[i];
            cp.cu_pix0_vectors[i] = db_det.pix0_vectors[i];
        }
        for(int i=0; i < db_det.close_distances.size();i++){
            cp.cu_close_distances[i] = db_det.close_distances[i];
        }
        if (db_flags.verbose>1)
          printf("H2D Done copying detector vectors\n");
    }
//  END  DETECTOR VECTORS

    if ( ALLOC || FORCE_COPY){
      for(int i=0; i< db_cryst.nominal_hkl.size(); i++){
        cp.cu_nominal_hkl[i] = db_cryst.nominal_hkl[i];
      }
      for (int i=0; i< db_cryst.atom_data.size(); i++){
        cp.cu_atom_data[i] = db_cryst.atom_data[i];
      }
      if (db_flags.verbose>1)
        printf("H2D Done copying atom data\n");
      for(int i=0; i< db_cryst.fpfdp.size(); i++){
        cp.cu_fpfdp[i] = db_cryst.fpfdp[i];
      }
      for(int i=0; i< db_cryst.fpfdp_derivs.size(); i++){
        cp.cu_fpfdp_derivs[i] = db_cryst.fpfdp_derivs[i];
      }
      if (db_flags.verbose>1)
        printf("H2D Done copying fprime and fdblprime\n");
    }


//  BEGIN REFINEMENT FLAGS
    if (db_cu_flags.update_refine_flags || ALLOC || FORCE_COPY){
        for (int i=0; i<3; i++){
            cp.cu_refine_Umat[i] = db_flags.refine_Umat[i];
            cp.cu_refine_Ncells[i] = db_flags.refine_Ncells[i];
            cp.cu_refine_panel_origin[i] = db_flags.refine_panel_origin[i];
            cp.cu_refine_panel_rot[i] = db_flags.refine_panel_rot[i];
        }
        for(int i=0; i<2; i++)
            cp.cu_refine_lambda[i] = db_flags.refine_lambda[i];
        for(int i=0; i<6; i++)
            cp.cu_refine_Bmat[i] = db_flags.refine_Bmat[i];
        if (db_flags.verbose>1)
          printf("H2D Done copying refinement flags\n");
    }
//  END REFINEMENT FLAGS


//  BEGIN Fhkl
    if (db_cu_flags.update_Fhkl || ALLOC || FORCE_COPY){
        for(int i=0; i < db_cryst.FhklLinear.size(); i++){
          cp.cu_Fhkl[i] = db_cryst.FhklLinear[i];
          if (db_flags.complex_miller)
              cp.cu_Fhkl2[i] = db_cryst.Fhkl2Linear[i];
        }
        if (db_flags.verbose>1)
            printf("H2D Done copying step Fhkl\n");
    }
//  END Fhkl

//  BEGIN panel derivative vecs
    if(db_cu_flags.update_panel_deriv_vecs || ALLOC || FORCE_COPY){
        for (int i=0; i<db_det.dF_vecs.size(); i++){
            cp.cu_dF_vecs[i] = db_det.dF_vecs[i];
            cp.cu_dS_vecs[i] = db_det.dS_vecs[i];
        }
        if (db_flags.verbose>1)
            printf("H2D Done copying step panel derivative vectors\n");
    }
//  END panel derivative vecs

//  BEGIN panels fasts slows
    if (db_cu_flags.update_panels_fasts_slows || ALLOC || FORCE_COPY){
        for (int i=0; i< panels_fasts_slows.size(); i++)
            cp.cu_panels_fasts_slows[i] = panels_fasts_slows[i];
        if (db_flags.verbose>1)
            printf("H2D Done copying panels_fasts_slows\n");
    }
//  END panels fasts slows


    gettimeofday(&t2, 0);
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    if (TIMERS.recording) TIMERS.cuda_copy_to_dev += time;
    if(db_flags.verbose>1)
        printf("TIME SPENT COPYING DATA HOST->DEV:  %3.10f ms \n", time);

    cp.device_is_allocated = true;
    error_msg(cudaGetLastError(), "after copy to device");

    gettimeofday(&t1, 0);

    int Npanels = db_det.fdet_vectors.size()/3;
    int num_atoms = db_cryst.atom_data.size()/5;
    if (db_cryst.fpfdp.size() == 0){ // note cannot use atom data if fpfdp is 0, make this cleaner
        num_atoms=0;
    }

    if (db_flags.use_diffuse) {
        if (db_cryst.laue_group_num < 1 || db_cryst.laue_group_num >14 ){
            throw std::string("Laue group number not in range 1-14");
        }
    }
    //int sm_size = number_of_sources*5*sizeof(CUDAREAL);
    //gpu_sum_over_steps<<<numblocks, blocksize, sm_size >>>(
    bool aniso_eta = db_cryst.UMATS_RXYZ.size() != db_cryst.UMATS_RXYZ_prime.size();
    bool use_nominal_hkl = !db_cryst.nominal_hkl.empty();
    gpu_sum_over_steps<<<numblocks, blocksize>>>(
        Npix_to_model, cp.cu_panels_fasts_slows,
        cp.cu_floatimage,
        cp.cu_wavelenimage,
        cp.cu_d_Umat_images, cp.cu_d2_Umat_images,
        cp.cu_d_Bmat_images, cp.cu_d2_Bmat_images,
        cp.cu_d_Ncells_images, cp.cu_d2_Ncells_images,
        cp.cu_d_fcell_images, cp.cu_d2_fcell_images,
        cp.cu_d_eta_images, cp.cu_d2_eta_images,
        cp.cu_d_lambda_images, cp.cu_d2_lambda_images,
        cp.cu_d_panel_rot_images, cp.cu_d2_panel_rot_images,
        cp.cu_d_panel_orig_images, cp.cu_d2_panel_orig_images,
        cp.cu_d_fp_fdp_images,
        db_steps.Nsteps, db_flags.printout_fpixel, db_flags.printout_spixel, db_flags.printout, db_cryst.default_F,
        db_det.oversample,  db_flags.oversample_omega, db_det.subpixel_size, db_det.pixel_size,
        db_det.detector_thickstep, db_det.detector_thick, cp.cu_close_distances, db_det.detector_attnlen,
        db_det.detector_thicksteps, db_beam.number_of_sources, db_cryst.phisteps, db_cryst.UMATS.size(),
        db_flags.use_lambda_coefficients, db_beam.lambda0, db_beam.lambda1,
        db_cryst.eig_U, db_cryst.eig_O, db_cryst.eig_B, db_cryst.RXYZ,
        cp.cu_dF_vecs,
        cp.cu_dS_vecs,
        cp.cu_UMATS_RXYZ,
        cp.cu_UMATS_RXYZ_prime,
        cp.cu_UMATS_RXYZ_dbl_prime,
        cp.cu_RotMats,
        cp.cu_dRotMats,
        cp.cu_d2RotMats,
        cp.cu_UMATS,
        cp.cu_dB_Mats,
        cp.cu_dB2_Mats,
        cp.cu_AMATS,
        cp.cu_source_X, cp.cu_source_Y, cp.cu_source_Z, cp.cu_source_lambda, cp.cu_source_I,
        db_beam.kahn_factor,
        db_cryst.Na, db_cryst.Nb, db_cryst.Nc,
        db_cryst.Nd, db_cryst.Ne, db_cryst.Nf,
        db_cryst.phi0, db_cryst.phistep,
        db_cryst.spindle_vec, db_beam.polarization_axis,
        db_cryst.h_range, db_cryst.k_range, db_cryst.l_range,
        db_cryst.h_max, db_cryst.h_min, db_cryst.k_max, db_cryst.k_min, db_cryst.l_max, db_cryst.l_min, db_cryst.dmin,
        db_cryst.fudge, db_flags.complex_miller, db_flags.verbose, db_flags.only_save_omega_kahn,
        db_flags.isotropic_ncells, db_flags.compute_curvatures,
        cp.cu_Fhkl, cp.cu_Fhkl2,
        cp.cu_refine_Bmat, cp.cu_refine_Ncells, db_flags.refine_Ncells_def, cp.cu_refine_panel_origin, cp.cu_refine_panel_rot,
        db_flags.refine_fcell, cp.cu_refine_lambda, db_flags.refine_eta, cp.cu_refine_Umat,
        cp.cu_fdet_vectors, cp.cu_sdet_vectors,
        cp.cu_odet_vectors, cp.cu_pix0_vectors,
        db_flags.nopolar, db_flags.point_pixel, db_beam.fluence, db_cryst.r_e_sqr, db_cryst.spot_scale, Npanels, aniso_eta, db_flags.no_Nabc_scale,
        cp.cu_fpfdp,  cp.cu_fpfdp_derivs, cp.cu_atom_data, num_atoms,
        db_flags.refine_fp_fdp, cp.cu_nominal_hkl, use_nominal_hkl,
        db_cryst.anisoU, db_cryst.anisoG, db_cryst.rotate_principal_axes,
        db_flags.use_diffuse, cp.cu_d_diffuse_gamma_images, cp.cu_d_diffuse_sigma_images,
        db_flags.refine_diffuse, db_flags.gamma_miller_units, db_flags.refine_Icell,
        db_flags.wavelength_img, db_cryst.laue_group_num, db_cryst.stencil_size,
        db_flags.Fhkl_gradient_mode, db_flags.Fhkl_errors_mode, db_flags.using_trusted_mask, db_beam.Fhkl_channels.empty(), db_flags.Fhkl_have_scale_factors,
        db_cryst.Num_ASU,
        cp.data_residual, cp.data_variance,
        cp.data_freq, cp.data_trusted,
        cp.FhklLinear_ASUid,
        cp.Fhkl_channels,
        cp.Fhkl_scale, cp.Fhkl_scale_deriv,
        db_cryst.xtal_shape==GAUSS_STAR,
        db_cryst.xtal_shape==SQUARE
        );

    error_msg(cudaGetLastError(), "after kernel call");

    cudaDeviceSynchronize();
    error_msg(cudaGetLastError(), "after kernel completion");

    if(db_flags.verbose>1)
        printf("KERNEL_COMPLETE gpu_sum_over_steps\n");
    gettimeofday(&t2, 0);
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    if (TIMERS.recording) TIMERS.cuda_kernel += time;
    if(db_flags.verbose>1)
        printf("TIME SPENT(KERNEL):  %3.10f ms \n", time);

    gettimeofday(&t1, 0);
//  COPY BACK FROM DEVICE
    for (int i=0; i< Npix_to_model; i++){
        floatimage[i] = cp.cu_floatimage[i];
    }
    if(db_flags.wavelength_img){
        for (int i=0; i< 4*Npix_to_model; i++){
            d_image.wavelength[i] = cp.cu_wavelenimage[i];
        }
    }
    if (db_flags.refine_fcell){
        for (int i=0; i<Npix_to_model; i++){
            d_image.fcell[i] = cp.cu_d_fcell_images[i];
            d2_image.fcell[i] = cp.cu_d2_fcell_images[i];
        }
    }
    if (db_flags.Fhkl_gradient_mode){
        if (db_flags.Fhkl_errors_mode){
            for (int i=0; i < d_image.Fhkl_hessian.size(); i++)
                d_image.Fhkl_hessian[i]= cp.Fhkl_scale_deriv[i];
        }
        else{
            for (int i=0; i < d_image.Fhkl_scale_deriv.size(); i++)
                d_image.Fhkl_scale_deriv[i]= cp.Fhkl_scale_deriv[i];
        }
    }
    if (std::count(db_flags.refine_Umat.begin(), db_flags.refine_Umat.end(), true) > 0){
        for (int i=0; i<3*Npix_to_model; i++){
            d_image.Umat[i] = cp.cu_d_Umat_images[i];
            d2_image.Umat[i] = cp.cu_d2_Umat_images[i];
        }
    }
    if (std::count(db_flags.refine_panel_rot.begin(), db_flags.refine_panel_rot.end(), true) > 0){
        for (int i=0; i<3*Npix_to_model; i++)
            d_image.panel_rot[i] = cp.cu_d_panel_rot_images[i];
    }
    if (std::count(db_flags.refine_panel_origin.begin(), db_flags.refine_panel_origin.end(), true) > 0){
        for (int i=0; i<3*Npix_to_model; i++)
            d_image.panel_orig[i] = cp.cu_d_panel_orig_images[i];
    }
    if (db_flags.refine_eta){
        for (int i=0; i<3*Npix_to_model; i++){
            d_image.eta[i] = cp.cu_d_eta_images[i];
            d2_image.eta[i] = cp.cu_d2_eta_images[i];
        }
    }
    if (std::count(db_flags.refine_Ncells.begin(), db_flags.refine_Ncells.end(), true) > 0 || db_flags.refine_Ncells_def){
        for(int i=0; i<6*Npix_to_model; i++){
            d_image.Ncells[i] = cp.cu_d_Ncells_images[i];
            d2_image.Ncells[i] = cp.cu_d2_Ncells_images[i];
        }
    }
    if (db_flags.refine_diffuse){
        for(int i=0; i<3*Npix_to_model; i++){
            d_image.diffuse_gamma[i] = cp.cu_d_diffuse_gamma_images[i];
            d_image.diffuse_sigma[i] = cp.cu_d_diffuse_sigma_images[i];
        }
    }
    if (std::count(db_flags.refine_Bmat.begin(), db_flags.refine_Bmat.end(), true) > 0){
        for(int i=0; i<6*Npix_to_model; i++){
            d_image.Bmat[i] = cp.cu_d_Bmat_images[i];
            d2_image.Bmat[i] = cp.cu_d2_Bmat_images[i];
        }
    }
    if (std::count(db_flags.refine_lambda.begin(), db_flags.refine_lambda.end(), true) > 0){
        for(int i=0; i<2*Npix_to_model; i++)
            d_image.lambda[i] = cp.cu_d_lambda_images[i];
    }

    if (db_flags.refine_fp_fdp){
        for (int i=0; i< 2*Npix_to_model; i++)
            d_image.fp_fdp[i] = cp.cu_d_fp_fdp_images[i];
    }

    gettimeofday(&t2, 0);
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    if (TIMERS.recording) TIMERS.cuda_copy_from_dev += time;
    if(db_flags.verbose>1)
        printf("TIME SPENT COPYING BACK :  %3.10f ms \n", time);
    error_msg(cudaGetLastError(), "After copy to host");


}


void freedom(diffBragg_cudaPointers& cp){

    if (cp.device_is_allocated){
        gpuErr(cudaFree( cp.cu_floatimage));
        gpuErr(cudaFree( cp.cu_wavelenimage));
        gpuErr(cudaFree( cp.cu_d_Umat_images));
        gpuErr(cudaFree( cp.cu_d_Bmat_images));
        gpuErr(cudaFree( cp.cu_d_Ncells_images));
        gpuErr(cudaFree( cp.cu_d_diffuse_gamma_images));
        gpuErr(cudaFree( cp.cu_d_diffuse_sigma_images));
        gpuErr(cudaFree( cp.cu_d2_Umat_images));
        gpuErr(cudaFree( cp.cu_d2_Bmat_images));
        gpuErr(cudaFree( cp.cu_d2_Ncells_images));
        gpuErr(cudaFree( cp.cu_d_eta_images));
        gpuErr(cudaFree( cp.cu_d2_eta_images));
        gpuErr(cudaFree( cp.cu_d_fcell_images));
        gpuErr(cudaFree( cp.cu_d2_fcell_images));
        gpuErr(cudaFree( cp.cu_d_lambda_images));
        gpuErr(cudaFree( cp.cu_d_panel_rot_images));
        gpuErr(cudaFree( cp.cu_d_panel_orig_images));
        gpuErr(cudaFree( cp.cu_d_sausage_XYZ_scale_images));
        gpuErr(cudaFree( cp.cu_d_fp_fdp_images));

        gpuErr(cudaFree(cp.cu_Fhkl));
        gpuErr(cudaFree(cp.cu_Fhkl2));

        gpuErr(cudaFree(cp.cu_fdet_vectors));
        gpuErr(cudaFree(cp.cu_sdet_vectors));
        gpuErr(cudaFree(cp.cu_odet_vectors));
        gpuErr(cudaFree(cp.cu_pix0_vectors));
        gpuErr(cudaFree(cp.cu_close_distances));
        gpuErr(cudaFree(cp.cu_nominal_hkl));
        gpuErr(cudaFree(cp.cu_atom_data));
        gpuErr(cudaFree(cp.cu_fpfdp));
        gpuErr(cudaFree(cp.cu_fpfdp_derivs));

        gpuErr(cudaFree(cp.Fhkl_channels));
        gpuErr(cudaFree(cp.cu_source_X));
        gpuErr(cudaFree(cp.cu_source_Y));
        gpuErr(cudaFree(cp.cu_source_Z));
        gpuErr(cudaFree(cp.cu_source_I));
        gpuErr(cudaFree(cp.cu_source_lambda));

        gpuErr(cudaFree(cp.cu_UMATS));
        gpuErr(cudaFree(cp.cu_UMATS_RXYZ));
        gpuErr(cudaFree(cp.cu_AMATS));
        gpuErr(cudaFree(cp.cu_UMATS_RXYZ_prime));
        gpuErr(cudaFree(cp.cu_UMATS_RXYZ_dbl_prime));
        gpuErr(cudaFree(cp.cu_RotMats));
        gpuErr(cudaFree(cp.cu_dRotMats));
        gpuErr(cudaFree(cp.cu_d2RotMats));
        gpuErr(cudaFree(cp.cu_dB_Mats));
        gpuErr(cudaFree(cp.cu_dB2_Mats));
        //gpuErr(cudaFree(cp.cu_sausages_RXYZ));
        //gpuErr(cudaFree(cp.cu_d_sausages_RXYZ));
        //gpuErr(cudaFree(cp.cu_sausages_U));
        //gpuErr(cudaFree(cp.cu_sausages_scale));

        gpuErr(cudaFree(cp.cu_dF_vecs));
        gpuErr(cudaFree(cp.cu_dS_vecs));

        gpuErr(cudaFree(cp.cu_refine_Bmat));
        gpuErr(cudaFree(cp.cu_refine_Umat));
        gpuErr(cudaFree(cp.cu_refine_Ncells));
        gpuErr(cudaFree(cp.cu_refine_lambda));
        gpuErr(cudaFree(cp.cu_refine_panel_origin));
        gpuErr(cudaFree(cp.cu_refine_panel_rot));

        gpuErr(cudaFree(cp.cu_panels_fasts_slows));

        cp.device_is_allocated = false;
        cp.npix_allocated = 0;
    }

    if (cp.Fhkl_grad_arrays_allocated){
        gpuErr(cudaFree(cp.data_trusted));
        gpuErr(cudaFree(cp.data_freq));
        gpuErr(cudaFree(cp.data_residual));
        gpuErr(cudaFree(cp.data_variance));
        gpuErr(cudaFree(cp.FhklLinear_ASUid));
        gpuErr(cudaFree(cp.Fhkl_scale));
        gpuErr(cudaFree(cp.Fhkl_scale_deriv));
        cp.Fhkl_grad_arrays_allocated=false;
    }
}
