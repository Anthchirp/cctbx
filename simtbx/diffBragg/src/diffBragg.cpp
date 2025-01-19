#include <sys/time.h>
#include <simtbx/diffBragg/src/diffBragg.h>
#include <assert.h>
#include <stdbool.h>
#include <cctbx/sgtbx/space_group.h>
#include <cctbx/miller/asu.h>
#include<unordered_map>
#include <cctbx/sgtbx/reciprocal_space_asu.h>
#include <boost/python/numpy.hpp>

namespace np=boost::python::numpy;

#pragma omp declare reduction(vec_double_plus : std::vector<double> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
                    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

namespace simtbx {
namespace nanoBragg {

// BEGIN derivative manager
derivative_manager::derivative_manager(){}

void derivative_manager::initialize(int Npix_total, bool curvatures)
{
    //raw_pixels = af::flex_double(af::flex_grid<>(sdim,fdim));
    raw_pixels = af::flex_double(Npix_total);
    dI=0;

    // for second derivatives
    dI2=0;
    //if (curvatures)
    //raw_pixels2 = af::flex_double(af::flex_grid<>(sdim,fdim));
    raw_pixels2 = af::flex_double(Npix_total);
}

void derivative_manager::increment_image(int idx, double value, double value2, bool curvatures){
    double* floatimage = raw_pixels.begin();
    floatimage[idx] = value;

    // increment second derivatives
    if (curvatures){
        double* floatimage2 = raw_pixels2.begin();
        floatimage2[idx] = value2;
    }
}
// END derivative manager

//BEGIN origin manager
origin_manager::origin_manager(){
    FF=0;
    FdF=0;
    dFdF=0;
    FdF2=0;
    }

// BEGIN Ncells_abc manager
Ncells_manager::Ncells_manager(){}

void Ncells_manager::increment(double dI_increment, double dI2_increment){
    dI += dI_increment;
    dI2 += dI2_increment;
};

//END Ncells_abc manager

// Begin Fcell manager
Fcell_manager::Fcell_manager(){}

void Fcell_manager::increment( double value, double value2)
{
    dI += value;
    dI2 += value2;
};
// END Fcell manager

// Begin eta manager
eta_manager::eta_manager(){}

void eta_manager::increment( double value, double value2)
{
    dI += value;
    dI2 += value2;
};
// END eta manager

// begin lambda_manager
lambda_manager::lambda_manager(){}

void lambda_manager::increment( double value, double value2)
{
    dI += value;
    dI2 += value2;
};
//END lambda manager

// begin panel_manager
panel_manager::panel_manager(){}

//END panel_manager

//BEGIN unit cell manager
ucell_manager::ucell_manager(){}

void ucell_manager::increment(double value, double value2)
{
  dI += value;
  dI2 += value2;
};

// BEGIN rotation manager begin
rot_manager::rot_manager(){}

void rot_manager::set_R(){assert (false);}

void rot_manager::increment(double value, double value2)
{
  dI += value;
  dI2 += value2;
}

rotX_manager::rotX_manager(){
    value = 0;
    set_R();
}
rotZ_manager::rotZ_manager(){
    value = 0;
    set_R();
}
rotY_manager::rotY_manager(){
    value = 0;
    set_R();
}

void rotX_manager::set_R(){
    R << 1,           0,           0,
          0,  cos(value), sin(value),
          0, -sin(value), cos(value);

    dR << 0,           0,           0,
             0,  -sin(value), cos(value),
             0, -cos(value), -sin(value);

    dR2 << 0,           0,           0,
              0,  -cos(value), -sin(value),
              0,   sin(value), -cos(value);


}
void rotY_manager::set_R(){
    R << cos(value),0, -sin(value),
             0,         1,             0,
            sin(value), 0, cos(value);

    dR << -sin(value),0, -cos(value),
                0,          0,             0,
                cos(value), 0, -sin(value);

    dR2 << -cos(value),0, sin(value),
              0,          0,          0,
             -sin(value), 0, -cos(value);
}
void rotZ_manager::set_R(){
    R  << cos(value),  sin(value), 0,
              -sin(value), cos(value), 0,
                         0,           0, 1;

    dR << -sin(value),  cos(value), 0,
               -cos(value), -sin(value), 0,
                           0,           0, 0;

    dR2 << -cos(value), -sin(value), 0,
               sin(value), -cos(value), 0,
                        0,           0, 0;
}
// END rot manager

// BEGIN diffBragg
diffBragg::diffBragg(const dxtbx::model::Detector& detector, const dxtbx::model::Beam& beam,
            int verbose):
    nanoBragg(detector, beam, verbose, 0)
    { // diffBragg init
    int Npanels = detector.size();
    Npix_total = Npanels * detector[0].get_image_size()[0]* detector[0].get_image_size()[1];
    db_det.fdet_vectors.clear();
    db_det.sdet_vectors.clear();
    db_det.odet_vectors.clear();
    db_det.pix0_vectors.clear();


    no_Nabc_scale = false;

    db_det.dF_vecs.clear();
    db_det.dS_vecs.clear();
    for (int ii=0; ii < Npanels*3; ii++){
        db_det.fdet_vectors.push_back(0);
        db_det.sdet_vectors.push_back(0);
        db_det.odet_vectors.push_back(0);
        db_det.pix0_vectors.push_back(0);

        Eigen::Vector3d vec(0,0,0);
        db_det.dF_vecs.push_back(vec);
        db_det.dS_vecs.push_back(vec);
    }

    EYE <<  1,0,0,
            0,1,0,
            0,0,1;
    db_cryst.eig_O << 1,0,0,
               0,1,0,
               0,0,1;
    psi = 0;

    db_cryst.RotMats.push_back(EYE);
    db_cryst.RotMats.push_back(EYE);
    db_cryst.RotMats.push_back(EYE);

    db_cryst.dRotMats.push_back(EYE);
    db_cryst.dRotMats.push_back(EYE);
    db_cryst.dRotMats.push_back(EYE);

    db_cryst.d2RotMats.push_back(EYE);
    db_cryst.d2RotMats.push_back(EYE);
    db_cryst.d2RotMats.push_back(EYE);


    boost::shared_ptr<rot_manager> rotX = boost::shared_ptr<rot_manager>(new rotX_manager());
    boost::shared_ptr<rot_manager> rotY = boost::shared_ptr<rot_manager>(new rotY_manager());
    boost::shared_ptr<rot_manager> rotZ = boost::shared_ptr<rot_manager>(new rotZ_manager());

    boost::shared_ptr<ucell_manager> uc1 = boost::shared_ptr<ucell_manager>(new ucell_manager());
    boost::shared_ptr<ucell_manager> uc2 = boost::shared_ptr<ucell_manager>(new ucell_manager());
    boost::shared_ptr<ucell_manager> uc3 = boost::shared_ptr<ucell_manager>(new ucell_manager());
    boost::shared_ptr<ucell_manager> uc4 = boost::shared_ptr<ucell_manager>(new ucell_manager());
    boost::shared_ptr<ucell_manager> uc5 = boost::shared_ptr<ucell_manager>(new ucell_manager());
    boost::shared_ptr<ucell_manager> uc6 = boost::shared_ptr<ucell_manager>(new ucell_manager());

    boost::shared_ptr<Ncells_manager> nc1 = boost::shared_ptr<Ncells_manager>(new Ncells_manager());
    boost::shared_ptr<Ncells_manager> nc2 = boost::shared_ptr<Ncells_manager>(new Ncells_manager());
    boost::shared_ptr<Ncells_manager> nc3 = boost::shared_ptr<Ncells_manager>(new Ncells_manager());
    boost::shared_ptr<Ncells_manager> nc4 = boost::shared_ptr<Ncells_manager>(new Ncells_manager());
    boost::shared_ptr<Ncells_manager> nc5 = boost::shared_ptr<Ncells_manager>(new Ncells_manager());
    boost::shared_ptr<Ncells_manager> nc6 = boost::shared_ptr<Ncells_manager>(new Ncells_manager());

    boost::shared_ptr<lambda_manager> lam1 = boost::shared_ptr<lambda_manager>(new lambda_manager());
    boost::shared_ptr<lambda_manager> lam2 = boost::shared_ptr<lambda_manager>(new lambda_manager());

    boost::shared_ptr<panel_manager> orig0 = boost::shared_ptr<panel_manager>(new panel_manager());
    boost::shared_ptr<panel_manager> origX = boost::shared_ptr<panel_manager>(new panel_manager());
    boost::shared_ptr<panel_manager> origY = boost::shared_ptr<panel_manager>(new panel_manager());
    //boost::shared_ptr<origin_manager> orig0 = boost::shared_ptr<origin_manager>(new origin_manager());

    boost::shared_ptr<derivative_manager> fcell_man0 = boost::shared_ptr<derivative_manager>(new Fcell_manager());
    boost::shared_ptr<derivative_manager> fcell_man1 = boost::shared_ptr<derivative_manager>(new Fcell_manager());
    boost::shared_ptr<derivative_manager> fcell_man2 = boost::shared_ptr<derivative_manager>(new Fcell_manager());
    fcell_man0->refine_me=false;
    fcell_man1->refine_me=false;
    fcell_man2->refine_me=false;

    fcell_managers.push_back(fcell_man0);
    fcell_managers.push_back(fcell_man1);
    fcell_managers.push_back(fcell_man2);
    //fcell_man = boost::shared_ptr<Fcell_manager>(new Fcell_manager());
    //fcell_man->refine_me = false;
    track_Fhkl=false;
    db_cryst.nominal_hkl.clear();

    boost::shared_ptr<eta_manager> eta0 = boost::shared_ptr<eta_manager>(new eta_manager());
    boost::shared_ptr<eta_manager> eta1 = boost::shared_ptr<eta_manager>(new eta_manager());
    boost::shared_ptr<eta_manager> eta2 = boost::shared_ptr<eta_manager>(new eta_manager());

    eta0->refine_me = false;
    eta1->refine_me = false;
    eta2->refine_me = false;
    mosaic_umats_prime = NULL;
    mosaic_umats_dbl_prime = NULL;
    eta_managers.push_back(eta0);
    eta_managers.push_back(eta1);
    eta_managers.push_back(eta2);

    panel_rot_man = boost::shared_ptr<panel_manager>(new panel_manager());
    panel_rot_man->refine_me = false;
    panel_rot_man->F_cross_dS << 0,0,0;
    panel_rot_man->dF_cross_S << 0,0,0;
    panel_rot_manF = boost::shared_ptr<panel_manager>(new panel_manager());
    panel_rot_manF->refine_me = false;
    panel_rot_manF->F_cross_dS <<0,0,0;
    panel_rot_manF->dF_cross_S <<0,0,0;

    panel_rot_manS = boost::shared_ptr<panel_manager>(new panel_manager());
    panel_rot_manS->refine_me = false;
    panel_rot_manS->F_cross_dS <<0,0,0;
    panel_rot_manS->dF_cross_S <<0,0,0;


    rotX->refine_me = false;
    rotY->refine_me = false;
    rotZ->refine_me = false;
    uc1->refine_me = false;
    uc2->refine_me = false;
    uc3->refine_me = false;
    uc4->refine_me = false;
    uc5->refine_me = false;
    uc6->refine_me = false;

    nc1->refine_me = false;
    nc2->refine_me = false;
    nc3->refine_me = false;
    nc4->refine_me = false;
    nc5->refine_me = false;
    nc6->refine_me = false;

    orig0->refine_me = false;
    orig0->dk << 0,0,1;

    origX->refine_me = false;
    origX->dk <<1,0,0;

    origY->refine_me = false;
    origY->dk <<0,1,0;

    lam1->refine_me= false;
    lam2->refine_me= false;

    lambda_managers.push_back(lam1);
    lambda_managers.push_back(lam2);

    rot_managers.push_back(rotX);
    rot_managers.push_back(rotY);
    rot_managers.push_back(rotZ);

    ucell_managers.push_back(uc1);
    ucell_managers.push_back(uc2);
    ucell_managers.push_back(uc3);
    ucell_managers.push_back(uc4);
    ucell_managers.push_back(uc5);
    ucell_managers.push_back(uc6);

    Ncells_managers.push_back(nc1);
    Ncells_managers.push_back(nc2);
    Ncells_managers.push_back(nc3);
    Ncells_managers.push_back(nc4);
    Ncells_managers.push_back(nc5);
    Ncells_managers.push_back(nc6);
    NABC << 1,0,0,
            0,1,0,
            0,0,1;

    panels.push_back(panel_rot_man);
    panels.push_back(orig0);
    panels.push_back(origX);
    panels.push_back(origY);
    panels.push_back(panel_rot_manF);
    panels.push_back(panel_rot_manS);

    boost::shared_ptr<Fcell_manager> fpfdp1 = boost::shared_ptr<Fcell_manager>(new Fcell_manager());
    boost::shared_ptr<Fcell_manager> fpfdp2 = boost::shared_ptr<Fcell_manager>(new Fcell_manager());
    fpfdp1->refine_me=false;
    fpfdp2->refine_me=false;
    fp_fdp_managers.push_back(fpfdp1);
    fp_fdp_managers.push_back(fpfdp2);

    db_cryst.fpfdp.clear();

    O_reference <<0,0,0;

    update_oversample_during_refinement = true;
    oversample_omega = true;
    only_save_omega_kahn = false;
    compute_curvatures = false; // why was this True by default?
    isotropic_ncells = false;
    nmats=0;
    modeling_anisotropic_mosaic_spread = false;
    refine_Ncells_def = false;
    Nd = 0;
    Ne = 0;
    Nf = 0;

    db_cryst.anisoG << 50,0,0,
                       0,50,0,
                       0,0,50;
    db_cryst.anisoU << .16,0,0,
                       0,.16,0,
                       0,0,.16;

    db_cryst.rotate_principal_axes << 0.70710678,-0.70710678,0.,
                                      0.70710678, 0.70710678,0.,
                                      0.,0.,1.;

    lambda_managers[0]->value = 0;
    lambda_managers[1]->value = 1;
    use_lambda_coefficients = false;

    // set ucell gradients, Bmatrix is upper triangular in diffBragg?
    for (int i=0; i <6; i++){
        ucell_managers[i]->dB << 0,0,0,
                                 0,0,0,
                                 0,0,0;
        ucell_managers[i]->dB2 << 0,0,0,
                                  0,0,0,
                                  0,0,0;
    }

    init_raw_pixels_roi();
    initialize_managers();

    Fhkl2 = NULL;
    F_cell2 = 0;
    complex_miller = false;
    pythony_amplitudes2.clear();

    for(int pid=0; pid < Npanels; pid++){
        update_dxtbx_geoms(detector, beam, pid, 0,0,0,0,0,0,false);
    }

    set_close_distances();

    linearize_Fhkl(false);
    //sanity_check_linear_Fhkl();
    //
    last_kernel_on_GPU=boost::python::object(); // object() returns  None

}

void diffBragg::set_close_distances(){
    db_det.close_distances.clear();
    int Npanels = db_det.pix0_vectors.size() / 3;
    for (int ii=0; ii<Npanels; ii++){
        Eigen::Vector3d pix0(db_det.pix0_vectors[ii*3], db_det.pix0_vectors[ii*3+1], db_det.pix0_vectors[ii*3+2]) ;
        Eigen::Vector3d OO(db_det.odet_vectors[ii*3], db_det.odet_vectors[ii*3+1], db_det.odet_vectors[ii*3+2]) ;
        double close_dist = pix0.dot(OO);
        db_det.close_distances.push_back(close_dist);
        if (verbose) printf("Panel %d: close distance %f\n", ii, close_dist);
    }
}


af::flex_double get_panel_increment( double Iincrement, double omega_pixel,
    const Eigen::Ref<const Eigen::Matrix3d>& M,
    double pix2, const Eigen::Ref<const Eigen::Vector3d>& o, const Eigen::Ref<const Eigen::Vector3d>& k_diffracted,
    double per_k, double per_k3, double per_k5, const Eigen::Ref<const Eigen::Vector3d>& V,
    const Eigen::Ref<const Eigen::Vector3d>& _dk)
{
    double G = _dk.dot(k_diffracted);
    Eigen::Vector3d dk_hat = -per_k3*G*k_diffracted + per_k*_dk;
    double coef = (M*dk_hat).dot(V);
    double coef2 = -3*pix2*per_k5*G * (o.dot(k_diffracted));
    coef2 += pix2*per_k3*(o.dot(_dk));
    double value = coef*Iincrement + coef2*Iincrement/omega_pixel;
    af::flex_double values = af::flex_double(2,0);
    values[0] = value;
    values[1] = 0;
    return values;
};



/* BEGIN panel Rot XYZ */
void diffBragg::rotate_fs_ss_vecs_3D(double panel_rot_angO, double panel_rot_angF, double panel_rot_angS){

    Eigen::Vector3d fs_vec(fdet_vector[1], fdet_vector[2], fdet_vector[3]);
    Eigen::Vector3d ss_vec(sdet_vector[1], sdet_vector[2], sdet_vector[3]);
    Eigen::Vector3d origin_vec(pix0_vector[1], pix0_vector[2], pix0_vector[3]);
    Eigen::Vector3d origin_diff_vec = origin_vec - O_reference; // difference between origin and a reference vector, this is the vector we will rotate

    Eigen::Matrix3d RO, RF, RS;
    double xx,yy,zz;
    xx = odet_vector[1];
    yy = odet_vector[2];
    zz = odet_vector[3];
    RO<< 0,-zz,  yy,
        zz,  0, -xx,
       -yy, xx,   0;
    Eigen::Matrix3d RO2 = RO*RO;

    xx = fdet_vector[1];
    yy = fdet_vector[2];
    zz = fdet_vector[3];
    RF<< 0,-zz,  yy,
        zz,  0, -xx,
       -yy, xx,   0;

    Eigen::Matrix3d RF2 = RF*RF;

    xx = sdet_vector[1];
    yy = sdet_vector[2];
    zz = sdet_vector[3];
    RS<< 0,-zz,  yy,
        zz,  0, -xx,
       -yy, xx,   0;
    Eigen::Matrix3d RS2 = RS*RS;

    Eigen::Matrix3d rotO = EYE + RO*sin(panel_rot_angO) + RO2*(1-cos(panel_rot_angO));
    Eigen::Matrix3d rotF = EYE + RF*sin(panel_rot_angF) + RF2*(1-cos(panel_rot_angF));
    Eigen::Matrix3d rotS = EYE + RS*sin(panel_rot_angS) + RS2*(1-cos(panel_rot_angS));

    boost::shared_ptr<panel_manager> pan_rot = boost::dynamic_pointer_cast<panel_manager>(panels[0]);
    pan_rot->dR = RO*cos(panel_rot_angO) + RO2*sin(panel_rot_angO);
    pan_rot->value = panel_rot_angO;
    Eigen::Matrix3d ROT = (pan_rot->dR)*rotF*rotS;
    pan_rot->dF = ROT*fs_vec;
    pan_rot->dS = ROT*ss_vec;

    pan_rot = boost::dynamic_pointer_cast<panel_manager>(panels[4]);
    pan_rot->dR = RF*cos(panel_rot_angF) + RF2*sin(panel_rot_angF);
    pan_rot->value = panel_rot_angF;
    ROT = rotO*(pan_rot->dR)*rotS;
    pan_rot->dF = ROT*fs_vec;
    pan_rot->dS = ROT*ss_vec;

    pan_rot = boost::dynamic_pointer_cast<panel_manager>(panels[5]);
    pan_rot->dR = RS*cos(panel_rot_angS) + RS2*sin(panel_rot_angS);
    pan_rot->value = panel_rot_angS;
    ROT = rotO*rotF*(pan_rot->dR);
    pan_rot->dF = ROT*fs_vec;
    pan_rot->dS = ROT*ss_vec;

    ROT = rotO*rotF*rotS;
    fs_vec = ROT*fs_vec;
    ss_vec = ROT*ss_vec;
    origin_vec = O_reference + ROT*origin_diff_vec; // add the rotated difference to the reference to recover the origin of the rotated panel
    for (int i=0; i < 3; i++){
        fdet_vector[i+1] = fs_vec[i];
        sdet_vector[i+1] = ss_vec[i];
        pix0_vector[i+1] = origin_vec[i];
    }

}
/* end panel Rot XYZ */


void diffBragg::update_dxtbx_geoms(
    const dxtbx::model::Detector& detector,
    const dxtbx::model::Beam& beam,
    int panel_id,
    double panel_rot_angO,
    double panel_rot_angF,
    double panel_rot_angS, double panel_offsetX, double panel_offsetY, double panel_offsetZ,
    bool force){

    db_cu_flags.update_detector = true;
    int old_verbose = verbose;
    verbose = 0;

    /* BEAM properties first */
    detector_panel_id = panel_id;

    double temp;
    vec3 xyz;
    /* direction in 3-space of beam vector */
    xyz = beam.get_unit_s0();
    beam_vector[1] = xyz[0];
    beam_vector[2] = xyz[1];
    beam_vector[3] = xyz[2];
    unitize(beam_vector,beam_vector);

    /* central wavelength, in Angstrom */
    db_beam.lambda0 = beam.get_wavelength()*1e-10;

    /* divergence, what are the DXTBX units? */
    temp = beam.get_divergence();
    if(temp>0.0) hdivrange = vdivrange = temp;

    /* assume this is photons/s, unless it is zero */
    temp = beam.get_flux();
    if(temp>0.0) flux = temp;

    /* assume this is Kahn polarization parameter */
    temp = beam.get_polarization_fraction();
    if(temp>=-1.0 && temp<=1.0) polarization = temp;

    /* dxtbx polarization points down B vector, we want the E vector */
    xyz = beam.get_polarization_normal();
    vert_vector[1] = xyz[0];
    vert_vector[2] = xyz[1];
    vert_vector[3] = xyz[2];
    unitize(vert_vector,vert_vector);
    cross_product(beam_vector,vert_vector,polar_vector);
    unitize(polar_vector,polar_vector);

    /* DETECTOR properties */
    /* size of the pixels in meters, this should not vary after instantiation */
    SCITBX_ASSERT(pixel_size == detector[panel_id].get_pixel_size()[0]/1000.);

    /* pixel count in short and fast-axis directions, should not change after instantiation */
    SCITBX_ASSERT(spixels == detector[panel_id].get_image_size()[1]);
    SCITBX_ASSERT(fpixels == detector[panel_id].get_image_size()[0]);

    /* direction in 3-space of detector axes */
    SCITBX_ASSERT (beam_convention == CUSTOM);

    fdet_vector[1] = detector[panel_id].get_fast_axis()[0];
    fdet_vector[2] = detector[panel_id].get_fast_axis()[1];
    fdet_vector[3] = detector[panel_id].get_fast_axis()[2];
    unitize(fdet_vector, fdet_vector);

    sdet_vector[1] = detector[panel_id].get_slow_axis()[0];
    sdet_vector[2] = detector[panel_id].get_slow_axis()[1];
    sdet_vector[3] = detector[panel_id].get_slow_axis()[2];
    unitize(sdet_vector,sdet_vector);
    /* set orthogonal vector to the detector pixel array */
    cross_product(fdet_vector,sdet_vector,odet_vector);
    unitize(odet_vector,odet_vector);
    if (! detector_is_righthanded)
        vector_scale(odet_vector, odet_vector, -1);

    /* dxtbx origin is location of outer corner of the first pixel */
    pix0_vector[1] = detector[panel_id].get_origin()[0]/1000.0;
    pix0_vector[2] = detector[panel_id].get_origin()[1]/1000.0;
    pix0_vector[3] = detector[panel_id].get_origin()[2]/1000.0;

    //if (panel_rot_ang != 0)
    //rotate_fs_ss_vecs(panel_rot_ang);
    int n_rot_refine = 0;
    if (panels[0]->refine_me)
        n_rot_refine ++;
    if (panels[4]->refine_me)
        n_rot_refine ++;
    if (panels[5]->refine_me)
        n_rot_refine ++;
    if (n_rot_refine  > 0 || force){
        rotate_fs_ss_vecs_3D(panel_rot_angO, panel_rot_angF, panel_rot_angS);

        /* reset orthogonal vector to the detector pixel array after rotation */
        cross_product(fdet_vector,sdet_vector,odet_vector);
        unitize(odet_vector,odet_vector);
        if (! detector_is_righthanded)
            vector_scale(odet_vector, odet_vector, -1);

        boost::shared_ptr<panel_manager> pan_rot;
        Eigen::Vector3d fs_vec(fdet_vector[1], fdet_vector[2], fdet_vector[3]);
        Eigen::Vector3d ss_vec(sdet_vector[1], sdet_vector[2], sdet_vector[3]);
        int pan_mans[3] = {0,4,5};
        for (int i_rot=0; i_rot <3; i_rot++ ){
            int panels_id = pan_mans[i_rot];
            pan_rot = boost::dynamic_pointer_cast<panel_manager>(panels[panels_id]);
            if (!detector_is_righthanded){
                pan_rot->F_cross_dS = (pan_rot->dS).cross(fs_vec);
                pan_rot->dF_cross_S = ss_vec.cross(pan_rot->dF);
            }
            else{
                pan_rot->F_cross_dS = fs_vec.cross(pan_rot->dS);
                pan_rot->dF_cross_S = (pan_rot->dF).cross(ss_vec);
            }
        }
    }

    pix0_vector[1] += panel_offsetX;///1000;
    pix0_vector[2] += panel_offsetY;///1000;
    pix0_vector[3] += panel_offsetZ;///1000;

    Fclose = Xclose = -dot_product(pix0_vector, fdet_vector);
    Sclose = Yclose = -dot_product(pix0_vector, sdet_vector);
    close_distance = distance = dot_product(pix0_vector, odet_vector);

    /* set beam centre */
    Eigen::Matrix3d dmat;
    dmat<<fdet_vector[1], sdet_vector[1], pix0_vector[1]*1000,
          fdet_vector[2], sdet_vector[2], pix0_vector[2]*1000,
          fdet_vector[3], sdet_vector[3], pix0_vector[3]*1000;
    Eigen::Matrix3d Dmat = dmat.inverse();
    Eigen::Vector3d s0(beam.get_s0()[0], beam.get_s0()[1], beam.get_s0()[2]);
    Eigen::Vector3d dxtbx_v = Dmat*s0;
    SCITBX_ASSERT(dxtbx_v[2] > 0);

    double rotated_center_x = dxtbx_v[0] / dxtbx_v[2];
    double rotated_center_y = dxtbx_v[1] / dxtbx_v[2];

    scitbx::vec2<double> dials_bc = detector[panel_id].get_beam_centre(beam.get_s0());
    dials_bc[0] = rotated_center_x;
    dials_bc[1] = rotated_center_y;
    Xbeam = dials_bc[0]/1000.0;
    Ybeam = dials_bc[1]/1000.0;

    /* detector sensor layer properties */
    detector_thick = detector[panel_id].get_thickness();
    temp = detector[panel_id].get_mu();        // is this really a mu? or mu/rho ?
    if(temp>0.0) detector_attnlen = 1.0/temp;

    /* quantum_gain = amp_gain * electrooptical_gain, does not include capture_fraction */
    quantum_gain = detector[panel_id].get_gain();
    //adc_offset = detector[panel_id].ADC_OFFSET;

    /* SPINDLE properties */

    /* By default align the rotation axis with the detector fast direction */
    spindle_vector[1] = fdet_vector[1];
    spindle_vector[2] = fdet_vector[2];
    spindle_vector[3] = fdet_vector[3];
    unitize(spindle_vector,spindle_vector);

    /* OMG So important otherwise center walks */
    ORGX=NAN;
    ORGY=NAN;

    user_beam=true;

    init_beam();
    init_beamcenter();
    update_beamcenter();

    int pan_rot_ids[3] = {0,4,5};
    boost::shared_ptr<panel_manager> pan;
    for (int ii=0; ii < 3; ii++){
        db_det.fdet_vectors[panel_id*3 + ii] = fdet_vector[ii+1];
        db_det.sdet_vectors[panel_id*3 + ii] = sdet_vector[ii+1];
        db_det.pix0_vectors[panel_id*3 + ii] = pix0_vector[ii+1];
        db_det.odet_vectors[panel_id*3 + ii] = odet_vector[ii+1];

        int i_rot = pan_rot_ids[ii];
        pan = boost::dynamic_pointer_cast<panel_manager>(panels[i_rot]);
        db_det.dF_vecs[panel_id*3 + ii] = pan->dF;
        db_det.dS_vecs[panel_id*3 + ii] = pan->dS;
    }

    SCITBX_ASSERT(close_distance > 0);
    verbose = old_verbose;
    set_close_distances();
    db_cu_flags.update_panel_deriv_vecs=true;
    db_cu_flags.update_detector=true;
    }

void diffBragg::shift_originZ(const dxtbx::model::Detector& detector, double shiftZ){
    if (shiftZ != prev_shiftZ){
        for (int pid=0; pid< detector.size(); pid++)
            db_det.pix0_vectors[pid*3 + 2] = detector[pid].get_origin()[2]/1000.0 + shiftZ;
        set_close_distances();
        db_cu_flags.update_detector=true;
        prev_shiftZ = shiftZ;
    }
}

void diffBragg::init_raw_pixels_roi(){
    raw_pixels_roi = af::flex_double(Npix_total);
}

void diffBragg::initialize_managers(){
    for (int i_rot=0; i_rot < 3; i_rot++){
        if (rot_managers[i_rot]->refine_me){
            rot_managers[i_rot]->initialize(Npix_total, compute_curvatures);
            update_rotmats_on_device = true;
        }
    }
    for (int i_uc=0; i_uc < 6; i_uc++){
        if (ucell_managers[i_uc]->refine_me){
            update_dB_matrices_on_device = true;
            ucell_managers[i_uc]->initialize(Npix_total, compute_curvatures);
        }
    }
    for (int i_nc=0; i_nc < 6; i_nc ++){
        if (Ncells_managers[i_nc]->refine_me)
            Ncells_managers[i_nc]->initialize(Npix_total, compute_curvatures);
    }

    boost::shared_ptr<panel_manager> pan_orig;
    for (int i_pan_orig=0; i_pan_orig  < 3; i_pan_orig ++){
        pan_orig = boost::dynamic_pointer_cast<panel_manager>(panels[1+i_pan_orig]);
        if (pan_orig->refine_me){
            pan_orig->initialize(Npix_total, compute_curvatures);
            db_cu_flags.update_detector=true;
        }
    }

    if (fcell_managers[0]->refine_me){
        fcell_managers[0]->initialize(Npix_total, compute_curvatures);
        //fcell_managers[1]->initialize(Npix_total, compute_curvatures);
        //fcell_managers[2]->initialize(Npix_total, compute_curvatures);
        db_cu_flags.update_Fhkl = true;
        db_cu_flags.update_Fhkl_scales = true;
        }

    for (int i_eta=0; i_eta<3; i_eta++){
        int counter =0;
        if (eta_managers[i_eta]->refine_me){
            if (verbose)printf("Initializing eta %d\n", i_eta);
            eta_managers[i_eta]->initialize(Npix_total, compute_curvatures);
            counter +=1;
            }
        if (counter>0){
            update_umats_on_device=true;
            vectorize_umats();
        }
    }

    for (int i_lam=0; i_lam < 2; i_lam++){
        if (lambda_managers[i_lam]->refine_me)
            lambda_managers[i_lam]->initialize(Npix_total, compute_curvatures);
    }

    int panrot_manager_indices[3] = {0,4,5};
    boost::shared_ptr<panel_manager> pan_rot;
    for (int i=0; i< 3; i++){
        int manager_idx = panrot_manager_indices[i];
        pan_rot = boost::dynamic_pointer_cast<panel_manager>(panels[manager_idx]);
        if (pan_rot->refine_me){
            update_panel_deriv_vecs_on_device=true;
            db_cu_flags.update_detector=true;
            pan_rot->initialize(Npix_total, compute_curvatures);
        }
    }
    if (fp_fdp_managers[0]->refine_me)
        fp_fdp_managers[0]->initialize(Npix_total, compute_curvatures);
    if (fp_fdp_managers[1]->refine_me)
        fp_fdp_managers[1]->initialize(Npix_total, compute_curvatures);


}

af::shared<mat3>
diffBragg::get_mosaic_blocks_prime() {
    af::shared<mat3> result;
    int num_blocks = mosaic_domains;
    // TODO if refining aniso eta, then num_blocks for the primes is 3*mosaic_domains, address this case
    for(mos_tic=0;mos_tic<num_blocks;++mos_tic) {
        result.push_back(mat3( mosaic_umats_prime[9*mos_tic+0], mosaic_umats_prime[9*mos_tic+1], mosaic_umats_prime[9*mos_tic+2],
                               mosaic_umats_prime[9*mos_tic+3], mosaic_umats_prime[9*mos_tic+4], mosaic_umats_prime[9*mos_tic+5],
                               mosaic_umats_prime[9*mos_tic+6], mosaic_umats_prime[9*mos_tic+7], mosaic_umats_prime[9*mos_tic+8]));
    }
    return result;
}


void diffBragg::vectorize_umats(){
    /* vector store two copies of Umats, one unperturbed for reference */
    if (db_cryst.UMATS.size() > 0){
        db_cryst.UMATS.clear();
        db_cryst.UMATS_RXYZ.clear();
    }
    if (db_cryst.UMATS_prime.size()> 0){
        db_cryst.UMATS_prime.clear();
        db_cryst.UMATS_RXYZ_prime.clear();
    }
    if (db_cryst.UMATS_dbl_prime.size()> 0){
        db_cryst.UMATS_dbl_prime.clear();
        db_cryst.UMATS_RXYZ_dbl_prime.clear();
    }
    for(mos_tic=0;mos_tic<mosaic_domains;++mos_tic){
        double uxx,uxy,uxz,uyx,uyy,uyz,uzx,uzy,uzz;
        uxx = mosaic_umats[mos_tic*9+0];
        uxy = mosaic_umats[mos_tic*9+1];
        uxz = mosaic_umats[mos_tic*9+2];
        uyx = mosaic_umats[mos_tic*9+3];
        uyy = mosaic_umats[mos_tic*9+4];
        uyz = mosaic_umats[mos_tic*9+5];
        uzx = mosaic_umats[mos_tic*9+6];
        uzy = mosaic_umats[mos_tic*9+7];
        uzz = mosaic_umats[mos_tic*9+8];
        Eigen::Matrix3d U;
        U << uxx, uxy, uxz,
             uyx, uyy, uyz,
             uzx, uzy, uzz;
        db_cryst.UMATS.push_back(U);
        db_cryst.UMATS_RXYZ.push_back(U);
    }

    for (int i_eta=0; i_eta<3; i_eta ++){
        if (eta_managers[i_eta]->refine_me){
            for (int i_mos=0; i_mos< mosaic_domains; i_mos++){
                SCITBX_ASSERT(mosaic_umats_prime != NULL);
                int mos_tic2 = mosaic_domains*i_eta + i_mos;
                double uxx = mosaic_umats_prime[mos_tic2*9+0];
                double uxy = mosaic_umats_prime[mos_tic2*9+1];
                double uxz = mosaic_umats_prime[mos_tic2*9+2];
                double uyx = mosaic_umats_prime[mos_tic2*9+3];
                double uyy = mosaic_umats_prime[mos_tic2*9+4];
                double uyz = mosaic_umats_prime[mos_tic2*9+5];
                double uzx = mosaic_umats_prime[mos_tic2*9+6];
                double uzy = mosaic_umats_prime[mos_tic2*9+7];
                double uzz = mosaic_umats_prime[mos_tic2*9+8];
                Eigen::Matrix3d Up;
                Up << uxx, uxy, uxz,
                     uyx, uyy, uyz,
                     uzx, uzy, uzz;
                db_cryst.UMATS_RXYZ_prime.push_back(Up);
                db_cryst.UMATS_prime.push_back(Up);

                if (mosaic_umats_dbl_prime != NULL){
                    if (verbose && mos_tic==0)
                        printf("Setting umts dbl prime\n");
                    uxx = mosaic_umats_dbl_prime[mos_tic2*9+0];
                    uxy = mosaic_umats_dbl_prime[mos_tic2*9+1];
                    uxz = mosaic_umats_dbl_prime[mos_tic2*9+2];
                    uyx = mosaic_umats_dbl_prime[mos_tic2*9+3];
                    uyy = mosaic_umats_dbl_prime[mos_tic2*9+4];
                    uyz = mosaic_umats_dbl_prime[mos_tic2*9+5];
                    uzx = mosaic_umats_dbl_prime[mos_tic2*9+6];
                    uzy = mosaic_umats_dbl_prime[mos_tic2*9+7];
                    uzz = mosaic_umats_dbl_prime[mos_tic2*9+8];
                    Eigen::Matrix3d Udp;
                    Udp << uxx, uxy, uxz,
                         uyx, uyy, uyz,
                         uzx, uzy, uzz;
                    db_cryst.UMATS_RXYZ_dbl_prime.push_back(Udp);
                    db_cryst.UMATS_dbl_prime.push_back(Udp);
                }
            }
        }
    }
}

void diffBragg::let_loose(int refine_id){
//experimental
    if (refine_id >= 0 && refine_id < 3  ){
        rot_managers[refine_id]->refine_me=true;
    }
    else if (refine_id >=3 and refine_id < 9 ){
        // 6 possible unit cell managers (a,b,c,al,be,ga)
        ucell_managers[refine_id-3]->refine_me=true;
    }
    else if (refine_id==9){
        for (int i_nc=0; i_nc < 3; i_nc ++){
            Ncells_managers[i_nc]->refine_me=true;
        }
    }
    else if (refine_id==10){
        boost::shared_ptr<panel_manager> pan_orig = boost::dynamic_pointer_cast<panel_manager>(panels[1]);
        pan_orig->refine_me=true;
    }
    if (refine_id==23)
        db_flags.refine_diffuse = true;
}

void diffBragg::fix(int refine_id){
    if (refine_id >= 0 && refine_id < 3  ){
        rot_managers[refine_id]->refine_me=false;
    }
    else if (refine_id >=3 and refine_id < 9 ){
        // 6 possible unit cell managers (a,b,c,al,be,ga)
        ucell_managers[refine_id-3]->refine_me=false;
    }
    else if (refine_id==9){
        for (int i_nc=0; i_nc < 3; i_nc ++){
            Ncells_managers[i_nc]->refine_me=false;
        }
    }
    else if (refine_id==21){
        refine_Ncells_def = false;
        for (int i_nc=3; i_nc< 6; i_nc++){
            Ncells_managers[i_nc]->refine_me=false;
        }
    }
    else if (refine_id==10){
        boost::shared_ptr<panel_manager> pan_orig = boost::dynamic_pointer_cast<panel_manager>(panels[1]);
        pan_orig->refine_me=false;
    }
    else if(refine_id==11){
        fcell_managers[0]->refine_me=false;
        //fcell_managers[1]->refine_me=false;
        //fcell_managers[2]->refine_me=false;
    }

    else if (refine_id==12 || refine_id==13){
        int i_lam = refine_id-12;
        lambda_managers[i_lam]->refine_me=false;
    }

    else if (refine_id==14){
        boost::shared_ptr<panel_manager> pan_rot = boost::dynamic_pointer_cast<panel_manager>(panels[0]);
        pan_rot->refine_me=false;
    }

    else if (refine_id==15){
        boost::shared_ptr<panel_manager> pan_orig = boost::dynamic_pointer_cast<panel_manager>(panels[2]);
        pan_orig->refine_me=false;
    }

    else if (refine_id==16){
        boost::shared_ptr<panel_manager> pan_orig = boost::dynamic_pointer_cast<panel_manager>(panels[3]);
        pan_orig->refine_me=false;
    }
    else if (refine_id==17){
        boost::shared_ptr<panel_manager> pan_rot = boost::dynamic_pointer_cast<panel_manager>(panels[4]);
        pan_rot->refine_me=false;
    }
    else if (refine_id==18){
        boost::shared_ptr<panel_manager> pan_rot = boost::dynamic_pointer_cast<panel_manager>(panels[5]);
        pan_rot->refine_me=false;
    }
    else if (refine_id==19){
        eta_managers[0]->refine_me=false;
        if (modeling_anisotropic_mosaic_spread){
            for (int i_eta=1; i_eta<3; i_eta++){
                eta_managers[i_eta]->refine_me=false;
            }
        }
    }
    else if (refine_id==22){
        fp_fdp_managers[0]->refine_me = false;
        fp_fdp_managers[1]->refine_me = false;
    }
    if (refine_id==23)
        db_flags.refine_diffuse = false;
}


void diffBragg::refine(int refine_id){
    if (refine_id >= 0 && refine_id < 3  ){
        // 3 possitle rotation managers (rotX, rotY, rotZ)
        rot_managers[refine_id]->refine_me=true;
        rot_managers[refine_id]->initialize(Npix_total, compute_curvatures);
        update_rotmats_on_device=true;
    }
    else if (refine_id >=3 and refine_id < 9 ){
        // 6 possible unit cell managers (a,b,c,al,be,ga)
        ucell_managers[refine_id-3]->refine_me=true;
        ucell_managers[refine_id-3]->initialize(Npix_total, compute_curvatures);
        update_dB_matrices_on_device = true;
    }
    else if (refine_id==9){
        for (int i_nc=0; i_nc < 3; i_nc ++){
            Ncells_managers[i_nc]->refine_me=true;
            Ncells_managers[i_nc]->initialize(Npix_total, compute_curvatures);
        }
    }
    else if (refine_id==21){
        refine_Ncells_def = true;
        for (int i_nc=3; i_nc< 6; i_nc++){
            Ncells_managers[i_nc]->refine_me=true;
            Ncells_managers[i_nc]->initialize(Npix_total, compute_curvatures);
        }
    }
    else if (refine_id==10){
        boost::shared_ptr<panel_manager> pan_orig = boost::dynamic_pointer_cast<panel_manager>(panels[1]);
        pan_orig->refine_me=true;
        pan_orig->initialize(Npix_total, compute_curvatures);
        db_cu_flags.update_detector=true;
    }
    else if(refine_id==11){
        fcell_managers[0]->refine_me=true;
        fcell_managers[0]->initialize(Npix_total, compute_curvatures);
        db_cu_flags.update_Fhkl = true;
        db_cu_flags.update_Fhkl_scales = true;
    }

    else if (refine_id==12 || refine_id==13){
        use_lambda_coefficients = true;
        int i_lam = refine_id-12;
        lambda_managers[i_lam]->refine_me=true;
        lambda_managers[i_lam]->initialize(Npix_total, compute_curvatures);
    }

    else if (refine_id==14){
        boost::shared_ptr<panel_manager> pan_rot = boost::dynamic_pointer_cast<panel_manager>(panels[0]);
        pan_rot->refine_me=true;
        rotate_fs_ss_vecs_3D(0,0,0);
        pan_rot->initialize(Npix_total, compute_curvatures);
        db_cu_flags.update_detector = true;
        update_panel_deriv_vecs_on_device = true;
    }

    else if (refine_id==15){
        boost::shared_ptr<panel_manager> pan_orig = boost::dynamic_pointer_cast<panel_manager>(panels[2]);
        pan_orig->refine_me=true;
        pan_orig->initialize(Npix_total, compute_curvatures);
        db_cu_flags.update_detector = true;
    }

    else if (refine_id==16){
        boost::shared_ptr<panel_manager> pan_orig = boost::dynamic_pointer_cast<panel_manager>(panels[3]);
        pan_orig->refine_me=true;
        pan_orig->initialize(Npix_total, compute_curvatures);
        db_cu_flags.update_detector = true;
    }
    else if (refine_id==17){
        boost::shared_ptr<panel_manager> pan_rot = boost::dynamic_pointer_cast<panel_manager>(panels[4]);
        pan_rot->refine_me=true;
        rotate_fs_ss_vecs_3D(0,0,0);
        pan_rot->initialize(Npix_total, compute_curvatures);
        db_cu_flags.update_detector = true;
        update_panel_deriv_vecs_on_device = true;
    }
    else if (refine_id==18){
        boost::shared_ptr<panel_manager> pan_rot = boost::dynamic_pointer_cast<panel_manager>(panels[5]);
        pan_rot->refine_me=true;
        rotate_fs_ss_vecs_3D(0,0,0);
        pan_rot->initialize(Npix_total, compute_curvatures);
        db_cu_flags.update_detector = true;
        update_panel_deriv_vecs_on_device = true;
    }
    else if (refine_id==19){
        eta_managers[0]->refine_me=true;
        SCITBX_ASSERT(mosaic_umats_prime != NULL);
        eta_managers[0]->initialize(Npix_total, compute_curvatures);
        update_umats_on_device = true;
        if (modeling_anisotropic_mosaic_spread){
            if (verbose){
                printf("Initializing for anisotropic mosaic spread modeling!\n");
            }
            for (int i_eta=1; i_eta<3; i_eta++){
                eta_managers[i_eta]->refine_me=true;
                eta_managers[i_eta]->initialize(Npix_total, compute_curvatures);
            }
        }
    }
    else if(refine_id==22){
        fp_fdp_managers[0]->refine_me = true;
        fp_fdp_managers[1]->refine_me = true;
        fp_fdp_managers[0]->initialize(Npix_total, compute_curvatures);
        fp_fdp_managers[1]->initialize(Npix_total, compute_curvatures);
    }
    if (refine_id==23)
        db_flags.refine_diffuse = true;
}

void diffBragg::print_if_refining(){
    for (int i=0; i< 3; i++){
        if (rot_managers[i]->refine_me)
            printf("Refining rot %d\n", i);
    }
    for (int i=0; i< 6; i++){
        if (ucell_managers[i]->refine_me)
            printf("Refining ucell %d\n", i);
        if (Ncells_managers[i]->refine_me)
            printf("Refining Ncells %d\n", i);
    }
    if (fcell_managers[0]->refine_me)
        printf("Refining Fcell\n");
    if (lambda_managers[0]->refine_me)
        printf("Refining Lambda0\n");
    if (lambda_managers[1]->refine_me)
        printf("Refining Lambda1\n");

    if (panels[0]->refine_me)
        printf("Refining panel rot O\n");
    if (panels[4]->refine_me)
        printf("Refining panel rot F\n");
    if (panels[5]->refine_me)
        printf("Refining panel rot S\n");
    if (panels[1]->refine_me)
        printf("Refining panel Z\n");
    if (panels[2]->refine_me)
        printf("Refining panel X\n");
    if (panels[3]->refine_me)
        printf("Refining panel Y\n");
    if (eta_managers[0]->refine_me)
        printf("refining eta 0\n");
    if (eta_managers[1]->refine_me)
        printf("refining eta 1\n");
    if (eta_managers[2]->refine_me)
        printf("refining eta 2\n");

    if(fp_fdp_managers[0]->refine_me)
      printf("refining fdp center\n");
    if(fp_fdp_managers[1]->refine_me)
      printf("refining fdp slope\n");
}

void diffBragg::update_Fhkl_channels(np::ndarray& channels){
    SCITBX_ASSERT (channels.shape(0)==sources);
    db_beam.Fhkl_channels.clear();
    int* channels_ptr = reinterpret_cast<int*>(channels.get_data());
    for (int i=0; i < sources; i++){
        int channel_id = *(channels_ptr+i);
        db_beam.Fhkl_channels.push_back(channel_id);
        if (verbose)
            printf("source=%d channel_id=%d\n", i, channel_id);
    }
    db_cu_flags.update_Fhkl_channels=true;
}

boost::python::list diffBragg::get_Fhkl_channels(){
    boost::python::list channels;
    for (int i=0; i < db_beam.Fhkl_channels.size(); i++){
        int channel_id = db_beam.Fhkl_channels [i];
        channels.append(channel_id);
    }
    return channels;
}

void diffBragg::update_xray_beams(scitbx::af::versa<dxtbx::model::Beam, scitbx::af::flex_grid<> > const& value) {
    pythony_beams = value;
    //SCITBX_ASSERT(sources==pythony_beams.size());

    if(verbose) printf("udpating pythony sources\n");
    //vec3 xyz,beamdir=vec3(0,0,0),polarvec = vec3(0,0,0);
    double flux_sum = 0.0, lambda_sum = 0.0;//, polar_sum = 0.0, div_sum = 0.0;
    for (i=0; i < sources; ++i)
    {
        source_I[i] = pythony_beams[i].get_flux();
        flux_sum += source_I[i];
        source_lambda[i] = pythony_beams[i].get_wavelength();
        lambda_sum += source_lambda[i];

    }
    /* update averaged parameters */
    if(lambda_sum>0.0) db_beam.lambda0 = lambda_sum/sources;

    /* take in total flux */
    if(flux_sum > 0)
    {
        flux = flux_sum;
        init_beam();
    }
    /* make sure stored source intensities are fractional */
    double norm = flux_sum ; //sources;
    for (i=0; i < sources && norm>0.0; ++i)
    {
        source_I[i] /= norm;
    }
    if(verbose) printf("done initializing sources:\n");
}



void diffBragg::quick_Fcell_update(boost::python::tuple const& value){
    pythony_indices = boost::python::extract<indices>(value[0]);
    pythony_amplitudes = boost::python::extract<af::shared<double> >(value[1]);
    if(verbose) printf("updating Fhkl with pythony indices and amplitudes\n");
    miller_t hkl;
    for (int i_h=0; i_h < pythony_indices.size(); ++i_h)
    {
        hkl = pythony_indices[i_h];
        F_cell = pythony_amplitudes[i_h];
        h0 = hkl[0];
        k0 = hkl[1];
        l0 = hkl[2];
        int h = h0-h_min;
        int k = k0-k_min;
        int l = l0-l_min;
        //Fhkl[h0-h_min][k0-k_min][l0-l_min]=F_cell;
        db_cryst.FhklLinear[h*k_range*l_range+ k*l_range + l] = F_cell;
        if(verbose>9) printf("F %d : %d %d %d = %g\n",i_h,h,k,l,F_cell);
    }
    //update_linear_Fhkl();
    if(verbose) printf("done with quick update of Fhkl:\n");
    db_cu_flags.update_Fhkl=true;
}


void diffBragg::init_Fhkl2()
{
    /* This should only be called if init_Fhkl has already been called with python indices/amplitudes*/
    /* free any previous allocations */
    if(Fhkl2 != NULL) {
        for (h0=0; h0<=h_range;h0++) {
            for (k0=0; k0<=k_range;k0++) {
                free(Fhkl2[h0][k0]);
            }
            free(Fhkl2[h0]);
        }
        free(Fhkl2);
    }

    /* allocate memory for 3d arrays */
    Fhkl2 = (double***) calloc(h_range+1,sizeof(double**));
    if(Fhkl2==NULL){perror("ERROR");exit(9);};
    for (h0=0; h0<=h_range;h0++) {
        Fhkl2[h0] = (double**) calloc(k_range+1,sizeof(double*));
        if(Fhkl2[h0]==NULL){perror("ERROR");exit(9);};
        for (k0=0; k0<=k_range;k0++) {
            Fhkl2[h0][k0] = (double*) calloc(l_range+1,sizeof(double));
            if(Fhkl2[h0][k0]==NULL){perror("ERROR");exit(9);};
        }
    }
    for (h0=0; h0<h_range;h0++) {
        for (k0=0; k0<k_range;k0++) {
            for (l0=0; l0<l_range;l0++) {
                Fhkl2[h0][k0][l0] = 0;
            }
        }
    }
    Fhkl2[-h_min][-k_min][-l_min] = 0;

    if(verbose) printf("initializing Fhkl2 with pythony indices and amplitudes\n");
    miller_t hkl;
    for (i=0; i < pythony_indices.size(); ++i)
    {
        hkl = pythony_indices[i];
        F_cell = pythony_amplitudes2[i];
        h0 = hkl[0];
        k0 = hkl[1];
        l0 = hkl[2];
        Fhkl2[h0-h_min][k0-k_min][l0-l_min]=F_cell;
    }
    Fhkl2[-h_min][-k_min][-l_min] = 0;
    if(verbose) printf("done initializing Fhkl2:\n");
}


void diffBragg::set_ucell_derivative_matrix(int refine_id, af::shared<double> const& value){
    int ucell_param_idx = refine_id-3;  // its just how the API works, pass in 3 for first ucell matrix
    if (ucell_param_idx < 0 || ucell_param_idx > 5)
      printf("WARNING, passing in wrong refine_id for unit cell parameter (should be 3-8).\nNothing done.\n");
    else
        ucell_managers[ucell_param_idx]->dB <<
                        value[0], value[1], value[2],
                        value[3], value[4], value[5],
                        value[6], value[7], value[8];
}

void diffBragg::set_ucell_second_derivative_matrix(int refine_id, af::shared<double> const& value){
    int ucell_param_idx = refine_id-3;  // its just how the API works, pass in 3 for first ucell matrix
    if (ucell_param_idx < 0 || ucell_param_idx > 5)
      printf("WARNING, passing in wrong refine_id for unit cell parameter (should be 3-8).\nNothing done.\n");
    else
        ucell_managers[ucell_param_idx]->dB2 <<
                        value[0], value[1], value[2],
                        value[3], value[4], value[5],
                        value[6], value[7], value[8];
}

/* Begin parameter set/get */

// TODO : rename set_value and get_value because they dont apply to ucell derivatives...
// this function will get exeedingly complicated because it will try to ensure all the dependent parameters get
// adjusted when we update a given parameter that we are refining
// For example updating Ncells_abc should also update oversample, and should also update xtal_size
void diffBragg::set_ncells_values( boost::python::tuple const& values){
    Na = boost::python::extract<double>(values[0]);
    Nb = boost::python::extract<double>(values[1]);
    Nc = boost::python::extract<double>(values[2]);
    Ncells_managers[0]->value=Na;
    Ncells_managers[1]->value=Nb;
    Ncells_managers[2]->value=Nc;
    xtal_size_x = -1;
    xtal_size_y = -1;
    xtal_size_z = -1;
    if (update_oversample_during_refinement)
       update_oversample();
    NABC(0,0) = Na;
    NABC(1,1) = Nb;
    NABC(2,2) = Nc;
}

void diffBragg::set_ncells_def_values( boost::python::tuple const& values){
    Nd = boost::python::extract<double>(values[0]);
    Ne = boost::python::extract<double>(values[1]);
    Nf = boost::python::extract<double>(values[2]);
    Ncells_managers[3]->value=Nd;
    Ncells_managers[4]->value=Ne;
    Ncells_managers[5]->value=Nf;
    //xtal_size_x = -1;
    //xtal_size_y = -1;
    //xtal_size_z = -1;
    //if (update_oversample_during_refinement)
    //   update_oversample();
    NABC(0,1) = Nd;
    NABC(1,0) = Nd;
    NABC(1,2) = Ne;
    NABC(2,1) = Ne;
    NABC(0,2) = Nf;
    NABC(2,0) = Nf;
}

boost::python::tuple diffBragg::get_ncells_values(){
    boost::python::tuple values;
    values = boost::python::make_tuple( NABC(0,0),  NABC(1,1), NABC(2,2));
    return values;
}

void diffBragg::show_heavy_atom_data(){
  int natom = atom_data.size()/5;
  for (i=0; i<natom;i++){
    double x =db_cryst.atom_data[i*5];
    double y =db_cryst.atom_data[i*5+1];
    double z =db_cryst.atom_data[i*5+2];
    double B =db_cryst.atom_data[i*5+3];
    double O =db_cryst.atom_data[i*5+4];
    printf("Atom %d at position %5.3g,%5.3g,%5.3g, Bfactor=%5.2g Ang^2, Occupancy=%2.3g\n",i,x,y,z,B,O);
  }
}

void diffBragg::show_fp_fdp(){
  for(int i=0; i< sources; i++){
    double fp = db_cryst.fpfdp[2*i];
    double fdp=db_cryst.fpfdp[2*i+1];
    printf("Source %d, fp=%8.3g, fdp=%8.3g\n", i,fp,fdp);
  }
}



void diffBragg::set_value( int refine_id, double value ){
    if (refine_id < 3){
        rot_managers[refine_id]->value = value;
        rot_managers[refine_id]->set_R();
    }
    if (refine_id==9){
        Ncells_managers[0]->value = value;
        Na=value;
        Nb=value;
        Nc=value;
        xtal_size_x = -1;
        xtal_size_y = -1;
        xtal_size_z = -1;
        //TODO make me optional!
        if (update_oversample_during_refinement)
            update_oversample();
        NABC(0,0) = value;
        NABC(1,1) = value;
        NABC(2,2) = value;
    }
}

double diffBragg::get_value(int refine_id){
    double value(0);
    if (refine_id < 3)
        value = rot_managers[refine_id]->value;
    else if (refine_id ==9)
        value = Ncells_managers[0]->value;
    return value;
}
/* End parameter set/get */



af::flex_double diffBragg::get_derivative_pixels(int refine_id){

    SCITBX_ASSERT(refine_id >=0 && refine_id <= 19);

    if (refine_id>=0 and refine_id < 3){
        SCITBX_ASSERT(rot_managers[refine_id]->refine_me);
        return rot_managers[refine_id]->raw_pixels;
        }
    else if(refine_id >=3 && refine_id < 9){
        int i_uc = refine_id-3;
        SCITBX_ASSERT(i_uc >= 0);
        SCITBX_ASSERT(i_uc < 6);
        SCITBX_ASSERT(ucell_managers[i_uc]->refine_me);
        return ucell_managers[i_uc]->raw_pixels;
        }
    else if (refine_id==9)
        return Ncells_managers[0]->raw_pixels;
    else if (refine_id==10){
        boost::shared_ptr<panel_manager> pan_orig = boost::dynamic_pointer_cast<panel_manager>(panels[1]);
        return pan_orig->raw_pixels;
        }
    else if (refine_id==11)
        return fcell_managers[0]->raw_pixels;
    else if (refine_id==12)
        return lambda_managers[0]->raw_pixels;
    else if  (refine_id==13)
        return lambda_managers[1]->raw_pixels;
    else if (refine_id==14){
        boost::shared_ptr<panel_manager> pan_rot = boost::dynamic_pointer_cast<panel_manager>(panels[0]);
        return pan_rot->raw_pixels;
    }
    else if (refine_id==15){
        boost::shared_ptr<panel_manager> pan_orig = boost::dynamic_pointer_cast<panel_manager>(panels[2]);
        return pan_orig->raw_pixels;
    }
    else if(refine_id==16){
        boost::shared_ptr<panel_manager> pan_orig = boost::dynamic_pointer_cast<panel_manager>(panels[3]);
        return pan_orig->raw_pixels;
    }
    else if(refine_id==17){
        boost::shared_ptr<panel_manager> pan_rot = boost::dynamic_pointer_cast<panel_manager>(panels[4]);
        return pan_rot->raw_pixels;
    }
    else if (refine_id==18){
        boost::shared_ptr<panel_manager> pan_rot = boost::dynamic_pointer_cast<panel_manager>(panels[5]);
        return pan_rot->raw_pixels;
    }
    else // refine_id == 19
        return eta_managers[0]->raw_pixels; // just one component of the mosaicity, legacy code because anisotropic mosaicity used to lack support
}


af::flex_double diffBragg::get_second_derivative_pixels(int refine_id){
    if (refine_id>=0 and refine_id < 3){
        SCITBX_ASSERT(rot_managers[refine_id]->refine_me);
        return rot_managers[refine_id]->raw_pixels2;
        }
    else if(refine_id >=3 && refine_id < 9){
        int i_uc = refine_id-3;
        SCITBX_ASSERT(i_uc >= 0);
        SCITBX_ASSERT(i_uc < 6);
        SCITBX_ASSERT(ucell_managers[i_uc]->refine_me);
        return ucell_managers[i_uc]->raw_pixels2;
        }
    else if (refine_id == 9)
        return Ncells_managers[0]->raw_pixels2;
    else if (refine_id==10){
        boost::shared_ptr<panel_manager> pan_orig = boost::dynamic_pointer_cast<panel_manager>(panels[1]);
        return pan_orig->raw_pixels2;}
    else if (refine_id==15){
        boost::shared_ptr<panel_manager> pan_orig = boost::dynamic_pointer_cast<panel_manager>(panels[2]);
        return pan_orig->raw_pixels2;}
    else if (refine_id==16){
        boost::shared_ptr<panel_manager> pan_orig = boost::dynamic_pointer_cast<panel_manager>(panels[3]);
        return pan_orig->raw_pixels2;}
    else if (refine_id==19)
        return eta_managers[0]->raw_pixels2;
    else if (refine_id==11)
        return fcell_managers[0]->raw_pixels2;
    else{
       printf("Not suppotrted for refine id %d\n", refine_id);
       SCITBX_ASSERT(false);
   }
}
//lkalsd
boost::python::tuple diffBragg::get_fp_fdp_derivative_pixels(){
    boost::python::tuple derivative_pixels;
    SCITBX_ASSERT(fp_fdp_managers[0]->refine_me);
    SCITBX_ASSERT(fp_fdp_managers[1]->refine_me);
    derivative_pixels = boost::python::make_tuple(fp_fdp_managers[0]->raw_pixels,
      fp_fdp_managers[1]->raw_pixels);
    return derivative_pixels;
}

boost::python::tuple diffBragg::get_ncells_derivative_pixels(){
    SCITBX_ASSERT(Ncells_managers[0]->refine_me);
    SCITBX_ASSERT(Ncells_managers[1]->refine_me);
    SCITBX_ASSERT(Ncells_managers[2]->refine_me);
    boost::python::tuple derivative_pixels;
    derivative_pixels = boost::python::make_tuple(Ncells_managers[0]->raw_pixels,
        Ncells_managers[1]->raw_pixels, Ncells_managers[2]->raw_pixels);
    return derivative_pixels;
}

boost::python::tuple diffBragg::get_diffuse_gamma_derivative_pixels(){
    SCITBX_ASSERT(db_flags.refine_diffuse);
    int Npix_total = first_deriv_imgs.diffuse_gamma.size() / 3;
    af::flex_double raw_pixels_a = af::flex_double(Npix_total);
    af::flex_double raw_pixels_b = af::flex_double(Npix_total);
    af::flex_double raw_pixels_c = af::flex_double(Npix_total);

    double* floatimage_a = raw_pixels_a.begin();
    double* floatimage_b = raw_pixels_b.begin();
    double* floatimage_c = raw_pixels_c.begin();
    for (int ii=0; ii< Npix_to_model; ii++){
        floatimage_a[ii] = first_deriv_imgs.diffuse_gamma[ii];
        floatimage_b[ii] = first_deriv_imgs.diffuse_gamma[Npix_to_model + ii];
        floatimage_c[ii] = first_deriv_imgs.diffuse_gamma[2*Npix_to_model + ii];
    }
    boost::python::tuple derivative_pixels;
    derivative_pixels = boost::python::make_tuple(raw_pixels_a, raw_pixels_b, raw_pixels_c);
    return derivative_pixels;
}

// TODO: Merge the gamma and sigma getters

  boost::python::tuple diffBragg::get_diffuse_sigma_derivative_pixels(){
    SCITBX_ASSERT(db_flags.refine_diffuse);
    int Npix_total = first_deriv_imgs.diffuse_sigma.size() / 3;
    af::flex_double raw_pixels_a = af::flex_double(Npix_total);
    af::flex_double raw_pixels_b = af::flex_double(Npix_total);
    af::flex_double raw_pixels_c = af::flex_double(Npix_total);

    double* floatimage_a = raw_pixels_a.begin();
    double* floatimage_b = raw_pixels_b.begin();
    double* floatimage_c = raw_pixels_c.begin();
    for (int ii=0; ii< Npix_to_model; ii++){
        floatimage_a[ii] = first_deriv_imgs.diffuse_sigma[ii];
        floatimage_b[ii] = first_deriv_imgs.diffuse_sigma[Npix_to_model + ii];
        floatimage_c[ii] = first_deriv_imgs.diffuse_sigma[2*Npix_to_model + ii];
    }
    boost::python::tuple derivative_pixels;
    derivative_pixels = boost::python::make_tuple(raw_pixels_a, raw_pixels_b, raw_pixels_c);
    return derivative_pixels;
}


boost::python::tuple diffBragg::get_ncells_def_derivative_pixels(){
    SCITBX_ASSERT(Ncells_managers[3]->refine_me);
    SCITBX_ASSERT(Ncells_managers[4]->refine_me);
    SCITBX_ASSERT(Ncells_managers[5]->refine_me);
    boost::python::tuple derivative_pixels;
    derivative_pixels = boost::python::make_tuple(Ncells_managers[3]->raw_pixels,
        Ncells_managers[4]->raw_pixels, Ncells_managers[5]->raw_pixels);
    return derivative_pixels;
}

boost::python::tuple diffBragg::get_aniso_eta_deriv_pixels(){
    SCITBX_ASSERT(eta_managers[0]->refine_me);
    SCITBX_ASSERT(eta_managers[1]->refine_me);
    SCITBX_ASSERT(eta_managers[2]->refine_me);
    boost::python::tuple derivative_pixels;
    derivative_pixels = boost::python::make_tuple(eta_managers[0]->raw_pixels,
        eta_managers[1]->raw_pixels, eta_managers[2]->raw_pixels);
    return derivative_pixels;
}
boost::python::tuple diffBragg::get_aniso_eta_second_deriv_pixels(){
    SCITBX_ASSERT(eta_managers[0]->refine_me);
    SCITBX_ASSERT(eta_managers[1]->refine_me);
    SCITBX_ASSERT(eta_managers[2]->refine_me);
    boost::python::tuple derivative_pixels;
    derivative_pixels = boost::python::make_tuple(eta_managers[0]->raw_pixels2,
        eta_managers[1]->raw_pixels2, eta_managers[2]->raw_pixels2);
    return derivative_pixels;
}


boost::python::tuple diffBragg::get_ncells_def_second_derivative_pixels(){
    SCITBX_ASSERT(Ncells_managers[3]->refine_me);
    SCITBX_ASSERT(Ncells_managers[4]->refine_me);
    SCITBX_ASSERT(Ncells_managers[5]->refine_me);
    boost::python::tuple second_derivative_pixels;
    second_derivative_pixels = boost::python::make_tuple(Ncells_managers[3]->raw_pixels2,
        Ncells_managers[4]->raw_pixels2, Ncells_managers[5]->raw_pixels2);
    return second_derivative_pixels;
}


boost::python::tuple diffBragg::get_lambda_derivative_pixels(){
    SCITBX_ASSERT(lambda_managers[0]->refine_me || lambda_managers[1]->refine_me);

    boost::python::tuple derivative_pixels;
    if (lambda_managers[0]->refine_me && lambda_managers[1]->refine_me){
        derivative_pixels = boost::python::make_tuple(lambda_managers[0]->raw_pixels,
            lambda_managers[1]->raw_pixels);
    }
    else{
        if (lambda_managers[0]->refine_me)
            derivative_pixels = boost::python::make_tuple(lambda_managers[0]->raw_pixels);
        else if (lambda_managers[1]->refine_me)
            derivative_pixels = boost::python::make_tuple(lambda_managers[1]->raw_pixels);
    }
    return derivative_pixels;
}

boost::python::tuple diffBragg::get_ncells_second_derivative_pixels(){
    SCITBX_ASSERT(Ncells_managers[0]->refine_me);
    SCITBX_ASSERT(Ncells_managers[1]->refine_me);
    SCITBX_ASSERT(Ncells_managers[2]->refine_me);
    boost::python::tuple second_derivative_pixels;
    second_derivative_pixels = boost::python::make_tuple(Ncells_managers[0]->raw_pixels2,
        Ncells_managers[1]->raw_pixels2, Ncells_managers[2]->raw_pixels2);
    return second_derivative_pixels;
}

void diffBragg::zero_raw_pixel_rois(){
    init_raw_pixels_roi();
    initialize_managers();
}

void diffBragg::set_mosaic_blocks_prime(af::shared<mat3> umat_in){
    /* free any previous allocations */
    if(mosaic_umats_prime != NULL) free(mosaic_umats_prime);

    /* allocate enough space */
    nmats = umat_in.size();
    SCITBX_ASSERT(mosaic_domains == nmats || 3*mosaic_domains==nmats);
    //int nfactor = umat_in.size() / mosaic_domains;
    // NOTE allow for at least 3 times the number of mosaic domains, in case of modeling anisotropic mosaicity
    mosaic_umats_prime = (double *) calloc(nmats+10,9*sizeof(double));

    /* now actually import the orientation of each domain */
    for(mos_tic=0;mos_tic<nmats;++mos_tic){
      int offset = 9 * mos_tic;
      mat3& domain = umat_in[mos_tic];
      mosaic_umats_prime[0+offset]=domain[0];mosaic_umats_prime[1+offset]=domain[1];mosaic_umats_prime[2+offset]=domain[2];
      mosaic_umats_prime[3+offset]=domain[3];mosaic_umats_prime[4+offset]=domain[4];mosaic_umats_prime[5+offset]=domain[5];
      mosaic_umats_prime[6+offset]=domain[6];mosaic_umats_prime[7+offset]=domain[7];mosaic_umats_prime[8+offset]=domain[8];
    }
    if(verbose) printf("  imported a total of %d mosaic domain derivative Umats\n", nmats);
}

void diffBragg::set_mosaic_blocks_dbl_prime(af::shared<mat3> umat_in){
    /* free any previous allocations */
    if(mosaic_umats_dbl_prime != NULL) free(mosaic_umats_dbl_prime);

    nmats = umat_in.size();

    /* allocate enough space */
    SCITBX_ASSERT(mosaic_domains == umat_in.size() || 3*mosaic_domains==umat_in.size());
    mosaic_umats_dbl_prime = (double *) calloc(nmats+10,9*sizeof(double));

    /* now actually import the orientation of each domain */
    for(mos_tic=0;mos_tic<nmats;++mos_tic){
      int offset = 9 * mos_tic;
      mat3& domain = umat_in[mos_tic];
      mosaic_umats_dbl_prime[0+offset]=domain[0];mosaic_umats_dbl_prime[1+offset]=domain[1];mosaic_umats_dbl_prime[2+offset]=domain[2];
      mosaic_umats_dbl_prime[3+offset]=domain[3];mosaic_umats_dbl_prime[4+offset]=domain[4];mosaic_umats_dbl_prime[5+offset]=domain[5];
      mosaic_umats_dbl_prime[6+offset]=domain[6];mosaic_umats_dbl_prime[7+offset]=domain[7];mosaic_umats_dbl_prime[8+offset]=domain[8];
    }
    if(verbose) printf("  imported a total of %d mosaic domain 2nd derivative Umats\n",nmats);
}

af::shared<double> diffBragg::add_diffBragg_spots_full(){

    struct timeval t1,t2;
    gettimeofday(&t1,0 );

    int Npanels = db_det.pix0_vectors.size() / 3;
    int fdim = roi_xmax-roi_xmin;
    int sdim = roi_ymax-roi_ymin;
    int npix = Npanels*spixels*fpixels;
    af::shared<size_t> pfs(npix*3);
    for (int pid=0; pid< Npanels; pid++){
        for (int s=0; s <spixels; s++){
            for (int f=0; f <fpixels; f++){
                int i = pid*sdim*fdim + s*fdim + f;
                pfs[i*3] = pid;
                pfs[i*3+1] = f;
                pfs[i*3+2] = s;
            }
        }
    }
    gettimeofday(&t2, 0);
    double time_make_pfs = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

    gettimeofday(&t1,0 );
    add_diffBragg_spots(pfs);
    gettimeofday(&t2, 0);
    double time_diffBragg_spots = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

    if (verbose)
        printf("Time to make PFS: %f, Time to run diffBragg spots: %f\n", time_make_pfs, time_diffBragg_spots);
    return raw_pixels_roi;
}

void diffBragg::add_diffBragg_spots(){
    int fdim = roi_xmax-roi_xmin;
    int sdim = roi_ymax-roi_ymin;
    int npix = fdim*sdim;
    af::shared<size_t> pfs(npix*3);
    for (int s=0; s <spixels; s++){
        for (int f=0; f <fpixels; f++){
            if(f < roi_xmin || f >= roi_xmax || s < roi_ymin || s >= roi_ymax)
                continue;
            int i=(s-roi_ymin)*fdim + (f-roi_xmin);
            //int i = s*fpixels + f;
            pfs[i*3] = detector_panel_id;
            pfs[i*3+1] = f;
            pfs[i*3+2] = s;
        }
    }
    add_diffBragg_spots(pfs);
    double* floatimage = raw_pixels.begin();
    double* floatimage_roi = raw_pixels_roi.begin();
    for (int i=0; i< npix;i++)
        floatimage[i] = floatimage_roi[i];
}


void diffBragg::add_diffBragg_spots(const af::shared<size_t>& panels_fasts_slows, boost::python::list per_pix_nominal_hkl){

    db_cryst.nominal_hkl.clear();
    Npix_to_model = panels_fasts_slows.size()/3;
    SCITBX_ASSERT(Npix_to_model==boost::python::len(per_pix_nominal_hkl));
    // NOTE each element of the list needs to be a 3-tuple
    for (int i_pix=0; i_pix <Npix_to_model; i_pix++){
        boost::python::tuple hkl = boost::python::extract<boost::python::tuple>( per_pix_nominal_hkl[i_pix]);
        int nom_h = boost::python::extract<int>(hkl[0]);
        int nom_k = boost::python::extract<int>(hkl[1]);
        int nom_l = boost::python::extract<int>(hkl[2]);
        db_cryst.nominal_hkl.push_back(nom_h);
        db_cryst.nominal_hkl.push_back(nom_k);
        db_cryst.nominal_hkl.push_back(nom_l);
    }
    add_diffBragg_spots(panels_fasts_slows);
}

np::ndarray diffBragg::add_Fhkl_gradients(const af::shared<size_t>& panels_fasts_slows,
    np::ndarray& residual, np::ndarray& variance, np::ndarray& trusted, np::ndarray& freq,
    int num_Fhkl_channels, double Gscale, bool track, bool errors){

    db_flags.track_Fhkl_indices = track;
    db_flags.Fhkl_errors_mode = errors;

    Npix_to_model = panels_fasts_slows.size()/3;

    double* resid_ptr = reinterpret_cast<double*>(residual.get_data());
    double* var_ptr = reinterpret_cast<double*>(variance.get_data());
    bool* trust_ptr = reinterpret_cast<bool*>(trusted.get_data());
    int* freq_ptr = reinterpret_cast<int*>(freq.get_data());

    int n_data_alloc = first_deriv_imgs.residual.size();
    while (n_data_alloc < Npix_to_model){
        first_deriv_imgs.residual.push_back(0);
        first_deriv_imgs.variance.push_back(0);
        first_deriv_imgs.trusted.push_back(0);
        first_deriv_imgs.freq.push_back(0);
        n_data_alloc ++;
    }

    for (int i=0; i < Npix_to_model; i++){
        double resid = *(resid_ptr+i);
        double var = *(var_ptr+i);
        bool trust = *(trust_ptr+i);
        int fr = *(freq_ptr+i);
        first_deriv_imgs.residual[i] = resid;
        first_deriv_imgs.variance[i] = var;
        first_deriv_imgs.trusted[i] = trust;
        first_deriv_imgs.freq[i] = fr;
    }

    db_flags.Fhkl_gradient_mode = true;
    db_flags.using_trusted_mask = true;
    bool is_empty  = first_deriv_imgs.Fhkl_scale_deriv.empty();
    for (int i=0; i < db_cryst.Num_ASU*num_Fhkl_channels; i ++){
        if (is_empty)
            first_deriv_imgs.Fhkl_scale_deriv.push_back(0);
        else
            first_deriv_imgs.Fhkl_scale_deriv[i] = 0;
    }
    if (db_flags.Fhkl_errors_mode){
        first_deriv_imgs.Fhkl_hessian.clear();
        for (int i=0; i < db_cryst.Num_ASU*num_Fhkl_channels; i++)
            first_deriv_imgs.Fhkl_hessian.push_back(0);
    }

    // we need to supply the spot scale parameter
    SCITBX_ASSERT(spot_scale==1);
    spot_scale=Gscale;
    add_diffBragg_spots(panels_fasts_slows);
    spot_scale=1;

    db_flags.using_trusted_mask = false;
    db_flags.Fhkl_gradient_mode = false;
    db_flags.track_Fhkl_indices = false;
    db_flags.Fhkl_errors_mode = false;

    //printf("3rd sanity check: Fhkl_scale_deriv[16128]=%10.7g\n", first_deriv_imgs.Fhkl_scale_deriv[16128]);
    boost::python::tuple shape = boost::python::make_tuple(db_cryst.Num_ASU*num_Fhkl_channels);
    boost::python::tuple stride = boost::python::make_tuple(sizeof(double));
    np::dtype dt = np::dtype::get_builtin<double>();
    np::ndarray output = np::empty(shape,dt);

    if (errors) {
        output = np::from_data(&first_deriv_imgs.Fhkl_hessian[0], dt, shape, stride, boost::python::object());
    }
    else {
        output = np::from_data(&first_deriv_imgs.Fhkl_scale_deriv[0], dt, shape, stride, boost::python::object());
    }
    return output.copy();
}

void diffBragg::update_Fhkl_scale_factors(np::ndarray& scale_factors, int num_Fhkl_channels){
    unsigned int Nscale = scale_factors.shape(0);
    SCITBX_ASSERT(Nscale==db_cryst.Num_ASU * num_Fhkl_channels);
    bool init_scales=first_deriv_imgs.Fhkl_scale.empty();
    db_flags.Fhkl_have_scale_factors = true;
    db_cryst.num_Fhkl_channels=num_Fhkl_channels;
    db_cu_flags.update_Fhkl_scales=true;

    double* scale_ptr = reinterpret_cast<double*>(scale_factors.get_data());
    for(int i_chan=0; i_chan< num_Fhkl_channels; i_chan++){
        for (int i_hkl=0; i_hkl< db_cryst.Num_ASU;i_hkl++){
            int i = i_hkl + db_cryst.Num_ASU*i_chan;
            double scale =  *(scale_ptr+i);
            if (init_scales)
                first_deriv_imgs.Fhkl_scale.push_back(scale);
            else
                first_deriv_imgs.Fhkl_scale[i] = scale;
        }
    }
}

// BEGIN diffBragg_add_spots
void diffBragg::add_diffBragg_spots(const af::shared<size_t>& panels_fasts_slows){

    TIMERS.recording = record_time;
    // timer variables
    struct timeval t1,t2, t3,t4;
    gettimeofday(&t1,0 );

    Npix_to_model = panels_fasts_slows.size()/3;
    SCITBX_ASSERT(Npix_to_model <= Npix_total);
    double * floatimage_roi = raw_pixels_roi.begin();

    diffBragg_rot_mats();
    /* make sure we are normalizing with the right number of sub-steps */
    gettimeofday(&t3,0 );
    steps = phisteps*mosaic_domains*oversample*oversample;
    subpixel_size = pixel_size/oversample;
    db_steps.Nsteps = oversample*oversample*detector_thicksteps*sources*phisteps*mosaic_domains;
    db_steps.subS_pos = new int[db_steps.Nsteps];
    db_steps.subF_pos = new int[db_steps.Nsteps];
    db_steps.thick_pos = new int[db_steps.Nsteps];
    db_steps.source_pos = new int[db_steps.Nsteps];
    db_steps.phi_pos = new int[db_steps.Nsteps];
    db_steps.mos_pos = new int[db_steps.Nsteps];
    diffBragg_list_steps(db_steps);
    gettimeofday(&t4,0 );
    double time_steps = (1000000.0*(t4.tv_sec-t3.tv_sec) + t4.tv_usec-t3.tv_usec)/1000.0;

    gettimeofday(&t3,0 );
    int pan_rot_ids[3] = {0,4,5};
    int pan_orig_ids[3] = {1,2,3};

    if (db_flags.refine_panel_rot.empty()){ // if one is empty, they're all empty
        db_flags.refine_panel_rot.resize(3, false);
        db_flags.refine_panel_origin.resize(3,false);
        db_flags.refine_lambda.resize(2,false);
        db_flags.refine_Bmat.resize(6,false);
        db_flags.refine_Umat.resize(3,false);
        db_flags.refine_Ncells.resize(3,false);
        db_cu_flags.update_refine_flags=true;
    }

    for(int i_pan=0;i_pan < 3; i_pan++){
        int i_pan_rot = pan_rot_ids[i_pan];
        int i_pan_orig = pan_orig_ids[i_pan];
        if (panels[i_pan_rot]->refine_me != db_flags.refine_panel_rot[i_pan]){
            db_flags.refine_panel_rot[i_pan] = panels[i_pan_rot]->refine_me;
            db_cu_flags.update_refine_flags=true;
        }
        if (panels[i_pan_orig]-> refine_me != db_flags.refine_panel_origin[i_pan_orig]){
            db_flags.refine_panel_origin[i_pan] = panels[i_pan_orig]->refine_me;
            db_cu_flags.update_refine_flags=true;
        }
    }
    for (int i_uc = 0; i_uc < 6; i_uc++){
        if (ucell_managers[i_uc]->refine_me != db_flags.refine_Bmat[i_uc]){
            db_flags.refine_Bmat[i_uc] = ucell_managers[i_uc]->refine_me;
            db_cu_flags.update_refine_flags=true;
        }
    }

    for (int i_rot =0; i_rot< 3; i_rot ++){
        if (rot_managers[i_rot]->refine_me != db_flags.refine_Umat[i_rot]){
            db_flags.refine_Umat[i_rot] = rot_managers[i_rot]->refine_me;
            db_cu_flags.update_refine_flags=true;
        }
    }

    for (int i_lam=0; i_lam< 2; i_lam++){
        if (lambda_managers[i_lam]->refine_me != db_flags.refine_lambda[i_lam]){
            db_flags.refine_lambda[i_lam] = lambda_managers[i_lam]->refine_me;
            db_cu_flags.update_refine_flags=true;
        }
    }

    for (int i_nc=0; i_nc<3; i_nc++){
        if (isotropic_ncells && i_nc >0)
            continue;
        if (Ncells_managers[i_nc]->refine_me != db_flags.refine_Ncells[i_nc]){
            db_flags.refine_Ncells[i_nc] = Ncells_managers[i_nc]->refine_me;
            db_cu_flags.update_refine_flags=true;
        }
    }

    db_flags.refine_fcell = fcell_managers[0]->refine_me;
    db_flags.refine_eta = eta_managers[0]->refine_me;
    db_flags.refine_Ncells_def = refine_Ncells_def;
    db_flags.printout_fpixel = printout_fpixel;
    db_flags.printout_spixel = printout_spixel;
    db_flags.printout = printout;
    db_flags.track_Fhkl = track_Fhkl;
    db_flags.printout = printout;
    db_flags.nopolar = nopolar;
    db_flags.point_pixel = point_pixel;
    db_flags.only_save_omega_kahn = only_save_omega_kahn;
    db_flags.compute_curvatures = compute_curvatures;
    db_flags.isotropic_ncells = isotropic_ncells;
    db_flags.complex_miller = complex_miller;
    db_flags.no_Nabc_scale = no_Nabc_scale;
    db_flags.refine_fp_fdp = fp_fdp_managers[0]->refine_me;
    db_flags.use_lambda_coefficients = use_lambda_coefficients;
    db_flags.oversample_omega = oversample_omega;
    db_flags.printout_fpixel = printout_fpixel;
    db_flags.printout_spixel = printout_spixel;
    db_flags.verbose = verbose;

    db_cryst.mosaic_domains = mosaic_domains;
    db_cryst.default_F = default_F;
    db_cryst.r_e_sqr = r_e_sqr;
    db_cryst.h_max = h_max;
    db_cryst.k_max = k_max;
    db_cryst.l_max = l_max;
    db_cryst.h_min = h_min;
    db_cryst.k_min = k_min;
    db_cryst.l_min = l_min;
    db_cryst.h_range = h_range;
    db_cryst.k_range = k_range;
    db_cryst.l_range = l_range;
    db_cryst.dmin = dmin;
    db_cryst.phi0 = phi0;
    db_cryst.phistep = phistep;
    db_cryst.phisteps = phisteps;
    db_cryst.fudge = fudge;
    db_cryst.spot_scale = spot_scale;
    db_cryst.Na = Na;
    db_cryst.Nb = Nb;
    db_cryst.Nc = Nc;
    db_cryst.Nd = Nd;
    db_cryst.Ne = Ne;
    db_cryst.Nf = Nf;
    db_cryst.xtal_shape = xtal_shape;

    db_beam.number_of_sources = sources;
    db_beam.source_X = source_X;
    db_beam.source_Y = source_Y;
    db_beam.source_Z = source_Z;
    db_beam.source_lambda = source_lambda;
    db_beam.source_I = source_I;
    db_beam.fluence = fluence;
    db_beam.kahn_factor = polarization;
    db_beam.lambda0 = lambda_managers[0]->value;
    db_beam.lambda1 = lambda_managers[1]->value;

    db_det.detector_thickstep = detector_thickstep;
    db_det.detector_thicksteps = detector_thicksteps;
    db_det.detector_thick = detector_thick;
    db_det.detector_attnlen = detector_attnlen;
    db_det.subpixel_size = subpixel_size;
    db_det.pixel_size = pixel_size;
    db_det.oversample = oversample;

    Eigen::Vector3d eig_spindle_vec(spindle_vector[1], spindle_vector[2], spindle_vector[3]);
    Eigen::Vector3d _polarization_axis(polar_vector[1], polar_vector[2], polar_vector[3]);

    db_cryst.dB_Mats.clear();
    db_cryst.dB2_Mats.clear();
    for(int i_uc=0; i_uc< 6; i_uc++){
        db_cryst.dB_Mats.push_back(ucell_managers[i_uc]->dB);
        db_cryst.dB2_Mats.push_back(ucell_managers[i_uc]->dB2);
    }

    std::vector<unsigned int> panels_fasts_slows_vec(panels_fasts_slows.begin(), panels_fasts_slows.begin() + panels_fasts_slows.size()) ;//(panels_fasts_slows.size());
    gettimeofday(&t4,0 );
    double time_other_vecs = (1000000.0*(t4.tv_sec-t3.tv_sec) + t4.tv_usec-t3.tv_usec)/1000.0;

    gettimeofday(&t3,0 );
    image_type image(Npix_to_model,0.0);
    if(db_flags.wavelength_img){
        // TODO: `wavelength` is not a first derivative image, change membership in the future
        first_deriv_imgs.wavelength.resize(4*Npix_to_model,0); // store average wavelength and h,k,l
    }
    if (std::count(db_flags.refine_Umat.begin(), db_flags.refine_Umat.end(), true) > 0){
        first_deriv_imgs.Umat.resize(Npix_to_model*3, 0);
        second_deriv_imgs.Umat.resize(Npix_to_model*3,0);
    }
    if (std::count(db_flags.refine_Bmat.begin(), db_flags.refine_Bmat.end(), true) > 0){
        first_deriv_imgs.Bmat.resize(Npix_to_model*6, 0);
        second_deriv_imgs.Bmat.resize(Npix_to_model*6,0);
    }
    if (std::count(db_flags.refine_Ncells.begin(), db_flags.refine_Ncells.end(), true) > 0 || refine_Ncells_def){
        first_deriv_imgs.Ncells.resize(Npix_to_model*6,0);
        second_deriv_imgs.Ncells.resize(Npix_to_model*6,0);
    }
    if (fcell_managers[0]->refine_me){
        first_deriv_imgs.fcell.resize(Npix_to_model,0);
        second_deriv_imgs.fcell.resize(Npix_to_model,0);
    }
    if (lambda_managers[0]->refine_me || lambda_managers[1]->refine_me){
        first_deriv_imgs.lambda.resize(Npix_to_model*2,0);
        second_deriv_imgs.lambda.resize(Npix_to_model*2,0);
    }
    if(eta_managers[0]->refine_me){
        first_deriv_imgs.eta.resize(Npix_to_model*3,0);
        second_deriv_imgs.eta.resize(Npix_to_model*3,0);
    }
    if (std::count(db_flags.refine_panel_rot.begin(), db_flags.refine_panel_rot.end(), true) > 0){
        first_deriv_imgs.panel_rot.resize(Npix_to_model*3,0);
        second_deriv_imgs.panel_rot.resize(Npix_to_model*3,0);
    }
    if (std::count(db_flags.refine_panel_origin.begin(), db_flags.refine_panel_origin.end(), true) > 0){
        first_deriv_imgs.panel_orig.resize(Npix_to_model*3,0);
        second_deriv_imgs.panel_orig.resize(Npix_to_model*3,0);
    }
    if(fp_fdp_managers[0]->refine_me){
        first_deriv_imgs.fp_fdp.resize(Npix_to_model*2,0);
    }
    if (db_flags.refine_diffuse){
        first_deriv_imgs.diffuse_gamma.resize(Npix_to_model*3,0);
        first_deriv_imgs.diffuse_sigma.resize(Npix_to_model*3,0);
    }
    gettimeofday(&t4,0 );
    double time_make_images = (1000000.0*(t4.tv_sec-t3.tv_sec) + t4.tv_usec-t3.tv_usec)/1000.0;

    db_cryst.spindle_vec = eig_spindle_vec;
    db_beam.polarization_axis = _polarization_axis;

    gettimeofday(&t2, 0);
    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    if (record_time)
        TIMERS.add_spots_pre += time;

    if (verbose){
        printf("Pre kernel: Time to list steps: %f msec\n", time_steps);
        printf("Pre kernel: Time to make other vecs: %f msec\n", time_other_vecs);
        printf("Pre kernel: Time to make images: %f msec\n", time_make_images);
    }

    //fudge = 1.1013986013; // from manuscript computation
    gettimeofday(&t1,0 );
    if ((! use_gpu && getenv("DIFFBRAGG_USE_CUDA")==NULL && getenv("DIFFBRAGG_USE_KOKKOS")==NULL ) || force_cpu){
        diffBragg_sum_over_steps(
            Npix_to_model, panels_fasts_slows_vec,
            image,
            first_deriv_imgs,
            second_deriv_imgs,
            db_steps,
            db_det,
            db_beam,
            db_cryst,
            db_flags);
        last_kernel_on_GPU=false;
        }
    else { // we are using cuda or kokkos
#if defined DIFFBRAGG_HAVE_CUDA || defined DIFFBRAGG_HAVE_KOKKOS
        db_cu_flags.device_Id = device_Id;
        db_cu_flags.update_step_positions = update_step_positions_on_device;
        db_cu_flags.update_panels_fasts_slows = update_panels_fasts_slows_on_device;
        db_cu_flags.update_umats = update_umats_on_device;
        db_cu_flags.update_dB_mats = update_dB_matrices_on_device;
        db_cu_flags.update_rotmats = update_rotmats_on_device;
        //db_cu_flags.update_Fhkl = update_Fhkl_on_device;
        db_cu_flags.update_panel_deriv_vecs = update_panel_deriv_vecs_on_device;
        db_cu_flags.Npix_to_allocate = Npix_to_allocate;

      bool use_cuda = false;
#ifdef DIFFBRAGG_HAVE_CUDA
      use_cuda = use_gpu;
#endif

      if (use_cuda || getenv("DIFFBRAGG_USE_CUDA")!=NULL){
#ifdef DIFFBRAGG_HAVE_CUDA
        diffBragg_sum_over_steps_cuda(
             Npix_to_model, panels_fasts_slows_vec,
             image,
             first_deriv_imgs,
             second_deriv_imgs,
             db_steps,
             db_det,
             db_beam,
             db_cryst,
             db_flags,
             db_cu_flags,
             device_pointers,
             TIMERS);
        if(verbose)
           printf("Ran the CUDA kernel\n");
#else
        bool DIFFBRAGG_USE_CUDA_flag_unsupported=false;
        SCITBX_ASSERT(DIFFBRAGG_USE_CUDA_flag_unsupported);
#endif
      }
      else if (use_gpu || getenv("DIFFBRAGG_USE_KOKKOS")!=NULL){
#ifdef DIFFBRAGG_HAVE_KOKKOS
        if (!diffBragg_runner) {
          diffBragg_runner = std::make_shared<diffBraggKOKKOS>();
        }

        diffBragg_runner->diffBragg_sum_over_steps_kokkos(
            Npix_to_model, panels_fasts_slows_vec,
            image,
            first_deriv_imgs,
            second_deriv_imgs,
            db_steps,
            db_det,
            db_beam,
            db_cryst,
            db_flags,
            db_cu_flags,
            TIMERS);
        if (verbose)
            printf("Ran the Kokkos kernel\n");
#else
        bool DIFFBRAGG_USE_KOKKOS_flag_unsupported=false;
        SCITBX_ASSERT(DIFFBRAGG_USE_KOKKOS_flag_unsupported);
#endif
        }
    last_kernel_on_GPU=true;
#else
    bool DIFFBRAGG_USE_KOKKOS_and_DIFFBRAGG_USE_CUDA_flags_unsupported=false;
    SCITBX_ASSERT(DIFFBRAGG_USE_KOKKOS_and_DIFFBRAGG_USE_CUDA_flags_unsupported);
#endif
    }
    gettimeofday(&t2, 0);
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    if(verbose){
        unsigned long long long_Nsteps = db_steps.Nsteps;
        unsigned long long long_Npix = Npix_to_model;
        unsigned long long n_total_iter =  long_Nsteps*long_Npix;
        printf("Nsteps=%d\noversample=%d\ndet_thick_steps=%d\nsources=%d\nphisteps=%d\nmosaic_domains=%d\n",
                db_steps.Nsteps,oversample,detector_thicksteps,sources,phisteps,mosaic_domains);
        printf("DIFFBRAGG isotropic Ncells=%d\n", isotropic_ncells);
        if(use_gpu || getenv("DIFFBRAGG_USE_CUDA")!= NULL || getenv("DIFFBRAGG_USE_KOKKOS")!= NULL)
            printf("TIME TO RUN DIFFBRAGG -GPU- (%llu iterations):  %3.10f ms \n",n_total_iter, time);
        else
            printf("TIME TO RUN DIFFBRAGG -CPU- (%llu iterations):  %3.10f ms \n",n_total_iter, time);
    }
    if (record_time) TIMERS.add_spots_kernel_wrapper += time;

    gettimeofday(&t1,0 );

    for (int i_pix=0; i_pix< Npix_to_model; i_pix++){
        floatimage_roi[i_pix] = image[i_pix];

        for (int i_rot=0; i_rot<3; i_rot++){
            if (rot_managers[i_rot]->refine_me){
                int idx = i_rot*Npix_to_model + i_pix;
                rot_managers[i_rot]->increment_image(i_pix, first_deriv_imgs.Umat[idx], second_deriv_imgs.Umat[idx], compute_curvatures);
            }
        }
        for (int i_uc=0; i_uc<6; i_uc++){
            if (ucell_managers[i_uc]->refine_me){
                int idx = i_uc*Npix_to_model + i_pix;
                ucell_managers[i_uc]->increment_image(i_pix, first_deriv_imgs.Bmat[idx], second_deriv_imgs.Bmat[idx], compute_curvatures);
            }
        }
        if (Ncells_managers[0]->refine_me){
            Ncells_managers[0]->increment_image(i_pix, first_deriv_imgs.Ncells[i_pix], second_deriv_imgs.Ncells[i_pix], compute_curvatures);
            if (! isotropic_ncells){
                int idx= Npix_to_model+i_pix;
                Ncells_managers[1]->increment_image(i_pix, first_deriv_imgs.Ncells[idx], second_deriv_imgs.Ncells[idx], compute_curvatures);
                idx = 2*Npix_to_model + i_pix;
                Ncells_managers[2]->increment_image(i_pix, first_deriv_imgs.Ncells[idx], second_deriv_imgs.Ncells[idx], compute_curvatures);
            }
        }

        if (refine_Ncells_def){
            for (int i_nc =3; i_nc < 6; i_nc++){
                int idx= i_nc*Npix_to_model+i_pix;
                Ncells_managers[i_nc]->increment_image(i_pix, first_deriv_imgs.Ncells[idx], second_deriv_imgs.Ncells[idx], compute_curvatures);
            }
        }

        if (fcell_managers[0]->refine_me){
            int idx= i_pix;
            fcell_managers[0]->increment_image(i_pix, first_deriv_imgs.fcell[idx], second_deriv_imgs.fcell[idx], compute_curvatures);
        }

        if (eta_managers[0]->refine_me){
            eta_managers[0]->increment_image(i_pix, first_deriv_imgs.eta[i_pix], second_deriv_imgs.eta[i_pix], compute_curvatures);
            if (modeling_anisotropic_mosaic_spread){
                if (verbose && i_pix==0)printf("copying aniso eta derivatives\n");
                for(int i_eta=1; i_eta < 3; i_eta++){
                    int idx = i_eta*Npix_to_model+i_pix;
                    eta_managers[i_eta]->increment_image(i_pix, first_deriv_imgs.eta[idx], second_deriv_imgs.eta[idx], compute_curvatures);
                }
            }
        }

        for(int i_lam=0; i_lam < 2; i_lam++){
            if (lambda_managers[i_lam]->refine_me){
                int idx= Npix_to_model*i_lam + i_pix;
                lambda_managers[i_lam]->increment_image(i_pix, first_deriv_imgs.lambda[idx], second_deriv_imgs.lambda[idx], compute_curvatures);
            }
        }

        for(int i_pan=0; i_pan <3; i_pan++){
            int i_rot = pan_rot_ids[i_pan];
            if (panels[i_rot]->refine_me){
                int idx = Npix_to_model*i_pan + i_pix;
                panels[i_rot]->increment_image(i_pix, first_deriv_imgs.panel_rot[idx], second_deriv_imgs.panel_rot[idx], compute_curvatures);
            }

            int i_orig = pan_orig_ids[i_pan];
            if(panels[i_orig]->refine_me){
                int idx= Npix_to_model*i_pan + i_pix;
                panels[i_orig]->increment_image(i_pix, first_deriv_imgs.panel_orig[idx], second_deriv_imgs.panel_orig[idx], compute_curvatures);
            }
        }

        if (fp_fdp_managers[0]->refine_me)
            fp_fdp_managers[0]->increment_image(i_pix, first_deriv_imgs.fp_fdp[i_pix], 0, compute_curvatures);
        if (fp_fdp_managers[1]->refine_me)
            fp_fdp_managers[1]->increment_image(i_pix, first_deriv_imgs.fp_fdp[i_pix+Npix_to_model], 0, compute_curvatures);

    } // END of flex array update

    delete[] db_steps.subS_pos;
    delete[] db_steps.subF_pos;
    delete[] db_steps.thick_pos;
    delete[] db_steps.source_pos;
    delete[] db_steps.phi_pos;
    delete[] db_steps.mos_pos;

    gettimeofday(&t2, 0);
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    if (record_time) {
        TIMERS.add_spots_post += time;
        TIMERS.timings += 1; // only increment timings at the end of the add_diffBragg_spots call
    }


    if(verbose) printf("done with pixel loop\n");
} // END  of add_diffBragg_spots

void diffBragg::diffBragg_rot_mats(){
    for (int i_rot=0; i_rot < 3; i_rot++){
        //if (rot_managers[i_rot]->refine_me){
            db_cryst.RotMats[i_rot] = rot_managers[i_rot]->R;
            db_cryst.dRotMats[i_rot] = rot_managers[i_rot]->dR;
            db_cryst.d2RotMats[i_rot] = rot_managers[i_rot]->dR2;
        //}
    }
    db_cryst.RXYZ = db_cryst.RotMats[0]*db_cryst.RotMats[1]*db_cryst.RotMats[2];

    /*  update Umats to be U*RXYZ   */
    for(mos_tic=0;mos_tic<mosaic_domains;++mos_tic){
        db_cryst.UMATS_RXYZ[mos_tic] = db_cryst.UMATS[mos_tic] * db_cryst.RXYZ;
        for (int i_eta=0; i_eta<3; i_eta++){
            if (eta_managers[i_eta]->refine_me){
                //printf("setting umat %d in vector of length %d\n" , mos_tic, UMATS_RXYZ_prime.size());
                int mos_tic2 = mosaic_domains*i_eta + mos_tic;
                db_cryst.UMATS_RXYZ_prime[mos_tic2] = db_cryst.UMATS_prime[mos_tic2]*db_cryst.RXYZ;
                if (db_cryst.UMATS_dbl_prime.size() > 0){
                    if (verbose && mos_tic ==0)
                        printf("setting UMATS_RXYZ_dblprime\n");
                    db_cryst.UMATS_RXYZ_dbl_prime[mos_tic2] = db_cryst.UMATS_dbl_prime[mos_tic2]*db_cryst.RXYZ;
                }
            }
        }
    }
}

void diffBragg::diffBragg_list_steps( step_arrays& db_steps){
    /* TODO theres probably a clever way to do this, but oh well */
    int i_step = 0;
    for(subS=0;subS<oversample;++subS){
        for(subF=0;subF<oversample;++subF){
            for(thick_tic=0;thick_tic<detector_thicksteps;++thick_tic){
                for(source=0;source<sources;++source){
                    for(phi_tic = 0; phi_tic < phisteps; ++phi_tic){
                        for(mos_tic=0;mos_tic<mosaic_domains;++mos_tic){
                            db_steps.subS_pos[i_step] = subS;
                            db_steps.subF_pos[i_step] = subF;
                            db_steps.thick_pos[i_step] = thick_tic;
                            db_steps.source_pos[i_step] = source;
                            db_steps.phi_pos[i_step] = phi_tic;
                            db_steps.mos_pos[i_step] = mos_tic;
                            i_step ++;
                        }
                    }
                }
            }
        }
    }
}

void diffBragg::sanity_check_linear_Fhkl(){
        for (int h = 0; h < h_range; h++) {
                for (int k = 0; k < k_range; k++) {
                        for (int l = 0; l < l_range; l++) {
                                SCITBX_ASSERT(db_cryst.FhklLinear[h*k_range*l_range + k*l_range  + l] == Fhkl[h][k][l]);
                        }
                }
        }
}


void diffBragg::update_linear_Fhkl(){
        for (int h = 0; h < h_range; h++) {
                for (int k = 0; k < k_range; k++) {
                        for (int l = 0; l < l_range; l++) {
                                db_cryst.FhklLinear[h*k_range*l_range + k*l_range  + l] = Fhkl[h][k][l];
                        }
                }
        }
}

std::string get_hkl_key(int h, int k , int l){
    std::string hkl_s ;
    hkl_s = std::to_string(h) + ","+ std::to_string(k) + "," + std::to_string(l);
    return hkl_s;
}

void diffBragg::linearize_Fhkl(bool compute_dists){
//      TODO assert eig_O and hall_symbol are properly set before proceeding
        cctbx::sgtbx::space_group sg = cctbx::sgtbx::space_group(db_cryst.hall_symbol);
        Eigen::Matrix3d O_inv = db_cryst.eig_O.inverse();
        if (verbose>1){
            printf("Hall symbol %s\n", db_cryst.hall_symbol.c_str());
            printf("O_inv:\n%.1f %.1f %.1f\n%.1f %.1f %.1f\n%.1f %.1f %.1f\n",
                O_inv(0,0), O_inv(0,1), O_inv(0,2),
                O_inv(1,0), O_inv(1,1), O_inv(1,2),
                O_inv(2,0), O_inv(2,1), O_inv(2,2));
        }

        cctbx::sgtbx::reciprocal_space::asu asu(sg.type());

        cctbx::uctbx::unit_cell ucell;
        if (compute_dists){
            double uc_a = a[0]*1e10;
            double uc_b = b[0]*1e10;
            double uc_c = c[0]*1e10;
            double uc_alpha = alpha*RTD;
            double uc_beta = beta*RTD;
            double uc_gamma =gamma*RTD;
            scitbx::af::double6 uc_par(uc_a, uc_b, uc_c,uc_alpha, uc_beta, uc_gamma);
            ucell = cctbx::uctbx::unit_cell(uc_par);
        }

        db_cryst.ASUid_map.clear();
        db_cryst.FhklLinear_ASUid.clear();
        db_cryst.FhklLinear.clear();
        db_cryst.Fhkl2Linear.clear();
        db_cryst.ASU_dspace.clear();
        db_cryst.ASU_Fcell.clear();

        int asu_count = 0;
        for (int h = 0; h < h_range; h++) {
                for (int k = 0; k < k_range; k++) {
                        for (int l = 0; l < l_range; l++) {
                                double fhkl_val = Fhkl[h][k][l];

                                int h0 = h+ h_min;
                                int k0 = k+ k_min;
                                int l0 = l+ l_min;

                                Eigen::Vector3d HKL((double)h0, (double)k0, (double)l0);
                                HKL = O_inv*HKL;
                                h0 = (int)HKL[0];
                                k0 = (int)HKL[1];
                                l0 = (int)HKL[2];

                                cctbx::miller::index<int> hkl0(h0,k0,l0);
                                cctbx::miller::asym_index ai(sg, asu, hkl0);
                                cctbx::miller::index_table_layout_adaptor ila = ai.one_column(true);
                                cctbx::miller::index<int> hkl0_asu = ila.h();

                                int h_asu = hkl0_asu[0];
                                int k_asu = hkl0_asu[1];
                                int l_asu = hkl0_asu[2];
                                std::string key = get_hkl_key(h_asu, k_asu, l_asu);
                                if (db_cryst.ASUid_map.count(key) == 0){
                                    db_cryst.ASUid_map[key] = asu_count;
                                    asu_count +=1;
                                    if(compute_dists){
                                        double d_spacing = ucell.d(hkl0);
                                        db_cryst.ASU_dspace.push_back(d_spacing);
                                        db_cryst.ASU_Fcell.push_back(fhkl_val);
                                    }
                                }
                                int asu_id = db_cryst.ASUid_map[key];
                                db_cryst.FhklLinear_ASUid.push_back(asu_id);

                                db_cryst.FhklLinear.push_back(fhkl_val);
                                if (complex_miller)
                                    db_cryst.Fhkl2Linear.push_back(Fhkl2[h][k][l]);
                        }
                }
        }
        db_cryst.Num_ASU = asu_count;
}

void diffBragg::show_timing_stats(int MPI_RANK){ //}, boost_adaptbx::python::streambuf & output){
    if (TIMERS.timings > 0){
        printf("Unit is milliseconds\n");
        printf("RANK%d TIMINGS: add_diffBragg_spots pre kernel wrapper: %10.3f\n", MPI_RANK, TIMERS.add_spots_pre );
        printf("RANK%d TIMINGS: add_diffBragg_spots post kernel wrapper: %10.3f\n", MPI_RANK , TIMERS.add_spots_post);
        printf("RANK%d TIMINGS: add_diffBragg_spots kernel wrapper: %10.3f\n", MPI_RANK, TIMERS.add_spots_kernel_wrapper );
        printf("RANK%d TIMINGS: add_diffBragg_spots device alloc: %10.3f\n", MPI_RANK, TIMERS.cuda_alloc );
        printf("RANK%d TIMINGS: add_diffBragg_spots copy host-to-dev: %10.3f\n", MPI_RANK, TIMERS.cuda_copy_to_dev );

        printf("RANK%d TIMINGS: host-to-dev Fhkl scales: %10.3f\n", MPI_RANK, TIMERS.copy_Fhkl_scale );
        printf("RANK%d TIMINGS: host-to-dev sources: %10.3f\n", MPI_RANK, TIMERS.copy_sources );
        printf("RANK%d TIMINGS: host-to-dev umats: %10.3f\n", MPI_RANK, TIMERS.copy_umats );
        printf("RANK%d TIMINGS: host-to-dev amats: %10.3f\n", MPI_RANK, TIMERS.copy_amats );
        printf("RANK%d TIMINGS: host-to-dev bmats: %10.3f\n", MPI_RANK, TIMERS.copy_bmats );
        printf("RANK%d TIMINGS: host-to-dev rotmats: %10.3f\n", MPI_RANK, TIMERS.copy_rotmats );
        printf("RANK%d TIMINGS: host-to-dev det: %10.3f\n", MPI_RANK, TIMERS.copy_det );
        printf("RANK%d TIMINGS: host-to-dev nomhkl: %10.3f\n", MPI_RANK, TIMERS.copy_nomhkl );
        printf("RANK%d TIMINGS: host-to-dev flags: %10.3f\n", MPI_RANK, TIMERS.copy_flags );
        printf("RANK%d TIMINGS: host-to-dev fhkl: %10.3f\n", MPI_RANK, TIMERS.copy_fhkl );
        printf("RANK%d TIMINGS: host-to-dev detderiv: %10.3f\n", MPI_RANK, TIMERS.copy_detderiv );
        printf("RANK%d TIMINGS: host-to-dev pfs: %10.3f\n", MPI_RANK, TIMERS.copy_pfs );

        printf("RANK%d TIMINGS: add_diffBragg_spots copy dev-to-host: %10.3f\n", MPI_RANK, TIMERS.cuda_copy_from_dev );
        printf("RANK%d TIMINGS: add_diffBragg_spots device kernel: %10.3f\n", MPI_RANK, TIMERS.cuda_kernel );
        printf("RANK%d TIMINGS: Total kernel calls=%d\n", MPI_RANK, TIMERS.timings );
    }
    else printf("RANK%d No timing has occured since instantiation of diffBragg\n", MPI_RANK);


}

af::flex_double diffBragg::ave_wavelength_img(){
  int Npix = first_deriv_imgs.wavelength.size()/4;
  af::flex_double wavelen_pixels = af::flex_double(Npix);
  for (int i=0; i <Npix; i++)
    wavelen_pixels[i] = first_deriv_imgs.wavelength[i*4];
  return wavelen_pixels;
}

boost::python::tuple diffBragg::ave_hkl_img(){
  int Npix = first_deriv_imgs.wavelength.size()/4;
  af::flex_double miller_h_pixels = af::flex_double(Npix);
  af::flex_double miller_k_pixels = af::flex_double(Npix);
  af::flex_double miller_l_pixels = af::flex_double(Npix);
  for (int i=0; i <Npix; i++){
    miller_h_pixels[i] = first_deriv_imgs.wavelength[i*4+1];
    miller_k_pixels[i] = first_deriv_imgs.wavelength[i*4+2];
    miller_l_pixels[i] = first_deriv_imgs.wavelength[i*4+3];
  }

  boost::python::tuple result = boost::python::make_tuple(miller_h_pixels, miller_k_pixels, miller_l_pixels);
  return result;
}


boost::python::tuple diffBragg::get_ave_I_cell(bool use_Fhkl_scale, int i_channel, bool use_geometric_mean){
    SCITBX_ASSERT(db_cryst.ASU_dspace.size() > 0 );
    SCITBX_ASSERT(db_cryst.dspace_bins.size()>0);
    db_cryst.use_geometric_mean = use_geometric_mean;
    std::vector<double> ave = I_cell_ave(db_cryst, use_Fhkl_scale, i_channel, first_deriv_imgs.Fhkl_scale);
    boost::python::list ave_out;
    boost::python::list counts_out;
    for (int i=1; i< ave.size()/2; i++) {// skip the first bin, its for out-of-bounds inserts
        ave_out.append(ave[i]);
        counts_out.append(ave[i + ave.size()/2] );
    }
    boost::python::tuple out = boost::python::make_tuple(ave_out, counts_out);
    return out ;
}


np::ndarray diffBragg::Fhkl_restraint_data(
        int i_channel, double Fhkl_beta,
        bool use_geometric_mean, int flag){
    if (flag==0){
        db_cryst.Fhkl_beta = Fhkl_beta;
        db_cryst.use_geometric_mean = use_geometric_mean;
    }
    else if (flag==1){
        db_cryst.Friedel_beta = Fhkl_beta;
    }
    else{
        db_cryst.use_geometric_mean = false;
        db_cryst.Finit_beta = Fhkl_beta;
    }

    std::vector<double> restraint_data;
    for (int i=0; i < db_cryst.Num_ASU+1; i++)
        restraint_data.push_back(0);
    if (flag==0)
        Ih_grad_terms(db_cryst, i_channel, first_deriv_imgs.Fhkl_scale, restraint_data);
    else if (flag==1)
        Friedel_grad_terms(db_cryst, i_channel, first_deriv_imgs.Fhkl_scale, restraint_data);
    else
        Finit_grad_terms(db_cryst, i_channel, first_deriv_imgs.Fhkl_scale, restraint_data);

    boost::python::tuple shape = boost::python::make_tuple(restraint_data.size());
    boost::python::tuple stride = boost::python::make_tuple(sizeof(double));
    np::dtype dt = np::dtype::get_builtin<double>();
    np::ndarray output = np::empty(shape,dt);

    output = np::from_data(&restraint_data[0], dt, shape, stride, boost::python::object());
    return output.copy();
}


void diffBragg::set_Friedel_mate_inds(boost::python::list pos_inds, boost::python::list neg_inds){
    int N = boost::python::len(pos_inds);
    SCITBX_ASSERT(boost::python::len(neg_inds) == N);
    db_cryst.pos_inds.clear();
    db_cryst.neg_inds.clear();
    for (int i=0; i<N; i++){
        int pos = boost::python::extract<int>(pos_inds[i]);
        int neg = boost::python::extract<int>(neg_inds[i]);
        db_cryst.pos_inds.push_back(pos);
        db_cryst.neg_inds.push_back(neg);
    }
}

} // end of namespace nanoBragg
} // end of namespace simtbx
