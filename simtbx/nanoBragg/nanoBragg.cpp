#include <simtbx/nanoBragg/nanoBragg.h>


//Contributed by James Holton, UCSF,LBNL,SLAC.

namespace simtbx {
namespace nanoBragg {

using boost::math::isnan;

/* constructor that takes a dxtbx "panel" detector model */
nanoBragg::nanoBragg(
    const dxtbx::model::Detector& detector, // no default
    const dxtbx::model::Beam& beam,  // no default
    int verbose, // =0
    int panel_id) //=0
{
    double temp;
    vec3 xyz;

    /* initialize to sensible default values */
    init_defaults();

    /* store arg verbosity as member */
    this->verbose=verbose;

    if(verbose) std::cout << detector << std::endl;
    if(verbose) std::cout << beam << std::endl;

    /* BEAM properties first */

    /* direction in 3-space of beam vector */
    xyz = beam.get_unit_s0();
    beam_vector[1] = xyz[0];
    beam_vector[2] = xyz[1];
    beam_vector[3] = xyz[2];
    unitize(beam_vector,beam_vector);

    /* central wavelength, in Angstrom */
    lambda0 = beam.get_wavelength()*1e-10;

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

    /* size of the pixels in meters */
    pixel_size = detector[panel_id].get_pixel_size()[0]/1000.;

    /* pixel count in short and fast-axis directions */
    spixels = detector[panel_id].get_image_size()[1];
    fpixels = detector[panel_id].get_image_size()[0];

    /* allocate all the other arrays */
    init_detector();

    /* direction in 3-space of detector axes */
    beam_convention = CUSTOM;

    set_dxtbx_detector_panel(detector[panel_id], beam.get_s0());

    /* SPINDLE properties */

    /* By default align the rotation axis with the detector fast direction */
    spindle_vector[1] = fdet_vector[1];
    spindle_vector[2] = fdet_vector[2];
    spindle_vector[3] = fdet_vector[3];
    unitize(spindle_vector,spindle_vector);
    user_beam=true;//needed for tilted dxtbx geometries, locks user (dxtbx) geom in place

    /* NOT IMPLEMENTED: read in any other stuff?  */
    /*TODO: consider reading in a crystal model as well, showing params without crystal model can be confusing*/
    if (verbose)
      show_params();

    /* sensible initialization of all unititialized values */
    reconcile_parameters();
    SCITBX_ASSERT(distance > 0);  // safe to check,  especially with multi panel dxtbx models
}


// constructor for the nanoBragg class that takes most any member as an argument, defaults in nanoBragg_ext.cpp
nanoBragg::nanoBragg(
        scitbx::vec2<int> detpixels_slowfast, // = 1024, 1024
        scitbx::vec3<double> Ncells_abc, // 1. 1. 1.
        cctbx::uctbx::unit_cell unitcell, // lysozyme
        vec3 missets_deg, // 0 0 0
        vec2 beam_center_mm, // NAN NAN
        double distance_mm, // =100,
        double pixel_size_mm, // =0.1,
        double wavelength_A, // =1,
        double divergence_mrad, // =0,
        double dispersion_pct, // =0,
        double mosaic_spread_deg, // =-1
        int oversample, // =0 =auto
        int verbose)    // =0
{
    /* initialize to sensible default values */
    init_defaults();

    /* now incorporate arguments to the constructor */
    this->spixels=detpixels_slowfast[0];
    this->fpixels=detpixels_slowfast[1];
    this->Na=Ncells_abc[0]; this->Nb=Ncells_abc[1]; this->Nc=Ncells_abc[2];
    this->a_A[0]=unitcell.parameters()[0];      // units = Angstrom
    this->b_A[0]=unitcell.parameters()[1];
    this->c_A[0]=unitcell.parameters()[2];
    this->alpha=unitcell.parameters()[3]/RTD;   // input units = Degrees
    this->beta=unitcell.parameters()[4]/RTD;
    this->gamma=unitcell.parameters()[5]/RTD;
    this->user_cell=true;
    this->misset[1]=missets_deg[0];
    this->misset[2]=missets_deg[1];
    this->misset[3]=missets_deg[2];
    if(this->misset[1] != 0.0 && this->misset[2] != 0.0 && this->misset[3] != 0.0) this->misset[0]=1;
    if(! isnan(beam_center_mm[0]) && ! isnan(beam_center_mm[1])) {
        this->user_beam = true;
        this->Xbeam = beam_center_mm[0]/1000.0;
        this->Ybeam = beam_center_mm[1]/1000.0;
    }
    this->distance=distance_mm/1000.0;
    this->pixel_size=pixel_size_mm/1000.0;
    this->lambda0=wavelength_A/1e10;
    this->hdivrange=divergence_mrad/1000.0;
    this->vdivrange=divergence_mrad/1000.0;
    this->dispersion=dispersion_pct/100.0;
    this->mosaic_spread=mosaic_spread_deg/RTD;
    this->oversample=oversample;
    if(oversample>0) this->user_oversample=true;
    this->verbose=verbose;

    /* sensible initialization of all unititialized values */
    reconcile_parameters();
}


/* sensible defaults */
void
nanoBragg::reconcile_parameters()
{
    /* now that constructor arguments are applied, fill in blanks */

    /* allocate detector pixel array */
    init_detector();

    /* initialize fluence from flux and check sample size fits inside beam size */
    init_beam();

    /* initialize interpretation of detector and beam geometry */
    init_beamcenter();

    /* sanitize options for steps across divergence, dispersion, mosaic spread and spindle rotation */
    init_steps();

    /* reconcile different conventions of beam center input and detector position */
    update_beamcenter();

    /* set up interpolation if need be */
    init_interpolator();

    /* reconcile different types of input of cell or crystal orientation */
    init_cell();

    /* decide if oversampling is needed based on crystal volume and pixel size */
    update_oversample();

    /* read in structure factors */
    init_Fhkl();

    /* read in centrosymmetric background profile (not implemented) */
    init_background();

    /* display all actual phi values */
    show_phisteps();

    /* display all detector layers */
    show_detector_thicksteps();

    /* read in or generate x-ray source properties */
    init_sources();

    /* allocate and set up mosaic domains */
    init_mosaicity();


    if(verbose) printf("CONSTRUCT!!! %d   x-ray beam: %f %f %f\n",(int) raw_pixels.size(),this->beam_vector[1],this->beam_vector[2],this->beam_vector[3]);
}


/* sensible defaults */
void
nanoBragg::init_defaults()
{
    double test = NAN;
    if(! isnan(test)) {
        if(verbose) printf("ERROR! isnan(NAN) != %g\n",test);
        exit(9);
    };

    /* write cbf files in int precision? */
    cbf_int = false;

    /* optional file stuff, to be removed eventually? */
    matfilename = NULL;
    hklfilename = NULL;
    stolfilename = NULL;
    imginfilename = NULL;
    maskfilename = NULL;
//    stoloutfilename = "output.stol\0";
    sourcefilename = NULL;
//    floatfilename = "floatimage.bin\0";
//    intfilename = "intimage.img\0";
//    pgmfilename = "image.pgm\0";
//    noisefilename = "noiseimage.img\0";
    infile = NULL;
//    FILE *outfile;  // = NULL;
//    FILE *stoloutfile;  // = NULL;

    /* default progress meter stuff */
    progress_pixel=progress_pixels=0;
    progress_meter = 1;
    babble = 1;
    printout = 0;
    printout_spixel=printout_fpixel=-1;
    verbose = 1;
#ifdef NANOBRAGG_HAVE_CUDA
    device_Id = 0; // for running the CUDA code
#endif

    /* default x-ray beam properties */
//    double beam_vector[4] = {0,1,0,0};  this->beam_vector = beam_vector;
    beam_vector[0] = 0;
    beam_vector[1] = 1;
    beam_vector[2] = 0;
    beam_vector[3] = 0;
    coherent = 0;
    far_source = 1;
    round_div = 1;
    lambda=1e-10;
    lambda_of = NULL;
    mosaic_spread=-1.0;
    mosaic_umats = NULL;
    dispersion=0.0;dispstep=-1.0;
    lambda0 = 1.0e-10;
    hdiv=0;hdivstep=-1.0; hdivrange= -1.0;
    vdiv=0;vdivstep=-1.0; vdivrange= -1.0;
    source_path=0;source_distance = 10.0;
    hdiv_tic=vdiv_tic=disp_tic=mos_tic=0;
    divsteps=hdivsteps=vdivsteps=dispsteps=-1;
    mosaic_domains=-1;
    weight=0;
    source=sources=0;
    source_X=source_Y=source_Z=source_I=source_lambda=NULL;
    allocated_sources=0;

    /* Thomson cross section (m^2) */
//    r_e_sqr = 7.94079248018965e-30;
    /* default incident x-ray fluence in photons/m^2 - exactly cancels Thomson cross section */
    fluence = 125932015286227086360700780544.0;
    flux=0.0;exposure=1.0;beamsize=1e-4;

    /* default sample size stuff */
    N=1;
    Na=Nb=Nc=1.0;
    xtalsize_max=xtalsize_a=xtalsize_b=xtalsize_c=0.0;
    reciprocal_pixel_size=0.0;

    xtal_shape = SQUARE;
    fudge=1;
    xtal_size_x   = 0;             /* m */
    xtal_size_y   = 0;             /* m */
    xtal_size_z   = 0;             /* m */
    xtal_density    = 1.0e6;         /* g/m^3 */
    xtal_molecular_weight = 18.0;    /* g/mol */
    xtal_volume=0.0;xtal_molecules = 0.0;
    /* scale factor = F^2*r_e_sqr*fluence*Avogadro*volume*density/molecular_weight
                           m^2     ph/m^2  /mol      m^3   g/m^3    g/mol   */

    /* amorphous material properties for background */
    amorphous_sample_x = 0.0;
    amorphous_sample_y = 0.0;
    amorphous_sample_z = 0.0;
    amorphous_volume = 0.0;
    amorphous_molecular_weight = 18.0;
    amorphous_density = 1.0e6;   // 1 g/cm^3
    amorphous_molecules = 0;
    /* water F = 2.57 in forward direction */

    /* default detector stuff */
    pixel_size = 0.1e-3;
    fpixels=spixels=0;
    pixels=0;
    distance = 100.0e-3;
    detsize_f = 102.4e-3;
    detsize_s = 102.4e-3;
    detector_attnlen= 234e-6; // default to Si at 1 A x-ray wavelength?
    detector_thick=0.0;
    detector_thickstep=0.0;
    detector_thicksteps=-1;
//    double fdet_vector[4]  = {0,0,0,1};  this->fdet_vector = fdet_vector;
    fdet_vector[0] = 0;
    fdet_vector[1] = 0;
    fdet_vector[2] = 0;
    fdet_vector[3] = 1;
//    double sdet_vector[4]  = {0,0,-1,0}; this->sdet_vector  = sdet_vector;
    sdet_vector[0] = 0;
    sdet_vector[1] = 0;
    sdet_vector[2] = -1;
    sdet_vector[3] = 0;
//    double odet_vector[4]  = {0,1,0,0};  this->odet_vector  = odet_vector;
    odet_vector[0] = 0;
    odet_vector[1] = 1;
    odet_vector[2] = 0;
    odet_vector[3] = 0;
//    double pix0_vector[4]  = {0,0,0,0};  this->pix0_vector  = pix0_vector;
    pix0_vector[0] = 0;
    pix0_vector[1] = 0;
    pix0_vector[2] = 0;
    pix0_vector[3] = 0;
    detector_rotx=0.0;detector_roty=0.0;detector_rotz=0.0;
//    double twotheta_axis[4] = {0,0,1,0}; this->twotheta_axis = twotheta_axis;
    twotheta_axis[0] = 0;
    twotheta_axis[1] = 0;
    twotheta_axis[2] = 1;
    twotheta_axis[3] = 0;
    detector_pivot = BEAM;
    beam_convention = MOSFLM;
    detector_twotheta = 0.0;
    airpath=omega_pixel=omega_Rsqr_pixel=omega_sum = 0.0;
    curved_detector = 0;
    point_pixel= 0;
    Xbeam=NAN;Ybeam=NAN;
    Fbeam=NAN;Sbeam=NAN;
    Fdet=Sdet=Odet=Fdet0=Sdet0=0.0;
    Xclose=NAN;Yclose=NAN;close_distance=NAN;
    Fclose=NAN;Sclose=NAN;
    ORGX=NAN;ORGY=NAN;
    detector_is_righthanded=true;
    adc_offset = 40.0;

    /* use these to remember "user" inputs */
    user_beam = false;
    user_distance =false;
    user_mosdomains = false;

    /* scattering vectors */
//    double incident[4] = {0,0,0,0};
//    double diffracted[4],diffracted0[4] = {0,0,0,0};
//    double scattering[4];
    stol=twotheta=theta=0.0;

    /* diffraction geometry stuff */
    costwotheta=sintwotheta=psi=0;
    xd=yd=zd=xd0=yd0=zd0=0.0;
//    double Ewald[4],Ewald0[4],relp[4];
    dmin=0;
    integral_form = 0; // experimental: use integral form of spots to avoid need for oversampling

    /* polarization stuff */
//    double polar_vector[4] = {0,0,0,1}; this->polar_vector = polar_vector;
    polar_vector[0] = 0;
    polar_vector[1] = 0;
    polar_vector[2] = 0;
    polar_vector[3] = 1;
//  vert_vector[4];
    polar=1.0;
    polarization=0.0;
    nopolar = 0;

    /* sampling */
    steps=0;
    roi_xmin=-1;roi_xmax=-1;roi_ymin=-1;roi_ymax=-1;
    oversample = -1;
    recommended_oversample=subS=subF=0;
    user_oversample=false;
    subpixel_size=0.0;

    /* spindle */
    phi0=0.0;phistep=-1.0;osc=-1.0;
    phisteps=-1;
//    double spindle_vector[4] = {0,0,0,1}; this->spindle_vector = spindle_vector;
    spindle_vector[0] = 0;
    spindle_vector[1] = 0;
    spindle_vector[2] = 0;
    spindle_vector[3] = 1;

    /* structure factor representation */
    Fhkl = NULL;
    F000 = 0.0;
    default_F = 0.0;

    /* background stuff */
    default_Fbg = 0.0;
    stol_of = NULL;
    Fbg_of  = NULL;
    allocated_stols = 0;
    nearest=0;
    stol_file_mult=1.0e10;
    ignore_values=0;
    Fmap_pixel=false;

    /* radial median filter stuff */
    pixels_in = NULL;
    bin_of = NULL;
    bin_start = NULL;
    invalid_pixel = NULL;
//    double median,mad,deviate,sign;
//    double sum_arej,avg_arej,sumd_arej,rms_arej,rmsd_arej;
    diffimage = NULL;
    stolimage = NULL;
    Fimage = NULL;

    pythony_indices.clear();
    pythony_amplitudes.clear();
    pythony_stolFbg.clear();
    pythony_source_XYZ.clear();

    /* intensity stats */
    max_I = 0.0;
    max_I_x = 0.0; max_I_y = 0.0;
    photon_scale = 0.0;
    intfile_scale = 0.0;
    pgm_scale = 0.0;
    sumn = 0;
    overloads = 0;

    /* image file data */
    floatimage = NULL;
    maskimage = NULL;
    intimage = NULL;
    pgmimage = NULL;
    allocated_pixels=0;

    /* random number seeds */
    seed = -time((time_t *)0);
//    if(verbose) printf("random number seed = %u\n",seed);
    mosaic_seed = 12345678;
    calib_seed = 123456789;

    /* point-spread function parameters */
    psf_type = FIBER;
    psf_fwhm = 80e-6;
    psf_radius = 0;
    readout_noise = 3.0;
    flicker_noise = 0.0;
    calibration_noise = 0.03;
    spot_scale = 1.0;
    quantum_gain = 1.0;

    /* interpolation defaults: auto-detect */
    interpolate = 2;
    i1=0;i2=0;i3=0;
    sub_Fhkl=NULL;

    /* unit cell stuff */
    user_cell = false;
    user_matrix = false;
//    double a[4] = {0,0,0,0}; this->a = a;
//    double b[4] = {0,0,0,0}; this->b = b;
//    double c[4] = {0,0,0,0}; this->c = c;
    a_A[0]=0;a_A[1]=0;a_A[2]=0;a_A[3]=0;
    b_A[0]=0;b_A[1]=0;b_A[2]=0;b_A[3]=0;
    c_A[0]=0;c_A[1]=0;c_A[2]=0;c_A[3]=0;
    a[0]=0;a[1]=0;a[2]=0;a[3]=0;
    b[0]=0;b[1]=0;b[2]=0;b[3]=0;
    c[0]=0;c[1]=0;c[2]=0;c[3]=0;
    alpha=0.0;beta=0.0;gamma=0.0;
//    double misset[4] = {0,0,0,0}; this->misset = misset;
    misset[0] = 0;
    misset[1] = 0;
    misset[2] = 0;
    misset[3] = 0;
    user_umat = false;
    umat[0]=1;umat[1]=0;umat[2]=0;
    umat[3]=0;umat[4]=1;umat[5]=0;
    umat[6]=0;umat[7]=0;umat[8]=1;

    /* special options */
//    calculate_noise = 1;
//    write_pgm = 1;
//    binary_spots = false;
}
// end of init_defaults



/* initialize detector size from mm or pixel specs */
void
nanoBragg::init_detector()
    {
    /* fill in blanks */
    if(fpixels) {
        detsize_f = pixel_size*fpixels;
    }
    if(spixels) {
        detsize_s = pixel_size*spixels;
    }
    if(verbose){
        printf("fpixels= %d detsize_f/pixel_size=%g roundoff=%g\n",fpixels,detsize_f/pixel_size,ceil(detsize_f/pixel_size-0.5));
    }

    /* sanity check */
    if(allocated_pixels==0 && invalid_pixel!=NULL) {
        printf("ERROR: problem with memory allocation! \n");
        exit(9);
    }

    /* initialize Pythony pixel array */
    this->fpixels = static_cast<int>(ceil(detsize_f/pixel_size-0.5));
    this->spixels = static_cast<int>(ceil(detsize_s/pixel_size-0.5));
    pixels = this->fpixels*this->spixels;

    if(pixels!=allocated_pixels) {
        /* actually allocate memory */
        if(verbose>6) printf("(re)allocating %d %ld-byte doubles for raw_pixels array\n",pixels,sizeof(double));
        raw_pixels = af::flex_double(af::flex_grid<>(spixels,fpixels));

        if(invalid_pixel!=NULL) {
            if(verbose>6) printf("freeing %d %ld-byte bool invalid_pixel at %p\n",pixels,sizeof(bool),invalid_pixel);
            free(invalid_pixel);
        }
        if(bin_of!=NULL) {
            if(verbose>6) printf("freeing %d %ld-byte unsigned int bin_of at %p\n",pixels,sizeof(unsigned int),bin_of);
            free(bin_of);
        }
        if(diffimage!=NULL) {
            if(verbose>6) printf("freeing %d %ld-byte double diffimage at %p\n",2*pixels,sizeof(double),diffimage);
            free(diffimage);
        }
        if(stolimage!=NULL) {
            if(verbose>6) printf("freeing %d %ld-byte double stolimage at %p\n",pixels,sizeof(double),stolimage);
            free(stolimage);
        }
        if(Fimage!=NULL) {
            if(verbose>6) printf("freeing %d %ld-byte double Fimage at %p\n",pixels,sizeof(double),Fimage);
            free(Fimage);
        }
        if(intimage!=NULL) {
            if(verbose>6) printf("freeing %d %ld-byte unsigned short int intimage at %p\n",pixels,sizeof(unsigned short int),intimage);
            free(intimage);
        }
        if(pgmimage!=NULL) {
            if(verbose>6) printf("freeing %d %ld-byte unsigned char invalid_pixel at %p\n",pixels,sizeof(unsigned char),invalid_pixel);
            free(pgmimage);
        }
        if(verbose>6) printf("allocating %d pixels 7 times for image flags\n",pixels);
        invalid_pixel  =         (bool *) calloc(pixels+10,sizeof(bool));
        bin_of         = (unsigned int *) calloc(pixels+10,sizeof(unsigned int));
        diffimage      =       (double *) calloc(2*pixels+10,sizeof(double));
        stolimage      =       (double *) calloc(pixels+10,sizeof(double));
        Fimage         =       (double *) calloc(pixels+10,sizeof(double));
        intimage = (unsigned short int *) calloc(pixels+10,sizeof(unsigned short int));
        pgmimage =      (unsigned char *) calloc(pixels+10,sizeof(unsigned char));
    }

    allocated_pixels = pixels;
}
// end of init_detector


/* initialize fluence from flux and check sample/beam size */
void
nanoBragg::init_beam()
{
    /* get fluence from flux */
    if(flux != 0.0 && exposure > 0.0 && beamsize >= 0){
        fluence = flux*exposure/beamsize/beamsize;
    }
    if(beamsize >= 0){
        if(beamsize < xtal_size_y){
            if(verbose) printf("WARNING: clipping xtal (%lg m high) with beam (%lg m)\n",xtal_size_y,beamsize);
            xtal_size_y = beamsize;
        }
        if(beamsize < xtal_size_z){
            if(verbose) printf("WARNING: clipping xtal (%lg m wide) with beam (%lg m)\n",xtal_size_z,beamsize);
            xtal_size_z = beamsize;
        }
        if(beamsize < amorphous_sample_y){
            if(verbose) printf("WARNING: clipping amorphous sample (%lg m high) with beam (%lg m)\n",amorphous_sample_y,beamsize);
            amorphous_sample_y = beamsize;
        }
        if(beamsize < amorphous_sample_z){
            if(verbose) printf("WARNING: clipping amorphous sample (%lg m wide) with beam (%lg m)\n",amorphous_sample_z,beamsize);
            amorphous_sample_z = beamsize;
        }
    }
    if(exposure > 0.0)
    {
        flux = fluence/exposure*beamsize*beamsize;
    }
    /* straighten up sample properties */
//    volume = sample_x*sample_y*sample_z;
//    molecules = volume*density*Avogadro/molecular_weight;
}
// end of init_beam


/* set up any uninitialized beam centers from others provided */
void
nanoBragg::init_beamcenter()
{
    /* use XDS origin, if provided */
    if(! isnan(ORGX)) Xclose = ORGX*pixel_size;
    if(! isnan(ORGY)) Yclose = ORGY*pixel_size;
    /* default to center of detector */
    if(isnan(Xclose)) Xclose = (detsize_f - pixel_size)/2.0;
    if(isnan(Yclose)) Yclose = (detsize_s + pixel_size)/2.0;
    /* apply defaults for anything not initialized */
    if(isnan(Xbeam)) Xbeam = Xclose;
    if(isnan(Ybeam)) Ybeam = Yclose;
    if(isnan(Fbeam)) Fbeam = Fclose;
    if(isnan(Sbeam)) Sbeam = Sclose;
    if(roi_xmin < 0) roi_xmin = 0;
    if(roi_xmax < 0) roi_xmax = fpixels;
    if(roi_ymin < 0) roi_ymin = 0;
    if(roi_ymax < 0) roi_ymax = spixels;
    progress_pixels = (roi_xmax-roi_xmin+1)*(roi_ymax-roi_ymin+1);

    if(beam_convention == ADXV)
    {
        /* first pixel is at 0,0 pix and pixel_size,pixel_size*npixels mm */
        if(! user_beam)
        {
            Xbeam = (detsize_f + pixel_size)/2.0;
            Ybeam = (detsize_s - pixel_size)/2.0;
        }
           beam_vector[1]=  0;    beam_vector[2]=  0;    beam_vector[3]=  1;
           fdet_vector[1]=  1;    fdet_vector[2]=  0;    fdet_vector[3]=  0;
           sdet_vector[1]=  0;    sdet_vector[2]= -1;    sdet_vector[3]=  0;
           odet_vector[1]=  0;    odet_vector[2]=  0;    odet_vector[3]=  1;
         twotheta_axis[1]= -1;  twotheta_axis[2]=  0;  twotheta_axis[3]=  0;
          polar_vector[1]=  1;   polar_vector[2]=  0;   polar_vector[3]=  0;
        spindle_vector[1]=  1; spindle_vector[2]=  0; spindle_vector[3]=  0;
        Fbeam = Xbeam;
        Sbeam = detsize_s - Ybeam;
        detector_pivot = BEAM;
        if(verbose) printf("beam center in adxv convention\n");
    }
    if(beam_convention == MOSFLM)
    {
        /* first pixel is at 0.5,0.5 pix and pixel_size/2,pixel_size/2 mm */
        if(! user_beam)
        {
            Xbeam = (detsize_s + pixel_size)/2.0;
            Ybeam = (detsize_f + pixel_size)/2.0;
        }
           beam_vector[1]=  1;    beam_vector[2]=  0;    beam_vector[3]=  0;
           odet_vector[1]=  1;    odet_vector[2]=  0;    odet_vector[3]=  0;
           fdet_vector[1]=  0;    fdet_vector[2]=  0;    fdet_vector[3]=  1;
           sdet_vector[1]=  0;    sdet_vector[2]= -1;    sdet_vector[3]=  0;
         twotheta_axis[1]=  0;  twotheta_axis[2]=  0;  twotheta_axis[3]= -1;
          polar_vector[1]=  0;   polar_vector[2]=  0;   polar_vector[3]=  1;
        spindle_vector[1]=  0; spindle_vector[2]=  0; spindle_vector[3]=  1;
        Fbeam = Ybeam + 0.5*pixel_size;
        Sbeam = Xbeam + 0.5*pixel_size;
        detector_pivot = BEAM;
        if(verbose) printf("beam center in mosflm convention\n");
    }
    if(beam_convention == DENZO)
    {
        if(! user_beam)
        {
            Xbeam = (detsize_s + pixel_size)/2.0;
            Ybeam = (detsize_f + pixel_size)/2.0;
        }
           beam_vector[1]=  1;    beam_vector[2]=  0;    beam_vector[3]=  0;
           odet_vector[1]=  1;    odet_vector[2]=  0;    odet_vector[3]=  0;
           fdet_vector[1]=  0;    fdet_vector[2]=  0;    fdet_vector[3]=  1;
           sdet_vector[1]=  0;    sdet_vector[2]= -1;    sdet_vector[3]=  0;
         twotheta_axis[1]=  0;  twotheta_axis[2]=  0;  twotheta_axis[3]= -1;
          polar_vector[1]=  0;   polar_vector[2]=  0;   polar_vector[3]=  1;
        spindle_vector[1]=  0; spindle_vector[2]=  0; spindle_vector[3]=  1;
        Fbeam = Ybeam + 0.0*pixel_size;
        Sbeam = Xbeam + 0.0*pixel_size;
        detector_pivot = BEAM;
        if(verbose) printf("beam center in denzo convention\n");
    }
    if(beam_convention == XDS)
    {
        if(! user_beam)
        {
            Xbeam = Xclose;
            Ybeam = Yclose;
        }
           beam_vector[1]=  0;    beam_vector[2]=  0;    beam_vector[3]=  1;
           fdet_vector[1]=  1;    fdet_vector[2]=  0;    fdet_vector[3]=  0;
           sdet_vector[1]=  0;    sdet_vector[2]=  1;    sdet_vector[3]=  0;
           odet_vector[1]=  0;    odet_vector[2]=  0;    odet_vector[3]=  1;
         twotheta_axis[1]=  1;  twotheta_axis[2]=  0;  twotheta_axis[3]=  0;
          polar_vector[1]=  1;   polar_vector[2]=  0;   polar_vector[3]=  0;
        spindle_vector[1]=  1; spindle_vector[2]=  0; spindle_vector[3]=  0;
        Fbeam = Xbeam;
        Sbeam = Ybeam;
        detector_pivot = SAMPLE;
        if(verbose) printf("beam center in XDS convention\n");
    }
    if(beam_convention == DIALS)
    {
        if(! user_beam)
        {
            Xbeam = Xclose;
            Ybeam = Yclose;
        }
           beam_vector[1]=  0;    beam_vector[2]=  0;    beam_vector[3]= -1;
           fdet_vector[1]=  1;    fdet_vector[2]=  0;    fdet_vector[3]=  0;
           sdet_vector[1]=  0;    sdet_vector[2]= -1;    sdet_vector[3]=  0;
           odet_vector[1]=  0;    odet_vector[2]=  0;    odet_vector[3]= -1;
         twotheta_axis[1]=  1;  twotheta_axis[2]=  0;  twotheta_axis[3]=  0;
          polar_vector[1]=  1;   polar_vector[2]=  0;   polar_vector[3]=  0;
        spindle_vector[1]=  1; spindle_vector[2]=  0; spindle_vector[3]=  0;
        Fbeam = Xbeam;
        Sbeam = Ybeam;
        detector_pivot = SAMPLE;
        if(verbose) printf("beam center in DIALS convention\n");
    }
    if(beam_convention == CUSTOM)
    {
        if(! user_beam)
        {
            Xbeam = Xclose;
            Ybeam = Yclose;
        }
        Fbeam = Xbeam;
        Sbeam = Ybeam;
        Fclose = Xbeam;
        Sclose = Ybeam;
        if(verbose) printf("beam center in custom convention\n");
    }

    /* straighten up vectors */
    unitize(beam_vector,beam_vector);
    unitize(fdet_vector,fdet_vector);
    unitize(sdet_vector,sdet_vector);
    if(unitize(odet_vector,odet_vector) != 1.0)
    {
        if(verbose) printf("WARNING: auto-generating odet_vector\n");
        cross_product(fdet_vector,sdet_vector,odet_vector);
        unitize(odet_vector,odet_vector);
        if (! detector_is_righthanded)
            vector_rescale(odet_vector, odet_vector, -1);
    }
    unitize(polar_vector,polar_vector);
    unitize(spindle_vector,spindle_vector);
    cross_product(beam_vector,polar_vector,vert_vector);
    unitize(vert_vector,vert_vector);

}
// init_beamcenter



/* count up and sanitize steps across divergence, dispersion, mosaic spread and spindle rotation */
void
nanoBragg::init_steps()
{
    /* default sampling logic */
    if(phisteps < 0){
        /* auto-select number of phi steps */
        if(osc < 0.0) {
            /* auto-select osc range */
            if(phistep <= 0.0) {
                /* user doesn't care about anything */
                phisteps = 1;
                osc = 0.0;
                phistep = 0.0;
            } else {
                /* user doesn't care about osc or steps, but specified step */
                osc = phistep;
                phisteps = 2;
            }
        } else {
            /* user-speficied oscillation */
            if(phistep <= 0.0) {
                /* osc specified, but nothing else */
                phisteps = 2;
                phistep = osc/2.0;
            } else {
                /* osc and phi step specified */
                phisteps = static_cast<int>(ceil(osc/phistep));
            }
        }
    } else {
        /* user-specified number of phi steps */
        if(phisteps == 0) phisteps = 1;
        if(osc < 0.0) {
            /* auto-select osc range */
            if(phistep <= 0.0) {
                /* user cares only about number of steps */
                osc = 1.0/RTD;
                phistep = osc/phisteps;
            } else {
                /* user doesn't care about osc, but specified step */
                osc = phistep;
                phisteps = 2;
            }
        } else {
            /* user-speficied oscillation */
            if(phistep < 0.0) {
                /* osc and steps specified */
                phistep = osc/phisteps;
            } else {
                /* everything specified */
            }
        }
    }

    if(hdivsteps <= 0){
        /* auto-select number of steps */
        if(hdivrange < 0.0) {
            /* auto-select range */
            if(hdivstep <= 0.0) {
                /* user doesn't care about anything */
                hdivsteps = 1;
                hdivrange = 0.0;
                hdivstep = 0.0;
            } else {
                /* user specified stepsize and nothing else */
                hdivrange = hdivstep;
                hdivsteps = 2;
            }
        } else {
            /* user-speficied range */
            if(hdivstep <= 0.0) {
                /* range specified, but nothing else */
                hdivstep = hdivrange;
                hdivsteps = 2;
            } else {
                /* range and step specified, but not number of steps */
                hdivsteps = static_cast<int>(ceil(hdivrange/hdivstep));
            }
        }
    } else {
        /* user-specified number of steps */
        if(hdivrange < 0.0) {
            /* auto-select range */
            if(hdivstep <= 0.0) {
                /* user cares only about number of steps */
                hdivrange = 1.0;
                hdivstep = hdivrange/hdivsteps;
            } else {
                /* user doesn't care about range */
                hdivrange = hdivstep;
                hdivsteps = 2;
            }
        } else {
            /* user-speficied range */
            if(hdivstep <= 0.0) {
                /* range and steps specified */
                if(hdivsteps <=1 ) hdivsteps = 2;
                hdivstep = hdivrange/(hdivsteps-1);
            } else {
                /* everything specified */
            }
        }
    }

    if(vdivsteps <= 0){
        /* auto-select number of steps */
        if(vdivrange < 0.0) {
            /* auto-select range */
            if(vdivstep <= 0.0) {
                /* user doesn't care about anything */
                vdivsteps = 1;
                vdivrange = 0.0;
                vdivstep = 0.0;
            } else {
                /* user specified stepsize and nothing else */
                vdivrange = vdivstep;
                vdivsteps = 2;
            }
        } else {
            /* user-speficied range */
            if(vdivstep <= 0.0) {
                /* range specified, but nothing else */
                vdivstep = vdivrange;
                vdivsteps = 2;
            } else {
                /* range and step specified, but not number of steps */
                vdivsteps = static_cast<int>(ceil(vdivrange/vdivstep));
            }
        }
    } else {
        /* user-specified number of steps */
        if(vdivrange < 0.0) {
            /* auto-select range */
            if(vdivstep <= 0.0) {
                /* user cares only about number of steps */
                vdivrange = 1.0;
                vdivstep = vdivrange/vdivsteps;
            } else {
                /* user doesn't care about range */
                vdivrange = vdivstep;
                vdivsteps = 2;
            }
        } else {
            /* user-speficied range */
            if(vdivstep <= 0.0) {
                /* range and steps specified */
                if(vdivsteps <=1 ) vdivsteps = 2;
                vdivstep = vdivrange/(vdivsteps-1);
            } else {
                /* everything specified */
            }
        }
    }


    if(dispsteps <= 0){
        /* auto-select number of steps */
        if(dispersion < 0.0) {
            /* auto-select range */
            if(dispstep <= 0.0) {
                /* user doesn't care about anything */
                dispsteps = 1;
                dispersion = 0.0;
                dispstep = 0.0;
            } else {
                /* user specified stepsize and nothing else */
                dispersion = dispstep;
                dispsteps = 2;
            }
        } else {
            /* user-speficied range */
            if(dispstep <= 0.0) {
                /* range specified, but nothing else */
                dispstep = dispersion;
                dispsteps = 2;
            } else {
                /* range and step specified, but not number of steps */
                dispsteps = static_cast<int>(ceil(dispersion/dispstep));
            }
        }
    } else {
        /* user-specified number of steps */
        if(dispersion < 0.0) {
            /* auto-select range */
            if(dispstep <= 0.0) {
                /* user cares only about number of steps */
                dispersion = 1.0;
                dispstep = dispersion/dispsteps;
            } else {
                /* user doesn't care about range */
                dispersion = dispstep;
                dispsteps = 2;
            }
        } else {
            /* user-speficied range */
            if(dispstep <= 0.0) {
                /* range and steps specified */
                if(dispsteps <=1 ) dispsteps = 2;
                dispstep = dispersion/(dispsteps-1);
            } else {
                /* everything specified */
            }
        }
    }


    if(detector_thicksteps <= 0){
        /* auto-select number of steps */
        if(detector_thick < 0.0) {
            /* auto-select range */
            if(detector_thickstep <= 0.0) {
                /* user doesn't care about anything */
                detector_thicksteps = 1;
                detector_thick = 0.0;
                detector_thickstep = 0.0;
            } else {
                /* user specified stepsize and nothing else */
                detector_thick = detector_thickstep;
                detector_thicksteps = 2;
            }
        } else {
            /* user-speficied range */
            if(detector_thickstep <= 0.0) {
                /* range specified, but nothing else */
                detector_thicksteps = 2;
                detector_thickstep = detector_thick/detector_thicksteps;
            } else {
                /* range and step specified, but not number of steps */
                detector_thicksteps = ceil(detector_thick/detector_thickstep);
            }
        }
    } else {
        /* user-specified number of steps */
        if(detector_thick < 0.0) {
            /* auto-select range */
            if(detector_thickstep <= 0.0) {
                /* user cares only about number of steps */
                detector_thick = 0.5e-6;
                detector_thickstep = detector_thick/detector_thicksteps;
            } else {
                /* user doesn't care about range */
                detector_thick = detector_thickstep;
                detector_thicksteps = 2;
            }
        } else {
            /* user-speficied range */
            if(detector_thickstep <= 0.0) {
                /* range and steps specified */
                if(detector_thicksteps <=1 ) detector_thicksteps = 2;
                detector_thickstep = detector_thick/(detector_thicksteps-1);
            } else {
                /* everything specified */
            }
        }
    }

    if(mosaic_domains <= 0){
        /* auto-select number of domains */
        if(mosaic_spread < 0.0) {
            /* user doesn't care about anything */
            mosaic_domains = 1;
            mosaic_spread = 0.0;
        } else {
            /* user-speficied mosaicity, but not number of domains */
            if(mosaic_spread == 0.0)
            {
                mosaic_domains = 1;
            }
            else
            {
                if(verbose) printf("WARNING: finite mosaicity with only one domain! upping to 10 mosaic domains\n");
                mosaic_domains = 10;
            }
        }
    } else {
        /* user-specified number of domains */
        if(mosaic_spread < 0.0) {
            /* number of domains specified, but no spread? */
            if(verbose) printf("WARNING: no mosaic spread specified.  setting mosaic_domains = 1\n");
            mosaic_spread = 0.0;
            mosaic_domains = 1;
        } else {
            /* user-speficied mosaicity and number of domains */
            if(mosaic_spread == 0.0)
            {
                if(verbose) printf("WARNING: zero mosaic spread specified.  setting mosaic_domains = 1\n");
                mosaic_domains = 1;
            }
        }
    }


    /* sanity checks */
    if(hdivrange <= 0.0 || hdivstep <= 0.0 || hdivsteps <= 0) {
        hdivsteps = 1;
        hdivrange = 0.0;
        hdivstep = 0.0;
    }
    if(vdivrange <= 0.0 || vdivstep <= 0.0 || vdivsteps <= 0) {
        vdivsteps = 1;
        vdivrange = 0.0;
        vdivstep = 0.0;
    }
    if(dispersion <= 0.0 || dispstep <= 0.0 || dispsteps <= 0) {
        dispsteps = 1;
        dispersion = 0.0;
        dispstep = 0.0;
    }
    if(detector_thick <= 0.0 || detector_thickstep <= 0.0 || detector_thicksteps <= 0) {
        detector_thicksteps = 1;
        detector_thick = 0.0;
        detector_thickstep = 0.0;
    }
    if(verbose) printf("dispersion= %g dispstep= %g  dispsteps= %d\n",dispersion,dispstep,dispsteps);

}
// end of init_steps()




/* reconcile different conventions of beam center input and detector position */
void
nanoBragg::update_beamcenter()
{
    /* initialize detector origin from a beam center and distance */
    /* there are different conventions here: mosflm, XDS etc */

    if(beam_convention == ADXV)
    {
        if(verbose) printf("adxv");
    }
    if(beam_convention == MOSFLM)
    {
        if(verbose) printf("mosflm");
    }
    if(beam_convention == XDS)
    {
        if(verbose) printf("xds");
    }
    if(beam_convention == DENZO)
    {
        if(verbose) printf("denzo");
    }
    if(beam_convention == DIALS)
    {
        if(verbose) printf("dials");
    }
    if(beam_convention == CUSTOM)
    {
        if(verbose) printf("custom");
    }
    if(verbose) printf(" convention selected.\n");

    /* first off, what is the relationship between the two "beam centers"? */
    rotate(odet_vector,vector,detector_rotx,detector_roty,detector_rotz);
    ratio = dot_product(beam_vector,vector);
    if(ratio == 0.0) {
        if(verbose) printf("WARNING: beam is parallel to detector surface! \n");
        ratio = DBL_MIN;
    }
    if(isnan(close_distance)) close_distance = fabs(ratio*distance);
    distance = close_distance/ratio;

    if(detector_pivot == SAMPLE){
        if(verbose) printf("pivoting detector around sample\n");
        /* initialize detector origin before rotating detector */
        pix0_vector[1] = -Fclose*fdet_vector[1]-Sclose*sdet_vector[1]+close_distance*odet_vector[1];
        pix0_vector[2] = -Fclose*fdet_vector[2]-Sclose*sdet_vector[2]+close_distance*odet_vector[2];
        pix0_vector[3] = -Fclose*fdet_vector[3]-Sclose*sdet_vector[3]+close_distance*odet_vector[3];

        /* now swing the detector origin around */
        rotate(pix0_vector,pix0_vector,detector_rotx,detector_roty,detector_rotz);
        rotate_axis(pix0_vector,pix0_vector,twotheta_axis,detector_twotheta);
    }
    /* now orient the detector plane */
    rotate(fdet_vector,fdet_vector,detector_rotx,detector_roty,detector_rotz);
    rotate(sdet_vector,sdet_vector,detector_rotx,detector_roty,detector_rotz);
    rotate(odet_vector,odet_vector,detector_rotx,detector_roty,detector_rotz);

    /* also apply orientation part of twotheta swing */
    rotate_axis(fdet_vector,fdet_vector,twotheta_axis,detector_twotheta);
    rotate_axis(sdet_vector,sdet_vector,twotheta_axis,detector_twotheta);
    rotate_axis(odet_vector,odet_vector,twotheta_axis,detector_twotheta);

    /* make sure beam center is preserved */
    if(detector_pivot == BEAM){
        if(verbose) printf("pivoting detector around direct beam spot\n");
        pix0_vector[1] = -Fbeam*fdet_vector[1]-Sbeam*sdet_vector[1]+distance*beam_vector[1];
        pix0_vector[2] = -Fbeam*fdet_vector[2]-Sbeam*sdet_vector[2]+distance*beam_vector[2];
        pix0_vector[3] = -Fbeam*fdet_vector[3]-Sbeam*sdet_vector[3]+distance*beam_vector[3];
    }

    /* what is the point of closest approach between sample and detector? */
    Fclose         = -dot_product(pix0_vector,fdet_vector);
    Sclose         = -dot_product(pix0_vector,sdet_vector);
    close_distance =  dot_product(pix0_vector,odet_vector);

    /* where is the direct beam now? */
    /* difference between beam impact vector and detector origin */
    newvector[1] = close_distance/ratio*beam_vector[1]-pix0_vector[1];
    newvector[2] = close_distance/ratio*beam_vector[2]-pix0_vector[2];
    newvector[3] = close_distance/ratio*beam_vector[3]-pix0_vector[3];
    /* extract components along detector vectors */
    Fbeam = dot_product(fdet_vector,newvector);
    Sbeam = dot_product(sdet_vector,newvector);
    distance = close_distance/ratio;

    /* find origin in XDS convention */
    ORGX=Fclose/pixel_size+0.5;
    ORGY=Sclose/pixel_size+0.5;

    /* find origin in DIALS convention
      from "CBF coordinate definition" at: https://sites.google.com/site/nexuscbf/home/cbf-dictionary
      The standard coordinate frame in imgCIF/CBF aligns the X-axis to the
      principal goniometer axis, chooses the Z-axis to point from the sample
      into the beam.  If the beam is not orthogonal to the X-axis, the Z-axis
      is the component of the vector points into the beam orthogonal to the
      X-axis.  The Y-axis is chosen to complete a right-handed axis system.
    */
    double cbfX[4],cbfY[4],cbfZ[4],negbeam[4];
    /* construct "X axis" */
    unitize(spindle_vector,cbfX);
    /* sample to source */
    vector_scale(beam_vector,negbeam,-1);
    /* construct "Z axis" */
    double comp = dot_product(negbeam,cbfX);
    cbfZ[1] = negbeam[1]-comp*cbfX[1];
    cbfZ[2] = negbeam[2]-comp*cbfX[2];
    cbfZ[3] = negbeam[3]-comp*cbfX[3];
    unitize(cbfZ,cbfZ);
    /* construct "Y axis" */
    cross_product(cbfZ,cbfX,cbfY);
    unitize(cbfY,cbfY);
    /* now get the components */
    dials_origin[1] = 1000.0*dot_product(pix0_vector,cbfX);
    dials_origin[2] = 1000.0*dot_product(pix0_vector,cbfY);
    dials_origin[3] = 1000.0*dot_product(pix0_vector,cbfZ);

    /* find the beam in the detector frame, for XDS */
    newvector[1] = dot_product(beam_vector,fdet_vector);
    newvector[2] = dot_product(beam_vector,sdet_vector);
    newvector[3] = dot_product(beam_vector,odet_vector);
    if(verbose) printf("XDS incident beam: %g %g %g\n",newvector[1],newvector[2],newvector[3]);
}
// end of update_beamcenter()


void nanoBragg::set_dxtbx_detector_panel(const dxtbx::model::Panel& panel, const vec3& s0_vector){

    int spixels_new = panel.get_image_size()[1];
    int fpixels_new = panel.get_image_size()[0];
    if (spixels_new != spixels || fpixels_new != fpixels){
        printf("New panel has different dimension than allocated panel!\n");
        pixel_size = panel.get_pixel_size()[0]/1000.;
        spixels = panel.get_image_size()[1];
        fpixels = panel.get_image_size()[0];
        init_detector();
    }

    /* typically: 1 0 0 */
    fdet_vector[1] = panel.get_fast_axis()[0];
    fdet_vector[2] = panel.get_fast_axis()[1];
    fdet_vector[3] = panel.get_fast_axis()[2];
    unitize(fdet_vector,fdet_vector);
    /* typically: 0 -1 0 */
    sdet_vector[1] = panel.get_slow_axis()[0];
    sdet_vector[2] = panel.get_slow_axis()[1];
    sdet_vector[3] = panel.get_slow_axis()[2];
    unitize(sdet_vector,sdet_vector);
    /* set orthogonal vector to the detector pixel array */
    cross_product(fdet_vector,sdet_vector,odet_vector);
    unitize(odet_vector,odet_vector);

    /* dxtbx origin is location of outer corner of the first pixel */
    pix0_vector[1] = panel.get_origin()[0]/1000.0;
    pix0_vector[2] = panel.get_origin()[1]/1000.0;
    pix0_vector[3] = panel.get_origin()[2]/1000.0;
    /* what is the point of closest approach between sample and detector? */
    Fclose = Xclose = -dot_product(pix0_vector,fdet_vector);
    Sclose = Yclose = -dot_product(pix0_vector,sdet_vector);
    close_distance = distance =  dot_product(pix0_vector,odet_vector);
    if (close_distance < 0){
        if(verbose)printf("WARNING: dxtbx model seems to be lefthanded. Inverting odet_vector.\n");
        vector_rescale(odet_vector, odet_vector, -1);
        close_distance = distance =  dot_product(pix0_vector,odet_vector);
        detector_is_righthanded = false;
    }

    /* set beam centre */
    scitbx::vec2<double> dials_bc = panel.get_beam_centre(s0_vector);
    Xbeam = dials_bc[0]/1000.0;
    Ybeam = dials_bc[1]/1000.0;

    user_beam = true; // necessary for tilted dxtbx geometries

    /* detector sensor layer properties */
    detector_thick   = panel.get_thickness();
    double temp = panel.get_mu();        // is this really a mu? or mu/rho ?
    if(temp>0.0) detector_attnlen = 1.0/temp;

    /* quantum_gain = amp_gain * electrooptical_gain, does not include capture_fraction */
    quantum_gain = panel.get_gain();
    //adc_offset = detector[panel_id].ADC_OFFSET;
    }


/* automatically decide if we are interpolating, allocate memory if yes */
void
nanoBragg::init_interpolator()
{
    /* free any previous allocations */
    if(sub_Fhkl != NULL) {
        for (h0=0; h0<=5;h0++) {
            for (k0=0; k0<=5;k0++) {
                if(verbose>6) printf("freeing %d %ld-byte double Fhkl[%d][%d] at %p\n",5,sizeof(double),h0,k0,sub_Fhkl[h0][k0]);
                free(*(*(sub_Fhkl +h0)+k0));
            }
            if(verbose>6) printf("freeing %d %ld-byte double* Fhkl[%d] at %p\n",5,sizeof(double*),h0,sub_Fhkl[h0]);
            free(*(sub_Fhkl +h0));
        }
        if(verbose>6) printf("freeing %d %ld-byte double** Fhkl at %p\n",5,sizeof(double**),sub_Fhkl);
        free(sub_Fhkl);
        sub_Fhkl = NULL;
    }

    if(interpolate > 1){
        /* no user options */
        if(( Na <= 2) || (Nb <= 2) || (Nc <= 2)){
            if(verbose) printf("auto-selected tricubic interpolation of structure factors\n");
            interpolate = 1;
        }
        else
        {
            if(verbose) printf("auto-selected no interpolation\n");
            interpolate = 0;
        }
    }
    if(interpolate){
        /* allocate interpolation array */
        sub_Fhkl = (double***) calloc(6,sizeof(double**));
        for (h0=0; h0<=5;h0++) {
            *(sub_Fhkl +h0) = (double**) calloc(6,sizeof(double*));
            for (k0=0; k0<=5;k0++) {
                *(*(sub_Fhkl +h0)+k0) = (double*) calloc(6,sizeof(double));
            }
        }
    }
}
// end of init_interpolator()




/* compute A matrix from cell, missets, mosflm matrix file, etc. */
void
nanoBragg::init_cell()
{
    /* do not run if things are not going to work */
    if(! user_cell && ! user_matrix && matfilename == NULL)
    {
        if(verbose) printf("ERROR: cannot initialize without a cell\n");
        return;
    }
    if(verbose>1) printf("raw CELL %f %f %f %f %f %f   %d\n",a_A[0],b_A[0],c_A[0],alpha,beta,gamma,user_cell);
    /* user-specified unit cell */
    if(user_cell)
    {
        /* a few random defaults */
        if(b_A[0]  <= 0.0) b_A[0] = a_A[0];
        if(c_A[0]  <= 0.0) c_A[0] = a_A[0];
        if(alpha <= 0.0) alpha = M_PI/2;
        if(beta  <= 0.0) beta  = M_PI/2;
        if(gamma <= 0.0) gamma = M_PI/2;

        /* get cell volume from angles */
        aavg = (alpha+beta+gamma)/2;
        skew = sin(aavg)*sin(aavg-alpha)*sin(aavg-beta)*sin(aavg-gamma);
        if(skew<0.0) skew=-skew;
        V_cell = 2.0*a_A[0]*b_A[0]*c_A[0]*sqrt(skew);
        if(V_cell <= 0.0)
        {
            if(verbose) printf("WARNING: impossible unit cell volume: %g\n",V_cell);
            V_cell = DBL_MIN;
        }
        V_star = 1.0/V_cell;

        /* now get reciprocal-cell lengths from the angles and volume */
        a_star[0] = b_A[0]*c_A[0]*sin(alpha)*V_star;
        b_star[0] = c_A[0]*a_A[0]*sin(beta)*V_star;
        c_star[0] = a_A[0]*b_A[0]*sin(gamma)*V_star;
        if(a_star[0] <= 0.0 || b_star[0] <= 0.0 || c_star[0] <= 0.0)
        {
            if(verbose) printf("WARNING: impossible reciprocal cell lengths: %g %g %g\n",
                a_star[0],b_star[0],c_star[0]);
            /* make up something non-zero */
            a_star[0] = fabs(a_star[0]);
            b_star[0] = fabs(b_star[0]);
            c_star[0] = fabs(c_star[0]);
            if(a_star[0] <= 0.0) a_star[0] = DBL_MIN;
            if(b_star[0] <= 0.0) b_star[0] = DBL_MIN;
            if(c_star[0] <= 0.0) c_star[0] = DBL_MIN;
        }

        /* for fun, compute the reciprocal-cell angles from direct-cell angles */
        sin_alpha_star = a_A[0]*V_star/b_star[0]/c_star[0];
        sin_beta_star  = b_A[0]*V_star/a_star[0]/c_star[0];
        sin_gamma_star = c_A[0]*V_star/a_star[0]/b_star[0];
        cos_alpha_star = (cos(beta)*cos(gamma)-cos(alpha))/(sin(beta)*sin(gamma));
        cos_beta_star  = (cos(gamma)*cos(alpha)-cos(beta))/(sin(gamma)*sin(alpha));
        cos_gamma_star = (cos(alpha)*cos(beta)-cos(gamma))/(sin(alpha)*sin(beta));
        if(sin_alpha_star>1.0000001 || sin_alpha_star<-1.0000001 ||
           sin_beta_star >1.0000001 || sin_beta_star <-1.0000001 ||
           sin_gamma_star>1.0000001 || sin_gamma_star<-1.0000001 ||
           cos_alpha_star>1.0000001 || cos_alpha_star<-1.0000001 ||
           cos_beta_star >1.0000001 || cos_beta_star <-1.0000001 ||
           cos_gamma_star>1.0000001 || cos_gamma_star<-1.0000001 )
        {
            if(verbose) printf("WARNING: oddball reciprocal cell angles:\n");
            if(verbose) printf("sin(alpha_star) = %.25g\n",sin_alpha_star);
            if(verbose) printf("cos(alpha_star) = %.25g\n",cos_alpha_star);
            if(verbose) printf("sin(beta_star)  = %.25g\n",sin_beta_star);
            if(verbose) printf("cos(beta_star)  = %.25g\n",cos_beta_star);
            if(verbose) printf("sin(gamma_star) = %.25g\n",sin_gamma_star);
            if(verbose) printf("cos(gamma_star) = %.25g\n",cos_gamma_star);
        }
        if(sin_alpha_star>1.0) sin_alpha_star=1.0;
        if(sin_beta_star >1.0) sin_beta_star =1.0;
        if(sin_gamma_star>1.0) sin_gamma_star=1.0;
        if(sin_alpha_star<-1.0) sin_alpha_star=-1.0;
        if(sin_beta_star <-1.0) sin_beta_star =-1.0;
        if(sin_gamma_star<-1.0) sin_gamma_star=-1.0;
        if(cos_alpha_star*cos_alpha_star>1.0) cos_alpha_star=1.0;
        if(cos_beta_star *cos_beta_star >1.0) cos_beta_star=1.0;
        if(cos_gamma_star*cos_gamma_star>1.0) cos_gamma_star=1.0;
        alpha_star = atan2(sin_alpha_star,cos_alpha_star);
        beta_star  = atan2(sin_beta_star ,cos_beta_star );
        gamma_star = atan2(sin_gamma_star,cos_gamma_star);

        if(verbose>1) printf("CELL %f %f %f %f %f %f   %d\n",a_A[0],b_A[0],c_A[0],alpha,beta,gamma,user_cell);

        /* construct default orientation */
        a_star[1] = a_star[0];
        b_star[1] = b_star[0]*cos_gamma_star;
        c_star[1] = c_star[0]*cos_beta_star;
        a_star[2] = 0.0;
        b_star[2] = b_star[0]*sin_gamma_star;
        c_star[2] = c_star[0]*(cos_alpha_star-cos_beta_star*cos_gamma_star)/sin_gamma_star;
        a_star[3] = 0.0;
        b_star[3] = 0.0;
        c_star[3] = c_star[0]*V_cell/(a_A[0]*b_A[0]*c_A[0]*sin_gamma_star);
    }

    /* load the lattice orientation (reciprocal cell vectors) from a mosflm matrix */
    if(matfilename != NULL)
    {
        infile = fopen(matfilename,"r");
        if(infile != NULL)
        {
            if(verbose) printf("reading %s\n",matfilename);
            if(! fscanf(infile,"%lg%lg%lg",a_star+1,b_star+1,c_star+1)) {perror("fscanf");};
            if(! fscanf(infile,"%lg%lg%lg",a_star+2,b_star+2,c_star+2)) {perror("fscanf");};
            if(! fscanf(infile,"%lg%lg%lg",a_star+3,b_star+3,c_star+3)) {perror("fscanf");};
            fclose(infile);

            /* mosflm A matrix includes the wavelength, so remove it */
            /* calculate reciprocal cell lengths, store in 0th element */
            vector_scale(a_star,a_star,1e-10/lambda0);
            vector_scale(b_star,b_star,1e-10/lambda0);
            vector_scale(c_star,c_star,1e-10/lambda0);
        }
    }

    if(verbose>1) printf("MISSET %f %f %f \n",misset[1],misset[2],misset[3]);
    /* apply any user-provided unitary rotation matrix? */
    if(user_umat)
    {
        rotate_umat(a_star,a_star,umat);
        rotate_umat(b_star,b_star,umat);
        rotate_umat(c_star,c_star,umat);
    }

    if(verbose>1) printf("MISSET %f %f %f \n",misset[1],misset[2],misset[3]);
    /* apply any missetting angle */
    if(misset[0] > 0.0)
    {
        rotate(a_star,a_star,misset[1],misset[2],misset[3]);
        rotate(b_star,b_star,misset[1],misset[2],misset[3]);
        rotate(c_star,c_star,misset[1],misset[2],misset[3]);
    }

    /* various cross products */
    cross_product(a_star,b_star,a_star_cross_b_star);
    cross_product(b_star,c_star,b_star_cross_c_star);
    cross_product(c_star,a_star,c_star_cross_a_star);

    /* reciprocal lattice vector "a_star" is defined as perpendicular to both b and c, and must also preserve volume
       converse is true for direct-space lattice: a is perpendicular to both b_star and c_star
       a = ( b_star cross c_star ) / V_star    */

    /* reciprocal unit cell volume, but is it lambda-corrected? */
    V_star = dot_product(a_star,b_star_cross_c_star);

    /* make sure any user-supplied cell takes */
    if(user_cell)
    {
        /* a,b,c and V_cell were generated above */

        /* force the cross-product vectors to have proper magnitude: b_star X c_star = a*V_star */
        vector_rescale(b_star_cross_c_star,b_star_cross_c_star,a_A[0]/V_cell);
        vector_rescale(c_star_cross_a_star,c_star_cross_a_star,b_A[0]/V_cell);
        vector_rescale(a_star_cross_b_star,a_star_cross_b_star,c_A[0]/V_cell);
        V_star = 1.0/V_cell;
    }

    /* direct-space cell volume in A^3 */
    V_cell = 1.0/V_star;

    /* generate direct-space cell vectors, also updates magnitudes */
    vector_scale(b_star_cross_c_star,a_A,V_cell);
    vector_scale(c_star_cross_a_star,b_A,V_cell);
    vector_scale(a_star_cross_b_star,c_A,V_cell);

    /* now that we have direct-space vectors, re-generate the reciprocal ones */
    cross_product(a_A,b_A,a_cross_b);
    cross_product(b_A,c_A,b_cross_c);
    cross_product(c_A,a_A,c_cross_a);
    vector_scale(b_cross_c,a_star,V_star);
    vector_scale(c_cross_a,b_star,V_star);
    vector_scale(a_cross_b,c_star,V_star);

    /* for fun, calculate the cell angles too */
    sin_alpha = a_star[0]*V_cell/b_A[0]/c_A[0];
    sin_beta  = b_star[0]*V_cell/a_A[0]/c_A[0];
    sin_gamma = c_star[0]*V_cell/a_A[0]/b_A[0];
    cos_alpha = dot_product(b_A,c_A)/b_A[0]/c_A[0];
    cos_beta  = dot_product(a_A,c_A)/a_A[0]/c_A[0];
    cos_gamma = dot_product(a_A,b_A)/a_A[0]/b_A[0];
    if(sin_alpha>1.0000001 || sin_alpha<-1.0000001 ||
       sin_beta >1.0000001 || sin_beta <-1.0000001 ||
       sin_gamma>1.0000001 || sin_gamma<-1.0000001 ||
       cos_alpha>1.0000001 || cos_alpha<-1.0000001 ||
       cos_beta >1.0000001 || cos_beta <-1.0000001 ||
       cos_gamma>1.0000001 || cos_gamma<-1.0000001 )
    {
        if(verbose) printf("WARNING: oddball cell angles:\n");
            if(verbose) printf("sin_alpha = %.25g\n",sin_alpha);
            if(verbose) printf("cos_alpha = %.25g\n",cos_alpha);
            if(verbose) printf("sin_beta  = %.25g\n",sin_beta);
            if(verbose) printf("cos_beta  = %.25g\n",cos_beta);
            if(verbose) printf("sin_gamma = %.25g\n",sin_gamma);
            if(verbose) printf("cos_gamma = %.25g\n",cos_gamma);
    }
    if(sin_alpha>1.0) sin_alpha=1.0;
    if(sin_beta >1.0) sin_beta =1.0;
    if(sin_gamma>1.0) sin_gamma=1.0;
    if(sin_alpha<-1.0) sin_alpha=-1.0;
    if(sin_beta <-1.0) sin_beta =-1.0;
    if(sin_gamma<-1.0) sin_gamma=-1.0;
    if(cos_alpha*cos_alpha>1.0) cos_alpha=1.0;
    if(cos_beta *cos_beta >1.0) cos_beta=1.0;
    if(cos_gamma*cos_gamma>1.0) cos_gamma=1.0;
    alpha = atan2(sin_alpha,cos_alpha);
    beta  = atan2(sin_beta ,cos_beta );
    gamma = atan2(sin_gamma,cos_gamma);


    /* reciprocal cell angles */
    sin_alpha_star = a_A[0]*V_star/b_star[0]/c_star[0];
    sin_beta_star  = b_A[0]*V_star/a_star[0]/c_star[0];
    sin_gamma_star = c_A[0]*V_star/a_star[0]/b_star[0];
    cos_alpha_star = dot_product(b_star,c_star)/b_star[0]/c_star[0];
    cos_beta_star  = dot_product(a_star,c_star)/a_star[0]/c_star[0];
    cos_gamma_star = dot_product(a_star,b_star)/a_star[0]/b_star[0];
    if(sin_alpha_star>1.0000001 || sin_alpha_star<-1.0000001 ||
       sin_beta_star >1.0000001 || sin_beta_star <-1.0000001 ||
       sin_gamma_star>1.0000001 || sin_gamma_star<-1.0000001 ||
       cos_alpha_star>1.0000001 || cos_alpha_star<-1.0000001 ||
       cos_beta_star >1.0000001 || cos_beta_star <-1.0000001 ||
       cos_gamma_star>1.0000001 || cos_gamma_star<-1.0000001 )
    {
            if(verbose) printf("WARNING: oddball reciprocal cell angles:\n");
            if(verbose) printf("sin(alpha_star) = %.25g\n",sin_alpha_star);
            if(verbose) printf("cos(alpha_star) = %.25g\n",cos_alpha_star);
            if(verbose) printf("sin(beta_star)  = %.25g\n",sin_beta_star);
            if(verbose) printf("cos(beta_star)  = %.25g\n",cos_beta_star);
            if(verbose) printf("sin(gamma_star) = %.25g\n",sin_gamma_star);
            if(verbose) printf("cos(gamma_star) = %.25g\n",cos_gamma_star);
    }
    if(sin_alpha_star>1.0) sin_alpha_star=1.0;
    if(sin_beta_star >1.0) sin_beta_star =1.0;
    if(sin_gamma_star>1.0) sin_gamma_star=1.0;
    if(sin_alpha_star<-1.0) sin_alpha_star=-1.0;
    if(sin_beta_star <-1.0) sin_beta_star =-1.0;
    if(sin_gamma_star<-1.0) sin_gamma_star=-1.0;
    if(cos_alpha_star*cos_alpha_star>1.0) cos_alpha_star=1.0;
    if(cos_beta_star *cos_beta_star >1.0) cos_beta_star=1.0;
    if(cos_gamma_star*cos_gamma_star>1.0) cos_gamma_star=1.0;
    alpha_star = atan2(sin_alpha_star,cos_alpha_star);
    beta_star  = atan2(sin_beta_star ,cos_beta_star );
    gamma_star = atan2(sin_gamma_star,cos_gamma_star);

    if(verbose) printf("Unit Cell: %g %g %g %g %g %g\n", a_A[0],b_A[0],c_A[0],alpha*RTD,beta*RTD,gamma*RTD);
    if(verbose) printf("Recp Cell: %g %g %g %g %g %g\n", a_star[0],b_star[0],c_star[0],alpha_star*RTD,beta_star*RTD,gamma_star*RTD);
    if(verbose) printf("volume = %g A^3\n",V_cell);

    /* print out the real-space matrix */
    if(verbose) printf("real-space cell vectors (Angstrom):\n");
    if(verbose) printf("     %-10s  %-10s  %-10s\n","a","b","c");
    if(verbose) printf("X: %11.8f %11.8f %11.8f\n",a_A[1],b_A[1],c_A[1]);
    if(verbose) printf("Y: %11.8f %11.8f %11.8f\n",a_A[2],b_A[2],c_A[2]);
    if(verbose) printf("Z: %11.8f %11.8f %11.8f\n",a_A[3],b_A[3],c_A[3]);
    if(verbose) printf("reciprocal-space cell vectors (Angstrom^-1):\n");
    if(verbose) printf("     %-10s  %-10s  %-10s\n","a_star","b_star","c_star");
    if(verbose) printf("X: %11.8f %11.8f %11.8f\n",a_star[1],b_star[1],c_star[1]);
    if(verbose) printf("Y: %11.8f %11.8f %11.8f\n",a_star[2],b_star[2],c_star[2]);
    if(verbose) printf("Z: %11.8f %11.8f %11.8f\n",a_star[3],b_star[3],c_star[3]);

    /* now convert Angstrom to meters */
    vector_scale(a_A,a,1e-10);
    vector_scale(b_A,b,1e-10);
    vector_scale(c_A,c,1e-10);

    /* define phi=0 mosaic=0 crystal orientation */
    vector_scale(a,a0,1.0);
    vector_scale(b,b0,1.0);
    vector_scale(c,c0,1.0);

    /* define phi=0 crystal orientation */
    vector_scale(a,ap,1.0);
    vector_scale(b,bp,1.0);
    vector_scale(c,cp,1.0);
}
// end of init_cell()


/* compute sample size and required oversampling */
void
nanoBragg::update_oversample()
{
    /* now we know the cell, calculate crystal size in meters */
    if(xtal_size_x > 0) Na = xtal_size_x/a[0];
    if(xtal_size_y > 0) Nb = xtal_size_y/b[0];
    if(xtal_size_z > 0) Nc = xtal_size_z/c[0];
    if(Na <= 1.0) Na = 1.0;
    if(Nb <= 1.0) Nb = 1.0;
    if(Nc <= 1.0) Nc = 1.0;
    xtalsize_a = a[0]*Na;
    xtalsize_b = b[0]*Nb;
    xtalsize_c = c[0]*Nc;
    if(verbose) printf("crystal is %g x %g x %g microns\n",xtalsize_a*1e6,xtalsize_b*1e6,xtalsize_c*1e6);
    xtalsize_max = xtalsize_a;
    if(xtalsize_max < xtalsize_b) xtalsize_max = xtalsize_b;
    if(xtalsize_max < xtalsize_c) xtalsize_max = xtalsize_c;
    reciprocal_pixel_size = lambda0*distance/pixel_size;
    recommended_oversample = static_cast<int>(ceil(3.0 * xtalsize_max/reciprocal_pixel_size));
    if(recommended_oversample <= 0) recommended_oversample = 1;
    if(! user_oversample) {
        oversample = recommended_oversample;
        if(verbose) printf("auto-selected %d-fold oversampling\n",oversample);
    }
    if(oversample < recommended_oversample)
    {
        if(verbose)
        {
        printf("WARNING: maximum dimension of xtal is %g A\n",xtalsize_max*1e10);
        printf("         but reciprocal pixel size is %g A\n", reciprocal_pixel_size*1e10 );
        printf("         intensity may vary significantly across a pixel!\n");
            printf("         recommend oversample=%d to work around this\n",recommended_oversample);
        }
    }

    /* rough estimate of sample properties */
    xtal_size_x = xtalsize_a;
    xtal_size_y = xtalsize_b;
    xtal_size_z = xtalsize_c;
    xtal_volume = xtal_size_x*xtal_size_y*xtal_size_z;
    xtal_density = 1.2e6;
    xtal_molecules = Na*Nb*Nc;
    xtal_molecular_weight = xtal_volume*xtal_density*Avogadro/xtal_molecules;
    if(verbose) printf("approximate MW = %g\n",xtal_molecular_weight);

}
// end of update_oversample()




/* read in structure factors vs hkl */
void
nanoBragg::init_Fhkl()
{
    /* free any previous allocations */
    if(Fhkl != NULL) {
        for (h0=0; h0<=h_range;h0++) {
            for (k0=0; k0<=k_range;k0++) {
                if(verbose>6) printf("freeing %d %ld-byte double Fhkl[%d][%d] at %p\n",l_range+1,sizeof(double),h0,k0,Fhkl[h0][k0]);
                free(Fhkl[h0][k0]);
            }
            if(verbose>6) printf("freeing %d %ld-byte double* Fhkl[%d] at %p\n",k_range+1,sizeof(double*),h0,Fhkl[h0]);
            free(Fhkl[h0]);
        }
        if(verbose>6) printf("freeing %d %ld-byte double** Fhkl at %p\n",h_range+1,sizeof(double**),Fhkl);
        free(Fhkl);
    }
    hkls = 0;

    /* load the structure factors */
    if(hklfilename == NULL)
    {
        /* try to recover Fs from a previous run? */
        h_min = h_max = 0;
        k_min = k_max = 0;
        l_min = l_max = 0;
    }
    else
    {
        infile = fopen(hklfilename,"r");
        if(infile == NULL)
        {
            if(verbose) printf("ERROR: unable to open %s.",hklfilename);
            exit(9);
        }
        h_min=k_min=l_min=static_cast<int>(1e9);
        h_max=k_max=l_max=static_cast<int>(-1e9);
        if(verbose) printf("counting entries in %s\n",hklfilename);
        while(4 == fscanf(infile,"%lg%lg%lg%lg",&h,&k,&l,&F_cell)){
            if(verbose && h != ceil(h-0.4)) printf("WARNING: non-integer value for h (%g) at line %d\n",h,hkls);
            if(verbose && k != ceil(k-0.4)) printf("WARNING: non-integer value for k (%g) at line %d\n",k,hkls);
            if(verbose && l != ceil(l-0.4)) printf("WARNING: non-integer value for l (%g) at line %d\n",l,hkls);
            if(h_min > h) h_min = static_cast<int>(h);
            if(k_min > k) k_min = static_cast<int>(k);
            if(l_min > l) l_min = static_cast<int>(l);
            if(h_max < h) h_max = static_cast<int>(h);
            if(k_max < k) k_max = static_cast<int>(k);
            if(l_max < l) l_max = static_cast<int>(l);
            ++hkls;
        }
        rewind(infile);
    }

    if(pythony_indices.size())
    {
        if(verbose) printf(" noticed pythony_indices.size() = %ld\n",pythony_indices.size());
        /* need to know how much memory to allocate for Fhkl array */
        h_min=k_min=l_min=1e9;
        h_max=k_max=l_max=-1e9;
        miller_t hkl;
        for (i=0; i < pythony_indices.size(); ++i)
        {
            hkl = pythony_indices[i];
            if(pythony_amplitudes.size()) F_cell = pythony_amplitudes[i];
            h0 = hkl[0];
            k0 = hkl[1];
            l0 = hkl[2];
            if(verbose>9) printf("DEBUG: hkl# %d : %d %d %d = %g\n",i,h0,k0,l0,F_cell);
            if(h_min > h0) h_min = h0;
            if(k_min > k0) k_min = k0;
            if(l_min > l0) l_min = l0;
            if(h_max < h0) h_max = h0;
            if(k_max < k0) k_max = k0;
            if(l_max < l0) l_max = l0;
            ++hkls;
        }
    }

    if(1)
    {
        h_range = h_max - h_min + 1;
        k_range = k_max - k_min + 1;
        l_range = l_max - l_min + 1;

        if(verbose) printf("h: %d - %d\n",h_min,h_max);
        if(verbose) printf("k: %d - %d\n",k_min,k_max);
        if(verbose) printf("l: %d - %d\n",l_min,l_max);
        if(h_range < 0 || k_range < 0 || l_range < 0) {
            if(verbose) printf("ERROR: not enough HKL indices in %s\n",hklfilename);
            exit(9);
        }

        /* allocate memory for 3d arrays */
        if(verbose>6) printf("allocating %d %ld-byte double**\n",h_range+1,sizeof(double**));
        Fhkl = (double***) calloc(h_range+1,sizeof(double**));
        if(Fhkl==NULL){perror("ERROR");exit(9);};
        for (h0=0; h0<=h_range;h0++) {
            if(verbose>6) printf("allocating %d %ld-byte double*\n",k_range+1,sizeof(double*));
            Fhkl[h0] = (double**) calloc(k_range+1,sizeof(double*));
            if(Fhkl[h0]==NULL){perror("ERROR");exit(9);};
            for (k0=0; k0<=k_range;k0++) {
            if(verbose>6) printf("allocating %d %ld-byte double\n",l_range+1,sizeof(double));
                Fhkl[h0][k0] = (double*) calloc(l_range+1,sizeof(double));
                if(Fhkl[h0][k0]==NULL){perror("ERROR");exit(9);};
            }
        }
        if(verbose) printf("initializing to default_F = %g:\n",default_F);
        for (h0=0; h0<h_range;h0++) {
            for (k0=0; k0<k_range;k0++) {
                for (l0=0; l0<l_range;l0++) {
                    if(verbose>9) printf("initializing %d %d %d to default_F = %g:\n",h0,k0,l0,default_F);
                    Fhkl[h0][k0][l0] = default_F;
                }
            }
        }
        if(verbose) printf("done initializing to default_F:\n");
        if(verbose) printf("settting F000 to %g\n",F000);
        Fhkl[-h_min][-k_min][-l_min] = F000;
    }

    if(hklfilename != NULL)
    {
        if(verbose) printf("re-reading %s\n",hklfilename);
        while(4 == fscanf(infile,"%d%d%d%lg",&h0,&k0,&l0,&F_cell))
        {
            Fhkl[h0-h_min][k0-k_min][l0-l_min]=F_cell;
        }
        fclose(infile);
    }

    if(pythony_indices.size() && pythony_amplitudes.size())
    {
        if(verbose) printf("initializing Fhkl with pythony indices and amplitudes\n");
        miller_t hkl;
        for (i=0; i < pythony_indices.size(); ++i)
        {
            hkl = pythony_indices[i];
            F_cell = pythony_amplitudes[i];
            h0 = hkl[0];
            k0 = hkl[1];
            l0 = hkl[2];
            Fhkl[h0-h_min][k0-k_min][l0-l_min]=F_cell;
            if(verbose>9) printf("F %d : %d %d %d = %g\n",i,h0,k0,l0,F_cell);
        }
        if(verbose) printf("settting F000 to %g\n",F000);
        Fhkl[-h_min][-k_min][-l_min] = F000;
        if(verbose) printf("done initializing Fhkl:\n");
    }
}
// end of init_Fhkl





/* read in or generate background profile */
void
nanoBragg::init_background()
{
    int i;
    /* straighten up amorphous material properties */
    if(amorphous_sample_y<=0.0) amorphous_sample_y = beamsize;
    if(amorphous_sample_z<=0.0) amorphous_sample_z = beamsize;
    amorphous_volume = amorphous_sample_x*amorphous_sample_y*amorphous_sample_z;
    if(amorphous_density==0 && amorphous_molecules!=0 && amorphous_volume!=0 )
    {
        amorphous_density = amorphous_molecules/amorphous_volume/Avogadro*amorphous_molecular_weight;
    }
    amorphous_molecules = amorphous_volume*amorphous_density*Avogadro/amorphous_molecular_weight;
    if(verbose>1) printf("amorphous_molecules= %g in beam: %g m^3 * %g g/m^3 * %g /mol / %g g/mol\n",
             amorphous_molecules,amorphous_volume,amorphous_density,Avogadro,amorphous_molecular_weight);

    /* now read in amorphous material structure factors */
    stols = 0;

    /* option to read from a text file */
    if(stolfilename != NULL)
    {
        if(verbose) printf("reading %s\n",stolfilename);
        stols = read_text_file(stolfilename,2,&stol_of,&Fbg_of);
        if(stols == 0){
            perror("no data in input file");
            exit(9);
        }
        allocated_stols=stols;
    }

    /* initialize from python flex array of vec2s */
    if(pythony_stolFbg.size())
    {
        /* we will need enough space for this */
        stols = pythony_stolFbg.size();
    }

    if(allocated_stols != stols)
    {
        /* free any previous allocations */
        if(stol_of != NULL) {
            if(verbose>6) printf("freeing %d %ld-byte double stol_of at %p\n",stols,sizeof(double),stol_of);
            free(stol_of);
        }
        if(Fbg_of != NULL) {
            if(verbose>6) printf("freeing %d %ld-byte double Fbg_of at %p\n",stols,sizeof(double),Fbg_of);
            free(Fbg_of);
        }
        if(bin_start != NULL){
            if(verbose>6) printf("freeing %d %ld-byte double *s from bin_start at %p\n",stols,sizeof(double *),bin_start);
            free(bin_start);
        }
        if(pixels_in != NULL){
            if(verbose>6) printf("freeing %d %ld-byte unsigned ints from pixels_in at %p\n",stols,sizeof(unsigned int),pixels_in);
            free(pixels_in);
        }

        if(verbose>6) printf("allocating %d %ld-byte doubles for stol_of\n",stols,sizeof(double));
        stol_of = (double *) calloc(stols+10,sizeof(double));
        if(verbose>6) printf("allocating %d %ld-byte doubles for Fbg_of\n",stols,sizeof(double));
        Fbg_of  = (double *) calloc(stols+10,sizeof(double));

        /* allocate memory for counting how many of these get used */
        /* starting point for pixel value data for each stol-bin */
        if(verbose>6) printf("allocating %d %ld-byte double *s for bin_start\n",stols,sizeof(double *));
        bin_start = (double **) calloc(stols+2,sizeof(double *));
        /* storage for counting number of pixels in each bin */
        if(verbose>6) printf("allocating %d %ld-byte unsigned ints for pixels_in\n",stols,sizeof(unsigned int));
        pixels_in = (unsigned int *) calloc(stols,sizeof(unsigned int));
        allocated_stols = stols;
    }

    if(pythony_stolFbg.size())
    {
        if(verbose) printf("initializing Fbg with pythony array %d\n",stols);
        for (i=0; i < stols; ++i)
        {
            stol_of[i] = pythony_stolFbg[i][0];
            Fbg_of[i]  = pythony_stolFbg[i][1];
            if(verbose>6) printf("Fbg # %d ( stol= %g ) = %g\n",i,stol_of[i],Fbg_of[i]);
        }
        /* flag to add padding on top and bottom */
        stol_of[stols] = NAN;
        if(verbose) printf("done initializing stol_of and Fbg_of\n");
    }

    if(stols == 0 && amorphous_volume != 0.0)
    {
        /* do something clever here */

    }

    if(stols > 0 && isnan(stol_of[stols]))
    {
        if(verbose>9) printf("shifting F_bg vs stol array up by 2\n");
        /* clear the flag */
        stol_of[stols] = 0.0;
        /* add two values at either end for interpolation */
        stols += 4;
        Fbg_highangle = NAN;
        for(i=stols-3;i>1;--i){
            /* shift data up, and convert stol to meters while we are at it */
            stol_of[i] = stol_of[i-2] * stol_file_mult;
            Fbg_of[i]   = Fbg_of[i-2];
            if(! isnan(Fbg_of[i])) {
                /* keep track of lowest-angle valid value */
                Fbg_lowangle = Fbg_of[i];
                if(isnan(Fbg_highangle)) {
                    /* also keep track of highest angle value */
                    Fbg_highangle = Fbg_of[i];
                }
            }
            else
            {
                /* missing values are set to default */
                Fbg_of[i] = default_Fbg;
            }
        }

        /* to turn interpolation into extrapolation, bottom two values should be out at stol=negative infinity */
        stol_of[0] = -1e99;
        stol_of[1] = -1e98;
        Fbg_of[0] = Fbg_of[1] = Fbg_lowangle;
        /* to turn interpolation into extrapolation, bottom two values should be out at stol=infinity */
        stol_of[stols-2] = 1e98;
        stol_of[stols-1] = 1e99;
        Fbg_of[stols-1] = Fbg_of[stols-2] = Fbg_highangle;
    }
}


/* print out actual phi values in sweep */
void
nanoBragg::show_phisteps()
{
    /* show phi steps with sweep over spindle axis */
    for(phi_tic = 0; phi_tic < phisteps; ++phi_tic){
        phi = phi0 + phistep*phi_tic;
        if(verbose) printf("phi%d = %g\n",phi_tic,phi*RTD);
    }
}


/* print out actual detector layers in sweep */
void
nanoBragg::show_detector_thicksteps()
{
    /* print out detector sensor thickness with sweep over all sensor layers */
    for(thick_tic=0;thick_tic<detector_thicksteps;++thick_tic){
        if(verbose) printf("thick%d = %g um\n",thick_tic,detector_thickstep*thick_tic*1e6);
    }
}


/* read in or generate x-ray sources */
void
nanoBragg::init_sources()
{
    /* import sources from user file */
    sources = 0;
    if(sourcefilename != NULL) {
        sources = read_text_file(sourcefilename,5,&source_X,&source_Y,&source_Z,&source_I,&source_lambda);
        if(sources == 0) {
            perror("reading source definition file");
            exit(9);
        }
        /* apply defaults to missing values */
        for(source=0;source<sources;++source){
            if(isnan(source_X[source])) {
                source_X[source] = -source_distance*beam_vector[1];
            }
            if(isnan(source_Y[source])) {
                source_Y[source] = -source_distance*beam_vector[2];
            }
            if(isnan(source_Z[source])) {
                source_Z[source] = -source_distance*beam_vector[3];
            }
            if(isnan(source_I[source])) {
                source_I[source] = 1.0;
            }
            if(isnan(source_lambda[source])) {
                source_lambda[source] = lambda0;
            }
        }
    }



    if(pythony_beams.size())
    {
        if(verbose>8) printf("pythony_beams.size()= %ld\n",pythony_beams.size());
        sources = pythony_beams.size();
        if(verbose>8) printf("total sources: %d\n",sources);
        if(allocated_sources != sources && sources>0)
        {
            /* free any previous allocation */
            if(source_X != NULL) free(source_X);
            if(source_Y != NULL) free(source_Y);
            if(source_Z != NULL) free(source_Z);
            if(source_I != NULL) free(source_I);
            if(source_lambda != NULL) free(source_lambda);

            /* allocate enough space */
            if(verbose>6) printf("allocating space for %d sources\n",sources);
            source_X = (double *) calloc(sources+10,sizeof(double));
            source_Y = (double *) calloc(sources+10,sizeof(double));
            source_Z = (double *) calloc(sources+10,sizeof(double));
            source_I = (double *) calloc(sources+10,sizeof(double));
            source_lambda = (double *) calloc(sources+10,sizeof(double));
            allocated_sources = sources;
        }

        if(verbose) printf("initializing sources with pythony sources\n");
        vec3 xyz,beamdir=vec3(0,0,0),polarvec = vec3(0,0,0);
        double flux_sum = 0.0, lambda_sum = 0.0, polar_sum = 0.0, div_sum = 0.0;
        for (i=0; i < sources; ++i)
        {
            xyz = pythony_beams[i].get_sample_to_source_direction();
            source_X[i] = xyz[0];
            source_Y[i] = xyz[1];
            source_Z[i] = xyz[2];
            source_I[i] = pythony_beams[i].get_flux();
            if(isnan(source_I) || source_I[i]==0.) source_I[i] = 1.0/sources;
            flux_sum += source_I[i];
            source_lambda[i] = pythony_beams[i].get_wavelength();
            if(isnan(source_lambda[i]) || source_lambda[i]==0.) source_lambda[i] = lambda0;
            lambda_sum += source_lambda[i];
            if(verbose>8) printf("source %d :  xyz= ( %g %g %g ), I= %g lambda=%g\n",i,source_X[i],source_Y[i],source_Z[i],source_I[i],source_lambda[i]);

            /* average quantities that we store as one value */
            div_sum += pythony_beams[i].get_divergence();
            polar_sum += pythony_beams[i].get_polarization_fraction();
            polarvec += pythony_beams[i].get_polarization_normal();
            beamdir += xyz;
        }
        /* update averaged parameters */
        if(lambda_sum>0.0) lambda0 = lambda_sum/sources;
        polarization = polar_sum/sources;
        hdivrange=vdivrange=div_sum/sources;
        vert_vector[1] = polarvec[0];vert_vector[2] = polarvec[1];vert_vector[3] = polarvec[2];
        beam_vector[1] = beamdir[0]; beam_vector[2] = beamdir[1]; beam_vector[3] = beamdir[2];
        unitize(beam_vector,beam_vector);
        unitize(vert_vector,vert_vector);
        cross_product(beam_vector,vert_vector,polar_vector);
        unitize(polar_vector,polar_vector);


        /* take in total flux */
        if(flux_sum > 0)
        {
            flux = flux_sum;
            init_beam();
        }
        /* make sure stored source intensities are fractional */
        double norm = flux_sum/sources;
        for (i=0; i < sources && norm>0.0; ++i)
        {
            source_I[i] /= norm;
        }

        if(verbose) printf("done initializing sources:\n");
    }



    if(pythony_source_XYZ.size() || pythony_source_intensity.size() || pythony_source_lambda.size())
    {
        if(verbose>8) printf("pythony_source_XYZ.size()= %ld\n",pythony_source_XYZ.size());
        if(verbose>8) printf("pythony_source_intensity.size()= %ld\n",pythony_source_intensity.size());
        if(verbose>8) printf("pythony_source_lambda.size()= %ld\n",pythony_source_lambda.size());
        sources = pythony_source_XYZ.size();
        if(sources < pythony_source_intensity.size()) sources = pythony_source_intensity.size();
        if(sources < pythony_source_lambda.size()) sources = pythony_source_lambda.size();
        if(verbose>8) printf("total sources: %d\n",sources);
        if(allocated_sources != sources && sources>0)
        {
            /* free any previous allocation */
            if(source_X != NULL) free(source_X);
            if(source_Y != NULL) free(source_Y);
            if(source_Z != NULL) free(source_Z);
            if(source_I != NULL) free(source_I);
            if(source_lambda != NULL) free(source_lambda);

            /* allocate enough space */
            if(verbose>6) printf("allocating space for %d sources\n",sources);
            source_X = (double *) calloc(sources+10,sizeof(double));
            source_Y = (double *) calloc(sources+10,sizeof(double));
            source_Z = (double *) calloc(sources+10,sizeof(double));
            source_I = (double *) calloc(sources+10,sizeof(double));
            source_lambda = (double *) calloc(sources+10,sizeof(double));
            allocated_sources = sources;
        }

        /* make sure sizes match, or else? */
        pythony_source_XYZ.resize(sources);
        pythony_source_intensity.resize(sources);
        pythony_source_lambda.resize(sources);

        if(verbose) printf("initializing sources with pythony sources\n");
        vec3 xyz;
        for (i=0; i < pythony_source_XYZ.size(); ++i)
        {
            xyz = pythony_source_XYZ[i];
            source_X[i] = xyz[0];
            source_Y[i] = xyz[1];
            source_Z[i] = xyz[2];
            source_I[i] = pythony_source_intensity[i];
            if(isnan(source_I[i]) || source_I[i]==0.) source_I[i] = 1.0/sources;
            source_lambda[i] = pythony_source_lambda[i];
            if(isnan(source_lambda[i]) || source_lambda[i]==0.) source_lambda[i] = lambda0;
            if(verbose>8) printf("source %d :  xyz= ( %g %g %g ), I= %g lambda=%g\n",i,source_X[i],source_Y[i],source_Z[i],source_I[i],source_lambda[i]);
        }
        if(verbose) printf("done initializing sources:\n");
    }


    if(sources == 0)
    {
        /* generate generic list of sources */

        /* count divsteps sweep over solid angle of beam divergence */
        divsteps = 0;
        for(hdiv_tic=0;hdiv_tic<hdivsteps;++hdiv_tic){
            for(vdiv_tic=0;vdiv_tic<vdivsteps;++vdiv_tic){
                hdiv = hdivstep * hdiv_tic - hdivrange/2.0 ;
                vdiv = vdivstep * vdiv_tic - vdivrange/2.0 ;
                /* force an elliptical divergence */
                test = (hdiv*hdiv-hdivstep*hdivstep/4.0*(1-hdivsteps%2))/hdivrange/hdivrange ;
                test += (vdiv*vdiv-vdivstep*vdivstep/4.0*(1-vdivsteps%2))/vdivrange/vdivrange ;
                if( round_div && test*4.0 > 1.1) continue;

                ++divsteps;
                if(verbose) printf("divergence deviation: %g %g\n",hdiv,vdiv);
            }
        }

        /* print out wavelength steps with sweep over spectral dispersion */
        for(disp_tic=0;disp_tic<dispsteps;++disp_tic){
            lambda = lambda0 * ( 1.0 + dispstep * disp_tic - dispersion/2.0 ) ;
            if(verbose) printf("lambda%d = %.15g\n",disp_tic,lambda);
        }
        if(verbose) printf("dispersion= %g dispstep= %g  dispsteps= %d\n",dispersion,dispstep,dispsteps);

        /* free any previous allocation */
        if(source_X != NULL) free(source_X);
        if(source_Y != NULL) free(source_Y);
        if(source_Z != NULL) free(source_Z);
        if(source_I != NULL) free(source_I);
        if(source_lambda != NULL) free(source_lambda);

        /* allocate enough space */
        sources = divsteps*dispsteps;
        if(verbose>6) printf("allocating space for %d sources\n",sources);
        source_X = (double *) calloc(sources+10,sizeof(double));
        source_Y = (double *) calloc(sources+10,sizeof(double));
        source_Z = (double *) calloc(sources+10,sizeof(double));
        source_I = (double *) calloc(sources+10,sizeof(double));
        source_lambda = (double *) calloc(sources+10,sizeof(double));

        /* now actually create the source entries */
        weight = 1.0/sources;
        sources = 0;
        for(hdiv_tic=0;hdiv_tic<hdivsteps;++hdiv_tic){
            for(vdiv_tic=0;vdiv_tic<vdivsteps;++vdiv_tic){
                hdiv = hdivstep * hdiv_tic - hdivrange/2.0 ;
                vdiv = vdivstep * vdiv_tic - vdivrange/2.0 ;
                /* force an elliptical divergence */
                test = (hdiv*hdiv-hdivstep*hdivstep/4.0*(1-hdivsteps%2))/hdivrange/hdivrange ;
                test += (vdiv*vdiv-vdivstep*vdivstep/4.0*(1-vdivsteps%2))/vdivrange/vdivrange ;
                if( round_div && test*4.0 > 1.1) continue;

                /* construct unit vector along "beam" */
                vector[1] = -source_distance*beam_vector[1];
                vector[2] = -source_distance*beam_vector[2];
                vector[3] = -source_distance*beam_vector[3];
                /* divergence is in angle space */
                /* define "horizontal" as the E-vector of the incident beam */
                rotate_axis(vector,newvector,polar_vector,vdiv);
                rotate_axis(newvector,vector,vert_vector,hdiv);

                /* one source at each position for each wavelength */
                for(disp_tic=0;disp_tic<dispsteps;++disp_tic){
                    lambda = lambda0 * ( 1.0 + dispstep * disp_tic - dispersion/2.0 ) ;

                    source_X[sources] = vector[1];
                    source_Y[sources] = vector[2];
                    source_Z[sources] = vector[3];
                    source_I[sources] = weight;
                    source_lambda[sources] = lambda;
                    ++sources;
                }
            }
        }
    }
    if(verbose) printf("  created a total of %d sources:\n",sources);
    for(source=0;source<sources;++source){

        /* retrieve stuff from cache */
        X = source_X[source];
        Y = source_Y[source];
        Z = source_Z[source];
        I = source_I[source];
        lambda = source_lambda[source];

        if(verbose) printf("%g %g %g   %g %.6g\n",X,Y,Z,I,lambda);
    }
}
// end of init_sources()





/* expose internal source data */
void
nanoBragg::show_sources()
{
    printf("  created a total of %d sources:\n",sources);
    for(source=0;source<sources;++source){

        /* retrieve stuff from cache */
        X = source_X[source];
        Y = source_Y[source];
        Z = source_Z[source];
        I = source_I[source];
        lambda = source_lambda[source];

        printf("%d %g %g %g   %g %.6g\n",source,X,Y,Z,I,lambda);
    }
    printf("pythony sources:\n");
    for(source=0;source<pythony_source_lambda.size();++source){

        /* retrieve stuff from cache */
        X = pythony_source_XYZ[source][0];
        Y = pythony_source_XYZ[source][1];
        Z = pythony_source_XYZ[source][2];
        I = pythony_source_intensity[source];
        lambda = pythony_source_lambda[source];

        printf("pythony: %d %g %g %g   %g %.6g\n",source,X,Y,Z,I,lambda);
    }
}
// end of init_sources()




/* read in? or generate mosaic domains */
void
nanoBragg::init_mosaicity()
{
    /* temporary local seed */
    long mseed;

    /* catch nks initialized values here? */
    if(user_mosdomains) return;

    /* free any previous allocations */
    if(mosaic_umats!=NULL) free(mosaic_umats);

    /* allocate enough space */
    if(mosaic_domains<1) mosaic_domains=1;
    if(verbose>6) printf("allocating enough space for %d mosaic domain orientation matrices\n",mosaic_domains);
    mosaic_umats = (double *) calloc(mosaic_domains+10,9*sizeof(double));

    /* re-initialize the RNG for the mosaic sequence */
    mseed = -labs(mosaic_seed);

    /* now actually create the orientation of each domain */
    for(mos_tic=0;mos_tic<mosaic_domains;++mos_tic)
    {
        if(mos_tic==0)
        {
            /* force at least one domain to be "aligned"? */
            mosaic_umats[0]=1.0;mosaic_umats[1]=0.0;mosaic_umats[2]=0.0;
            mosaic_umats[3]=0.0;mosaic_umats[4]=1.0;mosaic_umats[5]=0.0;
            mosaic_umats[6]=0.0;mosaic_umats[7]=0.0;mosaic_umats[8]=1.0;
        }
        else
        {
            mosaic_rotation_umat(mosaic_spread, mosaic_umats+9*mos_tic, &mseed);
        }
    }

    if(verbose) printf("  created a total of %d mosaic domains\n",mosaic_domains);
}
// end of init_mosaicity()




/* print out individual mosaic domain orientations */
void
nanoBragg::show_mosaic_blocks()
{
    /* assume init_mosaicity() was already run? */
    for(mos_tic=0;mos_tic<mosaic_domains;++mos_tic)
    {
//      printf("%d diagonal %f %f %f\n",mos_tic,mosaic_umats[mos_tic*9],mosaic_umats[mos_tic*9+4],mosaic_umats[mos_tic*9+8]);
        printf("%d by: %f deg\n",mos_tic,acos((mosaic_umats[mos_tic*9]+mosaic_umats[mos_tic*9+4]+mosaic_umats[mos_tic*9+8]-1)/2)*RTD);
        umat2misset(mosaic_umats+9*mos_tic,mosaic_missets);
        printf("%d by: %f %f %f deg\n",mos_tic,mosaic_missets[1]*RTD,mosaic_missets[2]*RTD,mosaic_missets[3]*RTD);
        printf("       %f %f %f\n",*(mosaic_umats+9*mos_tic+0),*(mosaic_umats+9*mos_tic+1),*(mosaic_umats+9*mos_tic+2));
        printf("       %f %f %f\n",*(mosaic_umats+9*mos_tic+3),*(mosaic_umats+9*mos_tic+4),*(mosaic_umats+9*mos_tic+5));
        printf("       %f %f %f\n",*(mosaic_umats+9*mos_tic+6),*(mosaic_umats+9*mos_tic+7),*(mosaic_umats+9*mos_tic+8));
    }

    printf("  total of %d mosaic domains\n",mosaic_domains);
}
// end of show_mosaic_blocks()


/* return the unitary matrices U that define the mosaic block distribution, cctbx format please */
af::shared<mat3>
nanoBragg::get_mosaic_blocks() {
    /* assume init_mosaicity() was already run? */
    af::shared<mat3> result;
    for(mos_tic=0;mos_tic<mosaic_domains;++mos_tic) {
        result.push_back(mat3( *(mosaic_umats+9*mos_tic+0), *(mosaic_umats+9*mos_tic+1), *(mosaic_umats+9*mos_tic+2),
                               *(mosaic_umats+9*mos_tic+3), *(mosaic_umats+9*mos_tic+4), *(mosaic_umats+9*mos_tic+5),
                               *(mosaic_umats+9*mos_tic+6), *(mosaic_umats+9*mos_tic+7), *(mosaic_umats+9*mos_tic+8)));
    }
    return result;
}
// end of get_mosaic_blocks()

/* set the mosaic domains */
void
nanoBragg::set_mosaic_blocks(af::shared<mat3> umat_in) {
    /* free any previous allocations */
    if(mosaic_umats!=NULL) free(mosaic_umats);

    /* flag to avoid over-writing with automatic random domains */
    user_mosdomains=true;

    /* allocate enough space */
    mosaic_domains = umat_in.size();
    if(verbose>6) printf("allocating enough space for %d mosaic domain orientation matrices\n",mosaic_domains);
    mosaic_umats = (double *) calloc(mosaic_domains+10,9*sizeof(double));

    /* now actually import the orientation of each domain */
    for(mos_tic=0;mos_tic<mosaic_domains;++mos_tic) {
      int offset = 9 * mos_tic;
      mat3& domain = umat_in[mos_tic];
      mosaic_umats[0+offset]=domain[0];mosaic_umats[1+offset]=domain[1];mosaic_umats[2+offset]=domain[2];
      mosaic_umats[3+offset]=domain[3];mosaic_umats[4+offset]=domain[4];mosaic_umats[5+offset]=domain[5];
      mosaic_umats[6+offset]=domain[6];mosaic_umats[7+offset]=domain[7];mosaic_umats[8+offset]=domain[8];
   }

    if(verbose) printf("  imported a total of %d mosaic domains\n",mosaic_domains);
}
// end of set_mosaic_blocks()


/* reconcile different conventions of beam center input and detector position */
void
nanoBragg::show_params()
{
    printf("nanoBragg nanocrystal diffraction simulator - James Holton and Ken Frankel 8-1-17\n");

    printf("  %d initialized hkls (all others =%g)\n",hkls,default_F);
    printf("  ");
    if(xtal_shape == ROUND)  printf("ellipsoidal");
    if(xtal_shape == SQUARE) printf("parallelpiped");
    if(xtal_shape == GAUSS ) printf("gaussian");
    if(xtal_shape == GAUSS_ARGCHK ) printf("gaussian_argchk");
    if(xtal_shape == GAUSS_STAR ) printf("gaussian_star");
    if(xtal_shape == TOPHAT) printf("tophat-spot");
    printf(" xtal: %.1fx%.1fx%.1f cells\n",Na,Nb,Nc);
    printf("Unit Cell: %g %g %g %g %g %g\n", a_A[0],b_A[0],c_A[0],alpha*RTD,beta*RTD,gamma*RTD);
    printf("Recp Cell: %g %g %g %g %g %g\n", a_star[0],b_star[0],c_star[0],alpha_star*RTD,beta_star*RTD,gamma_star*RTD);
    printf("volume = %g A^3\n",V_cell);
    printf("  amorphous part of sample is %lg m thick x %lg m high x %lg m wide, %lg g/cm^3 and %lg g/mol (%lg molecules)\n",
            amorphous_sample_x,amorphous_sample_y,amorphous_sample_z,amorphous_density/1e6,amorphous_molecular_weight,amorphous_molecules);

    printf("missets: %11.8f %11.8f %11.8f\n",misset[1]*RTD,misset[2]*RTD,misset[3]*RTD);

    /* print out the real-space matrix */
    printf("real-space cell vectors (Angstrom):\n");
    printf("     %-10s  %-10s  %-10s\n","a","b","c");
    printf("X: %11.8f %11.8f %11.8f\n",a_A[1],b_A[1],c_A[1]);
    printf("Y: %11.8f %11.8f %11.8f\n",a_A[2],b_A[2],c_A[2]);
    printf("Z: %11.8f %11.8f %11.8f\n",a_A[3],b_A[3],c_A[3]);
    printf("reciprocal-space cell vectors (Angstrom^-1):\n");
    printf("     %-10s  %-10s  %-10s\n","a_star","b_star","c_star");
    printf("X: %11.8f %11.8f %11.8f\n",a_star[1],b_star[1],c_star[1]);
    printf("Y: %11.8f %11.8f %11.8f\n",a_star[2],b_star[2],c_star[2]);
    printf("Z: %11.8f %11.8f %11.8f\n",a_star[3],b_star[3],c_star[3]);
    printf("  wave=%g meters +/- %g%% in %d steps\n",lambda0,dispersion*100,dispsteps);
    if(nopolar) { printf("  polarization effect disabled\n"); }
           else { printf("  Kahn polarization factor: %f\n",polarization); }
    if(curved_detector) printf("  curved detector: all pixels same distance from origin\n");
    if(point_pixel) printf("  pixel obliquity effect disabled\n");
    if(Fmap_pixel) printf("  Fmap mode: plotting structure factor at every pixel\n");
    printf("  incident fluence: %lg photons/m^2\n",fluence);
    printf("  distance=%lg detsize=%lgx%lg  pixel=%lg meters (%dx%d pixels)\n",distance,detsize_f,detsize_s,pixel_size,fpixels,spixels);
    printf("  Xbeam=%lg Ybeam=%lg\n",Xbeam,Ybeam);
    printf("  Fbeam=%lg Sbeam=%lg\n",Fbeam,Sbeam);
    printf("  Xclose=%lg Yclose=%lg\n",Xclose,Yclose);
    printf("  Fclose=%lg Sclose=%lg\n",Fclose,Sclose);
    printf("  DIRECTION_OF_DETECTOR_X-AXIS= %g %g %g\n",fdet_vector[1],fdet_vector[2],fdet_vector[3]);
    printf("  DIRECTION_OF_DETECTOR_Y-AXIS= %g %g %g\n",sdet_vector[1],sdet_vector[2],sdet_vector[3]);
    printf("  DIRECTION_OF_DETECTOR_Z-AXIS= %g %g %g\n",odet_vector[1],odet_vector[2],odet_vector[3]);
    printf("  INCIDENT_BEAM_DIRECTION= %g %g %g\n",beam_vector[1],beam_vector[2],beam_vector[3]);
    printf("  spindle ROTATION_AXIS= %g %g %g\n",spindle_vector[1],spindle_vector[2],spindle_vector[3]);
    cross_product(beam_vector,polar_vector,vector);
    printf("  POLARIZATION_PLANE_NORMAL= %g %g %g\n",vector[1],vector[2],vector[3]);
    printf("  dials origin= %g %g %g\n",dials_origin[1],dials_origin[2],dials_origin[3]);
    printf("  roi: %d < x < %d && %d < y < %d\n",roi_xmin,roi_xmax,roi_ymin,roi_ymax);
    printf("  hdivrange=%g hdivstep=%g  radians\n",hdivrange,hdivstep);
    printf("  vdivrange=%g vdivstep=%g  radians\n",vdivrange,vdivstep);
    printf("  %d divergence steps\n",divsteps);
    printf("  %d sources\n",sources);
    printf("  %d mosaic domains over mosaic spread of %g degrees\n",mosaic_domains,mosaic_spread*RTD);
    printf("  %d phi steps from %g to %g degrees\n",phisteps,phi0*RTD,(phi0+osc)*RTD);
    printf("  %dx%d pixel oversample steps\n",oversample,oversample);
    if(maskimage != NULL) printf("  skipping zero-flagged pixels in %s\n",maskfilename);
//    printf("  coherent source: %d\n",coherent);
}
// end of show_params()




/* add spots from nanocrystal simulation */
void
nanoBragg::randomize_orientation()
{
    if(verbose>1) printf("MISSET seed %ld\n",seed);
    /* re-initialize the RNG */
    seed = -labs(seed);
    /* use spherical cap as sphere to generate random orientation in umat */
    mosaic_rotation_umat(90.0, umat, &seed);
    /* get the missetting angles, in case we want to use them again on -misset option */
    umat2misset(umat,misset);
    if(verbose) printf("random orientation misset angles: %f %f %f deg\n",misset[1]*RTD,misset[2]*RTD,misset[3]*RTD);
    /* apply this orientation shift */
    //rotate_umat(a_star,a_star,umat);
    //rotate_umat(b_star,b_star,umat);
    //rotate_umat(c_star,c_star,umat);
    /* apply below */
    misset[0] = 1.0;
    init_cell();
    return;
}


/* add spots from nanocrystal simulation */
void
nanoBragg::add_nanoBragg_spots()
{
    max_I = 0.0;
    i = 0;
    floatimage = raw_pixels.begin();
//    double* floatimage(raw_pixels.begin());
//    floatimage = (double *) calloc(spixels*fpixels+10,sizeof(double));

    if(verbose) printf("TESTING sincg(1,1)= %f\n",sincg(1,1));

    /* make sure we are normalizing with the right number of sub-steps */
    steps = phisteps*mosaic_domains*oversample*oversample;
    subpixel_size = pixel_size/oversample;

    sum = sumsqr = 0.0;
    i = sumn = 0;
    progress_pixel = 0;
    omega_sum = 0.0;
    for(spixel=0;spixel<spixels;++spixel)
    {
        for(fpixel=0;fpixel<fpixels;++fpixel)
        {

            /* allow for just one part of detector to be rendered */
            if(fpixel < roi_xmin || fpixel > roi_xmax || spixel < roi_ymin || spixel > roi_ymax)
            {
                ++i; continue;
            }
            /* allow for the use of a mask */
            if(maskimage != NULL)
            {
                /* skip any flagged pixels in the mask */
                if(maskimage[i] == 0)
                {
                    ++i; continue;
                }
            }

            /* reset photon count for this pixel */
            I = 0;

            /* loop over sub-pixels */
            for(subS=0;subS<oversample;++subS)
            {
                for(subF=0;subF<oversample;++subF)
                {

                    /* absolute mm position on detector (relative to its origin) */
                    Fdet = subpixel_size*(fpixel*oversample + subF ) + subpixel_size/2.0;
                    Sdet = subpixel_size*(spixel*oversample + subS ) + subpixel_size/2.0;
//                  Fdet = pixel_size*fpixel;
//                  Sdet = pixel_size*spixel;

                    for(thick_tic=0;thick_tic<detector_thicksteps;++thick_tic)
                    {
                        /* assume "distance" is to the front of the detector sensor layer */
                        Odet = thick_tic*detector_thickstep;

                        /* construct detector subpixel position in 3D space */
//                      pixel_X = distance;
//                      pixel_Y = Sdet-Ybeam;
//                      pixel_Z = Fdet-Xbeam;
                        pixel_pos[1] = Fdet*fdet_vector[1]+Sdet*sdet_vector[1]+Odet*odet_vector[1]+pix0_vector[1];
                        pixel_pos[2] = Fdet*fdet_vector[2]+Sdet*sdet_vector[2]+Odet*odet_vector[2]+pix0_vector[2];
                        pixel_pos[3] = Fdet*fdet_vector[3]+Sdet*sdet_vector[3]+Odet*odet_vector[3]+pix0_vector[3];
                        pixel_pos[0] = 0.0;
                        if(curved_detector) {
                            /* construct detector pixel that is always "distance" from the sample */
                                vector[1] = distance*beam_vector[1];
                                vector[2]=distance*beam_vector[2] ;
                                vector[3]=distance*beam_vector[3];
                            /* treat detector pixel coordinates as radians */
                            rotate_axis(vector,newvector,sdet_vector,pixel_pos[2]/distance);
                            rotate_axis(newvector,pixel_pos,fdet_vector,pixel_pos[3]/distance);
    //                      rotate(vector,pixel_pos,0,pixel_pos[3]/distance,pixel_pos[2]/distance);
                        }
                        /* construct the diffracted-beam unit vector to this sub-pixel */
                        airpath = unitize(pixel_pos,diffracted);

                        /* solid angle subtended by a pixel: (pix/airpath)^2*cos(2theta) */
                        omega_pixel = pixel_size*pixel_size/airpath/airpath*close_distance/airpath;
                        /* option to turn off obliquity effect, inverse-square-law only */
                        if(point_pixel) omega_pixel = 1.0/airpath/airpath;
                        omega_sum += omega_pixel;

                        /* now calculate detector thickness effects */
                        if(detector_thick > 0.0 && detector_attnlen > 0.0)
                        {
                            /* inverse of effective thickness increase */
                            parallax = dot_product(diffracted,odet_vector);
                            capture_fraction = exp(-thick_tic*detector_thickstep/detector_attnlen/parallax)
                                              -exp(-(thick_tic+1)*detector_thickstep/detector_attnlen/parallax);
                        }
                        else
                        {
                            capture_fraction = 1.0;
                        }

                    /* loop over sources now */
                    for(source=0;source<sources;++source){

                        /* retrieve stuff from cache */
                        incident[1] = -source_X[source];
                        incident[2] = -source_Y[source];
                        incident[3] = -source_Z[source];
                        lambda = source_lambda[source];

                        /* construct the incident beam unit vector while recovering source distance */
                        source_path = unitize(incident,incident);

                        /* construct the scattering vector for this pixel */
                        scattering[1] = (diffracted[1]-incident[1])/lambda;
                        scattering[2] = (diffracted[2]-incident[2])/lambda;
                        scattering[3] = (diffracted[3]-incident[3])/lambda;

                        /* sin(theta)/lambda is half the scattering vector length */
                        stol = 0.5*magnitude(scattering);

                        /* rough cut to speed things up when we aren't using whole detector */
                        if(dmin > 0.0 && stol > 0.0)
                        {
                            if(dmin > 0.5/stol)
                            {
                                continue;
                            }
                        }

                        /* sweep over phi angles */
                        for(phi_tic = 0; phi_tic < phisteps; ++phi_tic)
                        {
                            phi = phi0 + phistep*phi_tic;

                            if( phi != 0.0 )
                            {
                                /* rotate about spindle if neccesary */
                                rotate_axis(a0,ap,spindle_vector,phi);
                                rotate_axis(b0,bp,spindle_vector,phi);
                                rotate_axis(c0,cp,spindle_vector,phi);
                            }

                            /* enumerate mosaic domains */
                            for(mos_tic=0;mos_tic<mosaic_domains;++mos_tic)
                            {
                                /* apply mosaic rotation after phi rotation */
                                if( mosaic_spread > 0.0 )
                                {
                                    rotate_umat(ap,a,&mosaic_umats[mos_tic*9]);
                                    rotate_umat(bp,b,&mosaic_umats[mos_tic*9]);
                                    rotate_umat(cp,c,&mosaic_umats[mos_tic*9]);
                                }
                                else
                                {
                                    a[1]=ap[1];a[2]=ap[2];a[3]=ap[3];
                                    b[1]=bp[1];b[2]=bp[2];b[3]=bp[3];
                                    c[1]=cp[1];c[2]=cp[2];c[3]=cp[3];
                                }
//                              printf("%d %f %f %f\n",mos_tic,mosaic_umats[mos_tic*9+0],mosaic_umats[mos_tic*9+1],mosaic_umats[mos_tic*9+2]);
//                              printf("%d %f %f %f\n",mos_tic,mosaic_umats[mos_tic*9+3],mosaic_umats[mos_tic*9+4],mosaic_umats[mos_tic*9+5]);
//                              printf("%d %f %f %f\n",mos_tic,mosaic_umats[mos_tic*9+6],mosaic_umats[mos_tic*9+7],mosaic_umats[mos_tic*9+8]);

                                /* construct fractional Miller indicies */
                                h = dot_product(a,scattering);
                                k = dot_product(b,scattering);
                                l = dot_product(c,scattering);

                                /* round off to nearest whole index */
                                h0 = static_cast<int>(ceil(h-0.5));
                                k0 = static_cast<int>(ceil(k-0.5));
                                l0 = static_cast<int>(ceil(l-0.5));


                                /* structure factor of the lattice (paralelpiped crystal)
                                    F_latt = sin(M_PI*Na*h)*sin(M_PI*Nb*k)*sin(M_PI*Nc*l)/sin(M_PI*h)/sin(M_PI*k)/sin(M_PI*l);
                                */
                                F_latt = 1.0;
                                    if(xtal_shape == SQUARE)
                                {
                                    /* xtal is a paralelpiped */
                                    if(Na>1){
                                        F_latt *= sincg(M_PI*h,Na);
                                    }
                                    if(Nb>1){
                                        F_latt *= sincg(M_PI*k,Nb);
                                    }
                                    if(Nc>1){
                                        F_latt *= sincg(M_PI*l,Nc);
                                    }
                                }
                                    else
                                    {
                                        /* handy radius in reciprocal space, squared */
                                        hrad_sqr = (h-h0)*(h-h0)*Na*Na + (k-k0)*(k-k0)*Nb*Nb + (l-l0)*(l-l0)*Nc*Nc ;
                                    }
                                    if(xtal_shape == ROUND)
                                    {
                                        /* use sinc3 for elliptical xtal shape,
                                           correcting for sqrt of volume ratio between cube and sphere */
                                        F_latt = Na*Nb*Nc*0.723601254558268*sinc3(M_PI*sqrt( hrad_sqr * fudge ) );
                                    }
                                    if(xtal_shape == GAUSS)
                                    {
                                        /* fudge the radius so that volume and FWHM are similar to square_xtal spots */
                                        F_latt = Na*Nb*Nc*exp(-( hrad_sqr / 0.63 * fudge ));
                                    }
                                    if (xtal_shape == GAUSS_ARGCHK)
                                    {
                                        /* fudge the radius so that volume and FWHM are similar to square_xtal spots */
                                        double my_arg = hrad_sqr / 0.63 * fudge; // pre-calculate to check for no Bragg signal
                                        if (my_arg<35.){ F_latt = Na * Nb * Nc * exp(-(my_arg));}
                                        else { F_latt = 0.; } // not expected to give performance gain on optimized C++, only on GPU
                                    }
                                    if (xtal_shape == GAUSS_STAR){
                                        double xtal_size_sq = pow(Na*Nb*Nc*V_cell, double(2)/double(3));
                                        vec3 delta_H {h-h0, k-k0, l-l0};
                                        mat3 At(a[1], a[2], a[3], b[1], b[2], b[3], c[1], c[2], c[3]);
                                        mat3 Ainverse =  (1e10*At).inverse();
                                        vec3 delta_Q = Ainverse*delta_H;
                                        double rad_star_sqr = (delta_Q*delta_Q)*xtal_size_sq; // dot product
                                        F_latt = Na*Nb*Nc*exp(-( rad_star_sqr*1.9*fudge ));
                                    }

                                    if(xtal_shape == TOPHAT)
                                    {
                                        /* make a flat-top spot of same height and volume as square_xtal spots */
                                        F_latt = Na*Nb*Nc*(hrad_sqr*fudge < 0.3969 );
                                }
                                /* no need to go further if result will be zero */
                                if(F_latt == 0.0) continue;


                                /* find nearest point on Ewald sphere surface? */
                                if( integral_form )
                                {

                                    if( phi != 0.0 || mos_tic > 0 )
                                    {
                                        /* need to re-calculate reciprocal matrix */

                                        /* various cross products */
                                        cross_product(a,b,a_cross_b);
                                        cross_product(b,c,b_cross_c);
                                        cross_product(c,a,c_cross_a);

                                        /* new reciprocal-space cell vectors */
                                        vector_scale(b_cross_c,a_star,1e20/V_cell);
                                        vector_scale(c_cross_a,b_star,1e20/V_cell);
                                        vector_scale(a_cross_b,c_star,1e20/V_cell);
                                    }

                                    /* reciprocal-space coordinates of nearest relp */
                                    relp[1] = h0*a_star[1] + k0*b_star[1] + l0*c_star[1];
                                    relp[2] = h0*a_star[2] + k0*b_star[2] + l0*c_star[2];
                                    relp[3] = h0*a_star[3] + k0*b_star[3] + l0*c_star[3];
//                                  d_star = magnitude(relp)

                                    /* reciprocal-space coordinates of center of Ewald sphere */
                                    Ewald0[1] = -incident[1]/lambda/1e10;
                                    Ewald0[2] = -incident[2]/lambda/1e10;
                                    Ewald0[3] = -incident[3]/lambda/1e10;
//                                  1/lambda = magnitude(Ewald0)

                                     /* distance from Ewald sphere in lambda=1 units */
                                    vector[1] = relp[1]-Ewald0[1];
                                    vector[2] = relp[2]-Ewald0[2];
                                    vector[3] = relp[3]-Ewald0[3];
                                    d_r = magnitude(vector)-1.0;

                                    /* unit vector of diffracted ray through relp */
                                    unitize(vector,diffracted0);

                                    /* intersection with detector plane */
                                    xd = dot_product(fdet_vector,diffracted0);
                                    yd = dot_product(sdet_vector,diffracted0);
                                    zd = dot_product(odet_vector,diffracted0);

                                    /* where does the central direct-beam hit */
                                    xd0 = dot_product(fdet_vector,incident);
                                    yd0 = dot_product(sdet_vector,incident);
                                    zd0 = dot_product(odet_vector,incident);

                                    /* convert to mm coordinates */
                                    Fdet0 = distance*(xd/zd) + Xbeam;
                                    Sdet0 = distance*(yd/zd) + Ybeam;

                                    if(verbose>8) printf("integral_form: %g %g   %g %g\n",Fdet,Sdet,Fdet0,Sdet0);
                                    test = exp(-( (Fdet-Fdet0)*(Fdet-Fdet0)+(Sdet-Sdet0)*(Sdet-Sdet0) + d_r*d_r )/1e-8);
                                } // end of integral form


                                /* structure factor of the unit cell */
                                if(interpolate){
                                    h0_flr = static_cast<int>(floor(h));
                                    k0_flr = static_cast<int>(floor(k));
                                    l0_flr = static_cast<int>(floor(l));


                                    if ( ((h-h_min+3)>h_range) ||
                                         (h-2<h_min)           ||
                                         ((k-k_min+3)>k_range) ||
                                         (k-2<k_min)           ||
                                         ((l-l_min+3)>l_range) ||
                                         (l-2<l_min)  ) {
                                        if(babble){
                                            babble=0;
                                            if(verbose) printf ("WARNING: out of range for three point interpolation: h,k,l,h0,k0,l0: %g,%g,%g,%d,%d,%d \n", h,k,l,h0,k0,l0);
                                            if(verbose) printf("WARNING: further warnings will not be printed! ");
                                        }
                                        F_cell = default_F;
                                        interpolate=0;
                                        continue;
                                    }

                                    /* integer versions of nearest HKL indicies */
                                    h_interp[0]=h0_flr-1;
                                    h_interp[1]=h0_flr;
                                    h_interp[2]=h0_flr+1;
                                    h_interp[3]=h0_flr+2;
                                    k_interp[0]=k0_flr-1;
                                    k_interp[1]=k0_flr;
                                    k_interp[2]=k0_flr+1;
                                    k_interp[3]=k0_flr+2;
                                    l_interp[0]=l0_flr-1;
                                    l_interp[1]=l0_flr;
                                    l_interp[2]=l0_flr+1;
                                    l_interp[3]=l0_flr+2;

                                    /* polin function needs doubles */
                                    h_interp_d[0] = (double) h_interp[0];
                                    h_interp_d[1] = (double) h_interp[1];
                                    h_interp_d[2] = (double) h_interp[2];
                                    h_interp_d[3] = (double) h_interp[3];
                                    k_interp_d[0] = (double) k_interp[0];
                                    k_interp_d[1] = (double) k_interp[1];
                                    k_interp_d[2] = (double) k_interp[2];
                                    k_interp_d[3] = (double) k_interp[3];
                                    l_interp_d[0] = (double) l_interp[0];
                                    l_interp_d[1] = (double) l_interp[1];
                                    l_interp_d[2] = (double) l_interp[2];
                                    l_interp_d[3] = (double) l_interp[3];

                                    /* now populate the "y" values (nearest four structure factors in each direction) */
                                    for (i1=0;i1<4;i1++) {
                                        for (i2=0;i2<4;i2++) {
                                           for (i3=0;i3<4;i3++) {
                                                  sub_Fhkl[i1][i2][i3]= Fhkl[h_interp[i1]-h_min][k_interp[i2]-k_min][l_interp[i3]-l_min];
                                           }
                                        }
                                     }


                                    /* run the tricubic polynomial interpolation */
                                    polin3(h_interp_d,k_interp_d,l_interp_d,sub_Fhkl,h,k,l,&F_cell);
                                }

                                if(! interpolate)
                                {
                                    if ( (h0<=h_max) && (h0>=h_min) && (k0<=k_max) && (k0>=k_min) && (l0<=l_max) && (l0>=l_min)  ) {
                                        /* just take nearest-neighbor */
                                        F_cell = Fhkl[h0-h_min][k0-k_min][l0-l_min];
                                    }
                                    else
                                    {
                                        F_cell = default_F; // usually zero
                                    }
                                }

                                /* now we have the structure factor for this pixel */

                                /* polarization factor */
                                if(! nopolar){
                                    /* need to compute polarization factor */
                                    polar = polarization_factor(polarization,incident,diffracted,polar_vector);
                                }
                                else
                                {
                                    polar = 1.0;
                                }

                                /* convert amplitudes into intensity (photons per steradian) */
                                    I += F_cell*F_cell*F_latt*F_latt*source_I[source]*capture_fraction*omega_pixel;
                            }
                            /* end of mosaic loop */
                        }
                        /* end of phi loop */
                    }
                    /* end of source loop */
                }
                    /* end of detector thickness loop */
                }
                /* end of sub-pixel y loop */
            }
            /* end of sub-pixel x loop */


            floatimage[i] += r_e_sqr*fluence*spot_scale*polar*I/steps;
//          floatimage[i] = test;
            if(floatimage[i] > max_I) {
                max_I = floatimage[i];
                max_I_x = Fdet;
                max_I_y = Sdet;
            }
            sum += floatimage[i];
            sumsqr += floatimage[i]*floatimage[i];
            ++sumn;

            if( printout )
            {
                if((fpixel==printout_fpixel && spixel==printout_spixel) || printout_fpixel < 0)
                {
                    twotheta = atan2(sqrt(pixel_pos[2]*pixel_pos[2]+pixel_pos[3]*pixel_pos[3]),pixel_pos[1]);
                    test = sin(twotheta/2.0)/(lambda0*1e10);
                    printf("%4d %4d : stol = %g or %g\n", fpixel,spixel,stol,test);
                    printf("at %g %g %g\n", pixel_pos[1],pixel_pos[2],pixel_pos[3]);
                    printf("hkl= %f %f %f  hkl0= %d %d %d\n", h,k,l,h0,k0,l0);
                    printf(" F_cell=%g  F_latt=%g   I = %g\n", F_cell,F_latt,I);
                    printf("I/steps %15.10g\n", I/steps);
                    printf("cap frac   %f\n", capture_fraction);
                    printf("polar   %15.10g\n", polar);
                    printf("omega   %15.10g\n", omega_pixel);
                    printf("pixel   %15.10g\n", floatimage[i]);
                    printf("real-space cell vectors (Angstrom):\n");
                    printf("     %-10s  %-10s  %-10s\n","a","b","c");
                    printf("X: %11.8f %11.8f %11.8f\n",a[1]*1e10,b[1]*1e10,c[1]*1e10);
                    printf("Y: %11.8f %11.8f %11.8f\n",a[2]*1e10,b[2]*1e10,c[2]*1e10);
                    printf("Z: %11.8f %11.8f %11.8f\n",a[3]*1e10,b[3]*1e10,c[3]*1e10);
                    SCITBX_EXAMINE(fluence);
                    SCITBX_EXAMINE(source_I[0]);
                    SCITBX_EXAMINE(spot_scale);
                    SCITBX_EXAMINE(Na);
                    SCITBX_EXAMINE(Nb);
                    SCITBX_EXAMINE(Nc);
                    SCITBX_EXAMINE(airpath);
                    SCITBX_EXAMINE(Fclose);
                    SCITBX_EXAMINE(Sclose);
                    SCITBX_EXAMINE(close_distance);
                    SCITBX_EXAMINE(pix0_vector[0]);
                    SCITBX_EXAMINE(pix0_vector[1]);
                    SCITBX_EXAMINE(pix0_vector[2]);
                    SCITBX_EXAMINE(pix0_vector[3]);
                    SCITBX_EXAMINE(odet_vector[0]);
                    SCITBX_EXAMINE(odet_vector[1]);
                    SCITBX_EXAMINE(odet_vector[2]);
                    SCITBX_EXAMINE(odet_vector[3]);
                }
            }
            else
            {
                if(progress_meter && verbose && progress_pixels/100 > 0)
                {
                    if(progress_pixel % ( progress_pixels/20 ) == 0 ||
                       ((10*progress_pixel<progress_pixels ||
                         10*progress_pixel>9*progress_pixels) &&
                        (progress_pixel % (progress_pixels/100) == 0)))
                    {
                        printf("%lu%% done\n",progress_pixel*100/progress_pixels);
                    }
                }
                ++progress_pixel;
            }
            ++i;
        }
    }
    if(verbose) printf("done with pixel loop\n");

    if(verbose) printf("solid angle subtended by detector = %g steradian ( %g%% sphere)\n",omega_sum/steps,100*omega_sum/steps/4/M_PI);
    if(verbose) printf("max_I= %g sum= %g avg= %g\n",max_I,sum,sum/sumn);

}
// end of add_nanoBragg_spots()




/* member function to generate background from Fbg vs stol list
   arguments allow override of features that usually just slow things down,
   like oversampling pixels and multiple sources.
   oversample: user can provide a smaller override value to save time.
   override_source: user can select a single source from the collection to save time.
   Providing these arguments does NOT change the values of the member variables */
void
nanoBragg::add_background( int oversample, int const& override_source )
{
    int i;
    int source_start = 0;
    int orig_sources = this->sources;
    int end_sources = this->sources;
    max_I = 0.0;
    floatimage = raw_pixels.begin();
//    double* floatimage(raw_pixels.begin());
//    floatimage = (double *) calloc(spixels*fpixels+10,sizeof(double));

    /* allow user to override automated oversampling decision at call time with arguments */
    if(oversample<=0) oversample = this->oversample;
    if(oversample<=0) oversample = 1;
    bool have_single_source = false;
    if(override_source>=0) {
        /* user-specified idx_single_source in the argument */
        source_start = override_source;
        end_sources = source_start +1;
        have_single_source = true;
    }

    /* make sure we are normalizing with the right number of sub-steps */
    steps = oversample*oversample;
    subpixel_size = pixel_size/oversample;

    /* sweep over detector */
    sum = sumsqr = 0.0;
    sumn = 0;
    progress_pixel = 0;
    omega_sum = 0.0;
    nearest = 0;
    i = 0;
    for(spixel=0;spixel<spixels;++spixel)
    {
        for(fpixel=0;fpixel<fpixels;++fpixel)
        {
            /* allow for just one part of detector to be rendered */
            if(fpixel < roi_xmin || fpixel > roi_xmax || spixel < roi_ymin || spixel > roi_ymax) {
                invalid_pixel[i] = true;
                ++i; continue;
            }

            /* reset background photon count for this pixel */
            Ibg = 0;

            /* loop over sub-pixels */
            for(subS=0;subS<oversample;++subS){
                for(subF=0;subF<oversample;++subF){

                    /* absolute mm position on detector (relative to its origin) */
                    Fdet = subpixel_size*(fpixel*oversample + subF ) + subpixel_size/2.0;
                    Sdet = subpixel_size*(spixel*oversample + subS ) + subpixel_size/2.0;
//                    Fdet = pixel_size*fpixel;
//                    Sdet = pixel_size*spixel;

                    for(thick_tic=0;thick_tic<detector_thicksteps;++thick_tic)
                    {
                        /* assume "distance" is to the front of the detector sensor layer */
                        Odet = thick_tic*detector_thickstep;

                        /* construct detector pixel position in 3D space */
    //                    pixel_X = distance;
    //                    pixel_Y = Sdet-Ybeam;
    //                    pixel_Z = Fdet-Xbeam;
                        pixel_pos[1] = Fdet*fdet_vector[1]+Sdet*sdet_vector[1]+Odet*odet_vector[1]+pix0_vector[1];
                        pixel_pos[2] = Fdet*fdet_vector[2]+Sdet*sdet_vector[2]+Odet*odet_vector[2]+pix0_vector[2];
                        pixel_pos[3] = Fdet*fdet_vector[3]+Sdet*sdet_vector[3]+Odet*odet_vector[3]+pix0_vector[3];
                        pixel_pos[0] = 0.0;
                        if(curved_detector) {
                            /* construct detector pixel that is always "distance" from the sample */
                            vector[1] = distance*beam_vector[1]; vector[2]=distance*beam_vector[2] ; vector[3]=distance*beam_vector[3];
                            /* treat detector pixel coordinates as radians */
                            rotate_axis(vector,newvector,sdet_vector,pixel_pos[2]/distance);
                            rotate_axis(newvector,pixel_pos,fdet_vector,pixel_pos[3]/distance);
    //                             rotate(vector,pixel_pos,0,pixel_pos[3]/distance,pixel_pos[2]/distance);
                        }
                        /* construct the diffracted-beam unit vector to this pixel */
                        airpath = unitize(pixel_pos,diffracted);

                        /* solid angle subtended by a pixel: (pix/airpath)^2*cos(2theta) */
                        omega_pixel = pixel_size*pixel_size/airpath/airpath*close_distance/airpath;
                        /* option to turn off obliquity effect, inverse-square-law only */
                        if(point_pixel) omega_pixel = 1.0/airpath/airpath;
                        omega_sum += omega_pixel;

                        /* now calculate detector thickness effects */
                        if(detector_thick > 0.0)
                        {
                            /* inverse of effective thickness increase */
                            parallax = dot_product(diffracted,odet_vector);
                            capture_fraction = exp(-thick_tic*detector_thickstep/detector_attnlen/parallax)
                                              -exp(-(thick_tic+1)*detector_thickstep/detector_attnlen/parallax);
                        }
                        else
                        {
                            capture_fraction = 1.0;
                        }

                        /* loop over sources now */
                        for(source=source_start; source < end_sources; ++source){
                            double n_source_scale = (have_single_source) ? orig_sources : source_I[source];

                            /* retrieve stuff from cache */
                            incident[1] = -source_X[source];
                            incident[2] = -source_Y[source];
                            incident[3] = -source_Z[source];
                            lambda = source_lambda[source];

                            /* construct the incident beam unit vector while recovering source distance */
                            source_path = unitize(incident,incident);

                            /* construct the scattering vector for this pixel */
                            scattering[1] = (diffracted[1]-incident[1])/lambda;
                            scattering[2] = (diffracted[2]-incident[2])/lambda;
                            scattering[3] = (diffracted[3]-incident[3])/lambda;

                            /* sin(theta)/lambda is half the scattering vector length */
                            stol = 0.5*magnitude(scattering);

                            /* now we need to find the nearest four "stol file" points */
                            while(stol > stol_of[nearest] && nearest <= stols){++nearest; };
                            while(stol < stol_of[nearest] && nearest >= 2){--nearest; };

                            /* cubic spline interpolation */
                            polint(stol_of+nearest-1, Fbg_of+nearest-1, stol, &Fbg);

                            /* allow negative F values to yield negative intensities */
                            sign=1.0;
                            if(Fbg<0.0) sign=-1.0;

                            /* now we have the structure factor for this pixel */

                            /* polarization factor */
                            if(! nopolar){
                                /* need to compute polarization factor */
                                polar = polarization_factor(polarization,incident,diffracted,polar_vector);
                            }
                            else
                            {
                                polar = 1.0;
                            }

                            /* accumulate unscaled pixel intensity from this */
                            Ibg += sign*Fbg*Fbg*polar*omega_pixel*capture_fraction*n_source_scale;
                            if(verbose>7 && i==1)printf("DEBUG: Fbg= %g polar= %g omega_pixel= %g source[%d]= %g capture_fraction= %g\n",
                                                           Fbg,polar,omega_pixel,source,source_I[source],capture_fraction);
                        }
                        /* end of source loop */
                    }
                    /* end of detector thickness loop */
                }
                /* end of sub-pixel y loop */
            }
            /* end of sub-pixel x loop */


            /* save photons/pixel (if fluence specified), or F^2/omega if no fluence given */
            floatimage[i] += Ibg*r_e_sqr*fluence*amorphous_molecules/steps;

            if(verbose>7 && i==1)printf(
              "DEBUG: Ibg= %g r_e_sqr= %g fluence= %g amorphous_molecules= %g parallax= %g capfrac= %g omega= %g polar= %g steps= %d\n",
                        Ibg,r_e_sqr,fluence,amorphous_molecules,parallax,capture_fraction,omega_pixel,polar,steps);

            /* override: just plot interpolated structure factor at every pixel, useful for making absorption masks */
            if(Fmap_pixel) floatimage[i]= Fbg;

            /* keep track of basic statistics */
            if(floatimage[i] > max_I || i==0) {
                max_I = floatimage[i];
                max_I_x = Fdet;
                max_I_y = Sdet;
            }
            sum += floatimage[i];
            sumsqr += floatimage[i]*floatimage[i];
            ++sumn;

            /* debugging infrastructure */
            if( printout )
            {
                if((fpixel==printout_fpixel && spixel==printout_spixel) || printout_fpixel < 0)
                {
                    twotheta = atan2(sqrt(pixel_pos[2]*pixel_pos[2]+pixel_pos[3]*pixel_pos[3]),pixel_pos[1]);
                    test = sin(twotheta/2.0)/(lambda0*1e10);
                    printf("%4d %4d : stol = %g or %g\n", fpixel,spixel,stol,test);
                    printf(" F=%g    I = %g\n", F,I);
                    printf("I/steps %15.10g\n", I/steps);
                    printf("polar   %15.10g\n", polar);
                    printf("omega   %15.10g\n", omega_pixel);
                    printf("pixel   %15.10g\n", floatimage[i]);
                }
            }
            else
            {
                if(progress_meter && progress_pixels/100 > 0)
                {
                    if(progress_pixel % ( progress_pixels/20 ) == 0 ||
                       ((10*progress_pixel<progress_pixels ||
                         10*progress_pixel>9*progress_pixels) &&
                        (progress_pixel % (progress_pixels/100) == 0)))
                    {
                        printf("%lu%% done\n",progress_pixel*100/progress_pixels);
                    }
                }
                ++progress_pixel;
            }
            /* end progress meter stuff */

            /* never ever forget to increment this */
            ++i;
        }
        /* end fpixel loop */
    }
    /* end spixel loop */

    if(verbose) printf("\nsolid angle subtended by detector = %g steradian ( %g%% sphere)\n",omega_sum/steps,100*omega_sum/steps/4/M_PI);
    if(verbose) printf("max_I= %g @ ( %g, %g) sum= %g avg= %g\n",max_I,max_I_x,max_I_y,sum,sum/sumn);

}
// end of add_background()






/* function for extracting stol-vs-Fbg data from an existing image */
void
nanoBragg::extract_background(int source)
{
    /* make sure we override k from hkl */
    int i,k;

    nearest = 0;
    max_I = 0.0;
    floatimage = raw_pixels.begin();
//    double* floatimage(raw_pixels.begin());
//    floatimage = (double *) calloc(spixels*fpixels+10,sizeof(double));

    /* override internal default if argument was given */
    if(source<0) {
        /* user did not specify source in the argument, use the first one */
        source = 0;
    }
    /* no oversampling, this is an extraction */
    steps = 1;

    /* sweep over detector */
    sum = sumsqr = 0.0;
    i = 0;
    progress_pixel = 0;
    valid_pixels = 0;
    omega_sum = 0.0;
    for(spixel=0;spixel<spixels;++spixel)
    {
        for(fpixel=0;fpixel<fpixels;++fpixel)
        {

            /* allow for just one part of detector to be rendered */
            if(fpixel < roi_xmin || fpixel > roi_xmax || spixel < roi_ymin || spixel > roi_ymax) {
                invalid_pixel[i] = true;
                ++i; continue;
            }

            /* is the pixel valid on the input image? */
            /* skip over any invalid values */
            for(k=1;k<=ignore_values;++k)
            {
                if(floatimage[i]==ignore_value[k]){
                    invalid_pixel[i] = true;
                }
            }
//            if(invalid_pixel[i])
//            {
//                ++i;
//                continue;
//            }

            /* still need polar, omega and stol, but cannot deconvolute sub-pixels and source size,
               so assume oversample=1, neutronium detector, and take the first source */
            thick_tic = 0;
            //source = 0;

            /* absolute mm position on detector (relative to its origin) */
            //Fdet = subpixel_size*(fpixel*oversample + subF ) + subpixel_size/2.0;
            //Sdet = subpixel_size*(spixel*oversample + subS ) + subpixel_size/2.0;
            Fdet = pixel_size*fpixel;
            Sdet = pixel_size*spixel;

            /* assume "distance" is to the front of the detector sensor layer */
            //Odet = thick_tic*detector_thickstep;
            Odet = 0.;

            /* construct detector pixel position in 3D space */
            //                    pixel_X = distance;
            //                    pixel_Y = Sdet-Ybeam;
            //                    pixel_Z = Fdet-Xbeam;
            pixel_pos[1] = Fdet*fdet_vector[1]+Sdet*sdet_vector[1]+Odet*odet_vector[1]+pix0_vector[1];
            pixel_pos[2] = Fdet*fdet_vector[2]+Sdet*sdet_vector[2]+Odet*odet_vector[2]+pix0_vector[2];
            pixel_pos[3] = Fdet*fdet_vector[3]+Sdet*sdet_vector[3]+Odet*odet_vector[3]+pix0_vector[3];
            pixel_pos[0] = 0.0;
            if(curved_detector) {
                /* construct detector pixel that is always "distance" from the sample */
                vector[1] = distance*beam_vector[1];
                vector[2] = distance*beam_vector[2] ;
                vector[3] = distance*beam_vector[3];
                /* treat detector pixel coordinates as radians */
                rotate_axis(vector,newvector,sdet_vector,pixel_pos[2]/distance);
                rotate_axis(newvector,pixel_pos,fdet_vector,pixel_pos[3]/distance);
                //rotate(vector,pixel_pos,0,pixel_pos[3]/distance,pixel_pos[2]/distance);
            }
            /* construct the diffracted-beam unit vector to this pixel */
            airpath = unitize(pixel_pos,diffracted);

            /* solid angle subtended by a pixel: (pix/airpath)^2*cos(2theta) */
            omega_pixel = pixel_size*pixel_size/airpath/airpath*close_distance/airpath;
            /* option to turn off obliquity effect, inverse-square-law only */
            if(point_pixel) omega_pixel = 1.0/airpath/airpath;
            omega_sum += omega_pixel;

            /* retrieve stuff from cache */
            incident[1] = -source_X[source];
            incident[2] = -source_Y[source];
            incident[3] = -source_Z[source];
            lambda = source_lambda[source];

            /* construct the incident beam unit vector while recovering source distance */
            source_path = unitize(incident,incident);

            /* construct the scattering vector for this pixel */
            scattering[1] = (diffracted[1]-incident[1])/lambda;
            scattering[2] = (diffracted[2]-incident[2])/lambda;
            scattering[3] = (diffracted[3]-incident[3])/lambda;

            /* sin(theta)/lambda is half the scattering vector length */
            stol = 0.5*magnitude(scattering);

            /* now we need to find the nearest four "stol file" points */
            while(stol > stol_of[nearest] && nearest <= stols){++nearest; };
            while(stol < stol_of[nearest] && nearest >= 2){--nearest; };

            /* polarization factor */
            if(! nopolar){
                /* need to compute polarization factor */
                polar = polarization_factor(polarization,incident,diffracted,polar_vector);
            }
            else
            {
                polar = 1.0;
            }

            /* now we have everything we need to transform pixel intensity back to a structure factor */
            /* note that for real image data we want to subtract ADC offset, but not for half-done simulated data */
            deviate=floatimage[i]-adc_offset;
            /* for negative intensities, make the structure factor negative too */
            sign = 1.0;
            if(deviate<0.0) sign = -1.0;
            deviate = fabs(deviate);
            pixel_F = sign*sqrt(deviate/polar/omega_pixel/fluence/r_e_sqr/amorphous_molecules);
            /* maintain F and stol images */
            stolimage[i] = stol/stol_file_mult;
            Fimage[i] = pixel_F;
            bin = 0;
            if(! invalid_pixel[i])
            {
                /* figure out which stol bin this pixel belongs to.  invalid pixels are in bin=0 */
                bin = nearest;
                /* move to next bin if it is more than halfway there */
                if(stol > (stol_of[bin]+stol_of[bin+1])/2.0) ++bin;
                /* only pixels with bin assignments are valid */
                ++valid_pixels;
            }
            ++pixels_in[bin];
            bin_of[i]=bin;

            /* now maybe do some stats? */
            if(floatimage[i] > max_I) {
                max_I = floatimage[i];
                max_I_x = Fdet;
                max_I_y = Sdet;
            }
            sum += floatimage[i];
            sumsqr += floatimage[i]*floatimage[i];
            ++n;

            /* debugging option: print out particular pixel, or progress meter */
            if( printout )
            {
                if((fpixel==printout_fpixel && spixel==printout_spixel) || printout_fpixel < 0)
                {
                    twotheta = atan2(sqrt(pixel_pos[2]*pixel_pos[2]+pixel_pos[3]*pixel_pos[3]),pixel_pos[1]);
                    test = sin(twotheta/2.0)/(lambda0*1e10);
                        printf("%4d %4d : stol = %g or %g\n", fpixel,spixel,stol,test);
                        printf(" F=%g    I = %g\n", F,I);
                        printf("I/steps %15.10g\n", I/steps);
                        printf("polar   %15.10g\n", polar);
                        printf("omega   %15.10g\n", omega_pixel);
                        printf("pixel   %15.10g\n", floatimage[i]);
                }
            }
            else
            {
                if(progress_meter && progress_pixels/100 > 0)
                {
                    if(progress_pixel % ( progress_pixels/20 ) == 0 ||
                       ((10*progress_pixel<progress_pixels ||
                         10*progress_pixel>9*progress_pixels) &&
                        (progress_pixel % (progress_pixels/100) == 0)))
                    {
                        printf("%lu%% done\n",progress_pixel*100/progress_pixels);
                    }
                }
                ++progress_pixel;
            }

            ++i;
        }
        /* end fpixel loop */
    }
    /* end spixel loop */

    /* now we need to organize Fpixel data into bins */

    /* set up pointers with enough space after each of them (2*n for median/mad filter) */
    bin_start[0]= (double *) calloc(2*pixels+10*stols,sizeof(double));
    ++bin_start[0];
    for(bin=1;bin<stols-1;++bin)
    {
        /* each array must have 2*n values in it */
        /* we counted number of pixels_in each bin in above loop */
        bin_start[bin]=bin_start[bin-1]+2*pixels_in[bin-1]+2;
    }

    /* populate each bin with appropriate pixel values */
    for(i=0;i<pixels;++i)
    {
        /* recover which bin this pixel is in */
        bin = bin_of[i];
        /* copy F value into start of bin data */
        *bin_start[bin] = Fimage[i];
        /* increment the pointer to the next value for this bin */
        ++bin_start[bin];
        /* yes, I know.  The bin_start now points to the end, we will reset the starting points in the next loop */
    }


    /* now we go through bins, rejecting outliers and taking mean of whats left */
    for(bin=2;bin<stols-2;++bin)
    {
        /* correct pointer drift in last loop, bin_start will now point to the start of its data */
        bin_start[bin] -= pixels_in[bin];

        /* recover cached value of stol for this bin */
        stol = stol_of[bin];
        /* this function looks at "input_n" elements, starting at 1 */
        median   = fmedian_with_rejection(pixels_in[bin],bin_start[bin]-1,6.0,&mad,&n);
        /* now that outliers are rejected, take the mean of what is left */
        avg_arej = fmean_with_rejection(n,bin_start[bin],6.0,&rmsd_arej,&n);
        if(n>100)
        {
//          fprintf(outfile,"%g %g %g  %d\n",stol/stol_file_mult,median,mad,n);
//          fprintf(outfile,"%g %g %g  %d\n",stol/stol_file_mult,avg_arej,rmsd_arej,n);
            if(verbose) printf("bin= %d stol= %g avg= %g rmsd= %g  n= %d\n",bin,stol/stol_file_mult,avg_arej,rmsd_arej,n);
            /* now populate the Fbg array.  This should automagically get passed back to Python? */
            Fbg_of[bin]=avg_arej;
        }
        else
        {
            if(verbose) printf("WARNING: not enough pixels in bin= %d n=%d stol= %g median= %g avg_arej= %g\n",bin,n,stol/stol_file_mult,median,avg_arej);
        }
    }

    if(verbose) printf("done with radial median filter\n");

    /* now we need to copy back into pythony array */
}
// end of extract_background()






/* function for applying the PSF, copies over raw pixels with blurred version of itself */
void
nanoBragg::apply_psf(shapetype psf_type, double fwhm_pixels, int user_psf_radius)
{
    max_I=0.0;
    double* inimage(raw_pixels.begin());
    double *outimage=NULL;
    double *kernel;
    int x0,y0,x,y,dx,dy;
    double g,rsq;
    double photon_noise,lost_photons=0.0,total_lost_photons=0.0;
    int maxwidth,kernel_size,psf_radius;
    int i,j,k;
    double photonloss_factor = 10.0; // inverse of maximum tolerable number of lost photons


    /* take the member value for PSF radius if it is set and nothing was in the function call */
    if(user_psf_radius <= 0 && this->psf_radius > 0) user_psf_radius = this->psf_radius;
    /* find a fwhm for the PSF in pixel units */
    if(fwhm_pixels <= 0.0) fwhm_pixels = this->psf_fwhm/this->pixel_size;
    /* update the members, for posterity */
    this->psf_fwhm = fwhm_pixels * this->pixel_size;
    this->psf_radius = user_psf_radius;

    if(verbose>7) printf("apply_psf(): user_psf_radius = %d\n",user_psf_radius);
    if(verbose>7) printf("apply_psf(): updated psf_fwhm = %g  pixel_size= %g\n",psf_fwhm,pixel_size);

    /* convert fwhm to "g" distance : fwhm = sqrt((2**(2./3)-1))/2*g */
    g = fwhm_pixels * 0.652383013252053;

    if(psf_type != GAUSS && psf_type != FIBER)
    {
        if(verbose) printf("ERROR: unknown PSF type\n");
        return;
    }

    pixels = fpixels*spixels;
    if(pixels == 0)
    {
        if(verbose) printf("ERROR: apply_psf image has zero size\n");
        return;
    }

    if(fwhm_pixels <= 0.0)
    {
        if(verbose) printf("WARNING: apply_psf function has zero size\n");
        return;
    }

    /* start with a clean slate */
    if(outimage!=NULL) free(outimage);
    outimage = (double *) calloc(pixels+10,sizeof(double));

    psf_radius = user_psf_radius;
    if(psf_radius <= 0)
    {
        /* auto-select radius */

        /* preliminary stats */
        max_I = 0.0;
        for(i=0;i<pixels;++i)
        {
            /* optionally scale the input file */
            if(max_I < inimage[i]) max_I = inimage[i];
        }
        if(verbose) printf("  maximum input photon/pixel: %g\n",max_I);

        if(max_I<=0.0)
        {
            /* nothing to blur */
            if(verbose) printf("WARNING: no photons, PSF skipped\n");
            return;
        }

        /* at what level will an error in intensity be lost? */
        photon_noise = sqrt(max_I);
        lost_photons = photon_noise/photonloss_factor;
        if(verbose) printf("apply_psf() predicting %g lost photons\n",lost_photons);

        if(psf_type == GAUSS)
        {
            /* calculate the radius beyond which only 0.5 photons will fall */
            psf_radius = 1+ceil( sqrt(-log(lost_photons/max_I)/log(4.0)/2.0)*fwhm_pixels );
            if(verbose) printf("  auto-selected psf_radius = %d x %d pixels for rendering kernel\n",psf_radius,psf_radius);
        }
        if(psf_type == FIBER)
        {
            /* calculate the radius r beyond which only 0.5 photons will fall */
            /* r = sqrt((g*(max_I/0.5))**2-g**2)
                 ~ 2*g*max_I */
            psf_radius = 1+ceil( g*(max_I/lost_photons)  );
            if(verbose) printf("  auto-selected psf_radius = %d x %d pixels for rendering kernel\n",psf_radius,psf_radius);
        }
        if(psf_radius == 0) psf_radius = 1;
    }
    /* limit psf kernel to be no bigger than 4x the input image */
    maxwidth = fpixels;
    if(spixels > maxwidth) maxwidth = spixels;
    if(psf_radius > maxwidth) psf_radius = maxwidth;
    kernel_size = 2*psf_radius+1;
    if(verbose>6) printf("apply_psf() kernel_size= %d\n",kernel_size);

    /* now alocate enough space to store the PSF kernel image */
    kernel = (double *) calloc(kernel_size*kernel_size,sizeof(double));
    if(kernel == NULL)
    {
        perror("apply_psf: could not allocate memory for PSF kernel");
        exit(9);
    }

    /* cache the PSF in an array */
    for(dy=-psf_radius;dy<=psf_radius;++dy)
    {
        for(dx=-psf_radius;dx<=psf_radius;++dx)
        {
            rsq = dx*dx+dy*dy;
            if(rsq > psf_radius*psf_radius) continue;

            /* this could be more efficient */
            k = kernel_size*(kernel_size/2+dy)+kernel_size/2+dx;


            if( psf_type == GAUSS ) {
                kernel[k] = integrate_gauss_over_pixel(dx,dy,fwhm_pixels,1.0);
            }
            if( psf_type == FIBER ) {
                kernel[k] = integrate_fiber_over_pixel(dx,dy,g,1.0);
            }
        }
    }

    /* implement PSF  */
    double sum_in = 0.0, sum_out = 0.0;
    sumn = 0;
    for(i=0;i<pixels;++i)
    {
        x0 = i%fpixels;
        y0 = (i-x0)/fpixels;

        /* skip if there is nothing to add */
        if(inimage[i] <= 0.0) continue;

        sum_in += inimage[i]; ++sumn;

        if(user_psf_radius != 0)
        {
            psf_radius = user_psf_radius;
        }
        else
        {
            /* at what level will an error in intensity be lost? */
            photon_noise = sqrt(inimage[i]);
            lost_photons = photon_noise/photonloss_factor;

            if(psf_type == GAUSS)
            {
                /* calculate the radius beyond which only 0.5 photons will fall
                   r = sqrt(-log(lost_photons/total_photons)/log(4)/2)*fwhm */
                psf_radius = 1+ceil( sqrt(-log(lost_photons/inimage[i])/log(16.0))*fwhm_pixels );
//              printf("  auto-selected psf_radius = %d pixels\n",psf_radius);
            }
            if(psf_type == FIBER)
            {
                /* calculate the radius beyond which only 0.5 photons will fall
                   r = sqrt((g*(total_photons/lost_photons))**2-g**2)
                     ~ g*total_photons/lost_photons */
                psf_radius = 1+ceil( g*(inimage[i]/lost_photons)  );
//              printf("  (%d,%d) auto-selected psf_radius = %d pixels\n",x0,y0,psf_radius);
            }
        }
        if(psf_radius == 0) psf_radius = 1;
        /* limit psf kernel to be no bigger than 4x the input image */
        maxwidth = fpixels;
        if(spixels > maxwidth) maxwidth = spixels;
        if(psf_radius > maxwidth) psf_radius = maxwidth;

        /* given the radius, how many photons will escape? */
        if(psf_type == GAUSS)
        {
            /* r = sqrt(-log(lost_photons/total_photons)/log(16))*fwhm */
            /* lost_photons = total_photons*exp(-log(16)*(r^2/fwhm^2)) */
            rsq = psf_radius;
            rsq = rsq/fwhm_pixels;
            rsq = rsq*rsq;
            lost_photons = inimage[i]*exp(-log(16.0)*rsq);
        }
        if(psf_type == FIBER)
        {
            /* r ~ g*total_photons/lost_photons
               normalized integral from r=inf to "r" :  g/sqrt(g**2+r**2) */
            lost_photons = inimage[i]*g/sqrt(g*g+psf_radius*psf_radius);
        }
        /* accumulate this so we can add it to the whole image */
        total_lost_photons += lost_photons;

        for(dx=-psf_radius;dx<=psf_radius;++dx)
        {
            for(dy=-psf_radius;dy<=psf_radius;++dy)
            {
                /* this could be more efficient */
                k = kernel_size*(kernel_size/2+dy)+kernel_size/2+dx;
                if(kernel[k] == 0.0) continue;

                rsq = dx*dx+dy*dy;
                if(rsq > psf_radius*psf_radius) continue;
                x = x0+dx;
                y = y0+dy;
                if(x<0 || x>fpixels) continue;
                if(y<0 || y>spixels) continue;

                /* index into output array */
                j = y*fpixels+x;
                /* do not wander off the output array */
                if(j<0 || j > pixels) continue;

                outimage[j] += inimage[i]*kernel[k];
            }
        }
    }
    /* now we have some lost photons, add them back "everywhere" */
    lost_photons = total_lost_photons/pixels;
    if(verbose) printf("adding back %g lost photons\n",total_lost_photons);
    for(i=0;i<pixels;++i)
    {
        sum_out += outimage[i];
        outimage[i] += lost_photons;
        if(verbose>7 && i==pixels/2) printf("apply_psf() pixel=%d in= %g out= %g \n",i,inimage[i],outimage[i]);
    }
    if(verbose>7) printf("apply_psf() sum_in=  %g\n",sum_in);
    if(verbose>7) printf("apply_psf() sum_out= %g\n",sum_out);
    if(verbose>7) printf("apply_psf() sum_out= %g (after correction)\n",sum_out+total_lost_photons);

    /* don't need kernel anymore. */
    free(kernel);

    i=pixels/2;
    if(verbose>7) printf("apply_psf() pixel=%d in= %g out= %g \n",i,inimage[i],outimage[i]);

    /* and now.  No idea how to exchange buffers without confusing Python, so lets just copy it back */
    memcpy(inimage,outimage,pixels*sizeof(double));

    free(outimage);
    return;
}
// end of apply_psf()




// function to add different types of noise to the image
void
nanoBragg::add_noise()
{
    encapsulated_twodev image_deviates;
    encapsulated_twodev pixel_deviates;
    int i = 0;
    long cseed;

    double expected_photons,observed_photons,adu;
    /* refer to raw_pixels pixel data */
    floatimage = raw_pixels.begin();

    /* don't bother with this loop if calibration is perfect
     NOTE: applying calibration before Poisson noise simulates loss of photons before the detector
     NOTE: applying calibration after Poisson noise simulates systematics in read-out electronics
     here we do the latter */

    /* re-start the RNG */
    seed = -labs(seed);

    if(verbose) printf("applying calibration at %g%%, flicker noise at %g%%\n",calibration_noise*100.,flicker_noise*100.);
    sum = max_I = 0.0;
    i = sumn = 0;
    for(spixel=0;spixel<spixels;++spixel)
    {
        for(fpixel=0;fpixel<fpixels;++fpixel)
        {

            /* allow for just one part of detector to be rendered */
            if(fpixel < roi_xmin || fpixel > roi_xmax || spixel < roi_ymin || spixel > roi_ymax)
            {
                ++i; continue;
            }
            /* allow for the use of a mask */
            if(maskimage != NULL)
            {
                /* skip any flagged pixels in the mask */
                if(maskimage[i] == 0)
                {
                    ++i; continue;
                }
            }

            /* take input image to be ideal photons/pixel */
            expected_photons = floatimage[i];

            /* negative photons should be taken as invalid? */
            if(expected_photons < 0.0)
            {
                ++i; continue;
            }

            /* simulate 1/f noise in source */
            if(flicker_noise > 0.0){
                expected_photons *= ( 1.0 + flicker_noise * image_deviates.gaussdev( &seed ) );
            }
            /* simulate photon-counting error */
            observed_photons = image_deviates.poidev( expected_photons, &seed );

            /* now we overwrite the flex array, it is now observed, rather than expected photons */
            floatimage[i] = observed_photons;

            /* accumulate number of photons, and keep track of max */
            if(floatimage[i] > max_I) {
                max_I = floatimage[i];
                max_I_x = fpixel;
                max_I_y = spixel;
            }
            sum += observed_photons;
            ++sumn;

            ++i;
        }
    }
    if(verbose) printf("%.0f photons generated on noise image, max= %f at ( %.0f, %.0f )\n",sum,max_I,max_I_x,max_I_y);



    if(calibration_noise > 0.0)
    {
        /* calibration is same from shot to shot, so use well-known seed */
        cseed = -labs(calib_seed);
        sum = max_I = 0.0;
        i = sumn = 0;
        for(spixel=0;spixel<spixels;++spixel)
        {
            for(fpixel=0;fpixel<fpixels;++fpixel)
            {
                /* allow for just one part of detector to be rendered */
                if(fpixel < roi_xmin || fpixel > roi_xmax || spixel < roi_ymin || spixel > roi_ymax)
                {
                    ++i; continue;
                }
                /* allow for the use of a mask */
                if(maskimage != NULL)
                {
                    /* skip any flagged pixels in the mask */
                    if(maskimage[i] == 0)
                    {
                        ++i; continue;
                    }
                }

                /* calibration is same from shot to shot, but varies from pixel to pixel */
                floatimage[i] *= ( 1.0 + calibration_noise * pixel_deviates.gaussdev( &cseed ) );

                /* accumulate number of photons, and keep track of max */
                if(floatimage[i] > max_I) {
                    max_I = floatimage[i];
                    max_I_x = fpixel;
                    max_I_y = spixel;
                }
                sum += floatimage[i];
                ++sumn;

                ++i;
            }
        }
    }
    if(verbose) printf("%.0f photons after calibration error, max= %f at ( %.0f, %.0f )\n",sum,max_I,max_I_x,max_I_y);


    /* now would be a good time to implement PSF?  before we add read-out noise */

    /* now that we have photon count at each point, implement any PSF */
    if(psf_type != UNKNOWN && psf_fwhm > 0.0)
    {
        /* report on sum before the PSF is applied */
        if(verbose) printf("%.0f photons on noise image before PSF\n",sum);
        /* start with a clean slate */
        if(verbose) printf("  applying PSF width = %g um\n",psf_fwhm*1e6);

        apply_psf(psf_type, psf_fwhm/pixel_size, 0);

        /* the flex array is now the blurred version of itself, ready for read-out noise */
    }


    if(verbose) printf("adu = quantum_gain= %g * observed_photons + offset= %g + readout_noise= %g\n",quantum_gain,adc_offset,readout_noise);
    sum = max_I = 0.0;
    i = sumn = 0;
    for(spixel=0;spixel<spixels;++spixel)
    {
        for(fpixel=0;fpixel<fpixels;++fpixel)
        {
            /* allow for just one part of detector to be rendered */
            if(fpixel < roi_xmin || fpixel > roi_xmax || spixel < roi_ymin || spixel > roi_ymax)
            {
                ++i; continue;
            }
            /* allow for the use of a mask */
            if(maskimage != NULL)
            {
                /* skip any flagged pixels in the mask */
                if(maskimage[i] == 0)
                {
                    ++i; continue;
                }
            }

                /* convert photon signal to pixel units */
                adu = floatimage[i]*quantum_gain + adc_offset;

                /* readout noise is in pixel units (adu) */
                if(readout_noise > 0.0){
                    adu += readout_noise * image_deviates.gaussdev( &seed );
            }

            /* once again, overwriting flex array, this time in ADU units */
            floatimage[i] = adu;

            if(adu > max_I) {
                max_I = adu;
                max_I_x = fpixel;
                max_I_y = spixel;
            }
            sum += adu;
            ++sumn;
            ++i;
        }
    }
    if(verbose) printf("%.0f net adu generated on final image, max= %f at ( %.0f, %.0f )\n",sum-adc_offset*sumn,max_I,max_I_x,max_I_y);
}
// end of add_noise()

double nanoBragg::get_intfile_scale(double intfile_scale) const {
    const double* floatimage = raw_pixels.begin();
    double max_value = (double)std::numeric_limits<unsigned short int>::max();
    double saturation = floor(max_value - 1 );
    /* output as ints */

    unsigned short int intimage;
    double max_I = this-> max_I;
    double max_I_x = this-> max_I_x;
    double max_I_y = this-> max_I_y;
    if(intfile_scale <= 0.0){
        /* need to auto-scale */
        int i=0;
        for(int spixel=0;spixel<spixels;++spixel)
        {
            for(int fpixel=0;fpixel<fpixels;++fpixel)
            {
                if(i==0 || max_I < floatimage[i])
                {
                    max_I = floatimage[i];
                    max_I_x = fpixel;
                    max_I_y = spixel;
                }
                ++i;
            }
        }
        if(verbose) printf("providing default scaling: max_I = %g @ (%g %g)\n",max_I,max_I_x,max_I_y);
        intfile_scale = 1.0;
        if(max_I>0.0) intfile_scale = 55000.0/(max_I);
    }
    return intfile_scale;
}

void
nanoBragg::to_smv_format_streambuf(boost_adaptbx::python::streambuf & output,
    double intfile_scale, int const&debug_x, int const& debug_y) const {
    boost_adaptbx::python::streambuf::ostream os(output);
    const double* floatimage = raw_pixels.begin();
    double max_value = (double)std::numeric_limits<unsigned short int>::max();
    double saturation = floor(max_value - 1 );
    /* output as ints */

    unsigned short int intimage;
    double max_I = this-> max_I;
    double max_I_x = this-> max_I_x;
    double max_I_y = this-> max_I_y;
    if(intfile_scale <= 0.0){
        /* need to auto-scale */
        int i=0;
        for(int spixel=0;spixel<spixels;++spixel)
        {
            for(int fpixel=0;fpixel<fpixels;++fpixel)
            {
                if(fpixel==debug_x && spixel==debug_y) printf("DEBUG: pixel # %d at (%d,%d) has value %g\n",i,fpixel,spixel,floatimage[i]);
                if(i==0 || max_I < floatimage[i])
                {
                    max_I = floatimage[i];
                    max_I_x = fpixel;
                    max_I_y = spixel;
                }
                ++i;
            }
        }
        if(verbose) printf("providing default scaling: max_I = %g @ (%g %g)\n",max_I,max_I_x,max_I_y);
        intfile_scale = 1.0;
        if(max_I>0.0) intfile_scale = 55000.0/(max_I);
    }
    if(verbose) printf("scaling data by: intfile_scale = %g\n",intfile_scale);

    double sum = 0.0;
    max_I = 0.0;
    int i = 0;
    for(int spixel=0;spixel<spixels;++spixel)
    {
        for(int fpixel=0;fpixel<fpixels;++fpixel)
        {
            /* no noise, just use intfile_scale */

            intimage = (unsigned short int) (std::min(saturation, floatimage[i]*intfile_scale ));
            os.write((char *) &intimage, sizeof(unsigned short int));

            if(verbose>90) printf("DEBUG #%d %g -> %g -> %d\n",i,floatimage[i],floatimage[i]*intfile_scale,intimage);

            if((double) intimage > max_I || i==0) {
                max_I = (double) intimage;
                max_I_x = fpixel;
                max_I_y = spixel;
            }
            if(fpixel==debug_x && spixel==debug_y) printf("DEBUG: pixel # %d at (%d,%d) has int value %d\n",i,fpixel,spixel,intimage);

            sum += intimage;
            ++i;
        }
    }
    // os << "Hello world";

}

void
nanoBragg::to_smv_format(
    std::string const& fileout, double intfile_scale, int debug_x, int debug_y)
{
    pixels = spixels * fpixels;
    floatimage = raw_pixels.begin();
    FILE* outfile;
    double max_value = (double)std::numeric_limits<unsigned short int>::max();
    double saturation = floor(max_value - 1 );
    static const char *byte_order = "little_endian";

    /* output as ints */
    af::versa<unsigned short int, af::c_grid<2> > intimage_v(
      af::c_grid<2> (spixels,fpixels));
    unsigned short int * intimage = intimage_v.begin();

    if(intfile_scale <= 0.0)
    {
        /* need to auto-scale */
        i=0;
        for(spixel=0;spixel<spixels;++spixel)
        {
            for(fpixel=0;fpixel<fpixels;++fpixel)
            {
                if(fpixel==debug_x && spixel==debug_y) printf("DEBUG: pixel # %d at (%d,%d) has value %g\n",i,fpixel,spixel,floatimage[i]);
                if(i==0 || max_I < floatimage[i])
                {
                    max_I = floatimage[i];
                    max_I_x = fpixel;
                    max_I_y = spixel;
                }
                ++i;
            }
        }
        if(verbose) printf("providing default scaling: max_I = %g @ (%g %g)\n",max_I,max_I_x,max_I_y);
        intfile_scale = 1.0;
        if(max_I>0.0) intfile_scale = 55000.0/(max_I);
    }
    if(verbose) printf("scaling data by: intfile_scale = %g\n",intfile_scale);

    sum = max_I = 0.0;
    int i = 0;
    for(spixel=0;spixel<spixels;++spixel)
    {
        for(fpixel=0;fpixel<fpixels;++fpixel)
        {

            /* no noise, just use intfile_scale */
            intimage[i] = (unsigned short int) (std::min(saturation,
                           floatimage[i]*intfile_scale ));

            if(verbose>90) printf("DEBUG #%d %g -> %g -> %d\n",i,floatimage[i],floatimage[i]*intfile_scale,intimage[i]);

            if((double) intimage[i] > max_I || i==0) {
                max_I = (double) intimage[i];
                max_I_x = fpixel;
                max_I_y = spixel;
            }
            if(fpixel==debug_x && spixel==debug_y) printf("DEBUG: pixel # %d at (%d,%d) has int value %d\n",i,fpixel,spixel,intimage[i]);

            sum += intimage[i];
            ++i;
        }
    }
    if (verbose){
      printf("writing %s as %d-byte integers; sum= %g max= %g @ ( %g %g)\n",fileout.c_str(),
            (int)sizeof(unsigned short int),sum,max_I,max_I_x,max_I_y);
    }
    outfile = fopen(fileout.c_str(),"wb");
    fprintf(outfile,"{\nHEADER_BYTES=1024;\nDIM=2;\nBYTE_ORDER=%s;\nTYPE=unsigned_short;\n",byte_order);
    fprintf(outfile,"SIZE1=%d;\nSIZE2=%d;\nPIXEL_SIZE=%g;\nDISTANCE=%g;\n",fpixels,spixels,pixel_size*1000.0,distance*1000.0);
    fprintf(outfile,"WAVELENGTH=%g;\n",lambda0*1e10);
    fprintf(outfile,"BEAM_CENTER_X=%g;\nBEAM_CENTER_Y=%g;\n",Xbeam*1000.0,Ybeam*1000);
    fprintf(outfile,"ADXV_CENTER_X=%g;\nADXV_CENTER_Y=%g;\n",Fbeam*1000.0,(detsize_s-Sbeam)*1000);
    fprintf(outfile,"MOSFLM_CENTER_X=%g;\nMOSFLM_CENTER_Y=%g;\n",(Sbeam-0.5*pixel_size)*1000.0,(Fbeam-0.5*pixel_size)*1000);
    fprintf(outfile,"DENZO_X_BEAM=%g;\nDENZO_Y_BEAM=%g;\n",(Sbeam+0.0*pixel_size)*1000.0,(Fbeam+0.0*pixel_size)*1000);
    fprintf(outfile,"DIALS_ORIGIN=%g,%g,%g\n",dials_origin[1],dials_origin[2],dials_origin[3]);
    fprintf(outfile,"XDS_ORGX=%g;\nXDS_ORGY=%g;\n",ORGX,ORGY);
    fprintf(outfile,"CLOSE_DISTANCE=%g;\n",close_distance*1000.0);
    fprintf(outfile,"PHI=%g;\nOSC_START=%g;\nOSC_RANGE=%g;\n",phi0*RTD,phi0*RTD,osc*RTD);
    fprintf(outfile,"TIME=%g;\n",exposure);
    fprintf(outfile,"TWOTHETA=%g;\n",detector_twotheta*RTD);
    fprintf(outfile,"DETECTOR_SN=000;\n");
    fprintf(outfile,"ADC_OFFSET=%g;\n",adc_offset);
    fprintf(outfile,"BEAMLINE=fake;\n");
    fprintf(outfile,"}\f");
    while ( ftell(outfile) < 1024 ){ fprintf(outfile," "); };
    fwrite(intimage,sizeof(unsigned short int),pixels,outfile);

    fclose(outfile);

    return;
}




/* Fourier transform of a grating */
double sincg(double x,double N) {
    if(x==0.0) return N;

    return sin(x*N)/sin(x);
}

/* Fourier transform of a sphere */
double sinc3(double x) {
    if(x==0.0) return 1.0;

    return 3.0*(sin(x)/x-cos(x))/(x*x);
}

double sinc_conv_sinc3(double x) {
    if(x==0.0) return 1.0;

    return 3.0*(sin(x)-x*cos(x))/(x*x*x);
}


double *rotate(double *v, double *newv, double phix, double phiy, double phiz) {

    double rxx,rxy,rxz,ryx,ryy,ryz,rzx,rzy,rzz;
    double new_x,new_y,new_z,rotated_x,rotated_y,rotated_z;

    new_x=v[1];
    new_y=v[2];
    new_z=v[3];

    if(phix != 0){
        /* rotate around x axis */
        //rxx= 1;         rxy= 0;         rxz= 0;
        ryx= 0;         ryy= cos(phix); ryz=-sin(phix);
        rzx= 0;         rzy= sin(phix); rzz= cos(phix);

        rotated_x = new_x;
        rotated_y = new_y*ryy + new_z*ryz;
        rotated_z = new_y*rzy + new_z*rzz;
        new_x = rotated_x; new_y = rotated_y; new_z = rotated_z;
    }

    if(phiy != 0) {
        /* rotate around y axis */
        rxx= cos(phiy); rxy= 0;         rxz= sin(phiy);
        //ryx= 0;         ryy= 1;         ryz= 0;
        rzx=-sin(phiy); rzy= 0;         rzz= cos(phiy);

        rotated_x = new_x*rxx + new_y*rxy + new_z*rxz;
        rotated_y = new_y;
        rotated_z = new_x*rzx + new_y*rzy + new_z*rzz;
        new_x = rotated_x; new_y = rotated_y; new_z = rotated_z;
    }

    if(phiz != 0){
        /* rotate around z axis */
        rxx= cos(phiz); rxy=-sin(phiz); rxz= 0;
        ryx= sin(phiz); ryy= cos(phiz); ryz= 0;
        //rzx= 0;         rzy= 0;         rzz= 1;

        rotated_x = new_x*rxx + new_y*rxy ;
        rotated_y = new_x*ryx + new_y*ryy;
        rotated_z = new_z;
        new_x = rotated_x; new_y = rotated_y; new_z = rotated_z;
    }

    newv[1]=new_x;
    newv[2]=new_y;
    newv[3]=new_z;

    return newv;
}



/* rotate a point about a unit vector axis */
double *rotate_axis(double *v, double *newv, double *axis, double phi) {

    double sinphi = sin(phi);
    double cosphi = cos(phi);
    double dot = (axis[1]*v[1]+axis[2]*v[2]+axis[3]*v[3])*(1.0-cosphi);
    double temp[4];

    temp[1] = axis[1]*dot+v[1]*cosphi+(-axis[3]*v[2]+axis[2]*v[3])*sinphi;
    temp[2] = axis[2]*dot+v[2]*cosphi+(+axis[3]*v[1]-axis[1]*v[3])*sinphi;
    temp[3] = axis[3]*dot+v[3]*cosphi+(-axis[2]*v[1]+axis[1]*v[2])*sinphi;
    newv[1]=temp[1]; newv[2]=temp[2]; newv[3]=temp[3];

    return newv;
}



/* rotate a vector using a 9-element unitary matrix */
double *rotate_umat(double *v, double *newv, double umat[9]) {

    double uxx,uxy,uxz,uyx,uyy,uyz,uzx,uzy,uzz;

    /* for convenience, assign matrix x-y coordinate */
    uxx = umat[0];
    uxy = umat[1];
    uxz = umat[2];
    uyx = umat[3];
    uyy = umat[4];
    uyz = umat[5];
    uzx = umat[6];
    uzy = umat[7];
    uzz = umat[8];

    /* rotate the vector (x=1,y=2,z=3) */
    newv[1] = uxx*v[1] + uxy*v[2] + uxz*v[3];
    newv[2] = uyx*v[1] + uyy*v[2] + uyz*v[3];
    newv[3] = uzx*v[1] + uzy*v[2] + uzz*v[3];

    return newv;
}




/* returns a 9-element unitary matrix for a random isotropic rotation on a spherical cap of diameter "mosaicity" */
/* mosaic = 90 deg is a full sphere */
double *mosaic_rotation_umat(double mosaicity, double umat[9], long *seed)
{
//    double ran1(long *idum);
    double r1,r2,r3,xyrad,rot;
    double v1,v2,v3;
    double t1,t2,t3,t6,t7,t8,t9,t11,t12,t15,t19,t20,t24;
    double uxx,uxy,uxz,uyx,uyy,uyz,uzx,uzy,uzz;

    /* make three random uniform deviates on [-1:1] */
    r1= (double) 2.0*ran1(seed)-1.0;
    r2= (double) 2.0*ran1(seed)-1.0;
    r3= (double) 2.0*ran1(seed)-1.0;

    xyrad = sqrt(1.0-r2*r2);
    rot = mosaicity*powf((1.0-r3*r3),(1.0/3.0));

    v1 = xyrad*sin(M_PI*r1);
    v2 = xyrad*cos(M_PI*r1);
    v3 = r2;

    /* commence incomprehensible quaternion calculation */
    t1 =  cos(rot);
    t2 =  1.0 - t1;
    t3 =  v1*v1;
    t6 =  t2*v1;
    t7 =  t6*v2;
    t8 =  sin(rot);
    t9 =  t8*v3;
    t11 = t6*v3;
    t12 = t8*v2;
    t15 = v2*v2;
    t19 = t2*v2*v3;
    t20 = t8*v1;
    t24 = v3*v3;

    /* populate the unitary rotation matrix */
    umat[0] = uxx = t1 + t2*t3;
    umat[1] = uxy = t7 - t9;
    umat[2] = uxz = t11 + t12;
    umat[3] = uyx = t7 + t9;
    umat[4] = uyy = t1 + t2*t15;
    umat[5] = uyz = t19 - t20;
    umat[6] = uzx = t11 - t12;
    umat[7] = uzy = t19 + t20;
    umat[8] = uzz = t1 + t2*t24;

    /* return pointer to the provided array, in case that is useful */
    return umat;
}

/* convert a unitary rotation matrix into misseting angles
   rotx roty rotz are returned as missets[1] missets[2] missets[3] */
double *umat2misset(double umat[9],double *missets)
{
    double uxx,uxy,uxz,uyx,uyy,uyz,uzx,uzy,uzz;
    double m,mx,my,mz;
    double xcy_x,xcy_y,xcy_z;
    double ycz_x,ycz_y,ycz_z;
    double zcx_x,zcx_y,zcx_z;
    double rotx,roty,rotz;

    uxx=umat[0];uxy=umat[1];uxz=umat[2];
    uyx=umat[3];uyy=umat[4];uyz=umat[5];
    uzx=umat[6];uzy=umat[7];uzz=umat[8];

    /* or transpose? */
//    uxx=umat[1];uyx=umat[2];uzx=umat[3];
//    uxy=umat[4];uyy=umat[5];uzy=umat[6];
//    uxz=umat[7];uyz=umat[8];uzz=umat[9];

    /* make sure it is unitary */
    mx = sqrt(uxx*uxx+uxy*uxy+uxz*uxz);
    my = sqrt(uyx*uyx+uyy*uyy+uyz*uyz);
    mz = sqrt(uzx*uzx+uzy*uzy+uzz*uzz);
    if(mx>0){uxx/=mx;uxy/=mx;uxz/=mx;};
    if(my>0){uyx/=my;uyy/=my;uyz/=my;};
    if(mz>0){uzx/=mz;uzy/=mz;uzz/=mz;};

    if(mx>=0 && my<=0 && mz<=0)
    {
        uyx=0;uyy=1;uyz=0;
        uzx=0;uzy=0;uzz=1;
    }
    if(mx<=0 && my>=0 && mz<=0)
    {
        uxx=1;uxy=0;uxz=0;
        uzx=0;uzy=0;uzz=1;
    }
    if(mx<=0 && my<=0 && mz>=0)
    {
        uxx=1;uxy=0;uxz=0;
        uyx=0;uyy=1;uyz=0;
    }

    /* cross products to check normality */
    xcy_x = uxy*uyz - uxz*uyy;
    xcy_y = uxz*uyx - uxx*uyz;
    xcy_z = uxx*uyy - uxy*uyx;
    m=sqrt(xcy_x*xcy_x+xcy_y*xcy_y+xcy_z*xcy_z);
    if(m>0){xcy_x/=m;xcy_y/=m;xcy_z/=m;};

    ycz_x = uyy*uzz - uyz*uzy;
    ycz_y = uyz*uzx - uyx*uzz;
    ycz_z = uyx*uzy - uyy*uzx;
    m=sqrt(ycz_x*ycz_x+ycz_y*ycz_y+ycz_z*ycz_z);
    if(m>0){ycz_x/=m;ycz_y/=m;ycz_z/=m;};

    zcx_x = uzy*uxz - uzz*uxy;
    zcx_y = uzz*uxx - uzx*uxz;
    zcx_z = uzx*uxy - uzy*uxx;
    m=sqrt(zcx_x*zcx_x+zcx_y*zcx_y+zcx_z*zcx_z);
    if(m>0){zcx_x/=m;zcx_y/=m;zcx_z/=m;};

    /* substitute any empty vectors for cross-product of other two */
    if(mx<=0){uxx=ycz_x;uxy=ycz_y;uxz=ycz_z;};
    if(my<=0){uyx=zcx_x;uyy=zcx_y;uyz=zcx_z;};
    if(mz<=0){uzx=xcy_x;uzy=xcy_y;uzz=xcy_z;};


    /* cross products to check normality */
    xcy_x = uxy*uyz - uxz*uyy;
    xcy_y = uxz*uyx - uxx*uyz;
    xcy_z = uxx*uyy - uxy*uyx;
    m=sqrt(xcy_x*xcy_x+xcy_y*xcy_y+xcy_z*xcy_z);
    if(m>0){xcy_x/=m;xcy_y/=m;xcy_z/=m;}

    ycz_x = uyy*uzz - uyz*uzy;
    ycz_y = uyz*uzx - uyx*uzz;
    ycz_z = uyx*uzy - uyy*uzx;
    m=sqrt(ycz_x*ycz_x+ycz_y*ycz_y+ycz_z*ycz_z);
    if(m>0){ycz_x/=m;ycz_y/=m;ycz_z/=m;};

    zcx_x = uzy*uxz - uzz*uxy;
    zcx_y = uzz*uxx - uzx*uxz;
    zcx_z = uzx*uxy - uzy*uxx;
    m=sqrt(zcx_x*zcx_x+zcx_y*zcx_y+zcx_z*zcx_z);
    if(m>0){zcx_x/=m;zcx_y/=m;zcx_z/=m;};


    /* substitute any empty vectors for cross-product of other two */
    if(mx<=0){uxx=ycz_x;uxy=ycz_y;uxz=ycz_z;};
    if(my<=0){uyx=zcx_x;uyy=zcx_y;uyz=zcx_z;};
    if(mz<=0){uzx=xcy_x;uzy=xcy_y;uzz=xcy_z;};



    /* make sure it is unitary */
    mx = sqrt(uxx*uxx+uxy*uxy+uxz*uxz);
    my = sqrt(uyx*uyx+uyy*uyy+uyz*uyz);
    mz = sqrt(uzx*uzx+uzy*uzy+uzz*uzz);
    if(mx>0){uxx/=mx;uxy/=mx;uxz/=mx;};
    if(my>0){uyx/=my;uyy/=my;uyz/=my;};
    if(mz>0){uzx/=mz;uzy/=mz;uzz/=mz;};

    /* see if its really orthonormal? */

    if(uzx*uzx < 1.0)
    {
        rotx = atan2(uzy,uzz);
        roty = atan2(-uzx,sqrt(uzy*uzy+uzz*uzz));
        rotz = atan2(uyx,uxx);
    }
    else
    {
        rotx = atan2(1.0,1.0)*4.0;
        roty = atan2(1.0,1.0)*2.0;
        rotz = atan2(uxy,-uyy);
    }

    missets[1] = rotx;
    missets[2] = roty;
    missets[3] = rotz;
    return missets;
}



/* random number generators */


/* generate Lorentzian deviate with FWHM = 2 */
double lorentzdev(long *seed) {
//    double ran1(long *idum);

    return tan(M_PI*(ran1(seed)-0.5));
}

/* return triangular deviate with FWHM = 1 */
double triangledev(long *seed) {
    double ran1(long *idum);
    double value;

    value = ran1(seed);
    if(value > 0.5){
        value = sqrt(2*(value-0.5))-1;
    }else{
        value = 1-sqrt(2*value);
    }

    return value;
}



double expdev(long *idum)
{
    double dum;

    do
    dum=ran1(idum);
    while( dum == 0.0);
    return -log(dum);
}



/* ln of the gamma function */
double gammln(double xx)
{
    double x,y,tmp,ser;
    static double cof[6]={76.18009172947146,-86.50532032941677,
    24.01409824083091,-1.231739572450155,
    0.1208650973866179e-2,-0.5395239384953e-5};
    int j;

    y=x=xx;
    tmp=x+5.5;
    tmp -= (x+0.5)*log(tmp);
    ser = 1.000000000190015;
    for(j=0;j<=5;++j) ser += cof[j]/++y;

    return -tmp+log(2.5066282746310005*ser/x);
}





/* returns a uniform random deviate between 0 and 1 */
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

double ran1(long *idum)
{
    int j;
    long k;
    static long iy=0;
    static long iv[NTAB];
    double temp;

    if (*idum <= 0 || !iy) {
        /* first time around.  don't want idum=0 */
        if(-(*idum) < 1) *idum=1;
        else *idum = -(*idum);

        /* load the shuffle table */
        for(j=NTAB+7;j>=0;j--) {
            k=(*idum)/IQ;
            *idum=IA*(*idum-k*IQ)-IR*k;
            if(*idum < 0) *idum += IM;
            if(j < NTAB) iv[j] = *idum;
        }
        iy=iv[0];
    }
    /* always start here after initializing */
    k=(*idum)/IQ;
    *idum=IA*(*idum-k*IQ)-IR*k;
    if (*idum < 0) *idum += IM;
    j=iy/NDIV;
    iy=iv[j];
    iv[j] = *idum;
    if((temp=AM*iy) > RNMX) return RNMX;
    else return temp;
}


void polint(double *xa, double *ya, double x, double *y)
{
        double x0,x1,x2,x3;
        x0 = (x-xa[1])*(x-xa[2])*(x-xa[3])*ya[0]/((xa[0]-xa[1])*(xa[0]-xa[2])*(xa[0]-xa[3]));
        x1 = (x-xa[0])*(x-xa[2])*(x-xa[3])*ya[1]/((xa[1]-xa[0])*(xa[1]-xa[2])*(xa[1]-xa[3]));
        x2 = (x-xa[0])*(x-xa[1])*(x-xa[3])*ya[2]/((xa[2]-xa[0])*(xa[2]-xa[1])*(xa[2]-xa[3]));
        x3 = (x-xa[0])*(x-xa[1])*(x-xa[2])*ya[3]/((xa[3]-xa[0])*(xa[3]-xa[1])*(xa[3]-xa[2]));
        *y = x0+x1+x2+x3;
}



void polin2(double *x1a, double *x2a, double **ya, double x1, double x2, double *y)
{
        void polint(double *xa, double *ya, double x, double *y);
        int j;
        double ymtmp[4];
        for (j=1;j<=4;j++) {
                polint(x2a,ya[j-1],x2,&ymtmp[j-1]);
        }
        polint(x1a,ymtmp,x1,y);
}


void polin3(double *x1a, double *x2a, double *x3a, double ***ya, double x1,
        double x2, double x3, double *y)
{
        void polint(double *xa, double ya[], double x, double *y);
        void polin2(double *x1a, double *x2a, double **ya, double x1,double x2, double *y);
        void polin1(double *x1a, double *ya, double x1, double *y);
        int j;
        double ymtmp[4];

        for (j=1;j<=4;j++) {
            polin2(x2a,x3a,&ya[j-1][0],x2,x3,&ymtmp[j-1]);
        }
        polint(x1a,ymtmp,x1,y);
}


/* FWHM = integral = 1 */
double ngauss2D(double x,double y)
{
    return log(16.0)/M_PI*exp(-log(16.0)*(x*x+y*y));
}
double ngauss2Dinteg(double x,double y)
{
    return 0.125*(erf(2.0*x*sqrt(log(2.0)))*erf(y*sqrt(log(16.0)))*sqrt(log(16.0)/log(2.0)));
}









/* read in multi-column text file to list of double arrays */
/* provide address of undeclared arrays on command line */
size_t read_text_file(char *filename, size_t nargs, ... )
{
    /* maximum of 10240-character lines? */
    char text[10240];
    char *token;
    const char delimiters[] = " \t,;:!";
    const char numberstuf[] = "0123456789-+.EGeg";

    unsigned long line,lines;
    unsigned long i,j;
    double value;
    double *data;
    double **pointer;
    va_list arglist;
    FILE *infile = NULL;

    infile = fopen(filename,"r");
    if(infile == NULL) {
        perror("fopen()");
        return 0;
    }
    lines=0;
    while ( fgets ( text, sizeof text, infile ) != NULL ) {
        token = text;
        token += strspn(token,delimiters);
        if(strcmp(token,"\n")==0) {
            //printf("blank\n");
            continue;
        }
        ++lines;
    }
    rewind(infile);

    /* allocate memory for arrays */
    va_start( arglist, nargs);
    for(i=0;i<nargs;++i){
        /* allocate the array */
        data = (double*) malloc((lines+10)*sizeof(double));
        /* initialize with missing number flags */
        for(j=0;j<lines+10;++j) {
            data[j] = NAN;
        }
        /* get argument (pointer to pointer) */
        pointer = va_arg(arglist, double **);
        /* change the value of what the arg points to */
        *pointer = data;
        /* now the pointer provided as an argument points to
        something */
    }
    va_end(arglist);

    line = 0;
    while ( fgets ( text, sizeof text, infile ) != NULL ) { /* read a line */

        token = text;
        token += strspn(token,delimiters);
        if(strcmp(token,"\n")==0) {
            //printf("blank\n");
            continue;
        }
        i=0;
        va_start( arglist, nargs);
        do
        {
            value=atof(token);
            /* get argument */
            pointer = va_arg(arglist, double **);
            /* retrieve data array's address */
            data = *pointer;
            data[line] = value;

            token += strspn(token,numberstuf);
            if (strcmp(token,"\n")==0) continue;
            token += strcspn(token,delimiters);
            token += strspn(token,delimiters);
            if (strcmp(token,"\n")==0) continue;

            ++i;
            if(i>=nargs) {
                break;
            }
        }
        while (strcmp(token,"\n")!=0) ;
        va_end(arglist);

//      printf("initializing:");
//        va_start( arglist, nargs);
//        for(i=0;i<nargs;++i){
//          pointer = va_arg(arglist, double **);
//          data = *pointer;
//          printf(" %g",data[line]);
//        }
//        va_end(arglist);
//      printf("\n");

        ++line;
    }
    fclose(infile);

    return lines;
}


/* measure magnitude of provided vector */
double magnitude(double *vector) {

    /* measure the magnitude */
    vector[0] = sqrt(vector[1]*vector[1]+vector[2]*vector[2]+vector[3]*vector[3]);

    return vector[0];
}

/* make provided vector a unit vector */
double unitize(double *vector, double *new_unit_vector) {
    double mag;

    /* measure the magnitude */
    mag = magnitude(vector);

    if(mag != 0.0){
        /* normalize it */
        new_unit_vector[1]=vector[1]/mag;
        new_unit_vector[2]=vector[2]/mag;
        new_unit_vector[3]=vector[3]/mag;
    }
    else
    {
        /* can't normalize, report zero vector */
        new_unit_vector[0] = 0.0;
        new_unit_vector[1] = 0.0;
        new_unit_vector[2] = 0.0;
        new_unit_vector[3] = 0.0;
    }
    return mag;
}

/* scale magnitude of provided vector */
double vector_scale(double *vector, double *new_vector, double scale) {

    new_vector[1] = scale*vector[1];
    new_vector[2] = scale*vector[2];
    new_vector[3] = scale*vector[3];

    return magnitude(new_vector);
}

/* enforce magnitude of provided vector */
double vector_rescale(double *vector, double *new_vector, double new_magnitude) {
    double oldmag;

    oldmag = magnitude(vector);
    if(oldmag <= 0.0) oldmag = 1.0;
    new_vector[1] = new_magnitude/oldmag*vector[1];
    new_vector[2] = new_magnitude/oldmag*vector[2];
    new_vector[3] = new_magnitude/oldmag*vector[3];

    return magnitude(new_vector);
}

/* difference between two given vectors */
double vector_diff(double *vector, double *origin_vector, double *new_vector) {

    new_vector[1] = vector[1]-origin_vector[1];
    new_vector[2] = vector[2]-origin_vector[2];
    new_vector[3] = vector[3]-origin_vector[3];
    return magnitude(new_vector);
}


/* vector cross product where vector magnitude is 0th element */
double *cross_product(double *x, double *y, double *z) {
    z[1] = x[2]*y[3] - x[3]*y[2];
    z[2] = x[3]*y[1] - x[1]*y[3];
    z[3] = x[1]*y[2] - x[2]*y[1];
    z[0] = 0.0;

    return z;
}
/* vector inner product where vector magnitude is 0th element */
double dot_product(double *x, double *y) {
    return x[1]*y[1]+x[2]*y[2]+x[3]*y[3];
}


/* polarization factor */
double polarization_factor(double kahn_factor, double *incident, double *diffracted, double *axis)
{
    double cos2theta,cos2theta_sqr,sin2theta_sqr;
    double psi=0.0;
    double E_in[4];
    double B_in[4];
    double E_out[4];
    double B_out[4];

    unitize(incident,incident);
    unitize(diffracted,diffracted);
    unitize(axis,axis);

    /* component of diffracted unit vector along incident beam unit vector */
    cos2theta = dot_product(incident,diffracted);
    cos2theta_sqr = cos2theta*cos2theta;
    sin2theta_sqr = 1-cos2theta_sqr;

    if(kahn_factor != 0.0){
        /* tricky bit here is deciding which direciton the E-vector lies in for each source
           here we assume it is closest to the "axis" defined above */

        /* cross product to get "vertical" axis that is orthogonal to the cannonical "polarization" */
        cross_product(axis,incident,B_in);
        /* make it a unit vector */
        unitize(B_in,B_in);

        /* cross product with incident beam to get E-vector direction */
        cross_product(incident,B_in,E_in);
        /* make it a unit vector */
        unitize(E_in,E_in);

        /* get components of diffracted ray projected onto the E-B plane */
        E_out[0] = dot_product(diffracted,E_in);
        B_out[0] = dot_product(diffracted,B_in);

        /* compute the angle of the diffracted ray projected onto the incident E-B plane */
        psi = -atan2(B_out[0],E_out[0]);
    }

    /* correction for polarized incident beam */
    return 0.5*(1.0 + cos2theta_sqr - kahn_factor*cos(2*psi)*sin2theta_sqr);
}





/* 2D Gaussian integral=1 */
double ngauss2D(double x, double y, double fwhm)
{
    return log(16.0)/M_PI*fwhm*fwhm*exp(-log(16.0)*((x*x+y*y)/(fwhm*fwhm) ));
}

/* integral of Gaussian fwhm=1 integral=1 */
double ngauss2D_integ(double x, double y)
{
    return 0.125*(erf(2.0*x*sqrt(log(2.0)))*erf(y*sqrt(log(16.0)))*sqrt(log(16.0)/log(2.0)));
}

/* unit volume integrated over a pixel, fwhm = 1 */
double ngauss2D_pixel(double x,double y,double pix)
{
    return ngauss2D_integ(x+pix/2.,y+pix/2.)-ngauss2D_integ(x+pix/2.,y-pix/2.)-ngauss2D_integ(x-pix/2.,y+pix/2.)+ngauss2D_integ(x-pix/2.,y-pix/2.);
}

double integrate_gauss_over_pixel(double x, double y, double fwhm, double pix)
{
    return ngauss2D_pixel(x/fwhm,y/fwhm,pix/fwhm);
}


double fiber2D(double x,double y,double g)
{
    /* g/(2*pi)*(g**2+x**2+y**2)**(-3/2) */
    double temp;
    temp = sqrt(g*g+x*x+y*y);
    if(temp <= 0.0) return 0.0;
    return g/2.0/M_PI/temp/temp/temp;
}
double fiber2D_integ(double x,double y,double g)
{
    return atan((x*y)/(g*sqrt(g*g + x*x + y*y)))/2.0/M_PI;
}
double fiber2D_pixel(double x,double y,double g,double pix)
{
     return fiber2D_integ(x+pix/2.,y+pix/2.,g)-fiber2D_integ(x+pix/2.,y-pix/2.,g)-fiber2D_integ(x-pix/2.,y+pix/2.,g)+fiber2D_integ(x-pix/2.,y-pix/2.,g);
}
double integrate_fiber_over_pixel(double x, double y, double g, double pix)
{
    return fiber2D_pixel(x,y,g,pix);
}



#define SWAP(a,b) temp=(a);(a)=(b);(b)=temp;
double fmedian(unsigned int n, double arr[])
{
    unsigned int i,j,k,l,ir,mid;
    double a,temp;

    l=1;
    ir=n;
    k=(n+1)/2;
//printf("n=%d; k=%d\n",n,k);

//for(i=1;i<=n;++i) printf("arr[%d]=%f\n",i,arr[i]);

    for(;;)
    {
        if(ir <= l+1)
        {
            if(ir == l+1 && arr[ir] < arr[l])
            {
                SWAP(arr[l],arr[ir]);
            }
//for(i=1;i<=n;++i) printf("arr[%d]=%f\n",i,arr[i]);
            return arr[k];
        } else {
            mid=(l+ir) >> 1;
            SWAP(arr[mid],arr[l+1]);
            if(arr[l+1] > arr[ir])
            {
                SWAP(arr[l+1],arr[ir]);
            }
            if(arr[l] > arr[ir])
            {
                SWAP(arr[l],arr[ir]);
            }
            if(arr[l+1] > arr[l])
            {
                SWAP(arr[l+1],arr[l]);
            }
            i=l+1;        // initialize pointers for partitioning
            j=ir;
            a=arr[l];        // partitioning element
            for(;;)        // innermost loop
            {
                do i++; while(arr[i]<a);        // scan up to find element > a
                do j--; while(arr[j]>a);        // scan down to find element < a
                if( j < i ) break;                // pointers crossed, median is in between
                SWAP(arr[i],arr[j]);
            }
            arr[l]=arr[j];                        // insert partitioning element
            arr[j]=a;
            if( j >= k ) ir=j-1;                // Keep partition that contains the median active
            if( j <= k ) l=i;
        }
    }
}


double fmedian_with_rejection(unsigned int n, double arr[],double sigma_cutoff, double *final_mad, int *final_n)
{
    double median_value;
    int i,done;
    double deviate,mad;

    done = 0;
    while(! done)
    {
        /* compute the median (centroid) value */
        median_value = fmedian(n,arr);

        /* now figure out what the mean absolute deviation from this value is */
        mad = fmedian_absolute_deviation(n,arr,median_value);
        //if(flag) printf("mad = %f\n",mad);

        done = 1;
        /* reject all outliers */
        for(i=1;i<=n;++i)
        {
            /* reject positive and negative outliers */
            deviate = fabs(arr[i]-median_value);
            if(deviate > sigma_cutoff*mad)
            {
                /* needs to go */
                /* move value at the end of the array to this "reject" and then shorten the array */
                //if(flag) printf("rejecting arr[%d] = %f (%f)\n",i,arr[i],deviate);
                //arr[worst]+=10000;
                if(i != n)
                {
                    //temp=arr[worst];
                    arr[i] = arr[n];
                    //arr[n]=temp;
                }
                --n;
                done = 0;
            }
        }
    }

    /* basically three return values */
    *final_mad = mad;
    *final_n = n;
    return median_value;
}

/* note: there must be 2*n elements in this array! */
double fmedian_absolute_deviation(unsigned int n, double arr[], double median_value)
{
    int i;
    for(i=1;i<=n;++i)
    {
        arr[i+n] = fabs(arr[i]-median_value);
    }

    return fmedian(n,arr+n);
}




/* this function keeps track of outliers by swapping them to the end of the array */
/* counting starts at 0 and "points" is the number of points */
double fmean_with_rejection(unsigned int starting_points, double arr[], double sigma_cutoff, double *final_rmsd, int *final_n)
{
    int points,i;
    int rejection,worst;
    double temp,sum,avg,sumd,rmsd,deviate,worst_deviate;

    points=starting_points;

    rejection = 1;
    avg = rmsd = NAN;
    while ( rejection && points>starting_points/2.0 )
    {
        /* find the mean and rms deivation */
        sum = sumd = 0.0;
        for(i=0;i<points;++i)
        {
            sum+=arr[i];
        }
        avg=sum/points;
        worst=-1;
        worst_deviate=0.0;
        for(i=0;i<points;++i)
        {
            deviate=fabs(arr[i]-avg);
            if(deviate > worst_deviate)
            {
                worst=i;
                worst_deviate=deviate;
            }
            sumd+=deviate*deviate;
        }
        rmsd=sqrt(sumd/points);

        rejection=0;
        if(worst_deviate>sigma_cutoff*rmsd)
        {
            /* we have a reject! */
            rejection=1;

            /* move it to end of the array and forget about it */
            SWAP(arr[worst],arr[points]);
            --points;
        }
    }

    *final_rmsd = rmsd;
    *final_n = points;
    return avg;
}


}}// namespace simtbx::nanoBragg
