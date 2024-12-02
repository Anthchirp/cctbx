#include <cctbx/boost_python/flex_fwd.h>

#include <boost/python/copy_const_reference.hpp>

#include <boost/python/module.hpp>
#include <boost/python/scope.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/extract.hpp>

#include <simtbx/nanoBragg/nanoBragg.h>

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif


using namespace boost::python;

namespace simtbx {
namespace nanoBragg {

namespace boost_python { namespace {

  boost::python::tuple
  testuple()
  {
    double test = NAN;
    return boost::python::make_tuple(boost::math::isnan(test),2,3,4);
  }

  /* getter/setter functions for convenient python-side members
     here we apply and scaling required to initialize parameters in convenient units
     internal storage is always in meters and radians
     after most initializations we need to update inter-related parameters sensibly,
     such as pixel size and detector size when the number of detector pixels has already been set */

  /* square pixel size, exposed to Python in mm */
  static double get_pixel_size_mm(nanoBragg const& nanoBragg) {return nanoBragg.pixel_size*1000.;}
  static void   set_pixel_size_mm(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.pixel_size = value/1000.;
      /* re-defines detector size */
      nanoBragg.init_detector();
  }

  /* number of pixels along each edge of detector */
  static scitbx::vec2<int> get_detpixels_fastslow(nanoBragg const& nanoBragg) {
      scitbx::vec2<int> value;
      value[0]=nanoBragg.fpixels;
      value[1]=nanoBragg.spixels;
      return value;
  }
  static void   set_detpixels_fastslow(nanoBragg& nanoBragg, scitbx::vec2<int> const& value) {
      nanoBragg.fpixels = value[0];
      nanoBragg.spixels = value[1];
      /* re-defines detector size */
      nanoBragg.detsize_f=nanoBragg.detsize_s=0.0;
      nanoBragg.init_detector();
  }

  /* size of each edge of detector in mm */
  static vec2 get_detsize_fastslow_mm(nanoBragg const& nanoBragg) {
      vec2 value;
      value[0]=nanoBragg.detsize_f*1000.0;
      value[1]=nanoBragg.detsize_s*1000.0;
      return value;
  }
  static void   set_detsize_fastslow_mm(nanoBragg& nanoBragg, vec2 const& value) {
      nanoBragg.detsize_f = value[0]/1000.0;
      nanoBragg.detsize_s = value[1]/1000.0;
      /* update number of pixels */
      nanoBragg.fpixels=nanoBragg.spixels=0;
      /* free the array? */
      nanoBragg.init_detector();
  }

  /* number of sub-pixels in each direction for oversampling */
  static int  get_oversample(nanoBragg const& nanoBragg) {return nanoBragg.oversample;}
  static void set_oversample(nanoBragg& nanoBragg, int const& value) {
      nanoBragg.oversample = value;
      /* don't keep re-setting this! */
      nanoBragg.user_oversample=true;
      nanoBragg.update_oversample();
}


  /* region of interest in pixels */
  static boost::python::tuple get_roi(nanoBragg const& nanoBragg) {
      boost::python::tuple frange;
      boost::python::tuple srange;
      frange = boost::python::make_tuple(nanoBragg.roi_xmin,nanoBragg.roi_xmax);
      srange = boost::python::make_tuple(nanoBragg.roi_ymin,nanoBragg.roi_ymax);

      return boost::python::make_tuple(frange,srange);
  }
  static void   set_roi(nanoBragg& nanoBragg, boost::python::tuple const& tupleoftuples) {
      boost::python::tuple frange;
      boost::python::tuple srange;
      frange = extract<boost::python::tuple>(tupleoftuples[0]);
      srange = extract<boost::python::tuple>(tupleoftuples[1]);
      nanoBragg.roi_xmin=extract<int>(frange[0]);
      nanoBragg.roi_xmax=extract<int>(frange[1]);
      nanoBragg.roi_ymin=extract<int>(srange[0]);
      nanoBragg.roi_ymax=extract<int>(srange[1]);
       /* fix this up so its not out-of-range */
      nanoBragg.init_beamcenter();
  }


  /* debugging option: select pixel to print out to screen */
  static scitbx::vec2<int> get_printout_pixel_fastslow(nanoBragg const& nanoBragg) {
      scitbx::vec2<int> value;
      value[0]=nanoBragg.printout_fpixel;
      value[1]=nanoBragg.printout_spixel;
      return value;
  }
  static void   set_printout_pixel_fastslow(nanoBragg& nanoBragg, scitbx::vec2<int> const& value) {
      nanoBragg.printout_fpixel = value[0];
      nanoBragg.printout_spixel = value[1];
      /* implicitly turn on printing, of course */
      if(value[0]>0) nanoBragg.printout = 1;
  }



  /* beam center in some arbitrary convention (conventions defined below) */
  static vec2 get_beam_center_mm(nanoBragg const& nanoBragg) {
      vec2 value;
      value[0]=nanoBragg.Xbeam*1000.;
      value[1]=nanoBragg.Ybeam*1000.;
      return value;
  }
  static void   set_beam_center_mm(nanoBragg& nanoBragg, vec2 const& value) {
      nanoBragg.Xbeam = value[0]/1000.;
      nanoBragg.Ybeam = value[1]/1000.;
      nanoBragg.Fbeam = value[0]/1000.;
      nanoBragg.Sbeam = value[1]/1000.;
      nanoBragg.Xclose = value[0]/1000.;
      nanoBragg.Yclose = value[1]/1000.;
      nanoBragg.Fclose = value[0]/1000.;
      nanoBragg.Sclose = value[1]/1000.;
      nanoBragg.user_beam = true;
      //nanoBragg.Fbeam = nanoBragg.Sbeam = NAN;
      //nanoBragg.Xclose = nanoBragg.Yclose = NAN;
      //nanoBragg.Fclose = nanoBragg.Sclose = NAN;
      nanoBragg.ORGX = nanoBragg.ORGY = NAN;
      nanoBragg.reconcile_parameters();
  }
  static vec2 get_XDS_ORGXY(nanoBragg const& nanoBragg) {
      vec2 value;
      value[0]=nanoBragg.ORGX;
      value[1]=nanoBragg.ORGY;
      return value;
  }
  static void   set_XDS_ORGXY(nanoBragg& nanoBragg, vec2 const& value) {
      nanoBragg.ORGX = value[0];
      nanoBragg.ORGY = value[1];
      nanoBragg.user_beam = true;
      nanoBragg.beam_convention = XDS;
      nanoBragg.Fbeam = nanoBragg.Sbeam = NAN;
      nanoBragg.Xclose = nanoBragg.Yclose = NAN;
      nanoBragg.Fclose = nanoBragg.Sclose = NAN;
      nanoBragg.Xbeam = nanoBragg.Ybeam = NAN;
      nanoBragg.reconcile_parameters();
  }

  static vec2 get_adxv_beam_center_mm(nanoBragg const& nanoBragg) {
      vec2 value;
      value[0]=nanoBragg.Fbeam*1000.;
      value[1]=(nanoBragg.detsize_s - nanoBragg.Sbeam)*1000.;
      return value;
  }
  static vec2 get_mosflm_beam_center_mm(nanoBragg const& nanoBragg) {
      vec2 value;
      value[0]=(nanoBragg.Sbeam-0.5*nanoBragg.pixel_size)*1000.0;
      value[1]=(nanoBragg.Fbeam-0.5*nanoBragg.pixel_size)*1000.0;
      return value;
  }
  static vec2 get_denzo_beam_center_mm(nanoBragg const& nanoBragg) {
      vec2 value;
      value[0]=(nanoBragg.Sbeam+0.0*nanoBragg.pixel_size)*1000.0;
      value[1]=(nanoBragg.Fbeam+0.0*nanoBragg.pixel_size)*1000;
      return value;
  }
  static vec3 get_dials_origin_mm(nanoBragg const& nanoBragg) {
      vec3 value;
      value[0]=nanoBragg.dials_origin[1];
      value[1]=nanoBragg.dials_origin[2];
      value[2]=nanoBragg.dials_origin[3];
      return value;
  }


  /* number of unit cells along edge in each cell axis direction */
  static scitbx::vec3<double> get_Nabc(nanoBragg const& nanoBragg) {
      scitbx::vec3<double> value;
      value[0]=static_cast<double>(nanoBragg.Na);
      value[1]=static_cast<double>(nanoBragg.Nb);
      value[2]=static_cast<double>(nanoBragg.Nc);
      return value;
  }
  static void   set_Nabc(nanoBragg& nanoBragg, scitbx::vec3<double> const& value) {
      nanoBragg.Na = value[0];
      nanoBragg.Nb = value[1];
      nanoBragg.Nc = value[2];
      /* clear things that might override it */
      nanoBragg.xtal_size_x = -1;
      nanoBragg.xtal_size_y = -1;
      nanoBragg.xtal_size_z = -1;
//      init_interpolator();
      nanoBragg.update_oversample();
  }


  /* specify overall crystal size instead of number of cells */
  static vec3 get_xtal_size_mm(nanoBragg const& nanoBragg) {
      vec3 value;
      value[0]=nanoBragg.xtal_size_x*1000.;
      value[1]=nanoBragg.xtal_size_y*1000.;
      value[2]=nanoBragg.xtal_size_z*1000.;
      return value;
  }
  static void   set_xtal_size_mm(nanoBragg& nanoBragg, vec3 const& value) {
      nanoBragg.xtal_size_x = value[0]/1000.;
      nanoBragg.xtal_size_y = value[1]/1000.;
      nanoBragg.xtal_size_z = value[2]/1000.;
      /* need to update oversampling */
      nanoBragg.update_oversample();
      nanoBragg.init_interpolator();
  }

  /* amorphous material size in 3D */
  static vec3 get_amorphous_sample_size_mm(nanoBragg const& nanoBragg) {
      vec3 value;
      value[0]=nanoBragg.amorphous_sample_x*1000.;
      value[1]=nanoBragg.amorphous_sample_y*1000.;
      value[2]=nanoBragg.amorphous_sample_z*1000.;
      return value;
  }
  static void   set_amorphous_sample_size_mm(nanoBragg& nanoBragg, vec3 const& value) {
      nanoBragg.amorphous_sample_x = value[0]/1000.;
      nanoBragg.amorphous_sample_y = value[1]/1000.;
      nanoBragg.amorphous_sample_z = value[2]/1000.;
      /* need to update amorphous molecules */
      nanoBragg.init_background();
  }

  /* amorphous material thickness along beam (mm)  */
  static double get_amorphous_sample_thick_mm(nanoBragg const& nanoBragg) {return nanoBragg.amorphous_sample_x*1000.;}
  static void   set_amorphous_sample_thick_mm(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.amorphous_sample_x = value/1000.;
      /* need to update amorphous molecules */
      nanoBragg.init_background();
  }
  /* amorphous material molecular weight (Da) */
  static double get_amorphous_molecular_weight_Da(nanoBragg const& nanoBragg) {return nanoBragg.amorphous_molecular_weight;}
  static void   set_amorphous_molecular_weight_Da(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.amorphous_molecular_weight = value;
      /* need to update amorphous molecules */
      nanoBragg.init_background();
  }
  /* amorphous material density (g/cm^3) */
  static double get_amorphous_density_gcm3(nanoBragg const& nanoBragg) {return nanoBragg.amorphous_density/1e6;}
  static void   set_amorphous_density_gcm3(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.amorphous_density = value*1e6;
      /* need to update amorphous molecules */
      nanoBragg.init_background();
  }
  /* now we can compute the scale factor of intensity from amorphous material structure factor */

  /* unit cell dimensions */
  static cctbx::uctbx::unit_cell get_cell_Adeg(nanoBragg const& nanoBragg) {
      scitbx::af::double6 six;
      six[0]=nanoBragg.a[0]*1e10; // convert from internal storage of meters and radians
      six[1]=nanoBragg.b[0]*1e10;
      six[2]=nanoBragg.c[0]*1e10;
      six[3]=nanoBragg.alpha*RTD;
      six[4]=nanoBragg.beta*RTD;
      six[5]=nanoBragg.gamma*RTD;
      cctbx::uctbx::unit_cell cell(six);
      return cell;
  }
  static void   set_cell_Adeg(nanoBragg& nanoBragg, cctbx::uctbx::unit_cell const& value) {
      nanoBragg.a_A[0] = value.parameters()[0];  // initialize Angstrom version, update at the end
      nanoBragg.b_A[0] = value.parameters()[1];
      nanoBragg.c_A[0] = value.parameters()[2];
      nanoBragg.alpha = value.parameters()[3]/RTD;
      nanoBragg.beta  = value.parameters()[4]/RTD;
      nanoBragg.gamma = value.parameters()[5]/RTD;
      nanoBragg.user_cell = 1;
      /* re-generate cell structure and apply any missetting angles to it */
      nanoBragg.init_cell();
//      reconcile_parameters();
  }


  /* unit cell dimensions as a tuple */
  static boost::python::tuple get_cell_astuple(nanoBragg const& nanoBragg) {
      double a,b,c,al,be,ga;
      a=nanoBragg.a[0]*1e10; // convert from internal storage of meters and radians
      b=nanoBragg.b[0]*1e10;
      c=nanoBragg.c[0]*1e10;
      al=nanoBragg.alpha*RTD;
      be=nanoBragg.beta*RTD;
      ga=nanoBragg.gamma*RTD;
      return boost::python::make_tuple(a,b,c,al,be,ga);
  }
  static void   set_cell_astuple(nanoBragg& nanoBragg, boost::python::tuple const& value) {
      nanoBragg.a_A[0] = extract<double>(value[0]);  // initialize Angstrom version, update at the end
      nanoBragg.b_A[0] = extract<double>(value[1]);
      nanoBragg.c_A[0] = extract<double>(value[2]);;
      nanoBragg.alpha  = extract<double>(value[3])/RTD;
      nanoBragg.beta   = extract<double>(value[4])/RTD;
      nanoBragg.gamma  = extract<double>(value[5])/RTD;
      nanoBragg.user_cell = 1;
      /* re-generate cell structure and apply any missetting angles to it */
      nanoBragg.init_cell();
//      reconcile_parameters();
  }


  /* use a MOSFLM-style A=UB matrix */
  static mat3 get_Amatrix(nanoBragg const& nanoBragg) {
      mat3 Amatrix;
      Amatrix(0,0)=nanoBragg.a_star[1]; // internal storage is reciprocal Angstrom
      Amatrix(0,1)=nanoBragg.a_star[2];
      Amatrix(0,2)=nanoBragg.a_star[3];
      Amatrix(1,0)=nanoBragg.b_star[1];
      Amatrix(1,1)=nanoBragg.b_star[2];
      Amatrix(1,2)=nanoBragg.b_star[3];
      Amatrix(2,0)=nanoBragg.c_star[1];
      Amatrix(2,1)=nanoBragg.c_star[2];
      Amatrix(2,2)=nanoBragg.c_star[3];
      return Amatrix;
  }
  static void   set_Amatrix(nanoBragg& nanoBragg, mat3 const& value) {
      nanoBragg.a_star[1] = value(0,0);  // initialize Angstrom version, update at the end
      nanoBragg.a_star[2] = value(0,1);
      nanoBragg.a_star[3] = value(0,2);
      nanoBragg.b_star[1] = value(1,0);
      nanoBragg.b_star[2] = value(1,1);
      nanoBragg.b_star[3] = value(1,2);
      nanoBragg.c_star[1] = value(2,0);
      nanoBragg.c_star[2] = value(2,1);
      nanoBragg.c_star[3] = value(2,2);
      nanoBragg.user_cell = 0;
      nanoBragg.user_matrix = 1;
      /* re-generate cell structure and apply any missetting angles to it */
      nanoBragg.init_cell();
//      reconcile_parameters();
  }
  static void   set_Amatrix_NKS_implementation(nanoBragg& nanoBragg, mat3 const& value) {
      // Input is direct space A matrix in angstroms.
      nanoBragg.user_cell = 1;
      // Assume input is A=UB.  No additional missetting angles accepted.
      nanoBragg.a_A[1] = value[0];
      nanoBragg.a_A[2] = value[1];
      nanoBragg.a_A[3] = value[2];
      nanoBragg.b_A[1] = value[3];
      nanoBragg.b_A[2] = value[4];
      nanoBragg.b_A[3] = value[5];
      nanoBragg.c_A[1] = value[6];
      nanoBragg.c_A[2] = value[7];
      nanoBragg.c_A[3] = value[8];
      /* now convert Angstrom to meters */
      magnitude(nanoBragg.a_A);
      magnitude(nanoBragg.b_A);
      magnitude(nanoBragg.c_A);
      //SCITBX_EXAMINE(nanoBragg.a_A[0]);
      //SCITBX_EXAMINE(nanoBragg.b_A[0]);
      //SCITBX_EXAMINE(nanoBragg.c_A[0]);
      vector_scale(nanoBragg.a_A,nanoBragg.a,1e-10);
      vector_scale(nanoBragg.b_A,nanoBragg.b,1e-10);
      vector_scale(nanoBragg.c_A,nanoBragg.c,1e-10);
      cctbx::uctbx::unit_cell cell(value.transpose());
      nanoBragg.alpha = cell.parameters()[3]/RTD;
      nanoBragg.beta = cell.parameters()[4]/RTD;
      nanoBragg.gamma = cell.parameters()[5]/RTD;
      //SCITBX_EXAMINE(cell.parameters()[3]);
      //SCITBX_EXAMINE(cell.parameters()[4]);
      //SCITBX_EXAMINE(cell.parameters()[5]);

      /* define phi=0 mosaic=0 crystal orientation */
      vector_scale(nanoBragg.a,nanoBragg.a0,1.0);
      vector_scale(nanoBragg.b,nanoBragg.b0,1.0);
      vector_scale(nanoBragg.c,nanoBragg.c0,1.0);
      /* define phi=0 crystal orientation */
      vector_scale(nanoBragg.a,nanoBragg.ap,1.0);
      vector_scale(nanoBragg.b,nanoBragg.bp,1.0);
      vector_scale(nanoBragg.c,nanoBragg.cp,1.0);
      /* Now set the reciprocal cell */
      mat3 invAmat = value.inverse();
      nanoBragg.a_star[1] = invAmat[0];
      nanoBragg.a_star[2] = invAmat[3];
      nanoBragg.a_star[3] = invAmat[6];
      magnitude(nanoBragg.a_star);
      nanoBragg.b_star[1] = invAmat[1];
      nanoBragg.b_star[2] = invAmat[4];
      nanoBragg.b_star[3] = invAmat[7];
      magnitude(nanoBragg.b_star);
      nanoBragg.c_star[1] = invAmat[2];
      nanoBragg.c_star[2] = invAmat[5];
      nanoBragg.c_star[3] = invAmat[8];
      magnitude(nanoBragg.c_star);
  }

  /* crystal misseting angles, will be added after any matrix */
  static vec3 get_misset_deg(nanoBragg const& nanoBragg) {
      vec3 value;
      value[0]=nanoBragg.misset[1]*RTD;
      value[1]=nanoBragg.misset[2]*RTD;
      value[2]=nanoBragg.misset[3]*RTD;
      return value;
  }
  static void   set_misset_deg(nanoBragg& nanoBragg, vec3 const& value) {
      nanoBragg.misset[0] = 1;
      nanoBragg.misset[1] = value[0]/RTD;
      nanoBragg.misset[2] = value[1]/RTD;
      nanoBragg.misset[3] = value[2]/RTD;
      /* need to update A matrix */
      nanoBragg.init_cell();
  }


  /* detector misseting angles, will be applied */
  static vec3 get_detector_rotation_deg(nanoBragg const& nanoBragg) {
      vec3 value;
      value[0]=nanoBragg.detector_rotx*RTD;
      value[1]=nanoBragg.detector_roty*RTD;
      value[2]=nanoBragg.detector_rotz*RTD;
      return value;
  }
  static void   set_detector_rotation_deg(nanoBragg& nanoBragg, vec3 const& value) {
      nanoBragg.detector_rotx = value[0]/RTD;
      nanoBragg.detector_roty = value[1]/RTD;
      nanoBragg.detector_rotz = value[2]/RTD;
      /* need to update detector stuff */
      nanoBragg.init_detector();
  }

  /* twotheta is a special extra missetting angle, applied in addition to above, but pivoting about sample instead of beam spot */
  static double get_detector_twotheta_deg(nanoBragg const& nanoBragg) {
      return nanoBragg.detector_twotheta*RTD;;
  }
  static void   set_detector_twotheta_deg(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.detector_twotheta = value/RTD;
      nanoBragg.detector_pivot = BEAM;
      /* need to update detector stuff */
      nanoBragg.init_detector();
  }

  /* sample-to-direct-beam-spot distance, exposed to Python in mm */
  static double get_distance_mm(nanoBragg const& nanoBragg) {return nanoBragg.distance*1000.;}
  static void   set_distance_mm(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.distance = value/1000.;
      nanoBragg.close_distance = value/1000.;
      nanoBragg.user_distance = true;
      nanoBragg.reconcile_parameters();
  }
  /* sample-to-nearest-point-on-detector distance, exposed to Python in mm */
  static double get_close_distance_mm(nanoBragg const& nanoBragg) {return nanoBragg.close_distance*1000.;}
  static void   set_close_distance_mm(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.close_distance = value/1000.;
      nanoBragg.distance = value/1000.;
      nanoBragg.user_distance = true;
      nanoBragg.reconcile_parameters();
  }


  /* re-set convention used to define diffractometer coordinates */
  static convention get_beam_convention(nanoBragg const& nanoBragg) {
      convention value;
      value=nanoBragg.beam_convention;
      return value;
  }
  static void   set_beam_convention(nanoBragg& nanoBragg, convention const& value) {
      nanoBragg.beam_convention = value;
      /* need to update everything? */
      nanoBragg.init_beamcenter();
  }


  /* unit vectors defining camera coordinate system */

  /* first, the X-ray beam, default is (1,0,0) */
  static vec3 get_beam_vector(nanoBragg const& nanoBragg) {
      vec3 value;
      value[0]=nanoBragg.beam_vector[1];
      value[1]=nanoBragg.beam_vector[2];
      value[2]=nanoBragg.beam_vector[3];
      return value;
  }
  static void   set_beam_vector(nanoBragg& nanoBragg, vec3 const& value) {
      nanoBragg.beam_vector[0] = 0;
      nanoBragg.beam_vector[1] = value[0];
      nanoBragg.beam_vector[2] = value[1];
      nanoBragg.beam_vector[3] = value[2];
      /* need to update everything? */
      nanoBragg.init_detector();
  }

  /* the fast direction of the detector pixel array, default is (0,0,1) */
  static vec3 get_fdet_vector(nanoBragg const& nanoBragg) {
      vec3 value;
      value[0]=nanoBragg.fdet_vector[1];
      value[1]=nanoBragg.fdet_vector[2];
      value[2]=nanoBragg.fdet_vector[3];
      return value;
  }
  static void   set_fdet_vector(nanoBragg& nanoBragg, vec3 const& value) {
      nanoBragg.fdet_vector[0] = 0;
      nanoBragg.fdet_vector[1] = value[0];
      nanoBragg.fdet_vector[2] = value[1];
      nanoBragg.fdet_vector[3] = value[2];
      /* need to update everything? */
      nanoBragg.init_detector();
  }

  /* the slow direction of the detector pixel array, default is (0,-1,0) */
  static vec3 get_sdet_vector(nanoBragg const& nanoBragg) {
      vec3 value;
      value[0]=nanoBragg.sdet_vector[1];
      value[1]=nanoBragg.sdet_vector[2];
      value[2]=nanoBragg.sdet_vector[3];
      return value;
  }
  static void   set_sdet_vector(nanoBragg& nanoBragg, vec3 const& value) {
      nanoBragg.sdet_vector[0] = 0;
      nanoBragg.sdet_vector[1] = value[0];
      nanoBragg.sdet_vector[2] = value[1];
      nanoBragg.sdet_vector[3] = value[2];
      /* need to update everything? */
      nanoBragg.init_detector();
  }

  /* the orthogonal vector to the detector pixel array, default is (1,0,0), same as the beam */
  /* note that this is updated every time using sdet_vector and fdet_vector bu init_detector() */
  static vec3 get_odet_vector(nanoBragg const& nanoBragg) {
      vec3 value;
      value[0]=nanoBragg.odet_vector[1];
      value[1]=nanoBragg.odet_vector[2];
      value[2]=nanoBragg.odet_vector[3];
      return value;
  }
  static void   set_odet_vector(nanoBragg& nanoBragg, vec3 const& value) {
      nanoBragg.odet_vector[0] = 0;
      nanoBragg.odet_vector[1] = value[0];
      nanoBragg.odet_vector[2] = value[1];
      nanoBragg.odet_vector[3] = value[2];
      /* need to update everything? */
      nanoBragg.init_detector();
  }
  static vec3 get_pix0_vector(nanoBragg const& nanoBragg) {
      vec3 value;
      value[0]=1000*nanoBragg.pix0_vector[1];
      value[1]=1000*nanoBragg.pix0_vector[2];
      value[2]=1000*nanoBragg.pix0_vector[3];
      return value;
  }
  /* the direction vector of detector_twotheta rotation axis (about the sample), default is (0,1,0) */
  static vec3 get_twotheta_axis(nanoBragg const& nanoBragg) {
      vec3 value;
      value[0]=nanoBragg.twotheta_axis[1];
      value[1]=nanoBragg.twotheta_axis[2];
      value[2]=nanoBragg.twotheta_axis[3];
      return value;
  }
  static void   set_twotheta_axis(nanoBragg& nanoBragg, vec3 const& value) {
      nanoBragg.twotheta_axis[0] = 0;
      nanoBragg.twotheta_axis[1] = value[0];
      nanoBragg.twotheta_axis[2] = value[1];
      nanoBragg.twotheta_axis[3] = value[2];
      /* need to update everything? */
      nanoBragg.init_detector();
  }

  /* direction of the E vector of the X-ray beam. will be made normal to beam direction, default: (0,0,1) */
  static vec3 get_polar_vector(nanoBragg const& nanoBragg) {
      vec3 value;
      value[0]=nanoBragg.polar_vector[1];
      value[1]=nanoBragg.polar_vector[2];
      value[2]=nanoBragg.polar_vector[3];
      return value;
  }
  static void   set_polar_vector(nanoBragg& nanoBragg, vec3 const& value) {
      nanoBragg.polar_vector[0] = 0;
      nanoBragg.polar_vector[1] = value[0];
      nanoBragg.polar_vector[2] = value[1];
      nanoBragg.polar_vector[3] = value[2];
      /* need to update everything? */
      nanoBragg.init_detector();
  }

  /* direction of the B vector of the X-ray beam. will be made normal to beam direction, default: (0,-1,0) */
  static vec3 get_polar_Bvector(nanoBragg const& nanoBragg) {
      vec3 Evector = vec3(nanoBragg.polar_vector[1],nanoBragg.polar_vector[2],nanoBragg.polar_vector[3]);
      vec3 Pvector = vec3(nanoBragg.beam_vector[1],nanoBragg.beam_vector[2],nanoBragg.beam_vector[3]);
      vec3 Bvector = Pvector.cross(Evector).normalize();
      return Bvector;
  }
  static void   set_polar_Bvector(nanoBragg& nanoBragg, vec3 const& value) {
      vec3 Bvector = value;
      vec3 Pvector = vec3(nanoBragg.beam_vector[1],nanoBragg.beam_vector[2],nanoBragg.beam_vector[3]);
      vec3 Evector = Bvector.cross(Pvector).normalize();
      nanoBragg.polar_vector[0] = 0;
      nanoBragg.polar_vector[1] = Evector[0];
      nanoBragg.polar_vector[2] = Evector[1];
      nanoBragg.polar_vector[3] = Evector[2];
      /* need to update everything? */
      nanoBragg.init_detector();
  }

  /* the direction vector of the phi spindle rotation axis, default is (0,0,1) */
  static vec3 get_spindle_vector(nanoBragg const& nanoBragg) {
      vec3 value;
      value[0]=nanoBragg.spindle_vector[1];
      value[1]=nanoBragg.spindle_vector[2];
      value[2]=nanoBragg.spindle_vector[3];
      return value;
  }
  static void   set_spindle_vector(nanoBragg& nanoBragg, vec3 const& value) {
      nanoBragg.spindle_vector[0] = 0;
      nanoBragg.spindle_vector[1] = value[0];
      nanoBragg.spindle_vector[2] = value[1];
      nanoBragg.spindle_vector[3] = value[2];
      /* need to update everything? */
      nanoBragg.init_detector();
  }




  /* hkl, without F */
  static indices get_indices(nanoBragg nanoBragg) {
      int i,h,k,l;
      /* initialize python from built-in ***Fhkl array */
      nanoBragg.pythony_indices = indices();
      int hkls = nanoBragg.h_range*nanoBragg.k_range*nanoBragg.l_range;
      nanoBragg.pythony_indices.resize(hkls);
      i=0;
      for(h=nanoBragg.h_min;h<=nanoBragg.h_max;++h){
          for(k=nanoBragg.k_min;k<=nanoBragg.k_max;++k){
              for(l=nanoBragg.l_min;l<=nanoBragg.l_max;++l){

                  if ( (h<=nanoBragg.h_max) && (h>=nanoBragg.h_min) && (k<=nanoBragg.k_max)
                    && (k>=nanoBragg.k_min) && (l<=nanoBragg.l_max) && (l>=nanoBragg.l_min)  ) {
                      /* populate only valid variables */
                      miller_t hkl (h,k,l);
                      nanoBragg.pythony_indices[i] = hkl;
                      ++i;
                  }
              }
          }
      }
      return nanoBragg.pythony_indices;
  }
  static void   set_indices(nanoBragg& nanoBragg, indices const& value) {
      if(nanoBragg.verbose>3) printf(" about to initialize indices\n");
      nanoBragg.pythony_indices = value;
      if(nanoBragg.verbose>3) printf(" done\n");

      /* re-initialize */
      nanoBragg.init_Fhkl();
  }


  /* F without hkl : translate F values, hope number is the same as miller indices */
  static af::shared<double> get_amplitudes(nanoBragg nanoBragg) {
      int i,h,k,l;
      nanoBragg.pythony_amplitudes = af::shared<double>();
      int hkls = nanoBragg.h_range*nanoBragg.k_range*nanoBragg.l_range;
      nanoBragg.pythony_amplitudes.resize(hkls);
      i=0;
      for(h=nanoBragg.h_min;h<=nanoBragg.h_max;++h){
          for(k=nanoBragg.k_min;k<=nanoBragg.k_max;++k){
              for(l=nanoBragg.l_min;l<=nanoBragg.l_max;++l){

                  if ( (h<=nanoBragg.h_max) && (h>=nanoBragg.h_min) && (k<=nanoBragg.k_max)
                    && (k>=nanoBragg.k_min) && (l<=nanoBragg.l_max) && (l>=nanoBragg.l_min)  ) {
                      /* populate only valid values */
                      nanoBragg.pythony_amplitudes[i] = nanoBragg.Fhkl[h-nanoBragg.h_min][k-nanoBragg.k_min][l-nanoBragg.l_min];
                      ++i;
                  }
              }
          }
      }
      return nanoBragg.pythony_amplitudes;
  }
  static void   set_amplitudes(nanoBragg& nanoBragg, af::shared<double> const& value) {
      if(nanoBragg.verbose>3) printf(" about to initialize amplitudes\n");
      nanoBragg.pythony_amplitudes = value;
      if(nanoBragg.verbose>3) printf(" done\n");

      /* re-initialize ***Fhkl array */
      nanoBragg.init_Fhkl();
  }


  /* F and hkl as a tuple: translate F values, hope number is the same as miller indices */
  static boost::python::tuple get_Fhkl_tuple(nanoBragg nanoBragg) {
      int h,k,l;
      double temp;
      int hkls = nanoBragg.h_range*nanoBragg.k_range*nanoBragg.l_range;
      if(nanoBragg.verbose>3) printf(" expecting %d hkls\n",hkls);
//      nanoBragg.pythony_indices = indices();
//      nanoBragg.pythony_amplitudes = af::shared<double>();
      nanoBragg.pythony_indices = indices(hkls,af::init_functor_null<scitbx::vec3<int> >());
      nanoBragg.pythony_amplitudes = af::shared<double>(hkls,af::init_functor_null<double>());
//      nanoBragg.pythony_indices = indices(hkls);
//      nanoBragg.pythony_amplitudes = af::shared<double>(hkls);
//      nanoBragg.pythony_indices.resize(hkls);
//      nanoBragg.pythony_amplitudes.resize(hkls);
      int i=0;

      for(h=nanoBragg.h_min;h<=nanoBragg.h_max;++h){
          for(k=nanoBragg.k_min;k<=nanoBragg.k_max;++k){
              for(l=nanoBragg.l_min;l<=nanoBragg.l_max;++l){

                  if ( (h<=nanoBragg.h_max) && (h>=nanoBragg.h_min) && (k<=nanoBragg.k_max)
                    && (k>=nanoBragg.k_min) && (l<=nanoBragg.l_max) && (l>=nanoBragg.l_min)  ) {
                      /* populate all stored values */
                      miller_t hkl (h,k,l);
                      if(nanoBragg.verbose>9) printf(" about to access indices[%d] at %p\n",i,&nanoBragg.pythony_indices[i]);
                      nanoBragg.pythony_indices[i] = hkl;
//                      nanoBragg.pythony_indices.push_back(hkl);
                      if(nanoBragg.verbose>9) printf(" about to access (%d,%d,%d) Fhkl[%d][%d][%d]\n",
                                           h,k,l,h-nanoBragg.h_min,k-nanoBragg.k_min,l-nanoBragg.l_min);
                      temp = nanoBragg.Fhkl[h-nanoBragg.h_min][k-nanoBragg.k_min][l-nanoBragg.l_min];
                      if(nanoBragg.verbose>9) printf(" about to access amplitudes[%d] at %p\n",
                                            i,&nanoBragg.pythony_amplitudes[i]);
                      nanoBragg.pythony_amplitudes[i] = temp;
//                      nanoBragg.pythony_amplitudes.push_back(temp);
                      if(nanoBragg.verbose>9) printf(" done\n");
                      ++i;
                  }
              }
          }
      }
      /* return a tuple so there is no confusion about order of initialization */
      if(nanoBragg.verbose>3) SCITBX_EXAMINE(nanoBragg.pythony_indices.size());
      return boost::python::make_tuple(nanoBragg.pythony_indices,nanoBragg.pythony_amplitudes);
  }
  static void   set_Fhkl_tuple(nanoBragg& nanoBragg, boost::python::tuple const& value) {
//      nanoBragg.pythony_indices = boost::python::tuple::get<0>(value);
//      nanoBragg.pythony_amplitudes = boost::python::tuple::get<1>(value);
//      boost::python::extract( nanoBragg.pythony_indices , nanoBragg.pythony_amplitudes ) = value;
      if(nanoBragg.verbose>3) printf(" about to initialize indices\n");
      nanoBragg.pythony_indices = extract<indices >(value[0]);
      if(nanoBragg.verbose>3) printf(" about to initialize amplitudes\n");
      nanoBragg.pythony_amplitudes = extract<af::shared<double> >(value[1]);

      /* re-initialize ***Fhkl array */
      nanoBragg.init_Fhkl();
  }

  /* allow default value for missing Fs */
  static double get_default_F(nanoBragg const& nanoBragg) {return nanoBragg.default_F;}
  static void   set_default_F(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.default_F = value;
      /* make sure this gets applied! */
      nanoBragg.init_Fhkl();
  }

  /* allow direct access to F000, mostly for marking beam center */
  static double get_F000(nanoBragg const& nanoBragg) {
      return nanoBragg.Fhkl[-nanoBragg.h_min][-nanoBragg.k_min][-nanoBragg.l_min];
  }
  static void   set_F000(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.F000 = value;
      nanoBragg.Fhkl[-nanoBragg.h_min][-nanoBragg.k_min][-nanoBragg.l_min] = nanoBragg.F000;
  }


  /* background-scattering structure factor Fbg vs sin(theta)/lambda (stol) : populate C++ interpolation arrays */
  static af::shared<vec2> get_Fbg_vs_stol(nanoBragg nanoBragg) {
      int i;
      /* create a new flex array */
      //nanoBragg.pythony_stolFbg = af::shared<vec2>();
      /* make sure it is big enough to hold all stol,Fbg pairs */
      nanoBragg.pythony_stolFbg.resize(nanoBragg.stols);
      /* copy the non-edge values into the flex array (interpolation buffer of 2 values on either end) */
printf("DEBUG: pythony_stolFbg[1]=(%g,%g)\n",nanoBragg.pythony_stolFbg[1][0],nanoBragg.pythony_stolFbg[1][1]);
      if(nanoBragg.verbose>3) printf(" about to initialize pythony_stolFbg\n");
      for(i=0;i<=nanoBragg.stols-4;++i){
          nanoBragg.pythony_stolFbg[i] = vec2(nanoBragg.stol_of[i+2]/nanoBragg.stol_file_mult,nanoBragg.Fbg_of[i+2]);
      }
printf("DEBUG: pythony_stolFbg[1]=(%g,%g)\n",nanoBragg.pythony_stolFbg[1][0],nanoBragg.pythony_stolFbg[1][1]);
      if(nanoBragg.verbose>3) printf("  done\n");
      return nanoBragg.pythony_stolFbg;
  }
  static void   set_Fbg_vs_stol(nanoBragg& nanoBragg, af::shared<vec2> const& value) {
      if(nanoBragg.verbose>3) printf(" about to initialize stol,Fbg\n");
      nanoBragg.pythony_stolFbg = value;
      if(nanoBragg.verbose>3) printf(" done\n");

      /* re-initialize stol and Fbg arrays */
      nanoBragg.init_background();
  }

  /* allow default value for background structure factor Fbg */
  static double get_default_Fbg(nanoBragg const& nanoBragg) {return nanoBragg.default_Fbg;}
  static void   set_default_Fbg(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.default_Fbg = value;
      /* make sure this gets applied! */
      nanoBragg.init_background();
  }


  /* table of sources, as dxtbx "beam"s */
  static scitbx::af::versa<dxtbx::model::Beam, scitbx::af::flex_grid<> > get_beams(nanoBragg& nanoBragg) {
      int i;
      /* allocate new flex array */
//      scitbx::af::versa<dxtbx::model::Beam, scitbx::af::flex_grid<> > nanoBragg_pythony_beams;
      nanoBragg.pythony_beams = scitbx::af::versa<dxtbx::model::Beam, scitbx::af::flex_grid<> >();
      /* make sure it is big enough to hold all sources */
      nanoBragg.pythony_beams.resize(nanoBragg.sources);

      /* polarization normal seems to be B vector */
      vec3 Evector = vec3(nanoBragg.polar_vector[1],nanoBragg.polar_vector[2],nanoBragg.polar_vector[3]);
      vec3 Pvector = vec3(nanoBragg.beam_vector[1],nanoBragg.beam_vector[2],nanoBragg.beam_vector[3]);
      vec3 Bvector = Pvector.cross(Evector).normalize();

      /* copy internal storage into the flex array */
      for(i=0;i<nanoBragg.sources;++i){
          nanoBragg.pythony_beams[i].set_direction(vec3(nanoBragg.source_X[i],nanoBragg.source_Y[i],nanoBragg.source_Z[i]));
          nanoBragg.pythony_beams[i].set_wavelength(nanoBragg.source_lambda[i]*1e10);
          nanoBragg.pythony_beams[i].set_flux(nanoBragg.source_I[i]);
          // how is this a fraction when it can be negative? (Kahn et al. 1982)
          nanoBragg.pythony_beams[i].set_polarization_fraction(nanoBragg.polarization);
          nanoBragg.pythony_beams[i].set_polarization_normal(Bvector);
      }
      /* pass this back to python */
      return nanoBragg.pythony_beams;
  }
  static void   set_beams(nanoBragg& nanoBragg, scitbx::af::versa<dxtbx::model::Beam, scitbx::af::flex_grid<> > const& value) {
      if(nanoBragg.verbose>3) printf(" about to initialize sources\n");
      nanoBragg.pythony_beams = value;
      if(nanoBragg.verbose>3) printf(" done\n");

      /* re-initialize source table from pythony array */
      nanoBragg.init_sources();
  }



  /* table of sources : position in space  */
  static af::shared<vec3> get_source_XYZ(nanoBragg nanoBragg) {
      int i;
      /* create new flex arrays */
      nanoBragg.pythony_source_XYZ = af::shared<vec3>();
      //nanoBragg.pythony_source_intensity = af::shared<double>();
      //nanoBragg.pythony_source_lambda = af::shared<double>();
      /* make sure it is big enough to hold all sources, and that they all match */
      nanoBragg.pythony_source_XYZ.resize(nanoBragg.sources);
      nanoBragg.pythony_source_intensity.resize(nanoBragg.sources);
      nanoBragg.pythony_source_lambda.resize(nanoBragg.sources);
      /* copy internal storage into the flex array */
      for(i=0;i<nanoBragg.sources;++i){
          nanoBragg.pythony_source_XYZ[i]       = vec3(nanoBragg.source_X[i],nanoBragg.source_Y[i],nanoBragg.source_Z[i]);
          nanoBragg.pythony_source_intensity[i] = nanoBragg.source_I[i];
          nanoBragg.pythony_source_lambda[i]    = nanoBragg.source_lambda[i]*1e10;
      }
      return nanoBragg.pythony_source_XYZ;
  }
  static void   set_source_XYZ(nanoBragg& nanoBragg, af::shared<vec3> const& value) {
      if(nanoBragg.verbose>3) printf(" about to initialize sources\n");
      nanoBragg.pythony_source_XYZ = value;
      if(nanoBragg.verbose>3) printf(" done\n");

      /* re-initialize source table from pythony array */
      nanoBragg.init_sources();
  }


  /* table of sources : wavelength  */
  static af::shared<double> get_source_lambda(nanoBragg nanoBragg) {
      int i;
      /* create new flex arrays */
      nanoBragg.pythony_source_lambda = af::shared<double>();
      /* make sure it is big enough to hold all sources, and that they all match */
      nanoBragg.pythony_source_XYZ.resize(nanoBragg.sources);
      nanoBragg.pythony_source_intensity.resize(nanoBragg.sources);
      nanoBragg.pythony_source_lambda.resize(nanoBragg.sources);
      /* copy internal storage into the flex array */
      for(i=0;i<nanoBragg.sources;++i){
          nanoBragg.pythony_source_XYZ[i]       = vec3(nanoBragg.source_X[i],nanoBragg.source_Y[i],nanoBragg.source_Z[i]);
          nanoBragg.pythony_source_intensity[i] = nanoBragg.source_I[i];
          nanoBragg.pythony_source_lambda[i]    = nanoBragg.source_lambda[i]*1e10;
      }
      return nanoBragg.pythony_source_lambda;
  }
  static void   set_source_lambda(nanoBragg& nanoBragg, af::shared<double> const& value) {
      if(nanoBragg.verbose>3) printf(" about to initialize sources\n");
      nanoBragg.pythony_source_lambda = value;
      if(nanoBragg.verbose>3) printf(" done\n");

      /* re-initialize source table from pythony array */
      nanoBragg.init_sources();
  }





  /* X-ray wavelength */
  static double get_lambda_A(nanoBragg const& nanoBragg) {return nanoBragg.lambda0*1e10;}
  static void   set_lambda_A(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.lambda0 = value/1e10;

      /* override any other provided lambda values, so they don't override us */
      int n = nanoBragg.pythony_source_lambda.size();
      if(n > 0 ) {
          printf("WARNING: resetting %d source wavelengths to %f A\n",n,value);
          for(int i=0;i<n;++i){
            nanoBragg.pythony_source_lambda[i] = value;
          }
      }
      /* re-initialize source table */
      nanoBragg.init_sources();
  }
  /* X-ray wavelength from energy */
  static double get_energy_eV(nanoBragg const& nanoBragg) {return 12398.42/(nanoBragg.lambda0*1e10);}
  static void   set_energy_eV(nanoBragg& nanoBragg, double const& value) {nanoBragg.lambda0 = (12398.42/value)/1e10;}

  /* X-ray fluence in photons/meter^2 */
  static double get_fluence(nanoBragg const& nanoBragg) {
      return nanoBragg.fluence;
  }
  static void   set_fluence(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.fluence = value;
      /* force update of the flux... */
      nanoBragg.flux = 0.0;
      nanoBragg.init_beam();
  }

  /* X-ray flux in photons/s */
  static double get_flux(nanoBragg const& nanoBragg) {
      return nanoBragg.flux;
  }
  static void   set_flux(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.flux = value;
      /* re-defines fluence */
      nanoBragg.init_beam();
  }

  /* X-ray exposure time, mainly defines fluence */
  static double get_exposure_s(nanoBragg const& nanoBragg) {
      return nanoBragg.exposure;
  }
  static void   set_exposure_s(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.exposure = value;
      /* re-defines fluence */
      nanoBragg.init_beam();
  }

  /* X-ray beam size, mainly defines fluence */
  static double get_beamsize_mm(nanoBragg const& nanoBragg) {
      return nanoBragg.beamsize*1000.0;
  }
  static void   set_beamsize_mm(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.beamsize = value/1000.0;
      /* re-defines fluence */
      nanoBragg.init_beam();
  }


  /* x-ray beam spectral dispersion in % */
  static double get_dispersion_pct(nanoBragg const& nanoBragg) {
      return nanoBragg.dispersion*100.0;
  }
  static void   set_dispersion_pct(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.dispersion = value/100.0;
      /* always force recompute of step size? */
      if(nanoBragg.dispsteps>0) nanoBragg.dispstep = -1.0;
      /* need to re-create source table */
      nanoBragg.init_steps();
      nanoBragg.init_sources();
  }

  /* number of discrete wavelengths in X-ray source, default: 1 or 2 ; you will want more */
  static int get_dispsteps(nanoBragg const& nanoBragg) {
      return nanoBragg.dispsteps;
  }
  static void   set_dispsteps(nanoBragg& nanoBragg, int const& value) {
      nanoBragg.dispsteps = value;
      /* always force recompute of step size? */
      nanoBragg.dispstep = -1;
//      nanoBragg.dispersion=-1.0;
      /* need to re-create source table */
      nanoBragg.init_steps();
      nanoBragg.init_sources();
  }




  /* x-ray beam horizontal,vertical divergence in mrad */
  static vec2 get_divergence_hv_mrad(nanoBragg const& nanoBragg) {
      vec2 value;
      value[0]=nanoBragg.hdivrange*1000.0;
      value[1]=nanoBragg.vdivrange*1000.0;
      return value;
  }
  static void   set_divergence_hv_mrad(nanoBragg& nanoBragg, vec2 const& value) {
      nanoBragg.hdivrange = value[0]/1000.0;
      nanoBragg.vdivrange = value[1]/1000.0;
      /* re-determine stepsize */
      nanoBragg.hdivstep=nanoBragg.vdivstep=-1.0;
      /* need to re-create source table */
      nanoBragg.init_steps();
      nanoBragg.init_sources();
  }

  /* number of discrete incidence angles in X-ray source, default: 1 or 2 ; you will want more */
  static scitbx::vec2<int> get_divsteps_hv(nanoBragg const& nanoBragg) {
      scitbx::vec2<int> value;
      value[0]=nanoBragg.hdivsteps;
      value[1]=nanoBragg.vdivsteps;
      return value;
  }
  static void   set_divsteps_hv(nanoBragg& nanoBragg, scitbx::vec2<int> const& value) {
      nanoBragg.hdivsteps = value[0];
      nanoBragg.vdivsteps = value[1];
      /* need to re-create source table */
      nanoBragg.vdivstep = nanoBragg.hdivstep = -1.0;
      nanoBragg.init_steps();
      nanoBragg.init_sources();
  }

  /* alternative to steps: x-ray beam divergence step size, horizontal,vertical in mrad */
  static vec2 get_divstep_hv_mrad(nanoBragg const& nanoBragg) {
      vec2 value;
      value[0]=nanoBragg.hdivstep*1000.0;
      value[1]=nanoBragg.vdivstep*1000.0;
      return value;
  }
  static void   set_divstep_hv_mrad(nanoBragg& nanoBragg, vec2 const& value) {
      nanoBragg.hdivstep = value[0]/1000.0;
      nanoBragg.vdivstep = value[1]/1000.0;
      /* need to re-create source table */
      nanoBragg.vdivsteps = nanoBragg.hdivsteps = -1;
      nanoBragg.init_steps();
      nanoBragg.init_sources();
  }





  /* crystal mosaic spread, in deg */
  static double get_mosaic_spread_deg(nanoBragg const& nanoBragg) {
      return nanoBragg.mosaic_spread*RTD;;
  }
  static void   set_mosaic_spread_deg(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.mosaic_spread = value/RTD;
      /* need to re-create mosaic domain table */
      nanoBragg.init_mosaicity();
      nanoBragg.init_steps();
      //nanoBragg.show_mosaic_blocks();
  }
  /* number of discrete mosaic domains, default: 1 or 2 ; you will want more */
  static int get_mosaic_domains(nanoBragg const& nanoBragg) {
      return nanoBragg.mosaic_domains;
  }
  static void   set_mosaic_domains(nanoBragg& nanoBragg, int const& value) {
      nanoBragg.mosaic_domains = value;
      /* need to re-create mosaic domain table */
      nanoBragg.init_mosaicity();
      nanoBragg.init_steps();
      //nanoBragg.show_mosaic_blocks();
  }



  /* rotate a vector using a 9-element unitary matrix */
  vec3 rotate_umat_nks(const double *v, const double umat[9]) {

    double uxx,uxy,uxz,uyx,uyy,uyz,uzx,uzy,uzz;

    /* for clarity, assign matrix x-y coordinate */
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
    return vec3 (uxx*v[1] + uxy*v[2] + uxz*v[3],
                 uyx*v[1] + uyy*v[2] + uyz*v[3],
                 uzx*v[1] + uzy*v[2] + uzz*v[3]);
  }
  static af::shared<mat3> get_mosaic_domains_abc_phi_0(nanoBragg const& X){

    /* return individual mosaic domain orientations */
    /* only defined (NKS) for the phi==0 case */
    SCITBX_ASSERT(X.mosaic_spread > 0 || X.mosaic_domains==1);
    af::shared<mat3> result = af::shared<mat3>();
    vec3 a,b,c;                  // cell vectors in meters
    for(int mos_tic=0;mos_tic<X.mosaic_domains;++mos_tic) {
                                /* apply mosaic rotation with zero phi rotation */
                                if( X.mosaic_spread > 0.0 )
                                {
                                    a = rotate_umat_nks(X.a0,&X.mosaic_umats[mos_tic*9]);
                                    b = rotate_umat_nks(X.b0,&X.mosaic_umats[mos_tic*9]);
                                    c = rotate_umat_nks(X.c0,&X.mosaic_umats[mos_tic*9]);
                                }
      result.push_back(
        mat3(1e10*a[0],1e10*a[1],1e10*a[2],1e10*b[0],1e10*b[1],1e10*b[2],1e10*c[0],1e10*c[1],1e10*c[2])
      );
    }
    return result;
  }

  /* spindle starting angle phi, in deg */
  static double get_phi_deg(nanoBragg const& nanoBragg) {
      return nanoBragg.phi0*RTD;;
  }
  static void   set_phi_deg(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.phi0 = value/RTD;
      /* need to re-create phi step table */
      nanoBragg.init_steps();
      if(nanoBragg.verbose) nanoBragg.show_phisteps();
  }

  /* spindle rotation ange, in deg */
  static double get_osc_deg(nanoBragg const& nanoBragg) {
      return nanoBragg.osc*RTD;;
  }
  static void   set_osc_deg(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.osc = value/RTD;
      /* need to re-create phi step table */
      nanoBragg.phistep=-1.0;
      nanoBragg.init_steps();
      if(nanoBragg.verbose) nanoBragg.show_phisteps();
  }

  /* number of discrete phi rotations, default: 1 or 2 ; you will want more */
  static int get_phisteps(nanoBragg const& nanoBragg) {
      return nanoBragg.phisteps;
  }
  static void   set_phisteps(nanoBragg& nanoBragg, int const& value) {
      nanoBragg.phisteps = value;
      /* need to re-create phi step table */
      nanoBragg.phistep=-1.0;
      nanoBragg.init_steps();
      if(nanoBragg.verbose) nanoBragg.show_phisteps();
  }

  /* spindle angle phi step, in deg */
  static double get_phistep_deg(nanoBragg const& nanoBragg) {
      return nanoBragg.phistep*RTD;;
  }
  static void   set_phistep_deg(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.phistep = value/RTD;
      /* need to re-create phi step table */
      nanoBragg.init_steps();
      if(nanoBragg.verbose) nanoBragg.show_phisteps();
  }


  /* detector active layer thickness in mm */
  static double get_detector_thick_mm(nanoBragg const& nanoBragg) {
      return nanoBragg.detector_thick*1000.;
  }
  static void   set_detector_thick_mm(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.detector_thick = value/1000.;
      /* need to re-create thick step table */
      nanoBragg.detector_thickstep=-1.0;
      nanoBragg.init_steps();
      if(nanoBragg.verbose) nanoBragg.show_detector_thicksteps();
  }

  /* number of discrete detector layers, default: 1 or 2 ; you will want more */
  static int get_detector_thicksteps(nanoBragg const& nanoBragg) {
      return nanoBragg.detector_thicksteps;
  }
  static void   set_detector_thicksteps(nanoBragg& nanoBragg, int const& value) {
      nanoBragg.detector_thicksteps = value;
      /* need to re-create detector layer table */
      nanoBragg.detector_thickstep=-1.0;
      nanoBragg.init_steps();
      if(nanoBragg.verbose) nanoBragg.show_detector_thicksteps();
  }

  /* optionally specify detector sub-layer thickness, in um */
  static double get_detector_thickstep_mm(nanoBragg const& nanoBragg) {
      return nanoBragg.detector_thickstep*1000.;
  }
  static void   set_detector_thickstep_mm(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.detector_thickstep = value/1000.;
      /* need to re-create detector layer table */
      nanoBragg.init_steps();
      if(nanoBragg.verbose) nanoBragg.show_detector_thicksteps();
  }

  /* detector active layer attenuation length in mm */
  static double get_detector_attenuation_mm(nanoBragg const& nanoBragg) {
      /* internal storage is always in meters */
      return nanoBragg.detector_attnlen*1000;
  }
  static void   set_detector_attenuation_mm(nanoBragg& nanoBragg, double const& value) {
      /* internal storage is always in meters */
      nanoBragg.detector_attnlen = value/1000.;
  }


  /* noise creation parameters */

  /* seed values need to be negated to re-initialize the generator */
  /* at the moment, trying to do this on-the-fly */

  /* prenoise scale need not be scaled */
  /* quantum_gain need not be scaled */
  /* adc_offset need not be scaled */

  /* detector calibration noise, usually 2-3%  Sometimse as low as 0.9%  */
  static double get_calibration_noise_pct(nanoBragg const& nanoBragg) {
      return nanoBragg.calibration_noise*100.;
  }
  static void   set_calibration_noise_pct(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.calibration_noise = value/100.;
  }

  /* incident beam flicker noise, usually ~0.1-0.2% at synchrotron, ~100% at XFEL  */
  static double get_flicker_noise_pct(nanoBragg const& nanoBragg) {
      return nanoBragg.flicker_noise*100.;
  }
  static void   set_flicker_noise_pct(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.flicker_noise = value/100.;
  }

  /* readout_noise need not be scaled  */
  /* psf_type need not be scaled  */

  /* detector point-spread function full-width at half-maximum value, in mm */
  static double get_psf_fwhm_mm(nanoBragg const& nanoBragg) {
      return nanoBragg.psf_fwhm*1000.;
  }
  static void   set_psf_fwhm_mm(nanoBragg& nanoBragg, double const& value) {
      nanoBragg.psf_fwhm = value/1000.;
  }


  static PyObject*
  raw_pixels_unsigned_short_as_python_bytes( nanoBragg const& nB,
    double intfile_scale, int const&debug_x, int const& debug_y) {
    std::basic_ostringstream<char> os;
    const double* floatimage = nB.raw_pixels.begin();
    double max_value = (double)std::numeric_limits<unsigned short int>::max();
    double saturation = floor(max_value - 1 );
    /* output as ints */

    unsigned short int intimage;
    double max_I = nB.max_I;
    double max_I_x = nB.max_I_x;
    double max_I_y = nB.max_I_y;
    if(intfile_scale <= 0.0){
        /* need to auto-scale */
        int i=0;
        for(int spixel=0;spixel<nB.spixels;++spixel)
        {
            for(int fpixel=0;fpixel<nB.fpixels;++fpixel)
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
        if(nB.verbose) printf("providing default scaling: max_I = %g @ (%g %g)\n",max_I,max_I_x,max_I_y);
        intfile_scale = 1.0;
        if(max_I>0.0) intfile_scale = 55000.0/(max_I);
    }
    if(nB.verbose) printf("scaling data by: intfile_scale = %g\n",intfile_scale);
    double sum = 0.0;
    max_I = 0.0;
    int i = 0;
    for(int spixel=0;spixel<nB.spixels;++spixel)
    {
        for(int fpixel=0;fpixel<nB.fpixels;++fpixel)
        {
            /* no noise, just use intfile_scale */
            intimage = (unsigned short int) (std::min(saturation, floatimage[i]*intfile_scale ));
            os.write((char *) &intimage, sizeof(unsigned short int));

            if(nB.verbose>90) printf("DEBUG #%d %g -> %g -> %d\n",i,floatimage[i],floatimage[i]*intfile_scale,intimage);

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
  #ifdef IS_PY3K
    return PyBytes_FromStringAndSize(os.str().c_str(), os.str().size());
  #else
    return PyString_FromStringAndSize(os.str().c_str(), os.str().size());
  #endif
  }


  void
  init_module() {
    using namespace boost::python;

    typedef return_value_policy<return_by_value> rbv;
    typedef return_value_policy<copy_const_reference> ccr;
    typedef return_internal_reference<> rir;
    typedef default_call_policies dcp;

    def("testuple", &testuple,
        "test that basic python stuff works, as well as NAN implementation");
    // Expose enums to Python
    enum_<pivot>("pivot",
     "description of detector pivot point for detector rotation angles")
      .value("Beam",BEAM)
      .value("Sample",SAMPLE)
      .export_values();
    enum_<shapetype>("shapetype",
       "select shape of the crystal, spot, or point-spread function" )
      .value("Square",SQUARE)
      .value("Round",ROUND)
      .value("Gauss",GAUSS)
      .value("Gauss_argchk",GAUSS_ARGCHK)
      .value("Gauss_star",GAUSS_STAR)
      .value("Tophat",TOPHAT)
      .value("Fiber",FIBER)
      .value("Unknown",UNKNOWN)
      .export_values();
    enum_<convention>("convention",
       "beam center convention to use when interpreting Xbeam Ybeam")
      .value("Custom",CUSTOM)
      .value("ADXV",ADXV)
      .value("MOSFLM",MOSFLM)
      .value("XDS",XDS)
      .value("DIALS",DIALS)
      .value("DENZO",DENZO)
      .export_values();
    // end of enum definitions

    class_<nanoBragg>("nanoBragg",no_init)
      /* constructor that takes a dxtbx detector and beam model */
      .def(init<const dxtbx::model::Detector&,
                const dxtbx::model::Beam&,
                int, int>(
        (arg_("detector"),
         arg_("beam"),
         arg_("verbose")=0,
         arg_("panel_id")=0),
        "nanoBragg simulation initialized from dxtbx detector and beam objects"))

       /* constructor that takes any and all parameters with sensible defaults */
      .def(init<scitbx::vec2<int>,
                scitbx::vec3<double>,
                cctbx::uctbx::unit_cell,
                vec3,
                vec2,
                double,
                double,
                double,
                double,
                double,
                double,
                int,
                int>(
        (arg_("detpixels_slowfast")=scitbx::vec2<int>(1024,1024),
         arg_("Ncells_abc")=scitbx::vec3<double>(1,1,1),
         arg_("unit_cell_Adeg")=cctbx::uctbx::unit_cell(scitbx::af::double6(78.0,78.0,38.0,90.0,90.0,90.0)),
         arg_("misset_deg")=vec3(0.0,0.0,0.0),
         arg_("beam_center_mm")=vec2(NAN,NAN),
         arg_("distance_mm")=100,
         arg_("pixel_size_mm")=0.1,
         arg_("wavelength_A")=1,
         arg_("divergence_mrad")=0,
         arg_("dispersion_pct")=0,
         arg_("mosaic_spread_deg")=0,
         arg_("oversample")=0,
         arg_("verbose")=0),
         "nanoBragg simulation initialized with most common parameters. All parameters have sensible defaults."))

      .def("free_all",&nanoBragg::free_all)
      /* toggle printing stuff to screen - does not work completely yet */
      .add_property("verbose",
                     make_getter(&nanoBragg::verbose,rbv()),
                     make_setter(&nanoBragg::verbose,dcp()),
                     "non-zero turns on debugging messages of increasing verbosity.")
      /* toggle detailed print-out of each and every pixel value and parameters */
      .add_property("printout",
                     make_getter(&nanoBragg::printout,rbv()),
                     make_setter(&nanoBragg::printout,dcp()),
                     "debugging option: print to screen all values associated with a pixel")
      /* select a single pixel to be verbose about */
      .add_property("printout_pixel_fastslow",
                     make_function(&get_printout_pixel_fastslow,rbv()),
                     make_function(&set_printout_pixel_fastslow,dcp()),
                     "fast/slow pixel coordinate of particular pixel to print to screen.  Default is all if printout=True.")
      /* toggle screen printing of progrees in percent */
      .add_property("progress_meter",
                     make_getter(&nanoBragg::progress_meter,rbv()),
                     make_setter(&nanoBragg::progress_meter,dcp()),
                     "toggle screen printing of simulation progress in percent.")
      /* expose image rendering progress */
      .add_property("progress_pixel",
                     make_getter(&nanoBragg::progress_pixel,rbv()),
                     make_setter(&nanoBragg::progress_pixel,dcp()),
                     "Pixel number that is currently being rendered.")



      /* number of unit cells in each direction */
      .add_property("Ncells_abc",
                     make_function(&get_Nabc,rbv()),
                     make_function(&set_Nabc,dcp()),
                     "number of unit cells along each unit cell axis")
      /* alternatively, specify crystal size in mm instead of number of cells */
      .add_property("xtal_size_mm",
                     make_function(&get_xtal_size_mm,rbv()),
                     make_function(&set_xtal_size_mm,dcp()),
                     "alternative: specify crystal size (mm) instead of Ncells_abc")



      /* specify properties of amorphous material contribution to background */
      .add_property("amorphous_sample_size_mm",
                     make_function(&get_amorphous_sample_size_mm,rbv()),
                     make_function(&set_amorphous_sample_size_mm,dcp()),
                     "specify dimensions of amorphous material in beam (mm)")
      .add_property("amorphous_sample_thick_mm",
                     make_function(&get_amorphous_sample_thick_mm,rbv()),
                     make_function(&set_amorphous_sample_thick_mm,dcp()),
                     "thickness of amorphous material (mm) is most important along beam path")
      .add_property("amorphous_density_gcm3",
                     make_function(&get_amorphous_density_gcm3,rbv()),
                     make_function(&set_amorphous_density_gcm3,dcp()),
                     "density of amorphous material (g/cm^3), needed to calculate scale factor")
      .add_property("amorphous_molecular_weight_Da",
                     make_function(&get_amorphous_molecular_weight_Da,rbv()),
                     make_function(&set_amorphous_molecular_weight_Da,dcp()),
                     "molecular weight of amorphous material (Da, or g/mol), needed to calculate scale factor")



      /* unit cell dimensions */
      .add_property("unit_cell_Adeg",
                     make_function(&get_cell_Adeg,rbv()),
                     make_function(&set_cell_Adeg,dcp()),
                     "uctbx unit cell")
      .add_property("unit_cell_tuple",
                     make_function(&get_cell_astuple,rbv()),
                     make_function(&set_cell_astuple,dcp()),
                     "unit cell as a 6-tuple")

      /* crystal orientation as missetting angles (deg) */
      .add_property("missets_deg",
                     make_function(&get_misset_deg,rbv()),
                     make_function(&set_misset_deg,dcp()),
                     "crystal mis-orientation about x,y,z axes as missetting angles (deg)")
      /* crystal orientation as an A=UB matrix */
      .add_property("Amatrix",
                     make_function(&get_Amatrix,rbv()),
                     make_function(&set_Amatrix,dcp()),
                     "unit cell, wavelength, and crystal orientation as Arndt & Wonacott A=UB matrix")
      /* crystal orientation as an A=UB matrix. No further missetting angles to be applied */
      .add_property("Amatrix_RUB",
                     make_function(&get_Amatrix,rbv()),
                     make_function(&set_Amatrix_NKS_implementation,dcp()),
                     "unit cell, wavelength, and crystal orientation as Arndt & Wonacott A=UB matrix")

      /* beam center, in convention specified below */
      .add_property("beam_center_mm",
                     make_function(&get_beam_center_mm,rbv()),
                     make_function(&set_beam_center_mm,dcp()),
                     "x-ray beam center (mm) in specified convention (dials, adxv, mosflm, denzo, XDS)")
      .add_property("XDS_ORGXY",
                     make_function(&get_XDS_ORGXY,rbv()),
                     make_function(&set_XDS_ORGXY,dcp()),
                     "XDS-convention beam center is nearest pixel in detector plane to sample position (fast,slow)")
      .add_property("adxv_beam_center_mm",
                     make_function(&get_adxv_beam_center_mm,rbv()),
                     "ADXV beam center (mm) (fast,slow)")

      .add_property("mosflm_beam_center_mm",
                     make_function(&get_mosflm_beam_center_mm,rbv()),
                     "MOSFLM beam center (mm) (slow,fast)")

      .add_property("denzo_beam_center_mm",
                     make_function(&get_denzo_beam_center_mm,rbv()),
                     "DENZO beam center (mm) (slow,fast)")

      .add_property("dials_origin_mm",
                     make_function(&get_dials_origin_mm,rbv()),
                     "DIALS detector origin (mm)")

      /* specify the detector pivot point, note: this is an enum */
      .add_property("detector_pivot",
                     make_getter(&nanoBragg::detector_pivot,rbv()),
                     make_setter(&nanoBragg::detector_pivot,dcp()),
                     "specify the center of detector rotations, either the sample, or the direct-beam spot. Must import pivot to use this")

      /* specify the crystal (and therefore spot) shape, note: this is an enum */
      .add_property("xtal_shape",
                     make_getter(&nanoBragg::xtal_shape,rbv()),
                     make_setter(&nanoBragg::xtal_shape,dcp()),
                     "specify the nano crystal shape, which determines the spot shape.  square is exact, round is approximate ellipsoid, gauss for gaussian spots, or tophat for top-hat-spots.  Must import shapetype to use this")

      /* specify the detector pivot point, note: this is an enum */
      .add_property("beamcenter_convention",
                     make_function(&get_beam_convention,rbv()),
                     make_function(&set_beam_convention,dcp()),
                     "specify the convention used to define the beam center (DIALS, MOSFLM, XDS, DENZO, ADXV, Custom).  Must import convention to use this")



      /* specify convention? Custom, MOSFLM, XDS, DENZO, ADXV, DIALS */

      /* dxtbx origin vector getter */
      .add_property("pix0_vector_mm", /* No setter allowed for pix0 */
                     make_function(&get_pix0_vector,rbv()),
                     "Current state of the internal origin vector (in mm) pointing to the first pixel in memory")
      /* unit vectors specifying coordinate system */
      .add_property("fdet_vector",
                     make_function(&get_fdet_vector,rbv()),
                     make_function(&set_fdet_vector,dcp()),
                     "unit vector representing 3-space direction of increasing fast-pixel-coordinate values")
      .add_property("sdet_vector",
                     make_function(&get_sdet_vector,rbv()),
                     make_function(&set_sdet_vector,dcp()),
                     "unit vector representing 3-space direction of increasing slow-pixel-coordinate values")
      .add_property("odet_vector",
                     make_function(&get_odet_vector,rbv()),
                     make_function(&set_odet_vector,dcp()),
                     "unit vector representing 3-space direction of increasing detector distance")
      .add_property("beam_vector",
                     make_function(&get_beam_vector,rbv()),
                     make_function(&set_beam_vector,dcp()),
                     "unit vector representing 3-space direction of x-ray beam propagation (the k-vector)")
      .add_property("polar_vector",
                     make_function(&get_polar_vector,rbv()),
                     make_function(&set_polar_vector,dcp()),
                     "unit vector representing 3-space direction of x-ray beam polarization (the E-vector)")
      .add_property("polar_Bvector",
                     make_function(&get_polar_Bvector,rbv()),
                     make_function(&set_polar_Bvector,dcp()),
                     "unit vector representing 3-space direction of x-ray beam polarization (the B-vector), equivalent to polarization_normal")
      .add_property("spindle_axis",
                     make_function(&get_spindle_vector,rbv()),
                     make_function(&set_spindle_vector,dcp()),
                     "unit vector representing 3-space direction of crystal phi spindle rotation axis")
      .add_property("twotheta_axis",
                     make_function(&get_twotheta_axis,rbv()),
                     make_function(&set_twotheta_axis,dcp()),
                     "unit vector representing 3-space direction of detector twotheta rotation axis (about sample)")

      /* sample-to-direct-beam-spot distance */
      .add_property("distance_meters",
                     make_getter(&nanoBragg::distance,rbv()),
                     make_setter(&nanoBragg::distance,dcp()),
                     "sample-to-detector distance (meters)")
      .add_property("distance_mm",
                     make_function(&get_distance_mm,rbv()),
                     make_function(&set_distance_mm,dcp()),
                     "sample-to-detector distance (mm)")
      /* sample-to-nearest-point-on-detector distance */
      .add_property("close_distance_mm",
                     make_function(&get_close_distance_mm,rbv()),
                     make_function(&set_close_distance_mm,dcp()),
                     "distance between sample and nearest point in the detector pixel plane (mm)")

      /* rotation of detector about sample */
      .add_property("detector_twotheta_deg",
                     make_function(&get_detector_twotheta_deg,rbv()),
                     make_function(&set_detector_twotheta_deg,dcp()),
                     "rotation of detector about twotheta_axis at sample position")

      /* detector face size in mm */
      .add_property("detsize_fastslow_mm",
                     make_function(&get_detsize_fastslow_mm,rbv()),
                     make_function(&set_detsize_fastslow_mm,dcp()),
                     "overall size of the detector fast and slow pixel directions (mm)")
      /* detector face size in pixels */
      .add_property("detpixels_fastslow",
                     make_function(&get_detpixels_fastslow,rbv()),
                     make_function(&set_detpixels_fastslow,dcp()),
                     "overall size of the detector fast and slow pixel directions (pixels)")

      /* rotation of detector about direct-beam-spot or sample, depending on pivot */
      .add_property("detector_rotation_deg",
                     make_function(&get_detector_rotation_deg,rbv()),
                     make_function(&set_detector_rotation_deg,dcp()),
                     "rotation of detector (deg) about direct-beam-spot (or sample position, depending on pivot)")

      /* True forces all pixels to be same distance from sample */
      .add_property("curved_detector",
                     make_getter(&nanoBragg::curved_detector,rbv()),
                     make_setter(&nanoBragg::curved_detector,dcp()),
                     "True forces all pixels to be same distance from sample")

      /* detector pixel size in mm */
      .add_property("pixel_size_mm",
                     make_function(&get_pixel_size_mm,rbv()),
                     make_function(&set_pixel_size_mm,dcp()),
                     "square detector pixel size (mm)")

      /* True turns solid angle effect off */
      .add_property("point_pixel",
                     make_getter(&nanoBragg::point_pixel,rbv()),
                     make_setter(&nanoBragg::point_pixel,dcp()),
                     "True turns pixel solid angle effect off, pixels behave like infinitely sharp points")

      /* set Kahn polarization factor between -1 and 1, default: 0  - note polarization=0 is DIFFERENT from nopolar */
      .add_property("polarization",
                     make_getter(&nanoBragg::polarization,rbv()),
                     make_setter(&nanoBragg::polarization,dcp()),
                     "set Kahn polarization factor between -1 and 1, default: 0  - note polarization=0 is DIFFERENT from nopolar")
      /* True turns polarization effect off */
      .add_property("nopolar",
                     make_getter(&nanoBragg::nopolar,rbv()),
                     make_setter(&nanoBragg::nopolar,dcp()),
                     "True turns polarization effect off, F^2 scattering in all directions")

     /* override oversampling, default: reciprocal sub-pixel fits xtal_size */
      .add_property("oversample",
                     make_function(&get_oversample,rbv()),
                     make_function(&set_oversample,dcp()),
                     "override pixel oversampling, speed is inversely proportional to square of this quantity, but critical if reciprocal crystal size is smaller than a pixel. default: set oversample so that reciprocal sub-pixel fits xtal_size")

     /* select a region-of-interest: xmin xmax ymin ymax */
      .add_property("region_of_interest",
                     make_function(&get_roi,rbv()),
                     make_function(&set_roi,dcp()),
                     "region of interst on detector: fast_min fast_max slow_min slow_max")

      /* wavelength in Angstrom */
      .add_property("wavelength_A",
                     make_function(&get_lambda_A,rbv()),
                     make_function(&set_lambda_A,dcp()),
                     "centroid x-ray wavelength (Angstrom)")
      /* or photon energy */
      .add_property("energy_eV",
                     make_function(&get_energy_eV,rbv()),
                     make_function(&set_energy_eV,dcp()),
                     "alternately specify x-ray wavelength as a photon energy (eV)")

      /* override value of x-ray fluence in photons/m^2 : default is sufficient for intensity=F^2 */
      .add_property("fluence",
                     make_function(&get_fluence,rbv()),
                     make_function(&set_fluence,dcp()),
                     "x-ray fluence upon sample (photons/meter^2), default makes photons/steradian=F^2 exactly")
      /* specify x-ray flux in photons/s, updates fluence */
      .add_property("flux",
                     make_function(&get_flux,rbv()),
                     make_function(&set_flux,dcp()),
                     "x-ray flux (photons/s), mainly to update fluence")
      /* exposure time, updates fluence */
      .add_property("exposure_s",
                     make_function(&get_exposure_s,rbv()),
                     make_function(&set_exposure_s,dcp()),
                     "x-ray exposure time (seconds), mainly to update fluence")
      /* x-ray beam size in mm, updates fluence */
      .add_property("beamsize_mm",
                     make_function(&get_beamsize_mm,rbv()),
                     make_function(&set_beamsize_mm,dcp()),
                     "x-ray beam size in both directions (mm), mainly to update fluence")

      /* x-ray beam spectral dispersion in percent */
      .add_property("dispersion_pct",
                     make_function(&get_dispersion_pct,rbv()),
                     make_function(&set_dispersion_pct,dcp()),
                     "x-ray beam spectral dispersion (%)")
      /* spectral dispersion steps */
      .add_property("dispsteps",
                     make_function(&get_dispsteps,rbv()),
                     make_function(&set_dispsteps,dcp()),
                     "number of micro-steps across wavelength range, speed is inversely proportional to this")

      /* x-ray beam divergence in mrad */
      .add_property("divergence_hv_mrad",
                     make_function(&get_divergence_hv_mrad,rbv()),
                     make_function(&set_divergence_hv_mrad,dcp()),
                     "x-ray beam divergence in horizontal and vertical directions (mrad)")
      /* beam divergence micro-steps in wavelength sweep */
      .add_property("divsteps_hv",
                     make_function(&get_divsteps_hv,rbv()),
                     make_function(&set_divsteps_hv,dcp()),
                     "number of x-ray beam divergence micro-steps across source in horizontal and vertical directions, speed is inversely proportional to this")
      /* beam divergence internal micro-step size in mrad */
      .add_property("divstep_hv_mrad",
                     make_function(&get_divstep_hv_mrad,rbv()),
                     make_function(&set_divstep_hv_mrad,dcp()),
                     "beam divergence internal micro-step size across source in horizontal and vertical directions (mrad)")
      /* Trim automatic range of sources for divergence to lie within an ellipse instead of a rectangle */
      .add_property("round_div",
                     make_getter(&nanoBragg::round_div,rbv()),
                     make_setter(&nanoBragg::round_div,dcp()),
                     "Trim automatic range of sources for divergence to lie within an ellipse instead of a rectangle")

      /* spindle phi rotation start in deg */
      .add_property("phi_deg",
                     make_function(&get_phi_deg,rbv()),
                     make_function(&set_phi_deg,dcp()),
                     "spindle phi rotation start angle (deg)")
      /* spindle phi rotation range in deg */
      .add_property("osc_deg",
                     make_function(&get_osc_deg,rbv()),
                     make_function(&set_osc_deg,dcp()),
                     "spindle phi rotation range (deg)")
      /* number of internal phi micro-steps in phi sweep */
      .add_property("phisteps",
                     make_function(&get_phisteps,rbv()),
                     make_function(&set_phisteps,dcp()),
                     "number of internal phi micro-steps in phi sweep, speed is inversely proportional to this")
      /* size of the internal phi micro-step, in deg */
      .add_property("phistep_deg",
                     make_function(&get_phistep_deg,rbv()),
                     make_function(&set_phistep_deg,dcp()),
                     "size of the internal phi micro-step (deg)")


      /* detector x-ray sensitive layer thickness in mm */
      .add_property("detector_thick_mm",
                     make_function(&get_detector_thick_mm,rbv()),
                     make_function(&set_detector_thick_mm,dcp()),
                     "detector x-ray sensitive layer thickness in millimeters (mm)")
      /* detector x-ray sensitive layer attenuation length in mm */
      .add_property("detector_attenuation_length_mm",
                     make_function(&get_detector_attenuation_mm,rbv()),
                     make_function(&set_detector_attenuation_mm,dcp()),
                     "detector x-ray sensitive layer attenuation length in millimeters (mm)")
      /* number of internal detector layers in thickness sweep */
      .add_property("detector_thicksteps",
                     make_function(&get_detector_thicksteps,rbv()),
                     make_function(&set_detector_thicksteps,dcp()),
                     "number of internal detector layers in thickness sweep, speed is inversely proportional to this")
      /* size of the internal detector thickness micro-step, in mm */
      .add_property("detector_thickstep_mm",
                     make_function(&get_detector_thickstep_mm,rbv()),
                     make_function(&set_detector_thickstep_mm,dcp()),
                     "size of the internal detector thickness micro-step, in millimeters (mm)")


      /* crystal mosaic spread spherical cap range in deg */
      .add_property("mosaic_spread_deg",
                     make_function(&get_mosaic_spread_deg,rbv()),
                     make_function(&set_mosaic_spread_deg,dcp()),
                     "crystal mosaic spread spherical cap radius, not range (deg)")
      /* number of discrete mosaic domains to generate, speed is inversely proportional to this */
      .add_property("mosaic_domains",
                     make_function(&get_mosaic_domains,rbv()),
                     make_function(&set_mosaic_domains,dcp()),
                     "number of discrete mosaic domains to generate, speed is inversely proportional to this")
      .def          ("get_mosaic_domains_abc_phi_0",&get_mosaic_domains_abc_phi_0,
                     "get an array of 3x3 matrices giving the a,b,c vectors of each mosaic domain at phi 0")
      .def          ("show_mosaic_blocks",
                     &nanoBragg::show_mosaic_blocks,
                     "print out individual mosaic domain orientations")
      .def          ("get_mosaic_blocks",
                     &nanoBragg::get_mosaic_blocks,
                     "return the unitary matrices U that define the mosaic block distribution")
      .def          ("set_mosaic_blocks",
                     &nanoBragg::set_mosaic_blocks,
                     "enter an arbitrary list of unitary matrices U to define the mosaic block distribution")

      /* hkl and F */
      .add_property("indices",
                     make_function(&get_indices,rbv()),
                     make_function(&set_indices,dcp()),
                     "miller flex array, use miller array Fhkl instead")
      .add_property("amplitudes",
                     make_function(&get_amplitudes,rbv()),
                     make_function(&set_amplitudes,dcp()),
                     "structure factor amplitude flex array, use miller array Fhkl instead")
      /* hkl and F as a 2-tuple of indices and flex-double */
      .add_property("Fhkl_tuple",
                     make_function(&get_Fhkl_tuple,rbv()),
                     make_function(&set_Fhkl_tuple,dcp()),
                     "hkl and F as a 2-tuple of indices and flex-double, use miller array Fhkl instead")
      /* override value of missing Fs, default 0 */
      .add_property("default_F",
                     make_function(&get_default_F,rbv()),
                     make_function(&set_default_F,dcp()),
                     "override value of missing Fs, default 0 (useful if you just want spots fast)")
      /* direct access to F000 term, will appear at beam center, default 0 */
      .add_property("F000",
                     make_function(&get_F000,rbv()),
                     make_function(&set_F000,dcp()),
                     "override value of F000, default 0 (useful for marking beam center)")


      /* background intensity structure factor vs sin(theta)/lambda */
      .add_property("Fbg_vs_stol",
                     make_function(&get_Fbg_vs_stol,rbv()),
                     make_function(&set_Fbg_vs_stol,dcp()),
                     "background-scattering structure factor (Fbg) is second member of vector, first is sin(theta)/lambda (stol), just like the International Tables for individual atomic form factors ")
      /* override value of missing background structure factor (Fbg), default 0 */
      .add_property("default_Fbg",
                     make_function(&get_default_Fbg,rbv()),
                     make_function(&set_default_Fbg,dcp()),
                     "override value of missing background-scatter structure factor (Fbg), default 0 (useful if you just want some uniform background)")


      /* x-ray source position, intensity and wavelength list */
      .add_property("xray_beams",
                     make_function(&get_beams,rbv()),
                     make_function(&set_beams,dcp()),
                     "list of dxtbx::Beam objects corresponding to each zero-divergence and monochromatic x-ray point source in the numerical simulation ")
      /* x-ray sources, raw position list */
      .add_property("xray_source_XYZ",
                     make_function(&get_source_XYZ,rbv()),
                     make_function(&set_source_XYZ,dcp()),
                     "list of the xyz positon in space of each x-ray point source, same axis convention as detector vectors ")
      /* x-ray sources, raw wavelength list */
      .add_property("xray_source_wavelengths_A",
                     make_function(&get_source_lambda,rbv()),
                     make_function(&set_source_lambda,dcp()),
                     "list of the wavelengths (A) for each x-ray point source. default is to initialize with nanoBragg.wavelength_A ")
      /* x-ray sources, raw position list */
      .add_property("xray_source_intensity_fraction",
                     make_getter(&nanoBragg::pythony_source_intensity,rbv()),
                     make_setter(&nanoBragg::pythony_source_intensity,dcp()),
                     "list of relative intensities of each x-ray point source, should always sum to unity ")


      /* toggle interpolation between spot structure factors */
      .add_property("interpolate",
                     make_getter(&nanoBragg::interpolate,rbv()),
                     make_setter(&nanoBragg::interpolate,dcp()),
                     "toggle interpolation between structure factors for inter-Bragg peaks")
      /* experimental: use integral form instead of oversampling */
      .add_property("integral_form",
                     make_getter(&nanoBragg::integral_form,rbv()),
                     make_setter(&nanoBragg::integral_form,dcp()),
                     "experimental: use integral form instead of oversapmling")

      /* random number seed for mosaic domain generation */
      .add_property("mosaic_seed",
                     make_getter(&nanoBragg::mosaic_seed,rbv()),
                     make_setter(&nanoBragg::mosaic_seed,dcp()),
                     "random number seed for mosaic domain generation, keep the same for a given crystal")
      /* random number seed for noise generation (use a different value for different images) */
      .add_property("seed",
                     make_getter(&nanoBragg::seed,rbv()),
                     make_setter(&nanoBragg::seed,dcp()),
                     "random number seed for noise generation, use a different value for different images")
      /* random number seed for detector calibration error (usually same for all frames) */
      .add_property("calib_seed",
                     make_getter(&nanoBragg::calib_seed,rbv()),
                     make_setter(&nanoBragg::calib_seed,dcp()),
                     "random number seed for detector calibration error, keep the same for a given detector")

      /* noise generation parameters */
      .add_property("fudge",
                     make_getter(&nanoBragg::fudge,rbv()),
                     make_setter(&nanoBragg::fudge,dcp()),
                     "fudge factor on F_latt falloff (default=1)")
      .add_property("spot_scale",
                     make_getter(&nanoBragg::spot_scale,rbv()),
                     make_setter(&nanoBragg::spot_scale,dcp()),
                     "scale factor on spot photons applied BEFORE computing photon-counting or other errors (default: 1, but may be increased to model multiple mosaic domains in the same orientation)")
       /* image file offset in converting photons to pixel values */
      .add_property("quantum_gain",
                     make_getter(&nanoBragg::quantum_gain,rbv()),
                     make_setter(&nanoBragg::quantum_gain,dcp()),
                     "scale factor on observed photons applied AFTER computing Poisson photon-counting error (usually 1 or more)")
       /* image file offset in converting photons to pixel values */
     .add_property("adc_offset_adu",
                     make_getter(&nanoBragg::adc_offset,rbv()),
                     make_setter(&nanoBragg::adc_offset,dcp()),
                     "constant offset added to all pixel values before rounding off to integers (40 on ADSC, 10 on Rayonix, 0 for Pilatus)")

      /* magnitude of calibration noise */
      .add_property("detector_calibration_noise_pct",
                     make_function(&get_calibration_noise_pct,rbv()),
                     make_function(&set_calibration_noise_pct,dcp()),
                     "detector calibration error; implemented as a fixed Gaussian deviate for each pixel. (%)")
      /* magnitude of flicker noise */
      .add_property("flicker_noise_pct",
                     make_function(&get_flicker_noise_pct,rbv()),
                     make_function(&set_flicker_noise_pct,dcp()),
                     "incident beam flicker or shot-to-shot intensity variation; implemented as a fixed Gaussian deviate for each image. (%)")
      /* magnitude of readout noise */
     .add_property("readout_noise_adu",
                     make_getter(&nanoBragg::readout_noise,rbv()),
                     make_setter(&nanoBragg::readout_noise,dcp()),
                     "extra noise typically associated with CCD read-out event; applied after any quantum gain or adc offset before converting to integer pixel value; implemented as a fixed Gaussian deviate for each pixel. units are Arbitrary Detector Units (ADU)")

      /* detector point-spread function shape type */
      .add_property("detector_psf_type",
                     make_getter(&nanoBragg::psf_type,rbv()),
                     make_setter(&nanoBragg::psf_type,dcp()),
                     "shape of the detector point-spread function. valid values are: gauss for Gaussian spots, fiber for CCD-like spots, or unknown.  Must import shapetype to use this")
      /* detector point-spread function FWHM */
      .add_property("detector_psf_fwhm_mm",
                     make_function(&get_psf_fwhm_mm,rbv()),
                     make_function(&set_psf_fwhm_mm,dcp()),
                     "width of the detector point-spread function at half maximum in mm.  Set to zero to disable PSF.")
      /* detector point-spread function rendering radius */
      .add_property("detector_psf_kernel_radius_pixels",
                     make_getter(&nanoBragg::psf_radius,rbv()),
                     make_setter(&nanoBragg::psf_radius,dcp()),
                     "override size of kernel used to compute PSF.  Set to zero for automatic (the default).")

      /* 2D flex array representing pixel values in expected photons, not neccesarily integers */
      .add_property("raw_pixels",
                     make_getter(&nanoBragg::raw_pixels,rbv()),
                     make_setter(&nanoBragg::raw_pixels,dcp()),
                     "2D flex array representing floating-point pixel values, this is expected photons before you call add_noise(), which converts it into detector pixel values, or ADU")

     .add_property("cbf_int",
                     make_getter(&nanoBragg::cbf_int,rbv()),
                     make_setter(&nanoBragg::cbf_int,dcp()),
                     "Write the cbf file using to_cbf with int32 precision")

      /* print to screen a summary of all initialized parameters */
      .def("show_params",&nanoBragg::show_params,
       "print out all simulation parameters, just like the standalone program")

      /* print to screen a summary of all initialized sources */
      .def("show_sources",&nanoBragg::show_sources,
       "print out all internal x-ray source parameters, just like the standalone program")

      /* randomize crystal orientation */
      .def("randomize_orientation",&nanoBragg::randomize_orientation,
       "rotate crystal to a random orientation, updates Amatrix and missets normally seeded with time, set nanoBragg.seed to get the same random orientation each time")

      /* actual run of the spot simulation */
      .def("add_nanoBragg_spots",&nanoBragg::add_nanoBragg_spots,
       "actually run the spot simulation, going pixel-by-pixel over the region-of-interst")

      /* actual run of the spot simulation, restricted implementation plus OpenMP */
      .def("add_nanoBragg_spots_nks",&nanoBragg::add_nanoBragg_spots_nks,
       "actually run the spot simulation, going pixel-by-pixel over the region-of-interest, restricted options, plus OpenMP")

      .add_property("device_Id",
                     make_getter(&nanoBragg::device_Id,rbv()),
                     make_setter(&nanoBragg::device_Id,dcp()),
                     "Which GPU device to simulate on (only relevant for CUDA enabled builds). ")

#ifdef NANOBRAGG_HAVE_CUDA
      /* actual run of the spot simulation, CUDA version */
      .def("add_nanoBragg_spots_cuda",&nanoBragg::add_nanoBragg_spots_cuda,
       "actually run the spot simulation, going pixel-by-pixel over the region-of-interest, CUDA version")
#endif

    .def("set_dxtbx_detector_panel", &nanoBragg::set_dxtbx_detector_panel,
        (arg_("panel"), arg_("s0_vector")),
        "Updates the pixel array geometry using a dxtbx panel model (panel dimension should be conserved)")

      /* actual run of the background simulation */
      .def("add_background",&nanoBragg::add_background,
        (arg_("oversample")=-1,arg_("override_source")=-1),
       "run the non-Bragg simulation, adding background from speficied amorphous materials")

      /* retrieve radial-median filtered average background from the image */
      .def("extract_background",&nanoBragg::extract_background,
        (arg_("source")=-1),
       "retrieve radial-median filtered average background from the image, populates stol_vs_Fbg, given the flux and properties of speficied amorphous materials")

      /* blur the image with specified point-spread function */
      .def("apply_psf",&nanoBragg::apply_psf,
        (arg_("psf_type")=FIBER,arg_("fwhm_pixels")=-1,arg_("psf_radius")=-1),
       "blur the image with specified point-spread function, may be done before or after adding noise")

      /* actual run of the noise simulation */
      .def("add_noise",static_cast<void(nanoBragg::*)()>(&nanoBragg::add_noise),
       "apply specified Poisson, calibration, flicker and read-out noise to the pixels")

      .def("to_smv_format",&nanoBragg::to_smv_format,
        (arg_("fileout"),arg_("intfile_scale")=0,arg_("debug_x")=-1,arg_("debug_y")=-1),
        "write an SMV-format image file to disk from the raw pixel array\n"
        "intfile_scale: multiplicative factor applied to raw pixels before rounding off to integral pixel values\n"
        "     intfile_scale > 0 : specify a value for the multiplicative factor\n"
        "     intfile_scale = 1 : do not apply a factor\n"
        "     intfile_scale = 0 (default): compute a reasonable scale factor to set max pixel to 55000; same value given by get_intfile_scale()"
        )
      .def("get_intfile_scale",&nanoBragg::get_intfile_scale,
        (arg_("intfile_scale")=0),
        "expose the intfile_scale multiplier to raw pixels that is normally hidden within the to_smv_format interface.\n"
        "user can apply the return value to the output pixels using to_smv_format(<value>) or to_cbf(<value>)\n"
        "     intfile_scale = 0 (default): compute a reasonable scale factor to set max pixel to 55000"
        )
      .def("raw_pixels_unsigned_short_as_python_bytes",&raw_pixels_unsigned_short_as_python_bytes,
        (arg_("intfile_scale")=0,arg_("debug_x")=-1,arg("debug_y")=-1),
        "get the unsigned short raw pixels as a Python bytes object.  Intfile_scale is applied before rounding off to integral pixel values")
      .def("to_smv_format_streambuf",&nanoBragg::to_smv_format_streambuf,
        (arg_("output"),arg_("intfile_scale")=0,arg_("debug_x")=-1,arg("debug_y")=-1),
        "provide the integer buffer only to be used in Python for SMV-format output.  Intfile_scale is applied before rounding off to integral pixel values")
    ;
    // end of nanoBragg class definition


  } // end of init_module

} // end of namespace
} // end of namespace boost_python
} // end of namespace nanoBragg
} // end of namespace simtbx


BOOST_PYTHON_MODULE(simtbx_nanoBragg_ext)
{
  simtbx::nanoBragg::boost_python::init_module();
}
