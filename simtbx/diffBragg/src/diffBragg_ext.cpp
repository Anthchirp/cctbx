#include <cctbx/boost_python/flex_fwd.h>
#include <boost/python.hpp>
#include <simtbx/diffBragg/src/diffBragg.h>
#include <simtbx/nanoBragg/nanoBragg.h>
#include <iostream>
#include <boost/python/numpy.hpp>
#include <dlpack/dlpack.h>

using namespace boost::python;
namespace simtbx{
namespace diffBragg{
namespace boost_python { namespace {

  void (simtbx::nanoBragg::diffBragg::*add_diffBragg_spots_A)() = &simtbx::nanoBragg::diffBragg::add_diffBragg_spots;
  void (simtbx::nanoBragg::diffBragg::*add_diffBragg_spots_B)(const nanoBragg::af::shared<size_t>&) = &simtbx::nanoBragg::diffBragg::add_diffBragg_spots;
  void (simtbx::nanoBragg::diffBragg::*add_diffBragg_spots_C)(const nanoBragg::af::shared<size_t>&, boost::python::list per_pix_nominal_hkl)
        = &simtbx::nanoBragg::diffBragg::add_diffBragg_spots;

  boost::python::list get_Fhkl_grad_inds(simtbx::nanoBragg::diffBragg& diffBragg){
      boost::python::list indices;
      for(auto& fhkl_idx: diffBragg.db_cryst.Fhkl_grad_idx_tracker){
        indices.append(fhkl_idx);
      }
      return indices;
  }

  static void set_beams(simtbx::nanoBragg::diffBragg& diffBragg, scitbx::af::versa<dxtbx::model::Beam, scitbx::af::flex_grid<> > const& value) {
      if(diffBragg.verbose>3) printf(" about to initialize sources\n");
      diffBragg.pythony_beams = value;
      if(diffBragg.verbose>3) printf(" done\n");
      diffBragg.db_cu_flags.update_sources=true;
      /* re-initialize source table from pythony array */
      diffBragg.init_sources();
  }
  // TODO: point to the get_beams defined in simtbx/nanoBragg/nanoBragg_ext.cpp if possible..
  ///* table of sources, as dxtbx "beam"s */
  static scitbx::af::versa<dxtbx::model::Beam, scitbx::af::flex_grid<> > get_beams(simtbx::nanoBragg::diffBragg& diffBragg) {
      int i;
      /* allocate new flex array */
//      scitbx::af::versa<dxtbx::model::Beam, scitbx::af::flex_grid<> > diffBragg_pythony_beams;
      diffBragg.pythony_beams = scitbx::af::versa<dxtbx::model::Beam, scitbx::af::flex_grid<> >();
      /* make sure it is big enough to hold all sources */
      diffBragg.pythony_beams.resize(diffBragg.sources);

      /* polarization normal seems to be B vector */
      scitbx::vec3<double> Evector = scitbx::vec3<double>(diffBragg.polar_vector[1],diffBragg.polar_vector[2],diffBragg.polar_vector[3]);
      scitbx::vec3<double> Pvector = scitbx::vec3<double>(diffBragg.beam_vector[1],diffBragg.beam_vector[2],diffBragg.beam_vector[3]);
      scitbx::vec3<double> Bvector = Pvector.cross(Evector).normalize();

      /* copy internal storage into the flex array */
      for(i=0;i<diffBragg.sources;++i){
          diffBragg.pythony_beams[i].set_direction(scitbx::vec3<double>(diffBragg.source_X[i],diffBragg.source_Y[i],diffBragg.source_Z[i]));
          diffBragg.pythony_beams[i].set_wavelength(diffBragg.source_lambda[i]*1e10);
          diffBragg.pythony_beams[i].set_flux(diffBragg.source_I[i]);
          // how is this a fraction when it can be negative? (Kahn et al. 1982)
          diffBragg.pythony_beams[i].set_polarization_fraction(diffBragg.polarization);
          diffBragg.pythony_beams[i].set_polarization_normal(Bvector);
      }
      /* pass this back to python */
      return diffBragg.pythony_beams;
  }


  void set_dspace_bins(simtbx::nanoBragg::diffBragg& diffBragg, boost::python::list bins){
    diffBragg.db_cryst.dspace_bins.clear();
    for (int i=0; i< boost::python::len(bins); i++ ){
        double bin_edge = boost::python::extract<double>(bins[i]);
        diffBragg.db_cryst.dspace_bins.push_back(bin_edge);
    }
  }

  boost::python::list get_dspace_bins(simtbx::nanoBragg::diffBragg& diffBragg){
    boost::python::list bins;
    for (int i=0; i < diffBragg.db_cryst.dspace_bins.size(); i++){
        bins.append(diffBragg.db_cryst.dspace_bins[i]);
    }
    return bins;
  }

  void set_hall(simtbx::nanoBragg::diffBragg& diffBragg, boost::python::str hall){
    diffBragg.db_cryst.hall_symbol = boost::python::extract<std::string>(hall);
  }
  boost::python::str get_hall(simtbx::nanoBragg::diffBragg& diffBragg){
    return boost::python::str(diffBragg.db_cryst.hall_symbol);
  }

  int get_Num_ASU(simtbx::nanoBragg::diffBragg& diffBragg){
    return diffBragg.db_cryst.Num_ASU;
  }

  bool get_Fhkl_have_scale_factors(simtbx::nanoBragg::diffBragg& diffBragg){
    return diffBragg.db_flags.Fhkl_have_scale_factors;
  }

  boost::python::dict get_ASUid_map(simtbx::nanoBragg::diffBragg& diffBragg){
        boost::python::dict asu_info;
        for(auto &x: diffBragg.db_cryst.ASUid_map){
            asu_info[x.first] = x.second;
        }
        return asu_info;
  }

  static void  set_diffuse_gamma(simtbx::nanoBragg::diffBragg& diffBragg, boost::python::tuple const& values) {
      double g0 = boost::python::extract<double>(values[0]);
      double g1 = boost::python::extract<double>(values[1]);
      double g2 = boost::python::extract<double>(values[2]);
      diffBragg.db_cryst.anisoG <<  g0,0,0,
                                    0,g1,0,
                                    0,0,g2;
  }

  static boost::python::tuple get_diffuse_gamma(simtbx::nanoBragg::diffBragg const& diffBragg) {
      return boost::python::make_tuple(diffBragg.db_cryst.anisoG(0,0),
                                      diffBragg.db_cryst.anisoG(1,1),
                                      diffBragg.db_cryst.anisoG(2,2));
  }

  static void  set_diffuse_sigma(simtbx::nanoBragg::diffBragg& diffBragg, boost::python::tuple const& values) {
      double sig0 = boost::python::extract<double>(values[0]);
      double sig1 = boost::python::extract<double>(values[1]);
      double sig2 = boost::python::extract<double>(values[2]);
      double sig_array[3] = {sig0,sig1,sig2};

      diffBragg.db_cryst.anisoU <<  sig0*sig0,0,0,
                                    0,sig1*sig1,0,
                                    0,0,sig2*sig2;
      diffBragg.db_cryst.dU_dsigma.clear();
      diffBragg.db_cryst.dU_dsigma.resize(3);

      for (int i=0; i<3; i++){
        diffBragg.db_cryst.dU_dsigma[i] <<  0,0,0,
                                    0,0,0,
                                    0,0,0;
        diffBragg.db_cryst.dU_dsigma[i](i,i) = 2.*sig_array[i];
      }
  }

  static boost::python::tuple get_diffuse_sigma(simtbx::nanoBragg::diffBragg const& diffBragg) {
      double sig0 = sqrt(diffBragg.db_cryst.anisoU(0,0));
      double sig1 = sqrt(diffBragg.db_cryst.anisoU(1,1));
      double sig2 = sqrt(diffBragg.db_cryst.anisoU(2,2));
      return boost::python::make_tuple(sig0, sig1, sig2);
  }

  static boost::python::tuple get_rotate_principal_axes(simtbx::nanoBragg::diffBragg& diffBragg){
    double val0=diffBragg.db_cryst.rotate_principal_axes(0,0);
    double val1=diffBragg.db_cryst.rotate_principal_axes(0,1);
    double val2=diffBragg.db_cryst.rotate_principal_axes(0,2);
    double val3=diffBragg.db_cryst.rotate_principal_axes(1,0);
    double val4=diffBragg.db_cryst.rotate_principal_axes(1,1);
    double val5=diffBragg.db_cryst.rotate_principal_axes(1,2);
    double val6=diffBragg.db_cryst.rotate_principal_axes(2,0);
    double val7=diffBragg.db_cryst.rotate_principal_axes(2,1);
    double val8=diffBragg.db_cryst.rotate_principal_axes(2,2);
    return boost::python::make_tuple(val0,val1,val2,val3,val4,val5,val6,val7,val8);
  }

  void set_rotate_principal_axes(simtbx::nanoBragg::diffBragg& diffBragg, nanoBragg::mat3 const& value){
    diffBragg.db_cryst.rotate_principal_axes << value[0], value[1], value[2],
            value[3], value[4], value[5],
            value[6], value[7], value[8];
  }

  static void  set_Ndef(simtbx::nanoBragg::diffBragg& diffBragg, boost::python::tuple const& values) {
      diffBragg.set_ncells_def_values(values);
  }

  static boost::python::tuple get_Ndef(simtbx::nanoBragg::diffBragg const& diffBragg) {
      return boost::python::make_tuple(diffBragg.Nd, diffBragg.Ne, diffBragg.Nf);
  }

  static void  set_Nabc_aniso(simtbx::nanoBragg::diffBragg& diffBragg, boost::python::tuple const& values) {
      diffBragg.isotropic_ncells=false;
      diffBragg.set_ncells_values(values);
  }

  static boost::python::tuple get_Nabc_aniso(simtbx::nanoBragg::diffBragg const& diffBragg) {
      return boost::python::make_tuple(diffBragg.Na, diffBragg.Nb, diffBragg.Nc);
  }

  static void  set_Nabc(simtbx::nanoBragg::diffBragg& diffBragg, double const& value) {
      // NOTE this is the old way, assumming isotropic Nabc, for anisotropic use set_Nabc_aniso
      diffBragg.isotropic_ncells=true;
      diffBragg.set_value(9, value); //  9 means Ncells parameter
  }

  static double get_Nabc(simtbx::nanoBragg::diffBragg const& diffBragg) {
      return diffBragg.Na;
  }

  boost::python::tuple get_atom_data(simtbx::nanoBragg::diffBragg &diffBragg){
    boost::python::tuple values;
    values = boost::python::make_tuple(0,0);
    return values;
  }
  //TODO override the set_sources function (or xray_beams) property in nanoBragg
  // to set the fpfdp accordingly (if Fhkl2 is set)
  static void set_atom_data(simtbx::nanoBragg::diffBragg & diffBragg,
            boost::python::tuple const& atom_XYZBO){
            /// atom XYZBO is tuple of x coord, ycoords, z coords, B-factors, occupancy
            // e.g. 5 parameters per atom
   boost::python::list atom_x = boost::python::extract<boost::python::list>(atom_XYZBO[0]);
   boost::python::list atom_y = boost::python::extract<boost::python::list>(atom_XYZBO[1]);
   boost::python::list atom_z = boost::python::extract<boost::python::list>(atom_XYZBO[2]);
   boost::python::list atom_B = boost::python::extract<boost::python::list>(atom_XYZBO[3]);
   boost::python::list atom_O = boost::python::extract<boost::python::list>(atom_XYZBO[4]);
    int natoms = boost::python::len(atom_x);
    SCITBX_ASSERT( boost::python::len(atom_y)== natoms);
    SCITBX_ASSERT( boost::python::len(atom_z)== natoms);
    SCITBX_ASSERT( boost::python::len(atom_B)== natoms);
    SCITBX_ASSERT( boost::python::len(atom_O)== natoms);
    diffBragg.db_cryst.atom_data.clear();
    for (int i=0; i < natoms; i++){
        diffBragg.db_cryst.atom_data.push_back( boost::python::extract<double>(atom_x[i]));
        diffBragg.db_cryst.atom_data.push_back(  boost::python::extract<double>(atom_y[i]));
        diffBragg.db_cryst.atom_data.push_back(  boost::python::extract<double>(atom_z[i]));
        diffBragg.db_cryst.atom_data.push_back( boost::python::extract<double>(atom_B[i]));
        diffBragg.db_cryst.atom_data.push_back(  boost::python::extract<double>(atom_O[i]));
    }
  }

  boost::python::tuple get_fpfdp_derivs(simtbx::nanoBragg::diffBragg & diffBragg){
        boost::python::tuple vals;
        vals = boost::python::make_tuple(0,0);
        //TODO implement
        return vals;
  }

  static void set_fpfdp_derivs(simtbx::nanoBragg::diffBragg & diffBragg,
                boost::python::tuple const& derivs) {
    boost::python::list fprime_derivs = boost::python::extract<boost::python::list>(derivs[0]);
    boost::python::list fdblprime_derivs = boost::python::extract<boost::python::list>(derivs[1]);
    int num_fprime = boost::python::len(fprime_derivs);
    int num_fdblprime =   boost::python::len(fdblprime_derivs);
    SCITBX_ASSERT(num_fprime % diffBragg.sources==0);
    SCITBX_ASSERT(num_fdblprime % diffBragg.sources == 0 );
    diffBragg.db_cryst.fpfdp_derivs.clear();
    //diffBragg.fdp_derivs.clear();
    int n_deriv = num_fprime / diffBragg.sources;
    for (int i_deriv=0; i_deriv < n_deriv; i_deriv++ ){
        for (int i=0; i<diffBragg.sources; i++){
            int idx=diffBragg.sources*i_deriv + i;
            double val = boost::python::extract<double>(fprime_derivs[idx]);
            diffBragg.db_cryst.fpfdp_derivs.push_back(val);
            val = boost::python::extract<double>(fdblprime_derivs[idx]);
            diffBragg.db_cryst.fpfdp_derivs.push_back(val);
        }
    }
  }


  static void set_fpfdp(simtbx::nanoBragg::diffBragg & diffBragg,
                boost::python::tuple const& fprime_fdblprime) {
    boost::python::list fprime = boost::python::extract<boost::python::list>(fprime_fdblprime[0]);
    boost::python::list fdblprime = boost::python::extract<boost::python::list>(fprime_fdblprime[1]);
    int num_fprime = boost::python::len(fprime);
    int num_fdblprime =   boost::python::len(fdblprime);
    SCITBX_ASSERT(num_fprime==diffBragg.sources);
    SCITBX_ASSERT(num_fdblprime==diffBragg.sources);
    diffBragg.db_cryst.fpfdp.clear();
    for (int i=0; i<diffBragg.sources; i++){
        double val = boost::python::extract<double>(fprime[i]);
        diffBragg.db_cryst.fpfdp.push_back(val);
        val = boost::python::extract<double>(fdblprime[i]);
        diffBragg.db_cryst.fpfdp.push_back(val);
    }
  }

  boost::python::tuple get_fpfdp(simtbx::nanoBragg::diffBragg const& diffBragg){
    // TODO implement
    boost::python::list fp;
    boost::python::list fdp;
    boost::python::tuple values;
    int n = diffBragg.db_cryst.fpfdp.size()/2;
    for(int i=0; i < n; i++ ){
        double fp_val = diffBragg.db_cryst.fpfdp[2*i];
        double fdp_val = diffBragg.db_cryst.fpfdp[2*i+1];
        fp.append(fp_val);
        fdp.append(fdp_val);
    }
    values = boost::python::make_tuple(fp,fdp);
    return values;
  }

  // change of basis operator see dxtbx.model.crystal.h
  static nanoBragg::mat3 get_O(simtbx::nanoBragg::diffBragg& diffBragg){
    return diffBragg.Omatrix;
  }

  void set_O(simtbx::nanoBragg::diffBragg& diffBragg, nanoBragg::mat3 const& value){
    diffBragg.Omatrix = value;
    diffBragg.db_cryst.eig_O << value[0], value[1], value[2],
            value[3], value[4], value[5],
            value[6], value[7], value[8];
    //std::cout << "eigO: " << diffBragg.eig_O << std::endl;
  }

  static nanoBragg::mat3 get_U(simtbx::nanoBragg::diffBragg& diffBragg){
    return diffBragg.Umatrix;
  }

  void set_laue_group_num(simtbx::nanoBragg::diffBragg& diffBragg, int value){
    diffBragg.db_cryst.laue_group_num = value;
  }

  int get_laue_group_num(simtbx::nanoBragg::diffBragg& diffBragg){
    return diffBragg.db_cryst.laue_group_num;
  }

  void set_stencil_size(simtbx::nanoBragg::diffBragg& diffBragg, int value){
    diffBragg.db_cryst.stencil_size = value;
  }

  int get_stencil_size(simtbx::nanoBragg::diffBragg& diffBragg){
    return diffBragg.db_cryst.stencil_size;
  }

  void set_U(simtbx::nanoBragg::diffBragg& diffBragg, nanoBragg::mat3 const& value){
    diffBragg.Umatrix = value;
    diffBragg.db_cryst.eig_U << value[0], value[1], value[2],
            value[3], value[4], value[5],
            value[6], value[7], value[8];
  }

  static nanoBragg::mat3 get_B(simtbx::nanoBragg::diffBragg& diffBragg){
    return diffBragg.Bmatrix;
  }

  void set_B(simtbx::nanoBragg::diffBragg& diffBragg, nanoBragg::mat3 const& value){
      diffBragg.Bmatrix = value;
      diffBragg.a_star[1] = value(0,0);  // initialize Angstrom version, update at the end
      diffBragg.a_star[2] = value(1,0);
      diffBragg.a_star[3] = value(2,0);
      diffBragg.b_star[1] = value(0,1);
      diffBragg.b_star[2] = value(1,1);
      diffBragg.b_star[3] = value(2,1);
      diffBragg.c_star[1] = value(0,2);
      diffBragg.c_star[2] = value(1,2);
      diffBragg.c_star[3] = value(2,2);

      diffBragg.user_cell = 0;
      diffBragg.user_matrix = 1;
      diffBragg.init_cell();

       diffBragg.db_cryst.eig_B << diffBragg.a0[1], diffBragg.b0[1], diffBragg.c0[1],
            diffBragg.a0[2], diffBragg.b0[2], diffBragg.c0[2],
            diffBragg.a0[3], diffBragg.b0[3], diffBragg.c0[3];
  }

  //void set_roi(simtbx::nanoBragg::diffBragg& diffBragg, boost::python::tuple const& tupleoftuples){
  //    boost::python::tuple frange;
  //    boost::python::tuple srange;
  //    frange = extract<boost::python::tuple>(tupleoftuples[0]);
  //    srange = extract<boost::python::tuple>(tupleoftuples[1]);
  //    diffBragg.roi_xmin=extract<int>(frange[0]);
  //    diffBragg.roi_xmax=extract<int>(frange[1]);
  //    diffBragg.roi_ymin=extract<int>(srange[0]);
  //    diffBragg.roi_ymax=extract<int>(srange[1]);
  //     /* fix this up so its not out-of-range */
  //    diffBragg.init_beamcenter();
  //    diffBragg.init_raw_pixels_roi();
  //    diffBragg.initialize_managers();
  //}

  /* region of interest in pixels */
  //static boost::python::tuple get_roi(simtbx::nanoBragg::diffBragg const& diffBragg) {
  //    boost::python::tuple frange;
  //    boost::python::tuple srange;
  //    frange = boost::python::make_tuple(diffBragg.roi_xmin,diffBragg.roi_xmax);
  //    srange = boost::python::make_tuple(diffBragg.roi_ymin,diffBragg.roi_ymax);
  //    return boost::python::make_tuple(frange,srange);
  //}

  static boost::python::tuple get_reference_origin(simtbx::nanoBragg::diffBragg & diffBragg){
    return boost::python::make_tuple(diffBragg.O_reference[0]*1000, diffBragg.O_reference[1]*1000, diffBragg.O_reference[2]*1000);
  }

  //simtbx::nanoBragg::af::shared<double> get_deriv_pix(simtbx::nanoBragg::diffBragg& diffBragg, int refine_id){
  //  simtbx::nanoBragg::af::shared<double> pixels = diffBragg.get_derivative_pixels(refine_id);
  //  // slice them
  //  //return pixels[simtbx::nanoBragg::af::slice(0,diffBragg.Npix_to_model)];
  //  return pixels[0:diffBragg.Npix_to_model];
  //}

  void set_reference_origin(simtbx::nanoBragg::diffBragg & diffBragg, boost::python::tuple const& values){
    diffBragg.O_reference[0] = extract<double>(values[0])/1000.;
    diffBragg.O_reference[1] = extract<double>(values[1])/1000.;
    diffBragg.O_reference[2] = extract<double>(values[2])/1000.;
  }

  void set_lambda_coef(simtbx::nanoBragg::diffBragg& diffBragg, boost::python::tuple const& values){
      double coef0 = extract<double>(values[0]);
      double coef1 = extract<double>(values[1]);
      diffBragg.lambda_managers[0]->value = coef0;
      diffBragg.lambda_managers[1]->value = coef1;
      diffBragg.use_lambda_coefficients = true;
  }

  static boost::python::tuple get_lambda_coef(simtbx::nanoBragg::diffBragg const& diffBragg) {
      double coef0=diffBragg.lambda_managers[0]->value;
      double coef1=diffBragg.lambda_managers[1]->value;
      return boost::python::make_tuple(coef0, coef1);
  }

  bool get_use_diffuse(simtbx::nanoBragg::diffBragg& diffBragg){
      return diffBragg.db_flags.use_diffuse;
  }

  void set_use_diffuse(simtbx::nanoBragg::diffBragg& diffBragg, bool val){
      diffBragg.db_flags.use_diffuse = val ;
  }

  bool get_gamma_miller_units(simtbx::nanoBragg::diffBragg& diffBragg){
      return diffBragg.db_flags.gamma_miller_units;
  }

  void set_gamma_miller_units(simtbx::nanoBragg::diffBragg& diffBragg, bool val){
      diffBragg.db_flags.gamma_miller_units = val ;
  }

  bool get_only_diffuse(simtbx::nanoBragg::diffBragg& diffBragg){
      return diffBragg.db_flags.only_diffuse;
  }

  void set_only_diffuse(simtbx::nanoBragg::diffBragg& diffBragg, bool val){
      diffBragg.db_flags.only_diffuse = val ;
  }

  bool get_wavelen_img_flag(simtbx::nanoBragg::diffBragg& diffBragg){
    return diffBragg.db_flags.wavelength_img;
  }

  void set_wavelen_img_flag(simtbx::nanoBragg::diffBragg& diffBragg, bool val){
    diffBragg.db_flags.wavelength_img = val;
  }

  bool get_refine_Icell(simtbx::nanoBragg::diffBragg& diffBragg){
      return diffBragg.db_flags.refine_Icell;
  }

  void set_refine_Icell(simtbx::nanoBragg::diffBragg& diffBragg, bool val){
      diffBragg.db_flags.refine_Icell = val ;
  }

  static boost::python::tuple get_origin(simtbx::nanoBragg::diffBragg& diffBragg){
    return boost::python::make_tuple(diffBragg.pix0_vector[1]*1000,
                                    diffBragg.pix0_vector[2]*1000, diffBragg.pix0_vector[3]*1000);
  }

  static boost::python::list get_close_distances(simtbx::nanoBragg::diffBragg& diffBragg){
      boost::python::list detList;
      for (int pid=0; pid < diffBragg.db_det.close_distances.size(); pid++)
          detList.append( diffBragg.db_det.close_distances[pid]);
      return detList;
  }

  static void  set_Fhkl_tuple(simtbx::nanoBragg::diffBragg& diffBragg, boost::python::tuple const& value) {
      //TODO nanoBragg set as well ?
      diffBragg.db_cu_flags.update_Fhkl=true;
      diffBragg.pythony_indices = extract<nanoBragg::indices >(value[0]);
      diffBragg.pythony_amplitudes = extract<nanoBragg::af::shared<double> >(value[1]);
      diffBragg.init_Fhkl();
      diffBragg.complex_miller = false;
      if (boost::python::len(value)==3){
          if (value[2]!=boost::python::object()){ // check if it is not None
              diffBragg.pythony_amplitudes2 = extract<nanoBragg::af::shared<double> >(value[2]);
              diffBragg.init_Fhkl2();
              diffBragg.complex_miller = true;
          }
      }
      diffBragg.linearize_Fhkl(true);
  }

  static boost::python::tuple get_Fhkl_tuple(nanoBragg::diffBragg diffBragg) {
      int h,k,l;
      double temp;
      int hkls = diffBragg.h_range*diffBragg.k_range*diffBragg.l_range;
      diffBragg.pythony_indices = nanoBragg::indices(hkls,nanoBragg::af::init_functor_null<scitbx::vec3<int> >());
      diffBragg.pythony_amplitudes = nanoBragg::af::shared<double>(hkls,nanoBragg::af::init_functor_null<double>());
      diffBragg.pythony_amplitudes2 = nanoBragg::af::shared<double>(hkls,nanoBragg::af::init_functor_null<double>());
      int i=0;
      for(h=diffBragg.h_min;h<=diffBragg.h_max;++h){
          for(k=diffBragg.k_min;k<=diffBragg.k_max;++k){
              for(l=diffBragg.l_min;l<=diffBragg.l_max;++l){
                  if ( (h<=diffBragg.h_max) && (h>=diffBragg.h_min) && (k<=diffBragg.k_max)
                    && (k>=diffBragg.k_min) && (l<=diffBragg.l_max) && (l>=diffBragg.l_min)  ) {
                      /* populate all stored values */
                      nanoBragg::miller_t hkl (h,k,l);
                      diffBragg.pythony_indices[i] = hkl;
                      temp = diffBragg.Fhkl[h-diffBragg.h_min][k-diffBragg.k_min][l-diffBragg.l_min];
                      diffBragg.pythony_amplitudes[i] = temp;
                      if (diffBragg.complex_miller){
                          temp = diffBragg.Fhkl2[h-diffBragg.h_min][k-diffBragg.k_min][l-diffBragg.l_min];
                          diffBragg.pythony_amplitudes2[i] = temp;
                      }
                      ++i;
                  }
              }
          }
      }
      if (diffBragg.complex_miller)
        return boost::python::make_tuple(diffBragg.pythony_indices,diffBragg.pythony_amplitudes, diffBragg.pythony_amplitudes2);
      else
        return boost::python::make_tuple(diffBragg.pythony_indices,diffBragg.pythony_amplitudes);
  }

std::string kokkos_device() {
  std::string backend = "cpu:0";
#ifdef DIFFBRAGG_HAVE_KOKKOS
  if (Kokkos::is_finalized() || !Kokkos::is_initialized()) {
    return backend;
  }
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  backend = "cuda:" + std::to_string( Kokkos::device_id() );
#endif
#endif
  return backend;
}

#ifdef DIFFBRAGG_HAVE_KOKKOS
  void finalize_kokkos(){
    Kokkos::finalize();
  }

  void initialize_kokkos(int dev){
    Kokkos::initialize(Kokkos::InitializationSettings()
                           .set_device_id(dev));
  }

  void PrintDLTensorParameters( PyObject* capsule ) {
    auto tensor = static_cast<DLTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
    if (tensor == nullptr) {
        std::cerr << "The input DLTensor is null." << std::endl;
        return;
    }

    // Print the number of dimensions.
    std::cout << "Number of dimensions (ndim): " << tensor->ndim << std::endl;

    // Print the shape of the tensor.
    std::cout << "Shape: [";
    for (int i = 0; i < tensor->ndim; ++i) {
        std::cout << tensor->shape[i];
        if (i < tensor->ndim - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    // Print the data type of the tensor.
    std::cout << "Data type (dtype): ";
    switch (tensor->dtype.code) {
        case kDLInt: std::cout << "Int"; break;
        case kDLUInt: std::cout << "UInt"; break;
        case kDLFloat: std::cout << "Float"; break;
        case kDLBfloat: std::cout << "BFloat"; break;
        default: std::cout << "Unknown"; break;
    }
    std::cout << " (" << int(tensor->dtype.bits) << " bits, " << tensor->dtype.lanes << " lanes)" << std::endl;

    // Print the device context (device type and device id).
    std::cout << "Device context: ";
    switch (tensor->device.device_type) {
        case kDLCPU: std::cout << "CPU"; break;
        case kDLCUDA: std::cout << "CUDA"; break;
        case kDLCUDAHost: std::cout << "CUDA Host"; break;
        case kDLOpenCL: std::cout << "OpenCL"; break;
        case kDLVulkan: std::cout << "Vulkan"; break;
        case kDLMetal: std::cout << "Metal"; break;
        case kDLVPI: std::cout << "VPI"; break;
        case kDLROCM: std::cout << "ROCM"; break;
        default: std::cout << "Unknown"; break;
    }
    std::cout << ", Device ID: " << tensor->device.device_id << std::endl;

    // Print the strides of the tensor, if available.
    if (tensor->strides != nullptr) {
        std::cout << "Strides: [";
        for (int i = 0; i < tensor->ndim; ++i) {
            std::cout << tensor->strides[i];
            if (i < tensor->ndim - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    } else {
        std::cout << "Strides: [Contiguous in memory]" << std::endl;
    }

    // Print the byte offset of the tensor data.
    std::cout << "Byte Offset: " << tensor->byte_offset << std::endl;
}

void dlpack_destructor(PyObject* capsule) {
  if (!PyCapsule_IsValid(capsule, "dltensor")) {
    std::cout << "Muh0 " << PyCapsule_GetPointer(capsule, "used_dltensor") << std::endl;
    return;
  }

  // If the capsule has not been used, we need to delete it
  std::cout << "Muh1" << std::endl;
  DLManagedTensor* dlpackTensor = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
  std::cout << "Muh2" << std::endl;
  dlpackTensor->deleter(dlpackTensor);
  std::cout << "Muh3" << std::endl;
  delete dlpackTensor;
  std::cout << "Muh4" << std::endl;
}

struct DLPackAPI {

  double container[50];

  PyObject* dlpack() {
    // Get the Kokkos view size and dimensions
    size_t numDims = 1;
    int64_t* shape = new int64_t[numDims];
    for (size_t i = 0; i < numDims; i++) {
      shape[i] = 50;
    }

    // Create a DLPack tensor
    DLManagedTensor* dlpackTensor = new DLManagedTensor;
    dlpackTensor->dl_tensor.data = static_cast<void*>(&container);
    dlpackTensor->dl_tensor.device.device_type = DLDeviceType::kDLCPU;
    dlpackTensor->dl_tensor.device.device_id = 0;
    dlpackTensor->dl_tensor.ndim = numDims;
    dlpackTensor->dl_tensor.dtype = getDLPackDataType();
    dlpackTensor->dl_tensor.shape = shape;
    dlpackTensor->dl_tensor.strides = nullptr;
    dlpackTensor->dl_tensor.byte_offset = 0;
    dlpackTensor->manager_ctx = nullptr;
    dlpackTensor->deleter = [](DLManagedTensor* tensor) {
        std::cout << "Blob" << std::endl;
        delete[] tensor->dl_tensor.shape;
    };

    // Create a PyCapsule with the DLPack tensor
    PyObject* capsule = PyCapsule_New(dlpackTensor, "dltensor", dlpack_destructor);

    return capsule;
  }

  DLDataType getDLPackDataType() {
    DLDataType dtype;
    dtype.code = kDLFloat;
    dtype.bits = sizeof(double) * 8;
    dtype.lanes = 1;
    return dtype;
  }

  void print_hello() {
    std::cout << "Hello Python!" << std::endl;
  }

  void print_container() {
    std::cout << "C = [ ";
    for (int i=0; i<50; i++) {
      std::cout << container[i] << " ";
    }
    std::cout << "]" << std::endl;
  }

  boost::python::tuple dlpack_device() {
    return boost::python::make_tuple(static_cast<int32_t>(DLDeviceType::kDLCPU), 0);
  }

  };

#endif

  void diffBragg_init_module() {
    Py_Initialize();
    boost::python::numpy::initialize();

    using namespace boost::python;
    typedef return_value_policy<return_by_value> rbv;
    typedef default_call_policies dcp;
    typedef return_internal_reference<> rir;

#ifdef DIFFBRAGG_HAVE_KOKKOS
    def("finalize_kokkos", finalize_kokkos,
        "calls Kokkos::finalize()");

    def("initialize_kokkos", initialize_kokkos,
        "the sole argument `dev` (an int from 0 to Ngpu-1) is passed to Kokkos::initialize()");

    def("kokkos_device", kokkos_device, "returns kokkos device for use in PyTorch");

    def("print_dlpack",PrintDLTensorParameters,"Print information about a dlpack");

    // def("get_d_Ncells_images", &get_dlpack, "Return DLPackTensor for d_Ncells_images; pot. on GPU")

    class_<DLPackAPI>("KokkosViewToDLPack", no_init)
      .def(init("DLPack init"))
      .def("__dlpack__", &DLPackAPI::dlpack, "Part of DLPack API")
      .def("__dlpack_device__", &DLPackAPI::dlpack_device, "Part of DLPack API")
      .def("hello", &DLPackAPI::print_hello, "Dummy test function")
      .def("print", &DLPackAPI::print_container, "Print container")
    ;
#endif


    class_<simtbx::nanoBragg::diffBragg, bases<simtbx::nanoBragg::nanoBragg> >
            ("diffBragg", no_init)
      /* constructor that takes a dxtbx detector and beam model */
      .def(init<const dxtbx::model::Detector&,
                const dxtbx::model::Beam&,
                int>(
        (arg_("detector"),
         arg_("beam"),
         arg_("verbose")=0),
        "nanoBragg simulation initialized from dxtbx detector and beam objects"))

      /* function for doing some differentiation */
      //.def("add_diffBragg_spots",&simtbx::nanoBragg::diffBragg::add_diffBragg_spots,
      //  "gives derivative of average photon count w.r.t. parameter of choice")

      .def("add_diffBragg_spots", add_diffBragg_spots_A,
        "gives derivative of average photon count w.r.t. parameter of choice")

      .def("add_diffBragg_spots", add_diffBragg_spots_B,
        "gives derivative of average photon count w.r.t. parameter of choice")

      .def("add_diffBragg_spots", add_diffBragg_spots_C,
        "gives derivative of average photon count w.r.t. parameter of choice")

      /* Run this before running add_diffBragg_spots */
      .def("vectorize_umats",&simtbx::nanoBragg::diffBragg::vectorize_umats,
        "caches the UMATS")

      .def("initialize_managers",&simtbx::nanoBragg::diffBragg::initialize_managers,
        "sets up the managers")

      .def("refine", &simtbx::nanoBragg::diffBragg::refine,
        "sets the refinement flag to true")

      .def("fix", &simtbx::nanoBragg::diffBragg::fix,
        "sets the refinement flag to false")

      .def("let_loose", &simtbx::nanoBragg::diffBragg::let_loose,
        "sets the refinement flag to True(but doesnt initialize)")

      .def("get_value", &simtbx::nanoBragg::diffBragg::get_value, "get value of the refinement parameter")

      .def("set_value", &simtbx::nanoBragg::diffBragg::set_value, "set value of the refinement parameter")

      .def("set_ncells_values", &simtbx::nanoBragg::diffBragg::set_ncells_values, "set Ncells values as a 3-tuple (Na, Nb, Nc)")

      .def("get_ncells_values", &simtbx::nanoBragg::diffBragg::get_ncells_values, "get Ncells values as a 3-tuple (Na, Nb, Nc)")

      .def("add_diffBragg_spots_full", &simtbx::nanoBragg::diffBragg::add_diffBragg_spots_full, "forward model and gradients at every pixel")

      .def("get_ncells_derivative_pixels", &simtbx::nanoBragg::diffBragg::get_ncells_derivative_pixels, "get derivatives of intensity w.r.t (Na, Nb, Nc)")

      .def("get_diffuse_gamma_derivative_pixels", &simtbx::nanoBragg::diffBragg::get_diffuse_gamma_derivative_pixels, "get derivatives of intensity w.r.t. diffuse (gamma_a, gamma_b, gamma_c)")

      .def("get_diffuse_sigma_derivative_pixels", &simtbx::nanoBragg::diffBragg::get_diffuse_sigma_derivative_pixels, "get derivatives of intensity w.r.t. diffuse (sigma_a, sigma_b, sigma_c)")

      .def("get_fp_fdp_derivative_pixels", &simtbx::nanoBragg::diffBragg::get_fp_fdp_derivative_pixels, "get derivatives of intensity w.r.t c,d that describe fprime and fdblprime (see diffBragg.utils)")

      .def("get_ncells_def_derivative_pixels", &simtbx::nanoBragg::diffBragg::get_ncells_def_derivative_pixels, "get derivatives of intensity w.r.t (Nd, Ne, Nf)")

      //.def("get_Na_derivative_pixels", &simtbx::nanoBragg::diffBragg::get_Na_derivative_pixels, "get derivatives of intensity w.r.t (Nd, Ne, Nf)")

      .def("get_aniso_eta_deriv_pixels", &simtbx::nanoBragg::diffBragg::get_aniso_eta_deriv_pixels, "get derivatives of intensity w.r.t anisotropic mosaicity model")

      .def("get_aniso_eta_second_deriv_pixels", &simtbx::nanoBragg::diffBragg::get_aniso_eta_second_deriv_pixels, "get second derivatives of intensity w.r.t anisotropic mosaicity model")

      .def("get_ncells_def_second_derivative_pixels", &simtbx::nanoBragg::diffBragg::get_ncells_def_second_derivative_pixels, "get 2nd derivatives of intensity w.r.t (Nd, Ne, Nf)")

      .def("get_ncells_second_derivative_pixels", &simtbx::nanoBragg::diffBragg::get_ncells_second_derivative_pixels, "get second derivatives of intensity w.r.t (Na, Nb, Nc)")

      .def("get_lambda_derivative_pixels", &simtbx::nanoBragg::diffBragg::get_lambda_derivative_pixels, "get derivatives of intensity w.r.t (lambda0, lambda1) as a 2-tuple")

      .def("print_if_refining", &simtbx::nanoBragg::diffBragg::print_if_refining, "print a statement if the refiner flag is set")

      .def("quick_Fhkl_update", &simtbx::nanoBragg::diffBragg::quick_Fcell_update, "update Fcell amplitudes without reallocating memory")
      .def("update_xray_beams", &simtbx::nanoBragg::diffBragg::update_xray_beams, "update xray beams without reallocating memory")

      .def("set_ucell_derivative_matrix",  &simtbx::nanoBragg::diffBragg::set_ucell_derivative_matrix, "Boo-ya")

      .def("set_ucell_second_derivative_matrix",  &simtbx::nanoBragg::diffBragg::set_ucell_second_derivative_matrix,
            "Two-ya")

      //.def("get_ucell_derivative_matrix",  &simtbx::nanoBragg::diffBragg::get_ucell_derivative_matrix, "scooby snacks")

      //.def("get_derivative_pixels", get_deriv_pix,
      //      "gets the manager raw image containing first derivatives")

      .def("__add_Fhkl_gradients", &simtbx::nanoBragg::diffBragg::add_Fhkl_gradients,
            "special mode for computing the gradients of the structure factors. Takes psf, residual, variance, and trusted mask as arguments")

      .def("__update_Fhkl_scale_factors", &simtbx::nanoBragg::diffBragg::update_Fhkl_scale_factors,
            "updates the scale factors for each ASU. Should be same length as the db_cryst.ADUid_map")

      .def("__update_Fhkl_channels", &simtbx::nanoBragg::diffBragg::update_Fhkl_channels,
            "pass this a numpy int array the same length as the number of energy sources, this specifies the mapping of structure factor to energy channel, allowing one to refine multiple energy-dependent structure factors for example in a two-color experiment")

      .def("get_Fhkl_channels", &simtbx::nanoBragg::diffBragg::get_Fhkl_channels,
            "get current list of Fhkl channels (if set, it will be the same length as number of sources)")

      .def("__get_derivative_pixels", &simtbx::nanoBragg::diffBragg::get_derivative_pixels,
            "gets the manager raw image containing first derivatives")

      .def("get_second_derivative_pixels", &simtbx::nanoBragg::diffBragg::get_second_derivative_pixels,
            "gets the manager raw image containing second derivatives")

      .def("zero_raw_pixel_rois", &simtbx::nanoBragg::diffBragg::zero_raw_pixel_rois,
           "reallocate the raw image ROIs (they are usually tiny)")

      .def("get_origin", get_origin, "get panel origin")

      .def("shift_origin_z",
            &simtbx::nanoBragg::diffBragg::shift_originZ,
            "shift the Z-component of the pixel0 vector in every detector panel")

      .def("update_dxtbx_geoms",
            &simtbx::nanoBragg::diffBragg::update_dxtbx_geoms,
             (arg_("detector"),
              arg_("beam"),
              arg_("panel_id")=0,
              arg_("panel_rot_angO")=0,
              arg_("panel_rot_angF")=0,
              arg_("panel_rot_angS")=0,
              arg_("panel_offsetX")=0,
              arg_("panel_offsetY")=0,
              arg_("panel_offsetZ")=0, arg_("force")=true),
           "update the geometries with new dxtbx models, number of pixels should remain constant")

      .def("free_Fhkl2",&simtbx::nanoBragg::diffBragg::free_Fhkl2)

      .def("show_fp_fdp", &simtbx::nanoBragg::diffBragg::show_fp_fdp)

      .def("show_heavy_atom_data", &simtbx::nanoBragg::diffBragg::show_heavy_atom_data)

      .def("gpu_free",&simtbx::nanoBragg::diffBragg::gpu_free)

#ifdef DIFFBRAGG_HAVE_CUDA
      .def("gpu_free",&simtbx::nanoBragg::diffBragg::cuda_free)
#endif

#ifdef DIFFBRAGG_HAVE_KOKKOS
      .def("kokkos_free",&simtbx::nanoBragg::diffBragg::kokkos_free)
#endif

      .def("set_mosaic_blocks_prime",
             &simtbx::nanoBragg::diffBragg::set_mosaic_blocks_prime,
             "enter a list of unitary matrix derivatives Uprime")

      .def("set_mosaic_blocks_dbl_prime",
             &simtbx::nanoBragg::diffBragg::set_mosaic_blocks_dbl_prime,
             "enter a list of unitary matrix 2nd derivatives Uprime")

      .def("show_timings",
           &simtbx::nanoBragg::diffBragg::show_timing_stats,
           (arg_("MPI_RANK")=0),
           "Shows a few key accumulated timings (in milliseconds) of specific code portions")

      //.add_property("region_of_interest",
      //       make_function(&get_roi,rbv()),
      //       make_function(&set_roi,dcp()),
      //       "region of interest on detector: fast_min fast_max slow_min slow_max")

      .add_property("close_distances",
             make_function(&get_close_distances,rbv()))
             //"get the effective detector distance for each panel")

      .add_property("raw_pixels_roi",
                     make_getter(&simtbx::nanoBragg::diffBragg::raw_pixels_roi,rbv()),
                     make_setter(&simtbx::nanoBragg::diffBragg::raw_pixels_roi,dcp()),
                    "raw pixels from region of interest only")

      .add_property("oversample_omega",
                     make_getter(&simtbx::nanoBragg::diffBragg::oversample_omega,rbv()),
                     make_setter(&simtbx::nanoBragg::diffBragg::oversample_omega,dcp()),
                    "whether to use an average solid angle correction per pixel, or one at the sub pixel level")

      .add_property("host_transfer",
                     make_getter(&simtbx::nanoBragg::diffBragg::host_transfer,rbv()),
                     make_setter(&simtbx::nanoBragg::diffBragg::host_transfer,dcp()),
                    "whether to transfer results from device to host")

      .add_property("force_cpu",
                     make_getter(&simtbx::nanoBragg::diffBragg::force_cpu,rbv()),
                     make_setter(&simtbx::nanoBragg::diffBragg::force_cpu,dcp()),
                    "force use of the CPU kernel, for example, if the environ var DIFFBRAGG_USE_CUDA is set")

      .add_property("track_Fhkl",
                     make_getter(&simtbx::nanoBragg::diffBragg::track_Fhkl,rbv()),
                     make_setter(&simtbx::nanoBragg::diffBragg::track_Fhkl,dcp()),
                    "whether to track Fhkl per pixel")

      .add_property("fprime_fdblprime",
             make_function(&get_fpfdp,rbv()),
             make_function(&set_fpfdp,dcp()),
             "fprime and fdoubleprime, passed as two lists whose length should be the same")

      .add_property("fprime_fdblprime_derivs",
             make_function(&get_fpfdp_derivs,rbv()), // TODO implement me
             make_function(&set_fpfdp_derivs,dcp()),
             "fprime and fdoubleprime derivatives, as a 2-tuple. e.g. 1st tuple element is fprime derivs, length should be multiple of number of sources")

      .add_property("heavy_atom_data",
             make_function(&get_atom_data,rbv()),
             make_function(&set_atom_data,dcp()),
             "5 tuple of fracx_list,fracy_list,fracz_list,Bfactor_list,Occupancy_list, each list is same length")

      .add_property("no_Nabc_scale",
                     make_getter(&simtbx::nanoBragg::diffBragg::no_Nabc_scale,rbv()),
                     make_setter(&simtbx::nanoBragg::diffBragg::no_Nabc_scale,dcp()),
                    "toggle off the Nabc scale factor in the forward model (such that it is replaced entirely by spot_scale)")

      .add_property("__number_of_pixels_modeled_using_diffBragg", // protect this by making it a long name
                     make_getter(&simtbx::nanoBragg::diffBragg::Npix_to_model,rbv()),
                     make_setter(&simtbx::nanoBragg::diffBragg::Npix_to_model,dcp()),
                    "num pix modeled most recently")

      .add_property("compute_curvatures",
                     make_getter(&simtbx::nanoBragg::diffBragg::compute_curvatures,rbv()),
                     make_setter(&simtbx::nanoBragg::diffBragg::compute_curvatures,dcp()),
                    "Whether to compute the curvatures")

      .add_property("only_save_omega_kahn",
                     make_getter(&simtbx::nanoBragg::diffBragg::only_save_omega_kahn,rbv()),
                     make_setter(&simtbx::nanoBragg::diffBragg::only_save_omega_kahn,dcp()),
                    "ONLY simulate the kahn polarization correction and solid angle components")

      .add_property("reference_origin",
            make_function(&get_reference_origin, rbv()),
            make_function(&set_reference_origin, dcp()),
            "reference origin for doing panel rotations for a group of panels")

      .add_property("Umatrix",
             make_function(&get_U,rbv()),
             make_function(&set_U,dcp()),
             "rotation portion of Amatrix (dxtbx format, C.get_U())")

      .add_property("laue_group_num",
             make_function(&get_laue_group_num,rbv()),
             make_function(&set_laue_group_num,dcp()),
             "relevant for diffuse model only. an integer (1-13) specifying the laue group number to assign transformation operators")

      .add_property("stencil_size",
             make_function(&get_stencil_size,rbv()),
             make_function(&set_stencil_size,dcp()),
             "relevant for diffuse model only. an integer specifying the number of additional reflections to include along each direction")

      .add_property("Bmatrix",
             make_function(&get_B,rbv()),
             make_function(&set_B,dcp()),
             "unit cell portion of Amatrix (dxtbx format; C.get_B())")

      .add_property("Omatrix",
             make_function(&get_O,rbv()),
             make_function(&set_O,dcp()),
             "change of basis Omatrix")

      .add_property("Ncells_abc",
             make_function(&get_Nabc,rbv()),
             make_function(&set_Nabc,dcp()),
             "number of mosaic domains along an axis")

      .add_property("Ncells_def",
             make_function(&get_Ndef,rbv()),
             make_function(&set_Ndef,dcp()),
             "off-axis components in mosaic domain size matrix")

      .add_property("Ncells_abc_aniso",
             make_function(&get_Nabc_aniso,rbv()),
             make_function(&set_Nabc_aniso,dcp()),
             "number of mosaic domains along an axis, can be anisotropic")

      .add_property("update_oversample_during_refinement",
                     make_getter(&simtbx::nanoBragg::diffBragg::update_oversample_during_refinement,rbv()),
                     make_setter(&simtbx::nanoBragg::diffBragg::update_oversample_during_refinement,dcp()),
                     "Allows oversample to change as Ncells abc changes")

      .add_property("isotropic_ncells",
                     make_getter(&simtbx::nanoBragg::diffBragg::isotropic_ncells,rbv()),
                     make_setter(&simtbx::nanoBragg::diffBragg::isotropic_ncells,dcp()),
                     "refine (Na, Nb, Nc) as three independent parameters")

      .add_property("has_anisotropic_mosaic_spread",
                     make_getter(&simtbx::nanoBragg::diffBragg::modeling_anisotropic_mosaic_spread,rbv()),
                     make_setter(&simtbx::nanoBragg::diffBragg::modeling_anisotropic_mosaic_spread,dcp()),
                     "compute derivatives for a single mosaic spread parameter")

      .add_property("use_lambda_coefficients",
                     make_getter(&simtbx::nanoBragg::diffBragg::use_lambda_coefficients,rbv()),
                     make_setter(&simtbx::nanoBragg::diffBragg::use_lambda_coefficients,dcp()),
                     "represent wavelength using coefficients a + b*source where a,b are coefficients and source is the source index")

      .add_property("Fhkl_tuple",
            make_function(&get_Fhkl_tuple,rbv()),
            make_function(&set_Fhkl_tuple,dcp()),
            "hkl and Freal and Fcomplex as a 3-tuple of (indices ,flex-double, flex-double). experimental, also accepts (indices, flex-double, None) for mod only")

      .add_property("lambda_coefficients",
             make_function(&get_lambda_coef,rbv()),
             make_function(&set_lambda_coef,dcp()),
             "coefficients for source_lambda refinement: `lambda = coef0 + coef1*source`  where `source` is the source index")

     // CUDA PROPERTIES
      .add_property("use_gpu",
             make_getter(&simtbx::nanoBragg::diffBragg::use_gpu,rbv()),
             make_setter(&simtbx::nanoBragg::diffBragg::use_gpu,dcp()),
             "use GPU acceleration")

      .add_property("record_time",
             make_getter(&simtbx::nanoBragg::diffBragg::record_time,rbv()),
             make_setter(&simtbx::nanoBragg::diffBragg::record_time,dcp()),
             "store timing data for key aspects of the code")

      .add_property("Npix_to_allocate",
            make_getter(&simtbx::nanoBragg::diffBragg::Npix_to_allocate,rbv()),
            make_setter(&simtbx::nanoBragg::diffBragg::Npix_to_allocate,dcp()),
            "number of pixels per parameter to allocate in the GPU (default value of -1 means auto-determine)")

      .add_property("use_diffuse",
            make_function(&get_use_diffuse,rbv()),
            make_function(&set_use_diffuse,dcp()),
            "sim with diffuse")

      .add_property("gamma_miller_units",
            make_function(&get_gamma_miller_units,rbv()),
            make_function(&set_gamma_miller_units,dcp()),
            "use Miller index units for gamma parameters")

      .add_property("only_diffuse",
            make_function(&get_only_diffuse,rbv()),
            make_function(&set_only_diffuse,dcp()),
            "sim with ONLY diffuse")

      .add_property("refine_Icell",
            make_function(&get_refine_Icell,rbv()),
            make_function(&set_refine_Icell,dcp()),
            "Assume whatever is stored in the structure factor array is an intensity (allowed to be negative)")

      .add_property("diffuse_gamma",
            make_function(&get_diffuse_gamma,rbv()),
            make_function(&set_diffuse_gamma,dcp()),
            "the 3 diffuse gamma factors")

      .add_property("diffuse_sigma",
            make_function(&get_diffuse_sigma,rbv()),
            make_function(&set_diffuse_sigma,dcp()),
            "set the 3 diffuse sigma factors")

      .add_property("_ext_rotate_principal_axes",
             make_function(&get_rotate_principal_axes,rbv()),
             make_function(&set_rotate_principal_axes,dcp()),
             "sets the matrix that rotates the principal axes of the anisotropic diffuse model")

      .add_property("most_recent_kernel_used_GPU",
            make_getter(&simtbx::nanoBragg::diffBragg::last_kernel_on_GPU,rbv()),
            "if True, the most recent kernel call utilized a GPU")
      .add_property("store_ave_wavelength_image",
            make_function(&get_wavelen_img_flag,rbv()),
            make_function(&set_wavelen_img_flag,dcp()),
            "if True, then record the average wavelength per pixel, weighted by Bragg intensity")
      .def("get_ASUid_map",
            &get_ASUid_map,
            "an internal map that specifies the ASU miller index for each entry in FhklLinear")
      .add_property("Num_ASU",
                     make_function(get_Num_ASU,rbv()),
                    "number of unique ASU miller indices")

      .add_property("Fhkl_have_scale_factors",
                     make_function(get_Fhkl_have_scale_factors,rbv()),
                    "boolean flag indicating whether the Fhkl have scale factors initialized (for doing Fhkl refinement)")

      .add_property("Fhkl_gradient_indices",
                     make_function(get_Fhkl_grad_inds,rbv()),
                    "return a list of indices")
      .add_property("hall_symbol",
            make_function(&get_hall,rbv()),
            make_function(&set_hall,dcp()),
            "an internal map that specifies the ASU miller index for each entry in FhklLinear")

      .add_property("dspace_bins",
            make_function(&get_dspace_bins,rbv()),
            make_function(&set_dspace_bins,dcp()),
            "set the bins for computing the average structure factor per resolution")

      .def("_ave_I_cell",
            &simtbx::nanoBragg::diffBragg::get_ave_I_cell,
            "return python list of average I_cell (provided dspace_bins was set)")

      .def("_Fhkl_restraint_data",
            &simtbx::nanoBragg::diffBragg::Fhkl_restraint_data,
            "return numpy array of Fhkl restraint gradients and target contribution")

      .def("ave_wavelength_image",
            &simtbx::nanoBragg::diffBragg::ave_wavelength_img,
            "return flex array containing average wavelen per pixel")

      .def("ave_hkl_image",
            &simtbx::nanoBragg::diffBragg::ave_hkl_img,
            "return 3-tuple of flex arrays each containing average h,k,l per pixel, respectively")

      .def("_set_Friedel_mate_inds",
            &simtbx::nanoBragg::diffBragg::set_Friedel_mate_inds,
            "Two arguments; each lists of the same length, pointing to the positive and negative mates in a Friedel pair, respectively")

      .def("get_mosaic_blocks_prime",
           &simtbx::nanoBragg::diffBragg::get_mosaic_blocks_prime,
           "return the deriv of the matrices U that define the mosaic block distribution w.r.t eta")

      .add_property("xray_beams",
                    make_function(&get_beams,rbv()),
                    make_function(&set_beams,dcp()),
                    "list of dxtbx::Beam objects corresponding to each zero-divergence and monochromatic x-ray point source in the numerical simulation ")

#ifdef DIFFBRAGG_HAVE_KOKKOS
      .def("get_floatimage", &simtbx::nanoBragg::diffBragg::get_floatimage, "get DLPackTensor for floatimage; pot. on GPU")

      .def("get_wavelenimage", &simtbx::nanoBragg::diffBragg::get_wavelenimage, "get DLPackTensor for wavelenimage; pot. on GPU")

      .def("get_d_diffuse_gamma_images", &simtbx::nanoBragg::diffBragg::get_d_diffuse_gamma_images, "get DLPackTensor for d_diffuse_gamma_images; pot. on GPU")

      .def("get_d_diffuse_sigma_images", &simtbx::nanoBragg::diffBragg::get_d_diffuse_sigma_images, "get DLPackTensor for d_diffuse_sigma_images; pot. on GPU")

      .def("get_d_Umat_images", &simtbx::nanoBragg::diffBragg::get_d_Umat_images, "get DLPackTensor for d_Umat_images; pot. on GPU")

      .def("get_d2_Umat_images", &simtbx::nanoBragg::diffBragg::get_d2_Umat_images, "get DLPackTensor for d2_Umat_images; pot. on GPU")

      .def("get_d_Bmat_images", &simtbx::nanoBragg::diffBragg::get_d_Bmat_images, "get DLPackTensor for d_Bmat_images; pot. on GPU")

      .def("get_d2_Bmat_images", &simtbx::nanoBragg::diffBragg::get_d2_Bmat_images, "get DLPackTensor for d2_Bmat_images; pot. on GPU")

      .def("get_d_Ncells_images", &simtbx::nanoBragg::diffBragg::get_d_Ncells_images, "get DLPackTensor for d_Ncells_images; pot. on GPU")

      .def("get_d2_Ncells_images", &simtbx::nanoBragg::diffBragg::get_d2_Ncells_images, "get DLPackTensor for d2_Ncells_images; pot. on GPU")

      .def("get_d_fcell_images", &simtbx::nanoBragg::diffBragg::get_d_fcell_images, "get DLPackTensor for d_fcell_images; pot. on GPU")

      .def("get_d2_fcell_images", &simtbx::nanoBragg::diffBragg::get_d2_fcell_images, "get DLPackTensor for d2_fcell_images; pot. on GPU")

      .def("get_d_eta_images", &simtbx::nanoBragg::diffBragg::get_d_eta_images, "get DLPackTensor for d_eta_images; pot. on GPU")

      .def("get_d2_eta_images", &simtbx::nanoBragg::diffBragg::get_d2_eta_images, "get DLPackTensor for d2_eta_images; pot. on GPU")

      .def("get_d_lambda_images", &simtbx::nanoBragg::diffBragg::get_d_lambda_images, "get DLPackTensor for d_lambda_images; pot. on GPU")

      .def("get_d2_lambda_images", &simtbx::nanoBragg::diffBragg::get_d2_lambda_images, "get DLPackTensor for d2_lambda_images; pot. on GPU")

      .def("get_d_panel_rot_images", &simtbx::nanoBragg::diffBragg::get_d_panel_rot_images, "get DLPackTensor for d_panel_rot_images; pot. on GPU")

      .def("get_d2_panel_rot_images", &simtbx::nanoBragg::diffBragg::get_d2_panel_rot_images, "get DLPackTensor for d2_panel_rot_images; pot. on GPU")

      .def("get_d_panel_orig_images", &simtbx::nanoBragg::diffBragg::get_d_panel_orig_images, "get DLPackTensor for d_panel_orig_images; pot. on GPU")

      .def("get_d2_panel_orig_images", &simtbx::nanoBragg::diffBragg::get_d2_panel_orig_images, "get DLPackTensor for d2_panel_orig_images; pot. on GPU")

      .def("get_d_fp_fdp_images", &simtbx::nanoBragg::diffBragg::get_d_fp_fdp_images, "get DLPackTensor for d_fp_fdp_images; pot. on GPU")

      .def("get_Fhkl_scale_deriv", &simtbx::nanoBragg::diffBragg::get_Fhkl_scale_deriv, "get DLPackTensor for Fhkl_scale_deriv; pot. on GPU")
#endif

    ; // end of diffBragg extention

  } // end of diffBragg_init_module
} // end of namespace
} // end of boost python namespace
} // end of refine
} // end of simtbx

BOOST_PYTHON_MODULE(simtbx_diffBragg_ext)
{
  simtbx::diffBragg::boost_python::diffBragg_init_module();
}
