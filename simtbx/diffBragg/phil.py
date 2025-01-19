from __future__ import absolute_import, division, print_function
from iotbx.phil import parse

#help_message = '''
#diffBragg command line utility
#'''

hopper_phil = """

filter_during_refinement {
  enable = False
    .type = bool
    .help = if True, filtering will occur each N iterations, controlled by parameter after_n
  after_n = 50
    .type = int
    .help = refiner will pause and check for outliers every after_n iterations
  threshold = 20
    .type = float
    .help = outliers are detected by looking at the distribution of per shoebox sigmaZ
    .help = and then using a median absolute deviation filter. Lower values of threshold will flag more pixels as outliers
}

filter_after_refinement {
  enable = False
    .type = bool
    .help = if True, filter, then rerun refinement if certain conditions are met (e.g. too few refinement iterations)
  max_attempts = 2
    .type = int
    .help = how many additional times to run hopper
  min_prev_niter = 50
    .type = int
    .help = only repeat if the previous refinement was fewer than this many iterations
  max_prev_sigz = 10
    .type = float
    .help = only repeat if the previous refinement had sigma Z more than this
  threshold = 20
    .type = float
    .help = outliers are detected by looking at the distribution of per shoebox sigmaZ
    .help = and then using a median absolute deviation filter. Lower values of threshold will flag more pixels as outliers
}

symmetrize_Flatt = False
  .type = bool
  .help = If True, add 3-fold symmetric mosaic blocks to the calculation of F_latt
record_device_timings = False
  .type = bool
  .help = Record the execution times of diffBragg host-dev copies and kernel executions
  .help = the results will be printed to the terminal
consider_multicrystal_shots = False
  .type = bool
  .help = If True, and if there are multiple crystals in the experiment list,
  .help = then try to model all crystals for a given shot.
debug_mode = False
  .type = bool
  .help = If True, many output files are written to explore the diffBragg models in great detail
nominal_Fhkl_only = True
  .type = bool
  .help = if refining Fhkls, only refine the ones that are assigned to a reflection table...
use_geometric_mean_Fhkl = False
  .type = bool
  .help = whether to use the geometric mean for Fhkl restraint (when betasFhkl is not None, restratin Fhkl to the mean in each res bin)
Fhkl_channel_bounds = None
  .type = floats
  .help = Energy bounds for energy-dependent structure factors. Units are eV.
  .help = If provided refine a unique structure factor correction for each bin defined by Fhkl_channel_bins
  .help = 0 and infinity are implicit. Providing a single number, e.g. 8950, will refine two sets of structure
  .help = factors: one for energies [0-8950), and another for energies [8950-infinity]
Fhkl_dspace_bins = 10
  .type = int
  .help = number of resolution bins (out to corner of detector) for computing the average structure factor intensity
  .help = One can then restrain to these values when fix.Fhkl=False by using the restraint strength betas.Fhkl (default is None in which case no restraints are applied)
try_strong_mask_only = False
  .type = bool
  .help = if strong spot masks are present in the input refls, then use them
  .help = as the trusted-flags, i.e. only run strong spot pixels through diffBragg
dilate_strong_mask = None
  .type = int
  .help = if using strong mask only for refinement, then dilate it this many iterations using
  .help = scipy.ndimage method binary_dilation (has to be >= 1)
hopper_save_freq = None
  .type = int
  .help = save the output files when the iteration number is a multiple of this argument
terminate_after_n_converged_iter = None
  .type = int
  .help = optionally converge if all parameters seem converged for this many iterations. See
  .help = converged_param_percent_change
converged_param_percent_change = 0.1
  .type = float
  .help = if a parameter's value changes by less than this amount (percent) between iterations, for a
  .help = a ceterain number of iters (determined by terminate_after_n_converged_iter), then
  .help = optimization is considered done, and is stopped
mask_highest_values = None
  .type = int
  .help = mask out the N highest-valued pixels in a shoebox when performing diffBragg refinement
use_float32 = False
  .type = bool
  .help = store pixel data and background models in 32bit arrays
  .expert_level=10
test_gathered_file = False
  .type = bool
  .help = run a quick test to ensure the gathered data file preserves information
  .expert_level=10
load_data_from_refls = False
  .type = bool
  .help = load image data, background etc from reflection tables
  .expert_level=10
gathered_output_file = None
  .type = str
  .help = optional file for storing a new hopper input file which points to the gathered data dumps
  .expert_level=10
only_dump_gathers = False
  .type = bool
  .help = only reads in image data, fits background planes, and dumps
  .help = results to disk, writes a new exper refl file at the end
  .expert_level=10
gathers_dir = None
  .type = str
  .help = folder where gathered data reflection tables
  .help = will be writen (if dump_gathers=True)
  .expert_level=10
dump_gathers = False
  .type = bool
  .help = optionally dump the loaded experimental data to reflection tables
  .help = for portability
  .expert_level=10
spectrum_from_imageset = False
  .type = bool
  .help = if True, load the spectrum from the imageset in the experiment, then probably downsample it
  .expert_level=0
gen_gauss_spec = False
  .type = bool
  .help = If the experimental data dont include spectra, one can try generating gaussian spectra.
  .help = See the diffBragg/phil.py under simulator.spectrum.gauss_spec.
use_perpixel_dark_rms = False
  .type = bool
  .help = some Jungfrau formats have per-pixel RMS values that change shot-to-shot with the dynamic
  .help = gain mode switching. If this flag is true, then the per-pixel gain modes will be extracted from the image
  .help = format . The image format class is expected to have a method named  get_pedestal_rms(self, index=None)
  .help = See the method xfel/util/jungfrau/get_pedestalRMS-from_jungfrau
isotropic {
  diffuse_gamma = False
    .type = bool
    .help = refine a single diffuse gamma parameter as opposed to 3
  diffuse_sigma = False
    .type = bool
    .help = refine a single diffuse gamma parameter as opposed to 3
}
downsamp_spec {
  skip = False
    .type = bool
    .help = if reading spectra from imageset, optionally skip the downsample portion
    .help = Note, if skip=True, then total flux will be determined by whats in the imageset spectrum (sum of the weights)
    .expert_level=10
  filt_freq = 0.07
    .type = float
    .help = low pass filter frequency in units of inverse spectrometer pixels (??)
    .expert_level=10
  filt_order = 3
    .type = int
    .help = order for bandpass butter filter
    .expert_level=10
  tail = 50
    .type = int
    .help = endpoints of the spectrum that are used in background estimation
    .expert_level=10
  delta_en = 0.5
    .type = float
    .help = final resolution of downsampled spectrum in eV
    .expert_level=0
}
filter_unpredicted_refls_in_output = True
  .type = bool
  .help = filter reflections in the output refl table for which there was no model bragg peak
  .help = after stage 1 termination
  .expert_level=10
tag = stage1
  .type = str
  .help = output name tag
  .expert_level=0
ignore_existing = False
  .type = bool
  .help = experimental, ignore expts that already have optimized models in the output dir
  .expert_level=0
global_method = *basinhopping annealing
  .type = choice
  .help = the method of global optimization to use
  .expert_level=10
nelder_mead_maxfev = 60
  .type = int
  .help = multiplied by total number of modeled pixels to get max number of iterations
  .expert_level=10
nelder_mead_fatol = 0.0001
  .type = float
  .help = nelder mead functional error tolerance
niter_per_J = 1
  .type = int
  .help = if using gradient descent, compute gradients
  .help = every niter_per_J iterations .
  .expert_level=10
rescale_params = True
  .type = bool
  .help = DEPRECATED, this parameter no longer has meaning.
  .expert_level=10
best_pickle = None
  .type = str
  .help = path to a pandas pickle containing the best models for the experiments
  .expert_level=0
betas
  .help = variances for the restraint targets
  .expert_level=0
{
  Finit = None
    .type = float
  Friedel = None
    .type = float
    .help = set this to some value to restraint Friedel mates during refinement (ensemble mode) . Lower values are
    .help = tightly restrained . (Exploratory, experimental phil param)
  ucell_a = None
    .type = float
    .help = restraint variance for unit cell a
  ucell_b = None
    .type = float
    .help = restraint variance for unit cell b
  ucell_c = None
    .type = float
    .help = restraint variance for unit cell c
  ucell_alpha = None
    .type = float
    .help = restraint variance for unit cell alpha angle
  ucell_beta = None
    .type = float
    .help = restraint variance for unit cell beta angle
  ucell_gamma = None
    .type = float
    .help = restraint variance for unit cell gamma angle
  Nvol = None
    .type = float
    .help = tightness of the Nabc volume contraint
  detz_shift = None
    .type = float
    .help = restraint variance for detector shift target
  ucell = None
    .type = floats
    .help = DEPRECATED: use e.g. betas.ucell_a instead
    .help = variances for unit cell constants in order determined by unit cell manager class (see diffBragg/refiners/crystal_systems)
  RotXYZ = None
    .type = float
    .help = restraint factor for the rotXYZ restraint
  Nabc = None
    .type = floats(size=3)
    .help = restraint factor for the ncells abc
  Ndef = None
    .type = floats(size=3)
    .help = restraint factor for the ncells def
  diffuse_sigma = None
    .type = floats(size=3)
    .help = restraint factor for diffuse sigma
  diffuse_gamma = None
    .type = floats(size=3)
    .help = restraint factor for diffuse gamma
  G = None
    .type = float
    .help = restraint factor for the scale G
  B = None
    .type = float
    .help = restraint factor for Bfactor
  eta_abc = None
    .type = floats(size=3)
    .help = restrain factor for mosaic spread angles
  spec = None
    .type = floats(size=2)
    .help = restraint factor for spectrum coefs
  Fhkl = None
    .type = float
    .help = restraint factor for structure factor intensity scales
}
dual
  .help = configuration parameters for dual annealing
  .expert_level=10
{
  initial_temp = 5230
    .type = float
    .help = init temp for dual annealing
  no_local_search = False
    .type = bool
    .help = whether to try local search procedure with dual annealing
    .help = if False, then falls back on classical simulated annealing
  visit = 2.62
    .type = float
    .help = dual_annealing visit param, see scipy optimize docs
  accept = -5
    .type = float
    .help = dual_annealing accept param, see scipy optimize docs
}
centers
  .help = restraint targets
  .expert_level=0
{
  ucell_a = None
    .type = float
    .help = restraint target for unit cell a (Angstrom)
  ucell_b = None
    .type = float
    .help = restraint target for unit cell b (Angstrom)
  ucell_c = None
    .type = float
    .help = restraint target for unit cell c (Angstrom)
  ucell_alpha = None
    .type = float
    .help = restraint target for unit cell alpha angle (deg.)
  ucell_beta = None
    .type = float
    .help = restraint target for unit cell beta angle (deg.)
  ucell_gamma = None
    .type = float
    .help = restraint target for unit cell gamma angle (deg.)
  Nvol = None
    .type = float
    .help = if provided, constrain the product Na*Nb*Nc to this value
  detz_shift = None
    .type = float
    .help = restraint target for detector shift along z-direction
  RotXYZ = None
    .type = floats(size=3)
    .help = restraint target for Umat rotations
  Nabc = None
    .type = floats(size=3)
    .help = restraint target for Nabc
  Ndef = None
    .type = floats(size=3)
    .help = restraint target for Ndef
  diffuse_sigma = None
    .type = floats(size=3)
    .help = restraint target for diffuse sigma
  diffuse_gamma = None
    .type = floats(size=3)
    .help = restraint target for diffuse gamma
  G = None
    .type = float
    .help = restraint target for scale G
  B = None
    .type = float
    .help = restraint target for Bfactor
  eta_abc = None
    .type = floats(size=3)
    .help = restraint target for mosaic spread angles in degrees
  spec = None
    .type = floats(size=2)
    .help = restraint target for specturm correction (0 + 1*Lambda )
}
skip = None
  .type = int
  .help = skip this many exp
  .expert_level=0
hess = None
  .type = str
  .help = scipy minimize hessian argument, 2-point, 3-point, cs, or None
  .expert_level=10
stepsize = 0.5
  .type = float
  .help = basinhopping stepsize
  .expert_level=10
temp = 1
  .type = float
  .help = temperature for basin hopping algo
  .expert_level=10
niter = 0
  .type = int
  .help = number of basin hopping iterations (0 just does a gradient descent and stops at the first minima encountered)
  .expert_level=0
exp_ref_spec_file = None
  .type = str
  .help = path to 3 col txt file containing file names for exper, refl, spectrum (.lam)
  .help = Note: only single-image experiment lists are supported! Uses dials.split_experiments or diffBragg.make_input_file if necessary
  .expert_level=0
method = None
  .type = str
  .help = minimizer method, usually this is L-BFGS-B (gradients) or Nelder-Mead (simplex)
  .help = other methods are experimental (see details in hopper_utils.py)
  .expert_level=0
opt_det = None
  .type = str
  .help = path to experiment with optimized detector model
  .expert_level=0
opt_beam = None
  .type = str
  .help = path to experiment with optimized beam model
  .expert_level=0
number_of_xtals = 1
  .type = int
  .help = number of crystal domains to model per shot
  .expert_level=10
sanity_test_input = True
  .type = bool
  .help = sanity test input
  .expert_level=10
outdir = None
  .type = str
  .help = output folder
  .expert_level=0
max_process = -1
  .type = int
  .help = max exp to process
  .expert_level=0
types
  .help = type of target to parameter (see diffBragg.refiners.parameters.py)
  .expert_level=10
{
  G = *ranged positive
    .type = choice
  Nabc = *ranged positive
    .type = choice
  diffuse_sigma = *ranged positive
    .type = choice
  diffuse_gamma = *ranged positive
    .type = choice
}
sigmas
  .help = sensitivity of target to parameter (experimental)
  .expert_level=10
{
  spec = [1,1]
    .type=floats(size=2)
    .help = spectrum offset and scale factor sigmas
  roiPerScale = 1
    .type = float
  detz_shift = 1
    .type = float
    .help = sensitivity shift for the overall detector shift along z-direction
  Nabc = [1,1,1]
    .type = floats(size=3)
    .help = sensitivity for Nabc
  Ndef = [1,1,1]
    .type = floats(size=3)
    .help = sensitivity for Ndef
  diffuse_sigma = [1,1,1]
    .type = floats(size=3)
    .help = sensitivity for diffuse sigma
  diffuse_gamma = [1,1,1]
    .type = floats(size=3)
    .help = sensitivity for diffuse gamma
  RotXYZ = [1e-3,1e-3,1e-3]
    .type = floats(size=3)
    .help = sensitivity for RotXYZ in radians
  G = 1
    .type = float
    .help = sensitivity for scale factor
  B = 1
    .type = float
    .help = sensitivity for Bfactor
  eta_abc = [1,1,1]
    .type = floats(size=3)
    .help = sensitivity of mosaic spread parameters
  ucell = [1,1,1,1,1,1]
    .type = floats
    .help = sensitivity for unit cell params
  Fhkl = 1
    .type = float
    .help = sensitivity for structure factors
}
init
  .help = initial value of model parameter (will be overrided if best pickle is provided)
  .expert_level=0
{
  random_Gs = None
    .type = floats
    .help = list of floats from which to select an init.G at random
  random_Nabcs = None
    .type = floats
    .help = list of random floats from which to select init.Nabc at random
  spec = [0,1]
    .type = floats(size=2)
    .help = initial offset and scale factor applied to each energy channel wavelength
  detz_shift = 0
    .type = float
    .help = initial value for the detector position overall shift along z-direction in millimeters
  Nabc = [100,100,100]
    .type = floats(size=3)
    .help = init for Nabc
  Ndef = [0,0,0]
    .type = floats(size=3)
    .help = init for Ndef
  diffuse_sigma = [.01,.01,.01]
    .type = floats(size=3)
    .help = init diffuse sigma
  diffuse_gamma = [1,1,1]
    .type = floats(size=3)
    .help = init for diffuse gamma
  RotXYZ = [0,0,0]
    .type = floats(size=3)
    .help = init for RotXYZ in radians
  G = 1
    .type = float
    .help = init for scale factor
  B = 0
    .type = float
    .help = init for B factor
  eta_abc = [0,0,0]
    .type = floats(size=3)
    .help = initial values (in degrees) for anisotropic mosaic spread about the 3 crystal axes a,b,c
    .help = Note, these can never be exactly 0 if fix.eta_abc=False
}
mins
  .help = min value allowed for parameter
  .expert_level = 0
{
  detz_shift = -10
    .type = float
    .help = min value for detector z-shift in millimeters
  Nabc = [3,3,3]
    .type = floats(size=3)
    .help = min for Nabc
  Ndef = [-200,-200,-200]
    .type = floats(size=3)
    .help = min for Ndef
  diffuse_sigma = [0,0,0]
    .type = floats(size=3)
    .help = min diffuse sigma
  diffuse_gamma = [0,0,0]
    .type = floats(size=3)
    .help = min for diffuse gamma
  RotXYZ = [-3.1415926, -3.1415926, -3.1415926]
    .type = floats(size=3)
    .help = min for rotXYZ in radians
  G = 0
    .type = float
    .help = min for scale G
  B = 0
    .type = float
    .help = min for Bfactor
  Fhkl = 0
    .type = float
    .help = min for structure factors
  eta_abc = [0,0,0]
    .type = floats(size=3)
    .help = min value (in degrees) for mosaic spread angles
  spec = [-0.01, 0.95]
    .type = floats(size=2)
    .help = min value for spectrum correction (-0.01 + Lambda *1.05)
}
maxs
  .help = max value allowed for parameter
  .expert_level = 0
{
  detz_shift = 10
    .type = float
    .help = max value for detector z-shift in millimeters
  eta = 0.1
    .type = float
    .help = maximum mosaic spread in degrees
  Nabc = [300,300,300]
    .type = floats(size=3)
    .help = max for Nabc
  Ndef = [200,200,200]
    .type = floats(size=3)
    .help = max for Ndef
  diffuse_sigma = [20,20,20]
    .type = floats(size=3)
    .help = max diffuse sigma
  diffuse_gamma = [10000,10000,10000]
    .type = floats(size=3)
    .help = max for diffuse gamma
  RotXYZ = [3.1415926, 3.1415926, 3.1415926]
    .type = floats(size=3)
    .help = max for rotXYZ in radians
  G = 1e12
    .type = float
    .help = max for scale G
  B = 1e3
    .type = float
    .help = max for Bfactor
  eta_abc = [10,10,10]
    .type = floats(size=3)
    .help = maximum value (in degrees) for mosaic spread angles
  Fhkl = 1e6
    .type = float
    .help = max for structure factors
  spec = [0.01, 1.05]
    .type = floats(size=2)
    .help = max value for spectrum correction (0.01 + Lambda *1.05)
}
fix
  .help = flags for fixing parameters during refinement
  .expert_level = 0
{
  Fhkl = True
    .type = bool
    .help = fix the structure factors scales during refinement
  spec = True
    .type = bool
    .help = fix the spectrum. If False, a spectrum correction is refined (wavelength shift and scale)
  perRoiScale = True
    .type = bool
    .help = a per-roi scale factor
  G = False
    .type = bool
    .help = fix the Bragg spot scale during refinement
  B = True
    .type = bool
    .help = fix the Bfactor during refinement
  eta_abc = True
    .type = bool
    .help = fix the mosaic spread parameters during refinement
  RotXYZ = False
    .type = bool
    .help = fix the misorientation matrix during refinement
  Nabc = False
    .type = bool
    .help = fix the diagonal mosaic domain size parameters during refinement
  Ndef = True
    .type = bool
    .help = fix the diagonal mosaic domain size parameters during refinement
  diffuse_sigma = True
    .type = bool
    .help = fix diffuse sigma
  diffuse_gamma = True
    .type = bool
    .help = fix diffuse gamma
  ucell = False
    .type = bool
    .help = fix the unit cell during refinement
  detz_shift = True
    .type = bool
    .help = fix the detector distance shift during refinement
}
relative_tilt = False
  .type = bool
  .help = fit tilt coef relative to roi corner
  .expert_level = 10
ucell_edge_perc = 10
  .type = float
  .help = precentage for allowing ucell to fluctuate during refinement
  .expert_level = 10
ucell_ang_abs = 5
  .type = float
  .help = absolute angle deviation in degrees for unit cell angles to vary during refinement
  .expert_level = 10
no_Nabc_scale = False
  .type = bool
  .help = toggle Nabc scaling of the intensity
  .expert_level = 10
use_diffuse_models = False
  .type = bool
  .help = if True, let the values of init.diffuse_sigma and init.diffuse_gamma
  .help = be used to define the diffuse scattering. Set e.g. fix.diffuse_sigma=True in order to refine them
  .expert_level = 10
diffuse_stencil_size = 0
  .type = int
  .help = Increase to add accuracy to diffuse scattering models, at the expense of longer computations
  .help = Best to increment by values of 1 when testing
diffuse_orientation = 1
  .type = int
  .help = orient the diffuse scattering features. 0 is along (a-b, a+b, c), 1 is along (a,b,c)
symmetrize_diffuse = True
  .type = bool
  .help = use the laue group rotation operators to symmetrize diffuse signals
gamma_miller_units = False
  .type = bool
  .help = if True, let the values of init.diffuse_gamma be expressed in Miller index units
  .expert_level = 10
sigma_frac = None
  .type = float
  .help = sigma for Fhkl restraints will be some fraction of the starting value
  .expert_level = 10
sanity_test_hkl_variation = False
  .type = bool
  .help = measure the variation of each HKL within the shoebox
  .expert_level = 10
sanity_test_models = False
  .type = bool
  .help = make sure best models from stage 1 are reproduced at the start
  .expert_level = 10
sanity_test_amplitudes = False
  .type = bool
  .help = if True, then quickly run a sanity check ensuring that all h,k,l are predicted
  .help = and/or in the starting miller array
  .expert_level = 10
x_write_freq = 25
  .type = int
  .help = save x arrays every x_write_freq iterations
  .expert_level = 10
percentile_cut = None
  .type = float
  .help = percentile below which pixels are masked
  .expert_level = 10
remove_duplicate_hkl = False
  .type = bool
  .help = for hopper, remove duplicate HKLs in input refl files
space_group = None
  .type = str
  .help = space group to refine structure factors in
  .expert_level = 0
first_n = None
  .type = int
  .help = refine the first n shots only
  .expert_level = 0
maxiter = 15000
  .type = int
  .help = stop refiner after this many iters
  .expert_level = 10
ftol = 1e-10
  .type = float
  .help = ftol convergence threshold for scipys L-BFGS-B
  .expert_level = 10
lbfgs_maxiter = 1e5
  .type = int
  .help = maximum number of L-BFGS-B iterations
disp = False
  .type = bool
  .help = scipy minimize convergence printouts
  .expert_level = 10
use_restraints = False
  .type = bool
  .help = enable the parameter restraints
  .expert_level = 0
min_multi = 2
  .type = int
  .help = minimum ASU multiplicity, obs that fall below this threshold
  .help = are removed from analysis
  .expert_level = 10
min_spot = 5
  .type = int
  .help = minimum spots on a shot in order to optimize that shot
  .expert_level = 10
store_wavelength_images = False
  .type = bool
  .help = for simtbx.diffBragg.hopper, optionally write subimages whose value
  .help = is the avereage wavelength per pixels, weighted by the model
logging
  .help = controls the logging module for hopper and stage_two
  .expert_level = 10
{
  show_params_at_minimum = True
    .type = bool
    .help = show the optimized parameters once a basinhopping minimum is reached
  parameters = False
    .type = bool
    .help = whether to display hopper refinement parameters at each iteration
  disable = False
    .type = bool
    .help = turn off logging
  logfiles_level = low *normal high
    .type = choice
    .help = level of the main log when writing logfiles
  logfiles = False
    .type = bool
    .help = write log files in the outputdir
  rank0_level = low *normal high
    .type = choice
    .help = console log level for rank 0, ignored if logfiles=True
  other_ranks_level = *low normal high
    .type = choice
    .help = console log level for all ranks > 0, ignored if logfiles=True
  overwrite = True
    .type = bool
    .help = overwrite the existing logfiles
  logname = mainLog
    .type = str
    .help = if logfiles=True, then write the log to this file, stored in the folder specified by outdir
    .help = if None, then defaults to main_stage1.log for hopper, main_pred.log for prediction, main_stage2.log for stage_two
  log_hostname = True
    .type = bool
    .help = prefix logfiles with host name
}
profile = False
  .type = bool
  .help = profile the workhorse functions
  .expert_level = 0
profile_name = lineProf
  .type = str
  .help = name of the output file that stores the line-by-line profile (written to folder specified by outdir)
  .help = if None, defaults to prof_stage1.log, prof_pred.log, prof_stage2.log for hopper, prediction, stage_two respectively
  .expert_level = 10
"""

simulator_phil = """
simulator {
  oversample = 0
    .type = int
    .help = pixel oversample rate (0 means auto-select)
  device_id = 0
    .type = int
    .help = device id for GPU simulation
  init_scale = 1
    .type = float
    .help = initial scale factor for this crystal simulation
  total_flux = 1e12
    .type = float
    .help = total photon flux for all energies
  crystal {
    ncells_abc = (10,10,10)
      .type = floats(size=3)
      .help = number of unit cells along each crystal axis making up a mosaic domain
    ncells_def = (0,0,0)
      .type = floats(size=3)
      .help = off-diagonal components for mosaic domain model (experimental)
    has_isotropic_ncells = False
      .type = bool
      .help = if True, ncells_abc are constrained to be the same values during refinement
    has_isotropic_mosaicity = False
      .type = bool
      .help = if True, eta_abc are constrained to be the same values during refinement
    mosaicity = 0
      .type = float
      .help = mosaic spread in degrees
    anisotropic_mosaicity = None
      .type = floats
      .help = mosaic spread 3-tuple or 6-tuple specifying anisotropic mosaicity
    num_mosaicity_samples = 1
      .type = int
      .help = the number of mosaic domains to use when simulating mosaic spread
    mos_angles_per_axis = 10
      .type = int
      .help = if doing a uniform mosaicity sampling, use this many angles per rotation axis
    num_mos_axes = 10
      .type = int
      .help = number of sampled rot axes if doing a uniform mosaicity sampling
    mosaicity_method = 2
      .type = int
      .help = 1 or 2. 1 is random sampling, 2 is even sampling
    rotXYZ_ucell = None
      .type = floats(size=9)
      .help = three missetting angles (about X,Y,Z axes), followed by
      .help = unit cell parameters. The crystal will be rotated according to
      .help = the matrix RotX*RotY*RotZ, and then the unit cell will be updated
  }
  gonio {
    delta_phi = None
      .type = float
      .help = Angular amount in degrees by which goniometer is rotated during shot
    phi_steps = 50
      .type = int
      .help = number of discrete angular positions to model
  }
  structure_factors {
    from_pdb {
      name = None
        .type = str
        .help = path to a pdb file
      add_anom = True
        .type = bool
        .help = Use the dxtbx beams wavelength to sample the henke tables
        .help = and add anomalous contributions from each atom
      k_sol = None
        .type = float
        .help = solvent component of structure factor: k_sol * exp(-b_sol*s^2/4)
      b_sol = None
        .type = float
        .help = solvent component of structure factor: k_sol * exp(-b_sol*s^2/4)
    }
    mtz_name = None
      .type = str
      .help = path to an MTZ file . If an mtz_name and from_pdb.name are both provided, then
      .help = the mtz takes precedence
    mtz_column = None
      .type = str
      .help = column in an MTZ file
    dmin = 1
      .type = float
      .help = minimum resolution for structure factor array (not applicable when F is loaded from mtz)
    dmax = None
      .type = float
      .help = maximum resolution for structure factor array (not applicable when F is loaded from mtz)
    default_F = 0
      .type = float
      .help = Default value for structure factor amps . MIssing structure factors will have this value
      .help = during simulation, for example if the mtz is incomplete. Also, if mtz_name and
      .help = from_pdb.name are both None, then a structure factor array will be created with this
      .help = value as every amplitude.
  }
  spectrum {
    filename = None
      .type = str
      .help = a .lam file (precognition) for inputting wavelength spectra
    stride = 1
      .type = int
      .help = stride of the spectrum (e.g. set to 10 to keep every 10th value in the spectrum file data)
    filename_list = None
      .type = str
      .help = path to a file containing 1 .lam filename per line
    gauss_spec {
      fwhm = 10
        .type = float
        .help = width of the gaussian in electron volts
      nchannels = 20
        .type = int
        .help = total number of spectrum energies, centered on the shots nominal energy as determined from format class
      res = 1
        .type = float
        .help = energy resolution of the spectrum
    }
  }
  beam {
    size_mm = 1
      .type = float
      .help = diameter of the beam in mm
  }
  detector {
    thick = None
      .type = float
      .help = sensor thickness in millimeters. Overrides dxtbx detector model.
      .help = Note: must also provide param `atten`, otherwise this param is ineffective
    atten = None
      .type = float
      .help = x-ray absorption length in millimeters
      .help = for sensor. Overrides dxtbx detector model.
      .help = Note: must also provide param `thick`, otherwise this param is ineffective
    force_zero_thickness = False
      .type = bool
      .help = if True, then set sensor thickness to 0
    thicksteps = 1
      .type = int
      .help = number of layers within sensor where scattering
      .help = will be averaged over (evenly divided). This is a nanoBragg attribute
  }
  psf {
    use = False
      .type = bool
      .help = optionally apply a point-spread-function to the model
    fwhm = 100
      .type = float
      .help = PSF full width half max in microns
    radius = 7
      .type = int
      .help = PSF kernel radius (in pixels)
  }
}
"""

refiner_phil = """
refiner {
  check_expt_format = True
    .type = bool
    .help = In some cases the expt is only used for the crystal model, in which case set check_expt_format=False.
    .help = If, however, the experimental data and/or spectra are to be extracted from the expt, then this
    .help = should  remain True.
  refldata_trusted = *allValid fg bg
    .type = choice
    .help = If loading data from reflection table, choose which pixels are flagged as trusted/
    .help = The default is allValid, meaning any shoebox mask value > 1.
    .help = fg is any foreground (integrated), valid pixel (mask==5).
    .help = bg is any valid pixel used for background fitting or foreground (integration).
    .help = Note, in this context, Foreground is usually the central shoebox pixels marked for integration.
  refldata_to_photons = False
    .type = bool
    .help = If loading data from reflection table, then optionally normalize by the refiner.adu_to_photon factor.
    .help = If the reflection tables were created using dials.integrate or dials.stils_process,
    .help = then you will need to set this flag to True.
  load_data_from_refl = False
    .type = bool
    .help = Rather than load data from the experiment, load data from the reflection table shoeboxes.
    .help = The data in shoeboxes is stored in float32. The shoebox bound boxes will determing the regions of
    .help = pixels used for refinement. It is assumed that shoebox background, data, and the mask are properly set.
    .help = See the method GatherFromReflectionTable in hopper_utils.
  test_gathered_file = False
    .type = bool
  gather_dir = None
    .type = str
    .help = optional dir for stashing loaded input data in refl files (mainly for tests/portability)
  break_signal = None
    .type = int
    .help = intended to be used to break out of a long refinement job prior to a timeout on a super computer
    .help = On summit, set this to 12 (SIGUSR2), at least thats what it was last I checked (July 2021)
  debug_pixel_panelfastslow = None
    .type = ints(size=3)
    .help = 3-tuple of panel ID, fast coord, slow coord. If set, show the state of diffBragg
    .help = for this pixel once refinement has finished
  gain_map_min_max = [.5,2]
    .type = floats(size=2)
    .help = the min, max allowed values for the gain correction terms
    .help = that are applied to each region (defined by region_size)
  refine_gain_map = False
    .type = bool
    .help = flag for refining a detector gain map, defined by the parameter region_size
  save_gain_freq=10
    .type = int
    .help = after how many iterations should we save the optimized gain map
  region_size = [50,50]
    .type = ints(size=2)
    .help = Used for gain correction. size of subregions in each detector module in pixels.
    .help = Each panel region will be divided into blocks of shape region_size
    .help = and a unique gain correction will be applied to each subregion.
    .help = Note, this will usually be square shaped, but its (slowDim,fastDim).
  res_ranges = None
    .type = str
    .help = resolution-defining strings, where each string is
    .help = is comma-separated substrings, formatted according to "%f-%f,%f-%f" where the first float
    .help = in each substr specifies the high-resolution for the refinement trial, and the second float
    .help = specifies the low-resolution for the refinement trial. Should be same length as max_calls
  force_symbol = None
    .type = str
    .help = a space group lookup symbol used to map input miller indices to ASU
  force_unit_cell = None
    .type = ints(size=6)
    .help = a unit cell tuple to use
  num_devices = 1
    .type = int
    .help = number of cuda devices on current node
  refine_Fcell = None
    .type = ints(size_min=1)
    .help = whether to refine the structure factor amplitudes
  refine_spot_scale = None
    .type = ints(size_min=1)
    .help = whether to refine the crystal scale factor
  refine_Nabc = False
    .type = bool
    .help = whether to refine the mosaic domain size tensor
  gain_restraint=None
    .type = floats(size=2)
    .help = if not None, apply a gain restraint to the data
    .help = This is two parameters, a center and a variance
  max_calls = [100]
    .type = ints(size_min=1)
    .help = maximum number of calls for the refinement trial
  panel_group_file = None
    .type = str
    .help = a text file with 2 columns, the first column is the panel_id and the second
    .help = column is the panel_group_id. Panels geometries in the same group are refined together
  update_oversample_during_refinement = False
    .type = bool
    .help = whether to update the oversample parameter as ncells changes
  sigma_r = 3
    .type = float
    .help = standard deviation of the dark signal fluctuation
  adu_per_photon = 1
    .type = float
    .help = how many ADUs (detector units) equal 1 photon
  use_curvatures_threshold = 10
    .type = int
    .help = how many consecutiv positive curvature results before switching to curvature mode
  curvatures = False
    .type = bool
    .help = whether to try using curvatures
  start_with_curvatures = False
    .type = bool
    .help = whether to try using curvatures in the first iteration
  tradeps = 1e-2
    .type = float
    .help = LBFGS termination parameter  (smaller means minimize for longer)
  io {
    restart_file = None
      .type = str
      .help = output file for re-starting a simulation
    output_dir = None
      .type = str
      .help = optional output directory
  }
  quiet = False
    .type = bool
    .help = silence the refiner
  verbose = 0
    .type = int
    .help = verbosity level (0-10) for nanoBragg
  num_macro_cycles = 1
    .type = int
    .help = keep repeating the same refinement scheme over and over, this many times
  ncells_mask = *000 110 101 011 111
    .type = choice
    .help = a mask specifying which ncells parameters should be the same
    .help = e.g. 110 specifies Na and Nb are refined together as one parameter
  reference_geom = None
    .type = str
    .help = path to expt list file containing a detector model
  stage_two {
    use_nominal_hkl = True
      .type = bool
      .help = use the nominal hkl as a filter for Fhkl gradients
    save_model_freq = 50
      .type = int
      .help = save the model  after this many iterations
    save_Z_freq = 25
      .type = int
      .help = save Z-scores for all pixels after this many iterations
    min_multiplicity = 1
      .type = int
      .help = structure factors whose multiplicity falls below this value
      .help = will not be refined
    Fref_mtzname = None
      .type = str
      .help = path to a reference MTZ file. if passed, this is used solely to
      .help = observe the R-factor and CC between it and the Fobs being optimized
    Fref_mtzcol = "Famp(+),Famp(-)"
      .type = str
      .help = column in the mtz file containing the data
    d_min = 2
      .type = float
      .help = high res lim for binner
    d_max = 999
      .type = float
      .help = low res lim for binner
    n_bin = 10
      .type = int
      .help = number of binner bins
  }
}
"""

roi_phil = """
roi {
  centroid = *obs cal
    .type = choice
    .help = Determines which refl table column contains the spot centroids
    .help = Shoeboxes are drawn around the centroids, and refinement uses pixels
    .help = within those shoeboxes.
    .help = obs: xyz.px.value  cal: xyzcal.px
  trusted_range = None
    .type = floats(size=2)
    .help = optional override for detector trusted range, should be (min,max)
  mask_all_if_any_outside_trusted_range = True
    .type = bool
    .help = If a reflection has any pixels which are outside the detectors
    .help = trusted range, then mask the entire reflections and surrounding
    .help = pixels. If False, then only pixels outside the range are masked.
    .help = Note: this only takes effect if mask_outside_trusted_range=True.
  mask_outside_trusted_range = False
    .type = bool
    .help = Check the dxtbx detector's trusted range and use that to mask
    .help = out-of-range pixels on a per-image basis
  only_filter_zingers_above_mean = True
    .type = bool
    .help = if fitting background, theres a zinger filter step (background_threshold)
    .help = and typically it only applies to pixels above the mean
    .help = Set this to False to filter zingers below the mean, which is useful for
    .help = data with low background signal.
  cache_dir_only = False
    .type = bool
    .help = if True, create the cache folder , populate it with the roi data, then exit
  fit_tilt = False
    .type = bool
    .help = fit tilt plane, or else background is simply an offset
  force_negative_background_to_zero = False
    .type = bool
    .help = if True and the background model evaluates to a negative number
    .help = within an ROI, then force the background to be 0 for all pixels in that ROI
  background_threshold = 3.5
    .type = float
    .help = for determining background pixels
  pad_shoebox_for_background_estimation = None
    .type = int
    .help = shoebox_size specifies the dimenstion of the shoebox used during refinement
    .help = and this parameter is used to increase that shoebox_size only during the background
    .help = estimation stage
  shoebox_size = 10
    .type = int
    .help = roi box dimension
  deltaQ = None
    .type = float
    .help = roi dimension in inverse Angstrom, such that shoeboxes at wider angles are larger.
    .help = If this parameter is supplied, shoebox_size will be ignored.
  reject_edge_reflections = True
    .type = bool
    .help = whether to reject ROIs if they occur near the detector panel edge
  reject_roi_with_hotpix = True
    .type = bool
    .help = whether to reject an ROI if it has a bad pixel
  hotpixel_mask = None
    .type = str
    .help = path to a hotpixel mask (hot pixels set to True)
  panels = None
    .type = str
    .help = panel list for refinement as a string, e.g. "0-8,10,32-40" . The ranges are inclusive,
    .help = e.g. 0-8 means panels 0,1,2,3,4,5,6,7,8
  fit_tilt_using_weights = False
    .type = bool
    .help = if not using robust estimation for background, and instead using tilt plane fit,
    .help = then this parameter will toggle the use of weights. Weights are the estimated
    .help = pixel variance, incuding readout and shot noises.
  allow_overlapping_spots = False
    .type = bool
    .help = if True, then model overlapping spots
  skip_roi_with_negative_bg = True
    .type = bool
    .help = If a region of interest contains negative background model, then skip entire region,else
    .help = mask the pixels with negative background model.
}

geometry {
  save_state_freq = 50
    .type = int
    .help = how often to save all model parameters
  save_state_overwrite = True
    .type = bool
    .help = whether to overwrite model parameter files each time they are saved
  pandas_dir = None
    .type = str
    .help = If provided, Pandas dataframes for each shot will be written here
    .help = These are the same format as those saved during stage 1
    .help = Also, data modelers will pickled and written here as well
  optimized_results_tag = None
    .type = str
    .help = optional tagname, if provided,write optimized refls/expts alongside the
    .help = input refls/expts however using this tag suffix
  refls_key = stage1_refls
    .type = str
    .help = column name for the input pickle which contains the reflection tables to be modeled
  optimize_method = *lbfgsb nelder
    .type = choice
    .help = lmfit optimization method (lbfgsb uses gradients, nelder is graientless)
  input_pkl = None
    .type = str
    .help = path to the input pickle (from hopper) containing models and experiment lists
  input_pkl_glob = None
    .type = str
    .help = path to a glob of input pandas pickles output by hopper or geometry_refiner
  optimize = False
    .type = bool
    .help = flag to specify whether to optimize the geometry
  save_optimized_det_freq = 1
    .type = int
    .help = Save the optimzied detector model every X iterations
  optimized_detector_name = "diffBragg_detector.expt"
    .type = str
    .help = basename of the experiment which will be written, and will contain the optimized detector. Note, this should be a basename only (not to be prefixed with a directory path). If a directory path is include, it will be stripped. The file will be stored in the output folder (pandas_dir)
  min {
    panel_rotations = -1,-1,-1
      .type = floats(size=3)
      .help = minimum value in degrees for a detector panel rotation
    panel_translations = -1,-1,-1
      .type = floats(size=3)
      .help = minimum value in mm for detector panel translations in X,Y,Z
  }
  max {
    panel_rotations = 1,1,1
      .type = floats(size=3)
      .help = maximum value in degrees for a detector panel rotation
    panel_translations = 1,1,1
      .type = floats(size=3)
      .help = maximum value in mm for detector panel translations in X,Y,Z
  }
  center {
    panel_rotations = 0,0,0
      .type = floats(size=3)
      .help = restraint target in degrees for panel rotations
    panel_translations = 0,0,0
      .type = floats(size=3)
      .help = restraint target in mm for detector panel translations in X,Y,Z
  }
  betas {
    panel_rot = 1e6,1e6,1e6
      .type = floats(size=3)
      .help = restraint factor for panel rotations (higher values lead to unrestrained parameters)
    panel_xyz = 1e6,1e6,1e6
      .type = floats(size=3)
      .help = restraint factor in mm for detector panel translations in X,Y,Z
    close_distances = None
      .type = float
      .help = restraint factor for the spread of detector panel Z-distances (#TODO think about this in context of tilt)
  }
  fix {
    panel_rotations = 0,0,0
      .type = ints(size=3)
      .help = refinement flags, 1 means to fix the parameter
    panel_translations = 0,0,0
      .type = ints(size=3)
      .help = refinement flags, 1 means to fix the parameter
  }
}
"""

predictions_phil = """
predictions {
  use_peak_detection = False
    .type = bool
    .help = If True, then simulations will be converted to refl tables
    .help = where each reflection is a single pixel corresponding
    .help = to the peak in the simulated spot. This is useful if one
    .help = models spatial overlaps with separate peaks
  verbose = False
    .type = bool
    .help = See more console output from prediction methods
  laue_mode = False
    .type = bool
    .help = if True, predict the per-pixel wavelenth from the model and use that for index assigment
  qcut = 0.1
    .type = float
    .help = Label predicted reflection as a strong reflection if its within this
    .help = many inverse Angstromg (q=2/Lambda sin(theta)) of an observed strong spot
  label_weak_col = "xyzobs.px.value"
    .type = str
  weak_fraction = 0.5
    .type = float
    .help = fraction of weak predictions to integrate
  threshold = 1e-3
    .type = float
    .help = value determining the cutoff for the forward model intensity. Bragg peaks will then be determined
    .help = as regions of connected values greater than the threshold
  thicksteps_override = None
    .type = int
    .help = Use to force the number of detector thickness steps to a specific value
  oversample_override = None
    .type = int
    .help = force the pixel oversample rate to this value during the forward model simulation
    .help = for maximum speed gains, set this to 1, but inspect the output!
    .expert_level=10
  use_diffBragg_mtz = False
    .type = bool
    .help = whether to use the mtz supplied to diffBragg for prediction
  Nabc_override = None
    .type = ints(size=3)
    .help = use this value of mosaic block size for every shot, useful to get more predicted spots
    .expert_level=10
  pink_stride_override = None
    .type = int
    .help = if specified, stride through the spectrum according to this interval
  default_Famplitude = 1e3
    .type = float
    .help = default structure factor amplitude for every miller index
    .help = this creates a flat prediction model, where the magnitude is dependent on the distance to the Ewald sphere
  resolution_range = [1,999]
    .type = floats(size=2)
    .help = high-res to low-res limit for prediction model
  symbol_override = None
    .type = str
    .help = specify the space group symbol to use in diffBragg (e,g, P43212),
    .help = if None, then it will be pulled from the crystal model
  method = *diffbragg exascale
    .type = choice
    .help = engine used for computing the forward model
    .help = diffbragg offers CUDA support via the DIFFBRAGG_USE_CUDA=1 environment variable specification
    .help = or openmp support using the OMP_NUM_THREADS flag
    .help = The exascale only uses CUDA (will raise error if CUDA is not confugured)
  mosaic_samples_override = None
    .type = int
    .help = Specify the number of mosaic spread samples
}
"""

philz = simulator_phil + refiner_phil + roi_phil + predictions_phil
phil_scope = parse(philz)
