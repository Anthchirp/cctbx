from __future__ import absolute_import, division, print_function

from iotbx.phil import parse

help_message = '''
Redesign script for merging xfel data
'''

dispatch_phil = """
dispatch {
  step_list = None
    .type = strings
    .help = List of steps to use. None means use the full set of steps to merge.
}
"""

input_phil = """
input {
  override_identifiers = False
    .type = bool
    .help = override whatever identifiers may be present in experiments, replacing with auto-generated hash
  alist {
    file = None
      .type = str
      .multiple = True
      .help = Path to a txt file containing experiment tags or experiment filenames to merge (1 per line, all of same type)
      .help = If contents are files: then only experiments whose filename is in the alist will be merged
      .help = If contents are tags: then only experiments whose filename contains one of the alist tags will be merged
    type = *tags files
      .type = choice
      .help = the contents of the input.alist_file, either dials.stills_process image_tags, or absolute expt file paths
      .help = actually the tag can be any unique substring of the expt filenames found under input.path
      .help = Note, for very large datasets, using type=files should perform better
    op = *keep reject
      .type = choice
      .help = whether we keep or reject experiments according to the alist(s)
  }
  persistent_refl_cols = None
    .type = str
    .multiple = True
    .help = Names of reflection table columns that will remain after all prune steps
    .help = If output.save_experiments_and_reflections=True, then these columns will be in the saved tables.
  keep_imagesets = True
    .type = bool
    .help = If True, keep imagesets attached to experiments
  read_image_headers = False
    .type = bool
    .help = If True, when loading data also read image headers. Not needed when merging integrated data.
    .help = Use when needing to read original image pixel data. Equivalent to check_format in other DIALS programs.
  path = None
    .type = str
    .multiple = True
    .help = paths are validated as a glob, directory or file.
    .help = however, validation is delayed until data are assigned to parallel ranks.
    .help = integrated experiments (.expt) and reflection tables (.refl) must both be
    .help = present as matching files.  Only one need be explicitly specified.
  reflections_suffix = _integrated.refl
    .type = str
    .help = Find file names with this suffix for reflections
  experiments_suffix = _integrated.expt
    .type = str
    .help = Find file names with this suffix for experiments

  parallel_file_load {
    method = *uniform node_memory
      .type = choice
      .help = uniform: distribute input experiments/reflections files uniformly over all available ranks
      .help = node_memory: distribute input experiments/reflections files over the nodes such that the node memory limit is not exceeded.
      .help = Within each node distribute the input files uniformly over all ranks of that node.
    node_memory {
      architecture = "Cori KNL"
        .type = str
        .help = node architecture name. Currently not used.
      limit = 90.0
        .type = float
        .help = node memory limit, GB. On Cori KNL each node has 96 GB of memory, but we use 6 GB as a cushion, so the default value is 90 GB.
      pickle_to_memory = 3.5
        .type = float
        .help = an empirical coefficient to convert pickle file size to anticipated run-time process memory required to load a file of that size
    }
    ranks_per_node = 68
        .type = int
        .help = number of MPI ranks per node
    balance = *global1 global2 per_node
      .type = choice
      .multiple = False
      .help = Balance the input file load by distributing experiments uniformly over all available ranks (global1/2) or over the ranks on each node (per_node)
      .help = The idea behind the "per_node" method is that it doesn't require MPI communications across nodes. But if the input file load varies strongly
      .help = between the nodes, "global1/2" is a much better option. "global1" is accomplished by reshuffling all data across all ranks while "global2" is
      .help = accomplished by sending the minimal necessary information between ranks to deterministically evenly balance the load.
    balance_verbose = False
      .type = bool
      .help = print load balancing details to the main log
    balance_mpi_alltoall_slices = 1
      .type = int
      .expert_level = 2
      .help = memory reduction factor for MPI alltoall.
      .help = Use mpi_alltoall_slices > 1, when available RAM is insufficient for doing MPI alltoall on all data at once.
      .help = The data will then be split into mpi_alltoall_slices parts and, correspondingly, alltoall will be performed in mpi_alltoall_slices iterations.
  }
}

mp {
  method = *mpi
    .type = choice
    .help = Muliprocessing method (only mpi at present)
  debug {
    cProfile = False
      .type = bool
      .help = Enable code profiling. Use (for example) runsnake to visualize processing performance
  }
}
"""

tdata_phil = """
tdata{
  output_path = None
    .type = path
    .help = If output_path is not None, the tdata worker writes out a list of unit cells to a file.
    .help = Generally speaking the program should then stop.  The tdata worker is not active by default, so it is necessary to have
    .help = the following phil configuration: dispatch.step_list=input,tdata.
    .help = The output_path assumes the *.tdata filename extension will be appended.
    .help = More information about using this option is given in the source code, xfel/merging/application/tdata/README.md
}
"""

filter_phil = """
filter
  .help = The filter section defines criteria to accept or reject whole experiments
  .help = or to modify the entire experiment by a reindexing operator
  .help = refer to the select section for filtering of individual reflections
  {
  algorithm = n_obs resolution unit_cell
    .type = choice(multi=True)
  n_obs {
    min = None
      .type = int
      .help = Minimum number of observations for subsequent processing
    max = None
      .type = int
      .help = Maximum number of observations for subsequent processing
  }
  resolution {
    d_min = None
      .type = float
      .help = Reject the experiment unless some reflections extend beyond this resolution limit
    model_or_image = model image
      .type = choice
      .help = Calculate resolution either using the scaling model unit cell or from the image itself
  }
  unit_cell
    .help = Various algorithms to restrict unit cell and space group
    {
    algorithm = range *value cluster
      .type = choice
    value
      .help = Discard lattices that are not close to the given target.
      .help = If the target is left as Auto, use the scaling model
      .help = (derived from either PDB file cryst1 record or MTZ header)
      {
      target_unit_cell = Auto
        .type = unit_cell
      relative_length_tolerance = 0.1
        .type = float
        .help = Fractional change in unit cell dimensions allowed (versus target cell).
      absolute_angle_tolerance = 2.
        .type = float
      target_space_group = Auto
        .type = space_group
      }
    cluster
      .help = CLUSTER implies an implementation (standalone program or fork?) where all the
      .help = unit cells are brought together prior to any postrefinement or merging,
      .help = and analyzed in a global sense to identify the isoforms.
      .help = the output of this program could potentially form the a_list for a subsequent
      .help = run where the pre-selected events are postrefined and merged.
      {
      algorithm = rodgriguez_laio dbscan *covariance
        .type = choice
      covariance
        .help = Read a pickle file containing the previously determined clusters,
        .help = represented by estimated covariance models for unit cell parameters.
        {
        file = None
          .type = path
        component = 0
          .type = int(value_min=0)
        skip_component = None
          .type = int(value_min=0)
          .multiple = True
          .help = If a lattice belongs to any of these components, exclude it from processing.
        mahalanobis = 4.0
          .type = float(value_min=0)
          .help = Is essentially the standard deviation cutoff. Given that the unit cells
          .help = are estimated to be distributed by a multivariate Gaussian, this is the
          .help = maximum deviation (in sigmas) from the central value that is allowable for the
          .help = unit cell to be considered part of the cluster.
        skip_mahalanobis = 4.0
          .type = float(value_min=0)
          .help = Cutoff distance for any components specified under skip_component.
        }
      isoform = None
        .type=str
        .help = unknown at present. if there is more than one cluster, such as in PSII,
        .help = perhaps the program should write separate a_lists.
        .help = Alternatively identify a particular isoform to carry forward for merging.
      }
  }
  outlier {
    mad_thresh = None
      .type = float
      .help = If provided, during  the actual merge step, symmetrically equivalent reflecitons (same ASU) are filtered
      .help = according to there median absolute deviation. mad_thresh=3 means a reflection is filtered
      .help = if its deviation is greater than 3 standard deviations of the median amongst ASU samples,
      .help = i.e. lower values of mad_thresh will filter more reflections.
    min_corr = 0.1
      .type = float
      .help = Correlation cutoff for rejecting individual experiments by comparing observed intensities to the model.
      .help = This filter is not applied if scaling.model==None. No experiments are rejected with min_corr=-1.
      .help = This either keeps or rejects the whole experiment.
    assmann_diederichs {}
  }
}
"""

modify_phil = """
modify
  .help = The MODIFY section defines operations on the integrated intensities
  {
  algorithm = *polarization
    .type = choice
    .multiple = True
  reindex_to_reference
    .help = An algorithm to match input experiments against a reference model to
    .help = break an indexing ambiguity
    {
    dataframe = None
      .type = path
      .help = if not None, save a list of which experiments were reindexed (requires pandas)
      .help = and plot a histogram of correlation coefficients (matplotlib)
    }
  cosym
    .help = Implement the ideas of Gildea and Winter doi:10.1107/S2059798318002978
    .help = to determine Laue symmetry from individual symops
    {
    include scope dials.command_line.cosym.phil_scope
    dataframe = None
      .type = path
      .help = if not None, save a list of which experiments were reindexed (requires pandas)
      .help = and plot a histogram of correlation coefficients (matplotlib)
    anchor = False
      .type = bool
      .help = Once the patterns are mutually aligned with the Gildea/Winter/Brehm/Diederichs methodology
      .help = flip the whole set so that it is aligned with a reference model.  For simplicity, the
      .help = reference model from scaling.model is used.  It should be emphasized that the scaling.model
      .help = is only used to choose the overall alignment, which may be chosen arbitrarily, it does not
      .help = bias the mutual alignment of the experimental diffraction patterns.
    tranch_size = 600
      .type = int(value_min=12)
      .help = Best guess as to the ideal tranch size for generating embedding plots.  Total number of
      .help = experiments will be divided among the MPI ranks so as to approximate the tranch size,
      .help = referring to the composite size (each experiment appears in 3 composite tranches).
      .help = Size represents a tradeoff,  larger size will take longer to compute but use fewer MPI ranks
      .help = and converge to more accurate coset assignment (higher % consensus).  Ideal value is data
      .help = dependent, default of 600 is good for a 200-Angstrom cell, but value should increase for smaller cell.
    twin_axis = None
      .type = ints(size=3)
      .multiple = True
      .help = Rotation axis to test as a reindexing operator. Given as a direct \
          axis, e.g. 1,0,0 is a rotation around the a axis.
    twin_rotation = None
      .type = int
      .multiple = True
      .help = Rotation corresponding to a value of twin_axis. Given as an \
          n-fold rotation, e.g. 2 denotes a twofold rotation.
    single_cb_op_to_minimum = False
      .type = bool
      .help = Internally the lattices are transformed to a primitive cell \
          before cosym analysis. Typically each cb_op_to_minimum is determined \
          individually, but if the options twin_axis and twin_rotation are \
          used, the transformation has to be the same for every lattice. \
          Forcing this is probably harmless but has not been tested extensively.
    plot
      {
      do_plot = True
        .type = bool
        .help = Generate embedding plots to assess quality of modify_cosym reindexing.
      n_max = 1
        .type = int
        .help = If shots were divided into tranches for alignment, generate embedding plots for
        .help = the first n_max tranches.
      interactive = False
        .type = bool
        .help = Open embedding plot in Matplotlib window instead of writing a file, overrides do_plot
      format = *png pdf
        .type = choice
        .multiple = False
      filename = cosym_embedding
        .type = str
      }
    }
  reindex_to_abc
    .help = Apply a reindexing operator
    {
      include scope dials.command_line.reindex.phil_scope
    }
}
"""

select_phil = """
select
  .help = The select section accepts or rejects specified reflections
  .help = refer to the filter section for filtering of whole experiments
  {
  algorithm = panel cspad_sensor significance_filter isolation_forest local_outlier_factor
    .type = choice(multi=True)
  cspad_sensor {
    number = None
      .type = int(value_min=0, value_max=31)
      .multiple = True
      .help = Index in the range(32) specifying sensor on the CSPAD to deselect from merging, for the purpose
      .help = of testing whether an individual sensor is poorly calibrated.
    operation = *deselect select
      .type = choice
      .multiple = True
  }
  significance_filter
    .help = If listed as an algorithm, apply a sigma cutoff (on unmerged data) to limit
    .help = the resolution from each diffraction pattern.
    .help = Implement an alternative filter for fuller-kapton geometry
    {
    n_bins = 12
      .type = int (value_min=2)
      .help = Initial target number of resolution bins for sigma cutoff calculation
    min_ct = 10
      .type = int
      .help = Decrease number of resolution bins to require mean bin population >= min_ct
    max_ct = 50
      .type = int
      .help = Increase number of resolution bins to require mean bin population <= max_ct
    sigma = 0.5
      .type = float
      .help = Remove highest resolution bins such that all accepted bins have <I/sigma> >= sigma
    d_min = None
      .type = float
      .help = Remove the entire lattice if the resolution is not at least this d_min
    }
  reflection_filter
    .help = Algorithms to identify single reflections as anomalies based on resolution and intensity. \
      Isolation Forest and Local Outlier Factor are available algorithms. \
      Refer to scikit-learn for more information: \
        https://scikit-learn.org/stable/modules/outlier_detection.html \
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html \
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
    {
    n_bins = 100
      .type = int
      .help = Number of bins used to determine the mean intensity for normalization. Also tails \
        are taken from within bins instead of over the entire dataset.
    tail_percentile = 0.01
      .type = float
      .help = Percentile used to select the tails of the data.
    apply_lower = True
      .type = bool
      .help = Apply reflection filter to the lower tail of the data.
    apply_upper = True
      .type = bool
      .help = Apply reflection filter to the upper tail of the data.
    do_diagnostics = False
      .type = bool
      .help = Make plots of the reflections selected as outliers
    contamination_lower = 0.0001
      .type = float
      .help = Fraction of lower tail reflections that are outliers
    contamination_upper = 0.0001
      .type = float
      .help = Fraction of upper tail reflections that are outliers
    local_outlier_factor {
      n_neighbors = 200
        .type = int
        .help = Number of neighbors used to determine local density.
      }
    isolation_forest {
      sampling_fraction = 0.05
        .type = float
        .help = Fraction of total dataset subsampled to train each decision tree.
      random_seed = 0
        .type = int
        .help = seed for the random forest model
      }
    }
}
"""

scaling_phil = """
scaling {
  model = None
    .type = str
    .help = PDB filename containing atomic coordinates & isomorphous cryst1 record
    .help = or MTZ filename from a previous cycle. If MTZ, specify mtz.mtz_column_F.
  unit_cell = None
    .type = unit_cell
    .help = Unit cell to be used during scaling and merging. Used if model is not provided
    .help = (e.g. mark1).
  space_group = None
    .type = space_group
    .help = Space group to be used during scaling and merging. Used if model is not provided
    .help = (e.g. mark1).
  model_reindex_op = h,k,l
    .type = str
    .help = Kludge for cases with an indexing ambiguity, need to be able to adjust scaling model
  resolution_scalar = 0.969
    .type = float
    .help = Accommodates a few more miller indices at the high resolution limit to account for
    .help = unit cell variation in the sample. merging.d_min is multiplied by resolution_scalar
    .help = when computing which reflections are within the resolution limit.
  mtz {
    mtz_column_F = fobs
      .type = str
      .help = scaling reference column name containing reference structure factors. Can be
      .help = intensities or amplitudes
    minimum_common_hkls = -1
      .type = int
      .help = minimum required number of common hkls between mtz reference and data
      .help = used to validate mtz-based model. No validation with -1.
  }
  pdb {
    include_bulk_solvent = True
      .type = bool
      .help = Whether to simulate bulk solvent
    k_sol = 0.35
      .type = float
      .help = If model is taken from coordinates, use k_sol for the bulk solvent scale factor
      .help = default is approximate mean value in PDB (according to Pavel)
    b_sol = 46.00
      .type = float
      .help = If model is taken from coordinates, use b_sol for bulk solvent B-factor
      .help = default is approximate mean value in PDB (according to Pavel)
    solvent_algorithm = *mosaic flat
      .type = choice
      .help = Mosaic solvent model is as in https://doi.org/10.1101/2021.12.09.471976
  }
  algorithm = *mark0 mark1
    .type = choice
    .help = "mark0: original per-image scaling by reference to isomorphous PDB model"
    .help = "mark1: no scaling, just averaging (i.e. Monte Carlo
             algorithm).  Individual image scale factors are set to 1."
  weights = *unit icalc icalc_sigma
    .type = choice
    .help = "Sigmas applied in the linear fit of Iobs vs Icalc. unit: sigmas"
            "equal to 1. icalc: Sigmas are proportional to the square root of"
            "Icalc (this relation is totally empirical). icalc_sigma: Error"
            "due to partiality is proportional to Icalc; this term is added to"
            "the sigma(Iobs) determined in integration so that sigma ="
            "sqrt(Icalc**2 + sigma(Iobs)**2)."
}
"""

postrefinement_phil = """
postrefinement {
  enable = True
    .type = bool
    .help = enable the preliminary postrefinement algorithm (monochromatic)
    .expert_level = 3
  algorithm = *rs rs2 rs_hybrid eta_deff
    .type = choice
    .help = rs only, eta_deff protocol 7
    .expert_level = 3
  partiality_threshold_hcfix = 0.2
    .type = float ( value_min = 0.0001 )
    .help = Throw out observations below this value. Hard coded as 0.2 for rs2
    .help = Minimum positive value is required because partiality appears in the denominator
  rs {
    fix = thetax thetay *RS G BFACTOR
      .type = choice(multi=True)
      .help = Which parameters to fix during postrefinement
  }
  rs2
    .help = Reimplement postrefinement with the following (Oct 2016):
    .help = Refinement engine now work on analytical derivatives instead of finite differences
    .help = Better convergence using "traditional convergence test"
    {}
  rs_hybrid
    .help = More aggressive postrefinement with the following (Oct 2016):
    .help = One round of 'rs2' using LBFGS minimizer as above to refine G,B,rotx,roty
    .help = Gentle weighting rather than unit weighting for the postrefinement target
    .help = Second round of LevMar adding an Rs refinement parameter
    .help = Option of weighting the merged terms by partiality
    {
    partiality_threshold = 0.2
      .type = float ( value_min = 0.01 )
      .help = throw out observations below this value. Hard coded as 0.2 for rs2, allow value for hybrid
      .help = must enforce minimum positive value because partiality appears in the denominator
    }
  target_weighting = *unit variance gentle extreme
    .type = choice
    .help = weights for the residuals in the postrefinement target (for rs2 or rs_hybrid)
    .help = Unit: each residual weighted by 1.0
    .help = Variance: weighted by 1/sigma**2.  Doesn't seem right, constructive feedback invited
    .help = Gentle: weighted by |I|/sigma**2.  Seems like best option
    .help = Extreme: weighted by (I/sigma)**2.  Also seems right, but severely downweights weak refl
  merge_weighting = *variance
    .type = choice
    .help = assumed that individual reflections are weighted by the counting variance
  merge_partiality_exponent = 0
    .type = float
    .help = additionally weight each measurement by partiality**exp when merging
    .help = 0 is no weighting, 1 is partiality weighting, 2 is weighting by partiality-squared
  lineshape = *lorentzian gaussian
    .type = choice
    .help = Soft sphere RLP modeled with Lorentzian radial profile as in prime
    .help = or Gaussian radial profile. (for rs2 or rs_hybrid)
  show_trumpet_plot = False
    .type = bool
    .help = each-image trumpet plot showing before-after plot. Spot color warmth indicates I/sigma
    .help = Spot radius for lower plot reflects partiality. Only implemented for rs_hybrid
}
"""
merging_phil = """
merging {
  minimum_multiplicity = 2
    .type = int(value_min=2)
    .help = If defined, merged structure factors not produced for the Miller indices below this threshold.
  error {
    model = ha14 *ev11 mm24 errors_from_sample_residuals
      .type = choice
      .multiple = False
      .help = ha14, formerly sdfac_auto, apply sdfac to each-image data assuming negative
      .help = intensities are normally distributed noise
      .help = errors_from_sample_residuals, use the distribution of intensities in a given miller index
      .help = to compute the error for each merged reflection
    ev11
      .help = formerly sdfac_refine, correct merged sigmas refining sdfac, sdb and sdadd as Evans 2011.
      {
      random_seed = None
        .help = Random seed. May be int or None. Only used for the simplex minimizer
        .type = int
        .expert_level = 1
      minimizer = *lbfgs LevMar
        .type = choice
        .help = Which minimizer to use while refining the Sdfac terms
      refine_propagated_errors = False
        .type = bool
        .help = If True then during sdfac refinement, also \
                refine the estimated error used for error propagation.
      show_finite_differences = False
        .type = bool
        .help = If True and minimizer is lbfgs, show the finite vs. analytical differences
      plot_refinement_steps = False
        .type = bool
        .help = If True, plot refinement steps during refinement.
    }
    mm24
      .help = Maximum log-likelihood from Mittan-Moreau 2024
      {
      expected_gain = None
        .help = Expected gain used for s_fac initialization.\
                If None, initialize s_fac using routine.
        .type = float
      number_of_intensity_bins = 100
        .help = Number of intensity bins
        .type = int
      tuning_param = 10
        .help = Tuning param for t-dist in maximum log likelihood
        .type = float
      n_max_differences = 100
        .help = Maximum number of pairwise differences per reflection.\
                If None, then do not limit the maximum number of differences
        .type = int
      random_seed = 50298
        .help = Seed used to establish the random number generator for\
                subsampling the pairwise differences.
        .type = int
      tuning_param_opt = False
        .type = bool
        .help = If True, optimize the t-distribution's tuning parameter
      likelihood = normal *t-dist
        .help = Choice for likelihood function.
        .type = choice
        .multiple = False
      cc_after_pr = True
        .type = bool
        .help = If True - use correlation coefficient determined after post-refinement.\
                If False - use correlation coefficient determined before. \
                If post-refinement is not performed, must be False.
      constant_sadd = False
        .type = bool
        .help = If False, parameterize sadd with the correlation coefficient.\
                If True, use a constant sadd.
      do_diagnostics = False
        .type = bool
        .help = Make diagnostic plots.
    }
  }
  plot_single_index_histograms = False
    .type = bool
  set_average_unit_cell = True
    .type = bool
    .help = Output file adopts the unit cell of the data rather than of the reference model.
    .help = How is it determined?  Not a simple average, use a cluster-driven method for
    .help = deriving the best unit cell value.
  d_min = None
    .type = float
    .help = limiting resolution for scaling and merging
  d_max = None
    .type = float
    .help = limiting resolution for scaling and merging.  Implementation currently affects only the CCiso cal
  merge_anomalous = False
    .type = bool
    .help = Merge anomalous contributors
  include_multiplicity_column = False
    .type = bool
    .help = If True, save multiplicity to output mtz as separate column
}
"""

output_phil = """
output {
  expanded_bookkeeping = False
    .type = bool
    .help = if True, and if save_experiments_and_reflections=True, then include in the saved refl tabls:
    .help = 1- modified experiment identifier that contains the image number and lattice number
    .help = 2- index corresponding to the particular reflection in the input file (usually something_integrated.refl)
    .help = 3- the is_odd flag
    .help = 4- the original exp id for the reflection
    .help = 5- a unique number mapping the reflection to its input expFile, refFile pair (see output_dir/file_list_mapping.json)
  prefix = iobs
    .type = str
    .help = Prefix for all output file names
  title = None
    .type = str
    .help = Title for run - will appear in MTZ file header
  output_dir = .
    .type = str
    .help = output file directory
  tmp_dir = None
    .type = str
    .help = temporary file directory
  do_timing = False
    .type = bool
    .help = When True, calculate and log elapsed time for execution steps
  log_level = 1
    .type = int
    .help = determines how much information to log. Level 0 means: log all, while a non-zero level reduces the logging amount.
  save_experiments_and_reflections = False
    .type = bool
    .help = If True, dump the final set of experiments and reflections from the last worker
}
"""

statistics_phil = """
statistics {
  shuffle_ids = False
    .type = bool
    .help = shuffle the IDs when dividing into even/odd. This adds variation to half dataset stats like CC1/2
  n_bins = 10
    .type = int(value_min=1)
    .help = Number of resolution bins in statistics table
  cc1_2 {
    hash_filenames = False
      .type = bool
      .help = For CC1/2, instead of using odd/even filenames to split images into two sets,
      .help = hash the filename using md5 and split the images using odd/even hashes.
  }
  cciso {
    mtz_file = None
      .type = str
      .help = The isomorphous reference structure factors for Riso/ CCiso.
      .help = a fake file is written out to this file name if model is None
      .help = The experimental reference file can be old type *.mtz or new *sf.cif
    mtz_column_F = fobs
      .type = str
      .help = For Riso/ CCiso, the array name containing reference structure factors
      .help = As an example, could be intensity from *sf.cif
      .help = Specifically this is a tag searched for in the array name, could be 'tensity' from 'intensity'
  }
  deltaccint
    .help = Parameters used when computing ΔCC½ (aka ΔCC internal), a means of filtering out lattices that
    .help = degrade the overall CC½. Uses the σ-τ method from Assmann 2016 to avoid splitting the data into
    .help = odd/even datasets. Enable by adding the deltaccint worker after the group worker.
  {
    iqr_ratio = 10
      .type = float
      .help = If the ΔCC½ filter is enabled, first compute CC½ when removing every image one at a time, then
      .help = compute the IQR of these CC½s. Remove all lattices whose contribution degrades CC½ by more than
      .help = IQR * iqr_ratio above the median. You can discover a good IQR by running the program once,
      .help = examining the log file where possible values are listed, and running it again with the best value.
    verbose = False
      .type = bool
      .help = If True, include the ΔCC½ for every lattice in the main log.
  }
  predictions_to_edge {
    apply = False
      .type = bool
      .help = If True and key 'indices_to_edge' not found in integration pickles, predictions
      .help = will be made to the edge of the detector based on current unit cell, orientation,
      .help = and mosaicity.
    image = None
      .type = path
      .help = Path to an example image from which to extract active areas and pixel size.
    detector_phil = None
      .type = path
      .help = Path to the detector version phil file used to generate the selected data.
  }
  report_ML = True
    .type = bool
    .help = Report statistics on per-frame attributes modeled by max-likelihood fit (expert only).
  uc_precision = 2
    .type = int
    .help = Decimal places for unit cell statistics
}
"""

group_phil = """
parallel {
  a2a = 1
    .type = int
    .expert_level = 2
    .help = memory reduction factor for MPI alltoall.
    .help = Use a2a > 1, when available RAM is insufficient for doing MPI alltoall on all data at once.
    .help = The data will be split into a2a parts and, correspondingly, alltoall will be performed in a2a iterations.
}
"""

publish_phil = """
publish {
  include scope xfel.command_line.upload_mtz.phil_scope
}
"""

lunus_phil = """
lunus {
  deck_file = None
    .type = path
}
"""

diffbragg_phil = """
diffBragg {
  include scope simtbx.diffBragg.phil.phil_scope
}
"""

monitor_phil = """
monitor {
  detail = *rank node rank0 none
    .type = choice
    .help = Detail of data to be collected: from every rank, from rank 0 only,
    .help = from first rank on every node, or none.
  period = 5.0
    .type = float
    .help = Interval between subsequent resource statistics checks in seconds.
    .help = Short periods might lead to inconsistent logging.
  plot = True
    .type = bool
    .help = Plot the resource usage history after the monitor is stopped.
  prefix = monitor
    .type = str
    .help = Filename prefix for log files and summary plot.
  write = True
    .type = bool
    .help = Write collected resource information to log files.
}
"""


filter_global_phil = """
filter_global {
  intensity_extrema_iqr_dist_threshold = 1000.0
    .type = float(value_min=0, value_max=None)
    .help = Maximum tolerated deviation of max(intensity.sum.value)
    .help = and min(intensity.sum.value) from expts' population's respective
    .help = medians, expressed in population's interquartile range units.
}
"""


# A place to override any defaults included from elsewhere
program_defaults_phil_str = """
modify.cosym.use_curvatures=False
"""

master_phil = dispatch_phil + input_phil + tdata_phil + filter_phil + modify_phil + \
              select_phil + scaling_phil + postrefinement_phil + merging_phil + \
              output_phil + statistics_phil + group_phil + lunus_phil + \
              publish_phil + diffbragg_phil + monitor_phil + filter_global_phil

import os, importlib
custom_phil_pathstr = os.environ.get('XFEL_CUSTOM_WORKER_PATH')
if custom_phil_pathstr is not None and os.path.isdir(custom_phil_pathstr):
  for dir in os.listdir(custom_phil_pathstr):
    path = os.path.join(custom_phil_pathstr, dir, 'phil.py')
    if not os.path.isfile(path): continue
    spec = importlib.util.spec_from_file_location('_', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    master_phil += module.phil_str


phil_scope = parse(master_phil, process_includes = True)
phil_scope = phil_scope.fetch(parse(program_defaults_phil_str))

class Script(object):
  '''A class for running the script.'''

  def __init__(self):
    # The script usage
    import libtbx.load_env
    self.usage = "usage: %s [options] [param.phil] " % libtbx.env.dispatcher_name
    self.parser = None

  def initialize(self):
    '''Initialise the script.'''
    from dials.util.options import ArgumentParser
    # Create the parser
    self.parser = ArgumentParser(
      usage=self.usage,
      phil=phil_scope,
      epilog=help_message)
    self.parser.add_option(
        '--plots',
        action='store_true',
        default=False,
        dest='show_plots',
        help='Show some plots.')

    # Parse the command line. quick_parse is required for MPI compatibility
    params, options = self.parser.parse_args(show_diff_phil=True,quick_parse=True)
    self.params = params
    self.options = options

  def validate(self):
    from xfel.merging.application.validation.application import application
    application(self.params)

  def modify(self, experiments, reflections):
    return experiments, reflections #nop

  def run(self):
    print('''Initializing and validating phil...''')

    self.initialize()
    self.validate()

    # do other stuff
    return

if __name__ == '__main__':
  script = Script()
  result = script.run()
  print ("OK")
