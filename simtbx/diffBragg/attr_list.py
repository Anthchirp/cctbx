from __future__ import division

"""
critical properties of diffBragg objects which should be logged for reproducibility
"""
# TODO :  implement a savestate and getstate for these objects

# attrs of diffBragg() instances
DIFFBRAGG_ATTRS = [
 'Amatrix',
 'Bmatrix',
 'Ncells_abc',
 'Ncells_abc_aniso',
 'Ncells_def',
 'Npix_to_allocate',
 'Omatrix',
 'Umatrix',
 'beamsize_mm',
 'compute_curvatures',
 'default_F',
 'detector_thick_mm',
 'detector_thickstep_mm',
 'detector_thicksteps',
 'detector_twotheta_deg',
 'device_Id',
 'diffuse_gamma',
 'diffuse_sigma',
 'exposure_s',
 'fluence',
 'flux',
 'has_anisotropic_mosaic_spread',
 'host_transfer',
 'interpolate',
 'isotropic_ncells',
 'lambda_coefficients',
 'mosaic_domains',
 'mosaic_spread_deg',
 'no_Nabc_scale',
 'nopolar',
 'only_diffuse',
 'only_save_omega_kahn',
 'oversample',
 'oversample_omega',
 'phi_deg',
 'phistep_deg',
 'phisteps',
 'point_pixel',
 'polar_vector',
 'polarization',
 'spindle_axis',
 'spot_scale',
 'twotheta_axis',
 'unit_cell_Adeg',
 'unit_cell_tuple',
 'use_diffuse',
 'use_lambda_coefficients']


# properties of nanoBragg_crystal.NBcryst instances
NB_CRYST_ATTRS = [
 'anisotropic_mos_spread_deg',
 'isotropic_ncells',
 'miller_is_complex',
 'mos_spread_deg',
 'n_mos_domains',
 'symbol',
 'xtal_shape']


# properties of nanoBragg_beam.NBbeam instances
NB_BEAM_ATTRS = [
 'divergence_mrad',
 'divsteps',
 'polarization_fraction',
 'size_mm',
 'number_of_sources',
 'unit_s0']
