/*
 ============================================================================
 Name        : nanoBraggCUDA.cu
 Author      :
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include "nanotypes.h"
#include "cuda_compatibility.h"
#include <simtbx/nanoBragg/nanoBraggCUDA.cuh>
using simtbx::nanoBragg::shapetype;
using simtbx::nanoBragg::hklParams;
using simtbx::nanoBragg::SQUARE;
using simtbx::nanoBragg::ROUND;
using simtbx::nanoBragg::GAUSS;
using simtbx::nanoBragg::GAUSS_ARGCHK;
using simtbx::nanoBragg::GAUSS_STAR;
using simtbx::nanoBragg::TOPHAT;

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define THREADS_PER_BLOCK_X 128
#define THREADS_PER_BLOCK_Y 1
#define THREADS_PER_BLOCK_TOTAL (THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y)
#define VECTOR_SIZE 4

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
	exit(1);
}

static cudaError_t cudaMemcpyVectorDoubleToDevice(CUDAREAL *dst, double *src, size_t vector_items) {
	CUDAREAL * temp = new CUDAREAL[vector_items];
	for (size_t i = 0; i < vector_items; i++) {
		temp[i] = src[i];
	}
	cudaError_t ret = cudaMemcpy(dst, temp, sizeof(*dst) * vector_items, cudaMemcpyHostToDevice);
	delete temp;
	return ret;
}

/* make a unit vector pointing in same direction and report magnitude (both args can be same vector) */
double cpu_unitize(double *vector, double *new_unit_vector);
double cpu_unitize(double * vector, double * new_unit_vector) {

	double v1 = vector[1];
	double v2 = vector[2];
	double v3 = vector[3];

	double mag = sqrt(v1 * v1 + v2 * v2 + v3 * v3);

	if (mag != 0.0) {
		/* normalize it */
		new_unit_vector[0] = mag;
		new_unit_vector[1] = v1 / mag;
		new_unit_vector[2] = v2 / mag;
		new_unit_vector[3] = v3 / mag;
	} else {
		/* can't normalize, report zero vector */
		new_unit_vector[0] = 0.0;
		new_unit_vector[1] = 0.0;
		new_unit_vector[2] = 0.0;
		new_unit_vector[3] = 0.0;
	}
	return mag;
}


__global__ void nanoBraggSpotsInitCUDAKernel(int spixels, int fpixesl, float * floatimage, float * omega_reduction, float * max_I_x_reduction,
		float * max_I_y_reduction, bool * rangemap);

__global__ void nanoBraggSpotsCUDAKernel(int spixels, int fpixels, int roi_xmin, int roi_xmax, int roi_ymin, int roi_ymax, int oversample, int point_pixel,
CUDAREAL pixel_size, CUDAREAL subpixel_size, int steps, CUDAREAL detector_thickstep, int detector_thicksteps, CUDAREAL detector_thick, CUDAREAL detector_mu,
		const CUDAREAL * __restrict__ sdet_vector, const CUDAREAL * __restrict__ fdet_vector, const CUDAREAL * __restrict__ odet_vector,
		const CUDAREAL * __restrict__ pix0_vector, int curved_detector, CUDAREAL distance, CUDAREAL close_distance, const CUDAREAL * __restrict__ beam_vector,
		CUDAREAL Xbeam, CUDAREAL Ybeam, CUDAREAL dmin, CUDAREAL phi0, CUDAREAL phistep, int phisteps, const CUDAREAL * __restrict__ spindle_vector, int sources,
		const CUDAREAL * __restrict__ source_X, const CUDAREAL * __restrict__ source_Y, const CUDAREAL * __restrict__ source_Z,
		const CUDAREAL * __restrict__ source_I, const CUDAREAL * __restrict__ source_lambda, const CUDAREAL * __restrict__ a0, const CUDAREAL * __restrict__ b0,
		const CUDAREAL * __restrict c0, shapetype xtal_shape, CUDAREAL mosaic_spread, int mosaic_domains, const CUDAREAL * __restrict__ mosaic_umats,
		CUDAREAL Na, CUDAREAL Nb,
		CUDAREAL Nc, CUDAREAL V_cell,
		CUDAREAL water_size, CUDAREAL water_F, CUDAREAL water_MW, CUDAREAL r_e_sqr, CUDAREAL fluence, CUDAREAL Avogadro, CUDAREAL spot_scale, int integral_form, CUDAREAL default_F,
		int interpolate, const CUDAREAL * __restrict__ Fhkl, const hklParams * __restrict__ Fhklparams, int nopolar, const CUDAREAL * __restrict__ polar_vector, CUDAREAL polarization, CUDAREAL fudge,
		const int unsigned short * __restrict__ maskimage, float * floatimage /*out*/, float * omega_reduction/*out*/, float * max_I_x_reduction/*out*/,
		float * max_I_y_reduction /*out*/, bool * rangemap);


extern "C" void nanoBraggSpotsCUDA(int deviceId, int spixels, int fpixels, int roi_xmin, int roi_xmax, int roi_ymin, int roi_ymax, int oversample, int point_pixel,
                double pixel_size, double subpixel_size, int steps, double detector_thickstep, int detector_thicksteps, double detector_thick, double detector_mu,
                double sdet_vector[4], double fdet_vector[4], double odet_vector[4], double pix0_vector[4], int curved_detector, double distance, double close_distance,
                double beam_vector[4], double Xbeam, double Ybeam, double dmin, double phi0, double phistep, int phisteps, double spindle_vector[4], int sources,
                double *source_X, double *source_Y, double * source_Z, double * source_I, double * source_lambda, double a0[4], double b0[4], double c0[4],
                shapetype xtal_shape, double mosaic_spread, int mosaic_domains, double * mosaic_umats, double Na, double Nb, double Nc, double V_cell,
                double water_size, double water_F, double water_MW, double r_e_sqr, double fluence, double Avogadro, int integral_form, double default_F,
                int interpolate, double *** Fhkl, int h_min, int h_max, int h_range, int k_min, int k_max, int k_range, int l_min, int l_max, int l_range, int hkls,
                int nopolar, double polar_vector[4], double polarization, double fudge, int unsigned short * maskimage, float * floatimage /*out*/,
                double * omega_sum/*out*/, int * sumn /*out*/, double * sum /*out*/, double * sumsqr /*out*/, double * max_I/*out*/, double * max_I_x/*out*/,
                double * max_I_y /*out*/, double spot_scale) {

	int total_pixels = spixels * fpixels;

    cudaSetDevice(deviceId);

	/*allocate and zero reductions */
	bool * rangemap = (bool*) calloc(total_pixels, sizeof(bool));
	float * omega_reduction = (float*) calloc(total_pixels, sizeof(float));
	float * max_I_x_reduction = (float*) calloc(total_pixels, sizeof(float));
	float * max_I_y_reduction = (float*) calloc(total_pixels, sizeof(float));

	/* clear memory (TODO: consider this being optional) */
	memset(floatimage, 0, sizeof(typeof(*floatimage)) * total_pixels);

	/*create transfer arguments to device space*/
	int cu_spixels = spixels, cu_fpixels = fpixels;
	int cu_roi_xmin = roi_xmin, cu_roi_xmax = roi_xmax, cu_roi_ymin = roi_ymin, cu_roi_ymax = roi_ymax;
	int cu_oversample = oversample;
	int cu_point_pixel = point_pixel;
	CUDAREAL cu_pixel_size = pixel_size, cu_subpixel_size = subpixel_size;
	int cu_steps = steps;
	CUDAREAL cu_detector_thickstep = detector_thickstep, cu_detector_thick = detector_thick, cu_detector_mu = detector_mu;
	int cu_detector_thicksteps = detector_thicksteps;
	int cu_curved_detector = curved_detector;

	CUDAREAL cu_distance = distance, cu_close_distance = close_distance;

	CUDAREAL cu_Xbeam = Xbeam, cu_Ybeam = Ybeam;
	CUDAREAL cu_dmin = dmin, cu_phi0 = phi0, cu_phistep = phistep;
	int cu_phisteps = phisteps;

	shapetype cu_xtal_shape = xtal_shape;

	int cu_sources = sources;

	CUDAREAL cu_mosaic_spread = mosaic_spread;
	int cu_mosaic_domains = mosaic_domains;

	CUDAREAL cu_Na = Na, cu_Nb = Nb, cu_Nc = Nc, cu_V_cell = V_cell, cu_water_size = water_size, cu_water_F = water_F, cu_water_MW = water_MW;
	CUDAREAL cu_r_e_sqr = r_e_sqr, cu_fluence = fluence, cu_Avogadro = Avogadro, cu_spot_scale = spot_scale;

	int cu_integral_form = integral_form;
	CUDAREAL cu_default_F = default_F;
	int cu_interpolate = interpolate;

//	int cu_h_min = h_min, cu_h_max = h_max, cu_h_range = h_range;
//	int cu_k_min = k_min, cu_k_max = k_max, cu_k_range = k_range;
//	int cu_l_min = l_min, cu_l_max = l_max, cu_l_range = l_range;
//	int cu_hkls = hkls;

	int cu_nopolar = nopolar;
	CUDAREAL cu_polarization = polarization, cu_fudge = fudge;

	hklParams FhklParams = { hkls, h_min, h_max, h_range, k_min, k_max, k_range, l_min, l_max, l_range };
	hklParams * cu_FhklParams;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_FhklParams, sizeof(*cu_FhklParams)));
	CUDA_CHECK_RETURN(cudaMemcpy(cu_FhklParams, &FhklParams, sizeof(*cu_FhklParams), cudaMemcpyHostToDevice));

	const int vector_length = 4;
	CUDAREAL * cu_sdet_vector;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_sdet_vector, sizeof(*cu_sdet_vector) * vector_length));
	CUDA_CHECK_RETURN(cudaMemcpyVectorDoubleToDevice(cu_sdet_vector, sdet_vector, vector_length));

	CUDAREAL * cu_fdet_vector;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_fdet_vector, sizeof(*cu_fdet_vector) * vector_length));
	CUDA_CHECK_RETURN(cudaMemcpyVectorDoubleToDevice(cu_fdet_vector, fdet_vector, vector_length));

	CUDAREAL * cu_odet_vector;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_odet_vector, sizeof(*cu_odet_vector) * vector_length));
	CUDA_CHECK_RETURN(cudaMemcpyVectorDoubleToDevice(cu_odet_vector, odet_vector, vector_length));

	CUDAREAL * cu_pix0_vector;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_pix0_vector, sizeof(*cu_pix0_vector) * vector_length));
	CUDA_CHECK_RETURN(cudaMemcpyVectorDoubleToDevice(cu_pix0_vector, pix0_vector, vector_length));

	CUDAREAL * cu_beam_vector;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_beam_vector, sizeof(*cu_beam_vector) * vector_length));
	CUDA_CHECK_RETURN(cudaMemcpyVectorDoubleToDevice(cu_beam_vector, beam_vector, vector_length));

	CUDAREAL * cu_spindle_vector;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_spindle_vector, sizeof(*cu_spindle_vector) * vector_length));
	CUDA_CHECK_RETURN(cudaMemcpyVectorDoubleToDevice(cu_spindle_vector, spindle_vector, vector_length));

	CUDAREAL * cu_a0;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_a0, sizeof(*cu_a0) * vector_length));
	CUDA_CHECK_RETURN(cudaMemcpyVectorDoubleToDevice(cu_a0, a0, vector_length));

	CUDAREAL * cu_b0;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_b0, sizeof(*cu_b0) * vector_length));
	CUDA_CHECK_RETURN(cudaMemcpyVectorDoubleToDevice(cu_b0, b0, vector_length));

	CUDAREAL * cu_c0;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_c0, sizeof(*cu_c0) * vector_length));
	CUDA_CHECK_RETURN(cudaMemcpyVectorDoubleToDevice(cu_c0, c0, vector_length));

	//	Unitize polar vector before sending it to the GPU. Optimization do it only once here rather than multiple time per pixel in the GPU.
	CUDAREAL * cu_polar_vector;
	double polar_vector_unitized[4];
	cpu_unitize(polar_vector, polar_vector_unitized);
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_polar_vector, sizeof(*cu_polar_vector) * vector_length));
	CUDA_CHECK_RETURN(cudaMemcpyVectorDoubleToDevice(cu_polar_vector, polar_vector_unitized, vector_length));

	CUDAREAL * cu_source_X = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_source_X, sizeof(*cu_source_X) * sources));
	CUDA_CHECK_RETURN(cudaMemcpyVectorDoubleToDevice(cu_source_X, source_X, sources));

	CUDAREAL * cu_source_Y = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_source_Y, sizeof(*cu_source_Y) * sources));
	CUDA_CHECK_RETURN(cudaMemcpyVectorDoubleToDevice(cu_source_Y, source_Y, sources));

	CUDAREAL * cu_source_Z = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_source_Z, sizeof(*cu_source_Z) * sources));
	CUDA_CHECK_RETURN(cudaMemcpyVectorDoubleToDevice(cu_source_Z, source_Z, sources));

	CUDAREAL * cu_source_I = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_source_I, sizeof(*cu_source_I) * sources));
	CUDA_CHECK_RETURN(cudaMemcpyVectorDoubleToDevice(cu_source_I, source_I, sources));

	CUDAREAL * cu_source_lambda = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_source_lambda, sizeof(*cu_source_lambda) * sources));
	CUDA_CHECK_RETURN(cudaMemcpyVectorDoubleToDevice(cu_source_lambda, source_lambda, sources));

	CUDAREAL * cu_mosaic_umats = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_mosaic_umats, sizeof(*cu_mosaic_umats) * mosaic_domains * 9));
	CUDA_CHECK_RETURN(cudaMemcpyVectorDoubleToDevice(cu_mosaic_umats, mosaic_umats, mosaic_domains * 9));

	float * cu_floatimage = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_floatimage, sizeof(*cu_floatimage) * total_pixels));
	CUDA_CHECK_RETURN(cudaMemcpy(cu_floatimage, floatimage, sizeof(*cu_floatimage) * total_pixels, cudaMemcpyHostToDevice));

	float * cu_omega_reduction = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_omega_reduction, sizeof(*cu_omega_reduction) * total_pixels));
	CUDA_CHECK_RETURN(cudaMemcpy(cu_omega_reduction, omega_reduction, sizeof(*cu_omega_reduction) * total_pixels, cudaMemcpyHostToDevice));

	float * cu_max_I_x_reduction = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_max_I_x_reduction, sizeof(*cu_max_I_x_reduction) * total_pixels));
	CUDA_CHECK_RETURN(cudaMemcpy(cu_max_I_x_reduction, max_I_x_reduction, sizeof(*cu_max_I_x_reduction) * total_pixels, cudaMemcpyHostToDevice));

	float * cu_max_I_y_reduction = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_max_I_y_reduction, sizeof(*cu_max_I_y_reduction) * total_pixels));
	CUDA_CHECK_RETURN(cudaMemcpy(cu_max_I_y_reduction, max_I_y_reduction, sizeof(*cu_max_I_y_reduction) * total_pixels, cudaMemcpyHostToDevice));

	bool * cu_rangemap = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_rangemap, sizeof(*cu_rangemap) * total_pixels));
	CUDA_CHECK_RETURN(cudaMemcpy(cu_rangemap, rangemap, sizeof(*cu_rangemap) * total_pixels, cudaMemcpyHostToDevice));

	int unsigned short * cu_maskimage = NULL;
	if (maskimage != NULL) {
		CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_maskimage, sizeof(*cu_maskimage) * total_pixels));
		CUDA_CHECK_RETURN(cudaMemcpy(cu_maskimage, maskimage, sizeof(*cu_maskimage) * total_pixels, cudaMemcpyHostToDevice));
	}

	int hklsize = h_range * k_range * l_range;
	CUDAREAL * FhklLinear = (CUDAREAL*) calloc(hklsize, sizeof(*FhklLinear));
	for (int h = 0; h < h_range; h++) {
		for (int k = 0; k < k_range; k++) {
//			memcpy(FhklLinear + (h * k_range * l_range + k * l_range), Fhkl[h][k], sizeof(*FhklLinear) * l_range);
			for (int l = 0; l < l_range; l++) {

				//	convert Fhkl double to CUDAREAL
				FhklLinear[h * k_range * l_range + k * l_range + l] = Fhkl[h][k][l];
			}
		}
	}

	CUDAREAL * cu_Fhkl = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_Fhkl, sizeof(*cu_Fhkl) * hklsize));
	CUDA_CHECK_RETURN(cudaMemcpy(cu_Fhkl, FhklLinear, sizeof(*cu_Fhkl) * hklsize, cudaMemcpyHostToDevice));
    free(FhklLinear);

	//int deviceId = 0;
	CUDA_CHECK_RETURN(cudaGetDevice(&deviceId));
	cudaDeviceProp deviceProps = { 0 };
	CUDA_CHECK_RETURN(cudaGetDeviceProperties(&deviceProps, deviceId));
	int smCount = deviceProps.multiProcessorCount;

//	CUDA_CHECK_RETURN(cudaFuncSetCacheConfig(nanoBraggSpotsCUDAKernel, cudaFuncCachePreferShared));
//	CUDA_CHECK_RETURN(cudaFuncSetCacheConfig(nanoBraggSpotsCUDAKernel, cudaFuncCachePreferL1));

	dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
	//  dim3 numBlocks((spixels - 1) / threadsPerBlock.x + 1, (fpixels - 1) / threadsPerBlock.y + 1);
	dim3 numBlocks(smCount * 8, 1);

	//  initialize the device memory within a kernel.
	//	nanoBraggSpotsInitCUDAKernel<<<numBlocks, threadsPerBlock>>>(cu_spixels, cu_fpixels, cu_floatimage, cu_omega_reduction, cu_max_I_x_reduction, cu_max_I_y_reduction, cu_rangemap);
	//  CUDA_CHECK_RETURN(cudaPeekAtLastError());
	//  CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	nanoBraggSpotsCUDAKernel<<<numBlocks, threadsPerBlock>>>(cu_spixels, cu_fpixels, cu_roi_xmin, cu_roi_xmax, cu_roi_ymin, cu_roi_ymax, cu_oversample,
			cu_point_pixel, cu_pixel_size, cu_subpixel_size, cu_steps, cu_detector_thickstep, cu_detector_thicksteps, cu_detector_thick, cu_detector_mu,
			cu_sdet_vector, cu_fdet_vector, cu_odet_vector, cu_pix0_vector, cu_curved_detector, cu_distance, cu_close_distance, cu_beam_vector,
			cu_Xbeam, cu_Ybeam, cu_dmin, cu_phi0, cu_phistep, cu_phisteps, cu_spindle_vector,
			cu_sources, cu_source_X, cu_source_Y, cu_source_Z, cu_source_I, cu_source_lambda, cu_a0, cu_b0, cu_c0, cu_xtal_shape,
			cu_mosaic_spread, cu_mosaic_domains, cu_mosaic_umats, cu_Na, cu_Nb, cu_Nc, cu_V_cell, cu_water_size, cu_water_F, cu_water_MW, cu_r_e_sqr, cu_fluence, 
			cu_Avogadro, cu_spot_scale, cu_integral_form, cu_default_F, cu_interpolate, cu_Fhkl, cu_FhklParams,
			cu_nopolar, cu_polar_vector, cu_polarization, cu_fudge, cu_maskimage,
			cu_floatimage /*out*/, cu_omega_reduction/*out*/, cu_max_I_x_reduction/*out*/, cu_max_I_y_reduction /*out*/, cu_rangemap /*out*/);

	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	CUDA_CHECK_RETURN(cudaMemcpy(floatimage, cu_floatimage, sizeof(*cu_floatimage) * total_pixels, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(omega_reduction, cu_omega_reduction, sizeof(*cu_omega_reduction) * total_pixels, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(max_I_x_reduction, cu_max_I_x_reduction, sizeof(*cu_max_I_x_reduction) * total_pixels, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(max_I_y_reduction, cu_max_I_y_reduction, sizeof(*cu_max_I_y_reduction) * total_pixels, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(rangemap, cu_rangemap, sizeof(*cu_rangemap) * total_pixels, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree(cu_sdet_vector));
	CUDA_CHECK_RETURN(cudaFree(cu_fdet_vector));
	CUDA_CHECK_RETURN(cudaFree(cu_odet_vector));
	CUDA_CHECK_RETURN(cudaFree(cu_pix0_vector));
	CUDA_CHECK_RETURN(cudaFree(cu_beam_vector));
	CUDA_CHECK_RETURN(cudaFree(cu_spindle_vector));
	CUDA_CHECK_RETURN(cudaFree(cu_polar_vector));
	CUDA_CHECK_RETURN(cudaFree(cu_a0));
	CUDA_CHECK_RETURN(cudaFree(cu_b0));
	CUDA_CHECK_RETURN(cudaFree(cu_c0));
	CUDA_CHECK_RETURN(cudaFree(cu_source_X));
	CUDA_CHECK_RETURN(cudaFree(cu_source_Y));
	CUDA_CHECK_RETURN(cudaFree(cu_source_Z));
	CUDA_CHECK_RETURN(cudaFree(cu_source_I));
	CUDA_CHECK_RETURN(cudaFree(cu_source_lambda));
	CUDA_CHECK_RETURN(cudaFree(cu_FhklParams));
	CUDA_CHECK_RETURN(cudaFree(cu_mosaic_umats));
	CUDA_CHECK_RETURN(cudaFree(cu_floatimage));
	CUDA_CHECK_RETURN(cudaFree(cu_omega_reduction));
	CUDA_CHECK_RETURN(cudaFree(cu_max_I_x_reduction));
	CUDA_CHECK_RETURN(cudaFree(cu_max_I_y_reduction));
	CUDA_CHECK_RETURN(cudaFree(cu_maskimage));
	CUDA_CHECK_RETURN(cudaFree(cu_rangemap));
    CUDA_CHECK_RETURN(cudaFree(cu_Fhkl));

	*max_I = 0;
	*max_I_x = 0;
	*max_I_y = 0;
	*sum = 0.0;
	*sumsqr = 0.0;
	*sumn = 0;
	*omega_sum = 0.0;

	for (int i = 0; i < total_pixels; i++) {
		if (!rangemap[i]) {
			continue;
		}
		float pixel = floatimage[i];
		if (pixel > (double) *max_I) {
			*max_I = pixel;
			*max_I_x = max_I_x_reduction[i];
			*max_I_y = max_I_y_reduction[i];
		}
		*sum += pixel;
		*sumsqr += pixel * pixel;
		++(*sumn);
		*omega_sum += omega_reduction[i];
	}
	free(rangemap);
	free(omega_reduction);
	free(max_I_x_reduction);
	free(max_I_y_reduction);
}

/* cubic spline interpolation functions */
__device__ static void polin2(CUDAREAL *x1a, CUDAREAL *x2a, CUDAREAL ya[4][4], CUDAREAL x1, CUDAREAL x2, CUDAREAL *y);
__device__ static void polin3(CUDAREAL *x1a, CUDAREAL *x2a, CUDAREAL *x3a, CUDAREAL ya[4][4][4], CUDAREAL x1, CUDAREAL x2, CUDAREAL x3, CUDAREAL *y);
/* rotate a 3-vector about a unit vector axis */
__device__ static CUDAREAL *rotate_axis(const CUDAREAL * __restrict__ v, CUDAREAL *newv, const CUDAREAL * __restrict__ axis, const CUDAREAL phi);
/* scale the magnitude of a vector */
__device__ static CUDAREAL vector_scale(CUDAREAL *vector, CUDAREAL *new_vector, CUDAREAL scale);
/* Fourier transform of a truncated lattice */
__device__ static CUDAREAL sincg(CUDAREAL x, CUDAREAL N);
//__device__ static CUDAREAL sincgrad(CUDAREAL x, CUDAREAL N);
/* Fourier transform of a sphere */
__device__ static CUDAREAL sinc3(CUDAREAL x);

__device__ __inline__ static int flatten3dindex(int x, int y, int z, int x_range, int y_range, int z_range);

__device__ __inline__ CUDAREAL quickFcell_ldg(int hkls, int h_max, int h_min, int k_max, int k_min, int l_min, int l_max, int h0, int k0, int l0, int h_range, int k_range, int l_range, CUDAREAL defaultF, const CUDAREAL * __restrict__ Fhkl);

__global__ void nanoBraggSpotsInitCUDAKernel(int spixels, int fpixels, float * floatimage, float * omega_reduction, float * max_I_x_reduction,
		float * max_I_y_reduction, bool * rangemap) {

	const int total_pixels = spixels * fpixels;
	const int fstride = gridDim.x * blockDim.x;
	const int sstride = gridDim.y * blockDim.y;
	const int stride = fstride * sstride;

	for (int pixIdx = (blockDim.y * blockIdx.y + threadIdx.y) * fstride + blockDim.x * blockIdx.x + threadIdx.x; pixIdx < total_pixels; pixIdx += stride) {
		const int fpixel = pixIdx % fpixels;
		const int spixel = pixIdx / fpixels;

		/* position in pixel array */
		int j = spixel * fpixels + fpixel;

		if (j < total_pixels) {
			floatimage[j] = 0;
			omega_reduction[j] = 0;
			max_I_x_reduction[j] = 0;
			max_I_y_reduction[j] = 0;
			rangemap[j] = false;
		}
	}
}

__global__ void nanoBraggSpotsCUDAKernel(int spixels, int fpixels, int roi_xmin, int roi_xmax, int roi_ymin, int roi_ymax, int oversample, int point_pixel,
CUDAREAL pixel_size, CUDAREAL subpixel_size, int steps, CUDAREAL detector_thickstep, int detector_thicksteps, CUDAREAL detector_thick, CUDAREAL detector_mu,
		const CUDAREAL * __restrict__ sdet_vector, const CUDAREAL * __restrict__ fdet_vector, const CUDAREAL * __restrict__ odet_vector,
		const CUDAREAL * __restrict__ pix0_vector, int curved_detector, CUDAREAL distance, CUDAREAL close_distance, const CUDAREAL * __restrict__ beam_vector,
		CUDAREAL Xbeam, CUDAREAL Ybeam, CUDAREAL dmin, CUDAREAL phi0, CUDAREAL phistep, int phisteps, const CUDAREAL * __restrict__ spindle_vector, int sources,
		const CUDAREAL * __restrict__ source_X, const CUDAREAL * __restrict__ source_Y, const CUDAREAL * __restrict__ source_Z,
		const CUDAREAL * __restrict__ source_I, const CUDAREAL * __restrict__ source_lambda, const CUDAREAL * __restrict__ a0, const CUDAREAL * __restrict__ b0,
		const CUDAREAL * __restrict c0, shapetype xtal_shape, CUDAREAL mosaic_spread, int mosaic_domains, const CUDAREAL * __restrict__ mosaic_umats,
		CUDAREAL Na, CUDAREAL Nb, CUDAREAL Nc, CUDAREAL V_cell, CUDAREAL water_size, CUDAREAL water_F, CUDAREAL water_MW, CUDAREAL r_e_sqr, CUDAREAL fluence,
		CUDAREAL Avogadro, CUDAREAL spot_scale, int integral_form, CUDAREAL default_F, int interpolate, const CUDAREAL * __restrict__ Fhkl, const hklParams * __restrict__ FhklParams, int nopolar, const CUDAREAL * __restrict__ polar_vector,
		CUDAREAL polarization, CUDAREAL fudge, const int unsigned short * __restrict__ maskimage, float * floatimage /*out*/, float * omega_reduction/*out*/,
		float * max_I_x_reduction/*out*/, float * max_I_y_reduction /*out*/, bool * rangemap) {

	__shared__ CUDAREAL s_dmin;

	__shared__ bool s_nopolar;

	__shared__ int s_phisteps;
	__shared__ CUDAREAL s_phi0, s_phistep;
	__shared__ int s_mosaic_domains;
	__shared__ CUDAREAL s_mosaic_spread;
	__shared__ shapetype s_xtal_shape;

	__shared__ CUDAREAL s_Na, s_Nb, s_Nc;
	__shared__ bool s_interpolate;
	__shared__ int s_hkls, s_h_max, s_h_min, s_k_max, s_k_min, s_l_max, s_l_min, s_h_range, s_k_range, s_l_range;

	if (threadIdx.x == 0 && threadIdx.y == 0) {

		s_dmin = dmin;

		s_nopolar = nopolar;

		s_phisteps = phisteps;
		s_phi0 = phi0;
		s_phistep = phistep;

		s_mosaic_domains = mosaic_domains;
		s_mosaic_spread = mosaic_spread;

		s_xtal_shape = xtal_shape;
		s_Na = Na;
		s_Nb = Nb;
		s_Nc = Nc;

		s_interpolate = interpolate;

		s_hkls = FhklParams->hkls;
		s_h_max = FhklParams->h_max;
		s_h_min = FhklParams->h_min;
		s_k_max = FhklParams->k_max;
		s_k_min = FhklParams->k_min;
		s_l_max = FhklParams->l_max;
		s_l_min = FhklParams->l_min;
		s_h_range = FhklParams->h_range;
		s_k_range = FhklParams->k_range;
		s_l_range = FhklParams->l_range;

	}
	__syncthreads();

	const int total_pixels = spixels * fpixels;
	const int fstride = gridDim.x * blockDim.x;
	const int sstride = gridDim.y * blockDim.y;
	const int stride = fstride * sstride;
//	const int tidx = blockDim.x * threadIdx.y * +threadIdx.x;

//	__shared__ int sharedVectors[THREADS_PER_BLOCK_TOTAL + 1][1][9];
//	__shared__ CUDAREAL sharedVectors[THREADS_PER_BLOCK_TOTAL + 1][1][VECTOR_SIZE];
//	CUDAREAL * tmpVector1 = sharedVectors[tidx][0];
//	CUDAREAL * tmpVector2 = sharedVectors[tidx][1];

	/* add background from something amorphous */
	CUDAREAL F_bg = water_F;
	CUDAREAL I_bg = F_bg * F_bg * r_e_sqr * fluence * water_size * water_size * water_size * 1e6 * Avogadro / water_MW;

//	hklParams[0] = h_min;
//	hklParams[1] = h_max;
//	hklParams[2] = h_range;
//	hklParams[3] = k_min;
//	hklParams[4] = k_max;
//	hklParams[5] = k_range;
//	hklParams[6] = l_min;
//	hklParams[7] = l_max;
//	hklParams[8] = l_range;

	for (int pixIdx = (blockDim.y * blockIdx.y + threadIdx.y) * fstride + blockDim.x * blockIdx.x + threadIdx.x; pixIdx < total_pixels; pixIdx += stride) {
		const int fpixel = pixIdx % fpixels;
		const int spixel = pixIdx / fpixels;

		/* allow for just one part of detector to be rendered */
		if (fpixel < roi_xmin || fpixel > roi_xmax || spixel < roi_ymin || spixel > roi_ymax) { //ROI region of interest
			continue;
		}

		/* position in pixel array */
		const int j = pixIdx;

		/* allow for the use of a mask */
		if (maskimage != NULL) {
			/* skip any flagged pixels in the mask */
			if (maskimage[j] == 0) {
				continue;
			}
		}

		/* reset photon count for this pixel */
		CUDAREAL I = I_bg;
		CUDAREAL omega_sub_reduction = 0.0;
		CUDAREAL max_I_x_sub_reduction = 0.0;
		CUDAREAL max_I_y_sub_reduction = 0.0;
		CUDAREAL polar = 0.0;
		if (s_nopolar) {
			polar = 1.0;
		}

		/* add this now to avoid problems with skipping later */
		// move this to the bottom to avoid accessing global device memory. floatimage[j] = I_bg;
		/* loop over sub-pixels */
		int subS, subF;
		for (subS = 0; subS < oversample; ++subS) { // Y voxel
			for (subF = 0; subF < oversample; ++subF) { // X voxel
				/* absolute mm position on detector (relative to its origin) */
				CUDAREAL Fdet = subpixel_size * (fpixel * oversample + subF) + subpixel_size / 2.0; // X voxel
				CUDAREAL Sdet = subpixel_size * (spixel * oversample + subS) + subpixel_size / 2.0; // Y voxel
				//                  Fdet = pixel_size*fpixel;
				//                  Sdet = pixel_size*spixel;

				max_I_x_sub_reduction = Fdet;
				max_I_y_sub_reduction = Sdet;

				int thick_tic;
				for (thick_tic = 0; thick_tic < detector_thicksteps; ++thick_tic) {
					/* assume "distance" is to the front of the detector sensor layer */
					CUDAREAL Odet = thick_tic * detector_thickstep; // Z Orthagonal voxel.

					/* construct detector subpixel position in 3D space */
					//                      pixel_X = distance;
					//                      pixel_Y = Sdet-Ybeam;
					//                      pixel_Z = Fdet-Xbeam;
					//CUDAREAL * pixel_pos = tmpVector1;
					CUDAREAL pixel_pos[4];
					pixel_pos[1] = Fdet * __ldg(&fdet_vector[1]) + Sdet * __ldg(&sdet_vector[1]) + Odet * __ldg(&odet_vector[1]) + __ldg(&pix0_vector[1]); // X
					pixel_pos[2] = Fdet * __ldg(&fdet_vector[2]) + Sdet * __ldg(&sdet_vector[2]) + Odet * __ldg(&odet_vector[2]) + __ldg(&pix0_vector[2]); // X
					pixel_pos[3] = Fdet * __ldg(&fdet_vector[3]) + Sdet * __ldg(&sdet_vector[3]) + Odet * __ldg(&odet_vector[3]) + __ldg(&pix0_vector[3]); // X
//					pixel_pos[1] = Fdet * fdet_vector[1] + Sdet * sdet_vector[1] + Odet * odet_vector[1] + pix0_vector[1]; // X
//					pixel_pos[2] = Fdet * fdet_vector[2] + Sdet * sdet_vector[2] + Odet * odet_vector[2] + pix0_vector[2]; // Y
//					pixel_pos[3] = Fdet * fdet_vector[3] + Sdet * sdet_vector[3] + Odet * odet_vector[3] + pix0_vector[3]; // Z
					if (curved_detector) {
						/* construct detector pixel that is always "distance" from the sample */
						CUDAREAL dbvector[4];
						dbvector[1] = distance * beam_vector[1];
						dbvector[2] = distance * beam_vector[2];
						dbvector[3] = distance * beam_vector[3];
						/* treat detector pixel coordinates as radians */
						CUDAREAL newvector[] = { 0.0, 0.0, 0.0, 0.0 };
						rotate_axis(dbvector, newvector, sdet_vector, pixel_pos[2] / distance);
						rotate_axis(newvector, pixel_pos, fdet_vector, pixel_pos[3] / distance);
						//                          rotate(vector,pixel_pos,0,pixel_pos[3]/distance,pixel_pos[2]/distance);
					}
					/* construct the diffracted-beam unit vector to this sub-pixel */
					//CUDAREAL * diffracted = tmpVector2;
					CUDAREAL diffracted[4];
					CUDAREAL airpath = unitize(pixel_pos, diffracted);

					/* solid angle subtended by a pixel: (pix/airpath)^2*cos(2theta) */
					CUDAREAL omega_pixel = pixel_size * pixel_size / airpath / airpath * close_distance / airpath;
					/* option to turn off obliquity effect, inverse-square-law only */
					if (point_pixel) {
						omega_pixel = 1.0 / airpath / airpath;
					}

					/* now calculate detector thickness effects */
					CUDAREAL capture_fraction = 1.0;
					if (detector_thick > 0.0 && detector_mu> 0.0) {
						/* inverse of effective thickness increase */
						CUDAREAL parallax = dot_product_ldg(odet_vector, diffracted);
						capture_fraction = exp(-thick_tic * detector_thickstep / detector_mu / parallax)
								- exp(-(thick_tic + 1) * detector_thickstep / detector_mu / parallax);
					}

					/* loop over sources now */
					int source;
					for (source = 0; source < sources; ++source) {

						/* retrieve stuff from cache */
						//CUDAREAL * incident = tmpVector1;
						CUDAREAL incident[4];
						incident[1] = -__ldg(&source_X[source]);
						incident[2] = -__ldg(&source_Y[source]);
						incident[3] = -__ldg(&source_Z[source]);
						CUDAREAL lambda = __ldg(&source_lambda[source]);
						CUDAREAL source_fraction = __ldg(&source_I[source]);

						/* construct the incident beam unit vector while recovering source distance */
						// TODO[Giles]: Optimization! We can unitize the source vectors before passing them in.
						unitize(incident, incident);
//						CUDAREAL source_path = unitize(incident, incident);
//						CUDAREAL source_path = norm3d(incident[1], incident[2], incident[3]);

//						CUDAREAL * d = tmpVector2;
//						d[0] = diffracted[0];
//						d[1] = diffracted[1];
//						d[2] = diffracted[2];
//						d[3] = diffracted[3];

						/* construct the scattering vector for this pixel */
//						CUDAREAL * scattering = tmpVector1;
						CUDAREAL scattering[4];
						scattering[1] = (diffracted[1] - incident[1]) / lambda;
						scattering[2] = (diffracted[2] - incident[2]) / lambda;
						scattering[3] = (diffracted[3] - incident[3]) / lambda;
//						CUDAREAL scattering[] = { 0.0, (diffracted[1] - incident[1]) / lambda, (diffracted[2] - incident[2]) / lambda, (diffracted[3]
//								- incident[3]) / lambda };

						/* sin(theta)/lambda is half the scattering vector length */
//						magnitude(scattering);
//						CUDAREAL stol = 0.5 * scattering[0];
						CUDAREAL stol = 0.5 * norm3d(scattering[1], scattering[2], scattering[3]);

						/* rough cut to speed things up when we aren't using whole detector */
						if (s_dmin > 0.0 && stol > 0.0) {
							if (s_dmin > 0.5 / stol) {
								continue;
							}
						}

						/* polarization factor */
						if (!s_nopolar) {
							/* need to compute polarization factor */
							polar = polarization_factor(polarization, incident, diffracted, polar_vector);
						} else {
							polar = 1.0;
						}

						/* sweep over phi angles */
						for (int phi_tic = 0; phi_tic < s_phisteps; ++phi_tic) {
							CUDAREAL phi = s_phistep * phi_tic + s_phi0;

//							CUDAREAL ap[] = { 0.0, 0.0, 0.0, 0.0 };
//							CUDAREAL bp[] = { 0.0, 0.0, 0.0, 0.0 };
//							CUDAREAL cp[] = { 0.0, 0.0, 0.0, 0.0 };
							CUDAREAL ap[4];
							CUDAREAL bp[4];
							CUDAREAL cp[4];

							/* rotate about spindle if necessary */
							rotate_axis_ldg(a0, ap, spindle_vector, phi);
							rotate_axis_ldg(b0, bp, spindle_vector, phi);
							rotate_axis_ldg(c0, cp, spindle_vector, phi);

							/* enumerate mosaic domains */
							for (int mos_tic = 0; mos_tic < s_mosaic_domains; ++mos_tic) {
								/* apply mosaic rotation after phi rotation */
								CUDAREAL a[4];
								CUDAREAL b[4];
								CUDAREAL c[4];
								if (s_mosaic_spread > 0.0) {
									rotate_umat_ldg(ap, a, &mosaic_umats[mos_tic * 9]);
									rotate_umat_ldg(bp, b, &mosaic_umats[mos_tic * 9]);
									rotate_umat_ldg(cp, c, &mosaic_umats[mos_tic * 9]);
								} else {
									a[1] = ap[1];
									a[2] = ap[2];
									a[3] = ap[3];
									b[1] = bp[1];
									b[2] = bp[2];
									b[3] = bp[3];
									c[1] = cp[1];
									c[2] = cp[2];
									c[3] = cp[3];
								}
								//                                  printf("%d %f %f %f\n",mos_tic,mosaic_umats[mos_tic*9+0],mosaic_umats[mos_tic*9+1],mosaic_umats[mos_tic*9+2]);
								//                                  printf("%d %f %f %f\n",mos_tic,mosaic_umats[mos_tic*9+3],mosaic_umats[mos_tic*9+4],mosaic_umats[mos_tic*9+5]);
								//                                  printf("%d %f %f %f\n",mos_tic,mosaic_umats[mos_tic*9+6],mosaic_umats[mos_tic*9+7],mosaic_umats[mos_tic*9+8]);

								/* construct fractional Miller indicies */

//								CUDAREAL * scat_s = tmpVector2;
//								scat_s[0] = scattering[0];
//								scat_s[1] = scattering[1];
//								scat_s[2] = scattering[2];
//								scat_s[3] = scattering[3];
//
//								CUDAREAL h = dot_product(a, scat_s);
//								CUDAREAL k = dot_product(b, scat_s);
//								CUDAREAL l = dot_product(c, scat_s);
								CUDAREAL h = dot_product(a, scattering);
								CUDAREAL k = dot_product(b, scattering);
								CUDAREAL l = dot_product(c, scattering);

								/* round off to nearest whole index */
								int h0 = ceil(h - 0.5);
								int k0 = ceil(k - 0.5);
								int l0 = ceil(l - 0.5);

								/* structure factor of the lattice (paralelpiped crystal)
								 F_latt = sin(M_PI*Na*h)*sin(M_PI*Nb*k)*sin(M_PI*Nc*l)/sin(M_PI*h)/sin(M_PI*k)/sin(M_PI*l);
								 */
								CUDAREAL F_latt = 1.0; // Shape transform for the crystal.
								CUDAREAL hrad_sqr = 0.0;
								if (s_xtal_shape == SQUARE) {
									/* xtal is a paralelpiped */
									if (Na > 1) {
//										F_latt *= sincgrad(h, s_Na);
										F_latt *= sincg(M_PI * h, s_Na);
									}
									if (Nb > 1) {
//										F_latt *= sincgrad(k, s_Nb);
										F_latt *= sincg(M_PI * k, s_Nb);
									}
									if (Nc > 1) {
//										F_latt *= sincgrad(l, s_Nc);
										F_latt *= sincg(M_PI * l, s_Nc);
									}
								} else {
									/* handy radius in reciprocal space, squared */
									hrad_sqr = (h - h0) * (h - h0) * Na * Na + (k - k0) * (k - k0) * Nb * Nb + (l - l0) * (l - l0) * Nc * Nc;
								}
								if (s_xtal_shape == ROUND) {
									/* use sinc3 for elliptical xtal shape,
									 correcting for sqrt of volume ratio between cube and sphere */
									F_latt = Na * Nb * Nc * 0.723601254558268 * sinc3(M_PI * sqrt(hrad_sqr * fudge));
								}
								if (s_xtal_shape == GAUSS) {
									/* fudge the radius so that volume and FWHM are similar to square_xtal spots */
									F_latt = Na * Nb * Nc * exp(-(hrad_sqr / 0.63 * fudge));
								}
                                if (s_xtal_shape == GAUSS_ARGCHK) {
                                   /* fudge the radius so that volume and FWHM are similar to square_xtal spots */
                                   double my_arg = hrad_sqr / 0.63 * fudge;
                                   if (my_arg<35.){ F_latt = Na * Nb * Nc * exp(-(my_arg));
                                   } else { F_latt = 0.; } // warps coalesce when blocks of 32 pixels have no Bragg signal
                                }
                                if (s_xtal_shape == GAUSS_STAR){
                                   CUDAREAL a_cross_b[] = { 0.0, 0.0, 0.0, 0.0 };
                                   CUDAREAL b_cross_c[] = { 0.0, 0.0, 0.0, 0.0 };
                                   CUDAREAL c_cross_a[] = { 0.0, 0.0, 0.0, 0.0 };
                                   cross_product(a,b,a_cross_b);
                                   cross_product(b,c,b_cross_c);
                                   cross_product(c,a,c_cross_a);
                                   /* new reciprocal-space cell vectors */
                                   CUDAREAL a_star_tic[] = {0,0,0,0};
                                   CUDAREAL b_star_tic[] = {0,0,0,0};
                                   CUDAREAL c_star_tic[] = {0,0,0,0};
                                   vector_scale(b_cross_c,a_star_tic,1e20/V_cell);
                                   vector_scale(c_cross_a,b_star_tic,1e20/V_cell);
                                   vector_scale(a_cross_b,c_star_tic,1e20/V_cell);
                                   //}
                                   CUDAREAL dh=h-h0;
                                   CUDAREAL dk=k-k0;
                                   CUDAREAL dl=l-l0;
                                   CUDAREAL dx_star = dh*a_star_tic[1] + dk*b_star_tic[1] + dl*c_star_tic[1];
                                   CUDAREAL dy_star = dh*a_star_tic[2] + dk*b_star_tic[2] + dl*c_star_tic[2];
                                   CUDAREAL dz_star = dh*a_star_tic[3] + dk*b_star_tic[3] + dl*c_star_tic[3];
                                   CUDAREAL Nvol=Na*Nb*Nc;
                                   CUDAREAL xtal_size_sq = pow(Nvol*V_cell, CUDAREAL(2)/CUDAREAL(3));
                                   CUDAREAL rad_star_sqr = ( dx_star*dx_star + dy_star*dy_star + dz_star*dz_star )
                                                  *xtal_size_sq;
                                   F_latt = Nvol*exp(-( rad_star_sqr *1.9 * fudge ));
                                }
								if (s_xtal_shape == TOPHAT) {
									/* make a flat-top spot of same height and volume as square_xtal spots */
									F_latt = Na * Nb * Nc * (hrad_sqr * fudge < 0.3969);
								}
								/* no need to go further if result will be zero? */
								if (F_latt == 0.0 && water_size == 0.0)
									continue;

								/* find nearest point on Ewald sphere surface? */
								if (integral_form) {

									/* need to calculate reciprocal matrix */
									/* various cross products */
									CUDAREAL a_cross_b[] = { 0.0, 0.0, 0.0, 0.0 };
									CUDAREAL b_cross_c[] = { 0.0, 0.0, 0.0, 0.0 };
									CUDAREAL c_cross_a[] = { 0.0, 0.0, 0.0, 0.0 };
									cross_product(a, b, a_cross_b);
									cross_product(b, c, b_cross_c);
									cross_product(c, a, c_cross_a);

									/* new reciprocal-space cell vectors */
									CUDAREAL a_star[] = { 0.0, 0.0, 0.0, 0.0 };
									CUDAREAL b_star[] = { 0.0, 0.0, 0.0, 0.0 };
									CUDAREAL c_star[] = { 0.0, 0.0, 0.0, 0.0 };
									vector_scale(b_cross_c, a_star, 1e20 / V_cell);
									vector_scale(c_cross_a, b_star, 1e20 / V_cell);
									vector_scale(a_cross_b, c_star, 1e20 / V_cell);

									/* reciprocal-space coordinates of nearest relp */
									CUDAREAL relp[] = { 0.0, 0.0, 0.0, 0.0 };
									relp[1] = h0 * a_star[1] + k0 * b_star[1] + l0 * c_star[1];
									relp[2] = h0 * a_star[2] + k0 * b_star[2] + l0 * c_star[2];
									relp[3] = h0 * a_star[3] + k0 * b_star[3] + l0 * c_star[3];
									//                                      d_star = magnitude(relp)

									/* reciprocal-space coordinates of center of Ewald sphere */
									CUDAREAL Ewald0[] = { 0.0, 0.0, 0.0, 0.0 };
									Ewald0[1] = -incident[1] / lambda / 1e10;
									Ewald0[2] = -incident[2] / lambda / 1e10;
									Ewald0[3] = -incident[3] / lambda / 1e10;
									//                                      1/lambda = magnitude(Ewald0)

									/* distance from Ewald sphere in lambda=1 units */
									CUDAREAL dEwald0[] = { 0.0, 0.0, 0.0, 0.0 };
									dEwald0[1] = relp[1] - Ewald0[1];
									dEwald0[2] = relp[2] - Ewald0[2];
									dEwald0[3] = relp[3] - Ewald0[3];
									magnitude(dEwald0);
									CUDAREAL d_r = dEwald0[0] - 1.0;

									/* unit vector of diffracted ray through relp */
									CUDAREAL diffracted0[] = { 0.0, 0.0, 0.0, 0.0 };
									unitize(dEwald0, diffracted0);

									/* intersection with detector plane */
									CUDAREAL xd = dot_product_ldg(fdet_vector, diffracted0);
									CUDAREAL yd = dot_product_ldg(sdet_vector, diffracted0);
									CUDAREAL zd = dot_product_ldg(odet_vector, diffracted0);

									/* where does the central direct-beam hit */
									CUDAREAL xd0 = dot_product_ldg(fdet_vector, incident);
									CUDAREAL yd0 = dot_product_ldg(sdet_vector, incident);
									CUDAREAL zd0 = dot_product_ldg(odet_vector, incident);

									/* convert to mm coordinates */
									CUDAREAL Fdet0 = distance * (xd / zd) + Xbeam;
									CUDAREAL Sdet0 = distance * (yd / zd) + Ybeam;

									//printf("GOTHERE %g %g   %g %g\n",Fdet,Sdet,Fdet0,Sdet0);
									CUDAREAL test = exp(-((Fdet - Fdet0) * (Fdet - Fdet0) + (Sdet - Sdet0) * (Sdet - Sdet0) + d_r * d_r) / 1e-8);
								} // end of integral form

								/* structure factor of the unit cell */
								CUDAREAL F_cell = default_F;
								if (s_interpolate) {
									int h0_flr = floor(h);
									int k0_flr = floor(k);
									int l0_flr = floor(l);

									if (((h - s_h_min + 3) > s_h_range) || (h - 2 < s_h_min) || ((k - s_k_min + 3) > s_k_range) || (k - 2 < s_k_min)
											|| ((l - s_l_min + 3) > s_l_range) || (l - 2 < s_l_min)) {
										//											if (babble) {
										//												babble = 0;
										//												printf("WARNING: out of range for three point interpolation: h,k,l,h0,k0,l0: %g,%g,%g,%d,%d,%d \n", h, k, l, h0,
										//														k0, l0);
										//												printf("WARNING: further warnings will not be printed! ");
										//											}
										F_cell = quickFcell_ldg(s_hkls, s_h_max, s_h_min, s_k_max, s_k_min, s_l_max, s_l_min, h0, k0, l0, s_h_range, s_k_range, s_l_range, default_F, Fhkl);
									} else {
										/* integer versions of nearest HKL indicies */
										int h_interp[] = { 0, 0, 0, 0 };
										int k_interp[] = { 0, 0, 0, 0 };
										int l_interp[] = { 0, 0, 0, 0 };
										h_interp[0] = h0_flr - 1;
										h_interp[1] = h0_flr;
										h_interp[2] = h0_flr + 1;
										h_interp[3] = h0_flr + 2;
										k_interp[0] = k0_flr - 1;
										k_interp[1] = k0_flr;
										k_interp[2] = k0_flr + 1;
										k_interp[3] = k0_flr + 2;
										l_interp[0] = l0_flr - 1;
										l_interp[1] = l0_flr;
										l_interp[2] = l0_flr + 1;
										l_interp[3] = l0_flr + 2;

										/* polin function needs doubles */
										CUDAREAL h_interp_d[] = { 0.0, 0.0, 0.0, 0.0 };
										CUDAREAL k_interp_d[] = { 0.0, 0.0, 0.0, 0.0 };
										CUDAREAL l_interp_d[] = { 0.0, 0.0, 0.0, 0.0 };
										h_interp_d[0] = (CUDAREAL) h_interp[0];
										h_interp_d[1] = (CUDAREAL) h_interp[1];
										h_interp_d[2] = (CUDAREAL) h_interp[2];
										h_interp_d[3] = (CUDAREAL) h_interp[3];
										k_interp_d[0] = (CUDAREAL) k_interp[0];
										k_interp_d[1] = (CUDAREAL) k_interp[1];
										k_interp_d[2] = (CUDAREAL) k_interp[2];
										k_interp_d[3] = (CUDAREAL) k_interp[3];
										l_interp_d[0] = (CUDAREAL) l_interp[0];
										l_interp_d[1] = (CUDAREAL) l_interp[1];
										l_interp_d[2] = (CUDAREAL) l_interp[2];
										l_interp_d[3] = (CUDAREAL) l_interp[3];

										/* now populate the "y" values (nearest four structure factors in each direction) */
										CUDAREAL sub_Fhkl[4][4][4];
										int i1, i2, i3;
										for (i1 = 0; i1 < 4; i1++) {
											for (i2 = 0; i2 < 4; i2++) {
												for (i3 = 0; i3 < 4; i3++) {
													sub_Fhkl[i1][i2][i3] = __ldg(
															&Fhkl[flatten3dindex(h_interp[i1] - s_h_min, k_interp[i2] - s_k_min, l_interp[i3] - s_l_min, s_h_range,
																	s_k_range, s_l_range)]);
												}
											}
										}

										/* run the tricubic polynomial interpolation */
										polin3(h_interp_d, k_interp_d, l_interp_d, sub_Fhkl, h, k, l, &F_cell);
									}
								} else {
//								if (!interpolate) {
//									if (hkls && (h0 <= hklParams[1]) && (h0 >= hklParams[0]) && (k0 <= hklParams[4]) && (k0 >= hklParams[3]) && (l0 <= hklParams[7]) && (l0 >= hklParams[6])) {
//										/* just take nearest-neighbor */
//										F_cell = __ldg(&Fhkl[flatten3dindex(h0 - hklParams[0], k0 - hklParams[3], l0 - hklParams[6], hklParams[2], hklParams[5], hklParams[8])]);
//									} else {
//										F_cell = default_F;  // usually zero
//									}
//								}
									F_cell = quickFcell_ldg(s_hkls, s_h_max, s_h_min, s_k_max, s_k_min, s_l_max, s_l_min, h0, k0, l0, s_h_range, s_k_range, s_l_range, default_F, Fhkl);
//									if (s_hkls && (h0 <= s_h_max) && (h0 >= s_h_min) && (k0 <= s_k_max) && (k0 >= s_k_min) && (l0 <= s_l_max) && (l0 >= s_l_min)) {
//										/* just take nearest-neighbor */
//										F_cell = __ldg(&Fhkl[flatten3dindex(h0 - s_h_min, k0 - s_k_min, l0 - s_l_min, s_h_range, s_k_range, s_l_range)]);
////										F_cell = __ldg(&Fhkl[flatten3dindex(h0 - __ldg(&FhklParams->h_min), k0 - __ldg(&FhklParams->k_min), l0 - __ldg(&FhklParams->l_min), s_h_range, s_k_range, s_l_range)]);
////										F_cell = __ldg(&Fhkl[flatten3dindex(h0 - FhklParams->h_min, k0 - FhklParams->k_min, l0 - FhklParams->l_min, FhklParams->h_range, FhklParams->k_range, FhklParams->l_range)]);
//									}
								}

								/* now we have the structure factor for this pixel */

								/* convert amplitudes into intensity (photons per steradian) */
								I += F_cell * F_cell * F_latt * F_latt * source_fraction * capture_fraction * omega_pixel;
								omega_sub_reduction += omega_pixel;
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
		const double photons = I_bg + (r_e_sqr * spot_scale * fluence * polar * I) / steps;
		floatimage[j] = photons;
		omega_reduction[j] = omega_sub_reduction; // shared contention
		max_I_x_reduction[j] = max_I_x_sub_reduction;
		max_I_y_reduction[j] = max_I_y_sub_reduction;
		rangemap[j] = true;
	}
}

__device__ __inline__ CUDAREAL quickFcell_ldg(int hkls, int h_max, int h_min, int k_max, int k_min, int l_max, int l_min, int h0, int k0, int l0, int h_range, int k_range, int l_range, CUDAREAL defaultF, const CUDAREAL * __restrict__ Fhkl) {
	if (hkls && (h0 <= h_max) && (h0 >= h_min) && (k0 <= k_max) && (k0 >= k_min) && (l0 <= l_max) && (l0 >= l_min)) {
		/* just take nearest-neighbor */
//      F_cell = __ldg(&Fhkl[flatten3dindex(h0 - s_h_min, k0 - s_k_min, l0 - s_l_min, s_h_range, s_k_range, s_l_range)]);
		return __ldg(&Fhkl[flatten3dindex(h0 - h_min, k0 - k_min, l0 - l_min, h_range, k_range, l_range)]);
	} else {
		return defaultF;  // usually zero
	}
}

__device__ __inline__ int flatten3dindex(int x, int y, int z, int x_range, int y_range, int z_range) {
	return x * y_range * z_range + y * z_range + z;
}

/* rotate a point about a unit vector axis */
__device__ CUDAREAL *rotate_axis(const CUDAREAL * __restrict__ v, CUDAREAL * newv, const CUDAREAL * __restrict__ axis, const CUDAREAL phi) {

	const CUDAREAL sinphi = sin(phi);
	const CUDAREAL cosphi = cos(phi);
	const CUDAREAL a1 = axis[1];
	const CUDAREAL a2 = axis[2];
	const CUDAREAL a3 = axis[3];
	const CUDAREAL v1 = v[1];
	const CUDAREAL v2 = v[2];
	const CUDAREAL v3 = v[3];
	const CUDAREAL dot = (a1 * v1 + a2 * v2 + a3 * v3) * (1.0 - cosphi);

	newv[1] = a1 * dot + v1 * cosphi + (-a3 * v2 + a2 * v3) * sinphi;
	newv[2] = a2 * dot + v2 * cosphi + (+a3 * v1 - a1 * v3) * sinphi;
	newv[3] = a3 * dot + v3 * cosphi + (-a2 * v1 + a1 * v2) * sinphi;

	return newv;
}

/* scale magnitude of provided vector */
__device__ CUDAREAL vector_scale(CUDAREAL *vector, CUDAREAL *new_vector, CUDAREAL scale) {

	new_vector[1] = scale * vector[1];
	new_vector[2] = scale * vector[2];
	new_vector[3] = scale * vector[3];
	magnitude(new_vector);

	return new_vector[0];
}

/* Fourier transform of a grating */
__device__ CUDAREAL sincg(CUDAREAL x, CUDAREAL N) {
	if (x != 0.0)
		return sin(x * N) / sin(x);

	return N;

}

__device__ CUDAREAL sincgrad(CUDAREAL x, CUDAREAL N) {
	if (x != 0.0)
		return sinpi(x * N) / sinpi(x);

	return N;
}

/* Fourier transform of a sphere */
__device__ CUDAREAL sinc3(CUDAREAL x) {
	if (x != 0.0)
		return 3.0 * (sin(x) / x - cos(x)) / (x * x);

	return 1.0;

}

__device__ void polin2(CUDAREAL *x1a, CUDAREAL *x2a, CUDAREAL ya[4][4], CUDAREAL x1, CUDAREAL x2, CUDAREAL *y) {
	int j;
	CUDAREAL ymtmp[4];
	for (j = 1; j <= 4; j++) {
		polint(x2a, ya[j - 1], x2, &ymtmp[j - 1]);
	}
	polint(x1a, ymtmp, x1, y);
}

__device__ void polin3(CUDAREAL *x1a, CUDAREAL *x2a, CUDAREAL *x3a, CUDAREAL ya[4][4][4], CUDAREAL x1, CUDAREAL x2, CUDAREAL x3, CUDAREAL *y) {
	int j;
	CUDAREAL ymtmp[4];

	for (j = 1; j <= 4; j++) {
		polin2(x2a, x3a, &ya[j - 1][0], x2, x3, &ymtmp[j - 1]);
	}
	polint(x1a, ymtmp, x1, y);
}


