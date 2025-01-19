/*
 * nanotypes.h
 *
 *  Created on: Jan 2, 2018
 *      Author: giles
 */

#ifndef NANOTYPES_H_
#define NANOTYPES_H_

namespace simtbx {
namespace nanoBragg {

struct hklParams {
        int hkls;
        int h_min;
        int h_max;
        int h_range;
        int k_min;
        int k_max;
        int k_range;
        int l_min;
        int l_max;
        int l_range;
};

typedef enum { SAMPLE, BEAM } pivot;
typedef enum { UNKNOWN, SQUARE, ROUND, GAUSS, GAUSS_ARGCHK, TOPHAT, FIBER, GAUSS_STAR } shapetype;
// GAUSS_ARGCHK provides a lightweight backdoor for efficient implementation of
// the GAUSS shapetype in CUDA. Precaluates the argument of the exponential.
typedef enum { CUSTOM, ADXV, MOSFLM, XDS, DIALS, DENZO } convention;

}}
#endif /* NANOTYPES_H_ */
