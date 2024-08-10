// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#ifndef MATH_UTILS_H_
#define MATH_UTILS_H_

int clz(uint b)
{
	return 31 - findMSB(b);
}

const int TAB_BITS_IN = 8; // 256 entry LUT.
const int TAB_BITS_OUT = 9; // 9 bits effective, 8 bits stored

uint umul32_hi(uint a, uint b)
{
	uint msb, lsb;
	umulExtended(a, b, msb, lsb);
	return msb;
}

#ifdef USE_RCP_FIXED
// ~23.99 bits of precision, basically perfect result for FP32.
// Not quite perfect 0.5 ULP because of initial rounding step to 32-bit, then round to 24-bit.
float rcp_fixed(uint nb, int exp_bias)
{
	// Based on solution in https://stackoverflow.com/a/66060171.
	uint idx = (nb >> (32u - 1u - TAB_BITS_IN)) & 0xffu;
	uint rcp = texelFetch(RCPLut, int(idx)).x | 0x100u; // add implicit msb
	// f = 2.0 * xn * xn
	uint f = (rcp * rcp) << (32u - 2u * TAB_BITS_OUT);
	// p = 0.5 * [(2.0 * xn * xn) * (a)] = xn * xn * a
	uint p = umul32_hi(f, nb);
	// rcp = 2.0 * xn - p = xn1
	rcp = (rcp << (32u - TAB_BITS_OUT)) - p;

	// rcp = 2.0 * xn1
	rcp = rcp << 1u;
	// p = 0.5 * [(2.0 * xn1) * (a)] = xn1 * a
	p = umul32_hi(rcp, nb);
	// Negating p is 2.0 - p due to fixed point.
	// rcp = xn2 = 0.5 * [2.0 * xn1 * (2.0 - xn1 * a)] = 2.0 * xn1 - xn1 * xn1 * a
	rcp = umul32_hi(rcp, -p);

	// Quantize to FP32.
	rcp = (rcp + 64u) >> 7u;
	rcp -= 1u << 23u;
	int exp = 127 - 32 + exp_bias;
	rcp += exp << 23;

	return uintBitsToFloat(rcp);
}

float rcp_fixed(uint b)
{
	int lzb = clz(b);
	uint nb = b << lzb;
	return rcp_fixed(nb, lzb);
}
#endif

#ifdef USE_RCP_FLOAT
float newton_rhapson_step(float xn, float a)
{
	precise float xn2 = xn * 2.0;
	precise float xn_sqr = xn * xn;
	precise float xn_sqr_a = xn_sqr * a;
	precise float result = xn2 - xn_sqr_a;
	return result;
}

float rcp_float(float v)
{
	uint u32 = floatBitsToUint(v);
	int exp = int(bitfieldExtract(u32, 23, 8));

	// Handle divide by subnormal or 0.
	if (exp == 0)
		return uintBitsToFloat(0x7f800000);

	int shifted_exp = 127 - exp;

	// Initialize division estimate. 1 KiB LUT. Results in (0.5, 1.0) range.
	uint idx = int(bitfieldExtract(u32, 23 - 8, 8));
	float xn0 = texelFetch(RCPLutFloat, int(idx)).x;

	// Extract manitissa.
	u32 &= (1u << 23) - 1u;
	// Normalize exponent to 0. Inputs in [1.0, 2.0) range.
	u32 |= 127u << 23u;
	float a = uintBitsToFloat(u32);
	// 4 iterations is probably overkill, but perf isn't super important since this is done once per-primitive.
	float xn1 = newton_rhapson_step(xn0, a);
	float xn2 = newton_rhapson_step(xn1, a);
	float xn3 = newton_rhapson_step(xn2, a);
	float xn4 = newton_rhapson_step(xn3, a);
	float result = ldexp(xn4, shifted_exp);
	return result;
}
#endif

#endif