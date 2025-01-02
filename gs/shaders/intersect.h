// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#ifndef INTERSECT_H_
#define INTERSECT_H_

#include "math_utils.h"
#include "utils.h"

int idot3(ivec3 a, ivec2 b)
{
	return a.x * b.x + a.y * b.y + a.z;
}

ivec4 clip_bounding_box(ivec4 bb, ivec4 bb2)
{
	ivec2 lo = max(bb.xy, bb2.xy);
	ivec2 hi = min(bb.zw, bb2.zw);
	return ivec4(lo, hi);
}

bool triangle_setup_intersects_tile(ivec3 a, ivec3 b, ivec3 c, ivec4 bb)
{
	ivec2 lo = bb.xy;
	ivec2 hi = bb.zw;
	bool intersects_quad = all(greaterThanEqual(hi, lo));

	bool parallelogram = (a.x & 1) != 0;
	int parallelogram_b_offset = int(bitfieldExtract(uint(a.x), 2, 2)) - 1;
	int parallelogram_c_offset = int(bitfieldExtract(uint(a.x), 4, 2)) - 1;
	a.x >>= 6;

	if (intersects_quad)
	{
		lo = lo << PGS_RASTER_SUBSAMPLE_BITS;
		hi = (hi << PGS_RASTER_SUBSAMPLE_BITS) | ((1 << PGS_RASTER_SUBSAMPLE_BITS) - 1);

		// Try to intersect overlapped area conservatively.
		ivec2 optimal_a = mix(lo, hi, greaterThanEqual(a.xy, ivec2(0)));
		ivec2 optimal_b = mix(lo, hi, greaterThanEqual(b.xy, ivec2(0)));
		ivec2 optimal_c = mix(lo, hi, greaterThanEqual(c.xy, ivec2(0)));
		int dot_a = idot3(a, optimal_a);
		int dot_b = idot3(b, optimal_b);
		int dot_c = idot3(c, optimal_c);

		intersects_quad = all(greaterThanEqual(ivec3(dot_a, dot_b, dot_c), ivec3(0)));

		if (parallelogram && !intersects_quad)
		{
			// Test against other side.
			bvec2 optimal_mix_a = greaterThanEqual(a.xy, ivec2(0));
			optimal_a = mix(lo, hi, optimal_mix_a);
			optimal_b = mix(lo, hi, greaterThanEqual(-c.xy, ivec2(0)));
			optimal_c = mix(lo, hi, greaterThanEqual(-b.xy, ivec2(0)));

			dot_a = idot3(a, optimal_a);
			dot_b = idot3(b, optimal_b);
			dot_c = idot3(c, optimal_c);
			dot_b += dot_a + parallelogram_b_offset;
			dot_c += dot_a + parallelogram_c_offset;

			// If all A are non-negative, we cannot intersect with the flipped primitive.
			optimal_a = mix(hi, lo, optimal_mix_a);
			dot_a = idot3(a, optimal_a);

			intersects_quad = all(greaterThanEqual(ivec3(-1, dot_b, dot_c), ivec3(dot_a, 0, 0)));
		}
	}

	return intersects_quad;
}

ivec4 pack_bounding_box(ivec2 a, ivec2 b, ivec2 c, ivec2 d, bool multisample, int sample_rate_log2)
{
	ivec2 lo = min(min(a, b), min(c, d));
	ivec2 hi = max(max(a, b), max(c, d));

	// If multisampling snap to previous whole pixel, otherwise snap lo to next whole pixel.
	// Ignore for higher sampling rates since multisampling on top of super-sampling is just silly.
	if (multisample && sample_rate_log2 == 0)
		lo >>= PGS_SUBPIXEL_BITS;
	else
		lo = (lo + ((PGS_SUBPIXELS - 1) >> sample_rate_log2)) >> PGS_SUBPIXEL_BITS;

	// Top-left rule. Bottom/right pixels can never generate coverage.
	hi -= 1;

	// Snap hi to previous whole pixel.
	hi >>= PGS_SUBPIXEL_BITS;

	return ivec4(lo, hi);
}

bool evaluate_multi_coverage_single(PrimitiveSetup setup, bool parallelogram, ivec2 parallelogram_offset,
                                    ivec2 coord)
{
	int a = idot3(setup.a, coord);
	int b = idot3(setup.b, coord);
	int c = idot3(setup.c, coord);

	if (parallelogram && a.x < 0)
	{
		b += a + parallelogram_offset.x;
		c += a + parallelogram_offset.y;
		a = 0;
	}

	return all(greaterThanEqual(ivec3(a, b, c), ivec3(0)));
}

vec2 evaluate_barycentric_ij(ivec3 setup_b, ivec3 setup_c, float inv_area,
                             float error_i, float error_j, ivec2 coord, int sample_rate_log2)
{
	ivec2 sample_coord = coord << (PGS_RASTER_SUBSAMPLE_BITS - sample_rate_log2);
	int b = idot3(setup_b, sample_coord);
	int c = idot3(setup_c, sample_coord);
	precise float i = float(b) * inv_area + error_i;
	precise float j = float(c) * inv_area + error_j;
	return vec2(i, j);
}

bool evaluate_coverage_single(PrimitiveSetup setup, bool parallelogram, ivec2 parallelogram_offset,
                              ivec2 coord, inout float i, inout float j)
{
	int a = idot3(setup.a, coord);
	int b = idot3(setup.b, coord);
	int c = idot3(setup.c, coord);

	precise float i_result = float(b) * setup.inv_area + setup.error_i;
	precise float j_result = float(c) * setup.inv_area + setup.error_j;
	i = i_result;
	j = j_result;

	if (parallelogram && a.x < 0)
	{
		b += a + parallelogram_offset.x;
		c += a + parallelogram_offset.y;
		a = 0;
	}

	return all(greaterThanEqual(ivec3(a, b, c), ivec3(0)));
}

int evaluate_coverage(PrimitiveSetup setup, ivec2 coord, out float i, out float j, bool multisample, int sample_rate_log2)
{
	ivec2 single_sampled_coord = coord >> sample_rate_log2;
	bool coverage = all(greaterThanEqual(ivec4(single_sampled_coord, setup.bb.zw), ivec4(setup.bb.xy, single_sampled_coord)));

	if (!coverage)
		return 0;

	bool parallelogram = (setup.a.x & 1) != 0;
	int parallelogram_b_offset = int(bitfieldExtract(uint(setup.a.x), 2, 2)) - 1;
	int parallelogram_c_offset = int(bitfieldExtract(uint(setup.a.x), 4, 2)) - 1;
	setup.a.x >>= 6;

	ivec2 sample_coord = coord << (PGS_RASTER_SUBSAMPLE_BITS - sample_rate_log2);
	ivec2 offset = ivec2(parallelogram_b_offset, parallelogram_c_offset);

	int coverage_count = 0;

	// Only interpolate at true pixel center.
	bool pixel_center_coverage = evaluate_coverage_single(setup, parallelogram, offset, sample_coord, i, j);

	if (pixel_center_coverage)
	{
		coverage_count = 4;
	}
	else if (multisample && sample_rate_log2 == 0)
	{
		// TODO: Observed behavior from GSdx is bizarre enough that I have serious doubts this is how
		// it's meant to work.
		// If we have pixel center coverage, it seems like we should treat it as full coverage (?!)
		// Only pixels that have such coverage should write Z.
		// i/j interpolation seems to be able to go outside the primitive, at least for RGBA.
		// Only interpolate at pixel center, otherwise, FF X looks funny.

		// These samples are completely arbitrary. Just fakes 4x MSAA with a sub-optimal sample pattern.
		ivec2 multi_coord = sample_coord + ivec2(4, 2);
		coverage_count += int(evaluate_multi_coverage_single(setup, parallelogram, offset, multi_coord));
		multi_coord = sample_coord + ivec2(6, 4);
		coverage_count += int(evaluate_multi_coverage_single(setup, parallelogram, offset, multi_coord));
		multi_coord = sample_coord + ivec2(2, 6);
		coverage_count += int(evaluate_multi_coverage_single(setup, parallelogram, offset, multi_coord));
	}

	return coverage_count;
}

#endif
