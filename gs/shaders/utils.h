// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#ifndef UTILS_H_
#define UTILS_H_

#include "data_structures.h"

bool state_is_z_test(uint state)
{
	return bitfieldExtract(state, STATE_BIT_Z_TEST, 1) != 0;
}

bool state_is_z_test_greater(uint state)
{
	return bitfieldExtract(state, STATE_BIT_Z_TEST_GREATER, 1) != 0;
}

bool state_is_z_write(uint state)
{
	return bitfieldExtract(state, STATE_BIT_Z_WRITE, 1) != 0;
}

bool state_is_opaque(uint state)
{
	return bitfieldExtract(state, STATE_BIT_OPAQUE, 1) != 0;
}

bool state_is_multisample(uint state)
{
	return bitfieldExtract(state, STATE_BIT_MULTISAMPLE, 1) != 0;
}

bool state_is_perspective(uint state)
{
	return bitfieldExtract(state, STATE_BIT_PERSPECTIVE, 1) != 0;
}

bool state_is_scanmsk_even(uint state)
{
	return bitfieldExtract(state, STATE_BIT_SCANMSK_EVEN, 1) != 0;
}

bool state_is_scanmsk_odd(uint state)
{
	return bitfieldExtract(state, STATE_BIT_SCANMSK_ODD, 1) != 0;
}

uint state_get_index(uint state)
{
	return bitfieldExtract(state, STATE_INDEX_BIT_OFFSET, STATE_INDEX_BIT_COUNT);
}

ivec4 unpack_color(uint c)
{
	return (ivec4(c) >> ivec4(0, 8, 16, 24)) & ivec4(0xff);
}

ivec4 unpack_color16(uint c)
{
	ivec4 rgb_32 = ivec4((ivec4(c) >> ivec4(0, 5, 10, 15)) & ivec4(0x1f, 0x1f, 0x1f, 0x1));
	return ivec4(rgb_32 << ivec4(3, 3, 3, 7));
}

uint pack_color(ivec4 c)
{
	c = clamp(c, ivec4(0), ivec4(255));
	c = c << ivec4(0, 8, 16, 24);
	return uint(c.x | c.y | c.z | c.w);
}

uint rgba16_to_rgba32(uint rgb_16, uint aem, uint ta0, uint ta1)
{
	ivec4 c = unpack_color16(rgb_16);

	bool zero_alpha = bool(aem) && rgb_16 == 0;
	c.w = zero_alpha ? 0 : (c.w == 0 ? int(ta0) : int(ta1));

	return pack_color(c);
}

uint rgba32_to_rgba16(uint rgb_32)
{
	ivec4 c = unpack_color(rgb_32);
	c = c >> ivec4(3, 3, 3, 7);
	c = c << ivec4(0, 5, 10, 15);
	return uint(c.x | c.y | c.z | c.w);
}

uint rgb24_to_rgba32(uint rgb_24, uint aem, uint ta0)
{
	ivec4 c = unpack_color(rgb_24);

	bool zero_alpha = bool(aem) && (c.x | c.y | c.z) == 0;
	c.w = zero_alpha ? 0 : int(ta0);

	return pack_color(c);
}

// Inputs are spec constants, so this function will collapse into a constant.
vec2 get_average_sampling_offset(int sample_rate_x_log2, int sample_rate_y_log2)
{
	// Ordered grid is very simple.
	if (sample_rate_x_log2 == sample_rate_y_log2)
	{
		int rate = 1 << sample_rate_y_log2;
		return vec2(0.5 * float(rate - 1) / float(rate));
	}
	else if (sample_rate_y_log2 == 1) // 2x checkerboard
		return vec2(1.0 / 4.0);
	else if (sample_rate_x_log2 == 1 && sample_rate_y_log2 == 2) // 8x checkerboard
		return vec2((0.125 + 0.625) * 0.5);
	else if (sample_rate_x_log2 == 0 && sample_rate_y_log2 == 2) // 4x sparse
		return vec2(6.0 / 16.0);
	else if (sample_rate_x_log2 == 1 && sample_rate_y_log2 == 3) // 16x sparse
		return vec2((6.0 + 22.0) / 64.0);
	else
		return vec2(0.0);
}

#endif