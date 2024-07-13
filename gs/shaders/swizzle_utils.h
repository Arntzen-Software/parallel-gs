// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#ifndef SWIZZLE_UTILS_H_
#define SWIZZLE_UTILS_H_

#include "data_structures.h"

#ifdef __cplusplus
#define INLINE static inline
namespace ParallelGS {
#else
#define INLINE
#endif

INLINE uint get_bits_per_pixel(uint format)
{
	uint bpp = 0;

	switch (format)
	{
	case PSMCT24:
	case PSMZ24:
		bpp = 24;
		break;

	case PSMCT16:
	case PSMCT16S:
	case PSMZ16:
	case PSMZ16S:
		bpp = 16;
		break;

	case PSMT8:
	case PSMT8H:
		bpp = 8;
		break;

	case PSMT4:
	case PSMT4HL:
	case PSMT4HH:
		bpp = 4;
		break;

	default:
	case PSMCT32:
	case PSMZ32:
		bpp = 32;
		break;
	}

	return bpp;
}

INLINE uint get_stride_per_pixel(uint format)
{
	uint bpp = 0;

	switch (format)
	{
	default:
	case PSMCT32:
	case PSMZ32:
	case PSMCT24:
	case PSMZ24:
	case PSMT8H:
	case PSMT4HL:
	case PSMT4HH:
		bpp = 32;
		break;

	case PSMCT16:
	case PSMCT16S:
	case PSMZ16:
	case PSMZ16S:
		bpp = 16;
		break;

	case PSMT8:
		bpp = 8;
		break;

	case PSMT4:
		bpp = 4;
		break;
	}

	return bpp;
}

INLINE LocalDataStructure get_data_structure(uint format)
{
	LocalDataStructure data_structure;

	switch (format)
	{
	case PSMCT16:
	case PSMCT16S:
	case PSMZ16:
	case PSMZ16S:
		data_structure.page_width = 64;
		data_structure.page_height = 64;
		data_structure.block_width = 16;
		data_structure.block_height = 8;
		data_structure.column_height = 2;

		data_structure.page_width_log2 = 6;
		data_structure.page_height_log2 = 6;
		data_structure.block_width_log2 = 4;
		data_structure.block_height_log2 = 3;
		data_structure.column_height_log2 = 1;
		break;

	case PSMT8:
		data_structure.page_width = 128;
		data_structure.page_height = 64;
		data_structure.block_width = 16;
		data_structure.block_height = 16;
		data_structure.column_height = 4;

		data_structure.page_width_log2 = 7;
		data_structure.page_height_log2 = 6;
		data_structure.block_width_log2 = 4;
		data_structure.block_height_log2 = 4;
		data_structure.column_height_log2 = 2;
		break;

	case PSMT4:
		data_structure.page_width = 128;
		data_structure.page_height = 128;
		data_structure.block_width = 32;
		data_structure.block_height = 16;
		data_structure.column_height = 4;

		data_structure.page_width_log2 = 7;
		data_structure.page_height_log2 = 7;
		data_structure.block_width_log2 = 5;
		data_structure.block_height_log2 = 4;
		data_structure.column_height_log2 = 2;
		break;

	default:
		data_structure.page_width = 64;
		data_structure.page_height = 32;
		data_structure.block_width = 8;
		data_structure.block_height = 8;
		data_structure.column_height = 2;

		data_structure.page_width_log2 = 6;
		data_structure.page_height_log2 = 5;
		data_structure.block_width_log2 = 3;
		data_structure.block_height_log2 = 3;
		data_structure.column_height_log2 = 1;
		break;
	}

	return data_structure;
}

INLINE bool is_palette_format(uint format)
{
	bool is_palette = false;

	switch (format)
	{
	case PSMT8:
	case PSMT4:
	case PSMT8H:
	case PSMT4HL:
	case PSMT4HH:
		is_palette = true;
		break;

	default:
		break;
	}

	return is_palette;
}

#ifdef __cplusplus
template <typename Func>
static inline uint64_t swizzle_block_iterate_mask(
		uint base_block,
		uint base_block_x, uint base_block_y,
		uint num_block_x, uint num_block_y,
		const Func &func)
{
	uint64_t block_mask = 0;
	for (uint y = 0; y < num_block_y; y++)
	{
		for (uint x = 0; x < num_block_x; x++)
		{
			uint block_x = base_block_x + x;
			uint block_y = base_block_y + y;
			uint block_index = func(base_block, block_x, block_y);
			block_mask |= 1ull << block_index;
		}
	}
	return block_mask;
}

INLINE uint64_t swizzle_block_mask_PS2(uint base_block, uint base_block_x, uint base_block_y,
                                       uint num_block_x, uint num_block_y,
                                       uint storage_format)
{
	uint64_t block_mask;

	switch (storage_format)
	{
	// Just assume default here since this is the "sane" swizzle layout.
	default:
	case PSMCT32:
	case PSMCT24:
	case PSMT4HH:
	case PSMT4HL:
	case PSMT8H:
	case PSMT8:
		block_mask = swizzle_block_iterate_mask(
				base_block, base_block_x, base_block_y, num_block_x, num_block_y,
				[](uint base, uint block_x, uint block_y) {
					uint block = ((block_x & 0x1u) | ((block_y & 0x1u) << 1) | ((block_x & 0x2u) << 1) |
					             ((block_y & 0x2u) << 2) | ((block_x & 0x4u) << 2));
					return block + base;
				});
		break;

	case PSMZ32:
	case PSMZ24:
		block_mask = swizzle_block_iterate_mask(
				base_block, base_block_x, base_block_y, num_block_x, num_block_y,
				[](uint base, uint block_x, uint block_y) {
					// XOR seems to happen after block offset?
					// This doesn't matter for rendering, but does matter for texture uploads (FF X glass shatter).
					uint block = (block_x & 0x1u) | ((block_y & 0x1u) << 1) | ((block_x & 0x2u) << 1) |
					             ((block_y & 0x2u) << 2) | ((block_x & 0x4u) << 2);
					block += base;
					block ^= 0x18;
					return block;
				});
		break;

	case PSMCT16:
	case PSMT4:
		block_mask = swizzle_block_iterate_mask(
				base_block, base_block_x, base_block_y, num_block_x, num_block_y,
				[](uint base, uint block_x, uint block_y) {
					uint block = (block_y & 0x1u) | ((block_x & 0x1u) << 1) | ((block_y & 0x2u) << 1) |
					             ((block_x & 0x2u) << 2) | ((block_y & 0x4u) << 2);
					return block + base;
				});
		break;

	case PSMZ16:
		block_mask = swizzle_block_iterate_mask(
				base_block, base_block_x, base_block_y, num_block_x, num_block_y,
				[](uint base, uint block_x, uint block_y) {
					// XOR seems to happen after block offset?
					// This doesn't matter for rendering, but does matter for texture uploads (FF X glass shatter).
					uint block = (block_y & 0x1u) | ((block_x & 0x1u) << 1) | ((block_y & 0x2u) << 1) |
					             ((block_x & 0x2u) << 2) | ((block_y & 0x4u) << 2);
					block += base;
					block ^= 0x18;
					return block;
				});
		break;

	case PSMCT16S:
		block_mask = swizzle_block_iterate_mask(
				base_block, base_block_x, base_block_y, num_block_x, num_block_y,
				[](uint base, uint block_x, uint block_y) {
					uint block = (block_y & 0x1u) | ((block_x & 0x1u) << 1) | (block_y & 0x4u) |
					             ((block_y & 0x2u) << 2) | ((block_x & 0x2u) << 3);
					return block + base;
				});
		break;

	case PSMZ16S:
		block_mask = swizzle_block_iterate_mask(
				base_block, base_block_x, base_block_y, num_block_x, num_block_y,
				[](uint base, uint block_x, uint block_y) {
					// XOR seems to happen after block offset?
					// This doesn't matter for rendering, but does matter for texture uploads (FF X glass shatter).
					uint block = (block_y & 0x1u) | ((block_x & 0x1u) << 1) | (block_y & 0x4u) |
					             ((block_y & 0x2u) << 2) | ((block_x & 0x2u) << 3);
					block += base;
					block ^= 0x18;
					return block;
				});
		break;
	}

	return block_mask;
}

INLINE uint swizzle_compat_key(uint storage_format)
{
	switch (storage_format)
	{
	default:
	case PSMCT32:
	case PSMT4HH:
	case PSMT4HL:
	case PSMT8H:
	case PSMCT24:
		return 1;
	case PSMZ32:
		return 2;
	case PSMCT16:
		return 3;
	case PSMZ16:
		return 4;
	case PSMCT16S:
		return 5;
	case PSMZ16S:
		return 6;
	case PSMT8:
		return 7;
	case PSMT4:
		return 8;
	case PSMZ24:
		return 9;
	}
}
#endif

INLINE uint swizzle_PS2(uint x, uint y, uint base_pointer, uint page_stride, uint storage_format, uint vram_mask)
{
	LocalDataStructure data_structure = get_data_structure(storage_format);

	const uint block_size = PGS_BLOCK_ALIGNMENT_BYTES;
	const uint column_size = PGS_BLOCK_ALIGNMENT_BYTES / 4;

	uint page_x = x >> data_structure.page_width_log2;
	uint page_y = y >> data_structure.page_height_log2;

	if (storage_format == PSMT4 || storage_format == PSMT8)
		page_stride >>= 1;

	uint page_index = page_y * page_stride + page_x;

	uint block_x = (x & (data_structure.page_width - 1)) >> data_structure.block_width_log2;
	uint block_y = (y & (data_structure.page_height - 1)) >> data_structure.block_height_log2;

	uint column_index = (y & (data_structure.block_height - 1)) >> data_structure.column_height_log2;

	uint pixel_x = x & (data_structure.block_width - 1);
	uint pixel_y = y & (data_structure.column_height - 1);

	uint block_offset = base_pointer;

	uint buffer_index = 0;

	switch (storage_format)
	{
	default:
	case PSMCT32:
	case PSMCT24:
	case PSMT4HH:
	case PSMT4HL:
	case PSMT8H:
	{
		const uint bytes_per_pixel = 4;

		// x2 y1 x1 y0 x0
		uint block_index = (block_x & 0x1u) | ((block_y & 0x1u) << 1) | ((block_x & 0x2u) << 1) |
		                   ((block_y & 0x2u) << 2) | ((block_x & 0x4u) << 2);
		block_index += block_offset;

		// x2 x1 y0 x0
		uint pixel_index =
				(pixel_x & 0x1u) | ((pixel_y & 0x1u) << 1) | ((pixel_x & 0x2u) << 1) | ((pixel_x & 0x4u) << 1);

		buffer_index = page_index * PGS_PAGE_ALIGNMENT_BYTES + block_index * block_size +
					   column_index * column_size + pixel_index * bytes_per_pixel;
		buffer_index = (buffer_index & vram_mask) >> 2;
		break;
	}

	case PSMZ32:
	case PSMZ24:
	{
		const uint bytes_per_pixel = 4;

		// ~x2 ~y1 x1 y0 x0
		uint block_index = (block_x & 0x1u) | ((block_y & 0x1u) << 1) | ((block_x & 0x2u) << 1) |
		                   ((block_y & 0x2u) << 2) | ((block_x & 0x4u) << 2);
		block_index += block_offset;

		// XOR seems to happen after offsets. (FF X glass shatter).
		block_index ^= 0x18;

		// x2 x1 y0 x0
		uint pixel_index =
				(pixel_x & 0x1u) | ((pixel_y & 0x1u) << 1) | ((pixel_x & 0x2u) << 1) | ((pixel_x & 0x4u) << 1);

		buffer_index = page_index * PGS_PAGE_ALIGNMENT_BYTES + block_index * block_size +
					   column_index * column_size + pixel_index * bytes_per_pixel;
		buffer_index = (buffer_index & vram_mask) >> 2;
		break;
	}

	case PSMCT16:
	{
		const uint bytes_per_pixel = 2;

		// y2 x1 y1 x0 y0
		uint block_index = (block_y & 0x1u) | ((block_x & 0x1u) << 1) | ((block_y & 0x2u) << 1) |
		                   ((block_x & 0x2u) << 2) | ((block_y & 0x4u) << 2);
		block_index += block_offset;

		// x2 x1 y0 x0 x3
		uint pixel_index = ((pixel_x & 0x8u) >> 3) | ((pixel_x & 0x1u) << 1) | ((pixel_y & 0x1u) << 2) |
						   ((pixel_x & 0x2u) << 2) | ((pixel_x & 0x4u) << 2);

		buffer_index = page_index * PGS_PAGE_ALIGNMENT_BYTES +
					   block_index * block_size + column_index * column_size + pixel_index * bytes_per_pixel;
		buffer_index = (buffer_index & vram_mask) >> 1;
		break;
	}

	case PSMZ16:
	{
		const uint bytes_per_pixel = 2;

		// ~y2 ~x1 y1 x0 y0
		uint block_index = (block_y & 0x1u) | ((block_x & 0x1u) << 1) | ((block_y & 0x2u) << 1) |
		                   ((block_x & 0x2u) << 2) | ((block_y & 0x4u) << 2);
		block_index += block_offset;
		block_index ^= 0x18;

		// x2 x1 y0 x0 x3
		uint pixel_index = ((pixel_x & 0x8u) >> 3) | ((pixel_x & 0x1u) << 1) | ((pixel_y & 0x1u) << 2) |
						   ((pixel_x & 0x2u) << 2) | ((pixel_x & 0x4u) << 2);

		buffer_index = page_index * PGS_PAGE_ALIGNMENT_BYTES + block_index * block_size +
					   column_index * column_size + pixel_index * bytes_per_pixel;
		buffer_index = (buffer_index & vram_mask) >> 1;
		break;
	}

	case PSMCT16S:
	{
		const uint bytes_per_pixel = 2;

		// x1 y1 y2 x0 y0
		uint block_index = (block_y & 0x1u) | ((block_x & 0x1u) << 1) | (block_y & 0x4u) |
		                   ((block_y & 0x2u) << 2) | ((block_x & 0x2u) << 3);
		block_index += block_offset;

		// x2 x1 y0 x0 x3
		uint pixel_index = ((pixel_x & 0x8u) >> 3) | ((pixel_x & 0x1u) << 1) | ((pixel_y & 0x1u) << 2) |
						   ((pixel_x & 0x2u) << 2) | ((pixel_x & 0x4u) << 2);

		buffer_index = page_index * PGS_PAGE_ALIGNMENT_BYTES +
					   block_index * block_size + column_index * column_size + pixel_index * bytes_per_pixel;
		buffer_index = (buffer_index & vram_mask) >> 1;
		break;
	}

	case PSMZ16S:
	{
		const uint bytes_per_pixel = 2;

		// ~x1 ~y1 y2 x0 y0
		uint block_index = (block_y & 0x1u) | ((block_x & 0x1u) << 1) | (block_y & 0x4u) |
						   ((block_y & 0x2u) << 2) | ((block_x & 0x2u) << 3);
		block_index += block_offset;
		block_index ^= 0x18;

		// x2 x1 y0 x0 x3
		uint pixel_index = ((pixel_x & 0x8u) >> 3) | ((pixel_x & 0x1u) << 1) | ((pixel_y & 0x1u) << 2) |
						   ((pixel_x & 0x2u) << 2) | ((pixel_x & 0x4u) << 2);

		buffer_index = page_index * PGS_PAGE_ALIGNMENT_BYTES + block_index * block_size +
					   column_index * column_size + pixel_index * bytes_per_pixel;
		buffer_index = (buffer_index & vram_mask) >> 1;
		break;
	}

	case PSMT8:
	{
		// x2 y1 x1 y0 x0 (same as PSMCT32)
		uint block_index = (block_x & 0x1u) | ((block_y & 0x1u) << 1) | ((block_x & 0x2u) << 1) |
		                   ((block_y & 0x2u) << 2) | ((block_x & 0x4u) << 2);
		block_index += block_offset;

		// (x2^y1)^c0 x1 y0 x0 x4 y1
		uint pixel_index = ((pixel_y & 0x2u) >> 1) | ((pixel_x & 0x8u) >> 2) | ((pixel_x & 0x1u) << 2) |
						   ((pixel_y & 0x1u) << 3) | ((pixel_x & 0x2u) << 3) |
						   ((((pixel_x & 0x4u) << 3) ^ ((pixel_y & 0x2u) << 4)) ^ (column_index & 0x1u) << 5);

		buffer_index =
				page_index * PGS_PAGE_ALIGNMENT_BYTES + block_index * block_size +
				column_index * column_size + pixel_index;
		buffer_index = buffer_index & vram_mask; // Byte address
		break;
	}

	case PSMT4:
	{
		// y2 x1 y1 x0 y0 (same as PSMCT16)
		uint block_index = (block_y & 0x1u) | ((block_x & 0x1u) << 1) | ((block_y & 0x2u) << 1) |
						   ((block_x & 0x2u) << 2) | ((block_y & 0x4u) << 2);
		block_index += block_offset;

		// (x2^y1)^c0 x1 y0 x0 x4 x3 y1
		uint pixel_index = ((pixel_y & 0x2u) >> 1) | ((pixel_x & 0x8u) >> 2) | ((pixel_x & 0x10u) >> 2) |
						   ((pixel_x & 0x1u) << 3) | ((pixel_y & 0x1u) << 4) | ((pixel_x & 0x2u) << 4) |
						   ((((pixel_x & 0x4u) << 4) ^ ((pixel_y & 0x2u) << 5)) ^ (column_index & 0x1u) << 6);

		buffer_index = ((page_index * PGS_PAGE_ALIGNMENT_BYTES + block_index * block_size +
						 column_index * column_size) << 1) + pixel_index;
		buffer_index = buffer_index & ((vram_mask << 1) | 1); // Nibble address
		break;
	}
	}

	return buffer_index;
}

#ifdef __cplusplus
}
#endif
#endif
