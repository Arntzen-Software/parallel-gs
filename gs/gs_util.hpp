// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#pragma once

#include "page_tracker.hpp"
#include "gs_registers.hpp"
#include "muglm/muglm_impl.hpp"
#include <stdint.h>
#include "shaders/data_structures.h"
#include "shaders/swizzle_utils.h"

namespace ParallelGS
{
// Convert common addressing modes from copies and uploads.
// base is addr / 256 (64 words, i.e. block aligned).
// row_length_64 is effectively page stride.
PageRect compute_page_rect(uint32_t base_256, uint32_t x, uint32_t y,
                           uint32_t width, uint32_t height,
                           uint32_t row_length_64, uint32_t psm);

uint32_t psm_word_write_mask(uint32_t psm);

bool triangle_is_parallelogram_candidate(const VertexPosition *pos, const VertexAttribute *attr,
                                         const muglm::ivec2 &lo, const muglm::ivec2 &hi, const PRIMBits &prim,
                                         muglm::ivec3 &parallelogram_order);

bool triangles_form_parallelogram(const VertexPosition *pos, const VertexAttribute *attr,
                                  const muglm::ivec3 &order,
                                  const VertexPosition *last_pos, const VertexAttribute *last_attr,
                                  const muglm::ivec3 &last_order, const PRIMBits &prim);

void compute_has_potential_feedback(const TEX0Bits &tex0, const CLAMPBits &clamp,
                                    const FRAMEBits &frame, const ZBUFBits &z,
                                    uint32_t pages_in_vram, bool &color_feedback, bool &depth_feedback);

struct Extent1D
{
	uint32_t base;
	uint32_t extent;
};

Extent1D compute_effective_texture_extent(uint32_t extent, uint32_t wrap_mode,
                                          uint32_t lo, uint32_t hi,
                                          uint32_t levels);

template <uint32_t PSM>
static inline void vram_readback(void *readback_data, const void *host_data, uint32_t base_256, uint32_t page_stride,
                                 uint32_t src_x, uint32_t src_y,
                                 uint32_t width, uint32_t height,
                                 uint32_t vram_mask)
{
	// This can definitely be optimized a lot if need be.
	// Especially if we are block-aligned, it should be possible to de-swizzle a full 8x8 block in one go in SIMD or something ...
	for (uint32_t y = 0; y < height; y++)
	{
		uint32_t effective_y = (y + src_y) & 2047;
		uint32_t output_pixel = y * width;
		for (uint32_t x = 0; x < width; x++, output_pixel++)
		{
			uint32_t effective_x = (x + src_x) & 2047;

			// Hopefully inlining with templated argument should collapse branches.
			uint32_t addr = swizzle_PS2(effective_x, effective_y, base_256, page_stride, PSM, vram_mask);

			// Templated, so should collapse into branchless code.
			switch (PSM)
			{
			case PSMCT32:
			case PSMZ32:
				static_cast<uint32_t *>(readback_data)[output_pixel] = static_cast<const uint32_t *>(host_data)[addr];
				break;

			case PSMCT24:
			case PSMZ24:
			{
				auto *dst = static_cast<uint8_t *>(readback_data) + output_pixel * 3;
				uint32_t word = static_cast<const uint32_t *>(host_data)[addr];
				dst[0] = uint8_t(word >> 0);
				dst[1] = uint8_t(word >> 8);
				dst[2] = uint8_t(word >> 16);
				break;
			}

			case PSMCT16:
			case PSMCT16S:
			case PSMZ16:
			case PSMZ16S:
				static_cast<uint16_t *>(readback_data)[output_pixel] = static_cast<const uint16_t *>(host_data)[addr];
				break;

			case PSMT8:
				static_cast<uint8_t *>(readback_data)[output_pixel] = static_cast<const uint8_t *>(host_data)[addr];
				break;

			case PSMT8H:
				static_cast<uint8_t *>(readback_data)[output_pixel] = static_cast<const uint8_t *>(host_data)[4 * addr + 3];
				break;

			// 4-bit is not allowed for readback.

			default:
				LOGW("Unrecognized fifo readback format.\n");
				static_cast<uint32_t *>(readback_data)[output_pixel] = 0;
			}
		}
	}
}

template <uint32_t PSM>
static inline void vram_upload(void *vram_data, const void *upload_data,
                               uint32_t base_256, uint32_t page_stride,
                               uint32_t dst_x, uint32_t dst_y,
                               uint32_t width, uint32_t height,
                               uint32_t vram_mask)
{
	// This can definitely be optimized a lot if need be.
	// Especially if we are block-aligned, it should be possible to de-swizzle a full 8x8 block in one go in SIMD or something ...
	for (uint32_t y = 0; y < height; y++)
	{
		uint32_t effective_y = (y + dst_y) & 2047;
		uint32_t input_pixel = y * width;
		for (uint32_t x = 0; x < width; x++, input_pixel++)
		{
			uint32_t effective_x = (x + dst_x) & 2047;

			// Hopefully inlining with templated argument should collapse branches.
			uint32_t addr = swizzle_PS2(effective_x, effective_y, base_256, page_stride, PSM, vram_mask);

			// Templated, so should collapse into branchless code.
			switch (PSM)
			{
			case PSMCT32:
			case PSMZ32:
				static_cast<uint32_t *>(vram_data)[addr] = static_cast<const uint32_t *>(upload_data)[input_pixel];
				break;

			case PSMCT24:
			case PSMZ24:
			{
				auto *dst = static_cast<uint8_t *>(vram_data) + addr * 4;
				auto *src = static_cast<const uint8_t *>(upload_data) + input_pixel * 3;
				dst[0] = src[0];
				dst[1] = src[1];
				dst[2] = src[2];
				break;
			}

			case PSMCT16:
			case PSMCT16S:
			case PSMZ16:
			case PSMZ16S:
				static_cast<uint16_t *>(vram_data)[addr] = static_cast<const uint16_t *>(upload_data)[input_pixel];
				break;

			case PSMT8:
				static_cast<uint8_t *>(vram_data)[addr] = static_cast<const uint8_t *>(upload_data)[input_pixel];
				break;

			case PSMT8H:
				static_cast<uint8_t *>(vram_data)[4 * addr + 3] = static_cast<const uint8_t *>(upload_data)[input_pixel];
				break;

			case PSMT4:
			{
				auto &pix = static_cast<uint8_t *>(vram_data)[addr / 2];

				uint8_t input_data = static_cast<const uint8_t *>(upload_data)[input_pixel / 2];
				if (input_pixel & 1)
					input_data >>= 4;
				input_data &= 0xfu;

				if (addr & 1)
					pix = (pix & 0xfu) | (input_data << 4u);
				else
					pix = (pix & 0xf0u) | input_data;

				break;
			}

			case PSMT4HL:
			case PSMT4HH:
			{
				auto &pix = static_cast<uint32_t *>(vram_data)[addr];
				uint32_t input_data = static_cast<const uint8_t *>(upload_data)[input_pixel / 2];
				if (input_pixel & 1)
					input_data >>= 4;
				input_data &= 0xfu;

				constexpr uint32_t shamt = PSM == PSMT4HL ? 24 : 28;
				constexpr uint32_t mask = PSM == PSMT4HL ? 0xf0ffffffu : 0x0fffffffu;
				input_data <<= shamt;
				pix = (pix & mask) | input_data;

				break;
			}

			default:
				break;
			}
		}
	}
}
}
