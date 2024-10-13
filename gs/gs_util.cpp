// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#include "gs_util.hpp"
#include "gs_renderer.hpp"
#include "shaders/data_structures.h"
#include "shaders/swizzle_utils.h"

namespace ParallelGS
{
uint32_t psm_word_write_mask(uint32_t psm)
{
	switch (psm)
	{
	case PSMCT24:
	case PSMZ24:
		return 0xffffffu;
	case PSMT8H:
		return 0xff << 24;
	case PSMT4HL:
		return 0xf << 24;
	case PSMT4HH:
		return 0xf << 28;
	default:
		return UINT32_MAX;
	}
}

Extent1D compute_effective_texture_extent(uint32_t extent, uint32_t wrap_mode, uint32_t lo, uint32_t hi, uint32_t levels)
{
	uint32_t base = 0;

	// Try to make an optimized region texture.
	if (wrap_mode == CLAMPBits::REGION_CLAMP)
	{
		lo = std::min<uint32_t>(lo, hi);
		extent = std::max<uint32_t>(hi, lo) - lo + 1;

		if (levels > 1)
		{
			uint32_t max_level = levels - 1;
			uint32_t mask = (1u << max_level) - 1u;

			// Snap extent and lo such that any mip level we use, we won't cause any issues with non-even mip size.
			extent += lo & mask;
			lo &= ~mask;
			extent = (extent + mask) & ~mask;
		}

		base = lo;
	}
	else if (wrap_mode == CLAMPBits::REGION_REPEAT)
	{
		auto msk = lo;
		auto fix = hi;

		if (msk == 0)
		{
			extent = 1;
			base = fix;
		}
		else
		{
			uint32_t msk_msb = 31 - Util::leading_zeroes(msk);
			uint32_t fix_lsb = Util::trailing_zeroes(fix);

			// If LSB > MSB, we can rewrite (x & A) | B -> (x & A) + B.
			if (fix_lsb > msk_msb)
			{
				extent = std::min<uint32_t>(1u << (msk_msb + 1), extent);
				base = fix;
			}
		}
	}

	return { base, extent };
}

void compute_has_potential_feedback(const TEX0Bits &tex0, const CLAMPBits &clamp,
                                    const FRAMEBits &frame, const ZBUFBits &z,
                                    uint32_t pages_in_vram, bool &color_feedback, bool &depth_feedback)
{
	uint32_t fb_base_page = frame.FBP;
	uint32_t z_base_page = z.ZBP;

	auto psm = uint32_t(tex0.PSM);
	auto row_length_64 = uint32_t(tex0.TBW);
	auto layout = get_data_structure(psm);
	uint32_t page_stride = psm == PSMT4 || psm == PSMT8 ? (row_length_64 >> 1) : row_length_64;

	uint32_t width = 1u << uint32_t(tex0.TW);
	uint32_t height = 1u << uint32_t(tex0.TH);

	auto u_extent = compute_effective_texture_extent(
			width, uint32_t(clamp.WMS), uint32_t(clamp.MINU), uint32_t(clamp.MAXU), 1);
	auto v_extent = compute_effective_texture_extent(
			height, uint32_t(clamp.WMT), uint32_t(clamp.MINV), uint32_t(clamp.MAXV), 1);

	// REGION clamps may cause us to go beyond the limits of the texture.
	width = u_extent.base + u_extent.extent;
	height = v_extent.base + v_extent.extent;

	// If the stride is lower than texture width, we can expect the real sampled area is smaller.
	// Only bother with this when W is an obvious dummy value like 1024.
	// For lower values, we risk a scenario where we should have marked a texture as partial feedback,
	// but we didn't, so we get a lot of false positive hazards.
	// Hazard tracking with PS2 is such a mess ... ;_;
	if (width >= 1024 && row_length_64)
		width = std::min<uint32_t>(width, row_length_64 * PGS_BUFFER_WIDTH_SCALE);

	auto base_block = uint32_t(tex0.TBP0);
	uint32_t base_page = base_block / PGS_BLOCKS_PER_PAGE;

	uint32_t fb_base_block = fb_base_page * PGS_BLOCKS_PER_PAGE;
	uint32_t z_base_block = z_base_page * PGS_BLOCKS_PER_PAGE;

	// Consider wrap-around in VRAM. Not 100% sure if PS2 does that, but assume it does.
	if (fb_base_block < base_block)
		fb_base_page += pages_in_vram;
	if (z_base_block < base_block)
		z_base_page += pages_in_vram;

	uint32_t last_page_x = (width - 1) >> layout.page_width_log2;
	uint32_t last_page_y = (height - 1) >> layout.page_height_log2;
	uint32_t last_page = base_page + last_page_x + last_page_y * page_stride;
	// Consider misalignment in the page, which could cause straddle.
	// Don't care too much about accuracy here since it's not necessary.
	if (base_block % PGS_BLOCKS_PER_PAGE)
		last_page++;

	last_page -= base_page;
	fb_base_page -= base_page;
	z_base_page -= base_page;

	color_feedback = last_page >= fb_base_page;
	depth_feedback = last_page >= z_base_page;
}

PageRect compute_page_rect(uint32_t base_256, uint32_t x, uint32_t y,
                           uint32_t width, uint32_t height,
                           uint32_t row_length_64, uint32_t psm)
{
	if (!width || !height)
		return {};

	// Pages are 64x32, 64x64, 128x64, 128x128 for 32-bit, 16-bit, 8-bit and 4-bit respectively.
	uint32_t page_stride = psm == PSMT4 || psm == PSMT8 ? (row_length_64 >> 1) : row_length_64;
	auto layout = get_data_structure(psm);

	constexpr uint32_t BlocksPerPage = PageSize / 256;

	PageRect rect = {};
	rect.base_page = base_256 / BlocksPerPage;
	rect.page_stride = page_stride;
	rect.write_mask = psm_word_write_mask(psm);

	uint32_t fixed_page_offset_x = x >> layout.page_width_log2;
	uint32_t fixed_page_offset_y = y >> layout.page_height_log2;
	rect.base_page += fixed_page_offset_x + page_stride * fixed_page_offset_y;
	x -= fixed_page_offset_x << layout.page_width_log2;
	y -= fixed_page_offset_y << layout.page_height_log2;

	// Any X/Y offset we cannot account for by page offsets should be considered extra width / height.
	// If we don't end up straddling, we can mask out lower coordinate blocks as needed.
	width += x;
	height += y;

	rect.page_width = ((width - 1) >> layout.page_width_log2) + 1;
	rect.page_height = ((height - 1) >> layout.page_height_log2) + 1;

	// In the case that base_256 does not align to a page, we might have to consider overlap due to straddle.
	// Increase effective width and height to catch this scenario.
	uint32_t block_offset = base_256 % BlocksPerPage;

	// Get maximum block index observed in the spill block region.
	uint32_t max_block_x = ((width - 1) >> layout.block_width_log2) % BlocksPerPage;
	uint32_t max_block_y = ((height - 1) >> layout.block_height_log2) % BlocksPerPage;

	// Have to analyze if we need to consider straddle behavior within a page.
	// Also, do fine-grained tracking for single page rects.
	const bool single_page_rect = rect.page_width == 1 && rect.page_height == 1;

	if (block_offset != 0 || single_page_rect)
	{
		uint32_t min_block_x = x >> layout.block_width_log2;
		uint32_t min_block_y = y >> layout.block_height_log2;
		uint64_t block_mask = swizzle_block_mask_PS2(block_offset,
		                                             min_block_x, min_block_y,
		                                             max_block_x - min_block_x + 1,
		                                             max_block_y - min_block_y + 1,
		                                             psm);

		auto hi_mask = uint32_t(block_mask >> 32);
		auto lo_mask = uint32_t(block_mask >> 0);

		if (hi_mask == 0)
		{
			// Easy case, no straddle.
			if (single_page_rect)
				rect.block_mask = lo_mask;
		}
		else
		{
			// Potential hazard.
			if (single_page_rect && lo_mask == 0)
			{
				// We only access the second page, just shift the base page.
				// This can happen with Z swizzle formats.
				rect.base_page++;
				rect.block_mask = hi_mask;
			}
			else
			{
				// Straddle. Every row of pages will access another page.
				rect.page_width++;
			}
		}
	}

	if (!rect.block_mask)
	{
		// If we're straddling page boundaries, assume we're touching the full pages.
		// Block-tracking is not very useful at this point.
		rect.block_mask = UINT32_MAX;
	}

	return rect;
}

using muglm::vec2;
using muglm::ivec2;
using muglm::ivec3;
using muglm::u16vec2;

bool triangle_is_parallelogram_candidate(const VertexPosition *pos, const VertexAttribute *attr,
                                         const ivec2 &lo, const ivec2 &hi, const PRIMBits &prim,
                                         ivec3 &out_parallelogram_order)
{
	// Expect flat Q
	if (prim.FST == 0 && prim.TME)
		if (attr[0].q != attr[1].q || attr[1].q != attr[2].q)
			return false;

	// Expect flat Z
	if (pos[0].z != pos[1].z || pos[1].z != pos[2].z)
		return false;

	// Expect flat Fog if enabled
	if (prim.FGE)
		if (attr[0].fog != attr[1].fog || attr[1].fog != attr[2].fog)
			return false;

	// Expect flat RGBA. Ignore IIP, since RGBA is still used, just not interpolated.
	if (attr[0].rgba != attr[1].rgba || attr[1].rgba != attr[2].rgba)
		return false;

	ivec2 ab = pos[1].pos - pos[0].pos;
	ivec2 ac = pos[2].pos - pos[0].pos;
	ivec2 bc = pos[2].pos - pos[1].pos;
	int area = std::abs(ab.x * ac.y - ab.y * ac.x);
	// Only a 90-degree triangle will have an area that matches the BB area.
	if (area != (hi.x - lo.x) * (hi.y - lo.y))
		return false;

	ivec3 parallelogram_order;

	if (ab.x != 0 && ab.y != 0)
	{
		// AB is the diagonal, C is provoking.
		parallelogram_order = ivec3(2, 1, 0);

		// Verify that the provoking corner is 90 degrees.
		// The area check alone doesn't guarantee that.
		int cos_angle = ac.x * bc.x + ac.y * bc.y;
		if (cos_angle != 0)
			return false;
	}
	else if (ac.x != 0 && ac.y != 0)
	{
		// AC is the diagonal, B is provoking.
		parallelogram_order = ivec3(1, 2, 0);

		int cos_angle = ac.x * ab.x + ac.y * ab.y;
		if (cos_angle != 0)
			return false;
	}
	else
	{
		// We're not on the diagonal. A is provoking.
		parallelogram_order = ivec3(0, 1, 2);

		int cos_angle = ac.x * ab.x + ac.y * ab.y;
		if (cos_angle != 0)
			return false;
	}

	// Choose diagonal points invariantly, so we can match them up.
	if (pos[parallelogram_order.y].pos.x > pos[parallelogram_order.z].pos.x)
		std::swap(parallelogram_order.y, parallelogram_order.z);

	out_parallelogram_order = parallelogram_order;
	return true;
}

bool triangles_form_parallelogram(const VertexPosition *pos, const VertexAttribute *attr,
                                  const ivec3 &order,
                                  const VertexPosition *last_pos, const VertexAttribute *last_attr,
                                  const ivec3 &last_order,
                                  const PRIMBits &prim)
{
	auto &pos0 = pos[order.x];
	auto &pos1 = pos[order.y];
	auto &pos2 = pos[order.z];
	auto pos3 = pos1.pos + pos2.pos - pos0.pos;

	auto &last_pos0 = last_pos[last_order.x];
	auto &last_pos1 = last_pos[last_order.y];
	auto &last_pos2 = last_pos[last_order.z];

	if (any(notEqual(pos3, last_pos0.pos)) ||
	    any(notEqual(pos1.pos, last_pos1.pos)) ||
	    any(notEqual(pos2.pos, last_pos2.pos)))
	{
		return false;
	}

	if (pos[0].z != last_pos[0].z)
		return false;
	if (!prim.FST && prim.TME && attr[0].q != last_attr[0].q)
		return false;
	if (prim.FGE && attr[0].fog != last_attr[0].fog)
		return false;
	if (attr[0].rgba != last_attr[0].rgba)
		return false;

	auto &attr0 = attr[order.x];
	auto &attr1 = attr[order.y];
	auto &attr2 = attr[order.z];

	auto &last_attr0 = last_attr[last_order.x];
	auto &last_attr1 = last_attr[last_order.y];
	auto &last_attr2 = last_attr[last_order.z];

	if (prim.TME)
	{
		if (prim.FST)
		{
			u16vec2 uv3 = attr1.uv + attr2.uv - attr0.uv;
			if (any(notEqual(uv3, last_attr0.uv)) ||
			    any(notEqual(attr1.uv, last_attr1.uv)) ||
			    any(notEqual(attr2.uv, last_attr2.uv)))
			{
				return false;
			}
		}
		else
		{
			// Accept some very minor error in the computation.
			// If the error is less than a subtexel at 1k x 1k resolution, we're definitely close enough.
			vec2 st3_error = muglm::abs((attr2.st + attr1.st - attr0.st - last_attr0.st) / attr0.q);

			if (any(notEqual(attr1.st, last_attr1.st)) ||
			    any(notEqual(attr2.st, last_attr2.st)) ||
			    any(greaterThan(st3_error, vec2(1e-4f))))
			{
				return false;
			}
		}
	}

	return true;
}
}