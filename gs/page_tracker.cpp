// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#include "page_tracker.hpp"
#include "gs_interface.hpp"
#include "unstable_remove_if.hpp"
#include <algorithm>

#if 0 && defined(PARALLEL_GS_DEBUG) && PARALLEL_GS_DEBUG
#define TRACE(...) printf(__VA_ARGS__)
#else
#define TRACE(...) ((void)0)
#endif

namespace ParallelGS
{
PageTracker::PageTracker(GSInterface &cb_)
	: cb(cb_)
{
}

void PageTracker::set_num_pages(unsigned num_pages)
{
	page_state_mask = num_pages - 1;
	page_state.resize(num_pages);

	potential_invalidated_indices.reserve(num_pages);
	accessed_fb_pages.reserve(num_pages);
	accessed_cache_pages.reserve(num_pages);
	accessed_copy_pages.reserve(num_pages);
	accessed_readback_pages.reserve(num_pages);
}

bool PageTracker::page_has_host_write_timeline_update(const PageRect &rect) const
{
	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;
			auto &state = page_state[page & page_state_mask];
			if ((state.need_host_write_timeline_mask & rect.block_mask) != 0)
				return true;
		}
	}

	return false;
}

bool PageTracker::page_has_host_read_timeline_update(const PageRect &rect) const
{
	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;
			auto &state = page_state[page & page_state_mask];
			if ((state.need_host_read_timeline_mask & rect.block_mask) != 0)
				return true;
		}
	}

	return false;
}

bool PageTracker::page_has_fb_read_write(const PageRect &rect) const
{
	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;
			auto &state = page_state[page & page_state_mask];
			if (((state.fb_write_mask | state.fb_read_mask) & rect.block_mask) != 0 &&
			    (state.pending_fb_access_mask & rect.write_mask) != 0)
			{
				return true;
			}
		}
	}

	return false;
}

bool PageTracker::page_has_fb_write(const PageRect &rect) const
{
	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;
			auto &state = page_state[page & page_state_mask];
			if ((state.fb_write_mask & rect.block_mask) != 0 &&
			    (state.pending_fb_access_mask & rect.write_mask) != 0)
			{
				return true;
			}
		}
	}

	return false;
}

bool PageTracker::page_is_copy_cached_sensitive(const PageRect& rect) const
{
	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;
			auto &state = page_state[page & page_state_mask];
			if (((state.cached_read_block_mask | state.copy_write_block_mask | state.copy_read_block_mask) & rect.block_mask) != 0)
				return true;
		}
	}

	return false;
}

void PageTracker::mark_external_write(const PageRect &rect)
{
	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;

			page &= page_state_mask;
			auto &state = page_state[page];

			register_accessed_readback_page(page);
			register_potential_invalidated_indices(page);

			state.need_host_read_timeline_mask |= rect.block_mask;
			state.need_host_write_timeline_mask |= rect.block_mask;
			state.texture_cache_needs_invalidate_block_mask |= rect.block_mask;
			state.texture_cache_needs_invalidate_write_mask |= rect.block_mask;
			TRACE("TRACKER || PAGE 0x%x, EXT write\n", page);
		}
	}

	invalidate_texture_cache(UINT32_MAX);
}

bool PageTracker::texture_may_super_sample(const PageRect &rect) const
{
	// Only check the base page since otherwise we may easily get false positives
	// due to W/H rounding up to nearest POT.
	auto &state = page_state[rect.base_page & page_state_mask];
	return (state.flags & PAGE_STATE_MAY_SUPER_SAMPLE_BIT) != 0;
}

void PageTracker::mark_fb_write(const PageRect &rect)
{
	assert(rect.page_width <= 2048 / 64);
	assert(rect.page_height <= 2048 / 32);

	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;

			page &= page_state_mask;
			auto &state = page_state[page];

			register_accessed_fb_pages(page);
			register_accessed_readback_page(page);

			state.fb_read_mask |= rect.block_mask;
			state.fb_write_mask |= rect.block_mask;
			state.need_host_write_timeline_mask |= rect.block_mask;
			state.need_host_read_timeline_mask |= rect.block_mask;
			state.flags |= PAGE_STATE_MAY_SUPER_SAMPLE_BIT;

			if (state.texture_cache_needs_invalidate_block_mask == 0)
				potential_invalidated_indices.push_back(page);
			state.texture_cache_needs_invalidate_block_mask |= UINT32_MAX;
			state.texture_cache_needs_invalidate_write_mask |= UINT32_MAX;
			state.pending_fb_access_mask |= rect.write_mask;
			TRACE("TRACKER || PAGE 0x%x, FB write\n", page);
		}
	}

	// To accelerate mark_texture_read spam, which is fairly hot in profiles.
	// Allows early out check.
	// Intentionally don't use wrapping logic here.
	pending_fb_write_page_lo = std::min<uint32_t>(rect.base_page, pending_fb_write_page_lo);
	pending_fb_write_page_hi = std::max<uint32_t>(rect.base_page +
	                                              (rect.page_height - 1) * rect.page_stride + rect.page_width - 1,
	                                              pending_fb_write_page_hi);
	pending_fb_write_mask |= rect.write_mask;
}

void PageTracker::mark_fb_read(const PageRect &rect)
{
	assert(rect.page_width <= 2048 / 64);
	assert(rect.page_height <= 2048 / 64);

	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;
			page &= page_state_mask;
			auto &state = page_state[page];

			register_accessed_fb_pages(page);
			register_accessed_readback_page(page);

			state.fb_read_mask |= rect.block_mask;
			state.need_host_write_timeline_mask |= rect.block_mask;
			state.pending_fb_access_mask |= rect.write_mask;
			TRACE("TRACKER || PAGE 0x%x, FB read\n", page & page_state_mask);
		}
	}
}

bool PageTracker::mark_transfer_copy(const PageRect &dst_rect, const PageRect &src_rect)
{
	auto dst_block = get_block_state(dst_rect);
	auto src_block = get_block_state(src_rect);

	bool need_tex_invalidate = false;
	bool has_hazard = false;

	if (page_has_fb_read_write(dst_rect) || page_has_fb_write(src_rect))
	{
		flush_render_pass(FlushReason::CopyHazard);
	}
	else if ((dst_block.cached_read_block_mask & dst_rect.block_mask) != 0)
	{
		flush_cached();
		need_tex_invalidate = true;
	}
	else if ((dst_block.copy_read_block_mask & dst_rect.block_mask) != 0 ||
	         (src_block.copy_write_block_mask & src_rect.block_mask) != 0)
	{
		flush_copy();
	}

	// Write-after-Write hazards for copies are handled internally through atomics.
	// We only need to care about write-after-read and read-after-write.

	for (unsigned y = 0; y < dst_rect.page_height; y++)
	{
		for (unsigned x = 0; x < dst_rect.page_width; x++)
		{
			unsigned page = dst_rect.base_page + y * dst_rect.page_stride + x;
			page &= page_state_mask;
			auto &state = page_state[page];

			register_accessed_readback_page(page);
			register_accessed_copy_pages(page);
			register_potential_invalidated_indices(page);

			state.need_host_write_timeline_mask |= dst_rect.block_mask;
			state.need_host_read_timeline_mask |= dst_rect.block_mask;
			state.copy_write_block_mask |= dst_rect.block_mask;
			state.texture_cache_needs_invalidate_block_mask |= dst_rect.block_mask;
			state.texture_cache_needs_invalidate_write_mask |= dst_rect.write_mask;

			state.flags &= ~PAGE_STATE_MAY_SUPER_SAMPLE_BIT;

			TRACE("TRACKER || PAGE 0x%x, WRITE |= 0x%x -> 0x%x\n",
			      page,
			      dst_rect.block_mask, state.copy_write_block_mask);

			if (invalidate_cached_textures(state.short_term_cached_textures, dst_rect.block_mask, dst_rect.write_mask, UINT32_MAX))
				flush_cached();
		}
	}

	for (unsigned y = 0; y < src_rect.page_height; y++)
	{
		for (unsigned x = 0; x < src_rect.page_width; x++)
		{
			unsigned page = src_rect.base_page + y * src_rect.page_stride + x;
			page &= page_state_mask;
			auto &state = page_state[page];

			register_accessed_copy_pages(page);
			register_accessed_readback_page(page);

			if ((src_rect.block_mask & state.copy_write_block_mask) != 0)
				has_hazard = true;

			state.need_host_write_timeline_mask |= dst_rect.block_mask;
			state.copy_read_block_mask |= src_rect.block_mask;

			TRACE("TRACKER || PAGE 0x%x, READ |= 0x%x -> 0x%x\n",
			      page & page_state_mask,
			      src_rect.block_mask, state.copy_read_block_mask);

			// If we detect a COPY hazard here, it means that we have overlapping copy, and need to handle it carefully.
		}
	}

	if (has_hazard)
	{
		for (unsigned y = 0; y < src_rect.page_height; y++)
		{
			for (unsigned x = 0; x < src_rect.page_width; x++)
			{
				unsigned page = src_rect.base_page + y * src_rect.page_stride + x;
				page &= page_state_mask;
				auto &state = page_state[page];

				if ((state.flags & PAGE_STATE_NEEDS_SHADOW_PAGE_BIT) == 0)
				{
					state.flags |= PAGE_STATE_NEEDS_SHADOW_PAGE_BIT;
					cb.sync_shadow_page(page);
					accessed_shadow_pages.push_back(page);
				}
			}
		}
	}

	if (need_tex_invalidate)
		invalidate_texture_cache(UINT32_MAX);

	return has_hazard;
}

void PageTracker::mark_texture_read(const PageRect &rect)
{
	// Early-out if there cannot possibly be a hazard.
	if ((rect.write_mask & pending_fb_write_mask) == 0)
		return;

	uint32_t start_page = rect.base_page;
	uint32_t end_page = rect.base_page +
	                    (rect.page_height - 1) * rect.page_stride +
	                    rect.page_width - 1;

	if (end_page < pending_fb_write_page_lo || start_page > pending_fb_write_page_hi)
		return;

	// Strict interpretation of minimal caching.
	// Lots of content forgets TEXFLUSH, insert it automatically if we're trying to read after write.
	if (page_has_fb_write(rect))
		flush_render_pass(FlushReason::TextureHazard);
}

void PageTracker::register_cached_clut_clobber(const PageRectCLUT &rect)
{
	csa_written_mask |= rect.csa_mask;
	//TRACE("TRACKER || CSA mask |= 0x%x -> 0x%x\n", rect.csa_mask, csa_written_mask);

	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;
			page &= page_state_mask;
			auto &state = page_state[page];
			// Need to resolve WAR hazard.
			register_accessed_cache_pages(page);
			register_accessed_readback_page(page);
			state.need_host_write_timeline_mask |= rect.block_mask;
			state.cached_read_block_mask |= rect.block_mask;
			TRACE("TRACKER || PAGE 0x%x, CACHED |= 0x%x -> 0x%x\n",
				  page,
				  rect.block_mask, state.cached_read_block_mask);
		}
	}
}

void PageTracker::register_short_term_cached_texture(
		const PageRect *level_rect, uint32_t levels, Util::Hash hash)
{
	CachedTexture *handle = cached_texture_pool.allocate(cached_texture_pool);
	CachedTextureHandle tex{handle};
	handle->set_hash(hash);

	for (unsigned level = 0; level < levels; level++)
	{
		auto &rect = level_rect[level];

		for (unsigned y = 0; y < rect.page_height; y++)
		{
			for (unsigned x = 0; x < rect.page_width; x++)
			{
				unsigned page = rect.base_page + y * rect.page_stride + x;
				page &= page_state_mask;

				auto &state = page_state[page];

				// Need to resolve WAR hazard.
				register_accessed_readback_page(page);
				register_accessed_cache_pages(page);
				state.need_host_write_timeline_mask |= rect.block_mask;
				state.cached_read_block_mask |= rect.block_mask;

				TRACE("TRACKER || PAGE 0x%x, CACHED |= 0x%x -> 0x%x\n",
				      page, rect.block_mask, state.cached_read_block_mask);

				garbage_collect_texture_masked_handles(state.short_term_cached_textures);

				if (state.short_term_cached_textures.empty())
					short_term_cache_pages.push_back(page);
				state.short_term_cached_textures.push_back({ tex, rect.block_mask, rect.write_mask, UINT32_MAX });
			}
		}
	}
}

bool PageTracker::has_punchthrough_host_write(const PageRect &rect) const
{
	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;
			auto &state = page_state[page & page_state_mask];

			// This is quite hacky, and should just be seen as a heuristic.
			if (state.punchthrough_host_write_mask == 0 ||
			    state.copy_write_block_mask != 0 ||
			    state.cached_read_block_mask != 0)
			{
				return false;
			}
		}
	}

	return true;
}

PageTracker::UploadStrategy
PageTracker::register_cached_texture(const PageRect *level_rect, uint32_t levels,
                                     uint32_t csa_mask, uint32_t clut_instance,
                                     Util::Hash hash, Vulkan::ImageHandle image)
{
	CachedTexture *handle = cached_texture_pool.allocate(cached_texture_pool);
	handle->set_hash(hash);
	handle->image = std::move(image);

	auto *delete_t = cached_textures.insert_yield(handle);
	// We should always have called find_cached_texture before creating a new one.
	(void)delete_t;
	assert(!delete_t);

	CachedTextureHandle tex{handle};

	assert(levels > 0);
	assert(level_rect[0].page_width && level_rect[0].page_height);

	// Only bother trying to optimize small uploads.
	// There must be no hazards, in the sense that it's safe to just read the CPU VRAM.
	// In this case we elide the marking of CACHED reads on the pages.
	bool promote_to_cpu = levels == 1 &&
	                      level_rect[0].page_width == 1 &&
	                      level_rect[0].page_height == 1 &&
	                      (has_punchthrough_host_write(level_rect[0]) ||
	                       get_host_read_timeline(level_rect[0]) <= cb.query_timeline());

	for (unsigned level = 0; level < levels; level++)
	{
		auto &rect = level_rect[level];

		for (unsigned y = 0; y < rect.page_height; y++)
		{
			for (unsigned x = 0; x < rect.page_width; x++)
			{
				unsigned page = rect.base_page + y * rect.page_stride + x;
				page &= page_state_mask;

				auto &state = page_state[page];

				if (!promote_to_cpu)
				{
					// Need to resolve WAR hazard.
					register_accessed_readback_page(page);
					register_accessed_cache_pages(page);
					state.need_host_write_timeline_mask |= rect.block_mask;
					state.cached_read_block_mask |= rect.block_mask;
				}

				TRACE("TRACKER || PAGE 0x%x, CACHED |= 0x%x -> 0x%x\n",
				      page, rect.block_mask, state.cached_read_block_mask);

				garbage_collect_texture_masked_handles(state.cached_textures);
				state.cached_textures.push_back({ tex, rect.block_mask, rect.write_mask, clut_instance });
			}
		}
	}

	// If CLUT is clobbered, we'll have to invalidate too.
	if (csa_mask != 0)
	{
		garbage_collect_texture_masked_handles(texture_cached_palette);
		texture_cached_palette.push_back({ std::move(tex), csa_mask, UINT32_MAX, clut_instance });
	}

	return promote_to_cpu ? UploadStrategy::CPU : UploadStrategy::GPU;
}

void PageTracker::flush_render_pass(FlushReason reason)
{
	clear_cache_pages();
	clear_copy_pages();
	clear_fb_pages();
	// While TEXFLUSH is necessary, plenty of content do not do this properly.
	invalidate_texture_cache(UINT32_MAX);

	// Call this after texture cache invalication so we can recycle textures more aggressively.
	cb.flush(PAGE_TRACKER_FLUSH_FB_ALL, reason);

	TRACE("TRACKER || FLUSH RENDER PASS\n");
}

void PageTracker::clear_copy_pages()
{
	for (uint32_t page_index : accessed_copy_pages)
	{
		auto &page = page_state[page_index];
		page.copy_read_block_mask = 0;
		page.copy_write_block_mask = 0;
	}
	accessed_copy_pages.clear();

	for (uint32_t page_index : accessed_shadow_pages)
	{
		auto &page = page_state[page_index];
		page.flags &= ~PAGE_STATE_NEEDS_SHADOW_PAGE_BIT;
	}
	accessed_shadow_pages.clear();
}

void PageTracker::clear_cache_pages()
{
	for (uint32_t page_index : accessed_cache_pages)
		page_state[page_index].cached_read_block_mask = 0;
	accessed_cache_pages.clear();

	// If we have memoized data earlier in this render pass, need to forget
	// that and requery properly.
	// This is important if we're doing copy -> sample -> copy -> sample.
	cb.forget_in_render_pass_memoization();
}

void PageTracker::clear_fb_pages()
{
	for (uint32_t page_index : accessed_fb_pages)
	{
		auto &page = page_state[page_index];
		page.fb_read_mask = 0;
		page.fb_write_mask = 0;
		page.pending_fb_access_mask = 0;
	}
	accessed_fb_pages.clear();

	pending_fb_write_page_lo = UINT32_MAX;
	pending_fb_write_page_hi = 0;
	pending_fb_write_mask = 0;

	// Once a FB is flushed, these references are dead.
	for (uint32_t page_index : short_term_cache_pages)
	{
		auto &page = page_state[page_index];
		page.short_term_cached_textures.clear();
	}
	short_term_cache_pages.clear();

	// Once a FB is flushed, we can no longer hold on to CLUT cached images
	// which are in a floating state.
	for (auto &cached : texture_cached_palette)
		if (cached.tex->status == CachedTexture::Status::Floating)
			cached.tex->status = CachedTexture::Status::Dead;
}

void PageTracker::flush_copy()
{
	cb.flush(PAGE_TRACKER_FLUSH_COPY_ALL, FlushReason::CopyHazard);
	clear_copy_pages();
	TRACE("TRACKER || FLUSH COPY\n");
}

void PageTracker::flush_cached()
{
	cb.flush(PAGE_TRACKER_FLUSH_CACHE_ALL, FlushReason::TextureHazard);

	clear_copy_pages();
	clear_cache_pages();

	TRACE("TRACKER || FLUSH CACHED\n");
}

BlockState PageTracker::get_block_state(const PageRect &rect) const
{
	BlockState block = {};

	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;
			auto &state = page_state[page & page_state_mask];
			block.cached_read_block_mask |= state.cached_read_block_mask;
			block.copy_write_block_mask |= state.copy_write_block_mask;
			block.copy_read_block_mask |= state.copy_read_block_mask;
		}
	}

	return block;
}

void PageTracker::invalidate_fb_write_short_term_references()
{
	for (uint32_t page : accessed_fb_pages)
	{
		auto &state = page_state[page];
		if (!state.short_term_cached_textures.empty())
			invalidate_cached_textures(state.short_term_cached_textures, UINT32_MAX, state.fb_write_mask, UINT32_MAX);
	}
}

void PageTracker::mark_transfer_write(const PageRect &rect)
{
	bool need_tex_invalidate = false;

	// There are hazards if there is pending work that is dispatched after or concurrently.
	auto block = get_block_state(rect);
	if (page_has_fb_read_write(rect))
	{
		flush_render_pass(FlushReason::CopyHazard);
	}
	else if ((block.cached_read_block_mask & rect.block_mask) != 0)
	{
		flush_cached();
		need_tex_invalidate = true;
	}
	else if ((block.copy_read_block_mask & rect.block_mask) != 0)
		flush_copy();

	// Write-after-Write hazards for copies are handled internally through atomics.
	// We only need to care about write-after-read and read-after-write.

	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;
			page &= page_state_mask;
			auto &state = page_state[page];

			register_accessed_readback_page(page);
			register_accessed_copy_pages(page);
			register_potential_invalidated_indices(page);

			state.need_host_write_timeline_mask |= rect.block_mask;
			state.need_host_read_timeline_mask |= rect.block_mask;
			state.copy_write_block_mask |= rect.block_mask;
			state.texture_cache_needs_invalidate_block_mask |= rect.block_mask;
			state.texture_cache_needs_invalidate_write_mask |= rect.write_mask;

			state.flags &= ~PAGE_STATE_MAY_SUPER_SAMPLE_BIT;

			TRACE("TRACKER || PAGE 0x%x, WRITE |= 0x%x -> 0x%x\n",
			      page,
			      rect.block_mask, state.copy_write_block_mask);

			if (invalidate_cached_textures(state.short_term_cached_textures, rect.block_mask, rect.write_mask, UINT32_MAX))
				flush_cached();
		}
	}

	if (need_tex_invalidate)
		invalidate_texture_cache(UINT32_MAX);
}

bool PageTracker::acquire_host_write(const PageRect &rect, uint64_t max_timeline)
{
	if (page_has_host_write_timeline_update(rect))
	{
		// We have not committed to a host write timeline yet, because there are unflushed reads or writes.
		return false;
	}

	uint64_t write_timeline = 0;

	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;
			page &= page_state_mask;
			write_timeline = std::max<uint64_t>(write_timeline, page_state[page].host_write_timeline);
		}
	}

	if (write_timeline > max_timeline)
		return false;

	return true;
}

void PageTracker::commit_host_write(const PageRect &rect)
{
	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;
			page &= page_state_mask;
			cb.sync_host_vram_page(page, rect.block_mask);
			auto &state = page_state[page];
			state.flags &= ~PAGE_STATE_MAY_SUPER_SAMPLE_BIT;

			register_potential_invalidated_indices(page);
			state.texture_cache_needs_invalidate_block_mask |= UINT32_MAX;
			state.texture_cache_needs_invalidate_write_mask |= UINT32_MAX;
		}
	}
}

void PageTracker::commit_punchthrough_host_write(const ParallelGS::PageRect &rect)
{
	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;
			page &= page_state_mask;
			auto &state = page_state[page];

			register_accessed_readback_page(page);
			state.punchthrough_host_write_mask |= rect.block_mask;
		}
	}
}

void PageTracker::register_potential_invalidated_indices(uint32_t page)
{
	auto &state = page_state[page];
	if (state.texture_cache_needs_invalidate_block_mask == 0)
		potential_invalidated_indices.push_back(page);
}

void PageTracker::register_accessed_cache_pages(uint32_t page)
{
	auto &state = page_state[page];
	if (state.cached_read_block_mask == 0)
		accessed_cache_pages.push_back(page);
}

void PageTracker::register_accessed_readback_page(uint32_t page)
{
	auto &state = page_state[page];
	if (state.need_host_write_timeline_mask == 0 &&
	    state.need_host_read_timeline_mask == 0 &&
	    state.punchthrough_host_write_mask == 0)
	{
		accessed_readback_pages.push_back(page);
	}
}

void PageTracker::register_accessed_fb_pages(uint32_t page)
{
	auto &state = page_state[page];
	if (state.fb_read_mask == 0 && state.fb_write_mask == 0)
		accessed_fb_pages.push_back(page);
}

void PageTracker::register_accessed_copy_pages(uint32_t page)
{
	auto &state = page_state[page];
	if (state.copy_read_block_mask == 0 && state.copy_write_block_mask == 0)
		accessed_copy_pages.push_back(page);
}

uint64_t PageTracker::get_host_read_timeline(const PageRect &rect) const
{
	if (page_has_host_read_timeline_update(rect))
	{
		// We have not committed to a host read timeline yet, because there are unflushed writes.
		return UINT64_MAX;
	}

	uint64_t read_timeline = 0;

	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;
			page &= page_state_mask;
			read_timeline = std::max<uint64_t>(read_timeline, page_state[page].host_read_timeline);
		}
	}

	return read_timeline;
}

uint64_t PageTracker::get_submitted_host_write_timeline(const PageRect &rect) const
{
	uint64_t write_timeline = 0;

	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;
			page &= page_state_mask;
			write_timeline = std::max<uint64_t>(write_timeline, page_state[page].host_write_timeline);
		}
	}

	return write_timeline;
}

uint64_t PageTracker::get_host_write_timeline(const PageRect &rect) const
{
	if (page_has_host_write_timeline_update(rect))
	{
		// We have not committed to a host write timeline yet, because there are unflushed writes or reads.
		return UINT64_MAX;
	}

	return get_submitted_host_write_timeline(rect);
}

Vulkan::ImageHandle PageTracker::find_cached_texture(Util::Hash hash) const
{
	auto *cached_texture = cached_textures.find(hash);
	if (!cached_texture)
		return {};
	return cached_texture->image;
}

void PageTracker::garbage_collect_texture_masked_handles(std::vector<CachedTextureMasked> &textures)
{
	auto itr = Util::unstable_remove_if(textures.begin(), textures.end(), [](const CachedTextureMasked &masked) {
		return masked.tex->status == CachedTexture::Status::Dead;
	});
	textures.erase(itr, textures.end());
}

bool PageTracker::invalidate_cached_textures(
		std::vector<CachedTextureMasked> &textures,
		uint32_t block_mask, uint32_t write_mask, uint32_t clut_instance)
{
	bool did_work = false;

	auto itr = Util::unstable_remove_if(
			textures.begin(), textures.end(),
			[this, block_mask, write_mask, clut_instance, &did_work](CachedTextureMasked &masked) {
				auto &tex = *masked.tex;

				// CLUT invalidation is a soft invalidation. As long as we can use CLUT memoization
				// to get back to the original CLUT index, we can keep reusing the image,
				// but only within the same render pass.
				bool is_clut_invalidation = clut_instance != UINT32_MAX;

				bool can_invalidate =
						tex.status == CachedTexture::Status::Live ||
						(tex.status == CachedTexture::Status::Floating && !is_clut_invalidation);

				if (can_invalidate &&
				    (masked.block_mask & block_mask) != 0 &&
				    (masked.write_mask & write_mask) != 0 &&
				    (clut_instance == UINT32_MAX || masked.clut_instance != clut_instance))
				{
					// When we transition away from Live status, remove it from the hashmap lookup.
					// Handles may persist until render pass end.
					if (tex.status == CachedTexture::Status::Live)
						cached_textures.erase(masked.tex.get());

					// If we only invalidated texture due to palette cache being clobbered,
					// we may be able to ignore the invalidation and keep it alive in the render pass cache if
					// we sample the texture with same memoized CLUT instance once again.
					// Essentially, we defer the invalidation until the same texture is used with a different palette instance.
					cb.invalidate_texture_hash(tex.get_hash(), is_clut_invalidation);
					if (tex.image)
					{
						cb.recycle_image_handle(std::move(tex.image));
						tex.image = {};
					}

					// If the image was only invalidated for CLUT, it can remain live until.
					tex.status = is_clut_invalidation ?
					             CachedTexture::Status::Floating :
					             CachedTexture::Status::Dead;

					did_work = true;
				}

				// Can only reap the handle when it's truly dead.
				return tex.status == CachedTexture::Status::Dead;
			});

	textures.erase(itr, textures.end());
	return did_work;
}

void PageTracker::invalidate_texture_cache(uint32_t clut_instance)
{
	bool invalidated = false;

	for (auto index : potential_invalidated_indices)
	{
		auto &page = page_state[index];

		if (!page.cached_textures.empty())
		{
			TRACE("TRACKER || PAGE 0x%x, invalidate texture mask 0x%x\n", unsigned(&page - page_state.data()),
			      page.texture_cache_needs_invalidate_mask);
			bool did_work = invalidate_cached_textures(
					page.cached_textures,
					page.texture_cache_needs_invalidate_block_mask,
					page.texture_cache_needs_invalidate_write_mask, UINT32_MAX);
			invalidated = invalidated || did_work;
		}
		page.texture_cache_needs_invalidate_block_mask = 0;
		page.texture_cache_needs_invalidate_write_mask = 0;
	}

	potential_invalidated_indices.clear();

	if (csa_written_mask)
	{
		TRACE("TRACKER || invalidate CSA mask 0x%x\n", csa_written_mask);
		bool did_work = invalidate_cached_textures(texture_cached_palette, csa_written_mask, UINT32_MAX, clut_instance);
		invalidated = invalidated || did_work;
		csa_written_mask = 0;
	}

	if (invalidated)
		cb.mark_texture_state_dirty_with_flush();
}

uint64_t PageTracker::mark_submission_timeline(FlushReason reason)
{
	++timeline;

	flush_render_pass(reason);

	for (uint32_t page_index : accessed_readback_pages)
	{
		auto &page = page_state[page_index];

		if (page.need_host_read_timeline_mask != 0)
		{
			page.host_read_timeline = timeline;
			cb.sync_vram_host_page(page_index);
		}

		if (page.need_host_write_timeline_mask != 0)
			page.host_write_timeline = timeline;

		page.need_host_read_timeline_mask = 0;
		page.need_host_write_timeline_mask = 0;
		page.punchthrough_host_write_mask = 0;
	}

	accessed_readback_pages.clear();

	cb.flush(PAGE_TRACKER_FLUSH_WRITE_BACK_BIT, reason);
	return timeline;
}

void CachedTextureDeleter::operator()(CachedTexture *texture)
{
	texture->pool.free(texture);
}
}