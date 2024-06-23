// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#include "page_tracker.hpp"
#include "unstable_remove_if.hpp"
#include <algorithm>

#if 0 && defined(PARALLEL_GS_DEBUG) && PARALLEL_GS_DEBUG
#define TRACE(...) printf(__VA_ARGS__)
#else
#define TRACE(...) ((void)0)
#endif

namespace ParallelGS
{
PageTracker::PageTracker(PageTrackerCallback &cb_)
	: cb(cb_)
{
}

void PageTracker::set_num_pages(unsigned num_pages)
{
	page_state_mask = num_pages - 1;
	page_state.resize(num_pages);
}

bool PageTracker::page_has_flag(const PageRect &rect, PageStateFlags flags) const
{
	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;
			auto &state = page_state[page & page_state_mask];
			if ((state.flags & flags) != 0)
				return true;
		}
	}

	return false;
}

bool PageTracker::page_has_flag_with_fb_access_mask(
		const PageRect &rect, PageStateFlags flags, uint32_t write_mask) const
{
	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;
			auto &state = page_state[page & page_state_mask];
			if ((state.flags & flags) != 0 && (state.pending_fb_access_mask & write_mask) != 0)
				return true;
		}
	}

	return false;
}

void PageTracker::mark_fb_write(const PageRect &rect)
{
	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;
			page &= page_state_mask;
			auto &state = page_state[page];
			state.flags |= PAGE_STATE_FB_WRITE_BIT | PAGE_STATE_FB_READ_BIT |
			               PAGE_STATE_TIMELINE_UPDATE_HOST_WRITE_BIT |
			               PAGE_STATE_TIMELINE_UPDATE_HOST_READ_BIT;
			if (state.texture_cache_needs_invalidate_block_mask == 0)
				potential_invalidated_indices.push_back(page);
			state.texture_cache_needs_invalidate_block_mask |= UINT32_MAX;
			state.texture_cache_needs_invalidate_write_mask |= UINT32_MAX;
			state.pending_fb_access_mask |= rect.write_mask;
			TRACE("TRACKER || PAGE 0x%x, FB write\n", page);
		}
	}
}

void PageTracker::mark_fb_read(const PageRect &rect)
{
	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;
			auto &state = page_state[page & page_state_mask];
			state.flags |= PAGE_STATE_FB_READ_BIT | PAGE_STATE_TIMELINE_UPDATE_HOST_WRITE_BIT;
			state.pending_fb_access_mask |= rect.write_mask;
			TRACE("TRACKER || PAGE 0x%x, FB read\n", page & page_state_mask);
		}
	}
}

// If 1, assumes that every primitive calls TEXFLUSH.
// If 0, assumes that we have an infinitely large texture cache
// which is only cleared on explicit TEXFLUSH.
#define PAGE_TRACKER_MINIMAL_CACHING 1

void PageTracker::mark_transfer_copy(const PageRect &dst_rect, const PageRect &src_rect)
{
	auto dst_block = get_block_state(dst_rect);
	auto src_block = get_block_state(src_rect);

#if PAGE_TRACKER_MINIMAL_CACHING
	bool need_tex_invalidate = false;
#endif

	if (page_has_flag_with_fb_access_mask(
			dst_rect, PAGE_STATE_FB_WRITE_BIT | PAGE_STATE_FB_READ_BIT, dst_rect.write_mask) ||
	    page_has_flag_with_fb_access_mask(
			src_rect, PAGE_STATE_FB_WRITE_BIT, src_rect.write_mask))
	{
		flush_render_pass(FlushReason::CopyHazard);
	}
	else if ((dst_block.cached_read_block_mask & dst_rect.block_mask) != 0)
	{
		flush_cached();
#if PAGE_TRACKER_MINIMAL_CACHING
		need_tex_invalidate = true;
#endif
	}
	else if (((dst_block.copy_write_block_mask | dst_block.copy_read_block_mask) & dst_rect.block_mask) != 0 ||
	         (src_block.copy_write_block_mask & src_rect.block_mask) != 0)
	{
		flush_copy();
	}

	for (unsigned y = 0; y < dst_rect.page_height; y++)
	{
		for (unsigned x = 0; x < dst_rect.page_width; x++)
		{
			unsigned page = dst_rect.base_page + y * dst_rect.page_stride + x;
			page &= page_state_mask;
			auto &state = page_state[page];
			state.flags |= PAGE_STATE_TIMELINE_UPDATE_HOST_WRITE_BIT |
			               PAGE_STATE_TIMELINE_UPDATE_HOST_READ_BIT;
			state.copy_write_block_mask |= dst_rect.block_mask;
			if (state.texture_cache_needs_invalidate_block_mask == 0)
				potential_invalidated_indices.push_back(page);
			state.texture_cache_needs_invalidate_block_mask |= dst_rect.block_mask;
			state.texture_cache_needs_invalidate_write_mask |= dst_rect.write_mask;
			TRACE("TRACKER || PAGE 0x%x, WRITE |= 0x%x -> 0x%x\n",
			      page,
			      dst_rect.block_mask, state.copy_write_block_mask);
		}
	}

	for (unsigned y = 0; y < src_rect.page_height; y++)
	{
		for (unsigned x = 0; x < src_rect.page_width; x++)
		{
			unsigned page = src_rect.base_page + y * src_rect.page_stride + x;
			auto &state = page_state[page & page_state_mask];
			state.flags |= PAGE_STATE_TIMELINE_UPDATE_HOST_WRITE_BIT;
			state.copy_read_block_mask |= src_rect.block_mask;
			TRACE("TRACKER || PAGE 0x%x, READ |= 0x%x -> 0x%x\n",
			      page & page_state_mask,
			      src_rect.block_mask, state.copy_read_block_mask);
		}
	}

#if PAGE_TRACKER_MINIMAL_CACHING
	if (need_tex_invalidate)
		invalidate_texture_cache(UINT32_MAX);
#endif
}

void PageTracker::mark_texture_read(const PageRect &rect)
{
#if PAGE_TRACKER_MINIMAL_CACHING
	// Strict interpretation of minimal caching.
	// Lots of content forgets TEXFLUSH, insert it automatically if we're trying to read after write.
	if (page_has_flag_with_fb_access_mask(
			rect, PAGE_STATE_FB_WRITE_BIT | PAGE_STATE_FLUSH_FB_ON_CACHED_READ_BIT, rect.write_mask))
		flush_render_pass(FlushReason::TextureHazard);
#else
	// Maximal caching assumption.
	if (page_has_flag_with_fb_access_mask(
			rect, PAGE_STATE_FLUSH_FB_ON_CACHED_READ_BIT, rect.write_mask))
		flush_render_pass(FlushReason::TextureHazard);
#endif
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
			auto &state = page_state[page & page_state_mask];
			// Need to resolve WAR hazard.
			state.flags |= PAGE_STATE_TIMELINE_UPDATE_HOST_WRITE_BIT;
			state.cached_read_block_mask |= rect.block_mask;
			TRACE("TRACKER || PAGE 0x%x, CACHED |= 0x%x -> 0x%x\n",
				  page & page_state_mask,
				  rect.block_mask, state.cached_read_block_mask);
		}
	}
}

void PageTracker::register_cached_texture(const PageRect *level_rect, uint32_t levels,
                                          uint32_t csa_mask, uint32_t clut_instance,
                                          Util::Hash hash, Vulkan::ImageHandle image)
{
	CachedTexture *handle = cached_texture_pool.allocate(cached_texture_pool);
	handle->set_hash(hash);
	handle->image = std::move(image);

	CachedTextureHandle delete_t{cached_textures.insert_yield(handle)};
	CachedTextureHandle tex{handle};

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
				state.flags |= PAGE_STATE_TIMELINE_UPDATE_HOST_WRITE_BIT;
				state.cached_read_block_mask |= rect.block_mask;

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
}

void PageTracker::clear_page_flags(PageStateFlags flags)
{
	for (auto &page : page_state)
		page.flags &= ~flags;
}

void PageTracker::flush_render_pass(FlushReason reason)
{
	cb.flush(PAGE_TRACKER_FLUSH_FB_ALL, reason);
	clear_page_flags(PAGE_STATE_FB_WRITE_BIT | PAGE_STATE_FB_READ_BIT | PAGE_STATE_FLUSH_FB_ON_CACHED_READ_BIT);
	for (auto &page : page_state)
	{
		page.copy_read_block_mask = 0;
		page.copy_write_block_mask = 0;
		page.cached_read_block_mask = 0;
		page.pending_fb_access_mask = 0;
	}

#if PAGE_TRACKER_MINIMAL_CACHING
	// While TEXFLUSH is necessary, plenty of content do not do this properly.
	invalidate_texture_cache(UINT32_MAX);
#endif

	TRACE("TRACKER || FLUSH RENDER PASS\n");
}

void PageTracker::flush_copy()
{
	cb.flush(PAGE_TRACKER_FLUSH_COPY_ALL, FlushReason::CopyHazard);
	for (auto &page : page_state)
	{
		page.copy_read_block_mask = 0;
		page.copy_write_block_mask = 0;
	}

	TRACE("TRACKER || FLUSH COPY\n");
}

void PageTracker::flush_cached()
{
	cb.flush(PAGE_TRACKER_FLUSH_CACHE_ALL, FlushReason::TextureHazard);
	for (auto &page : page_state)
	{
		page.copy_read_block_mask = 0;
		page.copy_write_block_mask = 0;
		page.cached_read_block_mask = 0;
	}

	// If we have memoized data earlier in this render pass, need to forget
	// that and requery properly.
	// This is important if we're doing copy -> sample -> copy -> sample.
	cb.forget_in_render_pass_memoization();

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

void PageTracker::mark_transfer_write(const PageRect &rect)
{
#if PAGE_TRACKER_MINIMAL_CACHING
	bool need_tex_invalidate = false;
#endif

	// There are hazards if there is pending work that is dispatched after or concurrently.
	auto block = get_block_state(rect);
	if (page_has_flag_with_fb_access_mask(
			rect, PAGE_STATE_FB_WRITE_BIT | PAGE_STATE_FB_READ_BIT, rect.write_mask) != 0)
	{
		flush_render_pass(FlushReason::CopyHazard);
	}
	else if ((block.cached_read_block_mask & rect.block_mask) != 0)
	{
		flush_cached();
#if PAGE_TRACKER_MINIMAL_CACHING
		need_tex_invalidate = true;
#endif
	}
	else if (((block.copy_read_block_mask | block.copy_write_block_mask) & rect.block_mask) != 0)
		flush_copy();

	for (unsigned y = 0; y < rect.page_height; y++)
	{
		for (unsigned x = 0; x < rect.page_width; x++)
		{
			unsigned page = rect.base_page + y * rect.page_stride + x;
			page &= page_state_mask;
			auto &state = page_state[page];
			state.flags |= PAGE_STATE_TIMELINE_UPDATE_HOST_READ_BIT |
			               PAGE_STATE_TIMELINE_UPDATE_HOST_WRITE_BIT;
			state.copy_write_block_mask |= rect.block_mask;
			if (state.texture_cache_needs_invalidate_block_mask == 0)
				potential_invalidated_indices.push_back(page);
			state.texture_cache_needs_invalidate_block_mask |= rect.block_mask;
			state.texture_cache_needs_invalidate_write_mask |= rect.write_mask;

			TRACE("TRACKER || PAGE 0x%x, WRITE |= 0x%x -> 0x%x\n",
			      page,
			      rect.block_mask, state.copy_write_block_mask);
		}
	}

#if PAGE_TRACKER_MINIMAL_CACHING
	if (need_tex_invalidate)
		invalidate_texture_cache(UINT32_MAX);
#endif
}

bool PageTracker::acquire_host_write(const PageRect &rect, uint64_t max_timeline)
{
	if (page_has_flag(rect, PAGE_STATE_TIMELINE_UPDATE_HOST_WRITE_BIT))
	{
		// We have not committed to a host write timeline yet, because there are unflushed writes.
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
			cb.sync_host_vram_page(page);
			auto &state = page_state[page];
			if (state.texture_cache_needs_invalidate_block_mask == 0)
				potential_invalidated_indices.push_back(page);
			state.texture_cache_needs_invalidate_block_mask |= UINT32_MAX;
			state.texture_cache_needs_invalidate_write_mask |= UINT32_MAX;
		}
	}
}

uint64_t PageTracker::get_host_read_timeline(const PageRect &rect) const
{
	if (page_has_flag(rect, PAGE_STATE_TIMELINE_UPDATE_HOST_READ_BIT))
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

uint64_t PageTracker::get_host_write_timeline(const PageRect &rect) const
{
	if (page_has_flag(rect, PAGE_STATE_TIMELINE_UPDATE_HOST_WRITE_BIT))
	{
		// We have not committed to a host write timeline yet, because there are unflushed writes or reads.
		return UINT64_MAX;
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

	return write_timeline;
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
		return !masked.tex->image;
	});
	textures.erase(itr, textures.end());
}

bool PageTracker::invalidate_cached_textures(
		std::vector<CachedTextureMasked> &textures, uint32_t block_mask, uint32_t write_mask, uint32_t clut_instance)
{
	auto itr = Util::unstable_remove_if(textures.begin(), textures.end(),
	                                    [block_mask, write_mask, clut_instance](const CachedTextureMasked &masked) {
		                                    return !masked.tex->image ||
		                                           ((masked.block_mask & block_mask) != 0 &&
		                                            (masked.write_mask & write_mask) != 0 &&
		                                            masked.clut_instance != clut_instance);
	                                    });

	bool did_work = false;

	for (auto erase_itr = itr; erase_itr != textures.end(); ++erase_itr)
	{
		auto &tex = *erase_itr;
		if (tex.tex->image)
		{
			// If we only invalidated texture due to palette cache being clobbered,
			// we may be able to ignore the invalidation and keep it alive in the render pass cache if
			// we sample the texture with same memoized CLUT instance once again.
			// Essentially, we defer the invalidation until the same texture is used with a different palette instance.
			cb.invalidate_texture_hash(tex.tex->get_hash(), clut_instance != UINT32_MAX);
			tex.tex->image.reset();
			cached_textures.erase(tex.tex.get());
			did_work = true;
		}
	}

	textures.erase(itr, textures.end());
	return did_work;
}

bool PageTracker::invalidate_texture_cache(uint32_t clut_instance)
{
	bool invalidated = false;

	for (auto index : potential_invalidated_indices)
	{
		auto &page = page_state[index];

		// Delay flushing copy and render pass until we know we have to,
		// i.e., we're actually reading from that page.
		if ((page.flags & PAGE_STATE_FB_WRITE_BIT) != 0)
			page.flags |= PAGE_STATE_FLUSH_FB_ON_CACHED_READ_BIT;

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

	return invalidated;
}

uint64_t PageTracker::mark_submission_timeline()
{
	++timeline;

	flush_render_pass(FlushReason::SubmissionFlush);

	uint32_t page_index = 0;

	for (auto &page : page_state)
	{
		if ((page.flags & PAGE_STATE_TIMELINE_UPDATE_HOST_READ_BIT) != 0)
		{
			page.host_read_timeline = timeline;
			cb.sync_vram_host_page(page_index);
		}

		if ((page.flags & PAGE_STATE_TIMELINE_UPDATE_HOST_WRITE_BIT) != 0)
			page.host_write_timeline = timeline;

		page.flags &= ~(PAGE_STATE_TIMELINE_UPDATE_HOST_READ_BIT | PAGE_STATE_TIMELINE_UPDATE_HOST_WRITE_BIT);
		page_index++;
	}

	cb.flush(PAGE_TRACKER_FLUSH_WRITE_BACK_BIT, FlushReason::SubmissionFlush);
	return timeline;
}

void CachedTextureDeleter::operator()(CachedTexture *texture)
{
	texture->pool.free(texture);
}
}