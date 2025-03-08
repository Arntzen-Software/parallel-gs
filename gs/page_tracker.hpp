// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#pragma once
#include <vector>
#include "intrusive_hash_map.hpp"
#include "hashmap.hpp"
#include "image.hpp"

namespace ParallelGS
{
struct CachedTexture;
struct CachedTextureDeleter
{
	void operator()(CachedTexture *);
};

struct CachedTexture : Util::IntrusiveHashMapEnabled<CachedTexture>,
                       Util::IntrusivePtrEnabled<CachedTexture, CachedTextureDeleter>
{
	explicit CachedTexture(Util::ObjectPool<CachedTexture> &pool_) : pool(pool_) {}
	Util::ObjectPool<CachedTexture> &pool;
	Vulkan::ImageHandle image;

	enum class Status { Live, Floating, Dead };
	Status status = Status::Live;
};
using CachedTextureHandle = Util::IntrusivePtr<CachedTexture>;

enum PageStateFlagBits : uint32_t
{
	PAGE_STATE_NEEDS_SHADOW_PAGE_BIT = 1 << 0,
	PAGE_STATE_MAY_SUPER_SAMPLE_BIT = 1 << 1
};
using PageStateFlags = uint32_t;

struct CachedTextureMasked
{
	CachedTextureHandle tex;
	uint32_t block_mask;
	uint32_t write_mask;
	uint32_t clut_instance;
};

struct BlockState
{
	uint32_t copy_write_block_mask = 0;
	uint32_t copy_read_block_mask = 0;
	uint32_t cached_read_block_mask = 0;
};

struct PageState
{
	// On TEXFLUSH, we may have to clobber these texture handles if there have been writes to the page.
	std::vector<CachedTextureMasked> cached_textures;
	std::vector<CachedTextureMasked> short_term_cached_textures;

	// To safely read from host memory, this timeline must be reached.
	uint64_t host_read_timeline = 0;
	// To safely write to host memory, this timeline must be reached.
	uint64_t host_write_timeline = 0;

	// Hazards which affect the entire page.
	PageStateFlags flags = 0;

	// Tracked on a per-block (256b) basis.
	// Copies and textures are aligned to 256b and tracking per-page is too pessimistic.
	uint32_t copy_write_block_mask = 0;
	uint32_t copy_read_block_mask = 0;
	uint32_t cached_read_block_mask = 0;
	uint32_t texture_cache_needs_invalidate_block_mask = 0;

	uint32_t fb_write_mask = 0;
	uint32_t fb_read_mask = 0;
	uint32_t need_host_read_timeline_mask = 0;
	uint32_t need_host_write_timeline_mask = 0;
	uint32_t punchthrough_host_write_mask = 0;

	uint32_t pending_fb_access_mask = 0;

	// If copy or rendering writes to a block, it might be a masked write. E.g. 24-bit FB and upper 8-bit is sampled from.
	// If there is no overlap, we don't have to invalidate.
	uint32_t texture_cache_needs_invalidate_write_mask = 0;
};

struct PageRect
{
	uint32_t base_page;
	uint32_t page_width;
	uint32_t page_height;
	uint32_t page_stride;
	uint32_t block_mask;
	uint32_t write_mask;
};

struct PageRectCLUT : PageRect
{
	uint32_t csa_mask;
};

enum PageTrackerFlushFlagBits : uint32_t
{
	PAGE_TRACKER_FLUSH_HOST_VRAM_SYNC_BIT = 1 << 0,
	// Flush all HOST -> LOCAL or LOCAL -> LOCAL copies.
	PAGE_TRACKER_FLUSH_COPY_BIT = 1 << 1,
	// Flush all work copying VRAM into textures.
	PAGE_TRACKER_FLUSH_CACHE_BIT = 1 << 2,
	// Flush render pass.
	PAGE_TRACKER_FLUSH_FB_BIT = 1 << 3,
	// Flush write-back.
	PAGE_TRACKER_FLUSH_WRITE_BACK_BIT = 1 << 4,
	PAGE_TRACKER_FLUSH_FB_ALL = PAGE_TRACKER_FLUSH_HOST_VRAM_SYNC_BIT |
	                            PAGE_TRACKER_FLUSH_CACHE_BIT | PAGE_TRACKER_FLUSH_COPY_BIT | PAGE_TRACKER_FLUSH_FB_BIT,
	PAGE_TRACKER_FLUSH_COPY_ALL = PAGE_TRACKER_FLUSH_HOST_VRAM_SYNC_BIT | PAGE_TRACKER_FLUSH_COPY_BIT,
	PAGE_TRACKER_FLUSH_CACHE_ALL = PAGE_TRACKER_FLUSH_HOST_VRAM_SYNC_BIT | PAGE_TRACKER_FLUSH_COPY_BIT |
	                               PAGE_TRACKER_FLUSH_CACHE_BIT
};
using PageTrackerFlushFlags = uint32_t;

enum class FlushReason
{
	FBPointer,
	Overflow,
	TextureHazard,
	CopyHazard,
	SubmissionFlush,
	HostAccess
};

class GSInterface;

class PageTracker
{
public:
	explicit PageTracker(GSInterface &cb);
	void set_num_pages(unsigned num_pages);

	// If renderer flushes itself due to external memory pressure,
	// need to call these.
	void clear_cache_pages();
	void clear_copy_pages();

	void mark_external_write(const PageRect &rect);

	void mark_fb_write(const PageRect &rect);
	// For read-only depth.
	void mark_fb_read(const PageRect &rect);

	// HOST -> LOCAL
	void mark_transfer_write(const PageRect &rect);
	// LOCAL -> LOCAL
	bool mark_transfer_copy(const PageRect &dst_rect, const PageRect &src_rect);

	// If there are existing writes on a page and TEXFLUSH is called,
	// invalidate all cached textures associated with that page.
	// For invalidation based on palette cache, ignore any flush which matches this CLUT instance.
	// This avoids some false positives where:
	// CLUT is written
	// Texture is uploaded
	// TEXFLUSH is called
	// This would invalidate the texture since CLUT has been written.
	// Use UINT32_MAX to always flush.
	// This kind of CLUT flushing is mostly relevant to avoid false invalidations inside a render pass.
	void invalidate_texture_cache(uint32_t clut_instance_match);

	void invalidate_fb_write_short_term_references();

	void mark_texture_read(const PageRect &rect);
	void register_cached_clut_clobber(const PageRectCLUT &rect);

	enum class UploadStrategy
	{
		GPU,
		CPU
	};
	UploadStrategy register_cached_texture(const PageRect *level_rects, uint32_t num_levels,
	                                       uint32_t csa_mask, uint32_t clut_instance,
	                                       Util::Hash hash, Vulkan::ImageHandle image);

	void register_short_term_cached_texture(const PageRect *level_rects, uint32_t num_levels, Util::Hash hash);

	Vulkan::ImageHandle find_cached_texture(Util::Hash hash) const;

	// If there are hazards, this returns UINT64_MAX. Must explicitly call mark_submission_timeline first.
	uint64_t get_host_read_timeline(const PageRect &rect) const;
	uint64_t get_host_write_timeline(const PageRect &rect) const;
	uint64_t get_submitted_host_write_timeline(const PageRect &rect) const;
	bool acquire_host_write(const PageRect &rect, uint64_t max_timeline);
	void commit_host_write(const PageRect &rect);
	void commit_punchthrough_host_write(const PageRect &rect);

	// Explicitly flush render pass, does not force a submit as well.
	void flush_render_pass(FlushReason reason);

	// Mark an explicit flush. All batched GPU operations will complete and resolve fully.
	// Once the timeline reaches the value in uint64_t, CPU can safely read host copy.
	uint64_t mark_submission_timeline(FlushReason reason = FlushReason::SubmissionFlush);

	bool texture_may_super_sample(const PageRect &rect) const;

	bool page_has_fb_read_write(const PageRect &rect) const;
	bool page_has_fb_write(const PageRect &rect) const;
	bool page_is_copy_cached_sensitive(const PageRect &rect) const;

private:
	GSInterface &cb;
	Util::ObjectPool<CachedTexture> cached_texture_pool;
	Util::IntrusiveHashMapHolder<CachedTexture> cached_textures;
	std::vector<PageState> page_state;
	unsigned page_state_mask = 0;
	uint64_t timeline = 0;
	uint32_t csa_written_mask = 0;
	std::vector<CachedTextureMasked> texture_cached_palette;

	// Accelerate mark_texture_read.
	uint32_t pending_fb_write_page_lo = UINT32_MAX;
	uint32_t pending_fb_write_page_hi = UINT32_MAX;
	uint32_t pending_fb_write_mask = 0;

	std::vector<uint32_t> accessed_fb_pages;
	std::vector<uint32_t> accessed_cache_pages;
	std::vector<uint32_t> accessed_copy_pages;
	std::vector<uint32_t> accessed_readback_pages;
	std::vector<uint32_t> accessed_shadow_pages;
	std::vector<uint32_t> short_term_cache_pages;

	void clear_fb_pages();

	bool invalidate_cached_textures(std::vector<CachedTextureMasked> &textures,
	                                uint32_t block_mask, uint32_t write_mask, uint32_t clut_instance);
	bool page_has_host_write_timeline_update(const PageRect &rect) const;
	bool page_has_host_read_timeline_update(const PageRect &rect) const;
	BlockState get_block_state(const PageRect &rect) const;

	bool has_punchthrough_host_write(const PageRect &rect) const;

	void flush_copy();
	void flush_cached();
	void garbage_collect_texture_masked_handles(std::vector<CachedTextureMasked> &textures);
	std::vector<uint32_t> potential_invalidated_indices;

	void register_potential_invalidated_indices(uint32_t page);
	void register_accessed_readback_page(uint32_t page);
	void register_accessed_copy_pages(uint32_t page);
	void register_accessed_fb_pages(uint32_t page);
	void register_accessed_cache_pages(uint32_t page);
};
}