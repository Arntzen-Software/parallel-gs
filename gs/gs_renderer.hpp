// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#pragma once

#include "gs_registers.hpp"
#include "muglm/muglm.hpp"
#include "device.hpp"
#include "image.hpp"
#include "page_tracker.hpp"
#include "shaders/data_structures.h"
#include "shaders/slangmosh_iface.hpp"
#include <queue>
#include <future>
#include <atomic>
#include <condition_variable>
#include <thread>

namespace ParallelGS
{
struct ScanoutResult
{
	Vulkan::ImageHandle image;
	// Always reported in the single-sampled domain.
	uint32_t internal_width;
	uint32_t internal_height;
	// This is relevant for aspect ratio correction.
	// If only a fraction of the horizontal or vertical region was scanned out,
	// we have to adjust the input aspect ratio.
	// Mode width/height are used as reference for the target output aspect ratio.
	uint32_t mode_width;
	uint32_t mode_height;

	// Set to true if we scanned out at a higher resolution.
	bool high_resolution_scanout;
};

struct FlushStats
{
	VkDeviceSize allocated_scratch_memory;
	VkDeviceSize allocated_image_memory;
	uint32_t num_primitives;
	uint32_t num_render_passes;
	uint32_t num_palette_updates;
	uint32_t num_copies;
	uint32_t num_copy_threads;
	uint32_t num_copy_barriers;
};

enum class TimestampType
{
	SyncHostToVRAM,
	CopyVRAM,
	PaletteUpdate,
	TextureUpload,
	TriangleSetup,
	Binning,
	Shading,
	Readback,
	VSync,
	Count
};

struct TexRect
{
	uint32_t x, y, width, height;
	uint32_t levels;
};

struct TextureDescriptor
{
	Reg64<TEX0Bits> tex0;
	Reg64<TEX1Bits> tex1;
	Reg64<MIPTBPBits> miptbp1_3;
	Reg64<MIPTBPBits> miptbp4_6;
	Reg64<TEXABits> texa;
	Reg64<CLAMPBits> clamp;
	uint32_t palette_bank;
	uint32_t samples;
	uint32_t latest_palette_bank; // Purely for debug, so we can observe CLUT memoization.
	Util::Hash hash; // Purely for debug.

	// This information is purely implied from the desc, so don't compare or hash it.
	TexRect rect;

	inline bool operator==(const TextureDescriptor &other) const
	{
		return tex0.bits == other.tex0.bits &&
		       tex1.bits == other.tex1.bits &&
		       miptbp1_3.bits == other.miptbp1_3.bits &&
		       miptbp4_6.bits == other.miptbp4_6.bits &&
		       texa.bits == other.texa.bits &&
		       clamp.bits == other.clamp.bits &&
		       palette_bank == other.palette_bank &&
		       samples == other.samples;
	}

	inline bool operator!=(const TextureDescriptor &other) const
	{
		return !(*this == other);
	}
};

struct PaletteUploadDescriptor
{
	Reg64<TEX0Bits> tex0;
	Reg64<TEXCLUTBits> texclut;
	// Allows for certain FB forwarding techniques.
	uint32_t csm2_x_bias;
	float csm2_x_scale;
	uint32_t csm1_reference_base;
	uint32_t csm1_mask;
	uint32_t incoming_clut_instance;
	bool fully_replaces_clut_upload(const PaletteUploadDescriptor &old) const;
};

struct CopyDescriptor
{
	Reg64<BITBLTBUFBits> bitbltbuf;
	Reg64<TRXPOSBits> trxpos;
	Reg64<TRXREGBits> trxreg;
	Reg64<TRXDIRBits> trxdir;

	const void *host_data;
	size_t host_data_size;
	size_t host_data_size_offset;
	size_t host_data_size_required;
	bool needs_shadow_vram;
};

struct FBDescriptor
{
	Reg64<FRAMEBits> frame;
	Reg64<ZBUFBits> z;
};

static constexpr uint32_t MaxPrimitivesPerFlush = 64 * 1024;
static constexpr uint32_t MaxStateVectors = Vulkan::VULKAN_MAX_UBO_SIZE / sizeof(StateVector);
static constexpr uint32_t MaxTextures = std::min<uint32_t>(
		Vulkan::VULKAN_MAX_UBO_SIZE / sizeof(TexInfo), Vulkan::VULKAN_NUM_BINDINGS_BINDLESS_VARYING);
static constexpr uint32_t CLUTInstances = 1024;
static constexpr uint32_t PageSize = 8 * 1024;
static constexpr uint32_t CLUTSize = 1024; // This cannot be larger unless we also increase texture index bits.
static constexpr uint32_t MaxRenderPassInstances = 8;

// While 64x MSAA is theoretically possible,
// it can only work on AMD since we'd need wave size of 64.
// It would also be far too slow to be practical.
static constexpr uint32_t MaxSamplingRateLog2 = 2;

struct TextureInfo
{
	const Vulkan::ImageView *view;
	TexInfo info;
};

struct RenderPass
{
	struct Instance
	{
		FBDescriptor fb;
		uint32_t base_x, base_y;
		uint32_t coarse_tiles_width;
		uint32_t coarse_tiles_height;
		uint32_t opaque_fbmask;

		// 0, 1 or 2.
		uint32_t sampling_rate_x_log2;
		uint32_t sampling_rate_y_log2;
		bool z_sensitive;
		bool z_write;
		bool channel_shuffle;
	};
	Instance instances[MaxRenderPassInstances];
	uint32_t num_instances;

	uint32_t coarse_tile_size_log2;

	uint32_t num_primitives;

	const StateVector *states;
	uint32_t num_states;

	const TextureInfo *textures;
	uint32_t num_textures;

	uint32_t label_key;
	uint32_t debug_capture_stride;

	bool has_aa1;
	bool has_scanmsk;
	bool allow_blend_demote;

	// For debugging. Aids capture tools.
	bool feedback_color;
	bool feedback_depth;

	enum Feedback
	{
		None,
		Color,
		Depth
	};

	Feedback feedback_mode;

	uint32_t feedback_texture_psm;
	uint32_t feedback_texture_cpsm;
	FlushReason flush_reason;
};

struct PrivRegisterState;
struct VSyncInfo;

enum class SuperSampling
{
	X1 = 1,
	X2 = 2,
	X4 = 4,
	X8 = 8,
	X16 = 16
};

struct GSOptions;
class PageTracker;

class GSRenderer
{
public:
	explicit GSRenderer(PageTracker &tracker);
	bool init(Vulkan::Device *device, const GSOptions &options);
	~GSRenderer();

	void reserve_primitive_buffers(uint32_t num_primitives);
	VertexPosition *get_reserved_vertex_positions() const;
	VertexAttribute *get_reserved_vertex_attributes() const;
	PrimitiveAttribute *get_reserved_primitive_attributes() const;

	// Copies host VRAM into GPU VRAM.
	// First logical stage.
	void flush_host_vram_copy(const uint32_t *block_indices, uint32_t num_indices);

	// Caching stage.
	uint32_t update_palette_cache(const PaletteUploadDescriptor &desc);
	void mark_clut_read(uint32_t clut_instance);
	void rewind_clut_instance(uint32_t index);

	Vulkan::ImageHandle create_cached_texture(const TextureDescriptor &desc);
	// Should always be called after create_cached_texture().
	// We'll be able to do some last minute modifications to the upload descriptor
	// depending on the page tracker state, e.g. to enable shadow copy strategy.
	void promote_cached_texture_upload_cpu(const PageRect &rect);
	void commit_cached_texture(uint32_t tex_info_index, bool sampler_feedback);

	Vulkan::ImageHandle copy_cached_texture(const Vulkan::Image &img, const VkRect2D &rect);

	// Creating 1k+ VkImages per frame can be a noticeable CPU burden on drivers.
	// Computing swizzling layouts and stuff is quite complicated and slow.
	// It's not just about memory allocation.
	void recycle_image_handle(Vulkan::ImageHandle image);
	// Uploads to CLUT cache and texture cache. Only reads VRAM.
	void flush_cache_upload();

	// Copy stage.
	void copy_vram(const CopyDescriptor &desc, const PageRect &damage_rect);
	void flush_transfer();
	void transfer_overlap_barrier();

	// To deal with local -> local hazards.
	void mark_shadow_page_sync(uint32_t page_index);

	// FB stage.
	void flush_rendering(const RenderPass &rp);

	// Readback stage.
	void flush_readback(const uint32_t *page_indices, uint32_t num_indices);

	// Submit to GPU and signal when done.
	void flush_submit(uint64_t timeline);

	// Wait until timeline reaches value. After waiting, it may be safe to read from host VRAM.
	void wait_timeline(uint64_t value);
	// Query current timeline value.
	uint64_t query_timeline();

	// Cache management mostly on GPUs that need it.
	void *begin_host_vram_access();
	void end_host_write_vram_access();

	ScanoutResult vsync(const PrivRegisterState &priv, const VSyncInfo &info,
	                    uint32_t sampling_rate_x_log2, uint32_t sampling_rate_y_log2,
	                    const Vulkan::Image *promoted1, const Vulkan::Image *promoted2);
	bool vsync_can_skip(const PrivRegisterState &priv, const VSyncInfo &info) const;

	static TexRect compute_effective_texture_rect(const TextureDescriptor &desc);

	FlushStats consume_flush_stats();

	double get_accumulated_timestamps(TimestampType type) const;
	void set_enable_timestamps(bool enable);

	void invalidate_super_sampling_state(uint32_t sampling_rate_x_log2, uint32_t sampling_rate_y_log2);

	SuperSampling get_max_supported_super_sampling() const;

	void set_field_aware_super_sampling(bool enable);

private:
	PageTracker &tracker;
	Vulkan::Device *device = nullptr;
	Vulkan::CommandBufferHandle direct_cmd;
	Vulkan::CommandBufferHandle async_transfer_cmd;
	Vulkan::CommandBufferHandle triangle_setup_cmd;
	Vulkan::CommandBufferHandle clear_cmd;
	Vulkan::CommandBufferHandle heuristic_cmd;
	Vulkan::CommandBufferHandle binning_cmd;
	uint32_t vram_size = 0;
	uint32_t next_clut_instance = 0;
	uint32_t base_clut_instance = 0;

	Vulkan::Semaphore timeline;
	std::thread timeline_thread;
	uint64_t last_submitted_timeline = 0;
	std::atomic<uint64_t> timeline_value;
	std::condition_variable timeline_cond;
	std::mutex timeline_lock;

	bool last_clut_update_is_read = false;
	bool field_aware_super_sampling = false;

	std::vector<VkImageMemoryBarrier2> pre_image_barriers;
	std::vector<VkImageMemoryBarrier2> post_image_barriers;
	std::vector<PaletteUploadDescriptor> palette_uploads;
	std::vector<VkDeviceAddress> qword_clears;

	struct Scratch
	{
		Vulkan::BufferHandle buffer;
		VkDeviceSize offset = 0;
		VkDeviceSize size = 0;
	};

	struct AttributeScratch : Scratch
	{
		Vulkan::BufferHandle gpu_buffer;
		VkDeviceSize flushed_to = 0;
	};

	struct TextureUpload
	{
		Vulkan::ImageHandle image;
		TextureDescriptor desc;
		Scratch scratch;
		struct
		{
			Vulkan::BufferHandle buffer;
			VkDeviceSize offset;
			VkDeviceSize size;
			Vulkan::BufferHandle indirect;
			VkDeviceSize indirect_offset;
		} indirection;
	};

	struct TextureAnalysis
	{
		VkDeviceAddress indirect_dispatch_va;
		VkDeviceAddress indirect_workgroups_va;
		VkDeviceAddress indirect_bitmask_va;
		uvec2 size_minus_1;
		u16vec2 base;
		uint16_t block_stride;
		uint16_t layers;
		uint32_t flags;
		enum { ENABLED_BIT = 1 << 0 };
	};
	static_assert(sizeof(TextureAnalysis) % 16 == 0, "Unaligned TextureAnalysis");
	static_assert(sizeof(TextureAnalysis) * MaxTextures <= Vulkan::VULKAN_MAX_UBO_SIZE, "Too large analysis struct.");

	std::vector<TextureUpload> texture_uploads;
	std::vector<TextureAnalysis> texture_analysis;

	struct PendingIndirectTextureUpload
	{
		Vulkan::BufferHandle indirect;
		VkDeviceSize indirect_offset;
		u16vec3 fallback_dispatch;
	};
	std::vector<PendingIndirectTextureUpload> pending_indirect_uploads;

	struct PendingIndirectAnalysis
	{
		Vulkan::BufferHandle indirect;
		VkDeviceSize indirect_offset;
	};
	std::vector<PendingIndirectAnalysis> pending_indirect_analysis;

	struct
	{
		Vulkan::BufferHandle clut;
		Vulkan::BufferHandle gpu;
		Vulkan::BufferHandle cpu;
		Vulkan::BufferHandle vram_copy_atomics;
		Vulkan::BufferHandle vram_copy_payloads;

		Scratch device_scratch, rebar_scratch;

		VkDeviceSize ssbo_alignment = 0;

		Vulkan::BufferHandle fixed_rcp_lut;
		Vulkan::BufferViewHandle fixed_rcp_lut_view;
		Vulkan::BufferHandle float_rcp_lut;
		Vulkan::BufferViewHandle float_rcp_lut_view;
		Vulkan::ImageHandle phase_lut;

		Vulkan::BufferHandle bug_feedback;

		// Attribute buffers, for zero-copy on UMA and single copy on dGPU.
		AttributeScratch pos_scratch, attr_scratch, prim_scratch;
		VertexPosition *pos = nullptr;
		VertexAttribute *attr = nullptr;
		PrimitiveAttribute *prim = nullptr;
	} buffers;

	Scratch indirect_single_sample_heuristic;
	Scratch work_list_single_sample;
	Scratch work_list_super_sample;

	VkDeviceSize allocate_device_scratch(VkDeviceSize size, Scratch &scratch, const void *data);
	void reserve_attribute_scratch(VkDeviceSize size, AttributeScratch &scratch);
	void commit_attribute_scratch(VkDeviceSize size, AttributeScratch &scratch);
	void flush_attribute_scratch(AttributeScratch &scratch);

	Vulkan::BindlessDescriptorPoolHandle bindless_allocator;
	struct ExhaustedDescriptorPool
	{
		Vulkan::BindlessDescriptorPoolHandle exhausted_pool;
		uint64_t timeline = 0;
	};
	std::queue<ExhaustedDescriptorPool> exhausted_descriptor_pools;
	Vulkan::BindlessDescriptorPoolHandle get_bindless_pool();
	Vulkan::Semaphore descriptor_timeline;
	uint64_t next_descriptor_timeline_signal = 1;

	void ensure_command_buffer(Vulkan::CommandBufferHandle &cmd, Vulkan::CommandBuffer::Type type);
	void init_luts();
	void init_phase_lut(uint32_t sampling_rate_x_log2, uint32_t sampling_rate_y_log2);
	void init_vram(const GSOptions &options);

	void upload_texture(const TextureUpload &upload);
	void bind_textures(Vulkan::CommandBuffer &cmd, const RenderPass &rp);

	bool bound_texture_has_array = false;

	void bind_frame_resources(const RenderPass &rp);
	void bind_frame_resources_instanced(const RenderPass &rp, uint32_t instance, uint32_t num_primitives);
	void allocate_scratch_buffers(Vulkan::CommandBuffer &cmd, const RenderPass &rp);
	void allocate_scratch_buffers_instanced(Vulkan::CommandBuffer &cmd, const RenderPass &rp,
	                                        uint32_t instance, uint32_t num_primitives);
	void dispatch_triangle_setup(Vulkan::CommandBuffer &cmd, const RenderPass &rp);
	void dispatch_binning(Vulkan::CommandBuffer &cmd, const RenderPass &rp, uint32_t instance,
	                      uint32_t base_primitive, uint32_t num_primitives);
	void dispatch_single_sample_heuristic(Vulkan::CommandBuffer &cmd, const RenderPass &rp);
	void dispatch_texture_analysis(Vulkan::CommandBuffer &cmd, const RenderPass &rp);
	void dispatch_shading(Vulkan::CommandBuffer &cmd, const RenderPass &rp, uint32_t instance,
	                      uint32_t base_primitive, uint32_t num_primitives);

	void dispatch_shading_commands(Vulkan::CommandBuffer &cmd,
	                               const RenderPass &rp,
	                               uint32_t instance,
	                               bool post_barrier,
	                               uint32_t base_primitive,
	                               uint32_t num_primitives);

	void dispatch_read_aliased_depth_passes(
			Vulkan::CommandBuffer &cmd, const RenderPass &rp, uint32_t instance, uint32_t depth_psm,
			ShadingDescriptor &push,
			uint32_t base_primitive, uint32_t first_z_sensitive, uint32_t num_primitives);

	void flush_palette_upload();

	bool render_pass_instance_is_deduced_blur(const RenderPass &rp, uint32_t instance) const;

	void bind_debug_resources(Vulkan::CommandBuffer &cmd, const RenderPass &rp, const RenderPass::Instance &instance);
	Vulkan::ImageHandle feedback_color, feedback_depth, feedback_prim, feedback_vary;

	void dispatch_cache_read_only_depth(Vulkan::CommandBuffer &cmd, const RenderPass &rp, uint32_t depth_psm, uint32_t instance);

	void flush_rendering(const RenderPass &rp, uint32_t instance, uint32_t base_primitive, uint32_t num_primitives);

	struct SamplingRect
	{
		VkExtent2D valid_extent;
		VkExtent2D image_extent;
		uint32_t phase_offset;
		uint32_t phase_stride;
	};

	void sample_crtc_circuit(Vulkan::CommandBuffer &cmd, const Vulkan::Image &img,
	                         const DISPFBBits &dispfb, const SamplingRect &rect, uint32_t super_samples,
	                         const Vulkan::Image *promoted);

	static SamplingRect compute_circuit_rect(const PrivRegisterState &priv, uint32_t phase,
	                                         const DISPLAYBits &display, bool force_progressive,
	                                         const Vulkan::Image *promoted);

	void copy_blocks(Vulkan::CommandBuffer &cmd, const Vulkan::Buffer &dst, const Vulkan::Buffer &src,
	                 const uint32_t *page_indices, uint32_t num_indices, bool invalidate_super_sampling,
	                 uint32_t block_size);

	struct CopyDescriptorPayload
	{
		CopyDescriptor copy;
		Vulkan::BufferBlockAllocation alloc;
	};
	std::vector<CopyDescriptorPayload> pending_copies;
	void emit_copy_vram(Vulkan::CommandBuffer &cmd,
	                    const uint32_t *dispatch_order,
	                    uint32_t num_dispatches, bool prepare_only);

	struct Timestamp
	{
		TimestampType type;
		Vulkan::QueryPoolHandle ts_start;
		Vulkan::QueryPoolHandle ts_end;
	};
	std::vector<Timestamp> timestamps;
	void log_timestamps();
	double timestamp_total_time[int(TimestampType::Count)] = {};

	FlushStats stats = {}, total_stats = {};
	void check_flush_stats();
	bool enable_timestamps = false;

	Vulkan::ImageHandle vsync_last_fields[4];
	Vulkan::ImageHandle fastmad_deinterlace(Vulkan::CommandBuffer &cmd, const VSyncInfo &vsync);

	// Slangmosh
	Shaders<> shaders;
	Vulkan::Program *blit_quad = nullptr;
	Vulkan::Program *sample_quad[2] = {};
	Vulkan::Program *weave_quad = nullptr;

	void drain_compilation_tasks();
	void drain_compilation_tasks_nonblock();
	void kick_compilation_tasks();
	std::atomic_bool compilation_tasks_active;
	std::vector<std::future<void>> compilation_tasks;

	uint64_t query_timeline(const Vulkan::SemaphoreHolder &sem) const;

	std::vector<Vulkan::ImageHandle> recycled_image_handles;
	// Only cache textures with reasonable POT size.
	// Small slab allocator basically.
	std::vector<Vulkan::ImageHandle> recycled_image_pool[7][11][11];
	std::vector<Vulkan::ImageHandle> super_sampled_recycled_image_pool[11][11];
	void move_image_handles_to_slab();
	Vulkan::ImageHandle pull_image_handle_from_slab(uint32_t width, uint32_t height, uint32_t levels, uint32_t samples);
	VkDeviceSize total_image_slab_size = 0;
	VkDeviceSize max_image_slab_size = 0;
	VkDeviceSize max_allocated_image_memory_per_flush = 0;
	VkDeviceSize image_slab_high_water_mark = 0;
	void flush_slab_cache();

	std::vector<uint32_t> vram_copy_write_pages;
	std::vector<uint32_t> sync_vram_shadow_pages;

	bool can_potentially_super_sample() const;

	void check_bug_feedback();

	bool scanout_is_interlaced(const PrivRegisterState &priv, const VSyncInfo &info) const;
	uint32_t get_target_hierarchical_binning(uint32_t num_primitives, uint32_t coarse_tiles_width, uint32_t coarse_tiles_height) const;
	void set_hierarchical_binning_subgroup_config(Vulkan::CommandBuffer &cmd, uint32_t hier_factor) const;

	void allocate_upload_indirection(TextureAnalysis &analysis, TextureUpload &upload);
	void ensure_conservative_indirect_texture_uploads();

	void flush_qword_clears();
	void ensure_clear_cmd();
};
}