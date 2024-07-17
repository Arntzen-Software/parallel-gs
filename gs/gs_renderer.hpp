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

namespace ParallelGS
{
struct ScanoutResult
{
	Vulkan::ImageHandle image;
};

struct FlushStats
{
	VkDeviceSize allocated_scratch_memory;
	VkDeviceSize allocated_image_memory;
	uint32_t num_primitives;
	uint32_t num_render_passes;
	uint32_t num_palette_updates;
	uint32_t num_copies;
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
		       palette_bank == other.palette_bank;
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
	FBDescriptor fb;

	uint32_t base_x, base_y;
	uint32_t coarse_tiles_width;
	uint32_t coarse_tiles_height;
	uint32_t coarse_tile_size_log2;

	const VertexPosition *positions;
	const VertexAttribute *attributes;
	const PrimitiveAttribute *prims;
	uint32_t num_primitives;

	const StateVector *states;
	uint32_t num_states;

	const TextureInfo *textures;
	uint32_t num_textures;

	uint32_t label_key;
	uint32_t debug_capture_stride;

	bool z_sensitive;
	bool has_aa1;
	bool has_scanmsk;

	// For debugging. Aids capture tools.
	bool feedback_color;
	bool feedback_depth;
	bool feedback_texture;

	uint32_t feedback_texture_psm;
	uint32_t feedback_texture_cpsm;
	FlushReason flush_reason;

	// 0, 1 or 2.
	uint32_t sampling_rate_x_log2;
	uint32_t sampling_rate_y_log2;
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

class GSRenderer
{
public:
	bool init(Vulkan::Device *device, const GSOptions &options);
	~GSRenderer();

	// Copies host VRAM into GPU VRAM.
	// First logical stage.
	void flush_host_vram_copy(const uint32_t *page_indices, uint32_t num_indices);

	// Caching stage.
	uint32_t update_palette_cache(const PaletteUploadDescriptor &desc);
	Vulkan::ImageHandle create_cached_texture(const TextureDescriptor &desc);
	// Creating 1k+ VkImages per frame can be a noticeable CPU burden on drivers.
	// Computing swizzling layouts and stuff is quite complicated and slow.
	// It's not just about memory allocation.
	void recycle_image_handle(Vulkan::ImageHandle image);
	// Uploads to CLUT cache and texture cache. Only reads VRAM.
	void flush_cache_upload();

	// Copy stage.
	void copy_vram(const CopyDescriptor &desc);
	void flush_transfer();
	void transfer_overlap_barrier();

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

	ScanoutResult vsync(const PrivRegisterState &priv, const VSyncInfo &info);

	static TexRect compute_effective_texture_rect(const TextureDescriptor &desc);

	FlushStats consume_flush_stats();

	double get_accumulated_timestamps(TimestampType type) const;
	void set_enable_timestamps(bool enable);

	void invalidate_super_sampling_state();

private:
	Vulkan::Device *device = nullptr;
	Vulkan::CommandBufferHandle direct_cmd;
	Vulkan::CommandBufferHandle async_transfer_cmd;
	Vulkan::CommandBufferHandle triangle_setup_cmd;
	Vulkan::CommandBufferHandle clear_cmd;
	Vulkan::CommandBufferHandle binning_cmd;
	uint32_t vram_size = 0;
	uint32_t next_clut_instance = 0;
	uint32_t base_clut_instance = 0;
	Vulkan::Semaphore timeline;

	std::vector<VkImageMemoryBarrier2> pre_image_barriers;
	std::vector<VkImageMemoryBarrier2> post_image_barriers;
	std::vector<PaletteUploadDescriptor> palette_uploads;

	struct TextureUpload
	{
		Vulkan::ImageHandle image;
		TextureDescriptor desc;
	};
	std::vector<TextureUpload> texture_uploads;

	struct Scratch
	{
		Vulkan::BufferHandle buffer;
		VkDeviceSize offset = 0;
		VkDeviceSize size = 0;
	};

	struct
	{
		Vulkan::BufferHandle clut;
		Vulkan::BufferHandle gpu;
		Vulkan::BufferHandle cpu;

		Scratch device_scratch, rebar_scratch;
		VkDeviceSize ssbo_alignment = 0;

		Vulkan::BufferHandle fixed_rcp_lut;
		Vulkan::BufferViewHandle fixed_rcp_lut_view;
		Vulkan::BufferHandle float_rcp_lut;
		Vulkan::BufferViewHandle float_rcp_lut_view;
	} buffers;

	Scratch indirect_single_sample_heuristic;

	VkDeviceSize allocate_device_scratch(VkDeviceSize size, Scratch &scratch, const void *data);

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
	void init_vram(const GSOptions &options);

	void upload_texture(const TextureDescriptor &desc, const Vulkan::Image &img);
	void bind_textures(Vulkan::CommandBuffer &cmd, const RenderPass &rp);
	void bind_frame_resources(const RenderPass &rp);
	void allocate_scratch_buffers(Vulkan::CommandBuffer &cmd, const RenderPass &rp);
	void dispatch_triangle_setup(Vulkan::CommandBuffer &cmd, const RenderPass &rp);
	void dispatch_binning(Vulkan::CommandBuffer &cmd, const RenderPass &rp);
	void dispatch_single_sample_heuristic(Vulkan::CommandBuffer &cmd, const RenderPass &rp);
	void dispatch_shading(Vulkan::CommandBuffer &cmd, const RenderPass &rp);
	void dispatch_shading_debug(
			Vulkan::CommandBuffer &cmd, const RenderPass &rp,
			uint32_t width, uint32_t height);
	void flush_palette_upload();

	struct SamplingRect
	{
		VkExtent2D valid_extent;
		VkExtent2D image_extent;
		uint32_t phase_offset;
		uint32_t phase_stride;
	};

	void sample_crtc_circuit(Vulkan::CommandBuffer &cmd, const Vulkan::Image &img,
	                         const DISPFBBits &dispfb, const SamplingRect &rect);

	static SamplingRect compute_circuit_rect(const PrivRegisterState &priv, uint32_t phase,
	                                         const DISPLAYBits &display, bool force_progressive);

	void copy_pages(Vulkan::CommandBuffer &cmd, const Vulkan::Buffer &dst, const Vulkan::Buffer &src,
	                const uint32_t *page_indices, uint32_t num_indices);

	struct CopyDescriptorPayload
	{
		CopyDescriptor copy;
		Vulkan::BufferBlockAllocation alloc;
	};
	std::vector<CopyDescriptorPayload> pending_copies;
	void emit_copy_vram(Vulkan::CommandBuffer &cmd, const CopyDescriptorPayload &desc);

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
	Vulkan::Program *sample_quad = nullptr;
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
	std::vector<Vulkan::ImageHandle> recycled_image_pool[11][11];
	void move_image_handles_to_slab();
	Vulkan::ImageHandle pull_image_handle_from_slab(uint32_t width, uint32_t height, uint32_t levels);
	VkDeviceSize total_image_slab_size = 0;
	VkDeviceSize max_image_slab_size = 0;
	VkDeviceSize image_slab_high_water_mark = 0;
};
}