// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#include "gs_renderer.hpp"
#include "logging.hpp"
#include "gs_interface.hpp"
#include "gs_registers_debug.hpp"
#include "bitops.hpp"
#include "shaders/slangmosh.hpp"
#include "shaders/swizzle_utils.h"
#include "thread_id.hpp"
#include "gs_util.hpp"
#include <utility>
#include <algorithm>
#include <cmath>

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wformat-security"
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wformat-security"
#endif

namespace ParallelGS
{
// If we've seen a normal render pass, just flush right away.
// Normal 3D games seem to be between 4k and 40k primitives per frame.
static constexpr uint32_t MinimumPrimitivesForFlush = 4096;
// If there is a lot of back to back (small render passes). This should only trigger first for insane feedback loops.
static constexpr uint32_t MinimumRenderPassForFlush = 1024;
// If someone spams the scratch allocators too hard, we have to yield eventually.
static constexpr VkDeviceSize MaximumAllocatedImageMemory = 100 * 1000 * 1000;
static constexpr VkDeviceSize MaximumAllocatedScratchMemory = 100 * 1000 * 1000;
static constexpr uint32_t MaxPendingPaletteUploads = 4096;
static constexpr uint32_t MaxPendingCopies = 4096;
static constexpr uint32_t MaxPendingCopyThreads = 4 * 1024 * 1024; // 22 bits.
static constexpr uint32_t MaxPendingCopiesWithoutFlush = 1023; // 10 bits. Reserve highest value for unlinked node.

// Pink-ish. Intended to look good in RenderDoc's default light theme.
static constexpr float LabelColor[] = { 1.0f, 0.8f, 0.8f, 1.0f };

template <typename... Args>
static void insert_label(Vulkan::CommandBuffer &cmd, const char *fmt, Args... args)
{
	char label[256];
	snprintf(label, sizeof(label), fmt, args...);
	cmd.insert_label(label, LabelColor);
}

template <typename... Args>
static void begin_region(Vulkan::CommandBuffer &cmd, const char *fmt, Args... args)
{
	char label[256];
	snprintf(label, sizeof(label), fmt, args...);
	cmd.begin_region(label);
}

struct RangeMerger
{
	VkDeviceSize merged_offset = 0;
	VkDeviceSize merged_range = 0;

	template <typename Op>
	void push(VkDeviceSize new_offset, VkDeviceSize new_range, Op &&op)
	{
		if (!merged_range)
		{
			merged_offset = new_offset;
			merged_range = new_range;
		}
		else if (merged_offset + merged_range == new_offset)
		{
			merged_range += new_range;
		}
		else
		{
			op(merged_offset, merged_range);
			merged_offset = new_offset;
			merged_range = new_range;
		}
	}

	template <typename Op>
	void flush(Op &&op)
	{
		if (merged_range)
			op(merged_offset, merged_range);
	}
};

void GSRenderer::invalidate_super_sampling_state()
{
	if (!device || !buffers.gpu)
		return;

	VkDeviceSize clear_size = buffers.gpu->get_create_info().size - vram_size;
	if (!clear_size)
		return;

	flush_submit(0);

	// Clear reference values and all super sample memory to 0.
	auto cmd = device->request_command_buffer();
	cmd->begin_region("invalidate-super-sampling");
	cmd->barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
	             VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
	cmd->fill_buffer(*buffers.gpu, 0, vram_size, clear_size);
	cmd->barrier(VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
	cmd->end_region();
	device->submit(cmd);
}

SuperSampling GSRenderer::get_max_supported_super_sampling() const
{
	SuperSampling max_ssaa = SuperSampling::X4;
	if (device->supports_subgroup_size_log2(true, 3, 6))
		max_ssaa = SuperSampling::X8;
	if (device->supports_subgroup_size_log2(true, 4, 6))
		max_ssaa = SuperSampling::X16;

	return max_ssaa;
}

void GSRenderer::init_vram(const GSOptions &options)
{
	Vulkan::BufferCreateInfo info = {};
	info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

	// One copy of VRAM for single-rate, one reference copy of VRAM, and up to 16 sample references.
	// About 78 MB. This isn't too bad.
	if (options.dynamic_super_sampling)
		info.size = vram_size * (1 + 1 + int(get_max_supported_super_sampling()));
	else if (options.super_sampling != SuperSampling::X1)
		info.size = vram_size * (1 + 1 + int(options.super_sampling));
	else
		info.size = vram_size;

	// Need a shadow copy of VRAM for various difficult feedback hazards.
	// Simpler to reuse the same buffer.
	info.size *= 2;

	info.domain = Vulkan::BufferDomain::Device;
	// For convenience.
	info.misc = Vulkan::BUFFER_MISC_ZERO_INITIALIZE_BIT;
	buffers.gpu = device->create_buffer(info);
	device->set_name(*buffers.gpu, "vram-gpu");

	// Could elide this on iGPU, perhaps?
	info.domain = Vulkan::BufferDomain::CachedHost;
	buffers.cpu = device->create_buffer(info);
	info.size = vram_size;
	device->set_name(*buffers.cpu, "vram-cpu");

	info.domain = Vulkan::BufferDomain::Device;
	info.size = CLUTInstances * CLUTSize;
	buffers.clut = device->create_buffer(info);
	device->set_name(*buffers.clut, "clut");

	info.misc = 0;
	// - One atomic counter
	// - VRAM (one u32 atomic variable per u32 dword of VRAM memory)
	// - One bit per u32 dword of VRAM memory.
	info.size = PGS_LINKED_VRAM_COPY_WRITE_LIST_OFFSET + vram_size + vram_size / 32;
	buffers.vram_copy_atomics = device->create_buffer(info);
	device->set_name(*buffers.vram_copy_atomics, "vram-copy-atomics");
	info.size = MaxPendingCopyThreads * sizeof(LinkedVRAMCopyWrite);
	buffers.vram_copy_payloads = device->create_buffer(info);
	device->set_name(*buffers.vram_copy_payloads, "vram-copy-payloads");

	sync_vram_shadow_pages.resize(vram_size / PageSize);
	vram_copy_write_pages.resize(vram_size / PageSize);
}

void GSRenderer::drain_compilation_tasks()
{
	compilation_tasks_active = false;
	for (auto &task : compilation_tasks)
		if (task.valid())
			task.get();
	compilation_tasks.clear();
}

void GSRenderer::drain_compilation_tasks_nonblock()
{
	auto itr = std::remove_if(compilation_tasks.begin(), compilation_tasks.end(), [](std::future<void> &fut) {
		if (fut.valid() && fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
		{
			fut.get();
			return true;
		}
		else
			return false;
	});
	compilation_tasks.erase(itr, compilation_tasks.end());
}

bool GSRenderer::can_potentially_super_sample() const
{
	return buffers.gpu->get_create_info().size > vram_size * 2;
}

void GSRenderer::kick_compilation_tasks()
{
	// Pre-prime all potential shader variants early.
	std::vector<Vulkan::DeferredPipelineCompile> tasks;
	compilation_tasks_active = true;

	{
		auto cmd = device->request_command_buffer();
		cmd->set_program(shaders.ubershader[0][0]);
		Vulkan::DeferredPipelineCompile deferred = {};

		static const struct { uint32_t sample_x, sample_y; } sampling_rates[] = {
			{ 0, 0 },
			{ 0, 1 },
			{ 1, 1 },
			{ 1, 2 },
			{ 2, 2 },
		};

		static const uint32_t variant_flags[] = {
			0,
			VARIANT_FLAG_HAS_AA1_BIT,
			VARIANT_FLAG_HAS_SCANMSK_BIT,
			VARIANT_FLAG_HAS_AA1_BIT | VARIANT_FLAG_HAS_SCANMSK_BIT,
			VARIANT_FLAG_FEEDBACK_BIT,
			VARIANT_FLAG_FEEDBACK_BIT | VARIANT_FLAG_HAS_AA1_BIT,
			VARIANT_FLAG_FEEDBACK_BIT | VARIANT_FLAG_HAS_SCANMSK_BIT,
			VARIANT_FLAG_FEEDBACK_BIT | VARIANT_FLAG_HAS_AA1_BIT | VARIANT_FLAG_HAS_SCANMSK_BIT,
		};

		// Prime all common combinations.
		static const struct { uint32_t color_psm, depth_psm; } formats[] = {
			{ PSMCT24, PSMZ24 },
			{ PSMCT32, PSMZ24 },
			{ PSMCT16S, PSMZ24 },
			{ PSMCT24, PSMZ16S },
			{ PSMCT32, PSMZ16S },
			{ PSMCT16S, PSMZ16S },
			{ PSMCT24, UINT32_MAX },
			{ PSMCT32, UINT32_MAX },
			{ PSMCT16S, UINT32_MAX },
			{ PSMZ24, UINT32_MAX },
			{ PSMZ32, UINT32_MAX },
			{ PSMZ16S, UINT32_MAX },
			{ PSMZ16, UINT32_MAX },
		};

		static const struct { uint32_t feedback_psm, feedback_cpsm; } feedbacks[] = {
			{ PSMCT32, 0 },
			{ PSMCT24, 0 },
			{ PSMCT16S, 0 },
			{ PSMT4HL, PSMCT32 },
			{ PSMT4HL, PSMCT16 },
			{ PSMT4HH, PSMCT32 },
			{ PSMT4HH, PSMCT16 },
			{ PSMT8H, PSMCT32 },
			{ PSMT8H, PSMCT16 },
		};

		cmd->set_specialization_constant_mask(0xff);
		cmd->enable_subgroup_size_control(true);
		// Prefer Wave64 if we can get away with it.
		if (device->supports_subgroup_size_log2(true, 6, 6))
			cmd->set_subgroup_size_log2(true, 6, 6);
		else if (device->supports_subgroup_size_log2(true, 4, 6))
			cmd->set_subgroup_size_log2(true, 4, 6);
		else if (device->supports_subgroup_size_log2(true, 3, 6))
			cmd->set_subgroup_size_log2(true, 3, 6);
		else
			cmd->set_subgroup_size_log2(true, 2, 6);

		for (auto &format : formats)
		{
			for (auto &flags : variant_flags)
			{
				for (auto &rates : sampling_rates)
				{
					for (auto &feedback : feedbacks)
					{
						if ((flags & VARIANT_FLAG_FEEDBACK_BIT) != 0)
						{
							if (swizzle_compat_key(feedback.feedback_psm) != swizzle_compat_key(format.color_psm))
								continue;
						}
						else if (feedback.feedback_psm != 0 || feedback.feedback_cpsm != 0)
							continue;

						cmd->set_specialization_constant(0, rates.sample_x);
						cmd->set_specialization_constant(1, rates.sample_y);
						cmd->set_specialization_constant(2, format.color_psm);
						cmd->set_specialization_constant(3, format.depth_psm);
						cmd->set_specialization_constant(4, vram_size - 1);
						cmd->set_specialization_constant(
								5, flags | (rates.sample_y ? VARIANT_FLAG_HAS_SUPER_SAMPLE_REFERENCE_BIT : 0));
						cmd->set_specialization_constant(6, feedback.feedback_psm);
						cmd->set_specialization_constant(7, feedback.feedback_cpsm);
						cmd->extract_pipeline_state(deferred);
						tasks.push_back(deferred);

						if (rates.sample_x == 0 && rates.sample_y == 0)
						{
							if (can_potentially_super_sample())
							{
								cmd->set_specialization_constant(
										5, flags | VARIANT_FLAG_HAS_SUPER_SAMPLE_REFERENCE_BIT);
								cmd->extract_pipeline_state(deferred);
								tasks.push_back(deferred);
							}
						}
					}
				}
			}
		}

		device->submit_discard(cmd);
	}

	{
		Vulkan::DeferredPipelineCompile deferred = {};
		auto cmd = device->request_command_buffer();
		cmd->set_program(shaders.vram_copy);

		static const struct { uint32_t wg_size, psm; } formats[] = {
			{ 64, PSMCT32 },
			{ 64, PSMCT24 },
			{ 64, PSMCT16 },
			{ 64, PSMCT16S },
			{ 64, PSMZ32 },
			{ 64, PSMZ24 },
			{ 64, PSMZ16 },
			{ 64, PSMZ16S },
			{ 64, PSMT8 },
			{ 64, PSMT8H },
			{ 64, PSMT4HH },
			{ 64, PSMT4HL },
			{ 32, PSMT4 },
			{ 64, PSMT4 },
		};

		cmd->set_specialization_constant_mask(0x7f);
		cmd->set_specialization_constant(3, vram_size - 1);
		cmd->set_specialization_constant(4, HOST_TO_LOCAL);
		cmd->set_specialization_constant(5, uint32_t(can_potentially_super_sample()));

		for (auto &format : formats)
		{
			cmd->set_specialization_constant(0, format.wg_size);
			cmd->set_specialization_constant(1, format.psm);
			cmd->set_specialization_constant(2, format.psm);

			for (unsigned prepare_only = 0; prepare_only < 2; prepare_only++)
			{
				cmd->set_specialization_constant(6, prepare_only);
				cmd->extract_pipeline_state(deferred);
				tasks.push_back(deferred);
			}
		}

		device->submit_discard(cmd);
	}

	{
		Vulkan::DeferredPipelineCompile deferred = {};
		auto cmd = device->request_command_buffer();
		cmd->set_program(shaders.upload);
		cmd->set_specialization_constant_mask(0x7);
		cmd->set_specialization_constant(1, vram_size - 1);

		static const struct { uint32_t psm; uint32_t cpsm; } formats[] = {
			{ PSMCT32, 0 },
			{ PSMCT24, 0 },
			{ PSMCT16, 0 },
			{ PSMCT16S, 0 },
			{ PSMZ32, 0 },
			{ PSMZ24, 0 },
			{ PSMZ16, 0 },
			{ PSMZ16S, 0 },
			{ PSMT8, PSMCT32 },
			{ PSMT8, PSMCT16 },
			{ PSMT4, PSMCT32 },
			{ PSMT4, PSMCT16 },
			{ PSMT4HH, PSMCT32 },
			{ PSMT4HH, PSMCT16 },
			{ PSMT4HL, PSMCT32 },
			{ PSMT4HL, PSMCT16 },
		};

		for (auto &format : formats)
		{
			cmd->set_specialization_constant(0, format.psm);
			cmd->set_specialization_constant(2, format.cpsm);
			cmd->extract_pipeline_state(deferred);
			tasks.push_back(deferred);
		}

		device->submit_discard(cmd);
	}

	size_t num_tasks = tasks.size();
	size_t target_threads = (std::thread::hardware_concurrency() + 1) / 2;
	size_t tasks_per_thread = num_tasks / target_threads;

	for (size_t thread_index = 0; thread_index < target_threads; thread_index++)
	{
		std::vector<Vulkan::DeferredPipelineCompile> deferred(
			tasks.data() + thread_index * tasks_per_thread,
			tasks.data() + std::min<size_t>((thread_index + 1) * tasks_per_thread, tasks.size())
		);

		auto async_task = std::async(std::launch::async, [this, moved_tasks = std::move(deferred)]()
		{
			// Just shuts up warnings.
			Util::register_thread_index(0);
			for (auto &task: moved_tasks)
			{
				// If we destroy the device before threads are done spinning.
				if (!compilation_tasks_active)
					break;
				Vulkan::CommandBuffer::build_compute_pipeline(
						device, task, Vulkan::CommandBuffer::CompileMode::AsyncThread);
			}
		});

		compilation_tasks.push_back(std::move(async_task));
	}
}

GSRenderer::GSRenderer(PageTracker &tracker_)
	: tracker(tracker_), compilation_tasks_active(false)
{
}

bool GSRenderer::init(Vulkan::Device *device_, const GSOptions &options)
{
	drain_compilation_tasks();

	Vulkan::ResourceLayout layout;
	shaders = Shaders<>(*device_, layout, 0);
	blit_quad = device_->request_program(shaders.quad, shaders.blit_circuit);
	sample_quad = device_->request_program(shaders.quad, shaders.sample_circuit);
	weave_quad = device_->request_program(shaders.quad, shaders.weave);

	flush_submit(0);
	// Descriptor indexing is a hard requirement, but timeline semaphore could be elided if really needed.
	// Wave-ops is critical.
	constexpr VkSubgroupFeatureFlags required_subgroup_flags =
			VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
			VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
			VK_SUBGROUP_FEATURE_VOTE_BIT |
			VK_SUBGROUP_FEATURE_BALLOT_BIT |
			VK_SUBGROUP_FEATURE_BASIC_BIT;

	const auto &ext = device_->get_device_features();
	if (!ext.vk12_features.descriptorIndexing ||
	    !ext.vk12_features.timelineSemaphore ||
		!ext.vk12_features.bufferDeviceAddress ||
	    !ext.vk12_features.storageBuffer8BitAccess ||
	    !ext.vk11_features.storageBuffer16BitAccess ||
	    !ext.enabled_features.shaderInt16 ||
	    (ext.vk11_props.subgroupSupportedOperations & required_subgroup_flags) != required_subgroup_flags ||
	    !device_->supports_subgroup_size_log2(true, 2, 6))
	{
		LOGE("Minimum requirements for parallel-gs are not met.\n");
		LOGE("  - descriptorIndexing\n");
		LOGE("  - timelineSemaphore\n");
		LOGE("  - bufferDeviceAddress\n");
		LOGE("  - storageBuffer8BitAccess\n");
		LOGE("  - storageBuffer16BitAccess\n");
		LOGE("  - shaderInt16\n");
		LOGE("  - Arithmetic / Shuffle / Vote / Ballot / Basic subgroup operations\n");
		LOGE("  - SubgroupSize control for [4, 64] invocations per subgroup\n");
		return false;
	}

	buffers = {};
	device = device_;
	vram_size = options.vram_size;
	next_clut_instance = 0;
	base_clut_instance = 0;

	buffers.ssbo_alignment =
			std::max<VkDeviceSize>(16, device->get_gpu_properties().limits.minStorageBufferOffsetAlignment);

	init_vram(options);
	timeline = device->request_semaphore(VK_SEMAPHORE_TYPE_TIMELINE);
	descriptor_timeline = device->request_semaphore(VK_SEMAPHORE_TYPE_TIMELINE);
	init_luts();

	kick_compilation_tasks();

	// Reserve 25% of our budget to slab-allocate image handles.
	Vulkan::HeapBudget budgets[VK_MAX_MEMORY_HEAPS] = {};
	device->get_memory_budget(budgets);
	for (uint32_t i = 0; i < device->get_memory_properties().memoryHeapCount; i++)
		if ((device->get_memory_properties().memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0)
			max_image_slab_size = std::max<VkDeviceSize>(max_image_slab_size, budgets[i].budget_size / 4);

	LOGI("Using image slab size of %llu MiB.\n", static_cast<unsigned long long>(max_image_slab_size / (1024 * 1024)));

	return true;
}

void GSRenderer::check_flush_stats()
{
	// Make sure that we flush as soon as there is a reasonable amount of work in flight.
	// We also want to keep memory usage under control. We have to garbage collect memory.

	if (stats.num_primitives >= MinimumPrimitivesForFlush ||
	    stats.num_render_passes >= MinimumRenderPassForFlush ||
	    stats.allocated_image_memory >= MaximumAllocatedImageMemory ||
	    stats.allocated_scratch_memory >= MaximumAllocatedScratchMemory ||
	    stats.num_palette_updates >= MaxPendingPaletteUploads ||
	    stats.num_copies >= MaxPendingCopies ||
		pending_copies.size() >= MaxPendingCopiesWithoutFlush ||
		stats.num_copy_threads >= MaxPendingCopyThreads)
	{
#ifdef PARALLEL_GS_DEBUG
		LOGI("Too much pending work, flushing:\n");
		LOGI("  %u primitives\n", stats.num_primitives);
		LOGI("  %u render passes\n", stats.num_render_passes);
		LOGI("  %u MiB allocated image memory\n", unsigned(stats.allocated_image_memory / (1024 * 1024)));
		LOGI("  %u MiB allocated scratch memory\n", unsigned(stats.allocated_scratch_memory / (1024 * 1024)));
		LOGI("  %u palette updates\n", stats.num_palette_updates);
		LOGI("  %u copies\n", stats.num_copies);
		LOGI("  %u copy threads\n", stats.num_copy_threads);
		LOGI("  %u copy barriers\n", stats.num_copy_barriers);
#endif
		// Flush the work that is considered pending right now.
		// Render passes always commit their work to a command buffer right away.
		flush_transfer();
		flush_cache_upload();
		// Calls next_frame_context and does garbage collection.
		flush_submit(0);

		// Notify that we have flushed without being triggered to do so by page tracker.
		tracker.notify_pressure_flush();
	}
}

VkDeviceSize GSRenderer::allocate_device_scratch(VkDeviceSize size, Scratch &scratch, const void *data)
{
	// Trivial linear allocator. Reduces pressure on Granite allocator.
	// It's important that we don't allocate too huge buffers here, then we don't get suballocation in Granite
	// (which would be quite bad).
	auto align = buffers.ssbo_alignment;
	stats.allocated_scratch_memory += size;

	scratch.offset = (scratch.offset + align - 1) & ~(align - 1);
	if (!scratch.buffer || scratch.offset + size > scratch.size)
	{
		Vulkan::BufferCreateInfo info = {};
		constexpr VkDeviceSize DefaultScratchBufferSize = 32 * 1024 * 1024;
		info.size = std::max<VkDeviceSize>(size, DefaultScratchBufferSize);
		info.domain = Vulkan::BufferDomain::Device;
		info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
		             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
		             VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

		scratch.buffer = device->create_buffer(info);
		scratch.offset = 0;
		scratch.size = info.size;
	}

	auto offset = scratch.offset;

	if (data)
	{
		// Fallback when sufficient ReBAR is not available.
		// This also happens when capturing with RenderDoc since persistently mapped ReBAR is horrible for capture perf.
		bool first_command = !async_transfer_cmd;
		ensure_command_buffer(async_transfer_cmd, Vulkan::CommandBuffer::Type::AsyncTransfer);
		if (first_command)
			async_transfer_cmd->begin_region("AsyncTransfer");
		memcpy(async_transfer_cmd->update_buffer(*scratch.buffer, offset, size), data, size);
	}

	scratch.offset += size;
	return offset;
}

GSRenderer::~GSRenderer()
{
	// Need to get rid of any command buffer handles at the very least. Otherwise, we deadlock the device.
	flush_submit(0);
	drain_compilation_tasks();
}

void GSRenderer::wait_timeline(uint64_t value)
{
	if (!device)
		return;
	timeline->wait_timeline(value);
}

uint64_t GSRenderer::query_timeline(const Vulkan::SemaphoreHolder &sem) const
{
	uint64_t value = 0;
	if (device->get_device_table().vkGetSemaphoreCounterValue(
			device->get_device(), sem.get_semaphore(), &value) != VK_SUCCESS)
	{
		return 0;
	}
	else
		return value;
}

uint64_t GSRenderer::query_timeline()
{
	return query_timeline(*timeline);
}

void GSRenderer::flush_submit(uint64_t value)
{
	if (!device)
		return;

	total_stats.allocated_scratch_memory += stats.allocated_scratch_memory;
	total_stats.allocated_image_memory += stats.allocated_image_memory;
	total_stats.num_copies += stats.num_copies;
	total_stats.num_primitives += stats.num_primitives;
	total_stats.num_palette_updates += stats.num_palette_updates;
	total_stats.num_render_passes += stats.num_render_passes;
	total_stats.num_copy_barriers += stats.num_copy_barriers;
	total_stats.num_copy_threads += stats.num_copy_threads;
	stats = {};

	if (direct_cmd)
	{
		// Copies may hold references to scratch buffers which must not outlive pending_cmd.
		flush_transfer();
	}

	// This must come before async transfer cmd since we risk allocating a transfer.
	if (clear_cmd && !qword_clears.empty())
	{
		uint32_t count = qword_clears.size();
		auto offset = allocate_device_scratch(count * sizeof(qword_clears.front()), buffers.rebar_scratch, qword_clears.data());
		clear_cmd->set_program(shaders.qword_clear);
		clear_cmd->set_storage_buffer(0, 0, *buffers.rebar_scratch.buffer, offset, count * sizeof(qword_clears.front()));
		clear_cmd->push_constants(&count, 0, sizeof(count));

		uint32_t wgx = (count + 63) / 64;

		if (wgx <= device->get_gpu_properties().limits.maxComputeWorkGroupCount[0])
		{
			clear_cmd->dispatch(wgx, 1, 1);
		}
		else
		{
			// Shouldn't really happen, but if it does, just eat the extra dummy threads.
			clear_cmd->dispatch(0xffff, (wgx + 0xfffe) / 0xffff, 1);
		}

		qword_clears.clear();
	}

	if (async_transfer_cmd)
	{
		Vulkan::Semaphore sem;
		async_transfer_cmd->end_region();
		device->submit(async_transfer_cmd, nullptr, 1, &sem);
		device->add_wait_semaphore(Vulkan::CommandBuffer::Type::Generic, std::move(sem),
		                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, true);
	}

	if (clear_cmd)
	{
		clear_cmd->barrier(VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		                   VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                   VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

		clear_cmd->end_region();
		device->submit(clear_cmd);
	}

	if (triangle_setup_cmd)
	{
		triangle_setup_cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
		                            VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
		                            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
		                            VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
		device->submit(triangle_setup_cmd);
	}

	if (heuristic_cmd)
	{
		heuristic_cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
		device->submit(heuristic_cmd);
	}

	if (binning_cmd)
	{
		binning_cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
		                     VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
		device->submit(binning_cmd);
	}

	if (direct_cmd)
		device->submit(direct_cmd);

	if (value)
	{
		auto binary = device->request_timeline_semaphore_as_binary(*timeline, value);
		device->submit_empty(Vulkan::CommandBuffer::Type::Generic, nullptr, binary.get());
		binary = device->request_timeline_semaphore_as_binary(*descriptor_timeline, next_descriptor_timeline_signal++);
		device->submit_empty(Vulkan::CommandBuffer::Type::Generic, nullptr, binary.get());
	}

	// This is a delayed sync-point between CPU and GPU, and garbage collection can happen here.
	drain_compilation_tasks_nonblock();
	device->next_frame_context();
	log_timestamps();
}

void GSRenderer::log_timestamps()
{
	auto itr = timestamps.begin();
	for (; itr != timestamps.end() && itr->ts_start->is_signalled() && itr->ts_end->is_signalled(); ++itr)
	{
		double t = device->convert_device_timestamp_delta(
				itr->ts_start->get_timestamp_ticks(), itr->ts_end->get_timestamp_ticks());
		timestamp_total_time[int(itr->type)] += t;
	}
	timestamps.erase(timestamps.begin(), itr);
}

FlushStats GSRenderer::consume_flush_stats()
{
	FlushStats s = total_stats;
	total_stats = {};
	return s;
}

void GSRenderer::set_enable_timestamps(bool enable)
{
	enable_timestamps = enable;
}

double GSRenderer::get_accumulated_timestamps(TimestampType type) const
{
	assert(int(type) < int(TimestampType::Count));
	return timestamp_total_time[int(type)];
}

TexRect GSRenderer::compute_effective_texture_rect(const TextureDescriptor &desc)
{
	uint32_t width_log2 = std::min<uint32_t>(desc.tex0.desc.TW, TEX0Bits::MAX_SIZE_LOG2);
	uint32_t height_log2 = std::min<uint32_t>(desc.tex0.desc.TH, TEX0Bits::MAX_SIZE_LOG2);
	uint32_t width = 1u << width_log2;
	uint32_t height = 1u << height_log2;

	uint32_t max_levels = std::min<uint32_t>(TEX0Bits::MAX_LEVELS, std::min<uint32_t>(width_log2, height_log2) + 1);
	uint32_t levels = std::min<uint32_t>(uint32_t(desc.tex1.desc.MXL) + 1, max_levels);

	TexRect rect = { 0, 0, width, height, levels };

	auto effective_u = compute_effective_texture_extent(
			width,
			uint32_t(desc.clamp.desc.WMS),
			uint32_t(desc.clamp.desc.MINU),
			uint32_t(desc.clamp.desc.MAXU), levels);

	auto effective_v = compute_effective_texture_extent(
			height,
			uint32_t(desc.clamp.desc.WMT),
			uint32_t(desc.clamp.desc.MINV),
			uint32_t(desc.clamp.desc.MAXV), levels);

	rect.x = effective_u.base;
	rect.y = effective_v.base;
	rect.width = effective_u.extent;
	rect.height = effective_v.extent;

	return rect;
}

void GSRenderer::recycle_image_handle(Vulkan::ImageHandle image)
{
	// Have to defer this until render pass is flushed, since an invalidate doesn't mean the texture is
	// immune from reuse.
	if (Util::is_pow2(image->get_width()) && Util::is_pow2(image->get_height()) &&
	    image->get_width() <= 1024 && image->get_height() <= 1024 && image->get_create_info().levels == 1)
	{
		recycled_image_handles.push_back(std::move(image));
	}
}

Vulkan::ImageHandle GSRenderer::create_cached_texture(const TextureDescriptor &desc)
{
	if (!device)
		return {};

	assert(desc.rect.width && desc.rect.height);

	Vulkan::ImageHandle img = pull_image_handle_from_slab(desc.rect.width, desc.rect.height, desc.rect.levels);

	if (!img)
	{
		Vulkan::ImageCreateInfo info = Vulkan::ImageCreateInfo::immutable_2d_image(
			desc.rect.width, desc.rect.height, VK_FORMAT_R8G8B8A8_UNORM);

		info.levels = desc.rect.levels;
		info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
		info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

		// Ignore mips. This is just a crude heuristic.
		stats.allocated_image_memory += info.width * info.height * sizeof(uint32_t);

		img = device->create_image(info);
	}

	if (device->consumes_debug_markers())
	{
		// If we're running in capture tools.
		char name_str[128];

		snprintf(name_str, sizeof(name_str), "%s - [%u x %u] + (%u, %u) - 0x%x - bank %u @ %u - %s",
		         psm_to_str(desc.tex0.desc.PSM),
		         desc.rect.width, desc.rect.height,
		         desc.rect.x, desc.rect.y,
		         uint32_t(desc.tex0.desc.TBP0) * PGS_BLOCK_ALIGNMENT_BYTES,
		         desc.palette_bank,
		         uint32_t(desc.tex0.desc.CSA),
		         psm_to_str(uint32_t(desc.tex0.desc.CSM)));
		device->set_name(*img, name_str);
	}

	VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
	barrier.image = img->get_image();
	barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
	barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
	barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS };

	pre_image_barriers.push_back(barrier);

	barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
	barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
	barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
	barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

	post_image_barriers.push_back(barrier);
	texture_uploads.push_back({ img, desc });

	check_flush_stats();

	return img;
}

void *GSRenderer::begin_host_vram_access()
{
	if (!device)
		return nullptr;
	return device->map_host_buffer(*buffers.cpu, Vulkan::MEMORY_ACCESS_READ_WRITE_BIT);
}

void GSRenderer::end_host_write_vram_access()
{
	if (!device)
		return;
	device->unmap_host_buffer(*buffers.cpu, Vulkan::MEMORY_ACCESS_WRITE_BIT);
}

void GSRenderer::copy_pages(Vulkan::CommandBuffer &cmd, const Vulkan::Buffer &dst, const Vulkan::Buffer &src,
                            const uint32_t *page_indices, uint32_t num_indices, bool invalidate_super_sampling)
{
	RangeMerger merger;

	// page_indices is implicitly sorted, so a trivial linear merger is fine and works well.

	const auto flush_cb = [&](VkDeviceSize offset, VkDeviceSize range)
	{
		cmd.copy_buffer(dst, offset, src, offset, range);
		// Invalidate all super-sampling state for pixels which CPU clobber.
		if (invalidate_super_sampling)
			cmd.fill_buffer(dst, 0, offset + vram_size, range);
	};

	for (uint32_t i = 0; i < num_indices; i++)
		merger.push(page_indices[i] * PageSize, PageSize, flush_cb);
	merger.flush(flush_cb);
}

void GSRenderer::flush_host_vram_copy(const uint32_t *page_indices, uint32_t num_indices)
{
	bool first_command = !async_transfer_cmd;
	ensure_command_buffer(async_transfer_cmd, Vulkan::CommandBuffer::Type::AsyncTransfer);
	auto &cmd = *async_transfer_cmd;

	if (first_command)
		cmd.begin_region("AsyncTransfer");

	cmd.begin_region("flush-host-vram-copy");

	Vulkan::QueryPoolHandle start_ts, end_ts;

	VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_COPY_BIT;
	bool invalidate_super_sampling = can_potentially_super_sample();
	if (invalidate_super_sampling)
		stages |= VK_PIPELINE_STAGE_2_CLEAR_BIT;

	if (enable_timestamps)
		start_ts = cmd.write_timestamp(stages);

	copy_pages(cmd, *buffers.gpu, *buffers.cpu, page_indices, num_indices, invalidate_super_sampling);
	stats.num_copies += num_indices;

	if (enable_timestamps)
	{
		end_ts = cmd.write_timestamp(stages);
		timestamps.push_back({ TimestampType::SyncHostToVRAM, std::move(start_ts), std::move(end_ts) });
	}

	cmd.barrier(stages, VK_ACCESS_2_TRANSFER_WRITE_BIT,
	            stages, VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);

	cmd.end_region();
	check_flush_stats();
}

void GSRenderer::flush_readback(const uint32_t *page_indices, uint32_t num_indices)
{
	ensure_command_buffer(direct_cmd, Vulkan::CommandBuffer::Type::Generic);
	auto &cmd = *direct_cmd;

	cmd.begin_region("flush-readback");

	Vulkan::QueryPoolHandle start_ts, end_ts;
	if (enable_timestamps)
		start_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_2_COPY_BIT);
	copy_pages(cmd, *buffers.cpu, *buffers.gpu, page_indices, num_indices, false);
	stats.num_copies += num_indices;

	if (enable_timestamps)
	{
		end_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_2_COPY_BIT);
		timestamps.push_back({ TimestampType::Readback, std::move(start_ts), std::move(end_ts) });
	}

	cmd.barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
	            VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
	            VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);

	cmd.end_region();
	check_flush_stats();
}

void GSRenderer::ensure_command_buffer(Vulkan::CommandBufferHandle &cmd, Vulkan::CommandBuffer::Type type)
{
	if (!cmd)
		cmd = device->request_command_buffer(type);
}

Vulkan::BindlessDescriptorPoolHandle GSRenderer::get_bindless_pool()
{
	uint64_t progress = query_timeline(*descriptor_timeline);

	if (!exhausted_descriptor_pools.empty() && progress >= exhausted_descriptor_pools.front().timeline)
	{
		auto ret = std::move(exhausted_descriptor_pools.front().exhausted_pool);
		exhausted_descriptor_pools.pop();
		ret->reset();
		return ret;
	}
	else
	{
		return device->create_bindless_descriptor_pool(Vulkan::BindlessResourceType::Image, 4096, 64 * 1024);
	}
}

void GSRenderer::bind_textures(Vulkan::CommandBuffer &cmd, const RenderPass &rp)
{
	VK_ASSERT(rp.num_textures <= MaxTextures);
	auto *tex_infos = cmd.allocate_typed_constant_data<TexInfo>(
			0, BINDING_TEXTURE_INFO, std::max<uint32_t>(1, rp.num_textures));

	if (!bindless_allocator)
		bindless_allocator = get_bindless_pool();

	if (!bindless_allocator)
	{
		LOGE("Failed to allocate bindless set.\n");
		return;
	}

	if (rp.num_textures == 0 && bindless_allocator->get_descriptor_set() != VK_NULL_HANDLE)
	{
		// Just bind dummy set and forget.
		cmd.set_bindless(DESCRIPTOR_SET_IMAGES, bindless_allocator->get_descriptor_set());
		return;
	}

	uint32_t to_allocate = std::max<uint32_t>(1, rp.num_textures);

	if (!bindless_allocator->allocate_descriptors(to_allocate))
	{
		exhausted_descriptor_pools.push({ std::move(bindless_allocator), next_descriptor_timeline_signal });
		bindless_allocator = get_bindless_pool();
		if (!bindless_allocator || !bindless_allocator->allocate_descriptors(to_allocate))
		{
			LOGE("Failed to allocate descriptors from a fresh set.\n");
			return;
		}
	}

	for (uint32_t i = 0; i < rp.num_textures; i++)
	{
		bindless_allocator->push_texture(*rp.textures[i].view);
		tex_infos[i] = rp.textures[i].info;
	}
	bindless_allocator->update();
	cmd.set_bindless(DESCRIPTOR_SET_IMAGES, bindless_allocator->get_descriptor_set());
}

void GSRenderer::bind_frame_resources(const RenderPass &rp)
{
	auto &cmd = *direct_cmd;

	memcpy(cmd.allocate_typed_constant_data<StateVector>(
			       0, BINDING_STATE_VECTORS, std::max<uint32_t>(1, rp.num_states)),
	       rp.states, rp.num_states * sizeof(StateVector));

	cmd.set_storage_buffer(0, BINDING_CLUT, *buffers.clut);
	cmd.set_sampler(0, BINDING_SAMPLER_NEAREST, Vulkan::StockSampler::NearestWrap);
	cmd.set_sampler(0, BINDING_SAMPLER_LINEAR, Vulkan::StockSampler::LinearWrap);

	cmd.set_storage_buffer(0, BINDING_VRAM, *buffers.gpu);

	triangle_setup_cmd->set_buffer_view(0, BINDING_FIXED_RCP_LUT, *buffers.fixed_rcp_lut_view);
	triangle_setup_cmd->set_buffer_view(0, BINDING_FLOAT_RCP_LUT, *buffers.float_rcp_lut_view);

	auto pos_offset = allocate_device_scratch(
			rp.num_primitives * 3 * sizeof(rp.positions[0]),
			buffers.rebar_scratch, rp.positions);
	triangle_setup_cmd->set_storage_buffer(0, BINDING_VERTEX_POSITION, *buffers.rebar_scratch.buffer,
	                                       pos_offset, rp.num_primitives * 3 * sizeof(rp.positions[0]));
	if (heuristic_cmd)
	{
		heuristic_cmd->set_storage_buffer(0, BINDING_VERTEX_POSITION, *buffers.rebar_scratch.buffer,
		                                  pos_offset, rp.num_primitives * 3 * sizeof(rp.positions[0]));
	}

	auto attr_offset = allocate_device_scratch(
			rp.num_primitives * 3 * sizeof(rp.attributes[0]),
			buffers.rebar_scratch, rp.attributes);
	triangle_setup_cmd->set_storage_buffer(0, BINDING_VERTEX_ATTRIBUTES, *buffers.rebar_scratch.buffer,
	                                       attr_offset, rp.num_primitives * 3 * sizeof(rp.attributes[0]));

	auto prim_offset = allocate_device_scratch(
			rp.num_primitives * sizeof(rp.prims[0]),
			buffers.rebar_scratch, rp.prims);
	cmd.set_storage_buffer(0, BINDING_PRIMITIVE_ATTRIBUTES, *buffers.rebar_scratch.buffer,
	                       prim_offset, rp.num_primitives * sizeof(rp.prims[0]));
	triangle_setup_cmd->set_storage_buffer(0, BINDING_PRIMITIVE_ATTRIBUTES, *buffers.rebar_scratch.buffer,
	                                       prim_offset, rp.num_primitives * sizeof(rp.prims[0]));

	if (heuristic_cmd)
	{
		heuristic_cmd->set_storage_buffer(0, BINDING_PRIMITIVE_ATTRIBUTES, *buffers.rebar_scratch.buffer,
		                                  prim_offset, rp.num_primitives * sizeof(rp.prims[0]));
	}
	binning_cmd->set_storage_buffer(0, BINDING_PRIMITIVE_ATTRIBUTES, *buffers.rebar_scratch.buffer,
	                                prim_offset, rp.num_primitives * sizeof(rp.prims[0]));
}

void GSRenderer::bind_frame_resources_instanced(const RenderPass &rp, uint32_t instance, uint32_t num_primitives)
{
	auto &cmd = *direct_cmd;

	GlobalConstants constants = {};

	auto &inst = rp.instances[instance];
	constants.base_pixel.x = int(inst.base_x);
	constants.base_pixel.y = int(inst.base_y);
	constants.coarse_tile_size_log2 = int(rp.coarse_tile_size_log2);
	constants.coarse_fb_width = int(inst.coarse_tiles_width);
	constants.coarse_primitive_list_stride = int(num_primitives);
	constants.fb_color_page = int(inst.fb.frame.desc.FBP);
	constants.fb_depth_page = int(inst.fb.z.desc.ZBP);
	constants.fb_page_stride = int(inst.fb.frame.desc.FBW);

	*cmd.allocate_typed_constant_data<GlobalConstants>(0, BINDING_CONSTANTS, 1) = constants;
	*binning_cmd->allocate_typed_constant_data<GlobalConstants>(0, BINDING_CONSTANTS, 1) = constants;
}

void GSRenderer::allocate_scratch_buffers(Vulkan::CommandBuffer &cmd, const RenderPass &rp)
{
	VkDeviceSize primitive_setup_size = sizeof(PrimitiveSetup) * rp.num_primitives;
	VkDeviceSize transformed_attributes_size = sizeof(TransformedAttributes) * rp.num_primitives;

	auto primitive_setup_offset = allocate_device_scratch(primitive_setup_size, buffers.device_scratch, nullptr);
	cmd.set_storage_buffer(0, BINDING_PRIMITIVE_SETUP,
	                       *buffers.device_scratch.buffer, primitive_setup_offset, primitive_setup_size);
	triangle_setup_cmd->set_storage_buffer(0, BINDING_PRIMITIVE_SETUP,
	                                       *buffers.device_scratch.buffer, primitive_setup_offset, primitive_setup_size);
	binning_cmd->set_storage_buffer(0, BINDING_PRIMITIVE_SETUP,
	                                *buffers.device_scratch.buffer, primitive_setup_offset, primitive_setup_size);

	auto transformed_attributes_offset = allocate_device_scratch(transformed_attributes_size, buffers.device_scratch, nullptr);
	cmd.set_storage_buffer(0, BINDING_TRANSFORMED_ATTRIBUTES,
	                       *buffers.device_scratch.buffer, transformed_attributes_offset, transformed_attributes_size);
	triangle_setup_cmd->set_storage_buffer(0, BINDING_TRANSFORMED_ATTRIBUTES,
	                                       *buffers.device_scratch.buffer, transformed_attributes_offset, transformed_attributes_size);

	auto single_sampled_heuristic_offset =
			allocate_device_scratch(sizeof(SingleSampleHeuristic), buffers.device_scratch, nullptr);
	triangle_setup_cmd->set_storage_buffer(0, BINDING_SINGLE_SAMPLE_HEURISTIC,
	                                       *buffers.device_scratch.buffer,
	                                       single_sampled_heuristic_offset, sizeof(SingleSampleHeuristic));

	for (uint32_t i = 0; i < rp.num_instances; i++)
	{
		if (rp.instances[i].sampling_rate_y_log2 != 0)
		{
			heuristic_cmd->set_storage_buffer(0, BINDING_SINGLE_SAMPLE_HEURISTIC,
			                                  *buffers.device_scratch.buffer,
			                                  single_sampled_heuristic_offset, sizeof(SingleSampleHeuristic));

			for (VkDeviceSize bda_offset = 0; bda_offset < sizeof(SingleSampleHeuristic); bda_offset += 16)
				qword_clears.push_back(buffers.device_scratch.buffer->get_device_address() + single_sampled_heuristic_offset + bda_offset);

			indirect_single_sample_heuristic = buffers.device_scratch;
			indirect_single_sample_heuristic.offset = single_sampled_heuristic_offset;
			indirect_single_sample_heuristic.size = sizeof(SingleSampleHeuristic);
			break;
		}
	}
}

void GSRenderer::allocate_scratch_buffers_instanced(Vulkan::CommandBuffer &cmd, const RenderPass &rp,
                                                    uint32_t instance, uint32_t num_primitives)
{
	auto &inst = rp.instances[instance];

	VkDeviceSize num_coarse_tiles = inst.coarse_tiles_width * inst.coarse_tiles_height;
	VkDeviceSize primitive_count_size = num_coarse_tiles * sizeof(uint32_t);
	VkDeviceSize primitive_list_size = num_coarse_tiles * num_primitives * sizeof(uint16_t);

	auto primitive_count_offset = allocate_device_scratch(primitive_count_size, buffers.device_scratch, nullptr);
	cmd.set_storage_buffer(0, BINDING_COARSE_PRIMITIVE_COUNT, *buffers.device_scratch.buffer,
	                       primitive_count_offset, primitive_count_size);
	binning_cmd->set_storage_buffer(0, BINDING_COARSE_PRIMITIVE_COUNT, *buffers.device_scratch.buffer,
	                                primitive_count_offset, primitive_count_size);

	auto primitive_list_offset = allocate_device_scratch(primitive_list_size, buffers.device_scratch, nullptr);
	cmd.set_storage_buffer(0, BINDING_COARSE_TILE_LIST, *buffers.device_scratch.buffer,
	                       primitive_list_offset, primitive_list_size);
	binning_cmd->set_storage_buffer(0, BINDING_COARSE_TILE_LIST, *buffers.device_scratch.buffer,
	                                primitive_list_offset, primitive_list_size);

	VkDeviceSize work_list_size = 256 + inst.coarse_tiles_width * inst.coarse_tiles_height * sizeof(uvec2);

	auto work_list_scratch = allocate_device_scratch(work_list_size, buffers.device_scratch, nullptr);
	qword_clears.push_back(buffers.device_scratch.buffer->get_device_address() + work_list_scratch);

	binning_cmd->set_storage_buffer(1, 0, *buffers.device_scratch.buffer, work_list_scratch, work_list_size);
	work_list_single_sample = buffers.device_scratch;
	work_list_single_sample.offset = work_list_scratch;
	work_list_single_sample.size = work_list_size;

	if (inst.sampling_rate_y_log2 != 0)
	{
		work_list_scratch = allocate_device_scratch(work_list_size, buffers.device_scratch, nullptr);
		qword_clears.push_back(buffers.device_scratch.buffer->get_device_address() + work_list_scratch);
		binning_cmd->set_storage_buffer(1, 1, *buffers.device_scratch.buffer, work_list_scratch, work_list_size);
		work_list_super_sample = buffers.device_scratch;
		work_list_super_sample.offset = work_list_scratch;
		work_list_super_sample.size = work_list_size;
	}
	else
		binning_cmd->set_storage_buffer(1, 1, *buffers.device_scratch.buffer, work_list_scratch, work_list_size);
}

void GSRenderer::dispatch_triangle_setup(Vulkan::CommandBuffer &cmd, const RenderPass &rp)
{
	struct Push
	{
		uint32_t num_primitives;
		uint32_t z_shift_to_bucket;
	} push = {};

	push.num_primitives = rp.num_primitives;
	uint32_t sampling_rate_y_log2 = 0;

	for (uint32_t i = 0; i < rp.num_instances; i++)
	{
		auto &inst = rp.instances[i];

		switch (inst.fb.z.desc.PSM | ZBUFBits::PSM_MSB)
		{
		case PSMZ16S:
		case PSMZ16:
			push.z_shift_to_bucket |= 1 << (4 * i);
			break;

		case PSMZ24:
			push.z_shift_to_bucket |= 2 << (4 * i);
			break;

		case PSMZ32:
			push.z_shift_to_bucket |= 3 << (4 * i);
			break;

		default:
			LOGE("Unexpected Z format: %u\n", inst.fb.z.desc.PSM);
			break;
		}

		sampling_rate_y_log2 = std::max<uint32_t>(sampling_rate_y_log2, inst.sampling_rate_y_log2);
	}

	cmd.push_constants(&push, 0, sizeof(push));

	cmd.set_program(shaders.triangle_setup);
	cmd.set_specialization_constant_mask(1);
	cmd.set_specialization_constant(0, sampling_rate_y_log2);
	Vulkan::QueryPoolHandle start_ts, end_ts;
	if (enable_timestamps)
		start_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	cmd.dispatch((rp.num_primitives + 63) / 64, 1, 1);
	if (enable_timestamps)
	{
		end_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		timestamps.push_back({ TimestampType::TriangleSetup, std::move(start_ts), std::move(end_ts) });
	}
}

void GSRenderer::dispatch_single_sample_heuristic(Vulkan::CommandBuffer &cmd, const RenderPass &rp)
{
	cmd.set_specialization_constant_mask(1);
	cmd.set_program(shaders.single_sample_heuristic);

	cmd.set_specialization_constant(
			0, std::min<uint32_t>(512u, device->get_gpu_properties().limits.maxComputeWorkGroupSize[0]));

	struct Push
	{
		uint32_t num_primitives;
		uint32_t z_shift_bucket;
		uint32_t z_shift;
		uint32_t z_max;
		uint32_t instance;
	} push = {};

	push.num_primitives = rp.num_primitives;

	for (uint32_t i = 0; i < rp.num_instances; i++)
	{
		auto &inst = rp.instances[i];
		push.instance = i;

		if (inst.sampling_rate_y_log2 == 0)
			continue;

		// If we don't know, assume 24-bit range. If Z buffer isn't used at all, it's unlikely there will be proper 3D objects anyway.
		uint32_t depth_psm = inst.z_sensitive ? (inst.fb.z.desc.PSM | ZBUFBits::PSM_MSB) : PSMZ24;

		switch (depth_psm)
		{
		case PSMZ16S:
		case PSMZ16:
			push.z_shift_bucket = 8;
			push.z_shift = 0;
			push.z_max = 0xffffu;
			break;

		case PSMZ24:
			push.z_shift_bucket = 16;
			push.z_shift = 0;
			push.z_max = 0xffffffu;
			break;

		case PSMZ32:
			push.z_shift_bucket = 16;
			push.z_shift = 8;
			push.z_max = UINT32_MAX;
			break;

		default:
			LOGE("Unexpected Z format: %u\n", inst.fb.z.desc.PSM);
			break;
		}

		cmd.push_constants(&push, 0, sizeof(push));
		cmd.dispatch_indirect(*indirect_single_sample_heuristic.buffer, indirect_single_sample_heuristic.offset);
	}

	cmd.set_specialization_constant_mask(0);
}

void GSRenderer::dispatch_binning(Vulkan::CommandBuffer &cmd, const RenderPass &rp,
                                  uint32_t instance, uint32_t base_primitive, uint32_t num_primitives)
{
	auto &inst = rp.instances[instance];

	cmd.enable_subgroup_size_control(true);
	cmd.set_specialization_constant_mask(0x7);
	cmd.set_specialization_constant(1, uint32_t(rp.feedback_color || rp.feedback_depth));
	cmd.set_specialization_constant(2, uint32_t(inst.sampling_rate_y_log2 != 0));

	// Prefer large waves if possible. Also, prefer to have just one subgroup per workgroup,
	// since the algorithms kind of rely on that.
	// Using larger workgroups it's feasible to do two-phase binning, but given the content and current perf-metrics,
	// it doesn't seem meaningful perf-wise.
	const struct
	{
		uint32_t lo;
		uint32_t hi;
		uint32_t wg_size;
	} attempts[] = {
		{ 6, 6, 64 },
		{ 5, 5, 32 },
		{ 4, 4, 16 },
		{ 3, 3, 8 },
		{ 2, 2, 4 },
		{ 2, 6, std::min<uint32_t>(64, device->get_device_features().vk11_props.subgroupSize) },
	};

	for (auto &attempt : attempts)
	{
		if (device->supports_subgroup_size_log2(true, attempt.lo, attempt.hi))
		{
			cmd.set_subgroup_size_log2(true, attempt.lo, attempt.hi);
			cmd.set_specialization_constant(0, attempt.wg_size);
		}
	}

	struct Push
	{
		uint32_t base_x, base_y;
		uint32_t base_primitive;
		uint32_t instance;
		uint32_t end_primitives;
		uint32_t num_samples;
	} push = {
		inst.base_x, inst.base_y,
		base_primitive, instance, base_primitive + num_primitives,
		1u << (inst.sampling_rate_x_log2 + inst.sampling_rate_y_log2),
	};

	cmd.push_constants(&push, 0, sizeof(push));
	cmd.set_program(shaders.binning);
	Vulkan::QueryPoolHandle start_ts, end_ts;
	if (enable_timestamps)
		start_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	cmd.dispatch(inst.coarse_tiles_width, inst.coarse_tiles_height, 1);
	if (enable_timestamps)
	{
		end_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		timestamps.push_back({ TimestampType::Binning, std::move(start_ts), std::move(end_ts) });
	}
	cmd.enable_subgroup_size_control(false);
	cmd.set_specialization_constant_mask(0);
}

static uint32_t deduce_effectice_z_psm(uint32_t color_psm, uint32_t depth_psm)
{
	// Somewhat speculative, and may not be relevant anymore.
	// Probably only really relevant if color is Z swizzle and depth needs to be CT swizzle (?).

	if (color_psm == PSMCT32 || color_psm == PSMCT24 || color_psm == PSMCT16S) // Group 1
	{
		if (depth_psm == PSMCT32)
			return PSMZ32;
		else if (depth_psm == PSMCT24)
			return PSMZ24;
		else if (depth_psm == PSMCT16 || depth_psm == PSMCT16S)
			return PSMZ16S;
	}
	else if (color_psm == PSMZ32 || color_psm == PSMZ24 || color_psm == PSMZ16S)
	{
		// Whacky group 1 (does this make sense? Does color or depth win?)
		if (depth_psm == PSMZ32)
			return PSMCT32;
		else if (depth_psm == PSMZ24)
			return PSMCT24;
		else if (depth_psm == PSMZ16S || depth_psm == PSMZ16)
			return PSMCT16S;
	}

#if 0
	// Trying to fix up anything here only seems to cause issues.
	// FIXME: Figure out what is supposed to happen.
	else if (color_psm == PSMCT16) // Group 2
		return PSMZ16;
	else if (color_psm == PSMZ16) // Whacky group 2
		return PSMCT16;
#endif

	return depth_psm;
}

void GSRenderer::dispatch_cache_read_only_depth(Vulkan::CommandBuffer &cmd, const RenderPass &rp,
                                                uint32_t depth_psm, uint32_t instance)
{
	auto &inst = rp.instances[instance];
	auto z_rect = compute_page_rect(inst.fb.z.desc.ZBP * PGS_BLOCKS_PER_PAGE,
	                                inst.base_x, inst.base_y, inst.coarse_tiles_width << rp.coarse_tile_size_log2,
	                                inst.coarse_tiles_height << rp.coarse_tile_size_log2, inst.fb.frame.desc.FBW,
	                                depth_psm);

	uint32_t num_vram_slices = 1;
	uint32_t sample_rate_log2 = inst.sampling_rate_y_log2 + inst.sampling_rate_x_log2;
	if (sample_rate_log2 > 0)
		num_vram_slices = 2u + (1u << sample_rate_log2);

	// Upper half of VRAM is reserved for read-only caching.
	VkDeviceSize read_only_offset = buffers.gpu->get_create_info().size / 2;

	cmd.begin_region("depth-read-only-cache");
	cmd.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
				VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	{
		RangeMerger merger;
		const auto flush_range = [&](VkDeviceSize offset, VkDeviceSize range) {
			for (uint32_t slice = 0; slice < num_vram_slices; slice++)
			{
				cmd.copy_buffer(*buffers.gpu, vram_size * slice + offset + read_only_offset,
				                *buffers.gpu, vram_size * slice + offset,
				                range);
			}
		};

		for (uint32_t y = 0; y < z_rect.page_height; y++)
		{
			for (uint32_t x = 0; x < z_rect.page_width; x++)
			{
				uint32_t page_index = z_rect.base_page + y * z_rect.page_stride + x;
				uint32_t page_offset = (page_index * PGS_PAGE_ALIGNMENT_BYTES) & (vram_size - 1);
				merger.push(page_offset, PGS_PAGE_ALIGNMENT_BYTES, flush_range);
			}
		}

		merger.flush(flush_range);
	}
	cmd.barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
	            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
	cmd.end_region();
}

void GSRenderer::bind_debug_resources(Vulkan::CommandBuffer &cmd,
                                      const RenderPass &rp,
                                      const RenderPass::Instance &inst)
{
	const auto image_info_need_recreate = [](const Vulkan::ImageCreateInfo &info,
	                                         const Vulkan::ImageHandle &handle) {
		return !handle ||
		       handle->get_width() < info.width ||
		       handle->get_height() < info.height ||
		       handle->get_format() != info.format ||
		       handle->get_create_info().layers != info.layers;
	};

	const auto clear_image = [&](const Vulkan::Image &img) {
		cmd.image_barrier(img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		                  VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
		cmd.clear_image(img, {}, VK_IMAGE_ASPECT_COLOR_BIT);
		cmd.image_barrier(img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
		                  VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
		                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
	};

	if (rp.feedback_color)
	{
		auto info = Vulkan::ImageCreateInfo::immutable_2d_image(
				inst.coarse_tiles_width << rp.coarse_tile_size_log2,
				inst.coarse_tiles_height << rp.coarse_tile_size_log2,
				VK_FORMAT_R8G8B8A8_UNORM);

		info.width <<= inst.sampling_rate_y_log2;
		info.height <<= inst.sampling_rate_y_log2;

		info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
		info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		info.layers = inst.sampling_rate_y_log2 || inst.sampling_rate_x_log2 ? 5 : 2;

		if (image_info_need_recreate(info, feedback_color))
			feedback_color = device->create_image(info);

		info.layers = 1;
		info.format = VK_FORMAT_R32G32B32A32_UINT;
		if (image_info_need_recreate(info, feedback_prim))
			feedback_prim = device->create_image(info);

		info.layers = 3;
		info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
		if (image_info_need_recreate(info, feedback_vary))
			feedback_vary = device->create_image(info);

		if (inst.sampling_rate_x_log2 || inst.sampling_rate_y_log2)
			device->set_name(*feedback_color, "DebugColor - [after, before, after 1x, before 1x, ref 1x]");
		else
			device->set_name(*feedback_color, "DebugColor - [after, before]");

		device->set_name(*feedback_prim, "DebugStat - [ShadeCount, LastShadePrim, CoverageCount, TexMask]");
		device->set_name(*feedback_vary, "DebugVary - [UV, Q, LOD] [RGBA] [Z, IJ]");

		clear_image(*feedback_color);
		clear_image(*feedback_prim);
		clear_image(*feedback_vary);

		cmd.set_storage_texture(0, BINDING_FEEDBACK_COLOR, feedback_color->get_view());
		cmd.set_storage_texture(0, BINDING_FEEDBACK_PRIM, feedback_prim->get_view());
		cmd.set_storage_texture(0, BINDING_FEEDBACK_VARY, feedback_vary->get_view());
	}

	if (rp.feedback_depth)
	{
		auto info = Vulkan::ImageCreateInfo::immutable_2d_image(
				inst.coarse_tiles_width << rp.coarse_tile_size_log2,
				inst.coarse_tiles_height << rp.coarse_tile_size_log2,
				VK_FORMAT_R32_UINT);

		info.width <<= inst.sampling_rate_y_log2;
		info.height <<= inst.sampling_rate_y_log2;

		info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
		info.usage = VK_IMAGE_USAGE_STORAGE_BIT;
		info.layers = 2;

		if (image_info_need_recreate(info, feedback_depth))
			feedback_depth = device->create_image(info);

		clear_image(*feedback_depth);
		cmd.set_storage_texture(0, BINDING_FEEDBACK_DEPTH, feedback_depth->get_view());
		device->set_name(*feedback_depth, "DebugDepth - layer 0 = after - layer 1 = before");
	}
}

void GSRenderer::dispatch_shading(Vulkan::CommandBuffer &cmd, const RenderPass &rp, uint32_t instance,
                                  uint32_t base_primitive, uint32_t num_primitives)
{
	cmd.begin_region("shading");
	auto &inst = rp.instances[instance];

	bind_debug_resources(cmd, rp, inst);

	// The shading itself runs at workgroup size == 64, we can handle multiple subgroups, but prefer large waves,
	// since we're doing so much scalar work.

	cmd.set_specialization_constant_mask(0xff);

	cmd.enable_subgroup_size_control(true);

	// Prefer Wave64 if we can get away with it.
	if (device->supports_subgroup_size_log2(true, 6, 6))
		cmd.set_subgroup_size_log2(true, 6, 6);
	else if (device->supports_subgroup_size_log2(true, 4, 6))
		cmd.set_subgroup_size_log2(true, 4, 6);
	else if (device->supports_subgroup_size_log2(true, 3, 6))
		cmd.set_subgroup_size_log2(true, 3, 6);
	else
		cmd.set_subgroup_size_log2(true, 2, 6);

	uint32_t color_psm = inst.fb.frame.desc.PSM;
	uint32_t depth_psm = inst.fb.z.desc.PSM | ZBUFBits::PSM_MSB;

	if (inst.z_sensitive)
	{
		depth_psm = deduce_effectice_z_psm(color_psm, depth_psm);

#ifdef PARALELL_GS_DEBUG
		if (depth_psm != (rp.fb.z.desc.PSM | ZBUFBits::PSM_MSB))
		{
			LOGW("Invalid Z format detected, changed PSM %s to %s.\n",
			     psm_to_str(rp.fb.z.desc.PSM | ZBUFBits::PSM_MSB), psm_to_str(depth_psm));
		}
#endif
	}

	cmd.set_specialization_constant(2, color_psm);
	cmd.set_specialization_constant(3, inst.z_sensitive ? depth_psm : UINT32_MAX);
	cmd.set_specialization_constant(4, vram_size - 1);

	uint32_t variant_flags = 0;

	bool fb_z_alias = inst.z_sensitive && inst.fb.frame.desc.FBP == inst.fb.z.desc.ZBP;
	uint32_t last_primitive_index_single_step = 0;
	uint32_t last_primitive_index_z_sensitive = 0;
	bool single_primitive_step = false;

	if (fb_z_alias)
	{
		for (uint32_t i = 0; i < num_primitives; i++)
		{
			// If we have duelling color and Z write we're in deep trouble, so have to fall back
			// to single primitive stepping.
			if ((rp.prims[base_primitive + i].state & (1u << STATE_BIT_Z_WRITE)) != 0)
			{
				single_primitive_step = true;
				last_primitive_index_single_step = i;
				last_primitive_index_z_sensitive = i;
			}
			else if ((rp.prims[base_primitive + i].state & (1u << STATE_BIT_Z_TEST)) != 0)
			{
				last_primitive_index_z_sensitive = i;
			}
		}
	}

	if (rp.has_aa1)
		variant_flags |= VARIANT_FLAG_HAS_AA1_BIT;
	if (rp.has_scanmsk)
		variant_flags |= VARIANT_FLAG_HAS_SCANMSK_BIT;

	if (single_primitive_step)
		variant_flags |= VARIANT_FLAG_HAS_PRIMITIVE_RANGE_BIT;

	if (rp.feedback_mode != RenderPass::Feedback::None)
	{
		cmd.set_specialization_constant(6, rp.feedback_texture_psm);
		cmd.set_specialization_constant(7, rp.feedback_texture_cpsm);
		variant_flags |= VARIANT_FLAG_FEEDBACK_BIT;
		if (rp.feedback_mode == RenderPass::Feedback::Depth)
			variant_flags |= VARIANT_FLAG_FEEDBACK_DEPTH_BIT;
	}
	else
	{
		cmd.set_specialization_constant(6, 0);
		cmd.set_specialization_constant(7, 0);
	}

	if (can_potentially_super_sample())
		variant_flags |= VARIANT_FLAG_HAS_SUPER_SAMPLE_REFERENCE_BIT;

	cmd.set_specialization_constant(5, variant_flags);

	assert(inst.sampling_rate_x_log2 <= 2);
	assert(inst.sampling_rate_y_log2 <= 2);

	// Only way to make this work is to cache VRAM into a shadow copy.
	uint32_t fb_index_depth_offset = 0;
	if (fb_z_alias && !single_primitive_step)
	{
		uint32_t half_gpu_size = buffers.gpu->get_create_info().size / 2;
		if (get_bits_per_pixel(depth_psm) == 16)
			fb_index_depth_offset = half_gpu_size / sizeof(uint16_t);
		else
			fb_index_depth_offset = half_gpu_size / sizeof(uint32_t);

		dispatch_cache_read_only_depth(cmd, rp, depth_psm, instance);
	}

	cmd.set_program(shaders.ubershader[int(rp.feedback_color)][int(rp.feedback_depth)]);
	ShadingDescriptor push = { base_primitive, base_primitive + num_primitives - 1, fb_index_depth_offset };

	if (rp.feedback_color)
	{
		dispatch_shading_debug(cmd, rp, push, instance, base_primitive, num_primitives);
	}
	else if (single_primitive_step)
	{
		// Pure mayhem, game will rely on non-local feedback effects due to Z swizzling.
		// Render one primitive at a time.
		// Not much we can do about this other than render two tiles in sync in a single workgroup,
		// and then do barrier() after every primitive and exchange color and depth values as needed.
		// That however, is complete insanity, but if we end up seeing content that really hammers
		// this hard, we might not have a choice ...
		for (uint32_t i = 0; i <= last_primitive_index_single_step; i++)
		{
			push.lo_primitive_index = base_primitive + i;
			push.hi_primitive_index = base_primitive + i;
			cmd.push_constants(&push, 0, sizeof(push));

			if (inst.sampling_rate_y_log2 != 0)
			{
				cmd.set_specialization_constant(0, inst.sampling_rate_x_log2);
				cmd.set_specialization_constant(1, inst.sampling_rate_y_log2);
				cmd.set_storage_buffer(DESCRIPTOR_SET_WORKGROUP_LIST, 0,
				                       *work_list_super_sample.buffer, work_list_super_sample.offset + 256, VK_WHOLE_SIZE);
				cmd.dispatch_indirect(*work_list_super_sample.buffer, work_list_super_sample.offset);
			}

			cmd.set_specialization_constant(0, 0);
			cmd.set_specialization_constant(1, 0);
			cmd.set_storage_buffer(DESCRIPTOR_SET_WORKGROUP_LIST, 0,
			                       *work_list_single_sample.buffer, work_list_single_sample.offset + 256, VK_WHOLE_SIZE);
			cmd.dispatch_indirect(*work_list_single_sample.buffer, work_list_single_sample.offset);

			cmd.barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
			            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
		}

		// If there is still work to do that is read-only, use cached depth for the rest of the render pass.
		if (last_primitive_index_single_step + 1 < num_primitives)
		{
			if (last_primitive_index_z_sensitive > last_primitive_index_single_step)
			{
				// If we still need to read depth, have to cache it read-only.
				uint32_t half_gpu_size = buffers.gpu->get_create_info().size / 2;
				if (get_bits_per_pixel(depth_psm) == 16)
					fb_index_depth_offset = half_gpu_size / sizeof(uint16_t);
				else
					fb_index_depth_offset = half_gpu_size / sizeof(uint32_t);
				push.fb_index_depth_offset = fb_index_depth_offset;

				dispatch_cache_read_only_depth(cmd, rp, depth_psm, instance);
			}
			else
			{
				// If no further primitives need to read-depth, treat it as non-depth to speed up things.
				cmd.set_specialization_constant(3, UINT32_MAX);
			}

			push.lo_primitive_index = base_primitive + last_primitive_index_single_step + 1;
			push.hi_primitive_index = UINT32_MAX;
			cmd.push_constants(&push, 0, sizeof(push));

			if (inst.sampling_rate_y_log2 != 0)
			{
				cmd.set_specialization_constant(0, inst.sampling_rate_x_log2);
				cmd.set_specialization_constant(1, inst.sampling_rate_y_log2);
				cmd.set_storage_buffer(DESCRIPTOR_SET_WORKGROUP_LIST, 0,
				                       *work_list_super_sample.buffer, work_list_super_sample.offset + 256, VK_WHOLE_SIZE);
				cmd.dispatch_indirect(*work_list_super_sample.buffer, work_list_super_sample.offset);
			}

			cmd.set_specialization_constant(0, 0);
			cmd.set_specialization_constant(1, 0);
			cmd.set_storage_buffer(DESCRIPTOR_SET_WORKGROUP_LIST, 0,
			                       *work_list_single_sample.buffer, work_list_single_sample.offset + 256, VK_WHOLE_SIZE);
			cmd.dispatch_indirect(*work_list_single_sample.buffer, work_list_single_sample.offset);
		}
	}
	else
	{
		cmd.push_constants(&push, 0, sizeof(push));

		if (inst.sampling_rate_y_log2 != 0)
		{
			cmd.set_specialization_constant(0, inst.sampling_rate_x_log2);
			cmd.set_specialization_constant(1, inst.sampling_rate_y_log2);
			cmd.set_storage_buffer(DESCRIPTOR_SET_WORKGROUP_LIST, 0,
			                       *work_list_super_sample.buffer, work_list_super_sample.offset + 256, VK_WHOLE_SIZE);
			cmd.dispatch_indirect(*work_list_super_sample.buffer, work_list_super_sample.offset);
		}

		cmd.set_specialization_constant(0, 0);
		cmd.set_specialization_constant(1, 0);
		cmd.set_storage_buffer(DESCRIPTOR_SET_WORKGROUP_LIST, 0,
		                       *work_list_single_sample.buffer, work_list_single_sample.offset + 256, VK_WHOLE_SIZE);
		cmd.dispatch_indirect(*work_list_single_sample.buffer, work_list_single_sample.offset);
	}

	cmd.end_region();
	cmd.enable_subgroup_size_control(false);
}

void GSRenderer::dispatch_shading_debug(Vulkan::CommandBuffer &cmd, const RenderPass &rp,
                                        ShadingDescriptor push, uint32_t instance,
                                        uint32_t base_primitive, uint32_t num_primitives)
{
	auto &inst = rp.instances[instance];

	uint32_t stride = rp.debug_capture_stride ? rp.debug_capture_stride : num_primitives;
	for (uint32_t i = 0; i < num_primitives; i += stride)
	{
		push.lo_primitive_index = i;
		push.hi_primitive_index = std::min<uint32_t>(num_primitives, i + stride) - 1;
		push.lo_primitive_index += base_primitive;
		push.hi_primitive_index += base_primitive;

		cmd.push_constants(&push, 0, sizeof(push));

		if (device->consumes_debug_markers())
		{
			begin_region(cmd, "Prim [%u, %u]", push.lo_primitive_index, push.hi_primitive_index);
			for (uint32_t j = push.lo_primitive_index; j <= push.hi_primitive_index; j++)
			{
				auto s = rp.prims[j].state;

				uint32_t prim_instance = (s >> STATE_VERTEX_RENDER_PASS_INSTANCE_OFFSET) &
				                         ((1 << STATE_VERTEX_RENDER_PASS_INSTANCE_COUNT) - 1);

				if (prim_instance != instance)
					continue;

				insert_label(cmd, "Prim #%u", j);
				insert_label(cmd,
				             "  State: %u, ZTST: %u, ZGE: %u, ZWRITE: %u, AA1: %u, OPAQUE: %u, SPRITE: %u, QUAD: %u, IIP: %u, LINE: %u, FBMSK: 0x%x",
				             (s >> STATE_INDEX_BIT_OFFSET) & ((1u << STATE_INDEX_BIT_COUNT) - 1u),
				             (s >> STATE_BIT_Z_TEST) & 1,
				             (s >> STATE_BIT_Z_TEST_GREATER) & 1,
				             (s >> STATE_BIT_Z_WRITE) & 1,
				             (s >> STATE_BIT_MULTISAMPLE) & 1,
				             (s >> STATE_BIT_OPAQUE) & 1,
				             (s >> STATE_BIT_SPRITE) & 1,
				             (s >> STATE_BIT_PARALLELOGRAM) & 1,
				             (s >> STATE_BIT_IIP) & 1,
				             (s >> STATE_BIT_LINE) & 1,
							 rp.prims[j].fbmsk);

				auto alpha = rp.prims[j].alpha;
				insert_label(cmd, "  AFIX: %u, AREF: %u",
				             (alpha >> ALPHA_AFIX_OFFSET) & ((1u << ALPHA_AFIX_BITS) - 1u),
				             (alpha >> ALPHA_AREF_OFFSET) & ((1u << ALPHA_AREF_BITS) - 1u));

				auto tex = rp.prims[j].tex;
				insert_label(cmd, "  TEX: %u, MXL: %u, CLAMPS: %u, CLAMPT: %u, MAG: %u, MIN: %u, MIP: %u",
				             (tex >> TEX_TEXTURE_INDEX_OFFSET) & ((1u << TEX_TEXTURE_INDEX_BITS) - 1u),
				             (tex >> TEX_MAX_MIP_LEVEL_OFFSET) & ((1u << TEX_MAX_MIP_LEVEL_BITS) - 1u),
				             (tex & TEX_SAMPLER_CLAMP_S_BIT) ? 1 : 0,
				             (tex & TEX_SAMPLER_CLAMP_T_BIT) ? 1 : 0,
				             (tex & TEX_SAMPLER_MAG_LINEAR_BIT) ? 1 : 0,
				             (tex & TEX_SAMPLER_MIN_LINEAR_BIT) ? 1 : 0,
				             (tex & TEX_SAMPLER_MIPMAP_LINEAR_BIT) ? 1 : 0);
			}
			cmd.end_region();
		}

		if (inst.sampling_rate_y_log2 != 0)
		{
			cmd.set_specialization_constant(0, inst.sampling_rate_x_log2);
			cmd.set_specialization_constant(1, inst.sampling_rate_y_log2);
			cmd.insert_label("super-sample");
			cmd.set_storage_buffer(DESCRIPTOR_SET_WORKGROUP_LIST, 0,
			                       *work_list_super_sample.buffer, work_list_super_sample.offset + 256, VK_WHOLE_SIZE);
			cmd.dispatch_indirect(*work_list_super_sample.buffer, work_list_super_sample.offset);
		}

		cmd.set_specialization_constant(0, 0);
		cmd.set_specialization_constant(1, 0);
		cmd.insert_label("single-sample");
		cmd.set_storage_buffer(DESCRIPTOR_SET_WORKGROUP_LIST, 0,
		                       *work_list_single_sample.buffer, work_list_single_sample.offset + 256, VK_WHOLE_SIZE);
		cmd.dispatch_indirect(*work_list_single_sample.buffer, work_list_single_sample.offset);

		cmd.barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
	}
}

static bool copy_is_fused_nibble(const CopyDescriptor &desc)
{
	// One thread will write two values in PSMT4 mode.
	bool is_fused_nibble = desc.bitbltbuf.desc.DPSM == PSMT4;

	// If the address offsets are quirky (i.e., do not align to an 8x8 block),
	// the fused nibble approach is not going to work (without extremely hackery).
	// Fallback to atomics in this case to get sub-byte writes.
	if (is_fused_nibble && ((desc.trxpos.desc.DSAX | desc.trxpos.desc.DSAY |
	                         desc.trxreg.desc.RRW | desc.trxreg.desc.RRH) & 7) != 0)
		is_fused_nibble = false;

	return is_fused_nibble;
}

static uint32_t copy_pipeline_key(const CopyDescriptor &desc)
{
	// One thread will write two values in PSMT4 mode.
	bool is_fused_nibble = copy_is_fused_nibble(desc);

	uint32_t key = 0;
	key |= desc.bitbltbuf.desc.SPSM << 0;
	key |= desc.bitbltbuf.desc.DPSM << 8;
	key |= desc.trxdir.desc.XDIR << 16;
	key |= int(is_fused_nibble) << 24;

	return key;
}

void GSRenderer::emit_copy_vram(Vulkan::CommandBuffer &cmd,
								const uint32_t *dispatch_order, uint32_t num_dispatches, bool prepare_only)
{
	cmd.set_program(shaders.vram_copy);
	auto &base_desc = pending_copies[dispatch_order[0]].copy;

	// One thread will write two values in PSMT4 fused mode.
	bool is_fused_nibble = copy_is_fused_nibble(base_desc);
	const uint32_t workgroup_size = is_fused_nibble ? 32 : 64;

	cmd.set_specialization_constant_mask(0x7f);
	cmd.set_specialization_constant(0, workgroup_size);
	cmd.set_specialization_constant(1, base_desc.bitbltbuf.desc.SPSM);
	cmd.set_specialization_constant(2, base_desc.bitbltbuf.desc.DPSM);
	cmd.set_specialization_constant(3, vram_size - 1);
	cmd.set_specialization_constant(4, base_desc.trxdir.desc.XDIR);
	cmd.set_specialization_constant(5, uint32_t(can_potentially_super_sample()));
	cmd.set_specialization_constant(6, uint32_t(prepare_only));

	cmd.set_storage_buffer(0, 0, *buffers.gpu);
	cmd.set_storage_buffer(0, 1, *buffers.vram_copy_atomics);
	cmd.set_storage_buffer(0, 2, *buffers.vram_copy_payloads);

	uint32_t max_wgs = 0;
	for (uint32_t i = 0; i < num_dispatches; i++)
	{
		auto &desc = pending_copies[dispatch_order[i]].copy;
		max_wgs += ((desc.trxreg.desc.RRW + 7) / 8) * ((desc.trxreg.desc.RRH + 7) / 8);
	}

	constexpr uint32_t MaxWorkgroups = 4096;

	auto *work_items = cmd.allocate_typed_constant_data<uvec4>(1, 0, std::min<uint32_t>(MaxWorkgroups, max_wgs));
	uint32_t num_wgs = 0;

	const auto push_work = [&](uint32_t group_x, uint32_t group_y, uint32_t transfer_id) {
		if (num_wgs == MaxWorkgroups)
		{
			cmd.dispatch(num_wgs, 1, 1);
			max_wgs -= num_wgs;
			work_items = cmd.allocate_typed_constant_data<uvec4>(1, 0, std::min<uint32_t>(MaxWorkgroups, max_wgs));
			num_wgs = 0;
		}

		work_items[num_wgs++] = uvec4(group_x, group_y, transfer_id, 0);
	};

	for (uint32_t i = 0; i < num_dispatches; i++)
	{
		auto &desc = pending_copies[dispatch_order[i]].copy;
		assert(copy_pipeline_key(desc) == copy_pipeline_key(base_desc));

		if (device->consumes_debug_markers())
		{
			if (base_desc.trxdir.desc.XDIR == HOST_TO_LOCAL)
			{
				insert_label(cmd, "VRAMUpload #%u - 0x%x - %s - %u x %u (stride %u) + (%u, %u)",
				             dispatch_order[i],
				             desc.bitbltbuf.desc.DBP * PGS_BLOCK_ALIGNMENT_BYTES,
				             psm_to_str(desc.bitbltbuf.desc.DPSM),
				             desc.trxreg.desc.RRW, desc.trxreg.desc.RRH,
				             desc.bitbltbuf.desc.DBW * PGS_BUFFER_WIDTH_SCALE,
				             desc.trxpos.desc.DSAX, desc.trxpos.desc.DSAY);

				if (desc.host_data_size < desc.host_data_size_required || desc.host_data_size_offset)
				{
					insert_label(cmd, "  Partial %zu / %zu + %zu",
					             desc.host_data_size, desc.host_data_size_required, desc.host_data_size_offset);
				}
			}
			else if (desc.trxdir.desc.XDIR == LOCAL_TO_LOCAL)
			{
				insert_label(cmd, "LocalCopyDst #%u - 0x%x - %s - %u x %u (stride %u) + (%u, %u)",
							 dispatch_order[i],
				             desc.bitbltbuf.desc.DBP * PGS_BLOCK_ALIGNMENT_BYTES,
				             psm_to_str(desc.bitbltbuf.desc.DPSM),
				             desc.trxreg.desc.RRW, desc.trxreg.desc.RRH,
				             desc.bitbltbuf.desc.DBW * PGS_BUFFER_WIDTH_SCALE,
				             desc.trxpos.desc.DSAX, desc.trxpos.desc.DSAY);

				insert_label(cmd, "LocalCopySrc #%u - 0x%x - %s - %u x %u (stride %u) + (%u, %u)",
				             dispatch_order[i],
				             desc.bitbltbuf.desc.SBP * PGS_BLOCK_ALIGNMENT_BYTES,
				             psm_to_str(desc.bitbltbuf.desc.SPSM),
				             desc.trxreg.desc.RRW, desc.trxreg.desc.RRH,
				             desc.bitbltbuf.desc.SBW * PGS_BUFFER_WIDTH_SCALE,
				             desc.trxpos.desc.SSAX, desc.trxpos.desc.SSAY);
			}
		}

		uint32_t wgs_x = (desc.trxreg.desc.RRW + 7) / 8;
		uint32_t wgs_y = (desc.trxreg.desc.RRH + 7) / 8;
		for (uint32_t y = 0; y < wgs_y; y++)
			for (uint32_t x = 0; x < wgs_x; x++)
				push_work(x, y, dispatch_order[i]);
	}

	if (num_wgs)
		cmd.dispatch(num_wgs, 1, 1);
}

void GSRenderer::copy_vram(const CopyDescriptor &desc)
{
	Vulkan::BufferBlockAllocation alloc = {};
	stats.num_copy_threads += desc.trxreg.desc.RRW * desc.trxreg.desc.RRH;
	if (stats.num_copy_threads > MaxPendingCopyThreads)
		flush_transfer();

	if (desc.trxdir.desc.XDIR == HOST_TO_LOCAL)
	{
		ensure_command_buffer(direct_cmd, Vulkan::CommandBuffer::Type::Generic);
		stats.allocated_scratch_memory += desc.host_data_size;
		alloc = direct_cmd->request_scratch_buffer_memory(desc.host_data_size);
		memcpy(alloc.host, desc.host_data, desc.host_data_size);
	}
	pending_copies.push_back({ desc, std::move(alloc) });

	stats.num_copies++;
	check_flush_stats();
}

static const char *reason_to_str(FlushReason reason)
{
	switch (reason)
	{
	case FlushReason::SubmissionFlush:
		return "Submission";
	case FlushReason::TextureHazard:
		return "TextureHazard";
	case FlushReason::CopyHazard:
		return "CopyHazard";
	case FlushReason::Overflow:
		return "Overflow";
	case FlushReason::FBPointer:
		return "FBPointer";
	default:
		return "";
	}
}

#ifdef PARALLEL_GS_DEBUG
static inline void sanitize_state_indices(const RenderPass &rp)
{
	for (uint32_t i = 0; i < rp.num_primitives; i++)
	{
		uint32_t tex_index = rp.prims[i].tex & ((1u << TEX_TEXTURE_INDEX_BITS) - 1u);
		uint32_t state_index = (rp.prims[i].state >> STATE_INDEX_BIT_OFFSET) &
		                       ((1u << STATE_INDEX_BIT_COUNT) - 1u);

		if (state_index >= rp.num_states)
			std::terminate();

		if (tex_index >= 0x8000)
		{
			if (!rp.feedback_texture)
				std::terminate();
		}
		else if ((rp.states[state_index].combiner & COMBINER_TME_BIT) != 0)
		{
			if (tex_index >= rp.num_textures)
				std::terminate();
		}

		if ((rp.states[state_index].combiner & COMBINER_TME_BIT) == 0 && tex_index != 0)
			std::terminate();
	}
}
#endif

void GSRenderer::flush_rendering(const RenderPass &rp, uint32_t instance,
                                 uint32_t base_primitive, uint32_t num_primitives)
{
	auto &cmd = *direct_cmd;
	bind_frame_resources_instanced(rp, instance, num_primitives);
	allocate_scratch_buffers_instanced(cmd, rp, instance, num_primitives);

	if (device->consumes_debug_markers())
	{
		begin_region(*binning_cmd, "Binning %u[%u] [%u, %u)", rp.label_key,
		             instance, base_primitive, base_primitive + num_primitives);
	}
	dispatch_binning(*binning_cmd, rp, instance, base_primitive, num_primitives);
	if (device->consumes_debug_markers())
		binning_cmd->end_region();

	if (device->consumes_debug_markers())
	{
		auto &inst = rp.instances[instance];
		uint32_t depth_psm = inst.fb.z.desc.PSM | (3u << 4);

		begin_region(cmd,
		             "RP %u[%u] - %u prims - %u tex%s - %ux%u + (%u, %u) - %ux%u - FB 0x%x %s - Z 0x%x %s - %s",
		             rp.label_key, instance,
		             num_primitives, rp.num_textures, (rp.feedback_mode != RenderPass::Feedback::None ? " (feedback)" : ""),
		             inst.coarse_tiles_width << rp.coarse_tile_size_log2,
		             inst.coarse_tiles_height << rp.coarse_tile_size_log2,
		             inst.base_x, inst.base_y,
		             1u << rp.coarse_tile_size_log2,
		             1u << rp.coarse_tile_size_log2,
		             inst.fb.frame.desc.FBP * PGS_PAGE_ALIGNMENT_BYTES,
		             psm_to_str(inst.fb.frame.desc.PSM),
		             (inst.z_sensitive ? inst.fb.z.desc.ZBP * PGS_PAGE_ALIGNMENT_BYTES : ~0u),
		             psm_to_str(depth_psm),
		             reason_to_str(rp.flush_reason));

		for (uint32_t i = 0; i < rp.num_states && instance == 0; i++)
		{
			auto &state = rp.states[i];
			begin_region(cmd, "State #%u", i);
			insert_label(cmd, "  TME: %u", (state.combiner & COMBINER_TME_BIT) ? 1 : 0);
			insert_label(cmd, "  TCC: %u", (state.combiner & COMBINER_TCC_BIT) ? 1 : 0);
			insert_label(cmd, "  FOG: %u", (state.combiner & COMBINER_FOG_BIT) ? 1 : 0);
			insert_label(cmd, "  COMBINER: %u",
			             (state.combiner >> COMBINER_MODE_OFFSET) & ((1u << COMBINER_MODE_BITS) - 1));
			insert_label(cmd, "  ABE: %u", (state.blend_mode & BLEND_MODE_ABE_BIT) ? 1 : 0);
			insert_label(cmd, "  DATE: %u", (state.blend_mode & BLEND_MODE_DATE_BIT) ? 1 : 0);
			insert_label(cmd, "  DATM: %u", (state.blend_mode & BLEND_MODE_DATM_BIT) ? 1 : 0);
			insert_label(cmd, "  DTHE: %u", (state.blend_mode & BLEND_MODE_DTHE_BIT) ? 1 : 0);
			insert_label(cmd, "  FBA: %u", (state.blend_mode & BLEND_MODE_FB_ALPHA_BIT) ? 1 : 0);
			insert_label(cmd, "  PABE: %u", (state.blend_mode & BLEND_MODE_PABE_BIT) ? 1 : 0);
			insert_label(cmd, "  ATE: %u", (state.blend_mode & BLEND_MODE_ATE_BIT) ? 1 : 0);
			insert_label(cmd, "  ATEMODE: %u", (state.blend_mode >> BLEND_MODE_ATE_MODE_OFFSET) &
			                                   ((1u << BLEND_MODE_ATE_MODE_BITS) - 1u));
			insert_label(cmd, "  AFAIL: %u", (state.blend_mode >> BLEND_MODE_AFAIL_MODE_OFFSET) &
			                                 ((1u << BLEND_MODE_AFAIL_MODE_BITS) - 1u));
			insert_label(cmd, "  COLCLAMP: %u", (state.blend_mode & BLEND_MODE_COLCLAMP_BIT) ? 1 : 0);
			insert_label(cmd, "  A: %u",
			             (state.blend_mode >> BLEND_MODE_A_MODE_OFFSET) & ((1u << BLEND_MODE_A_MODE_BITS) - 1u));
			insert_label(cmd, "  B: %u",
			             (state.blend_mode >> BLEND_MODE_B_MODE_OFFSET) & ((1u << BLEND_MODE_B_MODE_BITS) - 1u));
			insert_label(cmd, "  C: %u",
			             (state.blend_mode >> BLEND_MODE_C_MODE_OFFSET) & ((1u << BLEND_MODE_C_MODE_BITS) - 1u));
			insert_label(cmd, "  D: %u",
			             (state.blend_mode >> BLEND_MODE_D_MODE_OFFSET) & ((1u << BLEND_MODE_D_MODE_BITS) - 1u));
			insert_label(cmd, "  DIMX: %016llx",
			             (static_cast<unsigned long long>(state.dimx.y) << 32) | state.dimx.x);
			cmd.end_region();
		}
	}

	dispatch_shading(cmd, rp, instance, base_primitive, num_primitives);

	if (device->consumes_debug_markers())
		cmd.end_region();

	work_list_single_sample = {};
	work_list_super_sample = {};

	cmd.set_specialization_constant_mask(0);

	stats.num_render_passes++;
}

static bool page_rect_overlaps(const PageRect &a, const PageRect &b)
{
	// TODO: Ignore page wrapping for now ...
	uint32_t a_start_page = a.base_page;
	uint32_t a_end_page = a.base_page + (a.page_height - 1) * a.page_stride + a.page_width - 1;

	uint32_t b_start_page = b.base_page;
	uint32_t b_end_page = b.base_page + (b.page_height - 1) * b.page_stride + b.page_width - 1;

	uint32_t overlap_lo = std::max<uint32_t>(a_start_page, b_start_page);
	uint32_t overlap_hi = std::min<uint32_t>(a_end_page, b_end_page);

	return overlap_lo <= overlap_hi;
}

static bool instance_has_hazard(const RenderPass::Instance &a, const RenderPass::Instance &b,
                                const PageRect &a_fb_rect, const PageRect &a_z_rect,
                                const PageRect &b_fb_rect, const PageRect &b_z_rect)
{
	if (page_rect_overlaps(a_fb_rect, b_fb_rect))
		return true;
	if (b.z_sensitive && page_rect_overlaps(a_fb_rect, b_z_rect))
		return true;
	if (a.z_sensitive && page_rect_overlaps(b_fb_rect, a_z_rect))
		return true;
	if ((a.z_write || b.z_write) && page_rect_overlaps(a_z_rect, b_z_rect))
		return true;

	return false;
}

static bool compute_instance_hazard_mask(uint8_t (&mask)[MaxRenderPassInstances],
                                         const RenderPass::Instance *instance,
                                         uint32_t num_instances, uint32_t coarse_tile_size_log2)
{
	PageRect fb_rects[MaxRenderPassInstances];
	PageRect z_rects[MaxRenderPassInstances];
	memset(mask, 0, sizeof(mask));
	bool has_hazard = false;

	for (uint32_t i = 0; i < num_instances; i++)
	{
		auto &inst = instance[i];

		fb_rects[i] = compute_page_rect(inst.fb.frame.desc.FBP * PGS_BLOCKS_PER_PAGE,
		                                inst.base_x, inst.base_y, inst.coarse_tiles_width << coarse_tile_size_log2,
		                                inst.coarse_tiles_height << coarse_tile_size_log2, inst.fb.frame.desc.FBW,
		                                inst.fb.frame.desc.PSM);

		if (inst.z_sensitive)
		{
			z_rects[i] = compute_page_rect(inst.fb.z.desc.ZBP * PGS_BLOCKS_PER_PAGE,
			                               inst.base_x, inst.base_y, inst.coarse_tiles_width << coarse_tile_size_log2,
			                               inst.coarse_tiles_height << coarse_tile_size_log2, inst.fb.frame.desc.FBW,
			                               inst.fb.z.desc.PSM);
		}
	}

	for (uint32_t i = 0; i < num_instances; i++)
	{
		for (uint32_t j = i + 1; j < num_instances; j++)
		{
			if (instance_has_hazard(instance[i], instance[j], fb_rects[i], z_rects[i], fb_rects[j], z_rects[j]))
			{
				mask[i] |= 1u << j;
				mask[j] |= 1u << i;
				has_hazard = true;
			}
		}
	}

	return has_hazard;
}

void GSRenderer::flush_rendering(const RenderPass &rp)
{
	if (rp.num_primitives == 0)
		return;
	assert(rp.num_primitives <= MaxPrimitivesPerFlush);

#ifdef PARALLEL_GS_DEBUG
	sanitize_state_indices(rp);
#endif

	// Attempted async compute here for binning, etc, but it's not very useful in practice.
	bool first_clear_cmd = !clear_cmd;
	ensure_command_buffer(clear_cmd, Vulkan::CommandBuffer::Type::Generic);
	if (first_clear_cmd)
		clear_cmd->begin_region("clear-memory");

	ensure_command_buffer(triangle_setup_cmd, Vulkan::CommandBuffer::Type::Generic);
	bool needs_single_sample_heuristic = false;
	for (uint32_t i = 0; i < rp.num_instances; i++)
	{
		if (rp.instances[i].sampling_rate_y_log2 != 0)
		{
			ensure_command_buffer(heuristic_cmd, Vulkan::CommandBuffer::Type::Generic);
			needs_single_sample_heuristic = true;
			break;
		}
	}
	ensure_command_buffer(binning_cmd, Vulkan::CommandBuffer::Type::Generic);
	ensure_command_buffer(direct_cmd, Vulkan::CommandBuffer::Type::Generic);
	auto &cmd = *direct_cmd;

	bind_textures(cmd, rp);
	bind_frame_resources(rp);
	allocate_scratch_buffers(cmd, rp);

	if (device->consumes_debug_markers())
		begin_region(*triangle_setup_cmd, "TriangleSetup %u", rp.label_key);
	dispatch_triangle_setup(*triangle_setup_cmd, rp);
	if (device->consumes_debug_markers())
		triangle_setup_cmd->end_region();

	if (needs_single_sample_heuristic)
	{
		if (device->consumes_debug_markers())
			begin_region(*heuristic_cmd, "Heuristic %u", rp.label_key);
		dispatch_single_sample_heuristic(*heuristic_cmd, rp);
		if (device->consumes_debug_markers())
			heuristic_cmd->end_region();
	}
	indirect_single_sample_heuristic = {};

	// Flush rendering work.
	{
		Vulkan::QueryPoolHandle start_ts, end_ts;
		if (enable_timestamps)
			start_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		if (rp.num_instances == 1)
		{
			flush_rendering(rp, 0, 0, rp.num_primitives);
		}
		else
		{
			uint8_t hazard_mask[MaxRenderPassInstances];
			bool has_hazards = compute_instance_hazard_mask(
					hazard_mask, rp.instances, rp.num_instances, rp.coarse_tile_size_log2);

			if (has_hazards)
			{
				uint32_t active_instances = 0;
				uint32_t active_hazards = 0;
				uint32_t base_primitive = 0;

				for (uint32_t prim = 0; prim < rp.num_primitives; prim++)
				{
					uint32_t instance = (rp.prims[prim].state >> STATE_VERTEX_RENDER_PASS_INSTANCE_OFFSET) &
					                    ((1u << STATE_VERTEX_RENDER_PASS_INSTANCE_COUNT) - 1u);

					if (active_hazards & (1u << instance))
					{
						// We have a hazard. Need to flush now. These can run in parallel.
						Util::for_each_bit(active_instances, [&](uint32_t bit)
						{
							flush_rendering(rp, bit, base_primitive, prim - base_primitive);
						});

						cmd.barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
						            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
						            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

						base_primitive = prim;
						active_instances = 0;
						active_hazards = 0;
					}

					active_instances |= 1u << instance;
					active_hazards |= hazard_mask[instance];
				}

				// Flush the rest now.
				Util::for_each_bit(active_instances, [&](uint32_t bit)
				{
					flush_rendering(rp, bit, base_primitive, rp.num_primitives - base_primitive);
				});
			}
			else
			{
				for (uint32_t i = 0; i < rp.num_instances; i++)
					flush_rendering(rp, i, 0, rp.num_primitives);
			}
		}

		cmd.barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		            VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
		            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
		            VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
		            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

		if (enable_timestamps)
		{
			end_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
			timestamps.push_back({TimestampType::Shading, std::move(start_ts), std::move(end_ts)});
		}

		stats.num_primitives += rp.num_primitives;
		check_flush_stats();
	}

	move_image_handles_to_slab();
}

Vulkan::ImageHandle GSRenderer::pull_image_handle_from_slab(uint32_t width, uint32_t height, uint32_t levels)
{
	if (!Util::is_pow2(width) || !Util::is_pow2(height) || levels > 1 || width > 1024 || height > 1024)
		return {};

	uint32_t W = Util::floor_log2(width);
	uint32_t H = Util::floor_log2(height);
	auto &pool = recycled_image_pool[H][W];
	if (pool.empty())
		return {};

	auto res = std::move(pool.back());
	pool.pop_back();
	assert(total_image_slab_size >= (sizeof(uint32_t) << (W + H)));
	total_image_slab_size -= sizeof(uint32_t) << (W + H);
	return res;
}

void GSRenderer::move_image_handles_to_slab()
{
	for (auto &handle : recycled_image_handles)
	{
		uint32_t W = Util::floor_log2(handle->get_width());
		uint32_t H = Util::floor_log2(handle->get_height());
		assert(W <= 10 && H <= 10 && handle->get_create_info().levels == 1);
		total_image_slab_size += sizeof(uint32_t) << (W + H);
		recycled_image_pool[H][W].push_back(std::move(handle));
	}
	recycled_image_handles.clear();

	if (total_image_slab_size > image_slab_high_water_mark)
	{
		LOGW("New high watermark for image slab: %llu MiB.\n",
		     static_cast<unsigned long long>(total_image_slab_size / (1024 * 1024)));
		image_slab_high_water_mark = total_image_slab_size;
	}

	// If we end up exhausting this pool, just flush everything and start over, no need to be more clever about it.
	// Shouldn't happen in normal use.
	if (total_image_slab_size > max_image_slab_size)
	{
		for (auto &y : recycled_image_pool)
			for (auto &x : y)
				x.clear();
		total_image_slab_size = 0;
		LOGW("Image slab pool was exhausted, flushing it ...\n");
	}
}

uint32_t GSRenderer::update_palette_cache(const PaletteUploadDescriptor &desc)
{
	next_clut_instance = (next_clut_instance + 1) % CLUTInstances;
	palette_uploads.push_back(desc);
	stats.num_palette_updates++;
	check_flush_stats();
	return next_clut_instance;
}

void GSRenderer::upload_texture(const TextureDescriptor &desc, const Vulkan::Image &img)
{
	auto &cmd = *direct_cmd;

	uint32_t levels = img.get_create_info().levels;
	cmd.set_program(shaders.upload);
	cmd.set_storage_buffer(0, 0, *buffers.gpu);
	cmd.set_storage_buffer(0, BINDING_CLUT, *buffers.clut);

	struct UploadInfo
	{
		uint32_t off_x, off_y;
		uint32_t width, height;
		uint32_t addr_block;
		uint32_t stride_block;
		uint32_t clut_offset;
		uint32_t aem;
		uint32_t ta0;
		uint32_t ta1;
		uint32_t instance;
		uint32_t umsk, ufix, vmsk, vfix;
	};

	UploadInfo info = {};

	info.off_x = desc.rect.x;
	info.off_y = desc.rect.y;
	info.clut_offset = uint32_t(desc.tex0.desc.CSA);
	info.aem = uint32_t(desc.texa.desc.AEM);
	info.ta0 = uint32_t(desc.texa.desc.TA0);
	info.ta1 = uint32_t(desc.texa.desc.TA1);
	info.instance = desc.palette_bank;
	info.umsk = UINT32_MAX;
	info.vmsk = UINT32_MAX;

	bool s_region_repeat = uint32_t(desc.clamp.desc.WMS) == CLAMPBits::REGION_REPEAT;
	bool t_region_repeat = uint32_t(desc.clamp.desc.WMT) == CLAMPBits::REGION_REPEAT;

	if (s_region_repeat)
	{
		info.umsk = uint32_t(desc.clamp.desc.MINU);
		info.ufix = uint32_t(desc.clamp.desc.MAXU);
	}

	if (t_region_repeat)
	{
		info.vmsk = uint32_t(desc.clamp.desc.MINV);
		info.vfix = uint32_t(desc.clamp.desc.MAXV);
	}

	const struct
	{
		uint32_t addr, stride;
	} table[] = {
		{ uint32_t(desc.tex0.desc.TBP0), uint32_t(desc.tex0.desc.TBW) },
		{ uint32_t(desc.miptbp1_3.desc.TBP1), uint32_t(desc.miptbp1_3.desc.TBW1) },
		{ uint32_t(desc.miptbp1_3.desc.TBP2), uint32_t(desc.miptbp1_3.desc.TBW2) },
		{ uint32_t(desc.miptbp1_3.desc.TBP3), uint32_t(desc.miptbp1_3.desc.TBW3) },
		{ uint32_t(desc.miptbp4_6.desc.TBP1), uint32_t(desc.miptbp4_6.desc.TBW1) },
		{ uint32_t(desc.miptbp4_6.desc.TBP2), uint32_t(desc.miptbp4_6.desc.TBW2) },
		{ uint32_t(desc.miptbp4_6.desc.TBP3), uint32_t(desc.miptbp4_6.desc.TBW3) },
	};

	cmd.set_specialization_constant_mask(0x7);
	cmd.set_specialization_constant(0, uint32_t(desc.tex0.desc.PSM));
	cmd.set_specialization_constant(1, vram_size - 1);
	cmd.set_specialization_constant(2, uint32_t(desc.tex0.desc.CPSM));

	for (uint32_t level = 0; level < levels; level++)
	{
		info.addr_block = table[level].addr;
		info.stride_block = table[level].stride;
		info.width = img.get_width(level);
		info.height = img.get_height(level);

		if (device->consumes_debug_markers())
		{
			insert_label(cmd,
			             "Cache mip %u - 0x%x - %s - %u x %u (stride %u) + (%u, %u) - CPSM %s - CSA %u - bank %u / %u - %016llx",
			             level, info.addr_block * PGS_BLOCK_ALIGNMENT_BYTES,
			             psm_to_str(uint32_t(desc.tex0.desc.PSM)),
			             info.width, info.height, info.stride_block * PGS_BUFFER_WIDTH_SCALE,
			             info.off_x, info.off_y,
			             psm_to_str(uint32_t(desc.tex0.desc.CPSM)),
			             uint32_t(desc.tex0.desc.CSA),
			             desc.palette_bank, desc.latest_palette_bank,
			             static_cast<unsigned long long>(desc.hash));

			insert_label(cmd, "  AEM: %u, TA0: 0x%x, TA1: 0x%x", info.aem, info.ta0, info.ta1);
		}

		if (level == 0 && levels == 1)
		{
			// Common case by far. Mip-mapping seems to be rare.
			cmd.set_storage_texture(0, 1, img.get_view());
		}
		else
		{
			Vulkan::ImageViewCreateInfo view_info = {};
			view_info.image = &img;
			view_info.levels = 1;
			view_info.layers = 1;
			view_info.view_type = VK_IMAGE_VIEW_TYPE_2D;
			view_info.format = img.get_format();
			view_info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
			view_info.base_level = level;

			auto view = device->create_image_view(view_info);
			cmd.set_storage_texture(0, 1, *view);
		}

		cmd.push_constants(&info, 0, sizeof(info));
		cmd.dispatch((info.width + 7) / 8, (info.height + 7) / 8, 1);

		if (levels > 1)
		{
			if (s_region_repeat)
			{
				info.ufix >>= 1;
				info.umsk >>= 1;
			}

			if (t_region_repeat)
			{
				info.vfix >>= 1;
				info.vmsk >>= 1;
			}

			info.off_x >>= 1;
			info.off_y >>= 1;
		}
	}

	cmd.set_specialization_constant_mask(0);
}

void GSRenderer::init_luts()
{
	static const uint8_t rcp_tab[256] =
	{
		0xff, 0xfd, 0xfb, 0xf9, 0xf7, 0xf5, 0xf3, 0xf1,
		0xf0, 0xee, 0xec, 0xea, 0xe8, 0xe6, 0xe5, 0xe3,
		0xe1, 0xdf, 0xdd, 0xdc, 0xda, 0xd8, 0xd7, 0xd5,
		0xd3, 0xd2, 0xd0, 0xce, 0xcd, 0xcb, 0xc9, 0xc8,
		0xc6, 0xc5, 0xc3, 0xc2, 0xc0, 0xbf, 0xbd, 0xbc,
		0xba, 0xb9, 0xb7, 0xb6, 0xb4, 0xb3, 0xb1, 0xb0,
		0xae, 0xad, 0xac, 0xaa, 0xa9, 0xa7, 0xa6, 0xa5,
		0xa4, 0xa2, 0xa1, 0x9f, 0x9e, 0x9d, 0x9c, 0x9a,
		0x99, 0x98, 0x96, 0x95, 0x94, 0x93, 0x91, 0x90,
		0x8f, 0x8e, 0x8c, 0x8b, 0x8a, 0x89, 0x88, 0x87,
		0x86, 0x84, 0x83, 0x82, 0x81, 0x80, 0x7f, 0x7e,
		0x7c, 0x7b, 0x7a, 0x79, 0x78, 0x77, 0x76, 0x74,
		0x74, 0x73, 0x71, 0x71, 0x70, 0x6f, 0x6e, 0x6d,
		0x6b, 0x6b, 0x6a, 0x68, 0x67, 0x67, 0x66, 0x65,
		0x64, 0x63, 0x62, 0x61, 0x60, 0x5f, 0x5e, 0x5d,
		0x5c, 0x5b, 0x5b, 0x59, 0x58, 0x58, 0x56, 0x56,
		0x55, 0x54, 0x53, 0x52, 0x51, 0x51, 0x50, 0x4f,
		0x4e, 0x4e, 0x4c, 0x4b, 0x4b, 0x4a, 0x48, 0x48,
		0x48, 0x46, 0x46, 0x45, 0x44, 0x43, 0x43, 0x42,
		0x41, 0x40, 0x3f, 0x3f, 0x3e, 0x3d, 0x3c, 0x3b,
		0x3b, 0x3a, 0x39, 0x38, 0x38, 0x37, 0x36, 0x36,
		0x35, 0x34, 0x34, 0x33, 0x32, 0x31, 0x30, 0x30,
		0x2f, 0x2e, 0x2e, 0x2d, 0x2d, 0x2c, 0x2b, 0x2a,
		0x2a, 0x29, 0x28, 0x27, 0x27, 0x26, 0x26, 0x25,
		0x24, 0x23, 0x23, 0x22, 0x21, 0x21, 0x21, 0x20,
		0x1f, 0x1f, 0x1e, 0x1d, 0x1d, 0x1c, 0x1c, 0x1b,
		0x1a, 0x19, 0x19, 0x19, 0x18, 0x17, 0x17, 0x16,
		0x16, 0x15, 0x14, 0x13, 0x13, 0x12, 0x12, 0x11,
		0x11, 0x10, 0x0f, 0x0f, 0x0e, 0x0e, 0x0e, 0x0d,
		0x0c, 0x0c, 0x0b, 0x0b, 0x0a, 0x0a, 0x09, 0x08,
		0x08, 0x07, 0x07, 0x07, 0x06, 0x05, 0x05, 0x04,
		0x04, 0x03, 0x03, 0x02, 0x02, 0x01, 0x01, 0x01
	};

	float lut[256];
	for (int i = 0; i < 256; i++)
		lut[i] = 1.0f / (1.0f + (float(i) + 0.5f) / 256.0f);

	Vulkan::BufferCreateInfo buf_info = {};
	buf_info.size = sizeof(rcp_tab);
	buf_info.domain = Vulkan::BufferDomain::Device;
	buf_info.usage = VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
	buffers.fixed_rcp_lut = device->create_buffer(buf_info, rcp_tab);

	Vulkan::BufferViewCreateInfo view_info = {};
	view_info.offset = 0;
	view_info.range = VK_WHOLE_SIZE;
	view_info.format = VK_FORMAT_R8_UINT;
	view_info.buffer = buffers.fixed_rcp_lut.get();
	buffers.fixed_rcp_lut_view = device->create_buffer_view(view_info);

	buf_info.size = sizeof(lut);
	buffers.float_rcp_lut = device->create_buffer(buf_info, lut);
	view_info.format = VK_FORMAT_R32_SFLOAT;
	view_info.buffer = buffers.float_rcp_lut.get();
	buffers.float_rcp_lut_view = device->create_buffer_view(view_info);
}

void GSRenderer::flush_palette_upload()
{
	if (palette_uploads.empty())
		return;

	auto &cmd = *direct_cmd;
	cmd.set_program(shaders.clut_write);
	cmd.set_specialization_constant_mask(1);
	cmd.set_specialization_constant(0, vram_size - 1);

	cmd.set_storage_buffer(0, 0, *buffers.gpu);
	cmd.set_storage_buffer(0, BINDING_CLUT, *buffers.clut);
	auto *desc = cmd.allocate_typed_constant_data<CLUTDescriptor>(0, 1, palette_uploads.size());

	CLUTDescriptor clut_desc = {};

	uint32_t instance = (base_clut_instance + 1) % CLUTInstances;

	for (auto &upload : palette_uploads)
	{
		clut_desc.tex_format = uint32_t(upload.tex0.desc.PSM);
		clut_desc.format = uint32_t(upload.tex0.desc.CPSM);
		clut_desc.base_pointer = uint32_t(upload.tex0.desc.CBP);
		clut_desc.instance = instance;
		clut_desc.co_uv = uint32_t(upload.texclut.desc.COU) | (uint32_t(upload.texclut.desc.COV) << 16);
		clut_desc.cbw = uint32_t(upload.texclut.desc.CBW);
		clut_desc.csa = uint32_t(upload.tex0.desc.CSA);
		clut_desc.csm = uint32_t(upload.tex0.desc.CSM);

		if (!clut_desc.csm)
		{
			// CSM1
			clut_desc.co_uv = 0;
			clut_desc.cbw = 0;
		}
		else
		{
			// CSM2
			if (clut_desc.csa != 0)
				LOGW("CSM2: CSA is not 0.\n");
		}

		instance = (instance + 1) % CLUTInstances;
		*desc++ = clut_desc;
	}

	struct Push
	{
		uint32_t count;
		uint32_t read_index;
	};

	Push push = {};
	push.count = palette_uploads.size();
	push.read_index = base_clut_instance;
	cmd.push_constants(&push, 0, sizeof(push));

	cmd.begin_region("flush-palette-upload");

	if (device->consumes_debug_markers())
	{
		for (size_t i = 0, n = palette_uploads.size(); i < n; i++)
		{
			auto &upload = palette_uploads[i];
			insert_label(cmd, "Bank %u - 0x%x - %s - %u colors - CSA %u - CSM %u - COU/V %u, %u",
			             uint32_t(base_clut_instance + 1 + i) % CLUTInstances,
			             uint32_t(upload.tex0.desc.CBP) * PGS_BLOCK_ALIGNMENT_BYTES,
			             psm_to_str(uint32_t(upload.tex0.desc.CPSM)),
			             ((uint32_t(upload.tex0.desc.PSM) == PSMT8 ||
			               uint32_t(upload.tex0.desc.PSM) == PSMT8H) ? 256 : 16),
			             uint32_t(upload.tex0.desc.CSA),
			             uint32_t(upload.tex0.desc.CSM),
			             upload.texclut.desc.COU * 16,
			             upload.texclut.desc.COV);
		}
	}

	Vulkan::QueryPoolHandle start_ts, end_ts;
	if (enable_timestamps)
		start_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	// One workgroup iterates through every CLUT updates and snapshots current palette memory state to a 1 KiB slice.
	// The palette is large enough to hold 32-bit with 256 colors.
	cmd.dispatch(1, 1, 1);
	if (enable_timestamps)
	{
		end_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		timestamps.push_back({ TimestampType::PaletteUpdate, std::move(start_ts), std::move(end_ts) });
	}

	cmd.barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
	cmd.end_region();
	cmd.set_specialization_constant_mask(0);

	base_clut_instance = next_clut_instance;
	palette_uploads.clear();
}

void GSRenderer::flush_cache_upload()
{
	ensure_command_buffer(direct_cmd, Vulkan::CommandBuffer::Type::Generic);
	flush_palette_upload();
	if (texture_uploads.empty())
		return;

	auto &cmd = *direct_cmd;

	VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
	dep.imageMemoryBarrierCount = pre_image_barriers.size();
	dep.pImageMemoryBarriers = pre_image_barriers.data();
	cmd.barrier(dep);

	cmd.begin_region("cache-upload");
	Vulkan::QueryPoolHandle start_ts, end_ts;
	if (enable_timestamps)
		start_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	// TODO: We could potentially sort this based on shader key to avoid some context rolls, but eeeeeh.
	for (auto &upload : texture_uploads)
		upload_texture(upload.desc, *upload.image);

	if (enable_timestamps)
	{
		end_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		timestamps.push_back({ TimestampType::TextureUpload, std::move(start_ts), std::move(end_ts) });
	}
	cmd.end_region();

	dep.imageMemoryBarrierCount = post_image_barriers.size();
	dep.pImageMemoryBarriers = post_image_barriers.data();
	cmd.barrier(dep);

	texture_uploads.clear();
	pre_image_barriers.clear();
	post_image_barriers.clear();
}

void GSRenderer::mark_copy_write_page(uint32_t page_index)
{
	vram_copy_write_pages[page_index / 32u] |= 1u << (page_index & 31u);
}

void GSRenderer::mark_shadow_page_sync(uint32_t page_index)
{
	sync_vram_shadow_pages[page_index / 32u] |= 1u << (page_index & 31u);
}

void GSRenderer::flush_transfer()
{
	if (pending_copies.empty())
		return;

	ensure_command_buffer(direct_cmd, Vulkan::CommandBuffer::Type::Generic);
	auto &cmd = *direct_cmd;

	cmd.begin_region("vram-copy");

	Vulkan::QueryPoolHandle start_ts, end_ts;
	if (enable_timestamps)
		start_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	cmd.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
	            VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
				VK_ACCESS_TRANSFER_WRITE_BIT);

	// Prepare atomic buffer for a new linked list setup.
	{
		RangeMerger merger;
		const auto flush_range = [&](VkDeviceSize offset, VkDeviceSize range)
		{
			cmd.fill_buffer(*buffers.vram_copy_atomics, UINT32_MAX, offset + PGS_LINKED_VRAM_COPY_WRITE_LIST_OFFSET, range);
		};

		cmd.begin_region("reset-vram-copy-state");
		// Reset atomic counter.
		cmd.fill_buffer(*buffers.vram_copy_atomics, 0, 0, PGS_LINKED_VRAM_COPY_WRITE_LIST_OFFSET);
		// Reset hazard exist bitfield.
		cmd.fill_buffer(*buffers.vram_copy_atomics, 0, PGS_LINKED_VRAM_COPY_WRITE_LIST_OFFSET + vram_size, vram_size / 32);

		for (size_t i = 0, n = vram_copy_write_pages.size(); i < n; i++)
		{
			Util::for_each_bit(vram_copy_write_pages[i], [&](uint32_t bit) {
				merger.push((i * 32 + bit) * PageSize, PageSize, flush_range);
			});
			vram_copy_write_pages[i] = 0;
		}

		merger.flush(flush_range);
		cmd.end_region();
	}

	// Upper half of VRAM is reserved for read-only caching.
	{
		VkDeviceSize read_only_offset = buffers.gpu->get_create_info().size / 2;
		RangeMerger merger;

		const auto flush_range = [&](VkDeviceSize offset, VkDeviceSize range)
		{
			cmd.copy_buffer(*buffers.gpu, offset + read_only_offset, *buffers.gpu, offset, range);
		};

		cmd.begin_region("shadow-vram-cache");
		for (size_t i = 0, n = sync_vram_shadow_pages.size(); i < n; i++)
		{
			Util::for_each_bit(sync_vram_shadow_pages[i], [i, &merger, &flush_range](uint32_t bit) {
				merger.push((i * 32 + bit) * PageSize, PageSize, flush_range);
			});
			sync_vram_shadow_pages[i] = 0;
		}

		merger.flush(flush_range);
		cmd.end_region();
	}

	cmd.barrier(VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
	            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

	auto *ubo = cmd.allocate_typed_constant_data<TransferDescriptor>(0, 3, pending_copies.size());
	uint32_t copy_index = 0;
	for (auto &copy : pending_copies)
	{
		auto &desc = copy.copy;
		ubo->width = desc.trxreg.desc.RRW;
		ubo->height = desc.trxreg.desc.RRH;
		ubo->dest_x = desc.trxpos.desc.DSAX;
		ubo->dest_y = desc.trxpos.desc.DSAY;
		ubo->source_x = desc.trxpos.desc.SSAX;
		ubo->source_y = desc.trxpos.desc.SSAY;
		ubo->source_addr = desc.bitbltbuf.desc.SBP;
		ubo->dest_addr = desc.bitbltbuf.desc.DBP;
		ubo->source_stride = desc.bitbltbuf.desc.SBW;
		ubo->dest_stride = desc.bitbltbuf.desc.DBW;
		ubo->host_offset_qwords = desc.host_data_size_offset / sizeof(uint64_t);
		ubo->dispatch_order = copy_index++;

		// Use BDA to get bindless buffers.
		// We handle robustness anyway, so this is fine.
		if (desc.trxdir.desc.XDIR == HOST_TO_LOCAL)
		{
			ubo->source_bda = copy.alloc.buffer->get_device_address() + copy.alloc.offset;
			ubo->source_size = desc.host_data_size;
		}
		else
		{
			VkDeviceSize offset = copy.copy.needs_shadow_vram ? buffers.gpu->get_create_info().size / 2 : 0;
			ubo->source_bda = buffers.gpu->get_device_address() + offset;
			ubo->source_size = UINT32_MAX;
		}

		ubo->padding = 0;
		ubo++;
	}

	// Sort and batch copy work that uses same shader.
	uint32_t dispatch_order[MaxPendingCopiesWithoutFlush];
	struct
	{
		uint32_t offset;
		uint32_t range;
	} dispatches[MaxPendingCopiesWithoutFlush];
	uint32_t num_dispatches = 1;

	for (size_t i = 0, n = pending_copies.size(); i < n; i++)
		dispatch_order[i] = i;

	std::sort(dispatch_order, dispatch_order + pending_copies.size(), [&](uint32_t a, uint32_t b) {
		return copy_pipeline_key(pending_copies[a].copy) < copy_pipeline_key(pending_copies[b].copy);
	});

	uint32_t current_key = copy_pipeline_key(pending_copies[dispatch_order[0]].copy);
	dispatches[0] = { 0, 1 };

	for (size_t i = 1, n = pending_copies.size(); i < n; i++)
	{
		uint32_t next_key = copy_pipeline_key(pending_copies[dispatch_order[i]].copy);
		if (next_key == current_key)
		{
			dispatches[num_dispatches - 1].range++;
		}
		else
		{
			dispatches[num_dispatches].offset = i;
			dispatches[num_dispatches].range = 1;
			num_dispatches++;
			current_key = next_key;
		}
	}

	// Sort copies by pipeline key.

	cmd.begin_region("prepare-copy");
	for (uint32_t i = 0; i < num_dispatches; i++)
		emit_copy_vram(cmd, dispatch_order + dispatches[i].offset, dispatches[i].range, true);
	cmd.end_region();

	cmd.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

	cmd.begin_region("actual-copy");
	for (uint32_t i = 0; i < num_dispatches; i++)
		emit_copy_vram(cmd, dispatch_order + dispatches[i].offset, dispatches[i].range, false);
	cmd.end_region();

	if (enable_timestamps)
	{
		end_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		timestamps.push_back({ TimestampType::CopyVRAM, std::move(start_ts), std::move(end_ts) });
	}

	pending_copies.clear();

	cmd.barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
	            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
	            VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);

	cmd.end_region();
}

void GSRenderer::transfer_overlap_barrier()
{
	flush_transfer();
	stats.num_copy_barriers++;
}

void GSRenderer::sample_crtc_circuit(Vulkan::CommandBuffer &cmd, const Vulkan::Image &img, const DISPFBBits &dispfb,
                                     const SamplingRect &rect)
{
	cmd.image_barrier(img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	                  0, 0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

	Vulkan::RenderPassInfo rp_info;
	rp_info.num_color_attachments = 1;
	rp_info.color_attachments[0] = &img.get_view();
	rp_info.store_attachments = 1u << 0;
	if (memcmp(&rect.valid_extent, &rect.image_extent, sizeof(VkExtent2D)) != 0)
		rp_info.clear_attachments = 1u << 0;
	cmd.begin_render_pass(rp_info);

	cmd.set_opaque_sprite_state();
	cmd.set_storage_buffer(0, 0, *buffers.gpu);
	cmd.set_program(sample_quad);

	cmd.set_scissor({{ 0, 0 }, rect.valid_extent });

	cmd.set_specialization_constant_mask(0x3);
	cmd.set_specialization_constant(0, uint32_t(dispfb.PSM));
	cmd.set_specialization_constant(1, vram_size - 1);

	struct Registers
	{
		uint32_t fbp;
		uint32_t fbw;
		uint32_t dbx;
		uint32_t dby;
		uint32_t phase;
		uint32_t phase_stride;
	} push = {};

	push.fbp = uint32_t(dispfb.FBP);
	push.fbw = uint32_t(dispfb.FBW);
	push.dbx = uint32_t(dispfb.DBX);
	push.dby = uint32_t(dispfb.DBY);
	push.phase = rect.phase_offset;
	push.phase_stride = rect.phase_stride;
	cmd.push_constants(&push, 0, sizeof(push));

	cmd.draw(3);

	cmd.end_render_pass();
}

GSRenderer::SamplingRect GSRenderer::compute_circuit_rect(const PrivRegisterState &priv, uint32_t phase,
                                                          const DISPLAYBits &display, bool force_progressive)
{
	SamplingRect rect = {};

	bool is_interlaced = priv.smode2.INT;
	bool alternative_sampling = is_interlaced && !priv.smode2.FFMD;

	if (alternative_sampling && force_progressive)
		is_interlaced = false;

	uint32_t DW = display.DW + 1;
	uint32_t DH = display.DH + 1;
	uint32_t MAGH = display.MAGH + 1;
	uint32_t MAGV = display.MAGV + 1;

	rect.image_extent.width = DW / MAGH;
	rect.image_extent.height = DH / MAGV;

	rect.valid_extent.width = rect.image_extent.width;

	if (is_interlaced)
	{
		// No idea if this makes sense. Seems to create same effect as using DY 0/1 in EN1 and EN2.
		// In FFMD mode, just sample the field as-is.
		rect.phase_offset = alternative_sampling ? (display.DY & 1) : 0;
		rect.phase_stride = alternative_sampling ? 2 : 1;
		rect.phase_offset += phase;
	}
	else
	{
		// Read as-is.
		rect.phase_offset = 0;
		rect.phase_stride = 1;
	}

	if (is_interlaced)
	{
		// Half-height render. Either we read every other line, or every line, but half height (FFMD).
		// Avoid sampling beyond what the game expects us to sample.
		rect.valid_extent.height = (rect.image_extent.height - phase - rect.phase_offset + 1) >> 1;
		rect.image_extent.height = (rect.image_extent.height + 1) >> 1;
	}
	else
	{
		rect.valid_extent.height = rect.image_extent.height;
		// Round up to nearest even number since some games have an odd number of valid lines here.
		rect.image_extent.height = (rect.image_extent.height + 1) & ~1u;
	}

	return rect;
}

ScanoutResult GSRenderer::vsync(const PrivRegisterState &priv, const VSyncInfo &info)
{
	if (!device)
		return {};

	// Assume that all pending operations have been flushed.
	ensure_command_buffer(direct_cmd, Vulkan::CommandBuffer::Type::Generic);
	auto &cmd = *direct_cmd;

	cmd.begin_region("vsync");

	auto image_info = Vulkan::ImageCreateInfo::immutable_2d_image(1, 1, VK_FORMAT_R8G8B8A8_UNORM);
	image_info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
	image_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	image_info.misc |= Vulkan::IMAGE_MISC_MUTABLE_SRGB_BIT;

	Vulkan::ImageHandle circuit1, circuit2;

	const bool force_progressive = info.force_progressive;
	// True to include the overscan area.
	const bool overscan = info.overscan;
	// Tries to counteract field blending. It's just a blur that is overkill.
	const bool anti_blur = info.anti_blur;

	bool is_interlaced = priv.smode2.INT;
	bool alternative_sampling = is_interlaced && !priv.smode2.FFMD;
	bool force_deinterlace = priv.smode2.FFMD && priv.smode1.CMOD != SMODE1Bits::CMOD_PROGRESSIVE;
	if (alternative_sampling && force_progressive)
		is_interlaced = false;

	uint32_t phase = 0;
	uint32_t clock_divider = 1;
	int scan_offset_x = 0;
	int scan_offset_y = 0;
	uint32_t mode_width = 0;
	uint32_t mode_height = 0;

	if ((priv.smode1.CMOD == SMODE1Bits::CMOD_NTSC ||
	     (priv.smode1.CMOD == SMODE1Bits::CMOD_PROGRESSIVE && !priv.smode2.INT)) &&
	    priv.smode1.LC == SMODE1Bits::LC_ANALOG)
	{
		if (overscan)
		{
			mode_width = 712;
			mode_height = 240;
			scan_offset_x = 123;
			scan_offset_y = 17;
		}
		else
		{
			mode_width = 640;
			mode_height = 224;
			scan_offset_x = 159;
			scan_offset_y = 25;
		}

		if (!is_interlaced && !force_deinterlace)
		{
			mode_height *= 2;
			scan_offset_y *= 2;
		}

		clock_divider = priv.smode1.CMOD == SMODE1Bits::CMOD_PROGRESSIVE ?
		                SMODE1Bits::CLOCK_DIVIDER_COMPONENT : SMODE1Bits::CLOCK_DIVIDER_COMPOSITE;
		insert_label(cmd, "NTSC, field %u", info.phase);
		if (priv.smode1.CMOD == SMODE1Bits::CMOD_PROGRESSIVE)
			insert_label(cmd, "Progressive scan", info.phase);
	}
	else if (priv.smode1.CMOD == SMODE1Bits::CMOD_PAL && priv.smode1.LC == SMODE1Bits::LC_ANALOG)
	{
		// TODO: Does PAL output support progressive scan? I seem to recall PAL PS2s would output NTSC progressive
		// back in the day, but most TVs supported it ...
		if (overscan)
		{
			mode_width = 712;
			mode_height = 288;
			scan_offset_x = 133;
			scan_offset_y = 21;
		}
		else
		{
			mode_width = 640;
			mode_height = 256;
			scan_offset_x = 169;
			scan_offset_y = 36;
		}

		if (!is_interlaced && !force_deinterlace)
		{
			mode_height *= 2;
			scan_offset_y *= 2;
		}

		clock_divider = SMODE1Bits::CLOCK_DIVIDER_COMPOSITE;
		insert_label(cmd, "PAL, field %u", info.phase);
	}
	else if (priv.smode1.CMOD == SMODE1Bits::CMOD_PROGRESSIVE && priv.smode1.LC == SMODE1Bits::LC_HDTV)
	{
		if (priv.smode2.INT)
		{
			mode_width = 1920;
			mode_height = 540;
			// These numbers are probably very wrong.
			// Just fiddled with it until it looked kinda ok.
			scan_offset_x = 236;
			scan_offset_y = 0;
			insert_label(cmd, "HDTV 1080i, field %u", info.phase);
		}
		else
		{
			mode_width = 1280;
			mode_height = 720;
			// These numbers are probably very wrong.
			// Just fiddled with it until it looked kinda ok.
			scan_offset_x = 300;
			scan_offset_y = 0;
			insert_label(cmd, "HDTV 720p", info.phase);
		}

		if (!is_interlaced && !force_deinterlace)
		{
			mode_height *= 2;
			scan_offset_y *= 2;
		}

		clock_divider = SMODE1Bits::CLOCK_DIVIDER_HDTV;
	}
	else
	{
		LOGE("Unknown video format.\n");
		cmd.end_region();
		flush_submit(0);
		return {};
	}

	if (device->consumes_debug_markers())
	{
		insert_label(cmd,
		             "PMODE: SLBG %u - ALP %u - MMOD %u - CRTMD %u - AMOD %u",
		             priv.pmode.SLBG,
		             priv.pmode.ALP,
		             priv.pmode.MMOD,
		             priv.pmode.CRTMD,
		             priv.pmode.AMOD);

		insert_label(cmd, "SMODE1:");

		// This is mostly just noise. Never seen this matter.
#if 0
		insert_label(cmd,
		             "  CLKSEL %u, RC %u, LC %u",
		             priv.smode1.CLKSEL, priv.smode1.RC, priv.smode1.LC);

		insert_label(cmd,
		             "  T1248 %u, SLCK %u, CMOD %u",
		             priv.smode1.T1248, priv.smode1.SLCK, priv.smode1.CMOD);

		insert_label(cmd,
		             "  EX %u, PRST %u, SINT %u, XPCK %u",
		             priv.smode1.EX, priv.smode1.PRST, priv.smode1.SINT, priv.smode1.XPCK);

		insert_label(cmd,
		             "  PCK2 %u, SPML %u, GCONT %u, PHS %u, PVS %u",
		             priv.smode1.PCK2, priv.smode1.SPML, priv.smode1.GCONT, priv.smode1.PHS, priv.smode1.PVS);

		insert_label(cmd,
		             "  PEHS %u, PEVS %u, CLKSEL %u, NVCK %u, SLCK2 %u, VCKSEL %u, VHP %u",
		             priv.smode1.PEHS, priv.smode1.PEVS,
		             priv.smode1.CLKSEL, priv.smode1.NVCK, priv.smode1.SLCK2,
		             priv.smode1.VCKSEL, priv.smode1.VHP);

		insert_label(cmd,
			 "SYNCV: VBP %u, VBPE %u, VDP %u, VFP %u, VFPE %u, VS %u",
			 priv.syncv.VBP,
			 priv.syncv.VBPE,
			 priv.syncv.VDP,
			 priv.syncv.VFP,
			 priv.syncv.VFPE,
			 priv.syncv.VS);
#endif

		insert_label(cmd,
		             "SMODE2: DPMS %u, FFMD %u, INT %u",
		             priv.smode2.DPMS, priv.smode2.FFMD, priv.smode2.INT);

		// Unimplemented.
		insert_label(cmd, "EXTWRITE: %u", priv.extwrite.WRITE);

		insert_label(cmd,
		             "EXTBUF: 0x%x - stride %u - EMODA %u - EMODC %u - FBIN %u - WDX/Y %u, %u - WFFMD %u",
		             priv.extbuf.EXBP * PGS_BLOCK_ALIGNMENT_BYTES,
		             priv.extbuf.EXBW * PGS_BUFFER_WIDTH_SCALE,
		             priv.extbuf.EMODA,
		             priv.extbuf.EMODC,
		             priv.extbuf.FBIN,
		             priv.extbuf.WDX,
		             priv.extbuf.WDY,
		             priv.extbuf.WFFMD);

		insert_label(cmd,
		             "EXTDATA: SX/Y %u, %u - SMPH/V %u, %u - WW/WH %u, %u",
		             priv.extdata.SX, priv.extdata.SY,
		             priv.extdata.SMPH, priv.extdata.SMPV,
		             priv.extdata.WW, priv.extdata.WH);
	}

	if (alternative_sampling && !force_progressive)
	{
		// Full-height input, but CRTC can only sample half the lines.
		// Some games seem to blend two layers with DY = 0 and DY = 1,
		// meaning they will effectively blend line 0 and 1 in phase 0 and line 1 and 2 in phase 1.
		// Seems like a reasonable way to do it?
		// Seems like PCSX2 tries really hard to promote this style to progressive scan.
		// Can always revisit later.
		phase = info.phase;
	}

	bool EN1 = priv.pmode.EN1;
	bool EN2 = priv.pmode.EN2;
	uint32_t MMOD = priv.pmode.MMOD;
	uint32_t ALP = priv.pmode.ALP;
	uint32_t SLBG = priv.pmode.SLBG;
	VkOffset2D crtc_shift = { INT32_MAX, INT32_MAX };
	VkRect2D crtc_rects[2] = {};
	bool skip_shift_x = false;
	bool skip_shift_y = false;

	if (EN1 && EN2 && anti_blur && alternative_sampling)
	{
		if (priv.dispfb1.FBP == priv.dispfb2.FBP &&
		    priv.dispfb1.PSM == priv.dispfb2.PSM &&
		    priv.dispfb1.FBW == priv.dispfb2.FBW &&
		    priv.dispfb1.DBX == priv.dispfb2.DBX &&
		    priv.display1.MAGH == priv.display2.MAGH &&
		    priv.display1.MAGV == priv.display2.MAGV)
		{
			// Games tend to either offset DY by 1, or DBY by one. Detect various cases and disable.
			if (priv.display1.DY == priv.display2.DY)
			{
				if (priv.dispfb1.DBY + 1 == priv.dispfb2.DBY)
				{
					EN2 = false;
					ALP = 0xff;
					MMOD = PMODEBits::MMOD_ALPHA_ALP;
					insert_label(cmd, "Anti-blur, force layer 0");
				}
				else if (priv.dispfb1.DBY == priv.dispfb2.DBY + 1)
				{
					EN1 = false;
					SLBG = PMODEBits::SLBG_ALPHA_BLEND_CIRCUIT2;
					insert_label(cmd, "Anti-blur, force layer 1");
				}
			}
			else if (priv.dispfb1.DBY == priv.dispfb2.DBY)
			{
				if (priv.display1.DY + 1 == priv.display2.DY)
				{
					EN2 = false;
					ALP = 0xff;
					MMOD = PMODEBits::MMOD_ALPHA_ALP;
					insert_label(cmd, "Anti-blur, force layer 0");
				}
				else if (priv.display1.DY == priv.display2.DY + 1)
				{
					EN1 = false;
					SLBG = PMODEBits::SLBG_ALPHA_BLEND_CIRCUIT2;
					insert_label(cmd, "Anti-blur, force layer 1");
				}
			}
		}
	}

	if (EN1)
	{
		if (device->consumes_debug_markers())
		{
			insert_label(cmd,
			             "EN1: 0x%x - %s - stride %u - DBX/Y %u, %u - DX/Y %u, %u - DW/H %u, %u - MAGH/V %u, %u",
			             priv.dispfb1.FBP * PGS_PAGE_ALIGNMENT_BYTES,
			             psm_to_str(priv.dispfb1.PSM),
			             priv.dispfb1.FBW * PGS_BUFFER_WIDTH_SCALE,
			             priv.dispfb1.DBX, priv.dispfb1.DBY,
			             priv.display1.DX, priv.display1.DY,
			             priv.display1.DW, priv.display1.DH,
			             priv.display1.MAGH, priv.display1.MAGV);
		}

		auto rect = compute_circuit_rect(priv, phase, priv.display1, force_progressive);
		image_info.width = rect.image_extent.width;
		image_info.height = rect.image_extent.height;

		if (image_info.width && image_info.height)
		{
			circuit1 = device->create_image(image_info);
			sample_crtc_circuit(cmd, *circuit1, priv.dispfb1, rect);
			device->set_name(*circuit1, "Circuit1");
		}

		int off_x = int(priv.display1.DX) / int(clock_divider) - scan_offset_x;
		int off_y = ((int(priv.display1.DY) + int(alternative_sampling && !is_interlaced)) >> int(is_interlaced)) - scan_offset_y;
		uint32_t width = (priv.display1.DW + 1) / clock_divider;
		uint32_t height = (priv.display1.DH + 1 + int(is_interlaced)) >> int(is_interlaced);

		if (!is_interlaced)
			height = (height + 1) & ~1;

		crtc_rects[0] = {{ off_x, off_y }, { width, height }};

		// If game is not using 1:1 regions, we'll need to consider CRTC offsets.
		// We could try to center it ourselves, but it would likely break in some cases.
		if (width != mode_width)
			skip_shift_x = true;
		if (height != mode_height)
			skip_shift_y = true;

		crtc_shift.x = std::min<int32_t>(off_x, crtc_shift.x);
		crtc_shift.y = std::min<int32_t>(off_y, crtc_shift.y);
	}

	if (EN2)
	{
		if (device->consumes_debug_markers())
		{
			insert_label(cmd,
			             "EN2: 0x%x - %s - stride %u - DBX/Y %u, %u - DX/Y %u, %u - DW/H %u, %u - MAGH/V %u, %u",
			             priv.dispfb2.FBP * PGS_PAGE_ALIGNMENT_BYTES,
			             psm_to_str(priv.dispfb2.PSM),
			             priv.dispfb2.FBW * PGS_BUFFER_WIDTH_SCALE,
			             priv.dispfb2.DBX, priv.dispfb2.DBY,
			             priv.display2.DX, priv.display2.DY,
			             priv.display2.DW, priv.display2.DH,
			             priv.display2.MAGH, priv.display2.MAGV);
		}

		auto rect = compute_circuit_rect(priv, phase, priv.display2, force_progressive);
		image_info.width = rect.image_extent.width;
		image_info.height = rect.image_extent.height;

		if (image_info.width && image_info.height)
		{
			circuit2 = device->create_image(image_info);
			sample_crtc_circuit(cmd, *circuit2, priv.dispfb2, rect);
			device->set_name(*circuit2, "Circuit2");
		}

		int off_x = int(priv.display2.DX) / int(clock_divider) - scan_offset_x;
		int off_y = ((int(priv.display2.DY) + int(alternative_sampling && !is_interlaced)) >> int(is_interlaced)) - scan_offset_y;
		uint32_t width = (priv.display2.DW + 1) / clock_divider;
		uint32_t height = (priv.display2.DH + 1 + int(is_interlaced)) >> int(is_interlaced);

		if (!is_interlaced)
			height = (height + 1) & ~1;

		crtc_rects[1] = {{ off_x, off_y }, { width, height }};

		// If game is not using 1:1 regions, we'll need to consider CRTC offsets.
		// We could try to center it ourselves, but it would likely break in some cases.
		if (width != mode_width)
			skip_shift_x = true;
		if (height != mode_height)
			skip_shift_y = true;

		crtc_shift.x = std::min<int32_t>(off_x, crtc_shift.x);
		crtc_shift.y = std::min<int32_t>(off_y, crtc_shift.y);
	}

	if (!info.overscan && !info.crtc_offsets)
	{
		if (!skip_shift_x)
		{
			crtc_rects[0].offset.x -= crtc_shift.x;
			crtc_rects[1].offset.x -= crtc_shift.x;
		}

		if (!skip_shift_y)
		{
			crtc_rects[0].offset.y -= crtc_shift.y;
			crtc_rects[1].offset.y -= crtc_shift.y;
		}
	}

	if (info.adapt_to_internal_horizontal_resolution)
	{
		uint32_t horiz_resolution0 = circuit1 ? circuit1->get_width() : 0;
		uint32_t horiz_resolution1 = circuit2 ? circuit2->get_width() : 0;

		if (horiz_resolution0 == 0)
			horiz_resolution0 = horiz_resolution1;
		if (horiz_resolution1 == 0)
			horiz_resolution1 = horiz_resolution0;

		if (horiz_resolution0 && horiz_resolution0 == horiz_resolution1)
		{
			float width_scaling = float(horiz_resolution0) / float(mode_width);
			crtc_rects[0].offset.x = int32_t(std::round(float(crtc_rects[0].offset.x) * width_scaling));
			crtc_rects[1].offset.x = int32_t(std::round(float(crtc_rects[1].offset.x) * width_scaling));
			crtc_rects[0].extent.width = uint32_t(std::round(float(crtc_rects[0].extent.width) * width_scaling));
			crtc_rects[1].extent.width = uint32_t(std::round(float(crtc_rects[1].extent.width) * width_scaling));
			mode_width = horiz_resolution0;
		}
	}

	ScanoutResult result = {};
	result.mode_width = mode_width;
	result.mode_height = mode_height;

	if (info.raw_circuit_scanout &&
	    !info.crtc_offsets && !info.overscan &&
	    info.adapt_to_internal_horizontal_resolution &&
	    !force_deinterlace && !is_interlaced)
	{
		bool is_raw_circuit1 =
				circuit1 && !circuit2 && MMOD == PMODEBits::MMOD_ALPHA_ALP && ALP == 0xff &&
				circuit1->get_width() <= mode_width && circuit1->get_height() <= mode_height;
		bool is_raw_circuit2 =
				circuit2 && !circuit1 && SLBG == PMODEBits::SLBG_ALPHA_BLEND_CIRCUIT2 &&
				circuit2->get_width() <= mode_width && circuit2->get_height() <= mode_height;

		if (is_raw_circuit1)
			result.image = std::move(circuit1);
		else if (is_raw_circuit2)
			result.image = std::move(circuit2);

		if (result.image)
		{
			cmd.image_barrier(*result.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			                  info.dst_layout,
			                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			                  info.dst_stage, info.dst_access);

			result.internal_width = result.image->get_width();
			result.internal_height = result.image->get_height();
			flush_submit(0);
			return result;
		}
	}

	result.internal_width = mode_width;
	result.internal_height = mode_height;
	image_info.width = mode_width;
	image_info.height = mode_height;
	auto merged = device->create_image(image_info);

	device->set_name(*merged, "Merged field");

	if (circuit1)
	{
		cmd.image_barrier(*circuit1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
	}

	if (circuit2)
	{
		cmd.image_barrier(*circuit2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
	}

	cmd.image_barrier(*merged, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	                  0, 0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
	                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT);

	// Execution barrier so that we don't render to VRAM before we're done sampling.
	cmd.barrier(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0);

	Vulkan::RenderPassInfo rp = {};
	rp.num_color_attachments = 1;
	rp.color_attachments[0] = &merged->get_view();
	rp.clear_attachments = 1u << 0;
	rp.store_attachments = 1u << 0;

	// All of this is somewhat ad-hoc and incomplete. Works fine in the games I've tested so far :)

	rp.clear_color[0].float32[0] = float(priv.bgcolor.R) * (1.0f / 255.0f);
	rp.clear_color[0].float32[1] = float(priv.bgcolor.G) * (1.0f / 255.0f);
	rp.clear_color[0].float32[2] = float(priv.bgcolor.B) * (1.0f / 255.0f);

	Vulkan::QueryPoolHandle start_ts, end_ts;
	if (enable_timestamps)
		start_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT);

	cmd.begin_render_pass(rp);
	cmd.set_opaque_sprite_state();

	if (SLBG == PMODEBits::SLBG_ALPHA_BLEND_CIRCUIT2 && circuit2)
	{
		cmd.set_program(blit_quad);
		cmd.set_texture(0, 0, circuit2->get_view(), Vulkan::StockSampler::LinearClamp);

		if (crtc_rects[1].extent.width && crtc_rects[1].extent.height)
		{
			VkViewport vp = {};
			vp.x = float(crtc_rects[1].offset.x);
			vp.y = float(crtc_rects[1].offset.y);
			vp.width = float(crtc_rects[1].extent.width);
			vp.height = float(crtc_rects[1].extent.height);
			vp.minDepth = 0.0f;
			vp.maxDepth = 1.0f;
			cmd.set_viewport(vp);

			cmd.draw(3);
		}
	}

	if (circuit1)
	{
		cmd.set_program(blit_quad);
		cmd.set_texture(0, 0, circuit1->get_view(), Vulkan::StockSampler::LinearClamp);

		if (crtc_rects[0].extent.width && crtc_rects[0].extent.height)
		{
			VkViewport vp = {};
			vp.x = float(crtc_rects[0].offset.x);
			vp.y = float(crtc_rects[0].offset.y);
			vp.width = float(crtc_rects[0].extent.width);
			vp.height = float(crtc_rects[0].extent.height);
			vp.minDepth = 0.0f;
			vp.maxDepth = 1.0f;
			cmd.set_viewport(vp);

			if (MMOD == PMODEBits::MMOD_ALPHA_ALP)
			{
				// Constant blend factor blend.
				if (ALP != 0xff)
				{
					cmd.set_blend_enable(true);
					cmd.set_blend_op(VK_BLEND_OP_ADD);
					cmd.set_blend_factors(VK_BLEND_FACTOR_CONSTANT_ALPHA, VK_BLEND_FACTOR_ONE,
					                      VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA, VK_BLEND_FACTOR_ZERO);

					float alp = float(uint32_t(priv.pmode.ALP)) * (1.0f / 255.0f);
					const float alps[4] = { alp, alp, alp, alp };
					cmd.set_blend_constants(alps);
				}
			}
			else
			{
				// Normal alpha-blend.
				cmd.set_blend_enable(true);
				cmd.set_blend_op(VK_BLEND_OP_ADD);
				cmd.set_blend_factors(VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE,
					VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_ZERO);
			}

			cmd.draw(3);
		}
	}

	cmd.end_render_pass();

	const bool need_intermediate_pass = priv.extwrite.WRITE || is_interlaced || force_deinterlace;
	VkPipelineStageFlags2 dst_stage =
			need_intermediate_pass ? VkPipelineStageFlags2(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT) : info.dst_stage;
	if (priv.extwrite.WRITE)
		dst_stage |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

	cmd.image_barrier(*merged, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	                  need_intermediate_pass ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : info.dst_layout,
	                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
					  dst_stage, need_intermediate_pass ? VkAccessFlags2(VK_ACCESS_2_SHADER_SAMPLED_READ_BIT) : info.dst_access);

	if (priv.extwrite.WRITE)
	{
		cmd.set_program(shaders.extwrite);
		cmd.set_specialization_constant_mask(3);
		cmd.set_specialization_constant(0, vram_size - 1);
		cmd.set_specialization_constant(1, uint32_t(can_potentially_super_sample()));

		struct Registers
		{
			uvec2 resolution;
			uint wdx;
			uint wdy;
			vec2 uv_base;
			vec2 uv_scale;
			uint exbp;
			uint exbw;
			uint wffmd;
			uint emoda;
			uint emodc;
		} push = {};

		push.resolution.x = (priv.extdata.WW + 1) / clock_divider;
		push.resolution.y = priv.extdata.WH + 1;
		push.wdx = priv.extbuf.WDX;
		push.wdy = priv.extbuf.WDY;
		push.exbp = priv.extbuf.EXBP;
		push.exbw = priv.extbuf.EXBW;
		push.wffmd = priv.extbuf.WFFMD;
		push.emoda = priv.extbuf.EMODA;
		push.emodc = priv.extbuf.EMODC;

		// TODO: Handle WFFMD == 0 somehow?
		if (priv.extbuf.WFFMD)
			push.resolution.y = (push.resolution.y + 1) / 2;

		cmd.set_storage_buffer(0, 0, *buffers.gpu);
		const Vulkan::ImageView *view = nullptr;

		if (priv.extbuf.FBIN == 0)
			view = &merged->get_view();
		else if (circuit2)
			view = &circuit2->get_view();
		else
			view = &circuit1->get_view();

		// Not sure what should happen if we haven't enabled the output circuit.
		if (view)
		{
			auto write_rect = compute_page_rect(priv.extbuf.EXBP, priv.extbuf.WDX, priv.extbuf.WDY,
			                                    push.resolution.x, push.resolution.y,
			                                    priv.extbuf.EXBW, PSMCT32);
			tracker.mark_external_write(write_rect);

			// TODO: Consider SX, SY.
			push.uv_base = vec2(0.5f) / vec2(push.resolution);
			push.uv_scale.x = float(priv.extdata.SMPH + 1) / float(view->get_view_width() * clock_divider);
			push.uv_scale.y = float(priv.extdata.SMPV + 1) / float(view->get_view_height());
			cmd.push_constants(&push, 0, sizeof(push));

			// Is the write-back filtered at all? Probably not, but whatever.
			cmd.set_texture(0, 1, *view, Vulkan::StockSampler::NearestClamp);
			cmd.dispatch((push.resolution.x + 7) / 8, (push.resolution.y + 7) / 8, 1);
			cmd.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
			            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
		}

		if (!is_interlaced && !force_deinterlace &&
		    (info.dst_stage != VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT ||
		     info.dst_access != VK_ACCESS_2_SHADER_SAMPLED_READ_BIT ||
		     info.dst_layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL))
		{
			cmd.image_barrier(*merged, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, info.dst_layout,
			                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
			                  info.dst_stage, info.dst_access);
		}
	}

	if (is_interlaced || force_deinterlace)
	{
		for (int i = 3; i >= 1; i--)
			vsync_last_fields[i] = std::move(vsync_last_fields[i - 1]);
		vsync_last_fields[0] = std::move(merged);

		// Fill in holes for the first frames.
		if (!vsync_last_fields[1])
			vsync_last_fields[1] = vsync_last_fields[0];
		if (!vsync_last_fields[2])
			vsync_last_fields[2] = vsync_last_fields[0];
		if (!vsync_last_fields[3])
			vsync_last_fields[3] = vsync_last_fields[1];

		// Crude de-interlace. Get something working for now.
		merged = fastmad_deinterlace(cmd, info);
	}
	else
	{
		vsync_last_fields[0].reset();
		vsync_last_fields[1].reset();
	}

	cmd.end_region();

	if (enable_timestamps)
	{
		end_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT);
		timestamps.push_back({ TimestampType::VSync, std::move(start_ts), std::move(end_ts) });
	}

	result.image = std::move(merged);
	flush_submit(0);
	return result;
}

Vulkan::ImageHandle GSRenderer::fastmad_deinterlace(Vulkan::CommandBuffer &cmd, const VSyncInfo &vsync)
{
	auto image_info = Vulkan::ImageCreateInfo::immutable_2d_image(
			vsync_last_fields[0]->get_width(), vsync_last_fields[0]->get_height() * 2, VK_FORMAT_R8G8B8A8_UNORM);
	image_info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
	image_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	image_info.misc |= Vulkan::IMAGE_MISC_MUTABLE_SRGB_BIT;

	auto deinterlaced = device->create_image(image_info);
	device->set_name(*deinterlaced, "Deinterlaced");

	cmd.image_barrier(*deinterlaced, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	                  0, 0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

	cmd.begin_region("deinterlace");

	Vulkan::RenderPassInfo rp_info = {};
	rp_info.num_color_attachments = 1;
	rp_info.color_attachments[0] = &deinterlaced->get_view();
	rp_info.store_attachments = 1u << 0;
	cmd.begin_render_pass(rp_info);
	cmd.set_opaque_sprite_state();
	cmd.set_program(weave_quad);

	struct Push
	{
		uint32_t phase;
		uint32_t height_minus_1;
	} push = { vsync.phase, deinterlaced->get_height() - 1 };

	cmd.push_constants(&push, 0, sizeof(push));

	for (int i = 0; i < 4; i++)
		cmd.set_texture(0, i, vsync_last_fields[i]->get_view());

	cmd.draw(3);
	cmd.end_render_pass();

	cmd.image_barrier(*deinterlaced, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, vsync.dst_layout,
	                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
	                  vsync.dst_stage, vsync.dst_access);

	cmd.end_region();
	return deinterlaced;
}
}
