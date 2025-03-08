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
#include "thread_name.hpp"
#include <utility>
#include <algorithm>
#include <cmath>
#include <climits>

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
static constexpr VkDeviceSize MaximumAllocatedScratchMemory = 100 * 1000 * 1000;
static constexpr uint32_t MaxPendingPaletteUploads = 4096;
static constexpr uint32_t MaxPendingCopies = 16 * 1024;
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

static constexpr int MaxSamples = 16;
static constexpr int PhaseLUTGridSize = 64;
static constexpr int MaxPhaseLUTResults = MaxSamples * 3 * 3;

struct PhaseLUTResult
{
	int sample_id;
	ivec2 texel_offset;
	ivec2 dist;
	int max_dist;
};

static int compute_sample_points(
	ivec2 *sample_points, uint32_t sampling_rate_x_log2, uint32_t sampling_rate_y_log2, bool nearest)
{
	int num_sample_points = 1 << (sampling_rate_x_log2 + sampling_rate_y_log2);

	// Generate the same sampling patterns as ubershader.
	// Sparse sampling grid.
	if (sampling_rate_y_log2 - sampling_rate_x_log2 == 2)
	{
		for (int i = 0; i < num_sample_points; i++)
		{
			constexpr int sparse_offsets[4] = {0, 2, 3, 1};
			sample_points[i] = ivec2(i / 8, i % 8);
			sample_points[i].x *= 4;
			sample_points[i].x += sparse_offsets[i % 4];
		}
	}
	else
	{
		for (int i = 0; i < num_sample_points; i++)
		{
			sample_points[i].y = (i >> 0) & 1;
			sample_points[i].x = (i >> 1) & 1;
			sample_points[i].y += ((i >> 2) & 1) * 2;
			sample_points[i].x += ((i >> 3) & 1) * 2;

			// Checkerboards
			if (sampling_rate_y_log2 - sampling_rate_x_log2 == 1)
			{
				sample_points[i].x *= 2;
				sample_points[i].x += i % 2;
			}
		}
	}

	// Rescale sampling points to grid.
	for (int i = 0; i < num_sample_points; i++)
	{
		int scale_factor = PhaseLUTGridSize >> sampling_rate_y_log2;
		sample_points[i] *= scale_factor;
		if (nearest)
			sample_points[i] += (scale_factor - 1) >> 1;
	}

	return num_sample_points;
}

static int compute_phase_lut_samples(
	int x, int y,
	const ivec2 *sample_points, int num_sample_points,
	int texel_support,
	PhaseLUTResult *results)
{
	int num_results = 0;

	for (int i = 0; i < num_sample_points; i++)
	{
		for (int texel_y = -texel_support; texel_y <= texel_support; texel_y++)
		{
			for (int texel_x = -texel_support; texel_x <= texel_support; texel_x++)
			{
				int dist_x = (sample_points[i].x + PhaseLUTGridSize * texel_x) - x;
				int dist_y = (sample_points[i].y + PhaseLUTGridSize * texel_y) - y;

				auto &result = results[num_results++];
				result.sample_id = i;
				result.texel_offset = ivec2(texel_x, texel_y);
				result.dist = muglm::abs(ivec2(dist_x, dist_y));
				result.max_dist = muglm::max(result.dist.x, result.dist.y);
			}
		}
	}

	std::sort(results, results + num_results, [](const PhaseLUTResult &a, const PhaseLUTResult &b)
	{
		if (a.max_dist == b.max_dist)
			return a.dist.y < b.dist.y;
		else
			return a.max_dist < b.max_dist;
	});

	return num_results;
}

void GSRenderer::init_phase_lut(uint32_t sampling_rate_x_log2, uint32_t sampling_rate_y_log2)
{
	uvec2 samples[PhaseLUTGridSize][PhaseLUTGridSize] = {};
	uvec2 samples_nearest[PhaseLUTGridSize / 2][PhaseLUTGridSize / 2] = {};
	const Vulkan::ImageInitialData level_data[2] = {{samples}, {samples_nearest}};
	auto info = Vulkan::ImageCreateInfo::immutable_2d_image(
		PhaseLUTGridSize, PhaseLUTGridSize, VK_FORMAT_R32G32_UINT);
	info.levels = 2;

	if (!sampling_rate_x_log2 && !sampling_rate_y_log2)
	{
		// Just need to bind something to be spec compliant.
		info.width = 1;
		info.height = 1;
		info.levels = 1;
		buffers.phase_lut = device->create_image(info, level_data);
		return;
	}

	ivec2 sample_points[MaxSamples];
	ivec2 sample_points_nearest[MaxSamples];
	int num_sample_points = compute_sample_points(sample_points, sampling_rate_x_log2, sampling_rate_y_log2, false);
	compute_sample_points(sample_points_nearest, sampling_rate_x_log2, sampling_rate_y_log2, true);
	PhaseLUTResult results[MaxPhaseLUTResults];

	int falloff_dist = INT_MAX;

	// Figure out how wide the filter kernel can be. The window must be small enough that
	// only the nearest 4 samples have kernel support.
	for (int y = 0; y < PhaseLUTGridSize; y++)
	{
		for (int x = 0; x < PhaseLUTGridSize; x++)
		{
			compute_phase_lut_samples(x, y, sample_points, num_sample_points, 1, results);
			falloff_dist = std::min<int>(falloff_dist, results[4].max_dist);
		}
	}

	for (int y = 0; y < PhaseLUTGridSize; y++)
	{
		for (int x = 0; x < PhaseLUTGridSize; x++)
		{
			int num_results = compute_phase_lut_samples(x, y, sample_points, num_sample_points, 1, results);

			if (results[0].texel_offset.x != 0 || results[0].texel_offset.y != 0)
			{
				for (int i = 1; i < num_results; i++)
				{
					// Ensure that the first sample has texel offset (0, 0).
					if (results[i].texel_offset.x == 0 && results[i].texel_offset.y == 0)
					{
						if (i < 4)
						{
							// Found candidate in-bounds.
							std::swap(results[0], results[i]);
						}
						else
						{
							// There were no in-bounds samples for this phase.
							// Throw away the lowest weight OOB sample.
							std::swap(results[3], results[i]);
							std::swap(results[0], results[3]);
						}

						break;
					}
				}
			}

			assert(results[0].texel_offset.x == 0 && results[0].texel_offset.y == 0);

			float weights[4];
			float weight_sum = 0.0f;

			// Encode a filter kernel in 64-bit.
			// .x: lower 16 bits encode 4 sample IDs.
			// .x: higher 16 bits encode 4 i2x2 texel offsets.
			// .y: Kernel weights in unorm8x4.
			for (int i = 0; i < 4; i++)
			{
				int sample_id = results[i].sample_id;
				auto &texel_offset = results[i].texel_offset;
				samples[y][x].x |= sample_id << (4 * i);
				samples[y][x].x |= (texel_offset.x & 3u) << (4 * i + 16 + 0);
				samples[y][x].x |= (texel_offset.y & 3u) << (4 * i + 16 + 2);
				vec2 dist = vec2(results[i].dist) / vec2(falloff_dist);
				weights[i] = muglm::max(0.0f, 1.0f - dist.x) * muglm::max(0.0f, 1.0f - dist.y);
				weight_sum += weights[i];
			}

			// Safe-guard against a case where all samples end up at the falloff outer edge.
			if (weight_sum < 0.001f)
			{
				for (auto &w : weights)
					w = 1.0f;
				weight_sum = 4.0f;
			}

			uint32_t unorm_sum = 0;

			for (int i = 0; i < 3; i++)
			{
				uint32_t weight = uint32_t(weights[i] * 255.0f / weight_sum + 0.5f);
				weight = std::min<uint32_t>(weight, 255 - unorm_sum);
				unorm_sum += weight;
				samples[y][x].y |= weight << (8 * i);
			}

			// Error redistribution. Ensure that the sum in unorm8 space is 255.
			assert(unorm_sum <= 255);
			samples[y][x].y |= (255 - unorm_sum) << 24;
		}
	}

	for (int y = 0; y < PhaseLUTGridSize / 2; y++)
	{
		for (int x = 0; x < PhaseLUTGridSize / 2; x++)
		{
			compute_phase_lut_samples(2 * x, 2 * y, sample_points_nearest, num_sample_points, 0, results);
			samples_nearest[y][x].x = results[0].sample_id;
		}
	}

	buffers.phase_lut = device->create_image(info, level_data);
}

void GSRenderer::invalidate_super_sampling_state(
    uint32_t sampling_rate_x_log2, uint32_t sampling_rate_y_log2)
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

	init_phase_lut(sampling_rate_x_log2, sampling_rate_y_log2);

	flush_slab_cache();
	device->wait_idle();
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

void GSRenderer::set_field_aware_super_sampling(bool enable)
{
	field_aware_super_sampling = enable;
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

	// Ideally we just have one big memory pool.
	// On iGPU there should be no need to copy memory around.
	info.domain = Vulkan::BufferDomain::UMACachedCoherentPreferDevice;
	info.misc = Vulkan::BUFFER_MISC_ZERO_INITIALIZE_BIT;
	buffers.gpu = device->create_buffer(info);
	device->set_name(*buffers.gpu, "vram-gpu");

	if (!device->map_host_buffer(*buffers.gpu, 0))
	{
		info.domain = Vulkan::BufferDomain::CachedHost;
		buffers.cpu = device->create_buffer(info);
		info.size = vram_size;
		device->set_name(*buffers.cpu, "vram-cpu");
		LOGI("Discrete GPU detected. Opting in for PCI-e copies to keep CPU/GPU in sync.\n");
	}
	else
	{
		LOGI("UMA-style device detected. Avoiding redundant readback copies.\n");
		buffers.cpu = buffers.gpu;
	}

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

	uint32_t num_pages = vram_size / PageSize;
	uint32_t num_pages_u32 = (num_pages + 31) / 32;
	sync_vram_shadow_pages.resize(num_pages_u32);
	vram_copy_write_pages.resize(num_pages_u32);

#if 0
	info.size = sizeof(uint32_t);
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	info.domain = Vulkan::BufferDomain::CachedHost;
	info.misc = Vulkan::BUFFER_MISC_ZERO_INITIALIZE_BIT;
	buffers.bug_feedback = device->create_buffer(info);
#endif
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
			{ 0, 2 },
			{ 1, 2 },
			{ 2, 2 },
			{ 1, 3 },
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

		static const struct
		{
			uint32_t lo, hi;
		} subgroup_configs[] = {
			{ 6, 6 },
			{ 4, 6 },
			{ 3, 6 },
			{ 2, 6 },
		};

		for (auto &subgroup_config : subgroup_configs)
		{
			if (device->supports_subgroup_size_log2(true, subgroup_config.lo, subgroup_config.hi))
				cmd->set_subgroup_size_log2(true, subgroup_config.lo, subgroup_config.hi);
			else
				continue;

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
							cmd->set_specialization_constant(6, feedback.feedback_psm);
							cmd->set_specialization_constant(7, feedback.feedback_cpsm);

							uint32_t active_flags =
									flags | (rates.sample_y ? VARIANT_FLAG_HAS_SUPER_SAMPLE_REFERENCE_BIT : 0);
							cmd->set_specialization_constant(5, active_flags);
							cmd->extract_pipeline_state(deferred);
							tasks.push_back(deferred);

							if (rates.sample_y != 0)
							{
								active_flags |= VARIANT_FLAG_HAS_TEXTURE_ARRAY_BIT;
								cmd->set_specialization_constant(5, active_flags);
								cmd->extract_pipeline_state(deferred);
								tasks.push_back(deferred);
							}

							if (rates.sample_x == 0 && rates.sample_y == 0 && can_potentially_super_sample())
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

			// If device has wave64, we always use it.
			if (device->supports_subgroup_size_log2(true, 6, 6))
				break;
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
		cmd->set_specialization_constant_mask(0x7);

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

		const uint32_t vram_masks[] = { PageSize - 1, vram_size - 1 };

		for (auto &format : formats)
		{
			cmd->set_specialization_constant(0, format.psm);
			cmd->set_specialization_constant(2, format.cpsm);

			for (auto mask : vram_masks)
			{
				cmd->set_specialization_constant(1, mask);
				for (auto *prog : shaders.upload)
				{
					cmd->set_program(prog);
					cmd->extract_pipeline_state(deferred);
					tasks.push_back(deferred);
				}
			}
		}

		device->submit_discard(cmd);
	}

	{
		Vulkan::DeferredPipelineCompile deferred = {};
		auto cmd = device->request_command_buffer();
		cmd->set_program(shaders.binning);
		cmd->set_specialization_constant_mask(0x1f);
		cmd->enable_subgroup_size_control(true);

		for (uint32_t hier_binning_size = 1; hier_binning_size <= 4; hier_binning_size *= 2)
		{
			for (uint32_t sampling_rate_non_zero = 0; sampling_rate_non_zero < 2; sampling_rate_non_zero++)
			{
				for (uint32_t prim_size = 1; prim_size <= 64; prim_size *= 2)
				{
					if (prim_size != 1 && hier_binning_size == 1)
						continue;

					set_hierarchical_binning_subgroup_config(*cmd, hier_binning_size);
					cmd->set_specialization_constant(2, sampling_rate_non_zero);
					cmd->set_specialization_constant(3, hier_binning_size);
					cmd->set_specialization_constant(4, prim_size);

					for (uint32_t feedback = 0; feedback < 2; feedback++)
					{
						cmd->set_specialization_constant(1, feedback);
						cmd->extract_pipeline_state(deferred);
						tasks.push_back(deferred);
					}
				}
			}
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
	timeline_value = 0;
	timeline_thread = std::thread([this]() {
		Util::set_current_thread_name("PGS-Waiter");
		uint64_t last_waited = 0;

		for (;;)
		{
			{
				std::unique_lock<std::mutex> holder{timeline_lock};
				timeline_cond.wait(holder, [&]() { return last_waited < last_submitted_timeline; });
			}

			if (last_submitted_timeline == UINT64_MAX)
				break;

			last_waited++;
			timeline->wait_timeline(last_waited);

			{
				std::lock_guard<std::mutex> holder{timeline_lock};
				timeline_value.store(last_waited, std::memory_order_release);
				timeline_cond.notify_all();
			}
		}
	});
}

bool GSRenderer::init(Vulkan::Device *device_, const GSOptions &options)
{
	drain_compilation_tasks();

	Vulkan::ResourceLayout layout;
	shaders = Shaders<>(*device_, layout, 0);
	blit_quad = device_->request_program(shaders.quad, shaders.blit_circuit);
	sample_quad[0] = device_->request_program(shaders.quad, shaders.sample_circuit[0]);
	sample_quad[1] = device_->request_program(shaders.quad, shaders.sample_circuit[1]);
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
	    !ext.vk12_features.scalarBlockLayout ||
	    (ext.vk11_props.subgroupSupportedOperations & required_subgroup_flags) != required_subgroup_flags ||
	    !device_->supports_subgroup_size_log2(true, 2, 6) ||
	    device_->get_gpu_properties().limits.maxComputeSharedMemorySize < 32 * 1024)
	{
		LOGE("Minimum requirements for parallel-gs are not met.\n");
		LOGE("  - descriptorIndexing\n");
		LOGE("  - timelineSemaphore\n");
		LOGE("  - bufferDeviceAddress\n");
		LOGE("  - storageBuffer8BitAccess\n");
		LOGE("  - storageBuffer16BitAccess\n");
		LOGE("  - shaderInt16\n");
		LOGE("  - scalarBlockLayout\n");
		LOGE("  - Arithmetic / Shuffle / Vote / Ballot / Basic subgroup operations\n");
		LOGE("  - SubgroupSize control for [4, 64] invocations per subgroup\n");
		LOGE("  - 32 KiB shared memory\n");
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

	// Reserve 1/3 of our budget to slab-allocate image handles.
	Vulkan::HeapBudget budgets[VK_MAX_MEMORY_HEAPS] = {};
	device->get_memory_budget(budgets);
	for (uint32_t i = 0; i < device->get_memory_properties().memoryHeapCount; i++)
	{
		if ((device->get_memory_properties().memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0)
		{
			max_image_slab_size = std::max<VkDeviceSize>(
					max_image_slab_size, budgets[i].budget_size / 3);
			max_allocated_image_memory_per_flush =
					std::max<VkDeviceSize>(max_allocated_image_memory_per_flush, budgets[i].budget_size / 20);
		}
	}

	LOGI("Using image slab size of %llu MiB.\n",
	     static_cast<unsigned long long>(max_image_slab_size / (1024 * 1024)));
	LOGI("Using max allocated image memory per flush of %llu MiB.\n",
	     static_cast<unsigned long long>(max_allocated_image_memory_per_flush / (1024 * 1024)));

	return true;
}

void GSRenderer::check_flush_stats()
{
	// Make sure that we flush as soon as there is a reasonable amount of work in flight.
	// We also want to keep memory usage under control. We have to garbage collect memory.

	if (stats.num_primitives >= MinimumPrimitivesForFlush ||
	    stats.num_render_passes >= MinimumRenderPassForFlush ||
	    stats.allocated_image_memory >= max_allocated_image_memory_per_flush ||
	    stats.allocated_scratch_memory >= MaximumAllocatedScratchMemory ||
	    stats.num_copies >= MaxPendingCopies ||
	    stats.num_palette_updates >= MaxPendingPaletteUploads)
	{
#ifdef PARALLEL_GS_DEBUG
		LOGI("Too much pending work, flushing:\n");
		LOGI("  %u primitives\n", stats.num_primitives);
		LOGI("  %u render passes\n", stats.num_render_passes);
		LOGI("  %u MiB allocated image memory\n", unsigned(stats.allocated_image_memory / (1024 * 1024)));
		LOGI("  %u MiB allocated scratch memory\n", unsigned(stats.allocated_scratch_memory / (1024 * 1024)));
		LOGI("  %u palette updates\n", stats.num_palette_updates);
		LOGI("  %u copies\n", stats.num_copies);
		LOGI("  %zu pending copies\n", pending_copies.size());
		LOGI("  %u copy threads\n", stats.num_copy_threads);
		LOGI("  %u copy barriers\n", stats.num_copy_barriers);
#endif
		// Flush the work that is considered pending right now.
		// Render passes always commit their work to a command buffer right away.
		flush_transfer();
		// If we flush submissions now, we will never know if a future primitive depends on a texture,
		// so have to upload the full thing. We have already committed to using indirect dispatch.
		ensure_conservative_indirect_texture_uploads();
		flush_cache_upload();
		// Calls next_frame_context and does garbage collection.
		flush_submit(0);
	}
	else if (pending_copies.size() >= MaxPendingCopiesWithoutFlush || stats.num_copy_threads >= MaxPendingCopyThreads)
	{
		// Deal with soft limits. We have to flush since our algorithms need it, not because of pressure.
		flush_transfer();
	}
}

static VkDeviceSize align_offset(VkDeviceSize offset, VkDeviceSize align)
{
	return (offset + align - 1) & ~(align - 1);
}

void GSRenderer::flush_attribute_scratch(AttributeScratch &scratch)
{
	if (!scratch.buffer || scratch.offset == scratch.flushed_to)
		return;

	// Flush CPU caches if needed.
	device->unmap_host_buffer(*scratch.buffer, Vulkan::MEMORY_ACCESS_WRITE_BIT,
	                          scratch.flushed_to, scratch.offset - scratch.flushed_to);

	// Do PCI-e transfer if needed.
	if (scratch.gpu_buffer != scratch.buffer)
	{
		bool first_command = !async_transfer_cmd;
		ensure_command_buffer(async_transfer_cmd, Vulkan::CommandBuffer::Type::AsyncTransfer);
		if (first_command)
			async_transfer_cmd->begin_region("AsyncTransfer");

		async_transfer_cmd->begin_region("AttributeFlush");
		async_transfer_cmd->copy_buffer(*scratch.gpu_buffer, scratch.flushed_to,
		                                *scratch.buffer, scratch.flushed_to,
		                                scratch.offset - scratch.flushed_to);
		async_transfer_cmd->end_region();
	}

	scratch.flushed_to = scratch.offset;
}

void GSRenderer::reserve_attribute_scratch(VkDeviceSize size, AttributeScratch &scratch)
{
	auto align = buffers.ssbo_alignment;
	scratch.offset = align_offset(scratch.offset, align);

	if (!scratch.buffer || scratch.offset + size > scratch.size)
	{
		flush_attribute_scratch(scratch);

		Vulkan::BufferCreateInfo info = {};
		constexpr VkDeviceSize DefaultScratchBufferSize = 32 * 1024 * 1024;
		info.size = std::max<VkDeviceSize>(size, DefaultScratchBufferSize);
		info.domain = Vulkan::BufferDomain::UMACachedCoherentPreferDevice;
		info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

		scratch.gpu_buffer = device->create_buffer(info);
		scratch.offset = 0;
		scratch.size = info.size;
		scratch.flushed_to = 0;

		if (device->map_host_buffer(*scratch.gpu_buffer, 0))
		{
			scratch.buffer = scratch.gpu_buffer;
		}
		else
		{
			info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
			info.domain = Vulkan::BufferDomain::CachedHost;
			scratch.buffer = device->create_buffer(info);
		}
	}
}

void GSRenderer::commit_attribute_scratch(VkDeviceSize size, AttributeScratch &scratch)
{
	stats.allocated_scratch_memory += size;
	scratch.offset += size;
}

VkDeviceSize GSRenderer::allocate_device_scratch(VkDeviceSize size, Scratch &scratch, const void *data)
{
	// Trivial linear allocator. Reduces pressure on Granite allocator.
	// It's important that we don't allocate too huge buffers here, then we don't get suballocation in Granite
	// (which would be quite bad).
	auto align = buffers.ssbo_alignment;
	stats.allocated_scratch_memory += size;

	scratch.offset = align_offset(scratch.offset, align);
	if (!scratch.buffer || scratch.offset + size > scratch.size)
	{
		Vulkan::BufferCreateInfo info = {};
		constexpr VkDeviceSize DefaultScratchBufferSize = 32 * 1024 * 1024;
		info.size = std::max<VkDeviceSize>(size, DefaultScratchBufferSize);
		info.domain = data ? Vulkan::BufferDomain::LinkedDeviceHostPreferDevice : Vulkan::BufferDomain::Device;
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
		auto *mapped = static_cast<uint8_t *>(device->map_host_buffer(*scratch.buffer, Vulkan::MEMORY_ACCESS_WRITE_BIT));

		if (mapped)
		{
			memcpy(mapped + offset, data, size);
		}
		else
		{
			// Fallback when sufficient ReBAR is not available.
			// This also happens when capturing with RenderDoc since persistently mapped ReBAR is horrible for capture perf.
			bool first_command = !async_transfer_cmd;
			ensure_command_buffer(async_transfer_cmd, Vulkan::CommandBuffer::Type::AsyncTransfer);
			if (first_command)
				async_transfer_cmd->begin_region("AsyncTransfer");
			memcpy(async_transfer_cmd->update_buffer(*scratch.buffer, offset, size), data, size);
		}
	}

	scratch.offset += size;
	return offset;
}

GSRenderer::~GSRenderer()
{
	// Need to get rid of any command buffer handles at the very least. Otherwise, we deadlock the device.
	flush_submit(0);
	drain_compilation_tasks();

	{
		std::lock_guard<std::mutex> holder{timeline_lock};
		last_submitted_timeline = UINT64_MAX;
		timeline_cond.notify_all();
	}

	if (timeline_thread.joinable())
		timeline_thread.join();

	check_bug_feedback();
}

void GSRenderer::wait_timeline(uint64_t value)
{
	std::unique_lock<std::mutex> holder{timeline_lock};
	timeline_cond.wait(holder, [this, value]() {
		return timeline_value.load(std::memory_order_relaxed) >= value;
	});
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
	// Actually spam-calling vkGetSemaphoreValue a ton of times has serious CPU overhead.
	// We need to transition into kernel to do that.
	return timeline_value.load(std::memory_order_acquire);
}

void GSRenderer::flush_qword_clears()
{
	ensure_clear_cmd();

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

	flush_attribute_scratch(buffers.pos_scratch);
	flush_attribute_scratch(buffers.attr_scratch);
	flush_attribute_scratch(buffers.prim_scratch);

	if (direct_cmd)
	{
		// Copies may hold references to scratch buffers which must not outlive pending_cmd.
		flush_transfer();
	}

	// This must come before async transfer cmd since we risk allocating a transfer.
	if (!qword_clears.empty())
		flush_qword_clears();

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
		clear_cmd->barrier(VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_COPY_BIT |
		                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		                   VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
		                   VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
		                   VK_ACCESS_INDIRECT_COMMAND_READ_BIT);

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
		{
			std::lock_guard<std::mutex> holder{timeline_lock};
			last_submitted_timeline = value;
			timeline_cond.notify_all();
		}
		binary = device->request_timeline_semaphore_as_binary(*descriptor_timeline, next_descriptor_timeline_signal++);
		device->submit_empty(Vulkan::CommandBuffer::Type::Generic, nullptr, binary.get());
	}

	// This is a delayed sync-point between CPU and GPU, and garbage collection can happen here.
	drain_compilation_tasks_nonblock();
	device->next_frame_context();

	log_timestamps();
	check_bug_feedback();
}

void GSRenderer::check_bug_feedback()
{
#if 0
	if (!buffers.bug_feedback)
		return;

	uint32_t bug_value = *static_cast<const uint32_t *>(
			device->map_host_buffer(*buffers.bug_feedback, Vulkan::MEMORY_ACCESS_READ_BIT));
	if (bug_value != 0)
	{
		LOGE("Fatal bug detected in shaders.\n");
		abort();
	}
#endif
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
	    image->get_width() <= 1024 && image->get_height() <= 1024)
	{
		recycled_image_handles.push_back(std::move(image));
	}
}

Vulkan::ImageHandle GSRenderer::copy_cached_texture(const Vulkan::Image &img, const VkRect2D &rect)
{
	ensure_command_buffer(direct_cmd, Vulkan::CommandBuffer::Type::Generic);
	auto &cmd = *direct_cmd;

	auto info = img.get_create_info();
	info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	info.width = rect.extent.width;
	info.height = rect.extent.height;
	info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
	info.misc |= Vulkan::IMAGE_MISC_FORCE_ARRAY_BIT;

	auto copy_img = device->create_image(info);
	device->set_name(*copy_img, "Sharp Backbuffer");

	cmd.begin_region("Sharp Backbuffer Copy");
	cmd.image_barrier(img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
	                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	cmd.image_barrier(*copy_img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					  0, 0,
	                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);

	cmd.copy_image(*copy_img, img, {}, { rect.offset.x, rect.offset.y, 0 }, { rect.extent.width, rect.extent.height, 1 },
	               { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, info.layers }, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, info.layers });

	cmd.image_barrier(img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	                  VK_PIPELINE_STAGE_2_COPY_BIT, 0,
	                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
	cmd.image_barrier(*copy_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
	                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
	                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

	cmd.end_region();

	return copy_img;
}

Vulkan::ImageHandle GSRenderer::create_cached_texture(const TextureDescriptor &desc)
{
	if (!device)
		return {};

	assert(desc.rect.width && desc.rect.height);

	Vulkan::ImageHandle img = pull_image_handle_from_slab(desc.rect.width, desc.rect.height, desc.rect.levels, desc.samples);

	if (!img)
	{
		Vulkan::ImageCreateInfo info = Vulkan::ImageCreateInfo::immutable_2d_image(
			desc.rect.width, desc.rect.height, VK_FORMAT_R8G8B8A8_UNORM);

		info.levels = desc.rect.levels;
		info.layers = desc.samples;
		if (desc.samples > 1)
			info.layers++;
		info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
		info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		info.misc |= Vulkan::IMAGE_MISC_CREATE_PER_MIP_LEVEL_VIEWS_BIT;

		// Ignore mips. This is just a crude heuristic.
		stats.allocated_image_memory += info.width * info.height * info.layers * sizeof(uint32_t);

		img = device->create_image(info);
	}

	if (device->consumes_debug_markers())
	{
		// If we're running in capture tools.
		char name_str[128];

		snprintf(name_str, sizeof(name_str), "%s%s - [%u x %u] + (%u, %u) - 0x%x - bank %u @ %u - %s",
		         psm_to_str(desc.tex0.desc.PSM),
		         desc.samples > 1 ? " (SSAA)" : "",
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

	return img;
}

void GSRenderer::allocate_upload_indirection(TextureAnalysis &analysis, TextureUpload &upload)
{
	uint32_t horiz_blocks = (upload.desc.rect.width + 7) / 8;
	uint32_t vert_blocks = (upload.desc.rect.height + 7) / 8;
	uint32_t num_blocks = horiz_blocks * vert_blocks;
	uint32_t qword_blocks = (num_blocks + 127) / 128;

	analysis = {};
	analysis.flags = TextureAnalysis::ENABLED_BIT;
	analysis.block_stride = horiz_blocks;
	analysis.layers = upload.image->get_create_info().layers;

	VkDeviceSize indirect_workgroups_offset = allocate_device_scratch(sizeof(uvec4), buffers.device_scratch, nullptr);
	analysis.indirect_dispatch_va = buffers.device_scratch.buffer->get_device_address() + indirect_workgroups_offset;
	upload.indirection.indirect = buffers.device_scratch.buffer;
	upload.indirection.indirect_offset = indirect_workgroups_offset;
	qword_clears.push_back(analysis.indirect_dispatch_va);

	VkDeviceSize bit_block_offset = allocate_device_scratch(qword_blocks * sizeof(uvec4), buffers.device_scratch, nullptr);
	analysis.indirect_bitmask_va = buffers.device_scratch.buffer->get_device_address() + bit_block_offset;

	for (uint32_t i = 0; i < qword_blocks; i++)
		qword_clears.push_back(analysis.indirect_bitmask_va + sizeof(uvec4) * i);

	VkDeviceSize workgroups_offset = allocate_device_scratch(
			sizeof(uvec4) + num_blocks * sizeof(uvec2), buffers.device_scratch, nullptr);
	analysis.indirect_workgroups_va = buffers.device_scratch.buffer->get_device_address() + workgroups_offset;
	upload.indirection.buffer = buffers.device_scratch.buffer;
	upload.indirection.offset = workgroups_offset;
	upload.indirection.size = sizeof(uvec4) + num_blocks * sizeof(uvec2);
	qword_clears.push_back(analysis.indirect_workgroups_va);

	analysis.base = u16vec2(upload.desc.rect.x, upload.desc.rect.y);
	analysis.size_minus_1 = uvec2(upload.desc.rect.width - 1, upload.desc.rect.height - 1);

	pending_indirect_uploads.push_back({ upload.indirection.indirect, upload.indirection.indirect_offset,
	                                     { uint16_t(horiz_blocks),
	                                       uint16_t(vert_blocks),
	                                       uint16_t(upload.image->get_create_info().layers) }});
}

void GSRenderer::commit_cached_texture(uint32_t tex_info_index, bool sampler_feedback)
{
	// Assert that there were no stray flushes between create_cached_texture() and commit_cached_texture().
	assert(!texture_uploads.empty());

	if (sampler_feedback)
	{
		texture_analysis.resize(std::max<size_t>(texture_analysis.size(), tex_info_index + 1));
		allocate_upload_indirection(texture_analysis[tex_info_index], texture_uploads.back());
	}

	// Delay any flushing since we may want to modify the texture upload based on the page tracker later.
	check_flush_stats();
}

void GSRenderer::ensure_conservative_indirect_texture_uploads()
{
	if (pending_indirect_uploads.empty())
		return;

	if (!qword_clears.empty())
	{
		flush_qword_clears();
		// Ensure that update_buffer_inline is ordered after the qword clears.
		clear_cmd->barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
	}

	ensure_clear_cmd();

	// Ensure that any previously recorded command will indirect dispatch with the normal direct dispatch sizes.
	clear_cmd->begin_region("ensure-conservative-indirect-upload");
	for (auto &upload : pending_indirect_uploads)
	{
		uvec3 fallback = uvec3(upload.fallback_dispatch);
		clear_cmd->update_buffer_inline(*upload.indirect, upload.indirect_offset,
		                                sizeof(fallback), &fallback);
	}

	for (auto &dispatch : pending_indirect_analysis)
		qword_clears.push_back(dispatch.indirect->get_device_address() + dispatch.indirect_offset);

	clear_cmd->end_region();
	pending_indirect_uploads.clear();
	pending_indirect_analysis.clear();

	// For any pending work we've yet to record, disable the indirection.
	texture_analysis.clear();
	for (auto &upload : texture_uploads)
		upload.indirection = {};
}

void GSRenderer::promote_cached_texture_upload_cpu(const PageRect &rect)
{
	// Assert that there were no stray flushed between create_cached_texture() and commit_cached_texture().
	assert(!texture_uploads.empty());
	auto &upload = texture_uploads.back();

	assert(rect.page_width == 1 && rect.page_height == 1);
	auto *vram = static_cast<const uint8_t *>(begin_host_vram_access());
	vram += (rect.base_page * PageSize) & (vram_size - 1);

	// Only copy what we need.
	uint32_t scratch_size;

	if (rect.block_mask == UINT32_MAX)
		scratch_size = 32;
	else
		scratch_size = 32 - Util::leading_zeroes(rect.block_mask);

	scratch_size *= PGS_BLOCK_ALIGNMENT_BYTES;

	upload.scratch.offset = allocate_device_scratch(scratch_size, buffers.rebar_scratch, vram);
	upload.scratch.size = scratch_size;
	upload.scratch.buffer = buffers.rebar_scratch.buffer;
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

void GSRenderer::copy_blocks(Vulkan::CommandBuffer &cmd, const Vulkan::Buffer &dst, const Vulkan::Buffer &src,
                             const uint32_t *page_indices, uint32_t num_indices, bool invalidate_super_sampling,
                             uint32_t block_size)
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
		merger.push(page_indices[i] * block_size, block_size, flush_cb);
	merger.flush(flush_cb);
}

void GSRenderer::flush_host_vram_copy(const uint32_t *block_indices, uint32_t num_indices)
{
	if (buffers.gpu == buffers.cpu)
		return;

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

	// Cannot use more than one pipeline stage in timestamps.
	if (enable_timestamps)
		start_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_TRANSFER_BIT);

	copy_blocks(cmd, *buffers.gpu, *buffers.cpu, block_indices, num_indices, invalidate_super_sampling,
	            PGS_BLOCK_ALIGNMENT_BYTES);
	stats.num_copies++;

	if (enable_timestamps)
	{
		end_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_TRANSFER_BIT);
		timestamps.push_back({ TimestampType::SyncHostToVRAM, std::move(start_ts), std::move(end_ts) });
	}

	cmd.barrier(stages, VK_ACCESS_2_TRANSFER_WRITE_BIT,
	            stages, VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);

	cmd.end_region();
	check_flush_stats();
}

void GSRenderer::flush_readback(const uint32_t *page_indices, uint32_t num_indices)
{
	if (buffers.gpu == buffers.cpu)
		return;

	ensure_command_buffer(direct_cmd, Vulkan::CommandBuffer::Type::Generic);
	auto &cmd = *direct_cmd;

	cmd.begin_region("flush-readback");

	Vulkan::QueryPoolHandle start_ts, end_ts;
	if (enable_timestamps)
		start_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_2_COPY_BIT);
	copy_blocks(cmd, *buffers.cpu, *buffers.gpu, page_indices, num_indices, false, PageSize);
	stats.num_copies++;

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

	bound_texture_has_array = false;

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
		if (rp.textures[i].info.arrayed)
			bound_texture_has_array = true;
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
	memcpy(triangle_setup_cmd->allocate_typed_constant_data<StateVector>(
		       0, BINDING_STATE_VECTORS, std::max<uint32_t>(1, rp.num_states)),
	       rp.states, rp.num_states * sizeof(StateVector));

	cmd.set_storage_buffer(0, BINDING_CLUT, *buffers.clut);
	cmd.set_sampler(0, BINDING_SAMPLER_NEAREST, Vulkan::StockSampler::NearestWrap);
	cmd.set_sampler(0, BINDING_SAMPLER_LINEAR, Vulkan::StockSampler::LinearWrap);

	cmd.set_storage_buffer(0, BINDING_VRAM, *buffers.gpu);

	triangle_setup_cmd->set_buffer_view(0, BINDING_FIXED_RCP_LUT, *buffers.fixed_rcp_lut_view);
	triangle_setup_cmd->set_buffer_view(0, BINDING_FLOAT_RCP_LUT, *buffers.float_rcp_lut_view);
	cmd.set_texture(0, BINDING_PHASE_LUT, buffers.phase_lut->get_view(), Vulkan::StockSampler::NearestClamp);

	triangle_setup_cmd->set_storage_buffer(0, BINDING_VERTEX_POSITION, *buffers.pos_scratch.gpu_buffer,
	                                       buffers.pos_scratch.offset, rp.num_primitives * 3 * sizeof(VertexPosition));
	if (heuristic_cmd)
	{
		heuristic_cmd->set_storage_buffer(0, BINDING_VERTEX_POSITION, *buffers.pos_scratch.gpu_buffer,
		                                  buffers.pos_scratch.offset, rp.num_primitives * 3 * sizeof(VertexPosition));
	}

	triangle_setup_cmd->set_storage_buffer(0, BINDING_VERTEX_ATTRIBUTES, *buffers.attr_scratch.gpu_buffer,
	                                       buffers.attr_scratch.offset, rp.num_primitives * 3 * sizeof(VertexAttribute));

	cmd.set_storage_buffer(0, BINDING_PRIMITIVE_ATTRIBUTES, *buffers.prim_scratch.gpu_buffer,
	                       buffers.prim_scratch.offset, rp.num_primitives * sizeof(PrimitiveAttribute));
	triangle_setup_cmd->set_storage_buffer(0, BINDING_PRIMITIVE_ATTRIBUTES, *buffers.prim_scratch.gpu_buffer,
	                                       buffers.prim_scratch.offset, rp.num_primitives * sizeof(PrimitiveAttribute));

	if (heuristic_cmd)
	{
		heuristic_cmd->set_storage_buffer(0, BINDING_PRIMITIVE_ATTRIBUTES, *buffers.prim_scratch.gpu_buffer,
		                                  buffers.prim_scratch.offset, rp.num_primitives * sizeof(PrimitiveAttribute));
	}

	binning_cmd->set_storage_buffer(0, BINDING_PRIMITIVE_ATTRIBUTES, *buffers.prim_scratch.gpu_buffer,
	                                buffers.prim_scratch.offset, rp.num_primitives * sizeof(PrimitiveAttribute));

	commit_attribute_scratch(rp.num_primitives * 3 * sizeof(VertexPosition), buffers.pos_scratch);
	commit_attribute_scratch(rp.num_primitives * 3 * sizeof(VertexAttribute), buffers.attr_scratch);
	commit_attribute_scratch(rp.num_primitives * sizeof(PrimitiveAttribute), buffers.prim_scratch);
}

static uint32_t align_coarse_tiles(uint32_t num_tiles, uint32_t hier_binning)
{
	return (num_tiles + hier_binning - 1) & ~(hier_binning - 1);
}

void GSRenderer::bind_frame_resources_instanced(const RenderPass &rp, uint32_t instance, uint32_t num_primitives)
{
	auto &cmd = *direct_cmd;

	GlobalConstants constants = {};

	auto &inst = rp.instances[instance];

	uint32_t hier_binning = get_target_hierarchical_binning(
			num_primitives, inst.coarse_tiles_width, inst.coarse_tiles_height);

	constants.base_pixel.x = int(inst.base_x);
	constants.base_pixel.y = int(inst.base_y);
	constants.coarse_tile_size_log2 = int(rp.coarse_tile_size_log2);
	constants.coarse_fb_width = int(align_coarse_tiles(inst.coarse_tiles_width, hier_binning));
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
	binning_cmd->set_storage_buffer(0, BINDING_TRANSFORMED_ATTRIBUTES,
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

uint32_t GSRenderer::get_target_hierarchical_binning(
		uint32_t num_primitives, uint32_t coarse_tiles_width, uint32_t coarse_tiles_height) const
{
#ifdef __APPLE__
	// Broken Metal drivers can't deal with the hierarchical binning for some reason.
	return 1;
#endif

	// Only bother for large number of primitives.
	// Simpler full-screen blit passes and similar should just use the simplified flat binner.
	if (num_primitives < 256)
		return 1;

	// Small resolution, we won't really be able to go wide anyway.
	if (coarse_tiles_width <= 4 || coarse_tiles_height <= 4)
		return 1;

	uint32_t target_binning = num_primitives < 4096 ? 2 : 4;
	uint32_t maximum_invocations = device->get_device_features().vk11_props.subgroupSize * target_binning * target_binning;

	// Make sure that we can support the worst case size of the workgroup.
	while (maximum_invocations > device->get_gpu_properties().limits.maxComputeWorkGroupInvocations)
	{
		maximum_invocations /= 4;
		target_binning /= 2;
	}

	assert(target_binning > 0);

	return target_binning;
}

void GSRenderer::allocate_scratch_buffers_instanced(Vulkan::CommandBuffer &cmd, const RenderPass &rp,
                                                    uint32_t instance, uint32_t num_primitives)
{
	auto &inst = rp.instances[instance];

	uint32_t hier_binning = get_target_hierarchical_binning(
			num_primitives, inst.coarse_tiles_width, inst.coarse_tiles_height);
	uint32_t aligned_width = align_coarse_tiles(inst.coarse_tiles_width, hier_binning);
	uint32_t aligned_height = align_coarse_tiles(inst.coarse_tiles_height, hier_binning);

	VkDeviceSize num_coarse_tiles = aligned_width * aligned_height;
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

	VkDeviceSize work_list_size = 256 + aligned_width * aligned_height * sizeof(uvec2);

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

bool GSRenderer::render_pass_instance_is_deduced_blur(const RenderPass &rp, uint32_t instance) const
{
	// Crude heuristic to figure out if a render pass instance attempts a blur kernel.
	if (rp.num_primitives > 64)
		return false;

	uint32_t last_tex_index = UINT32_MAX;
	ivec2 phase_offset = {};

	for (uint32_t i = 0; i < rp.num_primitives; i++)
	{
		uint32_t prim_instance = (buffers.prim[i].state >> STATE_VERTEX_RENDER_PASS_INSTANCE_OFFSET) &
		                         ((1 << STATE_VERTEX_RENDER_PASS_INSTANCE_COUNT) - 1);

		if (prim_instance != instance || (buffers.prim[i].tex & TEX_PER_SAMPLE_BIT) == 0)
			continue;
		if ((buffers.prim[i].state & (1 << STATE_BIT_PERSPECTIVE)) != 0)
			return false;

		uint32_t tex_index = (buffers.prim[i].tex >> TEX_TEXTURE_INDEX_OFFSET) &
		                     ((1 << TEX_TEXTURE_INDEX_BITS) - 1);

		ivec2 phase = ivec2(buffers.attr[3 * i].uv) - buffers.pos[3 * i].pos;

		if (last_tex_index == tex_index)
		{
			if (any(notEqual(phase_offset, phase)))
				return true;
		}
		else
			phase_offset = phase;

		// When blurring, assume we're only using one texture.
		last_tex_index = tex_index;
	}

	return false;
}

static bool render_pass_instance_might_field_render(const RenderPass &rp, uint32_t instance)
{
	// Some "field" rendered games actually render the frame in full res first, then downsample.
	// In this case, we can deduce we should use normal snapping rules.
	// Very shaky heuristic since it obviously cannot catch every case.
	auto &inst = rp.instances[instance];
	bool is_tall_render_pass = (inst.base_y + (inst.coarse_tiles_height << rp.coarse_tile_size_log2)) > 256;
	return !is_tall_render_pass;
}

void GSRenderer::dispatch_triangle_setup(Vulkan::CommandBuffer &cmd, const RenderPass &rp)
{
	struct Push
	{
		uint32_t num_primitives;
		uint32_t z_shift_to_bucket;
		uint32_t rp_is_blur_mask;
	} push = {};

	push.num_primitives = rp.num_primitives;
	uint32_t sampling_rate_x_log2 = 0;
	uint32_t sampling_rate_y_log2 = 0;
	bool allow_field_render = false;

	auto *opaque_fbmasks = cmd.allocate_typed_constant_data<uint32_t>(0, BINDING_OPAQUE_FBMASKS, MaxRenderPassInstances);

	for (uint32_t i = 0; i < rp.num_instances; i++)
	{
		auto &inst = rp.instances[i];

		opaque_fbmasks[i] = inst.opaque_fbmask;

		// If we don't know, assume 24-bit range. If Z buffer isn't used at all, it's unlikely there will be proper 3D objects anyway.
		uint32_t depth_psm = inst.z_sensitive ? (inst.fb.z.desc.PSM | ZBUFBits::PSM_MSB) : PSMZ24;

		switch (depth_psm)
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

		sampling_rate_x_log2 = std::max<uint32_t>(sampling_rate_x_log2, inst.sampling_rate_x_log2);
		sampling_rate_y_log2 = std::max<uint32_t>(sampling_rate_y_log2, inst.sampling_rate_y_log2);

		if (bound_texture_has_array && render_pass_instance_is_deduced_blur(rp, i))
		{
			push.rp_is_blur_mask |= 1u << i;
			if (device->consumes_debug_markers())
				insert_label(cmd, "Instance %u, deduced blur kernel", i);
		}

		if (field_aware_super_sampling && render_pass_instance_might_field_render(rp, i))
			allow_field_render = true;
	}

	cmd.push_constants(&push, 0, sizeof(push));

	cmd.set_program(shaders.triangle_setup);
	cmd.set_specialization_constant_mask(0xf);
	cmd.set_specialization_constant(0, sampling_rate_x_log2);
	cmd.set_specialization_constant(1, sampling_rate_y_log2);
	cmd.set_specialization_constant(2, bound_texture_has_array);

	cmd.set_specialization_constant(3, bound_texture_has_array && allow_field_render);

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

void GSRenderer::set_hierarchical_binning_subgroup_config(Vulkan::CommandBuffer &cmd, uint32_t hier_factor) const
{
	const bool is_hierarchical = hier_factor != 1;
	const uint32_t subgroup_size = cmd.get_device().get_device_features().vk11_props.subgroupSize;

	// We cannot take good advantage of wave64 in the hierarchical binner, otherwise, we prefer large waves.
	uint32_t size_log2 = is_hierarchical ? 5 : 6;

	while (size_log2 >= 2)
	{
		if (device->supports_subgroup_size_log2(true, size_log2, size_log2))
		{
			cmd.set_subgroup_size_log2(true, size_log2, size_log2);
			uint32_t wg_size = std::min<uint32_t>(subgroup_size, 1u << size_log2);
			cmd.set_specialization_constant(0, wg_size * hier_factor * hier_factor);
			return;
		}

		size_log2--;
	}

	// Fallback case, allow whatever.
	cmd.set_subgroup_size_log2(true, 2, 7);
	cmd.set_specialization_constant(0, subgroup_size * hier_factor * hier_factor);
}

void GSRenderer::dispatch_binning(Vulkan::CommandBuffer &cmd, const RenderPass &rp,
                                  uint32_t instance, uint32_t base_primitive, uint32_t num_primitives)
{
	auto &inst = rp.instances[instance];

	uint32_t hier_binning = get_target_hierarchical_binning(
			num_primitives, inst.coarse_tiles_width, inst.coarse_tiles_height);

	cmd.enable_subgroup_size_control(true);
	cmd.set_specialization_constant_mask(0x1f);
	cmd.set_specialization_constant(1, uint32_t(rp.feedback_color || rp.feedback_depth));
	cmd.set_specialization_constant(2, uint32_t(inst.sampling_rate_y_log2 != 0));
	cmd.set_specialization_constant(3, hier_binning);

	bool allow_blend_demote = false;
	constexpr uint32_t BlendDemoteBudget = 32;

	if (rp.allow_blend_demote)
	{
		allow_blend_demote = true;
		for (uint32_t i = 0; i < rp.num_states && allow_blend_demote; i++)
			if ((rp.states[i].blend_mode & BLEND_MODE_ABE_BIT) == 0)
				allow_blend_demote = false;
	}

	if (hier_binning > 1)
	{
		// Hierarchical binning is a bit more special.
		// Use POT sized shared memory to not have too many shader variants.
		// We mostly just want to avoid swarming the GPU with large LDS buffers for mid-sized render passes.
		uint32_t clamped_num_primitives = Util::next_pow2(num_primitives);
		uint32_t clamped_num_primitives_1024 = (clamped_num_primitives + 1023u) / 1024u;
		cmd.set_specialization_constant(4, clamped_num_primitives_1024);
	}
	else
	{
		cmd.set_specialization_constant(4, 1u);
	}

	set_hierarchical_binning_subgroup_config(cmd, hier_binning);

	struct Push
	{
		uint32_t base_x, base_y;
		uint32_t base_primitive;
		uint32_t instance;
		uint32_t end_primitives;
		uint32_t num_samples;
		uint32_t force_super_sample;
		uint32_t allow_blend_demote;
	} push = {
		inst.base_x, inst.base_y,
		base_primitive, instance, base_primitive + num_primitives,
		1u << (inst.sampling_rate_x_log2 + inst.sampling_rate_y_log2),
		// Technically we can decay to 2x SSAA here and splat, but that too complicated to support.
		uint32_t(field_aware_super_sampling && render_pass_instance_might_field_render(rp, instance)),
		allow_blend_demote ? BlendDemoteBudget : 0,
	};

	cmd.push_constants(&push, 0, sizeof(push));
	cmd.set_program(shaders.binning);
	Vulkan::QueryPoolHandle start_ts, end_ts;
	if (enable_timestamps)
		start_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	uint32_t tiles_x = align_coarse_tiles(inst.coarse_tiles_width, hier_binning) / hier_binning;
	uint32_t tiles_y = align_coarse_tiles(inst.coarse_tiles_height, hier_binning) / hier_binning;
	cmd.dispatch(tiles_x, tiles_y, 1);

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

void GSRenderer::dispatch_shading_commands(
		Vulkan::CommandBuffer &cmd, const RenderPass &rp, uint32_t instance,
		bool post_barrier, uint32_t base_primitive, uint32_t num_primitives)
{
	auto &inst = rp.instances[instance];

	uint32_t stride = rp.debug_capture_stride ? rp.debug_capture_stride : num_primitives;
	uint32_t lo_primitive_index = 0;
	uint32_t hi_primitive_index = 0;

	bool debug = rp.feedback_color && device->consumes_debug_markers();

	for (uint32_t i = 0; i < num_primitives; i += stride)
	{
		if (debug)
		{
			lo_primitive_index = i;
			hi_primitive_index = std::min<uint32_t>(num_primitives, i + stride) - 1;
			lo_primitive_index += base_primitive;
			hi_primitive_index += base_primitive;
			cmd.push_constants(&lo_primitive_index, offsetof(ShadingDescriptor, lo_primitive_index), sizeof(uint32_t));
			cmd.push_constants(&hi_primitive_index, offsetof(ShadingDescriptor, hi_primitive_index), sizeof(uint32_t));

			begin_region(cmd, "Prim [%u, %u]", lo_primitive_index, hi_primitive_index);
			for (uint32_t j = lo_primitive_index; j <= hi_primitive_index; j++)
			{
				auto s = buffers.prim[j].state;

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
				             buffers.prim[j].fbmsk);

				auto alpha = buffers.prim[j].alpha;
				insert_label(cmd, "  AFIX: %u, AREF: %u",
				             (alpha >> ALPHA_AFIX_OFFSET) & ((1u << ALPHA_AFIX_BITS) - 1u),
				             (alpha >> ALPHA_AREF_OFFSET) & ((1u << ALPHA_AREF_BITS) - 1u));

				auto tex = buffers.prim[j].tex;
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
			cmd.set_storage_buffer(DESCRIPTOR_SET_WORKGROUP_LIST, 0,
			                       *work_list_super_sample.buffer, work_list_super_sample.offset + 256, VK_WHOLE_SIZE);
			if (debug)
				cmd.insert_label("super-sample");
			cmd.dispatch_indirect(*work_list_super_sample.buffer, work_list_super_sample.offset);
		}

		cmd.set_specialization_constant(0, 0);
		cmd.set_specialization_constant(1, 0);
		cmd.set_storage_buffer(DESCRIPTOR_SET_WORKGROUP_LIST, 0,
		                       *work_list_single_sample.buffer, work_list_single_sample.offset + 256,
		                       VK_WHOLE_SIZE);
		if (debug)
			cmd.insert_label("single-sample");
		cmd.dispatch_indirect(*work_list_single_sample.buffer, work_list_single_sample.offset);

		if (post_barrier || debug)
		{
			cmd.barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
			            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
		}
	}
}

void GSRenderer::dispatch_read_aliased_depth_passes(Vulkan::CommandBuffer &cmd, const RenderPass &rp,
                                                    uint32_t instance, uint32_t depth_psm,
                                                    ShadingDescriptor &push,
                                                    uint32_t base_primitive,
                                                    uint32_t next_lo_index, uint32_t num_primitives)
{
	auto bb = ivec4(INT_MAX, INT_MAX, INT_MIN, INT_MIN);

	uint32_t half_gpu_size = buffers.gpu->get_create_info().size / 2;
	if (get_bits_per_pixel(depth_psm) == 16)
		push.fb_index_depth_offset = half_gpu_size / sizeof(uint16_t);
	else
		push.fb_index_depth_offset = half_gpu_size / sizeof(uint32_t);

	for (uint32_t i = next_lo_index; i < num_primitives; i++)
	{
		auto &state = buffers.prim[base_primitive + i].state;
		uint32_t prim_instance = (state >> STATE_VERTEX_RENDER_PASS_INSTANCE_OFFSET) &
		                         ((1u << STATE_VERTEX_RENDER_PASS_INSTANCE_COUNT) - 1u);

		// Only care about sprites here. If we're doing normal triangle rendering,
		// it's doubtful we're doing anything interesting with feedback on aliasing.
		// Most likely some kind of weird post effect if anything.
		if (instance != prim_instance || (state & (1u << STATE_BIT_SPRITE)) == 0)
			continue;

		auto &prim_bb = buffers.prim[base_primitive + i].bb;

		auto hazard_bb = ivec4(std::max<int>(bb.x, prim_bb.x),
		                       std::max<int>(bb.y, prim_bb.y),
		                       std::min<int>(bb.z, prim_bb.z),
		                       std::min<int>(bb.w, prim_bb.w));

		bool overlap = hazard_bb.x <= hazard_bb.z && hazard_bb.y <= hazard_bb.w &&
		               (state & ((1u << STATE_BIT_Z_TEST) | (1u << STATE_BIT_Z_WRITE))) != 0;

		if (overlap)
		{
			push.lo_primitive_index = base_primitive + next_lo_index;
			push.hi_primitive_index = base_primitive + i - 1;
			cmd.push_constants(&push, 0, sizeof(push));
			dispatch_cache_read_only_depth(cmd, rp, depth_psm, instance);
			dispatch_shading_commands(cmd, rp, instance, true,
			                          push.lo_primitive_index,
			                          push.hi_primitive_index - push.lo_primitive_index + 1);

			next_lo_index = i;
			overlap = false;
			bb = ivec4(prim_bb);
		}
		else
		{
			// Expand the BB.
			bb = ivec4(std::min<int>(bb.x, prim_bb.x),
			           std::min<int>(bb.y, prim_bb.y),
			           std::max<int>(bb.z, prim_bb.z),
			           std::max<int>(bb.w, prim_bb.w));
		}
	}

	if (next_lo_index < num_primitives)
	{
		push.lo_primitive_index = base_primitive + next_lo_index;
		push.hi_primitive_index = UINT32_MAX;
		cmd.push_constants(&push, 0, sizeof(push));
		dispatch_cache_read_only_depth(cmd, rp, depth_psm, instance);
		dispatch_shading_commands(cmd, rp, instance, false, push.lo_primitive_index,
		                          (num_primitives + base_primitive) - push.lo_primitive_index);
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

	uint32_t minimum_subgroup_size_log2 = inst.sampling_rate_x_log2 + inst.sampling_rate_y_log2;

	// Prefer Wave64 if we can get away with it.
	if (device->supports_subgroup_size_log2(true, 6, 6))
		cmd.set_subgroup_size_log2(true, 6, 6);
	else if (device->supports_subgroup_size_log2(true, minimum_subgroup_size_log2, 6))
		cmd.set_subgroup_size_log2(true, minimum_subgroup_size_log2, 6);

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
	uint32_t first_primitive_index_z_sensitive = UINT32_MAX;
	uint32_t last_primitive_index_single_step = 0;
	uint32_t last_primitive_index_z_sensitive = 0;
	bool single_primitive_step = false;

	if (fb_z_alias)
	{
		for (uint32_t i = 0; i < num_primitives; i++)
		{
			auto &state = buffers.prim[base_primitive + i].state;

			uint32_t prim_instance = (state >> STATE_VERTEX_RENDER_PASS_INSTANCE_OFFSET) &
			                         ((1u << STATE_VERTEX_RENDER_PASS_INSTANCE_COUNT) - 1u);

			if (prim_instance != instance)
				continue;

			// If we have duelling color and Z write we're in deep trouble, so have to fall back
			// to single primitive stepping.
			if ((state & (1u << STATE_BIT_Z_WRITE)) != 0)
			{
				single_primitive_step = true;
				last_primitive_index_single_step = i;
			}

			if ((state & ((1u << STATE_BIT_Z_TEST) | (1u << STATE_BIT_Z_WRITE))) != 0)
			{
				if (first_primitive_index_z_sensitive == UINT32_MAX)
					first_primitive_index_z_sensitive = i;
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

	if (bound_texture_has_array)
		variant_flags |= VARIANT_FLAG_HAS_TEXTURE_ARRAY_BIT;

	cmd.set_specialization_constant(5, variant_flags);

	assert(inst.sampling_rate_x_log2 <= 2);
	assert(inst.sampling_rate_y_log2 <= 3);

	cmd.set_program(shaders.ubershader[int(rp.feedback_color)][int(rp.feedback_depth)]);
	auto snap_raster_mask = ivec2(-(1 << inst.sampling_rate_y_log2));
	// Snap single-sampled raster to half-texel instead.
	if (inst.sampling_rate_y_log2 && field_aware_super_sampling && render_pass_instance_might_field_render(rp, instance))
		snap_raster_mask.y >>= 1;

	// If we're doing channel shuffle, but insist on super-sampling, we must make sure
	// to never do any resolve since that leads to a non-sensical result.
	// The best we can do is to demote to single samples, but declare that the super samples
	// are still valid so that we can keep using the super samples until we're forced to decay.
	ShadingDescriptor push = {
		snap_raster_mask,
		uint32_t(inst.channel_shuffle && inst.sampling_rate_y_log2 != 0),
		base_primitive,
		base_primitive + num_primitives - 1,
		0, // Offset filled in later.
	};

	// First, dispatch any work which is not reliant on Z at all.
	if (fb_z_alias && first_primitive_index_z_sensitive != 0)
	{
		push.lo_primitive_index = base_primitive;
		push.hi_primitive_index = base_primitive + first_primitive_index_z_sensitive - 1;
		cmd.push_constants(&push, 0, sizeof(push));
		dispatch_shading_commands(cmd, rp, instance, true, push.lo_primitive_index, first_primitive_index_z_sensitive);
	}

	if (single_primitive_step)
	{
		// Pure mayhem, game will rely on non-local feedback effects due to Z swizzling.
		// Render one primitive at a time.
		// Not much we can do about this other than render two tiles in sync in a single workgroup,
		// and then do barrier() after every primitive and exchange color and depth values as needed.
		// That however, is complete insanity, but if we end up seeing content that really hammers
		// this hard, we might not have a choice ...
		for (uint32_t i = first_primitive_index_z_sensitive; i <= last_primitive_index_single_step; i++)
		{
			push.lo_primitive_index = base_primitive + i;
			push.hi_primitive_index = base_primitive + i;
			cmd.push_constants(&push, 0, sizeof(push));
			dispatch_shading_commands(cmd, rp, instance, true, push.lo_primitive_index, 1);
		}

		if (last_primitive_index_single_step + 1 < num_primitives)
		{
			if (last_primitive_index_z_sensitive > last_primitive_index_single_step)
			{
				// If there is still work to do that is read-only, use cached depth for the rest of the render pass.
				dispatch_read_aliased_depth_passes(
						cmd, rp, instance, depth_psm, push,
						base_primitive, last_primitive_index_single_step + 1,
						num_primitives);
			}
			else
			{
				// If no further primitives need to read-depth, treat it as non-depth to speed up things.
				cmd.set_specialization_constant(3, UINT32_MAX);
				push.lo_primitive_index = base_primitive + last_primitive_index_single_step + 1;
				push.hi_primitive_index = UINT32_MAX;
				cmd.push_constants(&push, 0, sizeof(push));
				dispatch_shading_commands(cmd, rp, instance, false,
				                          push.lo_primitive_index,
				                          (base_primitive + num_primitives) - push.lo_primitive_index);
			}
		}
	}
	else if (fb_z_alias)
	{
		dispatch_read_aliased_depth_passes(
				cmd, rp, instance, depth_psm, push,
				base_primitive, first_primitive_index_z_sensitive, num_primitives);
	}
	else
	{
		cmd.push_constants(&push, 0, sizeof(push));
		dispatch_shading_commands(cmd, rp, instance, false, base_primitive, num_primitives);
	}

	cmd.end_region();
	cmd.enable_subgroup_size_control(false);
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

	// If we need to handle broken case of different BPP in local copies,
	// have to deal with it in the fallback path.
	if (is_fused_nibble && desc.trxdir.desc.XDIR == LOCAL_TO_LOCAL &&
	    get_bits_per_pixel(desc.bitbltbuf.desc.SPSM) != 4)
	{
		is_fused_nibble = false;
	}

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
#if 0
	cmd.set_storage_buffer(0, 4, *buffers.bug_feedback);
#endif

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

void GSRenderer::copy_vram(const CopyDescriptor &desc, const PageRect &damage_rect)
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

	if (desc.trxreg.desc.RRW + desc.trxpos.desc.DSAX > 2048 || desc.trxreg.desc.RRH + desc.trxpos.desc.DSAY > 2048)
	{
		// When there's copy wraparound, we don't get exact page tracking atm, so be conservative
		// since the writes will be scattered all over the place.
		for (auto &v : vram_copy_write_pages)
			v = UINT32_MAX;
	}
	else
	{
		for (uint32_t y = 0; y < damage_rect.page_height; y++)
		{
			for (uint32_t x = 0; x < damage_rect.page_width; x++)
			{
				uint32_t effective_page = damage_rect.base_page + y * damage_rect.page_stride + x;
				effective_page &= vram_size / PageSize - 1;
				vram_copy_write_pages[effective_page / 32] |= 1u << (effective_page & 31);
			}
		}
	}

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
	case FlushReason::HostAccess:
		return "HostAccess";
	default:
		return "";
	}
}

#ifdef PARALLEL_GS_DEBUG
static inline void sanitize_state_indices(const PrimitiveAttribute *prims, const RenderPass &rp)
{
	for (uint32_t i = 0; i < rp.num_primitives; i++)
	{
		uint32_t tex_index = prims[i].tex & ((1u << TEX_TEXTURE_INDEX_BITS) - 1u);
		uint32_t state_index = (prims[i].state >> STATE_INDEX_BIT_OFFSET) &
		                       ((1u << STATE_INDEX_BIT_COUNT) - 1u);

		if (state_index >= rp.num_states)
			std::terminate();

		if (tex_index >= 0x8000)
		{
			if (rp.feedback_mode == RenderPass::Feedback::None)
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
	if (a.z_sensitive && b.z_sensitive && (a.z_write || b.z_write) && page_rect_overlaps(a_z_rect, b_z_rect))
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

void GSRenderer::reserve_primitive_buffers(uint32_t num_primitives)
{
	reserve_attribute_scratch(num_primitives * 3 * sizeof(VertexPosition), buffers.pos_scratch);
	reserve_attribute_scratch(num_primitives * 3 * sizeof(VertexAttribute), buffers.attr_scratch);
	reserve_attribute_scratch(num_primitives * sizeof(PrimitiveAttribute), buffers.prim_scratch);

	{
		auto *mapped = static_cast<uint8_t *>(device->map_host_buffer(*buffers.pos_scratch.buffer,
		                                                              Vulkan::MEMORY_ACCESS_WRITE_BIT));
		buffers.pos = reinterpret_cast<VertexPosition *>(mapped + buffers.pos_scratch.offset);
	}

	{
		auto *mapped = static_cast<uint8_t *>(device->map_host_buffer(*buffers.attr_scratch.buffer,
		                                                              Vulkan::MEMORY_ACCESS_WRITE_BIT));
		buffers.attr = reinterpret_cast<VertexAttribute *>(mapped + buffers.attr_scratch.offset);
	}

	{
		auto *mapped = static_cast<uint8_t *>(device->map_host_buffer(*buffers.prim_scratch.buffer,
		                                                              Vulkan::MEMORY_ACCESS_WRITE_BIT));
		buffers.prim = reinterpret_cast<PrimitiveAttribute *>(mapped + buffers.prim_scratch.offset);
	}
}

VertexPosition *GSRenderer::get_reserved_vertex_positions() const
{
	return buffers.pos;
}

VertexAttribute *GSRenderer::get_reserved_vertex_attributes() const
{
	return buffers.attr;
}

PrimitiveAttribute *GSRenderer::get_reserved_primitive_attributes() const
{
	return buffers.prim;
}

void GSRenderer::ensure_clear_cmd()
{
	// Attempted async compute here for binning, etc, but it's not very useful in practice.
	bool first_clear_cmd = !clear_cmd;
	ensure_command_buffer(clear_cmd, Vulkan::CommandBuffer::Type::Generic);
	if (first_clear_cmd)
		clear_cmd->begin_region("clear-memory");
}

void GSRenderer::flush_rendering(const RenderPass &rp)
{
	if (rp.num_primitives == 0)
		return;
	assert(rp.num_primitives <= MaxPrimitivesPerFlush);

	// We didn't end up flushing indirect texture uploads before flushing the full render pass, so we're safe.
	pending_indirect_uploads.clear();
	pending_indirect_analysis.clear();

#ifdef PARALLEL_GS_DEBUG
	sanitize_state_indices(buffers.prim, rp);
#endif

	ensure_clear_cmd();

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

	if (!texture_analysis.empty())
	{
		if (device->consumes_debug_markers())
			begin_region(*binning_cmd, "Texture analysis %u", rp.label_key);
		dispatch_texture_analysis(*binning_cmd, rp);
		if (device->consumes_debug_markers())
			binning_cmd->end_region();
		texture_analysis.clear();
	}

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
					uint32_t instance = (buffers.prim[prim].state >> STATE_VERTEX_RENDER_PASS_INSTANCE_OFFSET) &
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
				// This really matters if we need to deal with workarounds of any kind like Color/Z aliasing.
				// Compute effective stepping range.

				uint32_t prim_lo[MaxRenderPassInstances];
				uint32_t prim_hi[MaxRenderPassInstances];

				for (uint32_t i = 0; i < rp.num_instances; i++)
				{
					prim_lo[i] = UINT32_MAX;
					prim_hi[i] = 0;
				}

				for (uint32_t prim = 0; prim < rp.num_primitives; prim++)
				{
					uint32_t instance = (buffers.prim[prim].state >> STATE_VERTEX_RENDER_PASS_INSTANCE_OFFSET) &
					                    ((1u << STATE_VERTEX_RENDER_PASS_INSTANCE_COUNT) - 1u);

					if (prim_lo[instance] == UINT32_MAX)
						prim_lo[instance] = prim;
					prim_hi[instance] = prim;
				}

				for (uint32_t i = 0; i < rp.num_instances; i++)
					if (prim_lo[i] != UINT32_MAX)
						flush_rendering(rp, i, prim_lo[i], prim_hi[i] - prim_lo[i] + 1);
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

Vulkan::ImageHandle GSRenderer::pull_image_handle_from_slab(
	uint32_t width, uint32_t height,
	uint32_t levels, uint32_t samples)
{
	if (!Util::is_pow2(width) || !Util::is_pow2(height) || levels > 7 ||
	    width > 1024 || height > 1024 || width == 0 || height == 0)
	{
		return {};
	}

	assert(levels == 1 || samples == 1);
	assert(levels > 0);

	uint32_t W = Util::floor_log2(width);
	uint32_t H = Util::floor_log2(height);
	auto &pool = samples > 1 ? super_sampled_recycled_image_pool[H][W] : recycled_image_pool[levels - 1][H][W];
	if (pool.empty())
		return {};

	auto res = std::move(pool.back());
	pool.pop_back();
	assert(total_image_slab_size >= (sizeof(uint32_t) << (W + H)) * res->get_create_info().layers);
	total_image_slab_size -= (sizeof(uint32_t) << (W + H)) * res->get_create_info().layers;
	return res;
}

void GSRenderer::move_image_handles_to_slab()
{
	for (auto &handle : recycled_image_handles)
	{
		uint32_t W = Util::floor_log2(handle->get_width());
		uint32_t H = Util::floor_log2(handle->get_height());
		uint32_t levels = handle->get_create_info().levels;
		assert(W <= 10 && H <= 10 && levels <= 7);
		total_image_slab_size += (sizeof(uint32_t) << (W + H)) * handle->get_create_info().layers;
		if (handle->get_create_info().layers > 1)
			super_sampled_recycled_image_pool[H][W].push_back(std::move(handle));
		else
			recycled_image_pool[levels - 1][H][W].push_back(std::move(handle));
	}
	recycled_image_handles.clear();

	if (total_image_slab_size > image_slab_high_water_mark)
	{
#ifdef PARALLEL_GS_DEBUG
		LOGW("New high watermark for image slab: %llu MiB.\n",
		     static_cast<unsigned long long>(total_image_slab_size / (1024 * 1024)));
#endif
		image_slab_high_water_mark = total_image_slab_size;
	}

	// If we end up exhausting this pool, just flush everything and start over, no need to be more clever about it.
	// Shouldn't happen in normal use.
	if (total_image_slab_size > max_image_slab_size)
	{
		flush_slab_cache();

#ifdef PARALLEL_GS_DEBUG
		LOGW("Image slab pool was exhausted, flushing it ...\n");
#endif
	}
}

void GSRenderer::flush_slab_cache()
{
	for (auto &l : recycled_image_pool)
		for (auto &y : l)
			for (auto &x : y)
				x.clear();

	for (auto &h : super_sampled_recycled_image_pool)
		for (auto &w : h)
			w.clear();

	total_image_slab_size = 0;
}

void GSRenderer::mark_clut_read(uint32_t clut_instance)
{
	// We only care about the latest upload w.r.t. usage.
	// A new CLUT upload will mark all older uploads as fully committed.
	if (clut_instance == next_clut_instance)
		last_clut_update_is_read = true;
}

bool PaletteUploadDescriptor::fully_replaces_clut_upload(const PaletteUploadDescriptor &old) const
{
	bool is_8bit = tex0.desc.PSM == PSMT8 || tex0.desc.PSM == PSMT8H;
	bool old_is_8bit = old.tex0.desc.PSM == PSMT8 || old.tex0.desc.PSM == PSMT8H;

	// Technically we could just check that new write just covers all of old's footprint, but that's too cumbersome.
	// A straight equal check is sufficient.
	return old.tex0.desc.CPSM == tex0.desc.CPSM &&
	       old.tex0.desc.CSA == tex0.desc.CSA &&
	       is_8bit == old_is_8bit;
}

void GSRenderer::rewind_clut_instance(uint32_t index)
{
	assert(palette_uploads.empty());

	if (index != next_clut_instance)
	{
		PageRectCLUT clut = {};
		clut.csa_mask = UINT32_MAX;
		tracker.register_cached_clut_clobber(clut);
		tracker.invalidate_texture_cache(index);
	}

	base_clut_instance = index;
	next_clut_instance = index;
}

uint32_t GSRenderer::update_palette_cache(const PaletteUploadDescriptor &desc)
{
	if (!last_clut_update_is_read && !palette_uploads.empty() &&
	    desc.fully_replaces_clut_upload(palette_uploads.back()))
	{
		uint32_t old_incoming = palette_uploads.back().incoming_clut_instance;
		palette_uploads.back() = desc;
		palette_uploads.back().incoming_clut_instance = old_incoming;
	}
	else
	{
		next_clut_instance = (next_clut_instance + 1) % CLUTInstances;
		palette_uploads.push_back(desc);
		stats.num_palette_updates++;
		check_flush_stats();
	}

	last_clut_update_is_read = false;
	return next_clut_instance;
}

void GSRenderer::dispatch_texture_analysis(Vulkan::CommandBuffer &cmd, const RenderPass &rp)
{
	cmd.set_program(shaders.sampler_feedback);
	memcpy(cmd.allocate_typed_constant_data<TextureAnalysis>(1, 0, texture_analysis.size()),
	       texture_analysis.data(),
	       texture_analysis.size() * sizeof(TextureAnalysis));

	memcpy(cmd.allocate_typed_constant_data<StateVector>(
			       0, BINDING_STATE_VECTORS, std::max<uint32_t>(1, rp.num_states)),
	       rp.states, rp.num_states * sizeof(StateVector));

	VK_ASSERT(rp.num_textures <= MaxTextures);
	auto *tex_infos = cmd.allocate_typed_constant_data<TexInfo>(
			0, BINDING_TEXTURE_INFO, std::max<uint32_t>(1, rp.num_textures));
	for (uint32_t i = 0; i < rp.num_textures; i++)
		tex_infos[i] = rp.textures[i].info;

	uint32_t ssaa_sample_offset = 0;
	for (uint32_t i = 0; i < rp.num_instances; i++)
	{
		uint32_t max_local_coord = 1u << rp.instances[i].sampling_rate_y_log2;
		max_local_coord--;
		max_local_coord <<= (PGS_RASTER_SUBSAMPLE_BITS - rp.instances[i].sampling_rate_y_log2);
		ssaa_sample_offset = std::max<uint32_t>(ssaa_sample_offset, max_local_coord);
	}

	struct
	{
		uint32_t num_primitives;
		uint32_t num_textures;
		uint32_t ssaa_sample_offset;
	} push = { rp.num_primitives, uint32_t(texture_analysis.size()), ssaa_sample_offset };
	cmd.push_constants(&push, 0, sizeof(push));

	uint32_t indirect_data[] = { (rp.num_primitives + 255) / 256, 1, 1, 0 };
	VkDeviceSize indirect_offset = allocate_device_scratch(sizeof(indirect_data), buffers.rebar_scratch, indirect_data);
	cmd.dispatch_indirect(*buffers.rebar_scratch.buffer, indirect_offset);
	// In case we have to cancel the analysis late and fallback.
	pending_indirect_analysis.push_back({ buffers.rebar_scratch.buffer, indirect_offset });
}

void GSRenderer::upload_texture(const TextureUpload &upload)
{
	auto &desc = upload.desc;
	auto &img = *upload.image;
	auto &scratch = upload.scratch;
	auto &cmd = *direct_cmd;

	uint32_t levels = img.get_create_info().levels;
	cmd.set_program(shaders.upload[int(upload.desc.samples > 1)]);
	if (scratch.buffer)
		cmd.set_storage_buffer(0, 0, *scratch.buffer, scratch.offset, scratch.size);
	else
		cmd.set_storage_buffer(0, 0, *buffers.gpu);
	cmd.set_storage_buffer(0, BINDING_CLUT, *buffers.clut);

	if (upload.indirection.buffer)
		cmd.set_storage_buffer(0, 2, *upload.indirection.buffer, upload.indirection.offset, upload.indirection.size);
	else
		cmd.set_storage_buffer(0, 2, *buffers.gpu);

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

	cmd.set_specialization_constant_mask(0xf);
	cmd.set_specialization_constant(0, uint32_t(desc.tex0.desc.PSM));
	cmd.set_specialization_constant(2, uint32_t(desc.tex0.desc.CPSM));
	cmd.set_specialization_constant(3, bool(upload.indirection.buffer));

	// Makes it feasible to handle REGION_CLAMP where we only access one page, but at an offset.
	if (scratch.buffer)
		cmd.set_specialization_constant(1, PageSize - 1);
	else
		cmd.set_specialization_constant(1, vram_size - 1);

	for (uint32_t level = 0; level < levels; level++)
	{
		info.addr_block = table[level].addr;
		info.stride_block = table[level].stride;
		info.width = img.get_width(level);
		info.height = img.get_height(level);

		if (device->consumes_debug_markers())
		{
			insert_label(cmd,
			             "%s mip %u%s - 0x%x - %s - %u x %u (stride %u) + (%u, %u) - CPSM %s - CSA %u - bank %u / %u - %016llx",
			             (scratch.buffer ? "CPU" : "GPU"),
			             level, upload.desc.samples > 1 ? " (SSAA)" : "",
			             info.addr_block * PGS_BLOCK_ALIGNMENT_BYTES,
			             psm_to_str(uint32_t(desc.tex0.desc.PSM)),
			             info.width, info.height, info.stride_block * PGS_BUFFER_WIDTH_SCALE,
			             info.off_x, info.off_y,
			             psm_to_str(uint32_t(desc.tex0.desc.CPSM)),
			             uint32_t(desc.tex0.desc.CSA),
			             desc.palette_bank, desc.latest_palette_bank,
			             static_cast<unsigned long long>(desc.hash));

			insert_label(cmd, "  AEM: %u, TA0: 0x%x, TA1: 0x%x", info.aem, info.ta0, info.ta1);
		}

		cmd.set_storage_texture_level(0, 1, img.get_view(), level);
		cmd.push_constants(&info, 0, sizeof(info));

		if (upload.indirection.buffer)
			cmd.dispatch_indirect(*upload.indirection.indirect, upload.indirection.indirect_offset);
		else
			cmd.dispatch((info.width + 7) / 8, (info.height + 7) / 8, img.get_create_info().layers);

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
		clut_desc.csm2_x_bias = upload.csm2_x_bias;
		clut_desc.csm2_x_scale = upload.csm2_x_scale;
		clut_desc.csm1_reference_base = upload.csm1_reference_base;
		clut_desc.csm1_mask = upload.csm1_mask;

		if (!clut_desc.csm)
		{
			// CSM1
			clut_desc.co_uv = 0;
			clut_desc.cbw = 0;
		}
#ifdef PARALLEL_GS_DEBUG
		else
		{
			// CSM2
			if (clut_desc.csa != 0)
				LOGW("CSM2: CSA is not 0.\n");
		}
#endif

		instance = (instance + 1) % CLUTInstances;
		*desc++ = clut_desc;
	}

	struct Push
	{
		uint32_t count;
		uint32_t read_index;
		uint32_t wg_offset;
	};

	cmd.begin_region("flush-palette-upload");

	if (device->consumes_debug_markers())
	{
		for (size_t i = 0, n = palette_uploads.size(); i < n; i++)
		{
			auto &upload = palette_uploads[i];
			insert_label(cmd, "Bank %u - 0x%x - %s - %u colors - CSA %u - CSM %u - COU/V %u, %u - S/B %.3f, %u, incoming %u",
			             uint32_t(base_clut_instance + 1 + i) % CLUTInstances,
			             uint32_t(upload.tex0.desc.CBP) * PGS_BLOCK_ALIGNMENT_BYTES,
			             psm_to_str(uint32_t(upload.tex0.desc.CPSM)),
			             ((uint32_t(upload.tex0.desc.PSM) == PSMT8 ||
			               uint32_t(upload.tex0.desc.PSM) == PSMT8H) ? 256 : 16),
			             uint32_t(upload.tex0.desc.CSA),
			             uint32_t(upload.tex0.desc.CSM),
			             upload.texclut.desc.COU * 16,
			             upload.texclut.desc.COV,
			             upload.csm2_x_scale, upload.csm2_x_bias, upload.incoming_clut_instance);
		}
	}

	Vulkan::QueryPoolHandle start_ts, end_ts;
	if (enable_timestamps)
		start_ts = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	Push push = {};
	push.read_index = palette_uploads[0].incoming_clut_instance;

	size_t next_begin_index = 0;

	for (size_t i = 0, n = palette_uploads.size(); i < n; i++)
	{
		auto &clut = palette_uploads[i].tex0.desc;

		uint32_t write_clut_index = (base_clut_instance + 1 + i) % CLUTInstances;
		bool full_replacement = clut.CSA == 0 && clut.CPSM == PSMCT32 && get_bits_per_pixel(clut.PSM) == 8;
		bool out_of_order_read = (palette_uploads[i].incoming_clut_instance + 1) % CLUTInstances != write_clut_index;

		assert(write_clut_index != palette_uploads[i].incoming_clut_instance);

		if (full_replacement || out_of_order_read)
		{
			if (i != next_begin_index)
			{
				// Flush all pending work.
				push.count = uint32_t(i - next_begin_index);
				push.wg_offset = next_begin_index;
				cmd.push_constants(&push, 0, sizeof(push));
				if (device->consumes_debug_markers())
					insert_label(cmd, "Split [%u, %u)", push.wg_offset, push.wg_offset + push.count);
				cmd.dispatch(1, 1, 1);

				if (out_of_order_read)
				{
					cmd.barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
					            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
				}
			}

			next_begin_index = i;
		}

		if (full_replacement)
		{
			// This will overwrite the full CLUT, so can split dispatch.
			// The next batch will start with a full CLUT clear, so no need to read from CLUT.
			push.read_index = UINT32_MAX;
		}
		else if (out_of_order_read)
		{
			push.read_index = palette_uploads[i].incoming_clut_instance;
			insert_label(cmd, "Hazard - incoming %u", push.read_index);
		}
	}

	push.count = uint32_t(palette_uploads.size() - next_begin_index);
	push.wg_offset = next_begin_index;
	cmd.push_constants(&push, 0, sizeof(push));
	if (device->consumes_debug_markers())
		insert_label(cmd, "Split [%u, %u)", push.wg_offset, push.wg_offset + push.count);
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
	tracker.clear_cache_pages();

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
		upload_texture(upload);

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

void GSRenderer::mark_shadow_page_sync(uint32_t page_index)
{
	sync_vram_shadow_pages[page_index / 32u] |= 1u << (page_index & 31u);
}

void GSRenderer::flush_transfer()
{
	tracker.clear_copy_pages();
	total_stats.num_copy_threads += stats.num_copy_threads;
	stats.num_copy_threads = 0;

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
		cmd.fill_buffer(*buffers.vram_copy_atomics, 0, 0, PGS_VALID_PAGE_COPY_WRITE_OFFSET);
		// Reset hazard exist bitfield.
		cmd.fill_buffer(*buffers.vram_copy_atomics, 0, PGS_LINKED_VRAM_COPY_WRITE_LIST_OFFSET + vram_size, vram_size / 32);

#if 0
		// For safety reasons, make absolutely sure it's safe to traverse the linked list.
		assert(PGS_VALID_PAGE_COPY_WRITE_OFFSET + vram_copy_write_pages.size() * sizeof(uint32_t) <= PGS_LINKED_VRAM_COPY_WRITE_LIST_OFFSET);
		cmd.update_buffer_inline(*buffers.vram_copy_atomics, PGS_VALID_PAGE_COPY_WRITE_OFFSET,
		                         vram_copy_write_pages.size() * sizeof(uint32_t), vram_copy_write_pages.data());
#endif

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
                                     const SamplingRect &rect, uint32_t super_samples,
                                     const Vulkan::Image *promoted)
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
	cmd.set_program(sample_quad[promoted ? 1 : 0]);
	if (promoted)
		cmd.set_texture(0, 0, promoted->get_view());
	else
		cmd.set_storage_buffer(0, 0, *buffers.gpu);

	auto valid_extent = rect.valid_extent;
	if (super_samples > 1)
	{
		valid_extent.width *= 2;
		valid_extent.height *= 2;
	}
	cmd.set_scissor({{ 0, 0 }, valid_extent });

	cmd.set_specialization_constant_mask(0x7);
	cmd.set_specialization_constant(0, uint32_t(dispfb.PSM));
	cmd.set_specialization_constant(1, vram_size - 1);
	cmd.set_specialization_constant(2, super_samples);

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
                                                          const DISPLAYBits &display, bool force_progressive,
                                                          const Vulkan::Image *promoted)
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

	if (promoted)
	{
		DW = promoted->get_width() * MAGH;
		DH = promoted->get_height() * MAGV;
	}

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

bool GSRenderer::scanout_is_interlaced(const PrivRegisterState &priv, const VSyncInfo &info) const
{
	bool force_progressive = info.force_progressive;
	bool is_interlaced = priv.smode2.INT;
	bool alternative_sampling = is_interlaced && !priv.smode2.FFMD;
	if (alternative_sampling && force_progressive)
		is_interlaced = false;
	return is_interlaced;
}

bool GSRenderer::vsync_can_skip(const PrivRegisterState &priv, const VSyncInfo &info) const
{
	return !scanout_is_interlaced(priv, info);
}

ScanoutResult GSRenderer::vsync(const PrivRegisterState &priv, const VSyncInfo &info,
                                uint32_t sampling_rate_x_log2, uint32_t sampling_rate_y_log2,
                                const Vulkan::Image *promoted1, const Vulkan::Image *promoted2)
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
	bool high_resolution_scanout = info.high_resolution_scanout;
	bool is_interlaced = scanout_is_interlaced(priv, info);
	bool force_deinterlace = !high_resolution_scanout &&
	                         (priv.smode2.FFMD && priv.smode1.CMOD != SMODE1Bits::CMOD_PROGRESSIVE);
	bool alternative_sampling = priv.smode2.INT && !priv.smode2.FFMD;

	// We have to scan out tightly packed fields or upscaling breaks.
	if (!force_progressive || priv.extwrite.WRITE ||
	    !sampling_rate_x_log2 || !sampling_rate_y_log2 ||
	    force_deinterlace)
	{
		high_resolution_scanout = false;
	}

	bool field_aware_rendering = high_resolution_scanout &&
	                             sampling_rate_y_log2 &&
	                             priv.smode2.FFMD &&
	                             priv.smode1.CMOD != SMODE1Bits::CMOD_PROGRESSIVE;

	uint32_t super_samples = 1;
	if (high_resolution_scanout)
		super_samples = 1 << (sampling_rate_x_log2 + sampling_rate_y_log2);

	if (promoted1 && promoted1->get_create_info().layers < super_samples)
		promoted1 = nullptr;
	if (promoted2 && promoted2->get_create_info().layers < super_samples)
		promoted2 = nullptr;

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

		// Scanning out high-res with these resolutions is somewhat bogus.
		high_resolution_scanout = false;
		field_aware_rendering = false;
		super_samples = 1;
	}
	else
	{
		LOGE("Unknown video format.\n");
		cmd.end_region();
		flush_submit(0);
		return {};
	}

	// It's possible the input had higher resolution.
	// Make a fake mode in this case that has more lines to compensate.
	if (promoted1)
		mode_height = std::max<uint32_t>(mode_height, promoted1->get_height());
	if (promoted2)
		mode_height = std::max<uint32_t>(mode_height, promoted2->get_height());

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

		auto rect = compute_circuit_rect(priv, phase, priv.display1, force_progressive, promoted1);
		image_info.width = rect.image_extent.width;
		image_info.height = rect.image_extent.height;

		if (image_info.width && image_info.height)
		{
			if (high_resolution_scanout)
			{
				image_info.width *= 2;
				image_info.height *= 2;
			}
			circuit1 = device->create_image(image_info);
			sample_crtc_circuit(cmd, *circuit1, priv.dispfb1, rect, super_samples, promoted1);
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

		auto rect = compute_circuit_rect(priv, phase, priv.display2, force_progressive, promoted2);

		image_info.width = rect.image_extent.width;
		image_info.height = rect.image_extent.height;

		if (image_info.width && image_info.height)
		{
			if (high_resolution_scanout)
			{
				image_info.width *= 2;
				image_info.height *= 2;
			}
			circuit2 = device->create_image(image_info);
			sample_crtc_circuit(cmd, *circuit2, priv.dispfb2, rect, super_samples, promoted2);
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

	uint32_t real_mode_width = mode_width;

	if (info.adapt_to_internal_horizontal_resolution)
	{
		uint32_t horiz_resolution0 = circuit1 ? circuit1->get_width() : 0;
		uint32_t horiz_resolution1 = circuit2 ? circuit2->get_width() : 0;

		// Need to do all the CRTC offset math in single sampled domain to avoid lots of confusing cases later.
		if (high_resolution_scanout)
		{
			horiz_resolution0 >>= 1;
			horiz_resolution1 >>= 1;
		}

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
	result.high_resolution_scanout = high_resolution_scanout;

	if (info.raw_circuit_scanout &&
	    !info.crtc_offsets && !info.overscan &&
	    info.adapt_to_internal_horizontal_resolution &&
	    !force_deinterlace && !is_interlaced && !priv.extwrite.WRITE)
	{
		auto effective_mode_width = mode_width;
		auto effective_mode_height = mode_height;

		if (high_resolution_scanout)
		{
			effective_mode_width *= 2;
			effective_mode_height *= 2;
		}

		bool is_raw_circuit1 =
				circuit1 && !circuit2 && MMOD == PMODEBits::MMOD_ALPHA_ALP && ALP == 0xff &&
				circuit1->get_width() <= effective_mode_width && circuit1->get_height() <= effective_mode_height;
		bool is_raw_circuit2 =
				circuit2 && !circuit1 && SLBG == PMODEBits::SLBG_ALPHA_BLEND_CIRCUIT2 &&
				circuit2->get_width() <= effective_mode_width && circuit2->get_height() <= effective_mode_height;

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

			result.internal_width = result.image->get_width() >> int(high_resolution_scanout);
			result.internal_height = result.image->get_height() >> int(high_resolution_scanout);

			flush_submit(0);
			return result;
		}
	}

	result.internal_width = mode_width;
	result.internal_height = mode_height;
	image_info.width = mode_width << int(high_resolution_scanout);
	image_info.height = mode_height << int(high_resolution_scanout);

	if (field_aware_rendering)
	{
		// Avoid a one line flicker due to field rendering.
		image_info.height--;
	}

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

			if (high_resolution_scanout)
			{
				vp.x *= 2.0f;
				vp.y *= 2.0f;
				vp.width *= 2.0f;
				vp.height *= 2.0f;

				if (field_aware_rendering && !info.phase)
					vp.y -= 1.0f;
			}

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

			if (high_resolution_scanout)
			{
				vp.x *= 2.0f;
				vp.y *= 2.0f;
				vp.width *= 2.0f;
				vp.height *= 2.0f;

				if (field_aware_rendering && !info.phase)
					vp.y -= 1.0f;
			}

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

		// It seems like we should not use clock divider here ... ?
		push.resolution.x = priv.extdata.WW + 1;
		push.resolution.y = priv.extdata.WH + 1;
		push.wdx = priv.extbuf.WDX;
		push.wdy = priv.extbuf.WDY;
		push.exbp = priv.extbuf.EXBP;
		push.exbw = priv.extbuf.EXBW;
		push.wffmd = priv.extbuf.WFFMD;
		push.emoda = priv.extbuf.EMODA;
		push.emodc = priv.extbuf.EMODC;

		// When WFFMD is 0, it seems like we should treat it as number of lines to write
		// in an interlaced fashion. Just doubling the output resolution to fill in the blanks should work?
		if (!priv.extbuf.WFFMD)
			push.resolution.y *= 2;

		cmd.set_storage_buffer(0, 0, *buffers.gpu);
		const Vulkan::ImageView *view = nullptr;

		uint32_t scanout_width = real_mode_width;
		uint32_t scanout_height = merged->get_height();

		if (priv.extbuf.FBIN == 0)
			view = &merged->get_view();
		else if (circuit2)
			view = &circuit2->get_view();
		else
			view = &circuit1->get_view();

		// Not sure what should happen if we haven't enabled the output circuit.
		if (view)
		{
			// I've seen cases of WW/WH assuming that we need a clock divider, and other cases where games don't ...
			// Most likely the missing clock-divider is just a game bug and the hardware stops doing feedback writes
			// when the scanout is done. So the correct solution, might simply be to clamp the resolution.
			uint32_t max_feedback_width =
					(scanout_width * clock_divider + priv.extdata.SMPH) / (priv.extdata.SMPH + 1);
			uint32_t max_feedback_height =
					(scanout_height + priv.extdata.SMPV) / (priv.extdata.SMPV + 1);

			push.resolution.x = std::min<uint32_t>(push.resolution.x, max_feedback_width);
			push.resolution.y = std::min<uint32_t>(push.resolution.y, max_feedback_height);

			auto write_rect = compute_page_rect(priv.extbuf.EXBP, priv.extbuf.WDX, priv.extbuf.WDY,
			                                    push.resolution.x, push.resolution.y,
			                                    priv.extbuf.EXBW, PSMCT32);
			tracker.mark_external_write(write_rect);

			// TODO: Consider SX, SY.
			push.uv_base = vec2(0.5f) / vec2(push.resolution);
			push.uv_scale.x = float(priv.extdata.SMPH + 1) / float(scanout_width * clock_divider);
			push.uv_scale.y = float(priv.extdata.SMPV + 1) / float(scanout_height);
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

	if (!high_resolution_scanout && (is_interlaced || force_deinterlace))
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
		for (auto &field : vsync_last_fields)
			field.reset();
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
