// SPDX-FileCopyrightText: 2026 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-License-Identifier: LGPL-3.0+

#pragma once

#include "device.hpp"
#include "image.hpp"
#include "transforms.hpp"
#include "math.hpp"
#include <stdint.h>
#include "shaders/slangmosh_iface.hpp"

namespace ParallelGS
{
class AnalogVideoFilter
{
public:
	// Changes carrier frequency and modulation strategy.
	enum class System { NTSC, PAL };
	enum class Cable { Composite = 0, SVideo = 1, Component = 2 };

	struct Options
	{
		System system = System::NTSC;
		Cable cable = Cable::Composite;
	};

	bool init(Vulkan::Device &device, const Options &options);

	struct FilterOptions
	{
		// Standard BT.601.
		// Should not be changed unless the input image is scaled horizontally.
		// Normal 640 width output from parallel-gs is assumed to be standard BT.601 13.5 MHz.
		// For best results, should be 56 MHz divided by an integer.
		// input_sampling_rate_mhz = 56 / (MAGH + 1) for PS2.
		float input_sampling_rate_mhz = 13.5f;

		// Adds a temporary offset to line counter. Useful for testing.
		// Otherwise, this should be 0 to let the subcarrier work as intended.
		uint64_t total_line_offset = 0;

		uint32_t phase = 0; // Interlacing phase.
		bool line_comb = true;

		// For comb filter, skip the additional notch filter to clean up chroma rejection
		// artifacts. TODO: Maybe make it adaptive somehow ...
		bool skip_notch = false;

		// Where the final image will be consumed.
		VkImageLayout dst_layout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
		VkPipelineStageFlags2 dst_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
		VkAccessFlags2 dst_access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
	};

	void run_filter(Vulkan::CommandBuffer &cmd, const Vulkan::ImageView &input, const FilterOptions &options);

	void reset_line_counter(uint64_t value);

	// Y resolution is the same as input resolution.
	// X resolution is standard 720 horizontal active pixels, but doubled to bandlimited 1440 pixels.
	// Output is in linear light with the appropriate primaries for the system in question.
	const Vulkan::ImageView &get_output() const;

	const Options &get_options() const { return options; }

private:
	Vulkan::Device *device = nullptr;
	Options options = {};
	Vulkan::ImageHandle downsample_target;
	Vulkan::ImageHandle encode_target;
	Vulkan::ImageHandle dummy_1d_array;
	Vulkan::ImageHandle decode_target;
	Vulkan::ImageHandle bandpass_target;
	Vulkan::ImageHandle chroma_estimate_target;
	uint64_t total_line_counter = 0;
	Analog::Shaders<> shaders;
};

class CRTFilter
{
public:
	bool init(Vulkan::Device &device);

	enum class Primaries
	{
		NTSC_Legacy, // Legacy saturated primaries from 1953
		BT601_525, // SMPTE C, "modern" NTSC as defined by SMPTE
		BT601_625, // PAL
		BT709, // sRGB displays
		BT2020, // HDR10
	};

	struct FilterOptions
	{
		bool hdr10 = false;
		bool progressive = false;
		uint32_t phase = 0; // 0 = top, 1 = bottom. Irrelevant for progressive.
		Primaries phosphor_primaries = Primaries::BT709;

		// Where the final image will be consumed.
		VkImageLayout dst_layout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
		VkPipelineStageFlags2 dst_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
		VkAccessFlags2 dst_access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

		float hdr10_target_max_cll = 600.0f;
		float hdr10_target_paper_white = 203.0f;

		float input_strength = 1.0f;
		float feedback = 0.05f;

		// Normalized. Use to only sample part of the input to remove e.g. overscan.
		struct
		{
			float x = 0.0f;
			float y = 0.0f;
			float width = 1.0f;
			float height = 1.0f;
		} input_rect;
	};

	// Runs prepasses.
	bool run_filter_prepass(Vulkan::CommandBuffer &cmd, const Vulkan::ImageView &view,
	                        const FilterOptions &filter_options,
	                        uint32_t output_width, uint32_t output_height);

	// Renderpass must be active. Scales and encodes OETF straight to "backbuffer".
	// UNORM type is expected. For SDR, assumes 2.2 gamma with BT709 primaries, otherwise PQ for HDR10 with BT2020.
	bool run_filter_encode(Vulkan::CommandBuffer &cmd, const FilterOptions &filter_options);

	static const Granite::Primaries &get_primaries(Primaries primaries);
	static muglm::mat3 generate_primary_conversion(const Granite::Primaries &output, const Granite::Primaries &input);

private:
	void init_buffers(Vulkan::Device &device, const Vulkan::ImageView &input_view, uint32_t output_width, uint32_t output_height);
	uint32_t input_width = 0, input_height = 0;
	Vulkan::ImageHandle phosphor_layer_front;
	Vulkan::ImageHandle phosphor_layer_back;
	Vulkan::ImageHandle bloomed;
	Vulkan::ImageHandle sinc_vert;
	bool back_is_valid = false;
	bool front_is_valid = false;
	Vulkan::Program *scan = nullptr;
	Vulkan::Program *bloom = nullptr;
	Vulkan::Program *sinc[2] = {};
};

}