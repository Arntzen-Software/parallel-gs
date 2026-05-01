// SPDX-FileCopyrightText: 2026 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-License-Identifier: LGPL-3.0+

#include "analog_video.hpp"
#include "muglm/matrix_helper.hpp"
#include "muglm/muglm_impl.hpp"
#include "shaders/slangmosh.hpp"

namespace ParallelGS
{
using namespace Vulkan;
using namespace Granite;

static constexpr uint32_t BaseOutputResolution = 720;

bool AnalogVideoFilter::init(Device &device_, const Options &options_)
{
	device = &device_;
	options = options_;

	ImageCreateInfo image_info = {};
	// 1D array makes more sense from cache PoV since we process scanlines, not 2D images.
	image_info.type = VK_IMAGE_TYPE_1D;
	image_info.domain = ImageDomain::Physical;
	image_info.width = BaseOutputResolution * 2 + 64; // Some extra room for convolution trails.
	image_info.height = 1;
	image_info.layers = 625;
	image_info.format = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
	image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

	downsample_target = device->create_image(image_info);
	device->set_name(*downsample_target, "downsample-target");

	if (options.cable != Cable::Component)
	{
		image_info.format = options.cable == Cable::SVideo ? VK_FORMAT_R16G16_SFLOAT : VK_FORMAT_R16_SFLOAT;
		encode_target = device->create_image(image_info);
		device->set_name(*encode_target, "encode-target");

		bandpass_target = device->create_image(image_info);
		device->set_name(*bandpass_target, "bandpass-target");

		chroma_estimate_target = device->create_image(image_info);
		device->set_name(*chroma_estimate_target, "chroma-estimate");
	}

	image_info.width = 1;
	image_info.layers = 1;
	image_info.misc = IMAGE_MISC_FORCE_ARRAY_BIT;
	image_info.initial_layout = VK_IMAGE_LAYOUT_GENERAL;
	image_info.layout = ImageLayout::General;
	image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	dummy_1d_array = device->create_image(image_info);
	device->set_name(*dummy_1d_array, "dummy-1d-array");

	ResourceLayout layout;
	shaders = Analog::Shaders<>(*device, layout, 0);

	// Encode output on-demand.
	return true;
}

void AnalogVideoFilter::reset_line_counter(uint64_t value)
{
	total_line_counter = value;
}

static void storage_to_sampled_barrier(CommandBuffer &cmd, const Image &img,
                                       VkPipelineStageFlags2 extra_stages = VK_PIPELINE_STAGE_NONE)
{
	cmd.image_barrier(img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
	                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | extra_stages, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

void AnalogVideoFilter::run_filter(CommandBuffer &cmd, const ImageView &input,
                                   const FilterOptions &filter_options)
{
	// Out of spec. This is probably not an analog signal.
	if (input.get_view_height() > 625)
		return;

	if (!decode_target || decode_target->get_height() != input.get_view_height())
	{
		ImageCreateInfo image_info = {};
		// 1D array makes more sense from cache PoV since we process scanlines, not 2D images.
		image_info.type = VK_IMAGE_TYPE_2D;
		image_info.domain = ImageDomain::Physical;
		image_info.width = BaseOutputResolution * 2;
		image_info.height = input.get_view_height();
		image_info.format = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
		image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		decode_target = device->create_image(image_info);
		device->set_name(*decode_target, "decode-target");
	}

	int normalized_input_width = int(float(input.get_view_width()) * (13.5f / filter_options.input_sampling_rate_mhz) + 0.5f);
	// We need to center the image as best we can against the 720 pixel BT.601 container.

	// During the filter process we have to shift the image by some pixels.
	// Since the final output is at double BT.601 rate to make CRT filtering nicer,
	// skip the / 2 since it gets cancelled out.
	int horizontal_shift = int(BaseOutputResolution) - normalized_input_width;

	// Out of spec. More than 720 active pixels per line is not allowed.
	if (horizontal_shift < 0)
		return;

	struct
	{
		int32_t input_offset;
		int32_t output_offset;
		float subcarrier_phases_per_pixel; // For ENCODE pass.
		float subcarrier_phases_per_line; // For ENCODE pass.
		float subcarrier_phase_offset;
		float input_horiz_scale; // For DOWNSAMPLE pass.
		int32_t line_phase; // For PAL. V flips polarity every line.
		int32_t height; // For chroma reconstruction filter.
	} push = {};

	cmd.set_program(shaders.composite);
	cmd.set_specialization_constant_mask(0x1f);
	// Controls the filters. PAL has a bit more bandwidth.
	cmd.set_specialization_constant(1, options.system == System::PAL);
	cmd.set_specialization_constant(2, 0);
	cmd.set_specialization_constant(3, false);

	enum
	{
		DownsamplingFilterOffset = 8, // 16-tap FIR
		EncodeOffset = 15, // 31-tap FIR
		DecodeOffset = 31, // 63-tap chroma
		BandpassOffset = 15, // 31-tap FIR
	};

	// PS2 CRTC subpixel clock is 4x BT.601 it seems.
	push.input_offset = -DownsamplingFilterOffset;
	push.input_horiz_scale = filter_options.input_sampling_rate_mhz / (13.5f * 4.0f);
	push.height = int(input.get_view_height());
	push.subcarrier_phases_per_pixel =
			(options.system == System::NTSC ? (315.0f / 88.0f) : 4.43361875f) /
			(13.5f * 2.0f);

	auto line_counter = total_line_counter + filter_options.total_line_offset;

	if (options.cable != Cable::Component)
	{
		if (options.system == System::NTSC)
		{
			// NTSC scanlines flip phase every scanline. Very easy to deal with.
			push.subcarrier_phase_offset = (line_counter & 1) ? 0.5f : 0.0f;
			push.subcarrier_phases_per_line = -0.5f /* 227.5f but wrapped to make FP math more accurate */;
		}
		else
		{
			// PAL is more ... complicated.
			// Each scanline is 283.75f subcarrier cycles, but PAL adds a +25 Hz bias to the subcarrier
			// which was designed to mitigate Hanover Bars (how phase shifts manifest in PAL).
			// Prefer negative phase so that per-pixel offset creates values close-ish to 0.
			push.subcarrier_phases_per_line = float(muglm::fract(283.75 + 1.0 / 625.0) - 1.0);

			// The pattern repeats after 2500 lines (8 fields).
			// Avoid terrible FP precision when phase gets very large near the end of the cycle.
			push.subcarrier_phase_offset = float(muglm::fract(double(line_counter % 2500) * push.subcarrier_phases_per_line));
		}
	}

	push.line_phase = int(line_counter & 1);

	// The exact specifics shouldn't matter too much, just need some way to nudge the carrier phase.
	auto odd_lines = options.system == System::NTSC ? 263 : 262;
	auto even_lines = options.system == System::NTSC ? 313 : 312;
	total_line_counter += (filter_options.phase & 1) ? odd_lines : even_lines;

	cmd.set_unorm_texture(0, 0, input);
	cmd.set_texture(0, 1, dummy_1d_array->get_view());
	cmd.set_storage_texture(0, 2, downsample_target->get_view());
	cmd.set_storage_texture(0, 3, get_output());
	cmd.set_texture(0, 4, dummy_1d_array->get_view());

	const Image *images[] = {
		downsample_target.get(),
		encode_target.get(),
		decode_target.get(),
		bandpass_target.get(),
		chroma_estimate_target.get(),
	};

	cmd.begin_barrier_batch();
	for (auto *img : images)
	{
		if (img)
		{
			auto src_stage = img == decode_target.get() ? filter_options.dst_stage : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
			cmd.image_barrier(*img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
			                  src_stage, 0,
			                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
		}
	}
	cmd.end_barrier_batch();

	enum Pass
	{
		PassDownsample = 0,
		PassEncode,
		PassDecode,
		PassBandpass,
		PassSeparation
	};

	static const char *pass_to_str[] = {
		"analog-downsample",
		"analog-encode",
		"analog-decode",
		"analog-bandpass",
		"analog-separation",
	};

	const auto run_pass = [&](Pass pass, const ImageView *src, const ImageView *dst)
	{
		cmd.begin_region(pass_to_str[pass]);
		cmd.push_constants(&push, 0, sizeof(push));

		if (src)
			cmd.set_texture(0, 1, *src);

		uint32_t groups_x;

		if (dst)
		{
			cmd.set_storage_texture(0, 2, *dst);
			groups_x = (dst->get_view_width() + 255) / 256;
		}
		else
		{
			cmd.set_storage_texture(0, 2, dummy_1d_array->get_view());
			groups_x = (get_output().get_view_width() + 255) / 256;
		}

		cmd.set_specialization_constant(0, pass);
		cmd.dispatch(groups_x, input.get_view_height(), 1);
		if (dst)
			storage_to_sampled_barrier(cmd, dst->get_image());
		cmd.end_region();
	};

	cmd.set_specialization_constant(2, int(options.cable));

	// Run downsampling filter to 2x BT.601 rate. It's a comfortable and convenient sampling rate to do DSP on.
	run_pass(PassDownsample, nullptr, &downsample_target->get_view());

	if (options.cable == Cable::Component)
		push.input_offset = -EncodeOffset - horizontal_shift;
	else
		push.input_offset = -EncodeOffset;

	run_pass(PassEncode, &downsample_target->get_view(),
		options.cable == Cable::Component ? nullptr : &encode_target->get_view());

	if (options.cable != Cable::Component)
	{
		if (options.cable == Cable::Composite && filter_options.line_comb)
		{
			// PAL does a bandpass while NTSC does not, so shift accordingly.
			push.input_offset = 0;
			run_pass(PassSeparation, &encode_target->get_view(), &chroma_estimate_target->get_view());

			push.input_offset = -BandpassOffset;
			run_pass(PassBandpass, &chroma_estimate_target->get_view(), &bandpass_target->get_view());

			cmd.set_texture(0, 4, bandpass_target->get_view());
		}

		// No need for further processing.
		cmd.set_storage_texture(0, 2, dummy_1d_array->get_view());

		// We have Y + modulated C. Generate output RGB.
		// Shifts the output so that we end up being centered in a standard BT.601 720 pixel container.
		push.input_offset = -DecodeOffset - horizontal_shift;
		cmd.set_specialization_constant(3, filter_options.line_comb);
		cmd.set_specialization_constant(4, filter_options.skip_notch);
		run_pass(PassDecode, &encode_target->get_view(), nullptr);
	}

	cmd.image_barrier(*decode_target, VK_IMAGE_LAYOUT_GENERAL, filter_options.dst_layout,
					  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
					  filter_options.dst_stage, filter_options.dst_access);
}

const ImageView &AnalogVideoFilter::get_output() const
{
	return decode_target->get_view();
}

bool CRTFilter::init(Vulkan::Device &device)
{
	ResourceLayout layout;
	Analog::Shaders<> shaders(device, layout, 0);

	scan = device.request_program(shaders.blit, shaders.scan);
	bloom = device.request_program(shaders.blit, shaders.bloom);
	sinc[0] = device.request_program(shaders.blit, shaders.sinc[0]);
	sinc[1] = device.request_program(shaders.blit, shaders.sinc[1]);

	return true;
}

mat3 CRTFilter::generate_primary_conversion(const Granite::Primaries &output, const Granite::Primaries &input)
{
	return inverse(compute_xyz_matrix(output)) * compute_xyz_matrix(input);
}

const Granite::Primaries &CRTFilter::get_primaries(Primaries primaries)
{
	static const Granite::Primaries bt709 = {
		{ 0.640f, 0.330f },
		{ 0.300f, 0.600f },
		{ 0.150f, 0.060f },
		{ 0.3127f, 0.3290f },
	};

	static const Granite::Primaries bt2020 = {
		{ 0.708f, 0.292f },
		{ 0.170f, 0.797f },
		{ 0.131f, 0.046f },
		{ 0.3127f, 0.3290f },
	};

	static const Granite::Primaries bt601_525 = {
		{ 0.63f, 0.34f },
		{ 0.31f, 0.595f },
		{ 0.155f, 0.07f },
		{ 0.3127f, 0.3290f },
	};

	static const Granite::Primaries bt601_625 = {
		{ 0.64f, 0.33f },
		{ 0.29f, 0.60f },
		{ 0.15f, 0.06f },
		{ 0.3127f, 0.3290f },
	};

	static const Granite::Primaries ntsc_legacy = {
		{ 0.67f, 0.33f },
		{ 0.21f, 0.71f },
		{ 0.14f, 0.08f },
		{ 0.31f, 0.316f },
	};

	switch (primaries)
	{
	default:
	case Primaries::BT709: return bt709;
	case Primaries::BT2020: return bt2020;
	case Primaries::BT601_525: return bt601_525;
	case Primaries::BT601_625: return bt601_625;
	case Primaries::NTSC_Legacy: return ntsc_legacy;
	}
}

void CRTFilter::init_buffers(Vulkan::Device &device,
                             const Vulkan::ImageView &input_view,
                             uint32_t, uint32_t output_height)
{
	VkImageFormatProperties2 props2 = { VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2 };
	bool supports_rgb9e5 = device.get_image_format_properties(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, VK_IMAGE_TYPE_2D,
	                                                          VK_IMAGE_TILING_OPTIMAL,
	                                                          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
	                                                          VK_IMAGE_USAGE_SAMPLED_BIT,
	                                                          0, nullptr, &props2);

	auto fmt = supports_rgb9e5 ? VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 : VK_FORMAT_B10G11R11_UFLOAT_PACK32;

	if (input_width != input_view.get_view_width() || input_height != input_view.get_view_height())
	{
		input_width = input_view.get_view_width();
		input_height = input_view.get_view_height();

		auto info = Vulkan::ImageCreateInfo::render_target(
			BaseOutputResolution * 3, input_view.get_view_height(), fmt);

		// Sample the scanlines at even multiple resolution to avoid pumping effects due to aliasing.
		if (input_view.get_view_height() > 350)
			info.height *= 4;
		else
			info.height *= 8;

		info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
		info.initial_layout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
		phosphor_layer_front = device.create_image(info);
		phosphor_layer_back = device.create_image(info);
		bloomed = device.create_image(info);
		device.set_name(*phosphor_layer_front, "phosphor-front");
		device.set_name(*phosphor_layer_back, "phosphor-back");
		device.set_name(*bloomed, "bloom");
	}

	if (!sinc_vert || sinc_vert->get_height() != output_height || sinc_vert->get_width() != bloomed->get_width())
	{
		auto info = Vulkan::ImageCreateInfo::render_target(bloomed->get_width(), output_height, fmt);
		info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
		info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
		sinc_vert = device.create_image(info);
		device.set_name(*sinc_vert, "sinc-vert");
	}
}

bool CRTFilter::run_filter_encode(Vulkan::CommandBuffer &cmd, const FilterOptions &filter_options)
{
	if (!sinc_vert)
		return false;

	struct
	{
		vec4 input_sizes;
		vec4 output_sizes;
		vec2 range;
		float bandwidth;
		float max_cll;
	} push = {};

	push.input_sizes = vec4(
		float(sinc_vert->get_width()),
		float(sinc_vert->get_height()),
		1.0f / float(sinc_vert->get_width()),
		1.0f / float(sinc_vert->get_height()));

	push.output_sizes = vec4(
		cmd.get_viewport().width,
		cmd.get_viewport().height,
		1.0f / cmd.get_viewport().width,
		1.0f / cmd.get_viewport().height);

	push.range = vec2(filter_options.input_rect.x, filter_options.input_rect.width);

	// If we're downsampling, make sure we get a proper low-pass.
	float effective_horiz_resolution = filter_options.input_rect.width * float(sinc_vert->get_width());
	push.bandwidth = std::min(1.0f, 0.9f * push.output_sizes.x / effective_horiz_resolution);
	push.max_cll = filter_options.hdr10_target_max_cll;

	cmd.push_constants(&push, 0, sizeof(push));

	mat3 conv = generate_primary_conversion(
		get_primaries(filter_options.hdr10 ? Primaries::BT2020 : Primaries::BT709),
		get_primaries(filter_options.phosphor_primaries));

	float sdr_scale = filter_options.hdr10 ? filter_options.hdr10_target_paper_white : 1.0f;

	auto *primary_transform = cmd.allocate_typed_constant_data<vec4>(0, 2, 3);
	primary_transform[0] = vec4(sdr_scale * conv[0], 0.0f);
	primary_transform[1] = vec4(sdr_scale * conv[1], 0.0f);
	primary_transform[2] = vec4(sdr_scale * conv[2], 0.0f);
	cmd.set_opaque_sprite_state();
	cmd.set_depth_test(false, false);
	cmd.set_specialization_constant_mask(0x1);
	cmd.set_specialization_constant(0, filter_options.hdr10);
	cmd.set_program(sinc[1]);
	cmd.set_texture(0, 0, sinc_vert->get_view());

	cmd.begin_region("sinc-output");
	cmd.draw(3);
	cmd.end_region();

	return true;
}

bool CRTFilter::run_filter_prepass(Vulkan::CommandBuffer &cmd, const Vulkan::ImageView &view,
                                   const FilterOptions &filter_options,
                                   uint32_t output_width, uint32_t output_height)
{
	init_buffers(cmd.get_device(), view, output_width, output_height);

	std::swap(phosphor_layer_front, phosphor_layer_back);
	if (front_is_valid)
		back_is_valid = true;

	RenderPassInfo rp = {};
	rp.num_color_attachments = 1;
	rp.color_attachments[0] = &phosphor_layer_front->get_view();
	rp.store_attachments = 0x1;

	cmd.begin_barrier_batch();
	cmd.image_barrier(*phosphor_layer_front, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
	                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
	                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
	cmd.image_barrier(*bloomed, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
					  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
					  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
	cmd.image_barrier(*sinc_vert, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
					  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
					  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
	cmd.end_barrier_batch();

	struct
	{
		vec4 input_sizes;
		vec4 output_sizes;
		float phase;
		float feedback;
		float input_strength;
	} push = {};

	push.input_sizes = vec4(
		float(view.get_view_width()),
		float(view.get_view_height()),
		1.0f / float(view.get_view_width()),
		1.0f / float(view.get_view_height()));

	push.output_sizes = vec4(
		float(phosphor_layer_back->get_width()),
		float(phosphor_layer_back->get_height()),
		1.0f / float(phosphor_layer_back->get_width()),
		1.0f / float(phosphor_layer_back->get_height()));

	push.phase = filter_options.progressive ? 0.0f : (0.25f - 0.5f * float(filter_options.phase));

	// Simulate some kind of phosphor persistence. Should be very subtle but might aid
	// perceptual deinterlacing effect.
	push.feedback = back_is_valid ? filter_options.feedback : 0.0f;
	push.input_strength = filter_options.input_strength;

	cmd.begin_region("crt-scan");
	{
		cmd.begin_render_pass(rp);
		cmd.set_opaque_sprite_state();
		cmd.set_program(scan);
		cmd.set_texture(0, 0, view, StockSampler::LinearClamp);
		cmd.set_texture(0, 1, phosphor_layer_back->get_view(), StockSampler::NearestClamp);

		cmd.push_constants(&push, 0, sizeof(push));

		cmd.draw(3);
		cmd.end_render_pass();
	}
	cmd.end_region();

	cmd.image_barrier(*phosphor_layer_front, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
	                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
	                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

	cmd.begin_region("crt-bloom");
	{
		rp.color_attachments[0] = &bloomed->get_view();
		cmd.begin_render_pass(rp);
		cmd.set_opaque_sprite_state();
		cmd.set_program(bloom);
		cmd.set_texture(0, 0, phosphor_layer_front->get_view());
		cmd.draw(3);
		cmd.end_render_pass();
	}
	cmd.end_region();

	cmd.image_barrier(*bloomed, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
				  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

	cmd.begin_region("crt-sinc-vert");
	{
		struct
		{
			vec4 input_sizes;
			vec4 output_sizes;
			vec2 range;
			float bandwidth;
		} sinc_push = {};

		sinc_push.input_sizes.x = float(bloomed->get_width());
		sinc_push.input_sizes.y = float(bloomed->get_height());
		sinc_push.input_sizes.z = 1.0f / sinc_push.input_sizes.x;
		sinc_push.input_sizes.w = 1.0f / sinc_push.input_sizes.y;

		sinc_push.output_sizes.x = float(sinc_vert->get_width());
		sinc_push.output_sizes.y = float(sinc_vert->get_height());
		sinc_push.output_sizes.z = 1.0f / sinc_push.output_sizes.x;
		sinc_push.output_sizes.w = 1.0f / sinc_push.output_sizes.y;

		sinc_push.range = vec2(filter_options.input_rect.y, filter_options.input_rect.height);

		// If we're downsampling, make sure we get a proper low-pass.
		float effective_vert_resolution = filter_options.input_rect.height * float(bloomed->get_height());
		sinc_push.bandwidth = std::min(1.0f, 0.9f * sinc_push.output_sizes.y / effective_vert_resolution);

		cmd.push_constants(&sinc_push, 0, sizeof(sinc_push));

		rp.color_attachments[0] = &sinc_vert->get_view();
		cmd.begin_render_pass(rp);
		cmd.set_opaque_sprite_state();
		cmd.set_program(sinc[0]);
		cmd.set_texture(0, 0, bloomed->get_view());
		cmd.draw(3);
		cmd.end_render_pass();
	}
	cmd.end_region();

	cmd.image_barrier(*sinc_vert, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
				  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

	front_is_valid = true;
	return true;
}
}
