// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#include "application.hpp"
#include "command_buffer.hpp"
#include "device.hpp"
#include "os_filesystem.hpp"
#include "gs_interface.hpp"
#include "gs_dump_parser.hpp"
#include "flat_renderer.hpp"
#include "ui_manager.hpp"
#include "shaders/slangmosh.hpp"
#include "hash.hpp"
#include <string.h>

#include "muglm/matrix_helper.hpp"
#include "transforms.hpp"
#include "math.hpp"

using namespace Granite;
using namespace Vulkan;

static void FsrEasuCon(
		float *con0,
		float *con1,
		float *con2,
		float *con3,
		float inputViewportInPixelsX,
		float inputViewportInPixelsY,
		float inputSizeInPixelsX,
		float inputSizeInPixelsY,
		float outputSizeInPixelsX,
		float outputSizeInPixelsY)
{
	// Output integer position to a pixel position in viewport.
	con0[0] = inputViewportInPixelsX / outputSizeInPixelsX;
	con0[1] = inputViewportInPixelsY / outputSizeInPixelsY;
	con0[2] = 0.5f * inputViewportInPixelsX / outputSizeInPixelsX - 0.5f;
	con0[3] = 0.5f * inputViewportInPixelsY / outputSizeInPixelsY - 0.5f;
	con1[0] = 1.0f / inputSizeInPixelsX;
	con1[1] = 1.0f / inputSizeInPixelsY;
	con1[2] = 1.0f / inputSizeInPixelsX;
	con1[3] = -1.0f / inputSizeInPixelsY;
	con2[0] = -1.0f / inputSizeInPixelsX;
	con2[1] = 2.0f / inputSizeInPixelsY;
	con2[2] = 1.0f / inputSizeInPixelsX;
	con2[3] = 2.0f / inputSizeInPixelsY;
	con3[0] = 0.0f / inputSizeInPixelsX;
	con3[1] = 4.0f / inputSizeInPixelsY;
	con3[2] = con3[3] = 0.0f;
}

static void FsrRcasCon(float *con, float sharpness)
{
	sharpness = muglm::exp2(-sharpness);
	uint32_t half = floatToHalf(sharpness);
	con[0] = sharpness;
	uint32_t halves = half | (half << 16);
	memcpy(&con[1], &halves, sizeof(halves));
	con[2] = 0.0f;
	con[3] = 0.0f;
}

static constexpr float SDRScale = 800.0f;

class AnalogVideoFilter
{
public:
	// Changes carrier frequency and modulation strategy.
	enum class System { NTSC, PAL };
	enum class Cable { Composite, SVideo };

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
		// Normal 640/512 width output from parallel-gs is assumed to be standard BT.601 13.5 MHz.
		// For best results, should be 13.5 MHz times some power of two.
		float input_sampling_rate_mhz = 13.5f;
		float subcarrier_phase_offset = 0.0f;
		uint32_t phase = 0;
	};

	void run_filter(Vulkan::CommandBuffer &cmd, const Vulkan::ImageView &input, const FilterOptions &options);

	// Y resolution is the same as input resolution.
	// X resolution is standard 720 horizontal active pixels, but doubled to bandlimited 1440 pixels.
	// Output is in linear light with the appropriate primaries for the system in question.
	const Vulkan::ImageView &get_output() const;

	enum class Primaries
	{
		NTSC_Legacy, // Legacy saturated primaries from 1953
		BT601_525, // SMPTE C, "modern" NTSC as defined by SMPTE
		BT601_625, // PAL
		BT709, // sRGB displays
		BT2020, // HDR10
	};
	static const Granite::Primaries &get_primaries(Primaries primaries);
	static mat3 generate_primary_conversion(const Granite::Primaries &output, const Granite::Primaries &input);

private:
	Vulkan::Device *device = nullptr;
	Options options = {};
	Vulkan::ImageHandle downsample_target;
	Vulkan::ImageHandle encode_target;
	Vulkan::ImageHandle dummy_1d_array;
	Vulkan::ImageHandle decode_target;
	uint64_t total_line_counter = 0;
};

static constexpr uint32_t BaseOutputResolution = 720;

bool AnalogVideoFilter::init(Vulkan::Device &device_, const Options &options_)
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

	image_info.format = options.cable == Cable::Composite ? VK_FORMAT_R16_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
	encode_target = device->create_image(image_info);
	device->set_name(*encode_target, "encode-target");

	image_info.width = 1;
	image_info.layers = 1;
	image_info.misc = Vulkan::IMAGE_MISC_FORCE_ARRAY_BIT;
	image_info.initial_layout = VK_IMAGE_LAYOUT_GENERAL;
	image_info.layout = ImageLayout::General;
	image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	dummy_1d_array = device->create_image(image_info);
	device->set_name(*dummy_1d_array, "dummy-1d-array");

	// Encode output on-demand.
	return true;
}

void AnalogVideoFilter::run_filter(Vulkan::CommandBuffer &cmd, const Vulkan::ImageView &input,
                                   const FilterOptions &filter_options)
{
	// Out of spec.
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

	struct
	{
		int32_t input_offset;
		int32_t output_offset;
		float subcarrier_phases_per_pixel; // For ENCODE pass.
		float subcarrier_phases_per_line; // For ENCODE pass.
		float subcarrier_phase_offset;
		float input_horiz_scale; // For DOWNSAMPLE pass.
	} push = {};

	cmd.set_program("assets://composite.comp");
	cmd.set_specialization_constant_mask(0x3);
	// Controls the filters. PAL has a bit more bandwidth.
	cmd.set_specialization_constant(1, options.system == System::PAL);

	// PS2 CRTC subpixel clock is 4x BT.601 it seems.
	push.input_offset = -16;
	push.input_horiz_scale = filter_options.input_sampling_rate_mhz / (13.5f * 4.0f);
	push.subcarrier_phases_per_pixel =
			(options.system == System::NTSC ? (315.0f / 88.0f) : 4.43361875f) /
			(13.5f * 2.0f);

	// Flip carrier phase every frame. Unsure if this is how it is supposed to work.
	// NTSC scanlines flip phase every scanline.
	if (options.system == System::NTSC)
	{
		push.subcarrier_phase_offset = (total_line_counter & 1) ? 0.5f : 0.0f;
		push.subcarrier_phases_per_line = 227.5f;
	}
	else
	{
		// TODO: PAL. It's more complicated.
	}

	push.subcarrier_phase_offset += filter_options.subcarrier_phase_offset;

	// The exact specifics shouldn't matter too much, just need some way to nudge the carrier phase.
	auto odd_lines = options.system == System::NTSC ? 263 : 262;
	auto even_lines = options.system == System::NTSC ? 313 : 312;
	total_line_counter += (filter_options.phase & 1) ? odd_lines : even_lines;

	cmd.set_unorm_texture(0, 0, input);
	cmd.set_texture(0, 1, dummy_1d_array->get_view());
	cmd.set_storage_texture(0, 2, downsample_target->get_view());
	cmd.set_storage_texture(0, 3, get_output());

	cmd.begin_barrier_batch();
	cmd.image_barrier(*downsample_target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
	                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0,
	                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
	cmd.image_barrier(*encode_target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
	                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0,
	                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
	cmd.image_barrier(*decode_target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
	                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0,
	                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
	cmd.end_barrier_batch();

	enum
	{
		PassDownsample = 0,
		PassEncode,
		PassDecode
	};

	cmd.push_constants(&push, 0, sizeof(push));

	// Run downsampling filter to 2x BT.601 rate. It's a comfortable and convenient sampling rate to do DSP on.
	uint32_t groups_x = (downsample_target->get_width() + 255) / 256;
	cmd.set_specialization_constant(0, PassDownsample);
	cmd.dispatch(groups_x, input.get_view_height(), 1);
	cmd.image_barrier(*downsample_target, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
	                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

	cmd.set_storage_texture(0, 2, encode_target->get_view());
	cmd.set_specialization_constant(0, PassEncode);
	groups_x = (encode_target->get_width() + 255) / 256;
	cmd.set_texture(0, 1, downsample_target->get_view());
	cmd.dispatch(groups_x, input.get_view_height(), 1);

	cmd.image_barrier(*encode_target, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
				  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
				  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

	cmd.set_texture(0, 1, encode_target->get_view());
	cmd.set_specialization_constant(0, PassDecode);
	cmd.set_storage_texture(0, 2, dummy_1d_array->get_view());
	groups_x = (get_output().get_view_width() + 255) / 256;
	push.input_offset = -80; // Shifts the output so that we end up being centered in a standard BT.601 720 pixel container.
	cmd.push_constants(&push, 0, sizeof(push));
	cmd.dispatch(groups_x, input.get_view_height(), 1);

	cmd.image_barrier(*decode_target, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
	                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
	                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

const Vulkan::ImageView &AnalogVideoFilter::get_output() const
{
	return decode_target->get_view();
}

mat3 AnalogVideoFilter::generate_primary_conversion(const Granite::Primaries &output, const Granite::Primaries &input)
{
	return inverse(compute_xyz_matrix(output)) * compute_xyz_matrix(input);
}

const Granite::Primaries &AnalogVideoFilter::get_primaries(Primaries primaries)
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

namespace ParallelGS
{
struct StreamApplication : Granite::Application, Granite::EventHandler
{
	explicit StreamApplication(const char *path_)
		: path(path_)
	{
		EVENT_MANAGER_REGISTER_LATCH(StreamApplication, on_device_created, on_device_destroyed, DeviceCreatedEvent);
		EVENT_MANAGER_REGISTER_LATCH(StreamApplication, on_swapchain_created, on_swapchain_destroyed, SwapchainParameterEvent);
		EVENT_MANAGER_REGISTER(StreamApplication, on_key_pressed, KeyboardEvent);
		get_wsi().set_backbuffer_format(BackbufferFormat::UNORM);

		auto meta = get_wsi().get_hdr_metadata();
		meta.maxContentLightLevel = SDRScale;
		// The rest can be left undefined.
		get_wsi().set_hdr_metadata(meta);
	}

	std::unique_ptr<GSInterface> iface;
	GSDumpParser parser;
	std::string path;

	enum class IterationMode
	{
		Full,
		Step,
		Pause
	};
	IterationMode mode = IterationMode::Full;
	ScanoutResult vsync = {};
	unsigned vsync_index = 0;
	bool is_eof = false;
	FlatRenderer flat_renderer;
	bool has_renderdoc_capture = false;
	uint32_t capture_count = 0;
	DebugMode::DrawDebugMode draw_mode = DebugMode::DrawDebugMode::None;
	bool hdr = false;

	enum { NumCaptureFrames = 4 };

	bool on_key_pressed(const KeyboardEvent &e)
	{
		if (e.get_key_state() == KeyState::Released)
			return true;

		if (e.get_key() == Key::F)
		{
			mode = IterationMode::Step;
		}
		else if (e.get_key() == Key::R)
		{
			vsync_index = 0;
			is_eof = !parser.restart();
		}
		else if (e.get_key() == Key::C)
		{
			if (has_renderdoc_capture)
			{
				capture_count = NumCaptureFrames;
				mode = IterationMode::Step;
			}
		}
		else if (e.get_key() == Key::Space)
		{
			if (mode != IterationMode::Full)
				mode = IterationMode::Full;
			else
				mode = IterationMode::Pause;
		}
		else if (e.get_key() == Key::D)
		{
			draw_mode = DebugMode::DrawDebugMode((uint32_t(draw_mode) + 1) % uint32_t(DebugMode::DrawDebugMode::Count));
		}
		else if (e.get_key() == Key::_1 && iface)
			iface->set_super_sampling_rate(SuperSampling::X1, true, false);
		else if (e.get_key() == Key::_2 && iface)
			iface->set_super_sampling_rate(SuperSampling::X2, true, false);
		else if (e.get_key() == Key::_3 && iface)
			iface->set_super_sampling_rate(SuperSampling::X4, true, false);
		else if (e.get_key() == Key::_4 && iface)
			iface->set_super_sampling_rate(SuperSampling::X8, true, false);
		else if (e.get_key() == Key::_5 && iface)
			iface->set_super_sampling_rate(SuperSampling::X16, true, false);
		else if (e.get_key() == Key::M)
			get_wsi().set_present_mode(get_wsi().get_present_mode() == PresentMode::SyncToVBlank ? PresentMode::UnlockedMaybeTear : PresentMode::SyncToVBlank);
		else if (e.get_key() == Key::H)
		{
			hdr = !hdr;
			get_wsi().set_backbuffer_format(hdr ? BackbufferFormat::HDR10 : BackbufferFormat::UNORM);
		}

		return true;
	}

	ImageHandle fsr_render_target;
	Program *upscale_program = nullptr;
	Program *sharpen_program = nullptr;

	void on_swapchain_created(const SwapchainParameterEvent &e)
	{
		auto info = ImageCreateInfo::render_target(e.get_width(), e.get_height(), VK_FORMAT_R8G8B8A8_UNORM);
		info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
		info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		info.misc = IMAGE_MISC_MUTABLE_SRGB_BIT;
		fsr_render_target = e.get_device().create_image(info);
	}

	void on_swapchain_destroyed(const SwapchainParameterEvent &)
	{
		fsr_render_target.reset();
	}

	void on_device_created(const DeviceCreatedEvent &e)
	{
		// We will cycle through many memory contexts per frame most likely.
		e.get_device().init_frame_contexts(12);

		iface = std::make_unique<GSInterface>();

		GSOptions opts = {};
		opts.dynamic_super_sampling = true;

		if (!iface->init(&e.get_device(), opts))
		{
			request_shutdown();
			LOGE("Failed to init GSInterface.\n");
			return;
		}

		if (!parser.open_raw(path.c_str(), 4 * 1024 * 1024, iface.get()))
		{
			request_shutdown();
			iface.reset();
			LOGE("Failed to open stream.\n");
			return;
		}

		DebugMode debug_mode = {};
		debug_mode.timestamps = true;
		iface->set_debug_mode(debug_mode);

		has_renderdoc_capture = Device::init_renderdoc_capture();

		ResourceLayout layout;
		Shaders<> suite(e.get_device(), layout, 0);
		upscale_program = e.get_device().request_program(suite.upscale_vert, suite.upscale_frag);
		sharpen_program = e.get_device().request_program(suite.sharpen_vert, suite.sharpen_frag);
	}

	void on_device_destroyed(const DeviceCreatedEvent &)
	{
		iface.reset();
		vsync = {};
		upscale_program = nullptr;
		sharpen_program = nullptr;
	}

	FlushStats stats = {};
	double timestamp_stats[int(TimestampType::Count)] = {};
	double last_timestamp_stats[int(TimestampType::Count)] = {};

	void render_fsr(CommandBuffer &cmd, const ImageView &view)
	{
		cmd.image_barrier(*fsr_render_target,
		                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
		                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
		                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

		RenderPassInfo rp = {};
		rp.num_color_attachments = 1;
		rp.color_attachments[0] = &fsr_render_target->get_view();
		rp.store_attachments = 1u << 0;

		cmd.begin_render_pass(rp);
		{
			struct Constants
			{
				float params[4][4];
			} constants;

			struct Push
			{
				float width, height;
			} push;

			auto width = float(view.get_image().get_width());
			auto height = float(view.get_image().get_height());
			auto *params = cmd.allocate_typed_constant_data<Constants>(1, 0, 1);
			FsrEasuCon(constants.params[0], constants.params[1], constants.params[2], constants.params[3],
			           width, height, width, height, cmd.get_viewport().width, cmd.get_viewport().height);
			*params = constants;

			push.width = cmd.get_viewport().width;
			push.height = cmd.get_viewport().height;
			cmd.push_constants(&push, 0, sizeof(push));

			const vec2 vertex_data[] = { vec2(-1.0f, -1.0f), vec2(-1.0f, 3.0f), vec2(3.0f, -1.0f) };
			memcpy(cmd.allocate_vertex_data(0, sizeof(vertex_data), sizeof(vec2)), vertex_data, sizeof(vertex_data));
			cmd.set_vertex_attrib(0, 0, VK_FORMAT_R32G32_SFLOAT, 0);

			cmd.set_texture(0, 0, view, StockSampler::NearestClamp);

			cmd.set_program(upscale_program);
			cmd.set_opaque_state();
			cmd.set_depth_test(false, false);
			cmd.draw(3);
		}
		cmd.end_render_pass();

		cmd.image_barrier(*fsr_render_target,
		                  VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
		                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
	}

	void render_dummy(CommandBuffer &cmd, const ImageView &view)
	{
		cmd.set_opaque_sprite_state();
		cmd.set_depth_test(false, false);
		cmd.set_program("assets://blit.vert", "assets://blit.frag");
		cmd.set_specialization_constant_mask(1);
		cmd.set_specialization_constant(0, get_wsi().get_backbuffer_color_space() == VK_COLOR_SPACE_HDR10_ST2084_EXT);
		cmd.set_texture(0, 0, view, StockSampler::LinearClamp);

		mat3 conv = AnalogVideoFilter::generate_primary_conversion(
			AnalogVideoFilter::get_primaries(AnalogVideoFilter::Primaries::BT2020),
			AnalogVideoFilter::get_primaries(AnalogVideoFilter::Primaries::BT601_525));
		float sdr_scale = SDRScale;

		auto *primary_transform = cmd.allocate_typed_constant_data<vec4>(0, 1, 3);
		primary_transform[0] = vec4(sdr_scale * conv[0], 0.0f);
		primary_transform[1] = vec4(sdr_scale * conv[1], 0.0f);
		primary_transform[2] = vec4(sdr_scale * conv[2], 0.0f);

		struct
		{
			vec4 input_sizes;
			vec4 output_sizes;
		} push = {};

		push.input_sizes = vec4(
			float(view.get_view_width()),
			float(view.get_view_height()),
			1.0f / float(view.get_view_width()),
			1.0f / float(view.get_view_height()));

		push.output_sizes = vec4(
			cmd.get_viewport().width,
			cmd.get_viewport().height,
			1.0f / cmd.get_viewport().width,
			1.0f / cmd.get_viewport().height);

		cmd.push_constants(&push, 0, sizeof(push));

		cmd.draw(3);
	}

	void render_rcas(CommandBuffer &cmd, const ImageView &view)
	{
		struct Constants
		{
			float params[4];
			int32_t range[4];
		} constants;

		FsrRcasCon(constants.params, 0.5f);
		constants.range[0] = 0;
		constants.range[1] = 0;
		constants.range[2] = int(view.get_view_width()) - 1;
		constants.range[3] = int(view.get_view_height()) - 1;
		auto *params = cmd.allocate_typed_constant_data<Constants>(1, 0, 1);
		*params = constants;

		const vec2 vertex_data[] = { vec2(-1.0f, -1.0f), vec2(-1.0f, 3.0f), vec2(3.0f, -1.0f) };
		memcpy(cmd.allocate_vertex_data(0, sizeof(vertex_data), sizeof(vec2)), vertex_data, sizeof(vertex_data));
		cmd.set_vertex_attrib(0, 0, VK_FORMAT_R32G32_SFLOAT, 0);
		cmd.set_srgb_texture(0, 0, fsr_render_target->get_view());
		cmd.set_sampler(0, 0, Vulkan::StockSampler::NearestClamp);
		cmd.set_opaque_state();
		cmd.set_depth_test(false, false);
		cmd.set_program(sharpen_program);
		cmd.draw(3);
	}

#if 0
	Util::Hash last_hash = 0;

	void read_page_memory()
	{
		auto *mapped = static_cast<const uint32_t *>(iface->map_vram_read(0x300000, PGS_PAGE_ALIGNMENT_BYTES));
		Util::Hasher h;
		for (uint32_t i = 0; i < PGS_PAGE_ALIGNMENT_BYTES / sizeof(uint32_t); i++)
			h.u32(mapped[i] & 0x0f000000);

		if (h.get() != last_hash)
		{
			LOGI("Hash changed at V #%u: %016llx\n", vsync_index, static_cast<unsigned long long>(h.get()));
			last_hash = h.get();
		}
	}
#endif

	void test_filter()
	{
		auto &wsi = get_wsi();
		auto &device = wsi.get_device();

		bool rdoc = Device::init_renderdoc_capture();
		if (rdoc)
			device.begin_renderdoc_capture();

		auto cmd = device.request_command_buffer();

		const float test_input[] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		BufferCreateInfo bufinfo = {};
		bufinfo.usage = VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
		bufinfo.domain = BufferDomain::Device;
		bufinfo.size = sizeof(test_input);
		auto inputs = device.create_buffer(bufinfo, test_input);
		bufinfo.usage = VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
		bufinfo.domain = BufferDomain::CachedHost;
		bufinfo.size = 64 * sizeof(float);
		bufinfo.misc = BUFFER_MISC_ZERO_INITIALIZE_BIT;
		auto outputs = device.create_buffer(bufinfo);

		BufferViewCreateInfo view_info = {};
		view_info.buffer = inputs.get();
		view_info.format = VK_FORMAT_R32_SFLOAT;
		view_info.offset = 0;
		view_info.range = VK_WHOLE_SIZE;
		auto input_view = device.create_buffer_view(view_info);
		view_info.buffer = outputs.get();
		auto output_view = device.create_buffer_view(view_info);

		struct
		{
			int32_t input_offset;
			int32_t output_offset;
			float subcarrier_phases_per_pixel; // For ENCODE pass.
			float eotf_gamma; // For DECODE pass.
			float input_horiz_scale; // For DOWNSAMPLE pass.
			float carrier_phase_offset;
		} push = {};
		push.input_offset = -14;
		cmd->push_constants(&push, 0, sizeof(push));

		cmd->set_program("assets://composite.comp");
		cmd->set_specialization_constant_mask(0x1);
		cmd->set_specialization_constant(0, 0);
		cmd->set_buffer_view(0, 0, *input_view);
		cmd->set_storage_buffer_view(0, 1, *output_view);
		cmd->dispatch(1, 1, 1);
		cmd->barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		             VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);

		Fence fence;
		device.submit(cmd, &fence);
		fence->wait();

		auto *ptr = static_cast<const float *>(device.map_host_buffer(*outputs, MEMORY_ACCESS_READ_BIT));
		for (int i = 0; i < 64; i++)
			LOGI("Value %u = %f\n", i, ptr[i]);

		if (rdoc)
			device.end_renderdoc_capture();
		exit(0);
	}

	void render_frame(double, double)
	{
		if (!iface)
		{
			request_shutdown();
			return;
		}
		auto &wsi = get_wsi();
		auto &device = wsi.get_device();

		if (capture_count == NumCaptureFrames && has_renderdoc_capture)
		{
			device.begin_renderdoc_capture();
			DebugMode debug_mode = {};
			debug_mode.draw_mode = draw_mode;
			debug_mode.feedback_render_target = true;
			debug_mode.timestamps = true;
			iface->set_debug_mode(debug_mode);
		}

		bool empty_vsync = false;

		if (mode != IterationMode::Pause && !is_eof)
		{
			if (parser.iterate_until_vsync())
			{
				vsync = parser.consume_vsync_result();
				auto flush_stats = iface->consume_flush_stats();
				if (flush_stats.num_render_passes || flush_stats.num_copies)
				{
					stats = flush_stats;
					for (int i = 0; i < int(TimestampType::Count); i++)
					{
						double new_accum = iface->get_accumulated_timestamps(TimestampType(i));
						timestamp_stats[i] = 1e3 * (new_accum - last_timestamp_stats[i]);
						last_timestamp_stats[i] = new_accum;
					}
				}
				else
					empty_vsync = true;

				vsync_index++;
			}
			else
				is_eof = true;
		}

		//read_page_memory();

		auto cmd = device.request_command_buffer();
		//if (vsync.image)
		//	render_fsr(*cmd, vsync.image->get_view());

		AnalogVideoFilter filter;
		if (vsync.image)
		{
			if (!filter.init(cmd->get_device(), {}))
				return;
			AnalogVideoFilter::FilterOptions opts = {};
			static uint32_t phase;
			opts.phase = phase++;
			opts.subcarrier_phase_offset = (opts.phase & 1) ? 0.5f : 0.0f;
			filter.run_filter(*cmd, vsync.image->get_view(), opts);
		}

		cmd->begin_render_pass(device.get_swapchain_render_pass(SwapchainRenderPass::Depth));

		if (vsync.image)
		{
			//render_rcas(*cmd, fsr_render_target->get_view());
			render_dummy(*cmd, filter.get_output());
		}

#if 1
		flat_renderer.begin();

		vec2 ui_offset = vec2(cmd->get_viewport().width - 105.0f, 5.0f);

		const auto draw_mode_str = [](DebugMode::DrawDebugMode draw) {
			switch (draw)
			{
			case DebugMode::DrawDebugMode::None: return "Simple";
			case DebugMode::DrawDebugMode::Strided: return "Strided";
			case DebugMode::DrawDebugMode::Full: return "Full";
			default: return "?";
			}
		};

		render_text("V #%05u", ui_offset, vec2(100.0f, 30.0f), vsync_index);
		render_text("%s", ui_offset + vec2(0.0f, 30.0f), vec2(100.0f, 30.0f), draw_mode_str(draw_mode));
		if (is_eof)
			render_text("EOF", ui_offset + vec2(0.0f, 60.0f), vec2(100.0f, 30.0f));
		else if (mode == IterationMode::Pause)
			render_text("Paused", ui_offset + vec2(0.0f, 60.0f), vec2(100.0f, 30.0f));

		auto color_space = get_wsi().get_backbuffer_color_space();
		if (color_space == VK_COLOR_SPACE_HDR10_ST2084_EXT)
			render_text("HDR", ui_offset + vec2(0.0f, 90.0f), vec2(100.0f, 30.0f));
		else
			render_text("SDR", ui_offset + vec2(0.0f, 90.0f), vec2(100.0f, 30.0f));

		const vec2 SIZE = vec2(100.0f, 30.0f);
		const vec2 LARGE_SIZE = vec2(150.0f, 30.0f);

		ui_offset = vec2(cmd->get_viewport().width - 105.0f, 130.0f);
		render_text("RP %u", ui_offset, SIZE, stats.num_render_passes);
		ui_offset.y += 30.0f;
		render_text("CLUT %u", ui_offset, SIZE, stats.num_palette_updates);
		ui_offset.y += 30.0f;
		render_text("COPY %u", ui_offset, SIZE, stats.num_copies);
		ui_offset.y += 30.0f;
		render_text("PRIM %u", ui_offset, SIZE, stats.num_primitives);
		ui_offset.y += 30.0f;
		render_text("IMG %u MiB", ui_offset, SIZE, unsigned(stats.allocated_image_memory / (1024 * 1024)));
		ui_offset.y += 30.0f;
		render_text("BAR %u MiB", ui_offset, SIZE, unsigned(stats.allocated_scratch_memory / (1024 * 1024)));
		ui_offset.y += 30.0f;
		if (empty_vsync)
			render_text("EMPTY", ui_offset, SIZE);

		ui_offset = vec2(cmd->get_viewport().width - 255.0f, 5.0f);
		render_text("COPY %.3f ms", ui_offset + vec2(0.0f, 0.0f), LARGE_SIZE,
		            timestamp_stats[int(TimestampType::CopyVRAM)]);
		render_text("CACHE %.3f ms", ui_offset + vec2(0.0f, 30.0f), LARGE_SIZE,
		            timestamp_stats[int(TimestampType::PaletteUpdate)] +
		            timestamp_stats[int(TimestampType::TextureUpload)]);
		render_text("SETUP %.3f ms", ui_offset + vec2(0.0f, 60.0f), LARGE_SIZE,
		            timestamp_stats[int(TimestampType::TriangleSetup)]);
		render_text("BIN %.3f ms", ui_offset + vec2(0.0f, 90.0f), LARGE_SIZE,
		            timestamp_stats[int(TimestampType::Binning)]);
		render_text("SHADE %.3f ms", ui_offset + vec2(0.0f, 120.0f), LARGE_SIZE,
		            timestamp_stats[int(TimestampType::Shading)]);

		flat_renderer.flush(*cmd, vec3(0.0f), vec3(cmd->get_viewport().width, cmd->get_viewport().height, 1.0f));
#endif

		cmd->end_render_pass();
		device.submit(cmd);

		if (capture_count)
		{
			capture_count--;
			if (!capture_count)
			{
				iface->set_debug_mode({});
				device.end_renderdoc_capture();
			}
		}

		if (mode == IterationMode::Step && !capture_count)
			mode = IterationMode::Pause;

		// Just in case.
		iface->flush();
	}

	template <typename... Ts>
	inline void render_text(const char *fmt, vec2 offset, vec2 size, Ts... ts)
	{
		char label[256];
		snprintf(label, sizeof(label), fmt, ts...);
		flat_renderer.render_quad(vec3(offset, 0.5f), size, vec4(0.0f, 0.0f, 0.0f, 0.8f));
		flat_renderer.render_text(GRANITE_UI_MANAGER()->get_font(UI::FontSize::Normal), label,
		                          vec3(offset, 0.0f), size, vec4(1.0f),
		                          Font::Alignment::Center);
	}

	unsigned get_default_width() override
	{
		return 1280;
	}

	unsigned get_default_height() override
	{
		return 224 * 4;
	}

	ImageHandle render_target;
};
}

namespace Granite
{
Application *application_create(int argc, char **argv)
{
	if (argc != 2)
	{
		LOGE("Missing path to file.\n");
		return nullptr;
	}

	GRANITE_APPLICATION_SETUP_FILESYSTEM();

	try
	{
		auto *app = new ParallelGS::StreamApplication(argv[1]);
		return app;
	}
	catch (const std::exception &e)
	{
		LOGE("application_create() threw exception: %s\n", e.what());
		return nullptr;
	}
}
}
