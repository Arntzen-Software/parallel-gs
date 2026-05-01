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
#include "analog_video.hpp"
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

static constexpr float SDRScale = 600.0f;

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

	AnalogVideoFilter filter;
	CRTFilter crt_filter;

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

		AnalogVideoFilter::Options dev_opts = {};
		dev_opts.cable = AnalogVideoFilter::Cable::Composite;
		dev_opts.system = AnalogVideoFilter::System::PAL;
		if (!filter.init(e.get_device(), dev_opts))
			request_shutdown();
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

	void test_filter()
	{
		auto &wsi = get_wsi();
		auto &device = wsi.get_device();

		bool rdoc = Device::init_renderdoc_capture();
		if (rdoc)
			device.begin_renderdoc_capture();

		auto cmd = device.request_command_buffer();

		auto info = ImageCreateInfo::immutable_2d_image(1, 16, VK_FORMAT_R8G8B8A8_UNORM);
		const u8vec4 data[16] = {
#if 0
			// 75% and 100% tests as prescribed.
			u8vec4(191, 191, 191, 0),
			u8vec4(191, 191, 0, 0),
			u8vec4(0, 191, 191, 0),
			u8vec4(0, 191, 0, 0),
			u8vec4(191, 0, 191, 0),
			u8vec4(191, 0, 0, 0),
			u8vec4(0, 0, 191, 0),
			u8vec4(0, 0, 0, 0),
			u8vec4(255, 255, 255, 0),
			u8vec4(255, 255, 0, 0),
			u8vec4(0, 255, 255, 0),
			u8vec4(0, 255, 0, 0),
			u8vec4(255, 0, 255, 0),
			u8vec4(255, 0, 0, 0),
			u8vec4(0, 0, 255, 0),
			u8vec4(0, 0, 0, 0),
#endif
			u8vec4(0, 0, 191, 0),
			u8vec4(0, 0, 191, 0),
			u8vec4(0, 0, 191, 0),
			u8vec4(0, 0, 191, 0),
			u8vec4(0, 0, 191, 0),
			u8vec4(0, 0, 191, 0),
			u8vec4(0, 0, 191, 0),
			u8vec4(0, 0, 191, 0),
			u8vec4(191, 0, 0, 0),
			u8vec4(191, 0, 0, 0),
			u8vec4(191, 0, 0, 0),
			u8vec4(191, 0, 0, 0),
			u8vec4(191, 0, 0, 0),
			u8vec4(191, 0, 0, 0),
			u8vec4(191, 0, 0, 0),
			u8vec4(191, 0, 0, 0),
		};
		ImageInitialData initial = { data, 0, 0 };
		auto test_image = device.create_image(info, &initial);

		static const struct
		{
			const char *tag;
			AnalogVideoFilter::Cable cable;
			AnalogVideoFilter::System system;
			bool comb;
		} tests[] = {
			{ "PAL composite", AnalogVideoFilter::Cable::Composite, AnalogVideoFilter::System::PAL },
			{ "PAL svideo", AnalogVideoFilter::Cable::SVideo, AnalogVideoFilter::System::PAL },
			{ "NTSC composite", AnalogVideoFilter::Cable::Composite, AnalogVideoFilter::System::NTSC },
			{ "NTSC svideo", AnalogVideoFilter::Cable::SVideo, AnalogVideoFilter::System::NTSC },
			{ "NTSC composite + comb", AnalogVideoFilter::Cable::Composite, AnalogVideoFilter::System::NTSC, true },
			{ "PAL composite + comb", AnalogVideoFilter::Cable::Composite, AnalogVideoFilter::System::PAL, true },
		};

		for (auto &test : tests)
		{
			cmd->begin_region(test.tag);
			AnalogVideoFilter test_filter;
			AnalogVideoFilter::Options dev_opts = {};
			dev_opts.cable = test.cable;
			dev_opts.system = test.system;
			test_filter.init(device, dev_opts);
			AnalogVideoFilter::FilterOptions opts = {};
			opts.input_sampling_rate_mhz = 0.0f;
			opts.line_comb = test.comb;
			test_filter.run_filter(*cmd, test_image->get_view(), opts);
			cmd->end_region();
		}

		Fence fence;
		device.submit(cmd, &fence);

		if (rdoc)
		{
			device.end_renderdoc_capture();
			request_shutdown();
		}
	}

	uint32_t frame_multiplier = 1;
	uint32_t frame_multiplier_phase = 1;

	void render_frame(double, double)
	{
		if (!iface)
		{
			request_shutdown();
			return;
		}
		auto &wsi = get_wsi();
		auto &device = wsi.get_device();

		//test_filter();

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

		if (frame_multiplier_phase >= frame_multiplier && mode != IterationMode::Pause && !is_eof)
		{
			if (parser.iterate_until_vsync(false, true))
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
				frame_multiplier_phase = 0;
			}
			else
				is_eof = true;
		}

		if (mode == IterationMode::Pause)
			frame_multiplier_phase = 0;

#if 1
		wsi.set_enable_timing_feedback(true);

		RefreshRateInfo refresh_info;
		if (wsi.get_refresh_rate_info(refresh_info) && refresh_info.refresh_duration)
		{
			float fps = vsync.mode_height == 240 || vsync.mode_height == 480 ? 59.94f : 50.0f;

			// If monitor is over 100 Hz it's probably VRR.
			bool force_vrr = false;

			uint64_t target_period_ns = 1000000000ull / fps;

			if (refresh_info.mode != RefreshMode::VRR)
			{
				uint64_t interval = refresh_info.refresh_duration;

				// Try to align with monitor refresh rate if we're close enough.
				// If we cannot snap to a specific cycle we have to assume free-flowing relative timing.
				uint64_t alignment = target_period_ns % interval;
				if (alignment + interval / 256 >= interval || alignment <= interval / 256)
				{
					frame_multiplier = (target_period_ns + interval / 2) / interval;
					target_period_ns = interval;
				}
				else
				{
					// If we know we have VRR we can go to town with high refresh rate framegen.
					frame_multiplier = target_period_ns / refresh_info.refresh_duration;
					// Every individual frame should be paced at a fraction of intended refresh.
					target_period_ns = target_period_ns / frame_multiplier;
					force_vrr = true;
				}
			}
			else
			{
				// If we know we have VRR we can go to town with high refresh rate framegen.
				frame_multiplier = target_period_ns / refresh_info.refresh_duration;

				// Every individual frame should be paced at a fraction of intended refresh.
				target_period_ns = target_period_ns / frame_multiplier;
			}

			if (frame_multiplier > 1)
			{
				wsi.set_frame_duplication_aware(true, 10);
				wsi.set_present_wait_latency(2);
			}

			wsi.set_target_presentation_time(0, target_period_ns, force_vrr);
		}
#endif

		//read_page_memory();

		auto cmd = device.request_command_buffer();
		//if (vsync.image)
		//	render_fsr(*cmd, vsync.image->get_view());

		CRTFilter::FilterOptions crt_opts = {};
		crt_opts.hdr10 = get_wsi().get_backbuffer_color_space() == VK_COLOR_SPACE_HDR10_ST2084_EXT;
		crt_opts.phase = vsync.interlace_phase;
		crt_opts.phosphor_primaries = filter.get_options().system == AnalogVideoFilter::System::PAL
										  ? CRTFilter::Primaries::BT601_625
										  : CRTFilter::Primaries::BT601_525;
		crt_opts.progressive = !vsync.interlaced;
		crt_opts.feedback = 0.5f;
		crt_opts.input_strength = frame_multiplier_phase == 0 ? 1.0f : 0.0f;
		crt_opts.hdr10_target_max_cll = SDRScale;
		crt_opts.hdr10_target_paper_white = 0.75f * SDRScale;

		crt_opts.input_rect = { 0.1f, 0.05f, 0.8f, 0.9f };

		if (frame_multiplier_phase)
			wsi.set_next_present_is_duplicated();

		if (vsync.image)
		{
			AnalogVideoFilter::FilterOptions opts = {};
			opts.phase = vsync.interlace_phase;
			opts.input_sampling_rate_mhz = 13.5f * float(vsync.image->get_width()) / 640.0f;
			opts.line_comb = true;
			opts.skip_notch = false;
			filter.run_filter(*cmd, vsync.image->get_view(), opts);

			crt_filter.run_filter_prepass(*cmd, filter.get_output(), crt_opts,
			                              device.get_swapchain_view().get_view_width(),
			                              device.get_swapchain_view().get_view_height());
		}

		cmd->begin_render_pass(device.get_swapchain_render_pass(SwapchainRenderPass::Depth));

		if (vsync.image)
		{
			//render_rcas(*cmd, fsr_render_target->get_view());
			crt_filter.run_filter_encode(*cmd, crt_opts);
		}

		frame_multiplier_phase++;

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
