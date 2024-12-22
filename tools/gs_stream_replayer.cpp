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
		get_wsi().set_backbuffer_format(BackbufferFormat::sRGB);
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
		                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
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
		                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
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
		if (vsync.image)
			render_fsr(*cmd, vsync.image->get_view());

		cmd->begin_render_pass(device.get_swapchain_render_pass(SwapchainRenderPass::Depth));

		if (vsync.image)
			render_rcas(*cmd, fsr_render_target->get_view());

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

		const vec2 SIZE = vec2(100.0f, 30.0f);
		const vec2 LARGE_SIZE = vec2(150.0f, 30.0f);

		ui_offset = vec2(cmd->get_viewport().width - 105.0f, 100.0f);
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
