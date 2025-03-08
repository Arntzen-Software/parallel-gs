// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#include "gs_renderer.hpp"
#include "device.hpp"
#include "context.hpp"
#include "gs_dump_parser.hpp"
#include "global_managers_init.hpp"
#include "filesystem.hpp"
#include "thread_group.hpp"
#include "gs_interface.hpp"
#include "gs_dump_generator.hpp"
#include "cli_parser.hpp"
#include "timer.hpp"
#include <stdlib.h>

using namespace Vulkan;
using namespace ParallelGS;
using namespace Util;

static void print_help()
{
	LOGI("Usage: parallel-gs-replayer <dump.gs> [--ssaa <rate>] [--strided] [--full] [--iterations <count>] [--high-res-scanout] [--ssaa-textures] [--disable-sampler-feedback]\n");
}

int main(int argc, char **argv)
{
	std::string dump_path;
	DebugMode debug_mode;
	debug_mode.feedback_render_target = true;
	debug_mode.draw_mode = DebugMode::DrawDebugMode::None;
	unsigned total_iterations = 1;
	bool high_res_scanout = false;
	GSOptions opts = {};

	CLICallbacks cbs;
	cbs.add("--help", [&](CLIParser &parser) { parser.end(); print_help(); });
	cbs.add("--ssaa", [&](CLIParser &parser) { opts.super_sampling = SuperSampling(parser.next_uint()); });
	cbs.add("--strided", [&](CLIParser &) { debug_mode.draw_mode = DebugMode::DrawDebugMode::Strided; });
	cbs.add("--full", [&](CLIParser &) { debug_mode.draw_mode = DebugMode::DrawDebugMode::Full; });
	cbs.add("--iterations", [&](CLIParser &parser) { total_iterations = parser.next_uint(); });
	cbs.add("--high-res-scanout", [&](CLIParser &) { high_res_scanout = true; });
	cbs.add("--ssaa-textures", [&](CLIParser &) { opts.super_sampled_textures = true; });
	cbs.add("--disable-sampler-feedback", [&](CLIParser &) { debug_mode.disable_sampler_feedback = true; });
	cbs.default_handler = [&](const char *arg) { dump_path = arg; };

	CLIParser cli_parser(std::move(cbs), argc - 1, argv + 1);
	if (!cli_parser.parse())
	{
		print_help();
		return EXIT_FAILURE;
	}

	if (cli_parser.is_ended_state())
		return EXIT_SUCCESS;

	if (dump_path.empty())
	{
		LOGE("Must provide dump.\n");
		print_help();
		return EXIT_FAILURE;
	}

	if (!Context::init_loader(nullptr))
		return EXIT_FAILURE;

	Context ctx;
	ctx.set_num_thread_indices(1);
	if (!ctx.init_instance_and_device(nullptr, 0, nullptr, 0,
	                                  CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT))
		return EXIT_FAILURE;

	Device device;
	device.set_context(ctx);
	device.init_frame_contexts(4);

	GSInterface iface;
	if (!iface.init(&device, opts))
		return EXIT_FAILURE;

	bool use_rdoc = Device::init_renderdoc_capture();
	if (use_rdoc)
	{
		iface.set_debug_mode(debug_mode);
		device.begin_renderdoc_capture();
	}

	GSDumpParser parser;
	if (!parser.open(argv[1], 4 * 1024 * 1024, &iface))
		return EXIT_FAILURE;

	unsigned iterations = 0;
	uint64_t start_ns = 0;
	unsigned vsyncs = 0;

	do
	{
		do
		{
			LOGI("Running frame ...\n");
			if (iterations > 0)
				vsyncs++;
		} while (parser.iterate_until_vsync(high_res_scanout));

		if (!parser.restart())
		{
			LOGE("Failed to rewind capture.\n");
			break;
		}

		HeapBudget budget[VK_MAX_MEMORY_HEAPS] = {};
		device.get_memory_budget(budget);

		for (uint32_t i = 0; i < device.get_memory_properties().memoryHeapCount; i++)
		{
			LOGI("Memory usage - Heap %u - %s - %llu MiB / %llu MiB\n", i,
			     (device.get_memory_properties().memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 ? "DEVICE" : "HOST",
			     static_cast<unsigned long long>(budget[i].tracked_usage / (1024 * 1024)),
			     static_cast<unsigned long long>(budget[i].device_usage / (1024 * 1024)));
		}

		if (iterations == 0)
			start_ns = Util::get_current_time_nsecs();
	} while (++iterations < total_iterations);
	uint64_t end_ns = Util::get_current_time_nsecs();

	double total_time = double(end_ns - start_ns) * 1e-9;
	LOGI("Total time per VBlank: %.3f ms\n", 1e3 * total_time / double(vsyncs));

	LOGI("Done!\n");

	if (use_rdoc)
		device.end_renderdoc_capture();
}