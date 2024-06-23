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
#include "timer.hpp"
#include <stdlib.h>

using namespace Vulkan;
using namespace ParallelGS;

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		LOGE("Usage: parallel-gs-replayer <dump.gs>\n");
		return EXIT_FAILURE;
	}

	if (!Context::init_loader(nullptr))
		return EXIT_FAILURE;

	Context ctx;
	ctx.set_num_thread_indices(1);
	if (!ctx.init_instance_and_device(nullptr, 0, nullptr, 0))
		return EXIT_FAILURE;

	Device device;
	device.set_context(ctx);
	device.init_frame_contexts(4);

	GSOptions opts = {};

	GSInterface iface;
	if (!iface.init(&device, opts))
		return EXIT_FAILURE;

	bool use_rdoc = Device::init_renderdoc_capture();
	if (use_rdoc)
	{
		DebugMode debug_mode;
		debug_mode.feedback_render_target = true;
		debug_mode.draw_mode = DebugMode::DrawDebugMode::Strided;
		iface.set_debug_mode(debug_mode);
		device.begin_renderdoc_capture();
	}

	GSDumpParser parser;
	if (!parser.open(argv[1], 4 * 1024 * 1024, &iface))
		return EXIT_FAILURE;

	unsigned iterations = 0;
	uint64_t start_ns = 0, end_ns = 0;
	unsigned vsyncs = 0;

	do
	{
		do
		{
			LOGI("Running frame ...\n");
			if (iterations > 0)
				vsyncs++;
		} while (parser.iterate_until_vsync());

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
	} while (++iterations < 1);
	end_ns = Util::get_current_time_nsecs();

	double total_time = double(end_ns - start_ns) * 1e-9;
	LOGI("Total time per VBlank: %.3f ms\n", 1e3 * total_time / double(vsyncs));

	LOGI("Done!\n");

	if (use_rdoc)
		device.end_renderdoc_capture();
}