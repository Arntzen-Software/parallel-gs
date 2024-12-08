// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-License-Identifier: LGPL-3.0+

#include "cli_parser.hpp"
#include "gs_interface.hpp"
#include "gs_dump_parser.hpp"
#include "device.hpp"
#include "context.hpp"
#include "stb_image.h"
#include "stb_image_write.h"
#include "logging.hpp"
#include "rapidjson_wrapper.hpp"
#include "os_filesystem.hpp"
#include "path_utils.hpp"
#include <cmath>

using namespace Util;
using namespace Granite;
using namespace Vulkan;
using namespace rapidjson;
using namespace ParallelGS;

static void print_help()
{
	LOGI("Usage: parallel-gs-repro [--dir <folder>] [--update]\n");
}

static bool compare_reference_image(const u8vec4 *new_pixels, const unsigned char *ref_pixels,
                                    int w, int h, int c)
{
	int64_t peak_total = 255ll * 255ll * w * h;
	int64_t sqr_err_r = 0;
	int64_t sqr_err_g = 0;
	int64_t sqr_err_b = 0;

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			auto &new_pix = new_pixels[y * w + x];
			auto *ref = ref_pixels + (y * w + x) * c;
			int err_r = new_pix.x - ref[0];
			int err_g = new_pix.y - ref[1];
			int err_b = new_pix.z - ref[2];
			sqr_err_r += err_r * err_r;
			sqr_err_g += err_g * err_g;
			sqr_err_b += err_b * err_b;
		}
	}

	double psnr_r = 10.0 * std::log10(double(peak_total) / double(sqr_err_r));
	double psnr_g = 10.0 * std::log10(double(peak_total) / double(sqr_err_g));
	double psnr_b = 10.0 * std::log10(double(peak_total) / double(sqr_err_b));
	LOGI("PSNR is [%.3f, %.3f, %.3f] dB.\n", psnr_r, psnr_g, psnr_b);

	if (psnr_r < 40.0 || psnr_g < 40.0 || psnr_b < 40.0)
	{
		LOGE("PSNR is too low for reference.\n");
		return false;
	}

	return true;
}

static bool parse_flush_stats(Document &doc, FlushStats &stats)
{
	stats.num_primitives = doc["numPrimitives"].GetUint();
	stats.allocated_image_memory = doc["allocatedImageMemory"].GetUint64();
	stats.allocated_scratch_memory = doc["allocatedScratchMemory"].GetUint64();
	stats.num_copies = doc["numCopies"].GetUint();
	stats.num_copy_barriers = doc["numCopyBarriers"].GetUint();
	stats.num_palette_updates = doc["numPaletteUpdates"].GetUint();
	stats.num_render_passes = doc["numRenderPasses"].GetUint();
	return true;
}

template <typename Alloc>
static void serialize_flush_stats(Value &value, const FlushStats &stats, Alloc &allocator)
{
	value.AddMember("numPrimitives", stats.num_primitives, allocator);
	value.AddMember("allocatedImageMemory", uint64_t(stats.allocated_image_memory), allocator);
	value.AddMember("allocatedScratchMemory", uint64_t(stats.allocated_scratch_memory), allocator);
	value.AddMember("numCopies", stats.num_copies, allocator);
	value.AddMember("numCopyBarriers", stats.num_copy_barriers, allocator);
	value.AddMember("numPaletteUpdates", stats.num_palette_updates, allocator);
	value.AddMember("numRenderPasses", stats.num_render_passes, allocator);
}

static bool compare_flush_stats(const FlushStats &new_stats, const FlushStats &old_stats)
{
	bool has_mismatch = false;
	LOGI("numPrimitives: %u -> %u\n", old_stats.num_primitives, new_stats.num_primitives);
	if (new_stats.num_primitives != old_stats.num_primitives)
	{
		LOGE("numPrimitives mismatch.\n");
		has_mismatch = true;
	}

	LOGI("numCopies: %u -> %u\n", old_stats.num_copies, new_stats.num_copies);
	if (new_stats.num_copies != old_stats.num_copies)
	{
		LOGE("numCopies mismatch.\n");
		has_mismatch = true;
	}

	LOGI("numCopyBarriers: %u -> %u\n", old_stats.num_copy_barriers, new_stats.num_copy_barriers);
	if (new_stats.num_copy_barriers != old_stats.num_copy_barriers)
	{
		LOGE("numCopyBarriers mismatch.\n");
		has_mismatch = true;
	}

	LOGI("numRenderPasses: %u -> %u\n", old_stats.num_render_passes, new_stats.num_render_passes);
	if (new_stats.num_render_passes != old_stats.num_render_passes)
	{
		LOGE("numRenderPasses mismatch.\n");
		has_mismatch = true;
	}

	LOGI("numPaletteUpdates: %u -> %u\n", old_stats.num_palette_updates, new_stats.num_palette_updates);
	if (new_stats.num_palette_updates != old_stats.num_palette_updates)
	{
		LOGE("numPaletteUpdates mismatch.\n");
		has_mismatch = true;
	}

	LOGI("allocatedImageMemory: %llu -> %llu\n",
	     static_cast<unsigned long long>(old_stats.allocated_image_memory),
	     static_cast<unsigned long long>(new_stats.allocated_image_memory));
	if (new_stats.allocated_image_memory != old_stats.allocated_image_memory)
	{
		LOGE("allocatedImageMemory mismatch.\n");
		has_mismatch = true;
	}

	LOGI("allocatedScratchMemory: %llu -> %llu\n",
	     static_cast<unsigned long long>(old_stats.allocated_scratch_memory),
	     static_cast<unsigned long long>(new_stats.allocated_scratch_memory));
	if (new_stats.allocated_scratch_memory != old_stats.allocated_scratch_memory)
	{
		LOGE("allocatedScratchMemory mismatch.\n");
		has_mismatch = true;
	}

	return !has_mismatch;
}

static bool run_gs_dump(OSFilesystem &fs, Device &device, const std::string &path, bool update)
{
	LOGI("=== Testing %s ===\n", path.c_str());

	auto iface = std::make_unique<GSInterface>();
	GSOptions opts = {};
	opts.super_sampling = SuperSampling::X8;

	if (!iface->init(&device, opts))
	{
		LOGE("Failed to initialize GS replayer.\n");
		return false;
	}

	DebugMode debug_mode = {};
	debug_mode.deterministic_timeline_query = true;
	iface->set_debug_mode(debug_mode);

	GSDumpParser parser;
	if (!parser.open(path.c_str(), opts.vram_size, iface.get()))
	{
		LOGE("Failed to open dump: %s\n", path.c_str());
		return false;
	}

	while (parser.iterate_until_vsync()) {}
	auto result = parser.consume_vsync_result();

	if (!result.image)
	{
		LOGE("No image retrieved from vsync result.\n");
		return false;
	}

	BufferHandle readback;
	{
		auto cmd = device.request_command_buffer();
		cmd->image_barrier(*result.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
						   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
						   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
						   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

		BufferCreateInfo bufinfo = {};
		bufinfo.size = result.image->get_width() * result.image->get_height() * sizeof(uint32_t);
		bufinfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		bufinfo.domain = BufferDomain::CachedHost;
		readback = device.create_buffer(bufinfo);
		cmd->copy_image_to_buffer(*readback, *result.image, 0,  {},
		                          { result.image->get_width(), result.image->get_height(), 1 },
		                          0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 });
		cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
					 VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
		device.submit(cmd);
		device.wait_idle();
	}

	FlushStats flush_stats = iface->consume_flush_stats();
	auto *pixels = static_cast<u8vec4 *>(device.map_host_buffer(*readback, MEMORY_ACCESS_READ_BIT));
	for (size_t i = 0; i < readback->get_create_info().size / sizeof(u8vec4); i++)
		pixels[i].w = 0xff;

	auto ref_png_path = path + ".reference.png";
	auto ref_json_path = path + ".reference.json";
	int w, h, c;
	struct stbi_deleter { void operator()(unsigned char *ptr) { stbi_image_free(ptr); } };
	std::unique_ptr<unsigned char, stbi_deleter> ref_buffer{stbi_load(ref_png_path.c_str(), &w, &h, &c, 4)};

	if (ref_buffer)
	{
		bool mismatch_dimensions = w != int(result.image->get_width()) || h != int(result.image->get_height());
		if (mismatch_dimensions)
			LOGE("Mismatch in vsync image dimensions.\n");

		if ((mismatch_dimensions || !compare_reference_image(pixels, ref_buffer.get(), w, h, c)) && !update)
		{
			auto fault_png_path = path + ".fault.png";
			LOGI("Writing faulting image to %s.\n", fault_png_path.c_str());
			if (!stbi_write_png(fault_png_path.c_str(), w, h, 4, pixels, w * 4))
				LOGE("Failed to write faulting image to %s.\n", fault_png_path.c_str());
			return false;
		}
	}

	if (update || !ref_buffer)
	{
		w = int(result.image->get_width());
		h = int(result.image->get_height());

		if (!stbi_write_png(ref_png_path.c_str(), w, h, 4, pixels, w * 4))
		{
			LOGE("Failed to write reference image to %s.\n", ref_png_path.c_str());
			return false;
		}
	}

	Document doc;
	bool has_json = false;
	auto file_handle = fs.open(ref_json_path, FileMode::ReadOnly);

	if (file_handle)
	{
		auto map = file_handle->map();
		if (!map)
		{
			LOGE("Failed to map %s for reading.\n", ref_json_path.c_str());
			return false;
		}

		doc.Parse(map->data<char>(), map->get_size());
		if (doc.HasParseError())
		{
			LOGE("Failed to parse JSON: %s\n", ref_json_path.c_str());
			return false;
		}
		has_json = true;
	}

	if (has_json)
	{
		FlushStats parsed_stats = {};
		parse_flush_stats(doc, parsed_stats);
		if (!compare_flush_stats(flush_stats, parsed_stats) && !update)
			return false;
	}

	if (update || !has_json)
	{
		Document write_doc;
		auto &obj = write_doc.SetObject();
		serialize_flush_stats(obj, flush_stats, write_doc.GetAllocator());
		StringBuffer strbuf;
		PrettyWriter<StringBuffer> writer{strbuf};
		write_doc.Accept(writer);

		file_handle = fs.open(ref_json_path, FileMode::WriteOnly);
		if (!file_handle)
		{
			LOGE("Failed to open %s for writing.\n", ref_json_path.c_str());
			return false;
		}

		auto map = file_handle->map_write(strbuf.GetLength());
		if (!map)
		{
			LOGE("Failed to map %s for writing.\n", ref_json_path.c_str());
			return false;
		}

		memcpy(map->mutable_data(), strbuf.GetString(), strbuf.GetLength());
	}

	return true;
}

static int main_inner(const std::string &path, bool update)
{
	OSFilesystem fs(".");

	if (!Context::init_loader(nullptr))
		return EXIT_FAILURE;
	Context ctx;
	Device dev;
	if (!ctx.init_instance_and_device(nullptr, 0, nullptr, 0,
	                                  CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT))
		return EXIT_FAILURE;
	dev.set_context(ctx);

	auto files = fs.list(path);
	for (auto &file : files)
	{
		if (file.type == PathType::File && Path::ext(file.path) == "gs")
		{
			if (!run_gs_dump(fs, dev, file.path, update))
				return EXIT_FAILURE;
		}
	}

	return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
	std::string path;
	bool update = false;
	CLICallbacks cbs;
	cbs.add("--help", [&](CLIParser &parser) { print_help(); parser.end(); });
	cbs.add("--dir", [&](CLIParser &parser) { path = parser.next_string(); });
	cbs.add("--update", [&](CLIParser &) { update = true; });

	CLIParser parser(std::move(cbs), argc - 1, argv + 1);
	if (!parser.parse())
		return EXIT_FAILURE;
	if (parser.is_ended_state())
		return EXIT_SUCCESS;

	if (path.empty())
	{
		LOGE("Must provide --dir.\n");
		return EXIT_FAILURE;
	}

	return main_inner(path, update);
}