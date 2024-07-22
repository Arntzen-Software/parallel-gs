#include "context.hpp"
#include "device.hpp"
#include "filesystem.hpp"
#include "thread_group.hpp"
#include "global_managers_init.hpp"
#include "math.hpp"
#include "texture_files.hpp"

using namespace Granite;
using namespace Vulkan;

static Vulkan::ImageHandle compute_luminance_hierarchy(Device &device, const char *path)
{
	auto rgb = load_texture_from_file(*GRANITE_FILESYSTEM(), path);
	auto info = ImageCreateInfo::immutable_image(rgb.get_layout());
	auto staging = device.create_image_staging_buffer(rgb.get_layout());
	auto img = device.create_image_from_staging_buffer(info, &staging);

	uint32_t luma_width = (img->get_width() + 31u) & ~31u;
	uint32_t luma_height = (img->get_height() + 31u) & ~31u;

	info = ImageCreateInfo::immutable_2d_image(luma_width, luma_height, VK_FORMAT_R8_UNORM);
	info.levels = 5;
	info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;

	auto luma_img = device.create_image(info);
	ImageViewHandle views[5];

	for (unsigned level = 0; level < info.levels; level++)
	{
		ImageViewCreateInfo view_info = {};
		view_info.image = luma_img.get();
		view_info.format = VK_FORMAT_R8_UNORM;
		view_info.base_level = level;
		view_info.levels = 1;
		view_info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
		view_info.view_type = VK_IMAGE_VIEW_TYPE_2D;
		views[level] = device.create_image_view(view_info);
	}

	auto cmd = device.request_command_buffer();

	cmd->set_program("assets://luminance-pyramid.comp");
	for (unsigned level = 0; level < info.levels; level++)
		cmd->set_storage_texture(0, level, *views[level]);
	cmd->set_texture(0, 5, img->get_view(), StockSampler::NearestClamp);
	struct Push
	{
		vec2 inv_resolution;
		int mips;
	} push = {};
	push.inv_resolution.x = 1.0f / float(img->get_width());
	push.inv_resolution.y = 1.0f / float(img->get_height());
	push.mips = 5;
	cmd->push_constants(&push, 0, sizeof(push));
	cmd->image_barrier(*luma_img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
	                   0, 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
	cmd->dispatch(luma_width / 32, luma_height / 32, 1);
	cmd->image_barrier(*luma_img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
	device.submit(cmd);

	return luma_img;
}

static void run_optical_flow(Device &device, const Image &current, const Image &prev)
{
	uint32_t mv_width = current.get_width() / 8;
	uint32_t mv_height = current.get_height() / 8;
	auto cmd = device.request_command_buffer();

	SamplerCreateInfo samp_info = {};
	samp_info.address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	samp_info.address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	samp_info.address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	samp_info.mag_filter = VK_FILTER_NEAREST;
	samp_info.min_filter = VK_FILTER_NEAREST;
	samp_info.mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

	samp_info.border_color = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
	auto float_border_sampler = device.create_sampler(samp_info);
	samp_info.border_color = VK_BORDER_COLOR_INT_TRANSPARENT_BLACK;
	auto int_border_sampler = device.create_sampler(samp_info);

	ImageHandle previous_level_upscale;

	for (int level = 4; level >= 0; level--)
	{
		uint32_t mv_width_for_level = (mv_width + ((1 << level) - 1)) >> level;
		uint32_t mv_height_for_level = (mv_height + ((1 << level) - 1)) >> level;
		auto info = ImageCreateInfo::immutable_2d_image(
				mv_width_for_level, mv_height_for_level, VK_FORMAT_R8G8_SINT);
		info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
		info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

		ImageViewCreateInfo view_info = {};
		view_info.format = VK_FORMAT_R8_UNORM;
		view_info.view_type = VK_IMAGE_VIEW_TYPE_2D;
		view_info.base_level = level;
		view_info.levels = 1;
		view_info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
		view_info.image = &current;
		auto current_view = device.create_image_view(view_info);
		view_info.image = &prev;
		auto prev_view = device.create_image_view(view_info);

		ImageHandle search_img = previous_level_upscale ? previous_level_upscale : device.create_image(info);
		search_img->set_layout(Layout::General);
		auto filter_img = device.create_image(info);

		info.width *= 2;
		info.height *= 2;
		auto upscale_img = device.create_image(info);

		if (!previous_level_upscale)
		{
			cmd->image_barrier(*search_img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
			                   0, 0,
			                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
		}

		cmd->image_barrier(*filter_img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
		                   0, 0,
		                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

		cmd->image_barrier(*upscale_img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
		                   0, 0,
		                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

		{
			struct Push
			{
				vec2 inv_resolution;
			} push = {};

			push.inv_resolution.x = 1.0f / float(current_view->get_view_width());
			push.inv_resolution.y = 1.0f / float(current_view->get_view_height());

			cmd->set_program("assets://motion-search.comp");
			cmd->set_specialization_constant_mask(1);
			cmd->push_constants(&push, 0, sizeof(push));

			cmd->set_storage_texture(0, 0, search_img->get_view());
			cmd->set_texture(0, 1, *current_view, StockSampler::NearestClamp);
			cmd->set_texture(0, 2, *prev_view, *float_border_sampler);
			cmd->set_specialization_constant(0, uint32_t(bool(previous_level_upscale)));
			cmd->dispatch(mv_width_for_level, mv_height_for_level, 1);
			cmd->set_specialization_constant_mask(0);
		}

		cmd->image_barrier(*search_img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
		                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

		{
			struct Push
			{
				uvec2 resolution;
				vec2 inv_resolution;
			} push = {};

			push.resolution.x = mv_width_for_level;
			push.resolution.y = mv_height_for_level;
			push.inv_resolution.x = 1.0f / float(mv_width_for_level);
			push.inv_resolution.y = 1.0f / float(mv_height_for_level);

			cmd->set_program("assets://motion-filter.comp");
			cmd->push_constants(&push, 0, sizeof(push));

			cmd->set_storage_texture(0, 0, filter_img->get_view());
			cmd->set_texture(0, 1, search_img->get_view(), *int_border_sampler);
			cmd->dispatch((mv_width_for_level + 7) / 8, (mv_height_for_level + 7) / 8, 1);
		}

		cmd->image_barrier(*filter_img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

		{
			struct Push
			{
				uvec2 low_res_resolution_minus_1;
				vec2 inv_resolution;
			} push = {};

			push.low_res_resolution_minus_1.x = mv_width_for_level - 1;
			push.low_res_resolution_minus_1.y = mv_height_for_level - 1;
			push.inv_resolution.x = 0.5f / float(mv_width_for_level);
			push.inv_resolution.y = 0.5f / float(mv_height_for_level);

			if (level > 0)
			{
				int next_level = level - 1;
				mv_width_for_level = (mv_width + ((1 << next_level) - 1)) >> next_level;
				mv_height_for_level = (mv_height + ((1 << next_level) - 1)) >> next_level;
			}
			else
			{
				mv_width_for_level *= 2;
				mv_height_for_level *= 2;
			}

			cmd->set_program("assets://motion-upscale.comp");
			cmd->push_constants(&push, 0, sizeof(push));

			cmd->set_storage_texture(0, 0, upscale_img->get_view());
			cmd->set_texture(0, 1, *current_view, StockSampler::NearestClamp);
			cmd->set_texture(0, 2, *prev_view, StockSampler::NearestClamp);
			cmd->set_texture(0, 3, filter_img->get_view(), StockSampler::NearestClamp);
			cmd->dispatch(mv_width_for_level, mv_height_for_level, 1);
		}

		previous_level_upscale = upscale_img;
	}

	device.submit(cmd);
}

static void run_test(Device &device)
{
	auto luma0 = compute_luminance_hierarchy(device, "/tmp/vsync1.png");
	auto luma1 = compute_luminance_hierarchy(device, "/tmp/vsync2.png");
	run_optical_flow(device, *luma1, *luma0);
}

static int main_inner()
{
	if (!Context::init_loader(nullptr))
		return EXIT_FAILURE;
	Context::SystemHandles handles = {};
	handles.asset_manager = GRANITE_ASSET_MANAGER();
	handles.filesystem = GRANITE_FILESYSTEM();
	handles.thread_group = GRANITE_THREAD_GROUP();
	Context ctx;
	ctx.set_system_handles(handles);
	ctx.set_num_thread_indices(3);
	if (!ctx.init_instance_and_device(nullptr, 0, nullptr, 0))
		return EXIT_FAILURE;

	Device dev;
	dev.set_context(ctx);

	bool has_rdoc = Device::init_renderdoc_capture();
	if (has_rdoc)
		dev.begin_renderdoc_capture();
	run_test(dev);
	if (has_rdoc)
		dev.end_renderdoc_capture();

	return EXIT_SUCCESS;
}

int main()
{
	Global::init(Global::MANAGER_FEATURE_DEFAULT_BITS, 2);
	Filesystem::setup_default_filesystem(GRANITE_FILESYSTEM(), ASSET_DIRECTORY);
	int ret = main_inner();
	Global::deinit();
	return ret;
}