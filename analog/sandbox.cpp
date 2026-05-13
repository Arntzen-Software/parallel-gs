#include "shaders/slangmosh_iface.hpp"
#include "shaders/slangmosh.hpp"
#include "context.hpp"
#include "device.hpp"
#include <assert.h>

using namespace Vulkan;
using namespace Granite;

static void iir(float *buffer, size_t count, float b0, float b1, float b2, float a1, float a2)
{
	assert(count >= 2);

	// FIR part.
	for (size_t i = count - 1; i >= 2; i--)
		buffer[i] = buffer[i] * b0 + buffer[i - 1] * b1 + buffer[i - 2] * b2;
	buffer[1] = buffer[1] * b0 + buffer[0] * b1;
	buffer[0] = buffer[0] * b0;

	buffer[1] += buffer[0] * a1;
	for (size_t i = 2; i < count; i++)
		buffer[i] += buffer[i - 1] * a1 + buffer[i - 2] * a2;
}

static void run_test(Device &device)
{
	ResourceLayout layout;
	Analog::Shaders<> shaders(device, layout, 0);
	auto *prog = shaders.iir[1];

	struct
	{
		float b0, b1, b2, a1, a2;
	} push = {};

	push.b0 = 1.0f;
	push.b1 = 0.3f;
	push.b2 = -0.2f;
	push.a1 = 0.98f;
	push.a2 = -0.01f;

	float data[1024] = { 1.0f, 0.0f, 3.0f };

	BufferCreateInfo bufinfo = {};
	bufinfo.usage = VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
	bufinfo.size = sizeof(data);
	bufinfo.domain = BufferDomain::Device;
	auto buf = device.create_buffer(bufinfo, data);

	bufinfo.domain = BufferDomain::CachedHost;
	auto readback_buf = device.create_buffer(bufinfo);

	BufferViewCreateInfo view_info = {};
	view_info.buffer = buf.get();
	view_info.format = VK_FORMAT_R32_SFLOAT;
	view_info.range = VK_WHOLE_SIZE;
	auto view = device.create_buffer_view(view_info);

	auto cmd = device.request_command_buffer();
	cmd->set_program(prog);
	cmd->set_storage_buffer_view(0, 0, *view);
	cmd->set_specialization_constant_mask(3);
	cmd->set_specialization_constant(0, 4);
	cmd->set_specialization_constant(1, 16);
	cmd->push_constants(&push, 0, sizeof(push));
	cmd->enable_subgroup_size_control(true);
	cmd->set_subgroup_size_log2(true, 4, 7);

	auto start_ts = cmd->write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	for (int i = 0; i < 1; i++)
		cmd->dispatch(1, 1, 1);
	auto end_ts = cmd->write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
	cmd->copy_buffer(*readback_buf, *buf);
	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);

	Fence fence;
	device.submit(cmd, &fence);
	fence->wait();

	iir(data, 1024, push.b0, push.b1, push.b2, push.a1, push.a2);

	auto *ptr = static_cast<const float *>(device.map_host_buffer(*readback_buf, MEMORY_ACCESS_READ_BIT));
	for (int i = 0; i < 1024; i++)
		LOGI("Value %d = %.3g, reference = %.3g\n", i, ptr[i], data[i]);

	device.wait_idle();
	LOGI("Filter time: %.3f ms\n",
		1e3 * device.convert_device_timestamp_delta(start_ts->get_timestamp_ticks(), end_ts->get_timestamp_ticks()));
}

int main()
{
	if (!Context::init_loader(nullptr))
		return EXIT_FAILURE;

	Context ctx;
	if (!ctx.init_instance_and_device(nullptr, 0, nullptr, 0))
		return EXIT_FAILURE;

	Device dev;
	dev.set_context(ctx);
	run_test(dev);
}
