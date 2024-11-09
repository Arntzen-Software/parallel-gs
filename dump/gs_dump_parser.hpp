#pragma once

#include <memory>
#include <vector>
#include <stdio.h>
#include "gs_registers.hpp"
#include "gs_renderer.hpp"
#include "image.hpp"

namespace ParallelGS
{
class GSInterface;

// Current GSdx state version.
static constexpr uint32_t STATE_VERSION = 8;

enum class GSDumpPacketType : uint8_t
{
	Transfer = 0,
	Vsync = 1,
	ReadFIFO = 2,
	PrivRegisters = 3,
};

struct DumpHeader
{
	uint32_t version;
	uint32_t state_size;
	uint32_t serial_offset;
	uint32_t serial_size;
	uint32_t crc;
	uint32_t screenshot_width;
	uint32_t screenshot_height;
	uint32_t screenshot_offset;
	uint32_t screenshot_size;
};

class GSDumpParser
{
public:
	bool open(const char *path, uint32_t vram_size, GSInterface *iface);
	bool open_raw(const char *path, uint32_t vram_size, GSInterface *iface);
	bool iterate_until_vsync(bool high_res_scanout = false);
	ScanoutResult consume_vsync_result();
	bool restart();

private:
	struct FileDeleter { void operator()(FILE *file) { if (file) fclose(file); } };
	GSInterface *iface = nullptr;
	std::unique_ptr<FILE, FileDeleter> file;
	std::vector<GIFTagBits> gif_tag_buffer;
	ScanoutResult vsync_result = {};
	uint32_t vram_size;
	bool is_raw = false;

	uint8_t read_u8();
	uint32_t read_u32();
	float read_f32();
	uint64_t read_u64();
	template <typename T>
	void read_reg(T &reg);
	void read_data(void *data, size_t size);
	bool eof = false;

	void read_register_state();
	void read_skip(size_t size);
};
}