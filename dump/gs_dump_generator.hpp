#pragma once

#include "gs_registers.hpp"
#include <stdint.h>
#include <type_traits>
#include <stdio.h>
#include <memory>

namespace ParallelGS
{
class GSInterface;

class GSDumpGenerator
{
public:
	bool init(const char *path, uint32_t vram_size, GSInterface &iface);

	void write_register(RegisterAddr addr, uint64_t payload);

	template <typename T>
	void write_register(RegisterAddr addr, const T &t)
	{
		static_assert(std::is_pod<T>::value &&
		              sizeof(T) == sizeof(uint64_t), "Type is not 64-bit POD union");

		Reg64<T> reg{t};
		write_register(addr, reg.bits);
	}

	void write_packed(const PRIMBits &prim,
	                  const GIFAddr *registers, uint32_t num_registers,
	                  uint32_t num_loops, const void *data);

	void write_vsync(uint32_t field, const GSInterface &iface);

	void write_image_upload(uint32_t addr, uint32_t fmt,
	                        uint32_t width, uint32_t height,
	                        const void *data, size_t size);

private:
	struct FILEDeleter { void operator()(FILE *f) { if (f) fclose(f); } };
	std::unique_ptr<FILE, FILEDeleter> file;

	void write_u32(uint32_t value);
	void write_f32(float value);
	void write_u8(uint8_t value);
	void write_data(const void *data, size_t size);

	void write_register_state(const GSInterface &iface);

	template <typename T>
	void write_reg(const Reg64<T> &t);
};
}