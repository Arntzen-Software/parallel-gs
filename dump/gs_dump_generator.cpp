#include "gs_dump_generator.hpp"
#include "gs_interface.hpp"
#include "gs_dump_parser.hpp"
#include <stdexcept>
#include "shaders/swizzle_utils.h"

namespace ParallelGS
{
void GSDumpGenerator::write_u32(uint32_t value)
{
	if (fwrite(&value, sizeof(value), 1, file.get()) != 1)
		throw std::runtime_error("Failed to write.");
}

void GSDumpGenerator::write_u8(uint8_t value)
{
	if (fwrite(&value, sizeof(value), 1, file.get()) != 1)
		throw std::runtime_error("Failed to write.");
}

void GSDumpGenerator::write_f32(float value)
{
	if (fwrite(&value, sizeof(value), 1, file.get()) != 1)
		throw std::runtime_error("Failed to write.");
}

void GSDumpGenerator::write_data(const void *data, size_t size)
{
	if (fwrite(data, 1, size, file.get()) != size)
		throw std::runtime_error("Failed to write.");
}

template <typename T>
void GSDumpGenerator::write_reg(const Reg64<T> &t)
{
	if (fwrite(&t.bits, sizeof(t.bits), 1, file.get()) != 1)
		throw std::runtime_error("Failed to write.");
}

void GSDumpGenerator::write_register_state(const GSInterface &iface)
{
	auto &regs = iface.get_register_state();

	write_u32(STATE_VERSION);
	write_reg(regs.prim);
	write_reg(regs.prmodecont);
	write_reg(regs.texclut);
	write_reg(regs.scanmsk);
	write_reg(regs.texa);
	write_reg(regs.fogcol);
	write_reg(regs.dimx);
	write_reg(regs.dthe);
	write_reg(regs.colclamp);
	write_reg(regs.pabe);
	write_reg(regs.bitbltbuf);
	write_reg(regs.trxdir);
	write_reg(regs.trxpos);
	write_reg(regs.trxreg);
	write_reg(regs.trxreg); // Dummy value

	for (const auto &ctx : regs.ctx)
	{
		write_reg(ctx.xyoffset);
		write_reg(ctx.tex0);
		write_reg(ctx.tex1);
		write_reg(ctx.clamp);
		write_reg(ctx.miptbl_1_3);
		write_reg(ctx.miptbl_4_6);
		write_reg(ctx.scissor);
		write_reg(ctx.alpha);
		write_reg(ctx.test);
		write_reg(ctx.fba);
		write_reg(ctx.frame);
		write_reg(ctx.zbuf);
	}

	write_reg(regs.rgbaq);
	write_reg(regs.st);
	write_u32(regs.uv.words[0]);
	write_u32(regs.fog.words[0]);
	// XYZ register, fill with dummy.
	write_reg(Reg64<XYZBits>{0});

	write_u32(UINT32_MAX); // Dummy GIFReg
	write_u32(UINT32_MAX);

	// Dummy transfer X/Y
	write_u32(0);
	write_u32(0);
}

void GSDumpGenerator::write_vsync(uint32_t field, const ParallelGS::GSInterface &iface)
{
	if (!file)
		return;

	write_u8(uint8_t(GSDumpPacketType::PrivRegisters));
	write_data(&iface.get_priv_register_state(), sizeof(iface.get_priv_register_state()));

	write_u8(uint8_t(GSDumpPacketType::Vsync));
	write_u8(field);
}

void GSDumpGenerator::write_register(RegisterAddr addr, uint64_t payload)
{
	GIFTagBits tag = {};
	tag.NLOOP = 1;
	tag.EOP = 1;
	tag.REGS = uint32_t(GIFAddr::A_D);
	tag.NREG = 1;
	tag.FLG = 0; // PACKED

	const uint64_t data[2] = { payload, uint64_t(addr) };
	write_u8(uint8_t(GSDumpPacketType::Transfer));
	write_u8(1);
	write_u32(sizeof(GIFTagBits) * 2);
	write_data(&tag, sizeof(tag));
	write_data(data, sizeof(data));
}

void GSDumpGenerator::write_packed(const PRIMBits &prim, const GIFAddr *registers,
                                   uint32_t num_registers, uint32_t num_loops, const void *data)
{
	GIFTagBits tag = {};
	tag.PRE = 1;
	tag.PRIM = Reg64<PRIMBits>{prim}.bits;
	tag.NREG = num_registers & 15;
	tag.NLOOP = num_loops;
	tag.EOP = 1;
	tag.FLG = 0; // PACKED

	for (uint32_t i = 0; i < num_registers; i++)
		tag.REGS |= uint64_t(registers[i]) << (4 * i);

	write_u8(uint8_t(GSDumpPacketType::Transfer));
	write_u8(1);
	write_u32(sizeof(GIFTagBits) * (1 + num_registers * num_loops));
	write_data(&tag, sizeof(tag));
	write_data(data, sizeof(GIFTagBits) * num_registers * num_loops);
}

void GSDumpGenerator::write_image_upload(uint32_t addr, uint32_t fmt,
										 uint32_t width, uint32_t height,
										 const void *data_, size_t size)
{
	BITBLTBUFBits bitblt = {};
	bitblt.DPSM = fmt;
	bitblt.SPSM = fmt;
	bitblt.DBP = addr / PGS_BLOCK_ALIGNMENT_BYTES;
	bitblt.DBW = width / 64;

	write_register(RegisterAddr::BITBLTBUF, bitblt);

	TRXPOSBits trxpos = {};
	write_register(RegisterAddr::TRXPOS, trxpos);

	TRXREGBits trxreg = {};
	trxreg.RRW = width;
	trxreg.RRH = height;
	write_register(RegisterAddr::TRXREG, trxreg);

	TRXDIRBits trxdir = {};
	trxdir.XDIR = 0; // HOST-to-LOCAL
	write_register(RegisterAddr::TRXDIR, trxdir);

	GIFTagBits tag = {};
	tag.FLG = 2;

	auto *data = static_cast<const uint8_t *>(data_);

	for (size_t i = 0; i < size; )
	{
		size_t to_write = std::min<size_t>(size - i, 32 * 1024 * sizeof(GIFTagBits));
		tag.EOP = uint32_t(i + to_write == size);
		tag.NLOOP = to_write / sizeof(GIFTagBits);

		write_u8(uint8_t(GSDumpPacketType::Transfer));
		write_u8(1);
		write_u32(to_write + sizeof(tag));
		write_data(&tag, sizeof(tag));
		write_data(data + i, to_write);

		i += to_write;
	}
}

bool GSDumpGenerator::init(const char *path, uint32_t vram_size, GSInterface &iface)
{
	file.reset(fopen(path, "wb"));
	if (!file)
		return false;

	const void *vram = iface.map_vram_read(0, vram_size);

	try
	{
		// GS dump format from PCSX2.

		// fakeCRC
		write_u32(UINT32_MAX);

		DumpHeader header = {};

		header.version = STATE_VERSION;
		header.state_size = vram_size + (18 + 12 * 2) * sizeof(uint64_t) + 7 * sizeof(uint32_t) +
		                    4 * 5 * sizeof(uint32_t) + sizeof(float);

		write_u32(sizeof(DumpHeader));
		write_data(&header, sizeof(header));

		write_register_state(iface);
		write_data(vram, vram_size);

		// 4 GIF paths
		for (int i = 0; i < 4; i++)
		{
			auto gif_path = iface.get_gif_path(i);
			gif_path.tag.NLOOP -= gif_path.loop;
			write_data(&gif_path.tag, sizeof(gif_path.tag));
			write_u32(gif_path.reg);
		}

		// internal_Q
		write_f32(iface.get_register_state().internal_q);

		// PrivRegisterState
		write_data(&iface.get_priv_register_state(), sizeof(iface.get_priv_register_state()));
	}
	catch (const std::exception &e)
	{
		LOGE("Failed to write: %s\n", e.what());
		file.reset();
		return false;
	}

	return true;
}
}
