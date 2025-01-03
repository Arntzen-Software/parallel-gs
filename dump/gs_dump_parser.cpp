#include "gs_dump_parser.hpp"
#include "gs_interface.hpp"

namespace ParallelGS
{
bool GSDumpParser::restart()
{
	rewind(file.get());
	eof = feof(file.get()) != 0;

	if (is_raw)
		return !eof;

	// FakeCRC
	if (read_u32() != UINT32_MAX)
		return false;

	// Parse header
	DumpHeader header = {};
	uint32_t header_size = read_u32();
	if (header_size < sizeof(header))
		return false;
	read_data(&header, sizeof(header));
	if (header.version != STATE_VERSION)
		return false;

	// Skip over serial and screenshot data.
	read_skip(header_size - sizeof(header));

	// Context registers
	read_register_state();

	// VRAM
	read_data(iface->map_vram_write(0, vram_size), vram_size);
	iface->end_vram_write(0, vram_size);
	iface->write_register(RegisterAddr::TEXFLUSH, uint64_t(0));

	// GIFPaths
	for (uint32_t i = 0; i < 4; i++)
	{
		auto &gif_path = iface->get_gif_path(i);
		read_data(&gif_path.tag, sizeof(gif_path.tag));
		// The dump format doesn't have current loop counter, so assuming that real implementation
		// decrements loop counter after every iteration?
		gif_path.loop = 0;
		gif_path.reg = read_u32();
	}

	// InternalQ
	iface->get_register_state().internal_q = read_f32();

	// PrivRegisterState
	read_data(&iface->get_priv_register_state(), sizeof(iface->get_priv_register_state()));

	iface->clobber_register_state();

	return !eof;
}

bool GSDumpParser::open_raw(const char *path, uint32_t vram_size_, GSInterface *iface_)
{
	iface = iface_;
	vram_size = vram_size_;
	file.reset(fopen(path, "rb"));
	is_raw = true;
	return file != nullptr;
}

bool GSDumpParser::open(const char *path, uint32_t vram_size_, GSInterface *iface_)
{
	iface = iface_;
	vram_size = vram_size_;
	file.reset(fopen(path, "rb"));
	if (!file)
		return false;

	is_raw = false;
	return restart();
}

void GSDumpParser::read_register_state()
{
	auto &regs = iface->get_register_state();
	read_u32(); // STATE_VERSION
	read_reg(regs.prim);
	read_reg(regs.prmodecont);
	read_reg(regs.texclut);
	read_reg(regs.scanmsk);
	read_reg(regs.texa);
	read_reg(regs.fogcol);
	read_reg(regs.dimx);
	read_reg(regs.dthe);
	read_reg(regs.colclamp);
	read_reg(regs.pabe);
	read_reg(regs.bitbltbuf);
	read_reg(regs.trxdir);
	read_reg(regs.trxpos);
	read_reg(regs.trxreg);
	read_u64(); // Dummy value

	for (auto &ctx : regs.ctx)
	{
		read_reg(ctx.xyoffset);
		read_reg(ctx.tex0);
		read_reg(ctx.tex1);
		read_reg(ctx.clamp);
		read_reg(ctx.miptbl_1_3);
		read_reg(ctx.miptbl_4_6);
		read_reg(ctx.scissor);
		read_reg(ctx.alpha);
		read_reg(ctx.test);
		read_reg(ctx.fba);
		read_reg(ctx.frame);
		read_reg(ctx.zbuf);
	}

	read_reg(regs.rgbaq);
	read_reg(regs.st);
	regs.uv.words[0] = read_u32();
	regs.fog.words[0] = read_u32();
	read_u64(); // Dummy XYZ

	read_u32(); // Dummy GIFReg
	read_u32();

	// Dummy transfer X/Y
	read_u32();
	read_u32();
}

bool GSDumpParser::iterate_until_vsync(bool high_res_scanout)
{
	if (!file)
		return false;

	bool has_transfer = false;

	while (!eof)
	{
		auto type = GSDumpPacketType(read_u8());
		if (eof)
			return false;

		switch (type)
		{
		case GSDumpPacketType::Transfer:
		{
			auto path = read_u8();
			auto size = read_u32();
			auto num_words = size / sizeof(GIFTagBits);
			if (num_words > gif_tag_buffer.size())
				gif_tag_buffer.resize(num_words);
			read_data(gif_tag_buffer.data(), size);
			if (!eof)
				iface->gif_transfer(path, gif_tag_buffer.data(), size);
			has_transfer = true;
			break;
		}

		case GSDumpPacketType::Vsync:
		{
			VSyncInfo vsync = {};
			vsync.phase = read_u8();
			vsync.dst_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			vsync.dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			vsync.dst_access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
			vsync.high_resolution_scanout = high_res_scanout;
			vsync.force_progressive = true;
			vsync.anti_blur = true;
			vsync.overscan = false;
			iface->flush();
			vsync_result = iface->vsync(vsync);
			if (has_transfer)
				return true;
			break;
		}

		case GSDumpPacketType::PrivRegisters:
			read_data(&iface->get_priv_register_state(), sizeof(iface->get_priv_register_state()));
			break;

		case GSDumpPacketType::ReadFIFO:
			// Unimplemented.
			LOGW("Unimplemented ReadFIFO\n");
			uint32_t size = read_u32();
			(void)size;
			return true;
		}
	}

	return !eof;
}

ScanoutResult GSDumpParser::consume_vsync_result()
{
	ScanoutResult result = {};
	std::swap(result, vsync_result);
	return result;
}

uint8_t GSDumpParser::read_u8()
{
	uint8_t v = 0;
	if (fread(&v, sizeof(v), 1, file.get()) != 1)
		eof = true;
	return v;
}

uint32_t GSDumpParser::read_u32()
{
	uint32_t v = 0;
	if (fread(&v, sizeof(v), 1, file.get()) != 1)
		eof = true;
	return v;
}

float GSDumpParser::read_f32()
{
	float v = 0;
	if (fread(&v, sizeof(v), 1, file.get()) != 1)
		eof = true;
	return v;
}

uint64_t GSDumpParser::read_u64()
{
	uint64_t v = 0;
	if (fread(&v, sizeof(v), 1, file.get()) != 1)
		eof = true;
	return v;
}

template <typename T>
void GSDumpParser::read_reg(T &reg)
{
	reg.bits = read_u64();
}

void GSDumpParser::read_skip(size_t size)
{
	fseek(file.get(), long(size), SEEK_CUR);
}

void GSDumpParser::read_data(void *data, size_t size)
{
	if (fread(data, 1, size, file.get()) != size)
		eof = true;
}
}