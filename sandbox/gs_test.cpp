// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#include "gs_renderer.hpp"
#include "device.hpp"
#include "context.hpp"
#include "gs_dump_parser.hpp"
#include "gs_interface.hpp"
#include "gs_dump_generator.hpp"
#include <stdlib.h>

using namespace Vulkan;
using namespace ParallelGS;

static void write_point_primitive(GSDumpGenerator &iface, int x0, int y0)
{
	struct Vertex
	{
		PackedRGBAQBits rgbaq;
		PackedXYZBits xyz;
	} vertices[2] = {};

	for (auto &vert : vertices)
	{
		vert.rgbaq.R = 0x40;
		vert.rgbaq.G = 0xff;
		vert.rgbaq.B = 0x60;
		vert.rgbaq.A = 0xff;
	}

	vertices[1].rgbaq.R = 0xff;

	vertices[0].xyz.X = x0 << SUBPIXEL_BITS;
	vertices[0].xyz.Y = y0 << SUBPIXEL_BITS;
	vertices[1].xyz.X = (x0 << SUBPIXEL_BITS) + 9;
	vertices[1].xyz.Y = (y0 << SUBPIXEL_BITS) + 9;

	PRIMBits prim = {};
	prim.TME = 1;
	prim.IIP = 1;
	prim.PRIM = 0; // Point

	static const GIFAddr addr[] = { GIFAddr::RGBAQ, GIFAddr::XYZ2 };
	iface.write_packed(prim, addr, 2, 1, vertices);

	prim.TME = 0;
	prim.IIP = 1;
	prim.PRIM = 0; // Point

	iface.write_packed(prim, addr, 2, 1, vertices + 1);
}

static void write_sprite_primitive(GSDumpGenerator &iface, int x0, int y0, int x1, int y1)
{
	struct Vertex
	{
		PackedSTBits st;
		PackedRGBAQBits rgbaq;
		PackedXYZBits xyz;
	} vertices[2] = {};

	for (auto &vert : vertices)
	{
		vert.rgbaq.R = 0x80;
		vert.rgbaq.G = 0x80;
		vert.rgbaq.B = 0x80;
		vert.rgbaq.A = 0xff;
	}

	vertices[0].xyz.X = x0 << SUBPIXEL_BITS;
	vertices[0].xyz.Y = y0 << SUBPIXEL_BITS;
	vertices[1].xyz.X = x1 << SUBPIXEL_BITS;
	vertices[1].xyz.Y = y1 << SUBPIXEL_BITS;

	//vertices[1].rgbaq.R = 0;

#if 1
	vertices[0].st.S = 0.0f;
	vertices[0].st.T = 0.0f;
	vertices[0].st.Q = 1.0f;
	vertices[1].st.S = 1.0f;
	vertices[1].st.T = 1.0f;
	vertices[1].st.Q = 1.0f;
#else
	vertices[0].st.S = 0.0f;
	vertices[0].st.T = 0.9f;
	vertices[0].st.Q = 1.0f;
	vertices[1].st = vertices[0].st;
#endif

	PRIMBits prim = {};
	prim.TME = 1;
	prim.IIP = 1;
	prim.PRIM = 6; // Sprite

	static const GIFAddr addr[] = { GIFAddr::ST, GIFAddr::RGBAQ, GIFAddr::XYZ2 };
	iface.write_packed(prim, addr, 3, 2, vertices);
}

static void write_line_primitive(GSDumpGenerator &iface, int x0, int y0, int x1, int y1)
{
	struct Vertex
	{
		PackedRGBAQBits rgbaq;
		PackedXYZBits xyz;
	} vertices[2] = {};

	for (auto &vert : vertices)
	{
		vert.rgbaq.R = 0x7f;
		vert.rgbaq.G = 0x7f;
		vert.rgbaq.B = 0x7f;
		vert.rgbaq.A = 0x80;
	}

	vertices[0].xyz.X = x0 << SUBPIXEL_BITS;
	vertices[0].xyz.Y = y0 << SUBPIXEL_BITS;
	vertices[1].xyz.X = x1 << SUBPIXEL_BITS;
	vertices[1].xyz.Y = y1 << SUBPIXEL_BITS;

	vertices[0].xyz.X -= 7;
	vertices[0].xyz.Y -= 7;
	vertices[1].xyz.X -= 7;
	vertices[1].xyz.Y -= 7;

	//vertices[0].rgbaq.R = 0;
	//vertices[1].rgbaq.B = 0;

	PRIMBits prim = {};
	prim.TME = 0;
	prim.IIP = 1;
	prim.PRIM = 2; // Line
	prim.ABE = 1;

	// Additive blend
	ALPHABits alpha = {};
	alpha.A = 0;
	alpha.B = 2;
	alpha.C = 0;
	alpha.D = 1;
	iface.write_register(RegisterAddr::ALPHA_1, alpha);

	static const GIFAddr addr[] = { GIFAddr::RGBAQ, GIFAddr::XYZ2 };
	iface.write_packed(prim, addr, 2, 2, vertices);
}

static void write_quad_primitive(GSDumpGenerator &iface, int x0, int y0, int x1, int y1, bool is_fg)
{
	struct Vertex
	{
		PackedSTBits st;
		PackedRGBAQBits rgbaq;
		PackedXYZBits xyz;
	} vertices[3] = {};

	for (auto &vert : vertices)
	{
		vert.rgbaq.A = 0x80;
	}

#if 0
	vertices[0].rgbaq.R = 0xff;
	vertices[1].rgbaq.G = 0xff;
	vertices[2].rgbaq.B = 0xff;
	vertices[3].rgbaq.R = 0x80;
	vertices[3].rgbaq.G = 0x80;
	vertices[3].rgbaq.B = 0x80;
#endif

	vertices[0].st.S = 0.0f;
	vertices[0].st.T = 0.0f;
	vertices[0].st.Q = 0.5f;
	vertices[1].st.S = 1.0f;
	vertices[1].st.T = 0.0f;
	vertices[1].st.Q = 0.5f;
	vertices[2].st.S = 0.0f;
	vertices[2].st.T = 1.0f;
	vertices[2].st.Q = 0.5f;

	vertices[0].xyz.X = x0 << SUBPIXEL_BITS;
	vertices[0].xyz.Y = y0 << SUBPIXEL_BITS;
	vertices[1].xyz.X = x1 << SUBPIXEL_BITS;
	vertices[1].xyz.Y = y0 << SUBPIXEL_BITS;
	vertices[2].xyz.X = x0 << SUBPIXEL_BITS;
	vertices[2].xyz.Y = y1 << SUBPIXEL_BITS;

	if (is_fg)
	{
		vertices[0].xyz.X += 4; // 0.5 pixel offset
		vertices[0].xyz.Y += 4;
		vertices[1].xyz.X += 4;
		vertices[1].xyz.Y += 4;
		vertices[2].xyz.X += 4;
		vertices[2].xyz.Y += 4;
		vertices[0].rgbaq.G = 100;
		vertices[1].rgbaq.G = 100 + 32;
		vertices[2].rgbaq.G = 100;
	}
	else
	{
		vertices[1].xyz.X += 128;
		vertices[2].xyz.Y += 128;
		vertices[0].rgbaq.R = 0x20;
		vertices[1].rgbaq.R = 0x20;
		vertices[2].rgbaq.R = 0x20;
		vertices[0].rgbaq.B = 0;
		vertices[1].rgbaq.B = 0;
		vertices[2].rgbaq.B = 0;
		vertices[0].rgbaq.G = 0;
		vertices[1].rgbaq.G = 0;
		vertices[2].rgbaq.G = 0;
	}

	const uint32_t Z = is_fg ? 100 : 50;
	vertices[0].xyz.Z = Z;
	vertices[1].xyz.Z = Z;
	vertices[2].xyz.Z = Z;

	PRIMBits prim = {};
	prim.TME = 0;
	prim.IIP = 1;
	prim.AA1 = is_fg ? 1 : 0;
	prim.PRIM = 4; // TriStrip

	ALPHABits alpha = {};
	alpha.A = 0;
	alpha.B = 1;
	alpha.C = 0;
	alpha.D = 1;
	iface.write_register(RegisterAddr::ALPHA_1, alpha);

	static const GIFAddr addr[] = { GIFAddr::ST, GIFAddr::RGBAQ, GIFAddr::XYZ2 };
	iface.write_packed(prim, addr, 3, 3, vertices);
}

static void setup_frame_buffer(GSDumpGenerator &iface)
{
	TESTBits test = {};
	test.ZTE = 1;
	test.ZTST = 3;
	iface.write_register(RegisterAddr::TEST_1, test);

	FRAMEBits frame = {};
	frame.PSM = PSMCT32;
	frame.FBW = 640 / 64;
	iface.write_register(RegisterAddr::FRAME_1, frame);

	ZBUFBits zbuf = {};
	zbuf.ZMSK = 0;
	zbuf.ZBP = 0x100000 / PAGE_ALIGNMENT_BYTES;
	iface.write_register(RegisterAddr::ZBUF_1, zbuf);

	SCISSORBits scissor = {};
	scissor.SCAX0 = 0;
	scissor.SCAY0 = 0;
	scissor.SCAX1 = 350 - 1;
	scissor.SCAY1 = 350 - 1;
	iface.write_register(RegisterAddr::SCISSOR_1, scissor);
}

static constexpr uint32_t PALETTE_ADDR = 3 * 1024 * 1024;
static constexpr uint32_t TEXTURE_ADDR = 2 * 1024 * 1024;

static void upload_palettes(GSDumpGenerator &iface)
{
	constexpr uint16_t RED = 0x1f;
	constexpr uint16_t GREEN = 0x1f << 5;
	constexpr uint16_t BLUE = 0x1f << 10;
	constexpr uint16_t WHITE = RED | GREEN | BLUE;

	constexpr uint16_t HALF_RED = 0xf;
	constexpr uint16_t HALF_GREEN = 0xf << 5;
	constexpr uint16_t HALF_BLUE = 0xf << 10;
	constexpr uint16_t HALF_WHITE = HALF_RED | HALF_GREEN | HALF_BLUE;

	static const uint16_t texture[] = {
		RED, GREEN, BLUE, WHITE,
		HALF_RED, HALF_GREEN, HALF_BLUE, HALF_WHITE,
		WHITE, BLUE, GREEN, RED,
		HALF_WHITE, HALF_BLUE, HALF_GREEN, HALF_RED,
	};

	static const uint16_t texture1[] = {
		HALF_RED, HALF_GREEN, HALF_BLUE, HALF_WHITE,
		RED, GREEN, BLUE, WHITE,
		HALF_WHITE, HALF_BLUE, HALF_GREEN, HALF_RED,
		WHITE, BLUE, GREEN, RED,
	};

	iface.write_image_upload(PALETTE_ADDR, PSMCT16, 8, 2, texture, sizeof(texture));
	iface.write_image_upload(PALETTE_ADDR + BLOCK_ALIGNMENT_BYTES, PSMCT16, 8, 2, texture1, sizeof(texture1));
}

static void run_test(GSDumpGenerator &iface)
{
	setup_frame_buffer(iface);
	upload_palettes(iface);

	const uint8_t texture[] = {
		0, 1, 2, 3, 4, 5, 6, 7,
		8, 9, 10, 11, 12, 13, 14, 15,
		0, 1, 2, 3, 4, 5, 6, 7,
		8, 9, 10, 11, 12, 13, 14, 15,
		0, 1, 2, 3, 4, 5, 6, 7,
		8, 9, 10, 11, 12, 13, 14, 15,
		0, 1, 2, 3, 4, 5, 6, 7,
		8, 9, 10, 11, 12, 13, 14, 15,
	};

	iface.write_image_upload(TEXTURE_ADDR, PSMT8, 8, 8, texture, sizeof(texture));
	iface.write_register(RegisterAddr::TEXFLUSH, uint64_t(0));

	TEX0Bits tex0 = {};
	tex0.TBP0 = TEXTURE_ADDR / BLOCK_ALIGNMENT_BYTES;
	tex0.TBW = 8 / 64;
	tex0.PSM = PSMT8;
	tex0.TW = 3;
	tex0.TH = 3;
	tex0.TCC = 0;
	tex0.TFX = 0;
	tex0.CPSM = PSMCT16;
	tex0.CSM = 0;
	tex0.CLD = 1;

	tex0.CSA = 0;
	tex0.CBP = PALETTE_ADDR / BLOCK_ALIGNMENT_BYTES;
	iface.write_register(RegisterAddr::TEX0_1, tex0);

	tex0.CSA = 1;
	tex0.CBP = (PALETTE_ADDR + BLOCK_ALIGNMENT_BYTES) / BLOCK_ALIGNMENT_BYTES;
	iface.write_register(RegisterAddr::TEX0_1, tex0);

	tex0.CLD = 0;
	tex0.CSA = 0;
	tex0.CBP = PALETTE_ADDR / BLOCK_ALIGNMENT_BYTES;
	iface.write_register(RegisterAddr::TEX0_1, tex0);

#if 0
	write_quad_primitive(iface, 50, 50, 150, 150);

	tex0.CSA = 1;
	tex0.CBP = (PALETTE_ADDR + BLOCK_ALIGNMENT_BYTES) / BLOCK_ALIGNMENT_BYTES;
	iface.write_register(RegisterAddr::TEX0_1, tex0);

	TEX1Bits tex1 = {};
	tex1.MMAG = 0;
	tex1.MMIN = 3;
	iface.write_register(RegisterAddr::TEX1_1, tex1);

	CLAMPBits clamp = {};
	clamp.WMS = 3;
	clamp.WMT = 3;
	clamp.MINU = 2;
	clamp.MINV = 2;
	clamp.MAXU = 4;
	iface.write_register(RegisterAddr::CLAMP_1, clamp);

	// Technically needed when replacing which palette we're using.
	iface.write_register(RegisterAddr::TEXFLUSH, uint64_t(0));
	write_quad_primitive(iface, 200, 50, 300, 150);

	write_point_primitive(iface, 50, 200);

	constexpr int OFF = 80;

	// Top-left to Bottom-right
	write_sprite_primitive(iface, 50, 200, 100, 250);

	// Bottom-right to top-left
	write_sprite_primitive(iface, 100 + OFF, 250, 50 + OFF, 200);

	// Top-right to bottom-left
	write_sprite_primitive(iface, 100, 200 + OFF, 50, 250 + OFF);

	// Bottom-left to top-right
	write_sprite_primitive(iface, 50 + OFF, 250 + OFF, 100 + OFF, 200 + OFF);
#else
	write_quad_primitive(iface, 0, 0, 16, 16, false);
	//write_quad_primitive(iface, 0, 0, 4, 4, false);
	//write_line_primitive(iface, 50, 50, 100, 100);
	//write_line_primitive(iface, 100, 100, 120, 110);
	//write_line_primitive(iface, 120, 110, 130, 110);
	//write_line_primitive(iface, 50, 81, 100, 21);
#endif
}

int main()
{
	if (!Context::init_loader(nullptr))
		return EXIT_FAILURE;

	Context ctx;
	ctx.set_num_thread_indices(1);
	if (!ctx.init_instance_and_device(nullptr, 0, nullptr, 0))
		return EXIT_FAILURE;

	Device device;
	device.set_context(ctx);
	device.init_frame_contexts(4);

	GSInterface iface;
	GSOptions opts = {};
	if (!iface.init(&device, opts))
		return EXIT_FAILURE;

	DebugMode debug_mode;
	debug_mode.feedback_render_target = true;
	iface.set_debug_mode(debug_mode);

	bool use_rdoc = Device::init_renderdoc_capture();
	if (use_rdoc)
		device.begin_renderdoc_capture();

	{
		GSDumpGenerator dump;
		if (!dump.init("/tmp/test.gs", 4 * 1024 * 1024, iface))
			return EXIT_FAILURE;

		run_test(dump);

		auto &priv = iface.get_priv_register_state();

		priv.pmode.EN1 = 1;
		priv.pmode.EN2 = 0;
		priv.pmode.CRTMD = 1;
		priv.pmode.MMOD = 1;
		priv.smode1.CMOD = 2;
		priv.smode1.LC = 32;
		priv.bgcolor.R = 0x30;
		priv.bgcolor.G = 0x40;
		priv.bgcolor.B = 0x50;
		priv.pmode.SLBG = 1;
		priv.pmode.ALP = 0xff;
		priv.smode2.INT = 1;

		priv.dispfb1.FBP = 0;
		priv.dispfb1.FBW = 640 / 64;
		priv.dispfb1.PSM = PSMCT32;
		priv.dispfb1.DBX = 0;
		priv.dispfb1.DBY = 0;
		priv.display1.DX = 640;
		priv.display1.DY = 50;
		priv.display1.MAGH = 3;
		priv.display1.MAGV = 0;
		priv.display1.DW = 640 * 4 - 1;
		priv.display1.DH = 480 - 1;

		dump.write_vsync(0, iface);
		dump.write_vsync(1, iface);
	}

	GSDumpParser parser;
	if (!parser.open("/tmp/test.gs", 4 * 1024 * 1024, &iface))
		return EXIT_FAILURE;

	if (!parser.iterate_until_vsync())
		return EXIT_FAILURE;

	if (use_rdoc)
		device.end_renderdoc_capture();
}