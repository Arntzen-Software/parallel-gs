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

static void write_clear_quad(GSDumpGenerator &iface, int x0, int y0, int x1, int y1)
{
	struct Vertex
	{
		PackedRGBAQBits rgba;
		PackedXYZBits xyz;
	} vertices[2] = {};

	vertices[0].xyz.X = x0 << PGS_SUBPIXEL_BITS;
	vertices[0].xyz.Y = y0 << PGS_SUBPIXEL_BITS;
	vertices[1].xyz.X = x1 << PGS_SUBPIXEL_BITS;
	vertices[1].xyz.Y = y1 << PGS_SUBPIXEL_BITS;

	vertices[0].rgba.R = 0x10;
	vertices[0].rgba.G = 0x10;
	vertices[0].rgba.B = 0x10;
	vertices[0].rgba.A = 0x80;
	vertices[1].rgba.R = 0x10;
	vertices[1].rgba.G = 0x10;
	vertices[1].rgba.B = 0x10;
	vertices[1].rgba.A = 0x80;

	PRIMBits prim = {};
	prim.PRIM = 6; // Sprite

	static const GIFAddr addr[] = { GIFAddr::RGBAQ, GIFAddr::XYZ2 };
	iface.write_packed(prim, addr, 2, 2, vertices);
}

static void write_line_primitive(GSDumpGenerator &iface, float x0, float y0, float x1, float y1)
{
	struct Vertex
	{
		PackedRGBAQBits rgba;
		PackedXYZBits xyz;
	} vertices[2] = {};

	for (auto &v : vertices)
	{
		v.rgba.R = 0xff;
		v.rgba.G = 0xff;
		v.rgba.B = 0xff;
		v.rgba.A = 0x80;
	}

	vertices[0].xyz.X = int(x0 * float(1 << PGS_SUBPIXEL_BITS));
	vertices[0].xyz.Y = int(y0 * float(1 << PGS_SUBPIXEL_BITS));
	vertices[1].xyz.X = int(x1 * float(1 << PGS_SUBPIXEL_BITS));
	vertices[1].xyz.Y = int(y1 * float(1 << PGS_SUBPIXEL_BITS));

	PRIMBits prim = {};
	prim.PRIM = int(PRIMType::LineList);

	static const GIFAddr addr[] = { GIFAddr::RGBAQ, GIFAddr::XYZ2 };
	iface.write_packed(prim, addr, 2, 2, vertices);
}

static void write_sprite_primitive(GSDumpGenerator &iface, int x0, int y0, int x1, int y1)
{
	struct Vertex
	{
		PackedUVBits uv;
		PackedXYZBits xyz;
	} vertices[2] = {};

	vertices[0].xyz.X = x0 << PGS_SUBPIXEL_BITS;
	vertices[0].xyz.Y = y0 << PGS_SUBPIXEL_BITS;
	vertices[1].xyz.X = x1 << PGS_SUBPIXEL_BITS;
	vertices[1].xyz.Y = y1 << PGS_SUBPIXEL_BITS;

	vertices[0].uv.U = 0;
	vertices[0].uv.V = 0;
	vertices[1].uv.U = 16 << PGS_SUBPIXEL_BITS;
	vertices[1].uv.V = 16 << PGS_SUBPIXEL_BITS;

	PRIMBits prim = {};
	prim.TME = 1;
	prim.FST = 1;
	prim.ABE = 1;
	prim.PRIM = 6; // Sprite

	static const GIFAddr addr[] = { GIFAddr::UV, GIFAddr::XYZ2 };
	iface.write_packed(prim, addr, 2, 2, vertices);
}

static void setup_frame_buffer(GSDumpGenerator &iface)
{
	TESTBits test = {};
	test.ZTE = 0;
	iface.write_register(RegisterAddr::TEST_1, test);

	FRAMEBits frame = {};
	frame.PSM = PSMCT32;
	frame.FBW = 640 / 64;
	iface.write_register(RegisterAddr::FRAME_1, frame);

	ZBUFBits zbuf = {};
	zbuf.ZMSK = 1;
	iface.write_register(RegisterAddr::ZBUF_1, zbuf);

	SCISSORBits scissor = {};
	scissor.SCAX0 = 0;
	scissor.SCAY0 = 0;
	scissor.SCAX1 = 640 - 1;
	scissor.SCAY1 = 448 - 1;
	iface.write_register(RegisterAddr::SCISSOR_1, scissor);
}

static constexpr uint32_t PALETTE_ADDR = 3 * 1024 * 1024;
static constexpr uint32_t TEXTURE_ADDR = 2 * 1024 * 1024;

static void upload_palettes(GSDumpGenerator &iface)
{
	uint32_t lut[256];
	for (int i = 0; i < 256; i++)
		lut[i] = 0x010101 * i + (0x80u << 24);

	iface.write_image_upload(PALETTE_ADDR,
	                         PSMCT32, 16, 16, lut, sizeof(lut));
}

static void run_test(GSDumpGenerator &iface)
{
	setup_frame_buffer(iface);
	upload_palettes(iface);

	uint8_t texture[256];
	for (int i = 0; i < 256; i++)
		texture[i] = uint8_t(i);

	iface.write_image_upload(TEXTURE_ADDR, PSMT8, 16, 16, texture, sizeof(texture));
	iface.write_register(RegisterAddr::TEXFLUSH, uint64_t(0));

	TEX0Bits tex0 = {};
	tex0.TBW = 16 / 64;
	tex0.TW = 4;
	tex0.TH = 4;
	tex0.TCC = 1;
	tex0.TFX = COMBINER_DECAL;
	tex0.TBP0 = TEXTURE_ADDR / PGS_BLOCK_ALIGNMENT_BYTES;
	tex0.PSM = PSMT8;
	tex0.CPSM = PSMCT32;
	tex0.CSM = 0;
	tex0.CLD = 1;
	tex0.CBP = PALETTE_ADDR / PGS_BLOCK_ALIGNMENT_BYTES;
	tex0.CSA = 0;
	iface.write_register(RegisterAddr::TEX0_1, tex0);

	tex0.PSM = PSMT8;
	tex0.CSA = 0;
	tex0.CLD = 0;
	iface.write_register(RegisterAddr::TEX0_1, tex0);

	TEXABits texa = {};
	texa.TA0 = 0x80;
	texa.TA1 = 0x80;
	iface.write_register(RegisterAddr::TEXA, texa);

	ALPHABits alpha = {};
	alpha.A = BLEND_RGB_SOURCE;
	alpha.B = BLEND_RGB_ZERO;
	alpha.C = BLEND_ALPHA_SOURCE;
	alpha.D = BLEND_RGB_ZERO;
	iface.write_register(RegisterAddr::ALPHA_1, alpha);

	write_clear_quad(iface, 0, 0, 8, 8);
	write_sprite_primitive(iface, 0, 0, 16, 16);
	write_line_primitive(iface, 5.0f, 3.0f - 1.0f / 16.0f, 2.0f, 3.0f - 1.0f / 16.0f);
}

int main()
{
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
		priv.display1.DW = 16 * 4 - 1;
		priv.display1.DH = 16 - 1;

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
