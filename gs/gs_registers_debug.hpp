// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#pragma once

#include <iostream>

namespace ParallelGS
{
static inline const char *psm_to_str(uint32_t psm)
{
	switch (psm)
	{
	case PSMCT32: return "PSMCT32";
	case PSMCT24: return "PSMCT24";
	case PSMCT16: return "PSMCT16";
	case PSMCT16S: return "PSMCT16S";
	case PSMT8: return "PSMT8";
	case PSMT4: return "PSMT4";
	case PSMT8H: return "PSMT8H";
	case PSMT4HL: return "PSMT4HL";
	case PSMT4HH: return "PSMT4HH";
	case PSMZ32: return "PSMTZ32";
	case PSMZ24: return "PSMTZ24";
	case PSMZ16: return "PSMTZ16";
	case PSMZ16S: return "PSMTZ16S";
	default: return "?";
	}
}
}

#if defined(PARALLEL_GS_DEBUG) && PARALLEL_GS_DEBUG
namespace ParallelGS
{
template <typename Stream>
static inline Stream &operator<<(Stream &stream, const Reg64<RGBAQBits> &rgba)
{
	stream << "R: " << rgba.desc.R
	       << ", G: " << rgba.desc.G
	       << ", B: " << rgba.desc.B
	       << ", A: " << rgba.desc.A
	       << ", Q: " << rgba.desc.Q;

	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const Reg64<STBits> &st)
{
	stream << "S: " << st.desc.S
	       << ", T: " << st.desc.T;

	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const Reg64<CLAMPBits> &clamp)
{
	static const char *clamp_str[] = {
		"REPEAT",
		"CLAMP",
		"REGION_CLAMP",
		"REGION_REPEAT",
	};

	stream << "WMS: " << clamp_str[clamp.desc.WMS]
	       << ", WMT: " << clamp_str[clamp.desc.WMT]
	       << ", MINU: " << clamp.desc.MINU
	       << ", MAXU: " << clamp.desc.MAXU
	       << ", MINV: " << clamp.desc.MINV
	       << ", MAXV: " << clamp.desc.MAXV;
	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const Reg64<UVBits> &uv)
{
	stream << "U: " << float(uv.desc.U) / 16.0f
	       << ", V: " << float(uv.desc.V) / 16.0f;

	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const Reg64<XYZBits> &xyz)
{
	stream << "X: " << float(xyz.desc.X) / 16.0f
	       << ", Y: " << float(xyz.desc.Y) / 16.0f
	       << ", Z: " << xyz.desc.Z;

	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const Reg64<XYOFFSETBits> &xy)
{
	stream << "X: " << float(xy.desc.OFX) / 16.0f
	       << ", Y: " << float(xy.desc.OFY) / 16.0f;

	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const Reg64<XYZFBits> &xyzf)
{
	stream << "X: " << float(xyzf.desc.X) / 16.0f
	       << ", Y: " << float(xyzf.desc.Y) / 16.0f
	       << ", Z: " << xyzf.desc.Z
	       << ", F: " << xyzf.desc.F;

	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const Reg64<TEX0Bits> &tex0)
{
	static const char *cld_str[] = {
		"NONE",
		"LOAD",
		"LOAD (update CBP0)",
		"LOAD (update CBP1)",
		"COND CBP0",
		"COND CBP1",
		"invalid 6",
		"invalid 7",
	};

	static const char *tfx_str[] = { "MODULATE", "DECAL", "HIGHLIGHT", "HIGHLIGHT2" };

	stream << "ADDR: " << std::hex << tex0.desc.TBP0 * PGS_BLOCK_ALIGNMENT_BYTES << std::dec
	       << ", RowLength: " << tex0.desc.TBW * PGS_BUFFER_WIDTH_SCALE
	       << ", PSM: " << psm_to_str(tex0.desc.PSM)
	       << ", W: " << (1u << tex0.desc.TW)
	       << ", H: " << (1u << tex0.desc.TH)
	       << ", TFX: " << tfx_str[tex0.desc.TFX]
	       << ", CLD: " << cld_str[tex0.desc.CLD]
	       << ", CPSM: " << psm_to_str(tex0.desc.CPSM)
	       << ", CSM: " << tex0.desc.CSM
	       << ", CBP: " << std::hex << tex0.desc.CBP * PGS_BLOCK_ALIGNMENT_BYTES << std::dec
	       << ", CSA: " << tex0.desc.CSA;

	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const Reg64<TEX1Bits> &tex1)
{
	static const char *mmin_str[] = {
		"NEAREST",
		"LINEAR",
		"NEAREST_MIPMAP_NEAREST",
		"NEAREST_MIPMAP_LINEAR",
		"LINEAR_MIPMAP_NEAREST",
		"LINEAR_MIPMAP_LINEAR",
		"Invalid 6",
		"Invalid 7",
	};

	static const char *mmag_str[] = {
		"NEAREST",
		"LINEAR",
	};

	stream << "L: " << tex1.desc.L
	       << ", K: " << tex1.desc.K
	       << ", MMIN: " << mmin_str[tex1.desc.MMIN]
	       << ", MMAG: " << mmag_str[tex1.desc.MMAG]
	       << ", LCM: " << tex1.desc.LCM
	       << ", MXL: " << tex1.desc.MXL
	       << ", MTBA: " << tex1.desc.MTBA;

	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const Reg64<TEXCLUTBits> &texclut)
{
	stream << "CBW: " << texclut.desc.CBW * PGS_BLOCK_ALIGNMENT_BYTES
	       << ", COU: " << texclut.desc.COU
	       << ", COV: " << texclut.desc.COV;

	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const Reg64<FRAMEBits> &frame)
{
	stream << "ADDR: " << std::hex << frame.desc.FBP * PGS_PAGE_ALIGNMENT_BYTES << std::dec
	       << ", RowLength: " << frame.desc.FBW * PGS_BUFFER_WIDTH_SCALE
	       << ", PSM: " << psm_to_str(frame.desc.PSM)
		   << ", FBMASK: " << std::hex << frame.desc.FBMSK << std::dec;

	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const Reg64<ZBUFBits> &zbuf)
{
	stream << "ADDR: " << std::hex << zbuf.desc.ZBP * PGS_PAGE_ALIGNMENT_BYTES << std::dec
	       << ", PSM: " << psm_to_str(zbuf.desc.PSM)
	       << ", ZMSK: " << zbuf.desc.ZMSK;

	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, i16vec4 bb)
{
	stream << "LO: [" << bb.x << ", " << bb.y << "], HI: ["
	       << bb.z << ", " << bb.w << "]";

	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, ivec4 bb)
{
	stream << "LO: [" << bb.x << ", " << bb.y << "], HI: ["
	       << bb.z << ", " << bb.w << "]";

	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const PrimitiveAttribute &prim)
{
	uint32_t state_index = (prim.state >> STATE_INDEX_BIT_OFFSET) & ((1u << STATE_INDEX_BIT_COUNT) - 1u);
	uint32_t tex_index = (prim.tex >> TEX_TEXTURE_INDEX_OFFSET) & ((1u << TEX_TEXTURE_INDEX_BITS) - 1u);

	bool multisample = (prim.state & (1u << STATE_BIT_MULTISAMPLE)) != 0;
	bool perspective = (prim.state & (1u << STATE_BIT_PERSPECTIVE)) != 0;
	bool iip = (prim.state & (1u << STATE_BIT_IIP)) != 0;
	bool fix = (prim.state & (1u << STATE_BIT_FIX)) != 0;
	bool ztst = (prim.state & (1u << STATE_BIT_Z_TEST)) != 0;
	bool zwrite = (prim.state & (1u << STATE_BIT_Z_WRITE)) != 0;
	bool zgreater = (prim.state & (1u << STATE_BIT_Z_TEST_GREATER)) != 0;
	bool opaque = (prim.state & (1u << STATE_BIT_OPAQUE)) != 0;

	stream << "StateIndex: " << state_index
	       << ", TexIndex: " << tex_index
	       << ", BB: " << prim.bb
	       << ", AFIX: " << ((prim.alpha >> ALPHA_AFIX_OFFSET) & ((1u << ALPHA_AFIX_BITS) - 1u))
	       << ", AREF: " << ((prim.alpha >> ALPHA_AREF_OFFSET) & ((1u << ALPHA_AREF_BITS) - 1u));

	stream << "\n      AA: " << multisample;
	stream << ", STQ: " << perspective;
	stream << ", IIP: " << iip;
	stream << ", FIX: " << fix;
	stream << ", ZTST: " << ztst;
	stream << ", ZWRITE: " << zwrite;
	stream << ", ZGT: " << zgreater;
	stream << ", OPAQUE: " << opaque;

	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const StateVector &state)
{
	static const char *atst_str[] = {
		"NEVER",
		"ALWAYS",
		"LE",
		"LEQ",
		"EQ",
		"GEQ",
		"GT",
		"NEQ",
	};

	static const char *afail_str[] = {
		"KEEP",
		"FB_ONLY",
		"ZB_ONLY",
		"RGB_ONLY",
	};

	bool ate = (state.blend_mode & BLEND_MODE_ATE_BIT) != 0;

	stream << "\n       TME: " << bool(state.combiner & COMBINER_TME_BIT);
	stream << "\n       FOG: " << bool(state.combiner & COMBINER_FOG_BIT);
	stream << "\n       TCC: " << bool(state.combiner & COMBINER_TCC_BIT);
	stream << "\n      DATE: " << bool(state.blend_mode & BLEND_MODE_DATE_BIT);
	stream << "\n      DATM: " << bool(state.blend_mode & BLEND_MODE_DATM_BIT);
	stream << "\n      DTHE: " << bool(state.blend_mode & BLEND_MODE_DTHE_BIT);
	stream << "\n      ATST: " << (ate ? atst_str[((state.blend_mode >> BLEND_MODE_ATE_MODE_OFFSET) &
	                                               ((1u << BLEND_MODE_ATE_MODE_BITS) - 1u))] : "disabled");
	stream << "\n     AFAIL: " << afail_str[(state.blend_mode >> BLEND_MODE_AFAIL_MODE_OFFSET) &
	                                        ((1u << BLEND_MODE_AFAIL_MODE_BITS) - 1u)];

	stream << "\n       ABE: ";
	auto A = (state.blend_mode >> BLEND_MODE_A_MODE_OFFSET) &
			 ((1u << BLEND_MODE_A_MODE_BITS) - 1u);
	auto B = (state.blend_mode >> BLEND_MODE_B_MODE_OFFSET) &
			 ((1u << BLEND_MODE_B_MODE_BITS) - 1u);
	auto C = (state.blend_mode >> BLEND_MODE_C_MODE_OFFSET) &
			 ((1u << BLEND_MODE_C_MODE_BITS) - 1u);
	auto D = (state.blend_mode >> BLEND_MODE_D_MODE_OFFSET) &
			 ((1u << BLEND_MODE_D_MODE_BITS) - 1u);

	static const char *abd_to_str[] = { "Cs", "Cd", "0", "N/A" };
	static const char *c_to_str[] = { "As", "Ad", "FIX", "N/A" };
	stream << "(" << abd_to_str[A] << " - " << abd_to_str[B] << ") * " << c_to_str[C] << " + " << abd_to_str[D];

	if ((state.combiner & COMBINER_TME_BIT) != 0)
	{
		stream << "\n   COMBINER: " << ((state.combiner >> COMBINER_MODE_OFFSET) &
		                                ((1u << COMBINER_MODE_BITS) - 1u));
	}

	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const RenderPass &rp)
{
	for (uint32_t i = 0; i < rp.num_instances; i++)
	{
		auto &inst = rp.instances[i];
		stream << "LABEL: " << rp.label_key
		       << ", instance: " << i
		       << ", X: " << inst.base_x
		       << ", Y: " << inst.base_y
		       << ", NumPrims: " << rp.num_primitives
		       << ", W: " << (inst.coarse_tiles_width << rp.coarse_tile_size_log2)
		       << ", H: " << (inst.coarse_tiles_height << rp.coarse_tile_size_log2)
		       << ", NumTex: " << rp.num_textures
		       << ", NumState: " << rp.num_states
		       << ", Z: " << (inst.z_sensitive ? "ON" : "OFF");

		stream << " || FRAME - " << inst.fb.frame;
		if (inst.z_sensitive)
			stream << " || ZBUF - " << inst.fb.z;
	}

	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const Reg64<TRXPOSBits> &trxpos)
{
	stream << "SSAX: " << trxpos.desc.SSAX
	       << ", SSAY: " << trxpos.desc.SSAY
	       << ", DSAX: " << trxpos.desc.DSAX
	       << ", DSAY: " << trxpos.desc.DSAY
	       << ", DIR: " << trxpos.desc.DIR;

	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const Reg64<BITBLTBUFBits> &bitbltbuf)
{
	stream << "DST: " << std::hex << bitbltbuf.desc.DBP * PGS_BLOCK_ALIGNMENT_BYTES << std::dec
	       << ", DstRowLength: " << bitbltbuf.desc.DBW * PGS_BUFFER_WIDTH_SCALE
	       << ", DPSM: " << psm_to_str(bitbltbuf.desc.DPSM)
	       << ", SRC: " << std::hex << bitbltbuf.desc.SBP * PGS_BLOCK_ALIGNMENT_BYTES << std::dec
	       << ", SrcRowLength: " << bitbltbuf.desc.SBW * PGS_BUFFER_WIDTH_SCALE
	       << ", SPSM: " << psm_to_str(bitbltbuf.desc.SPSM);

	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const Reg64<TRXREGBits> &trxreg)
{
	stream << "W: " << trxreg.desc.RRW << ", H: " << trxreg.desc.RRH;
	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const CopyDescriptor &copy)
{
	stream << "TRXPOS: " << copy.trxpos
	       << ", BITBLTBUF: " << copy.bitbltbuf
	       << ", TRXREG: " << copy.trxreg
	       << ", TRXDIR: " << copy.trxdir
	       << ", size: " << copy.host_data_size;

	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const PaletteUploadDescriptor &pal)
{
	stream << "TEXCLUT: " << pal.texclut << ", TEX0: " << pal.tex0;
	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const TextureDescriptor &tex)
{
	stream << "\n    TEX0: " << tex.tex0;
	stream << "\n    TEX1: " << tex.tex1;
	stream << "\n    TEXA: " << tex.texa;
	stream << "\n    MIPTBP1_3: " << tex.miptbp1_3;
	stream << "\n    MIPTBP4_6: " << tex.miptbp4_6;
	stream << "\n    PaletteBank: " << tex.palette_bank;
	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const Reg64<PRIMBits> &prim)
{
	static const char *prim_str[] = {
		"Point",
		"Line",
		"LineStrip",
		"TriList",
		"TriStrip",
		"TriFan",
		"Sprite",
		"Invalid",
	};

	stream << "ABE: " << prim.desc.ABE
	       << ", CTXT: " << prim.desc.CTXT
	       << ", IIP: " << prim.desc.IIP
	       << ", AA1: " << prim.desc.AA1
	       << ", FGE: " << prim.desc.FGE
	       << ", FST: " << prim.desc.FST
	       << ", PRIM: " << prim_str[prim.desc.PRIM]
	       << ", FIX: " << prim.desc.FIX
	       << ", TME: " << prim.desc.TME;

	return stream;
}

template <typename Stream, typename T>
static inline Stream &operator<<(Stream &stream, const Reg64<T> &desc)
{
	stream << "raw hex: " << std::hex << desc.bits << std::dec;
	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const Reg64<DummyBits> &)
{
	return stream;
}

template <typename Stream>
static inline Stream &operator<<(Stream &stream, const GIFTagBits &tag)
{
	auto nregs = tag.NREG ? tag.NREG : 16u;
	static const char *flg_to_str[] = { "PACKED", "REGLIST", "IMAGE", "IMAGE" };

	stream << "FLG: " << flg_to_str[tag.FLG]
	       << ", NLOOP: " << tag.NLOOP
	       << ", NREGS: " << nregs
	       << ", PRE: " << tag.PRE
	       << ", EOP: " << tag.EOP;

	if (tag.FLG < 2)
	{
		static const char *regs_str[] = {
			"PRIM",
			"RGBAQ",
			"ST",
			"UV",
			"XYZF2",
			"XYZ2",
			"TEX0_1",
			"TEX0_2",
			"CLAMP_1",
			"CLAMP_2",
			"FOG",
			"RESERVED",
			"XYZF3",
			"XYZ3",
			"A_D",
			"NOP",
		};

		stream << ", REGS: [";
		for (unsigned i = 0; i < nregs; i++)
		{
			stream << regs_str[(tag.REGS >> (4 * i)) & 0xf];
			if (i + 1 < nregs)
				stream << ", ";
		}
		stream << "]";
	}

	return stream;
}

#define TRACE_HEADER(tag, x) std::cout << (tag) << ": " << (x) << std::endl
#define TRACE(tag, x) std::cout << "  " << (tag) << " || " << (x) << std::endl
#define TRACE_INDEXED(tag, index, x) std::cout << "  " << (tag) << " #" << (index) << " || " << (x) << std::endl
}
#else
#define TRACE_HEADER(tag, x) ((void)(tag)), ((void)(x))
#define TRACE(tag, x) ((void)(tag)), ((void)(x))
#define TRACE_INDEXED(tag, index, x) ((void)(tag)), ((void)(index)), ((void)(x))
#endif