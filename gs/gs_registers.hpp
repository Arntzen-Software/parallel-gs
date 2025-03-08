// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#pragma once

#include <stdint.h>

namespace ParallelGS
{
template <typename T>
union Reg64
{
	Reg64() : desc{} {}
	inline Reg64(uint64_t bits_) : bits(bits_) {}
	inline Reg64(const T &desc_) : desc(desc_) {}

	T desc;
	uint64_t bits;
	uint32_t words[2];
	static_assert(sizeof(T) == sizeof(uint64_t), "Reg64<T> is not 64-bits.");
};

template <typename T>
union Reg128
{
	Reg128() : desc{} {};
	inline Reg128(const T &desc_) : desc(desc_) {}

	T desc;
	uint64_t qwords[2];
	uint32_t words[4];
	static_assert(sizeof(T) == sizeof(uint64_t) * 2, "Reg128<T> is not 128-bits.");
};

#define FIELD(name, bits) uint64_t name : bits
#define PAD(b) uint64_t : b
#define FIELD32(name, bits) uint32_t name : bits
#define PAD32(b) uint32_t : b

struct TEX0Bits
{
	FIELD(TBP0, 14);
	FIELD(TBW, 6);
	FIELD(PSM, 6);
	FIELD(TW, 4);
	FIELD(TH, 4);
	FIELD(TCC, 1);
	FIELD(TFX, 2);
	FIELD(CBP, 14);
	FIELD(CPSM, 4);
	FIELD(CSM, 1);
	FIELD(CSA, 5);
	FIELD(CLD, 3);
	enum { MAX_SIZE_LOG2 = 10, MAX_LEVELS = 7 };
	enum
	{
		CLD_IGNORE = 0,
		CLD_LOAD = 1,
		CLD_LOAD_WRITE_CBP0 = 2,
		CLD_LOAD_WRITE_CBP1 = 3,
		CLD_COMPARE_LOAD_CBP0 = 4,
		CLD_COMPARE_LOAD_CBP1 = 5
	};
	enum { CSM_LAYOUT_RECT = 0, CSM_LAYOUT_LINE = 1 };
	enum { COU_SCALE = 16 };
};

struct TEX1Bits
{
	FIELD32(LCM, 1);
	PAD32(1);
	FIELD32(MXL, 3);
	FIELD32(MMAG, 1);
	FIELD32(MMIN, 3);
	FIELD32(MTBA, 1);
	PAD32(9);
	FIELD32(L, 2);
	PAD32(11);
	FIELD32(K, 12);
	PAD32(20);

	enum
	{
		NEAREST = 0,
		LINEAR = 1,
		NEAREST_MIPMAP_NEAREST = 2,
		NEAREST_MIPMAP_LINEAR = 3,
		LINEAR_MIPMAP_NEAREST = 4,
		LINEAR_MIPMAP_LINEAR = 5
	};

	inline bool mmin_has_mipmap() const { return MMIN > LINEAR && MMIN <= LINEAR_MIPMAP_LINEAR; }
	inline bool has_mipmap() const { return mmin_has_mipmap() && MXL != 0; }
};

struct MIPTBPBits
{
	FIELD(TBP1, 14);
	FIELD(TBW1, 6);
	FIELD(TBP2, 14);
	FIELD(TBW2, 6);
	FIELD(TBP3, 14);
	FIELD(TBW3, 6);
	PAD(4);
};

struct TEXABits
{
	FIELD32(TA0, 8);
	PAD32(7);
	FIELD32(AEM, 1);
	PAD32(16);
	FIELD32(TA1, 8);
	PAD32(24);
};

struct TEXCLUTBits
{
	FIELD32(CBW, 6);
	FIELD32(COU, 6);
	FIELD32(COV, 10);
	PAD32(10);
	PAD32(32);
};

struct FRAMEBits
{
	FIELD32(FBP, 9);
	PAD32(7);
	FIELD32(FBW, 6);
	PAD32(2);
	FIELD32(PSM, 6);
	PAD32(2);
	FIELD32(FBMSK, 32);

	inline bool compat(const FRAMEBits &other) const { return FBP == other.FBP && FBW == other.FBW && PSM == other.PSM; }
};

struct ZBUFBits
{
	FIELD32(ZBP, 9);
	PAD32(15);
	FIELD32(PSM, 4);
	PAD32(4);
	FIELD32(ZMSK, 1);
	PAD32(31);
	enum { PSM_MSB = 3 << 4 };

	inline bool compat(const ZBUFBits &other) const { return ZBP == other.ZBP && PSM == other.PSM; }
};

struct BITBLTBUFBits
{
	FIELD32(SBP, 14);
	PAD32(2);
	FIELD32(SBW, 6);
	PAD32(2);
	FIELD32(SPSM, 6);
	PAD32(2);
	FIELD32(DBP, 14);
	PAD32(2);
	FIELD32(DBW, 6);
	PAD32(2);
	FIELD32(DPSM, 6);
	PAD32(2);
};

struct TRXDIRBits
{
	FIELD32(XDIR, 2);
	PAD32(30);
	PAD32(32);
};

struct TRXPOSBits
{
	FIELD32(SSAX, 11);
	PAD32(5);
	FIELD32(SSAY, 11);
	PAD32(5);
	FIELD32(DSAX, 11);
	PAD32(5);
	FIELD32(DSAY, 11);
	FIELD32(DIR, 2);
	PAD32(3);
};

struct TRXREGBits
{
	FIELD32(RRW, 12);
	PAD32(20);
	FIELD32(RRH, 12);
	PAD32(20);
};

struct RGBAQBits
{
	FIELD32(R, 8);
	FIELD32(G, 8);
	FIELD32(B, 8);
	FIELD32(A, 8);
	float Q;
};

struct STBits
{
	float S;
	float T;
};

struct UVBits
{
	FIELD32(U, 14);
	PAD32(2);
	FIELD32(V, 14);
	PAD32(2);
	PAD32(32);
};

struct XYZFBits
{
	FIELD32(X, 16);
	FIELD32(Y, 16);
	FIELD32(Z, 24);
	FIELD32(F, 8);
};

struct XYZBits
{
	FIELD32(X, 16);
	FIELD32(Y, 16);
	FIELD32(Z, 32);
};

struct FOGBits
{
	PAD32(32);
	PAD32(24);
	FIELD32(FOG, 8);
};

struct PRMODECONTBits
{
	FIELD32(AC, 1);
	PAD32(31);
	PAD32(32);
	enum { AC_DEFAULT = 1 };
};

struct PRIMBits
{
	FIELD32(PRIM, 3);
	FIELD32(IIP, 1);
	FIELD32(TME, 1);
	FIELD32(FGE, 1);
	FIELD32(ABE, 1);
	FIELD32(AA1, 1);
	FIELD32(FST, 1);
	FIELD32(CTXT, 1);
	FIELD32(FIX, 1);
	PAD32(21);
	PAD32(32);
};

struct ALPHABits
{
	FIELD32(A, 2);
	FIELD32(B, 2);
	FIELD32(C, 2);
	FIELD32(D, 2);
	PAD32(24);
	FIELD32(FIX, 8);
	PAD32(24);
};

struct CLAMPBits
{
	FIELD(WMS, 2);
	FIELD(WMT, 2);
	FIELD(MINU, 10);
	FIELD(MAXU, 10);
	FIELD(MINV, 10);
	FIELD(MAXV, 10);
	PAD(20);
	enum { REPEAT = 0, CLAMP = 1, REGION_CLAMP = 2, REGION_REPEAT = 3 };

	inline bool has_horizontal_repeat() const { return WMS == REPEAT || WMS == REGION_REPEAT; }
	inline bool has_vertical_repeat() const { return WMT == REPEAT || WMT == REGION_REPEAT; }
	inline bool has_horizontal_region() const { return WMS >= REGION_CLAMP; }
	inline bool has_vertical_region() const { return WMT >= REGION_CLAMP; }
	inline bool has_region_repeat() const { return WMS == REGION_REPEAT || WMT == REGION_REPEAT; }
	inline bool has_horizontal_clamp() const { return WMS == CLAMP || WMS == REGION_CLAMP; }
	inline bool has_vertical_clamp() const { return WMT == CLAMP || WMT == REGION_CLAMP; }
};

struct COLCLAMPBits
{
	FIELD32(CLAMP, 1);
	PAD32(31);
	PAD32(32);
};

struct DIMXBits
{
	FIELD32(DM00, 3);
	PAD32(1);
	FIELD32(DM01, 3);
	PAD32(1);
	FIELD32(DM02, 3);
	PAD32(1);
	FIELD32(DM03, 3);
	PAD32(1);

	FIELD32(DM10, 3);
	PAD32(1);
	FIELD32(DM11, 3);
	PAD32(1);
	FIELD32(DM12, 3);
	PAD32(1);
	FIELD32(DM13, 3);
	PAD32(1);

	FIELD32(DM20, 3);
	PAD32(1);
	FIELD32(DM21, 3);
	PAD32(1);
	FIELD32(DM22, 3);
	PAD32(1);
	FIELD32(DM23, 3);
	PAD32(1);

	FIELD32(DM30, 3);
	PAD32(1);
	FIELD32(DM31, 3);
	PAD32(1);
	FIELD32(DM32, 3);
	PAD32(1);
	FIELD32(DM33, 3);
	PAD32(1);
};

struct DTHEBits
{
	FIELD32(DTHE, 1);
	PAD32(31);
	PAD32(32);
};

struct FBABits
{
	FIELD32(FBA, 1);
	PAD32(31);
	PAD32(32);
};

struct FOGCOLBits
{
	FIELD32(FCR, 8);
	FIELD32(FCG, 8);
	FIELD32(FCB, 8);
	PAD32(8);
	PAD32(32);
};

struct LABELBits
{
	FIELD32(ID, 32);
	FIELD32(IDMSK, 32);
};

struct PABEBits
{
	FIELD32(PABE, 1);
	PAD32(31);
	PAD32(32);
};

struct SCANMSKBits
{
	FIELD32(MSK, 2);
	PAD32(30);
	PAD32(32);
	enum { MSK_NONE = 0, MSK_SKIP_EVEN = 2, MSK_SKIP_ODD = 3 };
	inline bool has_mask() const { return MSK >= MSK_SKIP_EVEN; }
};

struct SCISSORBits
{
	FIELD32(SCAX0, 11);
	PAD32(5);
	FIELD32(SCAX1, 11);
	PAD32(5);
	FIELD32(SCAY0, 11);
	PAD32(5);
	FIELD32(SCAY1, 11);
	PAD32(5);
};

struct TESTBits
{
	FIELD32(ATE, 1);
	FIELD32(ATST, 3);
	FIELD32(AREF, 8);
	FIELD32(AFAIL, 2);
	FIELD32(DATE, 1);
	FIELD32(DATM, 1);
	FIELD32(ZTE, 1);
	FIELD32(ZTST, 2);
	PAD32(13);
	PAD32(32);
	enum { ZTE_ENABLED = 1, ZTE_UNDEFINED = 0 };
	enum { ZTST_NEVER = 0, ZTST_ALWAYS = 1, ZTST_GEQUAL = 2, ZTST_GREATER = 3 };
	inline bool has_z_test() const { return ZTST > ZTST_ALWAYS; };
};

struct XYOFFSETBits
{
	FIELD32(OFX, 16);
	PAD32(16);
	FIELD32(OFY, 16);
	PAD32(16);
};

// Aliased registers with subset of effective state.
using TEX2Bits = TEX0Bits;
using SIGNALBits = LABELBits;

struct PackedRGBAQBits
{
	FIELD32(R, 8);
	PAD32(24);
	FIELD32(G, 8);
	PAD32(24);
	FIELD32(B, 8);
	PAD32(24);
	FIELD32(A, 8);
	PAD32(24);
};

struct PackedSTBits
{
	float S;
	float T;
	float Q;
	PAD32(32);
};

struct PackedUVBits
{
	FIELD32(U, 14);
	PAD32(18);
	FIELD32(V, 14);
	PAD32(18);
	PAD(64);
};

struct PackedXYZFBits
{
	FIELD32(X, 16);
	PAD32(16);
	FIELD32(Y, 16);
	PAD32(16);
	PAD32(4);
	FIELD32(Z, 24);
	PAD32(4);
	PAD32(4);
	FIELD32(F, 8);
	PAD32(3);
	FIELD32(ADC, 1);
	PAD32(16);
};

struct PackedXYZBits
{
	FIELD32(X, 16);
	PAD32(16);
	FIELD32(Y, 16);
	PAD32(16);
	FIELD32(Z, 32);
	PAD32(15);
	FIELD32(ADC, 1);
	PAD32(16);
};

struct PackedFOGBits
{
	PAD32(32);
	PAD32(32);
	PAD32(32);
	PAD32(4);
	FIELD32(F, 8);
	PAD32(20);
};

struct PackedADBits
{
	uint64_t data;
	FIELD32(ADDR, 7);
	PAD32(25);
};

struct GIFTagBits
{
	FIELD32(NLOOP, 15);
	FIELD32(EOP, 1);
	PAD32(16);
	PAD32(14);
	FIELD32(PRE, 1);
	FIELD32(PRIM, 11);
	FIELD32(FLG, 2);
	FIELD32(NREG, 4);
	uint64_t REGS;

	enum { PACKED = 0, REGLIST = 1, IMAGE = 2, IMAGE_RESERVED = 3 };
};

enum class GIFAddr : uint32_t
{
	PRIM = 0x0,
	RGBAQ = 0x1,
	ST = 0x02,
	UV = 0x03,
	XYZF2 = 0x04,
	XYZ2 = 0x05,
	TEX0_1 = 0x06,
	TEX0_2 = 0x07,
	CLAMP_1 = 0x08,
	CLAMP_2 = 0x09,
	FOG = 0xa,
	RESERVED = 0xb,
	XYZF3 = 0xc,
	XYZ3 = 0xd,
	A_D = 0xe,
	NOP = 0xf
};

enum class PRIMType : uint32_t
{
	Point = 0,
	LineList = 1,
	LineStrip = 2,
	TriangleList = 3,
	TriangleStrip = 4,
	TriangleFan = 5,
	Sprite = 6,
	Invalid = 7
};

// For A+D addressing.
enum class RegisterAddr : uint32_t
{
#define DECL_REG(reg, addr) reg = addr,
#include "gs_register_addr.hpp"
#undef DECL_REG
};

struct PMODEBits
{
	FIELD32(EN1, 1);
	FIELD32(EN2, 1);
	FIELD32(CRTMD, 3);
	FIELD32(MMOD, 1);
	FIELD32(AMOD, 1);
	FIELD32(SLBG, 1);
	FIELD32(ALP, 8);
	PAD32(16);
	PAD32(32);
	enum { SLBG_ALPHA_BLEND_CIRCUIT2 = 0, SLBG_ALPHA_BLEND_BG = 1 };
	enum { MMOD_ALPHA_CIRCUIT1 = 0, MMOD_ALPHA_ALP = 1 };
};

struct BGCOLORBits
{
	FIELD32(R, 8);
	FIELD32(G, 8);
	FIELD32(B, 8);
	PAD32(8);
	PAD32(32);
};

struct BUSDIRBits
{
	FIELD32(DIR, 1);
	PAD32(31);
	PAD32(32);
};

struct CSRBits
{
	FIELD32(SIGNAL, 1);
	FIELD32(FINISH, 1);
	FIELD32(HSINT, 1);
	FIELD32(VSINT, 1);
	FIELD32(EDWINT, 1);
	PAD32(3);
	FIELD32(FLUSH, 1);
	FIELD32(RESET, 1);
	PAD32(2);
	FIELD32(NFIELD, 1);
	FIELD32(FIELD, 1);
	FIELD32(FIFO, 2);
	FIELD32(REV, 8);
	FIELD32(ID, 8);
	PAD32(32);
};

struct DISPFBBits
{
	FIELD32(FBP, 9);
	FIELD32(FBW, 6);
	FIELD32(PSM, 5);
	PAD32(12);
	FIELD32(DBX, 11);
	FIELD32(DBY, 11);
	PAD32(10);
};

struct DISPLAYBits
{
	FIELD32(DX, 12);
	FIELD32(DY, 11);
	FIELD32(MAGH, 4);
	FIELD32(MAGV, 2);
	PAD32(3);
	FIELD32(DW, 12);
	FIELD32(DH, 11);
	PAD32(9);
};

struct EXTBUFBits
{
	FIELD32(EXBP, 14);
	FIELD32(EXBW, 6);
	FIELD32(FBIN, 2);
	FIELD32(WFFMD, 1);
	FIELD32(EMODA, 2);
	FIELD32(EMODC, 2);
	PAD32(5);
	FIELD32(WDX, 11);
	FIELD32(WDY, 11);
	PAD32(10);
};

struct EXTDATABits
{
	FIELD32(SX, 12);
	FIELD32(SY, 11);
	FIELD32(SMPH, 4);
	FIELD32(SMPV, 2);
	PAD32(3);
	FIELD32(WW, 12);
	FIELD32(WH, 11);
	PAD32(9);
};

struct EXTWRITEBits
{
	FIELD32(WRITE, 1);
	PAD32(31);
	PAD32(32);
};

struct IMRBits
{
	PAD32(8);
	FIELD32(SIGMSK, 1);
	FIELD32(FINISHMSK, 1);
	FIELD32(HSMSK, 1);
	FIELD32(VSMSK, 1);
	FIELD32(EDWMSK, 1);
	PAD32(3);
	PAD32(16);
	PAD32(32);
};

struct SIGLBLIDBits
{
	FIELD32(SIGID, 32);
	FIELD32(LBLID, 32);
};

struct SMODE1Bits
{
	FIELD32(RC, 3);
	FIELD32(LC, 7);
	FIELD32(T1248, 2);
	FIELD32(SLCK, 1);
	FIELD32(CMOD, 2);
	FIELD32(EX, 1);
	FIELD32(PRST, 1);
	FIELD32(SINT, 1);
	FIELD32(XPCK, 1);
	FIELD32(PCK2, 2);
	FIELD32(SPML, 4);
	FIELD32(GCONT, 1);
	FIELD32(PHS, 1);
	FIELD32(PVS, 1);
	FIELD32(PEHS, 1);
	FIELD32(PEVS, 1);
	FIELD32(CLKSEL, 2);
	FIELD32(NVCK, 1);
	FIELD32(SLCK2, 1);
	FIELD32(VCKSEL, 2);
	FIELD32(VHP, 1);
	PAD32(27);
	enum { CMOD_PROGRESSIVE = 0, CMOD_NTSC = 2, CMOD_PAL = 3 };
	enum { LC_ANALOG = 32, LC_HDTV = 22 };
	enum {
		CLOCK_DIVIDER_COMPOSITE = 4,
		CLOCK_DIVIDER_COMPONENT = 2 /* Seems to be the case based on progressive scan games. */,
		CLOCK_DIVIDER_HDTV = 1
	};
};

struct SMODE2Bits
{
	FIELD32(INT, 1);
	FIELD32(FFMD, 1);
	FIELD32(DPMS, 2);
	PAD32(29);
	PAD32(32);
};

struct SYNCVBits
{
	FIELD32(VFP, 10);
	FIELD32(VFPE, 10);
	FIELD32(VBP, 12);
	FIELD32(VBPE, 10);
	FIELD32(VDP, 11);
	FIELD32(VS, 11);
};

struct DummyBits
{
	uint64_t dummy;
};

}