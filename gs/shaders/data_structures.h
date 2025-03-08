// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#ifndef DATA_STRUCTURES_H_
#define DATA_STRUCTURES_H_

// Header which is shared between CPU and GPU.

#ifdef __cplusplus
using namespace muglm;
#define CONSTEXPR static constexpr
namespace ParallelGS {
#else
#define CONSTEXPR const
#define VkDeviceAddress uvec2
#endif

CONSTEXPR int PGS_SUBPIXEL_BITS = 4;
CONSTEXPR int PGS_SUBPIXELS = 1 << PGS_SUBPIXEL_BITS;

// Allows us to interpolate and test rasterization with 1/8th pixel precision.
// Can allow for 8x upscale.
CONSTEXPR int PGS_RASTER_SUBSAMPLE_BITS = 3;
CONSTEXPR int PGS_SUBPIXEL_RASTER_BITS = PGS_SUBPIXEL_BITS - PGS_RASTER_SUBSAMPLE_BITS;
CONSTEXPR int PGS_SUBPIXELS_RASTER = 1 << PGS_SUBPIXEL_RASTER_BITS;

struct PrimitiveSetup
{
	ivec3 a; float inv_area;
	ivec3 b; float error_i;
	ivec3 c; float error_j;
	ivec4 bb; // Also serves as scissor.
	uvec4 z; // Pack it here since we need to read it in early-Z. Better for cache.
};

// Each primitive's attribute setup consumes 64 bytes.
// Quite compact and nice cache line alignment.
struct TransformedAttributes
{
	vec4 stqf0;
	vec4 stqf1;
	vec4 stqf2;
	uint rgba0;
	uint rgba1;
	uint rgba2;
	uint padding;
};

// Affects rasterization.
struct VertexPosition
{
	ivec2 pos; uint z; int padding;
};

// Varyings, not accessed until we need to actually shade.
struct VertexAttribute
{
	vec2 st;
	float q;
	uint rgba;
	float fog;
	u16vec2 uv;
};

// Primitive attributes which affect the entire primitive like state index / texture indices / etc.
struct PrimitiveAttribute
{
	i16vec4 bb; // Scissor
	uint state;
	uint tex;
	uint tex2;
	uint alpha;
	uint fbmsk;
	uint fogcol;
};

struct CLUTDescriptor
{
	uint tex_format;
	uint format;
	uint base_pointer;
	uint instance;
	uint csm;
	uint co_uv;
	uint cbw;
	uint csa;
	float csm2_x_scale;
	uint csm2_x_bias;
	uint csm1_reference_base;
	uint csm1_mask;
};

struct TexInfo
{
	vec4 sizes;
	vec4 region;
	vec2 bias;
	int arrayed;
	int flags;
};

CONSTEXPR int TEX_INFO_FORCE_SAMPLE_MAPPING = 0x1;
CONSTEXPR int TEX_INFO_LONG_TERM_REFERENCE = 0x2;

CONSTEXPR int PGS_FB_SWIZZLE_WIDTH_LOG2 = 3;
CONSTEXPR int PGS_FB_SWIZZLE_HEIGHT_LOG2 = 3;
CONSTEXPR int PGS_FB_SWIZZLE_WIDTH = 1 << PGS_FB_SWIZZLE_WIDTH_LOG2;
CONSTEXPR int PGS_FB_SWIZZLE_HEIGHT = 1 << PGS_FB_SWIZZLE_HEIGHT_LOG2;
CONSTEXPR int PGS_FB_SWIZZLE_BLOCK_SIZE = PGS_FB_SWIZZLE_WIDTH * PGS_FB_SWIZZLE_HEIGHT;

CONSTEXPR int PGS_BLOCK_ALIGNMENT_WORDS = 64;
CONSTEXPR int PGS_PAGE_ALIGNMENT_WORDS = 2048;
CONSTEXPR int PGS_BLOCK_ALIGNMENT_BYTES = PGS_BLOCK_ALIGNMENT_WORDS * 4;
CONSTEXPR int PGS_PAGE_ALIGNMENT_BYTES = PGS_PAGE_ALIGNMENT_WORDS * 4;
CONSTEXPR int PGS_BLOCKS_PER_PAGE = PGS_PAGE_ALIGNMENT_WORDS / PGS_BLOCK_ALIGNMENT_WORDS;
CONSTEXPR int PGS_CLUT_SIZE = 1024;
CONSTEXPR int PGS_CLUT_INSTANCES = 1024;

// FBW / TBW, etc. Basically intended to match page width,
// but edge case for tiny formats like PSMT4, etc.
CONSTEXPR int PGS_BUFFER_WIDTH_SCALE = 64;

struct GlobalConstants
{
	ivec2 base_pixel;
	int coarse_primitive_list_stride;
	int coarse_fb_width;
	int coarse_tile_size_log2;
	int fb_color_page;
	int fb_depth_page;
	int fb_page_stride;
};

struct StateVector
{
	uint combiner;
	uint blend_mode;
	uvec2 dimx;
};

struct LocalDataStructure
{
	uint page_width;
	uint page_height;
	uint block_width;
	uint block_height;
	uint column_height;

	uint page_width_log2;
	uint page_height_log2;
	uint block_width_log2;
	uint block_height_log2;
	uint column_height_log2;
};

struct TransferDescriptor
{
	uint source_addr;
	uint source_stride;
	uint dest_addr;
	uint dest_stride;
	uint source_x;
	uint source_y;
	uint dest_x;
	uint dest_y;
	uint width;
	uint height;
	uint host_offset_qwords;
	uint dispatch_order;
	VkDeviceAddress source_bda;
	uint source_size;
	uint padding;
};

#define PGS_LINKED_VRAM_COPY_WRITE_LIST_OFFSET 4096
#define PGS_VALID_PAGE_COPY_WRITE_OFFSET 256
CONSTEXPR int LINKED_VRAM_COPY_DISPATCH_ORDER_OFFSET = 22;
CONSTEXPR int LINKED_VRAM_COPY_DISPATCH_ORDER_BITS = 10;
struct LinkedVRAMCopyWrite
{
	uint write_mask;
	uint next;
};

struct SingleSampleHeuristic
{
	uvec4 single_sample_fixup_indirect;
	uint active_depth_range_mask_atomic[8];
	uint depth_range_offset[256];
};

struct ShadingDescriptor
{
	ivec2 snap_raster_mask;
	uint color_preserve_samples;
	uint lo_primitive_index;
	uint hi_primitive_index;
	uint fb_index_depth_offset;
};

// PrimitiveAttribute::state
// If set, performs a greater-than or greater-or-equals test. If false, depth test always passes.
CONSTEXPR int STATE_BIT_Z_TEST = 12;
// If set, writes depth if pixel tests pass. Can be used with Z_TEST = false.
CONSTEXPR int STATE_BIT_Z_WRITE = 13;
// If opaque, the pixel is considered opaque and early ZS update is considered safe.
// An opaque pixel cannot blend or perform alpha testing.
CONSTEXPR int STATE_BIT_OPAQUE = 14;
CONSTEXPR int STATE_BIT_PARALLELOGRAM = 15;
CONSTEXPR int STATE_BIT_MULTISAMPLE = 16;
CONSTEXPR int STATE_BIT_Z_TEST_GREATER = 17;
CONSTEXPR int STATE_BIT_PERSPECTIVE = 18;
CONSTEXPR int STATE_BIT_IIP = 19;
CONSTEXPR int STATE_BIT_FIX = 20;
CONSTEXPR int STATE_BIT_SPRITE = 21;
CONSTEXPR int STATE_BIT_LINE = 22;
CONSTEXPR int STATE_BIT_SCANMSK_EVEN = 23;
CONSTEXPR int STATE_BIT_SCANMSK_ODD = 24;
CONSTEXPR int STATE_BIT_SNAP_RASTER = 25;
CONSTEXPR int STATE_BIT_SNAP_ATTRIBUTE = 26;
CONSTEXPR int STATE_PARALLELOGRAM_PROVOKING_OFFSET = 27;
CONSTEXPR int STATE_PARALLELOGRAM_PROVOKING_COUNT = 2;
CONSTEXPR int STATE_VERTEX_RENDER_PASS_INSTANCE_OFFSET = 29;
CONSTEXPR int STATE_VERTEX_RENDER_PASS_INSTANCE_COUNT = 3;

CONSTEXPR int ZTST_GEQUALS = 0;
CONSTEXPR int ZTST_GREATER = 1;

CONSTEXPR int STATE_INDEX_BIT_OFFSET = 0;
CONSTEXPR int STATE_INDEX_BIT_COUNT = 12;
/////

// PrimitiveAttribute::tex
CONSTEXPR int TEX_TEXTURE_INDEX_OFFSET = 0;
CONSTEXPR int TEX_TEXTURE_INDEX_BITS = 16;
CONSTEXPR int SAMPLER_NEAREST = 0;
CONSTEXPR int SAMPLER_LINEAR = 1;
CONSTEXPR int SAMPLER_COUNT = 2;
CONSTEXPR int TEX_SAMPLER_MAG_LINEAR_BIT = 1 << 16;
CONSTEXPR int TEX_SAMPLER_MIN_LINEAR_BIT = 1 << 17;
CONSTEXPR int TEX_SAMPLER_MIPMAP_LINEAR_BIT = 1 << 18;
CONSTEXPR int TEX_SAMPLER_CLAMP_S_BIT = 1 << 19;
CONSTEXPR int TEX_SAMPLER_CLAMP_T_BIT = 1 << 20;
CONSTEXPR int TEX_MAX_MIP_LEVEL_OFFSET = 21;
CONSTEXPR int TEX_MAX_MIP_LEVEL_BITS = 3;
CONSTEXPR int TEX_PER_SAMPLE_BIT = 1 << 24;
CONSTEXPR int TEX_SAMPLE_MAPPING_BIT = 1 << 25;
CONSTEXPR int TEX_SAMPLE_RESOLVED_BIT = 1 << 26;
/////

// PrimitiveAttribute::tex2
CONSTEXPR int TEX2_FIXED_LOD_OFFSET = 0;
CONSTEXPR int TEX2_FIXED_LOD_BITS = 1;
CONSTEXPR int TEX2_L_OFFSET = 1;
CONSTEXPR int TEX2_L_BITS = 2;
CONSTEXPR int TEX2_K_OFFSET = 3;
CONSTEXPR int TEX2_K_BITS = 12;
CONSTEXPR int TEX2_FEEDBACK_AEM_OFFSET = 15;
CONSTEXPR int TEX2_FEEDBACK_TA0_OFFSET = 16;
CONSTEXPR int TEX2_FEEDBACK_TA1_OFFSET = 24;
/////

// PrimitiveAttribute::alpha
CONSTEXPR int ALPHA_AREF_OFFSET = 0;
CONSTEXPR int ALPHA_AREF_BITS = 8;
CONSTEXPR int ALPHA_AFIX_OFFSET = 8;
CONSTEXPR int ALPHA_AFIX_BITS = 8;
/////

// StateVector::combiner
CONSTEXPR int COMBINER_MODE_OFFSET = 0;
CONSTEXPR int COMBINER_MODE_BITS = 2;
CONSTEXPR int COMBINER_MODULATE = 0;
CONSTEXPR int COMBINER_DECAL = 1;
CONSTEXPR int COMBINER_HIGHLIGHT = 2;
CONSTEXPR int COMBINER_HIGHLIGHT2 = 3;
CONSTEXPR int COMBINER_TME_BIT = 1 << 2;
CONSTEXPR int COMBINER_FOG_BIT = 1 << 3;
CONSTEXPR int COMBINER_TCC_BIT = 1 << 4;
/////

// StateVector::blend_mode
CONSTEXPR int BLEND_MODE_ATE_BIT = 1 << 1;
CONSTEXPR int BLEND_MODE_ATE_MODE_OFFSET = 2;
CONSTEXPR int BLEND_MODE_ATE_MODE_BITS = 3;
CONSTEXPR int BLEND_MODE_AFAIL_MODE_OFFSET = 5;
CONSTEXPR int BLEND_MODE_AFAIL_MODE_BITS = 2;
CONSTEXPR int BLEND_MODE_DATE_BIT = 1 << 7;
CONSTEXPR int BLEND_MODE_DATM_BIT = 1 << 8;
CONSTEXPR int BLEND_MODE_ABE_BIT = 1 << 9;
CONSTEXPR int BLEND_MODE_PABE_BIT = 1 << 10;
CONSTEXPR int BLEND_MODE_A_MODE_OFFSET = 11;
CONSTEXPR int BLEND_MODE_A_MODE_BITS = 2;
CONSTEXPR int BLEND_MODE_B_MODE_OFFSET = 13;
CONSTEXPR int BLEND_MODE_B_MODE_BITS = 2;
CONSTEXPR int BLEND_MODE_C_MODE_OFFSET = 15;
CONSTEXPR int BLEND_MODE_C_MODE_BITS = 2;
CONSTEXPR int BLEND_MODE_D_MODE_OFFSET = 17;
CONSTEXPR int BLEND_MODE_D_MODE_BITS = 2;
CONSTEXPR int BLEND_MODE_COLCLAMP_BIT = 1 << 19;
CONSTEXPR int BLEND_MODE_FB_ALPHA_BIT = 1 << 20;
CONSTEXPR int BLEND_MODE_DTHE_BIT = 1 << 21;
/////

// PSM
CONSTEXPR int PSMCT32 = 0x00;
CONSTEXPR int PSMCT24 = 0x01;
CONSTEXPR int PSMCT16 = 0x02;
CONSTEXPR int PSMCT16S = 0x0a;
CONSTEXPR int PSMT8 = 0x13;
CONSTEXPR int PSMT4 = 0x14;
CONSTEXPR int PSMT8H = 0x1b;
CONSTEXPR int PSMT4HL = 0x24;
CONSTEXPR int PSMT4HH = 0x2c;
CONSTEXPR int PSMZ32 = 0x30;
CONSTEXPR int PSMZ24 = 0x31;
CONSTEXPR int PSMZ16 = 0x32;
CONSTEXPR int PSMZ16S = 0x3a;
CONSTEXPR int PS_GPU24 = 0x12; // Only used by CRTC?
/////

// XDIR
CONSTEXPR int HOST_TO_LOCAL = 0;
CONSTEXPR int LOCAL_TO_HOST = 1;
CONSTEXPR int LOCAL_TO_LOCAL = 2;
CONSTEXPR int TRX_DEACTIVATED = 3;
/////

// BLEND
CONSTEXPR int BLEND_RGB_SOURCE = 0;
CONSTEXPR int BLEND_RGB_DEST = 1;
CONSTEXPR int BLEND_RGB_ZERO = 2;
CONSTEXPR int BLEND_RGB_RESERVED = 3;

CONSTEXPR int BLEND_ALPHA_SOURCE = 0;
CONSTEXPR int BLEND_ALPHA_DEST = 1;
CONSTEXPR int BLEND_ALPHA_FIX = 2;
CONSTEXPR int BLEND_ALPHA_RESERVED = 3;
/////

// ATST
CONSTEXPR int ATST_NEVER = 0;
CONSTEXPR int ATST_ALWAYS = 1;
CONSTEXPR int ATST_LESS = 2;
CONSTEXPR int ATST_LEQUAL = 3;
CONSTEXPR int ATST_EQUAL = 4;
CONSTEXPR int ATST_GEQUAL = 5;
CONSTEXPR int ATST_GREATER = 6;
CONSTEXPR int ATST_NOTEQUAL = 7;
/////

// AFAIL
CONSTEXPR int AFAIL_KEEP = 0;
CONSTEXPR int AFAIL_FB_ONLY = 1;
CONSTEXPR int AFAIL_ZB_ONLY = 2;
CONSTEXPR int AFAIL_RGB_ONLY = 3;
/////

#define BINDING_VRAM 0
#define BINDING_COARSE_TILE_LIST 1
#define BINDING_CONSTANTS 2
#define BINDING_COARSE_PRIMITIVE_COUNT 3
#define BINDING_PRIMITIVE_SETUP 4
#define BINDING_VERTEX_POSITION 5
#define BINDING_VERTEX_ATTRIBUTES 6
#define BINDING_TRANSFORMED_ATTRIBUTES 7
#define BINDING_PRIMITIVE_ATTRIBUTES 8
#define BINDING_STATE_VECTORS 9
#define BINDING_FIXED_RCP_LUT 10
#define BINDING_FLOAT_RCP_LUT 11

#define BINDING_SAMPLER_BASE 12
#define BINDING_SAMPLER_NEAREST 12
#define BINDING_SAMPLER_LINEAR 13
#define BINDING_SAMPLER_COUNT 2

#define BINDING_CLUT 14
#define BINDING_TEXTURE_INFO 15

#define BINDING_FEEDBACK_COLOR 16
#define BINDING_FEEDBACK_DEPTH 17
#define BINDING_FEEDBACK_PRIM 18
#define BINDING_FEEDBACK_VARY 19

#define BINDING_SINGLE_SAMPLE_HEURISTIC 20
#define BINDING_OPAQUE_FBMASKS 21
#define BINDING_PHASE_LUT 22

#define DESCRIPTOR_SET_IMAGES 1
#define DESCRIPTOR_SET_WORKGROUP_LIST 2

CONSTEXPR int VARIANT_FLAG_FEEDBACK_BIT = 1 << 0;
CONSTEXPR int VARIANT_FLAG_HAS_AA1_BIT = 1 << 1;
CONSTEXPR int VARIANT_FLAG_HAS_SCANMSK_BIT = 1 << 2;
CONSTEXPR int VARIANT_FLAG_HAS_PRIMITIVE_RANGE_BIT = 1 << 3;
CONSTEXPR int VARIANT_FLAG_HAS_SUPER_SAMPLE_REFERENCE_BIT = 1 << 4;
CONSTEXPR int VARIANT_FLAG_FEEDBACK_DEPTH_BIT = 1 << 5;
CONSTEXPR int VARIANT_FLAG_HAS_TEXTURE_ARRAY_BIT = 1 << 6;

#ifdef __cplusplus
}
#endif

#endif
