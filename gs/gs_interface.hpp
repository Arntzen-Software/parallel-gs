// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#pragma once

#include "page_tracker.hpp"
#include "gs_renderer.hpp"
#include "gs_registers.hpp"
#include "device.hpp"
#include "intrusive_hash_map.hpp"
#include "dynamic_array.hpp"
#include <stddef.h>
#include <vector>
#include <type_traits>

namespace ParallelGS
{
struct ContextState
{
	Reg64<TEX0Bits> tex0;
	Reg64<TEX1Bits> tex1;
	Reg64<CLAMPBits> clamp;
	Reg64<XYOFFSETBits> xyoffset;
	Reg64<MIPTBPBits> miptbl_1_3;
	Reg64<MIPTBPBits> miptbl_4_6;
	Reg64<SCISSORBits> scissor;
	Reg64<ALPHABits> alpha;
	Reg64<TESTBits> test;
	Reg64<FBABits> fba;
	Reg64<FRAMEBits> frame;
	Reg64<ZBUFBits> zbuf;
};

struct RegisterState
{
	ContextState ctx[2];
	Reg64<PRIMBits> prim;
	Reg64<RGBAQBits> rgbaq;
	Reg64<STBits> st;
	Reg64<UVBits> uv;
	Reg64<FOGBits> fog;
	Reg64<PRMODECONTBits> prmodecont;
	Reg64<TEXCLUTBits> texclut;
	Reg64<TEXABits> texa;
	Reg64<FOGCOLBits> fogcol;
	Reg64<DIMXBits> dimx;
	Reg64<DTHEBits> dthe;
	Reg64<COLCLAMPBits> colclamp;
	Reg64<PABEBits> pabe;
	Reg64<BITBLTBUFBits> bitbltbuf;
	Reg64<TRXPOSBits> trxpos;
	Reg64<TRXREGBits> trxreg;
	Reg64<TRXDIRBits> trxdir;
	Reg64<SCANMSKBits> scanmsk;
	// XYZ and XYZF are consumed inline in vertex kick.

	float internal_q = 1.0f;
	uint32_t cached_cbp[2] = {};
};

struct GIFPath
{
	GIFTagBits tag;
	uint32_t reg;
	uint32_t loop;
};

struct PrivRegisterState
{
	union
	{
		struct
		{
			alignas(16) PMODEBits pmode;
			alignas(16) SMODE1Bits smode1;
			alignas(16) SMODE2Bits smode2;
			alignas(16) DummyBits srfsh;
			alignas(16) DummyBits synch1;
			alignas(16) DummyBits synch2;
			alignas(16) SYNCVBits syncv;
			alignas(16) DISPFBBits dispfb1;
			alignas(16) DISPLAYBits display1;
			alignas(16) DISPFBBits dispfb2;
			alignas(16) DISPLAYBits display2;
			alignas(16) EXTBUFBits extbuf;
			alignas(16) EXTDATABits extdata;
			alignas(16) EXTWRITEBits extwrite;
			alignas(16) BGCOLORBits bgcolor;
		};
		uint64_t qwords_lo[0x200];
	};

	union
	{
		struct
		{
			alignas(16) CSRBits csr;
			alignas(16) IMRBits imr;
			uint64_t pad0_[5];
			alignas(16) BUSDIRBits busdir;
			uint64_t pad1_[7];
			alignas(16) SIGLBLIDBits siglblid;
		};
		uint64_t qwords_hi[0x200];
	};
};

enum StateDirtyFlagBits : uint32_t
{
	STATE_DIRTY_FB_BIT = 1 << 0,
	STATE_DIRTY_TEX_BIT = 1 << 1,
	STATE_DIRTY_STATE_BIT = 1 << 2,
	STATE_DIRTY_PRIM_TEMPLATE_BIT = 1 << 3,
	STATE_DIRTY_FEEDBACK_BIT = 1 << 4,
	STATE_DIRTY_DEGENERATE_BIT = 1 << 5,
	STATE_DIRTY_ALL_BITS = STATE_DIRTY_FB_BIT |
	                       STATE_DIRTY_TEX_BIT |
	                       STATE_DIRTY_STATE_BIT |
	                       STATE_DIRTY_PRIM_TEMPLATE_BIT |
	                       STATE_DIRTY_FEEDBACK_BIT |
	                       STATE_DIRTY_DEGENERATE_BIT
};
using StateDirtyFlags = uint32_t;

struct DebugMode
{
	bool feedback_render_target = false;
	bool timestamps = false;

	enum class DrawDebugMode
	{
		None,
		Strided,
		Full,
		Count
	};
	DrawDebugMode draw_mode = DrawDebugMode::None;
};

struct VSyncInfo
{
	uint32_t phase;
	VkImageLayout dst_layout;
	VkPipelineStageFlags2 dst_stage;
	VkAccessFlags2 dst_access;

	// Attempts to force progressive scan when possible, i.e. INT = 1, FFMD = 0.
	// Rather than scanning out every other line as prescribed, just scan out the full resolution as-is.
	bool force_progressive;
	// If true, includes the overscan area. This results in black borders.
	bool overscan;
	// Tries to counteract field blending.
	// For force_progressive or super-sampling, setting this to true is usually a good idea.
	bool anti_blur;
	// Honor CRTC offsets. Otherwise, tries to avoid any slight pixel shifts causing small black borders.
	// Especially relevant in no-overscan mode.
	bool crtc_offsets;

	// Use MAGH to deduce desired scanout resolution horizontally.
	// Generally, analog video does not have a fixed horizontal resolution, so we're free to fudge it a bit.
	// Avoids extra bi-linear blur for 512x448 games, but will interact funny with integer scaling for example.
	bool adapt_to_internal_horizontal_resolution;
};

struct GSOptions
{
	SuperSampling super_sampling = SuperSampling::X1;
	uint32_t vram_size = 4 * 1024 * 1024; // This should generally not be touched.
	bool dynamic_super_sampling = false; // If super sampling rate can be toggled in-flight.
};

class GSInterface final : private PageTrackerCallback
{
public:
	GSInterface();
	bool init(Vulkan::Device *device, const GSOptions &options);

	void set_super_sampling_rate(SuperSampling super_sampling);
	void set_debug_mode(const DebugMode &mode);

	// GIF payload format.
	void gif_transfer(uint32_t path, const void *data, size_t size);

	// Manually tickle register state and perform any work associated with accessing said register.
	void write_register(RegisterAddr addr, uint64_t payload);

	template <typename T>
	void write_register(RegisterAddr addr, const T &t)
	{
		static_assert(std::is_pod<T>::value &&
		              sizeof(T) == sizeof(uint64_t), "Type is not 64-bit POD union");

		Reg64<T> reg{t};
		write_register(addr, reg.bits);
	}

	void *map_vram_write(size_t offset, size_t size);
	void end_vram_write(size_t offset, size_t size);
	const void *map_vram_read(size_t offset, size_t size);

	void flush();

	void clobber_register_state();

	RegisterState &get_register_state();
	const RegisterState &get_register_state() const;

	PrivRegisterState &get_priv_register_state();
	const PrivRegisterState &get_priv_register_state() const;

	GIFPath &get_gif_path(uint32_t path);
	const GIFPath &get_gif_path(uint32_t path) const;

	ScanoutResult vsync(const VSyncInfo &info);

	FlushStats consume_flush_stats();
	double get_accumulated_timestamps(TimestampType type) const;

	void read_transfer_fifo(void *data, uint32_t num_128b_words);

private:
	void flush(PageTrackerFlushFlags flags, FlushReason reason) override;
	void sync_host_vram_page(uint32_t page_index) override;
	void sync_vram_host_page(uint32_t page_index) override;
	void invalidate_texture_hash(Util::Hash hash, bool clut) override;
	void forget_in_render_pass_memoization() override;
	void recycle_image_handle(Vulkan::ImageHandle image) override;
	void flush_render_pass(FlushReason reason);

	void mark_texture_state_dirty();

	GSRenderer renderer;
	PageTracker tracker;
	uint32_t vram_size = 0;
	DebugMode debug_mode;

	std::vector<uint32_t> sync_host_vram_pages;
	std::vector<uint32_t> sync_vram_host_pages;
	std::vector<uint32_t> page_buffer;

	struct TransferState
	{
		std::vector<uint64_t> host_to_local_payload;
		bool host_to_local_active = false;
		uint32_t required_qwords = 0;
		uint32_t last_flushed_qwords = 0;
		CopyDescriptor copy = {};
		Util::DynamicArray<uint8_t> fifo_readback;
		uint32_t fifo_readback_128b_offset = 0;
		uint32_t fifo_readback_128b_size = 0;
	} transfer_state;

	void flush_pending_transfer(bool keep_alive);
	void check_pending_transfer();
	void init_transfer();

	RegisterState registers = {};
	PrivRegisterState priv_registers = {};

	// Tracks state for a render pass
	enum { NumMemoizedPalettes = 16 };
	struct RenderPassState
	{
		Util::DynamicArray<VertexPosition> positions;
		Util::DynamicArray<VertexAttribute> attributes;
		Util::DynamicArray<PrimitiveAttribute> prim;
		uint32_t primitive_count = 0;

		std::vector<StateVector> state_vectors;
		std::vector<Vulkan::ImageHandle> held_images;
		std::vector<TextureInfo> tex_infos;

		uint32_t clut_instance = 0;
		uint32_t latest_clut_instance = 0;

		struct TextureStateToLocalIndex : Util::IntrusiveHashMapEnabled<TextureStateToLocalIndex>
		{
			explicit TextureStateToLocalIndex(uint32_t index_) : index(index_) {}
			uint32_t index;
			bool valid = true;
		};

		Util::IntrusiveHashMap<Util::IntrusivePODWrapper<uint32_t>> state_vector_map;
		Util::IntrusiveHashMap<TextureStateToLocalIndex> texture_map;

		Reg64<FRAMEBits> frame = {};
		Reg64<ZBUFBits> zbuf = {};
		uint32_t pending_palette_updates = 0;

		struct MemoizedPaletteState
		{
			uint32_t clut_instance;
			uint32_t csa_mask;
			PaletteUploadDescriptor upload;
		};
		MemoizedPaletteState memoized_palettes[NumMemoizedPalettes];
		uint32_t num_memoized_palettes = 0;

		ivec4 bb = ivec4(INT32_MAX, INT32_MAX, INT32_MIN, INT32_MIN);
		uint32_t fb_page_width_log2 = 0;
		uint32_t fb_page_height_log2 = 0;
		uint32_t z_page_width_log2 = 0;
		uint32_t z_page_height_log2 = 0;
		uint32_t color_write_mask = 0;
		bool z_sensitive = false;
		bool z_write = false;
		bool is_color_feedback = false;
		bool is_awkward_color_feedback = false;
		bool is_potential_color_feedback = false;
		bool is_potential_depth_feedback = false;
		bool has_color_feedback = false;
		bool has_aa1 = false;
		bool has_scanmsk = false;

		uint32_t feedback_psm = 0;
		uint32_t feedback_cpsm = 0;

		// For debug
		uint32_t label_key = 0;

		int32_t ofx = 0;
		int32_t ofy = 0;
	} render_pass;

	// Register handlers

	using RegListHandler = void (GSInterface::*)(uint64_t payload);
	using PackedHandler = void (GSInterface::*)(const void *words);
	RegListHandler ad_handlers[128] = {};
	RegListHandler reglist_handlers[16] = {};
	PackedHandler packed_handlers[16] = {};
	void setup_handlers();

	void reglist_nop(uint64_t payload);
	void packed_nop(const void *words);

	template <RegListHandler Handler> void packed_a_d_forward(const void *words);
	void packed_RGBAQ(const void *words);
	void packed_ST(const void *words);
	void packed_UV(const void *words);
	void packed_FOG(const void *words);

#define DECL_REG(reg, addr) void a_d_##reg(uint64_t payload);
#include "gs_register_addr.hpp"
#undef DECL_REG
	template <int CTX> void a_d_TEX2(uint64_t payload);

	void shift_vertex_queue();
	void vertex_kick_xyz(Reg64<XYZBits> xyz);
	void vertex_kick_xyzf(Reg64<XYZFBits> xyzf);
	template <bool ADC> void packed_XYZF(const void *words);
	template <bool ADC> void packed_XYZ(const void *words);
	template <bool ADC, bool FOG, PRIMType PRIM> void packed_XYZ(const void *words);
	void drawing_kick(bool adc);
	template <PRIMType PRIM>
	void drawing_kick(bool adc);
	void reset_vertex_queue();

	using DrawHandler = void (GSInterface::*)(bool adc);
	// Specialized handlers that come up again and again.
	// Idea is from GSdx, but the concept is very obvious when you see some dumps.
	using OptimizedPacketHandler = void (GSInterface::*)(const void *words, uint32_t nloops);
	DrawHandler draw_handler = nullptr;
	// One per GIFPath.
	OptimizedPacketHandler optimized_draw_handler[4] = {};

	// Optimized handlers.
	template <bool FOG, PRIMType PRIM, int factor>
	void packed_STQRGBAXYZ(const void *words, uint32_t num_vertices);
	template <bool FOG, PRIMType PRIM, int factor>
	void packed_UVRGBAXYZ(const void *words, uint32_t num_vertices);
	template <int count>
	void packed_ADONLY(const void *words, uint32_t num_loops);
	template <bool FOG>
	void packed_STXYZSTRGBAXYZ_sprite(const void *workds, uint32_t num_loops);

	void update_draw_handler();
	void update_optimized_gif_handler(uint32_t path);

	enum class ColorFeedbackMode
	{
		None,
		Pixel,
		Sliced,
		BypassHazards
	};

	uint32_t drawing_kick_update_texture(ColorFeedbackMode feedback_mode, const ivec4 &uv_bb, const ivec4 &bb);

	struct PrimitiveTemplate
	{
		uint32_t state, tex, tex2;
	};

	struct StateTracker
	{
		PrimitiveTemplate prim_template = {};
		StateDirtyFlags dirty_flags = STATE_DIRTY_ALL_BITS;
		bool degenerate_draw = false;

		StateVector last_state_vector = {};
		uint32_t last_state_index = 0;
		TextureDescriptor last_texture_descriptor = {};
		uint32_t last_texture_index = 0;

		struct StateTrackerTexture
		{
			PageRect page_rects[7];
			TexRect rect;
			struct
			{
				uint32_t base;
				uint32_t stride;
			} levels[7];
		} tex = {};
	} state_tracker;

	void update_internal_register(uint64_t &reg, uint64_t value, StateDirtyFlags flags);

	void update_texture_page_rects_and_read();
	void texture_page_rects_read();
	bool get_and_clear_dirty_flag(StateDirtyFlags flags);

	void check_frame_buffer_state();
	void mark_render_pass_has_texture_feedback(const TEX0Bits &tex0);
	bool draw_is_degenerate();
	uint32_t find_or_place_unique_state_vector(const StateVector &state);
	uint32_t drawing_kick_update_state_vector();

	void drawing_kick_update_state(ColorFeedbackMode feedback_mode, const ivec4 &uv_bb, const ivec4 &bb);
	bool state_is_z_sensitive() const;

	template <bool quad, unsigned num_vertices>
	ColorFeedbackMode deduce_color_feedback_mode(const VertexPosition *pos, const VertexAttribute *attr,
	                                             const ContextState &ctx, const PRIMBits &prim,
	                                             ivec4 &uv_bb, const ivec4 &bb);

	template <bool list_primitive, bool fan_primitive, bool quad, unsigned num_vertices>
	void drawing_kick_primitive(bool adc);
	template <bool list_primitive, bool fan_primitive, bool quad, unsigned num_vertices>
	void drawing_kick_append();
	template <bool list_primitive, bool fan_primitive, bool quad, unsigned num_vertices>
	void drawing_kick_maintain_queue();

	void drawing_kick_invalid(bool);
	void post_draw_kick_handler();

	struct
	{
		enum { MaxEntries = 3 };
		VertexPosition pos[MaxEntries];
		VertexAttribute attr[MaxEntries];
		unsigned count = 0;
	} vertex_queue;

	void handle_tex0_write(uint32_t ctx);
	void handle_clut_upload(uint32_t ctx);
	void handle_miptbl_gen(uint32_t ctx);

	PageRect compute_fb_rect(const ivec4 &bb) const;
	PageRect compute_z_rect(const ivec4 &bb) const;

	GIFPath paths[4] = {};
	void a_d_HWREG_multi(const uint64_t *payload, size_t count);

	void update_color_feedback_state();

	uint32_t sampling_rate_x_log2 = 0;
	uint32_t sampling_rate_y_log2 = 0;
};
}
