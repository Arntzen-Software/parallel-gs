// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#version 450

#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_samplerless_texture_functions : require
#include "data_structures.h"
#include "swizzle_utils.h"
#include "utils.h"

layout(location = 0) out vec4 FragColor;

#if PROMOTED
layout(set = 0, binding = 0) uniform texture2DArray uPromoted;
#else
layout(set = 0, binding = 0) readonly buffer VRAM32
{
    uint data[];
} vram32;

layout(set = 0, binding = 0) readonly buffer VRAM16
{
    uint16_t data[];
} vram16;
#endif

layout(push_constant) uniform Registers
{
    uint fbp;
    uint fbw;
    uint dbx;
    uint dby;
    uint phase;
    uint phase_stride;
} registers;

layout(constant_id = 0) const int PSM = 0;
layout(constant_id = 1) const uint VRAM_MASK = 4 * 1024 * 1024 - 1;
layout(constant_id = 2) const uint SUPER_SAMPLES = 1;

const bool is_tex_16bit = PSM == PSMCT16 || PSM == PSMCT16S || PSM == PSMZ16 || PSM == PSMZ16S;

#if PROMOTED
vec4 sample_vram(uvec2 coord, uint slice)
{
    return texelFetch(uPromoted, ivec3(coord, slice), 0);
}
#else
vec4 sample_vram(uint addr, uint slice)
{
    uint payload;

    if (is_tex_16bit)
    {
        payload = rgba16_to_rgba32(uint(vram16.data[addr + slice * (VRAM_MASK + 1) / 2]), 0, 0x0, 0xff);
    }
    else
    {
        payload = vram32.data[addr + slice * (VRAM_MASK + 1) / 4];
        if (PSM != PSMCT32)
        {
            payload &= ~0xff000000u;
            payload |= 0x80000000u;
        }
    }

    return unpackUnorm4x8(payload);
}
#endif

#if PROMOTED
bool super_sample_is_valid(uvec2 coord) { return true; }
#else
bool super_sample_is_valid(uint addr)
{
    bool is_valid;

    if (is_tex_16bit)
    {
        is_valid = uint(vram16.data[addr + (VRAM_MASK + 1) / 2]) == 0xffff;
    }
    else
    {
        uint payload = vram32.data[addr + (VRAM_MASK + 1) / 4];
        if (PSM != PSMCT32)
            is_valid = (payload & 0xffffffu) == 0xffffffu;
        else
            is_valid = payload == ~0u;
    }

    return is_valid;
}
#endif

void main()
{
    uvec2 super_sampled_coord = uvec2(gl_FragCoord.xy);
    uvec2 single_sampled_coord;
    if (SUPER_SAMPLES >= 4)
        single_sampled_coord = super_sampled_coord >> 1;
    else
        single_sampled_coord = super_sampled_coord;

    // Is this how full-height interlace is supposed to work? :|
    uvec2 coord = single_sampled_coord * uvec2(1u, registers.phase_stride) +
        uvec2(registers.dbx, registers.dby + registers.phase);

#if PROMOTED
    #define addr coord
    const int BASE_SSAA_LAYER = 1;
#else
    uint addr = swizzle_PS2(coord.x, coord.y, registers.fbp * PGS_BLOCKS_PER_PAGE, registers.fbw, PSM, VRAM_MASK);
    const int BASE_SSAA_LAYER = 2;
#endif

    if (SUPER_SAMPLES == 1)
    {
        // SUPER_SAMPLES == 2 forces 1 path.
        FragColor = sample_vram(addr, 0);
    }
    else if (SUPER_SAMPLES >= 4)
    {
        if (super_sample_is_valid(addr))
        {
            uint quad_offset;

            const uint NUM_SSAA_SAMPLES = SUPER_SAMPLES / 4;

            // The swizzling pattern for checkerboard is a bit different.
            if (SUPER_SAMPLES != 8)
                quad_offset = (super_sampled_coord.y & 1u) + (super_sampled_coord.x & 1u) * 2u;
            else
                quad_offset = (super_sampled_coord.x & 1u) + (super_sampled_coord.y & 1u) * 2u;

            uint base_slice = BASE_SSAA_LAYER + NUM_SSAA_SAMPLES * quad_offset;

            FragColor = vec4(0.0);
            for (uint i = 0; i < NUM_SSAA_SAMPLES; i++)
                FragColor += sample_vram(addr, base_slice + i);
            FragColor /= float(NUM_SSAA_SAMPLES);
        }
        else
            FragColor = sample_vram(addr, 0);
    }
}
