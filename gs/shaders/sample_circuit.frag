// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#version 450

#extension GL_EXT_shader_16bit_storage : require
#include "data_structures.h"
#include "swizzle_utils.h"
#include "utils.h"

layout(location = 0) out vec4 FragColor;

layout(set = 0, binding = 0) readonly buffer VRAM32
{
    uint data[];
} vram32;

layout(set = 0, binding = 0) readonly buffer VRAM16
{
    uint data[];
} vram16;

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

const bool is_tex_16bit = PSM == PSMCT16 || PSM == PSMCT16S || PSM == PSMZ16 || PSM == PSMZ16S;

void main()
{
    // Is this how full-height interlace is supposed to work? :|
    uvec2 coord = uvec2(gl_FragCoord.xy) * uvec2(1u, registers.phase_stride) +
        uvec2(registers.dbx, registers.dby + registers.phase);

    uint addr = swizzle_PS2(coord.x, coord.y, registers.fbp * BLOCKS_PER_PAGE, registers.fbw, PSM, VRAM_MASK);

    uint payload = 0;

    if (is_tex_16bit)
        payload = rgba16_to_rgba32(uint(vram16.data[addr]), 0, 0x0, 0xff);
    else
    {
        payload = vram32.data[addr];
        if (PSM != PSMCT32)
        {
            payload &= ~0xff000000u;
            payload |= 0x80000000u;
        }
    }

    FragColor = unpackUnorm4x8(payload);
}