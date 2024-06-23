#version 450

// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#extension GL_EXT_samplerless_texture_functions : require

layout(set = 0, binding = 0) uniform mediump texture2D uPhase0;
layout(set = 0, binding = 1) uniform mediump texture2D uPhase1;

layout(location = 0) out vec4 FragColor;

void main()
{
    ivec2 coord = ivec2(gl_FragCoord.xy);
    if ((coord.y & 1) != 0)
        FragColor = texelFetch(uPhase1, ivec2(coord.x, coord.y >> 1), 0);
    else
        FragColor = texelFetch(uPhase0, ivec2(coord.x, coord.y >> 1), 0);
}
