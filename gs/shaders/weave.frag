#version 450

// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-License-Identifier: LGPL-3.0+

#extension GL_EXT_samplerless_texture_functions : require

layout(set = 0, binding = 0) uniform texture2D uField0;
layout(set = 0, binding = 1) uniform texture2D uField1;
layout(set = 0, binding = 2) uniform texture2D uField2;
layout(set = 0, binding = 3) uniform texture2D uField3;

layout(location = 0) out vec4 FragColor;

layout(push_constant) uniform Registers
{
    int phase;
    int height_minus_1;
};

// FastMAD algorithm with some minor tweaks.
// Original documentation: https://www.patreon.com/posts/fastmad-73291458
// It's basically an adaptive filter that selects dynamically between weave and bob
// based on local field difference.
// To improve further on this, local motion estimation and compensation should help.

const vec3 LUMA = vec3(0.299, 0.587, 0.114);
const float BOB_FACTOR_LO = 0.04;
const float BOB_FACTOR_HI = 0.06;

void main()
{
    ivec2 coord = ivec2(gl_FragCoord.xy);
    if ((coord.y & 1) == phase)
    {
        // If we have the pixel in current field, accept as-is.
        FragColor = texelFetch(uField0, ivec2(coord.x, coord.y >> 1), 0);
    }
    else if (coord.y == 0 || coord.y == height_minus_1)
    {
        // On Y edge, just accept the previous field as-is.
        FragColor = texelFetch(uField1, ivec2(coord.x, coord.y >> 1), 0);
    }
    else
    {
        int up_y = (coord.y - 1) >> 1;
        vec3 current0 = texelFetch(uField0, ivec2(coord.x, up_y), 0).rgb;
        vec3 current1 = texelFetchOffset(uField0, ivec2(coord.x, up_y), 0, ivec2(0, 1)).rgb;
        vec3 prev0 = texelFetch(uField2, ivec2(coord.x, up_y), 0).rgb;
        vec3 prev1 = texelFetchOffset(uField2, ivec2(coord.x, up_y), 0, ivec2(0, 1)).rgb;

        vec3 field1 = texelFetch(uField1, ivec2(coord.x, coord.y >> 1), 0).rgb;
        vec3 field3 = texelFetch(uField3, ivec2(coord.x, coord.y >> 1), 0).rgb;

        // Study local differences in luma for where we have sample points.
        float Mh = abs(dot(current0 - prev0, LUMA));
        float Ml = abs(dot(current1 - prev1, LUMA));
        float Mc = abs(dot(field1 - field3, LUMA));

        float diff = max(min(Mh, Ml), Mc);

        float bob_factor = smoothstep(BOB_FACTOR_LO, BOB_FACTOR_HI, diff);
        vec3 color = mix(field1, 0.5 * (current0 + current1), bob_factor);
        FragColor = vec4(color, 1.0);
    }
}
