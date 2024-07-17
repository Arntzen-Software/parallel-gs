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

// The algorithm is modified in two ways:
// - Instead of accepting Field1 directly, extrapolate one field's worth from Field1 and Field3 in time.
//   This should help fading effects.
//   Clamp this result between current field's Y neighbors to avoid overshoots.
//   When Field1 and Field3 differences are very small (or 0), accept Field1 without any changes and clamping.
// - Make the selection between bob and weave a smoothstep instead of step.

const vec3 LUMA = vec3(0.299, 0.587, 0.114);
const float BOB_FACTOR_LO = 0.04;
const float BOB_FACTOR_HI = 0.06;

const float EXTENDED_WEAVE_FACTOR_LO = 0.01;
const float EXTENDED_WEAVE_FACTOR_HI = 0.03;

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
        float extended_weave_factor = smoothstep(EXTENDED_WEAVE_FACTOR_LO, EXTENDED_WEAVE_FACTOR_HI, Mc);

        // If pixel didn't meaningfully change between field1 and field3, accept the color as-is without any neighborhood clamp.
        // We know field3 and field1 at correct position, so extrapolate any delta.
        // Clamp the result to within the neighborhood of current field to avoid overshoots.
        vec3 extended_weave_color = clamp(field1 * 1.5 - field3 * 0.5, min(current0, current1), max(current0, current1));
        vec3 weave_color = mix(field1, extended_weave_color, extended_weave_factor);

        vec3 color = mix(weave_color, 0.5 * (current0 + current1), bob_factor);
        FragColor = vec4(color, 1.0);
    }
}
