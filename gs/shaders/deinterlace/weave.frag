#version 450

// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-License-Identifier: LGPL-3.0+

#extension GL_EXT_samplerless_texture_functions : require

layout(set = 0, binding = 0) uniform texture2D uField0;
layout(set = 0, binding = 1) uniform sampler2D uField1;
layout(set = 0, binding = 2) uniform texture2D uField2;
layout(set = 0, binding = 3) uniform texture2D uField3;
layout(set = 0, binding = 4) uniform itexture2D uMotion;
layout(set = 0, binding = 5) uniform sampler2D uLuma1;

layout(location = 0) out vec4 FragColor;

layout(push_constant) uniform Registers
{
    vec2 inv_field_resolution;
    vec2 inv_qpel_resolution;
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
        ivec2 field_mv = texelFetch(uMotion, coord >> ivec2(2, 3), 0).xy + ivec2(0, 2 - 4 * phase);

        #if 1
        if (diff < 0.001 || all(equal(field_mv, ivec2(0))))
        {
            // If the field pixels remain stable, assume no motion, and accept the old field as-is.
            // Avoids distorting UI significantly with false positive motion.
            // If we detected empty motion vector, assume we have a fade effect or similar, and we'll accept the weave.
            FragColor = vec4(field1, 1.0);
        }
        else
    #endif
        {
            // Compensate for the fact we did motion estimation across fields.
            vec2 pix_uv = (vec2(coord.x, coord.y >> 1) + 0.5) * inv_field_resolution;
            vec2 field_uv = pix_uv + vec2(field_mv) * inv_qpel_resolution;
            vec3 reconstructed = textureLod(uField1, field_uv, 0.0).rgb;

            // Analyze the 3x3 neighborhood to attempt to detect weave artifacts where we don't have good MVs.
            // Fallback to bob in that case.
            vec2 top_left_uv = field_uv - 0.5 * inv_field_resolution;

            vec4 luma00 = textureGather(uLuma1, top_left_uv);
            vec2 luma10 = textureGatherOffset(uLuma1, top_left_uv, ivec2(1, 0)).yz;
            vec2 luma01 = textureGatherOffset(uLuma1, top_left_uv, ivec2(0, 1)).xy;
            float luma11 = textureGatherOffset(uLuma1, top_left_uv, ivec2(1, 1)).w;

            vec2 lo2 = min(min(luma01, luma10), min(luma00.xy, luma00.zw));
            float lo = min(min(lo2.x, lo2.y), luma11);
            vec2 hi2 = max(max(luma01, luma10), max(luma00.xy, luma00.zw));
            float hi = max(max(lo2.x, lo2.y), luma11);

            float luma0 = dot(current0, LUMA);
            float luma1 = dot(current1, LUMA);
            float current_lo = min(luma0, luma1);
            float current_hi = max(luma0, luma1);

            bool artifacting = current_lo > hi || current_hi < lo;
            if (artifacting)
                FragColor = vec4(0.5 * (current0 + current1), 1.0);
            else
                FragColor = vec4(reconstructed, 1.0);
        }
    }
}
