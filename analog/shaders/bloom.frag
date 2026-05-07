#version 450

// SPDX-FileCopyrightText: 2026 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-License-Identifier: LGPL-3.0+

#extension GL_EXT_control_flow_attributes : require

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec3 Bloom;

layout(set = 0, binding = 0) uniform sampler2D uTex;

layout(push_constant) uniform Registers
{
    vec2 input_size, inv_input_size;
    vec2 output_size, inv_output_size;
    float phase; float feedback;
    float input_strength;
} registers;

layout(set = 0, binding = 1) uniform BlurKernel
{
    vec4 weights;
    vec4 offsets;
} kernel;

layout(constant_id = 0) const bool APPLY = false;

vec3 tap(float weight, float off_x, float off_y)
{
    return weight * textureLod(uTex, vUV + registers.inv_output_size * vec2(off_x, off_y), 0.0).rgb;
}

void main()
{
    if (APPLY)
    {
        // Blending.
        Bloom = vec3(0.0);
        Bloom += textureLod(uTex, vUV + vec2(-0.5, -0.5) * registers.inv_output_size, 0.0).rgb;
        Bloom += textureLod(uTex, vUV + vec2(+0.5, -0.5) * registers.inv_output_size, 0.0).rgb;
        Bloom += textureLod(uTex, vUV + vec2(-0.5, +0.5) * registers.inv_output_size, 0.0).rgb;
        Bloom += textureLod(uTex, vUV + vec2(+0.5, +0.5) * registers.inv_output_size, 0.0).rgb;
        Bloom *= 0.25;
    }
    else
    {
        Bloom = vec3(0.0);

        vec4 weights = kernel.weights;
        vec4 offsets = kernel.offsets;

        [[unroll]]
        for (int y = -2; y <= 2; y++)
        {
            [[unroll]]
            for (int x = -2; x <= 2; x++)
            {
                Bloom += tap(weights[abs(x)] * weights[abs(y)],
                    sign(float(x)) * offsets[abs(x)],
                    sign(float(y)) * offsets[abs(y)]);
            }
        }

        float edge_weights = weights.x * weights.w;
        Bloom += tap(edge_weights, -offsets.w, 0.0);
        Bloom += tap(edge_weights, +offsets.w, 0.0);
        Bloom += tap(edge_weights, 0.0, -offsets.w);
        Bloom += tap(edge_weights, 0.0, +offsets.w);
    }
}