#version 450

// SPDX-FileCopyrightText: 2026 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-License-Identifier: LGPL-3.0+

#extension GL_EXT_samplerless_texture_functions : require

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec3 LinearReference;
layout(set = 0, binding = 0) uniform sampler2D uSampler;
layout(set = 0, binding = 1) uniform sampler2D uBack;

layout(push_constant) uniform Registers
{
    vec2 input_size, inv_input_size;
    vec2 output_size, inv_output_size;
    float phase; float feedback;
    float input_strength;
    float gamma;
    float bloom_strength;
    float scan_factor_narrow;
    float scan_factor_wide;
} registers;

void accumulate(vec3 sampled, inout vec3 color, int y, float phase)
{
    phase -= float(y);

    // Breathing effect. Lower intensities on the electron gun leads to narrower beams.
    vec3 inv_stddev = mix(vec3(registers.scan_factor_narrow), vec3(registers.scan_factor_wide), sampled);

    // For progressive scan, the lines need to be wider.
    // The parameters are calibrated for interlaced. Double the stddev, (half the inverse).
    if (registers.phase == 0.0)
        inv_stddev *= 0.5;

    // Basic gaussian, assume normal distribution for where electrons hit the phosphor.
    // For wider beams, compensate to ensure the integral remains fixed.

    // Sample the gaussian in multiple position to get a more correct integral estimate,
    // especially for sharper scanlines.
    vec3 gaussian = vec3(0.0);
    for (int samples = 0; samples < 4; samples++)
    {
        float biased_phase = phase - 0.0625 + 0.125 * float(samples) / 3.0;
        gaussian += exp(-0.5 * inv_stddev * biased_phase * biased_phase);
    }
    gaussian *= 0.3989422 * inv_stddev * 0.25;

    // A little unclear if we should do gamma before or after. Before makes a little more sense I think.
    color += pow(sampled, vec3(registers.gamma)) * gaussian;
}

vec3 sample_scan(vec2 coord)
{
    float input_coord_y = coord.y * registers.input_size.y + registers.phase;
    float floor_coord_y = floor(input_coord_y);
    float phase = (input_coord_y - floor_coord_y) - 0.5;
    coord.y = (floor_coord_y + 0.5) * registers.inv_input_size.y;

    vec3 sampled;
    vec3 color = vec3(0.0);

#define STEP(y) sampled = textureLodOffset(uSampler, coord, 0, ivec2(0, y)).rgb; accumulate(sampled, color, y, phase)
    STEP(-2);
    STEP(-1);
    STEP(+0);
    STEP(+1);
    STEP(+2);

    return color;
}

vec3 grille(vec3 color, vec2 pos)
{
    vec3 mask = vec3(0.125);
    pos.x = fract(pos.x / 3.0);

    if (pos.x < 0.333)
        mask.r = 1.0;
    else if (pos.x < 0.666)
        mask.g = 1.0;
    else
        mask.b = 1.0;

    return color * mask;
}

layout(constant_id = 0) const bool APERTURE_GRILLE = false;

void main()
{
    if (registers.input_strength > 0.0)
    {
        LinearReference = registers.input_strength * sample_scan(vUV);

        if (APERTURE_GRILLE)
            LinearReference = grille(LinearReference, vUV * registers.output_size);
        else
            LinearReference *= 0.5;
    }
    else
    {
        LinearReference = vec3(0.0);
    }

    if (registers.feedback > 0.0)
        LinearReference += registers.feedback * textureLod(uBack, vUV, 0.0).rgb;

    // Safety clamp
    LinearReference = clamp(LinearReference, vec3(0.0), vec3(4000.0));
}