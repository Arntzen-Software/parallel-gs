#version 450

// SPDX-FileCopyrightText: 2026 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-License-Identifier: LGPL-3.0+

#extension GL_EXT_samplerless_texture_functions : require

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec3 Output;
layout(set = 0, binding = 0) uniform texture2D uInput;

layout(push_constant) uniform Registers
{
    vec2 input_size, inv_input_size;
    vec2 output_size, inv_output_size;
    vec2 range;
    float bw;
    float max_cll;
} registers;

#if IS_HORIZ
layout(constant_id = 0) const bool HDR10 = false;

vec3 encode_pq(vec3 nits)
{
    // PQ
    vec3 y = nits / 10000.0;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    vec3 num = c1 + c2 * pow(y, vec3(m1));
    vec3 den = 1.0 + c3 * pow(y, vec3(m1));
    vec3 n = pow(num / den, vec3(m2));
    return n;
}

layout(set = 0, binding = 2) uniform UBO
{
    mat3 primary_transform;
};
#endif

const int LOBE = 16;
const float PI = 3.1415628;

float sinc(float phase)
{
    phase *= PI;

    if (abs(phase) < 0.0001)
        return 1.0;
    else
        return sin(phase) / phase;
}

void setup_filter(out ivec2 base_coord, out float phase)
{
    vec2 uv = vUV;

#if IS_HORIZ
    uv.x = uv.x * registers.range.y + registers.range.x;
#else
    uv.y = uv.y * registers.range.y + registers.range.x;
#endif

    vec2 input_coord = uv * registers.input_size;

#if IS_HORIZ
    float coord_x = input_coord.x - 0.5;
    float floor_coord_x = floor(coord_x);
    phase = coord_x - floor_coord_x;
    base_coord = ivec2(floor_coord_x, input_coord.y);
#else
    float coord_y = input_coord.y - 0.5;
    float floor_coord_y = floor(coord_y);
    phase = coord_y - floor_coord_y;
    base_coord = ivec2(input_coord.x, floor_coord_y);
#endif
}

vec3 tonemap(vec3 color, float max_cll)
{
    // Hue-shifting tonemapper. We shouldn't be clipping the curve too hard in practice, probably fine *shrug*.
    vec3 range = clamp(color / max_cll, vec3(0.0), vec3(2.0));
    // Very basic. Can revisit later. Squeezes [0, 2] range into [0, 1].
    vec3 parabola = range - 0.25 * range * range;
    return max_cll * parabola;
}

void main()
{
    vec3 filtered = vec3(0.0);
    ivec2 base_coord;
    float phase;
    setup_filter(base_coord, phase);

    float w = 0.0;

    for (int i = -LOBE; i <= LOBE; i++)
    {
        float filter_phase = phase - float(i);
        float window_phase = clamp(filter_phase / float(LOBE), -1.0, 1.0);

        // Basic windowed sinc approach.
        float kernel = sinc(filter_phase * registers.bw) * cos(0.5 * PI * window_phase);
        w += kernel;

#if IS_HORIZ
        ivec2 offset = ivec2(i, 0);
#else
        ivec2 offset = ivec2(0, i);
#endif
        filtered += kernel * texelFetch(uInput, base_coord + offset, 0).rgb;
    }

    filtered /= w;
    Output = filtered;

#if IS_HORIZ
    // Color space conversions.
    vec3 display_nits = clamp(primary_transform * Output, vec3(0.0), vec3(1000.0));
    if (HDR10)
    {
        Output = encode_pq(tonemap(display_nits, registers.max_cll));
    }
    else
    {
        // SDR will get a bit overexposed by default.
        Output = pow(clamp(tonemap(display_nits * 0.75, 1.0), vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));
    }
#endif
}