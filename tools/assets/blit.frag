#version 450
#extension GL_EXT_samplerless_texture_functions : require

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec3 Encoded;
layout(location = 1) out vec3 LinearReference;
layout(set = 0, binding = 0) uniform sampler2D uSampler;
layout(set = 0, binding = 1) uniform sampler2D uBack;

layout(push_constant) uniform Registers
{
    vec2 input_size, inv_input_size;
    vec2 output_size, inv_output_size;
    float phase; float feedback;
} registers;

layout(set = 0, binding = 2, std140) uniform Primary
{
    mat3 primary_transform;
};

const float VertFactor = 5.0;

layout(constant_id = 0) const bool HDR10 = false;

void accumulate(vec3 sampled, inout vec3 color, int y, float phase)
{
    phase -= float(y);

    float scanline_phase = abs(phase) * VertFactor;

    // For progressive scan, the lines need to be wider.
    if (registers.phase == 0.0)
        scanline_phase *= 0.5;

    float vert_weight = (VertFactor / 4.0) * cos((0.25 * 3.1415628) * min(scanline_phase, 2.0)) /* * exp2(scanline_phase)*/;
    color += sampled * vert_weight;
}

vec3 sample_scan(vec2 coord)
{
    float input_coord_y = coord.y * registers.input_size.y + registers.phase;
    float floor_coord_y = floor(input_coord_y);
    float phase = (input_coord_y - floor_coord_y) - 0.5;
    coord.y = (floor_coord_y + 0.5) * registers.inv_input_size.y;

    vec3 sampled;
    vec3 color = vec3(0.0);

#define STEP(y) sampled = pow(textureLodOffset(uSampler, coord, 0, ivec2(0, y)).rgb, vec3(2.4)); accumulate(sampled, color, y, phase)
    STEP(-1);
    STEP(+0);
    STEP(+1);

    if (registers.phase == 0.0)
    {
        STEP(-2);
        STEP(+2);
    }

    return color;
}

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

vec3 grille(vec3 color, vec2 pos)
{
    vec3 mask = vec3(0.25);
    pos.x += (pos.y - 0.01) * 3.0;
    pos.x = fract(pos.x / 6.0);
    //pos.x = fract(pos.x / 3.0);

    if (pos.x < 0.333)
        mask.r = 1.5;
    else if (pos.x < 0.666)
        mask.g = 1.5;
    else
        mask.b = 1.5;

    return color * mask;
}

// Distortion of scanlines, and end of screen alpha.
vec2 warp(vec2 pos)
{
    pos = pos * 2.0 - 1.0;

    #if 0
    const float warp_x = 0.02;
    const float warp_y = 0.02;
    pos *= vec2(1.0 + (pos.y * pos.y) * warp_x, 1.0 + (pos.x * pos.x) * warp_y);
    #endif

    return pos * 0.5 + 0.5;
}

void main()
{
    LinearReference = sample_scan(warp(vUV));
    //LinearReference = grille(LinearReference, vUV * registers.output_size);

    if (registers.feedback > 0.0)
        LinearReference += registers.feedback * textureLod(uBack, vUV, 0.0).rgb;

    // Safety clamp
    LinearReference = clamp(LinearReference, vec3(0.0), vec3(4000.0));

    // Color space conversions.
    vec3 display_nits = clamp(primary_transform * LinearReference, vec3(0.0), vec3(4000.0));

    if (HDR10)
        Encoded = encode_pq(display_nits);
    else
        Encoded = pow(clamp(display_nits, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));
}