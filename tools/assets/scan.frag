#version 450
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
} registers;

const float VertFactor = 3.0;

void accumulate(vec3 sampled, inout vec3 color, int y, float phase)
{
    phase -= float(y);

    float scanline_phase = abs(phase) * -VertFactor;

    // For progressive scan, the lines need to be wider.
    if (registers.phase == 0.0)
        scanline_phase *= 0.5;

    // Breathing effect. Lower intensities on the electron gun leads to narrower beams.
    vec3 factor_multiplier = mix(vec3(3.0), vec3(1.0), sampled);
    vec3 vert_weight = factor_multiplier * exp2(scanline_phase * factor_multiplier);
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
    STEP(-2);
    STEP(-1);
    STEP(+0);
    STEP(+1);
    STEP(+2);

    return color;
}

vec3 grille(vec3 color, vec2 pos)
{
    vec3 mask = vec3(0.0);
    pos.x = fract(pos.x / 3.0);

    if (pos.x < 0.333)
        mask.r = 1.0;
    else if (pos.x < 0.666)
        mask.g = 1.0;
    else
        mask.b = 1.0;

    return color * mask;
}

void main()
{
    LinearReference = sample_scan(vUV);
    LinearReference = grille(LinearReference, vUV * registers.output_size);

    if (registers.feedback > 0.0)
        LinearReference += registers.feedback * textureLod(uBack, vUV, 0.0).rgb;

    // Safety clamp
    LinearReference = clamp(LinearReference, vec3(0.0), vec3(4000.0));
}