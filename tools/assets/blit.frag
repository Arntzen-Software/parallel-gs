#version 450
#extension GL_EXT_samplerless_texture_functions : require

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec3 FragColor;
layout(set = 0, binding = 0) uniform texture2D uSampler;

layout(push_constant) uniform Registers
{
    vec2 input_size, inv_input_size;
    vec2 output_size, inv_output_size;
};

layout(set = 0, binding = 1, std140) uniform Primary
{
    mat3 primary_transform;
};

const float HorizFactor = 2.0;
const float VertFactor = 4.0;

layout(constant_id = 0) const bool HDR10 = false;

void accumulate(vec3 sampled, inout vec3 color, inout float w, int x, int y, vec2 phase)
{
    phase.x -= float(x);
    phase.y -= float(y);
    float horiz_weight = exp2(abs(phase.x) * -HorizFactor);
    float vert_weight = exp2(abs(phase.y) * -VertFactor);
    color += sampled * horiz_weight * vert_weight;
    w += horiz_weight;
}

vec3 sample_scan(vec2 coord)
{
    vec2 floor_coord = floor(coord);
    vec2 phase = (coord - floor_coord) - 0.5;
    ivec2 icoord = ivec2(floor_coord);

    vec3 color = vec3(0.0);
    float w = 0.0;
    vec3 sampled;

#define STEP(x, y) sampled = texelFetchOffset(uSampler, icoord, 0, ivec2(x, y)).rgb; accumulate(sampled, color, w, x, y, phase)
#define STEP_HORIZ(y) STEP(-3, y); STEP(-2, y); STEP(-1, y); STEP(0, y); STEP(+1, y); STEP(+2, y); STEP(+3, y)
    STEP_HORIZ(-1);
    STEP_HORIZ(+0);
    STEP_HORIZ(+1);

    return color / w;
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

void main()
{
    vec2 input_coord = vUV * input_size;
    FragColor = sample_scan(input_coord * 0.25);

    if (HDR10)
        FragColor = encode_pq(primary_transform * FragColor);
    else
        FragColor = pow(clamp(FragColor, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));
}