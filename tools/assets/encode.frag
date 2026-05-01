#version 450

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec3 Encoded;

layout(set = 0, binding = 0) uniform sampler2D uTex;
layout(set = 0, binding = 1) uniform sampler2D uBloom;
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

void main()
{
    vec3 color = textureLod(uTex, vUV, 0.0).rgb + textureLod(uBloom, vUV, 0.0).rgb;

    // Color space conversions.
    vec3 display_nits = clamp(primary_transform * color, vec3(0.0), vec3(4000.0));

    if (HDR10)
        Encoded = encode_pq(display_nits);
    else
        Encoded = pow(clamp(display_nits, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));
}
