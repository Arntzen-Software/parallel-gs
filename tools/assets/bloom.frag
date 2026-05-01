#version 450
#extension GL_EXT_samplerless_texture_functions : require

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec3 Bloom;

layout(set = 0, binding = 0) uniform texture2D uTex;

layout(push_constant) uniform Registers
{
    vec2 input_size, inv_input_size;
    vec2 output_size, inv_output_size;
    float phase; float feedback;
} registers;

void main()
{
    Bloom = vec3(0.0);

    ivec2 base_coord = 2 * ivec2(gl_FragCoord.xy);

    // Could be done better in compute or maybe separable bloom accumulation, but this is simple for now.
    // It runs at half-pixel rate, so shouldn't be a real concern.

    float w = 0.0;

    for (int y = -5; y <= 6; y++)
    {
        for (int x = -5; x <= 6; x++)
        {
            vec2 dist = vec2(x, y) - 0.5;
            float weight = exp2(-0.05 * dot(dist, dist));
            w += weight;
            Bloom += weight * texelFetch(uTex, base_coord + ivec2(x, y), 0).rgb;
        }
    }

    Bloom /= w;
}