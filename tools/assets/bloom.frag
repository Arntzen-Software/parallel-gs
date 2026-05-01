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

    ivec2 base_coord = ivec2(gl_FragCoord.xy);

    // Could be done better in compute or maybe separable bloom accumulation, but this is simple enough for now.

    for (int y = -4; y <= 4; y++)
    {
        for (int x = -4; x <= 4; x++)
        {
            // Supposed to consider scattering contributions from neighbors.
            if (x == 0 && y == 0)
                continue;

            vec2 dist = vec2(x, y);
            float weight = exp2(-0.25 * dot(dist, dist));
            Bloom += weight * texelFetch(uTex, base_coord + ivec2(x, y), 0).rgb;
        }
    }

    Bloom *= 0.25;
    Bloom += texelFetch(uTex, base_coord, 0).rgb;
}