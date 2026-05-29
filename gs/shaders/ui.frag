#version 450

// SPDX-FileCopyrightText: 2026 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-License-Identifier: LGPL-3.0+

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec4 vColor;
layout(set = 0, binding = 0) uniform sampler2D uTex;
layout(location = 0) out vec4 FragColor;

void main()
{
	FragColor = vColor;
#if TEX
	FragColor *= texture(uTex, vUV);
#endif

	// Hack for when backbuffer is sRGB encoded.
#if LINEAR
	FragColor.rgb = pow(FragColor.rgb, vec3(2.2));
#endif
}

