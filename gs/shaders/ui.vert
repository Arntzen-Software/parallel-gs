#version 450

// SPDX-FileCopyrightText: 2026 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-License-Identifier: LGPL-3.0+

layout(location = 0) in vec2 Pos;
layout(location = 1) in vec2 UV;
layout(location = 2) in vec4 Color;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec4 vColor;

layout(push_constant) uniform Registers
{
	vec2 inv_resolution;
};

void main()
{
	gl_Position = vec4(Pos * inv_resolution * 2.0 - 1.0, 0.0, 1.0);
	vUV = UV;
	vColor = Color;
}

