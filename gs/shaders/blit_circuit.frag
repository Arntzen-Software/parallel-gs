#version 450

// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

layout(set = 0, binding = 0) uniform mediump sampler2D uCircuit;
layout(location = 0) in vec2 vUV;
layout(location = 0) out mediump vec4 FragColor;

void main()
{
    FragColor = textureLod(uCircuit, vUV, 0.0);
}