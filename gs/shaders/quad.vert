// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#version 450

layout(location = 0) out vec2 vUV;

void main()
{
    float x = float(gl_VertexIndex & 2) * 2.0 - 1.0;
    float y = float(gl_VertexIndex & 1) * 4.0 - 1.0;
    gl_Position = vec4(x, y, 0.0, 1.0);

    vUV = 0.5 * gl_Position.xy + 0.5;
}