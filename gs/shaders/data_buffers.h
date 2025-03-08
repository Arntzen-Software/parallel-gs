// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#ifndef DATA_BUFFERS_H_
#define DATA_BUFFERS_H_

#include "data_structures.h"

#ifdef NEED_VRAM
layout(set = 0, binding = BINDING_VRAM, std430) buffer VRAM32
{
	uint data[];
} vram32;

layout(set = 0, binding = BINDING_VRAM, std430) buffer VRAM24
{
	u8vec3 data[];
} vram24;

layout(set = 0, binding = BINDING_VRAM, std430) buffer VRAM16
{
	uint16_t data[];
} vram16;
#endif

#ifdef NEED_PRIMITIVE_LIST
layout(set = 0, binding = BINDING_COARSE_TILE_LIST, std430)
PRIMITIVE_LIST_QUALIFIER buffer CoarseTileList
{
	uint16_t data[];
} coarse_primitive_list;
#endif

#ifdef NEED_CONSTANTS
layout(set = 0, binding = BINDING_CONSTANTS, std140) uniform UBO
{
	GlobalConstants constants;
};
#endif

#ifdef NEED_PRIMITIVE_COUNTS
layout(set = 0, binding = BINDING_COARSE_PRIMITIVE_COUNT, std430)
PRIMITIVE_LIST_QUALIFIER buffer CoarsePrimitiveCount
{
	int data[];
} coarse_primitive_counts;
#endif

#ifdef NEED_PRIMITIVE_SETUP
layout(set = 0, binding = BINDING_PRIMITIVE_SETUP, std430)
PRIMITIVE_SETUP_QUALIFIER buffer PrimitiveSetups
{
	PrimitiveSetup data[];
} primitive_setup;
#endif

#ifdef NEED_VERTEX_POSITION
layout(set = 0, binding = BINDING_VERTEX_POSITION, std430)
readonly buffer VertexPositions
{
	VertexPosition data[];
} vertex_position;
#endif

#ifdef NEED_VERTEX_ATTRIBUTE
layout(set = 0, binding = BINDING_VERTEX_ATTRIBUTES, std430)
readonly buffer VertexAttributes
{
	VertexAttribute data[];
} vertex_attr;
#endif

const int MAX_PRIMITIVES = 64 * 1024;

#ifdef NEED_TEXTURE_INFO
layout(set = 0, binding = BINDING_TEXTURE_INFO, std140)
uniform TextureInfo
{
	TexInfo data[1024];
} texture_info;
#endif

#ifdef NEED_TRANSFORMED_ATTRIBUTE
layout(set = 0, binding = BINDING_TRANSFORMED_ATTRIBUTES, std430)
PRIMITIVE_SETUP_QUALIFIER buffer TransformedAttribute
{
	TransformedAttributes data[];
} transformed_attr;
#endif

#ifdef NEED_PRIMITIVE_ATTRIBUTE
layout(set = 0, binding = BINDING_PRIMITIVE_ATTRIBUTES, std430)
buffer PrimitiveAttributes
{
	PrimitiveAttribute data[];
} primitive_attr;
#endif

#ifdef NEED_STATE_VECTORS
layout(set = 0, binding = BINDING_STATE_VECTORS, std140) uniform StateVectors
{
	StateVector data[1 << STATE_INDEX_BIT_COUNT];
} state_vectors;
#endif

#ifdef BINDLESS
layout(set = 0, binding = BINDING_SAMPLER_NEAREST) uniform sampler nearest_sampler;
layout(set = 0, binding = BINDING_SAMPLER_LINEAR) uniform sampler linear_sampler;
layout(set = DESCRIPTOR_SET_IMAGES, binding = 0) uniform texture2D bindless_textures[];
layout(set = DESCRIPTOR_SET_IMAGES, binding = 0) uniform texture2DArray bindless_textures_array[];
#endif

#ifdef USE_RCP_FIXED
layout(set = 0, binding = BINDING_FIXED_RCP_LUT) uniform utextureBuffer RCPLut;
#endif

#ifdef USE_RCP_FLOAT
layout(set = 0, binding = BINDING_FLOAT_RCP_LUT) uniform textureBuffer RCPLutFloat;
#endif

#endif
