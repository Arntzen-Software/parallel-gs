// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#pragma once

#include "page_tracker.hpp"
#include "gs_registers.hpp"
#include "muglm/muglm_impl.hpp"
#include <stdint.h>
#include "shaders/data_structures.h"

namespace ParallelGS
{
// Convert common addressing modes from copies and uploads.
// base is addr / 256 (64 words, i.e. block aligned).
// row_length_64 is effectively page stride.
PageRect compute_page_rect(uint32_t base_256, uint32_t x, uint32_t y,
                           uint32_t width, uint32_t height,
                           uint32_t row_length_64, uint32_t psm);

uint32_t psm_word_write_mask(uint32_t psm);

bool triangle_is_parallelogram_candidate(const VertexPosition *pos, const VertexAttribute *attr,
                                         const muglm::ivec2 &lo, const muglm::ivec2 &hi, const PRIMBits &prim,
                                         muglm::ivec3 &parallelogram_order);

bool triangles_form_parallelogram(const VertexPosition *pos, const VertexAttribute *attr,
                                  const muglm::ivec3 &order,
                                  const VertexPosition *last_pos, const VertexAttribute *last_attr,
                                  const muglm::ivec3 &last_order, const PRIMBits &prim);

void compute_has_potential_feedback(const TEX0Bits &tex0,
                                    uint32_t fb_base_page, uint32_t z_base_page,
                                    uint32_t pages_in_vram, bool &color_feedback, bool &depth_feedback);
}
