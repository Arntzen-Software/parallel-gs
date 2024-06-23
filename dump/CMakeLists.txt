# SPDX-FileCopyrightText: 2024 Arntzen Software AS
# SPDX-FileContributor: Hans-Kristian Arntzen
# SPDX-FileContributor: Runar Heyer
# SPDX-License-Identifier: LGPL-3.0+

add_library(parallel-gs-dump STATIC
        gs_dump_parser.cpp gs_dump_parser.hpp
        gs_dump_generator.cpp gs_dump_generator.hpp)
target_compile_options(parallel-gs-dump PRIVATE ${PARALLEL_GS_CXX_FLAGS})
target_link_libraries(parallel-gs-dump PUBLIC parallel-gs granite-vulkan granite-math)
target_include_directories(parallel-gs-dump PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
