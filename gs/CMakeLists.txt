# SPDX-FileCopyrightText: 2024 Arntzen Software AS
# SPDX-FileContributor: Hans-Kristian Arntzen
# SPDX-FileContributor: Runar Heyer
# SPDX-License-Identifier: LGPL-3.0+

add_library(parallel-gs STATIC
        gs_registers.hpp
        gs_registers_debug.hpp
        gs_register_addr.hpp
        gs_interface.cpp gs_interface.hpp
        gs_util.cpp gs_util.hpp
        shaders/slangmosh_iface.hpp shaders/slangmosh.hpp
        page_tracker.cpp page_tracker.hpp
        gs_renderer.cpp gs_renderer.hpp)
target_compile_options(parallel-gs PRIVATE ${PARALLEL_GS_CXX_FLAGS})
target_link_libraries(parallel-gs PUBLIC granite-vulkan granite-math)
target_include_directories(parallel-gs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (PARALLEL_GS_DEBUG)
    target_compile_definitions(parallel-gs PRIVATE PARALLEL_GS_DEBUG=1)
endif()
