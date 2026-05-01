# SPDX-FileCopyrightText: 2026 Arntzen Software AS
# SPDX-FileContributor: Hans-Kristian Arntzen
# SPDX-License-Identifier: LGPL-3.0+

add_library(analog-video STATIC analog_video.hpp analog_video.cpp)
target_compile_options(analog-video PRIVATE ${PARALLEL_GS_CXX_FLAGS})
target_link_libraries(analog-video PUBLIC granite-vulkan granite-math)
target_include_directories(analog-video PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
