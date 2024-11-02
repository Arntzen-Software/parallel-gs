# SPDX-FileCopyrightText: 2024 Arntzen Software AS
# SPDX-FileContributor: Hans-Kristian Arntzen
# SPDX-FileContributor: Runar Heyer
# SPDX-License-Identifier: LGPL-3.0+

if (POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()

if (NOT CMAKE_MSVC_RUNTIME_LIBRARY)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 99)
project(parallel-gs LANGUAGES CXX C)

if (CMAKE_COMPILER_IS_GNUCXX OR (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang"))
    set(PARALLEL_GS_CXX_FLAGS -Wshadow -Wall -Wextra -Wno-comment -Wno-missing-field-initializers -Wno-empty-body)
    if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
        set(PARALLEL_GS_CXX_FLAGS ${PARALLEL_GS_CXX_FLAGS} -Wno-backslash-newline-escape)
    endif()
    if (NOT (${CMAKE_BUILD_TYPE} MATCHES "Release"))
        message("Enabling frame pointer for profiling/debug.")
        set(PARALLEL_GS_CXX_FLAGS ${PARALLEL_GS_CXX_FLAGS} -fno-omit-frame-pointer)
    endif()
elseif (MSVC)
    set(PARALLEL_GS_CXX_FLAGS /D_CRT_SECURE_NO_WARNINGS /wd4267 /wd4244 /wd4309 /wd4005 /MP /DNOMINMAX /fp:precise)
endif()

option(PARALLEL_GS_STANDALONE "Build parallel-gs as a library." OFF)
option(PARALLEL_GS_DEBUG "Add extra debug logging." OFF)

set(GRANITE_FAST_MATH OFF CACHE BOOL "Fast math" FORCE)
set(GRANITE_VULKAN_FOSSILIZE OFF CACHE BOOL "" FORCE)

if (PARALLEL_GS_STANDALONE)
    set(GRANITE_RENDERER OFF CACHE BOOL "" FORCE)
    set(GRANITE_VULKAN_SPIRV_CROSS OFF CACHE BOOL "" FORCE)
    set(GRANITE_VULKAN_SHADER_MANAGER_RUNTIME_COMPILER OFF CACHE BOOL "" FORCE)
    set(GRANITE_VULKAN_SYSTEM_HANDLES OFF CACHE BOOL "" FORCE)
    set(GRANITE_SHADER_COMPILER_OPTIMIZE OFF CACHE BOOL "" FORCE)
    set(GRANITE_SHIPPING ON CACHE BOOL "" FORCE)
    set(GRANITE_PLATFORM "null" CACHE STRING "" FORCE)
else()
    set(GRANITE_RENDERER ON CACHE BOOL "" FORCE)
    set(GRANITE_VULKAN_SPIRV_CROSS ON CACHE BOOL "" FORCE)
    set(GRANITE_VULKAN_SHADER_MANAGER_RUNTIME_COMPILER ON CACHE BOOL "" FORCE)
    set(GRANITE_VULKAN_SYSTEM_HANDLES ON CACHE BOOL "" FORCE)
    set(GRANITE_SHADER_COMPILER_OPTIMIZE ON CACHE BOOL "" FORCE)
    set(GRANITE_SHIPPING OFF CACHE BOOL "" FORCE)
    set(GRANITE_PLATFORM "SDL" CACHE STRING "" FORCE)
endif()

add_subdirectory(Granite EXCLUDE_FROM_ALL)

add_subdirectory(gs)
add_subdirectory(dump)
add_subdirectory(sandbox)
if (NOT PARALLEL_GS_STANDALONE)
    add_subdirectory(tools)
endif()
