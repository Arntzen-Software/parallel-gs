# SPDX-FileCopyrightText: 2024 Arntzen Software AS
# SPDX-FileContributor: Hans-Kristian Arntzen
# SPDX-FileContributor: Runar Heyer
# SPDX-License-Identifier: LGPL-3.0+

add_granite_offline_tool(parallel-gs-replayer gs_dump_replayer.cpp)
target_link_libraries(parallel-gs-replayer PRIVATE parallel-gs parallel-gs-dump)

add_granite_application(parallel-gs-stream gs_stream_replayer.cpp)
target_link_libraries(parallel-gs-stream PRIVATE parallel-gs parallel-gs-dump)

add_granite_offline_tool(parallel-gs-repro gs_repro_replayer.cpp)
target_link_libraries(parallel-gs-repro PRIVATE parallel-gs parallel-gs-dump granite-stb granite-rapidjson)

granite_install_executable(parallel-gs-replayer)
granite_install_executable(parallel-gs-stream)
granite_install_executable(parallel-gs-repro)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/assets DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../Granite/assets DESTINATION ${CMAKE_INSTALL_BINDIR}/builtin)
