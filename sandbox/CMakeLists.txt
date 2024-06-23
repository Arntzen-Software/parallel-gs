# SPDX-FileCopyrightText: 2024 Arntzen Software AS
# SPDX-FileContributor: Hans-Kristian Arntzen
# SPDX-FileContributor: Runar Heyer
# SPDX-License-Identifier: LGPL-3.0+

add_granite_offline_tool(parallel-gs-sandbox gs_test.cpp)
target_link_libraries(parallel-gs-sandbox PRIVATE parallel-gs parallel-gs-dump)

