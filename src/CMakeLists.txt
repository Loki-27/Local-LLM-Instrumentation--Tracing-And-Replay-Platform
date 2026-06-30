cmake_minimum_required(VERSION 3.16)
project(llm_tracer CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)    

# ── llama.cpp build flags ────────────────────────────────────────────────────
# Turn off everything we don't need so the build stays fast
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)

# Apple Silicon: enable Metal backend automatically
if(APPLE)
    message(STATUS "[llm-tracer] Apple Silicon detected — enabling Metal backend")
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

# set(GGML_CUDA ON CACHE BOOL "" FORCE)

add_subdirectory(extern/llama.cpp)

# ── Shared include paths used by every target ────────────────────────────────
set(TRACER_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/extern/llama.cpp/ggml/include
)

# ── Main executable ──────────────────────────────────────────────────────────
add_executable(llm_tracer
    src/main.cpp
)
target_include_directories(llm_tracer PRIVATE ${TRACER_INCLUDES})
target_link_libraries(llm_tracer PRIVATE llama)

# ── Phase 0 test: baseline inference sanity check ────────────────────────────
# Extend this file in Phase 2 to dump tensor events
add_executable(test_hook_dump
    tests/test_hook_dump.cpp
)
target_include_directories(test_hook_dump PRIVATE ${TRACER_INCLUDES})
target_link_libraries(test_hook_dump PRIVATE llama)

# ── Future test targets — uncomment phase by phase ───────────────────────────
# Phase 1:
# add_executable(test_ring_buffer tests/test_ring_buffer.cpp)
# target_include_directories(test_ring_buffer PRIVATE ${TRACER_INCLUDES})
# target_link_libraries(test_ring_buffer PRIVATE llama)

# Phase 3:
# add_executable(test_topology_print tests/test_topology_print.cpp)
# target_include_directories(test_topology_print PRIVATE ${TRACER_INCLUDES})
# target_link_libraries(test_topology_print PRIVATE llama)

# Phase 4:
# add_executable(test_tui_mock tests/test_tui_mock.cpp)
# target_include_directories(test_tui_mock PRIVATE ${TRACER_INCLUDES})
# target_link_libraries(test_tui_mock PRIVATE llama ftxui::screen ftxui::dom ftxui::component)

# Phase 6:
# add_executable(test_hook_smoke tests/test_hook_smoke.cpp)
# target_include_directories(test_hook_smoke PRIVATE ${TRACER_INCLUDES})
# target_link_libraries(test_hook_smoke PRIVATE llama)
