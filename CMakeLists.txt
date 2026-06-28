# cmake_minimum_required(VERSION 3.16)
# project(llm_tracer CXX C)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
# set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)

# if(APPLE)
#     set(GGML_METAL ON CACHE BOOL "" FORCE)
# endif()

# add_subdirectory(extern/llama.cpp)

# set(TRACER_INCLUDES
#     ${CMAKE_SOURCE_DIR}/include
#     ${CMAKE_SOURCE_DIR}/extern/llama.cpp/include
#     ${CMAKE_SOURCE_DIR}/extern/llama.cpp/ggml/include
# )

# add_library(tracer_core STATIC
#     src/core/layer_event.cpp
#     src/core/attention_store.cpp
# )
# target_include_directories(tracer_core PUBLIC ${TRACER_INCLUDES})

# add_executable(llm_tracer src/main.cpp)
# target_include_directories(llm_tracer PRIVATE ${TRACER_INCLUDES})
# target_link_libraries(llm_tracer PRIVATE tracer_core llama)

# add_executable(test_hook_dump tests/test_hook_dump.cpp)
# target_include_directories(test_hook_dump PRIVATE ${TRACER_INCLUDES})
# target_link_libraries(test_hook_dump PRIVATE tracer_core llama)

# add_executable(test_ring_buffer tests/test_ring_buffer.cpp)
# target_include_directories(test_ring_buffer PRIVATE ${TRACER_INCLUDES})
# target_link_libraries(test_ring_buffer PRIVATE tracer_core)

cmake_minimum_required(VERSION 3.16)
project(llm_tracer CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── llama.cpp build flags ────────────────────────────────────────────────────
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)

if(APPLE)
    message(STATUS "[llm-tracer] Apple Silicon detected — enabling Metal backend")
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

# set(GGML_CUDA ON CACHE BOOL "" FORCE)   # Uncomment for NVIDIA GPU

add_subdirectory(extern/llama.cpp)

# ── Shared include paths ──────────────────────────────────────────────────────
set(TRACER_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/extern/llama.cpp/ggml/include
)

# ── tracer_core static library ────────────────────────────────────────────────
# Grows with each phase. Every binary links against this.
# Phase 1: LayerEvent helpers + AttentionStore
# Phase 2: + HookManager + AnomalyDetector
# Phase 3: + TopologyParser
add_library(tracer_core STATIC
    src/core/layer_event.cpp       # Phase 1
    src/core/attention_store.cpp   # Phase 1
    # src/hook/hook_manager.cpp    # Phase 2 — uncomment
    # src/hook/anomaly_detector.cpp
    # src/topo/topology_parser.cpp # Phase 3 — uncomment
)
target_include_directories(tracer_core PUBLIC ${TRACER_INCLUDES})

# ── Main executable ───────────────────────────────────────────────────────────
add_executable(llm_tracer src/main.cpp)
target_include_directories(llm_tracer PRIVATE ${TRACER_INCLUDES})
target_link_libraries(llm_tracer PRIVATE tracer_core llama)

# ── Phase 0 test ─────────────────────────────────────────────────────────────
add_executable(test_hook_dump tests/test_hook_dump.cpp)
target_include_directories(test_hook_dump PRIVATE ${TRACER_INCLUDES})
target_link_libraries(test_hook_dump PRIVATE tracer_core llama)

# ── Phase 1 test ─────────────────────────────────────────────────────────────
add_executable(test_ring_buffer tests/test_ring_buffer.cpp)
target_include_directories(test_ring_buffer PRIVATE ${TRACER_INCLUDES})
target_link_libraries(test_ring_buffer PRIVATE tracer_core)   # no llama needed

# ── Future test targets — uncomment phase by phase ───────────────────────────
# Phase 3:
# add_executable(test_topology_print tests/test_topology_print.cpp)
# target_include_directories(test_topology_print PRIVATE ${TRACER_INCLUDES})
# target_link_libraries(test_topology_print PRIVATE tracer_core llama)

# Phase 4:
# find_package(ftxui QUIET)   # or use FetchContent below when Phase 4 starts
# add_executable(test_tui_mock tests/test_tui_mock.cpp)
# target_include_directories(test_tui_mock PRIVATE ${TRACER_INCLUDES})
# target_link_libraries(test_tui_mock PRIVATE tracer_core llama
#     ftxui::screen ftxui::dom ftxui::component)

# Phase 6:
# add_executable(test_hook_smoke tests/test_hook_smoke.cpp)
# target_include_directories(test_hook_smoke PRIVATE ${TRACER_INCLUDES})
# target_link_libraries(test_hook_smoke PRIVATE tracer_core llama)