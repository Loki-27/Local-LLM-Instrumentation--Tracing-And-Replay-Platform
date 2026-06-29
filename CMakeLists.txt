cmake_minimum_required(VERSION 3.16)
project(llm_tracer CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)

if(APPLE)
    message(STATUS "[llm-tracer] Apple Silicon — enabling Metal")
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(extern/llama.cpp)

# ftxui as local submodule
add_subdirectory(extern/ftxui)

set(TRACER_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/extern/llama.cpp/ggml/include
)

# tracer_core: data structures + hook + topology (no ftxui)
add_library(tracer_core STATIC
    src/core/layer_event.cpp
    src/core/attention_store.cpp
    src/hook/hook_manager.cpp
    src/hook/anomaly_detector.cpp
    src/topo/topology_parser.cpp
)
target_include_directories(tracer_core PUBLIC ${TRACER_INCLUDES})
target_link_libraries(tracer_core PUBLIC llama)

# tracer_tui: TUI layer (depends on tracer_core + ftxui)
add_library(tracer_tui STATIC
    src/tui/tui_app.cpp
)
target_include_directories(tracer_tui PUBLIC ${TRACER_INCLUDES})
target_link_libraries(tracer_tui PUBLIC tracer_core
    ftxui::screen ftxui::dom ftxui::component)

# Main binary — full integration (Phase 5)
add_executable(llm_tracer src/main.cpp)
target_include_directories(llm_tracer PRIVATE ${TRACER_INCLUDES})
target_link_libraries(llm_tracer PRIVATE tracer_tui)

# Phase 0: baseline inference (no TUI)
add_executable(test_hook_dump tests/test_hook_dump.cpp)
target_include_directories(test_hook_dump PRIVATE ${TRACER_INCLUDES})
target_link_libraries(test_hook_dump PRIVATE tracer_core)

# Phase 1: data structure tests
add_executable(test_ring_buffer tests/test_ring_buffer.cpp)
target_include_directories(test_ring_buffer PRIVATE ${TRACER_INCLUDES})
target_link_libraries(test_ring_buffer PRIVATE tracer_core)

# Phase 3: topology test
add_executable(test_topology_print tests/test_topology_print.cpp)
target_include_directories(test_topology_print PRIVATE ${TRACER_INCLUDES})
target_link_libraries(test_topology_print PRIVATE tracer_core)

# Phase 4: TUI mock (no model needed)
add_executable(test_tui_mock tests/test_tui_mock.cpp)
target_include_directories(test_tui_mock PRIVATE ${TRACER_INCLUDES})
target_link_libraries(test_tui_mock PRIVATE tracer_tui)

# Phase 6: regression test (needs model)
add_executable(test_hook_smoke tests/test_hook_smoke.cpp)
target_include_directories(test_hook_smoke PRIVATE ${TRACER_INCLUDES})
target_link_libraries(test_hook_smoke PRIVATE tracer_core)
