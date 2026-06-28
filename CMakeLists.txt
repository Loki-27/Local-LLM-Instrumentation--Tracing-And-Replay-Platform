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

set(TRACER_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/extern/llama.cpp/ggml/include
)

# tracer_core grows each phase — add new .cpp files here
add_library(tracer_core STATIC
    src/core/layer_event.cpp        # Phase 1
    src/core/attention_store.cpp    # Phase 1
    src/hook/hook_manager.cpp       # Phase 2
    src/hook/anomaly_detector.cpp   # Phase 2
    # src/topo/topology_parser.cpp  # Phase 3 — uncomment
)
target_include_directories(tracer_core PUBLIC ${TRACER_INCLUDES})
target_link_libraries(tracer_core PUBLIC llama)

add_executable(llm_tracer src/main.cpp)
target_include_directories(llm_tracer PRIVATE ${TRACER_INCLUDES})
target_link_libraries(llm_tracer PRIVATE tracer_core)

add_executable(test_hook_dump tests/test_hook_dump.cpp)
target_include_directories(test_hook_dump PRIVATE ${TRACER_INCLUDES})
target_link_libraries(test_hook_dump PRIVATE tracer_core)

add_executable(test_ring_buffer tests/test_ring_buffer.cpp)
target_include_directories(test_ring_buffer PRIVATE ${TRACER_INCLUDES})
target_link_libraries(test_ring_buffer PRIVATE tracer_core)

# Phase 3 (uncomment when ready):
# add_executable(test_topology_print tests/test_topology_print.cpp)
# target_include_directories(test_topology_print PRIVATE ${TRACER_INCLUDES})
# target_link_libraries(test_topology_print PRIVATE tracer_core)

# Phase 4 (uncomment when ready):
# include(FetchContent)
# FetchContent_Declare(ftxui GIT_REPOSITORY https://github.com/ArthurSonzogni/ftxui GIT_TAG v0.11.1)
# FetchContent_MakeAvailable(ftxui)
# add_executable(test_tui_mock tests/test_tui_mock.cpp)
# target_include_directories(test_tui_mock PRIVATE ${TRACER_INCLUDES})
# target_link_libraries(test_tui_mock PRIVATE tracer_core ftxui::screen ftxui::dom ftxui::component)

# Phase 6 (uncomment when ready):
# add_executable(test_hook_smoke tests/test_hook_smoke.cpp)
# target_include_directories(test_hook_smoke PRIVATE ${TRACER_INCLUDES})
# target_link_libraries(test_hook_smoke PRIVATE tracer_core)
