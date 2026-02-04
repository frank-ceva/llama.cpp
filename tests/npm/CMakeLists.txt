# NPM Backend Tests

# test-npm-backend: Full ggml backend integration tests
add_executable(test-npm-backend test-npm-backend.cpp)
target_link_libraries(test-npm-backend PRIVATE ggml-npm ggml-base ggml-cpu)
target_include_directories(test-npm-backend PRIVATE
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-npm
)

add_test(NAME test-npm-backend COMMAND test-npm-backend)
set_property(TEST test-npm-backend PROPERTY LABELS "npm")

# test-npm-device: Device abstraction tests (tests the device implementations directly)
# This test compiles the mock device directly to be independent of ggml-npm's device type
add_executable(test-npm-device
    test-npm-device.cpp
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-npm/npm-device/npm-device-common.cpp
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-npm/npm-device/npm-device-mock.cpp
)
target_link_libraries(test-npm-device PRIVATE ggml-base)
target_include_directories(test-npm-device PRIVATE
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-npm
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-npm/npm-device
    ${CMAKE_SOURCE_DIR}/common/npm-protocol
)

add_test(NAME test-npm-device COMMAND test-npm-device)
set_property(TEST test-npm-device PROPERTY LABELS "npm")

# test-npm-emulator: Emulator integration tests (tests IPC to npm-emulator process)
# This test compiles the emulator device directly for testing
add_executable(test-npm-emulator
    test-npm-emulator.cpp
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-npm/npm-device/npm-device-common.cpp
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-npm/npm-device/npm-device-emulator.cpp
)
target_link_libraries(test-npm-emulator PRIVATE ggml-base)
target_include_directories(test-npm-emulator PRIVATE
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-npm
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-npm/npm-device
    ${CMAKE_SOURCE_DIR}/common/npm-protocol
)
# Link with rt for shared memory (shm_open)
if(UNIX AND NOT APPLE)
    target_link_libraries(test-npm-emulator PRIVATE rt)
endif()

add_test(NAME test-npm-emulator COMMAND test-npm-emulator --managed)
set_property(TEST test-npm-emulator PROPERTY LABELS "npm" "integration")

# test-npm-inference: Model inference integration tests
# Tests llama.cpp inference with NPM backend (mock or emulator device)
add_executable(test-npm-inference test-npm-inference.cpp)
target_link_libraries(test-npm-inference PRIVATE
    llama
    ggml
    common
)
target_include_directories(test-npm-inference PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/common
)

add_test(NAME test-npm-inference COMMAND test-npm-inference)
set_property(TEST test-npm-inference PROPERTY LABELS "npm" "integration" "model")
