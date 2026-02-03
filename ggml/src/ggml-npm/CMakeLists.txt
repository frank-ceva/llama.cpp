# NPM Backend for ggml
# Ceva NeuPro-M accelerator support

message(STATUS "Including NPM backend (Ceva NeuPro-M)")

# =============================================================================
# Device type selection - NOW AT RUNTIME
# =============================================================================

# Compile all device implementations into the binary, selection happens at runtime
# via NPM_DEVICE environment variable (mock, emulator, hardware)
message(STATUS "NPM: Compiling all device types (runtime selection via NPM_DEVICE env var)")

# =============================================================================
# Source files - Include ALL device implementations
# =============================================================================

set(NPM_SOURCES
    ggml-npm.cpp
    npm-device/npm-device-common.cpp
    npm-device/npm-device-mock.cpp
    npm-device/npm-device-emulator.cpp
)

# Only include hardware device if SDK is available
if(NPM_SDK_PATH)
    list(APPEND NPM_SOURCES npm-device/npm-device-hardware.cpp)
    set(NPM_USE_SDK ON)
    message(STATUS "NPM: Hardware device will be included (SDK found at: ${NPM_SDK_PATH})")
else()
    message(STATUS "NPM: Hardware device not included (NPM_SDK_PATH not set)")
endif()

# Signal that runtime device selection is enabled
set(NPM_DEVICE_DEFINE "NPM_DEVICE_RUNTIME_SELECT")
set(NPM_NEED_NETWORK ON)
set(NPM_NEED_SHM ON)

# =============================================================================
# Create the backend library
# =============================================================================

ggml_add_backend_library(ggml-npm ${NPM_SOURCES})

# Include directories
# Note: CMAKE_SOURCE_DIR is llama.cpp root, PROJECT_SOURCE_DIR is ggml root
target_include_directories(ggml-npm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/npm-device
    ${CMAKE_SOURCE_DIR}/common/npm-protocol
)

# Link to ggml-cpu for buffer type functions
target_link_libraries(ggml-npm PRIVATE ggml-cpu)

# Set compile definition for device type selection
target_compile_definitions(ggml-npm PRIVATE ${NPM_DEVICE_DEFINE})

# =============================================================================
# Platform-specific dependencies
# =============================================================================

if(NPM_NEED_NETWORK)
    if(WIN32)
        target_link_libraries(ggml-npm PRIVATE ws2_32)
    endif()
endif()

if(NPM_NEED_SHM)
    if(UNIX AND NOT APPLE)
        target_link_libraries(ggml-npm PRIVATE rt)
    endif()
endif()

# =============================================================================
# Hardware SDK support (Phase 2)
# =============================================================================

if(NPM_USE_SDK AND NPM_SDK_PATH)
    message(STATUS "NPM SDK found at: ${NPM_SDK_PATH}")

    target_include_directories(ggml-npm PRIVATE ${NPM_SDK_PATH}/include)
    target_link_directories(ggml-npm PRIVATE ${NPM_SDK_PATH}/lib)
    target_link_libraries(ggml-npm PRIVATE npm_runtime)

    target_compile_definitions(ggml-npm PRIVATE NPM_HW_ENABLED)
endif()
