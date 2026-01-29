# NPM Backend for ggml
# Ceva NeuPro-M accelerator support

message(STATUS "Including NPM backend (Ceva NeuPro-M)")

# =============================================================================
# Device type selection
# =============================================================================

set(NPM_DEVICE_TYPE "mock" CACHE STRING "NPM device implementation (mock, emulator, hardware)")
set_property(CACHE NPM_DEVICE_TYPE PROPERTY STRINGS mock emulator hardware)

message(STATUS "NPM device type: ${NPM_DEVICE_TYPE}")

# =============================================================================
# Source files
# =============================================================================

set(NPM_SOURCES
    ggml-npm.cpp
)

# Select device implementation based on NPM_DEVICE_TYPE
if(NPM_DEVICE_TYPE STREQUAL "mock")
    message(STATUS "NPM: Using mock device (in-process CPU)")
    list(APPEND NPM_SOURCES npm-device/npm-device-mock.cpp)
    set(NPM_DEVICE_DEFINE "NPM_DEVICE_MOCK")

elseif(NPM_DEVICE_TYPE STREQUAL "emulator")
    message(STATUS "NPM: Using emulator device (IPC to npm-emulator process)")
    list(APPEND NPM_SOURCES npm-device/npm-device-emulator.cpp)
    set(NPM_NEED_NETWORK ON)
    set(NPM_NEED_SHM ON)
    set(NPM_DEVICE_DEFINE "NPM_DEVICE_EMULATOR")

elseif(NPM_DEVICE_TYPE STREQUAL "hardware")
    if(NOT NPM_SDK_PATH)
        message(FATAL_ERROR "NPM_SDK_PATH must be set for hardware device type")
    endif()
    message(STATUS "NPM: Using hardware device (real NPM)")
    list(APPEND NPM_SOURCES npm-device/npm-device-hardware.cpp)
    set(NPM_USE_SDK ON)
    set(NPM_DEVICE_DEFINE "NPM_DEVICE_HARDWARE")

else()
    message(FATAL_ERROR "Unknown NPM_DEVICE_TYPE: ${NPM_DEVICE_TYPE}")
endif()

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
