# NPM Hardware Emulator
# Standalone process that emulates NPM hardware

add_executable(npm-emulator
    main.cpp
    npm-emu-server.cpp
    npm-trace.cpp
    npm-memory-model.cpp
    npm-dma-model.cpp
)

target_include_directories(npm-emulator PRIVATE
    ${PROJECT_SOURCE_DIR}/common/npm-protocol
)

# Link with rt for shared memory (shm_open)
if(UNIX AND NOT APPLE)
    target_link_libraries(npm-emulator PRIVATE rt)
endif()

# Install
install(TARGETS npm-emulator RUNTIME DESTINATION bin)
