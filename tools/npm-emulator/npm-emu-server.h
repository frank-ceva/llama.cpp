#pragma once

// NPM Emulator Server
//
// IPC server that accepts connections from npm-device-emulator clients
// and executes NPM operations on shared memory.

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <unordered_map>
#include <memory>

#include "npm-types.h"
#include "npm-emu-protocol.h"
#include "npm-memory-model.h"
#include "npm-dma-model.h"
#include "npm-trace.h"

// =============================================================================
// Buffer entry
// =============================================================================

struct npm_emu_buffer {
    size_t   shm_offset;    // Offset within shared memory
    size_t   size;          // Size of the buffer
    uint32_t flags;         // Allocation flags
};

// =============================================================================
// Server configuration
// =============================================================================

struct npm_emu_config {
    const char * socket_path;   // Unix socket path
    enum npm_sku sku;           // Device SKU to emulate
    size_t       l2_size;       // L2 cache size (0 = use default for SKU)
    bool         tiling_enabled; // Enable tiled matmul execution
    bool         timing_enabled; // Enable timing/cycle simulation
    bool         verbose;        // Verbose logging

    // Tracing options
    uint32_t     trace_categories; // Bitmask of npm_trace_category
    const char * trace_file;       // Trace output file (NULL = stdout)
};

// =============================================================================
// Server state
// =============================================================================

struct npm_emu_server {
    // Configuration
    npm_emu_config config;

    // Socket
    int listen_fd;
    int client_fd;

    // Shared memory (attached from client)
    void * shm_base;
    size_t shm_size;

    // Buffer registry: handle -> buffer info
    std::unordered_map<uint64_t, npm_emu_buffer> buffers;
    uint64_t next_handle;

    // Fence registry
    uint64_t next_fence_id;

    // Device info (derived from SKU)
    int      num_engines;
    size_t   l1_size;
    size_t   l2_size;

    // Statistics
    uint64_t total_matmul_ops;
    uint64_t total_bytes_transferred;

    // Hardware models (initialized when timing is enabled)
    std::unique_ptr<npm_memory_hierarchy> mem_hierarchy;
    std::unique_ptr<npm_dma_model> dma_model;

    // Tracing context
    npm_trace_ctx* trace_ctx;
};

// =============================================================================
// Server lifecycle
// =============================================================================

// Create and initialize server
npm_emu_server * npm_emu_server_create(const npm_emu_config * config);

// Destroy server
void npm_emu_server_destroy(npm_emu_server * server);

// Run the server (blocks until shutdown)
int npm_emu_server_run(npm_emu_server * server);

// Request shutdown (can be called from signal handler)
void npm_emu_server_shutdown(npm_emu_server * server);
