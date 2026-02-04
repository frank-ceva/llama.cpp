#pragma once

// NPM Device Abstraction Layer
//
// This header defines the abstract interface for NPM device implementations.
// Multiple implementations can exist:
//   - npm-device-mock.cpp     : In-process CPU execution (Phase 1)
//   - npm-device-emulator.cpp : IPC to separate emulator process (Phase 1.5)
//   - npm-device-hardware.cpp : Real NPM hardware (Phase 2)

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Forward declarations
// =============================================================================

struct npm_device;
struct npm_fence;

// =============================================================================
// NPM SKU types
// =============================================================================

enum npm_sku {
    NPM_SKU_4K = 0,   // 1 engine, 16K INT4 MACs
    NPM_SKU_8K,       // 1 engine, 32K INT4 MACs
    NPM_SKU_16K,      // 2 engines, 64K INT4 MACs
    NPM_SKU_32K,      // 4 engines, 128K INT4 MACs
    NPM_SKU_64K,      // 8 engines, 256K INT4 MACs
    NPM_SKU_MOCK,     // Mock implementation (no real hardware)
    NPM_SKU_EMULATOR, // Emulator implementation
};

// =============================================================================
// Memory allocation flags
// =============================================================================

enum npm_alloc_flags {
    NPM_ALLOC_DEFAULT = 0,
    NPM_ALLOC_L2      = 1 << 0,  // Prefer L2 cache placement
    NPM_ALLOC_PINNED  = 1 << 1,  // Pinned host memory for DMA
};

// =============================================================================
// MatMul parameters
// =============================================================================

struct npm_matmul_params {
    // Buffer handles (returned by register_buffer)
    uint64_t a_handle;      // Input matrix A handle
    uint64_t b_handle;      // Input matrix B (weights) handle
    uint64_t c_handle;      // Output matrix C handle

    // Offsets within buffers
    size_t a_offset;
    size_t b_offset;
    size_t c_offset;

    // Matrix dimensions
    int64_t M;              // Rows of A and C
    int64_t N;              // Cols of B and C
    int64_t K;              // Cols of A, Rows of B

    // Leading dimensions (row strides in elements)
    int64_t lda;            // Leading dimension of A
    int64_t ldb;            // Leading dimension of B
    int64_t ldc;            // Leading dimension of C

    // Data types (ggml type values)
    int type_a;             // ggml type of A
    int type_b;             // ggml type of B
    int type_c;             // ggml type of C
};

// =============================================================================
// Device operations interface
// =============================================================================

struct npm_device_ops {
    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    // Initialize the device
    // Returns 0 on success, negative error code on failure
    int (*init)(struct npm_device * dev, int device_id);

    // Shutdown the device and release resources
    void (*shutdown)(struct npm_device * dev);

    // -------------------------------------------------------------------------
    // Device info
    // -------------------------------------------------------------------------

    // Get device SKU
    enum npm_sku (*get_sku)(struct npm_device * dev);

    // Get number of compute engines
    int (*get_num_engines)(struct npm_device * dev);

    // Get L1 cache size per engine (bytes)
    size_t (*get_l1_size)(struct npm_device * dev);

    // Get L2 cache size (bytes)
    size_t (*get_l2_size)(struct npm_device * dev);

    // -------------------------------------------------------------------------
    // Memory management
    // CPU allocates memory, device registers it for access
    // -------------------------------------------------------------------------

    // Register a CPU-allocated buffer with the device
    // The device may need to map it (emulator) or validate it (hardware)
    // Returns 0 on success, populates handle for use in compute operations
    int (*register_buffer)(struct npm_device * dev, void * ptr, size_t size, uint64_t * handle);

    // Unregister a previously registered buffer
    void (*unregister_buffer)(struct npm_device * dev, uint64_t handle);

    // Update buffer data in device memory (for emulator: sync to shared memory)
    // Called when buffer content has changed and needs to be synced before compute
    // Returns 0 on success, negative error code on failure
    int (*update_buffer)(struct npm_device * dev, uint64_t handle, void * ptr, size_t size);

    // -------------------------------------------------------------------------
    // Compute operations
    // -------------------------------------------------------------------------

    // Execute matrix multiplication: C = A * B^T
    // All buffers must be registered first via register_buffer
    // Returns 0 on success, negative error code on failure
    int (*matmul)(struct npm_device * dev, const struct npm_matmul_params * params);

    // -------------------------------------------------------------------------
    // Synchronization
    // -------------------------------------------------------------------------

    // Wait for all pending operations to complete
    int (*sync)(struct npm_device * dev);

    // Create a fence for fine-grained synchronization
    int (*fence_create)(struct npm_device * dev, struct npm_fence ** fence);

    // Destroy a fence
    void (*fence_destroy)(struct npm_device * dev, struct npm_fence * fence);

    // Wait for a fence with timeout (nanoseconds, 0 = infinite)
    int (*fence_wait)(struct npm_device * dev, struct npm_fence * fence, uint64_t timeout_ns);
};

// =============================================================================
// Device context
// =============================================================================

struct npm_device {
    struct npm_device_ops ops;  // Operation function pointers
    void * context;             // Implementation-specific context
};

// =============================================================================
// Factory functions for each implementation
// =============================================================================

// Create mock device (Phase 1: in-process CPU execution)
struct npm_device * npm_device_mock_create(void);

// Create emulator device (Phase 1.5: IPC to separate emulator process)
// socket_path: Unix socket path to connect to emulator, or NULL for default
struct npm_device * npm_device_emulator_create(const char * socket_path);

// Create hardware device (Phase 2: real NPM hardware)
struct npm_device * npm_device_hardware_create(void);

// Destroy any device (calls shutdown and frees resources)
void npm_device_destroy(struct npm_device * dev);

// =============================================================================
// Utility functions
// =============================================================================

// Get human-readable name for SKU
const char * npm_sku_name(enum npm_sku sku);

#ifdef __cplusplus
}
#endif
