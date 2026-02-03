// NPM Mock Device Implementation
//
// Phase 1: In-process CPU execution for validation.
// This implementation executes all operations on the CPU, validating
// the device abstraction layer without requiring external processes or hardware.

#include "npm-device.h"

#ifdef GGML_TYPE_F32
// Already defined (standalone test)
#else
#include "ggml.h"
#endif

#include <cstdlib>
#include <cstring>
#include <unordered_map>

// Define GGML_TYPE_F32 if not already defined (for standalone testing)
#ifndef GGML_TYPE_F32
#define GGML_TYPE_F32 0
#endif

// =============================================================================
// Buffer registration entry
// =============================================================================

struct npm_buffer_entry {
    void * ptr;
    size_t size;
};

// =============================================================================
// Mock device context
// =============================================================================

struct npm_device_mock_context {
    enum npm_sku sku;
    int num_engines;
    size_t l1_size;
    size_t l2_size;

    // Buffer registry: handle -> buffer info
    std::unordered_map<uint64_t, npm_buffer_entry> buffers;
    uint64_t next_handle;
};

// =============================================================================
// Lifecycle
// =============================================================================

static int npm_device_mock_init(struct npm_device * dev, int device_id) {
    (void)device_id;

    struct npm_device_mock_context * ctx = (struct npm_device_mock_context *)dev->context;

    // Mock configuration: simulate NPM8K
    ctx->sku = NPM_SKU_MOCK;
    ctx->num_engines = 1;
    ctx->l1_size = 1024 * 1024;      // 1MB L1
    ctx->l2_size = 8 * 1024 * 1024;  // 8MB L2
    ctx->next_handle = 1;            // Handle 0 is reserved/invalid

    return 0;
}

static void npm_device_mock_shutdown(struct npm_device * dev) {
    struct npm_device_mock_context * ctx = (struct npm_device_mock_context *)dev->context;
    ctx->buffers.clear();
}

// =============================================================================
// Device info
// =============================================================================

static enum npm_sku npm_device_mock_get_sku(struct npm_device * dev) {
    struct npm_device_mock_context * ctx = (struct npm_device_mock_context *)dev->context;
    return ctx->sku;
}

static int npm_device_mock_get_num_engines(struct npm_device * dev) {
    struct npm_device_mock_context * ctx = (struct npm_device_mock_context *)dev->context;
    return ctx->num_engines;
}

static size_t npm_device_mock_get_l1_size(struct npm_device * dev) {
    struct npm_device_mock_context * ctx = (struct npm_device_mock_context *)dev->context;
    return ctx->l1_size;
}

static size_t npm_device_mock_get_l2_size(struct npm_device * dev) {
    struct npm_device_mock_context * ctx = (struct npm_device_mock_context *)dev->context;
    return ctx->l2_size;
}

// =============================================================================
// Memory management
// =============================================================================

static int npm_device_mock_register_buffer(struct npm_device * dev, void * ptr, size_t size, uint64_t * handle) {
    struct npm_device_mock_context * ctx = (struct npm_device_mock_context *)dev->context;

    if (!ptr || size == 0 || !handle) {
        return -1;
    }

    // Assign a new handle
    uint64_t h = ctx->next_handle++;

    // Register the buffer
    ctx->buffers[h] = { ptr, size };

    *handle = h;
    return 0;
}

static void npm_device_mock_unregister_buffer(struct npm_device * dev, uint64_t handle) {
    struct npm_device_mock_context * ctx = (struct npm_device_mock_context *)dev->context;
    ctx->buffers.erase(handle);
}

static int npm_device_mock_update_buffer(struct npm_device * dev, uint64_t handle, void * ptr, size_t size) {
    struct npm_device_mock_context * ctx = (struct npm_device_mock_context *)dev->context;

    auto it = ctx->buffers.find(handle);
    if (it == ctx->buffers.end()) {
        return -1;  // Handle not found
    }

    // For mock device, just update the pointer and size
    it->second.ptr = ptr;
    it->second.size = size;
    return 0;
}

// =============================================================================
// Helper: resolve handle to pointer
// =============================================================================

static void * npm_device_mock_resolve_handle(struct npm_device_mock_context * ctx, uint64_t handle, size_t offset) {
    auto it = ctx->buffers.find(handle);
    if (it == ctx->buffers.end()) {
        return nullptr;
    }
    return (char *)it->second.ptr + offset;
}

// =============================================================================
// Compute operations
// =============================================================================

static int npm_device_mock_matmul(struct npm_device * dev, const struct npm_matmul_params * params) {
    struct npm_device_mock_context * ctx = (struct npm_device_mock_context *)dev->context;

    // Phase 1: Only support FP32
    if (params->type_a != GGML_TYPE_F32 ||
        params->type_b != GGML_TYPE_F32 ||
        params->type_c != GGML_TYPE_F32) {
        return -1;
    }

    // Resolve buffer handles to pointers
    const float * A = (const float *)npm_device_mock_resolve_handle(ctx, params->a_handle, params->a_offset);
    const float * B = (const float *)npm_device_mock_resolve_handle(ctx, params->b_handle, params->b_offset);
    float * C = (float *)npm_device_mock_resolve_handle(ctx, params->c_handle, params->c_offset);

    if (!A || !B || !C) {
        return -2;  // Invalid buffer handle
    }

    const int64_t M = params->M;
    const int64_t N = params->N;
    const int64_t K = params->K;

    // Matrix layout for MUL_MAT:
    // C = A * B^T where:
    // - A has shape (K, M) stored row-major: A[m,k] = A[m * lda + k]
    // - B has shape (K, N) stored row-major: B[n,k] = B[n * ldb + k]
    // - C has shape (N, M) stored row-major: C[m,n] = C[m * ldc + n]
    //
    // Computation: C[m,n] = sum_k(A[m,k] * B[n,k])

    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                sum += A[m * params->lda + k] * B[n * params->ldb + k];
            }
            C[m * params->ldc + n] = sum;
        }
    }

    return 0;
}

// =============================================================================
// Synchronization
// =============================================================================

static int npm_device_mock_sync(struct npm_device * dev) {
    (void)dev;
    return 0;  // Everything is synchronous in mock
}

static int npm_device_mock_fence_create(struct npm_device * dev, struct npm_fence ** fence) {
    (void)dev;
    *fence = (struct npm_fence *)1;  // Dummy non-NULL pointer
    return 0;
}

static void npm_device_mock_fence_destroy(struct npm_device * dev, struct npm_fence * fence) {
    (void)dev;
    (void)fence;
}

static int npm_device_mock_fence_wait(struct npm_device * dev, struct npm_fence * fence, uint64_t timeout_ns) {
    (void)dev;
    (void)fence;
    (void)timeout_ns;
    return 0;  // Instant completion for mock
}

// =============================================================================
// Factory function
// =============================================================================

struct npm_device * npm_device_mock_create(void) {
    struct npm_device * dev = new npm_device;
    if (!dev) {
        return nullptr;
    }

    struct npm_device_mock_context * ctx = new npm_device_mock_context;
    if (!ctx) {
        delete dev;
        return nullptr;
    }

    dev->context = ctx;

    // Set up operations
    dev->ops.init = npm_device_mock_init;
    dev->ops.shutdown = npm_device_mock_shutdown;
    dev->ops.get_sku = npm_device_mock_get_sku;
    dev->ops.get_num_engines = npm_device_mock_get_num_engines;
    dev->ops.get_l1_size = npm_device_mock_get_l1_size;
    dev->ops.get_l2_size = npm_device_mock_get_l2_size;
    dev->ops.register_buffer = npm_device_mock_register_buffer;
    dev->ops.unregister_buffer = npm_device_mock_unregister_buffer;
    dev->ops.update_buffer = npm_device_mock_update_buffer;
    dev->ops.matmul = npm_device_mock_matmul;
    dev->ops.sync = npm_device_mock_sync;
    dev->ops.fence_create = npm_device_mock_fence_create;
    dev->ops.fence_destroy = npm_device_mock_fence_destroy;
    dev->ops.fence_wait = npm_device_mock_fence_wait;

    // Initialize
    if (dev->ops.init(dev, 0) != 0) {
        delete ctx;
        delete dev;
        return nullptr;
    }

    return dev;
}

// Note: npm_sku_name() and npm_device_destroy() are defined in npm-device-common.cpp
