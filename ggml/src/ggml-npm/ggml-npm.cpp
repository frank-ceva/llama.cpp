#include "ggml-npm.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "npm-device/npm-device.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

// =============================================================================
// NPM Backend Context
// =============================================================================

struct ggml_backend_npm_context {
    struct npm_device * dev;
    int device_id;

    // Buffer registration cache: tensor data ptr -> device handle
    // Buffers are registered lazily on first use
    std::unordered_map<void *, uint64_t> buffer_handles;

    // Dequantization buffer for quantized matmul
    // Reused across calls to avoid repeated allocations
    std::vector<float> dequant_buffer;

    // Tracked dequant buffer handle for shared memory reuse
    // This prevents allocating new shared memory for each matmul
    uint64_t dequant_handle;
    size_t   dequant_handle_size;  // Size of allocated shared memory for dequant
};

// =============================================================================
// Quantization helpers
// =============================================================================

// Check if a type is quantized (not F32/F16)
static bool ggml_type_is_quantized(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_I8:
        case GGML_TYPE_I16:
        case GGML_TYPE_I32:
        case GGML_TYPE_I64:
        case GGML_TYPE_F64:
            return false;
        default:
            return true;
    }
}

// =============================================================================
// NPM Device Context
// =============================================================================

struct ggml_backend_npm_device_context {
    int device_id;
    struct npm_device * dev;  // Shared device instance for device info
};

// =============================================================================
// Forward declarations
// =============================================================================

static const char * ggml_backend_npm_get_name(ggml_backend_t backend);
static void ggml_backend_npm_free(ggml_backend_t backend);
static enum ggml_status ggml_backend_npm_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph);

// =============================================================================
// Buffer registration helper
// =============================================================================

// Get or register a buffer handle for a tensor's data
// skip_cache: if true, don't use cached handle (for temporary buffers like dequant_buffer)
// update_data: if true and handle is cached, update the shared memory with current data
static uint64_t ggml_backend_npm_get_buffer_handle_ex(
    struct ggml_backend_npm_context * ctx,
    void * ptr,
    size_t size,
    bool skip_cache,
    bool update_data = false
) {
    const char * npm_debug = getenv("NPM_DEBUG");
    if (npm_debug) {
        fprintf(stderr, "[NPM] get_buffer_handle_ex: ptr=%p size=%zu skip_cache=%d update_data=%d\n", ptr, size, skip_cache, update_data);
        fflush(stderr);
    }

    if (!skip_cache) {
        // Check if already registered
        auto it = ctx->buffer_handles.find(ptr);
        if (it != ctx->buffer_handles.end()) {
            if (npm_debug) fprintf(stderr, "[NPM] Found cached handle=%lu\n", (unsigned long)it->second);

            // If update_data is true, sync current data to shared memory
            if (update_data && ctx->dev->ops.update_buffer) {
                int result = ctx->dev->ops.update_buffer(ctx->dev, it->second, ptr, size);
                if (result != 0 && npm_debug) {
                    fprintf(stderr, "[NPM] Warning: update_buffer failed for cached handle=%lu\n", (unsigned long)it->second);
                }
            }
            return it->second;
        }
    }

    if (npm_debug) {
        fprintf(stderr, "[NPM] Calling register_buffer dev=%p register_buffer=%p\n",
                (void*)ctx->dev, (void*)ctx->dev->ops.register_buffer);
        fflush(stderr);
    }

    if (!ctx->dev->ops.register_buffer) {
        GGML_LOG_ERROR("%s: register_buffer function pointer is NULL!\n", __func__);
        return 0;
    }

    // Register new buffer
    uint64_t handle = 0;
    int result = ctx->dev->ops.register_buffer(ctx->dev, ptr, size, &handle);
    if (result != 0) {
        GGML_LOG_ERROR("%s: failed to register buffer %p (size %zu)\n", __func__, ptr, size);
        return 0;
    }

    if (npm_debug) fprintf(stderr, "[NPM] Registered handle=%lu\n", (unsigned long)handle);

    if (!skip_cache) {
        ctx->buffer_handles[ptr] = handle;
    }
    return handle;
}

// Convenience wrapper with caching enabled
// update_data: if true, update shared memory even for cached handles (needed for activations)
static uint64_t ggml_backend_npm_get_buffer_handle(
    struct ggml_backend_npm_context * ctx,
    void * ptr,
    size_t size,
    bool update_data = false
) {
    return ggml_backend_npm_get_buffer_handle_ex(ctx, ptr, size, false, update_data);
}

// Get or update the dequant buffer handle
// This reuses shared memory allocation when possible to avoid exhausting the bump allocator
static uint64_t ggml_backend_npm_get_dequant_handle(
    struct ggml_backend_npm_context * ctx,
    void * ptr,
    size_t size
) {
    const char * npm_debug = getenv("NPM_DEBUG");

    // If we already have a handle and it's large enough, just update the data
    if (ctx->dequant_handle != 0 && ctx->dequant_handle_size >= size) {
        // Copy new data to the existing shared memory region
        // The device's update_buffer function will handle this
        if (ctx->dev->ops.update_buffer) {
            int result = ctx->dev->ops.update_buffer(ctx->dev, ctx->dequant_handle, ptr, size);
            if (result == 0) {
                if (npm_debug) {
                    fprintf(stderr, "[NPM] Reused dequant handle=%lu (size=%zu, capacity=%zu)\n",
                            (unsigned long)ctx->dequant_handle, size, ctx->dequant_handle_size);
                }
                return ctx->dequant_handle;
            }
            // If update fails, fall through to re-register
        }
    }

    // Need to allocate new or larger buffer
    // First unregister old handle if it exists
    if (ctx->dequant_handle != 0) {
        ctx->dev->ops.unregister_buffer(ctx->dev, ctx->dequant_handle);
        ctx->dequant_handle = 0;
        ctx->dequant_handle_size = 0;
    }

    // Register new buffer
    uint64_t handle = 0;
    int result = ctx->dev->ops.register_buffer(ctx->dev, ptr, size, &handle);
    if (result != 0) {
        GGML_LOG_ERROR("%s: failed to register dequant buffer %p (size %zu)\n", __func__, ptr, size);
        return 0;
    }

    ctx->dequant_handle = handle;
    ctx->dequant_handle_size = size;

    if (npm_debug) {
        fprintf(stderr, "[NPM] New dequant handle=%lu (size=%zu)\n", (unsigned long)handle, size);
    }

    return handle;
}

// =============================================================================
// NPM MatMul Implementation
// =============================================================================
//
// The backend registers tensor buffers with the device and dispatches matmul
// operations using buffer handles. The device implementation handles the
// actual computation (mock: CPU, emulator: IPC, hardware: NPM).
// =============================================================================

static void ggml_backend_npm_mul_mat(struct ggml_backend_npm_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // weights (B)
    const struct ggml_tensor * src1 = dst->src[1];  // input (A)

    GGML_TENSOR_BINARY_OP_LOCALS

    // In ggml MUL_MAT:
    // dst = src1 * src0^T
    // src0 (weights): (ne00, ne01) = (K, N)
    // src1 (input):   (ne10, ne11) = (K, M)
    // dst (output):   (ne0, ne1)   = (N, M)

    // Verify dimensions
    GGML_ASSERT(ne0 == ne01);  // N
    GGML_ASSERT(ne1 == ne11);  // M
    GGML_ASSERT(ne00 == ne10); // K

    // src1 (activations) must be FP32
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // Check contiguity for src1 and dst
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb0 == sizeof(float));

    // Handle quantized src0 (weights) - dequantize to FP32
    void * src0_data = src0->data;
    size_t src0_bytes = ggml_nbytes(src0);
    bool src0_dequantized = false;

    // Debug output
    static int debug_count = 0;
    const char * npm_debug = getenv("NPM_DEBUG");

    if (src0->type != GGML_TYPE_F32) {
        // Get dequantization function
        const struct ggml_type_traits * traits = ggml_get_type_traits(src0->type);
        GGML_ASSERT(traits != nullptr && traits->to_float != nullptr);

        // Calculate FP32 size: ne00 * ne01 * ne02 * ne03 floats
        size_t n_elements = ggml_nelements(src0);
        size_t fp32_bytes = n_elements * sizeof(float);

        // Resize dequant buffer if needed
        if (ctx->dequant_buffer.size() < n_elements) {
            ctx->dequant_buffer.resize(n_elements);
        }

        // Dequantize the weights
        traits->to_float(src0->data, ctx->dequant_buffer.data(), n_elements);

        if (npm_debug && debug_count < 5) {
            fprintf(stderr, "[NPM] Dequantized src0: type=%s n_elem=%zu ne=(%lld,%lld,%lld,%lld)\n",
                    ggml_type_name(src0->type), n_elements,
                    (long long)ne00, (long long)ne01, (long long)ne02, (long long)ne03);
            debug_count++;
        }

        src0_data = ctx->dequant_buffer.data();
        src0_bytes = fp32_bytes;
        src0_dequantized = true;
    } else {
        // Check contiguity for FP32 src0
        GGML_ASSERT(nb00 == sizeof(float));

        if (npm_debug && debug_count < 5) {
            fprintf(stderr, "[NPM] FP32 src0: ne=(%lld,%lld,%lld,%lld)\n",
                    (long long)ne00, (long long)ne01, (long long)ne02, (long long)ne03);
            debug_count++;
        }
    }

    if (npm_debug) {
        fprintf(stderr, "[NPM] Registering buffers: A=%p(%zu) B=%p(%zu) C=%p(%zu) dequant=%d\n",
                src1->data, ggml_nbytes(src1), src0_data, src0_bytes, dst->data, ggml_nbytes(dst), src0_dequantized);
        fprintf(stderr, "[NPM] ctx=%p dev=%p\n", (void*)ctx, ctx ? (void*)ctx->dev : nullptr);
        fflush(stderr);
    }

    // Register buffers with device (or get existing handles)
    // For activations (src1), always update shared memory since data changes between inference steps
    // For dequantized data, use dedicated dequant handle to reuse shared memory
    uint64_t handle_a = ggml_backend_npm_get_buffer_handle(ctx, src1->data, ggml_nbytes(src1), true /* update_data */);
    if (npm_debug) fprintf(stderr, "[NPM] handle_a=%lu\n", (unsigned long)handle_a);

    uint64_t handle_b;
    if (src0_dequantized) {
        handle_b = ggml_backend_npm_get_dequant_handle(ctx, src0_data, src0_bytes);
    } else {
        handle_b = ggml_backend_npm_get_buffer_handle(ctx, src0_data, src0_bytes);
    }
    if (npm_debug) fprintf(stderr, "[NPM] handle_b=%lu\n", (unsigned long)handle_b);

    uint64_t handle_c = ggml_backend_npm_get_buffer_handle(ctx, dst->data, ggml_nbytes(dst));
    if (npm_debug) fprintf(stderr, "[NPM] handle_c=%lu\n", (unsigned long)handle_c);

    if (!handle_a || !handle_b || !handle_c) {
        GGML_LOG_ERROR("%s: failed to register buffers\n", __func__);
        return;
    }

    // Handle batching (ne2, ne3 dimensions)
    // Debug: print dimensions before division
    if (npm_debug) {
        fprintf(stderr, "[NPM] Dimensions: ne02=%lld ne03=%lld ne12=%lld ne13=%lld\n",
                (long long)ne02, (long long)ne03, (long long)ne12, (long long)ne13);
        fprintf(stderr, "[NPM] Matmul: M=%lld N=%lld K=%lld\n",
                (long long)ne11, (long long)ne01, (long long)ne10);
        fflush(stderr);
    }

    // Guard against division by zero
    GGML_ASSERT(ne02 > 0 && "ne02 must be positive");
    GGML_ASSERT(ne03 > 0 && "ne03 must be positive");

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    struct npm_matmul_params params;
    params.type_a = GGML_TYPE_F32;
    params.type_b = GGML_TYPE_F32;
    params.type_c = GGML_TYPE_F32;

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i03 = i13 / r3;
            const int64_t i02 = i12 / r2;

            // Set buffer handles and offsets for this batch
            params.a_handle = handle_a;
            params.b_handle = handle_b;
            params.c_handle = handle_c;

            // Calculate offsets within buffers
            params.a_offset = i12 * nb12 + i13 * nb13;
            // For dequantized data, use FP32 strides instead of original quantized strides
            if (src0_dequantized) {
                // After dequantization: contiguous FP32 with shape (ne00, ne01, ne02, ne03)
                // Stride for dim 2 = ne00 * ne01 * sizeof(float)
                // Stride for dim 3 = ne00 * ne01 * ne02 * sizeof(float)
                size_t fp32_nb02 = ne00 * ne01 * sizeof(float);
                size_t fp32_nb03 = ne00 * ne01 * ne02 * sizeof(float);
                params.b_offset = i02 * fp32_nb02 + i03 * fp32_nb03;
            } else {
                params.b_offset = i02 * nb02 + i03 * nb03;
            }
            params.c_offset = i12 * nb2 + i13 * nb3;

            params.M = ne11;  // Rows of input
            params.N = ne01;  // Rows of weights (output columns)
            params.K = ne10;  // Columns of input = columns of weights

            params.lda = ne10;  // Leading dimension of A (input)
            params.ldb = ne00;  // Leading dimension of B (weights) - same for both quantized and dequantized since we dequantize to same logical shape
            params.ldc = ne0;   // Leading dimension of C (output)

            // Dispatch to device
            int result = ctx->dev->ops.matmul(ctx->dev, &params);
            if (result != 0) {
                GGML_LOG_ERROR("%s: NPM matmul failed with error %d\n", __func__, result);
            }
        }
    }

    // Note: dequant_handle is now reused across matmul calls and cleaned up in backend_free
}

// =============================================================================
// Backend Interface Implementation
// =============================================================================

static const char * ggml_backend_npm_get_name(ggml_backend_t backend) {
    (void)backend;
    return "NPM";
}

static void ggml_backend_npm_free(ggml_backend_t backend) {
    struct ggml_backend_npm_context * ctx = (struct ggml_backend_npm_context *)backend->context;

    // Unregister all buffers
    for (auto & entry : ctx->buffer_handles) {
        ctx->dev->ops.unregister_buffer(ctx->dev, entry.second);
    }
    ctx->buffer_handles.clear();

    // Unregister dequant buffer if allocated
    if (ctx->dequant_handle != 0) {
        ctx->dev->ops.unregister_buffer(ctx->dev, ctx->dequant_handle);
        ctx->dequant_handle = 0;
        ctx->dequant_handle_size = 0;
    }

    // Destroy device
    if (ctx->dev) {
        npm_device_destroy(ctx->dev);
    }
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_npm_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_npm_context * ctx = (struct ggml_backend_npm_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_npm_mul_mat(ctx, node);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                // No-op for these operations
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    // Synchronize with device
    ctx->dev->ops.sync(ctx->dev);

    return GGML_STATUS_SUCCESS;
}

static const struct ggml_backend_i ggml_backend_npm_i = {
    /* .get_name                = */ ggml_backend_npm_get_name,
    /* .free                    = */ ggml_backend_npm_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_npm_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

// =============================================================================
// Backend GUID
// =============================================================================

static ggml_guid_t ggml_backend_npm_guid(void) {
    // Unique GUID for NPM backend
    static ggml_guid guid = {
        0x4e, 0x50, 0x4d, 0x00,  // "NPM\0"
        0xce, 0xba, 0x00, 0x01,  // "CEVA" hint
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x01
    };
    return &guid;
}

// =============================================================================
// Backend Initialization with Runtime Device Selection
// =============================================================================

// Create an npm_device for a specific implementation (mock/emulator/hardware)
static struct npm_device * npm_device_factory_create_for(const char *device_type) {
    struct npm_device * dev = nullptr;

    if (!device_type) return nullptr;

    GGML_LOG_INFO("NPM: Creating device type: %s\n", device_type);

    if (strcmp(device_type, "mock") == 0) {
        dev = npm_device_mock_create();
        if (dev) GGML_LOG_INFO("NPM: Mock device initialized\n");
    } else if (strcmp(device_type, "emulator") == 0) {
        const char * socket_path = getenv("NPM_EMULATOR_SOCKET");
        dev = npm_device_emulator_create(socket_path);
        if (dev) GGML_LOG_INFO("NPM: Emulator device initialized (socket: %s)\n",
                               socket_path ? socket_path : "/tmp/npm-emulator.sock");
    }
#ifdef NPM_SDK_PATH
    else if (strcmp(device_type, "hardware") == 0) {
        dev = npm_device_hardware_create();
        if (dev) GGML_LOG_INFO("NPM: Hardware device initialized\n");
    }
#endif
    else {
#ifdef NPM_SDK_PATH
        GGML_LOG_ERROR("NPM: Unknown device type: %s (valid: mock, emulator, hardware)\n", device_type);
#else
        GGML_LOG_ERROR("NPM: Unknown device type: %s (valid: mock, emulator)\n", device_type);
#endif
        return nullptr;
    }

    return dev;
}

// Initialize the backend and create device based on the provided device type
static ggml_backend_t ggml_backend_npm_init_with_type(const char *device_type) {
    struct npm_device * dev = npm_device_factory_create_for(device_type);

    if (!dev) {
        GGML_LOG_ERROR("%s: failed to create NPM device\n", __func__);
        return nullptr;
    }

    // Create context
    struct ggml_backend_npm_context * ctx = new ggml_backend_npm_context;
    ctx->dev = dev;
    ctx->device_id = 0;
    ctx->dequant_handle = 0;
    ctx->dequant_handle_size = 0;

    // Create backend
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_npm_guid(),
        /* .iface   = */ ggml_backend_npm_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_npm_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

// Backwards-compatible init: read device type from env and call typed init
ggml_backend_t ggml_backend_npm_init(void) {
    const char * device_type = getenv("NPM_DEVICE");
    if (!device_type) device_type = "mock";
    return ggml_backend_npm_init_with_type(device_type);
}

bool ggml_backend_is_npm(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_npm_guid());
}

// =============================================================================
// Device Interface Implementation
// =============================================================================

static const char * ggml_backend_npm_device_get_name(ggml_backend_dev_t dev) {
    (void)dev;
    // Return dynamic name based on NPM_DEVICE env var
    const char * device_type = getenv("NPM_DEVICE");
    if (!device_type) device_type = "mock";

    if (strcmp(device_type, "mock") == 0) return "NPM Mock";
    if (strcmp(device_type, "emulator") == 0) return "NPM Emulator";
#ifdef NPM_SDK_PATH
    if (strcmp(device_type, "hardware") == 0) return "NPM Hardware";
#endif
    return "NPM";
}

static const char * ggml_backend_npm_device_get_description(ggml_backend_dev_t dev) {
    (void)dev;
    return "Ceva NeuPro-M";
}

static void ggml_backend_npm_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    (void)dev;
    // Report mock memory (L2 size) as default
    *free = 8 * 1024 * 1024;   // 8MB
    *total = 8 * 1024 * 1024;
}

static enum ggml_backend_dev_type ggml_backend_npm_device_get_type(ggml_backend_dev_t dev) {
    (void)dev;
    // NPM is an accelerator device (like BLAS)
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_npm_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = ggml_backend_npm_device_get_name(dev);
    props->description = ggml_backend_npm_device_get_description(dev);
    props->type = ggml_backend_npm_device_get_type(dev);
    ggml_backend_npm_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_npm_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    (void)params;
    const char * impl = nullptr;
    if (dev && dev->context) impl = (const char *)dev->context;
    if (!impl) impl = getenv("NPM_DEVICE");
    if (!impl) impl = "mock";
    GGML_LOG_INFO("NPM: device_init_backend: dev=%p, dev->context=%s, impl=%s\n",
                  (void*)dev, (dev && dev->context) ? (const char*)dev->context : "null", impl);
    return ggml_backend_npm_init_with_type(impl);
}

static ggml_backend_buffer_type_t ggml_backend_npm_device_get_buffer_type(ggml_backend_dev_t dev) {
    (void)dev;
    // For Phase 1, use CPU buffer type (like BLAS backend)
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_npm_device_buffer_from_host_ptr(
    ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    (void)dev;
    (void)max_tensor_size;
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

// Environment variable to enable CPU fallback logging
static bool npm_log_cpu_fallback_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("NPM_LOG_CPU_FALLBACK");
        enabled = (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0));
    }
    return enabled > 0;
}

static bool ggml_backend_npm_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void)dev;

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
            const struct ggml_tensor * src0 = op->src[0];  // weights
            const struct ggml_tensor * src1 = op->src[1];  // activations

            // Minimum batch size for efficiency (similar to BLAS)
            const int64_t ne00 = src0->ne[0];  // K dimension (weight columns)
            const int64_t ne10 = src1->ne[0];
            const int64_t ne0 = op->ne[0];
            const int64_t ne1 = op->ne[1];

            const int64_t min_batch = 1;  // TODO: restore to 32 after testing

            bool contiguous_ok = ggml_is_contiguous(src0) && ggml_is_contiguous(src1);

            // src0 (weights): accept FP32 or any quantized type with to_float support
            // src1 (activations): must be FP32
            bool src0_type_ok = (src0->type == GGML_TYPE_F32);
            if (!src0_type_ok && ggml_type_is_quantized(src0->type)) {
                // Check if this quantized type has dequantization support
                const struct ggml_type_traits * traits = ggml_get_type_traits(src0->type);
                src0_type_ok = (traits != nullptr && traits->to_float != nullptr);
            }
            bool src1_type_ok = (src1->type == GGML_TYPE_F32);
            bool type_ok = src0_type_ok && src1_type_ok;

            // Block alignment validation for quantized types
            // K dimension must be divisible by quantization block size
            bool alignment_ok = true;
            if (ggml_type_is_quantized(src0->type)) {
                switch (src0->type) {
                    // K-quants: 256 elements per block
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                        alignment_ok = (ne00 % 256 == 0);
                        break;
                    // Standard quants: 32 elements per block
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q8_1:
                        alignment_ok = (ne00 % 32 == 0);
                        break;
                    // I-quants: 256 elements per super-block
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                        alignment_ok = (ne00 % 256 == 0);
                        break;
                    // FP16/BF16: no alignment requirements
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                        alignment_ok = true;
                        break;
                    default:
                        alignment_ok = true;
                        break;
                }
            }

            bool size_ok = (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch);

            bool supported = contiguous_ok && type_ok && size_ok && alignment_ok;

            if (!supported && npm_log_cpu_fallback_enabled()) {
                GGML_LOG_INFO("[NPM->CPU] MUL_MAT fallback: contiguous=%d, types=(%s,%s), dims=(%lld,%lld,%lld), alignment=%d\n",
                              contiguous_ok ? 1 : 0,
                              ggml_type_name(src0->type), ggml_type_name(src1->type),
                              (long long)ne0, (long long)ne1, (long long)ne10,
                              alignment_ok ? 1 : 0);
            }

            return supported;
        }

        default:
            if (npm_log_cpu_fallback_enabled()) {
                GGML_LOG_INFO("[NPM->CPU] Unsupported op: %s\n", ggml_op_desc(op));
            }
            return false;
    }
}

static bool ggml_backend_npm_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    (void)dev;
    // NPM (Phase 1) works with host memory buffers
    return ggml_backend_buft_is_host(buft);
}

static const struct ggml_backend_device_i ggml_backend_npm_device_i = {
    /* .get_name             = */ ggml_backend_npm_device_get_name,
    /* .get_description      = */ ggml_backend_npm_device_get_description,
    /* .get_memory           = */ ggml_backend_npm_device_get_memory,
    /* .get_type             = */ ggml_backend_npm_device_get_type,
    /* .get_props            = */ ggml_backend_npm_device_get_props,
    /* .init_backend         = */ ggml_backend_npm_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_npm_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ ggml_backend_npm_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_npm_device_supports_op,
    /* .supports_buft        = */ ggml_backend_npm_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// =============================================================================
// Registry Interface Implementation
// =============================================================================

static const char * ggml_backend_npm_reg_get_name(ggml_backend_reg_t reg) {
    (void)reg;
    return "NPM";
}

static size_t ggml_backend_npm_reg_get_device_count(ggml_backend_reg_t reg) {
    (void)reg;
    return 1;  // Single NPM device - implementation selected at runtime
}

static ggml_backend_dev_t ggml_backend_npm_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_npm_device = {
        /* .iface   = */ ggml_backend_npm_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_npm_device;
}

static void * ggml_backend_npm_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    (void)reg;
    (void)name;
    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_npm_reg_i = {
    /* .get_name         = */ ggml_backend_npm_reg_get_name,
    /* .get_device_count = */ ggml_backend_npm_reg_get_device_count,
    /* .get_device       = */ ggml_backend_npm_reg_get_device,
    /* .get_proc_address = */ ggml_backend_npm_get_proc_address,
};

ggml_backend_reg_t ggml_backend_npm_reg(void) {
    static struct ggml_backend_reg ggml_backend_npm_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_npm_reg_i,
        /* .context     = */ nullptr,
    };

    return &ggml_backend_npm_reg;
}

// Dynamic loading support
GGML_BACKEND_DL_IMPL(ggml_backend_npm_reg)
