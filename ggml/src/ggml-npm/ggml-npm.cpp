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
// Supported quantized types for dequantization
// =============================================================================

static bool ggml_npm_is_quantized_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
    }
}

// =============================================================================
// NPM Backend Context
// =============================================================================

struct ggml_backend_npm_context {
    struct npm_device * dev;
    int device_id;

    // Buffer registration cache: tensor data ptr -> {device handle, registered size}
    // Buffers are registered lazily on first use
    struct buffer_reg_info {
        uint64_t handle;
        size_t size;
    };
    std::unordered_map<void *, buffer_reg_info> buffer_handles;

    // Dequantization buffer for quantized weights
    // Reused across matmul operations to avoid repeated allocations
    std::vector<float> dequant_buffer;
    uint64_t dequant_handle;
    size_t dequant_handle_size;
};

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
static uint64_t ggml_backend_npm_get_buffer_handle(
    struct ggml_backend_npm_context * ctx,
    void * ptr,
    size_t size,
    bool update_data = true
) {
    // Check if already registered
    auto it = ctx->buffer_handles.find(ptr);
    if (it != ctx->buffer_handles.end()) {
        // Buffer already registered - check if size increased
        if (size > it->second.size) {
            // Need to re-register with larger size
            if (getenv("NPM_DEBUG")) {
                GGML_LOG_INFO("[NPM] Re-registering buffer %p: size %zu -> %zu\n",
                              ptr, it->second.size, size);
            }
            ctx->dev->ops.unregister_buffer(ctx->dev, it->second.handle);
            // Fall through to register with new size
        } else {
            // Size fits, just update data in device memory if requested
            if (update_data && ctx->dev->ops.update_buffer) {
                ctx->dev->ops.update_buffer(ctx->dev, it->second.handle, ptr, size);
            }
            return it->second.handle;
        }
    }

    // Register new buffer
    uint64_t handle = 0;
    int result = ctx->dev->ops.register_buffer(ctx->dev, ptr, size, &handle);
    if (result != 0) {
        GGML_LOG_ERROR("%s: failed to register buffer %p (size %zu)\n", __func__, ptr, size);
        return 0;
    }

    ctx->buffer_handles[ptr] = { handle, size };
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

// Helper to get or resize the dequantization buffer handle
static uint64_t ggml_backend_npm_get_dequant_handle(
    struct ggml_backend_npm_context * ctx,
    size_t required_size
) {
    // If we have a handle that's big enough, reuse it
    if (ctx->dequant_handle != 0 && ctx->dequant_handle_size >= required_size) {
        return ctx->dequant_handle;
    }

    // Need to allocate or resize
    // Unregister old handle if exists
    if (ctx->dequant_handle != 0) {
        ctx->dev->ops.unregister_buffer(ctx->dev, ctx->dequant_handle);
        ctx->dequant_handle = 0;
        ctx->dequant_handle_size = 0;
    }

    // Resize buffer
    size_t num_floats = required_size / sizeof(float);
    ctx->dequant_buffer.resize(num_floats);

    // Register new buffer
    uint64_t handle = 0;
    int result = ctx->dev->ops.register_buffer(ctx->dev, ctx->dequant_buffer.data(), required_size, &handle);
    if (result != 0) {
        GGML_LOG_ERROR("%s: failed to register dequant buffer (size %zu)\n", __func__, required_size);
        return 0;
    }

    ctx->dequant_handle = handle;
    ctx->dequant_handle_size = required_size;
    return handle;
}

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

    // Input (activations) and output must be FP32
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // Check contiguity
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb0 == sizeof(float));

    // Register input and output buffers
    uint64_t handle_a = ggml_backend_npm_get_buffer_handle(ctx, src1->data, ggml_nbytes(src1), true);   // input - update
    uint64_t handle_c = ggml_backend_npm_get_buffer_handle(ctx, dst->data, ggml_nbytes(dst), false);    // output - no update needed

    if (!handle_a || !handle_c) {
        GGML_LOG_ERROR("%s: failed to register input/output buffers\n", __func__);
        return;
    }

    // Handle weights - may need dequantization
    uint64_t handle_b = 0;
    bool weights_dequantized = false;

    if (src0->type == GGML_TYPE_F32) {
        // FP32 weights - use directly
        handle_b = ggml_backend_npm_get_buffer_handle(ctx, src0->data, ggml_nbytes(src0), true);
    } else if (ggml_npm_is_quantized_type(src0->type)) {
        // Quantized weights - dequantize to FP32
        const size_t n_elements = ggml_nelements(src0);
        const size_t dequant_size = n_elements * sizeof(float);

        static bool debug_logged = false;
        if (!debug_logged || getenv("NPM_DEBUG")) {
            GGML_LOG_INFO("[NPM] Dequantizing %s: %zu elements -> %zu bytes FP32\n",
                          ggml_type_name(src0->type), n_elements, dequant_size);
            debug_logged = true;
        }

        handle_b = ggml_backend_npm_get_dequant_handle(ctx, dequant_size);
        if (!handle_b) {
            GGML_LOG_ERROR("%s: failed to get dequant buffer (size %zu)\n", __func__, dequant_size);
            return;
        }

        // Get the dequantization function for this type
        const ggml_type_traits * traits = ggml_get_type_traits(src0->type);
        if (!traits || !traits->to_float) {
            GGML_LOG_ERROR("%s: no dequantization function for type %s\n", __func__, ggml_type_name(src0->type));
            return;
        }

        // Dequantize weights to FP32 buffer
        traits->to_float(src0->data, ctx->dequant_buffer.data(), n_elements);

        // Debug: verify dequantized data
        if (getenv("NPM_DEBUG")) {
            float sum = 0.0f;
            for (size_t i = 0; i < std::min(n_elements, (size_t)100); i++) {
                sum += ctx->dequant_buffer[i];
            }
            GGML_LOG_INFO("[NPM] Dequant checksum (first 100): %.6f\n", sum);
        }

        // Update the device buffer with dequantized data
        if (ctx->dev->ops.update_buffer) {
            int upd_result = ctx->dev->ops.update_buffer(ctx->dev, handle_b, ctx->dequant_buffer.data(), dequant_size);
            if (upd_result != 0) {
                GGML_LOG_ERROR("[NPM] update_buffer failed for dequant buffer\n");
            }
        }

        weights_dequantized = true;
    } else {
        GGML_LOG_ERROR("%s: unsupported weight type %s\n", __func__, ggml_type_name(src0->type));
        return;
    }

    if (!handle_b) {
        GGML_LOG_ERROR("%s: failed to register weights buffer\n", __func__);
        return;
    }

    // Handle batching (ne2, ne3 dimensions)
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    struct npm_matmul_params params;
    params.type_a = GGML_TYPE_F32;
    params.type_b = GGML_TYPE_F32;  // Always FP32 after dequantization
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

            if (weights_dequantized) {
                // For dequantized weights, offset is in FP32 elements
                const size_t weights_per_batch = ne00 * ne01;
                params.b_offset = (i02 * ne01 + i03 * ne01 * ne02) * ne00 * sizeof(float);
            } else {
                params.b_offset = i02 * nb02 + i03 * nb03;
            }
            params.c_offset = i12 * nb2 + i13 * nb3;

            params.M = ne11;  // Rows of input
            params.N = ne01;  // Rows of weights (output columns)
            params.K = ne10;  // Columns of input = columns of weights

            params.lda = ne10;  // Leading dimension of A (input)
            params.ldb = ne00;  // Leading dimension of B (weights)
            params.ldc = ne0;   // Leading dimension of C (output)

            // Dispatch to device
            int result = ctx->dev->ops.matmul(ctx->dev, &params);
            if (result != 0) {
                GGML_LOG_ERROR("%s: NPM matmul failed with error %d\n", __func__, result);
            }
        }
    }
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

    // Unregister dequant buffer if allocated
    if (ctx->dequant_handle != 0) {
        ctx->dev->ops.unregister_buffer(ctx->dev, ctx->dequant_handle);
        ctx->dequant_handle = 0;
    }

    // Unregister all buffers
    for (auto & entry : ctx->buffer_handles) {
        ctx->dev->ops.unregister_buffer(ctx->dev, entry.second.handle);
    }
    ctx->buffer_handles.clear();

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
// Backend Initialization
// =============================================================================

ggml_backend_t ggml_backend_npm_init(void) {
    // Create device based on build configuration
    struct npm_device * dev = nullptr;

#if defined(NPM_DEVICE_EMULATOR)
    dev = npm_device_emulator_create(nullptr);
#elif defined(NPM_DEVICE_HARDWARE)
    dev = npm_device_hardware_create();
#else
    // Default to mock device
    dev = npm_device_mock_create();
#endif

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

bool ggml_backend_is_npm(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_npm_guid());
}

// =============================================================================
// Device Interface Implementation
// =============================================================================

static const char * ggml_backend_npm_device_get_name(ggml_backend_dev_t dev) {
    (void)dev;
    return "NPM";
}

static const char * ggml_backend_npm_device_get_description(ggml_backend_dev_t dev) {
    (void)dev;
    return "Ceva NeuPro-M";
}

static void ggml_backend_npm_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    (void)dev;
    // Report mock memory (L2 size)
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
    (void)dev;
    (void)params;
    return ggml_backend_npm_init();
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
            const struct ggml_tensor * src1 = op->src[1];  // input

            // Minimum batch size - set to 1 for emulation testing
            const int64_t ne00 = src0->ne[0];  // K dimension (weight columns)
            const int64_t ne10 = src1->ne[0];  // K dimension (input columns)
            const int64_t ne0 = op->ne[0];
            const int64_t ne1 = op->ne[1];

            const int64_t min_batch = 1;

            bool contiguous_ok = ggml_is_contiguous(src0) && ggml_is_contiguous(src1);

            // Input must be FP32, weights can be FP32 or quantized (will be dequantized)
            bool input_type_ok = (src1->type == GGML_TYPE_F32);
            bool weight_type_ok = (src0->type == GGML_TYPE_F32) || ggml_npm_is_quantized_type(src0->type);
            bool output_type_ok = (op->type == GGML_TYPE_F32);

            // Block alignment validation for quantized types
            // K dimension must be divisible by quantization block size
            bool alignment_ok = true;
            if (ggml_npm_is_quantized_type(src0->type)) {
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

            bool type_ok = input_type_ok && weight_type_ok && output_type_ok;
            bool size_ok = (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch);

            bool supported = contiguous_ok && type_ok && size_ok && alignment_ok;

            if (!supported && npm_log_cpu_fallback_enabled()) {
                GGML_LOG_INFO("[NPM->CPU] MUL_MAT fallback: contiguous=%d, types=(%s,%s->%s), dims=(%lld,%lld,%lld), alignment=%d\n",
                              contiguous_ok ? 1 : 0,
                              ggml_type_name(src0->type), ggml_type_name(src1->type), ggml_type_name(op->type),
                              (long long)ne0, (long long)ne1, (long long)ne00,
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
    return 1;  // Single NPM device for now
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
