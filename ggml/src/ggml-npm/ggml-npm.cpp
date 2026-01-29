#include "ggml-npm.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "npm-device/npm-device.h"

#include <cstring>
#include <memory>
#include <unordered_map>

// =============================================================================
// NPM Backend Context
// =============================================================================

struct ggml_backend_npm_context {
    struct npm_device * dev;
    int device_id;

    // Buffer registration cache: tensor data ptr -> device handle
    // Buffers are registered lazily on first use
    std::unordered_map<void *, uint64_t> buffer_handles;
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
    size_t size
) {
    // Check if already registered
    auto it = ctx->buffer_handles.find(ptr);
    if (it != ctx->buffer_handles.end()) {
        return it->second;
    }

    // Register new buffer
    uint64_t handle = 0;
    int result = ctx->dev->ops.register_buffer(ctx->dev, ptr, size, &handle);
    if (result != 0) {
        GGML_LOG_ERROR("%s: failed to register buffer %p (size %zu)\n", __func__, ptr, size);
        return 0;
    }

    ctx->buffer_handles[ptr] = handle;
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

    // Phase 1: only support FP32 contiguous tensors
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // Check contiguity
    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb0 == sizeof(float));

    // Register buffers with device (or get existing handles)
    uint64_t handle_a = ggml_backend_npm_get_buffer_handle(ctx, src1->data, ggml_nbytes(src1));
    uint64_t handle_b = ggml_backend_npm_get_buffer_handle(ctx, src0->data, ggml_nbytes(src0));
    uint64_t handle_c = ggml_backend_npm_get_buffer_handle(ctx, dst->data, ggml_nbytes(dst));

    if (!handle_a || !handle_b || !handle_c) {
        GGML_LOG_ERROR("%s: failed to register buffers\n", __func__);
        return;
    }

    // Handle batching (ne2, ne3 dimensions)
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
            params.b_offset = i02 * nb02 + i03 * nb03;
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

    // Unregister all buffers
    for (auto & entry : ctx->buffer_handles) {
        ctx->dev->ops.unregister_buffer(ctx->dev, entry.second);
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
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];

            // Phase 1: Only support FP32 contiguous tensors
            // Minimum batch size for efficiency (similar to BLAS)
            const int64_t ne10 = src1->ne[0];
            const int64_t ne0 = op->ne[0];
            const int64_t ne1 = op->ne[1];

            const int64_t min_batch = 32;

            return ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src0->type == GGML_TYPE_F32 &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch);
        }

        default:
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
