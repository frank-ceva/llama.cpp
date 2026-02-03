// NPM Emulator Device Implementation
//
// Phase 1.5: IPC-based device driver that communicates with the npm-emulator
// process via Unix socket. Data is shared through POSIX shared memory.

#include "npm-device.h"

#define NPM_SHM_IMPLEMENTATION
#include "npm-shm.h"

#include "npm-emu-protocol.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

// Default shared memory size (1.5 GB) - for dequantized weights
#define NPM_EMU_DEFAULT_SHM_SIZE (1536ULL * 1024 * 1024)

// =============================================================================
// Emulator device context
// =============================================================================

struct npm_device_emu_context {
    // Socket connection
    int socket_fd;
    uint32_t seq_id;

    // Shared memory
    npm_shm_region * shm;

    // Device info (from emulator)
    enum npm_sku sku;
    int num_engines;
    size_t l1_size;
    size_t l2_size;

    // Buffer registry: local ptr -> (handle, shm_offset)
    struct buffer_info {
        uint64_t handle;
        size_t   shm_offset;
        size_t   size;
    };
    std::unordered_map<void *, buffer_info> buffers;
};

// =============================================================================
// IPC helpers
// =============================================================================

static bool send_all(int fd, const void * buf, size_t size) {
    const char * p = (const char *)buf;
    while (size > 0) {
        ssize_t n = send(fd, p, size, 0);
        if (n <= 0) return false;
        p += n;
        size -= n;
    }
    return true;
}

static bool recv_all(int fd, void * buf, size_t size) {
    char * p = (char *)buf;
    while (size > 0) {
        ssize_t n = recv(fd, p, size, 0);
        if (n <= 0) return false;
        p += n;
        size -= n;
    }
    return true;
}

static bool send_message(npm_device_emu_context * ctx, enum npm_emu_cmd cmd,
                         const void * payload, size_t payload_size) {
    npm_emu_header hdr;
    npm_emu_header_init(&hdr, cmd, ctx->seq_id++, payload_size);

    if (!send_all(ctx->socket_fd, &hdr, sizeof(hdr))) {
        return false;
    }
    if (payload_size > 0 && payload) {
        if (!send_all(ctx->socket_fd, payload, payload_size)) {
            return false;
        }
    }
    return true;
}

static bool recv_response(npm_device_emu_context * ctx, npm_emu_header * hdr,
                          void * payload, size_t payload_size) {
    if (!recv_all(ctx->socket_fd, hdr, sizeof(*hdr))) {
        return false;
    }
    if (npm_emu_header_validate(hdr) != 0) {
        return false;
    }
    if (payload_size > 0 && payload) {
        if (!recv_all(ctx->socket_fd, payload, payload_size)) {
            return false;
        }
    }
    return true;
}

// =============================================================================
// Lifecycle
// =============================================================================

static int npm_device_emu_init(struct npm_device * dev, int device_id) {
    (void)device_id;

    npm_device_emu_context * ctx = (npm_device_emu_context *)dev->context;

    // Get socket path from environment or use default
    const char * socket_path = getenv("NPM_EMULATOR_SOCKET");
    if (!socket_path) {
        socket_path = NPM_EMU_DEFAULT_SOCKET;
    }

    // Connect to emulator
    ctx->socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (ctx->socket_fd < 0) {
        fprintf(stderr, "[npm-device-emulator] Failed to create socket\n");
        return -1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

    // Set socket timeouts (5 seconds) to avoid hanging if emulator is unresponsive
    struct timeval timeout;
    timeout.tv_sec = 5;
    timeout.tv_usec = 0;
    setsockopt(ctx->socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(ctx->socket_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

    if (connect(ctx->socket_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[npm-device-emulator] Failed to connect to emulator at %s\n", socket_path);
        fprintf(stderr, "[npm-device-emulator] Make sure npm-emulator is running: ./npm-emulator --tiling\n");
        close(ctx->socket_fd);
        ctx->socket_fd = -1;
        return -1;
    }

    // Create shared memory
    ctx->shm = npm_shm_create(NPM_EMU_DEFAULT_SHM_SIZE);
    if (!ctx->shm) {
        fprintf(stderr, "[npm-device-emulator] Failed to create shared memory\n");
        close(ctx->socket_fd);
        ctx->socket_fd = -1;
        return -1;
    }

    // Send HELLO
    npm_emu_hello_req hello_req;
    memset(&hello_req, 0, sizeof(hello_req));
    hello_req.version_major = NPM_EMU_VERSION_MAJOR;
    hello_req.version_minor = NPM_EMU_VERSION_MINOR;
    strncpy(hello_req.shm_name, ctx->shm->name, sizeof(hello_req.shm_name) - 1);
    hello_req.shm_size = ctx->shm->size;

    if (!send_message(ctx, NPM_EMU_CMD_HELLO, &hello_req, sizeof(hello_req))) {
        fprintf(stderr, "[npm-device-emulator] Failed to send HELLO\n");
        npm_shm_destroy(ctx->shm);
        close(ctx->socket_fd);
        ctx->socket_fd = -1;
        return -1;
    }

    // Receive HELLO response
    npm_emu_header hdr;
    npm_emu_hello_rsp hello_rsp;
    if (!recv_response(ctx, &hdr, &hello_rsp, sizeof(hello_rsp))) {
        fprintf(stderr, "[npm-device-emulator] Failed to receive HELLO response\n");
        npm_shm_destroy(ctx->shm);
        close(ctx->socket_fd);
        ctx->socket_fd = -1;
        return -1;
    }

    if (hello_rsp.status != NPM_EMU_STATUS_OK) {
        fprintf(stderr, "[npm-device-emulator] HELLO failed: status=%d\n", hello_rsp.status);
        npm_shm_destroy(ctx->shm);
        close(ctx->socket_fd);
        ctx->socket_fd = -1;
        return -1;
    }

    // Store device info
    ctx->sku = (enum npm_sku)hello_rsp.sku;
    ctx->num_engines = hello_rsp.num_engines;
    ctx->l1_size = hello_rsp.l1_size;
    ctx->l2_size = hello_rsp.l2_size;

    return 0;
}

static void npm_device_emu_shutdown(struct npm_device * dev) {
    npm_device_emu_context * ctx = (npm_device_emu_context *)dev->context;

    if (ctx->socket_fd >= 0) {
        // Send GOODBYE
        send_message(ctx, NPM_EMU_CMD_GOODBYE, nullptr, 0);

        // Receive response (ignore errors)
        npm_emu_header hdr;
        npm_emu_goodbye_rsp rsp;
        recv_response(ctx, &hdr, &rsp, sizeof(rsp));

        close(ctx->socket_fd);
        ctx->socket_fd = -1;
    }

    if (ctx->shm) {
        npm_shm_destroy(ctx->shm);
        ctx->shm = nullptr;
    }

    ctx->buffers.clear();
}

// =============================================================================
// Device info
// =============================================================================

static enum npm_sku npm_device_emu_get_sku(struct npm_device * dev) {
    npm_device_emu_context * ctx = (npm_device_emu_context *)dev->context;
    return ctx->sku;
}

static int npm_device_emu_get_num_engines(struct npm_device * dev) {
    npm_device_emu_context * ctx = (npm_device_emu_context *)dev->context;
    return ctx->num_engines;
}

static size_t npm_device_emu_get_l1_size(struct npm_device * dev) {
    npm_device_emu_context * ctx = (npm_device_emu_context *)dev->context;
    return ctx->l1_size;
}

static size_t npm_device_emu_get_l2_size(struct npm_device * dev) {
    npm_device_emu_context * ctx = (npm_device_emu_context *)dev->context;
    return ctx->l2_size;
}

// =============================================================================
// Memory management
// =============================================================================

static int npm_device_emu_register_buffer(struct npm_device * dev, void * ptr,
                                           size_t size, uint64_t * handle) {
    npm_device_emu_context * ctx = (npm_device_emu_context *)dev->context;

    if (!ptr || size == 0 || !handle) {
        return -1;
    }

    // Allocate in shared memory
    size_t shm_offset = npm_shm_alloc(ctx->shm, size, 64);
    if (shm_offset == (size_t)-1) {
        fprintf(stderr, "[npm-device-emulator] Shared memory allocation failed\n");
        return -1;
    }

    // Copy data to shared memory
    void * shm_ptr = npm_shm_get_ptr(ctx->shm, shm_offset);
    memcpy(shm_ptr, ptr, size);

    // Register with emulator
    npm_emu_register_buffer_req req;
    memset(&req, 0, sizeof(req));
    req.shm_offset = shm_offset;
    req.size = size;
    req.flags = 0;

    if (!send_message(ctx, NPM_EMU_CMD_REGISTER_BUFFER, &req, sizeof(req))) {
        return -1;
    }

    npm_emu_header hdr;
    npm_emu_register_buffer_rsp rsp;
    if (!recv_response(ctx, &hdr, &rsp, sizeof(rsp))) {
        return -1;
    }

    if (rsp.status != NPM_EMU_STATUS_OK) {
        return -1;
    }

    // Store mapping
    ctx->buffers[ptr] = { rsp.handle, shm_offset, size };
    *handle = rsp.handle;

    return 0;
}

static void npm_device_emu_unregister_buffer(struct npm_device * dev, uint64_t handle) {
    npm_device_emu_context * ctx = (npm_device_emu_context *)dev->context;

    // Find and remove from local map
    for (auto it = ctx->buffers.begin(); it != ctx->buffers.end(); ++it) {
        if (it->second.handle == handle) {
            ctx->buffers.erase(it);
            break;
        }
    }

    // Tell emulator
    npm_emu_unregister_buffer_req req;
    req.handle = handle;

    send_message(ctx, NPM_EMU_CMD_UNREGISTER_BUFFER, &req, sizeof(req));

    npm_emu_header hdr;
    npm_emu_unregister_buffer_rsp rsp;
    recv_response(ctx, &hdr, &rsp, sizeof(rsp));
}

static int npm_device_emu_update_buffer(struct npm_device * dev, uint64_t handle, void * ptr, size_t size) {
    npm_device_emu_context * ctx = (npm_device_emu_context *)dev->context;

    // Find the buffer info for this handle
    size_t shm_offset = (size_t)-1;
    size_t registered_size = 0;

    for (const auto & entry : ctx->buffers) {
        if (entry.second.handle == handle) {
            shm_offset = entry.second.shm_offset;
            registered_size = entry.second.size;
            break;
        }
    }

    if (shm_offset == (size_t)-1) {
        return -1;  // Handle not found
    }

    if (size > registered_size) {
        return -2;  // New data is larger than allocated space
    }

    // Copy new data to shared memory
    void * shm_ptr = npm_shm_get_ptr(ctx->shm, shm_offset);
    if (!shm_ptr) {
        return -3;
    }

    memcpy(shm_ptr, ptr, size);
    return 0;
}

// =============================================================================
// Helper: get shm offset for a handle
// =============================================================================

static size_t get_shm_offset_for_handle(npm_device_emu_context * ctx, uint64_t handle) {
    for (const auto & entry : ctx->buffers) {
        if (entry.second.handle == handle) {
            return entry.second.shm_offset;
        }
    }
    return (size_t)-1;
}

// =============================================================================
// Compute operations
// =============================================================================

static int npm_device_emu_matmul(struct npm_device * dev, const struct npm_matmul_params * params) {
    npm_device_emu_context * ctx = (npm_device_emu_context *)dev->context;

    // Before sending matmul, ensure data is synced to shared memory
    // (For input buffers, data should already be there from register_buffer)

    // Send MATMUL command
    npm_emu_matmul_req req;
    memset(&req, 0, sizeof(req));
    req.a_handle = params->a_handle;
    req.a_offset = params->a_offset;
    req.b_handle = params->b_handle;
    req.b_offset = params->b_offset;
    req.c_handle = params->c_handle;
    req.c_offset = params->c_offset;
    req.M = params->M;
    req.N = params->N;
    req.K = params->K;
    req.lda = params->lda;
    req.ldb = params->ldb;
    req.ldc = params->ldc;
    req.type_a = params->type_a;
    req.type_b = params->type_b;
    req.type_c = params->type_c;
    req.flags = 0;

    if (!send_message(ctx, NPM_EMU_CMD_MATMUL, &req, sizeof(req))) {
        return -1;
    }

    npm_emu_header hdr;
    npm_emu_matmul_rsp rsp;
    if (!recv_response(ctx, &hdr, &rsp, sizeof(rsp))) {
        return -1;
    }

    if (rsp.status != NPM_EMU_STATUS_OK) {
        return -1;
    }

    // Copy output back from shared memory to original buffer
    // Find the output buffer
    for (auto & entry : ctx->buffers) {
        if (entry.second.handle == params->c_handle) {
            void * shm_ptr = npm_shm_get_ptr(ctx->shm, entry.second.shm_offset);
            memcpy(entry.first, shm_ptr, entry.second.size);
            break;
        }
    }

    return 0;
}

// =============================================================================
// Synchronization
// =============================================================================

static int npm_device_emu_sync(struct npm_device * dev) {
    npm_device_emu_context * ctx = (npm_device_emu_context *)dev->context;

    if (!send_message(ctx, NPM_EMU_CMD_SYNC, nullptr, 0)) {
        return -1;
    }

    npm_emu_header hdr;
    npm_emu_sync_rsp rsp;
    if (!recv_response(ctx, &hdr, &rsp, sizeof(rsp))) {
        return -1;
    }

    return rsp.status == NPM_EMU_STATUS_OK ? 0 : -1;
}

static int npm_device_emu_fence_create(struct npm_device * dev, struct npm_fence ** fence) {
    npm_device_emu_context * ctx = (npm_device_emu_context *)dev->context;

    if (!send_message(ctx, NPM_EMU_CMD_FENCE_CREATE, nullptr, 0)) {
        return -1;
    }

    npm_emu_header hdr;
    npm_emu_fence_create_rsp rsp;
    if (!recv_response(ctx, &hdr, &rsp, sizeof(rsp))) {
        return -1;
    }

    if (rsp.status != NPM_EMU_STATUS_OK) {
        return -1;
    }

    // Store fence ID as pointer (hacky but works)
    *fence = (struct npm_fence *)(uintptr_t)rsp.fence_id;
    return 0;
}

static void npm_device_emu_fence_destroy(struct npm_device * dev, struct npm_fence * fence) {
    npm_device_emu_context * ctx = (npm_device_emu_context *)dev->context;

    npm_emu_fence_destroy_req req;
    req.fence_id = (uint64_t)(uintptr_t)fence;

    send_message(ctx, NPM_EMU_CMD_FENCE_DESTROY, &req, sizeof(req));

    npm_emu_header hdr;
    npm_emu_fence_destroy_rsp rsp;
    recv_response(ctx, &hdr, &rsp, sizeof(rsp));
}

static int npm_device_emu_fence_wait(struct npm_device * dev, struct npm_fence * fence,
                                      uint64_t timeout_ns) {
    npm_device_emu_context * ctx = (npm_device_emu_context *)dev->context;

    npm_emu_fence_wait_req req;
    req.fence_id = (uint64_t)(uintptr_t)fence;
    req.timeout_ns = timeout_ns;

    if (!send_message(ctx, NPM_EMU_CMD_FENCE_WAIT, &req, sizeof(req))) {
        return -1;
    }

    npm_emu_header hdr;
    npm_emu_fence_wait_rsp rsp;
    if (!recv_response(ctx, &hdr, &rsp, sizeof(rsp))) {
        return -1;
    }

    return rsp.status == NPM_EMU_STATUS_OK ? 0 : -1;
}

// =============================================================================
// Factory function
// =============================================================================

struct npm_device * npm_device_emulator_create(const char * socket_path) {
    (void)socket_path;  // Can set via env var NPM_EMULATOR_SOCKET

    npm_device * dev = new npm_device;
    if (!dev) {
        return nullptr;
    }

    npm_device_emu_context * ctx = new npm_device_emu_context;
    if (!ctx) {
        delete dev;
        return nullptr;
    }

    // Don't use memset - it corrupts std::unordered_map
    // Initialize members properly instead
    ctx->socket_fd = -1;
    ctx->seq_id = 0;
    ctx->shm = nullptr;
    ctx->sku = NPM_SKU_EMULATOR;
    ctx->num_engines = 0;
    ctx->l1_size = 0;
    ctx->l2_size = 0;
    // ctx->buffers is default-initialized by new

    dev->context = ctx;

    // Set up operations
    dev->ops.init = npm_device_emu_init;
    dev->ops.shutdown = npm_device_emu_shutdown;
    dev->ops.get_sku = npm_device_emu_get_sku;
    dev->ops.get_num_engines = npm_device_emu_get_num_engines;
    dev->ops.get_l1_size = npm_device_emu_get_l1_size;
    dev->ops.get_l2_size = npm_device_emu_get_l2_size;
    dev->ops.register_buffer = npm_device_emu_register_buffer;
    dev->ops.unregister_buffer = npm_device_emu_unregister_buffer;
    dev->ops.update_buffer = npm_device_emu_update_buffer;
    dev->ops.matmul = npm_device_emu_matmul;
    dev->ops.sync = npm_device_emu_sync;
    dev->ops.fence_create = npm_device_emu_fence_create;
    dev->ops.fence_destroy = npm_device_emu_fence_destroy;
    dev->ops.fence_wait = npm_device_emu_fence_wait;

    // Initialize
    if (dev->ops.init(dev, 0) != 0) {
        delete ctx;
        delete dev;
        return nullptr;
    }

    return dev;
}

// Note: npm_sku_name() and npm_device_destroy() are defined in npm-device-common.cpp
