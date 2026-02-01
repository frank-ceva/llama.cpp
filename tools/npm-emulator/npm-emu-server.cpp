// NPM Emulator Server Implementation

#include "npm-emu-server.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <unistd.h>
#include <signal.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <fcntl.h>

// Global shutdown flag for signal handling
static volatile sig_atomic_t g_shutdown_requested = 0;

// =============================================================================
// Helper: send/receive messages
// =============================================================================

static bool send_all(int fd, const void * buf, size_t size) {
    const char * p = (const char *)buf;
    while (size > 0) {
        ssize_t n = send(fd, p, size, 0);
        if (n <= 0) {
            return false;
        }
        p += n;
        size -= n;
    }
    return true;
}

static bool recv_all(int fd, void * buf, size_t size) {
    char * p = (char *)buf;
    while (size > 0) {
        ssize_t n = recv(fd, p, size, 0);
        if (n <= 0) {
            return false;
        }
        p += n;
        size -= n;
    }
    return true;
}

// =============================================================================
// Helper: resolve handle to pointer
// =============================================================================

static void * resolve_handle(npm_emu_server * server, uint64_t handle, size_t offset) {
    auto it = server->buffers.find(handle);
    if (it == server->buffers.end()) {
        return nullptr;
    }
    const npm_emu_buffer & buf = it->second;
    if (offset >= buf.size) {
        return nullptr;
    }
    return (char *)server->shm_base + buf.shm_offset + offset;
}

// =============================================================================
// Command handlers
// =============================================================================

static void handle_hello(npm_emu_server * server, const npm_emu_header * hdr) {
    npm_emu_hello_req req;
    if (!recv_all(server->client_fd, &req, sizeof(req))) {
        return;
    }

    if (server->config.verbose) {
        printf("[Server] HELLO from client v%d.%d, shm=%s size=%lu\n",
               req.version_major, req.version_minor, req.shm_name, (unsigned long)req.shm_size);
    }

    // Attach to shared memory
    int fd = shm_open(req.shm_name, O_RDWR, 0);
    if (fd >= 0) {
        server->shm_base = mmap(NULL, req.shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (server->shm_base != MAP_FAILED) {
            server->shm_size = req.shm_size;
        } else {
            server->shm_base = nullptr;
        }
    }

    // Send response
    npm_emu_header rsp_hdr;
    npm_emu_header_init(&rsp_hdr, NPM_EMU_CMD_HELLO, hdr->seq_id, sizeof(npm_emu_hello_rsp));

    npm_emu_hello_rsp rsp;
    memset(&rsp, 0, sizeof(rsp));
    rsp.status = server->shm_base ? NPM_EMU_STATUS_OK : NPM_EMU_STATUS_ERROR;
    rsp.version_major = NPM_EMU_VERSION_MAJOR;
    rsp.version_minor = NPM_EMU_VERSION_MINOR;
    rsp.sku = server->config.sku;
    rsp.num_engines = server->num_engines;
    rsp.l1_size = server->l1_size;
    rsp.l2_size = server->l2_size;

    send_all(server->client_fd, &rsp_hdr, sizeof(rsp_hdr));
    send_all(server->client_fd, &rsp, sizeof(rsp));
}

static void handle_goodbye(npm_emu_server * server, const npm_emu_header * hdr) {
    if (server->config.verbose) {
        printf("[Server] GOODBYE from client\n");
    }

    // Detach shared memory
    if (server->shm_base) {
        munmap(server->shm_base, server->shm_size);
        server->shm_base = nullptr;
        server->shm_size = 0;
    }

    // Clear buffers
    server->buffers.clear();

    // Send response
    npm_emu_header rsp_hdr;
    npm_emu_header_init(&rsp_hdr, NPM_EMU_CMD_GOODBYE, hdr->seq_id, sizeof(npm_emu_goodbye_rsp));

    npm_emu_goodbye_rsp rsp;
    memset(&rsp, 0, sizeof(rsp));
    rsp.status = NPM_EMU_STATUS_OK;

    send_all(server->client_fd, &rsp_hdr, sizeof(rsp_hdr));
    send_all(server->client_fd, &rsp, sizeof(rsp));
}

static void handle_register_buffer(npm_emu_server * server, const npm_emu_header * hdr) {
    npm_emu_register_buffer_req req;
    if (!recv_all(server->client_fd, &req, sizeof(req))) {
        return;
    }

    uint64_t handle = server->next_handle++;
    server->buffers[handle] = { req.shm_offset, (size_t)req.size, req.flags };

    if (server->config.verbose) {
        printf("[Server] REGISTER_BUFFER offset=%lu size=%lu -> handle=%lu\n",
               (unsigned long)req.shm_offset, (unsigned long)req.size, (unsigned long)handle);
    }

    npm_emu_header rsp_hdr;
    npm_emu_header_init(&rsp_hdr, NPM_EMU_CMD_REGISTER_BUFFER, hdr->seq_id, sizeof(npm_emu_register_buffer_rsp));

    npm_emu_register_buffer_rsp rsp;
    memset(&rsp, 0, sizeof(rsp));
    rsp.status = NPM_EMU_STATUS_OK;
    rsp.handle = handle;

    send_all(server->client_fd, &rsp_hdr, sizeof(rsp_hdr));
    send_all(server->client_fd, &rsp, sizeof(rsp));
}

static void handle_unregister_buffer(npm_emu_server * server, const npm_emu_header * hdr) {
    npm_emu_unregister_buffer_req req;
    if (!recv_all(server->client_fd, &req, sizeof(req))) {
        return;
    }

    server->buffers.erase(req.handle);

    if (server->config.verbose) {
        printf("[Server] UNREGISTER_BUFFER handle=%lu\n", (unsigned long)req.handle);
    }

    npm_emu_header rsp_hdr;
    npm_emu_header_init(&rsp_hdr, NPM_EMU_CMD_UNREGISTER_BUFFER, hdr->seq_id, sizeof(npm_emu_unregister_buffer_rsp));

    npm_emu_unregister_buffer_rsp rsp;
    memset(&rsp, 0, sizeof(rsp));
    rsp.status = NPM_EMU_STATUS_OK;

    send_all(server->client_fd, &rsp_hdr, sizeof(rsp_hdr));
    send_all(server->client_fd, &rsp, sizeof(rsp));
}

static void handle_matmul(npm_emu_server * server, const npm_emu_header * hdr) {
    npm_emu_matmul_req req;
    if (!recv_all(server->client_fd, &req, sizeof(req))) {
        return;
    }

    if (server->config.verbose) {
        printf("[Server] MATMUL M=%ld N=%ld K=%ld\n",
               (long)req.M, (long)req.N, (long)req.K);
    }

    uint8_t status = NPM_EMU_STATUS_OK;

    // Resolve buffer handles to pointers
    const float * A = (const float *)resolve_handle(server, req.a_handle, req.a_offset);
    const float * B = (const float *)resolve_handle(server, req.b_handle, req.b_offset);
    float * C = (float *)resolve_handle(server, req.c_handle, req.c_offset);

    if (!A || !B || !C) {
        status = NPM_EMU_STATUS_INVALID_HANDLE;
    } else {
        // Execute matmul (same naive GEMM as mock)
        // C = A * B^T
        // A: (M, K), B: (N, K), C: (M, N)
        for (int64_t m = 0; m < req.M; m++) {
            for (int64_t n = 0; n < req.N; n++) {
                float sum = 0.0f;
                for (int64_t k = 0; k < req.K; k++) {
                    sum += A[m * req.lda + k] * B[n * req.ldb + k];
                }
                C[m * req.ldc + n] = sum;
            }
        }

        server->total_matmul_ops++;
    }

    npm_emu_header rsp_hdr;
    npm_emu_header_init(&rsp_hdr, NPM_EMU_CMD_MATMUL, hdr->seq_id, sizeof(npm_emu_matmul_rsp));

    npm_emu_matmul_rsp rsp;
    memset(&rsp, 0, sizeof(rsp));
    rsp.status = status;
    rsp.cycles = 0;  // Timing not implemented yet
    rsp.dma_bytes = 0;

    send_all(server->client_fd, &rsp_hdr, sizeof(rsp_hdr));
    send_all(server->client_fd, &rsp, sizeof(rsp));
}

static void handle_sync(npm_emu_server * server, const npm_emu_header * hdr) {
    if (server->config.verbose) {
        printf("[Server] SYNC\n");
    }

    npm_emu_header rsp_hdr;
    npm_emu_header_init(&rsp_hdr, NPM_EMU_CMD_SYNC, hdr->seq_id, sizeof(npm_emu_sync_rsp));

    npm_emu_sync_rsp rsp;
    memset(&rsp, 0, sizeof(rsp));
    rsp.status = NPM_EMU_STATUS_OK;

    send_all(server->client_fd, &rsp_hdr, sizeof(rsp_hdr));
    send_all(server->client_fd, &rsp, sizeof(rsp));
}

static void handle_fence_create(npm_emu_server * server, const npm_emu_header * hdr) {
    uint64_t fence_id = server->next_fence_id++;

    npm_emu_header rsp_hdr;
    npm_emu_header_init(&rsp_hdr, NPM_EMU_CMD_FENCE_CREATE, hdr->seq_id, sizeof(npm_emu_fence_create_rsp));

    npm_emu_fence_create_rsp rsp;
    memset(&rsp, 0, sizeof(rsp));
    rsp.status = NPM_EMU_STATUS_OK;
    rsp.fence_id = fence_id;

    send_all(server->client_fd, &rsp_hdr, sizeof(rsp_hdr));
    send_all(server->client_fd, &rsp, sizeof(rsp));
}

static void handle_fence_destroy(npm_emu_server * server, const npm_emu_header * hdr) {
    npm_emu_fence_destroy_req req;
    if (!recv_all(server->client_fd, &req, sizeof(req))) {
        return;
    }

    npm_emu_header rsp_hdr;
    npm_emu_header_init(&rsp_hdr, NPM_EMU_CMD_FENCE_DESTROY, hdr->seq_id, sizeof(npm_emu_fence_destroy_rsp));

    npm_emu_fence_destroy_rsp rsp;
    memset(&rsp, 0, sizeof(rsp));
    rsp.status = NPM_EMU_STATUS_OK;

    send_all(server->client_fd, &rsp_hdr, sizeof(rsp_hdr));
    send_all(server->client_fd, &rsp, sizeof(rsp));
}

static void handle_fence_wait(npm_emu_server * server, const npm_emu_header * hdr) {
    npm_emu_fence_wait_req req;
    if (!recv_all(server->client_fd, &req, sizeof(req))) {
        return;
    }

    npm_emu_header rsp_hdr;
    npm_emu_header_init(&rsp_hdr, NPM_EMU_CMD_FENCE_WAIT, hdr->seq_id, sizeof(npm_emu_fence_wait_rsp));

    npm_emu_fence_wait_rsp rsp;
    memset(&rsp, 0, sizeof(rsp));
    rsp.status = NPM_EMU_STATUS_OK;  // Instant completion for now

    send_all(server->client_fd, &rsp_hdr, sizeof(rsp_hdr));
    send_all(server->client_fd, &rsp, sizeof(rsp));
}

// =============================================================================
// Server lifecycle
// =============================================================================

npm_emu_server * npm_emu_server_create(const npm_emu_config * config) {
    npm_emu_server * server = new npm_emu_server;
    // Don't use memset - it corrupts std::unordered_map
    // Initialize members properly instead

    server->config = *config;
    server->listen_fd = -1;
    server->client_fd = -1;
    server->shm_base = nullptr;
    server->shm_size = 0;
    // server->buffers is default-initialized by new
    server->next_handle = 1;
    server->next_fence_id = 1;
    server->total_matmul_ops = 0;
    server->total_bytes_transferred = 0;

    // Get SKU configuration
    const npm_sku_config * sku_config = npm_get_sku_config(config->sku);
    if (!sku_config) {
        delete server;
        return nullptr;
    }

    server->num_engines = sku_config->num_engines;
    server->l1_size = sku_config->l1_size;
    server->l2_size = config->l2_size > 0 ? config->l2_size : sku_config->l2_size_default;

    // Create listen socket
    server->listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server->listen_fd < 0) {
        delete server;
        return nullptr;
    }

    // Bind to socket path
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, config->socket_path, sizeof(addr.sun_path) - 1);

    // Remove existing socket file
    unlink(config->socket_path);

    if (bind(server->listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(server->listen_fd);
        delete server;
        return nullptr;
    }

    if (listen(server->listen_fd, 1) < 0) {
        close(server->listen_fd);
        unlink(config->socket_path);
        delete server;
        return nullptr;
    }

    return server;
}

void npm_emu_server_destroy(npm_emu_server * server) {
    if (!server) {
        return;
    }

    if (server->client_fd >= 0) {
        close(server->client_fd);
    }

    if (server->listen_fd >= 0) {
        close(server->listen_fd);
        unlink(server->config.socket_path);
    }

    if (server->shm_base) {
        munmap(server->shm_base, server->shm_size);
    }

    delete server;
}

int npm_emu_server_run(npm_emu_server * server) {
    printf("[Server] Listening on %s (SKU: %s, L2: %zu MB)\n",
           server->config.socket_path,
           npm_sku_to_string(server->config.sku),
           server->l2_size / (1024 * 1024));
    fflush(stdout);

    while (!g_shutdown_requested) {
        // Accept client connection
        server->client_fd = accept(server->listen_fd, nullptr, nullptr);
        if (server->client_fd < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }

        printf("[Server] Client connected\n");

        // Handle client messages
        while (!g_shutdown_requested) {
            npm_emu_header hdr;
            if (!recv_all(server->client_fd, &hdr, sizeof(hdr))) {
                break;  // Client disconnected
            }

            if (npm_emu_header_validate(&hdr) != 0) {
                fprintf(stderr, "[Server] Invalid message header\n");
                fflush(stderr);
                break;
            }

            switch (hdr.cmd) {
                case NPM_EMU_CMD_HELLO:
                    handle_hello(server, &hdr);
                    break;
                case NPM_EMU_CMD_GOODBYE:
                    handle_goodbye(server, &hdr);
                    goto client_done;
                case NPM_EMU_CMD_REGISTER_BUFFER:
                    handle_register_buffer(server, &hdr);
                    break;
                case NPM_EMU_CMD_UNREGISTER_BUFFER:
                    handle_unregister_buffer(server, &hdr);
                    break;
                case NPM_EMU_CMD_MATMUL:
                    handle_matmul(server, &hdr);
                    break;
                case NPM_EMU_CMD_SYNC:
                    handle_sync(server, &hdr);
                    break;
                case NPM_EMU_CMD_FENCE_CREATE:
                    handle_fence_create(server, &hdr);
                    break;
                case NPM_EMU_CMD_FENCE_DESTROY:
                    handle_fence_destroy(server, &hdr);
                    break;
                case NPM_EMU_CMD_FENCE_WAIT:
                    handle_fence_wait(server, &hdr);
                    break;
                default:
                    fprintf(stderr, "[Server] Unknown command: 0x%02x\n", hdr.cmd);
                    fflush(stderr);
                    break;
            }
        }

    client_done:
        printf("[Server] Client disconnected (matmul ops: %lu)\n",
               (unsigned long)server->total_matmul_ops);
        close(server->client_fd);
        server->client_fd = -1;
    }

    return 0;
}

void npm_emu_server_shutdown(npm_emu_server * server) {
    g_shutdown_requested = 1;
    // Close the listen socket to unblock accept()
    if (server && server->listen_fd >= 0) {
        shutdown(server->listen_fd, SHUT_RDWR);
        close(server->listen_fd);
        unlink(server->config.socket_path);
        server->listen_fd = -1;
    }
}
