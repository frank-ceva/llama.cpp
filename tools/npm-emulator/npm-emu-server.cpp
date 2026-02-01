// NPM Emulator Server Implementation

#include "npm-emu-server.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <ctime>
#include <cmath>
#include <algorithm>
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

    // Trace request
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_COMMANDS)) {
        char details[256];
        snprintf(details, sizeof(details), "{\"version\":\"%d.%d\",\"shm_name\":\"%s\",\"shm_size\":%lu}",
                 req.version_major, req.version_minor, req.shm_name, (unsigned long)req.shm_size);
        npm_trace_command(server->trace_ctx, NPM_TRACE_CMD_HELLO, hdr->seq_id, 0xFF, details);
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

    // Trace response
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_COMMANDS)) {
        char details[256];
        snprintf(details, sizeof(details), "{\"sku\":\"%s\",\"engines\":%d,\"l1_size\":%lu,\"l2_size\":%lu}",
                 npm_sku_to_string((npm_sku)rsp.sku), rsp.num_engines, (unsigned long)rsp.l1_size, (unsigned long)rsp.l2_size);
        npm_trace_command(server->trace_ctx, NPM_TRACE_CMD_HELLO, hdr->seq_id, rsp.status, details);
    }
}

static void handle_goodbye(npm_emu_server * server, const npm_emu_header * hdr) {
    // Trace request
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_COMMANDS)) {
        npm_trace_command(server->trace_ctx, NPM_TRACE_CMD_GOODBYE, hdr->seq_id, 0xFF, NULL);
    }

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

    // Trace response
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_COMMANDS)) {
        npm_trace_command(server->trace_ctx, NPM_TRACE_CMD_GOODBYE, hdr->seq_id, rsp.status, NULL);
    }
}

static void handle_ping(npm_emu_server * server, const npm_emu_header * hdr) {
    npm_emu_ping_req req;
    if (!recv_all(server->client_fd, &req, sizeof(req))) {
        return;
    }

    // Trace request
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_COMMANDS)) {
        char details[256];
        snprintf(details, sizeof(details), "{\"echo_data\":\"0x%016lx\",\"timestamp\":%lu}",
                 (unsigned long)req.echo_data, (unsigned long)req.timestamp);
        npm_trace_command(server->trace_ctx, NPM_TRACE_CMD_PING, hdr->seq_id, 0xFF, details);
    }

    if (server->config.verbose) {
        printf("[Server] PING: echo_data=0x%016lx timestamp=%lu\n",
               (unsigned long)req.echo_data, (unsigned long)req.timestamp);
    }

    // Get current timestamp (nanoseconds since epoch)
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t server_timestamp = ts.tv_sec * 1000000000ULL + ts.tv_nsec;

    // Send response
    npm_emu_header rsp_hdr;
    npm_emu_header_init(&rsp_hdr, NPM_EMU_CMD_PING, hdr->seq_id, sizeof(npm_emu_ping_rsp));

    npm_emu_ping_rsp rsp;
    memset(&rsp, 0, sizeof(rsp));
    rsp.status = NPM_EMU_STATUS_OK;
    rsp.client_timestamp = req.timestamp;
    rsp.server_timestamp = server_timestamp;
    rsp.echo_data = req.echo_data;

    send_all(server->client_fd, &rsp_hdr, sizeof(rsp_hdr));
    send_all(server->client_fd, &rsp, sizeof(rsp));

    // Trace response
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_COMMANDS)) {
        char details[256];
        snprintf(details, sizeof(details), "{\"client_timestamp\":%lu,\"server_timestamp\":%lu,\"echo_data\":\"0x%016lx\"}",
                 (unsigned long)rsp.client_timestamp, (unsigned long)rsp.server_timestamp, (unsigned long)rsp.echo_data);
        npm_trace_command(server->trace_ctx, NPM_TRACE_CMD_PING, hdr->seq_id, rsp.status, details);
    }
}

static void handle_register_buffer(npm_emu_server * server, const npm_emu_header * hdr) {
    npm_emu_register_buffer_req req;
    if (!recv_all(server->client_fd, &req, sizeof(req))) {
        return;
    }

    // Trace request
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_COMMANDS)) {
        char details[256];
        snprintf(details, sizeof(details), "{\"shm_offset\":%lu,\"size\":%lu,\"flags\":\"0x%x\"}",
                 (unsigned long)req.shm_offset, (unsigned long)req.size, req.flags);
        npm_trace_command(server->trace_ctx, NPM_TRACE_CMD_REGISTER_BUFFER, hdr->seq_id, 0xFF, details);
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

    // Trace response
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_COMMANDS)) {
        char details[256];
        snprintf(details, sizeof(details), "{\"handle\":%lu}", (unsigned long)rsp.handle);
        npm_trace_command(server->trace_ctx, NPM_TRACE_CMD_REGISTER_BUFFER, hdr->seq_id, rsp.status, details);
    }
}

static void handle_unregister_buffer(npm_emu_server * server, const npm_emu_header * hdr) {
    npm_emu_unregister_buffer_req req;
    if (!recv_all(server->client_fd, &req, sizeof(req))) {
        return;
    }

    // Trace request
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_COMMANDS)) {
        char details[256];
        snprintf(details, sizeof(details), "{\"handle\":%lu}", (unsigned long)req.handle);
        npm_trace_command(server->trace_ctx, NPM_TRACE_CMD_UNREGISTER_BUFFER, hdr->seq_id, 0xFF, details);
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

    // Trace response
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_COMMANDS)) {
        npm_trace_command(server->trace_ctx, NPM_TRACE_CMD_UNREGISTER_BUFFER, hdr->seq_id, rsp.status, NULL);
    }
}

// Calculate tile size based on L1 capacity
// For FP32: L1 can hold elements = L1_size / 4 bytes
// Need to fit 3 tiles (A, B, C) for accumulation
static int calculate_tile_size(size_t l1_size) {
    // Each float is 4 bytes
    // We need space for: A tile (tile_m * tile_k) + B tile (tile_n * tile_k) + C tile (tile_m * tile_n)
    // Simplified: assume square tiles and 3 matrices: 3 * tile^2 elements
    size_t elements = l1_size / sizeof(float);
    size_t tile_elements = elements / 3;
    int tile_size = (int)sqrt((double)tile_elements);

    // Round down to power of 2 for alignment (min 32)
    tile_size = std::max(32, tile_size);
    int pot = 1;
    while (pot * 2 <= tile_size) pot *= 2;
    return pot;
}

static void handle_matmul(npm_emu_server * server, const npm_emu_header * hdr) {
    npm_emu_matmul_req req;
    if (!recv_all(server->client_fd, &req, sizeof(req))) {
        return;
    }

    // Trace request with buffer sizes and destination
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_COMMANDS)) {
        // Look up buffer sizes
        size_t a_size = 0, b_size = 0, c_size = 0;
        auto it_a = server->buffers.find(req.a_handle);
        if (it_a != server->buffers.end()) a_size = it_a->second.size;
        auto it_b = server->buffers.find(req.b_handle);
        if (it_b != server->buffers.end()) b_size = it_b->second.size;
        auto it_c = server->buffers.find(req.c_handle);
        if (it_c != server->buffers.end()) c_size = it_c->second.size;

        char details[512];
        snprintf(details, sizeof(details),
                 "{\"M\":%ld,\"N\":%ld,\"K\":%ld,"
                 "\"a_handle\":%lu,\"b_handle\":%lu,\"c_handle\":%lu,"
                 "\"a_size\":%zu,\"b_size\":%zu,\"c_size\":%zu,"
                 "\"destination\":\"NPM\"}",
                 (long)req.M, (long)req.N, (long)req.K,
                 (unsigned long)req.a_handle, (unsigned long)req.b_handle, (unsigned long)req.c_handle,
                 a_size, b_size, c_size);
        npm_trace_command(server->trace_ctx, NPM_TRACE_CMD_MATMUL, hdr->seq_id, 0xFF, details);
    }

    if (server->config.verbose) {
        printf("[Server] MATMUL M=%ld N=%ld K=%ld (tiling=%s, timing=%s)\n",
               (long)req.M, (long)req.N, (long)req.K,
               server->config.tiling_enabled ? "on" : "off",
               server->config.timing_enabled ? "on" : "off");
    }

    // Calculate tile_size early so we can include it in MATMUL_START trace
    int tile_size = 0;
    if (server->config.tiling_enabled) {
        tile_size = calculate_tile_size(server->l1_size);
    }

    // Trace: MATMUL_START with tile_size and memory config
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_OPS)) {
        char details[256];
        snprintf(details, sizeof(details),
                 "{\"tiling\":%s,\"timing\":%s,\"tile_size\":%d,\"l1_size\":%zu,\"l2_size\":%zu}",
                 server->config.tiling_enabled ? "true" : "false",
                 server->config.timing_enabled ? "true" : "false",
                 tile_size, server->l1_size, server->l2_size);
        npm_trace_op(server->trace_ctx, NPM_TRACE_OP_MATMUL_START, req.M, req.N, req.K, 0, details);
    }

    uint8_t status = NPM_EMU_STATUS_OK;
    uint64_t total_cycles = 0;
    uint64_t total_dma_bytes = 0;

    // Resolve buffer handles to pointers
    const float * A = (const float *)resolve_handle(server, req.a_handle, req.a_offset);
    const float * B = (const float *)resolve_handle(server, req.b_handle, req.b_offset);
    float * C = (float *)resolve_handle(server, req.c_handle, req.c_offset);

    if (!A || !B || !C) {
        status = NPM_EMU_STATUS_INVALID_HANDLE;
    } else {
        if (server->config.tiling_enabled && server->dma_model && server->mem_hierarchy) {
            // Tiled execution with DMA simulation and L2 cache awareness
            // (tile_size already calculated above for MATMUL_START trace)

            // Get SKU config for MACs/cycle
            const npm_sku_config * sku_cfg = npm_get_sku_config(server->config.sku);
            int64_t fp32_macs_per_cycle = sku_cfg ? sku_cfg->fp16_macs / 2 : 2000;  // FP32 ~= FP16/2

            server->dma_model->reset_stats();
            server->mem_hierarchy->reset();

            // Trace: TILING_PLAN - comprehensive tiling strategy summary
            if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_OPS)) {
                int num_m_tiles = (req.M + tile_size - 1) / tile_size;
                int num_n_tiles = (req.N + tile_size - 1) / tile_size;
                int num_k_tiles = (req.K + tile_size - 1) / tile_size;
                size_t a_total = req.M * req.K * sizeof(float);
                size_t b_total = req.N * req.K * sizeof(float);
                size_t c_total = req.M * req.N * sizeof(float);

                char details[512];
                snprintf(details, sizeof(details),
                         "{\"tile_size\":%d,"
                         "\"num_m_tiles\":%d,\"num_n_tiles\":%d,\"num_k_tiles\":%d,"
                         "\"total_tiles\":%d,"
                         "\"a_total_bytes\":%zu,\"b_total_bytes\":%zu,\"c_total_bytes\":%zu}",
                         tile_size, num_m_tiles, num_n_tiles, num_k_tiles,
                         num_m_tiles * num_n_tiles,
                         a_total, b_total, c_total);
                npm_trace_op(server->trace_ctx, NPM_TRACE_OP_TILING_PLAN,
                             req.M, req.N, req.K, 0, details);
            }

            // Tiled matmul: C = A * B^T
            for (int64_t m_tile = 0; m_tile < req.M; m_tile += tile_size) {
                for (int64_t n_tile = 0; n_tile < req.N; n_tile += tile_size) {
                    int64_t actual_m = std::min((int64_t)tile_size, req.M - m_tile);
                    int64_t actual_n = std::min((int64_t)tile_size, req.N - n_tile);

                    // Initialize C tile to zero
                    for (int64_t m = 0; m < actual_m; m++) {
                        for (int64_t n = 0; n < actual_n; n++) {
                            C[(m_tile + m) * req.ldc + (n_tile + n)] = 0.0f;
                        }
                    }

                    // Accumulate over K tiles
                    for (int64_t k_tile = 0; k_tile < req.K; k_tile += tile_size) {
                        int64_t actual_k = std::min((int64_t)tile_size, req.K - k_tile);

                        // Calculate tile byte offsets for cache tracking
                        // A tile: rows [m_tile, m_tile+actual_m), cols [k_tile, k_tile+actual_k)
                        size_t a_tile_byte_offset = (m_tile * req.lda + k_tile) * sizeof(float);
                        size_t a_tile_bytes = actual_m * actual_k * sizeof(float);

                        // B tile: rows [n_tile, n_tile+actual_n), cols [k_tile, k_tile+actual_k)
                        size_t b_tile_byte_offset = (n_tile * req.ldb + k_tile) * sizeof(float);
                        size_t b_tile_bytes = actual_n * actual_k * sizeof(float);

                        // Stage A tile: DDR -> L2 (with cache check)
                        uint64_t l2_misses_before = server->mem_hierarchy->get_l2_misses();
                        // Cast away const - stage_to_l2 only reads the data to copy into L2
                        server->mem_hierarchy->stage_to_l2(req.a_handle, a_tile_byte_offset, a_tile_bytes,
                                                          const_cast<float*>(A + m_tile * req.lda + k_tile));
                        bool a_l2_miss = (server->mem_hierarchy->get_l2_misses() > l2_misses_before);

                        // Only count DMA cycles if it was a cache miss
                        if (a_l2_miss) {
                            server->dma_model->transfer(NPM_DMA_DDR_TO_L2, a_tile_bytes);
                            if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_DMA)) {
                                npm_trace_dma(server->trace_ctx, NPM_TRACE_DMA_DDR_TO_L2, a_tile_bytes,
                                              server->dma_model->get_current_cycle(), -1);
                            }
                        }
                        // L2 -> L1 transfer (always needed for computation)
                        server->mem_hierarchy->stage_to_l1(0, req.a_handle, a_tile_byte_offset, a_tile_bytes);
                        server->dma_model->transfer(NPM_DMA_L2_TO_L1, a_tile_bytes);
                        if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_DMA)) {
                            npm_trace_dma(server->trace_ctx, NPM_TRACE_DMA_L2_TO_L1, a_tile_bytes,
                                          server->dma_model->get_current_cycle(), 0);
                        }

                        // Stage B tile: DDR -> L2 (with cache check)
                        l2_misses_before = server->mem_hierarchy->get_l2_misses();
                        server->mem_hierarchy->stage_to_l2(req.b_handle, b_tile_byte_offset, b_tile_bytes,
                                                          const_cast<float*>(B + n_tile * req.ldb + k_tile));
                        bool b_l2_miss = (server->mem_hierarchy->get_l2_misses() > l2_misses_before);

                        if (b_l2_miss) {
                            server->dma_model->transfer(NPM_DMA_DDR_TO_L2, b_tile_bytes);
                            if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_DMA)) {
                                npm_trace_dma(server->trace_ctx, NPM_TRACE_DMA_DDR_TO_L2, b_tile_bytes,
                                              server->dma_model->get_current_cycle(), -1);
                            }
                        }
                        server->mem_hierarchy->stage_to_l1(0, req.b_handle, b_tile_byte_offset, b_tile_bytes);
                        server->dma_model->transfer(NPM_DMA_L2_TO_L1, b_tile_bytes);
                        if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_DMA)) {
                            npm_trace_dma(server->trace_ctx, NPM_TRACE_DMA_L2_TO_L1, b_tile_bytes,
                                          server->dma_model->get_current_cycle(), 0);
                        }

                        // Compute: C_tile += A_tile * B_tile^T
                        // Actual computation
                        for (int64_t m = 0; m < actual_m; m++) {
                            for (int64_t n = 0; n < actual_n; n++) {
                                float sum = 0.0f;
                                for (int64_t k = 0; k < actual_k; k++) {
                                    sum += A[(m_tile + m) * req.lda + (k_tile + k)] *
                                           B[(n_tile + n) * req.ldb + (k_tile + k)];
                                }
                                C[(m_tile + m) * req.ldc + (n_tile + n)] += sum;
                            }
                        }

                        // Compute cycles (only if timing enabled)
                        uint64_t compute_cycles = 0;
                        if (server->config.timing_enabled) {
                            int64_t ops = 2 * actual_m * actual_n * actual_k;
                            compute_cycles = (ops + fp32_macs_per_cycle - 1) / fp32_macs_per_cycle;
                            server->dma_model->advance_cycles(compute_cycles);
                        }

                        // Trace: MATMUL_TILE with actual dimensions, byte sizes, and cache hit info
                        if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_OPS)) {
                            char details[384];
                            snprintf(details, sizeof(details),
                                     "{\"m_off\":%lld,\"n_off\":%lld,\"k_off\":%lld,"
                                     "\"actual_m\":%lld,\"actual_n\":%lld,\"actual_k\":%lld,"
                                     "\"a_tile_bytes\":%zu,\"b_tile_bytes\":%zu,"
                                     "\"a_l2_hit\":%s,\"b_l2_hit\":%s}",
                                     (long long)m_tile, (long long)n_tile, (long long)k_tile,
                                     (long long)actual_m, (long long)actual_n, (long long)actual_k,
                                     a_tile_bytes, b_tile_bytes,
                                     a_l2_miss ? "false" : "true", b_l2_miss ? "false" : "true");
                            npm_trace_op(server->trace_ctx, NPM_TRACE_OP_MATMUL_TILE,
                                         actual_m, actual_n, actual_k, compute_cycles, details);
                        }
                    }

                    // C tile writeback: L1 -> L2 -> DDR
                    size_t c_tile_byte_offset = (m_tile * req.ldc + n_tile) * sizeof(float);
                    (void)c_tile_byte_offset;  // Used for documentation, may be used for cache tracking later
                    size_t c_tile_bytes = actual_m * actual_n * sizeof(float);
                    server->dma_model->transfer(NPM_DMA_L1_TO_L2, c_tile_bytes);
                    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_DMA)) {
                        npm_trace_dma(server->trace_ctx, NPM_TRACE_DMA_L1_TO_L2, c_tile_bytes,
                                      server->dma_model->get_current_cycle(), 0);
                    }
                    server->dma_model->transfer(NPM_DMA_L2_TO_DDR, c_tile_bytes);
                    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_DMA)) {
                        npm_trace_dma(server->trace_ctx, NPM_TRACE_DMA_L2_TO_DDR, c_tile_bytes,
                                      server->dma_model->get_current_cycle(), -1);
                    }
                }
            }

            // Get DMA stats (always available with tiling)
            total_dma_bytes = server->dma_model->get_total_bytes_transferred();
            // Only report cycles if timing is enabled
            if (server->config.timing_enabled) {
                total_cycles = server->dma_model->get_current_cycle();
            }

            // Get cache statistics
            uint64_t l2_hits = server->mem_hierarchy->get_l2_hits();
            uint64_t l2_misses = server->mem_hierarchy->get_l2_misses();

            // Trace: MATMUL_END with cache stats, DMA totals, and tile_size
            if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_OPS)) {
                char details[256];
                snprintf(details, sizeof(details),
                         "{\"l2_hits\":%lu,\"l2_misses\":%lu,"
                         "\"total_dma_bytes\":%lu,\"tile_size\":%d}",
                         (unsigned long)l2_hits, (unsigned long)l2_misses,
                         (unsigned long)total_dma_bytes, tile_size);
                npm_trace_op(server->trace_ctx, NPM_TRACE_OP_MATMUL_END, req.M, req.N, req.K, total_cycles, details);
            }

            if (server->config.verbose) {
                printf("[Server] MATMUL tiled: %lu bytes DMA, tile=%d, L2 hits=%lu, misses=%lu",
                       (unsigned long)total_dma_bytes, tile_size,
                       (unsigned long)l2_hits, (unsigned long)l2_misses);
                if (server->config.timing_enabled) {
                    printf(", cycles=%lu", (unsigned long)total_cycles);
                }
                printf("\n");
            }
        } else {
            // Simple execution without timing (same as before)
            // C = A * B^T, A: (M, K), B: (N, K), C: (M, N)
            for (int64_t m = 0; m < req.M; m++) {
                for (int64_t n = 0; n < req.N; n++) {
                    float sum = 0.0f;
                    for (int64_t k = 0; k < req.K; k++) {
                        sum += A[m * req.lda + k] * B[n * req.ldb + k];
                    }
                    C[m * req.ldc + n] = sum;
                }
            }
        }

        server->total_matmul_ops++;
    }

    npm_emu_header rsp_hdr;
    npm_emu_header_init(&rsp_hdr, NPM_EMU_CMD_MATMUL, hdr->seq_id, sizeof(npm_emu_matmul_rsp));

    npm_emu_matmul_rsp rsp;
    memset(&rsp, 0, sizeof(rsp));
    rsp.status = status;
    rsp.cycles = total_cycles;
    rsp.dma_bytes = total_dma_bytes;

    send_all(server->client_fd, &rsp_hdr, sizeof(rsp_hdr));
    send_all(server->client_fd, &rsp, sizeof(rsp));

    // Trace response
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_COMMANDS)) {
        char details[256];
        snprintf(details, sizeof(details), "{\"cycles\":%lu,\"dma_bytes\":%lu}",
                 (unsigned long)rsp.cycles, (unsigned long)rsp.dma_bytes);
        npm_trace_command(server->trace_ctx, NPM_TRACE_CMD_MATMUL, hdr->seq_id, rsp.status, details);
    }
}

static void handle_sync(npm_emu_server * server, const npm_emu_header * hdr) {
    // Trace request
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_COMMANDS)) {
        npm_trace_command(server->trace_ctx, NPM_TRACE_CMD_SYNC, hdr->seq_id, 0xFF, NULL);
    }

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

    // Trace response
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_COMMANDS)) {
        npm_trace_command(server->trace_ctx, NPM_TRACE_CMD_SYNC, hdr->seq_id, rsp.status, NULL);
    }
}

static void handle_fence_create(npm_emu_server * server, const npm_emu_header * hdr) {
    // Trace request
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_COMMANDS)) {
        npm_trace_command(server->trace_ctx, NPM_TRACE_CMD_FENCE_CREATE, hdr->seq_id, 0xFF, NULL);
    }

    uint64_t fence_id = server->next_fence_id++;

    npm_emu_header rsp_hdr;
    npm_emu_header_init(&rsp_hdr, NPM_EMU_CMD_FENCE_CREATE, hdr->seq_id, sizeof(npm_emu_fence_create_rsp));

    npm_emu_fence_create_rsp rsp;
    memset(&rsp, 0, sizeof(rsp));
    rsp.status = NPM_EMU_STATUS_OK;
    rsp.fence_id = fence_id;

    send_all(server->client_fd, &rsp_hdr, sizeof(rsp_hdr));
    send_all(server->client_fd, &rsp, sizeof(rsp));

    // Trace response
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_COMMANDS)) {
        char details[128];
        snprintf(details, sizeof(details), "{\"fence_id\":%lu}", (unsigned long)rsp.fence_id);
        npm_trace_command(server->trace_ctx, NPM_TRACE_CMD_FENCE_CREATE, hdr->seq_id, rsp.status, details);
    }
}

static void handle_fence_destroy(npm_emu_server * server, const npm_emu_header * hdr) {
    npm_emu_fence_destroy_req req;
    if (!recv_all(server->client_fd, &req, sizeof(req))) {
        return;
    }

    // Trace request
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_COMMANDS)) {
        char details[128];
        snprintf(details, sizeof(details), "{\"fence_id\":%lu}", (unsigned long)req.fence_id);
        npm_trace_command(server->trace_ctx, NPM_TRACE_CMD_FENCE_DESTROY, hdr->seq_id, 0xFF, details);
    }

    npm_emu_header rsp_hdr;
    npm_emu_header_init(&rsp_hdr, NPM_EMU_CMD_FENCE_DESTROY, hdr->seq_id, sizeof(npm_emu_fence_destroy_rsp));

    npm_emu_fence_destroy_rsp rsp;
    memset(&rsp, 0, sizeof(rsp));
    rsp.status = NPM_EMU_STATUS_OK;

    send_all(server->client_fd, &rsp_hdr, sizeof(rsp_hdr));
    send_all(server->client_fd, &rsp, sizeof(rsp));

    // Trace response
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_COMMANDS)) {
        npm_trace_command(server->trace_ctx, NPM_TRACE_CMD_FENCE_DESTROY, hdr->seq_id, rsp.status, NULL);
    }
}

static void handle_fence_wait(npm_emu_server * server, const npm_emu_header * hdr) {
    npm_emu_fence_wait_req req;
    if (!recv_all(server->client_fd, &req, sizeof(req))) {
        return;
    }

    // Trace request
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_COMMANDS)) {
        char details[128];
        snprintf(details, sizeof(details), "{\"fence_id\":%lu,\"timeout_ns\":%lu}",
                 (unsigned long)req.fence_id, (unsigned long)req.timeout_ns);
        npm_trace_command(server->trace_ctx, NPM_TRACE_CMD_FENCE_WAIT, hdr->seq_id, 0xFF, details);
    }

    npm_emu_header rsp_hdr;
    npm_emu_header_init(&rsp_hdr, NPM_EMU_CMD_FENCE_WAIT, hdr->seq_id, sizeof(npm_emu_fence_wait_rsp));

    npm_emu_fence_wait_rsp rsp;
    memset(&rsp, 0, sizeof(rsp));
    rsp.status = NPM_EMU_STATUS_OK;  // Instant completion for now

    send_all(server->client_fd, &rsp_hdr, sizeof(rsp_hdr));
    send_all(server->client_fd, &rsp, sizeof(rsp));

    // Trace response
    if (npm_trace_enabled(server->trace_ctx, NPM_TRACE_COMMANDS)) {
        npm_trace_command(server->trace_ctx, NPM_TRACE_CMD_FENCE_WAIT, hdr->seq_id, rsp.status, NULL);
    }
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

    // Initialize trace context from config
    npm_trace_config trace_config = {};
    trace_config.categories = config->trace_categories;
    trace_config.flush_immediate = true;

    // Open trace file if specified, otherwise use stdout
    if (config->trace_file && config->trace_file[0] != '\0') {
        trace_config.output = fopen(config->trace_file, "w");
        if (!trace_config.output) {
            fprintf(stderr, "Warning: Could not open trace file %s, using stdout\n", config->trace_file);
            trace_config.output = nullptr; // will default to stdout
        }
    } else {
        trace_config.output = nullptr; // defaults to stdout in npm_trace_create
    }
    server->trace_ctx = npm_trace_create(&trace_config);

    // Get SKU configuration
    const npm_sku_config * sku_config = npm_get_sku_config(config->sku);
    if (!sku_config) {
        delete server;
        return nullptr;
    }

    server->num_engines = sku_config->num_engines;
    server->l1_size = sku_config->l1_size;
    server->l2_size = config->l2_size > 0 ? config->l2_size : sku_config->l2_size_default;

    // Initialize memory hierarchy and DMA models
    server->mem_hierarchy = std::make_unique<npm_memory_hierarchy>(
        server->num_engines, server->l1_size, server->l2_size);

    npm_dma_config dma_config;
    // Use default bandwidth values from npm_dma_config constructor
    server->dma_model = std::make_unique<npm_dma_model>(dma_config);

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

    // Destroy trace context
    npm_trace_destroy(server->trace_ctx);

    delete server;
}

static void print_startup_banner(const npm_emu_server * server) {
    const npm_sku_config * cfg = npm_get_sku_config(server->config.sku);

    printf("\n");
    printf("+---------------------------------------------------------+\n");
    printf("|           NPM Hardware Emulator v%d.%d                    |\n",
           NPM_EMU_VERSION_MAJOR, NPM_EMU_VERSION_MINOR);
    printf("+---------------------------------------------------------+\n");
    printf("|  SKU:         %-10s                                |\n", npm_sku_to_string(server->config.sku));
    printf("|  Engines:     %-3d                                       |\n", server->num_engines);
    printf("|  L1 Size:     %-4zu KB (per engine)                      |\n", server->l1_size / 1024);
    printf("|  L2 Size:     %-4zu MB (shared)                          |\n", server->l2_size / (1024*1024));
    if (cfg && cfg->int4_macs > 0) {
        printf("|  INT4 MACs:   %-6ld /cycle                             |\n", (long)cfg->int4_macs);
        printf("|  INT8 MACs:   %-6ld /cycle                             |\n", (long)cfg->int8_macs);
        printf("|  FP16 MACs:   %-6ld /cycle                             |\n", (long)cfg->fp16_macs);
    }
    printf("+---------------------------------------------------------+\n");
    printf("|  Socket:      %-39s  |\n", server->config.socket_path);
    printf("|  Tiling:      %-8s                                  |\n", server->config.tiling_enabled ? "enabled" : "disabled");
    printf("|  Timing:      %-8s                                  |\n", server->config.timing_enabled ? "enabled" : "disabled");
    printf("|  Verbose:     %-8s                                  |\n", server->config.verbose ? "enabled" : "disabled");
    printf("+---------------------------------------------------------+\n");
    printf("\n");
    fflush(stdout);
}

int npm_emu_server_run(npm_emu_server * server) {
    print_startup_banner(server);

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
                break;
            }

            switch (hdr.cmd) {
                case NPM_EMU_CMD_HELLO:
                    handle_hello(server, &hdr);
                    break;
                case NPM_EMU_CMD_GOODBYE:
                    handle_goodbye(server, &hdr);
                    goto client_done;
                case NPM_EMU_CMD_PING:
                    handle_ping(server, &hdr);
                    break;
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
    (void)server;
    g_shutdown_requested = 1;
}
