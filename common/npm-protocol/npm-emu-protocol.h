#pragma once

// NPM Emulator IPC Protocol
//
// Binary protocol for communication between the NPM device driver
// (npm-device-emulator.cpp) and the NPM emulator process (npm-emulator).
//
// Communication uses Unix domain sockets with shared memory for data.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Protocol constants
// =============================================================================

#define NPM_EMU_MAGIC          0x454D504E  // "NPME" in little-endian
#define NPM_EMU_VERSION_MAJOR  1
#define NPM_EMU_VERSION_MINOR  0

#define NPM_EMU_DEFAULT_SOCKET "/tmp/npm-emulator.sock"
#define NPM_EMU_MAX_SHM_NAME   64

// =============================================================================
// Command enumeration
// =============================================================================

enum npm_emu_cmd {
    // Connection lifecycle
    NPM_EMU_CMD_HELLO             = 0x00,  // Version handshake + shared memory setup
    NPM_EMU_CMD_GOODBYE           = 0x01,  // Clean disconnect
    NPM_EMU_CMD_PING              = 0x02,  // Keep-alive

    // Device info
    NPM_EMU_CMD_GET_CONFIG        = 0x10,  // Get SKU, engine count, memory sizes

    // Buffer management
    NPM_EMU_CMD_REGISTER_BUFFER   = 0x20,  // Register a buffer (shm offset, size)
    NPM_EMU_CMD_UNREGISTER_BUFFER = 0x21,  // Unregister a buffer

    // Compute operations
    NPM_EMU_CMD_MATMUL            = 0x30,  // Matrix multiplication

    // Synchronization
    NPM_EMU_CMD_SYNC              = 0x40,  // Global sync
    NPM_EMU_CMD_FENCE_CREATE      = 0x41,  // Create fence
    NPM_EMU_CMD_FENCE_DESTROY     = 0x42,  // Destroy fence
    NPM_EMU_CMD_FENCE_WAIT        = 0x43,  // Wait on fence
};

// =============================================================================
// Response status codes
// =============================================================================

enum npm_emu_status {
    NPM_EMU_STATUS_OK             = 0x00,
    NPM_EMU_STATUS_ERROR          = 0x01,
    NPM_EMU_STATUS_VERSION_MISMATCH = 0x02,
    NPM_EMU_STATUS_INVALID_HANDLE = 0x03,
    NPM_EMU_STATUS_OUT_OF_MEMORY  = 0x04,
    NPM_EMU_STATUS_INVALID_PARAMS = 0x05,
    NPM_EMU_STATUS_TIMEOUT        = 0x06,
};

// =============================================================================
// Message header (all messages start with this)
// =============================================================================

#pragma pack(push, 1)

struct npm_emu_header {
    uint32_t magic;           // NPM_EMU_MAGIC
    uint8_t  version_major;   // Protocol version major
    uint8_t  version_minor;   // Protocol version minor
    uint8_t  cmd;             // enum npm_emu_cmd
    uint8_t  flags;           // Reserved (0 for now)
    uint32_t seq_id;          // Monotonic sequence ID for request/response matching
    uint32_t payload_size;    // Size of payload following this header
};

// =============================================================================
// HELLO command - establishes connection and shared memory
// =============================================================================

// Request: client sends its version and shared memory name
struct npm_emu_hello_req {
    uint8_t  version_major;
    uint8_t  version_minor;
    uint8_t  reserved[2];
    char     shm_name[NPM_EMU_MAX_SHM_NAME];  // Shared memory region name (e.g., "/npm-shm-12345")
    uint64_t shm_size;                         // Size of shared memory region
};

// Response: server sends device info
struct npm_emu_hello_rsp {
    uint8_t  status;          // npm_emu_status
    uint8_t  version_major;
    uint8_t  version_minor;
    uint8_t  reserved;
    uint32_t sku;             // enum npm_sku
    uint32_t num_engines;
    uint64_t l1_size;         // L1 size per engine
    uint64_t l2_size;         // L2 size (total)
};

// =============================================================================
// GOODBYE command - clean disconnect
// =============================================================================

// Request: no payload needed (just header)
// Response: no payload (just header with status)

struct npm_emu_goodbye_rsp {
    uint8_t  status;
    uint8_t  reserved[3];
};

// =============================================================================
// GET_CONFIG command - query device configuration
// =============================================================================

// Request: no payload needed
// Response: same as hello_rsp (device info)

// =============================================================================
// REGISTER_BUFFER command
// =============================================================================

// Request: register a buffer in shared memory
struct npm_emu_register_buffer_req {
    uint64_t shm_offset;      // Offset within the shared memory region
    uint64_t size;            // Size of the buffer
    uint32_t flags;           // enum npm_alloc_flags
    uint32_t reserved;
};

// Response: returns handle
struct npm_emu_register_buffer_rsp {
    uint8_t  status;
    uint8_t  reserved[3];
    uint64_t handle;          // Opaque handle for use in compute operations
};

// =============================================================================
// UNREGISTER_BUFFER command
// =============================================================================

// Request: unregister a buffer
struct npm_emu_unregister_buffer_req {
    uint64_t handle;
};

// Response: status only
struct npm_emu_unregister_buffer_rsp {
    uint8_t  status;
    uint8_t  reserved[3];
};

// =============================================================================
// MATMUL command
// =============================================================================

// Request: execute matrix multiplication
struct npm_emu_matmul_req {
    uint64_t a_handle;        // Input A buffer handle
    uint64_t a_offset;        // Offset within A buffer
    uint64_t b_handle;        // Input B buffer handle
    uint64_t b_offset;        // Offset within B buffer
    uint64_t c_handle;        // Output C buffer handle
    uint64_t c_offset;        // Offset within C buffer
    int64_t  M;               // Rows of A and C
    int64_t  N;               // Cols of B and C
    int64_t  K;               // Inner dimension
    int64_t  lda;             // Leading dimension of A
    int64_t  ldb;             // Leading dimension of B
    int64_t  ldc;             // Leading dimension of C
    uint32_t type_a;          // ggml_type
    uint32_t type_b;
    uint32_t type_c;
    uint32_t flags;           // Reserved
};

// Response: status and optional timing info
struct npm_emu_matmul_rsp {
    uint8_t  status;
    uint8_t  reserved[3];
    uint64_t cycles;          // Simulated cycle count (if timing enabled)
    uint64_t dma_bytes;       // Total DMA traffic (for debugging)
};

// =============================================================================
// SYNC command
// =============================================================================

// Request: no payload needed
// Response: status only

struct npm_emu_sync_rsp {
    uint8_t  status;
    uint8_t  reserved[3];
};

// =============================================================================
// FENCE commands
// =============================================================================

// FENCE_CREATE request: no payload
// FENCE_CREATE response:
struct npm_emu_fence_create_rsp {
    uint8_t  status;
    uint8_t  reserved[3];
    uint64_t fence_id;
};

// FENCE_DESTROY request:
struct npm_emu_fence_destroy_req {
    uint64_t fence_id;
};

// FENCE_DESTROY response: status only
struct npm_emu_fence_destroy_rsp {
    uint8_t  status;
    uint8_t  reserved[3];
};

// FENCE_WAIT request:
struct npm_emu_fence_wait_req {
    uint64_t fence_id;
    uint64_t timeout_ns;      // Timeout in nanoseconds (0 = infinite)
};

// FENCE_WAIT response: status only
struct npm_emu_fence_wait_rsp {
    uint8_t  status;
    uint8_t  reserved[3];
};

#pragma pack(pop)

// =============================================================================
// Helper functions
// =============================================================================

// Initialize a message header
static inline void npm_emu_header_init(struct npm_emu_header * hdr,
                                       enum npm_emu_cmd cmd,
                                       uint32_t seq_id,
                                       uint32_t payload_size) {
    hdr->magic = NPM_EMU_MAGIC;
    hdr->version_major = NPM_EMU_VERSION_MAJOR;
    hdr->version_minor = NPM_EMU_VERSION_MINOR;
    hdr->cmd = cmd;
    hdr->flags = 0;
    hdr->seq_id = seq_id;
    hdr->payload_size = payload_size;
}

// Validate a message header
static inline int npm_emu_header_validate(const struct npm_emu_header * hdr) {
    if (hdr->magic != NPM_EMU_MAGIC) {
        return -1;
    }
    if (hdr->version_major != NPM_EMU_VERSION_MAJOR) {
        return -2;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif
