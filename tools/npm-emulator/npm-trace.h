#pragma once

/**
 * NPM Emulator Tracing System
 *
 * Provides structured JSON tracing for debugging and analysis.
 *
 * Usage:
 *   npm_trace_config config = {};
 *   config.categories = NPM_TRACE_COMMANDS | NPM_TRACE_DMA;
 *   npm_trace_ctx* ctx = npm_trace_create(&config);
 *
 *   // Check before formatting (zero-overhead when disabled)
 *   if (npm_trace_enabled(ctx, NPM_TRACE_COMMANDS)) {
 *       npm_trace_command(ctx, NPM_TRACE_CMD_MATMUL, seq_id, status, details);
 *   }
 *
 *   npm_trace_destroy(ctx);
 *
 * JSON Output Schema:
 *   Command: {"ts":123,"cat":"cmd","type":"MATMUL","seq":42,"status":"OK","details":{...}}
 *   DMA:     {"ts":123,"cat":"dma","type":"DDR_TO_L2","bytes":4096,"cycles":64,"engine":-1}
 *   Op:      {"ts":123,"cat":"op","type":"MATMUL_END","M":64,"N":128,"K":64,"cycles":8192}
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Trace Categories (bit flags for combining)
// =============================================================================

typedef enum npm_trace_category {
    NPM_TRACE_NONE     = 0,
    NPM_TRACE_COMMANDS = 1 << 0,  // IPC command flow
    NPM_TRACE_DMA      = 1 << 1,  // DMA transfers
    NPM_TRACE_OPS      = 1 << 2,  // Compute operations
    NPM_TRACE_ALL      = 0xFFFFFFFF
} npm_trace_category;

// =============================================================================
// Trace Event Types
// =============================================================================

typedef enum npm_trace_event_type {
    // Commands (request/response pairs)
    NPM_TRACE_CMD_HELLO,
    NPM_TRACE_CMD_GOODBYE,
    NPM_TRACE_CMD_PING,
    NPM_TRACE_CMD_REGISTER_BUFFER,
    NPM_TRACE_CMD_UNREGISTER_BUFFER,
    NPM_TRACE_CMD_MATMUL,
    NPM_TRACE_CMD_SYNC,
    NPM_TRACE_CMD_FENCE_CREATE,
    NPM_TRACE_CMD_FENCE_DESTROY,
    NPM_TRACE_CMD_FENCE_WAIT,

    // DMA transfers
    NPM_TRACE_DMA_DDR_TO_L2,
    NPM_TRACE_DMA_L2_TO_DDR,
    NPM_TRACE_DMA_L2_TO_L1,
    NPM_TRACE_DMA_L1_TO_L2,

    // Compute operations
    NPM_TRACE_OP_MATMUL_START,
    NPM_TRACE_OP_MATMUL_TILE,
    NPM_TRACE_OP_MATMUL_END,
    NPM_TRACE_OP_TILING_PLAN,  // Tiling strategy summary
} npm_trace_event_type;

// =============================================================================
// Trace Configuration
// =============================================================================

typedef struct npm_trace_config {
    uint32_t categories;        // Bitmask of npm_trace_category
    FILE*    output;            // Output file (NULL = stdout)
    bool     flush_immediate;   // Flush after each event (default: true)
} npm_trace_config;

// =============================================================================
// Trace Context (opaque handle)
// =============================================================================

typedef struct npm_trace_ctx npm_trace_ctx;

// =============================================================================
// Lifecycle Functions
// =============================================================================

/**
 * Create a new trace context.
 * @param config Configuration options (NULL for defaults: all disabled, stdout)
 * @return New trace context, or NULL on failure
 */
npm_trace_ctx* npm_trace_create(const npm_trace_config* config);

/**
 * Destroy a trace context and free resources.
 * @param ctx Trace context (NULL is safe)
 */
void npm_trace_destroy(npm_trace_ctx* ctx);

// =============================================================================
// Category Check (inline for zero-overhead when disabled)
// =============================================================================

/**
 * Check if a trace category is enabled.
 * Call this before formatting trace data to avoid overhead when disabled.
 * @param ctx Trace context
 * @param cat Category to check
 * @return true if category is enabled
 */
bool npm_trace_enabled(npm_trace_ctx* ctx, npm_trace_category cat);

// =============================================================================
// Event Emission Functions
// =============================================================================

/**
 * Trace an IPC command (request or response).
 * @param ctx Trace context
 * @param type Event type (NPM_TRACE_CMD_*)
 * @param seq_id Sequence ID to correlate request/response
 * @param status Status code (0xFF for request, actual status for response)
 * @param details JSON object string with command-specific details (or NULL)
 */
void npm_trace_command(npm_trace_ctx* ctx, npm_trace_event_type type,
                       uint32_t seq_id, uint8_t status, const char* details);

/**
 * Trace a DMA transfer.
 * @param ctx Trace context
 * @param type Event type (NPM_TRACE_DMA_*)
 * @param bytes Number of bytes transferred
 * @param cycles Simulated cycle count
 * @param engine_id Engine ID (-1 for system DMA)
 */
void npm_trace_dma(npm_trace_ctx* ctx, npm_trace_event_type type,
                   size_t bytes, uint64_t cycles, int engine_id);

/**
 * Trace a compute operation.
 * @param ctx Trace context
 * @param type Event type (NPM_TRACE_OP_*)
 * @param M Matrix dimension M
 * @param N Matrix dimension N
 * @param K Matrix dimension K
 * @param cycles Cycle count (0 for START events)
 * @param details Additional JSON details (or NULL)
 */
void npm_trace_op(npm_trace_ctx* ctx, npm_trace_event_type type,
                  int64_t M, int64_t N, int64_t K,
                  uint64_t cycles, const char* details);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Get current timestamp in nanoseconds (monotonic).
 * @return Nanoseconds since an arbitrary reference point
 */
uint64_t npm_trace_timestamp_ns(void);

/**
 * Get event type name as string.
 * @param type Event type
 * @return String representation (e.g., "MATMUL", "DDR_TO_L2")
 */
const char* npm_trace_event_name(npm_trace_event_type type);

#ifdef __cplusplus
}
#endif
