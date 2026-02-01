/**
 * NPM Emulator Tracing System - Implementation
 */

#include "npm-trace.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// =============================================================================
// Trace Context Structure
// =============================================================================

struct npm_trace_ctx {
    uint32_t categories;
    FILE*    output;
    bool     flush_immediate;
    bool     owns_file;        // True if we opened the file (need to close)
    uint64_t start_time_ns;    // Reference time for relative timestamps
};

// =============================================================================
// Event Type Names
// =============================================================================

static const char* event_names[] = {
    // Commands
    "HELLO",
    "GOODBYE",
    "PING",
    "REGISTER_BUFFER",
    "UNREGISTER_BUFFER",
    "MATMUL",
    "SYNC",
    "FENCE_CREATE",
    "FENCE_DESTROY",
    "FENCE_WAIT",

    // DMA
    "DDR_TO_L2",
    "L2_TO_DDR",
    "L2_TO_L1",
    "L1_TO_L2",

    // Compute
    "MATMUL_START",
    "MATMUL_TILE",
    "MATMUL_END",
    "TILING_PLAN",
};

// =============================================================================
// Timestamp Implementation
// =============================================================================

uint64_t npm_trace_timestamp_ns(void) {
#ifdef _WIN32
    static LARGE_INTEGER frequency = {0};
    if (frequency.QuadPart == 0) {
        QueryPerformanceFrequency(&frequency);
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (uint64_t)(counter.QuadPart * 1000000000ULL / frequency.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

// =============================================================================
// Lifecycle Functions
// =============================================================================

npm_trace_ctx* npm_trace_create(const npm_trace_config* config) {
    npm_trace_ctx* ctx = (npm_trace_ctx*)calloc(1, sizeof(npm_trace_ctx));
    if (!ctx) {
        return NULL;
    }

    if (config) {
        ctx->categories = config->categories;
        ctx->output = config->output;
        ctx->flush_immediate = config->flush_immediate;
    } else {
        ctx->categories = NPM_TRACE_NONE;
        ctx->output = NULL;
        ctx->flush_immediate = true;
    }

    // Default to stdout if no output specified
    if (!ctx->output) {
        ctx->output = stdout;
        ctx->owns_file = false;
    }

    ctx->start_time_ns = npm_trace_timestamp_ns();

    return ctx;
}

void npm_trace_destroy(npm_trace_ctx* ctx) {
    if (!ctx) {
        return;
    }

    // Flush any pending output
    if (ctx->output) {
        fflush(ctx->output);
    }

    // Close file if we own it
    if (ctx->owns_file && ctx->output && ctx->output != stdout && ctx->output != stderr) {
        fclose(ctx->output);
    }

    free(ctx);
}

// =============================================================================
// Category Check
// =============================================================================

bool npm_trace_enabled(npm_trace_ctx* ctx, npm_trace_category cat) {
    if (!ctx) {
        return false;
    }
    return (ctx->categories & cat) != 0;
}

// =============================================================================
// Helper: Get relative timestamp
// =============================================================================

static uint64_t get_relative_ts(npm_trace_ctx* ctx) {
    return npm_trace_timestamp_ns() - ctx->start_time_ns;
}

// =============================================================================
// Helper: Status to string
// =============================================================================

static const char* status_to_string(uint8_t status) {
    if (status == 0xFF) {
        return "REQ";
    }
    switch (status) {
        case 0: return "OK";
        case 1: return "ERR_INVALID_CMD";
        case 2: return "ERR_INVALID_HANDLE";
        case 3: return "ERR_OUT_OF_MEMORY";
        case 4: return "ERR_TIMEOUT";
        case 5: return "ERR_INVALID_SIZE";
        default: return "ERR_UNKNOWN";
    }
}

// =============================================================================
// Event Name Lookup
// =============================================================================

const char* npm_trace_event_name(npm_trace_event_type type) {
    if (type < 0 || type >= (int)(sizeof(event_names) / sizeof(event_names[0]))) {
        return "UNKNOWN";
    }
    return event_names[type];
}

// =============================================================================
// Event Emission: Commands
// =============================================================================

void npm_trace_command(npm_trace_ctx* ctx, npm_trace_event_type type,
                       uint32_t seq_id, uint8_t status, const char* details) {
    if (!ctx || !(ctx->categories & NPM_TRACE_COMMANDS)) {
        return;
    }

    uint64_t ts = get_relative_ts(ctx);
    const char* type_name = npm_trace_event_name(type);
    const char* status_str = status_to_string(status);

    if (details && details[0] != '\0') {
        fprintf(ctx->output,
                "{\"ts\":%llu,\"cat\":\"cmd\",\"type\":\"%s\",\"seq\":%u,\"status\":\"%s\",\"details\":%s}\n",
                (unsigned long long)ts, type_name, seq_id, status_str, details);
    } else {
        fprintf(ctx->output,
                "{\"ts\":%llu,\"cat\":\"cmd\",\"type\":\"%s\",\"seq\":%u,\"status\":\"%s\"}\n",
                (unsigned long long)ts, type_name, seq_id, status_str);
    }

    if (ctx->flush_immediate) {
        fflush(ctx->output);
    }
}

// =============================================================================
// Event Emission: DMA
// =============================================================================

void npm_trace_dma(npm_trace_ctx* ctx, npm_trace_event_type type,
                   size_t bytes, uint64_t cycles, int engine_id) {
    if (!ctx || !(ctx->categories & NPM_TRACE_DMA)) {
        return;
    }

    uint64_t ts = get_relative_ts(ctx);
    const char* type_name = npm_trace_event_name(type);

    fprintf(ctx->output,
            "{\"ts\":%llu,\"cat\":\"dma\",\"type\":\"%s\",\"bytes\":%zu,\"cycles\":%llu,\"engine\":%d}\n",
            (unsigned long long)ts, type_name, bytes, (unsigned long long)cycles, engine_id);

    if (ctx->flush_immediate) {
        fflush(ctx->output);
    }
}

// =============================================================================
// Event Emission: Compute Operations
// =============================================================================

void npm_trace_op(npm_trace_ctx* ctx, npm_trace_event_type type,
                  int64_t M, int64_t N, int64_t K,
                  uint64_t cycles, const char* details) {
    if (!ctx || !(ctx->categories & NPM_TRACE_OPS)) {
        return;
    }

    uint64_t ts = get_relative_ts(ctx);
    const char* type_name = npm_trace_event_name(type);

    if (details && details[0] != '\0') {
        fprintf(ctx->output,
                "{\"ts\":%llu,\"cat\":\"op\",\"type\":\"%s\",\"M\":%lld,\"N\":%lld,\"K\":%lld,\"cycles\":%llu,\"details\":%s}\n",
                (unsigned long long)ts, type_name,
                (long long)M, (long long)N, (long long)K,
                (unsigned long long)cycles, details);
    } else {
        fprintf(ctx->output,
                "{\"ts\":%llu,\"cat\":\"op\",\"type\":\"%s\",\"M\":%lld,\"N\":%lld,\"K\":%lld,\"cycles\":%llu}\n",
                (unsigned long long)ts, type_name,
                (long long)M, (long long)N, (long long)K,
                (unsigned long long)cycles);
    }

    if (ctx->flush_immediate) {
        fflush(ctx->output);
    }
}
