#pragma once

// NPM Common Types
//
// Shared type definitions used by both the NPM device driver
// and the NPM emulator process.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// NPM SKU types (mirrors npm-device.h but standalone for emulator)
// =============================================================================

enum npm_sku {
    NPM_SKU_4K = 0,       // 1 engine, 16K INT4 MACs
    NPM_SKU_8K,           // 1 engine, 32K INT4 MACs
    NPM_SKU_16K,          // 2 engines, 64K INT4 MACs
    NPM_SKU_32K,          // 4 engines, 128K INT4 MACs
    NPM_SKU_64K,          // 8 engines, 256K INT4 MACs
    NPM_SKU_MOCK,         // Mock implementation
    NPM_SKU_EMULATOR,     // Emulator implementation
};

// =============================================================================
// Data types (mirroring ggml types for standalone usage)
// When building with ggml, these should match ggml_type values
// =============================================================================

#ifndef NPM_TYPE_F32
#define NPM_TYPE_F32  0   // ggml GGML_TYPE_F32
#define NPM_TYPE_F16  1   // ggml GGML_TYPE_F16
#define NPM_TYPE_Q4_0 2   // ggml GGML_TYPE_Q4_0
#define NPM_TYPE_Q4_1 3   // ggml GGML_TYPE_Q4_1
#define NPM_TYPE_Q8_0 8   // ggml GGML_TYPE_Q8_0
#endif

// =============================================================================
// Memory allocation flags
// =============================================================================

enum npm_alloc_flags {
    NPM_ALLOC_DEFAULT = 0,
    NPM_ALLOC_L2      = 1 << 0,  // Prefer L2 cache placement
    NPM_ALLOC_PINNED  = 1 << 1,  // Pinned memory for DMA
};

// =============================================================================
// SKU configuration
// =============================================================================

struct npm_sku_config {
    enum npm_sku sku;
    int          num_engines;
    size_t       l1_size;           // L1 size per engine (bytes)
    size_t       l2_size_default;   // Default L2 size (bytes)
    size_t       l2_size_min;       // Minimum L2 size (bytes)
    size_t       l2_size_max;       // Maximum L2 size (bytes)
    int64_t      int4_macs;         // INT4 MACs per cycle
    int64_t      int8_macs;         // INT8 MACs per cycle
    int64_t      fp16_macs;         // FP16 MACs per cycle
};

// SKU configurations (as per NPM spec)
static const struct npm_sku_config NPM_SKU_CONFIGS[] = {
    // SKU            Engines  L1(MB)     L2 default  L2 min     L2 max      INT4   INT8   FP16
    { NPM_SKU_4K,     1,       1*1024*1024, 8*1024*1024, 1*1024*1024, 32*1024*1024, 16000,  4000,  2000 },
    { NPM_SKU_8K,     1,       1*1024*1024, 8*1024*1024, 1*1024*1024, 32*1024*1024, 32000,  8000,  4000 },
    { NPM_SKU_16K,    2,       1*1024*1024, 8*1024*1024, 1*1024*1024, 32*1024*1024, 64000,  16000, 8000 },
    { NPM_SKU_32K,    4,       1*1024*1024, 8*1024*1024, 1*1024*1024, 32*1024*1024, 128000, 32000, 16000 },
    { NPM_SKU_64K,    8,       1*1024*1024, 8*1024*1024, 1*1024*1024, 32*1024*1024, 256000, 64000, 32000 },
    { NPM_SKU_MOCK,   1,       1*1024*1024, 8*1024*1024, 1*1024*1024, 32*1024*1024, 0,      0,     0 },
    { NPM_SKU_EMULATOR, 1,     1*1024*1024, 8*1024*1024, 1*1024*1024, 32*1024*1024, 0,      0,     0 },
};

#define NPM_SKU_COUNT (sizeof(NPM_SKU_CONFIGS) / sizeof(NPM_SKU_CONFIGS[0]))

// =============================================================================
// Helper functions
// =============================================================================

// Get configuration for a SKU
static inline const struct npm_sku_config * npm_get_sku_config(enum npm_sku sku) {
    for (size_t i = 0; i < NPM_SKU_COUNT; i++) {
        if (NPM_SKU_CONFIGS[i].sku == sku) {
            return &NPM_SKU_CONFIGS[i];
        }
    }
    return NULL;
}

// Get human-readable name for SKU
static inline const char * npm_sku_to_string(enum npm_sku sku) {
    switch (sku) {
        case NPM_SKU_4K:      return "NPM4K";
        case NPM_SKU_8K:      return "NPM8K";
        case NPM_SKU_16K:     return "NPM16K";
        case NPM_SKU_32K:     return "NPM32K";
        case NPM_SKU_64K:     return "NPM64K";
        case NPM_SKU_MOCK:    return "Mock";
        case NPM_SKU_EMULATOR: return "Emulator";
        default:              return "Unknown";
    }
}

// Parse SKU from string
static inline enum npm_sku npm_sku_from_string(const char * str) {
    if (!str) return NPM_SKU_8K;

    if (str[0] == 'N' || str[0] == 'n') {
        // NPM format
        if (str[3] == '4' || str[4] == '4') return NPM_SKU_4K;
        if (str[3] == '8' || str[4] == '8') return NPM_SKU_8K;
        if (str[3] == '1' || str[4] == '1') return NPM_SKU_16K;
        if (str[3] == '3' || str[4] == '3') return NPM_SKU_32K;
        if (str[3] == '6' || str[4] == '6') return NPM_SKU_64K;
    } else {
        // Numeric format
        if (str[0] == '4') return NPM_SKU_4K;
        if (str[0] == '8') return NPM_SKU_8K;
        if (str[0] == '1') return NPM_SKU_16K;
        if (str[0] == '3') return NPM_SKU_32K;
        if (str[0] == '6') return NPM_SKU_64K;
    }

    return NPM_SKU_8K;  // Default
}

#ifdef __cplusplus
}
#endif
