// NPM Device Common Implementation
//
// Common utility functions shared across all device implementations.
// Note: npm_device_destroy is defined in each device file due to type-specific cleanup.

#include "npm-device.h"

// =============================================================================
// Utility functions
// =============================================================================

const char * npm_sku_name(enum npm_sku sku) {
    switch (sku) {
        case NPM_SKU_4K:      return "NPM4K";
        case NPM_SKU_8K:      return "NPM8K";
        case NPM_SKU_16K:     return "NPM16K";
        case NPM_SKU_32K:     return "NPM32K";
        case NPM_SKU_64K:     return "NPM64K";
        case NPM_SKU_MOCK:    return "MOCK";
        case NPM_SKU_EMULATOR: return "EMULATOR";
        default:              return "UNKNOWN";
    }
}
