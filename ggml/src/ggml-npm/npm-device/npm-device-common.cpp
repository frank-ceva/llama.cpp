#include "npm-device.h"
#include <cstdlib>

// =============================================================================
// Common Device Utilities (shared across all device implementations)
// =============================================================================

const char * npm_sku_name(enum npm_sku sku) {
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

void npm_device_destroy(struct npm_device * dev) {
    if (dev) {
        if (dev->ops.shutdown) {
            dev->ops.shutdown(dev);
        }
        // Generic cleanup - actual context deletion is device-specific
        if (dev->context) {
            free(dev->context);
        }
        delete dev;
    }
}
