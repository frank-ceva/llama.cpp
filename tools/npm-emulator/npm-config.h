#pragma once

// NPM Configuration File Parser
//
// Simple INI-style configuration file support for npm-emulator.
// Format:
//   # Comment
//   key=value
//
// Supported keys:
//   sku              - Device SKU (NPM4K, NPM8K, NPM16K, NPM32K, NPM64K)
//   l2_size_mb       - L2 cache size in MB
//   tiling           - Enable tiled matmul execution (true/false)
//   timing           - Enable timing simulation (true/false)
//   verbose          - Enable verbose output (true/false)
//   socket           - Unix socket path
//   dma_system_bw_gbps - System DMA bandwidth (GB/s)
//   dma_l1_bw_gbps   - L1 DMA bandwidth (GB/s)
//   clock_freq_mhz   - Clock frequency (MHz)
//
// Tracing options:
//   trace_commands   - Trace IPC commands (true/false)
//   trace_dma        - Trace DMA transfers (true/false)
//   trace_ops        - Trace compute operations (true/false)
//   trace_file       - Output file for trace (default: stdout)
//
// Example config file:
//   # NPM Emulator Configuration
//   sku=NPM8K
//   l2_size_mb=8
//   timing=true
//   verbose=false
//
//   # Tracing
//   trace_commands=true
//   trace_dma=true
//   trace_ops=true
//   trace_file=/tmp/npm-trace.json

#include <cstdio>
#include <cstring>
#include <cstdlib>

#include "npm-types.h"

// Maximum line length in config file
#define NPM_CONFIG_MAX_LINE 256

// Configuration structure
struct npm_config {
    // Device settings
    enum npm_sku sku;
    size_t l2_size_mb;

    // Runtime settings
    bool tiling_enabled;
    bool timing_enabled;
    bool verbose;
    char socket_path[256];

    // DMA settings
    double dma_system_bw_gbps;
    double dma_l1_bw_gbps;
    uint64_t clock_freq_mhz;

    // Tracing settings
    bool trace_commands;
    bool trace_dma;
    bool trace_ops;
    char trace_file[256];

    // Initialize with defaults
    npm_config() {
        sku = NPM_SKU_8K;
        l2_size_mb = 8;
        tiling_enabled = false;
        timing_enabled = false;
        verbose = false;
        strncpy(socket_path, "/tmp/npm-emulator.sock", sizeof(socket_path) - 1);
        socket_path[sizeof(socket_path) - 1] = '\0';
        dma_system_bw_gbps = 50.0;
        dma_l1_bw_gbps = 100.0;
        clock_freq_mhz = 1000;
        trace_commands = false;
        trace_dma = false;
        trace_ops = false;
        trace_file[0] = '\0';
    }
};

// Helper: trim whitespace from both ends
static inline char * npm_config_trim(char * str) {
    // Trim leading
    while (*str && (*str == ' ' || *str == '\t')) str++;

    // Trim trailing
    char * end = str + strlen(str) - 1;
    while (end > str && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')) {
        *end = '\0';
        end--;
    }
    return str;
}

// Helper: parse boolean value
static inline bool npm_config_parse_bool(const char * value) {
    return (strcmp(value, "true") == 0 ||
            strcmp(value, "yes") == 0 ||
            strcmp(value, "1") == 0 ||
            strcmp(value, "on") == 0);
}

// Load configuration from file
// Returns true on success, false on error
static inline bool npm_config_load(const char * path, npm_config * config) {
    FILE * f = fopen(path, "r");
    if (!f) {
        return false;
    }

    char line[NPM_CONFIG_MAX_LINE];
    int line_num = 0;

    while (fgets(line, sizeof(line), f)) {
        line_num++;

        // Skip lines that are only whitespace
        bool only_whitespace = true;
        for (char * p = line; *p; p++) {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
                only_whitespace = false;
                break;
            }
        }
        if (only_whitespace) {
            continue;
        }

        // Trim and skip comments
        char * trimmed = npm_config_trim(line);
        if (*trimmed == '\0' || *trimmed == '#') {
            continue;
        }

        // Find '=' separator
        char * eq = strchr(trimmed, '=');
        if (!eq) {
            fprintf(stderr, "Config error line %d: missing '='\n", line_num);
            continue;
        }

        // Split into key and value
        *eq = '\0';
        char * key = npm_config_trim(trimmed);
        char * value = npm_config_trim(eq + 1);

        // Parse known keys
        if (strcmp(key, "sku") == 0) {
            config->sku = npm_sku_from_string(value);
        } else if (strcmp(key, "l2_size_mb") == 0) {
            config->l2_size_mb = (size_t)atoi(value);
        } else if (strcmp(key, "tiling") == 0) {
            config->tiling_enabled = npm_config_parse_bool(value);
        } else if (strcmp(key, "timing") == 0) {
            config->timing_enabled = npm_config_parse_bool(value);
        } else if (strcmp(key, "verbose") == 0) {
            config->verbose = npm_config_parse_bool(value);
        } else if (strcmp(key, "socket") == 0) {
            strncpy(config->socket_path, value, sizeof(config->socket_path) - 1);
            config->socket_path[sizeof(config->socket_path) - 1] = '\0';
        } else if (strcmp(key, "dma_system_bw_gbps") == 0) {
            config->dma_system_bw_gbps = atof(value);
        } else if (strcmp(key, "dma_l1_bw_gbps") == 0) {
            config->dma_l1_bw_gbps = atof(value);
        } else if (strcmp(key, "clock_freq_mhz") == 0) {
            config->clock_freq_mhz = (uint64_t)atoll(value);
        } else if (strcmp(key, "trace_commands") == 0) {
            config->trace_commands = npm_config_parse_bool(value);
        } else if (strcmp(key, "trace_dma") == 0) {
            config->trace_dma = npm_config_parse_bool(value);
        } else if (strcmp(key, "trace_ops") == 0) {
            config->trace_ops = npm_config_parse_bool(value);
        } else if (strcmp(key, "trace_file") == 0) {
            strncpy(config->trace_file, value, sizeof(config->trace_file) - 1);
            config->trace_file[sizeof(config->trace_file) - 1] = '\0';
        } else {
            fprintf(stderr, "Config warning line %d: unknown key '%s'\n", line_num, key);
        }
    }

    fclose(f);
    return true;
}

// Print configuration (for debugging)
static inline void npm_config_print(const npm_config * config) {
    printf("Configuration:\n");
    printf("  sku=%s\n", npm_sku_to_string(config->sku));
    printf("  l2_size_mb=%zu\n", config->l2_size_mb);
    printf("  tiling=%s\n", config->tiling_enabled ? "true" : "false");
    printf("  timing=%s\n", config->timing_enabled ? "true" : "false");
    printf("  verbose=%s\n", config->verbose ? "true" : "false");
    printf("  socket=%s\n", config->socket_path);
    printf("  dma_system_bw_gbps=%.1f\n", config->dma_system_bw_gbps);
    printf("  dma_l1_bw_gbps=%.1f\n", config->dma_l1_bw_gbps);
    printf("  clock_freq_mhz=%lu\n", (unsigned long)config->clock_freq_mhz);
    printf("  trace_commands=%s\n", config->trace_commands ? "true" : "false");
    printf("  trace_dma=%s\n", config->trace_dma ? "true" : "false");
    printf("  trace_ops=%s\n", config->trace_ops ? "true" : "false");
    if (config->trace_file[0] != '\0') {
        printf("  trace_file=%s\n", config->trace_file);
    }
}
