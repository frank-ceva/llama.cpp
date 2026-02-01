// NPM Emulator - Main Entry Point
//
// A standalone process that emulates NPM hardware behavior.
// Communicates with npm-device-emulator.cpp clients via Unix socket.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <getopt.h>

#include "npm-emu-server.h"
#include "npm-config.h"
#include "npm-trace.h"

// =============================================================================
// Global server for signal handler
// =============================================================================

static npm_emu_server * g_server = nullptr;

static void signal_handler(int sig) {
    (void)sig;
    if (g_server) {
        npm_emu_server_shutdown(g_server);
    }
}

// =============================================================================
// Usage
// =============================================================================

static void print_usage(const char * prog) {
    printf("NPM Hardware Emulator\n");
    printf("\n");
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("\n");
    printf("Options:\n");
    printf("  --config PATH      Load configuration from file\n");
    printf("  --socket PATH      Unix socket path (default: /tmp/npm-emulator.sock)\n");
    printf("  --sku SKU          Device SKU: NPM4K, NPM8K, NPM16K, NPM32K, NPM64K\n");
    printf("                     (default: NPM8K)\n");
    printf("  --l2-size SIZE     L2 cache size in MB (default: SKU default)\n");
    printf("  --tiling           Enable tiled matmul execution (DMA simulation)\n");
    printf("  --timing           Enable timing/cycle simulation\n");
    printf("  --verbose, -v      Verbose output\n");
    printf("  --help, -h         Show this help\n");
    printf("\n");
    printf("Tracing options:\n");
    printf("  --trace-commands   Trace IPC command flow (JSON output)\n");
    printf("  --trace-dma        Trace DMA transfers\n");
    printf("  --trace-ops        Trace compute operations\n");
    printf("  --trace-all        Enable all tracing categories\n");
    printf("  --trace-file PATH  Write trace output to file (default: stdout)\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s --sku NPM8K --l2-size 8 --verbose\n", prog);
    printf("  %s --config npm-config.ini\n", prog);
    printf("  %s --trace-all --trace-file /tmp/npm-trace.json\n", prog);
    printf("\n");
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char ** argv) {
    // Buffer to hold socket path (from config file or command line)
    static char socket_path_buffer[256];
    strncpy(socket_path_buffer, NPM_EMU_DEFAULT_SOCKET, sizeof(socket_path_buffer) - 1);

    // Buffer to hold trace file path
    static char trace_file_buffer[256] = "";

    npm_emu_config config;
    memset(&config, 0, sizeof(config));
    config.socket_path = socket_path_buffer;
    config.sku = NPM_SKU_8K;
    config.l2_size = 0;  // Use SKU default
    config.tiling_enabled = false;
    config.timing_enabled = false;
    config.verbose = false;
    config.trace_categories = NPM_TRACE_NONE;
    config.trace_file = nullptr;

    const char * config_file = nullptr;

    static struct option long_options[] = {
        { "config",         required_argument, nullptr, 'c' },
        { "socket",         required_argument, nullptr, 's' },
        { "sku",            required_argument, nullptr, 'k' },
        { "l2-size",        required_argument, nullptr, 'l' },
        { "tiling",         no_argument,       nullptr, 'i' },
        { "timing",         no_argument,       nullptr, 't' },
        { "verbose",        no_argument,       nullptr, 'v' },
        { "help",           no_argument,       nullptr, 'h' },
        { "trace-commands", no_argument,       nullptr, 'C' },
        { "trace-dma",      no_argument,       nullptr, 'D' },
        { "trace-ops",      no_argument,       nullptr, 'O' },
        { "trace-all",      no_argument,       nullptr, 'A' },
        { "trace-file",     required_argument, nullptr, 'T' },
        { nullptr,          0,                 nullptr, 0 }
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "c:s:k:l:itvhCDOAT:", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'c':
                config_file = optarg;
                break;
            case 's':
                strncpy(socket_path_buffer, optarg, sizeof(socket_path_buffer) - 1);
                break;
            case 'k':
                config.sku = npm_sku_from_string(optarg);
                break;
            case 'l':
                config.l2_size = (size_t)atoi(optarg) * 1024 * 1024;
                break;
            case 'i':
                config.tiling_enabled = true;
                break;
            case 't':
                config.timing_enabled = true;
                break;
            case 'v':
                config.verbose = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            case 'C':
                config.trace_categories |= NPM_TRACE_COMMANDS;
                break;
            case 'D':
                config.trace_categories |= NPM_TRACE_DMA;
                break;
            case 'O':
                config.trace_categories |= NPM_TRACE_OPS;
                break;
            case 'A':
                config.trace_categories = NPM_TRACE_ALL;
                break;
            case 'T':
                strncpy(trace_file_buffer, optarg, sizeof(trace_file_buffer) - 1);
                config.trace_file = trace_file_buffer;
                break;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    // Load config file if specified (command line overrides config file)
    if (config_file) {
        npm_config file_config;
        if (npm_config_load(config_file, &file_config)) {
            // Apply config file settings (only if not overridden by command line)
            config.sku = file_config.sku;
            if (config.l2_size == 0) {
                config.l2_size = file_config.l2_size_mb * 1024 * 1024;
            }
            config.tiling_enabled = file_config.tiling_enabled;
            config.timing_enabled = file_config.timing_enabled;
            config.verbose = file_config.verbose;
            strncpy(socket_path_buffer, file_config.socket_path, sizeof(socket_path_buffer) - 1);

            // Apply trace settings from config file (if not already set by CLI)
            if (config.trace_categories == NPM_TRACE_NONE) {
                if (file_config.trace_commands) config.trace_categories |= NPM_TRACE_COMMANDS;
                if (file_config.trace_dma) config.trace_categories |= NPM_TRACE_DMA;
                if (file_config.trace_ops) config.trace_categories |= NPM_TRACE_OPS;
            }
            if (config.trace_file == nullptr && file_config.trace_file[0] != '\0') {
                strncpy(trace_file_buffer, file_config.trace_file, sizeof(trace_file_buffer) - 1);
                config.trace_file = trace_file_buffer;
            }

            if (config.verbose) {
                printf("Loaded config from: %s\n", config_file);
                npm_config_print(&file_config);
            }
        } else {
            fprintf(stderr, "Warning: Could not load config file: %s\n", config_file);
        }
    }

    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Create server
    g_server = npm_emu_server_create(&config);
    if (!g_server) {
        fprintf(stderr, "Failed to create server\n");
        return 1;
    }

    // Run server
    int result = npm_emu_server_run(g_server);

    // Cleanup
    npm_emu_server_destroy(g_server);
    g_server = nullptr;

    return result;
}
