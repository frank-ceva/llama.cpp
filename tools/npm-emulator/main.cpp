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
    printf("  --socket PATH      Unix socket path (default: /tmp/npm-emulator.sock)\n");
    printf("  --sku SKU          Device SKU: NPM4K, NPM8K, NPM16K, NPM32K, NPM64K\n");
    printf("                     (default: NPM8K)\n");
    printf("  --l2-size SIZE     L2 cache size in MB (default: SKU default)\n");
    printf("  --timing           Enable timing simulation (not implemented yet)\n");
    printf("  --verbose, -v      Verbose output\n");
    printf("  --help, -h         Show this help\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s --sku NPM8K --l2-size 8 --verbose\n", prog);
    printf("\n");
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char ** argv) {
    // Disable stdio buffering for visibility when run as child process
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    npm_emu_config config;
    memset(&config, 0, sizeof(config));
    config.socket_path = NPM_EMU_DEFAULT_SOCKET;
    config.sku = NPM_SKU_8K;
    config.l2_size = 0;  // Use SKU default
    config.timing_enabled = false;
    config.verbose = false;

    static struct option long_options[] = {
        { "socket",   required_argument, nullptr, 's' },
        { "sku",      required_argument, nullptr, 'k' },
        { "l2-size",  required_argument, nullptr, 'l' },
        { "timing",   no_argument,       nullptr, 't' },
        { "verbose",  no_argument,       nullptr, 'v' },
        { "help",     no_argument,       nullptr, 'h' },
        { nullptr,    0,                 nullptr, 0 }
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "s:k:l:tvh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 's':
                config.socket_path = optarg;
                break;
            case 'k':
                config.sku = npm_sku_from_string(optarg);
                break;
            case 'l':
                config.l2_size = (size_t)atoi(optarg) * 1024 * 1024;
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
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Create server
    g_server = npm_emu_server_create(&config);
    if (!g_server) {
        fprintf(stderr, "Failed to create server\n");
        fflush(stderr);
        return 1;
    }

    // Run server
    int result = npm_emu_server_run(g_server);

    // Cleanup
    npm_emu_server_destroy(g_server);
    g_server = nullptr;

    return result;
}
