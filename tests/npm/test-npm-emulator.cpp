// Test for NPM Emulator Integration
// Tests the emulator device implementation via IPC to npm-emulator process
//
// IMPORTANT: This test requires the npm-emulator process to be running:
//   ./bin/npm-emulator --verbose &
//   ./bin/test-npm-emulator

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <csignal>
#include <unistd.h>
#include <sys/wait.h>

#include "ggml.h"
#include "npm-device.h"

static pid_t emulator_pid = -1;
static const char* SOCKET_PATH = "/tmp/npm-emulator.sock";

// Start the emulator process
static bool start_emulator() {
    // Check if socket already exists (emulator might be running)
    if (access(SOCKET_PATH, F_OK) == 0) {
        printf("  Emulator socket already exists, assuming emulator is running\n");
        return true;
    }

    emulator_pid = fork();
    if (emulator_pid < 0) {
        printf("  FAILED: Could not fork emulator process\n");
        return false;
    }

    if (emulator_pid == 0) {
        // Child process - exec the emulator
        execl("./bin/npm-emulator", "npm-emulator", "--socket", SOCKET_PATH, "--verbose", nullptr);
        // If exec fails
        perror("execl failed");
        exit(1);
    }

    // Parent process - wait for emulator to start
    printf("  Started emulator with PID %d\n", emulator_pid);

    // Wait for socket to appear
    for (int i = 0; i < 50; i++) {  // 5 seconds max
        if (access(SOCKET_PATH, F_OK) == 0) {
            printf("  Emulator socket ready\n");
            usleep(100000);  // Extra 100ms for emulator to be ready
            return true;
        }
        usleep(100000);  // 100ms
    }

    printf("  FAILED: Emulator socket not available after 5 seconds\n");
    return false;
}

// Stop the emulator process
static void stop_emulator() {
    if (emulator_pid > 0) {
        printf("  Stopping emulator (PID %d)\n", emulator_pid);
        kill(emulator_pid, SIGTERM);
        waitpid(emulator_pid, nullptr, 0);
        emulator_pid = -1;
    }

    // Clean up socket
    unlink(SOCKET_PATH);
}

// Signal handler for cleanup
static void signal_handler(int sig) {
    (void)sig;
    stop_emulator();
    exit(1);
}

// Test 1: Emulator Connection
static int test_emulator_connection() {
    printf("Test 1: Emulator Connection\n");

    struct npm_device* dev = npm_device_emulator_create(SOCKET_PATH);
    if (!dev) {
        printf("  FAILED: Could not create emulator device\n");
        printf("  Make sure npm-emulator is running: ./bin/npm-emulator &\n");
        return 1;
    }

    // Test device info
    enum npm_sku sku = dev->ops.get_sku(dev);
    int num_engines = dev->ops.get_num_engines(dev);
    size_t l1_size = dev->ops.get_l1_size(dev);
    size_t l2_size = dev->ops.get_l2_size(dev);

    printf("  SKU: %d (%s)\n", sku, npm_sku_name(sku));
    printf("  Engines: %d\n", num_engines);
    printf("  L1 Size: %zu bytes (%.1f MB)\n", l1_size, l1_size / (1024.0 * 1024.0));
    printf("  L2 Size: %zu bytes (%.1f MB)\n", l2_size, l2_size / (1024.0 * 1024.0));

    // Emulator should report valid device info
    if (num_engines <= 0 || l1_size == 0 || l2_size == 0) {
        printf("  FAILED: Invalid device info\n");
        npm_device_destroy(dev);
        return 1;
    }

    npm_device_destroy(dev);
    printf("  PASSED\n\n");
    return 0;
}

// Test 2: Buffer Registration via IPC
static int test_emulator_buffer() {
    printf("Test 2: Buffer Registration via IPC\n");

    struct npm_device* dev = npm_device_emulator_create(SOCKET_PATH);
    if (!dev) {
        printf("  FAILED: Could not create emulator device\n");
        return 1;
    }

    // Allocate some host memory
    size_t size = 4096;
    float* buffer = (float*)malloc(size);
    if (!buffer) {
        printf("  FAILED: Could not allocate host memory\n");
        npm_device_destroy(dev);
        return 1;
    }

    // Initialize with test data
    for (size_t i = 0; i < size / sizeof(float); i++) {
        buffer[i] = (float)i;
    }

    // Register buffer with device
    uint64_t handle = 0;
    int result = dev->ops.register_buffer(dev, buffer, size, &handle);
    if (result != 0) {
        printf("  FAILED: register_buffer returned error %d\n", result);
        free(buffer);
        npm_device_destroy(dev);
        return 1;
    }
    printf("  Buffer registered, handle: %lu\n", (unsigned long)handle);

    if (handle == 0) {
        printf("  FAILED: Invalid handle returned\n");
        free(buffer);
        npm_device_destroy(dev);
        return 1;
    }

    // Unregister buffer
    dev->ops.unregister_buffer(dev, handle);
    printf("  Buffer unregistered\n");

    free(buffer);
    npm_device_destroy(dev);
    printf("  PASSED\n\n");
    return 0;
}

// Test 3: MatMul via IPC - Small
static int test_emulator_matmul_small() {
    printf("Test 3: MatMul via IPC - Small (2x4x3)\n");

    struct npm_device* dev = npm_device_emulator_create(SOCKET_PATH);
    if (!dev) {
        printf("  FAILED: Could not create emulator device\n");
        return 1;
    }

    // Test small matmul: C = A * B^T
    const int M = 2;
    const int N = 4;
    const int K = 3;

    // Allocate host buffers
    float* A = (float*)malloc(M * K * sizeof(float));
    float* B = (float*)malloc(N * K * sizeof(float));
    float* C = (float*)malloc(M * N * sizeof(float));

    if (!A || !B || !C) {
        printf("  FAILED: Memory allocation failed\n");
        free(A); free(B); free(C);
        npm_device_destroy(dev);
        return 1;
    }

    // Initialize A (M x K) = 2x3
    float A_data[] = {
        1.0f, 2.0f, 3.0f,   // row 0
        4.0f, 5.0f, 6.0f    // row 1
    };
    memcpy(A, A_data, sizeof(A_data));

    // Initialize B (N x K) = 4x3
    float B_data[] = {
        1.0f, 0.0f, 0.0f,   // B^T row 0
        0.0f, 1.0f, 0.0f,   // B^T row 1
        0.0f, 0.0f, 1.0f,   // B^T row 2
        1.0f, 1.0f, 1.0f    // B^T row 3
    };
    memcpy(B, B_data, sizeof(B_data));

    memset(C, 0, M * N * sizeof(float));

    // Expected: C[m,n] = sum_k(A[m,k] * B[n,k])
    float expected[] = {
        1.0f, 2.0f, 3.0f, 6.0f,
        4.0f, 5.0f, 6.0f, 15.0f
    };

    // Register buffers
    uint64_t handle_a = 0, handle_b = 0, handle_c = 0;
    if (dev->ops.register_buffer(dev, A, M * K * sizeof(float), &handle_a) != 0 ||
        dev->ops.register_buffer(dev, B, N * K * sizeof(float), &handle_b) != 0 ||
        dev->ops.register_buffer(dev, C, M * N * sizeof(float), &handle_c) != 0) {
        printf("  FAILED: Could not register buffers\n");
        free(A); free(B); free(C);
        npm_device_destroy(dev);
        return 1;
    }
    printf("  Buffers registered: A=%lu, B=%lu, C=%lu\n",
           (unsigned long)handle_a, (unsigned long)handle_b, (unsigned long)handle_c);

    // Set up matmul params
    struct npm_matmul_params params;
    params.a_handle = handle_a;
    params.b_handle = handle_b;
    params.c_handle = handle_c;
    params.a_offset = 0;
    params.b_offset = 0;
    params.c_offset = 0;
    params.M = M;
    params.N = N;
    params.K = K;
    params.lda = K;
    params.ldb = K;
    params.ldc = N;
    params.type_a = GGML_TYPE_F32;
    params.type_b = GGML_TYPE_F32;
    params.type_c = GGML_TYPE_F32;

    int result = dev->ops.matmul(dev, &params);
    if (result != 0) {
        printf("  FAILED: MatMul returned error %d\n", result);
        dev->ops.unregister_buffer(dev, handle_a);
        dev->ops.unregister_buffer(dev, handle_b);
        dev->ops.unregister_buffer(dev, handle_c);
        free(A); free(B); free(C);
        npm_device_destroy(dev);
        return 1;
    }

    printf("  Result C:\n");
    for (int m = 0; m < M; m++) {
        printf("    ");
        for (int n = 0; n < N; n++) {
            printf("%6.1f ", C[m * N + n]);
        }
        printf("\n");
    }

    // Verify results
    bool correct = true;
    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = fabsf(C[i] - expected[i]);
        if (error > max_error) max_error = error;
        if (error > 1e-5f) {
            correct = false;
        }
    }

    printf("  Max error: %e\n", max_error);

    if (!correct) {
        printf("  Expected C:\n");
        for (int m = 0; m < M; m++) {
            printf("    ");
            for (int n = 0; n < N; n++) {
                printf("%6.1f ", expected[m * N + n]);
            }
            printf("\n");
        }
        printf("  FAILED: Results do not match expected\n");
        dev->ops.unregister_buffer(dev, handle_a);
        dev->ops.unregister_buffer(dev, handle_b);
        dev->ops.unregister_buffer(dev, handle_c);
        free(A); free(B); free(C);
        npm_device_destroy(dev);
        return 1;
    }

    dev->ops.unregister_buffer(dev, handle_a);
    dev->ops.unregister_buffer(dev, handle_b);
    dev->ops.unregister_buffer(dev, handle_c);
    free(A); free(B); free(C);
    npm_device_destroy(dev);
    printf("  PASSED\n\n");
    return 0;
}

// Test 4: Larger MatMul via IPC
static int test_emulator_matmul_large() {
    printf("Test 4: Larger MatMul via IPC (64x128x64)\n");

    struct npm_device* dev = npm_device_emulator_create(SOCKET_PATH);
    if (!dev) {
        printf("  FAILED: Could not create emulator device\n");
        return 1;
    }

    const int M = 64;
    const int N = 128;
    const int K = 64;

    float* A = (float*)malloc(M * K * sizeof(float));
    float* B = (float*)malloc(N * K * sizeof(float));
    float* C = (float*)malloc(M * N * sizeof(float));
    float* C_ref = (float*)malloc(M * N * sizeof(float));

    if (!A || !B || !C || !C_ref) {
        printf("  FAILED: Memory allocation failed\n");
        free(A); free(B); free(C); free(C_ref);
        npm_device_destroy(dev);
        return 1;
    }

    // Initialize with simple pattern
    for (int i = 0; i < M * K; i++) A[i] = (float)(i % 10) * 0.1f;
    for (int i = 0; i < N * K; i++) B[i] = (float)(i % 7) * 0.1f;
    memset(C, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    // Compute reference using naive implementation
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C_ref[m * N + n] = sum;
        }
    }

    // Register buffers
    uint64_t handle_a = 0, handle_b = 0, handle_c = 0;
    if (dev->ops.register_buffer(dev, A, M * K * sizeof(float), &handle_a) != 0 ||
        dev->ops.register_buffer(dev, B, N * K * sizeof(float), &handle_b) != 0 ||
        dev->ops.register_buffer(dev, C, M * N * sizeof(float), &handle_c) != 0) {
        printf("  FAILED: Could not register buffers\n");
        free(A); free(B); free(C); free(C_ref);
        npm_device_destroy(dev);
        return 1;
    }

    // Compute using device
    struct npm_matmul_params params;
    params.a_handle = handle_a;
    params.b_handle = handle_b;
    params.c_handle = handle_c;
    params.a_offset = 0;
    params.b_offset = 0;
    params.c_offset = 0;
    params.M = M;
    params.N = N;
    params.K = K;
    params.lda = K;
    params.ldb = K;
    params.ldc = N;
    params.type_a = GGML_TYPE_F32;
    params.type_b = GGML_TYPE_F32;
    params.type_c = GGML_TYPE_F32;

    int result = dev->ops.matmul(dev, &params);
    if (result != 0) {
        printf("  FAILED: MatMul returned error %d\n", result);
        dev->ops.unregister_buffer(dev, handle_a);
        dev->ops.unregister_buffer(dev, handle_b);
        dev->ops.unregister_buffer(dev, handle_c);
        free(A); free(B); free(C); free(C_ref);
        npm_device_destroy(dev);
        return 1;
    }

    // Verify results
    float max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < M * N; i++) {
        float error = fabsf(C[i] - C_ref[i]);
        if (error > max_error) max_error = error;
        if (error > 1e-4f) error_count++;
    }

    printf("  Max error vs reference: %e\n", max_error);
    printf("  Elements with error > 1e-4: %d / %d\n", error_count, M * N);

    if (max_error > 1e-4f) {
        printf("  FAILED: Error too large\n");
        dev->ops.unregister_buffer(dev, handle_a);
        dev->ops.unregister_buffer(dev, handle_b);
        dev->ops.unregister_buffer(dev, handle_c);
        free(A); free(B); free(C); free(C_ref);
        npm_device_destroy(dev);
        return 1;
    }

    dev->ops.unregister_buffer(dev, handle_a);
    dev->ops.unregister_buffer(dev, handle_b);
    dev->ops.unregister_buffer(dev, handle_c);
    free(A); free(B); free(C); free(C_ref);
    npm_device_destroy(dev);
    printf("  PASSED\n\n");
    return 0;
}

int main(int argc, char** argv) {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║     NPM Emulator Integration Tests       ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    // Set up signal handlers for cleanup
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Check if we should start our own emulator
    bool manage_emulator = (argc > 1 && strcmp(argv[1], "--managed") == 0);

    if (manage_emulator) {
        printf("Starting managed emulator...\n");
        if (!start_emulator()) {
            return 1;
        }
    } else {
        printf("Using external emulator (start with: ./bin/npm-emulator &)\n\n");
    }

    int failures = 0;

    failures += test_emulator_connection();
    failures += test_emulator_buffer();
    failures += test_emulator_matmul_small();
    failures += test_emulator_matmul_large();

    if (manage_emulator) {
        stop_emulator();
    }

    printf("╔══════════════════════════════════════════╗\n");
    if (failures == 0) {
        printf("║     All tests PASSED!                    ║\n");
    } else {
        printf("║     %d test(s) FAILED                     ║\n", failures);
    }
    printf("╚══════════════════════════════════════════╝\n");

    return failures;
}
