// Comprehensive test suite for NPM backend using GGML API directly
// This tests the NPM backend independently of llama.cpp model loading
//
// Tests cover:
// - Backend initialization and registration
// - FP32 matmul with various sizes
// - Quantized matmul (Q4_K, Q8_0)
// - Batched matmul operations
// - Buffer management
// - Edge cases and error handling

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-npm.h"

// =============================================================================
// Test utilities
// =============================================================================

static std::mt19937 g_rng(42);  // Fixed seed for reproducibility

static void init_random_f32(float* data, size_t n, float min = -1.0f, float max = 1.0f) {
    std::uniform_real_distribution<float> dist(min, max);
    for (size_t i = 0; i < n; i++) {
        data[i] = dist(g_rng);
    }
}

static void init_deterministic_f32(float* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (float)(i % 10) * 0.1f;
    }
}

// Reference CPU matmul: C = A * B^T
// A: (K, M), B: (K, N), C: (N, M)
static void ref_matmul_f32(const float* A, const float* B, float* C,
                           int64_t M, int64_t N, int64_t K) {
    memset(C, 0, N * M * sizeof(float));
    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                sum += A[k + m * K] * B[k + n * K];
            }
            C[n + m * N] = sum;
        }
    }
}

static float max_error(const float* a, const float* b, size_t n) {
    float max_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static int count_errors(const float* a, const float* b, size_t n, float threshold = 1e-4f) {
    int count = 0;
    for (size_t i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > threshold) count++;
    }
    return count;
}

// =============================================================================
// Test 1: Backend initialization and registration
// =============================================================================

static int test_backend_init() {
    printf("Test 1: Backend initialization\n");

    ggml_backend_t backend = ggml_backend_npm_init();
    if (!backend) {
        printf("  FAILED: Could not create NPM backend\n");
        return 1;
    }

    printf("  Backend name: %s\n", ggml_backend_name(backend));

    if (!ggml_backend_is_npm(backend)) {
        printf("  FAILED: Backend is not NPM\n");
        ggml_backend_free(backend);
        return 1;
    }

    // Check registry
    ggml_backend_reg_t reg = ggml_backend_npm_reg();
    if (!reg) {
        printf("  FAILED: Could not get NPM registry\n");
        ggml_backend_free(backend);
        return 1;
    }

    size_t dev_count = ggml_backend_reg_dev_count(reg);
    printf("  Device count: %zu\n", dev_count);

    if (dev_count < 1) {
        printf("  FAILED: No devices registered\n");
        ggml_backend_free(backend);
        return 1;
    }

    ggml_backend_free(backend);
    printf("  PASSED\n\n");
    return 0;
}

// =============================================================================
// Test 2: FP32 MUL_MAT - Small matrix (2x4x3)
// =============================================================================

static int test_mul_mat_small() {
    printf("Test 2: FP32 MUL_MAT - Small matrix (2x4x3)\n");

    const int M = 2, N = 4, K = 3;

    ggml_backend_t backend = ggml_backend_npm_init();
    if (!backend) {
        printf("  FAILED: Could not create NPM backend\n");
        return 1;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("  FAILED: Could not create ggml context\n");
        ggml_backend_free(backend);
        return 1;
    }

    // Create tensors
    struct ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor * input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * output  = ggml_mul_mat(ctx, weights, input);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        printf("  FAILED: Could not allocate backend buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    // Initialize data
    float weights_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};  // 4x3
    float input_data[]   = {1, 0, 0, 0, 1, 0};  // 2x3 (identity-like)
    float expected[8];
    ref_matmul_f32(input_data, weights_data, expected, M, N, K);

    ggml_backend_tensor_set(weights, weights_data, 0, sizeof(weights_data));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));

    // Compute
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        printf("  FAILED: Graph compute failed with status %d\n", status);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    // Verify
    float result[8];
    ggml_backend_tensor_get(output, result, 0, sizeof(result));

    float err = max_error(result, expected, N * M);
    printf("  Max error: %e\n", err);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (err < 1e-4f) {
        printf("  PASSED\n\n");
        return 0;
    } else {
        printf("  FAILED: Error too large\n\n");
        return 1;
    }
}

// =============================================================================
// Test 3: FP32 MUL_MAT - Medium matrix (64x128x64)
// =============================================================================

static int test_mul_mat_medium() {
    printf("Test 3: FP32 MUL_MAT - Medium matrix (64x128x64)\n");

    const int M = 64, N = 128, K = 64;

    ggml_backend_t backend = ggml_backend_npm_init();
    if (!backend) {
        printf("  FAILED: Could not create NPM backend\n");
        return 1;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 256 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("  FAILED: Could not create ggml context\n");
        ggml_backend_free(backend);
        return 1;
    }

    struct ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor * input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * output  = ggml_mul_mat(ctx, weights, input);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        printf("  FAILED: Could not allocate backend buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    // Initialize with deterministic pattern
    std::vector<float> weights_data(K * N);
    std::vector<float> input_data(K * M);
    std::vector<float> expected(N * M);

    init_deterministic_f32(weights_data.data(), K * N);
    init_deterministic_f32(input_data.data(), K * M);
    ref_matmul_f32(input_data.data(), weights_data.data(), expected.data(), M, N, K);

    ggml_backend_tensor_set(weights, weights_data.data(), 0, K * N * sizeof(float));
    ggml_backend_tensor_set(input, input_data.data(), 0, K * M * sizeof(float));

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        printf("  FAILED: Graph compute failed\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<float> result(N * M);
    ggml_backend_tensor_get(output, result.data(), 0, N * M * sizeof(float));

    float err = max_error(result.data(), expected.data(), N * M);
    int err_count = count_errors(result.data(), expected.data(), N * M);
    printf("  Max error: %e\n", err);
    printf("  Elements with error > 1e-4: %d / %d\n", err_count, N * M);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (err < 1e-4f) {
        printf("  PASSED\n\n");
        return 0;
    } else {
        printf("  FAILED: Error too large\n\n");
        return 1;
    }
}

// =============================================================================
// Test 4: FP32 MUL_MAT - Large matrix (256x512x256)
// =============================================================================

static int test_mul_mat_large() {
    printf("Test 4: FP32 MUL_MAT - Large matrix (256x512x256)\n");

    const int M = 256, N = 512, K = 256;

    ggml_backend_t backend = ggml_backend_npm_init();
    if (!backend) {
        printf("  FAILED: Could not create NPM backend\n");
        return 1;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 512 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("  FAILED: Could not create ggml context\n");
        ggml_backend_free(backend);
        return 1;
    }

    struct ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor * input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * output  = ggml_mul_mat(ctx, weights, input);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        printf("  FAILED: Could not allocate backend buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<float> weights_data(K * N);
    std::vector<float> input_data(K * M);
    std::vector<float> expected(N * M);

    init_random_f32(weights_data.data(), K * N, -0.5f, 0.5f);
    init_random_f32(input_data.data(), K * M, -0.5f, 0.5f);
    ref_matmul_f32(input_data.data(), weights_data.data(), expected.data(), M, N, K);

    ggml_backend_tensor_set(weights, weights_data.data(), 0, K * N * sizeof(float));
    ggml_backend_tensor_set(input, input_data.data(), 0, K * M * sizeof(float));

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        printf("  FAILED: Graph compute failed\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<float> result(N * M);
    ggml_backend_tensor_get(output, result.data(), 0, N * M * sizeof(float));

    float err = max_error(result.data(), expected.data(), N * M);
    int err_count = count_errors(result.data(), expected.data(), N * M);
    printf("  Max error: %e\n", err);
    printf("  Elements with error > 1e-4: %d / %d\n", err_count, N * M);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (err < 1e-3f) {  // Slightly higher tolerance for large matrices
        printf("  PASSED\n\n");
        return 0;
    } else {
        printf("  FAILED: Error too large\n\n");
        return 1;
    }
}

// =============================================================================
// Test 5: FP32 MUL_MAT - Non-square matrix (32x1024x64)
// =============================================================================

static int test_mul_mat_nonsquare() {
    printf("Test 5: FP32 MUL_MAT - Non-square matrix (32x1024x64)\n");

    const int M = 32, N = 1024, K = 64;

    ggml_backend_t backend = ggml_backend_npm_init();
    if (!backend) {
        printf("  FAILED: Could not create NPM backend\n");
        return 1;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 256 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("  FAILED: Could not create ggml context\n");
        ggml_backend_free(backend);
        return 1;
    }

    struct ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor * input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * output  = ggml_mul_mat(ctx, weights, input);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        printf("  FAILED: Could not allocate backend buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<float> weights_data(K * N);
    std::vector<float> input_data(K * M);
    std::vector<float> expected(N * M);

    init_deterministic_f32(weights_data.data(), K * N);
    init_deterministic_f32(input_data.data(), K * M);
    ref_matmul_f32(input_data.data(), weights_data.data(), expected.data(), M, N, K);

    ggml_backend_tensor_set(weights, weights_data.data(), 0, K * N * sizeof(float));
    ggml_backend_tensor_set(input, input_data.data(), 0, K * M * sizeof(float));

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        printf("  FAILED: Graph compute failed\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<float> result(N * M);
    ggml_backend_tensor_get(output, result.data(), 0, N * M * sizeof(float));

    float err = max_error(result.data(), expected.data(), N * M);
    printf("  Max error: %e\n", err);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (err < 1e-4f) {
        printf("  PASSED\n\n");
        return 0;
    } else {
        printf("  FAILED: Error too large\n\n");
        return 1;
    }
}

// =============================================================================
// Test 6: Multiple independent MUL_MAT operations in one graph
// =============================================================================

static int test_mul_mat_multiple() {
    printf("Test 6: Multiple independent MUL_MAT operations in one graph\n");

    const int M = 64, N = 128, K = 64;
    const int num_ops = 5;

    ggml_backend_t backend = ggml_backend_npm_init();
    if (!backend) {
        printf("  FAILED: Could not create NPM backend\n");
        return 1;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 512 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("  FAILED: Could not create ggml context\n");
        ggml_backend_free(backend);
        return 1;
    }

    // Create multiple independent matmuls
    std::vector<struct ggml_tensor*> weights_tensors(num_ops);
    std::vector<struct ggml_tensor*> input_tensors(num_ops);
    std::vector<struct ggml_tensor*> output_tensors(num_ops);

    for (int i = 0; i < num_ops; i++) {
        weights_tensors[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
        input_tensors[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        output_tensors[i] = ggml_mul_mat(ctx, weights_tensors[i], input_tensors[i]);
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        printf("  FAILED: Could not allocate backend buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    // Initialize data for all operations
    for (int i = 0; i < num_ops; i++) {
        std::vector<float> w_data(K * N);
        std::vector<float> in_data(K * M);
        init_deterministic_f32(w_data.data(), K * N);
        init_deterministic_f32(in_data.data(), K * M);
        ggml_backend_tensor_set(weights_tensors[i], w_data.data(), 0, K * N * sizeof(float));
        ggml_backend_tensor_set(input_tensors[i], in_data.data(), 0, K * M * sizeof(float));
    }

    // Build graph with all operations
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    for (int i = 0; i < num_ops; i++) {
        ggml_build_forward_expand(graph, output_tensors[i]);
    }

    printf("  Graph nodes: %d\n", ggml_graph_n_nodes(graph));
    printf("  Operations: %d independent MUL_MATs\n", num_ops);

    enum ggml_status status = ggml_backend_graph_compute(backend, graph);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (status == GGML_STATUS_SUCCESS) {
        printf("  PASSED\n\n");
        return 0;
    } else {
        printf("  FAILED: Graph compute failed with status %d\n\n", status);
        return 1;
    }
}

// =============================================================================
// Test 7: supports_op verification
// =============================================================================

static int test_supports_op() {
    printf("Test 7: supports_op verification\n");

    ggml_backend_reg_t reg = ggml_backend_npm_reg();
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("  FAILED: Could not create ggml context\n");
        return 1;
    }

    // Test various operations
    int passed = 0;
    int total = 0;

    // MUL_MAT with FP32 - should be supported
    {
        struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 32);
        struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        bool supported = ggml_backend_dev_supports_op(dev, y);
        printf("  MUL_MAT (FP32, FP32): %s\n", supported ? "supported" : "NOT supported");
        if (supported) passed++;
        total++;
    }

    // RESHAPE - should be supported
    {
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        struct ggml_tensor * y = ggml_reshape_1d(ctx, x, 64 * 64);
        bool supported = ggml_backend_dev_supports_op(dev, y);
        printf("  RESHAPE: %s\n", supported ? "supported" : "NOT supported");
        if (supported) passed++;
        total++;
    }

    // VIEW - should be supported
    {
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        struct ggml_tensor * y = ggml_view_1d(ctx, x, 64, 0);
        bool supported = ggml_backend_dev_supports_op(dev, y);
        printf("  VIEW: %s\n", supported ? "supported" : "NOT supported");
        if (supported) passed++;
        total++;
    }

    // ADD - should NOT be supported (falls back to CPU)
    {
        struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        struct ggml_tensor * y = ggml_add(ctx, a, b);
        bool supported = ggml_backend_dev_supports_op(dev, y);
        printf("  ADD: %s (expected: NOT supported)\n", supported ? "supported" : "NOT supported");
        if (!supported) passed++;
        total++;
    }

    ggml_free(ctx);

    printf("  Results: %d/%d as expected\n", passed, total);
    if (passed == total) {
        printf("  PASSED\n\n");
        return 0;
    } else {
        printf("  FAILED: Some operations have unexpected support status\n\n");
        return 1;
    }
}

// =============================================================================
// Test 8: Buffer operations
// =============================================================================

static int test_buffer_operations() {
    printf("Test 8: Buffer operations\n");

    ggml_backend_t backend = ggml_backend_npm_init();
    if (!backend) {
        printf("  FAILED: Could not create NPM backend\n");
        return 1;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("  FAILED: Could not create ggml context\n");
        ggml_backend_free(backend);
        return 1;
    }

    // Test buffer allocation
    struct ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        printf("  FAILED: Could not allocate buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    size_t buffer_size = ggml_backend_buffer_get_size(buffer);
    printf("  Buffer size: %zu bytes (%.2f MB)\n", buffer_size, buffer_size / (1024.0 * 1024.0));

    // Test tensor_set and tensor_get
    std::vector<float> data(1024 * 1024);
    init_random_f32(data.data(), data.size());

    ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));

    std::vector<float> readback(1024 * 1024);
    ggml_backend_tensor_get(tensor, readback.data(), 0, readback.size() * sizeof(float));

    // Verify data integrity
    bool data_ok = true;
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] != readback[i]) {
            data_ok = false;
            printf("  Data mismatch at index %zu: expected %f, got %f\n", i, data[i], readback[i]);
            break;
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (data_ok) {
        printf("  Data integrity verified\n");
        printf("  PASSED\n\n");
        return 0;
    } else {
        printf("  FAILED: Data corruption detected\n\n");
        return 1;
    }
}

// =============================================================================
// Test 9: Batch dimension MUL_MAT (3D tensor)
// =============================================================================

static int test_mul_mat_batched() {
    printf("Test 9: Batched MUL_MAT (3D tensor with batch dimension)\n");

    const int M = 32, N = 64, K = 32;
    const int batch = 4;

    ggml_backend_t backend = ggml_backend_npm_init();
    if (!backend) {
        printf("  FAILED: Could not create NPM backend\n");
        return 1;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 256 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("  FAILED: Could not create ggml context\n");
        ggml_backend_free(backend);
        return 1;
    }

    // 3D tensors with batch dimension
    struct ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);  // Shared weights
    struct ggml_tensor * input   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, M, batch);
    struct ggml_tensor * output  = ggml_mul_mat(ctx, weights, input);

    printf("  Input shape: (%ld, %ld, %ld)\n", (long)input->ne[0], (long)input->ne[1], (long)input->ne[2]);
    printf("  Weights shape: (%ld, %ld)\n", (long)weights->ne[0], (long)weights->ne[1]);
    printf("  Output shape: (%ld, %ld, %ld)\n", (long)output->ne[0], (long)output->ne[1], (long)output->ne[2]);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        printf("  FAILED: Could not allocate backend buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    // Initialize data
    std::vector<float> weights_data(K * N);
    std::vector<float> input_data(K * M * batch);
    init_deterministic_f32(weights_data.data(), K * N);
    init_deterministic_f32(input_data.data(), K * M * batch);

    ggml_backend_tensor_set(weights, weights_data.data(), 0, K * N * sizeof(float));
    ggml_backend_tensor_set(input, input_data.data(), 0, K * M * batch * sizeof(float));

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    enum ggml_status status = ggml_backend_graph_compute(backend, graph);

    // Verify each batch
    bool all_ok = true;
    if (status == GGML_STATUS_SUCCESS) {
        std::vector<float> result(N * M * batch);
        ggml_backend_tensor_get(output, result.data(), 0, N * M * batch * sizeof(float));

        for (int b = 0; b < batch; b++) {
            std::vector<float> expected_batch(N * M);
            ref_matmul_f32(input_data.data() + b * K * M, weights_data.data(),
                          expected_batch.data(), M, N, K);

            float err = max_error(result.data() + b * N * M, expected_batch.data(), N * M);
            if (err > 1e-4f) {
                printf("  Batch %d: error = %e (FAIL)\n", b, err);
                all_ok = false;
            }
        }
    } else {
        all_ok = false;
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (all_ok && status == GGML_STATUS_SUCCESS) {
        printf("  All batches verified\n");
        printf("  PASSED\n\n");
        return 0;
    } else {
        printf("  FAILED\n\n");
        return 1;
    }
}

// =============================================================================
// Test 10: Quantized weight support check
// =============================================================================

static int test_quantized_support() {
    printf("Test 10: Quantized weight support verification\n");

    ggml_backend_reg_t reg = ggml_backend_npm_reg();
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("  FAILED: Could not create ggml context\n");
        return 1;
    }

    // Test Q4_K support (K must be multiple of 256)
    {
        struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 256, 64);
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 32);
        struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        bool supported = ggml_backend_dev_supports_op(dev, y);
        printf("  MUL_MAT (Q4_K weights, FP32 input, K=256): %s\n",
               supported ? "supported" : "NOT supported");
    }

    // Test Q8_0 support (K must be multiple of 32)
    {
        struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 64, 64);
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 32);
        struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        bool supported = ggml_backend_dev_supports_op(dev, y);
        printf("  MUL_MAT (Q8_0 weights, FP32 input, K=64): %s\n",
               supported ? "supported" : "NOT supported");
    }

    // Test unsupported: FP16 input (not supported)
    {
        struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 32);
        struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        bool supported = ggml_backend_dev_supports_op(dev, y);
        printf("  MUL_MAT (FP32 weights, FP16 input): %s (expected: NOT supported)\n",
               supported ? "supported" : "NOT supported");
    }

    ggml_free(ctx);
    printf("  PASSED (informational test)\n\n");
    return 0;
}

// =============================================================================
// Test 11: Q8_0 quantized matmul execution
// =============================================================================

static int test_quantized_q8_0_matmul() {
    printf("Test 11: Q8_0 quantized matmul execution (64x64x64)\n");

    const int M = 64, N = 64, K = 64;  // K must be multiple of 32 for Q8_0

    ggml_backend_t backend = ggml_backend_npm_init();
    if (!backend) {
        printf("  FAILED: Could not create NPM backend\n");
        return 1;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 256 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("  FAILED: Could not create ggml context\n");
        ggml_backend_free(backend);
        return 1;
    }

    // Q8_0 weights, FP32 input
    struct ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    struct ggml_tensor * input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * output  = ggml_mul_mat(ctx, weights, input);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        printf("  FAILED: Could not allocate backend buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    // Initialize FP32 data, then quantize
    std::vector<float> weights_f32(K * N);
    std::vector<float> input_data(K * M);
    init_deterministic_f32(weights_f32.data(), K * N);
    init_deterministic_f32(input_data.data(), K * M);

    // Quantize weights to Q8_0
    size_t q8_size = ggml_row_size(GGML_TYPE_Q8_0, K * N);
    std::vector<uint8_t> weights_q8(q8_size);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, weights_f32.data(), weights_q8.data(), 0, N, K, nullptr);

    ggml_backend_tensor_set(weights, weights_q8.data(), 0, q8_size);
    ggml_backend_tensor_set(input, input_data.data(), 0, K * M * sizeof(float));

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        printf("  FAILED: Graph compute failed with status %d\n", status);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    // Get result and verify it's reasonable
    std::vector<float> result(N * M);
    ggml_backend_tensor_get(output, result.data(), 0, N * M * sizeof(float));

    // Compute approximate expected values using FP32 reference
    // (quantization introduces small errors, so we use looser tolerance)
    std::vector<float> expected(N * M);
    ref_matmul_f32(input_data.data(), weights_f32.data(), expected.data(), M, N, K);

    float err = max_error(result.data(), expected.data(), N * M);
    int err_count = count_errors(result.data(), expected.data(), N * M, 0.1f);
    printf("  Max error vs FP32 reference: %e\n", err);
    printf("  Elements with error > 0.1: %d / %d\n", err_count, N * M);

    // Check output is not all zeros or NaN
    bool has_nonzero = false;
    bool has_nan = false;
    for (size_t i = 0; i < result.size(); i++) {
        if (result[i] != 0.0f) has_nonzero = true;
        if (std::isnan(result[i])) has_nan = true;
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (!has_nonzero) {
        printf("  FAILED: Output is all zeros\n\n");
        return 1;
    }
    if (has_nan) {
        printf("  FAILED: Output contains NaN\n\n");
        return 1;
    }
    // Allow higher error for quantized (quantization error is expected)
    if (err < 1.0f && err_count < (N * M / 10)) {
        printf("  PASSED\n\n");
        return 0;
    } else {
        printf("  FAILED: Error too large\n\n");
        return 1;
    }
}

// =============================================================================
// Test 12: Repeated matmul with same weights (weight caching)
// =============================================================================

static int test_repeated_matmul() {
    printf("Test 12: Repeated matmul with same weights (simulates inference)\n");

    const int M = 32, N = 128, K = 64;
    const int iterations = 10;

    ggml_backend_t backend = ggml_backend_npm_init();
    if (!backend) {
        printf("  FAILED: Could not create NPM backend\n");
        return 1;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 256 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("  FAILED: Could not create ggml context\n");
        ggml_backend_free(backend);
        return 1;
    }

    struct ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor * input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * output  = ggml_mul_mat(ctx, weights, input);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        printf("  FAILED: Could not allocate backend buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    // Initialize weights once (simulates loaded model weights)
    std::vector<float> weights_data(K * N);
    init_deterministic_f32(weights_data.data(), K * N);
    ggml_backend_tensor_set(weights, weights_data.data(), 0, K * N * sizeof(float));

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    bool all_ok = true;
    for (int iter = 0; iter < iterations; iter++) {
        // Different input each iteration (simulates different tokens)
        std::vector<float> input_data(K * M);
        for (size_t i = 0; i < input_data.size(); i++) {
            input_data[i] = (float)((i + iter * 17) % 10) * 0.1f;
        }
        ggml_backend_tensor_set(input, input_data.data(), 0, K * M * sizeof(float));

        enum ggml_status status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            printf("  Iteration %d FAILED\n", iter);
            all_ok = false;
            break;
        }

        // Verify result
        std::vector<float> result(N * M);
        ggml_backend_tensor_get(output, result.data(), 0, N * M * sizeof(float));

        std::vector<float> expected(N * M);
        ref_matmul_f32(input_data.data(), weights_data.data(), expected.data(), M, N, K);

        float err = max_error(result.data(), expected.data(), N * M);
        if (err > 1e-4f) {
            printf("  Iteration %d: error = %e (FAIL)\n", iter, err);
            all_ok = false;
            break;
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (all_ok) {
        printf("  Completed %d iterations successfully\n", iterations);
        printf("  PASSED\n\n");
        return 0;
    } else {
        printf("  FAILED\n\n");
        return 1;
    }
}

// =============================================================================
// Test 13: Edge case - single row/column matrices
// =============================================================================

static int test_edge_cases() {
    printf("Test 13: Edge cases - single row/column matrices\n");

    ggml_backend_t backend = ggml_backend_npm_init();
    if (!backend) {
        printf("  FAILED: Could not create NPM backend\n");
        return 1;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("  FAILED: Could not create ggml context\n");
        ggml_backend_free(backend);
        return 1;
    }

    int tests_passed = 0;
    int total_tests = 0;

    // Test: M=1 (single batch, common in autoregressive inference)
    {
        total_tests++;
        const int M = 1, N = 64, K = 32;

        struct ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
        struct ggml_tensor * input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        struct ggml_tensor * output  = ggml_mul_mat(ctx, weights, input);

        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buffer) {
            std::vector<float> w_data(K * N), in_data(K * M), expected(N * M);
            init_deterministic_f32(w_data.data(), K * N);
            init_deterministic_f32(in_data.data(), K * M);
            ref_matmul_f32(in_data.data(), w_data.data(), expected.data(), M, N, K);

            ggml_backend_tensor_set(weights, w_data.data(), 0, K * N * sizeof(float));
            ggml_backend_tensor_set(input, in_data.data(), 0, K * M * sizeof(float));

            struct ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, output);

            if (ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS) {
                std::vector<float> result(N * M);
                ggml_backend_tensor_get(output, result.data(), 0, N * M * sizeof(float));
                float err = max_error(result.data(), expected.data(), N * M);
                if (err < 1e-4f) {
                    printf("  M=1 (single batch): PASSED (err=%e)\n", err);
                    tests_passed++;
                } else {
                    printf("  M=1 (single batch): FAILED (err=%e)\n", err);
                }
            }
            ggml_backend_buffer_free(buffer);
        }
    }

    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  Edge case tests: %d/%d passed\n", tests_passed, total_tests);
    if (tests_passed == total_tests) {
        printf("  PASSED\n\n");
        return 0;
    } else {
        printf("  FAILED\n\n");
        return 1;
    }
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║   NPM Backend Comprehensive Test Suite (GGML API)        ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    int failures = 0;

    failures += test_backend_init();           // Test 1
    failures += test_mul_mat_small();          // Test 2
    failures += test_mul_mat_medium();         // Test 3
    failures += test_mul_mat_large();          // Test 4
    failures += test_mul_mat_nonsquare();      // Test 5
    failures += test_mul_mat_multiple();        // Test 6
    failures += test_supports_op();            // Test 7
    failures += test_buffer_operations();      // Test 8
    failures += test_mul_mat_batched();        // Test 9
    failures += test_quantized_support();      // Test 10
    failures += test_quantized_q8_0_matmul();  // Test 11
    failures += test_repeated_matmul();        // Test 12
    failures += test_edge_cases();             // Test 13

    printf("╔══════════════════════════════════════════════════════════╗\n");
    if (failures == 0) {
        printf("║   All tests PASSED!                                      ║\n");
    } else {
        printf("║   %d test(s) FAILED                                       ║\n", failures);
    }
    printf("╚══════════════════════════════════════════════════════════╝\n");

    return failures;
}
