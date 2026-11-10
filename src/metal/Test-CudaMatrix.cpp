// Bounded, model-free SM52 matrix checks against a scalar CPU reference.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static bool Test_Matrix(ggml_backend_t backend, ggml_type type, int columns) {
    constexpr int width = 256, rows = 32;
    ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
    if (!ctx) return false;
    ggml_tensor * a = ggml_new_tensor_2d(ctx, type, width, rows);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, columns);
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, c);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return false; }
    std::vector<float> weights(width * rows), inputs(width * columns), decoded(width * rows);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = (int((i * 7 + i / width * 3) % 16) - 8) / 8.0f;
    // Each Q8 block has an exactly representable scale, keeping the comparison strict.
    for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = i % 32 == 0 ? 127 / 16.0f : (int(i % 31) - 15) / 16.0f;
    std::vector<unsigned char> packed(ggml_nbytes(a));
    if (type == GGML_TYPE_F32) {
        std::memcpy(packed.data(), weights.data(), packed.size());
        decoded = weights;
    } else if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(weights.data(), reinterpret_cast<ggml_fp16_t *>(packed.data()), weights.size());
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(packed.data()), decoded.data(), decoded.size());
    } else {
        ggml_quantize_chunk(type, weights.data(), packed.data(), 0, rows, width, nullptr);
        ggml_get_type_traits(type)->to_float(packed.data(), decoded.data(), decoded.size());
    }
    ggml_backend_tensor_set(a, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(b, inputs.data(), 0, inputs.size() * sizeof(float));
    bool passed = ggml_backend_supports_op(backend, c) && ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    std::vector<float> output(rows * columns);
    double max_error = 0;
    if (passed) {
        ggml_backend_tensor_get(c, output.data(), 0, output.size() * sizeof(float));
        for (int column = 0; column < columns; ++column) {
            for (int row = 0; row < rows; ++row) {
                double expected = 0;
                for (int k = 0; k < width; ++k) expected += double(decoded[row * width + k]) * inputs[column * width + k];
                double actual = output[column * rows + row];
                double error = std::abs(actual - expected);
                max_error = std::max(max_error, error);
                if (!std::isfinite(actual) || error > 1e-4 * (1 + std::abs(expected))) passed = false;
            }
        }
    }
    std::printf("{\"type\":\"%s\",\"columns\":%d,\"max_abs_error\":%.9g,\"passed\":%s}\n",
                ggml_type_name(type), columns, max_error, passed ? "true" : "false");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

int main() {
    cudaDeviceProp device{};
    if (cudaGetDeviceProperties(&device, 0) != cudaSuccess || device.major != 5 || device.minor != 2) {
        std::fprintf(stderr, "This check requires an actual Maxwell SM52 CUDA device\n");
        return 2;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) return 2;
    bool passed = true;
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}) {
        for (int columns : {1, 4}) passed = Test_Matrix(backend, type, columns) && passed;
    }
    ggml_backend_free(backend);
    ggml_quantize_free();
    std::printf("{\"compute_capability\":\"5.2\",\"cases\":10,\"passed\":%s}\n", passed ? "true" : "false");
    return passed ? 0 : 1;
}
