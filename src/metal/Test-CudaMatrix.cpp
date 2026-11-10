// Bounded, model-free SM52 matrix checks against a scalar CPU reference.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-momoe-mailbox.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <thread>

static bool Test_Mailbox(ggml_backend_t backend, bool delayed) {
    struct Payload {
        alignas(64) volatile uint32_t epoch, sent, done, timeout;
        alignas(64) float input[32], output[32];
        alignas(64) int32_t ids[4];
    };
    Payload * host = nullptr;
    void * device = nullptr;
    if (cudaHostAlloc(reinterpret_cast<void **>(&host), sizeof(Payload), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) return false;
    if (cudaHostGetDevicePointer(&device, host, 0) != cudaSuccess) { cudaFreeHost(host); return false; }
    std::memset(host, 0, sizeof(Payload));
    host->epoch = 1;
    auto mapped = [&](const volatile void * value) {
        return static_cast<char *>(device) + (static_cast<const volatile char *>(value) - reinterpret_cast<const char *>(host));
    };
    float input[32], expected[32], output[32];
    int32_t ids[] = {-7, 3, 5, 11};
    for (int i = 0; i < 32; ++i) expected[i] = (input[i] = i / 8.0f) * 2 + ids[i % 4];
    if (!delayed) {
        std::memcpy(host->output, expected, sizeof(expected));
        __atomic_store_n(&host->done, 1u, __ATOMIC_RELEASE);
    }
    momoe_mailbox_op send{MOMOE_MAILBOX_MAGIC, MOMOE_MAILBOX_SEND, 32, 4, mapped(host->input), mapped(host->ids),
        reinterpret_cast<volatile uint32_t *>(mapped(&host->sent)), reinterpret_cast<const volatile uint32_t *>(mapped(&host->epoch)),
        reinterpret_cast<volatile uint32_t *>(mapped(&host->timeout))};
    momoe_mailbox_op wait{MOMOE_MAILBOX_MAGIC, MOMOE_MAILBOX_WAIT, 32, 0, mapped(host->output), nullptr,
        reinterpret_cast<volatile uint32_t *>(mapped(&host->done)), reinterpret_cast<const volatile uint32_t *>(mapped(&host->epoch)),
        reinterpret_cast<volatile uint32_t *>(mapped(&host->timeout))};
    ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
    if (!ctx) { cudaFreeHost(host); return false; }
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
    ggml_tensor * choices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    ggml_tensor * sources[] = {x, choices};
    ggml_tensor * sent = ggml_custom_4d(ctx, GGML_TYPE_I32, 1, 1, 1, 1, sources, 2, momoe_mailbox_cpu_stub, 1, &send);
    ggml_tensor * after[] = {sent};
    ggml_tensor * result = ggml_custom_4d(ctx, GGML_TYPE_F32, 32, 1, 1, 1, after, 1, momoe_mailbox_cpu_stub, 1, &wait);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, result);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); cudaFreeHost(host); return false; }
    ggml_backend_tensor_set(x, input, 0, sizeof(input));
    ggml_backend_tensor_set(choices, ids, 0, sizeof(ids));
    bool passed = ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS;
    if (passed && delayed) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (__atomic_load_n(&host->sent, __ATOMIC_ACQUIRE) != 1 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        passed = __atomic_load_n(&host->sent, __ATOMIC_ACQUIRE) == 1;
        if (passed) for (int i = 0; i < 32; ++i) host->output[i] = host->input[i] * 2 + host->ids[i % 4];
        __atomic_store_n(&host->done, 1u, __ATOMIC_RELEASE); // release the GPU even if the host timed out
    }
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(result, output, 0, sizeof(output));
    passed = passed && host->sent == 1 && host->timeout == 0;
    for (int i = 0; i < 32; ++i) passed = passed && host->input[i] == input[i] && output[i] == expected[i];
    for (int i = 0; i < 4; ++i) passed = passed && host->ids[i] == ids[i];
    std::printf("{\"check\":\"mailbox\",\"delayed\":%s,\"passed\":%s}\n", delayed ? "true" : "false", passed ? "true" : "false");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    cudaFreeHost(host);
    return passed;
}

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
    for (bool delayed : {false, true}) passed = Test_Mailbox(backend, delayed) && passed;
    ggml_backend_free(backend);
    ggml_quantize_free();
    std::printf("{\"compute_capability\":\"5.2\",\"matrix_cases\":10,\"mailbox_cases\":2,\"passed\":%s}\n", passed ? "true" : "false");
    return passed ? 0 : 1;
}
