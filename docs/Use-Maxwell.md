# Experimental Maxwell SM52 build

This opt-in path builds Momoe's own GGML CUDA engine for a Maxwell compute-capability 5.2 card, such as the Quadro M4000. It uses CUDA 12.4 or newer within CUDA 12. CUDA 13 removed compilation for Maxwell. This is a source build, not a ready-made engine or a CPU fallback.

## Build

Use a CUDA 12 toolkit and a compatible installed driver. Keep the existing driver and model files; no firmware or kernel change is required by this build.

```bash
MOMOE_MAXWELL=1 NVCC=/usr/local/cuda-12.9/bin/nvcc JOBS=2 \
  sh src/metal/build_cuda.sh
```

The engine and check are written to `build-cuda-sm52/`, leaving the normal `build-cuda/` output untouched. Without `MOMOE_MAXWELL=1`, the normal build path is unchanged. The CMake option is `MOMOE_EXPERIMENTAL_SM52=ON`; it selects SM52 explicitly and rejects toolkits outside CUDA 12.4 to 12.x.

## Check before serving

Stop competing GPU work through its normal controller before testing. First run the bounded model-free check on the actual SM52 device:

```bash
./build-cuda-sm52/momoe-cuda-sm52-check
```

The check evaluates ten small matrix products: F32, F16, Q4_0, Q4_K and Q6_K, each with one and four input columns. CUDA results must match a scalar CPU reference within the stated absolute-plus-relative tolerance. It exits nonzero for another GPU architecture, unsupported operations, non-finite results or a numerical mismatch. It performs no throughput loop and loads no model.

Two mapped-memory mailbox checks also verify GPU-to-CPU payload visibility and CPU-to-GPU wakeups, with the completion flag already set and with a delayed host response. The host wait has a five-second deadline; the production GPU wait retains its existing clock-based timeout and memory fences. Pre-SM70 devices poll without the unsupported `__nanosleep` instruction.

This check does not establish full-model compatibility, capacity or performance. A model must still use an architecture supported by Momoe's own engine and fit its dense weights, KV state and working buffers into available memory. Validate a small, bounded model case separately before exposing a serving profile. Do not label an unloaded API as proof of successful inference.

## Validation status

The source build and actual SM52 checks are pending. No successful M4000 inference or numerical result is claimed by this document yet.
