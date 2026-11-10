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

One run, dated 2026-11-10, on a Quadro M4000 (compute capability 5.2, 8114 MiB) inside an Ubuntu 24.04 container
with CUDA 12.9.1 (nvcc 12.9.86) and driver 580.196.07:

- the build above produced `build-cuda-sm52/momoe-cuda` and the check;
- `momoe-cuda-sm52-check` passed all ten matrix cases (largest absolute error 1.90734863e-06, `q4_K` at four columns,
  every other case exact against the CPU reference) and both mailbox cases;
- before the mailbox polling fix the same source failed to compile for SM52 with `identifier "__nanosleep" is
  undefined`, and after it the check passed; configuring with CUDA 13 was rejected at CMake time;
- one bounded model case: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` with `--ctx 256 --pool-gib 0.25 --mtp 0` loaded
  (2.49 GiB dense weights, 0.75 GiB expert pool across 3 format groups), read a 25-token prompt at 3.3 tok/s
  and generated 4 tokens at 2.3 tok/s, ending in the model's end-of-sequence token;
- the `--serve` pipe protocol answered `READY 256 stop` and completed one `GEN` request on the same card,
  stopping at end of sequence.

What this does not claim: throughput or capacity figures worth comparing, concurrency, long runs, a serving
qualification, or that CUDA 13 builds - CUDA 13 removed Maxwell compilation, and no CUDA 13 result exists for
this path. These were single bounded checks on one card, not a benchmark.
