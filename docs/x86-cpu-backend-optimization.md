# x86 CPU Backend Optimization

This branch is a public, unmerged development snapshot of the x86 CPU backend
work submitted to NNTrainer. It collects the implementation stack from FP32
AVX2 kernels through FP16 GEMM/GEMV and FP16 CausalLM attention in one place.
The final code snapshot before this documentation is
[`e21b39e`](https://github.com/YongHyeon02/nntrainer/commit/e21b39ed6a5e1963eeae54fc4c025f11ec07a959).

The benchmark harness was submitted separately in
[PR #3820](https://github.com/nntrainer/nntrainer/pull/3820) and is not part of
this branch. Benchmark numbers below are self-measured results documented in
the linked PR descriptions; they were not independently reproduced by
upstream maintainers.

## Upstream PRs

| Area | Pull request |
| --- | --- |
| Reusable x86 CPU microbenchmark suite | [#3820](https://github.com/nntrainer/nntrainer/pull/3820) |
| FP32 AVX2 compute kernels | [#3834](https://github.com/nntrainer/nntrainer/pull/3834) |
| Cache-blocked FP16 GEMM/GEMV | [#3967](https://github.com/nntrainer/nntrainer/pull/3967) |
| FP16-input x86 attention | [#3988](https://github.com/nntrainer/nntrainer/pull/3988) |

See [all open and closed PRs by YongHyeon02](https://github.com/nntrainer/nntrainer/pulls?q=author%3AYongHyeon02)
for the later split-up series and current review state.

## Results

| Optimization | Result | Scope |
| --- | ---: | --- |
| FP32 AVX2 kernels | Up to **41.0x** | `tanh_gelu`, N=1024, versus scalar fallback |
| FP16 GEMV | Up to **18.93x** | 1x768, single thread, versus legacy convert+CBLAS path |
| FP16 GEMM | Up to **14.30x** | 1x4096x4096, six threads, versus legacy convert+CBLAS path |
| FP16 GEMM/GEMV allocations | **3 -> 0** | Steady-state per-call allocations using a reusable thread-local workspace |
| FP32-input QK scoring | **1.59x-1.84x** | Multiple sequence length, GQA, and head-dimension configurations |

The FP32 kernel suite also measured 36.6x for fused GELU-multiply and 17.3x
for softmax at their best reported sizes. These are per-kernel microbenchmarks,
not end-to-end model speedups.

### Benchmark environments

- FP32 kernels: Intel Core i5-12400F, Linux WSL2, GCC 11.4, 1,000 measured
  iterations after 50 warmups.
- FP16 GEMM/GEMV: Intel Core i5-12400F, Linux WSL2, GCC 12.3, 30 measured
  iterations after 8 warmups. The legacy and optimized paths were measured in
  the same binary.
- Attention: Intel Core i5-12400F, Linux WSL2, GCC 12.3, single-threaded,
  same-session interleaved A/B measurements against the pre-attention branch.

## Implementation

### FP32 AVX2 kernels

- Added AVX2 dispatch for activation, reduction, trigonometric, and
  element-wise operations that previously used scalar x86 fallbacks.
- Implemented vectorized softmax, GELU variants, max reduction, element-wise
  arithmetic, and FMA-based reciprocal-square-root refinement.
- Added scalar-reference correctness tests covering numerical tolerance,
  strided inputs, broadcast inputs, zero-length inputs, and edge cases.

Relevant code:
[`nntrainer/tensor/cpu_backend/x86/avx2_impl.cpp`](../nntrainer/tensor/cpu_backend/x86/avx2_impl.cpp)

### FP16 GEMM/GEMV

- Replaced the row-major FP16 path that created full FP32 copies of operands,
  called CBLAS, and narrowed the result back to FP16.
- Converts FP16 to FP32 while packing cache-sized panels and computes with
  AVX2+F16C micro-kernels, including a 6x16 primary kernel and edge kernels.
- Reuses thread-local workspace buffers and parallelizes output panels through
  NNTrainer's thread manager.
- Keeps column-major inputs on the legacy fallback for correctness.

Relevant code:
[`nntrainer/tensor/cpu_backend/x86/hgemm/`](../nntrainer/tensor/cpu_backend/x86/hgemm/)

### FP16 CausalLM attention

- Added x86 FP16-input QK, value-cache, and softmax kernels that previously
  existed only in the ARM backend.
- Uses F16C widening on load, FP32 accumulation, and FP16 narrowing on store.
- Fuses FP16-to-FP32 key conversion into QK dot-product loops and replaces
  repeated scratch allocation with reusable thread-local buffers.

Relevant code:
[`nntrainer/tensor/cpu_backend/x86/avx2_impl_fp16.cpp`](../nntrainer/tensor/cpu_backend/x86/avx2_impl_fp16.cpp)

## Limitations and status

- The work is present in this public fork but is not merged into NNTrainer's
  upstream `main` branch.
- The original PRs were closed by the repository's inactivity automation, not
  merged. Later smaller PRs remain visible in the author-filtered PR list.
- The new cache-blocked FP16 GEMM/GEMV path is row-major; column-major inputs
  use the legacy fallback.
- The AVX2+F16C target has no native FP16 FMA. FP16-input QK can therefore be
  slower than the FP32-input kernel at longer sequence lengths even though it
  enables FP16-mode inference and reduces KV-cache data size.
- PRs and commit history preserve co-author and maintainer attribution for
  collaborative changes.
