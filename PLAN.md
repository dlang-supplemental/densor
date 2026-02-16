= Implementation Plan: DGML

== Phase 1: Core Tensor Ops
- Implement `struct Tensor` wrapping `Slice!(float*, N)`.
- Implement `matmul` (Matrix Multiplication).
    - Start with naive implementation.
    - Optimize with tiled block multiplication.
    - Add SIMD support via `core.simd` or `ldc.simd`.

== Phase 2: GGUF Parsing
- Implement a GGUF reader in `source/dgml/format/gguf.d`.
- Map GGUF tensor types (F16, Q4_0, Q8_0) to internal storage.
- Implement dequantization kernels (Q4_0 -> F32) for inference.

== Phase 3: CLIP Architecture
- Implement the specific layers needed for CLIP (Vision Transformer).
- `Conv2d` (for patch embedding).
- `LayerNorm`.
- `MultiHeadAttention`.
- `MLP`.

== Phase 4: Validation
- Compare outputs layer-by-layer against Python `transformers` implementation using a known seed/input.
