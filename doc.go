// Package ivfpq is a small, dependency-free pure-Go trainer for IVFPQ vector
// indexes: coarse k-means (an IVF inverted-list quantizer) over the vectors plus
// product-quantization codebooks over the residuals — the build side of an IVF-ADC
// approximate nearest-neighbor index.
//
// It is the Go counterpart of the roaringrange Rust vector_build trainer. The
// generic primitives ([Rng], [KMeans]) are reproducible against the Rust
// implementation given the same seed and inputs — the testdata reference fixture
// asserts the Rng sequence and a k-means run bit-for-bit. The higher-level training
// output is validated by recall rather than bit-equality, because a quantized index
// is approximate by construction and chasing cross-language float bit-exactness
// (FMA fusion differs between Go and Rust/LLVM) buys nothing.
//
// The on-disk RRVI/RRVR serialization is intentionally NOT part of this package: it
// belongs with the index format, in roaringrange. This package returns the trained
// model as plain Go arrays for that serializer (or any other) to consume.
package ivfpq
