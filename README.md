# go-ivfpq

A small, dependency-free **pure-Go IVFPQ vector-quantization trainer** — coarse
k-means (an IVF inverted-list quantizer) over the vectors, plus product-quantization
codebooks over the residuals. The build side of an IVF-ADC approximate
nearest-neighbor index, with no cgo and no FAISS.

It is the Go counterpart of the [roaringrange](https://github.com/freeeve/roaringrange)
Rust `vector_build` trainer (the build mirror of its `RRVI` similarity index).

## Status

Early. Landed so far:

- `Rng` — the deterministic xorshift64\* PRNG that seeds k-means.
- `KMeans` — Lloyd's k-means (assign → mean → reseed empty clusters), plus
  `normalize` for the inner-product metric.

Coming: `Train` (full IVFPQ: residuals + per-subspace PQ codebooks + inverted lists)
and `FromParts` (assemble an externally-trained model, e.g. exported from FAISS).

## Conformance

Two different bars, on purpose:

- **Bit-exact, where it is meaningful.** `Rng` is pure integer arithmetic and
  `KMeans` over well-separated data is bit-for-bit reproducible against the Rust
  trainer (same seed, same reduction order). `testdata/kmeans_ref.txt` is generated
  by the Rust side (every `float32` carried as its raw `u32` bits) and the Go tests
  assert an exact match.
- **Recall, where bit-exactness is a trap.** A quantized index is approximate by
  construction, and cross-language `float32` bit-exactness is fragile (Go may fuse
  `a*b+c` into an FMA on arm64 where Rust/LLVM does not). So the higher-level
  training output is validated by **recall@k vs a brute-force baseline**, not golden
  bytes.

## Scope boundary

The on-disk `RRVI`/`RRVR` serialization is **not** here — it belongs with the index
format in roaringrange. This package trains and returns the model as plain Go arrays
for that serializer (or any other) to consume.

## License

MIT — see [LICENSE](LICENSE).
