package ivfpq

// Rng is the deterministic xorshift64* PRNG that seeds k-means, so a build is
// reproducible without any external dependency. It is a byte-for-byte port of the
// roaringrange Rust vector_build Rng: given the same seed it yields the identical
// u64 sequence (asserted by the testdata reference fixture).
type Rng struct {
	state uint64
}

// rngGoldenRatio replaces a zero seed so the generator never degenerates (matches
// the Rust trainer).
const rngGoldenRatio = 0x9E3779B97F4A7C15

// rngMix is the xorshift64* output multiplier.
const rngMix = 0x2545F4914F6CDD1D

// NewRng returns an Rng seeded with seed; a zero seed is replaced by the
// golden-ratio constant.
func NewRng(seed uint64) *Rng {
	if seed == 0 {
		seed = rngGoldenRatio
	}
	return &Rng{state: seed}
}

// NextU64 advances the generator and returns the next 64-bit value. The state is
// updated to the shifted word, and the returned value is that word times the mix
// constant (unsigned multiply wraps mod 2^64, matching Rust's wrapping_mul).
func (r *Rng) NextU64() uint64 {
	x := r.state
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	r.state = x
	return x * rngMix
}

// NextIndex returns a pseudo-random index in [0, n); n must be nonzero.
func (r *Rng) NextIndex(n int) int {
	return int(r.NextU64() % uint64(n))
}
