package ivfpq

import (
	"math"
	"strconv"
	"testing"
)

// parseHexF32 reconstructs float32 values from their raw u32 bit patterns (so the
// reference inputs/outputs cross the language boundary bit-exact).
func parseHexF32(t *testing.T, fields []string) []float32 {
	t.Helper()
	out := make([]float32, len(fields))
	for i, h := range fields {
		bits, err := strconv.ParseUint(h, 16, 32)
		if err != nil {
			t.Fatalf("bad hex f32 %q: %v", h, err)
		}
		out[i] = math.Float32frombits(uint32(bits))
	}
	return out
}

func atoiRef(t *testing.T, s string) int {
	t.Helper()
	v, err := strconv.Atoi(s)
	if err != nil {
		t.Fatalf("atoi %q: %v", s, err)
	}
	return v
}

// TestKMeansMatchesRustReference runs k-means over the exact points the Rust trainer
// used and asserts the assignments match exactly and the centroids match bit-for-bit
// — the cross-implementation conformance for the core primitive.
func TestKMeansMatchesRustReference(t *testing.T) {
	ref := loadRef(t)
	seed, err := strconv.ParseUint(ref["rng_seed"][0], 16, 64)
	if err != nil {
		t.Fatalf("parse seed: %v", err)
	}
	dim := atoiRef(t, ref["dim"][0])
	k := atoiRef(t, ref["k"][0])
	iters := atoiRef(t, ref["iters"][0])
	n := atoiRef(t, ref["n"][0])
	points := parseHexF32(t, ref["points"])
	wantCentroids := parseHexF32(t, ref["centroids"])
	wantAssign := ref["assign"]

	cent, assign := KMeans(points, n, dim, k, iters, NewRng(seed))

	if len(assign) != len(wantAssign) {
		t.Fatalf("assign len = %d, want %d", len(assign), len(wantAssign))
	}
	for i, a := range assign {
		if strconv.Itoa(int(a)) != wantAssign[i] {
			t.Errorf("assign[%d] = %d, want %s", i, a, wantAssign[i])
		}
	}
	if len(cent) != len(wantCentroids) {
		t.Fatalf("centroid len = %d, want %d", len(cent), len(wantCentroids))
	}
	for i := range cent {
		if math.Float32bits(cent[i]) != math.Float32bits(wantCentroids[i]) {
			t.Errorf("centroid[%d] bits = %08x, want %08x", i,
				math.Float32bits(cent[i]), math.Float32bits(wantCentroids[i]))
		}
	}
}

// TestKMeansRecoversSeparatedClusters is the standalone quality ("recall") test:
// k-means over well-separated synthetic clusters must perfectly partition them —
// every point of a true cluster shares one label, and distinct true clusters get
// distinct labels — with near-zero quantization error.
func TestKMeansRecoversSeparatedClusters(t *testing.T) {
	const dim, k, per = 4, 3, 12
	centers := [][]float32{{30, 0, 0, 0}, {0, 30, 0, 0}, {0, 0, 30, 0}}
	var pts []float32
	for c := range k {
		for j := range per {
			off := float32(j%3-1) * 0.05
			for d := range dim {
				pts = append(pts, centers[c][d]+off)
			}
		}
	}
	cent, assign := KMeans(pts, k*per, dim, k, 25, NewRng(0x5251564952525649))

	// Each true cluster's points share a label; labels across true clusters differ.
	labels := make([]uint32, k)
	seen := map[uint32]bool{}
	for c := range k {
		base := assign[c*per]
		labels[c] = base
		for j := range per {
			if assign[c*per+j] != base {
				t.Fatalf("true cluster %d split across labels: %v", c, assign[c*per:(c+1)*per])
			}
		}
		if seen[base] {
			t.Fatalf("two true clusters share label %d: %v", base, labels)
		}
		seen[base] = true
	}

	// Quantization error must be tiny (only the ±0.05 jitter), proving convergence.
	var sse float32
	for i := range k * per {
		cl := assign[i]
		for d := range dim {
			diff := pts[i*dim+d] - cent[int(cl)*dim+d]
			sse += diff * diff
		}
	}
	if sse > 1.0 {
		t.Errorf("quantization SSE = %g, want < 1.0 (clusters should be tight)", sse)
	}
}

// TestNormalizeUnitNorm checks normalize scales to unit L2 norm and passes a zero
// vector through unchanged.
func TestNormalizeUnitNorm(t *testing.T) {
	got := normalize([]float32{3, 4})
	if math.Abs(float64(got[0]-0.6)) > 1e-6 || math.Abs(float64(got[1]-0.8)) > 1e-6 {
		t.Errorf("normalize([3 4]) = %v, want [0.6 0.8]", got)
	}
	zero := normalize([]float32{0, 0, 0})
	for i, x := range zero {
		if x != 0 {
			t.Errorf("normalize(zero)[%d] = %g, want 0", i, x)
		}
	}
}
