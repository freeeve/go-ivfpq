package ivfpq

import (
	"math"
	"sort"
	"strconv"
	"testing"
)

// TestTrainMatchesRustReference trains over the exact corpus the Rust build_ivfpq
// used and asserts the cross-language-robust parts match: the coarse centroids
// bit-for-bit and the per-cluster list membership exactly. The PQ codebooks/codes
// are NOT byte-asserted — residual k-means is FMA-sensitive across languages, so PQ
// quality is checked by TestTrainRecall instead.
func TestTrainMatchesRustReference(t *testing.T) {
	ref := loadRefFile(t, "ivfpq_ref.txt")
	dim := atoiRef(t, ref["dim"][0])
	nlist := atoiRef(t, ref["nlist"][0])
	m := atoiRef(t, ref["m"][0])
	nbits := atoiRef(t, ref["nbits"][0])
	metric := atoiRef(t, ref["metric"][0])
	n := atoiRef(t, ref["n"][0])
	flat := parseHexF32(t, ref["vectors"])

	vectors := make([]Vector, n)
	for i := range n {
		vectors[i] = Vector{ID: uint32(i), Vec: flat[i*dim : (i+1)*dim]}
	}
	p := Params{
		Dim: dim, Nlist: nlist, M: m, Nbits: uint8(nbits),
		Metric: Metric(metric), KmeansIters: 10, Seed: 0x5251564952525649,
	}
	model, err := Train(vectors, p)
	if err != nil {
		t.Fatalf("Train: %v", err)
	}

	if model.Dim != dim || model.Nlist != nlist || model.M != m ||
		int(model.Nbits) != nbits || uint8(model.Metric) != uint8(metric) || model.N != uint64(n) {
		t.Fatalf("header mismatch: %+v", model)
	}
	if len(model.Codebooks) != m*(1<<nbits)*(dim/m) {
		t.Errorf("codebooks length = %d, want %d", len(model.Codebooks), m*(1<<nbits)*(dim/m))
	}

	// Coarse centroids: bit-for-bit with Rust.
	wantCent := parseHexF32(t, ref["centroids"])
	for i := range model.Centroids {
		if math.Float32bits(model.Centroids[i]) != math.Float32bits(wantCent[i]) {
			t.Errorf("centroid[%d] bits = %08x, want %08x", i,
				math.Float32bits(model.Centroids[i]), math.Float32bits(wantCent[i]))
		}
	}

	// List membership: exact (the coarse assignment is robust for separated data).
	for c := range nlist {
		if got, want := strconv.Itoa(len(model.ListIDs[c])), ref["list_sizes"][c]; got != want {
			t.Errorf("list %d size = %s, want %s", c, got, want)
		}
	}
	var gotIDs []string
	for c := range nlist {
		for _, id := range model.ListIDs[c] {
			gotIDs = append(gotIDs, strconv.FormatUint(uint64(id), 10))
		}
	}
	assertStrSliceEqual(t, "list_ids", gotIDs, ref["list_ids"])
}

// TestFromPartsScatterAndValidate checks FromParts scatters codes into the right
// lists and rejects malformed inputs.
func TestFromPartsScatterAndValidate(t *testing.T) {
	parts := Parts{
		Dim: 2, Nlist: 2, M: 2, Nbits: 8, Metric: L2,
		Centroids:   []float32{0, 0, 10, 10},
		Codebooks:   make([]float32, 2*256*1),
		IDs:         []uint32{7, 8, 9},
		Assignments: []uint32{0, 1, 0},
		Codes:       []byte{1, 2, 3, 4, 5, 6}, // 3 vectors × m=2
	}
	model, err := FromParts(parts)
	if err != nil {
		t.Fatalf("FromParts: %v", err)
	}
	if len(model.ListIDs[0]) != 2 || model.ListIDs[0][0] != 7 || model.ListIDs[0][1] != 9 {
		t.Errorf("list 0 ids = %v, want [7 9]", model.ListIDs[0])
	}
	if len(model.ListIDs[1]) != 1 || model.ListIDs[1][0] != 8 {
		t.Errorf("list 1 ids = %v, want [8]", model.ListIDs[1])
	}
	// codes for id 9 (second entry of list 0) must be {5,6}.
	if got := model.ListCodes[0][2:4]; got[0] != 5 || got[1] != 6 {
		t.Errorf("list 0 codes for id 9 = %v, want [5 6]", got)
	}

	bad := parts
	bad.Assignments = []uint32{0, 2, 0} // 2 >= nlist
	if _, err := FromParts(bad); err == nil {
		t.Errorf("FromParts accepted an assignment >= nlist")
	}
}

// TestTrainRecall is the PQ-quality check: over clustered synthetic vectors, the
// trained index's reconstructions (centroid + PQ-decoded residual) must recover most
// of each query's exact nearest neighbors. recall@10 against a brute-force baseline
// must clear a healthy bar.
func TestTrainRecall(t *testing.T) {
	const n, dim, k = 400, 16, 8
	vectors := clusteredVectors(n, dim, k, 0x1234_5678)
	p := DefaultParams(dim, 16, 8) // InnerProduct, nbits 8, m=8 (dsub=2)
	p.KmeansIters = 15
	model, err := Train(vectors, p)
	if err != nil {
		t.Fatalf("Train: %v", err)
	}

	// Reconstructed vectors keyed by doc ID, in normalized (inner-product) space.
	recon := make(map[uint32][]float32)
	for c := range model.Nlist {
		for j := range model.ListIDs[c] {
			recon[model.ListIDs[c][j]] = model.Reconstruct(c, j)
		}
	}
	if len(recon) != n {
		t.Fatalf("reconstructed %d vectors, want %d", len(recon), n)
	}

	const topK = 10
	var hits, total int
	for q := range n {
		query := normalize(vectors[q].Vec)
		exact := topKByDot(query, vectors, normalizeOriginal, topK)
		approx := topKByDotRecon(query, recon, topK)
		hits += overlap(exact, approx)
		total += topK
	}
	recall := float64(hits) / float64(total)
	t.Logf("recall@%d = %.3f over %d queries", topK, recall, n)
	if recall < 0.80 {
		t.Errorf("recall@%d = %.3f, want >= 0.80", topK, recall)
	}
}

// --- test helpers ---

func assertStrSliceEqual(t *testing.T, name string, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("%s[%d] = %s, want %s", name, i, got[i], want[i])
		}
	}
}

// clusteredVectors builds n vectors in dim dims drawn from k random centers plus
// noise — deterministic from seed, so the recall bar is stable.
func clusteredVectors(n, dim, k int, seed uint64) []Vector {
	rng := NewRng(seed)
	centers := make([][]float32, k)
	for c := range k {
		centers[c] = make([]float32, dim)
		for d := range dim {
			centers[c][d] = unitFloat(rng)
		}
	}
	vs := make([]Vector, n)
	for i := range n {
		c := i % k
		v := make([]float32, dim)
		for d := range dim {
			v[d] = centers[c][d] + 0.15*unitFloat(rng)
		}
		vs[i] = Vector{ID: uint32(i), Vec: v}
	}
	return vs
}

// unitFloat returns a pseudo-random float in [-1, 1) from the generator's top bits.
func unitFloat(rng *Rng) float32 {
	return float32(rng.NextU64()>>40)/float32(1<<23) - 1.0
}

func normalizeOriginal(v Vector) []float32 { return normalize(v.Vec) }

type scored struct {
	id uint32
	s  float32
}

// topKByDot returns the IDs of the topK vectors by inner product with query, scoring
// each via project.
func topKByDot(query []float32, vectors []Vector, project func(Vector) []float32, topK int) []uint32 {
	scores := make([]scored, len(vectors))
	for i, v := range vectors {
		scores[i] = scored{v.ID, dot(query, project(v))}
	}
	return topIDs(scores, topK)
}

func topKByDotRecon(query []float32, recon map[uint32][]float32, topK int) []uint32 {
	scores := make([]scored, 0, len(recon))
	for id, v := range recon {
		scores = append(scores, scored{id, dot(query, v)})
	}
	return topIDs(scores, topK)
}

func topIDs(scores []scored, topK int) []uint32 {
	sort.Slice(scores, func(i, j int) bool {
		if scores[i].s != scores[j].s {
			return scores[i].s > scores[j].s
		}
		return scores[i].id < scores[j].id
	})
	if topK > len(scores) {
		topK = len(scores)
	}
	out := make([]uint32, topK)
	for i := range topK {
		out[i] = scores[i].id
	}
	return out
}

func overlap(a, b []uint32) int {
	set := make(map[uint32]bool, len(a))
	for _, x := range a {
		set[x] = true
	}
	count := 0
	for _, x := range b {
		if set[x] {
			count++
		}
	}
	return count
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
