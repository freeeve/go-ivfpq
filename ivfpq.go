package ivfpq

import (
	"errors"
	"fmt"
)

// Metric is the similarity the index trains and scores for. The values match the
// roaringrange on-disk metric byte.
type Metric uint8

const (
	// InnerProduct scores inner product on L2-normalized vectors (cosine). Inputs are
	// normalized before training.
	InnerProduct Metric = 0
	// L2 scores raw L2 distance; inputs are used as-is.
	L2 Metric = 1
)

// Vector is one input: a document ID and its dense embedding.
type Vector struct {
	ID  uint32
	Vec []float32
}

// Params configures Train. Mirrors the Rust IvfpqParams.
type Params struct {
	// Dim is the vector dimensionality; every input must have this length.
	Dim int
	// Nlist is the number of coarse (IVF) clusters; a rule of thumb is ~4·√N.
	Nlist int
	// M is the number of PQ subquantizers; must divide Dim.
	M int
	// Nbits is bits per PQ code (1..=8); each subspace gets 2^Nbits codebook entries.
	Nbits uint8
	// Metric is the similarity to train for.
	Metric Metric
	// KmeansIters is Lloyd's-iteration count for every k-means run (coarse and PQ).
	KmeansIters int
	// Seed seeds the deterministic PRNG, so a build is reproducible.
	Seed uint64
}

// DefaultParams returns Params with the same defaults as the Rust IvfpqParams::new:
// 8-bit PQ codes, inner-product metric, 25 k-means iterations, a fixed seed.
func DefaultParams(dim, nlist, m int) Params {
	return Params{
		Dim:         dim,
		Nlist:       nlist,
		M:           m,
		Nbits:       8,
		Metric:      InnerProduct,
		KmeansIters: 25,
		Seed:        0x5251564952525649,
	}
}

// Model is a trained IVFPQ index as plain Go arrays, ready for an RRVI serializer (in
// roaringrange) or in-memory ADC search. Field meanings mirror the Rust Ivfpq.
type Model struct {
	Dim    int
	Nlist  int
	M      int
	Nbits  uint8
	Metric Metric
	N      uint64
	// OPQ is an optional dim×dim row-major rotation (nil unless imported via FromParts).
	OPQ []float32
	// Centroids are the coarse centroids, Nlist×Dim row-major.
	Centroids []float32
	// Codebooks are the PQ codebooks, M×(2^Nbits)×(Dim/M) row-major.
	Codebooks []float32
	// ListIDs[c] holds the doc IDs assigned to cluster c, in input order.
	ListIDs [][]uint32
	// ListCodes[c] holds the PQ codes for cluster c (len(ListIDs[c])×M bytes, row-major).
	ListCodes [][]byte
}

// Parts is an already-trained model (e.g. exported from FAISS) for FromParts. All
// arrays are row-major in the (optionally OPQ-rotated) space the centroids live in.
type Parts struct {
	Dim         int
	Nlist       int
	M           int
	Nbits       uint8
	Metric      Metric
	Centroids   []float32
	Codebooks   []float32
	OPQ         []float32 // optional, dim×dim
	IDs         []uint32
	Assignments []uint32 // per-vector cluster (< Nlist)
	Codes       []byte   // per-vector PQ codes, len(IDs)×M
}

// ErrEmpty is returned by Train when no vectors are supplied.
var ErrEmpty = errors.New("ivfpq: no input vectors")

// ErrDimMismatch is returned when a vector's length != Params.Dim.
var ErrDimMismatch = errors.New("ivfpq: a vector's length != params.Dim")

func badParams(msg string) error { return fmt.Errorf("ivfpq: bad params: %s", msg) }

// Train trains an IVFPQ index from vectors per p — a faithful port of the Rust
// build_ivfpq. Vectors are L2-normalized for the inner-product metric; coarse k-means
// partitions them into Nlist clusters; per-subspace k-means over the residuals
// (vector − assigned centroid) builds the M PQ codebooks and, in the same pass, the
// per-vector codes. A single PRNG threads through the coarse and all PQ k-means runs,
// matching the Rust draw order.
func Train(vectors []Vector, p Params) (*Model, error) {
	if len(vectors) == 0 {
		return nil, ErrEmpty
	}
	dim, m := p.Dim, p.M
	if dim == 0 || m == 0 || dim%m != 0 {
		return nil, badParams("dim/m: need dim>0, m>0, m|dim")
	}
	if p.Nbits == 0 || p.Nbits > 8 {
		return nil, badParams("nbits must be in 1..=8")
	}
	if p.Nlist == 0 {
		return nil, badParams("nlist must be >= 1")
	}
	for _, v := range vectors {
		if len(v.Vec) != dim {
			return nil, ErrDimMismatch
		}
	}

	n := len(vectors)
	ksub := 1 << p.Nbits
	dsub := dim / m
	iters := p.KmeansIters
	if iters < 1 {
		iters = 1
	}
	nlist := p.Nlist
	if nlist > n {
		nlist = n // never more clusters than vectors
	}

	pts := make([]float32, 0, n*dim)
	ids := make([]uint32, 0, n)
	for _, v := range vectors {
		ids = append(ids, v.ID)
		if p.Metric == InnerProduct {
			pts = append(pts, normalize(v.Vec)...)
		} else {
			pts = append(pts, v.Vec...)
		}
	}

	rng := NewRng(p.Seed)

	// Coarse quantizer.
	centroids, assign := KMeans(pts, n, dim, nlist, iters, rng)

	// Residuals: vector − its assigned centroid.
	res := make([]float32, n*dim)
	for i := range n {
		a := int(assign[i])
		pv := pts[i*dim : (i+1)*dim]
		c := centroids[a*dim : (a+1)*dim]
		r := res[i*dim : (i+1)*dim]
		for d := range dim {
			r[d] = pv[d] - c[d]
		}
	}

	// PQ codebooks + per-vector codes, one subspace at a time. The subspace k-means
	// assignment is the code byte for that subspace.
	codebooks := make([]float32, m*ksub*dsub)
	codes := make([]byte, n*m)
	subpts := make([]float32, n*dsub)
	for s := range m {
		for i := range n {
			copy(subpts[i*dsub:(i+1)*dsub], res[i*dim+s*dsub:i*dim+(s+1)*dsub])
		}
		cb, codeS := KMeans(subpts, n, dsub, ksub, iters, rng)
		copy(codebooks[s*ksub*dsub:(s+1)*ksub*dsub], cb)
		for i := range n {
			codes[i*m+s] = byte(codeS[i])
		}
	}

	// Scatter every vector into its cluster's list.
	listIDs := make([][]uint32, nlist)
	listCodes := make([][]byte, nlist)
	for i := range n {
		a := int(assign[i])
		listIDs[a] = append(listIDs[a], ids[i])
		listCodes[a] = append(listCodes[a], codes[i*m:(i+1)*m]...)
	}

	return &Model{
		Dim: dim, Nlist: nlist, M: m, Nbits: p.Nbits, Metric: p.Metric, N: uint64(n),
		Centroids: centroids, Codebooks: codebooks, ListIDs: listIDs, ListCodes: listCodes,
	}, nil
}

// FromParts assembles a Model from already-trained parts (e.g. exported from FAISS),
// scattering each vector's code into its assigned cluster's list — no training. A
// faithful port of build_ivfpq_from_parts: validates every array length against
// dim/nlist/m/nbits and that assignments and codes are in range.
func FromParts(parts Parts) (*Model, error) {
	dim, m := parts.Dim, parts.M
	if dim == 0 || m == 0 || dim%m != 0 {
		return nil, badParams("dim/m: need dim>0, m>0, m|dim")
	}
	if parts.Nbits == 0 || parts.Nbits > 8 {
		return nil, badParams("nbits must be in 1..=8")
	}
	if parts.Nlist == 0 {
		return nil, badParams("nlist must be >= 1")
	}
	ksub := 1 << parts.Nbits
	dsub := dim / m
	if len(parts.Centroids) != parts.Nlist*dim {
		return nil, badParams("centroids length != nlist*dim")
	}
	if len(parts.Codebooks) != m*ksub*dsub {
		return nil, badParams("codebooks length != m*ksub*dsub")
	}
	if parts.OPQ != nil && len(parts.OPQ) != dim*dim {
		return nil, badParams("opq length != dim*dim")
	}
	n := len(parts.IDs)
	if len(parts.Assignments) != n {
		return nil, badParams("assignments length != ids length")
	}
	if len(parts.Codes) != n*m {
		return nil, badParams("codes length != ids length * m")
	}
	for _, a := range parts.Assignments {
		if int(a) >= parts.Nlist {
			return nil, badParams("an assignment >= nlist")
		}
	}
	if parts.Nbits < 8 {
		for _, c := range parts.Codes {
			if int(c) >= ksub {
				return nil, badParams("a code >= ksub (1<<nbits)")
			}
		}
	}

	listIDs := make([][]uint32, parts.Nlist)
	listCodes := make([][]byte, parts.Nlist)
	for i := range n {
		a := int(parts.Assignments[i])
		listIDs[a] = append(listIDs[a], parts.IDs[i])
		listCodes[a] = append(listCodes[a], parts.Codes[i*m:(i+1)*m]...)
	}

	return &Model{
		Dim: dim, Nlist: parts.Nlist, M: m, Nbits: parts.Nbits, Metric: parts.Metric, N: uint64(n),
		OPQ: parts.OPQ, Centroids: parts.Centroids, Codebooks: parts.Codebooks,
		ListIDs: listIDs, ListCodes: listCodes,
	}, nil
}

// Reconstruct returns the approximate vector for the j-th entry of cluster c —
// its coarse centroid plus the PQ-decoded residual. This is the vector an ADC scan
// scores a query against; it is also the basis of the recall self-check.
func (mdl *Model) Reconstruct(c, j int) []float32 {
	dsub := mdl.Dim / mdl.M
	ksub := 1 << mdl.Nbits
	out := make([]float32, mdl.Dim)
	copy(out, mdl.Centroids[c*mdl.Dim:(c+1)*mdl.Dim])
	code := mdl.ListCodes[c][j*mdl.M : (j+1)*mdl.M]
	for s := range mdl.M {
		base := s*ksub*dsub + int(code[s])*dsub
		cb := mdl.Codebooks[base : base+dsub]
		for d := range dsub {
			out[s*dsub+d] += cb[d]
		}
	}
	return out
}
