package ivfpq

import "math"

// KMeans runs Lloyd's k-means on n points of dim dimensions (flattened row-major)
// into k clusters for iters iterations, returning the k×dim centroids and the final
// per-point assignment. Centroids are seeded from random points (via rng); an empty
// cluster is reseeded to a random point each iteration. A faithful port of the
// roaringrange Rust trainer's kmeans — same RNG, same reduction order — so on
// well-separated data it reproduces the Rust centroids bit-for-bit.
func KMeans(points []float32, n, dim, k, iters int, rng *Rng) (centroids []float32, assign []uint32) {
	centroids = make([]float32, k*dim)
	for c := range k {
		r := rng.NextIndex(n)
		copy(centroids[c*dim:(c+1)*dim], points[r*dim:(r+1)*dim])
	}

	assign = make([]uint32, n)
	sums := make([]float32, k*dim)
	counts := make([]uint32, k)
	for range iters {
		for i := range n {
			assign[i] = uint32(nearest(points[i*dim:(i+1)*dim], centroids, dim))
		}
		for i := range sums {
			sums[i] = 0
		}
		for i := range counts {
			counts[i] = 0
		}
		for i := range n {
			a := int(assign[i])
			counts[a]++
			p := points[i*dim : (i+1)*dim]
			s := sums[a*dim : (a+1)*dim]
			for d := range dim {
				s[d] += p[d]
			}
		}
		for c := range k {
			if counts[c] == 0 {
				r := rng.NextIndex(n)
				copy(centroids[c*dim:(c+1)*dim], points[r*dim:(r+1)*dim])
				continue
			}
			inv := float32(1.0) / float32(counts[c])
			cen := centroids[c*dim : (c+1)*dim]
			sm := sums[c*dim : (c+1)*dim]
			for d := range dim {
				cen[d] = sm[d] * inv
			}
		}
	}
	// Final assignment consistent with the returned centroids.
	for i := range n {
		assign[i] = uint32(nearest(points[i*dim:(i+1)*dim], centroids, dim))
	}
	return centroids, assign
}

// nearest returns the index of the centroid closest to p by squared L2 distance.
// Ties resolve to the lower index (strict <), matching the Rust trainer.
func nearest(p, centroids []float32, dim int) int {
	best := 0
	bestD := float32(math.Inf(1))
	for j := range len(centroids) / dim {
		c := centroids[j*dim : (j+1)*dim]
		var d float32
		for i := range dim {
			diff := p[i] - c[i]
			d += diff * diff
		}
		if d < bestD {
			bestD = d
			best = j
		}
	}
	return best
}

// normalize returns v scaled to unit L2 norm; a zero vector is returned unchanged.
// Mirrors the Rust trainer's normalize (used for the inner-product metric). Sqrt is
// computed in float64 then rounded to float32 — for sqrt this yields the
// correctly-rounded single-precision result, matching Rust's f32::sqrt.
func normalize(v []float32) []float32 {
	return appendNormalized(make([]float32, 0, len(v)), v)
}

// appendNormalized appends v scaled to unit L2 norm to dst (a zero vector is appended
// unchanged), allocating no temporary — the inner-product build hot path. Produces the
// same values as normalize.
func appendNormalized(dst, v []float32) []float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	norm := float32(math.Sqrt(float64(sum)))
	if norm == 0 {
		return append(dst, v...)
	}
	for _, x := range v {
		dst = append(dst, x/norm)
	}
	return dst
}
