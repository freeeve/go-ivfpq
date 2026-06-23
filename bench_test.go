package ivfpq

import "testing"

// benchVectors is a representative training set: 2000 vectors in 32 dims over 16 clusters.
func benchVectors() []Vector { return clusteredVectors(2000, 32, 16, 0xBEEF) }

// BenchmarkTrain exercises the full IVFPQ build: normalize → coarse k-means → residuals →
// 8 PQ subspace k-means → scatter.
func BenchmarkTrain(b *testing.B) {
	vectors := benchVectors()
	p := DefaultParams(32, 64, 8) // InnerProduct, nbits 8, m 8
	p.KmeansIters = 10
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := Train(vectors, p); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkKMeans isolates the coarse k-means hot path.
func BenchmarkKMeans(b *testing.B) {
	vectors := benchVectors()
	pts := make([]float32, 0, len(vectors)*32)
	for _, v := range vectors {
		pts = append(pts, v.Vec...)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		KMeans(pts, len(vectors), 32, 64, 10, NewRng(1))
	}
}
