module vnum

pub fn matmul_arr(mat_a []f64, mat_b []f64, m int, n int, p int) []f64 {
	mut result := []f64{len: m * n, init: 0}
	for i := 0; i < m; i += 1 {
		for j := 0; j < n; j += 1 {
			mut sum := 0.0
			for k := 0; k < p; k += 1 {
				sum += mat_a[i * p + k] * mat_b[k * n + j]
			}
			result[i * n + j] = sum
		}
	}
	return result
}
