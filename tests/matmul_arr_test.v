import vnum
import time { new_stopwatch }

// Test the matmul backend
fn test_matmul_arr() {
	mat_a := [
		1.0,
		2,
		3.0,
		4,
		5.0,
		6,
	]
	mat_b := [2.0, 6, 10, 1.0, 11, 10]

	mat_c := vnum.matmul_arr(mat_a, mat_b, 3, 3, 2)
	assert mat_c == [4.0, 28, 30, 10, 62, 70, 16, 96, 110]
}

fn test_matmul_big() {
	m := 1000
	n := 1000
	a := []f64{len: m * 1000, init: 0.0}
	b := []f64{len: 1000 * n, init: 0.0}

	sw := new_stopwatch()
	c := vnum.matmul_arr(a, b, m, n, 100)
	println(sw.elapsed().milliseconds())
	assert c == []f64{len: m * n, init: 0.0}
}

fn test_matmul_ndarr() {
	a := vnum.create_ndarray([1.0, 2, 3, 4], 2, 2)
	b := vnum.create_ndarray([1.0, 10], 2, 1)
	c := vnum.matmul(a, b)
	assert vnum.get_view_linear_data(c) == [21.0, 43.0]
}
