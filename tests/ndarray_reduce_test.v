import vnum { create_ndarray, get_view_linear_data, max, min, reduce, reduce_at_dim, sum }

fn add(args ...f64) f64 {
	return args[0] + args[1]
}

fn test_reduce_at_dim() {
	a := create_ndarray([]f64{len: 9, init: it + 1}, 3, 3)

	sum_row := reduce_at_dim(a, add, false, 0)
	assert get_view_linear_data(sum_row) == [12.0, 15, 18]
	assert sum_row.shape == [3]

	sum_col := reduce_at_dim(a, add, true, 1)
	assert get_view_linear_data(sum_col) == [6.0, 15, 24]
	assert sum_col.shape == [3, 1]
}

fn test_reduce() {
	a := create_ndarray([]f64{len: 9, init: it + 1}, 3, 3)
	sum_all_dim := reduce(a, add, true, 0, 1)
	assert get_view_linear_data(sum_all_dim) == [45.0]
	assert sum_all_dim.shape == [1, 1]
}

fn test_sum_3d_arr() {
	a := create_ndarray([]f64{len: 18, init: it + 1}, 2, 3, 3)
	sum_dim_0 := sum(a, false, 0)
	assert get_view_linear_data(sum_dim_0) == [11.0, 13, 15, 17, 19, 21, 23, 25, 27]
}

fn test_max() {
	a := create_ndarray([2.0, 1, 3, 0, 2, 2, 1, 5, 1], 3, 3)
	assert get_view_linear_data(max(a, false, 0)) == [2.0, 5, 3]
}

fn test_min() {
	a := create_ndarray([2.0, 1, 3, 0, 2, 2, 1, 5, 1], 3, 3)
	assert get_view_linear_data(min(a, false, 1)) == [1.0, 0, 1]
}
