import vnum { create_ndarray, get_view_linear_data, permute_dimension, transpose }

fn test_permute_dimension() {
	a := create_ndarray([1.0, 2, 3, 4, 5, 6], 3, 2)
	b := permute_dimension(a, 1, 0)

	assert get_view_linear_data(b) == [1.0, 3, 5, 2, 4, 6]
	assert get_view_linear_data(b) == get_view_linear_data(transpose(a))
}
