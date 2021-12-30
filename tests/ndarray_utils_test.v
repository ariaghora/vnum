import ndarray

fn test_index_to_offset_1darray() {
	a := ndarray.create_ndarray([1.0, 2.0, 3.0, 4.0], 4)
	assert ndarray.offset_to_index(a, 2) == [2]
}

fn test_index_to_offset_2darray() {
	a := ndarray.create_ndarray([1.0, 2.0, 3.0, 4.0], 2, 2)
	assert ndarray.offset_to_index(a, 2) == [1, 0]
	assert ndarray.offset_to_index(a, 0) == [0, 0]
	assert ndarray.offset_to_index(a, 3) == [1, 1]
}

fn test_index_to_offset_3darray() {
	/*
	[[[0, 1],
	   [2, 3]],

	  [[4, 5]
	   [6, 7]],

	  [[8, 9]
	   [10, 11]]]
	*/
	a := ndarray.create_ndarray([]f64{len: 12, init: it}, 3, 2, 2)
	assert ndarray.offset_to_index(a, 4) == [1, 0, 0]
	assert ndarray.offset_to_index(a, 9) == [2, 0, 1]
}

fn test_get_view_linear_data_simple() {
	a := ndarray.create_ndarray([1.0, 2.0, 3.0, 4.0], 2, 2)
	assert ndarray.get_view_linear_data(a) == [1.0, 2.0, 3.0, 4.0]
}
