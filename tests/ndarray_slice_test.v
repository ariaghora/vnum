import ndarray

fn test_get_view_linear_data_simple_after_one_slicing() {
	a := ndarray.create_ndarray([1.0, 2.0, 3.0, 4.0], 4)
	assert ndarray.get_view_linear_data(a.slice([1, 3])) == [2.0, 4.0]
	assert ndarray.get_view_linear_data(a.slice([3, 2, 0])) == [4.0, 3.0, 1.0]
}

fn test_get_view_linear_data_simple_after_two_slicings() {
	a := ndarray.create_ndarray([1.0, 2.0, 3.0, 4.0], 4)
	mut slice := a.slice([1, 3])
	assert ndarray.get_view_linear_data(slice) == [2.0, 4.0]
	slice = slice.slice([1])
	assert ndarray.get_view_linear_data(slice) == [4.0]
}

fn test_get_view_linear_data_2d_after_one_slicing() {
	a := ndarray.create_ndarray([1.0, 2.0, 3.0, 4.0], 2, 2)
	slice := a.slice(ndarray.all, [1])
	assert ndarray.get_view_linear_data(slice) == [2.0, 4.0]
}

fn test_get_view_linear_data_2d_after_two_slicings() {
	a := ndarray.create_ndarray([1.0, 2.0, 3.0, 4.0], 2, 2)
	mut slice := a.slice(ndarray.all, [1])
	assert ndarray.get_view_linear_data(slice) == [2.0, 4.0]
	slice = slice.slice([0])
	assert ndarray.get_view_linear_data(slice) == [2.0]
}

fn test_get_view_linear_data_3d_after_one_slicing() {
	a := ndarray.create_ndarray([]f64{len: 12, init: it}, 3, 2, 2)
	slice := a.slice([0, 2], [0])
	assert ndarray.get_view_linear_data(slice) == [0.0, 1.0, 8.0, 9.0]
}

fn test_get_view_linear_data_3d_after_two_slicings() {
	a := ndarray.create_ndarray([]f64{len: 12, init: it}, 3, 2, 2)
	mut slice := a.slice([0, 2], [0])
	assert ndarray.get_view_linear_data(slice) == [0.0, 1.0, 8.0, 9.0]
	slice = slice.slice([0])
	assert ndarray.get_view_linear_data(slice) == [0.0, 1.0]
}
