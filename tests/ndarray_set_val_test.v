import ndarray

fn test_1darray_set_val() {
	mut a := ndarray.create_ndarray([1.0, 2.0, 3.0, 4.0], 4)
	a.set_val(42, 1)
	assert a.data == [1.0, 42.0, 3.0, 4.0]
}

fn test_2darray_set_val() {
	mut a := ndarray.create_ndarray([1.0, 2.0, 3.0, 4.0], 2, 2)
	a.set_val(42, 1, 1)
	assert a.data == [1.0, 2.0, 3.0, 42.0]
	a.set_val(42, 0, 0)
	assert a.data == [42.0, 2.0, 3.0, 42.0]
}

fn test_3darray_set_val() {
	mut a := ndarray.create_ndarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 1, 2)
	a.set_val(42, 2, 0, 0)
	assert a.data == [1.0, 2.0, 3.0, 4.0, 42.0, 6.0]
}
