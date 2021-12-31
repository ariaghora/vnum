import vnum

fn test_1darray_get_val() {
	a := vnum.create_ndarray([1.0, 2.0, 3.0, 4.0], 4)
	assert a.get_val(2) == 3
}

fn test_2darray_get_val() {
	a := vnum.create_ndarray([1.0, 2.0, 3.0, 4.0], 2, 2)
	assert a.get_val(0, 1) == 2
	assert a.get_val(1, 1) == 4
}

fn test_3darray_get_val() {
	mut a := vnum.create_ndarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 1, 2)
	assert a.get_val(0, 0, 1) == 2.0
	assert a.get_val(2, 0, 1) == 6.0
}
