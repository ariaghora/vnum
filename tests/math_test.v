import vnum

fn test_negative() {
	a := vnum.create_ndarray([1.0, 2.0, 3.0, 4.0], 2, 2)
	assert vnum.get_view_linear_data(vnum.negative(a)) == [-1.0, -2.0, -3.0, -4.0]
}

fn test_add_subtract_divide() {
	a := vnum.create_ndarray([1.0, 2, 3, 4], 2, 2)
	b := vnum.create_ndarray([10.0], 1)
	assert vnum.get_view_linear_data(a + b) == [11.0, 12, 13, 14]
	assert vnum.get_view_linear_data(a - b) == [-9.0, -8, -7, -6]
	assert vnum.get_view_linear_data(a / b) == [0.1, 0.2, 0.3, 0.4]
	assert vnum.get_view_linear_data(a * b) == [10.0, 20, 30, 40]
}
