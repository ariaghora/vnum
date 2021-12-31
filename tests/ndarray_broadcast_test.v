import vnum

fn test_broadcast_add_2x2_2() {
	mut a := vnum.create_ndarray([1.0, 2.0, 3.0, 4.0], 2, 2)
	mut b := vnum.create_ndarray([10.0, 100.0], 2)
	assert vnum.get_view_linear_data(a + b) == [11.0, 102.0, 13.0, 104.0]
}

fn test_broadcast_add_2x2_2x1() {
	mut a := vnum.create_ndarray([1.0, 2.0, 3.0, 4.0], 2, 2)
	mut b := vnum.create_ndarray([10.0, 100.0], 2, 1)
	assert vnum.get_view_linear_data(a + b) == [11.0, 12.0, 103.0, 104.0]
}

fn test_broadcast_add_2x2x2_1() {
	mut a := vnum.create_ndarray([]f64{len: 8, init: it}, 2, 2, 2)
	mut b := vnum.create_ndarray([100.0], 1)
	assert vnum.get_view_linear_data(a + b) == []f64{len: 8, init: it + 100}
}