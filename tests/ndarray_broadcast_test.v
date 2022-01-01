import vnum

fn test_broadcast_add_2x2_2() {
	mut a := vnum.create_ndarray([1.0, 2.0, 3.0, 4.0], 2, 2)
	mut b := vnum.create_ndarray([10.0, 100.0], 2)
	println(vnum.get_view_linear_data(a))
	println(vnum.get_view_linear_data(b))
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

fn test_broadcast_add_2x2x2_2x1x1() {
	mut a := vnum.create_ndarray([]f64{len: 8, init: it}, 2, 2, 2)
	mut b := vnum.create_ndarray([100.0, 1000.0], 1, 2, 1, 1)
	assert vnum.get_view_linear_data(a + b) == [100.0, 101.0, 102.0, 103.0, 1004.0, 1005.0, 1006.0, 1007.0]
}