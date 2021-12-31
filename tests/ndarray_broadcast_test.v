import vnum

fn test_broadcast_2x2_2() {
	mut a := vnum.create_ndarray([1.0, 2.0, 3.0, 4.0], 2, 2)
	mut b := vnum.create_ndarray([10.0, 100.0], 2)
	a, b = vnum.broadcast_ndarrays(a, b)
	assert vnum.get_view_linear_data(a + b) == [11.0, 102.0, 13.0, 104.0]
}
