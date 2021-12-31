// *_after_{x>1}_slicings() means the test runs for sliced slices
import vnum

fn test_get_view_linear_data_simple_after_one_slicing() {
	a := vnum.create_ndarray([1.0, 2.0, 3.0, 4.0], 4)
	assert vnum.get_view_linear_data(a.slice([1, 3])) == [2.0, 4.0]
	assert vnum.get_view_linear_data(a.slice([3, 2, 0])) == [4.0, 3.0, 1.0]
}

fn test_get_view_linear_data_simple_after_two_slicings() {
	a := vnum.create_ndarray([1.0, 2.0, 3.0, 4.0], 4)
	mut slice := a.slice([1, 3])
	assert vnum.get_view_linear_data(slice) == [2.0, 4.0]
	slice = slice.slice([1])
	assert vnum.get_view_linear_data(slice) == [4.0]
}

fn test_get_view_linear_data_2d_after_one_slicing() {
	a := vnum.create_ndarray([1.0, 2.0, 3.0, 4.0], 2, 2)
	mut slice := a.slice(vnum.all, [1])
	assert vnum.get_view_linear_data(slice) == [2.0, 4.0]
	slice = a.slice([0])
	assert vnum.get_view_linear_data(slice) == [1.0, 2.0]
	slice = a.slice([1])
	assert vnum.get_view_linear_data(slice) == [3.0, 4.0]
}

fn test_get_view_linear_data_2d_after_two_slicings() {
	a := vnum.create_ndarray([1.0, 2.0, 3.0, 4.0], 2, 2)
	mut slice := a.slice(vnum.all, [1])
	assert vnum.get_view_linear_data(slice) == [2.0, 4.0]
	slice = slice.slice([0])
	assert vnum.get_view_linear_data(slice) == [2.0]
}

fn test_get_view_linear_data_3d_after_one_slicing() {
	a := vnum.create_ndarray([]f64{len: 12, init: it}, 3, 2, 2)
	mut slice := a.slice([0, 2], [0])
	assert vnum.get_view_linear_data(slice) == [0.0, 1.0, 8.0, 9.0]
	slice = a.slice([1])
	assert vnum.get_view_linear_data(slice) == [4.0, 5.0, 6.0, 7.0]
}

fn test_get_view_linear_data_3d_after_two_slicings() {
	a := vnum.create_ndarray([]f64{len: 12, init: it}, 3, 2, 2)
	mut slice := a.slice([0, 2], [0])
	assert vnum.get_view_linear_data(slice) == [0.0, 1.0, 8.0, 9.0]
	slice = slice.slice([0])
	assert vnum.get_view_linear_data(slice) == [0.0, 1.0]
}
