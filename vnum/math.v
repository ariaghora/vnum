module vnum

fn ufunc(func fn (args ...f64) f64, arrs ...NDArray) NDArray {
	mut result_data := []f64{}
	if arrs.len > 1 {
		// Check if all ndarrays in the argument list have the same
		// shape
		mut all_same := true
		for i, arr in arrs[1..] {
			all_same = all_same && (arrs[i].shape == arr.shape)
		}

		// If shapes are exactly same, apply regular ufunc. Otherwise,
		// try broadcasting.
		if all_same {
			for i in 0 .. arrs[0].get_size() {
				result_data << func(...arrs.map(it.get_by_offset(i)))
			}
			return create_ndarray(result_data, ...arrs[0].shape)
		} else {
			panic('Trying broadcasting but it is not implemented yet')
		}
	} else if arrs.len == 1 {
		// If there is only one ndarray, then treat this as an unary
		// function.
		for i in 0 .. arrs[0].get_size() {
			result_data << get_view_linear_data(arrs[0])[i]
		}
		return create_ndarray(result_data, ...arrs[0].shape)
	}
	panic('Cannot apply this ufunc due to different array shapes')
}

[inline]
fn add__(args ...f64) f64 {
	return args[0] + args[1]
}

pub fn add(arr1 NDArray, arr2 NDArray) NDArray {
	return ufunc(add__, arr1, arr2)
}

pub fn (a NDArray) + (b NDArray) NDArray {
	return add(a, b)
}
