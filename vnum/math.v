module vnum

fn ufunc(func fn (args ...f64) f64, arrs ...NDArray) NDArray {
	if arrs.len == 2 {
		// Check if all ndarrays in the argument list have the same
		// shape
		mut all_same := true
		for i, arr in arrs[1..] {
			all_same = all_same && (arrs[i].shape == arr.shape)
		}

		// If shapes are exactly same, apply regular ufunc. Otherwise,
		// try broadcasting two arrays each other.
		if all_same {
			mut result_data := []f64{len: arrs[0].get_size()}
			data_1 := get_view_linear_data(arrs[0])
			data_2 := get_view_linear_data(arrs[1])
			for i in 0 .. arrs[0].get_size() {
				result_data[i] = func(data_1[i], data_2[i])
			}
			return create_ndarray(result_data, ...arrs[0].shape)
		} else {
			arr_1, arr_2 := broadcast_ndarrays(arrs[0], arrs[1])
			return ufunc(func, arr_1, arr_2)
		}
	} else if arrs.len == 1 {
		// If there is only one ndarray, then treat this as an unary
		// function.
		mut result_data := []f64{len: arrs[0].get_size()}
		data := get_view_linear_data(arrs[0])
		for i in 0 .. arrs[0].get_size() {
			result_data[i] = func(data[i])
		}
		return create_ndarray(result_data, ...arrs[0].shape)
	} else if arrs.len > 2 {
		panic('Cannot apply ufunc on $arrs.len arrays yet')
	}
	panic('Cannot apply this ufunc due to different array shapes')
}

/*---------------------------------------------------------------------------
 * Wrapper for basic (float) arithmetic functions
 *--------------------------------------------------------------------------*/

[inline]
fn negative__(args ...f64) f64 {
	return -args[0]
}

[inline]
fn add__(args ...f64) f64 {
	return args[0] + args[1]
}

[inline]
fn subtract__(args ...f64) f64 {
	return args[0] - args[1]
}

[inline]
fn multiply__(args ...f64) f64 {
	return args[0] * args[1]
}

[inline]
fn divide__(args ...f64) f64 {
	return args[0] / args[1]
}

/*---------------------------------------------------------------------------
 * Unary and binary functions
 *--------------------------------------------------------------------------*/

// Unary functions

pub fn negative(arr NDArray) NDArray {
	return ufunc(negative__, arr)
}

// Binary functions
pub fn add(arr1 NDArray, arr2 NDArray) NDArray {
	return ufunc(add__, arr1, arr2)
}

pub fn subtract(arr1 NDArray, arr2 NDArray) NDArray {
	return ufunc(subtract__, arr1, arr2)
}

pub fn multiply(arr1 NDArray, arr2 NDArray) NDArray {
	return ufunc(multiply__, arr1, arr2)
}

pub fn divide(arr1 NDArray, arr2 NDArray) NDArray {
	return ufunc(divide__, arr1, arr2)
}
