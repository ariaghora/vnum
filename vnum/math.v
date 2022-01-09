module vnum

import math

/*---------------------------------------------------------------------------
 * Wrapper for basic (float) arithmetic functions. These functions are
 * applied to NDArrays element-wise via universal function. The corresponding
 * Universal function should be defined later. Please write the functions in
 * ascending order.
 *--------------------------------------------------------------------------*/

[inline]
fn add__(args ...f64) f64 {
	return args[0] + args[1]
}

[inline]
fn divide__(args ...f64) f64 {
	return args[0] / args[1]
}

[inline]
fn maximum__(args ...f64) f64 {
	return math.max(args[0], args[1])
}

[inline]
fn minimum__(args ...f64) f64 {
	return math.min(args[0], args[1])
}

[inline]
fn multiply__(args ...f64) f64 {
	return args[0] * args[1]
}

[inline]
fn negative__(args ...f64) f64 {
	return -args[0]
}

[inline]
fn subtract__(args ...f64) f64 {
	return args[0] - args[1]
}

[inline]
fn pow__(args ...f64) f64 {
	return math.pow(args[0], args[1])
}

/*---------------------------------------------------------------------------
 * Unary and binary functions
 *--------------------------------------------------------------------------*/

pub fn add(arr1 NDArray, arr2 NDArray) NDArray {
	return ufunc(add__, arr1, arr2)
}

pub fn divide(arr1 NDArray, arr2 NDArray) NDArray {
	return ufunc(divide__, arr1, arr2)
}

pub fn matmul(arr1 NDArray, arr2 NDArray) NDArray {
	data1 := get_view_linear_data(arr1)
	data2 := get_view_linear_data(arr2)
	data_result := matmul_arr(data1, data2, arr1.shape[0], arr2.shape[1], arr1.shape[0])
	return create_ndarray(data_result, arr1.shape[0], arr2.shape[1])
}

pub fn max(arr NDArray, keep_dims bool, dims ...int) NDArray {
	return reduce(arr, maximum__, keep_dims, ...dims)
}

pub fn min(arr NDArray, keep_dims bool, dims ...int) NDArray {
	return reduce(arr, minimum__, keep_dims, ...dims)
}

pub fn multiply(arr1 NDArray, arr2 NDArray) NDArray {
	return ufunc(multiply__, arr1, arr2)
}

pub fn negative(arr NDArray) NDArray {
	return ufunc(negative__, arr)
}

pub fn pow(arr NDArray, exponent f64) NDArray {
	return ufunc(pow__, arr, create_ndarray([exponent], 1))
}

pub fn subtract(arr1 NDArray, arr2 NDArray) NDArray {
	return ufunc(subtract__, arr1, arr2)
}

pub fn sum(arr NDArray, keep_dims bool, dims ...int) NDArray {
	return reduce(arr, add__, keep_dims, ...dims)
}
