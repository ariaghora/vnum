module vnum

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
