module vnum

import math

/*---------------------------------------------------------------------------
 * Const and aliases
 *--------------------------------------------------------------------------*/
const all = []int{len: 0}

/*---------------------------------------------------------------------------
 * The main NDArray struct
 *--------------------------------------------------------------------------*/
pub struct NDArray {
pub mut:
	indices [][]int
	data    []f64
	shape   []int
	strides []int
}

pub fn (arr NDArray) get_by_offset(offset int) f64 {
	index := offset_to_index(arr, offset)
	return arr.data[index_to_offset(index, arr.strides)]
}

pub fn (arr NDArray) get_size() int {
	mut res := 1
	for shp in arr.shape {
		res *= shp
	}
	return res
}

pub fn (arr NDArray) get_val(index ...int) f64 {
	check_arr_ndim_and_index_len_equals(arr, index)
	return arr.data[index_to_offset(index, arr.strides)]
}

// Generate initial linear index from 0..arr.shape[i] for i in 0..arr.shape.len
fn (mut arr NDArray) init_indices() {
	mut indices := [][]int{len: arr.shape.len}
	for i in 0 .. arr.shape.len {
		indices[i] = []int{len: arr.shape[i], init: it}
	}
	arr.indices = indices
}

pub fn (mut arr NDArray) set_val(val f64, index ...int) {
	check_arr_ndim_and_index_len_equals(arr, index)
	arr.data[index_to_offset(index, arr.strides)] = val
}

// Slice NDArray using `indices`
pub fn (arr NDArray) slice(indices ...[]int) NDArray {
	if indices.len > arr.shape.len {
		panic('the NDArray is $arr.shape.len-dimensional, but $indices.len were indexed')
	}

	// Initially, use to arr's metadata and point data to arr's data
	mut result := NDArray{
		data: arr.data
	}
	for index in arr.indices {
		result.indices << index
	}
	for shp in arr.shape {
		result.shape << shp
	}
	for stride in arr.strides {
		result.strides << stride
	}

	// Alter result's indices and shape
	for i in 0 .. indices.len {
		// only alter when indices[i].len > 0. Otherwise,
		// we use original arr's indices[i] and shape[i]
		if indices[i].len > 0 {
			mut new_index := []int{}
			for j in indices[i] {
				new_index << result.indices[i][j]
			}
			result.indices[i] = new_index
			result.shape[i] = indices[i].len
		}
	}
	return result.squeeze()
}

// Squeeze length-1 dimensions
// TODO: use after we get proper implementation of indexing. Squuezing
// at this moment will cause unexpected behavior.
pub fn (mut arr NDArray) squeeze() NDArray {
	mut new_shape := []int{}
	for shp in arr.shape {
		if shp > 1 {
			new_shape << shp
		}
	}
	if new_shape.len == 0 {
		new_shape = [1]
	}
	return create_ndarray(get_view_linear_data(arr), ...new_shape)
}

/*---------------------------------------------------------------------------
 * Utility functions
 *--------------------------------------------------------------------------*/
fn check_arr_ndim_and_index_len_equals(arr NDArray, index []int) {
	if arr.shape.len != index.len {
		panic('Array dimension ($arr.shape.len) does not match the number of index ($index.len)')
	}
}

// Basic ndarray creation function.
// It accepts an array of data and an array containing shape information, to
// which the ndarray will be reshaped.
pub fn create_ndarray(data []f64, shape ...int) NDArray {
	mut result := NDArray{
		data: data
		shape: shape
		strides: shape_to_strides(shape)
	}
	result.init_indices()
	return result
}

// Returns the data in a linear manner, regardless contiguousness
pub fn get_view_linear_data(arr NDArray) []f64 {
	mut size := 1
	for s in arr.shape {
		size *= s
	}

	mut result := []f64{len: size}
	for i in 0 .. size {
		result[i] = arr.get_by_offset(i)
	}
	return result
}

// ....
fn index_to_offset(index []int, strides []int) int {
	mut result := 0
	for i in 0 .. index.len {
		result += strides[i] * index[i]
	}
	return result
}

// Given an offset, recover it to multidimensional index according to `arr`'s'
// strides, shapes, and indices. This is the inverse of index to offset.
pub fn offset_to_index(arr NDArray, offset int) []int {
	mut index := []int{len: arr.shape.len}

	mut cnt := arr.shape.len - 1
	mut offset_ := offset
	for i := arr.shape.len - 1; i >= 0; i -= 1 {
		index[cnt] = arr.indices[cnt][offset_ % arr.shape[i]]
		offset_ /= arr.shape[i]
		cnt -= 1
	}
	return index
}

// Given a multidimensional index, calculate a proper array of strides.
// This will be invoked whenever an ndarray is initialized or an ndarray's shape
// is reset.
fn shape_to_strides(shape []int) []int {
	mut strides := []int{len: shape.len}
	for k in 0 .. strides.len {
		mut prod := 1
		for j in k + 1 .. strides.len {
			prod = prod * shape[j]
		}
		strides[k] = prod
	}
	return strides
}

pub fn broadcast_ndarrays(arr1 NDArray, arr2 NDArray) (NDArray, NDArray) {
	new_ndims := int(math.max(arr1.shape.len, arr2.shape.len))

	mut new_shape_1 := arr1.shape.reverse()
	mut new_shape_2 := arr2.shape.reverse()
	mut new_strides_1 := arr1.strides.reverse()
	mut new_strides_2 := arr2.strides.reverse()

	// Append zeros until having length of `new_ndims`
	new_shape_1.insert(new_shape_1.len, []int{len: new_ndims - new_shape_1.len, init: 0})
	new_shape_2.insert(new_shape_2.len, []int{len: new_ndims - new_shape_2.len, init: 0})
	new_strides_1.insert(new_strides_1.len, []int{len: new_ndims - new_strides_1.len, init: 0})
	new_strides_2.insert(new_strides_2.len, []int{len: new_ndims - new_strides_2.len, init: 0})

	// Broadcast rule checking...
	for i in 0 .. new_ndims {
		if new_shape_1[i] != new_shape_2[i] {
			if new_shape_1[i] < new_shape_2[i] && new_shape_1[i] <= 1 {
				new_strides_1[i] = 0
				new_shape_1[i] = new_shape_2[i]
			} else if new_shape_2[i] < new_shape_1[i] && new_shape_2[i] <= 1 {
				new_strides_2[i] = 0
				new_shape_2[i] = new_shape_1[i]
			}
		}
	}

	mut new_arr_1 := NDArray{
		data: arr1.data
		shape: new_shape_1.reverse()
		strides: new_strides_1.reverse()
		indices: arr1.indices
	}
	new_arr_1.init_indices()

	mut new_arr_2 := NDArray{
		data: arr2.data
		shape: new_shape_2.reverse()
		strides: new_strides_2.reverse()
		indices: arr2.indices
	}
	new_arr_2.init_indices()
	return new_arr_1, new_arr_2
}
