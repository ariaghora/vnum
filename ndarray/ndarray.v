module ndarray
import strings

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

pub fn (arr NDArray) get_val(index ...int) f64 {
	check_arr_ndim_and_index_len_equals(arr, index)
	return arr.data[index_to_offset(index, arr.strides)]
}

pub fn (arr NDArray) get_by_offset(offset int) f64 {
	index := offset_to_index(arr, offset)
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
	return result
}

// Squeeze length-1 dimensions
// TODO: use after we get proper implementation of indexing. Squuezing
// at this moment will cause unexpected behavior.
pub fn (mut arr NDArray) squeeze_in_place() {
	mut new_indices := [][]int{}
	mut new_shape := []int{}
	mut new_strides := []int{}
	for i, shp in arr.shape {
		if shp > 1 {
			new_indices << arr.indices[i]
			new_shape << shp
			new_strides << arr.strides[i]
		}
	}
	arr.indices = new_indices
	arr.shape = new_shape
	arr.strides = new_strides
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

fn print_ndarray_util(ndarray NDArray, ndim int) {
	for i in 0 .. ndarray.shape[0] {
		if ndim > 1 {
			print_ndarray_util(ndarray.slice([i]), ndim - 1)
			print(strings.repeat_string("\n", ndim - 1))
		} else {
			println(get_view_linear_data(ndarray))
		}
	}
}

pub fn print_ndarray(ndarray NDArray) {
	print_ndarray_util(ndarray, ndarray.shape.len - 1)
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
