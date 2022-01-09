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
	indices    [][]int
	data       []f64
	shape      []int
	strides    []int
	contiguous bool = true
}

pub fn (arr NDArray) get_by_offset(offset int) f64 {
	if arr.contiguous {
		return arr.data[offset]
	}
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

// Create a new ndarray as a contiguous version of `arr`
pub fn (mut arr NDArray) contiguous() NDArray {
	if arr.contiguous {
		return arr
	}
	return create_ndarray(get_view_linear_data(arr), ...arr.shape)
}

pub fn (arr NDArray) str() string {
	shape_str := arr.shape.map(fn (i int) string {return i.str()})
	mut res := "(${shape_str.join("×")} ndarray)\n"
	res += ndarray_to_string(arr)

	return res
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

	// Initially, use arr's metadata and point data to arr's data
	mut result := NDArray{
		data: arr.data
	}
	result.indices = arr.indices.clone()
	result.shape = arr.shape.clone()
	result.strides = arr.strides.clone()

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
	result.contiguous = false
	return result.squeeze()
}

// Squeeze length-1 dimensions
// TODO: try to squeeze without having to create new contiguous ndarray
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

// This function takes sub-array from ndarray's actual data.
// It meants to be invoked in several threads.
fn arr_fetch_worker(arr NDArray, lower int, upper int) []f64 {
	// If array is contiguous then gettin sub-array is very simple
	if arr.contiguous {
		return arr.data[lower..upper]
	}

	mut result := []f64{len: upper - lower}
	for i in lower .. upper {
		result[i - lower] = arr.get_by_offset(i)
	}
	return result
}

// Returns a list of pairs containing lower and upper bound of a sub-array
// in arr's linear data
fn get_array_chunk_bounds(arr NDArray, chunk_count int) [][]int {
	size := arr.get_size()
	chunk_size := size / chunk_count

	// Determine the lower and upper bounds for each chunk
	mut starts := []int{len: chunk_count, init: it * chunk_size}
	starts << arr.get_size()
	mut idx_bounds := [][]int{}
	for i in 0 .. starts.len - 1 {
		idx_bounds << [starts[i], starts[i + 1]]
	}
	return idx_bounds
}

// Returns the data in a linear manner, regardless contiguousness
pub fn get_view_linear_data(arr NDArray) []f64 {
	// We are going to split the task of fetching array data into several tasks.
	// First, determine the number chunks, in which we want to store the fetched data.
	chunk_count := 4
	idx_bounds := get_array_chunk_bounds(arr, chunk_count)

	// Concurrently execute the data fetching tasks
	mut handlers := []thread []f64{}
	for bound in idx_bounds {
		handlers << go arr_fetch_worker(arr, bound[0], bound[1])
	}

	// Obtain the actual data from the thread handler in each chunk, and append
	// them into result array.
	mut result := []f64{}
	chunk_results := handlers.map(fn (x thread []f64) []f64 {
		return x.wait()
	})

	for i, chunk in chunk_results {
		result.insert(idx_bounds[i][0], chunk)
	}

	return result
}

// Convert multidimensional index into offset pointing to the actual data
[inline]
fn index_to_offset(index []int, strides []int) int {
	mut result := 0
	for i in 0 .. index.len {
		result += strides[i] * index[i]
	}
	return result
}

// Given an offset, recover it to multidimensional index according to `arr`'s'
// strides, shapes, and indices. This is the inverse of index to offset.
[inline]
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

// Permute dimension of an NDArray
pub fn permute_dimension(arr NDArray, dims ...int) NDArray {
	if arr.shape.len != dims.len {
		panic('Number of dimension does not match the permuted dims')
	}

	// Check if the permuted dims are unique
	mut unique := []int{}
	for i in dims {
		if i !in unique {
			unique << i
		} else {
			panic('The permuted dimensions are not unique')
		}
	}

	shape := []int{len: dims.len, init: arr.shape[dims[it]]}
	strides := []int{len: dims.len, init: arr.strides[dims[it]]}
	mut indices := [][]int{}
	for i in dims {
		indices << arr.indices[i]
	}

	return NDArray{
		data: arr.data
		shape: shape
		strides: strides
		indices: indices
		contiguous: false
	}
}

// Pretty-printing ndarray.
pub fn ndarray_to_string(arr NDArray) string {
	mut out := ''
	for i in 0 .. arr.shape[0] {
		if arr.shape.len == 1 {
			out += arr.slice([i]).data[0].str()
			if i < arr.shape[0] - 1 {
				out += ', '
			}
		} else {
			out += ndarray_to_string(arr.slice([i]))
		}

		if (arr.shape.len > 1) && (i < arr.shape[0] - 1) {
			out += '\n'.repeat(arr.shape.len - 1)
		}
	}
	return out
}

// Transpose is a special case of permute_dimension where there order of dims
// is reversed
pub fn transpose(arr NDArray) NDArray {
	reversed_dims := []int{len: arr.shape.len, init: it}.reverse()
	return permute_dimension(arr, ...reversed_dims)
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

// Given two ndarrays with different shape, try to broadcast them into two
// arrays of the same shape. Raise error if broadcasting is impossible.
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
			} else {
				panic('Cannot broadcast arrays')
			}
		}
	}

	mut new_arr_1 := NDArray{
		data: arr1.data
		shape: new_shape_1.reverse()
		strides: new_strides_1.reverse()
	}
	new_arr_1.init_indices()
	if (new_arr_1.shape == arr1.shape) && (new_arr_1.strides == arr1.strides) {
		new_arr_1.contiguous = arr1.contiguous
	} else {
		new_arr_1.contiguous = false
	}

	mut new_arr_2 := NDArray{
		data: arr2.data
		shape: new_shape_2.reverse()
		strides: new_strides_2.reverse()
	}
	new_arr_2.init_indices()
	if (new_arr_2.shape == arr2.shape) && (new_arr_2.strides == arr2.strides) {
		new_arr_2.contiguous = arr2.contiguous
	} else {
		new_arr_2.contiguous = false
	}
	return new_arr_1, new_arr_2
}

// Function that operates on ndarrays element-wise
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

//
// TODO: raise error if func's args.len > 2. The reducer function should be a
// binary function
pub fn reduce_at_dim(arr NDArray, func fn (args ...f64) f64, keep_dims bool, dim int) NDArray {
	mut indices := [][]int{len: arr.shape.len, init: []int{len: 0}}
	indices[dim] = [0]
	mut result_data := get_view_linear_data(arr.slice(...indices))
	for i in 1 .. arr.shape[dim] {
		indices[dim] = [i]
		arr_data := get_view_linear_data(arr.slice(...indices))

		for j in 0 .. arr_data.len {
			result_data[j] = func(result_data[j], arr_data[j])
		}
	}
	mut new_shape := arr.shape.clone()
	new_shape[dim] = 1
	mut result := create_ndarray(result_data, ...new_shape)

	if !keep_dims {
		result = result.squeeze()
	}
	return result
}

pub fn reduce(arr NDArray, func fn (args ...f64) f64, keep_dims bool, dims ...int) NDArray {
	mut result := reduce_at_dim(arr, func, true, dims[0])
	for dim in dims[1..] {
		result = reduce_at_dim(result, func, true, dim)
	}
	return result
}
