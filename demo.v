import ndarray

fn main() {
	mut a := ndarray.create_ndarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
	mut b := ndarray.create_ndarray([11.0, 22.0, 33.0, 44.0, 55.0, 66.0], 2, 3)
	ndarray.print_ndarray(a + b)

	// printing needs contiguousness anw
}
