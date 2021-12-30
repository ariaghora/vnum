import ndarray

fn main() {
	mut a := ndarray.create_ndarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 1, 2)
	ndarray.print_ndarray(a)
}
