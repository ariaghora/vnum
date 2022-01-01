import time {new_stopwatch}
import vnum
import vtl

fn main() {
	arr1 := []f64{len: 64 * 32 * 28 * 28, init: it}
	shp1 := [64, 32, 28, 28]
	arr2 := []f64{len: 32, init: 28}
	shp2 := [32, 1, 1]

	println('Start')
	mut sw := new_stopwatch()
	a := vtl.from_array(arr1, shp1)
	b := vtl.from_array(arr2, shp2)
	vtl.add(a, b)
	println(sw.elapsed().milliseconds())

	sw = new_stopwatch()
	a_ := vnum.create_ndarray(arr1, ...shp1)
	b_ := vnum.create_ndarray(arr2, ...shp2)
	vnum.add(a_, b_)
	println(sw.elapsed().milliseconds())
}