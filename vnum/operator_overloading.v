module vnum

pub fn (a NDArray) + (b NDArray) NDArray {
	return add(a, b)
}

pub fn (a NDArray) - (b NDArray) NDArray {
	return subtract(a, b)
}

pub fn (a NDArray) * (b NDArray) NDArray {
	return multiply(a, b)
}

pub fn (a NDArray) / (b NDArray) NDArray {
	return divide(a, b)
}
