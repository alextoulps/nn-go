package nn

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

func RandomArray(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	data = make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
		if data[i] == 0 {
			data[i] = 0.001
		}
	}
	return data
}

func ZeroArray(size int) []float64 {
	return make([]float64, size)
}

func ZeroPointOneArray(size int) []float64 {
	arr := make([]float64, size)
	for i := 0; i < len(arr); i++ {
		arr[i] = 0.1
	}
	return arr
}

func MatrixProduct(x, y mat.Matrix) mat.Matrix {
	row, _ := x.Dims()
	_, column := y.Dims()
	out := mat.NewDense(row, column, nil)
	out.Product(x, y)
	return out
}

func MatrixAdd(x, y mat.Matrix) mat.Matrix {
	r, c := x.Dims()
	res := mat.NewDense(r, c, nil)
	res.Add(x, y)
	return res
}

func MatrixSub(x, y mat.Matrix) mat.Matrix {
	r, c := x.Dims()
	res := mat.NewDense(r, c, nil)
	res.Sub(x, y)
	return res
}

func MatrixMul(x, y mat.Matrix) mat.Matrix {
	r, c := x.Dims()
	res := mat.NewDense(r, c, nil)
	res.MulElem(x, y)
	return res
}

func MatrixApply(x mat.Matrix, f ActivationFunction, activate bool) mat.Matrix {
	r, c := x.Dims()
	res := mat.NewDense(r, c, nil)
	res.Apply(func(i, j int, v float64) float64 {
		if activate {
			return f.activate(v)
		}
		return f.derivative(v)
	}, x)
	return res
}

func MatrixScale(factor float64, x mat.Matrix) mat.Matrix {
	r, c := x.Dims()
	res := mat.NewDense(r, c, nil)
	res.Scale(factor, x)
	return res
}
