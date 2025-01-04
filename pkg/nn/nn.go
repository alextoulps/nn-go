package nn

import (
	"fmt"
	"math"
	"time"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
)

type NeuralNet struct {
	input        *InputLayer
	hiddenLayers []NeuralLayer
	inputs       int
	outputs      int
	learningRate float64
}

type InputLayer struct {
	inputs *mat.Dense
}

type NeuralLayer struct {
	weights  *mat.Dense
	bias     *mat.Dense
	outputs  *mat.Dense
	inputs   *mat.Dense
	activate ActivationFunction
	neurons  int
}

func NewNeuralNet(inputs, outputs int, hidden []int, learningRate float64) *NeuralNet {
	net := &NeuralNet{
		inputs:       inputs,
		outputs:      outputs,
		learningRate: learningRate,
	}
	var layers []NeuralLayer

	nextOutputs := inputs
	outLayerInputs := outputs
	if len(hidden) > 0 {
		outLayerInputs = hidden[len(hidden)-1]
	}
	//	layers = append(layers, *CreateNeuralLayer(inputOutputs, inputs))
	for i := 0; i < len(hidden); i++ {
		layers = append(layers, *CreateNeuralLayer(hidden[i], nextOutputs))
		nextOutputs = hidden[i]
	}

	layers = append(layers, *CreateNeuralLayer(outputs, outLayerInputs))
	net.hiddenLayers = layers
	return net
}

func (n *NeuralNet) Train(in []float64, actual []float64) {
	rand.Seed(uint64(time.Now().UnixNano()))
	input := mat.NewDense(len(in), 1, in)
	result := n.feedForward(input)

	// back propagation
	actualM := mat.NewDense(len(actual), 1, actual)
	// Error of output layer
	error := MatrixSub(actualM, result)

	for i := len(n.hiddenLayers) - 1; i >= 0; i-- {
		layer := &n.hiddenLayers[i]
		derivative := MatrixApply(layer.outputs, layer.activate, false)
		mulError := MatrixMul(error, derivative)
		costProd := MatrixProduct(mulError, layer.inputs.T())
		layer.bias = MatrixAdd(layer.bias, MatrixScale(n.learningRate, error)).(*mat.Dense)
		errorScale := MatrixScale(n.learningRate, costProd)

		error = MatrixProduct(layer.weights.T(), error)
		layer.weights = MatrixAdd(layer.weights, errorScale).(*mat.Dense)
	}
}

func (n *NeuralNet) Predict(in []float64) mat.Matrix {
	input := mat.NewDense(len(in), 1, in)

	resultMatrix := n.feedForward(input)
	return resultMatrix
}

func (n *NeuralNet) feedForward(input *mat.Dense) *mat.Dense {
	next := input
	for i := 0; i < len(n.hiddenLayers); i++ {
		layer := &n.hiddenLayers[i]
		layer.inputs = next
		layerInput := MatrixProduct(layer.weights, next)
		layerInput = MatrixAdd(layerInput, layer.bias)
		next = MatrixApply(layerInput, layer.activate, true).(*mat.Dense)
		layer.outputs = next
	}
	return SoftMax(next)
}

func (n *NeuralNet) DebugPrint() {
	fmt.Println("inputs", n.inputs, "hidden", len(n.hiddenLayers)-1, "outputs", n.outputs)
	for idx, layer := range n.hiddenLayers {
		fmt.Println("layer", idx)
		layer.DebugPrint()
	}
}

func CreateNeuralLayer(neurons, inputs int) *NeuralLayer {
	return &NeuralLayer{
		neurons:  neurons,
		activate: new(Relu),
		weights:  mat.NewDense(neurons, inputs, RandomArray(neurons*inputs, float64(inputs))),
		bias:     mat.NewDense(neurons, 1, ZeroPointOneArray(neurons)),
	}
}

func (l *NeuralLayer) DebugPrint() {
	fmt.Println("neurons", l.neurons, "inputs", l.inputs)
	wfa := mat.Formatted(l.weights, mat.Prefix("   "), mat.Squeeze())
	bfa := mat.Formatted(l.bias, mat.Prefix("   "), mat.Squeeze())
	fmt.Println("weights")
	fmt.Println(wfa)
	fmt.Println("bias")
	fmt.Println(bfa)
}

type ActivationFunction interface {
	activate(x float64) float64
	derivative(x float64) float64
}

type Relu struct{}

func (r *Relu) activate(x float64) float64 {
	return math.Max(x, 0)
}

func (r *Relu) derivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

type Sigmoid struct{}

func (s *Sigmoid) activate(x float64) float64 {
	return 1.0 / (1 + math.Exp(-1*x))
}

func (s *Sigmoid) derivative(x float64) float64 {
	return x * (1 - x)
}

func SoftMax(matrix *mat.Dense) *mat.Dense {
	var sum float64
	// Calculate the sum
	for _, v := range matrix.RawMatrix().Data {
		sum += math.Exp(v)
	}

	resultMatrix := mat.NewDense(matrix.RawMatrix().Rows, matrix.RawMatrix().Cols, nil)
	// Calculate softmax value for each element
	resultMatrix.Apply(func(i int, j int, v float64) float64 {
		return math.Exp(v) / sum
	}, matrix)

	return resultMatrix
}
