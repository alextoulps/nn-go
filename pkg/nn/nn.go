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
	weights     *mat.Dense
	bias        *mat.Dense
	outputs     *mat.Dense
	inputs      *mat.Dense
	preActivation *mat.Dense
	activate    ActivationFunction
	neurons     int
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

		// Compute derivative using pre-activation values
		var derivative *mat.Dense
		if i == len(n.hiddenLayers)-1 && n.outputs > 1 {
			// For output layer with softmax, handle derivative differently
			// Softmax derivative is handled implicitly in the error calculation
			derivative = mat.NewDense(layer.preActivation.RawMatrix().Rows, layer.preActivation.RawMatrix().Cols, nil)
			derivative.Apply(func(i, j int, v float64) float64 { return 1.0 }, layer.preActivation)
		} else {
			// Use pre-activation values for derivative calculation
			derivative = MatrixApply(layer.preActivation, layer.activate, false).(*mat.Dense)
		}

		mulError := MatrixMul(error, derivative)
		costProd := MatrixProduct(mulError, layer.inputs.T())

		// Update bias using the weighted error (mulError)
		layer.bias = MatrixAdd(layer.bias, MatrixScale(n.learningRate, mulError)).(*mat.Dense)
		errorScale := MatrixScale(n.learningRate, costProd)

		// Propagate error to previous layer
		error = MatrixProduct(layer.weights.T(), mulError)
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
		layer.preActivation = layerInput.(*mat.Dense)
		next = MatrixApply(layerInput, layer.activate, true).(*mat.Dense)
		layer.outputs = next
	}
	// Apply softmax only to final output for multi-class classification
	if n.outputs > 1 {
		return SoftMax(next)
	}
	return next
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

// Getter methods for accessing neural network properties
func (n *NeuralNet) GetInputs() int {
	return n.inputs
}

func (n *NeuralNet) GetOutputs() int {
	return n.outputs
}

func (n *NeuralNet) GetLearningRate() float64 {
	return n.learningRate
}

func (n *NeuralNet) GetHiddenLayers() []int {
	layers := make([]int, len(n.hiddenLayers))
	for i, layer := range n.hiddenLayers {
		layers[i] = layer.neurons
	}
	return layers
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
	// x should be the pre-activation value
	activated := s.activate(x)
	return activated * (1 - activated)
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
