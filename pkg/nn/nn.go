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

func NewNeuralNet(inputs, outputs, hidden, hiddenNeurons int, learningRate float64) *NeuralNet {
	net := &NeuralNet{
		inputs:       inputs,
		outputs:      outputs,
		learningRate: learningRate,
	}
	var layers []NeuralLayer

	layers = append(layers, *CreateNeuralLayer(hiddenNeurons, inputs))
	for i := 0; i < hidden; i++ {
		layers = append(layers, *CreateNeuralLayer(hiddenNeurons, hiddenNeurons))
	}

	layers = append(layers, *CreateNeuralLayer(outputs, hiddenNeurons))
	net.hiddenLayers = layers
	return net
}

func (n *NeuralNet) Train(in []float64, actual []float64) {
	rand.Seed(uint64(time.Now().UnixNano()))
	n.input = CreateInputLayer(in)

	result := n.feedForward(n.input.inputs)
	actualM := mat.NewDense(len(actual), 1, actual)
	// fmt.Println("back propagation")
	outError := MatrixSub(actualM, result)

	// back propagation

	error := outError
	for i := len(n.hiddenLayers) - 1; i >= 0; i-- {

		layer := &n.hiddenLayers[i]

		derivarite := MatrixApply(layer.outputs, layer.activate, false)
		mulError := MatrixMul(error, derivarite)
		// fmt.Println("inputs", layer.weights.At(0, 0), "bias", layer.bias.At(0, 0), "idx", i)
		costProd := MatrixProduct(mulError, layer.inputs.T())
		layer.weights = MatrixAdd(layer.weights, MatrixScale(n.learningRate, costProd)).(*mat.Dense)
		layer.bias = MatrixAdd(layer.bias, MatrixScale(n.learningRate, error)).(*mat.Dense)

		// fmt.Println("inputs", layer.weights.At(0, 0), "idx", i)
		// layerInput := MatrixProduct(layer.weights, layer.inputs)
		// layerInput = MatrixAdd(layerInput, layer.bias)
		// correctedOut := MatrixApply(layerInput, layer.activate, true).(*mat.Dense)
		error = MatrixProduct(layer.weights.T(), error)
		// fmt.Println("error", error)
		// fmt.Println("cout", correctedOut)

		// layer.outputs = correctedOut
		// layer.inputs = layerInput.(*mat.Dense)
	}
}

func (n *NeuralNet) Predict(in []float64, softmax bool) []float64 {
	n.input = CreateInputLayer(in)
	resultMatrix := n.feedForward(n.input.inputs)
	result := mat.Col(nil, 0, resultMatrix)
	if softmax {
		return SoftMax(result)
	}
	return result
}

func (n *NeuralNet) feedForward(input *mat.Dense) *mat.Dense {
	next := input
	for i := 0; i < len(n.hiddenLayers); i++ {
		layer := &n.hiddenLayers[i]
		layer.inputs = next
		// fmt.Println("inputs", layer.inputs)
		layerInput := MatrixProduct(layer.weights, next)
		layerInput = MatrixAdd(layerInput, layer.bias)
		// fmt.Println("linput", layerInput)
		next = MatrixApply(layerInput, layer.activate, true).(*mat.Dense)
		layer.outputs = next
		// r, c := next.Dims()
		// fmt.Println("outputs", next, "row", r, "column", c)
		// fmt.Println("kayer w", layer.weights)
		// fmt.Println("klayer b", layer.bias)
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
		activate: new(Sigmoid),
		weights:  mat.NewDense(neurons, inputs, RandomArray(neurons*inputs, float64(inputs))),
		bias:     mat.NewDense(neurons, 1, ZeroArray(neurons)),
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

func CreateInputLayer(inputs []float64) *InputLayer {
	return &InputLayer{
		inputs: mat.NewDense(len(inputs), 1, inputs),
	}
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

func SoftMax(logits []float64) []float64 {
	maxLogit := logits[0]
	for _, logit := range logits {
		if logit > maxLogit {
			maxLogit = logit
		}
	}

	expValues := make([]float64, len(logits))
	var expSum float64
	for i, logit := range logits {
		expValues[i] = math.Exp(logit - maxLogit) // Subtract max for stability
		expSum += expValues[i]
	}

	probabilities := make([]float64, len(logits))
	for i, expValue := range expValues {
		probabilities[i] = expValue / expSum
	}

	return probabilities
}
