package nn

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"gonum.org/v1/gonum/mat"
)

type SerializedNetwork struct {
	Version      uint32
	Inputs       int32
	Outputs      int32
	LearningRate float64
	LayerCount   int32
	Layers       []SerializedLayer
}

type SerializedLayer struct {
	Neurons        int32
	ActivationType int32 // 0=ReLU, 1=Sigmoid
	WeightsRows    int32
	WeightsCols    int32
	WeightsData    []float64
	BiasRows       int32
	BiasCols       int32
	BiasData       []float64
}

const (
	NetworkFileVersion = 1
	ActivationReLU     = 0
	ActivationSigmoid  = 1
)

func (n *NeuralNet) Save(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	serialized := SerializedNetwork{
		Version:      NetworkFileVersion,
		Inputs:       int32(n.inputs),
		Outputs:      int32(n.outputs),
		LearningRate: n.learningRate,
		LayerCount:   int32(len(n.hiddenLayers)),
		Layers:       make([]SerializedLayer, len(n.hiddenLayers)),
	}

	for i, layer := range n.hiddenLayers {
		activationType := ActivationReLU
		if _, ok := layer.activate.(*Sigmoid); ok {
			activationType = ActivationSigmoid
		}

		weightsRows, weightsCols := layer.weights.Dims()
		biasRows, biasCols := layer.bias.Dims()

		serialized.Layers[i] = SerializedLayer{
			Neurons:        int32(layer.neurons),
			ActivationType: int32(activationType),
			WeightsRows:    int32(weightsRows),
			WeightsCols:    int32(weightsCols),
			WeightsData:    layer.weights.RawMatrix().Data,
			BiasRows:       int32(biasRows),
			BiasCols:       int32(biasCols),
			BiasData:       layer.bias.RawMatrix().Data,
		}
	}

	return writeSerializedNetwork(file, &serialized)
}

func LoadNeuralNet(filename string) (*NeuralNet, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	serialized, err := readSerializedNetwork(file)
	if err != nil {
		return nil, err
	}

	if serialized.Version != NetworkFileVersion {
		return nil, fmt.Errorf("unsupported network file version: %d", serialized.Version)
	}

	net := &NeuralNet{
		inputs:       int(serialized.Inputs),
		outputs:      int(serialized.Outputs),
		learningRate: serialized.LearningRate,
		hiddenLayers: make([]NeuralLayer, serialized.LayerCount),
	}

	for i, serializedLayer := range serialized.Layers {
		var activate ActivationFunction
		switch serializedLayer.ActivationType {
		case ActivationReLU:
			activate = &Relu{}
		case ActivationSigmoid:
			activate = &Sigmoid{}
		default:
			return nil, fmt.Errorf("unknown activation type: %d", serializedLayer.ActivationType)
		}

		weights := mat.NewDense(int(serializedLayer.WeightsRows), int(serializedLayer.WeightsCols), serializedLayer.WeightsData)
		bias := mat.NewDense(int(serializedLayer.BiasRows), int(serializedLayer.BiasCols), serializedLayer.BiasData)

		net.hiddenLayers[i] = NeuralLayer{
			weights:  weights,
			bias:     bias,
			activate: activate,
			neurons:  int(serializedLayer.Neurons),
		}
	}

	return net, nil
}

func writeSerializedNetwork(w io.Writer, network *SerializedNetwork) error {
	if err := binary.Write(w, binary.LittleEndian, network.Version); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, network.Inputs); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, network.Outputs); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, network.LearningRate); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, network.LayerCount); err != nil {
		return err
	}

	for _, layer := range network.Layers {
		if err := writeSerializedLayer(w, &layer); err != nil {
			return err
		}
	}

	return nil
}

func writeSerializedLayer(w io.Writer, layer *SerializedLayer) error {
	if err := binary.Write(w, binary.LittleEndian, layer.Neurons); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, layer.ActivationType); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, layer.WeightsRows); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, layer.WeightsCols); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, layer.WeightsData); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, layer.BiasRows); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, layer.BiasCols); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, layer.BiasData); err != nil {
		return err
	}

	return nil
}

func readSerializedNetwork(r io.Reader) (*SerializedNetwork, error) {
	network := &SerializedNetwork{}

	if err := binary.Read(r, binary.LittleEndian, &network.Version); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &network.Inputs); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &network.Outputs); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &network.LearningRate); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &network.LayerCount); err != nil {
		return nil, err
	}

	network.Layers = make([]SerializedLayer, network.LayerCount)
	for i := int32(0); i < network.LayerCount; i++ {
		layer, err := readSerializedLayer(r)
		if err != nil {
			return nil, err
		}
		network.Layers[i] = *layer
	}

	return network, nil
}

func readSerializedLayer(r io.Reader) (*SerializedLayer, error) {
	layer := &SerializedLayer{}

	if err := binary.Read(r, binary.LittleEndian, &layer.Neurons); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &layer.ActivationType); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &layer.WeightsRows); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &layer.WeightsCols); err != nil {
		return nil, err
	}

	weightsSize := int(layer.WeightsRows * layer.WeightsCols)
	layer.WeightsData = make([]float64, weightsSize)
	if err := binary.Read(r, binary.LittleEndian, layer.WeightsData); err != nil {
		return nil, err
	}

	if err := binary.Read(r, binary.LittleEndian, &layer.BiasRows); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &layer.BiasCols); err != nil {
		return nil, err
	}

	biasSize := int(layer.BiasRows * layer.BiasCols)
	layer.BiasData = make([]float64, biasSize)
	if err := binary.Read(r, binary.LittleEndian, layer.BiasData); err != nil {
		return nil, err
	}

	return layer, nil
}
