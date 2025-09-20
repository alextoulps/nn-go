package main

import (
	"fmt"

	"github.com/alextoulps/nn-go/pkg/nn"
)

func main() {
	net := nn.NewNeuralNet(3, 2, []int{10, 15}, 0.1)
	for i := 0; i < 5000; i++ {

		net.Train([]float64{0.01, 0.1, 0.3}, []float64{0, 1})
		net.Train([]float64{0.01, 0.1, 0.3}, []float64{0, 1})
		net.Train([]float64{0.1, 0.91, 0.8}, []float64{1, 0})
		net.Train([]float64{0.2, 0.91, 0.8}, []float64{1, 0})
		net.Train([]float64{0.3, 0.91, 0.8}, []float64{1, 0})
		net.Train([]float64{0.01, 0.11, 0.2}, []float64{0, 1})
		net.Train([]float64{0.02, 0.14, 0.2}, []float64{0, 1})
		net.Train([]float64{0.1, 0.91, 0.8}, []float64{1, 0})
		net.Train([]float64{0.2, 0.91, 0.8}, []float64{1, 0})
		net.Train([]float64{0.3, 0.91, 0.8}, []float64{1, 0})
		net.Train([]float64{0.01, 0.11, 0.2}, []float64{0, 1})
		net.Train([]float64{0.02, 0.14, 0.2}, []float64{0, 1})
	}
	fmt.Println("prediction", net.Predict([]float64{0.01, 0.11, 0.2}))

	net = nn.NewNeuralNet(784, 10, []int{100, 50}, 0.01)
	nn.MnistTrain(net)
	nn.MnistPredict(net)
}
