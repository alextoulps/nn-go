package nn

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/mat"
)

func MnistPredict(net *NeuralNet) {
	MnistPredictWithParams(net, 10)
}

func MnistPredictWithParams(net *NeuralNet, testSamples int) {
	t1 := time.Now()

	filename := fmt.Sprintf("datasets/mnist_test_%d.csv", testSamples)
	if testSamples > 100 {
		filename = "datasets/mnist_test.csv"
	}

	checkFile, err := os.Open(filename)
	if err != nil {
		fmt.Printf("Error opening test file: %v\n", err)
		return
	}
	defer checkFile.Close()

	score := 0
	count := 0
	r := csv.NewReader(bufio.NewReader(checkFile))
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if count >= testSamples {
			break
		}
		count++

		inputs := make([]float64, net.inputs)
		for i := range inputs {
			x, _ := strconv.ParseFloat(record[i+1], 64)
			inputs[i] = (x / 255.0 * 0.999) + 0.001
		}
		outputsResult := net.Predict(inputs)
		outputs := mat.Col(nil, 0, outputsResult)
		best := 0
		highest := 0.0
		for i := 0; i < len(outputs); i++ {
			if outputs[i] > highest {
				best = i
				highest = outputs[i]
			}
		}
		target, _ := strconv.Atoi(record[0])
		fmt.Printf("[%d/%d] target: %d, predicted: %d, correct: %v\n", count, testSamples, target, best, best == target)
		if best == target {
			score++
		}
	}

	elapsed := time.Since(t1)
	fmt.Printf("\nTest Results:\n")
	fmt.Printf("  Time taken: %s\n", elapsed)
	fmt.Printf("  Correct: %d/%d\n", score, count)
	fmt.Printf("  Accuracy: %.2f%%\n", float64(score)/float64(count)*100)
}

func MnistTrain(net *NeuralNet) {
	MnistTrainWithParams(net, 2, 1000)
}

func MnistTrainWithParams(net *NeuralNet, epochs int, trainSamples int) {
	t1 := time.Now()

	filename := fmt.Sprintf("datasets/mnist_train_%d.csv", trainSamples)
	if trainSamples > 1000 {
		filename = "datasets/mnist_train.csv"
	}

	for epoch := 0; epoch < epochs; epoch++ {
		fmt.Printf("Epoch %d/%d\n", epoch+1, epochs)

		testFile, err := os.Open(filename)
		if err != nil {
			fmt.Printf("Error opening training file: %v\n", err)
			return
		}

		count := 0
		r := csv.NewReader(bufio.NewReader(testFile))
		for {
			record, err := r.Read()
			if err == io.EOF {
				break
			}
			if count >= trainSamples {
				break
			}
			count++

			if count%100 == 0 {
				fmt.Printf("  Training sample %d/%d\n", count, trainSamples)
			}

			inputs := make([]float64, net.inputs)
			for i := range inputs {
				x, _ := strconv.ParseFloat(record[i+1], 64)
				inputs[i] = (x / 255.0 * 0.999) + 0.001
			}

			targets := make([]float64, 10)
			for i := range targets {
				targets[i] = 0.001
			}
			x, _ := strconv.Atoi(record[0])
			targets[x] = 0.999

			net.Train(inputs, targets)
		}
		testFile.Close()
	}
	elapsed := time.Since(t1)
	fmt.Printf("\nTraining completed:\n")
	fmt.Printf("  Time taken: %s\n", elapsed)
	fmt.Printf("  Epochs: %d\n", epochs)
	fmt.Printf("  Samples per epoch: %d\n", trainSamples)
}
