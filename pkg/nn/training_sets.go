package nn

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"time"
)

func MnistPredict(net *NeuralNet, softmax bool) {
	t1 := time.Now()
	checkFile, _ := os.Open("datasets/mnist_test.csv")
	defer checkFile.Close()

	score := 0
	r := csv.NewReader(bufio.NewReader(checkFile))
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		inputs := make([]float64, net.inputs)
		for i := range inputs {
			x, _ := strconv.ParseFloat(record[i+1], 64)
			inputs[i] = (x / 255.0 * 0.999) + 0.001
		}
		outputs := net.Predict(inputs, softmax)
		best := 0
		highest := 0.0
		for i := 0; i < len(outputs); i++ {
			if outputs[i] > highest {
				best = i
				highest = outputs[i]
			}
		}
		target, _ := strconv.Atoi(record[0])
		fmt.Println("target", target, "correct", best == target, "pred", outputs, "highest", best)
		if best == target {
			score++
		}
	}

	elapsed := time.Since(t1)
	fmt.Printf("Time taken to check: %s\n", elapsed)
	fmt.Println("score:", score)
}

func MnistTrain(net *NeuralNet) {
	t1 := time.Now()

	for epochs := 0; epochs < 5; epochs++ {
		testFile, _ := os.Open("datasets/mnist_train.csv")
		r := csv.NewReader(bufio.NewReader(testFile))
		for {
			record, err := r.Read()
			if err == io.EOF {
				break
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
	fmt.Printf("\nTime taken to train: %s\n", elapsed)
}
