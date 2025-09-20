package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/alextoulps/nn-go/pkg/nn"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	cfgFile string

	// Training parameters
	epochs       int
	learningRate float64
	batchSize    int

	// Network architecture
	hiddenLayers []int
	hiddenLayersStr string

	// Dataset parameters
	trainSamples int
	testSamples  int

	// Model persistence
	saveModel string
	loadModel string
)

var rootCmd = &cobra.Command{
	Use:   "nn-go",
	Short: "A neural network implementation in Go",
	Long:  `A feedforward neural network with backpropagation for MNIST digit classification.`,
}

var trainCmd = &cobra.Command{
	Use:   "train",
	Short: "Train the neural network",
	Long:  `Train a neural network on the MNIST dataset with configurable parameters.`,
	Run: func(cmd *cobra.Command, args []string) {
		runTraining()
	},
}

var demoCmd = &cobra.Command{
	Use:   "demo",
	Short: "Run a simple demo",
	Long:  `Run a simple binary classification demo.`,
	Run: func(cmd *cobra.Command, args []string) {
		runDemo()
	},
}

var predictCmd = &cobra.Command{
	Use:   "predict",
	Short: "Use a trained model to make predictions",
	Long:  `Load a saved model and use it to make predictions on the test dataset.`,
	Run: func(cmd *cobra.Command, args []string) {
		runPrediction()
	},
}

func init() {
	cobra.OnInitialize(initConfig)

	// Global flags
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is ./nn-config.yaml)")

	// Training command flags
	trainCmd.Flags().IntVarP(&epochs, "epochs", "e", 5, "Number of training epochs")
	trainCmd.Flags().Float64VarP(&learningRate, "learning-rate", "l", 0.01, "Learning rate for training")
	trainCmd.Flags().StringVar(&hiddenLayersStr, "hidden-layers", "100,50", "Hidden layer sizes (comma-separated, e.g., '100,50,25')")
	trainCmd.Flags().IntVarP(&trainSamples, "train-samples", "t", 1000, "Number of training samples to use")
	trainCmd.Flags().IntVarP(&testSamples, "test-samples", "", 10, "Number of test samples to use")
	trainCmd.Flags().StringVar(&saveModel, "save", "", "Save trained model to file (e.g., --save=model.bin)")
	trainCmd.Flags().StringVar(&loadModel, "load", "", "Load pre-trained model from file (e.g., --load=model.bin)")

	// Predict command flags
	predictCmd.Flags().StringVar(&loadModel, "load", "", "Load model from file (required)")
	predictCmd.Flags().IntVarP(&testSamples, "test-samples", "", 10, "Number of test samples to use")
	predictCmd.MarkFlagRequired("load")

	// Bind flags to viper
	viper.BindPFlag("epochs", trainCmd.Flags().Lookup("epochs"))
	viper.BindPFlag("learning_rate", trainCmd.Flags().Lookup("learning-rate"))
	viper.BindPFlag("hidden_layers", trainCmd.Flags().Lookup("hidden-layers"))
	viper.BindPFlag("train_samples", trainCmd.Flags().Lookup("train-samples"))
	viper.BindPFlag("test_samples", trainCmd.Flags().Lookup("test-samples"))
	viper.BindPFlag("save_model", trainCmd.Flags().Lookup("save"))
	viper.BindPFlag("load_model", trainCmd.Flags().Lookup("load"))

	// Add commands
	rootCmd.AddCommand(trainCmd)
	rootCmd.AddCommand(demoCmd)
	rootCmd.AddCommand(predictCmd)
}

func initConfig() {
	if cfgFile != "" {
		viper.SetConfigFile(cfgFile)
	} else {
		viper.SetConfigName("nn-config")
		viper.SetConfigType("yaml")
		viper.AddConfigPath(".")
		viper.AddConfigPath("./config")
	}

	viper.SetEnvPrefix("NN")
	viper.AutomaticEnv()

	if err := viper.ReadInConfig(); err == nil {
		fmt.Println("Using config file:", viper.ConfigFileUsed())
	}
}

func runDemo() {
	fmt.Println("Running simple classification demo...")
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

	fmt.Println("Prediction for [0.01, 0.11, 0.2]:", net.Predict([]float64{0.01, 0.11, 0.2}))
}

func runTraining() {
	// Get values from viper (which includes flags, env vars, and config file)
	epochs := viper.GetInt("epochs")
	learningRate := viper.GetFloat64("learning_rate")
	trainSamples := viper.GetInt("train_samples")
	testSamples := viper.GetInt("test_samples")
	saveModel := viper.GetString("save_model")
	loadModel := viper.GetString("load_model")

	// Parse hidden layers
	var hiddenLayers []int
	if viper.IsSet("hidden_layers") {
		// Handle both string format from flags and array format from config
		hiddenLayersValue := viper.Get("hidden_layers")

		switch v := hiddenLayersValue.(type) {
		case string:
			// From command line flag: "100,50,25"
			hiddenLayers = parseHiddenLayersString(v)
		case []interface{}:
			// From config file: [100, 50, 25]
			for _, val := range v {
				if intVal, ok := val.(int); ok {
					hiddenLayers = append(hiddenLayers, intVal)
				}
			}
		case []int:
			// Direct int slice
			hiddenLayers = v
		default:
			// Fallback
			hiddenLayers = []int{100, 50}
		}
	} else {
		hiddenLayers = []int{100, 50}
	}

	var net *nn.NeuralNet

	// Load existing model or create new one
	if loadModel != "" {
		fmt.Printf("Loading model from: %s\n", loadModel)
		var err error
		net, err = nn.LoadNeuralNet(loadModel)
		if err != nil {
			fmt.Printf("Error loading model: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("✓ Model loaded successfully\n")
		fmt.Printf("  Architecture: %d inputs → %v hidden → %d outputs\n",
			net.GetInputs(), net.GetHiddenLayers(), net.GetOutputs())
		fmt.Printf("  Learning Rate: %.4f\n", net.GetLearningRate())
	} else {
		fmt.Printf("Creating new neural network with:\n")
		fmt.Printf("  Hidden Layers: %v\n", hiddenLayers)
		net = nn.NewNeuralNet(784, 10, hiddenLayers, learningRate)
	}

	fmt.Printf("\nTraining configuration:\n")
	fmt.Printf("  Epochs: %d\n", epochs)
	fmt.Printf("  Learning Rate: %.4f\n", learningRate)
	fmt.Printf("  Training Samples: %d\n", trainSamples)
	fmt.Printf("  Test Samples: %d\n", testSamples)
	if saveModel != "" {
		fmt.Printf("  Save to: %s\n", saveModel)
	}
	fmt.Println()

	// Train the network
	if epochs > 0 {
		nn.MnistTrainWithParams(net, epochs, trainSamples)
	}

	// Test the network
	nn.MnistPredictWithParams(net, testSamples)

	// Save the model if requested
	if saveModel != "" {
		fmt.Printf("\nSaving model to: %s\n", saveModel)
		if err := net.Save(saveModel); err != nil {
			fmt.Printf("Error saving model: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("✓ Model saved successfully\n")
	}
}

func parseHiddenLayersString(s string) []int {
	if s == "" {
		return []int{100, 50}
	}

	parts := strings.Split(s, ",")
	layers := make([]int, 0, len(parts))

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if val, err := strconv.Atoi(part); err == nil && val > 0 {
			layers = append(layers, val)
		}
	}

	if len(layers) == 0 {
		return []int{100, 50}
	}

	return layers
}

func runPrediction() {
	loadModel := viper.GetString("load_model")
	testSamples := viper.GetInt("test_samples")

	if loadModel == "" {
		loadModel = loadModel // Use the flag value directly
	}

	fmt.Printf("Loading model from: %s\n", loadModel)
	net, err := nn.LoadNeuralNet(loadModel)
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("✓ Model loaded successfully\n")
	fmt.Printf("  Architecture: %d inputs → %v hidden → %d outputs\n",
		net.GetInputs(), net.GetHiddenLayers(), net.GetOutputs())
	fmt.Printf("  Learning Rate: %.4f\n", net.GetLearningRate())
	fmt.Println()

	// Run predictions
	fmt.Printf("Running predictions on %d test samples:\n", testSamples)
	nn.MnistPredictWithParams(net, testSamples)
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}