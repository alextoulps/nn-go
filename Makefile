.PHONY: run build clean download-mnist

download-mnist:
	@chmod +x download_mnist.sh
	@./download_mnist.sh

build:
	go build -o nn-go cmd/main.go

demo: download-mnist
	go run cmd/main.go demo

train: download-mnist
	go run cmd/main.go train

train-fast: download-mnist
	go run cmd/main.go train --epochs=1 --train-samples=100 --test-samples=10 --hidden-layers=50,25

clean:
	rm -f nn-go
	rm -rf datasets

run: train