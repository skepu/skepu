// Builds a SkePU-DNN model from a model file, trains it on MNIST, saves it (structure and trained
// parameters) and loads it again to check that the loaded model gives the same test accuracy.
//
// Usage: dnn_model_file model_file batch_size learning_rate epochs backend [train_samples] [test_samples] [saved_model_file]
//   e.g. dnn_model_file models/mnist_cnn.dnn 32 0.05 2 cpu 3200 640 trained.dnn
//
// Sequential build (no precompiler):
//   clang++ -std=c++20 -O2 -I ../skepu-headers/src -Wno-attributes dnn_model_file.cpp -o dnn_model_file
#include <iostream>
#include <string>

#define SKEPU_ENABLE_EXCEPTIONS

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/ml.hpp>

#include "skepu-mnist.hpp"

using namespace skepu::dnn;

static skepu::Tensor4<float> first_n(skepu::Tensor4<float> &src, size_t n)
{
	skepu::Tensor4<float> dst(n, src.size_j(), src.size_k(), src.size_l(), "images-subset");
	for (size_t i = 0; i < n; ++i)
		for (size_t j = 0; j < src.size_j(); ++j)
			for (size_t k = 0; k < src.size_k(); ++k)
				for (size_t l = 0; l < src.size_l(); ++l)
					dst(i, j, k, l) = src(i, j, k, l);
	return dst;
}

static skepu::Tensor4<float> onehot(skepu::Vector<mnist::label_t> &labels, size_t n)
{
	skepu::Tensor4<float> y(n, 1, 1, 10, "labels", 0.f);
	for (size_t i = 0; i < n; ++i) y(i, 0, 0, labels(i)) = 1.f;
	return y;
}

int main(int argc, char *argv[])
{
	if (argc < 6)
	{
		std::cout << "Usage: " << argv[0] << " model_file batch_size learning_rate epochs backend [train_samples] [test_samples] [saved_model_file]\n";
		return 1;
	}
	const std::string model_file = argv[1];
	const size_t batch_size = atoi(argv[2]);
	const float learning_rate = atof(argv[3]);
	const int epochs = atoi(argv[4]);
	skepu::BackendSpec spec{argv[5]};
	skepu::setGlobalBackendSpec(spec);
	const size_t max_train = (argc > 6) ? atoi(argv[6]) : 0;
	const size_t max_test  = (argc > 7) ? atoi(argv[7]) : 0;
	const std::string saved_file = (argc > 8) ? argv[8] : "trained.dnn";

	const std::string data_path = "./data/mnist/";
	skepu::Tensor4<float> train_images = mnist::parse_mnist_images(data_path + "train-images.idx3-ubyte", 0, 1.0);
	skepu::Tensor4<float> test_images  = mnist::parse_mnist_images(data_path + "t10k-images.idx3-ubyte", 0, 1.0);
	skepu::Vector<mnist::label_t> train_labels = mnist::parse_mnist_labels(data_path + "train-labels.idx1-ubyte");
	skepu::Vector<mnist::label_t> test_labels  = mnist::parse_mnist_labels(data_path + "t10k-labels.idx1-ubyte");
	const size_t n_train = max_train ? std::min<size_t>(max_train, train_images.size_i()) : train_images.size_i();
	const size_t n_test  = max_test  ? std::min<size_t>(max_test,  test_images.size_i())  : test_images.size_i();
	skepu::Tensor4<float> train_x = first_n(train_images, n_train), test_x = first_n(test_images, n_test);
	skepu::Tensor4<float> train_y = onehot(train_labels, n_train), test_y = onehot(test_labels, n_test);

	// Build the model from its description
	auto model = load_model(model_file, batch_size);
	std::cout << model.summary();

	model.configure(Loss::CategoricalCrossEntropy, Opt::SGD);
	model.fit(train_x, train_y, epochs, learning_rate, 0.1, "model_file", true);
	auto score = model.evaluate(test_x, test_y);
	skepu::io::cout << "Test accuracy of the trained model: " << std::get<1>(score) << "\n";

	// Save, load again (the loaded model is ready for inference) and compare
	model.save(saved_file);
	auto loaded = load_model(saved_file, batch_size);
	loaded.configure(Loss::CategoricalCrossEntropy, Opt::SGD);
	auto loaded_score = loaded.evaluate(test_x, test_y);
	skepu::io::cout << "Test accuracy of the model loaded from " << saved_file << ": " << std::get<1>(loaded_score) << "\n";
	return 0;
}
