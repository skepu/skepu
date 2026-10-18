// Full 10-digit MNIST classification with the SkePU DNN library.
//
// Usage: dnn_mnist10 model batch_size learning_rate epochs backend [train_samples] [test_samples] [--adam]
//   model:  mlp | cnn | conv1d (image rows as a 28-step sequence of 28 channels)
//   --adam: use the Adam optimizer instead of SGD (use a smaller learning rate, e.g. 0.001)
//   backend: cpu | openmp | opencl | cuda ...
//   train_samples / test_samples: optionally use only the first N samples (0 = all)
//
// Sequential build (no precompiler):
//   clang++ -std=c++20 -O2 -I ../../skepu-headers/src -Wno-attributes dnn_mnist10.cpp -o dnn_mnist10
#include <iostream>
#include <string>

#define SKEPU_ENABLE_EXCEPTIONS

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/ml.hpp>
#include <skepu-lib/dnn/csv_io.hpp>

#include "skepu-mnist.hpp"

using namespace skepu::dnn;

static skepu::Tensor4<float>
labels_to_onehot(skepu::Vector<mnist::label_t> &labels, size_t count, size_t num_classes)
{
	skepu::Tensor4<float> result(count, 1, 1, num_classes, "labels-onehot", 0.f);
	skepu::dnn::one_hot(result, labels); // (the first count labels)
	return result;
}

int main(int argc, char *argv[])
{
	// Optional flags
	Opt optimizer = Opt::SGD;
	{
		int kept = 0;
		for (int i = 0; i < argc; ++i)
		{
			if (std::string(argv[i]) == "--adam") optimizer = Opt::Adam;
			else if (std::string(argv[i]) == "--sgd") optimizer = Opt::SGD;
			else argv[kept++] = argv[i];
		}
		argc = kept;
	}

	if (argc < 6)
	{
		skepu::io::cout << "Usage: " << argv[0] << " model(mlp|cnn|conv1d) batch_size learning_rate epochs backend [train_samples] [test_samples] [--adam]\n";
		return 1;
	}

	const std::string model_name = argv[1];
	const size_t batch_size = atoi(argv[2]);
	const float learning_rate = atof(argv[3]);
	const int epochs = atoi(argv[4]);
	skepu::BackendSpec spec{argv[5]};
	skepu::setGlobalBackendSpec(spec);
	const size_t max_train = (argc > 6) ? atoi(argv[6]) : 0;
	const size_t max_test  = (argc > 7) ? atoi(argv[7]) : 0;

	const size_t num_classes = 10;

	// Models. Input: batch x 28 x 28 x 1
	auto build_mlp = [&]
	{
		return SequentialModel(Dimensions{batch_size, 28, 28, 1})
			<< Flatten()
			<< Dense(128).weights(Init::He_uniform).biases(Init::Zero).activation(Act::ReLU)
			<< Dense(num_classes).weights(Init::Xavier_uniform).biases(Init::Zero).activation(Act::SoftMax);
	};
	auto build_cnn = [&]
	{
		return SequentialModel(Dimensions{batch_size, 28, 28, 1})
			<< Conv2D(8).kernel(3, 3).weights(Init::He_uniform).biases(Init::Zero).activation(Act::ReLU)
			<< MaxPooling2D().kernel(2, 2)
			<< Conv2D(16).kernel(3, 3).weights(Init::He_uniform).biases(Init::Zero).activation(Act::ReLU)
			<< MaxPooling2D().kernel(2, 2)
			<< Flatten()
			<< Dense(num_classes).weights(Init::Xavier_uniform).biases(Init::Zero).activation(Act::SoftMax);
	};

	// Image rows as a sequence: 28 time steps with 28 channels each (input shape batch x 1 x 28 x 28)
	auto build_conv1d = [&]
	{
		return SequentialModel(Dimensions{batch_size, 1, 28, 28})
			<< Conv1D(32).kernel(5).weights(Init::He_uniform).biases(Init::Zero).activation(Act::ReLU) // 28 -> 24
			<< MaxPooling1D().kernel(2)                                                                // 24 -> 12
			<< Conv1D(32).kernel(3).weights(Init::He_uniform).biases(Init::Zero).activation(Act::ReLU) // 12 -> 10
			<< Flatten()
			<< Dense(num_classes).weights(Init::Xavier_uniform).biases(Init::Zero).activation(Act::SoftMax);
	};

	const std::string data_path = "../data/mnist/";
	skepu::Tensor4<float> train_images = mnist::parse_mnist_images(data_path + "train-images.idx3-ubyte", 0, 1.0);
	skepu::Tensor4<float> test_images  = mnist::parse_mnist_images(data_path + "t10k-images.idx3-ubyte", 0, 1.0);
	skepu::Vector<mnist::label_t> train_labels_raw = mnist::parse_mnist_labels(data_path + "train-labels.idx1-ubyte");
	skepu::Vector<mnist::label_t> test_labels_raw  = mnist::parse_mnist_labels(data_path + "t10k-labels.idx1-ubyte");

	const size_t n_train = (max_train > 0) ? std::min<size_t>(max_train, train_images.size_i()) : train_images.size_i();
	const size_t n_test  = (max_test  > 0) ? std::min<size_t>(max_test,  test_images.size_i())  : test_images.size_i();

	skepu::Tensor4<float> train_x = (model_name == "conv1d") ? skepu::dnn::as_sequence(train_images, n_train, "train sequences") : skepu::dnn::take(train_images, n_train, "train images");
	skepu::Tensor4<float> test_x  = (model_name == "conv1d") ? skepu::dnn::as_sequence(test_images, n_test, "test sequences") : skepu::dnn::take(test_images, n_test, "test images");
	skepu::Tensor4<float> train_y = labels_to_onehot(train_labels_raw, n_train, num_classes);
	skepu::Tensor4<float> test_y  = labels_to_onehot(test_labels_raw, n_test, num_classes);

	skepu::io::cout << "Training on " << n_train << " samples, testing on " << n_test << " samples\n";

	auto run = [&](auto model)
	{
		model.init();
		model.configure(Loss::CategoricalCrossEntropy, optimizer);
		skepu::io::cout << model.summary();

		model.fit(train_x, train_y, epochs, learning_rate, 0.1, "mnist10", true);

		skepu::Matrix<int> confusion(num_classes, num_classes, "Confusion Matrix", 0);
		auto score = model.evaluate(test_x, test_y, &confusion);
		skepu::io::cout << "Test loss: " << std::get<0>(score) << " Test accuracy: " << std::get<1>(score) << "\n";
		skepu::io::cout << "Confusion matrix (rows: actual, columns: predicted):\n" << confusion << "\n";
	};

	if (model_name == "mlp") run(build_mlp());
	else if (model_name == "cnn") run(build_cnn());
	else if (model_name == "conv1d") run(build_conv1d());
	else { skepu::io::cout << "Unknown model '" << model_name << "'\n"; return 1; }
	return 0;
}
