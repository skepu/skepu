// CIFAR-10 classification with the SkePU DNN library.
//
// Usage: dnn_cifar10 model batch_size learning_rate epochs backend [train_samples] [test_samples] [--adam]
//   model: small | big
//   small: 2 conv layers (16, 32 filters) + dense;   big: 3 conv layers (32, 64, 64) + 2 dense layers
//   train_samples / test_samples: use only the first N samples (0 = all)
//
// Expects the CIFAR-10 binary version in ./data/cifar-10-batches-bin/.
// Sequential build (no precompiler):
//   clang++ -std=c++20 -O3 -I ../skepu-headers/src -Wno-attributes dnn_cifar10.cpp -o dnn_cifar10
#include <iostream>
#include <string>
#include <chrono>

#define SKEPU_ENABLE_EXCEPTIONS

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/ml.hpp>

#include "skepu-cifar10.hpp"

using namespace skepu::dnn;

int main(int argc, char *argv[])
{
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
		std::cout << "Usage: " << argv[0] << " model(small|big) batch_size learning_rate epochs backend [train_samples] [test_samples] [--adam]\n";
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

	const std::string data_path = "./data/cifar-10-batches-bin/";
	std::vector<std::string> train_files;
	for (int i = 1; i <= 5; ++i) train_files.push_back(data_path + "data_batch_" + std::to_string(i) + ".bin");

	skepu::Tensor4<float> *train_x, *train_y, *test_x, *test_y;
	cifar10::load(train_files, max_train, train_x, train_y);
	cifar10::load({data_path + "test_batch.bin"}, max_test, test_x, test_y);

	// Standardize with the statistics of the training set
	float mean[cifar10::CHANNELS], stddev[cifar10::CHANNELS];
	cifar10::channel_stats(*train_x, mean, stddev);
	cifar10::standardize(*train_x, mean, stddev);
	cifar10::standardize(*test_x, mean, stddev);
	train_x->flush(); train_y->flush(); test_x->flush(); test_y->flush();

	std::cout << "CIFAR-10: " << train_x->size_i() << " training samples, " << test_x->size_i() << " test samples\n";

	auto build_small = [&]
	{
		return SequentialModel(Dimensions{batch_size, 32, 32, 3})
			<< Conv2D(16).kernel(3, 3).weights(Init::He_uniform).biases(Init::Zero).activation(Act::ReLU) // 30x30
			<< MaxPooling2D().kernel(2, 2)                                                               // 15x15
			<< Conv2D(32).kernel(3, 3).weights(Init::He_uniform).biases(Init::Zero).activation(Act::ReLU) // 13x13
			<< MaxPooling2D().kernel(2, 2)                                                               // 6x6
			<< Flatten()
			<< Dense(cifar10::NUM_CLASSES).weights(Init::Xavier_uniform).biases(Init::Zero).activation(Act::SoftMax);
	};
	auto build_big = [&]
	{
		return SequentialModel(Dimensions{batch_size, 32, 32, 3})
			<< Conv2D(32).kernel(3, 3).weights(Init::He_uniform).biases(Init::Zero).activation(Act::ReLU) // 30x30
			<< MaxPooling2D().kernel(2, 2)                                                               // 15x15
			<< Conv2D(64).kernel(3, 3).weights(Init::He_uniform).biases(Init::Zero).activation(Act::ReLU) // 13x13
			<< MaxPooling2D().kernel(2, 2)                                                               // 6x6
			<< Conv2D(64).kernel(3, 3).weights(Init::He_uniform).biases(Init::Zero).activation(Act::ReLU) // 4x4
			<< Flatten()
			<< Dense(64).weights(Init::He_uniform).biases(Init::Zero).activation(Act::ReLU)
			<< Dense(cifar10::NUM_CLASSES).weights(Init::Xavier_uniform).biases(Init::Zero).activation(Act::SoftMax);
	};

	auto run = [&](auto model)
	{
		model.init();
		model.configure(Loss::CategoricalCrossEntropy, optimizer);
		std::cout << model.summary();

		auto t0 = std::chrono::steady_clock::now();
		model.fit(*train_x, *train_y, epochs, learning_rate, 0.1, "cifar10", true);
		double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

		skepu::Matrix<int> confusion(cifar10::NUM_CLASSES, cifar10::NUM_CLASSES, "Confusion Matrix", 0);
		auto score = model.evaluate(*test_x, *test_y, &confusion);
		skepu::io::cout << "Test loss: " << std::get<0>(score) << " Test accuracy: " << std::get<1>(score) << "\n";
		std::cout << "Training time: " << train_seconds << " s\n";
		skepu::io::cout << "Confusion matrix (rows: actual, columns: predicted):\n" << confusion << "\n";
		std::cout << "Classes:";
		for (auto name : cifar10::class_names) std::cout << " " << name;
		std::cout << "\n";
	};

	if (model_name == "small") run(build_small());
	else if (model_name == "big") run(build_big());
	else { std::cout << "Unknown model '" << model_name << "'\n"; return 1; }
	return 0;
}
