// CIFAR-10 classification with the SkePU DNN library.
//
// Usage: dnn_cifar10 model batch_size learning_rate epochs backend [train_samples] [test_samples] [--adam] [--gray]
//   model: small | big
//   --gray: a "DNN+X" program: the images are converted to grayscale by a MapPool skeleton before the network sees them (the
//           preprocessing is SkePU code on the same containers as the network, 32x32x1 images, no copies to the host)
//   small: 2 conv layers (16, 32 filters) + dense;   big: 3 conv layers (32, 64, 64) + 2 dense layers
//   train_samples / test_samples: use only the first N samples (0 = all)
//
// Expects the CIFAR-10 binary version in ./data/cifar-10-batches-bin/.
// Sequential build (no precompiler):
//   clang++ -std=c++20 -O3 -I ../../skepu-headers/src -Wno-attributes dnn_cifar10.cpp -o dnn_cifar10
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#define SKEPU_ENABLE_EXCEPTIONS

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/benchmark.hpp>
#include <skepu-lib/ml.hpp>

#include "skepu-cifar10.hpp"

using namespace skepu::dnn;

// Grayscale (luma) of a pixel: the pool is the three color channels of one pixel (pool size 1 x 1 x 1 x 3)
float rgb_to_gray(skepu::Pool4D<float> pool)
{
	return 0.2989f * pool(0, 0, 0, 0) + 0.5870f * pool(0, 0, 0, 1) + 0.1140f * pool(0, 0, 0, 2);
}

// The grayscale images (n x 32 x 32 x 1) of color images (n x 32 x 32 x 3)
static skepu::Tensor4<float> *to_grayscale(skepu::Tensor4<float> &images, const char *label)
{
	auto convert = skepu::MapPool(rgb_to_gray);
	convert.setPoolSize(1, 1, 1, cifar10::CHANNELS);
	auto *gray = new skepu::Tensor4<float>(images.size_i(), images.size_j(), images.size_k(), 1, label);
	convert(*gray, images);
	return gray;
}

int main(int argc, char *argv[])
{
	Opt optimizer = Opt::SGD;
	bool gray = false;
	{
		int kept = 0;
		for (int i = 0; i < argc; ++i)
		{
			if (std::string(argv[i]) == "--adam") optimizer = Opt::Adam;
			else if (std::string(argv[i]) == "--gray") gray = true;
			else if (std::string(argv[i]) == "--sgd") optimizer = Opt::SGD;
			else argv[kept++] = argv[i];
		}
		argc = kept;
	}

	if (argc < 6)
	{
		skepu::io::cout << "Usage: " << argv[0] << " model(small|big) batch_size learning_rate epochs backend [train_samples] [test_samples] [--adam] [--gray]\n";
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

	const std::string data_path = "../data/cifar-10-batches-bin/";
	std::vector<std::string> train_files;
	for (int i = 1; i <= 5; ++i) train_files.push_back(data_path + "data_batch_" + std::to_string(i) + ".bin");

	skepu::Tensor4<float> *train_x, *train_y, *test_x, *test_y;
	cifar10::load(train_files, max_train, train_x, train_y);
	cifar10::load({data_path + "test_batch.bin"}, max_test, test_x, test_y);

	// Optional preprocessing: color to grayscale, by a skeleton, on the containers that the network trains on
	if (gray)
	{
		train_x = to_grayscale(*train_x, "CIFAR-10 grayscale images");
		test_x = to_grayscale(*test_x, "CIFAR-10 grayscale test images");
	}
	const size_t channels = train_x->size_l();

	// Standardize with the statistics of the training set
	auto stats = skepu::dnn::channel_stats(*train_x, "CIFAR-10 channel");
	skepu::dnn::standardize(*train_x, stats);
	skepu::dnn::standardize(*test_x, stats);

	skepu::io::cout << "CIFAR-10: " << train_x->size_i() << " training samples, " << test_x->size_i() << " test samples\n";

	auto build_small = [&]
	{
		return SequentialModel(Dimensions{batch_size, 32, 32, channels})
			<< Conv2D(16).kernel(3, 3).weights(Init::He_uniform).biases(Init::Zero).activation(Act::ReLU) // 30x30
			<< MaxPooling2D().kernel(2, 2)                                                               // 15x15
			<< Conv2D(32).kernel(3, 3).weights(Init::He_uniform).biases(Init::Zero).activation(Act::ReLU) // 13x13
			<< MaxPooling2D().kernel(2, 2)                                                               // 6x6
			<< Flatten()
			<< Dense(cifar10::NUM_CLASSES).weights(Init::Xavier_uniform).biases(Init::Zero).activation(Act::SoftMax);
	};
	auto build_big = [&]
	{
		return SequentialModel(Dimensions{batch_size, 32, 32, channels})
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
		skepu::io::cout << model.summary();

		auto train_time = skepu::benchmark::measureExecTime([&]
		{
			model.fit(*train_x, *train_y, epochs, learning_rate, 0.1, "cifar10", true);
		});
		const double train_seconds = std::chrono::duration<double>(train_time).count();

		skepu::Matrix<int> confusion(cifar10::NUM_CLASSES, cifar10::NUM_CLASSES, "Confusion Matrix", 0);
		auto score = model.evaluate(*test_x, *test_y, &confusion);
		skepu::io::cout << "Test loss: " << std::get<0>(score) << " Test accuracy: " << std::get<1>(score) << "\n";
		skepu::io::cout << "Training time: " << train_seconds << " s\n";
		skepu::io::cout << "Confusion matrix (rows: actual, columns: predicted):\n" << confusion << "\n";
		skepu::io::cout << "Classes:";
		for (auto name : cifar10::class_names) skepu::io::cout << " " << name;
		skepu::io::cout << "\n";
	};

	if (model_name == "small") run(build_small());
	else if (model_name == "big") run(build_big());
	else { skepu::io::cout << "Unknown model '" << model_name << "'\n"; return 1; }
	return 0;
}
