// Toy classification on a data set that is generated on the device: no data files, so it also works as a
// quick check of a backend (PRNG kernels, Map with random numbers, training end to end).
//
// Each sample is a random integer class c in [0, classes), given to the network as a single input value;
// the target is the one-hot vector of c. A small MLP learns this mapping.
//
// Usage: dnn_artificial batch_size learning_rate epochs backend [classes] [train_samples]
//
// Sequential build (no precompiler):
//   clang++ -std=c++20 -O2 -I ../skepu-headers/src -Wno-attributes dnn_artificial.cpp -o dnn_artificial
#include <iostream>
#include <string>

#define SKEPU_ENABLE_EXCEPTIONS

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/ml.hpp>

using namespace skepu::dnn;

// A random class in [0, classes) for every sample
float random_class_uf(skepu::Random<1> &rand, int classes)
{
	return (int)(rand.getNormalized() * classes);
}

// One-hot encoding of the class stored in x(sample, 0, 0, 0)
float one_hot_uf(skepu::Index4D index, skepu::Ten4<float> x)
{
	return (index.l == (size_t)x(index.i, 0, 0, 0)) ? 1.f : 0.f;
}

int main(int argc, char *argv[])
{
	if (argc < 5)
	{
		std::cout << "Usage: " << argv[0] << " batch_size learning_rate epochs backend [classes] [train_samples]\n";
		return 1;
	}

	const size_t batch_size = atoi(argv[1]);
	const float learning_rate = atof(argv[2]);
	const int epochs = atoi(argv[3]);
	skepu::BackendSpec spec{argv[4]};
	skepu::setGlobalBackendSpec(spec);
	const int classes = (argc > 5) ? atoi(argv[5]) : 5;
	const size_t n_train = (argc > 6) ? atoi(argv[6]) : 1024;
	const size_t n_test = 256;

	auto generate_class = skepu::Map<0>(random_class_uf);
	auto generate_target = skepu::Map<0>(one_hot_uf);
	skepu::PRNG prng(0);
	generate_class.setPRNG(prng);

	skepu::Tensor4<float> train_x(n_train, 1, 1, 1, "train classes"), train_y(n_train, 1, 1, classes, "train targets");
	skepu::Tensor4<float> test_x(n_test, 1, 1, 1, "test classes"), test_y(n_test, 1, 1, classes, "test targets");
	generate_class(train_x, classes);
	generate_target(train_y, train_x);
	generate_class(test_x, classes);
	generate_target(test_y, test_x);

	auto model = SequentialModel(Dimensions{batch_size, 1, 1, 1})
		<< Dense(32).weights(Init::He_uniform).biases(Init::Uniform).activation(Act::ReLU)
		<< Dense(32).weights(Init::He_uniform).biases(Init::Zero).activation(Act::ReLU)
		<< Dense(classes).weights(Init::Xavier_uniform).biases(Init::Zero).activation(Act::SoftMax);
	model.init();
	model.configure(Loss::CategoricalCrossEntropy, Opt::Adam);
	model.fit(train_x, train_y, epochs, learning_rate, 0.1, "artificial", true);

	auto score = model.evaluate(test_x, test_y);
	skepu::io::cout << "Test loss: " << std::get<0>(score) << " Test accuracy: " << std::get<1>(score) << "\n";
	return 0;
}
