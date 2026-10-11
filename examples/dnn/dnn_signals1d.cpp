// 1D signal classification with the SkePU DNN library: a synthetic waveform classification task
// that exercises Conv1D / MaxPooling1D without needing any data files.
//
// Four classes of noisy periodic waveforms (sine, square, sawtooth, triangle) with random period,
// phase and amplitude. Phase and period invariance make this a natural fit for convolutions.
//
// Usage: dnn_signals1d model batch_size learning_rate epochs backend [train_samples] [test_samples] [--adam]
//   model: conv1d | mlp (fully connected baseline)
//
// Sequential build (no precompiler):
//   clang++ -std=c++20 -O2 -I ../../skepu-headers/src -Wno-attributes dnn_signals1d.cpp -o dnn_signals1d
#include <iostream>
#include <random>
#include <cmath>
#include <string>

#define SKEPU_ENABLE_EXCEPTIONS

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/ml.hpp>

using namespace skepu::dnn;

constexpr size_t SIGNAL_LENGTH = 64;
constexpr size_t NUM_CLASSES = 4;
static const char *class_names[NUM_CLASSES] = {"sine", "square", "sawtooth", "triangle"};

// std::uniform_real_distribution and std::normal_distribution are not specified exactly and give different
// numbers with different standard libraries, std::mt19937 is. Building the distributions on its raw output
// makes the data (and the logs of the regression test) the same with every compiler.
static float uniform01(std::mt19937 &rng) { return (float)(rng() >> 8) * (1.0f / 16777216.0f); } // [0, 1)
static float uniform(std::mt19937 &rng, float lo, float hi) { return lo + (hi - lo) * uniform01(rng); }
static float normal(std::mt19937 &rng, float sigma) // Box-Muller
{
	const float u1 = 1.0f - uniform01(rng), u2 = uniform01(rng);
	return sigma * std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
}

// Fill x (n x 1 x SIGNAL_LENGTH x 1) and one-hot y (n x 1 x 1 x NUM_CLASSES)
static void generate(skepu::Tensor4<float> &x, skepu::Tensor4<float> &y, std::mt19937 &rng, float noise)
{
	const float two_pi = 6.28318530718f;

	for (size_t i = 0; i < x.size_i(); ++i)
	{
		const size_t cls = i % NUM_CLASSES;
		const float period = uniform(rng, 8.f, 20.f), phase = uniform(rng, 0.f, 1.f), amp = uniform(rng, 0.7f, 1.3f);
		for (size_t t = 0; t < SIGNAL_LENGTH; ++t)
		{
			float u = (float)t / period + phase;
			u -= std::floor(u); // position within the period, [0, 1)
			float v = 0;
			switch (cls)
			{
			case 0: v = std::sin(two_pi * u); break;
			case 1: v = (u < 0.5f) ? 1.f : -1.f; break;
			case 2: v = 2.f * u - 1.f; break;
			case 3: v = 1.f - 4.f * std::fabs(u - 0.5f); break;
			}
			x(i, 0, t, 0) = amp * v + normal(rng, noise);
		}
		for (size_t c = 0; c < NUM_CLASSES; ++c)
			y(i, 0, 0, c) = (c == cls) ? 1.f : 0.f;
	}
}

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
		std::cout << "Usage: " << argv[0] << " model(conv1d|mlp) batch_size learning_rate epochs backend [train_samples] [test_samples] [--adam]\n";
		return 1;
	}

	const std::string model_name = argv[1];
	const size_t batch_size = atoi(argv[2]);
	const float learning_rate = atof(argv[3]);
	const int epochs = atoi(argv[4]);
	skepu::BackendSpec spec{argv[5]};
	skepu::setGlobalBackendSpec(spec);
	const size_t n_train = (argc > 6) ? atoi(argv[6]) : 4000;
	const size_t n_test  = (argc > 7) ? atoi(argv[7]) : 1000;

	std::mt19937 rng(42);
	skepu::Tensor4<float> train_x(n_train, 1, SIGNAL_LENGTH, 1, "train signals"), train_y(n_train, 1, 1, NUM_CLASSES, "train labels");
	skepu::Tensor4<float> test_x(n_test, 1, SIGNAL_LENGTH, 1, "test signals"), test_y(n_test, 1, 1, NUM_CLASSES, "test labels");
	generate(train_x, train_y, rng, 0.3f);
	generate(test_x, test_y, rng, 0.3f);

	// Input shape: batch x 1 x length x channels
	auto build_conv1d = [&]
	{
		return SequentialModel(Dimensions{batch_size, 1, SIGNAL_LENGTH, 1})
			<< Conv1D(8).kernel(5).weights(Init::He_uniform).biases(Init::Zero).activation(Act::ReLU)  // 64 -> 60
			<< MaxPooling1D().kernel(2)                                                              // 60 -> 30
			<< Conv1D(16).kernel(5).weights(Init::He_uniform).biases(Init::Zero).activation(Act::ReLU) // 30 -> 26
			<< MaxPooling1D().kernel(2)                                                              // 26 -> 13
			<< Flatten()
			<< Dense(NUM_CLASSES).weights(Init::Xavier_uniform).biases(Init::Zero).activation(Act::SoftMax);
	};
	auto build_mlp = [&]
	{
		return SequentialModel(Dimensions{batch_size, 1, SIGNAL_LENGTH, 1})
			<< Flatten()
			<< Dense(64).weights(Init::He_uniform).biases(Init::Zero).activation(Act::ReLU)
			<< Dense(NUM_CLASSES).weights(Init::Xavier_uniform).biases(Init::Zero).activation(Act::SoftMax);
	};

	std::cout << "Signals: " << n_train << " train / " << n_test << " test, length " << SIGNAL_LENGTH << ", classes:";
	for (auto name : class_names) std::cout << " " << name;
	std::cout << "\n";

	auto run = [&](auto model)
	{
		model.init();
		model.configure(Loss::CategoricalCrossEntropy, optimizer);
		std::cout << model.summary();
		model.fit(train_x, train_y, epochs, learning_rate, 0.1, "signals1d", true);

		skepu::Matrix<int> confusion(NUM_CLASSES, NUM_CLASSES, "Confusion Matrix", 0);
		auto score = model.evaluate(test_x, test_y, &confusion);
		skepu::io::cout << "Test loss: " << std::get<0>(score) << " Test accuracy: " << std::get<1>(score) << "\n";
		skepu::io::cout << "Confusion matrix (rows: actual, columns: predicted):\n" << confusion << "\n";
	};

	if (model_name == "conv1d") run(build_conv1d());
	else if (model_name == "mlp") run(build_mlp());
	else { std::cout << "Unknown model '" << model_name << "'\n"; return 1; }
	return 0;
}
