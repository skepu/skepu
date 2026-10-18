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
#include <cmath>
#include <string>

#define SKEPU_ENABLE_EXCEPTIONS

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/ml.hpp>
#include <skepu-lib/randomize.hpp>

using namespace skepu::dnn;

constexpr size_t SIGNAL_LENGTH = 64;
constexpr size_t NUM_CLASSES = 4;
static const char *class_names[NUM_CLASSES] = {"sine", "square", "sawtooth", "triangle"};

// The waveform of a sample, at the time t (the index k): the class of a sample is its number modulo the number of classes,
// and its period, phase and amplitude are random numbers of the sample. The noise is the value that is in the element.
float waveform(skepu::Index4D idx, float noise, skepu::Vec<float> period, skepu::Vec<float> phase, skepu::Vec<float> amplitude, int classes)
{
	const float two_pi = 6.28318530718f;
	const size_t sample = idx.i;
	float u = (float)idx.k / period[sample] + phase[sample];
	u -= floor(u); // position within the period, [0, 1)
	const int cls = (int)(sample % classes);
	float v = 0;
	if (cls == 0)      v = sin(two_pi * u);
	else if (cls == 1) v = (u < 0.5f) ? 1.f : -1.f;
	else if (cls == 2) v = 2.f * u - 1.f;
	else               v = 1.f - 4.f * fabs(u - 0.5f);
	return amplitude[sample] * v + noise;
}

// The class of a sample is its number modulo the number of classes
int sample_class(skepu::Index1D index, int classes)
{
	return (int)(index.i % classes);
}

// Fill x (n x 1 x SIGNAL_LENGTH x 1) and one-hot y (n x 1 x 1 x NUM_CLASSES). The random numbers come from the stream rng
// (skepu::randomize fills the containers by skeleton calls, so the data are the same with every backend and compiler), and the
// signals are made by a Map over the elements, in parallel.
static void generate(skepu::Tensor4<float> &x, skepu::Tensor4<float> &y, skepu::PRNG &rng, float noise)
{
	const size_t n = x.size_i();
	skepu::Vector<float> period(n, "signal periods"), phase(n, "signal phases"), amplitude(n, "signal amplitudes");

	skepu::randomize::Scope stream(rng);
	skepu::randomize::uniform(period, 8.f, 20.f);
	skepu::randomize::uniform(phase, 0.f, 1.f);
	skepu::randomize::uniform(amplitude, 0.7f, 1.3f);
	skepu::randomize::normal(x, 0.f, noise);                                 // the noise of every element
	skepu::Map<1>(waveform)(x, x, period, phase, amplitude, (int)NUM_CLASSES); // the waveform of the sample, plus the noise
	skepu::Vector<int> classes(n, "signal classes");
	skepu::Map<0>(sample_class)(classes, (int)NUM_CLASSES);
	skepu::dnn::one_hot(y, classes);
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
		skepu::io::cout << "Usage: " << argv[0] << " model(conv1d|mlp) batch_size learning_rate epochs backend [train_samples] [test_samples] [--adam]\n";
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

	skepu::PRNG rng(42); // the stream of the data
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

	skepu::io::cout << "Signals: " << n_train << " train / " << n_test << " test, length " << SIGNAL_LENGTH << ", classes:";
	for (auto name : class_names) skepu::io::cout << " " << name;
	skepu::io::cout << "\n";

	auto run = [&](auto model)
	{
		model.init();
		model.configure(Loss::CategoricalCrossEntropy, optimizer);
		skepu::io::cout << model.summary();
		model.fit(train_x, train_y, epochs, learning_rate, 0.1, "signals1d", true);

		skepu::Matrix<int> confusion(NUM_CLASSES, NUM_CLASSES, "Confusion Matrix", 0);
		auto score = model.evaluate(test_x, test_y, &confusion);
		skepu::io::cout << "Test loss: " << std::get<0>(score) << " Test accuracy: " << std::get<1>(score) << "\n";
		skepu::io::cout << "Confusion matrix (rows: actual, columns: predicted):\n" << confusion << "\n";
	};

	if (model_name == "conv1d") run(build_conv1d());
	else if (model_name == "mlp") run(build_mlp());
	else { skepu::io::cout << "Unknown model '" << model_name << "'\n"; return 1; }
	return 0;
}
