// Binary classification with the SkePU DNN library: the MNIST digits 0 and 1 (a single sigmoid output, binary cross-entropy).
//
// The data preparation is SkePU too: the images of the two classes are counted with a MapReduce (so that every process knows the
// count) and copied to new tensors in a skepu::external section; the binary labels are 0 for the first class and 1 for the second.
//
// Usage: dnn_binary batch_size learning_rate epochs backend run
//
// Sequential build (no precompiler):
//   clang++ -std=c++20 -O2 -I ../../skepu-headers/src -Wno-attributes dnn_binary.cpp -o dnn_binary
#include <iostream>
#include <string>
#include <tuple>

#define SKEPU_ENABLE_EXCEPTIONS

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/util.hpp>
#include <skepu-lib/ml.hpp>

#include "skepu-mnist.hpp"

// Is the label one of the two classes?
int in_the_two_classes(int label, int class_a, int class_b)
{
	return (label == class_a || label == class_b) ? 1 : 0;
}

// The images and labels of the two classes only, with the labels 0 (class_a) and 1 (class_b)
std::tuple<skepu::Tensor4<float>*, skepu::Tensor4<float>*>
binary_class(skepu::Tensor4<float> &images, skepu::Vector<mnist::label_t> &labels, int class_a, int class_b)
{
	auto count_classes = skepu::MapReduce<1>(in_the_two_classes, skepu::util::add<int>);
	const int count = static_cast<int>(count_classes(labels, class_a, class_b));

	auto *filtered_images = new skepu::Tensor4<float>(count, images.size_j(), images.size_k(), images.size_l(), "Filtered Images");
	auto *filtered_labels = new skepu::Tensor4<float>(count, 1, 1, 1, "Filtered Labels");

	skepu::external("Binary Classification Filtering", skepu::read(images, labels), [&]
	{
		int index = 0;
		for (size_t i = 0; i < labels.size_i(); i++)
		{
			if (labels(i) != class_a && labels(i) != class_b)
				continue;
			for (size_t h = 0; h < images.size_j(); h++)
				for (size_t w = 0; w < images.size_k(); w++)
					for (size_t c = 0; c < images.size_l(); c++)
						(*filtered_images)(index, h, w, c) = images(i, h, w, c);
			(*filtered_labels)(index, 0, 0, 0) = (labels(i) == class_a) ? 0 : 1;
			index++;
		}
	}, skepu::write(*filtered_images, *filtered_labels));

	return { filtered_images, filtered_labels };
}

int main(int argc, char *argv[])
{
	if (argc < 6)
	{
		skepu::io::cout << "Usage: " << argv[0] << " batch_size learning_rate epochs backend run\n";
		return 1;
	}

	const size_t batch_size = atoi(argv[1]);
	const float learning_rate = atof(argv[2]);
	const size_t epochs = atoi(argv[3]);
	auto spec = skepu::BackendSpec{argv[4]};
	spec.setCPUThreads(8);
	skepu::setGlobalBackendSpec(spec);

	using namespace skepu::dnn;
	auto model = SequentialModel(Dimensions{batch_size, 28, 28, 1})
		<< Conv2D(32).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Zero).activation(Act::ReLU)
		<< MaxPooling2D().kernel(2, 2)
		<< Conv2D(64).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Zero).activation(Act::ReLU)
		<< MaxPooling2D().kernel(2, 2)
		<< Conv2D(64).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Zero)
		<< Flatten()
		<< Dense(128).weights(Init::Xavier_uniform).biases(Init::Zero).activation(Act::ReLU)
		<< Dense(1).weights(Init::Xavier_uniform).biases(Init::Zero).activation(Act::Sigmoid);

	model.init();
	model.setBatchCountOverride(2); // a short run: two batches per epoch
	skepu::io::cout << model.summary();

	const std::string data_path = "../data/mnist/";
	skepu::Tensor4<float> train_images_all = mnist::parse_mnist_images(data_path + "train-images.idx3-ubyte", 0, 1.0);
	skepu::Tensor4<float> test_images_all  = mnist::parse_mnist_images(data_path + "t10k-images.idx3-ubyte", 0, 1.0);
	skepu::Vector<mnist::label_t> train_labels_all = mnist::parse_mnist_labels(data_path + "train-labels.idx1-ubyte");
	skepu::Vector<mnist::label_t> test_labels_all  = mnist::parse_mnist_labels(data_path + "t10k-labels.idx1-ubyte");

	const int class_a = 0, class_b = 1;
	skepu::Tensor4<float> *train_images, *train_labels, *test_images, *test_labels;
	std::tie(train_images, train_labels) = binary_class(train_images_all, train_labels_all, class_a, class_b);
	std::tie(test_images, test_labels)   = binary_class(test_images_all,  test_labels_all,  class_a, class_b);

	skepu::io::cout << "Train images dimensions: "
		<< train_images->size_i() << " x " << train_images->size_j() << " x "
		<< train_images->size_k() << " x " << train_images->size_l() << "\n";

	const std::string name = "mnist_bce_" + std::to_string(batch_size) + argv[4];
	model.configure(Loss::BinaryCrossEntropy, Opt::SGD);
	model.fit(*train_images, *train_labels, epochs, learning_rate, 0.1, name, true);

	skepu::Matrix<int> confusion_matrix(2, 2, "Confusion Matrix", 0);
	auto score = model.evaluate(*test_images, *test_labels, &confusion_matrix);
	skepu::io::cout << "Test loss: " << std::get<0>(score) << " Test accuracy: " << std::get<1>(score) << "\n";
	skepu::io::cout << "Confusion matrix (rows: actual, columns: predicted): " << confusion_matrix << "\n";
	return 0;
}
