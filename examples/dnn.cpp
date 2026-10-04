#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#define SKEPU_ENABLE_EXCEPTIONS
//#define DNN_DEBUG 1

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/ml.hpp>

#include <skepu-lib/dnn/csv_io.hpp>

#include "skepu-mnist.hpp"
//#include "skepu-cifar10.hpp"

skepu::Tensor4<float>
mnist_labels_to_onehot(skepu::Vector<mnist::label_t> &labels, size_t num_classes)
{
	skepu::Tensor4<float> result(labels.size(), 1, 1, num_classes, "MNIST-labels-onehot");
	auto onehot_mapper = skepu::Map<0>([](skepu::Index4D idx, skepu::Vec<int> labels) -> float
    {
  		int my_item  = idx.i;
  		int my_class = idx.l;
  		int my_label = labels(my_item);
  		return (my_label == my_class) ? 1 : 0;
    });
	onehot_mapper.setLabel("MNIST-to-onehot");
	onehot_mapper(result, labels);
	return result;
}

void convertCHWtoHWC(skepu::Tensor4<float> &images, skepu::Tensor4<float> &conv_images)
{
    skepu::external(skepu::read(images), [&]
    {
        for (size_t batch = 0; batch < images.size_i(); batch++)
            for (size_t channel = 0; channel < images.size_j(); channel++)
                for (size_t row = 0; row < images.size_k(); row++)
                    for (size_t col = 0; col < images.size_l(); col++)
                        conv_images(batch, row, col, channel) = images(batch, channel, row, col);
    }, skepu::write(conv_images));
}


std::tuple<skepu::Tensor4<float>*, skepu::Tensor4<float>*> 
binary_class(skepu::Tensor4<float> &train_images, skepu::Vector<mnist::label_t> &train_labels_temp, int num_classes, int class_a, int class_b)
{	
	// Count occurences of each class 
	std::vector<int> count_vector(num_classes);
	for (int i = 0; i < train_labels_temp.size_i(); i++)
        count_vector[train_labels_temp(i)]++;

	// Allocate memory
	int num_images_label = count_vector[class_a] + count_vector[class_b];
    
    skepu::Tensor4<float> *filtered_images = new skepu::Tensor4<float>(
        num_images_label,
        train_images.size_j(),
        train_images.size_k(),
        train_images.size_l(), "Filtered Images");
	skepu::Tensor4<float> *filtered_labels = new skepu::Tensor4<float>(
	    num_images_label, 1, 1, 1, "Filtered Labels");

	// Filter classes
	skepu::external("Binary Classification Filtering", skepu::read(train_images, train_labels_temp), [&]
	{
    	int index = 0;
        for (int i = 0; i < train_labels_temp.size_i(); i ++)
        {
            if (train_labels_temp(i) == class_a || train_labels_temp(i) == class_b)
            {
                for (int h = 0; h < train_images.size_j(); h++)
                    for (int w = 0; w < train_images.size_k(); w++)
                        for (int c  = 0; c < train_images.size_l(); c++)
                            (*filtered_images)(index, h, w, c) = train_images(i, h, w, c);
                (*filtered_labels)(index, 0, 0, 0) = (train_labels_temp(i) == class_a) ? 0 : 1;  
                index++;
            }   
        }
	}, skepu::write(*filtered_images, *filtered_labels));

	return { filtered_images, filtered_labels };
}


void csv_results(int batch_size, std::string backend, int run, double test_loss, double test_acc, double training_time, double testing_time, std::string name)
{ //, double preprocessing
	std::ofstream file;
	file.open("./result_final_1_7_Mar/" + name + "results_all.csv", std::ios::app);

	// If the file is empty, add a header line (optional)
	if (file.tellp() == 0)
	    file << "batch, backend, Run, Test_loss, Test_acc, Traning_time, testing_time \n";

	// Write the values to the file as a new line
	file << batch_size << ","
	     << backend << ","
	     << run << ","
         << test_loss << ","
         << test_acc << ","
	     << training_time<< ","
		 << testing_time<< "\n";
        //  "," 
        //  << preprocessing <<"\n";

	file.close();
}


int main(int argc, char *argv[])
{
	if (argc < 6)
	{
		std::cout << "Usage: " << argv[0] << " batch_size learning_rate epochs backend run \n";
		exit(1);
	}

	const size_t batch_size = atoi(argv[1]);
	const float learning_rate = atof(argv[2]);
	const size_t epochs = atof(argv[3]);
	auto spec = skepu::BackendSpec{argv[4]};
	spec.setCPUThreads(8);
	skepu::setGlobalBackendSpec(spec);
	auto run = atoi(argv[5]);

	// Dataset configuration
	const std::string param_path = "./data/pretrained/";
	const size_t num_classes = 10;
	using namespace skepu::dnn;
	
	// Keras MNIST convolutional model -> simple example 
	auto modela = SequentialModel(Dimensions{batch_size, 28, 28, 1})
		<< Conv2D(32).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Zero)
				// .pretrained(param_path + "params_layer_0.csv")
				.activation(Act::ReLU)
		<< MaxPooling2D().kernel(2, 2)
		<< Conv2D(64).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Zero)
				// .pretrained(param_path + "params_layer_2.csv")
				.activation(Act::ReLU)
		<< MaxPooling2D().kernel(2, 2) 
		<< Flatten() 
		<< Dropout(0.5)
		<< Dense(num_classes).weights(Init::Xavier_uniform).biases(Init::Zero)
				// .pretrained(param_path + "params_layer_6.csv")
				.activation(Act::SoftMax);

	// Multi-layer perceptron
	auto modelc = SequentialModel(Dimensions{batch_size, 28, 28, 1})
		<< Flatten() << Dropout(0.5)
		<< Dense(100).weights(Init::Uniform).biases(Init::Zero)
		.pretrained(param_path + "fc_hidden_params_layer_2.csv")
		.activation(Act::ReLU)
		<< Dense(num_classes).weights(Init::Uniform).biases(Init::Zero)
		.pretrained(param_path + "fc_hidden_params_layer_3.csv")
		.activation(Act::SoftMax);

	// Simpler conv network
	auto modeld = SequentialModel(Dimensions{batch_size, 28, 28, 1})
		<< Conv2D(1).kernel(3, 3)
				.pretrained(param_path + "simple_params_layer_0.csv")
				.activation(Act::ReLU)
		<< MaxPooling2D().kernel(2, 2)
		<< Flatten() //<< Dropout(0.5)
		<< Dense(num_classes)
				.pretrained(param_path + "simple_params_layer_3.csv")
				.activation(Act::SoftMax);

	// Simpler conv network
	auto modele = SequentialModel(Dimensions{batch_size, 28, 28, 1})
	//	<< Conv2D(1).kernel(3, 3)
	//			.activation(Act::ReLU)
	//	<< MaxPooling2D().kernel(2, 2)
		<< Conv2D(12).kernel(3, 3)
				.activation(Act::ReLU)
		<< MaxPooling2D().kernel(2, 2)
		<< Flatten() << Dropout(0.5)
		<< Dense(num_classes)
				.activation(Act::SoftMax);

	// TF simple CIFAR dataset model 
	auto modelf = SequentialModel(Dimensions{batch_size, 32, 32, 3})
		<< Conv2D(32).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Zero)
				// .pretrained(param_path + "params_layer_0.csv")
				.activation(Act::ReLU)
		<< MaxPooling2D().kernel(2, 2)
		<< Conv2D(64).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Zero)
				// .pretrained(param_path + "params_layer_2.csv")
				.activation(Act::ReLU)
		<< MaxPooling2D().kernel(2, 2) 
		<< Conv2D(64).kernel(3,3).weights(Init::Xavier_uniform).biases(Init::Zero)
		<< Flatten() //<< Dropout(0.5)
		<< Dense(64).weights(Init::Xavier_uniform).biases(Init::Zero).activation(Act::ReLU)
		<< Dense(num_classes).weights(Init::Xavier_uniform).biases(Init::Zero)
				// .pretrained(param_path + "params_layer_6.csv")
				.activation(Act::SoftMax);

    // Keras MNIST dataset model for binary
	auto model = SequentialModel(Dimensions{batch_size, 28,28,1})
    << Conv2D(32).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Zero)
            // .pretrained(param_path + "params_layer_0.csv")
            .activation(Act::ReLU)
    << MaxPooling2D().kernel(2, 2)
    << Conv2D(64).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Zero)
            // .pretrained(param_path + "params_layer_2.csv")
            .activation(Act::ReLU)
    << MaxPooling2D().kernel(2, 2) 
    << Conv2D(64).kernel(3,3).weights(Init::Xavier_uniform).biases(Init::Zero)
    << Flatten() //<< Dropout(0.5)
    << Dense(128).weights(Init::Xavier_uniform).biases(Init::Zero).activation(Act::ReLU)
    << Dense(1).weights(Init::Xavier_uniform).biases(Init::Zero)
            // .pretrained(param_path + "params_layer_6.csv")
            .activation(Act::Sigmoid);            
	
	
	// Initialize model
	model.init(); // memory and such
	model.setBatchCountOverride(2);
	std::cout << model.summary();

    /*********************** MINST loading ************************** */ 

    const std::string data_path = "./data/mnist/";
    
    skepu::Tensor4<float> train_images_temp = mnist::parse_mnist_images(data_path + "train-images.idx3-ubyte", 0, 1.0);
    skepu::Tensor4<float> test_images_temp  = mnist::parse_mnist_images(data_path + "t10k-images.idx3-ubyte", 0, 1.0);
    skepu::Vector<mnist::label_t> train_labels_temp = mnist::parse_mnist_labels(data_path + "train-labels.idx1-ubyte");
    skepu::Vector<mnist::label_t> test_labels_temp  = mnist::parse_mnist_labels(data_path + "t10k-labels.idx1-ubyte");

    // Multi classification
    // skepu::Tensor4<float> train_labels = mnist_labels_to_onehot(train_labels_temp, num_classes);
    // skepu::Tensor4<float> test_labels  = mnist_labels_to_onehot(test_labels_temp,  num_classes);
    
    // Binary classification 
	int class_a = 0, class_b = 1;
    skepu::Tensor4<float> *train_images, *train_labels, *test_images, *test_labels;
    std::tie(train_images, train_labels) = binary_class(train_images_temp, train_labels_temp, num_classes, class_a, class_b);
    std::tie(test_images, test_labels)   = binary_class(test_images_temp,  test_labels_temp,  num_classes, class_a, class_b);

	std::cout << "Train images dimensions: "
	    << train_images->size_i() << " x "
		<< train_images->size_j() << " x "
		<< train_images->size_k() << " x "
		<< train_images->size_l() << "\n";

	// model.display_image(train_images);
	// for(int i = 0; i < 5; i++){ //train_labels.size_i()
    //     std::cout << train_labels(i,0,0,0) << "\n";
    // }


	/*********************** CIFAR loading ************************** */
//	skepu::Vector<int> *train_labels_temp, test_labels_temp;
//  skepu::Tensor4<float> *train_Images_temp, test_images_temp;
	/*const std::string data_path = "/Users/sehqu50/Documents/PHD/Dataset/cifar-10-batches-bin";	
	std::tie(train_labels_temp, train_Images_temp) = cifar10::load_CIFAR10_Train_Dataset(data_path);
    std::tie(test_labels_temp,  test_Images_temp)  = cifar10::load_CIFAR10_Test_Dataset(data_path);

	skepu::Tensor4<float> train_labels = mnist_labels_to_onehot(*train_labels_temp, num_classes);
	skepu::Tensor4<float> test_labels  = mnist_labels_to_onehot(*test_labels_temp,  num_classes);
    
	// std::cout << "Train_Images_org dimensions: " <<train_Images_temp->size_i() << train_Images_temp->size_j() << train_Images_temp->size_k() << train_Images_temp->size_l() << "\n";
	// std::cout << "Train_Images_org dimensions: " <<test_Images_temp->size_i() << test_Images_temp->size_j() << test_Images_temp->size_k() << test_Images_temp->size_l() << "\n";
*/
	
    /*********************** Convert cifar data set to NHWC format  ************************** */ 
	/*
    skepu::Tensor4<float> c_train_images(train_Images_temp->size_i(),train_Images_temp->size_k(), train_Images_temp->size_l(),train_Images_temp->size_j());
	skepu::Tensor4<float> c_test_images(test_Images_temp->size_i(),test_Images_temp->size_k(), test_Images_temp->size_l(),test_Images_temp->size_j());
	// std::cout << "Train_Images_converted diemnsions: " <<train_images.size_i() << train_images.size_j() << train_images.size_k() << train_images.size_l() << "\n";
	// std::cout << "Test_Images_converted diemnsions: " <<test_images.size_i() << test_images.size_j() << test_images.size_k() << test_images.size_l() << "\n";
	// std::cout << "probllem here \n";

	convertCHWtoHWC(*train_Images_temp,c_train_images);
	convertCHWtoHWC(*test_Images_temp, c_test_images);
*/
    /*********************** Preprocessing cifar data set RGB to Gray scale  ************************** */ 
    /*
    skepu::Tensor4<float> train_images(c_train_images.size_i(),c_train_images.size_j(), c_train_images.size_k(),  1 );
    skepu::Tensor4<float> test_images(c_test_images.size_i(),c_test_images.size_j(), c_test_images.size_k(),  1 );

    // std::cout << "Train_Images_converted dimensions: " <<c_train_images.size_i() << c_train_images.size_j() << c_train_images.size_k() << c_train_images.size_l() << "\n";
	// std::cout << "Test_Images_converted dimensions: " <<c_test_images.size_i() << c_test_images.size_j() << c_test_images.size_k() << c_test_images.size_l() << "\n";
    auto convert_pool = skepu::MapPool([](skepu::Pool4D<float> pool)
    {
        float pr = pool(0,0,0,0);
        float pg = pool(0,0,0,1);
        float pb = pool(0,0,0,2);
        return 0.2989f * pr + 0.5870f * pg + 0.1140f * pb;
    });
    convert_pool.setPoolSize(1,1,1,3);

	auto timeTaken = skepu::benchmark::measureExecTime([&]
	{
		convert_pool(train_images, c_train_images);//, imageInfo.elementsPerPixel);
        convert_pool(test_images, c_test_images);
	});
	// WritePngFileMatrix(outputMatrix, outputFile + "-average.png", colorType, imageInfo);
	std::cout << "Time for pre processing : " << (timeTaken.count() / 10E6) << "\n";*/
    



/* 	train_labels = cut_tensor4(train_labels, train_labels.size_i());
	train_images = cut_tensor4(train_images, train_images.size_i());	
	test_labels = cut_tensor4(test_labels, test_labels.size_i());
	test_images = cut_tensor4(test_images, test_images.size_i());*/


	/*********************** name of the file   ************************** */     	
	std::string name = "mnist_bce_" + std::to_string(batch_size) + argv[4];
	 
	/*********************** Training   ************************** */     
	model.configure(Loss::BinaryCrossEntropy, Opt::SGD);
	auto trainingtimeTaken = skepu::benchmark::measureExecTime([&]
	{
	    model.fit(*train_images, *train_labels, epochs, learning_rate, 0.1, name, true); //true is logging ->progress bar
    });
	
	/*********************** Testing   ************************** */ 
    double test_loss = 0, test_acc = 0;    
	skepu::Matrix<int> confusion_matrix(2, 2, "Confusion Matrix", 0); // binary classification
	auto testingtimeTaken = skepu::benchmark::measureExecTime([&]
	{   
		auto score = model.evaluate(*test_images, *test_labels, &confusion_matrix);
	    skepu::io::cout
			<< "Test loss: " << std::get<0>(score)
			<< " Test accuracy: " << std::get<1>(score) << "\n";
    });   

    // std::cout << "TRAIN_TIME: : " << (trainingtimeTaken.count() / 10E6) << "\n"; 
	// std::cout << "TEST_TIME: " << (testingtimeTaken.count() / 10E6) << "\n";
	
	// std::cout << "TRAIN_TIME: " << train_time << " sec" << std::endl;
    // std::cout << "TEST_TIME: " << test_time << " sec" << std::endl;

	skepu::io::cout << "Confuision Matrix: " << confusion_matrix << "\n";

	/*********************** Writing to the csv file   ************************** */     
	// csv_results (batch_size, argv[4], run, test_loss, test_acc, (trainingtimeTaken.count() / 10E6), (testingtimeTaken.count() / 10E6), (timeTaken.count() / 10E6), "cifar_gray_" );
    // csv_results (batch_size, argv[4], run, test_loss, test_acc, (trainingtimeTaken.count() / 10E6), (testingtimeTaken.count() / 10E6), "mnist_loop_10" );
	
    /*********************** Serialize weights to file   ************************** */     
	// model.serialize("skepu_dnn_trained_model_weights"); // generate parameter csv files
	
}
