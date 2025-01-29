#pragma once
#include <skepu>
#ifndef SKEPU_PRECOMPILED

static skepu::PrecompilerMarker startOf_SequentialModel_HPP;

#include "layer.hpp"
#include "conv1d.hpp"
#include "conv2d.hpp"
#include "conv1d.hpp"


template <typename T = float>
class SequentialModel
{
public:
	SequentialModel(Dimensions arg_input_size)
	: m_input_size{arg_input_size}, m_batch_size{std::get<0>(arg_input_size)} {}

	SequentialModel(size_t arg_input_size, size_t arg_batch_size)
	: m_input_size{arg_input_size, 1, 1, 1}, m_batch_size{arg_batch_size} {}

	// move constructor
	SequentialModel(std::vector<std::shared_ptr<Layer<T>>> &&arg_layers)
	: m_layers{arg_layers}
	{
		this->init();
	}

	void init()
	{
		if (this->m_initialized)
			SKEPU_ERROR("SkePU-DNN: Attempting to initialize an already initialized SequentialModel");

		// Connect layers together in internal doubly-linked list
		std::shared_ptr<Layer<T>> *prev = nullptr;
		for (auto &current : this->m_layers)
		{
			current->predecessor = prev;
			if (prev)
				prev->get()->successor = &current;
		}

		this->m_layers[0]->m_layer_level = Level::Input;
		this->m_layers[this->m_layers.size() - 1]->m_layer_level = Level::Output;

		// Allocate data
		tracing::tracer().beginRegion("Model data allocation", __LINE__);
		Dimensions input_size = this->m_input_size;
		for (auto &current : this->m_layers)
		{
			//	dnn_debug << "Input size: " << std::get<0>(input_size) << ", " << std::get<1>(input_size) << ", " << std::get<2>(input_size) << ", " << std::get<3>(input_size) << ", " << "\n";
			//dnn_debug << current->summary() << "\n";
			current->setup(input_size); // on each layer
			input_size = current->m_output_size;
		}
		tracing::tracer().endRegion();

		this->m_initialized = true;
	}

	template <typename ConcreteLayer>
	void add(ConcreteLayer &&l)
	{
		Act activationFunc = l.m_act;
		l.m_batch_size = this->m_batch_size;
		l.m_layer_id = ++ this->m_layers_count;//this->m_layers.size();
		l.m_prng = &this->m_prng;
		this->m_layers.push_back(std::make_shared<typename std::remove_reference<ConcreteLayer>::type>(std::forward<ConcreteLayer>(l)));

		if (activationFunc != Act::None)
		{
			auto act_layer = std::make_shared<Activation<T>>(activationFunc);
			act_layer->m_batch_size = this->m_batch_size;
			act_layer->m_layer_id = this->m_layers_count; //this->m_layers.size(); // or are activations not their own layer?
			this->m_layers.push_back(act_layer);
		}
	}

	void init_parameters()
	{
		tracing::tracer().region("Model param init", __LINE__, [&]
		{
			for (auto &current : this->m_layers)
				current->init_parameters();
		});
	}

	void serialize(std::string base_file_name)
	{
		tracing::tracer().region("Serialize model parameters", __LINE__, [&]
		{
			for (auto &current : this->m_layers)
				current->serialize(base_file_name);
		});
	}

	Tensor4<T> &forward(Tensor4<T> &inputs, bool is_training=false)
	{
		if (!this->m_initialized)
			SKEPU_ERROR("SkePU-DNN: Attempting to use (forward pass) a not-initialized SequentialModel")

		Tensor4<T> *temp = &inputs;
		int i = 1;
		for (auto &current : this->m_layers)
		{
			dnn_debug << "Layer number " << i++ << ": " << current->summary() << "\n";
	//  auto time_good = skepu::benchmark::measureExecTime([&]() {
			temp = current->forward(temp, is_training);
	//	});
	//	dnn_debug << "Time " << time_good.count() / 1E6  << "\n\n";
		}
		this->m_result = temp;
		dnn_debug << "Final result: " << *this->m_result << "\n";
		return *temp;
	}

	std::tuple<float, float> evaluate(Tensor4<T> &test_x, Tensor4<T> &test_y, skepu::Matrix<int> *confusion_matrix = nullptr)
	{
		if (!this->m_initialized)
			this->init();

		Tensor4<T> batched_test_x(
				this->m_batch_size,
				std::get<1>(this->m_input_size),
				std::get<2>(this->m_input_size),
				std::get<3>(this->m_input_size),
				"Batch eval data X"
		);
		Tensor4<T> batched_test_y(this->m_batch_size, 1, 1, test_y.size_l(), "Batch eval data Y");

		skepu::Vector<float> temp_loss(this->m_batch_size, "Loss temp");
		skepu::Vector<int>   temp_acc(this->m_batch_size, "Accuracy temp");

		float total_loss = 0, total_accuracy =0 ;
		const size_t mini_batch_count = test_x.size_i() / this->m_batch_size;
		for (size_t mini_batch = 0; mini_batch < mini_batch_count; ++mini_batch) //mini_batch_count
		{
			skel_flatmap.setLabel("Copy batch eval data X");
			skel_flatmap(batched_test_x, test_x.begin() + mini_batch * batched_test_x.size());
			skel_flatmap.setLabel("Copy batch eval data Y");
			skel_flatmap(batched_test_y, test_y.begin() + mini_batch * batched_test_y.size());

			auto &predicted = this->forward(batched_test_x);

			// Loss and accuracy
		/*  skel_cross_entropy_batched_loss.setDefaultSize(this->m_batch_size);
			loss += -skel_cross_entropy_batched_loss(batched_test_y, predicted) / this->m_batch_size;
			skel_accuracy_batched.setDefaultSize(this->m_batch_size);
			accuracy += (float)skel_accuracy_batched(batched_test_y, predicted) / this->m_batch_size;*/

			// Map+Reduce variant
			skel_cross_entropy_batched_loss_map.setLabel("Crossentropy A");
			skel_cross_entropy_batched_loss_reduce.setLabel("Crossentropy B");
			skel_accuracy_batched_map.setLabel("Accuracy A");
			skel_accuracy_batched_reduce.setLabel("Accuracy B");
			skel_cross_entropy_batched_loss_map(temp_loss, batched_test_y, predicted);
			skel_accuracy_batched_map(temp_acc, batched_test_y, predicted);
			float loss = -skel_cross_entropy_batched_loss_reduce(temp_loss) / this->m_batch_size;
			float accuracy = (float)skel_accuracy_batched_reduce(temp_acc) / this->m_batch_size;

			total_loss += loss;
			total_accuracy += accuracy;

			std::cout << "Valid Batch " << mini_batch + 1 << " - Loss: " << loss << " - Accuracy: " << accuracy << std::endl;

			if (confusion_matrix)
			{
				for (size_t i = 0; i < batched_test_y.size_i(); ++i) { // rows
					int actual_label = 0; //true
					int predicted_label = 0; // predicted
					float max_actual = 0, max_predicted = 0;
					for (size_t j = 0; j < batched_test_y.size_l(); ++j) { // cols
						if (batched_test_y(i,0,0, j) > max_actual)
						{
							max_actual = batched_test_y(i, 0, 0, j);
							actual_label = j;
						}
						if (predicted(i,0,0, j) > max_predicted)
						{
							max_predicted = predicted(i, 0, 0, j);
							predicted_label = j;
						}
					}
          (*confusion_matrix)(actual_label, predicted_label) += 1;
        } // end of j loop
			} // end of if


		}

		return {total_loss / mini_batch_count, total_accuracy / mini_batch_count}; // correct?
	}

	void display_image(skepu::Tensor4<float> &train_images){

	  int batch = train_images.size_i();
	  int height = train_images.size_j();
	  int width = train_images.size_k();
	  int channel = train_images.size_l();

	  for(size_t b =0; b < 5; ++b){
	    for(size_t i =0; i < height; ++i){
	      for(size_t j =0; j < width; ++j){
	        for(size_t c =0; c < channel; ++c){

	          if (train_images(b,i,j,c) > 0.75) {
	               std::cout << "#"; // Dense pixel
	           } else if (train_images(b,i,j,c) > 0.5) {
	               std::cout << "+"; // Medium pixel
	           } else if(train_images(b,i,j,c) >0.25){
	             std::cout << "." ;
	           } else {
	               std::cout << " "; // Sparse pixel
	           }
	        }
	      }
	      std::cout << '\n';
	    }
	    std::cout << '\n';
	  }
	}



	void fit(Tensor4<T> &train_x, Tensor4<T> &train_y, int epochs, float learning_rate, float validation_split,  bool print_progress=true)
	{
		if (!this->m_initialized)
			this->init();

		// (Re-)initialize layer trainable parameters
		this->init_parameters();

		size_t valid_size = train_x.size_i() * validation_split;
		size_t train_size = train_x.size_i() - valid_size ;
		std::cout << "total train_x: " << train_x.size_i() << '\n';
		std::cout << "valid_size: " <<valid_size <<" train_size: " <<train_size << '\n';

		Tensor4<T> train_split_x(
				train_size,
				std::get<1>(this->m_input_size),
				std::get<2>(this->m_input_size),
				std::get<3>(this->m_input_size),
				"Split train data X"
		);
		Tensor4<T> train_split_y(train_size, 1, 1, train_y.size_l(), "Split train data Y");

		Tensor4<T> valid_split_x(
				valid_size,
				std::get<1>(this->m_input_size),
				std::get<2>(this->m_input_size),
				std::get<3>(this->m_input_size),
				"Split valid data X"
		);
		Tensor4<T> valid_split_y(valid_size, 1, 1, train_y.size_l(), "Split valid data Y");

		skel_flatmap.setLabel("Copy Split training data X");
		skel_flatmap(train_split_x, train_x.begin());
		skel_flatmap.setLabel("Copy Split training data Y");
		skel_flatmap(train_split_y, train_y.begin());

		// std::cout << "train_split_x: " <<train_split_x.size_i()<< " train_split_x " << '\n';
		// display_image(train_split_x);

		skel_flatmap.setLabel("Copy split valid data X");
		skel_flatmap(valid_split_x, train_x.begin() + train_size * train_x.size_j() * train_x.size_k() * train_x.size_l());
		skel_flatmap.setLabel("Copy split valid data Y");
		skel_flatmap(valid_split_y, train_y.begin() + train_size * train_y.size_j() * train_y.size_k() * train_y.size_l());

		// std::cout << "valid_split_x: " <<valid_split_x.size_i()<< " valid_split_x " << '\n';
		// display_image(valid_split_x);

		Tensor4<T> batched_train_x(
				this->m_batch_size,
				std::get<1>(this->m_input_size),
				std::get<2>(this->m_input_size),
				std::get<3>(this->m_input_size),
				"Batch train data X"
		);
		Tensor4<T> batched_train_y(this->m_batch_size, 1, 1, train_y.size_l(), "Batch train data Y");


		skepu::Vector<float> temp_loss(this->m_batch_size, "Loss temp");
		skepu::Vector<int>   temp_acc(this->m_batch_size, "Accuracy temp");

		for (size_t epoch = 0; epoch < epochs; ++epoch) // epoch
		{
			float total_train_loss = 0, total_train_accuracy = 0;
			// std::stringstream ss; ss << "Epoch " << (epoch + 1);
			// tracing::tracer().region("Epoch", __LINE__, [&]()
			// {
			// dnn_debug << "Epoch " << (epoch + 1) << "\n";

			const size_t mini_batch_count = train_split_x.size_i() / this->m_batch_size;
			std::cout << "batches: " <<mini_batch_count << '\n';

			for (size_t mini_batch = 0; mini_batch < mini_batch_count; ++mini_batch) // mini batches, mini_batch_count
			{
		// 		/************************ prepare batch - copying **************************/
				// dnn_debug << "\nMini-batch " << (mini_batch + 1) << " / " << mini_batch_count << "\n";
				skel_flatmap.setLabel("Copy batch training data X");
				skel_flatmap(batched_train_x, train_split_x.begin() + mini_batch * batched_train_x.size());
				skel_flatmap.setLabel("Copy batch training data Y");
				skel_flatmap(batched_train_y, train_split_y.begin() + mini_batch * batched_train_y.size());

				// std::cout << "batched_train_x: " <<batched_train_x.size_i()<< " batched_train_x  " << '\n';
				 // display_image(batched_train_x);

		// 		// skepu::io::cout << batched_train_x << '\n';
		// 		/************************ forward **************************/
		// 		std::stringstream ss; ss << "Epoch " << (epoch + 1) << " batch " << (mini_batch + 1) << " forward";
		// 		tracing::tracer().beginRegion("Forward", __LINE__);
		 			auto &predicted = this->forward(batched_train_x, true);
		// 		tracing::tracer().endRegion();
		//

				// std::cout << "predicted : " << predicted << '\n';

				Tensor4<T> initial_gradient( // move out of loop
						predicted.size_i(), predicted.size_j(),
						predicted.size_k(), predicted.size_l(),
						"Initial gradient"
				);

				skel_delta.setLabel("Loss gradient");
				skel_delta(initial_gradient, predicted, batched_train_y);
				Tensor4<T> *gradient = &initial_gradient;
				// dnn_debug << "Initial gradient: " << *gradient << "\n";
		//
		// 		// accuracy
				float loss = 0, accuracy = 0;

				// skel_cross_entropy_batched_loss_map.setLabel("Crossentropy A");
				// skel_cross_entropy_batched_loss_reduce.setLabel("Crossentropy B");
				// skel_accuracy_batched_map.setLabel("Accuracy A");
				// skel_accuracy_batched_reduce.setLabel("Accuracy B");

		// 		/// check
				skel_cross_entropy_batched_loss_map(temp_loss, batched_train_y, predicted);
				skel_accuracy_batched_map(temp_acc, batched_train_y, predicted);
				loss += -skel_cross_entropy_batched_loss_reduce(temp_loss) / this->m_batch_size;
				accuracy += (float)skel_accuracy_batched_reduce(temp_acc) / this->m_batch_size;

				total_train_loss +=loss;
				total_train_accuracy +=accuracy;

				std::cout << "Train Batch " << mini_batch + 1 << " - Loss: " << loss << " - Accuracy: " << accuracy << std::endl;

		// 		std::cout << "-- Mini-batch " << std::setw(6) << (mini_batch + 1) << " / " << std::setw(6) << mini_batch_count;
		// 		std::cout << " [";
		// 		const size_t bar_width = 30;
		// 		for (size_t i = 0; i < bar_width; ++i)
		// 		{
		// 			if (mini_batch / (float)mini_batch_count < i / (float)bar_width) std::cout << " ";
		// 			else std::cout << "=";
		// 		}
		// 		std::cout << "]";
		// 		std::cout	<< " - Loss: " << std::setw(10) << loss << ", Accuracy: " << std::setw(10) << accuracy << "\r";
		//
		//
		// 		/************** back ward pass ***************/
		// 		std::stringstream ss2; ss2 << "Epoch " << (epoch + 1) << " batch " << (mini_batch + 1) << " backward";
		// 		tracing::tracer().beginRegion("Backward", __LINE__);
				for (auto it = this->m_layers.rbegin(); it != this->m_layers.rend(); ++it)
				{
					auto &current_layer = **it;
					Tensor4<T> *forward_inputs = ((it + 1) != this->m_layers.rend()) ? &(**(it + 1)).m_vals : &batched_train_x;

					if (current_layer.is_trainable()) current_layer.m_learning_rate = learning_rate;

					gradient = current_layer.backward(forward_inputs, gradient, &batched_train_y);
		// 			dnn_debug << "Outgoing gradient: " << *gradient << "\n";
				}
		// 		tracing::tracer().endRegion();

		 	} // end of mini batch
		//
		// 	if (print_progress)
		// 	{
		// 		std::stringstream ss; ss << "Epoch " << (epoch + 1) << " evaluate";
		// 		tracing::tracer().beginRegion("Epoch evaluate", __LINE__);
				auto score = this->evaluate(valid_split_x, valid_split_y);

		// 		tracing::tracer().endRegion(); // \033[K
		// 		std::cout << "- Epoch " << std::setw(4) << (epoch + 1) << " / " << std::setw(4) << epochs << ":";
		// 		std::cout << " Valid Loss: " << std::setw(10) << std::get<0>(score) << ", Valid Accuracy: " << std::setw(10) << std::get<1>(score) << "\n";
		// 	}
		// }); // tracer epoch end


		 } // end of epoch loop


	} // end of fit function

	std::string summary()
	{
		std::stringstream os;
		for (auto &layer : this->m_layers)
		{
			os << layer->summary() << "\n";
		}
		return os.str();
	}

	Layer<T> &getLayer(size_t i) { return *this->m_layers[i]; }

protected:
	bool m_initialized = false;

	std::vector<std::shared_ptr<Layer<T>>> m_layers{};
	size_t m_layers_count = 0;
	skepu::PRNG m_prng{0}; // default seed

	Dimensions m_input_size;
	size_t m_batch_size;

	Tensor4<T> *m_result = nullptr;
};

template <typename T, typename ConcreteLayer>
SequentialModel<T> &&operator<<(SequentialModel<T> &&model, ConcreteLayer &&layer)
{
	model.add(std::forward<ConcreteLayer>(layer));
	return std::move(model);
}

template <typename T, typename ConcreteLayer>
SequentialModel<T> &operator<<(SequentialModel<T> &model, ConcreteLayer &&layer)
{
	model.add(std::forward<ConcreteLayer>(layer));
	return model;
}
} // namespace impl

template <typename T = float, typename... Args>
impl::Dense<T> Dense(Args &&...args)
{
	return impl::Dense<T>(std::forward<Args>(args)...);
}

template <typename T = float, typename... Args>
impl::Conv1D<T> Conv1D(Args &&...args)
{
	return impl::Conv1D<T>(std::forward<Args>(args)...);
}

template <typename T = float, typename... Args>
impl::Conv2D<T> Conv2D(Args &&...args)
{
	return impl::Conv2D<T>(std::forward<Args>(args)...);
}

template <typename T = float, typename... Args>
impl::MaxPooling2D<T> MaxPooling2D(Args &&...args)
{
	return impl::MaxPooling2D<T>(std::forward<Args>(args)...);
}

template <typename T = float, typename... Args>
impl::Flatten<T> Flatten(Args &&...args)
{
	return impl::Flatten<T>(std::forward<Args>(args)...);
}

template <typename T = float, typename... Args>
impl::Dropout<T> Dropout(Args &&...args)
{
	return impl::Dropout<T>(std::forward<Args>(args)...);
}

template <typename T = float, typename... Args>
impl::SequentialModel<T> SequentialModel(Args &&...args)
{
	return impl::SequentialModel<T>(std::forward<Args>(args)...);
}


static skepu::PrecompilerMarker endOf_SequentialModel_HPP;