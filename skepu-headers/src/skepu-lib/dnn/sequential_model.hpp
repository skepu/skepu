#pragma once
#include <skepu>
#ifndef SKEPU_PRECOMPILED
static skepu::PrecompilerMarker startOf_SequentialModel_HPP;
#include <map>



namespace skepu::dnn::impl
{
// Parameter optimizers (SGD, Adam), applied per trainable layer right after its backward pass.
// Adam keeps first/second moment estimates for every parameter, allocated lazily per layer.
template<typename T = float>
class Optimizer
{
public:
	Optimizer(Opt opt_type, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
	: m_optimizer_type(opt_type), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon) {}

	// Forget all optimizer state (e.g. when the parameters are re-initialized)
	void reset()
	{
		this->m_dense_state.clear();
		this->m_conv_state.clear();
	}

	void step(Layer<T>& layer)
	{
		if (auto* conv_layer = dynamic_cast<Conv2D<T>*>(&layer))
		{
			// Also covers Conv1D, which is a Conv2D with a (1 x K) kernel
			this->update(layer, "Conv2D", layer.get_weights(), layer.get_gradients(), this->m_conv_state);
		}
		else if (auto* dense_layer = dynamic_cast<Dense<T>*>(&layer))
		{
			this->update(layer, "Dense", layer.get_d_weights(), layer.get_d_gradients(), this->m_dense_state);
		}
	}

private:
	// Adam moment estimates and step counter of one layer
	template <typename Weights>
	struct AdamState
	{
		Weights m_w, v_w;
		Vector<T> m_b, v_b;
		size_t t = 0;
	};

	static Matrix<T> make_like(const Matrix<T> &w) { return Matrix<T>(w.size_i(), w.size_j(), "Adam moment"); }
	static Tensor4<T> make_like(const Tensor4<T> &w) { return Tensor4<T>(w.size_i(), w.size_j(), w.size_k(), w.size_l(), "Adam moment"); }

	template <typename Weights>
	AdamState<Weights> &state_for(std::map<const Layer<T>*, AdamState<Weights>> &states, const Layer<T> &layer, const Weights &w, const Vector<T> &b)
	{
		auto it = states.find(&layer);
		if (it == states.end())
		{
			it = states.emplace(&layer, AdamState<Weights>{make_like(w), make_like(w), Vector<T>(b.size(), "Adam moment"), Vector<T>(b.size(), "Adam moment"), 0}).first;
			skel_init_zero(it->second.m_w);
			skel_init_zero(it->second.v_w);
			skel_init_zero(it->second.m_b);
			skel_init_zero(it->second.v_b);
		}
		return it->second;
	}

	template <typename Weights>
	void update(Layer<T>& layer, const std::string &kind, Weights &w, Weights &dw, std::map<const Layer<T>*, AdamState<Weights>> &states)
	{
		Vector<T>& b = layer.get_biases();
		Vector<T>& db = layer.get_biases_gradients();
		const float lr = layer.m_learning_rate;

		if (this->m_optimizer_type == Opt::SGD)
		{
			skel_update_params.setLabel(layer.label() + " " + kind + " Update Weights");
			skel_update_params(w, w, dw, lr);
			skel_update_params.setLabel(layer.label() + " " + kind + " Update Biases");
			skel_update_params(b, b, db, lr);
		}
		else if (this->m_optimizer_type == Opt::Adam)
		{
			AdamState<Weights> &st = this->state_for(states, layer, w, b);
			st.t++;
			// Bias correction of the moment estimates
			const float corr1 = 1.f - std::pow(this->m_beta1, (float)st.t);
			const float corr2 = 1.f - std::pow(this->m_beta2, (float)st.t);

			skel_adam_update.setLabel(layer.label() + " " + kind + " Adam Weights");
			skel_adam_update(w, st.m_w, st.v_w, w, dw, st.m_w, st.v_w, lr, this->m_beta1, this->m_beta2, this->m_epsilon, corr1, corr2);
			skel_adam_update.setLabel(layer.label() + " " + kind + " Adam Biases");
			skel_adam_update(b, st.m_b, st.v_b, b, db, st.m_b, st.v_b, lr, this->m_beta1, this->m_beta2, this->m_epsilon, corr1, corr2);
		}
	}

	Opt m_optimizer_type;
	float m_beta1, m_beta2, m_epsilon;
	std::map<const Layer<T>*, AdamState<Matrix<T>>> m_dense_state;
	std::map<const Layer<T>*, AdamState<Tensor4<T>>> m_conv_state;
};



template <typename T = float>
class LossFunction
{
public:
   	LossFunction(Loss type): m_loss_type(type) {}

	// Per-sample (negated) log-likelihood terms: out(i) = sum_c y_c log(p_c) for the batch item i.
	// The loss of an item is -out(i).
	void compute_log_likelihoods(Tensor4<T>& batched_train_y, Tensor4<T>& predicted, skepu::Vector<float>& out)
	{
		if (this->m_loss_type == Loss::CategoricalCrossEntropy)
			skel_cross_entropy_batched_loss_map(out, batched_train_y, predicted);
		else if (this->m_loss_type == Loss::BinaryCrossEntropy)
			skel_binary_cross_entropy_loss_map(out, batched_train_y, predicted);
	}

   	// forward pass 
   	skepu::Scalar<float> compute_loss(Tensor4<T>& batched_train_y, Tensor4<T>& predicted, size_t batch_size)
    {
  		skepu::Vector<float> temp_loss(batch_size, "Loss temp");
  		skepu::Scalar<float> computed_loss = 0;

  		if (this->m_loss_type == Loss::CategoricalCrossEntropy)
  		{
 			skel_cross_entropy_batched_loss_map(temp_loss, batched_train_y, predicted);	
	 		computed_loss += -skel_cross_entropy_batched_loss_reduce(temp_loss) / (float)batch_size;
  		}
  		else if (this->m_loss_type == Loss::BinaryCrossEntropy)
  		{
 			skel_binary_cross_entropy_loss_map(temp_loss, batched_train_y, predicted);
  		    computed_loss += -skel_cross_entropy_batched_loss_reduce(temp_loss) / (float)batch_size;	
  		}
  		return computed_loss;
   	}

	// backward pass
	Tensor4<T> compute_gradient(Tensor4<T>& predicted, Tensor4<T>& batched_train_y)
	{
		Tensor4<T> initial_gradient( // move out of loop
			predicted.size_i(), predicted.size_j(),
			predicted.size_k(), predicted.size_l(),
			"Initial gradient"
		);

		if (this->m_loss_type == Loss::CategoricalCrossEntropy)
		{
			skel_delta_cce(initial_gradient, predicted, batched_train_y);
		}
		else if (this->m_loss_type == Loss::BinaryCrossEntropy)
		{
			skel_delta_bce(initial_gradient, predicted, batched_train_y);			
		}
		return initial_gradient;	
	}

private:
    Loss m_loss_type; 
};



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
	    SKEPU_TRACE_SCOPE("Model Init");
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
		Dimensions input_size = this->m_input_size;
		for (auto &current : this->m_layers)
		{
			//	dnn_debug << "Input size: " << std::get<0>(input_size) << ", " << std::get<1>(input_size) << ", " << std::get<2>(input_size) << ", " << std::get<3>(input_size) << ", " << "\n";
			//dnn_debug << current->summary() << "\n";
			current->setup(input_size); // on each layer
			input_size = current->m_output_size;
		}

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
	    SKEPU_TRACE_SCOPE("Model param init");
		for (auto &current : this->m_layers)
			current->init_parameters();
	}

	void serialize(std::string base_file_name)
	{
		SKEPU_TRACE_SCOPE("Serialize model parameters");
		for (auto &current : this->m_layers)
			current->serialize(base_file_name);
	}

	Tensor4<T> &forward(Tensor4<T> &inputs, bool is_training=false)
	{
	    SKEPU_TRACE_SCOPE("Forward");
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

	std::tuple<skepu::Scalar<float>, skepu::Scalar<float>>
	evaluate(Tensor4<T> &test_x, Tensor4<T> &test_y, skepu::Matrix<int> *confusion_matrix = nullptr)
	{
        SKEPU_TRACE_SCOPE("Model Evaluate");
		if (!this->m_initialized)
			this->init();

		if (test_x.size_i() == 0)
			return { skepu::Scalar<float>(0.f), skepu::Scalar<float>(0.f) };
		if (test_x.size_i() < this->m_batch_size)
			SKEPU_ERROR("SkePU-DNN: evaluation data (" << test_x.size_i() << " samples) is smaller than the batch size (" << this->m_batch_size << ")");
		
		size_t num_valid_batches = test_x.size_i() / this->m_batch_size;
 		size_t remainder_valid_batch = test_x.size_i() % this->m_batch_size;
		if (remainder_valid_batch > 0 ) num_valid_batches = num_valid_batches + 1;
		
		Tensor4<T> batched_test_x(
			this->m_batch_size,
			std::get<1>(this->m_input_size),
			std::get<2>(this->m_input_size),
			std::get<3>(this->m_input_size),
			"Batch eval data X"
		);
		Tensor4<T> batched_test_y(this->m_batch_size, 1, 1, test_y.size_l(), "Batch eval data Y");

		size_t remainder_test_batch_allocate = remainder_valid_batch;
		if (remainder_test_batch_allocate == 0) remainder_test_batch_allocate = 1;

       	// skepu::Tensor4<float> r_batch_data(remainder_test_batch_allocate, test_x.size_j(),test_x.size_k(), test_x.size_l());
       	// skepu::Tensor4<float> r_batch_label(remainder_test_batch_allocate, test_y.size_j(),test_y.size_k(), test_y.size_l());

		skepu::Vector<float> temp_loss(this->m_batch_size, "Loss temp");
		skepu::Vector<int>   temp_acc(this->m_batch_size, "Accuracy temp");

		// Exact per-sample sums: the last batch overlaps the previous one when the data size is not a
		// multiple of the batch size, so its first (batch_size - remainder) items must not be counted twice
		double sum_loss = 0, sum_correct = 0;
		size_t counted_samples = 0;
		// const size_t mini_batch_count = test_x.size_i() / this->m_batch_size;

		// Tensor4<float> *actual_batched_test_x = nullptr;
		// Tensor4<float> *actual_batched_test_y = nullptr;

		const size_t total_valid_batches = num_valid_batches; // before any batch-count override
		if (this->m_override_batch_count >= 0)
			num_valid_batches = std::min(num_valid_batches, (size_t)this->m_override_batch_count);

		for (size_t mini_batch = 0; mini_batch < num_valid_batches; ++mini_batch)
		{
		    SKEPU_TRACE_SCOPE("Batch");
			if (mini_batch == (total_valid_batches - 1) && remainder_valid_batch > 0)
            {
                // skel_flatmap(r_batch_data, test_x.begin() + mini_batch * r_batch_data.size());
                // skel_flatmap(r_batch_label, test_y.begin() + mini_batch * r_batch_label.size());
				// std::cout << "error 4 "<< mini_batch <<" \n";
				skel_flatmap.setLabel("Copy batch test data X");
				skel_flatmap(batched_test_x, test_x.begin() + mini_batch * batched_test_x.size() - (this->m_batch_size - remainder_valid_batch) * batched_test_x.size_j() * batched_test_x.size_k() * batched_test_x.size_l()); // remainder + (remainder - batch_size) 
				skel_flatmap.setLabel("Copy batch test data Y");
				skel_flatmap(batched_test_y, test_y.begin() + mini_batch * batched_test_y.size() - (this->m_batch_size - remainder_valid_batch) * batched_test_y.size_j() * batched_test_y.size_k() * batched_test_y.size_l());
				// actual_batched_test_x = &r_batch_data;
				// actual_batched_test_y = &r_batch_label;
                // std::cout << "remainder batch: " << r_batch_data.size_i()<< '\n';
                // display_image(r_batch_data);
 			}	
            else
            {
				// skepu::io::cout << batched_test_y << "\n";
				// std::cout << "size of batched test y: "<<string_tensor_size(batched_test_y)<< "\n";
				// std::cout << "size of  test y: "<<string_tensor_size(test_y) << "\n";

                skel_flatmap.setLabel("Copy batch eval data X");
				skel_flatmap(batched_test_x, test_x.begin() + mini_batch * batched_test_x.size());
				skel_flatmap.setLabel("Copy batch eval data Y");
				
				// test_y.flush();
				// batched_test_y.flush();
					
				skel_flatmap(batched_test_y, test_y.begin() + mini_batch * batched_test_y.size());
				
				// actual_batched_test_x = &batched_test_x;
				// actual_batched_test_y = &batched_test_y;

	            // std::cout << "full batch " <<batch_index <<" size: " <<batch_data.size_i() << '\n';
 			}

			// skel_flatmap.setLabel("Copy batch eval data X");
			// skel_flatmap(batched_test_x, test_x.begin() + mini_batch * batched_test_x.size());
			// skel_flatmap.setLabel("Copy batch eval data Y");
			// skel_flatmap(batched_test_y, test_y.begin() + mini_batch * batched_test_y.size());

			auto &predicted = this->forward(batched_test_x);

			// Loss and accuracy
		/*  skel_cross_entropy_batched_loss.setDefaultSize(this->m_batch_size);
			loss += -skel_cross_entropy_batched_loss(batched_test_y, predicted) / this->m_batch_size;
			skel_accuracy_batched.setDefaultSize(this->m_batch_size);
			accuracy += (float)skel_accuracy_batched(batched_test_y, predicted) / this->m_batch_size;*/

			// Map+Reduce variant

			// Tensor4<T> grad = loss_function->compute_gradient(predicted, batched_train_y);
			// Tensor4<T> *gradient = &grad;

			skel_accuracy_batched_map.setLabel("Accuracy A");
			skel_accuracy_batched_map(temp_acc, batched_test_y, predicted);
			loss_function->compute_log_likelihoods(batched_test_y, predicted, temp_loss);

			// Skip the items of an overlapping remainder batch that were already evaluated
			const size_t skip = (mini_batch == (total_valid_batches - 1) && remainder_valid_batch > 0)
				? (this->m_batch_size - remainder_valid_batch) : 0;

			skepu::external("Accumulate evaluation metrics", skepu::read(temp_loss, temp_acc), [&]
			{
				for (size_t i = skip; i < this->m_batch_size; ++i)
				{
					sum_loss -= temp_loss(i);
					sum_correct += temp_acc(i);
				}
			});
			counted_samples += this->m_batch_size - skip;

			if (confusion_matrix)
			{
			    skepu::external("Compute Confusion Matrix", skepu::read(batched_test_y, predicted), [&]
			    {
    				for (size_t i = skip; i < batched_test_y.size_i(); ++i)
    				{
    					int actual_label = 0;
    					int predicted_label = 0;
    					if (batched_test_y.size_l() == 1)
    					{
    						// Single output (binary classification): threshold at 0.5
    						actual_label = (batched_test_y(i, 0, 0, 0) > 0.5f) ? 1 : 0;
    						predicted_label = (predicted(i, 0, 0, 0) > 0.5f) ? 1 : 0;
    					}
    					else
    					{
    						float max_actual = -1e30f, max_predicted = -1e30f;
    						for (size_t j = 0; j < batched_test_y.size_l(); ++j)
    						{
    							if (batched_test_y(i, 0, 0, j) > max_actual)
    							{
    								max_actual = batched_test_y(i, 0, 0, j);
    								actual_label = j;
    							}
    							if (predicted(i, 0, 0, j) > max_predicted)
    							{
    								max_predicted = predicted(i, 0, 0, j);
    								predicted_label = j;
    							}
    						}
    					}
                        (*confusion_matrix)(actual_label, predicted_label) += 1;
                    }
    			}, skepu::write(*confusion_matrix));
            }
		}

		const double n = (double)std::max<size_t>(counted_samples, 1);
		return { skepu::Scalar<float>((float)(sum_loss / n)), skepu::Scalar<float>((float)(sum_correct / n)) };
	}












	
	void display_image(skepu::Tensor4<float> &train_images)
	{
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

	// std::string name = "29_Jan_2025_opencl";
	void store_results_per_batch(std::string name, int epoch, int batch_index, double loss, double accuracy){
	  std::ofstream file;

	// Open the file in append mode (std::ios::app)
	file.open("./result_final_1_7_Mar/training_results_per_batch_"+ name +".csv", std::ios::app);

	// If the file is empty, add a header line (optional)
	if (file.tellp() == 0) {
	    file << "Epoch, Batch_Index, Loss,  Accuracy \n";
	}

	// Write the values to the file as a new line
	file << epoch << ","
	     << batch_index << ","
	     << loss << ","
	     << accuracy << "\n";


	file.close();
	}

	void store_results(std::string name, int batch_size, double learning_rate, int e,double average_epoch_loss_t,
   double average_b_accuracy, double average_epoch_loss_v,double average_v_accuracy )
   {
     std::ofstream file;

   // Open the file in append mode (std::ios::app)
   file.open("./result_final_1_7_Mar/training_results_"+ name +".csv", std::ios::app);

   // If the file is empty, add a header line (optional)
   if (file.tellp() == 0) {
       file << "Batch Size,Learning Rate,Epoch,Train Loss,Train Accuracy,Validation Loss,Validation Accuracy\n";
   }

   // Write the values to the file as a new line
   file << batch_size << ","
        << learning_rate << ","
        << e << ","
        << average_epoch_loss_t << ","
        << average_b_accuracy << ","
        << average_epoch_loss_v << ","
        << average_v_accuracy << "\n";

   file.close();
}














	// compiling loss function and optimizer 
	// Adam hyperparameters (beta1, beta2, epsilon) are ignored by SGD
	void configure(Loss loss_type, Opt optimizer_type, float adam_beta1 = 0.9f, float adam_beta2 = 0.999f, float adam_epsilon = 1e-8f){
		loss_function = std::make_unique<LossFunction<T>>(loss_type);
		optimizer = std::make_unique<Optimizer<T>>(optimizer_type, adam_beta1, adam_beta2, adam_epsilon);
		compiled = true;
	}


	// fit -> training 
	void fit(Tensor4<T> &train_x, Tensor4<T> &train_y, int epochs, float learning_rate, float validation_split,  const std::string &name, bool print_progress=true)
	{
	    SKEPU_TRACE_SCOPE("Model Fit");
		if (!compiled) throw std::runtime_error("Model must be compiled before training ");

		if (!this->m_initialized)
			this->init();

		// (Re-)initialize layer trainable parameters
		this->init_parameters();
		optimizer->reset();

		size_t valid_size = train_x.size_i() * validation_split;
		size_t train_size = train_x.size_i() - valid_size ;
		if (train_size < this->m_batch_size)
			SKEPU_ERROR("SkePU-DNN: training data (" << train_size << " samples after validation split) is smaller than the batch size (" << this->m_batch_size << ")");
		// std::cout << "total train_x: " << train_x.size_i() << '\n';
		// std::cout << "valid_size: " <<valid_size <<" train_size: " <<train_size << '\n';

		Tensor4<T> train_split_x(
			train_size,
			std::get<1>(this->m_input_size),
			std::get<2>(this->m_input_size),
			std::get<3>(this->m_input_size),
			"Split train data X"
		);
		Tensor4<T> train_split_y(train_size, 1, 1, train_y.size_l(), "Split train data Y");

		// A validation split of 0 samples is allowed (validation is then skipped); containers must be non-empty
		const size_t valid_alloc = std::max<size_t>(valid_size, 1);
		Tensor4<T> valid_split_x(
			valid_alloc,
			std::get<1>(this->m_input_size),
			std::get<2>(this->m_input_size),
			std::get<3>(this->m_input_size),
			"Split valid data X"
		);
		Tensor4<T> valid_split_y(valid_alloc, 1, 1, train_y.size_l(), "Split valid data Y");

		skel_flatmap.setLabel("Copy Split training data X");
		skel_flatmap(train_split_x, train_x.begin());
		skel_flatmap.setLabel("Copy Split training data Y");
		skel_flatmap(train_split_y, train_y.begin());

		// std::cout << "train_split_x: " <<train_split_x.size_i()<< " train_split_x " << '\n';
		// display_image(train_split_x);

		if (valid_size > 0)
		{
			skel_flatmap.setLabel("Copy split valid data X");
			skel_flatmap(valid_split_x, train_x.begin() + train_size * train_x.size_j() * train_x.size_k() * train_x.size_l());
			skel_flatmap.setLabel("Copy split valid data Y");
			skel_flatmap(valid_split_y, train_y.begin() + train_size * train_y.size_j() * train_y.size_k() * train_y.size_l());
		}

		size_t num_train_batches = train_split_x.size_i() / this->m_batch_size;
 		size_t remainder_train_batch = train_split_x.size_i() % this->m_batch_size;
		
 		// size_t num_valid_batches = valid_x.size_i()/this->m_batch_size;
 		// size_t remainder_valid_batch = valid_x.size_i()%this->m_batch_size;

		// std::cout << "train batch count: " << num_train_batches <<" remainder: " <<remainder_train_batch <<"\n";
		// std::cout << "valid bacth count: " << num_valid_batches <<" remainder: " <<remainder_valid_batch <<"\n";

		if (remainder_train_batch > 0) num_train_batches = num_train_batches + 1 ;
		const size_t total_train_batches = num_train_batches; // before any batch-count override
		
        // GPU backend bug: following is needed to correct software coherency
		train_split_x.flush();
		train_split_y.flush();
		valid_split_x.flush();
		valid_split_y.flush();

		// train_split_y = cut_tensor4(train_split_y, train_split_y.size_i());
		// train_split_x = cut_tensor4(train_split_x, train_split_x.size_i());
		// valid_split_x = cut_tensor4(valid_split_x, valid_split_x.size_i());
		// valid_split_y = cut_tensor4(valid_split_y, valid_split_y.size_i());

		Tensor4<T> batched_train_x(
			this->m_batch_size,
			std::get<1>(this->m_input_size),
			std::get<2>(this->m_input_size),
			std::get<3>(this->m_input_size),
			"Batch train data X"
		);
		
		Tensor4<T> batched_train_y(this->m_batch_size, 1, 1, train_y.size_l(), "Batch train data Y");

		size_t remainder_train_batch_allocate = remainder_train_batch;
		if (remainder_train_batch_allocate == 0) remainder_train_batch_allocate = 1;
       	// skepu::Tensor4<float> r_batch_data(remainder_train_batch_allocate, train_split_x.size_j(),train_split_x.size_k(), train_split_x.size_l());
       	// skepu::Tensor4<float> r_batch_label(remainder_train_batch_allocate, train_split_y.size_j(),train_split_y.size_k(), train_split_y.size_l());

	/*	Tensor4<T> r_batch_data(
				remainder_train_batch_allocate,
				std::get<1>(this->m_input_size),
				std::get<2>(this->m_input_size),
				std::get<3>(this->m_input_size),
				"r_batch_data train data X"
		);
		Tensor4<T> r_batch_label(remainder_train_batch_allocate, 1, 1, train_y.size_l(), "r_batch_label train data Y");*/

		// skepu::Vector<float> temp_loss(this->m_batch_size, "Loss temp");
		skepu::Vector<int> temp_acc(this->m_batch_size, "Accuracy temp");

		for (size_t epoch = 0; epoch < epochs; ++epoch) // epoch
		{
		    SKEPU_TRACE_SCOPE("Epoch");
			skepu::Scalar<float> total_train_loss = 0, total_train_accuracy = 0;
			std::stringstream ss; ss << "Epoch " << (epoch + 1);
			// dnn_debug << "Epoch " << (epoch + 1) << "\n";

			// const size_t mini_batch_count = train_split_x.size_i() / this->m_batch_size;

			// std::cout << "batches: " <<mini_batch_count << '\n';

		//	Tensor4<float> *actual_batched_train_x = nullptr;
		//	Tensor4<float> *actual_batched_train_y = nullptr;
			// std::cout << "Loop execute for " << num_train_batches << " batches \n";

			if (this->m_override_batch_count >= 0)
			    num_train_batches = std::min(num_train_batches, (size_t)this->m_override_batch_count);
			
			for (size_t mini_batch = 0; mini_batch < num_train_batches ; ++mini_batch) //  mini_batch_count , mini batches, mini_batch_count
			{
			    SKEPU_TRACE_SCOPE("Batch");
				if (mini_batch == (total_train_batches - 1) && remainder_train_batch > 0)
     			{
					// std::cout <<"problem here, offset: " << mini_batch * batched_train_x.size() - (this->m_batch_size - remainder_train_batch) * batched_train_x.size_j() * batched_train_x.size_k() * batched_train_x.size_l()  << "\n";
                    skel_flatmap(batched_train_x, train_split_x.begin() + mini_batch * batched_train_x.size() - (this->m_batch_size - remainder_train_batch) * batched_train_x.size_j() * batched_train_x.size_k() * batched_train_x.size_l()); // remainder + (remainder - batch_size) 
       				skel_flatmap(batched_train_y, train_split_y.begin() + mini_batch * batched_train_y.size() - (this->m_batch_size - remainder_train_batch) * batched_train_y.size_j() * batched_train_y.size_k() * batched_train_y.size_l());
				//	actual_batched_train_x = &r_batch_data;
				//	actual_batched_train_y = &r_batch_label;				
                //  std::cout << "remainder batch: " << r_batch_data.size_i()<< '\n';
       			//	display_image(r_batch_data);
     			}
     			else
                {
      			 	skel_flatmap.setLabel("Copy batch training data X");
					skel_flatmap(batched_train_x, train_split_x.begin() + mini_batch * batched_train_x.size());
					skel_flatmap.setLabel("Copy batch training data Y");
					skel_flatmap(batched_train_y, train_split_y.begin() + mini_batch * batched_train_y.size());
				//	actual_batched_train_x = &batched_train_x;
				//	actual_batched_train_y = &batched_train_y;

		       // std::cout << "full batch " <<batch_index <<" size: " <<batch_data.size_i() << '\n';
     			}

		// 		/************************ prepare batch - copying **************************/
				// dnn_debug << "\nMini-batch " << (mini_batch + 1) << " / " << mini_batch_count << "\n";
				// skel_flatmap.setLabel("Copy batch training data X");
				// skel_flatmap(batched_train_x, train_split_x.begin() + mini_batch * batched_train_x.size());
				// skel_flatmap.setLabel("Copy batch training data Y");
				// skel_flatmap(batched_train_y, train_split_y.begin() + mini_batch * batched_train_y.size());

				// std::cout << "batched_train_x: " <<batched_train_x.size_i()<< " batched_train_x  " << '\n';
				 // display_image(batched_train_x);

		// 		// skepu::io::cout << batched_train_x << '\n';
		// 		/************************ forward **************************/
		// 		std::stringstream ss; ss << "Epoch " << (epoch + 1) << " batch " << (mini_batch + 1) << " forward";
				auto &predicted = this->forward(batched_train_x, true);

				// std::cout << "predicted : " << predicted << '\n';

				// Tensor4<T> initial_gradient( // move out of loop
				// 		predicted.size_i(), predicted.size_j(),
				// 		predicted.size_k(), predicted.size_l(),
				// 		"Initial gradient"
				// );

				// skel_delta.setLabel("Loss gradient");
				// skel_delta(initial_gradient, predicted, batched_train_y);
				// Tensor4<T> *gradient = &initial_gradient;
				Tensor4<T> grad = loss_function->compute_gradient(predicted, batched_train_y);
				Tensor4<T> *gradient = &grad;
				// dnn_debug << "Initial gradient: " << *gradient << "\n";
				skepu::Scalar<float> accuracy = 0;

				// skel_cross_entropy_batched_loss_map.setLabel("Crossentropy A");
				// skel_cross_entropy_batched_loss_reduce.setLabel("Crossentropy B");
				// skel_accuracy_batched_map.setLabel("Accuracy A");
				// skel_accuracy_batched_reduce.setLabel("Accuracy B");

		// 		/// check
				// skel_cross_entropy_batched_loss_map(temp_loss, batched_train_y, predicted);
				// loss += -skel_cross_entropy_batched_loss_reduce(temp_loss) / this->m_batch_size;
				
				skel_accuracy_batched_map(temp_acc, batched_train_y, predicted);
				accuracy += skel_accuracy_batched_reduce(temp_acc) / (float)this->m_batch_size;
				skepu::Scalar<float> loss = loss_function->compute_loss(batched_train_y, predicted, this->m_batch_size);

				total_train_loss += loss;
				total_train_accuracy += accuracy;

				skepu::io::cout
				    << "Epoch " << epoch
					<< " Train Batch " << mini_batch + 1
					<< " - Loss: " << loss
					<< " - Accuracy: " << accuracy << "\n";

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
		 		SKEPU_TRACE_REGION_BEGIN("Backward");
				for (auto it = this->m_layers.rbegin(); it != this->m_layers.rend(); ++it)
				{
					auto &current_layer = **it;
					Tensor4<T> *forward_inputs = ((it + 1) != this->m_layers.rend()) ? &(**(it + 1)).m_vals : &batched_train_x;

					if (current_layer.is_trainable()) current_layer.m_learning_rate = learning_rate;
					gradient = current_layer.backward(forward_inputs, gradient, &batched_train_y);
					if (current_layer.is_trainable()) optimizer->step(current_layer);

		// 			dnn_debug << "Outgoing gradient: " << *gradient << "\n";
				}
		 		SKEPU_TRACE_REGION_END();
				
                // store_results_per_batch(name, epoch + 1, mini_batch + 1, loss, accuracy);
		 	} // end of mini batch
		//
		// 	if (print_progress)
		// 	{
		// 		std::stringstream ss; ss << "Epoch " << (epoch + 1) << " evaluate";
			// Validation needs at least one full batch; otherwise it is skipped (reported as 0)
			std::tuple<skepu::Scalar<float>, skepu::Scalar<float>> score{0.f, 0.f};
			if (valid_size >= this->m_batch_size)
				score = this->evaluate(valid_split_x, valid_split_y);
			else if (epoch == 0)
				std::cout << "SkePU-DNN: validation set (" << valid_size << " samples) smaller than the batch size, skipping validation\n";
		// 		std::cout << "- Epoch " << std::setw(4) << (epoch + 1) << " / " << std::setw(4) << epochs << ":";
		// 		std::cout << " Valid Loss: " << std::setw(10) << std::get<0>(score) << ", Valid Accuracy: " << std::setw(10) << std::get<1>(score) << "\n";
		// 	}
		    
            skepu::io::cout
                << "Epoch " << epoch
                << " Train Loss: " << total_train_loss / num_train_batches
                << " Train Acc: " << total_train_accuracy / num_train_batches
                << " Validation Loss: " << std::get<0>(score)
                << " Validation Acc: " << std::get<1>(score)
                << "\n";
			
            // store_results(name , this->m_batch_size, learning_rate, epoch + 1, total_train_loss/num_train_batches, total_train_accuracy/num_train_batches, std::get<0>(score), std::get<1>(score) );
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

	size_t layerCount() const { return this->m_layers.size(); }

	void setBatchCountOverride(int bc)
	{
	    std::cout << "Overriding batch count with : " << bc << "\n";
	    this->m_override_batch_count = bc;
	}

protected:
	bool m_initialized = false;

	std::vector<std::shared_ptr<Layer<T>>> m_layers{};
	size_t m_layers_count = 0;
	skepu::PRNG m_prng{0}; // default seed

	Dimensions m_input_size;
	size_t m_batch_size;

	Tensor4<T> *m_result = nullptr;

	std::unique_ptr<LossFunction<T>> loss_function; 
	std::unique_ptr<Optimizer<T>> optimizer;
	bool compiled = false;

	int m_override_batch_count = -1;
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

}

namespace skepu::dnn
{
    template <typename T = float, typename... Args>
    impl::SequentialModel<T> SequentialModel(Args &&...args)
    {
    	return impl::SequentialModel<T>(std::forward<Args>(args)...);
    }
}

static skepu::PrecompilerMarker endOf_SequentialModel_HPP;
#endif // SKEPU_PRECOMPILED
