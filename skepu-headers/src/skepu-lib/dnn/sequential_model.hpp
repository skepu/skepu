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
		this->m_matrix_state.clear();
		this->m_tensor_state.clear();
	}

	// Update the parameters of one layer after its backward pass
	void step(Layer<T>& layer, float learning_rate)
	{
		layer.apply_update(*this, learning_rate);
	}

	// Called by layers (Layer::apply_update) with their parameters; works for any weight container
	// type that has a state map below
	template <typename Weights>
	void update(Layer<T>& layer, const std::string &kind, Weights &w, Weights &dw, Vector<T> &b, Vector<T> &db, float lr)
	{
		if (this->m_optimizer_type == Opt::SGD)
		{
			skel_update_params.setLabel(layer.label() + " " + kind + " Update Weights");
			skel_update_params(w, w, dw, lr);
			skel_update_params.setLabel(layer.label() + " " + kind + " Update Biases");
			skel_update_params(b, b, db, lr);
		}
		else if (this->m_optimizer_type == Opt::Adam)
		{
			AdamState<Weights> &st = this->state_for(this->states(w), layer, w, b);
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

	std::map<const Layer<T>*, AdamState<Matrix<T>>> &states(const Matrix<T> &) { return this->m_matrix_state; }
	std::map<const Layer<T>*, AdamState<Tensor4<T>>> &states(const Tensor4<T> &) { return this->m_tensor_state; }

	Opt m_optimizer_type;
	float m_beta1, m_beta2, m_epsilon;
	std::map<const Layer<T>*, AdamState<Matrix<T>>> m_matrix_state;
	std::map<const Layer<T>*, AdamState<Tensor4<T>>> m_tensor_state;
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

	// Sets up all layers (allocates their data). Safe to call more than once.
	void init()
	{
	    SKEPU_TRACE_SCOPE("Model Init");
		if (this->m_initialized)
			return;

		this->m_layers[0]->m_layer_level = Level::Input;
		this->m_layers[this->m_layers.size() - 1]->m_layer_level = Level::Output;

		// Allocate data
		Dimensions input_size = this->m_input_size;
		for (auto &current : this->m_layers)
		{
			current->setup(input_size);
			input_size = current->m_output_size;
		}

		this->m_initialized = true;
	}

	template <typename ConcreteLayer>
	void add(ConcreteLayer &&l)
	{
		Act activationFunc = l.m_act;
		l.m_batch_size = this->m_batch_size;
		l.m_layer_id = ++ this->m_layers_count;
		l.m_prng = &this->m_prng;
		this->m_layers.push_back(std::make_shared<typename std::remove_reference<ConcreteLayer>::type>(std::forward<ConcreteLayer>(l)));

		// The activation function of a layer is a separate layer that shares its id
		if (activationFunc != Act::None)
		{
			auto act_layer = std::make_shared<Activation<T>>(activationFunc);
			act_layer->m_batch_size = this->m_batch_size;
			act_layer->m_layer_id = this->m_layers_count;
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
			temp = current->forward(temp, is_training);
		}
		this->m_result = temp;
		dnn_debug << "Final result: " << *this->m_result << "\n";
		return *temp;
	}

	// Choose the loss function and optimizer. Required before fit() and evaluate().
	// Adam hyperparameters (beta1, beta2, epsilon) are ignored by SGD.
	void configure(Loss loss_type, Opt optimizer_type, float adam_beta1 = 0.9f, float adam_beta2 = 0.999f, float adam_epsilon = 1e-8f)
	{
		loss_function = std::make_unique<LossFunction<T>>(loss_type);
		optimizer = std::make_unique<Optimizer<T>>(optimizer_type, adam_beta1, adam_beta2, adam_epsilon);
		compiled = true;
	}

	// Runs inference on a whole dataset (any number of samples, at least one batch) and returns the
	// outputs of the last layer, shape (samples, ...).
	Tensor4<T> predict(Tensor4<T> &x)
	{
		SKEPU_TRACE_SCOPE("Model Predict");
		this->init();
		this->check_dataset_size(x.size_i(), "prediction");

		const Dimensions out = this->m_layers.back()->m_output_size;
		Tensor4<T> result(x.size_i(), std::get<1>(out), std::get<2>(out), std::get<3>(out), "Predictions");
		Tensor4<T> batched_x(this->m_batch_size, std::get<1>(this->m_input_size), std::get<2>(this->m_input_size), std::get<3>(this->m_input_size), "Batch predict data X");

		BatchPlan plan(x.size_i(), this->m_batch_size, -1);
		for (size_t b = 0; b < plan.used; ++b)
		{
			skel_flatmap.setLabel("Copy batch predict data X");
			skel_flatmap(batched_x, x.begin() + plan.first_sample(b) * sample_size(x));
			Tensor4<T> &predicted = this->forward(batched_x);

			// The last batch may overlap the previous one; the overlap is rewritten with identical values
			const size_t first = plan.first_sample(b);
			skepu::external("Collect predictions", skepu::read(predicted), [&]
			{
				for (size_t i = 0; i < predicted.size_i(); ++i)
					for (size_t j = 0; j < predicted.size_j(); ++j)
						for (size_t k = 0; k < predicted.size_k(); ++k)
							for (size_t l = 0; l < predicted.size_l(); ++l)
								result(first + i, j, k, l) = predicted(i, j, k, l);
			}, skepu::write(result));
		}
		return result;
	}

	// Mean loss and accuracy over a dataset; optionally fills a confusion matrix (actual x predicted).
	// Every sample is counted exactly once.
	std::tuple<skepu::Scalar<float>, skepu::Scalar<float>>
	evaluate(Tensor4<T> &test_x, Tensor4<T> &test_y, skepu::Matrix<int> *confusion_matrix = nullptr)
	{
        SKEPU_TRACE_SCOPE("Model Evaluate");
		this->init();
		this->require_configured("evaluate");

		if (test_x.size_i() == 0)
			return { skepu::Scalar<float>(0.f), skepu::Scalar<float>(0.f) };
		this->check_dataset_size(test_x.size_i(), "evaluation");

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

		// Exact per-sample sums: the last batch overlaps the previous one when the data size is not a
		// multiple of the batch size, so its first (batch_size - remainder) items must not be counted twice
		double sum_loss = 0, sum_correct = 0;
		size_t counted_samples = 0;

		BatchPlan plan(test_x.size_i(), this->m_batch_size, this->m_override_batch_count);
		for (size_t mini_batch = 0; mini_batch < plan.used; ++mini_batch)
		{
		    SKEPU_TRACE_SCOPE("Batch");
			this->load_batch(test_x, test_y, batched_test_x, batched_test_y, plan.first_sample(mini_batch));

			auto &predicted = this->forward(batched_test_x);

			skel_accuracy_batched_map.setLabel("Accuracy A");
			skel_accuracy_batched_map(temp_acc, batched_test_y, predicted);
			loss_function->compute_log_likelihoods(batched_test_y, predicted, temp_loss);

			// Skip the items of an overlapping remainder batch that were already evaluated
			const size_t skip = plan.skip(mini_batch);

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
				this->accumulate_confusion(batched_test_y, predicted, skip, *confusion_matrix);
		}

		const double n = (double)std::max<size_t>(counted_samples, 1);
		return { skepu::Scalar<float>((float)(sum_loss / n)), skepu::Scalar<float>((float)(sum_correct / n)) };
	}

	// Trains the model with mini-batch gradient descent. The last validation_split fraction of the data
	// is held out for validation after every epoch (the data is used in order, it is not shuffled).
	// `name` is currently unused.
	void fit(Tensor4<T> &train_x, Tensor4<T> &train_y, int epochs, float learning_rate, float validation_split,  const std::string &name, bool print_progress=true)
	{
	    SKEPU_TRACE_SCOPE("Model Fit");
		this->require_configured("fit");
		this->init();

		// (Re-)initialize layer trainable parameters and optimizer state
		this->init_parameters();
		optimizer->reset();

		const size_t valid_size = train_x.size_i() * validation_split;
		const size_t train_size = train_x.size_i() - valid_size;
		this->check_dataset_size(train_size, "training (after the validation split)");

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

		if (valid_size > 0)
		{
			skel_flatmap.setLabel("Copy split valid data X");
			skel_flatmap(valid_split_x, train_x.begin() + train_size * sample_size(train_x));
			skel_flatmap.setLabel("Copy split valid data Y");
			skel_flatmap(valid_split_y, train_y.begin() + train_size * sample_size(train_y));
		}

        // GPU backend bug: following is needed to correct software coherency
		train_split_x.flush();
		train_split_y.flush();
		valid_split_x.flush();
		valid_split_y.flush();

		Tensor4<T> batched_train_x(
			this->m_batch_size,
			std::get<1>(this->m_input_size),
			std::get<2>(this->m_input_size),
			std::get<3>(this->m_input_size),
			"Batch train data X"
		);
		Tensor4<T> batched_train_y(this->m_batch_size, 1, 1, train_y.size_l(), "Batch train data Y");

		skepu::Vector<int> temp_acc(this->m_batch_size, "Accuracy temp");

		// When the training size is not a multiple of the batch size, the last batch overlaps the
		// previous one, so a few samples are trained on twice per epoch
		BatchPlan plan(train_size, this->m_batch_size, this->m_override_batch_count);

		for (int epoch = 0; epoch < epochs; ++epoch)
		{
		    SKEPU_TRACE_SCOPE("Epoch");
			skepu::Scalar<float> total_train_loss = 0, total_train_accuracy = 0;

			for (size_t mini_batch = 0; mini_batch < plan.used; ++mini_batch)
			{
			    SKEPU_TRACE_SCOPE("Batch");
				this->load_batch(train_split_x, train_split_y, batched_train_x, batched_train_y, plan.first_sample(mini_batch));

				// Forward
				auto &predicted = this->forward(batched_train_x, true);

				Tensor4<T> grad = loss_function->compute_gradient(predicted, batched_train_y);
				Tensor4<T> *gradient = &grad;

				skepu::Scalar<float> accuracy = 0;
				skel_accuracy_batched_map(temp_acc, batched_train_y, predicted);
				accuracy += skel_accuracy_batched_reduce(temp_acc) / (float)this->m_batch_size;
				skepu::Scalar<float> loss = loss_function->compute_loss(batched_train_y, predicted, this->m_batch_size);

				total_train_loss += loss;
				total_train_accuracy += accuracy;

				if (print_progress)
					skepu::io::cout
						<< "Epoch " << epoch
						<< " Train Batch " << mini_batch + 1
						<< " - Loss: " << loss
						<< " - Accuracy: " << accuracy << "\n";

				// Backward, updating the parameters of each layer right after its backward pass
		 		SKEPU_TRACE_REGION_BEGIN("Backward");
				for (auto it = this->m_layers.rbegin(); it != this->m_layers.rend(); ++it)
				{
					auto &current_layer = **it;
					Tensor4<T> *forward_inputs = ((it + 1) != this->m_layers.rend()) ? &(**(it + 1)).m_vals : &batched_train_x;

					gradient = current_layer.backward(forward_inputs, gradient, &batched_train_y);
					if (current_layer.is_trainable()) optimizer->step(current_layer, learning_rate);
				}
		 		SKEPU_TRACE_REGION_END();
		 	}

			// Validation needs at least one full batch; otherwise it is skipped (reported as 0)
			std::tuple<skepu::Scalar<float>, skepu::Scalar<float>> score{0.f, 0.f};
			if (valid_size >= this->m_batch_size)
				score = this->evaluate(valid_split_x, valid_split_y);
			else if (epoch == 0 && print_progress)
				std::cout << "SkePU-DNN: validation set (" << valid_size << " samples) smaller than the batch size, skipping validation\n";

			if (print_progress)
				skepu::io::cout
					<< "Epoch " << epoch
					<< " Train Loss: " << total_train_loss / plan.used
					<< " Train Acc: " << total_train_accuracy / plan.used
					<< " Validation Loss: " << std::get<0>(score)
					<< " Validation Acc: " << std::get<1>(score)
					<< "\n";
		}
	}

	void display_image(skepu::Tensor4<float> &train_images)
	{
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
	// Splits a dataset of `samples` items into batches of `batch_size`. When `samples` is not a multiple
	// of the batch size, the last batch overlaps the previous one (it covers the final `batch_size`
	// samples) so that all batches are full; skip(b) is the number of items of batch b that were
	// already part of the previous batch. `used` is the number of batches to process
	// (limited by the batch-count override, if any).
	struct BatchPlan
	{
		size_t samples, batch_size, total, remainder, used;

		BatchPlan(size_t arg_samples, size_t arg_batch_size, int override_count)
		: samples(arg_samples), batch_size(arg_batch_size),
		  total(arg_samples / arg_batch_size + ((arg_samples % arg_batch_size) > 0 ? 1 : 0)),
		  remainder(arg_samples % arg_batch_size), used(total)
		{
			if (override_count >= 0)
				used = std::min(total, (size_t)override_count);
		}

		bool is_overlapping(size_t b) const { return b == total - 1 && remainder > 0; }
		size_t first_sample(size_t b) const { return is_overlapping(b) ? samples - batch_size : b * batch_size; }
		size_t skip(size_t b) const { return is_overlapping(b) ? batch_size - remainder : 0; }
	};

	// Number of elements of one sample (item along the first dimension)
	static size_t sample_size(const Tensor4<T> &t) { return t.size_j() * t.size_k() * t.size_l(); }

	// Copies the batch starting at sample `first_sample` of (x, y) into the batch buffers
	void load_batch(Tensor4<T> &x, Tensor4<T> &y, Tensor4<T> &batch_x, Tensor4<T> &batch_y, size_t first_sample)
	{
		skel_flatmap.setLabel("Copy batch data X");
		skel_flatmap(batch_x, x.begin() + first_sample * sample_size(x));
		skel_flatmap.setLabel("Copy batch data Y");
		skel_flatmap(batch_y, y.begin() + first_sample * sample_size(y));
	}

	void check_dataset_size(size_t samples, const char *what)
	{
		if (samples < this->m_batch_size)
			SKEPU_ERROR("SkePU-DNN: " << what << " data (" << samples << " samples) is smaller than the batch size (" << this->m_batch_size << ")");
	}

	void require_configured(const char *what)
	{
		if (!this->compiled || !this->loss_function || !this->optimizer)
			SKEPU_ERROR("SkePU-DNN: configure(loss, optimizer) must be called before " << what << "()");
	}

	// Adds the (actual, predicted) class of the batch items from `skip` on to the confusion matrix
	void accumulate_confusion(Tensor4<T> &actual, Tensor4<T> &predicted, size_t skip, skepu::Matrix<int> &confusion_matrix)
	{
		skepu::external("Compute Confusion Matrix", skepu::read(actual, predicted), [&]
		{
			for (size_t i = skip; i < actual.size_i(); ++i)
			{
				int actual_label = 0;
				int predicted_label = 0;
				if (actual.size_l() == 1)
				{
					// Single output (binary classification): threshold at 0.5
					actual_label = (actual(i, 0, 0, 0) > 0.5f) ? 1 : 0;
					predicted_label = (predicted(i, 0, 0, 0) > 0.5f) ? 1 : 0;
				}
				else
				{
					float max_actual = -1e30f, max_predicted = -1e30f;
					for (size_t j = 0; j < actual.size_l(); ++j)
					{
						if (actual(i, 0, 0, j) > max_actual)
						{
							max_actual = actual(i, 0, 0, j);
							actual_label = j;
						}
						if (predicted(i, 0, 0, j) > max_predicted)
						{
							max_predicted = predicted(i, 0, 0, j);
							predicted_label = j;
						}
					}
				}
				confusion_matrix(actual_label, predicted_label) += 1;
			}
		}, skepu::write(confusion_matrix));
	}

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
