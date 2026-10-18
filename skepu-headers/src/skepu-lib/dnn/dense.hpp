#pragma once
#include <skepu>
#ifndef SKEPU_PRECOMPILED
static skepu::PrecompilerMarker startOf_Dense_HPP;


float fully_connected_batched_uf(
	skepu::Index4D index,
	skepu::Vec<float> DNN_CONST bias,
	skepu::Mat<float> DNN_CONST weights,
	skepu::Ten4<float> DNN_CONST in
)
{
	size_t batch_index = index.i;
	size_t class_index = index.l;
	float res = bias(class_index);
	for (size_t i = 0; i < weights.rows; ++i)
		res += weights(i, class_index) * in(batch_index, 0, 0, i);
	return res;
}


float propagate_dense_weight_gradient_uf(
	skepu::Index2D index,
	skepu::Ten4<float> inputs,
	skepu::Ten4<float> gradient)
{
	size_t num_batches = inputs.size_i;
	float sum = 0;
	for (size_t b = 0; b < num_batches; ++b)
		sum += inputs(b, 0, 0, index.row) * gradient(b, 0, 0, index.col); // indexing?
	return sum / num_batches;
}

float propagate_dense_bias_gradient_uf(
	skepu::Index1D index,
	skepu::Ten4<float> gradient)
{
	size_t num_batches = gradient.size_i;
	float sum = 0;
	for (size_t b = 0; b < num_batches; ++b)
		sum += gradient(b, 0, 0, index.i);
	return sum / num_batches;
}


auto skel_dense = skepu::Map(fully_connected_batched_uf);
auto skel_propagate_dense_weight_gradient = skepu::Map(propagate_dense_weight_gradient_uf);
auto skel_propagate_dense_bias_gradient   = skepu::Map(propagate_dense_bias_gradient_uf);


namespace skepu::dnn::impl
{
	template <typename T = float>
	class Dense : public Layer<T>
	{
	public:
	Dense(size_t arg_output_size) : Layer<T>(), m_out(arg_output_size) {}

	Dense(Dense &&source) : Layer<T>(std::move(source))
	{
		this->m_weights = std::move(source.m_weights);
		this->m_biases = std::move(source.m_biases);
		this->m_out = source.m_out;
	}

	~Dense() override = default;

	Dense &&weights(Init arg_init) override
	{
		Layer<T>::weights(arg_init);
		return std::move(*this);
	}

	Dense &&biases(Init arg_init) override
	{
		Layer<T>::biases(arg_init);
		return std::move(*this);
	}

	void setup(Dimensions input_size) override
	{
		this->m_vals.setLabel(this->label() + " Dense values");
		this->m_weights.setLabel(this->label() + " Dense weights");
		this->m_biases.setLabel(this->label() + " Dense biases");
		this->m_dW.setLabel(this->label() + " Dense weight-deltas");
		this->m_db.setLabel(this->label() + " Dense bias-deltas");
		this->m_temp.setLabel(this->label() + " Dense gradient");

		size_t out = this->m_out;
		size_t in = std::get<3>(input_size);

		this->m_weights.init(in, out);
		this->m_biases.init(out);

		this->m_output_size = {std::get<0>(input_size), 1, 1, out};

		this->load_pretrained();
		Layer<T>::setup(input_size);

		// Backprop temporaries
		this->m_db.init(this->m_biases.size());
		this->m_dW.init(this->m_weights.size_i(), this->m_weights.size_j());
		if (this->m_layer_level != Level::Input)
		{
			this->m_temp.init(
				std::get<0>(input_size), std::get<1>(input_size),
				std::get<2>(input_size), std::get<3>(input_size)
			);
		}
	}

	virtual void serialize(std::string base_file_name) override
	{
		skepu::external(this->label() + " serialize", skepu::read(this->m_weights, this->m_biases), [&]
		{
			std::ofstream ofs(base_file_name + "_" + this->label() + ".csv");

			for (size_t i = 0; i < this->m_weights.size_i(); i++)
			{
				for (size_t j = 0; j < this->m_weights.size_j(); j++)
				{
					ofs << this->m_weights(i,j);
					ofs << "\n";
				}
			}

			for (size_t i = 0; i < this->m_biases.size_i(); i++)
				ofs << this->m_biases(i) << "\n";
		});
	}

	bool is_trainable() const override { return true; }

	Dense &&pretrained(std::string fname) override
	{
		Layer<T>::pretrained(fname);
		return std::move(*this);
	}

	void load_pretrained() override
	{
		if (this->m_pretrained_params != SKEPU_DNN_NO_PRETRAINED_PARAMS)
		{
			// Load data from file
			// The elements are written on the host, in an external section (a distributed container is updated from it)
			skepu::external(this->label() + " load pretrained", [&]
			{
				std::vector<float> temp = deserialize_csv_helper<float>(this->m_pretrained_params);
				auto it = load_from_vector(this->m_weights, temp, temp.begin());
				it = load_from_vector(this->m_biases, temp, it);
			}, skepu::write(this->m_weights, this->m_biases));
		}
	}

	void init_parameters() override
	{
		// Keep pretrained parameters
		if (this->m_pretrained_params != SKEPU_DNN_NO_PRETRAINED_PARAMS || this->m_keep_params)
			return;

		// Weights
		if (this->m_init_weights == Init::Xavier_uniform)
		{
			double limit = sqrt(6.0 / (this->m_weights.size_i() + this->m_out));

			skel_init_random_uniform.setLabel(this->label() + " Random-init dense weights");
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform(this->m_weights, -limit, limit);
		}
		else if(this->m_init_weights == Init::Xavier_normal){
			double limit = sqrt(6.0 / (this->m_weights.size_i() + this->m_out)); // uniform with the variance of Xavier-normal

			skel_init_random_uniform.setLabel(this->label() + " Random-init dense weights");
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform(this->m_weights, -limit, limit);
		}
		else if (this->m_init_weights == Init::He_uniform){
			double limit = sqrt(6.0 / (this->m_weights.size_i()));

			skel_init_random_uniform.setLabel(this->label() + " Random-init dense weights");
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform(this->m_weights, -limit, limit);
		}
		else if (this->m_init_weights == Init::He_normal){
			double limit = sqrt(6.0 / (this->m_weights.size_i())); // uniform with the variance of He-normal

			skel_init_random_uniform.setLabel(this->label() + " Random-init dense weights");
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform(this->m_weights, -limit, limit);
		}
		else if (this->m_init_weights == Init::Uniform)
		{
			double limit = 1.0;
			skel_init_random_uniform.setLabel(this->label() + " Random-init dense weights");
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform(this->m_weights, -limit, limit);
		}
		else if (this->m_init_weights == Init::Zero)
		{
			skel_init_zero.setLabel(this->label() + " Zero-init dense weights");
			skel_init_zero(this->m_weights);
		}

		// Biases
		if (this->m_init_biases == Init::Uniform)
		{
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform.setLabel(this->label() + " Random-init dense biases");
			skel_init_random_uniform(this->m_biases, -1.0, 1.0);
		}
		else if (this->m_init_biases == Init::Zero) {
			skel_init_zero.setLabel(this->label() + " Zero-init dense biases");
			skel_init_zero(this->m_biases);
		}
	}

	Dense &&activation(Act new_act) override
	{
		Layer<T>::activation(new_act);
		return std::move(*this);
	}

	std::string summary() override
	{
		std::ostringstream os;
		os << Layer<T>::summary() << "Dense layer params{("
				<< this->m_weights.size_i() << " x " << this->m_weights.size_j()
				<< ") + " << this->m_biases.size() << " = "
				<< (this->m_weights.size() + this->m_biases.size()) << "}";
		return os.str();
	}

	Tensor4<T> *forward(Tensor4<T> *inputs, bool /*is_training*/=false) override
	{
		skel_dense.setLabel(this->label() + " Dense forward");
		skel_dense(this->m_vals, this->m_biases, this->m_weights, *inputs);

		return &this->m_vals;
	}

	// inputs: tensor in shape (batch, 1, 1, N)
	Tensor4<T> *backward(Tensor4<T> *inputs, Tensor4<T> *gradient, Tensor4<T> */*y*/) override
	{
		skel_propagate_loss_gradient.setLabel(this->label() + " Dense backprop-grad");
		skel_propagate_dense_weight_gradient.setLabel(this->label() + " Dense backprop weights");
		skel_propagate_dense_bias_gradient.setLabel(this->label() + " Dense backprop biases");

		if (this->m_layer_level != Level::Input)
			skel_propagate_loss_gradient(this->m_temp, *gradient, this->m_weights);

		skel_propagate_dense_weight_gradient(this->m_dW, *inputs, *gradient);
		skel_propagate_dense_bias_gradient(this->m_db, *gradient);

		return (this->m_layer_level != Level::Input) ? &this->m_temp : nullptr;
	}

	std::string describe() const override
	{
		return "Dense units=" + std::to_string(this->m_out) + this->common_attributes(true);
	}

	void write_params(std::ostream &os) override
	{
		std::vector<float> v;
		skepu::external(this->label() + " write params", skepu::read(m_weights, m_biases), [&]
		{
			for (size_t i = 0; i < m_weights.size_i(); ++i)
				for (size_t j = 0; j < m_weights.size_j(); ++j) v.push_back(m_weights(i, j));
			for (size_t i = 0; i < m_biases.size(); ++i) v.push_back(m_biases(i));
		});
		write_values(os, v);
	}

	void read_params(std::istream &is) override
	{
		std::vector<float> v = read_values(is, m_weights.size() + m_biases.size());
		skepu::external(this->label() + " read params", [&]
		{
			size_t n = 0;
			for (size_t i = 0; i < m_weights.size_i(); ++i)
				for (size_t j = 0; j < m_weights.size_j(); ++j) m_weights(i, j) = v[n++];
			for (size_t i = 0; i < m_biases.size(); ++i) m_biases(i) = v[n++];
		}, skepu::write(m_weights, m_biases));
		this->m_keep_params = true;
	}

	void apply_update(Optimizer<T> &opt, float learning_rate) override
	{
		opt.update(*this, "Dense", m_weights, m_dW, m_biases, m_db, learning_rate);
	}

	Matrix<T>&  get_d_weights() { return m_weights; }
	Matrix<T>&  get_d_gradients() { return m_dW; }

	Vector<T>&  get_biases() { return m_biases; }
	Vector<T>&  get_biases_gradients() { return m_db; }


	protected:
	size_t m_out = 0;

	Matrix<T> m_weights;
	Vector<T> m_biases;

	Vector<T> m_db;
	Matrix<T> m_dW;
	Tensor4<T> m_temp;
	};
}

namespace skepu::dnn
{
	template <typename T = float, typename... Args>
	impl::Dense<T> Dense(Args &&...args)
	{
	return impl::Dense<T>(std::forward<Args>(args)...);
	}
}

static skepu::PrecompilerMarker endOf_Dense_HPP;
#endif // SKEPU_PRECOMPILED
