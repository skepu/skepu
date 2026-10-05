#pragma once
#include <skepu>
#include <iomanip>
#include <limits>
#include <sstream>
#include <fstream>
#include <vector>
#ifndef SKEPU_PRECOMPILED
static skepu::PrecompilerMarker startOf_Layer_HPP;


namespace skepu::dnn::impl
{
	template <typename T> class Optimizer; // defined in sequential_model.hpp

	// Text I/O of parameter values for model files: a count, then the values (9 significant digits
	// reproduce a float exactly)
	inline void write_values(std::ostream &os, const std::vector<float> &v)
	{
	os << v.size() << "\n" << std::setprecision(std::numeric_limits<float>::max_digits10);
	for (size_t i = 0; i < v.size(); ++i)
		os << v[i] << ((i % 8 == 7 || i + 1 == v.size()) ? "\n" : " ");
	}

	inline std::vector<float> read_values(std::istream &is, size_t expected)
	{
	size_t n = 0;
	is >> n;
	if (n != expected)
		SKEPU_ERROR("SkePU-DNN: model file has " << n << " parameter values for a layer that needs " << expected);
	std::vector<float> v(n);
	for (auto &x : v) is >> x;
	if (!is)
		SKEPU_ERROR("SkePU-DNN: model file ended while reading parameters");
	return v;
	}

	template <typename T = float>
	class Layer
	{
	public:
	Layer(Dimensions arg_output_size) : m_output_size{arg_output_size} {}
	Layer() : m_output_size{0, 0, 0, 0} {}

	Layer(Layer &) = default;
	Layer(Layer &&) = default;

	virtual ~Layer() {}

	public:
	virtual Layer &&activation(Act new_act)
	{
		this->m_act = new_act;
		return std::move(*this);
	}

	// Override for layer classes with trainable parameters.
	virtual Layer &&pretrained(std::string fname)
	{
		this->m_pretrained_params = fname;
		return std::move(*this);
	}

	virtual Layer &&weights(Init arg_init)
	{
		this->m_init_weights = arg_init;
		return std::move(*this);
	}

	virtual Layer &&biases(Init arg_init)
	{
		this->m_init_biases = arg_init;
		return std::move(*this);
	}

	// Override for layer classes with trainable parameters.
	virtual bool is_trainable() const { return false; }

	// Applies one optimizer step to the layer's trainable parameters (after its backward pass).
	// Layers with parameters override this and pass their weights, biases and gradients to the
	// optimizer; the optimizer picks the update for the container types it is given.
	virtual void apply_update(Optimizer<T> &/*opt*/, float /*learning_rate*/) { /* No parameters */ }

	// Description of the layer for model files: "<Type> key=value ...", including its activation
	// function. Activation layers return "" and are skipped (they belong to the layer before them).
	virtual std::string describe() const
	{
		SKEPU_ERROR("SkePU-DNN: this layer type cannot be saved to a model file");
		return "";
	}

	// Writes/reads the trainable parameters as text for model files (layers without parameters do nothing)
	virtual void write_params(std::ostream &/*os*/) {}
	virtual void read_params(std::istream &/*is*/) {}

	// " weights=X biases=Y activation=Z", the attributes shared by the layers with parameters
	std::string common_attributes(bool with_init) const
	{
		std::ostringstream os;
		if (with_init)
			os << " weights=" << to_string(this->m_init_weights) << " biases=" << to_string(this->m_init_biases);
		os << " activation=" << to_string(this->m_act);
		return os.str();
	}

	std::string label()
	{
		std::ostringstream os;
		os << "L" << this->m_layer_id;
		return os.str();
	}

	virtual std::string summary()
	{
		std::ostringstream os;
		os << "[";
		if (this->m_is_setup)
			os << "Layer " << this->m_layer_id;
		else
			os << "Not initialized";

		os << "; out-shape{" << std::get<0>(this->m_output_size) << " x "
				<< std::get<1>(this->m_output_size) << " x "
				<< std::get<2>(this->m_output_size) << " x "
				<< std::get<3>(this->m_output_size);
		os << "}";
		os << "; batch-size{";
		os << this->m_batch_size;
		os << "}";
		os << "] ";
		return os.str();
	}

	// Subclasses must set their output size before calling super here!
	virtual void setup(Dimensions /*input_size*/)
	{
		this->m_is_setup = true;

		this->m_vals.init(
				std::get<0>(this->m_output_size), std::get<1>(this->m_output_size),
				std::get<2>(this->m_output_size), std::get<3>(this->m_output_size));
	}

	virtual void serialize(std::string /*base_file_name*/)
	{
		// No default action
	}

	[[noreturn]] virtual Tensor4<T> *forward(Tensor4<T> */*inputs*/, bool /*is_training*/=false)
	{
		SKEPU_ERROR("Layer: calling pure virtual 'forward' member function");
	}

	[[noreturn]] virtual Tensor4<T> *backward(Tensor4<T> */*inputs*/, Tensor4<T> */*gradient*/, Tensor4<T> */*y*/)
	{
		SKEPU_ERROR("Layer: calling pure virtual 'backward' member function");
	}

	//		private:

	// Override for layer classes with trainable parameters.
	// For internal use. Requires the layer to be setup first.
	virtual void load_pretrained() { /* No default action */ }

	// Override for layer classes with trainable parameters.
	virtual void init_parameters() { /* No default action */ }

	public: // for now, protected later?
	int m_layer_id = -1;
	Level m_layer_level = Level::Hidden;

	bool m_is_setup = false;
	bool m_keep_params = false; // parameters were loaded from a model file: fit() must not re-initialize them
	bool m_in_place = false; // when possible

	PRNG *m_prng = nullptr; // owned by the model

	size_t m_batch_size;
	Dimensions m_output_size;

	Act m_act = Act::None; // default value?
	Init m_init_weights = Init::Uniform;
	Init m_init_biases = Init::Zero;

	std::string m_pretrained_params = SKEPU_DNN_NO_PRETRAINED_PARAMS;

	Tensor4<T> m_vals;

	friend class SequentialLayer;
	};
}

static skepu::PrecompilerMarker endOf_Layer_HPP;
#endif // SKEPU_PRECOMPILED
