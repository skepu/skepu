#pragma once
#include <skepu>
#ifndef SKEPU_PRECOMPILED

static skepu::PrecompilerMarker startOf_Layer_HPP;




namespace impl
{
template <typename T>
class SequentialModel;

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

	virtual Layer &&weights(skepu::ml::Init arg_init)
	{
		this->m_init_weights = arg_init;
		return std::move(*this);
	}

	virtual Layer &&biases(skepu::ml::Init arg_init)
	{
		this->m_init_biases = arg_init;
		return std::move(*this);
	}

	// Override for layer classes with trainable parameters.
	virtual bool is_trainable() const { return false; }

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
	virtual void setup(Dimensions input_size)
	{
		this->m_is_setup = true;

		this->m_vals.init(
				std::get<0>(this->m_output_size), std::get<1>(this->m_output_size),
				std::get<2>(this->m_output_size), std::get<3>(this->m_output_size));
	}

	virtual void serialize(std::string base_file_name)
	{
		// No default action
	}

	[[noreturn]] virtual Tensor4<T> *forward(Tensor4<T> *inputs, bool is_training=false)
	{
		SKEPU_ERROR("Layer: calling pure virtual 'forward' member function");
	}

	[[noreturn]] virtual Tensor4<T> *backward(Tensor4<T> *inputs, Tensor4<T> *gradient, Tensor4<T> *y)
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
	bool m_in_place = false; // when possible

	skepu::PRNG *m_prng;

	size_t m_batch_size;
	Dimensions m_output_size;

	Act m_act = Act::None; // default value?
	Init m_init_weights = Init::Uniform;
	Init m_init_biases = Init::Zero;

	std::string m_pretrained_params = SKEPU_DNN_NO_PRETRAINED_PARAMS;

	std::shared_ptr<Layer> *predecessor = nullptr;
	std::shared_ptr<Layer> *successor = nullptr;

	float m_learning_rate = 0;
	Tensor4<T> m_vals;

	friend class SequentialLayer;
};


static skepu::PrecompilerMarker endOf_Layer_HPP;