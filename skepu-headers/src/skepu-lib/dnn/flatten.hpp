#pragma once
#include <skepu>
#ifndef SKEPU_PRECOMPILED

static skepu::PrecompilerMarker startOf_Flatten_HPP;




template <typename T = float>
class Flatten : public Layer<T>
{
public:
	Flatten() : Layer<T>() {}
	Flatten(Flatten &&) = default;
	~Flatten() override = default;

	void setup(Dimensions input_size) override
	{
		this->m_vals.setLabel(this->label() + " Flatten values");
		this->m_temp.setLabel(this->label() + " Flatten gradient");

		this->m_output_size = {
			std::get<0>(input_size), 1, 1,
			std::get<1>(input_size) * std::get<2>(input_size) * std::get<3>(input_size)
		};

		this->m_temp.init(
			std::get<0>(input_size), std::get<1>(input_size),
			std::get<2>(input_size), std::get<3>(input_size)
		);

		Layer<T>::setup(input_size);
	}

	Tensor4<T> *forward(Tensor4<T> *inputs, bool is_training=false) override
	{
		skel_flatmap.setLabel(this->label() + " Flatten forward");
		skel_flatmap(this->m_vals, *inputs);
		return &this->m_vals;
	}

	Tensor4<T> *backward(Tensor4<T> *inputs, Tensor4<T> *gradient, Tensor4<T> *y) override
	{
		dnn_debug << "Flatten backprop\n";
		dnn_debug << "Flatten vals size: " << string_tensor_size(this->m_vals) << "\n";
		dnn_debug << "Flatten input size: " << string_tensor_size(*inputs) << "\n";
		dnn_debug << "Flatten gradient size: " << string_tensor_size(*gradient) << "\n";
		dnn_debug << "Flatten temp size: " << string_tensor_size(this->m_temp) << "\n";

		skel_flatmap.setLabel(this->label() + " Flatten backward");
		skel_flatmap(this->m_temp, *gradient);

		dnn_debug << "Flatten inputs: " << *inputs << "\n";
		dnn_debug << "Flatten gradient: " << *gradient << "\n";
		dnn_debug << "Flatten temp: " << this->m_temp << "\n";

		return &this->m_temp;
	}

	std::string summary() override
	{
		std::stringstream os;
		os << Layer<T>::summary();
		os << "Flatten layer ";
		return os.str();
	}

private:

	Tensor4<T> m_temp;
};



static skepu::PrecompilerMarker endOf_Flatten_HPP;