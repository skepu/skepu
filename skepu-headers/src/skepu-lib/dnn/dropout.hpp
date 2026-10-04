#pragma once
#include <skepu>
#ifndef SKEPU_PRECOMPILED
static skepu::PrecompilerMarker startOf_Dropout_HPP;

template <typename T> T dropout_uf(skepu::Random<1> &rand, T el, float rate)
{
	float p = rand.getNormalized();  
	return (p > rate) ? (el / (1.0 - rate)) : 0;
}

template <typename T> skepu::multiple<T, char> dropout_masked_uf(skepu::Random<1> &rand, T el, float rate)
{
	float p = rand.getNormalized();
	// char enabled = (p > rate);
	// T res = enabled ? (el / (1.0 - rate)) : 0;
	// return skepu::ret(res, enabled);
	if (p < rate){
    char mask = 0;
    return skepu::ret(0, mask);
  }
  else{
    char mask = 1;
    return skepu::ret( (el/(1 - rate)), mask);
  }

}

template <typename T> T dropout_masked_backprop_uf(T gradient, char enabled, float rate)
{
	return enabled ? gradient/* / (1.0 - rate)*/ : 0; // todo check
}


auto skel_dropout = skepu::Map<1>(dropout_uf<float>);
auto skel_dropout_masked = skepu::Map<1>(dropout_masked_uf<float>);
auto skel_dropout_backprop_masked = skepu::Map<2>(dropout_masked_backprop_uf<float>);



namespace skepu::dnn::impl
{
    template <typename T = float>
    class Dropout : public Layer<T>
    {
    public:
	Dropout(float arg_rate) : Layer<T>(), m_dropout_rate(arg_rate) {}
    
	// Move constructor
	Dropout(Dropout &&source)
	: Layer<T>(std::move(source)), m_dropout_rate(source.m_dropout_rate)
	{
		this->m_prng = std::move(source.m_prng);
	}
    
	~Dropout() override = default;
    
	void setup(Dimensions input_size) override
	{
		this->m_vals.setLabel(this->label() + " Dropout values");
		this->m_mask.setLabel(this->label() + " Dropout mask");
		this->m_temp.setLabel(this->label() + " Dropout gradient");
    
		size_t out = std::get<0>(this->m_output_size);
		size_t in = std::get<0>(input_size);
    
		this->m_output_size = input_size;
		//	if (!this->m_in_place) ...
		Layer<T>::setup(input_size);
		
		this->m_mask.init(
			std::get<0>(input_size), std::get<1>(input_size),
			std::get<2>(input_size), std::get<3>(input_size)
		);

		if (this->m_layer_level != Level::Input)
		{
    		this->m_temp.init(
    			std::get<0>(input_size), std::get<1>(input_size),
    			std::get<2>(input_size), std::get<3>(input_size)
    		);
		}
	}
    
	Tensor4<T> *forward(Tensor4<T> *inputs, bool is_training=false) override
	{
		if (!is_training) return inputs;
    
		// During training, drop values
		skel_dropout_masked.setPRNG(*this->m_prng);
		skel_dropout_masked.setLabel(this->label() + " Dropout forward");
		skel_dropout_masked(this->m_vals, this->m_mask, *inputs, this->m_dropout_rate);
		return &this->m_vals;
	}
    
	Tensor4<T> *backward(Tensor4<T> *inputs, Tensor4<T> *gradient, Tensor4<T> *y) override
	{
		skel_dropout_backprop_masked.setLabel(this->label() + " Dropout backward");
		skel_dropout_backprop_masked(this->m_temp, *gradient, this->m_mask, this->m_dropout_rate);
		return &this->m_temp;
	}
    
	std::string summary() override {
		std::stringstream os;
		os << Layer<T>::summary();
		os << "Dropout layer with rate " << this->m_dropout_rate;
		return os.str();
	}
    
    private:
	float m_dropout_rate;
	Tensor4<char> m_mask;
	Tensor4<T> m_temp;
    };
    
}

namespace skepu::dnn
{
    template <typename T = float, typename... Args>
    impl::Dropout<T> Dropout(Args &&...args)
    {
	return impl::Dropout<T>(std::forward<Args>(args)...);
    }
}

static skepu::PrecompilerMarker endOf_Dropout_HPP;
#endif // SKEPU_PRECOMPILED
