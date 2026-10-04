#pragma once
#include <skepu>
#ifndef SKEPU_PRECOMPILED
static skepu::PrecompilerMarker startOf_Activation_HPP;


float sigmoid_uf(float x)
{
	return 1 / (1 + exp(-x)); // or tanh
}

float sigmoid_prime_uf(float x)
{
	float sigma_x = sigmoid_uf(x);
	return sigma_x * (1 - sigma_x);
}

// forward oass
float relu_uf(float x) { return (x > 0) ? x : 0; }

float leaky_relu_uf(float x) { return (x > 0) ? x :(0.01 * x); }

float P_relu_uf(float x, float alpha ){ return (x > 0) ? x :(alpha * x);}

float elu_uf(float x, float alpha){
	return (x > 0) ? x : alpha * (std::exp(x) - 1);
}

// bacCKWRaRD PASS
float relu_prime_uf(float x, float g)
{
	return (x > 0) ? g : 0 ;
}

float leaky_relu_prime_uf(float x)
{
	return (x > 0) ? 1.0 : 0.01 ;
}

float P_relu_prime_uf(float x, float alpha)
{
	return (x > 0) ? 1.0 : alpha ;
}

float elu_prime_uf(float x, float alpha)
{
	return (x > 0) ? 1.0 : alpha * std::exp(x);
}




float tanh_uf(float x) { return tanh(x); }

float tanh_prime_uf(float x)
{
	float t = tanh(x);
	return 1 - t * t;
}


// SoftMax, non-batched
float softmax_uf_1_map(float x) { return exp(x); }

float softmax_uf_1_reduce(float lhs, float rhs) { return lhs + rhs; }

float softmax_uf_2(float x, float sum) { return exp(x) / sum; }

// SoftMax, batched
float softmax_batched_uf_1(skepu::Pool4D<float> DNN_CONST pool)
{
	float sum = 0;
	for (size_t l = 0; l < pool.sl; l++)
		sum += exp(pool(0, 0, 0, l));
	return sum;
}

float softmax_batched_uf_2(skepu::Index4D index, float x, skepu::Ten4<float> sums)
{
	return exp(x) / sums(index.i, index.j, index.k, 0);
}


/** new softmax without MapPool */
float new_softmax_uf_1(skepu::Index4D index, skepu::Ten4<float> input)
{
  size_t item = index.i;
  float max_value = input(item,0,0,0);

  for(size_t i = 0; i < input.size_l; i++){
    if( input(item,0,0,i) > max_value ){
      max_value = input(item,0,0,i);
    }
  }

  return max_value;

}

float new_softmax_uf_2(skepu::Index4D index, skepu::Ten4<float> input, skepu::Ten4<float>  max_vals)
{
	float sum = 0;
	for (size_t l = 0; l < input.size_l; l++){
      sum += exp(input(index.i, 0, 0, l) - max_vals(index.i,0,0,0));
  }
	return sum;
}

float new_softmax_uf_3(skepu::Index4D index, float x, skepu::Ten4<float> sums, skepu::Ten4<float>  max_vals)
{
	return exp(x - max_vals(index.i,0,0,0)) / sums(index.i, index.j, index.k, 0);
}

/****************************** Softmax derivative  *********************************************/

float softmax_prime_uf(skepu::Index4D index , skepu::Ten4<float> gradient, skepu::Ten3<float> jacobian, int num_classes){
    float res = 0.0;
    for(size_t j = 0; j < num_classes; ++j){
      res += jacobian(index.i , index.l, j ) * gradient(index.i, 0,0,j);
    }
  return res;
}


float softmax_jacobian_uf(skepu::Index3D index , skepu::Ten4<float> s ){

  if ( index.j == index.k ){
    return s(index.i ,0 ,0 , index.j) * ((float) 1.0 - s(index.i ,0 ,0 , index.j) );
  }
  else {
    return -s(index.i ,0 ,0 , index.j) * s(index.i ,0 ,0 , index.k);
  }

}



auto skel_act_sigmoid = skepu::Map(sigmoid_uf);
auto skel_act_relu = skepu::Map(relu_uf);
//	auto skel_act_softmax_1 = skepu::MapReduce(softmax_uf_1_map, softmax_uf_1_reduce);
auto skel_act_softmax_2 = skepu::Map<1>(softmax_uf_2);

auto skel_softmax_batched_1 = skepu::MapPool(softmax_batched_uf_1);
auto skel_softmax_batched_2 = skepu::Map(softmax_batched_uf_2);

auto new_softmax_skel_1 = skepu::Map<0>(new_softmax_uf_1);
auto new_softmax_skel_2 = skepu::Map<0>(new_softmax_uf_2);
auto new_softmax_skel_3 = skepu::Map<1>(new_softmax_uf_3);

auto softmax_jacobian = skepu::Map<0>(softmax_jacobian_uf);
auto softmax_prime  = skepu::Map<0>(softmax_prime_uf);

auto skel_act_tanh = skepu::Map(tanh_uf);


auto skel_act_sigmoid_prime = skepu::Map(sigmoid_prime_uf);
auto skel_act_relu_prime = skepu::Map(relu_prime_uf);
auto skel_act_tanh_prime = skepu::Map(tanh_prime_uf);



namespace skepu::dnn::impl
{
    template <typename T = float>
    class Activation : public Layer<T>
    {
    public:
	Activation(Act arg_act) : Layer<T>() { this->m_act = arg_act; }
    
	// Move constructor
	Activation(Activation<T> &&source) = default;
    
	void setup(Dimensions input_size) override
	{
		this->m_vals.setLabel(this->label() + " Activation values");
		this->m_temp_vals.setLabel(this->label() + " Activation temporaries");
	//	this->m_temp_backprop.setLabel(this->label() + " Activation gradients");
    
		this->m_output_size = input_size;
    
		//	if (!this->m_in_place) ...

		if (this->m_act == Act::SoftMax)
		{
    		this->m_temp_vals.init(
    			std::get<0>(this->m_output_size),
    			std::get<1>(this->m_output_size),
    			std::get<2>(this->m_output_size),
    			1
    		);
		}
    
	/* 	this->m_temp_backprop.init(
			std::get<0>(this->m_output_size), std::get<1>(this->m_output_size),
			std::get<2>(this->m_output_size), std::get<3>(this->m_output_size)
			);*/
    
		Layer<T>::setup(input_size);
	}
    
	Tensor4<T> *forward(Tensor4<T> *inputs, bool is_training=false) override
	{
		//	dnn_debug << "Act input: " << string_tensor_size(*inputs) << "\n";
		//	dnn_debug << "Act vals: " << string_tensor_size(this->m_vals) << "\n";
    
		Tensor4<T> *target = (this->m_in_place) ? inputs : &this->m_vals;
    
		if (this->m_act == Act::ReLU)
		{
			skel_act_relu.setLabel(this->label() + " ReLU");
			skel_act_relu(*target, *inputs);
		}
		else if (this->m_act == Act::Sigmoid)
		{
			skel_act_sigmoid.setLabel(this->label() + " Sigmoid");
			skel_act_sigmoid(*target, *inputs);
		}
		else if (this->m_act == Act::SoftMax)
		{
			dnn_debug << "Softmax input: " << *inputs << "\n";
			
			Tensor4<T> m_temp_value_max_reduce (inputs->size_i(),inputs->size_j(),inputs->size_k(), 1);
    
			new_softmax_skel_1(m_temp_value_max_reduce, *inputs);
			new_softmax_skel_2(this->m_temp_vals, *inputs, m_temp_value_max_reduce);
			new_softmax_skel_3(*target, *inputs, this->m_temp_vals, m_temp_value_max_reduce);
    
			// skel_softmax_batched_1.setPoolSize(1, 1, 1, inputs->size_l());
			// skel_softmax_batched_1.setStride(1, 1, 1, inputs->size_l());
			// skel_softmax_batched_1.setLabel(this->label() + " Softmax A");
			// skel_softmax_batched_2.setLabel(this->label() + " Softmax B");
			// skel_softmax_batched_1(this->m_temp_vals, *inputs);
			// skel_softmax_batched_2(*target, *inputs, this->m_temp_vals);
		}
		else if (this->m_act == Act::TanH)
		{
			skel_act_tanh.setLabel(this->label() + " TanH");
			skel_act_tanh(*target, *inputs);
		}
		return target;
	}
    
	Tensor4<T> *backward(Tensor4<T> *inputs, Tensor4<T> *gradient, Tensor4<T> *y) override
	{
		dnn_debug << "Activation function backprop\n";
		//	dnn_debug << "Act input: " << string_tensor_size(*inputs) << "\n";
		//	dnn_debug << "Act vals: " << string_tensor_size(this->m_vals) << "\n";
    
		//	Tensor4<T> *target = (this->m_in_place) ? inputs : &this->m_vals;
		Tensor4<T> *target = gradient;
    
		if (this->m_act == Act::ReLU)
		{
			skel_act_relu_prime.setLabel(this->label() + " ReLU derivative");
			skel_act_relu_prime(*target, *inputs, *gradient);
		}
		else if (this->m_act == Act::Sigmoid)
		{
			skel_act_sigmoid_prime.setLabel(this->label() + " Sigmoid derivative");
			skel_act_sigmoid_prime(*target, *inputs);
		}
		else if (this->m_act == Act::SoftMax)
		{
    
		  int n_class = this->m_vals.size_l();
    
		  skepu::Tensor3<float> jacobian(inputs->size_i(),n_class, n_class);
    
		  softmax_jacobian(jacobian, this->m_vals);
		  softmax_prime(*target, *gradient, jacobian, n_class);
    
		}
		else if (this->m_act == Act::TanH)
		{
			skel_act_tanh_prime.setLabel(this->label() + " TanH derivative");
			skel_act_tanh_prime(*target, *inputs);
		}
    
		return target;
	}
    
	std::string summary() override
	{
		std::ostringstream os;
		os << Layer<T>::summary();
		os << "Activation layer: ";
		if (this->m_act == Act::ReLU)
			os << "ReLU";
		else if (this->m_act == Act::Sigmoid)
			os << "Sigmoid";
		else if (this->m_act == Act::SoftMax)
			os << "SoftMax";
		else if (this->m_act == Act::TanH)
			os << "TanH";
		os << (this->m_in_place ? " (in-place) " : " (not in-place) ");
		return os.str();
	}
    
    private:
	Tensor4<T> m_temp_vals, m_temp_backprop;
    };
}

static skepu::PrecompilerMarker endOf_Activation_HPP;
#endif // SKEPU_PRECOMPILED
