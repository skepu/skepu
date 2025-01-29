#include <iostream>
#include "skepu-mnist.hpp"

#include <skepu>

#define DNN_CONST /*const&*/

float fully_connected_uf(
	float bias,
	skepu::MatRow<float> DNN_CONST weights,
	skepu::Vec<float> DNN_CONST in
)
{
	float res = bias;
	for (size_t i = 0; i < in.size; ++i)
		res += weights(i) * in(i);
	return res;
}

float fully_connected_batched_uf(
	skepu::Index2D index,
	skepu::Vec<float> DNN_CONST bias,
	skepu::Mat<float> DNN_CONST weights,
	skepu::Mat<float> DNN_CONST in
)
{
	size_t batch_index = index.row;
	float res = bias(index.col);
	for (size_t i = 0; i < in.cols; ++i)
		res += weights(index.col, i)
		* in(batch_index, i);
	return res;
}

float fully_connected_backprop_uf(
	//float bias,,
	float sigmoid_p,
	skepu::MatCol<float> DNN_CONST weights,
	skepu::Vec<float> DNN_CONST delta
)
{
	float res = 0;//bias;
	for (size_t i = 0; i < delta.size; ++i)
		res += weights(i) * delta(i);
	return res * sigmoid_p;
}

float fully_connected_batched_backprop_uf(
	//float bias,,
	skepu::Index2D index,
	float sigmoid_p,
	skepu::Mat<float> DNN_CONST weights,
	skepu::Mat<float> DNN_CONST delta
)
{
	size_t batch_index = index.row;
	float res = 0;//bias;
	for (size_t i = 0; i < delta.cols; ++i)
		res += weights(i, index.col) * delta(batch_index, i);
	return res * sigmoid_p;
}



float sigmoid_uf(float x)
{
	return 1 / (1 + exp(-x)); // or tanh
}

float sigmoid_prime_uf(float x)
{
	float sigma_x = sigmoid_uf(x);
	return sigma_x * (1 - sigma_x);
}

float relu_uf(float x)
{
	return (x < 0) ? 0 : x;
}

float flatten_uf(float x)
{
	return x;
}


float max_pooling_uf(skepu::Pool2D<float> DNN_CONST pool /*coefficient, bias*/)
{
	float maxval = 10e-10;
	for (size_t i = 0; i < pool.si; ++i)
		for (size_t j = 0; j < pool.sj; ++j)
			maxval = (maxval > pool(i, j)) ? maxval : pool(i, j);
	return maxval;
}

float avg_pooling_uf(skepu::Pool2D<float> DNN_CONST pool /*coefficient, bias*/)
{
	float sum = 0;
	for (size_t i = 0; i < pool.si; ++i)
		for (size_t j = 0; j < pool.sj; ++j)
			sum += pool(i, j);
	return sum / (pool.si + pool.sj);
}

float generalized_pooling_uf(
	skepu::Index3D index,
	skepu::Pool3D<float> DNN_CONST pool
	/*coefficient, bias*/
)
{
	size_t i = index.i; // find my feature map
	float sum = 0;
	for (size_t j = 0; j < pool.sj; ++j)
		for (size_t k = 0; k < pool.sk; ++k)
			sum += pool(i, j, k);
	return sum /* * coefficient + bias*/;
}


float convolutional_uf(
	skepu::Index3D index,
	skepu::Region3D<float> DNN_CONST in,
	skepu::Ten3<float> DNN_CONST weights,
	skepu::Mat<int> DNN_CONST connection_map
) // bias
{
	float res = 0;
	for (size_t i = -in.oi; i <= in.oi; ++i)
		for (size_t j = -in.oj; j <= in.oj; ++j)
			if (connection_map(index.i, index.j) == 1)
			{
				for (size_t k = 0; k < weights.size_k; ++k)
					res += weights(i + in.oi, j + in.oi, k) * in(i, j, k - index.k);
			}
	return res;
}


float mse_prime_uf(float a, float y)
{
	return (y - a) * (y - a) * 0.5;
}

float mse_uf(float a, float y)
{
	return y - a;
}

// Column-wise reduction over batch items
float generate_nabla_b_uf(
	skepu::Index1D index,
	skepu::Mat<float> delta
)
{
	size_t bias_index = index.i;
	size_t batch_size = delta.rows;
	float res = 0;
	for (size_t i = 0; i < batch_size; ++i)
		res += delta(i, bias_index);
	return res;
}

float generate_nabla_w_uf(
	skepu::Index2D index,
	skepu::Mat<float> delta,
	skepu::Mat<float> activations
)
{
	size_t batch_size = delta.rows;
	float res = 0;
	for (size_t i = 0; i < batch_size; ++i)
		res += delta(i, index.row) * activations(i, index.col);
	return res;
}

float update_uf(
	float old,
	float updated,
	float eta // learning rate
)
{
	return old - eta * updated;
}

template<typename T>
T sum_uf(T a, T b)
{
	return a + b;
}

float product_uf(float a, float b)
{
	return a * b;
}

float flatmap_uf(float a)
{
	return a;
}

float reshape_y_uf(skepu::Index2D index, skepu::Vec<size_t> DNN_CONST y)
{
	if (index.col == y(index.row))
		return 1.f;
	return 0.f;
}

size_t argmax_uf(skepu::Index1D index, skepu::MatRow<float> DNN_CONST y)
{
	float max_v = -999999.f;
	size_t max_i = 0;
	for (size_t i = 0; i < y.cols; ++i)
	{
		if (y(i) > max_v)
		{
			max_v = y(i);
			max_i = i;
		}
	}
	return max_i;
}

size_t eval_uf(size_t a, size_t b) { return (a == b) ? 1 : 0; }


void mlp()
{
	// MLP
	std::string data_dir_path = "./data/mnist";
	
	skepu::Vector<mnist::label_t> train_labels = mnist::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte");
	skepu::Tensor3<float> train_images = mnist::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte", -1.0, 1.0);
	skepu::Vector<mnist::label_t> test_labels = mnist::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte");
	skepu::Tensor3<float> test_images = mnist::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte", -1.0, 1.0);
	
	
	
	size_t BATCH_SIZE = 100;
	size_t RUN_EPOCHS = 30;
	
	size_t IN_WIDTH = 784;
	size_t H1_WIDTH = 30;
	size_t H2_WIDTH = 10;
	size_t OUT_WIDTH = 10;
	
	// Forward
	auto flatmap = skepu::Map(flatmap_uf);
	auto hidden_dense = skepu::Map(fully_connected_batched_uf);
	auto hidden_activation = skepu::Map(sigmoid_uf);
	
	// Backprop
	auto cost_prime = skepu::Map(mse_prime_uf);
	auto reshape_y = skepu::Map(reshape_y_uf);
	auto hidden_dense_backprop = skepu::Map(fully_connected_batched_backprop_uf);
	auto sigmoid_prime = skepu::Map(sigmoid_prime_uf);
	auto hadamard = skepu::Map(product_uf);
	auto generate_nabla_b = skepu::Map(generate_nabla_b_uf);
	auto generate_nabla_w = skepu::Map(generate_nabla_w_uf);
	auto update = skepu::Map<2>(update_uf);
	auto argmax = skepu::Map(argmax_uf);
	auto evaluate = skepu::MapReduce<2>(eval_uf, sum_uf<size_t>);
	auto cost_function = skepu::MapReduce(mse_uf, sum_uf<float>);


	// =========================================================
	// ===== INPUT
	// =========================================================
	
	skepu::Matrix<float> batched_train_input(BATCH_SIZE, IN_WIDTH); // 28x28 @ 1 feature map
	skepu::Matrix<float> test_input(test_images.size_i(), IN_WIDTH); // 28x28 @ 1 feature map
	
	skepu::Matrix<float> expected_y(BATCH_SIZE, OUT_WIDTH);
	
	
	// =========================================================
	// == LAYER 1
	// == Fully connected, from 784, to 120
	// =========================================================
	
	skepu::Matrix<float> weights_1(H1_WIDTH, IN_WIDTH);
	weights_1.randomizeReal(-1, 1);
	skepu::Vector<float> bias_1(H1_WIDTH);
	
	
	// =========================================================
	// == LAYER 2
	// == Fully connected, from 120, to 84
	// =========================================================
	
	skepu::Matrix<float> weights_2(H2_WIDTH, H1_WIDTH);
	weights_2.randomizeReal(-1, 1);
	skepu::Vector<float> bias_2(H2_WIDTH);
	
	
	// =========================================================
	// == LAYER 3
	// == Fully connected, from 84, to 10
	// =========================================================
	
	skepu::Matrix<float> weights_3(OUT_WIDTH, H2_WIDTH);
	weights_3.randomizeReal(-1, 1);
	skepu::Vector<float> bias_3(OUT_WIDTH);
	
	
	// =========================================================
	// == OUTPUT
	// =========================================================
	
	skepu::Matrix<float> hidden_state_3_val(BATCH_SIZE, OUT_WIDTH);
	skepu::Matrix<float> hidden_state_3_activation(BATCH_SIZE, OUT_WIDTH);
	skepu::Matrix<float> delta_3(BATCH_SIZE, OUT_WIDTH);
	skepu::Matrix<float> hidden_state_3_sigmoid_prime(BATCH_SIZE, OUT_WIDTH);
	skepu::Matrix<float> cost_derivative(BATCH_SIZE, OUT_WIDTH);
	skepu::Vector<float> nabla_b_3(bias_3.size());
	skepu::Matrix<float> nabla_w_3(weights_3.total_rows(), weights_3.total_cols());
	
	skepu::Matrix<float> hidden_state_2_val(BATCH_SIZE, H2_WIDTH);
	skepu::Matrix<float> hidden_state_2_activation(BATCH_SIZE, H2_WIDTH);
	skepu::Matrix<float> delta_2(BATCH_SIZE, H2_WIDTH);
	skepu::Matrix<float> hidden_state_2_sigmoid_prime(BATCH_SIZE, H2_WIDTH);
	skepu::Vector<float> nabla_b_2(bias_2.size());
	skepu::Matrix<float> nabla_w_2(weights_2.total_rows(), weights_2.total_cols());
	
	skepu::Matrix<float> hidden_state_1_val(BATCH_SIZE, H1_WIDTH);
	skepu::Matrix<float> hidden_state_1_activation(BATCH_SIZE, H1_WIDTH);
	skepu::Matrix<float> delta_1(BATCH_SIZE, H1_WIDTH);
	skepu::Matrix<float> hidden_state_1_sigmoid_prime(BATCH_SIZE, H1_WIDTH);
	skepu::Vector<float> nabla_b_1(bias_1.size());
	skepu::Matrix<float> nabla_w_1(weights_1.total_rows(), weights_1.total_cols());
	
	
	for (size_t epoch = 0; epoch < RUN_EPOCHS; ++epoch)
	{
		size_t mini_batch_count = train_labels.size() / BATCH_SIZE;
		
		std::cout << "Epoch " << epoch << "\n";
	
		if (epoch > 0)
		for (size_t mini_batch = 0; mini_batch < mini_batch_count; ++mini_batch)
		{
			///////////////////////
			// FORWARD PASS
			
			std::cout << "Mini-batch " << mini_batch << " / " << mini_batch_count << ": Start forward pass!\n";
			
			flatmap(batched_train_input, train_images.begin() + mini_batch * BATCH_SIZE * train_images.size_j() * train_images.size_k());
			
			hidden_dense(hidden_state_1_val, bias_1, weights_1, batched_train_input);
			hidden_activation(hidden_state_1_activation, hidden_state_1_val);
			
			hidden_dense(hidden_state_2_val, bias_2, weights_2, hidden_state_1_activation);
			hidden_activation(hidden_state_2_activation, hidden_state_2_val);
			
			hidden_dense(hidden_state_3_val, bias_3, weights_3, hidden_state_2_activation);
			hidden_activation(hidden_state_3_activation, hidden_state_3_val);
			
			float cost = cost_function(hidden_state_3_activation, expected_y);
			
			std::cout << "End forward pass!\n";
			
			
			//////////////////////////
			// BACKWARDS PASS
			
			float eta = 0.01; // learning rate
			
		//	std::cout << "Start backward pass!\n";
			
			// Last layer (3)
			reshape_y(expected_y, train_labels.begin() + mini_batch * BATCH_SIZE);
			
			sigmoid_prime(hidden_state_3_sigmoid_prime, hidden_state_3_val);
			
			cost_prime(cost_derivative, hidden_state_3_activation, expected_y);
			
			hadamard(delta_3, cost_derivative, hidden_state_3_sigmoid_prime);
			
			generate_nabla_b(nabla_b_3, delta_3); // sum up per input in epoch
			generate_nabla_w(nabla_w_3, delta_3, hidden_state_2_activation); // sum up per input in epoch with outer product
			
			update(bias_3, bias_3, nabla_b_3, eta / BATCH_SIZE);
			update(weights_3, weights_3, nabla_w_3, eta / BATCH_SIZE);
			
			
			// Previous layer (2)
			sigmoid_prime(hidden_state_2_sigmoid_prime, hidden_state_2_val);
			
			hidden_dense_backprop(delta_2, hidden_state_2_sigmoid_prime, weights_3, delta_3);
			cost_prime(cost_derivative, hidden_state_2_activation, expected_y);
			
			hadamard(delta_2, cost_derivative, hidden_state_2_sigmoid_prime);
			
			generate_nabla_b(nabla_b_2, delta_2); // sum up per input in epoch
			generate_nabla_w(nabla_w_2, delta_2, hidden_state_1_activation); // sum up per input in epoch with outer product
			
			update(bias_2, bias_2, nabla_b_2, eta / BATCH_SIZE);
			update(weights_2, weights_2, nabla_w_2, eta / BATCH_SIZE);
			
			
			// Previous layer (1)
			sigmoid_prime(hidden_state_1_sigmoid_prime, hidden_state_1_val);
			
			hidden_dense_backprop(delta_1, hidden_state_1_sigmoid_prime, weights_2, delta_2);
			
			generate_nabla_b(nabla_b_1, delta_1); // sum up per input in epoch
			generate_nabla_w(nabla_w_1, delta_1, batched_train_input); // sum up per input in epoch with outer product
			
			update(bias_1, bias_1, nabla_b_1, eta / BATCH_SIZE);
			update(weights_1, weights_1, nabla_w_1, eta / BATCH_SIZE);
			
			
		//	std::cout << "End backward pass!\n";
		}
		
		// Evaluate
		
		std::cout << "Evaluate " << test_images.size_i() << "\n";
		
		skepu::Matrix<float> eval_hidden_state_1_val(test_images.size_i(), H1_WIDTH);
		skepu::Matrix<float> eval_hidden_state_1_activation(test_images.size_i(), H1_WIDTH);
		
		skepu::Matrix<float> eval_hidden_state_2_val(test_images.size_i(), H2_WIDTH);
		skepu::Matrix<float> eval_hidden_state_2_activation(test_images.size_i(), H2_WIDTH);
		
		skepu::Matrix<float> eval_hidden_state_3_val(test_images.size_i(), OUT_WIDTH);
		skepu::Matrix<float> eval_hidden_state_3_activation(test_images.size_i(), OUT_WIDTH);
		
		
		
//		std::cout << "flatmap " << test_input.size() << "\n";
		flatmap(test_input, test_images);
		
//		std::cout << "hidden_dense_5\n5_val: " << eval_hidden_state_5_val.size_i() << " x " << eval_hidden_state_5_val.size_j() << "\n";
		
//		std::cout << "bias_5: " << bias_5.size() << "\n";
		
//		std::cout << "weights_5: " << weights_5.size_i() << " x " << weights_5.size_j() << "\n";
//		std::cout << "test_input: " << test_input.size_i() << " x " << test_input.size_j() << "\n";
		

	//	std::cout << "weights 5: " << weights_5 << "\n";
	//	std::cout << "weights 5: " << weights_6 << "\n";
	//	std::cout << "weights 5: " << weights_7 << "\n";
		
		hidden_dense(eval_hidden_state_1_val, bias_1, weights_1, test_input);
		hidden_activation(eval_hidden_state_1_activation, eval_hidden_state_1_val);
		
		hidden_dense(eval_hidden_state_2_val, bias_2, weights_2, eval_hidden_state_1_activation);
		hidden_activation(eval_hidden_state_2_activation, eval_hidden_state_2_val);
		
		hidden_dense(eval_hidden_state_3_val, bias_3, weights_3, eval_hidden_state_2_activation);
		hidden_activation(eval_hidden_state_3_activation, eval_hidden_state_3_val);
		
		// argmax
		skepu::Vector<size_t> argmax_output(test_images.size_i());
		argmax(argmax_output, eval_hidden_state_3_activation);
//		std::cout << argmax_output << "\n";
//		std::cout << eval_hidden_state_7_activation << "\n";
		
		
		auto correct = evaluate(argmax_output, test_labels);
		
		std::cout << "Correct: " << correct << " / " << test_labels.size() << "\n";
		
	}
	
	
	
}





/*


struct convolutional_params
{
	size_t radius[2] = {1, 1};
	size_t pad[2] = {0, 0};
	size_t fmaps_out = 1;
};


void lenet()
{
	// LeNet-5
	
//	convolutional_params L1;
//	L1.radius = {2, 2};
//	L1.pad = {2, 2};
//	L1.fmaps_out = 6;


	// =========================================================
	// ===== INPUT
	// =========================================================
	
	skepu::Tensor3<float> processed_input(28, 28, 1); // 28x28 @ 1 feature map
	

	// =========================================================
	// == LAYER 1
	// == Convolution, 5x5 kernel, 2 pad, 1->6 fmaps
	// =========================================================
	
	auto hidden_1_conv = skepu::MapOverlap(convolutional_uf);
	auto hidden_1_activation = skepu::Map(sigmoid_uf);
	
	hidden_1_conv.setOverlap(2, 2, 0);
	hidden_1_conv.setEdgeMode(skepu::Edge::Pad);
	hidden_1_conv.setPad(0); // verify
	hidden_1_conv.setStride(1, 1, 0);
	
	skepu::Tensor3<float> weights_1(5, 5, 6);
	skepu::Matrix<int> connection_map_1(1, 6, 1); // 1 -> 6 fmaps
	skepu::Tensor3<float> hidden_state_1_val(28, 28, 6); // 28x28 @ 6 feature maps
	skepu::Tensor3<float> hidden_state_1_activation(28, 28, 6); // 28x28 @ 6 feature maps
	
	
	// =========================================================
	// == LAYER 2
	// == Pooling, 2x2 kernel, 2x2 stride, average
	// =========================================================
	
	auto hidden_2_pool = skepu::MapPool(generalized_pooling_uf);
	
	hidden_2_pool.setPoolSize(2, 2, 1);
	
	skepu::Tensor3<float> hidden_state_2(14, 14, 6); // 14x14 @ 6 feature maps
	
	
	// =========================================================
	// == LAYER 3
	// == Convolution 5x5 kernel, 0 pad, 6->16 fmaps
	// =========================================================
	
	auto hidden_3_conv = skepu::MapOverlap(convolutional_uf);
	auto hidden_3_activation = skepu::Map(sigmoid_uf);
	
	hidden_3_conv.setOverlap(2, 2, 6);
	hidden_3_conv.setEdgeMode(skepu::Edge::None);
	hidden_3_conv.setStride(1, 1, 0);
	
	skepu::Tensor3<float> weights_3(5, 5, 16);
	skepu::Matrix<int> connection_map_3(6, 16);// 6 -> 16 fmaps
	connection_map_3 = {
      1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1,
      1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1,
      1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1,
      0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1,
      0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,
      0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1
  };
	skepu::Tensor3<float> hidden_state_3_val(10, 10, 16); // 10x10 @ 16 feature maps
	skepu::Tensor3<float> hidden_state_3_activation(10, 10, 16); // 10x10 @ 16 feature maps
	
	
	// =========================================================
	// == LAYER 4
	// == Pooling, 2x2 kernel, 2x2 stride, average; flatten
	// =========================================================
	
	auto hidden_4_pool = skepu::MapPool(generalized_pooling_uf);
	
	hidden_4_pool.setPoolSize(2, 2, 1);
	
	skepu::Tensor3<float> hidden_state_4(5, 5, 16); // 5x5 @ 16 feature maps
	skepu::Vector<float> hidden_state_4_flattened(5 * 5 * 16); // 400
	
	
	// =========================================================
	// == LAYER 5
	// == Fully connected, from 400, to 120
	// =========================================================
	
	auto hidden_5_dense = skepu::Map(fully_connected_uf);
	auto hidden_5_activation = skepu::Map(sigmoid_uf);
	
	skepu::Matrix<float> weights_5(120, 400);
	skepu::Vector<float> bias_5(120);
	skepu::Vector<float> hidden_state_5_val(120);
	skepu::Vector<float> hidden_state_5_activation(120);
	
	
	// =========================================================
	// == LAYER 6
	// == Fully connected, from 120, to 84
	// =========================================================
	
	auto hidden_6_dense = skepu::Map(fully_connected_uf);
	auto hidden_6_activation = skepu::Map(sigmoid_uf);
	
	skepu::Matrix<float> weights_6(84, 120);
	skepu::Vector<float> bias_6(84);
	skepu::Vector<float> hidden_state_6_val(84);
	skepu::Vector<float> hidden_state_6_activation(84);
	
	
	// =========================================================
	// == LAYER 7
	// == Fully connected, from 84, to 10
	// =========================================================
	
	auto hidden_7_dense = skepu::Map(fully_connected_uf);
	
	skepu::Matrix<float> weights_7(10, 84);
	skepu::Vector<float> bias_7(10);
	skepu::Vector<float> hidden_state_7_val(10);
	
	
	// =========================================================
	// == OUTPUT
	// =========================================================
	
	
	
	
	
	
	
	std::cout << "Start forward pass!\n";
	
	hidden_1_conv(hidden_state_1_val, processed_input, weights_1, connection_map_1);
	hidden_1_activation(hidden_state_1_activation, hidden_state_1_val);
	
	hidden_2_pool(hidden_state_2, hidden_state_1_activation);
	
	hidden_3_conv(hidden_state_3_val, hidden_state_2, weights_3, connection_map_3);
	hidden_3_activation(hidden_state_3_activation, hidden_state_3_val);
	
	hidden_4_pool(hidden_state_4, hidden_state_3_activation);
	
	hidden_state_4_flattened = std::move(hidden_state_4);
	
	hidden_5_dense(hidden_state_5_val, bias_5, weights_5, hidden_state_4_flattened);
	hidden_5_activation(hidden_state_5_activation, hidden_state_5_val);
	
	hidden_6_dense(hidden_state_6_val, bias_6, weights_6, hidden_state_5_activation);
	hidden_6_activation(hidden_state_6_activation, hidden_state_6_val);
	
	hidden_7_dense(hidden_state_7_val, bias_7, weights_7, hidden_state_6_activation);
	
	std::cout << "End forward pass!\n";
	
}

*/




int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		skepu::external([&]{
			std::cout << "Usage: " << argv[0] << " size backend\n";});
		exit(1);
	}

	const size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	skepu::setGlobalBackendSpec(spec);
	
	mlp();
	//lenet();
	
	return 0;
}

