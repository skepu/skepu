#include <skepu>
#include <skepu-lib/io.hpp>
#include "csv_io.hpp"

#include <vector>
#include <fstream>




int main(int argc, char *argv[])
{
	skepu::Vector<float> vec(288);
  skepu::Matrix<float> mat(3, 3);
  skepu::Tensor3<float> ten3(3, 3, 32);
  skepu::Tensor4<float> ten4(3, 3, 32, 64);
	
  std::vector<float> v1 = deserialize_csv_helper<float>("data/pretrained/0_kernel_weights_single_file.csv");
  std::vector<float> v2 = deserialize_csv_helper<float>("data/pretrained/2_kernel_weights_single_file.csv");
  std::vector<float> v3 = deserialize_csv_helper<float>("data/pretrained/6_Dense_weights_single_file.csv");
  
  auto it = load_from_vector(vec, v1, v1.begin());
  
	skepu::io::cout << vec << "\n";
	skepu::io::cout << mat << "\n";
	skepu::io::cout << ten3 << "\n";
	skepu::io::cout << ten4 << "\n";
	
	return 0;
}
