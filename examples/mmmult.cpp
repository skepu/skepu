#include <iostream>
#include <skepu>


template<typename T>
T mmmult_f(skepu::Index2D idx, const skepu::Mat<T> lhs, const skepu::Mat<T> rhs)
{
	T res = 0;
	for (size_t i = 0; i < lhs.cols; ++i)
		res += lhs.data[idx.row * lhs.cols + i] * rhs.data[i * rhs.cols + idx.col];
	return res;
}

// A helper function to calculate dense matrix-matrix product. Used to verify that the SkePU output is correct.
template<typename T>
void directMM(skepu::Matrix<T> &lhs, skepu::Matrix<T> &rhs, skepu::Matrix<T> &res)
{
	int rows  = res.size_i();
	int cols  = res.size_j();
	int inner = lhs.size_j();
	
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
		{
			T sum{};
			for (int k = 0; k < inner; ++k)
				sum += lhs(i, k) * rhs(k, j);
			
			res(i, j) = sum;
		}
}

int main(int argc, char *argv[])
{
	if (argc < 4)
	{
		if(!skepu::cluster::mpi_rank())
			std::cout << "Usage: " << argv[0] << " height width inner backend\n";
		exit(1);
	}
	
	size_t height = atoi(argv[1]);
	size_t width = atoi(argv[2]);
	size_t inner = atoi(argv[3]);
	auto spec = skepu::BackendSpec{skepu::Backend::typeFromString(argv[4])};
	skepu::setGlobalBackendSpec(spec);
	
	skepu::Matrix<float> lhs(height, inner), rhs(inner, width), res(height, width), res2(height, width);
	lhs.randomize(3, 9);
	rhs.randomize(0, 9);
	
	lhs.flush();
	rhs.flush();
/*	if(!skepu::cluster::mpi_rank())
	{
		std::cout << "lhs: " << lhs << "\n";
		std::cout << "rhs: " << rhs << "\n";
	}*/

	res2.flush();
	directMM(lhs, rhs, res2);
		
	auto mmprod = skepu::Map(mmmult_f<float>);
	mmprod(res, lhs, rhs);
	
	res.flush();
	res2.flush();
	if(!skepu::cluster::mpi_rank())
	{
	//	std::cout << "res: " << res << "\n";
	//	std::cout << "res2: " << res2 << "\n";
		
		for (size_t i = 0; i < height; i++)
			for (size_t j = 0; j < width; j++)
				if (res(i, j) != res2(i, j))
					std::cout << "Output error at index (" << i << "," << j << "): " << res2(i, j) << " vs " << res(i, j) << "\n";
	}

	return 0;
}
