#include <iostream>
#include <skepu>


template<typename T>
T arr(skepu::Index2D idx, const skepu::Mat<T> lhs, const skepu::Mat<T> rhs)
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
	int rows = lhs.size_i();
	int cols = rhs.size_j();
	
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < rows; ++j)
		{
			T sum{};
			for (int k = 0; k < cols; ++k)
				sum += lhs(i, k) * rhs(k, j);
			
			res(i, j) = sum;
		}
}

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		if(!skepu::cluster::mpi_rank())
			std::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
	
	size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{skepu::Backend::typeFromString(argv[2])};
	skepu::setGlobalBackendSpec(spec);
	
	skepu::Matrix<float> lhs(size, size), rhs(size, size), res(size, size), res2(size, size);
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
		
	auto mmprod = skepu::Map<0>(arr<float>);
	mmprod(res, lhs, rhs);
	
	res.flush();
	res2.flush();
	if(!skepu::cluster::mpi_rank())
	{
	//	std::cout << "res: " << res << "\n";
	//	std::cout << "res2: " << res2 << "\n";
		
		for (size_t i = 0; i < size; i++)
			for (size_t j = 0; j < size; j++)
				if (res(i, j) != res2(i, j))
					std::cout << "Output error at index (" << i << "," << j << "): " << res2(i, j) << " vs " << res(i, j) << "\n";
	}

	return 0;
}
