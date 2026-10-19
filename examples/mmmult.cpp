#include <iostream>
#include <cmath>
#include <algorithm>
#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/randomize.hpp>

// The results of a GPU may differ from those of the host in the last digits (for example by fused multiply-add)
static bool differs(float a, float b)
{
	return std::abs(a - b) > 1e-5f * std::max(1.0f, std::max(std::abs(a), std::abs(b)));
}


template<typename T>
T mmmult_f(skepu::Index2D idx, const skepu::Mat<T> lhs, const skepu::Mat<T> rhs)
{
	T res = 0;
	for (size_t i = 0; i < lhs.cols; ++i)
		res += lhs(idx.row, + i) * rhs(i, + idx.col);
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
		skepu::io::cout << "Usage: " << argv[0] << " height width inner backend\n";
		exit(1);
	}
	
	size_t height = atoi(argv[1]);
	size_t width = atoi(argv[2]);
	size_t inner = atoi(argv[3]);
	auto spec = skepu::BackendSpec{argv[4]};
	skepu::setGlobalBackendSpec(spec);
	
	skepu::Matrix<float> lhs(height, inner), rhs(inner, width), res(height, width), res2(height, width);
	skepu::randomize::uniform(lhs, 3, 9);
	skepu::randomize::uniform(rhs, 0, 9);
	
	skepu::io::cout << "lhs: " << lhs << "\n";
	skepu::io::cout << "rhs: " << rhs << "\n";

	res2.flush();
	directMM(lhs, rhs, res2);
	
	auto mmprod = skepu::Map(mmmult_f<float>);
	mmprod(res, lhs, rhs);
	
	skepu::external(skepu::read(res, res2), [&]
	{
		for (size_t i = 0; i < height; i++)
			for (size_t j = 0; j < width; j++)
				if (differs(res(i, j), res2(i, j)))
					std::cout << "Output error at index (" << i << "," << j << "): " << res2(i, j) << " vs " << res(i, j) << "\n";
	});

	return 0;
}
