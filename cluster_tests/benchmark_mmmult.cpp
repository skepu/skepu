#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>

#include "common.hpp"

#ifdef BENCHMARK_MMMULT


namespace benchmark_mmmult {

		template<typename T>
		T seq_init_impl(skepu2::Index2D index) {
				return (T)(index.row + index.col);
		}

		auto seq_init = skepu2::Map<0>(seq_init_impl<float>);

template<typename T>
T arr(skepu2::Index2D idx, const skepu2::Mat<T> lhs, const skepu2::Mat<T> rhs)
{
	T res = 0;
	for (size_t i = 0; i < lhs.cols; ++i)
		res += lhs.data[idx.row * lhs.cols + i] * rhs.data[i * rhs.cols + idx.col];
	return res;
}

// A helper function to calculate dense matrix-matrix product. Used to verify that the SkePU output is correct.
template<typename T>
void directMM(skepu2::Matrix<T> &lhs, skepu2::Matrix<T> &rhs, skepu2::Matrix<T> &res)
{
	int rows = lhs.total_rows();
	int cols = rhs.total_cols();

	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < rows; ++j)
		{
			T sum{};
			for (int k = 0; k < cols; ++k)
				sum += lhs(i, k) * rhs(k, j);

			res.set(i, j, sum);
		}
}

auto mmprod = skepu2::Map<0>(arr<float>);

void mmmult(skepu2::Matrix<float> &lhs, skepu2::Matrix<float> &rhs, skepu2::Matrix<float> &res)
{
	mmprod(res, lhs, rhs);
}


		TEST_CASE("Test mmmult") {
				size_t size = 128;
				skepu2::Matrix<float> lhs(size, size), rhs(size, size), res(size, size);//, res2(size, size);
				seq_init(lhs);
				seq_init(rhs);
				seq_init(res);

				//	directMM(lhs, rhs, res2);
//				mmmult(lhs, rhs, res);

				// for (size_t i = 0; i < size; i++)
				//			for (size_t j = 0; j < size; j++)
				//					CHECK(res(i, j) == res2(i, j));
		}

}

#endif
