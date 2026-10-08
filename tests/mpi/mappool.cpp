// MapPool on distributed containers. Same output on a single node and on any number of processes, with any backend.

#include <iostream>
#include <skepu>

int pool_sum(skepu::Pool2D<int> p)
{
	int sum = 0;
	for (int i = 0; i < p.si; ++i)
		for (int j = 0; j < p.sj; ++j)
			sum += p(i, j) * (i + 2 * j + 1);
	return sum;
}
int pool_max_indexed(skepu::Index2D index, skepu::Pool2D<int> p, int k)
{
	int best = p(0, 0);
	for (int i = 0; i < p.si; ++i)
		for (int j = 0; j < p.sj; ++j)
			best = p(i, j) > best ? p(i, j) : best;
	return best + (int)(index.row) * k + (int)(index.col);
}
int pool_sum_3d(skepu::Pool3D<int> p)
{
	int sum = 0;
	for (int i = 0; i < p.si; ++i)
		for (int j = 0; j < p.sj; ++j)
			for (int k = 0; k < p.sk; ++k)
				sum += p(i, j, k) * (i + 2 * j + 3 * k + 1);
	return sum;
}
int pool_sum_4d(skepu::Pool4D<int> p)
{
	int sum = 0;
	for (int i = 0; i < p.si; ++i)
		for (int j = 0; j < p.sj; ++j)
			for (int k = 0; k < p.sk; ++k)
				for (int l = 0; l < p.sl; ++l)
					sum += p(i, j, k, l) * (i + j + k + l + 1);
	return sum;
}

template<typename C>
static unsigned long long checksum(C &c)
{
	c.flush();
	unsigned long long h = 17;
	for (size_t i = 0; i < c.size(); ++i)
		h = h * 31 + (unsigned long long)(long long)c.getAddress()[i];
	return h;
}

static void pools(const size_t rows, const size_t cols)
{
	skepu::Matrix<int> in(rows, cols);
	skepu::external([&]
	{
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
				in(i, j) = (int)((i * 7 + j * 5) % 13) - 4;
	}, skepu::write(in));
	std::cout << "matrix " << rows << "x" << cols << "\n";

	// non-overlapping windows, 2x3
	{
		skepu::Matrix<int> out(rows / 2, cols / 3);
		skepu::MapPool(pool_sum).setPoolSize(2, 3).setStride(2, 3)(out, in);
		std::cout << " 2x3, stride 2x3: " << checksum(out) << "\n";
	}
	// overlapping windows: 3x3, stride 1x2
	{
		skepu::Matrix<int> out(rows - 2, (cols - 3) / 2 + 1);
		skepu::MapPool(pool_sum).setPoolSize(3, 3).setStride(1, 2)(out, in);
		std::cout << " 3x3, stride 1x2: " << checksum(out) << "\n";
	}
	// indices and a uniform; the pool size and the stride in one call
	{
		skepu::Matrix<int> out(rows / 2, cols / 2);
		skepu::MapPool(pool_max_indexed).setPoolSizeAndStride(2)(out, in, 5);
		std::cout << " indexed: " << checksum(out) << "\n";
	}
	// the input is larger than the windows need
	{
		skepu::Matrix<int> out((rows - 1) / 2, cols / 2);
		skepu::MapPool(pool_sum).setPoolSizeAndStride(2)(out, in);
		std::cout << " rest of the input unused: " << checksum(out) << "\n";
	}
}

int main()
{
	pools(14, 9);
	pools(8, 6);
	pools(7, 6); // fewer output rows than processes

	{
		const size_t si = 8, sj = 6, sk = 4;
		skepu::Tensor3<int> in(si, sj, sk), out(si / 2, sj / 2, sk / 2);
		skepu::external([&]
		{
			for (size_t i = 0; i < si; ++i)
				for (size_t j = 0; j < sj; ++j)
					for (size_t k = 0; k < sk; ++k)
						in(i, j, k) = (int)((i * 5 + j * 3 + k) % 9) - 2;
		}, skepu::write(in));
		skepu::MapPool(pool_sum_3d).setPoolSizeAndStride(2)(out, in);
		std::cout << "tensor3: " << checksum(out) << "\n";
	}
	{
		const size_t si = 6, sj = 4, sk = 4, sl = 2;
		skepu::Tensor4<int> in(si, sj, sk, sl), out(si / 2, sj / 2, sk / 2, sl / 2);
		skepu::external([&]
		{
			for (size_t i = 0; i < si; ++i)
				for (size_t j = 0; j < sj; ++j)
					for (size_t k = 0; k < sk; ++k)
						for (size_t l = 0; l < sl; ++l)
							in(i, j, k, l) = (int)((i * 7 + j * 5 + k * 3 + l) % 11) - 3;
		}, skepu::write(in));
		skepu::MapPool(pool_sum_4d).setPoolSizeAndStride(2)(out, in);
		std::cout << "tensor4: " << checksum(out) << "\n";
	}
	return 0;
}
