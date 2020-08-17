#include <iostream>
#include <skepu>


template<typename T>
T mmmult_f(const skepu::MatRow<T> ar, const skepu::MatCol<T> bc)
{
	T res = 0;
	for (size_t k = 0; k < ar.cols; ++k)
		res += ar(k) * bc(k);
	return res;
}

// A helper function to calculate dense matrix-matrix product. Used to verify that the SkePU output is correct.
template<typename T>
void directMM(skepu::Matrix<T> &lhs, skepu::Matrix<T> &rhs, skepu::Matrix<T> &res)
{
	int rows  = res.size_i();
	int cols  = res.size_j();
	int inner = lhs.total_cols();
	
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
		skepu::external([&]{
			std::cout << "Usage: " << argv[0] << " height width inner backend\n";
		});
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
	
	skepu::external(
		skepu::read(lhs,rhs),
		[&]{
			if(!skepu::cluster::mpi_rank())
			{
				std::cout << "lhs: " << lhs << "\n";
				std::cout << "rhs: " << rhs << "\n";
			}
		});

	skepu::external(
		skepu::read(lhs,rhs),
		[&]{
			directMM(lhs, rhs, res2);
		},
		skepu::write(res2));
		
	auto mmprod = skepu::Map(mmmult_f<float>);
	mmprod(res, lhs, rhs);
	
	skepu::external(
		skepu::read(res, res2),
		[&]{
			if(!skepu::cluster::mpi_rank())
			{
				for (size_t i = 0; i < height; i++)
					for (size_t j = 0; j < width; j++)
						if (res(i, j) != res2(i, j))
							std::cout << "Output error at index (" << i << "," << j << "): "
								<< res2(i, j) << " vs " << res(i, j) << "\n";
			}
		});

	return 0;
}
