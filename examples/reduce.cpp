#include <iostream>
#include <skepu>

float plus_f(float a, float b)
{
	return a + b;
}

float max_f(float a, float b)
{
	return (a > b) ? a : b;
}

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		if(!skepu::cluster::mpi_rank())
			std::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
	
	const size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{skepu::Backend::typeFromString(argv[2])};
	
	skepu::Matrix<float> m(size / 2, size / 2);
	skepu::Vector<float> v(size), rv(size / 2);
	m.randomize(0, 10);
	v.randomize(0, 10);
	
	m.flush();
	v.flush();
	if(!skepu::cluster::mpi_rank())
	{
		std::cout << "v: " << v << "\n";
		std::cout << "m: " << m << "\n";
	}

	auto sum = skepu::Reduce(plus_f);
	sum.setBackend(spec);
	
	auto max_sum = skepu::Reduce(plus_f, max_f);
	max_sum.setBackend(spec);
	
	// With containers
	float r = sum(v);
	if(!skepu::cluster::mpi_rank())
		std::cout << "Reduce: r = " << r << "\n";
	
	r = sum(m);
	if(!skepu::cluster::mpi_rank())
		std::cout << "Reduce: r = " << r << "\n";
	
	sum.setReduceMode(skepu::ReduceMode::RowWise);
	sum(rv, m);
	rv.flush();
	if(!skepu::cluster::mpi_rank())
		std::cout << "Reduce: r = " << rv << "\n";
	
	sum.setReduceMode(skepu::ReduceMode::ColWise);
	sum(rv, m);
	rv.flush();
	if(!skepu::cluster::mpi_rank())
		std::cout << "Reduce: r = " << rv << "\n";
	
	
	
	// 2D reduce
	max_sum.setReduceMode(skepu::ReduceMode::RowWise);
	r = max_sum(m);
	if(!skepu::cluster::mpi_rank())
		std::cout << "Reduce 2D max row-sum: r = " << r << "\n";
	
	max_sum.setReduceMode(skepu::ReduceMode::ColWise);
	r = max_sum(m);
	if(!skepu::cluster::mpi_rank())
		std::cout << "Reduce 2D max col-sum: r = " << r << "\n";
	
	r = max_sum(v);
	if(!skepu::cluster::mpi_rank())
		std::cout << "Reduce: r = " << r << "\n";
	
	return 0;
}

