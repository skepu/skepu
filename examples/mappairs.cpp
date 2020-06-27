#include <iostream>
#include <skepu>


int uf(int a, int b)
{
	return a * b;
}

void test1(size_t Vsize, size_t Hsize, skepu::BackendSpec spec)
{
	auto pairs = skepu::MapPairs(uf);
	
	skepu::Vector<int> v1(Vsize, 3), h1(Hsize, 7);
	
	for (int i = 0; i < Vsize; ++i) v1(i) = i+1;
	for (int i = 0; i < Hsize; ++i) h1(i) = (i+1)*10;
	
	std::cout << "\nv1: " << v1 << "\nh1: " << h1 << "\n\n";
	
	skepu::Matrix<int> res(Vsize, Hsize);
	
	pairs(res, v1, h1);
	
	std::cout << "\ntest 1 res: " << res;
}



int uf2(skepu::Index2D i, int ve1, int ve2, int ve3, int he1, int he2, skepu::Vec<int> test, int u1, int u2)
{
//	std:: cout << "(" << i.row << ", " << i.col << ")\n";
	return i.row + i.col + u1; // + ve1 + ve2 + ve3 + he1 + he2 + test.data[0] + u1 + u2;
}

void test2(size_t Vsize, size_t Hsize, skepu::BackendSpec spec)
{
	auto pairs2 = skepu::MapPairs<3, 2>(uf2);
	
	skepu::Vector<int> v1(Vsize), v2(Vsize), v3(Vsize);
	skepu::Vector<int> h1(Hsize), h2(Hsize);
	skepu::Vector<int> testarg(10);
	skepu::Matrix<int> res(Vsize, Hsize);
	
	pairs2(res, v1, v2, v3, h1, h2, testarg, 10, 2);
	
	std::cout << "\ntest 2 res: " << res << "\n";
}



int uf3(skepu::Index2D i, int u)
{
	return i.row + i.col + u;
}

void test3(size_t Vsize, size_t Hsize, skepu::BackendSpec spec)
{
	auto pairs3 = skepu::MapPairs<0, 0>(uf3);
	
	skepu::Matrix<int> res(Vsize, Hsize);
	
	pairs3(res, 2);
	
	std::cout << "\ntest 3 res: " << res << "\n";
}


int sum(int lhs, int rhs)
{
	return lhs + rhs;
}

int uf4(skepu::Index2D i, skepu::Vec<int> test, int u)
{
	return i.row + i.col + u;
}
/*
void testReduce(size_t Vsize, size_t Hsize, skepu::BackendSpec spec)
{
	auto pairs = skepu::MapPairsReduce(uf, sum);
	
	skepu::Vector<int> v1(Vsize, 3), h1(Hsize, 7);
	skepu::Vector<int> resV(Vsize), resH(Hsize);
	
	for (int i = 0; i < Vsize; ++i) v1(i) = i+1;
	for (int i = 0; i < Hsize; ++i) h1(i) = (i+1)*10;
	
	std::cout << "\nv1: " << v1 << "\nh1: " << h1 << "\n\n";
	
	pairs.setReduceMode(skepu::ReduceMode::ColWise);
	pairs(resH, v1, h1);
	std::cout << "\nresH: " << resH << "\n";
	
	pairs.setReduceMode(skepu::ReduceMode::RowWise);
	pairs(resV, v1, h1);
	std::cout << "\nresV: " << resV << "\n";
	
	
	// Test implicit dimensions
	{
		auto pairs4 = skepu::MapPairsReduce<0, 0>(uf4, sum);
		
		pairs4.setDefaultSize(Vsize, Hsize);
		pairs4.setReduceMode(skepu::ReduceMode::ColWise);
		pairs4(resH, v1, 0);
		std::cout << "\nresH: " << resH << "\n";
		
		pairs4.setReduceMode(skepu::ReduceMode::RowWise);
		pairs4(resV, v1, 0);
		std::cout << "\nresV: " << resV << "\n";
	}
}
*/




int main(int argc, char *argv[])
{
	if (argc < 4)
	{
		std::cout << "Usage: " << argv[0] << " size_v size_h backend\n";
		exit(1);
	}
	
	const size_t Vsize = atoi(argv[1]);
	const size_t Hsize = atoi(argv[2]);
	auto spec = skepu::BackendSpec{skepu::Backend::typeFromString(argv[3])};
	skepu::setGlobalBackendSpec(spec);
	
	test1(Vsize, Hsize, spec);
	for (int i = 0; i < 10; ++i) std::cout << std::endl;
	test2(Vsize, Hsize, spec);
	
	test3(Vsize, Hsize, spec);
	
//	testReduce(Vsize, Hsize, spec);
	
	return 0;
}
