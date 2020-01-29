#define SKEPU_DEBUG 1

#include <iostream>
#include <skepu>






int uf(int a, int b)
{
	return a * b;
}

void test1(size_t Vsize, size_t Hsize, skepu::BackendSpec spec)
{
	auto pairs = skepu::MapPairs(uf);
	pairs.setBackend(spec);
	
	skepu::Vector<int> v1(Vsize, 3), h1(Hsize, 7);
	
	for (int i = 0; i < Vsize; ++i) v1(i) = i+1;
	for (int i = 0; i < Hsize; ++i) h1(i) = (i+1)*10;
	
	std::cout << "\nv1: " << v1 << "\nh1: " << h1 << "\n\n";
	
	skepu::Matrix<int> res(Vsize, Hsize);
	
	pairs(res, v1, h1);
	
	std::cout << "\nres: " << res;
}







int uf2(skepu::Index2D i, int ve1, int ve2, int ve3, int he1, int he2, skepu::Vec<int> test, int u1, int u2)
{
//	std:: cout << "(" << i.row << ", " << i.col << ")\n";
	return i.row + i.col + u1; // + ve1 + ve2 + ve3 + he1 + he2 + test.data[0] + u1 + u2;
}

void test2(size_t Vsize, size_t Hsize, skepu::BackendSpec spec)
{
	auto pairs2 = skepu::MapPairs<3, 2>(uf2);
	pairs2.setBackend(spec);
	
	skepu::Vector<int> v1(Vsize), v2(Vsize), v3(Vsize);
	skepu::Vector<int> h1(Hsize), h2(Hsize);
	skepu::Vector<int> testarg(10);
	skepu::Matrix<int> res(Vsize, Hsize);
	
//	pairs2.setMask(stencil);
	pairs2(res, v1, v2, v3, h1, h2, testarg, 10, 2);
	
	std::cout << "\nres: " << res;
}









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
	
	test1(Vsize, Hsize, spec);
	for (int i = 0; i < 10; ++i) std::cout << std::endl;
	test2(Vsize, Hsize, spec);
	
	return 0;
}

