#include <iostream>
#include <skepu>


[[skepu::userconstant]]
constexpr size_t SIZE = 10;

template<typename T, typename V>
T arr(skepu::Index1D row,  skepu::Mat<T> m, const skepu::Vec<T> v, skepu::Vec<V> v2 [[skepu::out]])
{
	T res = 0;
	for (size_t i = 0; i < SIZE; ++i)
	{
		res = m.data[row.i * SIZE + i] * v.data[i];
	}
	v2.data[row.i * 3] = row.i;
	return res;
}


int main()
{
	auto mvprod = skepu::Map<0>(arr<float, size_t>);
	
	skepu::Vector<float> v0(SIZE);
	skepu::Vector<size_t> v1(SIZE*3);
	skepu::Matrix<float> m1(SIZE, SIZE);
	m1.randomize(3, 9);
	
	// Sets v0 = 1 2 3 4 5...
	for(int i = 0; i < SIZE; ++i)
		v0[i] = (float)(i+10);
	
//	std::cout<<"v0: " <<v0 <<"\n";
//	std::cout<<"m1: " <<m1 <<"\n";
	
	for (auto backend : skepu::Backend::availableTypes())
	{
		std::cout << "---------[ " << backend << " ]---------\n";
		
		skepu::Vector<float> r(SIZE);
		mvprod.setBackend(skepu::BackendSpec{backend});
		
		auto s = mvprod(r, m1, v0, v1);
		std::cout << "CPU: " << s << "\n";
		std::cout << "r: " << r <<"\n";
		std::cout << "v1: " << v1 <<"\n";
		std::cout << "m1: " << m1 <<"\n";
	}
	
	return 0;
}
