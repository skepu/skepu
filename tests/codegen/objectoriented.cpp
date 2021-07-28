#include <catch2/catch.hpp>

#include <skepu>


template<typename T>
class TestClass
{
	T _val;
	
public:
	
	TestClass(T val): _val{val}
	{}
	
	// Skeletons can be "hidden" inside classes like this.
	// The skeleton instance can also be declared static.
	skepu::Vector<T> scale(skepu::Vector<T> &v)
	{
		auto skel = skepu::Map<1>([](T e, T scale){ return e * scale; });
		skepu::Vector<T> res(v.size());
		return skel(res, v, this->_val);
	}
	
};



TEST_CASE("Object-oriented SkePU usage")
{
	const size_t size{100};
	
	skepu::Vector<int> v1(size);
	
	v1.flush();
	for (size_t i = 0; i < size; ++i)
	{
		v1(i) = i;
	}
	
	TestClass<int> test(5);
	
	auto res = test.scale(v1);
	
	res.flush();
	std::cout << "v1: " << v1 << "\nres: " << res << "\n\n";
	
}
