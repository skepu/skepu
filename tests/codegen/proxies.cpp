#include <catch2/catch.hpp>

#include <skepu>


template<typename T>
T test1_f(
	const skepu::Vec<T> v,
	const skepu::Mat<T> m,
	const skepu::MatRow<T> mr,
	const skepu::MatCol<T> mc,
	const skepu::Ten3<T> t3,
	const skepu::Ten4<T> t4)
{
	T a  = v(0);
	T b  = m(0,0);
	T c  = mr(0);
	T d  = mc(0);
	T e  = t3(0,0,0);
	T f  = t4(0,0,0,0);
	
	return a+b+c+d+e+f;
}

auto skel = skepu::Map(test1_f<float>);



// TEST PROXIES

int test1_proxy(const skepu::Vec<int> vec)
{
	return vec.size + vec(1);
}

int test2_proxy(const skepu::Mat<int> mat)
{
	return mat.rows * 100 + mat.cols + mat(1, 1);
}

int test3_proxy(const skepu::Ten3<int> ten3)
{
	return ten3.size_i * 10000 + ten3.size_j * 100 + ten3.size_k + ten3(1, 1, 1);
}

int test4_proxy(const skepu::Ten4<int> ten4)
{
	return ten4.size_i * 1000000 + ten4.size_j * 10000 + ten4.size_k * 100 + ten4.size_l + ten4(1, 1, 1, 1);
}

auto skel1_p = skepu::Map(test1_proxy);
auto skel2_p = skepu::Map(test2_proxy);
auto skel3_p = skepu::Map(test3_proxy);
auto skel4_p = skepu::Map(test4_proxy);



TEST_CASE("Container proxy reads in user function")
{
	size_t size{100};
	
	skepu::Vector<float> v(size), r(size);
	skepu::Matrix<float> m(size, size);
	skepu::Tensor3<float> t3(size, size, 1);
	skepu::Tensor4<float> t4(size, size, 1, 1);
	
	v(0) = 1;
	m(0,0) = 2;
	t3(0,0,0) = 3;
	t4(0,0,0,0) = 4;
	
	skel(r, v, m, m, m, t3, t4);
	r.flush();
	
	CHECK(r(0) == 1+2+2+2+3+4);
}
	
TEST_CASE("Container proxy sizes in user function")
{	
	constexpr int A = 903;
	skepu::Vector<int> dummy(1);

	const size_t size{100};
	
	skepu::Vector<int> vec(size, A);
	skepu::Matrix<int> mat(size, size+1, A);
	skepu::Tensor3<int> ten3(size, size+1, size+2, A);
	skepu::Tensor4<int> ten4(size, size+1, size+2, size+3, A);
	
	std::cout << skepu::is_skepu_container_proxy<decltype(ten3.hostProxy())>::value << std::endl;
	
	skel1_p(dummy, vec);
	dummy.flush();
	CHECK(dummy(0) == vec.size() + A);
	
	skel2_p(dummy, mat);
	dummy.flush();
	CHECK(dummy(0) == mat.total_rows() * 100 + mat.total_cols() + A);
	
	skel3_p(dummy, ten3);
	dummy.flush();
	CHECK(dummy(0) == ten3.size_i() * 10000 + ten3.size_j() * 100 + ten3.size_k() + A);
	
	skel4_p(dummy, ten4);
	dummy.flush();
	CHECK(dummy(0) == ten4.size_i() * 1000000 + ten4.size_j() * 10000 + ten4.size_k() * 100 + ten4.size_l() + A);
	
}
