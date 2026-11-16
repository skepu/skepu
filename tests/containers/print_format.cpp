// The stream output of the containers and of the proxies of user functions: Type[dims]: and the elements.
#include "../../external/catch2/catch.hpp"
#include <sstream>
#include <skepu>

namespace
{
template<typename T>
std::string text(T &&x)
{
	std::ostringstream os;
	os << x;
	return os.str();
}
}

TEST_CASE("print_format: containers")
{
	skepu::Vector<int> v{1, 2, 3};
	skepu::Matrix<int> m(2, 3, 7);
	REQUIRE(text(v) == "Vector[3]: 1 2 3 ");
	REQUIRE(text(m) == "Matrix[2×3]:\n7 7 7 \n7 7 7 \n");

	skepu::Tensor3<int> t3(1, 2, 2, "", 5);
	skepu::Tensor4<int> t4(1, 1, 1, 2, "", 5);
	REQUIRE(text(t3).rfind("Tensor3[1×2×2]:\n", 0) == 0);
	REQUIRE(text(t4).rfind("Tensor4[1×1×1×2]:\n", 0) == 0);
}

TEST_CASE("print_format: proxies of user functions")
{
	int data[6] = {0, 1, 2, 3, 4, 5};
	REQUIRE(text(skepu::Vec<int>(data, 3)) == "Vec[3]: 0 1 2 ");
	REQUIRE(text(skepu::Mat<int>{data, 2, 3}) == "Mat[2×3]:\n0 1 2 \n3 4 5 \n");
	REQUIRE(text(skepu::MatRow<int>{data + 3, 3}) == "MatRow[3]: 3 4 5 ");
	REQUIRE(text(skepu::MatCol<int>{data + 1, 2, 3}) == "MatCol[2]: 1 4 ");
	REQUIRE(text(skepu::Ten3<int>(data, 1, 2, 3)).rfind("Ten3[1×2×3]:\n", 0) == 0);
	REQUIRE(text(skepu::Ten4<int>(data, 1, 1, 2, 3)).rfind("Ten4[1×1×2×3]:\n", 0) == 0);
}
