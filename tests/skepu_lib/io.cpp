// The stream wrappers of skepu-lib/io.hpp: skepu::io::cout writes values and containers, skepu::io::cin reads them. (The standard streams
// that they wrap are redirected to strings.)

#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <skepu-lib/io.hpp>
#include <test_helper_automatic_setup.hpp>
#include <iostream>
#include <sstream>
#include <string>

namespace
{
	int plus_one(int x)
	{
		return x + 1;
	}

	// Redirects std::cout to a string for the life of the object
	struct CaptureCout
	{
		std::ostringstream buffer;
		std::streambuf *old;
		CaptureCout() : old(std::cout.rdbuf(buffer.rdbuf())) {}
		~CaptureCout() { std::cout.rdbuf(old); }
		std::string str() { return buffer.str(); }
	};

	// Makes std::cin read from a string for the life of the object
	struct FeedCin
	{
		std::istringstream source;
		std::streambuf *old;
		explicit FeedCin(const std::string &text) : source(text), old(std::cin.rdbuf(source.rdbuf())) {}
		~FeedCin() { std::cin.rdbuf(old); }
	};
}


TEST_CASE("io: cout writes values")
{
	CaptureCout capture;
	skepu::io::cout << "x=" << 42 << ' ' << 1.5 << "\n";
	CHECK(capture.str() == "x=42 1.5\n");
}

TEST_CASE("io: cout writes a container that a skeleton made")
{
	skepu::Vector<int> in(4), out(4);
	for (size_t i = 0; i < 4; ++i)
		in(i) = int(i) + 1;
	skepu::Map<1>(plus_one)(out, in);   // (the elements of out are on the device of the backend)

	CaptureCout capture;
	skepu::io::cout << out;
	const std::string text = capture.str();
	CHECK(text.find("2 3 4 5") != std::string::npos);
}

TEST_CASE("io: cin reads values")
{
	FeedCin feed("7 3.5 word");
	int a = 0;
	double b = 0;
	std::string c;
	skepu::io::cin >> a >> b >> c;
	CHECK(a == 7);
	CHECK(b == 3.5);
	CHECK(c == "word");
}

TEST_CASE("io: cin reads a vector, which the backend then has")
{
	FeedCin feed("10 20 30 99");
	skepu::Vector<int> v(3), out(3);
	skepu::io::cin >> v;
	skepu::Map<1>(plus_one)(out, v);   // (the skeleton must see the values that were read)

	out.updateHost();
	CHECK(out(0) == 11);
	CHECK(out(1) == 21);
	CHECK(out(2) == 31);

	int next = 0;   // (only as many values as the vector has elements are read)
	skepu::io::cin >> next;
	CHECK(next == 99);
}

TEST_CASE("io: cin reads a matrix by rows, and several things in a row")
{
	FeedCin feed("5 1 2 3 4 5 6");
	int first = 0;
	skepu::Matrix<int> m(2, 3);
	skepu::io::cin >> first >> m;

	m.updateHost();
	CHECK(first == 5);
	CHECK(m(0, 0) == 1);
	CHECK(m(0, 2) == 3);
	CHECK(m(1, 0) == 4);
	CHECK(m(1, 2) == 6);
}

TEST_CASE("io: the streams are the standard ones")
{
	CHECK(&skepu::io::cout.m_stream == &std::cout);
	CHECK(&skepu::io::cin.m_stream == &std::cin);
}
