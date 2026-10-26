// A probe of the data coherence of the containers when the program has more than one device backend (CUDA and OpenCL)
// and hands containers from one backend to another. It is a program, not a Catch2 test: each scenario is run on its own
// (coherence_probe <scenario>), prints "PASS" or "FAIL: <what>", and a crash or a hang of one scenario does not hide the others.
// coherence_probe list    prints the scenarios.
// The containers are read on the host with flush() and written with skepu::external, the API of the user guide.
#define SKEPU_ENABLE_EXCEPTIONS

#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <skepu>

int f_inc(int a) { return a + 1; }
int f_add(int a, int b) { return a + b; }
int f_dbl(int a) { return a * 2; }
int f_index(skepu::Index1D i) { return (int)i.i; }
int f_reverse_lookup(skepu::Index1D i, const skepu::Vec<int> table) { return table.data[table.size - 1 - i.i]; }

auto inc = skepu::Map<1>(f_inc);
auto dbl = skepu::Map<1>(f_dbl);
auto add = skepu::Map<2>(f_add);
auto make_index = skepu::Map<0>(f_index);
auto reverse_lookup = skepu::Map<0>(f_reverse_lookup);
auto sum = skepu::Reduce(f_add);

static constexpr size_t N = 1000;
static constexpr size_t R = 20, C = 30;

template<typename Skeleton>
static Skeleton &on(Skeleton &skeleton, const char *backend)
{
	skeleton.setBackend(skepu::BackendSpec{backend});
	return skeleton;
}

static void init(skepu::Vector<int> &v, int base)
{
	skepu::external([&] { for (size_t i = 0; i < v.size(); ++i) v(i) = base + (int)i; }, skepu::write(v));
}

static void init(skepu::Matrix<int> &m, int base)
{
	skepu::external([&] { for (size_t i = 0; i < m.total_rows(); ++i) for (size_t j = 0; j < m.total_cols(); ++j) m(i, j) = base + (int)(i * 100 + j); }, skepu::write(m));
}

// "" if the vector (read on the host with flush) has the expected values, else what is wrong
static std::string verify(skepu::Vector<int> &v, const std::function<int(size_t)> &expected)
{
	v.flush();
	size_t wrong = 0, first = 0;
	for (size_t i = 0; i < v.size(); ++i)
		if (v(i) != expected(i))
			if (wrong++ == 0) first = i;
	if (!wrong)
		return "";
	std::stringstream ss;
	ss << wrong << " of " << v.size() << " elements are wrong; the first is [" << first << "]: got " << v(first) << ", expected " << expected(first);
	return ss.str();
}

static std::string verify(skepu::Matrix<int> &m, const std::function<int(size_t, size_t)> &expected)
{
	m.flush();
	size_t wrong = 0, fi = 0, fj = 0;
	for (size_t i = 0; i < m.total_rows(); ++i)
		for (size_t j = 0; j < m.total_cols(); ++j)
			if (m(i, j) != expected(i, j))
				if (wrong++ == 0) { fi = i; fj = j; }
	if (!wrong)
		return "";
	std::stringstream ss;
	ss << wrong << " of " << m.size() << " elements are wrong; the first is [" << fi << "," << fj << "]: got " << m(fi, fj) << ", expected " << expected(fi, fj);
	return ss.str();
}

// Joins the descriptions of the checks that failed
static std::string both(std::initializer_list<std::pair<const char *, std::string>> checks)
{
	std::string result;
	for (auto &c : checks)
		if (!c.second.empty())
			result += std::string(result.empty() ? "" : "; ") + c.first + ": " + c.second;
	return result;
}

struct Scenario { std::string name, description; std::function<std::string()> run; };

static std::vector<Scenario> scenarios()
{
	std::vector<Scenario> s;
	auto E = [](size_t i, int k) { return (int)i + k; }; // (the value at i after k increments of an initial value of i)

	// ---- Controls: one backend, and the host as the meeting point
	s.push_back({"c01_cuda_chain", "a -> b -> c on CUDA only", [=]
	{
		skepu::Vector<int> a(N), b(N), c(N); init(a, 0);
		on(inc, "cuda")(b, a); on(inc, "cuda")(c, b);
		return verify(c, [&](size_t i) { return E(i, 2); });
	}});
	s.push_back({"c02_opencl_chain", "a -> b -> c on OpenCL only", [=]
	{
		skepu::Vector<int> a(N), b(N), c(N); init(a, 0);
		on(inc, "opencl")(b, a); on(inc, "opencl")(c, b);
		return verify(c, [&](size_t i) { return E(i, 2); });
	}});
	s.push_back({"c03_cuda_flush_opencl", "CUDA writes b, b.flush(), OpenCL reads b", [=]
	{
		skepu::Vector<int> a(N), b(N), c(N); init(a, 0);
		on(inc, "cuda")(b, a); b.flush(); on(inc, "opencl")(c, b);
		return verify(c, [&](size_t i) { return E(i, 2); });
	}});
	s.push_back({"c04_opencl_flush_cuda", "OpenCL writes b, b.flush(), CUDA reads b", [=]
	{
		skepu::Vector<int> a(N), b(N), c(N); init(a, 0);
		on(inc, "opencl")(b, a); b.flush(); on(inc, "cuda")(c, b);
		return verify(c, [&](size_t i) { return E(i, 2); });
	}});
	s.push_back({"c05_cuda_dealloc_opencl", "CUDA writes b, b.flush(Dealloc), OpenCL reads b", [=]
	{
		skepu::Vector<int> a(N), b(N), c(N); init(a, 0);
		on(inc, "cuda")(b, a); b.flush(skepu::FlushMode::Dealloc); on(inc, "opencl")(c, b);
		return verify(c, [&](size_t i) { return E(i, 2); });
	}});

	// ---- A device backend hands a container to another without a flush
	s.push_back({"d01_cuda_to_opencl", "CUDA writes b, OpenCL reads b", [=]
	{
		skepu::Vector<int> a(N), b(N), c(N); init(a, 0);
		on(inc, "cuda")(b, a); on(inc, "opencl")(c, b);
		return both({{"c", verify(c, [&](size_t i) { return E(i, 2); })}, {"b", verify(b, [&](size_t i) { return E(i, 1); })}});
	}});
	s.push_back({"d02_opencl_to_cuda", "OpenCL writes b, CUDA reads b", [=]
	{
		skepu::Vector<int> a(N), b(N), c(N); init(a, 0);
		on(inc, "opencl")(b, a); on(inc, "cuda")(c, b);
		return both({{"c", verify(c, [&](size_t i) { return E(i, 2); })}, {"b", verify(b, [&](size_t i) { return E(i, 1); })}});
	}});
	s.push_back({"d03_cuda_to_cpu", "CUDA writes b, CPU reads b", [=]
	{
		skepu::Vector<int> a(N), b(N), c(N); init(a, 0);
		on(inc, "cuda")(b, a); on(inc, "cpu")(c, b);
		return verify(c, [&](size_t i) { return E(i, 2); });
	}});
	s.push_back({"d04_opencl_to_cpu", "OpenCL writes b, CPU reads b", [=]
	{
		skepu::Vector<int> a(N), b(N), c(N); init(a, 0);
		on(inc, "opencl")(b, a); on(inc, "cpu")(c, b);
		return verify(c, [&](size_t i) { return E(i, 2); });
	}});
	s.push_back({"d05_cpu_to_cuda", "CPU writes b, CUDA reads b", [=]
	{
		skepu::Vector<int> a(N), b(N), c(N); init(a, 0);
		on(inc, "cpu")(b, a); on(inc, "cuda")(c, b);
		return verify(c, [&](size_t i) { return E(i, 2); });
	}});
	s.push_back({"d06_cpu_to_opencl", "CPU writes b, OpenCL reads b", [=]
	{
		skepu::Vector<int> a(N), b(N), c(N); init(a, 0);
		on(inc, "cpu")(b, a); on(inc, "opencl")(c, b);
		return verify(c, [&](size_t i) { return E(i, 2); });
	}});
	s.push_back({"d07_openmp_cuda_openmp", "OpenMP writes, CUDA writes, OpenMP reads", [=]
	{
		skepu::Vector<int> a(N), b(N), c(N), d(N); init(a, 0);
		on(inc, "openmp")(b, a); on(inc, "cuda")(c, b); on(inc, "openmp")(d, c);
		return verify(d, [&](size_t i) { return E(i, 3); });
	}});
	s.push_back({"d08_cuda_to_opencl_reduce", "CUDA writes b, OpenCL reduces b to a scalar", [=]
	{
		skepu::Vector<int> a(N), b(N); init(a, 0);
		on(inc, "cuda")(b, a);
		const int got = on(sum, "opencl")(b);
		const int expected = (int)(N * (N - 1) / 2 + N);
		return got == expected ? std::string() : "sum is " + std::to_string(got) + ", expected " + std::to_string(expected);
	}});
	s.push_back({"d09_cuda_to_opencl_proxy", "CUDA writes a table, OpenCL reads it through a Vec proxy", [=]
	{
		skepu::Vector<int> a(N), table(N), out(N); init(a, 0);
		on(inc, "cuda")(table, a);
		on(reverse_lookup, "opencl")(out, table);
		return verify(out, [&](size_t i) { return (int)(N - 1 - i) + 1; });
	}});
	s.push_back({"d10_index_cuda_then_opencl", "CUDA makes c with an index-only Map<0>, OpenCL reads c", [=]
	{
		skepu::Vector<int> c(N), d(N);
		on(make_index, "cuda")(c); on(inc, "opencl")(d, c);
		return verify(d, [&](size_t i) { return E(i, 1); });
	}});
	s.push_back({"d11_matrix_cuda_to_opencl", "Matrix: CUDA writes b, OpenCL reads b", [=]
	{
		skepu::Matrix<int> a(R, C), b(R, C), c(R, C); init(a, 0);
		on(inc, "cuda")(b, a); on(inc, "opencl")(c, b);
		return verify(c, [&](size_t i, size_t j) { return (int)(i * 100 + j) + 2; });
	}});
	s.push_back({"d12_matrix_opencl_to_cuda", "Matrix: OpenCL writes b, CUDA reads b", [=]
	{
		skepu::Matrix<int> a(R, C), b(R, C), c(R, C); init(a, 0);
		on(inc, "opencl")(b, a); on(inc, "cuda")(c, b);
		return verify(c, [&](size_t i, size_t j) { return (int)(i * 100 + j) + 2; });
	}});

	// ---- In place, and repeated switching
	s.push_back({"p01_pingpong_inplace", "v = inc(v), 10 times, alternating CUDA and OpenCL", [=]
	{
		skepu::Vector<int> v(N); init(v, 0);
		for (int k = 0; k < 10; ++k) on(inc, (k % 2) ? "opencl" : "cuda")(v, v);
		return verify(v, [&](size_t i) { return E(i, 10); });
	}});
	s.push_back({"p02_pingpong_two_vectors", "b = inc(a), a = inc(b), ..., 10 times, alternating CUDA and OpenCL", [=]
	{
		skepu::Vector<int> a(N), b(N); init(a, 0);
		for (int k = 0; k < 10; ++k)
		{
			if (k % 2 == 0) on(inc, (k % 4 == 0) ? "cuda" : "opencl")(b, a);
			else            on(inc, (k % 4 == 1) ? "cuda" : "opencl")(a, b);
		}
		return verify(a, [&](size_t i) { return E(i, 10); });
	}});
	s.push_back({"p03_alternate_same_input", "b = inc(a) on CUDA, c = inc(a) on OpenCL, a modified by the host, again", [=]
	{
		skepu::Vector<int> a(N), b(N), c(N); init(a, 0);
		on(inc, "cuda")(b, a); on(inc, "opencl")(c, a);
		auto first = both({{"b", verify(b, [&](size_t i) { return E(i, 1); })}, {"c", verify(c, [&](size_t i) { return E(i, 1); })}});
		init(a, 100);
		on(inc, "cuda")(b, a); on(inc, "opencl")(c, a);
		return both({{"first round", first}, {"b after the write of a", verify(b, [&](size_t i) { return E(i, 101); })}, {"c after the write of a", verify(c, [&](size_t i) { return E(i, 101); })}});
	}});
	s.push_back({"p04_stale_cuda_copy", "CUDA reads v, OpenCL then updates v in place, CUDA reads v again", [=]
	{
		skepu::Vector<int> v(N), w(N), w2(N); init(v, 0);
		on(inc, "cuda")(w, v);       // (CUDA has a copy of v)
		on(inc, "opencl")(v, v);     // (v is now i+1; the CUDA copy of v is stale)
		on(inc, "cuda")(w2, v);      // (expected i+2)
		return verify(w2, [&](size_t i) { return E(i, 2); });
	}});
	s.push_back({"p05_stale_opencl_copy", "OpenCL reads v, CUDA then updates v in place, OpenCL reads v again", [=]
	{
		skepu::Vector<int> v(N), w(N), w2(N); init(v, 0);
		on(inc, "opencl")(w, v);
		on(inc, "cuda")(v, v);
		on(inc, "opencl")(w2, v);
		return verify(w2, [&](size_t i) { return E(i, 2); });
	}});

	// ---- Both backends write the same container
	s.push_back({"w01_cuda_then_opencl_write", "c = inc(a) on CUDA, then c = dbl(a) on OpenCL (overwrite), read c on the host", [=]
	{
		skepu::Vector<int> a(N), c(N); init(a, 0);
		on(inc, "cuda")(c, a); on(dbl, "opencl")(c, a);
		return verify(c, [&](size_t i) { return 2 * (int)i; });
	}});
	s.push_back({"w02_opencl_then_cuda_write", "c = inc(a) on OpenCL, then c = dbl(a) on CUDA (overwrite), read c on the host", [=]
	{
		skepu::Vector<int> a(N), c(N); init(a, 0);
		on(inc, "opencl")(c, a); on(dbl, "cuda")(c, a);
		return verify(c, [&](size_t i) { return 2 * (int)i; });
	}});
	s.push_back({"w03_write_read_write", "CUDA writes c, OpenCL reads c, OpenCL overwrites c, CUDA reads c", [=]
	{
		skepu::Vector<int> a(N), c(N), d(N), e(N); init(a, 0);
		on(inc, "cuda")(c, a); on(inc, "opencl")(d, c); on(dbl, "opencl")(c, a); on(inc, "cuda")(e, c);
		return both({{"d", verify(d, [&](size_t i) { return E(i, 2); })}, {"e", verify(e, [&](size_t i) { return 2 * (int)i + 1; })}});
	}});

	// ---- Containers as values
	s.push_back({"v01_copy_assign", "CUDA writes b, c = b, OpenCL reads c", [=]
	{
		skepu::Vector<int> a(N), b(N), c(N); init(a, 0);
		on(inc, "cuda")(b, a); c = b; on(inc, "opencl")(a, c);
		return verify(a, [&](size_t i) { return E(i, 2); });
	}});
	s.push_back({"v02_swap", "CUDA writes b, swap(b, c), OpenCL reads c", [=]
	{
		skepu::Vector<int> a(N), b(N), c(N), d(N); init(a, 0); init(c, 50);
		on(inc, "cuda")(b, a); b.swap(c); on(inc, "opencl")(d, c);
		return both({{"d", verify(d, [&](size_t i) { return E(i, 2); })}, {"b", verify(b, [&](size_t i) { return (int)i + 50; })}});
	}});
	s.push_back({"v03_scope_end", "a temporary written on CUDA and read on OpenCL is destroyed, the result is read afterwards", [=]
	{
		skepu::Vector<int> a(N), out(N); init(a, 0);
		{
			skepu::Vector<int> tmp(N);
			on(inc, "cuda")(tmp, a); on(inc, "opencl")(out, tmp);
		}
		return verify(out, [&](size_t i) { return E(i, 2); });
	}});
	s.push_back({"v04_host_write_after_cuda", "CUDA writes b, the host writes b (external), OpenCL reads b", [=]
	{
		skepu::Vector<int> a(N), b(N), c(N); init(a, 0);
		on(inc, "cuda")(b, a); init(b, 1000); on(inc, "opencl")(c, b);
		return verify(c, [&](size_t i) { return E(i, 1001); });
	}});
	s.push_back({"v05_host_write_after_opencl", "OpenCL writes b, the host writes b (external), CUDA reads b", [=]
	{
		skepu::Vector<int> a(N), b(N), c(N); init(a, 0);
		on(inc, "opencl")(b, a); init(b, 1000); on(inc, "cuda")(c, b);
		return verify(c, [&](size_t i) { return E(i, 1001); });
	}});
	s.push_back({"v06_binary_two_backends", "x written on CUDA, y written on OpenCL, z = add(x, y) on CPU", [=]
	{
		skepu::Vector<int> a(N), x(N), y(N), z(N); init(a, 0);
		on(inc, "cuda")(x, a); on(dbl, "opencl")(y, a); on(add, "cpu")(z, x, y);
		return verify(z, [&](size_t i) { return 3 * (int)i + 1; });
	}});
	s.push_back({"v07_binary_on_cuda", "x written on OpenCL, y written on CUDA, z = add(x, y) on CUDA", [=]
	{
		skepu::Vector<int> a(N), x(N), y(N), z(N); init(a, 0);
		on(inc, "opencl")(x, a); on(dbl, "cuda")(y, a); on(add, "cuda")(z, x, y);
		return verify(z, [&](size_t i) { return 3 * (int)i + 1; });
	}});

	// ---- Which calls between the two backends make it right
	auto handoff = [&s, E](const char *name, const char *description, std::function<void(skepu::Vector<int> &)> between)
	{
		s.push_back({name, description, [=]
		{
			skepu::Vector<int> a(N), b(N), c(N); init(a, 0);
			on(inc, "cuda")(b, a); between(b); on(inc, "opencl")(c, b);
			return verify(c, [&](size_t i) { return E(i, 2); });
		}});
	};
	handoff("h01_updateHost", "CUDA writes b, b.updateHost(), OpenCL reads b", [](skepu::Vector<int> &b) { b.updateHost(); });
	handoff("h02_flush", "CUDA writes b, b.flush(), OpenCL reads b", [](skepu::Vector<int> &b) { b.flush(); });
	handoff("h03_updateHostAndInvalidate", "CUDA writes b, b.updateHostAndInvalidateDevice(), OpenCL reads b", [](skepu::Vector<int> &b) { b.updateHostAndInvalidateDevice(); });
	// (by contract these two throw the changes on the device away: the OpenCL backend then reads the old contents of the host, 0 + 1 = 1)
	auto discard = [&s](const char *name, const char *description, std::function<void(skepu::Vector<int> &)> between)
	{
		s.push_back({name, description, [=]
		{
			skepu::Vector<int> a(N), b(N), c(N); init(a, 0);
			on(inc, "cuda")(b, a); between(b); on(inc, "opencl")(c, b);
			return verify(c, [&](size_t) { return 1; });
		}});
	};
	discard("h04_invalidateOnly", "CUDA writes b, b.invalidateDeviceData() (no update): the device changes are discarded by contract", [](skepu::Vector<int> &b) { b.invalidateDeviceData(); });
	discard("h05_releaseOnly", "CUDA writes b, b.releaseDeviceAllocations() (no update): the device changes are discarded by contract", [](skepu::Vector<int> &b) { b.releaseDeviceAllocations(); });
	handoff("h06_external_read", "CUDA writes b, skepu::external(read(b), {}), OpenCL reads b", [](skepu::Vector<int> &b) { skepu::external(skepu::read(b), [] {}); });

	auto stale = [&s, E](const char *name, const char *description, const char *first, const char *second, std::function<void(skepu::Vector<int> &)> between)
	{
		s.push_back({name, description, [=]
		{
			skepu::Vector<int> v(N), w(N), w2(N); init(v, 0);
			on(inc, first)(w, v); on(inc, second)(v, v); between(v); on(inc, first)(w2, v);
			return verify(w2, [&](size_t i) { return E(i, 2); });
		}});
	};
	stale("s01_stale_cuda_flush", "CUDA reads v, OpenCL updates v in place, v.flush(), CUDA reads v", "cuda", "opencl", [](skepu::Vector<int> &v) { v.flush(); });
	stale("s02_stale_cuda_dealloc", "as s01 with v.flush(Dealloc)", "cuda", "opencl", [](skepu::Vector<int> &v) { v.flush(skepu::FlushMode::Dealloc); });
	stale("s03_stale_cuda_updateHostAndInvalidate", "as s01 with v.updateHostAndInvalidateDevice()", "cuda", "opencl", [](skepu::Vector<int> &v) { v.updateHostAndInvalidateDevice(); });
	stale("s04_stale_opencl_flush", "OpenCL reads v, CUDA updates v in place, v.flush(), OpenCL reads v", "opencl", "cuda", [](skepu::Vector<int> &v) { v.flush(); });
	stale("s05_stale_opencl_dealloc", "as s04 with v.flush(Dealloc)", "opencl", "cuda", [](skepu::Vector<int> &v) { v.flush(skepu::FlushMode::Dealloc); });
	stale("s06_stale_opencl_updateHostAndInvalidate", "as s04 with v.updateHostAndInvalidateDevice()", "opencl", "cuda", [](skepu::Vector<int> &v) { v.updateHostAndInvalidateDevice(); });

	auto pingpong = [&s, E](const char *name, const char *description, std::function<void(skepu::Vector<int> &)> between)
	{
		s.push_back({name, description, [=]
		{
			skepu::Vector<int> v(N); init(v, 0);
			for (int k = 0; k < 10; ++k) { on(inc, (k % 2) ? "opencl" : "cuda")(v, v); between(v); }
			return verify(v, [&](size_t i) { return E(i, 10); });
		}});
	};
	pingpong("q01_pingpong_flush", "the in-place ping-pong with v.flush() after each step", [](skepu::Vector<int> &v) { v.flush(); });
	pingpong("q02_pingpong_dealloc", "the in-place ping-pong with v.flush(Dealloc) after each step", [](skepu::Vector<int> &v) { v.flush(skepu::FlushMode::Dealloc); });
	pingpong("q03_pingpong_updateHostAndInvalidate", "the in-place ping-pong with v.updateHostAndInvalidateDevice() after each step", [](skepu::Vector<int> &v) { v.updateHostAndInvalidateDevice(); });

	s.push_back({"t01_tensor3_cuda_to_opencl", "Tensor3: CUDA writes b, OpenCL reads b", [=]
	{
		skepu::Tensor3<int> a(4, 5, 6), b(4, 5, 6), c(4, 5, 6);
		skepu::external([&] { for (size_t i = 0; i < 4; ++i) for (size_t j = 0; j < 5; ++j) for (size_t k = 0; k < 6; ++k) a(i, j, k) = (int)(i * 100 + j * 10 + k); }, skepu::write(a));
		on(inc, "cuda")(b, a); on(inc, "opencl")(c, b);
		c.flush();
		size_t wrong = 0;
		for (size_t i = 0; i < 4; ++i) for (size_t j = 0; j < 5; ++j) for (size_t k = 0; k < 6; ++k) wrong += c(i, j, k) != (int)(i * 100 + j * 10 + k) + 2;
		return wrong ? std::to_string(wrong) + " of 120 elements are wrong" : std::string();
	}});
	return s;
}

int main(int argc, char *argv[])
{
	const auto all = scenarios();
	if (argc < 2 || !std::strcmp(argv[1], "list"))
	{
		for (auto &s : all) std::cout << s.name << "\t" << s.description << "\n";
		return 0;
	}
	for (auto &s : all)
		if (s.name == argv[1])
		{
			try
			{
				const std::string problem = s.run();
				std::cout << (problem.empty() ? "PASS" : "FAIL: " + problem) << "\n";
				return problem.empty() ? 0 : 1;
			}
			catch (const std::exception &e) { std::cout << "EXCEPTION: " << e.what() << "\n"; return 2; }
		}
	std::cout << "no such scenario\n";
	return 3;
}
