#pragma once
#include <skepu>
#ifndef SKEPU_PRECOMPILED
static skepu::PrecompilerMarker startOfRandomizeHPP;

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

// skepu::randomize: random fills of the containers (vectors, matrices, tensors of 3 and 4 dimensions) with skeleton
// calls, in the way that skepu::blas does the linear algebra. The elements are filled in the order of the memory,
// from a stream of random numbers (skepu::PRNG), so the contents depend only on the state of the stream and the size
// of the container, not on the backend, the number of threads or the number of processes.
//
//   skepu::randomize::uniform(v, 99);          // integral elements: from 0 to 99
//   skepu::randomize::uniform(v, -5, 5);       // integral elements: from -5 to 5 (both included)
//   skepu::randomize::uniform(m);              // floating-point elements: from 0 up to 1
//   skepu::randomize::uniform(m, -1.f, 1.f);   // floating-point elements: from -1 up to 1
//   skepu::randomize::normal(t, 0.f, 2.f);     // floating-point elements: normal distribution (mean, standard deviation)
//   skepu::randomize::bernoulli(v, 0.3);       // 1 with the probability 0.3, else 0
//   skepu::randomize::poisson(v, 4.0);         // Poisson distribution with the mean 4 (at most 50)
//
// The stream is the one of the program (skepu::getGlobalPRNG()), unless another is set with set_prng() or, for a
// scope, with randomize::Scope. Every call uses the stream, and advances it by the same number of steps on all
// backends. The functions have no random stream as an argument, since it would interfere with the default values.
//
// The integers, the uniform and the normal numbers are made from 2 draws of the stream (31 bits each; 4 draws for a normal
// number), the Poisson numbers from 1 draw, so the integral ranges are at most 2^62 wide. The integral ranges are not exactly uniform (the remainder of a division is used), which is
// negligible unless the range is of the size of 2^62. The floating-point numbers have the resolution of their type.

namespace skepu {

namespace randomize {

	// ---------------- the stream ----------------

	inline PRNG *&stream_pointer()
	{
		static PRNG *stream = nullptr;
		return stream;
	}

	// The stream that the fills use now
	inline PRNG &prng()
	{
		PRNG *stream = stream_pointer();
		return stream ? *stream : getGlobalPRNG();
	}

	// Uses the stream (it is not copied: the fills advance the stream that is passed)
	inline void set_prng(PRNG &stream) { stream_pointer() = &stream; }

	// Uses the stream of the program again
	inline void reset_prng() { stream_pointer() = nullptr; }

	// Uses a stream for the lifetime of the object, then the previous stream again
	class Scope
	{
	public:
		explicit Scope(PRNG &stream): m_previous(stream_pointer()) { set_prng(stream); }
		~Scope() { stream_pointer() = m_previous; }
		Scope(const Scope &) = delete;
		Scope &operator=(const Scope &) = delete;
	private:
		PRNG *m_previous;
	};


	// ---------------- the containers ----------------

	namespace detail
	{
		// The element type of the containers that the fills can handle
		template<typename C>
		struct shape {};

		template<typename T> struct shape<Vector<T>>  { using value_type = T; };
		template<typename T> struct shape<Matrix<T>>  { using value_type = T; };
		template<typename T> struct shape<Tensor3<T>> { using value_type = T; };
		template<typename T> struct shape<Tensor4<T>> { using value_type = T; };

		template<typename C>
		using value_of = typename shape<C>::value_type;

		template<typename C>
		using integral_container = std::enable_if_t<std::is_integral<value_of<C>>::value, int>;

		template<typename C>
		using floating_container = std::enable_if_t<std::is_floating_point<value_of<C>>::value, int>;

		template<typename C>
		using numeric_container = std::enable_if_t<std::is_arithmetic<value_of<C>>::value, int>;

		// How a floating-point number from 0 up to 1 is made of the two draws of 31 bits (high and low): the high one
		// shifted right, times a scale, plus the low one shifted right, times a scale. The type gets the bits of its
		// mantissa: float 24 (all from the high draw), double 53 (31 from the high draw, 22 from the low one).
		// (32-bit integers only: Apple's OpenCL compiler fails on conversions of 64-bit integers to float)
		template<typename T>
		struct real_bits
		{
			int high_shift, low_shift;
			T high_scale, low_scale;

			real_bits()
			{
				const int digits = std::numeric_limits<T>::digits;
				if (digits <= 31)
				{
					high_shift = 31 - digits;
					high_scale = (T)std::ldexp(1.0, -digits);
					low_shift = 31;
					low_scale = 0;
				}
				else
				{
					high_shift = 0;
					high_scale = (T)std::ldexp(1.0, -31);
					low_shift = 31 - (digits - 31);
					low_scale = (T)std::ldexp(1.0, -digits);
				}
			}
		};

		inline size_t range_span(long long min, long long max)
		{
			if (max < min)
				SKEPU_ERROR("skepu::randomize::uniform: the maximum is below the minimum");
			const unsigned long long span = (unsigned long long)max - (unsigned long long)min + 1ull;
			if (span == 0 || span > (1ull << 62))
				SKEPU_ERROR("skepu::randomize::uniform: the range is wider than 2^62");
			return (size_t)span;
		}


		// ---------------- the user functions ----------------
		// (62 bits are made of two draws of 31 bits)

		template<typename T>
		T uniform_integer_uf(skepu::Random<2> &random, T min, size_t span)
		{
			unsigned long long high = random.get();
			unsigned long long low = random.get();
			unsigned long long bits = (high << 31) | low;
			return (T)((unsigned long long)min + bits % span);
		}

		template<typename T>
		T uniform_real_uf(skepu::Random<2> &random, T min, T width, int high_shift, T high_scale, int low_shift, T low_scale)
		{
			unsigned int high = random.get();
			unsigned int low = random.get();
			T u = (T)(high >> high_shift);
			u = u * high_scale;
			T v = (T)(low >> low_shift);
			v = v * low_scale;
			return min + (u + v) * width;
		}

		// Box-Muller: a normally distributed number from two uniform numbers, one of the two that they make
		template<typename T>
		T normal_uf(skepu::Random<4> &random, T mean, T deviation, int high_shift, T high_scale, int low_shift, T low_scale)
		{
			unsigned int high1 = random.get();
			unsigned int low1 = random.get();
			unsigned int high2 = random.get();
			unsigned int low2 = random.get();
			T u1 = (T)(high1 >> high_shift);
			u1 = u1 * high_scale;
			T v1 = (T)(low1 >> low_shift);
			v1 = v1 * low_scale;
			T u2 = (T)(high2 >> high_shift);
			u2 = u2 * high_scale;
			T v2 = (T)(low2 >> low_shift);
			v2 = v2 * low_scale;
			T a = (T)1 - (u1 + v1); // above 0, up to 1
			T b = u2 + v2;
			return mean + deviation * sqrt((T)-2 * log(a)) * cos((T)6.283185307179586 * b);
		}

		// (the threshold is compared as the two draws of 31 bits that it is made of, so that no 64-bit integer is passed on)
		template<typename T>
		T bernoulli_uf(skepu::Random<2> &random, unsigned int threshold_high, unsigned int threshold_low)
		{
			unsigned int high = random.get();
			unsigned int low = random.get();
			T result = 0;
			if (high < threshold_high || (high == threshold_high && low < threshold_low))
				result = 1;
			return result;
		}

		// Poisson: the smallest number k whose cumulative probability exceeds a uniform number, found by adding the
		// probabilities from 0 on (in float, which is also what the OpenCL devices all have)
		template<typename T>
		T poisson_uf(skepu::Random<1> &random, float lambda, int limit)
		{
			unsigned int draw = random.get();
			float u = (float)(draw >> 7);
			u = u / 16777216.0f;
			float p = exp(-lambda);
			float cumulative = p;
			int k = 0;
			while (u >= cumulative && k < limit)
			{
				k = k + 1;
				p = p * lambda / (float)k;
				cumulative = cumulative + p;
			}
			return (T)k;
		}
	}


	// ---------------- the uniform distribution ----------------

	// Integral elements, from min to max (both included). The maximum is always given.
	template<typename C, detail::integral_container<C> Enable = 0>
	void uniform(C &container, detail::value_of<C> min, detail::value_of<C> max SKEPU_CALL_SITE_PARAM)
	{
		SKEPU_TRACE_API("randomize.uniform");
		using T = detail::value_of<C>;
		const size_t span = detail::range_span((long long)min, (long long)max);
		auto skel = skepu::Map<0>(detail::uniform_integer_uf<T>);
		skel.setLabel("randomize::uniform");
		skel.setPRNG(prng());
		skel(container, min, span);
	}

	template<typename C, detail::integral_container<C> Enable = 0>
	void uniform(C &container, detail::value_of<C> max SKEPU_CALL_SITE_PARAM)
	{
		SKEPU_TRACE_API("randomize.uniform");
		uniform(container, detail::value_of<C>(0), max);
	}

	// Floating-point elements, from min up to max (0 and 1 by default)
	template<typename C, detail::floating_container<C> Enable = 0>
	void uniform(C &container, detail::value_of<C> min = 0, detail::value_of<C> max = 1 SKEPU_CALL_SITE_PARAM)
	{
		SKEPU_TRACE_API("randomize.uniform");
		using T = detail::value_of<C>;
		const detail::real_bits<T> bits;
		auto skel = skepu::Map<0>(detail::uniform_real_uf<T>);
		skel.setLabel("randomize::uniform");
		skel.setPRNG(prng());
		skel(container, min, (T)(max - min), bits.high_shift, bits.high_scale, bits.low_shift, bits.low_scale);
	}


	// ---------------- the normal distribution ----------------

	// Floating-point elements with the normal distribution of the mean and the standard deviation
	template<typename C, detail::floating_container<C> Enable = 0>
	void normal(C &container, detail::value_of<C> mean = 0, detail::value_of<C> deviation = 1 SKEPU_CALL_SITE_PARAM)
	{
		SKEPU_TRACE_API("randomize.normal");
		using T = detail::value_of<C>;
		const detail::real_bits<T> bits;
		auto skel = skepu::Map<0>(detail::normal_uf<T>);
		skel.setLabel("randomize::normal");
		skel.setPRNG(prng());
		skel(container, mean, deviation, bits.high_shift, bits.high_scale, bits.low_shift, bits.low_scale);
	}


	// ---------------- more distributions ----------------

	// The elements are 1 with the probability p (from 0 to 1), else 0
	template<typename C, detail::numeric_container<C> Enable = 0>
	void bernoulli(C &container, double p SKEPU_CALL_SITE_PARAM)
	{
		SKEPU_TRACE_API("randomize.bernoulli");
		using T = detail::value_of<C>;
		if (!(p >= 0 && p <= 1))
			SKEPU_ERROR("skepu::randomize::bernoulli: the probability is not from 0 to 1");
		const unsigned long long threshold = (unsigned long long)(p * 4611686018427387904.0); // p * 2^62
		auto skel = skepu::Map<0>(detail::bernoulli_uf<T>);
		skel.setLabel("randomize::bernoulli");
		skel.setPRNG(prng());
		skel(container, (unsigned int)(threshold >> 31), (unsigned int)(threshold & 0x7fffffffull));
	}

	// The elements have the Poisson distribution of the mean lambda (from 0 to 50)
	template<typename C, detail::numeric_container<C> Enable = 0>
	void poisson(C &container, double lambda SKEPU_CALL_SITE_PARAM)
	{
		SKEPU_TRACE_API("randomize.poisson");
		using T = detail::value_of<C>;
		if (!(lambda >= 0 && lambda <= 50))
			SKEPU_ERROR("skepu::randomize::poisson: the mean is not from 0 to 50");
		const int limit = (int)(lambda + 10 * std::sqrt(lambda) + 20);
		auto skel = skepu::Map<0>(detail::poisson_uf<T>);
		skel.setLabel("randomize::poisson");
		skel.setPRNG(prng());
		skel(container, (float)lambda, limit);
	}

}}

static skepu::PrecompilerMarker endOfRandomizeHPP;
#endif // SKEPU_PRECOMPILED
