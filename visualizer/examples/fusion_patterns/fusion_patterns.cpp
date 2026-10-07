// Synthetic example for SkeVU: calls that can be fused (skepu::Map(f) >> skepu::Map(g), a || b, see
// examples/fusion.cpp), and calls that look as if they could but cannot. Every block is one region; turn on
// Advanced > Fusion analysis to see the groups, and the reason why the other calls are not in one.
// (Illustrative: the trace was made by examples/synthetic_traces.py, this program has not been run.)
#include <skepu>
#include <skepu-lib/io.hpp>

int main()
{
	{ SKEPU_TRACE_SCOPE("Fusible");
		// Three Maps in a row, each reading only the result of the one before.
		//   fused:  auto f = scale >> shift >> clamp;  f(y, x);
		{ SKEPU_TRACE_SCOPE("Map, Map, Map");
			size_t n = 69632;
			skepu::Vector<float> x(n, "X");
			skepu::Vector<float> t1(n, "T1");
			skepu::Vector<float> t2(n, "T2");
			skepu::Vector<float> y(n, "Y");
			auto scale = skepu::Map([](float a) { return 2.0f * a; });
			auto shift = skepu::Map([](float a) { return a + 1.0f; });
			auto clamp = skepu::Map([](float a) { return fminf(a, 10.0f); });
			scale(t1, x);
			shift(t2, t1);
			clamp(y, t2);
		}

		// A Map and a Reduce of its result: a MapReduce, the squares are never stored.
		//   fused:  auto sum_of_squares = skepu::Map(square) >> skepu::Reduce(add);
		{ SKEPU_TRACE_SCOPE("Map into Reduce");
			size_t n = 73728;
			skepu::Vector<float> x(n, "X");
			skepu::Vector<float> t(n, "Squares");
			skepu::Scalar<float> sum("Sum");
			auto square_sum = skepu::Map([](float a) { return a * a; });
			auto add_all = skepu::Reduce([](float a, float b) { return a + b; });
			square_sum(t, x);
			add_all(sum, t);
		}

		// Three Maps over the same elements that do not depend on each other: one launch, X read once.
		//   fused:  auto f = skepu::Map(square) || skepu::Map(negate) || skepu::Map(half);
		{ SKEPU_TRACE_SCOPE("Map, Map, Map side by side");
			size_t n = 77824;
			skepu::Vector<float> x(n, "X");
			skepu::Vector<float> u(n, "Squares");
			skepu::Vector<float> v(n, "Negated");
			skepu::Vector<float> w(n, "Halves");
			auto sq = skepu::Map([](float a) { return a * a; });
			auto neg = skepu::Map([](float a) { return -a; });
			auto halve = skepu::Map([](float a) { return 0.5f * a; });
			sq(u, x);
			neg(v, x);
			halve(w, x);
		}

		// Two Maps feed a third: a tree, which fuses into one call (each of U and V is read by one call only).
		//   fused:  auto f = (skepu::Map(square) || skepu::Map(negate)) >> skepu::Map(add);
		{ SKEPU_TRACE_SCOPE("Two Maps into one");
			size_t n = 81920;
			skepu::Vector<float> x(n, "X");
			skepu::Vector<float> z(n, "Z");
			skepu::Vector<float> u(n, "U");
			skepu::Vector<float> v(n, "V");
			skepu::Vector<float> w(n, "W");
			auto square_u = skepu::Map([](float a) { return a * a; });
			auto negate_v = skepu::Map([](float a) { return -a; });
			auto add_uv = skepu::Map([](float a, float b) { return a + b; });
			square_u(u, x);
			negate_v(v, z);
			add_uv(w, u, v);
		}

		// Two Reduces over the same vector, in one pass.
		//   fused:  auto minmax = skepu::Reduce(max) || skepu::Reduce(min);  auto [hi, lo] = minmax(x, x);
		{ SKEPU_TRACE_SCOPE("Reduce and Reduce");
			size_t n = 86016;
			skepu::Vector<float> x(n, "X");
			skepu::Scalar<float> hi("Max");
			skepu::Scalar<float> lo("Min");
			auto max_of = skepu::Reduce([](float a, float b) { return fmaxf(a, b); });
			auto min_of = skepu::Reduce([](float a, float b) { return fminf(a, b); });
			max_of(hi, x);
			min_of(lo, x);
		}

		// A Map (ReLU) into a MapPool with windows that do not overlap (stride = window): no element is
		// computed twice, so this is as cheap as a Map into a Map. With overlapping windows it would recompute.
		{ SKEPU_TRACE_SCOPE("Map into MapPool");
			size_t n = 90112;
			skepu::Tensor4<float> x(8, 44, 44, 8, "X");
			skepu::Tensor4<float> relu_out(8, 44, 44, 8, "Activations");
			skepu::Tensor4<float> pooled(8, 22, 22, 8, "Pooled");
			auto relu = skepu::Map([](float a) { return fmaxf(a, 0.0f); });
			auto max_pool = skepu::MapOverlap(max_pooling_uf);   // a pool: stride = window
			relu(relu_out, x);
			max_pool(pooled, relu_out);
		}

		// A Map (brightness) into a 3x3 stencil. Fusing is possible, but the stencil reads each element nine times:
		// the fused call would compute the Map nine times for each element, trading memory traffic for work.
		{ SKEPU_TRACE_SCOPE("Map into MapOverlap (recomputes)");
			size_t n = 94208;
			skepu::Matrix<float> image(228, 228, "Image");
			skepu::Matrix<float> bright(228, 228, "Brightened");
			skepu::Matrix<float> edges(228, 228, "Edges");
			auto brighten = skepu::Map([](float a) { return 1.2f * a + 0.1f; });
			auto laplace = skepu::MapOverlap(laplace_uf);   // 3x3 stencil
			brighten(bright, image);
			laplace(edges, bright);
		}

		// A Map over one of the vectors of a MapPairs. MapPairs reads each element of it for every element of the other
		// vector, so the fused call computes the Map that many times.
		{ SKEPU_TRACE_SCOPE("Map into MapPairs (recomputes)");
			size_t n = 98304;
			skepu::Vector<float> raw_a(512, "Raw A");
			skepu::Vector<float> a(512, "A");
			skepu::Vector<float> b(160, "B");
			skepu::Matrix<float> dist(160, 512, "Distances");
			auto normalize = skepu::Map([](float a) { return a / 255.0f; });
			auto distance = skepu::MapPairs<1, 1>([](float a, float b) { return fabsf(a - b); });
			normalize(a, raw_a);
			distance(dist, a, b);
		}

		// The matrix of a MapPairs, scaled element by element: the matrix is never stored.
		//   fused:  auto f = skepu::MapPairs<1, 1>(add) >> skepu::Map(reciprocal);
		{ SKEPU_TRACE_SCOPE("MapPairs into Map");
			size_t n = 102400;
			skepu::Vector<float> u(512, "U");
			skepu::Vector<float> v(132, "V");
			skepu::Matrix<float> sums(132, 512, "Sums");
			skepu::Matrix<float> recip(132, 512, "Reciprocals");
			auto pair_sum = skepu::MapPairs<1, 1>([](float a, float b) { return a + b; });
			auto reciprocal = skepu::Map([](float a) { return 1.0f / a; });
			pair_sum(sums, u, v);
			reciprocal(recip, sums);
		}

	}

	{ SKEPU_TRACE_SCOPE("Not fusible");
		// A Map into a Scan: there is no MapScan, the Scan cannot take the Map's function.
		{ SKEPU_TRACE_SCOPE("Map into Scan");
			size_t n = 106496;
			skepu::Vector<float> x(n, "X");
			skepu::Vector<float> t(n, "T");
			skepu::Vector<float> y(n, "Prefix sums");
			auto double_x = skepu::Map([](float a) { return 2.0f * a; });
			auto prefix_sum = skepu::Scan([](float a, float b) { return a + b; });
			double_x(t, x);
			prefix_sum(y, t);
		}

		// T is read by two calls: it has to be stored. The two readers are independent of each other,
		// though, and can be fused side by side (g || h).
		{ SKEPU_TRACE_SCOPE("Two readers");
			size_t n = 110592;
			skepu::Vector<float> x(n, "X");
			skepu::Vector<float> t(n, "T");
			skepu::Vector<float> ya(n, "Ya");
			skepu::Vector<float> yb(n, "Yb");
			auto plus_one = skepu::Map([](float a) { return a + 1.0f; });
			auto times_two = skepu::Map([](float a) { return 2.0f * a; });
			auto negate_it = skepu::Map([](float a) { return -a; });
			plus_one(t, x);
			times_two(ya, t);
			negate_it(yb, t);
		}

		// The host reads an element of T between the two calls, so T has to exist.
		{ SKEPU_TRACE_SCOPE("The host looks in between");
			size_t n = 114688;
			skepu::Vector<float> x(n, "X");
			skepu::Vector<float> t(n, "T");
			skepu::Vector<float> y(n, "Y");
			auto add_one = skepu::Map([](float a) { return a + 1.0f; });
			auto scale_two = skepu::Map([](float a) { return 2.0f * a; });
			add_one(t, x);
			float first = t[0];
			scale_two(y, t);
		}

		// The second call reads T at any index (a proxy, skepu::Vec), not element by element:
		// it needs all of T finished before it starts.
		{ SKEPU_TRACE_SCOPE("Random access");
			size_t n = 118784;
			skepu::Vector<float> x(n, "X");
			skepu::Vector<float> w(n, "Indices");
			skepu::Vector<float> t(n, "T");
			skepu::Vector<float> y(n, "Y");
			auto bump = skepu::Map([](float a) { return a + 1.0f; });
			auto lookup = skepu::Map([](float i, skepu::Vec<float> table) { return table[(size_t)i % table.size]; });
			bump(t, x);
			lookup(y, w, t);
		}

		// The same data, read as a vector by the first call and as a matrix by the second:
		// the calls do not run over the same elements.
		{ SKEPU_TRACE_SCOPE("Other shapes");
			size_t n = 122880;
			skepu::Vector<float> x(n, "X");
			skepu::Vector<float> t(n, "T");
			skepu::Vector<float> y(n, "Y");
			auto shift_vec = skepu::Map([](float a) { return a + 1.0f; });
			auto scale_mat = skepu::Map([](float a) { return 2.0f * a; });
			shift_vec(t, x);
			scale_mat(y_matrix, t_matrix);   // the same memory, as a 256 x 480 matrix
		}

		// The first call runs on OpenMP, the second on OpenCL: T has to go to the device in between.
		{ SKEPU_TRACE_SCOPE("Different backends");
			size_t n = 126976;
			skepu::Vector<float> x(n, "X");
			skepu::Vector<float> t(n, "T");
			skepu::Vector<float> y(n, "Y");
			auto omp_add = skepu::Map([](float a) { return a + 1.0f; });
			auto ocl_scale = skepu::Map([](float a) { return 2.0f * a; });
			f.setBackend(skepu::BackendSpec(skepu::Backend::Type::OpenMP));
			g.setBackend(skepu::BackendSpec(skepu::Backend::Type::OpenCL));
			omp_add(t, x);
			ocl_scale(y, t);
		}

		// Both calls draw from the same random stream. In one call, the numbers would come in another order.
		{ SKEPU_TRACE_SCOPE("One random stream");
			size_t n = 131072;
			skepu::Vector<float> x(n, "X");
			skepu::Vector<float> t(n, "T");
			skepu::Vector<float> y(n, "Y");
			skepu::Random<> stream;   // shared by the calls below
			auto jitter = skepu::Map([](float a, skepu::Random<1> &r) { return a + r.f(); });
			auto noise = skepu::Map([](float a, skepu::Random<1> &r) { return a * (1.0f + 0.1f * r.f()); });
			jitter(t, x, stream);
			noise(y, t, stream);
		}

		// Between f and combine, prefix overwrites x, which f reads, and produces what combine reads.
		// f cannot move down past prefix, combine cannot move up past it.
		{ SKEPU_TRACE_SCOPE("A call in between");
			size_t n = 135168;
			skepu::Vector<float> x(n, "X");
			skepu::Vector<float> q(n, "Q");
			skepu::Vector<float> t(n, "T");
			skepu::Vector<float> y(n, "Y");
			auto increment = skepu::Map([](float a) { return a + 1.0f; });
			auto prefix = skepu::Scan([](float a, float b) { return a + b; });
			auto combine = skepu::Map([](float a, float b) { return a + b; });
			increment(t, x);
			prefix(x, q);        // overwrites x
			combine(y, t, x);
		}

		// Only a Map can be fused into the call that reads its output; the stencil's result is not a stream
		// of independent elements (and would have to be computed with its halo).
		{ SKEPU_TRACE_SCOPE("MapOverlap into Map");
			size_t n = 139264;
			skepu::Matrix<float> image(272, 272, "Image");
			skepu::Matrix<float> blurred(272, 272, "Blurred");
			skepu::Matrix<float> mask(272, 272, "Mask");
			auto blur = skepu::MapOverlap(blur_uf);   // 3x3 stencil
			auto threshold = skepu::Map([](float a) { return a > 0.5f ? 1.0f : 0.0f; });
			blur(blurred, image);
			threshold(mask, blurred);
		}

		// The Map needs the whole sum before its first element: a Reduce is a barrier.
		{ SKEPU_TRACE_SCOPE("A Reduce before a Map");
			size_t n = 143360;
			skepu::Vector<float> x(n, "X");
			skepu::Vector<float> y(n, "Normalized");
			skepu::Scalar<float> sum("Sum");
			auto total_of = skepu::Reduce([](float a, float b) { return a + b; });
			auto normalize_by = skepu::Map<1>([](float a, float total) { return a / total; });
			total_of(sum, x);
			normalize_by(y, x, sum);
		}

		// Three independent Maps; only the two over the same number of elements can share a launch.
		{ SKEPU_TRACE_SCOPE("Maps of different sizes");
			size_t n = 147456;
			skepu::Vector<float> x(n, "X");
			skepu::Vector<float> w(n / 2, "W");
			skepu::Vector<float> a(n, "A");
			skepu::Vector<float> b(n, "B");
			skepu::Vector<float> c(n / 2, "C");
			auto inc = skepu::Map([](float a) { return a + 1.0f; });
			auto dec = skepu::Map([](float a) { return a - 1.0f; });
			auto twice = skepu::Map([](float a) { return 2.0f * a; });
			inc(a, x);
			dec(b, x);
			twice(c, w);
		}

	}

	return 0;
}
