#include <skepu>
#include <skepu-lib/io.hpp>

template<typename T>
T iden(T a)
{
	return a;
}

template<typename T>
T mult(T a, T b)
{
	return a * b;
}

template<typename T>
T add(T a, T b)
{
	return a + b;
}

//#define SKEPU_TRACE_MARKER(dc, msg) skepu::external(msg, skepu::read(dc), [&]{}, skepu::write(dc))
#define SKEPU_TRACE_MARKER(msg, ...) skepu::external(msg, skepu::read(__VA_ARGS__), [&]{}, skepu::write(__VA_ARGS__))

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		skepu::io::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
	
	const size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	skepu::setGlobalBackendSpec(spec);
	
	// Map
	{
		SKEPU_TRACE_SCOPE("Map Tests");
		skepu::Vector<float> v(size, "Map input/output");
		SKEPU_TRACE_MARKER("Map", v);
		auto mapper = skepu::Map(iden<float>);
		mapper(v, v);
	}
	
	// Reduce
	{
		SKEPU_TRACE_SCOPE("Reduce Tests");
		
		// Reduce 1D
		{
			SKEPU_TRACE_SCOPE("Reduce1D Tests");
			skepu::Vector<float> v(size, "Reduce 1D-Full input");
			SKEPU_TRACE_MARKER("Reduce 1D-Full", v);
			auto reduce_1d_full = skepu::Reduce(add<float>);
			float res = reduce_1d_full(v);
			
			skepu::Matrix<float> m(size, size, "Reduce 1D-Partial input");
			skepu::Vector<float> res_v(size, "Reduce 1D-Partial output");
			SKEPU_TRACE_MARKER("Reduce 1D-Partial", m);
			auto reduce_1d_partial = skepu::Reduce(add<float>);
			reduce_1d_partial(res_v, m);
		}
		
		// Reduce 2D
		{
			SKEPU_TRACE_SCOPE("Reduce2D Tests");
			skepu::Matrix<float> m(size, size, "Reduce 2D input");
			SKEPU_TRACE_MARKER("Reduce 2D", m);
			auto reduce_2d = skepu::Reduce(mult<float>, add<float>);
			float res = reduce_2d(m);
		}
	}
	
	// Scan
	{
		SKEPU_TRACE_SCOPE("Scan Tests");
		skepu::Vector<float> v(size, "Scan input"), res(size, "Scan output");
		SKEPU_TRACE_MARKER("Scan", v);
		auto scan = skepu::Scan(add<float>);
		scan(res, v);
	}
	
	// MapReduce
	{
		SKEPU_TRACE_SCOPE("MapReduce Tests");
		skepu::Vector<float> v(size, "MapReduce input");
		SKEPU_TRACE_MARKER("MapReduce", v);
		auto mapreduce = skepu::MapReduce(iden<float>, add<float>);
		float res = mapreduce(v);
	}
	
	// MapOverlap
	{
		SKEPU_TRACE_SCOPE("MapOverlap Tests");
	
		// MapOverlap 1D
		{
			SKEPU_TRACE_SCOPE("MapOverlap1D Tests");
			skepu::Vector<float> v(size, "MapOverlap1D input"), res_v(size, "MapOverlap1D output");
			SKEPU_TRACE_MARKER("MapOverlap 1D", v);
			auto mapoverlap1d = skepu::MapOverlap([](skepu::Region1D<float> r) { return r(0); });
			mapoverlap1d.setEdgeMode(skepu::Edge::Pad);
			mapoverlap1d.setPad(0);
			mapoverlap1d(res_v, v);
			
			skepu::Matrix<float> m1(size, size, "MapOverlap1D input"), res_m1(size, size, "MapOverlap1D output");
			SKEPU_TRACE_MARKER("MapOverlap 1D RowWise", m1);
			mapoverlap1d.setOverlapMode(skepu::Overlap::RowWise);
		//	mapoverlap1d(res_m1, m1);

			skepu::Matrix<float> m2(size, size, "MapOverlap1D input"), res_m2(size, size, "MapOverlap1D output");
			SKEPU_TRACE_MARKER("MapOverlap 1D ColWise", m2);
			mapoverlap1d.setOverlapMode(skepu::Overlap::ColWise);
		//	mapoverlap1d(res_m2, m2);
		}

		// MapOverlap 2D
		{
			SKEPU_TRACE_SCOPE("MapOverlap2D Tests");
			skepu::Matrix<float> m(size, size, "MapOverlap2D input"), res(size, size, "MapOverlap2D output");
			SKEPU_TRACE_MARKER("MapOverlap 2D", m);
			auto mapoverlap2d = skepu::MapOverlap([](skepu::Region2D<float> r) { return r(0, 0); });
			mapoverlap2d.setEdgeMode(skepu::Edge::Pad);
			mapoverlap2d.setPad(0);
			mapoverlap2d(res, m);
		}

		// MapOverlap 3D
		{
			SKEPU_TRACE_SCOPE("MapOverlap3D Tests");
			skepu::Tensor3<float> t(size, size, size, "MapOverlap3D input"), res(size, size, size, "MapOverlap3D output");
			SKEPU_TRACE_MARKER("MapOverlap 3D", t);
			auto mapoverlap3d = skepu::MapOverlap([](skepu::Region3D<float> r) { return r(0, 0, 0); });
			mapoverlap3d.setEdgeMode(skepu::Edge::Pad);
			mapoverlap3d.setPad(0);
			mapoverlap3d(res, t);
		}

		// MapOverlap 4D
		{
			SKEPU_TRACE_SCOPE("MapOverlap4D Tests");
			skepu::Tensor4<float> t(size, size, size, size, "MapOverlap4D input"), res(size, size, size, size, "MapOverlap4D output");
			SKEPU_TRACE_MARKER("MapOverlap 4D", t);
			auto mapoverlap4d = skepu::MapOverlap([](skepu::Region4D<float> r) { return r(0, 0, 0, 0); });
			mapoverlap4d.setEdgeMode(skepu::Edge::Pad);
			mapoverlap4d.setPad(0);
			mapoverlap4d(res, t);
		}
	}
	
	// MapPairs
	{
		SKEPU_TRACE_SCOPE("MapPairs Tests");
		skepu::Vector<float> v1(size, "MapPairs input"), v2(size, "MapPairs input");
		skepu::Matrix<float> res(size, size, "MapPairs output");
		SKEPU_TRACE_MARKER("MapPairs", v1, v2);
		auto mappairs = skepu::MapPairs<1, 1>([](float e1, float e2) { return e1 + e2; });
		mappairs(res, v1, v2);
	}
	
	// MapPairsReduce
	{
		SKEPU_TRACE_SCOPE("MapPairsReduce Tests");
		skepu::Vector<float> v1(size, "MapPairsReduce input"), v2(size, "MapPairsReduce input"), res(size, "MapPairsReduce output");
		SKEPU_TRACE_MARKER("MapPairs", v1, v2);
		auto mappairsreduce = skepu::MapPairsReduce<1, 1>([](float e1, float e2) { return e1 + e2; }, [](float e1, float e2) { return e1 + e2; });
		mappairsreduce(res, v1, v2);
	}
	
	// PRNG
	{
		skepu::Vector<float> v1(size, "PRNG input 1"), v2(size, "PRNG input 2");
		
		auto prng_map = skepu::Map<1>([](skepu::Random<1> &rnd, float e) -> float
		{
			return e + rnd.getNormalized();
		});
		
		{
			SKEPU_TRACE_SCOPE("Global PRNG");
			prng_map(v1, v1);
			prng_map(v2, v2);
		}
		
		{
			SKEPU_TRACE_SCOPE("Local PRNG");
			skepu::PRNG local_prng(0);
			prng_map.setPRNG(local_prng);
			prng_map(v1, v1);
			prng_map(v2, v2);
		}
		
	}
	
	return 0;
}
