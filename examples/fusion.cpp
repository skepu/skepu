
#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/util.hpp>


float square(float a)
{
	return a * a;
}

float reciprocal(float a)
{
	return 1.f / a;
}



int main(int argc, char *argv[])
{
	size_t size {5};
	size_t Vsize {5};
	size_t Hsize {7};
	
	auto test = skepu::Map([](int a) { return a+1; });
	
	
	skepu::io::cout << "Map >> Map\n";
	{
		auto reciprocal_of_squares = skepu::Map(square) >> skepu::Map(reciprocal);

		skepu::Vector<float> v(size), res(size);
		for (size_t i = 0; i < size; ++i)
			v(i) = i+1;

		reciprocal_of_squares(res, v);
		skepu::io::cout << res << "\n";
	}
	
	
	skepu::io::cout << "Map >> Map >> Map\n";
	{
		auto reciprocal_of_squares = skepu::Map(square) >> skepu::Map(reciprocal) >> skepu::Map(square);

		skepu::Vector<float> v(size), res(size);
		for (size_t i = 0; i < size; ++i)
			v(i) = i+1;

		reciprocal_of_squares(res, v);
		skepu::io::cout << res << "\n";
	}
	
	
	
	skepu::io::cout << "Map >> Reduce\n";
	{
		auto dotproduct = skepu::Map(skepu::util::mul<float>) >> skepu::Reduce(skepu::util::add<float>);

		skepu::Vector<float> v1(size), v2(size, 1);
		for (size_t i = 0; i < size; ++i)
			v1(i) = i+1;

		float res = dotproduct(v1, v2);	
		skepu::io::cout << res << "\n";
	}
	
	
	
	
	skepu::io::cout << "Map >> (Map >> Reduce)\n";
	{
		auto reciprocal_of_squares
			=  skepu::Map(square)
			>> (skepu::Map(reciprocal)
					>> skepu::Reduce(skepu::util::add<float>)
				);

		skepu::Vector<float> v(size);
		for (size_t i = 0; i < size; ++i)
			v(i) = i+1;

		float res = reciprocal_of_squares(v);
		skepu::io::cout << res << "\n";
	}
	
	
	
	
	skepu::io::cout << "(Map >> Map) >> Reduce\n";
	{
		auto reciprocal_of_squares
			=  (skepu::Map(square)
					>> skepu::Map(reciprocal)
				)
			>> skepu::Reduce(skepu::util::add<float>);

		skepu::Vector<float> v(size);
		for (size_t i = 0; i < size; ++i)
			v(i) = i+1;

		float res = reciprocal_of_squares(v);
		skepu::io::cout << res << "\n";
	}
	
	
	
	skepu::io::cout << "MapPairs >> Reduce\n";
	{
		auto mpr_fusion = skepu::MapPairs<1, 1>(skepu::util::add<float>) >> skepu::Reduce(skepu::util::add<float>);

		skepu::Vector<float> v(Vsize), h(Hsize, 1), res(Vsize);
		for (size_t i = 0; i < Vsize; ++i)
			v(i) = i+1;

		mpr_fusion(res, v, h);	
		skepu::io::cout << res << "\n";
	}
	
	
	
	
	skepu::io::cout << "MapPairs >> Map\n";
	{
		auto fusion = skepu::MapPairs<1, 1>(skepu::util::add<float>) >> skepu::Map(reciprocal);

		skepu::Vector<float> v(Vsize), h(Hsize, 1);
		skepu::Matrix<float> res(Vsize, Hsize);
		for (size_t i = 0; i < Vsize; ++i)
			v(i) = i+1;

		fusion(res, v, h);	
		skepu::io::cout << res << "\n";
	}
	
	
	
	
	skepu::io::cout << "Map || Map\n";
	{
		auto reciprocal_and_squares = skepu::Map(square) || skepu::Map(reciprocal);

		skepu::Vector<float> v(size), resA(size), resB(size);
		for (size_t i = 0; i < size; ++i)
			v(i) = i+1;

		reciprocal_and_squares(resA, resB, v, v);
		skepu::io::cout << resA << "\n";
		skepu::io::cout << resB << "\n";
	}
	
	
	
	
	skepu::io::cout << "Reduce || Reduce\n";
	{
		auto skel	 = skepu::Reduce(skepu::util::max<float>)
		          || skepu::Reduce(skepu::util::min<float>);
		
		skepu::Vector<float> v(size);
		for (size_t i = 0; i < size; ++i)
			v(i) = i+1;

		auto [resA, resB] = skel(v, v);
		skepu::io::cout << resA << "\n";
		skepu::io::cout << resB << "\n";
	}
	
	
	/*
	// MapPairs || MapPairs
	{
		skepu::io::cout << "MapPairs || MapPairs\n";
		auto fusion = skepu::MapPairs<1, 1>(skepu::util::add<float>) || skepu::MapPairs<1, 1>(skepu::util::mul<float>);

		skepu::Vector<float> v1(Vsize), v2(Hsize, 1);
		skepu::Matrix<float> resA(Vsize, Hsize), resB(Vsize, Hsize);
		for (size_t i = 0; i < Vsize; ++i)
		{
			v1(i) = i;
		}
		for (size_t i = 0; i < Hsize; ++i)
		{
			v2(i) = i*100;
		}

		fusion(resA, resB, v1, v2, v1, v2);
		skepu::io::cout << resA << "\n";
		skepu::io::cout << resB << "\n";
	}
	*/
	
	
	// Map || Map || Map
	{
		skepu::io::cout << "Map || Map || Map\n";
		auto reciprocal_and_squares = skepu::Map(square) || skepu::Map(reciprocal) || skepu::Map(square);

		skepu::Vector<float> v(size), resA(size), resB(size), resC(size);
		for (size_t i = 0; i < size; ++i)
			v(i) = i+1;

		reciprocal_and_squares(resA, resB, resC, v, v, v);
		skepu::io::cout << resA << "\n";
		skepu::io::cout << resB << "\n";
		skepu::io::cout << resC << "\n";
	}
	
	
	
	// (Map >> Reduce) || (Map >> Reduce)
	{
		skepu::io::cout << "(Map >> Reduce) || (Map >> Reduce)\n";
		auto reciprocal_and_squares_sum
			=  (skepu::Map(square) >> skepu::Reduce(skepu::util::add<float>))
			|| (skepu::Map(reciprocal) >> skepu::Reduce(skepu::util::add<float>));

		skepu::Vector<float> v(size);
		for (size_t i = 0; i < size; ++i)
			v(i) = i+1;

	 	auto [resA, resB] = reciprocal_and_squares_sum(v, v);
		skepu::io::cout << resA << "\n";
		skepu::io::cout << resB << "\n";
	}
	
	
	
	// (Map || Map || Map) >> (Map || Map || Map)
	{
		skepu::io::cout << "(Map || Map || Map) >> (Map || Map || Map)\n";
		auto reciprocal_and_squares
		= (skepu::Map(square) || skepu::Map(reciprocal) || skepu::Map(square))
		>> (skepu::Map(reciprocal) || skepu::Map(square) || skepu::Map(square));

		skepu::Vector<float> v(size), resA(size), resB(size), resC(size);
		for (size_t i = 0; i < size; ++i)
			v(i) = i+1;

		reciprocal_and_squares(resA, resB, resC, v, v, v);
		skepu::io::cout << resA << "\n";
		skepu::io::cout << resB << "\n";
		skepu::io::cout << resC << "\n";
	}
	
	
	
	
	skepu::io::cout << "Map ^ N\n";
	{
		auto fusion = skepu::Map(skepu::util::increment<float>) ^ 5;

		skepu::Vector<float> v(size), res(size);
		for (size_t i = 0; i < size; ++i)
			v(i) = i+1;

		fusion(res, v);
		skepu::io::cout << res << "\n";
	}
	
	
	
	skepu::io::cout << "( (Map || Map || Map) >> (Map || Map || Map) ) ^ N\n";
	{
		auto reciprocal_and_squares
		= ((skepu::Map(square) || skepu::Map(reciprocal) || skepu::Map(square))
		>> (skepu::Map(reciprocal) || skepu::Map(square) || skepu::Map(sqrtf)))
		^ 5;

		skepu::Vector<float> v(size), resA(size), resB(size), resC(size);
		for (size_t i = 0; i < size; ++i)
			v(i) = i+1;

		reciprocal_and_squares(resA, resB, resC, v, v, v);
		skepu::io::cout << resA << "\n";
		skepu::io::cout << resB << "\n";
		skepu::io::cout << resC << "\n";
	}
	
	
	
	
	
	
}