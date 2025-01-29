#include <iostream>

#include <skepu>
#include <skepu-lib/io.hpp>



int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		skepu::external([&]{
			std::cout << "Usage: " << argv[0] << " size backend\n";});
		exit(1);
	}

	const size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	skepu::setGlobalBackendSpec(spec);
	
	// Overlap 4D forward
	{
		skepu::io::cout << "Overlap 4D Forward\n";
		size_t stride[4] = {1, 1, 1, 1};
		size_t overlap[4] = {0, 1, 1, 0};
		auto conv4 = skepu::MapOverlap([](skepu::Index4D index, skepu::Region4D<float> r, skepu::Ten4<float> stencil) -> float
		{
		//	std::cout << p;
			float res = 0;
			for (float i = -r.oi; i <= r.oi; ++i)
				for (float j = -r.oj; j <= r.oj; ++j)
					for (float k = -r.ok; k <= r.ok; ++k)
						for (float l = -r.ol; l <= r.ol; ++l)
							res += r(i, j, k, l) * stencil(i+r.oi, j+r.oj, k+r.ok, l+r.ol);
			
			return res;
			
		});
		conv4.setOverlap(overlap[0], overlap[1], overlap[2], overlap[3]);
		conv4.setStride(stride[0], stride[1], stride[2], stride[3]);

		skepu::Tensor4<float> ten4(size, size, size, size);
		skepu::Tensor4<float> ret_ten4( // TBD fix
			(size - overlap[0] * 2) / stride[0],
			(size - overlap[1] * 2) / stride[1],
			(size - overlap[2] * 2) / stride[2],
			(size - overlap[3] * 2) / stride[3]
		);
		skepu::Tensor4<float> stencil4(overlap[0]*2+1, overlap[1]*2+1, overlap[2]*2+1, overlap[3]*2+1, "", 1);

		skepu::external(
			[&]
			{
				auto i(1);
				for(auto & e : ten4)
					e = i++;
			},
			skepu::write(ten4));
		
		conv4(ret_ten4, ten4, stencil4);
		skepu::io::cout << "Tensor4 in: " << ten4 << "\n";
		skepu::io::cout << "Tensor4 out: " << ret_ten4 << "\n";
	}
	
	// Overlap 4D backward
	{
		skepu::io::cout << "Overlap 4D Backward\n";
		size_t stride[4] = {1, 1, 1, 1};
		size_t overlap[4] = {0, 1, 1, 0};
		auto conv4 = skepu::MapOverlap([](skepu::Index4D index, skepu::Region4D<float> r, skepu::Ten4<float> stencil) -> float
		{
		//	std::cout << p;
			float res = 0;
			for (float i = -r.oi; i <= r.oi; ++i)
				for (float j = -r.oj; j <= r.oj; ++j)
					for (float k = -r.ok; k <= r.ok; ++k)
						for (float l = -r.ol; l <= r.ol; ++l)
							res += r(i, j, k, l) * stencil(i+r.oi, j+r.oj, k+r.ok, l+r.ol);
			
			return res;
			
			
		});
		conv4.setEdgeMode(skepu::Edge::Pad);
		conv4.setPad(-1);
		conv4.setOverlap(overlap[0], overlap[1], overlap[2], overlap[3]);
		conv4.setStride(stride[0], stride[1], stride[2], stride[3]);

		skepu::Tensor4<float> ten4(size, size, size, size);
		skepu::Tensor4<float> ret_ten4(
			(size + overlap[0] * 2) / stride[0],
			(size + overlap[1] * 2) / stride[1],
			(size + overlap[2] * 2) / stride[2],
			(size + overlap[3] * 2) / stride[3]
		);
		skepu::Tensor4<float> stencil4(overlap[0]*2+1, overlap[1]*2+1, overlap[2]*2+1, overlap[3]*2+1, "", 1);

		skepu::external(
			[&]
			{
				auto i(1);
				for(auto & e : ten4)
					e = i++;
			},
			skepu::write(ten4));
		
		conv4(ret_ten4, ten4, stencil4);
		skepu::io::cout << "Tensor4 in: " << ten4 << "\n";
		skepu::io::cout << "Tensor4 out: " << ret_ten4 << "\n";
	}
	
	// Pool 4D
	{
		skepu::io::cout << "Pool 4D\n";
		size_t stride[4] = {1, 2, 2, 1};
		size_t pool[4] = {1, 2, 2, 1};
		auto conv4 = skepu::MapPool([](skepu::Pool4D<float> p, skepu::Ten4<float> stencil) -> float
		{
		//	std::cout << p;
			float maxval = p(0,0,0,0);
			for (size_t j = 0; j < p.sj; ++j)
				for (size_t k = 0; k < p.sk; ++k)
				{
					float val = p(0, j, k, 0);
					maxval = (maxval > val) ? maxval : val;
				}
			return maxval;
		});
		conv4.setPoolSize(pool[0], pool[1], pool[2], pool[3]);
		conv4.setStride(stride[0], stride[1], stride[2], stride[3]);

		skepu::Tensor4<float> ten4(1, 12, 12, 4); //(size, size, size, size);
		skepu::Tensor4<float> ret_ten4(
	//		(size - pool[0] + stride[0]) / stride[0],
	//		(size - pool[1] + stride[1]) / stride[1],
	//		(size - pool[2] + stride[2]) / stride[2],
	//		(size - pool[3] + stride[3]) / stride[3]
			(1) / pool[0],
			(12) / pool[1],
			(12) / pool[2],
			(4) / pool[3]
		);
		skepu::Tensor4<float> stencil4(pool[0], pool[1], pool[2], pool[3], "", 1);

		skepu::external(
			[&]
			{
				auto i(0);
				for(auto & e : ten4)
					e = i++;
				std::cout << "ten4: " << ten4 << "\n";
			},
			skepu::write(ten4));
		
		conv4(ret_ten4, ten4, stencil4);
		skepu::io::cout << "Tensor4: " << ret_ten4 << "\n";
	}
	
	return 0;
}

