#include <iostream>

#include <skepu>
#include <skepu-lib/io.hpp>

float over_1d(skepu::Pool1D<int> p, int scale)
{
	std::cout << p;
	return (p(0) + p(1)*2 + p(2)*2 + p(3)) / scale;
}

float over_2d(skepu::Pool2D<int> p, const skepu::Mat<float> stencil)
{
	std::cout << p;
	float res = 999999999;
	for (int i = 0; i < p.si; ++i)
		for (int j = 0; j < p.sj; ++j)
			res = std::min(res, p(i, j) * stencil(i, j));
	return res;
}

float over_3d(skepu::Pool3D<int> p, skepu::Ten3<float> stencil)
{
	std::cout << p;
	float res = 0;
	for (int i = 0; i < p.si; ++i)
		for (int j = 0; j < p.sj; ++j)
			for (int k = 0; k < p.sk; ++k)
				res += p(i, j, k) * stencil(i, j, k);
	return res;
}

float over_4d(skepu::Pool4D<int> p, skepu::Ten4<float> stencil)
{
	std::cout << p;
	float res = 0;
	for (int i = 0; i < p.si; ++i)
		for (int j = 0; j < p.sj; ++j)
			for (int k = 0; k < p.sk; ++k)
				for (int l = 0; l < p.sl; ++l)
					res += p(i, j, k, l) * stencil(i, j, k, l);
	return res;
}




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
	
	skepu::Vector<int> v(size);
	skepu::Matrix<int> m(size, size);
	
	skepu::external(
		skepu::read(v),
		[&]
		{
			for (size_t i = 0; i < size; ++i)
				v(i) = i;

			std::cout << "v: " << v <<"\n";
		},
		skepu::write(v)
	);
	
	skepu::external(
		skepu::read(m),
		[&]
		{
			int i = 0;
			for(size_t y = 0; y < size; ++y)
				for(size_t x = 0; x < size; ++x)
					m(y, x) = i++;
			std::cout << "m: " << m <<"\n";
		},
		skepu::write(m)
	);
	
	// 1D
/*	{
		auto conv = skepu::MapPool(over_1d);
		conv.setPoolSize(4);

		skepu::Vector<float> rv(size);
		
		conv(rv, v, 13);
		skepu::external(skepu::read(rv), [&]{
			std::cout << "Vector None:    rv = " << rv << "\n";});

		skepu::Matrix<float> rm((size - 4 + 1) / 1, (size - 4 + 1) / 1);
		
		conv.setPoolSize(4);
		conv.setOverlapMode(skepu::Overlap::RowWise);
		
		conv(rm, m, 13);
		skepu::external(skepu::read(rm), [&]{
			std::cout << "Matrix Row-wise None:    rm = " << rm << "\n"; });


		conv.setPoolSize(4);
		conv.setOverlapMode(skepu::Overlap::ColWise);
		
		conv(rm, m, 13);
		std::cout << "Matrix Col-wise None:    rm = " << rm << "\n";
	}*/
	
	// 2D
	{
		size_t stride[2] = {3, 1};
		size_t pool[2] = {2, 2};
		auto conv2 = skepu::MapPool(over_2d);
		conv2.setPoolSize(pool[0], pool[1]);
		conv2.setStride(stride[0], stride[1]);

		skepu::Matrix<float> filter(pool[0], pool[1], 1);
		skepu::Matrix<float> rm((size - pool[0] + stride[0]) / stride[0], (size - pool[1] + stride[1]) / stride[1]);
		
		std::cout << "Matrix 2D: m = " << m << "\n";
		std::cout << "Matrix 2D: filter = " << filter << "\n";
		std::cout << "Matrix 2D: rm = " << rm << "\n";
		conv2(rm, m, filter);
		std::cout << "Matrix 2D: rm = " << rm << "\n";
	}
	
	// 3D
	{
		size_t stride[3] = {1, 1, 1};
		size_t pool[3] = {3, 3, 3};
		auto conv3 = skepu::MapPool(over_3d);
		conv3.setPoolSize(pool[0], pool[1], pool[2]);
		conv3.setStride(stride[0], stride[1], stride[2]);

		skepu::Tensor3<int> ten3(size, size, size, 1);
		skepu::Tensor3<float> ret_ten3(
			(size - pool[0] + stride[0]) / stride[0],
			(size - pool[1] + stride[1]) / stride[1],
			(size - pool[2] + stride[2]) / stride[2]
		);
		skepu::Tensor3<float> stencil3(pool[0], pool[1], pool[2], 1);
		
		skepu::external(
			[&]
			{
				auto i(0);
				for(auto & e : ten3)
					e = i++;
			},
			skepu::write(ten3));
			
		skepu::io::cout << "\n--------------------\nTensor3 Map Pool edge mode NONE\n";
		skepu::io::cout << "Tensor3: in = " << ten3 << "\n";
		skepu::io::cout << "Tensor3: stencil = " << stencil3 << "\n";
		skepu::io::cout << "Tensor3: out = " << ret_ten3 << "\n";
		conv3(ret_ten3, ten3, stencil3);
		skepu::io::cout << "Tensor3: " << ret_ten3 << "\n";
	}
	
	
	// 3D with edge mode pad
	{
		size_t stride[3] = {1, 1, 1};
		size_t pool[3] = {3, 3, 3};
		auto conv3 = skepu::MapPool(over_3d);
		conv3.setPoolSize(pool[0], pool[1], pool[2]);
		conv3.setStride(stride[0], stride[1], stride[2]);
		conv3.setEdgeMode(skepu::Edge::Pad);
		conv3.setPad(0);

		skepu::Tensor3<int> ten3(size, size, size, 1);
		skepu::Tensor3<float> ret_ten3(
			(size - pool[0] + stride[0]) / stride[0] + 2 * (pool[0] - 1),
			(size - pool[1] + stride[1]) / stride[1] + 2 * (pool[1] - 1),
			(size - pool[2] + stride[2]) / stride[2] + 2 * (pool[2] - 1)
		);
		skepu::Tensor3<float> stencil3(pool[0], pool[1], pool[2], 1);
		
		skepu::external(
			[&]
			{
				auto i(0);
				for(auto & e : ten3)
					e = i++;
			},
			skepu::write(ten3));
			
		skepu::io::cout << "\n--------------------\nTensor3 Map Pool edge mode PAD 0\n";
		skepu::io::cout << "Tensor3: in = " << ten3 << "\n";
		skepu::io::cout << "Tensor3: stencil = " << stencil3 << "\n";
		skepu::io::cout << "Tensor3: out = " << ret_ten3 << "\n";
		conv3(ret_ten3, ten3, stencil3);
		skepu::io::cout << "Tensor3: " << ret_ten3 << "\n";
	}
	
	
	// 4D
	{
		size_t stride[4] = {2, 2, 2, 2};
		size_t pool[4] = {2, 1, 1, 1};
		auto conv4 = skepu::MapPool(over_4d);
		conv4.setPoolSize(pool[0], pool[1], pool[2], pool[3]);
		conv4.setStride(stride[0], stride[1], stride[2], stride[3]);

		skepu::Tensor4<int> ten4(size, size, size, size);
		skepu::Tensor4<float> ret_ten4(
			(size - pool[0] + stride[0]) / stride[0],
			(size - pool[1] + stride[1]) / stride[1],
			(size - pool[2] + stride[2]) / stride[2],
			(size - pool[3] + stride[3]) / stride[3]
		);
		skepu::Tensor4<float> stencil4(pool[0], pool[1], pool[2], pool[3], 1);

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

