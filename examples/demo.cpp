/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/*        GAME OF LIFE          */
/* SkePU demo for tutorial 2022 */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

#include <skepu>
#include <skepu-lib/io.hpp>

// Save output (8)
#include <skepu-lib/filter.hpp>
#include "lodepng.h"
void WritePngFileBinaryMatrix(skepu::Matrix<char> &imageData, size_t frame)
{
	using namespace skepu::filter;
	auto copy = skepu::container_like<GrayscalePixel>(imageData);
	auto multiplier = skepu::Map([](char e) -> GrayscalePixel { GrayscalePixel res; res.intensity = 255 - (e * 255); return res; });
	multiplier(copy, imageData);
	std::stringstream fileName;
	fileName << "demo/frame" << std::setw(4) << std::setfill('0') << frame << ".png";
	skepu::external(skepu::read(copy), [&]
	{
		unsigned error = lodepng::encode(fileName.str(), reinterpret_cast<unsigned char*>(copy.data()),
			copy.total_cols(), copy.total_rows(), LCT_GREY);
		if (error) SKEPU_ERROR("decoder error " << error << ": " << lodepng_error_text(error));
	});
}

// Default population generator user-function (7)
char initializer(skepu::Random<1> &prng)
{
	return (prng.get() % 100) < 10;
}

// Population update user-function (6)
char updater(skepu::Region2D<char> r)
{
	char neighbors = 0;
	neighbors += r(-1, -1) + r(-1, 0) + r(-1, +1);
	neighbors += r( 0, -1)            + r( 0, +1);
	neighbors += r(+1, -1) + r(+1, 0) + r(+1, +1);
	
	bool activate = !r(0, 0) && (neighbors == 3);
	bool stay_active = r(0, 0) && ((neighbors == 2) || (neighbors == 3));
	return (activate || stay_active) ? 1 : 0;
}


int main(int argc, char *argv[])
{
	// Command line inputs (1)
	if (argc < 5)
	{
	  skepu::io::cout << "Usage: " << argv[0] << " height width iterations backend\n";
	  exit(1);
	}
	const float height = atof(argv[1]);
	const float width = atof(argv[2]);
	const float iters = atof(argv[3]);
	auto spec = skepu::BackendSpec{argv[4]};
	spec.setCPUThreads(4);
	skepu::setGlobalBackendSpec(spec);
	
	// Skeletons (5)
	auto init = skepu::Map(initializer);
	auto update = skepu::MapOverlap(updater);
	update.setOverlap(1, 1);
	update.setEdgeMode(skepu::Edge::Cyclic);
	
	// Data containers (2)
	skepu::Matrix<char> current(height, width), next(height, width);
	
	// Default population (4)
	init(current);
	
	// Simulate evolution (3)
	for (size_t i = 0; i < iters; ++i)
	{
		WritePngFileBinaryMatrix(current, i);
	  update(next, current);
	  current.swap(next);
	}
  
	return 0;
}