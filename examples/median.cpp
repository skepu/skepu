#include <fstream>
#include <iostream>
#include <sstream>

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/filter.hpp>

#include "lodepng.h"

template<typename T>
struct PixelInfo {};

template<>
struct PixelInfo<skepu::filter::GrayscalePixel>
{
	static constexpr LodePNGColorType type = LCT_GREY;
	static constexpr size_t bytes = 1;
};

template<>
struct PixelInfo<skepu::filter::RGBPixel>
{
	static constexpr LodePNGColorType type = LCT_RGB;
	static constexpr size_t bytes = 3;
};

// Reads a file from png and retuns it as a skepu::Matrix. Uses a library called LodePNG.
template<typename Pixel>
void ReadPngFileToMatrix(skepu::Matrix<Pixel> &inputMatrix, std::string filePath)
{
	skepu::external([&]
	{
		std::vector<unsigned char> image;
		unsigned imageWidth, imageHeight;
		unsigned error = lodepng::decode(image, imageWidth, imageHeight, filePath, PixelInfo<Pixel>::type);
		if (error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
		inputMatrix.init(imageHeight, imageWidth);
		Pixel *imgView = reinterpret_cast<Pixel*>(image.data());
		std::copy(imgView, imgView + imageHeight * imageWidth, inputMatrix.data());
	}, skepu::write(inputMatrix));
}

template<typename Pixel>
void WritePngFileMatrix(skepu::Matrix<Pixel> &imageData, std::string filePath)
{
	skepu::external(skepu::read(imageData), [&]
	{
		unsigned error = lodepng::encode(filePath, reinterpret_cast<unsigned char*>(imageData.data()), imageData.total_cols(), imageData.total_rows(), PixelInfo<Pixel>::type);
		if (error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
	});
}


int main(int argc, char* argv[])
{
	if (argc < 4)
	{
		skepu::io::cout << "Usage: " << argv[0] << "input-image output-image radius [backend]\n";
		exit(1);
	}
	
	std::string inputFileName = argv[1];
	std::string outputFileName = argv[2];
	const int radius = atoi(argv[3]);
	skepu::BackendSpec spec;
	if (argc > 4) spec = skepu::BackendSpec{argv[4]};
	skepu::setGlobalBackendSpec(spec);
	
	auto calculateMedian = skepu::MapOverlap(skepu::filter::median_kernel);
	calculateMedian.setEdgeMode(skepu::Edge::Duplicate);
	
	// Create the full path for writing the image.
	std::stringstream ss;
	ss << (2 * radius + 1) << "x" << (2 * radius + 1);
	std::string outputFileNamePad = outputFileName + ss.str() + ".png";
	
	// Read the padded image into a matrix. Create the output matrix without padding.
	skepu::Matrix<skepu::filter::RGBPixel> inputImg;
	ReadPngFileToMatrix<skepu::filter::RGBPixel>(inputImg, inputFileName);
	skepu::Matrix<skepu::filter::RGBPixel> outputMatrix(inputImg.total_rows(), inputImg.total_cols());
	
	// Launch different kernels depending on filtersize.
	calculateMedian.setOverlap(radius, radius);
	calculateMedian(outputMatrix, inputImg);

	skepu::io::cout << "Inputfile : " << inputFileName << "\n";
	skepu::io::cout << "Filtersize : " << (2 * radius + 1) << "x" << (2 * radius + 1) << "\n";
	
	WritePngFileMatrix(outputMatrix, outputFileNamePad);
	
	return 0;
}
