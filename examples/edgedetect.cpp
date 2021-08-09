#include <iostream>

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
		skepu::io::cout << "Usage: " << argv[0] << "[blur|edge] input-image output-image blur-sigma [backend]\n";
		exit(1);
	}
	
	std::string mode(argv[1]);
	std::string inputFileName(argv[2]);
	std::string outputFileName(argv[3]);
	const float blur_sigma = atof(argv[4]);
	skepu::BackendSpec spec;
	if (argc > 5) spec = skepu::BackendSpec{argv[5]};
	skepu::setGlobalBackendSpec(spec);
	
	auto intensity       = skepu::Map(skepu::filter::intensity_kernel);
	auto convolution     = skepu::MapOverlap(skepu::filter::convolution_kernel);
	auto convolution_rgb = skepu::MapOverlap(skepu::filter::convolution_kernel_rgb);
	auto filter_gen      = skepu::Map<0>(skepu::filter::gauss_weights_kernel);
	auto distance        = skepu::Map<2>(skepu::filter::distance_kernel);
	
	// Read the padded image into a matrix. Create the output matrix without padding.
	skepu::Matrix<skepu::filter::RGBPixel> inputImg;
	ReadPngFileToMatrix<skepu::filter::RGBPixel>(inputImg, inputFileName);
	skepu::io::cout << "Image of height: " << inputImg.total_rows() << " and width: " << inputImg.total_cols() << "\n";
	
	// Gaussian blur
	if (mode == "blur")
	{
		const size_t blur_radius = ceil(3.0 * blur_sigma);
		skepu::Vector<float> filter(blur_radius * 2 + 1);
		filter_gen(filter, blur_radius, blur_sigma);
		
		skepu::Matrix<skepu::filter::RGBPixel>
			tempImgA (inputImg.total_rows(), inputImg.total_cols()),
			outputImg(inputImg.total_rows(), inputImg.total_cols());
		
		convolution_rgb.setOverlap(blur_radius);
		convolution_rgb.setEdgeMode(skepu::Edge::Duplicate);
		
		convolution_rgb.setOverlapMode(skepu::Overlap::RowWise);
		convolution_rgb(tempImgA, inputImg, filter, 0, 1.0);
		convolution_rgb.setOverlapMode(skepu::Overlap::ColWise);
		convolution_rgb(outputImg, tempImgA, filter, 0, 1.0);
		
		WritePngFileMatrix(outputImg, outputFileName + "_final.png");
	}
	else if (mode == "edge")
	{
		skepu::Matrix<skepu::filter::GrayscalePixel>
			tempImgA (inputImg.total_rows(), inputImg.total_cols()),
			tempImgB (inputImg.total_rows(), inputImg.total_cols()),
			outputImg(inputImg.total_rows(), inputImg.total_cols());
		
		intensity(tempImgA, inputImg);
		
		// Sobel edge detection
		skepu::Vector<float> averaging_filter {  1.0, 2.0, 1.0 };
		skepu::Vector<float> differentiation_filter { -1.0, 0.0, 1.0 };
		convolution.setOverlap(1);
		
		convolution.setOverlapMode(skepu::Overlap::RowWise);
		convolution(outputImg, tempImgA, differentiation_filter, 255.0, 0.5); // x-dir
		convolution(tempImgB, tempImgA, averaging_filter, 0, 0.25); // y-dir
		
		convolution.setOverlapMode(skepu::Overlap::ColWise);
		convolution(tempImgA, outputImg, averaging_filter, 0, 0.25); // x-dir
		convolution(outputImg, tempImgB, differentiation_filter, 255.0, 0.5); // y-dir
		
		// Final computation
		distance(outputImg, tempImgA, outputImg);
		WritePngFileMatrix(outputImg, outputFileName + "_final.png");
	}
	
	
	return 0;
}
