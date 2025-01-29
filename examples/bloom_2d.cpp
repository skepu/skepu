#include <skepu>
#include <skepu-lib/io.hpp>

#define USE_2D 1

#include "lodepng.h"

struct RGBPixel
{
	unsigned char r, g, b;
};

template<typename T>
struct PixelInfo {};

template<>
struct PixelInfo<RGBPixel>
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
		std::vector<unsigned char> image_buf;
		unsigned imageWidth, imageHeight;
		unsigned error = lodepng::decode(image_buf, imageWidth, imageHeight, filePath, PixelInfo<Pixel>::type);
		if (error) SKEPU_ERROR("decoder error " << error << ": " << lodepng_error_text(error));
		inputMatrix.init(imageHeight, imageWidth);
		Pixel *imgView = reinterpret_cast<Pixel*>(image_buf.data());
		std::copy(imgView, imgView + imageHeight * imageWidth, inputMatrix.data());
	}, skepu::write(inputMatrix));
}

template<typename Pixel>
void WritePngFileMatrix(skepu::Matrix<Pixel> &imageData, std::string filePath)
{
	skepu::external(skepu::read(imageData), [&]
	{
		unsigned error = lodepng::encode(filePath, reinterpret_cast<unsigned char*>(imageData.data()), imageData.total_cols(), imageData.total_rows(), PixelInfo<Pixel>::type);
		if (error) SKEPU_ERROR("decoder error " << error << ": " << lodepng_error_text(error));
	});
}

float gauss_weights_kernel(skepu::Index1D index, size_t r, float sigma)
{
	const float pi = 3.141592;
	float i = (float)index.i - r;
	return exp(-i*i / (2 * sigma * sigma)) / (sqrt(2 * pi) * sigma);
}

float gauss_weights_kernel_2(skepu::Index2D index, size_t r, float sigma)
{
	const float pi = 3.141592;
	float i = sqrt(((float)index.row - r) * ((float)index.row - r) + ((float)index.col - r) * ((float)index.col - r));
	return exp(-i*i / (2 * sigma * sigma)) / (sqrt(2 * pi) * sigma);
}

RGBPixel convolution_kernel(skepu::Region1D<RGBPixel> in, skepu::Vec<float> filter, float offset, float scaling, float avg)
{
	float r = 0;
	float g = 0;
	float b = 0;
	
	for (int i = -in.oi; i <= in.oi; i++)
	{
		RGBPixel p = in(i);
		float coeff = filter(i + in.oi);
		
		if ((p.r + p.g + p.b) / 3 < avg)
		{
			r += in(0).r * coeff;
			g += in(0).g * coeff;
			b += in(0).b * coeff;
		}
		else {
			r += p.r * coeff;
			g += p.g * coeff;
			b += p.b * coeff;
		}
	}
	
	RGBPixel result;
	result.r = (r + offset) * scaling;
	result.g = (g + offset) * scaling;
	result.b = (b + offset) * scaling;
	return result;
}

RGBPixel convolution_kernel_2(skepu::Region2D<RGBPixel> in, skepu::Mat<float> filter, float offset, float scaling, float avg)
{
	float r = 0;
	float g = 0;
	float b = 0;
	
	for (int i = -in.oi; i <= in.oi; i++)
	{
		for (int j = -in.oj; j <= in.oj; j++)
		{
			RGBPixel p = in(i, j);
			float coeff = filter(i + in.oi, j + in.oj);
			
			if ((p.r + p.g + p.b) / 3 < avg)
			{
				r += in(0, 0).r * coeff;
				g += in(0, 0).g * coeff;
				b += in(0, 0).b * coeff;
			}
				else {
				r += p.r * coeff;
				g += p.g * coeff;
				b += p.b * coeff;
			}
		}
	}
	
	RGBPixel result;
	result.r = (r + offset) * scaling;
	result.g = (g + offset) * scaling;
	result.b = (b + offset) * scaling;
	return result;
}

float intensity(RGBPixel p)
{
	return ((float)p.r + (float)p.g + (float)p.b) / 3;
}

float sum(float l, float r)
{
	return l + r;
}


int main(int argc, char *argv[])
{
	if (argc < 4)
	{
	  skepu::io::cout << "Usage: " << argv[0] << " in out radius\n";
	  exit(1);
	}
	
	std::string in{argv[1]};
	std::string out{argv[2]};
	const size_t blur_radius = atoi(argv[3]);
	
	skepu::Matrix<::RGBPixel> image;
	ReadPngFileToMatrix(image, in);
	skepu::io::cout << "Read image of height: " << image.total_rows() << " and width: " << image.total_cols() << "\n";
	skepu::Matrix<::RGBPixel> temp(image.total_rows(), image.total_cols());
	
	auto total_intensity = skepu::MapReduce(intensity, sum);
	
	float avg_intensity = total_intensity(image) / (image.total_rows() * image.total_cols());
	std::cout << "Avg intensity: " << avg_intensity << std::endl;
  
#ifndef USE_2D
	skepu::Vector<float> filter(blur_radius * 2 + 1);
	
	auto filter_gen = skepu::Map<0>(gauss_weights_kernel);
	auto convolution_rgb = skepu::MapOverlap(convolution_kernel);
	
	filter_gen(filter, blur_radius, 1);
	skepu::io::cout << filter << "\n";
	
	convolution_rgb.setOverlap(blur_radius);
	convolution_rgb.setEdgeMode(skepu::Edge::Duplicate);
	
	convolution_rgb.setOverlapMode(skepu::Overlap::RowWise);
	convolution_rgb(temp, image, filter, 0, 1.0, avg_intensity);
	convolution_rgb.setOverlapMode(skepu::Overlap::ColWise);
	convolution_rgb(image, temp, filter, 0, 1.0, avg_intensity);
	
	WritePngFileMatrix(image, out);
	
#else

	skepu::Matrix<float> filter(blur_radius * 2 + 1, blur_radius * 2 + 1);

	auto filter_gen = skepu::Map<0>(gauss_weights_kernel_2);
	auto convolution_rgb = skepu::MapOverlap(convolution_kernel_2);

	filter_gen(filter, blur_radius, sqrt(blur_radius));
	skepu::io::cout << filter << "\n";
	std::cout << "Filter mid: " << filter(blur_radius, blur_radius) << "\n";
	
	auto red = skepu::Reduce(sum);
	float filter_w = red(filter);
	std::cout << "Filter sum: " << filter_w << "\n";
	auto mapper = skepu::Map<1>([](float a, float w) { return a / w; });
	mapper(filter, filter, filter_w);
	filter_w = red(filter);
	std::cout << "Filter sum: " << filter_w << "\n";

	convolution_rgb.setOverlap(blur_radius);
	convolution_rgb.setEdgeMode(skepu::Edge::Duplicate);
	
	convolution_rgb(temp, image, filter, 0, 1.0, avg_intensity);

	WritePngFileMatrix(temp, out);
	
#endif
	
	skepu::io::cout << "Wrote image to file of height: " << image.total_rows() << " and width: " << image.total_cols() << "\n";
	
	return 0;
}