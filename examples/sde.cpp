#include <iostream>
#include <limits>
#include <algorithm>

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
	static constexpr unsigned char default_value = 0;
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

void normalize(skepu::Matrix<skepu::filter::GrayscalePixel>& img);

void applySobelEdgeDetection(
	skepu::Matrix<skepu::filter::RGBPixel>& input_img,
	skepu::Matrix<skepu::filter::GrayscalePixel>& output_img)
{
	auto intensity       = skepu::Map(skepu::filter::intensity_kernel);
	auto convolution     = skepu::MapOverlap(skepu::filter::convolution_kernel);
	auto distance        = skepu::Map<2>(skepu::filter::distance_kernel);

	skepu::Matrix<skepu::filter::GrayscalePixel>
		temp_img_a (input_img.total_rows(), input_img.total_cols()),
		temp_img_b (input_img.total_rows(), input_img.total_cols());

	intensity(temp_img_a, input_img);

	// Sobel edge detection
	skepu::Vector<float> averaging_filter {  1.0, 2.0, 1.0 };
	skepu::Vector<float> differentiation_filter { -1.0, 0.0, 1.0 };
	convolution.setOverlap(1);

	convolution.setOverlapMode(skepu::Overlap::RowWise);
	convolution(output_img, temp_img_a, differentiation_filter, 255.0, 0.5); // x-dir
	convolution(temp_img_b, temp_img_a, averaging_filter, 0, 0.25); // y-dir

	convolution.setOverlapMode(skepu::Overlap::ColWise);
	convolution(temp_img_a, output_img, averaging_filter, 0, 0.25); // x-dir
	convolution(output_img, temp_img_b, differentiation_filter, 255.0, 0.5); // y-dir

	// Final computation
	distance(output_img, temp_img_a, output_img);
	//normalize(output_img);
}

void applyGaussianBlur(
	skepu::Matrix<skepu::filter::RGBPixel>& img,
	float quantity)
{
	auto filter_gen      = skepu::Map<0>(skepu::filter::gauss_weights_kernel);
	auto convolution_rgb = skepu::MapOverlap(skepu::filter::convolution_kernel_rgb_1d);
	const size_t blur_radius = ceil(3.0 * quantity);
	skepu::Vector<float> filter(blur_radius * 2 + 1);
	filter_gen(filter, blur_radius, quantity);

	skepu::Matrix<skepu::filter::RGBPixel> temp(img.size_i(), img.size_j());

	convolution_rgb.setOverlap(blur_radius);
	convolution_rgb.setEdgeMode(skepu::Edge::Duplicate);

	convolution_rgb.setOverlapMode(skepu::Overlap::RowWise);
	convolution_rgb(temp, img, filter, 0, 1.0);
	convolution_rgb.setOverlapMode(skepu::Overlap::ColWise);
	convolution_rgb(img, temp, filter, 0, 1.0);
}

void normalize(skepu::Matrix<skepu::filter::GrayscalePixel>& img)
{
    int min = std::numeric_limits<unsigned char>::max();
    int max = std::numeric_limits<unsigned char>::min();

    // Determine the actual min and max values of the image
    for (size_t i = 0; i < img.size_i(); ++i)
        for (size_t j = 0; j < img.size_j(); ++j) {
            int intensity = img(i, j).intensity;
            if (intensity < min) min = intensity;
            if (intensity > max) max = intensity;
        }

    // Apply normalization if valid range
    if (max > min) {
        for (size_t i = 0; i < img.size_i(); ++i)
            for (size_t j = 0; j < img.size_j(); ++j)
                img(i, j).intensity = 255 * (img(i, j).intensity - min) / (max - min);
    }
}


// Moved this to global scope so they can be shared by both variants for now
static const int NUM_DISPARITIES = 64;
static const int MIN_DISPARITY_CONFIG = 0;
static const int BLOCK_SIZE = 17; // Must be odd
static const int PADDING = (BLOCK_SIZE-1) / 2; // The "-1" here is not necessary due to integer division, but it's there for clarity


// Non-SkePU variant
skepu::Matrix<skepu::filter::GrayscalePixel>
disparity_calc(
	skepu::Matrix<skepu::filter::GrayscalePixel>& left_img,
	skepu::Matrix<skepu::filter::GrayscalePixel>& right_img)
{
	skepu::Matrix<skepu::filter::GrayscalePixel> out_img(left_img.size_i(), left_img.size_j());

	// Initialize disparity map to minimum value (black)
	for (int i = 0; i < out_img.size_i(); i++)
	{
		for (int j = 0; j < out_img.size_j(); j++)
		{
			out_img(i, j).intensity = std::numeric_limits<unsigned char>::min();
		}
	}

	for (int i = PADDING; i < left_img.size_i() - PADDING; i++)
	{
		for (int j = PADDING + NUM_DISPARITIES; j < left_img.size_j() - PADDING; j++)
		{
			int min_sad = std::numeric_limits<int>::max();
			int min_disparity = 0;


			for (int d = MIN_DISPARITY_CONFIG; d < NUM_DISPARITIES; d++)
			{
				int sad = 0;
				// Calculate SAD for a block around the pixel
				for (int k = -PADDING; k <= PADDING; k++)
				{
					for (int l = -PADDING; l <= PADDING; l++)
					{
						sad += std::abs(left_img(i + k, j + l).intensity - right_img(i + k, j + l - d).intensity);
					}
				}
				if (sad < min_sad)
				{
					min_sad = sad;
					min_disparity = d;
				}
			}

			out_img(i, j).intensity = min_disparity * 255 / NUM_DISPARITIES;
		}
	}

	return out_img;
}


// User function for SkePU variant
namespace skepu { namespace filter {

	GrayscalePixel blockmatcher_uf(
		skepu::Index2D index,
		Mat<GrayscalePixel> left_img,
		Mat<GrayscalePixel> right_img,
		int padding,
		int num_disparities,
		int min_disparity_config)
	{
		int min_sad = 2 << 15; // just some large number for now, since we cannot use std::numeric_limits<int>::max();
		int min_disparity = 0;
		int i = index.row + padding;
		int j = index.col + padding;

		for (int d = min_disparity_config; d < num_disparities; d++)
		{
			int sad = 0;
			// Calculate SAD for a block around the pixel
			for (int k = -padding; k <= padding; k++)
			{
				for (int l = -padding; l <= padding; l++)
				{
					sad += abs(left_img(i + k, j + l).intensity - right_img(i + k, j + l - d).intensity);
				}
			}
			if (sad < min_sad)
			{
				min_sad = sad;
				min_disparity = d;
			}
		}

		GrayscalePixel p;
		p.intensity = min_disparity * 255 / num_disparities;
		return p;
	}

}} // namespace skepu::filter

// SkePU variant
skepu::Matrix<skepu::filter::GrayscalePixel>
disparity_calc_skepuized(
	skepu::Matrix<skepu::filter::GrayscalePixel>& left_img,
	skepu::Matrix<skepu::filter::GrayscalePixel>& right_img)
{
	skepu::Matrix<skepu::filter::GrayscalePixel> out_img(left_img.size_i() - PADDING*2, left_img.size_j() - PADDING*2);
	auto disparity_skel = skepu::Map<0>(skepu::filter::blockmatcher_uf);
	disparity_skel(out_img, left_img, right_img, PADDING, NUM_DISPARITIES, MIN_DISPARITY_CONFIG);
	return out_img;
}


//#define SDE_PREFILTER_BLUR 1.0

int main(int argc, char* argv[])
{
	if (argc < 5)
	{
		skepu::io::cout << "Usage: " << argv[0] << " left-image.png right-image.png output-file.png backend\n";
		exit(1);
	}

	std::string left_file(argv[1]);
	std::string right_file(argv[2]);
	std::string out_file(argv[3]);
	skepu::BackendSpec spec = skepu::BackendSpec{argv[4]};
	skepu::setGlobalBackendSpec(spec);

	// Use RGB matrices temporarily for reading and processing
	skepu::Matrix<skepu::filter::RGBPixel> temp_left_rgb, temp_right_rgb;
	ReadPngFileToMatrix(temp_left_rgb, left_file);
	ReadPngFileToMatrix(temp_right_rgb, right_file);

#ifdef SDE_PREFILTER_BLUR
	// Just for testing: see what happens if we apply blur filter first
	applyGaussianBlur(temp_left_rgb, SDE_PREFILTER_BLUR);
	applyGaussianBlur(temp_right_rgb, SDE_PREFILTER_BLUR);
#endif

	// Apply Sobel edge detection and store the results in grayscale matrices
  skepu::Matrix<skepu::filter::GrayscalePixel> left_img(temp_left_rgb.size_i(), temp_left_rgb.size_j());
	skepu::Matrix<skepu::filter::GrayscalePixel> right_img(temp_right_rgb.size_i(), temp_right_rgb.size_j());
	applySobelEdgeDetection(temp_left_rgb, left_img);
	applySobelEdgeDetection(temp_right_rgb, right_img);

	/* Disparity map calculation */
	//skepu::Matrix<skepu::filter::GrayscalePixel> out_img = disparity_calc(left_img, right_img); // raw C++
	skepu::Matrix<skepu::filter::GrayscalePixel> out_img = disparity_calc_skepuized(left_img, right_img); // SkePU

	WritePngFileMatrix(out_img, out_file);
}
