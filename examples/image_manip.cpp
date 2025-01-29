#include <iostream>
#include <list>
#include <memory>

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/util.hpp>
#include <skepu-lib/filter.hpp>

#include "lodepng.h"

using namespace skepu::filter;

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



// Thresholding (B&W)
// HSV median filtering
// Add noise (uses PRNG)


auto extend = skepu::Map(skepu::filter::precision_extend_rgb);
auto reduce = skepu::Map(skepu::filter::precision_reduce_rgb);

auto intensity       = skepu::Map(skepu::filter::intensity_kernel_ex);
auto convolution     = skepu::MapOverlap(skepu::filter::convolution_kernel_ex);
auto convolution_rgb = skepu::MapOverlap(skepu::filter::convolution_kernel_rgbex_1d);
auto convolution_rgb2d = skepu::MapOverlap(skepu::filter::convolution_kernel_rgbex_2d);

auto random_dist_rgb = skepu::MapOverlap(skepu::filter::random_displacement_rgbex);

auto filter_gen      = skepu::Map<0>(skepu::filter::gauss_weights_kernel);
auto distance_grey   = skepu::Map<2>(skepu::filter::distance_kernel_ex);
auto blend_gray      = skepu::Map<2>(skepu::filter::blend_rgbex_gray);
auto blend           = skepu::Map<2>(skepu::filter::blend_rgbex);
auto blend_gray_c    = skepu::Map<1>(skepu::filter::blend_rgbex_gray);

auto to_hsv = skepu::Map(skepu::filter::rgbex_to_hsv);
auto to_rgb = skepu::Map(skepu::filter::hsv_to_rgbex);
auto get_rgb_channels = skepu::Map(skepu::filter::channels_from_rgbex);
auto get_hsv_channels = skepu::Map(skepu::filter::channels_from_hsv_ex);

auto lighten     = skepu::Map<1>(skepu::filter::lighten_rgbex);
auto darken      = skepu::Map<1>(skepu::filter::darken_rgbex);
auto saturate    = skepu::Map<1>(skepu::filter::saturate_rgbex);
auto desaturate  = skepu::Map<1>(skepu::filter::desaturate_rgbex);
auto colorrotate = skepu::Map<1>(skepu::filter::hue_rotate_rgbex);
auto contrast    = skepu::Map<1>(skepu::filter::contrast_rgbex);

auto closest_palette = skepu::Map(skepu::filter::find_closest<skepu::filter::RGBExtended>);
auto uniform_sample = skepu::Map(skepu::filter::uniform_sample_picker<skepu::filter::RGBExtended>);
auto closest_hue_palette = skepu::Map(skepu::filter::find_closest_hue);


skepu::Matrix<GrayscaleExtended> edgedetect(skepu::Matrix<RGBExtended> &image)
{
	auto output = skepu::container_like<skepu::filter::GrayscaleExtended>(image);
	auto tempA  = skepu::container_like(output);
	auto tempB  = skepu::container_like(output);
	intensity(tempA, image);
	
	// Sobel edge detection
	skepu::Vector<float> averaging_filter       {  1.0, 2.0, 1.0 };
	skepu::Vector<float> differentiation_filter { -1.0, 0.0, 1.0 };
	convolution.setOverlap(1);
	
	convolution.setOverlapMode(skepu::Overlap::RowWise);
	convolution(output, tempA, differentiation_filter, 255.0, 0.5); // x-dir
	convolution(tempB, tempA, averaging_filter, 0, 0.25); // y-dir
	
	// TODO: Debug for OpenCL:
	convolution.setOverlapMode(skepu::Overlap::ColWise);
	convolution(tempA, output, averaging_filter, 0, 0.25); // x-dir
	convolution(output, tempB, differentiation_filter, 255.0, 0.5); // y-dir
	
	// Final computation
	distance_grey(output, tempA, output);
	return std::move(output);
}


int main(int argc, char* argv[])
{
	skepu::BackendSpec spec;
	if (argc > 1) spec = skepu::BackendSpec{argv[1]};
	skepu::setGlobalBackendSpec(spec);
	
	auto quantify   = skepu::Map<1>(skepu::filter::quantify_ex);
	auto invert     = skepu::Map<1>(skepu::filter::invert_ex);
	
	auto calculateMedian = skepu::MapOverlap(skepu::filter::median_kernel_ex);
	calculateMedian.setEdgeMode(skepu::Edge::Duplicate);
	
	// Read the padded *image into a matrix. Create the output matrix without padding.
	auto image = std::make_shared<skepu::Matrix<skepu::filter::RGBExtended>>();
	std::list<std::shared_ptr<skepu::Matrix<RGBExtended>>> stack;
	std::string mode;
	
	while (mode != "quit")
	{
		std::cin >> mode;
		std::cout << "Command: " << mode << "\n";
		
		if (mode == "load")
		{
			std::string file_name;
			std::cin >> file_name;
			skepu::Matrix<skepu::filter::RGBPixel> byte_image;
			ReadPngFileToMatrix(byte_image, file_name);
			image = std::make_shared<skepu::Matrix<skepu::filter::RGBExtended>>(byte_image.size_i(), byte_image.size_j());
			extend(*image, byte_image);
			skepu::io::cout << "Read image of height: " << image->total_rows() << " and width: " << image->total_cols() << "\n";
		}
		else if (mode == "save")
		{
			std::string file_name;
			std::cin >> file_name;
			auto byte_image = skepu::container_like<skepu::filter::RGBPixel>(*image);
			reduce(byte_image, *image);
			WritePngFileMatrix(byte_image, file_name);
			skepu::io::cout << "Wrote image to file of height: " << image->total_rows() << " and width: " << image->total_cols() << "\n";
		}
		else if (mode == "push")
		{
			stack.push_front(image);
			
			skepu::io::cout << "Stored image on stack. Current depth: " << stack.size() << "\n";
		}
		else if (mode == "peek")
		{
			image = stack.front();
			skepu::io::cout << "Loaded image from top of stack. Current depth: " << stack.size() << "\n";
		}
		else if (mode == "pop")
		{
			stack.pop_front();
			skepu::io::cout << "Removed image on top of stack. Current depth: " << stack.size() << "\n";
		}
		else if (mode == "intensity")
		{
			auto output = skepu::container_like<skepu::filter::GrayscaleExtended>(*image);
			intensity(output, *image);
			
			// TODO back to *image
		}
		else if (mode == "invert")
		{
			invert(*image, *image);
		}
		else if (mode == "saturate")
		{
			float quantity;
			std::cin >> quantity;
			saturate(*image, *image, quantity);
		}
		else if (mode == "desaturate")
		{
			float quantity;
			std::cin >> quantity;
			desaturate(*image, *image, quantity);
		}
		else if (mode == "lighten")
		{
			float quantity;
			std::cin >> quantity;
			lighten(*image, *image, quantity);
		}
		else if (mode == "darken")
		{
			float quantity;
			std::cin >> quantity;
			darken(*image, *image, quantity);
		}
		else if (mode == "contrast")
		{
			float quantity;
			std::cin >> quantity;
			contrast(*image, *image, quantity);
		}
		else if (mode == "rotate-color")
		{
			float quantity;
			std::cin >> quantity;
			colorrotate(*image, *image, quantity);
		}
		else if (mode == "quantify")
		{
			float quantity;
			std::cin >> quantity;
			quantify(*image, *image, quantity);
		}
		else if (mode == "smart-quantify")
		{
			size_t quantity;
			std::cin >> quantity;
			skepu::Matrix<RGBExtended> samples(quantity, quantity);
			uniform_sample(samples, *image, samples.size_i(), samples.size_j());
			skepu::Vector<RGBExtended> palette (quantity * quantity);// = std::move(samples);
			auto reshape = skepu::Map(skepu::util::identity<RGBExtended>);
			reshape(palette, samples);
			closest_palette(*image, *image, palette);
		}
		else if (mode == "palette")
		{
			skepu::Vector<RGBExtended> palette = {
				{0,0,0},
			//	{.5,.5,.5},
				{1,1,1},
				{1,0,0},
				{0,1,0},
				{0,0,1},
				{1,1,0},
				{1,0,1},
				{0,1,1},
			};
			closest_palette(*image, *image, palette);
		}
		else if (mode == "hue-quantify")
		{
			skepu::Vector<HSVPixel> palette = {
				{142,0,0},
				{194,0,0},
				{185,0,0},
				{ 71,0,0},
				{132,0,0},
				{ 48,0,0},
				{ 57,0,0},
			};
			auto tempA  = skepu::container_like<skepu::filter::HSVPixel>(*image);
			to_hsv(tempA, *image);
			closest_hue_palette(tempA, tempA, palette);
			to_rgb(*image, tempA);
		}
		else if (mode == "convert")
		{
			auto temp1  = skepu::container_like(*image);
			auto temp2  = skepu::container_like(*image);
			auto temp3  = skepu::container_like(*image);
			auto output = skepu::container_like(*image);
			auto tempA  = skepu::container_like<skepu::filter::HSVPixel>(*image);
			auto tempB  = skepu::container_like<skepu::filter::HSVPixel>(*image);
			auto outputR  = skepu::container_like<skepu::filter::GrayscaleExtended>(*image);
			auto outputG  = skepu::container_like<skepu::filter::GrayscaleExtended>(*image);
			auto outputB  = skepu::container_like<skepu::filter::GrayscaleExtended>(*image);
			auto outputH  = skepu::container_like<skepu::filter::GrayscaleExtended>(*image);
			auto outputS  = skepu::container_like<skepu::filter::GrayscaleExtended>(*image);
			auto outputV  = skepu::container_like<skepu::filter::GrayscaleExtended>(*image);
			auto outputI  = skepu::container_like<skepu::filter::GrayscaleExtended>(*image);
			
			get_rgb_channels(outputR, outputG, outputB, *image);
			intensity(outputI, *image);
		/*	WritePngFileMatrix(outputR, outputFileName + "_01_red.png");
			WritePngFileMatrix(outputG, outputFileName + "_02_green.png");
			WritePngFileMatrix(outputB, outputFileName + "_03_blue.png");
			WritePngFileMatrix(outputI, outputFileName + "_04_intensity.png");
			*/
			to_hsv(tempA, *image);
			
			get_hsv_channels(outputH, outputS, outputV, tempA);
		/*	WritePngFileMatrix(outputH, outputFileName + "_05_hue.png");
			WritePngFileMatrix(outputS, outputFileName + "_06_saturation.png");
			WritePngFileMatrix(outputV, outputFileName + "_07_value.png");
			*/
			to_rgb(*image, tempA);
		}
		else if (mode == "displace")
		{
			size_t radius;
			std::cin >> radius;
			auto output = skepu::container_like(*image);
			random_dist_rgb.setOverlap(radius, radius);
			random_dist_rgb.setEdgeMode(skepu::Edge::Duplicate);
			random_dist_rgb(output, *image);
			//*image = output;
			lighten(*image, output, 0.0);
		}
		else if (mode == "comic")
		{
			auto temp1  = skepu::container_like(*image);
			auto temp2  = skepu::container_like(*image);
			auto temp3  = skepu::container_like(*image);
			
			quantify(temp1, *image, 4);
			
			calculateMedian.setOverlap(2, 2);
			calculateMedian(temp2, temp1);
			
			lighten(temp3, temp2, 0.3);
			saturate(temp2, temp3, 0.5);
			
			skepu::Matrix<GrayscaleExtended> edges = edgedetect(temp2);
			
			blend_gray(*image, temp2, edges, skepu::filter::MULTIPLY);
			blend_gray(*image, *image, edges, skepu::filter::MULTIPLY);
			blend_gray(*image, *image, edges, skepu::filter::MULTIPLY);
		}
		else if (mode == "sharpen")
		{
			auto output = skepu::container_like(*image);
			skepu::Matrix<float> sharpen_filter {
				 0.0, -1.0,  0.0,
				-1.0,  5.0, -1.0,
				 0.0, -1.0,  0.0
			};
			convolution_rgb2d.setOverlap(1, 1);
			convolution_rgb2d.setEdgeMode(skepu::Edge::Duplicate);
			convolution_rgb2d(output, *image, sharpen_filter, 0, 1);
			//*image = output;
			lighten(*image, output, 0.0);
		}
		else if (mode == "box-blur")
		{
			size_t quantity;
			std::cin >> quantity;
			auto temp   = skepu::container_like(*image);
			const size_t blur_radius = ceil(quantity);
			skepu::Vector<float> filter(blur_radius * 2 + 1, 1.f / (blur_radius * 2 + 1));
			
			convolution_rgb.setOverlap(blur_radius);
			convolution_rgb.setEdgeMode(skepu::Edge::Duplicate);
			
			convolution_rgb.setOverlapMode(skepu::Overlap::RowWise);
			convolution_rgb(temp, *image, filter, 0, 1.0);
			convolution_rgb.setOverlapMode(skepu::Overlap::ColWise);
			convolution_rgb(*image, temp, filter, 0, 1.0);
		}
		else if (mode == "gaussian-blur")
		{
			size_t quantity;
			std::cin >> quantity;
			auto temp   = skepu::container_like(*image);
			auto output = skepu::container_like(*image);
			const size_t blur_radius = ceil(3.0 * quantity);
			skepu::Vector<float> filter(blur_radius * 2 + 1);
			filter_gen(filter, blur_radius, quantity);
			
			convolution_rgb.setOverlap(blur_radius);
			convolution_rgb.setEdgeMode(skepu::Edge::Duplicate);
			
			convolution_rgb.setOverlapMode(skepu::Overlap::RowWise);
			convolution_rgb(temp, *image, filter, 0, 1.0);
			convolution_rgb.setOverlapMode(skepu::Overlap::ColWise);
			convolution_rgb(output, temp, filter, 0, 1.0);
			lighten(*image, output, 0.0);
		}
		else if (mode == "median")
		{
			auto temp1  = skepu::container_like(*image);
			size_t quantity;
			std::cin >> quantity;
			calculateMedian.setOverlap(quantity, quantity);
			calculateMedian(temp1, *image);
			lighten(*image, temp1, 0.0);
		}
		else if (mode == "unsharp-mask")
		{
			float quantity;
			std::cin >> quantity;
			auto temp1  = skepu::container_like(*image);
			auto temp2  = skepu::container_like(*image);
			auto temp3  = skepu::container_like(*image);
			auto temp4  = skepu::container_like(*image);
			auto output = skepu::container_like(*image);
			
			const size_t blur_radius = 1;//ceil(3.0 * quantity);
			skepu::Vector<float> filter(blur_radius * 2 + 1);
			filter_gen(filter, blur_radius, quantity);
			
			convolution_rgb.setOverlap(blur_radius);
			convolution_rgb.setEdgeMode(skepu::Edge::Duplicate);
			
			convolution_rgb.setOverlapMode(skepu::Overlap::RowWise);
			convolution_rgb(temp1, *image, filter, 0, 1.0);
			convolution_rgb.setOverlapMode(skepu::Overlap::ColWise);
			convolution_rgb(temp2, temp1, filter, 0, 1.0);
			
			blend(temp3, *image, temp2, skepu::filter::SUBTRACT);
			blend_gray_c(temp4, temp3, GrayscaleExtended{quantity}, skepu::filter::MULTIPLY);
			blend(output, *image, temp4, skepu::filter::ADDITION);
			
			*image = output;
		}
		else if (mode == "edge")
		{
			skepu::Matrix<GrayscaleExtended> output = edgedetect(*image);
			
			// TODO back to *image
		}
		else if (mode == "quit")
		{
			break;
		}
		else
		{
			skepu::io::cout << "Command not recongnized\n";
		}
		
	}
	
	return 0;
}
