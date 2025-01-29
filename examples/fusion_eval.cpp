
#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/util.hpp>
#include <skepu-lib/filter.hpp>

#include "lodepng.h"


using namespace skepu::filter;


template<typename T>
struct PixelInfo {};

template<>
struct PixelInfo<GrayscalePixel>
{
	static constexpr LodePNGColorType type = LCT_GREY;
	static constexpr size_t bytes = 1;
};

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
		std::vector<unsigned char> image;
		unsigned imageWidth, imageHeight;
		unsigned error = lodepng::decode(image, imageWidth, imageHeight, filePath, PixelInfo<Pixel>::type);
		if (error) SKEPU_ERROR("decoder error " << error << ": " << lodepng_error_text(error));
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
		if (error) SKEPU_ERROR("decoder error " << error << ": " << lodepng_error_text(error));
	});
}






int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		skepu::io::cout << "Usage: " << argv[0] << "[backend]\n";
		exit(1);
	}
	skepu::BackendSpec spec;
	if (argc > 1) spec = skepu::BackendSpec{argv[1]};
	skepu::setGlobalBackendSpec(spec);
	
	const size_t repeats = 5;
	
	

	skepu::Matrix<RGBPixel> imageA;
	ReadPngFileToMatrix(imageA, "data/dragonfly.png");
	skepu::Matrix<RGBPixel> imageB = imageA, imageC = imageA;
	skepu::io::cout << "Image of height: " << imageA.total_rows() << " and width: " << imageA.total_cols() << "\n";
	
	
	
	/*
	auto pipeline =
		   skepu::Map(precision_extend_rgb)
		>> skepu::Map(gamma_expansion_ex)
 		>> skepu::Map(rgbex_to_hsv)
		>> skepu::Map(lighten_hsv_sp)   // Needs uniform
		>> skepu::Map(saturate_hsv_sp)  // Needs uniform
//		>> skepu::Map(blend_rgbex)     // Needs second elwise
		>> skepu::Map(hsv_to_rgbex)
		>> skepu::Map(gamma_compression_ex)
		>> skepu::Map(precision_reduce_rgb);
	
	auto timeFused = skepu::benchmark::measureExecTime([&]
	{
		pipeline(imageA, imageA);
	});
	std::cout << "Time fused: " << timeFused.count() / 1E6 << "\n";
	*/
	
	
	
	
	auto step1 = skepu::Map(precision_extend_rgb);
	auto step2 = skepu::Map(gamma_expansion_ex);
 	auto step3 = skepu::Map(rgbex_to_hsv);
	auto step4 = skepu::Map(lighten_hsv_sp);   // Needs uniform
	auto step5 = skepu::Map(saturate_hsv_sp);  // Needs uniform
//auto step6 = skepu::Map(blend_rgbex);     // Needs second elwise
	auto step7 = skepu::Map(hsv_to_rgbex);
	auto step8 = skepu::Map(gamma_compression_ex);
	auto step9 = skepu::Map(precision_reduce_rgb);
	
	auto temp1 = skepu::container_like<RGBExtended>(imageB);
	auto temp2 = skepu::container_like<HSVPixel>(imageB);
	
	auto timeNonFused = skepu::benchmark::basicBenchmark(repeats, 0, [&](size_t)
	{
		imageB.flush();
		temp1.flush();
		temp2.flush();
		step1(temp1, imageB);
		step2(temp1, temp1);
		step3(temp2, temp1);
		step4(temp2, temp2);
		step5(temp2, temp2);
	//	step6(temp2, temp2);
		step7(temp1, temp2);
		step8(temp1, temp1);
		step9(imageB, temp1);
		imageB.flush();
	});
	std::cout << "Time non-fused: " << timeNonFused.count() / 1E6 << "\n";
	
//	WritePngFileMatrix(imageB, "imageB.png");
	
	
	auto manual_pipeline = skepu::Map([](RGBPixel p) -> RGBPixel
	{
		RGBExtended ex = precision_extend_rgb(p);
		ex = gamma_expansion_ex(ex);
	 	HSVPixel hsv = rgbex_to_hsv(ex);
		hsv = lighten_hsv_sp(hsv);   // Needs uniform
		hsv = saturate_hsv_sp(hsv);  // Needs uniform
	//auto step6 = skepu::Map(blend_rgbex);     // Needs second elwise
		ex = hsv_to_rgbex(hsv);
		ex = gamma_compression_ex(ex);
		return precision_reduce_rgb(ex);
	});
	
	auto timeManuallyFused = skepu::benchmark::basicBenchmark(repeats, 0, [&](size_t)
	{
		imageC.flush();
		manual_pipeline(imageC, imageC);
		imageC.flush();
	});
	std::cout << "Time manually fused: " << timeManuallyFused.count() / 1E6 << "\n";
	
//	WritePngFileMatrix(imageC, "imageC.png");
	
	
}