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

template<typename Pixel>
void WritePngFilesTensor3(skepu::Tensor3<Pixel> &imageData, std::string filePath)
{
	skepu::external(skepu::read(imageData), [&]
	{
		for (size_t i = 0; i < imageData.size_i(); ++i)
		{
			std::stringstream fileName;
			fileName << filePath << std::setw(4) << std::setfill('0') << i << ".png";
			unsigned error = lodepng::encode(fileName.str(), reinterpret_cast<unsigned char*>(imageData.data() + i * imageData.size_j() * imageData.size_k()), imageData.size_j(), imageData.size_k(), PixelInfo<Pixel>::type);
			if (error) SKEPU_ERROR("decoder error " << error << ": " << lodepng_error_text(error));
		}
	});
}


float render_value_noise_3d(
	skepu::Random<1>& rnd
)
{
	return rnd.getNormalized();
}




using namespace skepu::filter;

#ifdef COLOR_OUTPUT
RGBPixel
#else
GrayscalePixel
#endif
to_pixel(float in, float scale)
{
#ifdef COLOR_OUTPUT
	HSVPixel temp;
	temp.h = (in * scale) * 360.f;
	temp.s = 0.8;
	temp.v = 0.8;
	RGBExtended temp1;
	RGBPixel res = hsv_to_rgb(temp);
#else
	GrayscalePixel res;
	res.intensity = (in * scale) * 255.f;
#endif
	return res;
}



int main(int argc, char* argv[])
{
	if (argc < 4)
	{
		skepu::io::cout << "Usage: " << argv[0] << " height width seed [backend]\n";
		exit(1);
	}
	
	skepu::BackendSpec spec;
	if (argc > 7) spec = skepu::BackendSpec{argv[7]};
	spec.setCPUPartitionRatio(0.1);
	skepu::setGlobalBackendSpec(spec);
	
	size_t textureHeight = atoi(argv[1]);
	size_t textureWidth = atoi(argv[2]);
	size_t textureDepth = 1;
	size_t seed = atoi(argv[3]);
	
	auto texture_renderer = skepu::Map<1>(to_pixel);
	skepu::PRNG prng(seed);
	float max_val = 1;
	
	
	
	skepu::Tensor3<float> noise(textureDepth, textureHeight, textureWidth, 0.f);
	auto noise_renderer = skepu::Map(render_value_noise_3d);
	
  noise_renderer.setPRNG(prng);
	
	auto duration = skepu::benchmark::measureExecTime([&]
	{
		noise_renderer(noise);
	});
	
	skepu::io::cout << "Duration: " << duration.count() / 1E6 << " s.\n";
	
//	skepu::io::cout << noise << "\n";
	
#ifdef COLOR_OUTPUT
	skepu::Tensor3<skepu::filter::RGBPixel>
#else
	skepu::Tensor3<skepu::filter::GrayscalePixel>
#endif
	texture(textureDepth, textureHeight, textureWidth);
	texture_renderer(texture, noise, 1 / max_val);
	WritePngFilesTensor3(texture, "uniform-frame");
	
	
	
	
	
	
	
	
	/*
	skepu::Matrix<float> noise(textureHeight, textureWidth, 0.f);
	
	bool gradient_mode = true;
	
	if (!gradient_mode)
	{
		auto grid_generator = skepu::Map(generate_grid_point_value);
		auto noise_renderer = skepu::Map(render_value_noise);
		
	  grid_generator.setPRNG(prng);
		
		for (size_t i = 2; i < octaves+2; ++i)
		{
			float frequency = base_frequency * pow(2, i);
			float amplitude = pow(persistence, i);
			size_t gridHeight = textureHeight / (textureWidth / frequency) + 1;
			size_t gridWidth = textureWidth / (textureWidth / frequency) + 1;
			skepu::Matrix<float> grid(gridHeight, gridWidth);
			
			grid_generator(grid);
			noise_renderer(noise, noise, grid, amplitude, textureHeight, textureWidth);
			max_val += amplitude;
		}
	}
	else
	{
		auto grid_generator = skepu::Map(generate_grid_point_gradient);
		auto noise_renderer = skepu::Map(render_gradient_noise);
		
	  grid_generator.setPRNG(prng);
		
		for (size_t i = 0; i < octaves; ++i)
		{
			float frequency = base_frequency * pow(2, i);
			float amplitude = pow(persistence, i);
			size_t gridHeight = textureHeight / (textureWidth / frequency) + 1;
			size_t gridWidth = textureWidth / (textureWidth / frequency) + 1;
			skepu::Matrix<vec2> grid(gridHeight, gridWidth);
			
			grid_generator(grid);
			noise_renderer(noise, noise, grid, amplitude, textureHeight, textureWidth);
			max_val += amplitude;
		}
	}
	
	skepu::Matrix<skepu::filter::GrayscalePixel> texture(textureHeight, textureWidth);
	texture_renderer(texture, noise, 1 / max_val);
	WritePngFileMatrix(texture, "perlin-noise.png");*/
	
	return 0;
}
