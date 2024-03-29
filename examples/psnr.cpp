/*!
* PSNR (Peak Signal to Noise Ratio): The PSNR represents a measure of the peak error between the compressed and the original image.
* It is clsely related to MSE which represents the cumulative squared error between the compressed and the original image.
*/

#include <skepu>
#include <skepu-lib/util.hpp>
#include <skepu-lib/io.hpp>

[[skepu::userconstant]] constexpr int
	MAX = 255,
	NOISE = 10;

float diff_squared(int a, int b)
{
	return (a - b) * (a - b);
}

template<typename T>
T clamp_sum(T a, T b)
{
	T temp = a + b;
	return temp < 0 ? 0 : (temp > MAX ? MAX : temp);
}

float psnr(skepu::Matrix<int> &img, skepu::Matrix<int> noise)
{
	const size_t rows = img.size_i();
	const size_t cols = img.size_j();
	
	auto clamped_sum = skepu::Map(clamp_sum<int>);
	auto squared_diff_sum = skepu::MapReduce(diff_squared, skepu::util::add<float>);
	skepu::Matrix<int> comp_img(rows, cols);
	
	// Add noise
	clamped_sum(comp_img, img, noise);
	
	float mse = squared_diff_sum(img, comp_img) / (rows * cols);
	return 10 * log10((MAX * MAX) / mse);
}

int main(int argc, char *argv[])
{
	if (argc < 4)
	{
		skepu::io::cout << "Usage: " << argv[0] << " rows cols backend\n";
		exit(1);
	}
	
	const size_t rows = std::stoul(argv[1]);
	const size_t cols = std::stoul(argv[2]);
	auto spec = skepu::BackendSpec{argv[3]};
	skepu::setGlobalBackendSpec(spec);
	
	skepu::Matrix<int> img(rows, cols), noise(rows, cols);
	
	// Generate random image and random noise
	img.randomize(0, MAX);
	noise.randomize(-NOISE, NOISE);
	
	skepu::io::cout << "Actual image: " << img << "\n";
	
	float psnrval = psnr(img, noise);
	
	skepu::io::cout << "PSNR of two images: " << psnrval << "\n";
	
	return 0;
}
