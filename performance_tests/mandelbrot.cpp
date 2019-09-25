/*!
Mandelbrot fractals. The Mandelbrot set
{
"B. B. Mandelbrot. Fractal aspects of the iteration of z → λz(1 − z) for complex λ and z.
Annals of the New York Academy of Sciences, 357:249–259, December 1980."
}
is a set of complex numbers which boundary draws a fractal in the complex numbers plane. A complex number c lies
within the Mandelbrot set, if the sequence
z_{i+1} = z_{i}2 + c
with i ∈ N, starting with z0 = 0 does not escape to infinity, otherwise c is not part of
the Mandelbrot set.

When computing a Mandelbrot fractal, the sequence in equation 3.1 is calculated
for every pixel of an image representing a section of the complex numbers plane. If a
given threshold is crossed, it is presumed that the sequence will escape to infinity and
that the pixel is not inside the Mandelbrot set. If the threshold is not crossed for a given
number of steps in the sequence, the pixel is taken as a member of the Mandelbrot
set. A pixel within the Mandelbrot set painted in black, other pixels are given a color
that corresponds to the number of sequence steps that have been calculated before
excluding the pixel from the Mandelbrot set. By setting the threshold and the number
of sequence steps accordingly, the calculation of the fractal can be a time-consuming
task. However, as all pixels are calculated independently, it is a common benchmark
application for data-parallel computations.
*/

#include <iostream>
#include <fstream>
#include <cstdlib>

#include <skepu>

#include "bitmap_image.hpp"
#include "performance_tests_common.hpp"


const size_t HEIGHT = 20;
const size_t WIDTH = 20;
std::string application = "Mandelbrot";


template<typename T>
void save_image(size_t width, size_t height, T *buf, T max)
{
	bitmap_image image(width, height);
	
	for (size_t y = 0; y < height; ++y)
	{
		for (size_t x = 0; x < width; ++x)
		{
			float val = buf[y*width + x];
			unsigned char shade = val / max * 255;
			rgb_store col = prism_colormap[shade];
			image.set_pixel(x, y, col.red, col.green, col.blue);
		}
	}
	
	image.save_image("generated_image.bmp");
}


[[skepu::userconstant]] constexpr float
	CENTER_X = -.5f,
	CENTER_Y = 0.f,
	SCALE = 2.5f;

[[skepu::userconstant]] constexpr size_t
	MAX_ITERS = 1000;

struct cplx 
{
	float a, b;
};

cplx mult_c(cplx lhs, cplx rhs)
{
	cplx r;
	r.a = lhs.a * rhs.a - lhs.b * rhs.b;
	r.b = lhs.b * rhs.a + lhs.a * rhs.b;
	return r;
}

cplx add_c(cplx lhs, cplx rhs)
{
	cplx r;
	r.a = lhs.a + rhs.a;
	r.b = lhs.b + rhs.b;
	return r;
}

size_t mandelbrot_f(skepu::Index2D index, size_t height, size_t width)
{
	cplx a;
	a.a = SCALE / height * (index.col - width/2.f) + CENTER_X;
	a.b = SCALE / height * (index.row - width/2.f) + CENTER_Y;
	cplx c = a;
	
	for (size_t i = 0; i < MAX_ITERS; ++i)
	{
		a = add_c(mult_c(a, a), c);
		if ((a.a * a.a + a.b * a.b) > 4)
			return i;
	}
	return MAX_ITERS;
}



auto mandelbroter = skepu::Map<0>(mandelbrot_f);

double mandelbrot() {
	skepu::Timer timer;
	for(size_t test = 0; test < NUM_REPEATS; ++test) {
		skepu::Matrix<size_t> iterations(HEIGHT, WIDTH);

		timer.start();
		mandelbroter(iterations, HEIGHT, WIDTH);
		timer.stop();
	}
	return timer.getMedianTime();
}



constexpr auto benchmarkFunc = mandelbrot;

void setBackend(const skepu::BackendSpec& spec) {
	mandelbroter.setBackend(spec);
}

void tune() {
	skepu::backend::tuner::hybridTune(mandelbroter, 16, 1, 32, 1024);
	mandelbroter.resetBackend();
}

int main(int argc, char* argv[]) {
	std::vector<double> times;
	
	std::cout << application << ": Running CPU backend" << std::endl;
	skepu::BackendSpec specCPU(skepu::Backend::Type::CPU);
	setBackend(specCPU);
	double cpuTime = benchmarkFunc();
	
	std::cout << application << ": Running OpenMP backend" << std::endl;
	skepu::BackendSpec specOpenMP(skepu::Backend::Type::OpenMP);
	specOpenMP.setCPUThreads(16);
	setBackend(specOpenMP);
	times.push_back(benchmarkFunc());
	
	std::cout << application << ": Running CUDA GPU backend" << std::endl;
	skepu::BackendSpec specGPU(skepu::Backend::Type::CUDA);
	specGPU.setDevices(1);
	setBackend(specGPU);
	times.push_back(benchmarkFunc());
	
	
	std::cout << application << ": Running Oracle" << std::endl;
	double bestOracleTime = 100000000.0;
	size_t bestOracleRatio = 9999999;
	std::vector<double> oracleTimes;
	
	for(size_t ratio = 0; ratio <= 100; ratio += 5) {
		double percentage = (double)ratio / 100.0;
		skepu::BackendSpec spec(skepu::Backend::Type::Hybrid);
		spec.setDevices(1);
		spec.setCPUThreads(16);
		spec.setCPUPartitionRatio(percentage);
		
		setBackend(spec);
		double time = benchmarkFunc();
		
		oracleTimes.push_back(time);
		if(time < bestOracleTime) {
			bestOracleTime = time;
			bestOracleRatio = ratio;
		}
		std::cout << "Ratio: " << percentage << " gave time: " << time << std::endl;
	}
	times.push_back(bestOracleTime);
	std::cout << "Optimal ratio was: " << bestOracleRatio << std::endl;
	
	std::cout << application << ": Running Hybrid backend" << std::endl;
	tune();
	times.push_back(benchmarkFunc());
	
	appendPerformanceResult("speedup_openmp.dat", application, cpuTime/times[0]);
	appendPerformanceResult("speedup_cuda.dat", application, cpuTime/times[1]);
	appendPerformanceResult("speedup_oracle.dat", application, cpuTime/times[2]);
	appendPerformanceResult("speedup_hybrid.dat", application, cpuTime/times[3]);
	
	return 0;
}
