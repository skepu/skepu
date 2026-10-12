// Heat diffusion in 1 to 4 dimensions: the steady state of a body with fixed temperatures on its boundary, found by
// relaxation. Each element of the domain is repeatedly replaced by the average of its neighbours (the Laplace equation).
//
// Two ways of updating the domain, to compare their convergence:
//
//   normal    Jacobi iteration. An iteration reads the old domain and writes a second one (double-buffered), and the two
//             are swapped.
//   redblack  Gauss-Seidel iteration with a red-black ordering. The domain is colored like a chess board (by the parity of
//             the sum of the indices of an element). One iteration updates the red elements and then the black ones, in
//             place and from the latest values, so a single domain is enough. The elements of one color do not depend on
//             each other, so each half-iteration is as parallel as a Jacobi iteration. It uses
//             skepu::UpdateMode::RedBlack, and usually converges in about half as many iterations.
//
// The boundary of the domain is not computed, it keeps its values: the edge mode None on an input of the size of the
// output (the "mirror mode" of MapOverlap) only updates the interior.
//
// Every few iterations the program prints the largest change of any element during the last iteration. The user function
// returns it together with the new temperature, so it costs one reduction and no extra pass over the domain.
//
//   Usage: heat_diffusion dim size iterations backend [normal|redblack] [tolerance]
//
// 'iterations' is the largest number of iterations; with a tolerance the program stops when the largest change is below it.

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include <skepu>
#include <skepu-lib/io.hpp>

// The user functions return the new temperature of an element, the average of its neighbours, and how much the temperature changed
float absolute(float x)
{
	return x < 0 ? -x : x;
}

skepu::multiple<float, float> heat1D(skepu::Region1D<float> r)
{
	const float average = (r(-1) + r(1)) / 2;
	return skepu::ret(average, absolute(average - r(0)));
}

skepu::multiple<float, float> heat2D(skepu::Region2D<float> r)
{
	const float average = (r(-1, 0) + r(1, 0) + r(0, -1) + r(0, 1)) / 4;
	return skepu::ret(average, absolute(average - r(0, 0)));
}

skepu::multiple<float, float> heat3D(skepu::Region3D<float> r)
{
	const float average = (r(-1, 0, 0) + r(1, 0, 0) + r(0, -1, 0) + r(0, 1, 0) + r(0, 0, -1) + r(0, 0, 1)) / 6;
	return skepu::ret(average, absolute(average - r(0, 0, 0)));
}

skepu::multiple<float, float> heat4D(skepu::Region4D<float> r)
{
	const float average = (r(-1, 0, 0, 0) + r(1, 0, 0, 0) + r(0, -1, 0, 0) + r(0, 1, 0, 0)
		+ r(0, 0, -1, 0) + r(0, 0, 1, 0) + r(0, 0, 0, -1) + r(0, 0, 0, 1)) / 8;
	return skepu::ret(average, absolute(average - r(0, 0, 0, 0)));
}

float largest(float a, float b)
{
	return a > b ? a : b;
}


struct Options
{
	size_t size;
	size_t iterations;
	bool red_black;
	float tolerance;
};

// Relaxes the domain until the iterations are used or the change is below the tolerance. 'update' writes the new domain
// and the changes of the elements. 'domain' has the boundary conditions. 'change' is 0: the boundary is not computed, so
// its changes stay 0.
template<typename Container, typename Update, typename MaxChange>
void relax(const Options &opt, Update &update, MaxChange &max_change, Container &domain, Container &change, size_t elements)
{
	// Jacobi needs a second domain, with the same boundary. Red-black updates the domain in place.
	std::unique_ptr<Container> other;
	if (!opt.red_black)
	{
		other = std::make_unique<Container>(domain);
		other->setLabel("Domain B");
		SKEPU_TRACE_VIRTUAL_DATASET("Domain", domain, *other);
	}

	Container *current = &domain, *next = other.get();
	// The change is measured every 10 iterations (it takes a reduction), and printed about ten times
	const size_t check_interval = std::min<size_t>(10, std::max<size_t>(1, opt.iterations));
	const size_t print_interval = std::max<size_t>(check_interval, opt.iterations / 10 / check_interval * check_interval);

	skepu::io::cout << "iteration  largest change\n";
	size_t done = 0;
	for (size_t i = 1; i <= opt.iterations; ++i)
	{
		if (opt.red_black)
			update(*current, change, *current);
		else
		{
			update(*next, change, *current);
			std::swap(current, next);
		}
		done = i;

		if (i == 1 || i % check_interval == 0 || i == opt.iterations)
		{
			const float largest_change = max_change(change);
			const bool converged = largest_change < opt.tolerance;
			if (i == 1 || i % print_interval == 0 || i == opt.iterations || converged)
				skepu::io::cout << i << "  " << largest_change << "\n";
			SKEPU_TRACE_SNAPSHOT(*current);
			if (converged)
				break;
		}
	}

	// (with the double-buffered update the result is in one of the two domains)
	skepu::io::cout << "Done after " << done << " iterations, " << (opt.red_black ? "red-black" : "double-buffered") << " update.\n";
	if (elements <= 256)
		skepu::io::cout << *current << "\n";
}


void run1D(const Options &opt)
{
	auto update = skepu::MapOverlap(heat1D);
	update.setOverlap(1);
	update.setEdgeMode(skepu::Edge::None);
	update.setUpdateMode(opt.red_black ? skepu::UpdateMode::RedBlack : skepu::UpdateMode::Normal);
	auto max_change = skepu::Reduce(largest);

	// A rod, 0 at one end and 5 at the other
	skepu::Vector<float> domain(opt.size, "Domain", 0);
	skepu::external([&]
	{
		domain(opt.size - 1) = 5;
	}, skepu::write(domain));

	skepu::Vector<float> change(opt.size, "Change", 0);
	relax(opt, update, max_change, domain, change, opt.size);
}

void run2D(const Options &opt)
{
	auto update = skepu::MapOverlap(heat2D);
	update.setOverlap(1, 1);
	update.setEdgeMode(skepu::Edge::None);
	update.setUpdateMode(opt.red_black ? skepu::UpdateMode::RedBlack : skepu::UpdateMode::Normal);
	auto max_change = skepu::Reduce(largest);

	// A plate, 0 at the top edge, 5 at the bottom edge, and 2 at the sides
	const size_t size = opt.size;
	skepu::Matrix<float> domain(size, size, "Domain", 0);
	skepu::external([&]
	{
		for (size_t i = 0; i < size; ++i)
		{
			domain(i, 0) = 2;
			domain(i, size - 1) = 2;
			domain(size - 1, i) = 5;
		}
	}, skepu::write(domain));

	skepu::Matrix<float> change(size, size, "Change", 0);
	relax(opt, update, max_change, domain, change, size * size);
}

void run3D(const Options &opt)
{
	auto update = skepu::MapOverlap(heat3D);
	update.setOverlap(1, 1, 1);
	update.setEdgeMode(skepu::Edge::None);
	update.setUpdateMode(opt.red_black ? skepu::UpdateMode::RedBlack : skepu::UpdateMode::Normal);
	auto max_change = skepu::Reduce(largest);

	// A block, with a different temperature on each face
	const size_t size = opt.size;
	skepu::Tensor3<float> domain(size, size, size, "Domain", 0);
	skepu::external([&]
	{
		for (size_t i = 0; i < size; ++i)
			for (size_t j = 0; j < size; ++j)
			{
				domain(0, i, j) = 1;
				domain(size - 1, i, j) = 5;
				domain(i, 0, j) = 2;
				domain(i, size - 1, j) = 2;
			}
	}, skepu::write(domain));

	skepu::Tensor3<float> change(size, size, size, "Change", 0);
	relax(opt, update, max_change, domain, change, size * size * size);
}

void run4D(const Options &opt)
{
	auto update = skepu::MapOverlap(heat4D);
	update.setOverlap(1, 1, 1, 1);
	update.setEdgeMode(skepu::Edge::None);
	update.setUpdateMode(opt.red_black ? skepu::UpdateMode::RedBlack : skepu::UpdateMode::Normal);
	auto max_change = skepu::Reduce(largest);

	// Two opposite faces at 1 and 5, the rest of the boundary at 0
	const size_t size = opt.size;
	skepu::Tensor4<float> domain(size, size, size, size, "Domain", 0);
	skepu::external([&]
	{
		for (size_t i = 0; i < size; ++i)
			for (size_t j = 0; j < size; ++j)
				for (size_t k = 0; k < size; ++k)
				{
					domain(0, i, j, k) = 1;
					domain(size - 1, i, j, k) = 5;
				}
	}, skepu::write(domain));

	skepu::Tensor4<float> change(size, size, size, size, "Change", 0);
	relax(opt, update, max_change, domain, change, size * size * size * size);
}


int main(int argc, char *argv[])
{
	if (argc < 5 || argc > 7)
	{
		skepu::io::cout << "Usage: " << argv[0] << " dim size iterations backend [normal|redblack] [tolerance]\n"
			<< "  normal     double-buffered (Jacobi) update (default)\n"
			<< "  redblack   in-place red-black (Gauss-Seidel) update\n";
		exit(1);
	}

	const int dim = atoi(argv[1]);
	Options opt;
	opt.size = atol(argv[2]);
	opt.iterations = atol(argv[3]);
	opt.red_black = false;
	opt.tolerance = 0;

	if (argc > 5)
	{
		const std::string mode = argv[5];
		if (mode == "redblack" || mode == "red-black")
			opt.red_black = true;
		else if (mode != "normal")
		{
			skepu::io::cout << "Unknown update mode '" << mode << "': use normal or redblack\n";
			exit(1);
		}
	}
	if (argc > 6)
		opt.tolerance = atof(argv[6]);

	if (opt.size < 3)
	{
		skepu::io::cout << "The size must be at least 3 (the boundary is not computed)\n";
		exit(1);
	}

	auto spec = skepu::BackendSpec{argv[4]};
	skepu::setGlobalBackendSpec(spec);

	if (dim == 1) { run1D(opt); return 0; }
	if (dim == 2) { run2D(opt); return 0; }
	if (dim == 3) { run3D(opt); return 0; }
	if (dim == 4) { run4D(opt); return 0; }

	skepu::io::cout << "The dimension must be 1 to 4\n";
	return 1;
}
