// Particle swarm optimization: minimizes a test function of several variables with a swarm of particles that move through the
// search space, each one pulled towards the best point that it has found itself and the best point that the swarm has found.
//
// The swarm is a Matrix with a row for each particle (positions, velocities and the best position of each particle). A step is
//   - a Map over all the elements of the swarm, which moves the particles: it draws random numbers from the PRNG of the Map
//     (Random<2>) and reads the best point of the swarm with a random-access proxy (Vec),
//   - a Map over the particles (the vector of results) that evaluates the test function on the row of each (MatRow),
//   - Maps that keep the best position of each particle (the small user functions are lambdas),
//   - a Reduce and a MapReduce that find the best particle: the lowest value, and the lowest index of a particle that has it
//     (so the result does not depend on how the work is divided), and a Map that copies its position to the best point.
//
// Usage: pso particles dimensions iterations [backend] [--function NAME] [--seed N] [--tolerance X]
#include <algorithm>
#include <cfloat>
#include <climits>
#include <iostream>
#include <string>

#include <skepu>
#include <skepu-lib/args.hpp>
#include <skepu-lib/benchmark.hpp>
#include <skepu-lib/io.hpp>
#include <skepu-lib/randomize.hpp>
#include <skepu-lib/util.hpp>


// The test functions, which all have their minimum 0 in the middle of the search space (the origin, but the point (1, ..., 1) for
// rosenbrock): 0 sphere, 1 rastrigin, 2 ackley, 3 rosenbrock
float test_function(skepu::MatRow<float> x, int function)
{
	const float pi = 3.14159265f;
	const int n = (int)x.cols;
	float sum = 0.f, sum_cos = 0.f;
	for (int i = 0; i < n; ++i)
	{
		const float v = x(i);
		if (function == 0)      sum += v * v;
		else if (function == 1) sum += v * v - 10.f * cos(2.f * pi * v) + 10.f;
		else if (function == 2) { sum += v * v; sum_cos += cos(2.f * pi * v); }
		else if (i + 1 < n)     sum += 100.f * (x(i + 1) - v * v) * (x(i + 1) - v * v) + (1.f - v) * (1.f - v);
	}
	if (function == 2)
		return -20.f * exp(-0.2f * sqrt(sum / n)) - exp(sum_cos / n) + 20.f + 2.718281828f;
	return sum;
}

// Moves a particle's coordinate: the new velocity is a part of the old one and a pull towards the best point of the particle
// itself and one towards the best point of the swarm (each with a random strength); a particle that leaves the search space
// stops at its edge
skepu::multiple<float, float> move_particle(
	skepu::Index2D index, skepu::Random<2> &prng,
	float x, float v, float best_of_particle, skepu::Vec<float> best_of_swarm,
	float inertia, float cognitive, float social, float lower, float upper)
{
	const float r1 = prng.getNormalized();
	const float r2 = prng.getNormalized();
	float velocity = inertia * v + cognitive * r1 * (best_of_particle - x) + social * r2 * (best_of_swarm(index.col) - x);
	float position = x + velocity;
	if (position < lower)      { position = lower; velocity = 0.f; }
	else if (position > upper) { position = upper; velocity = 0.f; }
	return skepu::ret(position, velocity);
}

// The swarm: a row for each particle in the matrices
struct Swarm
{
	skepu::Matrix<float> positions, velocities, best_positions;
	skepu::Vector<float> value, best_value, best_point; // the value of each particle, the best value of each, the best point of the swarm
	float lowest_value = FLT_MAX;                       // the value of the best point of the swarm

	Swarm(size_t particles, size_t dimensions)
	: positions(particles, dimensions, "positions", 0.f), velocities(particles, dimensions, "velocities", 0.f),
	  best_positions(particles, dimensions, "best positions", 0.f), value(particles, "values"),
	  best_value(particles, "best values", FLT_MAX), best_point(dimensions, "best point") {}
};

// The skeletons are made where they are called (anonymous skeleton instances: one instance of each is made, whatever the number of
// calls; the precompiler makes them in functions, not in lambdas or templates). The small user functions are lambdas, and the
// settings of a skeleton (its PRNG, the start value of a reduction) are set in the same expression.

// Moves the particles
static void move_swarm(Swarm &swarm, skepu::PRNG &prng, float inertia, float cognitive, float social, float lower, float upper)
{
	skepu::Map<3>(move_particle).setPRNG(prng)(swarm.positions, swarm.velocities, swarm.positions, swarm.velocities, swarm.best_positions,
		swarm.best_point, inertia, cognitive, social, lower, upper);
}

// Evaluates the particles, keeps their best positions and finds the best point of the swarm
static void track_best(Swarm &swarm, int function)
{
	// The values of the particles, then their best positions (with the best values from before) and the best values
	skepu::Map<0>(test_function)(swarm.value, swarm.positions, function);
	skepu::Map<2>([](skepu::Index2D index, float old_best, float x, skepu::Vec<float> value, skepu::Vec<float> best_value) -> float
	{
		return (value(index.row) < best_value(index.row)) ? x : old_best;
	})(swarm.best_positions, swarm.best_positions, swarm.positions, swarm.value, swarm.best_value);
	skepu::Map<2>(skepu::util::min<float>)(swarm.best_value, swarm.best_value, swarm.value);

	// The best point of the swarm: the lowest value (the start value of a reduction is 0 unless it is set), the first particle that
	// has it, and the position of that particle
	swarm.lowest_value = skepu::Reduce(skepu::util::min<float>).setStartValue(FLT_MAX)(swarm.best_value);
	const int best_particle = skepu::MapReduce<1>([](skepu::Index1D index, float best_value, float lowest) -> int
	{
		return (best_value == lowest) ? (int)index.i : INT_MAX;
	}, skepu::util::min<int>).setStartValue(INT_MAX)(swarm.best_value, swarm.lowest_value);
	skepu::Map<0>([](skepu::Index1D index, skepu::Mat<float> best_positions, int particle) -> float
	{
		return best_positions(particle, index.i);
	})(swarm.best_point, swarm.best_positions, best_particle);
}


int main(int argc, char *argv[])
{
	skepu::args::Parser args(argc, argv, "Particle swarm optimization of a test function: sphere, rastrigin, ackley or rosenbrock");
	const size_t particles   = args.get<size_t>("particles");
	const size_t dimensions  = args.get<size_t>("dimensions");
	const size_t iterations  = args.get<size_t>("iterations");
	const std::string name   = args.named<std::string>("function", "rastrigin", "the test function to minimize");
	const unsigned long seed = args.named<unsigned long>("seed", 1, "the seed of the random numbers");
	const float tolerance    = args.named<float>("tolerance", 1e-6f, "stop when the best value is below this");
	args.backend();

	const std::string names[] = {"sphere", "rastrigin", "ackley", "rosenbrock"};
	const float bounds[] = {5.12f, 5.12f, 32.768f, 5.f}; // the search space is [-bound, bound] in every dimension
	int function = -1;
	for (int i = 0; i < 4; ++i) if (name == names[i]) function = i;
	if (function < 0 || particles < 2 || dimensions < 1)
	{
		skepu::io::cout << "The function must be sphere, rastrigin, ackley or rosenbrock, with at least 2 particles and 1 dimension\n";
		return 1;
	}
	const float upper = bounds[function], lower = -upper;
	const float inertia = 0.7298f, cognitive = 1.49618f, social = 1.49618f; // (the constriction coefficients of Clerc and Kennedy)

	skepu::PRNG prng(seed);
	Swarm swarm(particles, dimensions);
	{
		skepu::randomize::Scope stream(prng);
		skepu::randomize::uniform(swarm.positions, lower, upper);
		skepu::randomize::uniform(swarm.velocities, -0.1f * (upper - lower), 0.1f * (upper - lower));
	}

	skepu::Map<1>(skepu::util::identity<float>)(swarm.best_positions, swarm.positions);
	size_t iteration = 0;
	auto duration = skepu::benchmark::measureExecTime([&]
	{
		track_best(swarm, function);
		for (; iteration < iterations && swarm.lowest_value > tolerance; ++iteration)
		{
			SKEPU_TRACE_SCOPE("Iteration");
			move_swarm(swarm, prng, inertia, cognitive, social, lower, upper);
			track_best(swarm, function);
			SKEPU_TRACE_SNAPSHOT(swarm.positions); // (if the program is built for it: make TRACE_SNAPSHOTS=1)
			if ((iteration + 1) % std::max<size_t>(1, iterations / 10) == 0)
				skepu::io::cout << "Iteration " << iteration + 1 << ": best value " << swarm.lowest_value << "\n";
		}
	});

	skepu::io::cout << "Function " << names[function] << ", " << particles << " particles, " << dimensions << " dimensions\n";
	skepu::io::cout << "Iterations: " << iteration << ", duration: " << duration.count() / 1E6 << " s\n";
	skepu::io::cout << "Best value: " << swarm.lowest_value << "\n";
	skepu::io::cout << "Best point:";
	for (size_t d = 0; d < std::min<size_t>(dimensions, 5); ++d)
		skepu::io::cout << " " << swarm.best_point[d];
	skepu::io::cout << (dimensions > 5 ? " ...\n" : "\n");
	return 0;
}
