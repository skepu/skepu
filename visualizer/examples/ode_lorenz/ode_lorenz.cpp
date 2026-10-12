// An ensemble of Lorenz systems, integrated with the classical Runge-Kutta method of the fourth order (RK4).
//
//   dx/dt = sigma (y - x)      dy/dt = x (rho - z) - y      dz/dt = x y - beta z
//
// Every member of the ensemble is a separate system, so the members are independent of each other and a Map integrates
// all of them in parallel: the user function takes the state (x, y, z) of one member, makes a number of RK4 steps, and
// returns the new state. The members start from a small normally distributed cloud around one point (randomize::normal).
// The Lorenz system is chaotic, so the cloud is stretched along the attractor and the spread of the members grows
// exponentially until it is the size of the attractor. After each round of steps the program prints the mean and the
// standard deviation of x over the ensemble (two reductions), which shows that growth.
//
// The type of the numbers is set once, by Real; everything else is a template of it.
//
//   Usage: ode_lorenz ensemble_size rounds backend

#include <cmath>
#include <cstdlib>

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/randomize.hpp>
#include <skepu-lib/util.hpp>

using Real = float;

// The parameters of the system (the classical ones) and the step of the integration are user constants: the user
// functions read them directly (cast to their own type), they are not arguments of the skeleton call. The parameter rho is
// an argument instead, so that the ensemble could be run with other values of it.
[[skepu::userconstant]] constexpr Real SIGMA = 10, BETA = 8.0 / 3.0, STEP = 0.01;
constexpr Real RHO = 28;
constexpr int STEPS_PER_ROUND = 100;

// The derivative of each of the variables
template<typename T>
T lorenz_dx(T x, T y)
{
	return (T)SIGMA * (y - x);
}

template<typename T>
T lorenz_dy(T x, T y, T z, T rho)
{
	return x * (rho - z) - y;
}

template<typename T>
T lorenz_dz(T x, T y, T z)
{
	return x * y - (T)BETA * z;
}

// 'steps' steps of RK4 from the state (x, y, z)
template<typename T>
skepu::multiple<T, T, T> lorenz_rk4(T x, T y, T z, T rho, int steps)
{
	const T h = (T)STEP;
	for (int i = 0; i < steps; ++i)
	{
		const T k1x = lorenz_dx<T>(x, y);
		const T k1y = lorenz_dy<T>(x, y, z, rho);
		const T k1z = lorenz_dz<T>(x, y, z);

		const T x2 = x + h / 2 * k1x, y2 = y + h / 2 * k1y, z2 = z + h / 2 * k1z;
		const T k2x = lorenz_dx<T>(x2, y2);
		const T k2y = lorenz_dy<T>(x2, y2, z2, rho);
		const T k2z = lorenz_dz<T>(x2, y2, z2);

		const T x3 = x + h / 2 * k2x, y3 = y + h / 2 * k2y, z3 = z + h / 2 * k2z;
		const T k3x = lorenz_dx<T>(x3, y3);
		const T k3y = lorenz_dy<T>(x3, y3, z3, rho);
		const T k3z = lorenz_dz<T>(x3, y3, z3);

		const T x4 = x + h * k3x, y4 = y + h * k3y, z4 = z + h * k3z;
		const T k4x = lorenz_dx<T>(x4, y4);
		const T k4y = lorenz_dy<T>(x4, y4, z4, rho);
		const T k4z = lorenz_dz<T>(x4, y4, z4);

		x += h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x);
		y += h / 6 * (k1y + 2 * k2y + 2 * k3y + k4y);
		z += h / 6 * (k1z + 2 * k2z + 2 * k3z + k4z);
	}
	return skepu::ret(x, y, z);
}

template<typename T>
T squared_deviation(T value, T mean)
{
	return (value - mean) * (value - mean);
}

// The mean and the standard deviation of the elements
template<typename T>
void statistics(skepu::Vector<T> &values, T &mean, T &deviation)
{
	auto sum = skepu::Reduce(skepu::util::add<T>);
	auto sum_squares = skepu::MapReduce<1>(squared_deviation<T>, skepu::util::add<T>);
	sum_squares.setLabel("sum of squared deviations");
	const T n = (T)values.size();
	mean = sum(values) / n;
	deviation = std::sqrt(sum_squares(values, mean) / n);
}

template<typename T>
void lorenz_ensemble(size_t size, int rounds)
{
	auto integrate = skepu::Map<3>(lorenz_rk4<T>);
	integrate.setLabel("RK4");

	// The members start from a cloud of the standard deviation 0.001 around (1, 1, 1)
	skepu::Vector<T> x(size), y(size), z(size);
	skepu::randomize::normal(x, (T)1, (T)0.001);
	skepu::randomize::normal(y, (T)1, (T)0.001);
	skepu::randomize::normal(z, (T)1, (T)0.001);

	skepu::io::cout << "time  mean(x)  stddev(x)\n";
	for (int round = 0; round <= rounds; ++round)
	{
		SKEPU_TRACE_SCOPE("Round");
		T mean, deviation;
		statistics<T>(x, mean, deviation);
		skepu::io::cout << round * STEPS_PER_ROUND * STEP << "  " << mean << "  " << deviation << "\n";
		SKEPU_TRACE_SNAPSHOT(x);

		if (round < rounds)
			integrate(x, y, z, x, y, z, (T)RHO, STEPS_PER_ROUND);
	}
}

int main(int argc, char *argv[])
{
	if (argc < 4)
	{
		skepu::io::cout << "Usage: " << argv[0] << " ensemble_size rounds backend\n";
		exit(1);
	}

	const size_t size = atoi(argv[1]);
	const int rounds = atoi(argv[2]);
	auto spec = skepu::BackendSpec{argv[3]};
	skepu::setGlobalBackendSpec(spec);

	lorenz_ensemble<Real>(size, rounds);
	return 0;
}
