// A reaction-diffusion equation, solved with the method of lines and an adaptive Runge-Kutta method.
//
//   du/dt = D (d2u/dx2 + d2u/dy2) + r u (1 - u)        (the Fisher-KPP equation)
//
// on a square grid of N x N points with the spacing 1 and zero flux over the boundary. A patch of u = 1 in a corner spreads
// as a wave front, with the speed 2 sqrt(D r). The space is discretized with the 5-point stencil, which turns the equation
// into one ordinary differential equation for each grid point, du_ij/dt = F(u)_ij, where F is a MapOverlap (the stencil
// and the reaction term).
//
// The ODE system is integrated with the embedded Runge-Kutta pair of Bogacki and Shampine (order 3 with an error estimate
// of order 2), with a step size that adapts to the tolerance. A step makes the stages
//
//   k1 = F(u)    k2 = F(u + h/2 k1)    k3 = F(u + 3h/4 k2)    u' = u + h (2/9 k1 + 1/3 k2 + 4/9 k3)    k4 = F(u')
//
// (the last stage of an accepted step is the first stage of the next one), so a step is four MapOverlap calls, the Map calls
// that combine the stages, and a MapReduce that finds the largest estimated error. If the error is above the tolerance the
// step is rejected and repeated with a smaller step; otherwise it is accepted and the next step is made larger or smaller
// according to the error. The step is a scalar of the host, so the size of the steps (a reduction result that decides what
// the next skeleton calls do) is visible in a trace.
//
// The type of the numbers is set once, by Real; everything else is a template of it. The parameters of the equation are user
// constants.
//
//   Usage: ode_fisher size end_time backend [tolerance]

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/util.hpp>

using Real = float;

[[skepu::userconstant]] constexpr Real DIFFUSION = 1, GROWTH = 1;

// The right-hand side F: the diffusion of the 5-point stencil and the reaction. The edges are the values of the nearest points,
// so there is no flux over the boundary.
template<typename T>
T rhs(skepu::Region2D<T> u)
{
	const T laplacian = u(-1, 0) + u(1, 0) + u(0, -1) + u(0, 1) - 4 * u(0, 0);
	return (T)DIFFUSION * laplacian + (T)GROWTH * u(0, 0) * (1 - u(0, 0));
}

// The state of a stage: u + c k
template<typename T>
T stage_state(T u, T k, T c)
{
	return u + c * k;
}

// The new state of the step: u + h (2/9 k1 + 1/3 k2 + 4/9 k3)
template<typename T>
T new_state(T u, T k1, T k2, T k3, T h)
{
	return u + h * (2 * k1 / 9 + k2 / 3 + 4 * k3 / 9);
}

// The estimated error of a point: h (-5/72 k1 + 1/12 k2 + 1/9 k3 - 1/8 k4), in absolute value
template<typename T>
T estimated_error(T k1, T k2, T k3, T k4, T h)
{
	const T error = h * (-5 * k1 / 72 + k2 / 12 + k3 / 9 - k4 / 8);
	return error < 0 ? -error : error;
}

// The initial state: a patch of 1 in the corner, 0 elsewhere
template<typename T>
T initial_state(skepu::Index2D index, int patch)
{
	return (int)index.row < patch && (int)index.col < patch ? (T)1 : (T)0;
}

template<typename T>
void fisher(size_t size, T end_time, T tolerance)
{
	auto derivative = skepu::MapOverlap(rhs<T>);
	derivative.setOverlap(1, 1);
	derivative.setEdgeMode(skepu::Edge::Duplicate);

	auto stage = skepu::Map<2>(stage_state<T>);
	auto combine = skepu::Map<4>(new_state<T>);
	auto largest_error = skepu::MapReduce<4>(estimated_error<T>, skepu::util::max<T>);
	auto total = skepu::Reduce(skepu::util::add<T>);
	auto initialize = skepu::Map<0>(initial_state<T>);

	// The state, the state of the stages and the new state, and the stages (a pointer to each, which are exchanged when a step is accepted)
	skepu::Matrix<T> u(size, size), u_new(size, size), u_stage(size, size);
	skepu::Matrix<T> stage_a(size, size), stage_b(size, size), stage_c(size, size), stage_d(size, size);
	skepu::Matrix<T> *k1 = &stage_a, *k2 = &stage_b, *k3 = &stage_c, *k4 = &stage_d;

	initialize(u, 8);
	SKEPU_TRACE_SNAPSHOT(u);
	derivative(*k1, u);

	T time = 0, h = (T)0.01;
	int accepted = 0, rejected = 0, next_report = 0;
	skepu::io::cout << "time  step  accepted  rejected  total of u\n";
	while (time < end_time)
	{
		SKEPU_TRACE_SCOPE("Step");
		if (accepted >= next_report)
		{
			skepu::io::cout << time << "  " << h << "  " << accepted << "  " << rejected << "  " << (T)total(u) << "\n";
			next_report += 20;
		}
		h = std::min(h, end_time - time);

		stage(u_stage, u, *k1, h / 2);
		derivative(*k2, u_stage);
		stage(u_stage, u, *k2, 3 * h / 4);
		derivative(*k3, u_stage);
		combine(u_new, u, *k1, *k2, *k3, h);
		derivative(*k4, u_new);

		const T error = largest_error(*k1, *k2, *k3, *k4, h);
		if (error <= tolerance)
		{
			time += h;
			++accepted;
			std::swap(u, u_new);
			std::swap(k1, k4);
			SKEPU_TRACE_SNAPSHOT(u);
		}
		else
			++rejected;

		// The next step: the error of a step of the order 3 is proportional to h^3
		const T factor = error > 0 ? (T)0.9 * std::cbrt(tolerance / error) : (T)5;
		h *= std::min((T)5, std::max((T)0.2, factor));
	}
	skepu::io::cout << time << "  " << h << "  " << accepted << "  " << rejected << "  " << (T)total(u) << "\n";
}

int main(int argc, char *argv[])
{
	if (argc < 4)
	{
		skepu::io::cout << "Usage: " << argv[0] << " size end_time backend [tolerance]\n";
		exit(1);
	}

	const size_t size = atoi(argv[1]);
	const Real end_time = atof(argv[2]);
	auto spec = skepu::BackendSpec{argv[3]};
	skepu::setGlobalBackendSpec(spec);
	const Real tolerance = argc > 4 ? atof(argv[4]) : 1e-4;

	fisher<Real>(size, end_time, tolerance);
	return 0;
}
