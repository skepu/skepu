// The Ozaki scheme: a matrix product of more than double precision made of exact products of small integers.
//
// C = A B is computed from slices of the matrices. A is cut into S slices of 7 bits each, A = diag(ra) sum_s Ahat_s 2^(-7 s), where every
// element of a slice is a signed 8-bit integer and ra is a power of two for each row (so that the largest element of the row is at most 1/2 of
// it). B is cut the same way by columns, with a power of two rb for each column. Then
//
//   C = diag(ra) ( sum_s sum_t (Ahat_s Bhat_t) 2^(-7 (s+t)) ) diag(rb)
//
// and every product Ahat_s Bhat_t is a product of integer matrices: the sum of n products of two numbers of 7 bits is an integer of
// 14 + log2(n) bits, so it is exact in 32 bits, the whole way. This is what an accelerator that multiplies 8-bit integers is good at (a tensor
// core), and what the scheme uses it for. The exact products are then added, with their weights, in a high precision: here a double-double
// number (the sum of two doubles that is exact up to 106 bits) for each element, so that the result has more digits than a double has.
// The precision is chosen by the number of slices, 7 bits for each.
//
// The pieces, with their skeletons:
//   * the power of two of each row of A and each column of B     Map with a MatRow or a MatCol
//   * the matrix of the products of these (ra_i rb_j)            MapPairs: the outer product of two vectors
//   * the slices (an 8-bit integer and the remainder)            Map, two results
//   * the exact products of the slices                           Map with a MatRow and a MatCol, the type of the slices (signed char) and of the products (int)
//   * the sum of the products, in double-double                  Map
//   * the error, in the program's own check                      MapReduce
//
// The products of the slices are the only part that is run on the backend that the program is given; the rest, which is in double precision,
// is run on the CPU backend. (The program can be built for OpenCL and run on a device that has no double precision, as the GPUs of Apple: the
// skeletons with doubles in their signatures are not built for the device and use another backend, see "Double precision on OpenCL devices" in the user guide.)
//
// The check: the elements of A and B are random integers of 40 bits, so the exact product is an integer of at most 88 bits that the host has in an
// __int128. After the products of each anti-diagonal of slice pairs (s + t = w) the program prints the largest error of the elements, relative to
// the largest element: it goes down by a factor of 2^7 for each anti-diagonal, and is zero (written as the smallest number that it prints, 1e-34) once
// the slices have all of the bits of the matrices: 6 slices have the 40 bits, and the anti-diagonal 12 has the products of the last of them.
//
//   Usage: ozaki_gemm n slices backend

#include <cmath>
#include <cstdlib>

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/randomize.hpp>
#include <skepu-lib/util.hpp>

constexpr int SLICE_BITS = 7;

// The smallest power of two that is at least twice the number m (1 if it is zero): the elements are at most 1/2 of it, in absolute value
double power_of_two_above(double m)
{
	double scale = 1;
	if (m == 0)
		return scale;
	while (scale < 2 * m)
		scale *= 2;
	while (scale >= 4 * m)
		scale /= 2;
	return scale;
}

double absolute(double x)
{
	return x < 0 ? -x : x;
}

// The scale of a row of A, and of a column of B
double row_scale(const skepu::MatRow<double> row)
{
	double largest = 0;
	for (size_t j = 0; j < row.cols; ++j)
		largest = absolute(row(j)) > largest ? absolute(row(j)) : largest;
	return power_of_two_above(largest);
}

double column_scale(const skepu::MatCol<double> column)
{
	double largest = 0;
	for (size_t i = 0; i < column.rows; ++i)
		largest = absolute(column(i)) > largest ? absolute(column(i)) : largest;
	return power_of_two_above(largest);
}

// The element divided by the scale of its row, or of its column (exact: the scales are powers of two)
double divide_by_row_scale(skepu::Index2D index, double x, const skepu::Vec<double> scale)
{
	return x / scale(index.row);
}

double divide_by_column_scale(skepu::Index2D index, double x, const skepu::Vec<double> scale)
{
	return x / scale(index.col);
}

// The scale of an element of the product: the outer product of the scales of the rows of A and of the columns of B
double outer_product(double row, double column)
{
	return row * column;
}

// A slice of a number u (|u| <= 1/2): the integer that is nearest to 2^7 u, and the remainder that is left for the next slices,
// which is again at most 1/2 in absolute value (the multiplication by 2^7 and the subtraction are exact)
skepu::multiple<signed char, double> slice(double u)
{
	const double v = u * 128;
	const double nearest = v < 0 ? -floor(-v + 0.5) : floor(v + 0.5);
	return skepu::ret((signed char)nearest, v - nearest);
}

// The product of a row of a slice of A and a column of a slice of B: integers, so it is exact
int slice_product(const skepu::MatRow<signed char> row, const skepu::MatCol<signed char> column)
{
	int sum = 0;
	for (size_t k = 0; k < row.cols; ++k)
		sum += (int)row(k) * (int)column(k);
	return sum;
}

// Adds a product, with its weight and the scale of the element, to a double-double number (hi, lo): the exact sum of two doubles
// (TwoSum) and a renormalization of the number. The term is exact: an integer of 31 bits times powers of two.
skepu::multiple<double, double> accumulate(double hi, double lo, int product, double scale, double weight)
{
	const double term = (double)product * weight * scale;
	const double sum = hi + term;
	const double virtual_term = sum - hi;
	const double error = (hi - (sum - virtual_term)) + (term - virtual_term);
	const double lo_sum = lo + error;
	const double new_hi = sum + lo_sum;
	return skepu::ret(new_hi, lo_sum - (new_hi - sum));
}

// The conversion of an integer to a double
double to_double(long long x)
{
	return (double)x;
}

// The error of an element of the result (hi, lo) against the exact one (exact_hi, exact_lo), relative to the largest element; at least 1e-34
double relative_error(double hi, double lo, double exact_hi, double exact_lo, double norm)
{
	const double difference = (hi - exact_hi) + (lo - exact_lo);
	const double error = absolute(difference) / norm;
	return error > 1e-34 ? error : 1e-34;
}

// A matrix cut into slices: the slices as signed bytes, from the scaled matrix u (|u| <= 1/2) that the cutting uses up
void cut_into_slices(skepu::Matrix<double> &u, std::vector<skepu::Matrix<signed char>> &slices, const std::string &name)
{
	auto cut = skepu::Map<1>(slice);
	for (size_t s = 0; s < slices.size(); ++s)
	{
		SKEPU_TRACE_SCOPE("Slice");
		slices[s] = skepu::Matrix<signed char>(u.size_i(), u.size_j(), name + " slice " + std::to_string(s + 1));
		cut(slices[s], u, u);
	}
}

int main(int argc, char *argv[])
{
	if (argc < 4)
	{
		skepu::io::cout << "Usage: " << argv[0] << " n slices backend\n";
		exit(1);
	}

	const size_t n = atoi(argv[1]);
	const size_t slices = atoi(argv[2]);
	auto products_backend = skepu::BackendSpec{argv[3]};

	// The doubles are for the CPU backend: an OpenCL device does not have to have them
	auto cpu = skepu::BackendSpec{"cpu"};
	skepu::setGlobalBackendSpec(cpu);

	// The matrices: random integers of 40 bits, as doubles (and the integers, for the exact product)
	skepu::Matrix<long long> integers_a(n, n), integers_b(n, n);
	skepu::randomize::uniform(integers_a, -(1LL << 39), (1LL << 39) - 1);
	skepu::randomize::uniform(integers_b, -(1LL << 39), (1LL << 39) - 1);
	skepu::Matrix<double> a(n, n), b(n, n);
	auto convert = skepu::Map<1>(to_double);
	convert(a, integers_a);
	convert(b, integers_b);

	// The exact product, as a double-double number (the host has it in 128 bits)
	skepu::Matrix<double> exact_hi(n, n), exact_lo(n, n);
	skepu::external(skepu::read(integers_a, integers_b), [&]
	{
		for (size_t i = 0; i < n; ++i)
			for (size_t j = 0; j < n; ++j)
			{
				__int128 sum = 0;
				for (size_t k = 0; k < n; ++k)
					sum += (__int128)integers_a(i, k) * (__int128)integers_b(k, j);
				const double high = (double)sum;
				exact_hi(i, j) = high;
				exact_lo(i, j) = (double)(sum - (__int128)high);
			}
	}, skepu::write(exact_hi, exact_lo));
	const double norm = skepu::MapReduce<1>(absolute, skepu::util::max<double>)(exact_hi);

	// The scales, and the matrix of the scales of the elements of C
	skepu::Vector<double> row_scales(n), column_scales(n);
	skepu::Map<0>(row_scale)(row_scales, a);
	skepu::Map<0>(column_scale)(column_scales, b);
	skepu::Matrix<double> scales(n, n);
	skepu::MapPairs<1, 1>(outer_product)(scales, row_scales, column_scales);

	// The slices of A and of B
	skepu::Map<1>(divide_by_row_scale)(a, a, row_scales);
	skepu::Map<1>(divide_by_column_scale)(b, b, column_scales);
	std::vector<skepu::Matrix<signed char>> a_slices(slices), b_slices(slices);
	cut_into_slices(a, a_slices, "A");
	cut_into_slices(b, b_slices, "B");

	// The products of the slices (on the backend of the program), added to the result: anti-diagonal by anti-diagonal, the more significant first
	auto product = skepu::Map<0>(slice_product);
	product.setBackend(products_backend);
	auto add_product = skepu::Map<4>(accumulate);
	auto largest_error = skepu::MapReduce<4>(relative_error, skepu::util::max<double>);
	skepu::Matrix<int> slice_products(n, n);
	skepu::Matrix<double> hi(n, n, 0.0), lo(n, n, 0.0);

	skepu::io::cout << "anti-diagonal  products  largest relative error\n";
	size_t products = 0;
	for (size_t w = 2; w <= 2 * slices; ++w)
	{
		SKEPU_TRACE_SCOPE("Anti-diagonal");
		for (size_t s = 1; s <= slices; ++s)
		{
			const size_t t = w - s;
			if (t < 1 || t > slices) continue;
			product(slice_products, a_slices[s - 1], b_slices[t - 1]);
			add_product(hi, lo, hi, lo, slice_products, scales, std::ldexp(1.0, -SLICE_BITS * (int)(s + t)));
			++products;
		}
		const double error = largest_error(hi, lo, exact_hi, exact_lo, norm);
		skepu::io::cout << w << "  " << products << "  " << error << "\n";
	}
	return 0;
}
