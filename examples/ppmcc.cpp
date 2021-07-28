/*!
 *  PPMCC stands for "Pearson product-moment correlation coefficient"
 *  In statistics, the Pearson coefficient of correlation is a measure by the
 *  linear dependence between two variables X and Y. The mathematical
 *  expression of the Pearson coefficient of correlation is as follows:
 *   r = ( (n*sum(X.Y)-sum(X)*sum(Y))/((n*sum(X^2)-(sum(X))^2)*(n*sum(Y^2)-(sum(Y))^2)) )
 */

#include <skepu>
#include <skepu-lib/util.hpp>
#include <skepu-lib/io.hpp>

using T = float;

T ppmcc(skepu::Vector<T> &x, skepu::Vector<T> &y)
{
	// Skeleton definitions
	auto sum = skepu::Reduce(skepu::util::add<T>);
	auto dotProduct = skepu::MapReduce(skepu::util::mul<T>, skepu::util::add<T>);
	auto sumSquare = skepu::MapReduce(skepu::util::square<T>, skepu::util::add<T>);
	
	size_t N = x.size();
	T sumX = sum(x);
	T sumY = sum(y);
	
	return (N * dotProduct(x, y) - sumX * sumY)
		/ sqrt((N * sumSquare(x) - pow(sumX, 2)) * (N * sumSquare(y) - pow(sumY, 2)));
}

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		skepu::io::cout << "Usage: " << argv[0] << " input_size backend\n";
		exit(1);
	}
	
	const size_t size = std::stoul(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	skepu::setGlobalBackendSpec(spec);
	
	// Vector operands
	skepu::Vector<T> x(size), y(size);
	x.randomize(1, 3);
	y.randomize(2, 4);
	
	skepu::io::cout << "X: " << x << "\nY: " << y << "\n";

	T res = ppmcc(x, y);
	
	skepu::io::cout << "res: " << res << "\n";
	
	return 0;
}
