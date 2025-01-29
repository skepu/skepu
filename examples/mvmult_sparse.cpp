
#include <skepu>
#include <skepu-lib/io.hpp>




/*!
 * A helper function to calculate SparseMatrix-Vector product. Used to verify that the SkePU output is correct.
 */
template<typename T>
void directspmv(skepu::Vector<T> &res, skepu::SparseMatrix<T> &m, skepu::Vector<T> &v)
{
	int rows = m.total_rows();
	int nnz = m.total_nnz();

	T *values= m.get_values();
	size_t * row_offsets = m.get_row_pointers();
	size_t * col_indices = m.get_col_indices();

	T sum;

	int rowIdx = 0;
	int nxtRowIdx = 0;

	for(int ii = 0; ii < rows; ii++)
	{
		sum = 0;

		rowIdx = row_offsets[ii];
		nxtRowIdx = row_offsets[ii+1];

		for (int jj=rowIdx; jj<nxtRowIdx; jj++)
		{
			sum += values[jj] * v[col_indices[jj]];
		}
		res[ii] = sum;
	}
}


template<typename T>
T arr(skepu::Index1D index, skepu::SparseMat<T> sm, skepu::Vec<T> v)
{
	size_t row = index.i;
	float res = 0;
	for (size_t i = sm.row_offsets[row]; i < sm.row_offsets[row + 1]; ++i)
		res += sm.data[i] * v.data[sm.col_indices[i]];
	return res;
}

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
	
	size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	
	auto reverse = skepu::Map<0>(arr<float>);

	// randomly initialize the SparseMatrix
	skepu::SparseMatrix<float> m1(size, size, (size * size) / 2, 3.f, 7.f);

	// result vectors
	skepu::Vector<float> v0(size), r(size), r2(size);

	//Sets v0 = 1 2 3 4 5...
	for(int i = 0; i < size; ++i)
	{
		v0[i] = (float)(i+10);
	}

	skepu::io::cout << "v0: " << v0 << "\n";
	m1.printMatrixInDenseFormat();

	reverse(r, m1, v0);
	skepu::io::cout << "Computed output: " << r <<"\n";

	directspmv<float>(r2, m1, v0);
	skepu::io::cout << "Direct output:   " << r2 <<"\n";

	return 0;
}
