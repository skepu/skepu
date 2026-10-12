#pragma once

// The snapshots of the trace (SKEPU_TRACE_SNAPSHOT) of distributed containers: the container is gathered (every process takes part)
// and the first process writes the file, in the format of the node-level library. Found by the tracer through the namespace of
// the containers.

#include <fstream>
#include <string>

#include "skepu3/mpi/vector.hpp"
#include "skepu3/mpi/matrix.hpp"
#include "skepu3/mpi/tensor.hpp"

namespace skepu
{
	template<typename T>
	inline void snapshot_serializer(Vector<T> &data, std::string &fname)
	{
		data.flush(); // (the gather)
		if (mpi::rank() != 0)
			return;
		const Vector<T> &view = data;
		std::ofstream fs(fname);
		fs << 1 << " " << data.size() << "\n";
		for (size_t i = 0; i < data.size(); ++i)
			fs << view(i) << " ";
	}

	template<typename T>
	inline void snapshot_serializer(Matrix<T> &data, std::string &fname)
	{
		data.flush(); // (the gather)
		if (mpi::rank() != 0)
			return;
		const Matrix<T> &view = data;
		std::ofstream fs(fname);
		fs << 2 << " " << data.size_i() << " " << data.size_j() << "\n";
		for (size_t i = 0; i < data.size_i(); ++i)
			for (size_t j = 0; j < data.size_j(); ++j)
				fs << view(i, j) << " ";
	}

	template<typename T>
	inline void snapshot_serializer(Tensor3<T> &data, std::string &fname)
	{
		data.flush(); // (the gather)
		if (mpi::rank() != 0)
			return;
		const Tensor3<T> &view = data;
		std::ofstream fs(fname);
		fs << 3 << " " << data.size_i() << " " << data.size_j() << " " << data.size_k() << "\n";
		for (size_t i = 0; i < data.size_i(); ++i)
			for (size_t j = 0; j < data.size_j(); ++j)
				for (size_t k = 0; k < data.size_k(); ++k)
					fs << view(i, j, k) << " ";
	}

	template<typename T>
	inline void snapshot_serializer(Tensor4<T> &data, std::string &fname)
	{
		data.flush(); // (the gather)
		if (mpi::rank() != 0)
			return;
		const Tensor4<T> &view = data;
		std::ofstream fs(fname);
		fs << 4 << " " << data.size_i() << " " << data.size_j() << " " << data.size_k() << " " << data.size_l() << "\n";
		for (size_t i = 0; i < data.size_i(); ++i)
			for (size_t j = 0; j < data.size_j(); ++j)
				for (size_t k = 0; k < data.size_k(); ++k)
					for (size_t l = 0; l < data.size_l(); ++l)
						fs << view(i, j, k, l) << " ";
	}
}
