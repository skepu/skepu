/*! \file reduce.h
 *  \brief Contains a class declaration for the Reduce skeleton.
 */

#ifndef REDUCE_H
#define REDUCE_H

#include "reduce_helpers.h"

namespace skepu
{
	namespace backend
	{

		/*!
		 *  \ingroup skeletons
		 */
		/*!
		 *
		 * \brief A specilalization of above class, used for 1D Reduce operation.
		 * Please note that the class name is same. The only difference is
		 * how you instantiate it either by passing 1 user function (i.e. 1D reduction)
		 * or 2 user function (i.e. 2D reduction). See code examples for more information.
		 */
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		class Reduce1D : public SkeletonBase
		{

		public:
			using T = typename ReduceFunc::Ret;

			static constexpr auto skeletonType = SkeletonType::Reduce1D;
			using ResultArg = std::tuple<>;
			using ElwiseArgs = std::tuple<T>;
			using ContainerArgs = std::tuple<>;
			using UniformArgs = std::tuple<>;
			static constexpr bool prefers_matrix = false;

		public:
			Reduce1D(std::string label, CUDAKernel kernel): SkeletonBase(label), m_cuda_kernel(kernel)
			{
#ifdef SKEPU_OPENCL
				this->template prepareOpenCL<CLKernel>(any_uf_requires_double<ReduceFunc>::value);
#endif
			}

			SKEPU_CHAIN_SKELETON_SETTERS(SkeletonBase)

			auto &setReduceMode(ReduceMode mode)
			{
				this->m_mode = mode;
				return *this;
			}

			auto &setStartValue(T val)
			{
				this->m_start = val;
				return *this;
			}

			// Where the instance and its user functions are defined (for the tracer)
			void traceDefinition(tracing::Definition &definition) const override { definition.template describe<ReduceFunc>(); }


			void traceParameters(tracing::Parameters &parameters) const override
			{
				if (this->m_mode == ReduceMode::ColWise)     // (the reduction of rows is the default)
					parameters.add("mode", to_string(this->m_mode));
				parameters.addNumber("start_value", this->m_start);
			}

			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}

		protected:
			CUDAKernel m_cuda_kernel;

			ReduceMode m_mode = ReduceMode::RowWise;
			T m_start{};


			void CPU(Vector<T> &res, Matrix<T>& arg);

			template<typename Iterator>
			Scalar<T> CPU(size_t size, T &res, Iterator arg);

#ifdef SKEPU_OPENMP

			void OMP(Vector<T> &res, Matrix<T>& arg);

			template<typename Iterator>
			Scalar<T> OMP(size_t size, T &res, Iterator arg);

#endif

#ifdef SKEPU_CUDA

			void reduceSingleThreadOneDim_CU(size_t deviceID, VectorIterator<T> &res, const MatrixIterator<T> &arg, size_t numRows);

			void reduceMultipleOneDim_CU(size_t numDevices, VectorIterator<T> &res, const MatrixIterator<T> &arg, size_t numRows);

			void CU(VectorIterator<T> &res, const MatrixIterator<T>& arg, size_t numRows);

			template<typename Iterator>
			Scalar<T> reduceSingleThread_CU(size_t deviceID, size_t size, T &res, Iterator arg);

			template<typename Iterator>
			Scalar<T> reduceMultiple_CU(size_t numDevices, size_t size, T &res, Iterator arg);

			template<typename Iterator>
			Scalar<T> CU(size_t size, T &res, Iterator arg);
#endif

#ifdef SKEPU_OPENCL

			void reduceSingleThreadOneDim_CL(size_t deviceID, VectorIterator<T> &res, const MatrixIterator<T> &arg, size_t numRows);

			void reduceMultipleOneDim_CL(size_t numDevices, VectorIterator<T> &res, const MatrixIterator<T> &arg, size_t numRows);

			void CL(VectorIterator<T> &res, const MatrixIterator<T>& arg, size_t numRows);

			template<typename Iterator>
			Scalar<T> reduceSingle_CL(size_t deviceID, size_t size, T &res, Iterator arg);

			template<typename Iterator>
			Scalar<T> reduceMultiple_CL(size_t numDevices, size_t size, T &res, Iterator arg);

			template<typename Iterator>
			Scalar<T> CL(size_t size, T &res, Iterator arg);

#endif


#ifdef SKEPU_HYBRID

			void Hybrid(Vector<T> &res, Matrix<T>& arg);

			template<typename Iterator>
			T Hybrid(size_t size, T &res, Iterator arg);

#endif


		public:
			template<template<class> class Container>
			Scalar<T> operator()(Container<T> &arg)
			{
				return this->backendDispatch(arg.size(), arg.begin());
			}

			template<typename Iterator>
			Scalar<T> operator()(Iterator arg, Iterator arg_end)
			{
				return this->backendDispatch(arg_end - arg, arg.begin());
			}

			// The entries of each row of a sparse matrix are reduced, to an element of the output Vector
			Vector<T> &operator()(Vector<T> &res, SparseMatrix<T> &arg)
			{
				if (this->m_mode != ReduceMode::RowWise)
					SKEPU_ERROR("Reduce of a sparse matrix: only the reduction of the rows is supported (use the transpose for columns)");
				if (!arg.has_pattern())
					SKEPU_ERROR("Reduce: the sparse matrix has no pattern");
				const size_t rows = arg.rows();
				if (res.size() != rows)
					SKEPU_ERROR("Reduce: the output Vector has " << res.size() << " elements, and the sparse matrix " << rows << " rows");

				this->selectBackend(arg.nnz());
				arg.updateHost();
				res.invalidateDeviceData();
				SKEPU_TRACE_START_EVENT(trace_handle);
				const SparseIndex *offsets = arg.pattern()->row_offsets().data();
				const T *values = arg.values().getAddress();
				T *result = res.getAddress();

				auto row = [&](size_t r)
				{
					T inner = this->m_start;
					for (SparseIndex k = offsets[r]; k < offsets[r + 1]; ++k)
						inner = ReduceFunc::CPU(inner, values[k]);
					result[r] = inner;
				};
#ifdef SKEPU_OPENMP
				if (this->m_selected_spec->activateBackend() == Backend::Type::OpenMP)
				{
#pragma omp parallel for schedule(runtime)
					for (size_t r = 0; r < rows; ++r)
						row(r);
				}
				else
#endif
				{
					for (size_t r = 0; r < rows; ++r)
						row(r);
				}
				SKEPU_TRACE_CALL(trace_handle, "Reduce 1D-Partial", this, {rows, arg.nnz()}, {res.getObjectID()}, {arg.getObjectID()}, {}, {});
				return res;
			}

			Vector<T> &operator()(Vector<T> &res, Matrix<T>& arg)
			{
			//	assert(this->m_execPlan != NULL && this->m_execPlan->isCalibrated());

				const size_t size = arg.size();

				// The result has an element for each row (row-wise) or column (column-wise), as in the sequential skeleton: the backends write
				// all of them
				if (res.size() != (this->m_mode == ReduceMode::ColWise ? arg.total_cols() : arg.total_rows()))
					SKEPU_ERROR("Reduce: Non-matching container sizes");

				this->selectBackend(size);

				VectorIterator<T> it = res.begin();
				Matrix<T> &arg_tr = (this->m_mode == ReduceMode::ColWise) ? arg.transpose(*this->m_selected_spec) : arg;

				switch (this->m_selected_spec->activateBackend())
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
					this->Hybrid(res, arg_tr);
					break;
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					this->CU(it, arg_tr.begin(), arg_tr.total_rows());
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					this->CL(it, arg_tr.begin(), arg_tr.total_rows());
					break;
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					this->OMP(res, arg_tr);
					break;
#endif
				default:
					this->CPU(res, arg_tr);
				}

				return res;
			}

		private:
			template<typename Iterator>
			Scalar<T> backendDispatch(size_t size, Iterator arg)
			{
			//	assert(this->m_execPlan != NULL && this->m_execPlan->isCalibrated());

				T res = this->m_start;

				this->selectBackend(size);

				switch (this->m_selected_spec->activateBackend())
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
					return this->Hybrid(size, res, arg);
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					return this->CU(size, res, arg);
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					return this->CL(size, res, arg);
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					return this->OMP(size, res, arg);
#endif
				default:
					return this->CPU(size, res, arg);
				}
			}


		};


		/*!
		 *  \class Reduce
		 *
		 *  \brief A class representing the Reduce skeleton both for 1D and 2D reduce operation for 1D Vector, 2D Dense Matrix/Sparse matrices.
		 *
		 *  This class defines the Reduce skeleton which support following reduction operations:
		 *  (a) (1D Reduction) Each element in the input range, yielding a scalar result by applying a commutative associative binary operator.
		 *     Here we consider dense/sparse matrix as vector thus reducing all (non-zero) elements of the matrix.
		 *  (b) (1D Reduction) Dense/Sparse matrix types: Where we reduce either row-wise or column-wise by applying a commutative associative binary operator. It returns
		 *     a \em SkePU vector of results that corresponds to reduction on either dimension.
		 *  (c) (2D Reduction) Dense/Sparse matrix types: Where we reduce both row- and column-wise by applying two commutative associative
		 *     binary operators, one for row-wise reduction and one for column-wise reduction. It returns a scalar result.
		 *  Two versions of this class are created using C++ partial class-template specialization to support
		 *  (a) 1D reduction (where a "single" reduction operator is applied on all elements or to 1 direction for 2D Dense/Sparse matrix).
		 *  (b) 2D reduction that works only for matrix (where two different reduction operations are used to reduce row-wise and column-wise separately.)
		 *  Once instantiated, it is meant to be used as a function and therefore overloading
		 *  \p operator(). The Reduce skeleton needs to be created with
		 *  a 1 or 2 binary user function for 1D reduction and 2D reduction respectively.
		 */
		template<typename ReduceFuncRowWise, typename ReduceFuncColWise, typename CUDARowWise, typename CUDAColWise, typename CLKernel>
		class Reduce2D : public Reduce1D<ReduceFuncRowWise, CUDARowWise, CLKernel>
		{
			using T = typename ReduceFuncRowWise::Ret;

		public:
			using SkepuChainBase = Reduce1D<ReduceFuncRowWise, CUDARowWise, CLKernel>;
			SKEPU_CHAIN_SKELETON_SETTERS(SkepuChainBase)
			SKEPU_CHAIN(setReduceMode, SkepuChainBase)
			SKEPU_CHAIN(setStartValue, SkepuChainBase)

			static constexpr auto skeletonType = SkeletonType::Reduce2D;
			static constexpr bool prefers_matrix = true;

			// Where the instance and its user functions are defined (for the tracer)
			void traceDefinition(tracing::Definition &definition) const override { definition.template describe<ReduceFuncRowWise, ReduceFuncColWise>(); }


			Reduce2D(std::string label, CUDARowWise row, CUDAColWise col) : Reduce1D<ReduceFuncRowWise, CUDARowWise, CLKernel>(label, row), m_cuda_colwise_kernel(col) {}

		private:
			CUDAColWise m_cuda_colwise_kernel;




		private:
			T CPU(T &res, Matrix<T>& arg);

#ifdef SKEPU_OPENMP

			T OMP(T &res, Matrix<T>& arg);

#endif

#ifdef SKEPU_CUDA

			T CU(T &res, const MatrixIterator<T>& arg, size_t numRows);

			T reduceSingleThread_CU(size_t deviceID, T &res, const MatrixIterator<T>& arg, size_t numRows);

			T reduceMultiple_CU(size_t numDevices, T &res, const MatrixIterator<T>& arg, size_t numRows);

#endif

#ifdef SKEPU_OPENCL

			T CL(T &res, const MatrixIterator<T>& arg, size_t numRows);

			T reduceSingle_CL(size_t deviceID, T &res, const MatrixIterator<T>& arg, size_t numRows);

			T reduceNumDevices_CL(size_t numDevices, T &res, const MatrixIterator<T>& arg, size_t numRows);

#endif

#ifdef SKEPU_HYBRID

			T Hybrid(T &res, Matrix<T>& arg);

#endif



		public:
			T operator()(Vector<T>& arg)
			{
				return Reduce1D<ReduceFuncRowWise, CUDARowWise, CLKernel>::operator()(arg);
			}

			T operator()(Matrix<T>& arg)
			{
			//	assert(this->m_execPlan != NULL && this->m_execPlan->isCalibrated());

				this->selectBackend(arg.size());

				T res = this->m_start;

				Matrix<T> &arg_tr = (this->m_mode == ReduceMode::ColWise) ? arg.transpose(*this->m_selected_spec) : arg;

				switch (this->m_selected_spec->activateBackend())
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
					return this->Hybrid(res, arg_tr);
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					return this->CU(res, arg_tr.begin(), arg_tr.total_rows());
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					return this->CL(res, arg_tr.begin(), arg_tr.total_rows());
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					return this->OMP(res, arg_tr);
#endif
				default:
					return this->CPU(res, arg_tr);
				}
			}

		};

	} // end namespace backend

} // end namespace skepu


#include "impl/reduce/reduce_cpu.inl"
#include "impl/reduce/reduce_omp.inl"
#include "impl/reduce/reduce_cl.inl"
#include "impl/reduce/reduce_cu.inl"
#include "impl/reduce/reduce_hy.inl"

#endif // REDUCE_H
