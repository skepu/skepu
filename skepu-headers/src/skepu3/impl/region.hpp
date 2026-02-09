#pragma once

namespace future_std
{
	template<class T>
	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	constexpr const T& clamp( const T& v, const T& lo, const T& hi )
	{
	  return (v < lo) ? lo : (hi < v) ? hi : v;
	}
}

namespace skepu
{
	// ----------------------------------------------------------------
	// Region proxy types
	// ----------------------------------------------------------------
	
	template<typename T>
	struct Region1D
	{
		int oi;
		size_t stride;
		const T *data;
		
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
		T operator()(int i) const
		{
			return data[i * this->stride];
		}
		
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
		Region1D(int arg_oi, size_t arg_stride, const T *arg_data)
		: oi(arg_oi), stride(arg_stride), data(arg_data) {}
	};
	
	template<typename T>
	struct Region2D
	{
		int oi, oj;
		size_t size_i, size_j;
		size_t stride;
		Index2D idx;
		const T *data;
		Edge edge = Edge::None;
		T pad;
		
#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		T operator()(int i, int j) const
		{
			int ii = this->idx.row + i;
			int jj = this->idx.col + j;
			size_t offset;
			
			switch (this->edge)
			{
			case Edge::Cyclic:
				offset =
					((ii + this->size_i) % this->size_i) * this->stride +
					((jj + this->size_j) % this->size_j);
				return data[offset];
			case Edge::Pad:
				if ((0 <= ii && ii < size_i) && (0 <= jj && jj < size_j))
					return data[ii * this->stride + jj];
				else return this->pad;
			case Edge::Duplicate:
				offset =
					future_std::clamp<int>(ii, 0, this->size_i-1) * this->stride +
					future_std::clamp<int>(jj, 0, this->size_j-1);
				return data[offset];
			default:
				return data[ii * this->stride + jj];
			}
		}
		
		Region2D(Matrix<T> const& mat, int arg_oi, int arg_oj, Edge arg_edge, T arg_pad)
		:	oi(arg_oi), oj(arg_oj),
			size_i(mat.size_i()), size_j(mat.size_j()),
			stride(this->size_j),
			edge(arg_edge),
			pad(arg_pad),
			data(mat.getAddress())
		{}

// For CUDA kernel
#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		Region2D(int arg_oi, int arg_oj, size_t arg_stride, T *arg_data)
		:	oi(arg_oi), oj(arg_oj),
			size_i(0), size_j(0), // unused
			stride(arg_stride),
			edge(Edge::None),
			data(arg_data),
			idx{0,0}
		{}

		// Called by test code.
		Region2D(int arg_oi, int arg_oj,
				 size_t arg_size_i, size_t arg_size_j,
				 size_t arg_stride, T *arg_data)
		:	oi(arg_oi), oj(arg_oj),
			size_i(arg_size_i), size_j(arg_size_j),
			stride(arg_stride),
			edge(Edge::None),
			data(arg_data),
			idx{0,0}
		{}
	};
	
	template<typename T>
	struct Region3D
	{
		int oi, oj, ok;
		size_t size_i, size_j, size_k;
		size_t m_offset[3];
		int stride1, stride2;
		Index3D idx;
		const T *data;
		Edge edge = Edge::None;
		T pad;
		
#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		T operator()(int i, int j, int k) const
		{
			int ii = m_offset[0] + this->idx.i + i;
			int jj = m_offset[1] + this->idx.j + j;
			int kk = m_offset[2] + this->idx.k + k;
			size_t offset;
			
			switch (this->edge)
			{
			case Edge::Cyclic:
				offset =
					((ii + this->size_i) % this->size_i) * this->stride1 +
					((jj + this->size_j) % this->size_j) * this->stride2 +
					((kk + this->size_k) % this->size_k);
				return data[offset];
			case Edge::Pad:
				if ((0 <= ii && ii < size_i) && (0 <= jj && jj < size_j) &&
						(0 <= kk && kk < size_k))
					return data[ii * this->stride1 + jj * this->stride2 + kk];
				else return this->pad;
			case Edge::Duplicate:
				offset =
					future_std::clamp<int>(ii, 0, this->size_i-1) * this->stride1 +
					future_std::clamp<int>(jj, 0, this->size_j-1) * this->stride2 +
					future_std::clamp<int>(kk, 0, this->size_k-1);
				return data[offset];
			default:
				return data[ii * this->stride1 + jj * this->stride2 + kk];
			}
		}
		
		Region3D(Tensor3<T> const& ten, int arg_oi, int arg_oj, int arg_ok, Edge arg_edge, T arg_pad)
		:	oi(arg_oi), oj(arg_oj), ok(arg_ok),
			size_i(ten.size_i()), size_j(ten.size_j()), size_k(ten.size_k()),
			stride1(ten.m_stride_1),
			stride2(ten.m_stride_2),
			m_offset{0,0,0},
			edge(arg_edge),
			pad(arg_pad),
			data(ten.getAddress())
		{}
			
		// For CUDA kernel
#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		Region3D(int arg_oi, int arg_oj, int arg_ok, size_t arg_stride1, size_t arg_stride2, T *arg_data)
		:	oi(arg_oi), oj(arg_oj), ok(arg_ok),
			size_i(0), size_j(0), size_k(0), // unused
			stride1(arg_stride1), stride2(arg_stride2),
			m_offset{0,0,0},
			edge(Edge::None),
			data(arg_data),
			idx{0,0,0}
		{}

		// Called by test code.
		Region3D(int arg_oi, int arg_oj, int arg_ok,
				 size_t arg_size_i, size_t arg_size_j, size_t arg_size_k,
				 size_t arg_stride1, size_t arg_stride2, T *arg_data)
		:	oi(arg_oi), oj(arg_oj), ok(arg_ok),
			size_i(arg_size_i), size_j(arg_size_j), size_k(arg_size_k),
			stride1(arg_stride1), stride2(arg_stride2),
			m_offset{0,0,0},
			edge(Edge::None),
			data(arg_data),
			idx{0,0,0}
		{}
	};
	
	template<typename T>
	struct Region4D
	{
		int oi, oj, ok, ol;
		size_t size_i, size_j, size_k, size_l;
		size_t offset_i, offset_j, offset_k, offset_l;
		size_t stride1, stride2, stride3;
		Index4D idx;
		const T *data;
		Edge edge = Edge::None;
		T pad;
		
		T operator()(int i, int j, int k, int l) const
		{
			int ii = this->idx.i + i + (this->edge != Edge::None ? -this->offset_i : 0);
			int jj = this->idx.j + j + (this->edge != Edge::None ? -this->offset_j : 0);
			int kk = this->idx.k + k + (this->edge != Edge::None ? -this->offset_k : 0);
			int ll = this->idx.l + l + (this->edge != Edge::None ? -this->offset_l : 0);
			size_t offset;
			
			switch (this->edge)
			{
			case Edge::Cyclic:
				offset =
					((ii + this->size_i) % this->size_i) * this->stride1 +
					((jj + this->size_j) % this->size_j) * this->stride2 +
					((kk + this->size_k) % this->size_k) * this->stride3 +
					((ll + this->size_l) % this->size_l);
				return data[offset];
			case Edge::Pad:
				if ((0 <= ii && ii < size_i) && (0 <= jj && jj < size_j) &&
					  (0 <= kk && kk < size_k) && (0 <= ll && ll < size_l))
					return data[ii * this->stride1 + jj * this->stride2 + kk * this->stride3 + ll];
				else return this->pad;
			case Edge::Duplicate:
				offset =
					future_std::clamp<int>(ii, 0, this->size_i-1) * this->stride1 +
					future_std::clamp<int>(jj, 0, this->size_j-1) * this->stride2 +
					future_std::clamp<int>(kk, 0, this->size_k-1) * this->stride3 +
					future_std::clamp<int>(ll, 0, this->size_l-1);
				return data[offset];
			default:
				return data[(ii + oi) * this->stride1 + (jj + oj) * this->stride2 + (kk + ok) * this->stride3 + (ll + ol)];
			}
		}

		Region4D(Tensor4<T> const& ten, int arg_oi, int arg_oj, int arg_ok, int arg_ol, Edge arg_edge, T arg_pad,
		size_t arg_offset_i = 0, size_t arg_offset_j = 0, size_t arg_offset_k = 0, size_t arg_offset_l = 0)
		:	oi(arg_oi), oj(arg_oj), ok(arg_ok), ol(arg_ol),
			size_i(ten.size_i()), size_j(ten.size_j()), size_k(ten.size_k()), size_l(ten.size_l()),
			stride1(ten.m_stride_1),
			stride2(ten.m_stride_2),
			stride3(ten.m_stride_3),
			edge(arg_edge),
			pad(arg_pad),
			offset_i(arg_offset_i), offset_j(arg_offset_j), offset_k(arg_offset_k), offset_l(arg_offset_l), 
			data(ten.getAddress())
		{}

		// Called by test code.
		Region4D(int arg_oi, int arg_oj, int arg_ok, int arg_ol,
				 size_t arg_size_i, size_t arg_size_j, size_t arg_size_k, size_t arg_size_l,
				 size_t arg_stride1, size_t arg_stride2, size_t arg_stride3, T *arg_data)
		:	oi(arg_oi), oj(arg_oj), ok(arg_ok), ol(arg_ol),
			size_i(arg_size_i), size_j(arg_size_j), size_k(arg_size_k), size_l(arg_size_l),
			stride1(arg_stride1), stride2(arg_stride2), stride3(arg_stride3),
			edge(Edge::None),
			data(arg_data),
			idx{0,0,0,0}
		{}
	};

	template <typename T>
	struct Region4DCU
	{
		int oi, oj, ok, ol;
		size_t stride1, stride2, stride3;
		const T *data;

#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		T operator()(int i, int j, int k, int l) const
		{
			return data[i * this->stride1 + j * this->stride2 + k * this->stride3 + l];
		}

#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		Region4DCU(int arg_oi, int arg_oj, int arg_ok, int arg_ol, size_t arg_stride1, size_t arg_stride2, size_t arg_stride3, T *arg_data)
		:	oi(arg_oi), oj(arg_oj), ok(arg_ok), ol(arg_ol),
			stride1(arg_stride1), stride2(arg_stride2), stride3(arg_stride3),
			data(arg_data)
		{}
	};
	
	
	
	template<typename T>
	std::ostream& operator<<(std::ostream &os, Region1D<T>& r)
	{
		os << "Region1: (" << (2 * r.oi + 1) << ")\n";
		
		for (int i = -r.oi; i <= r.oi; ++i)
		{
			auto val = r(i);
			std::cout << std::setw(5) << val << " ";
		}
		return os << "\n";
	}
	
	template<typename T>
	std::ostream& operator<<(std::ostream &os, Region2D<T>& r)
	{
		os << "Region2: (" << (2 * r.oi + 1)
			<< " x " << (2 * r.oj + 1) << ")\n";
		
		for (int i = -r.oi; i <= r.oi; ++i)
		{
			for (int j = -r.oj; j <= r.oj; ++j)
			{
				auto val = r(i,j);
				std::cout << std::setw(5) << val << " ";
			}
			os << "\n";
		}
		return os << "\n";
	}
	
	template<typename T>
	std::ostream& operator<<(std::ostream &os, Region3D<T>& r)
	{
		os << "Region3: (" << (2 * r.oi + 1)
			<< " x " << (2 * r.oj + 1)
			<< " x " << (2 * r.ok + 1) << ")\n";
		
		for (int j = -r.oj; j <= r.oj; ++j)
		{
			for (int i = -r.oi; i <= r.oi; ++i)
			{
				for (int k = -r.ok; k <= r.ok; ++k)
				{
					auto val = r(i,j,k);
					std::cout << std::setw(5) << val << " ";
				}
				os << " | ";
			}
			os << "\n";
		}
		return os << "\n";
	}
	
	template<typename T>
	std::ostream& operator<<(std::ostream &os, Region4D<T>& r)
	{
		os << "Region4: (" << (2 * r.oi + 1)
			<< " x " << (2 * r.oj + 1)
			<< " x " << (2 * r.ok + 1)
			<< " x " << (2 * r.ol + 1) << ")\n";
		
		os << "strides[" << r.stride1 << ", " << r.stride2 << ", " << r.stride3 << "]\n";
		os << "index[" << r.idx.i << ", " << r.idx.j << ", " << r.idx.k << ", " << r.idx.l << "]\n";
		os << "offset[" << r.offset_i << ", " << r.offset_j << ", " << r.offset_k << ", " << r.offset_l << "]\n";
		
		for (int i = -r.oi; i <= r.oi; ++i)
		{
			for (int j = -r.oj; j <= r.oj; ++j)
			{
				for (int k = -r.ok; k <= r.ok; ++k)
				{
					for (int l = -r.ol; l <= r.ol; ++l)
					{
						auto val = r(i,j,k,l);
						std::cout << std::setw(12) << std::fixed << val << " ";
					}
					os << " | ";
				}
				os << "\n";
			}
			os << "---------------------------------\n";
		}
		return os << "\n";
	}
	
	
	// ----------------------------------------------------------------
	// MapOverlap type deducer
	// ----------------------------------------------------------------
	
	template<typename T>
	struct region_type_h {};
	
	template<typename T>
	struct region_type_h<Region1D<T>> { using type = T; };
	
	template<typename T>
	struct region_type_h<Region2D<T>> { using type = T; };
	
	template<typename T>
	struct region_type_h<Region3D<T>> { using type = T; };
	
	template<typename T>
	struct region_type_h<Region4D<T>> { using type = T; };
	
	template<typename T>
	struct region_type : region_type_h<typename std::decay<T>::type> {};
	
	template<typename... Args>
	struct region_type_ext: region_type<typename pack_element<
			(is_indexed<Args...>::value ? 1 : 0)
			+ (has_random<Args...>::value ? 1 : 0), Args...>::type> {};
	
	
	// ----------------------------------------------------------------
	// MapOverlap dimensionality deducer
	// ----------------------------------------------------------------
	
	template<typename... Args>
	struct mapoverlap_dimensionality {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Region1D<T>, Args...>: std::integral_constant<int, 1> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Region2D<T>, Args...>: std::integral_constant<int, 2> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Region3D<T>, Args...>: std::integral_constant<int, 3> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Region4D<T>, Args...>: std::integral_constant<int, 4> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Index1D, Region1D<T>, Args...>: std::integral_constant<int, 1> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Index2D, Region2D<T>, Args...>: std::integral_constant<int, 2> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Index3D, Region3D<T>, Args...>: std::integral_constant<int, 3> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Index4D, Region4D<T>, Args...>: std::integral_constant<int, 4> {};
	
	
	template<size_t RC, typename T, typename... Args>
	struct mapoverlap_dimensionality<Random<RC>&, Region1D<T>, Args...>: std::integral_constant<int, 1> {};
	
	template<size_t RC, typename T, typename... Args>
	struct mapoverlap_dimensionality<Random<RC>&, Region2D<T>, Args...>: std::integral_constant<int, 2> {};
	
	template<size_t RC, typename T, typename... Args>
	struct mapoverlap_dimensionality<Random<RC>&, Region3D<T>, Args...>: std::integral_constant<int, 3> {};
	
	template<size_t RC, typename T, typename... Args>
	struct mapoverlap_dimensionality<Random<RC>&, Region4D<T>, Args...>: std::integral_constant<int, 4> {};
	
	template<size_t RC, typename T, typename... Args>
	struct mapoverlap_dimensionality<Index1D, Random<RC>&, Region1D<T>, Args...>: std::integral_constant<int, 1> {};
	
	template<size_t RC, typename T, typename... Args>
	struct mapoverlap_dimensionality<Index2D, Random<RC>&, Region2D<T>, Args...>: std::integral_constant<int, 2> {};
	
	template<size_t RC, typename T, typename... Args>
	struct mapoverlap_dimensionality<Index3D, Random<RC>&, Region3D<T>, Args...>: std::integral_constant<int, 3> {};
	
	template<size_t RC, typename T, typename... Args>
	struct mapoverlap_dimensionality<Index4D, Random<RC>&, Region4D<T>, Args...>: std::integral_constant<int, 4> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<T, Args...>: mapoverlap_dimensionality<typename std::decay<T>::type, Args...> {};
	
	
	
	
	
	
	
	
	
	template<typename T>
	struct Pool1D
	{
		int si;
		size_t stride;
		const T *data;
		
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
		T operator()(size_t i) const
		{
			return data[i * this->stride];
		}
		
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
		Pool1D(int arg_si, size_t arg_stride, const T *arg_data)
		: si(arg_si), stride(arg_stride), data(arg_data) {}
	};
	
	template<typename T>
	struct Pool2D
	{
		int si, sj;
		size_t stride;
		Index2D idx;
		const T *data;
		
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
		T operator()(size_t i, size_t j) const
		{
			int ii = this->idx.row + i;
			int jj = this->idx.col + j;
			return data[ii * this->stride + jj];
		}
		
	Pool2D(Matrix<T> const& mat, int arg_si, int arg_sj, Edge arg_edge, T arg_pad)
	:	si(arg_si), sj(arg_sj),
		stride(mat.size_j()),
		data(mat.getAddress())
	{}

	// For CUDA kernel
#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		Pool2D(int arg_si, int arg_sj, size_t arg_stride, T *arg_data)
		:	si(arg_si), sj(arg_sj),
			stride(arg_stride),
			data(arg_data),
			idx{0,0}
		{}
	};
	
	template<typename T>
	struct Pool3D
	{
		int si, sj, sk;
		size_t size_i, size_j, size_k;
		size_t offset[3];
		size_t stride1, stride2;
		Index3D idx;
		const T *data;
		Edge edge = Edge::None;
		T pad;
		
		#ifdef SKEPU_CUDA
				__host__ __device__
		#endif
				T operator()(int i, int j, int k) const
				{
					int ii = offset[0] + this->idx.i + i;
					int jj = offset[1] + this->idx.j + j;
					int kk = offset[2] + this->idx.k + k;
					size_t offset;
					
					switch (this->edge)
					{
					case Edge::Cyclic:
						offset =
							((ii + this->size_i) % this->size_i) * this->stride1 +
							((jj + this->size_j) % this->size_j) * this->stride2 +
							((kk + this->size_k) % this->size_k);
						return data[offset];
					case Edge::Pad:
						if ((0 <= ii && ii < size_i) && (0 <= jj && jj < size_j) &&
								(0 <= kk && kk < size_k))
							return data[ii * this->stride1 + jj * this->stride2 + kk];
						else return this->pad;
					case Edge::Duplicate:
						offset =
							future_std::clamp<int>(ii, 0, this->size_i-1) * this->stride1 +
							future_std::clamp<int>(jj, 0, this->size_j-1) * this->stride2 +
							future_std::clamp<int>(kk, 0, this->size_k-1);
						return data[offset];
					default:
						return data[ii * this->stride1 + jj * this->stride2 + kk];
					}
				}
		/*
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
		T operator()(size_t i, size_t j, size_t k) const
		{
			int ii = this->idx.i + i;
			int jj = this->idx.j + j;
			int kk = this->idx.k + k;
			return data[ii * this->stride1 + jj * this->stride2 + kk];
		}*/
		
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
	Pool3D(Tensor3<T> const& ten, int pool_size[3], size_t ofst[3], Edge arg_edge, T arg_pad)
	:	si(pool_size[0]), sj(pool_size[1]), sk(pool_size[2]),
		offset{ofst[0], ofst[1], ofst[2]},
		size_i(ten.size_i()), size_j(ten.size_j()), size_k(ten.size_k()),
		stride1(ten.m_stride_1),
		stride2(ten.m_stride_2),
		edge(arg_edge),
		pad(arg_pad),
		data(ten.getAddress())
	{}
	};
	
	template<typename T>
	struct Pool4D
	{
		int si, sj, sk, sl;
		size_t stride1, stride2, stride3;
		Index4D idx;
		const T *data;
		
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
		T operator()(size_t i, size_t j, size_t k, size_t l) const
		{
			int ii = this->idx.i + i;
			int jj = this->idx.j + j;
			int kk = this->idx.k + k;
			int ll = this->idx.l + l;
			return data[ii * this->stride1 + jj * this->stride2 + kk * this->stride3 + ll];
		}
	
	Pool4D(Tensor4<T> const& ten, int arg_si, int arg_sj, int arg_sk, int arg_sl, Edge arg_edge, T arg_pad,
	size_t arg_offset_i = 0, size_t arg_offset_j = 0, size_t arg_offset_k = 0, size_t arg_offset_l = 0)
	:	si(arg_si), sj(arg_sj), sk(arg_sk), sl(arg_sl),
		stride1(ten.m_stride_1),
		stride2(ten.m_stride_2),
		stride3(ten.m_stride_3),
		data(ten.getAddress())
	{}

#ifdef SKEPU_CUDA
	__host__ __device__
#endif
	Pool4D(int arg_si, int arg_sj, int arg_sk, int arg_sl, size_t arg_stride1, size_t arg_stride2, size_t arg_stride3, T *arg_data)
		:	si(arg_si), sj(arg_sj), sk(arg_sk), sl(arg_sl),
			stride1(arg_stride1), stride2(arg_stride2), stride3(arg_stride3),
			data(arg_data)
		{}
	};
	
	
	template<typename T>
	std::ostream& operator<<(std::ostream &os, Pool1D<T>& p)
	{
		os << "Pool1D: (" << p.si << ")\n";
		
		for (int i = 0; i < p.si; ++i)
		{
			auto val = p(i);
			std::cout << std::setw(5) << val << " ";
		}
		return os << "\n";
	}
	
	template<typename T>
	std::ostream& operator<<(std::ostream &os, Pool2D<T>& p)
	{
		os << "Pool2D: (" << p.si
			<< " x " << p.sj << ") @ [" << p.idx.row << ", " << p.idx.col << "]\n";
		
		for (size_t i = 0; i < p.si; ++i)
		{
			for (size_t j = 0; j < p.sj; ++j)
			{
				auto val = p(i,j);
				std::cout << std::setw(5) << val << " ";
			}
			os << "\n";
		}
		return os << "\n";
	}
	
	template<typename T>
	std::ostream& operator<<(std::ostream &os, Pool3D<T>& p)
	{
		os << "Pool3D: (" << p.si
			<< " x " << p.sj
			<< " x " << p.sk << ") @ [" << p.idx.i << ", " << p.idx.j << ", " << p.idx.k << "]\n";
		
		for (size_t j = 0; j < p.sj; ++j)
		{
			for (size_t i = 0; i < p.si; ++i)
			{
				for (size_t k = 0; k < p.sk; ++k)
				{
					auto val = p(i,j,k);
					std::cout << std::setw(5) << val << " ";
				}
				os << " | ";
			}
			os << "\n";
		}
		return os << "\n";
	}
	
	template<typename T>
	std::ostream& operator<<(std::ostream &os, Pool4D<T>& p)
	{
		os << "Pool4D: (" << p.si
			<< " x " << p.sj
			<< " x " << p.sk
			<< " x " << p.sl << ")\n";
		
		for (size_t i = 0; i < p.si; ++i)
		{
			for (size_t k = 0; k < p.sk; ++k)
			{
				for (size_t j = 0; j < p.sj; ++j)
				{
					for (size_t l = 0; l < p.sl; ++l)
					{
						auto val = p(i,j,k,l);
						std::cout << std::setw(5) << val << " ";
					}
					os << " | ";
				}
				os << "\n";
			}
			os << "---------------------------------\n";
		}
		return os << "\n";
	}
	
	
	
	template<typename T>
	struct region_type_h<Pool1D<T>> { using type = T; };
	
	template<typename T>
	struct region_type_h<Pool2D<T>> { using type = T; };
	
	template<typename T>
	struct region_type_h<Pool3D<T>> { using type = T; };
	
	template<typename T>
	struct region_type_h<Pool4D<T>> { using type = T; };
	
	
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Pool1D<T>, Args...>: std::integral_constant<int, 1> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Pool2D<T>, Args...>: std::integral_constant<int, 2> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Pool3D<T>, Args...>: std::integral_constant<int, 3> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Pool4D<T>, Args...>: std::integral_constant<int, 4> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Index1D, Pool1D<T>, Args...>: std::integral_constant<int, 1> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Index2D, Pool2D<T>, Args...>: std::integral_constant<int, 2> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Index3D, Pool3D<T>, Args...>: std::integral_constant<int, 3> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Index4D, Pool4D<T>, Args...>: std::integral_constant<int, 4> {};
	
	
	template<size_t RC, typename T, typename... Args>
	struct mapoverlap_dimensionality<Random<RC>&, Pool1D<T>, Args...>: std::integral_constant<int, 1> {};
	
	template<size_t RC, typename T, typename... Args>
	struct mapoverlap_dimensionality<Random<RC>&, Pool2D<T>, Args...>: std::integral_constant<int, 2> {};
	
	template<size_t RC, typename T, typename... Args>
	struct mapoverlap_dimensionality<Random<RC>&, Pool3D<T>, Args...>: std::integral_constant<int, 3> {};
	
	template<size_t RC, typename T, typename... Args>
	struct mapoverlap_dimensionality<Random<RC>&, Pool4D<T>, Args...>: std::integral_constant<int, 4> {};
	
	template<size_t RC, typename T, typename... Args>
	struct mapoverlap_dimensionality<Index1D, Random<RC>&, Pool1D<T>, Args...>: std::integral_constant<int, 1> {};
	
	template<size_t RC, typename T, typename... Args>
	struct mapoverlap_dimensionality<Index2D, Random<RC>&, Pool2D<T>, Args...>: std::integral_constant<int, 2> {};
	
	template<size_t RC, typename T, typename... Args>
	struct mapoverlap_dimensionality<Index3D, Random<RC>&, Pool3D<T>, Args...>: std::integral_constant<int, 3> {};
	
	template<size_t RC, typename T, typename... Args>
	struct mapoverlap_dimensionality<Index4D, Random<RC>&, Pool4D<T>, Args...>: std::integral_constant<int, 4> {};
	
	
}