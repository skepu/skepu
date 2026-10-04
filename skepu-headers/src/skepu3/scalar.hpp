/*! \file vector.h
*  \brief Contains a class declaration for the Vector container.
 */

#ifndef SCALAR_H
#define SCALAR_H


namespace skepu
{

template<typename T>
class Scalar
{
public:
	using ID = UniqueIdentifier::ID;

	T m_value;
	std::vector<ID> m_ids;

	Scalar(T value, ID id): m_value{value}, m_ids{id}
	{}

	Scalar(T value, std::vector<ID> id): m_value{value}, m_ids{id}
	{}

	Scalar(T value, std::vector<ID> id1, std::vector<ID> id2): m_value{value}
	{
		this->m_ids = id1;
		this->m_ids.insert(this->m_ids.end(), id2.begin(), id2.end());
	}

public:

	Scalar(T value): m_value{value}, m_ids{UniqueIdentifier::generate()}
	{

	}

	operator T() const
	{
		return this->m_value;
	}

	template<typename Raw>
	void operator+=(Raw rhs)
	{
		this->m_value += rhs;
	}

	template<typename Raw>
	void operator-=(Raw rhs)
	{
		this->m_value -= rhs;
	}

	template<typename Raw>
	void operator*=(Raw rhs)
	{
		this->m_value *= rhs;
	}

	template<typename Raw>
	void operator/=(Raw rhs)
	{
		this->m_value /= rhs;
	}


	
	template<typename Raw>
	void operator+=(Scalar<Raw> rhs)
	{
		this->m_value += rhs.m_value;
		this->m_ids.insert(this->m_ids.end(), rhs.m_ids.begin(), rhs.m_ids.end());
	}

	template<typename Raw>
	void operator-=(Scalar<Raw> rhs)
	{
		this->m_value -= rhs.m_value;
		this->m_ids.insert(this->m_ids.end(), rhs.m_ids.begin(), rhs.m_ids.end());
	}

	template<typename Raw>
	void operator*=(Scalar<Raw> rhs)
	{
		this->m_value *= rhs.m_value;
		this->m_ids.insert(this->m_ids.end(), rhs.m_ids.begin(), rhs.m_ids.end());
	}

	template<typename Raw>
	void operator/=(Scalar<Raw> rhs)
	{
		this->m_value /= rhs.m_value;
		this->m_ids.insert(this->m_ids.end(), rhs.m_ids.begin(), rhs.m_ids.end());
	}

	

//	friend std::ostream &operator<< <T>(std::ostream &, Scalar<T>);
};



// Metafunctions for handling scalars in the general case of skepu::multiple

template<typename T>
struct scalar_wrapper
{
	using type = skepu::Scalar<T>;
};

template<typename... Ts>
struct scalar_wrapper<skepu::multiple<Ts...>>
{
	using type = skepu::multiple<skepu::Scalar<Ts>...>;;
};

template<typename T>
using scalar_wrapper_t = typename scalar_wrapper<T>::type;



template<typename T>
struct is_skepu_scalar: std::false_type {};

template<typename T>
struct is_skepu_scalar<skepu::Scalar<T>>: std::true_type {};



// Operator overloads

template<typename T>
Scalar<T> operator+(Scalar<T> scal)
{
	return scal;
}

template<typename T>
Scalar<T> operator-(Scalar<T> scal)
{
	return Scalar<T>(-(scal.m_value), scal.m_ids);
}



template<typename T, typename Raw>
decltype(auto) operator+(Scalar<T> lhs_scal, Raw rhs)
{
	return Scalar<decltype(lhs_scal.m_value + rhs)>(lhs_scal.m_value + rhs, lhs_scal.m_ids);
}

template<typename T, typename Raw>
decltype(auto) operator-(Scalar<T> lhs_scal, Raw rhs)
{
	return Scalar<decltype(lhs_scal.m_value - rhs)>(lhs_scal.m_value - rhs, lhs_scal.m_ids);
}

template<typename T, typename Raw>
decltype(auto) operator*(Scalar<T> lhs_scal, Raw rhs)
{
	return Scalar<decltype(lhs_scal.m_value * rhs)>(lhs_scal.m_value * rhs, lhs_scal.m_ids);
}

template<typename T, typename Raw>
decltype(auto) operator/(Scalar<T> lhs_scal, Raw rhs)
{
	return Scalar<decltype(lhs_scal.m_value / rhs)>(lhs_scal.m_value / rhs, lhs_scal.m_ids);
}

template<typename T, typename Raw>
decltype(auto) operator%(Scalar<T> lhs_scal, Raw rhs)
{
	return Scalar<decltype(lhs_scal.m_value % rhs)>(lhs_scal.m_value % rhs, lhs_scal.m_ids);
}

template<typename T, typename Raw>
decltype(auto) operator&(Scalar<T> lhs_scal, Raw rhs)
{
	return Scalar<decltype(lhs_scal.m_value & rhs)>(lhs_scal.m_value & rhs, lhs_scal.m_ids);
}

template<typename T, typename Raw>
decltype(auto) operator|(Scalar<T> lhs_scal, Raw rhs)
{
	return Scalar<decltype(lhs_scal.m_value | rhs)>(lhs_scal.m_value | rhs, lhs_scal.m_ids);
}

template<typename T, typename Raw>
decltype(auto) operator^(Scalar<T> lhs_scal, Raw rhs)
{
	return Scalar<decltype(lhs_scal.m_value ^ rhs)>(lhs_scal.m_value ^ rhs, lhs_scal.m_ids);
}

template<typename T, typename Raw>
Scalar<bool> operator<(Scalar<T> lhs_scal, Raw rhs)
{
	return Scalar<bool>(lhs_scal.m_value < rhs, lhs_scal.m_ids);
}

template<typename T, typename Raw>
Scalar<bool> operator>(Scalar<T> lhs_scal, Raw rhs)
{
	return Scalar<bool>(lhs_scal.m_value > rhs, lhs_scal.m_ids);
}

template<typename T, typename Raw>
Scalar<bool> operator==(Scalar<T> lhs_scal, Raw rhs)
{
	return Scalar<bool>(lhs_scal.m_value == rhs, lhs_scal.m_ids);
}

template<typename T, typename Raw>
Scalar<bool> operator<=(Scalar<T> lhs_scal, Raw rhs)
{
	return Scalar<bool>(lhs_scal.m_value <= rhs, lhs_scal.m_ids);
}

template<typename T, typename Raw>
Scalar<bool> operator>=(Scalar<T> lhs_scal, Raw rhs)
{
	return Scalar<bool>(lhs_scal.m_value >= rhs, lhs_scal.m_ids);
}

template<typename T, typename Raw>
decltype(auto) operator<<(Scalar<T> lhs_scal, Raw rhs)
{
	return Scalar<decltype(lhs_scal.m_value << rhs)>(lhs_scal.m_value << rhs, lhs_scal.m_ids);
}

template<typename T, typename Raw>
decltype(auto) operator>>(Scalar<T> lhs_scal, Raw rhs)
{
	return Scalar<decltype(lhs_scal.m_value >> rhs)>(lhs_scal.m_value >> rhs, lhs_scal.m_ids);
}




template<typename T, typename Raw>
Scalar<T> operator+(Raw lhs, Scalar<T> rhs_scal)
{
	return Scalar<T>(lhs + rhs_scal.m_value, rhs_scal.m_ids);
}

template<typename T, typename Raw>
Scalar<T> operator-(Raw lhs, Scalar<T> rhs_scal)
{
	return Scalar<T>(lhs - rhs_scal.m_value, rhs_scal.m_ids);
}

template<typename T, typename Raw>
Scalar<T> operator*(Raw lhs, Scalar<T> rhs_scal)
{
	return Scalar<T>(lhs * rhs_scal.m_value, rhs_scal.m_ids);
}

template<typename T, typename Raw>
Scalar<T> operator/(Raw lhs, Scalar<T> rhs_scal)
{
	return Scalar<T>(lhs / rhs_scal.m_value, rhs_scal.m_ids);
}

template<typename T, typename Raw>
Scalar<T> operator%(Raw lhs, Scalar<T> rhs_scal)
{
	return Scalar<T>(lhs % rhs_scal.m_value, rhs_scal.m_ids);
}

template<typename T, typename Raw>
Scalar<T> operator&(Raw lhs, Scalar<T> rhs_scal)
{
	return Scalar<T>(lhs & rhs_scal.m_value, rhs_scal.m_ids);
}

template<typename T, typename Raw>
Scalar<T> operator|(Raw lhs, Scalar<T> rhs_scal)
{
	return Scalar<T>(lhs | rhs_scal.m_value, rhs_scal.m_ids);
}

template<typename T, typename Raw>
Scalar<T> operator^(Raw lhs, Scalar<T> rhs_scal)
{
	return Scalar<T>(lhs ^ rhs_scal.m_value, rhs_scal.m_ids);
}

template<typename T, typename Raw>
Scalar<bool> operator<(Raw lhs, Scalar<T> rhs_scal)
{
	return Scalar<bool>(lhs < rhs_scal.m_value, rhs_scal.m_ids);
}

template<typename T, typename Raw>
Scalar<bool> operator>(Raw lhs, Scalar<T> rhs_scal)
{
	return Scalar<bool>(lhs > rhs_scal.m_value, rhs_scal.m_ids);
}

template<typename T, typename Raw>
Scalar<bool> operator==(Raw lhs, Scalar<T> rhs_scal)
{
	return Scalar<bool>(lhs == rhs_scal.m_value, rhs_scal.m_ids);
}

template<typename T, typename Raw>
Scalar<bool> operator<=(Raw lhs, Scalar<T> rhs_scal)
{
	return Scalar<bool>(lhs <= rhs_scal.m_value, rhs_scal.m_ids);
}

template<typename T, typename Raw>
Scalar<bool> operator>=(Raw lhs, Scalar<T> rhs_scal)
{
	return Scalar<bool>(lhs >= rhs_scal.m_value, rhs_scal.m_ids);
}

/*
template<typename T, typename Raw>
Scalar<T> operator<<(Raw lhs, Scalar<T> rhs_scal)
{
	return Scalar<T>(lhs << rhs_scal.m_value, rhs_scal.m_ids);
}

template<typename T, typename Raw>
Scalar<T> operator>>(Raw lhs, Scalar<T> rhs_scal)
{
	return Scalar<T>(lhs << rhs_scal.m_value, rhs_scal.m_ids);
}
*/






template<typename T, typename U>
Scalar<T> operator+(Scalar<T> lhs_scal, Scalar<U> rhs_scal)
{
	return Scalar<T>(lhs_scal.m_value + rhs_scal.m_value, lhs_scal.m_ids, rhs_scal.m_ids);
}

template<typename T, typename U>
Scalar<T> operator-(Scalar<T> lhs_scal, Scalar<U> rhs_scal)
{
	return Scalar<T>(lhs_scal.m_value - rhs_scal.m_value, lhs_scal.m_ids, rhs_scal.m_ids);
}

template<typename T, typename U>
Scalar<T> operator*(Scalar<T> lhs_scal, Scalar<U> rhs_scal)
{
	return Scalar<T>(lhs_scal.m_value * rhs_scal.m_value, lhs_scal.m_ids, rhs_scal.m_ids);
}

template<typename T, typename U>
Scalar<T> operator/(Scalar<T> lhs_scal, Scalar<U> rhs_scal)
{
	return Scalar<T>(lhs_scal.m_value / rhs_scal.m_value, lhs_scal.m_ids, rhs_scal.m_ids);
}

template<typename T, typename U>
Scalar<T> operator%(Scalar<T> lhs_scal, Scalar<U> rhs_scal)
{
	return Scalar<T>(lhs_scal.m_value % rhs_scal.m_value, lhs_scal.m_ids, rhs_scal.m_ids);
}

template<typename T, typename U>
Scalar<T> operator&(Scalar<T> lhs_scal, Scalar<U> rhs_scal)
{
	return Scalar<T>(lhs_scal.m_value & rhs_scal.m_value, lhs_scal.m_ids, rhs_scal.m_ids);
}

template<typename T, typename U>
Scalar<T> operator|(Scalar<T> lhs_scal, Scalar<U> rhs_scal)
{
	return Scalar<T>(lhs_scal.m_value | rhs_scal.m_value, lhs_scal.m_ids, rhs_scal.m_ids);
}

template<typename T, typename U>
Scalar<T> operator^(Scalar<T> lhs_scal, Scalar<U> rhs_scal)
{
	return Scalar<T>(lhs_scal.m_value ^ rhs_scal.m_value, lhs_scal.m_ids, rhs_scal.m_ids);
}

template<typename T, typename U>
Scalar<bool> operator<(Scalar<T> lhs_scal, Scalar<U> rhs_scal)
{
	return Scalar<bool>(lhs_scal.m_value < rhs_scal.m_value, lhs_scal.m_ids, rhs_scal.m_ids);
}

template<typename T, typename U>
Scalar<bool> operator>(Scalar<T> lhs_scal, Scalar<U> rhs_scal)
{
	return Scalar<bool>(lhs_scal.m_value > rhs_scal.m_value, lhs_scal.m_ids, rhs_scal.m_ids);
}

template<typename T, typename U>
Scalar<bool> operator==(Scalar<T> lhs_scal, Scalar<U> rhs_scal)
{
	return Scalar<bool>(lhs_scal.m_value == rhs_scal.m_value, lhs_scal.m_ids, rhs_scal.m_ids);
}

template<typename T, typename U>
Scalar<bool> operator<=(Scalar<T> lhs_scal, Scalar<U> rhs_scal)
{
	return Scalar<bool>(lhs_scal.m_value <= rhs_scal.m_value, lhs_scal.m_ids, rhs_scal.m_ids);
}

template<typename T, typename U>
Scalar<bool> operator>=(Scalar<T> lhs_scal, Scalar<U> rhs_scal)
{
	return Scalar<bool>(lhs_scal.m_value >= rhs_scal.m_value, lhs_scal.m_ids, rhs_scal.m_ids);
}

template<typename T, typename U>
Scalar<T> operator<<(Scalar<T> lhs_scal, Scalar<U> rhs_scal)
{
	return Scalar<T>(lhs_scal.m_value << rhs_scal.m_value, lhs_scal.m_ids, rhs_scal.m_ids);
}

template<typename T, typename U>
Scalar<T> operator>>(Scalar<T> lhs_scal, Scalar<U> rhs_scal)
{
	return Scalar<T>(lhs_scal.m_value >> rhs_scal.m_value, lhs_scal.m_ids, rhs_scal.m_ids);
}








template<typename T>
std::ostream &operator<<(std::ostream &os, Scalar<T> scalar)
{
	os << scalar.m_value;
	return os;
}

} // namespace skepu


#endif // SCALAR_H
