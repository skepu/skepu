#pragma once

#include <vector>
#include <fstream>

#include <skepu>

template<typename T>
std::vector<T> deserialize_csv_helper(std::string fname)
{
	std::ifstream stream(fname);
	T val;
	std::vector<T> result;

	if (stream.is_open())
	{
		while (stream >> val)
		{
			result.push_back(val);
		}
	}
	else
	{
		std::cout << "Error: invalid file\n";
	}

	return result;
}


template<typename T>
typename std::vector<T>::iterator load_from_vector(skepu::Vector<T> &output, std::vector<T> &/*input*/, typename std::vector<T>::iterator it)
{
	for (size_t i = 0; i < output.size(); ++i)
	{
		output(i) = *it++;
	}
	return it;
}


template<typename T>
typename std::vector<T>::iterator load_from_vector(skepu::Matrix<T> &output, std::vector<T> &/*input*/, typename std::vector<T>::iterator it)
{
	for (size_t i = 0; i < output.size_i(); ++i)
	{
		for (size_t j = 0; j < output.size_j(); ++j)
		{
			output(i, j) = *it++;
		}
	}
	return it;
}


template<typename T>
typename std::vector<T>::iterator load_from_vector(skepu::Tensor3<T> &output, std::vector<T> &/*input*/, typename std::vector<T>::iterator it)
{
	for (size_t i = 0; i < output.size_i(); ++i)
	{
		for (size_t j = 0; j < output.size_j(); ++j)
		{
			for (size_t k = 0; k < output.size_k(); ++k)
			{
				output(i, j, k) = *it++;
			}
		}
	}
	return it;
}

template<typename T>
typename std::vector<T>::iterator load_from_vector(skepu::Tensor4<T> &output, std::vector<T> &/*input*/, typename std::vector<T>::iterator it)
{
	for (size_t i = 0; i < output.size_i(); ++i)
	{
		for (size_t j = 0; j < output.size_j(); ++j)
		{
			for (size_t k = 0; k < output.size_k(); ++k)
			{
				for (size_t l = 0; l < output.size_l(); ++l)
				{
					output(i, j, k, l) = *it++;
				}
			}
		}
	}
	return it;
}

