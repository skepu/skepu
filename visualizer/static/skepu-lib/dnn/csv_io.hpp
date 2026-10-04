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

//  std::cout << "Loaded CSV file into std::vector. Total " << result.size() << " elements\n";
  return result;
}


template<typename T>
typename std::vector<T>::iterator load_from_vector(skepu::Vector<T> &output, std::vector<T> &input, typename std::vector<T>::iterator it)
{
  for (size_t i = 0; i < output.size(); ++i)
  {
    output(i) = *it++;
  }
//  std::cout << "Loaded std::vector into Vector. Total " << output.size() << " elements\n";
  return it;
}


template<typename T>
typename std::vector<T>::iterator load_from_vector(skepu::Matrix<T> &output, std::vector<T> &input, typename std::vector<T>::iterator it)
{
  for (size_t i = 0; i < output.size_i(); ++i)
  {
    for (size_t j = 0; j < output.size_j(); ++j)
    {
      output(i, j) = *it++;
    }
  }
//  std::cout << "Loaded std::vector into Matrix. Total " << output.size() << " elements\n";
  return it;
}


template<typename T>
typename std::vector<T>::iterator load_from_vector(skepu::Tensor3<T> &output, std::vector<T> &input, typename std::vector<T>::iterator it)
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
//  std::cout << "Loaded std::vector into Tensor3. Total " << output.size() << " elements\n";
  return it;
}

template<typename T>
typename std::vector<T>::iterator load_from_vector(skepu::Tensor4<T> &output, std::vector<T> &input, typename std::vector<T>::iterator it)
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
//  std::cout << "Loaded std::vector into Tensor4. Total " << output.size() << " elements\n";
  return it;
}


template<typename T>
skepu::Tensor4<T> cut_tensor4(skepu::Tensor4<T> &old, size_t size_i)
{
  if (size_i > old.size_i()) SKEPU_ERROR("Invalid size for cutting tensor");
  std::stringstream ss;
  ss << old.getLabel() << " (cut down)";
  skepu::Tensor4<T> res(size_i, old.size_j(), old.size_k(), old.size_l(), ss.str());
  
  old.flush();

  for (size_t i = 0; i < size_i; ++i)
  {
    for (size_t j = 0; j < old.size_j(); ++j)
    {
      for (size_t k = 0; k < old.size_k(); ++k)
      {
        for (size_t l = 0; l < old.size_l(); ++l)
        {
          res(i, j, k, l) = old(i, j, k, l);
        }
      }
    }
  }
  return res;
}
