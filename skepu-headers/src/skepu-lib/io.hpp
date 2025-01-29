#pragma once

#include <iostream>

namespace skepu {
namespace io {

template<typename T>
class is_skepu_ostream: std::false_type {};

template<typename T>
class is_skepu_istream: std::false_type {};


class ostream
{
public:
  static constexpr bool is_skepu_ostream = true;
  std::ostream &m_stream;

  explicit ostream(std::ostream &dest)
  : m_stream(dest) {}
};


class istream
{
public:
  static constexpr bool is_skepu_istream = true;
  std::istream &m_stream;

  explicit istream(std::istream &source)
  : m_stream(source) {}
};


template<>
class is_skepu_ostream<ostream>: std::true_type {};

template<>
class is_skepu_istream<istream>: std::true_type {};


template<typename T>
ostream &operator<<(ostream &s, T const& arg)
{
  s.m_stream << arg;
  return s;
}

template<typename C, REQUIRES_VALUE(is_skepu_container<C>)>
ostream &operator<<(ostream &s, C &arg)
{
  skepu::external("I/O", skepu::read(arg), [&]{
    s.m_stream << arg;
  });
  return s;
}

template<typename T>
ostream &&operator<<(ostream &&s, T const& arg)
{
  s.m_stream << arg;
  return std::forward<ostream>(s);
}

template<typename C, REQUIRES_VALUE(is_skepu_container<C>)>
ostream &&operator<<(ostream &&s, C &arg)
{
  skepu::external("I/O", skepu::read(arg), [&]{
    s.m_stream << arg;
  });
  return std::forward<ostream>(s);
}




template<typename Stream, typename T, REQUIRES_VALUE(is_skepu_istream<Stream>)>
Stream &&operator<<(Stream &&s, T const& arg)
{
  s.m_stream << arg;
  return std::forward<istream>(s);
}

template<typename Stream, typename C, REQUIRES_VALUE(is_skepu_istream<Stream>), REQUIRES_VALUE(is_skepu_container<C>)>
Stream &&operator<<(Stream &&s, C &arg)
{
  skepu::external("I/O", [&]{
    s.m_stream << arg;
  }, skepu::write(arg));
  return std::forward<istream>(s);
}

static ostream cout{std::cout}; // TODO
static istream cin{std::cin}; // TODO

}} // skepu::io
