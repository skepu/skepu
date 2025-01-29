/*! \file debug.h
 *  \brief Defines a few macros that includes macros to output text when debugging. The macros use std::cerr.
 */

#ifndef DEBUG_H
#define DEBUG_H

#include <assert.h>
#include <sstream>
#include <iomanip>
#include <chrono>

#ifndef SKEPU_DEBUG
#define SKEPU_DEBUG 0
#endif


#if SKEPU_TUNING_DEBUG > 0
#include <iostream>
#endif

#if SKEPU_TUNING_DEBUG > 1
#define DEBUG_TUNING_LEVEL2(skepu_macro_text) std::cerr << "[SKEPU_TUNING_L1 " << __FILE__ << ":" << __LINE__ << "] " << skepu_macro_text << "\n";
#else
#define DEBUG_TUNING_LEVEL2(skepu_macro_text)
#endif

#if SKEPU_TUNING_DEBUG > 2
#define DEBUG_TUNING_LEVEL3(skepu_macro_text) std::cerr << "[SKEPU_TUNING_L2 " << __FILE__ << ":" << __LINE__ << "] " << skepu_macro_text << "\n";
#else
#define DEBUG_TUNING_LEVEL3(skepu_macro_text)
#endif

#if SKEPU_DEBUG > 0
#include <iostream>
#endif

#if SKEPU_DEBUG > 0
#define DEBUG_TEXT_LEVEL1(skepu_macro_text) std::cerr << "[" << debug_timestamp() << "][SKEPU_DEBUG_L1 " << __FILE__ << ":" << __LINE__ << "] " << skepu_macro_text << "\n";
#else
#define DEBUG_TEXT_LEVEL1(skepu_macro_text)
#endif

#if SKEPU_DEBUG > 1
#define DEBUG_TEXT_LEVEL2(skepu_macro_text) std::cerr << "[" << debug_timestamp() << "][SKEPU_DEBUG_L2 " << __FILE__ << ":" << __LINE__ << "] " << skepu_macro_text << "\n";
#else
#define DEBUG_TEXT_LEVEL2(skepu_macro_text)
#endif

#if SKEPU_DEBUG > 2
#define DEBUG_TEXT_LEVEL3(skepu_macro_text) std::cerr << "[" << debug_timestamp() << "][SKEPU_DEBUG_L3 " << __FILE__ << ":" << __LINE__ << "] " << skepu_macro_text << "\n";
#else
#define DEBUG_TEXT_LEVEL3(skepu_macro_text)
#endif


#ifndef SKEPU_ASSERT
#define SKEPU_ASSERT(expr) assert(expr)
#endif // SKEPU_ASSERT

#ifdef SKEPU_ENABLE_EXCEPTIONS
#define SKEPU_ERROR(skepu_macro_text) { std::stringstream skepu_macro_msg; skepu_macro_msg << skepu_macro_text; throw(skepu_macro_msg.str()); }
#else
#define SKEPU_ERROR(skepu_macro_text) { std::cerr << "[SKEPU_ERROR " << __FILE__ << ":" << __LINE__ << "] " << skepu_macro_text << "\n"; exit(1); }
#endif // SKEPU_ENABLE_EXCEPTIONS

#define SKEPU_WARNING(skepu_macro_text) { std::cerr << "[SKEPU_WARNING " << __FILE__ << ":" << __LINE__ << "] " << skepu_macro_text << "\n"; }

#define SKEPU_EXIT() exit(0)

#ifdef __GNUC__
#define SKEPU_UNLIKELY(expr)          (__builtin_expect(!!(expr),0))
#define SKEPU_LIKELY(expr)            (__builtin_expect(!!(expr),1))
#define SKEPU_ATTRIBUTE_UNUSED        __attribute__((unused))
#define SKEPU_ATTRIBUTE_INTERNAL      __attribute__ ((visibility ("internal")))
#else
#define SKEPU_UNLIKELY(expr)          (expr)
#define SKEPU_LIKELY(expr)            (expr)
#define SKEPU_ATTRIBUTE_UNUSED
#define SKEPU_ATTRIBUTE_INTERNAL
#endif

#ifndef SKEPU_NO_FORCE_INLINE
// Force inline in GCC and Clang (should also apply to NVCC?)
#if defined(__GNUC__) || defined(__clang__)
#define SKEPU_ATTRIBUTE_FORCE_INLINE __attribute__((always_inline))
// Force inline in MS VC
#elif defined(_MSC_VER)
#define SKEPU_ATTRIBUTE_FORCE_INLINE __forceinline
#else
// Intel compiler?
#define SKEPU_ATTRIBUTE_FORCE_INLINE
#endif
#else
#define SKEPU_ATTRIBUTE_FORCE_INLINE
#endif


inline std::string debug_timestamp()
{
	std::stringstream s;
/*	auto now = std::chrono::system_clock::now();
	auto now_c = std::chrono::system_clock::to_time_t(now);
	s << std::put_time(std::localtime(&now_c), "%H:%M:%S.");*/
	s << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	return s.str();
}

#endif
