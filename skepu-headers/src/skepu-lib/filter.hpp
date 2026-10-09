#pragma once

#include <cmath>

#include <skepu>

namespace skepu {
namespace filter {


struct GrayscalePixel
{
	unsigned char intensity;
};

struct RGBPixel
{
	unsigned char r, g, b;
};

struct RGBAPixel
{
	unsigned char r, g, b, a;
};

struct HSVPixel
{
	float h, s, v;
};

struct HSVAPixel
{
	float h, s, v, a;
};

struct GrayscaleExtended
{
	float intensity;
};

struct RGBExtended
{
	float r, g, b;
};

struct RGBAExtended
{
	float r, g, b, a;
};




template<typename T>
inline T min3(T a, T b, T c)
{
	if (a < b && a < c)
		return a;
	else if (b < c)
		return b;
	else
		return c;
}

template<typename T>
inline T max3(T a, T b, T c)
{
	if (a > b && a > c)
		return a;
	else if (b > c)
		return b;
	else
		return c;
}

inline RGBExtended precision_extend_rgb(RGBPixel p)
{
	RGBExtended res;
	res.r = p.r / 255.f;
	res.g = p.g / 255.f;
	res.b = p.b / 255.f;
	return res;
}

inline RGBPixel precision_reduce_rgb(RGBExtended p)
{
	RGBPixel res;
	res.r = round(fmin(fmax(p.r, 0.f), 1.f) * 255.f);
	res.g = round(fmin(fmax(p.g, 0.f), 1.f) * 255.f);
	res.b = round(fmin(fmax(p.b, 0.f), 1.f) * 255.f);
	return res;
}

inline float gamma_helper(float in, float a, float gamma)
{
	return a * pow(in, gamma);
}

inline RGBExtended gamma_expansion_ex(RGBExtended p)
{
	float a = 1.f;
	float gamma = 2.f;
	p.r = gamma_helper(p.r, a, gamma);
	p.g = gamma_helper(p.g, a, gamma);
	p.b = gamma_helper(p.b, a, gamma);
	return p;
}

inline RGBExtended gamma_compression_ex(RGBExtended p)
{
	float a = 1.f;
	float gamma = 0.5f;
	p.r = gamma_helper(p.r, a, gamma);
	p.g = gamma_helper(p.g, a, gamma);
	p.b = gamma_helper(p.b, a, gamma);
	return p;
}


inline HSVPixel rgbex_to_hsv(RGBExtended p)
{
	float r = p.r;
	float g = p.g;
	float b = p.b;
	
	float M = max3<float>(r, g, b);
	float m = min3<float>(r, g, b);
	
	float C = M - m;
	
	float Hprime;
	
	if (C == 0)
		Hprime = 0; // "undefined"
	else if (M == r)
		Hprime = fmod((g - b) / C, 6.f);
	else if (M == g)
		Hprime = (b - r) / C + 2;
	else // M == b
		Hprime = (r - g) / C + 4;
	
	float H = Hprime * 60;
	if (H < 0)
		H += 360.f;
	
	float V = M;
	
	float S = (V == 0) ? 0 : C / V;
	
	HSVPixel res;
	res.h = H;
	res.s = S;
	res.v = V;
	return res;
}

inline HSVPixel rgb_to_hsv(RGBPixel p)
{
	RGBExtended ex = precision_extend_rgb(p);
	HSVPixel res = rgbex_to_hsv(ex);
	return res;
}

inline RGBExtended hsv_to_rgbex(HSVPixel p)
{
	float C = p.v * p.s;
	
	float h = fmod(p.h, 360.f);
	if (h < 0)
		h += 360.f;
	float Hprime = h / 60;
	
	float X = C * (1 - fabs((fmod(Hprime, 2)) - 1));
	
	float m = p.v - C;
	
	float tempR = 0, tempG = 0, tempB = 0;
	
	if (Hprime <= 1)
	{
		tempR = C; tempG = X; tempB = 0;
	}
	else if (Hprime <= 2)
	{
		tempR = X; tempG = C; tempB = 0;
	}
	else if (Hprime <= 3)
	{
		tempR = 0; tempG = C; tempB = X;
	}
	else if (Hprime <= 4)
	{
		tempR = 0; tempG = X; tempB = C;
	}
	else if (Hprime <= 5)
	{
		tempR = X; tempG = 0; tempB = C;
	}
	else if (Hprime <= 6)
	{
		tempR = C; tempG = 0; tempB = X;
	}
	
	
	RGBExtended res;
	res.r = fmin(fmax(0.f, tempR + m), 1.f);
	res.g = fmin(fmax(0.f, tempG + m), 1.f);
	res.b = fmin(fmax(0.f, tempB + m), 1.f);
	return res;
}

inline RGBPixel hsv_to_rgb(HSVPixel p)
{
	RGBExtended temp = hsv_to_rgbex(p);
	RGBPixel ret = precision_reduce_rgb(temp);
	return ret;
}


inline HSVAPixel rgba_to_hsva(RGBAPixel p)
{
	RGBPixel temp;
	temp.r = p.r;
	temp.g = p.g;
	temp.b = p.b;
	HSVPixel temp2 = rgb_to_hsv(temp);
	HSVAPixel res;
	res.h = temp2.h;
	res.s = temp2.s;
	res.v = temp2.v;
	res.a = p.a;
	return res;
}

inline RGBAPixel hsva_to_rgba(HSVAPixel p)
{
	HSVPixel temp;
	temp.h = p.h;
	temp.s = p.s;
	temp.v = p.v;
	RGBPixel temp2 = hsv_to_rgb(temp);
	RGBAPixel res;
	res.r = temp2.r;
	res.g = temp2.g;
	res.b = temp2.b;
	res.a = p.a;
	return res;
}

inline GrayscalePixel red_channel(RGBPixel p)
{
	GrayscalePixel output;
	output.intensity = p.r;
	return output;
}

inline GrayscalePixel green_channel(RGBPixel p)
{
	GrayscalePixel output;
	output.intensity = p.g;
	return output;
}

inline GrayscalePixel blue_channel(RGBPixel p)
{
	GrayscalePixel output;
	output.intensity = p.b;
	return output;
}

inline skepu::multiple<GrayscalePixel, GrayscalePixel, GrayscalePixel>
channels_from_rgb(RGBPixel p)
{
	GrayscalePixel output_r, output_g, output_b;
	output_r.intensity = p.r;
	output_g.intensity = p.g;
	output_b.intensity = p.b;
	return skepu::ret(output_r, output_g, output_b);
}

inline skepu::multiple<GrayscalePixel, GrayscalePixel, GrayscalePixel, GrayscalePixel>
channels_from_rgba(RGBAPixel p)
{
	GrayscalePixel output_r, output_g, output_b, output_a;
	output_r.intensity = p.r / 360.f * 255.f;
	output_g.intensity = p.g * 255.f;
	output_b.intensity = p.b * 255.f;
	output_a.intensity = p.a * 255.f;
	return skepu::ret(output_r, output_g, output_b, output_a);
}

inline skepu::multiple<GrayscaleExtended, GrayscaleExtended, GrayscaleExtended>
channels_from_rgbex(RGBExtended p)
{
	GrayscaleExtended output_r, output_g, output_b;
	output_r.intensity = p.r;
	output_g.intensity = p.g;
	output_b.intensity = p.b;
	return skepu::ret(output_r, output_g, output_b);
}

inline skepu::multiple<GrayscaleExtended, GrayscaleExtended, GrayscaleExtended, GrayscaleExtended>
channels_from_rgbaex(RGBAExtended p)
{
	GrayscaleExtended output_r, output_g, output_b, output_a;
	output_r.intensity = p.r;
	output_g.intensity = p.g;
	output_b.intensity = p.b;
	output_a.intensity = p.a;
	return skepu::ret(output_r, output_g, output_b, output_a);
}

inline RGBPixel channels_to_rgb(GrayscalePixel r, GrayscalePixel g, GrayscalePixel b)
{
	RGBPixel output;
	output.r = r.intensity / 255.f;
	output.g = g.intensity / 255.f;
	output.b = b.intensity / 255.f;
	return output;
}

inline RGBAPixel channels_to_rgba(GrayscalePixel r, GrayscalePixel g, GrayscalePixel b, GrayscalePixel a)
{
	RGBAPixel output;
	output.r = r.intensity / 255.f;
	output.g = g.intensity / 255.f;
	output.b = b.intensity / 255.f;
	output.a = a.intensity / 255.f;
	return output;
}

inline GrayscalePixel intensity_kernel(RGBPixel input)
{
	GrayscalePixel output;
	output.intensity = ((unsigned int)input.r + (unsigned int)input.g + (unsigned int)input.b) / 3;
	return output;
}

inline GrayscaleExtended intensity_kernel_ex(RGBExtended input)
{
	GrayscaleExtended output;
	output.intensity = (input.r + input.g + input.b) / 3;
	return output;
}




inline GrayscalePixel hue_channel(HSVPixel p)
{
	GrayscalePixel output;
	output.intensity = p.h / 360.f * 255.f;
	return output;
}

inline GrayscalePixel saturation_channel(HSVPixel p)
{
	GrayscalePixel output;
	output.intensity = p.s * 255.f;
	return output;
}

inline GrayscalePixel value_channel(HSVPixel p)
{
	GrayscalePixel output;
	output.intensity = p.v * 255.f;
	return output;
}

inline GrayscaleExtended hue_channel_ex(HSVPixel p)
{
	GrayscaleExtended output;
	output.intensity = p.h / 360.f;
	return output;
}

inline GrayscaleExtended saturation_channel_ex(HSVPixel p)
{
	GrayscaleExtended output;
	output.intensity = p.s;
	return output;
}

inline GrayscaleExtended value_channel_ex(HSVPixel p)
{
	GrayscaleExtended output;
	output.intensity = p.v;
	return output;
}

inline skepu::multiple<GrayscalePixel, GrayscalePixel, GrayscalePixel>
channels_from_hsv(HSVPixel p)
{
	GrayscalePixel output_h, output_s, output_v;
	output_h.intensity = p.h / 360.f * 255.f;
	output_s.intensity = p.s * 255.f;
	output_v.intensity = p.v * 255.f;
	return skepu::ret(output_h, output_s, output_v);
}

inline skepu::multiple<GrayscalePixel, GrayscalePixel, GrayscalePixel, GrayscalePixel>
channels_from_hsva(HSVAPixel p)
{
	GrayscalePixel output_h, output_s, output_v, output_a;
	output_h.intensity = p.h / 360.f * 255.f;
	output_s.intensity = p.s * 255.f;
	output_v.intensity = p.v * 255.f;
	output_a.intensity = p.a * 255.f;
	return skepu::ret(output_h, output_s, output_v, output_a);
}

inline skepu::multiple<GrayscaleExtended, GrayscaleExtended, GrayscaleExtended>
channels_from_hsv_ex(HSVPixel p)
{
	GrayscaleExtended output_h, output_s, output_v;
	output_h.intensity = p.h / 360.f;
	output_s.intensity = p.s;
	output_v.intensity = p.v;
	return skepu::ret(output_h, output_s, output_v);
}

inline skepu::multiple<GrayscaleExtended, GrayscaleExtended, GrayscaleExtended, GrayscaleExtended>
channels_from_hsva_ex(HSVAPixel p)
{
	GrayscaleExtended output_h, output_s, output_v, output_a;
	output_h.intensity = p.h / 360.f;
	output_s.intensity = p.s;
	output_v.intensity = p.v;
	output_a.intensity = p.a;
	return skepu::ret(output_h, output_s, output_v, output_a);
}

inline HSVPixel channels_to_hsv(GrayscalePixel h, GrayscalePixel s, GrayscalePixel v)
{
	HSVPixel output;
	output.h = h.intensity / 255.f * 360.f;
	output.s = s.intensity / 255.f;
	output.v = v.intensity / 255.f;
	return output;
}

inline HSVAPixel channels_to_hsva(GrayscalePixel h, GrayscalePixel s, GrayscalePixel v, GrayscalePixel a)
{
	HSVAPixel output;
	output.h = h.intensity / 255.f * 360.f;
	output.s = s.intensity / 255.f;
	output.v = v.intensity / 255.f;
	output.a = a.intensity / 255.f;
	return output;
}

inline HSVPixel channels_to_hsv_ex(GrayscaleExtended h, GrayscaleExtended s, GrayscaleExtended v)
{
	HSVPixel output;
	output.h = h.intensity * 360.f;
	output.s = s.intensity;
	output.v = v.intensity;
	return output;
}

inline HSVAPixel channels_to_hsva_ex(GrayscaleExtended h, GrayscaleExtended s, GrayscaleExtended v, GrayscaleExtended a)
{
	HSVAPixel output;
	output.h = h.intensity * 360.f;
	output.s = s.intensity;
	output.v = v.intensity;
	output.a = a.intensity;
	return output;
}

[[skepu::userconstant]] constexpr int
	ADDITION = 1,
	SUBTRACT = 2,
	DIFFERENCE = 3,
	MULTIPLY = 4,
	DIVIDE = 5,
	SCREEN = 6,
	OVERLAY = 7,
	DISSOLVE = 8;


inline float blend_helper_ex(float a, float b, int mode)
{
	switch (mode)
	{
	case ADDITION:
		return fmin(a + b, 1.f);
	case SUBTRACT:
		return fmax(a - b, 0.f);
	case DIFFERENCE:
		return fabs(a - b);
	case MULTIPLY:
		return (a * b);
	case DIVIDE:
		return (a / b);
	case SCREEN:
		return (1 - (1 - a) * (1 - b));
	case OVERLAY:
		return (a < 0.5 ? 2*a*b : 1 - 2*(1 - a)*(1 - b));
	default:
		return a;
	}
}

inline unsigned char blend_helper(unsigned char a_in, unsigned char b_in, int mode)
{
	float a = a_in / 255.f;
	float b = b_in / 255.f;
	
	return 255.f * blend_helper_ex(a, b, mode);
}

inline RGBPixel blend_rgb(RGBPixel a, RGBPixel b, int mode)
{
	a.r = blend_helper(a.r, b.r, mode);
	a.g = blend_helper(a.g, b.g, mode);
	a.b = blend_helper(a.b, b.b, mode);
	return a;
}

inline RGBExtended blend_rgbex(RGBExtended a, RGBExtended b, int mode)
{
	a.r = blend_helper_ex(a.r, b.r, mode);
	a.g = blend_helper_ex(a.g, b.g, mode);
	a.b = blend_helper_ex(a.b, b.b, mode);
	return a;
}

inline RGBPixel blend_rgb_gray(RGBPixel a, GrayscalePixel b, int mode)
{
	a.r = blend_helper(a.r, b.intensity, mode);
	a.g = blend_helper(a.g, b.intensity, mode);
	a.b = blend_helper(a.b, b.intensity, mode);
	return a;
}

inline RGBExtended blend_rgbex_gray(RGBExtended a, GrayscaleExtended b, int mode)
{
	a.r = blend_helper_ex(a.r, b.intensity, mode);
	a.g = blend_helper_ex(a.g, b.intensity, mode);
	a.b = blend_helper_ex(a.b, b.intensity, mode);
	return a;
}


inline unsigned char quantify_channel(unsigned char in, int levels)
{
	return (round(in / 255.f * levels) / levels) * 255; 
}

inline RGBPixel quantify(RGBPixel p, int levels)
{
	p.r = quantify_channel(p.r, levels);
	p.g = quantify_channel(p.g, levels);
	p.b = quantify_channel(p.b, levels);
	return p;
}

inline float quantify_channel_ex(float in, int levels)
{
	return round(in * levels) / levels; 
}

inline RGBExtended quantify_ex(RGBExtended p, int levels)
{
	p.r = quantify_channel_ex(p.r, levels);
	p.g = quantify_channel_ex(p.g, levels);
	p.b = quantify_channel_ex(p.b, levels);
	return p;
}

inline float color_distance(RGBPixel a, RGBPixel b)
{
	float dist_r = (float)a.r - (float)b.r;
	float dist_g = (float)a.g - (float)b.g;
	float dist_b = (float)a.b - (float)b.b;
	return sqrt(dist_r * dist_r + dist_g * dist_g + dist_b * dist_b);
}

inline float color_distance(RGBExtended a, RGBExtended b)
{
	float dist_r = (float)a.r - (float)b.r;
	float dist_g = (float)a.g - (float)b.g;
	float dist_b = (float)a.b - (float)b.b;
	return sqrt(dist_r * dist_r + dist_g * dist_g + dist_b * dist_b);
}

template<typename T>
inline T find_closest(T p, skepu::Vec<T> palette)
{
	size_t closest_i = 0;
	float closest_distance = 1E10;
	for (size_t i = 0; i < palette.size; ++i)
	{
		float dist = color_distance(p, palette(i));
		if (dist < closest_distance)
		{
			closest_distance = dist;
			closest_i = i;
		}
	}
	return palette(closest_i);
}

inline HSVPixel find_closest_hue(HSVPixel p, skepu::Vec<HSVPixel> palette)
{
	size_t closest_i = 0;
	float closest_distance = 1E10;
	for (size_t i = 0; i < palette.size; ++i)
	{
		float dist = fabs(p.h - palette(i).h);
		dist = fmin(dist, 360.f - dist);
		if (dist < closest_distance)
		{
			closest_distance = dist;
			closest_i = i;
		}
	}
	p.h = palette(closest_i).h;
	return p;
}

template<typename T>
inline T uniform_sample_picker(skepu::Index2D index, skepu::Mat<T> image, size_t out_rows, size_t out_cols)
{
	size_t i = (index.row * image.rows) / out_rows;
	size_t j = (index.col * image.cols) / out_cols;
	return image(i, j);
}



inline RGBPixel invert(RGBPixel p)
{
	p.r = 255 - p.r;
	p.g = 255 - p.g;
	p.b = 255 - p.b;
	return p;
}

inline RGBExtended invert_ex(RGBExtended p)
{
	p.r = 1.f - p.r;
	p.g = 1.f - p.g;
	p.b = 1.f - p.b;
	return p;
}


inline HSVPixel lighten_hsv(HSVPixel p, float quantity)
{
	p.v = fmin(p.v * (1.f + quantity), 1.f);
	return p;
}

inline RGBPixel lighten_rgb(RGBPixel p, float quantity)
{
	HSVPixel temp = rgb_to_hsv(p);
	temp = lighten_hsv(temp, quantity);
	return hsv_to_rgb(temp);
}

inline RGBExtended lighten_rgbex(RGBExtended p, float quantity)
{
	HSVPixel temp = rgbex_to_hsv(p);
	temp = lighten_hsv(temp, quantity);
	return hsv_to_rgbex(temp);
}

inline HSVPixel darken_hsv(HSVPixel p, float quantity)
{
	p.v = fmax(0.f, p.v * (1.f - quantity));
	return p;
}

inline RGBPixel darken_rgb(RGBPixel p, float quantity)
{
	HSVPixel temp = rgb_to_hsv(p);
	HSVPixel temp2 = darken_hsv(temp, quantity);
	return hsv_to_rgb(temp2);
}

inline RGBExtended darken_rgbex(RGBExtended p, float quantity)
{
	HSVPixel temp = rgbex_to_hsv(p);
	HSVPixel temp2 = darken_hsv(temp, quantity);
	return hsv_to_rgbex(temp2);
}

inline HSVPixel saturate_hsv(HSVPixel p, float quantity)
{
	p.s = fmin(p.s * (1.f + quantity), 1.f);
	return p;
}

inline RGBPixel saturate_rgb(RGBPixel p, float quantity)
{
	HSVPixel temp = rgb_to_hsv(p);
	temp = saturate_hsv(temp, quantity);
	return hsv_to_rgb(temp);
}

inline RGBExtended saturate_rgbex(RGBExtended p, float quantity)
{
	HSVPixel temp = rgbex_to_hsv(p);
	temp = saturate_hsv(temp, quantity);
	return hsv_to_rgbex(temp);
}

inline HSVPixel desaturate_hsv(HSVPixel p, float quantity)
{
	p.s = fmax(0.f, p.s * (1.f - quantity));
	return p;
}

inline RGBPixel desaturate_rgb(RGBPixel p, float quantity)
{
	HSVPixel temp = rgb_to_hsv(p);
	temp = desaturate_hsv(temp, quantity);
	return hsv_to_rgb(temp);
}

inline RGBExtended desaturate_rgbex(RGBExtended p, float quantity)
{
	HSVPixel temp = rgbex_to_hsv(p);
	temp = desaturate_hsv(temp, quantity);
	return hsv_to_rgbex(temp);
}


inline HSVPixel hue_rotate_hsv(HSVPixel p, float deg)
{
	p.h = fmod(p.h + deg, 360.f);
	if (p.h < 0)
		p.h += 360.f;
	return p;
}

inline RGBPixel hue_rotate_rgb(RGBPixel p, float deg)
{
	HSVPixel temp = rgb_to_hsv(p);
	temp = hue_rotate_hsv(temp, deg);
	return hsv_to_rgb(temp);
}

inline RGBExtended hue_rotate_rgbex(RGBExtended p, float deg)
{
	HSVPixel temp = rgbex_to_hsv(p);
	temp = hue_rotate_hsv(temp, deg);
	return hsv_to_rgbex(temp);
}

inline HSVPixel contrast_hsv(HSVPixel p, float quantity)
{
	p.v = 2.f * p.v - 1.f; // Map to [-1, 1]
	p.v *= quantity;
	p.v = fmin(fmax(0.5f * p.v + 0.5f, 0.f), 1.f); // Map to [0, 1]
	
	p.s = 2.f * p.s - 1.f; // Map to [-1, 1]
	p.s *= quantity;
	p.s = fmin(fmax(0.5f * p.s + 0.5f, 0.f), 1.f); // Map to [0, 1]
	return p;
}

inline RGBPixel contrast_rgb(RGBPixel p, float quantity)
{
	HSVPixel temp = rgb_to_hsv(p);
	temp = contrast_hsv(temp, quantity);
	return hsv_to_rgb(temp);
}

inline RGBExtended contrast_rgbex(RGBExtended p, float quantity)
{
	HSVPixel temp = rgbex_to_hsv(p);
	temp = contrast_hsv(temp, quantity);
	return hsv_to_rgbex(temp);
}






inline HSVPixel saturate_hsv_sp(HSVPixel p)
{
	float quantity = 0.5f;
	p.s = fmin(p.s * (1.f + quantity), 1.f);
	return p;
}


inline HSVPixel lighten_hsv_sp(HSVPixel p)
{
	float quantity = 0.5f;
	p.v = fmin(p.v * (1.f + quantity), 1.f);
	return p;
}




inline GrayscalePixel convolution_kernel(skepu::Region1D<GrayscalePixel> r, skepu::Vec<float> filter, float offset, float scaling)
{
	GrayscalePixel result;
	float intensity = 0;
	
	for (int i = -r.oi; i <= r.oi; i++)
	{
		GrayscalePixel p = r(i);
		intensity += p.intensity * filter(i+r.oi);
	}
	
	result.intensity = round(fmin(fmax((intensity + offset) * scaling, 0.f), 255.f));
	return result;
}

inline GrayscaleExtended convolution_kernel_ex(skepu::Region1D<GrayscaleExtended> r, skepu::Vec<float> filter, float offset, float scaling)
{
	GrayscaleExtended result;
	float intensity = 0;
	
	for (int i = -r.oi; i <= r.oi; i++)
	{
		GrayscaleExtended p = r(i);
		intensity += p.intensity * filter(i+r.oi);
	}
	
	result.intensity = (intensity + offset) * scaling;
	return result;
}

inline RGBPixel convolution_kernel_rgb_1d(skepu::Region1D<RGBPixel> in, skepu::Vec<float> filter, float offset, float scaling)
{
	float r = 0;
	float g = 0;
	float b = 0;
	
	
	for (int i = -in.oi; i <= in.oi; i++)
	{
		RGBPixel p = in(i);
		float coeff = filter(i + in.oi);
		r += p.r * coeff;
		g += p.g * coeff;
		b += p.b * coeff;
	}
	
	RGBPixel result;
	result.r = round(fmin(fmax((r + offset) * scaling, 0.f), 255.f));
	result.g = round(fmin(fmax((g + offset) * scaling, 0.f), 255.f));
	result.b = round(fmin(fmax((b + offset) * scaling, 0.f), 255.f));
	return result;
}

inline RGBExtended convolution_kernel_rgbex_1d(skepu::Region1D<RGBExtended> in, skepu::Vec<float> filter, float offset, float scaling)
{
	float r = 0;
	float g = 0;
	float b = 0;
	
	
	for (int i = -in.oi; i <= in.oi; i++)
	{
		RGBExtended p = in(i);
		float coeff = filter(i + in.oi);
		r += p.r * coeff;
		g += p.g * coeff;
		b += p.b * coeff;
	}
	
	RGBExtended result;
	result.r = (r + offset) * scaling;
	result.g = (g + offset) * scaling;
	result.b = (b + offset) * scaling;
	return result;
}

inline RGBPixel convolution_kernel_rgb_2d(skepu::Region2D<RGBPixel> in, skepu::Mat<float> filter, float offset, float scaling)
{
	float r = 0;
	float g = 0;
	float b = 0;
	
	for (int i = -in.oi; i <= in.oi; i++)
	{
		for (int j = -in.oj; j <= in.oj; j++)
		{
			RGBPixel p = in(i, j);
			float coeff = filter(i + in.oi, j + in.oj);
			r += p.r * coeff;
			g += p.g * coeff;
			b += p.b * coeff;
		}
	}
	
	RGBPixel result;
	result.r = round(fmin(fmax((r + offset) * scaling, 0.f), 255.f));
	result.g = round(fmin(fmax((g + offset) * scaling, 0.f), 255.f));
	result.b = round(fmin(fmax((b + offset) * scaling, 0.f), 255.f));
	return result;
}

inline RGBExtended convolution_kernel_rgbex_2d(skepu::Region2D<RGBExtended> in, skepu::Mat<float> filter, float offset, float scaling)
{
	float r = 0;
	float g = 0;
	float b = 0;
	
	for (int i = -in.oi; i <= in.oi; i++)
	{
		for (int j = -in.oj; j <= in.oj; j++)
		{
			RGBExtended p = in(i, j);
			float coeff = filter(i + in.oi, j + in.oj);
			r += p.r * coeff;
			g += p.g * coeff;
			b += p.b * coeff;
		}
	}
	
	RGBExtended result;
	result.r = (r + offset) * scaling;
	result.g = (g + offset) * scaling;
	result.b = (b + offset) * scaling;
	return result;
}

inline float gauss_weights_kernel(skepu::Index1D index, size_t r, float sigma)
{
	const float pi = 3.141592;
	float i = (float)index.i - r;
	return exp(-i*i / (2 * sigma * sigma)) / sqrt(2* pi * sigma * sigma);
}

inline GrayscalePixel distance_kernel(GrayscalePixel x, GrayscalePixel y)
{
	GrayscalePixel result;
	float xf = (float)x.intensity / 255 - 0.5;
	float yf = (float)y.intensity / 255 - 0.5;
	result.intensity = 255 - sqrt(xf*xf + yf*yf) * 2 * 255;
	return result;
}

inline GrayscaleExtended distance_kernel_ex(GrayscaleExtended x, GrayscaleExtended y)
{
	GrayscaleExtended result;
	float xf = x.intensity - 0.5;
	float yf = y.intensity - 0.5;
	result.intensity = 1 - sqrt(xf*xf + yf*yf) * 2;
	return result;
}

inline RGBPixel random_displacement_rgb(skepu::Random<2>& rnd, skepu::Region2D<RGBPixel> in)
{
	int i = (rnd.get() % (2 * in.oi + 1)) - in.oi;
	int j = (rnd.get() % (2 * in.oj + 1)) - in.oj;
	
	return in(i, j);
}

inline RGBExtended random_displacement_rgbex(skepu::Random<2>& rnd, skepu::Region2D<RGBExtended> in)
{
	int i = (rnd.get() % (2 * in.oi + 1)) - in.oi;
	int j = (rnd.get() % (2 * in.oj + 1)) - in.oj;
	RGBExtended result =  in(i, j);
	return result;
}




// Kernel for filter with raduis R
inline RGBPixel median_kernel(skepu::Region2D<RGBPixel> image)
{
	long fineHistogram[3][256], coarseHistogram[3][16];
	
	for (int c = 0; c < 3; c++)
		for (int i = 0; i < 256; i++)
			fineHistogram[c][i] = 0;
	
	for (int c = 0; c < 3; c++)
		for (int i = 0; i < 16; i++)
			coarseHistogram[c][i] = 0;
	
	for (int row = -image.oi; row <= image.oi; row++)
	{
		for (int column = -image.oj; column <= image.oj; column++)
		{ 
			unsigned char imageValue = image(row, column).r;
			fineHistogram[0][imageValue]++;
			coarseHistogram[0][imageValue / 16]++;
			
			imageValue = image(row, column).g;
			fineHistogram[1][imageValue]++;
			coarseHistogram[1][imageValue / 16]++;
			
			imageValue = image(row, column).b;
			fineHistogram[2][imageValue]++;
			coarseHistogram[2][imageValue / 16]++;
		}
	}
	
	unsigned char fineIndex[3];
	
	for (int c = 0; c < 3; c++)
	{
		int count = 2 * image.oi * (image.oi + 1);
		unsigned char coarseIndex;
		for (coarseIndex = 0; coarseIndex < 16; ++coarseIndex)
		{
			if ((long)count - coarseHistogram[c][coarseIndex] < 0) break;
			count -= coarseHistogram[c][coarseIndex];
		}
		
		fineIndex[c] = coarseIndex * 16;
		while ((long)count - fineHistogram[c][fineIndex[c]] >= 0)
			count -= fineHistogram[c][fineIndex[c]++];
	}
	
	RGBPixel res;
	res.r = fineIndex[0];
	res.g = fineIndex[1];
	res.b = fineIndex[2];
	return res;
}


inline RGBExtended median_kernel_ex(skepu::Region2D<RGBExtended> image)
{
	long fineHistogram[3][256], coarseHistogram[3][16];
	
	for (int c = 0; c < 3; c++)
		for (int i = 0; i < 256; i++)
			fineHistogram[c][i] = 0;
	
	for (int c = 0; c < 3; c++)
		for (int i = 0; i < 16; i++)
			coarseHistogram[c][i] = 0;
	
	for (int row = -image.oi; row <= image.oi; row++)
	{
		for (int column = -image.oj; column <= image.oj; column++)
		{ 
			unsigned char imageValue = image(row, column).r * 256;
			fineHistogram[0][imageValue]++;
			coarseHistogram[0][imageValue / 16]++;
			
			imageValue = image(row, column).g * 256;
			fineHistogram[1][imageValue]++;
			coarseHistogram[1][imageValue / 16]++;
			
			imageValue = image(row, column).b * 256;
			fineHistogram[2][imageValue]++;
			coarseHistogram[2][imageValue / 16]++;
		}
	}
	
	unsigned int fineIndex[3];
	
	for (int c = 0; c < 3; c++)
	{
		int count = 2 * image.oi * (image.oi + 1);
		unsigned char coarseIndex;
		for (coarseIndex = 0; coarseIndex < 16; ++coarseIndex)
		{
			if ((long)count - coarseHistogram[c][coarseIndex] < 0) break;
			count -= coarseHistogram[c][coarseIndex];
		}
		
		fineIndex[c] = coarseIndex * 16;
		while ((long)count - fineHistogram[c][fineIndex[c]] >= 0)
			count -= fineHistogram[c][fineIndex[c]++];
	}
	
	RGBExtended res;
	res.r = fineIndex[0] / 256.f;
	res.g = fineIndex[1] / 256.f;
	res.b = fineIndex[2] / 256.f;
	return res;
}


inline HSVPixel median_kernel_hsv(skepu::Region2D<HSVPixel> image)
{
	long fineHistogram[3][256], coarseHistogram[3][16];
	
	for (int c = 0; c < 3; c++)
		for (int i = 0; i < 256; i++)
			fineHistogram[c][i] = 0;
	
	for (int c = 0; c < 3; c++)
		for (int i = 0; i < 16; i++)
			coarseHistogram[c][i] = 0;
	
	for (int row = -image.oi; row <= image.oi; row++)
	{
		for (int column = -image.oj; column <= image.oj; column++)
		{ 
			unsigned char imageValue = image(row, column).h * 256 / 360.f;
			fineHistogram[0][imageValue]++;
			coarseHistogram[0][imageValue / 16]++;
			
			imageValue = image(row, column).s * 256;
			fineHistogram[1][imageValue]++;
			coarseHistogram[1][imageValue / 16]++;
			
			imageValue = image(row, column).v * 256;
			fineHistogram[2][imageValue]++;
			coarseHistogram[2][imageValue / 16]++;
		}
	}
	
	unsigned int fineIndex[3];
	
	for (int c = 0; c < 3; c++)
	{
		int count = 2 * image.oi * (image.oi + 1);
		unsigned char coarseIndex;
		for (coarseIndex = 0; coarseIndex < 16; ++coarseIndex)
		{
			if ((long)count - coarseHistogram[c][coarseIndex] < 0) break;
			count -= coarseHistogram[c][coarseIndex];
		}
		
		fineIndex[c] = coarseIndex * 16;
		while ((long)count - fineHistogram[c][fineIndex[c]] >= 0)
			count -= fineHistogram[c][fineIndex[c]++];
	}
	
	HSVPixel res;
	res.h= fineIndex[0] / 256.f * 360.f;
	res.s = fineIndex[1] / 256.f;
	res.v = fineIndex[2] / 256.f;
	return res;
}






}} // skepu::filter
