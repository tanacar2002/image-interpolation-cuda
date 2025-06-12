#pragma once

#include <opencv2/opencv.hpp>

#define DISABLE_W_CALC // Results in speedup

// Structs from cccl vector_types.h
struct __attribute__((aligned(4))) uchar4{
    unsigned char x, y, z, w;
};
struct __attribute__((aligned(16))) float4{
    float x, y, z, w;
};

inline uchar4 operator+(const uchar4& lhs, const uchar4& rhs){
    uchar4 res;
    res.x = lhs.x + rhs.x;
    res.y = lhs.y + rhs.y;
    res.z = lhs.z + rhs.z;
#ifndef DISABLE_W_CALC
    res.w = lhs.w + rhs.w;
#endif
    return res;
}

inline uchar4 operator*(const uchar4& lhs, const uchar4& rhs){
    uchar4 res;
    res.x = lhs.x * rhs.x;
    res.y = lhs.y * rhs.y;
    res.z = lhs.z * rhs.z;
#ifndef DISABLE_W_CALC    
    res.w = lhs.w * rhs.w;
#endif
    return res;
}

inline float4 operator+(const float4& lhs, const float4& rhs){
    float4 res;
    res.x = lhs.x + rhs.x;
    res.y = lhs.y + rhs.y;
    res.z = lhs.z + rhs.z;
#ifndef DISABLE_W_CALC
    res.w = lhs.w + rhs.w;
#endif
    return res;
}

inline float4 operator*(const float4& lhs, const float4& rhs){
    float4 res;
    res.x = lhs.x * rhs.x;
    res.y = lhs.y * rhs.y;
    res.z = lhs.z * rhs.z;
#ifndef DISABLE_W_CALC
    res.w = lhs.w * rhs.w;
#endif
    return res;
}

inline float4 operator+(const float& lhs, const uchar4& rhs){
    float4 res;
    res.x = lhs + rhs.x;
    res.y = lhs + rhs.y;
    res.z = lhs + rhs.z;
#ifndef DISABLE_W_CALC
    res.w = lhs + rhs.w;
#endif
    return res;
}

inline float4 operator*(const float& lhs, const uchar4& rhs){
    float4 res;
    res.x = lhs * rhs.x;
    res.y = lhs * rhs.y;
    res.z = lhs * rhs.z;
#ifndef DISABLE_W_CALC
    res.w = lhs * rhs.w;
#endif
    return res;
}

inline float4 operator+(const float& lhs, const float4& rhs){
    float4 res;
    res.x = lhs + rhs.x;
    res.y = lhs + rhs.y;
    res.z = lhs + rhs.z;
#ifndef DISABLE_W_CALC
    res.w = lhs + rhs.w;
#endif
    return res;
}

inline float4 operator+=(float4& lhs, const float4& rhs){
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
#ifndef DISABLE_W_CALC
    lhs.w += rhs.w;
#endif
    return lhs;
}

inline float4 operator+=(float4& lhs, const uchar4& rhs){
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
#ifndef DISABLE_W_CALC
    lhs.w += rhs.w;
#endif
    return lhs;
}

inline float4 operator*(const float& lhs, const float4& rhs){
    float4 res;
    res.x = lhs * rhs.x;
    res.y = lhs * rhs.y;
    res.z = lhs * rhs.z;
#ifndef DISABLE_W_CALC
    res.w = lhs * rhs.w;
#endif
    return res;
}

template<typename T>
inline T clamp(const T &x, const T &vmin, const T &vmax){
    return MIN(MAX(x, vmin), vmax);
}

inline int clamp(const int &x, const int &vmax){
    return MIN(MAX(x, 0), vmax);
}

inline uint8_t clamp(const uint8_t &x, const uint8_t &vmax){
    return MIN(x, vmax);
}

inline uint32_t clamp(const uint32_t &x, const uint32_t &vmax){
    return MIN(x, vmax);
}

inline float devsinc(const float &x){
    return (x == 0.f) ? 1.f : (sinf(M_PIf*x)/(M_PIf*x));
}

template<int a = 4>
inline float devlanc(const float &x){
    return (x == 0.f) ? 1.f : (a*sinf(M_PIf*x)*sinf((M_PIf/a)*x)/(M_PIf*M_PIf*x*x));
}

// Needed because of overshoot, when float values exceed the range of uint8_t
inline uint8_t trunc2uint8(const float& val){
    return (val > 255.f) ? 255 : ((val < 0.f) ? 0 : ((uint8_t) val));
}

// Needed because of overshoot, when float values exceed the range of uint8_t
inline uchar4 trunc2uchar4(const float4& val){
    uchar4 res;
    res.x = trunc2uint8(val.x);
    res.y = trunc2uint8(val.y);
    res.z = trunc2uint8(val.z);
#ifndef DISABLE_W_CALC
    res.w = trunc2uint8(val.w);
#else
    res.w = 255;
#endif
    return res;
}

inline uchar4 cast2uchar4(const float4& val){
    uchar4 res;
    res.x = val.x;
    res.y = val.y;
    res.z = val.z;
#ifndef DISABLE_W_CALC
    res.w = val.w;
#else
    res.w = 255;
#endif
    return res;
}

// Catmull Algorithm from https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
// template<typename T>
// inline float cubic_interp(const float& t, const T& fn1, const T& f0, const T& f1, const T& f2){
//     return t*(-0.5f + 1.f*t - 0.5f*t*t)*fn1 + (1.f - 2.5f*t*t + 1.5f*t*t*t)*f0 + t*(0.5f + 2.f*t - 1.5f*t*t)*f1 + t*t*(-0.5f + 0.5f*t)*f2;
// }
// 
// template<typename T>
// inline float cubic_interp(const float& t, const T *f){
//     return t*(-0.5f + 1.f*t - 0.5f*t*t)*f[0] + (1.f - 2.5f*t*t + 1.5f*t*t*t)*f[1] + t*(0.5f + 2.f*t - 1.5f*t*t)*f[2] + t*t*(-0.5f + 0.5f*t)*f[3];
// }

inline void cubic_coeffs(const float& t, float* const coeffs){
    coeffs[0] = t*(-0.5f + 1.f*t - 0.5f*t*t);
    coeffs[1] = 1.f - 2.5f*t*t + 1.5f*t*t*t;
    coeffs[2] = t*(0.5f + 2.f*t - 1.5f*t*t);
    coeffs[3] = t*t*(-0.5f + 0.5f*t);
}

template<typename T, int step=1>
inline float cubic_interp_coeffs(float* const coeffs, const T* func){
    float sum(0.f);
    for(int i = 0; i < 4; i++){
        sum += coeffs[i]*func[i*step];
    }
    return sum;
}

inline float4 cubic_interp4_coeffs(float* const coeffs, const uchar4* func){
    float4 sum;
    sum.x = 0.f;
    sum.y = 0.f;
    sum.z = 0.f;
#ifndef DISABLE_W_CALC
    sum.w = 0.f;
#endif
    for(int i = 0; i < 4; i++){
        sum += coeffs[i]*func[i];
    }
    return sum;
}

inline float4 cubic_interp4_coeffs(float* const coeffs, const float4* func){
    float4 sum;
    sum.x = 0.f;
    sum.y = 0.f;
    sum.z = 0.f;
#ifndef DISABLE_W_CALC
    sum.w = 0.f;
#endif
    for(int i = 0; i < 4; i++){
        sum += coeffs[i]*func[i];
    }
    return sum;
}

template<typename T, int step=1>
inline float cubic_interp(const float& t, const T *f){
    return t*(-0.5f + 1.f*t - 0.5f*t*t)*f[0] + (1.f - 2.5f*t*t + 1.5f*t*t*t)*f[step] + t*(0.5f + 2.f*t - 1.5f*t*t)*f[2*step] + t*t*(-0.5f + 0.5f*t)*f[3*step];
}

inline float4 cubic_interp4(const float& t, const uchar4 *f){
    return t*(-0.5f + 1.f*t - 0.5f*t*t)*f[0] + (1.f - 2.5f*t*t + 1.5f*t*t*t)*f[1] + t*(0.5f + 2.f*t - 1.5f*t*t)*f[2] + t*t*(-0.5f + 0.5f*t)*f[3];
}

inline float4 cubic_interp4(const float& t, const float4 *f){
    return t*(-0.5f + 1.f*t - 0.5f*t*t)*f[0] + (1.f - 2.5f*t*t + 1.5f*t*t*t)*f[1] + t*(0.5f + 2.f*t - 1.5f*t*t)*f[2] + t*t*(-0.5f + 0.5f*t)*f[3];
}

inline void lancsoz4_coeffs(const float& t, float* const coeffs){
    coeffs[0] = devlanc(t+3.f);
    coeffs[1] = devlanc(t+2.f);
    coeffs[2] = devlanc(t+1.f);
    coeffs[3] = devlanc(t);
    coeffs[4] = devlanc(t-1.f);
    coeffs[5] = devlanc(t-2.f);
    coeffs[6] = devlanc(t-3.f);
    coeffs[7] = devlanc(t-4.f);
}

template<typename T, int step=1>
inline float lancsoz4_interp_coeffs(float* const coeffs, const T* func){
    float sum(0.f);
    for(int i = 0; i < 8; i++){
        sum += coeffs[i]*func[i*step];
    }
    return sum;
}

inline float4 lancsoz4_interp4_coeffs(float* const coeffs, const uchar4* func){
    float4 sum;
    sum.x = 0.f;
    sum.y = 0.f;
    sum.z = 0.f;
#ifndef DISABLE_W_CALC
    sum.w = 0.f;
#endif
    for(int i = 0; i < 8; i++){
        sum += coeffs[i]*func[i];
    }
    return sum;
}

inline float4 lancsoz4_interp4_coeffs(float* const coeffs, const float4* func){
    float4 sum;
    sum.x = 0.f;
    sum.y = 0.f;
    sum.z = 0.f;
#ifndef DISABLE_W_CALC
    sum.w = 0.f;
#endif
    for(int i = 0; i < 8; i++){
        sum += coeffs[i]*func[i];
    }
    return sum;
}

template<typename T, int step=1>
inline float lancsoz4_interp(const float& t, const T* func){
    return devlanc(t+3.f)*func[0] + devlanc(t+2.f)*func[step] + devlanc(t+1.f)*func[2*step] + devlanc(t)*func[3*step] + devlanc(t-1.f)*func[4*step] + devlanc(t-2.f)*func[5*step] + devlanc(t-3.f)*func[6*step] + devlanc(t-4.f)*func[7*step];
}

inline float4 lancsoz4_interp4(const float& t, const uchar4* func){
    return devlanc(t+3.f)*func[0] + devlanc(t+2.f)*func[1] + devlanc(t+1.f)*func[2] + devlanc(t)*func[3] + devlanc(t-1.f)*func[4] + devlanc(t-2.f)*func[5] + devlanc(t-3.f)*func[6] + devlanc(t-4.f)*func[7];
}

inline float4 lancsoz4_interp4(const float& t, const float4* func){
    return devlanc(t+3.f)*func[0] + devlanc(t+2.f)*func[1] + devlanc(t+1.f)*func[2] + devlanc(t)*func[3] + devlanc(t-1.f)*func[4] + devlanc(t-2.f)*func[5] + devlanc(t-3.f)*func[6] + devlanc(t-4.f)*func[7];
}
