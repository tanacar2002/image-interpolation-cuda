#pragma once

#define DISABLE_W_CALC // Results in speedup

__device__ __forceinline__ uchar4 operator+(const uchar4& lhs, const uchar4& rhs){
    uchar4 res;
    res.x = lhs.x + rhs.x;
    res.y = lhs.y + rhs.y;
    res.z = lhs.z + rhs.z;
#ifndef DISABLE_W_CALC
    res.w = lhs.w + rhs.w;
#endif
    return res;
}

__device__ __forceinline__ uchar4 operator*(const uchar4& lhs, const uchar4& rhs){
    uchar4 res;
    res.x = lhs.x * rhs.x;
    res.y = lhs.y * rhs.y;
    res.z = lhs.z * rhs.z;
#ifndef DISABLE_W_CALC    
    res.w = lhs.w * rhs.w;
#endif
    return res;
}

__device__ __forceinline__ float4 operator+(const float4& lhs, const float4& rhs){
    float4 res;
    res.x = lhs.x + rhs.x;
    res.y = lhs.y + rhs.y;
    res.z = lhs.z + rhs.z;
#ifndef DISABLE_W_CALC
    res.w = lhs.w + rhs.w;
#endif
    return res;
}

__device__ __forceinline__ float4 operator*(const float4& lhs, const float4& rhs){
    float4 res;
    res.x = lhs.x * rhs.x;
    res.y = lhs.y * rhs.y;
    res.z = lhs.z * rhs.z;
#ifndef DISABLE_W_CALC
    res.w = lhs.w * rhs.w;
#endif
    return res;
}

__device__ __forceinline__ float4 operator+(const float& lhs, const uchar4& rhs){
    float4 res;
    res.x = lhs + rhs.x;
    res.y = lhs + rhs.y;
    res.z = lhs + rhs.z;
#ifndef DISABLE_W_CALC
    res.w = lhs + rhs.w;
#endif
    return res;
}

__device__ __forceinline__ float4 operator*(const float& lhs, const uchar4& rhs){
    float4 res;
    res.x = lhs * rhs.x;
    res.y = lhs * rhs.y;
    res.z = lhs * rhs.z;
#ifndef DISABLE_W_CALC
    res.w = lhs * rhs.w;
#endif
    return res;
}

__device__ __forceinline__ float4 operator+(const float& lhs, const float4& rhs){
    float4 res;
    res.x = lhs + rhs.x;
    res.y = lhs + rhs.y;
    res.z = lhs + rhs.z;
#ifndef DISABLE_W_CALC
    res.w = lhs + rhs.w;
#endif
    return res;
}

__device__ __forceinline__ float4 operator*(const float& lhs, const float4& rhs){
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
__device__ __forceinline__ T clamp(const T &x, const T &vmin, const T &vmax){
    return min(max(x, vmin), vmax);
}

__device__ __forceinline__ int clamp(const int &x, const int &vmax){
    return min(max(x, 0), vmax);
}

__device__ __forceinline__ uint8_t clamp(const uint8_t &x, const uint8_t &vmax){
    return min(x, vmax);
}

__device__ __forceinline__ uint32_t clamp(const uint32_t &x, const uint32_t &vmax){
    return min(x, vmax);
}

__device__ __forceinline__ float devsinc(const float &x){
    return (x == 0.f) ? 1.f : (__sinf(M_PIf*x)/(M_PIf*x));
}

template<int a = 4>
__device__ __forceinline__ float devlanc(const float &x){
    return (x == 0.f) ? 1.f : (a*__sinf(M_PIf*x)*__sinf((M_PIf/a)*x)/(M_PIf*M_PIf*x*x));
}

// Needed because of overshoot, when float values exceed the range of uint8_t
__device__ __forceinline__ uint8_t trunc2uint8(const float& val){
    return (val > 255.f) ? 255 : ((val < 0.f) ? 0 : ((uint8_t) val));
}
// Needed because of overshoot, when float values exceed the range of uint8_t
__device__ __forceinline__ uchar4 trunc2uchar4(const float4& val){
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

// Catmull Algorithm from https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
template<typename T>
__device__ __forceinline__ float cubic_interp(const float& t, const T& fn1, const T& f0, const T& f1, const T& f2){
    return t*(-0.5f + 1.f*t - 0.5f*t*t)*fn1 + (1.f - 2.5f*t*t + 1.5f*t*t*t)*f0 + t*(0.5f + 2.f*t - 1.5f*t*t)*f1 + t*t*(-0.5f + 0.5f*t)*f2;
}

template<typename T>
__device__ __forceinline__ float lancsoz4_interp(const float& t, const T* func){
    return devlanc(t+3.f)*func[0] + devlanc(t+2.f)*func[1] + devlanc(t+1.f)*func[2] + devlanc(t)*func[3] + devlanc(t-1.f)*func[4] + devlanc(t-2.f)*func[5] + devlanc(t-3.f)*func[6] + devlanc(t-4.f)*func[7];
}

// Catmull Algorithm from https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
__device__ __forceinline__ float4 cubic_interp4(const float& t, const uchar4 *f){
    return t*(-0.5f + 1.f*t - 0.5f*t*t)*f[0] + (1.f - 2.5f*t*t + 1.5f*t*t*t)*f[1] + t*(0.5f + 2.f*t - 1.5f*t*t)*f[2] + t*t*(-0.5f + 0.5f*t)*f[3];
}

__device__ __forceinline__ float4 cubic_interp4(const float& t, const float4 *f){
    return t*(-0.5f + 1.f*t - 0.5f*t*t)*f[0] + (1.f - 2.5f*t*t + 1.5f*t*t*t)*f[1] + t*(0.5f + 2.f*t - 1.5f*t*t)*f[2] + t*t*(-0.5f + 0.5f*t)*f[3];
}

// Needed because of overshoot and denormalizing, when float values exceed the range of uint8_t
__device__ __forceinline__ uchar4 convert2uchar4(const float4& val){
    uchar4 res;
    res.x = __saturatef(val.x)*255;
    res.y = __saturatef(val.y)*255;
    res.z = __saturatef(val.z)*255;
#ifndef DISABLE_W_CALC
    res.w = __saturatef(val.w)*255;
#else
    res.w = 255;
#endif
    return res;
}

__device__ __forceinline__ float4 lancsoz4_interp4(const float& t, const uchar4* func){
    return devlanc(t+3.f)*func[0] + devlanc(t+2.f)*func[1] + devlanc(t+1.f)*func[2] + devlanc(t)*func[3] + devlanc(t-1.f)*func[4] + devlanc(t-2.f)*func[5] + devlanc(t-3.f)*func[6] + devlanc(t-4.f)*func[7];
}

__device__ __forceinline__ float4 lancsoz4_interp4(const float& t, const float4* func){
    return devlanc(t+3.f)*func[0] + devlanc(t+2.f)*func[1] + devlanc(t+1.f)*func[2] + devlanc(t)*func[3] + devlanc(t-1.f)*func[4] + devlanc(t-2.f)*func[5] + devlanc(t-3.f)*func[6] + devlanc(t-4.f)*func[7];
}
