// Do not alter the preprocessor directives
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdint.h>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/tuple.h>
#include <thrust/version.h>
#include <thrust/iterator/constant_iterator.h>

#define NUM_CHANNELS 1
struct multiply_functor{
    __device__ __host__ uint8_t operator()(uint8_t lhs, float rhs){
        return (uint8_t)(rhs*(float)lhs);
    }
};

struct scale_functor{
    uint8_t nMin = 0;
    float scale = 1.f;
    __device__ __host__ scale_functor(uint8_t nMin, float scale){
        this->nMin = nMin;
        this->scale = scale;
    }
    __device__ __host__ uint8_t operator()(uint8_t &i){
        i = (uint8_t)(this->scale*(float)(i-this->nMin));
        return i; //(uint8_t)(this->scale*(float)(i-this->nMin));
    }
};

struct minmax_functor{
    __device__ __host__ thrust::tuple<uint8_t,uint8_t> operator()(thrust::tuple<uint8_t,uint8_t> lhs, thrust::tuple<uint8_t,uint8_t> rhs){
        uint8_t rMin = thrust::get<0>(rhs);
        uint8_t rMax = thrust::get<1>(rhs);
        uint8_t lMin = thrust::get<0>(lhs);
        uint8_t lMax = thrust::get<1>(lhs);
        return thrust::make_tuple(lMin < rMin ? lMin : rMin, lMax < rMax ? rMax : lMax);
    }
};

// main routine that executes on the host
int main(int argc, char *argv[])
{
    int width; //image width
    int height; //image height
    int bpp;  //bytes per pixel if the image was RGB (not used)
    printf("Thrust Version: %d\n", THRUST_VERSION);

    // Create timer events
	cudaEvent_t start, mid, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&mid);
	cudaEventCreate(&stop);

    std::string filename("./samples/640x426.bmp");
    if(argc > 1) filename = argv[1];

    // Load a grayscale bmp image to an unsigned integer array with its height and weight.
    //  (uint8_t is an alias for "unsigned char")
    uint8_t* image = stbi_load(filename.c_str(), &width, &height, &bpp, NUM_CHANNELS);
    // Print for sanity check
    printf("Bytes per pixel: %d\n", bpp / 3); //Image is grayscale, so bpp / 3;
    printf("Height: %d\n", height);
    printf("Width: %d\n", width);

    size_t N = width * height;

    // thrust::device_vector<uint8_t> d_image(N);
    // thrust::host_vector<uint8_t> h_image(image, image+N);
    // thrust::copy(h_image.begin(), h_image.end(), d_image.begin());

    thrust::device_vector<uint8_t> d_image(image, image+N);
    thrust::device_vector<uint8_t> nMinMaxVec(2);
    thrust::host_vector<uint8_t> h_image(N);

	cudaEventRecord(start); // Start timer

    // Seperate Reductions 4.02009 ms
    // uint8_t nMax = thrust::reduce(d_image.begin(), d_image.end(), 0U, thrust::maximum<uint8_t>());
    // uint8_t nMin = thrust::reduce(d_image.begin(), d_image.end(), 255U, thrust::minimum<uint8_t>());

    // Combined Reduction 3.41223 ms
    thrust::tuple<uint8_t,uint8_t> nMinMax = thrust::reduce(thrust::make_zip_iterator(thrust::make_tuple(d_image.begin(),d_image.begin())), thrust::make_zip_iterator(thrust::make_tuple(d_image.end(),d_image.end())), thrust::make_tuple<uint8_t,uint8_t>(255U,0U), minmax_functor());
    // To Be Used in the Future (to avoid copying back to host)
    // thrust::tuple<uint8_t,uint8_t> nMinMax = thrust::reduce_into(thrust::make_zip_iterator(thrust::make_tuple(d_image.begin(),d_image.begin())), thrust::make_zip_iterator(thrust::make_tuple(d_image.end(),d_image.end())), thrust::make_tuple<uint8_t,uint8_t>(255U,0U), minmax_functor());
	cudaEventRecord(mid);
    
    uint8_t nMin = thrust::get<0>(nMinMax);
    uint8_t nMax = thrust::get<1>(nMinMax);

    float scale = 255.f/(nMax-nMin);
    
    // Separate Transforms 0.32209 ms
    // thrust::constant_iterator<uint8_t> nMin_iter(nMin);
    // thrust::constant_iterator<float> scale_iter(scale);
    // thrust::transform(d_image.begin(), d_image.end(), nMin_iter, d_image.begin(), thrust::minus<uint8_t>());
    // thrust::transform(d_image.begin(), d_image.end(), scale_iter, d_image.begin(), multiply_functor());

    scale_functor scale_func(nMin, scale);

    // Combined Transform 0.18114 ms
    // thrust::transform(d_image.begin(), d_image.end(), d_image.begin(), scale_func);

    // Inplace Computation 0.14415 ms
    thrust::for_each(d_image.begin(), d_image.end(), scale_func);
    

	cudaEventRecord(stop); // Stop timer
	cudaEventSynchronize(stop);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	float time;
	cudaEventElapsedTime(&time, start, mid);
	printf("minmax: %f  ms\n", time);
    cudaEventElapsedTime(&time, mid, stop);
	printf("subscale: %f  ms\n", time);
	cudaEventElapsedTime(&time, start, stop);
	printf("Time elapsed for %lu elements: %f ms\n", N, time);

    thrust::copy(d_image.begin(), d_image.end(), h_image.begin());
    uint8_t* h_image_ptr = thrust::raw_pointer_cast<uint8_t*>(&h_image[0]);

    printf("Min: %u\t Max: %u \n", nMin, nMax);
    // Write image array into a bmp file
    filename.replace(filename.find_last_of("."), filename.length(), "_out_thrust.bmp");
    stbi_write_bmp(filename.c_str(), width, height, 1, h_image_ptr);

    // Deallocate memory
    stbi_image_free(image);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
