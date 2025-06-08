// Do not alter the preprocessor directives
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdint.h>
#include <string>
#include <stdio.h>
#include <cuda.h>

#define NUM_CHANNELS 1

#define WARP 32
#define TPB 1024

__global__ void contrast_device(uint8_t *img, uint32_t *nMax, uint32_t *nMin, size_t N)
{
    __shared__ uint8_t out[2][WARP][WARP];

    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t bi = blockIdx.x;
    uint32_t BX = blockDim.x;
    uint32_t BY = blockDim.y;
    uint32_t Nthreads = BX*BY*gridDim.x;
    
    uint32_t n = tx + ty*BX + bi*BX*BY;
    uint32_t tmpMin = 255;
    uint32_t tmpMax = 0;
    *nMin = tmpMin;
    *nMax = tmpMax;
    // Align data with thread
    uint32_t i = n;
    while(i < N){
        tmpMin = min(img[i], tmpMin);
        tmpMax = max(img[i], tmpMax);
        i += Nthreads;
    }

    out[0][ty][tx] = tmpMin;
    out[1][ty][tx] = tmpMax;
    __syncwarp();
    {
        if(tx<16){
            out[0][ty][tx] = min(out[0][ty][tx], out[0][ty][tx+16]);
            out[1][ty][tx] = max(out[1][ty][tx], out[1][ty][tx+16]);
        }
        __syncwarp();
        if(tx<8){
            out[0][ty][tx] = min(out[0][ty][tx], out[0][ty][tx+8]);
            out[1][ty][tx] = max(out[1][ty][tx], out[1][ty][tx+8]);
        }
        __syncwarp();
        if(tx<4){
            out[0][ty][tx] = min(out[0][ty][tx], out[0][ty][tx+4]);
            out[1][ty][tx] = max(out[1][ty][tx], out[1][ty][tx+4]);
        }
        __syncwarp();
        if(tx<2){
            out[0][ty][tx] = min(out[0][ty][tx], out[0][ty][tx+2]);
            out[1][ty][tx] = max(out[1][ty][tx], out[1][ty][tx+2]);
        }
        __syncwarp();
        // diagonal to avoid bank conflicts
        if(tx<1){
            out[0][ty][ty] = min(out[0][ty][0], out[0][ty][1]);
            out[1][ty][ty] = max(out[1][ty][0], out[1][ty][1]);
        }
    }
    // synchronize all warps
    __syncthreads();

    if(ty==0){
        if(tx<16){
            out[0][0][tx] = min(out[0][tx][tx], out[0][tx+16][tx+16]);
            out[1][0][tx] = max(out[1][tx][tx], out[1][tx+16][tx+16]);
        }
        __syncwarp();
        if(tx<8){
            out[0][ty][tx] = min(out[0][0][tx], out[0][0][tx+8]);
            out[1][ty][tx] = max(out[1][0][tx], out[1][0][tx+8]);
        }
        __syncwarp();
        if(tx<4){
            out[0][ty][tx] = min(out[0][0][tx], out[0][0][tx+4]);
            out[1][ty][tx] = max(out[1][0][tx], out[1][0][tx+4]);
        }
        __syncwarp();
        if(tx<2){
            out[0][ty][tx] = min(out[0][0][tx], out[0][0][tx+2]);
            out[1][ty][tx] = max(out[1][0][tx], out[1][0][tx+2]);
        }
        __syncwarp();
        if( tx< 1){
            tmpMin = min(out[0][0][0], out[0][0][1]);
            tmpMax = max(out[1][0][0], out[1][0][1]);
            atomicMin(nMin, tmpMin);
            atomicMax(nMax, tmpMax);
        }
    }
    __syncthreads();
    uint32_t localmin = *nMin;
    float scale = 255. / (float)(*nMax-localmin);

    while(n < N){
        img[n] = ((float) (img[n]-localmin))* scale;
        n += Nthreads;
    }
}

// main routine that executes on the host
int main(int argc, char *argv[])
{
    int width; //image width
    int height; //image height
    int bpp;  //bytes per pixel if the image was RGB (not used)

    uint8_t *image_d; // Pointer to device image array
    uint32_t *nMin, *nMax; // min and max pixel values
    uint32_t *nMin_d, *nMax_d; // Pointer to device min and max pixel values

    // Create timer events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    std::string filename("./samples/640x426.bmp");
    if(argc > 1) {
        filename = argv[1];
    }

    // Load a grayscale bmp image to an unsigned integer array with its height and weight.
    //  (uint8_t is an alias for "unsigned char")
    uint8_t* image = stbi_load(filename.c_str(), &width, &height, &bpp, NUM_CHANNELS);
    // Print for sanity check
    printf("Bytes per pixel: %d \n", bpp / 3); //Image is grayscale, so bpp / 3;
    printf("Height: %d \n", height);
    printf("Width: %d \n", width);
    
    size_t N = width * height;
	size_t size = N * sizeof(uint8_t);

    size_t NUM_BLOCKS = N / TPB + (N % TPB == 0 ? 0 : 1);
    size_t num_block = NUM_BLOCKS / 16;
    if(argc > 2) {
        num_block = atoi(argv[2]);
    }
    nMin = (uint32_t*) malloc(sizeof(uint32_t));
    nMax = (uint32_t*) malloc(sizeof(uint32_t));
	// Allocate memory on device
	cudaMalloc((void **)&nMin_d, sizeof(uint32_t));
	cudaMalloc((void **)&nMax_d, sizeof(uint32_t));
	cudaMalloc((void **)&image_d, size);

	// Copy host array to device array
	cudaMemcpy(image_d, image, size, cudaMemcpyHostToDevice);

	cudaEventRecord(start); // Start timer
    contrast_device<<<min(NUM_BLOCKS, num_block), dim3(WARP, WARP)>>>(image_d, nMax_d, nMin_d, N);
	cudaEventRecord(stop); // Stop timer
	cudaEventSynchronize(stop);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("Time elapsed for %lu elements: %f ms\n", N, time);

	// Retrieve result from device and store it in host array
	cudaMemcpy(image, image_d, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(nMax, nMax_d, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(nMin, nMin_d, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("Min: %u\t Max: %u \n", *nMin, *nMax);
    // Write image array into a bmp file
    filename.replace(filename.find_last_of("."), filename.length(), "_out_gpu_single.bmp");
    stbi_write_bmp(filename.c_str(), width, height, 1, image);

    // Deallocate memory
    stbi_image_free(image);
	cudaFree(image_d);
    cudaFree(nMin_d);
    cudaFree(nMax_d);
    free(nMin);
    free(nMax);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
