// Do not alter the preprocessor directives
#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <cstdio>
#include <string>
#include <time.h>
#include <cuda.h>
#include <cmath>
#include "vector_math.cuh"

#define BLOCK_X 16
#define BLOCK_Y 16

// 0.071 ms
__global__ void resize_nn_device(const cudaTextureObject_t src_tex, uint8_t *dest_img, int width, int height, int width_out, int height_out, float px, float py){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int BX = blockDim.x;
    int BY = blockDim.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int dest_x = tx + bx*BX;
    int dest_y = ty + by*BY;

    if((dest_x < width_out)&&(dest_y < height_out)){
        uchar4 p = tex2D<uchar4>(src_tex, (px*dest_x), (py*dest_y));
        ((uchar4*)dest_img)[dest_x + dest_y*width_out] = p;
    }
}

// 0.075 ms
__global__ void resize_nn_v2_device(const cudaTextureObject_t src_tex, uchar4 *dest_img, int width, int height, int width_out, int height_out, float fx, float fy){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int BX = blockDim.x;
    int BY = blockDim.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int src_x = tx + bx*BX;
    int src_y = ty + by*BY;

    if((src_x < width)&&(src_y < height)){
        uchar4 p = tex2D<uchar4>(src_tex, src_x, src_y);

        int l_x = ceilf(fx*src_x);
        int l_y = ceilf(fy*src_y);
        int u_x = ceilf(fx*(src_x+1));
        int u_y = ceilf(fy*(src_y+1));
        for(int i = l_y; i < u_y; i++){
            for(int j = l_x; j < u_x; j++){
                dest_img[clamp(j, width_out-1) + clamp(i, height_out-1)*width_out] = p;
            }
        }
    }
}

//0.072 ms
__global__ void resize_lin_device(const cudaTextureObject_t src_tex, uint8_t *dest_img, int width, int height, int width_out, int height_out, float px, float py){    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int BX = blockDim.x;
    int BY = blockDim.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int dest_x = tx + bx*BX;
    int dest_y = ty + by*BY;

    if((dest_x < width_out)&&(dest_y < height_out)){
        float4 texel = tex2D<float4>(src_tex, (px*dest_x), (py*dest_y));
        uchar4 p;
        p.x = texel.x*255;
        p.y = texel.y*255;
        p.z = texel.z*255;
        p.w = 255;
        ((uchar4*)dest_img)[dest_x + dest_y*width_out] = p;
    }
}

// 0.259 ms
__global__ void resize_cub_device(const cudaTextureObject_t src_tex, uint8_t *dest_img, int width, int height, int width_out, int height_out, float px, float py){    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int BX = blockDim.x;
    int BY = blockDim.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int dest_x = tx + bx*BX;
    int dest_y = ty + by*BY;

    if((dest_x < width_out)&&(dest_y < height_out)){
        float x = px*dest_x;
        float y = py*dest_y;
        int src_lx = (int) x; // floor
        int src_ly = (int) y; // floor

        float4 pixels[4], b[4];
        float dx = x - src_lx;
        for(uint8_t i = 0; i < 4; i++){
            for(uint8_t j = 0; j < 4; j++){
                pixels[j] = tex2D<float4>(src_tex, x + j - 1.f, y + i - 1.f);
            }
            b[i] = cubic_interp4(dx, pixels);
        }

        float dy = y - src_ly;
        ((uchar4*)dest_img)[dest_x + dest_y*width_out] = convert2uchar4(cubic_interp4(dy, b));
    }
}

// 0.119 ms
__global__ void resize_cub_v2_device(const cudaTextureObject_t src_tex, uchar4 *dest_img, int width, int height, int width_out, int height_out, float fx, float fy){    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int BX = blockDim.x;
    int BY = blockDim.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int src_x = tx + bx*BX;
    int src_y = ty + by*BY;

    if((src_x < width)&&(src_y < height)){
        float4 pixels[4][4];

        for(uint8_t i = 0; i < 4; i++){
            for(uint8_t j = 0; j < 4; j++){
                pixels[i][j] = tex2D<float4>(src_tex, src_x + j - 0.5f, src_y + i - 0.5f);
            }
        }

        int l_x = ceilf(fx*src_x);
        int l_y = ceilf(fy*src_y);
        int u_x = ceilf(fx*(src_x+1));
        int u_y = ceilf(fy*(src_y+1));
        for(int i = l_y; i < u_y; i++){
            for(int j = l_x; j < u_x; j++){
                float x = fdividef(j,fx);
                float y = fdividef(i,fy);
                float dx = x - src_x;
                float dy = y - src_y;
                float4 b[4];
                for(int k = 0; k < 4; k++){
                    b[k] = cubic_interp4(dx, pixels[k]);
                }
                dest_img[clamp(j, width_out-1) + clamp(i, height_out-1)*width_out] = convert2uchar4(cubic_interp4(dy, b));
            }
        }
    }
}

// 0.966 ms
// Lancsoz resampling with a=4 window (8x8 kernel)
__global__ void resize_lan_device(const cudaTextureObject_t src_tex, uint8_t *dest_img, int width, int height, int width_out, int height_out, float px, float py){    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int BX = blockDim.x;
    int BY = blockDim.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int dest_x = tx + bx*BX;
    int dest_y = ty + by*BY;

    if((dest_x < width_out)&&(dest_y < height_out)){
        float4 pixels[8];
        float4 results[8];

        float x = px*dest_x;
        float y = py*dest_y;
        int src_lx = (int) x; // floor
        int src_ly = (int) y; // floor

        float dx = x - src_lx;
        for(uint8_t i = 0; i < 8; i++){
            for(uint8_t j = 0; j < 8; j++){
                pixels[j] = tex2D<float4>(src_tex, x - 2.5f + j, y - 2.5f + i);
            }
            results[i] = lancsoz4_interp4(dx, pixels);
        }

        float dy = y - src_ly;
        ((uchar4*)dest_img)[dest_x + dest_y*width_out] = convert2uchar4(lancsoz4_interp4(dy, results));
    }
}

// 0.907 ms
// With shared memory
template<int BX, int BY>
__global__ void resize_lan_v2_device(const cudaTextureObject_t src_tex, uint8_t *dest_img, int width, int height, int width_out, int height_out, float px, float py){    
    extern __shared__ float4 pixels[];
    float4 results[8];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int pixels_min_x = px*(bx*BX) - 3; // tx = 0
    int pixels_min_y = py*(by*BY) - 3; // ty pixels_min_x= 0
    int pixels_max_x = px*((bx+1)*BX - 1) + 5; // tx = BX - 1
    int pixels_max_y = py*((by+1)*BY - 1) + 5; // ty = BY - 1
    int pixels_width = pixels_max_x - pixels_min_x + 1;
    int pixels_height = pixels_max_y - pixels_min_y + 1;

    for(int i = ty; i < pixels_height; i += BY){
        for (int j = tx; j < pixels_width; j += BX){
            pixels[i*pixels_width + j] = tex2D<float4>(src_tex, pixels_min_x + j, pixels_min_y + i);
        }
    }

    __syncthreads();

    int dest_x = tx + bx*BX;
    int dest_y = ty + by*BY;

    if((dest_x >= width_out)||(dest_y >= height_out)) return;
    
    float x = px*dest_x;
    float y = py*dest_y;
    int src_lx = (int) x; // floor
    int src_ly = (int) y; // floor
    int p_x = x - pixels_min_x - 3;
    int p_y = y - pixels_min_y - 3;

    float dx = x - src_lx;
    for(int i = 0; i < 8; i++){
        results[i] = lancsoz4_interp4(dx, pixels + (p_y + i)*pixels_width + p_x);
    }

    float dy = y - src_ly;
    ((uchar4*)dest_img)[dest_x + dest_y*width_out] = convert2uchar4(lancsoz4_interp4(dy, results));
}

// 1.130 ms
__global__ void resize_lan_v3_device(const cudaTextureObject_t src_tex, uchar4 *dest_img, int width, int height, int width_out, int height_out, float fx, float fy){    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int BX = blockDim.x;
    int BY = blockDim.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int src_x = tx + bx*BX;
    int src_y = ty + by*BY;

    if((src_x < width)&&(src_y < height)){
        float4 pixels[8][8];
        for(int i = 0; i < 8; i++){
            for(int j = 0; j < 8; j++){
                pixels[i][j] = tex2D<float4>(src_tex, src_x + j - 3.f, src_y + i - 3.f);
            }
        }

        int l_x = ceilf(fx*src_x);
        int l_y = ceilf(fy*src_y);
        int u_x = ceilf(fx*(src_x+1));
        int u_y = ceilf(fy*(src_y+1));

        for(int i = l_y; i < u_y; i++){
            for(int j = l_x; j < u_x; j++){
                float x = fdividef(j,fx);
                float y = fdividef(i,fy);
                float dx = x - src_x;
                float dy = y - src_y;
                float4 b[8];
                for(int k = 0; k < 8; k++){
                    b[k] = lancsoz4_interp4(dx, pixels[k]);
                }
                dest_img[clamp(j, width_out-1) + clamp(i, height_out-1)*width_out] = convert2uchar4(lancsoz4_interp4(dy, b));
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int width, width_out; //image width
    int height, height_out; //image height
    int bpp;  //bytes per pixel
    float fx = 2.0, fy = 2.0, px, py;
    int alg = 1;
    int show = 0;
    dim3 BLOCK_SIZE(BLOCK_X, BLOCK_Y), GRID_SIZE;
    
    // Create timer events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    std::string filename("../samples/640x426.bmp");
    std::string algName("lin");
    if(argc > 1) filename = argv[1];
    if(argc > 2) fx = atof(argv[2]);
    if(argc > 3) fy = atof(argv[3]);
    else fy = fx;
    if(argc > 4){
        if(strcmp(argv[4], "nn") == 0){
            alg = 0;
        }else if(strcmp(argv[4], "lin") == 0){
            alg = 1;
        }else if(strcmp(argv[4], "cub") == 0){
            alg = 2;
        }else if(strcmp(argv[4], "lan") == 0){
            alg = 3;
        }else if(strcmp(argv[4], "lan_v2") == 0){
            alg = 4;
        }else if(strcmp(argv[4], "nn_v2") == 0){
            alg = 5;
        }else if(strcmp(argv[4], "cub_v2") == 0){
            alg = 6;
        }else if(strcmp(argv[4], "lan_v3") == 0){
            alg = 7;
        }else{
            std::cout << "Alg name not supported. Possible values: nn, nn_v2, lin, cub, cub_v2, lan, lan_v2, lan_v3." << std::endl;
            exit(EXIT_FAILURE);
        }
        algName = argv[4];
    }
    if(argc > 5) show = atoi(argv[5]);

    cv::Mat cv_image = cv::imread(filename.c_str());
    if (cv_image.empty()) {
        std::cout << "Could not open the image!" << std::endl;
        exit(EXIT_FAILURE);
    }
    bpp = cv_image.channels();
    width = cv_image.cols;
    height = cv_image.rows;
    width_out = lroundf(fx*width);
    height_out = lroundf(fy*height);
    px = 1./fx;
    py = 1./fy;
    cv::Mat cv_image_a;
    cv::cvtColor(cv_image, cv_image_a, cv::COLOR_BGR2BGRA);
    
    uint8_t *d_image_out, *image_out;
    //size_t in_size = width*height*bpp*sizeof(uint8_t);
    size_t out_size = width_out*height_out*4*sizeof(uint8_t);

	cudaMalloc((void **)&d_image_out, out_size);
	image_out = (uint8_t *) malloc(out_size);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    cudaMemcpy2DToArray(
        cuArray, 0, 0, cv_image_a.data,
        width * sizeof(uchar4),
        width * sizeof(uchar4), height,
        cudaMemcpyHostToDevice
    );

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;

    switch(alg){
        case 0:
            texDesc.filterMode = cudaFilterModePoint;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;
            break;
        case 1:
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeNormalizedFloat;
            texDesc.normalizedCoords = 0;
            break;
        case 5:
            texDesc.filterMode = cudaFilterModePoint;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;
            break;
        default:
            texDesc.filterMode = cudaFilterModePoint;
            texDesc.readMode = cudaReadModeNormalizedFloat;
            texDesc.normalizedCoords = 0;
            break;
    }

    cudaTextureObject_t texObj;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    // Print for sanity check
    printf("Bytes per pixel: %d \n", bpp);
    printf("Height: %d \n", height);
    printf("Width: %d \n", width);

    if(alg >= 5){
        GRID_SIZE = {(width + (BLOCK_SIZE.x - 1))/BLOCK_SIZE.x, (height + (BLOCK_SIZE.y - 1))/BLOCK_SIZE.y};
    }else{
        GRID_SIZE = {(width_out + (BLOCK_SIZE.x - 1))/BLOCK_SIZE.x, (height_out + (BLOCK_SIZE.y - 1))/BLOCK_SIZE.y};
    }

    int dyn_smem_size = (((int)(px*BLOCK_SIZE.x)) + 7) * (((int)(py*BLOCK_SIZE.y)) + 7) * sizeof(float4);

    cudaEventRecord(start); // Start timer
    switch(alg){
        case 0:
            resize_nn_device<<<GRID_SIZE, BLOCK_SIZE>>>(texObj, d_image_out, width, height, width_out, height_out, px, py);
            break;
        case 1:
            resize_lin_device<<<GRID_SIZE, BLOCK_SIZE>>>(texObj, d_image_out, width, height, width_out, height_out, px, py);
            break;
        case 2:
            resize_cub_device<<<GRID_SIZE, BLOCK_SIZE>>>(texObj, d_image_out, width, height, width_out, height_out, px, py);
            break;
        case 3:
            resize_lan_device<<<GRID_SIZE, BLOCK_SIZE>>>(texObj, d_image_out, width, height, width_out, height_out, px, py);
            break;
        case 4:
            resize_lan_v2_device<BLOCK_X, BLOCK_Y><<<GRID_SIZE, BLOCK_SIZE, dyn_smem_size>>>(texObj, d_image_out, width, height, width_out, height_out, px, py);
            break;
        case 5:
            resize_nn_v2_device<<<GRID_SIZE, BLOCK_SIZE>>>(texObj, (uchar4*)d_image_out, width, height, width_out, height_out, fx, fy);
            break;
        case 6:
            resize_cub_v2_device<<<GRID_SIZE, BLOCK_SIZE>>>(texObj, (uchar4*)d_image_out, width, height, width_out, height_out, fx, fy);
            break;
        case 7:
            resize_lan_v3_device<<<GRID_SIZE, BLOCK_SIZE>>>(texObj, (uchar4*)d_image_out, width, height, width_out, height_out, fx, fy);
            break;
    }
    cudaEventRecord(stop); // Stop timer
	cudaEventSynchronize(stop);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(image_out, d_image_out, out_size, cudaMemcpyDeviceToHost);

    cv::Mat scaled_image_a = cv::Mat(height_out, width_out, CV_8UC4, image_out);
    cv::Mat scaled_image;
    cv::cvtColor(scaled_image_a, scaled_image, cv::COLOR_BGRA2BGR);

    // Write image array into a bmp file
    char buff[50];
    sprintf(buff, "_gpu_tex_%s_out", algName.c_str());
    filename.insert(filename.find_last_of("."), buff);
    cv::imwrite(filename.c_str(), scaled_image);

    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("Time taken: %f ms\n", time);

    if(show){
        cv::imshow("Result Image", scaled_image);
        cv::waitKey(0);
    }

	cudaFree(d_image_out);
    free(image_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
