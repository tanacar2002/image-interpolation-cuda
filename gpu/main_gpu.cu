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

// 0.113 ms
__global__ void resize_nn_device(const uint8_t *src_img, uint8_t *dest_img, int width, int height, int width_out, int height_out, float px, float py){
    int tx = threadIdx.x; // Channel
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int BX = blockDim.x; // bpp
    int BY = blockDim.y;
    int BZ = blockDim.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int dest_x = ty + bx*BY;
    int dest_y = tz + by*BZ;

    if((dest_x < width_out)&&(dest_y < height_out)){
        // Using lroundf was ~0.050 ms slower, since we are only working with positive numbers, we can just cast to int
        int src_x = (int)((px*dest_x) + 0.5f); // lroundf(px*dest_x);
        int src_y = (int)((py*dest_y) + 0.5f); // lroundf(py*dest_y);

        uint8_t p = src_img[tx + src_x*BX + src_y*BX*width];

        dest_img[tx + dest_x*BX + dest_y*BX*width_out] = p;
    }
}

// 0.082 ms
__global__ void resize_nn_v2_device(const uint8_t *src_img, uint8_t *dest_img, int width, int height, int width_out, int height_out, float fx, float fy){
    int tx = threadIdx.x; // Channel
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int BX = blockDim.x; // bpp
    int BY = blockDim.y;
    int BZ = blockDim.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int src_x = ty + bx*BY;
    int src_y = tz + by*BZ;

    if((src_x < width)&&(src_y < height)){
        uint8_t p = src_img[tx + src_x*BX + src_y*BX*width];

        int l_x = ceilf(fx*src_x);
        int l_y = ceilf(fy*src_y);
        int u_x = ceilf(fx*(src_x+1));
        int u_y = ceilf(fy*(src_y+1));
        for(int i = l_y; i < u_y; i++){
            for(int j = l_x; j < u_x; j++){
                dest_img[tx + clamp(j, width_out-1)*BX + clamp(i, height_out-1)*BX*width_out] = p;
            }
        }
    }
}

// 0.072 ms
__global__ void resize_nn_v3_device(const uchar4 *src_img, uchar4 *dest_img, int width, int height, int width_out, int height_out, float fx, float fy){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int BX = blockDim.x;
    int BY = blockDim.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int src_x = tx + bx*BX;
    int src_y = ty + by*BY;

    if((src_x < width)&&(src_y < height)){
        uchar4 p = src_img[src_x + src_y*width];

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

// 0.155 ms
__global__ void resize_lin_device(const uint8_t *src_img, uint8_t *dest_img, int width, int height, int width_out, int height_out, float px, float py){    
    int tx = threadIdx.x; // Channel
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int BX = blockDim.x; // bpp
    int BY = blockDim.y;
    int BZ = blockDim.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int dest_x = ty + bx*BY;
    int dest_y = tz + by*BZ;

    if((dest_x < width_out)&&(dest_y < height_out)){
        float x = px*dest_x;
        float y = py*dest_y;
        int src_lx = (int) x; // floor
        int src_ly = (int) y; // floor
        float dx = x - src_lx;
        float dy = y - src_ly;

        uint8_t p00 = src_img[tx + src_lx*BX + src_ly*BX*width];
        uint8_t p01 = src_img[tx + clamp(src_lx+1, width-1)*BX + src_ly*BX*width];
        uint8_t p10 = src_img[tx + src_lx*BX + clamp(src_ly+1, height-1)*BX*width];
        uint8_t p11 = src_img[tx + clamp(src_lx+1, width-1)*BX + clamp(src_ly+1, height-1)*BX*width];

        dest_img[tx + dest_x*BX + dest_y*BX*width_out] = (uint8_t)((1.f-dx)*(1.f-dy)*p00 + dx*(1.f-dy)*p01 + (1.f-dx)*dy*p10 + dx*dy*p11);
    }
}

// 0.092 ms
__global__ void resize_lin_v2_device(const uchar4 *src_img, uchar4 *dest_img, int width, int height, int width_out, int height_out, float fx, float fy){    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int BX = blockDim.x;
    int BY = blockDim.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int src_x = tx + bx*BX;
    int src_y = ty + by*BY;

    if((src_x < width)&&(src_y < height)){
        uchar4 p00 = src_img[src_x + src_y*width];
        uchar4 p01 = src_img[clamp(src_x+1, width-1) + src_y*width];
        uchar4 p10 = src_img[src_x + clamp(src_y+1, height-1)*width];
        uchar4 p11 = src_img[clamp(src_x+1, width-1) + clamp(src_y+1, height-1)*width];

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
                dest_img[clamp(j, width_out-1) + clamp(i, height_out-1)*width_out] = trunc2uchar4((1.f-dx)*(1.f-dy)*p00 + dx*(1.f-dy)*p01 + (1.f-dx)*dy*p10 + dx*dy*p11);
            }
        }
    }
}

// 0.455 ms
__global__ void resize_cub_device(const uint8_t *src_img, uint8_t *dest_img, int width, int height, int width_out, int height_out, float px, float py){    
    int tx = threadIdx.x; // Channel
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int BX = blockDim.x; // bpp
    int BY = blockDim.y;
    int BZ = blockDim.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int dest_x = ty + bx*BY;
    int dest_y = tz + by*BZ;

    if((dest_x < width_out)&&(dest_y < height_out)){
        float x = px*dest_x;
        float y = py*dest_y;
        int src_lx = (int) x; // floor
        int src_ly = (int) y; // floor

        uint8_t p00 = src_img[tx + src_lx*BX + src_ly*BX*width];
        uint8_t p0n1 = (src_lx == 0) ? p00 : src_img[tx + (src_lx-1)*BX + src_ly*BX*width];
        uint8_t p01 = (src_lx == width) ? p00 : src_img[tx + (src_lx+1)*BX + src_ly*BX*width];
        uint8_t p02 = (src_lx >= width-1) ? p01 : src_img[tx + (src_lx+2)*BX + src_ly*BX*width];

        uint8_t pn1n1 = (src_lx == 0 || src_ly == 0) ? p0n1 : src_img[tx + (src_lx-1)*BX + (src_ly-1)*BX*width];
        uint8_t pn10 = (src_ly == 0) ? p00 : src_img[tx + src_lx*BX + (src_ly-1)*BX*width];
        uint8_t pn11 = (src_lx == width || src_ly == 0) ? p01 : src_img[tx + (src_lx+1)*BX + (src_ly-1)*BX*width];
        uint8_t pn12 = (src_lx >= width-1 || src_ly == 0) ? p02 : src_img[tx + (src_lx+2)*BX + (src_ly-1)*BX*width];

        uint8_t p1n1 = (src_lx == 0 || src_ly == height) ? p0n1 : src_img[tx + (src_lx-1)*BX + (src_ly+1)*BX*width];
        uint8_t p10 = (src_ly == height) ? p00 : src_img[tx + src_lx*BX + (src_ly+1)*BX*width];
        uint8_t p11 = (src_lx == width || src_ly == height) ? p01 : src_img[tx + (src_lx+1)*BX + (src_ly+1)*BX*width];
        uint8_t p12 = (src_lx >= width-1 || src_ly == height) ? p02 : src_img[tx + (src_lx+2)*BX + (src_ly+1)*BX*width];

        uint8_t p2n1 = (src_lx == 0 || src_ly >= height-1) ? p1n1 : src_img[tx + (src_lx-1)*BX + (src_ly+2)*BX*width];
        uint8_t p20 = (src_ly >= height-1) ? p10 : src_img[tx + src_lx*BX + (src_ly+2)*BX*width];
        uint8_t p21 = (src_lx == width || src_ly >= height-1) ? p11 : src_img[tx + (src_lx+1)*BX + (src_ly+2)*BX*width];
        uint8_t p22 = (src_lx >= width-1 || src_ly >= height-1) ? p12 : src_img[tx + (src_lx+2)*BX + (src_ly+2)*BX*width];

        float dx = x - src_lx;
        float bn1 = cubic_interp(dx, pn1n1, pn10, pn11, pn12);
        float b0 = cubic_interp(dx, p0n1, p00, p01, p02);
        float b1 = cubic_interp(dx, p1n1, p10, p11, p12);
        float b2 = cubic_interp(dx, p2n1, p20, p21, p22);

        float dy = y - src_ly;
        dest_img[tx + dest_x*BX + dest_y*BX*width_out] = trunc2uint8(cubic_interp(dy, bn1, b0, b1, b2));
    }
}

// 0.167 ms
__global__ void resize_cub_v2_device(const uchar4 *src_img, uchar4 *dest_img, int width, int height, int width_out, int height_out, float fx, float fy){    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int BX = blockDim.x;
    int BY = blockDim.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int src_x = tx + bx*BX;
    int src_y = ty + by*BY;

    if((src_x < width)&&(src_y < height)){
        uchar4 pixels[4][4];
        for(int i = 0; i < 4; i++){
            for(int j = 0; j < 4; j++){
                pixels[i][j] = src_img[clamp(src_x+j-1, width-1) + clamp(src_y+i-1, height-1)*width];
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
                dest_img[clamp(j, width_out-1) + clamp(i, height_out-1)*width_out] = trunc2uchar4(cubic_interp4(dy, b));
            }
        }
    }
}

// 2.960 ms
__global__ void resize_lan_device(const uint8_t *src_img, uint8_t *dest_img, int width, int height, int width_out, int height_out, float px, float py){    
    int tx = threadIdx.x; // Channel
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int BX = blockDim.x; // bpp
    int BY = blockDim.y;
    int BZ = blockDim.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int dest_x = ty + bx*BY;
    int dest_y = tz + by*BZ;

    if((dest_x < width_out)&&(dest_y < height_out)){
        uint8_t pixels[8];
        float b[8];

        float x = px*dest_x;
        float y = py*dest_y;
        int src_lx = (int) x; // floor x
        int src_ly = (int) y; // floor y
        float dx = x - src_lx;
        float dy = y - src_ly;

        for(int i = 0; i < 8; i++){
            for(int j = 0; j < 8; j++){
                pixels[j] = src_img[tx + clamp(src_lx+j-3, width-1)*BX + clamp(src_ly+i-3, height-1)*BX*width];
            }
            b[i] = lancsoz4_interp(dx, pixels);
        }

        dest_img[tx + dest_x*BX + dest_y*BX*width_out] = trunc2uint8(lancsoz4_interp(dy, b));
    }
}

// 1.514 ms
__global__ void resize_lan_v2_device(const uchar4 *src_img, uchar4 *dest_img, int width, int height, int width_out, int height_out, float px, float py){    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int BX = blockDim.x;
    int BY = blockDim.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int dest_x = tx + bx*BX;
    int dest_y = ty + by*BY;

    if((dest_x < width_out)&&(dest_y < height_out)){
        uchar4 pixels[8];
        float4 b[8];

        float x = px*dest_x;
        float y = py*dest_y;
        int src_lx = lroundf(x); // floor
        int src_ly = lroundf(y); // floor
        float dx = x - src_lx;
        float dy = y - src_ly;

        for(int i = 0; i < 8; i++){
            for(int j = 0; j < 8; j++){
                pixels[j] = src_img[clamp(src_lx+j-3, width-1) + clamp(src_ly+i-3, height-1)*width];
            }
            b[i] = lancsoz4_interp4(dx, pixels);
        }

        dest_img[dest_x + dest_y*width_out] = trunc2uchar4(lancsoz4_interp4(dy, b));
    }
}

// With shared memory
// 1.487 ms
template<int BX, int BY>
__global__ void resize_lan_v3_device(const uchar4 *src_img, uchar4 *dest_img, int width, int height, int width_out, int height_out, float px, float py){    
    extern __shared__ uchar4 pixels[];
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
            pixels[i*pixels_width + j] = src_img[clamp(pixels_min_x + j, width-1) + clamp(pixels_min_y + i, height-1)*width];
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
    dest_img[dest_x + dest_y*width_out] = trunc2uchar4(lancsoz4_interp4(dy, results));
}

// 1.195 ms
__global__ void resize_lan_v4_device(const uchar4 *src_img, uchar4 *dest_img, int width, int height, int width_out, int height_out, float fx, float fy){    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int BX = blockDim.x;
    int BY = blockDim.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int src_x = tx + bx*BX;
    int src_y = ty + by*BY;

    if((src_x < width)&&(src_y < height)){
        uchar4 pixels[8][8];
        for(int i = 0; i < 8; i++){
            for(int j = 0; j < 8; j++){
                pixels[i][j] = src_img[clamp(src_x+j-3, width-1) + clamp(src_y+i-3, height-1)*width];
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
                dest_img[clamp(j, width_out-1) + clamp(i, height_out-1)*width_out] = trunc2uchar4(lancsoz4_interp4(dy, b));
            }
        }
    }
}

// With shared memory
// 1.403 ms
template<int BX, int BY>
__global__ void resize_lan_v5_device(const uchar4 *src_img, uchar4 *dest_img, int width, int height, int width_out, int height_out, float fx, float fy){    
    __shared__ uchar4 pixels[BY + 8][BX + 8];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    for(int i = ty; i < BY + 8; i += BY){
        for(int j = tx; j < BX + 8; j += BX){
            pixels[i][j] = src_img[clamp(bx*BX+j-3, width-1) + clamp(by*BY+i-3, height-1)*width];
        }
    }

    __syncthreads();

    int src_x = tx + bx*BX;
    int src_y = ty + by*BY;

    if((src_x < width)&&(src_y < height)){
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
                    b[k] = lancsoz4_interp4(dx, pixels[ty + k] + tx);
                }
                dest_img[clamp(j, width_out-1) + clamp(i, height_out-1)*width_out] = trunc2uchar4(lancsoz4_interp4(dy, b));
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int width, width_out; //image width
    int height, height_out; //image height
    unsigned int bpp;  //bytes per pixel
    uint8_t *d_image, *d_image_out, *image_out;
    float fx = 2.0, fy = 2.0, px, py;
    int alg = 1;
    int show = 0;
    dim3 BLOCK_SIZE, GRID_SIZE;
    
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
        }else if(strcmp(argv[4], "nn_v2") == 0){
            alg = 4;
        }else if(strcmp(argv[4], "lan_v2") == 0){
            alg = 5;
        }else if(strcmp(argv[4], "lan_v3") == 0){
            alg = 6;
        }else if(strcmp(argv[4], "nn_v3") == 0){
            alg = 7;
        }else if(strcmp(argv[4], "lin_v2") == 0){
            alg = 8;
        }else if(strcmp(argv[4], "cub_v2") == 0){
            alg = 9;
        }else if(strcmp(argv[4], "lan_v4") == 0){
            alg = 10;
        }else if(strcmp(argv[4], "lan_v5") == 0){
            alg = 11;
        }else{
            std::cout << "Alg name not supported. Possible: nn, nn_v2, nn_v3, lin, lin_v2, cub, cub_v2, lan, lan_v2, lan_v3, lan_v4, lan_v5." << std::endl;
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

    if(alg >= 7){
        bpp = 4;
        cv::cvtColor(cv_image, cv_image, cv::COLOR_BGR2BGRA);
        BLOCK_SIZE = {BLOCK_X, BLOCK_Y};
        GRID_SIZE = {(width + (BLOCK_SIZE.x - 1))/BLOCK_SIZE.x, (height + (BLOCK_SIZE.y - 1))/BLOCK_SIZE.y};
    }else if(alg >= 5){
        bpp = 4;
        cv::cvtColor(cv_image, cv_image, cv::COLOR_BGR2BGRA);
        BLOCK_SIZE = {BLOCK_X, BLOCK_Y};
        GRID_SIZE = {(width_out + (BLOCK_SIZE.x - 1))/BLOCK_SIZE.x, (height_out + (BLOCK_SIZE.y - 1))/BLOCK_SIZE.y};
    }else if(alg == 4){
        BLOCK_SIZE = {bpp, BLOCK_X, BLOCK_Y};
        GRID_SIZE = {(width + (BLOCK_SIZE.y - 1))/BLOCK_SIZE.y, (height + (BLOCK_SIZE.z - 1))/BLOCK_SIZE.z};
    }else{
        BLOCK_SIZE = {bpp, BLOCK_X, BLOCK_Y};
        GRID_SIZE = {(width_out + (BLOCK_SIZE.y - 1))/BLOCK_SIZE.y, (height_out + (BLOCK_SIZE.z - 1))/BLOCK_SIZE.z};
    }
    size_t in_size = width*height*bpp*sizeof(uint8_t);
    size_t out_size = width_out*height_out*bpp*sizeof(uint8_t);
	cudaMalloc((void **)&d_image, in_size);
	cudaMalloc((void **)&d_image_out, out_size);
	image_out = (uint8_t *) malloc(out_size);

    cudaMemcpy(d_image, cv_image.data, in_size, cudaMemcpyHostToDevice);
    // Print for sanity check
    printf("Bytes per pixel: %d \n", bpp);
    printf("Height: %d \n", height);
    printf("Width: %d \n", width);

    int dyn_smem_size = (((int)(px*BLOCK_SIZE.x)) + 7) * (((int)(py*BLOCK_SIZE.y)) + 7) * sizeof(uchar4);

    cudaEventRecord(start); // Start timer
    switch(alg){
        case 0:
            resize_nn_device<<<GRID_SIZE, BLOCK_SIZE>>>(d_image, d_image_out, width, height, width_out, height_out, px, py);
            break;
        case 1:
            resize_lin_device<<<GRID_SIZE, BLOCK_SIZE>>>(d_image, d_image_out, width, height, width_out, height_out, px, py);
            break;
        case 2:
            resize_cub_device<<<GRID_SIZE, BLOCK_SIZE>>>(d_image, d_image_out, width, height, width_out, height_out, px, py);
            break;
        case 3:
            resize_lan_device<<<GRID_SIZE, BLOCK_SIZE>>>(d_image, d_image_out, width, height, width_out, height_out, px, py);
            break;
        case 4:
            resize_nn_v2_device<<<GRID_SIZE, BLOCK_SIZE>>>(d_image, d_image_out, width, height, width_out, height_out, fx, fy);
            break;
        case 5:
            resize_lan_v2_device<<<GRID_SIZE, BLOCK_SIZE>>>((uchar4*)d_image, (uchar4*)d_image_out, width, height, width_out, height_out, px, py);
            break;
        case 6:
            resize_lan_v3_device<BLOCK_X, BLOCK_Y><<<GRID_SIZE, BLOCK_SIZE, dyn_smem_size>>>((uchar4*)d_image, (uchar4*)d_image_out, width, height, width_out, height_out, px, py);
            break;
        case 7:
            resize_nn_v3_device<<<GRID_SIZE, BLOCK_SIZE>>>((uchar4*)d_image, (uchar4*)d_image_out, width, height, width_out, height_out, fx, fy);
            break;
        case 8:
            resize_lin_v2_device<<<GRID_SIZE, BLOCK_SIZE>>>((uchar4*)d_image, (uchar4*)d_image_out, width, height, width_out, height_out, fx, fy);
            break;
        case 9:
            resize_cub_v2_device<<<GRID_SIZE, BLOCK_SIZE>>>((uchar4*)d_image, (uchar4*)d_image_out, width, height, width_out, height_out, fx, fy);
            break;
        case 10:
            resize_lan_v4_device<<<GRID_SIZE, BLOCK_SIZE>>>((uchar4*)d_image, (uchar4*)d_image_out, width, height, width_out, height_out, fx, fy);
            break;
        case 11:
            resize_lan_v5_device<BLOCK_X, BLOCK_Y><<<GRID_SIZE, BLOCK_SIZE>>>((uchar4*)d_image, (uchar4*)d_image_out, width, height, width_out, height_out, fx, fy);
            break;
    }
    cudaEventRecord(stop); // Stop timer
	cudaEventSynchronize(stop);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(image_out, d_image_out, out_size, cudaMemcpyDeviceToHost);
    
    cv::Mat scaled_image;
    if(alg >= 5){
        scaled_image = cv::Mat(height_out, width_out, CV_8UC4, image_out);
        cv::cvtColor(scaled_image, scaled_image, cv::COLOR_BGRA2BGR);
    }else{
        scaled_image = cv::Mat(height_out, width_out, CV_8UC3, image_out);
    }

    // Write image array into a bmp file
    char buff[50];
    sprintf(buff, "_gpu_%s_out", algName.c_str());
    filename.insert(filename.find_last_of("."), buff);
    cv::imwrite(filename.c_str(), scaled_image);

    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("Time taken: %f ms\n", time);

    if(show){
        cv::imshow("Result Image", scaled_image);
        cv::waitKey(0);
    }

	cudaFree(d_image);
	cudaFree(d_image_out);
    free(image_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
