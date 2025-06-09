// Do not alter the preprocessor directives
#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <cstdio>
#include <string>
#include <time.h>
#include "vector_math.h"

enum Alg{
    NEAREST,
    BILINEAR,
    BICUBIC,
    LANCZOS
};

inline void resizeImg(const uint8_t * const image, uint8_t * const scaled_image, const int& width, const int& height, const int& width_out, const int& height_out, const float& px, const float& py, const Alg& alg){
    const int bpp = 3;
    switch(alg){
        case NEAREST:
            {
                for(int i = 0; i < height_out; i++){
                    int src_y = clamp(((float)i)*py + 0.5f, height-1);
                    for(int j = 0; j < width_out; j++){
                        int src_x = clamp(((float)j)*px + 0.5f, width-1);
                        for(int k = 0; k < bpp; k++){
                            scaled_image[k + (j + i*width_out)*bpp] = image[k + (src_x + src_y*width)*bpp];
                        }
                    }
                }
            }
        return;
        case BILINEAR:
            {
                for(int i = 0; i < height_out; i++){
                    float y = ((float)i)*py;
                    int src_ly = y;
                    float dy = y - src_ly;
                    float dy_n = 1.f - dy;
                    for(int j = 0; j < width_out; j++){
                        float x = ((float)j)*px;
                        int src_lx = x;
                        float dx = x - src_lx;
                        for(int k = 0; k < bpp; k++){
                            uint8_t p00 = image[k + (src_lx + src_ly*width)*bpp];
                            uint8_t p01 = image[k + (clamp(src_lx+1, width-1) + src_ly*width)*bpp];
                            uint8_t p10 = image[k + (src_lx + clamp(src_ly+1, height-1)*width)*bpp];
                            uint8_t p11 = image[k + (clamp(src_lx+1, width-1) + clamp(src_ly+1, height-1)*width)*bpp];
                            scaled_image[k + (j + i*width_out)*bpp] = (1.f-dx)*(dy_n*p00 + dy*p10) + dy_n*dx*p01 + dx*dy*p11;
                        }
                    }
                }
            }
        return;
        case BICUBIC:
            {
                float b[4];
                for(int i = 0; i < height_out; i++){
                        float y = ((float)i)*py;
                        int src_ly = y;
                        float dy = y - src_ly;
                    for(int j = 0; j < width_out; j++){
                        float x = ((float)j)*px;
                        int src_lx = x;
                        float dx = x - src_lx;
                        for(int k = 0; k < bpp; k++){
                            for(int m = 0; m < 4; m++){
                                b[m] = cubic_interp<uint8_t, bpp>(dx, image + k + (clamp(src_lx-1, width-1) + clamp(src_ly+m-1, height-1)*width)*bpp);
                            }
                            scaled_image[k + (j + i*width_out)*bpp] = trunc2uint8(cubic_interp(dy, b));
                        }
                    }
                }
            }
        return;
        case LANCZOS:
            {
                float b[8];
                for(int i = 0; i < height_out; i++){
                    float y = ((float)i)*py;
                    int src_ly = y;
                    float dy = y - src_ly;
                    for(int j = 0; j < width_out; j++){
                        float x = ((float)j)*px;
                        int src_lx = x;
                        float dx = x - src_lx;
                        for(int k = 0; k < bpp; k++){
                            for(int m = 0; m < 8; m++){
                                b[m] = lancsoz4_interp<uint8_t,bpp>(dx, image + k + (clamp(src_lx-1, width-1) + clamp(src_ly+m-1, height-1)*width)*bpp);
                            }
                            scaled_image[k + (j + i*width_out)*bpp] = trunc2uint8(lancsoz4_interp(dy, b));
                        }
                    }
                }
            }
        return;
    }
}

inline void resizeImgA(const uchar4 * const image, uchar4 * const scaled_image, const int& width, const int& height, const int& width_out, const int& height_out, const float& px, const float& py, const Alg& alg){
    const int bpp = 4;
    switch(alg){
        case NEAREST:
            {
                for(int i = 0; i < height_out; i++){
                    int src_y = clamp(((float)i)*py + 0.5f, height-1);
                    for(int j = 0; j < width_out; j++){
                        int src_x = clamp(((float)j)*px + 0.5f, width-1);
                        scaled_image[j + i*width_out] = image[src_x + src_y*width];
                    }
                }
            }
        return;
        case BILINEAR:
            {
                for(int i = 0; i < height_out; i++){
                    float y = ((float)i)*py;
                    int src_ly = y;
                    float dy = y - src_ly;
                    float dy_n = 1.f - dy;
                    for(int j = 0; j < width_out; j++){
                        float x = ((float)j)*px;
                        int src_lx = x;
                        float dx = x - src_lx;
                        uchar4 p00 = image[src_lx + src_ly*width];
                        uchar4 p01 = image[clamp(src_lx+1, width-1) + src_ly*width];
                        uchar4 p10 = image[src_lx + clamp(src_ly+1, height-1)*width];
                        uchar4 p11 = image[clamp(src_lx+1, width-1) + clamp(src_ly+1, height-1)*width];
                        scaled_image[j + i*width_out] = cast2uchar4((1.f-dx)*(dy_n*p00 + dy*p10) + dy_n*dx*p01 + dx*dy*p11);
                    }
                }
            }
        return;
        case BICUBIC:
            {
                float4 b[4];
                for(int i = 0; i < height_out; i++){
                    float y = ((float)i)*py;
                    int src_ly = y;
                    float dy = y - src_ly;
                    for(int j = 0; j < width_out; j++){
                        float x = ((float)j)*px;
                        int src_lx = x;
                        float dx = x - src_lx;
                        for(int m = 0; m < 4; m++){
                            b[m] = cubic_interp4(dx, image + clamp(src_lx-1, width-1) + clamp(src_ly+m-1, height-1)*width);
                        }
                        scaled_image[j + i*width_out] = trunc2uchar4(cubic_interp4(dy, b));
                    }
                }
            }
        return;
        case LANCZOS:
            {
                float4 b[8];
                for(int i = 0; i < height_out; i++){
                    float y = ((float)i)*py;
                    int src_ly = y;
                    float dy = y - src_ly;
                    for(int j = 0; j < width_out; j++){
                        float x = ((float)j)*px;
                        int src_lx = x;
                        float dx = x - src_lx;
                        for(int m = 0; m < 8; m++){
                            b[m] = lancsoz4_interp4(dx, image + clamp(src_lx-1, width-1) + clamp(src_ly+m-1, height-1)*width);
                        }
                        scaled_image[j + i*width_out] = trunc2uchar4(lancsoz4_interp4(dy, b));
                    }
                }
            }
        return;
    }
}

int main(int argc, char* argv[]) {
    int width; //image width
    int height; //image height
    int bpp;  //bytes per pixel
    double fx = 2.0, fy = 2.0;
    Alg alg = BILINEAR;
    int alignment = 0;
    int show = 0;
    
    clock_t start, end;
    std::string filename("../samples/640x426.bmp");
    std::string algName("lin");
    if(argc > 1) filename = argv[1];
    if(argc > 2) fx = atof(argv[2]);
    if(argc > 3) fy = atof(argv[3]);
    else fy = fx;
    if(argc > 4){
        if(strcmp(argv[4], "nn") == 0){
            alg = NEAREST;
        }else if(strcmp(argv[4], "lin") == 0){ //default
            alg = BILINEAR;
        }else if(strcmp(argv[4], "cub") == 0){
            alg = BICUBIC;
        }else if(strcmp(argv[4], "lan") == 0){
            alg = LANCZOS;
        }else if(strcmp(argv[4], "nn_a") == 0){
            alignment = 1;
            alg = NEAREST;
        }else if(strcmp(argv[4], "lin_a") == 0){
            alignment = 1;
            alg = BILINEAR;
        }else if(strcmp(argv[4], "cub_a") == 0){
            alignment = 1;
            alg = BICUBIC;
        }else if(strcmp(argv[4], "lan_a") == 0){
            alignment = 1;
            alg = LANCZOS;
        }else{
            std::cout << "Alg name not supported. Possible values: nn, lin, cub, lan." << std::endl;
            exit(EXIT_FAILURE);
        }
        algName = argv[4];
    }
    if(argc > 5) show = atoi(argv[5]);

    cv::Mat image = cv::imread(filename.c_str());
    if (image.empty()) {
        std::cout << "Could not open the image!" << std::endl;
        exit(EXIT_FAILURE);
    }
    bpp = image.channels();
    width = image.cols;
    height = image.rows;
    float px = 1.f/fx;
    float py = 1.f/fy;
    // Print for sanity check
    printf("Bytes per pixel: %d \n", bpp);
    printf("Height: %d \n", height);
    printf("Width: %d \n", width);
    if(alignment){
        bpp = 4;
        cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);
    }

    int width_out = lroundf(fx*width), height_out = lroundf(fy*height);
    size_t in_size = width*height*bpp*sizeof(uint8_t);
    size_t out_size = width_out*height_out*bpp*sizeof(uint8_t);
	uint8_t *image_out = (uint8_t *) malloc(out_size);

    start = clock();

    if(alignment){
        resizeImgA((uchar4 *)image.data, (uchar4 *)image_out, width, height, width_out, height_out, px, py, alg);
    }else{
        resizeImg(image.data, image_out, width, height, width_out, height_out, px, py, alg);
    }

    end = clock();
    // Write image array into a bmp file
    cv::Mat scaled_image(height_out, width_out, alignment ? CV_8UC4 : CV_8UC3, image_out);
    
    char buff[50];
    sprintf(buff, "_cpu_%s_out", algName.c_str());
    filename.insert(filename.find_last_of("."), buff);
    cv::imwrite(filename.c_str(), scaled_image);
    printf("Time taken: %f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000.);

    if(show){
        cv::imshow("Result Image", scaled_image);
        cv::waitKey(0);
    }
    return 0;
}
