// Do not alter the preprocessor directives
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <stdint.h>
#include <cstdio>
#include <string>
#include <time.h>

#define NUM_CHANNELS 1

int main(int argc, char* argv[]) {
    int width; //image width
    int height; //image height
    int bpp;  //bytes per pixel
    double fx = 2.0, fy = 2.0;
    cv::InterpolationFlags alg = cv::INTER_LINEAR;
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
            alg = cv::INTER_NEAREST;
        }else if(strcmp(argv[4], "lin") == 0){
            alg = cv::INTER_LINEAR;
        }else if(strcmp(argv[4], "cub") == 0){
            alg = cv::INTER_CUBIC;
        }else if(strcmp(argv[4], "lan") == 0){
            alg = cv::INTER_LANCZOS4;
            std::cout << "Lanczos is not supported by OpenCV!" << std::endl;
            exit(EXIT_FAILURE);
        }else{
            std::cout << "Alg name not supported. Possible: nn, lin, cub, lan." << std::endl;
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

    cv::cuda::GpuMat d_image;
    cv::cuda::GpuMat d_image_out;

    d_image.upload(image);
    // Print for sanity check
    printf("Bytes per pixel: %d \n", bpp);
    printf("Height: %d \n", height);
    printf("Width: %d \n", width);

    cv::Mat scaled_image;

    start = clock();

    cv::cuda::resize(d_image, d_image_out, cv::Size(), fx, fy, alg);

    end = clock();

    d_image_out.download(scaled_image);
    // Write image array into a bmp file
    char buff[50];
    sprintf(buff, "_gpu_cv_%s_out", algName.c_str());
    filename.insert(filename.find_last_of("."), buff);
    cv::imwrite(filename.c_str(), scaled_image);
    printf("Time taken: %f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000.);

    if(show){
        cv::imshow("Result Image", scaled_image);
        cv::waitKey(0);
    }
    return 0;
}
