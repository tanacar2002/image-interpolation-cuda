# Image Interpolation Resizing
## Environment Setup
The code is meant to run on a machine with Ubuntu 24.04 LTS with CUDA toolkit 12.9 installed. Also, OpenCV should be built with the options below and installed.
```
cmake -DWITH_CUDA=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules -DCUDA_ARCH_BIN=8.9 -DARCH=sm_89 -Dgencode=arch=compute_89,code=sm_89 -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_C_COMPILER=/usr/bin/gcc-12 ../opencv-4.x
```
## Build
To build the code, run the following commands:
```
mkdir build && cd build
cmake ..
make
```
