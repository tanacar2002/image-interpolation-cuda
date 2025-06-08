cmake_minimum_required(VERSION 3.27)
project( GPU_TEX_RESIZE LANGUAGES CXX CUDA)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS}
                     ${CUDA_INCLUDE_DIRS}
                     "../commonlib"
                    )
add_executable( gpu_texture main_gpu_texture.cu )
set_property(TARGET gpu_texture PROPERTY CUDA_ARCHITECTURES 89)
target_compile_options(gpu_texture PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -use_fast_math
                        -Xptxas="-v";
                        >)
target_link_libraries( gpu_texture ${OpenCV_LIBS} ${CUDA_LIBS} m)