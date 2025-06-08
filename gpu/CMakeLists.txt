cmake_minimum_required(VERSION 3.27)
project( GPU_RESIZE LANGUAGES CXX CUDA)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS}
                     ${CUDA_INCLUDE_DIRS}
                    "../commonlib"
                    )
add_executable( gpu main_gpu.cu )
set_property(TARGET gpu PROPERTY CUDA_ARCHITECTURES 89)
target_compile_options(gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -use_fast_math
                        -Xptxas="-v";
                        >)
target_link_libraries( gpu ${OpenCV_LIBS} ${CUDA_LIBS} m)