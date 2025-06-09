cmake_minimum_required(VERSION 3.28)
project( CPU_CV_RESIZE )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( cpu_cv main_cv_cpu.cpp )
target_compile_options(cpu_cv PRIVATE 
                        -O3 
                        -ffast-math
                        -march=native
                        )
target_link_libraries( cpu_cv ${OpenCV_LIBS} )