cmake_minimum_required(VERSION 3.28)
project( CPU_RESIZE )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} 
                     "../commonlib"
                    )
add_executable( cpu main_cpu.cpp )
target_compile_options(cpu PRIVATE 
                        -O3 
                        -ffast-math
                        -march=native
                        )
target_link_libraries( cpu ${OpenCV_LIBS} )