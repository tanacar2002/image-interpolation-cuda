cmake_minimum_required(VERSION 3.28)

project(IMAGE_RESIZE)

add_subdirectory("./cpu")
add_subdirectory("./gpu")
add_subdirectory("./gpu_cv")
add_subdirectory("./gpu_texture")