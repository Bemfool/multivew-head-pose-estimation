# Project name: Multi-View Head Pose Estimation
project(multiview-head-pose-estimation)
cmake_minimum_required(VERSION 2.8.12)

# Executable files:
#    hpe_multiview
#
# Link libraries:
#    bfm (for manipulating Basel Face Model)
#        - hdf5
#        - dlib
#        - opencv (for VideoCapture)
#        - ceres
#        - m (for mathematic operations)
#    dlib
#    ceres

if(UNIX)
    add_compile_options(-fPIC)
endif()
set(CMAKE_CXX_STANDARD 17)

# For accelaration
add_definitions(
    "-DUSE_AVX_INSTRUCTIONS"
    "-DUSE_SSE2_INSTRUCTIONS"
    "-DUSE_SSE4_INSTRUCTIONS"
    "-DCMAKE_BUILD_TYPE=Release"
)

add_library(multi-hpe SHARED src/hpe_problem.cpp src/string_utils.cpp lib/tinyxml2/tinyxml2.cpp)
# target_compile_features(hpe PRIVATE cxx_std_17)

# OpenCV
if(UNIX)
    find_package(OpenCV 3 REQUIRED HINTS "/usr/local/opencv3/share/OpenCV")
else()
    find_package(OpenCV 3 REQUIRED)
endif()
target_include_directories(multi-hpe PUBLIC ${OpenCV_INCLUDE_DIRS})

# Dlib
add_subdirectory(lib/dlib/dlib dlib_build)

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Hdf5
find_package(HDF5 REQUIRED C CXX)
include_directories(${HDF5_INCLUDE_DIRS})

# Bfm
add_subdirectory(lib/BFM-tools)
include_directories(lib/BFM-tools/include)

# stb_image.h
include_directories(lib/stb)

# tinyxml2
include_directories(lib/tinyxml2)

find_package(Boost REQUIRED system filesystem program_options)
include_directories(${Boost_INCLUDE_DIRS})

include_directories(./lib/tiny_progress)

# Head files
include_directories(./include)

if(UNIX)
    set(LIBS 
        /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.65.1
        /usr/lib/x86_64-linux-gnu/libboost_system.so.1.65.1
    )
endif()

# Source files
target_link_libraries(multi-hpe dlib::dlib ${CERES_LIBRARIES} bfm_manager ${Boost_LIBRARIES} boost_filesystem)

add_executable(hpe_multiview hpe_multiview.cpp)
target_link_libraries(hpe_multiview
    m Xrandr Xi X11 Xxf86vm pthread dl Xinerama Xcursor
    multi-hpe)