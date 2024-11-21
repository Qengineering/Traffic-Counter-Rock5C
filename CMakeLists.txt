cmake_minimum_required(VERSION 3.10)
project(Traffic)

# Enable C++17 standard and set common compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)
find_package(OpenCV REQUIRED)   
find_package(gflags REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS} 
    ${GFLAGS_INCLUDE_DIRS}
    include 
    /usr/include/eigen3
    /usr/local/include/eigen3
)

# Source files
set(SRC_FILES 
    src/BYTETracker.cpp 
    src/MJPG_sender.cpp 
    src/MJPGthread.cpp 
    src/MQTT.cpp 
    src/STrack.cpp 
    src/TChannel.cpp 
    src/Tjson.cpp 
    src/kalmanFilter.cpp 
    src/lapjv.cpp 
    src/main.cpp 
    src/postprocess.cpp
    src/utils.cpp 
)

# Apply common compiler flags
add_compile_options(${COMMON_FLAGS})

set(EXECUTABLE_OUTPUT_PATH "../")

# Add executable
add_executable(Traffic ${SRC_FILES})

# Link libraries using the new-style (keyword) approach
target_link_libraries(Traffic
    PRIVATE
    ${OpenCV_LIBS}
    ${GFLAGS_LIBRARIES}
    /usr/local/lib/librknnrt.so
    /usr/local/lib/libpaho-mqtt3c.so
    dl
    pthread
)

# Add link options
target_link_options(Traffic PRIVATE -fopenmp -s)
