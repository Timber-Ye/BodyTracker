cmake_minimum_required(VERSION 3.16)
project(OpenCV)

set(CMAKE_CXX_STANDARD 11)
set(OpenCV_DIR "D:\\Installation\\opencv\\mingw64-build\\install")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(OpenCV Src/untitled.cpp Src/MyObjDetect.cpp Inc/MyObjDetect.h Src/KCF_BP_Kalman_Tracker.cpp Inc/KCF_BP_Kalman_Tracker.h)

# add libs you need
#set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_objdetect)
# linking
target_link_libraries(OpenCV ${OpenCV_LIBS})

