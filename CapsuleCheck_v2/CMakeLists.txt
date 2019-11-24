cmake_minimum_required(VERSION 2.8)

project(CapsuleCheck)

find_package(OpenCV REQUIRED)
message(STATUS "version: ${OpenCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(
${PROJECT_SOURCE_DIR}/include
)

add_executable(CapsuleCheck src/main.cpp)
# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})

# opencv libraries
target_link_libraries(CapsuleCheck ${OpenCV_LIBRARIES})