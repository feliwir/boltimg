cmake_minimum_required(VERSION 3.14)
project(boltimg LANGUAGES C)

option(BOLTIMG_BUILD_APP "Build editing application" ON)
option(BOLTIMG_BUILD_BENCH "Build benchmarks" ON)
option(BOLTIMG_BUILD_TEST "Build tests" ON)

add_subdirectory(src)

include(FetchContent)
if(BOLTIMG_BUILD_APP)
  add_subdirectory(app)
endif()

if(BOLTIMG_BUILD_BENCH OR BOLTIMG_BUILD_TEST)
  # GoogleTest requires at least C++14
  enable_language(CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)

  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1)

  FetchContent_MakeAvailable(googletest)

  if(BOLTIMG_BUILD_BENCH)
    add_subdirectory(bench)
  endif()
  if(BOLTIMG_BUILD_TEST)
    add_subdirectory(test)
  endif()
endif()
