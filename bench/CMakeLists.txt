FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.7.1)

FetchContent_MakeAvailable(googlebenchmark)

add_executable(boltimg_bench bench_conv.cpp)
target_link_libraries(boltimg_bench boltimg benchmark::benchmark)
