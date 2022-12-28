enable_testing()

add_executable(boltimg_test main.cpp)
target_link_libraries(boltimg_test boltimg GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(boltimg_test)