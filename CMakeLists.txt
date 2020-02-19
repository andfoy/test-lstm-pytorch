cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(rnn_test)

find_package(Torch REQUIRED)

# INCLUDE_DIRECTORIES(
#   "/home/andfoy/git/Quansight/libtorch/include"
#   "/home/andfoy/git/Quansight/libtorch/include/torch/csrc/api/include/"
# )

# LINK_DIRECTORIES(
#   "/home/andfoy/git/Quansight/libtorch/lib"
# )

add_executable(rnn_test rnn_test.cpp)
target_compile_features(rnn_test PUBLIC cxx_range_for)
target_link_libraries(rnn_test ${TORCH_LIBRARIES})
# target_link_libraries(rnn_test
# PRIVATE torch m)