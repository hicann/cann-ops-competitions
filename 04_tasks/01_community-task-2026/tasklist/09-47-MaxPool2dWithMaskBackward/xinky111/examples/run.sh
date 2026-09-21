#!/bin/bash
set -e
echo "Building and running example..."
mkdir -p build && cd build
cmake ..
make -j4
./test_aclnn_example
