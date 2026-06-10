#!/bin/bash
if [ -z "$BASE_LIBS_PATH" ]; then
    if [ -z "$ASCEND_HOME_PATH" ]; then
        if [ -z "$ASCEND_AICPU_PATH" ]; then
            echo "please set env."
            exit 1
        else
            export ASCEND_HOME_PATH=$ASCEND_AICPU_PATH
        fi
    else
        export ASCEND_HOME_PATH=$ASCEND_HOME_PATH
    fi
else
    export ASCEND_HOME_PATH=$BASE_LIBS_PATH
fi
echo "using ASCEND_HOME_PATH: $ASCEND_HOME_PATH"

BUILD_DIR="build_out"
mkdir -p build_out
rm -rf build_out/*

target=package
if [ "$1"x != ""x ]; then target=$1; fi

cmake -S . -B "$BUILD_DIR" --preset=default \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}"
cmake --build "$BUILD_DIR" --target binary -j$(nproc)
cmake --build "$BUILD_DIR" --target $target -j$(nproc)
