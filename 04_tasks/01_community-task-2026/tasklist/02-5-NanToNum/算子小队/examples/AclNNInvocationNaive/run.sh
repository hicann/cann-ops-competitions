#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
source $_ASCEND_INSTALL_PATH/bin/setenv.bash
export DDK_PATH=$_ASCEND_INSTALL_PATH
export NPU_HOST_LIB=$_ASCEND_INSTALL_PATH/lib64

rm -rf $HOME/ascend/log/*
rm ./input/*.bin
rm ./output/*.bin

function help_info() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo
    echo "-h|--help            Displays help message."
    echo
    echo "-t|--test_case        Excute user's testcase."
    echo "                      For example: -t \"test_1\" or -t \"test_2\""
    echo
}

case_name='test_1'

while [[ $# -gt 0 ]]; do
    case $1 in
    -h|--help)
    help_info
    exit
    ;;
    -t|--testcase)
    echo "============Excute user's testcase $2."
    case_name=$2;
    shift 2
    ;;
    *)
    help_info
    exit 1
    ;;
    esac
done

`python3 gen_data.py $case_name`

if [ $? -ne 0 ]; then
    echo "ERROR: generate input data failed!"
    return 1
fi
echo "INFO: generate input data success!"
set -e
rm -rf build
mkdir -p build
cmake -B build
cmake --build build -j
(
    cd build
    ./execute_add_op $case_name
)

ret=`python3 verify_result.py output/output_z.bin output/golden.bin $case_name`
echo $ret
if [ "x$ret" == "xtest pass" ]; then
    echo ""
    echo "#####################################"
    echo "INFO: you have passed the Precision!"
    echo "#####################################"
    echo ""
fi