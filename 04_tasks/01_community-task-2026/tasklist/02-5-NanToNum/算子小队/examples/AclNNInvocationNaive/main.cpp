/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file main.cpp
 */
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <fcntl.h>
#include <map>

#include "acl/acl.h"
#include "aclnn_nan_to_num.h"


#define SUCCESS 0
#define FAILED 1

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

using namespace std;

static map<string, vector<int64_t>> allcase {
	{"test_1" ,{1} },
    {"test_2" ,{1} },
    {"test_3" ,{1} },
    {"test_4" ,{7}},
    {"test_5" ,{8, 1} },
    {"test_6" ,{8} },
    {"test_7" ,{16, 1} },
    {"test_8" ,{19} },
    {"test_9" ,{20} },
    {"test_10" ,{20} },
    {"test_11" ,{20}},
    {"test_12" ,{9,15} },
    {"test_13" ,{8,17} },
    {"test_14" ,{7,21} },
    {"test_15" ,{255} },
    {"test_16" ,{20,20} },
    {"test_17" ,{256,8} },
    {"test_18" ,{19,9,16} },
    {"test_19" ,{20,20,7} },
    {"test_20" ,{257, 15} },
    {"test_21" ,{9,1,9,8,7} },
    {"test_22" ,{1, 20, 16, 17} },
    {"test_23" ,{1, 21, 1, 20, 19} },
    {"test_24" ,{16, 1, 7, 9, 21} },
    {"test_25" ,{17, 7, 15, 1, 161} },
    {"test_26" ,{7, 257, 19} },
    {"test_27" ,{9, 15, 255} },
    {"test_28" ,{9, 1, 255, 17, 1} },
    {"test_29" ,{17, 255, 15, 1} },
    {"test_30" ,{1, 16, 257, 21} },
    {"test_31" ,{131073} },
    {"test_32" ,{9, 9, 15, 19, 7} },
    {"test_33" ,{9, 7, 20, 21, 20} },
    {"test_34" ,{19, 17, 9, 255} },
    {"test_35" ,{15, 1, 8, 1, 7, 7, 8, 21} },
    {"test_36" ,{16, 8, 1, 8, 8, 8, 1, 21} },
    {"test_37" ,{19, 16, 16, 20, 19} },
    {"test_38" ,{16, 131073} },
    {"test_39" ,{15, 255, 19, 7, 8} },
    {"test_40" ,{20, 19, 15, 1, 7, 1, 9, 16} },
    {"test_41" ,{20, 256, 9, 8, 16} },
    {"test_42" ,{1, 8, 21, 20, 8, 16, 16} },
    {"test_43" ,{20, 7, 17, 15, 255} },
    {"test_44" ,{21, 256, 255, 7} },
    {"test_45" ,{21, 19, 21, 7, 9, 21} },
    {"test_46" ,{21, 7, 19, 257, 1, 17} },
    {"test_47" ,{21, 8, 131073, 1} },
    {"test_48" ,{20, 9, 131073, 1} },
    {"test_49" ,{17, 16, 20, 20, 20, 16} },
    {"test_50" ,{1, 17, 1, 256, 15, 9, 9, 7} },
};
static map<string, aclDataType> allcasedt {
    {"test_1" , aclDataType::ACL_FLOAT16},
    {"test_2" , aclDataType::ACL_FLOAT},
    {"test_3" , aclDataType::ACL_FLOAT},
    {"test_4" , aclDataType::ACL_FLOAT},
    {"test_5" , aclDataType::ACL_FLOAT},
    {"test_6" , aclDataType::ACL_FLOAT},
    {"test_7" , aclDataType::ACL_FLOAT},
    {"test_8" , aclDataType::ACL_BF16},
    {"test_9" , aclDataType::ACL_BF16},
    {"test_10" , aclDataType::ACL_FLOAT},
    {"test_11" , aclDataType::ACL_FLOAT},
    {"test_12" , aclDataType::ACL_FLOAT16},
    {"test_13" , aclDataType::ACL_FLOAT16},
    {"test_14" , aclDataType::ACL_FLOAT16},
    {"test_15" , aclDataType::ACL_FLOAT16},
    {"test_16" , aclDataType::ACL_FLOAT16},
    {"test_17" , aclDataType::ACL_FLOAT16},
    {"test_18" , aclDataType::ACL_FLOAT16},
    {"test_19" , aclDataType::ACL_FLOAT16},
    {"test_20" , aclDataType::ACL_FLOAT16},
    {"test_21" , aclDataType::ACL_FLOAT16},
    {"test_22" , aclDataType::ACL_FLOAT16},
    {"test_23" , aclDataType::ACL_FLOAT16},
    {"test_24" , aclDataType::ACL_FLOAT16},
    {"test_25" , aclDataType::ACL_FLOAT16},
    {"test_26" , aclDataType::ACL_FLOAT16},
    {"test_27" , aclDataType::ACL_FLOAT16},
    {"test_28" , aclDataType::ACL_FLOAT16},
    {"test_29" , aclDataType::ACL_FLOAT16},
    {"test_30" , aclDataType::ACL_FLOAT16},
    {"test_31" , aclDataType::ACL_FLOAT16},
    {"test_32" , aclDataType::ACL_FLOAT16},
    {"test_33" , aclDataType::ACL_FLOAT16},
    {"test_34" , aclDataType::ACL_FLOAT16},
    {"test_35" , aclDataType::ACL_FLOAT16},
    {"test_36" , aclDataType::ACL_FLOAT16},
    {"test_37" , aclDataType::ACL_FLOAT16},
    {"test_38" , aclDataType::ACL_FLOAT16},
    {"test_39" , aclDataType::ACL_FLOAT16},
    {"test_40" , aclDataType::ACL_FLOAT16},
    {"test_41" , aclDataType::ACL_FLOAT16},
    {"test_42" , aclDataType::ACL_FLOAT16},
    {"test_43" , aclDataType::ACL_FLOAT16},
    {"test_44" , aclDataType::ACL_FLOAT16},
    {"test_45" , aclDataType::ACL_FLOAT16},
    {"test_46" , aclDataType::ACL_FLOAT16},
    {"test_47" , aclDataType::ACL_FLOAT16},
    {"test_48" , aclDataType::ACL_FLOAT16},
    {"test_49" , aclDataType::ACL_FLOAT16},
    {"test_50" , aclDataType::ACL_FLOAT16},
};

bool ReadFile(const std::string &filePath, size_t fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file %s", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    auto writeSize = write(fd, buffer, size);
    (void) close(fd);
    if (writeSize != size) {
        ERROR_LOG("Write file Failed.");
        return false;
    }

    return true;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，acl初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return FAILED);

    return SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return FAILED);

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return SUCCESS;
}

int main(int argc, char **argv)
{
    // 1. （固定写法）device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    std::vector<int64_t> shape;
    aclDataType dtype;
    for(int i = 0; i < argc; i++) {
        string test_name = string(argv[i]);
        shape = allcase[test_name];
        dtype = allcasedt[test_name]; 
    }
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return FAILED);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> inputXShape = shape;

    std::vector<int64_t> outputZShape = shape;
    void *inputXDeviceAddr = nullptr;

    void *outputZDeviceAddr = nullptr;
    aclTensor *inputX = nullptr;

    aclTensor *outputZ = nullptr;
    size_t inputXShapeSize = GetShapeSize(inputXShape);
    size_t outputZShapeSize = GetShapeSize(outputZShape);
    size_t dataType = 2;
    switch (dtype)
    {
    case aclDataType::ACL_FLOAT16:
        dataType = 2;
        break;
    case aclDataType::ACL_FLOAT:
        dataType = 4;
        break;
    case aclDataType::ACL_BF16:
        dataType = 2;
        break;
    default:
        break;
    }
    std::vector<float> inputXHostData(inputXShapeSize);
    std::vector<float> outputZHostData(outputZShapeSize);

    size_t fileSize = 0;
    void ** input1=(void **)(&inputXHostData);
    //读取数据
    ReadFile("../input/input_x.bin", fileSize, *input1, inputXShapeSize * dataType);

    INFO_LOG("Set input success");
    // 创建inputX aclTensor
    ret = CreateAclTensor(inputXHostData, inputXShape, &inputXDeviceAddr, dtype, &inputX);
    float nanValueIn = 0;
    float posInfIn = 1;
    float negInfIn = -1;
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    // 创建outputZ aclTensor
    ret = CreateAclTensor(outputZHostData, outputZShape, &outputZDeviceAddr, dtype, &outputZ);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);

    // 3. 调用CANN自定义算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 计算workspace大小并申请内存
    ret = aclnnNanToNumGetWorkspaceSize(inputX, nanValueIn, posInfIn, negInfIn, outputZ, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNanToNumGetWorkspaceSize failed. ERROR: %d\n", ret); return FAILED);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return FAILED;);
    }
    // 执行算子
    ret = aclnnNanToNum(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNanToNum failed. ERROR: %d\n", ret); return FAILED);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return FAILED);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outputZShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputZDeviceAddr,
                      size * dataType, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return FAILED);
    void ** output1=(void **)(&resultData);
    //写出数据
    WriteFile("../output/output_z.bin", *output1, outputZShapeSize * dataType);
    INFO_LOG("Write output success");

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(inputX);
    aclDestroyTensor(outputZ);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(inputXDeviceAddr);
    aclrtFree(outputZDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return SUCCESS;
}
