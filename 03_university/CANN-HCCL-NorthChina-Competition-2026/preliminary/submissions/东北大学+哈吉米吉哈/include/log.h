/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OPS_HCCL_LOG_H
#define OPS_HCCL_LOG_H

#include <cstdio>
#include <hccl/hccl_types.h>

#ifndef LOG_LEVEL
#define LOG_LEVEL LOG_LEVEL_ERROR // 默认日志级别为：ERROR
#endif

typedef enum {
    LOG_LEVEL_DEBUG = 0,   // DEBUG级别
    LOG_LEVEL_INFO = 1,    // INFO级别
    LOG_LEVEL_WARNING = 2, // WARNING级别
    LOG_LEVEL_ERROR = 3,   // ERROR级别
    LOG_LEVEL_NONE = 4     // 关闭所有日志
} LogLevel;

#ifndef LIKELY
#define LIKELY(x) (static_cast<bool>(__builtin_expect(static_cast<bool>(x), 1)))
#define UNLIKELY(x) (static_cast<bool>(__builtin_expect(static_cast<bool>(x), 0)))
#endif

#define HCCL_DEBUG(format, ...) \
    do { \
        if (LOG_LEVEL <= LOG_LEVEL_DEBUG) { \
            printf("[DEBUG][%s][%s:%d]" format "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__); \
        } \
    } while (0)

#define HCCL_INFO(format, ...) \
    do { \
        if (LOG_LEVEL <= LOG_LEVEL_INFO) { \
            printf("[INFO][%s][%s:%d]" format "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__); \
        } \
    } while (0)

#define HCCL_WARNING(format, ...) \
    do { \
        if (LOG_LEVEL <= LOG_LEVEL_WARNING) { \
            printf("[WARN][%s][%s:%d]" format "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__); \
        } \
    } while (0)

#define HCCL_ERROR(format, ...) \
    do { \
        if (LOG_LEVEL <= LOG_LEVEL_ERROR) { \
            printf("[ERROR][%s][%s:%d]" format "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__); \
        } \
    } while (0)

/* 检查指针, 若指针为NULL, 则记录日志, 并返回错误 */
#define CHK_PTR_NULL(ptr) \
    do { \
        if (UNLIKELY((ptr) == nullptr)) { \
            HCCL_ERROR("[%s] ptr [%s] is nullptr, return HCCL_E_PTR", __func__, #ptr); \
            return HCCL_E_PTR; \
        } \
    } while (0)

/* 检查函数返回值, 记录指定日志, 并返回指定错误码 */
#define CHK_PRT_RET(result, exeLog, retCode) \
    do { \
        if (UNLIKELY(result)) { \
            exeLog; \
            return retCode; \
        } \
    } while (0)

/* 检查函数返回值, 并返回指定错误码 */
#define CHK_RET(call) \
    do { \
        int32_t hcclRet = call; \
        if (UNLIKELY(hcclRet != HCCL_SUCCESS)) { \
            if (hcclRet == HCCL_E_AGAIN) { \
                HCCL_WARNING("[%s] call trace: hcclRet -> %d", __func__, hcclRet); \
            } else { \
                HCCL_ERROR("[%s] call trace: hcclRet -> %d", __func__, hcclRet); \
            } \
            return static_cast<HcclResult>(hcclRet); \
        } \
    } while (0)

#define ACLCHECK(cmd) \
    do { \
        aclError ret = cmd; \
        if (UNLIKELY(ret != ACL_SUCCESS)) { \
            HCCL_ERROR("acl interface return err %s:%d, retcode: %d.\n", __FILE__, __LINE__, ret); \
            if (ret == ACL_ERROR_RT_MEMORY_ALLOCATION) { \
                HCCL_ERROR("memory allocation error, check whether the current memory space is sufficient.\n"); \
            } \
            return HCCL_E_RUNTIME; \
        } \
    } while (0)

#endif // OPS_HCCL_LOG_H
