/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <iostream>
#include <string>
#include <typeinfo>

namespace aclco::test {

template<typename T>
std::string TypeName() {
    std::string name = typeid(T).name();
    
    if constexpr (std::is_same_v<T, uint32_t>) {
        return "uint32_t";
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return "uint64_t";
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return "uint8_t";
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return "int32_t";
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return "int8_t";
    } else if constexpr (std::is_same_v<T, float>) {
        return "float";
    } else if constexpr (std::is_same_v<T, double>) {
        return "double";
    } else if constexpr (std::is_same_v<T, int>) {
        return "int";
    } else if constexpr (std::is_same_v<T, unsigned int>) {
        return "unsigned int";
    } else if constexpr (std::is_same_v<T, long>) {
        return "long";
    } else if constexpr (std::is_same_v<T, unsigned long>) {
        return "unsigned long";
    } else if constexpr (std::is_same_v<T, long long>) {
        return "long long";
    } else if constexpr (std::is_same_v<T, unsigned long long>) {
        return "unsigned long long";
    } else if constexpr (std::is_same_v<T, char>) {
        return "char";
    } else if constexpr (std::is_same_v<T, unsigned char>) {
        return "unsigned char";
    } else if constexpr (std::is_same_v<T, short>) {
        return "short";
    } else if constexpr (std::is_same_v<T, unsigned short>) {
        return "unsigned short";
    }
    
    return name;
}

template<typename Probe>
std::string ProbingSchemeName() {
    std::string name = typeid(Probe).name();

    if(name.find("DoubleHashing") != std::string::npos) {
        return "DoubleHashing";
    } else if (name.find("LinearProbing") != std::string::npos) {
        return "LinearProbing";
    }
    return name;
}

class TestPrint {
public:
    TestPrint(const std::string& testName, 
               const std::string& keyType,
               const std::string& valueType,
               int bucketSize,
               std::size_t capacity,
               std::size_t numInsert,
               const std::string& extraParams = "",
               const std::string& probingScheme = "")
        : testName_(testName)
    {
        std::cout << "\n";
        std::cout << "========================================\n";
        std::cout << "Test: " << testName << "\n";
        std::cout << "========================================\n";
        std::cout << "Key Type: " << keyType << "\n";
        if (valueType != "void") {
            std::cout << "Value Type: " << valueType << "\n";
        }
        std::cout << "BucketSize: " << bucketSize << "\n";
        std::cout << "Capacity: " << capacity << "\n";
        if (numInsert != 0)
        std::cout << "Insert Numbers: " << numInsert << "\n";
        if (!extraParams.empty()) {
            std::cout << "Params: " << extraParams << "\n";
        }
        if (!probingScheme.empty()) {
            std::cout << "Probing Scheme: " << probingScheme << "\n";
        }
        std::cout << "----------------------------------------\n";
    }

    ~TestPrint() {
        std::cout << "----------------------------------------\n";
        std::cout << "Test [" << testName_ << "] " << (passed_ ? "PASSED" : "FAILED") << "\n";
        std::cout << "========================================\n";
        std::cout << "\n";
    }

    void Section(const std::string& sectionName) const {
        std::cout << "\n>>> Section: " << sectionName << "\n";
    }

    void Info(const std::string& message) const {
        std::cout << "  INFO: " << message << "\n";
    }

    void SetFailed() {
        passed_ = false;
    }

private:
    std::string testName_;
    bool passed_ = true;
};

#define PRINT_BEFORE_EXEC(testName, K, V, BS, capacity, numInsert) \
    aclco::test::TestPrint _test_output(testName, \
        aclco::test::TypeName<K>(), \
        aclco::test::TypeName<V>(), \
        BS, capacity, numInsert)

#define PRINT_BEFORE_EXEC_WITH_PARAMS(testName, K, V, BS, capacity, numInsert, params) \
    aclco::test::TestPrint _test_output(testName, \
        aclco::test::TypeName<K>(), \
        aclco::test::TypeName<V>(), \
        BS, capacity, numInsert, params)

#define PRINT_BEFORE_EXEC_SET(testName, K, BS, capacity, numInsert) \
    aclco::test::TestPrint _test_output(testName, \
        aclco::test::TypeName<K>(), \
        "void", \
        BS, capacity, numInsert)

#define PRINT_BEFORE_EXEC_SET_WITH_PARAMS(testName, K, BS, capacity, numInsert, params) \
    aclco::test::TestPrint _test_output(testName, \
        aclco::test::TypeName<K>(), \
        "void", \
        BS, capacity, numInsert, params)    
        
#define PRINT_BEFORE_EXEC_WITH_PROBE(testName, K, V, BS, capacity, numInsert, params, Probe) \
    aclco::test::TestPrint _test_output(testName, \
        aclco::test::TypeName<K>(), \
        aclco::test::TypeName<V>(), \
        BS, capacity, numInsert, params, aclco::test::ProbingSchemeName<Probe>())     

#define PRINT_BEFORE_EXEC_SET_WITH_PROBE(testName, K, BS, capacity, numInsert, params, Probe) \
    aclco::test::TestPrint _test_output(testName, \
        aclco::test::TypeName<K>(), \
        "void", \
        BS, capacity, numInsert, params, aclco::test::ProbingSchemeName<Probe>())           

#define PRINT_SECTION(name) _test_output.Section(name)
#define PRINT_INFO(msg) _test_output.Info(msg)
#define PRINT_FAILED() _test_output.SetFailed()

#define REQUIRE_PRINT(expr) \
    do { \
        if (!(expr)) { \
            PRINT_FAILED(); \
        } \
        REQUIRE(expr); \
    } while(0)

} // namespace aclco::test
