/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <cstddef>
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

#define CHECK_RET(cond, return_expr) \
do {                               \
if (!(cond)) {                   \
return_expr;                   \
}                                \
} while (0)

#define LOG_PRINT(message, ...)     \
do {                              \
printf(message, ##__VA_ARGS__); \
} while (0)

namespace {
	int64_t GetShapeSize(const std::vector<int64_t>& shape)
	{
		int64_t shapeSize = 1;
		for (auto dim : shape) {
			shapeSize *= dim;
		}
		return shapeSize;
	}
	
	std::vector<int64_t> ComputeContiguousStrides(const std::vector<int64_t>& shape)
	{
		std::vector<int64_t> strides(shape.size(), 1);
		if (shape.size() >= 2) {
			for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
				strides[static_cast<size_t>(i)] =
				shape[static_cast<size_t>(i) + 1] * strides[static_cast<size_t>(i) + 1];
			}
		}
		return strides;
	}
	
	int Init(int32_t deviceId, aclrtStream* stream)
	{
		auto ret = aclInit(nullptr);
		CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
		
		ret = aclrtSetDevice(deviceId);
		CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
		
		ret = aclrtCreateStream(stream);
		CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
		
		return 0;
	}
	
	void Finalize(int32_t deviceId, aclrtStream stream)
	{
		if (stream != nullptr) {
			aclrtDestroyStream(stream);
		}
		aclrtResetDevice(deviceId);
		aclFinalize();
	}
	
	struct TensorResource {
		void* deviceAddr = nullptr;
		aclTensor* tensor = nullptr;
		std::vector<int64_t> shape;
		std::vector<int64_t> strides;
		
		void Destroy()
		{
			if (tensor != nullptr) {
				aclDestroyTensor(tensor);
				tensor = nullptr;
			}
			if (deviceAddr != nullptr) {
				aclrtFree(deviceAddr);
				deviceAddr = nullptr;
			}
			shape.clear();
			strides.clear();
		}
	};
	
	struct ScalarResource {
		aclScalar* scalar = nullptr;
		alignas(std::max_align_t) unsigned char valueBuf[16] = {0};
		
		void Destroy()
		{
			if (scalar != nullptr) {
				aclDestroyScalar(scalar);
				scalar = nullptr;
			}
			std::memset(valueBuf, 0, sizeof(valueBuf));
		}
	};
	
	uint32_t FloatToBits(float x)
	{
		uint32_t bits = 0;
		std::memcpy(&bits, &x, sizeof(bits));
		return bits;
	}
	
	float BitsToFloat(uint32_t bits)
	{
		float x = 0.0f;
		std::memcpy(&x, &bits, sizeof(x));
		return x;
	}
	
	uint16_t FloatToHalfBits(float value)
	{
		uint32_t x = FloatToBits(value);
		uint32_t sign = (x >> 16) & 0x8000u;
		uint32_t exp = (x >> 23) & 0xFFu;
		uint32_t mant = x & 0x7FFFFFu;
		
		if (exp == 0xFFu) {
			if (mant != 0) {
				return static_cast<uint16_t>(sign | 0x7E00u);
			}
			return static_cast<uint16_t>(sign | 0x7C00u);
		}
		
		int32_t newExp = static_cast<int32_t>(exp) - 127 + 15;
		if (newExp >= 31) {
			return static_cast<uint16_t>(sign | 0x7C00u);
		}
		
		if (newExp <= 0) {
			if (newExp < -10) {
				return static_cast<uint16_t>(sign);
			}
			mant |= 0x800000u;
			uint32_t shift = static_cast<uint32_t>(14 - newExp);
			uint32_t halfMant = mant >> shift;
			uint32_t remainder = mant & ((1u << shift) - 1u);
			uint32_t halfway = 1u << (shift - 1u);
			if (remainder > halfway || (remainder == halfway && (halfMant & 1u))) {
				++halfMant;
			}
			return static_cast<uint16_t>(sign | halfMant);
		}
		
		uint32_t halfExp = static_cast<uint32_t>(newExp) << 10;
		uint32_t halfMant = mant >> 13;
		uint32_t remainder = mant & 0x1FFFu;
		if (remainder > 0x1000u || (remainder == 0x1000u && (halfMant & 1u))) {
			++halfMant;
			if (halfMant == 0x400u) {
				halfMant = 0;
				halfExp += 0x400u;
				if (halfExp >= 0x7C00u) {
					return static_cast<uint16_t>(sign | 0x7C00u);
				}
			}
		}
		return static_cast<uint16_t>(sign | halfExp | (halfMant & 0x3FFu));
	}
	
	float HalfBitsToFloat(uint16_t h)
	{
		uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
		uint32_t exp = (h >> 10) & 0x1Fu;
		uint32_t mant = h & 0x03FFu;
		uint32_t bits = 0;
		
		if (exp == 0) {
			if (mant == 0) {
				bits = sign;
			} else {
				int e = -14;
				while ((mant & 0x0400u) == 0) {
					mant <<= 1;
					--e;
				}
				mant &= 0x03FFu;
				uint32_t exp32 = static_cast<uint32_t>(e + 127);
				bits = sign | (exp32 << 23) | (mant << 13);
			}
		} else if (exp == 0x1Fu) {
			bits = sign | 0x7F800000u | (mant << 13);
			if (mant != 0) {
				bits |= 0x1u;
			}
		} else {
			bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
		}
		
		return BitsToFloat(bits);
	}
	
	uint16_t FloatToBf16Bits(float value)
	{
		uint32_t bits = FloatToBits(value);
		uint32_t lsb = (bits >> 16) & 1u;
		bits += 0x7FFFu + lsb;
		return static_cast<uint16_t>(bits >> 16);
	}
	
	float Bf16BitsToFloat(uint16_t bf16)
	{
		uint32_t bits = static_cast<uint32_t>(bf16) << 16;
		return BitsToFloat(bits);
	}
	
	std::vector<uint16_t> MakeFloat16Vector(const std::vector<float>& values)
	{
		std::vector<uint16_t> result;
		result.reserve(values.size());
		for (float v : values) {
			result.push_back(FloatToHalfBits(v));
		}
		return result;
	}
	
	std::vector<uint16_t> MakeBFloat16Vector(const std::vector<float>& values)
	{
		std::vector<uint16_t> result;
		result.reserve(values.size());
		for (float v : values) {
			result.push_back(FloatToBf16Bits(v));
		}
		return result;
	}
	
	template <typename T>
	int CreateAclTensor(const std::vector<T>& hostData,
						const std::vector<int64_t>& shape,
						aclDataType dataType,
						TensorResource* resource)
	{
		CHECK_RET(resource != nullptr, return -1);
		
		int64_t numel = GetShapeSize(shape);
		CHECK_RET(numel >= 0, LOG_PRINT("Invalid shape size.\n"); return -1);
		CHECK_RET(static_cast<int64_t>(hostData.size()) == numel,
				  LOG_PRINT("Host data size(%zu) does not match shape numel(%ld).\n",
							hostData.size(), numel);
							return -1);
		
		resource->shape = shape;
		resource->strides = ComputeContiguousStrides(shape);
		
		size_t bytes = static_cast<size_t>((numel > 0 ? numel : 1) * static_cast<int64_t>(sizeof(T)));
		auto ret = aclrtMalloc(&resource->deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
		CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
		
		if (numel > 0) {
			ret = aclrtMemcpy(resource->deviceAddr, bytes,
							  hostData.data(), static_cast<size_t>(numel) * sizeof(T),
							  ACL_MEMCPY_HOST_TO_DEVICE);
			CHECK_RET(ret == ACL_SUCCESS,
					  LOG_PRINT("aclrtMemcpy host->device failed. ERROR: %d\n", ret);
					  resource->Destroy();
					  return ret);
		}
		
		const int64_t* shapeData = shape.empty() ? nullptr : shape.data();
		const int64_t* strideData = resource->strides.empty() ? nullptr : resource->strides.data();
		
		resource->tensor = aclCreateTensor(shapeData, shape.size(), dataType,
										   strideData, 0,
										   aclFormat::ACL_FORMAT_ND,
										   shapeData, shape.size(),
										   resource->deviceAddr);
		CHECK_RET(resource->tensor != nullptr,
				  LOG_PRINT("aclCreateTensor failed.\n");
				  resource->Destroy();
				  return -1);
		
		return 0;
	}
	
	template <typename T>
	int CreateAclTensorWithCustomLayout(const std::vector<T>& storageData,
										const std::vector<int64_t>& viewShape,
										const std::vector<int64_t>& storageShape,
										const std::vector<int64_t>& customStrides,
										aclDataType dataType,
										TensorResource* resource)
	{
		CHECK_RET(resource != nullptr, return -1);
		CHECK_RET(viewShape.size() == customStrides.size(),
				  LOG_PRINT("viewShape.size() != customStrides.size().\n");
				  return -1);
		
		int64_t storageNumel = GetShapeSize(storageShape);
		CHECK_RET(storageNumel >= 0, LOG_PRINT("Invalid storage shape size.\n"); return -1);
		CHECK_RET(static_cast<int64_t>(storageData.size()) == storageNumel,
				  LOG_PRINT("storageData size(%zu) does not match storage numel(%ld).\n",
							storageData.size(), storageNumel);
							return -1);
		
		resource->shape = viewShape;
		resource->strides = customStrides;
		
		size_t bytes = static_cast<size_t>((storageNumel > 0 ? storageNumel : 1) * static_cast<int64_t>(sizeof(T)));
		auto ret = aclrtMalloc(&resource->deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
		CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
		
		if (storageNumel > 0) {
			ret = aclrtMemcpy(resource->deviceAddr, bytes,
							  storageData.data(), static_cast<size_t>(storageNumel) * sizeof(T),
							  ACL_MEMCPY_HOST_TO_DEVICE);
			CHECK_RET(ret == ACL_SUCCESS,
					  LOG_PRINT("aclrtMemcpy host->device failed. ERROR: %d\n", ret);
					  resource->Destroy();
					  return ret);
		}
		
		const int64_t* viewShapeData = viewShape.empty() ? nullptr : viewShape.data();
		const int64_t* storageShapeData = storageShape.empty() ? nullptr : storageShape.data();
		const int64_t* strideData = customStrides.empty() ? nullptr : customStrides.data();
		
		resource->tensor = aclCreateTensor(viewShapeData, viewShape.size(), dataType,
										   strideData, 0,
										   aclFormat::ACL_FORMAT_ND,
										   storageShapeData, storageShape.size(),
										   resource->deviceAddr);
		CHECK_RET(resource->tensor != nullptr,
				  LOG_PRINT("aclCreateTensor(custom layout) failed.\n");
				  resource->Destroy();
				  return -1);
		
		return 0;
	}
	
	template <typename T>
	int CopyDeviceToHost(const TensorResource& resource, std::vector<T>* hostData)
	{
		CHECK_RET(hostData != nullptr, return -1);
		
		int64_t numel = GetShapeSize(resource.shape);
		hostData->assign(static_cast<size_t>(numel), T{});
		if (numel == 0) {
			return 0;
		}
		
		size_t bytes = static_cast<size_t>(numel) * sizeof(T);
		auto ret = aclrtMemcpy(hostData->data(), bytes,
							   resource.deviceAddr, bytes,
							   ACL_MEMCPY_DEVICE_TO_HOST);
		CHECK_RET(ret == ACL_SUCCESS,
				  LOG_PRINT("aclrtMemcpy device->host failed. ERROR: %d\n", ret);
				  return ret);
		return 0;
	}
	
	template <typename T>
	bool CreateAclScalar(const T& value, aclDataType dataType, ScalarResource* resource)
	{
		CHECK_RET(resource != nullptr, return false);
		static_assert(sizeof(T) <= 16, "Scalar type too large for valueBuf");
		std::memset(resource->valueBuf, 0, sizeof(resource->valueBuf));
		std::memcpy(resource->valueBuf, &value, sizeof(T));
		resource->scalar = aclCreateScalar(reinterpret_cast<void*>(resource->valueBuf), dataType);
		CHECK_RET(resource->scalar != nullptr,
				  LOG_PRINT("aclCreateScalar failed.\n");
				  return false);
		return true;
	}
	
	template <typename T>
	double DecodeAsDouble(const T& value, aclDataType dtype)
	{
		switch (dtype) {
		case ACL_BF16:
			return static_cast<double>(Bf16BitsToFloat(static_cast<uint16_t>(value)));
		case ACL_FLOAT16:
			return static_cast<double>(HalfBitsToFloat(static_cast<uint16_t>(value)));
		case ACL_FLOAT:
			return static_cast<double>(static_cast<float>(value));
		case ACL_DOUBLE:
			return static_cast<double>(value);
		case ACL_INT8:
			return static_cast<double>(static_cast<int8_t>(value));
		case ACL_UINT8:
			return static_cast<double>(static_cast<uint8_t>(value));
		case ACL_INT16:
			return static_cast<double>(static_cast<int16_t>(value));
		case ACL_INT32:
			return static_cast<double>(static_cast<int32_t>(value));
		case ACL_INT64:
			return static_cast<double>(static_cast<int64_t>(value));
		case ACL_BOOL:
			return static_cast<double>((value != 0) ? 1 : 0);
		default:
			return 0.0;
		}
	}
	
	template <typename T>
	int64_t DecodeAsInt64(const T& value, aclDataType dtype)
	{
		switch (dtype) {
		case ACL_BF16:
			return static_cast<int64_t>(Bf16BitsToFloat(static_cast<uint16_t>(value)));
		case ACL_FLOAT16:
			return static_cast<int64_t>(HalfBitsToFloat(static_cast<uint16_t>(value)));
		case ACL_FLOAT:
			return static_cast<int64_t>(static_cast<float>(value));
		case ACL_DOUBLE:
			return static_cast<int64_t>(static_cast<double>(value));
		case ACL_INT8:
			return static_cast<int64_t>(static_cast<int8_t>(value));
		case ACL_UINT8:
			return static_cast<int64_t>(static_cast<uint8_t>(value));
		case ACL_INT16:
			return static_cast<int64_t>(static_cast<int16_t>(value));
		case ACL_INT32:
			return static_cast<int64_t>(static_cast<int32_t>(value));
		case ACL_INT64:
			return static_cast<int64_t>(value);
		case ACL_BOOL:
			return (value != 0) ? 1 : 0;
		default:
			return 0;
		}
	}
	
	template <typename T>
	uint64_t DecodeAsUInt64(const T& value, aclDataType dtype)
	{
		switch (dtype) {
		case ACL_BF16:
			return static_cast<uint64_t>(Bf16BitsToFloat(static_cast<uint16_t>(value)));
		case ACL_FLOAT16:
			return static_cast<uint64_t>(HalfBitsToFloat(static_cast<uint16_t>(value)));
		case ACL_FLOAT:
			return static_cast<uint64_t>(static_cast<float>(value));
		case ACL_DOUBLE:
			return static_cast<uint64_t>(static_cast<double>(value));
		case ACL_INT8:
			return static_cast<uint64_t>(static_cast<int8_t>(value));
		case ACL_UINT8:
			return static_cast<uint64_t>(static_cast<uint8_t>(value));
		case ACL_INT16:
			return static_cast<uint64_t>(static_cast<int16_t>(value));
		case ACL_INT32:
			return static_cast<uint64_t>(static_cast<int32_t>(value));
		case ACL_INT64:
			return static_cast<uint64_t>(static_cast<int64_t>(value));
		case ACL_BOOL:
			return static_cast<uint64_t>((value != 0) ? 1 : 0);
		default:
			return 0;
		}
	}
	
	template <typename T>
	bool DecodeAsBool(const T& value, aclDataType dtype)
	{
		switch (dtype) {
		case ACL_BF16:
			return Bf16BitsToFloat(static_cast<uint16_t>(value)) != 0.0f;
		case ACL_FLOAT16:
			return HalfBitsToFloat(static_cast<uint16_t>(value)) != 0.0f;
		case ACL_FLOAT:
			return static_cast<float>(value) != 0.0f;
		case ACL_DOUBLE:
			return static_cast<double>(value) != 0.0;
		default:
			return DecodeAsInt64(value, dtype) != 0;
		}
	}
	
	template <typename OutT, typename In1T, typename In2T, typename AlphaT>
	OutT CpuAddCast(const In1T& lhs, aclDataType lhsType,
					const In2T& rhs, aclDataType rhsType,
					const AlphaT& alpha, aclDataType alphaType,
					aclDataType outType)
	{
		long double l = static_cast<long double>(DecodeAsDouble(lhs, lhsType));
		long double r = static_cast<long double>(DecodeAsDouble(rhs, rhsType));
		long double a = static_cast<long double>(DecodeAsDouble(alpha, alphaType));
		long double v = l + a * r;
		
		switch (outType) {
		case ACL_BF16:
			return static_cast<OutT>(FloatToBf16Bits(static_cast<float>(v)));
		case ACL_FLOAT16:
			return static_cast<OutT>(FloatToHalfBits(static_cast<float>(v)));
		case ACL_FLOAT:
			return static_cast<OutT>(static_cast<float>(v));
		case ACL_DOUBLE:
			return static_cast<OutT>(static_cast<double>(v));
		case ACL_BOOL:
			return static_cast<OutT>((v != 0.0L) ? 1 : 0);
			case ACL_INT8: {
				long long iv = static_cast<long long>(v);
				if (iv > 127) iv = 127;
				if (iv < -128) iv = -128;
				return static_cast<OutT>(static_cast<int8_t>(iv));
			}
			case ACL_UINT8: {
				long long iv = static_cast<long long>(v);
				if (iv > 255) iv = 255;
				if (iv < 0) iv = 0;
				return static_cast<OutT>(static_cast<uint8_t>(iv));
			}
		case ACL_INT16:
			return static_cast<OutT>(static_cast<int16_t>(static_cast<long long>(v)));
		case ACL_INT32:
			return static_cast<OutT>(static_cast<int32_t>(static_cast<long long>(v)));
		case ACL_INT64:
			return static_cast<OutT>(static_cast<int64_t>(static_cast<long long>(v)));
		default:
			return OutT{};
		}
	}
	
	bool ComputeBroadcastShape(const std::vector<int64_t>& a,
							   const std::vector<int64_t>& b,
							   std::vector<int64_t>* out)
	{
		CHECK_RET(out != nullptr, return false);
		
		size_t rank = (a.size() > b.size()) ? a.size() : b.size();
		out->assign(rank, 1);
		
		for (size_t i = 0; i < rank; ++i) {
			int64_t da = (i < rank - a.size()) ? 1 : a[i - (rank - a.size())];
			int64_t db = (i < rank - b.size()) ? 1 : b[i - (rank - b.size())];
			
			if (da == db) {
				(*out)[i] = da;
			} else if (da == 1) {
				(*out)[i] = db;
			} else if (db == 1) {
				(*out)[i] = da;
			} else {
				return false;
			}
		}
		return true;
	}
	
	std::vector<int64_t> AlignShapeToRank(const std::vector<int64_t>& shape, size_t rank)
	{
		std::vector<int64_t> aligned(rank, 1);
		size_t offset = rank - shape.size();
		for (size_t i = 0; i < shape.size(); ++i) {
			aligned[offset + i] = shape[i];
		}
		return aligned;
	}
	
	std::vector<int64_t> AlignStridesToRank(const std::vector<int64_t>& shape, size_t rank)
	{
		std::vector<int64_t> aligned(rank, 0);
		std::vector<int64_t> strides = ComputeContiguousStrides(shape);
		size_t offset = rank - shape.size();
		for (size_t i = 0; i < shape.size(); ++i) {
			aligned[offset + i] = strides[i];
		}
		return aligned;
	}
	
	void NormalizeTolerance(aclDataType outType, double* atol, double* rtol)
	{
		CHECK_RET(atol != nullptr && rtol != nullptr, return);
		if (*atol >= 0.0 && *rtol >= 0.0) {
			return;
		}
		
		switch (outType) {
		case ACL_BF16:
			*atol = 1e-2;
			*rtol = 1e-2;
			break;
		case ACL_FLOAT16:
			*atol = 1e-3;
			*rtol = 1e-3;
			break;
		case ACL_FLOAT:
			*atol = 1e-5;
			*rtol = 1e-5;
			break;
		case ACL_DOUBLE:
			*atol = 1e-12;
			*rtol = 1e-12;
			break;
		default:
			*atol = 0.0;
			*rtol = 0.0;
			break;
		}
	}
	
	bool FpClose(long double actual, long double expect, double atol, double rtol)
	{
		if (std::isnan(static_cast<double>(expect))) {
			return std::isnan(static_cast<double>(actual));
		}
		if (std::isinf(static_cast<double>(expect))) {
			return std::isinf(static_cast<double>(actual)) &&
			(std::signbit(static_cast<double>(actual)) == std::signbit(static_cast<double>(expect)));
		}
		if (std::isnan(static_cast<double>(actual)) || std::isinf(static_cast<double>(actual))) {
			return false;
		}
		
		long double diff = std::fabs(actual - expect);
		long double limit = static_cast<long double>(atol) +
		static_cast<long double>(rtol) * std::fabs(expect);
		return diff <= limit;
	}
	
	template <typename T>
	std::string ValueToString(const T& value, aclDataType dtype)
	{
		std::ostringstream oss;
		oss << std::setprecision(17);
		
		switch (dtype) {
			case ACL_BF16: {
			uint16_t bits = static_cast<uint16_t>(value);
			oss << Bf16BitsToFloat(bits) << " (bf16:0x" << std::hex
			<< static_cast<unsigned int>(bits) << std::dec << ")";
			break;
		}
			case ACL_FLOAT16: {
				uint16_t bits = static_cast<uint16_t>(value);
				oss << HalfBitsToFloat(bits) << " (fp16:0x" << std::hex
				<< static_cast<unsigned int>(bits) << std::dec << ")";
				break;
			}
		case ACL_FLOAT:
			oss << static_cast<double>(static_cast<float>(value));
			break;
		case ACL_DOUBLE:
			oss << static_cast<double>(value);
			break;
		case ACL_BOOL:
			oss << ((value != 0) ? 1 : 0);
			break;
		case ACL_INT8:
			oss << static_cast<int>(static_cast<int8_t>(value));
			break;
		case ACL_UINT8:
			oss << static_cast<unsigned int>(static_cast<uint8_t>(value));
			break;
		case ACL_INT16:
			oss << static_cast<int>(static_cast<int16_t>(value));
			break;
		case ACL_INT32:
			oss << static_cast<int32_t>(value);
			break;
		case ACL_INT64:
			oss << static_cast<long long>(static_cast<int64_t>(value));
			break;
		default:
			oss << "?";
			break;
		}
		
		return oss.str();
	}
	
	template <typename T>
	bool CompareVectors(const std::vector<T>& actual,
						const std::vector<T>& expect,
						aclDataType outType,
						double atol,
						double rtol,
						const std::string& caseName)
	{
		if (actual.size() != expect.size()) {
			LOG_PRINT("[FAIL] %s: output size mismatch, actual=%zu expect=%zu\n",
					  caseName.c_str(), actual.size(), expect.size());
			return false;
		}
		
		for (size_t i = 0; i < actual.size(); ++i) {
			bool ok = false;
			switch (outType) {
				case ACL_BF16: {
				float a = Bf16BitsToFloat(static_cast<uint16_t>(actual[i]));
				float e = Bf16BitsToFloat(static_cast<uint16_t>(expect[i]));
				ok = FpClose(static_cast<long double>(a), static_cast<long double>(e), atol, rtol);
				break;
			}
				case ACL_FLOAT16: {
					float a = HalfBitsToFloat(static_cast<uint16_t>(actual[i]));
					float e = HalfBitsToFloat(static_cast<uint16_t>(expect[i]));
					ok = FpClose(static_cast<long double>(a), static_cast<long double>(e), atol, rtol);
					break;
				}
				case ACL_FLOAT: {
					float a = static_cast<float>(actual[i]);
					float e = static_cast<float>(expect[i]);
					ok = FpClose(static_cast<long double>(a), static_cast<long double>(e), atol, rtol);
					break;
				}
				case ACL_DOUBLE: {
					double a = static_cast<double>(actual[i]);
					double e = static_cast<double>(expect[i]);
					ok = FpClose(static_cast<long double>(a), static_cast<long double>(e), atol, rtol);
					break;
				}
			case ACL_BOOL:
				ok = ((actual[i] != 0) == (expect[i] != 0));
				break;
			default:
				ok = (actual[i] == expect[i]);
				break;
			}
			
			if (!ok) {
				LOG_PRINT("[FAIL] %s: mismatch at index=%zu, actual=%s, expect=%s\n",
						  caseName.c_str(), i,
						  ValueToString(actual[i], outType).c_str(),
						  ValueToString(expect[i], outType).c_str());
				return false;
			}
		}
		return true;
	}
	
	template <typename SelfT, typename OtherT, typename AlphaT, typename OutT>
	bool ComputeAddExpect(const std::vector<SelfT>& selfData,
						  const std::vector<int64_t>& selfShape,
						  aclDataType selfType,
						  const std::vector<OtherT>& otherData,
						  const std::vector<int64_t>& otherShape,
						  aclDataType otherType,
						  const AlphaT& alpha,
						  aclDataType alphaType,
						  const std::vector<int64_t>& outShape,
						  aclDataType outType,
						  std::vector<OutT>* expect)
	{
		CHECK_RET(expect != nullptr, return false);
		
		std::vector<int64_t> realOutShape;
		if (!ComputeBroadcastShape(selfShape, otherShape, &realOutShape)) {
			LOG_PRINT("CPU expect compute failed: self/other shapes are not broadcastable.\n");
			return false;
		}
		if (realOutShape != outShape) {
			LOG_PRINT("CPU expect compute failed: outShape mismatch.\n");
			return false;
		}
		
		int64_t outNumel = GetShapeSize(outShape);
		expect->assign(static_cast<size_t>(outNumel), OutT{});
		if (outNumel == 0) {
			return true;
		}
		
		size_t rank = outShape.size();
		std::vector<int64_t> outStrides = ComputeContiguousStrides(outShape);
		std::vector<int64_t> selfAlignedShape = AlignShapeToRank(selfShape, rank);
		std::vector<int64_t> otherAlignedShape = AlignShapeToRank(otherShape, rank);
		std::vector<int64_t> selfAlignedStrides = AlignStridesToRank(selfShape, rank);
		std::vector<int64_t> otherAlignedStrides = AlignStridesToRank(otherShape, rank);
		
		for (int64_t linear = 0; linear < outNumel; ++linear) {
			int64_t remain = linear;
			int64_t selfOffset = 0;
			int64_t otherOffset = 0;
			
			for (size_t d = 0; d < rank; ++d) {
				int64_t coord = remain / outStrides[d];
				remain %= outStrides[d];
				
				if (selfAlignedShape[d] != 1) {
					selfOffset += coord * selfAlignedStrides[d];
				}
				if (otherAlignedShape[d] != 1) {
					otherOffset += coord * otherAlignedStrides[d];
				}
			}
			
			(*expect)[static_cast<size_t>(linear)] =
			CpuAddCast<OutT>(selfData[static_cast<size_t>(selfOffset)], selfType,
							 otherData[static_cast<size_t>(otherOffset)], otherType,
							 alpha, alphaType, outType);
		}
		return true;
	}
	template <typename SelfScalarT, typename OtherT, typename AlphaT, typename OutT>
	bool ComputeAddV3Expect(const SelfScalarT& selfScalar,
							aclDataType selfScalarType,
							const std::vector<OtherT>& otherData,
							const std::vector<int64_t>& otherShape,
							aclDataType otherType,
							const AlphaT& alpha,
							aclDataType alphaType,
							aclDataType outType,
							std::vector<OutT>* expect)
	{
		CHECK_RET(expect != nullptr, return false);
		
		int64_t outNumel = GetShapeSize(otherShape);
		expect->assign(static_cast<size_t>(outNumel), OutT{});
		for (int64_t i = 0; i < outNumel; ++i) {
			(*expect)[static_cast<size_t>(i)] =
			CpuAddCast<OutT>(selfScalar, selfScalarType,
							 otherData[static_cast<size_t>(i)], otherType,
							 alpha, alphaType, outType);
		}
		return true;
	}
	
	
	template <typename SelfT, typename OtherT, typename AlphaT, typename OutT>
	struct AddExecCase {
		std::string name;
		std::vector<int64_t> selfShape;
		std::vector<int64_t> otherShape;
		std::vector<int64_t> outShape;
		std::vector<SelfT> selfData;
		std::vector<OtherT> otherData;
		AlphaT alpha;
		aclDataType selfType;
		aclDataType otherType;
		aclDataType alphaType;
		aclDataType outType;
		double atol = -1.0;
		double rtol = -1.0;
	};
	
	template <typename SelfT, typename OtherScalarT, typename AlphaT, typename OutT>
	struct AddsExecCase {
		std::string name;
		std::vector<int64_t> selfShape;
		std::vector<SelfT> selfData;
		OtherScalarT otherScalar;
		AlphaT alpha;
		aclDataType selfType;
		aclDataType otherScalarType;
		aclDataType alphaType;
		aclDataType outType;
		double atol = -1.0;
		double rtol = -1.0;
	};
	
	template <typename SelfT, typename OtherT, typename AlphaT>
	struct InplaceAddExecCase {
		std::string name;
		std::vector<int64_t> selfShape;
		std::vector<int64_t> otherShape;
		std::vector<SelfT> selfData;
		std::vector<OtherT> otherData;
		AlphaT alpha;
		aclDataType selfType;
		aclDataType otherType;
		aclDataType alphaType;
		double atol = -1.0;
		double rtol = -1.0;
	};
	
	template <typename SelfT, typename OtherScalarT, typename AlphaT>
	struct InplaceAddsExecCase {
		std::string name;
		std::vector<int64_t> selfShape;
		std::vector<SelfT> selfData;
		OtherScalarT otherScalar;
		AlphaT alpha;
		aclDataType selfType;
		aclDataType otherScalarType;
		aclDataType alphaType;
		double atol = -1.0;
		double rtol = -1.0;
	};
	
	template <typename SelfScalarT, typename OtherT, typename AlphaT, typename OutT>
	struct AddV3ExecCase {
		std::string name;
		SelfScalarT selfScalar;
		std::vector<int64_t> otherShape;
		std::vector<OtherT> otherData;
		AlphaT alpha;
		aclDataType selfScalarType;
		aclDataType otherType;
		aclDataType alphaType;
		aclDataType outType;
		double atol = -1.0;
		double rtol = -1.0;
	};
	
	
	template <typename SelfT, typename OtherT, typename AlphaT, typename OutT>
	bool RunAddExecCase(const AddExecCase<SelfT, OtherT, AlphaT, OutT>& tc, aclrtStream stream)
	{
		LOG_PRINT("\n[RUN] Add Exec Case: %s\n", tc.name.c_str());
		
		std::vector<OutT> expect;
		if (!ComputeAddExpect(tc.selfData, tc.selfShape, tc.selfType,
							  tc.otherData, tc.otherShape, tc.otherType,
							  tc.alpha, tc.alphaType,
							  tc.outShape, tc.outType, &expect)) {
			LOG_PRINT("[FAIL] %s: CPU expect compute failed.\n", tc.name.c_str());
			return false;
		}
		
		TensorResource selfRes;
		TensorResource otherRes;
		TensorResource outRes;
		ScalarResource alphaRes;
		void* workspaceAddr = nullptr;
		uint64_t workspaceSize = 0;
		bool success = false;
		
		auto Cleanup = [&]() {
			if (workspaceAddr != nullptr) {
				aclrtFree(workspaceAddr);
				workspaceAddr = nullptr;
			}
			alphaRes.Destroy();
			selfRes.Destroy();
			otherRes.Destroy();
			outRes.Destroy();
		};
		
		do {
			std::vector<OutT> outInit(static_cast<size_t>(GetShapeSize(tc.outShape)), OutT{});
			
			int ret = CreateAclTensor(tc.selfData, tc.selfShape, tc.selfType, &selfRes);
			if (ret != 0) break;
			ret = CreateAclTensor(tc.otherData, tc.otherShape, tc.otherType, &otherRes);
			if (ret != 0) break;
			ret = CreateAclTensor(outInit, tc.outShape, tc.outType, &outRes);
			if (ret != 0) break;
			if (!CreateAclScalar(tc.alpha, tc.alphaType, &alphaRes)) break;
			
			aclOpExecutor* executor = nullptr;
			auto aclRet = aclnnAddGetWorkspaceSize(selfRes.tensor, otherRes.tensor, alphaRes.scalar, outRes.tensor,
												   &workspaceSize, &executor);
			if (aclRet != ACL_SUCCESS) {
				LOG_PRINT("[FAIL] %s: aclnnAddGetWorkspaceSize failed. ERROR: %d\n",
						  tc.name.c_str(), static_cast<int>(aclRet));
				break;
			}
			
			if (workspaceSize > 0) {
				ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
				if (ret != ACL_SUCCESS) {
					LOG_PRINT("[FAIL] %s: workspace malloc failed. ERROR: %d\n",
							  tc.name.c_str(), ret);
					break;
				}
			}
			
			aclRet = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
			if (aclRet != ACL_SUCCESS) {
				LOG_PRINT("[FAIL] %s: aclnnAdd execute failed. ERROR: %d\n",
						  tc.name.c_str(), static_cast<int>(aclRet));
				break;
			}
			
			ret = aclrtSynchronizeStream(stream);
			if (ret != ACL_SUCCESS) {
				LOG_PRINT("[FAIL] %s: aclrtSynchronizeStream failed. ERROR: %d\n",
						  tc.name.c_str(), ret);
				break;
			}
			
			std::vector<OutT> actual;
			ret = CopyDeviceToHost(outRes, &actual);
			if (ret != 0) break;
			
			double atol = tc.atol;
			double rtol = tc.rtol;
			NormalizeTolerance(tc.outType, &atol, &rtol);
			
			if (!CompareVectors(actual, expect, tc.outType, atol, rtol, tc.name)) {
				break;
			}
			
			success = true;
		} while (0);
		
		Cleanup();
		LOG_PRINT(success ? "[PASS] %s\n" : "[FAIL] %s\n", tc.name.c_str());
		return success;
	}
	
	template <typename SelfT, typename OtherScalarT, typename AlphaT, typename OutT>
	bool RunAddsExecCase(const AddsExecCase<SelfT, OtherScalarT, AlphaT, OutT>& tc, aclrtStream stream)
	{
		LOG_PRINT("\n[RUN] Adds Exec Case: %s\n", tc.name.c_str());
		
		int64_t numel = GetShapeSize(tc.selfShape);
		std::vector<OutT> expect(static_cast<size_t>(numel), OutT{});
		for (int64_t i = 0; i < numel; ++i) {
			expect[static_cast<size_t>(i)] =
			CpuAddCast<OutT>(tc.selfData[static_cast<size_t>(i)], tc.selfType,
							 tc.otherScalar, tc.otherScalarType,
							 tc.alpha, tc.alphaType,
							 tc.outType);
		}
		
		TensorResource selfRes;
		TensorResource outRes;
		ScalarResource otherScalarRes;
		ScalarResource alphaRes;
		void* workspaceAddr = nullptr;
		uint64_t workspaceSize = 0;
		bool success = false;
		
		auto Cleanup = [&]() {
			if (workspaceAddr != nullptr) {
				aclrtFree(workspaceAddr);
				workspaceAddr = nullptr;
			}
			otherScalarRes.Destroy();
			alphaRes.Destroy();
			selfRes.Destroy();
			outRes.Destroy();
		};
		
		do {
			std::vector<OutT> outInit(static_cast<size_t>(numel), OutT{});
			
			int ret = CreateAclTensor(tc.selfData, tc.selfShape, tc.selfType, &selfRes);
			if (ret != 0) break;
			ret = CreateAclTensor(outInit, tc.selfShape, tc.outType, &outRes);
			if (ret != 0) break;
			if (!CreateAclScalar(tc.otherScalar, tc.otherScalarType, &otherScalarRes)) break;
			if (!CreateAclScalar(tc.alpha, tc.alphaType, &alphaRes)) break;
			
			aclOpExecutor* executor = nullptr;
			auto aclRet = aclnnAddsGetWorkspaceSize(selfRes.tensor, otherScalarRes.scalar, alphaRes.scalar, outRes.tensor,
													&workspaceSize, &executor);
			if (aclRet != ACL_SUCCESS) {
				LOG_PRINT("[FAIL] %s: aclnnAddsGetWorkspaceSize failed. ERROR: %d\n",
						  tc.name.c_str(), static_cast<int>(aclRet));
				break;
			}
			
			if (workspaceSize > 0) {
				ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
				if (ret != ACL_SUCCESS) {
					LOG_PRINT("[FAIL] %s: workspace malloc failed. ERROR: %d\n",
							  tc.name.c_str(), ret);
					break;
				}
			}
			
			aclRet = aclnnAdds(workspaceAddr, workspaceSize, executor, stream);
			if (aclRet != ACL_SUCCESS) {
				LOG_PRINT("[FAIL] %s: aclnnAdds execute failed. ERROR: %d\n",
						  tc.name.c_str(), static_cast<int>(aclRet));
				break;
			}
			
			ret = aclrtSynchronizeStream(stream);
			if (ret != ACL_SUCCESS) {
				LOG_PRINT("[FAIL] %s: aclrtSynchronizeStream failed. ERROR: %d\n",
						  tc.name.c_str(), ret);
				break;
			}
			
			std::vector<OutT> actual;
			ret = CopyDeviceToHost(outRes, &actual);
			if (ret != 0) break;
			
			double atol = tc.atol;
			double rtol = tc.rtol;
			NormalizeTolerance(tc.outType, &atol, &rtol);
			
			if (!CompareVectors(actual, expect, tc.outType, atol, rtol, tc.name)) {
				break;
			}
			
			success = true;
		} while (0);
		
		Cleanup();
		LOG_PRINT(success ? "[PASS] %s\n" : "[FAIL] %s\n", tc.name.c_str());
		return success;
	}
	
	template <typename SelfT, typename OtherT, typename AlphaT>
	bool RunInplaceAddExecCase(const InplaceAddExecCase<SelfT, OtherT, AlphaT>& tc, aclrtStream stream)
	{
		LOG_PRINT("\n[RUN] InplaceAdd Exec Case: %s\n", tc.name.c_str());
		
		std::vector<SelfT> expect;
		if (!ComputeAddExpect(tc.selfData, tc.selfShape, tc.selfType,
							  tc.otherData, tc.otherShape, tc.otherType,
							  tc.alpha, tc.alphaType,
							  tc.selfShape, tc.selfType, &expect)) {
			LOG_PRINT("[FAIL] %s: CPU expect compute failed.\n", tc.name.c_str());
			return false;
		}
		
		TensorResource selfRes;
		TensorResource otherRes;
		ScalarResource alphaRes;
		void* workspaceAddr = nullptr;
		uint64_t workspaceSize = 0;
		bool success = false;
		
		auto Cleanup = [&]() {
			if (workspaceAddr != nullptr) {
				aclrtFree(workspaceAddr);
				workspaceAddr = nullptr;
			}
			alphaRes.Destroy();
			selfRes.Destroy();
			otherRes.Destroy();
		};
		
		do {
			int ret = CreateAclTensor(tc.selfData, tc.selfShape, tc.selfType, &selfRes);
			if (ret != 0) break;
			ret = CreateAclTensor(tc.otherData, tc.otherShape, tc.otherType, &otherRes);
			if (ret != 0) break;
			if (!CreateAclScalar(tc.alpha, tc.alphaType, &alphaRes)) break;
			
			aclOpExecutor* executor = nullptr;
			auto aclRet = aclnnInplaceAddGetWorkspaceSize(selfRes.tensor, otherRes.tensor, alphaRes.scalar,
														  &workspaceSize, &executor);
			if (aclRet != ACL_SUCCESS) {
				LOG_PRINT("[FAIL] %s: aclnnInplaceAddGetWorkspaceSize failed. ERROR: %d\n",
						  tc.name.c_str(), static_cast<int>(aclRet));
				break;
			}
			
			if (workspaceSize > 0) {
				ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
				if (ret != ACL_SUCCESS) {
					LOG_PRINT("[FAIL] %s: workspace malloc failed. ERROR: %d\n",
							  tc.name.c_str(), ret);
					break;
				}
			}
			
			aclRet = aclnnInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
			if (aclRet != ACL_SUCCESS) {
				LOG_PRINT("[FAIL] %s: aclnnInplaceAdd execute failed. ERROR: %d\n",
						  tc.name.c_str(), static_cast<int>(aclRet));
				break;
			}
			
			ret = aclrtSynchronizeStream(stream);
			if (ret != ACL_SUCCESS) {
				LOG_PRINT("[FAIL] %s: aclrtSynchronizeStream failed. ERROR: %d\n",
						  tc.name.c_str(), ret);
				break;
			}
			
			std::vector<SelfT> actual;
			ret = CopyDeviceToHost(selfRes, &actual);
			if (ret != 0) break;
			
			double atol = tc.atol;
			double rtol = tc.rtol;
			NormalizeTolerance(tc.selfType, &atol, &rtol);
			
			if (!CompareVectors(actual, expect, tc.selfType, atol, rtol, tc.name)) {
				break;
			}
			
			success = true;
		} while (0);
		
		Cleanup();
		LOG_PRINT(success ? "[PASS] %s\n" : "[FAIL] %s\n", tc.name.c_str());
		return success;
	}
	
	template <typename SelfT, typename OtherScalarT, typename AlphaT>
	bool RunInplaceAddsExecCase(const InplaceAddsExecCase<SelfT, OtherScalarT, AlphaT>& tc, aclrtStream stream)
	{
		LOG_PRINT("\n[RUN] InplaceAdds Exec Case: %s\n", tc.name.c_str());
		
		int64_t numel = GetShapeSize(tc.selfShape);
		std::vector<SelfT> expect(static_cast<size_t>(numel), SelfT{});
		for (int64_t i = 0; i < numel; ++i) {
			expect[static_cast<size_t>(i)] =
			CpuAddCast<SelfT>(tc.selfData[static_cast<size_t>(i)], tc.selfType,
							  tc.otherScalar, tc.otherScalarType,
							  tc.alpha, tc.alphaType,
							  tc.selfType);
		}
		
		TensorResource selfRes;
		ScalarResource otherScalarRes;
		ScalarResource alphaRes;
		void* workspaceAddr = nullptr;
		uint64_t workspaceSize = 0;
		bool success = false;
		
		auto Cleanup = [&]() {
			if (workspaceAddr != nullptr) {
				aclrtFree(workspaceAddr);
				workspaceAddr = nullptr;
			}
			otherScalarRes.Destroy();
			alphaRes.Destroy();
			selfRes.Destroy();
		};
		
		do {
			int ret = CreateAclTensor(tc.selfData, tc.selfShape, tc.selfType, &selfRes);
			if (ret != 0) break;
			if (!CreateAclScalar(tc.otherScalar, tc.otherScalarType, &otherScalarRes)) break;
			if (!CreateAclScalar(tc.alpha, tc.alphaType, &alphaRes)) break;
			
			aclOpExecutor* executor = nullptr;
			auto aclRet = aclnnInplaceAddsGetWorkspaceSize(selfRes.tensor, otherScalarRes.scalar, alphaRes.scalar,
														   &workspaceSize, &executor);
			if (aclRet != ACL_SUCCESS) {
				LOG_PRINT("[FAIL] %s: aclnnInplaceAddsGetWorkspaceSize failed. ERROR: %d\n",
						  tc.name.c_str(), static_cast<int>(aclRet));
				break;
			}
			
			if (workspaceSize > 0) {
				ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
				if (ret != ACL_SUCCESS) {
					LOG_PRINT("[FAIL] %s: workspace malloc failed. ERROR: %d\n",
							  tc.name.c_str(), ret);
					break;
				}
			}
			
			aclRet = aclnnInplaceAdds(workspaceAddr, workspaceSize, executor, stream);
			if (aclRet != ACL_SUCCESS) {
				LOG_PRINT("[FAIL] %s: aclnnInplaceAdds execute failed. ERROR: %d\n",
						  tc.name.c_str(), static_cast<int>(aclRet));
				break;
			}
			
			ret = aclrtSynchronizeStream(stream);
			if (ret != ACL_SUCCESS) {
				LOG_PRINT("[FAIL] %s: aclrtSynchronizeStream failed. ERROR: %d\n",
						  tc.name.c_str(), ret);
				break;
			}
			
			std::vector<SelfT> actual;
			ret = CopyDeviceToHost(selfRes, &actual);
			if (ret != 0) break;
			
			double atol = tc.atol;
			double rtol = tc.rtol;
			NormalizeTolerance(tc.selfType, &atol, &rtol);
			
			if (!CompareVectors(actual, expect, tc.selfType, atol, rtol, tc.name)) {
				break;
			}
			
			success = true;
		} while (0);
		
		Cleanup();
		LOG_PRINT(success ? "[PASS] %s\n" : "[FAIL] %s\n", tc.name.c_str());
		return success;
	}
	
	
	template <typename SelfScalarT, typename OtherT, typename AlphaT, typename OutT>
	bool RunAddV3ExecCase(const AddV3ExecCase<SelfScalarT, OtherT, AlphaT, OutT>& tc, aclrtStream stream)
	{
		LOG_PRINT("\n[RUN] AddV3 Exec Case: %s\n", tc.name.c_str());
		
		std::vector<OutT> expect;
		if (!ComputeAddV3Expect(tc.selfScalar, tc.selfScalarType,
								tc.otherData, tc.otherShape, tc.otherType,
								tc.alpha, tc.alphaType,
								tc.outType, &expect)) {
			LOG_PRINT("[FAIL] %s: CPU expect compute failed.\n", tc.name.c_str());
			return false;
		}
		
		TensorResource otherRes;
		TensorResource outRes;
		ScalarResource selfRes;
		ScalarResource alphaRes;
		void* workspaceAddr = nullptr;
		uint64_t workspaceSize = 0;
		bool success = false;
		
		auto Cleanup = [&]() {
			if (workspaceAddr != nullptr) {
				aclrtFree(workspaceAddr);
				workspaceAddr = nullptr;
			}
			alphaRes.Destroy();
			selfRes.Destroy();
			otherRes.Destroy();
			outRes.Destroy();
		};
		
		do {
			std::vector<OutT> outInit(static_cast<size_t>(GetShapeSize(tc.otherShape)), OutT{});
			
			int ret = CreateAclTensor(tc.otherData, tc.otherShape, tc.otherType, &otherRes);
			if (ret != 0) break;
			ret = CreateAclTensor(outInit, tc.otherShape, tc.outType, &outRes);
			if (ret != 0) break;
			if (!CreateAclScalar(tc.selfScalar, tc.selfScalarType, &selfRes)) break;
			if (!CreateAclScalar(tc.alpha, tc.alphaType, &alphaRes)) break;
			
			aclOpExecutor* executor = nullptr;
			auto aclRet = aclnnAddV3GetWorkspaceSize(selfRes.scalar, otherRes.tensor, alphaRes.scalar, outRes.tensor,
													 &workspaceSize, &executor);
			if (aclRet != ACL_SUCCESS) {
				LOG_PRINT("[FAIL] %s: aclnnAddV3GetWorkspaceSize failed. ERROR: %d\n",
						  tc.name.c_str(), static_cast<int>(aclRet));
				break;
			}
			
			if (workspaceSize > 0) {
				ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
				if (ret != ACL_SUCCESS) {
					LOG_PRINT("[FAIL] %s: workspace malloc failed. ERROR: %d\n",
							  tc.name.c_str(), ret);
					break;
				}
			}
			
			aclRet = aclnnAddV3(workspaceAddr, workspaceSize, executor, stream);
			if (aclRet != ACL_SUCCESS) {
				LOG_PRINT("[FAIL] %s: aclnnAddV3 execute failed. ERROR: %d\n",
						  tc.name.c_str(), static_cast<int>(aclRet));
				break;
			}
			
			ret = aclrtSynchronizeStream(stream);
			if (ret != ACL_SUCCESS) {
				LOG_PRINT("[FAIL] %s: aclrtSynchronizeStream failed. ERROR: %d\n",
						  tc.name.c_str(), ret);
				break;
			}
			
			std::vector<OutT> actual;
			ret = CopyDeviceToHost(outRes, &actual);
			if (ret != 0) break;
			
			double atol = tc.atol;
			double rtol = tc.rtol;
			NormalizeTolerance(tc.outType, &atol, &rtol);
			
			if (!CompareVectors(actual, expect, tc.outType, atol, rtol, tc.name)) {
				break;
			}
			
			success = true;
		} while (0);
		
		Cleanup();
		LOG_PRINT(success ? "[PASS] %s\n" : "[FAIL] %s\n", tc.name.c_str());
		return success;
	}
	bool ExpectAddGetWorkspaceFail(const std::string& caseName,
								   aclTensor* self,
								   aclTensor* other,
								   aclScalar* alpha,
								   aclTensor* out)
	{
		uint64_t workspaceSize = 0;
		aclOpExecutor* executor = nullptr;
		auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
		if (ret == ACL_SUCCESS) {
			LOG_PRINT("[FAIL] %s: expected failure, but got ACL_SUCCESS.\n", caseName.c_str());
			return false;
		}
		LOG_PRINT("[PASS] %s: failed as expected. ERROR: %d\n",
				  caseName.c_str(), static_cast<int>(ret));
		return true;
	}
	
	bool ExpectAddsGetWorkspaceFail(const std::string& caseName,
									aclTensor* self,
									aclScalar* other,
									aclScalar* alpha,
									aclTensor* out)
	{
		uint64_t workspaceSize = 0;
		aclOpExecutor* executor = nullptr;
		auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
		if (ret == ACL_SUCCESS) {
			LOG_PRINT("[FAIL] %s: expected failure, but got ACL_SUCCESS.\n", caseName.c_str());
			return false;
		}
		LOG_PRINT("[PASS] %s: failed as expected. ERROR: %d\n",
				  caseName.c_str(), static_cast<int>(ret));
		return true;
	}
	
	bool ExpectInplaceAddGetWorkspaceFail(const std::string& caseName,
										  aclTensor* self,
										  aclTensor* other,
										  aclScalar* alpha)
	{
		uint64_t workspaceSize = 0;
		aclOpExecutor* executor = nullptr;
		auto ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
		if (ret == ACL_SUCCESS) {
			LOG_PRINT("[FAIL] %s: expected failure, but got ACL_SUCCESS.\n", caseName.c_str());
			return false;
		}
		LOG_PRINT("[PASS] %s: failed as expected. ERROR: %d\n",
				  caseName.c_str(), static_cast<int>(ret));
		return true;
	}
	
	bool ExpectInplaceAddsGetWorkspaceFail(const std::string& caseName,
										   aclTensor* self,
										   aclScalar* other,
										   aclScalar* alpha)
	{
		uint64_t workspaceSize = 0;
		aclOpExecutor* executor = nullptr;
		auto ret = aclnnInplaceAddsGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
		if (ret == ACL_SUCCESS) {
			LOG_PRINT("[FAIL] %s: expected failure, but got ACL_SUCCESS.\n", caseName.c_str());
			return false;
		}
		LOG_PRINT("[PASS] %s: failed as expected. ERROR: %d\n",
				  caseName.c_str(), static_cast<int>(ret));
		return true;
	}
	bool ExpectAddV3GetWorkspaceFail(const std::string& caseName,
									 aclScalar* self,
									 aclTensor* other,
									 aclScalar* alpha,
									 aclTensor* out)
	{
		uint64_t workspaceSize = 0;
		aclOpExecutor* executor = nullptr;
		auto ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
		if (ret == ACL_SUCCESS) {
			LOG_PRINT("[FAIL] %s: expected failure, but got ACL_SUCCESS.\n", caseName.c_str());
			return false;
		}
		LOG_PRINT("[PASS] %s: failed as expected. ERROR: %d\n",
				  caseName.c_str(), static_cast<int>(ret));
		return true;
	}
	bool TestInvalidNullptrSelf()
	{
		LOG_PRINT("\n[RUN] Invalid Case: nullptr self\n");
		TensorResource otherRes;
		TensorResource outRes;
		ScalarResource alphaRes;
		bool success = false;
		
		std::vector<int64_t> shape = {2, 2};
		std::vector<float> otherData = {1.f, 2.f, 3.f, 4.f};
		std::vector<float> outData(4, 0.f);
		float alpha = 1.0f;
		
		do {
			if (CreateAclTensor(otherData, shape, ACL_FLOAT, &otherRes) != 0) break;
			if (CreateAclTensor(outData, shape, ACL_FLOAT, &outRes) != 0) break;
			if (!CreateAclScalar(alpha, ACL_FLOAT, &alphaRes)) break;
			success = ExpectAddGetWorkspaceFail("invalid_nullptr_self", nullptr, otherRes.tensor, alphaRes.scalar, outRes.tensor);
		} while (0);
		
		alphaRes.Destroy();
		otherRes.Destroy();
		outRes.Destroy();
		return success;
	}
	
	bool TestInvalidNullptrOther()
	{
		LOG_PRINT("\n[RUN] Invalid Case: nullptr other\n");
		TensorResource selfRes;
		TensorResource outRes;
		ScalarResource alphaRes;
		bool success = false;
		
		std::vector<int64_t> shape = {2, 2};
		std::vector<float> selfData = {1.f, 2.f, 3.f, 4.f};
		std::vector<float> outData(4, 0.f);
		float alpha = 1.0f;
		
		do {
			if (CreateAclTensor(selfData, shape, ACL_FLOAT, &selfRes) != 0) break;
			if (CreateAclTensor(outData, shape, ACL_FLOAT, &outRes) != 0) break;
			if (!CreateAclScalar(alpha, ACL_FLOAT, &alphaRes)) break;
			success = ExpectAddGetWorkspaceFail("invalid_nullptr_other", selfRes.tensor, nullptr, alphaRes.scalar, outRes.tensor);
		} while (0);
		
		alphaRes.Destroy();
		selfRes.Destroy();
		outRes.Destroy();
		return success;
	}
	
	bool TestInvalidNullptrAlpha()
	{
		LOG_PRINT("\n[RUN] Invalid Case: nullptr alpha\n");
		TensorResource selfRes;
		TensorResource otherRes;
		TensorResource outRes;
		bool success = false;
		
		std::vector<int64_t> shape = {2, 2};
		std::vector<float> selfData = {1.f, 2.f, 3.f, 4.f};
		std::vector<float> otherData = {5.f, 6.f, 7.f, 8.f};
		std::vector<float> outData(4, 0.f);
		
		do {
			if (CreateAclTensor(selfData, shape, ACL_FLOAT, &selfRes) != 0) break;
			if (CreateAclTensor(otherData, shape, ACL_FLOAT, &otherRes) != 0) break;
			if (CreateAclTensor(outData, shape, ACL_FLOAT, &outRes) != 0) break;
			success = ExpectAddGetWorkspaceFail("invalid_nullptr_alpha", selfRes.tensor, otherRes.tensor, nullptr, outRes.tensor);
		} while (0);
		
		selfRes.Destroy();
		otherRes.Destroy();
		outRes.Destroy();
		return success;
	}
	
	bool TestInvalidNullptrOut()
	{
		LOG_PRINT("\n[RUN] Invalid Case: nullptr out\n");
		TensorResource selfRes;
		TensorResource otherRes;
		ScalarResource alphaRes;
		bool success = false;
		
		std::vector<int64_t> shape = {2, 2};
		std::vector<float> selfData = {1.f, 2.f, 3.f, 4.f};
		std::vector<float> otherData = {5.f, 6.f, 7.f, 8.f};
		float alpha = 1.0f;
		
		do {
			if (CreateAclTensor(selfData, shape, ACL_FLOAT, &selfRes) != 0) break;
			if (CreateAclTensor(otherData, shape, ACL_FLOAT, &otherRes) != 0) break;
			if (!CreateAclScalar(alpha, ACL_FLOAT, &alphaRes)) break;
			success = ExpectAddGetWorkspaceFail("invalid_nullptr_out", selfRes.tensor, otherRes.tensor, alphaRes.scalar, nullptr);
		} while (0);
		
		alphaRes.Destroy();
		selfRes.Destroy();
		otherRes.Destroy();
		return success;
	}
	
	bool TestInvalidIncompatibleShapes()
	{
		LOG_PRINT("\n[RUN] Invalid Case: incompatible input shapes\n");
		TensorResource selfRes;
		TensorResource otherRes;
		TensorResource outRes;
		ScalarResource alphaRes;
		bool success = false;
		
		std::vector<int64_t> selfShape = {2, 3};
		std::vector<int64_t> otherShape = {2, 2};
		std::vector<int64_t> outShape = {2, 3};
		
		std::vector<float> selfData = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
		std::vector<float> otherData = {1.f, 2.f, 3.f, 4.f};
		std::vector<float> outData(6, 0.f);
		float alpha = 1.0f;
		
		do {
			if (CreateAclTensor(selfData, selfShape, ACL_FLOAT, &selfRes) != 0) break;
			if (CreateAclTensor(otherData, otherShape, ACL_FLOAT, &otherRes) != 0) break;
			if (CreateAclTensor(outData, outShape, ACL_FLOAT, &outRes) != 0) break;
			if (!CreateAclScalar(alpha, ACL_FLOAT, &alphaRes)) break;
			success = ExpectAddGetWorkspaceFail("invalid_incompatible_shapes",
												selfRes.tensor, otherRes.tensor, alphaRes.scalar, outRes.tensor);
		} while (0);
		
		alphaRes.Destroy();
		selfRes.Destroy();
		otherRes.Destroy();
		outRes.Destroy();
		return success;
	}
	
	bool TestInvalidWrongOutShape()
	{
		LOG_PRINT("\n[RUN] Invalid Case: wrong output shape\n");
		TensorResource selfRes;
		TensorResource otherRes;
		TensorResource outRes;
		ScalarResource alphaRes;
		bool success = false;
		
		std::vector<int64_t> selfShape = {2, 3};
		std::vector<int64_t> otherShape = {3};
		std::vector<int64_t> outShape = {3, 2};
		
		std::vector<float> selfData = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
		std::vector<float> otherData = {2.f, 3.f, 4.f};
		std::vector<float> outData(6, 0.f);
		float alpha = 1.0f;
		
		do {
			if (CreateAclTensor(selfData, selfShape, ACL_FLOAT, &selfRes) != 0) break;
			if (CreateAclTensor(otherData, otherShape, ACL_FLOAT, &otherRes) != 0) break;
			if (CreateAclTensor(outData, outShape, ACL_FLOAT, &outRes) != 0) break;
			if (!CreateAclScalar(alpha, ACL_FLOAT, &alphaRes)) break;
			success = ExpectAddGetWorkspaceFail("invalid_wrong_out_shape",
												selfRes.tensor, otherRes.tensor, alphaRes.scalar, outRes.tensor);
		} while (0);
		
		alphaRes.Destroy();
		selfRes.Destroy();
		otherRes.Destroy();
		outRes.Destroy();
		return success;
	}
	
	bool TestInvalidUnsupportedDtypeUInt32()
	{
		LOG_PRINT("\n[RUN] Invalid Case: unsupported dtype uint32\n");
		TensorResource selfRes;
		TensorResource otherRes;
		TensorResource outRes;
		ScalarResource alphaRes;
		bool success = false;
		
		std::vector<int64_t> shape = {2, 2};
		std::vector<uint32_t> selfData = {1, 2, 3, 4};
		std::vector<uint32_t> otherData = {5, 6, 7, 8};
		std::vector<uint32_t> outData(4, 0);
		float alpha = 1.0f;
		
		do {
			if (CreateAclTensor(selfData, shape, ACL_UINT32, &selfRes) != 0) break;
			if (CreateAclTensor(otherData, shape, ACL_UINT32, &otherRes) != 0) break;
			if (CreateAclTensor(outData, shape, ACL_UINT32, &outRes) != 0) break;
			if (!CreateAclScalar(alpha, ACL_FLOAT, &alphaRes)) break;
			success = ExpectAddGetWorkspaceFail("invalid_unsupported_dtype_uint32",
												selfRes.tensor, otherRes.tensor, alphaRes.scalar, outRes.tensor);
		} while (0);
		
		alphaRes.Destroy();
		selfRes.Destroy();
		otherRes.Destroy();
		outRes.Destroy();
		return success;
	}
	
	bool TestInvalidOutputDtypeMismatch()
	{
		LOG_PRINT("\n[RUN] Invalid Case: output dtype mismatch\n");
		TensorResource selfRes;
		TensorResource otherRes;
		TensorResource outRes;
		ScalarResource alphaRes;
		bool success = false;
		
		std::vector<int64_t> shape = {2, 2};
		std::vector<float> selfData = {1.f, 2.f, 3.f, 4.f};
		std::vector<float> otherData = {5.f, 6.f, 7.f, 8.f};
		std::vector<int32_t> outData(4, 0);
		float alpha = 1.0f;
		
		do {
			if (CreateAclTensor(selfData, shape, ACL_FLOAT, &selfRes) != 0) break;
			if (CreateAclTensor(otherData, shape, ACL_FLOAT, &otherRes) != 0) break;
			if (CreateAclTensor(outData, shape, ACL_INT32, &outRes) != 0) break;
			if (!CreateAclScalar(alpha, ACL_FLOAT, &alphaRes)) break;
			success = ExpectAddGetWorkspaceFail("invalid_output_dtype_mismatch",
												selfRes.tensor, otherRes.tensor, alphaRes.scalar, outRes.tensor);
		} while (0);
		
		alphaRes.Destroy();
		selfRes.Destroy();
		otherRes.Destroy();
		outRes.Destroy();
		return success;
	}
	
	bool TestInvalidUnsupportedMixedFloatInt32()
	{
		LOG_PRINT("\n[RUN] Invalid Case: unsupported mixed float/int32\n");
		TensorResource selfRes;
		TensorResource otherRes;
		TensorResource outRes;
		ScalarResource alphaRes;
		bool success = false;
		
		std::vector<int64_t> shape = {2, 2};
		std::vector<float> selfData = {1.f, 2.f, 3.f, 4.f};
		std::vector<int32_t> otherData = {5, 6, 7, 8};
		std::vector<float> outData(4, 0.f);
		float alpha = 1.0f;
		
		do {
			if (CreateAclTensor(selfData, shape, ACL_FLOAT, &selfRes) != 0) break;
			if (CreateAclTensor(otherData, shape, ACL_INT32, &otherRes) != 0) break;
			if (CreateAclTensor(outData, shape, ACL_FLOAT, &outRes) != 0) break;
			if (!CreateAclScalar(alpha, ACL_FLOAT, &alphaRes)) break;
			success = ExpectAddGetWorkspaceFail("invalid_unsupported_mixed_float_int32",
												selfRes.tensor, otherRes.tensor, alphaRes.scalar, outRes.tensor);
		} while (0);
		
		alphaRes.Destroy();
		selfRes.Destroy();
		otherRes.Destroy();
		outRes.Destroy();
		return success;
	}
	
// scalar negative
	bool TestInvalidAddsNullSelf()
	{
		LOG_PRINT("\n[RUN] Invalid Case: adds nullptr self\n");
		TensorResource outRes;
		ScalarResource otherRes;
		ScalarResource alphaRes;
		bool success = false;
		
		std::vector<int64_t> shape = {2, 2};
		std::vector<float> outData(4, 0.f);
		float other = 2.0f;
		float alpha = 1.0f;
		
		do {
			if (CreateAclTensor(outData, shape, ACL_FLOAT, &outRes) != 0) break;
			if (!CreateAclScalar(other, ACL_FLOAT, &otherRes)) break;
			if (!CreateAclScalar(alpha, ACL_FLOAT, &alphaRes)) break;
			success = ExpectAddsGetWorkspaceFail("invalid_adds_null_self", nullptr, otherRes.scalar, alphaRes.scalar, outRes.tensor);
		} while (0);
		
		alphaRes.Destroy();
		otherRes.Destroy();
		outRes.Destroy();
		return success;
	}
	
	bool TestInvalidAddsNullOther()
	{
		LOG_PRINT("\n[RUN] Invalid Case: adds nullptr other scalar\n");
		TensorResource selfRes;
		TensorResource outRes;
		ScalarResource alphaRes;
		bool success = false;
		
		std::vector<int64_t> shape = {2, 2};
		std::vector<float> selfData = {1.f, 2.f, 3.f, 4.f};
		std::vector<float> outData(4, 0.f);
		float alpha = 1.0f;
		
		do {
			if (CreateAclTensor(selfData, shape, ACL_FLOAT, &selfRes) != 0) break;
			if (CreateAclTensor(outData, shape, ACL_FLOAT, &outRes) != 0) break;
			if (!CreateAclScalar(alpha, ACL_FLOAT, &alphaRes)) break;
			success = ExpectAddsGetWorkspaceFail("invalid_adds_null_other", selfRes.tensor, nullptr, alphaRes.scalar, outRes.tensor);
		} while (0);
		
		alphaRes.Destroy();
		selfRes.Destroy();
		outRes.Destroy();
		return success;
	}
	
	bool TestInvalidAddsNullAlpha()
	{
		LOG_PRINT("\n[RUN] Invalid Case: adds nullptr alpha\n");
		TensorResource selfRes;
		TensorResource outRes;
		ScalarResource otherRes;
		bool success = false;
		
		std::vector<int64_t> shape = {2, 2};
		std::vector<float> selfData = {1.f, 2.f, 3.f, 4.f};
		std::vector<float> outData(4, 0.f);
		float other = 2.0f;
		
		do {
			if (CreateAclTensor(selfData, shape, ACL_FLOAT, &selfRes) != 0) break;
			if (CreateAclTensor(outData, shape, ACL_FLOAT, &outRes) != 0) break;
			if (!CreateAclScalar(other, ACL_FLOAT, &otherRes)) break;
			success = ExpectAddsGetWorkspaceFail("invalid_adds_null_alpha", selfRes.tensor, otherRes.scalar, nullptr, outRes.tensor);
		} while (0);
		
		otherRes.Destroy();
		selfRes.Destroy();
		outRes.Destroy();
		return success;
	}
	
	bool TestInvalidAddsNullOut()
	{
		LOG_PRINT("\n[RUN] Invalid Case: adds nullptr out\n");
		TensorResource selfRes;
		ScalarResource otherRes;
		ScalarResource alphaRes;
		bool success = false;
		
		std::vector<int64_t> shape = {2, 2};
		std::vector<float> selfData = {1.f, 2.f, 3.f, 4.f};
		float other = 2.0f;
		float alpha = 1.0f;
		
		do {
			if (CreateAclTensor(selfData, shape, ACL_FLOAT, &selfRes) != 0) break;
			if (!CreateAclScalar(other, ACL_FLOAT, &otherRes)) break;
			if (!CreateAclScalar(alpha, ACL_FLOAT, &alphaRes)) break;
			success = ExpectAddsGetWorkspaceFail("invalid_adds_null_out", selfRes.tensor, otherRes.scalar, alphaRes.scalar, nullptr);
		} while (0);
		
		alphaRes.Destroy();
		otherRes.Destroy();
		selfRes.Destroy();
		return success;
	}
	
	bool TestInvalidAddsOutputDtypeMismatch()
	{
		LOG_PRINT("\n[RUN] Invalid Case: adds output dtype mismatch\n");
		TensorResource selfRes;
		TensorResource outRes;
		ScalarResource otherRes;
		ScalarResource alphaRes;
		bool success = false;
		
		std::vector<int64_t> shape = {2, 2};
		std::vector<float> selfData = {1.f, 2.f, 3.f, 4.f};
		std::vector<int32_t> outData(4, 0);
		float other = 2.0f;
		float alpha = 1.0f;
		
		do {
			if (CreateAclTensor(selfData, shape, ACL_FLOAT, &selfRes) != 0) break;
			if (CreateAclTensor(outData, shape, ACL_INT32, &outRes) != 0) break;
			if (!CreateAclScalar(other, ACL_FLOAT, &otherRes)) break;
			if (!CreateAclScalar(alpha, ACL_FLOAT, &alphaRes)) break;
			success = ExpectAddsGetWorkspaceFail("invalid_adds_output_dtype_mismatch",
												 selfRes.tensor, otherRes.scalar, alphaRes.scalar, outRes.tensor);
		} while (0);
		
		alphaRes.Destroy();
		otherRes.Destroy();
		selfRes.Destroy();
		outRes.Destroy();
		return success;
	}
	
	bool TestInvalidInplaceAddsNullSelf()
	{
		LOG_PRINT("\n[RUN] Invalid Case: inplace adds nullptr self\n");
		ScalarResource otherRes;
		ScalarResource alphaRes;
		bool success = false;
		float other = 2.0f;
		float alpha = 1.0f;
		
		do {
			if (!CreateAclScalar(other, ACL_FLOAT, &otherRes)) break;
			if (!CreateAclScalar(alpha, ACL_FLOAT, &alphaRes)) break;
			success = ExpectInplaceAddsGetWorkspaceFail("invalid_inplace_adds_null_self", nullptr, otherRes.scalar, alphaRes.scalar);
		} while (0);
		
		alphaRes.Destroy();
		otherRes.Destroy();
		return success;
	}
	
	bool TestInvalidInplaceAddsNullOther()
	{
		LOG_PRINT("\n[RUN] Invalid Case: inplace adds nullptr other\n");
		TensorResource selfRes;
		ScalarResource alphaRes;
		bool success = false;
		std::vector<int64_t> shape = {2, 2};
		std::vector<float> selfData = {1.f, 2.f, 3.f, 4.f};
		float alpha = 1.0f;
		
		do {
			if (CreateAclTensor(selfData, shape, ACL_FLOAT, &selfRes) != 0) break;
			if (!CreateAclScalar(alpha, ACL_FLOAT, &alphaRes)) break;
			success = ExpectInplaceAddsGetWorkspaceFail("invalid_inplace_adds_null_other", selfRes.tensor, nullptr, alphaRes.scalar);
		} while (0);
		
		alphaRes.Destroy();
		selfRes.Destroy();
		return success;
	}
	
	bool TestInvalidInplaceAddsNullAlpha()
	{
		LOG_PRINT("\n[RUN] Invalid Case: inplace adds nullptr alpha\n");
		TensorResource selfRes;
		ScalarResource otherRes;
		bool success = false;
		std::vector<int64_t> shape = {2, 2};
		std::vector<float> selfData = {1.f, 2.f, 3.f, 4.f};
		float other = 2.0f;
		
		do {
			if (CreateAclTensor(selfData, shape, ACL_FLOAT, &selfRes) != 0) break;
			if (!CreateAclScalar(other, ACL_FLOAT, &otherRes)) break;
			success = ExpectInplaceAddsGetWorkspaceFail("invalid_inplace_adds_null_alpha", selfRes.tensor, otherRes.scalar, nullptr);
		} while (0);
		
		otherRes.Destroy();
		selfRes.Destroy();
		return success;
	}
	
	bool TestInvalidInplaceAddNullOther()
	{
		LOG_PRINT("\n[RUN] Invalid Case: inplace add nullptr other\n");
		TensorResource selfRes;
		ScalarResource alphaRes;
		bool success = false;
		std::vector<int64_t> shape = {2, 2};
		std::vector<float> selfData = {1.f, 2.f, 3.f, 4.f};
		float alpha = 1.0f;
		
		do {
			if (CreateAclTensor(selfData, shape, ACL_FLOAT, &selfRes) != 0) break;
			if (!CreateAclScalar(alpha, ACL_FLOAT, &alphaRes)) break;
			success = ExpectInplaceAddGetWorkspaceFail("invalid_inplace_add_null_other", selfRes.tensor, nullptr, alphaRes.scalar);
		} while (0);
		
		alphaRes.Destroy();
		selfRes.Destroy();
		return success;
	}
	
	bool TestInvalidAddV3NullSelf()
	{
		LOG_PRINT("\n[RUN] Invalid Case: addv3 nullptr self\n");
		TensorResource otherRes;
		TensorResource outRes;
		ScalarResource alphaRes;
		bool success = false;
		
		std::vector<int64_t> shape = {2, 2};
		std::vector<float> otherData = {1.f, 2.f, 3.f, 4.f};
		std::vector<float> outData(4, 0.f);
		float alpha = 1.0f;
		
		do {
			if (CreateAclTensor(otherData, shape, ACL_FLOAT, &otherRes) != 0) break;
			if (CreateAclTensor(outData, shape, ACL_FLOAT, &outRes) != 0) break;
			if (!CreateAclScalar(alpha, ACL_FLOAT, &alphaRes)) break;
			success = ExpectAddV3GetWorkspaceFail("invalid_addv3_null_self", nullptr, otherRes.tensor, alphaRes.scalar, outRes.tensor);
		} while (0);
		
		alphaRes.Destroy();
		otherRes.Destroy();
		outRes.Destroy();
		return success;
	}
	
	bool TestInvalidAddV3NullOther()
	{
		LOG_PRINT("\n[RUN] Invalid Case: addv3 nullptr other\n");
		TensorResource outRes;
		ScalarResource selfRes;
		ScalarResource alphaRes;
		bool success = false;
		
		std::vector<int64_t> shape = {2, 2};
		std::vector<float> outData(4, 0.f);
		float selfScalar = 1.0f;
		float alpha = 1.0f;
		
		do {
			if (CreateAclTensor(outData, shape, ACL_FLOAT, &outRes) != 0) break;
			if (!CreateAclScalar(selfScalar, ACL_FLOAT, &selfRes)) break;
			if (!CreateAclScalar(alpha, ACL_FLOAT, &alphaRes)) break;
			success = ExpectAddV3GetWorkspaceFail("invalid_addv3_null_other", selfRes.scalar, nullptr, alphaRes.scalar, outRes.tensor);
		} while (0);
		
		alphaRes.Destroy();
		selfRes.Destroy();
		outRes.Destroy();
		return success;
	}
	
	bool TestInvalidAddV3NullOut()
	{
		LOG_PRINT("\n[RUN] Invalid Case: addv3 nullptr out\n");
		TensorResource otherRes;
		ScalarResource selfRes;
		ScalarResource alphaRes;
		bool success = false;
		
		std::vector<int64_t> shape = {2, 2};
		std::vector<float> otherData = {1.f, 2.f, 3.f, 4.f};
		float selfScalar = 1.0f;
		float alpha = 1.0f;
		
		do {
			if (CreateAclTensor(otherData, shape, ACL_FLOAT, &otherRes) != 0) break;
			if (!CreateAclScalar(selfScalar, ACL_FLOAT, &selfRes)) break;
			if (!CreateAclScalar(alpha, ACL_FLOAT, &alphaRes)) break;
			success = ExpectAddV3GetWorkspaceFail("invalid_addv3_null_out", selfRes.scalar, otherRes.tensor, alphaRes.scalar, nullptr);
		} while (0);
		
		alphaRes.Destroy();
		selfRes.Destroy();
		otherRes.Destroy();
		return success;
	}
	AddExecCase<float, float, float, float> BuildLargeFloatCase()
	{
		AddExecCase<float, float, float, float> tc;
		tc.name = "add_float_large_tensor_tiling";
		tc.selfShape = {32, 64};
		tc.otherShape = {32, 64};
		tc.outShape = {32, 64};
		tc.alpha = 1.2f;
		tc.selfType = ACL_FLOAT;
		tc.otherType = ACL_FLOAT;
		tc.alphaType = ACL_FLOAT;
		tc.outType = ACL_FLOAT;
		tc.atol = 1e-5;
		tc.rtol = 1e-5;
		
		int64_t numel = GetShapeSize(tc.selfShape);
		tc.selfData.resize(static_cast<size_t>(numel));
		tc.otherData.resize(static_cast<size_t>(numel));
		for (int64_t i = 0; i < numel; ++i) {
			tc.selfData[static_cast<size_t>(i)] = static_cast<float>((i % 29) - 14) / 3.0f;
			tc.otherData[static_cast<size_t>(i)] = static_cast<float>((i % 17) - 8) / 5.0f;
		}
		return tc;
	}
	
	AddExecCase<uint16_t, uint16_t, float, uint16_t> BuildFloat16SameTypeCase()
	{
		AddExecCase<uint16_t, uint16_t, float, uint16_t> tc;
		tc.name = "add_float16_same_shape";
		tc.selfShape = {2, 4};
		tc.otherShape = {2, 4};
		tc.outShape = {2, 4};
		tc.alpha = 1.0f;
		tc.selfType = ACL_FLOAT16;
		tc.otherType = ACL_FLOAT16;
		tc.alphaType = ACL_FLOAT;
		tc.outType = ACL_FLOAT16;
		tc.atol = 1e-3;
		tc.rtol = 1e-3;
		
		tc.selfData = MakeFloat16Vector({1.0f, -2.0f, 3.5f, 0.5f, -0.25f, 8.0f, 0.0f, 1.5f});
		tc.otherData = MakeFloat16Vector({2.0f, 0.5f, -1.0f, 4.0f, -8.0f, 0.25f, 3.0f, -2.0f});
		return tc;
	}
	
	AddExecCase<uint16_t, float, float, float> BuildFloat16FloatMixedCase()
	{
		AddExecCase<uint16_t, float, float, float> tc;
		tc.name = "add_float16_float_mixed";
		tc.selfShape = {4};
		tc.otherShape = {4};
		tc.outShape = {4};
		tc.alpha = 0.5f;
		tc.selfType = ACL_FLOAT16;
		tc.otherType = ACL_FLOAT;
		tc.alphaType = ACL_FLOAT;
		tc.outType = ACL_FLOAT;
		tc.atol = 1e-5;
		tc.rtol = 1e-5;
		
		tc.selfData = MakeFloat16Vector({1.5f, -2.0f, 0.25f, 4.0f});
		tc.otherData = {2.0f, 3.0f, -8.0f, 0.5f};
		return tc;
	}
	
	AddExecCase<float, uint16_t, float, float> BuildFloatBf16MixedCase()
	{
		AddExecCase<float, uint16_t, float, float> tc;
		tc.name = "add_float_bf16_mixed";
		tc.selfShape = {4};
		tc.otherShape = {4};
		tc.outShape = {4};
		tc.alpha = -2.0f;
		tc.selfType = ACL_FLOAT;
		tc.otherType = ACL_BF16;
		tc.alphaType = ACL_FLOAT;
		tc.outType = ACL_FLOAT;
		tc.atol = 1e-5;
		tc.rtol = 1e-5;
		
		tc.selfData = {2.0f, -3.0f, 8.0f, 0.125f};
		tc.otherData = MakeBFloat16Vector({1.5f, 2.0f, -0.5f, 16.0f});
		return tc;
	}
	
	AddExecCase<double, double, float, double> BuildDoubleSpecialCase()
	{
		const double nan = std::numeric_limits<double>::quiet_NaN();
		const double inf = std::numeric_limits<double>::infinity();
		
		AddExecCase<double, double, float, double> tc;
		tc.name = "add_double_special_values";
		tc.selfShape = {5};
		tc.otherShape = {5};
		tc.outShape = {5};
		tc.alpha = 1.0f;
		tc.selfType = ACL_DOUBLE;
		tc.otherType = ACL_DOUBLE;
		tc.alphaType = ACL_FLOAT;
		tc.outType = ACL_DOUBLE;
		tc.atol = 1e-12;
		tc.rtol = 1e-12;
		
		tc.selfData = {nan, inf, -1.5, 0.0, -inf};
		tc.otherData = {2.0, -3.0, inf, nan, -1.0};
		return tc;
	}
	AddV3ExecCase<float, float, float, float> BuildAddV3FloatBasicCase()
	{
		AddV3ExecCase<float, float, float, float> tc;
		tc.name = "addv3_float_basic";
		tc.selfScalar = 5.0f;
		tc.otherShape = {2, 2};
		tc.otherData = {1.f, 2.f, 3.f, 4.f};
		tc.alpha = 1.0f;
		tc.selfScalarType = ACL_FLOAT;
		tc.otherType = ACL_FLOAT;
		tc.alphaType = ACL_FLOAT;
		tc.outType = ACL_FLOAT;
		tc.atol = 1e-5;
		tc.rtol = 1e-5;
		return tc;
	}
	
	AddV3ExecCase<float, float, float, float> BuildAddV3Float1DCase()
	{
		AddV3ExecCase<float, float, float, float> tc;
		tc.name = "addv3_float_1d";
		tc.selfScalar = 2.0f;
		tc.otherShape = {5};
		tc.otherData = {1.f, 2.f, 3.f, 4.f, 5.f};
		tc.alpha = 1.0f;
		tc.selfScalarType = ACL_FLOAT;
		tc.otherType = ACL_FLOAT;
		tc.alphaType = ACL_FLOAT;
		tc.outType = ACL_FLOAT;
		tc.atol = 1e-5;
		tc.rtol = 1e-5;
		return tc;
	}
	
	AddV3ExecCase<float, float, float, float> BuildAddV3FloatLargeCase()
	{
		AddV3ExecCase<float, float, float, float> tc;
		tc.name = "addv3_float_large_tensor";
		tc.selfScalar = 1.0f;
		tc.otherShape = {32, 64};
		tc.alpha = 0.75f;
		tc.selfScalarType = ACL_FLOAT;
		tc.otherType = ACL_FLOAT;
		tc.alphaType = ACL_FLOAT;
		tc.outType = ACL_FLOAT;
		tc.atol = 1e-5;
		tc.rtol = 1e-5;
		
		int64_t numel = GetShapeSize(tc.otherShape);
		tc.otherData.resize(static_cast<size_t>(numel));
		for (int64_t i = 0; i < numel; ++i) {
			tc.otherData[static_cast<size_t>(i)] = static_cast<float>((i % 31) - 15) / 4.0f;
		}
		return tc;
	}
	bool RunNonContiguousAddCase(aclrtStream stream)
	{
		const std::string caseName = "add_float_non_contiguous_self_tensor";
		LOG_PRINT("\n[RUN] Add Exec Case: %s\n", caseName.c_str());
		
		TensorResource selfRes;
		TensorResource otherRes;
		TensorResource outRes;
		ScalarResource alphaRes;
		void* workspaceAddr = nullptr;
		uint64_t workspaceSize = 0;
		bool success = false;
		
		auto Cleanup = [&]() {
			if (workspaceAddr != nullptr) {
				aclrtFree(workspaceAddr);
				workspaceAddr = nullptr;
			}
			alphaRes.Destroy();
			selfRes.Destroy();
			otherRes.Destroy();
			outRes.Destroy();
		};
		
		do {
			std::vector<int64_t> selfViewShape = {2, 3};
			std::vector<int64_t> selfStorageShape = {2, 4};
			std::vector<int64_t> selfStrides = {4, 1};
			std::vector<float> selfStorageData = {
				1.0f, 2.0f, 3.0f, 99.0f,
				4.0f, 5.0f, 6.0f, 99.0f
			};
			
			std::vector<int64_t> otherShape = {2, 3};
			std::vector<float> otherData = {
				2.0f, 2.0f, 2.0f,
				2.0f, 2.0f, 2.0f
			};
			
			std::vector<int64_t> outShape = {2, 3};
			std::vector<float> outInit(6, 0.0f);
			std::vector<float> expect = {
				3.0f, 4.0f, 5.0f,
				6.0f, 7.0f, 8.0f
			};
			float alpha = 1.0f;
			
			int ret = CreateAclTensorWithCustomLayout(selfStorageData, selfViewShape, selfStorageShape,
													  selfStrides, ACL_FLOAT, &selfRes);
			if (ret != 0) break;
			ret = CreateAclTensor(otherData, otherShape, ACL_FLOAT, &otherRes);
			if (ret != 0) break;
			ret = CreateAclTensor(outInit, outShape, ACL_FLOAT, &outRes);
			if (ret != 0) break;
			if (!CreateAclScalar(alpha, ACL_FLOAT, &alphaRes)) break;
			
			aclOpExecutor* executor = nullptr;
			auto aclRet = aclnnAddGetWorkspaceSize(selfRes.tensor, otherRes.tensor, alphaRes.scalar, outRes.tensor,
												   &workspaceSize, &executor);
			if (aclRet != ACL_SUCCESS) {
				LOG_PRINT("[FAIL] %s: aclnnAddGetWorkspaceSize failed. ERROR: %d\n",
						  caseName.c_str(),                      static_cast<int>(aclRet));
				break;
			}
			
			if (workspaceSize > 0) {
				ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
				if (ret != ACL_SUCCESS) {
					LOG_PRINT("[FAIL] %s: workspace malloc failed. ERROR: %d\n",
							  caseName.c_str(), ret);
					break;
				}
			}
			
			aclRet = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
			if (aclRet != ACL_SUCCESS) {
				LOG_PRINT("[FAIL] %s: aclnnAdd execute failed. ERROR: %d\n",
						  caseName.c_str(), static_cast<int>(aclRet));
				break;
			}
			
			ret = aclrtSynchronizeStream(stream);
			if (ret != ACL_SUCCESS) {
				LOG_PRINT("[FAIL] %s: aclrtSynchronizeStream failed. ERROR: %d\n",
						  caseName.c_str(), ret);
				break;
			}
			
			std::vector<float> actual;
			ret = CopyDeviceToHost(outRes, &actual);
			if (ret != 0) break;
			
			if (!CompareVectors(actual, expect, ACL_FLOAT, 1e-5, 1e-5, caseName)) {
				break;
			}
			
			success = true;
		} while (0);
		
		Cleanup();
		LOG_PRINT(success ? "[PASS] %s\n" : "[FAIL] %s\n", caseName.c_str());
		return success;
	}
}
	int main()
	{
		int32_t deviceId = 0;
		aclrtStream stream = nullptr;
		
		auto ret = Init(deviceId, &stream);
		CHECK_RET(ret == ACL_SUCCESS,
				  LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
				  return ret);
		
		int total = 0;
		int passed = 0;
		
		auto CountCase = [&](bool ok) {
			++total;
			if (ok) {
				++passed;
			}
		};
		
		LOG_PRINT("\n==============================\n");
		LOG_PRINT("Add Example Test Start\n");
		LOG_PRINT("==============================\n");
		
		// ----------------------------
		// tensor-tensor: aclnnAdd
		// ----------------------------
		CountCase(RunAddExecCase<float, float, float, float>(
															 {"add_float_basic_same_shape",
																 {4, 2}, {4, 2}, {4, 2},
																 {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f},
																 {1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 3.f, 3.f},
																 1.2f,
																 ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
																 1e-5, 1e-5},
																 stream));
		
		CountCase(RunAddExecCase<float, float, float, float>(
															 {"add_float_broadcast_2x3_plus_3",
																 {2, 3}, {3}, {2, 3},
																 {1.f, -2.f, 3.f, 4.f, -5.f, 6.f},
																 {2.f, 0.5f, -1.f},
																 1.0f,
																 ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
																 1e-5, 1e-5},
																 stream));
		
		CountCase(RunAddExecCase<float, float, float, float>(
															 {"add_float_empty_tensor",
																 {0, 3}, {1, 3}, {0, 3},
																 {},
																 {1.f, 2.f, 3.f},
																 1.0f,
																 ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
																 1e-5, 1e-5},
																 stream));
		
		CountCase(RunAddExecCase<float, float, float, float>(BuildLargeFloatCase(), stream));
		CountCase(RunAddExecCase<uint16_t, uint16_t, float, uint16_t>(BuildFloat16SameTypeCase(), stream));
		CountCase(RunAddExecCase<uint16_t, float, float, float>(BuildFloat16FloatMixedCase(), stream));
		CountCase(RunAddExecCase<float, uint16_t, float, float>(BuildFloatBf16MixedCase(), stream));
		CountCase(RunAddExecCase<double, double, float, double>(BuildDoubleSpecialCase(), stream));
		CountCase(RunNonContiguousAddCase(stream));
		
		CountCase(RunAddExecCase<int32_t, int32_t, int32_t, int32_t>(
																	 {"add_int32_same_shape_alpha2",
																		 {6}, {6}, {6},
																		 {1000, -200, 0, 13, 9, -7},
																		 {3, -10, 5, -4, 0, 8},
																		 2,
																		 ACL_INT32, ACL_INT32, ACL_INT32, ACL_INT32,
																		 -1.0, -1.0},
																		 stream));
		
		CountCase(RunAddExecCase<int16_t, int16_t, int16_t, int16_t>(
																	 {"add_int16_same_shape_alpha_minus1",
																		 {6}, {6}, {6},
																		 {100, -20, 0, 13, 9, -7},
																		 {3, -10, 5, -4, 0, 8},
																		 static_cast<int16_t>(-1),
																		 ACL_INT16, ACL_INT16, ACL_INT16, ACL_INT16,
																		 -1.0, -1.0},
																		 stream));
		
		CountCase(RunAddExecCase<int64_t, int64_t, float, int64_t>(
																   {"add_int64_same_shape_alpha1",
																	   {4}, {4}, {4},
																	   {100000LL, -2000LL, 0LL, 13LL},
																	   {30LL, -5LL, 7LL, -9LL},
																	   1.0f,
																	   ACL_INT64, ACL_INT64, ACL_FLOAT, ACL_INT64,
																	   -1.0, -1.0},
																	   stream));
		
		CountCase(RunAddExecCase<uint8_t, uint8_t, float, uint8_t>(
																   {"add_uint8_alpha1",
																	   {6}, {6}, {6},
																	   {10, 20, 100, 200, 255, 0},
																	   {20, 13, 3, 2, 2, 5},
																	   1.0f,
																	   ACL_UINT8, ACL_UINT8, ACL_FLOAT, ACL_UINT8,
																	   -1.0, -1.0},
																	   stream));
		
		CountCase(RunAddExecCase<int8_t, int8_t, float, int8_t>(
																{"add_int8_alpha1",
																	{7}, {7}, {7},
																	{100, -100, 60, -60, 127, -128, 10},
																	{2, 2, 3, 3, 2, 2, -13},
																	1.0f,
																	ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_INT8,
																	-1.0, -1.0},
																	stream));
		
		CountCase(RunAddExecCase<uint8_t, uint8_t, float, uint8_t>(
																   {"add_bool_same_shape",
																	   {6}, {6}, {6},
																	   {1, 0, 1, 1, 0, 1},
																	   {1, 1, 0, 1, 0, 0},
																	   1.0f,
																	   ACL_BOOL, ACL_BOOL, ACL_FLOAT, ACL_BOOL,
																	   -1.0, -1.0},
																	   stream));
		
		CountCase(RunAddExecCase<float, float, float, float>(
															 {"add_float_alpha_zero",
																 {4}, {4}, {4},
																 {1.f, 2.f, 3.f, 4.f},
																 {10.f, 20.f, 30.f, 40.f},
																 0.0f,
																 ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
																 1e-5, 1e-5},
																 stream));
		
		CountCase(RunAddExecCase<float, float, float, float>(
															 {"add_float_alpha_negative",
																 {4}, {4}, {4},
																 {1.f, 2.f, 3.f, 4.f},
																 {10.f, 20.f, 30.f, 40.f},
																 -0.5f,
																 ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
																 1e-5, 1e-5},
																 stream));
		
		// ----------------------------
		// tensor-scalar: aclnnAdds
		// ----------------------------
		CountCase(RunAddsExecCase<float, float, float, float>(
															  {"adds_float_scalar",
																  {2, 3},
																  {1.f, -2.f, 3.f, 4.f, -5.f, 6.f},
																  2.0f,
																  1.5f,
																  ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
																  1e-5, 1e-5},
																  stream));
		
		CountCase(RunAddsExecCase<int32_t, int32_t, int32_t, int32_t>(
																	  {"adds_int32_scalar",
																		  {5},
																		  {1, -2, 3, 4, -5},
																		  7,
																		  2,
																		  ACL_INT32, ACL_INT32, ACL_INT32, ACL_INT32,
																		  -1.0, -1.0},
																		  stream));
		
		CountCase(RunAddsExecCase<uint16_t, float, float, float>(
																 {"adds_float16_scalar_to_float",
																	 {4},
																	 MakeFloat16Vector({1.5f, -2.0f, 0.25f, 4.0f}),
																	 2.0f,
																	 0.5f,
																	 ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
																	 1e-5, 1e-5},
																	 stream));
		
		CountCase(RunAddsExecCase<uint16_t, uint16_t, uint16_t, uint16_t>(
																		  {"adds_bf16_scalar",
																			  {4},
																			  MakeBFloat16Vector({1.0f, -2.0f, 3.5f, 0.5f}),
																			  FloatToBf16Bits(2.0f),
																			  FloatToBf16Bits(1.0f),
																			  ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16,
																			  1e-2, 1e-2},
																			  stream));
		
		
		CountCase(RunAddV3ExecCase<float, float, float, float>(BuildAddV3FloatBasicCase(), stream));
		CountCase(RunAddV3ExecCase<float, float, float, float>(BuildAddV3Float1DCase(), stream));
		CountCase(RunAddV3ExecCase<float, float, float, float>(BuildAddV3FloatLargeCase(), stream));
		
		// ----------------------------
		// inplace tensor-tensor
		// ----------------------------
		CountCase(RunInplaceAddExecCase<float, float, float>(
															 {"inplace_add_float_broadcast",
																 {2, 3}, {3},
																 {1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
																 {2.f, -1.f, 0.5f},
																 1.0f,
																 ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
																 1e-5, 1e-5},
																 stream));
		
		CountCase(RunInplaceAddExecCase<int32_t, int32_t, int32_t>(
																   {"inplace_add_int32_same_shape",
																	   {4}, {4},
																	   {10, -2, 3, 4},
																	   {-3, 5, 0, 7},
																	   2,
																	   ACL_INT32, ACL_INT32, ACL_INT32,
																	   -1.0, -1.0},
																	   stream));
		
		// ----------------------------
		// inplace tensor-scalar
		// ----------------------------
		CountCase(RunInplaceAddsExecCase<float, float, float>(
															  {"inplace_adds_float_scalar",
																  {4},
																  {1.f, -2.f, 3.5f, 0.f},
																  -3.0f,
																  2.0f,
																  ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
																  1e-5, 1e-5},
																  stream));
		
		CountCase(RunInplaceAddsExecCase<int16_t, int16_t, int16_t>(
																	{"inplace_adds_int16_scalar",
																		{5},
																		{1, -2, 3, 4, -5},
																		static_cast<int16_t>(2),
																		static_cast<int16_t>(3),
																		ACL_INT16, ACL_INT16, ACL_INT16,
																		-1.0, -1.0},
																		stream));
		
		// ----------------------------
		// invalid tensor-tensor
		// ----------------------------
		CountCase(TestInvalidNullptrSelf());
		CountCase(TestInvalidNullptrOther());
		CountCase(TestInvalidNullptrAlpha());
		CountCase(TestInvalidNullptrOut());
		CountCase(TestInvalidIncompatibleShapes());
		CountCase(TestInvalidWrongOutShape());
		CountCase(TestInvalidUnsupportedDtypeUInt32());
		CountCase(TestInvalidOutputDtypeMismatch());
		CountCase(TestInvalidUnsupportedMixedFloatInt32());
		
		// ----------------------------
		// invalid tensor-scalar / inplace
		// ----------------------------
		CountCase(TestInvalidAddsNullSelf());
		CountCase(TestInvalidAddsNullOther());
		CountCase(TestInvalidAddsNullAlpha());
		CountCase(TestInvalidAddsNullOut());
		CountCase(TestInvalidAddsOutputDtypeMismatch());
		CountCase(TestInvalidInplaceAddsNullSelf());
		CountCase(TestInvalidInplaceAddsNullOther());
		CountCase(TestInvalidInplaceAddsNullAlpha());
		CountCase(TestInvalidInplaceAddNullOther());
		CountCase(TestInvalidAddV3NullSelf());
		CountCase(TestInvalidAddV3NullOther());
		CountCase(TestInvalidAddV3NullOut());
		
		LOG_PRINT("\n==============================\n");
		LOG_PRINT("Add Example Test Summary: %d / %d passed\n", passed, total);
		LOG_PRINT("==============================\n");
		
		Finalize(deviceId, stream);
		return (passed == total) ? 0 : 1;
	}
