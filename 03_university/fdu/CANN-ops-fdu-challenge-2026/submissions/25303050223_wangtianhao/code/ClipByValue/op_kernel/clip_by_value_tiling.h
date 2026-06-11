#pragma once

#include <cstdint>

struct ClipByValueTilingData
{
	uint32_t length;
	uint32_t blockLength;
	uint32_t coreNum;
	float minValue;
	float maxValue;
};
