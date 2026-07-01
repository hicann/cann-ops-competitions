#pragma once

#include <cstdint>

struct ErfTilingData {
    uint32_t coreLenAligned;
    uint32_t fullCoreNum;
    uint32_t tailLen;
    uint32_t tailNeedPad;
};
