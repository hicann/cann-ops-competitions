#pragma once

  #include <cstdint>

  struct ErfTilingData {
      uint32_t totalLength;
      uint32_t usedCoreNum;
      uint32_t largeCoreLength;
      uint32_t smallCoreLength;
      uint32_t largeCoreCount;
      uint32_t tailLength;
      uint32_t tileLength;
  };
