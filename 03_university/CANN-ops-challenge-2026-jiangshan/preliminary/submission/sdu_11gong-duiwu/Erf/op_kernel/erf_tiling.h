// Tiling struct — host pre-computes core distribution
#pragma once
#include <cstdint>

constexpr uint32_t ERF_ALIGN_NUM = 8;
constexpr uint32_t ERF_TILE_LENGTH = 8192;

#ifndef ERF_PROBE_MODE
#define ERF_PROBE_MODE 8
#endif

#ifndef ERF_PROBE_SPIN
#define ERF_PROBE_SPIN 256
#endif

enum ErfProbeMode : uint32_t {
    ERF_PROBE_NONE = 0,
    ERF_PROBE_FIXED_SCALAR = 1,
    ERF_PROBE_PER_TILE_SCALAR = 2,
    ERF_PROBE_EXTRA_VECTOR = 3,
    ERF_PROBE_SCALAR_COPY = 4,
    ERF_PROBE_FORCE_SINGLE_CORE = 5,
    ERF_PROBE_FORCE_MAX_CORES = 6,
    ERF_PROBE_FORCE_EXACT_CORES = 7,
    ERF_PROBE_SUBTILE_EIGHT_CORES = 8,
    ERF_PROBE_SUBTILE_SIXTEEN_CORES = 9,
};

constexpr uint32_t kErfProbeMode = ERF_PROBE_MODE;
constexpr uint32_t kErfProbeSpin = ERF_PROBE_SPIN;

struct ErfTilingData {
    uint64_t totalLength;      // total element count
    uint64_t coreDataLength;   // per-core element count (aligned to host granularity)
    uint32_t coreNum;          // actual number of cores used
    uint32_t tileLength;       // per-data-point optimized tile size
    uint32_t fusionMode;       // reserved: 0 = erf, 1 = gelu
};
