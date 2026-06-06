// Tiling ABI marker for profile-only BatchToSpace dispatch.
# pragma once

# include <cstdint>

struct BatchToSpaceTilingData final {
    uint32_t abi_guard;
};
