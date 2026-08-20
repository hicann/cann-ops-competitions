#include "kernel_operator.h"
using namespace AscendC;

#define PRINTF(fmt, ...) \
    printf("================LINE %d================\n" fmt, __LINE__, ##__VA_ARGS__)

#define SCAST static_cast
