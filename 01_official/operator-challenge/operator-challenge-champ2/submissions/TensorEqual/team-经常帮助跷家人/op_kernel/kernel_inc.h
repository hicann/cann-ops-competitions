#include "kernel_operator.h"
using namespace AscendC;
using std::is_same_v;

constexpr uint64_t BUFFER_NUM(2);

#define DUMP(tensor)     DumpTensor(tensor, __LINE__, 16)
#define DUMPN(tensor, n) DumpTensor(tensor, __LINE__, n)
#define PRINTF(fmt, ...)                                                                           \
    printf("================LINE %d================\n" fmt, __LINE__, ##__VA_ARGS__);

#define SCAST         static_cast
#define VCAST(T, vec) vec.template ReinterpretCast<T>()
#define ALLOC(T, que) que.template AllocTensor<T>()
constexpr int MAX_SHAPE(4);
constexpr int DIM_LAST(MAX_SHAPE - 1);

template <class T> constexpr static int64_t block_n  = 32 / sizeof(T);
template <class T> constexpr static int64_t repeat_n = 256 / sizeof(T);

template <class T>
constexpr static bool is_casting = (is_same_v<T, bfloat16_t> || is_same_v<T, float16_t> || is_same_v<T, int16_t> || is_same_v<T, int32_t>);

template <class T>
constexpr static bool is_casting_8 = (is_same_v<T, int8_t> || is_same_v<T, uint8_t>);

