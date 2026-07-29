#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include <type_traits>

// Marcos
// Tensor Allocation stuff
#define ALLOCBUF(type, name, bufname)   LocalTensor<type> name = bufname.template Get<type>()
#define _ENQUE(type, qname, var)        qname.template EnQue<type>(var)
#define ENQUE(type, name)               _ENQUE(type, q_##name, name)
#define _DEQ(type, qname, name)         name = qname.template DeQue<type>()
#define _DEQ_DEFINE(type, qname, name)  LocalTensor<type> name = qname.template DeQue<type>()
#define _GET(type, bname, var)          var = bname.template Get<type>()
#define _GET_DEFINE(type, bname, var)   LocalTensor<type> var = bname.template Get<type>()
#define _ALLOC(type, qname, var)        var = qname.template AllocTensor<type>()
#define _ALLOC_DEFINE(type, qname, var) LocalTensor<type> var = qname.template AllocTensor<type>()
#define _FREE(qname, var)               qname.FreeTensor(var)
#define DEQUE(type, name)               _DEQ_DEFINE(type, q_##name, name)
#define GET(type, name)                 _GET_DEFINE(type, bf_##name, name)
#define ALLOC(type, name)               _ALLOC_DEFINE(type, q_##name, name)

#define FREE(var)                       _FREE(q_##var, var)

// Init function stuff
#define _SET_GLOBAL(type, var, add, of)     var.SetGlobalBuffer((__gm__ type *)add + (of))
#define SET_GLOBAL(type, add, offset)       _SET_GLOBAL(type, gm_##add, add, offset)
#define __INIT_QUEUE(p, tp, qn, bn, l)      p InitBuffer(qn, (bn), ((l) * sizeof(tp)))
#define __INIT_QUEUE_NOPTR(tp, qn, bn, l)   GetTPipePtr()->InitBuffer(qn, (bn), ((l) * sizeof(tp)))
#define INIT_QUEUE_1(type, qname, len)      __INIT_QUEUE_NOPTR(type, qname, 1, len)
#define INIT_QUEUE_2(type, qname, len)      __INIT_QUEUE_NOPTR(type, qname, 2, len)
#define INIT_QUEUE_3(type, qname, len)      __INIT_QUEUE_NOPTR(type, qname, 3, len)
#define INIT_QUEUE_N(type, q, bn, len)      __INIT_QUEUE_NOPTR(type, q, (bn), (len))
#define __INIT_BUFF(p, tp, bn, l)           p InitBuffer(bn, ((l) * sizeof(tp)))
#define __INIT_BUFF_NOPTR(tp, bn, l)        GetTPipePtr()->InitBuffer(bn, ((l) * sizeof(tp)))
#define INIT_BUFF_N(type, bfname, len)      __INIT_BUFF_NOPTR(type, bfname, len)

// bitwise hack
#define TREAT_AS(type)                  template ReinterpretCast<type>()
#define CAST(type, name)                reinterpret_cast<type>(name)

// tiling struct stuff
#define TI                              this->ti
#define TI_COPY_VAR(var_name)           .var_name = tiling_data.var_name
#define _TI_COPY_ARR(ti, arr_name, it)  ti.arr_name[it] = tiling_data.arr_name[it]
#define TI_COPY_ARR_I(arr_name)         _TI_COPY_ARR(ti, arr_name, i)
#define _COPY_ARR(ti, arr_name, it)     arr_name[it] = ti.arr_name[it]
#define COPY_ARR_I(arr_name)            _COPY_ARR(ti, arr_name, i)
#define DEF_VAR_FROM_TI(var_name)       auto var_name = TI.var_name
#define DEF_CVAR_FROM_TI(var_name)      const auto var_name = TI.var_name

// algorithms
#define CEIL(x, align_num)              (((x) + (align_num) - 1) / (align_num) * (align_num))
#define FLOOR(x, align_num)             ((x) / (align_num) * (align_num))

// Alias
using std::is_same_v;
using namespace AscendC;
