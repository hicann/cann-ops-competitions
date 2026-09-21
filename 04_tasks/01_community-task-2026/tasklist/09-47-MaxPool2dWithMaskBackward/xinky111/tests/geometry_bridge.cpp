#include "../op_host/geometry.h"
extern "C" int test_plan(const int64_t* x,const int64_t* g,const int64_t* m,const int64_t* k,int64_t nk,
                         const int64_t* s,int64_t ns,const int64_t* p,int64_t np,const int64_t* d,int64_t nd,
                         bool ceil,PoolTiling* out) {
    return MakePoolPlan(x,g,m,k,nk,s,ns,p,np,d,nd,ceil,*out)?0:1;
}
