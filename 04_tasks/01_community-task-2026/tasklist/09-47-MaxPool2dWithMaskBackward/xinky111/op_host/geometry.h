#pragma once
#include "../op_kernel/pool_tiling.h"
#include <climits>

inline bool PoolPair(const int64_t* values, int64_t count, int64_t& a, int64_t& b, bool empty=false) {
    if (count<0 || count>2 || (!empty && count==0) || (count && !values)) return false;
    if (!count) return true;
    a=values[0];b=values[count-1];
    return a>=0 && b>=0 && a<=INT32_MAX && b<=INT32_MAX;
}
inline int64_t PoolDim(int64_t input,int64_t k,int64_t s,int64_t p,bool ceil) {
    const int64_t numerator=input+2*p-k+(ceil?s-1:0);
    int64_t out=numerator>=0?numerator/s+1:-((-numerator+s-1)/s)+1;
    if (ceil && (out-1)*s>=input+p) --out;
    return out;
}
inline bool MakePoolPlan(const int64_t* xs,const int64_t* gs,const int64_t* ms,
                         const int64_t* k,int64_t nk,const int64_t* s,int64_t ns,
                         const int64_t* p,int64_t np,const int64_t* d,int64_t nd,
                         bool ceil,PoolTiling& t) {
    if (!xs || !gs || !ms) return false;
    t={};
    const int64_t n=xs[0],c=xs[1];t.height=xs[2];t.width=xs[3];
    if (n<=0 || c<=0 || n>INT64_MAX/c || t.height<=0 || t.width<=0 || t.height>INT32_MAX/t.width) return false;
    t.planes=n*c;
    if (t.planes>(INT64_MAX-8192)/(t.height*t.width)) return false;
    if (!PoolPair(k,nk,t.kernelH,t.kernelW) || t.kernelH==0 || t.kernelW==0) return false;
    t.strideH=t.kernelH;t.strideW=t.kernelW;
    int64_t dh=1,dw=1;
    if (!PoolPair(s,ns,t.strideH,t.strideW,true) || !PoolPair(p,np,t.padH,t.padW) || !PoolPair(d,nd,dh,dw) ||
        dh!=1 || dw!=1 || t.strideH==0 || t.strideW==0 || t.padH>t.kernelH || t.padW>t.kernelW) return false;
    t.outHeight=PoolDim(t.height,t.kernelH,t.strideH,t.padH,ceil);
    t.outWidth=PoolDim(t.width,t.kernelW,t.strideW,t.padW,ceil);
    if (t.outHeight<=0 || t.outWidth<=0 || t.outHeight>INT64_MAX/t.outWidth) return false;
    const int64_t hw=t.outHeight*t.outWidth;
    if (hw>(INT64_MAX-64)/4 || t.planes>INT64_MAX/(hw*4)) return false;
    const int64_t maskWidth=((hw+15)/16+1)*32;
    if (t.kernelH>INT64_MAX/t.kernelW || t.kernelH*t.kernelW>INT64_MAX/maskWidth) return false;
    if (t.kernelH*t.kernelW*maskWidth<4*hw) return false;
    return gs[0]==n && gs[1]==c && gs[2]==t.outHeight && gs[3]==t.outWidth &&
           ms[0]==n && ms[1]==c && ms[2]==t.kernelH*t.kernelW && ms[3]==maskWidth;
}
