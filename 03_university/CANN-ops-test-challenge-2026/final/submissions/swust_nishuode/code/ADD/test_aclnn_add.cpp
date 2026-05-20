#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"

#define CHECK(ret) do { if(ret != ACL_SUCCESS) { printf("Error\n"); return -1; } } while(0)

int64_t GetSize(const std::vector<int64_t>& s) {
    int64_t r = 1;
    for(auto d : s) r *= d;
    return r;
}

int CreateTensor(const std::vector<float>& data, const std::vector<int64_t>& shape,
                 void** dev, aclDataType dt, aclTensor** t) {
    auto size = GetSize(shape) * sizeof(float);
    CHECK(aclrtMalloc(dev, size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy(*dev, size, data.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
    std::vector<int64_t> strides(shape.size(), 1);
    for(int i = shape.size() - 2; i >= 0; i--) strides[i] = shape[i+1] * strides[i+1];
    *t = aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0,
                         ACL_FORMAT_ND, shape.data(), shape.size(), *dev);
    return 0;
}

int main() {
    printf("=== Add算子测试 ===\n");
    
    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));
    aclrtStream stream;
    CHECK(aclrtCreateStream(&stream));
    
    int total = 0, passed = 0;
    
    // 测试1: 基础2x2
    printf("\nTest1: Basic 2x2\n");
    total++;
    {
        std::vector<int64_t> shape = {2,2};
        std::vector<float> x1 = {1,2,3,4};
        std::vector<float> x2 = {5,6,7,8};
        std::vector<float> exp = {6,8,10,12};
        void *d1=0,*d2=0,*dout=0,*ws=0;
        aclTensor *t1=0,*t2=0,*tout=0;
        CreateTensor(x1,shape,&d1,ACL_FLOAT,&t1);
        CreateTensor(x2,shape,&d2,ACL_FLOAT,&t2);
        std::vector<float> out(4,0);
        CreateTensor(out,shape,&dout,ACL_FLOAT,&tout);
        float av=1.0f;
        aclScalar* alpha=aclCreateScalar(&av,ACL_FLOAT);
        uint64_t wsSize=0;
        aclOpExecutor* exec=0;
        aclnnAddGetWorkspaceSize(t1,t2,alpha,tout,&wsSize,&exec);
        if(wsSize>0) aclrtMalloc(&ws,wsSize,ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnAdd(ws,wsSize,exec,stream);
        aclrtSynchronizeStream(stream);
        aclrtMemcpy(out.data(),4*sizeof(float),dout,4*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
        bool ok=true;
        for(int i=0;i<4;i++) if(std::abs(out[i]-exp[i])>1e-5) ok=false;
        if(ok) passed++;
        printf("  %s\n",ok?"PASS":"FAIL");
        aclDestroyScalar(alpha);
        aclrtFree(d1);aclrtFree(d2);aclrtFree(dout);if(ws)aclrtFree(ws);
        aclDestroyTensor(t1);aclDestroyTensor(t2);aclDestroyTensor(tout);
    }
    
    // 测试2: Alpha参数
    printf("\nTest2: Alpha=2.5\n");
    total++;
    {
        std::vector<int64_t> shape={2,3};
        std::vector<float> x1={1,2,3,4,5,6};
        std::vector<float> x2={1,1,1,2,2,2};
        std::vector<float> exp={3.5,4.5,5.5,9,10,11};
        void *d1=0,*d2=0,*dout=0,*ws=0;
        aclTensor *t1=0,*t2=0,*tout=0;
        CreateTensor(x1,shape,&d1,ACL_FLOAT,&t1);
        CreateTensor(x2,shape,&d2,ACL_FLOAT,&t2);
        std::vector<float> out(6,0);
        CreateTensor(out,shape,&dout,ACL_FLOAT,&tout);
        float av=2.5f;
        aclScalar* alpha=aclCreateScalar(&av,ACL_FLOAT);
        uint64_t wsSize=0;
        aclOpExecutor* exec=0;
        aclnnAddGetWorkspaceSize(t1,t2,alpha,tout,&wsSize,&exec);
        if(wsSize>0) aclrtMalloc(&ws,wsSize,ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnAdd(ws,wsSize,exec,stream);
        aclrtSynchronizeStream(stream);
        aclrtMemcpy(out.data(),6*sizeof(float),dout,6*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
        bool ok=true;
        for(int i=0;i<6;i++) if(std::abs(out[i]-exp[i])>1e-5) ok=false;
        if(ok) passed++;
        printf("  %s\n",ok?"PASS":"FAIL");
        aclDestroyScalar(alpha);
        aclrtFree(d1);aclrtFree(d2);aclrtFree(dout);if(ws)aclrtFree(ws);
        aclDestroyTensor(t1);aclDestroyTensor(t2);aclDestroyTensor(tout);
    }
    
    // 测试3: 广播
    printf("\nTest3: Broadcasting\n");
    total++;
    {
        std::vector<int64_t> shape1={2,3},shape2={1};
        std::vector<float> x1={1,2,3,4,5,6};
        std::vector<float> x2={10};
        std::vector<float> exp={11,12,13,14,15,16};
        void *d1=0,*d2=0,*dout=0,*ws=0;
        aclTensor *t1=0,*t2=0,*tout=0;
        CreateTensor(x1,shape1,&d1,ACL_FLOAT,&t1);
        CreateTensor(x2,shape2,&d2,ACL_FLOAT,&t2);
        std::vector<float> out(6,0);
        CreateTensor(out,shape1,&dout,ACL_FLOAT,&tout);
        float av=1.0f;
        aclScalar* alpha=aclCreateScalar(&av,ACL_FLOAT);
        uint64_t wsSize=0;
        aclOpExecutor* exec=0;
        aclnnAddGetWorkspaceSize(t1,t2,alpha,tout,&wsSize,&exec);
        if(wsSize>0) aclrtMalloc(&ws,wsSize,ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnAdd(ws,wsSize,exec,stream);
        aclrtSynchronizeStream(stream);
        aclrtMemcpy(out.data(),6*sizeof(float),dout,6*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
        bool ok=true;
        for(int i=0;i<6;i++) if(std::abs(out[i]-exp[i])>1e-5) ok=false;
        if(ok) passed++;
        printf("  %s\n",ok?"PASS":"FAIL");
        aclDestroyScalar(alpha);
        aclrtFree(d1);aclrtFree(d2);aclrtFree(dout);if(ws)aclrtFree(ws);
        aclDestroyTensor(t1);aclDestroyTensor(t2);aclDestroyTensor(tout);
    }
    
    // 测试4: 边界值
    printf("\nTest4: Boundary (zeros & negatives)\n");
    total++;
    {
        std::vector<int64_t> shape={2,4};
        std::vector<float> x1={0,-1,-2,-3,0,1,2,3};
        std::vector<float> x2={0,1,2,3,0,-1,-2,-3};
        std::vector<float> exp={0,0,0,0,0,0,0,0};
        void *d1=0,*d2=0,*dout=0,*ws=0;
        aclTensor *t1=0,*t2=0,*tout=0;
        CreateTensor(x1,shape,&d1,ACL_FLOAT,&t1);
        CreateTensor(x2,shape,&d2,ACL_FLOAT,&t2);
        std::vector<float> out(8,0);
        CreateTensor(out,shape,&dout,ACL_FLOAT,&tout);
        float av=1.0f;
        aclScalar* alpha=aclCreateScalar(&av,ACL_FLOAT);
        uint64_t wsSize=0;
        aclOpExecutor* exec=0;
        aclnnAddGetWorkspaceSize(t1,t2,alpha,tout,&wsSize,&exec);
        if(wsSize>0) aclrtMalloc(&ws,wsSize,ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnAdd(ws,wsSize,exec,stream);
        aclrtSynchronizeStream(stream);
        aclrtMemcpy(out.data(),8*sizeof(float),dout,8*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
        bool ok=true;
        for(int i=0;i<8;i++) if(std::abs(out[i]-exp[i])>1e-5) ok=false;
        if(ok) passed++;
        printf("  %s\n",ok?"PASS":"FAIL");
        aclDestroyScalar(alpha);
        aclrtFree(d1);aclrtFree(d2);aclrtFree(dout);if(ws)aclrtFree(ws);
        aclDestroyTensor(t1);aclDestroyTensor(t2);aclDestroyTensor(tout);
    }
    
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    
    printf("\n=== Results: %d/%d passed ===\n", passed, total);
    return (passed==total)?0:1;
}
