#include <iostream>
#include "acl/acl.h"

// 演示 ACLNN 调用流程伪代码及接口骨架
int main() {
    std::cout << "[Example] Starting aclnnMaxPool2dWithMaskBackward call..." << std::endl;
    // 1. aclInit(nullptr);
    // 2. aclrtSetDevice(0);
    // 3. aclrtCreateStream(&stream);
    // 4. aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize(...)
    // 5. aclrtMalloc(&workspace, workspaceSize, ...)
    // 6. aclnnMaxPool2dWithMaskBackward(workspace, workspaceSize, executor, stream)
    // 7. aclrtSynchronizeStream(stream);
    std::cout << "[Example] aclnnMaxPool2dWithMaskBackward completed successfully!" << std::endl;
    return 0;
}
