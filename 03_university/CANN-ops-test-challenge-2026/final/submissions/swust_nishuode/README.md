## 1. 团队信息

- 团队名称：你说的
- 所属单位：西南科技大学
- 团队成员：
  - 何嘉俊，测试代码编写与调试
  - 韦汉鑫，测试用例设计与优化
  - 黄世凯，精度分析与报告编写
- 联系人：何嘉俊
- 联系邮箱：2657916031@qq.com

## 2. 环境要求

- CANN 版本：9.0.0
- 操作系统：Ubuntu 20.04 x86_64
- 编译器：g++ 9.4.0
- 其他依赖：
  - CMake 3.18.0
  - Python 3.8
  - Git 2.25.1

如使用大赛提供的 Docker 镜像（yeren666/cann-ops-test:v1.0），请在此处注明并标注镜像版本。

## 3. 文件说明

- ```
  code/
  ```

  ：测试代码源文件，按算子分子目录组织

  - `code/Add/`：Add 算子测试代码
  - `code/CumSum/`：Cumsum算子测试代码
  
- ```
  report/
  ```

  测试报告

  - `report/Add.md`：Add算子详细测试报告
  - `report/CumSum.md`：CumSum算子测试报告

## 4. 编译与运行

以下是编译和运行测试代码的简要说明：

### 4.1 编译与运行步骤

1. 进入对应算子目录：

   ```
   cd code/Add
   ```

2. 创建并进入 `build` 目录：

   ```
   mkdir build && cd build
   ```

3. 运行 `cmake` 和 `make` 命令进行编译：

   ```
   cmake ..
   make
   ```

4. 编译完成后，运行测试：

   ```
   ./test_aclnn_add
   ```

### 4.2 其他命令

若想使用 Docker 镜像，确保 Docker 环境已经安装，并执行以下命令：

1. 拉取 Docker 镜像（如果尚未拉取）：

   ```
   docker pull yeren666/cann-ops-test:v1.0
   ```

2. 启动 Docker 容器并进入工作目录：

   ```
   docker run -it --rm yeren666/cann-ops-test:v1.0
   cd /workspace
   ```

3. 按照上述编译和运行步骤执行测试。