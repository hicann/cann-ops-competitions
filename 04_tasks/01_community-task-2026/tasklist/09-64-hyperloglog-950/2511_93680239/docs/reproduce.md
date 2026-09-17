# 复现入口与执行边界

本页提供950PR独立复现步骤。已执行的自测范围与命令/退出码入口见[证据索引](evidence-index.md)；以下可移植命令用于复现，不代表新增实测结果。

已执行：候选bbd2d2abd2248a7c的1280原功能和一次72行原性能；提交9bee28c的全新16目标构建、补充测试、多TU、示例及显式内存审计。最终038596d的140项编译输入与该构建一致，文档更新没有被写成新一次真机运行。

环境：Linux Ascend950PR_9579、CANN9.1.0-beta.3、Bisheng dav-3510、C++17、CMake、g++、Python3、Git；Catch2 v3.5.4依赖包已附。容器实测预算32CPU/32GiB，功能8进程；性能单独串行，运行时设备不应有其他测试任务。其他CANN/硬件未验证。

## 1. 校验材料并从干净目录构建

先获得完整`hyperloglog-local-acceptance-20260917`包（尚未线上发布，不能从本文推导下载URL）。以下在该包的Linux目录启动；输出写到其下新建的`review-run`，不覆盖封存证据。CANN激活路径按实际安装选择，示例为已验证版本的常用安装位置。

```bash
set -euo pipefail
python3 verify_package.py
package_dir="$PWD"
mkdir review-run
cd review-run
run_dir="$PWD"
git -c core.autocrlf=false clone --no-checkout "$package_dir/source/hyperloglog.bundle" source
git -C source config core.autocrlf false
git -C source checkout --detach 038596d2fbe8e4f7a65b88f1f682faa08b476e05
cd source
test -z "$(git status --porcelain)"
mkdir -p 3rdparty/Catch2
tar -xzf "$package_dir/dependencies/Catch2-v3.5.4.tar.gz" -C 3rdparty/Catch2 --strip-components=1
source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh
export HLL_BUILD_JOBS=4
mkdir "$run_dir/evidence"
python3 - <<'PY'
import sys
sys.path.insert(0, 'scripts')
from run_hyperloglog_validation import implementation_hash, verify_target
assert implementation_hash() == '5107ffc5f5cda78361bc4c201d1077a486236f8a9f0cbdf86d8b55682831b05a'
assert len(verify_target()) == 13
print('IMPLEMENTATION_AND_ORIGINAL_TESTS_MATCH')
PY
# 保留命令输出与退出码；失败立即停止。
set +e
bash scripts/build_hyperloglog.sh > "$run_dir/evidence/build.log" 2>&1
build_status=$?
set -e
printf '%s\n' "$build_status" > "$run_dir/evidence/build-exit.txt"
test "$build_status" -eq 0
```

构建脚本生成`build/hyperloglog/hyperloglog-build-manifest.json`，记录源码、编译参数、二进制哈希。该脚本还会编译性能二进制，但不执行性能测试。不要复用已有build目录来宣称干净构建。

## 2. 补充运行与完整原始功能

仍在上述source目录和同一shell中运行：

```bash
set +e
python3 scripts/run_hyperloglog_supplemental.py --output "$run_dir/evidence" > "$run_dir/evidence/supplemental.log" 2>&1
supp_status=$?
set -e
printf '%s\n' "$supp_status" > "$run_dir/evidence/supplemental-exit.txt"
test "$supp_status" -eq 0

set +e
python3 scripts/run_hyperloglog_validation.py --phase functional \
  --output "$run_dir/evidence/functional" --jobs 8 \
  --functional-timeout 10800 --require-build-manifest \
  > "$run_dir/evidence/functional-driver.log" 2>&1
functional_status=$?
set -e
printf '%s\n' "$functional_status" > "$run_dir/evidence/functional-exit.txt"
test "$functional_status" -eq 0
test -z "$(git status --porcelain)"
git rev-parse HEAD > "$run_dir/evidence/source-commit.txt"
cp build/hyperloglog/hyperloglog-build-manifest.json "$run_dir/evidence/"
```

supplemental runner包含7组合、multi-TU、完整API示例和memory audit。原始功能runner按dtype、SECTION分28片，保留全部GENERATE，预计六操作计数38/24/180/726/192/120。检查新建时间戳目录的report.json、原始XML与parts.jsonl；数量不足或失败不能算1280通过。异常退出时保留driver日志和退出码，不删失败记录。

## 3. 性能复现

现成72行和原始日志已封存，直接按[性能附录](appendix-performance.md)审计即可。以下提供独立复现的完整入口。确认上一节所有进程退出，选择本次生成的唯一功能报告，不引用旧候选报告。

```bash
functional_report=$(find "$run_dir/evidence/functional" -mindepth 2 -maxdepth 2 -name report.json -type f)
test "$(printf '%s\n' "$functional_report" | wc -l)" -eq 1
test -f "$functional_report"
set +e
python3 scripts/run_hyperloglog_validation.py --phase performance \
  --output "$run_dir/evidence/performance" \
  --functional-report "$functional_report" --require-build-manifest \
  > "$run_dir/evidence/performance-driver.log" 2>&1
performance_status=$?
set -e
printf '%s\n' "$performance_status" > "$run_dir/evidence/performance-exit.txt"
test "$performance_status" -eq 0
```

runner强制性能串行并逐行比较baseline/0.4，不能以平均数或新旧结果拼接替代。新独立结果必须保留其环境、日期和版本，不覆盖已有唯一成功性能记录；耗时字段仍是同步API Host计时。

## 4. 已执行命令的原件

E3归档内`run_hll_final_functional.sh`、`run_hll_final_clean.sh`保存当时完整驱动命令，driver.log及job-exit.txt保存输出与退出状态。E2内性能report.json保存各操作实际argv、退出码及log SHA256。上面的可移植命令是为评审准备的复现步骤，不冒充当时原样执行的记录。
