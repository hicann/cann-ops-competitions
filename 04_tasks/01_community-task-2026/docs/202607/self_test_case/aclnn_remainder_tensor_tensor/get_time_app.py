import os
import csv
import re
import sys
import argparse
from collections import defaultdict

def find_and_read_csv(base_dir, test_dir):
    test_path = os.path.join(base_dir, test_dir)
    
    # 策略1: 查找 OpBasicInfo.csv (prof_floormod 格式)
    for root, _, files in os.walk(test_path):
        for f in files:
            if f == "OpBasicInfo.csv":
                return os.path.join(root, f)
    
    # 策略2: 查找 op_summary_*.csv (prof_bernoulli 格式)
    for root, _, files in os.walk(test_path):
        for f in files:
            if f.startswith("op_summary_") and f.endswith(".csv"):
                return os.path.join(root, f)
    
    return None

def parse_args():
    parser = argparse.ArgumentParser(
        description="从 prof_* 目录抓取 op_summary 或 OpBasicInfo 中的 TBE 性能数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python get_time_app.py prof_floormod
  python get_time_app.py prof_bernoulli output.csv
  python get_time_app.py prof_floormod ./result/tbe_time.csv
        """
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="prof_floormod",
        help="输入的 prof 目录路径，如 prof_floormod（默认: prof_floormod）"
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        help="输出的 CSV 文件路径，支持相对路径（默认: <input_dir>_tbe_duration.csv）"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 配置路径
    base_dir = os.path.abspath(args.input_dir) if not os.path.isabs(args.input_dir) else args.input_dir
    
    if args.output_file:
        output_csv = os.path.abspath(args.output_file) if not os.path.isabs(args.output_file) else args.output_file
    else:
        output_csv = f"{args.input_dir}_tbe_duration.csv"
    
    all_prof_path = os.path.join("prof_diff", "all_prof.csv")

    # 2. 遍历 Test_xxx 目录，计算每个用例的耗时总和
    tbe_duration = defaultdict(float)
    
    if not os.path.exists(base_dir):
        print(f"错误：目录 {base_dir} 不存在")
        return

    for test_dir in sorted(os.listdir(base_dir)):
        if not test_dir.startswith("Test_"):
            continue
        match = re.match(r"(Test_\d{3})", test_dir)
        if not match:
            continue
        case_name = match.group(1)

        csv_path = find_and_read_csv(base_dir, test_dir)
        if csv_path is None:
            print(f"  警告：{case_name} 未找到 CSV 文件")
            continue

        total = 0.0
        with open(csv_path, newline='', encoding='utf-8') as cf:
            reader = csv.DictReader(cf)
            for row in reader:
                dur_str = row.get("Task Duration(us)", "").strip()
                if dur_str:
                    try:
                        dur = float(dur_str)
                        total += dur
                    except ValueError:
                        continue
        tbe_duration[case_name] = total

    # 3. 读取 all_prof.csv 中的 custom 耗时（可选）
    custom_duration = {}
    has_custom = os.path.exists(all_prof_path)
    if has_custom:
        with open(all_prof_path, newline='', encoding='utf-8') as cf:
            reader = csv.DictReader(cf)
            for row in reader:
                case = row.get("case_name", "").strip()
                if not case:
                    continue
                match = re.match(r"(Test_\d{3})", case)
                if not match:
                    continue
                case_name = match.group(1)
                try:
                    time_use = float(row.get("time_use", 0))
                    custom_duration[case_name] = time_use
                except ValueError:
                    custom_duration[case_name] = 0.0
    else:
        print(f"提示：未找到文件 {all_prof_path}，仅输出 TBE 耗时数据")

    # 4. 确保输出目录存在
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 5. 写入结果到 CSV 文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if has_custom:
            writer.writerow([
                "用例名称",
                "Custom耗时(us)",
                "TBE耗时(us)",
                "差值(us)",
                "差值百分比(%)"
            ])
            for case in sorted(set(custom_duration.keys()) | set(tbe_duration.keys())):
                cust = custom_duration.get(case, 0.0)
                tbe = tbe_duration.get(case, 0.0)
                diff = cust - tbe
                diff_pct = (diff / tbe * 100) if tbe != 0 else 0.0
                writer.writerow([
                    case,
                    round(cust, 2),
                    round(tbe, 2),
                    round(diff, 2),
                    round(diff_pct, 2)
                ])
        else:
            writer.writerow([
                "用例名称",
                "TBE耗时(us)"
            ])
            for case in sorted(tbe_duration.keys()):
                writer.writerow([
                    case,
                    round(tbe_duration[case], 2)
                ])

    print(f"✅ 完成！共处理 {len(tbe_duration)} 个用例，结果已保存到：{output_csv}")

if __name__ == "__main__":
    main()
