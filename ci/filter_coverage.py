#!/usr/bin/env python3

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
使用 diff-cover 的解析结果来过滤 coverage.xml
只保留 diff.txt 中提到的变更文件的覆盖率数据
"""

import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def parse_diff_file(diff_file):
    """
    解析 diff.txt 文件，获取变更的文件和行号

    格式示例：
    --- a/src/file1.py
    +++ b/src/file1.py
    @@ -10,5 +10,7 @@

    返回: dict {文件名: set(变更行号)}
    """
    changed_files = defaultdict(set)
    current_file = None

    if not os.path.exists(diff_file):
        print(f"Error: Diff file not found: {diff_file}")
        return changed_files

    print(f"Parsing diff file: {diff_file}")

    with open(diff_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # 识别目标文件 (+++ b/path/to/file)
        if line.startswith("+++ b/"):
            current_file = line[6:]  # 去掉 '+++ b/'
            print(f"  Found changed file: {current_file}")
            if "src/paddlefleet" not in current_file:
                current_file = None  # 只关注 paddlefleet 相关文件
                print("    Skipping non-paddlefleet file")
                continue
            elif line.endswith(
                (
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                    ".bmp",
                    ".pdf",
                    ".zip",
                    ".tar",
                    ".gz",
                    ".so",
                    ".dll",
                    ".exe",
                )
            ):
                current_file = None  # 过滤二进制文件
                print("    Skipping binary file")
                continue

        # 解析行号范围 (@@ -old_start,old_length +new_start,new_length @@)
        elif line.startswith("@@") and current_file:
            # 示例: @@ -10,5 +10,7 @@
            # 我们关心 + 后面的新行号
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if match:
                start_line = int(match.group(1))
                line_count = int(match.group(2) or 1)

                # 添加变更的行号范围
                for line_num in range(start_line, start_line + line_count):
                    changed_files[current_file].add(line_num)

    print("\nDiff analysis:")
    total_changed_lines = sum(len(lines) for lines in changed_files.values())
    print(f"  Changed files: {len(changed_files)}")
    print(f"  Total changed lines: {total_changed_lines}")

    for filename, lines in changed_files.items():
        if len(lines) <= 10:
            print(
                f"    {filename}: lines {sorted(lines)[:10]}{'...' if len(lines) > 10 else ''}"
                )
        else:
            print(f"    {filename}: {len(lines)} changed lines")
    return changed_files


def filter_coverage_by_diff(coverage_file, diff_data):
    """
    基于 diff 数据过滤 coverage.xml - 修复版
    """
    if not os.path.exists(coverage_file):
        print(f"Error: Coverage file not found: {coverage_file}")
        return False

    print(f"\nFiltering coverage file: {coverage_file}")

    try:
        # 解析覆盖率文件
        tree = ET.parse(coverage_file)
        root = tree.getroot()

        # 备份
        import shutil

        backup_file = coverage_file + ".backup"
        shutil.copy2(coverage_file, backup_file)

        # 统计信息
        stats = {
            "files_processed": 0,
            "files_kept": 0,
            "lines_kept": 0,
            "lines_removed": 0,
            "lines_covered": 0,
        }

        # 构建文件名到 class 元素的映射
        filename_to_classes = defaultdict(list)

        # 第一遍：收集所有 class 元素
        for class_elem in root.findall(".//class"):
            filename = class_elem.get("filename", "")
            if filename:
                filename_to_classes[filename].append(class_elem)
                stats["files_processed"] += 1

        print(f"Found {stats['files_processed']} files in coverage report")

        # 第二遍：找出哪些文件需要处理
        files_to_keep = set()
        for coverage_filename in filename_to_classes.keys():
            for diff_filename in diff_data.keys():
                if filename_match(coverage_filename, diff_filename):
                    files_to_keep.add(coverage_filename)
                    break

        print(f"Files to keep after diff matching: {len(files_to_keep)}")

        # 如果没有匹配的文件，创建最小覆盖率
        if not files_to_keep:
            print("No matching files found, creating minimal coverage")
            create_minimal_coverage(coverage_file)
            return True

        # 创建新的 XML 结构
        new_root = ET.Element("coverage")
        new_root.set("version", "1.0")
        new_root.set("timestamp", root.get("timestamp", "0"))

        # 添加基本元素
        sources = ET.SubElement(new_root, "sources")
        ET.SubElement(sources, "source").text = "paddlefleet"

        packages = ET.SubElement(new_root, "packages")

        # 创建一个 package 包含所有变更文件
        package_name = "changed_files"
        package_elem = ET.SubElement(packages, "package")
        package_elem.set("name", package_name)
        package_elem.set("line-rate", "0")
        package_elem.set("branch-rate", "0")
        package_elem.set("complexity", "0")

        # 处理每个需要保留的文件
        for filename in files_to_keep:
            # 找到对应的 diff 文件（可能有多个匹配）
            matched_diff_file = None
            changed_lines = set()

            for diff_file, lines in diff_data.items():
                if filename_match(filename, diff_file):
                    matched_diff_file = diff_file
                    changed_lines = lines
                    break

            if not matched_diff_file:
                continue

            # 处理这个文件的所有 class 元素
            for class_elem in filename_to_classes[filename]:
                stats["files_kept"] += 1

                # 创建新的 class 元素
                new_class = ET.Element("class")

                # 复制所有属性
                for attr_name, attr_value in class_elem.items():
                    new_class.set(attr_name, attr_value)

                # 处理 lines 元素
                lines_elem = class_elem.find("lines")
                if lines_elem is not None:
                    new_lines_elem = ET.SubElement(new_class, "lines")

                    # 过滤行数据
                    for line_elem in lines_elem.findall("line"):
                        line_num = int(line_elem.get("number", 0))

                        if line_num in changed_lines:
                            # 复制这一行
                            new_line_elem = ET.SubElement(
                                new_lines_elem, "line"
                            )
                            for attr_name, attr_value in line_elem.items():
                                new_line_elem.set(attr_name, attr_value)

                            stats["lines_kept"] += 1

                            hits = int(line_elem.get("hits", 0))
                            if hits > 0:
                                stats["lines_covered"] += 1
                        else:
                            stats["lines_removed"] += 1

                # 添加到 package
                package_elem.append(new_class)

        # 计算覆盖率
        total_lines = stats["lines_kept"]
        covered_lines = stats["lines_covered"]

        if total_lines > 0:
            coverage_rate = covered_lines / total_lines
        else:
            coverage_rate = 0

        # 更新 package 和根元素的覆盖率
        package_elem.set("line-rate", str(coverage_rate))
        package_elem.set("branch-rate", str(coverage_rate))

        new_root.set("line-rate", str(coverage_rate))
        new_root.set("branch-rate", str(coverage_rate))
        new_root.set("lines-covered", str(covered_lines))
        new_root.set("lines-valid", str(total_lines))

        # 保存文件
        tree = ET.ElementTree(new_root)
        tree.write(coverage_file, encoding="utf-8", xml_declaration=True)

        # 打印统计信息
        print("\n📊 Filtering Statistics:")
        print(f"  Files processed: {stats['files_processed']}")
        print(f"  Files kept: {stats['files_kept']}")
        print(f"  Lines kept: {stats['lines_kept']}")
        print(f"  Lines removed: {stats['lines_removed']}")
        print(
            f"  Coverage: {covered_lines}/{total_lines} lines ({coverage_rate * 100:.1f}%)"
        )

        # 验证文件
        try:
            ET.parse(coverage_file)
            print("\n✅ Valid XML generated")
            return True
        except ET.ParseError as e:
            print(f"❌ Generated invalid XML: {e}")
            # 恢复备份
            shutil.copy2(backup_file, coverage_file)
            return False

    except Exception as e:
        print(f"❌ Error filtering coverage: {e}")
        import traceback

        traceback.print_exc()
        return False


def filename_match(coverage_filename, diff_filename):
    """
    智能文件名匹配

    处理情况：
    1. coverage: "module/file.py", diff: "src/module/file.py"
    2. coverage: "./module/file.py", diff: "module/file.py"
    3. coverage: "file.py", diff: "src/file.py"
    """
    import os

    # 标准化路径
    def normalize(path):
        # 去除开头的 ./
        path = path.removeprefix("./")
        # 统一分隔符
        path = path.replace("\\", "/")
        # 去除末尾的 /
        path = path.rstrip("/")
        return path.lower()

    cov_norm = normalize(coverage_filename)
    diff_norm = normalize(diff_filename)

    # 1. 完全匹配
    if cov_norm == diff_norm:
        return True

    # 2. coverage 是 diff 的后缀
    if diff_norm.endswith(cov_norm):
        return True

    # 3. diff 是 coverage 的后缀
    if cov_norm.endswith(diff_norm):
        return True

    # 4. 只比较文件名（不包括路径）
    cov_basename = os.path.basename(cov_norm)
    diff_basename = os.path.basename(diff_norm)

    if cov_basename and diff_basename and cov_basename == diff_basename:
        # 进一步检查路径相关性
        cov_dir = os.path.dirname(cov_norm)
        diff_dir = os.path.dirname(diff_norm)

        # 如果有一个是另一个的子目录
        if cov_dir in diff_dir or diff_dir in cov_dir:
            return True

    return False


def create_minimal_coverage(filename):
    """创建最小覆盖率文件"""
    minimal_xml = """<?xml version="1.0"?>
<coverage version="1.0" timestamp="0">
  <sources>
    <source>.</source>
  </sources>
  <packages>
    <package name="changed_files" line-rate="0" branch-rate="0" complexity="0">
      <classes>
        <class name="ChangedCode" filename="changed.py" line-rate="0" branch-rate="0" complexity="0">
          <methods/>
          <lines/>
        </class>
      </classes>
    </package>
  </packages>
</coverage>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(minimal_xml)
    print(f"Created minimal coverage file: {filename}")


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python filter_coverage_fixed.py <coverage.xml> <diff.txt>"
        )
        print("Example: python filter_coverage_fixed.py coverage.xml diff.txt")
        sys.exit(1)

    coverage_file = sys.argv[1]
    diff_file = sys.argv[2]

    # 解析 diff
    diff_data = parse_diff_file(diff_file)

    # 过滤覆盖率
    success = filter_coverage_by_diff(coverage_file, diff_data)

    if success:
        # 显示文件大小
        if os.path.exists(coverage_file):
            size = os.path.getsize(coverage_file)
            print(f"\n✅ Filtered coverage saved: {size:,} bytes")

            if os.path.exists(coverage_file + ".backup"):
                orig_size = os.path.getsize(coverage_file + ".backup")
                reduction = (1 - size / orig_size) * 100
                print(f"   Reduced by: {reduction:.1f}%")
    else:
        print("\n❌ Failed to filter coverage")
        sys.exit(1)


if __name__ == "__main__":
    main()
