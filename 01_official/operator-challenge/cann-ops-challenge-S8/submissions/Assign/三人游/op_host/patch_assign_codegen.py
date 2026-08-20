#!/usr/bin/env python3
"""Patch generated Assign metadata and ACLNN API for the official in-place wrapper."""

import argparse
import re
from pathlib import Path


ATTR_DEFAULTS = {
    "validate_shape": {"defaultValue": "true", "paramType": "optional"},
    "use_locking": {"defaultValue": "false", "paramType": "optional"},
}

HEADER_OLD = re.compile(
    r"aclnnStatus aclnnAssignGetWorkspaceSize\(\s*"
    r"aclTensor \*inputRef,\s*"
    r"const aclTensor \*other,\s*"
    r"bool validateShape,\s*"
    r"bool useLocking,\s*"
    r"uint64_t \*workspaceSize,\s*"
    r"aclOpExecutor \*\*executor\);",
    re.S,
)

HEADER_NEW = """aclnnStatus aclnnAssignGetWorkspaceSize(
    aclTensor *inputRef,
    const aclTensor *other,
    bool useLocking,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);"""

CPP_SIG_OLD = re.compile(
    r"aclnnStatus aclnnAssignGetWorkspaceSize\(\s*"
    r"aclTensor \*inputRef,\s*"
    r"const aclTensor \*other,\s*"
    r"bool validateShape,\s*"
    r"bool useLocking,\s*"
    r"uint64_t \*workspaceSize,\s*"
    r"aclOpExecutor \*\*executor\)\s*\{",
    re.S,
)

CPP_SIG_NEW = """aclnnStatus aclnnAssignGetWorkspaceSize(
    aclTensor *inputRef,
    const aclTensor *other,
    bool useLocking,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    bool validateShape = true;"""

MATCH_ARGS_BLOCK = re.compile(
    r"\s*if \(NnopbaseSetMatchArgsFlag != NULL\) \{\s*"
    r"NnopbaseSetMatchArgsFlag\(\*executor\);\s*\}\s*"
    r"NNOPBASE_ASSERT_OK_RETVAL\(NnopbaseSetRef\(\*executor, 0, 0\)\);\s*"
    r"NNOPBASE_ASSERT_OK_RETVAL\(NnopbaseAddInput\(\*executor, inputRef, 0\)\);\s*"
    r"NNOPBASE_ASSERT_OK_RETVAL\(NnopbaseAddInput\(\*executor, other, 1\)\);\s*"
    r"NNOPBASE_ASSERT_OK_RETVAL\(NnopbaseAddAttrWithDtype\(\*executor, static_cast<void\*>\(&validateShape\), sizeof\(bool\), 0, kNnopbaseBool\)\);\s*"
    r"NNOPBASE_ASSERT_OK_RETVAL\(NnopbaseAddAttrWithDtype\(\*executor, static_cast<void\*>\(&useLocking\), sizeof\(bool\), 1, kNnopbaseBool\)\);\s*"
    r"NNOPBASE_ASSERT_OK_RETVAL\(NnopbaseAddOutput\(\*executor, inputRef, 0\)\);\s*"
    r"if \(NnopbaseMatchArgs != NULL\) \{\s*"
    r"if \(NnopbaseMatchArgs\(\*executor, workspaceSize\)\) \{\s*"
    r"NnopbaseReportApiInfo\(timeStamp, dfxId\);\s*"
    r"return ACLNN_SUCCESS;\s*"
    r"\}\s*\}",
    re.S,
)

MATCH_ARGS_REPLACEMENT = """
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseSetRef(*executor, 0, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, inputRef, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, other, 1));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&validateShape), sizeof(bool), 0, kNnopbaseBool));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&useLocking), sizeof(bool), 1, kNnopbaseBool));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, inputRef, 0));"""


def upsert_key(lines, key, value, insert_at):
    replacement = f"{key}={value}\n"
    for index, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[index] = replacement
            return 0
    lines.insert(insert_at, replacement)
    return 1


def patch_ini(path):
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    section_start = None
    section_end = None
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if line == "[Assign]":
            section_start = index + 1
            continue
        if section_start is not None and line.startswith("[") and line.endswith("]"):
            section_end = index
            break
    if section_start is None:
        raise RuntimeError("Assign section not found in ops-info ini")
    if section_end is None:
        section_end = len(lines)

    inserted = 0
    for attr_name, settings in ATTR_DEFAULTS.items():
        for suffix in ("defaultValue", "paramType"):
            inserted += upsert_key(
                lines,
                f"attr_{attr_name}.{suffix}",
                settings[suffix],
                section_end + inserted,
            )
    path.write_text("".join(lines), encoding="utf-8")


def patch_header(path):
    text = path.read_text(encoding="utf-8")
    patched, count = HEADER_OLD.subn(HEADER_NEW, text, count=1)
    if count != 1:
        raise RuntimeError("failed to patch aclnn_assign.h signature")
    path.write_text(patched, encoding="utf-8")


def patch_source(path):
    text = path.read_text(encoding="utf-8")
    patched, count = CPP_SIG_OLD.subn(CPP_SIG_NEW, text, count=1)
    if count != 1:
        raise RuntimeError("failed to patch aclnn_assign.cpp signature")
    patched, count = MATCH_ARGS_BLOCK.subn(MATCH_ARGS_REPLACEMENT, patched, count=1)
    if count != 1:
        raise RuntimeError("failed to patch aclnn_assign.cpp match-args block")
    path.write_text(patched, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ini", required=True, type=Path)
    parser.add_argument("--header", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()

    patch_ini(args.ini)
    patch_header(args.header)
    patch_source(args.source)


if __name__ == "__main__":
    main()
