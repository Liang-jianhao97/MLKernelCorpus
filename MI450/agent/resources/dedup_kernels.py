#!/usr/bin/env python3
"""Deduplicate LLVM IR kernel files with identical content.

Usage:
    python3 dedup_kernels.py <kernels_dir> [--dry-run]

Walks all subdirectories of <kernels_dir>, hashes every .llir file, and for
each group of identical files keeps one canonical copy and removes the rest.
Updates each subdirectory's mapping.json so removed entries get
status="deduplicated" with a pointer to the kept file.

Canonical selection priority (highest first):
    best > OAI > gluon > aiter > triton

Within the same directory, the file whose mapping entry appears first is kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

DIR_PRIORITY = {"best": 0, "OAI": 1, "gluon": 2, "aiter": 3, "triton": 4}


def hash_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_canonical(files: list[str]) -> str:
    """Pick the canonical file from a group of identical files."""
    def sort_key(filepath):
        subdir = filepath.split(os.sep)[0]
        priority = DIR_PRIORITY.get(subdir, 99)
        is_helper = "_helper" in os.path.basename(filepath)
        return (priority, is_helper, filepath)
    return min(files, key=sort_key)


def load_mapping(subdir_path: str) -> dict | None:
    mapping_path = os.path.join(subdir_path, "mapping.json")
    if not os.path.exists(mapping_path):
        return None
    with open(mapping_path) as f:
        return json.load(f)


def save_mapping(subdir_path: str, mapping: dict):
    mapping_path = os.path.join(subdir_path, "mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
        f.write("\n")


def build_file_to_kernel_index(mapping: dict) -> dict:
    """Map filename -> (kernel_entry, field) for primary and helper files."""
    index = {}
    for entry in mapping.get("kernels", []):
        if entry.get("filename"):
            index[entry["filename"]] = (entry, "filename")
        for helper in entry.get("additional_files", []):
            index[helper] = (entry, "additional_files")
    return index


def run_dedup(kernels_dir: str, dry_run: bool = False) -> dict:
    os.chdir(kernels_dir)

    subdirs = sorted(
        d for d in os.listdir(".")
        if os.path.isdir(d) and not d.startswith(".")
    )

    hash_to_files: dict[str, list[str]] = defaultdict(list)
    for subdir in subdirs:
        for fname in sorted(os.listdir(subdir)):
            if fname.endswith(".llir"):
                rel = os.path.join(subdir, fname)
                h = hash_file(rel)
                hash_to_files[h].append(rel)

    total_files = sum(len(v) for v in hash_to_files.values())
    dup_groups = {h: fs for h, fs in hash_to_files.items() if len(fs) > 1}
    redundant = sum(len(fs) - 1 for fs in dup_groups.values())

    print(f"Total .llir files:      {total_files}")
    print(f"Unique IR hashes:       {len(hash_to_files)}")
    print(f"Duplicate groups:       {len(dup_groups)}")
    print(f"Redundant files:        {redundant}")
    print()

    mappings: dict[str, dict] = {}
    for subdir in subdirs:
        m = load_mapping(subdir)
        if m:
            mappings[subdir] = m

    removed_files = []
    kept_files = []

    for h, files in sorted(dup_groups.items(), key=lambda x: -len(x[1])):
        canonical = pick_canonical(files)
        kept_files.append(canonical)
        to_remove = [f for f in files if f != canonical]

        canonical_subdir = canonical.split(os.sep)[0]
        canonical_fname = os.path.basename(canonical)
        canonical_ref = canonical

        for filepath in to_remove:
            subdir = filepath.split(os.sep)[0]
            fname = os.path.basename(filepath)

            if subdir in mappings:
                mapping = mappings[subdir]
                for entry in mapping.get("kernels", []):
                    if entry.get("filename") == fname:
                        if not dry_run:
                            entry["filename"] = None
                            entry["status"] = "deduplicated"
                            entry["deduplicated_to"] = canonical_ref
                        break
                    if fname in entry.get("additional_files", []):
                        if not dry_run:
                            entry["additional_files"].remove(fname)
                            entry.setdefault("deduplicated_helpers", []).append({
                                "original": fname,
                                "deduplicated_to": canonical_ref,
                            })
                        break

            if not dry_run:
                os.remove(filepath)
            removed_files.append((filepath, canonical_ref))

    if not dry_run:
        for subdir, mapping in mappings.items():
            kernel_list = mapping.get("kernels", [])
            deduped = sum(
                1 for k in kernel_list if k.get("status") == "deduplicated"
            )
            success = sum(
                1 for k in kernel_list
                if k.get("status") in ("success", "launch_failed_ir_captured")
                and k.get("filename") is not None
            )
            failed = sum(
                1 for k in kernel_list
                if k.get("status") in ("compilation_failed", "import_error")
            )
            mapping["summary"] = {
                "total": len(kernel_list),
                "unique_ir": success,
                "deduplicated": deduped,
                "failed": failed,
            }
            save_mapping(subdir, mapping)

    report = {
        "total_before": total_files,
        "total_after": total_files - len(removed_files),
        "removed": len(removed_files),
        "duplicate_groups": len(dup_groups),
        "unique_hashes": len(hash_to_files),
        "removals": [
            {"removed": r, "canonical": c} for r, c in removed_files
        ],
    }

    if dry_run:
        print("[DRY RUN] Would remove:")
    else:
        print("Removed:")
    for filepath, canonical in removed_files:
        print(f"  {filepath}  ->  {canonical}")

    print()
    print(f"Files before:  {total_files}")
    print(f"Files after:   {total_files - len(removed_files)}")
    print(f"Removed:       {len(removed_files)}")

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernels_dir", help="Path to MI450/kernels/")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be removed without deleting anything"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.kernels_dir):
        print(f"Error: {args.kernels_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    report = run_dedup(args.kernels_dir, dry_run=args.dry_run)

    report_path = os.path.join(os.path.abspath(args.kernels_dir), "dedup_report.json")
    if not args.dry_run:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
