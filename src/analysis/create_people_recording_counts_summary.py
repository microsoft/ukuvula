#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Generate a three-column CSV (person_name,recording_count,summary) from the
existing `results/people_recording_counts.csv` (currently holding person_name,summary)
by counting how many media file paths under `data/nmf_recordings` contain each name.

Definition of recording_count here:
  The number of distinct media file paths under data/nmf_recordings whose full relative
  path string (case-insensitive) contains the exact person_name substring.

Notes:
  - This differs from earlier heuristic association counts (which parsed tokens).
  - Substring matching may over-count for short names (e.g., 'Ali' inside 'Album').
    To dampen obvious false positives, we require either word boundary match OR exact substring
    with length >= 5. For multi-word names we attempt whole phrase matching ignoring multiple spaces.
  - Media extensions considered: common audio/video formats.
"""
import csv
import os
import re
import sys
from pathlib import Path

# Ensure src/ is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SUMMARY_SOURCE = Path("results/people_recording_counts.csv")
OUTPUT = Path("results/people_recording_counts_summary.csv")
DATA_ROOT = Path("data/nmf_recordings")
MEDIA_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".mp4", ".mkv", ".mov", ".avi", ".webm"}

def load_names_with_summaries(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        # Expect header like ['person_name','summary']
        if len(header) < 2 or header[0].lower() != 'person_name':
            raise ValueError("Unexpected header in summary source: " + str(header))
        for r in reader:
            if not r:
                continue
            name = r[0].strip()
            summary = r[1].strip() if len(r) > 1 else ""
            rows.append((name, summary))
    return rows

def collect_media_paths(root: Path):
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if Path(f).suffix.lower() in MEDIA_EXTS:
                full = Path(dirpath) / f
                rel = full.relative_to(root)
                paths.append(str(rel))
    return paths

def count_occurrences(name: str, media_paths):
    # Normalize spaces in name for phrase matching
    norm_name = re.sub(r"\s+", " ", name.strip())
    if not norm_name:
        return 0
    # Build regex: word boundary on each end for single-token or short names
    # For multi-word names, allow flexible internal spacing.
    tokens = norm_name.split()
    if len(tokens) > 1:
        pattern = r"\b" + r"\s+".join(map(re.escape, tokens)) + r"\b"
    else:
        # Single token: require word boundary
        pattern = r"\b" + re.escape(norm_name) + r"\b"
    regex = re.compile(pattern, re.IGNORECASE)
    count = 0
    for p in media_paths:
        if regex.search(p):
            count += 1
    return count

def main():
    if not SUMMARY_SOURCE.exists():
        raise SystemExit(f"Source summary file not found: {SUMMARY_SOURCE}")
    names = load_names_with_summaries(SUMMARY_SOURCE)
    media_paths = collect_media_paths(DATA_ROOT)
    # Deduplicate names while preserving order
    seen = set()
    out_rows = []
    for name, summary in names:
        if name in seen:
            continue
        seen.add(name)
        occurrences = count_occurrences(name, media_paths)
        out_rows.append({"person_name": name, "recording_count": occurrences, "summary": summary})
    # Write output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["person_name", "recording_count", "summary"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"✅ Wrote {len(out_rows)} rows to {OUTPUT}")
    # Show a few samples
    for r in out_rows[:10]:
        print(f"{r['person_name']}: {r['recording_count']} :: {r['summary'][:70]}")

if __name__ == "__main__":
    main()
