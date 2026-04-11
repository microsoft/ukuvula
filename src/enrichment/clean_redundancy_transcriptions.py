#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Clean transcription redundancy while preserving provenance.
Generates:
  - results/final_transcriptions.cleaned.csv (adds cleaned_transcription)
  - results/cleaning_diffs/<row_index>.json (diff operations)
  - results/cleaning_metrics.json (aggregate statistics)
  - results/cleaning_config.json (configuration used)

Heuristics (configurable):
  1. Whitespace normalization
  2. Filler collapse/removal
  3. Contiguous n-gram repetition capping
  4. Long-range loop collapse
  5. Redundant clause similarity collapse
  6. Repeated word stretch compression
  7. Ellipsis normalization

Run:
  python clean_redundancy_transcriptions.py --input results/final_transcriptions.csv

Use --dry-run to only compute metrics on sample subset.
"""
from __future__ import annotations
import re, json, argparse, math, os, csv, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Ensure src/ is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_CONFIG = {
    "filler_list": ["um", "uh", "erm", "mmm", "hmm", "ah", "eh"],
    "remove_fillers": False,
    "max_contiguous_repeats": 2,
    "min_ngram": 2,
    "max_ngram": 6,
    "long_range_repeat_threshold": 4,
    "long_range_spacing_limit": 12,
    "clause_similarity_threshold": 0.85,
    "min_tokens_for_full_clean": 10,
    "preserve_entity_like_capitalized": True,
    "max_word_stretch": 3,
    "sample_size_dry_run": 200,
}
STOPWORDS = set("the a an and or but if so to of in on at for from with by is are was were be been being as that this these those it its they them he she we you i our your their his her not".split())

ELLIPSIS_REGEX = re.compile(r"\.{4,}")
# Updated token/word regexes to preserve internal apostrophes and key punctuation
# Words: sequences of alphanumerics possibly joined by internal apostrophes (e.g., Mandela's, we'll)
TOKEN_REGEX = re.compile(
    r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*"          # word with internal apostrophes
    r"|\[.*?\]"                               # bracketed markup
    r"|\.\.\."                               # ellipsis token
    r"|[.!?,;:]"                               # sentence punctuation
    r"|[""'“”‘’()\-–—]"                      # quotes, parens, dashes
)
WORD_REGEX = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*")

def tokenize(text: str) -> List[str]:
    """Tokenize while retaining internal apostrophes and essential punctuation.
    Normalizes curly quotes to straight quotes before matching.
    """
    text = (text
            .replace('’', "'")
            .replace('‘', "'")
            .replace('“', '"')
            .replace('”', '"'))
    return TOKEN_REGEX.findall(text)

def detokenize(tokens: List[str]) -> str:
    out = []
    for i, tok in enumerate(tokens):
        if i == 0:
            out.append(tok)
            continue
        # No space before punctuation
        if tok in ".,!?;:" or tok == "...":
            out.append(tok)
        else:
            prev = out[-1]
            if prev.endswith("["):
                out.append(tok)
            else:
                out.append(" " + tok)
    return "".join(out)

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def collapse_fillers(tokens: List[str], cfg: Dict[str, Any], ops: List[Dict[str, Any]]) -> List[str]:
    fillers = set([f.lower() for f in cfg["filler_list"]])
    out = []
    i = 0
    removed = 0
    while i < len(tokens):
        t = tokens[i]
        low = t.lower()
        if low in fillers:
            # Count consecutive fillers
            j = i + 1
            while j < len(tokens) and tokens[j].lower() in fillers:
                j += 1
            span = tokens[i:j]
            if cfg["remove_fillers"]:
                removed += len(span)
                ops.append({"type": "fillers_removed", "count": len(span), "tokens": span})
            else:
                out.append(span[0])  # keep one
                if len(span) > 1:
                    removed += len(span) - 1
                    ops.append({"type": "fillers_collapsed", "count": len(span) - 1, "tokens": span})
            i = j
        else:
            out.append(t)
            i += 1
    return out

def contiguous_ngram_cap(tokens: List[str], cfg: Dict[str, Any], ops: List[Dict[str, Any]]) -> List[str]:
    if len(tokens) < cfg["min_ngram"]:
        return tokens
    max_rep = cfg["max_contiguous_repeats"]
    out = []
    i = 0
    while i < len(tokens):
        # Try longest n-gram first for efficiency
        capped = False
        for n in range(cfg["max_ngram"], cfg["min_ngram"] - 1, -1):
            if i + n > len(tokens):
                continue
            seq = tokens[i:i+n]
            # Count contiguous repeats
            j = i + n
            rep_count = 1
            while j + n <= len(tokens) and tokens[j:j+n] == seq:
                rep_count += 1
                j += n
            if rep_count > 1:
                keep = min(rep_count, max_rep)
                out.extend(seq * keep)
                removed = (rep_count - keep) * n
                if removed > 0:
                    ops.append({"type": "contiguous_ngram_cap", "ngram": seq, "original_repeats": rep_count, "kept": keep, "removed_tokens": removed})
                i = j
                capped = True
                break
        if not capped:
            out.append(tokens[i])
            i += 1
    return out

def long_range_loop_collapse(tokens: List[str], cfg: Dict[str, Any], ops: List[Dict[str, Any]]) -> List[str]:
    if len(tokens) < cfg["min_tokens_for_full_clean"]:
        return tokens
    positions = defaultdict(list)
    max_n = cfg["max_ngram"]
    min_n = cfg["min_ngram"]
    # Index n-grams
    for i in range(len(tokens)):
        for n in range(min_n, max_n+1):
            if i + n <= len(tokens):
                ng = tuple(tokens[i:i+n])
                positions[ng].append(i)
    to_remove = set()
    for ng, pos_list in positions.items():
        if len(pos_list) >= cfg["long_range_repeat_threshold"]:
            # compute average spacing
            spacings = [pos_list[i+1]-pos_list[i] for i in range(len(pos_list)-1)]
            avg_spacing = sum(spacings)/len(spacings) if spacings else math.inf
            if avg_spacing <= cfg["long_range_spacing_limit"]:
                # keep first two occurrences only
                for idx in pos_list[2:]:
                    for k in range(idx, idx+len(ng)):
                        to_remove.add(k)
                ops.append({"type": "long_range_loop_collapse", "ngram": list(ng), "occurrences": len(pos_list), "removed_tokens": len(pos_list[2:]) * len(ng)})
    cleaned = [tok for i, tok in enumerate(tokens) if i not in to_remove]
    return cleaned

def clause_similarity_collapse(tokens: List[str], cfg: Dict[str, Any], ops: List[Dict[str, Any]]) -> List[str]:
    if len(tokens) < cfg["min_tokens_for_full_clean"]:
        return tokens
    # Split into clauses
    clause_indices = [0]
    for i, t in enumerate(tokens):
        if t in ['.', '?', '!', ';', ':']:
            clause_indices.append(i+1)
    if clause_indices[-1] != len(tokens):
        clause_indices.append(len(tokens))
    clauses = []
    for i in range(len(clause_indices)-1):
        start, end = clause_indices[i], clause_indices[i+1]
        clauses.append((start, end, tokens[start:end]))
    to_remove_ranges = []
    def clause_key(cl):
        words = [w.lower() for w in cl if WORD_REGEX.match(w)]
        return set([w for w in words if w not in STOPWORDS])
    for i in range(len(clauses)-1):
        a = clauses[i]; b = clauses[i+1]
        set_a = clause_key(a[2]); set_b = clause_key(b[2])
        if not set_a or not set_b:
            continue
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        sim = inter / union if union else 0
        # Basic entity-like detection: starts with Capitalized word
        entity_guard = False
        if cfg["preserve_entity_like_capitalized"]:
            first_b = b[2][0] if b[2] else ''
            if first_b and first_b[0].isupper():
                entity_guard = True
        if sim >= cfg["clause_similarity_threshold"] and not entity_guard:
            to_remove_ranges.append((b[0], b[1]))
            ops.append({"type": "clause_similarity_collapse", "similarity": round(sim,3), "removed_clause": detokenize(b[2])})
    remove_positions = set()
    for (s,e) in to_remove_ranges:
        for k in range(s,e):
            remove_positions.add(k)
    cleaned = [tok for i, tok in enumerate(tokens) if i not in remove_positions]
    return cleaned

def word_stretch_compress(tokens: List[str], cfg: Dict[str, Any], ops: List[Dict[str, Any]]) -> List[str]:
    out = []
    i = 0
    max_stretch = cfg["max_word_stretch"]
    while i < len(tokens):
        t = tokens[i]
        j = i+1
        while j < len(tokens) and tokens[j].lower() == t.lower():
            j += 1
        stretch_len = j - i
        if stretch_len > max_stretch:
            out.extend([t]*max_stretch)
            ops.append({"type": "word_stretch_compress", "word": t.lower(), "original_len": stretch_len, "kept": max_stretch, "removed_tokens": stretch_len - max_stretch})
        else:
            out.extend(tokens[i:j])
        i = j
    return out

def normalize_ellipsis(text: str) -> str:
    return ELLIPSIS_REGEX.sub("...", text)

def process_text(text: str, cfg: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    ops: List[Dict[str, Any]] = []
    original = text
    text = normalize_ellipsis(text)
    text = normalize_whitespace(text)
    tokens = tokenize(text)
    if len(tokens) < cfg["min_tokens_for_full_clean"]:
        return original.strip(), [{"type": "skipped_short_segment", "length": len(tokens)}]
    # Apply heuristics sequentially
    tokens = collapse_fillers(tokens, cfg, ops)
    tokens = contiguous_ngram_cap(tokens, cfg, ops)
    tokens = long_range_loop_collapse(tokens, cfg, ops)
    tokens = clause_similarity_collapse(tokens, cfg, ops)
    tokens = word_stretch_compress(tokens, cfg, ops)
    cleaned = detokenize(tokens)
    cleaned = normalize_whitespace(cleaned)
    return cleaned, ops

def compute_bigram_repetition_ratio(text: str) -> float:
    # Normalize curly quotes prior to word extraction for consistency
    text = (text
            .replace('’', "'")
            .replace('‘', "'")
            .replace('“', '"')
            .replace('”', '"'))
    words = [w for w in WORD_REGEX.findall(text.lower())]
    if len(words) < 2:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    c = Counter(bigrams)
    repeated = sum(v for k,v in c.items() if v>1)
    return repeated / len(bigrams)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to original transcription CSV')
    ap.add_argument('--output', default='results/final_transcriptions_cleaned_final.csv')
    ap.add_argument('--config', default='', help='Optional path to config JSON')
    ap.add_argument('--dry-run', action='store_true', help='Do not write cleaned CSV, just metrics on sample')
    ap.add_argument('--sample', type=int, default=0, help='Limit processing to first N rows (for quick tests)')
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config,'r') as f:
            user_cfg = json.load(f)
        cfg.update(user_cfg)
    os.makedirs('results/cleaning_diffs', exist_ok=True)

    # Load CSV (expect at least a 'transcription' column; pass through others)
    with open(args.input, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if args.sample > 0:
        rows = rows[:args.sample]

    # Dry-run subset
    if args.dry_run:
        subset = rows[:cfg['sample_size_dry_run']]
        ratios = [compute_bigram_repetition_ratio(r.get('transcription','')) for r in subset]
        print(f"Dry run: {len(subset)} rows, avg repeated bigram ratio={sum(ratios)/len(ratios):.4f}")
        print("No cleaning performed (dry-run).")
        return

    metrics = {
        'total_rows_processed': 0,
        'rows_changed': 0,
        'total_tokens_original': 0,
        'total_tokens_cleaned': 0,
        'fillers_removed': 0,
        'fillers_collapsed': 0,
        'ngram_caps': 0,
        'long_range_collapses': 0,
        'clause_collapses': 0,
        'word_stretch_compress': 0,
        'bigram_repetition_ratio_before': [],
        'bigram_repetition_ratio_after': [],
        'top_removed_ngrams': Counter(),
    }

    cleaned_rows = []
    for idx, row in enumerate(rows):
        original_text = row.get('transcription','')
        metrics['total_rows_processed'] += 1
        metrics['total_tokens_original'] += len(re.findall(r"\w+", original_text))
        ratio_before = compute_bigram_repetition_ratio(original_text)
        cleaned_text, ops = process_text(original_text, cfg)
        ratio_after = compute_bigram_repetition_ratio(cleaned_text)
        metrics['bigram_repetition_ratio_before'].append(ratio_before)
        metrics['bigram_repetition_ratio_after'].append(ratio_after)
        if cleaned_text != original_text.strip():
            metrics['rows_changed'] += 1
        metrics['total_tokens_cleaned'] += len(re.findall(r"\w+", cleaned_text))
        # Count operation types
        for op in ops:
            t = op['type']
            if t == 'fillers_removed':
                metrics['fillers_removed'] += op['count']
            elif t == 'fillers_collapsed':
                metrics['fillers_collapsed'] += op['count']
            elif t == 'contiguous_ngram_cap':
                metrics['ngram_caps'] += 1
                ng = ' '.join(op['ngram'])
                metrics['top_removed_ngrams'][ng] += op['removed_tokens']
            elif t == 'long_range_loop_collapse':
                metrics['long_range_collapses'] += 1
                ng = ' '.join(op['ngram'])
                metrics['top_removed_ngrams'][ng] += op['removed_tokens']
            elif t == 'clause_similarity_collapse':
                metrics['clause_collapses'] += 1
            elif t == 'word_stretch_compress':
                metrics['word_stretch_compress'] += 1
        # Diff log
        diff_path = f"results/cleaning_diffs/{idx}.json"
        with open(diff_path,'w') as df:
            json.dump({
                'index': idx,
                'original': original_text,
                'cleaned': cleaned_text,
                'operations': ops,
                'ratio_before': ratio_before,
                'ratio_after': ratio_after,
            }, df, ensure_ascii=False, indent=2)
        new_row = row.copy()
        new_row['cleaned_transcription'] = cleaned_text
        cleaned_rows.append(new_row)

    # Write cleaned CSV
    fieldnames = list(cleaned_rows[0].keys()) if cleaned_rows else []
    with open(args.output,'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    # Final metrics
    metrics_summary = {
        k: (sum(v)/len(v) if isinstance(v, list) and v and k.startswith('bigram_repetition_ratio') else v)
        for k,v in metrics.items()
    }
    metrics_summary['avg_bigram_repetition_ratio_before'] = metrics_summary.pop('bigram_repetition_ratio_before')
    metrics_summary['avg_bigram_repetition_ratio_after'] = metrics_summary.pop('bigram_repetition_ratio_after')
    metrics_summary['token_reduction_percent'] = (1 - (metrics['total_tokens_cleaned']/metrics['total_tokens_original']))*100 if metrics['total_tokens_original'] else 0
    metrics_summary['top_removed_ngrams'] = metrics['top_removed_ngrams'].most_common(20)

    with open('results/cleaning_metrics.json','w') as mf:
        json.dump(metrics_summary, mf, ensure_ascii=False, indent=2)
    with open('results/cleaning_config.json','w') as cf:
        json.dump(cfg, cf, ensure_ascii=False, indent=2)

    print("Cleaning complete.")
    print(json.dumps(metrics_summary, indent=2))

if __name__ == '__main__':
    main()
