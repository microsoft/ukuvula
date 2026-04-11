# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Aggregate transcriptions with GPT-generated scope notes for the Mandela Foundation archive.

This script reads the cleaned final transcription CSV, aggregates transcription text
per recording into a structured note, and generates a concise scope-and-content note
for each recording using Azure OpenAI GPT models. Output is written to a new CSV.
"""

import os
import csv
import sys
from collections import defaultdict, Counter
from pathlib import Path
import re
import json
import logging
import math
import pandas as pd
import argparse
from azure_openai_utils import setup_azure_openai

# Ensure src/ is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

INPUT_PATH = Path("results/final_transcriptions_cleaned_final.csv")
OUTPUT_PATH = Path("results/transcriptions_with_scope_and_aggregated_note.csv")

HEADER_LINE = "Transcription done using AI in collaboration with Microsoft AI for Good Lab"

STOPWORDS = set("""
a an the and or of to in on for with without within by from at as is are was were be been being
this that these those it its into over under above below up down out off about more most some any all no not
we you he she they them his her their our us me my your yours ours theirs
""".split())

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
TIME_FMT_RE = re.compile(r"^\d{2}:\d{2}(?::\d{2})?$")
ARTIFACTS_RE = re.compile(r"\b(?:al\s*icons?|gov(?:ernment)?\s*Province|concept\s*area|life\s*in\s*exile)\b", re.IGNORECASE)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


HEADER = HEADER_LINE
SENT_END_REGEX = re.compile(r"(?<=[.!?])\s+")
MULTISPACE = re.compile(r"\s{2,}")
DASH_FIX = re.compile(r"\s*[–—-]+\s*")
NOISE = re.compile(r"\b(?:uh|um|erm|hmm|inaudible|unintelligible|[\x00-\x1F])\b", re.IGNORECASE)
CURLY_QUOTES = re.compile(r"[""'']")

def build_note(rows):
    # Aggregate only cleaned transcriptions per segment, one per line, prefixed by header.
    parts = [HEADER]
    for _, r in rows.iterrows():
        text = str(r.get("cleaned_transcription", r.get("transcription", ""))).strip()
        # normalize quotes and spaces
        seg = CURLY_QUOTES.sub("'", text)
        seg = MULTISPACE.sub(" ", seg)
        parts.append(seg)
    return "\n".join(parts)

def split_sentences(text):
    # Basic sentence segmentation with cleanup
    raw = SENT_END_REGEX.split(text)
    sents = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        s = CURLY_QUOTES.sub("'", s)
        s = NOISE.sub("", s)
        s = DASH_FIX.sub(" - ", s)
        s = MULTISPACE.sub(" ", s)
        s = ARTIFACTS_RE.sub(" ", s)
        # drop header lines and timestamps-only
        if s.startswith(HEADER):
            continue
        if re.match(r"^\d{2}:\d{2}(?::\d{2})?\s*-\s*$", s):
            continue
        # remove leading timestamp if present
        s = re.sub(r"^\d{2}:\d{2}(?::\d{2})?\s*-\s*", "", s)
        sents.append(s)
    return sents

def score_sentence(sent, word_freq):
    words = re.findall(r"[A-Za-z0-9']+", sent.lower())
    if not words:
        return 0.0
    return sum(word_freq[w] for w in words) / math.sqrt(len(words))

def build_scope_with_gpt(client, collection: str, file_name: str, note_text: str, max_tokens=220):
    """Use Azure OpenAI GPT-4o to generate a clean 2–3 sentence scope summary (<=5 sentences max)."""
    # Strip header from Note; timestamps are not included in Note aggregation now
    lines = [ln.strip() for ln in note_text.splitlines() if ln.strip()]
    cleaned_lines = []
    for ln in lines:
        if ln.startswith(HEADER):
            continue
        cleaned_lines.append(ln)
    note_clean = " \n".join(cleaned_lines)

    # Construct prompt leveraging collection and file_name context and aggregated Note text
    context_line = f"Collection: {collection}\nFile name: {file_name}"
    instructions = (
        "You are an archivist summarizing a recording for a scope and content field. "
        "Write 2–3 well-formed sentences (maximum 5) describing what the recording is about. "
        "Focus on who is involved, the event or setting, and the main activity or discussion. "
        "Avoid copying verbatim lines, fix minor typos, and ensure proper punctuation."
    )
    user_content = (
        f"{instructions}\n\n{context_line}\n\nAggregated transcripts (Note, timestamps removed):\n" 
        + note_clean[:6000]  # safety cap on prompt size
    )
    messages = [
        {"role": "system", "content": "You are a professional archival cataloger. Produce concise, readable scope notes."},
        {"role": "user", "content": user_content}
    ]
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
            top_p=0.95,
        )
        summary = completion.choices[0].message.content.strip()
        # Post-cleanup: collapse excessive whitespace and enforce sentence cap
        summary = re.sub(r"\s+", " ", summary)
        # Hard cap sentences to 5
        sentences = re.split(r"(?<=[.!?])\s+", summary)
        summary = " ".join(sentences[:5]).strip()
        return summary
    except Exception as e:
        logger.warning(f"GPT summarization failed, falling back to simple heuristic: {e}")
        # Minimal fallback: first 2 cleaned sentences from note
        sents = split_sentences(note_text)
        return finalize_summary(sents[:3]) if sents else ""

def finalize_summary(sent_list):
    # Clean up spacing, ensure proper punctuation termination
    cleaned = []
    for s in sent_list:
        s = s.strip()
        s = CURLY_QUOTES.sub("'", s)
        s = MULTISPACE.sub(" ", s)
        # Basic capitalization fix
        if s and s[0].islower():
            s = s[0].upper() + s[1:]
        if s and s[-1] not in ".!?":
            s = s + "."
        cleaned.append(s)
    return " ".join(cleaned)

def main():
    parser = argparse.ArgumentParser(description="Aggregate notes and optionally summarize scope and content.")
    parser.add_argument("--mode", choices=["notes-only", "summarize", "full"], default="full",
                        help="Operation mode: 'notes-only' writes Collection/File name/Note; 'summarize' reads existing CSV and fills Scope and content; 'full' does both in one pass.")
    parser.add_argument("--input", default="results/final_transcriptions_cleaned_final.csv",
                        help="Input CSV of segments (default: results/final_transcriptions_cleaned_final.csv)")
    parser.add_argument("--output", default="results/transcriptions_with_scope_and_aggregated_note.csv",
                        help="Output CSV path (default: results/transcriptions_with_scope_and_aggregated_note.csv)")
    args = parser.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    if args.mode in ("notes-only", "full"):
        if not inp.exists():
            raise FileNotFoundError(f"Input file not found: {inp}")
        df = pd.read_csv(inp)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        required_cols = {"collection", "file_name", "start_time"}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns in input: {missing}")
        text_col = "cleaned_transcription" if "cleaned_transcription" in df.columns else "transcription"
        df = df.sort_values(["collection", "file_name", "start_time"], kind="stable")
        groups = df.groupby(["collection", "file_name"], sort=False)
        rows = []
        for (collection, file_name), g in groups:
            # Build Note from cleaned_transcription with timestamps and blank lines between segments
            note_parts = [HEADER, ""]  # Header followed by blank line
            for _, r in g.iterrows():
                seg_text = str(r.get(text_col, "")).strip()
                if seg_text:
                    # Format timestamp
                    start_time = str(r.get("start_time", "")).strip()
                    end_time = str(r.get("end_time", "")).strip()
                    if start_time and end_time:
                        timestamp = f"{start_time} - {end_time}:"
                    elif start_time:
                        timestamp = f"{start_time}:"
                    else:
                        timestamp = ""
                    
                    # Add timestamp and segment text, followed by blank line
                    if timestamp:
                        note_parts.append(timestamp)
                    note_parts.append(seg_text)
                    note_parts.append("")  # Blank line between segments
            
            # Remove trailing blank line
            if note_parts and note_parts[-1] == "":
                note_parts = note_parts[:-1]
            
            note_text = "\n".join(note_parts)
            rows.append({
                "Collection": collection,
                "File name": file_name,
                "Note": note_text,
            })
        out_df = pd.DataFrame(rows)
        # If full mode, create the file now; summarize will read and append column
        out_df.to_csv(outp, index=False)
        print(f"Wrote {len(out_df)} rows (notes-only) to {outp}")

    if args.mode in ("summarize", "full"):
        # Read current output and append/replace Scope and content via GPT
        if not outp.exists():
            raise FileNotFoundError(f"Output file for summarization not found: {outp}")
        df_out = pd.read_csv(outp)
        # Normalize columns for safety
        col_map = {c: c for c in df_out.columns}
        # Ensure required columns exist
        for req in ["Collection", "File name", "Note"]:
            if req not in df_out.columns:
                raise ValueError(f"Missing required column '{req}' in output for summarization.")
        client = setup_azure_openai()
        scopes = []
        for _, row in df_out.iterrows():
            scope = build_scope_with_gpt(client, str(row["Collection"]), str(row["File name"]), str(row["Note"]))
            scopes.append(scope)
        df_out["Scope and content"] = scopes
        df_out.to_csv(outp, index=False)
        print(f"Updated '{outp}' with Scope and content for {len(df_out)} rows")

if __name__ == "__main__":
    main()
