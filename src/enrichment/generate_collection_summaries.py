#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
GPT-based Collection Summary Generator

This script analyzes the transcribed content of each collection in results/aggregated_transcriptions/
and generates concise summaries describing the nature and themes of the content in each collection.
Uses Azure OpenAI with the same authentication approach as gpt_clustering_mandela.py.
"""

import sys
import pandas as pd
import json
import logging
import os
import csv
import random
from pathlib import Path
import time
from azure_openai_utils import setup_azure_openai
from typing import List, Dict

# Ensure src/ is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_collection_transcripts(aggregated_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all collection transcription CSV files"""
    collections = {}
    
    for csv_file in Path(aggregated_dir).glob("*.csv"):
        collection_name = csv_file.stem
        print(f"Loading collection: {collection_name}")
        
        try:
            df = pd.read_csv(csv_file)
            collections[collection_name] = df
            print(f"  - Loaded {len(df)} transcription segments")
        except Exception as e:
            print(f"  - Error loading {csv_file}: {e}")
    
    return collections

def sample_transcription_content(df: pd.DataFrame, max_chars: int = 8000) -> str:
    """
    Sample representative transcription content from a collection DataFrame
    to stay within GPT token limits while capturing diverse content
    """
    # Get transcription text
    if 'transcription' not in df.columns:
        return "No transcription content available"
    
    # Remove empty/null transcriptions
    valid_transcriptions = df['transcription'].dropna()
    valid_transcriptions = valid_transcriptions[valid_transcriptions.str.strip() != '']
    
    if len(valid_transcriptions) == 0:
        return "No valid transcription content available"
    
    # Sample from beginning, middle, and end to get diverse content
    total_segments = len(valid_transcriptions)
    
    if total_segments <= 20:
        # Use all segments for small collections
        sample_indices = range(total_segments)
    else:
        # Sample strategically: beginning (5), middle (5), end (5), random (5)
        beginning = list(range(0, min(5, total_segments)))
        middle_start = total_segments // 2 - 2
        middle = list(range(middle_start, min(middle_start + 5, total_segments)))
        end = list(range(max(0, total_segments - 5), total_segments))
        
        # Add some random samples
        random.seed(42)  # For reproducibility
        remaining = set(range(total_segments)) - set(beginning + middle + end)
        random_sample = random.sample(list(remaining), min(5, len(remaining)))
        
        sample_indices = sorted(set(beginning + middle + end + random_sample))
    
    # Combine sampled transcriptions
    sampled_text = ""
    for idx in sample_indices:
        segment_text = str(valid_transcriptions.iloc[idx]).strip()
        if len(sampled_text) + len(segment_text) + 10 > max_chars:
            break
        sampled_text += segment_text + " [SEGMENT_BREAK] "
    
    return sampled_text

def generate_collection_summary(collection_name: str, transcription_sample: str, client) -> str:
    """Generate a concise summary using Azure OpenAI GPT-4"""
    
    prompt = f"""Analyze the following transcribed audio content from the "{collection_name}" collection of the Nelson Mandela Foundation archive. 

Based on the sample transcriptions below, write a concise paragraph (3-5 sentences) that describes:
1. The primary type of content (interviews, speeches, conversations, etc.)
2. Main themes, topics, or subject matter discussed
3. Key figures, organizations, or events mentioned
4. The historical or cultural context of the materials

Keep the summary academic, informative, and suitable for a research paper. Focus on what researchers would find valuable in this collection.

Sample transcription content:
{transcription_sample}

Collection Summary:"""

    try:
        messages = [
            {"role": "system", "content": "You are an expert archivist and historian specializing in South African history and the Nelson Mandela Foundation's collections. Provide scholarly, concise analysis."},
            {"role": "user", "content": prompt}
        ]
        
        completion = client.chat.completions.create(
            model="gpt-4o",  # Updated model name for Azure OpenAI
            messages=messages,
            max_tokens=250,
            temperature=0.3,
            top_p=0.9,
            frequency_penalty=0.3,
            presence_penalty=0.1
        )
        
        summary = completion.choices[0].message.content.strip()
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary for {collection_name}: {e}")
        return f"Summary generation failed for {collection_name} collection."

def main():
    """Main function to process all collections and generate summaries"""
    
    # Setup
    aggregated_dir = "results/aggregated_transcriptions"
    output_file = "results/gpt_based_collection_summary.csv"
    
    print("🤖 GPT-Based Collection Summary Generator")
    print("=" * 60)
    
    # Setup Azure OpenAI client
    try:
        client = setup_azure_openai()
        print("✅ Azure OpenAI client initialized")
    except Exception as e:
        print(f"❌ Error initializing Azure OpenAI: {e}")
        print("Please ensure Azure credentials are properly configured")
        return
    
    # Load collections
    if not os.path.exists(aggregated_dir):
        print(f"❌ Directory not found: {aggregated_dir}")
        return
    
    collections = load_collection_transcripts(aggregated_dir)
    if not collections:
        print("❌ No collection files found")
        return
    
    print(f"\n📊 Found {len(collections)} collections to analyze")
    
    # Generate summaries
    results = []
    
    for i, (collection_name, df) in enumerate(collections.items(), 1):
        print(f"\n🔍 Analyzing {i}/{len(collections)}: {collection_name}")
        print(f"   Collection size: {len(df)} segments")
        
        # Sample content for GPT analysis
        sample_content = sample_transcription_content(df)
        content_length = len(sample_content)
        print(f"   Sample content length: {content_length} characters")
        
        if content_length < 100:
            print("   ⚠️ Very little content available for analysis")
        
        # Generate summary
        print("   🤖 Generating GPT-4 summary...")
        summary = generate_collection_summary(collection_name, sample_content, client)
        
        # Store result
        results.append({
            'collection_name': collection_name,
            'num_segments': len(df),
            'total_duration_hours': df['duration'].sum() / 3600 if 'duration' in df.columns else 0,
            'summary': summary
        })
        
        print(f"   ✅ Summary generated ({len(summary)} characters)")
        
        # Rate limiting - pause between requests
        if i < len(collections):
            print("   ⏳ Pausing to respect API rate limits...")
            time.sleep(2)
    
    # Save results
    print(f"\n💾 Saving summaries to {output_file}")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['collection_name', 'num_segments', 'total_duration_hours', 'summary']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print("✅ Collection summaries generated successfully!")
    
    # Display results
    print(f"\n📋 SUMMARY RESULTS:")
    print("=" * 60)
    
    for result in results:
        print(f"\n🏛️ {result['collection_name']}")
        print(f"   Segments: {result['num_segments']:,}")
        print(f"   Duration: {result['total_duration_hours']:.1f} hours")
        print(f"   Summary: {result['summary']}")

if __name__ == "__main__":
    main()