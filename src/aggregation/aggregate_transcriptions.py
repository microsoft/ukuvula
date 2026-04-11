#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Aggregate transcription CSVs from each directory under results/nmf_recordings/
into a single CSV per directory under results/aggregated_transcriptions/.

For each subdirectory in results/nmf_recordings/, collect all .csv files and
concatenate them into a single aggregated CSV with consistent columns:
file_name, start_time, end_time, speaker, transcription, confidence, duration, word_count

Usage:
    python aggregate_transcriptions.py
"""

import os
import sys
import pandas as pd
from pathlib import Path
import logging

# Ensure src/ is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def aggregate_directory_transcriptions(source_dir: Path, output_dir: Path):
    """
    Aggregate all CSV transcriptions from a single directory.
    
    Args:
        source_dir: Path to the source directory containing CSV files
        output_dir: Path to the output directory for aggregated CSV
    """
    # Find all CSV files in the source directory
    csv_files = list(source_dir.glob("*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {source_dir}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files in {source_dir.name}")
    
    # Expected columns (in order)
    expected_columns = [
        'file_name', 'start_time', 'end_time', 'speaker', 
        'transcription', 'confidence', 'duration', 'word_count'
    ]
    
    aggregated_dfs = []
    
    for csv_file in csv_files:
        try:
            logger.debug(f"Processing {csv_file.name}")
            df = pd.read_csv(csv_file)
            
            # Check if the CSV has the expected columns
            if not all(col in df.columns for col in expected_columns):
                logger.warning(f"CSV {csv_file.name} missing expected columns. "
                             f"Expected: {expected_columns}, Found: {list(df.columns)}")
                # Still include it but with available columns
                available_cols = [col for col in expected_columns if col in df.columns]
                df = df[available_cols]
                
                # Add missing columns with default values
                for col in expected_columns:
                    if col not in df.columns:
                        if col == 'speaker':
                            df[col] = 'Unknown'
                        elif col in ['confidence', 'duration']:
                            df[col] = 0.0
                        elif col == 'word_count':
                            df[col] = 0
                        else:
                            df[col] = ''
            
            # Reorder columns to match expected order
            df = df[expected_columns]
            aggregated_dfs.append(df)
            
        except Exception as e:
            logger.error(f"Error processing {csv_file.name}: {e}")
            continue
    
    if not aggregated_dfs:
        logger.error(f"No valid CSV files could be processed from {source_dir}")
        return
    
    # Concatenate all DataFrames
    try:
        aggregated_df = pd.concat(aggregated_dfs, ignore_index=True)
        
        # Sort by file_name and start_time for consistency
        aggregated_df = aggregated_df.sort_values(['file_name', 'start_time'], 
                                                  na_position='last')
        
        # Create output file path
        output_file = output_dir / f"{source_dir.name}.csv"
        
        # Save aggregated CSV
        aggregated_df.to_csv(output_file, index=False)
        
        logger.info(f"Aggregated {len(aggregated_dfs)} files into {output_file}")
        logger.info(f"Total rows: {len(aggregated_df)}")
        
    except Exception as e:
        logger.error(f"Error aggregating DataFrames for {source_dir}: {e}")

def main():
    """Main function to aggregate transcriptions from all directories."""
    
    # Define paths
    source_base = Path("results/nmf_recordings/")
    output_base = Path("results/aggregated_transcriptions")
    
    # Check if source directory exists
    if not source_base.exists():
        logger.error(f"Source directory {source_base} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_base.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_base}")
    
    # Get all subdirectories in source_base
    subdirectories = [d for d in source_base.iterdir() if d.is_dir()]
    
    if not subdirectories:
        logger.warning(f"No subdirectories found in {source_base}")
        return
    
    logger.info(f"Found {len(subdirectories)} directories to process")
    
    # Process each subdirectory
    processed_count = 0
    for subdir in subdirectories:
        logger.info(f"Processing directory: {subdir.name}")
        
        try:
            aggregate_directory_transcriptions(subdir, output_base)
            processed_count += 1
        except Exception as e:
            logger.error(f"Failed to process directory {subdir.name}: {e}")
            continue
    
    logger.info(f"Aggregation complete. Processed {processed_count}/{len(subdirectories)} directories")
    
    # Summary report
    if output_base.exists():
        output_files = list(output_base.glob("*.csv"))
        logger.info(f"Generated {len(output_files)} aggregated CSV files in {output_base}")
        
        # Log file sizes for verification
        for output_file in output_files:
            try:
                df = pd.read_csv(output_file)
                logger.info(f"  {output_file.name}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"  {output_file.name}: Error reading for summary - {e}")

if __name__ == "__main__":
    main()