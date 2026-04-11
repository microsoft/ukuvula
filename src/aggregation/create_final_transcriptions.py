#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Create a final aggregated transcription CSV from all collection-level aggregated transcriptions.

This script takes all CSV files from results/aggregated_transcriptions/ and combines them
into a single results/final_transcriptions.csv file with an additional 'collection' column
to identify the source collection.

Usage:
    python create_final_transcriptions.py
    python create_final_transcriptions.py --backup
    python create_final_transcriptions.py --input path/to/dir --output path/to/output.csv
"""

import argparse
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

def create_final_transcriptions(source_dir, output_file, backup=False):
    """
    Aggregate all collection-level transcription CSVs into a single final CSV.

    Args:
        source_dir: Path to the directory containing aggregated transcription CSVs.
        output_file: Path to the output CSV file.
        backup: If True, create a .csv.backup of any existing output file before overwriting.
    """

    source_dir = Path(source_dir)
    output_file = Path(output_file)

    # Check if source directory exists
    if not source_dir.exists():
        logger.error(f"Source directory {source_dir} does not exist")
        return None

    # Find all CSV files in the aggregated transcriptions directory
    csv_files = list(source_dir.glob("*.csv"))

    if not csv_files:
        logger.error(f"No CSV files found in {source_dir}")
        return None

    logger.info(f"Found {len(csv_files)} CSV files to aggregate")

    # Expected columns (in order)
    expected_columns = [
        'file_name', 'start_time', 'end_time', 'speaker',
        'transcription', 'confidence', 'duration', 'word_count'
    ]

    final_columns = ['collection'] + expected_columns

    aggregated_dfs = []
    total_rows = 0

    for csv_file in sorted(csv_files):
        try:
            logger.info(f"Processing {csv_file.name}")

            # Extract collection name from filename (remove .csv extension)
            collection_name = csv_file.stem

            # Read the CSV
            df = pd.read_csv(csv_file)

            # Check if the CSV has the expected columns
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"CSV {csv_file.name} missing columns: {missing_columns}")
                # Add missing columns with default values
                for col in missing_columns:
                    if col == 'speaker':
                        df[col] = 'Unknown'
                    elif col in ['confidence', 'duration']:
                        df[col] = 0.0
                    elif col == 'word_count':
                        df[col] = 0
                    else:
                        df[col] = ''

            # Add collection column
            df['collection'] = collection_name

            # Reorder columns to match final schema
            df = df[final_columns]

            aggregated_dfs.append(df)
            total_rows += len(df)

            logger.info(f"  Added {len(df)} rows from {collection_name}")

        except Exception as e:
            logger.error(f"Error processing {csv_file.name}: {e}")
            continue

    if not aggregated_dfs:
        logger.error("No valid CSV files could be processed")
        return None

    # Concatenate all DataFrames
    try:
        logger.info("Concatenating all collection transcriptions...")
        final_df = pd.concat(aggregated_dfs, ignore_index=True)

        # Sort by collection, file_name, and start_time for consistency
        logger.info("Sorting final dataset...")
        final_df = final_df.sort_values(['collection', 'file_name', 'start_time'],
                                       na_position='last')

        # Create results directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Create backup of existing file if requested
        if backup and output_file.exists():
            backup_file = output_file.with_suffix('.csv.backup')
            logger.info(f"Creating backup: {backup_file}")
            output_file.rename(backup_file)

        # Save final aggregated CSV
        logger.info(f"Saving final transcriptions to {output_file}")
        final_df.to_csv(output_file, index=False)

        # Summary statistics
        logger.info("=== Final Transcriptions Summary ===")
        logger.info(f"Total rows: {len(final_df):,}")
        logger.info(f"Total collections: {final_df['collection'].nunique()}")
        logger.info(f"Total unique files: {final_df['file_name'].nunique()}")
        logger.info(f"Output file: {output_file}")

        # Collection breakdown
        logger.info("\n=== Collection Breakdown ===")
        collection_counts = final_df['collection'].value_counts().sort_index()
        for collection, count in collection_counts.items():
            logger.info(f"  {collection}: {count:,} rows")

        # Additional statistics
        total_duration = final_df['duration'].sum() / 60  # Convert to minutes
        total_words = final_df['word_count'].sum()
        avg_confidence = final_df['confidence'].mean()

        logger.info(f"\n=== Content Statistics ===")
        logger.info(f"Total duration: {total_duration:,.1f} minutes ({total_duration/60:.1f} hours)")
        logger.info(f"Total words: {total_words:,}")
        logger.info(f"Average confidence: {avg_confidence:.3f}")

        return output_file

    except Exception as e:
        logger.error(f"Error creating final aggregation: {e}")
        return None

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate collection-level transcription CSVs into a single final CSV."
    )
    parser.add_argument(
        "--input",
        default="results/aggregated_transcriptions",
        help="Path to directory containing aggregated transcription CSVs (default: results/aggregated_transcriptions)",
    )
    parser.add_argument(
        "--output",
        default="results/final_transcriptions.csv",
        help="Path to the output CSV file (default: results/final_transcriptions.csv)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=False,
        help="Create a .csv.backup of any existing output file before overwriting",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite the output file without creating a backup (default behavior; opposite of --backup)",
    )
    return parser.parse_args()

def main():
    """Main function to create final transcriptions."""
    args = parse_args()

    # --force simply means "no backup", which is already the default.
    # If the user passes both --backup and --force, --force wins.
    do_backup = args.backup and not args.force

    logger.info("Starting final transcriptions aggregation...")

    output_file = create_final_transcriptions(
        source_dir=args.input,
        output_file=args.output,
        backup=do_backup,
    )

    if output_file and output_file.exists():
        logger.info(f"Successfully created {output_file}")

        # Verify the output file
        try:
            df = pd.read_csv(output_file)
            logger.info(f"Verification: Final CSV contains {len(df):,} rows and {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
        except Exception as e:
            logger.error(f"Error verifying output file: {e}")
    else:
        logger.error("Failed to create final transcriptions file")

if __name__ == "__main__":
    main()
