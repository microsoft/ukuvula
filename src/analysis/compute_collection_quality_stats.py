#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Compute collection-level quality statistics from segment-level quality evaluations.

This script aggregates quality scores by collection and generates a LaTeX table
suitable for inclusion in the paper.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure src/ is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def compute_collection_statistics(input_file):
    """
    Compute mean, std, median, and range for each quality metric by collection.
    
    Args:
        input_file: Path to quality evaluation CSV with collection column
        
    Returns:
        DataFrame with collection-level statistics
    """
    print(f"Loading quality evaluation data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Extract collection from file_name (first part before /)
    if 'collection' not in df.columns:
        print("Extracting collection from file_name...")
        df['collection'] = df['file_name'].str.split('/').str[0]
    
    # Quality metric columns
    metrics = [
        'Fluency / grammaticality',
        'Coherence / consistency',
        'Completeness',
        'Redundancy',
        'Lexical richness'
    ]
    
    # Check if metrics exist
    missing_metrics = [m for m in metrics if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing quality metrics in input file: {missing_metrics}")
    
    print(f"\nTotal segments: {len(df)}")
    print(f"Collections: {df['collection'].nunique()}")
    print(f"Quality metrics: {metrics}")
    
    # Compute statistics for each collection
    results = []
    
    for collection in sorted(df['collection'].unique()):
        collection_data = df[df['collection'] == collection]
        n_segments = len(collection_data)
        
        print(f"\nProcessing {collection}: {n_segments} segments")
        
        row = {'Collection': collection, 'Segments': n_segments}
        
        for metric in metrics:
            values = collection_data[metric].dropna()
            
            if len(values) > 0:
                row[f'{metric}_mean'] = values.mean()
                row[f'{metric}_std'] = values.std()
                row[f'{metric}_median'] = values.median()
                row[f'{metric}_min'] = values.min()
                row[f'{metric}_max'] = values.max()
            else:
                row[f'{metric}_mean'] = np.nan
                row[f'{metric}_std'] = np.nan
                row[f'{metric}_median'] = np.nan
                row[f'{metric}_min'] = np.nan
                row[f'{metric}_max'] = np.nan
        
        results.append(row)
    
    stats_df = pd.DataFrame(results)
    return stats_df, metrics


def generate_latex_table(stats_df, metrics, output_file):
    """
    Generate a comprehensive LaTeX table with collection-level quality statistics.
    
    Args:
        stats_df: DataFrame with collection statistics
        metrics: List of metric names
        output_file: Path to save LaTeX table
    """
    # Abbreviate metric names for table
    metric_abbrev = {
        'Fluency / grammaticality': 'Fluency',
        'Coherence / consistency': 'Coherence',
        'Completeness': 'Complete',
        'Redundancy': 'Redund.',
        'Lexical richness': 'Lexical'
    }
    
    # Start building LaTeX table
    latex = []
    latex.append("\\begin{table}[ht]")
    latex.append("\\centering")
    latex.append("\\caption{Collection-Level Quality Statistics: Mean (Std) / Median / [Range] for Each Metric. "
                 "Scores on 0--100 scale; Redundancy inverted (higher = worse quality). "
                 "Statistics computed from segment-level GPT-4o quality evaluations.}")
    latex.append("\\label{tab:collection-quality-stats}")
    latex.append("\\small")
    latex.append("\\begin{tabular}{@{}l r *{5}{c}@{}}")
    latex.append("\\toprule")
    
    # Header row 1: Metric names
    header1 = "\\textbf{Collection} & \\textbf{Seg.}"
    for metric in metrics:
        header1 += f" & \\textbf{{{metric_abbrev[metric]}}}"
    header1 += " \\\\"
    latex.append(header1)
    
    latex.append("\\midrule")
    
    # Data rows
    for _, row in stats_df.iterrows():
        collection_name = row['Collection']
        # Shorten collection names for table
        if len(collection_name) > 30:
            collection_name = collection_name[:27] + "..."
        
        line = f"{collection_name} & {int(row['Segments'])}"
        
        for metric in metrics:
            mean = row[f'{metric}_mean']
            std = row[f'{metric}_std']
            median = row[f'{metric}_median']
            min_val = row[f'{metric}_min']
            max_val = row[f'{metric}_max']
            
            if pd.notna(mean):
                # Format as: mean (std) / median / [min-max]
                cell = f"{mean:.1f} ({std:.1f}) / {median:.0f} / [{int(min_val)}--{int(max_val)}]"
            else:
                cell = "N/A"
            
            line += f" & \\footnotesize {cell}"
        
        line += " \\\\"
        latex.append(line)
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    # Write to file
    latex_content = "\n".join(latex)
    
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"\n✓ LaTeX table saved to {output_file}")
    print("\nPreview:")
    print("=" * 80)
    print(latex_content)
    print("=" * 80)
    
    return latex_content


def generate_compact_latex_table(stats_df, metrics, output_file):
    """
    Generate a more compact LaTeX table showing only mean ± std.
    
    Args:
        stats_df: DataFrame with collection statistics
        metrics: List of metric names
        output_file: Path to save LaTeX table
    """
    # Abbreviate metric names for table
    metric_abbrev = {
        'Fluency / grammaticality': 'Fluency',
        'Coherence / consistency': 'Coherence',
        'Completeness': 'Complete',
        'Redundancy': 'Redund.',
        'Lexical richness': 'Lexical'
    }
    
    # Start building LaTeX table
    latex = []
    latex.append("\\begin{table}[ht]")
    latex.append("\\centering")
    latex.append("\\caption{Collection-Level Quality Statistics: Mean $\\pm$ Std for Each Metric. "
                 "Scores on 0--100 scale; Redundancy inverted (higher = worse quality). "
                 "Based on segment-level GPT-4o evaluations.}")
    latex.append("\\label{tab:collection-quality-compact}")
    latex.append("\\footnotesize")
    latex.append("\\begin{tabular}{@{}l r *{5}{c}@{}}")
    latex.append("\\toprule")
    
    # Header row
    header = "\\textbf{Collection} & \\textbf{n}"
    for metric in metrics:
        header += f" & \\textbf{{{metric_abbrev[metric]}}}"
    header += " \\\\"
    latex.append(header)
    
    latex.append("\\midrule")
    
    # Data rows
    for _, row in stats_df.iterrows():
        collection_name = row['Collection']
        # Shorten collection names for table
        if len(collection_name) > 35:
            collection_name = collection_name[:32] + "..."
        
        line = f"{collection_name} & {int(row['Segments'])}"
        
        for metric in metrics:
            mean = row[f'{metric}_mean']
            std = row[f'{metric}_std']
            
            if pd.notna(mean):
                # Format as: mean ± std
                cell = f"{mean:.1f} $\\pm$ {std:.1f}"
            else:
                cell = "N/A"
            
            line += f" & {cell}"
        
        line += " \\\\"
        latex.append(line)
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    # Write to file
    latex_content = "\n".join(latex)
    
    output_compact = Path(str(output_file).replace('.tex', '_compact.tex'))
    with open(output_compact, 'w') as f:
        f.write(latex_content)
    
    print(f"\n✓ Compact LaTeX table saved to {output_compact}")
    print("\nPreview:")
    print("=" * 80)
    print(latex_content)
    print("=" * 80)
    
    return latex_content


def main():
    """Main execution function."""
    # Paths
    input_file = Path("results/quality_evaluation.csv")
    output_file = Path("results/collection_quality_stats_table.tex")
    stats_csv = Path("results/collection_quality_statistics.csv")
    
    if not input_file.exists():
        print(f"❌ Error: Quality evaluation file not found: {input_file}")
        print("\nPlease run quality evaluation first:")
        print("  python quality_evaluation.py")
        return 1
    
    try:
        # Compute statistics
        stats_df, metrics = compute_collection_statistics(input_file)
        
        # Save statistics CSV
        stats_df.to_csv(stats_csv, index=False)
        print(f"\n✓ Statistics saved to {stats_csv}")
        
        # Generate LaTeX tables
        generate_latex_table(stats_df, metrics, output_file)
        generate_compact_latex_table(stats_df, metrics, output_file)
        
        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS BY COLLECTION")
        print("=" * 80)
        for _, row in stats_df.iterrows():
            print(f"\n{row['Collection']} ({int(row['Segments'])} segments):")
            for metric in metrics:
                mean = row[f'{metric}_mean']
                std = row[f'{metric}_std']
                print(f"  {metric:30s}: {mean:5.1f} ± {std:4.1f}")
        
        print("\n" + "=" * 80)
        print("✓ Collection quality statistics generated successfully!")
        print("=" * 80)
        print("\nNext steps:")
        print(f"1. Review statistics: {stats_csv}")
        print(f"2. Copy LaTeX table from: {output_file}")
        print("3. Insert into paper.tex in Section: Transcription Quality Evaluation")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
