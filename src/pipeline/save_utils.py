# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Utilities for saving transcription results in various formats.

This module provides functions for saving processed transcription results
to structured files (CSV, JSON) with proper formatting and metadata.
"""

import os
import csv
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime

# Setup logger
logger = logging.getLogger(__name__)

class TranscriptionSaver:
    """
    Utility class for saving transcription results in different formats.
    """
    
    def __init__(self, output_dir: str, encoding: str = 'utf-8'):
        """
        Initialize the transcription saver.
        
        Args:
            output_dir (str): Output directory for saved files
            encoding (str): Text encoding for output files
        """
        self.output_dir = Path(output_dir)
        self.encoding = encoding
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV field names
        self.csv_fieldnames = [
            'file_name', 'start_time', 'end_time', 'speaker', 
            'transcription', 'confidence', 'duration', 'word_count'
        ]
    
    def format_time(self, seconds: float) -> str:
        """
        Format time in seconds to MM:SS format.
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            str: Formatted time string
        """
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def prepare_segment_data(self, segment: Dict[str, Any], file_name: str) -> Dict[str, Any]:
        """
        Prepare segment data for saving.
        
        Args:
            segment (Dict[str, Any]): Transcription segment
            file_name (str): Source file name
            
        Returns:
            Dict[str, Any]: Formatted segment data
        """
        # Extract timing information
        start_time = segment.get('start', 0.0)
        end_time = segment.get('end', start_time)
        duration = end_time - start_time
        
        # Extract text and metadata
        text = segment.get('text', '').strip()
        speaker = segment.get('speaker', 'Speaker 1')
        confidence = segment.get('confidence', segment.get('avg_logprob', 0.0))
        
        # Calculate word count
        word_count = len(text.split()) if text else 0
        
        # Prefer full relative path provenance if available (source_relpath set by pipeline); fallback to basename.
        display_name = segment.get('source_relpath') or segment.get('source_file', file_name)

        return {
            'file_name': display_name,
            'start_time': self.format_time(start_time),
            'end_time': self.format_time(end_time),
            'speaker': speaker,
            'transcription': text,
            'confidence': round(confidence, 3) if confidence else 0.0,
            'duration': round(duration, 2),
            'word_count': word_count,
            'start_seconds': start_time,
            'end_seconds': end_time
        }
    
    def save_to_csv(self, segments: List[Dict[str, Any]], file_name: str, 
                   output_filename: str = "transcription.csv") -> str:
        """
        Save transcription segments to CSV file.
        
        Args:
            segments (List[Dict[str, Any]]): Transcription segments
            file_name (str): Source file name
            output_filename (str): Output CSV filename
            
        Returns:
            str: Path to saved CSV file
        """
        try:
            output_path = self.output_dir / output_filename
            
            # Prepare data
            csv_data = []
            for segment in segments:
                segment_data = self.prepare_segment_data(segment, file_name)
                csv_data.append(segment_data)
            
            # Write CSV file
            with open(output_path, 'w', newline='', encoding=self.encoding) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_fieldnames)
                
                # Write header
                writer.writeheader()
                
                # Write data
                for row in csv_data:
                    # Only include fields that exist in fieldnames
                    filtered_row = {k: v for k, v in row.items() if k in self.csv_fieldnames}
                    writer.writerow(filtered_row)
            
            logger.info(f"CSV saved: {output_path} ({len(csv_data)} segments)")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save CSV: {str(e)}")
            raise
    
    def save_to_json(self, segments: List[Dict[str, Any]], file_name: str,
                    output_filename: str = "transcription.json", 
                    include_metadata: bool = True) -> str:
        """
        Save transcription segments to JSON file.
        
        Args:
            segments (List[Dict[str, Any]]): Transcription segments
            file_name (str): Source file name
            output_filename (str): Output JSON filename
            include_metadata (bool): Whether to include metadata
            
        Returns:
            str: Path to saved JSON file
        """
        try:
            output_path = self.output_dir / output_filename
            
            # Prepare data
            json_data = {
                "source_file": file_name,
                "total_segments": len(segments),
                "created_at": datetime.now().isoformat(),
                "segments": []
            }
            
            if include_metadata:
                json_data["metadata"] = {
                    "total_duration": max([seg.get('end', 0) for seg in segments]) if segments else 0,
                    "total_words": sum([len(seg.get('text', '').split()) for seg in segments]),
                    "average_confidence": sum([seg.get('confidence', seg.get('avg_logprob', 0)) for seg in segments]) / len(segments) if segments else 0,
                    "speakers": list(set([seg.get('speaker', 'Speaker 1') for seg in segments]))
                }
            
            # Add segments
            for segment in segments:
                segment_data = self.prepare_segment_data(segment, file_name)
                json_data["segments"].append(segment_data)
            
            # Write JSON file
            with open(output_path, 'w', encoding=self.encoding) as jsonfile:
                json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON saved: {output_path} ({len(segments)} segments)")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save JSON: {str(e)}")
            raise
    
    def save_to_txt(self, segments: List[Dict[str, Any]], file_name: str,
                   output_filename: str = "transcription.txt", 
                   include_timestamps: bool = True,
                   include_speakers: bool = True) -> str:
        """
        Save transcription segments to plain text file.
        
        Args:
            segments (List[Dict[str, Any]]): Transcription segments
            file_name (str): Source file name
            output_filename (str): Output text filename
            include_timestamps (bool): Whether to include timestamps
            include_speakers (bool): Whether to include speaker labels
            
        Returns:
            str: Path to saved text file
        """
        try:
            output_path = self.output_dir / output_filename
            
            with open(output_path, 'w', encoding=self.encoding) as txtfile:
                # Write header
                txtfile.write(f"Transcription for: {file_name}\n")
                txtfile.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                txtfile.write(f"Total segments: {len(segments)}\n")
                txtfile.write("-" * 50 + "\n\n")
                
                # Write segments
                for segment in segments:
                    segment_data = self.prepare_segment_data(segment, file_name)
                    
                    line_parts = []
                    
                    # Add timestamp
                    if include_timestamps:
                        line_parts.append(f"[{segment_data['start_time']} - {segment_data['end_time']}]")
                    
                    # Add speaker
                    if include_speakers:
                        line_parts.append(f"{segment_data['speaker']}:")
                    
                    # Add text
                    line_parts.append(segment_data['transcription'])
                    
                    txtfile.write(" ".join(line_parts) + "\n\n")
            
            logger.info(f"Text file saved: {output_path} ({len(segments)} segments)")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save text file: {str(e)}")
            raise
    
    def create_summary_report(self, all_results: Dict[str, List[Dict[str, Any]]], 
                            output_filename: str = "processing_summary.json") -> str:
        """
        Create a summary report for all processed files.
        
        Args:
            all_results (Dict[str, List[Dict[str, Any]]]): Results for all files
            output_filename (str): Output filename for summary
            
        Returns:
            str: Path to saved summary file
        """
        try:
            output_path = self.output_dir / output_filename
            
            # Calculate summary statistics
            total_files = len(all_results)
            total_segments = sum(len(segments) for segments in all_results.values())
            total_duration = 0
            total_words = 0
            all_speakers = set()
            
            file_details = []
            
            for file_name, segments in all_results.items():
                if segments:
                    file_duration = max([seg.get('end', 0) for seg in segments])
                    file_words = sum([len(seg.get('text', '').split()) for seg in segments])
                    file_speakers = set([seg.get('speaker', 'Speaker 1') for seg in segments])
                    avg_confidence = sum([seg.get('confidence', seg.get('avg_logprob', 0)) for seg in segments]) / len(segments)
                    
                    total_duration += file_duration
                    total_words += file_words
                    all_speakers.update(file_speakers)
                    
                    file_details.append({
                        'file_name': file_name,
                        'segments': len(segments),
                        'duration_seconds': round(file_duration, 2),
                        'duration_formatted': self.format_time(file_duration),
                        'word_count': file_words,
                        'speakers': list(file_speakers),
                        'average_confidence': round(avg_confidence, 3)
                    })
            
            # Create summary
            summary = {
                'processing_summary': {
                    'generated_at': datetime.now().isoformat(),
                    'total_files_processed': total_files,
                    'total_segments': total_segments,
                    'total_duration_seconds': round(total_duration, 2),
                    'total_duration_formatted': self.format_time(total_duration),
                    'total_words': total_words,
                    'unique_speakers': list(all_speakers),
                    'average_segments_per_file': round(total_segments / total_files, 2) if total_files > 0 else 0
                },
                'file_details': file_details
            }
            
            # Save summary
            with open(output_path, 'w', encoding=self.encoding) as jsonfile:
                json.dump(summary, jsonfile, indent=2, ensure_ascii=False)
            
            logger.info(f"Summary report saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to create summary report: {str(e)}")
            raise

def save_segments_csv(segments: List[Dict[str, Any]], output_path: str, 
                     file_name: str = "audio_file") -> bool:
    """
    Simple function to save segments to CSV.
    
    Args:
        segments (List[Dict[str, Any]]): Transcription segments
        output_path (str): Output CSV file path
        file_name (str): Source file name
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        output_dir = os.path.dirname(output_path)
        saver = TranscriptionSaver(output_dir)
        saver.save_to_csv(segments, file_name, os.path.basename(output_path))
        return True
    except Exception as e:
        logger.error(f"Failed to save CSV: {str(e)}")
        return False

def save_folder_transcription(segments: List[Dict[str, Any]], folder_path: str,
                            file_name: str, output_filename: str = "transcription.csv") -> str:
    """
    Save transcription results for a folder.
    
    Args:
        segments (List[Dict[str, Any]]): Transcription segments
        folder_path (str): Path to the folder
        file_name (str): Source file name
        output_filename (str): Output filename
        
    Returns:
        str: Path to saved file
    """
    saver = TranscriptionSaver(folder_path)
    return saver.save_to_csv(segments, file_name, output_filename)

def create_transcription_dataframe(segments: List[Dict[str, Any]], 
                                 file_name: str) -> pd.DataFrame:
    """
    Create a pandas DataFrame from transcription segments.
    
    Args:
        segments (List[Dict[str, Any]]): Transcription segments
        file_name (str): Source file name
        
    Returns:
        pd.DataFrame: DataFrame with transcription data
    """
    saver = TranscriptionSaver("")  # Dummy output dir
    
    data = []
    for segment in segments:
        segment_data = saver.prepare_segment_data(segment, file_name)
        data.append(segment_data)
    
    return pd.DataFrame(data)