#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Main script for the WhisperX ASR transcription pipeline.

This script processes audio files in the Mandela at 90 archive, generating
accurate timestamped transcriptions using WhisperX with GPU acceleration,
speaker diarization, and comprehensive post-processing.

Usage:
    python create_transcription_main.py [OPTIONS]
    
Example:
    python create_transcription_main.py --input_dir "data/nmf_recordings/Mandela at 90" 
                                       --model_size "large-v2" --use_gpu True --language "en"
"""

import os
import sys
import logging
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

# Add current directory and src/ root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import our modules
from config import *
from audio_utils import preprocess_audio, validate_audio_file, split_audio_chunks, get_audio_info
from transcriber import WhisperXTranscriber, create_transcriber
from postprocess import TranscriptionPostProcessor, create_post_processor
from save_utils import TranscriptionSaver, save_folder_transcription

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class TranscriptionPipeline:
    """
    Main orchestrator for the WhisperX transcription pipeline.
    """
    
    def __init__(self, args):
        """Initialize the pipeline with command-line arguments."""
        self.args = args
        
        # Setup logging
        self.setup_logging()
        
        # Create necessary directories
        create_directories()
        
        # Initialize components
        self.transcriber = None
        self.post_processor = None
        self.saver = None
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'skipped_files': 0,
            'total_segments': 0,
            'total_duration': 0.0,
            'start_time': datetime.now()
        }
        
        self.failed_files = []
        self.processed_folders = {}
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("TranscriptionPipeline initialized")
        
    def setup_logging(self):
        """Setup logging configuration."""
        # If --quiet provided, force WARNING level regardless of --log_level
        if getattr(self.args, 'quiet', False):
            log_level = logging.WARNING
        else:
            log_level = getattr(logging, self.args.log_level.upper(), logging.INFO)
        
        # Create logs directory
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format=LOG_FORMAT,
            handlers=[
                logging.FileHandler(LOGS_DIR / LOG_FILE),
                logging.StreamHandler(sys.stdout)
            ]
        )

        if getattr(self.args, 'quiet', False):
            # Silence noisy module loggers further
            for name in ['__main__', 'transcriber', 'audio_utils', 'postprocess']:
                logging.getLogger(name).setLevel(logging.WARNING)
            # Optionally suppress library info
            logging.getLogger('speechbrain').setLevel(logging.ERROR)
            logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
        
        # Suppress external library logs
        logging.getLogger('whisperx').setLevel(logging.WARNING)
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)
        
    def initialize_components(self):
        """Initialize transcriber, post-processor, and saver components."""
        try:
            self.logger.info("Initializing pipeline components...")
            
            # Initialize transcriber
            self.logger.info(f"Loading WhisperX model: {self.args.model_size}")
            self.transcriber = create_transcriber(
                model_size=self.args.model_size,
                language=self.args.language,
                device=get_device() if self.args.use_gpu else "cpu",
                enable_diarization=self.args.enable_diarization,
                vad_method=self.args.vad_method,
                vad_device=self.args.vad_device,
                vad_onset=self.args.vad_onset,
                vad_offset=self.args.vad_offset,
                vad_chunk_size=self.args.vad_chunk_size
            )
            
            # Initialize post-processor
            self.post_processor = create_post_processor(
                target_language=self.args.language,
                min_confidence=self.args.min_confidence
            )
            
            # Initialize saver
            self.saver = TranscriptionSaver(
                output_dir=self.args.output_dir,
                encoding='utf-8'
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    def find_audio_files(self, directory: str) -> List[str]:
        """
        Find all supported audio files in a directory.
        
        Args:
            directory (str): Directory to search
            
        Returns:
            List[str]: List of audio file paths
        """
        audio_files = []
        
        for ext in SUPPORTED_FORMATS:
            audio_files.extend(Path(directory).glob(f"*{ext}"))
            audio_files.extend(Path(directory).glob(f"*{ext.upper()}"))
        
        # Convert to strings and sort
        audio_files = sorted([str(f) for f in audio_files])
        
        self.logger.info(f"Found {len(audio_files)} audio files in {directory}")
        return audio_files
    
    def get_folder_structure(self, input_dir: str) -> Dict[str, List[str]]:
        """Recursively find all audio files, but group them by their top-level folder directly under input_dir.
        Each top-level folder gets one consolidated transcription.csv, containing segments for every audio file
        anywhere in its subtree. This matches the requirement: one transcription.csv per subfolder below the archive root.
        Returns mapping: absolute top-level folder path -> list of absolute audio file paths in its entire subtree.
        """
        input_path = Path(input_dir)
        grouped: Dict[str, List[str]] = {}
        if not input_path.exists():
            self.logger.error(f"Input directory does not exist: {input_dir}")
            return grouped

        supported_exts = set(SUPPORTED_FORMATS) | {ext.upper() for ext in SUPPORTED_FORMATS}
        for dirpath, _, filenames in os.walk(input_path):
            for fname in filenames:
                fpath = Path(dirpath) / fname
                if fpath.suffix in supported_exts:
                    rel_parts = fpath.relative_to(input_path).parts
                    if len(rel_parts) == 0:
                        continue
                    top_level = input_path / rel_parts[0]
                    grouped.setdefault(str(top_level), []).append(str(fpath))

        # Sort file lists deterministically
        for k in list(grouped.keys()):
            grouped[k] = sorted(grouped[k])
        self.logger.info(f"Top-level folders with audio: {len(grouped)} (recursive grouping applied)")
        return grouped
    
    def process_audio_file(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Process a single audio file through the complete pipeline.
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            Optional[List[Dict[str, Any]]]: Processed transcription segments or None
        """
        try:
            self.logger.info(f"Processing audio file: {os.path.basename(file_path)}")
            
            # Validate audio file
            if not validate_audio_file(file_path):
                self.logger.warning(f"Skipping invalid audio file: {file_path}")
                self.stats['skipped_files'] += 1
                return None
            
            # Get audio info
            audio_info = get_audio_info(file_path)
            self.logger.info(f"Audio info: {audio_info.get('duration', 0):.2f}s, "
                           f"{audio_info.get('sample_rate', 0)}Hz, "
                           f"{audio_info.get('file_size_mb', 0):.1f}MB")
            
            # Preprocess audio
            audio, sample_rate = preprocess_audio(
                file_path,
                target_sr=TARGET_SAMPLE_RATE,
                enable_noise_reduction=True,
                noise_strength=NOISE_REDUCTION_STRENGTH
            )
            
            all_segments = []
            
            # Check if audio needs chunking
            if len(audio) / sample_rate > CHUNK_DURATION:
                self.logger.info("Audio is long, processing in chunks...")
                
                # Split into chunks
                chunks = split_audio_chunks(
                    audio, 
                    sample_rate, 
                    chunk_duration=CHUNK_DURATION,
                    overlap=5.0
                )
                
                # Process each chunk
                for i, (chunk_audio, start_time, end_time) in enumerate(chunks):
                    self.logger.info(f"Processing chunk {i+1}/{len(chunks)}: "
                                   f"{start_time:.1f}s - {end_time:.1f}s")
                    
                    # Transcribe chunk
                    chunk_result = self.transcriber.process_audio_file(chunk_audio, sample_rate, quiet=getattr(self.args,'quiet',False))
                    chunk_segments = chunk_result.get('segments', [])
                    
                    # Adjust timestamps to global timeline
                    for segment in chunk_segments:
                        segment['start'] += start_time
                        segment['end'] += start_time
                    
                    all_segments.extend(chunk_segments)
                    
            else:
                # Process entire file at once
                self.logger.info("Processing entire audio file...")
                result = self.transcriber.process_audio_file(audio, sample_rate, quiet=getattr(self.args,'quiet',False))
                all_segments = result.get('segments', [])
            
            # Post-process segments (speech-level)
            processed_segments = self.post_processor.process_segments(all_segments)

            # Optional aggregation into fixed windows (e.g., every 2 minutes) to reduce row count
            window_size = getattr(self.args, 'fixed_window_duration', None)
            if window_size and window_size > 0:
                processed_segments = self.aggregate_fixed_windows(processed_segments, window_size)
            
            # Update statistics
            self.stats['processed_files'] += 1
            self.stats['total_segments'] += len(processed_segments)
            self.stats['total_duration'] += audio_info.get('duration', 0)
            
            self.logger.info(f"Successfully processed {file_path}: {len(processed_segments)} segments")
            return processed_segments
            
        except Exception as e:
            self.logger.error(f"Failed to process {file_path}: {str(e)}")
            self.stats['failed_files'] += 1
            self.failed_files.append((file_path, str(e)))
            return None
    
    def process_folder(self, folder_path: str, audio_files: List[str]) -> bool:
        """
        Process all audio files in a folder and save consolidated results.
        
        Args:
            folder_path (str): Path to the folder
            audio_files (List[str]): List of audio files in the folder
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            folder_name = os.path.basename(folder_path)
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Processing folder: {folder_name}")
            self.logger.info(f"Audio files: {len(audio_files)}")
            self.logger.info(f"{'='*50}")
            
            all_folder_segments = []
            
            # Process each audio file
            per_file_outputs = getattr(self.args, 'per_file_outputs', False)

            input_root = Path(self.args.input_dir)
            for i, audio_file in enumerate(audio_files, 1):
                # Log only the recording name (relative path), without progress counters
                rel_display = Path(audio_file).relative_to(input_root) if Path(audio_file).is_file() else Path(audio_file).name
                self.logger.info(f"Processing recording: {rel_display}")
                
                segments = self.process_audio_file(audio_file)
                
                if segments:
                    # Add file name to each segment
                    rel_path = str(Path(audio_file).relative_to(input_root)) if Path(audio_file).is_file() else os.path.basename(audio_file)
                    for segment in segments:
                        # Preserve both relative path (for provenance) and original basename
                        segment['source_file'] = os.path.basename(audio_file)
                        segment['source_relpath'] = rel_path
                    
                    all_folder_segments.extend(segments)
                    self.logger.info(f"Added {len(segments)} segments from {os.path.basename(audio_file)}")

                    # Optional per-file outputs (CSV + JSON)
                    if per_file_outputs:
                        from save_utils import TranscriptionSaver as _FileSaver
                        # Group all per-file outputs under the TOP-LEVEL directory (first path component after input root)
                        rel_parts = Path(audio_file).relative_to(input_root).parts
                        top_level = rel_parts[0] if rel_parts else Path(folder_path).name
                        folder_out_dir = Path(self.args.output_dir) / top_level
                        folder_out_dir.mkdir(parents=True, exist_ok=True)
                        # Flatten full relative path (minus extension) to avoid nested directories in output
                        flattened_name = "_".join([*rel_parts[:-1], Path(rel_parts[-1]).stem]) if rel_parts else Path(audio_file).stem
                        file_saver = _FileSaver(str(folder_out_dir))
                        try:
                            csv_name = f"{flattened_name}.csv"
                            json_name = f"{flattened_name}.json"
                            csv_path = file_saver.save_to_csv(segments, flattened_name, csv_name)
                            json_path = file_saver.save_to_json(segments, flattened_name, json_name)
                            self.logger.info(f"Per-file outputs saved (flattened): {csv_path}, {json_path}")
                        except Exception as pf_err:
                            self.logger.warning(f"Failed to save per-file outputs for {audio_file}: {pf_err}")
                else:
                    self.logger.warning(f"No segments extracted from {audio_file}")
            
            # Save consolidated results for the folder
            if all_folder_segments:
                # Save consolidated file into global output directory (one per folder)
                output_path = self.saver.save_to_csv(
                    all_folder_segments,
                    folder_name,
                    "transcription.csv"
                )

                self.processed_folders[folder_name] = all_folder_segments

                self.logger.info(f"Saved {len(all_folder_segments)} total segments for {folder_name}")
                self.logger.info(f"Output saved to: {output_path}")

                return True
            else:
                self.logger.warning(f"No valid segments found for folder: {folder_name}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to process folder {folder_path}: {str(e)}")
            return False

    def aggregate_fixed_windows(self, segments: List[Dict[str, Any]], window_size: int) -> List[Dict[str, Any]]:
        """Aggregate fine-grained speech segments into fixed-size windows.

        Args:
            segments: List of processed speech-level segments (with 'start','end','text').
            window_size: Window size in seconds (e.g., 120 for 2 minutes).

        Returns:
            List of aggregated segments where each covers a fixed time window.
        """
        if not segments:
            return []

        # Ensure segments sorted
        segments = sorted(segments, key=lambda s: s.get('start', 0.0))
        total_duration = max(s.get('end', 0.0) for s in segments)
        aggregated = []
        num_windows = int((total_duration + window_size - 1) // window_size)

        for w in range(num_windows):
            win_start = w * window_size
            win_end = min((w + 1) * window_size, total_duration)

            # Gather segments that overlap the window (any overlap)
            bucket = [s for s in segments if s.get('end', 0) > win_start and s.get('start', 0) < win_end]
            if not bucket:
                continue  # skip empty window

            # Concatenate texts in chronological order
            texts = [s.get('text', '').strip() for s in bucket if s.get('text')]
            combined_text = ' '.join(t for t in texts if t)
            if not combined_text:
                continue

            # Confidence: average of available boosted/computed/confidence fields
            conf_vals = []
            for s in bucket:
                for key in ('boosted_confidence','computed_confidence','confidence'):
                    if key in s and s[key] is not None:
                        try:
                            conf_vals.append(float(s[key]))
                            break
                        except Exception:
                            continue
            avg_conf = sum(conf_vals)/len(conf_vals) if conf_vals else 0.0

            aggregated.append({
                'start': win_start,
                'end': win_end,
                'text': combined_text,
                'confidence': avg_conf,
                'aggregated': True,
                'window_index': w,
                'window_size': window_size
            })

        self.logger.info(f"Aggregated {len(segments)} speech segments into {len(aggregated)} fixed {window_size}s windows")
        return aggregated
    
    def run_pipeline(self):
        """Run the complete transcription pipeline."""
        try:
            start_time = time.time()
            self.logger.info("Starting WhisperX ASR Transcription Pipeline")
            self.logger.info(f"Input directory: {self.args.input_dir}")
            self.logger.info(f"Model: {self.args.model_size}, Language: {self.args.language}")
            self.logger.info(f"GPU enabled: {self.args.use_gpu}, Device: {get_device()}")
            
            # Initialize components
            self.initialize_components()
            
            # Get folder structure
            folder_structure = self.get_folder_structure(self.args.input_dir)
            
            if not folder_structure:
                self.logger.error("No folders with audio files found!")
                return False
            
            # Update total files count
            self.stats['total_files'] = sum(len(files) for files in folder_structure.values())
            self.logger.info(f"Total audio files to process: {self.stats['total_files']}")
            
            # Process each folder
            successful_folders = 0
            
            for folder_path, audio_files in folder_structure.items():
                if self.process_folder(folder_path, audio_files):
                    successful_folders += 1
            
            # Create summary report
            if self.processed_folders:
                summary_path = self.saver.create_summary_report(
                    self.processed_folders,
                    "processing_summary.json"
                )
                self.logger.info(f"Summary report saved: {summary_path}")
            
            # Final statistics
            end_time = time.time()
            processing_time = end_time - start_time
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info("PIPELINE COMPLETED")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Processing time: {processing_time/60:.1f} minutes")
            self.logger.info(f"Folders processed: {successful_folders}/{len(folder_structure)}")
            self.logger.info(f"Files processed: {self.stats['processed_files']}/{self.stats['total_files']}")
            self.logger.info(f"Files failed: {self.stats['failed_files']}")
            self.logger.info(f"Files skipped: {self.stats['skipped_files']}")
            self.logger.info(f"Total segments: {self.stats['total_segments']}")
            self.logger.info(f"Total duration: {self.stats['total_duration']/60:.1f} minutes")
            
            if self.failed_files:
                self.logger.warning(f"\nFailed files ({len(self.failed_files)}):")
                for file_path, error in self.failed_files:
                    self.logger.warning(f"  {file_path}: {error}")
            
            return successful_folders > 0
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            return False
        finally:
            # Cleanup
            if self.transcriber:
                self.transcriber.cleanup()

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="WhisperX ASR Transcription Pipeline for Mandela Archives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python create_transcription_main.py
  
  # Custom model and directory
  python create_transcription_main.py --model_size large-v3 --input_dir "path/to/audio"
  
  # CPU-only processing
  python create_transcription_main.py --use_gpu false
  
  # High-quality processing
  python create_transcription_main.py --model_size large-v3 --min_confidence 0.8
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        default=str(INPUT_DIR),
        help=f'Input directory containing audio folders (default: {INPUT_DIR})'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(OUTPUT_DIR),
        help=f'Output directory for transcription results (default: {OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--model_size',
        type=str,
        default=MODEL_SIZE,
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
        help=f'WhisperX model size (default: {MODEL_SIZE})'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        default=LANGUAGE,
        help=f'Primary language for transcription (default: {LANGUAGE})'
    )
    
    parser.add_argument(
        '--use_gpu',
        type=lambda x: x.lower() in ['true', '1', 'yes', 'on'],
        default=USE_GPU,
        help=f'Enable GPU acceleration (default: {USE_GPU})'
    )
    
    parser.add_argument(
        '--enable_diarization',
        type=lambda x: x.lower() in ['true', '1', 'yes', 'on'],
        default=ENABLE_DIARIZATION,
        help=f'Enable speaker diarization (default: {ENABLE_DIARIZATION})'
    )

    parser.add_argument(
        '--vad_method',
        type=str,
        default=VAD_METHOD,
        choices=['pyannote', 'silero'],
        help=f'Voice activity detection backend (default: {VAD_METHOD})'
    )

    parser.add_argument(
        '--vad_device',
        type=str,
        default=VAD_DEVICE,
        help=f'Device for VAD processing (default: {VAD_DEVICE})'
    )

    parser.add_argument(
        '--vad_onset',
        type=float,
        default=VAD_ONSET,
        help=f'VAD onset threshold (default: {VAD_ONSET})'
    )

    parser.add_argument(
        '--vad_offset',
        type=float,
        default=VAD_OFFSET,
        help=f'VAD offset threshold (default: {VAD_OFFSET})'
    )

    parser.add_argument(
        '--vad_chunk_size',
        type=float,
        default=VAD_CHUNK_SIZE,
        help=f'VAD chunk size in seconds (default: {VAD_CHUNK_SIZE})'
    )
    
    parser.add_argument(
        '--min_confidence',
        type=float,
        default=MIN_CONFIDENCE_THRESHOLD,
        help=f'Minimum confidence threshold (default: {MIN_CONFIDENCE_THRESHOLD})'
    )
    
    parser.add_argument(
        '--chunk_duration',
        type=int,
        default=CHUNK_DURATION,
        help=f'Audio chunk duration in seconds (default: {CHUNK_DURATION})'
    )

    parser.add_argument(
        '--fixed_window_duration',
        type=int,
        default=0,
        help='If >0, aggregate output into fixed-size windows (seconds), e.g., 120 for 2-minute blocks.'
    )

    parser.add_argument(
        '--per_file_outputs',
        type=lambda x: x.lower() in ['true','1','yes','on'],
        default=False,
        help='If true, also write a CSV and JSON per original audio file under results/<Folder Name>/'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default=LOG_LEVEL,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help=f'Logging level (default: {LOG_LEVEL})'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress INFO logs (forces WARNING level for pipeline and related modules)'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the transcription pipeline."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Create and run pipeline
        pipeline = TranscriptionPipeline(args)
        success = pipeline.run_pipeline()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()