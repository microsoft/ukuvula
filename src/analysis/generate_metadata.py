#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Mandela Foundation Audio/Video Metadata Generator

This script generates metadata for audiovisual files with two modes:
- Simple mode (default): Basic file information using ffprobe only
- Full mode (--all): Comprehensive analysis including speaker diarization

Generated Columns (Simple mode):
- file_name: Full relative path starting with "nmf_recordings/" (e.g., "nmf_recordings/Bloody Miracle/Interview.wav") 
- file_size_mb: File size in megabytes
- duration: Duration in seconds
- unique_speakers: "Not analyzed (basic version)"
- data_shape: Audio data dimensions (samples or channels×samples)
- sample_rate: Audio sample rate in Hz
- channels: Number of audio channels

Additional Columns (Full mode):
- unique_speakers: Number of unique speakers (estimated via diarization)
- rms_energy: Root mean square energy (audio quality metric)
- zero_crossing_rate: Zero crossing rate (speech/music indicator) 
- spectral_centroid: Spectral centroid (brightness measure)

Requirements:
- ffprobe (from ffmpeg) - required for both modes
- pyannote.audio, librosa, dotenv - only required for full mode

Usage:
    python generate_metadata.py           # Simple mode (default)
    python generate_metadata.py --all     # Full analysis with speaker diarization
    python generate_metadata.py --simple  # Explicitly use simple mode
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime
import subprocess
import json
import argparse

# Ensure src/ is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Suppress warnings for cleaner output  
warnings.filterwarnings('ignore')


class SimpleAudioVideoMetadataExtractor:
    """Simple metadata extractor using only ffprobe."""
    
    def __init__(self, data_dir="data/nmf_recordings"):
        self.data_dir = data_dir
        self.supported_audio_formats = ['.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg']
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        self.results = []
    
    def get_file_size_mb(self, file_path):
        """Get file size in megabytes."""
        try:
            size_bytes = os.path.getsize(file_path)
            return round(size_bytes / (1024 * 1024), 2)
        except:
            return None
    
    def get_file_type(self, file_path):
        """Determine if file is audio or video based on extension."""
        ext = Path(file_path).suffix.lower()
        if ext in self.supported_audio_formats:
            return 'audio'
        elif ext in self.supported_video_formats:
            return 'video'
        else:
            return 'unknown'
    
    def get_ffprobe_metadata(self, file_path):
        """Extract metadata using ffprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', '-show_streams', file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return None
                
            data = json.loads(result.stdout)
            
            metadata = {}
            
            # Find video and audio streams
            video_stream = None
            audio_stream = None
            
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video' and video_stream is None:
                    video_stream = stream
                elif stream.get('codec_type') == 'audio' and audio_stream is None:
                    audio_stream = stream
            
            # Extract general format info
            format_info = data.get('format', {})
            duration = format_info.get('duration')
            if duration:
                metadata['duration'] = round(float(duration), 2)
            
            # Extract audio information
            if audio_stream:
                metadata['sample_rate'] = audio_stream.get('sample_rate')
                metadata['channels'] = audio_stream.get('channels', 0)
                
                # For audio files, create a simple data shape
                if not video_stream and metadata.get('sample_rate') and metadata.get('duration'):
                    total_samples = int(float(metadata['sample_rate']) * metadata['duration'])
                    if metadata['channels'] > 1:
                        metadata['data_shape'] = f"{metadata['channels']}x{total_samples}"
                    else:
                        metadata['data_shape'] = str(total_samples)
                else:
                    # For video files, use duration and sample rate
                    metadata['data_shape'] = None
            else:
                metadata.update({
                    'sample_rate': None,
                    'channels': None,
                    'data_shape': None
                })
            
            return metadata
            
        except Exception as e:
            print(f"⚠️ Error extracting metadata with ffprobe: {e}")
            return None
    
    def process_file(self, file_path):
        """Process a single audio/video file and extract metadata.

        The file_name recorded is now the full relative path from self.data_dir
        starting with "nmf_recordings/" (e.g., "nmf_recordings/Bloody Miracle/Some Interview.wav")
        This preserves the complete provenance and folder context required for 
        downstream analysis and grouping.
        """
        print(f"Processing: {file_path}")

        # Full relative path from the data directory
        rel_path = os.path.relpath(file_path, self.data_dir)
        # Normalize path separators to forward slashes for CSV portability
        clean_name = rel_path.replace(os.sep, "/")
        
        # Ensure path starts with "nmf_recordings/" prefix
        if not clean_name.startswith("nmf_recordings/"):
            clean_name = f"nmf_recordings/{clean_name}"

        file_type = self.get_file_type(file_path)
        file_size_mb = self.get_file_size_mb(file_path)

        metadata = {
            'file_name': clean_name,
            'file_size_mb': file_size_mb,
        }
        
        # Extract metadata using ffprobe
        ffprobe_metadata = self.get_ffprobe_metadata(file_path)
        if ffprobe_metadata:
            metadata.update(ffprobe_metadata)
            # Note: No speaker analysis in this simple version
            metadata['unique_speakers'] = "Not analyzed (basic version)"
        else:
            metadata.update({
                'duration': None,
                'sample_rate': None,
                'channels': None,
                'data_shape': None,
                'unique_speakers': None,
            })
        
        return metadata
    
    def scan_directory(self):
        """Scan the data directory for audio and video files."""
        files_found = []
        
        for root, dirs, files in os.walk(self.data_dir):
            # Sort directories numerically (e.g., Folder 1, Folder 2, ..., Folder 10, Folder 11)
            dirs.sort(key=lambda x: int(x.split()[-1]) if 'Folder' in x and x.split()[-1].isdigit() else float('inf'))
            
            # Sort files alphabetically within each directory
            files.sort()
            
            for file in files:
                file_path = os.path.join(root, file)
                ext = Path(file_path).suffix.lower()
                
                if ext in self.supported_audio_formats + self.supported_video_formats:
                    files_found.append(file_path)
        
        return files_found
    
    def generate_metadata(self, output_file="./results/audiovisual_metadata_simple.csv"):
        """Generate metadata for all files and save to CSV."""
        print("🎬 Starting Mandela Foundation Audiovisual Metadata Generation (Simple Version)")
        print("=" * 70)
        
        # Scan for files
        files = self.scan_directory()
        print(f"📁 Found {len(files)} audiovisual files")
        
        if not files:
            print("❌ No audiovisual files found in the data directory")
            return
        
        # Process each file
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] ", end="")
            metadata = self.process_file(file_path)
            self.results.append(metadata)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(self.results)
        
        # Reorder columns for better readability - keep only essential columns
        column_order = [
            'file_name', 'file_size_mb', 'duration', 'unique_speakers', 
            'data_shape', 'sample_rate', 'channels'
        ]
        
        # Only include columns that exist
        existing_columns = [col for col in column_order if col in df.columns]
        df = df.reindex(columns=existing_columns)
        
        # Ensure results directory exists
        import os
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Create full output path
        if not output_file.startswith("./results/"):
            output_file = os.path.join(results_dir, os.path.basename(output_file))
        
        # Save to CSV (append if file exists)
        if os.path.exists(output_file):
            print(f"📝 Appending to existing file: {output_file}")
            # Read existing file to get header consistency
            try:
                existing_df = pd.read_csv(output_file)
                # Append new data
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_csv(output_file, index=False)
                print(f"✅ Successfully appended {len(df)} new records to existing file")
            except Exception as e:
                print(f"⚠️  Warning: Could not append to existing file, creating backup and overwriting: {e}")
                backup_file = output_file.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                if os.path.exists(output_file):
                    os.rename(output_file, backup_file)
                df.to_csv(output_file, index=False)
        else:
            print(f"📝 Creating new file: {output_file}")
            df.to_csv(output_file, index=False)
        
        print(f"\n\n✅ Metadata generation complete!")
        print(f"📊 Results saved to: {output_file}")
        print(f"📈 Total files processed: {len(self.results)}")
        
        # Print summary statistics
        total_duration = pd.to_numeric(df['duration'], errors='coerce').sum()
        total_size = df['file_size_mb'].sum()
        
        if not pd.isna(total_duration):
            print(f"⏱️ Total duration: {total_duration/60:.1f} minutes")
        print(f"💾 Total size: {total_size:.1f} MB")
        
        print(f"\n📝 Note: This is the basic version without speaker diarization.")
        print(f"   For speaker analysis, use --all flag after installing pyannote.audio")
        
        return df


class FullAudioVideoMetadataExtractor:
    """Full metadata extractor with speaker diarization."""
    
    def __init__(self, data_dir="data/nmf_recordings"):
        self.data_dir = data_dir
        self.supported_audio_formats = ['.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg']
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        self.results = []
        self.pyannote_pipeline = None
        
        # Load advanced libraries
        self.load_advanced_libraries()
        
        # Load environment variables for Hugging Face tokens
        if self.load_dotenv:
            self.load_dotenv()
        
        # Initialize pyannote pipeline
        self._setup_pyannote_pipeline()
    
    def load_advanced_libraries(self):
        """Load libraries needed for full analysis mode."""
        try:
            import librosa
            import soundfile as sf
            from dotenv import load_dotenv
            
            self.librosa = librosa
            self.sf = sf
            self.load_dotenv = load_dotenv
            return True
        except ImportError as e:
            print(f"❌ Advanced libraries not available: {e}")
            print("💡 Install with: conda activate nmf && pip install librosa soundfile python-dotenv openai-whisper")
            raise ImportError("Required libraries not available for full mode")
    
    def _setup_pyannote_pipeline(self):
        """Initialize pyannote.audio pipeline with fallback models."""
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            print("⚠️ No HUGGINGFACE_TOKEN found in environment variables")
            print("📝 Speaker diarization will use basic estimation method")
            return
        
        models_to_try = [
            "pyannote/speaker-diarization-3.1",
            "pyannote/speaker-diarization-community-1"
        ]
        
        for model_name in models_to_try:
            try:
                print(f"🔄 Loading pyannote model: {model_name}")
                from pyannote.audio import Pipeline
                
                # Try newer parameter name first (pyannote 4.x)
                try:
                    self.pyannote_pipeline = Pipeline.from_pretrained(model_name, token=hf_token)
                except TypeError:
                    # Fallback to older parameter name (pyannote 3.x)
                    self.pyannote_pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
                
                print(f"✅ Successfully loaded: {model_name}")
                return
                
            except Exception as e:
                print(f"⚠️ Failed to load {model_name}: {str(e)[:100]}...")
                continue
        
        print("❌ No pyannote models available - using basic speaker estimation")
    
    def get_file_size_mb(self, file_path):
        """Get file size in megabytes."""
        try:
            size_bytes = os.path.getsize(file_path)
            return round(size_bytes / (1024 * 1024), 2)
        except:
            return None
    
    def get_audio_metadata(self, file_path):
        """Extract comprehensive audio metadata."""
        try:
            # Use soundfile for basic info
            info = self.sf.info(file_path)
            
            # Load with librosa for detailed analysis
            y, sr = self.librosa.load(file_path, sr=None, mono=False)
            
            metadata = {
                'duration': info.duration,
                'sample_rate': info.samplerate, 
                'channels': info.channels,
            }
            
            # Calculate data shape
            if len(y.shape) > 1:
                metadata['data_shape'] = f"{y.shape[0]}x{y.shape[1]}"
            else:
                metadata['data_shape'] = str(len(y))
            
            # Convert to mono for feature extraction
            if len(y.shape) > 1:
                y_mono = self.librosa.to_mono(y)
            else:
                y_mono = y
            
            # Calculate audio features
            rms_energy = float(np.sqrt(np.mean(y_mono**2)))
            zero_crossing_rate = float(np.mean(self.librosa.feature.zero_crossing_rate(y_mono)))
            spectral_centroid = float(np.mean(self.librosa.feature.spectral_centroid(y=y_mono, sr=sr)))
            
            metadata.update({
                'rms_energy': rms_energy,
                'zero_crossing_rate': zero_crossing_rate,
                'spectral_centroid': spectral_centroid
            })
            
            return metadata
            
        except Exception as e:
            print(f"⚠️ Error extracting audio metadata: {e}")
            return None
    
    def estimate_speakers_basic(self, audio_path):
        """Enhanced speaker estimation using multiple audio features."""
        try:
            # Load audio
            y, sr = self.librosa.load(audio_path, sr=22050)
            
            # Multiple feature approaches
            features = []
            
            # 1. MFCC variance approach
            mfccs = self.librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_var = np.var(mfccs, axis=1).mean()
            features.append(mfcc_var)
            
            # 2. Spectral centroid variation
            spectral_centroid = self.librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_var = np.var(spectral_centroid)
            features.append(spec_var)
            
            # 3. Zero crossing rate variation
            zcr = self.librosa.feature.zero_crossing_rate(y)
            zcr_var = np.var(zcr)
            features.append(zcr_var)
            
            # 4. RMS energy variation
            rms = self.librosa.feature.rms(y=y)
            rms_var = np.var(rms)
            features.append(rms_var)
            
            # Combined variance for speaker estimation
            combined_variance = np.mean(features)
            
            # Enhanced heuristics
            if combined_variance < 0.05:
                return 1
            elif combined_variance < 0.15:
                return 2
            elif combined_variance < 0.35:
                return 3
            else:
                return 4
                
        except Exception as e:
            print(f"Error in basic speaker estimation: {e}")
            return None
    
    def estimate_speakers_pyannote(self, audio_path):
        """Estimate speakers using pyannote.audio diarization."""
        try:
            if self.pyannote_pipeline is None:
                return self.estimate_speakers_basic(audio_path)
            
            # Apply diarization - CORRECTED API for pyannote 4.x
            diarization_output = self.pyannote_pipeline(audio_path)
            diarization = diarization_output.speaker_diarization
            
            # Count unique speakers
            unique_speakers = set()
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                unique_speakers.add(speaker)
            
            return len(unique_speakers)
            
        except Exception as e:
            print(f"⚠️ pyannote diarization failed: {str(e)[:100]}...")
            return self.estimate_speakers_basic(audio_path)
    
    def process_file(self, file_path):
        """Process a single audio/video file and extract comprehensive metadata.

        The file_name recorded is now the full relative path from self.data_dir
        starting with "nmf_recordings/" (e.g., "nmf_recordings/Bloody Miracle/Some Interview.wav")
        This preserves the complete provenance and folder context required for 
        downstream analysis and grouping.
        """
        print(f"Processing: {file_path}")
        
        # Full relative path from the data directory (consistent with SimpleAudioVideoMetadataExtractor)
        rel_path = os.path.relpath(file_path, self.data_dir)
        # Normalize path separators to forward slashes for CSV portability
        clean_name = rel_path.replace(os.sep, "/")
        
        # Ensure path starts with "nmf_recordings/" prefix
        if not clean_name.startswith("nmf_recordings/"):
            clean_name = f"nmf_recordings/{clean_name}"
        
        # Initialize metadata
        metadata = {
            'file_name': clean_name,
            'file_size_mb': self.get_file_size_mb(file_path),
        }
        
        # Audio analysis
        ext = Path(file_path).suffix.lower()
        if ext in self.supported_audio_formats:
            # Get comprehensive audio metadata
            audio_metadata = self.get_audio_metadata(file_path)
            if audio_metadata:
                metadata.update(audio_metadata)
                
                # Speaker estimation (full mode)
                speaker_count = self.estimate_speakers_pyannote(file_path)
                metadata['unique_speakers'] = speaker_count
            else:
                metadata.update({
                    'duration': None,
                    'sample_rate': None,
                    'channels': None,
                    'data_shape': None,
                    'rms_energy': None,
                    'zero_crossing_rate': None,
                    'spectral_centroid': None,
                    'unique_speakers': None
                })
        else:
            # For video files, basic info only
            metadata.update({
                'duration': None,
                'sample_rate': None,
                'channels': None,
                'data_shape': None,
                'rms_energy': None,
                'zero_crossing_rate': None,
                'spectral_centroid': None,
                'unique_speakers': None
            })
        
        return metadata
    
    def scan_directory(self):
        """Scan the data directory for audio and video files."""
        files_found = []
        
        for root, dirs, files in os.walk(self.data_dir):
            # Sort directories numerically (e.g., Folder 1, Folder 2, ..., Folder 10, Folder 11)
            dirs.sort(key=lambda x: int(x.split()[-1]) if 'Folder' in x and x.split()[-1].isdigit() else float('inf'))
            
            # Sort files alphabetically within each directory
            files.sort()
            
            for file in files:
                file_path = os.path.join(root, file)
                ext = Path(file_path).suffix.lower()
                
                if ext in self.supported_audio_formats + self.supported_video_formats:
                    files_found.append(file_path)
        
        return files_found
    
    def generate_metadata(self, output_file="./results/audiovisual_metadata_full.csv"):
        """Generate comprehensive metadata for all files and save to CSV."""
        print("🎬 Starting Mandela Foundation Audiovisual Metadata Generation (Full Analysis)")
        print("=" * 80)
        
        # Scan for files
        files = self.scan_directory()
        print(f"📁 Found {len(files)} audiovisual files")
        
        if not files:
            print("❌ No audiovisual files found in the data directory")
            return None
        
        # Process each file
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] ", end="")
            metadata = self.process_file(file_path)
            self.results.append(metadata)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(self.results)
        
        # Reorder columns for better readability
        column_order = [
            'file_name', 'file_size_mb', 'duration', 'unique_speakers', 
            'data_shape', 'sample_rate', 'channels', 'rms_energy',
            'zero_crossing_rate', 'spectral_centroid'
        ]
        
        # Only include columns that exist
        existing_columns = [col for col in column_order if col in df.columns]
        df = df.reindex(columns=existing_columns)
        
        # Ensure results directory exists
        import os
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Create full output path
        if not output_file.startswith("./results/"):
            output_file = os.path.join(results_dir, os.path.basename(output_file))
        
        # Save to CSV (append if file exists)
        if os.path.exists(output_file):
            print(f"📝 Appending to existing file: {output_file}")
            # Read existing file to get header consistency
            try:
                existing_df = pd.read_csv(output_file)
                # Append new data
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_csv(output_file, index=False)
                print(f"✅ Successfully appended {len(df)} new records to existing file")
            except Exception as e:
                print(f"⚠️  Warning: Could not append to existing file, creating backup and overwriting: {e}")
                backup_file = output_file.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                if os.path.exists(output_file):
                    os.rename(output_file, backup_file)
                df.to_csv(output_file, index=False)
        else:
            print(f"📝 Creating new file: {output_file}")
            df.to_csv(output_file, index=False)
        
        print(f"\n\n✅ Full metadata generation complete!")
        print(f"📊 Results saved to: {output_file}")
        print(f"📈 Total files processed: {len(self.results)}")
        
        # Print summary statistics
        total_duration = pd.to_numeric(df['duration'], errors='coerce').sum()
        total_size = df['file_size_mb'].sum()
        
        if not pd.isna(total_duration):
            print(f"⏱️ Total duration: {total_duration/60:.1f} minutes")
        print(f"💾 Total size: {total_size:.1f} MB")
        
        return df


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Mandela Foundation Audio/Video Metadata Generator",
        epilog="Use --all for comprehensive analysis including speaker diarization"
    )
    
    parser.add_argument(
        '--all', '--full', 
        action='store_true',
        help='Enable full analysis mode with speaker diarization (requires pyannote.audio)'
    )
    
    parser.add_argument(
        '--simple', 
        action='store_true',
        help='Explicitly use simple mode (ffprobe only) - this is the default'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output CSV filename (auto-generated if not specified)'
    )
    
    args = parser.parse_args()
    
    # Determine mode
    use_full_mode = args.all and not args.simple
    
    print(f"🎬 Mandela Foundation Metadata Generator")
    print(f"📋 Mode: {'Full Analysis' if use_full_mode else 'Simple Mode'}")
    print("=" * 50)
    
    # Generate output filename if not specified
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "full" if use_full_mode else "simple"
        # Ensure results directory exists
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        args.output = f"./results/audiovisual_metadata_{mode_suffix}.csv"
    
    try:
        if use_full_mode:
            # Use full analysis mode
            extractor = FullAudioVideoMetadataExtractor()
            df = extractor.generate_metadata(args.output)
        else:
            # Use simple mode (default)
            extractor = SimpleAudioVideoMetadataExtractor()
            df = extractor.generate_metadata(args.output)
        
        if df is not None:
            print(f"\n📋 Preview of generated metadata:")
            print(df.head().to_string())
            
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print(f"💡 Try using simple mode: python generate_metadata.py --simple")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())