#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Sophisticated Speaker Diarization Script for Mandela Foundation Audio Collection

This script uses state-of-the-art speaker diarization from pyannote.audio to accurately
estimate the number of unique speakers in each audio file. It updates the 
results/audiovisual_metadata_full.csv file with corrected unique_speakers values.

The script uses pyannote.audio's pre-trained speaker diarization models which:
- Detect voice activity (VAD)
- Extract speaker embeddings
- Cluster speakers using advanced algorithms
- Provide accurate speaker counts and timings

Requirements:
- pyannote.audio >= 3.1.0
- HUGGINGFACE_TOKEN environment variable (for model access)
- torch, torchaudio
- pandas, soundfile

Usage:
    python estimate_unique_speakers.py
    python estimate_unique_speakers.py --sample 10  # Test on first 10 files
    python estimate_unique_speakers.py --force      # Reprocess all files
"""

import os
import sys
import argparse
import pandas as pd
import warnings
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import torch

# Ensure src/ is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Suppress warnings
warnings.filterwarnings('ignore')

class SpeakerDiarizationEstimator:
    """Advanced speaker diarization using pyannote.audio."""
    
    def __init__(self, use_gpu=True):
        """Initialize the speaker diarization pipeline.
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        self.device = self._setup_device(use_gpu)
        self.pipeline = None
        self.data_dir = str(Path(__file__).resolve().parents[2] / "data" / "nmf_recordings")
        self._setup_pyannote_pipeline()
        
    def _setup_device(self, use_gpu):
        """Setup computation device (GPU or CPU)."""
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device("cpu")
            print("💻 Using CPU")
        return device
    
    def _setup_pyannote_pipeline(self):
        """Initialize pyannote.audio speaker diarization pipeline."""
        # Load environment variables
        load_dotenv()
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        
        if not hf_token:
            print("\n❌ ERROR: HUGGINGFACE_TOKEN not found in environment variables!")
            print("   Please set your Hugging Face token:")
            print("   1. Get token from: https://huggingface.co/settings/tokens")
            print("   2. Accept pyannote model terms: https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("   3. Set token: export HUGGINGFACE_TOKEN='your_token_here'")
            print("   4. Or add to .env file: HUGGINGFACE_TOKEN=your_token_here")
            sys.exit(1)
        
        # Try to load the best available model
        models_to_try = [
            "pyannote/speaker-diarization-3.1",  # Latest version
            "pyannote/speaker-diarization",      # Default version
        ]
        
        from pyannote.audio import Pipeline
        
        for model_name in models_to_try:
            try:
                print(f"\n🔄 Loading pyannote model: {model_name}")
                
                # Load pipeline with authentication token
                # The new API uses 'token' parameter instead of 'use_auth_token'
                try:
                    self.pipeline = Pipeline.from_pretrained(
                        model_name,
                        use_auth_token=hf_token
                    )
                except Exception as e1:
                    try:
                        # Try with newer 'token' parameter
                        self.pipeline = Pipeline.from_pretrained(
                            model_name,
                            token=hf_token
                        )
                    except Exception as e2:
                        raise Exception(f"Auth error: {str(e1)}, {str(e2)}")
                
                # Move pipeline to appropriate device
                if self.device.type == "cuda":
                    self.pipeline.to(self.device)
                
                print(f"✅ Successfully loaded: {model_name}")
                print(f"   Device: {self.device}")
                return
                
            except Exception as e:
                print(f"⚠️  Failed to load {model_name}: {str(e)[:150]}...")
                continue
        
        print("\n❌ ERROR: Could not load any pyannote.audio models!")
        print("   Please ensure you have:")
        print("   1. Accepted the model terms on Hugging Face:")
        print("      → Visit: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("      → Click 'Agree and access repository'")
        print("      → Also accept: https://huggingface.co/pyannote/segmentation-3.0")
        print("   2. Valid HUGGINGFACE_TOKEN with 'read' permissions")
        print("   3. pyannote.audio >= 3.1.0 installed")
        print("\n   After accepting terms, run: huggingface-cli login")
        sys.exit(1)
    
    def estimate_speakers(self, audio_path, verbose=False):
        """Estimate number of unique speakers using pyannote diarization.
        
        Args:
            audio_path (str): Path to audio file
            verbose (bool): Print detailed information
            
        Returns:
            int: Number of unique speakers detected, or None if failed
        """
        try:
            # Run diarization
            if verbose:
                print(f"   Processing: {Path(audio_path).name}")
            
            # Try normal processing first
            try:
                diarization = self.pipeline(audio_path)
            except RuntimeError as e:
                # Handle tensor size mismatch by loading and resampling audio properly
                if "Sizes of tensors must match" in str(e):
                    if verbose:
                        print(f"   Retrying with audio preprocessing...")
                    
                    import torchaudio
                    import tempfile
                    
                    # Load audio and ensure proper format
                    waveform, sample_rate = torchaudio.load(audio_path)
                    
                    # Convert to mono if needed
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    # Resample to 16kHz if needed
                    if sample_rate != 16000:
                        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                        waveform = resampler(waveform)
                        sample_rate = 16000
                    
                    # Save to temporary file and retry
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                        torchaudio.save(tmp_path, waveform, sample_rate)
                    
                    try:
                        diarization = self.pipeline(tmp_path)
                    finally:
                        # Clean up temporary file
                        import os
                        os.unlink(tmp_path)
                else:
                    raise
            
            # Extract unique speaker labels
            unique_speakers = set()
            total_speech_duration = 0.0
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                unique_speakers.add(speaker)
                total_speech_duration += turn.end - turn.start
            
            num_speakers = len(unique_speakers)
            
            if verbose:
                print(f"   ✓ Detected {num_speakers} unique speaker(s)")
                print(f"   Total speech: {total_speech_duration:.1f}s")
            
            return num_speakers
            
        except Exception as e:
            print(f"\n⚠️  Error processing {Path(audio_path).name}: {str(e)[:100]}")
            return None
    
    def process_audio_files(self, output_csv, force_reprocess=False, sample_size=None):
        """Process all audio files and create speakers_per_records.csv.
        
        Args:
            output_csv (str): Path to output CSV file
            force_reprocess (bool): Reprocess files even if output exists
            sample_size (int): Only process first N files (for testing)
        """
        # Check if output already exists
        if os.path.exists(output_csv) and not force_reprocess:
            print(f"\n📂 Loading existing results: {output_csv}")
            results_df = pd.read_csv(output_csv)
            processed_files = set(results_df['file_name'].tolist())
            print(f"   Already processed: {len(processed_files)} files")
        else:
            results_df = pd.DataFrame(columns=['file_name', 'unique_speakers'])
            processed_files = set()
        
        # Get list of all audio files
        print(f"\n📁 Scanning audio directory: {self.data_dir}")
        audio_files = []
        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.wma']
        
        for ext in audio_extensions:
            audio_files.extend(Path(self.data_dir).rglob(f'*{ext}'))
        
        # Convert to relative paths
        audio_files = [str(f.relative_to(self.data_dir)) for f in audio_files]
        print(f"   Found {len(audio_files)} audio files")
        
        # Filter out already processed files
        if not force_reprocess:
            to_process = [f for f in audio_files if f not in processed_files]
            print(f"   Files needing processing: {len(to_process)}")
        else:
            to_process = audio_files
            print(f"   Force reprocessing all files")
        
        if sample_size:
            to_process = to_process[:sample_size]
            print(f"   Processing sample: {len(to_process)} files")
        
        if not to_process:
            print("   ✅ All files already processed!")
            return
        
        # Process files
        print(f"\n🎙️  Starting speaker diarization...\n")
        successful = 0
        failed = 0
        new_results = []
        
        for file_name in tqdm(to_process, desc="Processing files"):
            # Construct full path
            full_path = os.path.join(self.data_dir, file_name)
            
            # Check if file exists
            if not os.path.exists(full_path):
                print(f"\n⚠️  File not found: {file_name}")
                failed += 1
                continue
            
            # Estimate speakers
            num_speakers = self.estimate_speakers(full_path, verbose=False)
            
            if num_speakers is not None:
                new_results.append({
                    'file_name': file_name,
                    'unique_speakers': num_speakers
                })
                successful += 1
            else:
                # Still record failed attempts
                new_results.append({
                    'file_name': file_name,
                    'unique_speakers': None
                })
                failed += 1
            
            # Save progress every 50 files
            if len(new_results) % 50 == 0 and len(new_results) > 0:
                temp_df = pd.DataFrame(new_results)
                combined_df = pd.concat([results_df, temp_df], ignore_index=True)
                combined_df.to_csv(output_csv, index=False)
                print(f"\n💾 Checkpoint saved ({successful + failed} processed)")
        
        # Combine with existing results and save final CSV
        if new_results:
            print(f"\n💾 Saving final results...")
            new_df = pd.DataFrame(new_results)
            combined_df = pd.concat([results_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['file_name'], keep='last')
            combined_df.to_csv(output_csv, index=False)
            print(f"   ✅ Saved: {output_csv}")
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"   Successfully processed: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Total processed: {successful + failed}")
        
        # Show distribution of speaker counts
        final_df = pd.read_csv(output_csv)
        speaker_counts = final_df['unique_speakers'].dropna()
        if len(speaker_counts) > 0:
            print(f"\n🎤 Speaker Distribution:")
            for count in sorted(speaker_counts.unique()):
                if isinstance(count, (int, float)) and not pd.isna(count):
                    num_files = (speaker_counts == count).sum()
                    print(f"   {int(count)} speaker(s): {num_files} files")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Estimate unique speakers using pyannote.audio speaker diarization"
    )
    parser.add_argument(
        '--output',
        default='results/speakers_per_records.csv',
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--sample',
        type=int,
        help='Process only first N files (for testing)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Reprocess all files, even if already processed'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage (disable GPU)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 70)
    print("SPEAKER DIARIZATION ESTIMATOR - Mandela Foundation".center(70))
    print("=" * 70)
    
    # Initialize estimator
    estimator = SpeakerDiarizationEstimator(use_gpu=not args.cpu)
    
    # Process audio files
    estimator.process_audio_files(
        output_csv=args.output,
        force_reprocess=args.force,
        sample_size=args.sample
    )
    
    print("\n✅ Processing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
