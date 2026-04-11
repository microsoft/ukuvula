# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Configuration settings for the WhisperX ASR pipeline.

This module contains all configurable parameters for the transcription pipeline
including model settings, audio processing parameters, and output formatting options.
"""

import os
from pathlib import Path

# Model Configuration
MODEL_SIZE = "large-v2"  # Options: tiny, base, small, medium, large, large-v2, large-v3
LANGUAGE = "en"  # Primary language for transcription
COMPUTE_TYPE = "float16"  # Options: int8, float16, float32
BATCH_SIZE = 32  # Batch size for inference (optimized for V100 32GB)
VAD_METHOD = "pyannote"  # Options: pyannote, silero
VAD_DEVICE = "cuda"  # Options: auto, cpu, cuda, cuda:<index>
VAD_ONSET = 0.5  # VAD onset threshold
VAD_OFFSET = 0.363  # VAD offset threshold
VAD_CHUNK_SIZE = 30  # Chunk size for VAD segmentation (seconds)

# Audio Processing Parameters
CHUNK_DURATION = 120  # Duration in seconds (2 minutes)
TARGET_SAMPLE_RATE = 16000  # WhisperX optimal sample rate
MIN_SEGMENT_DURATION = 1.0  # Minimum segment duration in seconds
MAX_SEGMENT_DURATION = 30.0  # Maximum segment duration in seconds

# Quality Thresholds
MIN_CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence for including transcription (archival audio)
MIN_SPEECH_PROBABILITY = 0.5  # Minimum probability for speech detection
NOISE_REDUCTION_STRENGTH = 0.8  # Noise reduction strength (0.0 - 1.0)

# Speaker Diarization
ENABLE_DIARIZATION = True  # Enable speaker diarization
MIN_SPEAKERS = 1  # Minimum number of speakers
MAX_SPEAKERS = 10  # Maximum number of speakers

# File Processing
SUPPORTED_FORMATS = ['.wav', '.mp3', '.m4a', '.flac']  # Supported audio formats
OUTPUT_FORMAT = 'csv'  # Output format for transcriptions
ENCODING = 'utf-8'  # Text encoding for output files

# GPU Configuration
USE_GPU = True  # Enable GPU acceleration if available
DEVICE = "auto"  # Options: auto, cuda, cpu

# Logging Configuration
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "transcription_pipeline.log"

# Directory Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "nmf_recordings" / "Mandela at 90"
OUTPUT_DIR = BASE_DIR / "transcription_outputs"
TEMP_DIR = BASE_DIR / "temp"
LOGS_DIR = BASE_DIR / "logs"

# Language Detection
SUPPORTED_LANGUAGES = ['en']  # Only process English audio
LANGUAGE_DETECTION_THRESHOLD = 0.8  # Confidence threshold for language detection

# Post-processing
REMOVE_GIBBERISH = True  # Remove likely gibberish text
STANDARDIZE_PUNCTUATION = True  # Clean and standardize punctuation
NORMALIZE_WHITESPACE = True  # Remove extra whitespace
MIN_WORD_LENGTH = 2  # Minimum word length to keep
MAX_CONSECUTIVE_REPEATS = 3  # Maximum allowed consecutive word repeats

# Performance Settings
MAX_PARALLEL_WORKERS = 4  # Maximum parallel processing workers
MEMORY_LIMIT_GB = 8  # Memory limit for processing large files
CHECKPOINT_INTERVAL = 100  # Save progress every N files

# Validation Settings
VALIDATE_AUDIO_INTEGRITY = True  # Check audio file integrity before processing
SKIP_CORRUPTED_FILES = True  # Skip corrupted files instead of failing
BACKUP_ORIGINAL_FILES = False  # Create backups of original files

def get_device():
    """Automatically detect the best available device for processing."""
    if DEVICE == "auto":
        try:
            import torch
            if USE_GPU and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    return DEVICE

def create_directories():
    """Create necessary directories for the pipeline."""
    directories = [OUTPUT_DIR, TEMP_DIR, LOGS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Environment-specific overrides
if os.getenv('WHISPER_MODEL_SIZE'):
    MODEL_SIZE = os.getenv('WHISPER_MODEL_SIZE')

if os.getenv('WHISPER_LANGUAGE'):
    LANGUAGE = os.getenv('WHISPER_LANGUAGE')

if os.getenv('WHISPER_BATCH_SIZE'):
    BATCH_SIZE = int(os.getenv('WHISPER_BATCH_SIZE'))

if os.getenv('WHISPER_VAD_METHOD'):
    VAD_METHOD = os.getenv('WHISPER_VAD_METHOD')

if os.getenv('WHISPER_VAD_DEVICE'):
    VAD_DEVICE = os.getenv('WHISPER_VAD_DEVICE')

if os.getenv('WHISPER_VAD_ONSET'):
    VAD_ONSET = float(os.getenv('WHISPER_VAD_ONSET'))

if os.getenv('WHISPER_VAD_OFFSET'):
    VAD_OFFSET = float(os.getenv('WHISPER_VAD_OFFSET'))

if os.getenv('WHISPER_VAD_CHUNK_SIZE'):
    VAD_CHUNK_SIZE = float(os.getenv('WHISPER_VAD_CHUNK_SIZE'))

if os.getenv('DISABLE_GPU'):
    USE_GPU = False