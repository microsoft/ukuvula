# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Audio processing utilities for the WhisperX ASR pipeline.

This module provides functions for audio preprocessing, validation, noise reduction,
and format conversion to prepare audio files for transcription.
"""

import os
import logging
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Tuple, Optional, List
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logger
logger = logging.getLogger(__name__)

def validate_audio_file(file_path: str) -> bool:
    """
    Validate audio file integrity and format compatibility.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return False
            
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.error(f"Audio file is empty: {file_path}")
            return False
            
        # Try to read audio metadata
        info = sf.info(file_path)
        
        # Check basic properties
        if info.duration <= 0:
            logger.error(f"Invalid audio duration: {file_path}")
            return False
            
        if info.samplerate <= 0:
            logger.error(f"Invalid sample rate: {file_path}")
            return False
            
        if info.channels <= 0:
            logger.error(f"Invalid number of channels: {file_path}")
            return False
            
        logger.info(f"Audio file validated: {file_path} - Duration: {info.duration:.2f}s, "
                   f"Sample rate: {info.samplerate}Hz, Channels: {info.channels}")
        return True
        
    except Exception as e:
        logger.error(f"Error validating audio file {file_path}: {str(e)}")
        return False

def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file with error handling and format conversion.
    
    Args:
        file_path (str): Path to the audio file
        target_sr (int): Target sample rate for output
        
    Returns:
        Tuple[np.ndarray, int]: Audio data and sample rate
        
    Raises:
        Exception: If audio loading fails
    """
    try:
        logger.info(f"Loading audio file: {file_path}")
        
        # Load audio using librosa for better format support
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True, dtype=np.float32)
        
        # Validate loaded audio
        if len(audio) == 0:
            raise ValueError("Loaded audio is empty")
            
        # Normalize audio to prevent clipping
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
            
        logger.info(f"Audio loaded successfully - Shape: {audio.shape}, Duration: {len(audio)/sr:.2f}s")
        return audio, sr
        
    except Exception as e:
        logger.error(f"Failed to load audio file {file_path}: {str(e)}")
        raise

def reduce_noise(audio: np.ndarray, sr: int, strength: float = 0.8) -> np.ndarray:
    """
    Apply noise reduction to audio signal.
    
    Args:
        audio (np.ndarray): Input audio signal
        sr (int): Sample rate
        strength (float): Noise reduction strength (0.0-1.0)
        
    Returns:
        np.ndarray: Noise-reduced audio signal
    """
    try:
        # Try to import noisereduce
        import noisereduce as nr
        
        # Apply noise reduction
        reduced_noise = nr.reduce_noise(y=audio, sr=sr, prop_decrease=strength)
        
        logger.debug(f"Noise reduction applied with strength: {strength}")
        return reduced_noise
        
    except ImportError:
        logger.warning("noisereduce not available, skipping noise reduction")
        return audio
    except Exception as e:
        logger.warning(f"Noise reduction failed: {str(e)}, returning original audio")
        return audio

def normalize_audio_level(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """
    Normalize audio level to target RMS value.
    
    Args:
        audio (np.ndarray): Input audio signal
        target_rms (float): Target RMS level
        
    Returns:
        np.ndarray: Level-normalized audio
    """
    try:
        # Calculate current RMS
        current_rms = np.sqrt(np.mean(audio**2))
        
        if current_rms > 0:
            # Calculate scaling factor
            scaling_factor = target_rms / current_rms
            
            # Apply scaling with clipping protection
            normalized = audio * scaling_factor
            normalized = np.clip(normalized, -1.0, 1.0)
            
            logger.debug(f"Audio level normalized - Original RMS: {current_rms:.4f}, "
                        f"Target RMS: {target_rms:.4f}")
            return normalized
        else:
            logger.warning("Audio signal has zero RMS, returning original")
            return audio
            
    except Exception as e:
        logger.warning(f"Audio normalization failed: {str(e)}, returning original audio")
        return audio

def split_audio_chunks(audio: np.ndarray, sr: int, chunk_duration: float = 120.0, 
                      overlap: float = 5.0) -> List[Tuple[np.ndarray, float, float]]:
    """
    Split audio into overlapping chunks for processing.
    
    Args:
        audio (np.ndarray): Input audio signal
        sr (int): Sample rate
        chunk_duration (float): Duration of each chunk in seconds
        overlap (float): Overlap between chunks in seconds
        
    Returns:
        List[Tuple[np.ndarray, float, float]]: List of (chunk_audio, start_time, end_time)
    """
    chunks = []
    
    total_duration = len(audio) / sr
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap * sr)
    step_samples = chunk_samples - overlap_samples
    
    start_sample = 0
    
    while start_sample < len(audio):
        # Calculate end sample
        end_sample = min(start_sample + chunk_samples, len(audio))
        
        # Extract chunk
        chunk_audio = audio[start_sample:end_sample]
        
        # Calculate time boundaries
        start_time = start_sample / sr
        end_time = end_sample / sr
        
        # Only add chunk if it has meaningful duration
        if len(chunk_audio) > sr:  # At least 1 second
            chunks.append((chunk_audio, start_time, end_time))
            logger.debug(f"Created chunk: {start_time:.2f}s - {end_time:.2f}s")
        
        # Move to next chunk
        start_sample += step_samples
        
        # If remaining audio is less than half chunk duration, include it in last chunk
        if len(audio) - start_sample < chunk_samples // 2:
            break
    
    logger.info(f"Audio split into {len(chunks)} chunks")
    return chunks

def detect_speech_activity(audio: np.ndarray, sr: int, 
                          energy_threshold: float = 0.01) -> List[Tuple[float, float]]:
    """
    Detect speech activity regions in audio using energy-based method.
    
    Args:
        audio (np.ndarray): Input audio signal
        sr (int): Sample rate
        energy_threshold (float): Energy threshold for speech detection
        
    Returns:
        List[Tuple[float, float]]: List of (start_time, end_time) for speech regions
    """
    try:
        # Calculate frame-wise energy
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Compute short-time energy
        energy = []
        for i in range(0, len(audio) - frame_length + 1, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sum(frame ** 2))
        
        energy = np.array(energy)
        
        # Normalize energy
        if np.max(energy) > 0:
            energy = energy / np.max(energy)
        
        # Detect speech regions
        speech_mask = energy > energy_threshold
        
        # Find contiguous speech regions
        speech_regions = []
        start_idx = None
        
        for i, is_speech in enumerate(speech_mask):
            if is_speech and start_idx is None:
                start_idx = i
            elif not is_speech and start_idx is not None:
                start_time = start_idx * hop_length / sr
                end_time = i * hop_length / sr
                speech_regions.append((start_time, end_time))
                start_idx = None
        
        # Handle case where audio ends with speech
        if start_idx is not None:
            start_time = start_idx * hop_length / sr
            end_time = len(audio) / sr
            speech_regions.append((start_time, end_time))
        
        logger.debug(f"Detected {len(speech_regions)} speech regions")
        return speech_regions
        
    except Exception as e:
        logger.warning(f"Speech activity detection failed: {str(e)}")
        # Return entire audio as single speech region
        return [(0.0, len(audio) / sr)]

def save_audio(audio: np.ndarray, sr: int, output_path: str) -> bool:
    """
    Save audio to file with error handling.
    
    Args:
        audio (np.ndarray): Audio signal to save
        sr (int): Sample rate
        output_path (str): Output file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save audio file
        sf.write(output_path, audio, sr, format='WAV', subtype='PCM_16')
        
        logger.info(f"Audio saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save audio to {output_path}: {str(e)}")
        return False

def get_audio_info(file_path: str) -> dict:
    """
    Get comprehensive audio file information.
    
    Args:
        file_path (str): Path to audio file
        
    Returns:
        dict: Audio file information
    """
    try:
        info = sf.info(file_path)
        file_size = os.path.getsize(file_path)
        
        return {
            'file_path': file_path,
            'file_size_mb': file_size / (1024 * 1024),
            'duration': info.duration,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'format': info.format,
            'subtype': info.subtype,
            'frames': info.frames
        }
        
    except Exception as e:
        logger.error(f"Failed to get audio info for {file_path}: {str(e)}")
        return {}

def preprocess_audio(file_path: str, target_sr: int = 16000, 
                    enable_noise_reduction: bool = True,
                    noise_strength: float = 0.8,
                    normalize_level: bool = True) -> Tuple[np.ndarray, int]:
    """
    Complete audio preprocessing pipeline.
    
    Args:
        file_path (str): Path to input audio file
        target_sr (int): Target sample rate
        enable_noise_reduction (bool): Whether to apply noise reduction
        noise_strength (float): Noise reduction strength
        normalize_level (bool): Whether to normalize audio level
        
    Returns:
        Tuple[np.ndarray, int]: Preprocessed audio and sample rate
        
    Raises:
        Exception: If preprocessing fails
    """
    try:
        # Validate file
        if not validate_audio_file(file_path):
            raise ValueError(f"Invalid audio file: {file_path}")
        
        # Load audio
        audio, sr = load_audio(file_path, target_sr)
        
        # Apply noise reduction
        if enable_noise_reduction:
            audio = reduce_noise(audio, sr, noise_strength)
        
        # Normalize audio level
        if normalize_level:
            audio = normalize_audio_level(audio)
        
        logger.info(f"Audio preprocessing completed for: {file_path}")
        return audio, sr
        
    except Exception as e:
        logger.error(f"Audio preprocessing failed for {file_path}: {str(e)}")
        raise