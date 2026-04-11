# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
WhisperX transcription module for the ASR pipeline.

This module handles transcription, forced alignment, and speaker diarization
using the WhisperX library for accurate timestamped transcriptions.
"""

import os
import logging
import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import torch

from config import (
    COMPUTE_TYPE,
    BATCH_SIZE,
    VAD_METHOD,
    VAD_DEVICE,
    VAD_ONSET,
    VAD_OFFSET,
    VAD_CHUNK_SIZE,
)

# Optional CUDA deterministic/compat settings (previously in cuda_compat)
def _apply_cuda_compat():
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    logger.info("Applied inline CUDA determinism settings")

if os.getenv("WHISPERX_CUDA_PATCH", "0") in ("1", "true", "True"):  # opt-in
    _apply_cuda_compat()

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logger
logger = logging.getLogger(__name__)

class WhisperXTranscriber:
    """
    WhisperX-based transcriber with alignment and diarization capabilities.
    """
    
    def __init__(self, model_size: str = "large-v2", language: str = "en", 
                 compute_type: str = "float16", device: str = "auto",
                 batch_size: int = 16, enable_diarization: bool = True,
                 vad_method: str = "pyannote", vad_device: str = "auto",
                 vad_onset: float = 0.5, vad_offset: float = 0.363,
                 vad_chunk_size: float = 30.0):
        """
        Initialize the WhisperX transcriber.
        
        Args:
            model_size (str): Whisper model size
            language (str): Primary language for transcription
            compute_type (str): Computation precision
            device (str): Device for inference
            batch_size (int): Batch size for processing
            enable_diarization (bool): Whether to enable speaker diarization
        """
        self.model_size = model_size
        self.language = language
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.enable_diarization = enable_diarization
        self.vad_method = vad_method.lower() if vad_method else "pyannote"
        self.vad_device = vad_device.lower() if vad_device else "auto"
        self.vad_onset = vad_onset
        self.vad_offset = vad_offset
        self.vad_chunk_size = vad_chunk_size
        
        # Determine device
        resolved_device = self._get_device(device)
        self.device_index = self._get_device_index(resolved_device)
        self.device = self._normalize_device(resolved_device)
        self.resolved_vad_device = self._resolve_vad_device()
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None
        
        self._load_models()
    
    def _get_device(self, device: str) -> str:
        """Determine the best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device

    def _normalize_device(self, device: str) -> str:
        """Normalize device strings to formats expected by WhisperX."""
        if device.startswith("cuda") and ":" in device:
            return "cuda"
        return device

    def _get_device_index(self, device: str) -> int:
        """Extract CUDA device index if available."""
        if device.startswith("cuda"):
            parts = device.split(":")
            if len(parts) > 1:
                try:
                    return int(parts[1])
                except ValueError:
                    logger.warning(f"Invalid CUDA device index in '{device}', defaulting to 0")
            return 0
        return 0

    def _resolve_vad_device(self) -> str:
        """Determine which device to run VAD on."""
        if self.vad_device in (None, "auto"):
            if self.device.startswith("cuda") and torch.cuda.is_available():
                return f"cuda:{self.device_index}"
            return "cpu"

        if self.vad_device == "cuda":
            if torch.cuda.is_available():
                return f"cuda:{self.device_index}"
            logger.warning("CUDA requested for VAD but no GPU available, falling back to CPU")
            return "cpu"

        if self.vad_device.startswith("cuda"):
            if torch.cuda.is_available():
                return self.vad_device
            logger.warning("CUDA requested for VAD but no GPU available, falling back to CPU")
            return "cpu"

        return self.vad_device
    
    def _load_models(self):
        """Load WhisperX models with error handling."""
        try:
            # Import WhisperX
            import whisperx
            
            # Load transcription model
            logger.info(f"Loading Whisper model: {self.model_size}")
            vad_options = {
                "vad_onset": self.vad_onset,
                "vad_offset": self.vad_offset,
                "chunk_size": self.vad_chunk_size
            }

            load_kwargs = {
                "compute_type": self.compute_type,
                "language": self.language,
                "device_index": self.device_index,
                "vad_method": self.vad_method,
                "vad_options": vad_options
            }

            vad_model = None

            if self.vad_method == "pyannote":
                logger.info(f"Configuring Pyannote VAD on {self.resolved_vad_device}")
                if not self.resolved_vad_device.startswith("cuda"):
                    try:
                        from whisperx.vads.pyannote import Pyannote
                        vad_model = Pyannote(torch.device(self.resolved_vad_device), use_auth_token=None, **vad_options)
                    except Exception as vad_error:
                        logger.warning(f"Failed to initialize Pyannote VAD on {self.resolved_vad_device}: {vad_error}. Falling back to Silero VAD.")
                        self.vad_method = "silero"
                        load_kwargs["vad_method"] = "silero"
                else:
                    logger.info("Using Pyannote VAD on GPU")
            elif self.vad_method == "silero":
                logger.info("Using Silero VAD")
            else:
                logger.warning(f"Unsupported VAD method '{self.vad_method}', falling back to Silero")
                self.vad_method = "silero"
                load_kwargs["vad_method"] = "silero"

            if vad_model is not None:
                load_kwargs["vad_model"] = vad_model

            self.model = whisperx.load_model(
                self.model_size,
                self.device,
                **load_kwargs
            )
            logger.info("Whisper model loaded successfully")
            
            # Load alignment model
            try:
                logger.info("Loading alignment model...")
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=self.language, 
                    device=self.device
                )
                logger.info("Alignment model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load alignment model: {str(e)}")
                self.align_model = None
                self.align_metadata = None
            
            # Load diarization model
            if self.enable_diarization:
                try:
                    logger.info("Loading diarization model...")
                    self.diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=None,  # You may need to set this for pyannote
                        device=self.device
                    )
                    logger.info("Diarization model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load diarization model: {str(e)}")
                    self.diarize_model = None
                    self.enable_diarization = False
            
        except ImportError as e:
            logger.error(f"WhisperX import failed: {str(e)}")
            raise ImportError("WhisperX is not installed. Please install it with: pip install whisperx")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise
    
    def transcribe_audio(self, audio: np.ndarray, sample_rate: int = 16000, quiet: bool = False) -> Dict[str, Any]:
        """
        Transcribe audio using WhisperX.
        
        Args:
            audio (np.ndarray): Audio signal
            sample_rate (int): Audio sample rate
            
        Returns:
            Dict[str, Any]: Transcription results
        """
        try:
            if self.model is None:
                raise ValueError("Whisper model not loaded")
            
            logger.info("Starting transcription...")
            
            # Prepare audio for WhisperX
            if len(audio.shape) > 1:
                audio = audio.flatten()
            
            # Ensure audio is float32
            audio = audio.astype(np.float32)
            
            # Normalize audio for better low-volume speech detection
            if np.max(np.abs(audio)) > 0:
                # Normalize to 80% of maximum to preserve dynamics but boost low volume
                peak = np.max(np.abs(audio))
                if peak < 0.1:  # Very low volume audio
                    audio = audio / peak * 0.8
                    logger.info(f"Audio volume boosted for low-volume speech (peak: {peak:.4f})")
                elif peak > 0.95:  # Very loud audio  
                    audio = audio * 0.7  # Slightly reduce to prevent clipping
                    logger.info(f"Audio volume reduced to prevent clipping (peak: {peak:.4f})")
            
            # Transcribe with enhanced settings for low-volume speech
            try:
                # First attempt: standard transcription
                result = self.model.transcribe(
                    audio,
                    batch_size=self.batch_size,
                    language=self.language,
                    print_progress=not quiet
                )
                
                # If no segments found, the issue might be VAD sensitivity
                # Let's try chunk-based processing for better detection
                if not result.get('segments') or len(result['segments']) == 0:
                    logger.warning("No segments found, trying chunk-based processing for low-volume speech...")
                    
                    # Process in smaller chunks for better low-volume detection
                    chunk_duration = 30  # 30-second chunks
                    sample_rate = 16000  # WhisperX expects 16kHz
                    chunk_samples = chunk_duration * sample_rate
                    all_segments = []
                    
                    for i in range(0, len(audio), chunk_samples):
                        chunk = audio[i:i + chunk_samples]
                        if len(chunk) < sample_rate:  # Skip very short chunks
                            continue
                        
                        # Boost chunk volume for better detection
                        if np.max(np.abs(chunk)) > 0:
                            chunk = chunk / np.max(np.abs(chunk)) * 0.8
                        
                        try:
                            chunk_result = self.model.transcribe(
                                chunk,
                                batch_size=self.batch_size,
                                language=self.language,
                                print_progress=False  # always suppress for chunks
                            )
                            
                            # Adjust timestamps for chunk offset
                            if chunk_result.get('segments'):
                                chunk_offset = i / sample_rate
                                for segment in chunk_result['segments']:
                                    segment['start'] += chunk_offset
                                    segment['end'] += chunk_offset
                                all_segments.extend(chunk_result['segments'])
                                
                        except Exception as chunk_e:
                            logger.warning(f"Chunk {i//chunk_samples} failed: {chunk_e}")
                            continue
                    
                    if all_segments:
                        result = {'segments': all_segments, 'language': self.language}
                        logger.info(f"Chunk-based processing successful - Found {len(all_segments)} segments")
                    else:
                        logger.warning("Even chunk-based processing found no speech segments")
                        
            except Exception as e:
                logger.error(f"All transcription methods failed: {e}")
                result = {'segments': [], 'language': self.language}
            
            logger.info(f"Transcription completed - Found {len(result.get('segments', []))} segments")
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise
    
    def align_transcription(self, transcription_result: Dict[str, Any], 
                          audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Perform forced alignment on transcription results.
        
        Args:
            transcription_result (Dict[str, Any]): Whisper transcription results
            audio (np.ndarray): Original audio signal
            sample_rate (int): Audio sample rate
            
        Returns:
            Dict[str, Any]: Aligned transcription results
        """
        try:
            if self.align_model is None or self.align_metadata is None:
                logger.warning("Alignment model not available, returning original transcription")
                return transcription_result
            
            logger.info("Starting forced alignment...")
            
            # Import WhisperX
            import whisperx
            
            # Prepare audio
            if len(audio.shape) > 1:
                audio = audio.flatten()
            audio = audio.astype(np.float32)
            
            # Perform alignment
            aligned_result = whisperx.align(
                transcription_result["segments"], 
                self.align_model, 
                self.align_metadata, 
                audio, 
                self.device, 
                return_char_alignments=False
            )
            
            logger.info("Forced alignment completed")
            return aligned_result
            
        except Exception as e:
            logger.warning(f"Alignment failed: {str(e)}, returning original transcription")
            return transcription_result
    
    def diarize_speakers(self, audio: np.ndarray, sample_rate: int = 16000,
                        min_speakers: int = 1, max_speakers: int = 10) -> Optional[Dict[str, Any]]:
        """
        Perform speaker diarization on audio.
        
        Args:
            audio (np.ndarray): Audio signal
            sample_rate (int): Audio sample rate
            min_speakers (int): Minimum number of speakers
            max_speakers (int): Maximum number of speakers
            
        Returns:
            Optional[Dict[str, Any]]: Diarization results
        """
        try:
            if not self.enable_diarization or self.diarize_model is None:
                logger.info("Speaker diarization not available")
                return None
            
            logger.info("Starting speaker diarization...")
            
            # Prepare audio
            if len(audio.shape) > 1:
                audio = audio.flatten()
            audio = audio.astype(np.float32)
            
            # Perform diarization
            diarize_segments = self.diarize_model(
                audio,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            logger.info(f"Speaker diarization completed - Found speakers: {len(set(seg['speaker'] for seg in diarize_segments))}")
            return diarize_segments
            
        except Exception as e:
            logger.warning(f"Speaker diarization failed: {str(e)}")
            return None
    
    def assign_speaker_labels(self, aligned_result: Dict[str, Any], 
                            diarize_segments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assign speaker labels to transcription segments.
        
        Args:
            aligned_result (Dict[str, Any]): Aligned transcription results
            diarize_segments (Optional[Dict[str, Any]]): Diarization results
            
        Returns:
            Dict[str, Any]: Transcription with speaker labels
        """
        try:
            if diarize_segments is None:
                # Assign default speaker if no diarization
                for segment in aligned_result.get("segments", []):
                    segment["speaker"] = "Speaker 1"
                logger.info("Assigned default speaker labels")
                return aligned_result
            
            logger.info("Assigning speaker labels...")
            
            # Import WhisperX
            import whisperx
            
            # Assign speaker labels
            result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
            
            logger.info("Speaker labels assigned successfully")
            return result
            
        except Exception as e:
            logger.warning(f"Speaker label assignment failed: {str(e)}")
            # Assign default speaker labels
            for segment in aligned_result.get("segments", []):
                segment["speaker"] = "Speaker 1"
            return aligned_result
    
    def process_audio_file(self, audio: np.ndarray, sample_rate: int = 16000,
                          min_speakers: int = 1, max_speakers: int = 10, quiet: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Complete transcription pipeline for audio file.
        
        Args:
            audio (np.ndarray): Audio signal
            sample_rate (int): Audio sample rate
            min_speakers (int): Minimum number of speakers
            max_speakers (int): Maximum number of speakers
            
        Returns:
            Dict[str, Any]: Complete transcription results with speakers and alignment
        """
        try:
            # Step 1: Transcribe
            # Support legacy callers that might pass quiet in kwargs
            if 'quiet' in kwargs and not quiet:
                quiet = kwargs.get('quiet', quiet)
            transcription_result = self.transcribe_audio(audio, sample_rate, quiet=quiet)
            
            # Step 2: Align
            aligned_result = self.align_transcription(transcription_result, audio, sample_rate)
            
            # Step 3: Diarize (if enabled)
            diarize_segments = None
            if self.enable_diarization:
                diarize_segments = self.diarize_speakers(audio, sample_rate, min_speakers, max_speakers)
            
            # Step 4: Assign speaker labels
            final_result = self.assign_speaker_labels(aligned_result, diarize_segments)
            
            # Add metadata
            final_result["metadata"] = {
                "model_size": self.model_size,
                "language": self.language,
                "device": self.device,
                "sample_rate": sample_rate,
                "diarization_enabled": self.enable_diarization,
                "total_segments": len(final_result.get("segments", []))
            }
            
            logger.info("Audio processing completed successfully")
            return final_result
            
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            raise
    
    def cleanup(self):
        """Clean up model resources."""
        try:
            # Clear CUDA cache if using GPU
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Model cleanup completed")
            
        except Exception as e:
            logger.warning(f"Model cleanup failed: {str(e)}")

def create_transcriber(model_size: str = "large-v2", language: str = "en",
                      device: str = "auto", enable_diarization: bool = True,
                      compute_type: str = COMPUTE_TYPE, batch_size: int = BATCH_SIZE,
                      vad_method: str = VAD_METHOD, vad_device: str = VAD_DEVICE,
                      vad_onset: float = VAD_ONSET, vad_offset: float = VAD_OFFSET,
                      vad_chunk_size: float = VAD_CHUNK_SIZE) -> WhisperXTranscriber:
    """
    Factory function to create a WhisperX transcriber instance.
    
    Args:
        model_size (str): Whisper model size
        language (str): Primary language
        device (str): Device for inference
        enable_diarization (bool): Enable speaker diarization
        
    Returns:
        WhisperXTranscriber: Configured transcriber instance
    """
    return WhisperXTranscriber(
        model_size=model_size,
        language=language,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        enable_diarization=enable_diarization,
        vad_method=vad_method,
        vad_device=vad_device,
        vad_onset=vad_onset,
        vad_offset=vad_offset,
        vad_chunk_size=vad_chunk_size
    )

def transcribe_audio_simple(audio: np.ndarray, sample_rate: int = 16000,
                          model_size: str = "large-v2", language: str = "en") -> List[Dict[str, Any]]:
    """
    Simple transcription function for single audio processing.
    
    Args:
        audio (np.ndarray): Audio signal
        sample_rate (int): Audio sample rate
        model_size (str): Whisper model size
        language (str): Language code
        
    Returns:
        List[Dict[str, Any]]: List of transcription segments
    """
    transcriber = create_transcriber(model_size=model_size, language=language)
    
    try:
        result = transcriber.process_audio_file(audio, sample_rate)
        return result.get("segments", [])
    finally:
        transcriber.cleanup()