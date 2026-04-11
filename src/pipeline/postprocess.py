# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Post-processing utilities for transcription results.

This module provides functions for cleaning, filtering, and enhancing
transcription results from the ASR pipeline.
"""

import re
import logging
from typing import List, Dict, Any, Optional
import string
from collections import Counter
import math

# Setup logger
logger = logging.getLogger(__name__)

# Language detection
try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0  # For consistent results
    LANGDETECT_AVAILABLE = True
except ImportError:
    logger.warning("langdetect not available, language filtering will be disabled")
    LANGDETECT_AVAILABLE = False

class TranscriptionPostProcessor:
    """
    Post-processor for cleaning and enhancing transcription results.
    """
    
    def __init__(self, target_language: str = "en", min_confidence: float = 0.4,
                 remove_gibberish: bool = True, normalize_text: bool = True):
        """
        Initialize the post-processor.
        
        Args:
            target_language (str): Target language code
            min_confidence (float): Minimum confidence threshold
            remove_gibberish (bool): Whether to remove likely gibberish
            normalize_text (bool): Whether to normalize text
        """
        self.target_language = target_language
        self.min_confidence = min_confidence
        self.remove_gibberish = remove_gibberish
        self.normalize_text = normalize_text
        
        # Common gibberish patterns
        self.gibberish_patterns = [
            r'^[a-z]{1,2}$',  # Very short words
            r'^[^a-zA-Z]*$',   # Only symbols/numbers
            r'(.)\1{4,}',      # Repeated characters (5+)
            r'^(ha|he|hi|ho|hu|ah|eh|ih|oh|uh){3,}$',  # Repeated sounds
            r'^\W+$',          # Only non-word characters
        ]
        
        # Common filler words and false starts
        self.filler_words = {
            'um', 'uh', 'er', 'ah', 'hmm', 'mhm', 'mm', 'hm',
            'like', 'you know', 'i mean', 'basically', 'actually', 'literally'
        }
        
        # Word validation patterns
        self.valid_word_pattern = re.compile(r'^[a-zA-Z][a-zA-Z0-9\'-]*[a-zA-Z0-9]$|^[a-zA-Z]$')
    
    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect the language of a text segment.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Optional[str]: Detected language code or None
        """
        if not LANGDETECT_AVAILABLE or not text.strip():
            return None
            
        try:
            # Clean text for language detection
            clean_text = re.sub(r'[^\w\s]', '', text)
            if len(clean_text.split()) < 3:  # Need at least 3 words for reliable detection
                return None
                
            detected_lang = detect(clean_text)
            return detected_lang
            
        except (LangDetectException, Exception) as e:
            logger.debug(f"Language detection failed for text '{text[:50]}...': {str(e)}")
            return None
    
    def is_gibberish(self, text: str) -> bool:
        """
        Check if text is likely gibberish.
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if text is likely gibberish
        """
        if not self.remove_gibberish:
            return False
            
        # Check against gibberish patterns
        for pattern in self.gibberish_patterns:
            if re.search(pattern, text.lower()):
                return True
        
        # Check word validity ratio (only meaningful for >=4 words)
        words = text.split()
        if not words:
            return True

        if len(words) >= 4:
            valid_words = sum(1 for word in words if self.valid_word_pattern.match(word))
            validity_ratio = valid_words / len(words)
            if validity_ratio < 0.5:
                return True

            # Repetition check only for longer segments
            word_counts = Counter(word.lower() for word in words)
            max_repetition = max(word_counts.values()) if word_counts else 0
            repetition_ratio = max_repetition / len(words) if words else 0
            if repetition_ratio > 0.5:  # slightly more lenient
                return True
        
        return False
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize transcription text.
        
        Args:
            text (str): Raw transcription text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        if not self.normalize_text:
            return text
        
        # Fix common transcription errors
        text = self._fix_common_errors(text)
        
        # Normalize punctuation
        text = self._normalize_punctuation(text)
        
        # Remove excessive filler words
        text = self._remove_excessive_fillers(text)
        
        # Standardize capitalization
        text = self._standardize_capitalization(text)
        
        return text.strip()
    
    def _fix_common_errors(self, text: str) -> str:
        """Fix common ASR transcription errors."""
        # Common replacements
        replacements = {
            r'\bmandalay\b': 'Mandela',
            r'\bmandala\b': 'Mandela',
            r'\bmandela\b': 'Mandela',
            r'\bi\s+m\b': "I'm",
            r'\bdont\b': "don't",
            r'\bcant\b': "can't",
            r'\bwont\b': "won't",
            r'\bthats\b': "that's",
            r'\bwhats\b': "what's",
            r'\bits\b': "it's",
            r'\byoure\b': "you're",
            r'\btheyre\b': "they're",
            r'\bwere\b': "we're",
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation in text."""
        # Remove multiple consecutive punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[,]{2,}', ',', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Remove punctuation at the beginning
        text = re.sub(r'^[.!?,:;]+\s*', '', text)
        
        return text
    
    def _remove_excessive_fillers(self, text: str) -> str:
        """Remove excessive filler words."""
        words = text.split()
        filtered_words = []
        
        i = 0
        while i < len(words):
            word = words[i].lower().strip(string.punctuation)
            
            # Check for repeated filler words
            if word in self.filler_words:
                # Count consecutive occurrences
                count = 1
                j = i + 1
                while j < len(words) and words[j].lower().strip(string.punctuation) == word:
                    count += 1
                    j += 1
                
                # Keep only one occurrence if repeated
                if count > 1:
                    filtered_words.append(words[i])
                    i = j
                else:
                    filtered_words.append(words[i])
                    i += 1
            else:
                filtered_words.append(words[i])
                i += 1
        
        return ' '.join(filtered_words)
    
    def _standardize_capitalization(self, text: str) -> str:
        """Standardize capitalization."""
        if not text:
            return text
        
        # Capitalize first letter
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Capitalize after sentence endings
        text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        # Capitalize proper nouns (basic)
        proper_nouns = ['mandela', 'nelson', 'south africa', 'african', 'apartheid']
        for noun in proper_nouns:
            pattern = r'\b' + re.escape(noun) + r'\b'
            text = re.sub(pattern, noun.title(), text, flags=re.IGNORECASE)
        
        return text
    
    def filter_segment(self, segment: Dict[str, Any]) -> bool:
        """
        Check if a transcription segment should be kept.
        
        Args:
            segment (Dict[str, Any]): Transcription segment
            
        Returns:
            bool: True if segment should be kept
        """
        # Derive a confidence-like score that works with WhisperX output
        raw_conf = None
        if 'confidence' in segment and segment['confidence'] is not None:
            raw_conf = segment['confidence']
        else:
            # Whisper / WhisperX typically provides avg_logprob (negative ~ -0.1 to -1.5)
            avg_logprob = segment.get('avg_logprob')
            no_speech_prob = segment.get('no_speech_prob')
            if avg_logprob is not None:
                # Convert log probability (natural log) to [0,1] range via exp then clamp
                try:
                    raw_conf = math.exp(float(avg_logprob))  # exp(-0.3) ~0.74, exp(-1.0) ~0.37
                except Exception:
                    raw_conf = 0.0
            elif no_speech_prob is not None:
                raw_conf = 1.0 - float(no_speech_prob)
            else:
                raw_conf = 1.0  # Fallback: keep segment

        # Attach computed confidence for downstream visibility
        segment['computed_confidence'] = raw_conf

        # Scale heuristic: remap exp(avg_logprob) into a boosted confidence so that
        # moderately negative logprobs are not over-penalized. Typical avg_logprob ~ -0.4 to -1.2.
        if 'avg_logprob' in segment and segment.get('avg_logprob') is not None:
            boosted = max(0.0, min(1.0, 1.2 * raw_conf))  # light boost
            segment['boosted_confidence'] = boosted
        else:
            segment['boosted_confidence'] = raw_conf

        effective_conf = segment['boosted_confidence']

        if effective_conf < self.min_confidence:
            # Allow at least the first temporal segment through to avoid empty outputs when heuristic is strict
            if segment.get('start', 0) == 0 and segment.get('end', 0) > 0:
                logger.debug("Keeping first segment despite low confidence to seed transcript (override)")
                segment['_override_keep'] = True
            else:
                logger.debug(f"Filtering low confidence segment (effective_confidence={effective_conf:.3f} < {self.min_confidence}) text='{segment.get('text','')[:60]}'")
                return False
        
        # Check text content
        text = segment.get('text', '').strip()
        if not text:
            return False
        
        # Check for gibberish
        if self.is_gibberish(text):
            if segment.get('_override_keep'):
                logger.debug("Gibberish heuristic suppressed for first low-confidence segment")
            else:
                logger.debug(f"Filtering gibberish segment: '{text}'")
                return False
        
        # Check language (if detection available)
        if LANGDETECT_AVAILABLE and len(text.split()) >= 3:
            detected_lang = self.detect_language(text)
            if detected_lang and detected_lang != self.target_language:
                logger.debug(f"Filtering non-{self.target_language} segment: '{text}' (detected: {detected_lang})")
                return False
        
        return True
    
    def process_segment(self, segment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single transcription segment.
        
        Args:
            segment (Dict[str, Any]): Input segment
            
        Returns:
            Optional[Dict[str, Any]]: Processed segment or None if filtered out
        """
        # Filter segment
        if not self.filter_segment(segment):
            return None
        
        # Clean text
        cleaned_text = self.clean_text(segment.get('text', ''))
        
        # Skip if text becomes empty after cleaning
        if not cleaned_text:
            return None
        
        # Create processed segment
        processed_segment = segment.copy()
        processed_segment['text'] = cleaned_text
        processed_segment['original_text'] = segment.get('text', '')
        processed_segment['processed'] = True
        
        return processed_segment
    
    def process_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of transcription segments.
        
        Args:
            segments (List[Dict[str, Any]]): Input segments
            
        Returns:
            List[Dict[str, Any]]: Processed segments
        """
        processed_segments = []
        
        for segment in segments:
            processed_segment = self.process_segment(segment)
            if processed_segment:
                processed_segments.append(processed_segment)

        if segments and not processed_segments:
            logger.warning(
                "All segments were filtered out. This may indicate the confidence heuristic is too strict. "
                f"min_confidence={self.min_confidence}. Consider lowering to e.g. 0.4 or 0.5."
            )
        else:
            logger.info(f"Processed {len(segments)} segments -> {len(processed_segments)} valid segments")
        return processed_segments

def create_post_processor(target_language: str = "en", min_confidence: float = 0.7) -> TranscriptionPostProcessor:
    """
    Factory function to create a post-processor instance.
    
    Args:
        target_language (str): Target language code
        min_confidence (float): Minimum confidence threshold
        
    Returns:
        TranscriptionPostProcessor: Configured post-processor
    """
    return TranscriptionPostProcessor(
        target_language=target_language,
        min_confidence=min_confidence
    )

def clean_transcription_text(text: str, target_language: str = "en") -> str:
    """
    Simple function to clean transcription text.
    
    Args:
        text (str): Input text
        target_language (str): Target language
        
    Returns:
        str: Cleaned text
    """
    processor = create_post_processor(target_language=target_language)
    return processor.clean_text(text)

def filter_segments_by_confidence(segments: List[Dict[str, Any]], 
                                min_confidence: float = 0.7) -> List[Dict[str, Any]]:
    """
    Filter segments by confidence threshold.
    
    Args:
        segments (List[Dict[str, Any]]): Input segments
        min_confidence (float): Minimum confidence threshold
        
    Returns:
        List[Dict[str, Any]]: Filtered segments
    """
    filtered = []
    for segment in segments:
        confidence = segment.get('confidence', segment.get('avg_logprob', 0))
        if confidence >= min_confidence:
            filtered.append(segment)
    
    logger.info(f"Filtered {len(segments)} -> {len(filtered)} segments by confidence")
    return filtered