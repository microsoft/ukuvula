#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Transcription Quality Evaluation for Nelson Mandela Foundation Archive

This script uses Azure OpenAI GPT models to evaluate transcription quality across
multiple linguistic dimensions: fluency, coherence, completeness, redundancy, and
lexical richness.

Usage:
    python quality_evaluation.py
    python quality_evaluation.py --input results/final_transcriptions.cleaned.csv --output results/quality_evaluation.csv
    python quality_evaluation.py --model gpt-4o --batch-size 10
"""

import os
import sys
# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
import time
from tqdm import tqdm
from azure_openai_utils import setup_azure_openai

# Ensure src/ is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/quality_eval_errors.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TranscriptionQualityEvaluator:
    """
    Evaluates transcription quality using Azure OpenAI GPT models.
    
    Quality metrics:
    - Fluency / grammaticality (0-100)
    - Coherence / consistency (0-100)
    - Completeness (0-100)
    - Redundancy (0-100, higher = more redundant → lower quality)
    - Lexical richness (0-100)
    """
    
    def __init__(self, input_file, output_file, model_deployment="gpt-4o", batch_size=10):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.model_deployment = model_deployment
        self.batch_size = batch_size
        self.df = None
        self.results = []
        self.failed_entries = []
        self.client = self.setup_azure_openai()
        
        # Quality metrics definition
        self.metrics = [
            "Fluency / grammaticality",
            "Coherence / consistency",
            "Completeness",
            "Redundancy",
            "Lexical richness"
        ]
        
    def setup_azure_openai(self):
        """Setup Azure OpenAI client with Entra ID authentication."""
        return setup_azure_openai()
    
    def load_data(self):
        """Load transcription data from CSV."""
        logger.info(f"Loading transcription data from {self.input_file}")
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        self.df = pd.read_csv(self.input_file)
        
        # Verify required columns: 'File name' and 'Note' for aggregated transcriptions
        required_cols = ['File name', 'Note']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Rename columns for internal consistency
        self.df['file_name'] = self.df['File name']
        self.df['transcription'] = self.df['Note']
        
        # Filter out rows with empty transcriptions
        self.df = self.df[self.df['transcription'].notna() & (self.df['transcription'].str.strip() != '')]
        
        print(f"✓ Loaded {len(self.df)} recordings with aggregated transcriptions", flush=True)
        logger.info(f"Loaded {len(self.df)} recordings with aggregated transcriptions")
        return self.df
    
    def create_evaluation_prompt(self, transcriptions_batch):
        """
        Create a structured prompt for GPT to evaluate transcription quality.
        
        Args:
            transcriptions_batch: List of dicts with transcription metadata
            
        Returns:
            System and user messages for API call
        """
        system_message = """You are an expert linguistic evaluator specializing in transcription quality assessment. 

Your task is to evaluate aggregated transcriptions from historical archival audio recordings on the following five quality metrics, each scored from 0 to 100:

1. **Fluency / grammaticality** (0-100): How natural, grammatically correct, and well-formed is the text? 
   - 100 = Perfect grammar, natural flow, reads like native speaker text
   - 0 = Incomprehensible, severe grammatical errors, fragmented

2. **Coherence / consistency** (0-100): Does the text make logical and contextual sense as a standalone excerpt?
   - 100 = Perfectly coherent, ideas flow logically, clear context
   - 0 = Incoherent, contradictory, no clear meaning

3. **Completeness** (0-100): Are sentences and ideas complete, or are they missing, cut off, or truncated?
   - 100 = All sentences complete, no missing content indicators
   - 0 = Mostly fragments, severe truncation, critical missing content

4. **Redundancy** (0-100): How repetitive or verbose is the content? 
   - **Higher score = MORE redundant/repetitive = LOWER quality**
   - 100 = Extremely repetitive, excessive verbosity, constant loops
   - 0 = No repetition, concise, each phrase adds value

5. **Lexical richness** (0-100): How varied and rich is the vocabulary?
   - 100 = Diverse vocabulary, sophisticated word choice, rich expression
   - 0 = Extremely limited vocabulary, constant word repetition

**Important**: For archival oral history transcriptions, expect some disfluency (hesitations, false starts) which is normal for spontaneous speech. Focus on whether the transcription captures intelligible, meaningful content.

Return your response as a valid JSON array with one object per recording. Each object must include:
- Original metadata fields: file_name, transcription
- Numeric scores (integers 0-100) for each of the five metrics

Example format:
```json
[
  {
    "file_name": "example.mp3",
    "transcription": "...",
    "Fluency / grammaticality": 85,
    "Coherence / consistency": 78,
    "Completeness": 92,
    "Redundancy": 15,
    "Lexical richness": 68
  }
]
```

Evaluate each recording independently. Be consistent and objective in your scoring."""

        # Format transcriptions batch as JSON for the user message
        transcriptions_json = json.dumps(transcriptions_batch, indent=2, ensure_ascii=False)
        
        user_message = f"""Please evaluate the following aggregated transcription(s) according to the five quality metrics described. Return valid JSON only.

Transcriptions to evaluate:
{transcriptions_json}"""
        
        return system_message, user_message
    
    def call_gpt_api(self, system_message, user_message, max_retries=3):
        """
        Call Azure OpenAI API with retry logic.
        
        Args:
            system_message: System prompt
            user_message: User prompt with transcriptions
            max_retries: Maximum number of retry attempts
            
        Returns:
            Parsed JSON response or None on failure
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_deployment,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.1,  # Low temperature for consistent evaluation
                    max_tokens=4000,
                    response_format={"type": "json_object"} if "gpt-4" in self.model_deployment else None
                )
                
                content = response.choices[0].message.content.strip()
                
                # Try to parse JSON
                try:
                    # Handle both direct array and object with array wrapper
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and 'evaluations' in parsed:
                        return parsed['evaluations']
                    elif isinstance(parsed, dict) and 'results' in parsed:
                        return parsed['results']
                    elif isinstance(parsed, list):
                        return parsed
                    else:
                        # Assume it's a single evaluation wrapped in object
                        return [parsed]
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error (attempt {attempt+1}/{max_retries}): {e}")
                    logger.debug(f"Response content: {content[:500]}")
                    
                    # Try to extract JSON from markdown code blocks
                    if "```json" in content:
                        json_start = content.find("```json") + 7
                        json_end = content.find("```", json_start)
                        content = content[json_start:json_end].strip()
                        parsed = json.loads(content)
                        if isinstance(parsed, list):
                            return parsed
                        else:
                            return [parsed]
                    
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                logger.error(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def validate_evaluation(self, evaluation):
        """
        Validate that an evaluation has all required fields and valid scores.
        
        Args:
            evaluation: Dict with evaluation results
            
        Returns:
            Tuple (is_valid, cleaned_evaluation)
        """
        required_fields = ['file_name', 'transcription'] + self.metrics
        
        # Check all fields present
        missing_fields = [f for f in required_fields if f not in evaluation]
        if missing_fields:
            logger.warning(f"Missing fields in evaluation: {missing_fields}")
            return False, evaluation
        
        # Validate score ranges (0-100)
        for metric in self.metrics:
            try:
                score = int(evaluation[metric])
                if not 0 <= score <= 100:
                    logger.warning(f"Score out of range for {metric}: {score}")
                    evaluation[metric] = max(0, min(100, score))  # Clamp to valid range
            except (ValueError, TypeError):
                logger.warning(f"Invalid score for {metric}: {evaluation[metric]}")
                return False, evaluation
        
        return True, evaluation
    
    def process_batch(self, batch):
        """
        Process a batch of transcriptions and get quality evaluations.
        
        Args:
            batch: DataFrame slice with transcriptions
            
        Returns:
            List of evaluation results
        """
        # Prepare batch data
        transcriptions_batch = []
        for idx, row in batch.iterrows():
            file_name = str(row['file_name'])
            print(f"  → Evaluating: {file_name}", flush=True)
            transcriptions_batch.append({
                "file_name": file_name,
                "transcription": str(row['transcription'])[:8000]  # Limit length for aggregated transcriptions
            })
        
        # Create prompt
        system_message, user_message = self.create_evaluation_prompt(transcriptions_batch)
        
        # Call API
        evaluations = self.call_gpt_api(system_message, user_message)
        
        if evaluations is None:
            logger.error(f"Failed to get evaluations for batch")
            self.failed_entries.extend(transcriptions_batch)
            return []
        
        # Validate and clean results
        validated_results = []
        for eval_result in evaluations:
            is_valid, cleaned_eval = self.validate_evaluation(eval_result)
            if is_valid:
                validated_results.append(cleaned_eval)
            else:
                logger.warning(f"Invalid evaluation result: {eval_result}")
                self.failed_entries.append(eval_result)
        
        return validated_results
    
    def evaluate_all(self):
        """Evaluate all transcriptions in batches with progress tracking."""
        logger.info(f"Starting quality evaluation of {len(self.df)} segments")
        logger.info(f"Using batch size: {self.batch_size}")
        
        # Process in batches
        num_batches = (len(self.df) + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=len(self.df), desc="Evaluating transcriptions") as pbar:
            for i in range(0, len(self.df), self.batch_size):
                batch = self.df.iloc[i:i+self.batch_size]
                batch_results = self.process_batch(batch)
                self.results.extend(batch_results)
                
                pbar.update(len(batch))
                
                # Rate limiting: small delay between batches
                if i + self.batch_size < len(self.df):
                    time.sleep(0.5)
        
        logger.info(f"Evaluation complete. Successful: {len(self.results)}, Failed: {len(self.failed_entries)}")
        
        if self.failed_entries:
            logger.warning(f"Failed to evaluate {len(self.failed_entries)} entries. Check logs for details.")
    
    def save_results(self):
        """Save evaluation results to CSV."""
        if not self.results:
            logger.error("No results to save!")
            return
        
        # Create output directory if needed
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Ensure proper column order (excluding transcription from output)
        column_order = ['file_name'] + self.metrics
        results_df = results_df[column_order]
        
        # Save to CSV
        results_df.to_csv(self.output_file, index=False, encoding='utf-8')
        logger.info(f"Results saved to {self.output_file}")
        
        # Compute and display statistics
        self.compute_statistics(results_df)
    
    def compute_statistics(self, results_df):
        """Compute and display summary statistics for quality metrics."""
        logger.info("\n" + "="*70)
        logger.info("QUALITY EVALUATION SUMMARY STATISTICS")
        logger.info("="*70)
        
        for metric in self.metrics:
            mean_score = results_df[metric].mean()
            std_score = results_df[metric].std()
            median_score = results_df[metric].median()
            min_score = results_df[metric].min()
            max_score = results_df[metric].max()
            
            logger.info(f"\n{metric}:")
            logger.info(f"  Mean:   {mean_score:.2f}")
            logger.info(f"  Std:    {std_score:.2f}")
            logger.info(f"  Median: {median_score:.2f}")
            logger.info(f"  Range:  {min_score:.0f} - {max_score:.0f}")
        
        logger.info("\n" + "="*70)
        
        # Overall quality interpretation
        avg_fluency = results_df["Fluency / grammaticality"].mean()
        avg_coherence = results_df["Coherence / consistency"].mean()
        avg_completeness = results_df["Completeness"].mean()
        avg_redundancy = results_df["Redundancy"].mean()
        avg_richness = results_df["Lexical richness"].mean()
        
        # Compute composite quality score (redundancy is inverted)
        results_df['Overall Quality'] = (
            avg_fluency + avg_coherence + avg_completeness + 
            (100 - avg_redundancy) + avg_richness
        ) / 5
        
        logger.info(f"\nOverall Quality Score: {results_df['Overall Quality'].mean():.2f}/100")
        logger.info(f"  (Note: Redundancy inverted for composite score)")
        logger.info("="*70 + "\n")
    
    def run(self):
        """Execute the full evaluation pipeline."""
        try:
            # Load data
            self.load_data()
            
            # Evaluate
            self.evaluate_all()
            
            # Save results
            self.save_results()
            
            logger.info("Quality evaluation completed successfully!")
            
        except Exception as e:
            logger.error(f"Evaluation pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Evaluate transcription quality using Azure OpenAI GPT models"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/transcriptions_with_scope_and_aggregated_note.csv",
        help="Input CSV file with transcriptions (default: results/transcriptions_with_scope_and_aggregated_note.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/quality_evaluation_across_recordings.csv",
        help="Output CSV file for quality evaluations (default: results/quality_evaluation_across_recordings.csv)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        choices=["gpt-4o"],  # Only gpt-4o is deployed in this Azure OpenAI resource
        help="Azure OpenAI deployment name (default: gpt-4o)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of transcriptions to evaluate per API call (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Initialize and run evaluator
    evaluator = TranscriptionQualityEvaluator(
        input_file=args.input,
        output_file=args.output,
        model_deployment=args.model,
        batch_size=args.batch_size
    )
    
    evaluator.run()


if __name__ == "__main__":
    main()
