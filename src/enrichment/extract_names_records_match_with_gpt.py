#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
GPT-Powered Entity Extraction for Nelson Mandela Foundation Archive

This script uses Azure OpenAI GPT models to intelligently extract and classify
entities (persons, locations, organizations) from transcription data.

Output: results/entities_with_records_gpt.csv with columns:
    - Name/Entity: The extracted entity name
    - Entity Type: PERSON, LOCATION, or ORGANIZATION  
    - Corresponding Records: List of files/segments where entity appears

Usage:
    python extract_names_records_match.py
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
import time
import re
from pathlib import Path
from collections import defaultdict, Counter
from azure_openai_utils import setup_azure_openai
import warnings

# Ensure src/ is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPTEntityExtractor:
    """GPT-powered entity extraction system for the Nelson Mandela Foundation archive."""
    
    def __init__(self, input_file="results/final_transcriptions.csv"):
        self.input_file = Path(input_file)
        self.df = None
        
        # Initialize Azure OpenAI client
        self.initialize_openai_client()
        
        # Entity storage
        self.entities = defaultdict(lambda: {
            'type': None,
            'records': set(),
            'mentions': 0,
            'contexts': []
        })
    
    def initialize_openai_client(self):
        """Initialize Azure OpenAI client with proper authentication."""
        try:
            self.client = setup_azure_openai()
            self.model_name = "gpt-4o"
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise
    
    def load_transcription_data(self):
        """Load transcription data for entity extraction."""
        logger.info(f"Loading transcription data from {self.input_file}")
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"Transcription file {self.input_file} not found")
            
        self.df = pd.read_csv(self.input_file)
        logger.info(f"Loaded {len(self.df)} transcription records")
        
        # Basic data cleaning
        self.df['transcription'] = self.df['transcription'].fillna('')
        self.df['transcription'] = self.df['transcription'].astype(str)
        
        # Filter out very short transcriptions
        min_length = 50
        initial_count = len(self.df)
        self.df = self.df[self.df['transcription'].str.len() >= min_length]
        logger.info(f"Filtered to {len(self.df)} records (removed {initial_count - len(self.df)} short transcriptions)")
    
    def create_entity_extraction_prompt(self, text_segment):
        """Create a specialized prompt for entity extraction."""
        
        prompt = f"""You are an expert historian specializing in Nelson Mandela and South African history. Extract named entities from this transcription segment.

TRANSCRIPTION: "{text_segment}"

Extract all significant named entities and classify them as:
1. PERSON - Individual people (e.g., Nelson Mandela, Desmond Tutu, Winnie Mandela)
2. ORGANIZATION - Groups, institutions, movements (e.g., ANC, UN, Robben Island Museum)
3. LOCATION - Geographic places (e.g., Johannesburg, Robben Island, Soweto)
4. DATE_TIME - Dates, time periods, eras (e.g., July 18, 1990s, apartheid era)
5. EVENT - Significant events, trials, processes (e.g., Rivonia Trial, 1994 Election, Truth and Reconciliation Commission)

Return JSON format:
[{{"entity": "Name", "type": "PERSON|ORGANIZATION|LOCATION|DATE_TIME|EVENT", "context": "brief context"}}]

Focus on proper nouns and significant historical references. Include politicians, activists, places, organizations, dates, and events relevant to South African liberation history.

Your response:"""
        
        return prompt
    
    def extract_entities_from_segment(self, text_segment, max_retries=3):
        """Extract entities using GPT."""
        
        if len(text_segment) > 3000:
            text_segment = text_segment[:3000] + "..."
        
        prompt = self.create_entity_extraction_prompt(text_segment)
        
        for attempt in range(max_retries):
            try:
                messages = [
                    {"role": "system", "content": "You are an expert in South African liberation history. Provide accurate named entity extraction."},
                    {"role": "user", "content": prompt}
                ]
                
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.1,
                    top_p=0.95
                )
                
                response_text = completion.choices[0].message.content.strip()
                
                # Parse JSON response
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].strip()
                
                entities = json.loads(response_text)
                
                if isinstance(entities, list):
                    valid_entities = []
                    for entity in entities:
                        if isinstance(entity, dict) and 'entity' in entity and 'type' in entity:
                            if entity['type'] in ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE_TIME', 'EVENT']:
                                valid_entities.append(entity)
                    return valid_entities
                    
            except Exception as e:
                logger.warning(f"Error in attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        return []

    def process_all_transcriptions(self, batch_size=100):
        """Process all transcription segments to extract entities."""
        logger.info(f"Starting entity extraction from {len(self.df)} segments...")
        
        total_processed = 0
        
        for idx, row in self.df.iterrows():
            if (total_processed + 1) % batch_size == 0:
                logger.info(f"Processed {total_processed + 1}/{len(self.df)} segments...")
            
            text_segment = row['transcription']
            entities = self.extract_entities_from_segment(text_segment)
            
            record_info = f"{row['collection']}/{row['file_name']}"
            
            for entity_info in entities:
                entity_name = entity_info['entity'].strip()
                entity_type = entity_info['type']
                context = entity_info.get('context', '')
                
                # Normalize entity name
                entity_name = self.normalize_entity_name(entity_name)
                
                if entity_name:
                    self.entities[entity_name]['type'] = entity_type
                    self.entities[entity_name]['records'].add(record_info)
                    self.entities[entity_name]['mentions'] += 1
                    self.entities[entity_name]['contexts'].append(context[:100])
            
            total_processed += 1
            time.sleep(0.1)  # Rate limiting
            
            if total_processed % 500 == 0:
                self.save_intermediate_results(f"entities_intermediate_{total_processed}.json")
        
        logger.info(f"Entity extraction completed. Found {len(self.entities)} unique entities.")

    def normalize_entity_name(self, name):
        """Normalize entity names for consistency."""
        name = name.strip()
        name = re.sub(r'\s+', ' ', name)
        name = re.sub(r'^["\'\[\(]|["\'\]\)]$', '', name)
        
        if len(name) < 2 or not re.search(r'[A-Za-z]', name):
            return None
            
        return name

    def save_intermediate_results(self, filename):
        """Save intermediate extraction results."""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        serializable_entities = {}
        for entity, info in self.entities.items():
            serializable_entities[entity] = {
                'type': info['type'],
                'records': list(info['records']),
                'mentions': info['mentions'],
                'contexts': info['contexts'][:5]
            }
        
        filepath = results_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_entities, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Intermediate results saved to {filepath}")

    def create_final_results(self, min_mentions=2):
        """Create final results CSV with entity information."""
        logger.info("Creating final entity extraction results...")
        
        results = []
        
        for entity_name, info in self.entities.items():
            if info['mentions'] >= min_mentions or 'mandela' in entity_name.lower():
                records_list = sorted(list(info['records']))
                
                if len(records_list) == 1:
                    records_str = records_list[0]
                else:
                    records_str = str(records_list)
                
                results.append({
                    'Name/Entity': entity_name,
                    'Entity Type': info['type'],
                    'Corresponding Records': records_str,
                    'Mention Count': info['mentions'],
                    'Number of Files': len(info['records'])
                })
        
        results.sort(key=lambda x: x['Mention Count'], reverse=True)
        return results

    def save_results(self, results, output_file="results/entities_with_records_gpt.csv"):
        """Save entity extraction results to CSV."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to {output_path}")
        
        df_results = pd.DataFrame(results)
        column_order = ['Name/Entity', 'Entity Type', 'Corresponding Records']
        
        if 'Mention Count' in df_results.columns:
            column_order.extend(['Mention Count', 'Number of Files'])
        
        df_results = df_results[column_order]
        df_results.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Saved {len(results)} entities to {output_path}")
        
        # Print summary
        type_counts = df_results['Entity Type'].value_counts()
        logger.info("Entity summary by type:")
        for entity_type, count in type_counts.items():
            logger.info(f"  {entity_type}: {count} entities")
        
        # Save detailed analysis
        analysis_file = output_path.parent / "entity_extraction_analysis_gpt.json"
        analysis_data = {
            'total_entities': len(results),
            'entity_type_counts': type_counts.to_dict(),
            'top_entities_by_mentions': results[:20],
            'extraction_date': pd.Timestamp.now().isoformat()
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        return output_path

    def run_complete_extraction(self, min_mentions=1):
        """Run the complete entity extraction pipeline."""
        logger.info("Starting GPT-powered entity extraction...")
        
        try:
            self.load_transcription_data()
            self.process_all_transcriptions()
            results = self.create_final_results(min_mentions=min_mentions)
            output_file = self.save_results(results)
            
            logger.info("Entity extraction completed successfully!")
            logger.info(f"Extracted {len(results)} entities")
            logger.info(f"Results saved to: {output_file}")
            
            return results, output_file
            
        except Exception as e:
            logger.error(f"Error during entity extraction: {e}")
            raise

def main():
    """Main function to run entity extraction."""
    extractor = GPTEntityExtractor()
    results, output_file = extractor.run_complete_extraction()
    
    print(f"\n{'='*70}")
    print("NELSON MANDELA FOUNDATION - GPT ENTITY EXTRACTION")
    print(f"{'='*70}")
    print(f"Total entities extracted: {len(results)}")
    print(f"Output file: {output_file}")
    print(f"{'='*70}")
    
    # Print top entities by type
    df = pd.DataFrame(results)
    print("\nTOP ENTITIES BY TYPE:")
    
    for entity_type in ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE_TIME', 'EVENT']:
        type_entities = df[df['Entity Type'] == entity_type].head(10)
        if not type_entities.empty:
            type_label = entity_type.replace('_', '/') + "S" if entity_type != 'DATE_TIME' else 'DATES/TIMES'
            print(f"\n{type_label}:")
            for i, (_, row) in enumerate(type_entities.iterrows(), 1):
                print(f"  {i:2d}. {row['Name/Entity']:40s} ({row['Mention Count']:3d} mentions)")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
