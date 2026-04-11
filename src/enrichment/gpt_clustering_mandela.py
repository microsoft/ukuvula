#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
GPT-Based Clustering Analysis for Nelson Mandela Foundation Archive

This script uses Azure OpenAI GPT models to intelligently classify transcription
segments into predefined, meaningful clusters that reflect different dimensions
of Nelson Mandela's life and legacy.

Usage:
    python gpt_clustering_mandela.py
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
import time
from azure_openai_utils import setup_azure_openai

# Ensure src/ is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPTMandelaClusterAnalyzer:
    """
    Analyzes transcriptions using GPT to classify segments into predefined clusters
    relevant to Nelson Mandela's life and legacy.
    """
    
    def __init__(self, input_file="results/final_transcriptions.csv"):
        self.input_file = Path(input_file)
        self.df = None
        self.predefined_clusters = self.define_mandela_clusters()
        self.client = self.setup_azure_openai()
        
    def setup_azure_openai(self):
        """Setup Azure OpenAI client with authentication."""
        return setup_azure_openai()
        
    def define_mandela_clusters(self):
        """Define predefined clusters relevant to Nelson Mandela's life and legacy."""
        
        clusters = {
            "Early Life and Education": {
                "description": "Discussions about Mandela's childhood, youth, early education, Fort Hare University, family upbringing in Transkei, and formative experiences that shaped his character and worldview.",
                "keywords": ["youth", "childhood", "school", "university", "fort hare", "transkei", "family", "education", "early years"]
            },
            
            "Legal Career and Law Practice": {
                "description": "Content related to Mandela's career as a lawyer, his law firm Mandela & Tambo, legal education, court cases, and his work as an advocate for justice through legal means.",
                "keywords": ["lawyer", "legal", "law firm", "mandela and tambo", "attorney", "court", "legal practice", "advocate"]
            },
            
            "ANC and Political Activism": {
                "description": "Discussions about Mandela's involvement with the African National Congress, political activism, leadership roles, and contributions to the liberation movement and political strategy.",
                "keywords": ["african national congress", "anc", "political activism", "liberation movement", "political leadership", "congress"]
            },
            
            "Armed Struggle and MK": {
                "description": "Content addressing Mandela's role in Umkhonto we Sizwe (MK), the ANC's armed wing, decisions about armed struggle, military training, and sabotage activities.",
                "keywords": ["umkhonto we sizwe", "mk", "armed struggle", "sabotage", "military wing", "armed resistance", "guerrilla"]
            },
            
            "Rivonia Trial and Legal Battles": {
                "description": "Discussions about the famous Rivonia Trial, court proceedings, Mandela's dock speech, legal defense strategies, and the impact of the trial on the liberation struggle.",
                "keywords": ["rivonia trial", "trial", "dock", "courtroom", "legal defense", "sentence", "court case"]
            },
            
            "Prison Life and Robben Island": {
                "description": "Content covering Mandela's 27 years of imprisonment, life on Robben Island, Pollsmoor and Victor Verster prisons, prison conditions, fellow prisoners, and the experience of incarceration.",
                "keywords": ["robben island", "prison", "prisoner", "imprisonment", "pollsmoor", "victor verster", "cell", "warders"]
            },
            
            "Release and Transition to Freedom": {
                "description": "Discussions about Mandela's release from prison in 1990, the immediate aftermath, transition from prisoner to free man, and the symbolic importance of this moment.",
                "keywords": ["release", "freedom", "walk to freedom", "liberation", "free mandela", "coming out"]
            },
            
            "Negotiations and Peace Process": {
                "description": "Content about the negotiation process leading to democratic transition, CODESA talks, bilateral negotiations, and the peaceful transition to democracy.",
                "keywords": ["negotiations", "peace process", "codesa", "democratic transition", "talks", "constitutional negotiations"]
            },
            
            "Presidency and Democratic Leadership": {
                "description": "Discussions about Mandela's tenure as South Africa's first democratically elected president, leadership style, policy decisions, and the Government of National Unity.",
                "keywords": ["president", "presidency", "democratic government", "cabinet", "government of national unity", "leadership"]
            },
            
            "Reconciliation and Truth Commission": {
                "description": "Content addressing Mandela's commitment to reconciliation, the Truth and Reconciliation Commission, forgiveness, healing, and building a unified nation.",
                "keywords": ["reconciliation", "truth and reconciliation", "trc", "forgiveness", "healing", "unity", "rainbow nation"]
            },
            
            "Family and Personal Relationships": {
                "description": "Discussions about Mandela's personal and family life, marriages to Evelyn Mase, Winnie Mandela, and Graça Machel, relationships with children, and impact of political life on family.",
                "keywords": ["family", "wife", "winnie", "graca machel", "evelyn", "children", "marriage", "personal life"]
            },
            
            "International Relations and Global Impact": {
                "description": "Content about Mandela's international relationships, global impact, role as world statesman, relationships with world leaders, and influence on global politics.",
                "keywords": ["international", "global", "world leader", "diplomatic", "foreign policy", "state visits", "united nations"]
            },
            
            "Apartheid System and Racial Oppression": {
                "description": "Discussions about the apartheid system, racial discrimination, oppressive laws, and the systematic nature of racial oppression that motivated the liberation struggle.",
                "keywords": ["apartheid", "racial discrimination", "segregation", "oppression", "pass laws", "bantustans", "racial laws"]
            },
            
            "Philosophy and Values": {
                "description": "Content exploring Mandela's philosophy, values, worldview, beliefs about ubuntu, human dignity, justice, equality, and the philosophical foundations that guided his actions.",
                "keywords": ["philosophy", "values", "ubuntu", "human dignity", "justice", "equality", "beliefs", "principles"]
            },
            
            "Health and Aging": {
                "description": "Discussions about Mandela's health, aging process, medical care, physical challenges, and reflections on mortality, particularly in his later years.",
                "keywords": ["health", "illness", "hospital", "medical", "aging", "old age", "frail", "health condition"]
            },
            
            "Legacy and Historical Significance": {
                "description": "Content about Mandela's legacy, historical significance, commemorative activities, and reflections on how he should be remembered by future generations.",
                "keywords": ["legacy", "memory", "commemoration", "historical", "influence", "impact", "inspiration", "mandela day"]
            },
            
            "Culture and Traditional Heritage": {
                "description": "Discussions about Mandela's connection to Xhosa cultural heritage, African traditions, and the role of culture in his identity and leadership style.",
                "keywords": ["culture", "tradition", "xhosa", "african culture", "traditional", "customs", "cultural heritage"]
            },
            
            "Economic and Social Development": {
                "description": "Content focusing on economic and social development, poverty alleviation, inequality, and efforts to address socio-economic challenges in post-apartheid South Africa.",
                "keywords": ["economy", "development", "poverty", "inequality", "reconstruction", "economic policy", "social development"]
            },
            
            "Interviews and Conversations": {
                "description": "General interviews, media interactions, and recorded conversations that capture Mandela's direct words and thoughts on various topics.",
                "keywords": ["interview", "conversation", "discussion", "media", "journalist", "questions", "dialogue"]
            },
            
            "General and Miscellaneous": {
                "description": "General conversations, discussions, and content that doesn't clearly fit into other specific thematic categories.",
                "keywords": ["general", "conversation", "discussion", "various", "miscellaneous"]
            }
        }
        
        return clusters
    
    def load_data(self, sample_size=None):
        """Load and preprocess the transcription data."""
        logger.info(f"Loading transcription data from {self.input_file}")
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file {self.input_file} not found")
            
        self.df = pd.read_csv(self.input_file)
        logger.info(f"Loaded {len(self.df)} transcription records")
        
        # Basic data cleaning
        self.df['transcription'] = self.df['transcription'].fillna('')
        self.df['transcription'] = self.df['transcription'].astype(str)
        
        # Filter out very short transcriptions
        min_length = 50  # Longer minimum for GPT analysis
        initial_count = len(self.df)
        self.df = self.df[self.df['transcription'].str.len() >= min_length]
        logger.info(f"Filtered to {len(self.df)} records (removed {initial_count - len(self.df)} short transcriptions)")
        
        # Sample data if requested (for testing/development)
        if sample_size and sample_size < len(self.df):
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled {sample_size} records for analysis")
        
        # Reset index to ensure continuous integers
        self.df = self.df.reset_index(drop=True)
    
    def create_classification_prompt(self, text_segment):
        """Create a prompt for GPT to classify a text segment."""
        
        cluster_descriptions = "\n".join([
            f"{i+1}. {name}: {info['description']}"
            for i, (name, info) in enumerate(self.predefined_clusters.items())
        ])
        
        prompt = f"""You are analyzing transcription segments from the Nelson Mandela Foundation archive. These are recordings of conversations with or about Nelson Mandela, covering different aspects of his life and legacy.

AVAILABLE CLUSTERS:
{cluster_descriptions}

TRANSCRIPTION SEGMENT TO CLASSIFY:
"{text_segment}"

TASK: Classify this transcription segment into the MOST APPROPRIATE cluster from the list above.

INSTRUCTIONS:
1. Read the transcription segment carefully
2. Consider the main topics, themes, and context discussed
3. Select the ONE cluster that best represents the primary theme of this segment
4. If the segment covers multiple themes, choose the most prominent one
5. If none of the specific clusters fit well, use "General and Miscellaneous"

RESPONSE FORMAT:
Provide your response as a JSON object with exactly this structure:
{{
    "cluster_name": "Exact cluster name from the list",
    "confidence": 0.95,
    "reasoning": "Brief explanation of why this cluster was chosen"
}}

Your response:"""
        
        return prompt
    
    def classify_segment_with_gpt(self, text_segment, max_retries=3):
        """Use GPT to classify a single text segment."""
        
        prompt = self.create_classification_prompt(text_segment)
        
        for attempt in range(max_retries):
            try:
                messages = [
                    {"role": "system", "content": "You are an expert historian and archivist specializing in Nelson Mandela and South African liberation history."},
                    {"role": "user", "content": prompt}
                ]
                
                completion = self.client.chat.completions.create(
                    model="gpt-4o",  # Updated model name for text analysis
                    messages=messages,
                    max_tokens=200,
                    temperature=0.1,  # Low temperature for consistent classification
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    stream=False
                )
                
                response_text = completion.choices[0].message.content.strip()
                
                # Parse JSON response
                try:
                    # Handle both raw JSON and markdown-wrapped JSON
                    response_text = response_text.strip()
                    
                    # Remove markdown code block if present
                    if response_text.startswith('```json'):
                        response_text = response_text[7:]  # Remove ```json
                    if response_text.startswith('```'):
                        response_text = response_text[3:]  # Remove ```
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]  # Remove trailing ```
                    
                    response_text = response_text.strip()
                    result = json.loads(response_text)
                    
                    # Validate required fields
                    if 'cluster_name' not in result:
                        raise ValueError("Missing cluster_name in response")
                    
                    # Validate cluster name exists
                    if result['cluster_name'] not in self.predefined_clusters:
                        logger.warning(f"Invalid cluster name returned: {result['cluster_name']}")
                        result['cluster_name'] = "General and Miscellaneous"
                    
                    return result
                    
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON response (attempt {attempt + 1}): {response_text}")
                    if attempt == max_retries - 1:
                        return {
                            "cluster_name": "General and Miscellaneous",
                            "confidence": 0.5,
                            "reasoning": "Failed to parse GPT response"
                        }
                    
            except Exception as e:
                logger.warning(f"Error calling GPT API (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return {
                        "cluster_name": "General and Miscellaneous",
                        "confidence": 0.0,
                        "reasoning": f"API error: {str(e)}"
                    }
                time.sleep(2)  # Wait before retry
    
    def classify_all_segments(self, batch_size=200):
        """Classify all transcription segments using GPT."""
        logger.info(f"Starting GPT classification of {len(self.df)} segments...")
        
        cluster_assignments = defaultdict(list)
        classification_results = []
        
        for idx, row in self.df.iterrows():
            if (idx + 1) % batch_size == 0:
                logger.info(f"Processing segment {idx + 1}/{len(self.df)}")
            
            text_segment = row['transcription']
            
            # Truncate very long segments to avoid token limits
            if len(text_segment) > 2000:
                text_segment = text_segment[:2000] + "..."
            
            # Classify with GPT
            classification = self.classify_segment_with_gpt(text_segment)
            
            cluster_name = classification['cluster_name']
            cluster_assignments[cluster_name].append(idx)
            
            classification_results.append({
                'segment_idx': idx,
                'cluster_name': cluster_name,
                'confidence': classification.get('confidence', 0.0),
                'reasoning': classification.get('reasoning', ''),
                'file_name': row['file_name'],
                'collection': row['collection']
            })
            
            # Add small delay to avoid rate limiting (reduced for full processing)
            time.sleep(0.05)
            
            # Save intermediate results every 1000 segments
            if (idx + 1) % 1000 == 0:
                self.save_intermediate_results(classification_results, f"intermediate_results_{idx + 1}.json")
                logger.info(f"Saved intermediate results for {idx + 1} segments")
        
        # Save final intermediate results
        self.save_intermediate_results(classification_results, "final_classification_results.json")
        logger.info(f"GPT classification completed. Assigned segments to {len(cluster_assignments)} clusters.")
        
        return cluster_assignments, classification_results
    
    def save_intermediate_results(self, results, filename):
        """Save intermediate classification results."""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def create_cluster_results(self, cluster_assignments):
        """Create comprehensive cluster analysis results."""
        logger.info("Generating cluster analysis results...")
        
        results = []
        
        for cluster_name, segment_indices in cluster_assignments.items():
            if len(segment_indices) == 0:
                continue
            
            # Get cluster info
            cluster_info = self.predefined_clusters.get(cluster_name, {})
            summary = cluster_info.get('description', f'Cluster containing {len(segment_indices)} segments related to {cluster_name.lower()}.')
            
            # Get records for this cluster
            cluster_records = self.df.iloc[segment_indices]
            num_records = len(cluster_records)
            
            # Create records list with file paths
            records = []
            for _, row in cluster_records.iterrows():
                record_path = f"{row['collection']}/{row['file_name']}"
                # Clean up the path to avoid double collection names
                if record_path.startswith(row['collection'] + '/' + row['collection']):
                    record_path = record_path.replace(row['collection'] + '/', '', 1)
                records.append(record_path)
            
            # Remove duplicates while preserving order
            unique_records = []
            seen = set()
            for record in records:
                if record not in seen:
                    unique_records.append(record)
                    seen.add(record)
            
            results.append({
                'Cluster Name': cluster_name,
                'Summary': summary,
                'Number of Records': num_records,
                'Records': unique_records
            })
            
            logger.info(f"Cluster '{cluster_name}': {num_records} records")
        
        # Sort by number of records (descending)
        results.sort(key=lambda x: x['Number of Records'], reverse=True)
        
        return results
    
    def save_results(self, results, classification_results, output_file="results/gpt_cluster_results.csv"):
        """Save clustering results to CSV and detailed analysis to JSON."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to {output_path}")
        
        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        # Convert Records list to string representation
        df_results['Records'] = df_results['Records'].apply(lambda x: str(x))
        
        # Save to CSV
        df_results.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Saved {len(results)} clusters to {output_path}")
        
        # Save detailed analysis
        analysis_file = output_path.parent / "gpt_cluster_analysis_detailed.json"
        detailed_analysis = {
            'cluster_results': results,
            'classification_details': classification_results,
            'summary_statistics': {
                'total_segments': len(classification_results),
                'total_clusters': len(results),
                'average_confidence': np.mean([r.get('confidence', 0) for r in classification_results])
            }
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved detailed analysis to {analysis_file}")
        
        return output_path
    
    def run_complete_analysis(self, sample_size=None):
        """Run the complete GPT-powered clustering analysis pipeline."""
        logger.info("Starting GPT-powered clustering analysis for Nelson Mandela Foundation archive...")
        
        try:
            # Load and preprocess data
            self.load_data()
            
            # Use all data unless sample_size is specified
            if sample_size and sample_size < len(self.df):
                logger.info(f"Sampling {sample_size} segments from {len(self.df)} total segments")
                sample_df = self.df.sample(n=sample_size, random_state=42)
                # Reset index for consistent indexing
                sample_df = sample_df.reset_index(drop=True)
                # Temporarily set df to sample for processing
                original_df = self.df
                self.df = sample_df
            else:
                logger.info(f"Processing all {len(self.df)} segments")
                sample_df = self.df
            
            # Analyze with GPT
            cluster_assignments, classification_results = self.classify_all_segments()
            
            # Generate results
            results = self.create_cluster_results(cluster_assignments)
            
            # Save results
            output_file = self.save_results(results, classification_results)
            
            logger.info("GPT clustering analysis completed successfully!")
            logger.info(f"Generated {len(results)} thematic clusters")
            logger.info(f"Results saved to: {output_file}")
            
            return results, output_file
            
        except Exception as e:
            logger.error(f"Error during clustering analysis: {e}")
            raise

def main():
    """Main function to run GPT-based clustering analysis."""
    analyzer = GPTMandelaClusterAnalyzer()
    
    # Run analysis on a smaller sample for demonstration/testing
    # For full analysis, change sample_size to None or a larger number
    results, output_file = analyzer.run_complete_analysis(sample_size=None)

    # Print summary
    print(f"\n{'='*70}")
    print("NELSON MANDELA FOUNDATION - GPT-BASED CLUSTERING ANALYSIS")
    print(f"{'='*70}")
    print(f"Total clusters generated: {len(results)}")
    print(f"Output file: {output_file}")
    print(f"{'='*70}")
    
    # Print clusters by size
    print("\nCLUSTERS BY SIZE:")
    for i, result in enumerate(results, 1):
        print(f"{i:2d}. {result['Cluster Name']:50s} ({result['Number of Records']:5d} records)")
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()