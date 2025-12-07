#!/usr/bin/env python3
"""
Optimized multi-turn CIR data filtering system.
Efficient filtering utilizing existing JSON and CSV files.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import torch
from torch.nn.functional import normalize
import hashlib
from collections import defaultdict
import random

# CLIP availability check
CLIP_AVAILABLE = False
try:
    import clip
    from PIL import Image
    CLIP_AVAILABLE = True
except ImportError:
    print("Warning: CLIP not available. CLIP similarity filtering will be disabled.")

class OptimizedMultiTurnFilter:
    """Optimized multi-turn CIR filtering system utilizing existing data"""
    
    def __init__(self, similarity_threshold: float = 0.8, rank_margin: int = 30):
        """
        Args:
            similarity_threshold: CLIP similarity threshold
            rank_margin: Rank margin threshold
        """
        self.similarity_threshold = similarity_threshold
        self.rank_margin = rank_margin
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.clip_available = CLIP_AVAILABLE
        
        # CLIP initialization
        if self.clip_available:
            try:
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
                print(f"CLIP model loaded on {self.device}")
            except Exception as e:
                print(f"Failed to load CLIP: {e}")
                self.clip_available = False
        
        # Image hash mapping
        self.image_id_to_hash = {}
    
    def _compute_image_hash(self, image_path: str) -> str:
        """Compute MD5 hash of image file"""
        try:
            with open(image_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            print(f"Warning: Failed to compute hash for {image_path}: {e}")
            return None
    
    def _initialize_fashioniq_hash_mapping(self, dataset_name: str):
        """Initialize hash mapping for FashionIQ images"""
        category = dataset_name.split('_')[1] if '_' in dataset_name else 'dress'
        image_dir = f'fashion-iq/images'

        # Get image IDs from split files
        splits = ['train', 'val', 'test']
        for split in splits:
            split_file = f'fashion-iq/image_splits/split.{category}.{split}.json'
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    image_ids = json.load(f)
                
                for image_id in image_ids:
                    image_path = os.path.join(image_dir, category, f"{image_id}.jpg")
                    
                    if os.path.exists(image_path):
                        img_hash = self._compute_image_hash(image_path)
                        if img_hash:
                            self.image_id_to_hash[image_id] = img_hash
                            self.image_id_to_hash[f"{image_id}.jpg"] = img_hash

    def _initialize_cirr_hash_mapping(self, dataset_name: str):
        """Initialize hash mapping for CIRR images"""
        image_dir = 'cirr/img_raw'

        # Get image IDs from split files
        splits = ['train', 'val', 'test1']
        for split in splits:
            split_file = f'cirr/image_splits/split.rc2.{split}.json'
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    split_data = json.load(f)
                
                for image_id, relative_path in split_data.items():
                    # relative_path is in format like "./train/34/train-11041-2-img0.png"
                    full_path = os.path.join(image_dir, relative_path.lstrip('./'))
                    
                    if os.path.exists(full_path):
                        img_hash = self._compute_image_hash(full_path)
                        if img_hash:
                            self.image_id_to_hash[image_id] = img_hash

    def _initialize_circo_hash_mapping(self, dataset_name: str):
        """Initialize hash mapping for CIRCO images"""
        try:
            import torch
            
            # Load existing hash mapping from CIRCO metadata file
            metadata_file = 'CIRCO/metadata_blip.pt'
            if not os.path.exists(metadata_file):
                print(f"Warning: CIRCO metadata file not found: {metadata_file}")
                return
            
            metadata = torch.load(metadata_file, map_location='cpu', weights_only=False)
            hash_to_idx = metadata['hash_to_idx']
            idx_to_info = metadata['idx_to_info']
            
            # Create hash to filename mapping
            hash_to_filename = {}
            for hash_val, idx in hash_to_idx.items():
                info = idx_to_info.get(idx, {})
                image_id = info.get('image_id', '')
                if image_id:
                    filename = f"{int(image_id):012d}.jpg"
                    hash_to_filename[hash_val] = filename
            
            # Create image ID to hash mapping
            for hash_val, filename in hash_to_filename.items():
                # Extract image ID from filename (e.g., 000000000001.jpg → 1)
                image_id = filename.replace('.jpg', '').lstrip('0') or '0'
                self.image_id_to_hash[image_id] = hash_val
                # Also add 12-digit format
                padded_id = f"{int(image_id):012d}"
                self.image_id_to_hash[padded_id] = hash_val
                
        except ImportError:
            print("Warning: PyTorch not available, cannot load CIRCO metadata")
        except Exception as e:
            print(f"Warning: Failed to load CIRCO metadata: {e}")

    def get_image_hash(self, image_id: str) -> str:
        """Get hash value from image ID"""
        if not image_id:
            return None
            
        # Try direct match
        if image_id in self.image_id_to_hash:
            return self.image_id_to_hash[image_id]
        
        # Consider with/without extension
        if image_id.endswith('.jpg'):
            base_id = image_id[:-4]
            if base_id in self.image_id_to_hash:
                return self.image_id_to_hash[base_id]
        else:
            jpg_id = f"{image_id}.jpg"
            if jpg_id in self.image_id_to_hash:
                return self.image_id_to_hash[jpg_id]
        
        # For CIRCO, also try 12-digit format
        if hasattr(self, 'dataset_name') and self.dataset_name == 'circo':
            try:
                padded_id = f"{int(image_id):012d}"
                if padded_id in self.image_id_to_hash:
                    return self.image_id_to_hash[padded_id]
            except ValueError:
                pass
        
        # For FashionIQ, try additional patterns
        if hasattr(self, 'dataset_name') and self.dataset_name.startswith('fashioniq'):
            # Try formats like B00006M009.jpg
            variations = [
                image_id,
                f"{image_id}.jpg",
                image_id.replace('.jpg', '') if image_id.endswith('.jpg') else f"{image_id}.jpg"
            ]
            
            for variation in variations:
                if variation in self.image_id_to_hash:
                    return self.image_id_to_hash[variation]
        
        # For debugging (display only first few)
        if not hasattr(self, '_hash_miss_count'):
            self._hash_miss_count = 0
        
        self._hash_miss_count += 1
        if self._hash_miss_count <= 5:
            print(f"Debug: Hash not found for image_id: '{image_id}' (miss #{self._hash_miss_count})")
            if self._hash_miss_count == 5:
                print("Debug: Suppressing further hash miss messages...")
        
        return None
    
    def load_dataset_files(self, dataset_name: str, input_dir: str = '.') -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
        """Load dataset files (also works with JSON only)"""
        # File paths
        json_file = os.path.join(input_dir, f"multiturn_cir_results_{dataset_name}.json")
        summary_file = os.path.join(input_dir, f"multiturn_cir_summary_{dataset_name}.csv")
        detailed_file = os.path.join(input_dir, f"multiturn_cir_detailed_rankings_{dataset_name}.csv")

        # Check if JSON file exists
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Required JSON file not found: {json_file}")
        
        # Load data
        print(f"Loading {dataset_name} data...")
        
        # JSON results
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        # Check if CSV files exist
        csv_files_exist = os.path.exists(summary_file) and os.path.exists(detailed_file)
        
        if csv_files_exist:
            # If CSV files exist (FashionIQ etc.)
            summary_df = pd.read_csv(summary_file)
            detailed_df = pd.read_csv(detailed_file)
            print(f"Loaded: {len(json_data['results'])} results, {len(summary_df)} summary rows, {len(detailed_df)} detailed rows")
        else:
            # If CSV files don't exist (CIRCO, CIRR etc.) - generate from JSON
            print(f"CSV files not found. Generating from JSON data...")
            summary_df, detailed_df = self._generate_dataframes_from_json(json_data)
            print(f"Loaded: {len(json_data['results'])} results, {len(summary_df)} summary rows (generated), {len(detailed_df)} detailed rows (generated)")
        
        return json_data, summary_df, detailed_df
    
    def _generate_dataframes_from_json(self, json_data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate CSV-equivalent DataFrames from JSON data"""
        summary_data = []
        detailed_data = []
        
        for result in json_data['results']:
            query_id = int(result['query_id'])
            success = result.get('success', False)
            success_turn = result.get('success_turn', -1) if success else -1
            turns = result.get('turns', [])
            total_turns = len(turns)
            
            # Calculate improvement from first and last ranks
            initial_rank = turns[0].get('best_gt_rank', float('inf')) if turns else float('inf')
            final_rank = turns[-1].get('best_gt_rank', float('inf')) if turns else float('inf')
            rank_improvement = initial_rank - final_rank if initial_rank != float('inf') and final_rank != float('inf') else 0
            
            # Summary data
            summary_data.append({
                'query_id': query_id,
                'success': success,
                'success_turn': success_turn,
                'total_turns': total_turns,
                'gt_rank_improvement': rank_improvement,
                'initial_rank': initial_rank,
                'final_rank': final_rank
            })
            
            # Detailed data (each turn)
            for turn_data in turns:
                detailed_data.append({
                    'query_id': query_id,
                    'turn': turn_data.get('turn', 0),
                    'best_gt_rank': turn_data.get('best_gt_rank', float('inf')),
                    'query_text': turn_data.get('query_text', ''),
                    'relative_caption': turn_data.get('relative_caption', ''),
                    'selected_image': turn_data.get('selected_image', '')
                })
        
        summary_df = pd.DataFrame(summary_data)
        detailed_df = pd.DataFrame(detailed_data)
        
        return summary_df, detailed_df
    
    def get_clip_text_feature(self, text: str) -> np.ndarray:
        """Get CLIP text features"""
        if not self.clip_available:
            return np.random.rand(512)  # Dummy features
        
        with torch.no_grad():
            tokens = clip.tokenize(text, truncate=True).to(self.device)
            feat = self.clip_model.encode_text(tokens)
            return normalize(feat, dim=-1).squeeze(0).cpu().numpy()
    
    def apply_success_filter(self, json_data: Dict, summary_df: pd.DataFrame) -> List[int]:
        """Apply success filter"""
        # Identify successful queries directly from JSON data
        successful_query_ids = []
        
        for result in json_data['results']:
            try:
                query_id = int(result['query_id'])
                
                # Check success field at result level
                if result.get('success', False):
                    successful_query_ids.append(query_id)
                    
            except (ValueError, TypeError):
                continue
        
        return successful_query_ids

    def apply_multiturn_filter(self, json_data: Dict, query_ids: List[int]) -> List[int]:
        """Apply multi-turn filter (only those that succeeded at turn 1 or later)"""
        # Convert JSON results to dictionary
        results_dict = {}
        for result in json_data['results']:
            try:
                query_id = int(result['query_id'])
                results_dict[query_id] = result
            except (ValueError, TypeError):
                continue
        
        multiturn_query_ids = []
        
        for query_id in query_ids:
            if query_id not in results_dict:
                continue
            
            result = results_dict[query_id]
            success_turn = result.get('success_turn', 0)
            
            # Multi-turn condition (success at turn 1 or later)
            if success_turn > 0:
                multiturn_query_ids.append(query_id)
        
        return multiturn_query_ids

    def apply_original_query_filter(self, json_data: Dict, query_ids: List[int]) -> List[int]:
        """Apply original query filter (exclude queries with empty original_query)"""
        # Convert JSON results to dictionary
        results_dict = {}
        for result in json_data['results']:
            try:
                query_id = int(result['query_id'])
                results_dict[query_id] = result
            except (ValueError, TypeError):
                continue
        
        valid_query_ids = []
        empty_query_count = 0
        
        for query_id in query_ids:
            if query_id not in results_dict:
                continue
            
            result = results_dict[query_id]
            original_query = result.get('original_query', '')
            
            # Keep only if original_query exists and is not empty
            if original_query and original_query.strip():
                valid_query_ids.append(query_id)
            else:
                empty_query_count += 1
        
        print(f"Original query filter: {empty_query_count} queries removed for having empty original_query")
        return valid_query_ids

    def apply_rank_margin_filter(self, json_data: Dict, detailed_df: pd.DataFrame, query_ids: List[int]) -> List[int]:
        """Apply rank margin filter (exclude if rank degraded by 30+ from previous turn)"""

        # Collect JSON query IDs converted to integers
        json_query_ids = set()
        for result in json_data['results']:
            try:
                query_id = int(result['query_id'])
                json_query_ids.add(query_id)
            except (ValueError, TypeError):
                continue
        
        csv_query_ids = set(detailed_df['query_id'].tolist())
        
        # Only target common query IDs
        valid_query_ids = [qid for qid in query_ids if qid in json_query_ids and qid in csv_query_ids]
        
        print(f"Debug: JSON IDs: {len(json_query_ids)}, CSV IDs: {len(csv_query_ids)}, Valid IDs: {len(valid_query_ids)}")
        
        if len(valid_query_ids) == 0:
            print("Warning: No common query IDs found between JSON and CSV data - skipping rank margin filter")
            return query_ids  # Skip rank margin filter
        
        filtered_query_ids = []
        
        for query_id in tqdm(valid_query_ids, desc="Rank margin filtering"):
            # Get ranking data for the query
            query_ranks = detailed_df[detailed_df['query_id'] == query_id].sort_values('turn')
            
            if len(query_ranks) <= 1:
                # Exclude single-turn cases
                continue
            
            ranks = query_ranks['best_gt_rank'].tolist()
            
            # Process only valid ranks, excluding float('inf')
            valid_ranks = [r for r in ranks if r != float('inf')]
            if len(valid_ranks) <= 1:
                # Skip if there's one or fewer valid ranks
                continue
            
            keep = True
            
            # Check rank margin between consecutive turns
            for i in range(1, len(valid_ranks)):
                # If rank degraded by 30 or more (numerical value increased)
                if valid_ranks[i] > valid_ranks[i-1] + self.rank_margin:
                    keep = False
                    break
            
            if keep:
                filtered_query_ids.append(query_id)
        
        # Include query IDs that were in original query_ids but not in CSV
        missing_ids = [qid for qid in query_ids if qid not in csv_query_ids]
        filtered_query_ids.extend(missing_ids)
        
        return filtered_query_ids
    
    def apply_clip_similarity_filter(self, json_data: Dict, query_ids: List[int]) -> List[int]:
        """Apply CLIP similarity filter"""
        if not self.clip_available:
            print("CLIP not available, skipping similarity filter")
            return query_ids
        
        # Convert JSON results to dictionary (using integer IDs as keys)
        results_dict = {}
        for result in json_data['results']:
            try:
                query_id = int(result['query_id'])
                results_dict[query_id] = result
            except (ValueError, TypeError):
                continue
        
        filtered_query_ids = []
        
        for query_id in tqdm(query_ids, desc="CLIP similarity filtering"):
            if query_id not in results_dict:
                # Skip query IDs not found in JSON
                continue
            
            result = results_dict[query_id]
            turns = result.get('turns', [])
            
            if len(turns) <= 1:
                # Single-turn is not subject to similarity filter
                filtered_query_ids.append(query_id)
                continue
            
            # Check modification text similarity
            past_features = []
            keep = True
            
            for i, turn in enumerate(turns):
                # Get modification text
                if i == 0:
                    modification_text = result.get('original_query', '')
                else:
                    modification_text = turn.get('relative_caption', '')
                
                if not modification_text:
                    continue
                
                feat = self.get_clip_text_feature(modification_text)
                
                # Check similarity with past features
                for prev_feat in past_features:
                    similarity = np.dot(feat, prev_feat)
                    if similarity >= self.similarity_threshold:
                        keep = False
                        break
                
                if not keep:
                    break
                
                past_features.append(feat)
            
            if keep:
                filtered_query_ids.append(query_id)
        
        return filtered_query_ids
    
    def apply_image_duplication_filter(self, json_data: Dict, query_ids: List[int]) -> List[int]:
        """Apply image duplication filter (exclude queries that select the same image multiple times within a dialog, and queries where selected_image duplicates reference_image)"""
        
        # Convert JSON results to dictionary
        results_dict = {}
        for result in json_data['results']:
            try:
                query_id = int(result['query_id'])
                results_dict[query_id] = result
            except (ValueError, TypeError):
                continue

        # For collecting statistics
        hash_success_count = 0
        hash_failure_count = 0
        total_images_processed = 0
        
        filtered_query_ids = []
        duplicate_within_dialog_count = 0
        reference_selected_duplicate_count = 0
        duplicate_examples = []
        reference_selected_examples = []
        
        print("Checking for image duplication within individual dialogs and reference-selected duplications...")
        for query_id in tqdm(query_ids, desc="Processing queries"):
            if query_id not in results_dict:
                # Skip if query not found
                continue
            
            result = results_dict[query_id]
            turns = result.get('turns', [])
            reference_image_id = result.get('reference_image_id', '')
            
            # Get hash of reference_image
            reference_hash = None
            if reference_image_id:
                reference_hash = self.get_image_hash(reference_image_id)
                if reference_hash:
                    hash_success_count += 1
                else:
                    hash_failure_count += 1
                total_images_processed += 1
            
            # Track hashes of images selected in this dialog
            dialog_image_hashes = []
            dialog_images_info = []  # For debugging
            has_duplicate_in_dialog = False
            has_reference_selected_duplicate = False
            
            for turn_data in turns:
                turn_num = turn_data.get('turn', 0)
                selected_image = turn_data.get('selected_image')
                
                if selected_image:
                    total_images_processed += 1
                    image_hash = self.get_image_hash(selected_image)
                    
                    if image_hash:
                        hash_success_count += 1
                        
                        # Check for duplication between reference_image and selected_image
                        if reference_hash and image_hash == reference_hash:
                            has_reference_selected_duplicate = True
                            # Record duplication examples (first few only)
                            if len(reference_selected_examples) < 5:
                                reference_selected_examples.append({
                                    'query_id': query_id,
                                    'reference_image': reference_image_id,
                                    'selected_image': selected_image,
                                    'turn': turn_num,
                                    'image_hash': image_hash[:8] + '...'
                                })
                            break
                        
                        # Check if an image with the same hash was already selected in this dialog
                        if image_hash in dialog_image_hashes:
                            has_duplicate_in_dialog = True
                            # Record duplication examples (first few only)
                            if len(duplicate_examples) < 5:
                                duplicate_turn = dialog_image_hashes.index(image_hash)
                                duplicate_examples.append({
                                    'query_id': query_id,
                                    'image_id': selected_image,
                                    'image_hash': image_hash[:8] + '...',
                                    'duplicate_turns': [duplicate_turn, turn_num],
                                    'dialog_info': dialog_images_info + [(turn_num, selected_image, image_hash[:8] + '...')]
                                })
                            break
                        else:
                            dialog_image_hashes.append(image_hash)
                            dialog_images_info.append((turn_num, selected_image, image_hash[:8] + '...'))
                    else:
                        hash_failure_count += 1
                        # Skip duplication check if hash cannot be obtained
                        dialog_images_info.append((turn_num, selected_image, 'NO_HASH'))
            
            # Keep only if there's no duplication within dialog and no reference-selected duplication
            if not has_duplicate_in_dialog and not has_reference_selected_duplicate:
                filtered_query_ids.append(query_id)
            else:
                if has_duplicate_in_dialog:
                    duplicate_within_dialog_count += 1
                if has_reference_selected_duplicate:
                    reference_selected_duplicate_count += 1
        
        # Display statistics
        print(f"Hash statistics: {hash_success_count} success, {hash_failure_count} failures out of {total_images_processed} total images")
        print(f"Image duplication filter: {duplicate_within_dialog_count} queries removed for selecting duplicate images within same dialog")
        print(f"Reference-Selected duplication filter: {reference_selected_duplicate_count} queries removed for selecting reference image as selected image")
        
        # Display duplication examples
        if duplicate_examples:
            print(f"Examples of within-dialog duplications:")
            for i, example in enumerate(duplicate_examples):
                print(f"  Query {example['query_id']}: Image {example['image_id']} (hash: {example['image_hash']}) selected in turns {example['duplicate_turns']}")
                print(f"    Full dialog: {example['dialog_info']}")
        
        if reference_selected_examples:
            print(f"Examples of reference-selected duplications:")
            for i, example in enumerate(reference_selected_examples):
                print(f"  Query {example['query_id']}: Reference image {example['reference_image']} == Selected image {example['selected_image']} at turn {example['turn']} (hash: {example['image_hash']})")
        
        return filtered_query_ids
    
    def create_filtered_dataset(self, json_data: Dict, filtered_query_ids: List[int]) -> Dict:
        """Create filtered dataset (preserving original dialog structure)"""
        filtered_results = []
        filtered_query_ids_set = set(filtered_query_ids)  # Convert to set for faster lookup
        
        # Filter directly from original results list
        for result in json_data['results']:
            try:
                # Get query_id as integer
                query_id = int(result['query_id'])
                # Check if query ID is in filtered query IDs
                if query_id in filtered_query_ids_set:
                    # Check success_turn individually even for same query ID
                    success_turn = result.get('success_turn', 0)
                    if success_turn > 0:  # Re-apply multi-turn filter condition
                        filtered_results.append(result)
                    # Exclude if success_turn == 0 (with log output)
                    elif success_turn == 0:
                        print(f"Debug: Excluding duplicate entry with success_turn=0 for query_id {query_id}")
            except (ValueError, TypeError):
                # Output warning and skip if query_id conversion fails
                print(f"Warning: Invalid query_id in create_filtered_dataset: {result.get('query_id', 'None')}")
                continue
        
        # Create filtered dataset while preserving original data structure
        filtered_data = {
            'dataset_name': json_data.get('dataset_name', ''),
            'config': json_data.get('config', {}),
            'data_completeness': json_data.get('data_completeness', {}),
            'results': filtered_results,
            'filtering_info': {
                'original_count': len(json_data['results']),
                'filtered_count': len(filtered_results),
                'filtering_rate': len(filtered_results) / len(json_data['results']) * 100 if len(json_data['results']) > 0 else 0,
                'similarity_threshold': self.similarity_threshold,
                'rank_margin': self.rank_margin,
                'applied_filters': []  # Add applied filter info later
            }
        }
        
        return filtered_data
    
    def analyze_filtering_effects(self, summary_df: pd.DataFrame, filtered_query_ids: List[int]) -> Dict:
        """Analyze filtering effects"""
        original_df = summary_df
        filtered_df = summary_df[summary_df['query_id'].isin(filtered_query_ids)]
        
        analysis = {
            'original_stats': {
                'total_queries': len(original_df),
                'success_queries': len(original_df[original_df['success'] == True]),
                'success_rate': len(original_df[original_df['success'] == True]) / len(original_df) * 100,
                'avg_turns': original_df['total_turns'].mean(),
                'avg_rank_improvement': original_df[original_df['success'] == True]['gt_rank_improvement'].mean()
            },
            'filtered_stats': {
                'total_queries': len(filtered_df),
                'success_queries': len(filtered_df[filtered_df['success'] == True]),
                'success_rate': len(filtered_df[filtered_df['success'] == True]) / len(filtered_df) * 100 if len(filtered_df) > 0 else 0,
                'avg_turns': filtered_df['total_turns'].mean() if len(filtered_df) > 0 else 0,
                'avg_rank_improvement': filtered_df[filtered_df['success'] == True]['gt_rank_improvement'].mean() if len(filtered_df[filtered_df['success'] == True]) > 0 else 0
            },
            'filtering_rate': len(filtered_df) / len(original_df) * 100 if len(original_df) > 0 else 0
        }
        
        return analysis
    
    def filter_dataset(self, dataset_name: str, apply_success: bool = True,
                       apply_multiturn: bool = True,
                       apply_original_query: bool = True,
                       apply_rank_margin: bool = True, apply_clip_similarity: bool = True,
                       apply_image_duplication: bool = True,
                       save_filtered: bool = True,
                       input_dir: str = '.', output_dir: str = '.') -> Dict:
        """Execute dataset filtering"""

        print(f"Starting filtering for dataset: {dataset_name}")

        # Save dataset name and directories
        self.dataset_name = dataset_name
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Initialize image hash mapping
        if apply_image_duplication:
            print("Initializing image hash mapping...")
            if dataset_name.startswith('fashioniq'):
                self._initialize_fashioniq_hash_mapping(dataset_name)
                print(f"Initialized hash mapping for {len(self.image_id_to_hash)} images")
            elif dataset_name.startswith('cirr'):
                self._initialize_cirr_hash_mapping(dataset_name)
                print(f"Initialized hash mapping for {len(self.image_id_to_hash)} images")
            elif dataset_name == 'circo':
                self._initialize_circo_hash_mapping(dataset_name)
                print(f"Initialized hash mapping for {len(self.image_id_to_hash)} images")
            else:
                print("Warning: Image duplication filter only supports FashionIQ, CIRR, or CIRCO dataset currently")
                apply_image_duplication = False
        
        print(f"=== Filtering {dataset_name} ===")
        print(f"Loading {dataset_name} data...")

        # Load data
        json_data, summary_df, detailed_df = self.load_dataset_files(dataset_name, input_dir=input_dir)
        print(f"Loaded: {len(json_data['results'])} results, {len(summary_df)} summary rows, {len(detailed_df)} detailed rows")
        
        # Record filtering stages
        filtering_stages = []
        
        # Initial state - get actual query_ids
        initial_count = len(json_data['results'])
        current_query_ids = []
        for result in json_data['results']:
            try:
                query_id = int(result['query_id'])
                current_query_ids.append(query_id)
            except (ValueError, TypeError):
                # Output warning and skip if query_id conversion fails
                print(f"Warning: Invalid query_id found: {result.get('query_id', 'None')}")
                continue
        
        print(f"Initial queries: {len(current_query_ids)} (valid query_ids extracted)")
        
        # 1. Success filter
        if apply_success:
            print("Applying success filter...")
            successful_query_ids = self.apply_success_filter(json_data, summary_df)
            filtering_stages.append({
                'name': 'Success Filter',
                'input_count': len(current_query_ids),
                'output_count': len(successful_query_ids),
                'description': 'Filter queries where success == True'
            })
            current_query_ids = successful_query_ids
            print(f"Success filter: {len(current_query_ids)}/{initial_count} ({len(current_query_ids)/initial_count*100:.1f}%)")
        else:
            successful_query_ids = current_query_ids
        
        # 2. Multi-turn filter
        if apply_multiturn and len(current_query_ids) > 0:
            print("Applying multiturn filter...")
            pre_multiturn_count = len(current_query_ids)
            current_query_ids = self.apply_multiturn_filter(json_data, current_query_ids)
            filtering_stages.append({
                'name': 'Multiturn Filter',
                'input_count': pre_multiturn_count,
                'output_count': len(current_query_ids),
                'description': 'Filter queries with success_turn > 0'
            })
            print(f"Multiturn filter: {len(current_query_ids)}/{len(successful_query_ids) if apply_success else initial_count} ({len(current_query_ids)/(len(successful_query_ids) if apply_success else initial_count)*100:.1f}%)")
        
        # 3. Original query filter
        if apply_original_query and len(current_query_ids) > 0:
            print("Applying original query filter...")
            pre_original_count = len(current_query_ids)
            current_query_ids = self.apply_original_query_filter(json_data, current_query_ids)
            filtering_stages.append({
                'name': 'Original Query Filter',
                'input_count': pre_original_count,
                'output_count': len(current_query_ids),
                'description': 'Filter queries with empty original_query'
            })
            print(f"Original query filter: {len(current_query_ids)}/{len(successful_query_ids) if apply_success else initial_count} ({len(current_query_ids)/(len(successful_query_ids) if apply_success else initial_count)*100:.1f}%)")
        
        # 4. Rank margin filter
        if apply_rank_margin and len(current_query_ids) > 0:
            print("Applying rank margin filter...")
            pre_rank_count = len(current_query_ids)
            current_query_ids = self.apply_rank_margin_filter(json_data, detailed_df, current_query_ids)
            filtering_stages.append({
                'name': 'Rank Margin Filter',
                'input_count': pre_rank_count,
                'output_count': len(current_query_ids),
                'description': f'Filter queries with rank degradation > {self.rank_margin}'
            })
            print(f"Rank margin filter: {len(current_query_ids)}/{len(successful_query_ids) if apply_success else initial_count} ({len(current_query_ids)/(len(successful_query_ids) if apply_success else initial_count)*100:.1f}%)")
        
        # 5. CLIP similarity filter
        if apply_clip_similarity and len(current_query_ids) > 0:
            print("Applying CLIP similarity filter...")
            pre_clip_count = len(current_query_ids)
            current_query_ids = self.apply_clip_similarity_filter(json_data, current_query_ids)
            filtering_stages.append({
                'name': 'CLIP Similarity Filter',
                'input_count': pre_clip_count,
                'output_count': len(current_query_ids),
                'description': f'Filter queries with text similarity > {self.similarity_threshold}'
            })
            print(f"CLIP similarity filter: {len(current_query_ids)}/{len(successful_query_ids) if apply_success else initial_count} ({len(current_query_ids)/(len(successful_query_ids) if apply_success else initial_count)*100:.1f}%)")
        
        # 6. Image duplication filter
        if apply_image_duplication and len(current_query_ids) > 0:
            print("Applying image duplication filter...")
            pre_duplication_count = len(current_query_ids)
            current_query_ids = self.apply_image_duplication_filter(json_data, current_query_ids)
            filtering_stages.append({
                'name': 'Image Duplication Filter',
                'input_count': pre_duplication_count,
                'output_count': len(current_query_ids),
                'description': 'Filter queries selecting duplicate images within same dialog'
            })
            print(f"Image duplication filter: {len(current_query_ids)}/{len(successful_query_ids) if apply_success else initial_count} ({len(current_query_ids)/(len(successful_query_ids) if apply_success else initial_count)*100:.1f}%)")
        
        # Create filtered dataset
        filtered_data = self.create_filtered_dataset(json_data, current_query_ids)
        
        # Add applied filter information
        applied_filters = []
        for stage in filtering_stages:
            applied_filters.append({
                'name': stage['name'],
                'description': stage['description']
            })
        filtered_data['filtering_info']['applied_filters'] = applied_filters
        
        # Generate detailed report
        detailed_report = self.generate_detailed_report(
            dataset_name, json_data, filtering_stages, filtered_data
        )
        
        # Save report
        report_filename = os.path.join(output_dir, f"filtering_report_{dataset_name}.json")
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(detailed_report, f, indent=2, ensure_ascii=False)
        print(f"Detailed report saved to: {report_filename}")

        if save_filtered:
            # Save filtered data
            output_filename = os.path.join(output_dir, f"filtered_multiturn_cir_{dataset_name}.json")
            with open(output_filename, 'w') as f:
                json.dump(filtered_data, f, indent=2)
            print(f"Filtered dataset saved to: {output_filename}")
        
        # Display simple statistics
        self.print_simple_statistics(detailed_report)
        
        return filtered_data

    def print_simple_statistics(self, report: Dict):
        """Display simple statistics"""
        dataset_name = report['dataset_info']['dataset_name']
        
        print("\n" + "="*80)
        print("Multi-turn CIR Filtering Detailed Report")
        print("="*80)
        
        # Dataset information
        print(f"\n【{dataset_name}】")
        print("-" * 50)
        baseline = report['dataset_info']['baseline_stats']
        if 'total_queries_in_dataset' in baseline:
            print(f"Total queries in dataset: {baseline['total_queries_in_dataset']:,}")
            if 'total_image_pairs' in baseline:
                print(f"Image pairs: {baseline['total_image_pairs']:,}")
        
        experiment = report['dataset_info']['experiment_stats']
        print(f"Experiment processed queries: {experiment['total_processed_queries']:,}")
        if isinstance(experiment['processing_coverage'], (int, float)):
            print(f"Processing coverage: {experiment['processing_coverage']:.1f}%")
        
        # Display dataset sample examples
        if 'displayed_sample_queries' in baseline and baseline['displayed_sample_queries']:
            print(f"\nDataset sample examples:")
            # Use already randomly selected samples
            random_samples = baseline['displayed_sample_queries']
            
            for i, sample in enumerate(random_samples, 1):
                if dataset_name.startswith('fashioniq'):
                    print(f"  Example {i}: {sample.get('candidate', '')} → {sample.get('target', '')}")
                    print(f"       Modification: \"{sample.get('caption', '')}\"")
                else:
                    print(f"  Example {i}: {sample}")
        
        # Original data analysis
        print(f"\n[Original Data Analysis]")
        original = report['original_analysis']
        print(f"Total queries: {original['total_queries']:,}")
        print(f"Success rate: {original['summary']['success_rate']:.1f}%")
        
        print("\nSuccess statistics by turn:")
        for turn_key, stats in original['turn_success_breakdown'].items():
            if turn_key != 'no_success':
                turn_num = turn_key.replace('turn_', '')
                print(f"  Success at Turn {turn_num}: {stats['count']:,} ({stats['percentage']:.1f}%)")
                
                # Display one dialog example
                if stats.get('examples') and len(stats['examples']) > 0:
                    example = stats['examples'][0]
                    print(f"    === Dialog Example (Query {example['query_id']}) ===")
                    print(f"    Reference image: {example['reference_image']} → Target image: {example['target_image']}")
                    
                    # Display each turn of the dialog
                    for dialogue in example.get('dialogue', []):
                        turn_idx = dialogue['turn']
                        user_input = dialogue.get('user_input', '')
                        system_action = dialogue.get('system_action', '')

                        print(f"    Turn {turn_idx}:")
                        print(f"      User: \"{user_input}\"")
                        print(f"      System: {system_action}")
                        
                        if dialogue.get('is_success_turn', False):
                            print(f"      → Success! Final rank: {example.get('final_rank', 'N/A')}")
                            break
                    print()

        no_success = original['turn_success_breakdown'].get('no_success', {})
        print(f"  No success: {no_success.get('count', 0):,} ({no_success.get('percentage', 0):.1f}%)")
        if no_success.get('examples') and len(no_success.get('examples', [])) > 0:
            example = no_success['examples'][0]
            print(f"    === Failed Dialog Example (Query {example['query_id']}) ===")
            print(f"    Reference image: {example['reference_image']} → Target image: {example['target_image']}")
            
            # Display all turns
            for dialogue in example.get('dialogue', []):
                turn_idx = dialogue['turn']
                user_input = dialogue.get('user_input', '')
                system_action = dialogue.get('system_action', '')

                print(f"    Turn {turn_idx}:")
                print(f"      User: \"{user_input}\"")
                print(f"      System: {system_action}")

            print(f"    → Did not succeed (Final rank: {example.get('final_rank', 'N/A')})")
            print()
        
        # Filtering analysis
        print(f"\n[Stage-by-Stage Filtering]")
        for stage in report['filtering_analysis']['stages']:
            print(f"{stage['stage_name']}:")
            print(f"  Input: {stage['input_count']:,} → Output: {stage['output_count']:,}")
            print(f"  Excluded: {stage['filtered_count']:,} ({stage['filtering_rate']:.1f}%)")
            print(f"  Retention rate: {stage['retention_rate']:.1f}%")
        
        # Final results
        print(f"\n[Post-Filtering Analysis]")
        final = report['final_analysis']
        print(f"Total queries: {final['total_queries']:,}")
        print(f"Success rate: {final['summary']['success_rate']:.1f}%")
        
        print("\nSuccess statistics by turn:")
        for turn_key, stats in final['turn_success_breakdown'].items():
            if turn_key != 'no_success':
                turn_num = turn_key.replace('turn_', '')
                print(f"  Success at Turn {turn_num}: {stats['count']:,} ({stats['percentage']:.1f}%)")
                
                # Display one post-filtering dialog example
                if stats.get('examples') and len(stats['examples']) > 0:
                    example = stats['examples'][0]
                    print(f"    === Post-Filtering Dialog Example (Query {example['query_id']}) ===")
                    print(f"    Reference image: {example['reference_image']} → Target image: {example['target_image']}")
                    
                    # Display dialog until success
                    for dialogue in example.get('dialogue', []):
                        turn_idx = dialogue['turn']
                        user_input = dialogue.get('user_input', '')
                        system_action = dialogue.get('system_action', '')

                        print(f"    Turn {turn_idx}:")
                        print(f"      User: \"{user_input}\"")
                        print(f"      System: {system_action}")
                        
                        if dialogue.get('is_success_turn', False):
                            print(f"      → Success! Final rank: {example.get('final_rank', 'N/A')}")
                            break
                    print()

        no_success = final['turn_success_breakdown'].get('no_success', {})
        print(f"  No success: {no_success.get('count', 0):,} ({no_success.get('percentage', 0):.1f}%)")
        
        # Quality improvement
        print(f"\n[Quality Improvement]")
        comparison = report['comparison']['quality_improvement']
        print(f"Original success rate: {comparison['original_success_rate']:.1f}%")
        print(f"Final success rate: {comparison['final_success_rate']:.1f}%")
        print(f"Success rate improvement: {comparison['final_success_rate'] - comparison['original_success_rate']:+.1f}%")
        
        # Overall summary
        overall = report['filtering_analysis']['overall_summary']
        print(f"\n[Overall Summary]")
        print(f"Overall filtering rate: {overall['overall_filtering_rate']:.1f}%")
        print(f"Overall retention rate: {overall['overall_retention_rate']:.1f}%")

    def analyze_turn_success(self, json_data: Dict, query_ids: List[int]) -> Dict:
        """Analyze success statistics by turn"""

        # Convert JSON results to dictionary
        results_dict = {}
        for result in json_data['results']:
            try:
                query_id = int(result['query_id'])
                results_dict[query_id] = result
            except (ValueError, TypeError):
                continue
        
        turn_success_stats = {}
        turn_candidates = {}  # Store candidates for random selection
        no_success_candidates = []
        
        for query_id in query_ids:
            if query_id not in results_dict:
                continue
            
            result = results_dict[query_id]
            
            # Check success field at result level
            if result.get('success', False):
                # Identify successful turn
                success_turn = result.get('success_turn', 0)
                turn_key = f"turn_{success_turn}"
                
                if turn_key not in turn_success_stats:
                    turn_success_stats[turn_key] = []
                    turn_candidates[turn_key] = []
                
                turn_success_stats[turn_key].append(query_id)
                
                # Create detailed dialog example
                dialogue_example = {
                    'query_id': query_id,
                    'reference_image': result.get('reference_image_id', ''),
                    'target_image': result.get('target_image_id', ''),
                    'original_query': result.get('original_query', ''),
                    'success_turn': success_turn,
                    'dialogue': []
                }
                
                # Build dialog by turn
                turns = result.get('turns', [])
                for i, turn_data in enumerate(turns):
                    turn_info = {
                        'turn': i,
                        'query_text': turn_data.get('query_text', ''),
                        'relative_caption': turn_data.get('relative_caption', ''),
                        'selected_image': turn_data.get('selected_image', ''),
                        'best_gt_rank': turn_data.get('best_gt_rank', 'N/A'),
                        'is_success_turn': (i == success_turn)
                    }
                    
                    # Express as user-system dialog format
                    if i == 0:
                        turn_info['user_input'] = result.get('original_query', '')
                        turn_info['system_action'] = f"Image search executed → Rank {turn_data.get('best_gt_rank', 'N/A')}"
                    else:
                        turn_info['user_input'] = turn_data.get('relative_caption', '')
                        turn_info['system_action'] = f"Modified search executed → Rank {turn_data.get('best_gt_rank', 'N/A')}"

                    dialogue_example['dialogue'].append(turn_info)
                
                # Final result information
                final_turn = turns[success_turn] if success_turn < len(turns) else {}
                dialogue_example['final_rank'] = final_turn.get('best_gt_rank', 'N/A')
                dialogue_example['final_query'] = final_turn.get('query_text', '')
                
                turn_candidates[turn_key].append(dialogue_example)
            else:
                # Queries that did not succeed
                no_success_example = {
                    'query_id': query_id,
                    'reference_image': result.get('reference_image_id', ''),
                    'target_image': result.get('target_image_id', ''),
                    'original_query': result.get('original_query', ''),
                    'final_status': 'No success achieved',
                    'dialogue': []
                }
                
                # Record dialog for all turns
                turns = result.get('turns', [])
                max_turn = len(turns) - 1
                for i, turn_data in enumerate(turns):
                    turn_info = {
                        'turn': i,
                        'query_text': turn_data.get('query_text', ''),
                        'relative_caption': turn_data.get('relative_caption', ''),
                        'selected_image': turn_data.get('selected_image', ''),
                        'best_gt_rank': turn_data.get('best_gt_rank', 'N/A'),
                        'is_final_turn': (i == max_turn)
                    }
                    
                    if i == 0:
                        turn_info['user_input'] = result.get('original_query', '')
                        turn_info['system_action'] = f"Image search executed → Rank {turn_data.get('best_gt_rank', 'N/A')}"
                    else:
                        turn_info['user_input'] = turn_data.get('relative_caption', '')
                        turn_info['system_action'] = f"Modified search executed → Rank {turn_data.get('best_gt_rank', 'N/A')}"

                    no_success_example['dialogue'].append(turn_info)

                # Final rank information
                if turns:
                    final_turn = turns[-1]
                    no_success_example['final_rank'] = final_turn.get('best_gt_rank', 'N/A')
                    no_success_example['final_query'] = final_turn.get('query_text', '')
                
                no_success_candidates.append(no_success_example)
        
        # Summarize statistics (randomly select 3)
        turn_stats = {}
        for turn_key in sorted(turn_success_stats.keys()):
            # Randomly select 3 examples
            candidates = turn_candidates[turn_key]
            random_examples = random.sample(candidates, min(3, len(candidates))) if candidates else []
            
            turn_stats[turn_key] = {
                'count': len(turn_success_stats[turn_key]),
                'percentage': len(turn_success_stats[turn_key]) / len(query_ids) * 100 if query_ids else 0,
                'examples': random_examples
            }
        
        # Also randomly select 3 examples of queries that did not succeed
        random_no_success = random.sample(no_success_candidates, min(3, len(no_success_candidates))) if no_success_candidates else []
        
        turn_stats['no_success'] = {
            'count': len(no_success_candidates),
            'percentage': len(no_success_candidates) / len(query_ids) * 100 if query_ids else 0,
            'examples': random_no_success
        }
        
        return turn_stats

    def get_dataset_baseline_stats(self, dataset_name: str) -> Dict:
        """Get basic statistics of the dataset"""
        stats = {}

        # Get input_dir from instance variable
        input_dir = getattr(self, 'input_dir', '.')

        # Check actually processed data
        json_file = os.path.join(input_dir, f"multiturn_cir_results_{dataset_name}.json")
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                actual_processed = len(json_data.get('results', []))
                stats['actual_processed_queries'] = actual_processed
                
                # Get estimated processable queries from data_completeness
                completeness = json_data.get('data_completeness', {})
                estimated_processable = completeness.get('estimated_processable_queries', actual_processed)
                stats['estimated_processable_queries'] = estimated_processable
        
        # Dataset-specific statistics
        if dataset_name.startswith('fashioniq'):
            category = dataset_name.split('_')[1]  # dress, shirt, toptee
            split = 'val' if 'val' in dataset_name else 'train'
            
            annotation_file = f'fashion-iq/captions/cap.{category}.{split}.json'
            if os.path.exists(annotation_file):
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                    
                    # Fashion-IQ uses relative_captions array format
                    total_triplets = 0
                    for item in data:
                        total_triplets += len(item.get('captions', []))
                    
                    stats['total_queries_in_dataset'] = total_triplets
                    stats['total_image_pairs'] = len(data)
                    stats['annotation_file'] = annotation_file
            
        elif dataset_name.startswith('cirr'):
            split = 'val' if 'val' in dataset_name else 'train'
            annotation_file = f'cirr/captions/cap.rc2.{split}.json'
            if os.path.exists(annotation_file):
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                    
                    # For CIRR, preserve total count from original annotation file
                    stats['total_queries_in_dataset'] = len(data)
                    stats['annotation_file'] = annotation_file
                    
                    # Sample examples are randomly selected from actually processed data
                    if 'actual_processed_queries' in stats and stats['actual_processed_queries'] > 0:
                        if os.path.exists(json_file):
                            with open(json_file, 'r') as f:
                                json_data = json.load(f)
                                results = json_data.get('results', [])
                                if results:
                                    # Select random sample from actual results
                                    import random
                                    sample_size = min(5, len(results))
                                    stats['sample_queries'] = random.sample(results, sample_size)
                                else:
                                    stats['sample_queries'] = []
                        else:
                            stats['sample_queries'] = data
                    
        elif dataset_name == 'circo':
            annotation_file = 'CIRCO/annotations/val.json'
            if os.path.exists(annotation_file):
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                    stats['total_queries_in_dataset'] = len(data)
                    stats['annotation_file'] = annotation_file
                    
                    # Save all sample examples
                    stats['sample_queries'] = data
        
        return stats

    def generate_detailed_report(self, dataset_name: str, original_data: Dict, 
                               filtering_stages: List[Dict], final_data: Dict) -> Dict:
        """Generate detailed analysis report"""

        # Get basic statistics
        baseline_stats = self.get_dataset_baseline_stats(dataset_name)
        
        # Randomly select sample examples (same logic as print display)
        if 'sample_queries' in baseline_stats and baseline_stats['sample_queries']:
            sample_queries = baseline_stats['sample_queries']
            random_samples = random.sample(sample_queries, min(2, len(sample_queries))) if sample_queries else []
            baseline_stats['displayed_sample_queries'] = random_samples
            # Delete all samples to reduce file size
            del baseline_stats['sample_queries']
        
        # Original data statistics
        all_query_ids = []
        for result in original_data['results']:
            try:
                query_id = int(result['query_id'])
                all_query_ids.append(query_id)
            except (ValueError, TypeError):
                continue
        
        original_turn_stats = self.analyze_turn_success(original_data, all_query_ids)
        
        # Post-filtering statistics
        final_query_ids = []
        for result in final_data['results']:
            try:
                query_id = int(result['query_id'])
                final_query_ids.append(query_id)
            except (ValueError, TypeError):
                continue
        
        final_turn_stats = self.analyze_turn_success(original_data, final_query_ids)
        
        # Stage-by-stage filtering statistics
        stage_stats = []
        for stage in filtering_stages:
            stage_info = {
                'stage_name': stage['name'],
                'input_count': stage['input_count'],
                'output_count': stage['output_count'],
                'filtered_count': stage['input_count'] - stage['output_count'],
                'filtering_rate': (stage['input_count'] - stage['output_count']) / stage['input_count'] * 100 if stage['input_count'] > 0 else 0,
                'retention_rate': stage['output_count'] / stage['input_count'] * 100 if stage['input_count'] > 0 else 0
            }
            
            # Add specific examples if available
            if 'examples' in stage:
                stage_info['examples'] = stage['examples']
                
            stage_stats.append(stage_info)
        
        # Use actually processed query count
        actual_processed = baseline_stats.get('actual_processed_queries', len(all_query_ids))
        
        # Calculate processing coverage from original dataset total query count
        total_queries_in_dataset = baseline_stats.get('total_queries_in_dataset', actual_processed)
        processing_coverage = len(original_data['results']) / total_queries_in_dataset * 100 if total_queries_in_dataset > 0 else 100.0
        
        # Create detailed report
        report = {
            'dataset_info': {
                'dataset_name': dataset_name,
                'baseline_stats': baseline_stats,
                'experiment_stats': {
                    'total_processed_queries': len(original_data['results']),
                    'processing_coverage': processing_coverage
                }
            },
            'original_analysis': {
                'total_queries': len(all_query_ids),
                'turn_success_breakdown': original_turn_stats,
                'summary': {
                    'successful_queries': sum(stats['count'] for key, stats in original_turn_stats.items() if key != 'no_success'),
                    'unsuccessful_queries': original_turn_stats.get('no_success', {}).get('count', 0),
                    'success_rate': sum(stats['count'] for key, stats in original_turn_stats.items() if key != 'no_success') / len(all_query_ids) * 100 if all_query_ids else 0
                }
            },
            'filtering_analysis': {
                'stages': stage_stats,
                'overall_summary': {
                    'initial_count': len(all_query_ids),
                    'final_count': len(final_query_ids),
                    'total_filtered': len(all_query_ids) - len(final_query_ids),
                    'overall_filtering_rate': (len(all_query_ids) - len(final_query_ids)) / len(all_query_ids) * 100 if all_query_ids else 0,
                    'overall_retention_rate': len(final_query_ids) / len(all_query_ids) * 100 if all_query_ids else 0
                }
            },
            'final_analysis': {
                'total_queries': len(final_query_ids),
                'turn_success_breakdown': final_turn_stats,
                'summary': {
                    'successful_queries': sum(stats['count'] for key, stats in final_turn_stats.items() if key != 'no_success'),
                    'unsuccessful_queries': final_turn_stats.get('no_success', {}).get('count', 0),
                    'success_rate': sum(stats['count'] for key, stats in final_turn_stats.items() if key != 'no_success') / len(final_query_ids) * 100 if final_query_ids else 0
                }
            },
            'comparison': {
                'quality_improvement': {
                    'original_success_rate': sum(stats['count'] for key, stats in original_turn_stats.items() if key != 'no_success') / len(all_query_ids) * 100 if all_query_ids else 0,
                    'final_success_rate': sum(stats['count'] for key, stats in final_turn_stats.items() if key != 'no_success') / len(final_query_ids) * 100 if final_query_ids else 0
                }
            }
        }
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Optimized Multi-turn CIR Filtering")
    parser.add_argument('--datasets', nargs='+', 
                       default=['fashioniq_dress_val', 'fashioniq_shirt_val', 'fashioniq_toptee_val', 'cirr_val', 'cirr_train', 'circo', 'fashioniq_dress_train', 'fashioniq_shirt_train', 'fashioniq_toptee_train'],
                       help='Dataset names to filter')
    parser.add_argument('--similarity-threshold', type=float, default=0.8,
                       help='CLIP similarity threshold')
    parser.add_argument('--rank-margin', type=int, default=30,
                       help='Rank margin threshold')
    parser.add_argument('--no-success-filter', action='store_true',
                       help='Skip success filter')
    parser.add_argument('--no-multiturn-filter', action='store_true',
                       help='Skip multiturn filter')
    parser.add_argument('--no-original-query-filter', action='store_true',
                       help='Skip original query filter')
    parser.add_argument('--no-rank-margin-filter', action='store_true',
                       help='Skip rank margin filter')
    parser.add_argument('--no-clip-filter', action='store_true',
                       help='Skip CLIP similarity filter')
    parser.add_argument('--no-image-duplication-filter', action='store_true',
                       help='Skip image duplication filter')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze, do not save filtered data')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducible example selection')
    parser.add_argument('--data_dir', type=str, default='.',
                       help='Base directory containing dataset folders')
    parser.add_argument('--input_dir', type=str, default='output/raw',
                       help='Input directory for raw result files (default: output/raw)')
    parser.add_argument('--output_dir', type=str, default='output/filtered',
                       help='Output directory for filtered files (default: output/filtered)')

    args = parser.parse_args()

    # Convert input_dir and output_dir to absolute paths before changing directory
    # This allows input/output to be independent of data_dir
    args.input_dir = os.path.abspath(args.input_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    # Change to data directory
    if args.data_dir != '.':
        os.chdir(args.data_dir)
        print(f"Working directory: {os.getcwd()}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Initialize filter
    filter_system = OptimizedMultiTurnFilter(
        similarity_threshold=args.similarity_threshold,
        rank_margin=args.rank_margin
    )
    
    # Filter each dataset
    results = []
    for dataset_name in args.datasets:
        try:
            result = filter_system.filter_dataset(
                dataset_name=dataset_name,
                apply_success=not args.no_success_filter,
                apply_multiturn=not args.no_multiturn_filter,
                apply_original_query=not args.no_original_query_filter,
                apply_rank_margin=not args.no_rank_margin_filter,
                apply_clip_similarity=not args.no_clip_filter,
                apply_image_duplication=not args.no_image_duplication_filter,
                save_filtered=not args.analyze_only,
                input_dir=args.input_dir,
                output_dir=args.output_dir
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
    
    print(f"\nProcessing completed for {len(results)} datasets")

if __name__ == "__main__":
    main() 