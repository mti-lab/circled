#!/usr/bin/env python3
"""
Multi-turn CIR System - Supports CIRCO, CIRR, and FashionIQ
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
import clip
from tqdm import tqdm
import time
import warnings
from typing import List, Dict, Tuple, Optional, Any
from transformers import (
    BlipForImageTextRetrieval, AutoProcessor
)
from torch.nn.functional import normalize
import csv
import argparse
import openai
import asyncio
import aiohttp
import base64
from io import BytesIO
from openai import AsyncOpenAI
import hashlib

warnings.filterwarnings("ignore")

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Make sure OPENAI_API_KEY is set as environment variable.")

# Set OpenAI API key
openai.api_key = os.getenv('OPEN_API_KEY')
if not openai.api_key:
    raise ValueError("OPEN_API_KEY environment variable is not set. Please set it in .env file or as environment variable.")

# Cache directory can be set via environment variables:
# HF_HOME, TRANSFORMERS_CACHE

class BlipForRetrieval(BlipForImageTextRetrieval):
    def get_text_features(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        question_embeds = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_feat = normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)
        return text_feat

    def get_image_features(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_feat = normalize(self.vision_proj(vision_outputs[0][:, 0, :]), dim=-1)
        return image_feat

class DatasetLoader:
    """Dataset-specific loader class"""
    
    @staticmethod
    def load_circo_data(annotation_file: str) -> List[Dict]:
        """Load CIRCO data"""
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Create image ID → hash value mapping from metadata file
        import torch
        metadata_file = 'CIRCO/metadata_blip.pt'
        metadata = torch.load(metadata_file, map_location='cpu', weights_only=False)
        
        # Create image ID (6-digit number) → hash value mapping
        image_id_to_hash = {}
        hash_to_idx = metadata['hash_to_idx']
        idx_to_info = metadata['idx_to_info']
        
        for hash_val, idx in hash_to_idx.items():
            info = idx_to_info.get(idx, {})
            image_id = info.get('image_id', '')
            if image_id:
                image_id_to_hash[image_id] = hash_val
        
        print(f"Created image_id to hash mapping for {len(image_id_to_hash)} images")
        
        formatted_data = []
        for item in data:
            # Convert numeric ID to 6-digit string and get hash value
            ref_id_str = str(item['reference_img_id'])
            target_id_str = str(item['target_img_id'])
            gt_id_strs = [str(gt_id) for gt_id in item['gt_img_ids']]
            
            # Convert to hash value (keep original 12-digit format if not found)
            ref_hash = image_id_to_hash.get(ref_id_str, f"{item['reference_img_id']:012d}.jpg")
            target_hash = image_id_to_hash.get(target_id_str, f"{item['target_img_id']:012d}.jpg")
            gt_hashes = [image_id_to_hash.get(gt_id_str, f"{int(gt_id_str):012d}.jpg") for gt_id_str in gt_id_strs]
            
            formatted_data.append({
                'reference_image_id': ref_hash,
                'target_image_id': target_hash,
                'relative_caption': item['relative_caption'],
                'ground_truth_ids': gt_hashes,
                'shared_concept': item.get('shared_concept', ''),
                'id': item['id'],
                # Keep original IDs for debugging
                'original_reference_id': f"{item['reference_img_id']:012d}.jpg",
                'original_target_id': f"{item['target_img_id']:012d}.jpg",
                'original_gt_ids': [f"{gt_id:012d}.jpg" for gt_id in item['gt_img_ids']]
            })
        return formatted_data
    
    @staticmethod
    def load_cirr_data(annotation_file: str) -> List[Dict]:
        """Load CIRR data (using only single hard target as ground truth)"""
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        formatted_data = []
        for item in data:
            # CIRR basically evaluates against a single hard target
            formatted_data.append({
                'reference_image_id': item['reference'],
                'target_image_id': item['target_hard'],  # Main target
                'relative_caption': item['caption'],
                'ground_truth_ids': [item['target_hard']],  # Single hard target only
                'id': item.get('pairid', item.get('id', 0))
            })
        return formatted_data
    
    @staticmethod
    def load_fashioniq_data(annotation_file: str, caption_mode: str = 'separate') -> List[Dict]:
        """Load FashionIQ data

        Args:
            annotation_file: Path to annotation file
            caption_mode: Caption processing mode
                - 'separate': Treat each caption as an independent sample (recommended/standard)
                - 'combined': Combine multiple captions
                - 'first_only': Use only the first caption
        """
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        formatted_data = []
        query_id_counter = 0  # Explicit ID counter
        
        # Statistics
        total_items = len(data)
        skipped_empty_captions = 0
        skipped_no_valid_captions = 0
        
        for item in data:
            # Adapt to actual Fashion-IQ file format
            target_id = item['target']      # Target image ID
            candidate_id = item['candidate'] # Reference image ID (candidate is the actual reference image)
            captions = item['captions']     # List of relative captions
            
            # Exclude empty strings and whitespace-only captions
            valid_captions = [caption.strip() for caption in captions if caption and caption.strip()]
            
            # Skip if there are no valid captions
            if not valid_captions:
                skipped_no_valid_captions += 1
                continue
            
            if caption_mode == 'separate':
                # Method 1: Treat each caption as an independent sample (standard approach)
                for i, caption in enumerate(valid_captions):
                    formatted_data.append({
                        'reference_image_id': candidate_id,
                        'target_image_id': target_id, 
                        'relative_caption': caption,  # Individual caption (empty strings excluded)
                        'ground_truth_ids': [target_id],
                        'id': query_id_counter,  # Set explicit ID
                        'caption_index': i,  # Record which caption this is
                        'original_captions': captions,  # Original captions (before filtering)
                        'valid_captions': valid_captions  # Valid captions only
                    })
                    query_id_counter += 1
                    
            elif caption_mode == 'combined':
                # Method 2: Combine multiple captions as natural text
                if len(valid_captions) == 1:
                    combined_caption = valid_captions[0]
                elif len(valid_captions) == 2:
                    combined_caption = f"{valid_captions[0]} and {valid_captions[1]}"
                else:
                    # For 3 or more captions (rare)
                    combined_caption = ", ".join(valid_captions[:-1]) + f", and {valid_captions[-1]}"
                
                formatted_data.append({
                    'reference_image_id': candidate_id,
                    'target_image_id': target_id, 
                    'relative_caption': combined_caption,  # Combined caption
                    'ground_truth_ids': [target_id],
                    'id': query_id_counter,  # Set explicit ID
                    'original_captions': captions,  # Original captions (before filtering)
                    'valid_captions': valid_captions  # Valid captions only
                })
                query_id_counter += 1
                
            elif caption_mode == 'first_only':
                # Method 3: Use only the first valid caption
                formatted_data.append({
                    'reference_image_id': candidate_id,
                    'target_image_id': target_id, 
                    'relative_caption': valid_captions[0],  # Only the first valid caption
                    'ground_truth_ids': [target_id],
                    'id': query_id_counter,  # Set explicit ID
                    'original_captions': captions,  # Original captions (before filtering)
                    'valid_captions': valid_captions  # Valid captions only
                })
                query_id_counter += 1
            
            # Statistics for empty string captions
            if len(valid_captions) < len(captions):
                skipped_empty_captions += len(captions) - len(valid_captions)
        
        # Display debug information
        print(f"Fashion-IQ data loading summary:")
        print(f"  Total annotation items: {total_items}")
        print(f"  Items with no valid captions (skipped): {skipped_no_valid_captions}")
        print(f"  Empty/whitespace captions filtered: {skipped_empty_captions}")
        print(f"  Generated query samples: {len(formatted_data)}")
        print(f"  Caption mode: {caption_mode}")
        
        return formatted_data

class ModelManager:
    """Model management class"""
    
    def __init__(self, use_blip: bool = True):
        self.use_blip = use_blip
        self.load_retrieval_models()
        self.load_clip_model()  # For CLIP similarity checking
        self.setup_openai_client()
    
    def load_retrieval_models(self):
        """Load retrieval models"""
        if self.use_blip:
            self.retrieval_model = BlipForRetrieval.from_pretrained("Salesforce/blip-itm-large-coco")
            self.retrieval_processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
            self.retrieval_model = self.retrieval_model.to(device)
            
            self.dialog_encoder = lambda text: self.retrieval_model.get_text_features(
                **self.retrieval_processor(text=text, padding=True, truncation=True, return_tensors="pt").to(device)
            )
        else:
            self.retrieval_model, self.retrieval_preprocess = clip.load("ViT-B/32", device=device)
            self.dialog_encoder = lambda text: self.retrieval_model.encode_text(
                clip.tokenize(text, truncate=True).to(device)
            )
    
    def load_clip_model(self):
        """Load CLIP model for similarity checking"""
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            print("CLIP model loaded for similarity checking")
        except ImportError:
            print("Warning: CLIP not available. Similarity checking will be disabled.")
            self.clip_model = None
            self.clip_preprocess = None
        except Exception as e:
            print(f"Warning: Failed to load CLIP model: {e}")
            self.clip_model = None
            self.clip_preprocess = None
    
    def setup_openai_client(self):
        """Setup OpenAI API client"""
        # Verify that OpenAI API key is set
        if not openai.api_key:
            raise ValueError("OpenAI API key is not set. Please set OPENAI_API_KEY environment variable.")
        
        print("OpenAI GPT-4o-mini client ready for caption generation")
    
    async def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to Base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    async def call_gpt4o_mini(self, messages: List[Dict], max_tokens: int = 100) -> str:
        """Make API call to GPT-4o-mini"""
        client = AsyncOpenAI(api_key=openai.api_key)
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()

    async def combine_captions_with_gpt4o(self, original_caption: str, relative_caption: str) -> str:
        """Combine captions using GPT-4o-mini"""
        messages = [
            {
                "role": "system",
                "content": "Combine two descriptions into one short caption. Output only the caption, no explanations or prefixes."
            },
            {
                "role": "user", 
                "content": f"Base: {original_caption}\nChange: {relative_caption}\nResult:"
            }
        ]
        
        response = await self.call_gpt4o_mini(messages, max_tokens=50)

        # Remove unnecessary prefixes and redundant expressions
        response = self._clean_caption_text(response)

        return response.strip()

    def _clean_caption_text(self, text: str) -> str:
        """Clean caption text by removing redundant prefixes and expressions.

        Combines patterns from clean_captions.py and GPT output formatting prefixes.
        """
        import re

        if not text or not isinstance(text, str):
            return text

        cleaned = text.strip()

        # GPT output formatting prefixes (exact match, case-insensitive)
        gpt_prefixes = [
            "Comprehensive Caption:",
            "New caption:",
            "Caption:",
            "Combined caption:",
            "Result:",
        ]

        for prefix in gpt_prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
                if cleaned.startswith(':'):
                    cleaned = cleaned[1:].strip()

        # Redundant image description patterns (regex-based, from clean_captions.py)
        redundant_patterns = [
            r'^The image features?\s+',
            r'^The image shows?\s+',
            r'^The image depicts?\s+',
            r'^The image captures?\s+',
            r'^The image displays?\s+',
            r'^The image presents?\s+',
            r'^The image contains?\s+',
            r'^The image includes?\s+',
            r'^The image showcases?\s+',
            r'^This image features?\s+',
            r'^This image shows?\s+',
            r'^This image depicts?\s+',
            r'^This image captures?\s+',
            r'^This image displays?\s+',
            r'^This image presents?\s+',
            r'^This image contains?\s+',
            r'^This image includes?\s+',
            r'^This striking image\s+',
            r'^In the image,?\s+',
            r'^In this image,?\s+',
            r'^The photo features?\s+',
            r'^The photo shows?\s+',
            r'^The picture features?\s+',
            r'^The picture shows?\s+',
        ]

        for pattern in redundant_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # Capitalize first letter if needed
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()

        return cleaned.strip()

    async def generate_relative_caption_with_gpt4o(self, ref_image_path: str, target_image_path: str,
                                                  previous_captions: List[str] = None,
                                                  similarity_threshold: float = 0.8,
                                                  max_retries: int = 3) -> str:
        """Generate relative caption using GPT-4o-mini (with CLIP similarity checking)"""
        if not os.path.exists(ref_image_path):
            raise FileNotFoundError(f"Reference image file not found: {ref_image_path}")
        if not os.path.exists(target_image_path):
            raise FileNotFoundError(f"Target image file not found: {target_image_path}")
        
        # If CLIP model is not available, only perform regular generation
        if not (hasattr(self, 'clip_model') and self.clip_model is not None):
            print("CLIP model not available. Skipping similarity check.")
            # Regular relative caption generation (using core part of existing loop processing)
            ref_image_b64 = await self.encode_image_to_base64(ref_image_path)
            target_image_b64 = await self.encode_image_to_base64(target_image_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will see two images.\n"
                                    "**Image 1**: This is the REFERENCE image that needs to be modified.\n"
                                    "**Image 2**: This is the TARGET image showing the desired result.\n\n"
                                    "Write exactly ONE imperative instruction to transform Image 1 into Image 2.\n"
                                    "Requirements:\n"
                                    "1. Start with a verb\n"
                                    "2. Be extremely specific about colors, positions, or actions\n"
                                    "3. Avoid relative terms like 'left' or 'right'\n"
                                    "4. Do not use quotes or explanatory text\n"
                                    "5. Focus on a single, clear change"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{ref_image_b64}",
                                "detail": "low"
                            }
                        },
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{target_image_b64}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ]
            
            generated_caption = await self.call_gpt4o_mini(messages, max_tokens=60)
            # Remove quotes and extra characters, then clean redundant expressions
            cleaned_caption = self._clean_caption_text(generated_caption.strip('"\''))
            return cleaned_caption

        # Function for CLIP feature computation
        def get_clip_text_feature(text: str) -> torch.Tensor:
            """Get CLIP text features"""
            import clip
            device = next(self.clip_model.parameters()).device
            with torch.no_grad():
                tokens = clip.tokenize(text, truncate=True).to(device)
                feat = self.clip_model.encode_text(tokens)
                return torch.nn.functional.normalize(feat, dim=-1).squeeze(0)
        
        # Compute features for previous captions
        previous_features = []
        if previous_captions:
            for caption in previous_captions:
                if caption.strip():
                    feat = get_clip_text_feature(caption)
                    previous_features.append(feat)
        
        # Encode images to Base64
        ref_image_b64 = await self.encode_image_to_base64(ref_image_path)
        target_image_b64 = await self.encode_image_to_base64(target_image_path)
        
        # Retry loop
        for attempt in range(max_retries):
            # Instructions when previous captions exist
            previous_instructions = ""
            if previous_captions and len(previous_captions) > 0:
                previous_instructions = (
                    "IMPORTANT - Previous changes have already been suggested:\n" +
                    "\n".join(f'• "{cap}"' for cap in previous_captions) +
                    "\n\nYour task is to identify a COMPLETELY DIFFERENT visual change. "
                    "Focus on aspects that have NOT been mentioned before.\n\n"
                )
            
            # Additional instructions for retries
            retry_instructions = ""
            if attempt > 0:
                retry_instructions = (
                    f"RETRY #{attempt + 1}: The previous suggestion was too similar to existing ones. "
                    "Please provide a MORE DISTINCTIVE and DIFFERENT instruction. "
                    "Consider completely different visual aspects like:\n"
                    "- Different objects or people\n"
                    "- Different colors or lighting\n"
                    "- Different actions or poses\n"
                    "- Different background elements\n"
                    "- Different clothing or accessories\n\n"
                )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": previous_instructions + retry_instructions +
                                    "You will see two images.\n"
                                    "**Image 1**: This is the REFERENCE image that needs to be modified.\n"
                                    "**Image 2**: This is the TARGET image showing the desired result.\n\n"
                                    "Write exactly ONE imperative instruction to transform Image 1 into Image 2.\n"
                                    "Requirements:\n"
                                    "1. Start with a verb\n"
                                    "2. Be extremely specific about colors, positions, or actions\n"
                                    "3. Avoid relative terms like 'left' or 'right'\n"
                                    "4. Do not use quotes or explanatory text\n"
                                    "5. Focus on a single, clear change\n"
                                    "6. Must be different from previous suggestions"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{ref_image_b64}",
                                "detail": "low"
                            }
                        },
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{target_image_b64}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ]
            
            generated_caption = await self.call_gpt4o_mini(messages, max_tokens=60)
            
            # Remove quotes and extra characters
            generated_caption = generated_caption.strip('"\'')
            # Clean redundant expressions
            generated_caption = self._clean_caption_text(generated_caption)

            if not generated_caption:
                if attempt == max_retries - 1:
                    raise ValueError(f"GPT-4o generated empty caption after {max_retries} attempts for images: {ref_image_path}, {target_image_path}")
                continue
            
            # CLIP similarity check
            if previous_features:
                current_feature = get_clip_text_feature(generated_caption)
                
                # Calculate similarity with previous captions
                is_too_similar = False
                max_similarity = 0.0
                
                for prev_feat in previous_features:
                    similarity = torch.dot(current_feature, prev_feat).item()
                    max_similarity = max(max_similarity, similarity)
                    
                    if similarity >= similarity_threshold:
                        is_too_similar = True
                        break
                
                # Success if similarity is below threshold
                if not is_too_similar:
                    if attempt > 0:
                        print(f"  ✓ Generated distinctive caption after {attempt + 1} attempts (max similarity: {max_similarity:.3f})")
                    return generated_caption
                else:
                    print(f"  ⚠ Attempt {attempt + 1}: Generated caption too similar (similarity: {max_similarity:.3f} >= {similarity_threshold})")
                    if attempt == max_retries - 1:
                        print(f"  ⚠ Using similar caption after {max_retries} attempts: '{generated_caption}'")
                        return generated_caption
            else:
                # Return as-is if there are no previous captions
                return generated_caption

        # Should not reach here, but for safety
        return generated_caption

class RetrievalEngine:
    """Image retrieval engine"""

    def __init__(self, corpus_vectors_file: str, search_space_file: str, model_manager: ModelManager):
        # Check if feature file exists
        if not os.path.exists(corpus_vectors_file):
            raise FileNotFoundError(f"Corpus vectors file not found: {corpus_vectors_file}")
        
        # Load feature file
        corpus_data = torch.load(corpus_vectors_file, map_location=device, weights_only=False)
        
        # Process according to feature file format
        if isinstance(corpus_data, dict):
            # Dictionary format: {'features': tensor, 'hashes': list, ...}
            if 'features' in corpus_data and 'hashes' in corpus_data:
                all_corpus_features = corpus_data['features']
                all_corpus_ids = corpus_data['hashes']
            else:
                raise ValueError(f"Expected 'features' and 'hashes' keys in corpus file: {corpus_vectors_file}")
        elif isinstance(corpus_data, (tuple, list)) and len(corpus_data) == 2:
            # Tuple format: (corpus_ids, corpus_features)
            all_corpus_ids, all_corpus_features = corpus_data
        else:
            raise ValueError(f"Unsupported corpus vectors format in file: {corpus_vectors_file}")
        
        # Check if search space file exists
        if not os.path.exists(search_space_file):
            raise FileNotFoundError(f"Search space file not found: {search_space_file}")
        
        # Change loading method according to search space file format
        if search_space_file.endswith('.pt'):
            # For PyTorch files (metadata files)
            metadata = torch.load(search_space_file, map_location='cpu', weights_only=False)
            if isinstance(metadata, dict) and 'image_ids' in metadata:
                self.search_space = metadata['image_ids']
            elif isinstance(metadata, dict) and 'idx_to_info' in metadata:
                # For CIRCO: Extract image IDs from metadata file
                idx_to_info = metadata['idx_to_info']
                # For CIRCO, get hash from hash_to_idx (matches corpus ID)
                if 'circo' in search_space_file.lower() and 'hash_to_idx' in metadata:
                    self.search_space = list(metadata['hash_to_idx'].keys())
                    print(f"Extracted {len(self.search_space)} image hashes from CIRCO metadata")
                else:
                    self.search_space = [info.get('image_id', '') for info in idx_to_info.values() if info.get('image_id')]
                    print(f"Extracted {len(self.search_space)} image IDs from metadata")
            elif isinstance(metadata, list):
                self.search_space = metadata
            else:
                # Extract image IDs from metadata
                self.search_space = list(metadata.keys()) if isinstance(metadata, dict) else metadata
        else:
            # For JSON files
            with open(search_space_file, 'r') as f:
                search_data = json.load(f)
                if isinstance(search_data, dict):
                    # For dictionary format, get list of keys (for CIRR)
                    self.search_space = list(search_data.keys())
                    # Also keep values (relative path information)
                    self.search_space_paths = search_data
                else:
                    # For list format
                    self.search_space = search_data
                    self.search_space_paths = None
        
        # Verify that search space is not empty
        if not self.search_space:
            raise ValueError(f"Search space is empty. Check file: {search_space_file}")
        
        # Extract only features for images in search space
        valid_indices = []
        self.corpus_ids = []
        self.search_space_filtered = []  # Only images that actually exist
        
        print(f"Filtering features for search space ({len(self.search_space)} images)...")
        
        # Set image directory (per dataset)
        if 'fashion-iq' in corpus_vectors_file.lower():
            image_base_dir = 'fashion-iq/images'
        elif 'cirr' in corpus_vectors_file.lower():
            image_base_dir = 'cirr/img_raw'
        elif 'circo' in corpus_vectors_file.lower():
            image_base_dir = 'CIRCO/unlabeled2017'
        else:
            image_base_dir = ''  # Fallback
        
        def check_image_file_exists(image_id: str, image_path: str = None) -> bool:
            """Check if image file actually exists"""
            if not image_base_dir:
                return True  # Skip if directory is unknown

            # Use metadata image_path if available
            if image_path and os.path.exists(image_path):
                return True
                
            # For FashionIQ, use category-specific directories
            if 'fashion-iq' in corpus_vectors_file.lower():
                for category in ['dress', 'shirt', 'toptee']:
                    image_path = os.path.join(image_base_dir, category, f"{image_id}.jpg")
                    if os.path.exists(image_path):
                        return True
                return False
            elif 'cirr' in corpus_vectors_file.lower():
                # For CIRR: Structure differs by split
                # train: Hierarchical structure (0-99 subdirectories)
                # val, dev, test1: Flat structure (files directly under)

                # For train: Hierarchical structure
                for subdir in range(100):  # Search 0-99 subdirectories
                    for ext in ['.jpg', '.jpeg', '.png']:
                        image_path = os.path.join(image_base_dir, 'train', str(subdir), f"{image_id}{ext}")
                        if os.path.exists(image_path):
                            return True
                
                # For val, dev, test1: Flat structure
                for split in ['val', 'dev', 'test1']:
                    for ext in ['.png', '.jpg', '.jpeg']:
                        image_path = os.path.join(image_base_dir, split, f"{image_id}{ext}")
                        if os.path.exists(image_path):
                            return True
                
                return False
            else:
                # For CIRCO (convert image_id to 12-digit zero-padded format)
                if image_path and os.path.exists(image_path):
                    return True
                
                # Convert image_id to 12-digit zero-padded filename
                padded_filename = f"{int(image_id):012d}.jpg"
                image_path = os.path.join(image_base_dir, padded_filename)
                return os.path.exists(image_path)
        
        # Unified approach: Prioritize metadata file, fallback to direct matching
        metadata_file = corpus_vectors_file.replace('features_', 'metadata_')
        use_metadata = False
        
        # For CIRCO, search_space_file may be the same as metadata file
        if 'circo' in corpus_vectors_file.lower() and search_space_file == metadata_file:
            # For CIRCO, search space and metadata are the same file
            use_metadata = True
            metadata = torch.load(metadata_file, map_location='cpu', weights_only=False)
            
            # CIRCO metadata processing
            matched_count = 0
            search_space_set = set(self.search_space)  # For fast lookup

            # Use hash_to_idx from feature file (this is the actual feature array index)
            if isinstance(corpus_data, dict) and 'hash_to_idx' in corpus_data:
                features_hash_to_idx = corpus_data['hash_to_idx']
                
                for hash_val in search_space_set:
                    if hash_val in features_hash_to_idx:
                        corpus_idx = features_hash_to_idx[hash_val]
                        
                        # Get image information from metadata (for existence check)
                        if 'hash_to_idx' in metadata and hash_val in metadata['hash_to_idx']:
                            metadata_idx = metadata['hash_to_idx'][hash_val]
                            info = metadata['idx_to_info'].get(metadata_idx, {})
                            image_id = info.get('image_id', '')
                            image_path = info.get('image_path', '')
                            
                            # Check if image file exists
                            file_exists = check_image_file_exists(image_id, image_path)

                            if file_exists:
                                valid_indices.append(corpus_idx)
                                self.corpus_ids.append(all_corpus_ids[corpus_idx])
                                self.search_space_filtered.append(hash_val)  # Use hash
                                matched_count += 1
                        else:
                            # Use even if not in metadata, as long as features exist
                            valid_indices.append(corpus_idx)
                            self.corpus_ids.append(all_corpus_ids[corpus_idx])
                            self.search_space_filtered.append(hash_val)
                            matched_count += 1

            print(f"Successfully matched {matched_count} images using CIRCO metadata")
        
        # For CIRR, use metadata file for matching
        elif 'cirr' in corpus_vectors_file.lower() and os.path.exists(metadata_file):
            use_metadata = True
            metadata = torch.load(metadata_file, map_location='cpu', weights_only=False)
            
            # CIRR metadata processing
            matched_count = 0
            search_space_set = set(self.search_space)  # For fast lookup
            
            if 'idx_to_info' in metadata:
                idx_to_info = metadata['idx_to_info']
                for idx, info in idx_to_info.items():
                    image_id = info.get('image_id', '')
                    image_path = info.get('image_path', '')
                    
                    # Check if in search space
                    if image_id in search_space_set:
                        # Check if image file exists
                        file_exists = check_image_file_exists(image_id, image_path)

                        if file_exists and idx < len(all_corpus_ids):
                            valid_indices.append(idx)
                            self.corpus_ids.append(all_corpus_ids[idx])
                            self.search_space_filtered.append(image_id)
                            matched_count += 1

            print(f"Successfully matched {matched_count} images using CIRR metadata")

        # For FashionIQ, use metadata file for matching
        elif 'fashion-iq' in corpus_vectors_file.lower() and os.path.exists(metadata_file):
            use_metadata = True
            metadata = torch.load(metadata_file, map_location='cpu', weights_only=False)
            
            # FashionIQ metadata processing
            matched_count = 0
            search_space_set = set(self.search_space)  # For fast lookup
            
            if 'idx_to_info' in metadata:
                idx_to_info = metadata['idx_to_info']
                for idx, info in idx_to_info.items():
                    image_id = info.get('image_id', '')
                    image_path = info.get('image_path', '')
                    
                    # Check if in search space (based on product ID)
                    if image_id in search_space_set:
                        # Check if image file exists
                        file_exists = check_image_file_exists(image_id, image_path)

                        if file_exists and idx < len(all_corpus_ids):
                            valid_indices.append(idx)
                            self.corpus_ids.append(all_corpus_ids[idx])
                            self.search_space_filtered.append(image_id)  # Use product ID
                            matched_count += 1
            
            print(f"Successfully matched {matched_count} images using FashionIQ metadata")
        
        if not use_metadata:
            # Direct image ID matching (when metadata file is not available)
            print("Using direct image ID matching")
            for search_img_id in self.search_space:
                # First check if image file exists
                if '/' in search_img_id:
                    # For relative path format, extract filename part
                    base_id = search_img_id.split('/')[-1]
                    if '.' in base_id:
                        base_id = base_id.rsplit('.', 1)[0]
                else:
                    base_id = search_img_id.rsplit('.', 1)[0] if '.' in search_img_id else search_img_id
                
                if not check_image_file_exists(base_id):
                    continue  # Skip if image file does not exist

                # Create candidate list considering extension presence
                search_candidates = [search_img_id]
                
                # For CIRR, also add candidates with filename extracted from relative path
                if 'cirr' in corpus_vectors_file.lower() and '/' in search_img_id:
                    filename = search_img_id.split('/')[-1]
                    search_candidates.append(filename)
                    if '.' in filename:
                        search_candidates.append(filename.rsplit('.', 1)[0])
                
                if not search_img_id.endswith(('.png', '.jpg', '.jpeg')):
                    search_candidates.extend([f"{search_img_id}.png", f"{search_img_id}.jpg"])
                if search_img_id.endswith(('.png', '.jpg', '.jpeg')):
                    search_candidates.append(search_img_id.rsplit('.', 1)[0])
                
                # Match within feature file
                found_idx = None
                for candidate in search_candidates:
                    for idx, corpus_id in enumerate(all_corpus_ids):
                        if corpus_id == candidate:
                            found_idx = idx
                            break
                    if found_idx is not None:
                        break
                
                if found_idx is not None:
                    valid_indices.append(found_idx)
                    self.corpus_ids.append(all_corpus_ids[found_idx])
                    self.search_space_filtered.append(base_id)
        
        if not valid_indices:
            raise ValueError(f"No matching images found between search space and corpus. "
                           f"Search space sample: {self.search_space[:5]}, "
                           f"Corpus sample: {all_corpus_ids[:5]}")
        
        # Extract only features corresponding to search space
        self.corpus_features = all_corpus_features[valid_indices]
        
        print(f"Filtered corpus: {len(self.corpus_ids)} images (from {len(all_corpus_ids)} total)")
        
        self.model_manager = model_manager
    
    def search_images(self, query_features: torch.Tensor) -> List[Tuple[str, float]]:
        """Execute image search"""
        corpus_ids, corpus_features = self.corpus_ids, self.corpus_features
        
        # Normalize query features
        query_features = normalize(query_features, dim=-1)
        corpus_features = normalize(corpus_features, dim=-1)
        
        # Calculate cosine similarity
        similarities = (query_features @ corpus_features.T).squeeze(0).cpu().numpy()
        
        # Create search results (using filtered search space)
        image_similarities = [
            (self.search_space_filtered[index], similarities[index]) 
            for index in range(len(corpus_ids))
        ]
        
        # Sort by similarity
        images = sorted(image_similarities, key=lambda x: x[1], reverse=True)
        return images
    
    def get_image_features(self, image_id: str) -> torch.Tensor:
        """Get features for a specific image"""
        corpus_ids, corpus_features = self.corpus_ids, self.corpus_features
        
        if image_id not in self.search_space_filtered:
            raise ValueError(f"Image ID '{image_id}' not found in filtered search space. "
                           f"Available images: {len(self.search_space_filtered)}")
        
        index = self.search_space_filtered.index(image_id)
        return corpus_features[index].clone().detach().to(device)
    
    def update_query_with_feedback(self, current_query: torch.Tensor, selected_img: str,
                                 unselected_imgs: List[str], new_text_query: str,
                                 alpha: float = 0.08, beta: float = 0.29, gamma: float = 0.44) -> torch.Tensor:
        """Update query using feedback"""
        with torch.no_grad():
            text_features = self.model_manager.dialog_encoder(new_text_query)
        
        # Normalize features
        text_features = normalize(text_features, dim=-1)
        current_query = normalize(current_query, dim=-1)

        # Features of selected image
        selected_features = normalize(self.get_image_features(selected_img), dim=-1)
        
        # Features of unselected images
        unselected_features = torch.stack([
            self.get_image_features(img_id) for img_id in unselected_imgs
        ])
        unselected_features = normalize(unselected_features, dim=-1)
        
        # Update query
        updated_query = (
            current_query + 
            gamma * text_features + 
            alpha * selected_features - 
            beta * torch.mean(unselected_features, dim=0)
        )
        
        return normalize(updated_query, dim=-1)

class MultiTurnCIRSystem:
    """Main class for multi-turn CIR system"""

    def __init__(self, dataset_name: str, config: Dict[str, Any]):
        """Initialize"""
        self.dataset_name = dataset_name
        self.config = config
        self.max_turns = config.get('max_turns', 5)
        self.results = []
        self.completeness_info = {}
        
        # Log max_turns setting
        print(f"Initializing {dataset_name} with max_turns = {self.max_turns}")

        # For CIRCO, create hash → image filename mapping
        self.hash_to_filename = {}
        if dataset_name == 'circo':
            import torch
            metadata_file = 'CIRCO/metadata_blip.pt'
            metadata = torch.load(metadata_file, map_location='cpu', weights_only=False)
            hash_to_idx = metadata['hash_to_idx']
            idx_to_info = metadata['idx_to_info']
            
            for hash_val, idx in hash_to_idx.items():
                info = idx_to_info.get(idx, {})
                image_id = info.get('image_id', '')
                if image_id:
                    filename = f"{int(image_id):012d}.jpg"
                    self.hash_to_filename[hash_val] = filename
            
            print(f"Created hash to filename mapping for {len(self.hash_to_filename)} images")
        
        # Initialize image ID → hash value mapping
        self.image_id_to_hash = {}
        self._initialize_image_hash_mapping()
        
        # Initialize model manager and retrieval engine
        self.model_manager = ModelManager(use_blip=config.get('use_blip', True))
        self.retrieval_engine = RetrievalEngine(
            config['corpus_vectors_file'], 
            config['search_space_file'], 
            self.model_manager
        )
        
        # Load data and caption features
        self.data = self.load_dataset()
        self.reference_captions = self.load_reference_captions()
        self.caption_features = self.load_caption_features()
        
        # Load existing results (for resume mode)
        self.load_existing_results()
    
    def _initialize_image_hash_mapping(self):
        """Initialize image ID → hash value mapping"""
        print("Initializing image ID to hash mapping...")
        
        if self.dataset_name == 'circo':
            # For CIRCO, create reverse mapping from existing hash_to_filename
            for hash_val, filename in self.hash_to_filename.items():
                # Extract image ID from filename (e.g., 000000000001.jpg → 1)
                image_id = filename.replace('.jpg', '').lstrip('0') or '0'
                self.image_id_to_hash[image_id] = hash_val
                # Also add 12-digit format
                padded_id = f"{int(image_id):012d}"
                self.image_id_to_hash[padded_id] = hash_val
        
        elif self.dataset_name.startswith('cirr'):
            # For CIRR, compute hash from actual image files
            self._compute_cirr_image_hashes()
        
        elif self.dataset_name.startswith('fashioniq'):
            # For FashionIQ, compute hash from actual image files
            self._compute_fashioniq_image_hashes()
        
        print(f"Created image ID to hash mapping for {len(self.image_id_to_hash)} images")
    
    def _compute_image_hash(self, image_path: str) -> str:
        """Compute MD5 hash of image file"""
        try:
            with open(image_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            print(f"Warning: Failed to compute hash for {image_path}: {e}")
            return None
    
    def _compute_cirr_image_hashes(self):
        """Compute hash values for CIRR images"""
        image_dir = self.config.get('image_dir', 'cirr/img_raw')

        # Get image IDs from split file
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
    
    def _compute_fashioniq_image_hashes(self):
        """Compute hash values for FashionIQ images"""
        image_dir = self.config.get('image_dir', 'fashion-iq/images')
        category = self.dataset_name.split('_')[1] if '_' in self.dataset_name else 'dress'

        # Get image IDs from split file
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
                            # Also add version with extension
                            self.image_id_to_hash[f"{image_id}.jpg"] = img_hash
    
    def get_image_hash(self, image_id: str) -> str:
        """Get hash value from image ID"""
        # Get from direct mapping
        if image_id in self.image_id_to_hash:
            return self.image_id_to_hash[image_id]
        
        # For FashionIQ, consider presence of extension
        if self.dataset_name.startswith('fashioniq'):
            if image_id.endswith('.jpg'):
                base_id = image_id[:-4]
                if base_id in self.image_id_to_hash:
                    return self.image_id_to_hash[base_id]
            else:
                jpg_id = f"{image_id}.jpg"
                if jpg_id in self.image_id_to_hash:
                    return self.image_id_to_hash[jpg_id]
        
        # For CIRCO, also try 12-digit format
        if self.dataset_name == 'circo':
            try:
                padded_id = f"{int(image_id):012d}"
                if padded_id in self.image_id_to_hash:
                    return self.image_id_to_hash[padded_id]
            except ValueError:
                pass
        
        return None
    
    def load_existing_results(self) -> None:
        """Load existing results file for resume preparation"""
        output_dir = self.config.get('output_dir', '.')
        output_file = os.path.join(output_dir, f"multiturn_cir_results_{self.dataset_name}.json")
        
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    existing_data = json.load(f)
                
                if 'results' in existing_data:
                    self.results = existing_data['results']
                    
                    # For Fashion-IQ with caption_mode=separate, match using triplets
                    if self.dataset_name.startswith('fashioniq') and self.config.get('caption_mode') == 'separate':
                        processed_query_triplets = set()
                        
                        # Manage processed queries by (reference_id, target_id, relative_caption) combination
                        for result in self.results:
                            ref_id = result.get('reference_image_id', '')
                            target_id = result.get('target_image_id', '')
                            original_query = result.get('original_query', '')  # relative_caption
                            
                            if ref_id and target_id and original_query:
                                processed_query_triplets.add((ref_id, target_id, original_query))
                        
                        print(f"Found existing results file with {len(self.results)} processed queries")
                        print(f"Processed query triplets (ref, target, caption): {len(processed_query_triplets)}")
                        
                        # Keep only unprocessed queries
                        original_data_count = len(self.data)
                        remaining_data = []

                        for query_item in self.data:
                            # Get current query's (reference_id, target_id, relative_caption) triplet
                            ref_id = query_item['reference_image_id']
                            target_id = query_item['target_image_id']
                            relative_caption = query_item['relative_caption']
                            query_triplet = (ref_id, target_id, relative_caption)
                            
                            # Keep only if this triplet has not been processed
                            if query_triplet not in processed_query_triplets:
                                remaining_data.append(query_item)
                        
                        self.data = remaining_data
                        
                        print(f"Resume mode (separate caption mode): {original_data_count - len(remaining_data)} queries already processed")
                        print(f"Remaining queries to process: {len(remaining_data)}")
                        
                    elif self.dataset_name == 'circo':
                        # Special handling for CIRCO: match by reference_id + multiple ground_truths
                        processed_query_signatures = set()
                        
                        # Identify processed queries from saved results
                        for result in self.results:
                            # Since hash values are converted to 12-digit IDs when saving,
                            # match with original_*_id in current data
                            ref_id = result.get('reference_image_id', '')  # 12-digit ID format
                            gt_ids = result.get('ground_truth_ids', [])    # List in 12-digit ID format
                            
                            if ref_id and gt_ids:
                                # Sort and convert ground_truth_ids to tuple (create order-independent unique identifier)
                                gt_ids_sorted = tuple(sorted(gt_ids))
                                query_signature = (ref_id, gt_ids_sorted)
                                processed_query_signatures.add(query_signature)
                        
                        print(f"Found existing results file with {len(self.results)} processed queries")
                        print(f"Processed query signatures (CIRCO - ref+GTs): {len(processed_query_signatures)}")
                        
                        # Keep only unprocessed queries
                        original_data_count = len(self.data)
                        remaining_data = []

                        for query_item in self.data:
                            # Match using original_*_id in current data
                            original_ref_id = query_item.get('original_reference_id', '')
                            original_gt_ids = query_item.get('original_gt_ids', [])
                            
                            # Remove extension to get 12-digit numeric ID
                            if original_ref_id.endswith('.jpg'):
                                original_ref_id = original_ref_id[:-4]
                            
                            # Also remove extension from ground truths
                            normalized_gt_ids = []
                            for gt_id in original_gt_ids:
                                if gt_id.endswith('.jpg'):
                                    normalized_gt_ids.append(gt_id[:-4])
                                else:
                                    normalized_gt_ids.append(gt_id)
                            
                            # Sort ground_truth_ids and convert to tuple
                            gt_ids_sorted = tuple(sorted(normalized_gt_ids))
                            query_signature = (original_ref_id, gt_ids_sorted)
                            
                            # Keep only if this signature has not been processed
                            if query_signature not in processed_query_signatures:
                                remaining_data.append(query_item)
                        
                        self.data = remaining_data
                        
                        print(f"Resume mode (CIRCO with GTs): {original_data_count - len(remaining_data)} queries already processed")
                        print(f"Remaining queries to process: {len(remaining_data)}")
                        
                        # For debugging: show sample processed queries
                        if processed_query_signatures and len(processed_query_signatures) <= 5:
                            print("Sample processed query signatures:")
                            for i, sig in enumerate(list(processed_query_signatures)[:3]):
                                ref_id, gt_tuple = sig
                                print(f"  {i+1}. ref: {ref_id}, GTs: {gt_tuple}")
                        elif len(processed_query_signatures) > 5:
                            print(f"Large number of processed signatures ({len(processed_query_signatures)}), showing first 3:")
                            for i, sig in enumerate(list(processed_query_signatures)[:3]):
                                ref_id, gt_tuple = sig
                                print(f"  {i+1}. ref: {ref_id}, GTs: {gt_tuple}")
                        
                    else:
                        # Traditional method: match by (reference_id, target_id) pair
                        processed_query_pairs = set()
                        
                        # Manage processed queries by (reference_id, target_id) combination
                        for result in self.results:
                            # Get reference_image_id and target_image_id combination from result
                            ref_id = result.get('reference_image_id', '')
                            target_id = result.get('target_image_id', '')
                            
                            if ref_id and target_id:
                                processed_query_pairs.add((ref_id, target_id))
                        
                        print(f"Found existing results file with {len(self.results)} processed queries")
                        print(f"Processed query pairs: {len(processed_query_pairs)}")
                        
                        # Keep only unprocessed queries
                        original_data_count = len(self.data)
                        remaining_data = []

                        for query_item in self.data:
                            # Get current query's (reference_id, target_id) pair
                            ref_id = query_item['reference_image_id']
                            target_id = query_item['target_image_id']
                            query_pair = (ref_id, target_id)
                            
                            # Keep only if this pair has not been processed
                            if query_pair not in processed_query_pairs:
                                remaining_data.append(query_item)
                        
                        self.data = remaining_data
                        
                        print(f"Resume mode: {original_data_count - len(remaining_data)} queries already processed")
                        print(f"Remaining queries to process: {len(remaining_data)}")
                    
                    if len(self.data) == 0:
                        print("All queries have been processed! No remaining work.")
                    
                else:
                    print("Existing results file found but no 'results' key - starting fresh")
                    
            except Exception as e:
                print(f"Warning: Failed to load existing results from {output_file}: {e}")
                print("Starting fresh evaluation")
        else:
            print(f"No existing results file found ({output_file}) - starting fresh evaluation")
    
    def load_dataset(self) -> List[Dict]:
        """Load dataset"""
        if self.dataset_name == 'circo':
            return DatasetLoader.load_circo_data(self.config['annotation_file'])
        elif self.dataset_name == 'cirr_train':
            return DatasetLoader.load_cirr_data(self.config['annotation_file'])
        elif self.dataset_name == 'cirr_val':
            return DatasetLoader.load_cirr_data(self.config['annotation_file'])
        elif self.dataset_name.startswith('fashioniq'):
            return DatasetLoader.load_fashioniq_data(self.config['annotation_file'], self.config['caption_mode'])
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def load_reference_captions(self) -> Dict[str, str]:
        """Load pre-computed captions"""
        captions = {}
        
        # If caption file is specified in config
        if 'caption_file' in self.config and os.path.exists(self.config['caption_file']):
            with open(self.config['caption_file'], 'r') as f:
                caption_data = json.load(f)
                
            # Process according to caption data format
            if isinstance(caption_data, dict):
                captions = caption_data
            elif isinstance(caption_data, list):
                # For list format, map with appropriate keys
                for item in caption_data:
                    if 'image_id' in item and 'caption' in item:
                        captions[item['image_id']] = item['caption']
        
        # Load gpt4omini caption file (PyTorch format)
        if 'gpt4omini_caption_features_file' in self.config and os.path.exists(self.config['gpt4omini_caption_features_file']):
            try:
                features_data = torch.load(self.config['gpt4omini_caption_features_file'], map_location='cpu', weights_only=False)
                if isinstance(features_data, dict):
                    # Get dictionary directly from PyTorch file
                    captions.update(features_data)
                elif hasattr(features_data, 'items'):
                    # Other dictionary formats
                    captions.update(dict(features_data.items()))
            except Exception as e:
                print(f"Warning: Failed to load GPT-4o-mini captions from {self.config['gpt4omini_caption_features_file']}: {e}")
        
        # Dataset-specific processing
        if self.dataset_name == 'circo':
            # For CIRCO, COCO captions can also be used
            coco_caption_file = 'CIRCO/annotations/captions_val2017.json'
            if os.path.exists(coco_caption_file):
                with open(coco_caption_file, 'r') as f:
                    coco_captions = json.load(f)
                
                for ann in coco_captions.get('annotations', []):
                    img_id = f"{ann['image_id']:06d}.jpg"
                    if img_id not in captions:
                        captions[img_id] = ann['caption']
            
        elif self.dataset_name.startswith('fashioniq'):
            # For Fashion-IQ, pre-generated caption file is required
            if not captions:
                raise ValueError(f"No captions loaded for Fashion-IQ dataset. "
                               f"Caption file: {self.config.get('caption_file', 'Not specified')} "
                               f"GPT-4o-mini caption file: {self.config.get('gpt4omini_caption_features_file', 'Not specified')} "
                               f"Please ensure at least one caption file exists and contains valid data.")
        
        return captions
    
    def get_reference_caption(self, reference_image_id: str) -> str:
        """Get caption for reference image"""
        if reference_image_id in self.reference_captions:
            return self.reference_captions[reference_image_id]
        
        # Try full path format key (unified for all datasets)
        potential_keys = []
        
        if self.dataset_name.startswith('fashioniq'):
            # Fashion-IQ: fashion-iq/images/{category}/{image_id}
            category = self.dataset_name.split('_')[1]  # dress, shirt, toptee
            
            # Candidates considering extension presence
            base_key = f"fashion-iq/images/{category}/{reference_image_id}"
            potential_keys.append(base_key)
            
            # If no extension, also try with .jpg
            if not reference_image_id.endswith('.jpg'):
                potential_keys.append(f"fashion-iq/images/{category}/{reference_image_id}.jpg")
            # If has extension, also try without
            elif reference_image_id.endswith('.jpg'):
                base_id = reference_image_id[:-4]
                potential_keys.append(f"fashion-iq/images/{category}/{base_id}")
            
        elif self.dataset_name.startswith('cirr'):
            # CIRR: Search matching actual caption file structure
            # Captions are stored in full path format
            
            # Basic key candidates
            potential_keys.extend([
                reference_image_id,
                f"{reference_image_id}.png",
            ])
            
            # Full path format key candidates (actual storage format)
            # Train: Hierarchical structure (0-99 subdirectories)
            for subdir in range(100):
                potential_keys.extend([
                    f"cirr/img_raw/train/{subdir}/{reference_image_id}",
                    f"cirr/img_raw/train/{subdir}/{reference_image_id}.png",
                ])
            
            # Dev/Val: Flat structure
            potential_keys.extend([
                f"cirr/img_raw/dev/{reference_image_id}",
                f"cirr/img_raw/dev/{reference_image_id}.png",
                f"cirr/img_raw/val/{reference_image_id}",
                f"cirr/img_raw/val/{reference_image_id}.png",
            ])
        
        elif self.dataset_name == 'circo':
            # CIRCO: Convert hash value to actual filename and search for caption
            if reference_image_id in self.hash_to_filename:
                filename = self.hash_to_filename[reference_image_id]
                potential_keys.extend([
                    f"CIRCO/unlabeled2017/{filename}",
                    filename
                ])
            else:
                # If not hash value (12-digit format), use as-is
                potential_keys.extend([
                    f"CIRCO/unlabeled2017/{reference_image_id}",
                    f"CIRCO/unlabeled2017/{reference_image_id}.jpg"
                ])
        
        # Search with candidate keys
        for key in potential_keys:
            if key in self.reference_captions:
                return self.reference_captions[key]
        
        # Error if not found
        raise KeyError(f"Caption not found for image: {reference_image_id}. "
                      f"Available captions: {len(self.reference_captions)} images. "
                      f"Check caption file: {self.config.get('caption_file', 'Not specified')}. "
                      f"Tried keys: {potential_keys[:3]}... "
                      f"Sample actual keys: {list(self.reference_captions.keys())[:3]}")
    
    def check_success(self, ground_truth_ids: List[str], search_results: List[Tuple[str, float]], k: int = 10) -> bool:
        """Check success (whether ground truth is in top-k)"""
        if not ground_truth_ids:
            raise ValueError("Ground truth IDs list is empty")
        if not search_results:
            raise ValueError("Search results list is empty")
            
        top_k_ids = [result[0] for result in search_results[:k]]
        return any(gt_id in top_k_ids for gt_id in ground_truth_ids)
    
    def find_most_similar_to_gt(self, ground_truth_ids: List[str], search_results: List[Tuple[str, float]], 
                               selected_images: set = None) -> str:
        """Select image most similar to ground truth from search results (only when GT is not in top10)"""
        if not ground_truth_ids:
            raise ValueError("Ground truth IDs list is empty")
        if not search_results:
            raise ValueError("Search results list is empty")
            
        top_k_ids = [r[0] for r in search_results[:10]]
        
        # Initialize empty set if selected_images is None
        if selected_images is None:
            selected_images = set()
        
        # Create set of hash values for selected images
        selected_hashes = set()
        for img_id in selected_images:
            img_hash = self.get_image_hash(img_id)
            if img_hash:
                selected_hashes.add(img_hash)
        
        # Get unselected candidate images (check duplicates by hash value)
        available_images = []
        for img_id in top_k_ids:
            img_hash = self.get_image_hash(img_id)
            if img_hash and img_hash not in selected_hashes:
                available_images.append(img_id)
            elif not img_hash:
                # If hash value cannot be obtained, use ID as before
                if img_id not in selected_images:
                    available_images.append(img_id)
        
        if not available_images:
            raise ValueError(f"No available images for selection. "
                           f"Top-k IDs: {top_k_ids}, Selected images: {len(selected_images)}, "
                           f"Selected hashes: {len(selected_hashes)}")
        
        if self.dataset_name == 'circo':
            # For CIRCO: Use GT with highest search rank as caption similarity reference
            # First check search rank for all GTs and identify the top-ranked GT
            best_gt_position = len(search_results)
            best_gt_id = None
            
            for gt_id in ground_truth_ids:
                try:
                    # Get rank in all search results
                    all_ids = [r[0] for r in search_results]
                    position = all_ids.index(gt_id)
                    if position < best_gt_position:
                        best_gt_position = position
                        best_gt_id = gt_id
                except ValueError:
                    # Skip if GT is not in search results
                    continue
            
            if best_gt_id is None:
                raise ValueError(f"No ground truth images found in search results for CIRCO. "
                               f"GT IDs: {ground_truth_ids}, Search results: {[r[0] for r in search_results[:20]]}")
            
            # Get caption features for the top-ranked GT
            if best_gt_id in self.caption_features:
                gt_caption_features = self.caption_features[best_gt_id].to(device)
                gt_caption_features = normalize(gt_caption_features.unsqueeze(0), dim=-1)
            else:
                # Clear error if caption features not found
                raise KeyError(f"Caption features not found for best GT image '{best_gt_id}' in CIRCO. "
                             f"Available caption features: {len(self.caption_features)} images. "
                             f"Check gpt4omini_caption_features_file: {self.config.get('gpt4omini_caption_features_file', 'Not specified')}")
            
            # Calculate caption features and similarity for each candidate image
            best_similarity = -1.0
            best_image = None
            similarities_computed = 0
            
            for img_id in available_images:
                if img_id in self.caption_features:
                    # Use pre-computed features
                    img_caption_features = self.caption_features[img_id].to(device)
                    img_caption_features = normalize(img_caption_features.unsqueeze(0), dim=-1)
                    
                    # Calculate cosine similarity
                    similarity = torch.cosine_similarity(gt_caption_features, img_caption_features, dim=-1).item()
                    similarities_computed += 1
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_image = img_id
                else:
                    # Only warn if caption features not found (continue computation)
                    print(f"Warning: Caption features not found for candidate image '{img_id}' in CIRCO")
                    continue
            
            if best_image is None:
                raise ValueError(f"No suitable image found for selection in CIRCO. "
                               f"Available images: {len(available_images)}, "
                               f"Similarities computed: {similarities_computed}, "
                               f"Best GT ID: {best_gt_id}")
            
            return best_image
        
        elif self.dataset_name.startswith('cirr'):
            # For CIRR: Select image closest to GT using caption features
            # Get caption features for ground truth
            gt_id = ground_truth_ids[0]  # CIRR has single GT
            if gt_id in self.caption_features:
                gt_caption_features = self.caption_features[gt_id]
                # Convert to tensor if numpy array
                if isinstance(gt_caption_features, np.ndarray):
                    gt_caption_features = torch.from_numpy(gt_caption_features).float()
                gt_caption_features = gt_caption_features.to(device)
                gt_caption_features = normalize(gt_caption_features.unsqueeze(0), dim=-1)
            else:
                raise KeyError(f"Caption features not found for GT image '{gt_id}' in CIRR. "
                             f"Available caption features: {len(self.caption_features)} images. "
                             f"Check gpt4omini_caption_features_file: {self.config.get('gpt4omini_caption_features_file', 'Not specified')}")
            
            # Calculate caption features and similarity for each candidate image
            best_similarity = -1.0
            best_image = None
            similarities_computed = 0
            
            for img_id in available_images:
                if img_id in self.caption_features:
                    # Use pre-computed features
                    img_caption_features = self.caption_features[img_id]
                    # Convert to tensor if numpy array
                    if isinstance(img_caption_features, np.ndarray):
                        img_caption_features = torch.from_numpy(img_caption_features).float()
                    img_caption_features = img_caption_features.to(device)
                    img_caption_features = normalize(img_caption_features.unsqueeze(0), dim=-1)
                    
                    # Calculate cosine similarity
                    similarity = torch.cosine_similarity(gt_caption_features, img_caption_features, dim=-1).item()
                    similarities_computed += 1
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_image = img_id
                else:
                    # Only warn if caption features not found (continue computation)
                    print(f"Warning: Caption features not found for candidate image '{img_id}' in CIRR")
                    continue
            
            if best_image is None:
                raise ValueError(f"No suitable image found for selection in CIRR. "
                               f"Available images: {len(available_images)}, "
                               f"Similarities computed: {similarities_computed}, "
                               f"GT ID: {gt_id}")
            
            return best_image
        
        elif self.dataset_name.startswith('fashioniq'):
            # For FashionIQ: Select image closest to GT using caption features
            # Get caption features for ground truth
            gt_id = ground_truth_ids[0]  # FashionIQ has single GT
            
            # For FashionIQ, search GT caption features considering extension presence
            gt_caption_features = None
            
            # Candidate 1: Original ID
            if gt_id in self.caption_features:
                gt_caption_features = self.caption_features[gt_id]
            # Candidate 2: With .jpg extension
            elif f"{gt_id}.jpg" in self.caption_features:
                gt_caption_features = self.caption_features[f"{gt_id}.jpg"]
            # Candidate 3: Without extension (remove extension from original ID)
            elif gt_id.endswith('.jpg') and gt_id[:-4] in self.caption_features:
                gt_caption_features = self.caption_features[gt_id[:-4]]
            
            if gt_caption_features is not None:
                # Convert to tensor if numpy array
                if isinstance(gt_caption_features, np.ndarray):
                    gt_caption_features = torch.from_numpy(gt_caption_features).float()
                gt_caption_features = gt_caption_features.to(device)
                gt_caption_features = normalize(gt_caption_features.unsqueeze(0), dim=-1)
            else:
                raise KeyError(f"Caption features not found for GT image '{gt_id}' in FashionIQ. "
                             f"Available caption features: {len(self.caption_features)} images. "
                             f"Check gpt4omini_caption_features_file: {self.config.get('gpt4omini_caption_features_file', 'Not specified')}")
            
            # Calculate caption features and similarity for each candidate image
            best_similarity = -1.0
            best_image = None
            similarities_computed = 0
            
            for img_id in available_images:
                # For FashionIQ, search considering extension presence
                caption_features = None
                
                # Candidate 1: Original ID
                if img_id in self.caption_features:
                    caption_features = self.caption_features[img_id]
                # Candidate 2: With .jpg extension
                elif f"{img_id}.jpg" in self.caption_features:
                    caption_features = self.caption_features[f"{img_id}.jpg"]
                # Candidate 3: Without extension (remove extension from original ID)
                elif img_id.endswith('.jpg') and img_id[:-4] in self.caption_features:
                    caption_features = self.caption_features[img_id[:-4]]
                
                if caption_features is not None:
                    # Use pre-computed features
                    img_caption_features = caption_features
                    # Convert to tensor if numpy array
                    if isinstance(img_caption_features, np.ndarray):
                        img_caption_features = torch.from_numpy(img_caption_features).float()
                    img_caption_features = img_caption_features.to(device)
                    img_caption_features = normalize(img_caption_features.unsqueeze(0), dim=-1)
                    
                    # Calculate cosine similarity
                    similarity = torch.cosine_similarity(gt_caption_features, img_caption_features, dim=-1).item()
                    similarities_computed += 1
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_image = img_id
                else:
                    # Only warn if caption features not found (continue computation)
                    print(f"Warning: Caption features not found for candidate image '{img_id}' in FashionIQ")
                    continue
            
            if best_image is None:
                raise ValueError(f"No suitable image found for selection in FashionIQ. "
                               f"Available images: {len(available_images)}, "
                               f"Similarities computed: {similarities_computed}, "
                               f"GT ID: {gt_id}")
            
            return best_image
        
        else:
            # For other datasets, select first available image
            return available_images[0]
    
    def get_gt_rankings(self, ground_truth_ids: List[str], search_results: List[Tuple[str, float]]) -> Dict[str, int]:
        """Get ground truth rankings (1-indexed)"""
        gt_rankings = {}
        result_ids = [result[0] for result in search_results]
        
        for gt_id in ground_truth_ids:
            try:
                # Record rank as 1-indexed
                rank = result_ids.index(gt_id) + 1
                gt_rankings[gt_id] = rank
            except ValueError:
                # -1 if GT is not in search results
                gt_rankings[gt_id] = -1
        
        return gt_rankings
    
    def get_best_gt_rank(self, gt_rankings: Dict[str, int]) -> int:
        """Get the best (smallest) GT rank"""
        valid_ranks = [rank for rank in gt_rankings.values() if rank > 0]
        return min(valid_ranks) if valid_ranks else -1
    
    async def process_single_query(self, query_item: Dict) -> Dict:
        """Process single query"""
        # Set query ID uniformly (original method)
        query_id = query_item.get('id')
        if query_id is None:
            query_id = query_item.get('query_id')
        if query_id is None:
            # Fallback: use reference_image_id
            query_id = query_item.get('reference_image_id', 'unknown')
        
        # Convert numeric ID to string
        if isinstance(query_id, int):
            query_id = str(query_id)
        
        results = {
            'query_id': query_id,
            'reference_image_id': query_item['reference_image_id'],
            'target_image_id': query_item['target_image_id'],
            'ground_truth_ids': query_item['ground_truth_ids'],
            'original_query': query_item['relative_caption'],
            'success': False,
            'success_turn': -1,
            'turns': []
        }
        
        # Record selected images (add reference_image first to prevent duplicates)
        selected_images = set()
        selected_images.add(query_item['reference_image_id'])  # Prevent duplicates with reference_image
        previous_relative_captions = []
        
        # Get reference caption
        reference_caption = self.get_reference_caption(query_item['reference_image_id'])
        
        # Turn 0: Initial search
        initial_combined_caption = await self.model_manager.combine_captions_with_gpt4o(
            reference_caption, query_item['relative_caption']
        )
        
        with torch.no_grad():
            current_query_features = self.model_manager.dialog_encoder(initial_combined_caption)
        
        search_results = self.retrieval_engine.search_images(current_query_features)
        
        # Get GT rank
        gt_rankings = self.get_gt_rankings(query_item['ground_truth_ids'], search_results)
        best_gt_rank = self.get_best_gt_rank(gt_rankings)
        
        turn_result = {
            'turn': 0,
            'query_text': initial_combined_caption,
            'search_results': search_results[:10],
            'selected_image': None,
            'selected_image_caption': None,  # No selected image in turn 0
            'relative_caption': None,
            'gt_rankings': gt_rankings,
            'best_gt_rank': best_gt_rank
        }
        results['turns'].append(turn_result)
        
        # Success check
        if self.check_success(query_item['ground_truth_ids'], search_results):
            results['success'] = True
            results['success_turn'] = 0
            return results
        
        # Multi-turn processing
        for turn in range(1, self.max_turns + 1):
            # Check if GT is in top10
            if self.check_success(query_item['ground_truth_ids'], search_results):
                # Early termination if GT is in top10
                results['success'] = True
                results['success_turn'] = turn - 1  # Success in previous turn
                break
            
            # Select most similar image
            if self.dataset_name == 'circo':
                # For CIRCO: Select GT with highest search rank
                selected_image = self.find_most_similar_to_gt(query_item['ground_truth_ids'], search_results, selected_images)
            else:
                # For other datasets (CIRR, FashionIQ), select considering caption similarity
                selected_image = self.find_most_similar_to_gt(
                    query_item['ground_truth_ids'], search_results, selected_images
                )
            
            selected_images.add(selected_image)
            
            # Get caption for selected image
            try:
                selected_image_caption = self.get_reference_caption(selected_image)
            except KeyError:
                selected_image_caption = f"Caption not found for {selected_image}"
            
            # Generate relative caption with GPT-4o-mini
            # From turn 1 onwards, compare selected image with ground truth image
            if self.dataset_name.startswith('fashioniq'):
                # For Fashion-IQ, include category subdirectory and add extension
                category = self.dataset_name.split('_')[1]  # dress, shirt, toptee
                
                # Auto-add extension
                selected_img_with_ext = selected_image if selected_image.endswith('.jpg') else f"{selected_image}.jpg"
                target_img_with_ext = query_item['target_image_id'] if query_item['target_image_id'].endswith('.jpg') else f"{query_item['target_image_id']}.jpg"
                
                selected_image_path = os.path.join(self.config['image_dir'], category, selected_img_with_ext)
                target_image_path = os.path.join(self.config['image_dir'], category, target_img_with_ext)
            elif self.dataset_name.startswith('cirr'):
                # For CIRR, search for correct path using the same logic as RetrievalEngine
                def find_cirr_image_path(image_id: str) -> str:
                    """Search for CIRR image path from hierarchical structure"""
                    # For train: hierarchical structure (0-99 subdirectories)
                    for subdir in range(100):  # Search 0-99 subdirectories
                        for ext in ['.png', '.jpg', '.jpeg']:
                            image_path = os.path.join(self.config['image_dir'], 'train', str(subdir), f"{image_id}{ext}")
                            if os.path.exists(image_path):
                                return image_path
                    
                    # For val, dev, test1: Flat structure
                    for split in ['val', 'dev', 'test1']:
                        for ext in ['.png', '.jpg', '.jpeg']:
                            image_path = os.path.join(self.config['image_dir'], split, f"{image_id}{ext}")
                            if os.path.exists(image_path):
                                return image_path
                    
                    raise FileNotFoundError(f"CIRR image not found: {image_id}")
                
                selected_image_path = find_cirr_image_path(selected_image)
                target_image_path = find_cirr_image_path(query_item['target_image_id'])
            else:
                # For other datasets (CIRCO), convert hash value to actual filename
                def get_circo_image_path(image_id: str) -> str:
                    if image_id in self.hash_to_filename:
                        filename = self.hash_to_filename[image_id]
                        return os.path.join(self.config['image_dir'], filename)
                    else:
                        # Use as-is if not hash value
                        return os.path.join(self.config['image_dir'], image_id)
                
                selected_image_path = get_circo_image_path(selected_image)
                target_image_path = get_circo_image_path(query_item['target_image_id'])
            
            # Check file existence
            if not os.path.exists(selected_image_path):
                raise FileNotFoundError(f"Selected image not found: {selected_image_path}")
            if not os.path.exists(target_image_path):
                raise FileNotFoundError(f"Target image not found: {target_image_path}")
            
            # Generate relative caption with GPT-4o-mini (transform selected image → ground truth image)
            new_relative_caption = await self.model_manager.generate_relative_caption_with_gpt4o(
                selected_image_path, target_image_path, previous_relative_captions,
                similarity_threshold=0.8,  # CLIP similarity threshold
                max_retries=3  # Maximum retry count
            )
            
            previous_relative_captions.append(new_relative_caption)
            
            # Generate new query
            new_combined_caption = await self.model_manager.combine_captions_with_gpt4o(
                selected_image_caption, new_relative_caption
            )
            
            # Update query
            unselected_images = [r[0] for r in search_results[:10] if r[0] != selected_image]
            current_query_features = self.retrieval_engine.update_query_with_feedback(
                current_query_features, selected_image, unselected_images, new_combined_caption
            )
            
            # Re-search
            search_results = self.retrieval_engine.search_images(current_query_features)
            
            # Get GT rank
            gt_rankings = self.get_gt_rankings(query_item['ground_truth_ids'], search_results)
            best_gt_rank = self.get_best_gt_rank(gt_rankings)
            
            turn_result = {
                'turn': turn,
                'query_text': new_combined_caption,
                'search_results': search_results[:10],
                'selected_image': selected_image,
                'selected_image_caption': selected_image_caption,  # Caption for selected image
                'relative_caption': new_relative_caption,
                'gt_rankings': gt_rankings,
                'best_gt_rank': best_gt_rank
            }
            results['turns'].append(turn_result)
            
            # Success check after re-search
            if self.check_success(query_item['ground_truth_ids'], search_results):
                results['success'] = True
                results['success_turn'] = turn
                break
            
            # Brief wait
            time.sleep(0.5)
        
        return results
    
    def run_evaluation(self) -> None:
        """Run evaluation"""
        # Display resume mode information
        already_processed = len(self.results)
        if already_processed > 0:
            print(f"=== RESUME MODE ===")
            print(f"Already processed: {already_processed} queries")
            print(f"Remaining to process: {len(self.data)} queries")
            print(f"Total queries in dataset: {already_processed + len(self.data)}")
            print(f"==================")
        else:
            print(f"Starting fresh evaluation on {self.dataset_name}")
        
        print(f"Processing {len(self.data)} queries for {self.dataset_name}")
        
        # Run data completeness check
        self.completeness_info = self.check_data_completeness()
        
        # Set image directory
        image_dir = self.config.get('image_dir', '')
        
        def check_image_exists(image_id: str) -> bool:
            """Check if image file exists"""
            if self.dataset_name.startswith('fashioniq'):
                # Fashion-IQ: Include category subdirectory and auto-add extension
                category = self.dataset_name.split('_')[1]  # dress, shirt, toptee
                img_with_ext = image_id if image_id.endswith('.jpg') else f"{image_id}.jpg"
                image_path = os.path.join(image_dir, category, img_with_ext)
                return os.path.exists(image_path)
            elif self.dataset_name.startswith('cirr'):
                # For CIRR: structure differs by split
                # train: hierarchical structure (0-99 subdirectories)
                # val, dev, test1: flat structure (files directly under directory)
                
                # For train: hierarchical structure
                for subdir in range(100):  # Search 0-99 subdirectories
                    for ext in ['.jpg', '.jpeg', '.png']:
                        image_path = os.path.join(image_dir, 'train', str(subdir), f"{image_id}{ext}")
                        if os.path.exists(image_path):
                            return True
                
                # For val, dev, test1: Flat structure
                for split in ['val', 'dev', 'test1']:
                    for ext in ['.jpg', '.jpeg', '.png']:
                        image_path = os.path.join(image_dir, split, f"{image_id}{ext}")
                        if os.path.exists(image_path):
                            return True
                
                return False
            else:
                # CIRCO: Convert hash value to actual filename
                if image_id in self.hash_to_filename:
                    filename = self.hash_to_filename[image_id]
                    image_path = os.path.join(image_dir, filename)
                    return os.path.exists(image_path)
                else:
                    # If not hash value (12-digit format), use as-is
                    image_path = os.path.join(image_dir, image_id)
                    return os.path.exists(image_path)
        
        skipped_queries = 0
        processed_queries = 0
        
        async def process_queries():
            nonlocal skipped_queries, processed_queries
            
            for query_item in tqdm(self.data, desc=f"Processing {self.dataset_name} queries"):
                # Check if Ground Truth image files exist
                gt_ids = query_item['ground_truth_ids']
                missing_gt_files = []
                
                for gt_id in gt_ids:
                    if not check_image_exists(gt_id):
                        missing_gt_files.append(gt_id)
                
                # Check if Reference image file exists
                ref_id = query_item['reference_image_id']
                missing_ref_file = not check_image_exists(ref_id)
                
                # Check if Target image file exists
                target_id = query_item['target_image_id']
                missing_target_file = not check_image_exists(target_id)
                
                # Skip if image files are missing
                if missing_gt_files or missing_ref_file or missing_target_file:
                    skipped_queries += 1
                    if missing_gt_files:
                        print(f"Skipping query {query_item.get('id', '?')}: Missing GT image files {missing_gt_files}")
                    if missing_ref_file:
                        print(f"Skipping query {query_item.get('id', '?')}: Missing reference image file {ref_id}")
                    if missing_target_file:
                        print(f"Skipping query {query_item.get('id', '?')}: Missing target image file {target_id}")
                    continue
                
                # Process query
                try:
                    result = await self.process_single_query(query_item)
                    self.results.append(result)
                    processed_queries += 1
                
                    # Periodically save results
                    if len(self.results) % 10 == 0:
                        self.save_results()
                            
                except Exception as e:
                    print(f"Error processing query {query_item.get('id', '?')}: {e}")
                    skipped_queries += 1
                    continue
        
        # Execute async processing
        asyncio.run(process_queries())
        
        # Display statistics considering resume mode
        total_processed_in_session = processed_queries
        total_skipped_in_session = skipped_queries
        total_results = len(self.results)  # Existing results + new processing results
        
        print(f"\nEvaluation session completed:")
        print(f"  Queries in this session: {len(self.data)} (remaining)")
        print(f"  Processed in this session: {total_processed_in_session}")
        print(f"  Skipped in this session: {total_skipped_in_session}")
        print(f"  Total results accumulated: {total_results}")
        
        if total_processed_in_session > 0:
            session_success_rate = len([r for r in self.results[-total_processed_in_session:] if r['success']])/total_processed_in_session*100
            print(f"  Success rate in this session: {session_success_rate:.1f}%")
        
        overall_success_rate = len([r for r in self.results if r['success']])/max(1, total_results)*100
        print(f"  Overall success rate: {overall_success_rate:.1f}%")
        
        # Save final results
        self.save_results()
        self.print_statistics()
    
    def save_results(self) -> None:
        """Save results to JSON file"""
        
        def convert_to_json_serializable(obj):
            """Convert PyTorch tensors and numpy arrays to JSON-compatible format"""
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
                return float(obj) if isinstance(obj, (np.float32, np.float64)) else int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_json_serializable(item) for item in obj)
            else:
                return obj
        
        def convert_hash_to_image_id(hash_or_id: str) -> str:
            """For CIRCO, convert hash value to 12-digit image ID"""
            if self.dataset_name == 'circo' and hasattr(self, 'hash_to_filename'):
                if hash_or_id in self.hash_to_filename:
                    # For hash values, convert to 12-digit filename and remove extension
                    filename = self.hash_to_filename[hash_or_id]
                    if filename.endswith('.jpg'):
                        return filename[:-4]  # Remove extension to get 12-digit numeric ID
                    return filename
            return hash_or_id
        
        # Convert results to JSON-compatible format (also convert hash values to image IDs)
        json_results = []
        for result in self.results:
            json_result = convert_to_json_serializable(result.copy())
            
            # For CIRCO, convert hash values to image IDs
            if self.dataset_name == 'circo':
                # Convert reference_image_id, target_image_id, ground_truth_ids
                json_result['reference_image_id'] = convert_hash_to_image_id(json_result['reference_image_id'])
                json_result['target_image_id'] = convert_hash_to_image_id(json_result['target_image_id'])
                json_result['ground_truth_ids'] = [convert_hash_to_image_id(gt_id) for gt_id in json_result['ground_truth_ids']]
                
                # Also convert search results and selected_image for each turn
                for turn in json_result.get('turns', []):
                    if 'search_results' in turn:
                        turn['search_results'] = [
                            [convert_hash_to_image_id(img_id), score] 
                            for img_id, score in turn['search_results']
                        ]
                    if 'selected_image' in turn:
                        turn['selected_image'] = convert_hash_to_image_id(turn['selected_image'])
                    
                    # Also convert gt_rankings keys
                    if 'gt_rankings' in turn:
                        turn['gt_rankings'] = {
                            convert_hash_to_image_id(gt_id): rank 
                            for gt_id, rank in turn['gt_rankings'].items()
                        }
            
            json_results.append(json_result)
        
        # Structure results in the same format as Fashion-IQ
        final_results = {
            "dataset_name": self.dataset_name,
            "config": convert_to_json_serializable(self.config),
            "data_completeness": convert_to_json_serializable(self.completeness_info) if hasattr(self, 'completeness_info') else {},
            "results": json_results
        }
        
        # Save to results file
        output_dir = self.config.get('output_dir', '.')
        results_file = os.path.join(output_dir, f"multiturn_cir_results_{self.dataset_name}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {results_file}")
        print(f"Total results: {len(json_results)}")
    
    def print_statistics(self) -> None:
        """Output statistics"""
        total_queries = len(self.results)
        if total_queries == 0:
            print(f"\n=== {self.dataset_name.upper()} Results ===")
            print("No queries were processed successfully.")
            return
            
        successful_queries = sum(1 for r in self.results if r['success'])
        success_rate = successful_queries / total_queries * 100
        
        print(f"\n=== {self.dataset_name.upper()} Results ===")
        print(f"Total queries: {total_queries}")
        print(f"Successful queries: {successful_queries}")
        print(f"Success rate: {success_rate:.2f}%")
        
        # Fixed version of turn-wise success rate
        turn_success = {}
        turn_distribution = {}  # Multi-turn count distribution (0 = initial search only)
        max_turns_observed = 0
        
        for result in self.results:
            actual_turns_executed = len(result['turns'])  # Number of turns actually executed
            multiturn_count = actual_turns_executed - 1   # Multi-turn count (excluding turn 0)
            max_turns_observed = max(max_turns_observed, actual_turns_executed)
            
            # Count multi-turn occurrences
            turn_distribution[multiturn_count] = turn_distribution.get(multiturn_count, 0) + 1
            
            if result['success']:
                success_turn = result['success_turn']
                turn_success[success_turn] = turn_success.get(success_turn, 0) + 1
        
        print(f"Max turns observed: {max_turns_observed}")
        print(f"Turn distribution: {turn_distribution}")
        print(f"Turn distribution sum: {sum(turn_distribution.values())}")
        
        # Calculate number of queries reaching each turn
        turn_counts = {}
        turn_recall_counts = {}  # Number of queries actually evaluated
        
        for turn in range(max_turns_observed):
            # Number of queries reaching that turn
            queries_reaching_turn = sum(1 for r in self.results if len(r['turns']) > turn)
            turn_counts[turn + 1] = queries_reaching_turn  # Display as 1-indexed
            
            # Number of queries actually evaluated at that turn (those with recorded GT rank)
            queries_evaluated = sum(1 for r in self.results 
                                   if len(r['turns']) > turn and 
                                   r['turns'][turn].get('best_gt_rank', -1) > 0)
            turn_recall_counts[turn + 1] = queries_evaluated
        
        print(f"Turn counts (queries reaching each turn): {turn_counts}")
        print(f"Turn recall counts (actual evaluations): {turn_recall_counts}")
        
        print(f"\n=== {self.dataset_name} (text_only, average) ===")
        print(f"\nDataset Statistics:")
        print(f"  Total queries: {total_queries}")
        print(f"  Successful queries: {successful_queries}")
        print(f"  Max turns: {max_turns_observed}")
        print(f"  Turn distribution:")
        
        # Display multi-turn count distribution correctly
        for multiturn_count in sorted(turn_distribution.keys()):
            count = turn_distribution[multiturn_count]
            percentage = count / total_queries * 100
            if multiturn_count == 0:
                print(f"    Initial search only: {count} queries ({percentage:.1f}%)")
            else:
                actual_turns = multiturn_count + 1  # Turn 0 + multi-turns
                print(f"    {actual_turns} turns: {count} queries ({percentage:.1f}%)")
        
        # Cumulative success rate (Hits@T)
        print(f"\nHits@T (Cumulative success rate):")
        cumulative_success = 0
        for turn in range(max_turns_observed):
            turn_successes = turn_success.get(turn, 0)  # 0-indexed turn number
            cumulative_success += turn_successes
            cumulative_rate = cumulative_success / total_queries
            if turn == 0:
                print(f"  Initial search: {cumulative_rate:.3f} ({cumulative_rate * 100:.1f}%)")
            else:
                print(f"  Turn {turn + 1}: {cumulative_rate:.3f} ({cumulative_rate * 100:.1f}%)")
        
        # Turn-wise Recall@10
        print(f"\nRecall@10 per Turn:")
        for turn in range(max_turns_observed):
            if turn + 1 in turn_recall_counts and turn_recall_counts[turn + 1] > 0:
                turn_successes = turn_success.get(turn, 0)
                queries_evaluated = turn_recall_counts[turn + 1]
                recall_rate = turn_successes / queries_evaluated
                if turn == 0:
                    print(f"  Initial search: {recall_rate:.3f} ({recall_rate * 100:.1f}%) [{turn_successes}/{queries_evaluated} queries]")
                else:
                    print(f"  Turn {turn + 1}: {recall_rate:.3f} ({recall_rate * 100:.1f}%) [{turn_successes}/{queries_evaluated} queries]")
        
        # Final Recall@10 (overall success rate)
        final_recall = successful_queries / total_queries
        print(f"\nFinal Recall@10: {final_recall:.3f} ({final_recall * 100:.1f}%)")
        
        # AUC score calculation (simplified)
        auc_sum = 0
        for turn in range(max_turns_observed):
            turn_successes = turn_success.get(turn, 0)
            cumulative_success = sum(turn_success.get(t, 0) for t in range(turn + 1))
            auc_sum += cumulative_success / total_queries
        
        auc_score = auc_sum / max_turns_observed if max_turns_observed > 0 else 0
        print(f"AUC Score: {auc_score:.3f}")
        
        # Turn-wise success rate (maintain legacy display)
        print("\nSuccess by turn:")
        for turn in sorted(turn_success.keys()):
            count = turn_success[turn]
            if turn == 0:
                print(f"  Initial search: {count} queries ({count/total_queries*100:.1f}%)")
            else:
                print(f"  Turn {turn}: {count} queries ({count/total_queries*100:.1f}%)")
        
        # GT rank statistical analysis
        print("\n=== Ground Truth Ranking Analysis ===")
        
        # GT rank statistics per turn
        for turn in range(max_turns_observed):
            turn_gt_ranks = []
            turn_queries_with_data = 0
            
            for result in self.results:
                if turn < len(result['turns']):
                    turn_data = result['turns'][turn]
                    if 'best_gt_rank' in turn_data and turn_data['best_gt_rank'] > 0:
                        turn_gt_ranks.append(turn_data['best_gt_rank'])
                        turn_queries_with_data += 1
            
            if turn_gt_ranks:
                avg_rank = sum(turn_gt_ranks) / len(turn_gt_ranks)
                min_rank = min(turn_gt_ranks)
                max_rank = max(turn_gt_ranks)
                
                # Percentage within Top-k
                top1_count = sum(1 for rank in turn_gt_ranks if rank == 1)
                top5_count = sum(1 for rank in turn_gt_ranks if rank <= 5)
                top10_count = sum(1 for rank in turn_gt_ranks if rank <= 10)
                
                if turn == 0:
                    print(f"\nInitial Search GT Rankings (from {turn_queries_with_data} queries):")
                else:
                    print(f"\nTurn {turn + 1} GT Rankings (from {turn_queries_with_data} queries):")
                print(f"  Average rank: {avg_rank:.2f}")
                print(f"  Best rank: {min_rank}, Worst rank: {max_rank}")
                print(f"  Top-1: {top1_count} ({top1_count/len(turn_gt_ranks)*100:.1f}%)")
                print(f"  Top-5: {top5_count} ({top5_count/len(turn_gt_ranks)*100:.1f}%)")
                print(f"  Top-10: {top10_count} ({top10_count/len(turn_gt_ranks)*100:.1f}%)")
        
        # Analysis of GT rank improvement/degradation
        print("\n=== GT Ranking Improvement Analysis ===")
        
        improved_queries = 0
        worsened_queries = 0
        unchanged_queries = 0
        
        for result in self.results:
            if len(result['turns']) >= 2:
                initial_rank = result['turns'][0].get('best_gt_rank', -1)
                final_rank = result['turns'][-1].get('best_gt_rank', -1)
                
                if initial_rank > 0 and final_rank > 0:
                    if final_rank < initial_rank:
                        improved_queries += 1
                    elif final_rank > initial_rank:
                        worsened_queries += 1
                    else:
                        unchanged_queries += 1
        
        total_comparable = improved_queries + worsened_queries + unchanged_queries
        if total_comparable > 0:
            print(f"Queries with multi-turn data: {total_comparable}")
            print(f"  Improved ranking: {improved_queries} ({improved_queries/total_comparable*100:.1f}%)")
            print(f"  Worsened ranking: {worsened_queries} ({worsened_queries/total_comparable*100:.1f}%)")
            print(f"  Unchanged ranking: {unchanged_queries} ({unchanged_queries/total_comparable*100:.1f}%)")
        
        # Individual query detail examples (first 5)
        print(f"\n=== Sample Query GT Ranking Progression ===")
        
        def convert_hash_to_image_id_for_display(hash_or_id: str) -> str:
            """For CIRCO, convert hash value to 12-digit image ID (for display)"""
            if self.dataset_name == 'circo' and hasattr(self, 'hash_to_filename'):
                if hash_or_id in self.hash_to_filename:
                    # For hash values, convert to 12-digit filename and remove extension
                    filename = self.hash_to_filename[hash_or_id]
                    if filename.endswith('.jpg'):
                        return filename[:-4]  # Remove extension to get 12-digit numeric ID
                    return filename
            return hash_or_id
        
        for i, result in enumerate(self.results[:5]):
            # Also convert Ground truth IDs
            display_gt_ids = [convert_hash_to_image_id_for_display(gt_id) for gt_id in result['ground_truth_ids']]
            print(f"\nQuery {result['query_id']} (GT: {display_gt_ids}):")
            
            for turn_data in result['turns']:
                turn = turn_data['turn']
                best_rank = turn_data.get('best_gt_rank', -1)
                gt_rankings = turn_data.get('gt_rankings', {})
                
                if best_rank > 0:
                    print(f"  Turn {turn}: Best GT rank = {best_rank}")
                    if len(gt_rankings) > 1:
                        # Show details for multiple GTs (convert hash values to image IDs)
                        gt_details = [f"{convert_hash_to_image_id_for_display(gt_id)}:{rank}" 
                                    for gt_id, rank in gt_rankings.items() if rank > 0]
                        print(f"    GT details: {', '.join(gt_details)}")
                else:
                    print(f"  Turn {turn}: GT not found in search results")
            
            if result['success']:
                print(f"  → SUCCESS at turn {result['success_turn']}")
            else:
                print(f"  → FAILED after {len(result['turns'])} turns")

    def load_caption_features(self) -> Dict[str, torch.Tensor]:
        """Load pre-computed caption features"""
        caption_features = {}
        
        # Load gpt4omini caption features file (PyTorch format)
        if 'gpt4omini_caption_features_file' in self.config and os.path.exists(self.config['gpt4omini_caption_features_file']):
            try:
                features_data = torch.load(self.config['gpt4omini_caption_features_file'], map_location='cpu', weights_only=False)
                
                if isinstance(features_data, dict):
                    # Check if data is in new array format
                    if 'features' in features_data and 'image_ids' in features_data:
                        # Array format: {'features': tensor, 'image_ids': list}
                        features_tensor = features_data['features']  # Shape: (N, feature_dim)
                        image_ids = features_data['image_ids']  # List of image paths
                        
                        # Convert to tensor if numpy array
                        if isinstance(features_tensor, np.ndarray):
                            features_tensor = torch.from_numpy(features_tensor).float()
                        
                        # Convert to dictionary by image ID
                        for i, image_id in enumerate(image_ids):
                            # Extract image filename from full path (without extension)
                            if '/' in image_id:
                                # fashion-iq/images/dress/B008BHCT58.jpg -> B008BHCT58
                                # CIRCO/unlabeled2017/000000212560.jpg -> 000000212560.jpg
                                image_filename = image_id.split('/')[-1]
                                if '.' in image_filename:
                                    image_filename_no_ext = image_filename.rsplit('.', 1)[0]  # Remove extension
                                else:
                                    image_filename_no_ext = image_filename
                            else:
                                # If already filename only
                                image_filename = image_id
                                if '.' in image_filename:
                                    image_filename_no_ext = image_filename.rsplit('.', 1)[0]  # Remove extension
                                else:
                                    image_filename_no_ext = image_filename
                            
                            # For CIRCO, also save filename with extension as key
                            if self.dataset_name == 'circo':
                                caption_features[image_filename] = features_tensor[i]  # With extension
                                caption_features[image_filename_no_ext] = features_tensor[i]  # Without extension
                            else:
                                caption_features[image_filename_no_ext] = features_tensor[i]
                        
                        print(f"Loaded caption features for {len(caption_features)} images from array format")
                    else:
                        # Legacy dictionary format: {image_id: tensor, ...}
                        for image_id, features in features_data.items():
                            # Convert to tensor if numpy array
                            if isinstance(features, np.ndarray):
                                features = torch.from_numpy(features).float()
                            caption_features[image_id] = features
                        print(f"Loaded caption features for {len(caption_features)} images from dict format")
                else:
                    print(f"Warning: Unexpected format in caption features file: {self.config['gpt4omini_caption_features_file']}")
                    
            except Exception as e:
                print(f"Warning: Failed to load caption features from {self.config['gpt4omini_caption_features_file']}: {e}")
        
        # For CIRCO, add mapping to allow access by hash value
        if self.dataset_name == 'circo' and hasattr(self, 'hash_to_filename'):
            filename_to_hash = {filename: hash_val for hash_val, filename in self.hash_to_filename.items()}
            
            # Create mapping from 12-digit filename to hash value
            additional_mappings = {}
            for key, features in caption_features.items():
                # If key is 12-digit filename (with or without extension)
                if key.endswith('.jpg'):
                    # If filename with extension
                    if key in filename_to_hash:
                        hash_val = filename_to_hash[key]
                        additional_mappings[hash_val] = features
                else:
                    # If filename without extension, search with extension
                    filename_with_ext = f"{key}.jpg"
                    if filename_with_ext in filename_to_hash:
                        hash_val = filename_to_hash[filename_with_ext]
                        additional_mappings[hash_val] = features
            
            # Add access by hash value
            caption_features.update(additional_mappings)
            print(f"Added {len(additional_mappings)} hash-based mappings for CIRCO caption features")
        
        return caption_features

    def check_data_completeness(self) -> Dict[str, Any]:
        """Check data completeness and return missing information"""
        print(f"\n=== Checking data completeness for {self.dataset_name} ===")
        
        # Set image directory
        image_dir = self.config.get('image_dir', '')
        
        # Collect image IDs used in dataset queries
        reference_ids = set()
        target_ids = set()
        gt_ids = set()
        
        for query_item in self.data:
            reference_ids.add(query_item['reference_image_id'])
            target_ids.add(query_item['target_image_id'])
            gt_ids.update(query_item['ground_truth_ids'])
        
        # Check image file existence
        def check_image_exists(image_id: str) -> bool:
            """Check if image file exists"""
            if self.dataset_name.startswith('fashioniq'):
                # Fashion-IQ: Include category subdirectory and auto-add extension
                category = self.dataset_name.split('_')[1]  # dress, shirt, toptee
                img_with_ext = image_id if image_id.endswith('.jpg') else f"{image_id}.jpg"
                image_path = os.path.join(image_dir, category, img_with_ext)
                return os.path.exists(image_path)
            elif self.dataset_name.startswith('cirr'):
                # For CIRR: structure differs by split
                # train: hierarchical structure (0-99 subdirectories)
                # val, dev, test1: flat structure (files directly under directory)
                
                # For train: hierarchical structure
                for subdir in range(100):  # Search 0-99 subdirectories
                    for ext in ['.jpg', '.jpeg', '.png']:
                        image_path = os.path.join(image_dir, 'train', str(subdir), f"{image_id}{ext}")
                        if os.path.exists(image_path):
                            return True
                
                # For val, dev, test1: Flat structure
                for split in ['val', 'dev', 'test1']:
                    for ext in ['.jpg', '.jpeg', '.png']:
                        image_path = os.path.join(image_dir, split, f"{image_id}{ext}")
                        if os.path.exists(image_path):
                            return True
                
                return False
            else:
                # CIRCO: Convert hash value to actual filename
                if image_id in self.hash_to_filename:
                    filename = self.hash_to_filename[image_id]
                    image_path = os.path.join(image_dir, filename)
                    return os.path.exists(image_path)
                else:
                    # If not hash value (12-digit format), use as-is
                    image_path = os.path.join(image_dir, image_id)
                    return os.path.exists(image_path)
        
        # Missing data analysis
        print("Checking image file existence...")
        missing_ref_images = [img_id for img_id in reference_ids if not check_image_exists(img_id)]
        missing_target_images = [img_id for img_id in target_ids if not check_image_exists(img_id)]
        missing_gt_images = [img_id for img_id in gt_ids if not check_image_exists(img_id)]
        
        # Image ID set in search space
        search_space_set = set(self.retrieval_engine.search_space)
        missing_ref_search = reference_ids - search_space_set
        missing_target_search = target_ids - search_space_set
        missing_gt_search = gt_ids - search_space_set
        
        # Also display available caption features as reference
        available_caption_features = set(self.caption_features.keys())
        missing_ref_features = reference_ids - available_caption_features
        missing_target_features = target_ids - available_caption_features
        missing_gt_features = gt_ids - available_caption_features
        
        # Summarize results
        completeness_info = {
            'total_queries': len(self.data),
            'image_dir': image_dir,
            'available_caption_features': len(available_caption_features),
            'search_space_size': len(search_space_set),
            'reference_images': {
                'total': len(reference_ids),
                'missing_image_files': len(missing_ref_images),
                'missing_caption_features': len(missing_ref_features),
                'missing_in_search_space': len(missing_ref_search)
            },
            'target_images': {
                'total': len(target_ids),
                'missing_image_files': len(missing_target_images),
                'missing_caption_features': len(missing_target_features),
                'missing_in_search_space': len(missing_target_search)
            },
            'ground_truth_images': {
                'total': len(gt_ids),
                'missing_image_files': len(missing_gt_images),
                'missing_caption_features': len(missing_gt_features),
                'missing_in_search_space': len(missing_gt_search)
            },
            'missing_image_files': {
                'reference': missing_ref_images,
                'target': missing_target_images,
                'ground_truth': missing_gt_images
            }
        }
        
        # Output detailed report
        print(f"Total queries: {completeness_info['total_queries']}")
        print(f"Image directory: {image_dir}")
        print(f"Available caption features: {completeness_info['available_caption_features']}")
        print(f"Search space size: {completeness_info['search_space_size']}")
        
        print(f"\nReference images:")
        print(f"  Total: {completeness_info['reference_images']['total']}")
        print(f"  Missing image files: {completeness_info['reference_images']['missing_image_files']}")
        print(f"  Missing caption features: {completeness_info['reference_images']['missing_caption_features']}")
        print(f"  Missing in search space: {completeness_info['reference_images']['missing_in_search_space']}")
        
        print(f"\nTarget images:")
        print(f"  Total: {completeness_info['target_images']['total']}")
        print(f"  Missing image files: {completeness_info['target_images']['missing_image_files']}")
        print(f"  Missing caption features: {completeness_info['target_images']['missing_caption_features']}")
        print(f"  Missing in search space: {completeness_info['target_images']['missing_in_search_space']}")
        
        print(f"\nGround truth images:")
        print(f"  Total: {completeness_info['ground_truth_images']['total']}")
        print(f"  Missing image files: {completeness_info['ground_truth_images']['missing_image_files']}")
        print(f"  Missing caption features: {completeness_info['ground_truth_images']['missing_caption_features']}")
        print(f"  Missing in search space: {completeness_info['ground_truth_images']['missing_in_search_space']}")
        
        # Estimate processable queries (based on image file existence)
        processable_queries = 0
        for query_item in self.data:
            gt_available = all(check_image_exists(gt_id) for gt_id in query_item['ground_truth_ids'])
            ref_available = check_image_exists(query_item['reference_image_id'])
            target_available = check_image_exists(query_item['target_image_id'])
            
            if gt_available and ref_available and target_available:
                processable_queries += 1
        
        print(f"\nEstimated processable queries (based on image files): {processable_queries}/{len(self.data)} "
              f"({processable_queries/len(self.data)*100:.1f}%)")
        
        # Display sample of missing images (for debugging)
        if missing_gt_images:
            sample_missing = missing_gt_images[:5]
            print(f"\nSample missing GT image files: {sample_missing}")
        if missing_ref_images:
            sample_missing = missing_ref_images[:3]
            print(f"Sample missing reference image files: {sample_missing}")
        
        completeness_info['estimated_processable_queries'] = processable_queries
        
        return completeness_info

def main():
    """Main function"""
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Multi-turn CIR System')
    parser.add_argument('--dataset', type=str, default='circo', 
                       choices=['circo', 'cirr_train', 'cirr_val', 
                               'fashioniq_dress_train', 'fashioniq_dress_val',
                               'fashioniq_shirt_train', 'fashioniq_shirt_val',
                               'fashioniq_toptee_train', 'fashioniq_toptee_val'],
                       help='Dataset to evaluate on')
    parser.add_argument('--max_turns', type=int, default=5,
                       help='Maximum number of turns')
    parser.add_argument('--caption_mode', type=str, default='separate',
                       choices=['combined', 'separate', 'first_only'],
                       help='Caption processing mode for Fashion-IQ: separate (default, standard), combined, first_only')
    parser.add_argument('--test_mode', action='store_true',
                       help='Run in test mode with limited data')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use: auto (default, use CUDA if available), cpu, cuda')
    parser.add_argument('--data_dir', type=str, default='.',
                       help='Base directory containing dataset folders (fashion-iq/, cirr/, CIRCO/)')
    parser.add_argument('--output_dir', type=str, default='output/raw',
                       help='Output directory for result files (default: output/raw)')

    args = parser.parse_args()

    # Convert output_dir to absolute path before changing directory
    # This allows output to be independent of data_dir
    args.output_dir = os.path.abspath(args.output_dir)

    # Change to data directory
    if args.data_dir != '.':
        os.chdir(args.data_dir)
        print(f"Working directory: {os.getcwd()}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Update global device setting
    global device
    if args.device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    
    # Configuration
    datasets_config = {
        'circo': {
            'annotation_file': 'CIRCO/annotations/val.json',
            'corpus_vectors_file': 'CIRCO/features_blip.pt',  # BLIP feature file
            'search_space_file': 'CIRCO/metadata_blip.pt',    # Search space metadata
            'image_dir': 'CIRCO/unlabeled2017',  # COCO2017 unlabeled 120k image pool
            'caption_file': 'CIRCO/captions_gpt4omini.json',     # GPT-4o-mini captions
            'gpt4omini_caption_features_file': 'CIRCO/gpt4omini_captions_blip_features.pt',  # gpt4omini caption features
            'use_blip': True,
            'max_turns': args.max_turns,
            'dataset_split': 'val',  # Using val because test ground truth labels are not public
            'caption_mode': 'combined'
        },
        'cirr_train': {
            'annotation_file': 'cirr/captions/cap.rc2.train.json',
            'corpus_vectors_file': 'cirr/features_blip.pt',
            'search_space_file': 'cirr/image_splits/split.rc2.train.json',
            'image_dir': 'cirr/img_raw',
            'caption_file': 'cirr/captions_gpt4omini.json',
            'gpt4omini_caption_features_file': 'cirr/gpt4omini_captions_blip_features.pt',
            'use_blip': True,
            'max_turns': args.max_turns,
            'dataset_split': 'train',
            'caption_mode': 'combined'
        },
        'cirr_val': {
            'annotation_file': 'cirr/captions/cap.rc2.val.json',  # Using val set (test ground truth not public)
            'corpus_vectors_file': 'cirr/features_blip.pt',       # BLIP feature file
            'search_space_file': 'cirr/image_splits/split.rc2.val.json',  # val image pool
            'image_dir': 'cirr/img_raw',                          # CIRR image directory
            'caption_file': 'cirr/captions_gpt4omini.json',          # GPT-4o-mini captions
            'gpt4omini_caption_features_file': 'cirr/gpt4omini_captions_blip_features.pt',  # gpt4omini caption features
            'use_blip': True,
            'max_turns': args.max_turns,
            'dataset_split': 'val',  # Using val because test ground truth labels are not public
            'caption_mode': 'combined'
        },
        # Fashion-IQ Dress category
        'fashioniq_dress_train': {
            'annotation_file': 'fashion-iq/captions/cap.dress.train.json',
            'corpus_vectors_file': 'fashion-iq/features_blip.pt',
            'search_space_file': 'fashion-iq/image_splits/split.dress.train.json',
            'image_dir': 'fashion-iq/images',
            'caption_file': 'fashion-iq/captions_gpt4omini.json',
            'gpt4omini_caption_features_file': 'fashion-iq/gpt4omini_captions_blip_features.pt',
            'use_blip': True,
            'max_turns': args.max_turns,
            'dataset_split': 'train',
            'caption_mode': args.caption_mode
        },
        'fashioniq_dress_val': {
            'annotation_file': 'fashion-iq/captions/cap.dress.val.json',
            'corpus_vectors_file': 'fashion-iq/features_blip.pt',
            'search_space_file': 'fashion-iq/image_splits/split.dress.val.json',
            'image_dir': 'fashion-iq/images',
            'caption_file': 'fashion-iq/captions_gpt4omini.json',
            'gpt4omini_caption_features_file': 'fashion-iq/gpt4omini_captions_blip_features.pt',
            'use_blip': True,
            'max_turns': args.max_turns,
            'dataset_split': 'val',
            'caption_mode': args.caption_mode
        },
        # Fashion-IQ Shirt category
        'fashioniq_shirt_train': {
            'annotation_file': 'fashion-iq/captions/cap.shirt.train.json',
            'corpus_vectors_file': 'fashion-iq/features_blip.pt',
            'search_space_file': 'fashion-iq/image_splits/split.shirt.train.json',
            'image_dir': 'fashion-iq/images',
            'caption_file': 'fashion-iq/captions_gpt4omini.json',
            'gpt4omini_caption_features_file': 'fashion-iq/gpt4omini_captions_blip_features.pt',
            'use_blip': True,
            'max_turns': args.max_turns,
            'dataset_split': 'train',
            'caption_mode': args.caption_mode
        },
        'fashioniq_shirt_val': {
            'annotation_file': 'fashion-iq/captions/cap.shirt.val.json',
            'corpus_vectors_file': 'fashion-iq/features_blip.pt',
            'search_space_file': 'fashion-iq/image_splits/split.shirt.val.json',
            'image_dir': 'fashion-iq/images',
            'caption_file': 'fashion-iq/captions_gpt4omini.json',
            'gpt4omini_caption_features_file': 'fashion-iq/gpt4omini_captions_blip_features.pt',
            'use_blip': True,
            'max_turns': args.max_turns,
            'dataset_split': 'val',
            'caption_mode': args.caption_mode
        },
        # Fashion-IQ Toptee category
        'fashioniq_toptee_train': {
            'annotation_file': 'fashion-iq/captions/cap.toptee.train.json',
            'corpus_vectors_file': 'fashion-iq/features_blip.pt',
            'search_space_file': 'fashion-iq/image_splits/split.toptee.train.json',
            'image_dir': 'fashion-iq/images',
            'caption_file': 'fashion-iq/captions_gpt4omini.json',
            'gpt4omini_caption_features_file': 'fashion-iq/gpt4omini_captions_blip_features.pt',
            'use_blip': True,
            'max_turns': args.max_turns,
            'dataset_split': 'train',
            'caption_mode': args.caption_mode
        },
        'fashioniq_toptee_val': {
            'annotation_file': 'fashion-iq/captions/cap.toptee.val.json',
            'corpus_vectors_file': 'fashion-iq/features_blip.pt',
            'search_space_file': 'fashion-iq/image_splits/split.toptee.val.json',
            'image_dir': 'fashion-iq/images',
            'caption_file': 'fashion-iq/captions_gpt4omini.json',
            'gpt4omini_caption_features_file': 'fashion-iq/gpt4omini_captions_blip_features.pt',
            'use_blip': True,
            'max_turns': args.max_turns,
            'dataset_split': 'val',
            'caption_mode': args.caption_mode
        }
    }
    
    # Select dataset to run
    dataset_name = args.dataset
    
    if dataset_name not in datasets_config:
        print(f"Unknown dataset: {dataset_name}")
        return
    
    config = datasets_config[dataset_name]
    config['output_dir'] = args.output_dir

    # For Fashion-IQ dataset, set caption processing mode
    if dataset_name.startswith('fashioniq'):
        config['caption_mode'] = args.caption_mode
        print(f"Using caption mode: {args.caption_mode}")
    
    # Initialize system
    system = MultiTurnCIRSystem(dataset_name, config)
    
    # Limit for test mode
    if args.test_mode:
        print("Running in test mode - limiting to first 5 queries")
        system.data = system.data[:5]
    
    # Resume functionality explanation
    if len(system.results) == 0:
        print(f"\n=== Starting Fresh Evaluation ===")
        print(f"No existing results found. Starting evaluation from the beginning.")
    else:
        print(f"\n=== Resume Mode Active ===")
        print(f"Existing results detected. Evaluation will resume from where it left off.")
        print(f"To start fresh, delete the existing result files:")
        print(f"  - multiturn_cir_results_{dataset_name}.json")
        print(f"  - multiturn_cir_summary_{dataset_name}.csv")
        print(f"  - multiturn_cir_detailed_rankings_{dataset_name}.csv")
    
    if len(system.data) == 0:
        print(f"\n=== All Queries Completed ===")
        print(f"All queries for {dataset_name} have been processed!")
        print(f"Total results: {len(system.results)}")
        system.print_statistics()
        return
    
    # Run evaluation
    system.run_evaluation()

if __name__ == "__main__":
    main() 