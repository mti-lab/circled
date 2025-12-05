#!/usr/bin/env python3
"""
MTCIR Caption Generation System with OpenAI GPT-4o
Asynchronously generate captions for FashionIQ, CIRCO, and CIRR images using OpenAI GPT-4o
"""

import json
import os
import asyncio
import aiohttp
import base64
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict
import time
from typing import List, Dict, Any, Optional
from asyncio_throttle import Throttler
from dotenv import load_dotenv
import logging
import tiktoken

# Load environment variables from .env file
load_dotenv()

class OpenAIGPT4oCaptionGenerator:
    """Caption generation class using OpenAI GPT-4o"""

    def __init__(self, api_key: Optional[str] = None, max_concurrent: int = 100, requests_per_minute: int = 1500):
        """
        Initialize the caption generator.

        Args:
            api_key: OpenAI API Key (if None, retrieved from environment variable)
            max_concurrent: Maximum concurrent requests (default: 100)
            requests_per_minute: Maximum requests per minute (default: 1500)
        """
        self.api_key = api_key or os.getenv('OPEN_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPEN_API_KEY in .env file or pass as parameter.")
        
        self.max_concurrent = max_concurrent
        self.requests_per_minute = requests_per_minute
        
        # Throttler for rate limiting
        self.throttler = Throttler(rate_limit=requests_per_minute, period=60)

        # Semaphore for concurrent request limiting
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # OpenAI API configuration
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Logging configuration
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 2  # seconds

        # Token counting encoder
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Cost calculation (2024 pricing)
        self.cost_per_input_token = 0.00015 / 1000  # $0.15 per 1M input tokens
        self.cost_per_output_token = 0.0006 / 1000  # $0.60 per 1M output tokens

        # Statistics
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        
    def load_dataset_configs(self):
        """Load dataset configurations"""
        configs = {
            'fashion-iq': {
                'data_dir': 'fashion-iq/images',
                'split_files': {
                    'train': [
                        'fashion-iq/image_splits/split.dress.train.json',
                        'fashion-iq/image_splits/split.shirt.train.json',
                        'fashion-iq/image_splits/split.toptee.train.json'
                    ],
                    'val': [
                        'fashion-iq/image_splits/split.dress.val.json',
                        'fashion-iq/image_splits/split.shirt.val.json',
                        'fashion-iq/image_splits/split.toptee.val.json'
                    ],
                    'test': [
                        'fashion-iq/image_splits/split.dress.test.json',
                        'fashion-iq/image_splits/split.shirt.test.json',
                        'fashion-iq/image_splits/split.toptee.test.json'
                    ]
                },
                'categories': ['dress', 'shirt', 'toptee']
            },
            'cirr': {
                'data_dir': 'cirr/img_raw',
                'split_files': {
                    'train': 'cirr/image_splits/split.rc2.train.json',
                    'val': 'cirr/image_splits/split.rc2.val.json',
                    'test': 'cirr/image_splits/split.rc2.test1.json'
                },
                'categories': ['all'],
                'subdirs': {
                    'train': 'train',
                    'val': 'dev', 
                    'test': 'test1'
                }
            },
            'circo': {
                'data_dir': 'CIRCO/COCO2017_unlabeled/unlabeled2017',
                'categories': ['all']
            }
        }
        return configs

    def collect_dataset_images(self, dataset_name: str, configs: Dict, splits: List[str] = ['train', 'val', 'test']):
        """Collect image paths from the specified dataset"""
        if dataset_name not in configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = configs[dataset_name]
        image_paths = []
        image_info = {}
        
        # Statistics
        total_ids = 0
        found_images = 0
        missing_images = 0

        if dataset_name == 'fashion-iq':
            # Process FashionIQ
            data_dir = config['data_dir']
            
            for split in splits:
                if split not in config['split_files']:
                    continue
                
                for category in config['categories']:
                    # Find the split file for this category
                    category_split_file = None
                    for split_file in config['split_files'][split]:
                        if f'.{category}.' in split_file:
                            category_split_file = split_file
                            break
                    
                    if category_split_file is None:
                        continue
                    
                    print(f"Processing {dataset_name}/{category}/{split}: {category_split_file}")
                    
                    if not os.path.exists(category_split_file):
                        print(f"Warning: Split file not found: {category_split_file}")
                        continue
                    
                    with open(category_split_file, 'r') as f:
                        image_ids = json.load(f)
                    
                    category_total = len(image_ids)
                    category_found = 0
                    category_missing = 0
                    
                    for image_id in image_ids:
                        total_ids += 1
                        # FashionIQ uses IDs without extension, organized by category subdirectory
                        image_filename = f"{image_id}.jpg"
                        image_path = os.path.join(data_dir, category, image_filename)

                        if os.path.exists(image_path):
                            # Hash for duplicate checking
                            img_hash = self._get_image_hash(image_path)

                            if img_hash not in image_info:
                                # Add as new image
                                image_paths.append(image_path)
                                image_info[img_hash] = {
                                    'dataset': dataset_name,
                                    'category': category,
                                    'split': split,
                                    'image_id': image_id,
                                    'image_path': image_path,
                                    'duplicate_paths': []
                                }
                                found_images += 1
                                category_found += 1
                            else:
                                # Record path if duplicate image found
                                image_info[img_hash]['duplicate_paths'].append(image_path)
                                found_images += 1
                                category_found += 1
                        else:
                            missing_images += 1
                            category_missing += 1
                    
                    print(f"  {category}/{split}: {category_found}/{category_total} images found ({category_missing} missing)")
        
        elif dataset_name == 'cirr':
            # Process CIRR
            data_dir = config['data_dir']
            
            for split in splits:
                if split not in config['split_files']:
                    continue
                
                split_file = config['split_files'][split]
                print(f"Processing CIRR/{split}: {split_file}")
                
                if not os.path.exists(split_file):
                    print(f"Warning: Split file not found: {split_file}")
                    continue
                
                with open(split_file, 'r') as f:
                    split_data = json.load(f)
                
                split_total = 0
                split_found = 0
                
                # CIRR split file is dict format (image_id: path)
                for image_id, image_path in split_data.items():
                    total_ids += 1
                    split_total += 1
                    # If path is relative (./train/34/...), combine with data_dir
                    if image_path.startswith('./'):
                        full_path = os.path.join(data_dir, image_path[2:])  # Remove "./"
                    else:
                        full_path = os.path.join(data_dir, image_path)
                    
                    if os.path.exists(full_path):
                        # Hash for duplicate checking
                        img_hash = self._get_image_hash(full_path)

                        if img_hash not in image_info:
                            # Add as new image
                            image_paths.append(full_path)
                            image_info[img_hash] = {
                                'dataset': dataset_name,
                                'category': 'all',
                                'split': split,
                                'image_id': image_id,
                                'image_path': full_path,
                                'duplicate_paths': []
                            }
                            found_images += 1
                            split_found += 1
                        else:
                            # Record path if duplicate image found
                            image_info[img_hash]['duplicate_paths'].append(full_path)
                            found_images += 1
                            split_found += 1
                    else:
                        missing_images += 1
                
                print(f"  Found {split_found}/{split_total} images")
        
        elif dataset_name == 'circo':
            # Process CIRCO
            data_dir = config['data_dir']
            print(f"Processing CIRCO: scanning all images in {data_dir}")
            
            if not os.path.exists(data_dir):
                print(f"Warning: Image directory not found: {data_dir}")
            else:
                image_files = []
                for file in os.listdir(data_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_files.append(os.path.join(data_dir, file))
                
                print(f"Found {len(image_files)} image files in CIRCO")
                
                for image_path in image_files:
                    total_ids += 1
                    if os.path.exists(image_path):
                        img_hash = self._get_image_hash(image_path)
                        if img_hash not in image_info:
                            image_paths.append(image_path)
                            image_info[img_hash] = {
                                'dataset': dataset_name,
                                'category': 'all',
                                'split': 'all',
                                'image_id': os.path.basename(image_path),
                                'image_path': image_path,
                                'duplicate_paths': []
                            }
                            found_images += 1
                        else:
                            image_info[img_hash]['duplicate_paths'].append(image_path)
                            found_images += 1
                    else:
                        missing_images += 1
        
        print(f"\nDataset collection summary for {dataset_name}:")
        print(f"  Total IDs processed: {total_ids}")
        print(f"  Unique images found: {len(image_paths)}")
        print(f"  Total images found: {found_images}")
        print(f"  Missing images: {missing_images}")
        
        return image_paths, image_info

    def calculate_tokens(self, text: str) -> int:
        """Calculate token count for text"""
        return len(self.encoding.encode(text))

    def estimate_image_tokens(self) -> int:
        """Estimate token count for image (85 tokens for detail='low')"""
        return 85  # Per OpenAI official documentation

    def update_costs(self, input_tokens: int, output_tokens: int):
        """Update cost statistics"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        input_cost = input_tokens * self.cost_per_input_token
        output_cost = output_tokens * self.cost_per_output_token
        self.total_cost += input_cost + output_cost
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost statistics summary"""
        processed_count = getattr(self, '_processed_count', 1)
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'input_cost': self.total_input_tokens * self.cost_per_input_token,
            'output_cost': self.total_output_tokens * self.cost_per_output_token,
            'total_cost': self.total_cost,
            'cost_per_image': self.total_cost / max(1, processed_count)
        }

    def _get_image_hash(self, image_path: str) -> str:
        """Calculate hash of image file"""
        hasher = hashlib.md5()
        with open(image_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_category_from_path(self, image_path: str) -> str:
        """Infer category from image path"""
        path_parts = Path(image_path).parts
        
        if 'dress' in path_parts:
            return 'dress'
        elif 'shirt' in path_parts:
            return 'shirt'
        elif 'toptee' in path_parts:
            return 'toptee'
        elif 'cirr' in str(image_path).lower():
            return 'general'
        elif 'circo' in str(image_path).lower():
            return 'coco'
        else:
            return 'unknown'

    def generate_category_prompt(self, category: str) -> str:
        """Generate prompt based on category"""
        prompts = {
            'dress': "Describe this dress in 1-2 sentences. Focus on color, style, length, and key design features.",
            'shirt': "Describe this shirt in 1-2 sentences. Focus on color, style, collar, sleeves, and key design features.",
            'toptee': "Describe this top in 1-2 sentences. Focus on color, style, neckline, sleeves, and key design features.",
            'general': "Describe this image in 1-2 sentences. Include main objects, people, setting, and notable features.",
            'coco': "Describe this image in 1-2 sentences. Include main objects, composition, and notable features.",
            'unknown': "Describe this image in 1-2 sentences. Include main elements, colors, and notable features."
        }
        return prompts.get(category, prompts['unknown'])

    def encode_image_base64(self, image_path: str) -> str:
        """Encode image to Base64"""
        try:
            # Open and resize image (for API limits)
            with Image.open(image_path) as image:
                # Convert to RGB (handle transparency)
                if image.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                    image = background
                elif image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Size limit (for OpenAI API limits)
                max_size = (512, 512)  # Smaller size to reduce cost
                if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                    image.thumbnail(max_size, Image.Resampling.LANCZOS)

                # Save as JPEG temporarily and encode to Base64
                import io
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=75)  # Lower quality to reduce size
                image_data = buffer.getvalue()
                return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error encoding image {image_path}: {e}")
            raise

    async def generate_caption_single_with_retry(self, session: aiohttp.ClientSession, image_path: str, category: str, image_index: int) -> Dict[str, Any]:
        """Generate caption for single image with retry functionality"""
        for attempt in range(self.max_retries):
            try:
                result = await self.generate_caption_single(session, image_path, category, image_index)
                if result['success']:
                    return result

                # Increase wait time for rate limit errors
                if 'rate limit' in result.get('error', '').lower():
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"Rate limit hit for {image_path}, waiting {wait_time}s before retry {attempt+1}/{self.max_retries}")
                    await asyncio.sleep(wait_time)
                else:
                    break
                    
            except Exception as e:
                self.logger.error(f"Attempt {attempt+1} failed for {image_path}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    
        return {
            'image_path': image_path,
            'category': category,
            'caption': None,
            'success': False,
            'error': f'Failed after {self.max_retries} attempts',
            'image_index': image_index
        }

    async def generate_caption_single(self, session: aiohttp.ClientSession, image_path: str, category: str, image_index: int) -> Dict[str, Any]:
        """Asynchronously generate caption for single image"""
        async with self.semaphore:  # Concurrent request limit
            async with self.throttler:  # Rate limit
                try:
                    # Encode image to Base64
                    image_base64 = self.encode_image_base64(image_path)

                    # Generate prompt
                    prompt = self.generate_category_prompt(category)

                    # Payload for OpenAI API
                    payload = {
                        "model": "gpt-4o-mini",  # Using more affordable model
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_base64}",
                                            "detail": "low"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 100,  # Reduced tokens for shorter captions
                        "temperature": 0.1
                    }

                    # API call
                    async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            caption = result['choices'][0]['message']['content'].strip()

                            # Calculate and record token counts
                            prompt_tokens = self.calculate_tokens(prompt) + self.estimate_image_tokens()
                            output_tokens = self.calculate_tokens(caption)
                            self.update_costs(prompt_tokens, output_tokens)
                            
                            return {
                                'image_path': image_path,
                                'category': category,
                                'caption': caption,
                                'success': True,
                                'error': None,
                                'image_index': image_index,
                                'tokens': {
                                    'input': prompt_tokens,
                                    'output': output_tokens,
                                    'total': prompt_tokens + output_tokens
                                }
                            }
                        else:
                            error_text = await response.text()
                            self.logger.error(f"API error for {image_path}: {response.status} - {error_text}")
                            return {
                                'image_path': image_path,
                                'category': category,
                                'caption': None,
                                'success': False,
                                'error': f"API error: {response.status} - {error_text}",
                                'image_index': image_index
                            }
                            
                except Exception as e:
                    self.logger.error(f"Error processing {image_path}: {e}")
                    return {
                        'image_path': image_path,
                        'category': category,
                        'caption': None,
                        'success': False,
                        'error': str(e),
                        'image_index': image_index
                    }

    async def generate_captions_for_dataset(self, dataset_name: str, splits: List[str] = ['train', 'val', 'test'], output_file: str = None) -> Dict[str, Any]:
        """Asynchronously generate captions for entire dataset (with incremental save and resume support)"""
        print(f"Starting caption generation for dataset: {dataset_name}")
        
        configs = self.load_dataset_configs()
        image_paths, image_info = self.collect_dataset_images(dataset_name, configs, splits)
        if not image_paths:
            print(f"No images found for dataset {dataset_name}")
            return {}
        
        if output_file is None:
            output_file = f"captions_gpt4omini_{dataset_name}.json"
        temp_output_file = output_file + ".tmp"
        
        # Load existing caption file (resume functionality)
        existing_captions = {}
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_captions = json.load(f)
                print(f"Found existing caption file with {len(existing_captions)} entries")
            except Exception as e:
                print(f"Warning: Could not load existing caption file: {e}")

        # Also load from temp file (if more recent)
        if os.path.exists(temp_output_file):
            try:
                with open(temp_output_file, 'r', encoding='utf-8') as f:
                    temp_captions = json.load(f)
                if len(temp_captions) > len(existing_captions):
                    existing_captions = temp_captions
                    print(f"Loaded more recent data from temp file: {len(temp_captions)} entries")
            except Exception as e:
                print(f"Warning: Could not load temp caption file: {e}")

        # Extract only captions for unique image paths from existing captions
        # (Exclude duplicate image captions, consider only original image captions)
        existing_unique_captions = {}
        for path in image_paths:
            if path in existing_captions:
                existing_unique_captions[path] = existing_captions[path]

        print(f"Existing captions for unique images: {len(existing_unique_captions)}")

        # Extract only unprocessed image paths
        remaining_image_paths = [path for path in image_paths if path not in existing_unique_captions]
        already_processed = len(image_paths) - len(remaining_image_paths)
        
        print(f"Total unique images: {len(image_paths)}")
        print(f"Existing captions count: {len(existing_unique_captions)}")
        print(f"Already processed: {already_processed}")
        print(f"Remaining to process: {len(remaining_image_paths)}")
        
        # Debug info: Check duplicate image status
        if len(existing_unique_captions) > 0:
            # Check if existing_unique_captions keys are in image_paths
            existing_in_unique = sum(1 for path in existing_unique_captions.keys() if path in image_paths)
            print(f"Debug: Existing captions in unique paths: {existing_in_unique}")
            print(f"Debug: Existing captions not in unique paths: {len(existing_unique_captions) - existing_in_unique}")
        
        if len(remaining_image_paths) == 0:
            print("All images already have captions!")
            return existing_unique_captions
        
        print(f"Generating captions for {len(remaining_image_paths)} remaining images...")
        print(f"Rate limit: {self.requests_per_minute} requests/minute, Concurrent: {self.max_concurrent}")
        
        # Start from existing captions
        captions_dict = existing_unique_captions.copy()
        successful = 0
        failed = 0
        save_every = 100
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            tasks = [
                self.generate_caption_single_with_retry(session, image_path, self.get_category_from_path(image_path), i)
                for i, image_path in enumerate(remaining_image_paths)
            ]
            results = []
            with tqdm(total=len(tasks), desc=f"Generating captions for {dataset_name}") as pbar:
                for idx, coro in enumerate(asyncio.as_completed(tasks)):
                    result = await coro
                    results.append(result)
                    pbar.update(1)
                    image_path = result['image_path']
                    if result['success']:
                        captions_dict[image_path] = result['caption']
                        successful += 1
                    else:
                        failed += 1
                    # Save to temp file every 100 items
                    if (idx + 1) % save_every == 0:
                        self.save_captions(captions_dict, temp_output_file)
                        print(f"[Incremental save] Saved up to {already_processed + idx + 1} items")
            # Save to final file
            self.save_captions(captions_dict, output_file)
            if os.path.exists(temp_output_file):
                os.remove(temp_output_file)
        
        print(f"\nCaption generation completed:")
        print(f"  Previously processed: {already_processed}")
        print(f"  Newly successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total processed: {already_processed + successful}")
        print(f"  Success rate (new): {successful/(successful+failed)*100:.2f}%" if (successful+failed) > 0 else "  No new processing needed")
        
        # Copy captions to duplicate images across all datasets
        duplicate_count = 0
        for img_hash, info in image_info.items():
            if 'duplicate_paths' in info and info['duplicate_paths']:
                # Get caption from original image
                original_path = info['image_path']
                if original_path in captions_dict:
                    original_caption = captions_dict[original_path]
                    # Copy caption to duplicate image paths
                    for dup_path in info['duplicate_paths']:
                        captions_dict[dup_path] = original_caption
                        duplicate_count += 1
        
        if duplicate_count > 0:
            print(f"Copied captions to {duplicate_count} duplicate images")
            print(f"Total captions after duplication: {len(captions_dict)}")
        
        # Display cost statistics (new processing only)
        if successful + failed > 0:
            self._processed_count = successful + failed
            cost_summary = self.get_cost_summary()
            print(f"\n=== Cost Summary (New Processing Only) ===")
            print(f"  Total input tokens: {cost_summary['total_input_tokens']:,}")
            print(f"  Total output tokens: {cost_summary['total_output_tokens']:,}")
            print(f"  Total tokens: {cost_summary['total_tokens']:,}")
            print(f"  Input cost: ${cost_summary['input_cost']:.4f}")
            print(f"  Output cost: ${cost_summary['output_cost']:.4f}")
            print(f"  Total cost: ${cost_summary['total_cost']:.4f}")
            print(f"  Cost per image: ${cost_summary['cost_per_image']:.4f}")
            
            # Calculate estimated cost for remaining images
            if len(remaining_image_paths) > 0:
                estimated_remaining_cost = cost_summary['cost_per_image'] * len(remaining_image_paths)
                print(f"  Estimated cost for remaining images ({len(remaining_image_paths)} images): ${estimated_remaining_cost:.2f}")
        
        return captions_dict

    def save_captions(self, captions_dict: Dict[str, Any], output_file: str):
        """Save captions to JSON file"""
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(captions_dict, f, ensure_ascii=False, indent=2)
        
        print(f"Captions saved to: {output_file}")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate captions using OpenAI GPT-4o')
    parser.add_argument('--dataset', required=True, choices=['fashion-iq', 'cirr', 'circo'], 
                       help='Dataset to process')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'], 
                       help='Dataset splits to process')
    parser.add_argument('--output', 
                       help='Output file path (default: captions_gpt4omini_{dataset}.json)')
    parser.add_argument('--max-concurrent', type=int, default=100, 
                       help='Maximum concurrent requests (default: 100)')
    parser.add_argument('--requests-per-minute', type=int, default=1500, 
                       help='Maximum requests per minute (default: 1500)')
    parser.add_argument('--force-restart', action='store_true',
                       help='Force restart from beginning, ignoring existing caption files')
    parser.add_argument('--check-progress', action='store_true',
                       help='Only check progress without generating new captions')
    
    args = parser.parse_args()
    
    # Set output file name
    if not args.output:
        args.output = f'captions_gpt4omini_{args.dataset}.json'
    
    try:
        # Initialize caption generator
        generator = OpenAIGPT4oCaptionGenerator(
            max_concurrent=args.max_concurrent,
            requests_per_minute=args.requests_per_minute
        )
        
        # Progress check only mode
        if args.check_progress:
            configs = generator.load_dataset_configs()
            image_paths, _ = generator.collect_dataset_images(args.dataset, configs, args.splits)
            
            existing_captions = {}
            if os.path.exists(args.output):
                with open(args.output, 'r', encoding='utf-8') as f:
                    existing_captions = json.load(f)
            
            remaining = len(image_paths) - len(existing_captions)
            print(f"\n=== Progress Check ===")
            print(f"Dataset: {args.dataset}")
            print(f"Total images: {len(image_paths)}")
            print(f"Completed: {len(existing_captions)}")
            print(f"Remaining: {remaining}")
            print(f"Progress: {len(existing_captions)/len(image_paths)*100:.1f}%")
            return 0
        
        # If force restart, delete existing files
        if args.force_restart:
            if os.path.exists(args.output):
                os.remove(args.output)
                print(f"Removed existing caption file: {args.output}")
            temp_file = args.output + ".tmp"
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Removed existing temp file: {temp_file}")
        
        # Generate captions
        captions_dict = await generator.generate_captions_for_dataset(args.dataset, args.splits, args.output)
        
        # Save results
        if captions_dict:
            generator.save_captions(captions_dict, args.output)
        else:
            print("No captions were generated.")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main())) 