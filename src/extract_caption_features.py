#!/usr/bin/env python3
"""
Script to convert GPT-4o mini generated captions to CLIP and BLIP feature vectors.
Supports CIRR, CIRCO, and Fashion-IQ datasets.
"""

import os
import json
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import clip
from transformers import (
    BlipProcessor, BlipForImageTextRetrieval
)

# Cache directory can be set via environment variables:
# HF_HOME, TRANSFORMERS_CACHE

class CaptionFeatureExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize CLIP model (OpenCLIP ViT-L/14)
        print("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)
        self.clip_model.eval()
        
        # Initialize BLIP model (using large version)
        print("Loading BLIP model...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
        self.blip_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to(device)
        self.blip_model.eval()
        
        print("Model loading complete")
    
    def extract_clip_features(self, captions, batch_size=32):
        """Extract caption features using CLIP"""
        print(f"Starting CLIP feature extraction: {len(captions)} captions")
        
        features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(captions), batch_size), desc="CLIP feature extraction"):
                batch_captions = captions[i:i+batch_size]
                
                # Tokenize text with OpenCLIP
                text_tokens = clip.tokenize(batch_captions, truncate=True).to(self.device)
                
                # Extract features
                text_features = self.clip_model.encode_text(text_tokens).to(torch.float32)
                text_features = F.normalize(text_features, dim=-1)
                
                features.append(text_features.cpu().numpy())
        
        return np.vstack(features)
    
    def extract_blip_features(self, captions, batch_size=16):
        """Extract caption features using BLIP"""
        print(f"Starting BLIP feature extraction: {len(captions)} captions")
        
        features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(captions), batch_size), desc="BLIP feature extraction"):
                batch_captions = captions[i:i+batch_size]
                
                # Tokenize text
                inputs = self.blip_processor(
                    text=batch_captions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # Extract text features using BLIP model (same method as reference code)
                question_embeds = self.blip_model.text_encoder(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    return_dict=True
                ).last_hidden_state
                
                # Use CLS token (first token) and pass through text_proj
                text_features = self.blip_model.text_proj(question_embeds[:, 0, :])
                text_features = F.normalize(text_features, dim=-1)
                
                features.append(text_features.cpu().numpy())
        
        return np.vstack(features)

def load_gpt4omini_captions(caption_file):
    """Load GPT-4o mini caption file"""
    print(f"Loading caption file: {caption_file}")
    
    with open(caption_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    captions = []
    image_ids = []
    
    # Process according to data structure
    if isinstance(data, dict):
        for image_id, caption in data.items():
            image_ids.append(image_id)
            captions.append(caption)
    elif isinstance(data, list):
        for item in data:
            if 'image_id' in item and 'caption' in item:
                image_ids.append(item['image_id'])
                captions.append(item['caption'])
            elif 'id' in item and 'caption' in item:
                image_ids.append(item['id'])
                captions.append(item['caption'])
    
    print(f"Loading complete: {len(captions)} captions")
    return captions, image_ids

def save_features(features, image_ids, output_file, feature_type):
    """Save features to file"""
    print(f"Saving {feature_type} features: {output_file}")
    
    # Feature data structure
    feature_data = {
        'features': features,
        'image_ids': image_ids,
        'feature_type': feature_type,
        'feature_dim': features.shape[1]
    }
    
    torch.save(feature_data, output_file)
    print(f"Saved: {features.shape} -> {output_file}")

def merge_features(existing_features, existing_image_ids, new_features, new_image_ids, feature_type):
    """Merge existing features with new features"""
    print(f"Merging {feature_type} features...")
    print(f"  Existing: {len(existing_image_ids)} images")
    print(f"  New: {len(new_image_ids)} images")
    
    # Convert numpy arrays to tensors if needed
    if isinstance(existing_features, np.ndarray):
        existing_features = torch.from_numpy(existing_features).float()
    if isinstance(new_features, np.ndarray):
        new_features = torch.from_numpy(new_features).float()
    
    # Concatenate features
    merged_features = torch.cat([existing_features, new_features], dim=0)
    merged_image_ids = existing_image_ids + new_image_ids
    
    print(f"  After merge: {len(merged_image_ids)} images")
    return merged_features, merged_image_ids

def extract_missing_features(extractor, all_captions, all_image_ids, existing_image_ids, feature_type):
    """Extract features only for missing images"""
    # Identify missing image IDs
    existing_set = set(existing_image_ids)
    missing_indices = [i for i, img_id in enumerate(all_image_ids) if img_id not in existing_set]
    
    if not missing_indices:
        print(f"{feature_type}: No missing images")
        return None, []
    
    print(f"{feature_type}: Detected {len(missing_indices)} missing images")
    
    # Extract captions for missing images
    missing_captions = [all_captions[i] for i in missing_indices]
    missing_image_ids = [all_image_ids[i] for i in missing_indices]
    
    # Extract features
    if feature_type == 'CLIP':
        missing_features = extractor.extract_clip_features(missing_captions)
    elif feature_type == 'BLIP':
        missing_features = extractor.extract_blip_features(missing_captions)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    return missing_features, missing_image_ids

def process_dataset(dataset_name, caption_file, output_dir, extractor, force_reprocess=False):
    """Process each dataset (with incremental update support)"""
    print(f"\n=== Processing {dataset_name} dataset ===")
    
    # Load captions
    captions, image_ids = load_gpt4omini_captions(caption_file)
    
    if len(captions) == 0:
        print(f"Warning: No captions found for {dataset_name}")
        return
    
    # Create output directory
    dataset_output_dir = Path(output_dir) / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file paths
    clip_output_file = dataset_output_dir / 'gpt4omini_captions_clip_features.pt'
    blip_output_file = dataset_output_dir / 'gpt4omini_captions_blip_features.pt'
    
    # CLIP feature processing (with incremental update support)
    if clip_output_file.exists() and not force_reprocess:
        print(f"CLIP feature file already exists: {clip_output_file}")
        try:
            existing_data = torch.load(clip_output_file, weights_only=False)
            existing_image_ids = existing_data['image_ids']
            existing_features = existing_data['features']
            
            if len(existing_image_ids) == len(image_ids) and existing_image_ids == image_ids:
                print("CLIP features already complete. Skipping.")
            else:
                print(f"Existing CLIP features: {len(existing_image_ids)} images, Required: {len(image_ids)} images")
                
                # Extract only missing features
                missing_features, missing_image_ids = extract_missing_features(
                    extractor, captions, image_ids, existing_image_ids, 'CLIP'
                )

                if missing_features is not None:
                    # Merge existing and new features
                    merged_features, merged_image_ids = merge_features(
                        existing_features, existing_image_ids, 
                        missing_features, missing_image_ids, 'CLIP'
                    )
                    save_features(merged_features, merged_image_ids, clip_output_file, 'CLIP')
                else:
                    print("CLIP features: No images to add")
                    
        except Exception as e:
            print(f"Failed to load existing CLIP feature file: {e}")
            print("Reprocessing CLIP features from scratch.")
            clip_features = extractor.extract_clip_features(captions)
            save_features(clip_features, image_ids, clip_output_file, 'CLIP')
    else:
        if force_reprocess and clip_output_file.exists():
            print("Force reprocess mode: Overwriting existing CLIP feature file.")
        else:
            print("Creating new CLIP features.")
        clip_features = extractor.extract_clip_features(captions)
        save_features(clip_features, image_ids, clip_output_file, 'CLIP')
    
    # BLIP feature processing (with incremental update support)
    if blip_output_file.exists() and not force_reprocess:
        print(f"BLIP feature file already exists: {blip_output_file}")
        try:
            existing_data = torch.load(blip_output_file, weights_only=False)
            existing_image_ids = existing_data['image_ids']
            existing_features = existing_data['features']
            
            if len(existing_image_ids) == len(image_ids) and existing_image_ids == image_ids:
                print("BLIP features already complete. Skipping.")
            else:
                print(f"Existing BLIP features: {len(existing_image_ids)} images, Required: {len(image_ids)} images")
                
                # Extract only missing features
                missing_features, missing_image_ids = extract_missing_features(
                    extractor, captions, image_ids, existing_image_ids, 'BLIP'
                )

                if missing_features is not None:
                    # Merge existing and new features
                    merged_features, merged_image_ids = merge_features(
                        existing_features, existing_image_ids, 
                        missing_features, missing_image_ids, 'BLIP'
                    )
                    save_features(merged_features, merged_image_ids, blip_output_file, 'BLIP')
                else:
                    print("BLIP features: No images to add")
                    
        except Exception as e:
            print(f"Failed to load existing BLIP feature file: {e}")
            print("Reprocessing BLIP features from scratch.")
            blip_features = extractor.extract_blip_features(captions)
            save_features(blip_features, image_ids, blip_output_file, 'BLIP')
    else:
        if force_reprocess and blip_output_file.exists():
            print("Force reprocess mode: Overwriting existing BLIP feature file.")
        else:
            print("Creating new BLIP features.")
        blip_features = extractor.extract_blip_features(captions)
        save_features(blip_features, image_ids, blip_output_file, 'BLIP')
    
    print(f"=== {dataset_name} dataset processing complete ===")

def main():
    parser = argparse.ArgumentParser(description='Extract features from GPT-4o mini captions')
    parser.add_argument('--datasets', nargs='+',
                       choices=['cirr', 'circo', 'fashion-iq', 'all'],
                       default=['all'],
                       help='Datasets to process')
    parser.add_argument('--output-dir', type=str,
                       default='caption_features',
                       help='Output directory')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Force reprocess ignoring existing feature files')
    
    args = parser.parse_args()
    
    print("=== Starting GPT-4o mini caption feature extraction ===")
    print(f"Device: {args.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if args.force_reprocess:
        print("Force reprocess mode: Overwriting existing files")
    
    # Initialize feature extractor
    extractor = CaptionFeatureExtractor(device=args.device)
    
    # Dataset configuration (using cleaned caption files)
    dataset_configs = {
        'cirr': {
            'caption_file': './cirr/captions_gpt4omini.json',
            'name': 'CIRR'
        },
        'circo': {
            'caption_file': './circo/captions_gpt4omini_circo.json',
            'name': 'CIRCO'
        },
        'fashion-iq': {
            'caption_file': './fashion-iq/captions_gpt4omini_fashion-iq.json',
            'name': 'Fashion-IQ'
        }
    }
    
    # Determine which datasets to process
    if 'all' in args.datasets:
        datasets_to_process = list(dataset_configs.keys())
    else:
        datasets_to_process = args.datasets
    
    # Process each dataset
    for dataset_key in datasets_to_process:
        if dataset_key in dataset_configs:
            config = dataset_configs[dataset_key]
            caption_file = config['caption_file']
            
            # Check if file exists
            if os.path.exists(caption_file):
                try:
                    process_dataset(
                        config['name'], 
                        caption_file, 
                        args.output_dir, 
                        extractor,
                        force_reprocess=args.force_reprocess
                    )
                except Exception as e:
                    print(f"Error: An error occurred while processing {config['name']}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Warning: {caption_file} not found")
        else:
            print(f"Warning: Unknown dataset: {dataset_key}")
    
    print("\n=== All processing complete ===")

if __name__ == "__main__":
    main() 