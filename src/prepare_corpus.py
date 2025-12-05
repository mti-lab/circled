import json
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, BlipForImageTextRetrieval
import torch
import torch.nn.functional as F
from tqdm import tqdm
import hashlib
from pathlib import Path
import pickle
from collections import defaultdict
import argparse
import clip

class MultiDatasetImageProcessor:
    """Multi-dataset image processor (PyTorch version)"""
    
    def __init__(self, model_name="blip", device='cuda'):
        self.device = device
        self.model_name = model_name.lower()
        
        if self.model_name == "blip":
            # BLIP model setup
            self.processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
            self.model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to(device)
            self.feature_dim = 768
            print(f"Loaded BLIP model: Salesforce/blip-itm-large-coco")
        elif self.model_name == "clip":
            # CLIP model setup
            self.model, self.preprocess = clip.load("ViT-L/14", device=device)
            self.feature_dim = 768  # ViT-L/14 feature dimension
            print(f"Loaded CLIP model: ViT-L/14")
        else:
            raise ValueError(f"Unsupported model: {model_name}. Use 'blip' or 'clip'")

    def load_dataset_configs(self):
        """Load dataset configuration"""
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
                'data_dir': 'cirr/img_raw/images',
                'split_files': {
                    'train': ['cirr/image_splits/split.rc2.train.json'],
                    'val': ['cirr/image_splits/split.rc2.val.json'],
                    'test': ['cirr/image_splits/split.rc2.test1.json']
                },
                'categories': ['all']
            },
            'circo': {
                'data_dir': 'CIRCO/COCO2017_unlabeled',
                'split_files': {
                    'val': ['CIRCO/annotations/val.json'],
                    'test': ['CIRCO/annotations/test.json']
                },
                'categories': ['all']
            }
        }
        return configs

    def get_image_hash(self, image_path):
        """Calculate image file hash"""
        with open(image_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash

    def collect_all_images(self, configs, splits=['train', 'val', 'test']):
        """Collect images from all datasets"""
        all_images = []
        all_info = {}
        
        for dataset_name, config in configs.items():
            data_dir = config['data_dir']
            
            for split in splits:
                if split not in config['split_files']:
                    continue
                
                for category in config['categories']:
                    for split_file in config['split_files'][split]:
                        print(f"Processing {dataset_name}/{category}/{split}: {split_file}")
                        
                        if not os.path.exists(split_file):
                            print(f"Warning: Split file not found: {split_file}")
                            continue
                        
                        with open(split_file, 'r') as f:
                            if dataset_name == 'circo' and 'annotations' in split_file:
                                # CIRCO JSON file format
                                data = json.load(f)
                                image_ids = set()
                                for item in data:
                                    if 'reference_img_id' in item:
                                        image_ids.add(str(item['reference_img_id']))
                                    if 'target_img_id' in item:
                                        image_ids.add(str(item['target_img_id']))
                                image_ids = list(image_ids)
                            else:
                                # Normal JSON file format (list of image IDs)
                                image_ids = json.load(f)
                        
                        for image_id in image_ids:
                            if dataset_name == 'circo':
                                # CIRCO uses COCO format filenames
                                image_filename = f"{image_id.zfill(12)}.jpg"
                            else:
                                # FashionIQ and CIRR have extensions
                                image_filename = image_id
                            
                            image_path = os.path.join(data_dir, image_filename)
                            
                            if os.path.exists(image_path):
                                img_hash = self.get_image_hash(image_path)
                                
                                if img_hash not in all_info:
                                    all_images.append(image_path)
                                    all_info[img_hash] = {
                                        'dataset': dataset_name,
                                        'category': category,
                                        'split': split,
                                        'image_id': image_id,
                                        'image_path': image_path
                                    }
                            else:
                                print(f"Warning: Image not found: {image_path}")
        
        print(f"\nTotal unique images collected: {len(all_images)}")
        print(f"Dataset distribution:")
        
        dataset_counts = {}
        for info in all_info.values():
            key = f"{info['dataset']}/{info['split']}"
            dataset_counts[key] = dataset_counts.get(key, 0) + 1
        
        for key, count in dataset_counts.items():
            print(f"  {key}: {count}")
        
        return all_images, all_info

class ImageFeatureDataset(Dataset):
    """Dataset for image feature extraction"""
    
    def __init__(self, image_list, processor_instance):
        self.image_list = image_list
        self.processor_instance = processor_instance

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.processor_instance.model_name == "blip":
                # BLIP processing
                processed_image = self.processor_instance.processor(images=image, return_tensors="pt")['pixel_values'][0]
            elif self.processor_instance.model_name == "clip":
                # CLIP processing
                processed_image = self.processor_instance.preprocess(image)
            
            return {
                'image': processed_image,
                'image_path': image_path,
                'idx': idx
            }
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Return black image on error
            if self.processor_instance.model_name == "blip":
                processed_image = torch.zeros(3, 384, 384)  # BLIP input size
            else:
                processed_image = torch.zeros(3, 224, 224)  # CLIP input size
            
            return {
                'image': processed_image,
                'image_path': image_path,
                'idx': idx
            }

def collate_fn(batch):
    """Custom batch processing function"""
    # Exclude None
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    images = torch.stack([item['image'] for item in batch])
    image_paths = [item['image_path'] for item in batch]
    indices = torch.tensor([item['idx'] for item in batch])
    
    return {
        'images': images,
        'image_paths': image_paths,
        'indices': indices
    }

def extract_and_save_features_pytorch(processor_instance, output_dir, batch_size=32, num_workers=4):
    """Extract and save features in PyTorch format"""
    
    # Include model name in output filename
    model_suffix = processor_instance.model_name
    features_file = os.path.join(output_dir, f'features_{model_suffix}.pt')
    metadata_file = os.path.join(output_dir, f'metadata_{model_suffix}.pt')
    
    # Load dataset configuration
    configs = processor_instance.load_dataset_configs()
    
    # Collect all images
    all_images, all_info = processor_instance.collect_all_images(configs)
    
    if len(all_images) == 0:
        print("No images found!")
        return None, None
    
    # Create dataset and dataloader
    dataset = ImageFeatureDataset(all_images, processor_instance)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if processor_instance.device == 'cuda' else False
    )
    
    print(f"\nExtracting features using {processor_instance.model_name.upper()}...")
    print(f"Total images: {len(all_images)}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {processor_instance.device}")
    
    # Feature extraction
    all_features = []
    all_hashes = []
    hash_to_idx = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
            if batch is None:
                continue
            
            images = batch['images'].to(processor_instance.device)
            image_paths = batch['image_paths']
            indices = batch['indices']
            
            # Model-specific feature extraction
            if processor_instance.model_name == "blip":
                # BLIP feature extraction
                vision_outputs = processor_instance.model.vision_model(pixel_values=images)
                features = vision_outputs.last_hidden_state[:, 0, :].to(torch.float32)
                features = F.normalize(processor_instance.model.vision_proj(features), dim=-1)
            
            elif processor_instance.model_name == "clip":
                # CLIP feature extraction
                features = processor_instance.model.encode_image(images).to(torch.float32)
                features = F.normalize(features, dim=-1)
            
            # Move to CPU and save
            features_cpu = features.cpu()
            all_features.append(features_cpu)
            
            # Calculate hash
            for i, (path, idx) in enumerate(zip(image_paths, indices)):
                img_hash = processor_instance.get_image_hash(path)
                all_hashes.append(img_hash)
                hash_to_idx[img_hash] = len(all_hashes) - 1
    
    # Concatenate features
    features_tensor = torch.cat(all_features, dim=0)
    
    print(f"\nFeature extraction completed!")
    print(f"Features shape: {features_tensor.shape}")
    print(f"Features dtype: {features_tensor.dtype}")
    
    # Prepare metadata
    idx_to_info = {}
    dataset_splits = {}
    
    for idx, img_hash in enumerate(all_hashes):
        if img_hash in all_info:
            info = all_info[img_hash]
            idx_to_info[idx] = info
            
            # Record indices by dataset/split
            key = f"{info['dataset']}/{info['split']}"
            if key not in dataset_splits:
                dataset_splits[key] = []
            dataset_splits[key].append(idx)
    
    metadata = {
        'model_name': processor_instance.model_name,
        'feature_dim': processor_instance.feature_dim,
        'total_images': len(all_hashes),
        'hash_to_idx': hash_to_idx,
        'idx_to_info': idx_to_info,
        'dataset_splits': dataset_splits,
        'extraction_info': {
            'batch_size': batch_size,
            'device': processor_instance.device,
            'num_workers': num_workers
        }
    }
    
    # Save
    print(f"\nSaving features to: {features_file}")
    torch.save({
        'features': features_tensor,
        'hashes': all_hashes,
        'hash_to_idx': hash_to_idx
    }, features_file)
    
    print(f"Saving metadata to: {metadata_file}")
    torch.save(metadata, metadata_file)
    
    # Display statistics
    print(f"\n=== Extraction Summary ===")
    print(f"Model: {processor_instance.model_name.upper()}")
    print(f"Total images processed: {len(all_hashes)}")
    print(f"Feature dimension: {processor_instance.feature_dim}")
    print(f"Dataset breakdown:")
    for key, indices in dataset_splits.items():
        print(f"  {key}: {len(indices)} images")
    
    return features_file, metadata_file

class PyTorchFeatureLoader:
    """PyTorch format feature loader"""
    
    def __init__(self, features_file, metadata_file, device='cpu'):
        self.device = device
        self.features_file = features_file
        self.metadata_file = metadata_file
        
        # Load metadata
        self.metadata = torch.load(metadata_file, map_location='cpu')
        
        # Features are lazy loaded
        self._features = None
        self._hash_to_idx = self.metadata['hash_to_idx']
        
        print(f"Loaded metadata for {self.metadata['model_name'].upper()} features")
        print(f"Total images: {self.metadata['total_images']}")
        print(f"Feature dimension: {self.metadata['feature_dim']}")

    def _load_features(self):
        """Load features (lazy loading)"""
        if self._features is None:
            print(f"Loading features to {self.device}...")
            feature_data = torch.load(self.features_file, map_location=self.device)
            self._features = feature_data['features']
        return self._features

    @property
    def features(self):
        """Get all features"""
        return self._load_features()

    def get_features_by_dataset(self, dataset, split):
        """Get features by dataset and split"""
        key = f"{dataset}/{split}"
        if key not in self.metadata['dataset_splits']:
            raise KeyError(f"Dataset split not found: {key}")
        
        indices = self.metadata['dataset_splits'][key]
        features = self._load_features()
        
        return features[indices], torch.tensor(indices)

    def get_features_by_category(self, dataset, category, split):
        """Get features by category"""
        key = f"{dataset}/{split}"
        if key not in self.metadata['dataset_splits']:
            raise KeyError(f"Dataset split not found: {key}")
        
        indices = []
        for idx in self.metadata['dataset_splits'][key]:
            info = self.metadata['idx_to_info'][idx]
            if info['category'] == category:
                indices.append(idx)
        
        features = self._load_features()
        return features[indices], torch.tensor(indices)

    def get_feature_by_hash(self, img_hash):
        """Get features by hash"""
        if img_hash not in self._hash_to_idx:
            raise KeyError(f"Image hash not found: {img_hash}")
        
        idx = self._hash_to_idx[img_hash]
        features = self._load_features()
        return features[idx]

    def get_features_by_indices(self, indices):
        """Get features by index list"""
        features = self._load_features()
        return features[indices]

    def get_info_by_idx(self, idx):
        """Get image info by index"""
        return self.metadata['idx_to_info'][idx]

    def compute_similarity(self, query_features, corpus_features=None):
        """Calculate cosine similarity"""
        if corpus_features is None:
            corpus_features = self._load_features()
        
        # Fast computation on GPU
        query_features = query_features.to(self.device)
        corpus_features = corpus_features.to(self.device)
        
        return torch.mm(query_features, corpus_features.t())

def main():
    parser = argparse.ArgumentParser(description='Extract image features for multi-dataset CIR (PyTorch format)')
    parser.add_argument('--model', choices=['blip', 'clip'], default='blip', 
                       help='Model to use for feature extraction')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', default='cuda', help='Device for feature computation')
    parser.add_argument('--output_dir', default='.', help='Output directory for features')
    
    args = parser.parse_args()
    
    print(f"=== Multi-Dataset Feature Extraction ===")
    print(f"Model: {args.model.upper()}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize processor
    processor = MultiDatasetImageProcessor(model_name=args.model, device=args.device)
    
    # Feature extraction
    features_file, metadata_file = extract_and_save_features_pytorch(
        processor,
        args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    if features_file and metadata_file:
        print(f"\n=== Successfully completed ===")
        print(f"Features: {features_file}")
        print(f"Metadata: {metadata_file}")
        
        # Quick demo
        print(f"\n=== Quick Demo ===")
        loader = PyTorchFeatureLoader(features_file, metadata_file, device=args.device)
        print(f"Feature tensor shape: {loader.features.shape}")
        print(f"Feature tensor device: {loader.features.device}")
    else:
        print("Feature extraction failed!")

if __name__ == "__main__":
    main() 