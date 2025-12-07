import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, BlipForImageTextRetrieval
import torch
import torch.nn.functional as F
from tqdm import tqdm
import hashlib
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
        """Load dataset configuration

        Dataset structures:
        - FashionIQ: images/{category}/{image_id}.jpg, split files are lists of IDs
        - CIRR: img_raw/{split}/{image_id}.png, split files are dict {id: relative_path}
        - CIRCO: unlabeled2017/{id}.jpg, scan all images
        """
        configs = {
            'fashion-iq': {
                'type': 'fashioniq',
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
                'type': 'cirr',
                'data_dir': 'cirr/img_raw',  # Base directory for relative paths
                'split_files': {
                    'train': ['cirr/image_splits/split.rc2.train.json'],
                    'val': ['cirr/image_splits/split.rc2.val.json'],
                    'test': ['cirr/image_splits/split.rc2.test1.json']
                },
                'categories': ['all']
            },
            'circo': {
                'type': 'circo',
                'data_dir': 'CIRCO/unlabeled2017',  # Direct path to images
                'annotation_files': {
                    'val': 'CIRCO/annotations/val.json',
                    'test': 'CIRCO/annotations/test.json'
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
        """Collect images from all datasets

        Handles different dataset structures:
        - FashionIQ: Split file is list of IDs, images in category subdirs
        - CIRR: Split file is dict {id: relative_path}
        - CIRCO: Scan all images in the image directory
        """
        all_images = []
        all_info = {}

        for dataset_name, config in configs.items():
            dataset_type = config.get('type', dataset_name)
            data_dir = config['data_dir']

            if dataset_type == 'fashioniq':
                self._collect_fashioniq_images(
                    config, splits, all_images, all_info
                )
            elif dataset_type == 'cirr':
                self._collect_cirr_images(
                    config, splits, all_images, all_info
                )
            elif dataset_type == 'circo':
                self._collect_circo_images(
                    config, all_images, all_info
                )
            else:
                print(f"Warning: Unknown dataset type: {dataset_type}")

        print(f"\nTotal unique images collected: {len(all_images)}")
        print(f"Dataset distribution:")

        dataset_counts = {}
        for info in all_info.values():
            key = f"{info['dataset']}/{info['split']}"
            dataset_counts[key] = dataset_counts.get(key, 0) + 1

        for key, count in sorted(dataset_counts.items()):
            print(f"  {key}: {count}")

        return all_images, all_info

    def _collect_fashioniq_images(self, config, splits, all_images, all_info):
        """Collect FashionIQ images

        FashionIQ structure:
        - Split file: list of image IDs (without extension)
        - Image path: images/{category}/{image_id}.jpg
        """
        data_dir = config['data_dir']

        for split in splits:
            if split not in config.get('split_files', {}):
                continue

            for split_file in config['split_files'][split]:
                # Extract category from filename (e.g., split.dress.train.json -> dress)
                filename = os.path.basename(split_file)
                category = None
                for cat in ['dress', 'shirt', 'toptee']:
                    if cat in filename:
                        category = cat
                        break

                if category is None:
                    print(f"Warning: Cannot determine category for {split_file}")
                    continue

                print(f"Processing fashion-iq/{category}/{split}: {split_file}")

                if not os.path.exists(split_file):
                    print(f"  Warning: Split file not found: {split_file}")
                    continue

                with open(split_file, 'r') as f:
                    image_ids = json.load(f)

                existing_count = 0
                for image_id in image_ids:
                    # FashionIQ: images/{category}/{image_id}.jpg
                    image_path = os.path.join(data_dir, category, f"{image_id}.jpg")

                    if os.path.exists(image_path):
                        img_hash = self.get_image_hash(image_path)

                        if img_hash not in all_info:
                            all_images.append(image_path)
                            all_info[img_hash] = {
                                'dataset': 'fashion-iq',
                                'category': category,
                                'split': split,
                                'image_id': image_id,
                                'image_path': image_path
                            }
                        existing_count += 1

                print(f"  Found {existing_count}/{len(image_ids)} images")

    def _collect_cirr_images(self, config, splits, all_images, all_info):
        """Collect CIRR images

        CIRR structure:
        - Split file: dict {image_id: relative_path} (e.g., "./dev/dev-xxx.png")
        - Image path: img_raw/{relative_path without ./}
        """
        data_dir = config['data_dir']

        for split in splits:
            if split not in config.get('split_files', {}):
                continue

            for split_file in config['split_files'][split]:
                print(f"Processing cirr/{split}: {split_file}")

                if not os.path.exists(split_file):
                    print(f"  Warning: Split file not found: {split_file}")
                    continue

                with open(split_file, 'r') as f:
                    # CIRR format: {image_id: "./split/image_id.png"}
                    images_dict = json.load(f)

                existing_count = 0
                for image_id, relative_path in images_dict.items():
                    # Remove leading "./" from relative path
                    clean_path = relative_path.lstrip('./')
                    image_path = os.path.join(data_dir, clean_path)

                    if os.path.exists(image_path):
                        img_hash = self.get_image_hash(image_path)

                        if img_hash not in all_info:
                            all_images.append(image_path)
                            all_info[img_hash] = {
                                'dataset': 'cirr',
                                'category': 'all',
                                'split': split,
                                'image_id': image_id,
                                'image_path': image_path
                            }
                        existing_count += 1

                print(f"  Found {existing_count}/{len(images_dict)} images")

    def _collect_circo_images(self, config, all_images, all_info):
        """Collect CIRCO images

        CIRCO structure:
        - Scan all images in unlabeled2017/
        - Image ID is filename without extension and leading zeros
        """
        data_dir = config['data_dir']

        print(f"Processing circo: scanning {data_dir}")

        if not os.path.exists(data_dir):
            print(f"  Warning: Image directory not found: {data_dir}")
            return

        # Scan all image files in directory
        image_files = []
        for filename in os.listdir(data_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(filename)

        print(f"  Found {len(image_files)} image files")

        for image_filename in image_files:
            image_path = os.path.join(data_dir, image_filename)

            if os.path.exists(image_path):
                img_hash = self.get_image_hash(image_path)

                if img_hash not in all_info:
                    # Image ID: filename without extension, strip leading zeros
                    image_id = os.path.splitext(image_filename)[0].lstrip('0') or '0'

                    all_images.append(image_path)
                    all_info[img_hash] = {
                        'dataset': 'circo',
                        'category': 'all',
                        'split': 'unlabeled',  # All CIRCO images as 'unlabeled'
                        'image_id': image_id,
                        'image_path': image_path
                    }

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

def extract_features_for_dataset(processor_instance, dataset_name, config, splits, batch_size=32, num_workers=4):
    """Extract features for a single dataset and save to its directory"""

    # Determine output directory based on dataset
    dataset_output_dirs = {
        'fashion-iq': 'fashion-iq',
        'cirr': 'cirr',
        'circo': 'CIRCO'
    }
    output_dir = dataset_output_dirs.get(dataset_name, dataset_name)

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Include model name in output filename
    model_suffix = processor_instance.model_name
    features_file = os.path.join(output_dir, f'features_{model_suffix}.pt')
    metadata_file = os.path.join(output_dir, f'metadata_{model_suffix}.pt')

    # Collect images for this dataset only
    all_images = []
    all_info = {}

    dataset_type = config.get('type', dataset_name)

    if dataset_type == 'fashioniq':
        processor_instance._collect_fashioniq_images(config, splits, all_images, all_info)
    elif dataset_type == 'cirr':
        processor_instance._collect_cirr_images(config, splits, all_images, all_info)
    elif dataset_type == 'circo':
        processor_instance._collect_circo_images(config, all_images, all_info)

    if len(all_images) == 0:
        print(f"No images found for {dataset_name}!")
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

    print(f"\nExtracting features for {dataset_name} using {processor_instance.model_name.upper()}...")
    print(f"Total images: {len(all_images)}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {processor_instance.device}")

    # Feature extraction
    all_features = []
    all_hashes = []
    hash_to_idx = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Extracting {dataset_name}")):
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

    print(f"\nFeature extraction for {dataset_name} completed!")
    print(f"Features shape: {features_tensor.shape}")
    print(f"Features dtype: {features_tensor.dtype}")

    # Prepare metadata
    idx_to_info = {}
    dataset_splits = {}

    for idx, img_hash in enumerate(all_hashes):
        if img_hash in all_info:
            info = all_info[img_hash]
            idx_to_info[idx] = info

            # Record indices by split
            key = info['split']
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
    print(f"\n=== {dataset_name} Extraction Summary ===")
    print(f"Model: {processor_instance.model_name.upper()}")
    print(f"Total images processed: {len(all_hashes)}")
    print(f"Feature dimension: {processor_instance.feature_dim}")
    print(f"Split breakdown:")
    for key, indices in dataset_splits.items():
        print(f"  {key}: {len(indices)} images")

    return features_file, metadata_file


def extract_and_save_features_pytorch(processor_instance, output_dir, batch_size=32, num_workers=4):
    """Extract and save features for each dataset separately"""

    # Load dataset configuration
    configs = processor_instance.load_dataset_configs()

    # Define splits to process
    splits = ['train', 'val', 'test']

    print(f"=== Extracting features for each dataset ===")
    print(f"Datasets: {list(configs.keys())}")
    print(f"Splits: {splits}")

    results = {}

    for dataset_name, config in configs.items():
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*50}")

        features_file, metadata_file = extract_features_for_dataset(
            processor_instance,
            dataset_name,
            config,
            splits,
            batch_size=batch_size,
            num_workers=num_workers
        )

        if features_file and metadata_file:
            results[dataset_name] = {
                'features_file': features_file,
                'metadata_file': metadata_file
            }

    print(f"\n{'='*50}")
    print(f"=== All Datasets Completed ===")
    print(f"{'='*50}")
    for dataset_name, files in results.items():
        print(f"{dataset_name}:")
        print(f"  Features: {files['features_file']}")
        print(f"  Metadata: {files['metadata_file']}")

    return results

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
    parser.add_argument('--data_dir', type=str, default='.',
                       help='Base directory containing dataset folders (fashion-iq/, cirr/, CIRCO/)')

    args = parser.parse_args()

    # Change to data directory
    if args.data_dir != '.':
        os.chdir(args.data_dir)
        print(f"Working directory: {os.getcwd()}")

    print(f"=== Multi-Dataset Feature Extraction ===")
    print(f"Model: {args.model.upper()}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: Separate files per dataset (fashion-iq/, cirr/, CIRCO/)")

    # Initialize processor
    processor = MultiDatasetImageProcessor(model_name=args.model, device=args.device)

    # Feature extraction for each dataset
    results = extract_and_save_features_pytorch(
        processor,
        output_dir=None,  # Not used, each dataset outputs to its own directory
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    if results:
        print(f"\n=== Successfully completed ===")
        for dataset_name, files in results.items():
            print(f"\n{dataset_name}:")
            print(f"  Features: {files['features_file']}")
            print(f"  Metadata: {files['metadata_file']}")

            # Quick demo for each dataset
            print(f"  --- Quick Demo ---")
            loader = PyTorchFeatureLoader(
                files['features_file'],
                files['metadata_file'],
                device=args.device
            )
            print(f"  Feature tensor shape: {loader.features.shape}")
    else:
        print("Feature extraction failed!")

if __name__ == "__main__":
    main() 