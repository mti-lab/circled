# CIRCLED: Dataset Construction Code

This repository contains the code used to construct the CIRCLED dataset, a multi-turn Composed Image Retrieval (CIR) dataset with consistent dialogues across domains.

## Overview

CIRCLED extends existing single-turn CIR datasets (FashionIQ, CIRR, CIRCO) to multi-turn scenarios through a CIReVL-based retrieval pipeline. The dataset construction involves:

1. **Image Feature Extraction**: Extract BLIP/CLIP features from source dataset images
2. **Caption Generation**: Generate image descriptions using GPT-4o-mini
3. **Multi-turn Search Simulation**: Simulate multi-turn CIR sessions using the `fmerge` and `fdiff` operations
4. **Quality Filtering**: Apply 4-stage filtering to ensure dataset quality

## Repository Structure

```
CIRCLED-code/
├── src/
│   ├── prepare_corpus.py          # Image feature extraction (BLIP/CLIP)
│   ├── extract_caption_features.py # Caption text feature extraction
│   ├── generate_captions.py       # GPT-4o-mini caption generation
│   ├── clean_captions.py          # Remove redundant phrases from captions
│   ├── multiturn_cir_system.py    # Main multi-turn CIR search system
│   ├── filtering.py               # 4-stage quality filtering
│   └── convert_to_public.py       # Convert to public dataset format
├── requirements.txt
├── .env.example
├── LICENSE
└── README.md
```

## Prerequisites

### 1. Download Source Datasets

The code expects the following source datasets to be downloaded and organized in a specific structure.

#### FashionIQ

Download from [Fashion-IQ Repository](https://github.com/XiaoxiaoGuo/fashion-iq):

```bash
# Clone the repository
git clone https://github.com/XiaoxiaoGuo/fashion-iq.git

# Download images (follow instructions in the repository)
# Images should be organized by category: dress/, shirt/, toptee/
```

#### CIRR

Download from [CIRR Repository](https://github.com/Cuberick-Orion/CIRR):

```bash
# Clone the repository
git clone https://github.com/Cuberick-Orion/CIRR.git

# Download images using the provided scripts
cd CIRR
python download_train.py
python download_dev.py
```

#### CIRCO

Download from [CIRCO Repository](https://github.com/miccunifi/CIRCO):

```bash
# Clone the repository
git clone https://github.com/miccunifi/CIRCO.git

# Download COCO 2017 unlabeled images
# See: https://cocodataset.org/#download
```

### 2. Expected Directory Structure

Organize the datasets as follows (relative to the working directory):

```
working_directory/
├── fashion-iq/
│   ├── images/
│   │   ├── dress/
│   │   │   ├── B00006M009.jpg
│   │   │   └── ...
│   │   ├── shirt/
│   │   │   └── ...
│   │   └── toptee/
│   │       └── ...
│   ├── image_splits/
│   │   ├── split.dress.train.json
│   │   ├── split.dress.val.json
│   │   ├── split.dress.test.json
│   │   ├── split.shirt.train.json
│   │   ├── split.shirt.val.json
│   │   ├── split.shirt.test.json
│   │   ├── split.toptee.train.json
│   │   ├── split.toptee.val.json
│   │   └── split.toptee.test.json
│   └── captions/
│       ├── cap.dress.train.json
│       ├── cap.dress.val.json
│       ├── cap.shirt.train.json
│       ├── cap.shirt.val.json
│       ├── cap.toptee.train.json
│       └── cap.toptee.val.json
│
├── cirr/
│   ├── img_raw/
│   │   ├── train/
│   │   │   ├── 0/
│   │   │   ├── 1/
│   │   │   └── ...
│   │   ├── dev/
│   │   │   └── ...
│   │   └── test1/
│   │       └── ...
│   ├── image_splits/
│   │   ├── split.rc2.train.json
│   │   ├── split.rc2.val.json
│   │   └── split.rc2.test1.json
│   └── captions/
│       ├── cap.rc2.train.json
│       └── cap.rc2.val.json
│
├── CIRCO/
│   ├── COCO2017_unlabeled/
│   │   └── unlabeled2017/
│   │       ├── 000000000001.jpg
│   │       └── ...
│   └── annotations/
│       ├── val.json
│       └── test.json
│
└── src/
    └── (code files from this repository)
```

### 3. Installation

```bash
pip install -r requirements.txt
```

### 4. API Keys

Copy `.env.example` to `.env` and set your OpenAI API key:

```bash
cp .env.example .env
# Edit .env and add your OPEN_API_KEY
```

## Usage

All commands should be run from the working directory containing the dataset folders.

### Step 1: Extract Image Features

Extract BLIP features for all images:

```bash
python src/prepare_corpus.py --model blip --batch_size 32 --device cuda --output_dir .
```

This generates:
- `features_blip.pt` - Image feature vectors
- `metadata_blip.pt` - Image metadata and hash mappings

### Step 2: Generate Image Captions

Generate captions for each dataset using GPT-4o-mini:

```bash
# FashionIQ
python src/generate_captions.py --dataset fashion-iq --splits train val

# CIRR
python src/generate_captions.py --dataset cirr --splits train val

# CIRCO
python src/generate_captions.py --dataset circo --splits val
```

This generates caption files like `captions_gpt4omini_fashion-iq.json`.

### Step 3: Clean Captions

Remove redundant phrases like "The image features...":

```bash
python src/clean_captions.py captions_gpt4omini_fashion-iq.json --output captions_cleaned_fashion-iq.json
python src/clean_captions.py captions_gpt4omini_cirr.json --output captions_cleaned_cirr.json
python src/clean_captions.py captions_gpt4omini_circo.json --output captions_cleaned_circo.json
```

### Step 4: Extract Caption Features

Extract CLIP/BLIP features from the generated captions:

```bash
python src/extract_caption_features.py --datasets all --output-dir caption_features
```

### Step 5: Run Multi-turn CIR System

Simulate multi-turn retrieval sessions:

```bash
# FashionIQ subsets
python src/multiturn_cir_system.py --dataset fashioniq_dress_val --max_turns 6
python src/multiturn_cir_system.py --dataset fashioniq_shirt_val --max_turns 6
python src/multiturn_cir_system.py --dataset fashioniq_toptee_val --max_turns 6

# CIRR
python src/multiturn_cir_system.py --dataset cirr_val --max_turns 6

# CIRCO
python src/multiturn_cir_system.py --dataset circo --max_turns 6
```

This generates `multiturn_cir_results_{dataset}.json` files.

### Step 6: Apply Quality Filtering

Apply 4-stage filtering:
- **Retrieval Success Filter**: Keep only successful retrievals
- **Multi-turn Filter**: Keep only sessions with 2+ turns
- **Rank Margin Filter (ε=30)**: Filter inconsistent rank progressions
- **Text Redundancy Filter (τ=0.8)**: Filter redundant text via CLIP similarity

```bash
python src/filtering.py \
    --datasets fashioniq_dress_val fashioniq_shirt_val fashioniq_toptee_val cirr_val circo \
    --similarity-threshold 0.8 \
    --rank-margin 30
```

This generates `filtered_multiturn_cir_{dataset}.json` files.

### Step 7: Convert to Public Format

Convert filtered results to the public CIRCLED format:

```bash
python src/convert_to_public.py \
    --input-dir . \
    --output-dir ./circled_public
```

## Output Format

The final public format JSON files contain:

```json
{
  "session_id": "cirr_val_0000",
  "subset": "cirr_val",
  "ground_truth_ids": ["dev-1042-2-img1"],
  "num_turns": 3,
  "turns": [
    {
      "turn": 1,
      "reference_image_id": "dev-1044-1-img1",
      "relative_caption": "Human and one animal from a different species..."
    },
    {
      "turn": 2,
      "reference_image_id": "dev-1044-0-img0",
      "relative_caption": "Replace the manta ray with a large jellyfish..."
    }
  ]
}
```

## Dataset Properties

The resulting CIRCLED dataset satisfies:

- **ε-consistency**: Each turn progressively approaches the ground truth (rank degradation ≤ ε)
- **τ-diversity**: Modification texts are sufficiently diverse (CLIP similarity < τ)

## Citation

```bibtex
@article{circled2025,
  title={CIRCLED: A Multi-turn CIR Dataset with Consistent Dialogues across Domains},
  author={Anonymous},
  journal={Under review},
  year={2025}
}
```

## License

This code is released under the CC BY 4.0 license. See [LICENSE](LICENSE) for details.

## Related Resources

- **CIRCLED Dataset**: Available on [Hugging Face](https://huggingface.co/datasets/tk1441/CIRCLED)
