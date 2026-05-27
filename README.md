# CIRCLED: Dataset Construction Code

[![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/tk1441/CIRCLED)

This repository contains the code used to construct the CIRCLED dataset, a multi-turn Composed Image Retrieval (CIR) dataset with consistent dialogues across domains.

## Overview

CIRCLED extends existing single-turn CIR datasets (FashionIQ, CIRR, CIRCO) to multi-turn scenarios through a CIReVL-based retrieval pipeline. The dataset construction involves:

1. **Image Feature Extraction**: Extract BLIP/CLIP features from source dataset images
2. **Caption Generation**: Generate image descriptions using GPT-4o-mini
3. **Multi-turn Search Simulation**: Simulate multi-turn CIR sessions using the `fmerge` and `fdiff` operations
4. **Quality Filtering**: Apply 4-stage filtering to ensure dataset quality

## Prerequisites

### 1. Download Source Datasets

Download the following datasets:

- **FashionIQ**: [GitHub](https://github.com/XiaoxiaoGuo/fashion-iq)
- **CIRR**: [GitHub](https://github.com/Cuberick-Orion/CIRR)
- **CIRCO**: [GitHub](https://github.com/miccunifi/CIRCO) + [COCO 2017 unlabeled](https://cocodataset.org/#download)

### 2. Directory Structure

```
data/                            # Source data (--data_dir)
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
│   │   ├── split.shirt.train.json
│   │   ├── split.shirt.val.json
│   │   ├── split.toptee.train.json
│   │   └── split.toptee.val.json
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
└── CIRCO/
    ├── unlabeled2017/
    │   ├── 000000000001.jpg
    │   └── ...
    └── annotations/
        ├── val.json
        └── test.json

output/                          # Generated results (separate from data)
├── raw/                         # multiturn_cir_system.py output
│   └── multiturn_cir_results_*.json
├── filtered/                    # filtering.py output
│   └── filtered_multiturn_cir_*.json
└── public/                      # convert_to_public.py output
    └── {subset}.json
```

### 3. Installation

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your OPENAI_API_KEY
```

## Usage

All scripts support `--data_dir` to specify the data directory. Run from the data directory or use `--data_dir /path/to/data`.

### Step 1: Extract Image Features

```bash
python src/prepare_corpus.py --model blip --batch_size 32 --device cuda
```

### Step 2: Generate Image Captions

```bash
python src/generate_captions.py --dataset fashion-iq --splits train val
python src/generate_captions.py --dataset cirr --splits train val
python src/generate_captions.py --dataset circo --splits val
```

### Step 3: Extract Caption Features

```bash
python src/extract_caption_features.py --datasets all
```

### Step 4: Run Multi-turn CIR System

```bash
python src/multiturn_cir_system.py --dataset fashioniq_dress_val --max_turns 6
python src/multiturn_cir_system.py --dataset cirr_val --max_turns 6
python src/multiturn_cir_system.py --dataset circo --max_turns 6
```

Output: `output/raw/multiturn_cir_results_{dataset}.json`

### Step 5: Apply Quality Filtering

```bash
python src/filtering.py \
    --datasets fashioniq_dress_val fashioniq_shirt_val fashioniq_toptee_val cirr_val circo \
    --similarity-threshold 0.8 \
    --rank-margin 30
```

Output: `output/filtered/filtered_multiturn_cir_{dataset}.json`

### Step 6: Convert to Public Format

```bash
python src/convert_to_public.py --input-dir output/filtered --output-dir output/public
```

## Output Format

```json
{
  "session_id": "cirr_val_0000",
  "subset": "cirr_val",
  "ground_truth_ids": ["dev-1042-2-img1"],
  "num_turns": 3,
  "turns": [
    {"turn": 1, "reference_image_id": "dev-1044-1-img1", "relative_caption": "..."},
    {"turn": 2, "reference_image_id": "dev-1044-0-img0", "relative_caption": "..."}
  ]
}
```

## Dataset Properties

- **ε-consistency**: Each turn progressively approaches the ground truth (rank degradation ≤ ε)
- **τ-diversity**: Modification texts are sufficiently diverse (CLIP similarity < τ)

## Citation

```bibtex
@article{circled2026,
  title={CIRCLED: A Multi-turn CIR Dataset with Consistent Dialogues across Domains},
  author={Tomohisa Takeda and Yu-Chieh Lin and Yuji Nozawa and Youyang Ng and Osamu Torii and Yusuke Matsui},
  year={2026},
  note={Under review}
}
```

## License

CC BY 4.0. See [LICENSE](LICENSE) for details.

## Related Resources

- **CIRCLED Dataset**: [Hugging Face](https://huggingface.co/datasets/tk1441/CIRCLED)
