#!/usr/bin/env python3
"""
マルチターンCIRシステム - CIRCO、CIRR、FashionIQに対応
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

# .envファイルから環境変数を読み込み
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Make sure OPENAI_API_KEY is set as environment variable.")

# OpenAI APIキーの設定
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
    """データセット別のローダークラス"""
    
    @staticmethod
    def load_circo_data(annotation_file: str) -> List[Dict]:
        """CIRCOデータを読み込み"""
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # メタデータファイルから画像ID→ハッシュ値のマッピングを作成
        import torch
        metadata_file = 'CIRCO/metadata_blip.pt'
        metadata = torch.load(metadata_file, map_location='cpu', weights_only=False)
        
        # 画像ID（6桁数値）→ハッシュ値のマッピングを作成
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
            # 数値IDを6桁文字列に変換してハッシュ値を取得
            ref_id_str = str(item['reference_img_id'])
            target_id_str = str(item['target_img_id'])
            gt_id_strs = [str(gt_id) for gt_id in item['gt_img_ids']]
            
            # ハッシュ値に変換（見つからない場合は元の12桁形式を保持）
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
                # デバッグ用に元のIDも保持
                'original_reference_id': f"{item['reference_img_id']:012d}.jpg",
                'original_target_id': f"{item['target_img_id']:012d}.jpg",
                'original_gt_ids': [f"{gt_id:012d}.jpg" for gt_id in item['gt_img_ids']]
            })
        return formatted_data
    
    @staticmethod
    def load_cirr_data(annotation_file: str) -> List[Dict]:
        """CIRRデータを読み込み（単一のhard targetのみをground truthとする）"""
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        formatted_data = []
        for item in data:
            # CIRRは基本的に単一のhard targetを評価対象とする
            formatted_data.append({
                'reference_image_id': item['reference'],
                'target_image_id': item['target_hard'],  # メインターゲット
                'relative_caption': item['caption'],
                'ground_truth_ids': [item['target_hard']],  # 単一のhard targetのみ
                'id': item.get('pairid', item.get('id', 0))
            })
        return formatted_data
    
    @staticmethod
    def load_fashioniq_data(annotation_file: str, caption_mode: str = 'separate') -> List[Dict]:
        """FashionIQデータを読み込み
        
        Args:
            annotation_file: アノテーションファイルのパス
            caption_mode: キャプション処理方式
                - 'separate': 各キャプションを独立したサンプルとして扱う（推奨・標準）
                - 'combined': 複数キャプションを統合
                - 'first_only': 最初のキャプションのみ使用
        """
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        formatted_data = []
        query_id_counter = 0  # 明示的なIDカウンター
        
        # 統計情報
        total_items = len(data)
        skipped_empty_captions = 0
        skipped_no_valid_captions = 0
        
        for item in data:
            # 実際のFashion-IQファイル形式に対応
            target_id = item['target']      # ターゲット画像ID
            candidate_id = item['candidate'] # 参照画像ID（candidateが実際の参照画像）
            captions = item['captions']     # relative captions のリスト
            
            # 空文字列やホワイトスペースのみのキャプションを除外
            valid_captions = [caption.strip() for caption in captions if caption and caption.strip()]
            
            # 有効なキャプションがない場合はスキップ
            if not valid_captions:
                skipped_no_valid_captions += 1
                continue
            
            if caption_mode == 'separate':
                # 方針1: 各キャプションを独立したサンプルとして扱う（標準アプローチ）
                for i, caption in enumerate(valid_captions):
                    formatted_data.append({
                        'reference_image_id': candidate_id,
                        'target_image_id': target_id, 
                        'relative_caption': caption,  # 個別のキャプション（空文字列除外済み）
                        'ground_truth_ids': [target_id],
                        'id': query_id_counter,  # 明示的なIDを設定
                        'caption_index': i,  # どのキャプションかを記録
                        'original_captions': captions,  # 元のキャプション（フィルタ前）
                        'valid_captions': valid_captions  # 有効なキャプションのみ
                    })
                    query_id_counter += 1
                    
            elif caption_mode == 'combined':
                # 方針2: 複数キャプションを自然な文章として統合
                if len(valid_captions) == 1:
                    combined_caption = valid_captions[0]
                elif len(valid_captions) == 2:
                    combined_caption = f"{valid_captions[0]} and {valid_captions[1]}"
                else:
                    # 3つ以上の場合（稀）
                    combined_caption = ", ".join(valid_captions[:-1]) + f", and {valid_captions[-1]}"
                
                formatted_data.append({
                    'reference_image_id': candidate_id,
                    'target_image_id': target_id, 
                    'relative_caption': combined_caption,  # 統合されたキャプション
                    'ground_truth_ids': [target_id],
                    'id': query_id_counter,  # 明示的なIDを設定
                    'original_captions': captions,  # 元のキャプション（フィルタ前）
                    'valid_captions': valid_captions  # 有効なキャプションのみ
                })
                query_id_counter += 1
                
            elif caption_mode == 'first_only':
                # 方針3: 最初の有効なキャプションのみ使用
                formatted_data.append({
                    'reference_image_id': candidate_id,
                    'target_image_id': target_id, 
                    'relative_caption': valid_captions[0],  # 最初の有効なキャプションのみ
                    'ground_truth_ids': [target_id],
                    'id': query_id_counter,  # 明示的なIDを設定
                    'original_captions': captions,  # 元のキャプション（フィルタ前）
                    'valid_captions': valid_captions  # 有効なキャプションのみ
                })
                query_id_counter += 1
            
            # 空文字列キャプションがあった場合の統計
            if len(valid_captions) < len(captions):
                skipped_empty_captions += len(captions) - len(valid_captions)
        
        # デバッグ情報の表示
        print(f"Fashion-IQ data loading summary:")
        print(f"  Total annotation items: {total_items}")
        print(f"  Items with no valid captions (skipped): {skipped_no_valid_captions}")
        print(f"  Empty/whitespace captions filtered: {skipped_empty_captions}")
        print(f"  Generated query samples: {len(formatted_data)}")
        print(f"  Caption mode: {caption_mode}")
        
        return formatted_data

class ModelManager:
    """モデル管理クラス"""
    
    def __init__(self, use_blip: bool = True):
        self.use_blip = use_blip
        self.load_retrieval_models()
        self.load_clip_model()  # CLIP類似度チェック用
        self.setup_openai_client()
    
    def load_retrieval_models(self):
        """検索用モデルをロード"""
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
        """CLIP類似度チェック用モデルをロード"""
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
        """OpenAI APIクライアントの設定"""
        # OpenAI APIキーが設定されていることを確認
        if not openai.api_key:
            raise ValueError("OpenAI API key is not set. Please set OPENAI_API_KEY environment variable.")
        
        print("OpenAI GPT-4o-mini client ready for caption generation")
    
    async def encode_image_to_base64(self, image_path: str) -> str:
        """画像をBase64エンコードする"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    async def call_gpt4o_mini(self, messages: List[Dict], max_tokens: int = 100) -> str:
        """GPT-4o-miniにAPIコールする"""
        client = AsyncOpenAI(api_key=openai.api_key)
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()

    async def combine_captions_with_gpt4o(self, original_caption: str, relative_caption: str) -> str:
        """GPT-4o-miniを使ってキャプションを組み合わせ"""
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
        
        # 不要な接頭語や冗長な表現を除去
        response = response.strip()
        
        # より包括的な不要接頭語リスト
        unwanted_prefixes = [
            "Comprehensive Caption:",
            "New caption:",
            "Caption:",
            "Combined caption:",
            "Result:",
            "This striking image",
            "The image shows",
            "In this image"
        ]
        
        for prefix in unwanted_prefixes:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()
                if response.startswith(':'):
                    response = response[1:].strip()
        
        return response.strip()

    async def generate_relative_caption_with_gpt4o(self, ref_image_path: str, target_image_path: str, 
                                                  previous_captions: List[str] = None, 
                                                  similarity_threshold: float = 0.8,
                                                  max_retries: int = 3) -> str:
        """GPT-4o-miniを使ってrelative captionを生成（CLIP類似度チェック付き）"""
        if not os.path.exists(ref_image_path):
            raise FileNotFoundError(f"Reference image file not found: {ref_image_path}")
        if not os.path.exists(target_image_path):
            raise FileNotFoundError(f"Target image file not found: {target_image_path}")
        
        # CLIPモデルが利用できない場合は通常の生成のみ実行
        if not (hasattr(self, 'clip_model') and self.clip_model is not None):
            print("CLIP model not available. Skipping similarity check.")
            # 通常のrelative caption生成（既存のループ処理の中核部分を使用）
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
            # 引用符や余分な文字を除去
            return generated_caption.strip('"\'')
        
        # CLIP特徴量計算用の関数
        def get_clip_text_feature(text: str) -> torch.Tensor:
            """CLIPテキスト特徴量を取得"""
            import clip
            device = next(self.clip_model.parameters()).device
            with torch.no_grad():
                tokens = clip.tokenize(text, truncate=True).to(device)
                feat = self.clip_model.encode_text(tokens)
                return torch.nn.functional.normalize(feat, dim=-1).squeeze(0)
        
        # 過去のキャプションの特徴量を計算
        previous_features = []
        if previous_captions:
            for caption in previous_captions:
                if caption.strip():
                    feat = get_clip_text_feature(caption)
                    previous_features.append(feat)
        
        # 画像をBase64エンコード
        ref_image_b64 = await self.encode_image_to_base64(ref_image_path)
        target_image_b64 = await self.encode_image_to_base64(target_image_path)
        
        # 再試行ループ
        for attempt in range(max_retries):
            # 前回のキャプションがある場合の指示
            previous_instructions = ""
            if previous_captions and len(previous_captions) > 0:
                previous_instructions = (
                    "IMPORTANT - Previous changes have already been suggested:\n" +
                    "\n".join(f'• "{cap}"' for cap in previous_captions) +
                    "\n\nYour task is to identify a COMPLETELY DIFFERENT visual change. "
                    "Focus on aspects that have NOT been mentioned before.\n\n"
                )
            
            # 再試行の場合の追加指示
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
            
            # 引用符や余分な文字を除去
            generated_caption = generated_caption.strip('"\'')
            
            if not generated_caption:
                if attempt == max_retries - 1:
                    raise ValueError(f"GPT-4o generated empty caption after {max_retries} attempts for images: {ref_image_path}, {target_image_path}")
                continue
            
            # CLIP類似度チェック
            if previous_features:
                current_feature = get_clip_text_feature(generated_caption)
                
                # 過去のキャプションとの類似度を計算
                is_too_similar = False
                max_similarity = 0.0
                
                for prev_feat in previous_features:
                    similarity = torch.dot(current_feature, prev_feat).item()
                    max_similarity = max(max_similarity, similarity)
                    
                    if similarity >= similarity_threshold:
                        is_too_similar = True
                        break
                
                # 類似度が閾値以下の場合は成功
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
                # 過去のキャプションがない場合はそのまま返す
                return generated_caption
        
        # ここには到達しないはずだが、安全のため
        return generated_caption

class RetrievalEngine:
    """画像検索エンジン"""
    
    def __init__(self, corpus_vectors_file: str, search_space_file: str, model_manager: ModelManager):
        # 特徴量ファイルの存在確認
        if not os.path.exists(corpus_vectors_file):
            raise FileNotFoundError(f"Corpus vectors file not found: {corpus_vectors_file}")
        
        # 特徴量ファイルを読み込み
        corpus_data = torch.load(corpus_vectors_file, map_location=device, weights_only=False)
        
        # 特徴量ファイルの形式に応じて処理
        if isinstance(corpus_data, dict):
            # 辞書形式の場合：{'features': tensor, 'hashes': list, ...}
            if 'features' in corpus_data and 'hashes' in corpus_data:
                all_corpus_features = corpus_data['features']
                all_corpus_ids = corpus_data['hashes']
            else:
                raise ValueError(f"Expected 'features' and 'hashes' keys in corpus file: {corpus_vectors_file}")
        elif isinstance(corpus_data, (tuple, list)) and len(corpus_data) == 2:
            # タプル形式の場合：(corpus_ids, corpus_features)
            all_corpus_ids, all_corpus_features = corpus_data
        else:
            raise ValueError(f"Unsupported corpus vectors format in file: {corpus_vectors_file}")
        
        # 検索空間ファイルの存在確認
        if not os.path.exists(search_space_file):
            raise FileNotFoundError(f"Search space file not found: {search_space_file}")
        
        # 検索空間ファイルの形式に応じて読み込み方法を変更
        if search_space_file.endswith('.pt'):
            # PyTorchファイルの場合（メタデータファイル）
            metadata = torch.load(search_space_file, map_location='cpu', weights_only=False)
            if isinstance(metadata, dict) and 'image_ids' in metadata:
                self.search_space = metadata['image_ids']
            elif isinstance(metadata, dict) and 'idx_to_info' in metadata:
                # CIRCOの場合：メタデータファイルから画像IDを抽出
                idx_to_info = metadata['idx_to_info']
                # CIRCOの場合はhash_to_idxからハッシュを取得（コーパスのIDと一致）
                if 'circo' in search_space_file.lower() and 'hash_to_idx' in metadata:
                    self.search_space = list(metadata['hash_to_idx'].keys())
                    print(f"Extracted {len(self.search_space)} image hashes from CIRCO metadata")
                else:
                    self.search_space = [info.get('image_id', '') for info in idx_to_info.values() if info.get('image_id')]
                    print(f"Extracted {len(self.search_space)} image IDs from metadata")
            elif isinstance(metadata, list):
                self.search_space = metadata
            else:
                # メタデータから画像IDを抽出
                self.search_space = list(metadata.keys()) if isinstance(metadata, dict) else metadata
        else:
            # JSONファイルの場合
            with open(search_space_file, 'r') as f:
                search_data = json.load(f)
                if isinstance(search_data, dict):
                    # 辞書形式の場合、キーのリストを取得（CIRRの場合）
                    self.search_space = list(search_data.keys())
                    # 値も保持（相対パス情報）
                    self.search_space_paths = search_data
                else:
                    # リスト形式の場合
                    self.search_space = search_data
                    self.search_space_paths = None
        
        # 検索空間が空でないことを確認
        if not self.search_space:
            raise ValueError(f"Search space is empty. Check file: {search_space_file}")
        
        # 検索空間に含まれる画像の特徴量のみを抽出
        valid_indices = []
        self.corpus_ids = []
        self.search_space_filtered = []  # 実際に存在する画像のみ
        
        print(f"Filtering features for search space ({len(self.search_space)} images)...")
        
        # 画像ディレクトリを設定（データセット別）
        if 'fashion-iq' in corpus_vectors_file.lower():
            image_base_dir = 'fashion-iq/images'
        elif 'cirr' in corpus_vectors_file.lower():
            image_base_dir = 'cirr/img_raw'
        elif 'circo' in corpus_vectors_file.lower():
            image_base_dir = 'CIRCO/COCO2017_unlabeled/unlabeled2017'
        else:
            image_base_dir = ''  # フォールバック
        
        def check_image_file_exists(image_id: str, image_path: str = None) -> bool:
            """画像ファイルが実際に存在するかチェック"""
            if not image_base_dir:
                return True  # ディレクトリが不明な場合はスキップ
                
            # メタデータのimage_pathが利用可能な場合はそれを使用
            if image_path and os.path.exists(image_path):
                return True
                
            # FashionIQの場合はカテゴリ別ディレクトリ
            if 'fashion-iq' in corpus_vectors_file.lower():
                for category in ['dress', 'shirt', 'toptee']:
                    image_path = os.path.join(image_base_dir, category, f"{image_id}.jpg")
                    if os.path.exists(image_path):
                        return True
                return False
            elif 'cirr' in corpus_vectors_file.lower():
                # CIRRの場合：splitによって構造が異なる
                # train: 階層構造（0-99のサブディレクトリ）
                # val, dev, test1: フラット構造（直下にファイル）
                
                # trainの場合：階層構造
                for subdir in range(100):  # 0-99のサブディレクトリを検索
                    for ext in ['.jpg', '.jpeg', '.png']:
                        image_path = os.path.join(image_base_dir, 'train', str(subdir), f"{image_id}{ext}")
                        if os.path.exists(image_path):
                            return True
                
                # val, dev, test1の場合：フラット構造
                for split in ['val', 'dev', 'test1']:
                    for ext in ['.png', '.jpg', '.jpeg']:
                        image_path = os.path.join(image_base_dir, split, f"{image_id}{ext}")
                        if os.path.exists(image_path):
                            return True
                
                return False
            else:
                # CIRCOの場合（image_idを12桁ゼロパディング形式に変換）
                if image_path and os.path.exists(image_path):
                    return True
                
                # image_idを12桁ゼロパディング形式のファイル名に変換
                padded_filename = f"{int(image_id):012d}.jpg"
                image_path = os.path.join(image_base_dir, padded_filename)
                return os.path.exists(image_path)
        
        # 統一的なアプローチ：メタデータファイルを優先し、なければ直接照合
        metadata_file = corpus_vectors_file.replace('features_', 'metadata_')
        use_metadata = False
        
        # CIRCOの場合、search_space_fileがメタデータファイルと同じ場合がある
        if 'circo' in corpus_vectors_file.lower() and search_space_file == metadata_file:
            # CIRCOの場合、検索空間とメタデータが同じファイル
            use_metadata = True
            metadata = torch.load(metadata_file, map_location='cpu', weights_only=False)
            
            # CIRCOの場合のメタデータ処理
            matched_count = 0
            search_space_set = set(self.search_space)  # 高速検索用
            
            # 特徴量ファイルのhash_to_idxを使用（これが実際の特徴量配列のインデックス）
            if isinstance(corpus_data, dict) and 'hash_to_idx' in corpus_data:
                features_hash_to_idx = corpus_data['hash_to_idx']
                
                for hash_val in search_space_set:
                    if hash_val in features_hash_to_idx:
                        corpus_idx = features_hash_to_idx[hash_val]
                        
                        # メタデータから画像情報を取得（存在チェック用）
                        if 'hash_to_idx' in metadata and hash_val in metadata['hash_to_idx']:
                            metadata_idx = metadata['hash_to_idx'][hash_val]
                            info = metadata['idx_to_info'].get(metadata_idx, {})
                            image_id = info.get('image_id', '')
                            image_path = info.get('image_path', '')
                            
                            # 画像ファイルの存在チェック
                            file_exists = check_image_file_exists(image_id, image_path)
                            
                            if file_exists:
                                valid_indices.append(corpus_idx)
                                self.corpus_ids.append(all_corpus_ids[corpus_idx])
                                self.search_space_filtered.append(hash_val)  # ハッシュを使用
                                matched_count += 1
                        else:
                            # メタデータにない場合でも、特徴量が存在すれば使用
                            valid_indices.append(corpus_idx)
                            self.corpus_ids.append(all_corpus_ids[corpus_idx])
                            self.search_space_filtered.append(hash_val)
                            matched_count += 1

            print(f"Successfully matched {matched_count} images using CIRCO metadata")
        
        # CIRRの場合、メタデータファイルを使用して対応付け
        elif 'cirr' in corpus_vectors_file.lower() and os.path.exists(metadata_file):
            use_metadata = True
            metadata = torch.load(metadata_file, map_location='cpu', weights_only=False)
            
            # CIRRの場合のメタデータ処理
            matched_count = 0
            search_space_set = set(self.search_space)  # 高速検索用
            
            if 'idx_to_info' in metadata:
                idx_to_info = metadata['idx_to_info']
                for idx, info in idx_to_info.items():
                    image_id = info.get('image_id', '')
                    image_path = info.get('image_path', '')
                    
                    # 検索空間に含まれているかチェック
                    if image_id in search_space_set:
                        # 画像ファイルの存在チェック
                        file_exists = check_image_file_exists(image_id, image_path)
                        
                        if file_exists and idx < len(all_corpus_ids):
                            valid_indices.append(idx)
                            self.corpus_ids.append(all_corpus_ids[idx])
                            self.search_space_filtered.append(image_id)
                            matched_count += 1
            
            print(f"Successfully matched {matched_count} images using CIRR metadata")
        
        # FashionIQの場合、メタデータファイルを使用して対応付け
        elif 'fashion-iq' in corpus_vectors_file.lower() and os.path.exists(metadata_file):
            use_metadata = True
            metadata = torch.load(metadata_file, map_location='cpu', weights_only=False)
            
            # FashionIQの場合のメタデータ処理
            matched_count = 0
            search_space_set = set(self.search_space)  # 高速検索用
            
            if 'idx_to_info' in metadata:
                idx_to_info = metadata['idx_to_info']
                for idx, info in idx_to_info.items():
                    image_id = info.get('image_id', '')
                    image_path = info.get('image_path', '')
                    
                    # 検索空間に含まれているかチェック（商品IDベース）
                    if image_id in search_space_set:
                        # 画像ファイルの存在チェック
                        file_exists = check_image_file_exists(image_id, image_path)
                        
                        if file_exists and idx < len(all_corpus_ids):
                            valid_indices.append(idx)
                            self.corpus_ids.append(all_corpus_ids[idx])
                            self.search_space_filtered.append(image_id)  # 商品IDを使用
                            matched_count += 1
            
            print(f"Successfully matched {matched_count} images using FashionIQ metadata")
        
        if not use_metadata:
            # 直接画像ID照合（メタデータファイルがない場合）
            print("Using direct image ID matching")
            for search_img_id in self.search_space:
                # 画像ファイルの存在をまずチェック
                if '/' in search_img_id:
                    # 相対パス形式の場合、ファイル名部分を抽出
                    base_id = search_img_id.split('/')[-1]
                    if '.' in base_id:
                        base_id = base_id.rsplit('.', 1)[0]
                else:
                    base_id = search_img_id.rsplit('.', 1)[0] if '.' in search_img_id else search_img_id
                
                if not check_image_file_exists(base_id):
                    continue  # 画像ファイルが存在しない場合はスキップ
                
                # 拡張子の有無を考慮した候補リストを作成
                search_candidates = [search_img_id]
                
                # CIRRの場合、相対パスからファイル名を抽出した候補も追加
                if 'cirr' in corpus_vectors_file.lower() and '/' in search_img_id:
                    filename = search_img_id.split('/')[-1]
                    search_candidates.append(filename)
                    if '.' in filename:
                        search_candidates.append(filename.rsplit('.', 1)[0])
                
                if not search_img_id.endswith(('.png', '.jpg', '.jpeg')):
                    search_candidates.extend([f"{search_img_id}.png", f"{search_img_id}.jpg"])
                if search_img_id.endswith(('.png', '.jpg', '.jpeg')):
                    search_candidates.append(search_img_id.rsplit('.', 1)[0])
                
                # 特徴量ファイル内で照合
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
        
        # 検索空間に対応する特徴量のみを抽出
        self.corpus_features = all_corpus_features[valid_indices]
        
        print(f"Filtered corpus: {len(self.corpus_ids)} images (from {len(all_corpus_ids)} total)")
        
        self.model_manager = model_manager
    
    def search_images(self, query_features: torch.Tensor) -> List[Tuple[str, float]]:
        """画像検索を実行"""
        corpus_ids, corpus_features = self.corpus_ids, self.corpus_features
        
        # クエリ特徴量を正規化
        query_features = normalize(query_features, dim=-1)
        corpus_features = normalize(corpus_features, dim=-1)
        
        # コサイン類似度を計算
        similarities = (query_features @ corpus_features.T).squeeze(0).cpu().numpy()
        
        # 検索結果を作成（フィルタリング済み検索空間を使用）
        image_similarities = [
            (self.search_space_filtered[index], similarities[index]) 
            for index in range(len(corpus_ids))
        ]
        
        # 類似度順にソート
        images = sorted(image_similarities, key=lambda x: x[1], reverse=True)
        return images
    
    def get_image_features(self, image_id: str) -> torch.Tensor:
        """特定の画像の特徴量を取得"""
        corpus_ids, corpus_features = self.corpus_ids, self.corpus_features
        
        if image_id not in self.search_space_filtered:
            raise ValueError(f"Image ID '{image_id}' not found in filtered search space. "
                           f"Available images: {len(self.search_space_filtered)}")
        
        index = self.search_space_filtered.index(image_id)
        return corpus_features[index].clone().detach().to(device)
    
    def update_query_with_feedback(self, current_query: torch.Tensor, selected_img: str, 
                                 unselected_imgs: List[str], new_text_query: str,
                                 alpha: float = 0.08, beta: float = 0.29, gamma: float = 0.44) -> torch.Tensor:
        """フィードバックを使ってクエリを更新"""
        with torch.no_grad():
            text_features = self.model_manager.dialog_encoder(new_text_query)
        
        # 特徴量を正規化
        text_features = normalize(text_features, dim=-1)
        current_query = normalize(current_query, dim=-1)
        
        # 選択された画像の特徴量
        selected_features = normalize(self.get_image_features(selected_img), dim=-1)
        
        # 未選択画像の特徴量
        unselected_features = torch.stack([
            self.get_image_features(img_id) for img_id in unselected_imgs
        ])
        unselected_features = normalize(unselected_features, dim=-1)
        
        # クエリを更新
        updated_query = (
            current_query + 
            gamma * text_features + 
            alpha * selected_features - 
            beta * torch.mean(unselected_features, dim=0)
        )
        
        return normalize(updated_query, dim=-1)

class MultiTurnCIRSystem:
    """マルチターンCIRシステムのメインクラス"""
    
    def __init__(self, dataset_name: str, config: Dict[str, Any]):
        """初期化"""
        self.dataset_name = dataset_name
        self.config = config
        self.max_turns = config.get('max_turns', 5)
        self.results = []
        self.completeness_info = {}
        
        # max_turnsの設定値をログ出力
        print(f"Initializing {dataset_name} with max_turns = {self.max_turns}")
        
        # CIRCOの場合、ハッシュ→画像ファイル名のマッピングを作成
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
        
        # 画像ID→ハッシュ値のマッピングを初期化
        self.image_id_to_hash = {}
        self._initialize_image_hash_mapping()
        
        # モデルマネージャーとリトリーバルエンジンを初期化
        self.model_manager = ModelManager(use_blip=config.get('use_blip', True))
        self.retrieval_engine = RetrievalEngine(
            config['corpus_vectors_file'], 
            config['search_space_file'], 
            self.model_manager
        )
        
        # データとキャプション特徴量を読み込み
        self.data = self.load_dataset()
        self.reference_captions = self.load_reference_captions()
        self.caption_features = self.load_caption_features()
        
        # 既存結果を読み込み（再開モード対応）
        self.load_existing_results()
    
    def _initialize_image_hash_mapping(self):
        """画像ID→ハッシュ値のマッピングを初期化"""
        print("Initializing image ID to hash mapping...")
        
        if self.dataset_name == 'circo':
            # CIRCOの場合、既存のhash_to_filenameから逆マッピングを作成
            for hash_val, filename in self.hash_to_filename.items():
                # ファイル名から画像IDを抽出（例：000000000001.jpg → 1）
                image_id = filename.replace('.jpg', '').lstrip('0') or '0'
                self.image_id_to_hash[image_id] = hash_val
                # 12桁形式も追加
                padded_id = f"{int(image_id):012d}"
                self.image_id_to_hash[padded_id] = hash_val
        
        elif self.dataset_name.startswith('cirr'):
            # CIRRの場合、実際の画像ファイルからハッシュを計算
            self._compute_cirr_image_hashes()
        
        elif self.dataset_name.startswith('fashioniq'):
            # FashionIQの場合、実際の画像ファイルからハッシュを計算
            self._compute_fashioniq_image_hashes()
        
        print(f"Created image ID to hash mapping for {len(self.image_id_to_hash)} images")
    
    def _compute_image_hash(self, image_path: str) -> str:
        """画像ファイルのMD5ハッシュを計算"""
        try:
            with open(image_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            print(f"Warning: Failed to compute hash for {image_path}: {e}")
            return None
    
    def _compute_cirr_image_hashes(self):
        """CIRR画像のハッシュ値を計算"""
        image_dir = self.config.get('image_dir', 'cirr/img_raw')
        
        # スプリットファイルから画像IDを取得
        splits = ['train', 'val', 'test1']
        for split in splits:
            split_file = f'cirr/image_splits/split.rc2.{split}.json'
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    split_data = json.load(f)
                
                for image_id, relative_path in split_data.items():
                    # relative_pathは "./train/34/train-11041-2-img0.png" のような形式
                    full_path = os.path.join(image_dir, relative_path.lstrip('./'))
                    
                    if os.path.exists(full_path):
                        img_hash = self._compute_image_hash(full_path)
                        if img_hash:
                            self.image_id_to_hash[image_id] = img_hash
    
    def _compute_fashioniq_image_hashes(self):
        """FashionIQ画像のハッシュ値を計算"""
        image_dir = self.config.get('image_dir', 'fashion-iq/images')
        category = self.dataset_name.split('_')[1] if '_' in self.dataset_name else 'dress'
        
        # スプリットファイルから画像IDを取得
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
                            # 拡張子付きバージョンも追加
                            self.image_id_to_hash[f"{image_id}.jpg"] = img_hash
    
    def get_image_hash(self, image_id: str) -> str:
        """画像IDからハッシュ値を取得"""
        # 直接マッピングから取得
        if image_id in self.image_id_to_hash:
            return self.image_id_to_hash[image_id]
        
        # FashionIQの場合、拡張子の有無を考慮
        if self.dataset_name.startswith('fashioniq'):
            if image_id.endswith('.jpg'):
                base_id = image_id[:-4]
                if base_id in self.image_id_to_hash:
                    return self.image_id_to_hash[base_id]
            else:
                jpg_id = f"{image_id}.jpg"
                if jpg_id in self.image_id_to_hash:
                    return self.image_id_to_hash[jpg_id]
        
        # CIRCOの場合、12桁形式も試す
        if self.dataset_name == 'circo':
            try:
                padded_id = f"{int(image_id):012d}"
                if padded_id in self.image_id_to_hash:
                    return self.image_id_to_hash[padded_id]
            except ValueError:
                pass
        
        return None
    
    def load_existing_results(self) -> None:
        """既存の結果ファイルを読み込んで再開準備"""
        output_file = f"multiturn_cir_results_{self.dataset_name}.json"
        
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    existing_data = json.load(f)
                
                if 'results' in existing_data:
                    self.results = existing_data['results']
                    
                    # Fashion-IQでcaption_mode=separateの場合は3つ組みで照合
                    if self.dataset_name.startswith('fashioniq') and self.config.get('caption_mode') == 'separate':
                        processed_query_triplets = set()
                        
                        # 処理済みクエリを(reference_id, target_id, relative_caption)の組み合わせで管理
                        for result in self.results:
                            ref_id = result.get('reference_image_id', '')
                            target_id = result.get('target_image_id', '')
                            original_query = result.get('original_query', '')  # relative_caption
                            
                            if ref_id and target_id and original_query:
                                processed_query_triplets.add((ref_id, target_id, original_query))
                        
                        print(f"Found existing results file with {len(self.results)} processed queries")
                        print(f"Processed query triplets (ref, target, caption): {len(processed_query_triplets)}")
                        
                        # 未処理のクエリのみを残す
                        original_data_count = len(self.data)
                        remaining_data = []
                        
                        for query_item in self.data:
                            # 現在のクエリの(reference_id, target_id, relative_caption)の3つ組みを取得
                            ref_id = query_item['reference_image_id']
                            target_id = query_item['target_image_id']
                            relative_caption = query_item['relative_caption']
                            query_triplet = (ref_id, target_id, relative_caption)
                            
                            # この3つ組みが処理済みでない場合のみ残す
                            if query_triplet not in processed_query_triplets:
                                remaining_data.append(query_item)
                        
                        self.data = remaining_data
                        
                        print(f"Resume mode (separate caption mode): {original_data_count - len(remaining_data)} queries already processed")
                        print(f"Remaining queries to process: {len(remaining_data)}")
                        
                    elif self.dataset_name == 'circo':
                        # CIRCOの特別処理：reference_id + 複数ground_truthsで照合
                        processed_query_signatures = set()
                        
                        # 保存された結果から処理済みクエリを特定
                        for result in self.results:
                            # 保存時にはハッシュ値→12桁IDに変換されているため、
                            # current dataのoriginal_*_idと照合
                            ref_id = result.get('reference_image_id', '')  # 12桁ID形式
                            gt_ids = result.get('ground_truth_ids', [])    # 12桁ID形式のリスト
                            
                            if ref_id and gt_ids:
                                # ground_truth_idsをソートしてタプル化（順序に依存しない一意な識別子を作成）
                                gt_ids_sorted = tuple(sorted(gt_ids))
                                query_signature = (ref_id, gt_ids_sorted)
                                processed_query_signatures.add(query_signature)
                        
                        print(f"Found existing results file with {len(self.results)} processed queries")
                        print(f"Processed query signatures (CIRCO - ref+GTs): {len(processed_query_signatures)}")
                        
                        # 未処理のクエリのみを残す
                        original_data_count = len(self.data)
                        remaining_data = []
                        
                        for query_item in self.data:
                            # current dataのoriginal_*_idを使用して照合
                            original_ref_id = query_item.get('original_reference_id', '')
                            original_gt_ids = query_item.get('original_gt_ids', [])
                            
                            # 拡張子を除去して12桁数値IDにする
                            if original_ref_id.endswith('.jpg'):
                                original_ref_id = original_ref_id[:-4]
                            
                            # ground truthsの拡張子も除去
                            normalized_gt_ids = []
                            for gt_id in original_gt_ids:
                                if gt_id.endswith('.jpg'):
                                    normalized_gt_ids.append(gt_id[:-4])
                                else:
                                    normalized_gt_ids.append(gt_id)
                            
                            # ground_truth_idsをソートしてタプル化
                            gt_ids_sorted = tuple(sorted(normalized_gt_ids))
                            query_signature = (original_ref_id, gt_ids_sorted)
                            
                            # この署名が処理済みでない場合のみ残す
                            if query_signature not in processed_query_signatures:
                                remaining_data.append(query_item)
                        
                        self.data = remaining_data
                        
                        print(f"Resume mode (CIRCO with GTs): {original_data_count - len(remaining_data)} queries already processed")
                        print(f"Remaining queries to process: {len(remaining_data)}")
                        
                        # デバッグ用：処理済みクエリの例を表示
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
                        # 従来の方式：(reference_id, target_id)ペアで照合
                        processed_query_pairs = set()
                        
                        # 処理済みクエリを(reference_id, target_id)の組み合わせで管理
                        for result in self.results:
                            # 結果からreference_image_idとtarget_image_idの組み合わせを取得
                            ref_id = result.get('reference_image_id', '')
                            target_id = result.get('target_image_id', '')
                            
                            if ref_id and target_id:
                                processed_query_pairs.add((ref_id, target_id))
                        
                        print(f"Found existing results file with {len(self.results)} processed queries")
                        print(f"Processed query pairs: {len(processed_query_pairs)}")
                        
                        # 未処理のクエリのみを残す
                        original_data_count = len(self.data)
                        remaining_data = []
                        
                        for query_item in self.data:
                            # 現在のクエリの(reference_id, target_id)ペアを取得
                            ref_id = query_item['reference_image_id']
                            target_id = query_item['target_image_id']
                            query_pair = (ref_id, target_id)
                            
                            # このペアが処理済みでない場合のみ残す
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
        """データセットをロード"""
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
        """事前計算されたキャプションを読み込み"""
        captions = {}
        
        # 設定にキャプションファイルが指定されている場合
        if 'caption_file' in self.config and os.path.exists(self.config['caption_file']):
            with open(self.config['caption_file'], 'r') as f:
                caption_data = json.load(f)
                
            # キャプションデータの形式に応じて処理
            if isinstance(caption_data, dict):
                captions = caption_data
            elif isinstance(caption_data, list):
                # リスト形式の場合、適切なキーでマッピング
                for item in caption_data:
                    if 'image_id' in item and 'caption' in item:
                        captions[item['image_id']] = item['caption']
        
        # gpt4ominiキャプションファイル（PyTorch形式）の読み込み
        if 'gpt4omini_caption_features_file' in self.config and os.path.exists(self.config['gpt4omini_caption_features_file']):
            try:
                features_data = torch.load(self.config['gpt4omini_caption_features_file'], map_location='cpu', weights_only=False)
                if isinstance(features_data, dict):
                    # PyTorchファイルから直接辞書を取得
                    captions.update(features_data)
                elif hasattr(features_data, 'items'):
                    # その他の辞書形式
                    captions.update(dict(features_data.items()))
            except Exception as e:
                print(f"Warning: Failed to load GPT-4o-mini captions from {self.config['gpt4omini_caption_features_file']}: {e}")
        
        # データセット固有の処理
        if self.dataset_name == 'circo':
            # CIRCOの場合、COCOキャプションも併用可能
            coco_caption_file = 'CIRCO/annotations/captions_val2017.json'
            if os.path.exists(coco_caption_file):
                with open(coco_caption_file, 'r') as f:
                    coco_captions = json.load(f)
                
                for ann in coco_captions.get('annotations', []):
                    img_id = f"{ann['image_id']:06d}.jpg"
                    if img_id not in captions:
                        captions[img_id] = ann['caption']
            
        elif self.dataset_name.startswith('fashioniq'):
            # Fashion-IQの場合、事前生成されたキャプションファイルが必要
            if not captions:
                raise ValueError(f"No captions loaded for Fashion-IQ dataset. "
                               f"Caption file: {self.config.get('caption_file', 'Not specified')} "
                               f"GPT-4o-mini caption file: {self.config.get('gpt4omini_caption_features_file', 'Not specified')} "
                               f"Please ensure at least one caption file exists and contains valid data.")
        
        return captions
    
    def get_reference_caption(self, reference_image_id: str) -> str:
        """reference画像のキャプションを取得"""
        if reference_image_id in self.reference_captions:
            return self.reference_captions[reference_image_id]
        
        # フルパス形式のキーを試す（全データセット統一対応）
        potential_keys = []
        
        if self.dataset_name.startswith('fashioniq'):
            # Fashion-IQ: fashion-iq/images/{category}/{image_id}
            category = self.dataset_name.split('_')[1]  # dress, shirt, toptee
            
            # 拡張子の有無を考慮した候補
            base_key = f"fashion-iq/images/{category}/{reference_image_id}"
            potential_keys.append(base_key)
            
            # 拡張子なしの場合、.jpg付きも試す
            if not reference_image_id.endswith('.jpg'):
                potential_keys.append(f"fashion-iq/images/{category}/{reference_image_id}.jpg")
            # 拡張子ありの場合、なしも試す
            elif reference_image_id.endswith('.jpg'):
                base_id = reference_image_id[:-4]
                potential_keys.append(f"fashion-iq/images/{category}/{base_id}")
            
        elif self.dataset_name.startswith('cirr'):
            # CIRR: 実際のキャプションファイル構造に合わせて検索
            # キャプションファイルではフルパス形式で保存されている
            
            # 基本的なキー候補
            potential_keys.extend([
                reference_image_id,
                f"{reference_image_id}.png",
            ])
            
            # フルパス形式のキー候補（実際の保存形式）
            # Train: 階層構造 (0-99のサブディレクトリ)
            for subdir in range(100):
                potential_keys.extend([
                    f"cirr/img_raw/train/{subdir}/{reference_image_id}",
                    f"cirr/img_raw/train/{subdir}/{reference_image_id}.png",
                ])
            
            # Dev/Val: フラット構造
            potential_keys.extend([
                f"cirr/img_raw/dev/{reference_image_id}",
                f"cirr/img_raw/dev/{reference_image_id}.png",
                f"cirr/img_raw/val/{reference_image_id}",
                f"cirr/img_raw/val/{reference_image_id}.png",
            ])
        
        elif self.dataset_name == 'circo':
            # CIRCO: ハッシュ値から実際のファイル名に変換してキャプションを検索
            if reference_image_id in self.hash_to_filename:
                filename = self.hash_to_filename[reference_image_id]
                potential_keys.extend([
                    f"CIRCO/COCO2017_unlabeled/unlabeled2017/{filename}",
                    filename
                ])
            else:
                # ハッシュ値でない場合（12桁形式）はそのまま使用
                potential_keys.extend([
                    f"CIRCO/COCO2017_unlabeled/unlabeled2017/{reference_image_id}",
                    f"CIRCO/COCO2017_unlabeled/unlabeled2017/{reference_image_id}.jpg"
                ])
        
        # 候補キーで検索
        for key in potential_keys:
            if key in self.reference_captions:
                return self.reference_captions[key]
        
        # 見つからない場合はエラー
        raise KeyError(f"Caption not found for image: {reference_image_id}. "
                      f"Available captions: {len(self.reference_captions)} images. "
                      f"Check caption file: {self.config.get('caption_file', 'Not specified')}. "
                      f"Tried keys: {potential_keys[:3]}... "
                      f"Sample actual keys: {list(self.reference_captions.keys())[:3]}")
    
    def check_success(self, ground_truth_ids: List[str], search_results: List[Tuple[str, float]], k: int = 10) -> bool:
        """成功判定（ground truthがtop-kに含まれるかチェック）"""
        if not ground_truth_ids:
            raise ValueError("Ground truth IDs list is empty")
        if not search_results:
            raise ValueError("Search results list is empty")
            
        top_k_ids = [result[0] for result in search_results[:k]]
        return any(gt_id in top_k_ids for gt_id in ground_truth_ids)
    
    def find_most_similar_to_gt(self, ground_truth_ids: List[str], search_results: List[Tuple[str, float]], 
                               selected_images: set = None) -> str:
        """Ground truthに最も類似した画像を検索結果から選択（GTがtop10に入っていない場合のみ）"""
        if not ground_truth_ids:
            raise ValueError("Ground truth IDs list is empty")
        if not search_results:
            raise ValueError("Search results list is empty")
            
        top_k_ids = [r[0] for r in search_results[:10]]
        
        # selected_imagesがNoneの場合は空のセットで初期化
        if selected_images is None:
            selected_images = set()
        
        # 選択済み画像のハッシュ値セットを作成
        selected_hashes = set()
        for img_id in selected_images:
            img_hash = self.get_image_hash(img_id)
            if img_hash:
                selected_hashes.add(img_hash)
        
        # 未選択の候補画像を取得（ハッシュ値ベースで重複チェック）
        available_images = []
        for img_id in top_k_ids:
            img_hash = self.get_image_hash(img_id)
            if img_hash and img_hash not in selected_hashes:
                available_images.append(img_id)
            elif not img_hash:
                # ハッシュ値が取得できない場合は従来通りIDで判定
                if img_id not in selected_images:
                    available_images.append(img_id)
        
        if not available_images:
            raise ValueError(f"No available images for selection. "
                           f"Top-k IDs: {top_k_ids}, Selected images: {len(selected_images)}, "
                           f"Selected hashes: {len(selected_hashes)}")
        
        if self.dataset_name == 'circo':
            # CIRCO用: 最も検索上位にいたGTをキャプション類似度の基準として使用
            # まず全てのGTの検索順位を調べて、最も上位のGTを特定
            best_gt_position = len(search_results)
            best_gt_id = None
            
            for gt_id in ground_truth_ids:
                try:
                    # 全検索結果での順位を取得
                    all_ids = [r[0] for r in search_results]
                    position = all_ids.index(gt_id)
                    if position < best_gt_position:
                        best_gt_position = position
                        best_gt_id = gt_id
                except ValueError:
                    # GTが検索結果に含まれない場合はスキップ
                    continue
            
            if best_gt_id is None:
                raise ValueError(f"No ground truth images found in search results for CIRCO. "
                               f"GT IDs: {ground_truth_ids}, Search results: {[r[0] for r in search_results[:20]]}")
            
            # 最も上位のGTのキャプション特徴量を取得
            if best_gt_id in self.caption_features:
                gt_caption_features = self.caption_features[best_gt_id].to(device)
                gt_caption_features = normalize(gt_caption_features.unsqueeze(0), dim=-1)
            else:
                # キャプション特徴量が見つからない場合は明確なエラー
                raise KeyError(f"Caption features not found for best GT image '{best_gt_id}' in CIRCO. "
                             f"Available caption features: {len(self.caption_features)} images. "
                             f"Check gpt4omini_caption_features_file: {self.config.get('gpt4omini_caption_features_file', 'Not specified')}")
            
            # 各候補画像のキャプション特徴量と類似度を計算
            best_similarity = -1.0
            best_image = None
            similarities_computed = 0
            
            for img_id in available_images:
                if img_id in self.caption_features:
                    # 事前計算された特徴量を使用
                    img_caption_features = self.caption_features[img_id].to(device)
                    img_caption_features = normalize(img_caption_features.unsqueeze(0), dim=-1)
                    
                    # コサイン類似度を計算
                    similarity = torch.cosine_similarity(gt_caption_features, img_caption_features, dim=-1).item()
                    similarities_computed += 1
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_image = img_id
                else:
                    # キャプション特徴量が見つからない場合は警告のみ（計算続行）
                    print(f"Warning: Caption features not found for candidate image '{img_id}' in CIRCO")
                    continue
            
            if best_image is None:
                raise ValueError(f"No suitable image found for selection in CIRCO. "
                               f"Available images: {len(available_images)}, "
                               f"Similarities computed: {similarities_computed}, "
                               f"Best GT ID: {best_gt_id}")
            
            return best_image
        
        elif self.dataset_name.startswith('cirr'):
            # CIRR用: キャプションの特徴量を使ってGTに最も近い画像を選択
            # Ground truthのキャプション特徴量を取得
            gt_id = ground_truth_ids[0]  # CIRRは単一GT
            if gt_id in self.caption_features:
                gt_caption_features = self.caption_features[gt_id]
                # numpy配列の場合はテンソルに変換
                if isinstance(gt_caption_features, np.ndarray):
                    gt_caption_features = torch.from_numpy(gt_caption_features).float()
                gt_caption_features = gt_caption_features.to(device)
                gt_caption_features = normalize(gt_caption_features.unsqueeze(0), dim=-1)
            else:
                raise KeyError(f"Caption features not found for GT image '{gt_id}' in CIRR. "
                             f"Available caption features: {len(self.caption_features)} images. "
                             f"Check gpt4omini_caption_features_file: {self.config.get('gpt4omini_caption_features_file', 'Not specified')}")
            
            # 各候補画像のキャプション特徴量と類似度を計算
            best_similarity = -1.0
            best_image = None
            similarities_computed = 0
            
            for img_id in available_images:
                if img_id in self.caption_features:
                    # 事前計算された特徴量を使用
                    img_caption_features = self.caption_features[img_id]
                    # numpy配列の場合はテンソルに変換
                    if isinstance(img_caption_features, np.ndarray):
                        img_caption_features = torch.from_numpy(img_caption_features).float()
                    img_caption_features = img_caption_features.to(device)
                    img_caption_features = normalize(img_caption_features.unsqueeze(0), dim=-1)
                    
                    # コサイン類似度を計算
                    similarity = torch.cosine_similarity(gt_caption_features, img_caption_features, dim=-1).item()
                    similarities_computed += 1
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_image = img_id
                else:
                    # キャプション特徴量が見つからない場合は警告のみ（計算続行）
                    print(f"Warning: Caption features not found for candidate image '{img_id}' in CIRR")
                    continue
            
            if best_image is None:
                raise ValueError(f"No suitable image found for selection in CIRR. "
                               f"Available images: {len(available_images)}, "
                               f"Similarities computed: {similarities_computed}, "
                               f"GT ID: {gt_id}")
            
            return best_image
        
        elif self.dataset_name.startswith('fashioniq'):
            # FashionIQ用: キャプションの特徴量を使ってGTに最も近い画像を選択
            # Ground truthのキャプション特徴量を取得
            gt_id = ground_truth_ids[0]  # FashionIQは単一GT
            
            # FashionIQでは拡張子の有無を考慮してGTキャプション特徴量を検索
            gt_caption_features = None
            
            # 候補1: 元のID
            if gt_id in self.caption_features:
                gt_caption_features = self.caption_features[gt_id]
            # 候補2: .jpg拡張子付き
            elif f"{gt_id}.jpg" in self.caption_features:
                gt_caption_features = self.caption_features[f"{gt_id}.jpg"]
            # 候補3: 拡張子なし（元のIDから拡張子を除去）
            elif gt_id.endswith('.jpg') and gt_id[:-4] in self.caption_features:
                gt_caption_features = self.caption_features[gt_id[:-4]]
            
            if gt_caption_features is not None:
                # numpy配列の場合はテンソルに変換
                if isinstance(gt_caption_features, np.ndarray):
                    gt_caption_features = torch.from_numpy(gt_caption_features).float()
                gt_caption_features = gt_caption_features.to(device)
                gt_caption_features = normalize(gt_caption_features.unsqueeze(0), dim=-1)
            else:
                raise KeyError(f"Caption features not found for GT image '{gt_id}' in FashionIQ. "
                             f"Available caption features: {len(self.caption_features)} images. "
                             f"Check gpt4omini_caption_features_file: {self.config.get('gpt4omini_caption_features_file', 'Not specified')}")
            
            # 各候補画像のキャプション特徴量と類似度を計算
            best_similarity = -1.0
            best_image = None
            similarities_computed = 0
            
            for img_id in available_images:
                # FashionIQでは拡張子の有無を考慮して検索
                caption_features = None
                
                # 候補1: 元のID
                if img_id in self.caption_features:
                    caption_features = self.caption_features[img_id]
                # 候補2: .jpg拡張子付き
                elif f"{img_id}.jpg" in self.caption_features:
                    caption_features = self.caption_features[f"{img_id}.jpg"]
                # 候補3: 拡張子なし（元のIDから拡張子を除去）
                elif img_id.endswith('.jpg') and img_id[:-4] in self.caption_features:
                    caption_features = self.caption_features[img_id[:-4]]
                
                if caption_features is not None:
                    # 事前計算された特徴量を使用
                    img_caption_features = caption_features
                    # numpy配列の場合はテンソルに変換
                    if isinstance(img_caption_features, np.ndarray):
                        img_caption_features = torch.from_numpy(img_caption_features).float()
                    img_caption_features = img_caption_features.to(device)
                    img_caption_features = normalize(img_caption_features.unsqueeze(0), dim=-1)
                    
                    # コサイン類似度を計算
                    similarity = torch.cosine_similarity(gt_caption_features, img_caption_features, dim=-1).item()
                    similarities_computed += 1
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_image = img_id
                else:
                    # キャプション特徴量が見つからない場合は警告のみ（計算続行）
                    print(f"Warning: Caption features not found for candidate image '{img_id}' in FashionIQ")
                    continue
            
            if best_image is None:
                raise ValueError(f"No suitable image found for selection in FashionIQ. "
                               f"Available images: {len(available_images)}, "
                               f"Similarities computed: {similarities_computed}, "
                               f"GT ID: {gt_id}")
            
            return best_image
        
        else:
            # 他のデータセットでは、利用可能な画像から最初の画像を選択
            return available_images[0]
    
    def get_gt_rankings(self, ground_truth_ids: List[str], search_results: List[Tuple[str, float]]) -> Dict[str, int]:
        """Ground truthの順位を取得（1-indexed）"""
        gt_rankings = {}
        result_ids = [result[0] for result in search_results]
        
        for gt_id in ground_truth_ids:
            try:
                # 1-indexedで順位を記録
                rank = result_ids.index(gt_id) + 1
                gt_rankings[gt_id] = rank
            except ValueError:
                # GTが検索結果に含まれない場合は-1
                gt_rankings[gt_id] = -1
        
        return gt_rankings
    
    def get_best_gt_rank(self, gt_rankings: Dict[str, int]) -> int:
        """最も良い（小さい）GT順位を取得"""
        valid_ranks = [rank for rank in gt_rankings.values() if rank > 0]
        return min(valid_ranks) if valid_ranks else -1
    
    async def process_single_query(self, query_item: Dict) -> Dict:
        """単一クエリを処理"""
        # クエリIDを統一的に設定（元の方式）
        query_id = query_item.get('id')
        if query_id is None:
            query_id = query_item.get('query_id')
        if query_id is None:
            # フォールバック：reference_image_idを使用
            query_id = query_item.get('reference_image_id', 'unknown')
        
        # 数値IDを文字列に変換
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
        
        # 選択済み画像の記録（reference_imageを事前に追加して重複を防ぐ）
        selected_images = set()
        selected_images.add(query_item['reference_image_id'])  # reference_imageとの重複を防ぐ
        previous_relative_captions = []
        
        # リファレンスキャプションを取得
        reference_caption = self.get_reference_caption(query_item['reference_image_id'])
        
        # ターン0: 初期検索
        initial_combined_caption = await self.model_manager.combine_captions_with_gpt4o(
            reference_caption, query_item['relative_caption']
        )
        
        with torch.no_grad():
            current_query_features = self.model_manager.dialog_encoder(initial_combined_caption)
        
        search_results = self.retrieval_engine.search_images(current_query_features)
        
        # GT順位を取得
        gt_rankings = self.get_gt_rankings(query_item['ground_truth_ids'], search_results)
        best_gt_rank = self.get_best_gt_rank(gt_rankings)
        
        turn_result = {
            'turn': 0,
            'query_text': initial_combined_caption,
            'search_results': search_results[:10],
            'selected_image': None,
            'selected_image_caption': None,  # ターン0では選択画像なし
            'relative_caption': None,
            'gt_rankings': gt_rankings,
            'best_gt_rank': best_gt_rank
        }
        results['turns'].append(turn_result)
        
        # 成功判定
        if self.check_success(query_item['ground_truth_ids'], search_results):
            results['success'] = True
            results['success_turn'] = 0
            return results
        
        # マルチターン処理
        for turn in range(1, self.max_turns + 1):
            # GTがtop10に入っているかチェック
            if self.check_success(query_item['ground_truth_ids'], search_results):
                # GTがtop10に入っている場合は早期終了
                results['success'] = True
                results['success_turn'] = turn - 1  # 前のターンで成功
                break
            
            # 最も類似した画像を選択
            if self.dataset_name == 'circo':
                # CIRCOの場合：最も検索上位にいたGTを選択
                selected_image = self.find_most_similar_to_gt(query_item['ground_truth_ids'], search_results, selected_images)
            else:
                # 他のデータセット（CIRR、FashionIQ）では、キャプション類似度を考慮して選択
                selected_image = self.find_most_similar_to_gt(
                    query_item['ground_truth_ids'], search_results, selected_images
                )
            
            selected_images.add(selected_image)
            
            # 選択された画像のキャプションを取得
            try:
                selected_image_caption = self.get_reference_caption(selected_image)
            except KeyError:
                selected_image_caption = f"Caption not found for {selected_image}"
            
            # GPT-4o-miniでrelative captionを生成
            # ターン1以降では選択された画像とground truth画像を比較
            if self.dataset_name.startswith('fashioniq'):
                # Fashion-IQの場合、カテゴリサブディレクトリを含め、拡張子も追加
                category = self.dataset_name.split('_')[1]  # dress, shirt, toptee
                
                # 拡張子を自動追加
                selected_img_with_ext = selected_image if selected_image.endswith('.jpg') else f"{selected_image}.jpg"
                target_img_with_ext = query_item['target_image_id'] if query_item['target_image_id'].endswith('.jpg') else f"{query_item['target_image_id']}.jpg"
                
                selected_image_path = os.path.join(self.config['image_dir'], category, selected_img_with_ext)
                target_image_path = os.path.join(self.config['image_dir'], category, target_img_with_ext)
            elif self.dataset_name.startswith('cirr'):
                # CIRRの場合、RetrievalEngineと同じロジックで正しいパスを検索
                def find_cirr_image_path(image_id: str) -> str:
                    """CIRRの画像パスを階層構造から検索"""
                    # trainの場合：階層構造（0-99のサブディレクトリ）
                    for subdir in range(100):  # 0-99のサブディレクトリを検索
                        for ext in ['.png', '.jpg', '.jpeg']:
                            image_path = os.path.join(self.config['image_dir'], 'train', str(subdir), f"{image_id}{ext}")
                            if os.path.exists(image_path):
                                return image_path
                    
                    # val, dev, test1の場合：フラット構造
                    for split in ['val', 'dev', 'test1']:
                        for ext in ['.png', '.jpg', '.jpeg']:
                            image_path = os.path.join(self.config['image_dir'], split, f"{image_id}{ext}")
                            if os.path.exists(image_path):
                                return image_path
                    
                    raise FileNotFoundError(f"CIRR image not found: {image_id}")
                
                selected_image_path = find_cirr_image_path(selected_image)
                target_image_path = find_cirr_image_path(query_item['target_image_id'])
            else:
                # 他のデータセット（CIRCO）ではハッシュ値から実際のファイル名に変換
                def get_circo_image_path(image_id: str) -> str:
                    if image_id in self.hash_to_filename:
                        filename = self.hash_to_filename[image_id]
                        return os.path.join(self.config['image_dir'], filename)
                    else:
                        # ハッシュ値でない場合はそのまま使用
                        return os.path.join(self.config['image_dir'], image_id)
                
                selected_image_path = get_circo_image_path(selected_image)
                target_image_path = get_circo_image_path(query_item['target_image_id'])
            
            # ファイルの存在確認
            if not os.path.exists(selected_image_path):
                raise FileNotFoundError(f"Selected image not found: {selected_image_path}")
            if not os.path.exists(target_image_path):
                raise FileNotFoundError(f"Target image not found: {target_image_path}")
            
            # GPT-4o-miniでrelative captionを生成（選択された画像 → ground truth画像への変換）
            new_relative_caption = await self.model_manager.generate_relative_caption_with_gpt4o(
                selected_image_path, target_image_path, previous_relative_captions,
                similarity_threshold=0.8,  # CLIP類似度の閾値
                max_retries=3  # 最大再試行回数
            )
            
            previous_relative_captions.append(new_relative_caption)
            
            # 新しいクエリを生成
            new_combined_caption = await self.model_manager.combine_captions_with_gpt4o(
                selected_image_caption, new_relative_caption
            )
            
            # クエリを更新
            unselected_images = [r[0] for r in search_results[:10] if r[0] != selected_image]
            current_query_features = self.retrieval_engine.update_query_with_feedback(
                current_query_features, selected_image, unselected_images, new_combined_caption
            )
            
            # 再検索
            search_results = self.retrieval_engine.search_images(current_query_features)
            
            # GT順位を取得
            gt_rankings = self.get_gt_rankings(query_item['ground_truth_ids'], search_results)
            best_gt_rank = self.get_best_gt_rank(gt_rankings)
            
            turn_result = {
                'turn': turn,
                'query_text': new_combined_caption,
                'search_results': search_results[:10],
                'selected_image': selected_image,
                'selected_image_caption': selected_image_caption,  # 選択画像のキャプション
                'relative_caption': new_relative_caption,
                'gt_rankings': gt_rankings,
                'best_gt_rank': best_gt_rank
            }
            results['turns'].append(turn_result)
            
            # 再検索後の成功判定
            if self.check_success(query_item['ground_truth_ids'], search_results):
                results['success'] = True
                results['success_turn'] = turn
                break
            
            # 少し待機
            time.sleep(0.5)
        
        return results
    
    def run_evaluation(self) -> None:
        """評価を実行"""
        # 再開モードの情報表示
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
        
        # データ完全性チェックを実行
        self.completeness_info = self.check_data_completeness()
        
        # 画像ディレクトリの設定
        image_dir = self.config.get('image_dir', '')
        
        def check_image_exists(image_id: str) -> bool:
            """画像ファイルが存在するかチェック"""
            if self.dataset_name.startswith('fashioniq'):
                # Fashion-IQ: カテゴリサブディレクトリを含め、拡張子も自動追加
                category = self.dataset_name.split('_')[1]  # dress, shirt, toptee
                img_with_ext = image_id if image_id.endswith('.jpg') else f"{image_id}.jpg"
                image_path = os.path.join(image_dir, category, img_with_ext)
                return os.path.exists(image_path)
            elif self.dataset_name.startswith('cirr'):
                # CIRRの場合：splitによって構造が異なる
                # train: 階層構造（0-99のサブディレクトリ）
                # val, dev, test1: フラット構造（直下にファイル）
                
                # trainの場合：階層構造
                for subdir in range(100):  # 0-99のサブディレクトリを検索
                    for ext in ['.jpg', '.jpeg', '.png']:
                        image_path = os.path.join(image_dir, 'train', str(subdir), f"{image_id}{ext}")
                        if os.path.exists(image_path):
                            return True
                
                # val, dev, test1の場合：フラット構造
                for split in ['val', 'dev', 'test1']:
                    for ext in ['.jpg', '.jpeg', '.png']:
                        image_path = os.path.join(image_dir, split, f"{image_id}{ext}")
                        if os.path.exists(image_path):
                            return True
                
                return False
            else:
                # CIRCO: ハッシュ値から実際のファイル名に変換
                if image_id in self.hash_to_filename:
                    filename = self.hash_to_filename[image_id]
                    image_path = os.path.join(image_dir, filename)
                    return os.path.exists(image_path)
                else:
                    # ハッシュ値でない場合（12桁形式）はそのまま使用
                    image_path = os.path.join(image_dir, image_id)
                    return os.path.exists(image_path)
        
        skipped_queries = 0
        processed_queries = 0
        
        async def process_queries():
            nonlocal skipped_queries, processed_queries
            
            for query_item in tqdm(self.data, desc=f"Processing {self.dataset_name} queries"):
                # Ground Truth画像ファイルの存在をチェック
                gt_ids = query_item['ground_truth_ids']
                missing_gt_files = []
                
                for gt_id in gt_ids:
                    if not check_image_exists(gt_id):
                        missing_gt_files.append(gt_id)
                
                # Reference画像ファイルの存在をチェック
                ref_id = query_item['reference_image_id']
                missing_ref_file = not check_image_exists(ref_id)
                
                # Target画像ファイルの存在をチェック
                target_id = query_item['target_image_id']
                missing_target_file = not check_image_exists(target_id)
                
                # 画像ファイルが欠損している場合はスキップ
                if missing_gt_files or missing_ref_file or missing_target_file:
                    skipped_queries += 1
                    if missing_gt_files:
                        print(f"Skipping query {query_item.get('id', '?')}: Missing GT image files {missing_gt_files}")
                    if missing_ref_file:
                        print(f"Skipping query {query_item.get('id', '?')}: Missing reference image file {ref_id}")
                    if missing_target_file:
                        print(f"Skipping query {query_item.get('id', '?')}: Missing target image file {target_id}")
                    continue
                
                # クエリを処理
                try:
                    result = await self.process_single_query(query_item)
                    self.results.append(result)
                    processed_queries += 1
                
                    # 定期的に結果を保存
                    if len(self.results) % 10 == 0:
                        self.save_results()
                            
                except Exception as e:
                    print(f"Error processing query {query_item.get('id', '?')}: {e}")
                    skipped_queries += 1
                    continue
        
        # 非同期処理を実行
        asyncio.run(process_queries())
        
        # 再開モードを考慮した統計表示
        total_processed_in_session = processed_queries
        total_skipped_in_session = skipped_queries
        total_results = len(self.results)  # 既存結果 + 新規処理結果
        
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
        
        # 最終結果を保存
        self.save_results()
        self.print_statistics()
    
    def save_results(self) -> None:
        """結果をJSONファイルに保存"""
        
        def convert_to_json_serializable(obj):
            """PyTorchテンソルやnumpy配列をJSON対応形式に変換"""
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
            """CIRCOの場合、ハッシュ値を12桁の画像IDに変換"""
            if self.dataset_name == 'circo' and hasattr(self, 'hash_to_filename'):
                if hash_or_id in self.hash_to_filename:
                    # ハッシュ値の場合、12桁のファイル名に変換して拡張子を除去
                    filename = self.hash_to_filename[hash_or_id]
                    if filename.endswith('.jpg'):
                        return filename[:-4]  # 拡張子を除去して12桁の数値IDに
                    return filename
            return hash_or_id
        
        # 結果をJSON対応形式に変換（ハッシュ値も画像IDに変換）
        json_results = []
        for result in self.results:
            json_result = convert_to_json_serializable(result.copy())
            
            # CIRCOの場合、ハッシュ値を画像IDに変換
            if self.dataset_name == 'circo':
                # reference_image_id, target_image_id, ground_truth_idsを変換
                json_result['reference_image_id'] = convert_hash_to_image_id(json_result['reference_image_id'])
                json_result['target_image_id'] = convert_hash_to_image_id(json_result['target_image_id'])
                json_result['ground_truth_ids'] = [convert_hash_to_image_id(gt_id) for gt_id in json_result['ground_truth_ids']]
                
                # 各ターンの検索結果とselected_imageも変換
                for turn in json_result.get('turns', []):
                    if 'search_results' in turn:
                        turn['search_results'] = [
                            [convert_hash_to_image_id(img_id), score] 
                            for img_id, score in turn['search_results']
                        ]
                    if 'selected_image' in turn:
                        turn['selected_image'] = convert_hash_to_image_id(turn['selected_image'])
                    
                    # gt_rankingsのキーも変換
                    if 'gt_rankings' in turn:
                        turn['gt_rankings'] = {
                            convert_hash_to_image_id(gt_id): rank 
                            for gt_id, rank in turn['gt_rankings'].items()
                        }
            
            json_results.append(json_result)
        
        # Fashion-IQと同じ形式で結果を構造化
        final_results = {
            "dataset_name": self.dataset_name,
            "config": convert_to_json_serializable(self.config),
            "data_completeness": convert_to_json_serializable(self.completeness_info) if hasattr(self, 'completeness_info') else {},
            "results": json_results
        }
        
        # 結果ファイルに保存
        results_file = f"multiturn_cir_results_{self.dataset_name}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {results_file}")
        print(f"Total results: {len(json_results)}")
    
    def print_statistics(self) -> None:
        """統計情報を出力"""
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
        
        # ターン別成功率の修正版
        turn_success = {}
        turn_distribution = {}  # マルチターン回数の分布（0 = 初期検索のみ）
        max_turns_observed = 0
        
        for result in self.results:
            actual_turns_executed = len(result['turns'])  # 実際に実行されたターン数
            multiturn_count = actual_turns_executed - 1   # マルチターン回数（ターン0を除く）
            max_turns_observed = max(max_turns_observed, actual_turns_executed)
            
            # マルチターン回数をカウント
            turn_distribution[multiturn_count] = turn_distribution.get(multiturn_count, 0) + 1
            
            if result['success']:
                success_turn = result['success_turn']
                turn_success[success_turn] = turn_success.get(success_turn, 0) + 1
        
        print(f"Max turns observed: {max_turns_observed}")
        print(f"Turn distribution: {turn_distribution}")
        print(f"Turn distribution sum: {sum(turn_distribution.values())}")
        
        # 各ターンに到達したクエリ数を計算
        turn_counts = {}
        turn_recall_counts = {}  # 実際に評価されたクエリ数
        
        for turn in range(max_turns_observed):
            # そのターンに到達したクエリ数
            queries_reaching_turn = sum(1 for r in self.results if len(r['turns']) > turn)
            turn_counts[turn + 1] = queries_reaching_turn  # 1-indexedで表示
            
            # そのターンで実際に評価されたクエリ数（GT順位が記録されているもの）
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
        
        # マルチターン回数の分布を正しく表示
        for multiturn_count in sorted(turn_distribution.keys()):
            count = turn_distribution[multiturn_count]
            percentage = count / total_queries * 100
            if multiturn_count == 0:
                print(f"    Initial search only: {count} queries ({percentage:.1f}%)")
            else:
                actual_turns = multiturn_count + 1  # ターン0 + マルチターン
                print(f"    {actual_turns} turns: {count} queries ({percentage:.1f}%)")
        
        # 累積成功率（Hits@T）
        print(f"\nHits@T (Cumulative success rate):")
        cumulative_success = 0
        for turn in range(max_turns_observed):
            turn_successes = turn_success.get(turn, 0)  # 0-indexedのターン番号
            cumulative_success += turn_successes
            cumulative_rate = cumulative_success / total_queries
            if turn == 0:
                print(f"  Initial search: {cumulative_rate:.3f} ({cumulative_rate * 100:.1f}%)")
            else:
                print(f"  Turn {turn + 1}: {cumulative_rate:.3f} ({cumulative_rate * 100:.1f}%)")
        
        # ターン別Recall@10
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
        
        # 最終的なRecall@10（全体の成功率）
        final_recall = successful_queries / total_queries
        print(f"\nFinal Recall@10: {final_recall:.3f} ({final_recall * 100:.1f}%)")
        
        # AUCスコアの計算（簡易版）
        auc_sum = 0
        for turn in range(max_turns_observed):
            turn_successes = turn_success.get(turn, 0)
            cumulative_success = sum(turn_success.get(t, 0) for t in range(turn + 1))
            auc_sum += cumulative_success / total_queries
        
        auc_score = auc_sum / max_turns_observed if max_turns_observed > 0 else 0
        print(f"AUC Score: {auc_score:.3f}")
        
        # ターン別成功率（従来の表示も維持）
        print("\nSuccess by turn:")
        for turn in sorted(turn_success.keys()):
            count = turn_success[turn]
            if turn == 0:
                print(f"  Initial search: {count} queries ({count/total_queries*100:.1f}%)")
            else:
                print(f"  Turn {turn}: {count} queries ({count/total_queries*100:.1f}%)")
        
        # GT順位の統計分析
        print("\n=== Ground Truth Ranking Analysis ===")
        
        # 各ターンでのGT順位統計
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
                
                # Top-k内の割合
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
        
        # GT順位の改善/悪化の分析
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
        
        # 個別クエリの詳細例（最初の5つ）
        print(f"\n=== Sample Query GT Ranking Progression ===")
        
        def convert_hash_to_image_id_for_display(hash_or_id: str) -> str:
            """CIRCOの場合、ハッシュ値を12桁の画像IDに変換（表示用）"""
            if self.dataset_name == 'circo' and hasattr(self, 'hash_to_filename'):
                if hash_or_id in self.hash_to_filename:
                    # ハッシュ値の場合、12桁のファイル名に変換して拡張子を除去
                    filename = self.hash_to_filename[hash_or_id]
                    if filename.endswith('.jpg'):
                        return filename[:-4]  # 拡張子を除去して12桁の数値IDに
                    return filename
            return hash_or_id
        
        for i, result in enumerate(self.results[:5]):
            # Ground truth IDsも変換
            display_gt_ids = [convert_hash_to_image_id_for_display(gt_id) for gt_id in result['ground_truth_ids']]
            print(f"\nQuery {result['query_id']} (GT: {display_gt_ids}):")
            
            for turn_data in result['turns']:
                turn = turn_data['turn']
                best_rank = turn_data.get('best_gt_rank', -1)
                gt_rankings = turn_data.get('gt_rankings', {})
                
                if best_rank > 0:
                    print(f"  Turn {turn}: Best GT rank = {best_rank}")
                    if len(gt_rankings) > 1:
                        # 複数GTがある場合は詳細表示（ハッシュ値を画像IDに変換）
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
        """事前計算されたキャプション特徴量を読み込み"""
        caption_features = {}
        
        # gpt4ominiキャプション特徴量ファイル（PyTorch形式）の読み込み
        if 'gpt4omini_caption_features_file' in self.config and os.path.exists(self.config['gpt4omini_caption_features_file']):
            try:
                features_data = torch.load(self.config['gpt4omini_caption_features_file'], map_location='cpu', weights_only=False)
                
                if isinstance(features_data, dict):
                    # 新しい配列形式のデータかチェック
                    if 'features' in features_data and 'image_ids' in features_data:
                        # 配列形式: {'features': tensor, 'image_ids': list}
                        features_tensor = features_data['features']  # Shape: (N, feature_dim)
                        image_ids = features_data['image_ids']  # List of image paths
                        
                        # numpy配列の場合はテンソルに変換
                        if isinstance(features_tensor, np.ndarray):
                            features_tensor = torch.from_numpy(features_tensor).float()
                        
                        # 画像ID別の辞書に変換
                        for i, image_id in enumerate(image_ids):
                            # フルパスから画像ファイル名を抽出（拡張子なし）
                            if '/' in image_id:
                                # fashion-iq/images/dress/B008BHCT58.jpg -> B008BHCT58
                                # CIRCO/COCO2017_unlabeled/unlabeled2017/000000212560.jpg -> 000000212560.jpg
                                image_filename = image_id.split('/')[-1]
                                if '.' in image_filename:
                                    image_filename_no_ext = image_filename.rsplit('.', 1)[0]  # 拡張子を除去
                                else:
                                    image_filename_no_ext = image_filename
                            else:
                                # 既にファイル名のみの場合
                                image_filename = image_id
                                if '.' in image_filename:
                                    image_filename_no_ext = image_filename.rsplit('.', 1)[0]  # 拡張子を除去
                                else:
                                    image_filename_no_ext = image_filename
                            
                            # CIRCOの場合は拡張子付きファイル名もキーとして保存
                            if self.dataset_name == 'circo':
                                caption_features[image_filename] = features_tensor[i]  # 拡張子付き
                                caption_features[image_filename_no_ext] = features_tensor[i]  # 拡張子なし
                            else:
                                caption_features[image_filename_no_ext] = features_tensor[i]
                        
                        print(f"Loaded caption features for {len(caption_features)} images from array format")
                    else:
                        # 従来の辞書形式: {image_id: tensor, ...}
                        for image_id, features in features_data.items():
                            # numpy配列の場合はテンソルに変換
                            if isinstance(features, np.ndarray):
                                features = torch.from_numpy(features).float()
                            caption_features[image_id] = features
                        print(f"Loaded caption features for {len(caption_features)} images from dict format")
                else:
                    print(f"Warning: Unexpected format in caption features file: {self.config['gpt4omini_caption_features_file']}")
                    
            except Exception as e:
                print(f"Warning: Failed to load caption features from {self.config['gpt4omini_caption_features_file']}: {e}")
        
        # CIRCOの場合、ハッシュ値でもアクセスできるようにマッピングを追加
        if self.dataset_name == 'circo' and hasattr(self, 'hash_to_filename'):
            filename_to_hash = {filename: hash_val for hash_val, filename in self.hash_to_filename.items()}
            
            # 12桁ファイル名からハッシュ値へのマッピングを作成
            additional_mappings = {}
            for key, features in caption_features.items():
                # キーが12桁のファイル名（拡張子付きまたはなし）の場合
                if key.endswith('.jpg'):
                    # 拡張子付きファイル名の場合
                    if key in filename_to_hash:
                        hash_val = filename_to_hash[key]
                        additional_mappings[hash_val] = features
                else:
                    # 拡張子なしファイル名の場合、拡張子付きで検索
                    filename_with_ext = f"{key}.jpg"
                    if filename_with_ext in filename_to_hash:
                        hash_val = filename_to_hash[filename_with_ext]
                        additional_mappings[hash_val] = features
            
            # ハッシュ値でのアクセスを追加
            caption_features.update(additional_mappings)
            print(f"Added {len(additional_mappings)} hash-based mappings for CIRCO caption features")
        
        return caption_features

    def check_data_completeness(self) -> Dict[str, Any]:
        """データの完全性をチェックし、欠損情報を返す"""
        print(f"\n=== Checking data completeness for {self.dataset_name} ===")
        
        # 画像ディレクトリの設定
        image_dir = self.config.get('image_dir', '')
        
        # データセットクエリで使用される画像IDを収集
        reference_ids = set()
        target_ids = set()
        gt_ids = set()
        
        for query_item in self.data:
            reference_ids.add(query_item['reference_image_id'])
            target_ids.add(query_item['target_image_id'])
            gt_ids.update(query_item['ground_truth_ids'])
        
        # 画像ファイルの存在確認
        def check_image_exists(image_id: str) -> bool:
            """画像ファイルが存在するかチェック"""
            if self.dataset_name.startswith('fashioniq'):
                # Fashion-IQ: カテゴリサブディレクトリを含め、拡張子も自動追加
                category = self.dataset_name.split('_')[1]  # dress, shirt, toptee
                img_with_ext = image_id if image_id.endswith('.jpg') else f"{image_id}.jpg"
                image_path = os.path.join(image_dir, category, img_with_ext)
                return os.path.exists(image_path)
            elif self.dataset_name.startswith('cirr'):
                # CIRRの場合：splitによって構造が異なる
                # train: 階層構造（0-99のサブディレクトリ）
                # val, dev, test1: フラット構造（直下にファイル）
                
                # trainの場合：階層構造
                for subdir in range(100):  # 0-99のサブディレクトリを検索
                    for ext in ['.jpg', '.jpeg', '.png']:
                        image_path = os.path.join(image_dir, 'train', str(subdir), f"{image_id}{ext}")
                        if os.path.exists(image_path):
                            return True
                
                # val, dev, test1の場合：フラット構造
                for split in ['val', 'dev', 'test1']:
                    for ext in ['.jpg', '.jpeg', '.png']:
                        image_path = os.path.join(image_dir, split, f"{image_id}{ext}")
                        if os.path.exists(image_path):
                            return True
                
                return False
            else:
                # CIRCO: ハッシュ値から実際のファイル名に変換
                if image_id in self.hash_to_filename:
                    filename = self.hash_to_filename[image_id]
                    image_path = os.path.join(image_dir, filename)
                    return os.path.exists(image_path)
                else:
                    # ハッシュ値でない場合（12桁形式）はそのまま使用
                    image_path = os.path.join(image_dir, image_id)
                    return os.path.exists(image_path)
        
        # 欠損分析
        print("Checking image file existence...")
        missing_ref_images = [img_id for img_id in reference_ids if not check_image_exists(img_id)]
        missing_target_images = [img_id for img_id in target_ids if not check_image_exists(img_id)]
        missing_gt_images = [img_id for img_id in gt_ids if not check_image_exists(img_id)]
        
        # 検索空間の画像IDセット
        search_space_set = set(self.retrieval_engine.search_space)
        missing_ref_search = reference_ids - search_space_set
        missing_target_search = target_ids - search_space_set
        missing_gt_search = gt_ids - search_space_set
        
        # 利用可能なキャプション特徴量も参考情報として表示
        available_caption_features = set(self.caption_features.keys())
        missing_ref_features = reference_ids - available_caption_features
        missing_target_features = target_ids - available_caption_features
        missing_gt_features = gt_ids - available_caption_features
        
        # 結果をまとめ
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
        
        # 詳細レポート出力
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
        
        # 処理可能なクエリの推定（画像ファイル存在ベース）
        processable_queries = 0
        for query_item in self.data:
            gt_available = all(check_image_exists(gt_id) for gt_id in query_item['ground_truth_ids'])
            ref_available = check_image_exists(query_item['reference_image_id'])
            target_available = check_image_exists(query_item['target_image_id'])
            
            if gt_available and ref_available and target_available:
                processable_queries += 1
        
        print(f"\nEstimated processable queries (based on image files): {processable_queries}/{len(self.data)} "
              f"({processable_queries/len(self.data)*100:.1f}%)")
        
        # 欠損画像のサンプルを表示（デバッグ用）
        if missing_gt_images:
            sample_missing = missing_gt_images[:5]
            print(f"\nSample missing GT image files: {sample_missing}")
        if missing_ref_images:
            sample_missing = missing_ref_images[:3]
            print(f"Sample missing reference image files: {sample_missing}")
        
        completeness_info['estimated_processable_queries'] = processable_queries
        
        return completeness_info

def main():
    """メイン関数"""
    # コマンドライン引数のパーサーを作成
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
    
    args = parser.parse_args()
    
    # グローバルなデバイス設定を更新
    global device
    if args.device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # 設定
    datasets_config = {
        'circo': {
            'annotation_file': 'CIRCO/annotations/val.json',
            'corpus_vectors_file': 'CIRCO/features_blip.pt',  # BLIPの特徴量ファイル
            'search_space_file': 'CIRCO/metadata_blip.pt',    # 検索空間のメタデータ
            'image_dir': 'CIRCO/COCO2017_unlabeled/unlabeled2017',  # COCO2017 unlabeled 120k画像プール
            'caption_file': 'CIRCO/captions_gpt4omini.json',     # GPT-4o-miniキャプション
            'gpt4omini_caption_features_file': 'CIRCO/gpt4omini_captions_blip_features.pt',  # gpt4ominiキャプション特徴量
            'use_blip': True,
            'max_turns': args.max_turns,
            'dataset_split': 'val',  # testは正解ラベル非公開のためval使用
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
            'annotation_file': 'cirr/captions/cap.rc2.val.json',  # valセット使用（testは正解非公開）
            'corpus_vectors_file': 'cirr/features_blip.pt',       # BLIPの特徴量ファイル
            'search_space_file': 'cirr/image_splits/split.rc2.val.json',  # val画像プール
            'image_dir': 'cirr/img_raw',                          # CIRR画像ディレクトリ
            'caption_file': 'cirr/captions_gpt4omini.json',          # GPT-4o-miniキャプション
            'gpt4omini_caption_features_file': 'cirr/gpt4omini_captions_blip_features.pt',  # gpt4ominiキャプション特徴量
            'use_blip': True,
            'max_turns': args.max_turns,
            'dataset_split': 'val',  # testは正解ラベル非公開のためval使用
            'caption_mode': 'combined'
        },
        # Fashion-IQ Dress カテゴリ
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
        # Fashion-IQ Shirt カテゴリ
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
        # Fashion-IQ Toptee カテゴリ
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
    
    # 実行するデータセットを選択
    dataset_name = args.dataset
    
    if dataset_name not in datasets_config:
        print(f"Unknown dataset: {dataset_name}")
        return
    
    config = datasets_config[dataset_name]
    
    # Fashion-IQデータセットの場合、キャプション処理方式を設定
    if dataset_name.startswith('fashioniq'):
        config['caption_mode'] = args.caption_mode
        print(f"Using caption mode: {args.caption_mode}")
    
    # システムを初期化
    system = MultiTurnCIRSystem(dataset_name, config)
    
    # テストモードの場合は制限
    if args.test_mode:
        print("Running in test mode - limiting to first 5 queries")
        system.data = system.data[:5]
    
    # 再開機能の説明
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
    
    # 評価を実行
    system.run_evaluation()

if __name__ == "__main__":
    main() 