#!/usr/bin/env python3
"""
GPT-4o miniで生成されたキャプションをCLIPとBLIPで特徴量変換して保存するスクリプト
CIRR、CIRCO、Fashion-IQデータセット対応
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
        print(f"使用デバイス: {self.device}")
        
        # CLIPモデルの初期化（OpenCLIP ViT-L/14）
        print("CLIPモデルをロード中...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)
        self.clip_model.eval()
        
        # BLIPモデルの初期化（large版に統一）
        print("BLIPモデルをロード中...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
        self.blip_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to(device)
        self.blip_model.eval()
        
        print("モデルのロードが完了しました")
    
    def extract_clip_features(self, captions, batch_size=32):
        """CLIPでキャプションの特徴量を抽出"""
        print(f"CLIP特徴量抽出開始: {len(captions)}件のキャプション")
        
        features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(captions), batch_size), desc="CLIP特徴量抽出"):
                batch_captions = captions[i:i+batch_size]
                
                # OpenCLIPでテキストをトークン化
                text_tokens = clip.tokenize(batch_captions, truncate=True).to(self.device)
                
                # 特徴量を抽出
                text_features = self.clip_model.encode_text(text_tokens).to(torch.float32)
                text_features = F.normalize(text_features, dim=-1)
                
                features.append(text_features.cpu().numpy())
        
        return np.vstack(features)
    
    def extract_blip_features(self, captions, batch_size=16):
        """BLIPでキャプションの特徴量を抽出"""
        print(f"BLIP特徴量抽出開始: {len(captions)}件のキャプション")
        
        features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(captions), batch_size), desc="BLIP特徴量抽出"):
                batch_captions = captions[i:i+batch_size]
                
                # テキストをトークン化
                inputs = self.blip_processor(
                    text=batch_captions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # BLIPモデルでテキスト特徴量を抽出（参考コードと同じ方法）
                question_embeds = self.blip_model.text_encoder(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    return_dict=True
                ).last_hidden_state
                
                # CLSトークン（最初のトークン）を使用してtext_projを通す
                text_features = self.blip_model.text_proj(question_embeds[:, 0, :])
                text_features = F.normalize(text_features, dim=-1)
                
                features.append(text_features.cpu().numpy())
        
        return np.vstack(features)

def load_gpt4omini_captions(caption_file):
    """GPT-4o miniキャプションファイルを読み込み"""
    print(f"キャプションファイル読み込み: {caption_file}")
    
    with open(caption_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    captions = []
    image_ids = []
    
    # データ構造に応じて処理
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
    
    print(f"読み込み完了: {len(captions)}件のキャプション")
    return captions, image_ids

def save_features(features, image_ids, output_file, feature_type):
    """特徴量をファイルに保存"""
    print(f"{feature_type}特徴量を保存中: {output_file}")
    
    # 特徴量データの構造
    feature_data = {
        'features': features,
        'image_ids': image_ids,
        'feature_type': feature_type,
        'feature_dim': features.shape[1]
    }
    
    torch.save(feature_data, output_file)
    print(f"保存完了: {features.shape} -> {output_file}")

def merge_features(existing_features, existing_image_ids, new_features, new_image_ids, feature_type):
    """既存の特徴量と新しい特徴量をマージ"""
    print(f"{feature_type}特徴量をマージ中...")
    print(f"  既存: {len(existing_image_ids)}枚")
    print(f"  新規: {len(new_image_ids)}枚")
    
    # numpy配列の場合はテンソルに変換
    if isinstance(existing_features, np.ndarray):
        existing_features = torch.from_numpy(existing_features).float()
    if isinstance(new_features, np.ndarray):
        new_features = torch.from_numpy(new_features).float()
    
    # 特徴量を結合
    merged_features = torch.cat([existing_features, new_features], dim=0)
    merged_image_ids = existing_image_ids + new_image_ids
    
    print(f"  マージ後: {len(merged_image_ids)}枚")
    return merged_features, merged_image_ids

def extract_missing_features(extractor, all_captions, all_image_ids, existing_image_ids, feature_type):
    """不足している画像の特徴量のみを抽出"""
    # 不足している画像IDを特定
    existing_set = set(existing_image_ids)
    missing_indices = [i for i, img_id in enumerate(all_image_ids) if img_id not in existing_set]
    
    if not missing_indices:
        print(f"{feature_type}: 不足している画像はありません")
        return None, []
    
    print(f"{feature_type}: {len(missing_indices)}枚の不足画像を検出")
    
    # 不足している画像のキャプションを抽出
    missing_captions = [all_captions[i] for i in missing_indices]
    missing_image_ids = [all_image_ids[i] for i in missing_indices]
    
    # 特徴量抽出
    if feature_type == 'CLIP':
        missing_features = extractor.extract_clip_features(missing_captions)
    elif feature_type == 'BLIP':
        missing_features = extractor.extract_blip_features(missing_captions)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    return missing_features, missing_image_ids

def process_dataset(dataset_name, caption_file, output_dir, extractor, force_reprocess=False):
    """各データセットの処理（増分更新対応）"""
    print(f"\n=== {dataset_name}データセット処理開始 ===")
    
    # キャプション読み込み
    captions, image_ids = load_gpt4omini_captions(caption_file)
    
    if len(captions) == 0:
        print(f"警告: {dataset_name}のキャプションが見つかりません")
        return
    
    # 出力ディレクトリ作成
    dataset_output_dir = Path(output_dir) / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 出力ファイルパス
    clip_output_file = dataset_output_dir / 'gpt4omini_captions_clip_features.pt'
    blip_output_file = dataset_output_dir / 'gpt4omini_captions_blip_features.pt'
    
    # CLIP特徴量の処理（増分更新対応）
    if clip_output_file.exists() and not force_reprocess:
        print(f"CLIP特徴量ファイルが既に存在します: {clip_output_file}")
        try:
            existing_data = torch.load(clip_output_file, weights_only=False)
            existing_image_ids = existing_data['image_ids']
            existing_features = existing_data['features']
            
            if len(existing_image_ids) == len(image_ids) and existing_image_ids == image_ids:
                print("CLIP特徴量は既に完了しています。スキップします。")
            else:
                print(f"既存CLIP特徴量: {len(existing_image_ids)}枚, 必要: {len(image_ids)}枚")
                
                # 不足分のみ抽出
                missing_features, missing_image_ids = extract_missing_features(
                    extractor, captions, image_ids, existing_image_ids, 'CLIP'
                )
                
                if missing_features is not None:
                    # 既存と新規をマージ
                    merged_features, merged_image_ids = merge_features(
                        existing_features, existing_image_ids, 
                        missing_features, missing_image_ids, 'CLIP'
                    )
                    save_features(merged_features, merged_image_ids, clip_output_file, 'CLIP')
                else:
                    print("CLIP特徴量: 追加する画像がありません")
                    
        except Exception as e:
            print(f"既存のCLIP特徴量ファイルの読み込みに失敗: {e}")
            print("CLIP特徴量を一から再処理します。")
            clip_features = extractor.extract_clip_features(captions)
            save_features(clip_features, image_ids, clip_output_file, 'CLIP')
    else:
        if force_reprocess and clip_output_file.exists():
            print("強制再処理モード: 既存のCLIP特徴量ファイルを上書きします。")
        else:
            print("CLIP特徴量を新規作成します。")
        clip_features = extractor.extract_clip_features(captions)
        save_features(clip_features, image_ids, clip_output_file, 'CLIP')
    
    # BLIP特徴量の処理（増分更新対応）
    if blip_output_file.exists() and not force_reprocess:
        print(f"BLIP特徴量ファイルが既に存在します: {blip_output_file}")
        try:
            existing_data = torch.load(blip_output_file, weights_only=False)
            existing_image_ids = existing_data['image_ids']
            existing_features = existing_data['features']
            
            if len(existing_image_ids) == len(image_ids) and existing_image_ids == image_ids:
                print("BLIP特徴量は既に完了しています。スキップします。")
            else:
                print(f"既存BLIP特徴量: {len(existing_image_ids)}枚, 必要: {len(image_ids)}枚")
                
                # 不足分のみ抽出
                missing_features, missing_image_ids = extract_missing_features(
                    extractor, captions, image_ids, existing_image_ids, 'BLIP'
                )
                
                if missing_features is not None:
                    # 既存と新規をマージ
                    merged_features, merged_image_ids = merge_features(
                        existing_features, existing_image_ids, 
                        missing_features, missing_image_ids, 'BLIP'
                    )
                    save_features(merged_features, merged_image_ids, blip_output_file, 'BLIP')
                else:
                    print("BLIP特徴量: 追加する画像がありません")
                    
        except Exception as e:
            print(f"既存のBLIP特徴量ファイルの読み込みに失敗: {e}")
            print("BLIP特徴量を一から再処理します。")
            blip_features = extractor.extract_blip_features(captions)
            save_features(blip_features, image_ids, blip_output_file, 'BLIP')
    else:
        if force_reprocess and blip_output_file.exists():
            print("強制再処理モード: 既存のBLIP特徴量ファイルを上書きします。")
        else:
            print("BLIP特徴量を新規作成します。")
        blip_features = extractor.extract_blip_features(captions)
        save_features(blip_features, image_ids, blip_output_file, 'BLIP')
    
    print(f"=== {dataset_name}データセット処理完了 ===")

def main():
    parser = argparse.ArgumentParser(description='GPT-4o miniキャプションから特徴量抽出')
    parser.add_argument('--datasets', nargs='+', 
                       choices=['cirr', 'circo', 'fashion-iq', 'all'],
                       default=['all'],
                       help='処理するデータセット')
    parser.add_argument('--output-dir', type=str, 
                       default='caption_features',
                       help='出力ディレクトリ')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='使用デバイス')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='既存の特徴量ファイルを無視して強制的に再処理')
    
    args = parser.parse_args()
    
    print("=== GPT-4o miniキャプション特徴量抽出開始 ===")
    print(f"使用デバイス: {args.device}")
    print(f"CUDA利用可能: {torch.cuda.is_available()}")
    if args.force_reprocess:
        print("強制再処理モード: 既存ファイルを上書きします")
    
    # 特徴量抽出器を初期化
    extractor = CaptionFeatureExtractor(device=args.device)
    
    # データセット設定（クリーンアップ済みファイルを使用）
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
    
    # 処理するデータセットを決定
    if 'all' in args.datasets:
        datasets_to_process = list(dataset_configs.keys())
    else:
        datasets_to_process = args.datasets
    
    # 各データセットを処理
    for dataset_key in datasets_to_process:
        if dataset_key in dataset_configs:
            config = dataset_configs[dataset_key]
            caption_file = config['caption_file']
            
            # ファイル存在確認
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
                    print(f"エラー: {config['name']}の処理中にエラーが発生しました: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"警告: {caption_file}が見つかりません")
        else:
            print(f"警告: 不明なデータセット: {dataset_key}")
    
    print("\n=== 全ての処理が完了しました ===")

if __name__ == "__main__":
    main() 