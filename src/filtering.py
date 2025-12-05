#!/usr/bin/env python3
"""
最適化されたマルチターンCIRデータフィルタリングシステム
既存のJSONとCSVファイルを活用した効率的なフィルタリング
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
    """既存データを活用した最適化されたマルチターンCIRフィルタリングシステム"""
    
    def __init__(self, similarity_threshold: float = 0.8, rank_margin: int = 30):
        """
        Args:
            similarity_threshold: CLIP類似度の閾値
            rank_margin: ランクマージンの閾値
        """
        self.similarity_threshold = similarity_threshold
        self.rank_margin = rank_margin
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.clip_available = CLIP_AVAILABLE
        
        # CLIP初期化
        if self.clip_available:
            try:
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
                print(f"CLIP model loaded on {self.device}")
            except Exception as e:
                print(f"Failed to load CLIP: {e}")
                self.clip_available = False
        
        # 画像ハッシュマッピング
        self.image_id_to_hash = {}
    
    def _compute_image_hash(self, image_path: str) -> str:
        """画像ファイルのMD5ハッシュを計算"""
        try:
            with open(image_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            print(f"Warning: Failed to compute hash for {image_path}: {e}")
            return None
    
    def _initialize_fashioniq_hash_mapping(self, dataset_name: str):
        """FashionIQ画像のハッシュマッピングを初期化"""
        category = dataset_name.split('_')[1] if '_' in dataset_name else 'dress'
        image_dir = f'fashion-iq/images'
        
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
                            self.image_id_to_hash[f"{image_id}.jpg"] = img_hash

    def _initialize_cirr_hash_mapping(self, dataset_name: str):
        """CIRR画像のハッシュマッピングを初期化"""
        image_dir = 'cirr/img_raw'
        
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

    def _initialize_circo_hash_mapping(self, dataset_name: str):
        """CIRCO画像のハッシュマッピングを初期化"""
        try:
            import torch
            
            # CIRCOのメタデータファイルから既存のハッシュマッピングを読み込み
            metadata_file = 'CIRCO/metadata_blip.pt'
            if not os.path.exists(metadata_file):
                print(f"Warning: CIRCO metadata file not found: {metadata_file}")
                return
            
            metadata = torch.load(metadata_file, map_location='cpu', weights_only=False)
            hash_to_idx = metadata['hash_to_idx']
            idx_to_info = metadata['idx_to_info']
            
            # ハッシュ→ファイル名のマッピングを作成
            hash_to_filename = {}
            for hash_val, idx in hash_to_idx.items():
                info = idx_to_info.get(idx, {})
                image_id = info.get('image_id', '')
                if image_id:
                    filename = f"{int(image_id):012d}.jpg"
                    hash_to_filename[hash_val] = filename
            
            # 画像ID→ハッシュのマッピングを作成
            for hash_val, filename in hash_to_filename.items():
                # ファイル名から画像IDを抽出（例：000000000001.jpg → 1）
                image_id = filename.replace('.jpg', '').lstrip('0') or '0'
                self.image_id_to_hash[image_id] = hash_val
                # 12桁形式も追加
                padded_id = f"{int(image_id):012d}"
                self.image_id_to_hash[padded_id] = hash_val
                
        except ImportError:
            print("Warning: PyTorch not available, cannot load CIRCO metadata")
        except Exception as e:
            print(f"Warning: Failed to load CIRCO metadata: {e}")

    def get_image_hash(self, image_id: str) -> str:
        """画像IDからハッシュ値を取得"""
        if not image_id:
            return None
            
        # 直接マッチを試す
        if image_id in self.image_id_to_hash:
            return self.image_id_to_hash[image_id]
        
        # 拡張子の有無を考慮
        if image_id.endswith('.jpg'):
            base_id = image_id[:-4]
            if base_id in self.image_id_to_hash:
                return self.image_id_to_hash[base_id]
        else:
            jpg_id = f"{image_id}.jpg"
            if jpg_id in self.image_id_to_hash:
                return self.image_id_to_hash[jpg_id]
        
        # CIRCOの場合、12桁形式も試す
        if hasattr(self, 'dataset_name') and self.dataset_name == 'circo':
            try:
                padded_id = f"{int(image_id):012d}"
                if padded_id in self.image_id_to_hash:
                    return self.image_id_to_hash[padded_id]
            except ValueError:
                pass
        
        # FashionIQの場合、追加のパターンを試す
        if hasattr(self, 'dataset_name') and self.dataset_name.startswith('fashioniq'):
            # B00006M009.jpg のような形式を試す
            variations = [
                image_id,
                f"{image_id}.jpg",
                image_id.replace('.jpg', '') if image_id.endswith('.jpg') else f"{image_id}.jpg"
            ]
            
            for variation in variations:
                if variation in self.image_id_to_hash:
                    return self.image_id_to_hash[variation]
        
        # デバッグ用（最初の数個のみ表示）
        if not hasattr(self, '_hash_miss_count'):
            self._hash_miss_count = 0
        
        self._hash_miss_count += 1
        if self._hash_miss_count <= 5:
            print(f"Debug: Hash not found for image_id: '{image_id}' (miss #{self._hash_miss_count})")
            if self._hash_miss_count == 5:
                print("Debug: Suppressing further hash miss messages...")
        
        return None
    
    def load_dataset_files(self, dataset_name: str) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
        """データセットファイルを読み込み（JSONのみでも対応）"""
        # ファイルパス
        json_file = f"multiturn_cir_results_{dataset_name}.json"
        summary_file = f"multiturn_cir_summary_{dataset_name}.csv"
        detailed_file = f"multiturn_cir_detailed_rankings_{dataset_name}.csv"
        
        # JSONファイルの存在確認
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Required JSON file not found: {json_file}")
        
        # データ読み込み
        print(f"Loading {dataset_name} data...")
        
        # JSON結果
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        # CSVファイルの存在確認
        csv_files_exist = os.path.exists(summary_file) and os.path.exists(detailed_file)
        
        if csv_files_exist:
            # CSVファイルが存在する場合（FashionIQ等）
            summary_df = pd.read_csv(summary_file)
            detailed_df = pd.read_csv(detailed_file)
            print(f"Loaded: {len(json_data['results'])} results, {len(summary_df)} summary rows, {len(detailed_df)} detailed rows")
        else:
            # CSVファイルが存在しない場合（CIRCO, CIRR等）- JSONから生成
            print(f"CSV files not found. Generating from JSON data...")
            summary_df, detailed_df = self._generate_dataframes_from_json(json_data)
            print(f"Loaded: {len(json_data['results'])} results, {len(summary_df)} summary rows (generated), {len(detailed_df)} detailed rows (generated)")
        
        return json_data, summary_df, detailed_df
    
    def _generate_dataframes_from_json(self, json_data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """JSONデータからCSV相当のDataFrameを生成"""
        summary_data = []
        detailed_data = []
        
        for result in json_data['results']:
            query_id = int(result['query_id'])
            success = result.get('success', False)
            success_turn = result.get('success_turn', -1) if success else -1
            turns = result.get('turns', [])
            total_turns = len(turns)
            
            # 最初と最後のランクから改善値を計算
            initial_rank = turns[0].get('best_gt_rank', float('inf')) if turns else float('inf')
            final_rank = turns[-1].get('best_gt_rank', float('inf')) if turns else float('inf')
            rank_improvement = initial_rank - final_rank if initial_rank != float('inf') and final_rank != float('inf') else 0
            
            # サマリーデータ
            summary_data.append({
                'query_id': query_id,
                'success': success,
                'success_turn': success_turn,
                'total_turns': total_turns,
                'gt_rank_improvement': rank_improvement,
                'initial_rank': initial_rank,
                'final_rank': final_rank
            })
            
            # 詳細データ（各ターン）
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
        """CLIPテキスト特徴量を取得"""
        if not self.clip_available:
            return np.random.rand(512)  # ダミー特徴量
        
        with torch.no_grad():
            tokens = clip.tokenize(text, truncate=True).to(self.device)
            feat = self.clip_model.encode_text(tokens)
            return normalize(feat, dim=-1).squeeze(0).cpu().numpy()
    
    def apply_success_filter(self, json_data: Dict, summary_df: pd.DataFrame) -> List[int]:
        """成功フィルタを適用"""
        # JSONデータから直接成功クエリを特定
        successful_query_ids = []
        
        for result in json_data['results']:
            try:
                query_id = int(result['query_id'])
                
                # 結果レベルのsuccessフィールドを確認
                if result.get('success', False):
                    successful_query_ids.append(query_id)
                    
            except (ValueError, TypeError):
                continue
        
        return successful_query_ids

    def apply_multiturn_filter(self, json_data: Dict, query_ids: List[int]) -> List[int]:
        """マルチターンフィルタを適用（ターン1以降で成功したもののみ）"""
        # JSONの結果を辞書に変換
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
            
            # マルチターン（ターン1以降で成功）の条件
            if success_turn > 0:
                multiturn_query_ids.append(query_id)
        
        return multiturn_query_ids

    def apply_original_query_filter(self, json_data: Dict, query_ids: List[int]) -> List[int]:
        """オリジナルクエリフィルタを適用（original_queryが空のクエリを除外）"""
        # JSONの結果を辞書に変換
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
            
            # original_queryが存在し、空でない場合のみ保持
            if original_query and original_query.strip():
                valid_query_ids.append(query_id)
            else:
                empty_query_count += 1
        
        print(f"Original query filter: {empty_query_count} queries removed for having empty original_query")
        return valid_query_ids

    def apply_rank_margin_filter(self, json_data: Dict, detailed_df: pd.DataFrame, query_ids: List[int]) -> List[int]:
        """ランクマージンフィルタを適用（前ターンから30以上ランクが悪化した場合に除外）"""
        
        # JSONクエリIDを整数に変換して収集
        json_query_ids = set()
        for result in json_data['results']:
            try:
                query_id = int(result['query_id'])
                json_query_ids.add(query_id)
            except (ValueError, TypeError):
                continue
        
        csv_query_ids = set(detailed_df['query_id'].tolist())
        
        # 共通のクエリIDのみを対象とする
        valid_query_ids = [qid for qid in query_ids if qid in json_query_ids and qid in csv_query_ids]
        
        print(f"Debug: JSON IDs: {len(json_query_ids)}, CSV IDs: {len(csv_query_ids)}, Valid IDs: {len(valid_query_ids)}")
        
        if len(valid_query_ids) == 0:
            print("Warning: No common query IDs found between JSON and CSV data - skipping rank margin filter")
            return query_ids  # ランクマージンフィルタをスキップ
        
        filtered_query_ids = []
        
        for query_id in tqdm(valid_query_ids, desc="Rank margin filtering"):
            # 該当クエリのランキングデータを取得
            query_ranks = detailed_df[detailed_df['query_id'] == query_id].sort_values('turn')
            
            if len(query_ranks) <= 1:
                # シングルターンの場合は除外
                continue
            
            ranks = query_ranks['best_gt_rank'].tolist()
            
            # float('inf')を除外して有効なランクのみを処理
            valid_ranks = [r for r in ranks if r != float('inf')]
            if len(valid_ranks) <= 1:
                # 有効なランクが1つ以下の場合はスキップ
                continue
            
            keep = True
            
            # 連続するターン間でランクマージンをチェック
            for i in range(1, len(valid_ranks)):
                # ランクが30以上悪化した場合（数値が大きくなった場合）
                if valid_ranks[i] > valid_ranks[i-1] + self.rank_margin:
                    keep = False
                    break
            
            if keep:
                filtered_query_ids.append(query_id)
        
        # 元のquery_idsに含まれていたが、CSVに存在しないクエリIDも含める
        missing_ids = [qid for qid in query_ids if qid not in csv_query_ids]
        filtered_query_ids.extend(missing_ids)
        
        return filtered_query_ids
    
    def apply_clip_similarity_filter(self, json_data: Dict, query_ids: List[int]) -> List[int]:
        """CLIP類似度フィルタを適用"""
        if not self.clip_available:
            print("CLIP not available, skipping similarity filter")
            return query_ids
        
        # JSONの結果を辞書に変換（整数IDをキーとして使用）
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
                # JSONに存在しないクエリIDはスキップ
                continue
            
            result = results_dict[query_id]
            turns = result.get('turns', [])
            
            if len(turns) <= 1:
                # シングルターンは類似度フィルタの対象外
                filtered_query_ids.append(query_id)
                continue
            
            # 修正テキストの類似度チェック
            past_features = []
            keep = True
            
            for i, turn in enumerate(turns):
                # 修正テキストを取得
                if i == 0:
                    modification_text = result.get('original_query', '')
                else:
                    modification_text = turn.get('relative_caption', '')
                
                if not modification_text:
                    continue
                
                feat = self.get_clip_text_feature(modification_text)
                
                # 過去の特徴量との類似度チェック
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
        """画像重複フィルタを適用（同一ダイアログ内で同じ画像を複数回選択したクエリ、およびselected_imageとreference_imageが重複するクエリを除外）"""
        
        # JSONの結果を辞書に変換
        results_dict = {}
        for result in json_data['results']:
            try:
                query_id = int(result['query_id'])
                results_dict[query_id] = result
            except (ValueError, TypeError):
                continue
        
        # 統計情報収集用
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
                # クエリが見つからない場合はスキップ
                continue
            
            result = results_dict[query_id]
            turns = result.get('turns', [])
            reference_image_id = result.get('reference_image_id', '')
            
            # reference_imageのハッシュを取得
            reference_hash = None
            if reference_image_id:
                reference_hash = self.get_image_hash(reference_image_id)
                if reference_hash:
                    hash_success_count += 1
                else:
                    hash_failure_count += 1
                total_images_processed += 1
            
            # このダイアログで選択された画像のハッシュを追跡
            dialog_image_hashes = []
            dialog_images_info = []  # デバッグ用
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
                        
                        # reference_imageとselected_imageの重複チェック
                        if reference_hash and image_hash == reference_hash:
                            has_reference_selected_duplicate = True
                            # 重複例を記録（最初の数例のみ）
                            if len(reference_selected_examples) < 5:
                                reference_selected_examples.append({
                                    'query_id': query_id,
                                    'reference_image': reference_image_id,
                                    'selected_image': selected_image,
                                    'turn': turn_num,
                                    'image_hash': image_hash[:8] + '...'
                                })
                            break
                        
                        # このダイアログ内で既に同じハッシュの画像が選択されているかチェック
                        if image_hash in dialog_image_hashes:
                            has_duplicate_in_dialog = True
                            # 重複例を記録（最初の数例のみ）
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
                        # ハッシュが取得できない場合は重複チェックをスキップ
                        dialog_images_info.append((turn_num, selected_image, 'NO_HASH'))
            
            # ダイアログ内で重複がなく、reference-selected重複もない場合のみ保持
            if not has_duplicate_in_dialog and not has_reference_selected_duplicate:
                filtered_query_ids.append(query_id)
            else:
                if has_duplicate_in_dialog:
                    duplicate_within_dialog_count += 1
                if has_reference_selected_duplicate:
                    reference_selected_duplicate_count += 1
        
        # 統計情報を表示
        print(f"Hash statistics: {hash_success_count} success, {hash_failure_count} failures out of {total_images_processed} total images")
        print(f"Image duplication filter: {duplicate_within_dialog_count} queries removed for selecting duplicate images within same dialog")
        print(f"Reference-Selected duplication filter: {reference_selected_duplicate_count} queries removed for selecting reference image as selected image")
        
        # 重複例を表示
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
        """フィルタリング済みデータセットを作成（元のダイアログ構造を保持）"""
        filtered_results = []
        filtered_query_ids_set = set(filtered_query_ids)  # 高速な検索のためにセットに変換
        
        # 元のresultsリストから直接フィルタリング
        for result in json_data['results']:
            try:
                # query_idを整数として取得
                query_id = int(result['query_id'])
                # フィルタリング済みクエリIDに含まれるかを確認
                if query_id in filtered_query_ids_set:
                    # 同じクエリIDでも個別にsuccess_turnをチェック
                    success_turn = result.get('success_turn', 0)
                    if success_turn > 0:  # マルチターンフィルタの条件を再適用
                        filtered_results.append(result)
                    # success_turn == 0 の場合は除外（ログ出力）
                    elif success_turn == 0:
                        print(f"Debug: Excluding duplicate entry with success_turn=0 for query_id {query_id}")
            except (ValueError, TypeError):
                # query_idの変換に失敗した場合は警告を出力してスキップ
                print(f"Warning: Invalid query_id in create_filtered_dataset: {result.get('query_id', 'None')}")
                continue
        
        # 元のデータ構造を保持してフィルタリング済みデータセットを作成
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
                'applied_filters': []  # 適用されたフィルタの情報を後で追加
            }
        }
        
        return filtered_data
    
    def analyze_filtering_effects(self, summary_df: pd.DataFrame, filtered_query_ids: List[int]) -> Dict:
        """フィルタリング効果を分析"""
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
                       save_filtered: bool = True) -> Dict:
        """データセットフィルタリングを実行"""
        
        print(f"Starting filtering for dataset: {dataset_name}")
        
        # データセット名を保存
        self.dataset_name = dataset_name
        
        # 画像ハッシュマッピングの初期化
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
        
        # データ読み込み
        json_data, summary_df, detailed_df = self.load_dataset_files(dataset_name)
        print(f"Loaded: {len(json_data['results'])} results, {len(summary_df)} summary rows, {len(detailed_df)} detailed rows")
        
        # フィルタリング段階を記録
        filtering_stages = []
        
        # 初期状態 - 実際のquery_idを取得
        initial_count = len(json_data['results'])
        current_query_ids = []
        for result in json_data['results']:
            try:
                query_id = int(result['query_id'])
                current_query_ids.append(query_id)
            except (ValueError, TypeError):
                # query_idの変換に失敗した場合は警告を出力してスキップ
                print(f"Warning: Invalid query_id found: {result.get('query_id', 'None')}")
                continue
        
        print(f"Initial queries: {len(current_query_ids)} (valid query_ids extracted)")
        
        # 1. 成功フィルタ
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
        
        # 2. マルチターンフィルタ
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
        
        # 3. オリジナルクエリフィルタ
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
        
        # 4. ランクマージンフィルタ
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
        
        # 5. CLIP類似度フィルタ
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
        
        # 6. 画像重複フィルタ
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
        
        # フィルタリング済みデータセット作成
        filtered_data = self.create_filtered_dataset(json_data, current_query_ids)
        
        # 適用されたフィルタ情報を追加
        applied_filters = []
        for stage in filtering_stages:
            applied_filters.append({
                'name': stage['name'],
                'description': stage['description']
            })
        filtered_data['filtering_info']['applied_filters'] = applied_filters
        
        # 詳細レポート生成
        detailed_report = self.generate_detailed_report(
            dataset_name, json_data, filtering_stages, filtered_data
        )
        
        # レポート保存
        report_filename = f"filtering_report_{dataset_name}.json"
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(detailed_report, f, indent=2, ensure_ascii=False)
        print(f"Detailed report saved to: {report_filename}")
        
        if save_filtered:
            # フィルタリング済みデータを保存
            output_filename = f"filtered_multiturn_cir_{dataset_name}.json"
            with open(output_filename, 'w') as f:
                json.dump(filtered_data, f, indent=2)
            print(f"Filtered dataset saved to: {output_filename}")
        
        # 簡易統計表示
        self.print_simple_statistics(detailed_report)
        
        return filtered_data

    def print_simple_statistics(self, report: Dict):
        """簡易統計を表示"""
        dataset_name = report['dataset_info']['dataset_name']
        
        print("\n" + "="*80)
        print("マルチターンCIRフィルタリング詳細レポート")
        print("="*80)
        
        # データセット情報
        print(f"\n【{dataset_name}】")
        print("-" * 50)
        baseline = report['dataset_info']['baseline_stats']
        if 'total_queries_in_dataset' in baseline:
            print(f"データセット総クエリ数: {baseline['total_queries_in_dataset']:,}")
            if 'total_image_pairs' in baseline:
                print(f"画像ペア数: {baseline['total_image_pairs']:,}")
        
        experiment = report['dataset_info']['experiment_stats']
        print(f"実験処理クエリ数: {experiment['total_processed_queries']:,}")
        if isinstance(experiment['processing_coverage'], (int, float)):
            print(f"処理カバー率: {experiment['processing_coverage']:.1f}%")
        
        # データセットサンプル例を表示
        if 'displayed_sample_queries' in baseline and baseline['displayed_sample_queries']:
            print(f"\nデータセットサンプル例:")
            # 既にランダム選択済みのサンプルを使用
            random_samples = baseline['displayed_sample_queries']
            
            for i, sample in enumerate(random_samples, 1):
                if dataset_name.startswith('fashioniq'):
                    print(f"  例{i}: {sample.get('candidate', '')} → {sample.get('target', '')}")
                    print(f"       修正指示: \"{sample.get('caption', '')}\"")
                else:
                    print(f"  例{i}: {sample}")
        
        # 元データ分析
        print(f"\n【元データ分析】")
        original = report['original_analysis']
        print(f"総クエリ数: {original['total_queries']:,}")
        print(f"成功率: {original['summary']['success_rate']:.1f}%")
        
        print("\nターン別成功統計:")
        for turn_key, stats in original['turn_success_breakdown'].items():
            if turn_key != 'no_success':
                turn_num = turn_key.replace('turn_', '')
                print(f"  Turn {turn_num}で成功: {stats['count']:,} ({stats['percentage']:.1f}%)")
                
                # ダイアログ例を1つ表示
                if stats.get('examples') and len(stats['examples']) > 0:
                    example = stats['examples'][0]
                    print(f"    === ダイアログ例 (Query {example['query_id']}) ===")
                    print(f"    参照画像: {example['reference_image']} → 目標画像: {example['target_image']}")
                    
                    # ダイアログの各ターンを表示
                    for dialogue in example.get('dialogue', []):
                        turn_idx = dialogue['turn']
                        user_input = dialogue.get('user_input', '')
                        system_action = dialogue.get('system_action', '')
                        
                        print(f"    Turn {turn_idx}:")
                        print(f"      ユーザー: \"{user_input}\"")
                        print(f"      システム: {system_action}")
                        
                        if dialogue.get('is_success_turn', False):
                            print(f"      → 成功！最終ランク: {example.get('final_rank', 'N/A')}")
                            break
                    print()
        
        no_success = original['turn_success_breakdown'].get('no_success', {})
        print(f"  成功なし: {no_success.get('count', 0):,} ({no_success.get('percentage', 0):.1f}%)")
        if no_success.get('examples') and len(no_success.get('examples', [])) > 0:
            example = no_success['examples'][0]
            print(f"    === 失敗ダイアログ例 (Query {example['query_id']}) ===")
            print(f"    参照画像: {example['reference_image']} → 目標画像: {example['target_image']}")
            
            # 全ターンを表示
            for dialogue in example.get('dialogue', []):
                turn_idx = dialogue['turn']
                user_input = dialogue.get('user_input', '')
                system_action = dialogue.get('system_action', '')
                
                print(f"    Turn {turn_idx}:")
                print(f"      ユーザー: \"{user_input}\"")
                print(f"      システム: {system_action}")
            
            print(f"    → 最終的に成功せず (最終ランク: {example.get('final_rank', 'N/A')})")
            print()
        
        # フィルタリング分析
        print(f"\n【段階別フィルタリング】")
        for stage in report['filtering_analysis']['stages']:
            print(f"{stage['stage_name']}:")
            print(f"  入力: {stage['input_count']:,} → 出力: {stage['output_count']:,}")
            print(f"  除外: {stage['filtered_count']:,} ({stage['filtering_rate']:.1f}%)")
            print(f"  保持率: {stage['retention_rate']:.1f}%")
        
        # 最終結果
        print(f"\n【フィルタリング後分析】")
        final = report['final_analysis']
        print(f"総クエリ数: {final['total_queries']:,}")
        print(f"成功率: {final['summary']['success_rate']:.1f}%")
        
        print("\nターン別成功統計:")
        for turn_key, stats in final['turn_success_breakdown'].items():
            if turn_key != 'no_success':
                turn_num = turn_key.replace('turn_', '')
                print(f"  Turn {turn_num}で成功: {stats['count']:,} ({stats['percentage']:.1f}%)")
                
                # フィルタリング後のダイアログ例を1つ表示
                if stats.get('examples') and len(stats['examples']) > 0:
                    example = stats['examples'][0]
                    print(f"    === フィルタリング後ダイアログ例 (Query {example['query_id']}) ===")
                    print(f"    参照画像: {example['reference_image']} → 目標画像: {example['target_image']}")
                    
                    # 成功までのダイアログを表示
                    for dialogue in example.get('dialogue', []):
                        turn_idx = dialogue['turn']
                        user_input = dialogue.get('user_input', '')
                        system_action = dialogue.get('system_action', '')
                        
                        print(f"    Turn {turn_idx}:")
                        print(f"      ユーザー: \"{user_input}\"")
                        print(f"      システム: {system_action}")
                        
                        if dialogue.get('is_success_turn', False):
                            print(f"      → 成功！最終ランク: {example.get('final_rank', 'N/A')}")
                            break
                    print()
        
        no_success = final['turn_success_breakdown'].get('no_success', {})
        print(f"  成功なし: {no_success.get('count', 0):,} ({no_success.get('percentage', 0):.1f}%)")
        
        # 品質改善
        print(f"\n【品質改善】")
        comparison = report['comparison']['quality_improvement']
        print(f"元の成功率: {comparison['original_success_rate']:.1f}%")
        print(f"最終成功率: {comparison['final_success_rate']:.1f}%")
        print(f"成功率向上: {comparison['final_success_rate'] - comparison['original_success_rate']:+.1f}%")
        
        # 全体サマリー
        overall = report['filtering_analysis']['overall_summary']
        print(f"\n【全体サマリー】")
        print(f"全体フィルタリング率: {overall['overall_filtering_rate']:.1f}%")
        print(f"全体保持率: {overall['overall_retention_rate']:.1f}%")

    def analyze_turn_success(self, json_data: Dict, query_ids: List[int]) -> Dict:
        """ターン別の成功統計を分析"""
        
        # JSONの結果を辞書に変換
        results_dict = {}
        for result in json_data['results']:
            try:
                query_id = int(result['query_id'])
                results_dict[query_id] = result
            except (ValueError, TypeError):
                continue
        
        turn_success_stats = {}
        turn_candidates = {}  # ランダム選択用の候補を格納
        no_success_candidates = []
        
        for query_id in query_ids:
            if query_id not in results_dict:
                continue
            
            result = results_dict[query_id]
            
            # 結果レベルのsuccessフィールドを確認
            if result.get('success', False):
                # 成功したターンを特定
                success_turn = result.get('success_turn', 0)
                turn_key = f"turn_{success_turn}"
                
                if turn_key not in turn_success_stats:
                    turn_success_stats[turn_key] = []
                    turn_candidates[turn_key] = []
                
                turn_success_stats[turn_key].append(query_id)
                
                # 詳細なダイアログ例を作成
                dialogue_example = {
                    'query_id': query_id,
                    'reference_image': result.get('reference_image_id', ''),
                    'target_image': result.get('target_image_id', ''),
                    'original_query': result.get('original_query', ''),
                    'success_turn': success_turn,
                    'dialogue': []
                }
                
                # ターン別のダイアログを構築
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
                    
                    # ユーザーとシステムの対話形式で表現
                    if i == 0:
                        turn_info['user_input'] = result.get('original_query', '')
                        turn_info['system_action'] = f"画像検索実行 → ランク{turn_data.get('best_gt_rank', 'N/A')}"
                    else:
                        turn_info['user_input'] = turn_data.get('relative_caption', '')
                        turn_info['system_action'] = f"修正検索実行 → ランク{turn_data.get('best_gt_rank', 'N/A')}"
                    
                    dialogue_example['dialogue'].append(turn_info)
                
                # 最終結果情報
                final_turn = turns[success_turn] if success_turn < len(turns) else {}
                dialogue_example['final_rank'] = final_turn.get('best_gt_rank', 'N/A')
                dialogue_example['final_query'] = final_turn.get('query_text', '')
                
                turn_candidates[turn_key].append(dialogue_example)
            else:
                # 成功しなかったクエリ
                no_success_example = {
                    'query_id': query_id,
                    'reference_image': result.get('reference_image_id', ''),
                    'target_image': result.get('target_image_id', ''),
                    'original_query': result.get('original_query', ''),
                    'final_status': 'No success achieved',
                    'dialogue': []
                }
                
                # 全ターンのダイアログを記録
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
                        turn_info['system_action'] = f"画像検索実行 → ランク{turn_data.get('best_gt_rank', 'N/A')}"
                    else:
                        turn_info['user_input'] = turn_data.get('relative_caption', '')
                        turn_info['system_action'] = f"修正検索実行 → ランク{turn_data.get('best_gt_rank', 'N/A')}"
                    
                    no_success_example['dialogue'].append(turn_info)
                
                # 最終ランク情報
                if turns:
                    final_turn = turns[-1]
                    no_success_example['final_rank'] = final_turn.get('best_gt_rank', 'N/A')
                    no_success_example['final_query'] = final_turn.get('query_text', '')
                
                no_success_candidates.append(no_success_example)
        
        # 統計をまとめる（ランダムに3つ選択）
        turn_stats = {}
        for turn_key in sorted(turn_success_stats.keys()):
            # ランダムに3つの例を選択
            candidates = turn_candidates[turn_key]
            random_examples = random.sample(candidates, min(3, len(candidates))) if candidates else []
            
            turn_stats[turn_key] = {
                'count': len(turn_success_stats[turn_key]),
                'percentage': len(turn_success_stats[turn_key]) / len(query_ids) * 100 if query_ids else 0,
                'examples': random_examples
            }
        
        # 成功しなかったクエリの例もランダムに3つ選択
        random_no_success = random.sample(no_success_candidates, min(3, len(no_success_candidates))) if no_success_candidates else []
        
        turn_stats['no_success'] = {
            'count': len(no_success_candidates),
            'percentage': len(no_success_candidates) / len(query_ids) * 100 if query_ids else 0,
            'examples': random_no_success
        }
        
        return turn_stats

    def get_dataset_baseline_stats(self, dataset_name: str) -> Dict:
        """データセットの基本統計を取得"""
        stats = {}
        
        # 実際に処理されたデータを確認
        json_file = f"multiturn_cir_results_{dataset_name}.json"
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                actual_processed = len(json_data.get('results', []))
                stats['actual_processed_queries'] = actual_processed
                
                # data_completenessから推定可能クエリ数を取得
                completeness = json_data.get('data_completeness', {})
                estimated_processable = completeness.get('estimated_processable_queries', actual_processed)
                stats['estimated_processable_queries'] = estimated_processable
        
        # データセット別の統計
        if dataset_name.startswith('fashioniq'):
            category = dataset_name.split('_')[1]  # dress, shirt, toptee
            split = 'val' if 'val' in dataset_name else 'train'
            
            annotation_file = f'fashion-iq/captions/cap.{category}.{split}.json'
            if os.path.exists(annotation_file):
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                    
                    # Fashion-IQはrelative_captionsの配列形式
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
                    
                    # CIRRの場合、元のアノテーションファイルの総数を保持
                    stats['total_queries_in_dataset'] = len(data)
                    stats['annotation_file'] = annotation_file
                    
                    # サンプル例は実際に処理されたデータからランダム選択
                    if 'actual_processed_queries' in stats and stats['actual_processed_queries'] > 0:
                        if os.path.exists(json_file):
                            with open(json_file, 'r') as f:
                                json_data = json.load(f)
                                results = json_data.get('results', [])
                                if results:
                                    # 実際の結果からランダムサンプルを選択
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
                    
                    # 全てのサンプル例を保存
                    stats['sample_queries'] = data
        
        return stats

    def generate_detailed_report(self, dataset_name: str, original_data: Dict, 
                               filtering_stages: List[Dict], final_data: Dict) -> Dict:
        """詳細な分析レポートを生成"""
        
        # 基本統計を取得
        baseline_stats = self.get_dataset_baseline_stats(dataset_name)
        
        # サンプル例をランダム選択（print表示と同じロジック）
        if 'sample_queries' in baseline_stats and baseline_stats['sample_queries']:
            sample_queries = baseline_stats['sample_queries']
            random_samples = random.sample(sample_queries, min(2, len(sample_queries))) if sample_queries else []
            baseline_stats['displayed_sample_queries'] = random_samples
            # 全てのサンプルは削除してファイルサイズを抑制
            del baseline_stats['sample_queries']
        
        # 元データの統計
        all_query_ids = []
        for result in original_data['results']:
            try:
                query_id = int(result['query_id'])
                all_query_ids.append(query_id)
            except (ValueError, TypeError):
                continue
        
        original_turn_stats = self.analyze_turn_success(original_data, all_query_ids)
        
        # フィルタリング後の統計  
        final_query_ids = []
        for result in final_data['results']:
            try:
                query_id = int(result['query_id'])
                final_query_ids.append(query_id)
            except (ValueError, TypeError):
                continue
        
        final_turn_stats = self.analyze_turn_success(original_data, final_query_ids)
        
        # 段階別フィルタリング統計
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
            
            # 具体例があれば追加
            if 'examples' in stage:
                stage_info['examples'] = stage['examples']
                
            stage_stats.append(stage_info)
        
        # 実際に処理されたクエリ数を使用
        actual_processed = baseline_stats.get('actual_processed_queries', len(all_query_ids))
        
        # 処理カバー率を元のデータセット総クエリ数から計算
        total_queries_in_dataset = baseline_stats.get('total_queries_in_dataset', actual_processed)
        processing_coverage = len(original_data['results']) / total_queries_in_dataset * 100 if total_queries_in_dataset > 0 else 100.0
        
        # 詳細レポート作成
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
    
    args = parser.parse_args()
    
    # ランダムシードを設定
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # フィルタ初期化
    filter_system = OptimizedMultiTurnFilter(
        similarity_threshold=args.similarity_threshold,
        rank_margin=args.rank_margin
    )
    
    # 各データセットをフィルタリング
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
                save_filtered=not args.analyze_only
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
    
    print(f"\nProcessing completed for {len(results)} datasets")

if __name__ == "__main__":
    main() 