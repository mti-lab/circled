#!/usr/bin/env python3
"""
キャプションから冗長な表現を除去するスクリプト
GPT-4oが生成したキャプションの「The image features」などの定型句を除去
"""

import json
import re
import argparse
from pathlib import Path

class CaptionCleaner:
    """キャプション清浄化クラス"""
    
    def __init__(self):
        # 除去する冗長な表現のパターン
        self.redundant_patterns = [
            r'^The image features?\s+',
            r'^The image shows?\s+',
            r'^The image depicts?\s+',
            r'^The image captures?\s+',
            r'^The image displays?\s+',
            r'^The image presents?\s+',
            r'^The image contains?\s+',
            r'^The image includes?\s+',
            r'^This image features?\s+',
            r'^This image shows?\s+',
            r'^This image depicts?\s+',
            r'^This image captures?\s+',
            r'^This image displays?\s+',
            r'^This image presents?\s+',
            r'^This image contains?\s+',
            r'^This image includes?\s+',
            r'^In the image,?\s+',
            r'^In this image,?\s+',
            r'^The photo features?\s+',
            r'^The photo shows?\s+',
            r'^The picture features?\s+',
            r'^The picture shows?\s+',
            r'^The image showcases?\s+',
        ]
        
        # コンパイル済み正規表現パターン
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.redundant_patterns]
    
    def clean_caption(self, caption: str) -> str:
        """単一キャプションをクリーンアップ"""
        if not caption or not isinstance(caption, str):
            return caption
        
        cleaned = caption.strip()
        
        # 冗長な表現を除去
        for pattern in self.compiled_patterns:
            cleaned = pattern.sub('', cleaned)
        
        # 先頭を大文字に
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
        
        return cleaned.strip()
    
    def clean_captions_dict(self, captions_dict: dict) -> dict:
        """キャプション辞書全体をクリーンアップ"""
        cleaned_dict = {}
        total_count = 0
        cleaned_count = 0
        
        for image_path, caption in captions_dict.items():
            original_caption = caption
            cleaned_caption = self.clean_caption(caption)
            cleaned_dict[image_path] = cleaned_caption
            
            total_count += 1
            if original_caption != cleaned_caption:
                cleaned_count += 1
        
        print(f"クリーンアップ統計:")
        print(f"  総キャプション数: {total_count:,}")
        print(f"  変更されたキャプション数: {cleaned_count:,}")
        print(f"  変更率: {cleaned_count/total_count*100:.1f}%")
        
        return cleaned_dict
    
    def show_examples(self, original_dict: dict, cleaned_dict: dict, num_examples: int = 5):
        """クリーンアップの例を表示"""
        print(f"\nクリーンアップ例 (最初の{num_examples}件):")
        print("=" * 80)
        
        count = 0
        for image_path, original_caption in original_dict.items():
            if count >= num_examples:
                break
                
            cleaned_caption = cleaned_dict[image_path]
            if original_caption != cleaned_caption:
                print(f"画像: {Path(image_path).name}")
                print(f"元: {original_caption}")
                print(f"後: {cleaned_caption}")
                print("-" * 40)
                count += 1

def main():
    parser = argparse.ArgumentParser(description='キャプションから冗長な表現を除去')
    parser.add_argument('input_file', help='入力JSONファイル')
    parser.add_argument('--output', help='出力JSONファイル (デフォルト: {input}_cleaned.json)')
    parser.add_argument('--examples', type=int, default=5, help='表示する例の数 (デフォルト: 5)')
    parser.add_argument('--dry-run', action='store_true', help='実際の保存は行わず、例のみ表示')
    
    args = parser.parse_args()
    
    # 入力ファイル確認
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {args.input_file}")
        return 1
    
    # 出力ファイル名設定
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
    
    print(f"入力ファイル: {input_path}")
    print(f"出力ファイル: {output_path}")
    
    try:
        # JSONファイル読み込み
        print("JSONファイルを読み込み中...")
        with open(input_path, 'r', encoding='utf-8') as f:
            original_captions = json.load(f)
        
        print(f"読み込み完了: {len(original_captions):,}件のキャプション")
        
        # キャプションクリーンアップ
        cleaner = CaptionCleaner()
        cleaned_captions = cleaner.clean_captions_dict(original_captions)
        
        # 例を表示
        if args.examples > 0:
            cleaner.show_examples(original_captions, cleaned_captions, args.examples)
        
        # ファイル保存
        if not args.dry_run:
            print(f"\nクリーンアップされたキャプションを保存中: {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_captions, f, ensure_ascii=False, indent=2)
            
            print(f"保存完了: {output_path}")
            
            # ファイルサイズ比較
            original_size = input_path.stat().st_size
            cleaned_size = output_path.stat().st_size
            print(f"ファイルサイズ: {original_size:,} → {cleaned_size:,} bytes")
            print(f"サイズ削減: {(original_size-cleaned_size)/original_size*100:.1f}%")
        else:
            print("\n--dry-run モードのため、ファイルは保存されませんでした")
        
    except Exception as e:
        print(f"エラー: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 