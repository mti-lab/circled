#!/usr/bin/env python3
"""
Script to remove redundant expressions from captions.
Removes boilerplate phrases like "The image features" from GPT-4o generated captions.
"""

import json
import re
import argparse
from pathlib import Path

class CaptionCleaner:
    """Caption cleaning class"""
    
    def __init__(self):
        # Patterns of redundant expressions to remove
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
        
        # Pre-compiled regex patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.redundant_patterns]
    
    def clean_caption(self, caption: str) -> str:
        """Clean up a single caption"""
        if not caption or not isinstance(caption, str):
            return caption
        
        cleaned = caption.strip()
        
        # Remove redundant expressions
        for pattern in self.compiled_patterns:
            cleaned = pattern.sub('', cleaned)
        
        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
        
        return cleaned.strip()
    
    def clean_captions_dict(self, captions_dict: dict) -> dict:
        """Clean up the entire captions dictionary"""
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
        
        print(f"Cleanup statistics:")
        print(f"  Total captions: {total_count:,}")
        print(f"  Modified captions: {cleaned_count:,}")
        print(f"  Modification rate: {cleaned_count/total_count*100:.1f}%")
        
        return cleaned_dict
    
    def show_examples(self, original_dict: dict, cleaned_dict: dict, num_examples: int = 5):
        """Display cleanup examples"""
        print(f"\nCleanup examples (first {num_examples}):")
        print("=" * 80)
        
        count = 0
        for image_path, original_caption in original_dict.items():
            if count >= num_examples:
                break
                
            cleaned_caption = cleaned_dict[image_path]
            if original_caption != cleaned_caption:
                print(f"Image: {Path(image_path).name}")
                print(f"Before: {original_caption}")
                print(f"After: {cleaned_caption}")
                print("-" * 40)
                count += 1

def main():
    parser = argparse.ArgumentParser(description='Remove redundant expressions from captions')
    parser.add_argument('input_file', help='Input JSON file')
    parser.add_argument('--output', help='Output JSON file (default: {input}_cleaned.json)')
    parser.add_argument('--examples', type=int, default=5, help='Number of examples to display (default: 5)')
    parser.add_argument('--dry-run', action='store_true', help='Only display examples without saving')
    
    args = parser.parse_args()
    
    # Check input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1
    
    # Set output file name
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
    
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    
    try:
        # Load JSON file
        print("Loading JSON file...")
        with open(input_path, 'r', encoding='utf-8') as f:
            original_captions = json.load(f)
        
        print(f"Loaded: {len(original_captions):,} captions")
        
        # Clean up captions
        cleaner = CaptionCleaner()
        cleaned_captions = cleaner.clean_captions_dict(original_captions)
        
        # Display examples
        if args.examples > 0:
            cleaner.show_examples(original_captions, cleaned_captions, args.examples)
        
        # Save file
        if not args.dry_run:
            print(f"\nSaving cleaned captions: {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_captions, f, ensure_ascii=False, indent=2)
            
            print(f"Saved: {output_path}")
            
            # Compare file sizes
            original_size = input_path.stat().st_size
            cleaned_size = output_path.stat().st_size
            print(f"File size: {original_size:,} → {cleaned_size:,} bytes")
            print(f"Size reduction: {(original_size-cleaned_size)/original_size*100:.1f}%")
        else:
            print("\n--dry-run mode: file was not saved")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 