#!/usr/bin/env python3
"""
Convert internal MTCIR dataset format to public CIRCLED format.

Internal format:
- session.reference_image_id -> I_1
- session.original_query -> T_1
- turns[k].selected_image -> I_{k+1} (for k >= 1)
- turns[k].relative_caption -> T_{k+1} (for k >= 1)

Public format (1-indexed):
- turns[0].turn = 1, reference_image_id = I_1, relative_caption = T_1
- turns[k].turn = k+1, reference_image_id = I_{k+1}, relative_caption = T_{k+1}
"""

import json
import os
import argparse
from pathlib import Path
from typing import Any


def convert_session(session: dict, subset: str, session_idx: int) -> dict:
    """Convert a single session to public format."""

    # Generate session_id
    session_id = f"{subset}_{session_idx:04d}"

    # Build turns list (1-indexed)
    public_turns = []

    # Turn 1: from session level
    public_turns.append({
        "turn": 1,
        "reference_image_id": session["reference_image_id"],
        "relative_caption": session["original_query"]
    })

    # Turn 2+: from turns array (skip turn 0, use turn 1+)
    original_turns = session.get("turns", [])
    for orig_turn in original_turns:
        turn_idx = orig_turn.get("turn", 0)
        if turn_idx >= 1:  # Skip turn 0 (no selected_image)
            selected_image = orig_turn.get("selected_image")
            relative_caption = orig_turn.get("relative_caption")

            if selected_image and relative_caption:
                public_turns.append({
                    "turn": turn_idx + 1,  # Convert to 1-indexed
                    "reference_image_id": selected_image,
                    "relative_caption": relative_caption
                })

    return {
        "session_id": session_id,
        "subset": subset,
        "ground_truth_ids": session.get("ground_truth_ids", []),
        "num_turns": len(public_turns),
        "turns": public_turns
    }


def convert_dataset_file(input_path: str, output_path: str, subset: str) -> dict:
    """Convert a single dataset file."""

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get("results", [])

    converted_sessions = []
    for idx, session in enumerate(results):
        converted = convert_session(session, subset, idx)
        converted_sessions.append(converted)

    # Save converted data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_sessions, f, indent=2, ensure_ascii=False)

    return {
        "subset": subset,
        "input_sessions": len(results),
        "output_sessions": len(converted_sessions),
        "output_path": output_path
    }


def main():
    parser = argparse.ArgumentParser(description='Convert internal MTCIR format to public CIRCLED format')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Input directory containing filtered_multiturn_cir_*.json files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for public format JSON files')
    parser.add_argument('--subsets', nargs='+', default=None,
                       help='Specific subsets to convert (default: all)')

    args = parser.parse_args()

    # Define subset mappings
    all_subsets = {
        "fashioniq_dress_train": "filtered_multiturn_cir_fashioniq_dress_train.json",
        "fashioniq_dress_val": "filtered_multiturn_cir_fashioniq_dress_val.json",
        "fashioniq_shirt_train": "filtered_multiturn_cir_fashioniq_shirt_train.json",
        "fashioniq_shirt_val": "filtered_multiturn_cir_fashioniq_shirt_val.json",
        "fashioniq_toptee_train": "filtered_multiturn_cir_fashioniq_toptee_train.json",
        "fashioniq_toptee_val": "filtered_multiturn_cir_fashioniq_toptee_val.json",
        "cirr_train": "filtered_multiturn_cir_cirr_train.json",
        "cirr_val": "filtered_multiturn_cir_cirr_val.json",
        "circo_val": "filtered_multiturn_cir_circo.json",
    }

    # Filter subsets if specified
    if args.subsets:
        subsets = {k: v for k, v in all_subsets.items() if k in args.subsets}
    else:
        subsets = all_subsets

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = []
    for subset, filename in subsets.items():
        input_path = input_dir / filename
        output_path = output_dir / f"{subset}.json"

        if input_path.exists():
            print(f"Converting {subset}...")
            stat = convert_dataset_file(str(input_path), str(output_path), subset)
            stats.append(stat)
            print(f"  {stat['input_sessions']} -> {stat['output_sessions']} sessions")
        else:
            print(f"Warning: {input_path} not found, skipping.")

    # Print summary
    print("\n" + "="*50)
    print("Conversion Summary")
    print("="*50)
    total_sessions = sum(s['output_sessions'] for s in stats)
    for stat in stats:
        print(f"  {stat['subset']}: {stat['output_sessions']} sessions")
    print(f"\nTotal: {total_sessions} sessions")


if __name__ == "__main__":
    main()
