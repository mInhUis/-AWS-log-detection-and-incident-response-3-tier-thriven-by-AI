"""
src/detector/run_eval.py
─────────────────────────────────────────────────────────────────────────────
Pipeline position: Tier 2 Final Evaluation.
This is a consolidated, bulletproof evaluation script. It guarantees timeline
isolation and safe OOV masking without relying on external file states.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Final

import torch

# Ensure project root is on sys.path
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from deeplog import DeepLogModel
from detect import evaluate_predictions
from src.preprocessing.normalize import normalize_event
from drain_parser import DrainParser

# ── Configurable Paths ─────────────────────────────────────────────────────────
MODEL_PATH: Final[Path] = _PROJECT_ROOT / "models1" / "deeplog" / "deeplog_best.pt"
# The KNOWN GOOD normal data
VAL_KEYS_PATH: Final[Path] = _PROJECT_ROOT / "data" / "processeded1" / "val_log_keys.json"
# The mixed data to extract attacks from
MIXED_KEYS_PATH: Final[Path] = _PROJECT_ROOT / "data" / "processeded1" / "mixed_val_log_keys.json"
MIXED_RAW_PATH: Final[Path] = _PROJECT_ROOT / "data" / "processeded1" / "synth_val_mixed.jsonl"

def detect_anomalies_safe(model, sequences, window_size, top_k, device, model_num_keys):
    """Embedded inference engine guaranteeing safe OOV masking."""
    model.eval()
    all_flags = []
    
    with torch.no_grad():
        for seq in sequences:
            seq_len = len(seq)
            flags = [False] * seq_len
            
            if seq_len <= window_size:
                all_flags.append(flags)
                continue
            
            for i in range(window_size, seq_len):
                window = seq[i - window_size : i]
                true_key = seq[i]

                # 1. Current Event Check
                if true_key >= model_num_keys:
                    flags[i] = True
                    continue
                    
                # 2. Safe Past Window Masking
                safe_window = [k if k < model_num_keys else 0 for k in window]
                window_tensor = torch.tensor([safe_window], dtype=torch.long, device=device)
                
                logits, _ = model(window_tensor)
                _, top_indices = torch.topk(logits, k=top_k, dim=-1)
                top_set = set(top_indices.squeeze(0).tolist())
                
                if true_key not in top_set:
                    flags[i] = True
                    
            all_flags.append(flags)
            
    return all_flags

def main() -> None:
    print(f"\n{'='*72}")
    print(f" Tier 2: Final F1 Sweep (The Frankenstein Merge)")
    print(f"{'='*72}\n")

    # 1. Load the KNOWN GOOD normal sequence
    if not VAL_KEYS_PATH.exists():
        print(f"ERROR: Could not find {VAL_KEYS_PATH}")
        sys.exit(1)
        
    with open(VAL_KEYS_PATH, "r", encoding="utf-8") as f:
        normal_seq = json.load(f)["sequences"][0]

    # 2. Extract the attack sequence from the mixed data
    if not MIXED_KEYS_PATH.exists() or not MIXED_RAW_PATH.exists():
        print(f"ERROR: Could not find mixed files.")
        sys.exit(1)
        
    with open(MIXED_KEYS_PATH, "r", encoding="utf-8") as f:
        mixed_keys = json.load(f)["sequences"][0]
        
    all_events = []
    with open(MIXED_RAW_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                all_events.append(json.loads(line))
                
    all_events.sort(key=lambda e: e.get("eventTime", ""))
    mixed_labels = [e.get("_source", "synth_normal") == "flaws_cloud" for e in all_events]
    
    attack_seq = [k for k, is_atk in zip(mixed_keys, mixed_labels) if is_atk]

    # 3. Combine into pure evaluation timelines
    sequences = [normal_seq, attack_seq]
    ground_truth = [[False] * len(normal_seq), [True] * len(attack_seq)]

    print(f"[Data] Loaded pure Normal timeline: {len(normal_seq)} events (from val_log_keys).")
    print(f"[Data] Extracted pure Attack timeline: {len(attack_seq)} events (from mixed).\n")

    # 4. Load the PyTorch Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)

    model = DeepLogModel(
        num_keys=ckpt["num_keys"],
        embedding_dim=ckpt["embedding_dim"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    
    window_size = ckpt["window_size"]
    model_num_keys = ckpt["num_keys"]
    print(f"[Model] Loaded DeepLogModel (vocab={model_num_keys}, window={window_size}) on {device}.\n")

    # 5. Sweep top_k
    top_k_candidates = [3, 5, 7, 9, 11, 13, 15]

    print(f"{'top_k':<6} | {'Precision':<9} | {'Recall':<9} | {'F1 Score':<9} | {'TP':<6} | {'FP':<6} | {'FN':<6}")
    print("-" * 72)

    for k in top_k_candidates:
        predicted_flags = detect_anomalies_safe(
            model=model, 
            sequences=sequences, 
            window_size=window_size, 
            top_k=k, 
            device=device,
            model_num_keys=model_num_keys
        )
        
        metrics = evaluate_predictions(predicted_flags, ground_truth)
        
        print(f"{k:<6} | {metrics['precision']:<9.3f} | {metrics['recall']:<9.3f} | {metrics['f1']:<9.3f} | {metrics['tp']:<6} | {metrics['fp']:<6} | {metrics['fn']:<6}")

if __name__ == "__main__":
    main()