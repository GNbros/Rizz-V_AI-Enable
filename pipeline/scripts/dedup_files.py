#!/usr/bin/env python3
"""
dedup_files.py – Detect near-duplicate source files to prevent data leakage.

Input:  Directory of normalized .S files
Output: JSON report mapping filename -> cluster_id

Method:
1. Tokenize assembly (ignoring comments/whitespace)
2. Compute MinHash signature for each file
3. Cluster files with Jaccard similarity > threshold
4. Output cluster mapping

CLI usage:
    python scripts/dedup_files.py --config configs/dataset.yaml --output data/dedup_report.json
"""

import argparse
import hashlib
import json
import re
import struct
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ── MinHash Config ──
NUM_PERM = 64
SEED = 42

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def tokenize(text: str) -> Set[str]:
    """Simple tokenization: lowercase alphanumeric sequences."""
    # Split by non-alphanumeric, filter empty, lowercase
    tokens = re.split(r"[^a-zA-Z0-9_]+", text.lower())
    return {t for t in tokens if t}

def compute_signature(tokens: Set[str], num_perm: int) -> List[int]:
    """Compute MinHash signature using simple hashing."""
    # We use md5 with different seeds (by appending salt) as hash functions
    # efficient enough for this scale (3.4k files)
    
    hashes = [float('inf')] * num_perm
    
    for token in tokens:
        # Base hash of the token
        base = token.encode("utf-8")
        
        for i in range(num_perm):
            # Salt with permutation index
            h = hashlib.md5(base + struct.pack("<I", i)).digest()
            # Interpret as int
            val = struct.unpack("<I", h[:4])[0]
            if val < hashes[i]:
                hashes[i] = val
                
    return hashes

def compute_jaccard(sig1: List[int], sig2: List[int]) -> float:
    match = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return match / len(sig1)

def find_clusters(signatures: Dict[str, List[int]], threshold: float) -> Dict[str, int]:
    """Greedy clustering of files."""
    files = sorted(signatures.keys())
    clusters: Dict[str, int] = {}
    cluster_count = 0
    
    # Pre-compute active files to avoid dict lookups
    # O(N^2) naive comparison is fine for N=3400 (approx 5M pairs, fast enough in python)
    
    # Optimisation: use LSH if N grows, but for <5k files, brute force is acceptable (~seconds)
    
    assigned = set()
    
    for i, f1 in enumerate(files):
        if f1 in assigned:
            continue
            
        # Start new cluster
        cid = cluster_count
        cluster_count += 1
        clusters[f1] = cid
        assigned.add(f1)
        
        # Determine signature
        sig1 = signatures[f1]
        
        # Check all subsequent unassigned files
        for j in range(i + 1, len(files)):
            f2 = files[j]
            if f2 in assigned:
                continue
                
            sig2 = signatures[f2]
            sim = compute_jaccard(sig1, sig2)
            
            if sim >= threshold:
                clusters[f2] = cid
                assigned.add(f2)
                
    return clusters

def main():
    parser = argparse.ArgumentParser(description="Near-duplicate detection for .S files")
    parser.add_argument("--config", required=True, help="Path to dataset.yaml")
    parser.add_argument("--output", required=True, help="Path to write JSON report")
    parser.add_argument("--threshold", type=float, default=0.8, help="Jaccard similarity threshold (0.0-1.0)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    project_root = Path(args.config).resolve().parent.parent
    norm_dir = project_root / cfg["paths"]["normalized_dir"]
    
    if not norm_dir.exists():
        print(f"Error: {norm_dir} not found.", file=sys.stderr)
        sys.exit(1)
        
    print(f"[dedup] Scanning {norm_dir}...")
    files = sorted(list(norm_dir.glob("*.S")))
    if not files:
        print("No .S files found.", file=sys.stderr)
        sys.exit(1)
        
    # Compute signatures
    signatures = {}
    for fpath in files:
        text = fpath.read_text(encoding="utf-8")
        tokens = tokenize(text)
        if not tokens:
            continue
        signatures[fpath.name] = compute_signature(tokens, NUM_PERM)
        
    print(f"[dedup] Computed signatures for {len(signatures)} files. Clustering...")
    
    # Cluster
    file_clusters = find_clusters(signatures, args.threshold)
    
    # Invert to see size distribution
    cluster_to_files = {}
    for fname, cid in file_clusters.items():
        cluster_to_files.setdefault(cid, []).append(fname)
        
    multi_file_clusters = {c: fl for c, fl in cluster_to_files.items() if len(fl) > 1}
    print(f"[dedup] Found {len(multi_file_clusters)} clusters with duplicates (threshold={args.threshold}).")
    
    # Write report
    report = {
        "start_time": "2026-02-18", # Placeholder
        "config": {"threshold": args.threshold, "num_perm": NUM_PERM},
        "file_clusters": file_clusters,
        "cluster_stats": {
            "total_files": len(signatures),
            "unique_clusters": len(cluster_to_files),
            "duplicate_files": sum(len(fl) for fl in multi_file_clusters.values())
        }
    }
    
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"[dedup] Report written to {out_path}")

if __name__ == "__main__":
    main()
