#!/usr/bin/env python3
"""
Experiment runner. See run_defense.py for the full implementation pattern.
Usage: python scripts/SCRIPT_NAME.py --config configs/default.yaml --dataset mimic
"""
import argparse
import numpy as np
import yaml
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, choices=["mimic", "legalbench", "wiki"])
    parser.add_argument("--output", type=str, default="results/")
    parser.add_argument("--epsilon", type=float, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Running experiment with config: {args.config}, dataset: {args.dataset}")
    print("Implementation follows the same pattern as run_defense.py")
    print("See the paper §6 for detailed methodology.")

if __name__ == "__main__":
    main()
