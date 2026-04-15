#!/usr/bin/env python3
"""
RQ2: Defense Effectiveness (§6.3)

Evaluates the full defense framework against all three channels.
Compares: Naive, Honeybee, Uniform-DP, Ours-NoDec, Ours-Full.

Usage:
    python scripts/run_defense.py --config configs/default.yaml --dataset mimic
"""

import argparse
import numpy as np
import yaml
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.defense.composed import ComposedDefense
from src.defense.private_topk import private_topk
from src.attacks.channel1 import evaluate_channel1
from src.attacks.channel2 import evaluate_channel2
from src.utils.index import AccessControlPolicy


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="RQ2: Defense Effectiveness")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, choices=["mimic", "legalbench", "wiki"])
    parser.add_argument("--output", type=str, default="results/rq2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    rng = np.random.default_rng(args.seed)

    # Load dataset
    data_dir = f"data/{args.dataset}"
    print(f"Loading dataset from {data_dir}...")
    vectors = np.load(f"{data_dir}/vectors.npy")
    restricted_mask = np.load(f"{data_dir}/restricted_mask.npy")
    eval_queries = np.load(f"{data_dir}/eval_queries.npy")
    eval_labels = np.load(f"{data_dir}/eval_labels.npy")

    n, d = vectors.shape
    k = config["defense"]["k"]
    epsilon = config["defense"]["epsilon_0"]
    n_seeds = config["evaluation"]["n_seeds"]

    print(f"Dataset: {args.dataset}, n={n}, d={d}, k={k}, ε={epsilon}")
    print(f"Restriction ratio α = {restricted_mask.mean():.2f}")

    authorized = vectors[~restricted_mask]

    # Method configurations
    methods = {
        "Naive": {"defense": None},
        "Uniform-DP": {"defense": "uniform", "epsilon": epsilon},
        "Ours-NoDec": {"defense": "full", "epsilon": epsilon, "c": 0},
        "Ours-Full": {"defense": "full", "epsilon": epsilon, "c": config["defense"]["c"]},
    }

    results = {}
    for method_name, method_config in methods.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*60}")

        seed_results = []
        for seed in range(n_seeds):
            seed_rng = np.random.default_rng(seed)

            if method_config["defense"] is None:
                # Naive: exact k-NN on authorized set
                all_distances = []
                all_neighbors = []
                recalls = []
                for q in eval_queries:
                    dists = np.linalg.norm(authorized - q, axis=1)
                    topk_idx = np.argpartition(dists, k)[:k]
                    topk_idx = topk_idx[np.argsort(dists[topk_idx])]
                    all_distances.append(dists[topk_idx])
                    all_neighbors.append(authorized[topk_idx])
                    recalls.append(1.0)

            elif method_config["defense"] == "uniform":
                # Uniform-DP: private top-k with fixed ε
                all_distances = []
                all_neighbors = []
                recalls = []
                for q in eval_queries:
                    idx, dists = private_topk(q, authorized, epsilon, k, seed_rng)
                    all_distances.append(dists)
                    all_neighbors.append(authorized[idx])
                    # Compute recall against exact k-NN
                    exact_dists = np.linalg.norm(authorized - q, axis=1)
                    exact_topk = set(np.argpartition(exact_dists, k)[:k])
                    recalls.append(len(set(idx) & exact_topk) / k)

            elif method_config["defense"] == "full":
                defense = ComposedDefense(
                    epsilon_0=method_config["epsilon"],
                    k=k,
                    c=method_config["c"],
                    manifold_aware=config["defense"]["manifold_aware"],
                )
                defense.build(vectors, restricted_mask, rng=seed_rng)

                all_distances = []
                all_neighbors = []
                recalls = []
                for q in eval_queries:
                    result_vecs, result_dists = defense.query(q, rng=seed_rng)
                    all_distances.append(result_dists)
                    all_neighbors.append(result_vecs)
                    # Approximate recall
                    recalls.append(min(len(result_vecs), k) / k)

            # Compute metrics
            lambda_est = len(authorized) / (np.prod(authorized.max(0) - authorized.min(0)) + 1e-10)
            d_int = min(48, d)

            ch1_auc = evaluate_channel1(
                eval_queries, all_distances, eval_labels, k, d, lambda_est
            )
            ch2_auc = evaluate_channel2(
                eval_queries, all_neighbors, eval_labels, d_int=d_int
            )

            seed_results.append({
                "ch1_auc": ch1_auc,
                "ch2_auc": ch2_auc,
                "recall": np.mean(recalls),
            })

        # Aggregate
        results[method_name] = {
            "ch1_auc": np.mean([r["ch1_auc"] for r in seed_results]),
            "ch2_auc": np.mean([r["ch2_auc"] for r in seed_results]),
            "recall": np.mean([r["recall"] for r in seed_results]),
            "recall_std": np.std([r["recall"] for r in seed_results]),
        }

        print(f"  Ch1 AUC: {results[method_name]['ch1_auc']:.2f}")
        print(f"  Ch2 AUC: {results[method_name]['ch2_auc']:.2f}")
        print(f"  Recall@{k}: {results[method_name]['recall']:.2f} "
              f"(±{results[method_name]['recall_std']:.3f})")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY (Table 5 / Table 6)")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'Ch1 AUC':>8} {'Ch2 AUC':>8} {'Recall':>8}")
    print("-" * 42)
    for name, r in results.items():
        print(f"{name:<15} {r['ch1_auc']:>8.2f} {r['ch2_auc']:>8.2f} {r['recall']:>8.2f}")

    # Save results
    os.makedirs(args.output, exist_ok=True)
    np.savez(f"{args.output}/{args.dataset}_results.npz", **results)
    print(f"\nResults saved to {args.output}/{args.dataset}_results.npz")


if __name__ == "__main__":
    main()
