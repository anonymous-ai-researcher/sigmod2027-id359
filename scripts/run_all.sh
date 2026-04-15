#!/bin/bash
# Full reproduction pipeline for Phantom Neighbors (SIGMOD 2027)
# Estimated runtime: ~8 hours on A100 + EPYC
#
# Usage: bash scripts/run_all.sh

set -e

echo "=============================================="
echo "Phantom Neighbors - Full Reproduction Pipeline"
echo "=============================================="
echo ""

DATASETS="mimic legalbench wiki"
CONFIG="configs/default.yaml"
RESULTS_DIR="results"

mkdir -p $RESULTS_DIR

# Step 1: Data Preprocessing
echo "[Step 1/6] Data Preprocessing"
echo "  Note: Requires raw data files. See README.md for download instructions."
echo "  Skipping if processed data already exists."
for ds in $DATASETS; do
    if [ ! -f "data/$ds/vectors.npy" ]; then
        echo "  Processing $ds..."
        python scripts/preprocess_${ds}.py --output data/$ds/ --config $CONFIG
    else
        echo "  $ds already preprocessed, skipping."
    fi
done

# Step 2: RQ1 - Leakage Severity
echo ""
echo "[Step 2/6] RQ1: Leakage Severity without Defense"
for ds in $DATASETS; do
    echo "  Dataset: $ds"
    python scripts/run_leakage.py --config $CONFIG --dataset $ds \
        --output $RESULTS_DIR/rq1
done

# Step 3: RQ2 - Defense Effectiveness
echo ""
echo "[Step 3/6] RQ2: Defense Effectiveness"
for ds in $DATASETS; do
    echo "  Dataset: $ds"
    python scripts/run_defense.py --config $CONFIG --dataset $ds \
        --output $RESULTS_DIR/rq2
done

# Step 4: RQ3 - Privacy-Utility Tradeoff
echo ""
echo "[Step 4/6] RQ3: Privacy-Utility Tradeoff"
for ds in $DATASETS; do
    echo "  Dataset: $ds"
    python scripts/run_tradeoff.py --config $CONFIG --dataset $ds \
        --epsilon 0.1 0.5 1.0 3.0 5.0 10.0 \
        --output $RESULTS_DIR/rq3
done

# Step 5: RQ4 - Ablation and Sensitivity
echo ""
echo "[Step 5/6] RQ4: Ablation and Sensitivity"
for ds in $DATASETS; do
    echo "  Dataset: $ds"
    python scripts/run_ablation.py --config $CONFIG --dataset $ds \
        --output $RESULTS_DIR/rq4
done

# Step 6: RQ5 - System Overhead
echo ""
echo "[Step 6/6] RQ5: System Overhead"
for ds in $DATASETS; do
    echo "  Dataset: $ds"
    python scripts/run_overhead.py --config $CONFIG --dataset $ds \
        --output $RESULTS_DIR/rq5
done

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "Results saved to $RESULTS_DIR/"
echo "=============================================="
echo ""
echo "To generate paper figures/tables:"
echo "  jupyter notebook notebooks/visualization.ipynb"
