#!/bin/bash
# Complete training and submission pipeline
# Usage: ./scripts/run_pipeline.sh [subset|full]

set -e  # Exit on error

MODE=${1:-subset}

echo "================================================================================"
echo "FME-UPC DATATHON 2025 - COMPLETE PIPELINE"
echo "================================================================================"
echo "Mode: $MODE"
echo ""

# Check if data exists
if [ ! -d "data/raw/train/train" ]; then
    echo "❌ Error: Training data not found at data/raw/train/train"
    echo "   Please extract the dataset first:"
    echo "   unzip data/raw/smadex-challenge-predict-the-revenue.zip -d data/raw/"
    exit 1
fi

echo "📊 Step 1: Training teacher models (CatBoost + LightGBM)..."
echo "────────────────────────────────────────────────────────────────────────────────"
uv run python scripts/train_teachers.py --${MODE}

echo ""
echo "🎓 Step 2: Training student models (distillation)..."
echo "────────────────────────────────────────────────────────────────────────────────"
uv run python scripts/train_students.py

echo ""
echo "📝 Step 3: Generating submission file..."
echo "────────────────────────────────────────────────────────────────────────────────"
uv run python scripts/make_submission.py

echo ""
echo "================================================================================"
echo "✅ PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "Submission file: data/submissions/submission.csv"
echo ""
echo "You can now submit this file to the competition!"



