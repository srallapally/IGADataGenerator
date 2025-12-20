#!/bin/bash
# Quick start script for synthetic IGA dataset generator

set -e

echo "=========================================="
echo "Synthetic IGA Dataset Generator"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not found"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import pandas, numpy, faker" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install -r requirements.txt
fi

echo "Dependencies OK"
echo ""

# Generate dataset with default parameters
echo "Generating synthetic IGA dataset..."
echo "  Users: 2000"
echo "  Entitlements: 1000"
echo "  Apps: 10"
echo "  Seed: 42"
echo ""

python3 generate_synthetic_iga.py \
    --num-users 2000 \
    --num-entitlements 1000 \
    --num-apps 10 \
    --seed 42 \
    --out-dir ./output

echo ""
echo "=========================================="
echo "Dataset generated successfully!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  ./output/identities.csv"
echo "  ./output/entitlements.csv"
echo "  ./output/assignments.csv"
echo ""

# Validate if mlxtend is available
echo "Validating generated rules..."
python3 -c "import mlxtend" 2>/dev/null
if [ $? -eq 0 ]; then
    python3 validate_rules.py \
        --identities ./output/identities.csv \
        --assignments ./output/assignments.csv \
        --minsup 0.02 \
        --minconf 0.4
else
    echo "Note: mlxtend not installed. Install with 'pip install mlxtend' for rule validation"
    echo "Running validation with internal miner..."
    python3 validate_rules.py \
        --identities ./output/identities.csv \
        --assignments ./output/assignments.csv \
        --minsup 0.02 \
        --minconf 0.4
fi

echo ""
echo "=========================================="
echo "Complete!"
echo "=========================================="
