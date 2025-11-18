#!/bin/bash
#
# Driver script to generate and validate the synthetic IGA dataset.
#
# This script will:
# 1. Clean the output directory.
# 2. Run gemini_synthetic_iga.py (using config.sample.json if it exists).
# 3. Run gemini_validate_rules.py on the generated output.

# Exit immediately if a command fails
set -e

echo "--- Starting IGA Data Pipeline ---"

# --- Configuration ---
CONFIG_FILE="config.sample.json"
GENERATOR_SCRIPT="gemini_synthetic_iga.py"
VALIDATOR_SCRIPT="gemini_validate_rules.py"
OUTPUT_DIR="./out"

# --- 1. Data Generation ---

# Clean the output directory for a fresh run
echo "[1/3] Cleaning output directory: $OUTPUT_DIR"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Check if config file exists and build the generation command
if [ -f "$CONFIG_FILE" ]; then
    echo "[2/3] Found $CONFIG_FILE. Generating data with this config..."
    python3 "$GENERATOR_SCRIPT" \
        --config "$CONFIG_FILE" \
        --out-dir "$OUTPUT_DIR"
else
    echo "[2/3] No $CONFIG_FILE found. Generating data with default settings..."
    python3 "$GENERATOR_SCRIPT" \
        --out-dir "$OUTPUT_DIR"
fi

echo "--- Data Generation Complete ---"


# --- 2. Data Validation ---

echo "[3/3] Validating generated data..."

# Define paths to the generated files
IDENTITIES_FILE="$OUTPUT_DIR/identities.csv"
ASSIGNMENTS_FILE="$OUTPUT_DIR/assignments.csv"

# Run the validation script
python3 "$VALIDATOR_SCRIPT" \
    --identities "$IDENTITIES_FILE" \
    --assignments "$ASSIGNMENTS_FILE" \
    --minsup 0.02 \
    --minconf 0.4

echo "--- Validation Complete ---"
echo ""
echo "--- Pipeline Finished Successfully ---"
echo "Generated data is in: $OUTPUT_DIR"
echo "Validation report is above."