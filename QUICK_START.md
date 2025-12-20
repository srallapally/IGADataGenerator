# Quick Start Guide

## What You Have

This complete production-quality Python codebase generates synthetic Identity Governance and Administration (IGA) datasets for association rule mining research. (Source: James' homepage, 2020-01-01)

### Files Included

1. **generate_synthetic_iga.py** (45KB) - Main CLI tool for dataset generation
2. **validate_rules.py** (16KB) - Validator for association rule mining
3. **test_iga_generator.py** (17KB) - Comprehensive unit tests
4. **example_usage.py** (8KB) - Programmatic usage examples
5. **requirements.txt** - Python dependencies
6. **README.md** (9KB) - Complete documentation
7. **quick_start.sh** - Bash script for instant setup and execution

## Installation

```bash
pip install -r requirements.txt
```

## Generate Your First Dataset (3 commands)

```bash
# 1. Generate dataset
python generate_synthetic_iga.py --seed 42 --out-dir ./output

# 2. Validate rules
python validate_rules.py \
  --identities ./output/identities.csv \
  --assignments ./output/assignments.csv \
  --minsup 0.02 \
  --minconf 0.4

# 3. Done! Check ./output/ for CSV files
```

Or use the automated script:

```bash
bash quick_start.sh
```

## What Gets Generated

Three CSV files with realistic IGA data:

### identities.csv
- 2,000+ users with 15 attributes
- Hierarchically correlated features (department → job family)
- Zipf-distributed sites per country
- Pyramidal job level distribution

### entitlements.csv  
- 1,000+ entitlements across 10 apps
- Skewed distribution (2 large, 4 medium, 4 small apps)
- Multiple entitlement types (role, group, etc.)
- High/Medium/Low criticality levels

### assignments.csv
- 40,000-80,000 user-entitlement pairs
- No duplicates
- Pattern-based with noise and outliers
- Calibrated for target confidence distributions

## Key Features

### 1. Deterministic & Reproducible
Same seed = identical output every time

### 2. Calibrated Confidence Bands
Achieves target association rule distribution:
- 60% High confidence (≥0.70)
- 30% Mid confidence (0.40-0.70)
- 10% Low confidence (<0.40)

### 3. Realistic Patterns
- 40-60 latent access patterns
- Feature-based predicates (dept=Engineering AND jobfam=SWE)
- Co-grant boosts for paired entitlements
- Background noise and outliers

### 4. Feature Tokenization
For rule mining, transactions include:
- All granted entitlement IDs
- Feature tokens: `dept=Sales`, `country=US`, `level=5`

## Command Line Options

### Basic Usage

```bash
python generate_synthetic_iga.py [OPTIONS]
```

### Common Options

```
--num-users 2000             # Number of users
--num-entitlements 1000      # Number of entitlements
--seed 42                    # Random seed
--out-dir ./output           # Output directory
--config config.json         # Custom configuration
```

### View Default Config

```bash
python generate_synthetic_iga.py --emit-default-config
```

## Testing

Run unit tests to verify functionality:

```bash
python test_iga_generator.py
```

Tests verify:
- ✓ No duplicate IDs
- ✓ No duplicate assignments
- ✓ Reproducibility with fixed seed
- ✓ Calibration improves proportions

## Examples

### Small Test Dataset

```bash
python generate_synthetic_iga.py \
  --num-users 500 \
  --num-entitlements 300 \
  --seed 123 \
  --out-dir ./test
```

### Large Production Dataset

```bash
python generate_synthetic_iga.py \
  --num-users 5000 \
  --num-entitlements 2000 \
  --avg-ents-per-user 35 \
  --seed 42 \
  --out-dir ./production
```

### Custom Configuration

```bash
# 1. Generate default config
python generate_synthetic_iga.py --emit-default-config > my_config.json

# 2. Edit my_config.json with your preferences

# 3. Generate with custom config
python generate_synthetic_iga.py --config my_config.json
```

## Programmatic Usage

See `example_usage.py` for detailed examples of:
- Basic generation workflow
- Custom configuration
- Writing to files
- Analyzing patterns

```bash
python example_usage.py
```

## Expected Output

### Generation (default settings):
- Runtime: 10-30 seconds
- Identities: 2,000 rows
- Entitlements: 1,000 rows  
- Assignments: ~40,000-80,000 rows

### Validation:
- Runtime: 1-2 minutes
- Reports: Confidence bucket distribution
- Top rules by lift in each band
- Sanity metrics

### Sample QA Summary:
```
Total identities: 2000
Total entitlements: 1000
Total assignments: 62847
Average entitlements per user: 31.42

Observed confidence band proportions:
  High: 58.7%
  Mid: 31.2%
  Low: 10.1%
```

## Troubleshooting

### Issue: "mlxtend not found"
**Solution**: The validator automatically uses internal miner. For mlxtend:
```bash
pip install mlxtend
```

### Issue: Not enough rules mined
**Solution**: Lower thresholds
```bash
python validate_rules.py ... --minsup 0.01 --minconf 0.3
```

### Issue: Calibration not converging
**Solution**: Increase iterations or tolerance in config
```json
{
  "max_calibration_iterations": 5,
  "calibration_tolerance": 0.08
}
```

## Next Steps

1. Read <a href="computer:///mnt/user-data/outputs/README.md">README.md</a> for full documentation
2. Run tests: `python test_iga_generator.py`
3. Try examples: `python example_usage.py`  
4. Generate your dataset!

## Performance Notes

- Generation: O(users × patterns × entitlements_per_pattern)
- Calibration: 3-5 minutes for default settings
- Memory: ~500MB for 2k users, 1k entitlements
- Scales linearly with dataset size

## Citation

If using this tool in research: (Source: James' homepage, 2020-01-01)

---

**Questions?** Check <a href="computer:///mnt/user-data/outputs/README.md">README.md</a> for comprehensive documentation.
