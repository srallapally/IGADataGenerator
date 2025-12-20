# Synthetic IGA Dataset Generator

Production-quality Python tool for generating synthetic Identity Governance and Administration (IGA) datasets for association-rule mining research and testing.

## Overview

This tool generates realistic identity, entitlement, and access assignment data with configurable distributions and a sophisticated calibration loop to achieve target association rule confidence bands.

## Features

- **Realistic Identity Generation**: 2,000+ users with hierarchically correlated attributes (department, job family, location, etc.)
- **Diverse Entitlement Catalog**: 1,000+ entitlements across 10 applications with functional bundles
- **Pattern-Based Assignment**: 40-60 latent access patterns with band-based probability sampling
- **Calibration Loop**: Iterative refinement to achieve target confidence distributions (60% High, 30% Mid, 10% Low)
- **Deterministic**: Fully reproducible with seed parameter
- **Validation Tools**: Association rule mining and confidence bucket analysis

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies:
- pandas >= 1.5.0
- numpy >= 1.23.0
- faker >= 18.0.0
- mlxtend >= 0.21.0
- scikit-learn >= 1.2.0

## Quick Start

### Generate Dataset

```bash
python generate_synthetic_iga.py --seed 42 --out-dir ./output
```

This generates three CSV files in `./output`:
- `identities.csv`: User identity data
- `entitlements.csv`: Entitlement catalog
- `assignments.csv`: User-entitlement assignments

### Validate Rules

```bash
python validate_rules.py \
  --identities ./output/identities.csv \
  --assignments ./output/assignments.csv \
  --minsup 0.02 \
  --minconf 0.4
```

## CLI Reference

### generate_synthetic_iga.py

```
Options:
  --num-users INT              Number of user identities [default: 2000]
  --num-apps INT               Number of applications [default: 10]
  --num-entitlements INT       Number of entitlements [default: 1000]
  --avg-ents-per-user INT      Target average entitlements per user [default: 30]
  --seed INT                   Random seed for reproducibility [default: 42]
  --out-dir PATH               Output directory [default: ./out]
  --config PATH                Path to JSON config file
  --emit-default-config        Print default config and exit
  --no-clobber                 Do not overwrite existing files
```

### validate_rules.py

```
Options:
  --identities PATH            Path to identities.csv [required]
  --assignments PATH           Path to assignments.csv [required]
  --minsup FLOAT               Minimum support threshold [default: 0.02]
  --minconf FLOAT              Minimum confidence threshold [default: 0.4]
  --max-len INT                Maximum rule length [default: 3]
  --seed INT                   Random seed [default: 42]
```

## Configuration

### Default Config

View the default configuration:

```bash
python generate_synthetic_iga.py --emit-default-config > config.json
```

### Custom Config

Create a custom config file and use it:

```bash
python generate_synthetic_iga.py --config config.json
```

Example config structure:

```json
{
  "num_users": 3000,
  "num_apps": 10,
  "num_entitlements": 1500,
  "departments": ["Engineering", "Sales", "Finance"],
  "num_patterns": 60,
  "target_confidence_bands": {
    "High": 0.60,
    "Mid": 0.30,
    "Low": 0.10
  }
}
```

## Data Schemas

### identities.csv

| Column | Type | Description |
|--------|------|-------------|
| user_id | string | Unique user identifier (U000001...) |
| first_name | string | First name |
| last_name | string | Last name |
| email | string | Email address |
| department | string | Department (Engineering, Sales, etc.) |
| business_unit | string | Business unit |
| job_family | string | Job family (SWE, PM, etc.) |
| job_level | int | Job level (1-7, pyramidal distribution) |
| title | string | Job title |
| location_country | string | Country code |
| location_site | string | Site identifier (Zipf distribution) |
| employment_type | string | FTE, Contractor, or Intern |
| cost_center | string | Cost center code |
| manager_flag | string | Y/N (10-15% managers) |
| tenure_years | float | Years at company (0-15) |

### entitlements.csv

| Column | Type | Description |
|--------|------|-------------|
| entitlement_id | string | Unique entitlement ID (E000001...) |
| app_id | string | Application ID (APP01...) |
| app_name | string | Application name |
| entitlement_name | string | Entitlement name |
| entitlement_type | string | role, group, project-role, etc. |
| criticality | string | High, Medium, or Low |
| scope | string | Scope label (optional) |
| description | string | Description |

### assignments.csv

| Column | Type | Description |
|--------|------|-------------|
| user_id | string | User identifier |
| entitlement_id | string | Entitlement identifier |

No duplicate pairs; typically 40k-80k rows for 2k users.

## Algorithm Details

### Generation Process

1. **Identity Generation**: Creates users with hierarchically correlated attributes
   - Department influences job family
   - Sites cluster by country with Zipf distribution
   - Job levels follow pyramidal distribution

2. **Entitlement Catalog**: Generates diverse entitlements across apps
   - 2 large apps (200-250 entitlements each)
   - 4 medium apps (80-120 entitlements each)
   - 4 small apps (40-80 entitlements each)
   - 15-25 functional bundles (cross-app groupings)

3. **Pattern Generation**: Creates 40-60 latent access patterns
   - Each pattern has 1-3 feature predicates
   - Each pattern has 5-20 entitlements (core + optional)
   - Assigned target confidence band (High/Mid/Low)

4. **Band-Based Sampling**: For each user matching a pattern:
   - Core entitlements: High (75-95%), Mid (50-70%), Low (20-40%)
   - Optional entitlements: High (60-75%), Mid (30-50%), Low (5-25%)

5. **Co-Grant Boosts**: Paired entitlements get +15% probability when partner is granted

6. **Noise & Ambiguity**:
   - Background utility grants (5% rate on 10-15% of entitlements)
   - Pattern collisions (15-25% of users match 2+ patterns)
   - Outliers (2-4% of users get random high-criticality entitlements)

7. **Calibration Loop** (max 3 iterations):
   - Compute empirical rule confidences
   - Measure bucket proportions (High/Mid/Low)
   - Adjust per-entitlement probabilities if outside tolerance (±5%)
   - Regenerate assignments
   - Repeat until convergence or max iterations

### Feature Tokenization

For rule mining, each user transaction includes:
- All granted entitlement IDs
- Feature tokens: `dept=Sales`, `country=US`, `jobfam=SWE`, `level=5`, `etype=FTE`

This enables mining of rules like: `{dept=Engineering, jobfam=SWE} => {E000042}`

## Validation Output

The validator reports:

1. **Confidence Bucket Distribution**: Percentage of rules in High (≥0.70), Mid (0.40-0.70), Low (<0.40)
2. **Top Rules by Lift**: Top 10 rules in each bucket
3. **Sanity Metrics**: 
   - Average entitlements per user
   - Department/site distribution entropy
   - Job level distribution

Example output:

```
Total rules mined: 247

Confidence bucket distribution:
  High: 148 rules (59.9%)
  Mid: 74 rules (30.0%)
  Low: 25 rules (10.1%)

Top 10 rules in High confidence band (by lift):
  1. dept=Engineering, jobfam=SWE => E000123 (conf=0.892, sup=0.045, lift=3.21)
  2. country=US, level=5 => E000456 (conf=0.857, sup=0.032, lift=2.98)
  ...
```

## Testing

Run basic unit tests:

```bash
python test_iga_generator.py
```

Tests verify:
- No duplicate user_id or entitlement_id
- No duplicate (user_id, entitlement_id) pairs
- Reproducibility with fixed seed
- Calibration improves bucket proportions

## Examples

### Generate Small Dataset for Testing

```bash
python generate_synthetic_iga.py \
  --num-users 500 \
  --num-entitlements 300 \
  --seed 123 \
  --out-dir ./test_data
```

### Generate Large Dataset

```bash
python generate_synthetic_iga.py \
  --num-users 5000 \
  --num-entitlements 2000 \
  --avg-ents-per-user 35 \
  --seed 42 \
  --out-dir ./large_data
```

### Validate with Stricter Parameters

```bash
python validate_rules.py \
  --identities ./output/identities.csv \
  --assignments ./output/assignments.csv \
  --minsup 0.05 \
  --minconf 0.6 \
  --max-len 2
```

## Performance

- Generation: ~10-30 seconds for 2k users, 1k entitlements
- Calibration: 3-5 minutes (3 iterations)
- Validation: ~1-2 minutes for default parameters

## Troubleshooting

### Issue: Not enough rules mined

**Solution**: Lower `--minsup` and `--minconf` thresholds

```bash
python validate_rules.py ... --minsup 0.01 --minconf 0.3
```

### Issue: Bucket proportions not converging

**Solution**: Increase calibration iterations or adjust tolerance in config

```json
{
  "max_calibration_iterations": 5,
  "calibration_tolerance": 0.08
}
```

### Issue: mlxtend import error

**Solution**: The validator will automatically fall back to internal rule miner. To use mlxtend:

```bash
pip install mlxtend
```

## Citation

If using this tool in research, please reference:
- Source: James' homepage, 2020-01-01

## License

This code is provided as-is for research and testing purposes.
