# IGA Data Generator

A sophisticated synthetic data generation system for Identity Governance and Administration (IGA) testing and analytics. Creates statistically realistic datasets with meaningful feature-entitlement associations that mirror production enterprise environments.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Machine Learning Concepts](#machine-learning-concepts)
- [Output Files](#output-files)
- [Validation](#validation)
- [Advanced Usage](#advanced-usage)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

## Overview

The IGA Data Generator creates synthetic datasets that include:

- **User Identities**: 20+ attributes including department, job level, location, manager hierarchy
- **Application Entitlements**: Role catalogs for AWS, Salesforce, ServiceNow, SAP, and custom apps
- **Access Assignments**: User-to-entitlement mappings with confidence scores
- **Statistical Associations**: Feature-to-entitlement rules with configurable support/confidence

### Why This Matters

Unlike simple random data generators, this system:

1. **Avoids Overfitting**: Uses schema-based rule generation instead of mining patterns from generated data
2. **Ensures Statistical Validity**: Produces data with predictable Cramér's V, support, and confidence metrics
3. **Models Real Patterns**: Cross-application rules reflect how enterprise access actually works
4. **Enables Testing**: Provides ground truth for role mining, anomaly detection, and access analytics

## Key Features

### Schema-Based Rule Generation

Rules are generated **before** identities using abstract feature schemas:

```
Define Rule Schemas → Generate Rules → Generate Identities → Apply Rules → Validate
```

This prevents the circular dependency of needing data to create rules and needing rules to create realistic data.

### Deterministic Quota Assignment

Ensures mined confidence matches target confidence using the formula:

```
mined_confidence = freqUnion / freq
freqUnion_target = confidence × freq
```

For a rule `[Department=Finance] → SAP_FI_001` with 85% confidence:
- 1000 Finance users (freq)
- Exactly 850 get the entitlement (freqUnion_target)
- Mined confidence = 850/1000 = 85% ✓

### Cross-Application Rules

Models realistic enterprise access patterns:

```json
{
  "rule_id": "R001",
  "antecedent": {"department": "Finance", "job_level": "Senior"},
  "consequent": {
    "SAP": ["FI_ACCOUNTANT", "FI_ANALYST"],
    "AWS": ["PowerUserAccess"],
    "Salesforce": ["Finance_User"]
  },
  "confidence": 0.85
}
```

### Statistical Feature Selection

Multi-stage filtering using:
1. **Cardinality Filter**: Max 50 unique values
2. **Cramér's V**: Association strength ≥ 0.1
3. **Chi-Square Test**: Statistical significance p < 0.05
4. **Mutual Information**: Predictive power ranking

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Configuration (synthetic_iga_data_generator_config.json)   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Schema Definition & Rule Generation               │
│  • Load feature schemas (departments, job titles, etc.)    │
│  • Generate association rules with target metrics          │
│  • Configure cross-app vs per-app mode                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Identity Generation                               │
│  • Extract rule patterns                                   │
│  • Calculate quotas (support × √confidence weighting)      │
│  • Generate identities matching patterns                   │
│  • Fill remaining with random identities                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Entitlement Assignment                            │
│  • Compute rule quotas (freqUnion_target)                  │
│  • Deterministically assign entitlements                   │
│  • Calculate confidence scores                             │
│  • Enforce distribution targets (80/20 rule)               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Validation & Output                               │
│  • Feature validation (Cramér's V, chi-square)             │
│  • Distribution validation                                 │
│  • Export CSV files                                        │
│  • Generate QA summary                                     │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Libraries

```
pandas>=1.5.0
numpy>=1.23.0
faker>=18.0.0
mlxtend>=0.21.0
scikit-learn>=1.2.0
scipy
```

## Quick Start

### 1. Basic Generation

```bash
python synthetic_iga_data_generator.py --config synthetic_iga_data_generator_config.json
```

### 2. Validate Output

```bash
python validate_generated_data.py --identities out/identities.csv --accounts-dir out/
```

### 3. Check for Zero-Entitlement Users

```bash
python validate_zero_entitlements.py out/
```

## Configuration

### Key Configuration Sections

#### Global Settings

```json
{
  "global": {
    "seed": 42,
    "output_directory": "./out",
    "rules_directory": "./rules",
    "log_level": "INFO"
  }
}
```

#### Identity Generation

```json
{
  "identity": {
    "num_identities": 3500,
    "pct_users_without_manager": 0.10,
    "distribution_employee_contractor": {
      "Employee": 0.70,
      "Contractor": 0.20,
      "Intern": 0.10
    }
  }
}
```

#### Dynamic Rule Generation

```json
{
  "dynamic_rules": {
    "enabled": true,
    "use_cross_app_rules": true,
    "num_cross_app_rules": 700,
    "num_rules_per_app": 20,
    "confidence_distribution": {
      "high": 0.50,
      "medium": 0.35,
      "low": 0.15
    },
    "confidence_ranges": {
      "high": {"min": 0.80, "max": 1.00},
      "medium": {"min": 0.50, "max": 0.79},
      "low": {"min": 0.20, "max": 0.49}
    },
    "max_cardinality": 20
  }
}
```

#### Confidence Score Distribution

```json
{
  "confidence": {
    "distribution": {
      "high": 0.35,
      "medium": 0.30,
      "low": 0.30,
      "none": 0.05
    },
    "thresholds": {
      "high": {"min": 0.70},
      "medium": {"min": 0.40},
      "low": {"min": 0.01}
    },
    "pct_modelled_users": 0.93
  }
}
```

### Feature Selection

```json
{
  "features": {
    "mandatory_features": ["job_level", "business_unit", "department_type"],
    "additional_features": ["location_country", "employment_type", "is_manager"],
    "num_features_for_rules": 8,
    "feature_selection_method": "cramers_v"
  }
}
```

## Machine Learning Concepts

### Association Rule Mining (ARM)

The system uses ARM principles in reverse:

**Traditional Approach:**
```
Data → Mine Patterns → Discover Rules
```

**This System:**
```
Define Schemas → Generate Rules → Generate Data → Validate Patterns
```

#### Key Metrics

- **Support**: `P(features AND entitlements)` - frequency of pattern in population
- **Confidence**: `P(entitlements | features)` - conditional probability
- **Lift**: How much more likely the consequent is given antecedent vs. random
- **Cramér's V**: Effect size measuring association strength (0 to 1)

### Feature Selection Pipeline

```
Stage 1: Cardinality Filter (≤50 unique values)
    ↓
Stage 2: Statistical Significance (Cramér's V ≥ 0.1 or p < 0.05)
    ↓
Stage 3: Low-Cardinality for MI (preparation)
    ↓
Stage 4: Mutual Information Calculation
    ↓
Stage 5: Aggregation & Ranking
```

### Distribution Modeling

#### Tenure (Beta Distribution)

```python
# Right-skewed: most employees newer
beta_a = 2.0
beta_b = 5.0
tenure = beta(a, b) × 20 years
```

#### Job Level (Discrete Distribution)

```python
{
  'Junior': 0.20,
  'Mid': 0.30,
  'Senior': 0.20,
  'Manager': 0.10,
  'Executive': 0.02
}
```

## Output Files

### identities.csv

User profiles with 20+ attributes:

| Column | Description |
|--------|-------------|
| user_id | Unique identifier (U0000001) |
| user_name | Username (firstnamelastname) |
| department | Department name |
| job_level | Junior/Mid/Senior/Manager/Director/VP/Executive |
| business_unit | Industry/business unit |
| location_country | Country code (US, GB, IN, DE, AU) |
| manager | Manager's user_id |
| is_manager | Y/N flag |
| tenure_years | Years of service (Beta distribution) |

### {app_name}_entitlements.csv

Entitlement catalogs per application:

| Column | Description |
|--------|-------------|
| entitlement_id | Unique entitlement identifier |
| entitlement_name | Human-readable name |
| app_name | Application name |
| entitlement_type | standard/License/PermissionSet/linkedTemplates |
| criticality | High/Medium/Low |

### {app_name}_accounts.csv

User-entitlement assignments:

| Column | Description |
|--------|-------------|
| user_id | User identifier |
| user_name | Username |
| entitlement_grants | Pipe-delimited entitlement IDs |
| confidence_score | Numeric score 0.0-1.0 |
| confidence_bucket | High/Medium/Low/None |

### QA Summary

Validation report including:
- Identity distribution statistics
- Entitlement coverage per app
- Confidence distribution breakdown
- Feature validation results

## Validation

### Automated Validation

The system includes built-in validation:

```bash
# Full validation with confidence distribution check
python validate_generated_data.py \
  --identities out/identities.csv \
  --accounts-dir out/ \
  --config synthetic_iga_data_generator_config.json \
  --output validation_report.json
```

### Validation Checks

1. **Feature Quality**
   - Cardinality within limits
   - Cramér's V ≥ threshold
   - Non-constant features

2. **Confidence Distribution**
   - Actual vs. target bucket distribution
   - Tolerance: ±10%

3. **Data Integrity**
   - No duplicate user_id per app
   - No users with zero entitlements
   - All mandatory entitlements used

4. **Statistical Associations**
   - Mined rules match target confidence
   - Support levels within expected ranges

### Expected Validation Output

```
=== VALIDATION SUMMARY ===
Bucket      Target    Actual    Difference   Status
High        35.0%     34.2%     -0.8%        ✓ PASS
Medium      30.0%     31.1%     +1.1%        ✓ PASS
Low         30.0%     29.8%     -0.2%        ✓ PASS
None         5.0%      4.9%     -0.1%        ✓ PASS

✓ VALIDATION PASSED
```

## Advanced Usage

### Custom Applications

Add applications beyond the mandatory four (AWS, Salesforce, ServiceNow, SAP):

```json
{
  "applications": {
    "num_apps": 6,
    "additional_app_pool": ["Workday", "Okta", "GitHub", "Slack"],
    "apps": [
      {
        "app_name": "Workday",
        "app_id": "APP_WORKDAY",
        "enabled": true,
        "num_entitlements": 75,
        "criticality_distribution": {
          "High": 0.20,
          "Medium": 0.50,
          "Low": 0.30
        }
      }
    ]
  }
}
```

### Coordinated Cross-App Rules

Generate rules that reuse the same feature patterns across apps:

```json
{
  "dynamic_rules": {
    "coordinate_rules_across_apps": true,
    "num_unique_feature_patterns": 10
  }
}
```

This creates 10 feature patterns (e.g., `{department=Finance, job_level=Senior}`) and generates one rule per app using each pattern.

### Feature Schema Customization

Control which features are used in rules by adjusting cardinality limits:

```json
{
  "dynamic_rules": {
    "max_cardinality": 20,
    "min_cardinality": 2
  }
}
```

Features with too many unique values (e.g., 198 departments) are automatically excluded.

### Per-App Distribution Control

Enforce that 80% of users have 3+ entitlements **per application**:

```json
{
  "grants": {
    "pct_users_with_3_plus_per_app": 0.80
  }
}
```

## Technical Details

### Rule-Aware Identity Generation

The system implements a two-phase approach:

**Phase 1: Pattern-Based Generation (93% of users)**

```python
# Extract patterns from rules
patterns = extract_rule_patterns(rules)

# Calculate quotas using weighted allocation
weight = support × √confidence  # Square root dampens high-confidence dominance
quota = budget × (weight / total_weight)

# Generate identities matching patterns
for pattern, quota in pattern_quotas:
    generate_identities_matching(pattern, quota)
```

**Phase 2: Random Generation (7% of users)**

Fills remaining slots with random identities to prevent 100% rule coverage.

### Deterministic Assignment Algorithm

```python
# 1. Compute exact quota
freq = count_users_matching_antecedent(rule)
freqUnion_target = int(confidence × freq)

# 2. Sort users by ID (CRITICAL for nested patterns)
matching_users = sorted(matching_users, key=lambda u: u.user_id)

# 3. Select first N users deterministically
selected = matching_users[:freqUnion_target]

# 4. Grant entitlements to selected users
for user in selected:
    assign_entitlements(user, rule.consequent)
```

### Cross-App Atomicity

Users are pre-determined to follow rules consistently across all apps:

```python
# Single decision per user (not per user-app)
if random() < pct_modelled_users:
    rule_following_users.add(user_id)

# Applied consistently to ALL apps
for app in apps:
    if user_id in rule_following_users:
        apply_rules(user_id, app)
```

### Feature Cardinality Filtering

Only low-cardinality features are used in rules to ensure statistical power:

```python
# Skip high-cardinality features
if feature_cardinality > max_cardinality:
    logger.info(f"Skipping '{feature}': {cardinality} > {max_cardinality}")
    continue

# Example: department with 198 values → skipped
# Example: job_level with 8 values → included
```

This prevents overly-specific rules that have poor mined confidence.

## Troubleshooting

### Issue: Low Mined Confidence Scores

**Symptoms**: Validation shows 82% Low confidence instead of target 30%

**Causes**:
1. High-cardinality features in rules
2. Probabilistic assignment instead of quota-based
3. Mismatched features between rule generation and validation

**Solutions**:
```json
{
  "dynamic_rules": {
    "max_cardinality": 20,  // Lower to exclude high-cardinality features
    "confidence_ranges": {
      "high": {"min": 0.80, "max": 1.00}  // Raise minimum
    }
  }
}
```

### Issue: Users with Zero Entitlements

**Symptoms**: Some users have no entitlements across all apps

**Cause**: Edge case in random assignment

**Solution**: System auto-corrects with final safeguard:

```python
# Automatic fix in generate_all()
_ensure_no_users_without_entitlements()
```

### Issue: Rule Pattern Not Covered

**Symptoms**: Generated rules don't cover expected feature combinations

**Causes**:
1. Too few rules configured
2. Feature excluded due to cardinality
3. Random selection didn't pick pattern

**Solutions**:
```json
{
  "dynamic_rules": {
    "num_cross_app_rules": 700,  // Increase to cover more patterns
    "max_features_per_rule": 2,  // Lower for broader coverage
    "min_features_per_rule": 1
  }
}
```

### Issue: Duplicate Accounts

**Symptoms**: Multiple rows per user_id in accounts file

**Cause**: Bug in account generation

**Solution**: System includes deduplication:

```python
# Automatic deduplication in generate_for_app()
accounts = _deduplicate_accounts(accounts, app_name)
```

### Debug Mode

Enable verbose logging:

```bash
python synthetic_iga_data_generator.py --config config.json --verbose
```

Or in configuration:

```json
{
  "global": {
    "log_level": "DEBUG"
  }
}
```

## Performance Considerations

### Memory Usage

- **3,500 users**: ~50 MB
- **10,000 users**: ~150 MB
- **100,000 users**: ~1.5 GB

### Generation Time

- **3,500 users, 6 apps, 700 rules**: ~2-3 minutes
- **10,000 users, 6 apps, 700 rules**: ~5-7 minutes
- **100,000 users, 10 apps, 1000 rules**: ~30-45 minutes

### Optimization Tips

1. **Reduce MI calculation overhead**:
   ```json
   {"feature_validation": {"validation_sample_size": 5000}}
   ```

2. **Limit cross-app rules**:
   ```json
   {"dynamic_rules": {"num_cross_app_rules": 500}}
   ```

3. **Disable validation**:
   ```json
   {"feature_validation": {"enabled": false}}
   ```

## References

- **Association Rule Mining**: Agrawal & Srikant (1994)
- **Cramér's V**: Cramér (1946) - Mathematical Methods of Statistics
- **Feature Selection**: Guyon & Elisseeff (2003) - An Introduction to Variable and Feature Selection
- **Mutual Information**: Cover & Thomas (2006) - Elements of Information Theory
