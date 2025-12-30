#!/usr/bin/env python3
"""
validate_generated_data.py

Validates that generated synthetic IGA data matches target confidence distribution
by mining association rules using the production analytics methodology.

This script:
1. Analyzes recommended features from identities.csv using Cramér's V
2. Mines frequent itemsets from HR attributes (Stage 1)
3. Mines association rules from attributes to entitlements (Stage 2)
4. Calculates confidence = freqUnion / freq (production formula)
5. Compares actual confidence distribution to target distribution

Usage:
    python validate_generated_data.py --identities identities.csv --accounts-dir out/ --config config.json
"""

import argparse
import json
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


# =============================================================================
# Feature Analyzer (using Cramér's V)
# =============================================================================

def calculate_cramers_v(chi2_stat: float, n: int, r: int, c: int) -> float:
    """Calculate Cramér's V from chi-square statistic."""
    if n == 0 or min(r - 1, c - 1) == 0:
        return 0.0
    return np.sqrt(chi2_stat / (n * min(r - 1, c - 1)))


def recommend_features_cramers_v(
        identities_df: pd.DataFrame,
        max_features: int = 10,
        max_unique_values: int = 50,
        cramers_v_threshold: float = 0.1
) -> List[str]:
    """
    Recommend features using Cramér's V for association strength.

    Returns features sorted by statistical significance.
    """
    logger = logging.getLogger("FeatureRecommender")

    skip_columns = {'user_id', 'user_name', 'email', 'first_name', 'last_name',
                    'employee_id', 'hire_date', 'tenure_years'}

    # Filter categorical columns with reasonable cardinality
    candidate_features = []
    for col in identities_df.columns:
        if col in skip_columns:
            continue

        unique_count = identities_df[col].nunique()
        if 2 <= unique_count <= max_unique_values:
            candidate_features.append(col)

    logger.info(f"Analyzing {len(candidate_features)} candidate features")

    # Calculate Cramér's V between all feature pairs
    feature_scores = defaultdict(list)

    for i, feat1 in enumerate(candidate_features):
        for feat2 in candidate_features[i + 1:]:
            try:
                ct = pd.crosstab(
                    identities_df[feat1].fillna('NA'),
                    identities_df[feat2].fillna('NA')
                )

                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    continue

                chi2, p, dof, expected = chi2_contingency(ct)
                n = len(identities_df)
                r, c = ct.shape
                cramers_v = calculate_cramers_v(chi2, n, r, c)

                if cramers_v >= cramers_v_threshold:
                    feature_scores[feat1].append(cramers_v)
                    feature_scores[feat2].append(cramers_v)

            except Exception as e:
                continue

    # Rank features by average Cramér's V
    feature_rankings = []
    for feat, scores in feature_scores.items():
        if scores:
            avg_v = np.mean(scores)
            max_v = max(scores)
            feature_rankings.append((feat, avg_v, max_v, len(scores)))

    # Sort by average Cramér's V (descending)
    feature_rankings.sort(key=lambda x: x[1], reverse=True)

    recommended = [f[0] for f in feature_rankings[:max_features]]

    logger.info(f"Recommended {len(recommended)} features:")
    for feat, avg_v, max_v, count in feature_rankings[:max_features]:
        logger.info(f"  {feat}: avg_v={avg_v:.3f}, max_v={max_v:.3f}, pairs={count}")

    return recommended


# =============================================================================
# Production-Style Rule Miner
# =============================================================================

@dataclass
class MinedRule:
    """Association rule mined from data."""
    antecedent: Tuple[str, ...]  # Feature markers
    consequent: str  # Single entitlement
    freq: int  # Users with antecedent
    freqUnion: int  # Users with antecedent + entitlement
    confidence: float  # freqUnion / freq


class ProductionRuleMiner:
    """
    Mines association rules using production analytics methodology.

    Stage 1: Mine frequent itemsets from HR attributes
    Stage 2: Mine rules from attributes to entitlements
    """

    def __init__(self,
                 identities_df: pd.DataFrame,
                 accounts_df: pd.DataFrame,
                 feature_columns: List[str],
                 min_support: int = 3):
        self.identities_df = identities_df
        self.accounts_df = accounts_df
        self.feature_columns = feature_columns
        self.min_support = min_support
        self.logger = logging.getLogger(self.__class__.__name__)

        # Stage 1 results
        self.frequent_itemsets: Dict[Tuple[str, ...], int] = {}

        # Stage 2 results
        self.rules: List[MinedRule] = []

    def mine_rules(self) -> List[MinedRule]:
        """Execute two-stage mining process."""
        self.logger.info("=" * 60)
        self.logger.info("PRODUCTION-STYLE RULE MINING")
        self.logger.info("=" * 60)

        # Stage 1: Mine frequent itemsets
        self._mine_frequent_itemsets()

        # Stage 2: Mine association rules
        self._mine_association_rules()

        return self.rules

    def _mine_frequent_itemsets(self) -> None:
        """
        Stage 1: Mine frequent itemsets from HR attributes only.

        Finds patterns like [Dublin], [Finance], [Dublin, Finance]
        that appear at least min_support times.
        """
        self.logger.info("Stage 1: Mining frequent itemsets from HR attributes")

        # Build transactions (one per user, HR attributes only)
        transactions = []
        for _, row in self.identities_df.iterrows():
            transaction = []
            for col in self.feature_columns:
                if col in row and pd.notna(row[col]) and str(row[col]) != '':
                    # Format: [Feature_Name=Value]
                    transaction.append(f"{col}={row[col]}")

            if transaction:
                transactions.append(transaction)

        self.logger.info(f"Built {len(transactions)} transactions from identities")

        # Count all itemsets up to size 3
        for transaction in transactions:
            for size in range(1, min(4, len(transaction) + 1)):
                for itemset in combinations(sorted(transaction), size):
                    itemset_tuple = tuple(sorted(itemset))
                    self.frequent_itemsets[itemset_tuple] = \
                        self.frequent_itemsets.get(itemset_tuple, 0) + 1

        # Filter by minimum support
        self.frequent_itemsets = {
            itemset: freq
            for itemset, freq in self.frequent_itemsets.items()
            if freq >= self.min_support
        }

        self.logger.info(f"Found {len(self.frequent_itemsets)} frequent itemsets")

        # Show top 10
        top_itemsets = sorted(
            self.frequent_itemsets.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        self.logger.info("Top 10 frequent itemsets:")
        for itemset, freq in top_itemsets:
            self.logger.info(f"  {list(itemset)}: {freq}")

    def _mine_association_rules(self) -> None:
        """
        Stage 2: Mine association rules from attributes to entitlements.

        Creates expanded table (one row per user-entitlement),
        mines rules, and calculates confidence = freqUnion / freq.
        """
        self.logger.info("Stage 2: Mining association rules (Attributes → Entitlements)")

        # Build expanded table (one row per user-entitlement assignment)
        expanded_rows = []

        for _, account_row in self.accounts_df.iterrows():
            user_id = account_row['user_id']

            # Get user's HR attributes
            user_info = self.identities_df[self.identities_df['user_id'] == user_id]
            if user_info.empty:
                continue

            user_info = user_info.iloc[0]

            # Get entitlement grants
            entitlement_grants = account_row.get('entitlement_grants', '')
            if pd.isna(entitlement_grants) or entitlement_grants == '':
                continue

            # Parse entitlement list
            delimiter = '#' if '#' in str(entitlement_grants) else ','
            entitlements = [e.strip() for e in str(entitlement_grants).split(delimiter) if e.strip()]

            # Create one row per entitlement
            for ent_id in entitlements:
                expanded_row = {
                    'user_id': user_id,
                    'entitlement': ent_id
                }

                # Add HR attributes
                for col in self.feature_columns:
                    if col in user_info and pd.notna(user_info[col]):
                        expanded_row[col] = f"{col}={user_info[col]}"
                    else:
                        expanded_row[col] = None

                expanded_rows.append(expanded_row)

        if not expanded_rows:
            self.logger.warning("No user-entitlement rows found")
            return

        expanded_df = pd.DataFrame(expanded_rows)
        self.logger.info(f"Expanded to {len(expanded_df)} user-entitlement rows")

        # Mine rules from expanded table
        rule_candidates = defaultdict(lambda: {'users': set()})

        for _, row in expanded_df.iterrows():
            user_id = row['user_id']
            entitlement = row['entitlement']

            # Build attribute set for this row
            attributes = []
            for col in self.feature_columns:
                if row[col] is not None and pd.notna(row[col]):
                    attributes.append(row[col])

            # Generate all itemset combinations as potential antecedents
            for size in range(1, len(attributes) + 1):
                if size > 3:  # Limit to 3 attributes max
                    break

                for itemset in combinations(sorted(attributes), size):
                    itemset_tuple = tuple(sorted(itemset))

                    # Only consider frequent itemsets from Stage 1
                    if itemset_tuple not in self.frequent_itemsets:
                        continue

                    key = (itemset_tuple, entitlement)
                    rule_candidates[key]['users'].add(user_id)

        # Build rules with corrected frequencies
        for (antecedent_tuple, entitlement), data in rule_candidates.items():
            freqUnion = len(data['users'])  # Users with antecedent AND entitlement

            # Filter by minimum support on freqUnion
            if freqUnion < self.min_support:
                continue

            # Get corrected freq from Stage 1
            freq = self.frequent_itemsets.get(antecedent_tuple, 0)

            if freq == 0:
                continue

            # Calculate confidence (production formula)
            confidence = freqUnion / freq

            rule = MinedRule(
                antecedent=antecedent_tuple,
                consequent=entitlement,
                freq=freq,
                freqUnion=freqUnion,
                confidence=confidence
            )

            self.rules.append(rule)

        self.logger.info(f"Mined {len(self.rules)} association rules")


# =============================================================================
# Validator
# =============================================================================

class DataValidator:
    """Validates generated data against target confidence distribution."""

    def __init__(self,
                 identities_file: Path,
                 accounts_dir: Path,
                 target_distribution: Dict[str, float],
                 confidence_thresholds: Dict[str, float]):
        self.identities_file = identities_file
        self.accounts_dir = accounts_dir
        self.target_distribution = target_distribution
        self.confidence_thresholds = confidence_thresholds
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load data
        self.identities_df = pd.read_csv(identities_file)
        self.logger.info(f"Loaded {len(self.identities_df)} identities")

        # Load all accounts
        self.accounts_by_app = {}
        for app_file in accounts_dir.glob("*_accounts.csv"):
            app_name = app_file.stem.replace("_accounts", "")
            self.accounts_by_app[app_name] = pd.read_csv(app_file)
            self.logger.info(f"Loaded {len(self.accounts_by_app[app_name])} accounts for {app_name}")

    def validate(self) -> Dict[str, any]:
        """
        Execute full validation pipeline.

        Returns validation report with:
        - Recommended features
        - Mined rules per app
        - Actual vs target confidence distribution
        - Pass/fail status
        """
        self.logger.info("=" * 60)
        self.logger.info("VALIDATION REPORT")
        self.logger.info("=" * 60)

        report = {
            'identities_count': len(self.identities_df),
            'apps': {},
            'overall': {},
            'validation': {
                'passed': False,
                'errors': [],
                'warnings': []
            }
        }

        # Step 1: Recommend features using Cramér's V
        self.logger.info("\n1. Analyzing features using Cramér's V")
        recommended_features = recommend_features_cramers_v(
            self.identities_df,
            max_features=10,
            max_unique_values=50,
            cramers_v_threshold=0.1
        )

        report['recommended_features'] = recommended_features

        if not recommended_features:
            report['validation']['errors'].append("No features passed Cramér's V threshold")
            return report

        # Step 2: Mine rules for each app
        all_rules = []

        for app_name, accounts_df in self.accounts_by_app.items():
            self.logger.info(f"\n2. Mining rules for {app_name}")

            miner = ProductionRuleMiner(
                identities_df=self.identities_df,
                accounts_df=accounts_df,
                feature_columns=recommended_features,
                min_support=3
            )

            app_rules = miner.mine_rules()
            all_rules.extend(app_rules)

            # Analyze confidence distribution for this app
            confidences = [r.confidence for r in app_rules]

            if confidences:
                report['apps'][app_name] = {
                    'rules_count': len(app_rules),
                    'confidence_stats': {
                        'mean': float(np.mean(confidences)),
                        'median': float(np.median(confidences)),
                        'min': float(np.min(confidences)),
                        'max': float(np.max(confidences))
                    }
                }

        # Step 3: Analyze overall confidence distribution
        self.logger.info("\n3. Analyzing overall confidence distribution")

        if not all_rules:
            report['validation']['errors'].append("No rules mined from data")
            return report

        all_confidences = [r.confidence for r in all_rules]

        # Bucket rules by confidence
        actual_distribution = self._bucket_confidences(all_confidences)

        report['overall'] = {
            'total_rules': len(all_rules),
            'target_distribution': self.target_distribution,
            'actual_distribution': actual_distribution,
            'confidence_stats': {
                'mean': float(np.mean(all_confidences)),
                'median': float(np.median(all_confidences)),
                'min': float(np.min(all_confidences)),
                'max': float(np.max(all_confidences)),
                'std': float(np.std(all_confidences))
            }
        }

        # Step 4: Compare to target distribution
        self.logger.info("\n4. Comparing to target distribution")
        self._compare_distributions(report)

        # Print summary
        self._print_summary(report)

        return report

    def _bucket_confidences(self, confidences: List[float]) -> Dict[str, float]:
        """Bucket confidence scores using configured thresholds."""
        buckets = {'high': 0, 'medium': 0, 'low': 0, 'none': 0}

        for conf in confidences:
            if conf >= self.confidence_thresholds['high']:
                buckets['high'] += 1
            elif conf >= self.confidence_thresholds['medium']:
                buckets['medium'] += 1
            elif conf >= self.confidence_thresholds['low']:
                buckets['low'] += 1
            else:
                buckets['none'] += 1

        # Convert to percentages
        total = len(confidences)
        return {k: v / total for k, v in buckets.items()}

    def _compare_distributions(self, report: Dict) -> None:
        """Compare actual vs target distribution and set validation status."""
        actual = report['overall']['actual_distribution']
        target = report['overall']['target_distribution']

        tolerance = 0.10  # 10% tolerance

        passed = True
        for bucket in ['high', 'medium', 'low', 'none']:
            actual_pct = actual.get(bucket, 0)
            target_pct = target.get(bucket, 0)

            diff = abs(actual_pct - target_pct)

            if diff > tolerance:
                passed = False
                report['validation']['errors'].append(
                    f"Bucket '{bucket}': actual={actual_pct:.1%}, target={target_pct:.1%}, "
                    f"diff={diff:.1%} (exceeds {tolerance:.0%} tolerance)"
                )
            elif diff > tolerance / 2:
                report['validation']['warnings'].append(
                    f"Bucket '{bucket}': actual={actual_pct:.1%}, target={target_pct:.1%}, "
                    f"diff={diff:.1%}"
                )

        report['validation']['passed'] = passed

    def _print_summary(self, report: Dict) -> None:
        """Print validation summary."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("VALIDATION SUMMARY")
        self.logger.info("=" * 60)

        self.logger.info(f"\nRecommended Features ({len(report['recommended_features'])}):")
        for feat in report['recommended_features']:
            self.logger.info(f"  - {feat}")

        self.logger.info(f"\nTotal Rules Mined: {report['overall']['total_rules']}")

        self.logger.info("\nConfidence Distribution Comparison:")
        self.logger.info(f"{'Bucket':<10} {'Target':<10} {'Actual':<10} {'Difference':<12} {'Status'}")
        self.logger.info("-" * 60)

        actual = report['overall']['actual_distribution']
        target = report['overall']['target_distribution']

        for bucket in ['high', 'medium', 'low', 'none']:
            actual_pct = actual.get(bucket, 0)
            target_pct = target.get(bucket, 0)
            diff = actual_pct - target_pct

            status = "✓ PASS" if abs(diff) <= 0.10 else "✗ FAIL"

            self.logger.info(
                f"{bucket:<10} {target_pct:>8.1%} {actual_pct:>8.1%} "
                f"{diff:>+10.1%}  {status}"
            )

        self.logger.info("\n" + "=" * 60)

        if report['validation']['passed']:
            self.logger.info("✓ VALIDATION PASSED")
        else:
            self.logger.error("✗ VALIDATION FAILED")

            if report['validation']['errors']:
                self.logger.error("\nErrors:")
                for error in report['validation']['errors']:
                    self.logger.error(f"  - {error}")

        if report['validation']['warnings']:
            self.logger.warning("\nWarnings:")
            for warning in report['validation']['warnings']:
                self.logger.warning(f"  - {warning}")

        self.logger.info("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def setup_logging(level: str = 'INFO'):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s'
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Validate Synthetic IGA Data Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--identities', '-i', required=True, type=Path,
                        help='Path to identities.csv')
    parser.add_argument('--accounts-dir', '-a', required=True, type=Path,
                        help='Directory containing {app}_accounts.csv files')
    parser.add_argument('--config', '-c', type=Path,
                        help='Path to config JSON (for target distribution)')
    parser.add_argument('--output', '-o', type=Path,
                        help='Path to save validation report JSON')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose (DEBUG) logging')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level)

    # Load target distribution from config
    target_distribution = {
        'high': 0.35,
        'medium': 0.30,
        'low': 0.30,
        'none': 0.05
    }

    confidence_thresholds = {
        'high': 0.70,
        'medium': 0.40,
        'low': 0.01
    }

    if args.config and args.config.exists():
        with open(args.config) as f:
            config = json.load(f)

            # Extract target distribution
            conf_cfg = config.get('confidence', {})
            if 'distribution' in conf_cfg:
                dist = conf_cfg['distribution']
                if isinstance(dist, dict) and 'value' in dist:
                    target_distribution = dist['value']
                elif isinstance(dist, dict):
                    target_distribution = dist

            # Extract thresholds
            if 'thresholds' in conf_cfg:
                thresh = conf_cfg['thresholds']
                if isinstance(thresh, dict) and 'value' in thresh:
                    thresh = thresh['value']

                confidence_thresholds = {
                    'high': thresh.get('high', {}).get('min', 0.70),
                    'medium': thresh.get('medium', {}).get('min', 0.40),
                    'low': thresh.get('low', {}).get('min', 0.01)
                }

    # Run validation
    validator = DataValidator(
        identities_file=args.identities,
        accounts_dir=args.accounts_dir,
        target_distribution=target_distribution,
        confidence_thresholds=confidence_thresholds
    )

    report = validator.validate()

    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        logging.info(f"\nValidation report saved to {args.output}")

    # Exit with appropriate code
    exit_code = 0 if report['validation']['passed'] else 1
    exit(exit_code)


if __name__ == '__main__':
    main()