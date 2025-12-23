#!/usr/bin/env python3
"""
dynamic_rule_generator_cross_app.py

Enhanced version that generates TRUE cross-app rules where a single feature pattern
triggers entitlements across multiple applications simultaneously.

Key difference from original:
- Original: coordinate_rules_across_apps=true creates N rules per app using same patterns
  Result: N*M total rules (N patterns × M apps), but each rule is app-specific

- This version: Creates N cross-app rules total
  Result: N total rules, each rule grants entitlements across multiple apps

Example output:
{
  "rule_id": "R001",
  "description": "Users with business_unit='Hospital' get entitlements in SAP, AWS, Salesforce",
  "antecedent": {"business_unit": "Hospital"},
  "consequent": {
    "SAP": ["SAP_MM_PUR_SERVICE", ...],
    "AWS": ["AmazonCloudFrontReadOnlyAccess", ...],
    "Salesforce": [...]
  },
  "strength": {...}
}
"""

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np
from numpy.random import Generator


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class CrossAppRuleGeneratorConfig:
    """Configuration for cross-app rule generation."""
    num_cross_app_rules: int = 10  # Total number of cross-app rules to generate
    apps_per_rule_min: int = 2  # Minimum apps per rule
    apps_per_rule_max: int = 4  # Maximum apps per rule

    confidence_distribution: Dict[str, float] = None
    confidence_ranges: Dict[str, Tuple[float, float]] = None
    support_range: Tuple[float, float] = (0.01, 0.20)
    cramers_v_range: Tuple[float, float] = (0.30, 0.50)

    min_features_per_rule: int = 1
    max_features_per_rule: int = 3
    min_entitlements_per_app: int = 1
    max_entitlements_per_app: int = 4

    def __post_init__(self):
        if self.confidence_distribution is None:
            self.confidence_distribution = {
                'high': 0.40,
                'medium': 0.35,
                'low': 0.25
            }

        if self.confidence_ranges is None:
            self.confidence_ranges = {
                'high': (0.85, 0.95),
                'medium': (0.75, 0.84),
                'low': (0.65, 0.74)
            }


# =============================================================================
# Feature Analyzer (reused from original)
# =============================================================================

class FeatureAnalyzer:
    """Analyzes user data to identify good candidate features for rules."""

    def __init__(self, users_df: pd.DataFrame, rng: Generator):
        self.users_df = users_df
        self.rng = rng
        self.logger = logging.getLogger(self.__class__.__name__)
        self.feature_stats = self._analyze_features()

    def _analyze_features(self) -> Dict[str, Dict]:
        """Analyze each feature's characteristics."""
        stats = {}

        for col in self.users_df.columns:
            if col in ['user_id', 'user_name', 'email', 'first_name', 'last_name',
                       'employee_id', 'hire_date', 'tenure_years']:
                continue

            unique_vals = self.users_df[col].nunique()
            total_vals = len(self.users_df)
            value_counts = self.users_df[col].value_counts()
            top_values = value_counts.head(10).to_dict()

            stats[col] = {
                'unique_count': unique_vals,
                'cardinality_ratio': unique_vals / total_vals,
                'top_values': top_values,
                'is_categorical': unique_vals < total_vals * 0.5,
                'entropy': self._calculate_entropy(value_counts / total_vals)
            }

        return stats

    def _calculate_entropy(self, probabilities: pd.Series) -> float:
        """Calculate Shannon entropy for a distribution."""
        probs = probabilities[probabilities > 0]
        return -np.sum(probs * np.log2(probs))

    def get_good_features(self, min_cardinality: int = 2, max_cardinality: int = 50) -> List[str]:
        """Get features suitable for rule generation."""
        good_features = []

        for feature, stats in self.feature_stats.items():
            if (min_cardinality <= stats['unique_count'] <= max_cardinality and
                    stats['is_categorical']):
                good_features.append(feature)

        self.logger.info(f"Found {len(good_features)} suitable features for rules")
        return good_features

    def get_feature_values(self, feature: str, max_values: int = 10) -> List[str]:
        """Get the most common values for a feature."""
        return list(self.feature_stats[feature]['top_values'].keys())[:max_values]

    def estimate_population_with_features(self, feature_dict: Dict[str, str]) -> int:
        """Estimate how many users match the given feature combination."""
        mask = pd.Series([True] * len(self.users_df))

        for feature, value in feature_dict.items():
            if feature in self.users_df.columns:
                mask &= (self.users_df[feature] == value)

        return mask.sum()


# =============================================================================
# Cross-App Rule Generator
# =============================================================================

class CrossAppRuleGenerator:
    """Generates cross-app association rules."""

    def __init__(self,
                 users_df: pd.DataFrame,
                 apps: List[Dict[str, any]],
                 entitlements_by_app: Dict[str, List[Dict[str, str]]],
                 config: CrossAppRuleGeneratorConfig,
                 rng: Generator):
        self.users_df = users_df
        self.apps = apps
        self.entitlements_by_app = entitlements_by_app
        self.config = config
        self.rng = rng
        self.logger = logging.getLogger(self.__class__.__name__)

        self.feature_analyzer = FeatureAnalyzer(users_df, rng)
        self.good_features = self.feature_analyzer.get_good_features()
        self.used_combinations: Set[Tuple] = set()

    def generate_all_rules(self) -> List[Dict]:
        """Generate cross-app rules."""
        self.logger.info("=" * 60)
        self.logger.info("GENERATING CROSS-APP RULES")
        self.logger.info(f"Target: {self.config.num_cross_app_rules} rules")
        self.logger.info(f"Apps per rule: {self.config.apps_per_rule_min}-{self.config.apps_per_rule_max}")
        self.logger.info("=" * 60)

        all_rules = []
        confidence_buckets = self._sample_confidence_buckets(self.config.num_cross_app_rules)

        failed_attempts = 0
        max_total_attempts = self.config.num_cross_app_rules * 100

        for i in range(self.config.num_cross_app_rules):
            rule_id = f"R{i + 1:03d}"
            confidence_bucket = confidence_buckets[i]

            rule = None
            attempts = 0
            while rule is None and attempts < 100 and failed_attempts < max_total_attempts:
                rule = self._generate_single_cross_app_rule(
                    rule_id=rule_id,
                    confidence_bucket=confidence_bucket
                )
                attempts += 1
                if rule is None:
                    failed_attempts += 1

            if rule:
                all_rules.append(rule)
                self.logger.info(f"Generated {rule_id}: {len(rule['consequent'])} apps, "
                                 f"feature pattern: {rule['antecedent']}")
            else:
                self.logger.warning(
                    f"Could not generate rule {rule_id} after {attempts} attempts"
                )

        self.logger.info(f"Generated {len(all_rules)} cross-app rules")
        return all_rules

    def _generate_single_cross_app_rule(self,
                                        rule_id: str,
                                        confidence_bucket: str) -> Optional[Dict]:
        """Generate a single cross-app rule."""
        max_attempts = 50

        for attempt in range(max_attempts):
            # Step 1: Select feature combination (antecedent)
            antecedent = self._select_feature_combination()

            if not antecedent:
                continue

            combo_key = tuple(sorted(antecedent.items()))
            if combo_key in self.used_combinations:
                continue

            # Step 2: Estimate population
            matching_population = self.feature_analyzer.estimate_population_with_features(antecedent)
            total_population = len(self.users_df)

            if matching_population < 10:  # Need more users for cross-app rules
                continue

            # Step 3: Select which apps this rule applies to
            num_apps = self.rng.integers(
                self.config.apps_per_rule_min,
                min(self.config.apps_per_rule_max + 1, len(self.apps) + 1)
            )
            selected_apps = self.rng.choice(self.apps, size=num_apps, replace=False)

            # Step 4: Sample confidence, support, Cramér's V
            conf_range = self.config.confidence_ranges[confidence_bucket]
            confidence = self.rng.uniform(conf_range[0], conf_range[1])

            max_support = matching_population / total_population
            support_min, support_max = self.config.support_range

            actual_min_support = max(support_min, 0.001)
            actual_max_support = min(support_max, max_support * 0.9)

            if actual_min_support >= actual_max_support:
                continue

            support = self.rng.uniform(actual_min_support, actual_max_support)
            cramers_v = self.rng.uniform(
                self.config.cramers_v_range[0],
                self.config.cramers_v_range[1]
            )

            # Step 5: Select entitlements for each app (consequent)
            consequent = {}
            app_names = []

            for app in selected_apps:
                app_name = app['app_name']
                app_names.append(app_name)

                entitlements = self.entitlements_by_app.get(app_name, [])
                if not entitlements:
                    continue

                num_ents = self.rng.integers(
                    self.config.min_entitlements_per_app,
                    min(self.config.max_entitlements_per_app + 1, len(entitlements) + 1)
                )

                selected_ents = self.rng.choice(entitlements, size=num_ents, replace=False)
                consequent[app_name] = [e['entitlement_id'] for e in selected_ents]

            if not consequent:
                continue

            # Step 6: Create description
            description = self._create_cross_app_description(
                antecedent=antecedent,
                app_names=app_names
            )

            # Step 7: Build rule
            rule = {
                'rule_id': rule_id,
                'description': description,
                'antecedent': antecedent,
                'consequent': consequent,  # Dict[app_name -> List[entitlement_ids]]
                'strength': {
                    'confidence': round(confidence, 3),
                    'support': round(support, 3),
                    'target_cramers_v': round(cramers_v, 3)
                },
                'metadata': {
                    'confidence_bucket': confidence_bucket,
                    'matching_population': int(matching_population),
                    'total_population': int(total_population),
                    'num_apps': len(consequent),
                    'apps': list(consequent.keys()),
                    'cross_app': True
                }
            }

            self.used_combinations.add(combo_key)
            return rule

        return None

    def _select_feature_combination(self) -> Optional[Dict[str, str]]:
        """Select a random combination of features and values."""
        if not self.good_features:
            return None

        num_features = self.rng.integers(
            self.config.min_features_per_rule,
            self.config.max_features_per_rule + 1
        )
        num_features = min(num_features, len(self.good_features))

        selected_features = self.rng.choice(
            self.good_features,
            size=num_features,
            replace=False
        )

        antecedent = {}
        for feature in selected_features:
            values = self.feature_analyzer.get_feature_values(feature)
            if values:
                value = self.rng.choice(values)
                antecedent[feature] = str(value)

        return antecedent if antecedent else None

    def _create_cross_app_description(self,
                                      antecedent: Dict[str, str],
                                      app_names: List[str]) -> str:
        """Create a human-readable description for the cross-app rule."""
        # Build antecedent description
        conditions = []
        for feature, value in antecedent.items():
            conditions.append(f"{feature}='{value}'")

        antecedent_desc = " AND ".join(conditions)

        # Build consequent description
        if len(app_names) == 1:
            apps_desc = app_names[0]
        elif len(app_names) == 2:
            apps_desc = " and ".join(app_names)
        else:
            apps_desc = ", ".join(app_names[:-1]) + f", and {app_names[-1]}"

        return f"Users with {antecedent_desc} get entitlements in {apps_desc}"

    def _sample_confidence_buckets(self, num_rules: int) -> List[str]:
        """Sample confidence buckets according to distribution."""
        buckets = list(self.config.confidence_distribution.keys())
        probs = list(self.config.confidence_distribution.values())

        total = sum(probs)
        probs = [p / total for p in probs]

        return self.rng.choice(buckets, size=num_rules, p=probs).tolist()


# =============================================================================
# Main Orchestrator
# =============================================================================

class CrossAppRuleGenerationOrchestrator:
    """Main orchestrator for cross-app rule generation."""

    def __init__(self,
                 users_file: Path,
                 apps_config: List[Dict],
                 entitlements_dir: Path,
                 output_file: Path,
                 config: CrossAppRuleGeneratorConfig,
                 seed: int = 42):
        self.users_file = users_file
        self.apps_config = apps_config
        self.entitlements_dir = entitlements_dir
        self.output_file = output_file
        self.config = config
        self.seed = seed

        self.logger = logging.getLogger(self.__class__.__name__)
        self.rng = np.random.default_rng(seed)

        self.users_df: Optional[pd.DataFrame] = None
        self.entitlements_by_app: Dict[str, List[Dict]] = {}
        self.rules: List[Dict] = []

    def run(self):
        """Execute the rule generation pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("CROSS-APP RULE GENERATION - STARTING")
        self.logger.info("=" * 60)

        self._load_users()
        self._load_entitlements()
        self._generate_rules()
        self._save_rules()

        self.logger.info("=" * 60)
        self.logger.info("CROSS-APP RULE GENERATION - COMPLETE")
        self.logger.info("=" * 60)

    def _load_users(self):
        """Load user data from CSV."""
        self.logger.info(f"Loading users from {self.users_file}")

        if not self.users_file.exists():
            raise FileNotFoundError(f"Users file not found: {self.users_file}")

        self.users_df = pd.read_csv(self.users_file)
        self.logger.info(f"Loaded {len(self.users_df)} users with {len(self.users_df.columns)} columns")

    def _load_entitlements(self):
        """Load entitlements for each application."""
        self.logger.info("Loading entitlements for applications")

        for app in self.apps_config:
            app_name = app['app_name']
            ent_file = self.entitlements_dir / f"{app_name}_entitlements.csv"

            if not ent_file.exists():
                self.logger.warning(f"Entitlements file not found: {ent_file}")
                continue

            ent_df = pd.read_csv(ent_file)
            entitlements = []
            for _, row in ent_df.iterrows():
                entitlements.append({
                    'entitlement_id': row.get('entitlement_id', ''),
                    'entitlement_name': row.get('entitlement_name', ''),
                    'entitlement_type': row.get('entitlement_type', 'standard')
                })

            self.entitlements_by_app[app_name] = entitlements
            self.logger.info(f"Loaded {len(entitlements)} entitlements for {app_name}")

    def _generate_rules(self):
        """Generate cross-app rules."""
        self.logger.info("Generating cross-app rules")

        generator = CrossAppRuleGenerator(
            users_df=self.users_df,
            apps=self.apps_config,
            entitlements_by_app=self.entitlements_by_app,
            config=self.config,
            rng=self.rng
        )

        self.rules = generator.generate_all_rules()
        self.logger.info(f"Generated {len(self.rules)} cross-app rules")

    def _save_rules(self):
        """Save rules to JSON file."""
        self.logger.info(f"Saving rules to {self.output_file}")

        # Convert numpy types to native Python types
        def convert_to_native_types(obj):
            import numpy as np

            if isinstance(obj, dict):
                return {k: convert_to_native_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        rules_native = convert_to_native_types(self.rules)

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_file, 'w') as f:
            json.dump(rules_native, f, indent=2)

        self.logger.info(f"Rules saved successfully")
        self.logger.info(f"  Format: List of {len(rules_native)} cross-app rules")


# =============================================================================
# CLI
# =============================================================================

def setup_logging(level: str = 'INFO'):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(name)-25s - %(levelname)-8s - %(message)s'
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Cross-App Rule Generator for IGA Synthetic Data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--users', '-u', required=True, type=Path,
                        help='Path to users CSV file (identities.csv)')
    parser.add_argument('--entitlements-dir', '-e', required=True, type=Path,
                        help='Directory containing entitlement CSV files')
    parser.add_argument('--output', '-o', default=Path('cross_app_rules.json'), type=Path,
                        help='Output file for generated rules')
    parser.add_argument('--apps', nargs='+', default=['AWS', 'Salesforce', 'ServiceNow', 'Epic', 'SAP'],
                        help='List of application names')
    parser.add_argument('--num-rules', '-n', type=int, default=10,
                        help='Number of cross-app rules to generate')
    parser.add_argument('--apps-per-rule-min', type=int, default=2,
                        help='Minimum apps per rule')
    parser.add_argument('--apps-per-rule-max', type=int, default=4,
                        help='Maximum apps per rule')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose (DEBUG) logging')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level)

    # Build apps config
    apps_config = [{'app_name': app, 'app_id': f'APP_{app.upper()}'}
                   for app in args.apps]

    # Create config
    config = CrossAppRuleGeneratorConfig(
        num_cross_app_rules=args.num_rules,
        apps_per_rule_min=args.apps_per_rule_min,
        apps_per_rule_max=args.apps_per_rule_max
    )

    # Run orchestrator
    orchestrator = CrossAppRuleGenerationOrchestrator(
        users_file=args.users,
        apps_config=apps_config,
        entitlements_dir=args.entitlements_dir,
        output_file=args.output,
        config=config,
        seed=args.seed
    )

    orchestrator.run()


if __name__ == '__main__':
    main()