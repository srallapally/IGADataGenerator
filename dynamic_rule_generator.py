#!/usr/bin/env python3
"""
dynamic_rule_generator.py

Dynamically generates feature-to-entitlement association rules based on:
- User population and their attributes
- Available entitlements per application
- Target statistical distributions (confidence, support, Cramér's V)

This inverts the data generation pipeline: instead of generating random users
and hoping for statistical significance, we define rules with target strength
and then generate users to match those rules.
"""

import argparse
import json
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np
from numpy.random import Generator


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class RuleGeneratorConfig:
    """Configuration for dynamic rule generation."""
    num_rules_per_app: int = 5
    confidence_distribution: Dict[str, float] = None  # {bucket: probability}
    confidence_ranges: Dict[str, Tuple[float, float]] = None
    support_range: Tuple[float, float] = (0.01, 0.20)
    cramers_v_range: Tuple[float, float] = (0.30, 0.50)

    # Feature selection
    min_features_per_rule: int = 1
    max_features_per_rule: int = 3

    # Entitlement selection
    min_entitlements_per_rule: int = 1
    max_entitlements_per_rule: int = 4

    def __post_init__(self):
        if self.confidence_distribution is None:
            self.confidence_distribution = {
                'high': 0.40,  # 40% high confidence rules
                'medium': 0.35,  # 35% medium confidence
                'low': 0.25  # 25% low confidence
            }

        if self.confidence_ranges is None:
            self.confidence_ranges = {
                'high': (0.85, 0.95),
                'medium': (0.75, 0.84),
                'low': (0.65, 0.74)
            }


# =============================================================================
# Feature Analyzer
# =============================================================================

class FeatureAnalyzer:
    """Analyzes user data to identify good candidate features for rules."""

    def __init__(self, users_df: pd.DataFrame, rng: Generator):
        self.users_df = users_df
        self.rng = rng
        self.logger = logging.getLogger(self.__class__.__name__)

        # Analyze features
        self.feature_stats = self._analyze_features()

    def _analyze_features(self) -> Dict[str, Dict]:
        """Analyze each feature's characteristics."""
        stats = {}

        for col in self.users_df.columns:
            if col in ['user_id', 'user_name', 'email', 'first_name', 'last_name',
                       'employee_id', 'hire_date', 'tenure_years']:
                continue  # Skip identifier columns

            unique_vals = self.users_df[col].nunique()
            total_vals = len(self.users_df)

            # Get value distribution
            value_counts = self.users_df[col].value_counts()
            top_values = value_counts.head(10).to_dict()

            stats[col] = {
                'unique_count': unique_vals,
                'cardinality_ratio': unique_vals / total_vals,
                'top_values': top_values,
                'is_categorical': unique_vals < total_vals * 0.5,  # Heuristic
                'entropy': self._calculate_entropy(value_counts / total_vals)
            }

        return stats

    def _calculate_entropy(self, probabilities: pd.Series) -> float:
        """Calculate Shannon entropy for a distribution."""
        probs = probabilities[probabilities > 0]
        return -np.sum(probs * np.log2(probs))

    def get_good_features(self, min_cardinality: int = 2, max_cardinality: int = 50) -> List[str]:
        """
        Get features suitable for rule generation.

        Good features:
        - Have 2+ unique values (not constant)
        - Have < 50 unique values (not too high cardinality)
        - Are categorical in nature
        """
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
# Dynamic Rule Generator
# =============================================================================

class DynamicRuleGenerator:
    """Generates association rules dynamically based on configuration."""

    def __init__(self,
                 users_df: pd.DataFrame,
                 apps: List[Dict[str, any]],
                 entitlements_by_app: Dict[str, List[Dict[str, str]]],
                 config: RuleGeneratorConfig,
                 rng: Generator):
        self.users_df = users_df
        self.apps = apps
        self.entitlements_by_app = entitlements_by_app
        self.config = config
        self.rng = rng
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize feature analyzer
        self.feature_analyzer = FeatureAnalyzer(users_df, rng)
        self.good_features = self.feature_analyzer.get_good_features()

        # Track used combinations to avoid duplicates
        self.used_combinations: Set[Tuple] = set()

    def generate_all_rules(self) -> List[Dict]:
        """Generate rules for all applications."""
        all_rules = []
        rule_counter = 1

        for app in self.apps:
            app_name = app['app_name']

            if app_name not in self.entitlements_by_app:
                self.logger.warning(f"No entitlements found for {app_name}, skipping")
                continue

            entitlements = self.entitlements_by_app[app_name]

            if not entitlements:
                self.logger.warning(f"Empty entitlement list for {app_name}, skipping")
                continue

            self.logger.info(f"Generating {self.config.num_rules_per_app} rules for {app_name}")

            app_rules = self._generate_rules_for_app(
                app_name=app_name,
                entitlements=entitlements,
                num_rules=self.config.num_rules_per_app,
                start_id=rule_counter
            )

            all_rules.extend(app_rules)
            rule_counter += len(app_rules)

        self.logger.info(f"Generated {len(all_rules)} total rules across {len(self.apps)} apps")
        return all_rules

    def _generate_rules_for_app(self,
                                app_name: str,
                                entitlements: List[Dict[str, str]],
                                num_rules: int,
                                start_id: int) -> List[Dict]:
        """Generate rules for a single application."""
        rules = []

        # Determine confidence bucket distribution for this app
        confidence_buckets = self._sample_confidence_buckets(num_rules)

        for i in range(num_rules):
            rule_id = f"R{start_id + i:03d}"
            confidence_bucket = confidence_buckets[i]

            # Generate rule
            rule = self._generate_single_rule(
                rule_id=rule_id,
                app_name=app_name,
                entitlements=entitlements,
                confidence_bucket=confidence_bucket
            )

            if rule:
                rules.append(rule)

        return rules

    def _sample_confidence_buckets(self, num_rules: int) -> List[str]:
        """Sample confidence buckets according to distribution."""
        buckets = list(self.config.confidence_distribution.keys())
        probs = list(self.config.confidence_distribution.values())

        # Normalize probabilities
        total = sum(probs)
        probs = [p / total for p in probs]

        return self.rng.choice(buckets, size=num_rules, p=probs).tolist()

    def _generate_single_rule(self,
                              rule_id: str,
                              app_name: str,
                              entitlements: List[Dict[str, str]],
                              confidence_bucket: str) -> Optional[Dict]:
        """Generate a single rule."""
        max_attempts = 50

        for attempt in range(max_attempts):
            # Step 1: Select feature combination (antecedent)
            antecedent = self._select_feature_combination()

            if not antecedent:
                continue

            # Check if this combination was already used
            combo_key = tuple(sorted(antecedent.items()))
            if combo_key in self.used_combinations:
                continue

            # Step 2: Estimate population matching this antecedent
            matching_population = self.feature_analyzer.estimate_population_with_features(antecedent)
            total_population = len(self.users_df)

            if matching_population < 5:  # Need at least 5 users for statistical validity
                continue

            # Step 3: Sample confidence, support, and Cramér's V
            conf_range = self.config.confidence_ranges[confidence_bucket]
            confidence = self.rng.uniform(conf_range[0], conf_range[1])

            # Support must be achievable given population
            max_support = matching_population / total_population
            support_min, support_max = self.config.support_range
            support = self.rng.uniform(
                max(support_min, 0.01),
                min(support_max, max_support * 0.9)  # Leave some room for noise
            )

            cramers_v = self.rng.uniform(
                self.config.cramers_v_range[0],
                self.config.cramers_v_range[1]
            )

            # Step 4: Select entitlements (consequent)
            num_entitlements = self.rng.integers(
                self.config.min_entitlements_per_rule,
                self.config.max_entitlements_per_rule + 1
            )
            num_entitlements = min(num_entitlements, len(entitlements))

            selected_ents = self.rng.choice(entitlements, size=num_entitlements, replace=False)
            entitlement_ids = [e['entitlement_id'] for e in selected_ents]

            # Step 5: Create description
            description = self._create_rule_description(
                antecedent=antecedent,
                app_name=app_name,
                entitlements=selected_ents
            )

            # Step 6: Build rule
            rule = {
                'rule_id': rule_id,
                'app_name': app_name,
                'description': description,
                'antecedent': antecedent,
                'consequent': {
                    'entitlements': entitlement_ids
                },
                'strength': {
                    'confidence': round(confidence, 3),
                    'support': round(support, 3),
                    'target_cramers_v': round(cramers_v, 3)
                },
                'metadata': {
                    'confidence_bucket': confidence_bucket,
                    'estimated_matching_users': matching_population,
                    'total_users': total_population
                }
            }

            # Mark combination as used
            self.used_combinations.add(combo_key)

            return rule

        self.logger.warning(f"Failed to generate rule {rule_id} after {max_attempts} attempts")
        return None

    def _select_feature_combination(self) -> Optional[Dict[str, str]]:
        """Select a random combination of features and values."""
        if not self.good_features:
            return None

        # Decide how many features to use
        num_features = self.rng.integers(
            self.config.min_features_per_rule,
            self.config.max_features_per_rule + 1
        )
        num_features = min(num_features, len(self.good_features))

        # Select features
        selected_features = self.rng.choice(
            self.good_features,
            size=num_features,
            replace=False
        )

        # Select values for each feature
        antecedent = {}
        for feature in selected_features:
            values = self.feature_analyzer.get_feature_values(feature)
            if values:
                value = self.rng.choice(values)
                antecedent[feature] = str(value)

        return antecedent if antecedent else None

    def _create_rule_description(self,
                                 antecedent: Dict[str, str],
                                 app_name: str,
                                 entitlements: List[Dict[str, str]]) -> str:
        """Create a human-readable description for the rule."""
        # Build antecedent description
        conditions = []
        for feature, value in antecedent.items():
            conditions.append(f"{feature}='{value}'")

        antecedent_desc = " AND ".join(conditions)

        # Build consequent description
        if len(entitlements) == 1:
            ent_desc = entitlements[0].get('entitlement_name', entitlements[0]['entitlement_id'])
        elif len(entitlements) <= 3:
            names = [e.get('entitlement_name', e['entitlement_id']) for e in entitlements]
            ent_desc = ", ".join(names)
        else:
            ent_desc = f"{len(entitlements)} entitlements"

        return f"Users with {antecedent_desc} get {ent_desc} in {app_name}"


# =============================================================================
# Rule Validator
# =============================================================================

class RuleValidator:
    """Validates that generated rules are achievable."""

    def __init__(self, users_df: pd.DataFrame):
        self.users_df = users_df
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate_rules(self, rules: List[Dict]) -> Dict[str, any]:
        """Validate all rules and return validation report."""
        results = {
            'valid': True,
            'total_rules': len(rules),
            'errors': [],
            'warnings': [],
            'rule_details': []
        }

        total_users = len(self.users_df)

        for rule in rules:
            rule_id = rule['rule_id']
            antecedent = rule['antecedent']
            strength = rule['strength']

            # Count users matching antecedent
            matching = self._count_matching_users(antecedent)

            # Calculate required populations
            support = strength['support']
            confidence = strength['confidence']

            # support = P(A ∩ C) = users_with_both / total_users
            users_with_both = int(support * total_users)

            # confidence = P(C|A) = users_with_both / users_matching_antecedent
            # Therefore: users_matching_antecedent = users_with_both / confidence
            required_matching = int(users_with_both / confidence) if confidence > 0 else 0

            rule_detail = {
                'rule_id': rule_id,
                'actual_matching_users': matching,
                'required_matching_users': required_matching,
                'users_with_both_needed': users_with_both,
                'achievable': matching >= required_matching
            }

            results['rule_details'].append(rule_detail)

            # Validation checks
            if matching < required_matching:
                msg = (f"Rule {rule_id}: Requires {required_matching} users matching antecedent, "
                       f"but only {matching} exist")
                results['errors'].append(msg)
                results['valid'] = False
            elif matching < required_matching * 1.2:
                msg = (f"Rule {rule_id}: Only {matching} users match (need {required_matching}). "
                       f"Low margin for noise.")
                results['warnings'].append(msg)

            if support > 0.5:
                msg = f"Rule {rule_id}: High support ({support:.2%}) may cause population overlap issues"
                results['warnings'].append(msg)

        # Summary
        achievable = sum(1 for r in results['rule_details'] if r['achievable'])
        results['achievable_rules'] = achievable
        results['achievable_pct'] = achievable / len(rules) if rules else 0

        self.logger.info(f"Validation: {achievable}/{len(rules)} rules achievable")

        return results

    def _count_matching_users(self, antecedent: Dict[str, str]) -> int:
        """Count users matching the antecedent conditions."""
        mask = pd.Series([True] * len(self.users_df))

        for feature, value in antecedent.items():
            if feature in self.users_df.columns:
                mask &= (self.users_df[feature].astype(str) == str(value))

        return mask.sum()


# =============================================================================
# Main Orchestrator
# =============================================================================

class RuleGenerationOrchestrator:
    """Main orchestrator for dynamic rule generation."""

    def __init__(self,
                 users_file: Path,
                 apps_config: List[Dict],
                 entitlements_dir: Path,
                 output_file: Path,
                 config: RuleGeneratorConfig,
                 seed: int = 42):
        self.users_file = users_file
        self.apps_config = apps_config
        self.entitlements_dir = entitlements_dir
        self.output_file = output_file
        self.config = config
        self.seed = seed

        self.logger = logging.getLogger(self.__class__.__name__)
        self.rng = np.random.default_rng(seed)

        # Data
        self.users_df: Optional[pd.DataFrame] = None
        self.entitlements_by_app: Dict[str, List[Dict]] = {}
        self.rules: List[Dict] = []

    def run(self):
        """Execute the rule generation pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("DYNAMIC RULE GENERATION - STARTING")
        self.logger.info("=" * 60)

        # Step 1: Load users
        self._load_users()

        # Step 2: Load entitlements
        self._load_entitlements()

        # Step 3: Generate rules
        self._generate_rules()

        # Step 4: Validate rules
        self._validate_rules()

        # Step 5: Save rules
        self._save_rules()

        self.logger.info("=" * 60)
        self.logger.info("DYNAMIC RULE GENERATION - COMPLETE")
        self.logger.info("=" * 60)

    def _load_users(self):
        """Load user data from CSV."""
        self.logger.info(f"Loading users from {self.users_file}")

        if not self.users_file.exists():
            raise FileNotFoundError(f"Users file not found: {self.users_file}")

        self.users_df = pd.read_csv(self.users_file)
        self.logger.info(f"Loaded {len(self.users_df)} users with {len(self.users_df.columns)} columns")
        self.logger.info(f"Columns: {list(self.users_df.columns)}")

    def _load_entitlements(self):
        """Load entitlements for each application."""
        self.logger.info("Loading entitlements for applications")

        for app in self.apps_config:
            app_name = app['app_name']

            # Try to find entitlements file
            ent_file = self.entitlements_dir / f"{app_name}_entitlements.csv"

            if not ent_file.exists():
                self.logger.warning(f"Entitlements file not found: {ent_file}")
                continue

            # Load entitlements
            ent_df = pd.read_csv(ent_file)

            # Convert to list of dicts
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
        """Generate rules dynamically."""
        self.logger.info("Generating rules")

        generator = DynamicRuleGenerator(
            users_df=self.users_df,
            apps=self.apps_config,
            entitlements_by_app=self.entitlements_by_app,
            config=self.config,
            rng=self.rng
        )

        self.rules = generator.generate_all_rules()
        self.logger.info(f"Generated {len(self.rules)} rules")

    def _validate_rules(self):
        """Validate generated rules."""
        self.logger.info("Validating rules")

        validator = RuleValidator(self.users_df)
        validation = validator.validate_rules(self.rules)

        self.logger.info(f"Validation: {validation['achievable_rules']}/{validation['total_rules']} "
                         f"rules achievable ({validation['achievable_pct']:.1%})")

        if validation['errors']:
            self.logger.error("Validation errors:")
            for error in validation['errors'][:5]:  # Show first 5
                self.logger.error(f"  {error}")

        if validation['warnings']:
            self.logger.warning("Validation warnings:")
            for warning in validation['warnings'][:5]:  # Show first 5
                self.logger.warning(f"  {warning}")

    def _save_rules(self):
        """Save rules to JSON file."""
        self.logger.info(f"Saving rules to {self.output_file}")

        output = {
            'rules': self.rules,
            'metadata': {
                'total_identities': len(self.users_df),
                'total_rules': len(self.rules),
                'cramers_v_tolerance': 0.05,
                'generation_seed': self.seed,
                'applications': [app['app_name'] for app in self.apps_config],
                'generation_config': {
                    'num_rules_per_app': self.config.num_rules_per_app,
                    'confidence_distribution': self.config.confidence_distribution,
                    'support_range': self.config.support_range,
                    'cramers_v_range': self.config.cramers_v_range
                }
            }
        }

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_file, 'w') as f:
            json.dump(output, f, indent=2)

        self.logger.info(f"Rules saved successfully")


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
        description='Dynamic Rule Generator for IGA Synthetic Data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--users', '-u', required=True, type=Path,
                        help='Path to users CSV file (identities.csv)')
    parser.add_argument('--entitlements-dir', '-e', required=True, type=Path,
                        help='Directory containing entitlement CSV files')
    parser.add_argument('--output', '-o', default=Path('generated_rules.json'), type=Path,
                        help='Output file for generated rules')
    parser.add_argument('--apps', nargs='+', default=['AWS', 'Salesforce', 'ServiceNow', 'Epic'],
                        help='List of application names')
    parser.add_argument('--num-rules', '-n', type=int, default=5,
                        help='Number of rules to generate per application')
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
    config = RuleGeneratorConfig(
        num_rules_per_app=args.num_rules
    )

    # Run orchestrator
    orchestrator = RuleGenerationOrchestrator(
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