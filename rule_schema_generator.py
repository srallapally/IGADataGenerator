#!/usr/bin/env python3
"""
RuleSchemaGenerator - Schema-Based Rule Generation

Generates association rules based on abstract feature schemas,
WITHOUT requiring actual identity data. This prevents overfitting.

Key Difference from DynamicRuleGenerator:
- OLD: Analyzes existing identities → generates rules
- NEW: Uses abstract schema → generates rules → identities generated to match
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from numpy.random import Generator

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

    # Coordinated cross-app rule generation
    coordinate_rules_across_apps: bool = False
    num_unique_feature_patterns: Optional[int] = None

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



@dataclass
class FeatureSchema:
    """Abstract definition of a feature's characteristics."""
    name: str
    type: str  # 'categorical', 'numeric', 'boolean'
    values: List[str]  # Possible values (for categorical)
    cardinality: int  # Number of unique values
    distribution: Dict[str, float]  # Value probabilities


class RuleSchemaGenerator:
    """
    Generates rules based on abstract feature schemas.

    Does NOT require actual identity data - works from schema definitions.
    This eliminates the circular dependency and overfitting problem.
    """

    def __init__(self,
                 apps: List[Dict[str, any]],
                 entitlements_by_app: Dict[str, List[Dict[str, str]]],
                 feature_schema: Dict[str, Dict[str, any]],
                 config: 'RuleGeneratorConfig',
                 rng: Generator):
        self.apps = apps
        self.entitlements_by_app = entitlements_by_app
        self.feature_schema = feature_schema
        self.config = config
        self.rng = rng
        self.logger = logging.getLogger(self.__class__.__name__)

        # Parse schema into structured objects
        self.features = self._parse_feature_schema()

        # Track used combinations
        self.used_combinations: Set[Tuple] = set()

    def _parse_feature_schema(self) -> Dict[str, FeatureSchema]:
        """Convert raw schema dict into FeatureSchema objects."""
        # BEGIN NEW METHOD
        features = {}

        for name, schema in self.feature_schema.items():
            if schema.get('type') != 'categorical':
                continue  # Only categorical features for now

            values = schema.get('values', [])
            cardinality = schema.get('cardinality', len(values))

            # Handle distribution
            dist_config = schema.get('distribution', 'uniform')
            if isinstance(dist_config, dict):
                distribution = dist_config
            elif dist_config == 'uniform':
                distribution = {v: 1.0 / len(values) for v in values}
            else:
                distribution = {v: 1.0 / len(values) for v in values}

            features[name] = FeatureSchema(
                name=name,
                type=schema.get('type', 'categorical'),
                values=values,
                cardinality=cardinality,
                distribution=distribution
            )

        self.logger.info(f"Parsed {len(features)} features from schema")
        return features
        # END NEW METHOD

    def generate_all_rules(self) -> List[Dict]:
        """Generate rules for all applications using schema."""
        # BEGIN NEW METHOD
        all_rules = []
        rule_counter = 1

        coordinate_mode = getattr(self.config, 'coordinate_rules_across_apps', False)

        if coordinate_mode:
            # Generate shared patterns first
            num_patterns = getattr(self.config, 'num_unique_feature_patterns',
                                   self.config.num_rules_per_app)
            if num_patterns is None:
                num_patterns = self.config.num_rules_per_app
            if isinstance(num_patterns, dict):
                num_patterns = num_patterns.get('value', self.config.num_rules_per_app)
            num_patterns = int(num_patterns)

            shared_patterns = self._generate_shared_feature_patterns(num_patterns)

            # Generate rules for each app using shared patterns
            for app in self.apps:
                app_name = app['app_name']
                entitlements = self.entitlements_by_app.get(app_name, [])

                if not entitlements:
                    self.logger.warning(f"No entitlements for {app_name}, skipping")
                    continue

                app_rules = self._generate_rules_from_patterns(
                    app_name=app_name,
                    entitlements=entitlements,
                    feature_patterns=shared_patterns,
                    start_id=rule_counter
                )

                all_rules.extend(app_rules)
                rule_counter += len(app_rules)
        else:
            # Independent mode - generate unique rules per app
            for app in self.apps:
                app_name = app['app_name']
                entitlements = self.entitlements_by_app.get(app_name, [])

                if not entitlements:
                    continue

                app_rules = self._generate_rules_for_app(
                    app_name=app_name,
                    entitlements=entitlements,
                    num_rules=self.config.num_rules_per_app,
                    start_id=rule_counter
                )

                all_rules.extend(app_rules)
                rule_counter += len(app_rules)

        self.logger.info(f"Generated {len(all_rules)} schema-based rules")
        return all_rules
        # END NEW METHOD

    def _generate_rules_for_app(self,
                                app_name: str,
                                entitlements: List[Dict[str, str]],
                                num_rules: int,
                                start_id: int) -> List[Dict]:
        """Generate rules for one app using feature schema."""
        # BEGIN NEW METHOD
        rules = []
        confidence_buckets = self._sample_confidence_buckets(num_rules)

        for i in range(num_rules):
            rule_id = f"R{start_id + i:03d}"
            confidence_bucket = confidence_buckets[i]

            rule = self._generate_single_rule_from_schema(
                rule_id=rule_id,
                app_name=app_name,
                entitlements=entitlements,
                confidence_bucket=confidence_bucket
            )

            if rule:
                rules.append(rule)
            else:
                self.logger.warning(f"Could not generate rule {rule_id}")

        return rules
        # END NEW METHOD

    def _generate_single_rule_from_schema(self,
                                          rule_id: str,
                                          app_name: str,
                                          entitlements: List[Dict[str, str]],
                                          confidence_bucket: str) -> Optional[Dict]:
        """Generate a single rule using feature schema."""
        # BEGIN NEW METHOD
        max_attempts = 50

        for attempt in range(max_attempts):
            # Select feature combination
            antecedent = self._select_feature_combination_from_schema()

            if not antecedent:
                continue

            # Check uniqueness
            combo_key = tuple(sorted(antecedent.items()))
            if combo_key in self.used_combinations:
                continue

            # Estimate expected population (from schema distributions)
            expected_support = self._calculate_expected_support(antecedent)

            # Sample metrics
            conf_range = self.config.confidence_ranges[confidence_bucket]
            confidence = self.rng.uniform(conf_range[0], conf_range[1])

            support_min, support_max = self.config.support_range
            support = self.rng.uniform(
                max(support_min, expected_support * 0.5),
                min(support_max, expected_support * 1.5)
            )

            cramers_v = self.rng.uniform(
                self.config.cramers_v_range[0],
                self.config.cramers_v_range[1]
            )

            # Select entitlements
            num_ents = self.rng.integers(
                self.config.min_entitlements_per_rule,
                self.config.max_entitlements_per_rule + 1
            )
            num_ents = min(num_ents, len(entitlements))

            selected_ents = self.rng.choice(entitlements, size=num_ents, replace=False)
            entitlement_ids = [e['entitlement_id'] for e in selected_ents]

            # Create rule
            antecedent_markers = [
                f"FEATURE:{key}={value}"
                for key, value in antecedent.items()
            ]
            # Create rule in AssociationRule-compatible format
            rule = {
                'rule_id': rule_id,
                'app_name': app_name,
                'description': self._create_description(antecedent, app_name, selected_ents),
                'antecedent_entitlements': antecedent_markers,  # Changed from 'antecedent'
                'consequent_entitlements': entitlement_ids,  # Changed from nested dict
                'support': round(support, 3),  # Moved from 'strength' dict
                'confidence': round(confidence, 3),  # Moved from 'strength' dict
                'lift': 1.0,  # Added default lift value
                'metadata': {
                    'confidence_bucket': confidence_bucket,
                    'schema_based': True,
                    'target_cramers_v': round(cramers_v, 3)  # Moved from 'strength' dict
                }
            }

            self.used_combinations.add(combo_key)
            return rule

        return None
        # END NEW METHOD

    def _select_feature_combination_from_schema(self) -> Dict[str, str]:
        """Select random feature combination using schema."""
        # BEGIN NEW METHOD
        if not self.features:
            return {}

        num_features = self.rng.integers(
            self.config.min_features_per_rule,
            self.config.max_features_per_rule + 1
        )
        num_features = min(num_features, len(self.features))

        # Select features
        feature_names = list(self.features.keys())
        selected_names = self.rng.choice(feature_names, size=num_features, replace=False)

        # Select values for each feature based on schema distribution
        antecedent = {}
        for name in selected_names:
            feature = self.features[name]

            # Sample value based on distribution
            values = list(feature.distribution.keys())
            probs = list(feature.distribution.values())

            # Normalize probabilities
            total = sum(probs)
            probs = [p / total for p in probs]

            value = self.rng.choice(values, p=probs)
            antecedent[name] = str(value)

        return antecedent
        # END NEW METHOD

    def _calculate_expected_support(self, antecedent: Dict[str, str]) -> float:
        """
        Calculate expected support based on feature distributions in schema.

        support ≈ P(feature1=value1) × P(feature2=value2) × ...
        """
        # BEGIN NEW METHOD
        probability = 1.0

        for feature_name, value in antecedent.items():
            feature = self.features.get(feature_name)
            if feature and value in feature.distribution:
                probability *= feature.distribution[value]
            else:
                # Default to 1/cardinality if not in distribution
                probability *= 1.0 / feature.cardinality if feature else 0.1

        return probability
        # END NEW METHOD

    def _generate_shared_feature_patterns(self, num_patterns: int) -> List[Dict[str, str]]:
        """Generate shared patterns for coordinated cross-app rules."""
        # BEGIN NEW METHOD
        patterns = []
        attempts = 0
        max_attempts = num_patterns * 20

        while len(patterns) < num_patterns and attempts < max_attempts:
            attempts += 1

            pattern = self._select_feature_combination_from_schema()

            if not pattern:
                continue

            # Check uniqueness
            pattern_tuple = tuple(sorted(pattern.items()))
            is_duplicate = any(
                tuple(sorted(p.items())) == pattern_tuple
                for p in patterns
            )

            if not is_duplicate:
                patterns.append(pattern)

        self.logger.info(f"Generated {len(patterns)} shared patterns from schema")
        return patterns
        # END NEW METHOD

    def _generate_rules_from_patterns(self,
                                      app_name: str,
                                      entitlements: List[Dict[str, str]],
                                      feature_patterns: List[Dict[str, str]],
                                      start_id: int) -> List[Dict]:
        """
        Generate rules for an app using pre-defined feature patterns.

        Args:
            app_name: Name of the application
            entitlements: Available entitlements for this app
            feature_patterns: Pre-generated feature combinations to use
            start_id: Starting rule ID number

        Returns:
            List of generated rules
        """
        # BEGIN COMPLETE IMPLEMENTATION
        rules = []

        # Sample confidence buckets
        num_patterns = len(feature_patterns)
        confidence_buckets = self._sample_confidence_buckets(num_patterns)

        for i, pattern in enumerate(feature_patterns):
            rule_id = f"R{start_id + i:03d}"
            confidence_bucket = confidence_buckets[i]

            # Calculate expected support from schema
            expected_support = self._calculate_expected_support(pattern)

            # Sample confidence
            conf_range = self.config.confidence_ranges[confidence_bucket]
            confidence = self.rng.uniform(conf_range[0], conf_range[1])

            # Sample support around expected value
            support_min, support_max = self.config.support_range
            actual_min_support = max(support_min, expected_support * 0.5)
            actual_max_support = min(support_max, expected_support * 1.5)

            if actual_min_support >= actual_max_support:
                actual_max_support = actual_min_support + 0.01

            support = self.rng.uniform(actual_min_support, actual_max_support)

            # Sample Cramér's V
            cramers_v = self.rng.uniform(
                self.config.cramers_v_range[0],
                self.config.cramers_v_range[1]
            )

            # Select entitlements
            num_entitlements = self.rng.integers(
                self.config.min_entitlements_per_rule,
                self.config.max_entitlements_per_rule + 1
            )
            num_entitlements = min(num_entitlements, len(entitlements))

            selected_ents = self.rng.choice(entitlements, size=num_entitlements, replace=False)
            entitlement_ids = [e.entitlement_id for e in selected_ents]

            # Create description
            description = self._create_description(
                antecedent=pattern,
                app_name=app_name,
                entitlements=selected_ents
            )

            antecedent_markers = [
                f"FEATURE:{key}={value}"
                for key, value in pattern.items()
            ]

            # Build rule in AssociationRule-compatible format
            rule = {
                'rule_id': rule_id,
                'app_name': app_name,
                'description': description,
                'antecedent_entitlements': antecedent_markers,  # Changed from 'antecedent'
                'consequent_entitlements': entitlement_ids,  # Changed from nested dict
                'support': round(support, 3),  # Moved from 'strength' dict
                'confidence': round(confidence, 3),  # Moved from 'strength' dict
                'lift': 1.0,  # Added default lift value
                'metadata': {
                    'confidence_bucket': confidence_bucket,
                    'expected_support': round(expected_support, 4),
                    'coordinated': True,  # Flag to indicate shared pattern
                    'schema_based': True,
                    'target_cramers_v': round(cramers_v, 3)  # Moved from 'strength' dict
                }
            }

            rules.append(rule)
            self.logger.debug(f"Created rule {rule_id} for {app_name} using pattern: {pattern}")

        return rules
        # END COMPLETE IMPLEMENTATION

    def _sample_confidence_buckets(self, num_rules: int) -> List[str]:
        """Sample confidence buckets according to distribution."""
        buckets = list(self.config.confidence_distribution.keys())
        probs = list(self.config.confidence_distribution.values())
        total = sum(probs)
        probs = [p / total for p in probs]
        return self.rng.choice(buckets, size=num_rules, p=probs).tolist()

    def _create_description(self, antecedent: Dict[str, str],
                            app_name: str,
                            entitlements: List[Dict[str, str]]) -> str:
        """Create human-readable description."""
        conditions = [f"{k}='{v}'" for k, v in antecedent.items()]
        ant_desc = " AND ".join(conditions)

        if len(entitlements) == 1:
            ent_desc = getattr(entitlements[0], 'entitlement_name', entitlements[0].entitlement_id)
        else:
            ent_desc = f"{len(entitlements)} entitlements"

        return f"Users with {ant_desc} get {ent_desc} in {app_name}"