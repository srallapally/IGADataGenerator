#!/usr/bin/env python3
"""
CrossAppRuleSchemaGenerator - Schema-Based Cross-App Rule Generation

Generates cross-app association rules based on abstract feature schemas,
WITHOUT requiring actual identity data.

Key difference from CrossAppRuleGenerator:
- OLD: Analyzes existing identities → generates cross-app rules
- NEW: Uses abstract schema → generates cross-app rules
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from numpy.random import Generator


# Reuse the same config class structure from dynamic_rule_generator_cross_app
@dataclass
class CrossAppRuleGeneratorConfig:
    """Configuration for cross-app rule generation."""
    num_cross_app_rules: int = 10
    apps_per_rule_min: int = 2
    apps_per_rule_max: int = 4

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


@dataclass
class FeatureSchema:
    """Abstract definition of a feature's characteristics."""
    name: str
    type: str  # 'categorical', 'numeric', 'boolean'
    values: List[str]  # Possible values (for categorical)
    cardinality: int  # Number of unique values
    distribution: Dict[str, float]  # Value probabilities


class CrossAppRuleSchemaGenerator:
    """
    Generates cross-app rules based on abstract feature schemas.

    Does NOT require actual identity data - works from schema definitions.
    This eliminates the circular dependency and overfitting problem.
    """

    def __init__(self,
                 apps: List[Dict[str, any]],
                 entitlements_by_app: Dict[str, List[Dict[str, str]]],
                 feature_schema: Dict[str, Dict[str, any]],
                 config: CrossAppRuleGeneratorConfig,
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
        """Generate cross-app rules using schema."""
        # BEGIN NEW METHOD
        self.logger.info("=" * 60)
        self.logger.info("GENERATING CROSS-APP RULES FROM SCHEMA")
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
                rule = self._generate_single_cross_app_rule_from_schema(
                    rule_id=rule_id,
                    confidence_bucket=confidence_bucket
                )
                attempts += 1
                if rule is None:
                    failed_attempts += 1

            if rule:
                all_rules.append(rule)
                self.logger.info(
                    f"Generated {rule_id}: {len(rule['consequent'])} apps, "
                    f"feature pattern: {rule['antecedent']}"
                )
            else:
                self.logger.warning(
                    f"Could not generate rule {rule_id} after {attempts} attempts"
                )

        self.logger.info(f"Generated {len(all_rules)} cross-app schema-based rules")
        return all_rules
        # END NEW METHOD

    def _generate_single_cross_app_rule_from_schema(self,
                                                    rule_id: str,
                                                    confidence_bucket: str) -> Optional[Dict]:
        """Generate a single cross-app rule using feature schema."""
        # BEGIN NEW METHOD
        max_attempts = 50

        for attempt in range(max_attempts):
            # Step 1: Select feature combination (antecedent)
            antecedent = self._select_feature_combination_from_schema()

            if not antecedent:
                continue

            combo_key = tuple(sorted(antecedent.items()))
            if combo_key in self.used_combinations:
                continue

            # Step 2: Calculate expected support from schema
            expected_support = self._calculate_expected_support(antecedent)

            # Step 3: Select which apps this rule applies to
            num_apps = self.rng.integers(
                self.config.apps_per_rule_min,
                min(self.config.apps_per_rule_max + 1, len(self.apps) + 1)
            )
            selected_apps = self.rng.choice(self.apps, size=num_apps, replace=False)

            # Step 4: Sample confidence, support, Cramér's V
            conf_range = self.config.confidence_ranges[confidence_bucket]
            confidence = self.rng.uniform(conf_range[0], conf_range[1])

            support_min, support_max = self.config.support_range
            actual_min_support = max(support_min, expected_support * 0.3)
            actual_max_support = min(support_max, expected_support * 2.0)

            if actual_min_support >= actual_max_support:
                actual_max_support = actual_min_support + 0.01

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
                consequent[app_name] = [e.entitlement_id for e in selected_ents]

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
                    'expected_support': round(expected_support, 4),
                    'num_apps': len(consequent),
                    'apps': list(consequent.keys()),
                    'cross_app': True,
                    'schema_based': True
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

    def _create_cross_app_description(self,
                                      antecedent: Dict[str, str],
                                      app_names: List[str]) -> str:
        """Create a human-readable description for the cross-app rule."""
        # BEGIN NEW METHOD
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
        # END NEW METHOD

    def _sample_confidence_buckets(self, num_rules: int) -> List[str]:
        """Sample confidence buckets according to distribution."""
        buckets = list(self.config.confidence_distribution.keys())
        probs = list(self.config.confidence_distribution.values())

        total = sum(probs)
        probs = [p / total for p in probs]

        return self.rng.choice(buckets, size=num_rules, p=probs).tolist()