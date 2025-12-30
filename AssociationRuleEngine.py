import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class AssociationRule:
    """Simple in-memory representation of an association rule for one app."""
    id: str
    app_name: str
    antecedent_entitlements: List[str]
    consequent_entitlements: List[str]
    support: float
    confidence: float
    lift: float


class RuleEngine:
    """
    Loads pre-mined association rules from JSON and provides
    rule-based entitlement recommendations and coverage scores.

    Supports two rule formats:
    1. Per-app rules: Standard format with single app_name and entitlement list
    2. Cross-app rules: Rules that grant entitlements across multiple apps

    Cross-app rules are automatically decomposed into per-app rules during loading.
    """

    def __init__(self, rules_dir: Path, rng: np.random.Generator, logger: Optional[logging.Logger] = None):
        self.rules_dir = Path(rules_dir)
        self.rng = rng
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.rules_by_app: Dict[str, List[AssociationRule]] = {}

        self._load_all_rules()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _load_all_rules(self) -> None:
        """
        Load all *.json rule files in rules_dir.
        Expected file naming: <app_name>_rules.json OR generated_rules.json (cross-app)

        Handles two formats:
        1. Per-app format: consequent = {"entitlements": [...]}
        2. Cross-app format: consequent = {"SAP": [...], "AWS": [...]}

        Cross-app rules are decomposed (projected) into separate per-app rules.
        """
        if not self.rules_dir.exists():
            self.logger.warning(f"RuleEngine: rules directory does not exist: {self.rules_dir}")
            return

        # BEGIN MODIFICATION: Process all .json files, not just *_rules.json
        for path in self.rules_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw_rules = json.load(f)
            except Exception as e:
                self.logger.error(f"RuleEngine: failed to read {path}: {e}")
                continue

            if not isinstance(raw_rules, list):
                self.logger.warning(f"RuleEngine: rules file {path} is not a list; skipping")
                continue

            # Process each rule, detecting and handling format
            for r in raw_rules:
                try:
                    self._process_rule(r, path)
                except Exception as e:
                    self.logger.warning(f"RuleEngine: could not process rule from {path}: {e}")
        # END MODIFICATION

        # Log summary of loaded rules
        for app_name, rules in self.rules_by_app.items():
            self.logger.info(f"RuleEngine: loaded {len(rules)} rules for app '{app_name}'")

    def _process_rule(self, rule_dict: Dict[str, Any], source_path: Path) -> None:
        """
        Process a single rule, detecting format and normalizing to AssociationRule.

        Detects format by examining the 'consequent' field:
        - Cross-app: consequent is a dict with app names as keys
        - Per-app: consequent has 'entitlements' key or is from mined rules
        """
        # BEGIN NEW METHOD: Unified rule processing
        consequent = rule_dict.get("consequent", {})

        # Detect format by examining consequent structure
        if self._is_cross_app_format(consequent):
            # Cross-app format: decompose into per-app rules
            self._load_cross_app_rule(rule_dict)
        else:
            # Per-app format: load directly
            self._load_per_app_rule(rule_dict, source_path)
        # END NEW METHOD

    def _is_cross_app_format(self, consequent: Any) -> bool:
        """
        Determine if a rule is in cross-app format.

        Cross-app format: {"SAP": [...], "AWS": [...]}
        Per-app format: {"entitlements": [...]} or legacy mined format
        """
        # BEGIN NEW METHOD: Format detection
        if not isinstance(consequent, dict):
            return False

        # If it has 'entitlements' key, it's per-app format
        if "entitlements" in consequent:
            return False

        # If all keys look like app names (not 'entitlements'), it's cross-app
        # Cross-app rules have app names as keys, each mapping to a list
        if consequent:
            # Check if values are lists (expected for cross-app)
            all_lists = all(isinstance(v, list) for v in consequent.values())
            # Check if keys don't include standard per-app fields
            no_standard_fields = "entitlements" not in consequent
            return all_lists and no_standard_fields

        return False
        # END NEW METHOD

    def _load_cross_app_rule(self, rule_dict: Dict[str, Any]) -> None:
        """
        Decompose a cross-app rule into multiple per-app rules.

        Input format:
        {
          "rule_id": "R001",
          "antecedent": {"department": "Finance", "job_level": "Senior"},
          "consequent": {
            "SAP": ["role_1", "role_2"],
            "AWS": ["policy_1"],
            "Salesforce": ["license_1"]
          },
          "strength": {"confidence": 0.85, "support": 0.12, ...}
        }

        Creates separate AssociationRule objects for each app (SAP, AWS, Salesforce).
        Antecedent features are converted to pseudo-entitlement markers for matching.
        """
        # BEGIN NEW METHOD: Cross-app rule decomposition
        base_id = rule_dict.get("rule_id", rule_dict.get("id", "unknown"))
        antecedent = rule_dict.get("antecedent", {})
        consequent = rule_dict.get("consequent", {})
        strength = rule_dict.get("strength", {})

        # Convert antecedent features to pseudo-entitlement markers
        # Cross-app rules use identity features (department, job_level, etc.)
        # We encode these as special markers so they can be matched
        antecedent_markers = []
        if isinstance(antecedent, dict):
            antecedent_markers = [f"FEATURE:{k}={v}" for k, v in antecedent.items()]
        elif isinstance(antecedent, list):
            # Handle legacy format where antecedent might be a list
            antecedent_markers = list(antecedent)

        # Extract strength metrics
        support = float(strength.get("support", 0.0))
        confidence = float(strength.get("confidence", 0.0))
        lift = float(strength.get("lift", 1.0))

        # Project rule into per-app rules
        decomposed_count = 0
        for app_name, entitlements in consequent.items():
            if not isinstance(entitlements, list):
                self.logger.warning(
                    f"RuleEngine: Skipping app '{app_name}' in rule {base_id} "
                    f"(entitlements not a list)"
                )
                continue

            # Create unique ID for this app's projection
            projected_id = f"{base_id}_PROJ_{app_name}"

            rule = AssociationRule(
                id=projected_id,
                app_name=app_name,
                antecedent_entitlements=antecedent_markers,
                consequent_entitlements=entitlements,
                support=support,
                confidence=confidence,
                lift=lift,
            )

            # Add to the app's rule collection
            if app_name not in self.rules_by_app:
                self.rules_by_app[app_name] = []
            self.rules_by_app[app_name].append(rule)
            decomposed_count += 1

        if decomposed_count > 0:
            self.logger.debug(
                f"RuleEngine: Decomposed cross-app rule {base_id} into "
                f"{decomposed_count} per-app rules"
            )
        # END NEW METHOD

    def _load_per_app_rule(self, rule_dict: Dict[str, Any], source_path: Path) -> None:
        """
        Load a standard per-app rule.

        Handles two per-app formats:
        1. Dynamic generator format: consequent = {"entitlements": [...]}
        2. Mined rules format: antecedent_entitlements and consequent_entitlements at top level
        """
        # BEGIN NEW METHOD: Per-app rule loading
        # Infer app_name from file name as fallback
        default_app_name = source_path.name.replace("_rules.json", "").replace(".json", "")

        # Allow rule to specify app_name
        app_name = rule_dict.get("app_name", default_app_name)

        # Extract rule ID
        rule_id = str(rule_dict.get("id", rule_dict.get("rule_id", "unknown")))

        # Extract antecedent
        antecedent_ents = list(rule_dict.get("antecedent_entitlements", []))

        # Extract consequent - handle both formats
        consequent = rule_dict.get("consequent", {})
        if isinstance(consequent, dict) and "entitlements" in consequent:
            # Dynamic generator format
            consequent_ents = list(consequent.get("entitlements", []))
        else:
            # Mined rules format (consequent_entitlements at top level)
            consequent_ents = list(rule_dict.get("consequent_entitlements", []))

        # Extract strength metrics
        support = float(rule_dict.get("support", 0.0))
        confidence = float(rule_dict.get("confidence", 0.0))
        lift = float(rule_dict.get("lift", 1.0))

        # Create rule
        rule = AssociationRule(
            id=rule_id,
            app_name=app_name,
            antecedent_entitlements=antecedent_ents,
            consequent_entitlements=consequent_ents,
            support=support,
            confidence=confidence,
            lift=lift,
        )

        # Add to app's rule collection
        if app_name not in self.rules_by_app:
            self.rules_by_app[app_name] = []
        self.rules_by_app[app_name].append(rule)
        # END NEW METHOD

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #

    def has_rules(self, app_name: str) -> bool:
        return app_name in self.rules_by_app and len(self.rules_by_app[app_name]) > 0

    def get_rules_for_app(self, app_name: str) -> List[AssociationRule]:
        return self.rules_by_app.get(app_name, [])

    def suggest_entitlements_for_app(
            self,
            app_name: str,
            current_entitlements: Sequence[str],
            max_rules: int = 3,
            user_identity: Optional[Dict[str, str]] = None,  # NEW: User attributes for feature matching
    ) -> Tuple[List[str], List[AssociationRule]]:
        """
        Given a set of current entitlements for a user in an app, choose up to
        `max_rules` rules whose antecedent is a subset of current_entitlements,
        and return a set of recommended consequent entitlements plus the rules used.

        This is primarily useful when you *add* entitlements relative to a small seed.

        Args:
            app_name: Application name
            current_entitlements: Entitlements user currently has
            max_rules: Maximum rules to apply
            user_identity: User attributes for feature-based rule matching (e.g., {"department": "Finance"})
        """

        rules = self.rules_by_app.get(app_name, [])
        if not rules:
            return [], []

        current_set = set(current_entitlements)

        # BEGIN FIX: Build feature markers from user identity
        user_feature_markers = set()
        if user_identity:
            user_feature_markers = {
                f"FEATURE:{k}={v}"
                for k, v in user_identity.items()
                if v is not None
            }
        # END FIX

        candidate_rules: List[AssociationRule] = []

        for r in rules:
            ant = set(r.antecedent_entitlements)

            # BEGIN FIX: Handle empty antecedent
            if not ant:
                candidate_rules.append(r)
                continue
            # END FIX

            # BEGIN FIX: Detect rule type and match appropriately
            # Feature-based rules have markers like "FEATURE:department=Finance"
            # Entitlement-based rules have actual entitlement IDs
            if any(a.startswith("FEATURE:") for a in ant):
                # Feature-based rule: match against user identity
                if user_identity and ant.issubset(user_feature_markers):
                    candidate_rules.append(r)
            else:
                # Entitlement-based rule: match against current entitlements
                if ant.issubset(current_set):
                    candidate_rules.append(r)
            # END FIX

        if not candidate_rules:
            return [], []

        # Weight by support * confidence (can be tuned)
        weights = np.array([max(1e-6, r.support * r.confidence) for r in candidate_rules], dtype=float)
        weights = weights / weights.sum()

        k = min(max_rules, len(candidate_rules))
        chosen_idx = self.rng.choice(len(candidate_rules), size=k, replace=False, p=weights)
        chosen_rules = [candidate_rules[i] for i in chosen_idx]

        recommended: set[str] = set()
        for r in chosen_rules:
            recommended.update(r.consequent_entitlements)

        # Only return entitlements not already held
        new_ents = list(recommended - current_set)
        return new_ents, chosen_rules

    # ------------------------------------------------------------------ #
    # Coverage / Confidence
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_rule_coverage(
            entitlements: Sequence[str],
            rules_used: Sequence[AssociationRule],
    ) -> float:
        """
        Compute coverage = fraction of entitlements explained by rule consequents.
        """
        if not entitlements:
            return 0.0
        ent_set = set(entitlements)
        covered: set[str] = set()

        for r in rules_used:
            covered.update(r.consequent_entitlements)

        return len(covered & ent_set) / len(ent_set)

    @staticmethod
    def score_from_coverage_and_conf(
            coverage: float,
            rules_used: Sequence[AssociationRule],
    ) -> float:
        """
        Turn coverage and average rule confidence into a 0..1 score.

        Uses a weighted formula that emphasizes rule confidence over coverage:
        score = (0.7 × avg_confidence) + (0.3 × coverage)

        This produces realistic confidence distributions because:
        - High-confidence rules (0.85-0.95) yield high scores (0.65-0.75) even with 50% coverage
        - Medium-confidence rules (0.75-0.84) yield medium scores (0.50-0.65)
        - Low-confidence rules (0.65-0.74) yield low scores (0.40-0.55)

        Coverage acts as a modifier: perfect coverage adds up to +0.3 to the score.
        """
        if not rules_used:
            return 0.0

        # BEGIN FIX: Weighted formula emphasizing rule confidence
        avg_conf = sum(r.confidence for r in rules_used) / len(rules_used)

        # Weight: 70% from rule confidence, 30% from coverage
        # This ensures high-confidence rules produce high scores
        score = (0.7 * avg_conf) + (0.3 * coverage)
        # END FIX

        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def bucket_from_score(score: Optional[float]) -> str:
        """Map numeric score into High/Medium/Low/None."""
        if score is None:
            return "None"
        if score >= 0.8:
            return "High"
        if score >= 0.5:
            return "Medium"
        if score >= 0.2:
            return "Low"
        return "Low"