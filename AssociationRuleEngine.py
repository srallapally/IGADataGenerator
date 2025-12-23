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
        Expected file naming: <app_name>_rules.json
        """
        if not self.rules_dir.exists():
            self.logger.warning(f"RuleEngine: rules directory does not exist: {self.rules_dir}")
            return

        for path in self.rules_dir.glob("*_rules.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw_rules = json.load(f)
            except Exception as e:
                self.logger.error(f"RuleEngine: failed to read {path}: {e}")
                continue

            if not isinstance(raw_rules, list):
                self.logger.warning(f"RuleEngine: rules file {path} is not a list; skipping")
                continue

            # Infer app_name from file name; JSON also contains app_name per rule
            app_name = path.name.replace("_rules.json", "")
            rules: List[AssociationRule] = []

            for r in raw_rules:
                try:
                    rules.append(
                        AssociationRule(
                            id=str(r.get("id", "")),
                            app_name=r.get("app_name", app_name),
                            antecedent_entitlements=list(r.get("antecedent_entitlements", [])),
                            consequent_entitlements=list(r.get("consequent_entitlements", [])),
                            support=float(r.get("support", 0.0)),
                            confidence=float(r.get("confidence", 0.0)),
                            lift=float(r.get("lift", 1.0)),
                        )
                    )
                except Exception as e:
                    self.logger.warning(f"RuleEngine: could not parse rule from {path}: {e}")

            if rules:
                self.rules_by_app[app_name] = rules
                self.logger.info(f"RuleEngine: loaded {len(rules)} rules for app '{app_name}' from {path}")

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
    ) -> Tuple[List[str], List[AssociationRule]]:
        """
        Given a set of current entitlements for a user in an app, choose up to
        `max_rules` rules whose antecedent is a subset of current_entitlements,
        and return a set of recommended consequent entitlements plus the rules used.

        This is primarily useful when you *add* entitlements relative to a small seed.
        """
        rules = self.rules_by_app.get(app_name, [])
        if not rules:
            return [], []

        current_set = set(current_entitlements)
        candidate_rules: List[AssociationRule] = []

        for r in rules:
            ant = set(r.antecedent_entitlements)
            # If rule has no antecedent, treat it as always applicable
            if not ant or ant.issubset(current_set):
                candidate_rules.append(r)

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
        """Turn coverage and average rule confidence into a 0..1 score."""
        if not rules_used:
            return 0.0
        avg_conf = sum(r.confidence for r in rules_used) / len(rules_used)
        raw = coverage * avg_conf
        return float(max(0.0, min(1.0, raw)))

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
