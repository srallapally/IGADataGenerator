import json
from pathlib import Path

import numpy as np

from AssociationRuleEngine import AssociationRule, RuleEngine


def _write_rules(path: Path, rules) -> None:
    path.write_text(json.dumps(rules), encoding="utf-8")


def test_load_all_rules_skips_invalid_json(tmp_path: Path) -> None:
    rules_path = tmp_path / "app_rules.json"
    rules_path.write_text("{not valid json", encoding="utf-8")

    engine = RuleEngine(tmp_path, np.random.default_rng(0))

    assert engine.rules_by_app == {}


def test_load_all_rules_skips_non_list_json(tmp_path: Path) -> None:
    rules_path = tmp_path / "app_rules.json"
    _write_rules(rules_path, {"not": "a list"})

    engine = RuleEngine(tmp_path, np.random.default_rng(0))

    assert engine.rules_by_app == {}


def test_load_all_rules_with_missing_fields(tmp_path: Path) -> None:
    rules_path = tmp_path / "crm_rules.json"
    _write_rules(rules_path, [{"consequent_entitlements": ["role.viewer"]}])

    engine = RuleEngine(tmp_path, np.random.default_rng(0))

    assert "crm" in engine.rules_by_app
    rule = engine.rules_by_app["crm"][0]
    assert rule.app_name == "crm"
    assert rule.antecedent_entitlements == []
    assert rule.consequent_entitlements == ["role.viewer"]
    assert rule.support == 0.0
    assert rule.confidence == 0.0
    assert rule.lift == 1.0


def test_suggest_entitlements_empty_rules(tmp_path: Path) -> None:
    engine = RuleEngine(tmp_path, np.random.default_rng(0))

    entitlements, rules = engine.suggest_entitlements_for_app("app", ["role1"])

    assert entitlements == []
    assert rules == []


def test_suggest_entitlements_empty_entitlements(tmp_path: Path) -> None:
    rules_path = tmp_path / "app_rules.json"
    _write_rules(
        rules_path,
        [
            {
                "id": "r1",
                "app_name": "app",
                "antecedent_entitlements": ["role1"],
                "consequent_entitlements": ["role2"],
                "support": 0.2,
                "confidence": 0.6,
                "lift": 1.1,
            }
        ],
    )

    engine = RuleEngine(tmp_path, np.random.default_rng(0))

    entitlements, rules = engine.suggest_entitlements_for_app("app", [])

    assert entitlements == []
    assert rules == []


def test_suggest_entitlements_subset_matching(tmp_path: Path) -> None:
    rules_path = tmp_path / "app_rules.json"
    _write_rules(
        rules_path,
        [
            {
                "id": "r1",
                "app_name": "app",
                "antecedent_entitlements": ["role1"],
                "consequent_entitlements": ["role2"],
                "support": 0.4,
                "confidence": 0.7,
                "lift": 1.2,
            }
        ],
    )

    engine = RuleEngine(tmp_path, np.random.default_rng(0))

    entitlements, rules = engine.suggest_entitlements_for_app("app", ["role1"])

    assert set(entitlements) == {"role2"}
    assert len(rules) == 1
    assert rules[0].id == "r1"


def test_rule_coverage_and_score_edge_cases() -> None:
    rule = AssociationRule(
        id="r1",
        app_name="app",
        antecedent_entitlements=["role1"],
        consequent_entitlements=["role2"],
        support=0.4,
        confidence=0.7,
        lift=1.2,
    )

    assert RuleEngine.compute_rule_coverage([], [rule]) == 0.0
    assert RuleEngine.compute_rule_coverage(["role2"], []) == 0.0
    assert RuleEngine.score_from_coverage_and_conf(0.5, []) == 0.0
