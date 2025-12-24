import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from AssociationRuleEngine import RuleEngine
from MineRulesFromAccounts import mine_rules_for_app


REQUIRED_RULE_KEYS = {
    "id",
    "app_name",
    "antecedent_entitlements",
    "consequent_entitlements",
    "support",
    "confidence",
    "lift",
}


def assert_required_columns(path: Path, required: set[str]) -> None:
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    assert not missing, f"Missing required columns: {sorted(missing)}"


def test_rule_engine_rules_json_required_fields(tmp_path: Path) -> None:
    records = [
        {
            "id": "rule_000001",
            "app_name": "SampleApp",
            "antecedent_entitlements": ["entitlement_a"],
            "consequent_entitlements": ["entitlement_b"],
            "support": 0.5,
            "confidence": 0.75,
            "lift": 1.2,
        }
    ]
    assert REQUIRED_RULE_KEYS.issubset(records[0].keys())

    rules_path = tmp_path / "SampleApp_rules.json"
    with open(rules_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle)

    engine = RuleEngine(tmp_path, rng=np.random.default_rng(0))
    rules = engine.get_rules_for_app("SampleApp")

    assert len(rules) == 1
    assert rules[0].app_name == "SampleApp"
    assert rules[0].antecedent_entitlements == ["entitlement_a"]
    assert rules[0].consequent_entitlements == ["entitlement_b"]


def test_mine_rules_for_app_output_schema(tmp_path: Path) -> None:
    accounts_path = tmp_path / "TestApp_accounts.csv"
    df = pd.DataFrame(
        {
            "entitlement_grants": [
                "ent1#ent2",
                "ent1#ent2",
                "ent1",
                "ent2",
            ]
        }
    )
    df.to_csv(accounts_path, index=False)

    out_dir = tmp_path / "rules"
    mine_rules_for_app(
        app_name="TestApp",
        accounts_path=accounts_path,
        out_dir=out_dir,
        minsup=0.25,
        minconf=0.1,
    )

    out_path = out_dir / "TestApp_rules.json"
    with open(out_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    assert isinstance(data, list)
    assert data, "Expected at least one rule record"

    for record in data:
        assert isinstance(record, dict)
        assert REQUIRED_RULE_KEYS.issubset(record.keys())
        assert isinstance(record["antecedent_entitlements"], list)
        assert isinstance(record["consequent_entitlements"], list)


def test_entitlements_csv_required_columns(tmp_path: Path) -> None:
    entitlements_path = tmp_path / "App_entitlements.csv"
    df = pd.DataFrame(
        [
            {
                "entitlement_id": "ENT-001",
                "entitlement_name": "Example Entitlement",
            }
        ]
    )
    df.to_csv(entitlements_path, index=False)

    assert_required_columns(entitlements_path, {"entitlement_id", "entitlement_name"})


def test_entitlements_csv_missing_required_columns_raises(tmp_path: Path) -> None:
    entitlements_path = tmp_path / "App_entitlements.csv"
    df = pd.DataFrame(
        [
            {
                "entitlement_id": "ENT-002",
            }
        ]
    )
    df.to_csv(entitlements_path, index=False)

    with pytest.raises(AssertionError):
        assert_required_columns(entitlements_path, {"entitlement_id", "entitlement_name"})
