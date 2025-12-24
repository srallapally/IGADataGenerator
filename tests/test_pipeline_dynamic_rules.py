import json
from pathlib import Path

import numpy as np
import pandas as pd

from dynamic_rule_generator import RuleGenerationOrchestrator, RuleGeneratorConfig
from AssociationRuleEngine import RuleEngine


def _write_entitlements(entitlements_dir: Path, app_name: str, entitlements: list[dict[str, str]]) -> None:
    entitlements_dir.mkdir(parents=True, exist_ok=True)
    entitlements_df = pd.DataFrame(entitlements)
    entitlements_df.to_csv(entitlements_dir / f"{app_name}_entitlements.csv", index=False)


def _build_users_df(rng: np.random.Generator, num_rows: int = 12) -> pd.DataFrame:
    departments = ["IT", "Finance", "HR"]
    locations = ["NYC", "Remote", "London"]
    levels = ["L1", "L2", "L3"]

    return pd.DataFrame(
        {
            "user_id": [f"U{i:03d}" for i in range(num_rows)],
            "department": rng.choice(departments, size=num_rows),
            "location": rng.choice(locations, size=num_rows),
            "job_level": rng.choice(levels, size=num_rows),
        }
    )


def test_dynamic_rules_pipeline(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)

    users_df = _build_users_df(rng)
    users_file = tmp_path / "users.csv"
    users_df.to_csv(users_file, index=False)

    entitlements_dir = tmp_path / "entitlements"
    apps = ["AlphaApp", "BetaApp"]

    for app in apps:
        _write_entitlements(
            entitlements_dir,
            app,
            [
                {
                    "entitlement_id": f"{app}-read",
                    "entitlement_name": f"{app} Read",
                    "entitlement_type": "standard",
                },
                {
                    "entitlement_id": f"{app}-write",
                    "entitlement_name": f"{app} Write",
                    "entitlement_type": "standard",
                },
            ],
        )

    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)

    config = RuleGeneratorConfig(
        num_rules_per_app=2,
        min_features_per_rule=1,
        max_features_per_rule=1,
        min_entitlements_per_rule=1,
        max_entitlements_per_rule=1,
    )

    for app in apps:
        orchestrator = RuleGenerationOrchestrator(
            users_file=users_file,
            apps_config=[{"app_name": app, "app_id": f"APP_{app}"}],
            entitlements_dir=entitlements_dir,
            output_file=rules_dir / f"{app}_rules.json",
            config=config,
            seed=42,
        )
        orchestrator.run()

        output_data = json.loads((rules_dir / f"{app}_rules.json").read_text())
        assert isinstance(output_data, list)
        assert output_data, f"Expected rules for {app}"

    engine = RuleEngine(rules_dir=rules_dir, rng=np.random.default_rng(321))

    for app in apps:
        assert engine.has_rules(app)
