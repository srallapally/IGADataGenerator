# IGA Data Generator Architecture

This diagram summarizes how the synthetic IGA data generator loads configuration, generates rules, and emits datasets.

```mermaid
flowchart TD
    Config[synthetic_iga_data_generator_config.json] --> Loader[ConfigLoader]
    Loader --> Generator[synthetic_iga_data_generator.py]

    subgraph Rules[Rule generation]
        SchemaRules[rule_schema_generator.py]
        DynamicRules[dynamic_rule_generator.py]
        CrossSchema[cross_app_rule_schema_generator.py]
        CrossDynamic[dynamic_rule_generator_cross_app.py]
    end

    subgraph Engines[Rule execution]
        RuleEngine[AssociationRuleEngine.RuleEngine]
        FeatureRec[feature_recommender.py]
    end

    SchemaRules --> Generator
    DynamicRules --> Generator
    CrossSchema --> Generator
    CrossDynamic --> Generator

    Generator --> RuleEngine
    RuleEngine --> FeatureRec

    Generator --> Identities[identities.csv]
    Generator --> Entitlements[<app>_entitlements.csv]
    Generator --> Accounts[<app>_accounts.csv]

    FeatureRec --> Recommendations[Feature/entitlement recommendations]
```

## Notes
- `synthetic_iga_data_generator.py` is the primary entry point; it orchestrates configuration loading, rule ingestion, and dataset output.
- Rule generators produce association rules (per-app or cross-app) that the `AssociationRuleEngine` loads to drive entitlement assignments.
- The generator emits identity records plus per-application entitlement catalogs and account assignments.
