# IGA Data Generator

## Overview
The IGA Data Generator creates synthetic Identity Governance and Administration (IGA)
datasets that include identity profiles, application entitlement catalogs, and
account assignments with confidence scores. It is designed to simulate realistic
access patterns, support analytics, and enable testing or prototyping of IGA
pipelines without exposing real data.

## Overall Design
The generator is built around a small set of modules that each handle a specific
stage of the pipeline:

- **Configuration loading** (`synthetic_iga_data_generator.py`, `ConfigLoader`)
  reads and validates `synthetic_iga_data_generator_config.json`, expands the
  application list, and normalizes config values.
- **Rule generation**
  - **Static schema-driven rules** (`rule_schema_generator.py`) define rules based
    on the configured schema.
  - **Dynamic rules** (`dynamic_rule_generator.py`) optionally create rules with
    target support/confidence/Cramér’s V ranges by analyzing generated identity
    attributes.
  - **Cross-application rules** (`cross_app_rule_schema_generator.py`,
    `dynamic_rule_generator_cross_app.py`) describe entitlements that span
    multiple apps.
- **Rule execution** (`AssociationRuleEngine.py`) loads rule JSON and projects
  cross-app rules into per-application rule sets for evaluation.
- **Feature recommendation** (`feature_recommender.py`) analyzes generated
  identities and highlights the most informative attributes for downstream
  modeling.
- **Dataset output** (`synthetic_iga_data_generator.py`) emits:
  - `identities.csv`
  - `<app>_entitlements.csv`
  - `<app>_accounts.csv`

For a visual flow of these components, see `ARCHITECTURE.md`.

## ML and Statistical Principles
The generator relies on classic statistical and machine learning concepts to
create realistic relationships between user attributes and entitlements.

### Association Rule Metrics
Rules capture relationships between identity features (antecedents) and
entitlements (consequents). The system tracks:

- **Support**: how frequently a rule occurs in the population.
- **Confidence**: how often the consequent appears when the antecedent is
  present.
- **Lift**: how much more likely the consequent is given the antecedent compared
  to random chance.

These metrics are used both when generating dynamic rules and when applying
rules in `AssociationRuleEngine.RuleEngine`.

### Feature Selection and Statistical Tests
`feature_recommender.py` identifies the most informative identity attributes by
combining multiple filters:

- **Cardinality filtering** to drop extremely high-cardinality attributes that
  are unlikely to generalize.
- **Chi-square tests** and **Cramér’s V** to measure statistical association
  between candidate features.
- **Mutual information** to quantify predictive value across categorical
  features and identify strong signals.

### Entropy and Distribution Targets
Dynamic rule generation (`dynamic_rule_generator.py`) analyzes attribute
entropy and distribution balance to choose features that are neither constant
nor overly sparse. This ensures the synthetic dataset contains both common and
rare access patterns that resemble production variability.

### Cross-App Correlations
Cross-application rules model shared access behavior (for example, a department
implies a set of entitlements across multiple systems). The rule engine
decomposes these relationships into per-app rules for evaluation while
preserving the original joint structure.

## Getting Started
Install dependencies and run the generator:

```bash
pip install -r requirements.txt
python synthetic_iga_data_generator.py --config synthetic_iga_data_generator_config.json
```

The output CSVs are generated in the working directory unless configured
otherwise.
