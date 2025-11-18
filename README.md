# Auto ID Data Generator

Synthetic Identity, Entitlement & Assignment Dataset Generator for Identity Governance & AI Analytics.

This repository contains a configurable, reproducible pipeline for generating **fully synthetic** IGA-style datasets:
- **Identities** (users, departments, employment types, locations, managers, etc.)
- **Entitlements** (application roles, groups, permissions, templates)
- **Assignments** (user ↔ entitlement grants)
- **Per‑application breakdowns** for entitlements and grants
- **Optional rule mining & validation** to verify that generated access patterns exhibit expected statistical properties

The generated data is suitable for:

- Prototyping Identity Governance & Administration (IGA) analytics
- Building & testing access pattern‑mining and role‑mining algorithms
- Creating sandbox datasets for demos and training
- Simulating AI‑agent access, segregation of duties, and risk models
- Load‑testing and benchmarking data pipelines

> ⚠️ All data is synthetic and randomly generated. There is no real PII.

---

## ✨ Features

- **Config‑driven generation** via `config.sample.json`
- **Realistic identity attributes** (departments, business units, locations, job families, job levels, managers, employment type, etc.)
- **Configurable application and entitlement universe**
- **Latent access patterns** to simulate real‑world access behavior
- **Noise and outliers** (background entitlements, outlier users)
- **Calibration engine** to align observed rule confidence with target bands
- **Per‑app CSVs** for entitlements and user grants
- **Joining identity context into grants**, including `user_name` (and optionally `password`) in `*_grants.csv`
- **Reproducibility support** via deterministic seeds and built‑in tests
- **Validation pipeline** to mine association rules and analyze confidence bands

---

## 📁 Repository Structure

```text
.
├── gemini_synthetic_iga.py      # Main synthetic data generator and pipeline
├── gemini_validate_rules.py     # Association rule mining + validation/QA
├── example_usage.py             # Example usages (small sample generations)
├── config.sample.json           # Example configuration file
├── run_pipeline.sh              # One‑shot script to generate + validate
├── requirements.txt             # Python dependencies
├── out/                         # Default output directory (git‑ignored)
│   ├── identities.csv
│   ├── entitlements.csv
│   ├── assignments.csv
│   ├── entitlements_by_app/
│   └── entitlement_grants_by_app/
└── README.md
```

> The `out/` directory is created/overwritten by the pipeline and is not meant to be tracked in version control.

---

## 🧰 Requirements

- Python **3.9+** (earlier versions may work but are not tested)
- Recommended: virtual environment (`venv`, `pyenv`, or similar)

Install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

Key libraries:
- `pandas`
- `numpy`
- `faker`
- `mlxtend`
- `scikit-learn`

---

## 🚀 Quick Start

The easiest way to run the full pipeline is via the provided shell script:

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

What this does:

1. **Clean** the `./out` directory (if it exists)
2. **Generate** synthetic data using `gemini_synthetic_iga.py`
3. **Validate** the generated data using `gemini_validate_rules.py`

At the end you should see:

```text
out/
  identities.csv
  entitlements.csv
  assignments.csv
  entitlements_by_app/
  entitlement_grants_by_app/
```

---

## 🎛 Manual Usage

You can call the generator directly if you want more control.

### 1. Generate Data

```bash
python3 gemini_synthetic_iga.py \
  --config config.sample.json \
  --out-dir ./out
```

Common CLI options:

| Flag                       | Description                                   |
|----------------------------|-----------------------------------------------|
| `--config PATH`           | JSON or YAML config file                      |
| `--out-dir PATH`          | Output directory (default: `./out`)           |
| `--seed INT`              | Override PRNG seed                            |
| `--num-users INT`         | Override number of generated users            |
| `--num-apps INT`          | Override number of applications               |
| `--num-entitlements INT`  | Override number of entitlements               |
| `--emit-default-config`   | Print default config JSON and exit            |
| `--run-tests`             | Run built-in tests and exit                   |
| `--no-clobber`            | Do not overwrite existing CSV files           |
| `-v/--verbose`            | Enable DEBUG logging                          |

### 2. Run Validation / Rule Mining

```bash
python3 gemini_validate_rules.py \
  --identities ./out/identities.csv \
  --assignments ./out/assignments.csv \
  --minsup 0.02 \
  --minconf 0.4
```

This will:

- Load identities and assignments
- Build transactions mixing entitlements and identity features
- Mine association rules
- Classify them into confidence buckets (`High`, `Mid`, `Low`)
- Print summary statistics to the console

---

## 🧬 Data Model Overview

### `identities.csv`

Each row represents a synthetic identity (user). Example columns (may vary slightly based on the current Identity schema):

```text
user_id,
user_name,
first_name,
last_name,
email,
department,
business_unit,
line_of_business,
job_family,
job_level,
title,
location_country,
location_site,
employment_type,
cost_center,
manager_flag,
manager_id,
status,
tenure_years
```

- `user_id`: Stable synthetic internal identifier (e.g., `U0001234`)
- `user_name`: Login/user name derived from name, unique
- `manager_id`: `user_id` of the manager (if applicable)
- `status`: Typically `"active"` / `"inactive"`

---

### `entitlements.csv`

Each row is an entitlement for some application.

```text
entitlement_id,
app_id,
app_name,
entitlement_name,
entitlement_type,
criticality,
scope,
description
```

- `app_id`: Synthetic application key (e.g., `APP01`)
- `entitlement_type`: Such as `role`, `group`, `project-role`, `repo-permission`, etc.
- `criticality`: `High`, `Medium`, `Low`

---

### `assignments.csv`

Represents grants of entitlements to users:

```text
user_id,
entitlement_id
```

No derived columns here; it is a clean many‑to‑many mapping between identities and entitlements.

---

### Per‑Application Entitlement Files

#### `entitlements_by_app/APPxx_entitlements.csv`

Subset of `entitlements.csv` for a single application:

```text
entitlement_id,
entitlement_name,
app_id,
description
```

This is useful for app‑scoped importers or app‑specific analysis.

---

### Per‑Application Grant Files

#### `entitlement_grants_by_app/APPxx_grants.csv`

Each file contains the users and their aggregated grants *for a single application*.

Format (current convention):

```text
user_id,
user_name,
entitlement_grants,
application_id,
password
```

- `entitlement_grants`: `#`‑delimited list of entitlement_ids the user has in this application
- `user_name`: Joined from `identities.csv` using `user_id`
- `password`: Optional – depends on whether a `password` column is present in `identities.csv`. It can be blank or synthetically generated. This column is included for scenarios where you need per‑app credentials in synthetic datasets (e.g., for testing account provisioning flows).

The logic for writing these files lives in `write_entitlement_grants_by_app(...)` inside `gemini_synthetic_iga.py`.

---

## ⚙️ Configuration

All generator behavior is controlled by a nested configuration structure. A reference file is provided:

- `config.sample.json`

Top-level sections include (but are not limited to):

- `global`  
  - `num_users`, `num_apps`, `num_entitlements`, `seed`, etc.
- `identity`  
  - Department distributions, job families, countries, sites, employment type mix, manager probability, etc.
- `entitlement`  
  - App names, app sizes, entitlement types, criticality probabilities, bundle configuration.
- `pattern`  
  - Number of patterns, confidence bands, probabilities for core vs optional entitlements, bundle application span.
- `noise`  
  - Background entitlement fraction, background user fraction, outlier user fraction, collision probability reduction.
- `calibration`  
  - Target bucket shares (`High`, `Mid`, `Low`), tolerance, max iterations, learning rate, bucket thresholds.

You can either:
- Edit `config.sample.json` directly and pass it via `--config`, or
- Copy it to a new file (e.g., `config.my-lab.json`) and customize that.

To see the *built‑in* default config:

```bash
python3 gemini_synthetic_iga.py --emit-default-config
```

---

## 🔍 Calibration & Tests

The calibration engine attempts to ensure that **observed** rule confidence across patterns roughly matches target proportions in each band:

- Target bands (e.g., `High`: 0.6, `Mid`: 0.3, `Low`: 0.1)
- Bucket thresholds (e.g., High ≥ 0.7, Mid ≥ 0.4)

The generator iteratively:

1. Generates patterns and assignments
2. Measures confidence of pattern‑driven rules
3. Adjusts base probabilities of core entitlements to nudge towards target

You can run built‑in tests to verify that:

- There are no duplicate `(user_id, entitlement_id)` pairs
- The pipeline is reproducible for a given seed

```bash
python3 gemini_synthetic_iga.py --run-tests
```

---

## 🧪 Example Usage Script

`example_usage.py` contains several small, standalone examples that:

- Generate a smaller dataset with a custom config
- Write identities/entitlements/assignments to a local folder
- Run basic QA reporting

Run it directly if you want a “playground” version of the generator:

```bash
python3 example_usage.py
```

---

## 🛠 Extending the Project

Here are common extension points:

### Add New Identity Attributes

1. Update the `Identity` dataclass in `gemini_synthetic_iga.py`.
2. Extend the identity generation logic to populate the new attribute.
3. Ensure the attribute is included in the `identities.csv` DataFrame before writing.

### Add or Modify Applications & Entitlements

- Update the `entitlement` section in `config.sample.json` (app_names, app sizes, entitlement types).
- The `EntitlementGenerator` uses this configuration to build the entitlement universe.

### Customize Per‑App Grants Format

- Modify `write_entitlement_grants_by_app(...)` in `gemini_synthetic_iga.py`.
- You can change aggregation (e.g., keep entitlements as separate rows instead of `#`‑joined), add new columns, or output additional files.

### Integrate with Your Own Pipelines

- Use the CSV outputs directly (common for ETL / data warehouses).
- Or import the generator module functions/classes directly into another Python project and invoke them programmatically.

---

## 🧾 Logging

The generator and validator use Python’s `logging` module.

- Default level: `INFO`
- Use `-v` / `--verbose` to enable `DEBUG` logging for troubleshooting.

You can also customize logging behavior by editing the logger configuration at the top of the scripts or by wrapping script entry points in your own logging setup.

---

## 🤝 Contributing

Contributions, issue reports, and feature requests are welcome.

Ideas for contributions:

- Additional identity attributes (e.g., regions, legal entities)
- New entitlement/role models (e.g., SaaS‑specific patterns like Salesforce, Jira, Epic, etc.)
- More advanced pattern generation
- Time‑series data (join/leave dates, grant/revoke timestamps)
- Synthetic request and approval events
- Direct ML‑ready output (feature matrices, label columns)

If you open a PR, please include:

- A clear description of the change
- Any config changes
- A short note on expected impact on dataset size or shape
- Updated README snippets if behavior or schema changed

---

## 📄 License

This project is licensed under the MIT License (or update this section to your preferred license).
