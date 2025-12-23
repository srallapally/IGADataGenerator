#!/usr/bin/env python3
"""
Mine association rules from existing <app>_accounts.csv and save them
as <app>_rules.json in a rules/ directory.

Usage:
    python mine_rules_from_accounts.py --app Epic --accounts out/Epic_accounts.csv --out-dir rules
"""

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def parse_multi(col: pd.Series, delimiter: str = "#") -> List[List[str]]:
    """Turn a multi-valued column into a list of transactions."""
    tx: List[List[str]] = []
    for val in col.fillna(""):
        if not val:
            tx.append([])
        else:
            tx.append([x for x in val.split(delimiter) if x])
    return tx


def mine_rules_for_app(
    app_name: str,
    accounts_path: Path,
    out_dir: Path,
    delimiter: str = "#",
    minsup: float = 0.02,
    minconf: float = 0.4,
) -> None:
    df = pd.read_csv(accounts_path)

    if app_name == "Epic":
        # Combine both Epic multi-valued columns
        tx: List[List[str]] = []
        for _, row in df.iterrows():
            ents = []
            for col in ("linkedTemplates", "linkedSubTemplates"):
                if col in df.columns:
                    val = row.get(col, "")
                    if isinstance(val, str) and val:
                        ents.extend([x for x in val.split(delimiter) if x])
            tx.append(sorted(set(ents)))
    else:
        if "entitlement_grants" not in df.columns:
            raise ValueError(f"No entitlement_grants column found in {accounts_path}")
        tx = parse_multi(df["entitlement_grants"], delimiter=delimiter)

    # Remove empty transactions
    tx = [t for t in tx if t]

    if not tx:
        print(f"No non-empty transactions for {app_name}; skipping.")
        return

    te = TransactionEncoder()
    te_ary = te.fit(tx).transform(tx)
    df_tx = pd.DataFrame(te_ary, columns=te.columns_)

    freq = apriori(df_tx, min_support=minsup, use_colnames=True)
    if freq.empty:
        print(f"No frequent itemsets for {app_name}; nothing to mine.")
        return

    rules = association_rules(freq, metric="confidence", min_threshold=minconf)
    if rules.empty:
        print(f"No rules above confidence {minconf} for {app_name}.")
        return

    # Keep only usable rules
    rules = rules[
        (rules["antecedents"].apply(len) >= 1)
        & (rules["consequents"].apply(len) >= 1)
    ].copy()

    records = []
    for idx, row in rules.iterrows():
        records.append(
            {
                "id": f"rule_{idx:06d}",
                "app_name": app_name,
                "antecedent_entitlements": sorted(list(row["antecedents"])),
                "consequent_entitlements": sorted(list(row["consequents"])),
                "support": float(row["support"]),
                "confidence": float(row["confidence"]),
                "lift": float(row["lift"]),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{app_name}_rules.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Wrote {len(records)} rules for {app_name} to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True, help="Application name, e.g. Epic or AWS")
    parser.add_argument("--accounts", required=True, help="Path to <app>_accounts.csv")
    parser.add_argument("--out-dir", default="rules", help="Directory to write rules JSON into")
    parser.add_argument("--minsup", type=float, default=0.02)
    parser.add_argument("--minconf", type=float, default=0.4)

    args = parser.parse_args()
    mine_rules_for_app(
        app_name=args.app,
        accounts_path=Path(args.accounts),
        out_dir=Path(args.out_dir),
        minsup=args.minsup,
        minconf=args.minconf,
    )


if __name__ == "__main__":
    main()
