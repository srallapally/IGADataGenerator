#!/usr/bin/env python3

"""
gemini_validate_rules.py

Loads generated IGA data, builds a transactions matrix, and mines
association rules using mlxtend. Reports on rule counts and confidence
bucket distributions.
"""

import argparse
import logging
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Any, Tuple

# External libraries, provide guidance if missing
try:
    import numpy as np
    import pandas as pd
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import (
        apriori,
        fpgrowth,
        association_rules,
    )
except ImportError:
    print(
        "Error: Missing required libraries. Please install them via:",
        file=sys.stderr,
    )
    print("pip install pandas mlxtend", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---

logger = logging.getLogger(__name__)

# --- BEGIN ChatGPT change: Phase 3 - extend feature columns ---
FEATURE_COLS = [
    "department",
    "business_unit",
    "job_family",
    "job_level",
    "location_country",
    "location_site",
    "employment_type",
    "manager_flag",
    "line_of_business",
    "status",
    "has_manager",
]

TOKEN_PREFIX_MAP = {
    "department": "dept",
    "business_unit": "bu",
    "job_family": "jobfam",
    "job_level": "level",
    "location_country": "country",
    "location_site": "site",
    "employment_type": "etype",
    "manager_flag": "mgr",
    "line_of_business": "lob",
    "status": "status",
    "has_manager": "has_mgr",
}
# --- END ChatGPT change: Phase 3 - extend feature columns ---


# --- Helper Functions ---


def setup_logging(level=logging.INFO):
    """Configure logging."""
    log_format = (
        "%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s"
    )
    logging.basicConfig(level=level, format=log_format)


def load_data(
    identities_path: Path,
    assignments_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load identities and assignments CSVs."""
    logger.info(f"Loading identities from {identities_path}")
    try:
        identities_df = pd.read_csv(identities_path)
    except FileNotFoundError:
        logger.error(f"Identity file not found: {identities_path}")
        sys.exit(1)

    logger.info(f"Loading assignments from {assignments_path}")
    try:
        assignments_df = pd.read_csv(assignments_path)
    except FileNotFoundError:
        logger.error(f"Assignment file not found: {assignments_path}")
        sys.exit(1)

    return identities_df, assignments_df


def _tokenize_identity_row(user_row: pd.Series) -> Set[str]:
    """Convert a user's identity attributes into feature tokens."""
    tokens: Set[str] = set()
    for col in FEATURE_COLS:
        if col in user_row:
            prefix = TOKEN_PREFIX_MAP.get(col, col)
            # Ensure values are strings for tokenization
            tokens.add(f"{prefix}={str(user_row[col])}")
    return tokens


def create_transactions(
    identities_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
) -> List[List[str]]:
    """
    Create a list of transactions, where each transaction is a
    list of a user's entitlements and their identity feature tokens.
    """
    # --- BEGIN ChatGPT change: Phase 3 - derive has_manager feature ---
    # Derive a simple boolean/flag feature based on manager_id to avoid
    # using the raw foreign key as a high-cardinality token.
    if (
        "manager_id" in identities_df.columns
        and "has_manager" not in identities_df.columns
    ):
        identities_df["has_manager"] = np.where(
            identities_df["manager_id"].notna(), "Y", "N"
        )
    # --- END ChatGPT change: Phase 3 - derive has_manager feature ---

    logger.info("Tokenizing identities...")
    identity_tokens: Dict[str, Set[str]] = {
        row["user_id"]: _tokenize_identity_row(row)
        for _, row in identities_df.iterrows()
    }

    logger.info("Grouping entitlements by user...")
    ents_by_user: Dict[str, Set[str]] = defaultdict(set)
    if "user_id" not in assignments_df.columns:
        logger.error(
            "assignments.csv is missing required column 'user_id'."
        )
        sys.exit(1)
    if "entitlement_id" not in assignments_df.columns:
        logger.error(
            "assignments.csv is missing required column 'entitlement_id'."
        )
        sys.exit(1)

    for _, row in assignments_df.iterrows():
        uid = row["user_id"]
        ent = row["entitlement_id"]
        ents_by_user[uid].add(f"ent={ent}")

    logger.info("Building final transactions list...")
    transactions: List[List[str]] = []
    missing_identities = 0

    for user_id, entitlements in ents_by_user.items():
        if user_id not in identity_tokens:
            # This can happen if there are assignments pointing to
            # identities that were filtered out or not generated.
            missing_identities += 1
            continue

        features = identity_tokens[user_id]
        transaction = list(entitlements.union(features))
        transactions.append(transaction)

    if missing_identities > 0:
        logger.warning(
            f"{missing_identities} users in assignments.csv "
            f"were not found in identities.csv"
        )

    logger.info(
        f"Built {len(transactions)} transactions "
        f"for {len(ents_by_user)} users (with entitlements)."
    )

    return transactions


def run_association_mining(
    transactions: List[List[str]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """
    Run the full association rule mining pipeline using mlxtend.
    """
    logger.info("Encoding transactions...")
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions, sparse=True)

    # Create sparse DataFrame
    trans_df_sparse = pd.DataFrame.sparse.from_spmatrix(
        te_ary, columns=te.columns_
    )

    logger.info(
        f"Transaction matrix shape: {trans_df_sparse.shape} "
        f"(Users, Items)"
    )

    logger.info(
        f"Finding frequent itemsets with minsup={args.minsup}..."
    )

    # Use fpgrowth as it's often faster; fall back to apriori if needed
    try:
        frequent_itemsets = fpgrowth(
            trans_df_sparse,
            min_support=args.minsup,
            use_colnames=True,
            max_len=args.max_len,
        )
    except Exception as e_fpgrowth:
        logger.error(
            f"fpgrowth failed with error: {e_fpgrowth}. "
            f"Trying apriori as fallback..."
        )
        try:
            frequent_itemsets = apriori(
                trans_df_sparse,
                min_support=args.minsup,
                use_colnames=True,
                max_len=args.max_len,
            )
        except Exception as e_apriori:
            logger.error(
                f"Apriori fallback also failed: {e_apriori}"
            )
            return pd.DataFrame()

    logger.info(
        f"Found {len(frequent_itemsets)} frequent itemsets."
    )

    if frequent_itemsets.empty:
        logger.warning(
            "No frequent itemsets found. Try lowering --minsup."
        )
        return pd.DataFrame()

    logger.info(
        f"Generating association rules with "
        f"minconf={args.minconf}..."
    )
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=args.minconf,
    )

    logger.info(f"Generated {len(rules)} rules.")
    return rules


def analyze_and_report(
    rules: pd.DataFrame,
    identities_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
) -> None:
    """Analyze and print a report of the mined rules."""

    print("\n--- Sanity Metrics ---")
    avg_ents = len(assignments_df) / len(identities_df)
    print(f"Avg. entitlements per user: {avg_ents:.2f}")

    # --- BEGIN ChatGPT change: Phase 3 - extended entropy metrics ---
    # Calculate entropy for a broader set of identity fields,
    # including the new governance-relevant dimensions.
    entropy_cols = [
        "department",
        "location_site",
        "job_family",
        "line_of_business",
        "status",
        "manager_flag",
        "has_manager",
    ]
    for col in entropy_cols:
        if col not in identities_df.columns:
            continue
        probs = identities_df[col].value_counts(normalize=True)
        if probs.empty:
            continue
        entropy = -np.sum(probs * np.log2(probs))
        print(f"Entropy ({col}): {entropy:.2f} bits")
    # --- END ChatGPT change: Phase 3 - extended entropy metrics ---

    # --- BEGIN ChatGPT change: Phase 3 - manager/status summary ---
    if "manager_flag" in identities_df.columns:
        mgr_frac = (identities_df["manager_flag"] == "Y").mean()
        print(f"Manager ratio (manager_flag=Y): {mgr_frac:.2%}")

    if "status" in identities_df.columns:
        status_counts = identities_df["status"].value_counts(
            normalize=True
        )
        print(
            "Status distribution:",
            {k: f"{v:.2%}" for k, v in status_counts.items()},
        )
    # --- END ChatGPT change: Phase 3 - manager/status summary ---

    if rules is None or rules.empty:
        print("\nNo rules to analyze.")
        return

    print("\n--- Rule Statistics ---")
    print(f"Total rules: {len(rules)}")

    # Confidence buckets
    buckets = {
        "[0.40, 0.60)": (0.40, 0.60),
        "[0.60, 0.80)": (0.60, 0.80),
        "[0.80, 0.90)": (0.80, 0.90),
        "[0.90, 0.95)": (0.90, 0.95),
        "[0.95, 1.00]": (0.95, 1.01),
    }

    conf = rules["confidence"]
    for label, (low, high) in buckets.items():
        count = ((conf >= low) & (conf < high)).sum()
        print(f"{label}: {count} rules")

    # Top rules by confidence
    print("\nTop 10 rules by confidence:")
    top_rules = rules.sort_values(
        by="confidence", ascending=False
    ).head(10)

    for _, r in top_rules.iterrows():
        antecedent = ", ".join(sorted(list(r["antecedents"])))
        consequent = ", ".join(sorted(list(r["consequents"])))
        print(
            f"{antecedent} => {consequent} "
            f"(support={r['support']:.3f}, "
            f"confidence={r['confidence']:.3f}, "
            f"lift={r['lift']:.3f})"
        )


def main():
    """Main validator execution."""
    parser = argparse.ArgumentParser(
        description="Validate IGA data by mining association rules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--identities",
        type=Path,
        default=Path("./out/identities.csv"),
        help="Path to the generated identities.csv file.",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path("./out/assignments.csv"),
        help="Path to the generated assignments.csv file.",
    )
    parser.add_argument(
        "--minsup",
        type=float,
        default=0.02,
        help="Minimum support threshold for frequent itemsets.",
    )
    parser.add_argument(
        "--minconf",
        type=float,
        default=0.4,
        help="Minimum confidence threshold for association rules.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=3,
        help=(
            "Maximum length of frequent itemsets "
            "(e.g., 3 for A,B => C)."
        ),
    )
    # Seed is not needed for the validator as mlxtend is deterministic

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG level logging.",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    # 1. Load Data
    identities, assignments = load_data(
        args.identities, args.assignments
    )

    # 2. Build Transactions
    transactions = create_transactions(identities, assignments)

    if not transactions:
        logger.error(
            "No transactions could be built. Exiting."
        )
        sys.exit(1)

    # 3. Run Mining
    rules = run_association_mining(transactions, args)

    # 4. Report
    analyze_and_report(rules, identities, assignments)

    logger.info("Validation script finished.")


if __name__ == "__main__":
    main()
