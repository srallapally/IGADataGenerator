#!/usr/bin/env python3
"""
Validation script to check for users with zero entitlements
and verify entitlement distribution across all apps.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict


def validate_no_zero_entitlement_users(output_dir: Path):
    """
    Validate that no user has zero entitlements across all applications.

    Returns:
        (success: bool, report: dict)
    """
    print("=" * 70)
    print("ZERO-ENTITLEMENT USER VALIDATION")
    print("=" * 70)

    # Find all account files
    account_files = list(output_dir.glob("*_accounts.csv"))

    if not account_files:
        return False, {"error": "No account files found"}

    print(f"\nFound {len(account_files)} application account files")

    # Load identities
    identities_file = output_dir / "identities.csv"
    if not identities_file.exists():
        return False, {"error": "identities.csv not found"}

    identities_df = pd.read_csv(identities_file)
    total_identities = len(identities_df)
    print(f"Total identities: {total_identities}")

    # Build user -> entitlement count map
    user_ent_count = defaultdict(int)
    app_stats = {}

    for account_file in account_files:
        app_name = account_file.stem.replace("_accounts", "")
        df = pd.read_csv(account_file)

        zero_in_app = 0
        for _, row in df.iterrows():
            user_id = row['user_id']
            ent_str = row['entitlement_grants']

            if pd.notna(ent_str) and ent_str != '':
                count = ent_str.count('#') + 1
            else:
                count = 0
                zero_in_app += 1

            user_ent_count[user_id] += count

        app_stats[app_name] = {
            'total_accounts': len(df),
            'accounts_with_zero': zero_in_app,
            'pct_with_zero': (zero_in_app / len(df) * 100) if len(df) > 0 else 0
        }

    # Print per-app statistics
    print(f"\n{'Application':<20} {'Total':>8} {'Zero':>8} {'%Zero':>8}")
    print("-" * 50)
    for app_name, stats in sorted(app_stats.items()):
        print(f"{app_name:<20} {stats['total_accounts']:>8} "
              f"{stats['accounts_with_zero']:>8} {stats['pct_with_zero']:>7.1f}%")

    # Check for users with zero entitlements across ALL apps
    users_with_zero = [uid for uid, count in user_ent_count.items() if count == 0]

    print("\n" + "=" * 70)
    print("CROSS-APP VALIDATION")
    print("=" * 70)
    print(f"Users with ZERO entitlements across ALL apps: {len(users_with_zero)}")

    if users_with_zero:
        print(f"\n⚠ VALIDATION FAILED!")
        print(f"Found {len(users_with_zero)} users with no entitlements:")
        for uid in users_with_zero[:10]:
            print(f"  - {uid}")
        if len(users_with_zero) > 10:
            print(f"  ... and {len(users_with_zero) - 10} more")
        success = False
    else:
        print(f"\n✓ VALIDATION PASSED!")
        print(f"All {total_identities} users have at least 1 entitlement")
        success = True

    # Entitlement distribution statistics
    counts = list(user_ent_count.values())
    import numpy as np

    print("\n" + "=" * 70)
    print("ENTITLEMENT DISTRIBUTION")
    print("=" * 70)
    print(f"Min entitlements per user:     {np.min(counts)}")
    print(f"Max entitlements per user:     {np.max(counts)}")
    print(f"Mean entitlements per user:    {np.mean(counts):.1f}")
    print(f"Median entitlements per user:  {np.median(counts):.1f}")
    print(f"\nUsers with 0 entitlements:     {sum(1 for c in counts if c == 0)}")
    print(f"Users with 1-2 entitlements:   {sum(1 for c in counts if 1 <= c <= 2)}")
    print(f"Users with 3+ entitlements:    {sum(1 for c in counts if c >= 3)} "
          f"({sum(1 for c in counts if c >= 3) / len(counts) * 100:.1f}%)")

    # Build report
    report = {
        'total_identities': total_identities,
        'users_with_zero_entitlements': len(users_with_zero),
        'users_with_zero_ids': users_with_zero[:100],  # First 100
        'app_stats': app_stats,
        'distribution': {
            'min': int(np.min(counts)),
            'max': int(np.max(counts)),
            'mean': float(np.mean(counts)),
            'median': float(np.median(counts)),
            'pct_with_3_plus': float(sum(1 for c in counts if c >= 3) / len(counts) * 100)
        }
    }

    return success, report


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path("./out")

    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)

    success, report = validate_no_zero_entitlement_users(output_dir)

    if not success:
        sys.exit(1)