#!/usr/bin/env python3

import sys
import json
import argparse
import base64
import urllib.request
import urllib.error

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

DEFAULT_MAX_NUM_OF_FEATURES = 13
SKIP_COLUMNS = [
    'USR_ID', 'USR_NAME', 'USR_DEPARTMENT_NAME',
    'USR_DISPLAY_NAME', 'MANAGER_NAME', 'IS_ACTIVE'
]


def cramers_v(chi2_stat, n, r, c):
    """
    Calculate Cramér's V from chi-square statistic.

    Args:
        chi2_stat: Chi-square test statistic
        n: Total number of observations
        r: Number of rows in contingency table
        c: Number of columns in contingency table

    Returns:
        Cramér's V value (0 to 1)
    """
    if n == 0 or min(r - 1, c - 1) == 0:
        return 0.0
    return np.sqrt(chi2_stat / (n * min(r - 1, c - 1)))


def _http_request(url, method="GET", body=None, headers=None, timeout=60):
    """Minimal HTTP helper using urllib (no external deps)."""
    if headers is None:
        headers = {}

    if body is not None and not isinstance(body, (bytes, bytearray)):
        body = body.encode("utf-8")

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def load_es_index_to_df(
    es_host: str,
    es_port: int,
    es_user: str,
    es_password: str,
    es_index: str,
    sample_limit: int = 50000,
):
    """
    Load documents from an Elasticsearch index into a pandas DataFrame using a
    single _search request (no scroll), capped at sample_limit docs.

    Args:
        es_host: e.g. 'http://localhost' or 'https://my-es-host'
                 (if scheme is missing, 'https://' will be prefixed)
        es_port: e.g. 9200 or 443
        es_user: basic auth username
        es_password: basic auth password
        es_index: index name (e.g. 'users')
        sample_limit: max docs to pull (default: 50_000, clipped at ES page limit)

    Returns:
        pandas.DataFrame with one row per document (_source).
    """
    # Normalize base URL
    base = es_host
    if not base.startswith("http://") and not base.startswith("https://"):
        base = f"https://{base}"
    base = f"{base}:{es_port}"

    # Basic Auth header
    auth_str = f"{es_user}:{es_password}"
    auth_b64 = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
    headers = {
        "Authorization": f"Basic {auth_b64}",
        "Content-Type": "application/json",
    }

    # ES default max_result_window is typically 10_000; respect that
    MAX_ES_PAGE_SIZE = 10_000
    size = min(sample_limit, MAX_ES_PAGE_SIZE)

    query_body = json.dumps({
        "size": size,
        "query": {"match_all": {}}
    })

    search_url = f"{base}/{es_index}/_search"

    try:
        raw = _http_request(
            search_url,
            method="POST",
            body=query_body,
            headers=headers,
            timeout=120,  # a bit more generous than default 60
        )
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Error during search: {e.read().decode('utf-8')}") from e
    except Exception as e:
        raise RuntimeError(f"Error during search: {e}") from e

    data = json.loads(raw)
    hits = data.get("hits", {}).get("hits", [])

    docs = [h["_source"] for h in hits if "_source" in h]

    if not docs:
        raise ValueError(f"No documents found in index '{es_index}'.")

    df = pd.DataFrame(docs)
    return df


def compute_features_df(df: pd.DataFrame, max_features: int, verbose=False):
    """
    Compute feature recommendations with optional verbose output.

    Args:
        df: Input DataFrame
        max_features: Maximum number of features to return
        verbose: If True, print progress messages
    """
    # Replace string "NULL" with actual NaN
    df = df.replace(['NULL', 'null', 'Null'], np.nan)

    # Drop columns that are entirely empty or unnamed
    df = df.dropna(axis=1, how='all')
    unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
    df = df.drop(columns=unnamed_cols)

    if verbose:
        print(f"After cleaning: {len(df.columns)} columns, {len(df)} rows")

    # For very large datasets, sample rows
    LARGE_DATASET_THRESHOLD = 50000  # rows
    if len(df) > LARGE_DATASET_THRESHOLD:
        if verbose:
            print(
                f"Large dataset detected ({len(df)} rows). "
                f"Sampling {LARGE_DATASET_THRESHOLD} rows for analysis..."
            )
        df = df.sample(n=LARGE_DATASET_THRESHOLD, random_state=42)

    # Consider only 'object' dtype columns, excluding SKIP_COLUMNS
    categorical_columns = [
        col for col in df.columns
        if col not in SKIP_COLUMNS and df[col].dtype == 'object'
    ]

    if verbose:
        print(f"Found {len(categorical_columns)} categorical columns")

    if max_features >= len(categorical_columns):
        raise ValueError("max_features must be less than number of categorical columns.")

    # Limit columns with too many unique values
    MAX_UNIQUE_VALUES = 50

    categorical_columns = [
        col for col in categorical_columns
        if df[col].dropna().nunique() > 1 and df[col].dropna().nunique() <= MAX_UNIQUE_VALUES
    ]

    if verbose:
        print(
            "After filtering high-cardinality and single-value columns: "
            f"{len(categorical_columns)} columns remain"
        )

    if len(categorical_columns) < 2:
        raise ValueError("Not enough categorical columns with reasonable cardinality for analysis.")

    # 1) Chi-square filtering with Cramér's V calculation
    chi2_results_all = {}
    top_features_all = {}

    if verbose:
        print("Computing chi-square tests and Cramér's V...")

    for idx, target in enumerate(categorical_columns):
        if verbose and idx % 5 == 0:
            print(f"  Processing target column {idx + 1}/{len(categorical_columns)}: {target}")

        chi2_results = {}

        for feature in categorical_columns:
            if feature == target:
                continue

            # Filter out null values
            valid_mask = df[feature].notna() & df[target].notna()

            # Need at least 5 observations
            if valid_mask.sum() < 5:
                continue

            valid_feature = df[feature][valid_mask]
            valid_target = df[target][valid_mask]

            # Need at least 2 unique values in each
            if valid_feature.nunique() < 2 or valid_target.nunique() < 2:
                continue

            ct = pd.crosstab(valid_feature, valid_target)

            if ct.size == 0 or ct.sum().sum() == 0:
                continue

            chi2_stat, p, _, _ = chi2_contingency(ct)

            # Cramér's V
            n = ct.sum().sum()
            r, c = ct.shape
            v = cramers_v(chi2_stat, n, r, c)

            chi2_results[feature] = {'p_value': p, 'cramers_v': v}

        chi2_results_all[target] = chi2_results

        # Sort by p-value and keep significant ones
        sorted_feats = sorted(chi2_results.items(), key=lambda x: x[1]['p_value'])
        sig = [f for f, stats in sorted_feats if stats['p_value'] < 0.05][:10]
        top_features_all[target] = sig

    if verbose:
        print("Computing mutual information...")

    # 2) Mutual information
    mi_results_all = {}
    for target, feats in top_features_all.items():
        if not feats:
            continue

        # Only rows where target is not null
        target_valid_mask = df[target].notna()

        # Ensure features have valid data
        features_to_encode = []
        for feat in feats:
            if df[feat].notna().sum() > 0:
                features_to_encode.append(feat)

        if not features_to_encode:
            continue

        df_valid = df[target_valid_mask].copy()

        # One-hot encode features
        one_hot = pd.get_dummies(df_valid[features_to_encode], dummy_na=False)

        target_values = df_valid[target].values

        # Skip if target still has NaN
        if pd.isna(target_values).any():
            continue

        try:
            mi = mutual_info_classif(one_hot, target_values, random_state=42)
            mi_df = (
                pd.DataFrame({'Feature': one_hot.columns, 'MI': mi})
                .assign(Original=lambda d: d['Feature'].str.split('_').str[0])
            )
            agg = mi_df.groupby('Original', as_index=False)['MI'].max()
            mi_results_all[target] = agg
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not compute MI for target {target}: {e}")
            continue

    if verbose:
        print("Aggregating results...")

    # 3) Aggregate across targets
    aggregated = {}
    for target, feats in top_features_all.items():
        for feat in feats:
            p = chi2_results_all[target][feat]['p_value']
            v = chi2_results_all[target][feat]['cramers_v']

            mi_val = None
            if target in mi_results_all:
                row = mi_results_all[target]
                filt = row['Original'] == feat
                if filt.any():
                    mi_val = row.loc[filt, 'MI'].iat[0]

            entry = aggregated.setdefault(
                feat,
                {'count': 0, 'p_values': [], 'cramers_v_values': [], 'mi_values': []}
            )
            entry['count'] += 1
            entry['p_values'].append(p)
            entry['cramers_v_values'].append(v)
            if mi_val is not None:
                entry['mi_values'].append(mi_val)

    # 4) Build final list
    final = []
    for feat, data in aggregated.items():
        top3_series = df[feat].value_counts(dropna=True).head(3)

        final.append({
            'feature': feat,
            'times_recommended': data['count'],
            'min_chi2_p_value': min(data['p_values']),
            'max_cramers_v': max(data['cramers_v_values']) if data['cramers_v_values'] else None,
            'max_mutual_info': max(data['mi_values']) if data['mi_values'] else None,
            'unique_count': df[feat].dropna().nunique(),
            'top_3_values': [
                f"{val} ({cnt})"
                for val, cnt in zip(top3_series.index, top3_series.values)
            ]
        })

    # Sort: high count, low p-value, high Cramér's V, high MI
    final.sort(key=lambda x: (
        -x['times_recommended'],
        x['min_chi2_p_value'],
        -(x['max_cramers_v'] or 0),
        -(x['max_mutual_info'] or 0)
    ))

    if verbose:
        print(f"Analysis complete! Found {len(final)} features.")

    return final[:max_features]


def cli_mode_es(
    es_host: str,
    es_port: int,
    es_user: str,
    es_password: str,
    es_index: str,
    max_features: int,
    output_file: str = None,
):
    """
    Run the feature service in batch mode, using an Elasticsearch index as input.
    """
    print("=" * 60)
    print("FEATURE RECOMMENDATION SERVICE - ES MODE")
    print("=" * 60)
    print(f"Elasticsearch host : {es_host}:{es_port}")
    print(f"Index              : {es_index}")
    print(f"Max features       : {max_features}")
    print("=" * 60)

    try:
        df = load_es_index_to_df(es_host, es_port, es_user, es_password, es_index)
    except Exception as e:
        print(f"Error reading from Elasticsearch index '{es_index}': {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from index '{es_index}'")

    try:
        results = compute_features_df(df, max_features, verbose=True)
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("RESULTS (ES MODE)")
    print("=" * 60)

    if not results:
        print("No significant features were found.")
    else:
        print(f"Top {len(results)} recommended features:\n")
        print(f"{'#':<4} {'Feature':<25} {'Times':<7} {'p-value':<12} {'Cramér V':<10} {'MI':<10}")
        print("-" * 80)
        for idx, result in enumerate(results, 1):
            feature = result['feature'][:24]
            times = result['times_recommended']
            p_val = f"{result['min_chi2_p_value']:.4e}"
            cramer = (
                f"{result['max_cramers_v']:.4f}"
                if result['max_cramers_v'] is not None
                else "N/A"
            )
            mi = (
                f"{result['max_mutual_info']:.4f}"
                if result['max_mutual_info'] is not None
                else "N/A"
            )
            print(f"{idx:<4} {feature:<25} {times:<7} {p_val:<12} {cramer:<10} {mi:<10}")
            print(f"     Top values: {', '.join(result['top_3_values'])}")
            print()

    if output_file:
        output_data = {
            'source': f"elasticsearch:{es_index}",
            'rows': len(df),
            'columns': len(df.columns),
            'max_features_requested': max_features,
            'recommended_features': results,
        }

        try:
            import subprocess, tempfile, os

            local_tmp = tempfile.mktemp(suffix=".json")
            with open(local_tmp, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)

            subprocess.check_call(["gsutil", "cp", local_tmp, output_file])
            os.remove(local_tmp)

            print(f"\nResults written to GCS: {output_file}")

        except Exception as e:
            print(f"Warning: Failed to write output JSON file to GCS: {e}")



def main():
    """Dataproc entry point: analyze features from an Elasticsearch index."""
    parser = argparse.ArgumentParser(
        description='Feature Recommendation Service (Elasticsearch source)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:

  python feature_recommender.py \\
      --es-host https://my-es-host \\
      --es-port 443 \\
      --es-user elastic \\
      --es-password ******** \\
      --es-index users \\
      --max-features 15 \\
      --output results.json
        """
    )

    parser.add_argument(
        '--es-host',
        type=str,
        required=True,
        help="Elasticsearch host (e.g., http://localhost or https://cloud.es.io)"
    )
    parser.add_argument(
        '--es-port',
        type=int,
        default=9200,
        help="Elasticsearch port (default: 9200)"
    )
    parser.add_argument(
        '--es-user',
        type=str,
        required=True,
        help="Elasticsearch username for basic auth"
    )
    parser.add_argument(
        '--es-password',
        type=str,
        required=True,
        help="Elasticsearch password for basic auth"
    )
    parser.add_argument(
        '--es-index',
        type=str,
        required=True,
        help="Elasticsearch index name to analyze (e.g., 'users')"
    )
    parser.add_argument(
        '--max-features', '-m',
        type=int,
        default=DEFAULT_MAX_NUM_OF_FEATURES,
        help=f'Maximum number of features to return (default: {DEFAULT_MAX_NUM_OF_FEATURES})'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON file path (optional)'
    )

    args = parser.parse_args()

    cli_mode_es(
        es_host=args.es_host,
        es_port=args.es_port,
        es_user=args.es_user,
        es_password=args.es_password,
        es_index=args.es_index,
        max_features=args.max_features,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
