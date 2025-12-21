import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

pd.options.future.infer_string = True


def _cramers_v(chi2_stat, n, r, c):
    """
    Calculate Cramér's V from chi-square statistic.

    Cramér's V measures the strength of association between two categorical variables.
    Range: 0 (no association) to 1 (perfect association)

    Args:
        chi2_stat (float): Chi-square test statistic
        n (int): Total number of observations
        r (int): Number of rows in contingency table
        c (int): Number of columns in contingency table

    Returns:
        float: Cramér's V value (0 to 1)
    """
    if n == 0 or min(r - 1, c - 1) == 0:
        return 0.0
    return np.sqrt(chi2_stat / (n * min(r - 1, c - 1)))


def _process_attribute_value(value):
    """
    Process attribute values to handle:
    - Multi-valued attributes (take first value)
    - Numeric attributes (treat as strings)
    - Missing/null values

    Args:
        value: The attribute value to process

    Returns:
        str: Processed string value
    """
    if pd.isna(value) or value == '':
        return ''

    value_str = str(value)

    # For multi-valued attributes (typically separated by semicolon or pipe),
    # extract the first value
    if ';' in value_str:
        return value_str.split(';')[0].strip()
    if '|' in value_str:
        return value_str.split('|')[0].strip()

    return value_str


def recommend_features(df_users, config: dict):
    """
    Recommend features using statistical significance and mutual information.

    This function performs 4-stage filtering:
    1. Cardinality filter (max unique values)
    2. Statistical significance (Cramér's V or chi-square)
    3. Low-cardinality target restriction for MI
    4. Mutual information aggregation

    Args:
        df_users (pd.DataFrame): User data with columns as features
        config (dict): Configuration dictionary with keys:
            - max_num_features: Max features to return
            - skip_features: List of features to skip
            - max_unique_values: Max unique values per feature (cardinality filter)
            - feature_selection_method: 'chi_square' or 'cramers_v'
            - chi_square_p_threshold: P-value threshold for chi-square
            - cramers_v_threshold: Minimum Cramér's V value

    Returns:
        list: List of recommended features (sorted by quality)
    """
    max_features = config['max_num_features']
    skip_columns = config['skip_features']

    # === OPTIMIZATION CONFIG ===
    # Tune these based on your accuracy vs speed tradeoff
    MAX_UNIQUE_VALUES = config.get('max_unique_values', 50)
    MI_SAMPLE_SIZE = 5000  # Sample size for MI calculation (None = use all)
    MAX_MI_TARGETS = 20  # Limit number of targets for MI calculation

    # Feature selection method and thresholds
    FEATURE_SELECTION_METHOD = config.get('feature_selection_method', 'cramers_v').lower()
    CHI_SQUARE_P_THRESHOLD = config.get('chi_square_p_threshold', 0.05)
    CRAMERS_V_THRESHOLD = config.get('cramers_v_threshold', 0.1)
    # ===========================

    # Validate feature selection method
    if FEATURE_SELECTION_METHOD not in ['chi_square', 'cramers_v']:
        raise ValueError(
            f"Invalid feature_selection_method: {FEATURE_SELECTION_METHOD}. "
            f"Must be 'chi_square' or 'cramers_v'")

    # Convert all columns to string type to handle numeric and multi-valued attributes
    df_users = df_users.astype(str)

    # Process multi-valued and numeric attributes
    # Convert all values to strings, taking first value for multi-valued attributes
    for col in df_users.columns:
        df_users[col] = df_users[col].apply(_process_attribute_value)

    categorical_columns = [
        col for col in df_users.columns
        if col not in skip_columns
    ]

    print(f'- Number of users: {len(df_users)}')
    print(f'- Feature selection method: {FEATURE_SELECTION_METHOD}')
    print(f'- All available columns: {len(df_users.columns)}')
    print(f'- After skip_features filter: {len(categorical_columns)}')

    df_users = df_users.fillna('')

    # --- FILTER 1: Cardinality Filter ---
    low_cardinality_cols = [
        col for col in categorical_columns
        if df_users[col].nunique() <= MAX_UNIQUE_VALUES
    ]
    high_cardinality_cols = [
        col for col in categorical_columns
        if df_users[col].nunique() > MAX_UNIQUE_VALUES
    ]
    print(f'\n=== FILTER 1: Cardinality (max {MAX_UNIQUE_VALUES} unique values) ===')
    print(f'- Survived cardinality filter: {len(low_cardinality_cols)}')
    if low_cardinality_cols:
        print(f'  Columns: {low_cardinality_cols}')
    print(f'- Filtered out (too high cardinality): {len(high_cardinality_cols)}')
    if high_cardinality_cols:
        for col in sorted(high_cardinality_cols):
            unique_count = df_users[col].nunique()
            print(f'  - {col}: {unique_count} unique values')
    else:
        print('  (None)')
    # --for col in high_cardinality_cols:
    # --     unique_count = df_users[col].nunique()
    #    print(f'  - {col}: {unique_count} unique values')

    # --- FILTER 2: Statistical Significance Filter ---
    print(f'\n=== FILTER 2: Statistical Significance (Cramér\'s V >= {CRAMERS_V_THRESHOLD}) ===')
    metric_results_all = {}
    top_features_all = {}
    all_feature_scores = {}  # Track all scores for debugging

    for target in categorical_columns:
        metric_results = {}
        for feature in categorical_columns:
            if feature == target:
                continue

            ct = pd.crosstab(df_users[feature], df_users[target])
            chi2_stat, p_value, _, _ = chi2_contingency(ct)

            if FEATURE_SELECTION_METHOD == 'chi_square':
                # Use chi-square p-value
                metric_results[feature] = p_value
            else:  # cramers_v
                # Use Cramér's V (effect size)
                n = len(df_users)
                r, c = ct.shape
                v = _cramers_v(chi2_stat, n, r, c)
                metric_results[feature] = v

        metric_results_all[target] = metric_results

        # Filter based on selected method
        if FEATURE_SELECTION_METHOD == 'chi_square':
            sorted_feats = sorted(metric_results.items(), key=lambda x: x[1])
            sig = [f for f, p in sorted_feats if p < CHI_SQUARE_P_THRESHOLD][:10]
        else:  # cramers_v
            sorted_feats = sorted(metric_results.items(), key=lambda x: x[1], reverse=True)
            sig = [f for f, v in sorted_feats if v >= CRAMERS_V_THRESHOLD][:10]

        top_features_all[target] = sig

        # Store all scores for debugging
        all_feature_scores[target] = dict(sorted_feats)

    # Print detailed results
    total_sig = sum(len(feats) for feats in top_features_all.values())
    print(f'- Total feature-target associations meeting threshold: {total_sig}')
    print(f'\nDetailed breakdown by target:')
    for target in categorical_columns:
        features = top_features_all[target]
        print(f'\n  Target: {target}')
        if features:
            print(f'    Features meeting threshold: {len(features)}')
            for feat in features[:5]:  # Show top 5
                score = metric_results_all[target][feat]
                print(f'      - {feat}: {score:.4f}')
        else:
            print(f'    Features meeting threshold: 0')
            # Show top candidates that didn't make it
            scores = all_feature_scores[target]
            top_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f'    Top candidates (didn\'t meet threshold):')
            for feat, score in top_candidates:
                print(f'      - {feat}: {score:.4f}')

    # --- FILTER 3: Low-cardinality restriction for MI ---
    print(f'\n=== FILTER 3: Low-cardinality restriction for MI calculation ===')
    targets_with_features = [
        (target, feats) for target, feats in top_features_all.items()
        if feats and target in low_cardinality_cols
    ]
    targets_with_features.sort(key=lambda x: len(x[1]), reverse=True)
    targets_for_mi = targets_with_features[:MAX_MI_TARGETS]
    print(f'- Targets with significant low-cardinality features: {len(targets_with_features)}')
    print(f'- Computing MI for {len(targets_for_mi)} targets (limited to {MAX_MI_TARGETS})')

    # --- BEGIN FIX: Collect all targets with significant features (for aggregation) ---
    # This includes ALL targets that have features meeting the threshold,
    # regardless of whether the target itself is low-cardinality.
    # MI will only be computed for low-cardinality targets, but aggregation
    # will include features from all targets.
    all_targets_with_features = [
        (target, feats) for target, feats in top_features_all.items()
        if feats
    ]
    print(f'- Total targets with significant features (for aggregation): {len(all_targets_with_features)}')
    # --- END FIX ---

    # Early exit if no targets have any significant features
    if not all_targets_with_features:
        print('\n⚠ WARNING: No targets have features meeting the threshold!')
        print('Consider:')
        print('  - Lowering cramers_v_threshold')
        print('  - Increasing max_unique_values')
        print('  - Reviewing skip_features list')
        return []

    # --- BEGIN OPTIMIZATION: Sample data for MI calculation ---
    if MI_SAMPLE_SIZE and len(df_users) > MI_SAMPLE_SIZE:
        df_mi_sample = df_users.sample(n=MI_SAMPLE_SIZE, random_state=42)
        print(f'- Using {MI_SAMPLE_SIZE} sample rows for MI calculation')
    else:
        df_mi_sample = df_users
    # --- END OPTIMIZATION ---

    # --- FILTER 4: Mutual Information Calculation ---
    # MI is only calculated for low-cardinality targets
    print(f'\n=== FILTER 4: Mutual Information Calculation ===')
    mi_results_all = {}
    for target, feats in targets_for_mi:
        if not feats:
            continue

        # --- BEGIN OPTIMIZATION: Only use low-cardinality features for one-hot ---
        feats_filtered = [f for f in feats if f in low_cardinality_cols]
        if not feats_filtered:
            print(f'  ⚠ Warning: No low-cardinality features for target {target}')
            continue
        # --- END OPTIMIZATION ---

        one_hot = pd.get_dummies(df_mi_sample[feats_filtered])

        # --- BEGIN OPTIMIZATION: Use discrete_features=True for categorical data ---
        # This is faster and more appropriate for categorical features
        mi = mutual_info_classif(
            one_hot,
            df_mi_sample[target],
            discrete_features=True,  # Faster for categorical data
            random_state=42
        )
        # --- END OPTIMIZATION ---

        # Fix: Use rsplit with maxsplit=1 to handle column names with underscores
        mi_df_users = (
            pd.DataFrame({'Feature': one_hot.columns, 'MI': mi})
            .assign(Original=lambda d: d['Feature'].str.rsplit(pat='_', n=1).str[0])
        )
        agg = mi_df_users.groupby('Original', as_index=False)['MI'].max()
        mi_results_all[target] = agg

    print(f'- MI calculations completed for {len(mi_results_all)} targets')

    # --- FILTER 5: Aggregation across targets ---
    print(f'\n=== FILTER 5: Aggregation and Final Ranking ===')
    aggregated = {}

    # --- BEGIN FIX: Aggregate features from ALL targets with significant features ---
    # Previously this only iterated over targets_for_mi (low-cardinality targets).
    # Now we iterate over all_targets_with_features to include features from
    # high-cardinality targets as well. MI values will only be available for
    # features associated with low-cardinality targets.
    for target, feats in all_targets_with_features:
        if not feats:
            continue

        for feat in feats:
            metric_val = metric_results_all[target][feat]
            mi_val = None

            # Get MI value if it was calculated (only for low-cardinality targets)
            if target in mi_results_all:
                row = mi_results_all[target]
                filt = row['Original'] == feat
                if filt.any():
                    mi_val = row.loc[filt, 'MI'].iat[0]

            entry = aggregated.setdefault(feat, {
                'count': 0, 'metric_values': [], 'mi_values': []
            })
            entry['count'] += 1
            entry['metric_values'].append(metric_val)
            if mi_val is not None:
                entry['mi_values'].append(mi_val)
    # --- END FIX ---

    print(f'- Unique features in final aggregation: {len(aggregated)}')
    if aggregated:
        print(f'  Features: {list(aggregated.keys())}')

    # 4) Build final list
    final = []
    for feat, data in aggregated.items():
        top3_series = df_users[feat].value_counts().head(3)

        if FEATURE_SELECTION_METHOD == 'chi_square':
            metric_stat = min(data['metric_values'])  # For p-values, min is best
            metric_label = 'min_chi2_p_value'
        else:  # cramers_v
            metric_stat = max(data['metric_values'])  # For Cramér's V, max is best
            metric_label = 'max_cramers_v'

        result = {
            'feature': feat,
            'times_recommended': data['count'],
            metric_label: metric_stat,
            'max_mutual_info': max(data['mi_values']) if data['mi_values'] else None,
            'unique_count': df_users[feat].nunique(),
            'top_3_values': [
                f"{val} ({cnt})"
                for val, cnt in zip(top3_series.index, top3_series.values)
            ]
        }
        final.append(result)

    # Sort by: times recommended (desc), then metric (best for method), then MI (desc)
    if FEATURE_SELECTION_METHOD == 'chi_square':
        final.sort(key=lambda x: (
            -x['times_recommended'],
            x['min_chi2_p_value'],
            -(x['max_mutual_info'] or 0)
        ))
    else:  # cramers_v
        final.sort(key=lambda x: (
            -x['times_recommended'],
            -x['max_cramers_v'],
            -(x['max_mutual_info'] or 0)
        ))

    print(f'\n=== FINAL RESULTS ===')
    print(f'- Top {max_features} features selected from {len(final)} candidates')

    return final[:max_features]