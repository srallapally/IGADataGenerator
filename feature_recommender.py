import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

pd.options.future.infer_string = True


def recommend_features(df_users, config: dict):
    max_features = config['max_num_features']
    skip_columns = config['skip_features']

    # === OPTIMIZATION CONFIG ===
    # Tune these based on your accuracy vs speed tradeoff
    MAX_UNIQUE_VALUES = 50  # Skip high-cardinality columns in MI
    MI_SAMPLE_SIZE = 5000  # Sample size for MI calculation (None = use all)
    MAX_MI_TARGETS = 20  # Limit number of targets for MI calculation
    # ===========================

    df_users = df_users.select_dtypes(include='str')

    categorical_columns = [
        col for col in df_users.columns
        if col not in skip_columns
    ]
    print(f'- Number of users: {len(df_users)}')

    df_users = df_users.fillna('')

    # --- BEGIN OPTIMIZATION: Filter high-cardinality columns early ---
    # This prevents one-hot encoding explosion
    low_cardinality_cols = [
        col for col in categorical_columns
        if df_users[col].nunique() <= MAX_UNIQUE_VALUES
    ]
    print(
        f'- Columns after cardinality filter ({MAX_UNIQUE_VALUES} max unique): {len(low_cardinality_cols)} of {len(categorical_columns)}')
    # --- END OPTIMIZATION ---

    # 1) Chi-square filtering (unchanged, already fast)
    chi2_results_all = {}
    top_features_all = {}
    for target in categorical_columns:
        chi2_results = {}
        for feature in categorical_columns:
            if feature == target:
                continue
            ct = pd.crosstab(df_users[feature], df_users[target])
            _, p, _, _ = chi2_contingency(ct)
            chi2_results[feature] = p
        chi2_results_all[target] = chi2_results
        sorted_feats = sorted(chi2_results.items(), key=lambda x: x[1])
        sig = [f for f, p in sorted_feats if p < 0.05][:10]
        top_features_all[target] = sig

    # --- BEGIN OPTIMIZATION: Sample data for MI calculation ---
    if MI_SAMPLE_SIZE and len(df_users) > MI_SAMPLE_SIZE:
        df_mi_sample = df_users.sample(n=MI_SAMPLE_SIZE, random_state=42)
        print(f'- Using {MI_SAMPLE_SIZE} sample rows for MI calculation')
    else:
        df_mi_sample = df_users
    # --- END OPTIMIZATION ---

    # --- BEGIN OPTIMIZATION: Limit targets for MI calculation ---
    # Only compute MI for targets that have the most significant features
    targets_with_features = [
        (target, feats) for target, feats in top_features_all.items()
        if feats and target in low_cardinality_cols
    ]
    # Sort by number of significant features (descending) and limit
    targets_with_features.sort(key=lambda x: len(x[1]), reverse=True)
    targets_for_mi = targets_with_features[:MAX_MI_TARGETS]
    print(f'- Computing MI for {len(targets_for_mi)} targets (of {len(targets_with_features)} with features)')
    # --- END OPTIMIZATION ---

    # 2) Mutual information - OPTIMIZED
    mi_results_all = {}
    for target, feats in targets_for_mi:
        if not feats:
            continue

        # --- BEGIN OPTIMIZATION: Only use low-cardinality features for one-hot ---
        feats_filtered = [f for f in feats if f in low_cardinality_cols]
        if not feats_filtered:
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

        mi_df_users = (
            pd.DataFrame({'Feature': one_hot.columns, 'MI': mi})
            .assign(Original=lambda d: d['Feature'].str.split('_').str[0])
        )
        agg = mi_df_users.groupby('Original', as_index=False)['MI'].max()
        mi_results_all[target] = agg

    # 3) Aggregate across targets (unchanged)
    aggregated = {}
    for target, feats in top_features_all.items():
        for feat in feats:
            p = chi2_results_all[target][feat]
            mi_val = None
            if target in mi_results_all:
                row = mi_results_all[target]
                filt = row['Original'] == feat
                if filt.any():
                    mi_val = row.loc[filt, 'MI'].iat[0]
            entry = aggregated.setdefault(feat, {
                'count': 0, 'p_values': [], 'mi_values': []
            })
            entry['count'] += 1
            entry['p_values'].append(p)
            if mi_val is not None:
                entry['mi_values'].append(mi_val)

    # 4) Build final list (unchanged)
    final = []
    for feat, data in aggregated.items():
        top3_series = df_users[feat].value_counts().head(3)
        final.append({
            'feature': feat,
            'times_recommended': data['count'],
            'min_chi2_p_value': min(data['p_values']),
            'max_mutual_info': max(data['mi_values']) if data['mi_values'] else None,
            'unique_count': df_users[feat].nunique(),
            'top_3_values': [
                f"{val} ({cnt})"
                for val, cnt in zip(top3_series.index, top3_series.values)
            ]
        })
    final.sort(key=lambda x: (
        -x['times_recommended'],
        x['min_chi2_p_value'],
        - (x['max_mutual_info'] or 0)
    ))
    return final[:max_features]