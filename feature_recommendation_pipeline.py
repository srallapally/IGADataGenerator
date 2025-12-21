import pandas as pd
import json
import argparse
import sys
from pathlib import Path
from feature_recommender import recommend_features

def main():
    parser = argparse.ArgumentParser(description='Recommend features from a local IGA identities CSV.')
    parser.add_argument('input_file', help='Path to the identities.csv file')
    parser.add_argument('--output', '-o', help='Path to save recommendations (JSON)', default='recommended_features.json')
    parser.add_argument('--max-features', '-m', type=int, default=10, help='Maximum features to recommend')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)
        
    print(f"Reading dataset: {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Basic configuration for the recommender
    # We skip unique identifiers and common PII that shouldn't be used for role mining
    config = {
        'max_num_features': args.max_features,
        'skip_features': [
            'user_id', 'user_name', 'email', 'first_name', 'last_name', 'distinguished_name', 'external_id'
        ]
    }

    print("Analyzing features (this may take a minute for large datasets)...")
    recommendations = recommend_features(df, config)

    # Print summary to console
    print("\nTop Recommended Features:")
    print("-" * 30)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['feature']}")
        print(f"   - Recommended {rec['times_recommended']} times")
        print(f"   - Unique values: {rec['unique_count']}")
        print(f"   - Sample values: {', '.join(rec['top_3_values'])}")
        print()

    # Save to file
    with open(args.output, 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    print(f"Full analysis saved to: {args.output}")

if __name__ == "__main__":
    main()
