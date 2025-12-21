import argparse
import json
import ssl
import sys
import urllib.parse
import urllib.request
from typing import List, Dict, Any

import pandas as pd

from feature_recommender import recommend_features

# Disable SSL verification for development/QA environments if needed
ssl._create_default_https_context = ssl._create_unverified_context


def get_access_token(base_url: str, client_id: str, client_secret: str) -> str:
    """
    Obtains an access token using the client_credentials grant type.
    """
    token_url = f"{base_url}/am/oauth2/alpha/access_token"
    print(f"DEBUG: Requesting token from {token_url}")
    print(f"DEBUG: Client ID: {client_id}")
    # Prepare data for POST request
    data = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "scope": "fr:idm:*",
        "client_id": client_id,
        "client_secret": client_secret
    }).encode("utf-8")

    req = urllib.request.Request(token_url, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                response_data = json.loads(response.read().decode("utf-8"))
                print(f"DEBUG: Response: {response_data["access_token"]}")
                return response_data["access_token"]
            else:
                raise Exception(
                    f"Failed to get token. Status: {response.status}, Body: {response.read().decode('utf-8')}")
    except urllib.error.HTTPError as e:
        raise Exception(f"HTTP Error getting token: {e.code} {e.reason} - {e.read().decode('utf-8')}")
    except Exception as e:
        raise Exception(f"Error getting token: {str(e)}")


def fetch_users(base_url: str, access_token: str, page_size: int = 10) -> List[Dict[str, Any]]:
    """
    Fetches all users from IDM using pagination.
    """
    users = []
    offset = 0
    more_results = True

    base_endpoint = f"{base_url}/openidm/managed/alpha_user"

    # Define fields to fetch - basic profile fields usually needed for analysis
    # Adjust this list based on specific needs or config
    fields = "userName,givenName,sn,mail,accountStatus,department,jobTitle,city,country,employeeNumber,manager,costCenter,division"

    print(f"Starting user fetch from {base_endpoint}...")

    while more_results:
        query_params = urllib.parse.urlencode({
            "_queryFilter": "true",
            "_pageSize": page_size,
            "_totalPagedResultsPolicy": "EXACT",
    #        "_fields": fields,
            "_pagedResultsOffset": offset
        })

        url = f"{base_endpoint}?{query_params}"

        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {access_token}")

        try:
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    result = data.get("result", [])

                    if not result:
                        more_results = False
                        break

                    users.extend(result)
                    print(f"Fetched {len(result)} users (Total: {len(users)})")

                    # Check if we've reached the end
                    if len(result) < page_size:
                        more_results = False
                    else:
                        offset += page_size
                else:
                    print(f"Error fetching users: {response.status}")
                    more_results = False
        except urllib.error.HTTPError as e:
            print(f"HTTP Error fetching users at offset {offset}: {e.code} {e.read().decode('utf-8')}")
            more_results = False
        except Exception as e:
            print(f"Error fetching users: {str(e)}")
            more_results = False

    return users


def flatten_user_data(users: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Flattens user data, handling nested objects like 'manager'.
    """
    flattened_data = []

    for user in users:
        flat_user = user.copy()

        # Handle manager specifically if it's a relationship object
        if "manager" in user and isinstance(user["manager"], dict):
            # IDM relationship often looks like {"_ref": "managed/alpha_user/UUID", ...}
            # We usually just want the ID or a display name
            ref = user["manager"].get("_ref", "")
            flat_user["manager_id"] = ref.split("/")[-1] if "/" in ref else ref
            del flat_user["manager"]  # Remove the complex object

        flattened_data.append(flat_user)

    return pd.DataFrame(flattened_data)


# --- BEGIN ADDITION: Function to export users with all attributes to CSV ---
def export_identities_csv(df: pd.DataFrame, exclude_attributes: List[str],
                          output_path: str = "identities.csv") -> None:
    """
    Exports users with all attributes to a CSV file, excluding specified attributes.

    Args:
        df (pd.DataFrame): The full user DataFrame
        exclude_attributes (List[str]): List of attribute names to exclude from export
        output_path (str): Path to the output CSV file
    """
    # Get all columns except those in the exclude list
    columns_to_export = [col for col in df.columns if col not in exclude_attributes]

    if not columns_to_export:
        print("Warning: No columns available to export to CSV (all columns excluded).")
        return

    # Create the subset DataFrame
    df_export = df[columns_to_export].copy()

    # Export to CSV
    df_export.to_csv(output_path, index=False)
    print(f"Identities exported to {output_path} with {len(df_export)} rows and {len(columns_to_export)} columns")
    print(f"  Columns: {columns_to_export}")
    if exclude_attributes:
        print(f"  Excluded: {exclude_attributes}")


# --- END ADDITION ---


def main():
    parser = argparse.ArgumentParser(description='ForgeRock IDM Feature Recommendation Pipeline')
    parser.add_argument('--config', required=False, default='config.json', help='Path to configuration JSON file')
    parser.add_argument('--output', '-o', default='recommended_features.json', help='Output file for recommendations')
    # --- BEGIN ADDITION: New argument for identities CSV output ---
    parser.add_argument('--identities-output', '-i', default='identities.csv',
                        help='Output CSV file for user identities with selected attributes')
    # --- END ADDITION ---

    args = parser.parse_args()

    # 1. Load Config
    config_path = args.config if args.config else 'config.json'
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Failed to load config file: {e}")
        sys.exit(1)

    base_url = config.get("base_url")
    client_id = config.get("client_id")
    client_secret = config.get("client_secret")
    page_size = config.get("page_size", 100)

    if not all([base_url, client_id, client_secret]):
        print("Error: Config file must contain 'base_url', 'client_id', and 'client_secret'.")
        sys.exit(1)

    # 2. Get Access Token
    print("Authenticating...")
    try:
        token = get_access_token(base_url, client_id, client_secret)
        print("Authentication successful.")
    except Exception as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)

    # 3. Fetch Users
    users = fetch_users(base_url, token, page_size)
    if not users:
        print("No users found or failed to fetch users.")
        sys.exit(1)

    # 4. Prepare DataFrame
    print("Processing data...")
    df = flatten_user_data(users)
    print(f"DataFrame created with shape: {df.shape}")

    # --- BEGIN FIX: Export identities CSV BEFORE recommend_features modifies the DataFrame ---
    # 5. Export Identities CSV (using original DataFrame)
    export_exclude_attributes = config.get('export_exclude_attributes', [])
    try:
        export_identities_csv(df, export_exclude_attributes, args.identities_output)
    except Exception as e:
        print(f"Failed to export identities CSV: {e}")
        import traceback
        traceback.print_exc()
    # --- END FIX ---

    # 6. Recommend Features
    # Ensure config has the right keys for the recommender
    rec_config = {
        'max_num_features': config.get('max_num_features', 15),
        'skip_features': config.get('skip_features', []),
        'max_unique_values': config.get('max_unique_values', 50),
        'feature_selection_method': config.get('feature_selection_method', 'cramers_v'),
        'chi_square_p_threshold': config.get('chi_square_p_threshold', 0.05),
        'cramers_v_threshold': config.get('cramers_v_threshold', 0.1)
    }

    print("Running feature recommendation...")
    try:
        recommendations = recommend_features(df, rec_config)
    except Exception as e:
        print(f"Error during feature recommendation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 7. Output Results
    print("\nTop Recommended Features:")
    print("-" * 30)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['feature']}")
        print(f"   - Recommended {rec['times_recommended']} times")
        print(f"   - Unique values: {rec['unique_count']}")
        print(f"   - Sample values: {', '.join(rec['top_3_values'])}")
        print()

    try:
        with open(args.output, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"Recommendations saved to {args.output}")
    except Exception as e:
        print(f"Failed to save output: {e}")


if __name__ == "__main__":
    main()
