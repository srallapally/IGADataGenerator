import pandas as pd

from feature_recommender_aic import flatten_user_data, export_identities_csv
from feature_recommender import recommend_features


def test_flatten_user_data_handles_manager_ref():
    users = [
        {
            "userName": "jsmith",
            "mail": "jsmith@example.com",
            "manager": {"_ref": "managed/alpha_user/manager-123"},
        },
        {
            "userName": "adoe",
            "mail": "adoe@example.com",
            "manager": {"_ref": "manager-456"},
        },
    ]

    df = flatten_user_data(users)

    assert "manager" not in df.columns
    assert "manager_id" in df.columns
    assert df.loc[0, "manager_id"] == "manager-123"
    assert df.loc[1, "manager_id"] == "manager-456"


def test_export_identities_csv_respects_exclude_attributes(tmp_path):
    df = pd.DataFrame(
        [
            {
                "userName": "jsmith",
                "mail": "jsmith@example.com",
                "department": "Engineering",
                "manager_id": "manager-123",
            }
        ]
    )
    output_path = tmp_path / "identities.csv"

    export_identities_csv(df, exclude_attributes=["mail", "manager_id"], output_path=str(output_path))

    exported = pd.read_csv(output_path)
    assert list(exported.columns) == ["userName", "department"]


def test_recommend_features_shape_and_ordering():
    df = pd.DataFrame(
        {
            "department": [
                "Eng",
                "Eng",
                "Eng",
                "Eng",
                "Sales",
                "Sales",
                "Sales",
                "Sales",
            ],
            "job_title": [
                "Engineer",
                "Engineer",
                "Engineer",
                "Engineer",
                "SalesRep",
                "SalesRep",
                "SalesRep",
                "SalesRep",
            ],
            "city": [
                "NY",
                "SF",
                "LA",
                "CHI",
                "NY",
                "SF",
                "LA",
                "CHI",
            ],
        }
    )

    config = {
        "max_num_features": 5,
        "skip_features": [],
        "max_unique_values": 10,
        "feature_selection_method": "cramers_v",
        "cramers_v_threshold": 0.5,
    }

    recommendations = recommend_features(df, config)

    assert len(recommendations) == 2
    assert [rec["feature"] for rec in recommendations] == ["job_title", "department"]
