#!/usr/bin/env python3

"""
generate_synthetic_iga.py

Generates a synthetic Identity Governance (IGA) dataset including identities,
entitlements, and assignments. The generation process is based on latent
patterns and includes a calibration loop to ensure generated association
rules (feature -> entitlement) meet a target confidence distribution.
"""

import argparse
import json
import logging
import random
import sys
import csv  # Added for quoting
from pathlib import Path
from typing import (
    Any, Dict, List, Set, Tuple, Optional, Callable, DefaultDict
)
from dataclasses import dataclass, field
from collections import Counter, defaultdict

# Try to import external libraries, provide guidance if missing
try:
    import numpy as np
    from numpy.random import Generator, PCG64
    import pandas as pd
    from faker import Faker
except ImportError:
    print(
        "Error: Missing required libraries. Please install them via:",
        file=sys.stderr
    )
    print("pip install pandas numpy faker", file=sys.stderr)
    sys.exit(1)

# --- Default Configuration ---

DEFAULT_CONFIG = {
    "global": {
        "seed": 42,
        "num_users": 10000,
        "num_apps": 10,
        "num_entitlements": 12000,
    },
    "identity": {
        "departments": {
            "Engineering": 0.20,
            "Sales": 0.15,
            "Support": 0.15,
            "Finance": 0.10,
            "HR": 0.05,
            "IT": 0.10,
            "Marketing": 0.10,
            "Legal": 0.05,
            "Operations": 0.10,
        },
        "business_units": ["Corp", "Product", "Services", "Shared"],
        "job_families_by_dept": {
            "Engineering": ["SWE", "SRE", "Data", "PM", "Sec"],
            "Sales": ["AE", "SE", "SalesOps", "BDR"],
            "Support": ["CSM", "TechSupport", "SupportOps"],
            "Finance": ["Finance", "Accounting", "Procurement"],
            "HR": ["HRBP", "Recruiting", "Comp", "PeopleOps"],
            "IT": ["ITOps", "Helpdesk", "Infra", "BizSys"],
            "Marketing": ["ProductMktg", "DemandGen", "CorpComm"],
            "Legal": ["Counsel", "Paralegal", "Compliance"],
            "Operations": ["BizOps", "SupplyChain", "Logistics"],
        },
        "job_levels": {
            "values": [1, 2, 3, 4, 5, 6, 7],
            "probs": [0.10, 0.20, 0.30, 0.20, 0.12, 0.05, 0.03],  # Pyramidal
        },
        "job_titles_map": {
            "SWE": {
                1: "Associate Engineer", 2: "Software Engineer",
                3: "Senior Engineer", 4: "Staff Engineer",
                5: "Senior Staff Engineer", 6: "Principal Engineer",
                7: "Distinguished Engineer"
            },
            "AE": {
                1: "Associate AE", 2: "Account Executive",
                3: "Senior AE", 4: "Major Accounts AE",
                5: "Strategic AE", 6: "Regional VP", 7: "VP Sales"
            },
            "CSM": {
                1: "Associate CSM", 2: "CSM", 3: "Senior CSM",
                4: "Strategic CSM", 5: "CSM Manager", 6: "Director CS",
                7: "VP CS"
            },
            "Data": {1: "Data Analyst", 2: "Data Scientist", 3: "Sr. Data Scientist",
                     4: "Staff Data Scientist", 5: "Principal Data Scientist",
                     6: "Director, Data", 7: "VP, Data"}
        },
        "countries": {
            "US": 0.60,
            "GB": 0.15,
            "IN": 0.15,
            "DE": 0.05,
            "AU": 0.05,
        },
        "sites_by_country": {
            # Zipf-like distribution (manually defined for simplicity)
            "US": {"SFO": 0.3, "NYC": 0.2, "AUS": 0.15, "CHI": 0.1, "SEA": 0.1,
                   "BOS": 0.05, "ATL": 0.05, "RMT": 0.05},
            "GB": {"LON": 0.7, "MAN": 0.2, "EDI": 0.1},
            "IN": {"BLR": 0.5, "MUM": 0.3, "DEL": 0.2},
            "DE": {"BER": 0.6, "MUN": 0.4},
            "AU": {"SYD": 0.7, "MEL": 0.3},
        },
        "employment_types": {
            "FTE": 0.88,
            "Contractor": 0.10,
            "Intern": 0.02,
        },
        # --- BEGIN ChatGPT change: Phase 1 - add line_of_business and status config ---
        "lines_of_business": {
            "Retail": 0.20,
            "Corporate": 0.25,
            "SMB": 0.15,
            "PublicSector": 0.10,
            "Internal": 0.15,
            "SharedServices": 0.15,
        },
        "status_probs": {
            "active": 0.93,
            "inactive": 0.07,
        },
        # --- END ChatGPT change: Phase 1 - add line_of_business and status config ---
        "manager_prob": 0.12,
        "tenure_beta_params": {"a": 2.0, "b": 5.0, "scale": 15.0},  # Skewed right
    },
    "entitlement": {
        "app_names": [
            "Epic", "Salesforce", "GitHub", "AWS", "Zendesk",
            "Tableau", "Snowflake", "Slack", "GDrive", "Workday"
        ],
        "app_size_dist": {"big": 2, "medium": 4, "small": 4},
        "app_size_ents": {
            "big": (200, 250),
            "medium": (80, 120),
            "small": (40, 80),
        },
        "ent_types": [
            "role", "group", "project-role", "repo-permission",
            "dataset-access", "channel-access", "license-tier",
            "template-access"
        ],
        "criticality_probs": {
            "High": 0.15,
            "Medium": 0.40,
            "Low": 0.45,
        },
        "epic_templates": [
            "INP - RN", "INP - MD", "AMB - MA", "AMB - Sched",
            "RX - Pharmacist", "LAB - Tech", "RAD - Tech", "BILL - Coder",
            "ADMIN - Super"
        ],
        "num_bundles": 20,
        "bundle_size_range": (5, 20),
    },
    "pattern": {
        "num_patterns": 50,
        "antecedent_size_range": (1, 3),
        "confidence_bands": {
            "High": 0.6,
            "Mid": 0.3,
            "Low": 0.1,
        },
        "band_probs": {
            "High": {"core": (0.75, 0.95), "opt": (0.60, 0.75)},
            "Mid": {"core": (0.50, 0.70), "opt": (0.30, 0.50)},
            "Low": {"core": (0.20, 0.40), "opt": (0.05, 0.25)},
        },
        "core_ent_pct_range": (0.4, 0.8),
        "co_grant_boost": 0.15,
        "co_grant_pairs_per_bundle": 2,
        "super_bundle_prob_range": (0.2, 0.4),
    },
    "noise": {
        "background_ent_pct": 0.12,
        "background_user_pct": 0.05,
        "outlier_user_pct": 0.03,
        "outlier_ent_count_range": (1, 3),
        "collision_prob_reduction": 0.15,
    },
    "calibration": {
        "target_buckets": {
            "High": 0.6,
            "Mid": 0.3,
            "Low": 0.1,
        },
        "tolerance": 0.05,
        "max_iterations": 3,
        "learning_rate_alpha": 0.2,
        "target_midpoints": {
            "High": 0.85,  # Mid of (0.75, 0.95)
            "Mid": 0.60,  # Mid of (0.50, 0.70)
            "Low": 0.30,  # Mid of (0.20, 0.40)
        },
        "bucket_thresholds": {
            "High": 0.7,
            "Mid": 0.4,
        }
    }
}


# --- Data Structures ---
# --- BEGIN ChatGPT change: Phase 1 - extend Identity schema ---
@dataclass
class Identity:
    """Represents a user identity."""
    user_id: str
    user_name: str  # Derived from first_name + last_name, unique, alphanumeric only
    first_name: str
    last_name: str
    email: str
    department: str
    business_unit: str
    line_of_business: str
    job_family: str
    job_level: int
    title: str
    location_country: str
    location_site: str
    employment_type: str
    cost_center: str
    manager_flag: str
    manager_id: Optional[str]
    status: str  # "active" or "inactive"
    tenure_years: float
# --- END ChatGPT change: Phase 1 - extend Identity schema ---

@dataclass
class Entitlement:
    """Represents an access entitlement."""
    entitlement_id: str
    app_id: str
    app_name: str
    entitlement_name: str
    entitlement_type: str
    criticality: str
    scope: str
    description: str


@dataclass
class Pattern:
    """Represents a latent access pattern."""
    id: str
    antecedent: Dict[str, Any]
    target_band: str
    core_ents_p: Dict[str, float] = field(default_factory=dict)
    opt_ents_p: Dict[str, float] = field(default_factory=dict)
    co_grant_pairs: List[Tuple[str, str]] = field(default_factory=list)
    super_bundle_prob: float = 0.0

    def __repr__(self):
        antecedent_str = " AND ".join(
            f"{k}={v}" for k, v in self.antecedent.items()
        )
        return (
            f"<Pattern(id={self.id}, band={self.target_band}, "
            f"antecedent='{antecedent_str}', "
            f"core={len(self.core_ents_p)}, opt={len(self.opt_ents_p)})>"
        )


# --- Generator Classes ---

class IdentityGenerator:
    """Generates correlated identity data."""

    def __init__(self, config: Dict[str, Any], rng: Generator, faker: Faker):
        self.config = config["identity"]
        self.rng = rng
        self.faker = faker
        self.logger = logging.getLogger(self.__class__.__name__)

    def _safe_sample(
            self, choices: List[str], probs: List[float], size: int = 1
    ) -> Any:
        """Handles sampling from potentially empty lists."""
        if not choices or not probs or len(choices) != len(probs):
            self.logger.warning("Invalid choices or probs for sampling.")
            return self.rng.choice(choices, size=size) if choices else \
                (["NA"] * size)

        # Normalize probs just in case
        probs = np.array(probs)
        probs /= probs.sum()
        return self.rng.choice(choices, size=size, p=probs)

    def _get_job_title(self, job_family: str, job_level: int) -> str:
        """Generates a job title based on family and level."""
        family_map = self.config["job_titles_map"].get(job_family, {})
        if job_level in family_map:
            return family_map[job_level]

        # Fallback
        level_map = {
            1: "Analyst I", 2: "Analyst II", 3: "Senior Analyst",
            4: "Lead", 5: "Manager", 6: "Senior Manager", 7: "Director"
        }
        return f"{job_family} {level_map.get(job_level, 'Specialist')}"
    # --- BEGIN ChatGPT change: Phase 1 - identity helper methods (username, LOB, status) ---
    def _build_username(
            self,
            first_name: str,
            last_name: str,
            used_usernames: Set[str]
    ) -> str:
        """
        Build a deterministic, unique, alphanumeric username from first/last name.

        Examples:
            "Jane", "Doe" -> "janedoe"
            Collisions resolved as "janedoe1", "janedoe2", ...
        """
        base = f"{first_name}{last_name}".lower()
        # Remove any non-alphanumeric characters to satisfy "no special characters"
        filtered = "".join(ch for ch in base if ch.isalnum())
        if not filtered:
            filtered = "user"

        username = filtered
        suffix = 1
        while username in used_usernames:
            username = f"{filtered}{suffix}"
            suffix += 1

        used_usernames.add(username)
        return username

    def _assign_line_of_business(
            self,
            department: str,
            business_unit: str
    ) -> str:
        """
        Assign a line_of_business.

        Primary behavior:
            - Sample from config["lines_of_business"] distribution if present.
        Fallback behavior:
            - Derive a reasonable default from department / business_unit.
        """
        lob_cfg = self.config.get("lines_of_business", {})
        if isinstance(lob_cfg, dict) and lob_cfg:
            lob_choices = list(lob_cfg.keys())
            lob_probs = list(lob_cfg.values())
            return self._safe_sample(lob_choices, lob_probs)[0]

        # Fallback heuristics if config is missing/misconfigured
        bu = (business_unit or "").lower()
        dept = (department or "").lower()

        if "sales" in dept or "marketing" in dept:
            return "Retail"
        if "engineering" in dept or "product" in dept:
            return "Corporate"
        if "support" in dept or "cs" in dept:
            return "SMB"
        if "public" in bu:
            return "PublicSector"
        if dept in {"hr", "finance", "it", "legal"}:
            return "Internal"
        return "SharedServices"

    def _assign_status(self) -> str:
        """
        Assign an identity status ("active" or "inactive") using configuration.

        Falls back to a sane default distribution if config is missing.
        """
        status_cfg = self.config.get(
            "status_probs",
            {"active": 0.93, "inactive": 0.07}
        )
        choices = list(status_cfg.keys())
        probs = list(status_cfg.values())
        return self._safe_sample(choices, probs)[0]
    # --- END ChatGPT change: Phase 1 - identity helper methods (username, LOB, status) ---

    # --- BEGIN ChatGPT change: Phase 2 - assign managers and hierarchy ---
    def _assign_managers_and_hierarchy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Post-process identities to enforce manager ratios and hierarchy.

            Constraints (best-effort, especially for small populations):
            - ~25% of identities are managers (manager_flag == "Y")
            - >= 90% of identities have a non-null manager_id
            - A management hierarchy at least 3 levels deep when enough managers exist.
        """
        num_users = len(df)
        if num_users == 0:
            return df

        # Ensure required columns exist
        if "manager_flag" not in df.columns:
            df["manager_flag"] = "N"
        if "manager_id" not in df.columns:
            df["manager_id"] = pd.NA

        rng = self.rng

        # Target number of managers (~25% of population)
        target_mgr_ratio = 0.25
        target_managers = max(1, int(round(target_mgr_ratio * num_users)))

        # Rank potential managers: FTE > others, higher job_level, longer tenure
        tmp = df.copy()
        tmp["job_level_rank"] = tmp.get("job_level", 1)
        tmp["tenure_rank"] = tmp.get("tenure_years", 0.0)
        tmp["emp_rank"] = (tmp.get("employment_type", "FTE") == "FTE").astype(int)
        # Add a small random jitter for stable but varied ordering
        tmp["rand"] = rng.random(num_users)

        tmp = tmp.sort_values(
            by=["emp_rank", "job_level_rank", "tenure_rank", "rand"],
            ascending=[False, False, False, False],
        )

        manager_indices = list(tmp.index[:target_managers])

        # Set manager_flag according to selection
        df["manager_flag"] = "N"
        df.loc[manager_indices, "manager_flag"] = "Y"

        mgr_df = df[df["manager_flag"] == "Y"].copy()
        mgr_indices = list(mgr_df.index)
        mgr_count = len(mgr_indices)

        # Initialize manager_id as null for everyone
        df["manager_id"] = pd.NA

        # Allowed number of identities without a manager (max 10%)
        allowed_no_manager = int(0.10 * num_users)

        # Degenerate case: 1 manager only
        if mgr_count <= 1:
            root_idx = mgr_indices[0]
            root_id = df.at[root_idx, "user_id"]
            for idx in df.index:
                if idx == root_idx:
                    continue
                df.at[idx, "manager_id"] = root_id
            return df

        # Shuffle managers deterministically using numpy permutation
        mgr_indices_arr = np.array(mgr_indices)
        perm = rng.permutation(len(mgr_indices_arr))
        mgr_indices = [int(mgr_indices_arr[i]) for i in perm]

        # Choose root (level 1) managers
        root_count = max(1, int(round(0.05 * mgr_count)))
        if allowed_no_manager > 0:
            root_count = min(root_count, allowed_no_manager)
        else:
            root_count = 1

        root_indices = mgr_indices[:root_count]
        remaining_mgrs = mgr_indices[root_count:]

        # Fallback if no remaining managers (all roots)
        if not remaining_mgrs:
            root_id = df.at[root_indices[0], "user_id"]
            for idx in mgr_indices:
                if idx == root_indices[0]:
                    continue
                df.at[idx, "manager_id"] = root_id
            # Non-managers also report to root
            non_mgr_indices = df.index[df["manager_flag"] == "N"].tolist()
            for idx in non_mgr_indices:
                df.at[idx, "manager_id"] = root_id
            return df

        # Split remaining managers into level 2 and level 3
        half = len(remaining_mgrs) // 2
        if half == 0:
            level2_indices = remaining_mgrs
            level3_indices = []
        else:
            level2_indices = remaining_mgrs[:half]
            level3_indices = remaining_mgrs[half:]

        root_ids = [df.at[idx, "user_id"] for idx in root_indices]
        level2_ids = [df.at[idx, "user_id"] for idx in level2_indices] or root_ids

        # Assign level 2 managers to roots
        for idx in level2_indices:
            df.at[idx, "manager_id"] = rng.choice(root_ids)

        # Assign level 3 managers to level 2 (or roots if no level 2)
        if level3_indices:
            parent_pool = level2_ids if level2_ids else root_ids
            for idx in level3_indices:
                df.at[idx, "manager_id"] = rng.choice(parent_pool)

        # Now assign managers to non-managers from full manager pool
        manager_pool_ids = [df.at[idx, "user_id"] for idx in mgr_indices]
        non_mgr_indices = df.index[df["manager_flag"] == "N"].tolist()
        for idx in non_mgr_indices:
            df.at[idx, "manager_id"] = rng.choice(manager_pool_ids)

        # At this point, only the roots should have null manager_id.
        null_indices = df.index[df["manager_id"].isna()].tolist()
        if len(null_indices) > allowed_no_manager and allowed_no_manager >= 0:
            # Keep only allowed_no_manager roots without a manager;
            # reassign the rest to one of the kept roots.
            np_null_indices = np.array(null_indices)
            perm = rng.permutation(len(np_null_indices))
            shuffled_null = [int(np_null_indices[i]) for i in perm]

            keep_roots = shuffled_null[:max(1, allowed_no_manager)]
            reassign = shuffled_null[max(1, allowed_no_manager):]

            if keep_roots:
                root_ids_final = [df.at[idx, "user_id"] for idx in keep_roots]
            else:
                root_ids_final = manager_pool_ids

            for idx in reassign:
                df.at[idx, "manager_id"] = rng.choice(root_ids_final)

        return df
# --- END ChatGPT change: Phase 2 - assign managers and hierarchy ---
    def generate_identities(self, num_users: int) -> pd.DataFrame:
        """Generates the main identity DataFrame."""
        self.logger.info(f"Generating {num_users} identities...")

        data = []
        used_usernames: Set[str] = set()
        dept_choices = list(self.config["departments"].keys())
        dept_probs = list(self.config["departments"].values())

        for i in range(num_users):
            user_id = f"U{i + 1:07d}"

            first_name = self.faker.first_name()
            last_name = self.faker.last_name()
            email = (
                f"{first_name.lower()}.{last_name.lower()}"
                f"@{self.faker.domain_name()}"
            )

            user_name = self._build_username(
                first_name,
                last_name,
                used_usernames
            )

            department = self._safe_sample(dept_choices, dept_probs)[0]

            bu_choices = self.config["business_units"]
            business_unit = self.rng.choice(bu_choices)

            line_of_business = self._assign_line_of_business(
                department,
                business_unit
            )

            jf_choices = self.config["job_families_by_dept"].get(
                department, ["General"]
            )
            job_family = self.rng.choice(jf_choices)

            jl_values = self.config["job_levels"]["values"]
            jl_probs = self.config["job_levels"]["probs"]
            job_level = self._safe_sample(jl_values, jl_probs)[0]

            title = self._get_job_title(job_family, job_level)

            country_choices = list(self.config["countries"].keys())
            country_probs = list(self.config["countries"].values())
            location_country = self._safe_sample(
                country_choices, country_probs
            )[0]

            site_choices = list(
                self.config["sites_by_country"]
                .get(location_country, {})
                .keys()
            )
            site_probs = list(
                self.config["sites_by_country"]
                .get(location_country, {})
                .values()
            )
            location_site = self._safe_sample(site_choices, site_probs)[0]

            emp_choices = list(self.config["employment_types"].keys())
            emp_probs = list(self.config["employment_types"].values())
            employment_type = self._safe_sample(emp_choices, emp_probs)[0]

            cost_center = (
                f"CC_{department[:3].upper()}_{self.rng.integers(100, 999)}"
            )

            # Initial manager_flag (will be recalibrated in Phase 2)
            manager_flag = (
                "Y" if self.rng.random() < self.config["manager_prob"] else "N"
            )

            # Use Beta distribution for right-skewed tenure
            tenure_params = self.config["tenure_beta_params"]
            tenure_years = self.rng.beta(
                tenure_params["a"], tenure_params["b"]
            ) * tenure_params["scale"]

            status = self._assign_status()

            data.append({
                "user_id": user_id,
                "user_name": user_name,
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "department": department,
                "business_unit": business_unit,
                "line_of_business": line_of_business,
                "job_family": job_family,
                "job_level": job_level,
                "title": title,
                "location_country": location_country,
                "location_site": location_site,
                "employment_type": employment_type,
                "cost_center": cost_center,
                "manager_flag": manager_flag,
                "manager_id": None,  # Will be populated by _assign_managers_and_hierarchy
                "status": status,
                "tenure_years": round(tenure_years, 2),
            })

        df = pd.DataFrame(data)
        self.logger.info("Identity generation complete.")

        # --- BEGIN ChatGPT change: Phase 2 - apply manager hierarchy ---
        df = self._assign_managers_and_hierarchy(df)
        self.logger.info("Manager hierarchy assignment complete.")
        # --- END ChatGPT change: Phase 2 - apply manager hierarchy ---

        return df

class EntitlementGenerator:
    """Generates the entitlement catalog and functional bundles."""

    def __init__(self, config: Dict[str, Any], rng: Generator, faker: Faker):
        self.e_config = config["entitlement"]
        self.g_config = config["global"]
        self.rng = rng
        self.faker = faker
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ent_counter = 1

    def _safe_sample(
            self, choices: List[str], probs: List[float], size: int = 1
    ) -> Any:
        """Handles sampling from potentially empty lists."""
        if not choices or not probs or len(choices) != len(probs):
            self.logger.warning("Invalid choices or probs for sampling.")
            return self.rng.choice(choices, size=size) if choices else \
                (["NA"] * size)

        # Normalize probs just in case
        probs = np.array(probs)
        probs /= probs.sum()
        return self.rng.choice(choices, size=size, p=probs)

    def _generate_ent_name(self, app_name: str) -> Tuple[str, str]:
        """Generate a realistic entitlement name and type."""
        ent_type = self.rng.choice(self.e_config["ent_types"])
        name = "NA"

        if app_name == "Epic":
            template = self.rng.choice(self.e_config["epic_templates"])
            name = f"tpl_{template}_{self.faker.word()}"
            ent_type = "template-access"
        elif app_name == "GitHub":
            repo = self.faker.word().lower()
            perm = self.rng.choice(["read", "write", "admin", "triage"])
            name = f"repo:{repo}:{perm}"
            ent_type = "repo-permission"
        elif app_name == "AWS":
            role = self.faker.word().lower().capitalize()
            name = f"arn:aws:iam::{self.faker.msisdn()[:12]}:role/{role}Access"
            ent_type = "role"
        elif app_name == "Salesforce":
            obj = self.rng.choice(["Account", "Opportunity", "Lead", "Contact"])
            perm = self.rng.choice(["Read", "Edit", "Create", "Delete"])
            name = f"Profile: {self.faker.word()} ({obj} {perm})"
            ent_type = "role"
        elif app_name == "Snowflake":
            db = self.faker.word().upper()
            role = self.rng.choice(["ANALYST", "DEVELOPER", "READER", "ADMIN"])
            name = f"role::{db}_{role}"
            ent_type = "role"
        elif app_name == "GDrive":
            folder = self.faker.word().capitalize()
            name = f"SharedFolder: {folder}"
            ent_type = "group"
        elif app_name == "Slack":
            channel = self.faker.word().lower()
            name = f"channel:{channel}"
            ent_type = "channel-access"
        else:
            name = f"{self.faker.word().capitalize()} {ent_type.capitalize()}"

        # Add near-duplicates
        if self.rng.random() < 0.05:
            name = f"{name} (Legacy)"
        elif self.rng.random() < 0.05:
            name = f"{name}_v2"

        return name, ent_type

    def _generate_description(self, app_name: str, ent_type: str, ent_name: str) -> str:
        """Generates a meaningful description under 100 characters."""

        # Try a few description patterns
        if "role" in ent_type:
            desc = f"Grants {ent_name} {ent_type.replace('-', ' ')} in {app_name}."
        elif "access" in ent_type:
            desc = f"Provides {ent_type.replace('-', ' ')} to {ent_name} in {app_name}."
        elif "permission" in ent_type:
            desc = f"Confers {ent_name} permission for {app_name}."
        else:
            desc = f"General {ent_type} for {app_name} application."

        # Truncate to max 99 chars
        if len(desc) > 99:
            desc = desc[:96] + "..."
        return desc

    def generate_entitlements(self) -> pd.DataFrame:
        """Generates the main entitlement catalog DataFrame."""
        num_ents = self.g_config["num_entitlements"]
        num_apps = self.g_config["num_apps"]
        app_names = self.rng.choice(
            self.e_config["app_names"], num_apps, replace=False
        )
        # Ensure Epic is included
        if "Epic" not in app_names:
            app_names[self.rng.integers(0, num_apps)] = "Epic"

        self.logger.info(
            f"Generating {num_ents} entitlements across "
            f"{num_apps} apps: {app_names}"
        )

        # Assign app sizes
        sizes = (
                ["big"] * self.e_config["app_size_dist"]["big"] +
                ["medium"] * self.e_config["app_size_dist"]["medium"] +
                ["small"] * self.e_config["app_size_dist"]["small"]
        )
        self.rng.shuffle(sizes)
        app_size_map = dict(zip(app_names, sizes[:num_apps]))

        # Distribute entitlement counts
        ent_counts = {}
        total_assigned = 0
        for app, size in app_size_map.items():
            min_ents, max_ents = self.e_config["app_size_ents"][size]
            count = self.rng.integers(min_ents, max_ents + 1)
            ent_counts[app] = count
            total_assigned += count

        # Adjust counts to match target num_entitlements
        delta = num_ents - total_assigned
        for _ in range(abs(delta)):
            app_to_adjust = self.rng.choice(app_names)
            if delta > 0:
                ent_counts[app_to_adjust] += 1
            elif ent_counts[app_to_adjust] > 20:  # Don't go too low
                ent_counts[app_to_adjust] -= 1

        self.logger.debug(f"Entitlement counts per app: {ent_counts}")

        data = []
        crit_choices = list(self.e_config["criticality_probs"].keys())
        crit_probs = list(self.e_config["criticality_probs"].values())

        for app_name, count in ent_counts.items():
            for _ in range(count):
                ent_id = f"E{self.ent_counter:07d}"
                self.ent_counter += 1

                name, ent_type = self._generate_ent_name(app_name)
                criticality = self._safe_sample(
                    crit_choices, crit_probs
                )[0]

                # Generate meaningful description
                description = self._generate_description(app_name, ent_type, name)

                data.append({
                    "entitlement_id": ent_id,
                    "app_id": f"APP{app_names.tolist().index(app_name) + 1:02d}",
                    "app_name": app_name,
                    "entitlement_name": name,
                    "entitlement_type": ent_type,
                    "criticality": criticality,
                    "scope": self.faker.word() if self.rng.random() < 0.3 else "",
                    "description": description,  # Use new description
                })

        df = pd.DataFrame(data)
        self.logger.info(f"Generated {len(df)} total entitlements.")
        return df

    def generate_bundles(
            self, entitlements_df: pd.DataFrame
    ) -> List[List[str]]:
        """Generates cross-app functional bundles."""
        num_bundles = self.e_config["num_bundles"]
        min_size, max_size = self.e_config["bundle_size_range"]
        app_names = entitlements_df["app_name"].unique()
        ents_by_app = {
            app: list(df["entitlement_id"])
            for app, df in entitlements_df.groupby("app_name")
        }

        bundles = []
        for i in range(num_bundles):
            bundle = set()
            num_apps_in_bundle = self.rng.integers(2, 5)  # 2-4 apps
            apps_for_bundle = self.rng.choice(
                app_names, num_apps_in_bundle, replace=False
            )

            bundle_size = self.rng.integers(min_size, max_size + 1)

            # Distribute size across apps
            ents_per_app = self.rng.multinomial(
                bundle_size, [1 / num_apps_in_bundle] * num_apps_in_bundle
            )

            for app, count in zip(apps_for_bundle, ents_per_app):
                if count > 0 and ents_by_app[app]:
                    chosen_ents = self.rng.choice(
                        ents_by_app[app],
                        min(count, len(ents_by_app[app])),
                        replace=False
                    )
                    bundle.update(chosen_ents)

            if bundle:
                bundles.append(list(bundle))

        self.logger.info(f"Generated {len(bundles)} functional bundles.")
        return bundles


class PatternGenerator:
    """Generates latent patterns linking identities to bundles."""

    def __init__(self, config: Dict[str, Any], rng: Generator):
        self.p_config = config["pattern"]
        self.i_config = config["identity"]
        self.rng = rng
        self.logger = logging.getLogger(self.__class__.__name__)

    def _create_antecedent(
            self, identities_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create a random antecedent dict."""
        num_features = self.rng.integers(
            self.p_config["antecedent_size_range"][0],
            self.p_config["antecedent_size_range"][1] + 1,
        )

        features = ["department", "job_family", "job_level",
                    "location_country", "location_site", "employment_type"]

        chosen_features = self.rng.choice(
            features, num_features, replace=False
        )

        antecedent = {}
        for feature in chosen_features:
            if feature in identities_df.columns:
                possible_values = identities_df[feature].unique()
                if possible_values.size > 0:
                    antecedent[feature] = self.rng.choice(possible_values)
        return antecedent

    def generate_patterns(
            self, identities_df: pd.DataFrame, bundles: List[List[str]]
    ) -> List[Pattern]:
        """Generates the list of Pattern objects."""
        num_patterns = self.p_config["num_patterns"]
        if not bundles:
            self.logger.error("No bundles provided, cannot generate patterns.")
            return []

        self.logger.info(f"Generating {num_patterns} latent patterns...")

        band_choices = list(self.p_config["confidence_bands"].keys())
        band_probs = list(self.p_config["confidence_bands"].values())

        patterns = []
        for i in range(num_patterns):
            antecedent = self._create_antecedent(identities_df)
            # --- THIS IS THE FIX ---
            # Was: bundle = list(self.rng.choice(bundles))
            # Replaced self.rng.choice (NumPy) with random.choice (Python stdlib)
            # to handle lists of lists with different lengths ("ragged array").
            bundle = list(random.choice(bundles))  # Pick one bundle  <--- THIS LINE
            # --- END FIX ---
            target_band = self.rng.choice(band_choices, p=band_probs)

            if not bundle:
                self.logger.warning(f"Skipping pattern {i} due to empty bundle.")
                continue

            # Split into core/optional
            core_pct = self.rng.uniform(
                self.p_config["core_ent_pct_range"][0],
                self.p_config["core_ent_pct_range"][1]
            )
            n_core = max(1, int(len(bundle) * core_pct))
            self.rng.shuffle(bundle)
            core_ents = bundle[:n_core]
            opt_ents = bundle[n_core:]

            # Sample base probabilities
            band_p = self.p_config["band_probs"][target_band]
            core_p_min, core_p_max = band_p["core"]
            opt_p_min, opt_p_max = band_p["opt"]

            core_ents_p = {
                ent: self.rng.uniform(core_p_min, core_p_max)
                for ent in core_ents
            }
            opt_ents_p = {
                ent: self.rng.uniform(opt_p_min, opt_p_max)
                for ent in opt_ents
            }

            # Create co-grant pairs (from core entitlements)
            co_grant_pairs = []
            if len(core_ents) >= 2:
                n_pairs = self.p_config["co_grant_pairs_per_bundle"]
                for _ in range(n_pairs):
                    pair = self.rng.choice(core_ents, 2, replace=False)
                    co_grant_pairs.append(tuple(pair))

            super_bundle_prob = self.rng.uniform(
                self.p_config["super_bundle_prob_range"][0],
                self.p_config["super_bundle_prob_range"][1]
            )

            patterns.append(Pattern(
                id=f"P{i + 1:04d}",
                antecedent=antecedent,
                target_band=target_band,
                core_ents_p=core_ents_p,
                opt_ents_p=opt_ents_p,
                co_grant_pairs=co_grant_pairs,
                super_bundle_prob=super_bundle_prob,
            ))

        self.logger.info(f"Generated {len(patterns)} patterns.")
        return patterns


class AssignmentGenerator:
    """Generates assignments based on patterns and noise."""

    def __init__(self, config: Dict[str, Any], rng: Generator):
        self.a_config = config
        self.n_config = config["noise"]
        self.p_config = config["pattern"]
        self.rng = rng
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_matching_patterns(
            self, user: pd.Series, patterns: List[Pattern]
    ) -> List[Pattern]:
        """Find all patterns that match a user's identity."""
        matches = []
        for p in patterns:
            match = True
            for key, value in p.antecedent.items():
                if user[key] != value:
                    match = False
                    break
            if match:
                matches.append(p)
        return matches

    def _apply_pattern_to_user(
            self,
            pattern: Pattern,
            prob_reduction: float = 0.0
    ) -> Set[str]:
        """
        Apply a single pattern's logic to a user and return granted
        entitlements.
        """
        granted_ents = set()

        # Check for super-bundle grant first
        if self.rng.random() < pattern.super_bundle_prob:
            granted_ents.update(pattern.core_ents_p.keys())
            granted_ents.update(pattern.opt_ents_p.keys())
            return granted_ents  # All granted, skip individual logic

        # Base probabilities, adjusted for collision
        base_core_p = {
            ent: max(0.01, p - prob_reduction)
            for ent, p in pattern.core_ents_p.items()
        }
        base_opt_p = {
            ent: max(0.01, p - prob_reduction)
            for ent, p in pattern.opt_ents_p.items()
        }

        # Co-grant boosts
        boosts = defaultdict(float)
        co_grant_boost = self.p_config["co_grant_boost"]

        # Grant core entitlements
        granted_core = set()
        for ent, p in base_core_p.items():
            if self.rng.random() < p:
                granted_ents.add(ent)
                granted_core.add(ent)
                # Check if this grant boosts others
                for p1, p2 in pattern.co_grant_pairs:
                    if p1 == ent:
                        boosts[p2] += co_grant_boost
                    elif p2 == ent:
                        boosts[p1] += co_grant_boost

        # Re-check core ents that didn't get granted, applying boosts
        for ent, p in base_core_p.items():
            if ent not in granted_ents:
                boosted_p = np.clip(p + boosts[ent], 0, 0.98)
                if self.rng.random() < boosted_p:
                    granted_ents.add(ent)

        # Grant optional entitlements (boosts don't apply)
        for ent, p in base_opt_p.items():
            if self.rng.random() < p:
                granted_ents.add(ent)

        return granted_ents

    def _apply_noise(
            self,
            user_ids: List[str],
            entitlements_df: pd.DataFrame,
            assignments: Set[Tuple[str, str]]
    ) -> Set[Tuple[str, str]]:
        """Apply background utility and outlier noise."""
        self.logger.debug("Applying background and outlier noise...")

        # 1. Background utility grants
        all_ents = entitlements_df["entitlement_id"].values
        low_med_ents = entitlements_df[
            entitlements_df["criticality"] != "High"
            ]["entitlement_id"].values

        if len(low_med_ents) == 0:
            low_med_ents = all_ents

        n_bg_ents = int(len(low_med_ents) * self.n_config["background_ent_pct"])
        bg_ents_to_grant = self.rng.choice(
            low_med_ents, n_bg_ents, replace=False
        )

        n_bg_users = int(len(user_ids) * self.n_config["background_user_pct"])

        for ent in bg_ents_to_grant:
            users_to_get_ent = self.rng.choice(
                user_ids, n_bg_users, replace=False
            )
            for user in users_to_get_ent:
                assignments.add((user, ent))

        # 2. Outlier grants (high criticality)
        high_crit_ents = entitlements_df[
            entitlements_df["criticality"] == "High"
            ]["entitlement_id"].values

        if len(high_crit_ents) == 0:
            high_crit_ents = all_ents

        n_outlier_users = int(len(user_ids) * self.n_config["outlier_user_pct"])
        outlier_users = self.rng.choice(
            user_ids, n_outlier_users, replace=False
        )

        min_ents, max_ents = self.n_config["outlier_ent_count_range"]

        for user in outlier_users:
            n_to_grant = self.rng.integers(min_ents, max_ents + 1)
            if len(high_crit_ents) > 0:
                ents_to_grant = self.rng.choice(
                    high_crit_ents, n_to_grant,
                    replace=len(high_crit_ents) < n_to_grant
                )
                for ent in ents_to_grant:
                    assignments.add((user, ent))

        return assignments

    def generate_assignments_pass(
            self,
            identities_df: pd.DataFrame,
            entitlements_df: pd.DataFrame,
            patterns: List[Pattern]
    ) -> pd.DataFrame:
        """Run a single pass of assignment generation."""
        self.logger.info("Generating assignments...")

        all_assignments: Set[Tuple[str, str]] = set()

        for _, user in identities_df.iterrows():
            user_id = user["user_id"]
            matching_patterns = self._get_matching_patterns(user, patterns)

            if not matching_patterns:
                continue

            # Sort to make collision handling deterministic (given a seed)
            matching_patterns.sort(key=lambda p: p.id)

            reduction = 0.0
            for pattern in matching_patterns:
                granted = self._apply_pattern_to_user(
                    pattern, prob_reduction=reduction
                )
                for ent in granted:
                    all_assignments.add((user_id, ent))

                # Apply reduction for subsequent colliding patterns
                reduction += self.n_config["collision_prob_reduction"]

        # Apply global noise
        all_assignments = self._apply_noise(
            list(identities_df["user_id"]),
            entitlements_df,
            all_assignments
        )

        assignments_df = pd.DataFrame(
            list(all_assignments), columns=["user_id", "entitlement_id"]
        )
        self.logger.info(f"Generated {len(assignments_df)} assignment edges.")
        return assignments_df.drop_duplicates()


class TransactionConverter:
    """Converts identity and assignment data into transactions for mining."""

    FEATURE_COLS = [
        "department", "business_unit", "job_family", "job_level",
        "location_country", "location_site", "employment_type", "manager_flag"
    ]
    TOKEN_PREFIX_MAP = {
        "department": "dept",
        "business_unit": "bu",
        "job_family": "jobfam",
        "job_level": "level",
        "location_country": "country",
        "location_site": "site",
        "employment_type": "etype",
        "manager_flag": "mgr"
    }

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.identity_tokens_cache: Dict[str, Set[str]] = {}

    def _tokenize_identity_row(self, user_row: pd.Series) -> Set[str]:
        """Convert a user's identity attributes into feature tokens."""
        tokens = set()
        for col in self.FEATURE_COLS:
            if col in user_row:
                prefix = self.TOKEN_PREFIX_MAP.get(col, col)
                tokens.add(f"{prefix}={user_row[col]}")
        return tokens

    def get_identity_tokens(
            self, identities_df: pd.DataFrame
    ) -> Dict[str, Set[str]]:
        """Pre-compute and cache feature tokens for all users."""
        if not self.identity_tokens_cache:
            self.logger.debug("Tokenizing identities for transactions...")
            self.identity_tokens_cache = {
                row["user_id"]: self._tokenize_identity_row(row)
                for _, row in identities_df.iterrows()
            }
        return self.identity_tokens_cache

    def tokenize_antecedent(
            self, antecedent: Dict[str, Any]
    ) -> Set[str]:
        """Convert a pattern antecedent dict into a set of feature tokens."""
        tokens = set()
        for col, value in antecedent.items():
            if col in self.TOKEN_PREFIX_MAP:
                prefix = self.TOKEN_PREFIX_MAP[col]
                tokens.add(f"{prefix}={value}")
        return tokens

    def create_transactions(
            self,
            identities_df: pd.DataFrame,
            assignments_df: pd.DataFrame
    ) -> List[List[str]]:
        """
        Create a list of transactions, where each transaction is a
        list of a user's entitlements and their identity feature tokens.
        """
        self.logger.info("Creating transactions for calibration...")

        identity_tokens = self.get_identity_tokens(identities_df)

        # Group entitlements by user
        ents_by_user = defaultdict(set)
        for _, row in assignments_df.iterrows():
            ents_by_user[row["user_id"]].add(row["entitlement_id"])

        transactions = []
        for user_id, id_tokens in identity_tokens.items():
            user_ents = ents_by_user.get(user_id, set())  # Use .get for users with no ents
            # Combine identity tokens and entitlement IDs
            transactions.append(list(id_tokens) + list(user_ents))

        self.logger.info(f"Created {len(transactions)} transactions.")
        return transactions


class CalibrationEngine:
    """
    Runs the feedback loop to adjust pattern probabilities to meet
    target confidence bucket distributions.
    """

    def __init__(self, config: Dict[str, Any], rng: Generator):
        self.c_config = config["calibration"]
        self.g_config = config
        self.rng = rng
        self.logger = logging.getLogger(self.__class__.__name__)

        self.assign_gen = AssignmentGenerator(config, rng)
        self.trans_conv = TransactionConverter()

    def _get_bucket(self, confidence: float) -> str:
        """Classify a confidence score into a bucket."""
        if confidence >= self.c_config["bucket_thresholds"]["High"]:
            return "High"
        if confidence >= self.c_config["bucket_thresholds"]["Mid"]:
            return "Mid"
        return "Low"

    def _measure_confidences(
            self,
            transactions: List[List[str]],
            patterns: List[Pattern]
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Measure the empirical confidence of all (antecedent => core_ent)
        rules defined by the patterns.
        """
        self.logger.debug("Measuring empirical confidences...")
        transactions_as_sets = [set(t) for t in transactions]
        antecedent_token_map = {
            p.id: self.trans_conv.tokenize_antecedent(p.antecedent)
            for p in patterns
        }

        observed_rules = []

        for pattern in patterns:
            antecedent_set = antecedent_token_map[pattern.id]
            if not antecedent_set:
                continue  # Skip patterns with no valid tokens

            # Find all transactions matching the antecedent
            matching_transactions = [
                t for t in transactions_as_sets if antecedent_set.issubset(t)
            ]
            N_antecedent = len(matching_transactions)

            if N_antecedent == 0:
                continue  # No support for this antecedent

            # For each core entitlement, calculate confidence
            for core_ent in pattern.core_ents_p.keys():
                N_both = sum(1 for t in matching_transactions if core_ent in t)
                obs_conf = N_both / N_antecedent

                observed_rules.append({
                    "pattern_id": pattern.id,
                    "ent_id": core_ent,
                    "obs_conf": obs_conf,
                    "target_band": pattern.target_band,
                    "bucket": self._get_bucket(obs_conf),
                })

        if not observed_rules:
            self.logger.warning("No observed rules found. Calibration will stop.")
            return pd.DataFrame(), {"High": 0, "Mid": 0, "Low": 0}

        rules_df = pd.DataFrame(observed_rules)
        bucket_props = (
            rules_df["bucket"].value_counts(normalize=True)
            .reindex(["High", "Mid", "Low"], fill_value=0.0)
            .to_dict()
        )

        return rules_df, bucket_props

    def _adjust_probabilities(
            self,
            patterns: List[Pattern],
            observed_rules_df: pd.DataFrame
    ) -> None:
        """
        Nudge the base probabilities (p) stored in the Pattern objects
        based on the error (target_midpoint - observed_confidence).
        """
        self.logger.debug("Adjusting pattern probabilities...")
        patterns_by_id = {p.id: p for p in patterns}
        midpoints = self.c_config["target_midpoints"]
        alpha = self.c_config["learning_rate_alpha"]

        for row in observed_rules_df.itertuples():
            pattern = patterns_by_id.get(row.pattern_id)
            if not pattern:
                continue

            ent_id = row.ent_id
            if ent_id not in pattern.core_ents_p:
                continue

            obs_conf = row.obs_conf
            target_mid = midpoints[row.target_band]

            error = target_mid - obs_conf
            current_p = pattern.core_ents_p[ent_id]

            # Nudge: p_new = p_old + alpha * error
            new_p = np.clip(current_p + alpha * error, 0.02, 0.98)

            pattern.core_ents_p[ent_id] = new_p

    def run_calibration_loop(
            self,
            identities_df: pd.DataFrame,
            entitlements_df: pd.DataFrame,
            patterns: List[Pattern]
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Execute the full calibration loop.

        Returns the final assignments_df and the final bucket proportions.
        """
        self.logger.info("Starting calibration loop...")

        assignments_df = pd.DataFrame()  # Init
        proportions = {}

        for i in range(self.c_config["max_iterations"]):
            self.logger.info(f"--- Calibration Iteration {i + 1} ---")

            # 1. Generate assignments
            assignments_df = self.assign_gen.generate_assignments_pass(
                identities_df, entitlements_df, patterns
            )

            # 2. Create transactions
            # Clear cache to get fresh tokens (though identities don't change)
            self.trans_conv.identity_tokens_cache.clear()
            transactions = self.trans_conv.create_transactions(
                identities_df, assignments_df
            )

            if not transactions:
                self.logger.error("No transactions generated. Stopping loop.")
                break

            # 3. Measure confidences and bucket proportions
            rules_df, proportions = self._measure_confidences(
                transactions, patterns
            )

            if rules_df.empty:
                self.logger.error("No rules measured. Stopping loop.")
                break

            self.logger.info(
                f"Observed bucket proportions: "
                f"High={proportions['High']:.2%}, "
                f"Mid={proportions['Mid']:.2%}, "
                f"Low={proportions['Low']:.2%}"
            )

            # 4. Check for convergence
            targets = self.c_config["target_buckets"]
            tolerance = self.c_config["tolerance"]
            converged = True
            for band in targets.keys():
                if abs(proportions[band] - targets[band]) > tolerance:
                    converged = False
                    break

            if converged:
                self.logger.info(
                    f"Convergence achieved within {tolerance:.0%} tolerance."
                )
                break
            else:
                self.logger.warning("Targets not met. Adjusting probabilities.")

            # 5. Adjust probabilities (if not last iteration)
            if i < self.c_config["max_iterations"] - 1:
                self._adjust_probabilities(patterns, rules_df)
            else:
                self.logger.warning(
                    "Max iterations reached. Using last generated assignments."
                )

        return assignments_df, proportions


# --- Built-in Tests (New) ---

def _run_tests() -> None:
    """
    Run a minimal, self-contained test of the generation pipeline
    to check for errors and reproducibility.
    """
    logger = logging.getLogger("_run_tests")
    logger.info("--- Running Built-in Tests ---")

    # 1. Define a minimal test configuration
    test_seed = 123
    test_config = json.loads(json.dumps(DEFAULT_CONFIG))  # Deep copy
    test_config["global"]["seed"] = test_seed
    test_config["global"]["num_users"] = 100
    test_config["global"]["num_apps"] = 3
    test_config["global"]["num_entitlements"] = 50
    test_config["calibration"]["max_iterations"] = 1  # Fast
    test_config["pattern"]["num_patterns"] = 10

    def run_pipeline(seed: int, config: Dict[str, Any]) -> pd.DataFrame:
        """Helper to run the full pipeline once."""
        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        Faker.seed(seed)
        faker_instance = Faker()

        id_gen = IdentityGenerator(config, rng, faker_instance)
        ids = id_gen.generate_identities(config["global"]["num_users"])

        ent_gen = EntitlementGenerator(config, rng, faker_instance)
        ents = ent_gen.generate_entitlements()
        bundles = ent_gen.generate_bundles(ents)

        pat_gen = PatternGenerator(config, rng)
        patterns = pat_gen.generate_patterns(ids, bundles)

        cal_engine = CalibrationEngine(config, rng)
        assignments_df, _ = cal_engine.run_calibration_loop(
            ids, ents, patterns
        )
        return assignments_df

    # 2. Run pipeline once
    logger.info("Running first test pass...")
    assignments_1 = run_pipeline(test_seed, test_config)

    assert not assignments_1.empty, "Test failed: First pass produced no assignments"
    assert assignments_1.duplicated(subset=["user_id", "entitlement_id"]).sum() == 0, \
        "Test failed: Found duplicate assignments"
    len1 = len(assignments_1)

    # 3. Run pipeline a second time with the same seed
    logger.info("Running second test pass for reproducibility...")
    assignments_2 = run_pipeline(test_seed, test_config)
    len2 = len(assignments_2)

    # 4. Check for reproducibility
    assert len1 == len2, \
        f"Test failed: Reproducibility check. {len1} != {len2}"

    # Check DataFrame equality (more rigorous)
    pd.testing.assert_frame_equal(
        assignments_1.sort_values(['user_id', 'entitlement_id']).reset_index(drop=True),
        assignments_2.sort_values(['user_id', 'entitlement_id']).reset_index(drop=True)
    )

    logger.info("--- All Tests Passed ---")


# ============================================================================
# Data Writer
# ============================================================================

class DataWriter:
    """Writes generated data to CSV files."""

    # Note: This class is deprecated in favor of pandas `to_csv` functions
    # kept here for compatibility with `test_iga_generator.py` if it's used.

    # --- BEGIN ChatGPT change: Phase 1 - extend identity CSV columns ---
    @staticmethod
    def write_identities(identities: List[Identity], output_path: Path):
        """Write identities to CSV."""
        with open(output_path, 'w', newline='') as f:
            # Use csv.QUOTE_ALL
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow([
                'user_id', 'user_name', 'first_name', 'last_name', 'email',
                'department', 'business_unit', 'line_of_business', 'job_family',
                'job_level', 'title', 'location_country', 'location_site',
                'employment_type', 'cost_center', 'manager_flag', 'manager_id',
                'status', 'tenure_years'
            ])

            for identity in identities:
                writer.writerow([
                    identity.user_id, identity.user_name,
                    identity.first_name, identity.last_name,
                    identity.email, identity.department,
                    identity.business_unit, identity.line_of_business,
                    identity.job_family, identity.job_level, identity.title,
                    identity.location_country, identity.location_site,
                    identity.employment_type, identity.cost_center,
                    identity.manager_flag, identity.manager_id,
                    identity.status, identity.tenure_years
                ])

    # --- END ChatGPT change: Phase 1 - extend identity CSV columns ---
    @staticmethod
    def write_entitlements(entitlements: List[Entitlement], output_path: Path):
        """Write entitlements to CSV."""
        with open(output_path, 'w', newline='') as f:
            # Use csv.QUOTE_ALL
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow([
                'entitlement_id', 'app_id', 'app_name', 'entitlement_name',
                'entitlement_type', 'criticality', 'scope', 'description'
            ])

            for ent in entitlements:
                writer.writerow([
                    ent.entitlement_id, ent.app_id, ent.app_name,
                    ent.entitlement_name, ent.entitlement_type,
                    ent.criticality, ent.scope, ent.description
                ])

    @staticmethod
    def write_assignments(assignments: List[Tuple[str, str]], output_path: Path):
        """Write assignments to CSV."""
        with open(output_path, 'w', newline='') as f:
            # Use csv.QUOTE_ALL
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(['user_id', 'entitlement_id'])

            for user_id, ent_id in assignments:
                writer.writerow([user_id, ent_id])


# --- Main Execution ---

def setup_logging(level=logging.INFO):
    """Configure logging."""
    log_format = (
        "%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s"
    )
    logging.basicConfig(level=level, format=log_format)
    # Suppress verbose logs from faker
    logging.getLogger("faker").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Synthetic IGA Data Generator for Association Rule Mining.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a JSON or YAML config file (overrides defaults)."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default="./out",
        help="Directory to save output CSV files."
    )
    parser.add_argument(
        "--num-users",
        type=int,
        help="Number of identities to generate."
    )
    parser.add_argument(
        "--num-entitlements",
        type=int,
        help="Total number of entitlements to generate."
    )
    parser.add_argument(
        "--num-apps",
        type=int,
        help="Number of applications to generate."
    )
    # Note: avg-ents-per-user is an *outcome* of the pattern simulation,
    # not a direct input, so it's omitted as a flag.

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for deterministic generation."
    )
    parser.add_argument(
        "--no-clobber",
        action="store_true",
        help="Do not overwrite existing files in out-dir."
    )
    parser.add_argument(
        "--emit-default-config",
        action="store_true",
        help="Print the default config as JSON and exit."
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run built-in unit tests and exit."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG level logging."
    )

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Load config from default, file, and CLI args."""
    config = DEFAULT_CONFIG.copy()

    if args.config:
        try:
            with open(args.config, 'r') as f:
                if args.config.suffix.lower() in ('.yaml', '.yml'):
                    # Add YAML support if library is available
                    try:
                        import yaml
                        file_config = yaml.safe_load(f)
                    except ImportError:
                        logging.error(
                            "YAML config requires 'PyYAML'. "
                            "Please install or use JSON."
                        )
                        sys.exit(1)
                else:
                    file_config = json.load(f)

            # Deep merge file_config into config
            for key, value in file_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
            logging.info(f"Loaded config from {args.config}")
        except Exception as e:
            logging.error(f"Failed to load config file {args.config}: {e}")
            sys.exit(1)

    # CLI flags override everything
    if args.seed:
        config["global"]["seed"] = args.seed
    if args.num_users:
        config["global"]["num_users"] = args.num_users
    if args.num_apps:
        config["global"]["num_apps"] = args.num_apps
    if args.num_entitlements:
        config["global"]["num_entitlements"] = args.num_entitlements

    return config


def save_data(
        out_dir: Path,
        no_clobber: bool,
        **dataframes: pd.DataFrame
) -> None:
    """Save DataFrames to CSV files in the output directory."""
    logger = logging.getLogger("save_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, df in dataframes.items():
        outfile = out_dir / f"{name}.csv"
        if no_clobber and outfile.exists():
            logger.warning(
                f"File {outfile} exists and --no-clobber is set. Skipping."
            )
            continue

        logger.info(f"Saving {name}.csv to {outfile}...")
        # Add quoting=csv.QUOTE_ALL
        df.to_csv(outfile, index=False, quoting=csv.QUOTE_ALL)


def write_entitlements_by_app(
        entitlements_df: pd.DataFrame, out_dir: Path
) -> None:
    """Split entitlements into one file per app."""
    logger = logging.getLogger("write_ent_by_app")
    split_dir = out_dir / "entitlements_by_app"
    split_dir.mkdir(parents=True, exist_ok=True)

    # Use specified headers
    cols_to_save = ["entitlement_id", "entitlement_name", "app_id", "description"]
    # Filter columns that exist in the dataframe
    cols_to_save = [col for col in cols_to_save if col in entitlements_df.columns]

    for app_id, group_df in entitlements_df.groupby("app_id"):
        outfile = split_dir / f"{app_id}.csv"
        logger.debug(f"Saving split entitlements to {outfile}")

        df_to_save = group_df[cols_to_save]

        # Rename app_id to application_id
        if "app_id" in df_to_save.columns:
            df_to_save = df_to_save.rename(columns={"app_id": "application_id"})

        # Add quoting=csv.QUOTE_ALL
        df_to_save.to_csv(outfile, index=False, quoting=csv.QUOTE_ALL)

    logger.info(f"Split entitlements saved to {split_dir}/")

def write_entitlement_grants_by_app(
        assignments_df: pd.DataFrame,
        entitlements_df: pd.DataFrame,
        identities_df: pd.DataFrame,
        out_dir: Path
) -> None:
    """
       Writes entitlement grants grouped by user, per app, to separate CSV files.
       Format: "user_id","user_name","entitlement_grants","application_id","password"
       Example: "U0000908","jsmith","E0007260#E0011489","APP01",""
       """

    logger = logging.getLogger("write_ent_grants_by_app")
    split_dir = out_dir / "entitlement_grants_by_app"
    split_dir.mkdir(parents=True, exist_ok=True)

    # Need app_id from entitlements_df
    assignments_with_app = pd.merge(
        assignments_df,
        entitlements_df[['entitlement_id', 'app_id']],
        on='entitlement_id'
    )

    logger.info(f"Generating entitlement grant files in {split_dir}/")

    # Group by app
    for app_id, app_group_df in assignments_with_app.groupby("app_id"):
        # Group by user, aggregate entitlements into a #-separated string
        user_grants_df = app_group_df.groupby('user_id')['entitlement_id'].apply(
            lambda ents: '#'.join(sorted(ents))
        ).reset_index()

        # Rename columns to match spec
        user_grants_df = user_grants_df.rename(
            columns={'entitlement_id': 'entitlement_grants'}
        )

        # Join in user_name (and password if present) from identities_df
        merge_cols = ['user_id', 'user_name']
        if 'password' in identities_df.columns:
            merge_cols.append('password')

        user_grants_df = user_grants_df.merge(
            identities_df[merge_cols].drop_duplicates('user_id'),
            on='user_id',
            how='left'
        )

        # If identities_df does not yet have a password column, add an empty one
        if 'password' not in user_grants_df.columns:
            user_grants_df['password'] = ""

        # Add the application_id column
        user_grants_df['application_id'] = app_id

        # Reorder columns: user_name second, password last
        user_grants_df = user_grants_df[
            ['user_id', 'user_name', 'entitlement_grants', 'application_id', 'password']
        ]

        # Save the CSV
        outfile = split_dir / f"{app_id}_grants.csv"
        logger.debug(f"Saving app grants to {outfile}")

        # Save with double quotes
        user_grants_df.to_csv(
            outfile,
            index=False,
            quoting=csv.QUOTE_ALL
        )

    logger.info(f"Split entitlement grants saved to {split_dir}/")


def run_qa_summary(
        identities: pd.DataFrame,
        entitlements: pd.DataFrame,
        assignments: pd.DataFrame,
        proportions: Dict[str, float],
        out_dir: Path
) -> None:
    """Log a final QA summary and save it to qa_summary.txt."""
    logger = logging.getLogger("QASummary")
    report_lines = []

    report_lines.append("--- Generation QA Summary ---")

    # Counts
    report_lines.append(f"Identities:     {len(identities)} rows (Unique IDs: "
                        f"{identities['user_id'].nunique()})")
    report_lines.append(f"Entitlements:   {len(entitlements)} rows (Unique IDs: "
                        f"{entitlements['entitlement_id'].nunique()})")
    report_lines.append(f"Assignments:    {len(assignments)} rows (Unique pairs: "
                        f"{len(assignments.drop_duplicates(['user_id', 'entitlement_id']))})")

    # Identity Stats
    report_lines.append(f"Avg tenure:     {identities['tenure_years'].mean():.2f} years")
    report_lines.append("Top 5 Departments:")
    report_lines.append(identities['department'].value_counts().head(5).to_string())
    report_lines.append("Top 5 Job Families:")
    report_lines.append(identities['job_family'].value_counts().head(5).to_string())
    report_lines.append("Top 5 Sites:")
    report_lines.append(identities['location_site'].value_counts().head(5).to_string())

    # Entitlement Stats
    report_lines.append("Ents per App:")
    report_lines.append(entitlements['app_name'].value_counts().to_string())
    report_lines.append("Criticality Distribution:")
    report_lines.append(entitlements['criticality'].value_counts(normalize=True).to_string())
    report_lines.append("Sample Descriptions:")
    report_lines.append(entitlements['description'].sample(5, random_state=1).to_string(index=False))

    # Assignment Stats
    ents_per_user = assignments.groupby("user_id").size()
    report_lines.append("Assignments per User:")
    report_lines.append(f"  Avg:    {ents_per_user.mean():.2f}")
    report_lines.append(f"  Median: {ents_per_user.median():.0f}")
    report_lines.append(f"  Min:    {ents_per_user.min():.0f}")
    report_lines.append(f"  Max:    {ents_per_user.max():.0f}")

    users_per_ent = assignments.groupby("entitlement_id").size()
    top_10_ents = users_per_ent.nlargest(10).index
    top_10_names = entitlements[
        entitlements['entitlement_id'].isin(top_10_ents)
    ][['entitlement_name', 'app_name']]

    report_lines.append(f"Top 10 Entitlements by Support (Count):")
    report_lines.append(users_per_ent.nlargest(10).to_string())
    report_lines.append(f"Top 10 Entitlement Names:")
    report_lines.append(top_10_names.to_string(index=False))

    # Calibration Results
    report_lines.append("--- Calibration Summary ---")
    report_lines.append("Final (Pattern Antecedent => Core Ent) Rule Proportions:")
    if proportions:
        report_lines.append(f"  High (>=0.70): {proportions.get('High', 0.0):.2%}")
        report_lines.append(f"  Mid  (0.40-0.70): {proportions.get('Mid', 0.0):.2%}")
        report_lines.append(f"  Low  (<0.40): {proportions.get('Low', 0.0):.2%}")
    else:
        report_lines.append("  Calibration did not produce valid proportions.")
    report_lines.append("-----------------------------")

    # Log complete report to console
    logger.info("\n" + "\n".join(report_lines))

    # Save complete report to file
    report_path = out_dir / "qa_summary.txt"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        logger.info(f"QA Summary saved to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save QA summary to {report_path}: {e}")


def main():
    """Main script execution."""
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    if args.emit_default_config:
        print(json.dumps(DEFAULT_CONFIG, indent=2))
        sys.exit(0)

    if args.run_tests:
        _run_tests()
        sys.exit(0)

    logger = logging.getLogger("main")
    config = load_config(args)

    # --- Setup ---
    seed = config["global"]["seed"]
    num_users = config["global"]["num_users"]

    logger.info(f"Starting data generation with seed: {seed}")

    # Initialize RNGs and Faker for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    Faker.seed(seed)
    faker_instance = Faker()

    # --- Generation Steps ---

    # 1. Identities
    id_gen = IdentityGenerator(config, rng, faker_instance)
    identities_df = id_gen.generate_identities(num_users)

    # 2. Entitlements and Bundles
    ent_gen = EntitlementGenerator(config, rng, faker_instance)
    entitlements_df = ent_gen.generate_entitlements()
    bundles = ent_gen.generate_bundles(entitlements_df)

    # 3. Patterns
    pat_gen = PatternGenerator(config, rng)
    patterns = pat_gen.generate_patterns(identities_df, bundles)

    if not patterns:
        logger.error("No patterns were generated. Aborting.")
        sys.exit(1)

    # 4. Calibration Loop (finds final assignments)
    cal_engine = CalibrationEngine(config, rng)
    assignments_df, final_proportions = cal_engine.run_calibration_loop(
        identities_df, entitlements_df, patterns
    )

    # --- Save Output ---
    save_data(
        args.out_dir,
        args.no_clobber,
        identities=identities_df,
        entitlements=entitlements_df,
        assignments=assignments_df
    )

    # Save split entitlement files
    write_entitlements_by_app(entitlements_df, args.out_dir)

    # Save split entitlement grant files (New)
    write_entitlement_grants_by_app(
        assignments_df,
        entitlements_df,
        identities_df,
        args.out_dir
    )

    # --- Final Report ---
    run_qa_summary(
        identities_df,
        entitlements_df,
        assignments_df,
        final_proportions,
        args.out_dir  # Pass output dir for saving report
    )

    logger.info(f"Generation complete. Output saved to {args.out_dir}")


if __name__ == "__main__":
    main()