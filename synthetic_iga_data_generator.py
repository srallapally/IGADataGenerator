#!/usr/bin/env python3
"""
synthetic_data_generator.py

Generates synthetic Identity Governance and Administration (IGA) data including:
- identities.csv: User identities with 20+ attributes
- <app_name>_entitlements.csv: Entitlement catalogs per application
- <app_name>_accounts.csv: User-entitlement assignments with confidence scores

Special handling for Epic with dual entitlement attributes (linkedTemplates, linkedSubTemplates).

Usage:
    python synthetic_data_generator.py --config data_generator_config.json
"""

import argparse
import csv
import json
import logging
import random
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# External dependencies
try:
    import numpy as np
    from numpy.random import Generator, PCG64
    import pandas as pd
    from faker import Faker
    from scipy.stats import chi2_contingency
except ImportError as e:
    print(f"Error: Missing required library: {e}", file=sys.stderr)
    print("Please install dependencies: pip install pandas numpy faker scipy", file=sys.stderr)
    sys.exit(1)


# =============================================================================
# Configuration Loader
# =============================================================================

class ConfigLoader:
    """Loads and validates the configuration file."""

    REQUIRED_APPS = {"AWS", "Salesforce", "ServiceNow", "Epic"}

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def load(self) -> Dict[str, Any]:
        """Load and validate configuration from JSON file."""
        self.logger.info(f"Loading configuration from {self.config_path}")

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self._flatten_values()  # Must come BEFORE _validate()
        self._validate()

        return self.config

    def _validate(self) -> None:
        """Validate required configuration sections and values."""
        required_sections = ['global', 'identity', 'applications', 'grants', 'confidence', 'features']

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

        # Validate mandatory applications
        app_names = {app['app_name'] for app in self.config['applications']['apps']}
        missing_apps = self.REQUIRED_APPS - app_names
        if missing_apps:
            raise ValueError(f"Missing mandatory applications: {missing_apps}")

        # Validate distributions sum to 1.0
        self._validate_distribution('identity', 'distribution_employee_contractor')
        self._validate_distribution('identity', 'distribution_human_ai_agent')
        self._validate_distribution('confidence', 'distribution')

        self.logger.info("Configuration validation passed")

    def _validate_distribution(self, section: str, key: str) -> None:
        """Validate that a distribution sums to 1.0."""
        dist = self.config[section].get(key, {})
        if isinstance(dist, dict) and 'value' in dist:
            dist = dist['value']

        if isinstance(dist, dict):
            total = sum(dist.values())
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"Distribution {section}.{key} must sum to 1.0, got {total}")

    def _flatten_values(self) -> None:
        """Flatten nested 'value' keys for easier access."""
        self.config = self._flatten_recursive(self.config)

    def _flatten_recursive(self, obj: Any) -> Any:
        """Recursively flatten objects with 'value' keys."""
        if isinstance(obj, dict):
            # If dict has only 'value' and '_comment' keys, return the value
            keys = set(obj.keys())
            if keys == {'value'} or keys == {'value', '_comment'}:
                return self._flatten_recursive(obj['value'])

            # Otherwise, recurse into all values
            return {k: self._flatten_recursive(v) for k, v in obj.items()
                    if not k.startswith('_comment')}

        elif isinstance(obj, list):
            return [self._flatten_recursive(item) for item in obj]

        return obj

    def get(self, *keys, default=None) -> Any:
        """Get nested config value using dot notation."""
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value


# =============================================================================
# Input Data Reader
# =============================================================================

class InputDataReader:
    """Reads and parses input CSV files."""

    def __init__(self, base_path: Path = Path(".")):
        self.base_path = base_path
        self.cache: Dict[str, List[Dict[str, str]]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def read_csv(self, file_path: str) -> List[Dict[str, str]]:
        """Read a CSV file and return list of dictionaries."""
        if file_path in self.cache:
            return self.cache[file_path]

        full_path = self.base_path / file_path
        if not full_path.exists():
            self.logger.warning(f"Input file not found: {full_path}")
            return []

        try:
            with open(full_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.cache[file_path] = rows
                self.logger.info(f"Loaded {len(rows)} rows from {file_path}")
                return rows
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return []

    def get_column_values(self, file_path: str, column: str) -> List[str]:
        """Get unique values from a specific column."""
        rows = self.read_csv(file_path)
        values = [row.get(column, '') for row in rows if row.get(column)]
        return values

    def get_unique_column_values(self, file_path: str, column: str) -> List[str]:
        """Get unique values from a specific column."""
        return list(set(self.get_column_values(file_path, column)))


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Identity:
    """Represents a user identity with all attributes."""
    user_id: str
    user_name: str
    first_name: str
    last_name: str
    email: str
    department: str
    department_id: str
    business_unit: str
    department_type: str
    jobcode: str
    job_category: str
    job_level: str
    manager: Optional[str]
    manager_name: Optional[str]
    location_country: str
    location_site: str
    employment_type: str
    identity_type: str
    status: str
    cost_center: str
    tenure_years: str
    hire_date: str
    is_manager: str
    employee_id: str


@dataclass
class Entitlement:
    """Represents an entitlement."""
    entitlement_id: str
    entitlement_name: str
    app_id: str
    app_name: str
    entitlement_type: str
    criticality: str
    description: str


@dataclass
class Account:
    """Represents a user's account in an application."""
    user_id: str
    user_name: str
    app_id: str
    app_name: str
    entitlement_grants: Dict[str, List[str]]  # attribute_name -> list of entitlement IDs
    confidence_score: Optional[float]
    confidence_bucket: Optional[str]


# =============================================================================
# Identity Generator
# =============================================================================

class IdentityGenerator:
    """Generates synthetic user identities."""

    JOB_LEVEL_KEYWORDS = {
        'Executive': ['CEO', 'CFO', 'CIO', 'CTO', 'COO', 'Chief', 'President'],
        'VP': ['VP', 'Vice President'],
        'Director': ['Director'],
        'Manager': ['Manager', 'Supervisor', 'Head'],
        'Lead': ['Lead', 'Principal', 'Senior Staff'],
        'Senior': ['Senior', 'Sr.', 'Sr '],
        'Mid': ['Specialist', 'Analyst', 'Engineer', 'Coordinator'],
        'Junior': ['Associate', 'Assistant', 'Junior', 'Jr.', 'Entry', 'Intern']
    }

    def __init__(self, config: Dict[str, Any], rng: Generator, faker: Faker,
                 input_reader: InputDataReader):
        self.config = config.get('identity', {})
        self.rng = rng
        self.faker = faker
        self.input_reader = input_reader
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load input data
        self._load_input_data()

        # Track generated data for lookups
        self.identities: List[Identity] = []
        self.used_usernames: Set[str] = set()
        self.managers: List[str] = []  # List of user_ids who are managers

    def _load_input_data(self) -> None:
        """Load departments and job titles from input files."""
        input_files = self.config.get('input_files', {})

        # Load departments
        dept_file = input_files.get('departments', 'input/departments.csv')
        self.departments = self.input_reader.read_csv(dept_file)
        self.logger.info(f"Loaded {len(self.departments)} departments")

        # Load job titles
        job_file = input_files.get('jobtitles', 'input/jobtitles.csv')
        self.jobtitles = self.input_reader.read_csv(job_file)
        self.logger.info(f"Loaded {len(self.jobtitles)} job titles")

        # Build job title lookup by department/industry
        self.jobs_by_industry = defaultdict(list)
        for job in self.jobtitles:
            industry = job.get('Industry', 'General')
            self.jobs_by_industry[industry].append(job)

    def _weighted_choice(self, choices: Dict[str, float]) -> str:
        """Make a weighted random choice."""
        items = list(choices.keys())
        weights = list(choices.values())
        return self.rng.choice(items, p=np.array(weights) / sum(weights))

    def _generate_username(self, first_name: str, last_name: str) -> str:
        """Generate unique username from first and last name."""
        base = f"{first_name}{last_name}".lower()
        # Remove non-alphanumeric characters
        base = ''.join(c for c in base if c.isalnum())
        if not base:
            base = "user"

        username = base
        suffix = 1
        while username in self.used_usernames:
            username = f"{base}{suffix}"
            suffix += 1

        self.used_usernames.add(username)
        return username

    def _infer_job_level(self, job_title: str) -> str:
        """Infer job level from job title keywords."""
        title_upper = job_title.upper()
        for level, keywords in self.JOB_LEVEL_KEYWORDS.items():
            for keyword in keywords:
                if keyword.upper() in title_upper:
                    return level
        return "Mid"  # Default

    def _select_department(self) -> Dict[str, str]:
        """Select a random department from input data."""
        if not self.departments:
            return {
                'department_id': 'DEPT_DEFAULT',
                'department_name': 'General',
                'Industry': 'General',
                'Type': 'Business'
            }
        return self.rng.choice(self.departments)

    def _select_job_for_department(self, industry: str) -> Dict[str, str]:
        """Select a job title appropriate for the department's industry."""
        jobs = self.jobs_by_industry.get(industry, [])
        if not jobs:
            # Fallback to any job
            jobs = self.jobtitles if self.jobtitles else [{'Job Title': 'Specialist', 'Category': 'Business'}]
        return self.rng.choice(jobs) if jobs else {'Job Title': 'Specialist', 'Category': 'Business'}

    def _generate_tenure(self) -> float:
        """Generate tenure using Beta distribution."""
        tenure_cfg = self.config.get('tenure', {})
        a = tenure_cfg.get('beta_a', 2.0)
        b = tenure_cfg.get('beta_b', 5.0)
        scale = tenure_cfg.get('scale_years', 20.0)
        return float(self.rng.beta(a, b) * scale)

    def generate(self, num_identities: int) -> List[Identity]:
        """Generate the specified number of identities."""
        self.logger.info(f"Generating {num_identities} identities...")

        # Configuration values
        pct_no_manager = self.config.get('pct_users_without_manager', 0.10)
        if pct_no_manager is None:
            raise ValueError("identity.pct_users_without_manager must be set in the configuration")

        emp_dist = self.config.get('distribution_employee_contractor', {'Employee': 0.9, 'Contractor': 0.1})
        identity_dist = self.config.get('distribution_human_ai_agent', {'Human': 0.95, 'AI_Agent': 0.05})
        status_dist = self.config.get('distribution_active_inactive', {'Active': 0.93, 'Inactive': 0.07})

        # Location distributions
        country_dist = {'US': 0.60, 'GB': 0.15, 'IN': 0.15, 'DE': 0.05, 'AU': 0.05}
        site_by_country = {
            'US': {'SFO': 0.3, 'NYC': 0.25, 'AUS': 0.15, 'CHI': 0.1, 'SEA': 0.1, 'BOS': 0.1},
            'GB': {'LON': 0.7, 'MAN': 0.2, 'EDI': 0.1},
            'IN': {'BLR': 0.5, 'MUM': 0.3, 'HYD': 0.2},
            'DE': {'BER': 0.6, 'MUN': 0.4},
            'AU': {'SYD': 0.7, 'MEL': 0.3}
        }

        # First pass: generate basic identities
        for i in range(num_identities):
            user_id = f"U{i + 1:07d}"
            employee_id = f"EMP{i + 1:06d}"

            first_name = self.faker.first_name()
            last_name = self.faker.last_name()
            user_name = self._generate_username(first_name, last_name)
            email = f"{first_name.lower()}.{last_name.lower()}@company.com"

            # Select department and job
            dept = self._select_department()
            department = dept.get('department_name', 'General')
            department_id = dept.get('department_id', 'DEPT_GEN')
            business_unit = dept.get('Industry', 'General')
            department_type = dept.get('Type', 'Business')

            job = self._select_job_for_department(business_unit)
            jobcode = job.get('Job Title', 'Specialist')
            job_category = job.get('Category', 'Business')
            job_level = self._infer_job_level(jobcode)

            # Location
            country = self._weighted_choice(country_dist)
            site_options = site_by_country.get(country, {'HQ': 1.0})
            site = self._weighted_choice(site_options)

            # Employment and identity type
            employment_type = self._weighted_choice(emp_dist)
            identity_type = self._weighted_choice(identity_dist)
            status = self._weighted_choice(status_dist)

            # Tenure
            tenure = self._generate_tenure()
            hire_date = (datetime.now() - timedelta(days=int(tenure * 365))).strftime('%Y-%m-%d')

            # Cost center
            cost_center = f"CC_{department_id[:10]}_{self.rng.integers(100, 999)}"

            identity = Identity(
                user_id=user_id,
                user_name=user_name,
                first_name=first_name,
                last_name=last_name,
                email=email,
                department=department,
                department_id=department_id,
                business_unit=business_unit,
                department_type=department_type,
                jobcode=jobcode,
                job_category=job_category,
                job_level=job_level,
                manager=None,  # Set in second pass
                manager_name=None,
                location_country=country,
                location_site=site,
                employment_type=employment_type,
                identity_type=identity_type,
                status=status,
                cost_center=cost_center,
                tenure_years=f"{tenure:.1f}",
                hire_date=hire_date,
                is_manager='N',  # Set in second pass
                employee_id=employee_id
            )
            self.identities.append(identity)

        # Second pass: assign managers and hierarchy
        self._assign_managers(pct_no_manager)

        self.logger.info(f"Generated {len(self.identities)} identities")
        return self.identities

    def _assign_managers(self, pct_no_manager: float) -> None:
        """Assign managers to create hierarchy."""
        self.logger.info("Assigning manager hierarchy...")

        num_identities = len(self.identities)
        hierarchy_cfg = self.config.get('manager_hierarchy', {})
        pct_managers = hierarchy_cfg.get('pct_managers', 0.25)

        # Determine who will be managers (based on job level)
        manager_candidates = []
        for i, identity in enumerate(self.identities):
            if identity.job_level in ['Manager', 'Director', 'VP', 'Executive', 'Lead']:
                manager_candidates.append(i)

        # If not enough manager candidates, add more based on tenure
        target_managers = int(num_identities * pct_managers)
        if len(manager_candidates) < target_managers:
            non_managers = [i for i in range(num_identities) if i not in manager_candidates]
            # Sort by tenure (higher tenure = more likely to be manager)
            non_managers.sort(key=lambda i: float(self.identities[i].tenure_years), reverse=True)
            additional = non_managers[:target_managers - len(manager_candidates)]
            manager_candidates.extend(additional)

        # Shuffle and select final managers
        self.rng.shuffle(manager_candidates)
        manager_indices = set(manager_candidates[:target_managers])

        # Mark managers
        for idx in manager_indices:
            self.identities[idx].is_manager = 'Y'
            self.managers.append(self.identities[idx].user_id)

        # Determine users without managers
        num_no_manager = int(num_identities * pct_no_manager)

        # Top-level managers (executives) don't have managers
        executive_indices = [i for i in manager_indices
                             if self.identities[i].job_level in ['Executive', 'VP']]

        # If not enough executives, pick from manager pool
        if len(executive_indices) < num_no_manager:
            remaining_managers = [i for i in manager_indices if i not in executive_indices]
            self.rng.shuffle(remaining_managers)
            executive_indices.extend(remaining_managers[:num_no_manager - len(executive_indices)])

        no_manager_indices = set(executive_indices[:num_no_manager])

        # Assign managers to everyone else
        manager_list = list(manager_indices - no_manager_indices)
        if not manager_list:
            manager_list = list(manager_indices)  # Fallback

        for i, identity in enumerate(self.identities):
            if i in no_manager_indices:
                identity.manager = None
                identity.manager_name = None
            else:
                # Assign a random manager
                mgr_idx = self.rng.choice(manager_list)
                mgr = self.identities[mgr_idx]
                identity.manager = mgr.user_id
                identity.manager_name = mgr.user_name

        self.logger.info(f"Assigned {len(manager_indices)} managers, {len(no_manager_indices)} without managers")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert identities to DataFrame."""
        data = []
        for identity in self.identities:
            data.append({
                'user_id': identity.user_id,
                'user_name': identity.user_name,
                'first_name': identity.first_name,
                'last_name': identity.last_name,
                'email': identity.email,
                'department': identity.department,
                'department_id': identity.department_id,
                'business_unit': identity.business_unit,
                'department_type': identity.department_type,
                'jobcode': identity.jobcode,
                'job_category': identity.job_category,
                'job_level': identity.job_level,
                'manager': identity.manager or '',
                'manager_name': identity.manager_name or '',
                'location_country': identity.location_country,
                'location_site': identity.location_site,
                'employment_type': identity.employment_type,
                'identity_type': identity.identity_type,
                'status': identity.status,
                'cost_center': identity.cost_center,
                'tenure_years': identity.tenure_years,
                'hire_date': identity.hire_date,
                'is_manager': identity.is_manager,
                'employee_id': identity.employee_id
            })
        return pd.DataFrame(data)


# =============================================================================
# Entitlement Generator
# =============================================================================

class EntitlementGenerator:
    """Generates entitlement catalogs for each application."""

    def __init__(self, config: Dict[str, Any], rng: Generator, faker: Faker,
                 input_reader: InputDataReader):
        self.config = config
        self.rng = rng
        self.faker = faker
        self.input_reader = input_reader
        self.logger = logging.getLogger(self.__class__.__name__)

        self.entitlements_by_app: Dict[str, List[Entitlement]] = {}
        self.ent_counter = 1

    def generate_for_app(self, app_config: Dict[str, Any]) -> List[Entitlement]:
        """Generate entitlements for a single application."""
        app_name = app_config['app_name']
        app_id = app_config.get('app_id', f"APP_{app_name.upper()}")
        self.logger.info(f"Generating entitlements for {app_name}...")

        entitlements = []
        criticality_dist = app_config.get('criticality_distribution', {'High': 0.2, 'Medium': 0.5, 'Low': 0.3})

        # Handle Epic special case with multiple entitlement attributes
        if 'entitlement_attributes' in app_config:
            for attr_config in app_config['entitlement_attributes']:
                attr_name = attr_config['attribute_name']
                input_cfg = attr_config.get('input_file', {})

                if input_cfg:
                    file_path = input_cfg.get('path', '')
                    id_col = input_cfg.get('entitlement_id_column', 'Value')
                    name_col = input_cfg.get('entitlement_name_column', 'Title')

                    rows = self.input_reader.read_csv(file_path)
                    for row in rows:
                        ent_id = row.get(id_col, f"E{self.ent_counter:07d}")
                        ent_name = row.get(name_col, f"Entitlement_{self.ent_counter}")

                        criticality = self._weighted_choice(criticality_dist)
                        entitlements.append(Entitlement(
                            entitlement_id=ent_id,
                            entitlement_name=ent_name,
                            app_id=app_id,
                            app_name=app_name,
                            entitlement_type=attr_name,
                            criticality=criticality,
                            description=f"{app_name} {attr_name}: {ent_name[:50]}"
                        ))
                        self.ent_counter += 1

        # Handle standard single input file
        elif 'input_file' in app_config:
            input_cfg = app_config['input_file']
            file_path = input_cfg.get('path', '')
            id_col = input_cfg.get('entitlement_id_column', 'id')
            name_col = input_cfg.get('entitlement_name_column', 'name')
            desc_col = input_cfg.get('description_column', '')

            rows = self.input_reader.read_csv(file_path)
            num_ents = app_config.get('num_entitlements', len(rows))

            # Sample if we have more rows than needed
            if len(rows) > num_ents:
                indices = self.rng.choice(len(rows), size=num_ents, replace=False)
                rows = [rows[i] for i in indices]

            for row in rows:
                ent_id = row.get(id_col, f"E{self.ent_counter:07d}")
                ent_name = row.get(name_col, f"Entitlement_{self.ent_counter}")
                description = row.get(desc_col, '') if desc_col else f"{app_name} entitlement: {ent_name[:50]}"

                criticality = self._weighted_choice(criticality_dist)
                entitlements.append(Entitlement(
                    entitlement_id=ent_id,
                    entitlement_name=ent_name,
                    app_id=app_id,
                    app_name=app_name,
                    entitlement_type='standard',
                    criticality=criticality,
                    description=description[:100] if description else ''
                ))
                self.ent_counter += 1

        # Handle multiple input files (e.g., Salesforce)
        elif 'input_files' in app_config:
            for file_cfg in app_config['input_files']:
                file_path = file_cfg.get('path', '')
                id_col = file_cfg.get('entitlement_id_column', 'id')
                name_col = file_cfg.get('entitlement_name_column', 'name')
                desc_col = file_cfg.get('description_column', '')
                ent_type = file_cfg.get('entitlement_type', 'standard')

                rows = self.input_reader.read_csv(file_path)
                for row in rows:
                    ent_id = row.get(id_col, f"E{self.ent_counter:07d}")
                    ent_name = row.get(name_col, f"Entitlement_{self.ent_counter}")
                    description = row.get(desc_col, '') if desc_col else ''

                    criticality = self._weighted_choice(criticality_dist)
                    entitlements.append(Entitlement(
                        entitlement_id=ent_id,
                        entitlement_name=ent_name,
                        app_id=app_id,
                        app_name=app_name,
                        entitlement_type=ent_type,
                        criticality=criticality,
                        description=description[:100] if description else f"{app_name} {ent_type}: {ent_name[:40]}"
                    ))
                    self.ent_counter += 1

        self.entitlements_by_app[app_name] = entitlements
        self.logger.info(f"Generated {len(entitlements)} entitlements for {app_name}")
        return entitlements

    def _weighted_choice(self, choices: Dict[str, float]) -> str:
        """Make a weighted random choice."""
        items = list(choices.keys())
        weights = list(choices.values())
        return self.rng.choice(items, p=np.array(weights) / sum(weights))

    def generate_all(self) -> Dict[str, List[Entitlement]]:
        """Generate entitlements for all configured applications."""
        apps_cfg = self.config.get('applications', {})
        app_list = apps_cfg.get('apps', [])
        mandatory_apps = set(apps_cfg.get('mandatory_apps', []))

        for app_config in app_list:
            if app_config.get('enabled', True):
                self.generate_for_app(app_config)

        # Ensure mandatory apps always have at least one entitlement
        for mandatory_app in mandatory_apps:
            ents = self.entitlements_by_app.get(mandatory_app, [])
            if ents:
                continue

            self.logger.warning(
                f"No entitlements generated for mandatory app '{mandatory_app}'. "
                f"Creating fallback entitlements."
            )

            # Create a small set of generic fallback entitlements
            fallback_entitlements: List[Entitlement] = []
            criticality_dist = {
                "High": 0.2,
                "Medium": 0.5,
                "Low": 0.3,
            }
            app_id = f"APP_{mandatory_app.upper()}"

            for i in range(5):
                ent_id = f"FALLBACK_E{self.ent_counter:07d}"
                self.ent_counter += 1
                ent_name = f"{mandatory_app}_Fallback_{i + 1}"
                criticality = self._weighted_choice(criticality_dist)
                fallback_entitlements.append(
                    Entitlement(
                        entitlement_id=ent_id,
                        entitlement_name=ent_name,
                        app_id=app_id,
                        app_name=mandatory_app,
                        entitlement_type="standard",
                        criticality=criticality,
                        description=f"{mandatory_app} fallback entitlement {i + 1}",
                    )
                )

            self.entitlements_by_app[mandatory_app] = fallback_entitlements

        return self.entitlements_by_app

    def to_dataframe(self, app_name: str) -> pd.DataFrame:
        """Convert entitlements for an app to DataFrame."""
        entitlements = self.entitlements_by_app.get(app_name, [])
        data = []
        for ent in entitlements:
            data.append({
                'entitlement_id': ent.entitlement_id,
                'entitlement_name': ent.entitlement_name,
                'app_id': ent.app_id,
                'app_name': ent.app_name,
                'entitlement_type': ent.entitlement_type,
                'criticality': ent.criticality,
                'description': ent.description
            })
        return pd.DataFrame(data)


# =============================================================================
# Account/Grant Generator
# =============================================================================

class AccountGenerator:
    """Generates user accounts with entitlement grants."""

    def __init__(self, config: Dict[str, Any], rng: Generator,
                 identities: List[Identity],
                 entitlements_by_app: Dict[str, List[Entitlement]]):
        self.config = config
        self.rng = rng
        self.identities = identities
        self.entitlements_by_app = entitlements_by_app
        self.logger = logging.getLogger(self.__class__.__name__)

        self.accounts_by_app: Dict[str, List[Account]] = {}
        self.delimiter = config.get('global', {}).get('multi_value_delimiter', '#')

    def _get_grant_count(self, app_config: Dict[str, Any], attr_name: str = None) -> int:
        """Get number of entitlements to grant using bell curve distribution."""
        grants_cfg = self.config.get('grants', {})

        # Check for attribute-specific override (e.g., Epic)
        if attr_name and 'entitlement_attributes' in app_config:
            for attr_cfg in app_config['entitlement_attributes']:
                if attr_cfg.get('attribute_name') == attr_name:
                    attr_grants = attr_cfg.get('grants_per_user', {})
                    if attr_grants:
                        mean = attr_grants.get('mean', grants_cfg.get('mean_entitlements_per_user', 8))
                        std = attr_grants.get('std_dev', grants_cfg.get('std_dev_entitlements', 3))
                        min_val = attr_grants.get('min', grants_cfg.get('min_entitlements', 0))
                        max_val = attr_grants.get('max', grants_cfg.get('max_entitlements', 25))

                        count = int(self.rng.normal(mean, std))
                        return max(min_val, min(max_val, count))

        # Default distribution
        mean = grants_cfg.get('mean_entitlements_per_user', 8)
        std = grants_cfg.get('std_dev_entitlements', 3)
        min_val = grants_cfg.get('min_entitlements', 0)
        max_val = grants_cfg.get('max_entitlements', 25)

        count = int(self.rng.normal(mean, std))
        return max(min_val, min(max_val, count))

    def generate_for_app(self, app_config: Dict[str, Any]) -> List[Account]:
        """Generate accounts for a single application."""
        app_name = app_config['app_name']
        app_id = app_config.get('app_id', f"APP_{app_name.upper()}")
        self.logger.info(f"Generating accounts for {app_name}...")

        entitlements = self.entitlements_by_app.get(app_name, [])
        if not entitlements:
            self.logger.warning(f"No entitlements found for {app_name}")
            return []

        accounts = []

        # Handle Epic special case
        if app_name == "Epic":
            accounts = self._generate_epic_accounts(app_config, app_id)
        else:
            # Standard account generation
            ent_ids = [e.entitlement_id for e in entitlements]

            for identity in self.identities:
                num_grants = self._get_grant_count(app_config)

                if num_grants > 0 and ent_ids:
                    granted = self.rng.choice(
                        ent_ids,
                        size=min(num_grants, len(ent_ids)),
                        replace=False
                    ).tolist()
                else:
                    granted = []

                accounts.append(Account(
                    user_id=identity.user_id,
                    user_name=identity.user_name,
                    app_id=app_id,
                    app_name=app_name,
                    entitlement_grants={'entitlement_grants': granted},
                    confidence_score=None,  # Set later
                    confidence_bucket=None
                ))

        self.accounts_by_app[app_name] = accounts
        self.logger.info(f"Generated {len(accounts)} accounts for {app_name}")
        return accounts

    def _generate_epic_accounts(self, app_config: Dict[str, Any], app_id: str) -> List[Account]:
        """Generate Epic accounts with dual entitlement attributes."""
        accounts = []
        entitlements = self.entitlements_by_app.get('Epic', [])

        # Separate by type
        linked_templates = [e.entitlement_id for e in entitlements if e.entitlement_type == 'linkedTemplates']
        sub_templates = [e.entitlement_id for e in entitlements if e.entitlement_type == 'linkedSubTemplates']

        for identity in self.identities:
            grants = {}

            # Grant linked templates
            num_linked = self._get_grant_count(app_config, 'linkedTemplates')
            if num_linked > 0 and linked_templates:
                grants['linkedTemplates'] = self.rng.choice(
                    linked_templates,
                    size=min(num_linked, len(linked_templates)),
                    replace=False
                ).tolist()
            else:
                grants['linkedTemplates'] = []

            # Grant sub templates
            num_sub = self._get_grant_count(app_config, 'linkedSubTemplates')
            if num_sub > 0 and sub_templates:
                grants['linkedSubTemplates'] = self.rng.choice(
                    sub_templates,
                    size=min(num_sub, len(sub_templates)),
                    replace=False
                ).tolist()
            else:
                grants['linkedSubTemplates'] = []

            accounts.append(Account(
                user_id=identity.user_id,
                user_name=identity.user_name,
                app_id=app_id,
                app_name='Epic',
                entitlement_grants=grants,
                confidence_score=None,
                confidence_bucket=None
            ))

        return accounts

    def generate_all(self) -> Dict[str, List[Account]]:
        """Generate accounts for all configured applications."""
        apps_cfg = self.config.get('applications', {})
        app_list = apps_cfg.get('apps', [])

        for app_config in app_list:
            if app_config.get('enabled', True):
                self.generate_for_app(app_config)

        # Enforce 80/20 rule across all apps
        self._enforce_grant_distribution()

        # Ensure mandatory apps have at least one account with grants
        self._ensure_mandatory_app_grants()

        # Ensure all entitlements of all mandatory apps are used at least once
        self._ensure_mandatory_entitlement_coverage()

        return self.accounts_by_app

    def _ensure_mandatory_app_grants(self) -> None:
        """Guarantee that each mandatory app has at least one account with entitlement grants."""
        apps_cfg = self.config.get('applications', {})
        mandatory_apps = set(apps_cfg.get('mandatory_apps', []))

        for mandatory_app in mandatory_apps:
            accounts = self.accounts_by_app.get(mandatory_app)
            entitlements = self.entitlements_by_app.get(mandatory_app)

            # If we don't have accounts or entitlements for this app, nothing to do here.
            if not accounts or not entitlements:
                continue

            # Check if any account already has at least one grant
            has_grants = any(
                sum(len(grants) for grants in account.entitlement_grants.values()) > 0
                for account in accounts
            )
            if has_grants:
                continue

            self.logger.warning(
                f"Mandatory app '{mandatory_app}' has accounts but no grants. "
                f"Adding fallback grants to the first account."
            )

            ent_ids = [e.entitlement_id for e in entitlements]
            if not ent_ids:
                continue

            # Assign a small set of entitlements to the first account
            grants_to_add = self.rng.choice(
                ent_ids,
                size=min(3, len(ent_ids)),
                replace=False
            ).tolist()

            first_account = accounts[0]
            # Use the first entitlement attribute on the account
            attr_name = list(first_account.entitlement_grants.keys())[0]
            current = set(first_account.entitlement_grants.get(attr_name, []))
            new_grants = [e for e in grants_to_add if e not in current]
            first_account.entitlement_grants.setdefault(attr_name, [])
            first_account.entitlement_grants[attr_name].extend(new_grants)

    def _ensure_mandatory_entitlement_coverage(self) -> None:
        """
        Ensure that every entitlement of each mandatory app is used
        at least once in entitlement grants.

        For Epic, we respect the entitlement_type -> attribute_name mapping
        (linkedTemplates vs linkedSubTemplates).
        """
        apps_cfg = self.config.get('applications', {})
        mandatory_apps = set(apps_cfg.get('mandatory_apps', []))

        for mandatory_app in mandatory_apps:
            entitlements = self.entitlements_by_app.get(mandatory_app)
            accounts = self.accounts_by_app.get(mandatory_app)

            if not entitlements or not accounts:
                continue

            # Build current usage set
            used_entitlements: set[str] = set()
            for account in accounts:
                for grants in account.entitlement_grants.values():
                    used_entitlements.update(grants)

            # Assign any missing entitlement at least once
            for ent in entitlements:
                if ent.entitlement_id in used_entitlements:
                    continue

                # Choose a random account to receive this entitlement
                account = self.rng.choice(accounts)

                if mandatory_app == "Epic":
                    # Map entitlement_type to the correct Epic grant attribute
                    attr_name = 'linkedTemplates' if ent.entitlement_type == 'linkedTemplates' else 'linkedSubTemplates'
                else:
                    attr_name = 'entitlement_grants'

                current_list = account.entitlement_grants.get(attr_name, [])
                if ent.entitlement_id not in current_list:
                    account.entitlement_grants.setdefault(attr_name, [])
                    account.entitlement_grants[attr_name].append(ent.entitlement_id)

            self.logger.info(
                f"Ensured coverage for all {len(entitlements)} entitlements "
                f"of mandatory app '{mandatory_app}'."
            )

    def _enforce_grant_distribution(self) -> None:
        """Ensure 80% of users have at least 3 entitlements across all apps."""
        grants_cfg = self.config.get('grants', {})
        target_pct = grants_cfg.get('pct_users_with_3_plus', 0.80)

        self.logger.info(f"Enforcing {target_pct:.0%} users with 3+ entitlements...")

        # Count total entitlements per user
        user_total_grants = defaultdict(int)
        for app_name, accounts in self.accounts_by_app.items():
            for account in accounts:
                for attr_name, grants in account.entitlement_grants.items():
                    user_total_grants[account.user_id] += len(grants)

        # Check current distribution
        num_users = len(self.identities)
        users_with_3_plus = sum(1 for count in user_total_grants.values() if count >= 3)
        current_pct = users_with_3_plus / num_users if num_users > 0 else 0

        self.logger.info(f"Current: {current_pct:.1%} users have 3+ entitlements")

        if current_pct >= target_pct:
            return  # Already meeting target

        # Boost users who need more entitlements
        target_count = int(num_users * target_pct)
        shortfall = target_count - users_with_3_plus

        users_needing_boost = [
            uid for uid, count in user_total_grants.items() if count < 3
        ]
        self.rng.shuffle(users_needing_boost)
        users_to_boost = users_needing_boost[:shortfall]

        # Add random entitlements to these users
        for user_id in users_to_boost:
            needed = 3 - user_total_grants[user_id]

            # Pick a random app to add entitlements
            app_name = self.rng.choice(list(self.accounts_by_app.keys()))
            entitlements = self.entitlements_by_app.get(app_name, [])

            if not entitlements:
                continue

            ent_ids = [e.entitlement_id for e in entitlements]

            # Find the account for this user in this app
            for account in self.accounts_by_app[app_name]:
                if account.user_id == user_id:
                    # Get first attribute name
                    attr_name = list(account.entitlement_grants.keys())[0]
                    current_grants = set(account.entitlement_grants[attr_name])

                    # Add new entitlements
                    available = [e for e in ent_ids if e not in current_grants]
                    if available:
                        new_grants = self.rng.choice(
                            available,
                            size=min(needed, len(available)),
                            replace=False
                        ).tolist()
                        account.entitlement_grants[attr_name].extend(new_grants)
                    break

        self.logger.info(f"Boosted {len(users_to_boost)} users to meet 80/20 target")

    def to_dataframe(self, app_name: str) -> pd.DataFrame:
        """Convert accounts for an app to DataFrame."""
        accounts = self.accounts_by_app.get(app_name, [])
        data = []

        for account in accounts:
            row = {
                'user_id': account.user_id,
                'user_name': account.user_name,
            }

            # Add entitlement grant columns
            for attr_name, grants in account.entitlement_grants.items():
                row[attr_name] = self.delimiter.join(grants) if grants else ''

            row['confidence_score'] = account.confidence_score if account.confidence_score is not None else ''
            row['confidence_bucket'] = account.confidence_bucket or ''

            data.append(row)

        return pd.DataFrame(data)


# =============================================================================
# Confidence Score Calculator
# =============================================================================

class ConfidenceScoreCalculator:
    """Calculates confidence scores based on feature-entitlement associations."""

    def __init__(self, config: Dict[str, Any], rng: Generator,
                 identities_df: pd.DataFrame,
                 accounts_by_app: Dict[str, List[Account]]):
        self.config = config
        self.rng = rng
        self.identities_df = identities_df
        self.accounts_by_app = accounts_by_app
        self.logger = logging.getLogger(self.__class__.__name__)

        self.confidence_cfg = config.get('confidence', {})
        self.features_cfg = config.get('features', {})

    def calculate_scores(self) -> None:
        """Calculate confidence scores for all accounts."""
        self.logger.info("Calculating confidence scores...")

        distribution = self.confidence_cfg.get('distribution', {
            'high': 0.35, 'medium': 0.30, 'low': 0.30, 'none': 0.05
        })
        thresholds = self.confidence_cfg.get('thresholds', {
            'high': {'min': 0.70}, 'medium': {'min': 0.40}, 'low': {'min': 0.01}
        })

        # Get features to use, respecting desired number from config
        mandatory = self.features_cfg.get('mandatory_features', ['manager', 'department', 'jobcode'])
        additional = self.features_cfg.get('additional_features', [])
        desired_num = int(self.features_cfg.get('num_features_for_rules', len(mandatory)))

        if desired_num <= len(mandatory):
            features = mandatory[:desired_num]
        else:
            extra_needed = desired_num - len(mandatory)
            features = mandatory + additional[:max(0, extra_needed)]

        self.logger.info(f"Using features for confidence: {features}")

        # Calculate for each app
        for app_name, accounts in self.accounts_by_app.items():
            self._calculate_for_app(app_name, accounts, distribution, thresholds)

    def _calculate_for_app(self, app_name: str, accounts: List[Account],
                           distribution: Dict[str, float],
                           thresholds: Dict[str, Dict]) -> None:
        """Calculate confidence scores for accounts in one app."""
        num_accounts = len(accounts)
        if num_accounts == 0:
            return

        # Determine how many fall into each bucket
        bucket_counts = {
            'none': int(num_accounts * distribution.get('none', 0.05)),
            'low': int(num_accounts * distribution.get('low', 0.30)),
            'medium': int(num_accounts * distribution.get('medium', 0.30)),
            'high': int(num_accounts * distribution.get('high', 0.35))
        }

        # Adjust for rounding
        total_assigned = sum(bucket_counts.values())
        if total_assigned < num_accounts:
            bucket_counts['high'] += num_accounts - total_assigned

        # Create index mapping
        indices = list(range(num_accounts))
        self.rng.shuffle(indices)

        idx = 0

        # Assign 'none' (no confidence score)
        for _ in range(bucket_counts['none']):
            if idx < len(indices):
                accounts[indices[idx]].confidence_score = None
                accounts[indices[idx]].confidence_bucket = 'None'
                idx += 1

        # Assign 'low'
        low_min = thresholds.get('low', {}).get('min', 0.01)
        med_min = thresholds.get('medium', {}).get('min', 0.40)
        for _ in range(bucket_counts['low']):
            if idx < len(indices):
                score = self.rng.uniform(low_min, med_min - 0.01)
                accounts[indices[idx]].confidence_score = round(score, 3)
                accounts[indices[idx]].confidence_bucket = 'Low'
                idx += 1

        # Assign 'medium'
        high_min = thresholds.get('high', {}).get('min', 0.70)
        for _ in range(bucket_counts['medium']):
            if idx < len(indices):
                score = self.rng.uniform(med_min, high_min - 0.01)
                accounts[indices[idx]].confidence_score = round(score, 3)
                accounts[indices[idx]].confidence_bucket = 'Medium'
                idx += 1

        # Assign 'high'
        for _ in range(bucket_counts['high']):
            if idx < len(indices):
                score = self.rng.uniform(high_min, 0.99)
                accounts[indices[idx]].confidence_score = round(score, 3)
                accounts[indices[idx]].confidence_bucket = 'High'
                idx += 1

        self.logger.info(f"{app_name}: Assigned confidence - None:{bucket_counts['none']}, "
                         f"Low:{bucket_counts['low']}, Medium:{bucket_counts['medium']}, "
                         f"High:{bucket_counts['high']}")


# =============================================================================
# Feature Validator
# =============================================================================

class FeatureValidator:
    """Validates that generated data produces meaningful features."""

    def __init__(self, config: Dict[str, Any], identities_df: pd.DataFrame,
                 accounts_by_app: Dict[str, List[Account]]):
        self.config = config
        self.identities_df = identities_df
        self.accounts_by_app = accounts_by_app
        self.logger = logging.getLogger(self.__class__.__name__)

        self.validation_cfg = config.get('feature_validation', {})
        self.features_cfg = config.get('features', {})

    def validate(self) -> Dict[str, Any]:
        """Run all validation checks and return results."""
        if not self.validation_cfg.get('enabled', True):
            self.logger.info("Feature validation is disabled")
            return {'status': 'skipped'}

        self.logger.info("Running feature validation...")
        results = {
            'status': 'passed',
            'warnings': [],
            'errors': [],
            'feature_stats': {}
        }

        mandatory_features = self.features_cfg.get('mandatory_features', ['manager', 'department', 'jobcode'])
        max_unique = self.validation_cfg.get('max_unique_values', 50)
        cramers_threshold = self.validation_cfg.get('cramers_v_threshold', 0.1)

        # Check cardinality
        for feature in mandatory_features:
            if feature in self.identities_df.columns:
                unique_count = self.identities_df[feature].nunique()
                results['feature_stats'][feature] = {'unique_values': unique_count}

                if unique_count <= 1:
                    results['errors'].append(f"Feature '{feature}' is constant (only {unique_count} unique value)")
                    results['status'] = 'failed'
                elif unique_count > max_unique:
                    results['warnings'].append(f"Feature '{feature}' has high cardinality ({unique_count} > {max_unique})")
            else:
                results['errors'].append(f"Mandatory feature '{feature}' not found in identities")
                results['status'] = 'failed'

        # Check Cramér's V for feature-entitlement associations
        self._check_cramers_v(results, mandatory_features, cramers_threshold)

        # Log results
        if results['errors']:
            self.logger.error(f"Validation FAILED: {results['errors']}")
        if results['warnings']:
            self.logger.warning(f"Validation warnings: {results['warnings']}")
        if results['status'] == 'passed':
            self.logger.info("Feature validation PASSED")

        # Handle fail_on_validation_error
        if results['status'] == 'failed' and self.validation_cfg.get('fail_on_validation_error', False):
            raise ValueError(f"Feature validation failed: {results['errors']}")

        return results

    def _check_cramers_v(self, results: Dict, features: List[str], threshold: float) -> None:
        """Check Cramér's V between features and entitlement presence."""
        if not self.accounts_by_app:
            return

        # Pick first app for validation
        app_name = list(self.accounts_by_app.keys())[0]
        accounts = self.accounts_by_app[app_name]

        # Build a simple target variable: has any entitlements
        user_has_ents = {}
        for account in accounts:
            total_ents = sum(len(grants) for grants in account.entitlement_grants.values())
            user_has_ents[account.user_id] = 'Yes' if total_ents > 0 else 'No'

        # Merge with identities
        df = self.identities_df.copy()
        df['has_entitlements'] = df['user_id'].map(user_has_ents)

        for feature in features:
            if feature not in df.columns:
                continue

            try:
                ct = pd.crosstab(df[feature].fillna('NA'), df['has_entitlements'].fillna('NA'))
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    continue

                chi2, p, dof, expected = chi2_contingency(ct)
                n = len(df)
                r, c = ct.shape
                cramers_v = np.sqrt(chi2 / (n * min(r - 1, c - 1))) if n > 0 else 0

                results['feature_stats'].setdefault(feature, {})['cramers_v'] = round(cramers_v, 4)

                if cramers_v < threshold:
                    results['warnings'].append(
                        f"Feature '{feature}' has low Cramér's V ({cramers_v:.3f} < {threshold})"
                    )

            except Exception as e:
                self.logger.warning(f"Could not calculate Cramér's V for '{feature}': {e}")


# =============================================================================
# Data Writer
# =============================================================================

class DataWriter:
    """Writes generated data to CSV files."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        self.output_dir = Path(config.get('global', {}).get('output_directory', './out'))
        self.quoting = csv.QUOTE_ALL if config.get('global', {}).get('csv_quoting', 'all') == 'all' else csv.QUOTE_MINIMAL

    def setup_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory: {self.output_dir}")

    def write_identities(self, df: pd.DataFrame) -> Path:
        """Write identities to CSV."""
        output_path = self.output_dir / "identities.csv"
        df.to_csv(output_path, index=False, quoting=self.quoting)
        self.logger.info(f"Written {len(df)} identities to {output_path}")
        return output_path

    def write_entitlements(self, df: pd.DataFrame, app_name: str) -> Path:
        """Write entitlements for an app to CSV."""
        output_path = self.output_dir / f"{app_name}_entitlements.csv"
        df.to_csv(output_path, index=False, quoting=self.quoting)
        self.logger.info(f"Written {len(df)} entitlements to {output_path}")
        return output_path

    def write_accounts(self, df: pd.DataFrame, app_name: str) -> Path:
        """Write accounts for an app to CSV."""
        output_path = self.output_dir / f"{app_name}_accounts.csv"
        df.to_csv(output_path, index=False, quoting=self.quoting)
        self.logger.info(f"Written {len(df)} accounts to {output_path}")
        return output_path

    def write_qa_summary(self, summary: Dict[str, Any]) -> Path:
        """Write QA summary report."""
        output_path = self.output_dir / "qa_summary.txt"

        lines = [
            "=" * 60,
            "SYNTHETIC DATA GENERATION - QA SUMMARY",
            "=" * 60,
            f"Generated at: {datetime.now().isoformat()}",
            "",
        ]

        for section, data in summary.items():
            lines.append(f"--- {section} ---")
            if isinstance(data, dict):
                for key, value in data.items():
                    lines.append(f"  {key}: {value}")
            else:
                lines.append(f"  {data}")
            lines.append("")

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        self.logger.info(f"Written QA summary to {output_path}")
        return output_path

    def write_statistics(self, stats: Dict[str, Any]) -> Path:
        """Write generation statistics to JSON."""
        output_path = self.output_dir / "generation_statistics.json"

        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        self.logger.info(f"Written statistics to {output_path}")
        return output_path


# =============================================================================
# QA Reporter
# =============================================================================

class QAReporter:
    """Generates QA summary and statistics."""

    def __init__(self, identities_df: pd.DataFrame,
                 entitlements_by_app: Dict[str, List[Entitlement]],
                 accounts_by_app: Dict[str, List[Account]],
                 validation_results: Dict[str, Any]):
        self.identities_df = identities_df
        self.entitlements_by_app = entitlements_by_app
        self.accounts_by_app = accounts_by_app
        self.validation_results = validation_results
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive QA summary."""
        summary = {}

        # Identity statistics
        summary['Identities'] = {
            'Total count': len(self.identities_df),
            'Unique user_ids': self.identities_df['user_id'].nunique(),
            'With manager': (self.identities_df['manager'] != '').sum(),
            'Without manager': (self.identities_df['manager'] == '').sum(),
            'Managers (is_manager=Y)': (self.identities_df['is_manager'] == 'Y').sum(),
        }

        # Distribution checks
        summary['Distributions'] = {}

        if 'employment_type' in self.identities_df.columns:
            emp_dist = self.identities_df['employment_type'].value_counts(normalize=True).to_dict()
            summary['Distributions']['Employment Type'] = {k: f"{v:.1%}" for k, v in emp_dist.items()}

        if 'identity_type' in self.identities_df.columns:
            id_dist = self.identities_df['identity_type'].value_counts(normalize=True).to_dict()
            summary['Distributions']['Identity Type'] = {k: f"{v:.1%}" for k, v in id_dist.items()}

        if 'status' in self.identities_df.columns:
            status_dist = self.identities_df['status'].value_counts(normalize=True).to_dict()
            summary['Distributions']['Status'] = {k: f"{v:.1%}" for k, v in status_dist.items()}

        # Entitlement statistics
        summary['Entitlements'] = {}
        for app_name, ents in self.entitlements_by_app.items():
            summary['Entitlements'][app_name] = len(ents)

        # Account/Grant statistics
        summary['Grants'] = {}
        total_grants_per_user = defaultdict(int)

        for app_name, accounts in self.accounts_by_app.items():
            app_grants = []
            for account in accounts:
                user_grants = sum(len(grants) for grants in account.entitlement_grants.values())
                app_grants.append(user_grants)
                total_grants_per_user[account.user_id] += user_grants

            if app_grants:
                summary['Grants'][f'{app_name} - Mean'] = f"{np.mean(app_grants):.2f}"
                summary['Grants'][f'{app_name} - Std'] = f"{np.std(app_grants):.2f}"
                summary['Grants'][f'{app_name} - Min'] = min(app_grants)
                summary['Grants'][f'{app_name} - Max'] = max(app_grants)

        # 80/20 check
        users_with_3_plus = sum(1 for count in total_grants_per_user.values() if count >= 3)
        pct_3_plus = users_with_3_plus / len(self.identities_df) if len(self.identities_df) > 0 else 0
        summary['Grants']['Users with 3+ entitlements'] = f"{pct_3_plus:.1%}"

        # Confidence distribution
        summary['Confidence'] = {}
        all_confidence_buckets = []
        for app_name, accounts in self.accounts_by_app.items():
            for account in accounts:
                if account.confidence_bucket:
                    all_confidence_buckets.append(account.confidence_bucket)

        if all_confidence_buckets:
            conf_counts = Counter(all_confidence_buckets)
            total = len(all_confidence_buckets)
            for bucket in ['High', 'Medium', 'Low', 'None']:
                count = conf_counts.get(bucket, 0)
                summary['Confidence'][bucket] = f"{count} ({count/total:.1%})"

        # Validation results
        summary['Feature Validation'] = self.validation_results

        return summary

    def generate_statistics(self) -> Dict[str, Any]:
        """Generate detailed statistics for JSON export."""
        stats = {
            'generated_at': datetime.now().isoformat(),
            'identities': {
                'count': len(self.identities_df),
                'columns': list(self.identities_df.columns),
            },
            'entitlements': {
                app: len(ents) for app, ents in self.entitlements_by_app.items()
            },
            'accounts': {
                app: len(accs) for app, accs in self.accounts_by_app.items()
            },
            'validation': self.validation_results
        }
        return stats


# =============================================================================
# Main Orchestrator
# =============================================================================

class SyntheticDataGenerator:
    """Main orchestrator for synthetic data generation."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Components
        self.config_loader: Optional[ConfigLoader] = None
        self.input_reader: Optional[InputDataReader] = None
        self.identity_generator: Optional[IdentityGenerator] = None
        self.entitlement_generator: Optional[EntitlementGenerator] = None
        self.account_generator: Optional[AccountGenerator] = None
        self.confidence_calculator: Optional[ConfidenceScoreCalculator] = None
        self.feature_validator: Optional[FeatureValidator] = None
        self.data_writer: Optional[DataWriter] = None

        # Generated data
        self.identities: List[Identity] = []
        self.identities_df: Optional[pd.DataFrame] = None
        self.entitlements_by_app: Dict[str, List[Entitlement]] = {}
        self.accounts_by_app: Dict[str, List[Account]] = {}

    def run(self) -> None:
        """Execute the full data generation pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("SYNTHETIC DATA GENERATION - STARTING")
        self.logger.info("=" * 60)

        # Step 1: Load configuration
        self.config_loader = ConfigLoader(self.config_path)
        self.config = self.config_loader.load()

        # Step 2: Initialize RNG and Faker
        seed = self.config.get('global', {}).get('seed', 42)
        self.logger.info(f"Using seed: {seed}")

        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        Faker.seed(seed)
        faker = Faker()

        # Step 3: Initialize components
        self.input_reader = InputDataReader()
        self.data_writer = DataWriter(self.config)
        self.data_writer.setup_output_dir()

        # Step 4: Generate identities
        self.identity_generator = IdentityGenerator(self.config, rng, faker, self.input_reader)
        num_identities = self.config.get('identity', {}).get('num_identities', 10000)
        self.identities = self.identity_generator.generate(num_identities)
        self.identities_df = self.identity_generator.to_dataframe()

        # Step 5: Generate entitlements
        self.entitlement_generator = EntitlementGenerator(self.config, rng, faker, self.input_reader)
        self.entitlements_by_app = self.entitlement_generator.generate_all()

        # Step 6: Generate accounts/grants
        self.account_generator = AccountGenerator(
            self.config, rng, self.identities, self.entitlements_by_app
        )
        self.accounts_by_app = self.account_generator.generate_all()

        # Step 7: Calculate confidence scores
        self.confidence_calculator = ConfidenceScoreCalculator(
            self.config, rng, self.identities_df, self.accounts_by_app
        )
        self.confidence_calculator.calculate_scores()

        # Step 8: Validate features
        self.feature_validator = FeatureValidator(
            self.config, self.identities_df, self.accounts_by_app
        )
        validation_results = self.feature_validator.validate()

        # Step 9: Write output files
        self._write_outputs(validation_results)

        self.logger.info("=" * 60)
        self.logger.info("SYNTHETIC DATA GENERATION - COMPLETE")
        self.logger.info("=" * 60)

    def _write_outputs(self, validation_results: Dict[str, Any]) -> None:
        """Write all output files."""
        # Write identities
        self.data_writer.write_identities(self.identities_df)

        # Write entitlements and accounts per app
        for app_name in self.entitlements_by_app.keys():
            ent_df = self.entitlement_generator.to_dataframe(app_name)
            self.data_writer.write_entitlements(ent_df, app_name)

            acc_df = self.account_generator.to_dataframe(app_name)
            self.data_writer.write_accounts(acc_df, app_name)

        # Generate and write QA summary
        qa_reporter = QAReporter(
            self.identities_df,
            self.entitlements_by_app,
            self.accounts_by_app,
            validation_results
        )

        if self.config.get('output', {}).get('generate_qa_summary', True):
            summary = qa_reporter.generate_summary()
            self.data_writer.write_qa_summary(summary)

        if self.config.get('output', {}).get('generate_statistics', True):
            stats = qa_reporter.generate_statistics()
            self.data_writer.write_statistics(stats)


# =============================================================================
# CLI Entry Point
# =============================================================================

def setup_logging(level: str = 'INFO') -> None:
    """Configure logging."""
    log_format = "%(asctime)s - %(name)-25s - %(levelname)-8s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format
    )
    # Suppress verbose logs from faker
    logging.getLogger("faker").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Synthetic IGA Data Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data_generator_config.json"),
        help="Path to configuration JSON file"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level)

    try:
        generator = SyntheticDataGenerator(args.config)
        generator.run()
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()