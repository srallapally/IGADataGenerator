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
    12/21/25 Initial vers
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
from AssociationRuleEngine import RuleEngine, AssociationRule
from typing import Any, Dict, List, Optional, Set, Tuple

from dynamic_rule_generator_cross_app import CrossAppRuleGenerationOrchestrator

# BEGIN PATCH 1: Import dynamic rule generator
try:
    from dynamic_rule_generator import (
        RuleGeneratorConfig,
        RuleGenerationOrchestrator
    )

    DYNAMIC_RULES_AVAILABLE = True
except ImportError:
    DYNAMIC_RULES_AVAILABLE = False
    import warnings

    warnings.warn("dynamic_rule_generator not available. Dynamic rule generation will be disabled.")
# END PATCH 1

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
# BEGIN FIX: RuleQuota dataclass for deterministic quota-based assignment
# =============================================================================

@dataclass
class RuleQuota:
    """
    Tracks quota for a single rule in deterministic assignment.

    Used to ensure mined confidence matches target confidence by selecting
    exactly freqUnion_target users from those matching the antecedent.
    """
    rule_id: str
    app_name: str
    antecedent_markers: Set[str]
    consequent_entitlements: List[str]
    confidence: float
    freq: int  # Users matching antecedent
    freqUnion_target: int  # Users who should get entitlements
    assigned_users: Set[str] = field(default_factory=set)  # Users already assigned
    matching_users: List[str] = field(default_factory=list)  # Users matching antecedent


# =============================================================================
# END FIX: RuleQuota dataclass
# =============================================================================


# =============================================================================
# Configuration Loader
# =============================================================================

class ConfigLoader:
    """Loads and validates the configuration file."""

    REQUIRED_APPS = {"AWS", "Salesforce", "ServiceNow", "SAP"}

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
        self._expand_applications()
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

    def _expand_applications(self) -> None:
        """
        Ensure the applications list matches applications.num_apps.

        If num_apps > len(apps), auto-generate additional apps using
        additional_app_pool and additional_app_defaults.
        """
        apps_cfg = self.config.get('applications', {})
        num_apps = apps_cfg.get('num_apps', 4)
        apps_list = apps_cfg.get('apps', [])

        current_count = len(apps_list)
        if num_apps <= current_count:
            self.logger.info(f"Applications already configured: {current_count} (num_apps={num_apps})")
            return

        additional_needed = num_apps - current_count
        self.logger.info(f"Generating {additional_needed} additional applications to reach num_apps={num_apps}")

        # Pool of candidate names for auto-generated apps
        app_pool = apps_cfg.get('additional_app_pool', [])
        existing_names = {app.get('app_name') for app in apps_list}
        available_names = [name for name in app_pool if name not in existing_names]

        # Defaults for additional apps
        defaults = apps_cfg.get('additional_app_defaults', {})
        ent_range = defaults.get('entitlements_range', [30, 80])
        crit_dist = defaults.get('criticality_distribution', {
            "High": 0.15,
            "Medium": 0.50,
            "Low": 0.35,
        })

        for i in range(additional_needed):
            if i < len(available_names):
                app_name = available_names[i]
            else:
                # Fallback name if pool is exhausted
                app_name = f"App_{current_count + i + 1}"

            # Choose a reasonable entitlement count (midpoint of range)
            if isinstance(ent_range, list) and len(ent_range) == 2:
                min_ents, max_ents = ent_range
                num_ents = int((min_ents + max_ents) / 2)
            else:
                num_ents = 50

            app_id = f"APP_{app_name.upper().replace(' ', '_').replace('-', '_')}"

            new_app = {
                "app_name": app_name,
                "app_id": app_id,
                "enabled": True,
                "num_entitlements": num_ents,
                # No input_file(s); entitlements will be synthetic via EntitlementGenerator
                "criticality_distribution": crit_dist
            }

            apps_list.append(new_app)
            self.logger.info(f"  Added auto-generated app: {app_name} ({app_id}), num_entitlements={num_ents}")

        # Persist back to config
        self.config["applications"]["apps"] = apps_list
        self.logger.info(f"Total applications configured after expansion: {len(apps_list)}")


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
    create_from_rule: bool = False


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
    LOCATION_COUNTRY_DIST = {
        'US': 0.60,
        'GB': 0.15,
        'IN': 0.15,
        'DE': 0.05,
        'AU': 0.05
    }

    LOCATION_SITES_BY_COUNTRY = {
        'US': {'SFO': 0.3, 'NYC': 0.25, 'AUS': 0.15, 'CHI': 0.1, 'SEA': 0.1, 'BOS': 0.1},
        'GB': {'LON': 0.7, 'MAN': 0.2, 'EDI': 0.1},
        'IN': {'BLR': 0.5, 'MUM': 0.3, 'HYD': 0.2},
        'DE': {'BER': 0.6, 'MUN': 0.4},
        'AU': {'SYD': 0.7, 'MEL': 0.3}
    }
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
    JOB_LEVEL_DISTRIBUTION = {
        'Junior': 0.20,
        'Mid': 0.30,
        'Senior': 0.20,
        'Lead': 0.10,
        'Manager': 0.10,
        'Director': 0.05,
        'VP': 0.03,
        'Executive': 0.02
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
        # Allow config overrides for distributions
        identity_cfg = config.get('identity', {})

        # Override location distribution if provided
        custom_countries = identity_cfg.get('location_country_distribution')
        if custom_countries:
            self.location_country_dist = custom_countries
        else:
            self.location_country_dist = self.LOCATION_COUNTRY_DIST

        # Override job level distribution if provided
        custom_job_levels = identity_cfg.get('job_level_distribution')
        if custom_job_levels:
            self.job_level_dist = custom_job_levels
        else:
            self.job_level_dist = self.JOB_LEVEL_DISTRIBUTION

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
        country_dist = self.LOCATION_COUNTRY_DIST
        site_by_country = self.LOCATION_SITES_BY_COUNTRY

        # First pass: generate basic identities
        # BEGIN FIX: Use correct starting index to avoid duplicate user_ids
        # When called from _generate_random_identities, self.identities may already
        # have pattern-based identities. We need to start from the next available ID.
        starting_index = len(self.identities)
        # END FIX

        for i in range(num_identities):
            # BEGIN FIX: Offset by starting_index to avoid ID collisions
            user_id = f"U{starting_index + i + 1:07d}"
            employee_id = f"EMP{starting_index + i + 1:06d}"
            # END FIX

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

    def generate_rule_aware(self, num_identities: int, rule_engine: RuleEngine) -> List[Identity]:
        """
        Generate identities that conform to rule antecedents.

        This creates a population where rule patterns have the target support levels,
        ensuring meaningful statistical associations without overfitting.

        Args:
            num_identities: Total number of identities to generate
            rule_engine: RuleEngine containing the rules to satisfy

        Returns:
            List of Identity objects designed to satisfy rule patterns
        """
        self.logger.info("=" * 60)
        self.logger.info("RULE-AWARE IDENTITY GENERATION (Phase 2)")
        self.logger.info("=" * 60)

        # Step 1: Extract rule patterns across all apps
        rule_patterns = self._extract_rule_patterns(rule_engine)
        self.logger.info(f"Extracted {len(rule_patterns)} unique rule patterns")

        # Step 2: Calculate how many identities should match each pattern
        pattern_quotas = self._calculate_pattern_quotas(rule_patterns, num_identities)
        self.logger.info(f"Calculated quotas for {len(pattern_quotas)} patterns")

        # Step 3: Generate identities for each pattern
        identities = []
        for pattern, quota in pattern_quotas.items():
            if quota > 0:
                pattern_identities = self._generate_identities_for_pattern(pattern, quota)
                identities.extend(pattern_identities)
                self.logger.debug(f"Generated {len(pattern_identities)} identities for pattern: {pattern}")

        self.logger.info(f"Generated {len(identities)} rule-based identities")

        # Step 4: Fill remaining slots with random identities
        remaining = num_identities - len(identities)
        if remaining > 0:
            self.logger.info(f"Filling {remaining} remaining slots with random identities")
            random_identities = self._generate_random_identities(remaining)
            identities.extend(random_identities)

        # Step 5: Shuffle to avoid clustering
        self.rng.shuffle(identities)

        # Step 6: Assign managers (second pass)
        pct_no_manager = self.config.get('pct_users_without_manager', 0.10)
        self._assign_managers(pct_no_manager)

        self.logger.info(f"✓ Generated {len(identities)} rule-aware identities")
        return identities

    def _extract_rule_patterns(self, rule_engine: RuleEngine) -> List[Dict[str, any]]:
        """
        Extract all unique rule patterns from the rule engine.

        Returns list of patterns with metadata:
        [
            {
                'pattern': {'department': 'Engineering', 'job_level': 'Senior'},
                'apps': ['AWS', 'GitHub'],
                'target_support': 0.06,
                'confidence': 0.85
            },
            ...
        ]
        """
        patterns = []
        seen_patterns = set()

        # Iterate through all apps and their rules
        for app_name in rule_engine.rules_by_app.keys():
            rules = rule_engine.get_rules_for_app(app_name)

            for rule in rules:
                # Parse pattern from antecedent markers
                pattern = self._parse_antecedent_to_pattern(rule.antecedent_entitlements)

                if not pattern:
                    continue

                # Create a hashable key
                pattern_key = tuple(sorted(pattern.items()))

                if pattern_key not in seen_patterns:
                    seen_patterns.add(pattern_key)
                    patterns.append({
                        'pattern': pattern,
                        'apps': [app_name],
                        'target_support': rule.support,
                        'confidence': rule.confidence,
                        'rule_ids': [rule.id]
                    })
                else:
                    # Pattern already exists, add this app to it
                    for p in patterns:
                        if tuple(sorted(p['pattern'].items())) == pattern_key:
                            if app_name not in p['apps']:
                                p['apps'].append(app_name)
                            p['rule_ids'].append(rule.id)
                            # Use maximum support/confidence across rules
                            p['target_support'] = max(p['target_support'], rule.support)
                            p['confidence'] = max(p['confidence'], rule.confidence)
                            break

        return patterns

    def _parse_antecedent_to_pattern(self, antecedent_entitlements: List[str]) -> Dict[str, str]:
        """
        Parse antecedent markers into a feature pattern.

        Cross-app rules use markers like: 'FEATURE:department=Engineering'
        Per-app rules might use actual entitlements (skip these)

        Returns:
            Dict mapping feature names to values
        """
        pattern = {}

        for marker in antecedent_entitlements:
            if isinstance(marker, str) and marker.startswith('FEATURE:'):
                # Parse "FEATURE:feature_name=value"
                parts = marker[8:].split('=', 1)  # Remove "FEATURE:" prefix
                if len(parts) == 2:
                    feature_name, value = parts
                    pattern[feature_name] = value

        return pattern

    def _calculate_pattern_quotas(self,
                                  rule_patterns: List[Dict[str, any]],
                                  num_identities: int) -> Dict[Tuple, int]:
        """
        Calculate how many identities should match each pattern.

        Uses target support levels to determine quotas, with adjustments
        to ensure we don't exceed total population.

        TUNED FIX: Weight allocation by (support × √confidence) to balance
        the distribution. Linear weighting was too aggressive, causing 58% high
        instead of 35%. Square root dampens the effect while still biasing
        toward high-confidence rules.

        Returns:
            Dict mapping pattern tuples to identity counts
        """
        # BEGIN TUNED FIX: Use square root of confidence for balanced weighting
        quotas = {}
        total_allocated = 0

        # Calculate weights: support × √confidence
        # This gives modest preference to high-confidence rules without dominating
        weighted_patterns = []
        total_weight = 0.0

        for pattern_info in rule_patterns:
            # Square root dampens the confidence effect
            # conf=0.90 → √0.90 = 0.95 (small boost)
            # conf=0.50 → √0.50 = 0.71 (moderate)
            # conf=0.20 → √0.20 = 0.45 (not too penalized)
            confidence_weight = pattern_info['confidence'] ** 0.5
            weight = pattern_info['target_support'] * confidence_weight
            total_weight += weight
            weighted_patterns.append({
                **pattern_info,
                'weight': weight,
                'confidence_weight': confidence_weight
            })

        # Sort by weight (descending) to allocate high-weight patterns first
        sorted_patterns = sorted(weighted_patterns, key=lambda p: p['weight'], reverse=True)

        # Allocate proportionally to weights
        budget = int(num_identities * 0.95)  # Reserve 5% for random

        for pattern_info in sorted_patterns:
            pattern = pattern_info['pattern']
            weight = pattern_info['weight']

            # Proportional allocation based on weight
            if total_weight > 0:
                target_count = int(budget * (weight / total_weight))
            else:
                target_count = 0

            # Ensure minimum viable population
            if target_count < 10:  # Increased from 5 for better statistics
                target_count = 10

            # Don't exceed remaining budget
            remaining = budget - total_allocated
            if target_count > remaining:
                target_count = remaining

            if target_count > 0:
                pattern_key = tuple(sorted(pattern.items()))
                quotas[pattern_key] = target_count
                total_allocated += target_count

                self.logger.debug(
                    f"Pattern {pattern}: allocated {target_count} users "
                    f"(conf={pattern_info['confidence']:.3f}, "
                    f"√conf={pattern_info['confidence_weight']:.3f}, "
                    f"support={pattern_info['target_support']:.3f}, "
                    f"weight={weight:.4f})"
                )

            # Stop if budget exhausted
            if total_allocated >= budget:
                break

        self.logger.info(
            f"Allocated {total_allocated}/{num_identities} identities to "
            f"{len(quotas)} patterns (weighted by support × √confidence)"
        )
        return quotas
        # END TUNED FIX

        self.logger.info(
            f"Allocated {total_allocated}/{num_identities} identities to "
            f"{len(quotas)} patterns (weighted by confidence×support)"
        )
        return quotas
        # END FIX

    def _generate_identities_for_pattern(self,
                                         pattern: Tuple[Tuple[str, str], ...],
                                         count: int) -> List[Identity]:
        """
        Generate identities that match a specific feature pattern.

        Args:
            pattern: Tuple of (feature, value) pairs
            count: Number of identities to generate

        Returns:
            List of Identity objects matching the pattern
        """
        pattern_dict = dict(pattern)
        identities = []

        for i in range(count):
            identity = self._generate_single_identity_with_pattern(pattern_dict)
            identities.append(identity)

        return identities

    def _generate_single_identity_with_pattern(self, pattern: Dict[str, str]) -> Identity:
        """
        Generate a single identity that matches the given feature pattern.

        Args:
            pattern: Dict of feature constraints, e.g. {'department': 'Engineering', 'job_level': 'Senior'}

        Returns:
            Identity object with specified features
        """
        # Generate base identity attributes
        user_id = f"U{len(self.identities) + 1:07d}"
        employee_id = f"EMP{len(self.identities) + 1:06d}"

        first_name = self.faker.first_name()
        last_name = self.faker.last_name()
        user_name = self._generate_username(first_name, last_name)
        email = f"{first_name.lower()}.{last_name.lower()}@company.com"

        # Apply pattern constraints
        department, dept_data = self._select_department_matching_pattern(pattern)
        business_unit = dept_data.get('Industry', pattern.get('business_unit', 'General'))
        department_type = dept_data.get('Type', pattern.get('department_type', 'Business'))
        department_id = dept_data.get('department_id', f'DEPT_{department.upper()[:10]}')

        # Select job matching pattern
        jobcode, job_data = self._select_job_matching_pattern(pattern, business_unit)
        job_category = job_data.get('Category', 'Business')
        job_level = pattern.get('job_level', self._infer_job_level(jobcode))

        # Apply other pattern constraints
        employment_type = pattern.get('employment_type',
                                      self._weighted_choice(self.config.get('distribution_employee_contractor',
                                                                            {'Employee': 0.9, 'Contractor': 0.1})))

        identity_type = pattern.get('identity_type',
                                    self._weighted_choice(self.config.get('distribution_human_ai_agent',
                                                                          {'Human': 0.95, 'AI_Agent': 0.05})))

        status = pattern.get('status',
                             self._weighted_choice(self.config.get('distribution_active_inactive',
                                                                   {'Active': 0.93, 'Inactive': 0.07})))

        # Location
        location_country, location_site = self._select_location_matching_pattern(pattern)

        # Tenure
        tenure = self._generate_tenure()
        hire_date = (datetime.now() - timedelta(days=int(tenure * 365))).strftime('%Y-%m-%d')

        # Cost center
        cost_center = f"CC_{department_id[:10]}_{self.rng.integers(100, 999)}"

        # Create identity
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
            manager=None,  # Assigned in second pass
            manager_name=None,
            location_country=location_country,
            location_site=location_site,
            employment_type=employment_type,
            identity_type=identity_type,
            status=status,
            cost_center=cost_center,
            tenure_years=f"{tenure:.1f}",
            hire_date=hire_date,
            is_manager='N',  # Assigned in second pass
            employee_id=employee_id
        )

        self.identities.append(identity)
        return identity

    def _select_department_matching_pattern(self, pattern: Dict[str, str]) -> Tuple[str, Dict]:
        """
        Select a department that matches pattern constraints.

        Returns:
            (department_name, department_data)
        """
        if 'department' in pattern:
            # Find exact match
            target_dept = pattern['department']
            for dept in self.departments:
                if dept.get('department_name') == target_dept:
                    return target_dept, dept
            # If not found, return the pattern value with synthetic data
            return target_dept, {
                'department_id': f'DEPT_{target_dept.upper()[:10]}',
                'department_name': target_dept,
                'Industry': pattern.get('business_unit', 'General'),
                'Type': pattern.get('department_type', 'Business')
            }

        if 'business_unit' in pattern:
            # Find department in matching business unit
            target_bu = pattern['business_unit']
            matching_depts = [d for d in self.departments if d.get('Industry') == target_bu]
            if matching_depts:
                dept = self.rng.choice(matching_depts)
                return dept.get('department_name', 'General'), dept

        # Random selection
        dept = self._select_department()
        return dept.get('department_name', 'General'), dept

    def _select_job_matching_pattern(self, pattern: Dict[str, str], business_unit: str) -> Tuple[str, Dict]:
        """
        Select a job title that matches pattern constraints.

        Returns:
            (job_title, job_data)
        """
        if 'jobcode' in pattern:
            # Use exact job from pattern
            target_job = pattern['jobcode']
            for job in self.jobtitles:
                if job.get('Job Title') == target_job:
                    return target_job, job
            # If not found, return pattern value
            return target_job, {'Job Title': target_job, 'Category': pattern.get('job_category', 'Business')}

        if 'job_level' in pattern:
            # Filter jobs by level keywords
            target_level = pattern['job_level']
            level_keywords = self.JOB_LEVEL_KEYWORDS.get(target_level, [])

            matching_jobs = []
            for job in self.jobtitles:
                job_title = job.get('Job Title', '')
                for keyword in level_keywords:
                    if keyword.upper() in job_title.upper():
                        matching_jobs.append(job)
                        break

            if matching_jobs:
                job = self.rng.choice(matching_jobs)
                return job.get('Job Title', 'Specialist'), job

        # Random selection for business unit
        job = self._select_job_for_department(business_unit)
        return job.get('Job Title', 'Specialist'), job

    def _select_location_matching_pattern(self, pattern: Dict[str, str]) -> Tuple[str, str]:
        """
        Select location matching pattern constraints.

        Returns:
            (country, site)
        """
        country_dist = self.LOCATION_COUNTRY_DIST
        site_by_country = self.LOCATION_SITES_BY_COUNTRY

        if 'location_country' in pattern:
            country = pattern['location_country']
        else:
            country = self._weighted_choice(country_dist)

        if 'location_site' in pattern:
            site = pattern['location_site']
        else:
            site_options = site_by_country.get(country, {'HQ': 1.0})
            site = self._weighted_choice(site_options)

        return country, site

    def _generate_random_identities(self, count: int) -> List[Identity]:
        """
        Generate random identities (no pattern constraints).

        This fills in the remaining population after rule-based generation.
        """
        return self.generate(count)

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
        else:
            num_ents = app_config.get('num_entitlements', 50)
            self.logger.info(f"No input files for {app_name}, generating {num_ents} synthetic entitlements")

            for i in range(num_ents):
                ent_id = f"{app_id}_E{self.ent_counter:07d}"
                ent_name = f"{app_name}_Entitlement_{i + 1}"

                criticality = self._weighted_choice(criticality_dist)
                entitlements.append(Entitlement(
                    entitlement_id=ent_id,
                    entitlement_name=ent_name,
                    app_id=app_id,
                    app_name=app_name,
                    entitlement_type='standard',
                    criticality=criticality,
                    description=f"{app_name} synthetic entitlement {i + 1}"
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
                 entitlements_by_app: Dict[str, List[Entitlement]],
                 rule_engine: Optional[RuleEngine] = None):
        self.config = config
        self.rng = rng
        self.identities = identities
        self.entitlements_by_app = entitlements_by_app
        self.logger = logging.getLogger(self.__class__.__name__)

        self.accounts_by_app: Dict[str, List[Account]] = {}
        self.delimiter = config.get('global', {}).get('multi_value_delimiter', '#')
        self.rule_engine = rule_engine
        # Internal: keep rules used per (app_name, user_id) if needed
        self._rules_used: Dict[Tuple[str, str], List[AssociationRule]] = {}
        # BEGIN FIX: Pre-determine which users follow rules (for cross-app atomicity)
        self.rule_following_users: Set[str] = set()
        self._initialize_rule_following_users()

        # Load confidence thresholds from config for bucket classification
        confidence_cfg = config.get('confidence', {})
        thresholds = confidence_cfg.get('thresholds', {})
        self.high_threshold = thresholds.get('high', {}).get('min', 0.70)
        self.medium_threshold = thresholds.get('medium', {}).get('min', 0.40)
        self.low_threshold = thresholds.get('low', {}).get('min', 0.01)

    # =========================================================================
    # BEGIN FIX: Deduplication method to ensure unique accounts per app
    # =========================================================================
    def _deduplicate_accounts(self, accounts: List[Account], app_name: str) -> List[Account]:
        """
        Remove duplicate accounts based on user_id and user_name.

        Each user should have exactly one account per app. If duplicates exist,
        keep the first occurrence (which typically has rule-based assignments).

        Args:
            accounts: List of Account objects (may contain duplicates)
            app_name: Application name for logging

        Returns:
            Deduplicated list of Account objects
        """
        seen_user_ids = set()
        seen_user_names = set()
        deduplicated = []
        duplicates_removed = 0

        for account in accounts:
            # Check both user_id and user_name for uniqueness
            if account.user_id in seen_user_ids or account.user_name in seen_user_names:
                duplicates_removed += 1
                continue

            seen_user_ids.add(account.user_id)
            seen_user_names.add(account.user_name)
            deduplicated.append(account)

        if duplicates_removed > 0:
            self.logger.warning(
                f"{app_name}: Removed {duplicates_removed} duplicate accounts. "
                f"Reduced from {len(accounts)} to {len(deduplicated)} accounts."
            )

        return deduplicated

    # =========================================================================
    # END FIX: Deduplication method
    # =========================================================================

    def _bucket_from_score(self, score: Optional[float]) -> str:
        """
        Map numeric confidence score to bucket using config-based thresholds.
        Uses confidence.thresholds from config instead of hardcoded values.
        """
        if score is None:
            return "None"
        if score >= self.high_threshold:
            return "High"
        if score >= self.medium_threshold:
            return "Medium"
        if score >= self.low_threshold:
            return "Low"
        return "Low"

    def _apply_rules_probabilistically(
            self,
            app_name: str,
            user_identity: Dict[str, str],
            identity: 'Identity'
    ) -> Tuple[List[str], Optional[float], Optional[str]]:
        """
        Apply rules probabilistically based on their confidence values.

        This implements the production analytics methodology in reverse:
        - Production: mines data to discover confidence = freqUnion / freq
        - Synthetic: uses target confidence as firing probability

        For rule [Department=Finance] → SAP_FI_001 with confidence=0.85:
        - If user is in Finance, grant entitlement with probability 0.85
        - Result: mined confidence ≈ 0.85 (freqUnion/freq)

        Args:
            app_name: Application name
            user_identity: User's attributes for matching
            identity: User identity object

        Returns:
            (entitlements, confidence_score, confidence_bucket)
        """
        # Get all rules for this app
        rules = self.rule_engine.get_rules_for_app(app_name)

        if not rules:
            return [], None, None

        # Build user feature markers for matching
        user_feature_markers = {
            f"FEATURE:{k}={v}"
            for k, v in user_identity.items()
            if v is not None
        }

        # Track which rules match and fire
        granted_entitlements = set()
        matching_rules = []
        fired_rules = []

        for rule in rules:
            # Check if rule matches user's attributes
            antecedent = set(rule.antecedent_entitlements)

            # Skip if empty antecedent
            if not antecedent:
                continue

            # Check if this is a feature-based rule
            if any(a.startswith("FEATURE:") for a in antecedent):
                # Feature-based rule: match against user identity
                if not antecedent.issubset(user_feature_markers):
                    continue  # Rule doesn't match user
            else:
                # Entitlement-based rule: skip for now (we don't have seed entitlements)
                continue

            # Rule matches! Track it
            matching_rules.append(rule)

            # Apply rule probabilistically using its confidence as firing probability
            # This is the KEY to ensuring mined confidence matches target confidence
            if self.rng.random() < rule.confidence:
                # Rule fires! Grant the consequent entitlements
                granted_entitlements.update(rule.consequent_entitlements)
                fired_rules.append(rule)

        # Calculate confidence score and bucket
        # For rule-based users, we use the average confidence of fired rules
        if fired_rules:
            avg_confidence = sum(r.confidence for r in fired_rules) / len(fired_rules)
            score = avg_confidence
            bucket = self._bucket_from_score(score)  # Use config-based thresholds
        elif matching_rules:
            # BEGIN FIX: User matched rules but none fired due to probabilistic selection
            # The probabilistic firing (line 1395) already handles confidence distribution
            # by using rule.confidence as firing probability
            # Don't penalize with 0.3 multiplier - use rule confidence directly
            avg_match_confidence = sum(r.confidence for r in matching_rules) / len(matching_rules)
            score = avg_match_confidence  # Removed 0.3 penalty - use rule confidence as-is
            bucket = self._bucket_from_score(score)  # Use config-based thresholds
            # END FIX
        else:
            # No rules matched at all
            score = None
            bucket = None

        return sorted(list(granted_entitlements)), score, bucket

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
        """
        Generate accounts for a single application using deterministic quota-based assignment.

        This replaces the probabilistic approach with quota-based assignment that
        correctly produces the target mined confidence distribution.

        The key formula is: mined_confidence = freqUnion / freq

        To achieve target confidence, we deterministically select exactly
        freqUnion_target = confidence × freq users to receive each entitlement.
        """
        app_name = app_config['app_name']
        app_id = app_config.get('app_id', f"APP_{app_name.upper()}")
        self.logger.info(f"Generating accounts for {app_name} (deterministic quota-based)...")

        entitlements = self.entitlements_by_app.get(app_name, [])
        if not entitlements:
            self.logger.warning(f"No entitlements found for {app_name}")
            return []

        ent_ids = [e.entitlement_id for e in entitlements]
        accounts = []

        # Handle Epic special case
        if app_name == "Epic":
            accounts = self._generate_epic_accounts(app_config, app_id)
            self.accounts_by_app[app_name] = accounts
            self.logger.info(f"Generated {len(accounts)} accounts for {app_name}")
            return accounts

        # =================================================================
        # BEGIN FIX: Deterministic Quota-Based Assignment
        # =================================================================

        # Step 1: Compute rule quotas
        quotas = []
        if self.rule_engine and self.rule_engine.has_rules(app_name):
            quotas = self._compute_rule_quotas(app_name)

        # Step 2: Assign entitlements based on quotas
        user_assignments = {}
        if quotas:
            user_assignments = self._assign_entitlements_by_quota(app_name, quotas)
            self.logger.info(
                f"{app_name}: Quota-based assignment completed for {len(user_assignments)} users"
            )

        # =================================================================
        # BEGIN FIX: Exclude rule consequent entitlements from random pool
        # =================================================================
        # Problem: If non-rule users randomly get the same entitlements that rules
        # are assigning, it dilutes the mined confidence because those users don't
        # match the rule's feature pattern.
        #
        # Example: Rule says "dept=Finance -> Ent_X with 85% confidence"
        # - 1000 Finance users, 850 should get Ent_X (85%)
        # - If 200 random non-Finance users also get Ent_X
        # - Total with Ent_X = 1050, but mined confidence for Finance is still 850/1000 = 85%
        # - HOWEVER, if the random users ARE Finance users who weren't selected,
        #   then mined confidence = (850+N)/1000 where N is Finance users who got it randomly
        #
        # Solution: Non-rule users should NOT receive entitlements that are rule consequents.
        # This ensures only the quota-selected users have those entitlements for each pattern.

        rule_consequent_ents = set()
        if quotas:
            for quota in quotas:
                rule_consequent_ents.update(quota.consequent_entitlements)
            self.logger.info(
                f"{app_name}: Reserving {len(rule_consequent_ents)} entitlements for rule-based assignment only"
            )

        # Create a filtered pool for random assignment (excludes rule consequents)
        random_pool_ent_ids = [e for e in ent_ids if e not in rule_consequent_ents]

        if len(random_pool_ent_ids) < len(ent_ids):
            self.logger.info(
                f"{app_name}: Random assignment pool reduced from {len(ent_ids)} to {len(random_pool_ent_ids)} entitlements"
            )
        # =================================================================
        # END FIX: Exclude rule consequent entitlements from random pool
        # =================================================================

        # Step 3: Generate accounts for all users
        for identity in self.identities:
            user_id = identity.user_id

            # Check if user is in rule-following population
            use_rules = (
                    self.rule_engine is not None
                    and self.rule_engine.has_rules(app_name)
                    and user_id in self.rule_following_users
            )

            if use_rules and user_id in user_assignments:
                # User has rule-based assignments from quota system
                assignment = user_assignments[user_id]
                final_ents = sorted(list(assignment['entitlements']))

                # Calculate confidence based on which rules applied
                score, bucket = self._calculate_user_confidence(
                    assignment['rules_applied'],
                    assignment['rules_matched']
                )

                # Fallback: if no entitlements from rules, give minimal random
                # BEGIN FIX: Use random pool (excludes rule consequents)
                if not final_ents and random_pool_ent_ids:
                    num_grants = max(1, self._get_grant_count(app_config) // 3)
                    final_ents = self.rng.choice(
                        random_pool_ent_ids,
                        size=min(num_grants, len(random_pool_ent_ids)),
                        replace=False
                    ).tolist()
                # END FIX

            elif use_rules:
                # User is in rule-following population but no rules matched
                # Give them random entitlements with no confidence score
                # BEGIN FIX: Use random pool (excludes rule consequents)
                num_grants = self._get_grant_count(app_config)
                if num_grants == 0 and random_pool_ent_ids:
                    num_grants = 1

                if num_grants > 0 and random_pool_ent_ids:
                    final_ents = self.rng.choice(
                        random_pool_ent_ids,
                        size=min(num_grants, len(random_pool_ent_ids)),
                        replace=False
                    ).tolist()
                else:
                    final_ents = []
                # END FIX

                score = None
                bucket = None

            else:
                # Non-rule user: random assignment
                # BEGIN FIX: Use random pool (excludes rule consequents)
                num_grants = self._get_grant_count(app_config)
                if num_grants == 0 and random_pool_ent_ids:
                    num_grants = 1

                if num_grants > 0 and random_pool_ent_ids:
                    final_ents = self.rng.choice(
                        random_pool_ent_ids,
                        size=min(num_grants, len(random_pool_ent_ids)),
                        replace=False
                    ).tolist()
                else:
                    final_ents = []
                # END FIX

                score = None
                bucket = None

            accounts.append(Account(
                user_id=identity.user_id,
                user_name=identity.user_name,
                app_id=app_id,
                app_name=app_name,
                entitlement_grants={'entitlement_grants': final_ents},
                confidence_score=score,
                confidence_bucket=bucket
            ))

        # =================================================================
        # END FIX: Deterministic Quota-Based Assignment
        # =================================================================

        # =================================================================
        # BEGIN FIX: Deduplicate accounts to ensure uniqueness
        # =================================================================
        accounts = self._deduplicate_accounts(accounts, app_name)
        # =================================================================
        # END FIX: Deduplicate accounts
        # =================================================================

        self.accounts_by_app[app_name] = accounts

        # Log statistics
        rule_users = sum(1 for a in accounts if a.confidence_score is not None)
        self.logger.info(
            f"{app_name}: Generated {len(accounts)} accounts "
            f"({rule_users} with rule-based confidence)"
        )

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

        # BEGIN FIX: Deduplicate Epic accounts
        accounts = self._deduplicate_accounts(accounts, 'Epic')
        # END FIX

        return accounts

    def generate_all(self) -> Dict[str, List[Account]]:
        """Generate accounts for all configured applications."""
        apps_cfg = self.config.get('applications', {})
        app_list = apps_cfg.get('apps', [])

        for app_config in app_list:
            if app_config.get('enabled', True):
                self.generate_for_app(app_config)

        self._enforce_per_app_grant_distribution()

        # Enforce 80/20 rule across all apps
        self._enforce_grant_distribution()

        # Ensure mandatory apps have at least one account with grants
        self._ensure_mandatory_app_grants()

        # Ensure all entitlements of all mandatory apps are used at least once
        self._ensure_mandatory_entitlement_coverage()

        # BEGIN NEW: Final safeguard - ensure NO user has zero entitlements across ALL apps
        self._ensure_no_users_without_entitlements()
        # END NEW

        return self.accounts_by_app

    def _enforce_per_app_grant_distribution(self) -> None:
        """Ensure target percentage of users have at least 3 entitlements PER APPLICATION."""
        grants_cfg = self.config.get('grants', {})
        target_pct_per_app = grants_cfg.get('pct_users_with_3_plus_per_app')

        # If not configured or set to null, skip per-app enforcement
        if target_pct_per_app is None:
            self.logger.info("Per-app entitlement distribution enforcement disabled")
            return

        target_pct_per_app = float(target_pct_per_app)
        self.logger.info(f"Enforcing {target_pct_per_app:.0%} users with 3+ entitlements per app...")

        for app_name, accounts in self.accounts_by_app.items():
            num_users = len(accounts)
            if num_users == 0:
                continue

            # Count users with 3+ entitlements in this app
            users_with_3_plus = 0
            users_needing_boost = []

            for account in accounts:
                total_ents = sum(len(grants) for grants in account.entitlement_grants.values())
                if total_ents >= 3:
                    users_with_3_plus += 1
                else:
                    users_needing_boost.append(account)

            current_pct = users_with_3_plus / num_users if num_users > 0 else 0
            self.logger.info(f"{app_name}: Current {current_pct:.1%} users have 3+ entitlements")

            if current_pct >= target_pct_per_app:
                continue  # Already meeting target for this app

            # Calculate how many users need boosting
            target_count = int(num_users * target_pct_per_app)
            shortfall = target_count - users_with_3_plus

            # Get available entitlements for this app
            entitlements = self.entitlements_by_app.get(app_name, [])
            if not entitlements:
                self.logger.warning(f"No entitlements available for {app_name} to enforce distribution")
                continue

            # Shuffle users needing boost and select the number needed
            self.rng.shuffle(users_needing_boost)
            users_to_boost = users_needing_boost[:min(shortfall, len(users_needing_boost))]

            # Boost selected users to have at least 3 entitlements
            for account in users_to_boost:
                current_total = sum(len(grants) for grants in account.entitlement_grants.values())
                needed = 3 - current_total

                if needed <= 0:
                    continue

                # Get the first attribute name for this app
                attr_name = list(account.entitlement_grants.keys())[0]
                current_grants = set(account.entitlement_grants[attr_name])

                # For Epic, handle both attribute types
                if app_name == "Epic":
                    # Distribute across both linkedTemplates and linkedSubTemplates
                    linked_templates = [e.entitlement_id for e in entitlements if
                                        e.entitlement_type == 'linkedTemplates']
                    sub_templates = [e.entitlement_id for e in entitlements if
                                     e.entitlement_type == 'linkedSubTemplates']

                    # Add to linkedTemplates first
                    current_linked = set(account.entitlement_grants.get('linkedTemplates', []))
                    available_linked = [e for e in linked_templates if e not in current_linked]
                    if available_linked and needed > 0:
                        to_add = min(needed, len(available_linked), 2)  # Add up to 2
                        new_grants = self.rng.choice(available_linked, size=to_add, replace=False).tolist()
                        account.entitlement_grants.setdefault('linkedTemplates', [])
                        account.entitlement_grants['linkedTemplates'].extend(new_grants)
                        needed -= to_add

                    # Add to linkedSubTemplates if still needed
                    current_sub = set(account.entitlement_grants.get('linkedSubTemplates', []))
                    available_sub = [e for e in sub_templates if e not in current_sub]
                    if available_sub and needed > 0:
                        to_add = min(needed, len(available_sub))
                        new_grants = self.rng.choice(available_sub, size=to_add, replace=False).tolist()
                        account.entitlement_grants.setdefault('linkedSubTemplates', [])
                        account.entitlement_grants['linkedSubTemplates'].extend(new_grants)
                else:
                    # Standard apps - add to the single entitlement attribute
                    ent_ids = [e.entitlement_id for e in entitlements]
                    available = [e for e in ent_ids if e not in current_grants]

                    if available:
                        to_add = min(needed, len(available))
                        new_grants = self.rng.choice(available, size=to_add, replace=False).tolist()
                        account.entitlement_grants[attr_name].extend(new_grants)

            self.logger.info(f"{app_name}: Boosted {len(users_to_boost)} users to meet {target_pct_per_app:.0%} target")

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

    def _ensure_no_users_without_entitlements(self) -> None:
        """
        Final safeguard: Ensure NO user has zero entitlements across ALL applications.

        This catches any edge cases where a user might have slipped through with
        empty entitlement lists in every app.
        """
        self.logger.info("Running final check: ensuring all users have at least 1 entitlement...")

        # Build map of user_id -> total entitlements across all apps
        user_total_ents = defaultdict(int)
        for app_name, accounts in self.accounts_by_app.items():
            for account in accounts:
                total = sum(len(grants) for grants in account.entitlement_grants.values())
                user_total_ents[account.user_id] += total

        # Find users with zero entitlements
        users_with_zero = [uid for uid, count in user_total_ents.items() if count == 0]

        if not users_with_zero:
            self.logger.info(f"✓ All {len(self.identities)} users have at least 1 entitlement")
            return

        self.logger.warning(f"Found {len(users_with_zero)} users with ZERO entitlements - fixing...")

        # Fix each user by giving them 1 entitlement in a random app
        for user_id in users_with_zero:
            # Pick a random app that has entitlements
            available_apps = [
                app_name for app_name, ents in self.entitlements_by_app.items()
                if ents  # Has entitlements available
            ]

            if not available_apps:
                self.logger.error("No apps have entitlements available - cannot fix zero-entitlement users")
                continue

            # Choose random app
            app_name = self.rng.choice(available_apps)
            entitlements = self.entitlements_by_app[app_name]

            # Find this user's account in the chosen app
            accounts = self.accounts_by_app.get(app_name, [])
            user_account = None
            for account in accounts:
                if account.user_id == user_id:
                    user_account = account
                    break

            if not user_account:
                self.logger.warning(f"Could not find account for user {user_id} in {app_name}")
                continue

            # Grant 1 random entitlement
            random_ent = self.rng.choice(entitlements)

            # Determine attribute name (handle Epic special case)
            if app_name == "Epic":
                attr_name = 'linkedTemplates' if random_ent.entitlement_type == 'linkedTemplates' else 'linkedSubTemplates'
            else:
                attr_name = 'entitlement_grants'

            # Add the entitlement
            user_account.entitlement_grants.setdefault(attr_name, [])
            if random_ent.entitlement_id not in user_account.entitlement_grants[attr_name]:
                user_account.entitlement_grants[attr_name].append(random_ent.entitlement_id)
                self.logger.debug(f"  Fixed user {user_id}: granted {random_ent.entitlement_id} in {app_name}")

        self.logger.info(f"✓ Fixed {len(users_with_zero)} users who had zero entitlements")

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

    def _initialize_rule_following_users(self) -> None:
        """
        Pre-determine which users will follow rules across ALL apps.

        This ensures cross-app atomicity: if a user follows rules, they do so
        consistently across all applications. This prevents partial rule application
        where a user gets entitlements from a cross-app rule in one app but not another.

        Critical for cross-app rules like:
        "Users with department='Engineering' get entitlements in SAP AND AWS"

        Without this, the random check would happen per app, potentially causing:
        - User follows rules for SAP (random < 0.93) ✓
        - User doesn't follow rules for AWS (random > 0.93) ✗
        Result: Cross-app rule broken

        With this fix:
        - Decision made once: User follows rules ✓
        - Applied to SAP: Gets SAP entitlements from rules ✓
        - Applied to AWS: Gets AWS entitlements from rules ✓
        Result: Cross-app rule maintains atomicity
        """
        if not self.rule_engine:
            self.logger.info("No rule engine provided - all users will use random assignment")
            return

        pct_modelled = float(
            self.config.get('confidence', {}).get('pct_modelled_users', 0.93)
        )

        self.logger.info(
            f"Pre-determining rule-following users: {pct_modelled:.1%} of {len(self.identities)} users"
        )
        count_forced = 0
        count_random = 0
        # Single random check per user (not per user-app combination)
        for identity in self.identities:
            # Case 1: Rule-Based Identity (Phase 2)
            # These users were created specifically to match a rule, so they MUST follow it.
            if getattr(identity, 'created_from_rule', False):
                self.rule_following_users.add(identity.user_id)
                count_forced += 1

            # Case 2: Random Identity (Phase 1/Standard)
            # Apply random coin flip to decide if they should also follow rules.
            elif self.rng.random() < pct_modelled:
                self.rule_following_users.add(identity.user_id)
                count_random += 1

        self.logger.info(
            f"✓ Rule-following population: {len(self.rule_following_users)} total "
            f"({count_forced} forced from schema, {count_random} random selections)"
        )

    # =========================================================================
    # BEGIN FIX: Deterministic Quota-Based Rule Application
    # =========================================================================

    def _build_user_feature_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Build an index of user_id -> feature markers for efficient rule matching.

        Returns:
            Dict mapping user_id to dict containing 'markers' (set) and 'raw' (dict)
        """
        user_features = {}

        for identity in self.identities:
            features = {
                'department': identity.department,
                'job_level': identity.job_level,
                'employment_type': identity.employment_type,
                'business_unit': identity.business_unit,
                'job_category': identity.job_category,
                'location_country': identity.location_country,
                'has_manager': 'Yes' if identity.manager else 'No',
                'is_manager': identity.is_manager,
                'department_type': identity.department_type,
                'location_site': identity.location_site,
                'jobcode': identity.jobcode,
            }

            # Convert to FEATURE: markers
            markers = {
                f"FEATURE:{k}={v}"
                for k, v in features.items()
                if v is not None
            }

            user_features[identity.user_id] = {
                'markers': markers,
                'raw': features
            }

        return user_features

    def _compute_rule_quotas(self, app_name: str) -> List[RuleQuota]:
        """
        Compute quotas for each rule based on target confidence.

        For each rule:
        1. Count users matching antecedent (freq)
        2. Calculate freqUnion_target = confidence × freq

        This ensures that when rules are mined from the generated data,
        the mined confidence will match the target confidence.

        Args:
            app_name: Application name

        Returns:
            List of RuleQuota objects with computed quotas
        """
        rules = self.rule_engine.get_rules_for_app(app_name)
        if not rules:
            return []

        # Build user feature index if not already built
        if not hasattr(self, '_user_feature_index'):
            self._user_feature_index = self._build_user_feature_index()

        quotas = []

        for rule in rules:
            antecedent = set(rule.antecedent_entitlements)

            # Skip rules without feature-based antecedents
            if not antecedent or not any(a.startswith("FEATURE:") for a in antecedent):
                continue

            # Count users matching this antecedent (freq)
            # Only consider users in rule_following_users population
            matching_user_ids = []
            for user_id in self.rule_following_users:
                user_data = self._user_feature_index.get(user_id)
                if user_data and antecedent.issubset(user_data['markers']):
                    matching_user_ids.append(user_id)

            freq = len(matching_user_ids)

            if freq == 0:
                continue

            # Calculate target freqUnion = confidence × freq
            # This is the KEY formula: to achieve mined confidence = target confidence,
            # we need exactly this many users to have the entitlement
            freqUnion_target = int(round(rule.confidence * freq))

            # Ensure at least 1 if freq > 0 and confidence > 0
            if freqUnion_target == 0 and freq > 0 and rule.confidence > 0:
                freqUnion_target = 1

            quota = RuleQuota(
                rule_id=rule.id,
                app_name=app_name,
                antecedent_markers=antecedent,
                consequent_entitlements=list(rule.consequent_entitlements),
                confidence=rule.confidence,
                freq=freq,
                freqUnion_target=freqUnion_target,
                matching_users=matching_user_ids
            )

            quotas.append(quota)

        self.logger.info(
            f"{app_name}: Computed quotas for {len(quotas)} rules"
        )

        # Log sample quotas for debugging
        for quota in quotas[:3]:
            self.logger.debug(
                f"  Rule {quota.rule_id}: freq={quota.freq}, "
                f"freqUnion_target={quota.freqUnion_target}, "
                f"target_conf={quota.confidence:.2%}"
            )

        return quotas

    def _assign_entitlements_by_quota(
            self,
            app_name: str,
            quotas: List[RuleQuota]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Deterministically assign entitlements based on rule quotas.

        For each rule, selects exactly freqUnion_target users from those
        matching the antecedent to receive the consequent entitlements.

        CRITICAL FIX: Uses stable user ordering to ensure nested/overlapping
        patterns select CONSISTENT user subsets. Without this, rules with
        overlapping patterns would select different random subsets, diluting
        the mined confidence.

        Example problem without stable ordering:
        - Rule A: {dept=Finance} → Ent_X (85% conf) selects users [1,2,3,4,5]
        - Rule B: {dept=Finance, level=Mid} → Ent_X (85% conf) selects [2,6,7,8,9]
        - Mined confidence for {dept=Finance} → Ent_X becomes ~50% instead of 85%

        With stable ordering:
        - Users are sorted by user_id before selection
        - Rule A selects first 85% = [1,2,3,4,5]
        - Rule B (subset of A's users) selects first 85% = [2,3] (subset remains consistent)
        - Mined confidence stays at 85%

        Args:
            app_name: Application name
            quotas: List of RuleQuota objects

        Returns:
            Dict mapping user_id -> {
                'entitlements': set of entitlement IDs,
                'rules_applied': list of RuleQuota that granted entitlements,
                'rules_matched': list of RuleQuota that matched but didn't grant
            }
        """
        # Track assignments per user
        user_assignments: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                'entitlements': set(),
                'rules_applied': [],
                'rules_matched': []
            }
        )

        # =================================================================
        # BEGIN FIX: Stable user ordering for consistent selection
        # =================================================================

        # Sort quotas by confidence (descending) to prioritize high-confidence rules
        sorted_quotas = sorted(quotas, key=lambda q: q.confidence, reverse=True)

        for quota in sorted_quotas:
            matching_users = list(quota.matching_users)

            if not matching_users:
                continue

            # CRITICAL FIX: Sort users by user_id for STABLE ordering
            # This ensures that for overlapping patterns (e.g., {dept=Finance} vs
            # {dept=Finance, level=Mid}), the selected users are consistent subsets.
            #
            # Without this: random shuffle causes different 85% selections for each rule
            # With this: first 85% is always the same users (sorted by ID)
            matching_users_sorted = sorted(matching_users)

            # Select exactly freqUnion_target users from the SORTED list
            # This is deterministic: always selects the first N users by ID
            selected_users = matching_users_sorted[:quota.freqUnion_target]
            non_selected_users = matching_users_sorted[quota.freqUnion_target:]

            # Assign entitlements to selected users
            for user_id in selected_users:
                user_assignments[user_id]['entitlements'].update(
                    quota.consequent_entitlements
                )
                user_assignments[user_id]['rules_applied'].append(quota)
                quota.assigned_users.add(user_id)

            # Track matched-but-not-granted for non-selected users
            for user_id in non_selected_users:
                user_assignments[user_id]['rules_matched'].append(quota)

            self.logger.debug(
                f"Rule {quota.rule_id}: Assigned to {len(selected_users)}/{quota.freq} users "
                f"(target confidence: {quota.confidence:.2%})"
            )

        # =================================================================
        # END FIX: Stable user ordering
        # =================================================================

        return dict(user_assignments)

    def _calculate_user_confidence(
            self,
            rules_applied: List[RuleQuota],
            rules_matched: List[RuleQuota]
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Calculate confidence score and bucket for a user based on rules.

        Args:
            rules_applied: Rules that granted entitlements to this user
            rules_matched: Rules that matched but didn't grant (user wasn't selected)

        Returns:
            (confidence_score, confidence_bucket)
        """
        if rules_applied:
            # User got entitlements from rules - use average confidence of applied rules
            avg_confidence = sum(r.confidence for r in rules_applied) / len(rules_applied)
            score = avg_confidence
            bucket = self._bucket_from_score(score)
        elif rules_matched:
            # User matched rules but wasn't selected for entitlements
            # This represents the (1 - confidence) portion of users
            # Give them a reduced score to indicate partial match
            avg_confidence = sum(r.confidence for r in rules_matched) / len(rules_matched)
            score = avg_confidence * 0.3  # Reduced score for non-granted matches
            bucket = self._bucket_from_score(score)
        else:
            # No rules matched at all
            score = None
            bucket = None

        return score, bucket

    # =========================================================================
    # END FIX: Deterministic Quota-Based Rule Application
    # =========================================================================


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
        method = str(self.confidence_cfg.get('calculation_method', 'random_weighted')).lower()
        if method == "rule_support":
            # Workflow A: scores already set during account generation.
            self.logger.info("Confidence method 'rule_support' – skipping random bucket assignment.")
            return

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
                    results['warnings'].append(
                        f"Feature '{feature}' has high cardinality ({unique_count} > {max_unique})")
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
        self.quoting = csv.QUOTE_ALL if config.get('global', {}).get('csv_quoting',
                                                                     'all') == 'all' else csv.QUOTE_MINIMAL

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
        """Write accounts for an app to CSV and validate no duplicates."""
        output_path = self.output_dir / f"{app_name}_accounts.csv"
        df.to_csv(output_path, index=False, quoting=self.quoting)
        self.logger.info(f"Written {len(df)} accounts to {output_path}")

        # =====================================================================
        # BEGIN FIX: Validate no duplicate accounts after writing
        # =====================================================================
        # Read back and verify uniqueness
        df_verify = pd.read_csv(output_path)
        total_rows = len(df_verify)
        unique_user_ids = df_verify['user_id'].nunique()
        unique_user_names = df_verify['user_name'].nunique()

        if total_rows != unique_user_ids:
            self.logger.error(
                f"{app_name}: DUPLICATE user_id DETECTED! "
                f"Total rows: {total_rows}, Unique user_ids: {unique_user_ids}"
            )
            # Log sample duplicates for debugging
            dup_ids = df_verify[df_verify.duplicated(subset=['user_id'], keep=False)]
            if len(dup_ids) > 0:
                sample = dup_ids.head(5)['user_id'].tolist()
                self.logger.error(f"{app_name}: Sample duplicate user_ids: {sample}")
            raise ValueError(f"Duplicate user_id found in {app_name}_accounts.csv")

        if total_rows != unique_user_names:
            self.logger.error(
                f"{app_name}: DUPLICATE user_name DETECTED! "
                f"Total rows: {total_rows}, Unique user_names: {unique_user_names}"
            )
            raise ValueError(f"Duplicate user_name found in {app_name}_accounts.csv")

        self.logger.info(
            f"{app_name}: Validated - {total_rows} rows, "
            f"{unique_user_ids} unique user_ids, {unique_user_names} unique user_names ✓"
        )
        # =====================================================================
        # END FIX: Validate no duplicate accounts
        # =====================================================================

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
                summary['Confidence'][bucket] = f"{count} ({count / total:.1%})"

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

        # BEGIN CORRECTED ARCHITECTURE

        # CORRECTED Step 4: Generate entitlements FIRST
        # This must happen before rules because rules reference entitlement IDs
        self.entitlement_generator = EntitlementGenerator(self.config, rng, faker, self.input_reader)
        self.entitlements_by_app = self.entitlement_generator.generate_all()
        self.logger.info("✓ Generated entitlements catalog")

        # CORRECTED Step 5: Generate/Load rules BEFORE identities
        # Rules define the target patterns we want in the data
        rules_dir = Path(self.config.get('global', {}).get('rules_directory', 'rules'))

        dynamic_config = self.config.get('dynamic_rules', {})
        if dynamic_config.get('enabled', False):
            # Generate rules with target statistical properties
            self._generate_rules_first(seed, rng)
            self.logger.info("✓ Generated rules with target distributions")
        else:
            # Load pre-defined rules
            if not rules_dir.exists() or not list(rules_dir.glob("*.json")):
                raise ValueError(
                    "No rules found and dynamic_rules.enabled=false. "
                    "Either enable dynamic rules or provide rule files in rules_directory."
                )
            self.logger.info("✓ Using pre-defined rules from rules_directory")

        # Initialize rule engine
        rule_engine = RuleEngine(rules_dir=rules_dir, rng=rng)

        # CORRECTED Step 6: Generate identities conforming to rule antecedents
        # Instead of random identities, generate them to satisfy rule patterns
        self.identity_generator = IdentityGenerator(
            self.config, rng, faker, self.input_reader
        )

        # NEW: Pass rule engine to identity generator so it can create
        # identities that satisfy rule antecedents
        num_identities = self.config.get('identity', {}).get('num_identities', 10000)
        self.logger.info(
            "Note: Using standard identity generation. "
            "Rule-aware generation (Phase 2) not yet implemented."
        )
        self.identities = self.identity_generator.generate_rule_aware(
            num_identities=num_identities,
            rule_engine=rule_engine
        )
        self.identities_df = self.identity_generator.to_dataframe()
        self.logger.info("✓ Generated identities conforming to rule patterns")

        # CORRECTED Step 7: Apply rules to assign entitlements
        # Now rules are applied to identities designed to satisfy them
        self.account_generator = AccountGenerator(
            self.config, rng, self.identities, self.entitlements_by_app, rule_engine=rule_engine
        )
        self.accounts_by_app = self.account_generator.generate_all()
        self.logger.info("✓ Applied rules to assign entitlements")

        # END CORRECTED ARCHITECTURE

        # Step 8: Calculate confidence scores
        self.confidence_calculator = ConfidenceScoreCalculator(
            self.config, rng, self.identities_df, self.accounts_by_app
        )
        self.confidence_calculator.calculate_scores()

        # Step 9: Validate features
        self.feature_validator = FeatureValidator(
            self.config, self.identities_df, self.accounts_by_app
        )
        validation_results = self.feature_validator.validate()

        # Step 10: Write output files
        self._write_outputs(validation_results)

        self.logger.info("=" * 60)
        self.logger.info("SYNTHETIC DATA GENERATION - COMPLETE")
        self.logger.info("=" * 60)

    def _generate_rules_first(self, seed: int, rng: np.random.Generator) -> None:
        """
        CORRECTED: Generate rules BEFORE identities.

        Rules are defined with target statistical properties independent of any
        specific identity population. This prevents overfitting.
        """
        self.logger.info("=" * 60)
        self.logger.info("GENERATING RULES (BEFORE IDENTITIES)")
        self.logger.info("=" * 60)

        dynamic_config = self.config.get('dynamic_rules', {})
        rules_dir = Path(self.config.get('global', {}).get('rules_directory', './rules'))
        rules_dir.mkdir(parents=True, exist_ok=True)

        output_filename = dynamic_config.get('output_file', 'generated_rules.json')
        output_path = rules_dir / output_filename

        # Get apps configuration
        apps_cfg = self.config.get('applications', {})
        apps_list = apps_cfg.get('apps', [])
        enabled_apps = [
            {'app_name': app['app_name'], 'app_id': app.get('app_id', f"APP_{app['app_name'].upper()}")}
            for app in apps_list if app.get('enabled', True)
        ]

        use_cross_app = dynamic_config.get('use_cross_app_rules', False)
        if isinstance(use_cross_app, dict):
            use_cross_app = use_cross_app.get('value', False)
        use_cross_app = bool(use_cross_app)

        # BEGIN NEW: Rule generation without requiring existing identities
        if use_cross_app:
            self._generate_cross_app_rules_schema(
                dynamic_config, enabled_apps, output_path, seed
            )
        else:
            self._generate_per_app_rules_schema(
                dynamic_config, enabled_apps, output_path, seed
            )
        # END NEW

        self.logger.info(f"✓ Rules generated and saved to {output_path}")

    def _generate_per_app_rules_schema(
            self,
            dynamic_config: Dict[str, Any],
            enabled_apps: List[Dict[str, Any]],
            output_path: Path,
            seed: int
    ) -> None:
        """
        Generate per-app rules based on abstract schema, not actual identities.

        Instead of mining patterns from generated data, we define rules with
        target statistical properties that will guide identity generation.
        """
        # BEGIN NEW METHOD
        from rule_schema_generator import RuleGeneratorConfig, RuleSchemaGenerator

        # Extract configuration
        confidence_ranges_config = dynamic_config.get('confidence_ranges', {})
        confidence_ranges = {}
        for bucket, range_cfg in confidence_ranges_config.items():
            if isinstance(range_cfg, dict):
                confidence_ranges[bucket] = (
                    range_cfg.get('min', 0.0),
                    range_cfg.get('max', 1.0)
                )
            else:
                confidence_ranges[bucket] = (0.65, 0.95)

        support_range_cfg = dynamic_config.get('support_range', {})
        support_range = (
            support_range_cfg.get('min', 0.01),
            support_range_cfg.get('max', 0.20)
        ) if isinstance(support_range_cfg, dict) else (0.01, 0.20)

        cramers_v_range_cfg = dynamic_config.get('cramers_v_range', {})
        cramers_v_range = (
            cramers_v_range_cfg.get('min', 0.30),
            cramers_v_range_cfg.get('max', 0.50)
        ) if isinstance(cramers_v_range_cfg, dict) else (0.30, 0.50)

        coordinate_rules = dynamic_config.get('coordinate_rules_across_apps', False)
        if isinstance(coordinate_rules, dict):
            coordinate_rules = coordinate_rules.get('value', False)

        num_patterns = dynamic_config.get('num_unique_feature_patterns', None)
        if isinstance(num_patterns, dict):
            num_patterns = num_patterns.get('value', None)

        config = RuleGeneratorConfig(
            num_rules_per_app=dynamic_config.get('num_rules_per_app', 5),
            confidence_distribution=dynamic_config.get('confidence_distribution', {
                'high': 0.40,
                'medium': 0.35,
                'low': 0.25
            }),
            confidence_ranges=confidence_ranges,
            support_range=support_range,
            cramers_v_range=cramers_v_range,
            min_features_per_rule=dynamic_config.get('min_features_per_rule', 1),
            max_features_per_rule=dynamic_config.get('max_features_per_rule', 3),
            min_entitlements_per_rule=dynamic_config.get('min_entitlements_per_rule', 1),
            max_entitlements_per_rule=dynamic_config.get('max_entitlements_per_rule', 4),
            coordinate_rules_across_apps=coordinate_rules,
            num_unique_feature_patterns=num_patterns
        )

        # NEW: Schema-based generation (no identities required)
        generator = RuleSchemaGenerator(
            apps=enabled_apps,
            entitlements_by_app=self.entitlements_by_app,
            feature_schema=self._get_feature_schema(),  # Abstract feature definitions
            config=config,
            rng=np.random.default_rng(seed)
        )

        rules = generator.generate_all_rules()

        # Save rules
        self._save_rules_json(rules, output_path)
        # END NEW METHOD

    def _generate_cross_app_rules_schema(
            self,
            dynamic_config: Dict[str, Any],
            enabled_apps: List[Dict[str, Any]],
            output_path: Path,
            seed: int
    ) -> None:
        """
        Generate cross-app rules based on abstract schema.
        """
        # BEGIN NEW METHOD
        from cross_app_rule_schema_generator import (
            CrossAppRuleGeneratorConfig,
            CrossAppRuleSchemaGenerator
        )

        # Extract configuration for cross-app rules
        confidence_ranges_config = dynamic_config.get('confidence_ranges', {})
        confidence_ranges = {}
        for bucket, range_cfg in confidence_ranges_config.items():
            if isinstance(range_cfg, dict):
                confidence_ranges[bucket] = (
                    range_cfg.get('min', 0.0),
                    range_cfg.get('max', 1.0)
                )
            else:
                confidence_ranges[bucket] = (0.65, 0.95)

        support_range_cfg = dynamic_config.get('support_range', {})
        if isinstance(support_range_cfg, dict):
            support_range = (
                support_range_cfg.get('min', 0.01),
                support_range_cfg.get('max', 0.20)
            )
        else:
            support_range = (0.01, 0.20)

        cramers_v_range_cfg = dynamic_config.get('cramers_v_range', {})
        if isinstance(cramers_v_range_cfg, dict):
            cramers_v_range = (
                cramers_v_range_cfg.get('min', 0.30),
                cramers_v_range_cfg.get('max', 0.50)
            )
        else:
            cramers_v_range = (0.30, 0.50)

        # Extract cross-app specific parameters
        num_cross_app_rules = dynamic_config.get('num_cross_app_rules', 10)
        if isinstance(num_cross_app_rules, dict):
            num_cross_app_rules = num_cross_app_rules.get('value', 10)
        num_cross_app_rules = int(num_cross_app_rules)

        apps_per_rule_min = dynamic_config.get('apps_per_rule_min', 2)
        if isinstance(apps_per_rule_min, dict):
            apps_per_rule_min = apps_per_rule_min.get('value', 2)
        apps_per_rule_min = int(apps_per_rule_min)

        apps_per_rule_max = dynamic_config.get('apps_per_rule_max', 4)
        if isinstance(apps_per_rule_max, dict):
            apps_per_rule_max = apps_per_rule_max.get('value', 4)
        apps_per_rule_max = int(apps_per_rule_max)

        # Create cross-app configuration
        config = CrossAppRuleGeneratorConfig(
            num_cross_app_rules=num_cross_app_rules,
            apps_per_rule_min=apps_per_rule_min,
            apps_per_rule_max=apps_per_rule_max,
            confidence_distribution=dynamic_config.get('confidence_distribution', {
                'high': 0.40,
                'medium': 0.35,
                'low': 0.25
            }),
            confidence_ranges=confidence_ranges if confidence_ranges else None,
            support_range=support_range,
            cramers_v_range=cramers_v_range,
            min_features_per_rule=dynamic_config.get('min_features_per_rule', 1),
            max_features_per_rule=dynamic_config.get('max_features_per_rule', 3),
            min_entitlements_per_app=dynamic_config.get('min_entitlements_per_app', 1),
            max_entitlements_per_app=dynamic_config.get('max_entitlements_per_app', 4)
        )

        self.logger.info(f"Generating {num_cross_app_rules} cross-app rules from schema")
        self.logger.info(f"Apps per rule: {apps_per_rule_min}-{apps_per_rule_max}")

        # NEW: Schema-based cross-app generation
        generator = CrossAppRuleSchemaGenerator(
            apps=enabled_apps,
            entitlements_by_app=self.entitlements_by_app,
            feature_schema=self._get_feature_schema(),
            config=config,
            rng=np.random.default_rng(seed)
        )

        rules = generator.generate_all_rules()
        self._save_rules_json(rules, output_path)

        self.logger.info(f"Generated {len(rules)} cross-app schema-based rules")
        # END NEW METHOD

    def _get_feature_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Define abstract feature schema for rule generation.

        This describes what features exist and their possible values,
        WITHOUT requiring actual identity data.

        CRITICAL FIX: Only include features with cardinality <= max_cardinality.
        High-cardinality features (like department with 198 values) create rules
        that are too specific and result in low mined confidence when validated.

        The validator uses Cramér's V to select features, which naturally
        filters out high-cardinality features. Rules must use the same
        low-cardinality features for the mined confidence to match targets.

        Returns:
            Dict mapping feature names to their schemas:
            {
                'job_level': {
                    'type': 'categorical',
                    'values': ['Junior', 'Mid', 'Senior', 'Executive'],
                    'distribution': {...}
                },
                ...
            }
        """
        # BEGIN FIX: Get max_cardinality from config to filter features
        dynamic_rules_cfg = self.config.get('dynamic_rules', {})
        max_cardinality = dynamic_rules_cfg.get('max_cardinality', 20)
        self.logger.info(f"Feature schema: filtering features with cardinality <= {max_cardinality}")
        # END FIX

        features_cfg = self.config.get('features', {})
        mandatory_features = features_cfg.get('mandatory_features', [])
        additional_features = features_cfg.get('additional_features', [])

        all_features = mandatory_features + additional_features

        schema = {}

        # Load department values from input file
        input_files = self.config.get('identity', {}).get('input_files', {})
        dept_file = input_files.get('departments', 'input/departments.csv')
        departments = self.input_reader.read_csv(dept_file)

        # BEGIN FIX: Skip department if cardinality exceeds max_cardinality
        if 'department' in all_features and departments:
            dept_names = [d.get('department_name', '') for d in departments if d.get('department_name')]
            # Only include if cardinality is within limit
            if len(dept_names) <= max_cardinality:
                schema['department'] = {
                    'type': 'categorical',
                    'values': dept_names,
                    'cardinality': len(dept_names),
                    'distribution': 'uniform'
                }
            else:
                self.logger.info(f"  Skipping 'department': cardinality {len(dept_names)} > {max_cardinality}")
        # END FIX

        # Load business unit values from departments
        if 'business_unit' in all_features and departments:
            business_units = list(set([
                d.get('Industry', '') for d in departments
                if d.get('Industry')
            ]))

            schema['business_unit'] = {
                'type': 'categorical',
                'values': business_units,
                'cardinality': len(business_units),
                'distribution': 'uniform'
            }

        # Load department type from departments
        if 'department_type' in all_features and departments:
            dept_types = list(set([
                d.get('Type', '') for d in departments
                if d.get('Type')
            ]))

            schema['department_type'] = {
                'type': 'categorical',
                'values': dept_types,
                'cardinality': len(dept_types),
                'distribution': 'uniform'
            }

        # Load job titles from input file
        job_file = input_files.get('jobtitles', 'input/jobtitles.csv')
        jobtitles = self.input_reader.read_csv(job_file)

        # BEGIN FIX: Skip jobcode if cardinality exceeds max_cardinality
        if 'jobcode' in all_features and jobtitles:
            job_names = [j.get('Job Title', '') for j in jobtitles if j.get('Job Title')]
            # Only include if cardinality is within limit
            if len(job_names) <= max_cardinality:
                schema['jobcode'] = {
                    'type': 'categorical',
                    'values': job_names,
                    'cardinality': len(job_names),
                    'distribution': 'uniform'
                }
            else:
                self.logger.info(f"  Skipping 'jobcode': cardinality {len(job_names)} > {max_cardinality}")
        # END FIX

        # Load job category from jobtitles
        if 'job_category' in all_features and jobtitles:
            job_categories = list(set([
                j.get('Category', '') for j in jobtitles
                if j.get('Category')
            ]))

            schema['job_category'] = {
                'type': 'categorical',
                'values': job_categories,
                'cardinality': len(job_categories),
                'distribution': 'uniform'
            }

        # Define job_level (inferred from job titles, not loaded from file)
        if 'job_level' in all_features:
            schema['job_level'] = {
                'type': 'categorical',
                'values': list(IdentityGenerator.JOB_LEVEL_KEYWORDS.keys()),
                'cardinality': len(IdentityGenerator.JOB_LEVEL_KEYWORDS),
                'distribution': IdentityGenerator.JOB_LEVEL_DISTRIBUTION
            }

        # Define employment_type from config
        if 'employment_type' in all_features:
            emp_dist = self.config.get('identity', {}).get('distribution_employee_contractor', {
                'Employee': 0.70,
                'Contractor': 0.20,
                'Intern': 0.10
            })

            schema['employment_type'] = {
                'type': 'categorical',
                'values': list(emp_dist.keys()),
                'cardinality': len(emp_dist),
                'distribution': emp_dist
            }

        # Define identity_type from config
        if 'identity_type' in all_features:
            identity_dist = self.config.get('identity', {}).get('distribution_human_ai_agent', {
                'Human': 0.95,
                'AI_Agent': 0.05
            })

            schema['identity_type'] = {
                'type': 'categorical',
                'values': list(identity_dist.keys()),
                'cardinality': len(identity_dist),
                'distribution': identity_dist
            }

        # Define location_country
        if 'location_country' in all_features:
            schema['location_country'] = {
                'type': 'categorical',
                'values': list(IdentityGenerator.LOCATION_COUNTRY_DIST.keys()),
                'cardinality': len(IdentityGenerator.LOCATION_COUNTRY_DIST),
                'distribution': IdentityGenerator.LOCATION_COUNTRY_DIST
            }

        # Define location_site
        if 'location_site' in all_features:
            all_sites = []
            for sites in IdentityGenerator.LOCATION_SITES_BY_COUNTRY.values():
                all_sites.extend(sites.keys())

            # BEGIN FIX: Respect max_cardinality for location_site
            if len(all_sites) <= max_cardinality:
                schema['location_site'] = {
                    'type': 'categorical',
                    'values': all_sites,
                    'cardinality': len(all_sites),
                    'distribution': 'uniform'  # Complex nested distribution
                }
            else:
                self.logger.info(f"  Skipping 'location_site': cardinality {len(all_sites)} > {max_cardinality}")
            # END FIX

        # Define status
        if 'status' in all_features:
            status_dist = self.config.get('identity', {}).get('distribution_active_inactive', {
                'Active': 0.93,
                'Inactive': 0.07
            })

            schema['status'] = {
                'type': 'categorical',
                'values': list(status_dist.keys()),
                'cardinality': len(status_dist),
                'distribution': status_dist
            }

        # Define is_manager (boolean but treated as categorical)
        if 'is_manager' in all_features:
            manager_pct = self.config.get('identity', {}).get('manager_hierarchy', {}).get('pct_managers', 0.15)

            schema['is_manager'] = {
                'type': 'categorical',
                'values': ['Y', 'N'],
                'cardinality': 2,
                'distribution': {
                    'Y': manager_pct,
                    'N': 1.0 - manager_pct
                }
            }

        # Define manager (reference to another user)
        # For rule purposes, we'll treat this as a boolean "has manager"
        if 'manager' in all_features:
            pct_no_manager = self.config.get('identity', {}).get('pct_users_without_manager', 0.10)

            schema['has_manager'] = {
                'type': 'categorical',
                'values': ['Yes', 'No'],
                'cardinality': 2,
                'distribution': {
                    'Yes': 1.0 - pct_no_manager,
                    'No': pct_no_manager
                }
            }

        self.logger.info(f"Generated feature schema with {len(schema)} features")
        for feature_name, feature_spec in schema.items():
            self.logger.debug(
                f"  {feature_name}: {feature_spec['cardinality']} values"
            )

        return schema
        # END NEW METHOD

    def _save_rules_json(self, rules: List[Dict], output_path: Path) -> None:
        """Save rules to JSON with proper type conversion."""

        # BEGIN NEW METHOD
        def convert_to_native_types(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_to_native_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        rules_native = convert_to_native_types(rules)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(rules_native, f, indent=2)
        # END NEW METHOD

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

    def _generate_dynamic_rules(self, seed: int) -> None:
        """
        Generate association rules dynamically based on actual user population
        and available entitlements.

        """
        dynamic_config = self.config.get('dynamic_rules', {})

        if not dynamic_config.get('enabled', False):
            self.logger.info("Dynamic rule generation is disabled. Using pre-defined rules from rules_directory.")
            return

        if not DYNAMIC_RULES_AVAILABLE:
            self.logger.warning("Dynamic rule generator not available. Skipping dynamic rule generation.")
            return

        self.logger.info("=" * 60)
        self.logger.info("GENERATING ASSOCIATION RULES DYNAMICALLY")
        self.logger.info("=" * 60)

        # BEGIN NEW: Check if cross-app mode is requested
        use_cross_app = dynamic_config.get('use_cross_app_rules', False)
        if isinstance(use_cross_app, dict):
            use_cross_app = use_cross_app.get('value', False)
            use_cross_app = bool(use_cross_app)
        # END NEW

        # Prepare common configuration elements
        confidence_ranges_config = dynamic_config.get('confidence_ranges', {})

        # Convert nested config structure to tuple format
        confidence_ranges = {}
        for bucket, range_cfg in confidence_ranges_config.items():
            if isinstance(range_cfg, dict):
                confidence_ranges[bucket] = (
                    range_cfg.get('min', 0.0),
                    range_cfg.get('max', 1.0)
                )
            else:
                confidence_ranges[bucket] = (0.65, 0.95)

        # Get support range
        support_range_cfg = dynamic_config.get('support_range', {})
        if isinstance(support_range_cfg, dict):
            support_range = (
                support_range_cfg.get('min', 0.01),
                support_range_cfg.get('max', 0.20)
            )
        else:
            support_range = (0.01, 0.20)

        # Get cramers_v range
        cramers_v_range_cfg = dynamic_config.get('cramers_v_range', {})
        if isinstance(cramers_v_range_cfg, dict):
            cramers_v_range = (
                cramers_v_range_cfg.get('min', 0.30),
                cramers_v_range_cfg.get('max', 0.50)
            )
        else:
            cramers_v_range = (0.30, 0.50)

        # Prepare rules directory
        rules_dir = Path(self.config.get('global', {}).get('rules_directory', './rules'))
        rules_dir.mkdir(parents=True, exist_ok=True)

        # Output file path
        output_filename = dynamic_config.get('output_file', 'generated_rules.json')
        output_path = rules_dir / output_filename

        # Get apps configuration
        apps_cfg = self.config.get('applications', {})
        apps_list = apps_cfg.get('apps', [])
        enabled_apps = [
            {'app_name': app['app_name'], 'app_id': app.get('app_id', f"APP_{app['app_name'].upper()}")}
            for app in apps_list
            if app.get('enabled', True)
        ]

        # BEGIN NEW: Branch based on rule generation mode
        try:
            if use_cross_app:
                # ===================================================================
                # NEW: CROSS-APP RULE GENERATION MODE
                # ===================================================================
                self.logger.info("Using CROSS-APP rule generation mode")
                self.logger.info("Rules will grant entitlements across multiple apps simultaneously")

                # Import cross-app generator
                try:
                    from cross_app_rule_schema_generator import (
                        CrossAppRuleGeneratorConfig,
                        CrossAppRuleSchemaGenerator
                    )
                except ImportError:
                    self.logger.error(
                        "dynamic_rule_generator_cross_app.py not found. "
                        "Falling back to per-app coordinated mode."
                    )
                    use_cross_app = False

                if use_cross_app:
                    # Get cross-app specific configuration
                    num_cross_app_rules = dynamic_config.get('num_cross_app_rules', 10)
                    if isinstance(num_cross_app_rules, dict):
                        num_cross_app_rules = num_cross_app_rules.get('value', 10)
                    num_cross_app_rules = int(num_cross_app_rules)

                    apps_per_rule_min = dynamic_config.get('apps_per_rule_min', 2)
                    if isinstance(apps_per_rule_min, dict):
                        apps_per_rule_min = apps_per_rule_min.get('value', 2)
                    apps_per_rule_min = int(apps_per_rule_min)

                    apps_per_rule_max = dynamic_config.get('apps_per_rule_max', 4)
                    if isinstance(apps_per_rule_max, dict):
                        apps_per_rule_max = apps_per_rule_max.get('value', 4)
                    apps_per_rule_max = int(apps_per_rule_max)

                    # Create cross-app configuration
                    cross_app_config = CrossAppRuleGeneratorConfig(
                        num_cross_app_rules=num_cross_app_rules,
                        apps_per_rule_min=apps_per_rule_min,
                        apps_per_rule_max=apps_per_rule_max,
                        confidence_distribution=dynamic_config.get('confidence_distribution', {
                            'high': 0.40,
                            'medium': 0.35,
                            'low': 0.25
                        }),
                        confidence_ranges=confidence_ranges if confidence_ranges else None,
                        support_range=support_range,
                        cramers_v_range=cramers_v_range,
                        min_features_per_rule=dynamic_config.get('min_features_per_rule', 1),
                        max_features_per_rule=dynamic_config.get('max_features_per_rule', 3),
                        min_entitlements_per_app=dynamic_config.get('min_entitlements_per_app', 1),
                        max_entitlements_per_app=dynamic_config.get('max_entitlements_per_app', 4)
                    )

                    self.logger.info(f"Generating {num_cross_app_rules} cross-app rules")
                    self.logger.info(f"Apps per rule: {apps_per_rule_min}-{apps_per_rule_max}")

                    # Run cross-app rule generator
                    orchestrator = CrossAppRuleGenerationOrchestrator(
                        users_file=self.data_writer.output_dir / 'identities.csv',
                        apps_config=enabled_apps,
                        entitlements_dir=self.data_writer.output_dir,
                        output_file=output_path,
                        config=cross_app_config,
                        seed=seed
                    )

                    orchestrator.run()

                    self.logger.info(f"Cross-app rules successfully generated: {output_path}")
                    return

            # ===================================================================
            # ORIGINAL: PER-APP COORDINATED MODE (FALLBACK)
            # ===================================================================
            if not use_cross_app:
                self.logger.info("Using PER-APP coordinated rule generation mode")
                self.logger.info("Each rule grants entitlements for a single app")

                # Extract coordinate_rules_across_apps
                coordinate_rules = dynamic_config.get('coordinate_rules_across_apps', False)
                if isinstance(coordinate_rules, dict):
                    coordinate_rules = coordinate_rules.get('value', False)
                coordinate_rules = bool(coordinate_rules)

                # Extract num_unique_feature_patterns
                num_patterns = dynamic_config.get('num_unique_feature_patterns', None)
                if isinstance(num_patterns, dict):
                    num_patterns = num_patterns.get('value', None)
                if num_patterns is not None:
                    num_patterns = int(num_patterns)

                rule_gen_config = RuleGeneratorConfig(
                    num_rules_per_app=dynamic_config.get('num_rules_per_app', 5),
                    confidence_distribution=dynamic_config.get('confidence_distribution', {
                        'high': 0.40,
                        'medium': 0.35,
                        'low': 0.25
                    }),
                    confidence_ranges=confidence_ranges if confidence_ranges else None,
                    support_range=support_range,
                    cramers_v_range=cramers_v_range,
                    min_features_per_rule=dynamic_config.get('min_features_per_rule', 1),
                    max_features_per_rule=dynamic_config.get('max_features_per_rule', 3),
                    min_entitlements_per_rule=dynamic_config.get('min_entitlements_per_rule', 1),
                    max_entitlements_per_rule=dynamic_config.get('max_entitlements_per_rule', 4),
                    coordinate_rules_across_apps=coordinate_rules,
                    num_unique_feature_patterns=num_patterns
                )

            # Run original per-app rule generation orchestrator
            orchestrator = RuleGenerationOrchestrator(
                users_file=self.data_writer.output_dir / 'identities.csv',
                apps_config=enabled_apps,
                entitlements_dir=self.data_writer.output_dir,
                output_file=output_path,
                config=rule_gen_config,
                seed=seed
            )

            orchestrator.run()

            self.logger.info(f"Per-app rules successfully generated: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to generate dynamic rules: {e}")
            import traceback
            traceback.print_exc()

            # Fall back to using existing rules if available
            self.logger.warning("Falling back to pre-defined rules from rules_directory")
    # END NEW


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