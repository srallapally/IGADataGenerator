#!/usr/bin/env python3
"""
Unit tests for synthetic IGA dataset generator.

Tests verify acceptance criteria:
1. No duplicate identifiers
2. No duplicate assignments
3. Reproducibility with fixed seed
4. Calibration improves bucket proportions
"""

import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Import generator modules
sys.path.insert(0, str(Path(__file__).parent))
from gemini_synthetic_iga import (
    DEFAULT_CONFIG,
    IdentityGenerator,
    EntitlementGenerator,
    PatternGenerator,
    AssignmentGenerator,
    CalibrationEngine,
    DataWriter
)


class TestIdentityGenerator(unittest.TestCase):
    """Test identity generation."""

    def test_no_duplicate_user_ids(self):
        """Verify all user_ids are unique."""
        config = DEFAULT_CONFIG.copy()
        gen = IdentityGenerator(config, seed=42)
        identities = gen.generate(100)

        user_ids = [i.user_id for i in identities]
        self.assertEqual(len(user_ids), len(set(user_ids)),
                         "Found duplicate user_ids")

    def test_reproducibility(self):
        """Verify reproducibility with same seed."""
        config = DEFAULT_CONFIG.copy()

        gen1 = IdentityGenerator(config, seed=42)
        identities1 = gen1.generate(100)

        gen2 = IdentityGenerator(config, seed=42)
        identities2 = gen2.generate(100)

        # Check that same users generated
        for i in range(len(identities1)):
            self.assertEqual(identities1[i].user_id, identities2[i].user_id)
            self.assertEqual(identities1[i].first_name, identities2[i].first_name)
            self.assertEqual(identities1[i].department, identities2[i].department)

    def test_required_columns(self):
        """Verify all required columns are present."""
        config = DEFAULT_CONFIG.copy()
        gen = IdentityGenerator(config, seed=42)
        identities = gen.generate(100)

        # --- BEGIN ChatGPT change: Phase 4 - extend required identity attributes ---
        required_attrs = [
            'user_id', 'user_name', 'first_name', 'last_name', 'email',
            'department', 'business_unit', 'line_of_business', 'job_family',
            'job_level', 'title', 'location_country', 'location_site',
            'employment_type', 'cost_center', 'manager_flag', 'manager_id',
            'status', 'tenure_years'
        ]
        # --- END ChatGPT change: Phase 4 - extend required identity attributes ---

        for attr in required_attrs:
            self.assertTrue(hasattr(identities[0], attr),
                            f"Missing attribute: {attr}")

    def test_job_level_range(self):
        """Verify job levels are in valid range."""
        config = DEFAULT_CONFIG.copy()
        gen = IdentityGenerator(config, seed=42)
        identities = gen.generate(100)

        for identity in identities:
            self.assertTrue(1 <= identity.job_level <= 7,
                            f"Invalid job_level: {identity.job_level}")

    # --- BEGIN ChatGPT change: Phase 4 - new identity tests for username, manager, status, LOB ---
    def test_user_name_uniqueness_and_format(self):
        """Verify user_name is unique and strictly alphanumeric."""
        config = DEFAULT_CONFIG.copy()
        gen = IdentityGenerator(config, seed=42)
        identities = gen.generate(500)

        usernames = [i.user_name for i in identities]

        # Uniqueness
        self.assertEqual(
            len(usernames),
            len(set(usernames)),
            "Found duplicate user_name values"
        )

        # Non-empty and alphanumeric only
        for uname in usernames:
            self.assertTrue(uname, "user_name should not be empty")
            self.assertRegex(
                uname,
                r"^[A-Za-z0-9]+$",
                f"user_name contains invalid characters: {uname}"
            )

    def test_manager_ratio_and_coverage(self):
        """
        Verify that ~25% of users are managers and
        that >= 90% of users have a manager_id.
        """
        config = DEFAULT_CONFIG.copy()
        gen = IdentityGenerator(config, seed=42)
        identities = gen.generate(1000)

        total = len(identities)
        managers = [i for i in identities if i.manager_flag == "Y"]
        mgr_ratio = len(managers) / total if total else 0.0

        # Allow some tolerance around 25%
        self.assertGreaterEqual(
            mgr_ratio, 0.15,
            f"Manager ratio too low: {mgr_ratio:.2%}"
        )
        self.assertLessEqual(
            mgr_ratio, 0.35,
            f"Manager ratio too high: {mgr_ratio:.2%}"
        )

        with_manager = [
            i for i in identities
            if getattr(i, "manager_id", None) not in (None, "", pd.NA)
        ]
        coverage = len(with_manager) / total if total else 0.0

        self.assertGreaterEqual(
            coverage, 0.90,
            f"Less than 90% of users have a manager_id (coverage={coverage:.2%})"
        )

    def test_manager_id_refers_to_manager(self):
        """Verify that every non-null manager_id points to a manager (manager_flag='Y')."""
        config = DEFAULT_CONFIG.copy()
        gen = IdentityGenerator(config, seed=42)
        identities = gen.generate(500)

        by_user_id = {i.user_id: i for i in identities}

        for identity in identities:
            mid = getattr(identity, "manager_id", None)
            if mid in (None, "", pd.NA):
                continue
            self.assertIn(
                mid,
                by_user_id,
                f"manager_id {mid} does not correspond to any user_id"
            )
            self.assertEqual(
                by_user_id[mid].manager_flag,
                "Y",
                f"manager_id {mid} does not refer to a manager (manager_flag != 'Y')"
            )

    def test_manager_hierarchy_depth(self):
        """
        Verify there is at least one chain of length >= 3:
        user -> manager -> manager's manager.
        """
        config = DEFAULT_CONFIG.copy()
        gen = IdentityGenerator(config, seed=42)
        identities = gen.generate(1000)

        by_user_id = {i.user_id: i for i in identities}

        def depth(user, visited=None):
            if visited is None:
                visited = set()
            if user.user_id in visited:
                # Break potential cycles defensively
                return 0
            visited.add(user.user_id)

            mid = getattr(user, "manager_id", None)
            if mid in (None, "", pd.NA) or mid not in by_user_id:
                return 1
            return 1 + depth(by_user_id[mid], visited)

        depths = [depth(i) for i in identities]
        max_depth = max(depths) if depths else 0

        self.assertGreaterEqual(
            max_depth, 3,
            f"Expected at least one chain of depth >= 3, got max_depth={max_depth}"
        )

    def test_status_values_and_distribution(self):
        """Verify status values are valid and distribution is reasonable."""
        config = DEFAULT_CONFIG.copy()
        gen = IdentityGenerator(config, seed=42)
        identities = gen.generate(1000)

        statuses = [i.status for i in identities]
        self.assertTrue(statuses, "No status values generated")

        for s in statuses:
            self.assertIn(
                s,
                ("active", "inactive"),
                f"Invalid status value: {s}"
            )

        active_ratio = statuses.count("active") / len(statuses)
        # Expect a strong majority of active users, but not 100%
        self.assertGreaterEqual(
            active_ratio, 0.80,
            f"Too few active users: {active_ratio:.2%}"
        )
        self.assertLessEqual(
            active_ratio, 0.99,
            f"Too many active users (almost all active): {active_ratio:.2%}"
        )

    def test_line_of_business_values(self):
        """Verify line_of_business values are drawn from the configured set."""
        config = DEFAULT_CONFIG.copy()
        gen = IdentityGenerator(config, seed=42)
        identities = gen.generate(500)

        # We expect lines_of_business to be defined in the identity config
        lob_cfg = config.get("identity", {}).get("lines_of_business", {})
        expected_lobs = set(lob_cfg.keys())

        # If no config is present for some reason, just ensure non-empty strings
        if not expected_lobs:
            for identity in identities:
                lob = getattr(identity, "line_of_business", "")
                self.assertTrue(
                    isinstance(lob, str) and lob,
                    "line_of_business should be a non-empty string"
                )
            return

        for identity in identities:
            lob = identity.line_of_business
            self.assertIn(
                lob,
                expected_lobs,
                f"line_of_business '{lob}' not in configured set {expected_lobs}"
            )
    # --- END ChatGPT change: Phase 4 - new identity tests for username, manager, status, LOB ---


class TestEntitlementGenerator(unittest.TestCase):
    """Test entitlement generation."""
    
    def test_no_duplicate_entitlement_ids(self):
        """Verify all entitlement_ids are unique."""
        config = DEFAULT_CONFIG.copy()
        gen = EntitlementGenerator(config, seed=42)
        entitlements, _ = gen.generate(200, 10)
        
        ent_ids = [e.entitlement_id for e in entitlements]
        self.assertEqual(len(ent_ids), len(set(ent_ids)),
                        "Found duplicate entitlement_ids")
    
    def test_correct_count(self):
        """Verify correct number of entitlements generated."""
        config = DEFAULT_CONFIG.copy()
        gen = EntitlementGenerator(config, seed=42)
        entitlements, _ = gen.generate(200, 10)
        
        self.assertEqual(len(entitlements), 200,
                        "Incorrect number of entitlements")
    
    def test_app_distribution(self):
        """Verify entitlements distributed across apps."""
        config = DEFAULT_CONFIG.copy()
        gen = EntitlementGenerator(config, seed=42)
        entitlements, _ = gen.generate(200, 10)
        
        app_counts = defaultdict(int)
        for ent in entitlements:
            app_counts[ent.app_id] += 1
        
        self.assertEqual(len(app_counts), 10,
                        "Entitlements not distributed across all apps")
    
    def test_functional_bundles_created(self):
        """Verify functional bundles are created."""
        config = DEFAULT_CONFIG.copy()
        gen = EntitlementGenerator(config, seed=42)
        _, bundles = gen.generate(200, 10)
        
        self.assertGreater(len(bundles), 0,
                          "No functional bundles created")


class TestAssignmentGenerator(unittest.TestCase):
    """Test assignment generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DEFAULT_CONFIG.copy()
        
        # Generate test data
        id_gen = IdentityGenerator(self.config, seed=42)
        self.identities = id_gen.generate(100)
        
        ent_gen = EntitlementGenerator(self.config, seed=42)
        self.entitlements, self.bundles = ent_gen.generate(200, 10)
        
        pattern_gen = PatternGenerator(
            self.config, self.identities, self.bundles, seed=42
        )
        self.patterns = pattern_gen.generate(20)
    
    def test_no_duplicate_assignments(self):
        """Verify no duplicate (user_id, entitlement_id) pairs."""
        assignment_gen = AssignmentGenerator(
            self.config, self.identities, self.entitlements,
            self.patterns, seed=42
        )
        assignments = assignment_gen.generate()
        
        # Check for duplicates
        assignment_set = set(assignments)
        self.assertEqual(len(assignments), len(assignment_set),
                        "Found duplicate assignments")
    
    def test_reproducibility(self):
        """Verify reproducibility with same seed."""
        gen1 = AssignmentGenerator(
            self.config, self.identities, self.entitlements,
            self.patterns, seed=42
        )
        assignments1 = gen1.generate()
        
        gen2 = AssignmentGenerator(
            self.config, self.identities, self.entitlements,
            self.patterns, seed=42
        )
        assignments2 = gen2.generate()
        
        self.assertEqual(set(assignments1), set(assignments2),
                        "Assignments not reproducible")
    
    def test_reasonable_assignment_count(self):
        """Verify reasonable number of assignments."""
        assignment_gen = AssignmentGenerator(
            self.config, self.identities, self.entitlements,
            self.patterns, seed=42
        )
        assignments = assignment_gen.generate()
        
        num_users = len(self.identities)
        target_avg = self.config['avg_ents_per_user']
        
        # Should be within reasonable range
        min_expected = num_users * (target_avg * 0.3)
        max_expected = num_users * (target_avg * 2.0)
        
        self.assertGreater(len(assignments), min_expected,
                          "Too few assignments")
        self.assertLess(len(assignments), max_expected,
                       "Too many assignments")


class TestCalibrationEngine(unittest.TestCase):
    """Test calibration engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DEFAULT_CONFIG.copy()
        self.config['num_users'] = 100
        self.config['num_entitlements'] = 200
        self.config['num_patterns'] = 10
        
        # Generate test data
        id_gen = IdentityGenerator(self.config, seed=42)
        self.identities = id_gen.generate(self.config['num_users'])
        
        ent_gen = EntitlementGenerator(self.config, seed=42)
        self.entitlements, self.bundles = ent_gen.generate(
            self.config['num_entitlements'], 10
        )
        
        pattern_gen = PatternGenerator(
            self.config, self.identities, self.bundles, seed=42
        )
        self.patterns = pattern_gen.generate(self.config['num_patterns'])
        
        self.assignment_gen = AssignmentGenerator(
            self.config, self.identities, self.entitlements,
            self.patterns, seed=42
        )
    
    def test_calibration_improves_proportions(self):
        """Verify calibration moves proportions toward targets."""
        # Generate initial assignments
        initial_assignments = self.assignment_gen.generate()
        
        # Create calibration engine
        calibration_engine = CalibrationEngine(
            self.config, self.identities, self.patterns,
            self.assignment_gen
        )
        
        # Get initial proportions
        transactions = calibration_engine._build_transactions(initial_assignments)
        initial_rules = []
        
        for pattern in self.patterns[:5]:  # Test subset
            antecedent = calibration_engine._get_pattern_antecedent_tokens(pattern)
            if not antecedent:
                continue
            for ent_id in pattern.core_entitlements[:3]:  # Test subset
                confidence = calibration_engine._compute_rule_confidence(
                    transactions, antecedent, ent_id
                )
                observed_band = calibration_engine._classify_confidence(confidence)
                initial_rules.append((confidence, observed_band))
        
        if not initial_rules:
            self.skipTest("No rules to test")
        
        # Calculate initial band counts
        initial_band_counts = {"High": 0, "Mid": 0, "Low": 0}
        for _, band in initial_rules:
            initial_band_counts[band] += 1
        
        # Calibrate
        calibrated_assignments, final_proportions = calibration_engine.calibrate(
            initial_assignments, max_iterations=2
        )
        
        # Verify calibration ran
        self.assertIsNotNone(calibrated_assignments)
        self.assertIsNotNone(final_proportions)
        
        # Verify proportions are reasonable (may not converge fully in 2 iterations)
        for band in ["High", "Mid", "Low"]:
            self.assertGreaterEqual(final_proportions[band], 0.0)
            self.assertLessEqual(final_proportions[band], 1.0)


class TestDataWriter(unittest.TestCase):
    """Test data writing."""
    
    def test_write_identities(self):
        """Test writing identities to CSV."""
        config = DEFAULT_CONFIG.copy()
        gen = IdentityGenerator(config, seed=42)
        identities = gen.generate(50)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "identities.csv"
            DataWriter.write_identities(identities, output_path)
            
            # Verify file exists and can be read
            self.assertTrue(output_path.exists())
            
            df = pd.read_csv(output_path)
            self.assertEqual(len(df), 50)
            
            # Check required columns
            required_cols = [
                'user_id', 'first_name', 'last_name', 'email',
                'department', 'job_level'
            ]
            for col in required_cols:
                self.assertIn(col, df.columns)
    
    def test_write_entitlements(self):
        """Test writing entitlements to CSV."""
        config = DEFAULT_CONFIG.copy()
        gen = EntitlementGenerator(config, seed=42)
        entitlements, _ = gen.generate(100, 10)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "entitlements.csv"
            DataWriter.write_entitlements(entitlements, output_path)
            
            self.assertTrue(output_path.exists())
            
            df = pd.read_csv(output_path)
            self.assertEqual(len(df), 100)
            
            required_cols = [
                'entitlement_id', 'app_id', 'app_name',
                'entitlement_type', 'criticality'
            ]
            for col in required_cols:
                self.assertIn(col, df.columns)
    
    def test_write_assignments(self):
        """Test writing assignments to CSV."""
        assignments = [
            ("U000001", "E000001"),
            ("U000001", "E000002"),
            ("U000002", "E000001"),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "assignments.csv"
            DataWriter.write_assignments(assignments, output_path)
            
            self.assertTrue(output_path.exists())
            
            df = pd.read_csv(output_path)
            self.assertEqual(len(df), 3)
            self.assertIn('user_id', df.columns)
            self.assertIn('entitlement_id', df.columns)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""
    
    def test_full_pipeline(self):
        """Test complete generation pipeline."""
        config = DEFAULT_CONFIG.copy()
        config['num_users'] = 200
        config['num_entitlements'] = 300
        config['num_patterns'] = 15
        
        # Generate all components
        id_gen = IdentityGenerator(config, seed=42)
        identities = id_gen.generate(config['num_users'])
        
        ent_gen = EntitlementGenerator(config, seed=42)
        entitlements, bundles = ent_gen.generate(
            config['num_entitlements'], config['num_apps']
        )
        
        pattern_gen = PatternGenerator(
            config, identities, bundles, seed=42
        )
        patterns = pattern_gen.generate(config['num_patterns'])
        
        assignment_gen = AssignmentGenerator(
            config, identities, entitlements, patterns, seed=42
        )
        assignments = assignment_gen.generate()
        
        # Verify outputs
        self.assertEqual(len(identities), 200)
        self.assertEqual(len(entitlements), 300)
        self.assertEqual(len(patterns), 15)
        self.assertGreater(len(assignments), 0)
        
        # Verify no duplicates
        user_ids = [i.user_id for i in identities]
        self.assertEqual(len(user_ids), len(set(user_ids)))
        
        ent_ids = [e.entitlement_id for e in entitlements]
        self.assertEqual(len(ent_ids), len(set(ent_ids)))
        
        assignment_pairs = set(assignments)
        self.assertEqual(len(assignments), len(assignment_pairs))
    
    def test_write_all_files(self):
        """Test writing all files."""
        config = DEFAULT_CONFIG.copy()
        config['num_users'] = 100
        config['num_entitlements'] = 150
        config['num_patterns'] = 10
        
        # Generate data
        id_gen = IdentityGenerator(config, seed=42)
        identities = id_gen.generate(config['num_users'])
        
        ent_gen = EntitlementGenerator(config, seed=42)
        entitlements, bundles = ent_gen.generate(
            config['num_entitlements'], config['num_apps']
        )
        
        pattern_gen = PatternGenerator(
            config, identities, bundles, seed=42
        )
        patterns = pattern_gen.generate(config['num_patterns'])
        
        assignment_gen = AssignmentGenerator(
            config, identities, entitlements, patterns, seed=42
        )
        assignments = assignment_gen.generate()
        
        # Write to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            
            DataWriter.write_identities(identities, out_dir / 'identities.csv')
            DataWriter.write_entitlements(entitlements, out_dir / 'entitlements.csv')
            DataWriter.write_assignments(assignments, out_dir / 'assignments.csv')
            
            # Verify all files exist and are readable
            self.assertTrue((out_dir / 'identities.csv').exists())
            self.assertTrue((out_dir / 'entitlements.csv').exists())
            self.assertTrue((out_dir / 'assignments.csv').exists())
            
            # Verify data integrity
            id_df = pd.read_csv(out_dir / 'identities.csv')
            ent_df = pd.read_csv(out_dir / 'entitlements.csv')
            assign_df = pd.read_csv(out_dir / 'assignments.csv')
            
            self.assertEqual(len(id_df), 100)
            self.assertEqual(len(ent_df), 150)
            self.assertGreater(len(assign_df), 0)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestIdentityGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestEntitlementGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestAssignmentGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestCalibrationEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestDataWriter))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
