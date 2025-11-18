#!/usr/bin/env python3
"""
Example usage of the synthetic IGA dataset generator.

This script demonstrates various ways to use the generator programmatically.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_synthetic_iga import (
    DEFAULT_CONFIG,
    IdentityGenerator,
    EntitlementGenerator,
    PatternGenerator,
    AssignmentGenerator,
    CalibrationEngine,
    DataWriter,
    QAReporter
)


def example_basic_generation():
    """Example 1: Basic dataset generation."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Dataset Generation")
    print("=" * 70)
    
    config = DEFAULT_CONFIG.copy()
    config['num_users'] = 500
    config['num_entitlements'] = 400
    config['num_patterns'] = 20
    
    seed = 42
    
    # Generate identities
    print("Generating identities...")
    id_gen = IdentityGenerator(config, seed=seed)
    identities = id_gen.generate(config['num_users'])
    print(f"Generated {len(identities)} identities")
    
    # Generate entitlements
    print("Generating entitlements...")
    ent_gen = EntitlementGenerator(config, seed=seed)
    entitlements, bundles = ent_gen.generate(
        config['num_entitlements'],
        config['num_apps']
    )
    print(f"Generated {len(entitlements)} entitlements and {len(bundles)} bundles")
    
    # Generate patterns
    print("Generating patterns...")
    pattern_gen = PatternGenerator(config, identities, bundles, seed=seed)
    patterns = pattern_gen.generate(config['num_patterns'])
    print(f"Generated {len(patterns)} patterns")
    
    # Generate assignments
    print("Generating assignments...")
    assignment_gen = AssignmentGenerator(
        config, identities, entitlements, patterns, seed=seed
    )
    assignments = assignment_gen.generate()
    print(f"Generated {len(assignments)} assignments")
    
    # Calibrate (optional but recommended)
    print("Calibrating...")
    calibration_engine = CalibrationEngine(
        config, identities, patterns, assignment_gen
    )
    assignments, observed_proportions = calibration_engine.calibrate(
        assignments, max_iterations=2
    )
    
    print(f"\nObserved confidence proportions:")
    for band, prop in observed_proportions.items():
        print(f"  {band}: {prop:.2%}")
    
    print("\n")


def example_custom_configuration():
    """Example 2: Custom configuration."""
    print("=" * 70)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 70)
    
    # Create custom config
    config = {
        "num_users": 1000,
        "num_apps": 8,
        "num_entitlements": 800,
        "avg_ents_per_user": 25,
        "departments": ["Engineering", "Sales", "Marketing", "Finance"],
        "business_units": ["North", "South", "East", "West"],
        "job_families": ["SWE", "PM", "AE", "Finance"],
        "countries": ["US", "UK", "DE"],
        "employment_types": {
            "FTE": 0.90,
            "Contractor": 0.08,
            "Intern": 0.02
        },
        "criticality_distribution": {
            "High": 0.15,
            "Medium": 0.35,
            "Low": 0.50
        },
        "num_patterns": 30,
        "target_confidence_bands": {
            "High": 0.65,
            "Mid": 0.25,
            "Low": 0.10
        },
        "band_confidence_ranges": {
            "High": [0.70, 1.0],
            "Mid": [0.40, 0.70],
            "Low": [0.0, 0.40]
        },
        "calibration_tolerance": 0.05,
        "max_calibration_iterations": 3,
        "background_noise_rate": 0.04,
        "outlier_rate": 0.02
    }
    
    print("Custom configuration:")
    print(json.dumps(config, indent=2))
    print("\n")
    
    seed = 123
    
    # Generate with custom config
    id_gen = IdentityGenerator(config, seed=seed)
    identities = id_gen.generate(config['num_users'])
    
    ent_gen = EntitlementGenerator(config, seed=seed)
    entitlements, bundles = ent_gen.generate(
        config['num_entitlements'],
        config['num_apps']
    )
    
    pattern_gen = PatternGenerator(config, identities, bundles, seed=seed)
    patterns = pattern_gen.generate(config['num_patterns'])
    
    assignment_gen = AssignmentGenerator(
        config, identities, entitlements, patterns, seed=seed
    )
    assignments = assignment_gen.generate()
    
    print(f"Generated dataset with {len(identities)} users, "
          f"{len(entitlements)} entitlements, {len(assignments)} assignments")
    print("\n")


def example_write_to_files():
    """Example 3: Write dataset to files."""
    print("=" * 70)
    print("EXAMPLE 3: Write Dataset to Files")
    print("=" * 70)
    
    config = DEFAULT_CONFIG.copy()
    config['num_users'] = 300
    config['num_entitlements'] = 250
    config['num_patterns'] = 15
    
    seed = 42
    
    # Generate data
    id_gen = IdentityGenerator(config, seed=seed)
    identities = id_gen.generate(config['num_users'])
    
    ent_gen = EntitlementGenerator(config, seed=seed)
    entitlements, bundles = ent_gen.generate(
        config['num_entitlements'],
        config['num_apps']
    )
    
    pattern_gen = PatternGenerator(config, identities, bundles, seed=seed)
    patterns = pattern_gen.generate(config['num_patterns'])
    
    assignment_gen = AssignmentGenerator(
        config, identities, entitlements, patterns, seed=seed
    )
    assignments = assignment_gen.generate()
    
    calibration_engine = CalibrationEngine(
        config, identities, patterns, assignment_gen
    )
    assignments, observed_proportions = calibration_engine.calibrate(
        assignments, max_iterations=2
    )
    
    # Write to files
    output_dir = Path("./example_output")
    output_dir.mkdir(exist_ok=True)
    
    DataWriter.write_identities(identities, output_dir / "identities.csv")
    DataWriter.write_entitlements(entitlements, output_dir / "entitlements.csv")
    DataWriter.write_assignments(assignments, output_dir / "assignments.csv")
    
    print(f"Dataset written to {output_dir}/")
    print(f"  - identities.csv: {len(identities)} rows")
    print(f"  - entitlements.csv: {len(entitlements)} rows")
    print(f"  - assignments.csv: {len(assignments)} rows")
    
    # Generate QA report
    print("\n")
    QAReporter.report(identities, entitlements, assignments, observed_proportions)


def example_analyze_patterns():
    """Example 4: Analyze generated patterns."""
    print("=" * 70)
    print("EXAMPLE 4: Analyze Generated Patterns")
    print("=" * 70)
    
    config = DEFAULT_CONFIG.copy()
    config['num_users'] = 200
    config['num_entitlements'] = 200
    config['num_patterns'] = 10
    
    seed = 42
    
    # Generate data
    id_gen = IdentityGenerator(config, seed=seed)
    identities = id_gen.generate(config['num_users'])
    
    ent_gen = EntitlementGenerator(config, seed=seed)
    entitlements, bundles = ent_gen.generate(
        config['num_entitlements'],
        config['num_apps']
    )
    
    pattern_gen = PatternGenerator(config, identities, bundles, seed=seed)
    patterns = pattern_gen.generate(config['num_patterns'])
    
    # Analyze patterns
    print(f"Generated {len(patterns)} patterns\n")
    
    for i, pattern in enumerate(patterns[:5], 1):  # Show first 5
        print(f"Pattern {i} ({pattern.pattern_id}):")
        print(f"  Target Band: {pattern.target_band}")
        print(f"  Predicates: {pattern.feature_predicates}")
        print(f"  Core Entitlements: {len(pattern.core_entitlements)}")
        print(f"  Optional Entitlements: {len(pattern.optional_entitlements)}")
        print(f"  Paired Entitlements: {len(pattern.paired_entitlements)}")
        print(f"  Super Bundle Prob: {pattern.super_bundle_prob:.2f}")
        print()


def main():
    """Run all examples."""
    print("\n")
    print("*" * 70)
    print("Synthetic IGA Dataset Generator - Usage Examples")
    print("*" * 70)
    print("\n")
    
    # Run examples
    example_basic_generation()
    example_custom_configuration()
    example_write_to_files()
    example_analyze_patterns()
    
    print("\n")
    print("*" * 70)
    print("Examples Complete!")
    print("*" * 70)
    print("\n")


if __name__ == '__main__':
    main()
