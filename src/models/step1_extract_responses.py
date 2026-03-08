"""
STEP 1: Extract Morphine Responses

Purpose: For each morphine-treated animal, calculate response as:
         (post-dose activity - baseline activity) / baseline activity × 100%

Input:  S3 data from Morph2REP study
Output: morphine_responses.csv

Expected: 36 animals (18 at 5mg, 18 at 25mg)
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 80)
print("STEP 1: EXTRACT MORPHINE RESPONSES")
print("=" * 80)

# Study configuration
STUDY_CONFIG = {
    'rep1': {
        'baseline_dates': ['2025-01-10', '2025-01-11', '2025-01-12', '2025-01-13'],
        'doses': [
            {'date': datetime(2025, 1, 14, 6, 0, 0), 'dose_mg_kg': 5, 'cages': [4917, 4921, 4925]},
            {'date': datetime(2025, 1, 17, 17, 0, 0), 'dose_mg_kg': 25, 'cages': [4919, 4920, 4924]}
        ]
    },
    'rep2': {
        'baseline_dates': ['2025-01-25', '2025-01-26', '2025-01-27'],
        'doses': [
            {'date': datetime(2025, 1, 28, 17, 0, 0), 'dose_mg_kg': 5, 'cages': [4927, 4931, 4932]},
            {'date': datetime(2025, 1, 31, 6, 0, 0), 'dose_mg_kg': 25, 'cages': [4926, 4930, 4933]}
        ]
    }
}

def initialize_duckdb():
    """Initialize DuckDB connection"""
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs;")
    conn.execute("LOAD httpfs;")
    conn.execute("SET s3_region='us-east-1';")
    return conn

def get_baseline_activity(conn, cage_id, animal_id, dates):
    """Get baseline mean locomotion"""
    date_filter = " OR ".join([f"date = '{d}'" for d in dates])
    query = f"""
    SELECT AVG(value) as baseline_mean
    FROM read_parquet('s3://jax-envision-public-data/study_1001/2025v3.3/tabular/cage_id={cage_id}/date=*/animal_activity_db.parquet')
    WHERE ({date_filter})
        AND animal_id = {animal_id}
        AND name = 'animal_bouts.locomotion'
        AND resolution = 1
    """
    result = conn.execute(query).fetchdf()
    return result['baseline_mean'].iloc[0] if len(result) > 0 else None

def get_post_dose_activity(conn, cage_id, animal_id, dose_date):
    """Get activity during hyperlocomotion window (300-500 min post-dose)"""
    start_time = dose_date + timedelta(minutes=300)
    end_time = dose_date + timedelta(minutes=500)
    dose_date_str = dose_date.strftime('%Y-%m-%d')
    
    query = f"""
    SELECT AVG(value) as post_dose_mean
    FROM read_parquet('s3://jax-envision-public-data/study_1001/2025v3.3/tabular/cage_id={cage_id}/date={dose_date_str}/animal_activity_db.parquet')
    WHERE animal_id = {animal_id}
        AND name = 'animal_bouts.locomotion'
        AND resolution = 1
        AND CAST(time AS TIMESTAMP) >= CAST('{start_time.strftime('%Y-%m-%d %H:%M:%S')}' AS TIMESTAMP)
        AND CAST(time AS TIMESTAMP) < CAST('{end_time.strftime('%Y-%m-%d %H:%M:%S')}' AS TIMESTAMP)
    """
    result = conn.execute(query).fetchdf()
    return result['post_dose_mean'].iloc[0] if len(result) > 0 else None

def get_animal_ids(conn, cage_id, dates):
    """Get animal IDs in a cage"""
    date_filter = " OR ".join([f"date = '{d}'" for d in dates])
    query = f"""
    SELECT DISTINCT animal_id
    FROM read_parquet('s3://jax-envision-public-data/study_1001/2025v3.3/tabular/cage_id={cage_id}/date=*/animal_activity_db.parquet')
    WHERE ({date_filter})
        AND name = 'animal_bouts.locomotion'
    ORDER BY animal_id
    """
    result = conn.execute(query).fetchdf()
    return result['animal_id'].tolist()

# Initialize
conn = initialize_duckdb()
all_responses = []

# Extract responses
for rep_name, rep_config in STUDY_CONFIG.items():
    print(f"\n{'='*60}")
    print(f"Processing {rep_name.upper()}")
    print(f"{'='*60}")
    
    for dose_info in rep_config['doses']:
        dose_mg_kg = dose_info['dose_mg_kg']
        print(f"\n{dose_mg_kg} mg/kg dose group:")
        
        for cage_id in dose_info['cages']:
            animal_ids = get_animal_ids(conn, cage_id, rep_config['baseline_dates'])
            
            for animal_id in animal_ids:
                baseline = get_baseline_activity(conn, cage_id, animal_id, rep_config['baseline_dates'])
                post_dose = get_post_dose_activity(conn, cage_id, animal_id, dose_info['date'])
                
                if baseline and post_dose and baseline > 0:
                    response_pct = ((post_dose - baseline) / baseline) * 100
                    all_responses.append({
                        'animal_id': animal_id,
                        'cage_id': cage_id,
                        'replicate': 1 if rep_name == 'rep1' else 2,
                        'dose_mg_kg': dose_mg_kg,
                        'baseline_mean': baseline,
                        'post_dose_mean': post_dose,
                        'response_pct': response_pct
                    })

# Create DataFrame
df = pd.DataFrame(all_responses)

# Save
df.to_csv('morphine_responses.csv', index=False)

# SANITY CHECKS
print(f"\n{'='*80}")
print("SANITY CHECKS")
print(f"{'='*80}")

checks_passed = 0
total_checks = 5

# Check 1: Total count
n_animals = len(df)
print(f"\n✓ CHECK 1: Total animals extracted")
print(f"  Expected: 36")
print(f"  Actual: {n_animals}")
if n_animals == 36:
    print("  ✓ PASS")
    checks_passed += 1
else:
    print(f"  ✗ FAIL - Got {n_animals} animals")

# Check 2: Dose distribution
dose_counts = df['dose_mg_kg'].value_counts().sort_index()
print(f"\n✓ CHECK 2: Animals per dose")
print(f"  Expected: 18 at 5mg, 18 at 25mg")
print(f"  Actual:")
for dose, count in dose_counts.items():
    print(f"    {dose} mg/kg: {count} animals")
if len(dose_counts) == 2 and all(dose_counts == 18):
    print("  ✓ PASS")
    checks_passed += 1
else:
    print("  ✗ FAIL")

# Check 3: All responses positive
all_positive = (df['response_pct'] > -100).all()
print(f"\n✓ CHECK 3: Response values reasonable")
print(f"  All responses > -100%?: {all_positive}")
print(f"  Range: {df['response_pct'].min():.1f}% to {df['response_pct'].max():.1f}%")
if all_positive:
    print("  ✓ PASS")
    checks_passed += 1
else:
    print("  ✗ FAIL - Some responses are impossibly negative")

# Check 4: Dose groups different
mean_5mg = df[df['dose_mg_kg']==5]['response_pct'].mean()
mean_25mg = df[df['dose_mg_kg']==25]['response_pct'].mean()
print(f"\n✓ CHECK 4: Dose groups have different responses")
print(f"  5 mg/kg mean: {mean_5mg:.1f}%")
print(f"  25 mg/kg mean: {mean_25mg:.1f}%")
print(f"  Difference: {mean_25mg - mean_5mg:.1f}%")
if mean_25mg > mean_5mg * 2:  # 25mg should be ~5x higher
    print("  ✓ PASS - 25mg > 2x the 5mg response")
    checks_passed += 1
else:
    print("  ✗ FAIL - Dose groups not different enough")

# Check 5: No missing data
no_missing = not df.isnull().any().any()
print(f"\n✓ CHECK 5: No missing data")
print(f"  Any NaN values?: {df.isnull().any().any()}")
if no_missing:
    print("  ✓ PASS")
    checks_passed += 1
else:
    print("  ✗ FAIL - Missing data detected")

# Summary
print(f"\n{'='*80}")
print(f"STEP 1 COMPLETE: {checks_passed}/{total_checks} checks passed")
print(f"{'='*80}")

if checks_passed >= 4:
    print("✓ Data extraction successful - safe to proceed to Step 2")
else:
    print("⚠️  WARNING: Multiple checks failed - verify data before proceeding")

print(f"\nOutput: morphine_responses.csv ({n_animals} animals)")
print("\nNext step: python step2_regression_analysis.py")
