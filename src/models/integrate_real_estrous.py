"""
INTEGRATE REAL ESTROUS DATA WITH MORPHINE MODEL

Steps:
1. Analyze p_lowU estrous probabilities
2. Determine estrous phase at dosing
3. Create new estrous features
4. Compare: Baseline (24 feat) vs Baseline+Real Estrous (24+4 feat)
5. Test if REAL estrous improves prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("INTEGRATING REAL ESTROUS DATA")
print("="*80)

# Load data
estrous_df = pd.read_csv('p_lowU_clipped_table.csv')
responses_df = pd.read_csv('morphine_responses.csv')

print(f"\nEstrous data: {len(estrous_df)} rows")
print(f"Response data: {len(responses_df)} animals")

# Explore estrous data
print("\n" + "="*80)
print("ESTROUS DATA EXPLORATION")
print("="*80)

print(f"\nUnique animals in estrous data: {estrous_df['animal_id'].nunique()}")
print(f"Groups: {estrous_df['group'].unique()}")
print(f"Days recorded: {estrous_df['day'].min()}-{estrous_df['day'].max()}")

# p_lowU statistics
print(f"\np_lowU statistics:")
print(f"  Mean: {estrous_df['p_lowU'].mean():.4f}")
print(f"  Median: {estrous_df['p_lowU'].median():.4f}")
print(f"  Range: [{estrous_df['p_lowU'].min():.2e}, {estrous_df['p_lowU'].max():.4f}]")

# Classify estrous days (high p_lowU = estrous)
threshold_estrous = 0.5
estrous_df['is_estrous_day'] = (estrous_df['p_lowU'] > threshold_estrous).astype(int)

print(f"\nEstrous day classification (threshold={threshold_estrous}):")
print(f"  Estrous days: {estrous_df['is_estrous_day'].sum()}/{len(estrous_df)} ({estrous_df['is_estrous_day'].mean()*100:.1f}%)")

# Check cycling by animal
print("\nCycling detection by animal (% estrous days):")
cycling_by_animal = estrous_df.groupby('animal_id').agg({
    'is_estrous_day': 'mean',
    'p_lowU': 'mean'
}).sort_values('is_estrous_day', ascending=False)

print("Top 5 most cycling:")
print(cycling_by_animal.head())
print("\nBottom 5 least cycling:")
print(cycling_by_animal.tail())

# Animals with strong cycling
strong_cycling = cycling_by_animal[cycling_by_animal['is_estrous_day'] > 0.3].index.tolist()
print(f"\nAnimals with >30% estrous days: {len(strong_cycling)}")

# ============================================================================
# DETERMINE DOSING DAY
# ============================================================================
print("\n" + "="*80)
print("DETERMINING DOSING DAY")
print("="*80)

# Morphine was likely given after baseline recording
# Baseline = days 2-5 or 2-6, then dose on day 6 or 7
# Let's check what days are available

print("\nDays available per animal:")
days_per_animal = estrous_df.groupby('animal_id')['day'].agg(['min', 'max', 'count'])
print(days_per_animal.describe())

# Assume dosing day is around day 6-7 (after 4-5 days baseline)
# Let's use day 7 as likely dosing day
DOSING_DAY = 7

print(f"\nAssuming dosing day: {DOSING_DAY}")
print("(This is when morphine was administered)")

# ============================================================================
# CREATE ESTROUS FEATURES
# ============================================================================
print("\n" + "="*80)
print("CREATING ESTROUS FEATURES")
print("="*80)

# Feature 1: p_lowU at dosing day
estrous_at_dose = estrous_df[estrous_df['day'] == DOSING_DAY][['animal_id', 'p_lowU']].copy()
estrous_at_dose.rename(columns={'p_lowU': 'p_lowU_at_dose'}, inplace=True)

print(f"\nAnimals with dosing day data: {len(estrous_at_dose)}")

# Feature 2: Average p_lowU during baseline (days 2-6)
baseline_days = range(2, DOSING_DAY)
baseline_estrous = estrous_df[estrous_df['day'].isin(baseline_days)].groupby('animal_id').agg({
    'p_lowU': ['mean', 'std', 'max']
}).reset_index()
baseline_estrous.columns = ['animal_id', 'baseline_p_lowU_mean', 'baseline_p_lowU_std', 'baseline_p_lowU_max']

# Feature 3: Cycling strength (how variable is p_lowU?)
# High variability = strong cycling
baseline_estrous['cycling_strength_real'] = baseline_estrous['baseline_p_lowU_std']

# Feature 4: Number of estrous days in baseline
estrous_days_baseline = estrous_df[estrous_df['day'].isin(baseline_days)].groupby('animal_id')['is_estrous_day'].sum().reset_index()
estrous_days_baseline.rename(columns={'is_estrous_day': 'n_estrous_days_baseline'}, inplace=True)

# Feature 5: Binary - was animal in estrous at dosing?
estrous_at_dose['in_estrous_at_dose'] = (estrous_at_dose['p_lowU_at_dose'] > threshold_estrous).astype(int)

# Merge all features
estrous_features = estrous_at_dose.merge(baseline_estrous, on='animal_id', how='outer')
estrous_features = estrous_features.merge(estrous_days_baseline, on='animal_id', how='outer')

print(f"\nCreated estrous features for {len(estrous_features)} animals")
print("\nFeatures created:")
print("  1. p_lowU_at_dose - Estrous probability at dosing")
print("  2. in_estrous_at_dose - Binary: In estrous? (1=yes, 0=no)")
print("  3. baseline_p_lowU_mean - Average estrous prob during baseline")
print("  4. baseline_p_lowU_std - Variability (cycling strength)")
print("  5. baseline_p_lowU_max - Peak estrous probability")
print("  6. cycling_strength_real - Same as std (for compatibility)")
print("  7. n_estrous_days_baseline - Count of estrous days")

print("\nEstrous feature statistics:")
print(estrous_features[['p_lowU_at_dose', 'baseline_p_lowU_mean', 'cycling_strength_real']].describe())

print(f"\nAnimals in estrous at dosing: {estrous_features['in_estrous_at_dose'].sum()}/{len(estrous_features)} ({estrous_features['in_estrous_at_dose'].mean()*100:.1f}%)")

# ============================================================================
# MERGE WITH MORPHINE DATA
# ============================================================================
print("\n" + "="*80)
print("MERGING WITH MORPHINE RESPONSE DATA")
print("="*80)

# Load full features
features_df = pd.read_csv('animal_features_with_estrus.csv')

# Merge responses with features
data_full = features_df.merge(responses_df, on='animal_id', how='inner')

print(f"\nMorphine data (before merging estrous): {len(data_full)} animals")

# Merge with real estrous
data_with_estrous = data_full.merge(estrous_features, on='animal_id', how='left')

print(f"After merging real estrous: {len(data_with_estrous)} animals")
print(f"Animals with estrous data: {data_with_estrous['p_lowU_at_dose'].notna().sum()}")

# Check which animals are missing estrous data
missing_estrous = data_with_estrous[data_with_estrous['p_lowU_at_dose'].isna()]['animal_id'].tolist()
if len(missing_estrous) > 0:
    print(f"\n⚠ Animals missing estrous data: {missing_estrous}")
    print("  These will be excluded from estrous analysis")

# Save merged data
data_with_estrous.to_csv('morphine_with_real_estrous.csv', index=False)
print("\n✓ Saved: morphine_with_real_estrous.csv")

# ============================================================================
# VISUALIZE ESTROUS DATA
# ============================================================================
print("\n" + "="*80)
print("VISUALIZING ESTROUS PATTERNS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Panel A: p_lowU distribution
ax = axes[0, 0]
ax.hist(estrous_df['p_lowU'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(threshold_estrous, color='red', linestyle='--', linewidth=2, 
          label=f'Threshold ({threshold_estrous})')
ax.set_xlabel('p_lowU (Estrous Probability)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('A. Distribution of Estrous Probabilities', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel B: Cycling by animal
ax = axes[0, 1]
cycling_pct = cycling_by_animal['is_estrous_day'].values * 100
ax.hist(cycling_pct, bins=20, edgecolor='black', alpha=0.7, color='coral')
ax.axvline(50, color='red', linestyle='--', linewidth=2, label='50% threshold')
ax.set_xlabel('% Estrous Days', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Animals', fontsize=11, fontweight='bold')
ax.set_title('B. Cycling Strength by Animal', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel C: Example cycling pattern (pick a strong cycler)
ax = axes[0, 2]
if len(strong_cycling) > 0:
    example_animal = strong_cycling[0]
    animal_data = estrous_df[estrous_df['animal_id'] == example_animal].sort_values('day')
    ax.plot(animal_data['day'], animal_data['p_lowU'], marker='o', linewidth=2, markersize=8)
    ax.axhline(threshold_estrous, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.fill_between(animal_data['day'], 0, 1, 
                    where=animal_data['p_lowU'] > threshold_estrous,
                    alpha=0.3, color='pink', label='Estrous')
    ax.set_xlabel('Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('p_lowU', fontsize=11, fontweight='bold')
    ax.set_title(f'C. Example Cycling Pattern\nAnimal {example_animal}', 
                fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'No strong cyclers\ndetected', 
           transform=ax.transAxes, ha='center', va='center', fontsize=12)

# Panel D: Estrous at dosing vs response (if data available)
ax = axes[1, 0]
data_plot = data_with_estrous.dropna(subset=['p_lowU_at_dose', 'response_pct'])
if len(data_plot) > 0:
    colors_dose = ['red' if d==5 else 'blue' for d in data_plot['dose_mg_kg_y']]
    ax.scatter(data_plot['p_lowU_at_dose'], data_plot['response_pct'],
              c=colors_dose, alpha=0.6, s=100, edgecolors='black', linewidth=1)
    ax.set_xlabel('p_lowU at Dosing', fontsize=11, fontweight='bold')
    ax.set_ylabel('Morphine Response (%)', fontsize=11, fontweight='bold')
    ax.set_title('D. Estrous at Dose vs Response', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='5 mg/kg'),
                      Patch(facecolor='blue', label='25 mg/kg')]
    ax.legend(handles=legend_elements)
else:
    ax.text(0.5, 0.5, 'No merged data\navailable yet',
           transform=ax.transAxes, ha='center', va='center', fontsize=12)

# Panel E: Cycling strength vs response
ax = axes[1, 1]
data_plot = data_with_estrous.dropna(subset=['cycling_strength_real', 'response_pct'])
if len(data_plot) > 0:
    colors_dose = ['red' if d==5 else 'blue' for d in data_plot['dose_mg_kg_y']]
    ax.scatter(data_plot['cycling_strength_real'], data_plot['response_pct'],
              c=colors_dose, alpha=0.6, s=100, edgecolors='black', linewidth=1)
    ax.set_xlabel('Cycling Strength (Std of p_lowU)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Morphine Response (%)', fontsize=11, fontweight='bold')
    ax.set_title('E. Cycling Strength vs Response', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    legend_elements = [Patch(facecolor='red', label='5 mg/kg'),
                      Patch(facecolor='blue', label='25 mg/kg')]
    ax.legend(handles=legend_elements)
else:
    ax.text(0.5, 0.5, 'No merged data\navailable yet',
           transform=ax.transAxes, ha='center', va='center', fontsize=12)

# Panel F: Summary stats
ax = axes[1, 2]
if len(data_with_estrous.dropna(subset=['p_lowU_at_dose'])) > 0:
    summary_text = f"""
Real Estrous Data Summary

Total animals: {len(estrous_features)}
With estrous data: {estrous_features['p_lowU_at_dose'].notna().sum()}

At dosing (day {DOSING_DAY}):
  In estrous: {estrous_features['in_estrous_at_dose'].sum()}
  Not in estrous: {(~estrous_features['in_estrous_at_dose'].astype(bool)).sum()}
  
Baseline cycling:
  Mean p_lowU: {estrous_features['baseline_p_lowU_mean'].mean():.3f}
  Strong cyclers: {len(strong_cycling)}
  
Next: Test if this improves
morphine response prediction!
"""
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
else:
    ax.text(0.5, 0.5, 'Calculating\nsummary stats...',
           transform=ax.transAxes, ha='center', va='center', fontsize=12)
ax.axis('off')

plt.tight_layout()
plt.savefig('real_estrous_exploration.png', dpi=300, bbox_inches='tight')
print("✓ Saved: real_estrous_exploration.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n✓ Successfully integrated real estrous data!")
print(f"\nEstrous features created:")
print(f"  - p_lowU_at_dose: Estrous prob at dosing (day {DOSING_DAY})")
print(f"  - in_estrous_at_dose: Binary estrous status")
print(f"  - baseline_p_lowU_mean: Average baseline estrous")
print(f"  - cycling_strength_real: Cycling variability")
print(f"  - baseline_p_lowU_max: Peak estrous probability")
print(f"  - n_estrous_days_baseline: Estrous day count")

print(f"\nData saved:")
print(f"  ✓ morphine_with_real_estrous.csv")
print(f"  ✓ real_estrous_exploration.png")

print(f"\n" + "="*80)
print("NEXT STEPS")
print(f"="*80)
print("""
Now that you have REAL estrous data:

1. Run comparison: Baseline (24) vs Baseline+RealEstrous (24+X)
   → Test if real estrous improves 72.2% baseline

2. Test estrous × dose interaction
   → Does estrous affect 5mg vs 25mg differently?

3. Compare to old (broken) estrous features
   → Real estrous should work better!

Ready to test! Your hypothesis about estrous and opioid sensitivity
can now be properly evaluated with accurate data.
""")

print("="*80)
