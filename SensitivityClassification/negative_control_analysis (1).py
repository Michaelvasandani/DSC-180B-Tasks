"""
NEGATIVE CONTROL ANALYSIS
Test if baseline activity predicts "response" in mice that never got morphine

Logic:
- Use 14-day activity data from untreated mice
- Days 1-7: Extract baseline features (same as morphine study)
- Days 8-14: Calculate fake "response" metric
- Test if baseline predicts fake response (should NOT)

This validates that the 25mg morphine finding isn't spurious correlation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from scipy.stats import mannwhitneyu
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("NEGATIVE CONTROL ANALYSIS")
print("="*80)
print("\nTesting: Does baseline activity predict 'response' in untreated mice?")
print("Expected result: NO (if our morphine finding is real)")
print("="*80)

# Load data
print("\nLoading data...")
df = pd.read_excel('Mouse_Data_Student_Copy.xlsx', sheet_name='Fem Act')
print(f"Data shape: {df.shape}")
print(f"Mice: {len(df.columns)}")
print(f"Timepoints: {len(df)} (14 days × 1440 min/day)")

# Split into baseline (days 1-7) and test (days 8-14)
MINS_PER_DAY = 1440
baseline_data = df.iloc[:7*MINS_PER_DAY].copy()  # Days 1-7
test_data = df.iloc[7*MINS_PER_DAY:].copy()      # Days 8-14

print(f"\nBaseline period: {len(baseline_data)} minutes (days 1-7)")
print(f"Test period: {len(test_data)} minutes (days 8-14)")

# Feature extraction function (same as morphine study)
def extract_features(data_col):
    """Extract same 28 features used in morphine study"""
    
    # Reshape to hourly bins for feature extraction
    hourly = data_col.values.reshape(-1, 60).sum(axis=1)  # 60 min -> 1 hour
    daily = hourly.reshape(-1, 24)  # 24 hours per day
    
    features = {}
    
    # Basic stats
    features['mean_locomotion'] = data_col.mean()
    features['std_locomotion'] = data_col.std()
    features['cv_locomotion'] = data_col.std() / (data_col.mean() + 1e-10)
    features['max_locomotion'] = data_col.max()
    features['min_locomotion'] = data_col.min()
    
    # Light/dark phase (assume lights on 6am-6pm, off 6pm-6am)
    light_phase = daily[:, 6:18]  # Hours 6-18
    dark_phase = daily[:, list(range(0,6)) + list(range(18,24))]  # Hours 0-6 and 18-24
    
    features['light_mean'] = light_phase.mean()
    features['dark_mean'] = dark_phase.mean()
    features['light_dark_ratio'] = light_phase.mean() / (dark_phase.mean() + 1e-10)
    
    # Circadian analysis (simplified)
    avg_day = daily.mean(axis=0)
    features['circadian_amplitude'] = avg_day.max() - avg_day.min()
    features['circadian_mesor'] = avg_day.mean()
    features['circadian_acrophase'] = avg_day.argmax()
    
    # Plateau (active phase variability)
    features['plateau_mean'] = dark_phase.mean()
    features['plateau_std'] = dark_phase.std()
    features['plateau_cv'] = dark_phase.std() / (dark_phase.mean() + 1e-10)
    features['plateau_max'] = dark_phase.max()
    
    # Ultradian (within-day patterns)
    features['ultradian_mean'] = hourly.mean()
    features['ultradian_std'] = hourly.std()
    features['ultradian_cv'] = hourly.std() / (hourly.mean() + 1e-10)
    
    # Circadian stats
    features['circadian_mean'] = daily.mean()
    features['circadian_std'] = daily.std()
    features['circadian_to_ultradian_ratio'] = daily.std() / (hourly.std() + 1e-10)
    
    # Hourly variability
    features['hourly_variability'] = np.diff(hourly).std()
    features['hourly_variability_std'] = np.diff(hourly).std()
    
    # Placeholder for estrus-related features (not available here)
    features['has_4day_cycle'] = 0
    features['cycling_strength'] = 0
    features['cycle_phase_at_dose'] = 0
    features['near_estrus_at_dose'] = 0
    
    # Circadian R-squared (goodness of fit)
    features['circadian_r_squared'] = 0.5  # Placeholder
    
    return features

# Extract baseline features for all mice
print("\nExtracting baseline features (days 1-7)...")
baseline_features = []
for col in df.columns:
    features = extract_features(baseline_data[col])
    features['mouse_id'] = col
    baseline_features.append(features)

baseline_df = pd.DataFrame(baseline_features)
print(f"Extracted {len(baseline_df.columns)-1} features from {len(baseline_df)} mice")

# Calculate fake "response" metric
# Use same approach: compare activity between baseline and test periods
print("\nCalculating fake 'response' metric...")
responses = []
for col in df.columns:
    baseline_activity = baseline_data[col].mean()
    test_activity = test_data[col].mean()
    
    # Same formula as morphine study
    response_pct = ((test_activity - baseline_activity) / (baseline_activity + 1e-10)) * 100
    responses.append({
        'mouse_id': col,
        'baseline_mean': baseline_activity,
        'test_mean': test_activity,
        'response_pct': response_pct
    })

response_df = pd.DataFrame(responses)
print(f"\nFake 'response' statistics:")
print(f"  Mean: {response_df['response_pct'].mean():.1f}%")
print(f"  Std: {response_df['response_pct'].std():.1f}%")
print(f"  Range: {response_df['response_pct'].min():.1f}% to {response_df['response_pct'].max():.1f}%")

# Merge features and responses
data = baseline_df.merge(response_df, on='mouse_id')

# Create labels (median split, same as morphine study)
median_response = data['response_pct'].median()
data['label'] = (data['response_pct'] >= median_response).astype(int)

print(f"\nMedian split at {median_response:.1f}%:")
print(f"  Low 'responders': {(data['label']==0).sum()} mice")
print(f"  High 'responders': {(data['label']==1).sum()} mice")

# Prepare for classification
feature_cols = [col for col in baseline_df.columns 
                if col not in ['mouse_id']]
X = data[feature_cols].fillna(data[feature_cols].median())
y = data['label'].values

print(f"\nFeatures: {len(feature_cols)}")
print(f"Samples: {len(X)}")

# Get response values for each group
low_resp = data[data['label']==0]['response_pct'].values
high_resp = data[data['label']==1]['response_pct'].values

# Mann-Whitney test
stat, p_bio = mannwhitneyu(low_resp, high_resp)
print(f"\nMann-Whitney U test:")
print(f"  Low group: {low_resp.mean():.1f}%")
print(f"  High group: {high_resp.mean():.1f}%")
print(f"  p-value: {p_bio:.6f}")
print(f"  Result: {'✓ SIGNIFICANT' if p_bio < 0.05 else '✗ NOT SIGNIFICANT'}")

# Check dark-phase activity
low_dark = data[data['label']==0]['dark_mean'].values
high_dark = data[data['label']==1]['dark_mean'].values
stat_dark, p_dark = mannwhitneyu(low_dark, high_dark)
print(f"\nBaseline dark-phase activity:")
print(f"  Low 'responders': {low_dark.mean():.4f}")
print(f"  High 'responders': {high_dark.mean():.4f}")
print(f"  p-value: {p_dark:.6f}")

# Classification with LOO-CV (no cages here, so use Leave-One-Out)
print("\n" + "="*80)
print("CLASSIFICATION TEST")
print("="*80)

loo = LeaveOneOut()
predictions = np.zeros(len(y))

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, 
                                 min_samples_leaf=2, random_state=42)
    clf.fit(X_train, y_train)
    predictions[test_idx] = clf.predict(X_test)

observed_accuracy = (predictions == y).mean()
print(f"\nLOO-CV Accuracy: {observed_accuracy:.3f} ({observed_accuracy*100:.1f}%)")

# Permutation test
print("\nRunning permutation test (1000 iterations)...")
n_perms = 1000
perm_accuracies = []

for i in range(n_perms):
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{n_perms}...")
    
    y_perm = np.random.permutation(y)
    preds = np.zeros(len(y))
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_perm[train_idx], y_perm[test_idx]
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, 
                                     min_samples_leaf=2, random_state=42)
        clf.fit(X_train, y_train)
        preds[test_idx] = clf.predict(X_test)
    
    perm_accuracies.append((preds == y_perm).mean())

perm_accuracies = np.array(perm_accuracies)
p_value = (perm_accuracies >= observed_accuracy).mean()
percentile = (perm_accuracies < observed_accuracy).mean() * 100

print(f"\nPermutation results:")
print(f"  Null mean: {perm_accuracies.mean():.3f}")
print(f"  Observed: {observed_accuracy:.3f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Percentile: {percentile:.1f}th")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Permutation distribution
axes[0].hist(perm_accuracies, bins=30, alpha=0.7, edgecolor='black', color='lightgray')
axes[0].axvline(observed_accuracy, color='red', linestyle='--', linewidth=2.5,
                label=f'Observed: {observed_accuracy:.3f}')
axes[0].axvline(0.5, color='green', linestyle=':', linewidth=2, alpha=0.7,
                label='Chance: 0.500')
axes[0].set_xlabel('Accuracy', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title(f'Negative Control Permutation Test\np = {p_value:.4f}',
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: Response groups
bp = axes[1].boxplot([low_resp, high_resp], labels=['Low', 'High'],
                      widths=0.6, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightgray')
    patch.set_alpha(0.7)

axes[1].scatter([1]*len(low_resp), low_resp, alpha=0.6, s=80, 
                color='gray', edgecolors='black', linewidth=0.5)
axes[1].scatter([2]*len(high_resp), high_resp, alpha=0.6, s=80,
                color='darkgray', edgecolors='black', linewidth=0.5)
axes[1].set_ylabel('Fake "Response" (%)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Group', fontsize=12, fontweight='bold')
axes[1].set_title(f'Mann-Whitney p = {p_bio:.6f}',
                  fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].axhline(0, color='black', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('negative_control_results.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Figure saved: negative_control_results.png")

# Interpretation
print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

print(f"\nNegative control results:")
print(f"  Classification accuracy: {observed_accuracy*100:.1f}%")
print(f"  Permutation p-value: {p_value:.4f}")
print(f"  Dark activity predicts: p={p_dark:.4f}")

print("\nComparison to morphine study (25 mg/kg):")
print(f"  Morphine: 72.2% accuracy (p=0.054)")
print(f"  Negative control: {observed_accuracy*100:.1f}% accuracy (p={p_value:.4f})")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if p_value > 0.10 and observed_accuracy < 0.65:
    print("\n✓✓ NEGATIVE CONTROL PASSED!")
    print("  Baseline does NOT predict fake 'response' in untreated mice")
    print("  This VALIDATES your 25mg morphine finding")
    print("  Your original result is likely REAL, not spurious correlation")
elif p_value < 0.05:
    print("\n✗✗ NEGATIVE CONTROL FAILED!")
    print("  Baseline DOES predict fake 'response' in untreated mice")
    print("  This suggests your 25mg finding might be spurious")
    print("  Need to reconsider the original interpretation")
else:
    print("\n⚠ NEGATIVE CONTROL AMBIGUOUS")
    print("  Some prediction exists but marginal")
    print("  Interpret with caution")

print("\n" + "="*80)
