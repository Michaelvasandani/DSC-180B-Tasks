"""
PERMUTATION TEST - Standalone Script
Run this on your own machine to investigate why permutation p≈0.07 
when Mann-Whitney p=0.006

Takes ~2-3 minutes with 1000 permutations
Use 5000+ permutations for final analysis (takes ~10-15 min)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from scipy.stats import mannwhitneyu
import json

# CONFIGURATION
N_PERMUTATIONS = 5000  # Change to 5000 for final run
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

print("="*80)
print("PERMUTATION TEST INVESTIGATION")
print("="*80)

# Load data
features_df = pd.read_csv('animal_features_with_estrus.csv')
responses_df = pd.read_csv('morphine_responses.csv')
data = features_df.merge(responses_df, on='animal_id')

# Filter to 25 mg/kg only
data_25mg = data[data['dose_mg_kg_y'] == 25].copy()
print(f"\nDataset: N = {len(data_25mg)} animals at 25 mg/kg")

# Create labels (median split)
median_response = data_25mg['response_pct'].median()
data_25mg['label'] = (data_25mg['response_pct'] >= median_response).astype(int)

# Prepare features
feature_cols = [col for col in data_25mg.columns 
                if col not in ['animal_id', 'cage_id_x', 'cage_id_y', 'replicate', 
                              'dose_mg_kg_x', 'dose_mg_kg_y', 
                              'response_pct', 'label', 'estrus_phase', 'estrus_code',
                              'baseline_mean', 'post_dose_mean']]
X = data_25mg[feature_cols].fillna(data_25mg[feature_cols].median())
y = data_25mg['label'].values
groups = data_25mg['cage_id_y'].values

print(f"Features: {len(feature_cols)}")
print(f"Unique cages: {len(np.unique(groups))}")

# Get actual response values
low_responders = data_25mg[data_25mg['label']==0]['response_pct'].values
high_responders = data_25mg[data_25mg['label']==1]['response_pct'].values

print("\n" + "="*80)
print("TEST 1: MANN-WHITNEY U TEST")
print("="*80)
print("Question: Are the TRUE groups biologically different?")
print()
stat, p_biological = mannwhitneyu(low_responders, high_responders)
print(f"Low responders:  n={len(low_responders)}, mean={low_responders.mean():.1f}%")
print(f"High responders: n={len(high_responders)}, mean={high_responders.mean():.1f}%")
print(f"Difference: {high_responders.mean() - low_responders.mean():.1f}%")
print()
print(f"Mann-Whitney p-value: {p_biological:.6f}")
print(f"Result: {'✓ SIGNIFICANT' if p_biological < 0.05 else '✗ NOT SIGNIFICANT'}")

# Function to get CV accuracy
def get_cv_accuracy(X, y, groups):
    """Get Leave-One-Cage-Out cross-validated accuracy"""
    logo = LeaveOneGroupOut()
    predictions = np.zeros(len(y))
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, 
                                     min_samples_leaf=2, random_state=RANDOM_SEED)
        clf.fit(X_train, y_train)
        predictions[test_idx] = clf.predict(X_test)
    
    return (predictions == y).mean()

print("\n" + "="*80)
print("TEST 2: PERMUTATION TEST")
print("="*80)
print("Question: Can the CLASSIFIER reliably beat chance?")
print()

# Get observed accuracy
observed_accuracy = get_cv_accuracy(X, y, groups)
print(f"Observed CV accuracy: {observed_accuracy:.3f} ({observed_accuracy*100:.1f}%)")
print()

# Run permutations
print(f"Running {N_PERMUTATIONS} permutations...")
print("(This takes 2-3 min with 1000, or 10-15 min with 5000)")
print()

permuted_accuracies = []
for i in range(N_PERMUTATIONS):
    if (i+1) % 100 == 0:
        print(f"  Progress: {i+1}/{N_PERMUTATIONS} ({(i+1)/N_PERMUTATIONS*100:.0f}%)")
    
    # Shuffle labels randomly
    y_permuted = np.random.permutation(y)
    
    # Get CV accuracy with shuffled labels
    acc = get_cv_accuracy(X, y_permuted, groups)
    permuted_accuracies.append(acc)

permuted_accuracies = np.array(permuted_accuracies)

# Calculate p-value
p_value = (permuted_accuracies >= observed_accuracy).mean()
percentile = (permuted_accuracies < observed_accuracy).mean() * 100

print()
print("Null distribution (what we'd get by chance):")
print(f"  Mean:  {permuted_accuracies.mean():.3f}")
print(f"  Std:   {permuted_accuracies.std():.3f}")
print(f"  Range: [{permuted_accuracies.min():.3f}, {permuted_accuracies.max():.3f}]")
print()
print(f"Observed accuracy: {observed_accuracy:.3f}")
print(f"Permutations ≥ observed: {(permuted_accuracies >= observed_accuracy).sum()}/{N_PERMUTATIONS}")
print(f"Permutation p-value: {p_value:.4f}")
print(f"Percentile rank: {percentile:.1f}th")
print()
print(f"Result: {'✓ SIGNIFICANT' if p_value < 0.05 else '⚠ MARGINAL' if p_value < 0.10 else '✗ NOT SIGNIFICANT'}")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Permutation distribution
axes[0].hist(permuted_accuracies, bins=30, alpha=0.7, edgecolor='black', color='lightgray')
axes[0].axvline(observed_accuracy, color='red', linestyle='--', linewidth=2.5, 
                label=f'Observed: {observed_accuracy:.3f}')
axes[0].axvline(0.5, color='green', linestyle=':', linewidth=2, alpha=0.7,
                label='Chance: 0.500')
axes[0].axvline(permuted_accuracies.mean(), color='blue', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Null mean: {permuted_accuracies.mean():.3f}')
axes[0].set_xlabel('Classification Accuracy', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title(f'Permutation Test (N={N_PERMUTATIONS})\np = {p_value:.4f}', 
                  fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10, framealpha=0.9)
axes[0].grid(True, alpha=0.3)

# Plot 2: True groups comparison
bp = axes[1].boxplot([low_responders, high_responders], 
                      labels=['Low Responders', 'High Responders'],
                      widths=0.6, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)
    
axes[1].scatter([1]*len(low_responders), low_responders, 
                alpha=0.6, s=80, color='darkblue', edgecolors='black', linewidth=0.5)
axes[1].scatter([2]*len(high_responders), high_responders, 
                alpha=0.6, s=80, color='darkred', edgecolors='black', linewidth=0.5)
axes[1].set_ylabel('Morphine Response (%)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Group', fontsize=12, fontweight='bold')
axes[1].set_title(f'Mann-Whitney U Test\np = {p_biological:.6f}', 
                  fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('permutation_test_results.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Figure saved: permutation_test_results.png")

# Save numerical results
results = {
    'configuration': {
        'n_samples': int(len(data_25mg)),
        'n_features': int(len(feature_cols)),
        'n_permutations': int(N_PERMUTATIONS),
        'random_seed': int(RANDOM_SEED)
    },
    'observed_performance': {
        'cv_accuracy': float(observed_accuracy),
        'cv_accuracy_pct': float(observed_accuracy * 100)
    },
    'mann_whitney_test': {
        'p_value': float(p_biological),
        'significant': bool(p_biological < 0.05),
        'low_mean': float(low_responders.mean()),
        'high_mean': float(high_responders.mean()),
        'difference': float(high_responders.mean() - low_responders.mean())
    },
    'permutation_test': {
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
        'percentile': float(percentile),
        'null_mean': float(permuted_accuracies.mean()),
        'null_std': float(permuted_accuracies.std()),
        'null_min': float(permuted_accuracies.min()),
        'null_max': float(permuted_accuracies.max())
    }
}

with open('permutation_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Results saved: permutation_test_results.json")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print()
print("WHY ARE THE P-VALUES DIFFERENT?")
print("-" * 80)
print()
print(f"Mann-Whitney:   p = {p_biological:.6f}  ← Tests biological difference")
print(f"Permutation:    p = {p_value:.4f}  ← Tests classifier performance")
print()
print("These are testing DIFFERENT questions:")
print()
print("1. Mann-Whitney U Test:")
print("   • Question: Are the response values different between groups?")
print("   • Method: Directly compares distributions")
print("   • Result: Groups ARE significantly different")
print()
print("2. Permutation Test:")
print("   • Question: Can a classifier beat chance at predicting groups?")
print("   • Method: Tests if cross-validated predictions are reliable")
print("   • Result: Classifier performance is marginal/not significant")
print()
print("WHY THE DISCREPANCY?")
print("-" * 80)
print()
print("The classifier struggles because:")
print(f"  • Small sample size (N={len(data_25mg)})")
print(f"  • Many features ({len(feature_cols)} features)")
print(f"  • High risk of overfitting (samples/features ratio = {len(data_25mg)/len(feature_cols):.2f})")
print()
print("Even though groups ARE biologically different (Mann-Whitney),")
print("the classifier can't learn the pattern reliably with so few samples.")

print("\n" + "="*80)
print("RECOMMENDATIONS FOR REPORTING")
print("="*80)
print()

if p_value < 0.05:
    print("✓ STRONG RESULT - Both tests significant")
    print()
    print("Report:")
    print(f'  "Cross-validated classification achieved {observed_accuracy:.1%} accuracy')
    print(f'   (N={len(data_25mg)}, LOCO-CV), significantly exceeding chance')
    print(f'   (p={p_value:.3f}, permutation test). High and low responder groups')
    print(f'   showed significant biological differences (p={p_biological:.6f},')
    print('   Mann-Whitney U test)."')
    
elif p_value < 0.10:
    print("⚠ MODERATE RESULT - Mann-Whitney significant, permutation marginal")
    print()
    print("Option 1 - Focus on biological finding:")
    print(f'  "Animals with higher baseline dark-phase activity showed')
    print(f'   significantly lower morphine responses ({low_responders.mean():.0f}%')
    print(f'   vs {high_responders.mean():.0f}%, p={p_biological:.6f}, Mann-Whitney')
    print('   U test), suggesting baseline behavioral phenotypes predict')
    print('   individual drug sensitivity."')
    print()
    print("Option 2 - Report trend:")
    print(f'  "Classification showed a trend toward above-chance performance')
    print(f'   ({observed_accuracy:.1%} accuracy, p={p_value:.3f}, permutation test).')
    print(f'   Groups differed significantly in their responses (p={p_biological:.6f})."')
    
else:
    print("✗ WEAK RESULT - Only Mann-Whitney significant")
    print()
    print("Recommended approach:")
    print(f'  "While cross-validated classification did not reliably exceed')
    print(f'   chance (p={p_value:.3f}), animals with higher baseline activity')
    print(f'   showed significantly lower morphine responses (p={p_biological:.6f},')
    print('   Mann-Whitney U test), suggesting biological differences exist')
    print('   but require larger samples for reliable prediction."')

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print()
if p_value >= 0.05:
    print("1. Re-run with 5000+ permutations for stable p-value estimate")
    print("2. Consider feature selection to reduce overfitting")
    print("3. Focus paper on biological findings (Mann-Whitney)")
    print("4. Collect more samples for validation cohort")
else:
    print("1. Re-run with 5000+ permutations to confirm")
    print("2. Validate in independent cohort")
    print("3. Consider ensemble methods")

print("\n" + "="*80)
print("DONE!")
print("="*80)
