"""
NO-ESTROUS MODEL COMPARISON
Tests if estrous cycle features improve morphine response prediction

Compares two models:
1. FULL MODEL: All 28 features (including 4 estrous features)
2. NO-ESTROUS MODEL: 24 features (excluding estrous features)

Question: Does including estrous cycle information improve prediction accuracy?

Estrous features excluded in NO-ESTROUS model:
- has_4day_cycle
- cycling_strength
- cycle_phase_at_dose
- near_estrus_at_dose
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from scipy.stats import mannwhitneyu
import json
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("ESTROUS FEATURE COMPARISON ANALYSIS")
print("="*80)
print("\nQuestion: Does including estrous cycle features improve prediction?")
print("="*80)

# Load data
print("\nLoading data...")
features_df = pd.read_csv('animal_features_with_estrus.csv')
responses_df = pd.read_csv('morphine_responses.csv')
data = features_df.merge(responses_df, on='animal_id')

# Filter to 25mg (where we found the effect)
data_25mg = data[data['dose_mg_kg_y'] == 25].copy()
print(f"Dataset: N = {len(data_25mg)} animals at 25 mg/kg")

# Create labels
median_response = data_25mg['response_pct'].median()
data_25mg['label'] = (data_25mg['response_pct'] >= median_response).astype(int)

print(f"Median response: {median_response:.1f}%")
print(f"Low responders: {(data_25mg['label']==0).sum()}")
print(f"High responders: {(data_25mg['label']==1).sum()}")

# Define estrous features to exclude
ESTROUS_FEATURES = [
    'has_4day_cycle',
    'cycling_strength',
    'cycle_phase_at_dose',
    'near_estrus_at_dose'
]

# Get all features
all_feature_cols = [col for col in data_25mg.columns 
                    if col not in ['animal_id', 'cage_id_x', 'cage_id_y', 'replicate', 
                                  'dose_mg_kg_x', 'dose_mg_kg_y', 
                                  'response_pct', 'label', 'estrus_phase', 'estrus_code',
                                  'baseline_mean', 'post_dose_mean']]

# Create feature sets
full_features = all_feature_cols
no_estrous_features = [f for f in all_feature_cols if f not in ESTROUS_FEATURES]

print(f"\n" + "="*80)
print("FEATURE SETS")
print("="*80)
print(f"Full model: {len(full_features)} features")
print(f"No-estrous model: {len(no_estrous_features)} features")
print(f"Estrous features excluded: {ESTROUS_FEATURES}")
print(f"Difference: {len(full_features) - len(no_estrous_features)} features")

# Prepare data
X_full = data_25mg[full_features].fillna(data_25mg[full_features].median())
X_no_estrous = data_25mg[no_estrous_features].fillna(data_25mg[no_estrous_features].median())
y = data_25mg['label'].values
groups = data_25mg['cage_id_y'].values

print(f"\nSamples: {len(y)}")
print(f"Cages: {len(np.unique(groups))}")

# Function to run LOCO-CV
def run_loco_cv(X, y, groups, model_name="Model"):
    """Run Leave-One-Cage-Out cross-validation"""
    logo = LeaveOneGroupOut()
    predictions = np.zeros(len(y))
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, 
                                     min_samples_leaf=2, random_state=42)
        clf.fit(X_train, y_train)
        predictions[test_idx] = clf.predict(X_test)
    
    accuracy = (predictions == y).mean()
    cm = confusion_matrix(y, predictions)
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'confusion_matrix': cm
    }

# Function to run permutation test
def run_permutation_test(X, y, groups, observed_acc, n_perms=1000):
    """Run permutation test"""
    perm_accuracies = []
    
    for i in range(n_perms):
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{n_perms}...")
        
        y_perm = np.random.permutation(y)
        result = run_loco_cv(X, y_perm, groups)
        perm_accuracies.append(result['accuracy'])
    
    perm_accuracies = np.array(perm_accuracies)
    p_value = (perm_accuracies >= observed_acc).mean()
    percentile = (perm_accuracies < observed_acc).mean() * 100
    
    return {
        'perm_accuracies': perm_accuracies,
        'p_value': p_value,
        'percentile': percentile,
        'mean': perm_accuracies.mean(),
        'std': perm_accuracies.std()
    }

# ============================================================================
# RUN FULL MODEL (28 features with estrous)
# ============================================================================
print("\n" + "="*80)
print("MODEL 1: FULL MODEL (28 features, INCLUDING estrous)")
print("="*80)

print("\nRunning LOCO-CV...")
results_full = run_loco_cv(X_full, y, groups, "Full")

print(f"\nResults:")
print(f"  Accuracy: {results_full['accuracy']:.3f} ({results_full['accuracy']*100:.1f}%)")
print(f"\nConfusion Matrix:")
print(f"                Predicted Low | Predicted High")
print(f"  Actual Low:         {results_full['confusion_matrix'][0,0]}      |        {results_full['confusion_matrix'][0,1]}")
print(f"  Actual High:        {results_full['confusion_matrix'][1,0]}      |        {results_full['confusion_matrix'][1,1]}")

print("\nRunning permutation test (1000 iterations)...")
perm_full = run_permutation_test(X_full, y, groups, results_full['accuracy'], n_perms=1000)

print(f"\nPermutation test:")
print(f"  Null mean: {perm_full['mean']:.3f}")
print(f"  Observed: {results_full['accuracy']:.3f}")
print(f"  p-value: {perm_full['p_value']:.4f}")
print(f"  Percentile: {perm_full['percentile']:.1f}th")

# ============================================================================
# RUN NO-ESTROUS MODEL (24 features without estrous)
# ============================================================================
print("\n" + "="*80)
print("MODEL 2: NO-ESTROUS MODEL (24 features, EXCLUDING estrous)")
print("="*80)

print("\nRunning LOCO-CV...")
results_no_estrous = run_loco_cv(X_no_estrous, y, groups, "No-Estrous")

print(f"\nResults:")
print(f"  Accuracy: {results_no_estrous['accuracy']:.3f} ({results_no_estrous['accuracy']*100:.1f}%)")
print(f"\nConfusion Matrix:")
print(f"                Predicted Low | Predicted High")
print(f"  Actual Low:         {results_no_estrous['confusion_matrix'][0,0]}      |        {results_no_estrous['confusion_matrix'][0,1]}")
print(f"  Actual High:        {results_no_estrous['confusion_matrix'][1,0]}      |        {results_no_estrous['confusion_matrix'][1,1]}")

print("\nRunning permutation test (1000 iterations)...")
perm_no_estrous = run_permutation_test(X_no_estrous, y, groups, results_no_estrous['accuracy'], n_perms=1000)

print(f"\nPermutation test:")
print(f"  Null mean: {perm_no_estrous['mean']:.3f}")
print(f"  Observed: {results_no_estrous['accuracy']:.3f}")
print(f"  p-value: {perm_no_estrous['p_value']:.4f}")
print(f"  Percentile: {perm_no_estrous['percentile']:.1f}th")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("DIRECT COMPARISON")
print("="*80)

acc_diff = results_full['accuracy'] - results_no_estrous['accuracy']
p_diff = perm_full['p_value'] - perm_no_estrous['p_value']

print(f"\nAccuracy:")
print(f"  Full model (28 features):      {results_full['accuracy']:.3f} ({results_full['accuracy']*100:.1f}%)")
print(f"  No-estrous model (24 features): {results_no_estrous['accuracy']:.3f} ({results_no_estrous['accuracy']*100:.1f}%)")
print(f"  Difference: {acc_diff:+.3f} ({acc_diff*100:+.1f} percentage points)")

print(f"\nPermutation p-values:")
print(f"  Full model:      p = {perm_full['p_value']:.4f}")
print(f"  No-estrous model: p = {perm_no_estrous['p_value']:.4f}")
print(f"  Difference: {p_diff:+.4f}")

print(f"\nPrediction agreement:")
agreement = (results_full['predictions'] == results_no_estrous['predictions']).mean()
print(f"  Models agree on {agreement*100:.1f}% of animals ({int(agreement*len(y))}/{len(y)})")

disagreements = np.where(results_full['predictions'] != results_no_estrous['predictions'])[0]
print(f"  Models disagree on {len(disagreements)} animals: {disagreements.tolist()}")

# Check if disagreements are meaningful
if len(disagreements) > 0:
    # Get actual responses for disagreement cases
    disagreement_data = data_25mg.iloc[disagreements]
    print(f"\n  Disagreement cases:")
    for idx in disagreements:
        actual = y[idx]
        pred_full = results_full['predictions'][idx]
        pred_no_est = results_no_estrous['predictions'][idx]
        response = data_25mg.iloc[idx]['response_pct']
        print(f"    Animal {idx}: Actual={actual}, Full={int(pred_full)}, No-Est={int(pred_no_est)}, Response={response:.1f}%")

# ============================================================================
# FEATURE IMPORTANCE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE")
print("="*80)

# Train final models to get feature importance
clf_full = RandomForestClassifier(n_estimators=100, max_depth=5, 
                                 min_samples_leaf=2, random_state=42)
clf_full.fit(X_full, y)

clf_no_estrous = RandomForestClassifier(n_estimators=100, max_depth=5, 
                                       min_samples_leaf=2, random_state=42)
clf_no_estrous.fit(X_no_estrous, y)

# Get feature importances
feat_imp_full = pd.DataFrame({
    'feature': full_features,
    'importance': clf_full.feature_importances_
}).sort_values('importance', ascending=False)

feat_imp_no_estrous = pd.DataFrame({
    'feature': no_estrous_features,
    'importance': clf_no_estrous.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 features - FULL MODEL:")
for i, row in feat_imp_full.head(5).iterrows():
    estrous_marker = " (ESTROUS)" if row['feature'] in ESTROUS_FEATURES else ""
    print(f"  {row['feature']}{estrous_marker}: {row['importance']:.4f}")

print("\nTop 5 features - NO-ESTROUS MODEL:")
for i, row in feat_imp_no_estrous.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Check if any estrous features were in top 10
estrous_in_top10 = feat_imp_full.head(10)['feature'].isin(ESTROUS_FEATURES).sum()
print(f"\nEstrous features in top 10 (full model): {estrous_in_top10}/4")
if estrous_in_top10 > 0:
    estrous_top = feat_imp_full[feat_imp_full['feature'].isin(ESTROUS_FEATURES)].head(10)
    print("  Ranks:")
    for i, row in estrous_top.iterrows():
        rank = feat_imp_full.index.get_loc(i) + 1
        print(f"    #{rank}: {row['feature']} ({row['importance']:.4f})")

# ============================================================================
# VISUALIZE
# ============================================================================
print("\n" + "="*80)
print("GENERATING FIGURES")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Accuracy comparison
ax = axes[0, 0]
models = ['Full Model\n(28 features)', 'No-Estrous\n(24 features)']
accs = [results_full['accuracy'], results_no_estrous['accuracy']]
colors = ['#3182ce', '#e07a5f']

bars = ax.bar(models, accs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Chance', alpha=0.7)

for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
           f'{acc*100:.1f}%', ha='center', va='bottom',
           fontweight='bold', fontsize=12)

ax.set_ylabel('Classification Accuracy', fontsize=12, fontweight='bold')
ax.set_ylim([0, 1])
ax.set_title('A. Accuracy Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Panel B: Permutation distributions
ax = axes[0, 1]
ax.hist(perm_full['perm_accuracies'], bins=30, alpha=0.5, 
       label=f'Full (p={perm_full["p_value"]:.3f})', color=colors[0])
ax.hist(perm_no_estrous['perm_accuracies'], bins=30, alpha=0.5,
       label=f'No-Estrous (p={perm_no_estrous["p_value"]:.3f})', color=colors[1])
ax.axvline(results_full['accuracy'], color=colors[0], linestyle='--', linewidth=2)
ax.axvline(results_no_estrous['accuracy'], color=colors[1], linestyle='--', linewidth=2)
ax.axvline(0.5, color='red', linestyle=':', linewidth=1.5, alpha=0.5)

ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('B. Permutation Test Distributions', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel C: Feature importance top 10 comparison
ax = axes[1, 0]
top_features_full = feat_imp_full.head(10)
y_pos = np.arange(len(top_features_full))
colors_bars = ['red' if f in ESTROUS_FEATURES else 'steelblue' 
               for f in top_features_full['feature']]

ax.barh(y_pos, top_features_full['importance'], color=colors_bars, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_features_full['feature'], fontsize=9)
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('C. Top 10 Features (Full Model)\nRed = Estrous features', 
            fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# Panel D: Prediction agreement
ax = axes[1, 1]
agreement_matrix = np.zeros((2, 2))
for i in range(len(y)):
    agreement_matrix[int(results_full['predictions'][i]), 
                    int(results_no_estrous['predictions'][i])] += 1

im = ax.imshow(agreement_matrix, cmap='Blues', alpha=0.6)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['No-Est: Low', 'No-Est: High'])
ax.set_yticklabels(['Full: Low', 'Full: High'])
ax.set_xlabel('No-Estrous Model Prediction', fontsize=12, fontweight='bold')
ax.set_ylabel('Full Model Prediction', fontsize=12, fontweight='bold')
ax.set_title(f'D. Model Agreement ({agreement*100:.0f}%)', 
            fontsize=13, fontweight='bold')

for i in range(2):
    for j in range(2):
        text = ax.text(j, i, f'{int(agreement_matrix[i, j])}',
                      ha="center", va="center", color="black",
                      fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('estrous_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: estrous_comparison.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================
results_comparison = {
    'full_model': {
        'n_features': len(full_features),
        'accuracy': float(results_full['accuracy']),
        'p_value': float(perm_full['p_value']),
        'percentile': float(perm_full['percentile']),
        'confusion_matrix': results_full['confusion_matrix'].tolist()
    },
    'no_estrous_model': {
        'n_features': len(no_estrous_features),
        'accuracy': float(results_no_estrous['accuracy']),
        'p_value': float(perm_no_estrous['p_value']),
        'percentile': float(perm_no_estrous['percentile']),
        'confusion_matrix': results_no_estrous['confusion_matrix'].tolist()
    },
    'comparison': {
        'accuracy_difference': float(acc_diff),
        'p_value_difference': float(p_diff),
        'agreement_rate': float(agreement),
        'n_disagreements': int(len(disagreements)),
        'estrous_features_in_top10': int(estrous_in_top10)
    },
    'estrous_features_excluded': ESTROUS_FEATURES
}

with open('estrous_comparison_results.json', 'w') as f:
    json.dump(results_comparison, f, indent=2)
print("✓ Saved: estrous_comparison_results.json")

# ============================================================================
# INTERPRETATION
# ============================================================================
print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

print(f"\nAccuracy change: {acc_diff*100:+.1f} percentage points")
print(f"P-value change: {p_diff:+.4f}")

if abs(acc_diff) < 0.05:
    print("\n✓ MINIMAL DIFFERENCE")
    print("  Estrous features have little impact on prediction accuracy")
    print("  The 4 estrous features add minimal predictive value")
elif acc_diff > 0.05:
    print("\n✓ ESTROUS IMPROVES PREDICTION")
    print("  Including estrous features improves accuracy")
    print("  Estrous cycle information is valuable for prediction")
else:
    print("\n⚠ ESTROUS HURTS PREDICTION")
    print("  Including estrous features decreases accuracy")
    print("  Estrous cycle may be adding noise rather than signal")

if estrous_in_top10 == 0:
    print("\n✓ NO ESTROUS FEATURES IN TOP 10")
    print("  Estrous features are not among most important predictors")
    print("  Other behavioral features dominate prediction")
elif estrous_in_top10 >= 2:
    print(f"\n⚠ {estrous_in_top10} ESTROUS FEATURES IN TOP 10")
    print("  Estrous cycle contributes meaningfully to prediction")
    print("  Hormonal effects may be important for drug response")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if abs(acc_diff) < 0.05 and estrous_in_top10 == 0:
    print("\nEstrous cycle features do NOT substantially improve morphine")
    print("response prediction in this dataset. The primary predictive")
    print("information comes from non-estrous behavioral features like")
    print("dark-phase activity and circadian rhythms.")
    print("\nConclusion: Estrous features can be EXCLUDED without loss of accuracy.")
elif acc_diff > 0.05 or estrous_in_top10 >= 2:
    print("\nEstrous cycle features DO improve morphine response prediction.")
    print("Including hormonal cycle information provides additional")
    print("predictive value beyond baseline activity patterns.")
    print("\nConclusion: Estrous features should be INCLUDED for best accuracy.")
else:
    print("\nResults are mixed - estrous features have marginal impact.")
    print("May depend on specific dataset or dosing conditions.")

print("\n" + "="*80)
print("DONE!")
print("="*80)
