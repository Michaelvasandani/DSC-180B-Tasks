"""
TEST IF REAL ESTROUS IMPROVES BASELINE MODEL

Comparison:
- Model A: Baseline features only (24 features) - Current: 72.2%
- Model B: Baseline + REAL Estrous (24 + 6 features) - New: ???%

Question: Does REAL estrous data improve morphine response prediction?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from scipy.stats import mannwhitneyu
import json

np.random.seed(42)

print("="*80)
print("TESTING REAL ESTROUS IMPROVEMENT")
print("="*80)

# Load data with real estrous
data = pd.read_csv('morphine_with_real_estrous.csv')

# Filter to 25mg (where we found the effect)
data_25mg = data[data['dose_mg_kg_y'] == 25].copy()

print(f"\nDataset: N = {len(data_25mg)} (25 mg/kg)")
print(f"Animals with real estrous data: {data_25mg['p_lowU_at_dose'].notna().sum()}")

# Create labels
median_response = data_25mg['response_pct'].median()
data_25mg['label'] = (data_25mg['response_pct'] >= median_response).astype(int)

# ==================================================================
# DEFINE FEATURE SETS
# ==================================================================

# OLD broken estrous features (to exclude)
BROKEN_ESTROUS = ['has_4day_cycle', 'cycling_strength', 'cycle_phase_at_dose', 'near_estrus_at_dose']

# NEW real estrous features
REAL_ESTROUS = ['p_lowU_at_dose', 'in_estrous_at_dose', 'baseline_p_lowU_mean', 
                'baseline_p_lowU_std', 'cycling_strength_real', 
                'baseline_p_lowU_max', 'n_estrous_days_baseline']

# All features
all_cols = [col for col in data_25mg.columns 
            if col not in ['animal_id', 'cage_id_x', 'cage_id_y', 'replicate',
                          'dose_mg_kg_x', 'dose_mg_kg_y', 'response_pct', 'label',
                          'estrus_phase', 'estrus_code', 'baseline_mean', 'post_dose_mean',
                          'date', 'day', 'group']]

# Baseline features (no estrous at all)
baseline_features = [f for f in all_cols if f not in BROKEN_ESTROUS + REAL_ESTROUS]

# With real estrous
with_estrous_features = baseline_features + [f for f in REAL_ESTROUS if f in data_25mg.columns]

print(f"\n" + "="*80)
print("FEATURE SETS")
print("="*80)
print(f"Baseline (no estrous): {len(baseline_features)} features")
print(f"With REAL estrous: {len(with_estrous_features)} features")
print(f"\nReal estrous features available:")
for feat in REAL_ESTROUS:
    if feat in data_25mg.columns:
        available = data_25mg[feat].notna().sum()
        print(f"  ✓ {feat}: {available}/{len(data_25mg)} animals")
    else:
        print(f"  ✗ {feat}: NOT FOUND")

# ==================================================================
# PREPARE DATA
# ==================================================================

# Only use animals with real estrous data
data_with_estrous = data_25mg.dropna(subset=[f for f in REAL_ESTROUS if f in data_25mg.columns])

print(f"\n" + "="*80)
print(f"SAMPLE SIZE")
print(f"="*80)
print(f"Total 25mg animals: {len(data_25mg)}")
print(f"With real estrous: {len(data_with_estrous)}")

if len(data_with_estrous) < 10:
    print("\n⚠ WARNING: Too few animals with estrous data for robust analysis")
    print("  Need more samples for reliable comparison")

# Use animals with estrous data
X_baseline = data_with_estrous[baseline_features].fillna(data_with_estrous[baseline_features].median())
X_with_estrous = data_with_estrous[with_estrous_features].fillna(data_with_estrous[with_estrous_features].median())
y = data_with_estrous['label'].values
groups = data_with_estrous['cage_id_y'].values

print(f"\nFinal sample size: {len(y)}")
print(f"Features baseline: {len(baseline_features)}")
print(f"Features with estrous: {len(with_estrous_features)}")

# ==================================================================
# RUN COMPARISON
# ==================================================================

def run_loco_cv(X, y, groups):
    logo = LeaveOneGroupOut()
    predictions = np.zeros(len(y))
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y[train_idx]
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                     min_samples_leaf=2, random_state=42)
        clf.fit(X_train, y_train)
        predictions[test_idx] = clf.predict(X_test)
    
    return predictions

print("\n" + "="*80)
print("RUNNING COMPARISON")
print("="*80)

print("\nModel A: Baseline only (24 features)...")
pred_baseline = run_loco_cv(X_baseline, y, groups)
acc_baseline = (pred_baseline == y).mean()

print("Model B: Baseline + REAL Estrous...")
pred_with_estrous = run_loco_cv(X_with_estrous, y, groups)
acc_with_estrous = (pred_with_estrous == y).mean()

# ==================================================================
# RESULTS
# ==================================================================

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nModel A (Baseline only):       {acc_baseline:.3f} ({acc_baseline*100:.1f}%)")
print(f"Model B (+ Real Estrous):      {acc_with_estrous:.3f} ({acc_with_estrous*100:.1f}%)")
print(f"Difference:                     {(acc_with_estrous-acc_baseline)*100:+.1f} percentage points")

# Agreement
agreement = (pred_baseline == pred_with_estrous).mean()
print(f"\nModels agree on {agreement*100:.0f}% of animals ({int(agreement*len(y))}/{len(y)})")

# Feature importance with real estrous
clf_estrous = RandomForestClassifier(n_estimators=100, max_depth=5,
                                    min_samples_leaf=2, random_state=42)
clf_estrous.fit(X_with_estrous, y)

feat_imp = pd.DataFrame({
    'feature': with_estrous_features,
    'importance': clf_estrous.feature_importances_
}).sort_values('importance', ascending=False)

# Check real estrous in top 10
estrous_in_top10 = feat_imp.head(10)['feature'].isin(REAL_ESTROUS).sum()
print(f"\nReal estrous features in top 10: {estrous_in_top10}/{len([f for f in REAL_ESTROUS if f in data_with_estrous.columns])}")

if estrous_in_top10 > 0:
    print("  Real estrous features ranked:")
    for feat in REAL_ESTROUS:
        if feat in feat_imp.head(10)['feature'].values:
            rank = feat_imp.index[feat_imp['feature']==feat].tolist()[0] + 1
            importance = feat_imp[feat_imp['feature']==feat]['importance'].values[0]
            print(f"    #{rank}: {feat} ({importance:.4f})")

# ==================================================================
# VISUALIZE
# ==================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Accuracy comparison
ax = axes[0]
models = ['Baseline\n(24 feat)', 'Baseline +\nReal Estrous']
accs = [acc_baseline, acc_with_estrous]
colors = ['#3182ce', '#2d8659']
bars = ax.bar(models, accs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.axhline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Chance')

for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
           f'{acc*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)

ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_ylim([0, 1])
ax.set_title('A. Model Comparison\nDoes Real Estrous Improve Prediction?',
            fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel B: Feature importance (top 10)
ax = axes[1]
top_features = feat_imp.head(10)
colors_bars = ['#2d8659' if f in REAL_ESTROUS else 'steelblue' for f in top_features['feature']]
ax.barh(range(len(top_features)), top_features['importance'], color=colors_bars, alpha=0.7)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'], fontsize=9)
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('B. Top 10 Features\n(Green = Real Estrous)',
            fontsize=13, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# Panel C: Estrous at dose vs response
ax = axes[2]
data_plot = data_with_estrous.copy()
colors_estrous = ['red' if e==1 else 'blue' for e in data_plot['in_estrous_at_dose']]
ax.scatter(data_plot['p_lowU_at_dose'], data_plot['response_pct'],
          c=colors_estrous, alpha=0.6, s=100, edgecolors='black', linewidth=1)
ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Estrous threshold')
ax.set_xlabel('p_lowU at Dosing', fontsize=12, fontweight='bold')
ax.set_ylabel('Morphine Response (%)', fontsize=12, fontweight='bold')
ax.set_title('C. Estrous Status vs Response',
            fontsize=13, fontweight='bold')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', label='In Estrous'),
                  Patch(facecolor='blue', label='Not Estrous')]
ax.legend(handles=legend_elements)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('estrous_improvement_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: estrous_improvement_results.png")

# Save results
results = {
    'baseline_accuracy': float(acc_baseline),
    'with_estrous_accuracy': float(acc_with_estrous),
    'improvement': float(acc_with_estrous - acc_baseline),
    'improvement_pct': float((acc_with_estrous - acc_baseline) * 100),
    'agreement_rate': float(agreement),
    'estrous_in_top10': int(estrous_in_top10),
    'n_samples': int(len(y)),
    'n_features_baseline': len(baseline_features),
    'n_features_with_estrous': len(with_estrous_features)
}

with open('estrous_improvement_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Saved: estrous_improvement_results.json")

# ==================================================================
# INTERPRETATION
# ==================================================================

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

improvement = acc_with_estrous - acc_baseline

if improvement > 0.1:
    print("\n✓✓ REAL ESTROUS SUBSTANTIALLY IMPROVES PREDICTION!")
    print(f"  Improvement: +{improvement*100:.1f} percentage points")
    print(f"  Estrous features in top 10: {estrous_in_top10}")
    print("\n  CONCLUSION:")
    print("  → Estrous cycle DOES affect morphine response")
    print("  → Include real estrous in final model")
    print("  → Major finding for your hypothesis!")
    
elif improvement > 0.05:
    print("\n✓ REAL ESTROUS MODERATELY IMPROVES PREDICTION")
    print(f"  Improvement: +{improvement*100:.1f} percentage points")
    print(f"  Estrous features in top 10: {estrous_in_top10}")
    print("\n  CONCLUSION:")
    print("  → Estrous cycle adds some predictive value")
    print("  → Worth including if precision matters")
    print("  → Secondary effect on morphine response")
    
elif improvement > 0:
    print("\n→ REAL ESTROUS HAS MINIMAL IMPACT")
    print(f"  Improvement: +{improvement*100:.1f} percentage points")
    print(f"  Estrous features in top 10: {estrous_in_top10}")
    print("\n  CONCLUSION:")
    print("  → Baseline features capture most variability")
    print("  → Estrous effects may be indirect")
    print("  → Simpler baseline model preferred")
    
else:
    print("\n✗ REAL ESTROUS DOES NOT IMPROVE PREDICTION")
    print(f"  Change: {improvement*100:.1f} percentage points")
    print(f"  Estrous features in top 10: {estrous_in_top10}")
    print("\n  CONCLUSION:")
    print("  → Baseline activity captures relevant info")
    print("  → Estrous effects integrated into behavior")
    print("  → Keep 24-feature baseline model")

print("\n" + "="*80)
print("COMPARISON TO ORIGINAL QUESTION")
print("="*80)

print(f"\nYour research question: Does estrous affect opioid sensitivity?")
print(f"\nWith BROKEN estrous features:")
print(f"  Accuracy: 72.2% (same as baseline)")
print(f"  Features in top 10: 0/4")
print(f"  Conclusion: Couldn't test hypothesis ✗")

print(f"\nWith REAL estrous features:")
print(f"  Accuracy: {acc_with_estrous*100:.1f}%")
print(f"  Improvement: {improvement*100:+.1f} percentage points")
print(f"  Features in top 10: {estrous_in_top10}")

if improvement > 0.05:
    print(f"  Conclusion: Hypothesis SUPPORTED! ✓✓")
else:
    print(f"  Conclusion: Hypothesis NOT supported")

print("\n" + "="*80)
