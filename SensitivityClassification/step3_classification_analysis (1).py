"""
STEP 3: Classification Analysis (25 mg/kg only)

Purpose: Test if baseline features can classify HIGH vs LOW responders

Question: At 25 mg/kg, can we distinguish animals that will have
          above-median vs below-median responses?
          
Method: Random Forest classification with LOCO-CV
Expected: 60-70% accuracy (modest but better than 50% chance)

Why 25mg only?: Earlier analysis showed 5mg has too much noise/overlap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import mannwhitneyu

print("=" * 80)
print("STEP 3: CLASSIFICATION ANALYSIS (25 mg/kg)")
print("=" * 80)
print("\nQuestion: Can we classify HIGH vs LOW responders at 25 mg/kg?")

# Load data
features_df = pd.read_csv('animal_features_with_estrus.csv')
response_df = pd.read_csv('morphine_responses.csv')

df = features_df.merge(response_df, on=['animal_id', 'cage_id', 'dose_mg_kg'])
df_25 = df[df['dose_mg_kg'] == 25].copy()

print(f"\nDataset: {len(df_25)} animals at 25 mg/kg")

# Prepare features
exclude_cols = ['animal_id', 'cage_id', 'dose_mg_kg', 'replicate', 
               'baseline_mean', 'post_dose_mean', 'response_pct']
feature_cols = [col for col in df_25.columns if col not in exclude_cols]

X = df_25[feature_cols].values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Create labels (median split)
median_response = df_25['response_pct'].median()
y = (df_25['response_pct'] >= median_response).astype(int)
cage_ids = df_25['cage_id'].values

print(f"\nFeatures: {len(feature_cols)}")
print(f"Median response: {median_response:.1f}%")
print(f"Low responders (< median): {np.sum(y==0)}")
print(f"High responders (≥ median): {np.sum(y==1)}")
print(f"Unique cages: {len(np.unique(cage_ids))}")

# PRE-FLIGHT CHECKS
print(f"\n{'='*60}")
print("PRE-FLIGHT CHECKS")
print(f"{'='*60}")

# Check 1: Balanced classes
class_balance = min(np.sum(y==0), np.sum(y==1)) / len(y)
print(f"\n✓ Class balance: {class_balance:.2f}")
if class_balance > 0.4:
    print("  ✓ Well balanced (>40% in minority class)")
else:
    print("  ⚠️  Imbalanced classes - results may be biased")

# Check 2: Multiple animals per cage
animals_per_cage = len(y) / len(np.unique(cage_ids))
print(f"\n✓ Animals per cage: {animals_per_cage:.1f}")
if animals_per_cage >= 2:
    print("  ✓ Multiple animals per cage (good for LOCO-CV)")
else:
    print("  ⚠️  Few animals per cage - LOCO-CV may be unstable")

# Check 3: CV setup
logo = LeaveOneGroupOut()
n_splits = logo.get_n_splits(X, y, cage_ids)
print(f"\n✓ CV folds: {n_splits}")
if n_splits == len(np.unique(cage_ids)):
    print("  ✓ CV configured correctly (one fold per cage)")
else:
    print("  ✗ ERROR: CV not configured correctly!")

# Train classifier
print(f"\n{'='*60}")
print("TRAINING CLASSIFIER")
print(f"{'='*60}")

clf = RandomForestClassifier(n_estimators=200, max_depth=5, 
                             min_samples_leaf=2, class_weight='balanced',
                             random_state=42)

# Cross-validated predictions (HONEST ESTIMATE)
y_pred = cross_val_predict(clf, X, y, groups=cage_ids, cv=logo)

# Metrics
acc = accuracy_score(y, y_pred)
baseline_acc = max(np.sum(y==0), np.sum(y==1)) / len(y)
improvement = acc - baseline_acc

print(f"\nCross-Validated Performance:")
print(f"  Accuracy: {acc:.3f}")
print(f"  Baseline (majority class): {baseline_acc:.3f}")
print(f"  Improvement: {improvement:.3f}")

# Confusion matrix
cm = confusion_matrix(y, y_pred)
print(f"\nConfusion Matrix:")
print(f"                Predicted Low | Predicted High")
print(f"  Actual Low:        {cm[0,0]:3d}      |      {cm[0,1]:3d}")
print(f"  Actual High:       {cm[1,0]:3d}      |      {cm[1,1]:3d}")

# Per-class accuracy
low_acc = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
high_acc = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
print(f"\nPer-class accuracy:")
print(f"  Low responders: {low_acc:.3f}")
print(f"  High responders: {high_acc:.3f}")

# Train on all data for overfitting check
clf.fit(X, y)
y_pred_train = clf.predict(X)
acc_train = accuracy_score(y, y_pred_train)

print(f"\nOverfitting Check:")
print(f"  Training accuracy: {acc_train:.3f}")
print(f"  CV accuracy: {acc:.3f}")
print(f"  Gap: {acc_train - acc:.3f}")

# Feature importance
importances = clf.feature_importances_
fi_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

print(f"\nTop 5 Features:")
for idx, row in fi_df.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# SANITY CHECKS
print(f"\n{'='*80}")
print("SANITY CHECKS")
print(f"{'='*80}")

checks_passed = 0
total_checks = 6

# Check 1: Better than chance
print(f"\n✓ CHECK 1: Better than random guessing")
print(f"  Accuracy: {acc:.3f}")
print(f"  Chance (50%): 0.500")
if acc > 0.55:
    print("  ✓ PASS - Better than chance")
    checks_passed += 1
else:
    print("  ✗ FAIL - Not better than random")

# Check 2: Better than baseline
print(f"\n✓ CHECK 2: Better than majority class baseline")
print(f"  Improvement: {improvement:.3f}")
if improvement > 0.10:
    print("  ✓ PASS - Substantial improvement")
    checks_passed += 1
elif improvement > 0.05:
    print("  ~ MARGINAL - Small improvement")
    checks_passed += 0.5
else:
    print("  ✗ FAIL - No improvement over baseline")

# Check 3: Not overfitting severely
print(f"\n✓ CHECK 3: Overfitting check")
print(f"  Train - CV accuracy: {acc_train - acc:.3f}")
if acc_train - acc < 0.30:
    print("  ✓ PASS - Reasonable overfitting")
    checks_passed += 1
elif acc_train - acc < 0.40:
    print("  ~ WARNING - Moderate overfitting")
    checks_passed += 0.5
else:
    print("  ✗ FAIL - Severe overfitting")

# Check 4: Groups actually different
# Test if feature values differ between groups
dark_mean_idx = feature_cols.index('dark_mean')
dark_low = X[y==0, dark_mean_idx]
dark_high = X[y==1, dark_mean_idx]
u_stat, p_val = mannwhitneyu(dark_low, dark_high, alternative='two-sided')

print(f"\n✓ CHECK 4: Groups biologically different")
print(f"  dark_mean in low responders: {dark_low.mean():.4f}")
print(f"  dark_mean in high responders: {dark_high.mean():.4f}")
print(f"  Mann-Whitney U p-value: {p_val:.4f}")
if p_val < 0.05:
    print("  ✓ PASS - Significantly different")
    checks_passed += 1
else:
    print("  ✗ FAIL - Not significantly different")

# Check 5: Both classes predicted
print(f"\n✓ CHECK 5: Predicts both classes")
unique_preds = len(np.unique(y_pred))
print(f"  Unique predictions: {unique_preds}")
if unique_preds == 2:
    print("  ✓ PASS - Predicts both high and low")
    checks_passed += 1
else:
    print("  ✗ FAIL - Only predicting one class")

# Check 6: Actual response groups different
responses_low = df_25[y==0]['response_pct']
responses_high = df_25[y==1]['response_pct']
u_resp, p_resp = mannwhitneyu(responses_low, responses_high, alternative='two-sided')

print(f"\n✓ CHECK 6: Response groups actually different")
print(f"  Low group mean: {responses_low.mean():.1f}%")
print(f"  High group mean: {responses_high.mean():.1f}%")
print(f"  p-value: {p_resp:.6f}")
if p_resp < 0.001:
    print("  ✓ PASS - Groups very different")
    checks_passed += 1
else:
    print("  ⚠️  Groups overlap (expected with median split)")
    checks_passed += 0.5

# Create visualizations
fig = plt.figure(figsize=(16, 6))

# Confusion matrix
ax1 = plt.subplot(1, 3, 1)
im = ax1.imshow(cm, cmap='Blues', aspect='auto')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Pred Low', 'Pred High'])
ax1.set_yticklabels(['Actual Low', 'Actual High'])
for i in range(2):
    for j in range(2):
        ax1.text(j, i, str(cm[i, j]), ha='center', va='center', 
                fontsize=20, fontweight='bold')
ax1.set_title(f'Confusion Matrix\nAccuracy = {acc:.3f}', fontweight='bold')

# Feature importance
ax2 = plt.subplot(1, 3, 2)
top_n = 10
top_fi = fi_df.head(top_n)
y_pos = np.arange(top_n)
ax2.barh(y_pos, top_fi['importance'].values, alpha=0.7)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(top_fi['feature'].values, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel('Importance')
ax2.set_title('Top 10 Features', fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Response distributions
ax3 = plt.subplot(1, 3, 3)
ax3.hist(responses_low, bins=6, alpha=0.6, color='blue', 
        label=f'Low (N={len(responses_low)})', edgecolor='black')
ax3.hist(responses_high, bins=6, alpha=0.6, color='red',
        label=f'High (N={len(responses_high)})', edgecolor='black')
ax3.axvline(median_response, color='black', linestyle='--', 
           linewidth=2, label='Median')
ax3.set_xlabel('Response (%)')
ax3.set_ylabel('Count')
ax3.set_title('Response Distribution', fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('step3_classification_results.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Figure saved: step3_classification_results.png")

# Save results
results_df = df_25.copy()
results_df['true_class'] = y
results_df['predicted_class'] = y_pred
results_df['correct'] = (y == y_pred)
results_df.to_csv('step3_classification_predictions.csv', index=False)
print(f"✓ Results saved: step3_classification_predictions.csv")

# Save feature importance
fi_df.to_csv('step3_feature_importance.csv', index=False)
print(f"✓ Feature importance saved: step3_feature_importance.csv")

# INTERPRETATION
print(f"\n{'='*80}")
print("INTERPRETATION")
print(f"{'='*80}")

print(f"\nCan we classify high vs low responders at 25 mg/kg?")
if acc > 0.70:
    print(f"  ✓ YES - Good classification (Accuracy = {acc:.3f})")
    result = "SUCCESS"
elif acc > 0.60:
    print(f"  ~ MODEST - Moderate classification (Accuracy = {acc:.3f})")
    result = "MODEST"
else:
    print(f"  ✗ NO - Weak classification (Accuracy = {acc:.3f})")
    result = "WEAK"

if p_val < 0.05:
    print(f"  Groups are biologically different (p = {p_val:.4f})")
else:
    print(f"  Groups not significantly different (p = {p_val:.4f})")

print(f"\nBiological finding:")
if dark_low.mean() > dark_high.mean():
    print(f"  Higher baseline dark-phase activity → Lower morphine response")
    print(f"  Interpretation: More active animals are less drug-sensitive")
else:
    print(f"  Lower baseline dark-phase activity → Higher morphine response")
    print(f"  Interpretation: Less active animals are more drug-sensitive")

# Summary
print(f"\n{'='*80}")
print(f"STEP 3 COMPLETE: {checks_passed}/{total_checks} checks passed")
print(f"{'='*80}")

print(f"\nKey Findings:")
print(f"  Classification accuracy: {acc:.3f}")
print(f"  Improvement over baseline: {improvement:.3f}")
print(f"  Group difference p-value: {p_val:.4f}")
print(f"  Result: {result}")

print(f"\nNext step: python step4_validation_and_summary.py")

# Store for summary
summary_data = {
    'accuracy': acc,
    'baseline_acc': baseline_acc,
    'improvement': improvement,
    'group_p_value': p_val,
    'overfitting_gap': acc_train - acc,
    'checks_passed': checks_passed,
    'total_checks': total_checks,
    'result': result
}

import json
with open('step3_summary.json', 'w') as f:
    json.dump(summary_data, f, indent=2)
print(f"✓ Summary data saved: step3_summary.json")
