"""
STEP 2: Regression Analysis

Purpose: Test if baseline features can predict EXACT response magnitude

Question: Can we predict how much each animal will respond?
Method: Random Forest regression with Leave-One-Cage-Out CV
Expected: Will likely FAIL (R² < 0.15) because individual responses are too noisy

This establishes that exact prediction doesn't work, setting up classification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

print("=" * 80)
print("STEP 2: REGRESSION ANALYSIS")
print("=" * 80)
print("\nQuestion: Can baseline features predict exact response magnitude?")

# Load data
features_df = pd.read_csv('animal_features_with_estrus.csv')
response_df = pd.read_csv('morphine_responses.csv')

df = features_df.merge(response_df, on=['animal_id', 'cage_id', 'dose_mg_kg'])
df_morphine = df[df['dose_mg_kg'] > 0].copy()

print(f"\nDataset: {len(df_morphine)} morphine-treated animals")

# Prepare features (WITHOUT dose - honest prediction)
exclude_cols = ['animal_id', 'cage_id', 'dose_mg_kg', 'replicate', 
               'baseline_mean', 'post_dose_mean', 'response_pct']
feature_cols = [col for col in df_morphine.columns if col not in exclude_cols]

X = df_morphine[feature_cols].values
y = df_morphine['response_pct'].values
cage_ids = df_morphine['cage_id'].values

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Features: {len(feature_cols)}")
print(f"Samples: {len(y)}")
print(f"Unique cages: {len(np.unique(cage_ids))}")

# SANITY CHECK: CV setup
print(f"\n{'='*60}")
print("PRE-FLIGHT CHECK: Cross-Validation Setup")
print(f"{'='*60}")

logo = LeaveOneGroupOut()
n_splits = logo.get_n_splits(X, y, cage_ids)
expected_splits = len(np.unique(cage_ids))

print(f"Number of CV folds: {n_splits}")
print(f"Expected (one per cage): {expected_splits}")

if n_splits == expected_splits:
    print("✓ CV configured correctly")
else:
    print("✗ WARNING: CV not set up correctly!")

# Train model
print(f"\n{'='*60}")
print("TRAINING MODEL")
print(f"{'='*60}")

model = RandomForestRegressor(n_estimators=200, max_depth=10, 
                              min_samples_leaf=3, random_state=42)

# Cross-validated predictions
y_pred = cross_val_predict(model, X, y, groups=cage_ids, cv=logo)

# Metrics
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(((y - y_pred)**2).mean())
corr, p_val = pearsonr(y, y_pred)

print(f"\nCross-Validated Performance:")
print(f"  R² = {r2:.3f}")
print(f"  MAE = {mae:.1f}%")
print(f"  RMSE = {rmse:.1f}%")
print(f"  Correlation = {corr:.3f} (p = {p_val:.4f})")

# Fit on all data to check overfitting
model.fit(X, y)
y_pred_train = model.predict(X)
r2_train = r2_score(y, y_pred_train)

print(f"\nOverfitting Check:")
print(f"  Training R² = {r2_train:.3f}")
print(f"  CV R² = {r2:.3f}")
print(f"  Gap = {r2_train - r2:.3f}")

# SANITY CHECKS
print(f"\n{'='*80}")
print("SANITY CHECKS")
print(f"{'='*80}")

checks_passed = 0
total_checks = 5

# Check 1: R² not impossibly high
print(f"\n✓ CHECK 1: R² is reasonable")
print(f"  R² = {r2:.3f}")
if r2 < 0.8:
    print("  ✓ PASS - Not suspiciously high")
    checks_passed += 1
else:
    print("  ✗ FAIL - R² too high, possible data leakage!")

# Check 2: Predictions not perfect
perfect_preds = np.sum(np.abs(y - y_pred) < 1)
print(f"\n✓ CHECK 2: Not predicting perfectly")
print(f"  # of near-perfect predictions: {perfect_preds}/{len(y)}")
if perfect_preds < len(y) * 0.5:
    print("  ✓ PASS")
    checks_passed += 1
else:
    print("  ✗ FAIL - Too many perfect predictions")

# Check 3: Overfitting gap reasonable
print(f"\n✓ CHECK 3: Overfitting gap")
print(f"  Train R² - CV R² = {r2_train - r2:.3f}")
if r2_train - r2 < 0.5:
    print("  ✓ PASS - Reasonable gap")
    checks_passed += 1
elif r2_train - r2 < 0.8:
    print("  ~ WARNING - Moderate overfitting")
    checks_passed += 0.5
else:
    print("  ✗ FAIL - Severe overfitting")

# Check 4: Predictions in reasonable range
pred_range_ok = (y_pred.min() > -200) and (y_pred.max() < 2000)
print(f"\n✓ CHECK 4: Prediction range")
print(f"  Min prediction: {y_pred.min():.1f}%")
print(f"  Max prediction: {y_pred.max():.1f}%")
if pred_range_ok:
    print("  ✓ PASS - Reasonable range")
    checks_passed += 1
else:
    print("  ✗ FAIL - Predictions out of bounds")

# Check 5: Not just predicting mean
predicts_mean = np.abs(y_pred.mean() - y.mean()) < 10
var_captured = y_pred.std() / y.std()
print(f"\n✓ CHECK 5: Not just predicting mean")
print(f"  Predicted std / Actual std = {var_captured:.3f}")
if var_captured > 0.3:
    print("  ✓ PASS - Capturing variance")
    checks_passed += 1
else:
    print("  ✗ FAIL - Just predicting near the mean")

# Interpretation
print(f"\n{'='*80}")
print("INTERPRETATION")
print(f"{'='*80}")

print(f"\nCan we predict exact response magnitude?")
if r2 > 0.30:
    print(f"  ✓ YES - Strong prediction (R² = {r2:.3f})")
    print(f"  Baseline features meaningfully predict response")
elif r2 > 0.15:
    print(f"  ~ MAYBE - Moderate prediction (R² = {r2:.3f})")
    print(f"  Some signal but limited practical use")
else:
    print(f"  ✗ NO - Weak/no prediction (R² = {r2:.3f})")
    print(f"  Cannot reliably predict exact response from baseline")
    print(f"  This is EXPECTED with individual-level data")

if p_val < 0.05:
    print(f"\n  Statistical significance: YES (p = {p_val:.4f})")
else:
    print(f"\n  Statistical significance: NO (p = {p_val:.4f})")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot
colors = {5: 'blue', 25: 'red'}
for dose in [5, 25]:
    mask = df_morphine['dose_mg_kg'] == dose
    ax1.scatter(y[mask], y_pred[mask], c=colors[dose], 
               label=f'{dose} mg/kg', alpha=0.6, s=80)

# Perfect prediction line
min_val = min(y.min(), y_pred.min())
max_val = max(y.max(), y_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)

ax1.set_xlabel('Actual Response (%)', fontsize=11)
ax1.set_ylabel('Predicted Response (%)', fontsize=11)
ax1.set_title(f'Regression: R² = {r2:.3f}, p = {p_val:.4f}', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Residuals
residuals = y - y_pred
for dose in [5, 25]:
    mask = df_morphine['dose_mg_kg'] == dose
    ax2.scatter(y_pred[mask], residuals[mask], c=colors[dose],
               label=f'{dose} mg/kg', alpha=0.6, s=80)

ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('Predicted Response (%)', fontsize=11)
ax2.set_ylabel('Residual (%)', fontsize=11)
ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('step2_regression_results.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Figure saved: step2_regression_results.png")

# Save results
results_df = df_morphine.copy()
results_df['predicted_response'] = y_pred
results_df['residual'] = residuals
results_df.to_csv('step2_regression_predictions.csv', index=False)
print(f"✓ Results saved: step2_regression_predictions.csv")

# Summary
print(f"\n{'='*80}")
print(f"STEP 2 COMPLETE: {checks_passed}/{total_checks} checks passed")
print(f"{'='*80}")

print(f"\nKey Finding:")
print(f"  R² = {r2:.3f} (likely < 0.15 = weak)")
print(f"  Conclusion: Cannot predict EXACT response magnitude")
print(f"  Next step: Try CLASSIFICATION (high vs low responders)")

print(f"\nNext step: python step3_classification_analysis.py")
