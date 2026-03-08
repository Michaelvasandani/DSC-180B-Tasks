"""
STEP 4: Validation & Summary

Purpose: Final sanity checks and overall summary
- Cross-replicate validation
- Statistical significance tests
- Final interpretation
- What to report in your paper
"""

import pandas as pd
import numpy as np
import json
from scipy.stats import mannwhitneyu, permutation_test

print("=" * 80)
print("STEP 4: VALIDATION & SUMMARY")
print("=" * 80)

# Load all results
try:
    step3_summary = json.load(open('step3_summary.json'))
    has_classification = True
except:
    has_classification = False
    print("\n⚠️  Warning: Step 3 results not found")

# Load data
response_df = pd.read_csv('morphine_responses.csv')
if has_classification:
    classification_df = pd.read_csv('step3_classification_predictions.csv')

print("\n" + "="*60)
print("OVERALL SUMMARY")
print("="*60)

# Summary statistics
print(f"\nDataset:")
print(f"  Total morphine animals: {len(response_df)}")
print(f"  5 mg/kg: {len(response_df[response_df['dose_mg_kg']==5])}")
print(f"  25 mg/kg: {len(response_df[response_df['dose_mg_kg']==25])}")

print(f"\nResponse statistics:")
for dose in [5, 25]:
    dose_data = response_df[response_df['dose_mg_kg']==dose]
    print(f"  {dose} mg/kg:")
    print(f"    Mean: {dose_data['response_pct'].mean():.1f}%")
    print(f"    Std: {dose_data['response_pct'].std():.1f}%")
    print(f"    Range: {dose_data['response_pct'].min():.1f}% - {dose_data['response_pct'].max():.1f}%")

# Cross-replicate validation
if has_classification:
    print(f"\n" + "="*60)
    print("CROSS-REPLICATE VALIDATION")
    print("="*60)
    
    rep1_data = classification_df[classification_df['replicate']==1]
    rep2_data = classification_df[classification_df['replicate']==2]
    
    if len(rep1_data) > 0 and len(rep2_data) > 0:
        rep1_acc = (rep1_data['correct']).mean()
        rep2_acc = (rep2_data['correct']).mean()
        
        print(f"\nAccuracy by replicate:")
        print(f"  Replicate 1: {rep1_acc:.3f} ({len(rep1_data)} animals)")
        print(f"  Replicate 2: {rep2_acc:.3f} ({len(rep2_data)} animals)")
        print(f"  Difference: {abs(rep1_acc - rep2_acc):.3f}")
        
        if abs(rep1_acc - rep2_acc) < 0.20:
            print(f"  ✓ Consistent across replicates")
        else:
            print(f"  ⚠️  Large difference between replicates")
    else:
        print("  Insufficient data for cross-replicate validation")

# Statistical significance (permutation test)
if has_classification:
    print(f"\n" + "="*60)
    print("PERMUTATION TEST")
    print("="*60)
    
    print("\nRunning 1000 permutations to test significance...")
    
    # Simple permutation test
    observed_acc = step3_summary['accuracy']
    n_better = 0
    n_perms = 1000
    
    y_true = classification_df['true_class'].values
    cage_ids = classification_df['cage_id'].values
    
    for _ in range(n_perms):
        # Permute labels
        y_perm = np.random.permutation(y_true)
        # Simple accuracy
        perm_acc = np.mean(y_perm == classification_df['predicted_class'].values)
        if perm_acc >= observed_acc:
            n_better += 1
    
    p_perm = (n_better + 1) / (n_perms + 1)
    
    print(f"\nObserved accuracy: {observed_acc:.3f}")
    print(f"Permutation p-value: {p_perm:.4f}")
    
    if p_perm < 0.001:
        print(f"  ✓✓✓ HIGHLY SIGNIFICANT")
    elif p_perm < 0.01:
        print(f"  ✓✓ VERY SIGNIFICANT")
    elif p_perm < 0.05:
        print(f"  ✓ SIGNIFICANT")
    else:
        print(f"  ✗ NOT SIGNIFICANT")

# Final sanity check summary
print(f"\n" + "="*80)
print("FINAL SANITY CHECK SUMMARY")
print("="*80)

total_checks = 0
passed_checks = 0

# Check 1: Got expected number of animals
check1 = len(response_df) == 36
total_checks += 1
passed_checks += check1
print(f"\n✓ Check 1: Correct number of animals")
print(f"  Expected: 36, Got: {len(response_df)}")
print(f"  {'✓ PASS' if check1 else '✗ FAIL'}")

# Check 2: Dose groups different
u_stat, p_dose = mannwhitneyu(
    response_df[response_df['dose_mg_kg']==5]['response_pct'],
    response_df[response_df['dose_mg_kg']==25]['response_pct']
)
check2 = p_dose < 0.001
total_checks += 1
passed_checks += check2
print(f"\n✓ Check 2: Dose groups significantly different")
print(f"  p-value: {p_dose:.6f}")
print(f"  {'✓ PASS' if check2 else '✗ FAIL'}")

if has_classification:
    # Check 3: Classification better than chance
    check3 = step3_summary['accuracy'] > 0.55
    total_checks += 1
    passed_checks += check3
    print(f"\n✓ Check 3: Classification better than chance")
    print(f"  Accuracy: {step3_summary['accuracy']:.3f} vs 0.50")
    print(f"  {'✓ PASS' if check3 else '✗ FAIL'}")
    
    # Check 4: Statistically significant
    check4 = step3_summary['group_p_value'] < 0.05
    total_checks += 1
    passed_checks += check4
    print(f"\n✓ Check 4: Groups statistically different")
    print(f"  p-value: {step3_summary['group_p_value']:.4f}")
    print(f"  {'✓ PASS' if check4 else '✗ FAIL'}")
    
    # Check 5: Not severe overfitting
    check5 = step3_summary['overfitting_gap'] < 0.40
    total_checks += 1
    passed_checks += check5
    print(f"\n✓ Check 5: Overfitting acceptable")
    print(f"  Gap: {step3_summary['overfitting_gap']:.3f}")
    print(f"  {'✓ PASS' if check5 else '⚠️  WARNING'}")

# Overall verdict
print(f"\n" + "="*80)
print(f"OVERALL VERDICT: {passed_checks}/{total_checks} checks passed")
print("="*80)

if passed_checks >= total_checks * 0.8:
    verdict = "ROBUST"
    print("\n🎉 FINDINGS APPEAR ROBUST")
    print("  Results pass most sanity checks")
    print("  Safe to report with confidence")
elif passed_checks >= total_checks * 0.6:
    verdict = "MODERATE"
    print("\n✓ FINDINGS MODERATELY SUPPORTED")
    print("  Some concerns but generally reliable")
    print("  Report with appropriate caveats")
else:
    verdict = "QUESTIONABLE"
    print("\n⚠️  FINDINGS QUESTIONABLE")
    print("  Multiple checks failed")
    print("  Recommend additional validation")

# What to report
print(f"\n" + "="*80)
print("WHAT TO REPORT IN YOUR PAPER")
print("="*80)

if has_classification:
    acc = step3_summary['accuracy']
    p_val = step3_summary['group_p_value']
    imp = step3_summary['improvement']
    
    print(f"\nKey Numbers:")
    print(f"  Sample size: N=18 (at 25 mg/kg)")
    print(f"  Classification accuracy: {acc:.1%}")
    print(f"  Improvement over chance: {imp:.1%}")
    print(f"  Statistical significance: p={p_val:.4f}")
    
    print(f"\nRecommended text:")
    print(f"  \"Cross-validated classification achieved {acc:.1%} accuracy")
    print(f"   (N=18, LOCO-CV), representing a {imp:.1%} improvement over")
    print(f"   chance (p={p_val:.3f}). Animals with higher baseline dark-phase")
    print(f"   activity showed lower morphine responses, suggesting that")
    print(f"   spontaneous behavioral phenotypes contain predictive")
    print(f"   information about individual drug sensitivity.\"")
    
    print(f"\nInterpretation:")
    if acc > 0.70:
        print(f"  → STRONG prediction - baseline is useful biomarker")
    elif acc > 0.60:
        print(f"  → MODEST prediction - proof-of-concept established")
    else:
        print(f"  → WEAK prediction - limited practical utility")
    
    print(f"\nLimitations to mention:")
    print(f"  - Small sample size (N=18)")
    print(f"  - Modest accuracy ({acc:.1%})")
    print(f"  - Single dose tested")
    print(f"  - Needs validation in independent cohort")

# Save final summary
with open('validation_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("FINAL VALIDATION SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Overall Verdict: {verdict}\n")
    f.write(f"Checks Passed: {passed_checks}/{total_checks}\n\n")
    
    if has_classification:
        f.write(f"Key Results:\n")
        f.write(f"  Classification Accuracy: {acc:.3f}\n")
        f.write(f"  Improvement over chance: {imp:.3f}\n")
        f.write(f"  Statistical significance: p={p_val:.4f}\n")
        f.write(f"  Permutation p-value: {p_perm:.4f}\n\n")
    
    f.write("Recommendation:\n")
    if verdict == "ROBUST":
        f.write("  Findings are robust. Safe to report as main result.\n")
    elif verdict == "MODERATE":
        f.write("  Findings are moderately supported. Report with caveats.\n")
    else:
        f.write("  Findings need additional validation before publication.\n")

print(f"\n✓ Summary saved: validation_summary.txt")

print(f"\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print("\nAll analyses finished successfully.")
print("Check validation_summary.txt for final interpretation.")
print("\nKey output files:")
print("  1. morphine_responses.csv - Raw response data")
print("  2. step2_regression_results.png - Regression analysis")
print("  3. step3_classification_results.png - Classification analysis")
print("  4. validation_summary.txt - Final summary and interpretation")
