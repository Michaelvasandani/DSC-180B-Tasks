"""
DIRECT STATISTICAL TEST: DOES ESTROUS AFFECT MORPHINE RESPONSE?

This is different from prediction!

Prediction question: Can we predict who will respond high vs low?
Effect question: Does being in estrous change the response magnitude?

Tests:
1. In estrous vs not in estrous - response comparison
2. Cycling vs non-cycling - response comparison  
3. Estrous × Dose interaction - does estrous affect 5mg vs 25mg differently?
4. Continuous relationships - correlation between estrous metrics and response
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
import seaborn as sns

print("="*80)
print("ESTROUS EFFECTS ON MORPHINE RESPONSE")
print("="*80)
print("\nDirect statistical tests (not prediction!)")
print("="*80)

# Load data
data = pd.read_csv('morphine_with_real_estrous.csv')

print(f"\nTotal animals: {len(data)}")
print(f"Doses: {sorted(data['dose_mg_kg_y'].unique())}")
print(f"  5 mg/kg: {(data['dose_mg_kg_y']==5).sum()}")
print(f"  25 mg/kg: {(data['dose_mg_kg_y']==25).sum()}")

# Filter to animals with estrous data
data_with_estrous = data[data['p_lowU_at_dose'].notna()].copy()
print(f"\nAnimals with estrous data: {len(data_with_estrous)}")

# ============================================================================
# TEST 1: IN ESTROUS VS NOT IN ESTROUS
# ============================================================================

print("\n" + "="*80)
print("TEST 1: IN ESTROUS VS NOT IN ESTROUS AT DOSING")
print("="*80)

# Binary classification: in estrous (p_lowU > 0.5) or not
data_with_estrous['in_estrous'] = (data_with_estrous['p_lowU_at_dose'] > 0.5).astype(int)

print(f"\nThreshold: p_lowU > 0.5")
print(f"In estrous: {data_with_estrous['in_estrous'].sum()} animals")
print(f"Not in estrous: {(data_with_estrous['in_estrous']==0).sum()} animals")

# Overall comparison (all doses)
print("\n" + "-"*80)
print("ALL DOSES COMBINED")
print("-"*80)

in_estrous_response = data_with_estrous[data_with_estrous['in_estrous']==1]['response_pct']
not_estrous_response = data_with_estrous[data_with_estrous['in_estrous']==0]['response_pct']

print(f"\nIn estrous (n={len(in_estrous_response)}):")
print(f"  Mean: {in_estrous_response.mean():.1f}%")
print(f"  Median: {in_estrous_response.median():.1f}%")
print(f"  Std: {in_estrous_response.std():.1f}%")
print(f"  Range: [{in_estrous_response.min():.1f}%, {in_estrous_response.max():.1f}%]")

print(f"\nNot in estrous (n={len(not_estrous_response)}):")
print(f"  Mean: {not_estrous_response.mean():.1f}%")
print(f"  Median: {not_estrous_response.median():.1f}%")
print(f"  Std: {not_estrous_response.std():.1f}%")
print(f"  Range: [{not_estrous_response.min():.1f}%, {not_estrous_response.max():.1f}%]")

if len(in_estrous_response) > 0 and len(not_estrous_response) > 0:
    stat, p_val = mannwhitneyu(in_estrous_response, not_estrous_response)
    print(f"\nMann-Whitney U test:")
    print(f"  U statistic: {stat:.2f}")
    print(f"  p-value: {p_val:.4f}")
    
    if p_val < 0.05:
        print(f"  ✓ SIGNIFICANT - Estrous affects response!")
        if in_estrous_response.mean() > not_estrous_response.mean():
            print(f"  → Animals IN estrous have HIGHER response")
        else:
            print(f"  → Animals IN estrous have LOWER response")
    else:
        print(f"  ✗ Not significant - No estrous effect")

# By dose
print("\n" + "-"*80)
print("BY DOSE")
print("-"*80)

for dose in sorted(data_with_estrous['dose_mg_kg_y'].unique()):
    dose_data = data_with_estrous[data_with_estrous['dose_mg_kg_y']==dose]
    
    in_est = dose_data[dose_data['in_estrous']==1]['response_pct']
    not_est = dose_data[dose_data['in_estrous']==0]['response_pct']
    
    print(f"\n{dose} mg/kg:")
    print(f"  In estrous: {in_est.mean():.1f}% (n={len(in_est)})")
    print(f"  Not in estrous: {not_est.mean():.1f}% (n={len(not_est)})")
    
    if len(in_est) > 0 and len(not_est) > 0:
        stat, p_val = mannwhitneyu(in_est, not_est)
        print(f"  Mann-Whitney p: {p_val:.4f}", end='')
        if p_val < 0.05:
            print(" ✓ SIGNIFICANT")
        else:
            print(" ✗ Not significant")

# ============================================================================
# TEST 2: CYCLING VS NON-CYCLING
# ============================================================================

print("\n" + "="*80)
print("TEST 2: STRONG CYCLERS VS WEAK CYCLERS")
print("="*80)

# Use median split on cycling strength
median_cycling = data_with_estrous['cycling_strength_real'].median()
data_with_estrous['strong_cycler'] = (data_with_estrous['cycling_strength_real'] > median_cycling).astype(int)

print(f"\nMedian cycling strength: {median_cycling:.3f}")
print(f"Strong cyclers: {data_with_estrous['strong_cycler'].sum()} animals")
print(f"Weak cyclers: {(data_with_estrous['strong_cycler']==0).sum()} animals")

strong_response = data_with_estrous[data_with_estrous['strong_cycler']==1]['response_pct']
weak_response = data_with_estrous[data_with_estrous['strong_cycler']==0]['response_pct']

print(f"\nStrong cyclers (n={len(strong_response)}):")
print(f"  Mean: {strong_response.mean():.1f}%")
print(f"  Median: {strong_response.median():.1f}%")

print(f"\nWeak cyclers (n={len(weak_response)}):")
print(f"  Mean: {weak_response.mean():.1f}%")
print(f"  Median: {weak_response.median():.1f}%")

if len(strong_response) > 0 and len(weak_response) > 0:
    stat, p_val = mannwhitneyu(strong_response, weak_response)
    print(f"\nMann-Whitney U test:")
    print(f"  p-value: {p_val:.4f}")
    
    if p_val < 0.05:
        print(f"  ✓ SIGNIFICANT - Cycling strength affects response!")
    else:
        print(f"  ✗ Not significant - No cycling effect")

# ============================================================================
# TEST 3: CONTINUOUS CORRELATIONS
# ============================================================================

print("\n" + "="*80)
print("TEST 3: CONTINUOUS CORRELATIONS")
print("="*80)

estrous_vars = ['p_lowU_at_dose', 'baseline_p_lowU_mean', 'cycling_strength_real',
                'baseline_p_lowU_max', 'n_estrous_days_baseline']

print("\nCorrelations with morphine response:")
print(f"{'Variable':<30} {'Pearson r':<12} {'p-value':<12} {'Spearman ρ':<12} {'p-value'}")
print("-"*80)

for var in estrous_vars:
    if var in data_with_estrous.columns:
        valid_data = data_with_estrous[[var, 'response_pct']].dropna()
        
        if len(valid_data) > 3:
            # Pearson (linear)
            r_pearson, p_pearson = pearsonr(valid_data[var], valid_data['response_pct'])
            
            # Spearman (monotonic, non-linear)
            r_spearman, p_spearman = spearmanr(valid_data[var], valid_data['response_pct'])
            
            print(f"{var:<30} {r_pearson:>6.3f}      {p_pearson:>7.4f}     "
                  f"{r_spearman:>6.3f}      {p_spearman:>7.4f}")
            
            if p_pearson < 0.05 or p_spearman < 0.05:
                print(f"  → ✓ Significant correlation found!")

# ============================================================================
# TEST 4: ESTROUS × DOSE INTERACTION
# ============================================================================

print("\n" + "="*80)
print("TEST 4: ESTROUS × DOSE INTERACTION")
print("="*80)
print("\nQuestion: Does estrous affect 5mg vs 25mg DIFFERENTLY?")

# Create groups
data_with_estrous['group'] = (
    data_with_estrous['dose_mg_kg_y'].astype(str) + '_' + 
    data_with_estrous['in_estrous'].astype(str)
)

print("\n" + "-"*80)
print("GROUP MEANS")
print("-"*80)

group_stats = data_with_estrous.groupby(['dose_mg_kg_y', 'in_estrous'])['response_pct'].agg([
    ('n', 'count'),
    ('mean', 'mean'),
    ('std', 'std')
]).reset_index()

print(f"\n{'Dose':<10} {'Estrous':<15} {'N':<5} {'Mean Response':<15} {'Std'}")
print("-"*80)
for _, row in group_stats.iterrows():
    estrous_label = 'In Estrous' if row['in_estrous']==1 else 'Not Estrous'
    print(f"{row['dose_mg_kg_y']:>4.0f} mg/kg {estrous_label:<15} {row['n']:>3.0f}  "
          f"{row['mean']:>7.1f}%         {row['std']:>6.1f}%")

# Two-way ANOVA (if enough samples)
print("\n" + "-"*80)
print("TWO-WAY ANOVA")
print("-"*80)

# Check if we have enough samples
if len(group_stats) == 4 and all(group_stats['n'] >= 2):
    try:
        from statsmodels.formula.api import ols
        import statsmodels.api as sm
        
        # Prepare data
        anova_data = data_with_estrous.copy()
        anova_data['dose_str'] = anova_data['dose_mg_kg_y'].astype(str) + 'mg'
        anova_data['estrous_str'] = anova_data['in_estrous'].map({0: 'NotEstrous', 1: 'Estrous'})
        
        # Two-way ANOVA
        model = ols('response_pct ~ C(dose_str) * C(estrous_str)', data=anova_data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        print("\nANOVA Results:")
        print(anova_table)
        
        # Extract p-values
        p_dose = anova_table.loc['C(dose_str)', 'PR(>F)']
        p_estrous = anova_table.loc['C(estrous_str)', 'PR(>F)']
        p_interaction = anova_table.loc['C(dose_str):C(estrous_str)', 'PR(>F)']
        
        print(f"\nMain effects:")
        print(f"  Dose: p={p_dose:.4f}", "✓ SIGNIFICANT" if p_dose < 0.05 else "✗ Not significant")
        print(f"  Estrous: p={p_estrous:.4f}", "✓ SIGNIFICANT" if p_estrous < 0.05 else "✗ Not significant")
        print(f"\nInteraction:")
        print(f"  Dose × Estrous: p={p_interaction:.4f}", "✓✓ SIGNIFICANT INTERACTION!" if p_interaction < 0.05 else "✗ No interaction")
        
        if p_interaction < 0.05:
            print("\n  → Estrous affects 5mg and 25mg DIFFERENTLY!")
        else:
            print("\n  → Estrous effect is same across doses (or absent)")
            
    except Exception as e:
        print(f"⚠ Cannot run ANOVA: {e}")
        print("  (Need more balanced samples)")
else:
    print("⚠ Not enough samples for two-way ANOVA")
    print("  (Need at least 2 animals in each of 4 groups)")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Panel A: In estrous vs not (all doses)
ax = axes[0, 0]
if len(in_estrous_response) > 0 and len(not_estrous_response) > 0:
    data_plot = [not_estrous_response, in_estrous_response]
    bp = ax.boxplot(data_plot, labels=['Not Estrous', 'In Estrous'],
                   widths=0.6, patch_artist=True, showfliers=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    # Add scatter
    for i, d in enumerate(data_plot, 1):
        y = d
        x = np.random.normal(i, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.5, s=60, edgecolors='black', linewidth=0.5)
    
    ax.set_ylabel('Morphine Response (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'A. Estrous Effect (All Doses)\np={p_val:.4f}',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

# Panel B: By dose
ax = axes[0, 1]
doses = sorted(data_with_estrous['dose_mg_kg_y'].unique())
x_pos = np.arange(len(doses))
width = 0.35

means_not_est = []
means_in_est = []
sems_not_est = []
sems_in_est = []

for dose in doses:
    dose_data = data_with_estrous[data_with_estrous['dose_mg_kg_y']==dose]
    not_est = dose_data[dose_data['in_estrous']==0]['response_pct']
    in_est = dose_data[dose_data['in_estrous']==1]['response_pct']
    
    means_not_est.append(not_est.mean() if len(not_est) > 0 else 0)
    means_in_est.append(in_est.mean() if len(in_est) > 0 else 0)
    sems_not_est.append(not_est.sem() if len(not_est) > 0 else 0)
    sems_in_est.append(in_est.sem() if len(in_est) > 0 else 0)

bars1 = ax.bar(x_pos - width/2, means_not_est, width, yerr=sems_not_est,
              label='Not Estrous', color='lightblue', alpha=0.7, 
              edgecolor='black', capsize=5)
bars2 = ax.bar(x_pos + width/2, means_in_est, width, yerr=sems_in_est,
              label='In Estrous', color='lightcoral', alpha=0.7,
              edgecolor='black', capsize=5)

ax.set_ylabel('Mean Response (%)', fontsize=11, fontweight='bold')
ax.set_xlabel('Dose', fontsize=11, fontweight='bold')
ax.set_title('B. Estrous Effect by Dose', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{d} mg/kg' for d in doses])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel C: Interaction plot
ax = axes[0, 2]
for estrous_status in [0, 1]:
    means = []
    sems = []
    for dose in doses:
        subset = data_with_estrous[(data_with_estrous['dose_mg_kg_y']==dose) &
                                  (data_with_estrous['in_estrous']==estrous_status)]
        means.append(subset['response_pct'].mean() if len(subset) > 0 else 0)
        sems.append(subset['response_pct'].sem() if len(subset) > 0 else 0)
    
    label = 'In Estrous' if estrous_status == 1 else 'Not Estrous'
    color = 'red' if estrous_status == 1 else 'blue'
    ax.errorbar(doses, means, yerr=sems, marker='o', linewidth=2.5, 
               markersize=10, capsize=5, label=label, color=color)

ax.set_xlabel('Dose (mg/kg)', fontsize=11, fontweight='bold')
ax.set_ylabel('Mean Response (%)', fontsize=11, fontweight='bold')
ax.set_title('C. Interaction Plot', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel D: p_lowU vs response scatter
ax = axes[1, 0]
colors_dose = ['red' if d==5 else 'blue' for d in data_with_estrous['dose_mg_kg_y']]
ax.scatter(data_with_estrous['p_lowU_at_dose'], data_with_estrous['response_pct'],
          c=colors_dose, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('p_lowU at Dosing', fontsize=11, fontweight='bold')
ax.set_ylabel('Morphine Response (%)', fontsize=11, fontweight='bold')
ax.set_title('D. Continuous Relationship', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', label='5 mg/kg'),
                  Patch(facecolor='blue', label='25 mg/kg')]
ax.legend(handles=legend_elements)

# Panel E: Cycling strength vs response
ax = axes[1, 1]
ax.scatter(data_with_estrous['cycling_strength_real'], data_with_estrous['response_pct'],
          c=colors_dose, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Cycling Strength', fontsize=11, fontweight='bold')
ax.set_ylabel('Morphine Response (%)', fontsize=11, fontweight='bold')
ax.set_title('E. Cycling vs Response', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(handles=legend_elements)

# Panel F: Summary statistics
ax = axes[1, 2]
summary_text = f"""
ESTROUS EFFECTS SUMMARY

In Estrous vs Not:
  All doses: p={p_val:.4f}
  {'✓ SIGNIFICANT' if p_val < 0.05 else '✗ Not significant'}

Strong vs Weak Cyclers:
  p={mannwhitneyu(strong_response, weak_response)[1]:.4f if len(strong_response)>0 and len(weak_response)>0 else 'N/A'}
  {'✓ SIGNIFICANT' if len(strong_response)>0 and len(weak_response)>0 and mannwhitneyu(strong_response, weak_response)[1]<0.05 else '✗ Not significant'}

Continuous Correlations:
  All p > 0.05
  {'✗ No significant correlations' if all(pearsonr(data_with_estrous[var].dropna(), data_with_estrous.loc[data_with_estrous[var].notna(), 'response_pct'])[1] > 0.05 for var in estrous_vars if var in data_with_estrous.columns and len(data_with_estrous[var].dropna()) > 3) else '✓ Some correlations found'}

CONCLUSION:
{'Estrous does NOT affect morphine' if p_val > 0.05 else 'Estrous DOES affect morphine!'}
response magnitude
"""

ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
       fontsize=10, verticalalignment='center', family='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.axis('off')

plt.tight_layout()
plt.savefig('estrous_direct_effects.png', dpi=300, bbox_inches='tight')
print("✓ Saved: estrous_direct_effects.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print("\nDirect statistical tests of estrous effects on morphine response:")
print(f"\n1. In estrous vs not: p={p_val:.4f}", "✓ SIGNIFICANT" if p_val < 0.05 else "✗ Not significant")

if len(strong_response) > 0 and len(weak_response) > 0:
    cycling_p = mannwhitneyu(strong_response, weak_response)[1]
    print(f"2. Strong vs weak cyclers: p={cycling_p:.4f}", "✓ SIGNIFICANT" if cycling_p < 0.05 else "✗ Not significant")

sig_corr = sum(1 for var in estrous_vars if var in data_with_estrous.columns 
               and len(data_with_estrous[var].dropna()) > 3
               and pearsonr(data_with_estrous[var].dropna(), 
                          data_with_estrous.loc[data_with_estrous[var].notna(), 'response_pct'])[1] < 0.05)
print(f"3. Continuous correlations: {sig_corr}/{len(estrous_vars)} significant")

print("\n" + "="*80)
print("FOR YOUR PAPER")
print("="*80)

if p_val < 0.05:
    print("\n✓ ESTROUS AFFECTS MORPHINE RESPONSE!")
    print("\nResults section:")
    print(f"  'Animals in estrous at dosing showed {'higher' if in_estrous_response.mean() > not_estrous_response.mean() else 'lower'}")
    print(f"   morphine responses ({in_estrous_response.mean():.1f}%) compared to animals")
    print(f"   not in estrous ({not_estrous_response.mean():.1f}%), p={p_val:.4f}.'")
else:
    print("\n✗ Estrous does NOT affect morphine response magnitude")
    print("\nResults section:")
    print(f"  'Estrous cycle phase at dosing did not significantly affect")
    print(f"   morphine response magnitude (in estrous: {in_estrous_response.mean():.1f}%,")
    print(f"   not in estrous: {not_estrous_response.mean():.1f}%, p={p_val:.4f}).'")

print("\n" + "="*80)
