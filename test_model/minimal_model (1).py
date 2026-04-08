"""
Minimal 3-Feature Model with Diagnostics
For N=12 samples, use only 3 features (4:1 ratio)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt


def run_minimal_model():
    """Run with just 3 features for N=12 samples"""
    
    print("="*70)
    print(" MINIMAL 3-FEATURE MODEL (N=12)")
    print("="*70)
    
    # Load data
    features_df = pd.read_csv('baseline_features.csv')
    responses_df = pd.read_csv('morphine_response.csv')
    
    # Merge
    df = features_df.merge(
        responses_df[['cage_id', 'morphine_response']],
        on='cage_id',
        how='inner'
    )
    
    print(f"\nTotal samples: {len(df)}")
    
    # Select ONLY 3 features based on hypotheses
    features_to_use = [
        'mean_distance_cm_per_s',      # Locomotion (known predictor)
        'circadian_amplitude',          # H1: Circadian strength
        'rest_fragmentation_index',     # H2: Rest disruption
    ]
    
    # Check which are available
    available = [f for f in features_to_use if f in df.columns]
    print(f"\nUsing {len(available)} features:")
    for f in available:
        print(f"  - {f}")
    
    # Prepare data
    X = df[available].copy()
    y = df['morphine_response'].copy()
    cages = df['cage_id'].values
    
    # Handle missing values
    for col in X.columns:
        if X[col].isnull().any():
            print(f"\n⚠️  {col} has {X[col].isnull().sum()} missing values, filling with median")
            X[col].fillna(X[col].median(), inplace=True)
    
    print(f"\nData shape: X = {X.shape}, y = {y.shape}")
    print(f"Samples per feature: {X.shape[0] / X.shape[1]:.2f}")
    
    # Check for infinite or extreme values
    print("\nData quality check:")
    for col in X.columns:
        if np.isinf(X[col]).any():
            print(f"  ⚠️  {col} has infinite values!")
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            X[col].fillna(X[col].median(), inplace=True)
        
        print(f"  {col}: range [{X[col].min():.2f}, {X[col].max():.2f}], mean={X[col].mean():.2f}")
    
    print(f"\nOutcome range: [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\n" + "="*70)
    print(" LEAVE-ONE-CAGE-OUT CROSS-VALIDATION")
    print("="*70)
    
    logo = LeaveOneGroupOut()
    predictions = []
    actuals = []
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_scaled, y, groups=cages)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train Ridge with high regularization
        model = Ridge(alpha=10.0)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        predictions.extend(y_pred)
        actuals.extend(y_test)
        
        # Calculate fold metrics
        mae = mean_absolute_error([y_test.iloc[0]], [y_pred[0]])
        
        fold_results.append({
            'fold': fold_idx + 1,
            'test_cage': cages[test_idx][0],
            'actual': y_test.iloc[0],
            'predicted': y_pred[0],
            'error': y_pred[0] - y_test.iloc[0],
            'abs_error': mae
        })
        
        print(f"Fold {fold_idx+1:2d}: Cage {cages[test_idx][0]} | "
              f"Actual={y_test.iloc[0]:7.2f} | Pred={y_pred[0]:7.2f} | "
              f"Error={y_pred[0] - y_test.iloc[0]:7.2f}")
    
    # Calculate overall metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Manual R² calculation with diagnostics
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    
    print(f"\n" + "="*70)
    print(" CROSS-VALIDATION RESULTS")
    print("="*70)
    print(f"\nSS_res (residual sum of squares): {ss_res:.2f}")
    print(f"SS_tot (total sum of squares): {ss_tot:.2f}")
    
    if ss_tot > 0:
        r2 = 1 - (ss_res / ss_tot)
        print(f"R² = 1 - (SS_res / SS_tot) = 1 - ({ss_res:.2f} / {ss_tot:.2f}) = {r2:.3f}")
    else:
        r2 = np.nan
        print(f"R² = NaN (SS_tot = 0, no variance in outcome)")
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    correlation = np.corrcoef(actuals, predictions)[0, 1]
    
    print(f"\nMean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Correlation (actual vs predicted): {correlation:.3f}")
    
    # Interpretation
    print(f"\n" + "="*70)
    print(" INTERPRETATION")
    print("="*70)
    
    if not np.isnan(r2):
        if r2 >= 0.30:
            print(f"\n✓ STRONG PREDICTION (R² = {r2:.3f})")
            print("  Baseline features meaningfully predict morphine response")
        elif r2 >= 0.15:
            print(f"\n~ MODERATE PREDICTION (R² = {r2:.3f})")
            print("  Some predictive signal, but substantial unexplained variance")
        elif r2 >= 0:
            print(f"\n⚠️  WEAK PREDICTION (R² = {r2:.3f})")
            print("  Baseline features show minimal predictive power")
        else:
            print(f"\n✗ NEGATIVE R² (R² = {r2:.3f})")
            print("  Model performs worse than predicting the mean")
            print("  This often happens with severe overfitting in small samples")
    else:
        print(f"\n✗ UNDEFINED R² (NaN)")
        print("  No variance in outcome, or mathematical instability")
    
    # Train final model on all data for feature importance
    print(f"\n" + "="*70)
    print(" FEATURE IMPORTANCE (Full Model)")
    print("="*70)
    
    model_full = Ridge(alpha=10.0)
    model_full.fit(X_scaled, y)
    
    importance = pd.DataFrame({
        'feature': available,
        'coefficient': model_full.coef_,
        'abs_coefficient': np.abs(model_full.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    print(f"\nIntercept: {model_full.intercept_:.2f}")
    print("\nFeature coefficients (standardized):")
    for _, row in importance.iterrows():
        print(f"  {row['feature']:40s} {row['coefficient']:7.3f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Predicted vs Actual
    axes[0].scatter(actuals, predictions, alpha=0.7, s=100)
    axes[0].plot([actuals.min(), actuals.max()], 
                 [actuals.min(), actuals.max()], 
                 'k--', linewidth=1)
    axes[0].set_xlabel('Actual Response', fontsize=12)
    axes[0].set_ylabel('Predicted Response', fontsize=12)
    axes[0].set_title(f'Predicted vs Actual (R² = {r2:.3f})', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Add text with metrics
    textstr = f'MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nr = {correlation:.3f}'
    axes[0].text(0.05, 0.95, textstr, transform=axes[0].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Feature importance
    axes[1].barh(range(len(importance)), importance['abs_coefficient'])
    axes[1].set_yticks(range(len(importance)))
    axes[1].set_yticklabels(importance['feature'])
    axes[1].set_xlabel('Absolute Coefficient', fontsize=12)
    axes[1].set_title('Feature Importance', fontsize=14)
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('minimal_model_results.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: minimal_model_results.png")
    
    # Save detailed results
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv('cv_fold_results.csv', index=False)
    print(f"✓ Saved: cv_fold_results.csv")
    
    # Summary stats
    print(f"\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print(f"\nSample size: N = {len(df)}")
    print(f"Features: K = {len(available)}")
    print(f"Ratio: N/K = {len(df) / len(available):.2f}")
    print(f"\nCross-validated R² = {r2:.3f}" if not np.isnan(r2) else "\nCross-validated R² = NaN")
    print(f"Cross-validated MAE = {mae:.2f}")
    print(f"Correlation (r) = {correlation:.3f}")
    
    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation,
        'fold_results': results_df
    }


if __name__ == '__main__':
    results = run_minimal_model()
