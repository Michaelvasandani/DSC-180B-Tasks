"""
Reduced Feature Set for Small Sample Size
Use only top 5-8 most theoretically important features
"""

import pandas as pd
from modeling_pipeline import OpioidSensitivityPredictor


def select_core_features(features_df):
    """
    Select only core features for small sample size (N=12)
    
    Based on research hypotheses and theoretical importance
    """
    
    # Core features (5-8 only for N=12)
    core_features = [
        # Locomotion (known predictor)
        'mean_distance_cm_per_s',
        'distance_cv',  # Hypothesis H3
        
        # Circadian (Hypothesis H1)
        'circadian_amplitude',
        'circadian_r_squared',
        'light_dark_ratio',
        
        # Rest fragmentation (Hypothesis H2)
        'rest_fragmentation_index',
        
        # Bout structure (Hypothesis H4)
        'locomotion_bout_frequency_per_hour',
        'freq_duration_tradeoff_residual',  # Will be calculated
    ]
    
    # Keep only core features + metadata
    metadata_cols = ['cage_id', 'replicate', 'dose_mg_kg', 'baseline_start', 
                     'baseline_end', 'baseline_days']
    
    available_features = [f for f in core_features if f in features_df.columns]
    keep_cols = metadata_cols + available_features
    
    reduced_df = features_df[keep_cols].copy()
    
    print(f"Reduced from {len(features_df.columns)} to {len(available_features)} core features")
    print(f"Core features: {available_features}")
    
    return reduced_df


def run_reduced_model(features_file='baseline_features.csv', 
                     outcomes_file='morphine_response.csv'):
    """Run modeling with reduced feature set"""
    
    print("="*70)
    print(" REDUCED FEATURE MODEL (FOR SMALL SAMPLE SIZE)")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    features_df = pd.read_csv(features_file)
    responses_df = pd.read_csv(outcomes_file)
    
    # Reduce features
    print("\nReducing to core features...")
    reduced_features = select_core_features(features_df)
    
    # Merge
    modeling_df = reduced_features.merge(
        responses_df[['cage_id', 'morphine_response', 'baseline_mean', 'post_dose_mean']],
        on='cage_id',
        how='inner'
    )
    
    print(f"\nFinal dataset: {len(modeling_df)} cages")
    
    # Initialize predictor
    predictor = OpioidSensitivityPredictor(modeling_df)
    predictor.data = modeling_df
    
    # Prepare data
    print("\nPreparing data...")
    X, y = predictor.prepare_data(outcome_col='morphine_response')
    
    # Calculate trade-off residual if needed
    if 'locomotion_bout_freq_raw' in X.columns:
        predictor.calculate_freq_duration_tradeoff()
    
    print(f"\n{'='*70}")
    print(" TRAINING MODELS")
    print(f"{'='*70}")
    print(f"Samples: {len(X)}")
    print(f"Features: {len(X.columns)}")
    print(f"Ratio: {len(X) / len(X.columns):.2f} samples per feature")
    
    if len(X) / len(X.columns) < 2:
        print("\n⚠️  WARNING: Very low sample-to-feature ratio!")
        print("   Results may be unreliable. Consider:")
        print("   1. Reducing features further (use only 3-5)")
        print("   2. Pooling both replicates as separate samples")
        print("   3. Using simpler models (Ridge only)")
    
    # Train models
    print("\nTraining Ridge Regression...")
    ridge_results = predictor.train_model('ridge', alpha=10.0)  # Higher regularization
    
    print("\nTraining Elastic Net...")
    elastic_results = predictor.train_model('elastic_net', alpha=5.0, l1_ratio=0.5)
    
    # Get best model
    best_model = max(predictor.results.keys(), 
                    key=lambda x: predictor.results[x]['mean_r2'])
    
    print(f"\n{'='*70}")
    print(" RESULTS")
    print(f"{'='*70}")
    
    for model_name, results in predictor.results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  R² = {results['mean_r2']:.3f} ± {results['std_r2']:.3f}")
        print(f"  MAE = {results['mean_mae']:.2f}")
    
    print(f"\nBest model: {best_model}")
    
    # Feature importance
    print(f"\n{'='*70}")
    print(" FEATURE IMPORTANCE")
    print(f"{'='*70}")
    
    if best_model in predictor.feature_importance:
        importance = predictor.feature_importance[best_model]
        for _, row in importance.iterrows():
            print(f"  {row['feature']:40s} {row['importance']:.4f}")
    
    # Generate outputs
    print("\nGenerating outputs...")
    predictor.plot_results('model_results_reduced.png')
    print("✓ Saved: model_results_reduced.png")
    
    predictor.generate_report('model_report_reduced.txt')
    print("✓ Saved: model_report_reduced.txt")
    
    return predictor


if __name__ == '__main__':
    predictor = run_reduced_model()
