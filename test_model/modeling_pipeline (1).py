"""
Opioid Sensitivity Predictive Modeling Pipeline
Implements cross-validated models with cage-aware splitting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class OpioidSensitivityPredictor:
    """
    Predictive model for opioid sensitivity based on baseline behavioral features
    """
    
    def __init__(self, features_df):
        """
        Initialize predictor with feature dataframe
        
        Args:
            features_df: DataFrame with baseline features (from feature_extraction.py)
        """
        self.features_df = features_df.copy()
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def load_outcome_data(self, outcome_file):
        """
        Load morphine response data (outcome variable)
        
        Args:
            outcome_file: CSV file with columns ['animal_id', 'morphine_response']
        
        The morphine_response should be calculated as:
        - Peak activity in 0-60 minutes post-injection
        - Or percent change from baseline
        - Or area under the curve
        """
        outcome_df = pd.read_csv(outcome_file)
        
        # Merge with features
        self.data = self.features_df.merge(outcome_df, on='animal_id', how='inner')
        
        print(f"Loaded outcome data for {len(self.data)} animals")
        
        return self.data
    
    def prepare_data(self, feature_cols=None, outcome_col='morphine_response'):
        """
        Prepare feature matrix and outcome vector
        
        Args:
            feature_cols: List of feature column names (if None, auto-detect)
            outcome_col: Name of outcome column
        """
        # Auto-detect feature columns (exclude metadata)
        if feature_cols is None:
            exclude_cols = ['cage_id', 'animal_id', 'animal_string_id', 'replicate', 
                          'dose_mg_kg', 'baseline_start', 'baseline_end', 
                          'baseline_days', outcome_col]
            feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        # Remove columns with too many missing values (>50%)
        missing_pct = self.data[feature_cols].isnull().sum() / len(self.data)
        valid_features = missing_pct[missing_pct < 0.5].index.tolist()
        
        print(f"Selected {len(valid_features)} features (from {len(feature_cols)} total)")
        print(f"Removed {len(feature_cols) - len(valid_features)} features with >50% missing data")
        
        # Extract features and outcome
        X = self.data[valid_features].copy()
        y = self.data[outcome_col].copy()
        
        # Handle remaining missing values (impute with median)
        for col in X.columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)
        
        # Store for later use
        self.X = X
        self.y = y
        self.feature_cols = valid_features
        self.outcome_col = outcome_col
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Outcome vector shape: {y.shape}")
        
        return X, y
    
    def get_cage_groups(self):
        """
        Get cage IDs for group-based cross-validation
        """
        return self.data['cage_id'].values
    
    def calculate_freq_duration_tradeoff(self):
        """
        Calculate bout frequency-duration trade-off residual
        Must be done across all animals after feature extraction
        """
        if 'locomotion_bout_freq_raw' in self.X.columns and 'locomotion_bout_duration_raw' in self.X.columns:
            freq = self.X['locomotion_bout_freq_raw']
            duration = self.X['locomotion_bout_duration_raw']
            
            # Fit linear regression: duration ~ frequency
            valid_idx = freq.notna() & duration.notna()
            if valid_idx.sum() > 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    freq[valid_idx], duration[valid_idx]
                )
                
                # Calculate residuals
                predicted = slope * freq + intercept
                residuals = duration - predicted
                
                self.X['freq_duration_tradeoff_residual'] = residuals
                self.feature_cols.append('freq_duration_tradeoff_residual')
                
                print(f"Calculated frequency-duration trade-off (R² = {r_value**2:.3f}, p = {p_value:.3f})")
    
    def train_model(self, model_type='elastic_net', **kwargs):
        """
        Train a predictive model with leave-one-cage-out cross-validation
        
        Args:
            model_type: One of ['ridge', 'lasso', 'elastic_net', 'random_forest', 'gradient_boosting']
            **kwargs: Model-specific hyperparameters
        
        Returns:
            Dictionary with cross-validation results
        """
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # Initialize model
        if model_type == 'ridge':
            model = Ridge(alpha=kwargs.get('alpha', 1.0))
        elif model_type == 'lasso':
            model = Lasso(alpha=kwargs.get('alpha', 1.0), max_iter=10000)
        elif model_type == 'elastic_net':
            model = ElasticNet(
                alpha=kwargs.get('alpha', 1.0),
                l1_ratio=kwargs.get('l1_ratio', 0.5),
                max_iter=10000
            )
        elif model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 5),
                min_samples_split=kwargs.get('min_samples_split', 5),
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 3),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Leave-one-cage-out cross-validation
        logo = LeaveOneGroupOut()
        groups = self.get_cage_groups()
        
        print(f"\nTraining {model_type} with leave-one-cage-out CV ({logo.get_n_splits(groups=groups)} folds)...")
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X_scaled, self.y, groups=groups, cv=logo,
            scoring=['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'],
            return_train_score=True,
            return_estimator=True
        )
        
        # Calculate performance metrics
        results = {
            'model_type': model_type,
            'n_folds': logo.get_n_splits(groups=groups),
            'mean_r2': cv_results['test_r2'].mean(),
            'std_r2': cv_results['test_r2'].std(),
            'mean_mae': -cv_results['test_neg_mean_absolute_error'].mean(),
            'std_mae': -cv_results['test_neg_mean_absolute_error'].std(),
            'mean_rmse': np.sqrt(-cv_results['test_neg_mean_squared_error'].mean()),
            'fold_r2': cv_results['test_r2'],
            'estimators': cv_results['estimator']
        }
        
        # Train final model on all data
        model.fit(X_scaled, self.y)
        
        # Store results
        self.models[model_type] = {
            'model': model,
            'scaler': scaler,
            'results': results
        }
        self.results[model_type] = results
        
        # Extract feature importance
        if hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            importance = None
        
        if importance is not None:
            self.feature_importance[model_type] = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        return results
    
    def permutation_test(self, model_type='elastic_net', n_permutations=1000, **kwargs):
        """
        Perform permutation test to assess statistical significance
        
        Args:
            model_type: Model to test
            n_permutations: Number of random permutations
            **kwargs: Model hyperparameters
        
        Returns:
            Dictionary with permutation test results
        """
        print(f"\nPerforming permutation test with {n_permutations} permutations...")
        
        # Train on actual data
        actual_results = self.train_model(model_type, **kwargs)
        actual_r2 = actual_results['mean_r2']
        
        # Permutation tests
        perm_r2_scores = []
        
        for i in range(n_permutations):
            if (i + 1) % 100 == 0:
                print(f"  Permutation {i+1}/{n_permutations}")
            
            # Shuffle outcome
            y_permuted = self.y.sample(frac=1, random_state=i).values
            
            # Train model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.X)
            
            if model_type == 'ridge':
                model = Ridge(alpha=kwargs.get('alpha', 1.0))
            elif model_type == 'lasso':
                model = Lasso(alpha=kwargs.get('alpha', 1.0), max_iter=10000)
            elif model_type == 'elastic_net':
                model = ElasticNet(
                    alpha=kwargs.get('alpha', 1.0),
                    l1_ratio=kwargs.get('l1_ratio', 0.5),
                    max_iter=10000
                )
            elif model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=kwargs.get('n_estimators', 100),
                    max_depth=kwargs.get('max_depth', 5),
                    min_samples_split=kwargs.get('min_samples_split', 5),
                    random_state=i
                )
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(
                    n_estimators=kwargs.get('n_estimators', 100),
                    max_depth=kwargs.get('max_depth', 3),
                    learning_rate=kwargs.get('learning_rate', 0.1),
                    random_state=i
                )
            
            # Cross-validate
            logo = LeaveOneGroupOut()
            groups = self.get_cage_groups()
            
            cv_scores = cross_val_score(
                model, X_scaled, y_permuted, groups=groups, cv=logo, scoring='r2'
            )
            
            perm_r2_scores.append(cv_scores.mean())
        
        # Calculate p-value
        p_value = (np.sum(np.array(perm_r2_scores) >= actual_r2) + 1) / (n_permutations + 1)
        
        perm_results = {
            'actual_r2': actual_r2,
            'permuted_r2_mean': np.mean(perm_r2_scores),
            'permuted_r2_std': np.std(perm_r2_scores),
            'p_value': p_value,
            'permuted_scores': perm_r2_scores
        }
        
        print(f"\nPermutation Test Results:")
        print(f"  Actual R² = {actual_r2:.3f}")
        print(f"  Permuted R² (mean ± std) = {perm_results['permuted_r2_mean']:.3f} ± {perm_results['permuted_r2_std']:.3f}")
        print(f"  p-value = {p_value:.4f}")
        
        return perm_results
    
    def bootstrap_confidence_intervals(self, model_type='elastic_net', n_bootstrap=1000, **kwargs):
        """
        Calculate bootstrap confidence intervals for R² and feature importance
        
        Args:
            model_type: Model to use
            n_bootstrap: Number of bootstrap samples
            **kwargs: Model hyperparameters
        
        Returns:
            Dictionary with confidence intervals
        """
        print(f"\nCalculating bootstrap confidence intervals ({n_bootstrap} samples)...")
        
        bootstrap_r2 = []
        bootstrap_importance = []
        
        n_samples = len(self.X)
        
        for i in range(n_bootstrap):
            if (i + 1) % 100 == 0:
                print(f"  Bootstrap {i+1}/{n_bootstrap}")
            
            # Resample with replacement (by cage to preserve clustering)
            unique_cages = self.data['cage_id'].unique()
            sampled_cages = np.random.choice(unique_cages, size=len(unique_cages), replace=True)
            
            sample_idx = []
            for cage in sampled_cages:
                cage_idx = self.data[self.data['cage_id'] == cage].index.tolist()
                sample_idx.extend(cage_idx)
            
            X_boot = self.X.iloc[sample_idx]
            y_boot = self.y.iloc[sample_idx]
            
            # Standardize
            scaler = StandardScaler()
            X_boot_scaled = scaler.fit_transform(X_boot)
            
            # Train model
            if model_type == 'ridge':
                model = Ridge(alpha=kwargs.get('alpha', 1.0))
            elif model_type == 'lasso':
                model = Lasso(alpha=kwargs.get('alpha', 1.0), max_iter=10000)
            elif model_type == 'elastic_net':
                model = ElasticNet(
                    alpha=kwargs.get('alpha', 1.0),
                    l1_ratio=kwargs.get('l1_ratio', 0.5),
                    max_iter=10000
                )
            elif model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=kwargs.get('n_estimators', 100),
                    max_depth=kwargs.get('max_depth', 5),
                    min_samples_split=kwargs.get('min_samples_split', 5),
                    random_state=i
                )
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(
                    n_estimators=kwargs.get('n_estimators', 100),
                    max_depth=kwargs.get('max_depth', 3),
                    learning_rate=kwargs.get('learning_rate', 0.1),
                    random_state=i
                )
            
            model.fit(X_boot_scaled, y_boot)
            
            # Calculate R² on out-of-bag samples
            oob_idx = [i for i in range(n_samples) if i not in sample_idx]
            if len(oob_idx) > 0:
                X_oob = self.X.iloc[oob_idx]
                y_oob = self.y.iloc[oob_idx]
                X_oob_scaled = scaler.transform(X_oob)
                
                y_pred = model.predict(X_oob_scaled)
                r2 = r2_score(y_oob, y_pred)
                bootstrap_r2.append(r2)
            
            # Store feature importance
            if hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            elif hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                importance = None
            
            if importance is not None:
                bootstrap_importance.append(importance)
        
        # Calculate confidence intervals
        bootstrap_r2 = np.array(bootstrap_r2)
        bootstrap_importance = np.array(bootstrap_importance)
        
        ci_results = {
            'r2_mean': np.mean(bootstrap_r2),
            'r2_median': np.median(bootstrap_r2),
            'r2_ci_lower': np.percentile(bootstrap_r2, 2.5),
            'r2_ci_upper': np.percentile(bootstrap_r2, 97.5),
            'feature_importance_mean': np.mean(bootstrap_importance, axis=0),
            'feature_importance_ci_lower': np.percentile(bootstrap_importance, 2.5, axis=0),
            'feature_importance_ci_upper': np.percentile(bootstrap_importance, 97.5, axis=0),
            'bootstrap_r2': bootstrap_r2,
            'bootstrap_importance': bootstrap_importance
        }
        
        print(f"\nBootstrap Results:")
        print(f"  R² = {ci_results['r2_mean']:.3f} (95% CI: [{ci_results['r2_ci_lower']:.3f}, {ci_results['r2_ci_upper']:.3f}])")
        
        return ci_results
    
    def cross_replicate_validation(self, model_type='elastic_net', **kwargs):
        """
        Train on one replicate, test on the other
        
        Returns:
            Dictionary with cross-replicate results
        """
        print("\nPerforming cross-replicate validation...")
        
        # Split by replicate
        rep1_data = self.data[self.data['replicate'] == 'replicate_1']
        rep2_data = self.data[self.data['replicate'] == 'replicate_2']
        
        results = {}
        
        # Train on Rep1, test on Rep2
        print("  Training on Replicate 1, testing on Replicate 2...")
        X_train = rep1_data[self.feature_cols].fillna(rep1_data[self.feature_cols].median())
        y_train = rep1_data[self.outcome_col]
        X_test = rep2_data[self.feature_cols].fillna(rep2_data[self.feature_cols].median())
        y_test = rep2_data[self.outcome_col]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_type == 'elastic_net':
            model = ElasticNet(
                alpha=kwargs.get('alpha', 1.0),
                l1_ratio=kwargs.get('l1_ratio', 0.5),
                max_iter=10000
            )
        # Add other model types as needed
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        results['rep1_to_rep2'] = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'correlation': np.corrcoef(y_test, y_pred)[0, 1]
        }
        
        # Train on Rep2, test on Rep1
        print("  Training on Replicate 2, testing on Replicate 1...")
        X_train = rep2_data[self.feature_cols].fillna(rep2_data[self.feature_cols].median())
        y_train = rep2_data[self.outcome_col]
        X_test = rep1_data[self.feature_cols].fillna(rep1_data[self.feature_cols].median())
        y_test = rep1_data[self.outcome_col]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        results['rep2_to_rep1'] = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'correlation': np.corrcoef(y_test, y_pred)[0, 1]
        }
        
        print(f"\n  Rep1→Rep2: R² = {results['rep1_to_rep2']['r2']:.3f}, r = {results['rep1_to_rep2']['correlation']:.3f}")
        print(f"  Rep2→Rep1: R² = {results['rep2_to_rep1']['r2']:.3f}, r = {results['rep2_to_rep1']['correlation']:.3f}")
        
        return results
    
    def plot_results(self, save_path='model_results.png'):
        """
        Create visualization of model results
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Model comparison (R² scores)
        if len(self.results) > 0:
            model_names = list(self.results.keys())
            r2_scores = [self.results[m]['mean_r2'] for m in model_names]
            r2_stds = [self.results[m]['std_r2'] for m in model_names]
            
            axes[0, 0].bar(model_names, r2_scores, yerr=r2_stds, capsize=5)
            axes[0, 0].set_ylabel('R² Score')
            axes[0, 0].set_title('Model Comparison (Leave-One-Cage-Out CV)')
            axes[0, 0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Feature importance (top 15)
        if len(self.feature_importance) > 0:
            # Use the best model
            best_model = max(self.results.keys(), key=lambda x: self.results[x]['mean_r2'])
            top_features = self.feature_importance[best_model].head(15)
            
            axes[0, 1].barh(range(len(top_features)), top_features['importance'])
            axes[0, 1].set_yticks(range(len(top_features)))
            axes[0, 1].set_yticklabels(top_features['feature'], fontsize=8)
            axes[0, 1].set_xlabel('Importance')
            axes[0, 1].set_title(f'Top 15 Features ({best_model})')
            axes[0, 1].invert_yaxis()
        
        # Plot 3: Predicted vs Actual
        if len(self.models) > 0:
            best_model = max(self.results.keys(), key=lambda x: self.results[x]['mean_r2'])
            model_obj = self.models[best_model]['model']
            scaler = self.models[best_model]['scaler']
            
            X_scaled = scaler.transform(self.X)
            y_pred = model_obj.predict(X_scaled)
            
            axes[1, 0].scatter(self.y, y_pred, alpha=0.6)
            axes[1, 0].plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], 
                           'k--', linewidth=1)
            axes[1, 0].set_xlabel('Actual Response')
            axes[1, 0].set_ylabel('Predicted Response')
            axes[1, 0].set_title(f'Predicted vs Actual ({best_model})')
            
            # Add R² to plot
            r2 = r2_score(self.y, y_pred)
            axes[1, 0].text(0.05, 0.95, f'R² = {r2:.3f}', 
                          transform=axes[1, 0].transAxes, verticalalignment='top')
        
        # Plot 4: Feature correlation heatmap (top features)
        if len(self.feature_importance) > 0:
            best_model = max(self.results.keys(), key=lambda x: self.results[x]['mean_r2'])
            top_features = self.feature_importance[best_model].head(10)['feature'].tolist()
            
            corr_matrix = self.X[top_features].corr()
            
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=axes[1, 1], cbar_kws={'label': 'Correlation'})
            axes[1, 1].set_title('Feature Correlations (Top 10)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nResults plot saved to {save_path}")
        
        return fig
    
    def generate_report(self, output_file='model_report.txt'):
        """
        Generate a text report of all results
        """
        with open(output_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("OPIOID SENSITIVITY PREDICTION MODEL REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Dataset: {len(self.data)} animals\n")
            f.write(f"Features: {len(self.feature_cols)}\n")
            f.write(f"Outcome: {self.outcome_col}\n\n")
            
            f.write("MODEL PERFORMANCE\n")
            f.write("-"*60 + "\n")
            
            for model_name, results in self.results.items():
                f.write(f"\n{model_name.upper()}:\n")
                f.write(f"  R² = {results['mean_r2']:.3f} ± {results['std_r2']:.3f}\n")
                f.write(f"  MAE = {results['mean_mae']:.3f} ± {results['std_mae']:.3f}\n")
                f.write(f"  RMSE = {results['mean_rmse']:.3f}\n")
                f.write(f"  Cross-validation folds: {results['n_folds']}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("FEATURE IMPORTANCE (Top 20)\n")
            f.write("="*60 + "\n")
            
            if len(self.feature_importance) > 0:
                best_model = max(self.results.keys(), key=lambda x: self.results[x]['mean_r2'])
                top_features = self.feature_importance[best_model].head(20)
                
                f.write(f"\nModel: {best_model}\n\n")
                for idx, row in top_features.iterrows():
                    f.write(f"{row['feature']:50s} {row['importance']:.4f}\n")
        
        print(f"\nReport saved to {output_file}")


def main():
    """
    Example usage of the modeling pipeline
    """
    # Load features
    print("Loading baseline features...")
    features_df = pd.read_csv('baseline_features.csv')
    
    # Initialize predictor
    predictor = OpioidSensitivityPredictor(features_df)
    
    # Load outcome data (you'll need to create this file)
    # outcome_df should have columns: ['animal_id', 'morphine_response']
    # predictor.load_outcome_data('morphine_response.csv')
    
    # For demo purposes, create synthetic outcome
    print("\nCreating synthetic outcome for demonstration...")
    features_df['morphine_response'] = np.random.randn(len(features_df)) * 10 + 50
    predictor.data = features_df
    
    # Prepare data
    X, y = predictor.prepare_data()
    
    # Calculate trade-off residual
    predictor.calculate_freq_duration_tradeoff()
    
    # Train models
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    # Ridge regression
    ridge_results = predictor.train_model('ridge', alpha=1.0)
    
    # Lasso regression
    lasso_results = predictor.train_model('lasso', alpha=0.1)
    
    # Elastic Net (recommended)
    elastic_results = predictor.train_model('elastic_net', alpha=0.5, l1_ratio=0.5)
    
    # Random Forest
    rf_results = predictor.train_model('random_forest', n_estimators=100, max_depth=5)
    
    # Permutation test (using best model)
    best_model = max(predictor.results.keys(), key=lambda x: predictor.results[x]['mean_r2'])
    print(f"\nRunning permutation test for {best_model}...")
    # perm_results = predictor.permutation_test(best_model, n_permutations=100)
    
    # Bootstrap confidence intervals
    # boot_results = predictor.bootstrap_confidence_intervals(best_model, n_bootstrap=100)
    
    # Cross-replicate validation
    # xrep_results = predictor.cross_replicate_validation(best_model)
    
    # Generate visualizations
    predictor.plot_results('model_results.png')
    
    # Generate report
    predictor.generate_report('model_report.txt')
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nBest model: {best_model}")
    print(f"R² = {predictor.results[best_model]['mean_r2']:.3f}")
    print(f"MAE = {predictor.results[best_model]['mean_mae']:.3f}")


if __name__ == '__main__':
    main()
