"""
Feature Extraction Using DuckDB for S3 Access
You will run this in your environment with network access
"""

import duckdb
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractor:
    """Extract baseline behavioral features using DuckDB for S3 access"""
    
    def __init__(self, s3_base='s3://jax-envision-public-data/study_1001/2025v3.3/tabular'):
        self.s3_base = s3_base
        
        # Initialize DuckDB connection
        self.conn = duckdb.connect()
        
        # Install and load httpfs for S3 access
        self.conn.execute("INSTALL httpfs;")
        self.conn.execute("LOAD httpfs;")
        self.conn.execute("SET s3_region='us-east-1';")
        
        # Baseline windows (pre-drug periods)
        self.baseline_windows = {
            'replicate_1': {
                'start': '2025-01-10',
                'end': '2025-01-13',
                'cages': list(range(4917, 4926))
            },
            'replicate_2': {
                'start': '2025-01-25',
                'end': '2025-01-27',
                'cages': list(range(4926, 4935))
            }
        }
        
        # Cage treatment mapping
        self.cage_treatments = {
            4917: {'dose': 5}, 4918: {'dose': 0}, 4919: {'dose': 25},
            4920: {'dose': 25}, 4921: {'dose': 5}, 4922: {'dose': 0},
            4923: {'dose': 0}, 4924: {'dose': 25}, 4925: {'dose': 5},
            4926: {'dose': 25}, 4927: {'dose': 5}, 4928: {'dose': 0},
            4929: {'dose': 0}, 4930: {'dose': 25}, 4931: {'dose': 5},
            4932: {'dose': 5}, 4933: {'dose': 25}, 4934: {'dose': 0},
        }
    
    def query(self, sql):
        """Execute DuckDB query and return DataFrame"""
        try:
            return self.conn.execute(sql).fetchdf()
        except Exception as e:
            print(f"    Query error: {e}")
            return pd.DataFrame()
    
    def extract_locomotion_features(self, cage_id, dates):
        """Extract locomotion and activity state features"""
        
        # Build date filter
        date_filter = " OR ".join([f"date = '{d}'" for d in dates])
        
        # Query 1: Activity state proportions
        query = f"""
        SELECT 
            name,
            AVG(value) as mean_proportion,
            SUM(value) as total_seconds
        FROM read_parquet('{self.s3_base}/cage_id={cage_id}/date=*/animal_activity_db.parquet')
        WHERE ({date_filter})
            AND name IN ('animal_bouts.locomotion', 'animal_bouts.active', 
                         'animal_bouts.inactive', 'animal_bouts.climbing')
            AND resolution = 1
        GROUP BY name
        """
        
        df_states = self.query(query)
        
        # Query 2: Distance traveled metrics
        query = f"""
        SELECT 
            AVG(value) as mean_distance,
            STDDEV(value) as std_distance,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY value) as peak_distance,
            SUM(value * 60) as total_distance,
            COUNT(*) as sample_count
        FROM read_parquet('{self.s3_base}/cage_id={cage_id}/date=*/animal_aggs_short_id.parquet')
        WHERE ({date_filter})
            AND name = 'animal.distance_travelled'
            AND resolution = 60
        """
        
        df_dist = self.query(query)
        
        features = {}
        
        # Process activity states
        for _, row in df_states.iterrows():
            state = row['name'].replace('animal_bouts.', '')
            features[f'{state}_proportion'] = row['mean_proportion']
            features[f'{state}_total_seconds'] = row['total_seconds']
        
        # Process distance metrics
        if len(df_dist) > 0:
            features['mean_distance_cm_per_s'] = df_dist['mean_distance'].iloc[0]
            features['distance_std'] = df_dist['std_distance'].iloc[0]
            features['peak_distance_95pct'] = df_dist['peak_distance'].iloc[0]
            features['total_distance_cm'] = df_dist['total_distance'].iloc[0]
            
            if features['mean_distance_cm_per_s'] > 0:
                features['distance_cv'] = features['distance_std'] / features['mean_distance_cm_per_s']
        
        return features
    
    def extract_bout_features(self, cage_id, dates):
        """Extract bout structure features"""
        
        date_filter = " OR ".join([f"date = '{d}'" for d in dates])
        
        query = f"""
        SELECT 
            state_name,
            COUNT(*) as bout_count,
            AVG(bout_length_seconds) as mean_duration,
            STDDEV(bout_length_seconds) as std_duration,
            SUM(bout_length_seconds) as total_duration,
            MIN(bout_length_seconds) as min_duration,
            MAX(bout_length_seconds) as max_duration
        FROM read_parquet('{self.s3_base}/cage_id={cage_id}/date=*/animal_bouts.parquet')
        WHERE ({date_filter})
            AND state_name IN ('animal_bouts.locomotion', 'animal_bouts.inactive', 
                               'animal_bouts.active', 'animal_bouts.climbing')
        GROUP BY state_name
        """
        
        df = self.query(query)
        
        features = {}
        total_hours = len(dates) * 24
        
        for _, row in df.iterrows():
            state = row['state_name'].replace('animal_bouts.', '')
            
            features[f'{state}_bout_count'] = row['bout_count']
            features[f'{state}_bout_frequency_per_hour'] = row['bout_count'] / total_hours
            features[f'{state}_bout_mean_duration'] = row['mean_duration']
            features[f'{state}_bout_std_duration'] = row['std_duration']
            
            if row['mean_duration'] > 0:
                features[f'{state}_bout_duration_cv'] = row['std_duration'] / row['mean_duration']
            
            # Store raw values for frequency-duration trade-off (calculated later)
            if state == 'locomotion':
                features['locomotion_bout_freq_raw'] = row['bout_count'] / total_hours
                features['locomotion_bout_duration_raw'] = row['mean_duration']
            
            # Rest fragmentation
            if state == 'inactive':
                total_rest_hours = row['total_duration'] / 3600
                if total_rest_hours > 0:
                    features['rest_fragmentation_index'] = row['bout_count'] / total_rest_hours
                features['rest_total_hours'] = total_rest_hours
        
        return features
    
    def extract_circadian_features(self, cage_id, dates):
        """Extract circadian rhythm features"""
        
        date_filter = " OR ".join([f"date = '{d}'" for d in dates])
        
        query = f"""
        SELECT 
            HOUR(CAST(time AS TIMESTAMP)) as hour,
            AVG(value) as mean_activity,
            COUNT(*) as sample_count
        FROM read_parquet('{self.s3_base}/cage_id={cage_id}/date=*/animal_aggs_short_id.parquet')
        WHERE ({date_filter})
            AND name = 'animal.distance_travelled'
            AND resolution = 60
        GROUP BY HOUR(CAST(time AS TIMESTAMP))
        ORDER BY hour
        """
        
        df = self.query(query)
        
        features = {}
        
        if len(df) >= 20:  # Need most hours represented
            # Create full 24-hour profile
            full_hours = pd.DataFrame({'hour': range(24)})
            df_merged = full_hours.merge(df, on='hour', how='left')
            df_merged['mean_activity'] = df_merged['mean_activity'].fillna(df_merged['mean_activity'].mean())
            
            hourly_activity = df_merged['mean_activity'].values
            
            # Cosinor fit: activity = amplitude * cos(2π(hour - acrophase)/24) + mesor
            try:
                def cosinor(hour, amplitude, acrophase, mesor):
                    return amplitude * np.cos(2 * np.pi * (hour - acrophase) / 24) + mesor
                
                # Initial guesses
                amplitude_guess = (hourly_activity.max() - hourly_activity.min()) / 2
                acrophase_guess = float(hourly_activity.argmax())
                mesor_guess = hourly_activity.mean()
                
                params, _ = curve_fit(
                    cosinor,
                    range(24),
                    hourly_activity,
                    p0=[amplitude_guess, acrophase_guess, mesor_guess],
                    maxfev=10000
                )
                
                amplitude, acrophase, mesor = params
                
                # Calculate R² (goodness of fit)
                y_pred = cosinor(np.array(range(24)), *params)
                ss_res = np.sum((hourly_activity - y_pred) ** 2)
                ss_tot = np.sum((hourly_activity - hourly_activity.mean()) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                features['circadian_amplitude'] = amplitude
                features['circadian_acrophase'] = acrophase
                features['circadian_mesor'] = mesor
                features['circadian_r_squared'] = r_squared
                
            except Exception as e:
                print(f"      Cosinor fit warning: {e}")
            
            # Light/dark ratio (lights on 6am-6pm, off 6pm-6am)
            light_activity = hourly_activity[6:18].mean()
            dark_activity = np.concatenate([hourly_activity[18:24], hourly_activity[0:6]]).mean()
            
            features['light_phase_activity'] = light_activity
            features['dark_phase_activity'] = dark_activity
            
            if dark_activity > 0:
                features['light_dark_ratio'] = light_activity / dark_activity
        
        return features
    
    def extract_temporal_patterns(self, cage_id, dates):
        """Extract inter-bout intervals and temporal patterns"""
        
        date_filter = " OR ".join([f"date = '{d}'" for d in dates])
        
        # Get locomotion bout start times
        query = f"""
        WITH sorted_bouts AS (
            SELECT 
                start_time,
                LAG(start_time) OVER (ORDER BY start_time) as prev_start_time
            FROM read_parquet('{self.s3_base}/cage_id={cage_id}/date=*/animal_bouts.parquet')
            WHERE ({date_filter})
                AND state_name = 'animal_bouts.locomotion'
            ORDER BY start_time
        )
        SELECT 
            EPOCH(start_time - prev_start_time) as interval_seconds
        FROM sorted_bouts
        WHERE prev_start_time IS NOT NULL
        """
        
        df = self.query(query)
        
        features = {}
        
        if len(df) > 1:
            intervals = df['interval_seconds'].values
            
            features['inter_bout_interval_median'] = np.median(intervals)
            features['inter_bout_interval_mean'] = np.mean(intervals)
            features['inter_bout_interval_std'] = np.std(intervals)
            
            if np.mean(intervals) > 0:
                features['inter_bout_interval_cv'] = np.std(intervals) / np.mean(intervals)
            
            # Burstiness index: (σ - μ) / (σ + μ)
            mean_int = np.mean(intervals)
            std_int = np.std(intervals)
            if (std_int + mean_int) > 0:
                features['burstiness_index'] = (std_int - mean_int) / (std_int + mean_int)
        
        return features
    
    def extract_state_transitions(self, cage_id, dates):
        """Extract state transition rate"""
        
        date_filter = " OR ".join([f"date = '{d}'" for d in dates])
        
        query = f"""
        WITH all_bouts AS (
            SELECT 
                start_time,
                state_name,
                LAG(state_name) OVER (ORDER BY start_time) as prev_state
            FROM read_parquet('{self.s3_base}/cage_id={cage_id}/date=*/animal_bouts.parquet')
            WHERE ({date_filter})
            ORDER BY start_time
        )
        SELECT 
            COUNT(*) as transition_count
        FROM all_bouts
        WHERE state_name != prev_state AND prev_state IS NOT NULL
        """
        
        df = self.query(query)
        
        features = {}
        
        if len(df) > 0:
            total_hours = len(dates) * 24
            features['state_transition_rate_per_hour'] = df['transition_count'].iloc[0] / total_hours
        
        return features
    
    def extract_social_features(self, cage_id, dates):
        """Extract social distance features"""
        
        date_filter = " OR ".join([f"date = '{d}'" for d in dates])
        
        query = f"""
        SELECT 
            name,
            AVG(value) as mean_value,
            STDDEV(value) as std_value
        FROM read_parquet('{self.s3_base}/cage_id={cage_id}/date=*/animal_tsdb_mvp.parquet')
        WHERE ({date_filter})
            AND (name LIKE 'distance_mean%social.animal.cm' 
                 OR name LIKE 'distance_nearest%social.animal.cm')
        GROUP BY name
        """
        
        df = self.query(query)
        
        features = {}
        
        for _, row in df.iterrows():
            if 'mean_all' in row['name']:
                features['mean_distance_to_cagemates_cm'] = row['mean_value']
            elif 'nearest_all' in row['name']:
                features['nearest_neighbor_distance_cm'] = row['mean_value']
        
        return features
    
    def extract_respiration_features(self, cage_id, dates):
        """Extract respiration features"""
        
        date_filter = " OR ".join([f"date = '{d}'" for d in dates])
        
        query = f"""
        SELECT 
            AVG(value) as mean_resp,
            STDDEV(value) as std_resp,
            COUNT(*) as sample_count
        FROM read_parquet('{self.s3_base}/cage_id={cage_id}/date=*/animal_respiration.parquet')
        WHERE ({date_filter})
            AND name = 'animal.respiration_rate_lucas_kanade_psd'
        """
        
        df = self.query(query)
        
        features = {}
        
        if len(df) > 0 and pd.notna(df['mean_resp'].iloc[0]):
            features['mean_respiration_rate_bpm'] = df['mean_resp'].iloc[0]
            features['respiration_rate_std'] = df['std_resp'].iloc[0]
            
            if features['mean_respiration_rate_bpm'] > 0:
                features['respiration_rate_cv'] = features['respiration_rate_std'] / features['mean_respiration_rate_bpm']
        
        return features
    
    def extract_drinking_features(self, cage_id, dates):
        """Extract drinking behavior features"""
        
        date_filter = " OR ".join([f"date = '{d}'" for d in dates])
        
        query = f"""
        SELECT 
            SUM(value) as total_drinking_seconds,
            AVG(value) as mean_drinking_fraction
        FROM read_parquet('{self.s3_base}/cage_id={cage_id}/date=*/animal_drinking.parquet')
        WHERE ({date_filter})
            AND name = 'animal_bouts.drinking'
            AND resolution = 1
        """
        
        df = self.query(query)
        
        features = {}
        
        if len(df) > 0 and pd.notna(df['total_drinking_seconds'].iloc[0]):
            features['total_drinking_seconds'] = df['total_drinking_seconds'].iloc[0]
        
        return features
    
    def extract_all_features_for_cage(self, cage_id, replicate):
        """Extract all features for one cage"""
        
        baseline_info = self.baseline_windows[replicate]
        dates = pd.date_range(
            start=baseline_info['start'],
            end=baseline_info['end'],
            freq='D'
        ).strftime('%Y-%m-%d').tolist()
        
        print(f"\n  Cage {cage_id}: {self.cage_treatments[cage_id]['dose']} mg/kg")
        
        features = {
            'cage_id': cage_id,
            'replicate': replicate,
            'dose_mg_kg': self.cage_treatments[cage_id]['dose'],
            'baseline_start': baseline_info['start'],
            'baseline_end': baseline_info['end'],
            'baseline_days': len(dates)
        }
        
        print(f"    Extracting locomotion...", end='', flush=True)
        locomotion = self.extract_locomotion_features(cage_id, dates)
        features.update(locomotion)
        print(f" {len(locomotion)} features")
        
        print(f"    Extracting bouts...", end='', flush=True)
        bouts = self.extract_bout_features(cage_id, dates)
        features.update(bouts)
        print(f" {len(bouts)} features")
        
        print(f"    Extracting circadian...", end='', flush=True)
        circadian = self.extract_circadian_features(cage_id, dates)
        features.update(circadian)
        print(f" {len(circadian)} features")
        
        print(f"    Extracting temporal patterns...", end='', flush=True)
        temporal = self.extract_temporal_patterns(cage_id, dates)
        features.update(temporal)
        print(f" {len(temporal)} features")
        
        print(f"    Extracting state transitions...", end='', flush=True)
        transitions = self.extract_state_transitions(cage_id, dates)
        features.update(transitions)
        print(f" {len(transitions)} features")
        
        print(f"    Extracting social...", end='', flush=True)
        social = self.extract_social_features(cage_id, dates)
        features.update(social)
        print(f" {len(social)} features")
        
        print(f"    Extracting respiration...", end='', flush=True)
        respiration = self.extract_respiration_features(cage_id, dates)
        features.update(respiration)
        print(f" {len(respiration)} features")
        
        print(f"    Extracting drinking...", end='', flush=True)
        drinking = self.extract_drinking_features(cage_id, dates)
        features.update(drinking)
        print(f" {len(drinking)} features")
        
        total_features = len([k for k in features.keys() if k not in ['cage_id', 'replicate', 'dose_mg_kg', 'baseline_start', 'baseline_end', 'baseline_days']])
        print(f"    ✓ Total: {total_features} features extracted")
        
        return features
    
    def extract_all_features(self, morphine_only=True):
        """Extract features for all cages"""
        
        all_features = []
        
        for replicate_name, replicate_info in self.baseline_windows.items():
            print(f"\n{'='*70}")
            print(f" {replicate_name.upper().replace('_', ' ')}")
            print(f"{'='*70}")
            
            for cage_id in replicate_info['cages']:
                dose = self.cage_treatments[cage_id]['dose']
                
                if morphine_only and dose == 0:
                    print(f"\n  Cage {cage_id}: SKIPPED (saline control, 0 mg/kg)")
                    continue
                
                try:
                    features = self.extract_all_features_for_cage(cage_id, replicate_name)
                    all_features.append(features)
                except Exception as e:
                    print(f"    ✗ ERROR extracting features: {e}")
                    import traceback
                    traceback.print_exc()
        
        df = pd.DataFrame(all_features)
        return df


def main():
    """Main execution function"""
    
    print("="*70)
    print(" BASELINE FEATURE EXTRACTION - OPIOID SENSITIVITY PREDICTION")
    print("="*70)
    print("\nInitializing DuckDB connection to S3...")
    
    try:
        extractor = FeatureExtractor()
        print("✓ DuckDB initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize DuckDB: {e}")
        print("\nMake sure DuckDB is installed: pip install duckdb")
        return None
    
    print("\nExtracting baseline features for morphine-treated mice...")
    print("(Excluding saline controls)")
    
    features_df = extractor.extract_all_features(morphine_only=True)
    
    # Save results
    output_file = 'baseline_features.csv'
    features_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print(" EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"\n✓ Saved to: {output_file}")
    print(f"  - Cages: {len(features_df)}")
    print(f"  - Total columns: {len(features_df.columns)}")
    
    # Count feature columns (exclude metadata)
    metadata_cols = ['cage_id', 'replicate', 'dose_mg_kg', 'baseline_start', 'baseline_end', 'baseline_days']
    feature_cols = [c for c in features_df.columns if c not in metadata_cols]
    print(f"  - Feature columns: {len(feature_cols)}")
    
    print(f"\n Summary by dose group:")
    print(features_df.groupby('dose_mg_kg').agg({
        'cage_id': 'count',
        'replicate': lambda x: x.value_counts().to_dict()
    }).rename(columns={'cage_id': 'count'}))
    
    print(f"\n Sample feature values (first cage):")
    sample_features = {k: v for k, v in features_df.iloc[0].items() if k in feature_cols[:10]}
    for feat, val in sample_features.items():
        print(f"  {feat}: {val:.3f}" if isinstance(val, (int, float)) else f"  {feat}: {val}")
    
    return features_df


if __name__ == '__main__':
    features_df = main()
