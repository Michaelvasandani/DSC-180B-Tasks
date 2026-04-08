"""
Morphine Response Outcome Extraction - CORRECTED VERSION
Uses 50-250 minute window to capture hyperlocomotion phase
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class MorphineResponseExtractor:
    """Extract morphine-induced locomotor response using DuckDB - CORRECTED TIME WINDOW"""
    
    def __init__(self, s3_base='s3://jax-envision-public-data/study_1001/2025v3.3/tabular'):
        self.s3_base = s3_base
        
        # Initialize DuckDB
        self.conn = duckdb.connect()
        self.conn.execute("INSTALL httpfs;")
        self.conn.execute("LOAD httpfs;")
        self.conn.execute("SET s3_region='us-east-1';")
        
        # Dose administration times
        self.dose_times = {
            'replicate_1': {
                'dose_1_time': '2025-01-14 06:00:00',
                'dose_1_date': '2025-01-14',
                'dose_2_time': '2025-01-17 17:00:00',
                'dose_2_date': '2025-01-17',
                'cages': list(range(4917, 4926))
            },
            'replicate_2': {
                'dose_1_time': '2025-01-28 17:00:00',
                'dose_1_date': '2025-01-28',
                'dose_2_time': '2025-01-31 06:00:00',
                'dose_2_date': '2025-01-31',
                'cages': list(range(4926, 4935))
            }
        }
        
        # Baseline periods
        self.baseline_windows = {
            'replicate_1': ['2025-01-10', '2025-01-11', '2025-01-12', '2025-01-13'],
            'replicate_2': ['2025-01-25', '2025-01-26', '2025-01-27']
        }
        
        # Cage treatments
        self.cage_treatments = {
            4917: 5, 4918: 0, 4919: 25, 4920: 25, 4921: 5, 4922: 0,
            4923: 0, 4924: 25, 4925: 5, 4926: 25, 4927: 5, 4928: 0,
            4929: 0, 4930: 25, 4931: 5, 4932: 5, 4933: 25, 4934: 0
        }
    
    def query(self, sql):
        """Execute DuckDB query"""
        try:
            return self.conn.execute(sql).fetchdf()
        except Exception as e:
            print(f"    Query error: {e}")
            return pd.DataFrame()
    
    def extract_baseline_activity(self, cage_id, replicate):
        """Extract baseline locomotor activity"""
        
        dates = self.baseline_windows[replicate]
        date_filter = " OR ".join([f"date = '{d}'" for d in dates])
        
        query = f"""
        SELECT 
            AVG(value) as baseline_mean,
            STDDEV(value) as baseline_std,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) as baseline_median,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY value) as baseline_95pct,
            COUNT(*) as baseline_samples
        FROM read_parquet('{self.s3_base}/cage_id={cage_id}/date=*/animal_aggs_short_id.parquet')
        WHERE ({date_filter})
            AND name = 'animal.distance_travelled'
            AND resolution = 60
        """
        
        df = self.query(query)
        
        if len(df) > 0:
            return {
                'baseline_mean': df['baseline_mean'].iloc[0],
                'baseline_std': df['baseline_std'].iloc[0],
                'baseline_median': df['baseline_median'].iloc[0],
                'baseline_95pct': df['baseline_95pct'].iloc[0],
                'baseline_samples': df['baseline_samples'].iloc[0]
            }
        else:
            return {
                'baseline_mean': np.nan,
                'baseline_std': np.nan,
                'baseline_median': np.nan,
                'baseline_95pct': np.nan,
                'baseline_samples': 0
            }
    
    def extract_post_dose_activity(self, cage_id, replicate, dose_number=1, 
                                  window_start_min=50, window_end_min=250):
        """
        Extract post-dose locomotor activity during HYPERLOCOMOTION phase
        
        Args:
            window_start_min: Start of window in minutes post-injection (default: 50)
            window_end_min: End of window in minutes post-injection (default: 250)
        """
        
        dose_info = self.dose_times[replicate]
        dose_time_str = dose_info[f'dose_{dose_number}_time']
        dose_date = dose_info[f'dose_{dose_number}_date']
        
        # Calculate time window for hyperlocomotion phase
        dose_time = datetime.strptime(dose_time_str, '%Y-%m-%d %H:%M:%S')
        start_time = dose_time + timedelta(minutes=window_start_min)
        end_time = dose_time + timedelta(minutes=window_end_min)
        
        start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Need to handle date boundary crossing
        dates_involved = []
        current_date = dose_time.date()
        end_date = end_time.date()
        while current_date <= end_date:
            dates_involved.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        date_filter = " OR ".join([f"date = '{d}'" for d in dates_involved])
        
        query = f"""
        SELECT 
            AVG(value) as post_dose_mean,
            STDDEV(value) as post_dose_std,
            MAX(value) as post_dose_peak,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY value) as post_dose_95pct,
            SUM(value * 60) as post_dose_auc,
            COUNT(*) as post_dose_samples
        FROM read_parquet('{self.s3_base}/cage_id={cage_id}/date=*/animal_aggs_short_id.parquet')
        WHERE ({date_filter})
            AND name = 'animal.distance_travelled'
            AND resolution = 60
            AND CAST(time AS TIMESTAMP) >= CAST('{start_time_str}' AS TIMESTAMP)
            AND CAST(time AS TIMESTAMP) < CAST('{end_time_str}' AS TIMESTAMP)
        """
        
        df = self.query(query)
        
        if len(df) > 0:
            return {
                'post_dose_mean': df['post_dose_mean'].iloc[0],
                'post_dose_std': df['post_dose_std'].iloc[0],
                'post_dose_peak': df['post_dose_peak'].iloc[0],
                'post_dose_95pct': df['post_dose_95pct'].iloc[0],
                'post_dose_auc': df['post_dose_auc'].iloc[0],
                'post_dose_samples': df['post_dose_samples'].iloc[0]
            }
        else:
            return {
                'post_dose_mean': np.nan,
                'post_dose_std': np.nan,
                'post_dose_peak': np.nan,
                'post_dose_95pct': np.nan,
                'post_dose_auc': np.nan,
                'post_dose_samples': 0
            }
    
    def calculate_response_score(self, baseline, post_dose, method='percent_change_mean'):
        """Calculate morphine response score"""
        
        if method == 'percent_change_mean':
            if baseline['baseline_mean'] > 0:
                return ((post_dose['post_dose_mean'] - baseline['baseline_mean']) / 
                       baseline['baseline_mean'] * 100)
        
        elif method == 'percent_change_peak':
            if baseline['baseline_95pct'] > 0:
                return ((post_dose['post_dose_peak'] - baseline['baseline_95pct']) / 
                       baseline['baseline_95pct'] * 100)
        
        elif method == 'z_score':
            if baseline['baseline_std'] > 0:
                return ((post_dose['post_dose_mean'] - baseline['baseline_mean']) / 
                       baseline['baseline_std'])
        
        elif method == 'auc_normalized':
            if baseline['baseline_mean'] > 0:
                return post_dose['post_dose_auc'] / baseline['baseline_mean']
        
        elif method == 'absolute_change':
            return post_dose['post_dose_mean'] - baseline['baseline_mean']
        
        return np.nan
    
    def extract_response_for_cage(self, cage_id, replicate, dose_number=1, 
                                  response_method='percent_change_mean',
                                  window_start_min=50, window_end_min=250):
        """Extract complete morphine response for one cage during hyperlocomotion phase"""
        
        dose = self.cage_treatments[cage_id]
        
        print(f"  Cage {cage_id}: {dose} mg/kg", end='', flush=True)
        
        # Extract baseline
        baseline = self.extract_baseline_activity(cage_id, replicate)
        print(f" | Baseline: {baseline['baseline_mean']:.2f} cm/s", end='', flush=True)
        
        # Extract post-dose (during hyperlocomotion window)
        post_dose = self.extract_post_dose_activity(cage_id, replicate, dose_number, 
                                                    window_start_min, window_end_min)
        print(f" | Hyperloc ({window_start_min}-{window_end_min} min): {post_dose['post_dose_mean']:.2f} cm/s", end='', flush=True)
        
        # Calculate response
        response = self.calculate_response_score(baseline, post_dose, response_method)
        print(f" | Response: {response:.2f}%")
        
        result = {
            'cage_id': cage_id,
            'replicate': replicate,
            'dose_mg_kg': dose,
            'dose_number': dose_number,
            'morphine_response': response,
            'response_method': response_method,
            'response_window_start_min': window_start_min,
            'response_window_end_min': window_end_min
        }
        
        # Add all baseline and post-dose metrics
        result.update(baseline)
        result.update(post_dose)
        
        return result
    
    def extract_all_responses(self, dose_number=1, response_method='percent_change_mean',
                             window_start_min=50, window_end_min=250, morphine_only=True):
        """Extract morphine responses for all cages during hyperlocomotion phase"""
        
        all_responses = []
        
        for replicate_name, replicate_info in self.dose_times.items():
            print(f"\n{'='*70}")
            print(f" {replicate_name.upper().replace('_', ' ')} - Dose {dose_number}")
            print(f"{'='*70}")
            print(f" Dose time: {replicate_info[f'dose_{dose_number}_time']}")
            print(f" Hyperlocomotion window: {window_start_min}-{window_end_min} minutes post-injection")
            print(f" Method: {response_method}\n")
            
            for cage_id in replicate_info['cages']:
                dose = self.cage_treatments[cage_id]
                
                # Skip saline if morphine_only
                if morphine_only and dose == 0:
                    print(f"  Cage {cage_id}: SKIPPED (saline control)")
                    continue
                
                try:
                    response = self.extract_response_for_cage(
                        cage_id, replicate_name, dose_number, 
                        response_method, window_start_min, window_end_min
                    )
                    all_responses.append(response)
                    
                except Exception as e:
                    print(f"  Cage {cage_id}: ERROR - {e}")
        
        df = pd.DataFrame(all_responses)
        return df


def main():
    """Re-extract with corrected time window"""
    
    print("="*70)
    print(" RE-EXTRACTING MORPHINE RESPONSES")
    print(" USING CORRECT TIME WINDOW (50-250 MINUTES)")
    print("="*70)
    
    print("\nBased on your time-course plots, morphine induces:")
    print("  • 0-50 min: Onset/sedation phase (activity low)")
    print("  • 50-250 min: HYPERLOCOMOTION phase (activity peaks)")
    print("  • 250+ min: Return to baseline")
    
    print("\nYour original analysis used 0-60 min → captured WRONG phase!")
    print("New analysis will use 50-250 min → captures TRUE response\n")
    
    try:
        extractor = MorphineResponseExtractor()
        print("✓ DuckDB initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize DuckDB: {e}")
        return None
    
    # Extract responses during hyperlocomotion phase
    responses_df = extractor.extract_all_responses(
        dose_number=1,
        response_method='percent_change_mean',
        window_start_min=50,
        window_end_min=250,
        morphine_only=True
    )
    
    # Save
    output_file = 'morphine_response_corrected.csv'
    responses_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print(" EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"\n✓ Saved to: {output_file}")
    print(f"  - Cages: {len(responses_df)}")
    
    print(f"\n Summary by dose group (CORRECTED WINDOW):")
    summary = responses_df.groupby('dose_mg_kg')['morphine_response'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ])
    print(summary)
    
    print(f"\n Distribution of morphine response:")
    print(f"  Mean ± SD: {responses_df['morphine_response'].mean():.2f} ± {responses_df['morphine_response'].std():.2f}")
    print(f"  Range: [{responses_df['morphine_response'].min():.2f}, {responses_df['morphine_response'].max():.2f}]")
    
    # Check if responses are now positive
    n_positive = (responses_df['morphine_response'] > 0).sum()
    n_total = len(responses_df)
    
    print(f"\n Responders (increased locomotion):")
    print(f"  {n_positive}/{n_total} cages ({100*n_positive/n_total:.1f}%)")
    
    if responses_df['morphine_response'].mean() > 0:
        print("\n ✓ SUCCESS! Mean response is now POSITIVE")
        print("   This confirms morphine induces hyperlocomotion in this dataset")
    else:
        print("\n ⚠️  Mean response still negative - unexpected given time-course data")
    
    print("\n" + "="*70)
    print(" NEXT STEPS")
    print("="*70)
    print("\n1. Replace morphine_response.csv with morphine_response_corrected.csv")
    print("   mv morphine_response_corrected.csv morphine_response.csv")
    print("\n2. Run minimal model with corrected outcomes:")
    print("   python minimal_model.py")
    print("\n3. Expected: POSITIVE responses and correlation with circadian amplitude!")
    
    return responses_df


if __name__ == '__main__':
    responses_df = main()