"""
FINAL CORRECTED Morphine Response Extraction
Uses: animal_bouts.locomotion (proportion) at 300-500 min window
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class MorphineResponseExtractorFinal:
    """Extract morphine response using CORRECT metric and time window"""
    
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
                'cages': list(range(4917, 4926))
            },
            'replicate_2': {
                'dose_1_time': '2025-01-28 17:00:00',
                'dose_1_date': '2025-01-28',
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
    
    def extract_baseline_locomotion(self, cage_id, replicate):
        """Extract baseline locomotion PROPORTION (not distance!)"""
        
        dates = self.baseline_windows[replicate]
        date_filter = " OR ".join([f"date = '{d}'" for d in dates])
        
        # Use animal_bouts.locomotion - proportion of time locomoting
        query = f"""
        SELECT 
            AVG(value) as baseline_mean,
            STDDEV(value) as baseline_std,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) as baseline_median,
            COUNT(*) as baseline_samples
        FROM read_parquet('{self.s3_base}/cage_id={cage_id}/date=*/animal_activity_db.parquet')
        WHERE ({date_filter})
            AND name = 'animal_bouts.locomotion'
            AND resolution = 1
        """
        
        df = self.query(query)
        
        if len(df) > 0:
            return {
                'baseline_mean': df['baseline_mean'].iloc[0],
                'baseline_std': df['baseline_std'].iloc[0],
                'baseline_median': df['baseline_median'].iloc[0],
                'baseline_samples': df['baseline_samples'].iloc[0]
            }
        else:
            return {
                'baseline_mean': np.nan,
                'baseline_std': np.nan,
                'baseline_median': np.nan,
                'baseline_samples': 0
            }
    
    def extract_post_dose_locomotion(self, cage_id, replicate, dose_number=1, 
                                    window_start_min=300, window_end_min=500):
        """
        Extract post-dose locomotion PROPORTION during peak hyperlocomotion
        
        Based on diagnostic: peak response is 300-500 minutes post-injection
        """
        
        dose_info = self.dose_times[replicate]
        dose_time_str = dose_info[f'dose_{dose_number}_time']
        dose_date = dose_info[f'dose_{dose_number}_date']
        
        # Calculate time window
        dose_time = datetime.strptime(dose_time_str, '%Y-%m-%d %H:%M:%S')
        start_time = dose_time + timedelta(minutes=window_start_min)
        end_time = dose_time + timedelta(minutes=window_end_min)
        
        start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Handle date boundary crossing
        dates_involved = []
        current_date = dose_time.date()
        end_date = end_time.date()
        while current_date <= end_date:
            dates_involved.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        date_filter = " OR ".join([f"date = '{d}'" for d in dates_involved])
        
        # Use animal_bouts.locomotion - proportion of time locomoting
        query = f"""
        SELECT 
            AVG(value) as post_dose_mean,
            STDDEV(value) as post_dose_std,
            MAX(value) as post_dose_peak,
            COUNT(*) as post_dose_samples
        FROM read_parquet('{self.s3_base}/cage_id={cage_id}/date=*/animal_activity_db.parquet')
        WHERE ({date_filter})
            AND name = 'animal_bouts.locomotion'
            AND resolution = 1
            AND CAST(time AS TIMESTAMP) >= CAST('{start_time_str}' AS TIMESTAMP)
            AND CAST(time AS TIMESTAMP) < CAST('{end_time_str}' AS TIMESTAMP)
        """
        
        df = self.query(query)
        
        if len(df) > 0:
            return {
                'post_dose_mean': df['post_dose_mean'].iloc[0],
                'post_dose_std': df['post_dose_std'].iloc[0],
                'post_dose_peak': df['post_dose_peak'].iloc[0],
                'post_dose_samples': df['post_dose_samples'].iloc[0]
            }
        else:
            return {
                'post_dose_mean': np.nan,
                'post_dose_std': np.nan,
                'post_dose_peak': np.nan,
                'post_dose_samples': 0
            }
    
    def calculate_response(self, baseline, post_dose):
        """Calculate percent change in locomotion proportion"""
        
        if baseline['baseline_mean'] > 0 and not np.isnan(post_dose['post_dose_mean']):
            return ((post_dose['post_dose_mean'] - baseline['baseline_mean']) / 
                   baseline['baseline_mean'] * 100)
        return np.nan
    
    def extract_response_for_cage(self, cage_id, replicate, dose_number=1):
        """Extract complete morphine response for one cage"""
        
        dose = self.cage_treatments[cage_id]
        
        print(f"  Cage {cage_id}: {dose} mg/kg", end='', flush=True)
        
        # Extract baseline
        baseline = self.extract_baseline_locomotion(cage_id, replicate)
        print(f" | Baseline: {baseline['baseline_mean']:.4f}", end='', flush=True)
        
        # Extract post-dose (peak hyperlocomotion: 300-500 min)
        post_dose = self.extract_post_dose_locomotion(cage_id, replicate, dose_number)
        print(f" | Peak (300-500 min): {post_dose['post_dose_mean']:.4f}", end='', flush=True)
        
        # Calculate response
        response = self.calculate_response(baseline, post_dose)
        print(f" | Response: {response:.1f}%")
        
        result = {
            'cage_id': cage_id,
            'replicate': replicate,
            'dose_mg_kg': dose,
            'dose_number': dose_number,
            'morphine_response': response,
            'response_window': '300-500 min',
            'metric': 'locomotion_proportion'
        }
        
        result.update(baseline)
        result.update(post_dose)
        
        return result
    
    def extract_all_responses(self, dose_number=1, morphine_only=True):
        """Extract morphine responses for all cages"""
        
        all_responses = []
        
        for replicate_name, replicate_info in self.dose_times.items():
            print(f"\n{'='*70}")
            print(f" {replicate_name.upper().replace('_', ' ')} - Dose {dose_number}")
            print(f"{'='*70}")
            print(f" Dose time: {replicate_info[f'dose_{dose_number}_time']}")
            print(f" Peak hyperlocomotion window: 300-500 minutes (5-8 hours) post-injection")
            print(f" Metric: animal_bouts.locomotion (proportion of time)\n")
            
            for cage_id in replicate_info['cages']:
                dose = self.cage_treatments[cage_id]
                
                if morphine_only and dose == 0:
                    print(f"  Cage {cage_id}: SKIPPED (saline control)")
                    continue
                
                try:
                    response = self.extract_response_for_cage(
                        cage_id, replicate_name, dose_number
                    )
                    all_responses.append(response)
                    
                except Exception as e:
                    print(f"  Cage {cage_id}: ERROR - {e}")
                    import traceback
                    traceback.print_exc()
        
        df = pd.DataFrame(all_responses)
        return df


def main():
    """Extract with CORRECT metric and time window"""
    
    print("="*70)
    print(" FINAL CORRECTED MORPHINE RESPONSE EXTRACTION")
    print("="*70)
    
    print("\nBased on diagnostic analysis:")
    print("  ✓ Metric: animal_bouts.locomotion (proportion, not distance!)")
    print("  ✓ Time window: 300-500 minutes (5-8 hours post-injection)")
    print("  ✓ Expected: POSITIVE responses matching your time-course plot\n")
    
    try:
        extractor = MorphineResponseExtractorFinal()
        print("✓ DuckDB initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize DuckDB: {e}")
        return None
    
    # Extract responses
    responses_df = extractor.extract_all_responses(
        dose_number=1,
        morphine_only=True
    )
    
    # Save
    output_file = 'morphine_response_FINAL.csv'
    responses_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print(" EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"\n✓ Saved to: {output_file}")
    print(f"  - Cages: {len(responses_df)}")
    
    print(f"\n Summary by dose group:")
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
        print("\n ✓✓✓ SUCCESS! Mean response is now POSITIVE")
        print("     Morphine induces hyperlocomotion as expected!")
        print("     Ready for predictive modeling with baseline features")
    else:
        print("\n ⚠️  Still getting negative responses - needs further investigation")
    
    print("\n" + "="*70)
    print(" NEXT STEPS")
    print("="*70)
    print("\n1. Replace old outcomes:")
    print("   mv morphine_response_FINAL.csv morphine_response.csv")
    print("\n2. Run minimal model:")
    print("   python minimal_model.py")
    print("\n3. Expected: Positive R² and correlation with circadian amplitude!")
    
    return responses_df


if __name__ == '__main__':
    responses_df = main()
