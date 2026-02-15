# Opioid Sensitivity Prediction


### 3. Run the Pipeline

```bash
python run_pipeline_final.py
```

That's it! The pipeline will:
1. Download data from S3 using DuckDB (no AWS credentials needed)
2. Extract ~40 baseline behavioral features per cage
3. Calculate morphine response outcomes
4. Train and validate 4 predictive models
5. Generate visualizations and reports

**Expected runtime**: 30-60 minutes total

## What Gets Extracted

### Baseline Features (40+)

**Tier 1 - High Priority:**
1. **Circadian rhythm** (6 features)
   - Amplitude, acrophase, robustness, light/dark ratio
   
2. **Locomotion** (7 features)
   - Distance, speed, variability, peak activity
   
3. **Rest fragmentation** (5 features)
   - Bout duration, fragmentation index, transitions
   
4. **Bout structure** (8 features)
   - Frequency, duration, CV, trade-off residual, burstiness
   
5. **Temporal patterns** (4 features)
   - Inter-bout intervals, variability
   
6. **Activity states** (4 features)
   - Active, inactive, climbing proportions

**Tier 2 - Moderate Priority:**
7. Social behavior, respiration, drinking

### Outcome Variable

- **Morphine response**: Percent change in locomotion (0-60 min post-injection)
- Multiple calculation methods available

## Command Line Options

```bash
# Full run (recommended first time)
python run_pipeline_final.py

# Use existing features/outcomes (skip extraction)
python run_pipeline_final.py --skip-features --skip-outcomes

# Add permutation test for significance
python run_pipeline_final.py --permutation

# Add bootstrap confidence intervals
python run_pipeline_final.py --bootstrap

# Quick test run (100 permutations instead of 1000)
python run_pipeline_final.py --permutation --quick
```

## Complete Feature List

### Locomotion & Activity
- `mean_distance_cm_per_s` - Average velocity
- `distance_cv` - Variability coefficient
- `peak_distance_95pct` - Peak speed
- `total_distance_cm` - Total distance traveled
- `active_proportion`, `inactive_proportion`, `climbing_proportion`
- `locomotion_total_seconds`, `active_total_seconds`, etc.

### Circadian Rhythm
- `circadian_amplitude` - Rhythm strength (cosinor fit)
- `circadian_acrophase` - Peak activity time (hours)
- `circadian_r_squared` - Rhythm robustness
- `light_dark_ratio` - Day/night activity ratio
- `light_phase_activity`, `dark_phase_activity`

### Bout Structure
- `locomotion_bout_frequency_per_hour` - Activity initiation rate
- `locomotion_bout_mean_duration` - Average bout length
- `locomotion_bout_duration_cv` - Bout variability
- `rest_fragmentation_index` - Sleep disruption metric
- `inactive_bout_mean_duration` - Rest consolidation

### Temporal Patterns
- `inter_bout_interval_median` - Time between bouts
- `inter_bout_interval_cv` - Temporal variability
- `burstiness_index` - Activity regularity (-1 to +1)
- `state_transition_rate_per_hour` - Behavioral instability

### Social Behavior
- `mean_distance_to_cagemates_cm` - Social proximity
- `nearest_neighbor_distance_cm` - Closest approach

### Physiological
- `mean_respiration_rate_bpm` - Baseline respiration
- `respiration_rate_cv` - Respiratory variability

### Consummatory
- `total_drinking_seconds` - Drinking time

## Research Hypotheses Tested

This pipeline tests 5 hypotheses from your research plan:

**H1**: Higher circadian amplitude → greater morphine response
- Features: `circadian_amplitude`, `circadian_r_squared`, `light_dark_ratio`

**H2**: Fragmented rest → greater morphine sensitivity
- Features: `rest_fragmentation_index`, `rest_bout_mean_duration`, `state_transition_rate_per_hour`

**H3**: Higher baseline variability → greater response
- Features: `distance_cv`, `locomotion_bout_duration_cv`, `inter_bout_interval_cv`

**H4**: Bout frequency-duration trade-off predicts response
- Feature: `freq_duration_tradeoff_residual` (calculated from regression)

**H5**: Stronger prediction at lower doses
- Tested via dose-stratified analysis

## License

This code is provided for research use with public JAX Envision data.

---


