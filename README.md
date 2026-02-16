# Predicting Opioid Sensitivity from Baseline Behavioral Patterns (Capstone B22)

Individual differences in opioid sensitivity remain poorly understood. This project examines whether baseline behavioral patterns can predict acute morphine response in female mice. Using automated home-cage monitoring data from 54 female C57BL/6J mice, we measured morphine sensitivity as the percent increase in locomotor activity following injection relative to baseline levels.

We developed a predictive model using three pre-drug features: baseline locomotion, circadian rhythm strength, and rest fragmentation. The model was evaluated using cage-level cross-validation to account for shared housing effects and demonstrated that baseline features could meaningfully predict variability in morphine response.

**Key finding:** Circadian rhythm strength emerged as the strongest predictor—mice with more robust circadian rhythms showed greater morphine-induced hyperactivity. These results suggest that baseline circadian organization may serve as a behavioral marker of opioid sensitivity and highlight the potential of home-cage monitoring to predict individual differences in drug responsiveness.

**Study:** Morph2REP (Study 1001, v3.3)
**Species:** Female C57BL/6J mice (strain 664)
**Sample Size:** 54 mice across 2 replicates
**Treatment Groups:** Vehicle, 5 mg/kg morphine, 25 mg/kg morphine

---

## Prerequisites

- **Python 3.8+** (3.11 recommended)
- **Docker** (optional, for containerized environment)

## Setup Instructions

### Option 1: Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Michaelvasandani/DSC-180B-Tasks.git
   cd DSC-180B-Tasks
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n morphine-analysis python=3.11
   conda activate morphine-analysis
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Lab**
   ```bash
   jupyter lab
   ```
   Navigate to the analysis notebooks and open any notebook.

### Option 2: Docker Setup

1. **Build the Docker image**
   ```bash
   docker build -t morphine-analysis .
   ```

2. **Run the container**
   ```bash
   docker run -p 8888:8888 -v $(pwd):/workspace morphine-analysis
   ```

3. **Access Jupyter Lab**
   Open your browser and navigate to: `http://localhost:8888`

### Verify Installation

Test that all packages are installed correctly:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
from sklearn.decomposition import PCA

print("✓ All packages imported successfully!")
```

---

## Directory Structure

```
DSC-180B-Tasks/
├── morphine_analysis/          # Morphine behavioral analysis
│   ├── Morphine_eda_all_doses.ipynb
│   ├── Morphine_feature_interaction.ipynb
│   └── pca_analysis/
│       └── PCA_analysis.ipynb
├── morphine_features/          # Morphine feature exploration & heatmaps
│   ├── morphine_features.ipynb
│   ├── morphine_heatmap.ipynb
│   └── morphine_heatmap2.ipynb
├── test_model/                 # Opioid sensitivity prediction pipeline
│   ├── README_FINAL (1).md
│   ├── feature_extraction_final (1).py
│   ├── outcome_extraction_final (1) (1).py
│   ├── modeling_pipeline (1).py
│   ├── dose_stratified_analysis (1).py
│   └── ... (additional scripts)
├── estrous/                    # Estrous cycle detection
│   ├── morph2rep_estrous_calendar_days.ipynb
│   ├── estrous.ipynb
│   ├── morph2rep_estrous_analysis.ipynb
│   ├── smarr.ipynb
│   ├── estrus_plateau_core_analysis.ipynb
│   └── estrus_plateau_extended_analysis_per_cage.ipynb
├── fault_detection/            # Tracking quality analysis
│   └── fault_analysis.ipynb
├── Dockerfile                  # Docker configuration
├── .dockerignore               # Docker ignore patterns
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── technical_documentation.md  # Detailed PCA documentation
```

---

## Analysis Notebooks

### Morphine Analysis

#### [Morphine_eda_all_doses.ipynb](morphine_analysis/Morphine_eda_all_doses.ipynb)
**Primary morphine analysis across all injection events**

- Analyzes all 4 injection events (2 replicates × 2 doses each)
- Compares dose-response relationships (Vehicle vs 5 mg/kg vs 25 mg/kg)
- Examines circadian effects (6 AM vs 5 PM injections)
- Tests for sensitization/tolerance (first vs second dose)
- Statistical validation using Wilcoxon signed-rank tests
- Generates comprehensive visualizations and summary statistics

**Key outputs:**
- Locomotor activity time series by dose group
- Individual animal response traces
- Circadian comparison plots
- Dose-response heatmaps
- Summary statistics CSVs

---

#### [Morphine_feature_interaction.ipynb](morphine_analysis/Morphine_feature_interaction.ipynb)
**Individual consistency analysis across behavioral dimensions**

- Examines correlations between 7 behavioral features per animal
- Features: active, inactive, locomotion, climbing, drinking, feeding, sleep
- Computes percent change from baseline for multiple time windows
- Assesses whether animals with strong locomotor responses also show strong changes in other behaviors
- Identifies individual response patterns

**Key outputs:**
- Correlation matrices for feature interactions
- Individual response profiles
- Consistency plots by dose group

---

#### [pca_analysis/PCA_analysis.ipynb](morphine_analysis/pca_analysis/PCA_analysis.ipynb)
**Dimensionality reduction to identify behavioral state signatures**

- Reduces 4 key features (active, locomotion, drinking, feeding) to 2 principal components
- Uses 30-minute aggregated data for noise reduction
- Visualizes behavioral trajectories through PC space over time
- Compares dose groups in unified behavioral space
- Analyzes centroid trajectories to quantify dose effects

**Key outputs:**
- PCA scatter plots colored by time window
- PCA scatter plots colored by feature intensity
- Dose comparison overlays
- Feature loading bar charts

**See [technical_documentation.md](technical_documentation.md) for detailed PCA methodology.**

---

#### [morphine_features/](morphine_features/)
**Morphine feature exploration and heatmap visualizations**

- **[morphine_features.ipynb](morphine_features/morphine_features.ipynb)** - Feature extraction and exploration
- **[morphine_heatmap.ipynb](morphine_features/morphine_heatmap.ipynb)** - Heatmap visualization of morphine effects
- **[morphine_heatmap2.ipynb](morphine_features/morphine_heatmap2.ipynb)** - Additional heatmap analyses

---

### Fault Detection

#### [fault_detection/fault_analysis.ipynb](fault_detection/fault_analysis.ipynb)
Validates tracking system quality by identifying missing data, unusual inactivity patterns, and potential sensor failures across both replicates. Aggregates quality metrics per animal and cage using heatmaps and streak detection to flag animals with poor tracking coverage.

---

### Estrous Cycle Analysis

#### [morph2rep_estrous_calendar_days.ipynb](estrous/morph2rep_estrous_calendar_days.ipynb)
Detects estrous cycle phases from activity patterns across calendar days using wavelet analysis and activity thresholding methods. Aligns with morphine injection dates for future integration analysis.

#### [estrous.ipynb](estrous/estrous.ipynb)
Estrous detection methods development and exploration.

#### [morph2rep_estrous_analysis.ipynb](estrous/morph2rep_estrous_analysis.ipynb)
Alternative approach to estrous cycle analysis for morphine study animals.

#### [smarr.ipynb](estrous/smarr.ipynb)
Implementation of the Smarr et al. wavelet-based estrous detection method.

#### [estrus_plateau_core_analysis.ipynb](estrous/estrus_plateau_core_analysis.ipynb)
Estrus-linked locomotor plateau detection using 60-second resolution per-animal locomotion data.

#### [estrus_plateau_extended_analysis_per_cage.ipynb](estrous/estrus_plateau_extended_analysis_per_cage.ipynb)
Extended permutation testing and visualization of estrus-linked plateau structure analyzed per cage.

---

## Modeling Pipeline

### [test_model/](test_model/)
**Opioid sensitivity prediction pipeline**

A complete automated pipeline for predicting morphine response from baseline behavioral features. The pipeline downloads data from S3, extracts features, and trains predictive models—no manual data preparation required.

**Quick Start:**
```bash
cd test_model/
python run_pipeline_final.py
```

**Expected runtime:** 30-60 minutes

**What the pipeline does:**
1. Downloads data from S3 using DuckDB (no AWS credentials needed)
2. Extracts ~40 baseline behavioral features per cage
3. Calculates morphine response outcomes
4. Trains and validates 4 predictive models
5. Generates visualizations and reports

**Command line options:**
```bash
# Use existing features/outcomes (skip extraction)
python run_pipeline_final.py --skip-features --skip-outcomes

# Add permutation test for significance
python run_pipeline_final.py --permutation

# Add bootstrap confidence intervals
python run_pipeline_final.py --bootstrap
```

**Key Scripts:**
- **[run_pipeline_final.py](test_model/run_pipeline_final.py)** - Main entry point
- **[feature_extraction_final (1).py](test_model/feature_extraction_final%20(1).py)** - Extracts 40+ baseline behavioral features
- **[outcome_extraction_final (1) (1).py](test_model/outcome_extraction_final%20(1)%20(1).py)** - Calculates morphine response outcomes
- **[modeling_pipeline (1).py](test_model/modeling_pipeline%20(1).py)** - Trains and validates 4 predictive models
- **[dose_stratified_analysis (1).py](test_model/dose_stratified_analysis%20(1).py)** - Dose-specific prediction analysis

**Features extracted (40+):**
- **Circadian rhythm** (6 features): amplitude, acrophase, robustness, light/dark ratio
- **Locomotion** (7 features): distance, speed, variability, peak activity
- **Rest fragmentation** (5 features): bout duration, fragmentation index, transitions
- **Bout structure** (8 features): frequency, duration, CV, trade-off residual, burstiness
- **Temporal patterns** (4 features): inter-bout intervals, variability
- **Activity states** (4 features): active, inactive, climbing proportions
- Social behavior, respiration, drinking

**Research hypotheses tested:**
- **H1:** Higher circadian amplitude → greater morphine response
- **H2:** Fragmented rest → greater morphine sensitivity
- **H3:** Higher baseline variability → greater response
- **H4:** Bout frequency-duration trade-off predicts response
- **H5:** Stronger prediction at lower doses

**See [README_FINAL (1).md](test_model/README_FINAL%20(1).md) for complete documentation.**

---

## Data Sources

All data loaded from AWS S3:
```
s3://jax-envision-public-data/study_1001/2025v3.3/tabular/
└── cage_id={cage_id}/
    └── date={date}/
        ├── animal_activity_db.parquet
        ├── animal_drinking.parquet
        └── animal_tsdb_mvp.parquet
```

**Key metrics:**
- Locomotion, active/inactive states, climbing
- Drinking, feeding behavior
- Inferred sleep states
- All at 60-second resolution

---

## Experimental Design

### Injection Schedule

| Replicate | Dose 1          | Dose 2          |
|-----------|-----------------|-----------------|
| Rep 1     | Jan 14, 6:00 AM | Jan 17, 5:00 PM |
| Rep 2     | Jan 28, 5:00 PM | Jan 31, 6:00 AM |

### Dose Groups (n=18 per group)

| Group     | Description                  |
|-----------|------------------------------|
| Vehicle   | Saline control               |
| 5 mg/kg   | Low dose morphine            |
| 25 mg/kg  | High dose morphine           |

---

## References

- Smarr et al. wavelet estrous detection method
- Study protocol: Morph2REP v3.3 (internal)

---

## Contributors

- Michael Vasandani
- Matthew Budding
- Pansy Kuang
- Kyra Deng
