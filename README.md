# DSC-180B Tasks - Morphine & Estrous Analysis

> Comprehensive behavioral analysis of morphine effects and estrous cycle interactions in mice using automated monitoring data from the Morph2REP study.

## Overview

This repository contains analyses for a DSC-180B capstone project examining:
1. **Morphine behavioral effects** - Dose-response, temporal dynamics, and multi-feature behavioral signatures
2. **Estrous cycle detection** - Activity-based cycle phase classification
3. **Integration** - Understanding how estrous phase modulates drug response (future work)

**Study:** Morph2REP (Study 1001, v3.3)
**Species:** Female C57BL/6J mice (strain 664)
**Sample Size:** 54 mice across 2 replicates
**Treatment Groups:** Vehicle, 5 mg/kg morphine, 25 mg/kg morphine

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Conda (Anaconda or Miniconda)

### Quick Start Installation

1. **Navigate to the project directory**
```bash
cd DSC-180B-Tasks
```

2. **Create and activate conda environment**
```bash
conda create -n morphine-analysis python=3.11
conda activate morphine-analysis
```

3. **Install all required packages**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter**
```bash
jupyter notebook
```

### Alternative: Optimized Installation (Recommended for Production)

For better performance with large datasets, install scientific packages via conda first:

```bash
# Steps 1-2 same as above, then:
conda install pandas numpy scipy matplotlib seaborn scikit-learn jupyter
pip install -r requirements.txt  # Will only install duckdb
jupyter notebook
```

This uses optimized BLAS/LAPACK libraries for faster numerical computations.

### AWS Credentials (Optional)

The notebooks load data from a public S3 bucket (`s3://jax-envision-public-data/`). No credentials are required for accessing public data.

If you encounter authentication issues, you can configure AWS:
```bash
aws configure
# Press Enter for all prompts if accessing public data only
```

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
├── estrous/                    # Estrous cycle detection
│   ├── morph2rep_estrous_calendar_days.ipynb
│   ├── estrous.ipynb
│   ├── morph2rep_estrous_analysis.ipynb
│   └── smarr.ipynb
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

### Estrous Cycle Analysis

#### [morph2rep_estrous_calendar_days.ipynb](estrous/morph2rep_estrous_calendar_days.ipynb)
**Primary estrous cycle detection for morphine study animals**

- Detects estrous cycle phases from activity patterns across calendar days
- Uses wavelet analysis and/or activity thresholding methods
- Aligns with morphine injection dates for future integration analysis

**Key outputs:**
- Estrous phase classifications by day
- Cycle length distributions
- Phase assignment for each animal

---

#### Other Estrous Notebooks
- **[estrous.ipynb](estrous/estrous.ipynb)** - Estrous detection methods development
- **[morph2rep_estrous_analysis.ipynb](estrous/morph2rep_estrous_analysis.ipynb)** - Alternative analysis approach
- **[smarr.ipynb](estrous/smarr.ipynb)** - Implementation of Smarr et al. wavelet method

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

## Key Findings

### Morphine Effects
- **Strong dose-response:** 25 mg/kg produces 7-54× increase in locomotion vs baseline
- **Reproducible:** Effects consistent across both replicates
- **Circadian interaction:** Higher fold-change at 5 PM injections (due to lower baseline)
- **No sensitization:** Similar responses between first and second doses
- **Multi-dimensional:** Morphine affects not just locomotion but also drinking, feeding patterns

### PCA Insights
- PC1 captures 57% of variance (active/locomotion vs drinking/feeding trade-off)
- PC2 captures 22% of variance
- Clear separation between dose groups in PC space
- Temporal trajectories show peak effect at 60-180 min post-injection

---

## Next Steps

1. **Estrous-Morphine Integration**
   - Combine estrous phase assignments with morphine response data
   - Test hypothesis: estrous phase modulates morphine sensitivity

2. **Individual Response Prediction**
   - Use baseline behavioral features to predict morphine response magnitude

3. **Extended Time Window**
   - Analyze longer-term effects (24-48 hours post-injection)

---

## References

- Envision platform documentation: [https://envision.jax.org](https://envision.jax.org)
- Smarr et al. wavelet estrous detection method
- Study protocol: Morph2REP v3.3 (internal)
