# Comprehensive PCA Analysis Documentation
## Morphine Effects on Mouse Behavior - Morph2REP Study

**Study:** Morph2REP (Study 1001 Version 2025v3.3)  
**Analysis by:** Michael (UC San Diego / TLR Ventures Capstone)  
**Purpose:** Characterize behavioral signature of morphine using PCA

---

## Table of Contents

1. [Study Background](#1-study-background)
2. [Analysis Objectives](#2-analysis-objectives)
3. [Data Sources](#3-data-sources)
4. [Features Used](#4-features-used)
5. [Data Processing Pipeline](#5-data-processing-pipeline)
6. [Outlier Filtering](#6-outlier-filtering)
7. [PCA Methodology](#7-pca-methodology)
8. [Output Directory Structure](#8-output-directory-structure)
9. [Plot Descriptions](#9-plot-descriptions)
10. [Interpretation Guide](#10-interpretation-guide)
11. [Technical Specifications](#11-technical-specifications)
12. [Appendix](#appendix)

---

## 1. Study Background

### 1.1 Experimental Design

The **Morph2REP** (Morphine Treatment Two Replicates) study evaluated morphine's physiological and behavioral effects at two doses in female strain 664 mice.

| Parameter | Value |
|-----------|-------|
| **Total Animals** | 54 female mice |
| **Animals per Dose Group** | 18 mice (6 cages × 3 mice/cage) |
| **Treatment Groups** | Vehicle (control), 5 mg/kg morphine, 25 mg/kg morphine |
| **Replicates** | 2 independent replicates |
| **Injections per Replicate** | 2 doses (pooled for this analysis) |
| **Monitoring System** | Envision 2025 v3.0 automated behavioral monitoring |

### 1.2 Temporal Structure

#### Replicate 1 (Cages 4917-4925)
| Event | Date | Time (EST) | Time (UTC) |
|-------|------|------------|------------|
| Dose 1 | January 14, 2025 | 6:00 AM | 11:00 AM |
| Dose 2 | January 17, 2025 | 5:00 PM | 10:00 PM |

#### Replicate 2 (Cages 4926-4934)
| Event | Date | Time (EST) | Time (UTC) |
|-------|------|------------|------------|
| Dose 1 | January 28, 2025 | 5:00 PM | 10:00 PM |
| Dose 2 | January 31, 2025 | 6:00 AM | 11:00 AM |

### 1.3 Cage-to-Dose Mapping

#### Replicate 1
| Cage ID | Dose Group | Cage Name | Animal IDs |
|---------|------------|-----------|------------|
| 4917 | 5 mg/kg | Resp-1 (A1) | 523-1, 523-2, 523-3 |
| 4918 | Vehicle | Resp-2 (B1) | 523-4, 523-5, 523-6 |
| 4919 | 25 mg/kg | Resp-3 (C1) | 523-7, 523-8, 523-9 |
| 4920 | 25 mg/kg | Resp-4 (A2) | 523-10, 523-11, 523-12 |
| 4921 | 5 mg/kg | Resp-5 (B2) | 523-13, 523-14, 523-15 |
| 4922 | Vehicle | Resp-6 (C2) | 523-16, 523-17, 523-18 |
| 4923 | Vehicle | Resp-7 (A3) | 523-19, 523-20, 523-21 |
| 4924 | 25 mg/kg | Resp-8 (B3) | 523-22, 523-23, 523-24 |
| 4925 | 5 mg/kg | Resp-9 (C3) | 523-25, 523-26, 523-27 |

#### Replicate 2
| Cage ID | Dose Group | Cage Name | Animal IDs |
|---------|------------|-----------|------------|
| 4926 | 25 mg/kg | Resp-10 (A1) | 523-28, 523-29, 523-30 |
| 4927 | 5 mg/kg | Resp-11 (B1) | 523-31, 523-32, 523-33 |
| 4928 | Vehicle | Resp-12 (C1) | 523-34, 523-35, 523-36 |
| 4929 | Vehicle | Resp-13 (A2) | 523-37, 523-38, 523-39 |
| 4930 | 25 mg/kg | Resp-14 (B2) | 523-40, 523-41, 523-42 |
| 4931 | 5 mg/kg | Resp-15 (C2) | 523-43, 523-44, 523-45 |
| 4932 | 5 mg/kg | Resp-16 (A3→B3) | 523-46, 523-47, 523-48 |
| 4933 | 25 mg/kg | Resp-17 (B3→A3) | 523-49, 523-50, 523-51 |
| 4934 | Vehicle | Resp-18 (C3) | 523-52, 523-53, 523-54 |

---

## 2. Analysis Objectives

### 2.1 Primary Goal
Determine whether morphine creates a **distinct, separable behavioral state** that can be visualized in a reduced-dimensional space (PCA).

### 2.2 Specific Questions

1. **Dose-Response:** Does 25 mg/kg produce a more distinct behavioral signature than 5 mg/kg?
2. **Temporal Dynamics:** How does the behavioral state evolve over time post-injection?
3. **Replicate Consistency:** Are findings reproducible across independent replicates?
4. **Feature Contributions:** Which behavioral features drive the observed separation?

### 2.3 Downstream Purpose
This morphine analysis will be combined with a partner's estrous cycle analysis to study **morphine-estrous interactions** in subsequent work.

---

## 3. Data Sources

### 3.1 Primary Data Tables

| Table | Description | Resolution | Key Metrics |
|-------|-------------|------------|-------------|
| `animal_activity_db.parquet` | Activity state proportions | 1s, 10s, 60s | active, inactive, locomotion, climbing |
| `animal_drinking.parquet` | Consummatory behaviors | 1s, 10s, 60s | drinking, feeding |

### 3.2 S3 Data Location
```
s3://jax-envision-public-data/study_1001/2025v3.3/tabular/
└── cage_id={cage_id}/
    └── date={date}/
        ├── animal_activity_db.parquet
        └── animal_drinking.parquet
```

### 3.3 Data Loading Strategy
- **Query Engine:** DuckDB with S3 integration
- **Resolution Filter:** 60-second aggregation only (reduces noise)
- **Time Window:** -180 to +540 minutes from injection
- **Events Loaded:** All 4 injection events (2 per replicate)
- **Pooling:** Doses pooled within each replicate

---

## 4. Features Used

### 4.1 Selected Features (4 total)

| Feature Name | Full Column Name | Units | Value Range | Description |
|--------------|------------------|-------|-------------|-------------|
| **Active** | `animal_bouts.active` | Proportion | 0.0 - 1.0 | Fraction of time in general active state |
| **Locomotion** | `animal_bouts.locomotion` | Proportion | 0.0 - 1.0 | Fraction of time in horizontal movement |
| **Drinking** | `animal_bouts.drinking` | Proportion | 0.0 - 1.0 | Fraction of time at water source |
| **Feeding** | `animal_bouts.feeding` | Proportion | 0.0 - 1.0 | Fraction of time at food hopper |

### 4.2 Feature Selection Rationale

| Included | Excluded | Reason |
|----------|----------|--------|
| Active | Inactive | Inactive is redundant (≈ 1 - Active), high negative correlation |
| Locomotion | Climbing | Climbing showed low variance; locomotion captures main movement |
| Drinking | - | Independent signal for consummatory behavior |
| Feeding | - | Independent signal for consummatory behavior |

---

## 5. Data Processing Pipeline

### 5.1 Pipeline Overview

```
Raw Data (S3)
     │
     ▼
┌─────────────────────────────────────────┐
│ 1. LOAD DATA                            │
│    - animal_activity_db (all cages)     │
│    - animal_drinking (all cages)        │
│    - Filter: resolution = 60 seconds    │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ 2. ALIGN TO INJECTION TIME              │
│    - Convert timestamps to UTC          │
│    - Calculate minutes_from_injection   │
│    - Filter: -180 to +540 minutes       │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ 3. MERGE & PIVOT                        │
│    - Combine activity + drinking        │
│    - Pivot: rows = observations         │
│             columns = features          │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ 4. AGGREGATE TO 30-MIN BINS             │
│    - Group by: cage, animal, time_bin   │
│    - Aggregate: mean of each feature    │
│    - Assign time_window labels          │
└─────────────────────────────────────────┘
     │
     ├──────────────────┬──────────────────┐
     ▼                  ▼                  │
┌──────────────┐  ┌──────────────────┐     │
│ UNFILTERED   │  │ 5. FILTER        │     │
│ → pca_analysis│  │    OUTLIERS      │     │
└──────────────┘  │    (1st-99th %)  │     │
                  │    → pca_filtered │     │
                  └──────────────────┘     │
     │                  │                  │
     ▼                  ▼                  │
┌─────────────────────────────────────────┐
│ 6. RUN PCA                              │
│    - Standardize features (z-score)     │
│    - Extract 2 principal components     │
│    - Generate visualizations            │
└─────────────────────────────────────────┘
```

### 5.2 Time Window Definitions

| Window Name | Minutes from Injection | Description |
|-------------|----------------------|-------------|
| `baseline` | -180 to -60 | Pre-injection normal behavior |
| `pre_injection` | -60 to 0 | Immediate pre-injection period |
| `immediate` | 0 to 30 | Drug onset phase |
| `peak_early` | 30 to 90 | Early peak effect |
| `peak_sustained` | 90 to 180 | Sustained peak effect |
| `decline_early` | 180 to 300 | Initial drug washout |
| `decline_late` | 300 to 420 | Late washout phase |
| `post_6hr+` | 420 to 540 | Recovery period |

### 5.3 Aggregation Levels

#### Level 1: 30-Minute Bins (Primary Analysis)
- **Grouping:** cage_id, dose_group, animal_id, time_bin (30 min), replicate
- **Aggregation:** Mean of each feature within bin
- **Purpose:** Reduces minute-to-minute noise while preserving temporal dynamics

#### Level 2: Animal × Time Window (Aggregated Analysis)
- **Grouping:** cage_id, dose_group, animal_id, time_window, replicate
- **Aggregation:** Mean of each feature within time window
- **Purpose:** Further noise reduction for cleaner visualizations

---

## 6. Outlier Filtering

### 6.1 Purpose
Raw behavioral data contains extreme values that can compress the main data distribution in PCA visualizations. Outlier filtering removes these extremes to better visualize the core behavioral patterns.

### 6.2 Filtering Method

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Method** | Quantile-based | Remove observations outside specified percentiles |
| **Lower Bound** | 1st percentile | Remove bottom 1% of values |
| **Upper Bound** | 99th percentile | Remove top 1% of values |
| **Applied To** | All 4 features | active, locomotion, drinking, feeding |
| **Logic** | AND across features | Point removed if ANY feature is outside bounds |

### 6.3 Filtering Formula

For each feature, observations are kept if:
```
Q1(feature) ≤ value ≤ Q99(feature)
```

An observation is included only if ALL features pass this check.

### 6.4 Output Directories

Two separate analysis pipelines are produced:

| Directory | Description | Use Case |
|-----------|-------------|----------|
| `pca_analysis/` | Unfiltered data | Complete picture, includes all outliers |
| `pca_filtered/` | Outlier-filtered data | Cleaner visualization of main patterns |

### 6.5 Filtering Impact

Typical filtering removes ~2-4% of observations while preserving the main behavioral distributions. This improves:
- Visual clarity of dose group separation
- Interpretability of PC loadings
- Identification of temporal trajectories

---

## 7. PCA Methodology

### 7.1 Algorithm Details

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Algorithm** | Principal Component Analysis (PCA) | Linear dimensionality reduction |
| **Library** | scikit-learn `PCA` | Industry standard implementation |
| **Number of Components** | 2 | Visualization in 2D; captures majority of variance |
| **Preprocessing** | StandardScaler (z-score) | Required for PCA; equalizes feature scales |
| **Missing Data** | Row-wise deletion | Rows with any NaN excluded from PCA |

### 7.2 Standardization Formula

For each feature *x*:

```
z = (x - μ) / σ
```

Where:
- μ = mean of feature across all observations
- σ = standard deviation of feature

### 7.3 PCA Computation Steps

1. Compute covariance matrix of standardized features
2. Extract eigenvalues and eigenvectors
3. Sort by eigenvalue (variance explained)
4. Project data onto top 2 eigenvectors (PC1, PC2)

### 7.4 PCA Fitting Strategies

| Analysis Type | PCA Fit On | Transform | Purpose |
|---------------|------------|-----------|---------|
| **Dose-Specific** | Single dose group only | Same dose group | See within-dose structure |
| **Combined** | All dose groups together | All dose groups | Compare doses in same space |
| **Aggregated** | Animal-level means | Animal-level means | Reduce noise, cleaner patterns |

---

## 8. Output Directory Structure

### 8.1 Complete Tree

Two output directories are generated: `pca_analysis/` (unfiltered) and `pca_filtered/` (outlier-filtered). Both have identical structure:

```
pca_analysis/                   # UNFILTERED DATA
│
├── all_time_windows/           # FULL ANALYSIS: -180 to +540 min
│   │
│   ├── by_dose/                # DOSE-SPECIFIC PCAs
│   │   │                       # (PCA fit separately on each dose)
│   │   │
│   │   ├── vehicle/            # Vehicle control only
│   │   │   ├── pca_time_window.png    # Colored by time window
│   │   │   ├── pca_active.png         # Colored by active intensity
│   │   │   ├── pca_locomotion.png     # Colored by locomotion intensity
│   │   │   ├── pca_drinking.png       # Colored by drinking intensity
│   │   │   └── pca_feeding.png        # Colored by feeding intensity
│   │   │
│   │   ├── 5mg/                # 5 mg/kg morphine only
│   │   │   └── (same 5 plots)
│   │   │
│   │   └── 25mg/               # 25 mg/kg morphine only
│   │       └── (same 5 plots)
│   │
│   ├── combined/               # COMBINED PCAs
│   │   │                       # (PCA fit on all doses together)
│   │   │
│   │   ├── pca_replicate_dose.png     # 6-way coloring (Rep×Dose)
│   │   ├── pca_faceted_by_dose.png    # Side-by-side, time colors
│   │   ├── pca_by_dose.png            # All points, colored by dose
│   │   ├── pca_by_replicate.png       # All points, colored by replicate
│   │   ├── pca_by_active.png          # All points, colored by active
│   │   ├── pca_by_locomotion.png      # All points, colored by locomotion
│   │   ├── pca_by_drinking.png        # All points, colored by drinking
│   │   ├── pca_by_feeding.png         # All points, colored by feeding
│   │   └── pca_loadings.png           # Feature contribution bars
│   │
│   └── aggregated/             # AGGREGATED TO ANIMAL LEVEL
│       │                       # (Mean per animal per time_window)
│       │
│       ├── vehicle/            # Vehicle aggregated
│       │   ├── pca_time_window.png
│       │   ├── pca_active.png
│       │   ├── pca_locomotion.png
│       │   ├── pca_drinking.png
│       │   └── pca_feeding.png
│       │
│       ├── 5mg/                # 5 mg/kg aggregated
│       │   └── (same 5 plots)
│       │
│       ├── 25mg/               # 25 mg/kg aggregated
│       │   └── (same 5 plots)
│       │
│       └── combined/           # All doses aggregated combined
│           ├── pca_replicate_dose.png
│           ├── pca_faceted_by_dose.png
│           ├── pca_by_dose.png
│           ├── pca_by_replicate.png
│           ├── pca_by_active.png
│           ├── pca_by_locomotion.png
│           ├── pca_by_drinking.png
│           └── pca_by_feeding.png
│
└── peak_effect/                # PEAK EFFECT ONLY: 0 to 180 min
    │                           # (Same structure as all_time_windows)
    │
    ├── by_dose/
    │   ├── vehicle/
    │   ├── 5mg/
    │   └── 25mg/
    │
    ├── combined/
    │
    └── aggregated/
        ├── vehicle/
        ├── 5mg/
        ├── 25mg/
        └── combined/

pca_filtered/                   # OUTLIER-FILTERED DATA
│                               # (Same structure as pca_analysis/)
└── ...
```

### 8.2 Folder Purpose Descriptions

| Folder | Purpose | When to Use |
|--------|---------|-------------|
| `pca_analysis/` | Unfiltered data with all observations | Complete picture including outliers |
| `pca_filtered/` | Outlier-filtered data (1st-99th percentile) | Cleaner visualization of main patterns |
| `all_time_windows/` | Full temporal analysis (-180 to +540 min) | Understanding complete drug dynamics |
| `peak_effect/` | Focused on max drug effect (0 to 180 min) | Testing dose separation at peak |
| `by_dose/` | Each dose analyzed separately (PCA fit per dose) | Understanding within-dose structure |
| `combined/` | All doses in same PC space (single PCA fit) | Comparing doses directly |
| `aggregated/` | Reduced noise via animal-level averaging | Cleaner visualizations, presentations |
| `vehicle/` | Vehicle control group only | Baseline behavioral reference |
| `5mg/` | 5 mg/kg morphine group only | Low dose analysis |
| `25mg/` | 25 mg/kg morphine group only | High dose analysis |

### 8.3 File Count Summary

| Analysis | Dose-Specific | Combined | Aggregated | Total per Directory |
|----------|---------------|----------|------------|---------------------|
| All Time Windows | 15 (5×3 doses) | 9 | 23 (15 dose + 8 combined) | 47 |
| Peak Effect | 15 (5×3 doses) | 9 | 23 (15 dose + 8 combined) | 47 |
| **Total per Directory** | 30 | 18 | 46 | **94 plots** |
| **Grand Total (both directories)** | | | | **188 plots** |

---

## 9. Plot Descriptions

### 9.1 Time Window Plot (`pca_time_window.png`)

**Purpose:** Visualize temporal trajectory through PC space

**Coloring Scheme:**
| Color | Window | Minutes |
|-------|--------|---------|
| Dark Green | baseline | -180 to -60 |
| Light Green | pre_injection | -60 to 0 |
| Orange | immediate | 0 to 30 |
| Bright Red | peak_early | 30 to 90 |
| Dark Red | peak_sustained | 90 to 180 |
| Purple | decline_early | 180 to 300 |
| Blue | decline_late | 300 to 420 |
| Teal | post_6hr+ | 420 to 540 |

**What to Look For:**
- Temporal progression through PC space
- Clustering vs. spreading at different time points
- Recovery patterns

### 9.2 Feature-Colored Plots (`pca_[feature].png` and `pca_by_[feature].png`)

**Purpose:** Understand which features drive position in PC space

**Available Features:**
- `pca_active.png` / `pca_by_active.png` - Colored by active state proportion
- `pca_locomotion.png` / `pca_by_locomotion.png` - Colored by locomotion proportion
- `pca_drinking.png` / `pca_by_drinking.png` - Colored by drinking proportion
- `pca_feeding.png` / `pca_by_feeding.png` - Colored by feeding proportion

**Coloring Scheme:** Viridis colormap (purple = low, yellow = high)

**What to Look For:**
- Gradient direction indicates feature's contribution to that PC axis
- Sharp gradients = strong contribution
- No gradient = minimal contribution

### 9.3 Replicate-Dose Plot (`pca_replicate_dose.png`)

**Purpose:** Assess replicate consistency and batch effects

**Coloring Scheme (6-way):**
| Color | Group |
|-------|-------|
| Light Gray | Rep1_Vehicle |
| Dark Gray | Rep2_Vehicle |
| Light Blue | Rep1_5 mg/kg |
| Dark Blue | Rep2_5 mg/kg |
| Light Red | Rep1_25 mg/kg |
| Dark Red | Rep2_25 mg/kg |

**What to Look For:**
- Overlap between same-dose groups across replicates
- Separation between different dose groups

### 9.4 Faceted by Dose Plot (`pca_faceted_by_dose.png`)

**Purpose:** Side-by-side comparison of dose groups in same PC space

**Layout:** 3 panels (Vehicle | 5 mg/kg | 25 mg/kg)

**Coloring:** Time windows (same as 9.1)

**What to Look For:**
- Compare spread/variance between doses
- Compare trajectory shape between doses

### 9.5 By Dose Plot (`pca_by_dose.png`)

**Purpose:** Single plot with all points colored by dose group

**Coloring:**
| Color | Dose |
|-------|------|
| Gray | Vehicle |
| Blue | 5 mg/kg |
| Red | 25 mg/kg |

**What to Look For:**
- Separation or overlap between dose groups
- Relative clustering/spread of each dose

### 9.6 By Replicate Plot (`pca_by_replicate.png`)

**Purpose:** Single plot with all points colored by replicate

**Coloring:**
| Color | Replicate |
|-------|-----------|
| Blue | Rep1 |
| Red | Rep2 |

**What to Look For:**
- Overlap indicates good replicate consistency
- Separation indicates batch effects

### 9.7 Loadings Plot (`pca_loadings.png`)

**Purpose:** Interpret what each PC represents

**Layout:** 2 bar charts (PC1 | PC2)

**Coloring:** Red = positive loading, Blue = negative loading

**Reading the Plot:**
- Longer bars = stronger contribution to that PC
- Sign indicates direction of relationship
- Compare relative magnitudes across features

---

## 10. Interpretation Guide

### 10.1 How to Read PCA Plots

#### Position in PC Space
- **Similar positions** = similar behavioral profiles
- **Distant positions** = different behavioral states
- **Tight clusters** = constrained behavior
- **Spread/scatter** = variable behavior

#### Trajectories
- **Linear trajectory** = systematic change along one behavioral dimension
- **Curved trajectory** = complex state evolution
- **Return to origin** = recovery to baseline

### 10.2 PC Interpretation Template

Based on loadings, interpret each PC as:

**PC1:**
- Positive direction: [Features with positive loadings]
- Negative direction: [Features with negative loadings]

**PC2:**
- Positive direction: [Features with positive loadings]
- Negative direction: [Features with negative loadings]

### 10.3 Comparing Groups

| Pattern | Interpretation |
|---------|----------------|
| Complete overlap | Similar behavioral profiles |
| Partial overlap | Some behavioral differences |
| Clear separation | Distinct behavioral states |
| Different spread | Different behavioral variability |
| Different trajectory | Different temporal dynamics |

### 10.4 Filtered vs Unfiltered

| Use Case | Recommended Directory |
|----------|----------------------|
| Publication figures | `pca_filtered/` - cleaner visuals |
| Complete data analysis | `pca_analysis/` - includes all observations |
| Identifying outliers | Compare both directories |
| Sensitivity analysis | Run analyses on both, compare results |

---

## 11. Technical Specifications

### 11.1 Software Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.8+ | Runtime |
| pandas | ≥1.3.0 | Data manipulation |
| numpy | ≥1.20.0 | Numerical operations |
| scikit-learn | ≥0.24.0 | PCA, StandardScaler |
| matplotlib | ≥3.4.0 | Visualization |
| duckdb | ≥0.8.0 | S3 data loading |

### 11.2 Computational Parameters

| Parameter | Value |
|-----------|-------|
| PCA Components | 2 |
| Standardization | Z-score (mean=0, std=1) |
| Aggregation Resolution | 30-minute bins |
| Time Window | -180 to +540 minutes |
| Missing Data Handling | Listwise deletion |
| Outlier Filter (filtered only) | 1st-99th percentile |

### 11.3 Runtime Estimates

| Stage | Approximate Time |
|-------|-----------------|
| Data loading (S3) | 3-5 minutes |
| Processing | 30 seconds |
| PCA + Plotting | 2-3 minutes |
| **Total** | **5-10 minutes** |

---

## Appendix

### A.1 Color Reference

#### Time Window Colors (Hex Codes)
```
baseline:        #27ae60
pre_injection:   #2ecc71
immediate:       #f39c12
peak_early:      #e74c3c
peak_sustained:  #c0392b
decline_early:   #9b59b6
decline_late:    #3498db
post_6hr+:       #1abc9c
```

#### Dose Colors (Hex Codes)
```
Vehicle:   #95a5a6
5 mg/kg:   #3498db
25 mg/kg:  #e74c3c
```

#### Replicate × Dose Colors (Hex Codes)
```
Rep1_Vehicle:   #bdc3c7
Rep2_Vehicle:   #7f8c8d
Rep1_5 mg/kg:   #85c1e9
Rep2_5 mg/kg:   #2874a6
Rep1_25 mg/kg:  #f1948a
Rep2_25 mg/kg:  #922b21
```

#### Replicate Colors (Hex Codes)
```
Rep1:  #3498db
Rep2:  #e74c3c
```

### A.2 Quick Reference Card

| To Find... | Look At... |
|------------|------------|
| Dose separation | `combined/pca_by_dose.png` |
| Feature contributions | `combined/pca_loadings.png` |
| Replicate consistency | `combined/pca_by_replicate.png` |
| Replicate + dose combined | `combined/pca_replicate_dose.png` |
| Temporal dynamics per dose | `by_dose/[dose]/pca_time_window.png` |
| Feature gradients (all data) | `combined/pca_by_[feature].png` |
| Feature gradients (per dose) | `by_dose/[dose]/pca_[feature].png` |
| Cleaner patterns | `aggregated/` folder plots |
| Max drug effect | `peak_effect/` folder plots |
| Cleaner visualization | `pca_filtered/` directory |

### A.3 Glossary

| Term | Definition |
|------|------------|
| **PCA** | Principal Component Analysis - dimensionality reduction technique |
| **PC1/PC2** | First/second principal components (axes of maximum variance) |
| **Loading** | Correlation between original feature and principal component |
| **Variance Explained** | Percentage of total data variance captured by a PC |
| **Standardization** | Z-score transformation to mean=0, std=1 |
| **Time Window** | Categorized period relative to injection time |
| **Dose Group** | Treatment condition (Vehicle, 5 mg/kg, 25 mg/kg) |
| **Replicate** | Independent experimental batch (Rep1, Rep2) |
| **Aggregated** | Data summarized to animal × time_window level |
| **Filtered** | Outliers removed using 1st-99th percentile cutoff |

---

*Document Version: 1.1*
*Updated to include outlier filtering and additional plot types*