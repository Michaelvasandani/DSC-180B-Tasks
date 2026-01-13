# Data Description - Morph2REP Study (Study 1001 Version 2025v3.3)

## Table of Contents

- [Overview](#overview)
- [Experimental Design](#experimental-design)
- [Cage and Animal Mappings](#cage-and-animal-mappings)
- [Event Timeline](#event-timeline)
- [Data Organization](#data-organization)
- [File Schemas](#file-schemas)
- [Data Access](#data-access)
- [Query Examples](#query-examples)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Known Issues](#known-issues)

## Overview

The **Morph2REP** (Morphine Treatment Two Replicates) study evaluated morphine's physiological and behavioral effects at two doses in female strain 664 mice. Conducted at JAX Envision East, this study utilized continuous automated monitoring to capture high-resolution behavioral and physiological data under controlled environmental conditions.

### Study Design

- **Primary Objective:** Evaluate dose-dependent effects of morphine on behavior and physiology
- **Treatment Groups:**
  - 5 mg/kg morphine (n=18 mice, 6 cages)
  - 25 mg/kg morphine (n=18 mice, 6 cages)
  - Vehicle control (n=18 mice, 6 cages)
- **Experimental Approach:** Two independent replicates for reproducibility
- **Data Collection Platform:** Envision 2025 v3.0 automated monitoring system
- **Location:** JAX Envision East
- **Animal Model:** Female strain 664 mice (n=54 total)

### Temporal Coverage

**Complete Morph2REP Study (Both Replicates Included):**
- **Study Period:** January 7 - February 4, 2025
- **Replicate 1:** January 7-22, 2025 (16 days, cage_id=4917-4925)
- **Replicate 2:** January 22 - February 4, 2025 (14 days, cage_id=4926-4934)
- **Light Cycle:** 6:00 AM - 6:00 PM EST (12h:12h light:dark)
- **Sampling Frequency:** Continuous automated monitoring with millisecond-level temporal resolution

### Spatial Coverage

**Complete Dataset (Both Replicates):**
- **Total Cages:** 18 (cage_id=4917 to cage_id=4934)
- **Total Animals:** 54 mice (Animal IDs: 523-1 through 523-54)
- **Animals per Cage:** 3 mice per cage
- **Replicate 1:**
  - Cages: 9 (cage_id=4917-4925, Database IDs 3307-3315)
  - Animals: 523-1 to 523-27
  - Dates: January 7-22, 2025
- **Replicate 2:**
  - Cages: 9 (cage_id=4926-4934, Database IDs 3316-3324)
  - Animals: 523-28 to 523-54
  - Dates: January 22 - February 4, 2025

## Experimental Design

### Treatment Schedule

Each replicate followed a standardized timeline:

1. **Acclimation Phase** (3 days): Mice adapt to monitoring environment
2. **Baseline Recording** (3 days): Pre-treatment behavioral and physiological baselines
3. **Treatment Phase**: Two morphine doses administered at specific circadian times
4. **Continuous Monitoring**: Uninterrupted data collection throughout all phases

### Dosing Times

Morphine administration was carefully synchronized with the light cycle:

**Replicate 1:**
- Dose #1: January 14, 2025 at 6:00 AM (start of light cycle)
- Dose #2: January 17, 2025 at 5:00 PM (end of light cycle, before dark phase)

**Replicate 2:**
- Dose #1: January 28, 2025 at 5:00 PM (end of light cycle)
- Dose #2: January 31, 2025 at 6:00 AM (start of light cycle)

### Environmental Control

- **Light Cycle:** 6:00 AM - 6:00 PM EST (lights on), 6:00 PM - 6:00 AM EST (lights off)
- **Cage Changes:** Standardized mid-experiment (January 15, 2025 at 12:00 PM for Rep 1; January 29, 2025 at 12:00 PM for Rep 2)
- **Enrichment:** Nesting material increased during cage changes

## Cage and Animal Mappings

The following tables provide complete mappings between cage identifiers, database IDs (DBIDs), animal IDs, and treatment assignments for both replicates.

### Replicate 1 Cage Assignments

| Dose Group | Cage Name | Cage DBID | Animal IDs | Cage ID (Data) |
|------------|-----------|-----------|------------|----------------|
| 5 mg/kg | Resp-1 (A1) | 3307 | 523-1, 523-2, 523-3 | 4917 |
| Vehicle | Resp-2 (B1) | 3308 | 523-4, 523-5, 523-6 | 4918 |
| 25 mg/kg | Resp-3 (C1) | 3309 | 523-7, 523-8, 523-9 | 4919 |
| 25 mg/kg | Resp-4 (A2) | 3310 | 523-10, 523-11, 523-12 | 4920 |
| 5 mg/kg | Resp-5 (B2) | 3311 | 523-13, 523-14, 523-15 | 4921 |
| Vehicle | Resp-6 (C2) | 3312 | 523-16, 523-17, 523-18 | 4922 |
| Vehicle | Resp-7 (A3) | 3313 | 523-19, 523-20, 523-21 | 4923 |
| 25 mg/kg | Resp-8 (B3) | 3314 | 523-22, 523-23, 523-24 | 4924 |
| 5 mg/kg | Resp-9 (C3) | 3315 | 523-25, 523-26, 523-27 | 4925 |


### Replicate 2 Cage Assignments

| Dose Group | Cage Name | Cage DBID | Animal IDs | Cage ID (Data) |
|------------|-----------|-----------|------------|----------------|
| 25 mg/kg | Resp-10 (A1) | 3316 | 523-28, 523-29, 523-30 | 4926 |
| 5 mg/kg | Resp-11 (B1) | 3317 | 523-31, 523-32, 523-33 | 4927 |
| Vehicle | Resp-12 (C1) | 3318 | 523-34, 523-35, 523-36 | 4928 |
| Vehicle | Resp-13 (A2) | 3319 | 523-37, 523-38, 523-39 | 4929 |
| 25 mg/kg | Resp-14 (B2) | 3320 | 523-40, 523-41, 523-42 | 4930 |
| 5 mg/kg | Resp-15 (C2) | 3321 | 523-43, 523-44, 523-45 | 4931 |
| 5 mg/kg | Resp-16 (A3â†’B3) | 3322 | 523-46, 523-47, 523-48 | 4932 |
| 25 mg/kg | Resp-17 (B3â†’A3) | 3323 | 523-49, 523-50, 523-51 | 4933 |
| Vehicle | Resp-18 (C3) | 3324 | 523-52, 523-53, 523-54 | 4934 |


## Event Timeline

All times are in Eastern Standard Time (EST). These event timestamps are critical for analyzing treatment effects and accounting for environmental perturbations.

### Replicate 1 Events (January 7-22, 2025)

| Event | Date | Time | Description |
|-------|------|------|-------------|
| Start Cohort 1 Baseline | Jan 7, 2025 | 6:00 AM | Initiated first replicate; acclimation begins |
| Baseline Recording Start | Jan 10, 2025 | 6:00 AM | Pre-treatment baseline period begins |
| Morphine Dose #1 | Jan 14, 2025 | 6:00 AM | First dose administered at light cycle onset |
| Cage Handling Event | Jan 14, 2025 | 6:42-6:46 AM | Resp-3 (C1) and Resp-4 (A2) removed/inserted |
| Cage Change & Enrichment | Jan 15, 2025 | 12:00 PM | All cages changed; nesting material increased |
| Morphine Dose #2 | Jan 17, 2025 | 5:00 PM | Second dose before dark cycle onset |

### Replicate 2 Events (January 22 - February 4, 2025)

| Event | Date | Time | Description |
|-------|------|------|-------------|
| Start Cohort 2 Baseline | Jan 20, 2025 | 6:00 AM | Initiated second replicate |
| Acclimation Begins | Jan 22, 2025 | 6:00 AM | Second cohort enters monitoring |
| Baseline Recording Start | Jan 25, 2025 | 6:00 AM | Pre-treatment baseline period begins |
| Morphine Dose #1 | Jan 28, 2025 | 5:00 PM | First dose before dark cycle onset |
| Cage Change & Enrichment | Jan 29, 2025 | 12:00 PM | All cages changed; nesting material increased |
| Morphine Dose #2 | Jan 31, 2025 | 6:00 AM | Second dose at light cycle onset |

### Important Notes on Event Timing

1. **Circadian Considerations:** Dose timing alternated between light cycle onset (6:00 AM) and offset (5:00 PM) to assess circadian interactions with morphine effects.

2. **Cage Handling Effects:** Brief cage removals on Jan 14 (6:42-6:46 AM) for cages Resp-3 and Resp-4 may cause transient behavioral artifacts.

3. **Cage Change Effects:** Mid-experiment cage changes (12:00 PM) represent environmental perturbations that should be considered in temporal analyses.

4. **Light Cycle Boundaries:** The 6:00 AM and 6:00 PM transitions represent major circadian transitions when baseline behavior naturally changes.

## Data Organization

The dataset is organized using Hive-style partitioning on S3:

```
s3://jax-envision-public-data/study_1001/2025v3.3/tabular/
â”œâ”€â”€ cage_id=4917/
â”‚   â”œâ”€â”€ date=2025-01-07/
â”‚   â”‚   â”œâ”€â”€ animal_activity_db.parquet
â”‚   â”‚   â”œâ”€â”€ animal_activity_features.parquet
â”‚   â”‚   â”œâ”€â”€ animal_activity_state_bboxes.parquet
â”‚   â”‚   â”œâ”€â”€ animal_aggs_long.parquet
â”‚   â”‚   â”œâ”€â”€ animal_aggs_long_id.parquet
â”‚   â”‚   â”œâ”€â”€ animal_aggs_short.parquet
â”‚   â”‚   â”œâ”€â”€ animal_aggs_short_id.parquet
â”‚   â”‚   â”œâ”€â”€ animal_detections.parquet
â”‚   â”‚   â”œâ”€â”€ animal_drinking.parquet
â”‚   â”‚   â”œâ”€â”€ animal_eartags.parquet
â”‚   â”‚   â”œâ”€â”€ animal_ids.parquet
â”‚   â”‚   â”œâ”€â”€ animal_tsdb_mvp.parquet
â”‚   â”‚   â”œâ”€â”€ cage_aggs_short.parquet
â”‚   â”‚   â”œâ”€â”€ cage_detections.parquet
â”‚   â”‚   â””â”€â”€ cage_food_level.parquet / cage_water_level.parquet
â”‚   â”œâ”€â”€ date=2025-01-08/
â”‚   â”‚   â””â”€â”€ ... (same structure)
â”‚   â””â”€â”€ ...
â”œâ”€â”€ cage_id=4918/
â”‚   â””â”€â”€ ...
â””â”€â”€ ...
```

## File Schemas

All files are in Apache Parquet format. Most files share common metadata columns: `organization_id`, `cage_id`, `study_id`, `device_id`, `run_id`, `version_str`, `ULID`, `filename`, `source_file`.

### animal_activity_db.parquet

**Description:** Activity database records with behavioral metrics (distance, speed, activity level)

**Columns:** 17 total columns

**Sample Size (cage 4917, date 2025-01-07):** 72,159 rows, 0.00 MiB

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `time` | timestamp[ns] | Timestamp |
| `animal_id` | int64 | Animal identifier |
| `predicted_identity` | string | Predicted animal identity (e.g., "A", "B", "C") |
| `name` | string | Metric name |
| `value` | double | Metric value |
| `units` | string | Units of measurement |
| `resolution` | int64 | Temporal resolution (seconds) |

**Note:** 10 additional columns available (including metadata and computed features).


### animal_activity_features.parquet

**Description:** 155 extracted activity features including displacement, velocity, acceleration, bounding box metrics, keypoint distances, Hu moments, and pose angles

**Columns:** 155 total columns

**Sample Size (cage 4917, date 2025-01-07):** 21,600 rows, 0.02 MiB

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `time` | timestamp[ns] | Timestamp |
| `predicted_identity` | string | Predicted animal identity (e.g., "A", "B", "C") |
| `total_displacement` | double | Distance moved (pixels) |
| `average_velocity` | double | Movement velocity (pixels/frame) |
| `max_velocity` | double | Movement velocity (pixels/frame) |
| `stationary_ratio` | double | Computed ratio metric |
| `predicted_state` | double | Activity state |

**Note:** 148 additional columns available (including metadata and computed features).


### animal_activity_state_bboxes.parquet

**Description:** Bounding boxes for activity state detections with frame-by-frame animal positions

**Columns:** 17 total columns

**Sample Size (cage 4917, date 2025-01-07):** 603,313 rows, 0.01 MiB

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `time` | timestamp[ns] | Timestamp |
| `frame` | int64 | Video frame number |
| `animal_id` | int64 | Animal identifier |
| `predicted_identity` | string | Predicted animal identity (e.g., "A", "B", "C") |
| `bb_left` | double | Bounding box coordinate |
| `bb_top` | double | Bounding box coordinate |
| `bb_width` | double | Bounding box coordinate |
| `bb_height` | double | Bounding box coordinate |
| `predicted_state` | int64 | Activity state |

**Note:** 8 additional columns available (including metadata and computed features).


### animal_aggs_long_id.parquet

**Description:** Long-interval aggregated metrics by animal (typical resolution: hours)

**Columns:** 15 total columns

**Sample Size (cage 4917, date 2025-01-07):** 108 rows, 0.00 MiB

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `time` | timestamp[us] | Timestamp |
| `animal_id` | int64 | Animal identifier |
| `name` | string | Metric name |
| `value` | double | Metric value |
| `units` | string | Units of measurement |
| `resolution` | int64 | Temporal resolution (seconds) |

**Note:** 9 additional columns available (including metadata and computed features).


### animal_aggs_short_id.parquet

**Description:** Short-interval aggregated metrics by animal (typical resolution: minutes)

**Columns:** 15 total columns

**Sample Size (cage 4917, date 2025-01-07):** 47,560 rows, 0.00 MiB

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `time` | timestamp[ns] | Timestamp |
| `animal_id` | int64 | Animal identifier |
| `name` | string | Metric name |
| `value` | double | Metric value |
| `units` | string | Units of measurement |
| `resolution` | int32 | Temporal resolution (seconds) |

**Note:** 9 additional columns available (including metadata and computed features).


### animal_bout_metrics.parquet

**Description:** Behavioral bout metrics with start/end times, state names, and computed statistics per bout

**Columns:** 14 total columns

**Sample Size (cage 4917, date 2025-01-07):** 11,412 rows, 0.00 MiB

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `start_time` | timestamp[ns] | Timestamp |
| `end_time` | timestamp[us] | Timestamp |
| `animal_id` | int64 | Animal identifier |
| `predicted_identity` | string | Predicted animal identity (e.g., "A", "B", "C") |
| `state_name` | string | Activity state |
| `bout_length_seconds` | int64 | Duration of behavioral bout (seconds) |
| `metric_name` | string | Name of computed metric |
| `metric_value` | double | Value of computed metric |

**Note:** 6 additional columns available (including metadata and computed features).


### animal_bouts.parquet

**Description:** Behavioral bout data with start/end times and bout durations by state

**Columns:** 11 total columns

**Sample Size (cage 4917, date 2025-01-07):** 12,685 rows, 0.00 MiB

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `start_time` | timestamp[ns] | Timestamp |
| `end_time` | timestamp[us] | Timestamp |
| `animal_id` | int64 | Animal identifier |
| `predicted_identity` | string | Predicted animal identity (e.g., "A", "B", "C") |
| `state_name` | string | Activity state |
| `bout_length_seconds` | int64 | Duration of behavioral bout (seconds) |


### animal_drinking.parquet

**Description:** Drinking behavior events with detected drinking episodes

**Columns:** 17 total columns

**Sample Size (cage 4917, date 2025-01-07):** 48,240 rows, 0.00 MiB

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `time` | timestamp[us] | Timestamp |
| `animal_id` | int64 | Animal identifier |
| `predicted_identity` | string | Predicted animal identity (e.g., "A", "B", "C") |
| `name` | string | Metric name |
| `value` | double | Metric value |
| `units` | string | Units of measurement |
| `resolution` | int64 | Temporal resolution (seconds) |

**Note:** 10 additional columns available (including metadata and computed features).


### animal_eartags.parquet

**Description:** Eartag detection data with keypoints, bounding boxes, and classification confidence scores

**Columns:** 38 total columns

**Sample Size (cage 4917, date 2025-01-07):** 628,555 rows, 0.03 MiB

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `frame` | int64 | Video frame number |
| `time_stamp` | timestamp[us] | Timestamp |
| `animal_id` | double | Animal identifier |
| `predicted_class` | string | Eartag classification |
| `bb_left` | double | Bounding box coordinate |
| `bb_top` | double | Bounding box coordinate |
| `kpt_1_x` | double | Keypoint coordinate |
| `kpt_1_y` | double | Keypoint coordinate |
| `confidence` | double | Detection confidence score |

**Note:** 28 additional columns available (including metadata and computed features).


### animal_ids.parquet

**Description:** Animal ID tracking data with keypoints and bounding boxes for identification

**Columns:** 26 total columns

**Sample Size (cage 4917, date 2025-01-07):** 620,669 rows, 0.02 MiB

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `frame` | int64 | Video frame number |
| `animal_id` | int64 | Animal identifier |
| `predicted_identity` | string | Predicted animal identity (e.g., "A", "B", "C") |
| `eartag_code` | string | Eartag classification |
| `bb_left` | double | Bounding box coordinate |
| `bb_top` | double | Bounding box coordinate |
| `kpt_1_x` | double | Keypoint coordinate |
| `kpt_1_y` | double | Keypoint coordinate |

**Note:** 18 additional columns available (including metadata and computed features).


### animal_respiration.parquet

**Description:** Respiration rate measurements per animal

**Columns:** 17 total columns

**Sample Size (cage 4917, date 2025-01-07):** 1,030 rows, 0.00 MiB

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `time` | timestamp[us] | Timestamp |
| `animal_id` | int64 | Animal identifier |
| `predicted_identity` | string | Predicted animal identity (e.g., "A", "B", "C") |
| `name` | string | Metric name |
| `value` | double | Metric value |
| `units` | string | Units of measurement |
| `resolution` | int64 | Temporal resolution (seconds) |

**Note:** 10 additional columns available (including metadata and computed features).


### animal_sociability_pairwise.parquet

**Description:** Pairwise sociability metrics between pairs of animals

**Columns:** 17 total columns

**Sample Size (cage 4917, date 2025-01-07):** 20,966 rows, 0.00 MiB

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `time` | timestamp[us] | Timestamp |
| `animal_id_a` | int64 | Animal pair identifier |
| `animal_id_b` | int64 | Animal pair identifier |
| `predicted_identity_a` | string | Animal pair identifier |
| `predicted_identity_b` | string | Animal pair identifier |
| `name` | string | Metric name |
| `value` | double | Metric value |
| `units` | string | Units of measurement |

**Note:** 9 additional columns available (including metadata and computed features).


### animal_tsdb_mvp.parquet

**Description:** Time-series database for animal-level metrics (motion vectors, respiration)

**Columns:** 15 total columns

**Sample Size (cage 4917, date 2025-01-07):** 238,520 rows, 0.00 MiB

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `time` | timestamp[us] | Timestamp |
| `animal_id` | int64 | Animal identifier |
| `name` | string | Metric name |
| `value` | double | Metric value |
| `units` | string | Units of measurement |
| `resolution` | int32 | Temporal resolution (seconds) |
| `count` | int64 | Sample count for aggregation |

**Note:** 8 additional columns available (including metadata and computed features).


### cage_aggs_long.parquet

**Description:** Long-interval cage-level aggregated metrics

**Columns:** 15 total columns

**Sample Size (cage 4917, date 2025-01-07):** 42 rows, 0.00 MiB

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `time` | timestamp[us] | Timestamp |
| `cage_id` | int64 |  |
| `name` | string | Metric name |
| `value` | double | Metric value |
| `units` | string | Units of measurement |
| `resolution` | int64 | Temporal resolution (seconds) |

**Note:** 9 additional columns available (including metadata and computed features).


### cage_aggs_short.parquet

**Description:** Short-interval cage-level aggregated metrics

**Columns:** 15 total columns

**Sample Size (cage 4917, date 2025-01-07):** 23,997 rows, 0.00 MiB

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `time` | timestamp[us] | Timestamp |
| `cage_id` | int64 |  |
| `name` | string | Metric name |
| `value` | double | Metric value |
| `units` | string | Units of measurement |
| `resolution` | int64 | Temporal resolution (seconds) |

**Note:** 9 additional columns available (including metadata and computed features).


### cage_detections.parquet

**Description:** Cage-level detection events with keypoints and bounding boxes

**Columns:** 25 total columns

**Sample Size (cage 4917, date 2025-01-07):** 627,862 rows, 0.02 MiB

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `time_stamp` | timestamp[us] | Timestamp |
| `frame_number` | int64 | Video frame number |
| `cage_id` | int64 |  |
| `animal_id` | double | Animal identifier |
| `bb_left` | double | Bounding box coordinate |
| `bb_top` | double | Bounding box coordinate |
| `kpt_1_x` | double | Keypoint coordinate |
| `kpt_1_y` | double | Keypoint coordinate |

**Note:** 17 additional columns available (including metadata and computed features).


### cage_food_level.parquet

**Description:** Food level monitoring over time

**Columns:** 14 total columns

**Sample Size (cage 4917, date 2025-01-07):** 120 rows, 0.00 MiB

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `time` | timestamp[us] | Timestamp |
| `cage_id` | int64 |  |
| `name` | string | Metric name |
| `value` | int64 | Metric value |
| `units` | string | Units of measurement |
| `resolution` | int64 | Temporal resolution (seconds) |

**Note:** 8 additional columns available (including metadata and computed features).


### cage_motion_vector.parquet

**Description:** Cage-level motion vector data

**Columns:** 15 total columns

**Sample Size (cage 4917, date 2025-01-07):** 7,917 rows, 0.00 MiB

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `time` | timestamp[ns] | Timestamp |
| `cage_id` | int64 |  |
| `name` | string | Metric name |
| `value` | double | Metric value |
| `units` | string | Units of measurement |
| `resolution` | int64 | Temporal resolution (seconds) |

**Note:** 9 additional columns available (including metadata and computed features).


### cage_tsdb_mvp.parquet

**Description:** Time-series database for cage-level metrics

**Columns:** 14 total columns

**Sample Size (cage 4917, date 2025-01-07):** 26 rows, 0.00 MiB

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `time` | timestamp[us] | Timestamp |
| `cage_id` | int64 |  |
| `name` | string | Metric name |
| `value` | double | Metric value |
| `units` | string | Units of measurement |
| `resolution` | int32 | Temporal resolution (seconds) |
| `count` | int64 | Sample count for aggregation |

**Note:** 7 additional columns available (including metadata and computed features).


### cage_water_level.parquet

**Description:** Water level monitoring over time

**Columns:** 14 total columns

**Sample Size (cage 4917, date 2025-01-07):** 360 rows, 0.00 MiB

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `time` | timestamp[ns] | Timestamp |
| `cage_id` | int64 |  |
| `name` | string | Metric name |
| `value` | double | Metric value |
| `units` | string | Units of measurement |
| `resolution` | int64 | Temporal resolution (seconds) |

**Note:** 8 additional columns available (including metadata and computed features).


## Common Schema Patterns

### Time-Series Data Files
Most time-series files follow this pattern:
- `time`: Timestamp (nanosecond or microsecond precision)
- `name`: Metric name (e.g., 'distance_traveled', 'water_consumed')
- `value`: Numeric measurement
- `units`: Unit of measurement (e.g., 'cm', 'ml', 'seconds')
- `resolution`: Temporal resolution in seconds

### Detection/Tracking Data Files
Detection and tracking files typically include:
- `frame` or `frame_number`: Video frame identifier
- `time` or `time_stamp`: Timestamp
- `bb_left`, `bb_top`, `bb_right`, `bb_bottom`, `bb_width`, `bb_height`: Bounding box coordinates
- `kpt_1_x`, `kpt_1_y`, ... `kpt_6_x`, `kpt_6_y`: Six keypoint coordinates per animal
- `predicted_identity` or `animal_id`: Animal identifier

### Aggregation Files
Aggregation files provide summarized metrics:
- **Short-interval** (`aggs_short`): Higher frequency aggregations (typically per minute)
- **Long-interval** (`aggs_long`): Lower frequency aggregations (typically per hour)
- Both include `resolution` field indicating aggregation window in seconds

## Keypoint Schema

Many files include 6 keypoints per animal detection:
- **kpt_1**: Nose
- **kpt_2**: Left ear
- **kpt_3**: Right ear
- **kpt_4**: Neck/shoulder
- **kpt_5**: Mid-body
- **kpt_6**: Tail base

Each keypoint has X and Y pixel coordinates in the camera frame.

## Data Access

### Prerequisites

- **[mouseclick](https://github.com/murine-org/mouseclick)** - Python library for querying behavioral neuroscience data (recommended)
- Python 3.8+
- ClickHouse Local (installed automatically via `mouseclick setup`)

### Installation

```bash
git clone https://github.com/murine-org/mouseclick.git
cd mouseclick
pip install -e .
mouseclick setup
```

This dataset is configured as the default public dataset in mouseclick and works out of the box. No AWS credentials required for public access.

## Query Examples

### Basic Queries with Mouseclick

#### View Activity Data for One Cage/Date

```python
from mouseclick import MouseClickClient

client = MouseClickClient()

# Query activity database for cage 4917 on Jan 7, 2025
df = client.query(
    cage="4917",
    date="2025-01-07",
    file="animal_activity_db.parquet",
    columns=["timestamp", "animal_id", "activity_type"]
)

print(df.head(100))
```

#### Query Drinking Behavior

```python
# Query drinking events
df = client.query(
    cage="4920",
    date="2025-01-10",
    file="animal_drinking.parquet",
    columns=["timestamp", "animal_id", "duration", "volume"]
)

# Analyze drinking patterns
daily_drinking = df.groupby('animal_id')['volume'].sum()
print(f"Total drinking per animal:\n{daily_drinking}")
```

#### View Respiration/Motion Data

```python
# Query motion vector and respiration data
df = client.query(
    cage="4925",
    date="2025-01-15",
    file="animal_tsdb_mvp.parquet",
    columns=["timestamp", "animal_id", "respiration_rate",
             "motion_vector_x", "motion_vector_y"]
)

# Calculate average respiration by animal
resp_avg = df.groupby('animal_id')['respiration_rate'].mean()
```

### Advanced Queries

#### Time-Series Analysis with Pandas Aggregation

```python
# Get activity features for analysis
df = client.query(
    cage="4917",
    date="2025-01-07",
    file="animal_activity_features.parquet"
)

# Aggregate to 1-hour bins using Pandas
df['timestamp'] = pd.to_datetime(df['timestamp'])
hourly = df.set_index('timestamp').resample('1H').agg({
    'feature_value': ['mean', 'std']
})
```

#### Multi-Cage Queries

Query drinking behavior across all cages for a specific date:

```python
# Query all cages using wildcard
df = client.query(
    cage="*",  # All cages
    date="2025-01-10",
    file="animal_drinking.parquet",
    columns=["timestamp", "animal_id", "duration", "volume"]
)

# Analyze across cages
cage_summary = df.groupby('cage_id').agg({
    'volume': ['sum', 'mean'],
    'duration': ['sum', 'count']
})
```

#### Filtered Queries with Pandas

```python
# Get activity data
df = client.query(
    cage="4920",
    date="2025-01-12",
    file="animal_activity_db.parquet"
)

# Filter for high-activity periods
high_activity = df[df['activity_score'] > threshold]

# Aggregate by 10-minute bins
df['timestamp'] = pd.to_datetime(df['timestamp'])
activity_10min = df.set_index('timestamp').resample('10T')['activity_score'].max()
```

#### Cross-Date Analysis

Query activity across all dates for one cage:

```python
# Query all dates using wildcard
df = client.query(
    cage="4930",
    date="*",  # All available dates
    file="animal_activity_features.parquet"
)

# Daily averages
df['timestamp'] = pd.to_datetime(df['timestamp'])
daily = df.groupby([df['timestamp'].dt.date, 'animal_id']).mean()
```

### Interactive Explorer

Launch the rich terminal UI for interactive exploration:

```bash
mouseclick-explore
```

### Command Line Interface

```bash
# Basic query
mouseclick query --cage 4917 --date 2025-01-08 \
  --file cage_motion_vector.parquet

# Query with output to file
mouseclick query --cage 4920 --date 2025-01-10 \
  --file animal_drinking.parquet --output results.csv
```

### Programmatic Access

#### Python Example with PyArrow

```python
import pyarrow.parquet as pq
import s3fs

# Initialize S3 filesystem
s3 = s3fs.S3FileSystem(anon=True)  # anon=True for public access

# Read a single parquet file
table = pq.read_table(
    's3://jax-envision-public-data/study_1001/2025v3.3/tabular/'
    'cage_id=4917/date=2025-01-07/animal_drinking.parquet',
    filesystem=s3
)

# Convert to pandas DataFrame
df = table.to_pandas()
print(df.head())
```

#### Python Example with DuckDB

```python
import duckdb

# Connect to DuckDB and query S3 directly
con = duckdb.connect()

# Configure S3 access (for public bucket)
con.execute("SET s3_region='us-east-1';")
con.execute("SET s3_access_key_id='';")  # Empty for public access

# Query parquet files
query = """
SELECT timestamp, animal_id, duration, volume
FROM read_parquet('s3://jax-envision-public-data/study_1001/2025v3.3/tabular/*/date=2025-01-10/animal_drinking.parquet')
WHERE duration > 5.0
ORDER BY timestamp
LIMIT 100;
"""

result = con.execute(query).fetchdf()
print(result)
```

## Contact

For technical questions about the data:

- **Primary Contact:** Timothy L. Robertson, Ph.D.
- **Title:** Leader, Digital In Vivo Initiative
- **Email:** timothy.robertson@jax.org
- **Organization:** The Jackson Laboratory, Bar Harbor, ME

## Acknowledgments

**Funding:** This work was supported by JAX Mice Clinical and Research Services (JMCRS) R&D.

**Collaborators:** The Envision Team and The Digital In Vivo Alliance.

**Data Collection Platform:** Envision 2025 v3.0 automated monitoring system at JAX Envision East.