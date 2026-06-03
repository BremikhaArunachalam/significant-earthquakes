# Significant Earthquakes 1965–2016

A machine learning project to predict the magnitude of significant earthquakes worldwide based on seismic measurements, geographic coordinates, and event metadata recorded between 1965 and 2016.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Model Results](#model-results)
- [Author](#author)

---

## Project Overview

This project builds an end-to-end ML regression pipeline to predict the magnitude of significant seismic events using geographic and measurement-based features. The dataset contains 23,412 earthquake records sourced from the USGS Earthquake Catalog, covering events from 1965 to 2016 across the globe.

The target variable is `Magnitude` — a continuous value representing the seismic magnitude of the event (5.5 – 9.1). The pipeline covers the full ML workflow: data loading, missing value handling, categorical encoding, outlier treatment, feature scaling, feature selection, model training, and R² evaluation across 7 regression models.

---

## Repository Structure

```
significant-earthquakes-1965-2016/
├── Significant_Earthquakes_1965_2016.ipynb   # End-to-end ML pipeline notebook
└── database.csv                               # Dataset (23,412 seismic records)
```

---

## Dataset

**File:** `database.csv`  
**Source:** Kaggle — USGS Earthquake Catalog  
**Total Records:** 23,412  
**Period Covered:** 1965 – 2016  
**Target Variable:** `Magnitude` — seismic magnitude of the event (5.5 – 9.1)

| Feature | Type | Description |
|---|---|---|
| `Date` | Categorical | Date of the seismic event |
| `Time` | Categorical | Time of the seismic event (UTC) |
| `Latitude` | Float | Geographic latitude of the epicentre |
| `Longitude` | Float | Geographic longitude of the epicentre |
| `Type` | Categorical | Event type (Earthquake / Nuclear Explosion / Explosion / Rock Burst) |
| `Depth` | Float | Depth of the event in km (-1.1 – 700.0) |
| `Depth Error` | Float | Error margin in depth measurement |
| `Depth Seismic Stations` | Float | Number of seismic stations reporting depth |
| `Magnitude` | Float | Target — seismic magnitude (5.5 – 9.1) |
| `Magnitude Type` | Categorical | Scale used for magnitude (MW, ML, MS, MB, etc.) |
| `Magnitude Error` | Float | Error margin in magnitude measurement |
| `Magnitude Seismic Stations` | Float | Number of stations reporting magnitude |
| `Azimuthal Gap` | Float | Largest azimuthal gap between reporting stations |
| `Horizontal Distance` | Float | Distance from nearest station to epicentre |
| `Horizontal Error` | Float | Error in horizontal location |
| `Root Mean Square` | Float | RMS travel time residual |
| `ID` | Categorical | Unique event identifier |
| `Source` | Categorical | Data source agency |
| `Location Source` | Categorical | Agency responsible for location |
| `Magnitude Source` | Categorical | Agency responsible for magnitude |
| `Status` | Categorical | Review status (Automatic / Reviewed) |

### Summary Statistics

| Feature | Mean | Std Dev | Min | Max |
|---|---|---|---|---|
| Latitude | 1.68 | 30.11 | -77.08 | 86.01 |
| Longitude | 39.64 | 125.51 | -179.99 | 179.99 |
| Depth (km) | — | — | -1.1 | 700.0 |
| Magnitude | — | — | 5.5 | 9.1 |
| Root Mean Square | 1.02 | 0.19 | 0.00 | 3.44 |

### Event Type Distribution

| Type | Count |
|---|---|
| Earthquake | 23,232 |
| Nuclear Explosion | 175 |
| Explosion | 4 |
| Rock Burst | 1 |

---

## ML Pipeline

### 1. Data Loading and Exploration

- Loaded dataset using `pandas`
- Inspected shape, data types, and missing value counts
- Reviewed distribution of event types and geographic spread

### 2. Missing Value Handling

- **Numerical columns** — imputed using mean via `SimpleImputer(strategy='mean')`
- **Categorical columns** — imputed using mode via `SimpleImputer(strategy='most_frequent')`

### 3. Categorical Encoding

- Applied **One-Hot Encoding** on multi-class columns (`Type`, `Magnitude Type`, `Source`, `Location Source`, `Magnitude Source`, `Status`) using `pd.get_dummies()` with `drop_first=True` to avoid multicollinearity
- Columns `Date`, `Time`, and `ID` were dropped before model training as they carry no predictive signal

### 4. Outlier Treatment

- Used **IQR (Interquartile Range)** method for all numerical columns
- Capped values below `Q1 - 1.5 * IQR` to the lower bound
- Capped values above `Q3 + 1.5 * IQR` to the upper bound

### 5. Feature Selection

- Applied `SelectKBest` with `f_regression` and `k='all'` to rank all features by their correlation with `Magnitude`
- All statistically significant features were retained for model training

### 6. Feature Scaling

- Applied `StandardScaler` to normalize all feature values to zero mean and unit variance before model training

### 7. Train-Test Split

| Parameter | Value |
|---|---|
| Test size | 30% (7,024 records) |
| Train size | 70% (16,388 records) |
| Random state | 100 |

---

## Model Results

All 7 regression models were trained on the same preprocessed dataset and evaluated using **R² Score (Coefficient of Determination)**.

| # | Model | R² Score |
|---|---|---|
| 1 | **Gradient Boosting** | **0.1562** |
| 2 | Ridge Regression | 0.1344 |
| 3 | Linear Regression | 0.1342 |
| 4 | Support Vector Regression | 0.1091 |
| 5 | Random Forest | 0.0979 |
| 6 | K-Nearest Neighbors | 0.0814 |
| 7 | Decision Tree | -0.5965 |

**Best performing model: Gradient Boosting — R² Score of 0.1562**

> Note: The low R² scores across all models indicate that earthquake magnitude is inherently difficult to predict from the available features alone. Seismic magnitude is influenced by complex geological factors not fully captured in this dataset.

---

## Author

**Bremikha Arunachalam**  
GitHub: [github.com/BremikhaArunachalam](https://github.com/BremikhaArunachalam)
