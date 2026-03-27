# Automated Brugada Syndrome Detection from 12-Lead ECG: 
## Machine Learning Classification Report

---

## 1. Problem Statement

Brugada syndrome is a rare but life-threatening inherited cardiac disorder characterized by a distinctive coved-type ST-segment elevation in right precordial leads (V1–V3), often accompanied by a right bundle branch block pattern. The condition carries significant risk of sudden cardiac death, particularly in asymptomatic individuals. 

**Clinical Challenge:** Traditional Brugada diagnosis relies on manual visual inspection of ECG patterns by trained cardiologists, which is:
- **Subjective** — Inter-observer variability affects diagnosis
- **Time-consuming** — Delays in identifying high-risk patients
- **Geographically limited** — Few specialists available in resource-constrained settings

**Project Objective:** Develop an automated machine learning classifier to detect Brugada syndrome from 12-lead ECG recordings with sufficient sensitivity to support clinical decision-making. The goal is to achieve a model that can:
1. Accurately identify Brugada patients (high sensitivity/recall)
2. Minimize false negatives (missed diagnoses are clinically dangerous)
3. Provide interpretable features for clinical validation

**Classification Task:** Binary classification problem:
- **Class 0 (Negative):** Healthy individuals or non-Brugada patterns
- **Class 1 (Positive):** Confirmed Brugada Syndrome diagnosis

---

## 2. Dataset Justification

### Data Source
We utilized the **PhysioNet Brugada Syndrome ECG Dataset**, a publicly available, ethically approved dataset of 363 ECG recordings from individuals with suspected Brugada syndrome. After filtering for binary classification (excluding ambiguous cases, n=7), our final dataset comprised **356 samples**.

### Dataset Characteristics

| Metric | Value |
|--------|-------|
| Total Samples | 356 |
| Brugada Positive (Class 1) | 94 (26.4%) |
| Healthy/Negative (Class 0) | 262 (73.6%) |
| Sampling Frequency | 100 Hz |
| Recording Duration | 12 seconds |
| ECG Leads | 12 standard leads |
| Total samples per lead | 1,200 |

### Imbalanced Class Distribution
The dataset exhibits moderate class imbalance (26.4% positive vs. 73.6% negative). This mimics real-world clinical scenarios where Brugada syndrome is relatively rare. To address this during model training, we employed:
- Stratified train-test splits to maintain class proportions
- Class-weighted loss functions (class_weight={0: 1, 1: 3})
- 5-fold stratified cross-validation for robust evaluation

### Data Quality & Justification
- **Standardization:** All ECGs recorded at consistent 100 Hz sampling
- **Clinical Validity:** Curated by PhysioNet using rigorous standards
- **Sufficient Size:** 356 samples adequate for feature engineering with dimensionality ~126
- **Real-world Distribution:** Imbalanced distribution reflects actual clinical prevalence

---

## 3. ML Methodology

### Algorithm Selection: Random Forest Classifier

We selected **Random Forest** as our primary classifier for the following reasons:
1. **Robustness to imbalanced data** — Class weighting naturally supported
2. **Non-linear relationships** — ECG patterns are complex and non-linear
3. **Feature importance** — Interpretable feature rankings for clinical insights
4. **No scaling required** — Tree-based, invariant to feature magnitude
5. **Regularization built-in** — Multiple hyperparameters (depth, splits) to prevent overfitting

### Model Configuration

```
RandomForestClassifier(
    n_estimators      = 300        # trees
    max_depth         = 8          # prevents overfitting
    class_weight      = {0:1, 1:3} # penalizes false negatives 3x
    random_state      = 42         # reproducibility
)
```

**Hyperparameter Justification:**
- **300 trees:** Sufficient for ensemble stability; balance between performance & computational cost
- **max_depth=8:** Limits individual tree depth to reduce overfitting on 126 features
- **class_weight={0:1, 1:3}:** Prioritizes detection of Brugada cases (higher penalty for false negatives)

### Evaluation Strategy
- **Train-Test Split:** 80-20 stratified split to preserve class distribution
- **Primary Validation:** 5-fold stratified cross-validation on full dataset
- **Metrics:** ROC-AUC, F1-score, Precision, Recall, Confusion Matrix

---

## 4. Feature Engineering & Preprocessing

### Signal Processing Pipeline

**Input:** 12-lead ECG signals (12 × 1,200 samples per patient)

**Feature Extraction Strategy:** We engineered 126 discriminative features across three domains:

#### **A. Time-Domain Features (72 features)**
Per lead (12 leads × 6 features each):
- Mean value
- Standard deviation
- Minimum value
- Maximum value
- Median value
- First-order difference (delta) standard deviation

*Rationale:* Captures amplitude characteristics and signal variability; ST-segment elevation manifests as changes in mean and range.

#### **B. Frequency-Domain Features (36 features)**
Per lead (12 leads × 3 features each):
- FFT magnitude mean
- FFT magnitude standard deviation
- FFT magnitude maximum

*Rationale:* Brugada patterns exhibit distinctive spectral signatures; frequency-domain analysis reveals rhythmic abnormalities invisible in time domain.

#### **C. Energy Features (18 features)**
Per lead (12 leads × 1 + special):
- Signal energy (sum of squared amplitudes)
- Special focus: V1–V3 leads (indices 6, 7, 8)
  - Mean absolute value
  - Maximum absolute value

*Rationale:* Energy captures signal intensity. V1–V3 leads are diagnostic for Brugada (coved ST-elevation front); specialized features target this region.

### Feature Statistics
- **Total features:** 126
- **Feature dimensionality:** 12 leads × 10-11 features per lead
- **No scaling applied:** Tree-based models are scale-invariant
- **No feature selection:** All 126 features retained; Random Forest handles feature importance internally

---

## 5. Results

### Cross-Validation Performance (5-Fold Stratified)

| Metric | Score |
|--------|-------|
| **ROC-AUC** | 0.702 ± 0.089 |
| **F1-Score** | 0.176 ± 0.062 |
| **Precision** | 0.179 ± 0.076 |
| **Recall** | 0.102 ± 0.034 |

### Interpretation of Results

**Strengths:**
- **Moderate ROC-AUC (0.702):** Better than random guessing (0.50) and approaching clinical utility threshold
- **Cross-validation consistency:** Standard deviations indicate stable generalization across folds

**Limitations:**
- **Low Recall (10.2%):** Only detects ~1 in 10 true Brugada cases; **clinically insufficient** for deployment
- **Low F1-Score (0.176):** Poor balance between precision and recall
- **High False Negative Rate:** Misses 89.8% of positive cases — unacceptable risk given sudden cardiac death risk

### Confusion Matrix (Aggregate)
```
                Predicted
                Negative  Positive
Actual Negative    240        22
       Positive     84        10
```
- **True Negatives:** 240/262 = 91.6% specificity
- **True Positives:** 10/94 = 10.6% sensitivity (recall)

---

## 6. Interpretability & Feature Analysis

### Top Contributing Features

Random Forest feature importance revealed:
1. **V1–V3 mean values** (leads 6-8) — V1–V3 ST-segment elevation is primary diagnostic marker
2. **Time-domain variability** (standard deviations) — Distinguishes pathological from normal patterns
3. **Energy metrics** — Signal intensity differs in Brugada syndrome

### Clinical Alignment
The automated feature importance aligns with clinical diagnostic criteria:
- **V1–V3 emphasis:** Medical literature emphasizes right precordial leads for Brugada detection
- **Spectral characteristics:** FFT features capture conduction abnormalities reflected in frequency spectrum
- **Signal variability:** Reduced variability in Brugada records correlates with abnormal conduction

### Model Transparency
- Model predictions provide probability scores (0–1 confidence)
- Feature importance rankings interpretable by cardiologists
- Decision trees within the ensemble can be visualized for individual cases

---

## 7. Clinical Impact & Recommendations

### Current Limitations for Clinical Deployment
1. **Sensitivity too low (10–12%):** Cannot be used as primary diagnostic tool
2. **High false negative rate:** Misses majority of Brugada cases; unacceptable for life-threatening condition
3. **Requires validation cohort:** Clinical validation on prospective data needed before use

### Potential Clinical Applications (With Improvements)
- **Screening tool** (ensemble with human review, not standalone)
- **Flagging high-risk cases** for specialist review
- **Resource allocation** in telemetry-limited settings
- **Quality control** in ECG laboratories (catch obviously normal records)

### Recommendations for Model Improvement

**1. Feature Enhancement:**
- Incorporate phase information (not just magnitude) from Wavelet transforms
- Extract ST-segment slope features (specific to Brugada pattern)
- Add temporal dynamics (R-R intervals, T-wave characteristics)

**2. Data Augmentation:**
- Increase dataset size (current 356 samples is limiting)
- Include noise-augmented ECG variations
- Balance classes or use weighted resampling techniques

**3. Advanced Architectures:**
- **Deep Learning (CNN/LSTM):** 1D convolutional networks for end-to-end ECG-to-prediction
- **Ensemble Methods:** Combine Random Forest with Gradient Boosting (XGBoost) or neural networks
- **Attention Mechanisms:** Learn which time-segments and leads matter most

**4. Clinical Validation:**
- Prospective validation on independent cohort
- Cardiologist reader agreement study
- Assessment of clinical utility vs. standard visual analysis

**5. Bias & Fairness:**
- Evaluate performance across demographic groups (age, sex, ethnicity)
- Ensure model generalizes beyond PhysioNet dataset

### Conclusion

This study demonstrates the feasibility of automated ECG analysis for Brugada syndrome detection using engineered features and Random Forest classification. However, **the current model is not clinically deployable** due to insufficient sensitivity (10.2% recall). Further development incorporating deeper learning architectures, expanded feature engineering, and larger datasets is necessary to achieve clinically acceptable performance (>90% sensitivity) for this life-threatening condition. We recommend the model as a foundation for future research rather than clinical implementation.

---

**Project Metadata:**
- Model: Random Forest (300 trees, max_depth=8)
- Features: 126 engineered from time-, frequency-, and energy domains
- Dataset: 356 ECG records (PhysioNet Brugada Syndrome Cohort)
- Evaluation: 5-fold stratified cross-validation
- Cross-validation ROC-AUC: 0.702
