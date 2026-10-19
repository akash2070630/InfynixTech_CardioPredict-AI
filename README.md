# CardioPredict AI — Clinical Intelligence & Risk Classification

> **Tagline:** Understand Risk. Explore Data. Support Better Decisions.  
> **Category:** Clinical Intelligence & Medical Disease Prediction  
> **Focus:** Educational Heart Disease Classification from Structured Clinical Data  
> **Architecture:** Scikit-Learn Leakage-Safe Pipelines + Streamlit & Interactive Web UI

---

## 1. Abstract & Problem Statement

Coronary Artery Disease (CAD) remains the primary cause of global cardiovascular mortality. Early identification of symptomatic individuals facilitates timely clinical triage, prioritized cardiac catheterization, and lifestyle modification before myocardial infarction occurs.

**CardioPredict AI** is a rigorous, research-inspired clinical machine learning framework that evaluates demographic, symptom, and hemodynamic markers from the canonical Cleveland Clinic Foundation cohort. Engineered to clinical data science standards, this platform features:

- **Strict Data Integrity:** Preprocessing is encapsulated in `Pipeline` and `ColumnTransformer` constructs, ensuring zero training-to-test data leakage.
- **Model Tournament:** 4 distinct classification models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) are systematically cross-validated and tested on held-out samples.
- **Recall-Centric Optimization:** Model selection explicitly minimizes false negatives, recognizing that an undetected cardiac patient presents far higher clinical risk than an incidental false positive.
- **Clinical Interpretability:** Transparent feature attributions, interactive What-If sensitivity curves, and granular false-negative diagnostic error audits.

---

## 2. Proposed Solution & Project Objectives

1. **Leakage-Safe Modeling:** Build a standardized preprocessing pipeline where numerical features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) are scaled using `StandardScaler` and categorical features (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`) are encoded via `OneHotEncoder` strictly within training splits.
2. **Objective Clinical Selection:** Rather than blindly selecting models by raw accuracy, prioritize clinical sensitivity (Recall) and discrimination capacity (ROC-AUC) using the composite clinical score:
   $$\text{Score} = 0.50 \times \text{Recall} + 0.30 \times \text{ROC-AUC} + 0.20 \times \text{F1}$$
3. **Interactive Decision Support Interface:** Deliver an interactive clinical dashboard featuring an Executive Overview, Data Explorer, Prediction Studio, Model Intelligence Tournament, and Diagnostic Error Lab.
4. **Reproducible Engineering Standards:** Deliver modular code structure, unit test suite (`pytest`), automated dataset ingestion, and serialized model artifacts.

---

## 3. System Architecture & Workflow

```text
       Raw UCI Heart Disease Dataset (Cleveland Clinic, 303 cases)
                                  │
                                  ▼
      Data Cleaning, Median Imputation, Target Binarization (0/1)
                                  │
                                  ▼
                  Stratified 80/20 Train-Test Split
                                  │
                                  ▼
           Leakage-Safe Preprocessor (Scikit-Learn Pipeline)
             ├── Continuous: StandardScaler (age, bp, chol, thalach, oldpeak)
             └── Categorical: OneHotEncoder (sex, cp, fbs, ecg, exang, slope, ca, thal)
                                  │
                                  ▼
                Stratified 5-Fold Cross-Validation Tournament
       [Logistic Regression]  [Decision Tree]  [Random Forest]  [Gradient Boosting]
                                  │
                                  ▼
             Clinical Composite Scoring: (0.50*Rec + 0.30*AUC + 0.20*F1)
                                  │
                                  ▼
                Model & Metadata Serialization (`models/`, `outputs/`)
                                  │
                                  ▼
                  Interactive Streamlit Application (`app.py`)
   Executive Overview ── Clinical Data Explorer ── Prediction Studio ── Model Lab
```

---

## 4. Dataset Source, Feature Descriptions & Preprocessing

The project utilizes the verified **Cleveland Heart Disease** dataset from the UCI Machine Learning Repository (Detrano et al., 1989).

### Feature Dictionary

| Feature | Variable | Type | Clinical Description |
| :--- | :--- | :--- | :--- |
| **Age** | `age` | Continuous | Age in years (range: 29–77). |
| **Sex** | `sex` | Binary | Biological sex (0 = Female, 1 = Male). |
| **Chest Pain** | `cp` | Categorical | 1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal, 4: Asymptomatic. |
| **Resting BP** | `trestbps` | Continuous | Resting blood pressure at hospital intake (mm Hg). |
| **Cholesterol** | `chol` | Continuous | Serum total cholesterol (mg/dl). |
| **Fasting Blood Sugar**| `fbs` | Binary | Fasting blood sugar > 120 mg/dl (0 = False, 1 = True). |
| **Resting ECG** | `restecg` | Categorical | 0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy. |
| **Max Heart Rate** | `thalach` | Continuous | Maximum heart rate achieved during Bruce treadmill exercise. |
| **Exercise Angina** | `exang` | Binary | Exercise-induced angina (0 = No, 1 = Yes). |
| **ST Depression** | `oldpeak` | Continuous | Exercise-induced ST depression relative to rest (mm). |
| **ST Slope** | `slope` | Categorical | Slope of the peak exercise ST segment (1: Up, 2: Flat, 3: Down). |
| **Major Vessels** | `ca` | Ordinal | Number of major vessels (0–3) colored by fluoroscopy. |
| **Thallium Perfusion**| `thal` | Categorical | 3.0: Normal, 6.0: Fixed defect, 7.0: Reversible defect. |
| **Target** | `target` | Binary | 0 = No significant angiographic CAD (< 50% stenosis), 1 = CAD present. |

### Preprocessing & Data Cleaning
- **Target Binarization:** In the original Cleveland clinical study, targets 1 through 4 represent increasing stenosis severity (>50% vessel narrowing). For clinical triage, these are binarized: 0 (Absence) vs 1 (Presence).
- **Missing Value Handling:** Missing tokens (`?`) in fluoroscopy vessels (`ca`) and thallium scintigraphy (`thal`) are imputed using the cohort median to preserve sample volume without introducing data distribution skew.
- **Pipeline Segregation:** Scaling statistics ($\mu, \sigma$) and one-hot categorical matrices are calculated strictly on training folds to prevent diagnostic leakage into test sets.

---

## 5. Model Methodology & Evaluation Results

Evaluated across a stratified 20% holdout split ($N=61$ patients) strictly segregated during pipeline training:

| Model Architecture | Cross-Val Accuracy | Held-Out Accuracy | Sensitivity (Recall) | Precision | F1-Score | Held-Out ROC-AUC | Composite Clinical Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)** | **82.6%** | **85.2%** | **89.3%** | **80.6%** | **0.847** | **0.912** | **0.890** |
| **Random Forest Classifier** | 81.8% | 83.6% | 85.7% | 80.0% | 0.828 | 0.898 | 0.864 |
| **Gradient Boosting Classifier** | 79.7% | 80.3% | 82.1% | 76.7% | 0.793 | 0.884 | 0.835 |
| **Decision Tree Classifier** | 76.4% | 75.4% | 78.6% | 71.0% | 0.746 | 0.781 | 0.777 |

### Why Logistic Regression was Selected
While non-linear tree ensembles can capture complex feature interactions, with $N=303$ clinical cases and 13 biomarkers, complex ensembles exhibit higher variance and risk overfitting. Regularized Logistic Regression achieved the highest sensitivity (**89.3%**) and ROC-AUC (**0.912**), minimizing catastrophic false negatives while providing clear, monotonic odds-ratio interpretability.

---

## 6. Installation and Usage Instructions

Follow these terminal commands to configure and execute the project locally.

### Step 1: Clone or Navigate to the Directory
```bash
cd cardiopredict-ai
```

### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv .venv
```
- On Linux / macOS:
  ```bash
  source .venv/bin/activate
  ```
- On Windows (Command Prompt):
  ```cmd
  .venv\Scripts\activate.bat
  ```
- On Windows (PowerShell):
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Execute the Training & Tournament Engine
```bash
python train_model.py
```
*This downloads the Cleveland dataset, runs stratified 5-fold cross-validation on 4 architectures, evaluates held-out splits, exports serialized model artifacts to `models/`, and generates reports in `outputs/`.*

### Step 5: Run Automated Unit & Pipeline Tests
```bash
pytest tests/ -v
```

### Step 6: Launch the Streamlit Clinical Application
```bash
streamlit run app.py
```

---

## 7. Screenshots Checklist for Evaluation Submission

Ensure you capture the following views for your project submission portfolio:

1. [x] **Executive Overview:** Hero section with clinical value pillars and cohort summary KPI cards.
2. [x] **Clinical Data Explorer:** Interactive continuous biomarker histograms and Pearson correlation matrix.
3. [x] **Prediction Studio:** Form input fields, dynamic probability gauge, and risk categorization verdict.
4. [x] **What-If Explorer:** Dynamic sensitivity curve demonstrating predicted probability shifts as ST depression or maximum heart rate varies.
5. [x] **Model Intelligence Tournament:** Comparative leaderboard and multi-metric bar chart.
6. [x] **Evaluation Lab:** Held-out confusion matrix and detailed false-negative misclassification audit table.
7. [x] **Exportable Report:** Verified JSON evaluation download payload.

---

## 8. Limitations & Ethical Considerations

- **Historical Demographics:** The Cleveland cohort was collected in the late 1980s. Sociodemographic characteristics, baseline lifestyle factors, and biomarker measurement standards have evolved since then.
- **Observational Correlation vs Clinical Causality:** Identified high-importance factors (`ca`, `oldpeak`, `thal`) reflect statistical correlation with angiographic stenosis, not direct causal interventions.
- **Non-Diagnostic Scope:** This software is engineered strictly for educational research and decision-support simulation. It is never a substitute for formal clinical diagnosis by a licensed cardiologist.

---

## 9. Conclusion & Future Improvements

**CardioPredict AI** proves that medical machine learning systems must prioritize clinical utility over superficial test metrics. By combining leakage-safe pipelines, recall-centric model tournaments, and transparent What-If sensitivity simulation, the platform demonstrates both technical rigor and thoughtful healthcare data science.

### Future Roadmap
1. **Multi-Cohort Validation:** Incorporating the Hungarian, Swiss, and Long Beach cohorts from UCI to evaluate geographical model drift.
2. **Probability Calibration:** Implementing Isotonic Regression or Platt scaling to calibrate raw model scores into true empirical risk probabilities.
3. **EHR Interoperability:** Wrapping the prediction pipeline into a HL7/FHIR-compliant REST API for integration into clinical trial workflows.


