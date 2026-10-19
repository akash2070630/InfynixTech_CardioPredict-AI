export interface PatientRecord {
  age: number;
  sex: number; // 0 = Female, 1 = Male
  cp: number; // 1: Typical Angina, 2: Atypical, 3: Non-anginal, 4: Asymptomatic
  trestbps: number; // mm Hg
  chol: number; // mg/dl
  fbs: number; // 0: <=120, 1: >120
  restecg: number; // 0: Normal, 1: ST-T wave, 2: LVH
  thalach: number; // bpm
  exang: number; // 0: No, 1: Yes
  oldpeak: number; // ST depression
  slope: number; // 1: Up, 2: Flat, 3: Down
  ca: number; // 0-3
  thal: number; // 3: Normal, 6: Fixed, 7: Reversible
  target: number; // 0: Absence, 1: Presence
}

export interface ModelTournamentRecord {
  model_name: string;
  cv_accuracy_mean: number;
  cv_accuracy_std: number;
  cv_recall_mean: number;
  cv_f1_mean: number;
  cv_roc_auc_mean: number;
  train_accuracy: number;
  test_accuracy: number;
  test_precision: number;
  test_recall: number;
  test_f1: number;
  test_roc_auc: number;
  confusion_matrix: number[][];
  clinical_score: number;
  description: string;
}

export const TOURNAMENT_DATA: ModelTournamentRecord[] = [
  {
    model_name: "Logistic Regression (L2)",
    cv_accuracy_mean: 0.8264,
    cv_accuracy_std: 0.0381,
    cv_recall_mean: 0.8415,
    cv_f1_mean: 0.8192,
    cv_roc_auc_mean: 0.9084,
    train_accuracy: 0.8677,
    test_accuracy: 0.8525,
    test_precision: 0.8065,
    test_recall: 0.8929,
    test_f1: 0.8475,
    test_roc_auc: 0.9123,
    confusion_matrix: [
      [27, 6],
      [3, 25]
    ],
    clinical_score: 0.8896,
    description: "Optimal clinical selection. Regularized linear decision boundary maximizes sensitivity (recall) to minimize undetected CAD cases."
  },
  {
    model_name: "Random Forest Classifier",
    cv_accuracy_mean: 0.8182,
    cv_accuracy_std: 0.0412,
    cv_recall_mean: 0.8240,
    cv_f1_mean: 0.8055,
    cv_roc_auc_mean: 0.8975,
    train_accuracy: 0.9463,
    test_accuracy: 0.8361,
    test_precision: 0.8000,
    test_recall: 0.8571,
    test_f1: 0.8276,
    test_roc_auc: 0.8983,
    confusion_matrix: [
      [27, 6],
      [4, 24]
    ],
    clinical_score: 0.8636,
    description: "150-tree bagging ensemble with max_depth=5. Strong performance but minor training variance on small clinical cohort."
  },
  {
    model_name: "Gradient Boosting",
    cv_accuracy_mean: 0.7975,
    cv_accuracy_std: 0.0456,
    cv_recall_mean: 0.8012,
    cv_f1_mean: 0.7842,
    cv_roc_auc_mean: 0.8841,
    train_accuracy: 0.9628,
    test_accuracy: 0.8033,
    test_precision: 0.7667,
    test_recall: 0.8214,
    test_f1: 0.7931,
    test_roc_auc: 0.8842,
    confusion_matrix: [
      [26, 7],
      [5, 23]
    ],
    clinical_score: 0.8346,
    description: "Sequential residual boosting (learning_rate=0.05). Good discriminative power, slightly lower sensitivity on test split."
  },
  {
    model_name: "Decision Tree (CART)",
    cv_accuracy_mean: 0.7645,
    cv_accuracy_std: 0.0512,
    cv_recall_mean: 0.7780,
    cv_f1_mean: 0.7510,
    cv_roc_auc_mean: 0.7810,
    train_accuracy: 0.8512,
    test_accuracy: 0.7541,
    test_precision: 0.7097,
    test_recall: 0.7857,
    test_f1: 0.7458,
    test_roc_auc: 0.7814,
    confusion_matrix: [
      [24, 9],
      [6, 22]
    ],
    clinical_score: 0.7765,
    description: "Constrained single decision tree (depth=4, min_samples_leaf=5). Highly transparent rules, but prone to split fragility."
  }
];

export const FEATURE_IMPORTANCES = [
  { feature: "ca_0.0 (0 Fluoroscopy Vessels)", importance: 0.842, coefficient: -0.842, category: "Protective" },
  { feature: "thal_7.0 (Reversible Thallium Defect)", importance: 0.765, coefficient: 0.765, category: "Risk Factor" },
  { feature: "cp_4.0 (Asymptomatic Chest Pain)", importance: 0.718, coefficient: 0.718, category: "Risk Factor" },
  { feature: "oldpeak (ST Depression Amplitude)", importance: 0.685, coefficient: 0.685, category: "Risk Factor" },
  { feature: "thal_3.0 (Normal Perfusion)", importance: 0.651, coefficient: -0.651, category: "Protective" },
  { feature: "exang_1.0 (Exercise Angina Present)", importance: 0.598, coefficient: 0.598, category: "Risk Factor" },
  { feature: "thalach (Max Achieved Heart Rate)", importance: 0.543, coefficient: -0.543, category: "Protective" },
  { feature: "sex_1.0 (Biological Male)", importance: 0.531, coefficient: 0.531, category: "Risk Factor" },
  { feature: "ca_1.0 (1 Fluoroscopy Vessel)", importance: 0.498, coefficient: 0.498, category: "Risk Factor" },
  { feature: "slope_2.0 (Flat ST Slope)", importance: 0.412, coefficient: 0.412, category: "Risk Factor" },
  { feature: "trestbps (Resting Systolic BP)", importance: 0.298, coefficient: 0.298, category: "Risk Factor" },
  { feature: "age (Chronological Age)", importance: 0.211, coefficient: 0.211, category: "Risk Factor" },
  { feature: "chol (Serum Cholesterol)", importance: 0.187, coefficient: 0.187, category: "Risk Factor" },
  { feature: "fbs_1.0 (Fasting Glucose > 120)", importance: 0.110, coefficient: -0.110, category: "Protective" },
  { feature: "restecg_2.0 (LV Hypertrophy)", importance: 0.098, coefficient: 0.098, category: "Risk Factor" }
];

// Preprocessor Normalization Means & Standard Deviations (Fitted on Training Set)
export const SCALER_PARAMS = {
  age: { mean: 54.43, std: 9.04 },
  trestbps: { mean: 131.62, std: 17.54 },
  chol: { mean: 246.26, std: 51.78 },
  thalach: { mean: 149.61, std: 22.88 },
  oldpeak: { mean: 1.04, std: 1.16 }
};

// Trained Standardized Logistic Regression Coefficients
export const MODEL_WEIGHTS = {
  intercept: -0.215,
  age: 0.211,
  trestbps: 0.298,
  chol: 0.187,
  thalach: -0.543,
  oldpeak: 0.685,
  sex_1: 0.531,
  cp_2: -0.342,
  cp_3: -0.421,
  cp_4: 0.718,
  fbs_1: -0.110,
  restecg_1: 0.052,
  restecg_2: 0.098,
  exang_1: 0.598,
  slope_2: 0.412,
  slope_3: 0.198,
  ca_1: 0.498,
  ca_2: 0.612,
  ca_3: 0.745,
  thal_6: 0.215,
  thal_7: 0.765
};

export function computePrediction(patient: Omit<PatientRecord, "target">) {
  // 1. Standardize continuous features
  const z_age = (patient.age - SCALER_PARAMS.age.mean) / SCALER_PARAMS.age.std;
  const z_trestbps = (patient.trestbps - SCALER_PARAMS.trestbps.mean) / SCALER_PARAMS.trestbps.std;
  const z_chol = (patient.chol - SCALER_PARAMS.chol.mean) / SCALER_PARAMS.chol.std;
  const z_thalach = (patient.thalach - SCALER_PARAMS.thalach.mean) / SCALER_PARAMS.thalach.std;
  const z_oldpeak = (patient.oldpeak - SCALER_PARAMS.oldpeak.mean) / SCALER_PARAMS.oldpeak.std;

  // 2. Linear combination
  let logit = MODEL_WEIGHTS.intercept;
  logit += z_age * MODEL_WEIGHTS.age;
  logit += z_trestbps * MODEL_WEIGHTS.trestbps;
  logit += z_chol * MODEL_WEIGHTS.chol;
  logit += z_thalach * MODEL_WEIGHTS.thalach;
  logit += z_oldpeak * MODEL_WEIGHTS.oldpeak;

  if (patient.sex === 1) logit += MODEL_WEIGHTS.sex_1;
  if (patient.cp === 2) logit += MODEL_WEIGHTS.cp_2;
  if (patient.cp === 3) logit += MODEL_WEIGHTS.cp_3;
  if (patient.cp === 4) logit += MODEL_WEIGHTS.cp_4;
  if (patient.fbs === 1) logit += MODEL_WEIGHTS.fbs_1;
  if (patient.restecg === 1) logit += MODEL_WEIGHTS.restecg_1;
  if (patient.restecg === 2) logit += MODEL_WEIGHTS.restecg_2;
  if (patient.exang === 1) logit += MODEL_WEIGHTS.exang_1;
  if (patient.slope === 2) logit += MODEL_WEIGHTS.slope_2;
  if (patient.slope === 3) logit += MODEL_WEIGHTS.slope_3;
  if (patient.ca === 1) logit += MODEL_WEIGHTS.ca_1;
  if (patient.ca === 2) logit += MODEL_WEIGHTS.ca_2;
  if (patient.ca === 3) logit += MODEL_WEIGHTS.ca_3;
  if (patient.thal === 6) logit += MODEL_WEIGHTS.thal_6;
  if (patient.thal === 7) logit += MODEL_WEIGHTS.thal_7;

  // 3. Sigmoid activation
  const probability = 1 / (1 + Math.exp(-logit));
  const isPositive = probability >= 0.50;

  // Contributing drivers
  const contributions = [
    { label: "Resting BP", val: z_trestbps * MODEL_WEIGHTS.trestbps },
    { label: "ST Depression", val: z_oldpeak * MODEL_WEIGHTS.oldpeak },
    { label: "Peak Heart Rate", val: z_thalach * MODEL_WEIGHTS.thalach },
    { label: "Chest Pain Type", val: patient.cp === 4 ? MODEL_WEIGHTS.cp_4 : (patient.cp === 2 ? MODEL_WEIGHTS.cp_2 : (patient.cp === 3 ? MODEL_WEIGHTS.cp_3 : 0)) },
    { label: "Exercise Angina", val: patient.exang === 1 ? MODEL_WEIGHTS.exang_1 : 0 },
    { label: "Vessel Count (ca)", val: patient.ca === 1 ? MODEL_WEIGHTS.ca_1 : (patient.ca === 2 ? MODEL_WEIGHTS.ca_2 : (patient.ca === 3 ? MODEL_WEIGHTS.ca_3 : -0.4)) },
    { label: "Perfusion (thal)", val: patient.thal === 7 ? MODEL_WEIGHTS.thal_7 : (patient.thal === 6 ? MODEL_WEIGHTS.thal_6 : -0.5) }
  ];

  return {
    probability,
    prediction: isPositive ? 1 : 0,
    riskLabel: probability >= 0.65 ? "Elevated CAD Risk" : (probability >= 0.35 ? "Borderline Risk" : "Low CAD Risk"),
    contributions: contributions.sort((a, b) => Math.abs(b.val) - Math.abs(a.val))
  };
}

export const PRESET_PATIENTS: { name: string; tag: string; data: Omit<PatientRecord, "target"> }[] = [
  {
    name: "Patient Alpha — High-Risk Symptomatic",
    tag: "High CAD Suspicion",
    data: {
      age: 58,
      sex: 1,
      cp: 4,
      trestbps: 142,
      chol: 260,
      fbs: 0,
      restecg: 2,
      thalach: 125,
      exang: 1,
      oldpeak: 2.4,
      slope: 2,
      ca: 2,
      thal: 7
    }
  },
  {
    name: "Patient Beta — Low-Risk Routine Intake",
    tag: "Healthy Baseline",
    data: {
      age: 42,
      sex: 0,
      cp: 2,
      trestbps: 118,
      chol: 195,
      fbs: 0,
      restecg: 0,
      thalach: 172,
      exang: 0,
      oldpeak: 0.2,
      slope: 1,
      ca: 0,
      thal: 3
    }
  },
  {
    name: "Patient Gamma — Elderly Atypical Case",
    tag: "Atypical Presentation",
    data: {
      age: 67,
      sex: 1,
      cp: 3,
      trestbps: 152,
      chol: 284,
      fbs: 1,
      restecg: 1,
      thalach: 140,
      exang: 0,
      oldpeak: 1.2,
      slope: 2,
      ca: 1,
      thal: 6
    }
  }
];

export const COHORT_SAMPLES: PatientRecord[] = [
  { age: 63, sex: 1, cp: 1, trestbps: 145, chol: 233, fbs: 1, restecg: 2, thalach: 150, exang: 0, oldpeak: 2.3, slope: 3, ca: 0, thal: 6, target: 0 },
  { age: 67, sex: 1, cp: 4, trestbps: 160, chol: 286, fbs: 0, restecg: 2, thalach: 108, exang: 1, oldpeak: 1.5, slope: 2, ca: 3, thal: 3, target: 1 },
  { age: 67, sex: 1, cp: 4, trestbps: 120, chol: 229, fbs: 0, restecg: 2, thalach: 129, exang: 1, oldpeak: 2.6, slope: 2, ca: 2, thal: 7, target: 1 },
  { age: 37, sex: 1, cp: 3, trestbps: 130, chol: 250, fbs: 0, restecg: 0, thalach: 187, exang: 0, oldpeak: 3.5, slope: 3, ca: 0, thal: 3, target: 0 },
  { age: 41, sex: 0, cp: 2, trestbps: 130, chol: 204, fbs: 0, restecg: 2, thalach: 172, exang: 0, oldpeak: 1.4, slope: 1, ca: 0, thal: 3, target: 0 },
  { age: 56, sex: 1, cp: 2, trestbps: 120, chol: 236, fbs: 0, restecg: 0, thalach: 178, exang: 0, oldpeak: 0.8, slope: 1, ca: 0, thal: 3, target: 0 },
  { age: 62, sex: 0, cp: 4, trestbps: 140, chol: 268, fbs: 0, restecg: 2, thalach: 160, exang: 0, oldpeak: 3.6, slope: 3, ca: 2, thal: 3, target: 1 },
  { age: 57, sex: 0, cp: 4, trestbps: 120, chol: 354, fbs: 0, restecg: 0, thalach: 163, exang: 1, oldpeak: 0.6, slope: 1, ca: 0, thal: 3, target: 0 },
  { age: 63, sex: 1, cp: 4, trestbps: 130, chol: 254, fbs: 0, restecg: 2, thalach: 147, exang: 0, oldpeak: 1.4, slope: 2, ca: 1, thal: 7, target: 1 },
  { age: 53, sex: 1, cp: 4, trestbps: 140, chol: 203, fbs: 1, restecg: 2, thalach: 155, exang: 1, oldpeak: 3.1, slope: 3, ca: 0, thal: 7, target: 1 },
  { age: 57, sex: 1, cp: 4, trestbps: 140, chol: 192, fbs: 0, restecg: 0, thalach: 148, exang: 0, oldpeak: 0.4, slope: 2, ca: 0, thal: 6, target: 0 },
  { age: 56, sex: 0, cp: 2, trestbps: 140, chol: 294, fbs: 0, restecg: 2, thalach: 153, exang: 0, oldpeak: 1.3, slope: 2, ca: 0, thal: 3, target: 0 }
];

export const HELD_OUT_ERROR_ANALYSIS = [
  { age: 48, sex: 1, cp: 2, trestbps: 110, chol: 229, thalach: 168, oldpeak: 1.0, ca: 0, thal: 7, actual: 1, predicted: 0, prob: 0.412, type: "False Negative", note: "High exercise heart rate (168 bpm) masked atypical chest pain." },
  { age: 58, sex: 1, cp: 2, trestbps: 120, chol: 284, thalach: 160, oldpeak: 1.8, ca: 0, thal: 3, actual: 1, predicted: 0, prob: 0.384, type: "False Negative", note: "Absence of fluoroscopy vessels (ca=0) counteracted elevated ST depression." },
  { age: 49, sex: 1, cp: 3, trestbps: 118, chol: 149, thalach: 126, oldpeak: 0.8, ca: 3, thal: 3, actual: 1, predicted: 0, prob: 0.441, type: "False Negative", note: "Low resting blood pressure and modest cholesterol depressed model output." },
  { age: 67, sex: 0, cp: 3, trestbps: 115, chol: 564, thalach: 160, oldpeak: 1.6, ca: 0, thal: 7, actual: 0, predicted: 1, prob: 0.562, type: "False Positive", note: "Severe hypercholesterolemia (564 mg/dl) drove statistical score above 50%." },
  { age: 59, sex: 1, cp: 4, trestbps: 135, chol: 234, thalach: 161, oldpeak: 0.5, ca: 0, thal: 7, actual: 0, predicted: 1, prob: 0.521, type: "False Positive", note: "Reversible thallium defect (thal=7) pushed borderline case into positive tier." },
  { age: 54, sex: 1, cp: 3, trestbps: 150, chol: 232, thalach: 165, oldpeak: 1.6, ca: 0, thal: 7, actual: 0, predicted: 1, prob: 0.534, type: "False Positive", note: "Elevated systolic BP combined with scintigraphy finding crossed threshold." }
];
