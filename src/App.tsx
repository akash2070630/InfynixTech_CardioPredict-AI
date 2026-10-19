import React, { useState, useMemo } from 'react';
import {
  Activity,
  Heart,
  BarChart3,
  Sliders,
  Award,
  FlaskConical,
  BookOpen,
  Download,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  FileText,
  TrendingUp,
  Search,
  ExternalLink,
  ChevronRight,
  ShieldAlert,
  ArrowRight,
  Code2,
  Copy,
  Check,
  FileCode2,
  Terminal
} from 'lucide-react';
import {
  TOURNAMENT_DATA,
  FEATURE_IMPORTANCES,
  COHORT_SAMPLES,
  HELD_OUT_ERROR_ANALYSIS,
  PRESET_PATIENTS,
  computePrediction,
  PatientRecord
} from './data/heartData';
import { PROJECT_CODE_FILES, CodeFile } from './data/codeFiles';

type TabType = 'overview' | 'explorer' | 'studio' | 'tournament' | 'lab' | 'code' | 'docs';

export default function App() {
  const [activeTab, setActiveTab] = useState<TabType>('overview');

  // Prediction Studio State
  const [patientForm, setPatientForm] = useState<Omit<PatientRecord, 'target'>>(PRESET_PATIENTS[0].data);
  const [selectedPreset, setSelectedPreset] = useState<string>(PRESET_PATIENTS[0].name);

  // What-If Explorer State
  const [sweepFeature, setSweepFeature] = useState<'thalach' | 'oldpeak' | 'trestbps' | 'chol' | 'age'>('thalach');

  // Data Explorer State
  const [selectedBiomarker, setSelectedBiomarker] = useState<'thalach' | 'oldpeak' | 'trestbps' | 'chol' | 'age'>('thalach');
  const [docSubTab, setDocSubTab] = useState<'architecture' | 'methodology' | 'report'>('architecture');

  // Code Viewer State
  const [selectedCodeFile, setSelectedCodeFile] = useState<CodeFile>(PROJECT_CODE_FILES[0]);
  const [isCopied, setIsCopied] = useState<boolean>(false);

  // Compute live prediction
  const currentPrediction = useMemo(() => computePrediction(patientForm), [patientForm]);

  // Compute sensitivity sweep data
  const sweepData = useMemo(() => {
    const sweeps: { val: number; prob: number }[] = [];
    let min = 80, max = 200, step = 10;
    if (sweepFeature === 'oldpeak') { min = 0; max = 5.0; step = 0.5; }
    else if (sweepFeature === 'trestbps') { min = 95; max = 195; step = 10; }
    else if (sweepFeature === 'chol') { min = 130; max = 450; step = 25; }
    else if (sweepFeature === 'age') { min = 30; max = 75; step = 5; }

    for (let v = min; v <= max; v += step) {
      const cloned = { ...patientForm, [sweepFeature]: v };
      const res = computePrediction(cloned);
      sweeps.push({ val: Number(v.toFixed(1)), prob: Number((res.probability * 100).toFixed(1)) });
    }
    return sweeps;
  }, [patientForm, sweepFeature]);

  const handlePresetSelect = (presetName: string) => {
    const found = PRESET_PATIENTS.find(p => p.name === presetName);
    if (found) {
      setPatientForm(found.data);
      setSelectedPreset(found.name);
    }
  };

  const handleCopyCode = (text: string) => {
    navigator.clipboard.writeText(text);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  const handleDownloadFile = (file: CodeFile) => {
    const blob = new Blob([file.content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = file.name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleDownloadReport = () => {
    const reportData = {
      project: "CardioPredict AI",
      track: "Machine Learning & Clinical Data Engineering Capstone",
      timestamp: new Date().toISOString(),
      cohort_spec: "Cleveland Clinic Foundation (UCI)",
      selected_model: "Logistic Regression (L2 Regularized)",
      held_out_metrics: {
        accuracy: "85.25%",
        sensitivity_recall: "89.29%",
        specificity: "81.82%",
        precision: "80.65%",
        f1_score: "0.8475",
        roc_auc: "0.9123"
      },
      selection_criteria: "Weighted Clinical Score: 0.50*Recall + 0.30*ROC_AUC + 0.20*F1 = 0.8896",
      tournament: TOURNAMENT_DATA
    };
    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "cardiopredict_ai_evaluation_report.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans">
      {/* ---------------- TOP BAR CONTRACT ---------------- */}
      <header className="sticky top-0 z-30 bg-white/95 backdrop-blur-md border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-8">
          {/* Zone 1: Brand Wordmark */}
          <div className="flex items-center gap-3 shrink-0">
            <div className="w-9 h-9 rounded-lg bg-sky-900 text-white flex items-center justify-center font-bold text-lg shadow-sm">
              <Heart className="w-5 h-5 text-rose-400 fill-rose-400/30" />
            </div>
            <div>
              <span className="text-lg font-bold tracking-tight text-sky-950 block leading-tight">
                CardioPredict AI
              </span>
              <span className="text-[11px] text-slate-500 font-medium">Clinical ML Intelligence</span>
            </div>
          </div>

          {/* Zone 2: Navigation Links (Single-Line) */}
          <nav className="hidden md:flex items-center gap-1 lg:gap-1.5">
            {[
              { id: 'overview', label: 'Executive Overview', icon: Activity },
              { id: 'explorer', label: 'Data Explorer', icon: BarChart3 },
              { id: 'studio', label: 'Prediction Studio', icon: Sliders },
              { id: 'tournament', label: 'Model Tournament', icon: Award },
              { id: 'lab', label: 'Evaluation Lab', icon: FlaskConical },
              { id: 'code', label: 'Project Source Code', icon: Code2 },
              { id: 'docs', label: 'Documentation', icon: BookOpen },
            ].map(tab => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as TabType)}
                  className={`px-2.5 py-1.5 text-xs font-medium rounded-md transition-all flex items-center gap-1.5 whitespace-nowrap shrink-0 ${
                    isActive
                      ? 'bg-sky-50 text-sky-900 font-semibold shadow-xs'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-sky-700' : 'text-slate-400'}`} />
                  {tab.label}
                </button>
              );
            })}
          </nav>

          {/* Zone 3: Primary Action */}
          <div className="flex items-center gap-3 shrink-0">
            <button
              onClick={handleDownloadReport}
              className="px-3.5 py-1.5 text-xs font-semibold text-white bg-sky-900 rounded-md hover:bg-sky-800 transition-colors shadow-xs flex items-center gap-1.5 whitespace-nowrap shrink-0"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export Report</span>
            </button>
          </div>
        </div>
      </header>

      {/* Mobile Navigation Strip */}
      <div className="md:hidden flex overflow-x-auto gap-2 px-4 py-2 bg-white border-b border-slate-200">
        {[
          { id: 'overview', label: 'Overview' },
          { id: 'explorer', label: 'Explorer' },
          { id: 'studio', label: 'Studio' },
          { id: 'tournament', label: 'Tournament' },
          { id: 'lab', label: 'Lab' },
          { id: 'code', label: 'Source Code' },
          { id: 'docs', label: 'Docs' },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as TabType)}
            className={`px-3 py-1 text-xs rounded-md whitespace-nowrap ${
              activeTab === tab.id ? 'bg-sky-900 text-white font-semibold' : 'bg-slate-100 text-slate-700'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* ============================================================ */}
        {/* MODULE A: EXECUTIVE OVERVIEW */}
        {/* ============================================================ */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Clinical Value Banner */}
            <div className="bg-sky-50 border border-sky-200 rounded-lg p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-sky-950">
              <div>
                <p className="text-sm font-semibold text-sky-900">
                  CardioPredict AI — Clinical Decision Support & Machine Learning System
                </p>
                <p className="text-xs text-sky-800 mt-0.5">
                  Trained on validated Cleveland Clinic hemodynamic cohorts. Built with zero-leakage Scikit-learn pipelines and sensitivity-optimized classification.
                </p>
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-600 font-mono shrink-0">
                <span>Holdout ROC-AUC: 0.912</span>
                <span aria-hidden="true">·</span>
                <span>Recall: 89.3%</span>
              </div>
            </div>

            {/* KPI Metric Strip */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-xs">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">Validated Cohort</span>
                <span className="text-2xl font-bold text-sky-950 font-mono tabular-nums block mt-1">303 Patients</span>
                <span className="text-xs text-slate-600 mt-1 block">Cleveland Clinic Foundation</span>
              </div>
              <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-xs">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">Clinical Features</span>
                <span className="text-2xl font-bold text-sky-950 font-mono tabular-nums block mt-1">13 Predictors</span>
                <span className="text-xs text-slate-600 mt-1 block">Hemodynamic & Scintigraphy</span>
              </div>
              <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-xs">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">Held-Out ROC-AUC</span>
                <span className="text-2xl font-bold text-sky-900 font-mono tabular-nums block mt-1">0.912</span>
                <span className="text-xs text-slate-600 mt-1 block">Stratified 20% Holdout Split</span>
              </div>
              <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-xs">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">Held-Out Sensitivity</span>
                <span className="text-2xl font-bold text-emerald-700 font-mono tabular-nums block mt-1">89.3%</span>
                <span className="text-xs text-slate-600 mt-1 block">Minimizing False Negatives</span>
              </div>
            </div>

            {/* Three Core Pillars */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs">
                <div className="w-8 h-8 rounded-md bg-blue-100 text-blue-800 flex items-center justify-center font-bold text-sm mb-3">
                  01
                </div>
                <h3 className="text-base font-semibold text-slate-900">Leakage-Safe Architecture</h3>
                <p className="text-xs text-slate-600 mt-2 leading-relaxed">
                  Continuous biomarkers are scaled and categoricals are one-hot encoded exclusively inside Scikit-learn Pipeline abstractions, guaranteeing that training fold statistics never contaminate held-out test evaluations.
                </p>
              </div>
              <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs">
                <div className="w-8 h-8 rounded-md bg-teal-100 text-teal-800 flex items-center justify-center font-bold text-sm mb-3">
                  02
                </div>
                <h3 className="text-base font-semibold text-slate-900">Recall-Centric Optimization</h3>
                <p className="text-xs text-slate-600 mt-2 leading-relaxed">
                  In coronary triage, an undetected disease case (false negative) carries critical clinical peril. Candidate models are selected using a weighted clinical utility function that prioritizes sensitivity (50%) and ROC-AUC (30%).
                </p>
              </div>
              <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs">
                <div className="w-8 h-8 rounded-md bg-amber-100 text-amber-800 flex items-center justify-center font-bold text-sm mb-3">
                  03
                </div>
                <h3 className="text-base font-semibold text-slate-900">Explainable Inference</h3>
                <p className="text-xs text-slate-600 mt-2 leading-relaxed">
                  Rather than behaving as an opaque black box, CardioPredict surfaces standardized odds multipliers, feature attribution rank-orders, and local What-If sensitivity trajectories for each patient profile.
                </p>
              </div>
            </div>

            {/* Dataset Quick-Inspection Table */}
            <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                <div>
                  <h3 className="text-sm font-semibold text-slate-900">Validated Clinical Cohort Snapshot</h3>
                  <p className="text-xs text-slate-500">Representative patient records from the Cleveland Foundation repository.</p>
                </div>
                <button
                  onClick={() => setActiveTab('explorer')}
                  className="text-xs font-semibold text-sky-900 hover:text-sky-700 flex items-center gap-1"
                >
                  <span>Open Full Clinical Explorer</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-100 text-slate-700 uppercase font-semibold text-[10px] tracking-wider border-b border-slate-200">
                    <tr>
                      <th className="py-2.5 px-3">Age</th>
                      <th className="py-2.5 px-3">Sex</th>
                      <th className="py-2.5 px-3">Chest Pain</th>
                      <th className="py-2.5 px-3 text-right">Resting BP</th>
                      <th className="py-2.5 px-3 text-right">Cholesterol</th>
                      <th className="py-2.5 px-3 text-right">Peak HR</th>
                      <th className="py-2.5 px-3 text-right">ST Depr</th>
                      <th className="py-2.5 px-3 text-center">Vessels (ca)</th>
                      <th className="py-2.5 px-3 text-center">Thal</th>
                      <th className="py-2.5 px-3 text-center">CAD Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {COHORT_SAMPLES.slice(0, 7).map((p, idx) => (
                      <tr key={idx} className="hover:bg-slate-50/75 transition-colors font-mono">
                        <td className="py-2.5 px-3 font-sans font-medium text-slate-900">{p.age} yrs</td>
                        <td className="py-2.5 px-3 font-sans">{p.sex === 1 ? 'Male' : 'Female'}</td>
                        <td className="py-2.5 px-3 font-sans">
                          {p.cp === 1 ? 'Typical Angina' : p.cp === 2 ? 'Atypical' : p.cp === 3 ? 'Non-anginal' : 'Asymptomatic'}
                        </td>
                        <td className="py-2.5 px-3 text-right tabular-nums">{p.trestbps}</td>
                        <td className="py-2.5 px-3 text-right tabular-nums">{p.chol}</td>
                        <td className="py-2.5 px-3 text-right tabular-nums">{p.thalach}</td>
                        <td className="py-2.5 px-3 text-right tabular-nums">{p.oldpeak.toFixed(1)}</td>
                        <td className="py-2.5 px-3 text-center tabular-nums">{p.ca}</td>
                        <td className="py-2.5 px-3 text-center tabular-nums">{p.thal}</td>
                        <td className="py-2.5 px-3 text-center">
                          <span className={`inline-block px-2 py-0.5 rounded text-[11px] font-sans font-medium ${
                            p.target === 1 ? 'bg-amber-100 text-amber-900' : 'bg-slate-100 text-slate-700'
                          }`}>
                            {p.target === 1 ? 'Present (>50%)' : 'Absent'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Educational Disclaimer */}
            <div className="bg-amber-50/80 border border-amber-200 rounded-lg p-4 text-xs text-amber-900 flex items-start gap-3">
              <ShieldAlert className="w-5 h-5 text-amber-700 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold block">Academic & Decision-Support Scope</span>
                This system is built as an educational demonstration of healthcare machine learning and clinical triage algorithms. Predicted probabilities reflect observational statistical associations from historic research cohorts. It must not be deployed as an autonomous diagnostic device or override physician evaluation.
              </div>
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* MODULE B: CLINICAL DATA EXPLORER */}
        {/* ============================================================ */}
        {activeTab === 'explorer' && (
          <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
              <div>
                <h2 className="text-xl font-bold text-slate-900">Clinical Data Explorer</h2>
                <p className="text-xs text-slate-500">Distribution analysis, feature dictionaries, and correlation metrics from the 303-patient Cleveland cohort.</p>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-500">Stratify Biomarker:</span>
                <select
                  value={selectedBiomarker}
                  onChange={(e) => setSelectedBiomarker(e.target.value as any)}
                  className="bg-white border border-slate-300 text-slate-900 text-xs rounded-md px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-sky-600"
                >
                  <option value="thalach">Max Heart Rate (thalach)</option>
                  <option value="oldpeak">ST Depression (oldpeak)</option>
                  <option value="trestbps">Resting BP (trestbps)</option>
                  <option value="chol">Serum Cholesterol (chol)</option>
                  <option value="age">Patient Age (age)</option>
                </select>
              </div>
            </div>

            {/* Distribution Analysis Cards */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 bg-white border border-slate-200 rounded-lg p-5 shadow-xs">
                <h3 className="text-sm font-semibold text-slate-900 mb-1">
                  Hemodynamic Distribution Comparison: Healthy vs CAD Cohorts
                </h3>
                <p className="text-xs text-slate-500 mb-4">
                  Aggregated biomarker distributions contrast healthy participants (absence) against confirmed CAD cases.
                </p>

                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-xs text-slate-600 mb-1">
                      <span>Normal / Absence (Target = 0) Mean Baseline</span>
                      <span className="font-mono tabular-nums font-semibold">
                        {selectedBiomarker === 'thalach' ? '158.4 bpm' : selectedBiomarker === 'oldpeak' ? '0.58 mm' : selectedBiomarker === 'trestbps' ? '129.2 mm Hg' : selectedBiomarker === 'chol' ? '242.6 mg/dl' : '52.6 yrs'}
                      </span>
                    </div>
                    <div className="h-6 w-full bg-slate-100 rounded-md overflow-hidden flex">
                      <div className="h-full bg-sky-800 rounded-md transition-all duration-500" style={{ width: selectedBiomarker === 'thalach' ? '78%' : selectedBiomarker === 'oldpeak' ? '18%' : selectedBiomarker === 'trestbps' ? '65%' : selectedBiomarker === 'chol' ? '55%' : '60%' }}></div>
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs text-amber-900 mb-1">
                      <span>Confirmed CAD Present (Target = 1) Mean Baseline</span>
                      <span className="font-mono tabular-nums font-semibold">
                        {selectedBiomarker === 'thalach' ? '139.3 bpm' : selectedBiomarker === 'oldpeak' ? '1.57 mm' : selectedBiomarker === 'trestbps' ? '134.5 mm Hg' : selectedBiomarker === 'chol' ? '251.5 mg/dl' : '56.6 yrs'}
                      </span>
                    </div>
                    <div className="h-6 w-full bg-slate-100 rounded-md overflow-hidden flex">
                      <div className="h-full bg-amber-600 rounded-md transition-all duration-500" style={{ width: selectedBiomarker === 'thalach' ? '62%' : selectedBiomarker === 'oldpeak' ? '64%' : selectedBiomarker === 'trestbps' ? '71%' : selectedBiomarker === 'chol' ? '61%' : '69%' }}></div>
                    </div>
                  </div>
                </div>

                <div className="mt-5 p-3.5 bg-slate-50 border border-slate-200 rounded-md text-xs text-slate-600 leading-relaxed">
                  <span className="font-semibold text-slate-800">Physiological Insight: </span>
                  {selectedBiomarker === 'thalach' && "Patients with coronary artery disease consistently achieve significantly lower peak heart rates (mean ~139 bpm vs ~158 bpm in healthy subjects), reflecting exertional ischemia and chronotropic incompetence during stress tests."}
                  {selectedBiomarker === 'oldpeak' && "Exercise-induced ST depression is nearly three-fold higher in CAD patients (mean 1.57 mm vs 0.58 mm), representing classic electrophysiological evidence of subendocardial myocardial ischemia."}
                  {selectedBiomarker === 'trestbps' && "Resting systolic blood pressure demonstrates modest positive shift in CAD patients, serving as a contributing vascular stressor rather than a standalone discriminator."}
                  {selectedBiomarker === 'chol' && "Serum cholesterol displays substantial cohort overlap, underscoring that raw cholesterol alone is insufficient without scintigraphy and stress biomarkers."}
                  {selectedBiomarker === 'age' && "Average age in symptomatic patients is ~56.6 years versus ~52.6 in the healthy control group, corroborating known cardiovascular demographic risk curves."}
                </div>
              </div>

              {/* Class Imbalance & Data Quality Card */}
              <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs flex flex-col justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-slate-900 mb-1">Cohort Balance & Quality Audit</h3>
                  <p className="text-xs text-slate-500 mb-4">Inspection of missing records and diagnostic balance.</p>

                  <div className="space-y-3">
                    <div className="flex justify-between items-center text-xs pb-2 border-b border-slate-100">
                      <span className="text-slate-600">Total Valid Records</span>
                      <span className="font-mono font-semibold text-slate-900">303 cases</span>
                    </div>
                    <div className="flex justify-between items-center text-xs pb-2 border-b border-slate-100">
                      <span className="text-slate-600">Absence of CAD (Target = 0)</span>
                      <span className="font-mono font-semibold text-sky-800">164 (54.1%)</span>
                    </div>
                    <div className="flex justify-between items-center text-xs pb-2 border-b border-slate-100">
                      <span className="text-slate-600">CAD Present (Target = 1)</span>
                      <span className="font-mono font-semibold text-amber-800">139 (45.9%)</span>
                    </div>
                    <div className="flex justify-between items-center text-xs pb-2 border-b border-slate-100">
                      <span className="text-slate-600">Missing Values in ca / thal</span>
                      <span className="font-mono font-semibold text-emerald-700">6 imputed (median)</span>
                    </div>
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-slate-600">Data Leakage Invariants</span>
                      <span className="font-mono font-semibold text-emerald-700">Verified 0% Leakage</span>
                    </div>
                  </div>
                </div>

                <div className="mt-4 p-3 bg-emerald-50 border border-emerald-200 rounded-md text-[11px] text-emerald-800">
                  Class proportions are well-balanced (54% vs 46%), avoiding severe class imbalance artifacts while StratifiedKFold maintains identical ratios across all 5 folds.
                </div>
              </div>
            </div>

            {/* Feature Dictionary Reference Table */}
            <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs">
              <h3 className="text-sm font-semibold text-slate-900 mb-1">Clinical Biomarker Dictionary & Ranges</h3>
              <p className="text-xs text-slate-500 mb-4">Standardized clinical mapping for the 13 predictive variables.</p>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-100 text-slate-700 uppercase font-semibold text-[10px] tracking-wider border-b border-slate-200">
                    <tr>
                      <th className="py-2.5 px-3">Variable</th>
                      <th className="py-2.5 px-3">Biomarker Name</th>
                      <th className="py-2.5 px-3">Data Type</th>
                      <th className="py-2.5 px-3">Reference / Category Range</th>
                      <th className="py-2.5 px-3">Diagnostic Significance</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {[
                      { var: "age", name: "Patient Age", type: "Continuous", range: "29 – 77 yrs", note: "Cardiovascular risk rises with chronological age due to arterial stiffening." },
                      { var: "sex", name: "Biological Sex", type: "Binary", range: "0: Female, 1: Male", note: "Higher baseline CAD prevalence observed in males within this cohort." },
                      { var: "cp", name: "Chest Pain Type", type: "Categorical", range: "1: Typical, 2: Atypical, 3: Non-anginal, 4: Asymptomatic", note: "Asymptomatic (type 4) often correlates with severe silent ischemia in clinical testing." },
                      { var: "trestbps", name: "Resting Blood Pressure", type: "Continuous", range: "94 – 200 mm Hg", note: "Resting systolic pressure at admission. Chronic hypertension damages endothelium." },
                      { var: "chol", name: "Serum Cholesterol", type: "Continuous", range: "126 – 564 mg/dl", note: "Fasting total cholesterol. Marker of systemic atherogenic lipid burden." },
                      { var: "fbs", name: "Fasting Blood Sugar", type: "Binary", range: "0: <= 120, 1: > 120 mg/dl", note: "Indicates impaired glucose metabolism or diabetes comorbidity." },
                      { var: "restecg", name: "Resting ECG", type: "Categorical", range: "0: Normal, 1: ST-T wave, 2: LVH", note: "Baseline electrical conduction abnormalities and ventricular strain." },
                      { var: "thalach", name: "Max Achieved Heart Rate", type: "Continuous", range: "71 – 202 bpm", note: "Chronotropic response during Bruce treadmill stress test. Protective when high." },
                      { var: "exang", name: "Exercise-Induced Angina", type: "Binary", range: "0: No, 1: Yes", note: "Precipitation of substernal chest discomfort during physical exertion." },
                      { var: "oldpeak", name: "ST Depression", type: "Continuous", range: "0.0 – 6.2 mm", note: "Degree of horizontal or downsloping ST segment displacement during stress." },
                      { var: "slope", name: "ST Segment Slope", type: "Categorical", range: "1: Upsloping, 2: Flat, 3: Downsloping", note: "Flat or downsloping patterns reflect severe subendocardial ischemia." },
                      { var: "ca", name: "Fluoroscopy Vessels", type: "Ordinal", range: "0 – 3 major vessels", note: "Number of major coronary arteries stained during barium fluoroscopy." },
                      { var: "thal", name: "Thallium Scintigraphy", type: "Categorical", range: "3: Normal, 6: Fixed, 7: Reversible", note: "Myocardial perfusion scintigraphy evaluating viable vs ischemic tissue." },
                    ].map((row, i) => (
                      <tr key={i} className="hover:bg-slate-50/75 transition-colors">
                        <td className="py-2.5 px-3 font-mono font-semibold text-sky-950">{row.var}</td>
                        <td className="py-2.5 px-3 font-medium text-slate-900">{row.name}</td>
                        <td className="py-2.5 px-3 text-slate-500">{row.type}</td>
                        <td className="py-2.5 px-3 font-mono text-slate-700">{row.range}</td>
                        <td className="py-2.5 px-3 text-slate-600">{row.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* MODULE C: PREDICTION STUDIO & WHAT-IF EXPLORER */}
        {/* ============================================================ */}
        {activeTab === 'studio' && (
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h2 className="text-xl font-bold text-slate-900">Prediction Studio & What-If Simulator</h2>
                <p className="text-xs text-slate-500">Live patient parameter intake, risk stratification, and sensitivity perturbation.</p>
              </div>

              {/* Preset Selector */}
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-500">Clinical Preset:</span>
                <div className="flex items-center gap-1 bg-white p-1 border border-slate-200 rounded-md">
                  {PRESET_PATIENTS.map(p => (
                    <button
                      key={p.name}
                      onClick={() => handlePresetSelect(p.name)}
                      className={`px-2.5 py-1 text-xs rounded transition-colors whitespace-nowrap ${
                        selectedPreset === p.name ? 'bg-sky-900 text-white font-medium' : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      {p.name.split('—')[0].trim()}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Patient Parameter Input Form */}
              <div className="lg:col-span-7 bg-white border border-slate-200 rounded-lg p-5 shadow-xs space-y-5">
                <div>
                  <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
                    01. Demographics & Intake Vitals
                  </h3>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div>
                      <label className="text-xs font-medium text-slate-700 block mb-1">
                        Age ({patientForm.age} yrs)
                      </label>
                      <input
                        type="range"
                        min="28"
                        max="78"
                        value={patientForm.age}
                        onChange={(e) => setPatientForm({ ...patientForm, age: Number(e.target.value) })}
                        className="w-full accent-sky-900"
                      />
                    </div>
                    <div>
                      <label className="text-xs font-medium text-slate-700 block mb-1">Biological Sex</label>
                      <select
                        value={patientForm.sex}
                        onChange={(e) => setPatientForm({ ...patientForm, sex: Number(e.target.value) })}
                        className="w-full bg-slate-50 border border-slate-300 text-xs rounded px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-sky-700"
                      >
                        <option value={1}>1: Male</option>
                        <option value={0}>0: Female</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs font-medium text-slate-700 block mb-1">
                        Resting BP ({patientForm.trestbps} mm Hg)
                      </label>
                      <input
                        type="range"
                        min="94"
                        max="200"
                        step="2"
                        value={patientForm.trestbps}
                        onChange={(e) => setPatientForm({ ...patientForm, trestbps: Number(e.target.value) })}
                        className="w-full accent-sky-900"
                      />
                    </div>
                  </div>
                </div>

                <div className="pt-3 border-t border-slate-100">
                  <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
                    02. Symptom Presentation & Treadmill Stress
                  </h3>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div>
                      <label className="text-xs font-medium text-slate-700 block mb-1">Chest Pain Type</label>
                      <select
                        value={patientForm.cp}
                        onChange={(e) => setPatientForm({ ...patientForm, cp: Number(e.target.value) })}
                        className="w-full bg-slate-50 border border-slate-300 text-xs rounded px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-sky-700"
                      >
                        <option value={1}>1: Typical Angina</option>
                        <option value={2}>2: Atypical Angina</option>
                        <option value={3}>3: Non-Anginal Pain</option>
                        <option value={4}>4: Asymptomatic</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs font-medium text-slate-700 block mb-1">
                        Peak Heart Rate ({patientForm.thalach} bpm)
                      </label>
                      <input
                        type="range"
                        min="71"
                        max="202"
                        value={patientForm.thalach}
                        onChange={(e) => setPatientForm({ ...patientForm, thalach: Number(e.target.value) })}
                        className="w-full accent-sky-900"
                      />
                    </div>
                    <div>
                      <label className="text-xs font-medium text-slate-700 block mb-1">Exertional Angina</label>
                      <select
                        value={patientForm.exang}
                        onChange={(e) => setPatientForm({ ...patientForm, exang: Number(e.target.value) })}
                        className="w-full bg-slate-50 border border-slate-300 text-xs rounded px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-sky-700"
                      >
                        <option value={0}>0: Absent (No)</option>
                        <option value={1}>1: Present (Yes)</option>
                      </select>
                    </div>
                  </div>
                </div>

                <div className="pt-3 border-t border-slate-100">
                  <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
                    03. Diagnostic Biomarkers & Scintigraphy
                  </h3>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div>
                      <label className="text-xs font-medium text-slate-700 block mb-1">
                        Cholesterol ({patientForm.chol} mg/dl)
                      </label>
                      <input
                        type="range"
                        min="126"
                        max="480"
                        step="5"
                        value={patientForm.chol}
                        onChange={(e) => setPatientForm({ ...patientForm, chol: Number(e.target.value) })}
                        className="w-full accent-sky-900"
                      />
                    </div>
                    <div>
                      <label className="text-xs font-medium text-slate-700 block mb-1">
                        ST Depr ({patientForm.oldpeak.toFixed(1)} mm)
                      </label>
                      <input
                        type="range"
                        min="0.0"
                        max="5.0"
                        step="0.1"
                        value={patientForm.oldpeak}
                        onChange={(e) => setPatientForm({ ...patientForm, oldpeak: Number(e.target.value) })}
                        className="w-full accent-sky-900"
                      />
                    </div>
                    <div>
                      <label className="text-xs font-medium text-slate-700 block mb-1">ST Slope</label>
                      <select
                        value={patientForm.slope}
                        onChange={(e) => setPatientForm({ ...patientForm, slope: Number(e.target.value) })}
                        className="w-full bg-slate-50 border border-slate-300 text-xs rounded px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-sky-700"
                      >
                        <option value={1}>1: Upsloping</option>
                        <option value={2}>2: Flat</option>
                        <option value={3}>3: Downsloping</option>
                      </select>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-3">
                    <div>
                      <label className="text-xs font-medium text-slate-700 block mb-1">Fasting Sugar &gt; 120</label>
                      <select
                        value={patientForm.fbs}
                        onChange={(e) => setPatientForm({ ...patientForm, fbs: Number(e.target.value) })}
                        className="w-full bg-slate-50 border border-slate-300 text-xs rounded px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-sky-700"
                      >
                        <option value={0}>0: Normal (&lt;=120)</option>
                        <option value={1}>1: Elevated (&gt;120)</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs font-medium text-slate-700 block mb-1">Fluoroscopy Vessels (ca)</label>
                      <select
                        value={patientForm.ca}
                        onChange={(e) => setPatientForm({ ...patientForm, ca: Number(e.target.value) })}
                        className="w-full bg-slate-50 border border-slate-300 text-xs rounded px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-sky-700"
                      >
                        <option value={0}>0 Major Vessels</option>
                        <option value={1}>1 Major Vessel</option>
                        <option value={2}>2 Major Vessels</option>
                        <option value={3}>3 Major Vessels</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs font-medium text-slate-700 block mb-1">Thallium Perfusion (thal)</label>
                      <select
                        value={patientForm.thal}
                        onChange={(e) => setPatientForm({ ...patientForm, thal: Number(e.target.value) })}
                        className="w-full bg-slate-50 border border-slate-300 text-xs rounded px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-sky-700"
                      >
                        <option value={3}>3.0: Normal Perfusion</option>
                        <option value={6}>6.0: Fixed Defect</option>
                        <option value={7}>7.0: Reversible Defect</option>
                      </select>
                    </div>
                  </div>
                </div>
              </div>

              {/* Prediction Decision Card */}
              <div className="lg:col-span-5 flex flex-col gap-4">
                <div className={`border rounded-lg p-5 shadow-xs transition-colors ${
                  currentPrediction.prediction === 1
                    ? 'bg-amber-50/70 border-amber-300 text-amber-950'
                    : 'bg-emerald-50/70 border-emerald-300 text-emerald-950'
                }`}>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-600">Model Decision Verdict</span>
                    <span className={`text-xs px-2.5 py-0.5 rounded font-medium ${
                      currentPrediction.prediction === 1 ? 'bg-amber-200 text-amber-900' : 'bg-emerald-200 text-emerald-900'
                    }`}>
                      {currentPrediction.riskLabel}
                    </span>
                  </div>

                  <div className="text-3xl font-bold font-mono tabular-nums tracking-tight mb-1">
                    {(currentPrediction.probability * 100).toFixed(1)}%
                  </div>
                  <p className="text-xs text-slate-600 mb-4">
                    Computed statistical risk of angiographic stenosis (&gt;=50% narrowing).
                  </p>

                  {/* Progress Gauge */}
                  <div className="w-full h-3 bg-white/80 border border-slate-200 rounded-full overflow-hidden mb-3">
                    <div
                      className={`h-full transition-all duration-300 ${
                        currentPrediction.probability >= 0.5 ? 'bg-amber-600' : 'bg-sky-700'
                      }`}
                      style={{ width: `${Math.min(100, Math.max(5, currentPrediction.probability * 100))}%` }}
                    ></div>
                  </div>

                  <div className="flex justify-between text-[11px] text-slate-500 font-mono">
                    <span>0% (Low Risk)</span>
                    <span className="text-rose-600 font-semibold">50% Threshold</span>
                    <span>100% (High Risk)</span>
                  </div>
                </div>

                {/* Local Feature Attribution Breakdown */}
                <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs flex-1">
                  <h4 className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
                    Predictive Influence Drivers
                  </h4>
                  <p className="text-xs text-slate-500 mb-3">
                    Contribution of patient features relative to cohort baseline.
                  </p>

                  <div className="space-y-2">
                    {currentPrediction.contributions.slice(0, 5).map((c, i) => (
                      <div key={i} className="text-xs flex items-center justify-between">
                        <span className="text-slate-700 font-medium">{c.label}</span>
                        <div className="flex items-center gap-2">
                          <div className={`w-16 h-1.5 rounded-full overflow-hidden bg-slate-100 flex ${c.val >= 0 ? 'justify-start' : 'justify-end'}`}>
                            <div
                              className={`h-full ${c.val >= 0 ? 'bg-amber-500' : 'bg-sky-600'}`}
                              style={{ width: `${Math.min(100, Math.abs(c.val) * 70)}%` }}
                            ></div>
                          </div>
                          <span className={`font-mono text-[11px] font-semibold w-12 text-right ${c.val >= 0 ? 'text-amber-700' : 'text-sky-700'}`}>
                            {c.val >= 0 ? `+${c.val.toFixed(2)}` : c.val.toFixed(2)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* WHAT-IF SENSITIVITY SIMULATOR */}
            <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                <div>
                  <h3 className="text-sm font-semibold text-slate-900">What-If Sensitivity Simulation</h3>
                  <p className="text-xs text-slate-500">
                    Simulate how adjusting a single clinical factor sweeps the model's computed risk trajectory.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-500">Sweep Variable:</span>
                  <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-md">
                    {[
                      { id: 'thalach', label: 'Max HR' },
                      { id: 'oldpeak', label: 'ST Depr' },
                      { id: 'trestbps', label: 'Resting BP' },
                      { id: 'chol', label: 'Cholesterol' },
                      { id: 'age', label: 'Age' }
                    ].map(btn => (
                      <button
                        key={btn.id}
                        onClick={() => setSweepFeature(btn.id as any)}
                        className={`px-2.5 py-1 text-xs rounded transition-colors ${
                          sweepFeature === btn.id ? 'bg-white text-slate-900 font-semibold shadow-xs' : 'text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        {btn.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Sensitivity Grid Data */}
              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2 pt-2">
                {sweepData.map((pt, idx) => (
                  <div
                    key={idx}
                    className={`border rounded-md p-2 text-center text-xs transition-colors ${
                      pt.prob >= 50 ? 'bg-amber-50/60 border-amber-200' : 'bg-slate-50 border-slate-200'
                    }`}
                  >
                    <span className="text-[10px] text-slate-500 block font-mono">
                      {pt.val} {sweepFeature === 'thalach' ? 'bpm' : sweepFeature === 'oldpeak' ? 'mm' : sweepFeature === 'trestbps' ? 'mm Hg' : sweepFeature === 'chol' ? 'mg/dl' : 'yrs'}
                    </span>
                    <span className={`font-bold font-mono block mt-1 ${
                      pt.prob >= 50 ? 'text-amber-800' : 'text-sky-900'
                    }`}>
                      {pt.prob}%
                    </span>
                  </div>
                ))}
              </div>
              <p className="text-[11px] text-slate-500 mt-3 italic">
                Notice: Higher peak heart rate (thalach) exerts a protective monotonic decrease in risk probability, confirming physiological validity.
              </p>
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* MODULE D: MODEL TOURNAMENT & INTELLIGENCE */}
        {/* ============================================================ */}
        {activeTab === 'tournament' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-xl font-bold text-slate-900">Model Intelligence & Comparison Tournament</h2>
              <p className="text-xs text-slate-500">
                Rigorous empirical benchmarking across 4 machine learning architectures under identical stratified 5-fold cross-validation.
              </p>
            </div>

            {/* Tournament Leaderboard Table */}
            <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-slate-900">Tournament Evaluation Leaderboard</h3>
                <span className="text-xs text-slate-500 font-mono">Stratified 20% Holdout Split (N=61)</span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-100 text-slate-700 uppercase font-semibold text-[10px] tracking-wider border-b border-slate-200">
                    <tr>
                      <th className="py-2.5 px-3">Architecture</th>
                      <th className="py-2.5 px-3 text-right">CV Accuracy</th>
                      <th className="py-2.5 px-3 text-right">Held-Out Acc</th>
                      <th className="py-2.5 px-3 text-right font-bold text-sky-900">Sensitivity (Recall)</th>
                      <th className="py-2.5 px-3 text-right">Precision</th>
                      <th className="py-2.5 px-3 text-right">F1-Score</th>
                      <th className="py-2.5 px-3 text-right">Held-Out ROC-AUC</th>
                      <th className="py-2.5 px-3 text-right text-emerald-800 font-bold">Clinical Score</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {TOURNAMENT_DATA.map((m, i) => (
                      <tr key={i} className={`hover:bg-slate-50 transition-colors ${i === 0 ? 'bg-sky-50/50' : ''}`}>
                        <td className="py-3 px-3 font-semibold text-slate-900">
                          {m.model_name}
                          {i === 0 && (
                            <span className="ml-2 inline-block px-2 py-0.5 rounded text-[10px] font-sans font-medium bg-emerald-100 text-emerald-800">
                              Selected Model
                            </span>
                          )}
                        </td>
                        <td className="py-3 px-3 text-right font-mono tabular-nums">{(m.cv_accuracy_mean * 100).toFixed(1)}%</td>
                        <td className="py-3 px-3 text-right font-mono tabular-nums">{(m.test_accuracy * 100).toFixed(1)}%</td>
                        <td className="py-3 px-3 text-right font-mono tabular-nums font-bold text-sky-900">{(m.test_recall * 100).toFixed(1)}%</td>
                        <td className="py-3 px-3 text-right font-mono tabular-nums">{(m.test_precision * 100).toFixed(1)}%</td>
                        <td className="py-3 px-3 text-right font-mono tabular-nums">{m.test_f1.toFixed(3)}</td>
                        <td className="py-3 px-3 text-right font-mono tabular-nums">{m.test_roc_auc.toFixed(3)}</td>
                        <td className="py-3 px-3 text-right font-mono tabular-nums font-bold text-emerald-700">{m.clinical_score.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-4 p-3.5 bg-slate-50 border border-slate-200 rounded-md text-xs text-slate-600 leading-relaxed">
                <span className="font-semibold text-slate-800">Clinical Selection Rationale: </span>
                Rather than selecting solely on raw test accuracy, models are scored via the clinical objective:
                <span className="font-mono text-slate-900 font-semibold block my-1">
                  Clinical Score = 0.50 * Recall + 0.30 * ROC-AUC + 0.20 * F1
                </span>
                Logistic Regression achieves the highest sensitivity (<strong>89.3%</strong>) and ROC-AUC (<strong>0.912</strong>), outperforming tree ensembles by mitigating small-cohort overfitting while preventing dangerous false negatives.
              </div>
            </div>

            {/* Feature Importance Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs">
                <h3 className="text-sm font-semibold text-slate-900 mb-1">Global Feature Importance (Attributions)</h3>
                <p className="text-xs text-slate-500 mb-4">Standardized magnitude of influence in the selected classifier.</p>

                <div className="space-y-2.5">
                  {FEATURE_IMPORTANCES.slice(0, 8).map((f, idx) => (
                    <div key={idx} className="text-xs">
                      <div className="flex justify-between text-slate-700 mb-1 font-medium">
                        <span>{f.feature}</span>
                        <span className={`font-mono tabular-nums font-semibold ${f.category === 'Risk Factor' ? 'text-amber-700' : 'text-sky-700'}`}>
                          {f.importance.toFixed(3)} ({f.category})
                        </span>
                      </div>
                      <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${f.category === 'Risk Factor' ? 'bg-amber-500' : 'bg-sky-600'}`}
                          style={{ width: `${(f.importance / 0.85) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs flex flex-col justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-slate-900 mb-1">Model Architecture Breakdown</h3>
                  <p className="text-xs text-slate-500 mb-4">Comparison of structural inductive biases.</p>

                  <div className="space-y-3 text-xs">
                    <div className="p-3 bg-slate-50 rounded-md border border-slate-200">
                      <span className="font-semibold text-slate-900 block">Logistic Regression (Selected)</span>
                      <p className="text-slate-600 mt-1">L2 penalty (C=0.8) stabilizes multi-collinear hemodynamic biomarkers. High interpretability and lowest generalization variance on N=303 cohort.</p>
                    </div>
                    <div className="p-3 bg-slate-50 rounded-md border border-slate-200">
                      <span className="font-semibold text-slate-900 block">Random Forest (Ensemble)</span>
                      <p className="text-slate-600 mt-1">150 estimators with depth limitation. High train accuracy (94.6%) reveals slight over-specialization on smaller sub-splits.</p>
                    </div>
                    <div className="p-3 bg-slate-50 rounded-md border border-slate-200">
                      <span className="font-semibold text-slate-900 block">Gradient Boosting</span>
                      <p className="text-slate-600 mt-1">Sequential residual minimization. Strong discriminative score (0.884 ROC-AUC), but lower sensitivity on the clinical holdout.</p>
                    </div>
                  </div>
                </div>

                <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] text-slate-500 font-mono">
                  Random Seed: 42 · Scikit-learn Pipeline v1.3+ · Standardized Inputs
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* MODULE E: EVALUATION LAB & ERROR ANALYSIS */}
        {/* ============================================================ */}
        {activeTab === 'lab' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-xl font-bold text-slate-900">Evaluation Lab & Diagnostic Error Analysis</h2>
              <p className="text-xs text-slate-500">Held-out confusion matrix, error breakdown, and granular inspection of false negative edge cases.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Confusion Matrix Card */}
              <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs">
                <h3 className="text-sm font-semibold text-slate-900 mb-1">Held-Out Confusion Matrix (N = 61)</h3>
                <p className="text-xs text-slate-500 mb-4">Evaluated on strictly segregated 20% test partition.</p>

                <div className="grid grid-cols-2 gap-3 text-center">
                  <div className="p-4 bg-sky-50 border border-sky-200 rounded-md">
                    <span className="text-xs text-sky-800 font-medium block">True Negatives (TN)</span>
                    <span className="text-2xl font-bold font-mono text-sky-950 mt-1 block">27</span>
                    <span className="text-[11px] text-slate-500">Correctly classified healthy</span>
                  </div>
                  <div className="p-4 bg-amber-50 border border-amber-200 rounded-md">
                    <span className="text-xs text-amber-800 font-medium block">False Positives (FP)</span>
                    <span className="text-2xl font-bold font-mono text-amber-900 mt-1 block">6</span>
                    <span className="text-[11px] text-slate-500">Healthy misclassified as CAD</span>
                  </div>
                  <div className="p-4 bg-rose-50 border border-rose-200 rounded-md">
                    <span className="text-xs text-rose-800 font-medium block">False Negatives (FN)</span>
                    <span className="text-2xl font-bold font-mono text-rose-900 mt-1 block">3</span>
                    <span className="text-[11px] text-slate-500">CAD missed (Dangerous error)</span>
                  </div>
                  <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-md">
                    <span className="text-xs text-emerald-800 font-medium block">True Positives (TP)</span>
                    <span className="text-2xl font-bold font-mono text-emerald-950 mt-1 block">25</span>
                    <span className="text-[11px] text-slate-500">Correctly diagnosed CAD</span>
                  </div>
                </div>

                <div className="mt-4 pt-3 border-t border-slate-100 flex justify-between text-xs text-slate-600 font-mono">
                  <span>Accuracy: 85.2%</span>
                  <span>Sensitivity: 89.3%</span>
                  <span>Specificity: 81.8%</span>
                </div>
              </div>

              {/* Error Distribution Card */}
              <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs flex flex-col justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-slate-900 mb-1">Diagnostic Error Distribution</h3>
                  <p className="text-xs text-slate-500 mb-4">Breakdown of accurate classifications vs error types.</p>

                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-700 font-medium">Correct Diagnostic Decisions</span>
                        <span className="font-mono text-emerald-700 font-semibold">52 cases (85.2%)</span>
                      </div>
                      <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden">
                        <div className="h-full bg-emerald-600" style={{ width: '85.2%' }}></div>
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-700 font-medium">False Positives (Secondary Screening Needed)</span>
                        <span className="font-mono text-amber-700 font-semibold">6 cases (9.8%)</span>
                      </div>
                      <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden">
                        <div className="h-full bg-amber-500" style={{ width: '9.8%' }}></div>
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-700 font-medium">False Negatives (Critical Missed Cases)</span>
                        <span className="font-mono text-rose-700 font-semibold">3 cases (4.9%)</span>
                      </div>
                      <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden">
                        <div className="h-full bg-rose-600" style={{ width: '4.9%' }}></div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mt-4 p-3 bg-slate-50 border border-slate-200 rounded-md text-xs text-slate-600">
                  By tuning regularized thresholds and optimizing for clinical recall, the false negative rate was constrained to <strong>4.9%</strong> on the held-out dataset.
                </div>
              </div>
            </div>

            {/* Error Audit Table */}
            <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs">
              <h3 className="text-sm font-semibold text-slate-900 mb-1">Held-Out Diagnostic Misclassification Audit</h3>
              <p className="text-xs text-slate-500 mb-4">Granular forensic analysis of patients misclassified by the model.</p>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-100 text-slate-700 uppercase font-semibold text-[10px] tracking-wider border-b border-slate-200">
                    <tr>
                      <th className="py-2.5 px-3">Patient Profile</th>
                      <th className="py-2.5 px-3 text-center">Actual CAD</th>
                      <th className="py-2.5 px-3 text-center">Predicted CAD</th>
                      <th className="py-2.5 px-3 text-right">Computed Risk</th>
                      <th className="py-2.5 px-3">Error Classification</th>
                      <th className="py-2.5 px-3">Forensic Clinical Cause</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {HELD_OUT_ERROR_ANALYSIS.map((err, i) => (
                      <tr key={i} className="hover:bg-slate-50 transition-colors">
                        <td className="py-2.5 px-3 font-medium text-slate-900">
                          {err.age}yo {err.sex === 1 ? 'Male' : 'Female'}, BP {err.trestbps}, HR {err.thalach}
                        </td>
                        <td className="py-2.5 px-3 text-center font-mono">{err.actual}</td>
                        <td className="py-2.5 px-3 text-center font-mono">{err.predicted}</td>
                        <td className="py-2.5 px-3 text-right font-mono tabular-nums font-semibold">
                          {(err.prob * 100).toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-3">
                          <span className={`inline-block px-2 py-0.5 rounded text-[11px] font-medium ${
                            err.type === 'False Negative' ? 'bg-rose-100 text-rose-800' : 'bg-amber-100 text-amber-800'
                          }`}>
                            {err.type}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-slate-600">{err.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* MODULE F: PROJECT SOURCE CODE INSPECTOR */}
        {/* ============================================================ */}
        {activeTab === 'code' && (
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h2 className="text-xl font-bold text-slate-900">Project Source Code & Architecture</h2>
                <p className="text-xs text-slate-500">
                  Directly inspect, copy, or download the genuine machine learning pipeline, training code, test suites, and clinical configuration files.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-500 font-mono">6 Project Files</span>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* File List / Explorer Sidebar */}
              <div className="lg:col-span-4 bg-white border border-slate-200 rounded-lg p-4 shadow-xs space-y-2">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block mb-2 px-2">
                  Repository File Tree
                </span>
                <div className="space-y-1">
                  {PROJECT_CODE_FILES.map((file) => {
                    const isSelected = selectedCodeFile.path === file.path;
                    return (
                      <button
                        key={file.path}
                        onClick={() => setSelectedCodeFile(file)}
                        className={`w-full text-left p-2.5 rounded-md transition-all flex items-start gap-2.5 ${
                          isSelected
                            ? 'bg-sky-50 border border-sky-200 text-sky-950 font-medium shadow-xs'
                            : 'text-slate-700 hover:bg-slate-50 hover:text-slate-900 border border-transparent'
                        }`}
                      >
                        <FileCode2 className={`w-4 h-4 mt-0.5 shrink-0 ${isSelected ? 'text-sky-700' : 'text-slate-400'}`} />
                        <div className="truncate">
                          <span className="text-xs font-mono font-semibold block truncate">{file.name}</span>
                          <span className="text-[11px] text-slate-500 line-clamp-1 block mt-0.5">{file.description}</span>
                        </div>
                      </button>
                    );
                  })}
                </div>

                <div className="pt-4 mt-4 border-t border-slate-100 px-2 text-xs text-slate-600 space-y-2">
                  <span className="font-semibold text-slate-900 block">Local Terminal Commands:</span>
                  <p className="text-[11px] text-slate-500">
                    Run training pipeline:
                  </p>
                  <div className="bg-slate-900 text-slate-100 p-2 rounded text-[11px] font-mono select-all">
                    python train_model.py
                  </div>
                  <p className="text-[11px] text-slate-500">
                    Execute automated tests:
                  </p>
                  <div className="bg-slate-900 text-slate-100 p-2 rounded text-[11px] font-mono select-all">
                    pytest tests/ -v
                  </div>
                  <p className="text-[11px] text-slate-500">
                    Launch Streamlit application:
                  </p>
                  <div className="bg-slate-900 text-slate-100 p-2 rounded text-[11px] font-mono select-all">
                    streamlit run app.py
                  </div>
                </div>
              </div>

              {/* Code Viewer Panel */}
              <div className="lg:col-span-8 bg-white border border-slate-200 rounded-lg shadow-xs overflow-hidden flex flex-col">
                {/* File Header */}
                <div className="p-4 bg-slate-50 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono font-bold text-sky-950 bg-white px-2.5 py-1 rounded border border-slate-200 shadow-xs">
                      {selectedCodeFile.path}
                    </span>
                    <span className="text-[11px] uppercase tracking-wider font-semibold text-slate-500 font-mono">
                      {selectedCodeFile.language} · {selectedCodeFile.content.split('\n').length} lines
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleCopyCode(selectedCodeFile.content)}
                      className="px-3 py-1.5 text-xs font-medium bg-white border border-slate-200 rounded text-slate-700 hover:text-slate-900 hover:bg-slate-50 transition-colors shadow-xs flex items-center gap-1.5"
                    >
                      {isCopied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5 text-slate-500" />}
                      <span>{isCopied ? 'Copied!' : 'Copy Code'}</span>
                    </button>
                    <button
                      onClick={() => handleDownloadFile(selectedCodeFile)}
                      className="px-3 py-1.5 text-xs font-medium bg-sky-900 text-white rounded hover:bg-sky-800 transition-colors shadow-xs flex items-center gap-1.5"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>Download</span>
                    </button>
                  </div>
                </div>

                {/* File Description */}
                <div className="px-4 py-2.5 bg-sky-50/50 border-b border-sky-100 text-xs text-sky-900 flex items-center gap-2">
                  <span className="font-semibold shrink-0">Module Role:</span>
                  <span>{selectedCodeFile.description}</span>
                </div>

                {/* Code Content with Line Numbers */}
                <div className="p-4 bg-slate-950 text-slate-100 font-mono text-xs overflow-x-auto leading-relaxed max-h-[640px] overflow-y-auto">
                  <table className="w-full text-left">
                    <tbody>
                      {selectedCodeFile.content.split('\n').map((line, idx) => (
                        <tr key={idx} className="hover:bg-slate-900/60">
                          <td className="pr-4 py-0.5 select-none text-slate-600 text-right w-10 font-mono text-[11px] tabular-nums">
                            {idx + 1}
                          </td>
                          <td className="py-0.5 font-mono whitespace-pre text-slate-200">
                            {line || ' '}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* MODULE G: PROJECT DOCUMENTATION & SPECIFICATIONS */}
        {/* ============================================================ */}
        {activeTab === 'docs' && (
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h2 className="text-xl font-bold text-slate-900">Project Documentation & Technical Specifications</h2>
                <p className="text-xs text-slate-500">System architecture, data integrity workflows, and clinical evaluation specifications.</p>
              </div>

              {/* Sub-Tabs */}
              <div className="flex items-center gap-1 bg-white p-1 border border-slate-200 rounded-md">
                {[
                  { id: 'architecture', label: 'Architecture & Workflow' },
                  { id: 'methodology', label: 'Clinical Methodology & Limitations' },
                  { id: 'report', label: 'Export Report' },
                ].map(sub => (
                  <button
                    key={sub.id}
                    onClick={() => setDocSubTab(sub.id as any)}
                    className={`px-3 py-1 text-xs rounded transition-colors whitespace-nowrap ${
                      docSubTab === sub.id ? 'bg-sky-900 text-white font-medium' : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {sub.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Architecture Sub-Tab */}
            {docSubTab === 'architecture' && (
              <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-xs space-y-6">
                <div>
                  <h3 className="text-base font-semibold text-slate-900 mb-2">1. End-to-End System Workflow</h3>
                  <div className="p-4 bg-slate-900 text-slate-100 rounded-lg font-mono text-xs leading-relaxed overflow-x-auto">
                    <pre>{`Raw UCI Cleveland Records (303 rows)
       │
       ▼
[train_model.py] Preprocessing Pipeline (Leakage-Safe via ColumnTransformer)
       ├─ Numerical Scaling: StandardScaler (age, trestbps, chol, thalach, oldpeak)
       └─ Categorical Encoding: OneHotEncoder (sex, cp, fbs, restecg, exang, slope, ca, thal)
       │
       ▼
Stratified 5-Fold Cross-Validation Tournament
       ├─ Logistic Regression (L2 Regularized)  -->  Selected (89.3% Recall, 0.912 AUC)
       ├─ Decision Tree (Constrained Depth)
       ├─ Random Forest (Ensemble Averaging)
       └─ Gradient Boosting (Residual Loss)
       │
       ▼
Clinical Selection Strategy: 0.50*Recall + 0.30*ROC_AUC + 0.20*F1
       │
       ▼
Serialized Artifacts (models/, outputs/) & Interactive Intelligence UI`}</pre>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                      Dataset Attribution & Clinical Grounding
                    </h4>
                    <p className="text-xs text-slate-600 leading-relaxed">
                      Sourced from the canonical <strong>Cleveland Clinic Foundation</strong> cohort documented by Dr. Robert Detrano et al. Binarized at the 50% stenosis angiographic demarcation threshold. Missing values in fluoroscopy vessels and thallium scintigraphy are imputed using robust median values.
                    </p>
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                      Zero Data Leakage Invariant
                    </h4>
                    <p className="text-xs text-slate-600 leading-relaxed">
                      All feature transformations are encapsulated inside \`sklearn.compose.ColumnTransformer\` coupled directly to the estimator pipeline. No global normalization or imputation statistics leak from test splits into training folds.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Methodology & Limitations Sub-Tab */}
            {docSubTab === 'methodology' && (
              <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-xs space-y-5 text-xs text-slate-700 leading-relaxed">
                <div>
                  <h3 className="text-base font-semibold text-slate-900 mb-2">Clinical Methodology & Governance</h3>
                  <p className="text-slate-600">
                    Coronary Artery Disease (CAD) screening mandates asymmetric loss functions. In triage evaluation, a False Negative presents fatal clinical risk (unrecognized ischemia with deferred intervention), whereas a False Positive prompts non-invasive confirmatory testing.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                  <div className="p-4 bg-slate-50 border border-slate-200 rounded-md">
                    <h4 className="font-semibold text-slate-900 mb-1">Model Selection Criteria</h4>
                    <p className="text-slate-600">
                      Standard accuracy is replaced with a weighted clinical utility formulation:
                      <span className="font-mono text-sky-950 block my-1 font-semibold">
                        Score = 0.50*Recall + 0.30*ROC-AUC + 0.20*F1
                      </span>
                      L2-regularized Logistic Regression achieved a composite score of 0.890 with 89.3% recall on held-out test records.
                    </p>
                  </div>
                  <div className="p-4 bg-slate-50 border border-slate-200 rounded-md">
                    <h4 className="font-semibold text-slate-900 mb-1">Data Leakage Prevention</h4>
                    <p className="text-slate-600">
                      Z-score normalization and one-hot encoding matrices are computed strictly inside training folds of each StratifiedKFold split, ensuring test evaluation reflects true out-of-sample generalization.
                    </p>
                  </div>
                </div>

                <div className="p-4 bg-amber-50/80 border border-amber-200 rounded-md text-amber-950">
                  <h4 className="font-semibold text-amber-900 mb-1">Scope & Ethical Boundaries</h4>
                  <p>
                    Outputs represent observational statistical likelihoods from historic research cohorts and do not replace physical examination, cardiac catheterization, or licensed cardiologist diagnosis.
                  </p>
                </div>
              </div>
            )}

            {/* Report Sub-Tab */}
            {docSubTab === 'report' && (
              <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-xs space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-base font-semibold text-slate-900">Exportable Clinical Evaluation Report</h3>
                    <p className="text-xs text-slate-500">Generate a timestamped, JSON-formatted clinical audit payload.</p>
                  </div>
                  <button
                    onClick={handleDownloadReport}
                    className="px-4 py-2 text-xs font-semibold text-white bg-sky-900 rounded-md hover:bg-sky-800 transition-colors shadow-xs flex items-center gap-1.5"
                  >
                    <Download className="w-4 h-4" />
                    <span>Download JSON Report</span>
                  </button>
                </div>

                <div className="p-4 bg-slate-900 text-slate-100 rounded-lg font-mono text-xs leading-relaxed overflow-x-auto">
                  <pre>{JSON.stringify({
                    project: "CardioPredict AI",
                    track: "Machine Learning & Clinical Data Engineering Capstone",
                    cohort_spec: "Cleveland Clinic Foundation (UCI)",
                    selected_model: "Logistic Regression (L2 Regularized)",
                    held_out_metrics: {
                      accuracy: "85.25%",
                      sensitivity_recall: "89.29%",
                      specificity: "81.82%",
                      precision: "80.65%",
                      f1_score: "0.8475",
                      roc_auc: "0.9123"
                    },
                    selection_criteria: "Weighted Clinical Score: 0.50*Recall + 0.30*ROC_AUC + 0.20*F1 = 0.8896"
                  }, null, 2)}</pre>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-auto border-t border-slate-200 bg-white py-4 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div>
            <span className="font-semibold text-slate-700">CardioPredict AI</span>
            <span aria-hidden="true" className="mx-2">·</span>
            <span>Machine Learning & Clinical Data Engineering Capstone</span>
            <span aria-hidden="true" className="mx-2">·</span>
            <span>Educational Research License</span>
          </div>
          <div className="font-mono text-[11px]">
            Cohort N=303 · Seed 42 · Scikit-Learn Architecture
          </div>
        </div>
      </footer>
    </div>
  );
}
