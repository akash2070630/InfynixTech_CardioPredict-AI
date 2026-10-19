"""
Unit and Integration Tests for CardioPredict AI Pipeline
"""

import os
import pytest
import pandas as pd
import numpy as np
import joblib

from train_model import (
    download_or_load_dataset,
    build_preprocessor,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    MODELS_DIR,
    OUTPUTS_DIR
)


def test_dataset_loading_and_shape():
    """Ensure dataset loads properly and has expected features and records."""
    df = download_or_load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 300
    assert "target" in df.columns
    for col in NUMERICAL_FEATURES:
        assert col in df.columns
    for col in CATEGORICAL_FEATURES:
        assert col in df.columns


def test_no_unhandled_nulls():
    """Verify that there are no unhandled NaN values in the processed clinical dataset."""
    df = download_or_load_dataset()
    assert df[NUMERICAL_FEATURES].isna().sum().sum() == 0
    assert df[CATEGORICAL_FEATURES].isna().sum().sum() == 0


def test_preprocessor_pipeline_transformation():
    """Test that the Scikit-learn ColumnTransformer transforms inputs without dimension mismatch."""
    df = download_or_load_dataset()
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] > len(NUMERICAL_FEATURES)


def test_model_artifact_loading_and_inference():
    """Ensure trained model artifact exists, loads smoothly, and produces valid predictions."""
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    if not os.path.exists(model_path):
        pytest.skip("Model artifact not yet trained. Run train_model.py first.")

    pipeline = joblib.load(model_path)
    sample_patient = pd.DataFrame([{
        "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 240,
        "fbs": 0, "restecg": 0, "thalach": 130, "exang": 1,
        "oldpeak": 2.2, "slope": 2, "ca": 2.0, "thal": 7.0
    }])

    pred = pipeline.predict(sample_patient)
    assert pred[0] in [0, 1]

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(sample_patient)
        assert proba.shape == (1, 2)
        assert 0.0 <= proba[0, 1] <= 1.0
