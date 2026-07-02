"""Tests for training_module."""

import os
import sys
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from training_module import (
    train_classification_models, train_regression_models,
    evaluate_classification, evaluate_regression
)
from exceptions import TrainingError


class TestClassification:
    @pytest.fixture
    def classification_data(self):
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        return X[:80], X[80:], y[:80], y[80:]

    def test_trains_all_models(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        results = train_classification_models(X_train, y_train, X_test, y_test)
        successful = [r for r in results if not r.failed]
        assert len(successful) >= 7  # 8 without XGBoost

    def test_sorted_by_accuracy(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        results = train_classification_models(X_train, y_train, X_test, y_test)
        successful = [r for r in results if not r.failed]
        accuracies = [r.metrics['accuracy'] for r in successful]
        assert accuracies == sorted(accuracies, reverse=True)

    def test_metrics_in_range(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        results = train_classification_models(X_train, y_train, X_test, y_test)
        for r in results:
            if not r.failed:
                assert 0 <= r.metrics['accuracy'] <= 1
                assert 0 <= r.metrics['precision'] <= 1
                assert 0 <= r.metrics['recall'] <= 1
                assert 0 <= r.metrics['f1'] <= 1


class TestRegression:
    @pytest.fixture
    def regression_data(self):
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        return X[:80], X[80:], y[:80], y[80:]

    def test_trains_all_models(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        results = train_regression_models(X_train, y_train, X_test, y_test)
        successful = [r for r in results if not r.failed]
        assert len(successful) >= 6

    def test_sorted_by_r2(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        results = train_regression_models(X_train, y_train, X_test, y_test)
        successful = [r for r in results if not r.failed]
        r2s = [r.metrics['r2'] for r in successful]
        assert r2s == sorted(r2s, reverse=True)

    def test_metrics_valid(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        results = train_regression_models(X_train, y_train, X_test, y_test)
        for r in results:
            if not r.failed:
                assert r.metrics['mae'] >= 0
                assert r.metrics['rmse'] >= 0
