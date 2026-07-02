"""Model training and evaluation module."""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error,
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from exceptions import TrainingError

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Container for a single model's training result."""

    name: str
    model: Any
    metrics: dict[str, float] = field(default_factory=dict)
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    failed: bool = False
    error_message: str = ""


def evaluate_classification(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Compute classification metrics.

    Args:
        model: Trained classifier.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dict with accuracy, precision, recall, f1 (all in [0,1]).
    """
    predictions = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
        'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
        'f1': f1_score(y_test, predictions, average='weighted', zero_division=0),
    }


def evaluate_regression(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Compute regression metrics.

    Args:
        model: Trained regressor.
        X_test: Test features.
        y_test: Test targets.

    Returns:
        Dict with r2, mae, rmse.
    """
    predictions = model.predict(X_test)
    return {
        'r2': r2_score(y_test, predictions),
        'mae': mean_absolute_error(y_test, predictions),
        'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
    }


def train_classification_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> list[ModelResult]:
    """Train all classification models.

    Models: Logistic Regression, Decision Tree, Random Forest, AdaBoost,
    KNN, SVM, Naive Bayes.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        List of ModelResult sorted by accuracy descending.

    Raises:
        TrainingError: If all models fail.
    """
    classifiers = [
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
        ('Decision Tree', DecisionTreeClassifier(random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('AdaBoost', AdaBoostClassifier(n_estimators=50, random_state=42, algorithm='SAMME')),
        ('K-Nearest Neighbors', KNeighborsClassifier()),
        ('Support Vector Machine', SVC(random_state=42)),
        ('Naive Bayes', GaussianNB()),
    ]

    if XGBOOST_AVAILABLE:
        classifiers.append(('XGBoost', XGBClassifier(
            n_estimators=100, random_state=42, use_label_encoder=False,
            eval_metric='logloss', verbosity=0,
        )))

    results = []
    for name, model in classifiers:
        try:
            logger.info(f'Training classifier: {name}')
            model.fit(X_train, y_train)
            metrics = evaluate_classification(model, X_test, y_test)
            predictions = model.predict(X_test)
            results.append(ModelResult(
                name=name,
                model=model,
                metrics=metrics,
                predictions=predictions,
            ))
            logger.info(f'  {name}: accuracy={metrics["accuracy"]:.4f}')
        except Exception as e:
            logger.warning(f'  {name} failed: {type(e).__name__}: {e}')
            results.append(ModelResult(
                name=name,
                model=None,
                failed=True,
                error_message=f'{type(e).__name__}: {e}',
            ))

    successful = [r for r in results if not r.failed]
    if not successful:
        error_details = '; '.join(f'{r.name}: {r.error_message}' for r in results)
        raise TrainingError(f'All models failed to train. Reasons: {error_details}')

    # Sort by accuracy descending
    successful.sort(key=lambda r: r.metrics.get('accuracy', 0), reverse=True)
    failed = [r for r in results if r.failed]

    return successful + failed


def train_regression_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> list[ModelResult]:
    """Train all regression models.

    Models: Linear Regression, Decision Tree, Random Forest, Gradient Boosting,
    SVR, KNN Regressor.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.

    Returns:
        List of ModelResult sorted by R-squared descending.

    Raises:
        TrainingError: If all models fail.
    """
    regressors = [
        ('Linear Regression', LinearRegression()),
        ('Decision Tree', DecisionTreeRegressor(random_state=42)),
        ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('Support Vector Regressor', SVR()),
        ('K-Nearest Neighbors', KNeighborsRegressor()),
    ]

    if XGBOOST_AVAILABLE:
        regressors.append(('XGBoost', XGBRegressor(
            n_estimators=100, random_state=42, verbosity=0,
        )))

    results = []
    for name, model in regressors:
        try:
            logger.info(f'Training regressor: {name}')
            model.fit(X_train, y_train)
            metrics = evaluate_regression(model, X_test, y_test)
            predictions = model.predict(X_test)
            results.append(ModelResult(
                name=name,
                model=model,
                metrics=metrics,
                predictions=predictions,
            ))
            logger.info(f'  {name}: R²={metrics["r2"]:.4f}')
        except Exception as e:
            logger.warning(f'  {name} failed: {type(e).__name__}: {e}')
            results.append(ModelResult(
                name=name,
                model=None,
                failed=True,
                error_message=f'{type(e).__name__}: {e}',
            ))

    successful = [r for r in results if not r.failed]
    if not successful:
        error_details = '; '.join(f'{r.name}: {r.error_message}' for r in results)
        raise TrainingError(f'All models failed to train. Reasons: {error_details}')

    # Sort by R² descending
    successful.sort(key=lambda r: r.metrics.get('r2', float('-inf')), reverse=True)
    failed = [r for r in results if r.failed]

    return successful + failed
