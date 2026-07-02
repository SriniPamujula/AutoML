"""Results compilation module — builds comparison tables and charts."""

import os
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from training_module import ModelResult

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResults:
    """Container for compiled results."""

    comparison_table: pd.DataFrame      # Models sorted by primary metric
    best_model_name: str
    best_model_metrics: dict[str, float]
    sample_predictions: pd.DataFrame    # Sample predictions from best model
    chart_path: str                     # Path to bar chart image


def compile_results(
    model_results: list[ModelResult],
    task_type: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str,
    sample_size: int = 10,
) -> ComparisonResults:
    """Build comparison table, identify best model, generate chart, get sample predictions.

    Args:
        model_results: List of ModelResult from training.
        task_type: 'classification' or 'regression'.
        X_test: Test features.
        y_test: Test targets.
        output_dir: Directory to save chart image.
        sample_size: Number of sample predictions to show.

    Returns:
        ComparisonResults with all compiled data.
    """
    successful = [r for r in model_results if not r.failed]

    if not successful:
        raise ValueError('No successful models to compile results from.')

    # Determine primary metric
    primary_metric = 'accuracy' if task_type == 'classification' else 'r2'

    # Build comparison table
    rows = []
    for result in successful:
        row = {'Model': result.name}
        row.update(result.metrics)
        rows.append(row)

    comparison_table = pd.DataFrame(rows)
    comparison_table = comparison_table.sort_values(by=primary_metric, ascending=False).reset_index(drop=True)

    # Add failed models at the bottom
    for result in model_results:
        if result.failed:
            row = {'Model': f'{result.name} (failed)'}
            comparison_table = pd.concat([comparison_table, pd.DataFrame([row])], ignore_index=True)

    # Identify best model
    best = successful[0]
    best_model_name = best.name
    best_model_metrics = best.metrics

    # Generate sample predictions
    actual_sample_size = min(sample_size, len(y_test))
    sample_indices = np.arange(actual_sample_size)

    sample_predictions = pd.DataFrame({
        'Row': sample_indices + 1,
        'Actual': y_test[sample_indices],
        'Predicted': best.predictions[sample_indices] if len(best.predictions) > 0 else [],
    })

    # Add match indicator
    sample_predictions['Match'] = sample_predictions['Actual'] == sample_predictions['Predicted']
    sample_predictions['Match'] = sample_predictions['Match'].map({True: '✓', False: '✗'})

    # Generate comparison chart
    os.makedirs(output_dir, exist_ok=True)
    from visualization_module import generate_comparison_chart as gen_chart
    primary_label = 'Accuracy' if task_type == 'classification' else 'R² Score'
    names = [r.name for r in successful]
    scores = [r.metrics.get(primary_metric, 0) for r in successful]
    chart_path = gen_chart(names, scores, primary_label, os.path.join(output_dir, 'model_comparison.png'))

    logger.info(f'Results compiled: best model = {best_model_name} ({primary_metric}={best_model_metrics.get(primary_metric, 0):.4f})')

    return ComparisonResults(
        comparison_table=comparison_table,
        best_model_name=best_model_name,
        best_model_metrics=best_model_metrics,
        sample_predictions=sample_predictions,
        chart_path=chart_path,
    )
