"""Visualization module — generates interactive Plotly charts as HTML."""

import os
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# BetterAI theme colors
COLORS = {
    'bg': '#070B17',
    'surface': '#101827',
    'primary': '#7C3AED',
    'secondary': '#22D3EE',
    'accent': '#F97316',
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'text': '#F8FAFC',
    'muted': '#94A3B8',
    'grid': 'rgba(255,255,255,0.05)',
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS['surface'],
    plot_bgcolor=COLORS['bg'],
    font=dict(color=COLORS['text'], family='Inter, sans-serif'),
    margin=dict(l=40, r=20, t=40, b=40),
)


def generate_boxplot(df_numeric: pd.DataFrame, output_path: str) -> str:
    """Generate interactive box plot of numeric features.

    Args:
        df_numeric: DataFrame containing only numeric columns.
        output_path: Full path to save the HTML file.

    Returns:
        The output_path where the HTML was saved.
    """
    if df_numeric.empty or df_numeric.shape[1] == 0:
        logger.warning('No numeric columns for boxplot.')
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Normalize for comparison
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df_numeric),
        columns=df_numeric.columns
    )

    fig = go.Figure()
    for col in scaled.columns:
        fig.add_trace(go.Box(
            y=scaled[col],
            name=col,
            marker_color=COLORS['primary'],
            line_color=COLORS['secondary'],
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title='Feature Distribution (Normalized)',
        showlegend=False,
        xaxis=dict(gridcolor=COLORS['grid']),
        yaxis=dict(gridcolor=COLORS['grid']),
    )

    # Save as HTML
    html_path = output_path.replace('.png', '.html')
    fig.write_html(html_path, include_plotlyjs='cdn', full_html=False)
    logger.info(f'Boxplot saved: {html_path}')
    return html_path


def generate_heatmap(
    df_numeric: pd.DataFrame,
    target_col: str,
    output_path: str,
    max_features: int = 50
) -> str:
    """Generate interactive correlation heatmap.

    Args:
        df_numeric: DataFrame containing only numeric columns.
        target_col: The target column name (for feature selection if >50 cols).
        output_path: Full path to save the HTML file.
        max_features: Maximum features to include.

    Returns:
        The output_path where the HTML was saved.
    """
    if df_numeric.empty or df_numeric.shape[1] < 2:
        logger.warning('Not enough numeric columns for heatmap.')
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_plot = df_numeric.copy()

    # Limit to top N features if too many
    if df_plot.shape[1] > max_features:
        if target_col in df_plot.columns:
            correlations = df_plot.corr()[target_col].abs().sort_values(ascending=False)
            top_features = correlations.head(max_features).index.tolist()
            df_plot = df_plot[top_features]
            logger.info(f'Heatmap limited to top {max_features} features')
        else:
            df_plot = df_plot.iloc[:, :max_features]

    corr = df_plot.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr.values, 2) if corr.shape[0] <= 20 else None,
        texttemplate='%{text}' if corr.shape[0] <= 20 else None,
        textfont=dict(size=8),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title='Feature Correlations',
        height=max(400, corr.shape[0] * 25),
    )

    html_path = output_path.replace('.png', '.html')
    fig.write_html(html_path, include_plotlyjs='cdn', full_html=False)
    logger.info(f'Heatmap saved: {html_path}')
    return html_path


def generate_countplots(df_categorical: pd.DataFrame, output_dir: str) -> list[str]:
    """Generate interactive count plots for each categorical column.

    Args:
        df_categorical: DataFrame containing only categorical columns.
        output_dir: Directory to save the HTML files.

    Returns:
        List of HTML file paths.
    """
    if df_categorical.empty or df_categorical.shape[1] == 0:
        logger.warning('No categorical columns for countplots.')
        return []

    os.makedirs(output_dir, exist_ok=True)
    html_paths = []

    for column in df_categorical.columns:
        value_counts = df_categorical[column].value_counts().head(20)

        fig = go.Figure(data=go.Bar(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            marker_color=COLORS['secondary'],
        ))

        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f'Distribution: {column}',
            xaxis_title=column,
            yaxis_title='Count',
            xaxis=dict(gridcolor=COLORS['grid']),
            yaxis=dict(gridcolor=COLORS['grid']),
        )

        html_path = os.path.join(output_dir, f'countplot_{column}.html')
        fig.write_html(html_path, include_plotlyjs='cdn', full_html=False)
        html_paths.append(html_path)
        logger.info(f'Countplot saved: {html_path}')

    return html_paths


def generate_comparison_chart(model_names: list[str], scores: list[float], metric_label: str, output_path: str) -> str:
    """Generate model comparison bar chart.

    Args:
        model_names: List of model names.
        scores: List of scores (same order as names).
        metric_label: Label for the metric (e.g., 'Accuracy', 'R² Score').
        output_path: Full path to save the HTML file.

    Returns:
        The output_path where the HTML was saved.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sort by score descending
    sorted_pairs = sorted(zip(model_names, scores), key=lambda x: x[1], reverse=True)
    names = [p[0] for p in sorted_pairs]
    values = [p[1] for p in sorted_pairs]

    colors = [COLORS['primary'] if i == 0 else COLORS['secondary'] for i in range(len(names))]

    fig = go.Figure(data=go.Bar(
        y=names,
        x=values,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.4f}' for v in values],
        textposition='outside',
        textfont=dict(color=COLORS['text']),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f'Model Comparison — {metric_label}',
        xaxis_title=metric_label,
        yaxis=dict(autorange='reversed'),
        xaxis=dict(gridcolor=COLORS['grid'], range=[0, max(1.0, max(values) * 1.15)]),
        height=max(300, len(names) * 45),
    )

    html_path = output_path.replace('.png', '.html')
    fig.write_html(html_path, include_plotlyjs='cdn', full_html=False)
    logger.info(f'Comparison chart saved: {html_path}')
    return html_path
