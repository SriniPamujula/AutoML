# Implementation Plan: AutoML Web Platform

## Overview

This plan implements a modular Flask-based AutoML web application. The existing prototype is restructured into clean, separated modules with modern scikit-learn APIs, proper file upload handling, comprehensive error handling, and production deployment configuration for EC2 at automl.betterai.guru.

## Tasks

- [ ] 1. Set up project structure, configuration, and base application
  - [ ] 1.1 Create configuration module and project scaffolding
    - Create `config.py` with `Config` class containing SECRET_KEY, UPLOAD_FOLDER, GENERATED_IMAGES_FOLDER, MAX_CONTENT_LENGTH (50MB), ALLOWED_EXTENSIONS, TEST_SIZE, RANDOM_STATE, MIN_ROWS
    - Create `uploads/` and `static/generated/` directories with `.gitkeep` files
    - Create custom exception classes (`ValidationError`, `PreprocessingError`, `TrainingError`) in a shared `exceptions.py` module
    - _Requirements: 10.3, 10.6, 11.5_

  - [ ] 1.2 Create base Flask application and route structure
    - Create `app.py` with Flask app factory pattern, register configuration, set up logging
    - Create `templates/base.html` with Bootstrap layout, navigation, and progress indicator placeholder
    - Create `templates/index.html` with file upload form accepting CSV files
    - Create `templates/error.html` for displaying error messages
    - Wire up the `/` GET route to render the upload form
    - _Requirements: 1.1, 9.5, 10.4_

  - [ ] 1.3 Create `.gitignore` and `requirements.txt`
    - Create `.gitignore` excluding `uploads/`, `static/generated/`, `venv/`, `__pycache__/`, `*.pyc`, `.env`
    - Create `requirements.txt` with pinned versions: Flask, gunicorn, pandas>=2.0, scikit-learn>=1.3, matplotlib>=3.7, seaborn, numpy, hypothesis, pytest
    - _Requirements: 11.1, 11.7, 10.5_

- [ ] 2. Implement file upload handling
  - [ ] 2.1 Implement upload_handler.py module
    - Implement `validate_upload(file)` — check file is present, has `.csv` extension, and is within size limit; return `(is_valid, error_message)` tuple
    - Implement `save_upload(file, upload_dir)` — use `werkzeug.utils.secure_filename` to sanitize filename, save to uploads directory, return file path
    - Implement `extract_columns(file_path)` — read CSV with pandas, return list of column names
    - Implement `detect_task_type(series)` — return "classification" if categorical or ≤10 unique numeric values, else "regression"
    - Raise `ValidationError` with specific messages for each failure case
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 2.1, 2.3, 2.4, 2.5_

  - [ ] 2.2 Create upload and target selection routes
    - Implement `/upload` POST route: call validate_upload, save_upload, extract_columns; render column selection page on success, render error page on failure
    - Create `templates/select_target.html` with dropdown of columns and submit button
    - Implement `/select-target` GET route and `/analyze` POST route skeleton (to be completed in later tasks)
    - Handle Flask's `RequestEntityTooLarge` exception for files >50MB
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2_

  - [ ]* 2.3 Write property tests for upload_handler
    - **Property 1: Secure filename sanitization** — for any uploaded filename string (including path traversal sequences, special characters, unicode), `save_upload` produces a path contained within uploads directory
    - **Validates: Requirements 1.2**
    - **Property 2: Non-CSV file rejection** — for any filename with extension not `.csv`, `validate_upload` returns `is_valid=False`
    - **Validates: Requirements 1.3**

  - [ ]* 2.4 Write property test for task type detection
    - **Property 3: Task type detection** — for any Series with categorical values or ≤10 unique numeric values return "classification"; for >10 unique numeric values return "regression"
    - **Validates: Requirements 2.3, 2.4**

- [ ] 3. Implement EDA module
  - [ ] 3.1 Implement eda_module.py
    - Define `EDAResults` dataclass with fields: head, shape, describe, null_counts, dtypes, unique_counts
    - Implement `analyze(dataframe, target_col)` — compute all EDA metrics and return `EDAResults`
    - Validate that target column exists and is not all-null (raise `ValidationError` if so)
    - Validate minimum row count (≥10 rows, raise `ValidationError` if not)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 9.1, 2.5_

  - [ ]* 3.2 Write property tests for EDA module
    - **Property 4: EDA results correctness** — for any valid DataFrame, `analyze` returns correct shape, null_counts, dtypes, unique_counts, and head rows
    - **Validates: Requirements 3.1, 3.2, 3.4, 3.5, 3.6**
    - **Property 5: Descriptive statistics completeness** — for any DataFrame with at least one numeric column, `describe` contains mean, std, min, max, and quartiles for every numeric column
    - **Validates: Requirements 3.3**

- [ ] 4. Implement visualization module
  - [ ] 4.1 Implement visualization_module.py
    - Implement `generate_boxplot(df_numeric, output_path)` — generate box plot of numeric features, save as PNG, return path
    - Implement `generate_heatmap(df_numeric, target_col, output_path, max_features=50)` — generate correlation heatmap; if >50 numeric columns, limit to top 50 most correlated with target
    - Implement `generate_countplots(df_categorical, output_dir)` — generate count plot for each categorical column, return list of image paths
    - Use matplotlib with `Agg` backend for headless rendering
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ]* 4.2 Write property test for heatmap feature limiting
    - **Property 6: Heatmap feature limiting** — for any DataFrame with >50 numeric columns, `generate_heatmap` uses only the 50 columns most correlated with target
    - **Validates: Requirements 4.5**

- [ ] 5. Implement data wrangling module
  - [ ] 5.1 Implement wrangling_module.py
    - Define `PreprocessedData` dataclass with fields: X_train, X_test, y_train, y_test, feature_names, dropped_columns, duplicates_removed
    - Implement `remove_null_columns(df)` — drop columns that are entirely null, return cleaned df and list of dropped column names
    - Implement `remove_duplicates(df)` — remove duplicate rows, return cleaned df and count removed
    - Implement `impute_numeric(df)` — use `sklearn.impute.SimpleImputer(strategy='median')` to fill missing numeric values
    - Implement `encode_categorical(df)` — label-encode binary columns, one-hot encode multi-category columns
    - Implement `scale_features(df)` — apply `StandardScaler` to all numeric features
    - Implement `preprocess(dataframe, target_col, test_size=0.2, random_state=42)` — orchestrate full pipeline, use `train_test_split`, return `PreprocessedData`
    - Raise `PreprocessingError` if zero features remain after preprocessing
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 9.4, 10.1, 10.2_

  - [ ]* 5.2 Write property tests for wrangling module
    - **Property 7: Null column removal** — for any DataFrame with all-null columns, `remove_null_columns` removes exactly those columns
    - **Validates: Requirements 5.1**
    - **Property 8: Duplicate row removal** — for any DataFrame with duplicates, `remove_duplicates` returns DataFrame with no duplicates and correct count
    - **Validates: Requirements 5.2**
    - **Property 9: Median imputation correctness** — for any numeric column with nulls, `impute_numeric` replaces nulls with column median, leaves non-null unchanged
    - **Validates: Requirements 5.3**
    - **Property 10: Categorical encoding strategy** — for binary columns produce label encoding, for multi-category produce one-hot encoding
    - **Validates: Requirements 5.4**
    - **Property 11: Standard scaling invariant** — after `scale_features`, each column has mean ≈ 0 and std ≈ 1
    - **Validates: Requirements 5.5**
    - **Property 12: Train/test split ratio and reproducibility** — split produces ~80/20 ratio and is deterministic with same seed
    - **Validates: Requirements 5.6**

- [ ] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement model training module
  - [ ] 7.1 Implement training_module.py for classification
    - Define `ModelResult` dataclass with fields: name, model, metrics, predictions, failed, error_message
    - Implement `train_classification_models(X_train, y_train, X_test, y_test)` — train Logistic Regression, Decision Tree, Random Forest, AdaBoost, KNN, SVM, Naive Bayes; catch convergence failures per model; return list of `ModelResult` sorted by accuracy descending
    - Implement `evaluate_classification(model, X_test, y_test)` — compute accuracy, precision (weighted), recall (weighted), F1 (weighted)
    - Raise `TrainingError` if all models fail
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 9.3, 10.2, 10.6_

  - [ ] 7.2 Implement training_module.py for regression
    - Implement `train_regression_models(X_train, y_train, X_test, y_test)` — train Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, SVR, KNN Regressor; catch convergence failures per model; return list of `ModelResult` sorted by R² descending
    - Implement `evaluate_regression(model, X_test, y_test)` — compute R², MAE, RMSE
    - Raise `TrainingError` if all models fail
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 9.3, 10.2, 10.6_

  - [ ]* 7.3 Write property test for model evaluation metrics
    - **Property 13: Model evaluation metrics validity** — for any trained classifier, metrics contain accuracy/precision/recall/f1 in [0,1]; for any regressor, metrics contain r2/mae/rmse with mae≥0 and rmse≥0
    - **Validates: Requirements 6.3, 7.3**

- [ ] 8. Implement results compilation module
  - [ ] 8.1 Implement results_module.py
    - Define `ComparisonResults` dataclass with fields: comparison_table, best_model_name, best_model_metrics, sample_predictions, chart_path
    - Implement `compile_results(model_results, task_type, X_test, y_test, output_dir)` — build comparison table sorted by primary metric (accuracy/R²) descending, identify best model, generate sample predictions from best model on test subset
    - Implement `generate_comparison_chart(model_results, task_type, output_path)` — generate bar chart comparing primary metric across models
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ]* 8.2 Write property tests for results module
    - **Property 14: Results table sorting invariant** — comparison table is sorted by primary metric descending with one row per model
    - **Validates: Requirements 8.1, 8.2**
    - **Property 15: Sample predictions length** — sample predictions row count matches requested sample size or test set size if smaller
    - **Validates: Requirements 8.5**

- [ ] 9. Wire full pipeline and create results template
  - [ ] 9.1 Complete the `/analyze` route and results template
    - Implement the full `/analyze` POST route: read saved CSV, validate target column, run EDA → visualization → wrangling → training → results
    - Determine task type via `detect_task_type` and route to classification or regression training
    - Create `templates/results.html` displaying: EDA summary (head, shape, stats, nulls, types, uniques), visualization images (boxplot, heatmap, countplots), model comparison table with best model highlighted, bar chart, sample predictions
    - Handle all module exceptions and render error template with appropriate messages
    - _Requirements: 2.2, 3.1-3.6, 4.4, 8.1-8.5, 9.1-9.4, 10.6_

  - [ ] 9.2 Implement progress indicator with server-sent events
    - Implement `/progress/<job_id>` GET route with SSE to stream pipeline stage updates
    - Add JavaScript in templates to display a progress indicator while pipeline executes
    - Stages: "Analyzing data...", "Generating visualizations...", "Preprocessing...", "Training models...", "Compiling results..."
    - _Requirements: 9.5_

- [ ] 10. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. Create deployment configuration and documentation
  - [ ] 11.1 Create deployment scripts and configuration
    - Create `deploy/automl.service` systemd unit file for gunicorn with 2 workers bound to 127.0.0.1:8000
    - Create `deploy/automl-nginx.conf` with HTTP→HTTPS redirect, SSL via Let's Encrypt, proxy_pass to gunicorn, static file serving, client_max_body_size 50M
    - Create `deploy/setup.sh` script: install system packages, create venv, install requirements, configure systemd, install certbot, obtain SSL cert, configure nginx
    - _Requirements: 11.2, 11.3, 11.8, 11.9_

  - [ ] 11.2 Create README and documentation
    - Create comprehensive `README.md` with: project purpose, prerequisites, local development setup, EC2 deployment guide (including DNS configuration for automl.betterai.guru subdomain), usage guide, example screenshots placeholders, and project structure overview
    - _Requirements: 11.4_

  - [ ]* 11.3 Write unit tests for wrangling and training modules
    - Create `tests/test_wrangling.py` with example-based tests: known CSV produces expected preprocessing output, all-null columns removed, duplicates removed, encoding verified
    - Create `tests/test_training.py` with example-based tests: known dataset produces valid metrics, single model failure handled gracefully, all-model failure raises TrainingError
    - Create `tests/test_upload_handler.py` with example-based tests: valid CSV accepted, non-CSV rejected, empty upload rejected
    - Create `tests/test_routes.py` with Flask test client integration tests for upload flow
    - _Requirements: 11.6_

- [ ] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document using Hypothesis
- Unit tests validate specific examples and edge cases using pytest
- The project uses Python with Flask, scikit-learn, pandas, matplotlib/seaborn
- All deprecated APIs (sklearn.preprocessing.Imputer, bare except, axis parameter) are replaced with modern equivalents
- No duplicate code between modules — each function is defined once

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.3"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["2.1", "3.1", "4.1", "5.1"] },
    { "id": 3, "tasks": ["2.2", "2.3", "2.4", "3.2", "4.2", "5.2"] },
    { "id": 4, "tasks": ["7.1", "7.2"] },
    { "id": 5, "tasks": ["7.3", "8.1"] },
    { "id": 6, "tasks": ["8.2", "9.1"] },
    { "id": 7, "tasks": ["9.2"] },
    { "id": 8, "tasks": ["11.1", "11.2", "11.3"] }
  ]
}
```
