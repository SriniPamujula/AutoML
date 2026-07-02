"""AutoML Web Platform — Flask application entry point."""

import os
import logging

import pandas as pd
from flask import Flask, render_template, request

from config import Config
from exceptions import ValidationError, PreprocessingError, TrainingError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(config_class=Config):
    """Application factory."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure required directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['GENERATED_IMAGES_FOLDER'], exist_ok=True)

    @app.route('/', methods=['GET'])
    def index():
        """Home page with upload form and sample datasets."""
        from content.samples import SAMPLE_DATASETS
        return render_template('index.html', samples=SAMPLE_DATASETS)

    @app.route('/sample/<sample_id>', methods=['GET'])
    def run_sample(sample_id):
        """Run analysis on a pre-configured sample dataset."""
        from content.samples import SAMPLE_DATASETS
        from upload_handler import detect_task_type
        from eda_module import analyze as run_eda
        from visualization_module import generate_boxplot, generate_heatmap, generate_countplots
        from wrangling_module import preprocess
        from training_module import train_classification_models, train_regression_models
        from results_module import compile_results

        # Find the sample
        sample = next((s for s in SAMPLE_DATASETS if s['id'] == sample_id), None)
        if not sample:
            return render_template('error.html', error_message='Sample dataset not found.')

        file_path = os.path.join(os.path.dirname(__file__), 'Data', sample['file'])
        if not os.path.exists(file_path):
            return render_template('error.html', error_message='Sample data file missing on server.')

        target_column = sample['target']

        try:
            df = pd.read_csv(file_path)
            filename = sample['file']
            output_dir = app.config['GENERATED_IMAGES_FOLDER']

            task_type = detect_task_type(df[target_column])
            eda_results = run_eda(df, target_column)

            df_numeric = df.select_dtypes(include=['number'])
            df_categorical = df.select_dtypes(include=['object', 'category', 'bool'])

            boxplot_path = ''
            if not df_numeric.empty:
                boxplot_path = generate_boxplot(df_numeric, os.path.join(output_dir, 'boxplot.png'))

            heatmap_path = ''
            if df_numeric.shape[1] >= 2:
                heatmap_path = generate_heatmap(df_numeric, target_column, os.path.join(output_dir, 'heatmap.png'))

            countplot_paths = []
            if not df_categorical.empty:
                countplot_paths = generate_countplots(df_categorical, output_dir)

            preprocessed = preprocess(df, target_column)

            if task_type == 'classification':
                model_results = train_classification_models(
                    preprocessed.X_train, preprocessed.y_train,
                    preprocessed.X_test, preprocessed.y_test
                )
            else:
                model_results = train_regression_models(
                    preprocessed.X_train, preprocessed.y_train,
                    preprocessed.X_test, preprocessed.y_test
                )

            comparison = compile_results(
                model_results, task_type,
                preprocessed.X_test, preprocessed.y_test,
                output_dir
            )

            head_html = eda_results.head.to_html(classes='table table-sm table-dark-custom', index=False)
            describe_html = eda_results.describe.to_html(classes='table table-sm table-dark-custom')
            comparison_html = comparison.comparison_table.to_html(classes='table table-sm table-dark-custom', index=False, float_format='%.4f')
            predictions_html = comparison.sample_predictions.to_html(classes='table table-sm table-dark-custom', index=False)

            def read_plotly_html(path):
                if not path or not os.path.exists(path):
                    return ''
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()

            return render_template(
                'results.html',
                filename=filename,
                target_column=target_column,
                task_type=task_type,
                eda=eda_results,
                total_nulls=int(eda_results.null_counts.sum()),
                duplicates_removed=preprocessed.duplicates_removed,
                head_html=head_html,
                describe_html=describe_html,
                boxplot_html=read_plotly_html(boxplot_path),
                heatmap_html=read_plotly_html(heatmap_path),
                countplot_htmls=[read_plotly_html(p) for p in countplot_paths],
                best_model_name=comparison.best_model_name,
                best_metrics=comparison.best_model_metrics,
                chart_html=read_plotly_html(comparison.chart_path),
                comparison_html=comparison_html,
                predictions_html=predictions_html,
            )

        except (ValidationError, PreprocessingError, TrainingError) as e:
            return render_template('error.html', error_message=str(e))
        except Exception as e:
            logger.exception(f'Error running sample {sample_id}: {e}')
            return render_template('error.html', error_message='An error occurred. Please try again.')

    @app.route('/upload', methods=['POST'])
    def upload():
        """Handle file upload, validate, and show column selection."""
        from upload_handler import validate_upload, save_upload, extract_columns

        file = request.files.get('file')

        is_valid, error_msg = validate_upload(file)
        if not is_valid:
            logger.warning(f'Upload validation failed: {error_msg}')
            return render_template('error.html', error_message=error_msg)

        try:
            file_path = save_upload(file, app.config['UPLOAD_FOLDER'])
            logger.info(f'File saved: {file_path}')
        except ValidationError as e:
            logger.warning(f'Save failed: {e}')
            return render_template('error.html', error_message=str(e))

        try:
            columns = extract_columns(file_path)
        except ValidationError as e:
            logger.warning(f'Column extraction failed: {e}')
            return render_template('error.html', error_message=str(e))

        filename = os.path.basename(file_path)
        return render_template(
            'select_target.html',
            columns=columns,
            file_path=file_path,
            filename=filename
        )

    @app.route('/analyze', methods=['POST'])
    def analyze():
        """Run full ML pipeline: EDA → Viz → Wrangle → Train → Results."""
        from upload_handler import detect_task_type
        from eda_module import analyze as run_eda
        from visualization_module import generate_boxplot, generate_heatmap, generate_countplots
        from wrangling_module import preprocess
        from training_module import train_classification_models, train_regression_models
        from results_module import compile_results

        file_path = request.form.get('file_path', '')
        target_column = request.form.get('target_column', '')

        if not file_path or not os.path.exists(file_path):
            return render_template('error.html', error_message='File not found. Please upload again.')

        if not target_column:
            return render_template('error.html', error_message='Please select a target column.')

        try:
            # Read the CSV
            df = pd.read_csv(file_path)
            filename = os.path.basename(file_path)
            output_dir = app.config['GENERATED_IMAGES_FOLDER']

            # Determine task type
            task_type = detect_task_type(df[target_column])
            logger.info(f'Task type: {task_type} for target: {target_column}')

            # Step 1: EDA
            eda_results = run_eda(df, target_column)

            # Step 2: Visualizations
            df_numeric = df.select_dtypes(include=['number'])
            df_categorical = df.select_dtypes(include=['object', 'category', 'bool'])

            boxplot_path = ''
            if not df_numeric.empty:
                boxplot_path = generate_boxplot(df_numeric, os.path.join(output_dir, 'boxplot.png'))

            heatmap_path = ''
            if df_numeric.shape[1] >= 2:
                heatmap_path = generate_heatmap(
                    df_numeric, target_column, os.path.join(output_dir, 'heatmap.png')
                )

            countplot_paths = []
            if not df_categorical.empty:
                countplot_paths = generate_countplots(df_categorical, output_dir)

            # Step 3: Preprocessing
            preprocessed = preprocess(df, target_column)

            # Step 4: Model training
            if task_type == 'classification':
                model_results = train_classification_models(
                    preprocessed.X_train, preprocessed.y_train,
                    preprocessed.X_test, preprocessed.y_test
                )
            else:
                model_results = train_regression_models(
                    preprocessed.X_train, preprocessed.y_train,
                    preprocessed.X_test, preprocessed.y_test
                )

            # Step 5: Compile results
            comparison = compile_results(
                model_results, task_type,
                preprocessed.X_test, preprocessed.y_test,
                output_dir
            )

            # Prepare template data
            head_html = eda_results.head.to_html(
                classes='table table-sm table-dark-custom', index=False
            )
            describe_html = eda_results.describe.to_html(
                classes='table table-sm table-dark-custom'
            )
            comparison_html = comparison.comparison_table.to_html(
                classes='table table-sm table-dark-custom', index=False,
                float_format='%.4f'
            )
            predictions_html = comparison.sample_predictions.to_html(
                classes='table table-sm table-dark-custom', index=False
            )

            # Read Plotly HTML files as inline content
            def read_plotly_html(path):
                if not path or not os.path.exists(path):
                    return ''
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()

            return render_template(
                'results.html',
                filename=filename,
                target_column=target_column,
                task_type=task_type,
                eda=eda_results,
                total_nulls=int(eda_results.null_counts.sum()),
                duplicates_removed=preprocessed.duplicates_removed,
                head_html=head_html,
                describe_html=describe_html,
                boxplot_html=read_plotly_html(boxplot_path),
                heatmap_html=read_plotly_html(heatmap_path),
                countplot_htmls=[read_plotly_html(p) for p in countplot_paths],
                best_model_name=comparison.best_model_name,
                best_metrics=comparison.best_model_metrics,
                chart_html=read_plotly_html(comparison.chart_path),
                comparison_html=comparison_html,
                predictions_html=predictions_html,
            )

        except ValidationError as e:
            logger.warning(f'Validation error: {e}')
            return render_template('error.html', error_message=str(e))
        except PreprocessingError as e:
            logger.warning(f'Preprocessing error: {e}')
            return render_template('error.html', error_message=str(e))
        except TrainingError as e:
            logger.error(f'Training error: {e}')
            return render_template('error.html', error_message=str(e))
        except Exception as e:
            logger.exception(f'Unexpected error during analysis: {e}')
            return render_template('error.html', error_message='An unexpected error occurred during analysis. Please try again.')

    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle file too large error."""
        return render_template('error.html', error_message='File exceeds the 50 MB maximum. Please upload a smaller file.'), 413

    @app.errorhandler(404)
    def not_found(error):
        """Handle 404."""
        return render_template('error.html', error_message='Page not found.'), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500."""
        logger.exception('Internal server error')
        return render_template('error.html', error_message='An internal error occurred. Please try again.'), 500

    return app


# Create the application instance
app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
