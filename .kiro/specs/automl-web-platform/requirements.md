# Requirements Document

## Introduction

The AutoML Web Platform is a browser-based application that allows visitors to upload a CSV dataset and automatically receive a complete machine learning analysis. The platform performs exploratory data analysis, generates visualizations, preprocesses the data, trains multiple classification and regression models, evaluates model performance, and presents comparative results with the best model's predictions — all without requiring the visitor to write code or configure parameters.

## Glossary

- **Platform**: The AutoML Web Platform Flask application that serves the web interface and orchestrates the ML pipeline
- **Visitor**: A person who accesses the Platform through a web browser to analyze a CSV dataset
- **CSV_File**: A comma-separated values file uploaded by the Visitor containing tabular data
- **EDA_Module**: The component responsible for exploratory data analysis including statistics, null checks, and column type detection
- **Visualization_Module**: The component responsible for generating statistical plots including box plots, correlation heatmaps, and distribution plots
- **Wrangling_Module**: The component responsible for data preprocessing including null imputation, encoding, scaling, and duplicate removal
- **Training_Module**: The component responsible for training classification and regression models on preprocessed data
- **Results_Module**: The component responsible for evaluating trained models and presenting comparison metrics to the Visitor
- **Target_Column**: The column in the CSV_File that the Visitor selects as the prediction target
- **Pipeline**: The sequential execution of EDA, visualization, wrangling, training, and results presentation

## Requirements

### Requirement 1: CSV File Upload

**User Story:** As a Visitor, I want to upload any CSV file through the web interface, so that I can analyze my own dataset without hardcoded file restrictions.

#### Acceptance Criteria

1. WHEN the Visitor navigates to the Platform home page, THE Platform SHALL display a file upload form that accepts CSV files
2. WHEN the Visitor submits a CSV_File through the upload form, THE Platform SHALL save the file to a designated uploads directory with a secure filename
3. IF the Visitor submits a file that is not in CSV format, THEN THE Platform SHALL display an error message indicating only CSV files are accepted
4. IF the Visitor submits a CSV_File larger than 50 MB, THEN THE Platform SHALL display an error message indicating the file exceeds the maximum allowed size
5. IF the Visitor submits the upload form without selecting a file, THEN THE Platform SHALL display an error message prompting the Visitor to select a file

### Requirement 2: Target Column Selection

**User Story:** As a Visitor, I want to select which column in my dataset is the prediction target, so that the platform trains models to predict the correct variable.

#### Acceptance Criteria

1. WHEN the Platform receives a valid CSV_File, THE Platform SHALL parse the column names and display them to the Visitor for target selection
2. WHEN the Visitor selects a Target_Column, THE Platform SHALL use that column as the dependent variable for model training
3. WHEN the Target_Column contains categorical values with 10 or fewer unique categories, THE Platform SHALL classify the task as classification
4. WHEN the Target_Column contains continuous numeric values or more than 10 unique numeric values, THE Platform SHALL classify the task as regression
5. IF the Visitor selects a Target_Column that contains only null values, THEN THE Platform SHALL display an error message indicating the selected column has no valid data

### Requirement 3: Exploratory Data Analysis

**User Story:** As a Visitor, I want to see summary statistics and data quality information about my dataset, so that I can understand the data before model training begins.

#### Acceptance Criteria

1. WHEN the Visitor uploads a CSV_File and selects a Target_Column, THE EDA_Module SHALL compute and display the first five rows of the dataset
2. WHEN the EDA_Module processes the CSV_File, THE EDA_Module SHALL display the dataset shape including row count and column count
3. WHEN the EDA_Module processes the CSV_File, THE EDA_Module SHALL display descriptive statistics including mean, standard deviation, min, max, and quartiles for each numeric column
4. WHEN the EDA_Module processes the CSV_File, THE EDA_Module SHALL display the count of null values per column
5. WHEN the EDA_Module processes the CSV_File, THE EDA_Module SHALL display the data type of each column
6. WHEN the EDA_Module processes the CSV_File, THE EDA_Module SHALL display the number of unique values per column

### Requirement 4: Data Visualizations

**User Story:** As a Visitor, I want to see visual plots of my dataset, so that I can identify patterns, outliers, and correlations in the data.

#### Acceptance Criteria

1. WHEN the EDA_Module completes analysis, THE Visualization_Module SHALL generate a box plot showing the distribution and outliers of all numeric features
2. WHEN the EDA_Module completes analysis, THE Visualization_Module SHALL generate a correlation heatmap showing pairwise correlations between numeric features
3. WHEN the EDA_Module completes analysis, THE Visualization_Module SHALL generate distribution count plots for each categorical feature
4. THE Visualization_Module SHALL render all generated plots as images embedded in the results page presented to the Visitor
5. IF the dataset contains more than 50 numeric features, THEN THE Visualization_Module SHALL limit the heatmap to the 50 features most correlated with the Target_Column

### Requirement 5: Data Wrangling and Preprocessing

**User Story:** As a Visitor, I want the platform to automatically clean and preprocess my data, so that the models receive properly formatted input without manual intervention.

#### Acceptance Criteria

1. WHEN the Wrangling_Module receives the dataset, THE Wrangling_Module SHALL remove columns that contain only null values
2. WHEN the Wrangling_Module receives the dataset, THE Wrangling_Module SHALL remove rows that are complete duplicates
3. WHEN the Wrangling_Module encounters numeric columns with null values, THE Wrangling_Module SHALL impute missing values using the column median
4. WHEN the Wrangling_Module encounters categorical columns, THE Wrangling_Module SHALL encode binary categorical columns using label encoding and multi-category columns using one-hot encoding
5. WHEN the Wrangling_Module completes encoding, THE Wrangling_Module SHALL scale all numeric features using standard scaling with zero mean and unit variance
6. THE Wrangling_Module SHALL split the preprocessed dataset into training and testing sets using an 80/20 ratio with a fixed random seed for reproducibility

### Requirement 6: Classification Model Training

**User Story:** As a Visitor, I want multiple classification algorithms trained on my data, so that I can compare their performance and find the best classifier.

#### Acceptance Criteria

1. WHEN the task is classified as classification, THE Training_Module SHALL train all of the following models: Logistic Regression, Decision Tree, Random Forest, AdaBoost, K-Nearest Neighbors, Support Vector Machine, and Naive Bayes
2. WHEN the Training_Module trains each classification model, THE Training_Module SHALL use the training set produced by the Wrangling_Module
3. WHEN the Training_Module completes training of a classification model, THE Training_Module SHALL evaluate the model on the test set and record accuracy, precision, recall, and F1-score
4. IF a classification model fails to converge during training, THEN THE Training_Module SHALL skip that model and continue training remaining models

### Requirement 7: Regression Model Training

**User Story:** As a Visitor, I want multiple regression algorithms trained on my data, so that I can compare their performance and find the best regressor.

#### Acceptance Criteria

1. WHEN the task is classified as regression, THE Training_Module SHALL train all of the following models: Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, Support Vector Regressor, and K-Nearest Neighbors Regressor
2. WHEN the Training_Module trains each regression model, THE Training_Module SHALL use the training set produced by the Wrangling_Module
3. WHEN the Training_Module completes training of a regression model, THE Training_Module SHALL evaluate the model on the test set and record R-squared score, Mean Absolute Error, and Root Mean Squared Error
4. IF a regression model fails to converge during training, THEN THE Training_Module SHALL skip that model and continue training remaining models

### Requirement 8: Model Comparison and Results Presentation

**User Story:** As a Visitor, I want to see a comparison of all trained models with the best model highlighted, so that I can understand which algorithm works best for my dataset.

#### Acceptance Criteria

1. WHEN all models complete training, THE Results_Module SHALL display a comparison table listing each model name and its evaluation metrics
2. WHEN all models complete training, THE Results_Module SHALL sort the comparison table by the primary metric in descending order — accuracy for classification tasks and R-squared for regression tasks
3. WHEN all models complete training, THE Results_Module SHALL highlight the best-performing model in the comparison table
4. WHEN all models complete training, THE Results_Module SHALL display a bar chart comparing the primary metric across all trained models
5. WHEN the Visitor views the results page, THE Results_Module SHALL display sample predictions from the best model on a subset of the test data

### Requirement 9: Pipeline Error Handling

**User Story:** As a Visitor, I want clear error messages when something goes wrong during analysis, so that I can understand what happened and take corrective action.

#### Acceptance Criteria

1. IF the uploaded CSV_File contains fewer than 10 rows of data, THEN THE Platform SHALL display an error message indicating insufficient data for model training
2. IF the CSV_File cannot be parsed due to encoding issues or malformed content, THEN THE Platform SHALL display an error message indicating the file could not be read
3. IF all models fail during training, THEN THE Platform SHALL display an error message listing the reasons each model failed
4. IF the Wrangling_Module produces a dataset with zero features after preprocessing, THEN THE Platform SHALL display an error message indicating no usable features remain after preprocessing
5. THE Platform SHALL display a progress indicator to the Visitor while the Pipeline executes

### Requirement 10: Code Modernization and Compatibility

**User Story:** As a developer, I want the codebase to use current library APIs and Python best practices, so that the application runs without deprecation errors on modern environments.

#### Acceptance Criteria

1. THE Platform SHALL use scikit-learn SimpleImputer from sklearn.impute instead of the deprecated sklearn.preprocessing.Imputer class
2. THE Platform SHALL eliminate all deprecated API calls including the removed axis parameter in imputation and deprecated import paths
3. THE Platform SHALL consolidate duplicate code between eda_api.py and predictor_api.py into shared modules with no repeated function definitions
4. THE Platform SHALL use Flask's built-in development server with the standard app.run() method and provide a production WSGI configuration separately using gunicorn
5. THE Platform SHALL be compatible with Python 3.10 or later, scikit-learn 1.3 or later, pandas 2.0 or later, and matplotlib 3.7 or later
6. THE Platform SHALL replace bare except clauses with specific exception handling that logs the error type and message

### Requirement 11: Project Structure and Deployment Readiness

**User Story:** As a developer, I want the project properly structured with dependencies documented and deployable on EC2 at automl.betterai.guru, so that the project runs independently from the main static website while being accessible through a subdomain.

#### Acceptance Criteria

1. THE Platform SHALL include a requirements.txt file listing all Python dependencies with pinned versions compatible with the version constraints in Requirement 10
2. THE Platform SHALL include deployment scripts for AWS EC2 (systemd service file, nginx reverse proxy config with SSL, and setup instructions)
3. THE Platform SHALL run as a standalone application on EC2 accessible at https://automl.betterai.guru independently from the betterai.guru S3/CloudFront static website
4. THE Platform SHALL include a README file documenting project purpose, setup instructions, EC2 deployment guide, DNS configuration for subdomain, usage guide, and example screenshots
5. THE Platform SHALL organize source code into separate modules for upload handling, EDA, visualization, wrangling, training, and results with no duplicate function definitions across modules
6. THE Platform SHALL include a test suite with unit tests covering the Wrangling_Module data transformations and the Training_Module model evaluation logic
7. THE Platform SHALL include a .gitignore file excluding uploaded data files, generated images, virtual environments, and Python cache directories
8. THE Platform SHALL use gunicorn as the production WSGI server behind nginx on EC2
9. THE Platform SHALL use Let's Encrypt (certbot) for SSL certificate provisioning on the automl.betterai.guru subdomain
10. THE betterai.guru static website SHALL include a navigation link or page that directs visitors to https://automl.betterai.guru
