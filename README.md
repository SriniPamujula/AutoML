# AutoML Web Platform

A web-based automated machine learning platform that takes a CSV dataset through the complete ML pipeline — EDA, visualizations, preprocessing, model training, and comparison — all through an interactive browser interface.

🌐 **Live:** [automl.betterai.guru](https://automl.betterai.guru) (when deployed)

## Features

- **Upload any CSV** — drag and drop or file picker (up to 50 MB)
- **Auto-detect task type** — classification or regression based on target column
- **Exploratory Data Analysis** — shape, statistics, null counts, data types, unique values
- **Interactive Plotly Visualizations** — box plots, correlation heatmaps, distribution charts
- **Automated Preprocessing** — null imputation (median), encoding (label + one-hot), standard scaling, 80/20 split
- **Multi-model Training** — trains 9 classification or 7 regression models simultaneously
- **Results Comparison** — sorted table, interactive bar chart, best model highlighted, sample predictions
- **Dark SaaS theme** — matches BetterAI brand (Space Grotesk + Inter, purple/cyan palette)

## Models

### Classification (9 models)
| Model | Type |
|-------|------|
| Logistic Regression | Linear |
| Decision Tree | Tree |
| Random Forest | Ensemble |
| Gradient Boosting | Ensemble |
| XGBoost | Ensemble |
| AdaBoost | Ensemble |
| K-Nearest Neighbors | Instance-based |
| Support Vector Machine | Kernel |
| Naive Bayes | Probabilistic |

### Regression (7 models)
| Model | Type |
|-------|------|
| Linear Regression | Linear |
| Decision Tree | Tree |
| Random Forest | Ensemble |
| Gradient Boosting | Ensemble |
| XGBoost | Ensemble |
| Support Vector Regressor | Kernel |
| K-Nearest Neighbors | Instance-based |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.10+, Flask |
| ML | scikit-learn, XGBoost |
| Data | pandas, numpy |
| Visualization | Plotly (interactive, dark-themed) |
| Frontend | Bootstrap 5, Inter + Space Grotesk fonts |
| Production | gunicorn + nginx + Let's Encrypt |
| Deployment | AWS EC2 |

## Local Development

```bash
# Clone
git clone https://github.com/SriniPamujula/WebAnalytics.git AutoML
cd AutoML

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Run development server
python app.py
```

Open [http://localhost:5000](http://localhost:5000)

## Usage

1. Open the platform in your browser
2. Upload a CSV file (any tabular dataset)
3. Select the target column to predict
4. Wait for the pipeline to complete (~10-30 seconds)
5. Review results: EDA → Visualizations → Model Comparison → Predictions

## Sample Datasets

The `Data/` folder includes sample datasets for testing:
- `breast-cancer-data.csv` — UCI Breast Cancer (classification, 569 rows, 32 columns)
- `cereal.csv` — Cereal nutrition data (regression)

## EC2 Deployment

### Prerequisites
- Ubuntu 22.04+ EC2 instance (t3.medium recommended for ML workloads)
- DNS A record: `automl.betterai.guru` → EC2 public IP

### Deploy

```bash
ssh ubuntu@<your-ec2-ip>
git clone https://github.com/SriniPamujula/WebAnalytics.git /home/ubuntu/AutoML
cd /home/ubuntu/AutoML
sudo bash deploy/setup.sh

# After DNS propagates:
sudo certbot --nginx -d automl.betterai.guru
```

### Management

```bash
sudo systemctl status automl     # Check status
sudo systemctl restart automl    # Restart
sudo journalctl -u automl -f     # View logs
```

## Project Structure

```
AutoML/
├── app.py                    # Flask application (routes, pipeline orchestration)
├── config.py                 # Configuration (limits, paths, ML params)
├── exceptions.py             # Custom exceptions (Validation, Preprocessing, Training)
├── upload_handler.py         # File upload validation + task type detection
├── eda_module.py             # Exploratory data analysis
├── visualization_module.py   # Plotly chart generation (boxplot, heatmap, countplot, comparison)
├── wrangling_module.py       # Data preprocessing pipeline
├── training_module.py        # Model training + evaluation (classification & regression)
├── results_module.py         # Results compilation (table, chart, predictions)
├── templates/                # Jinja2 HTML templates (dark BetterAI theme)
│   ├── base.html             # Base layout with navbar, footer, styles
│   ├── index.html            # Upload page
│   ├── select_target.html    # Target column selection
│   ├── results.html          # Full results display
│   └── error.html            # Error page
├── static/
│   ├── css/main.css          # Custom styles
│   ├── js/main.js            # Upload progress indicator
│   └── generated/            # Runtime-generated chart HTML (gitignored)
├── Data/                     # Sample datasets for testing
├── tests/                    # pytest unit tests
├── deploy/                   # EC2 deployment configuration
│   ├── automl.service        # systemd unit file
│   ├── automl-nginx.conf     # nginx reverse proxy + SSL
│   └── setup.sh              # Full EC2 setup script
├── requirements.txt          # Pinned Python dependencies
├── LICENSE                   # MIT License
└── .gitignore
```

## Running Tests

```bash
pytest tests/ -v
```

## Architecture

```
Browser → nginx (HTTPS, automl.betterai.guru)
           → gunicorn (127.0.0.1:8000)
              → Flask App
                 ├── upload_handler    (validate, save, detect task)
                 ├── eda_module        (statistics, quality checks)
                 ├── visualization_module  (Plotly charts)
                 ├── wrangling_module  (impute, encode, scale, split)
                 ├── training_module   (9 classifiers / 7 regressors + XGBoost)
                 └── results_module    (compare, rank, chart, predict)
```

## Related Projects

- [BetterAI Cost Advisor™](https://betterai.guru) — AI-Powered Databricks FinOps Platform
- [CostOptimization](https://github.com/SriniPamujula/CostOptimization) — Databricks Cost Optimization Agent

## License

MIT — © 2025 BetterAI LLC. See [LICENSE](./LICENSE).
