# Federated Learning Comparison

Compare federated vs centralized machine learning performance across multiple models and datasets.

## Quick Start

```bash
# Run complete pipeline (recommended)
chmod +x pipeline_script.sh
./pipeline_script.sh
```

**Or run individual components:**

```bash
# Install dependencies
pip install -e .

# Run federated experiments
python federated_ml.py

# Run centralized experiments  
python centralized_ml.py

# Compare results
python comparison.py
```

## Models & Datasets

**Models**: Logistic Regression, Random Forest, SVM  
**Datasets**: Iris, Adult Income  
**Strategies**: FedAvg, FedProx, FedMedian

## Pipeline Script

The `pipeline_script.sh` automates the complete workflow:
- Sets up virtual environment
- Installs dependencies
- Runs federated learning experiments
- Runs centralized learning experiments
- Performs comparison analysis
- Generates all result files

## Project Structure

```
fed_learning/           # Core federated learning modules
data/                   # Dataset files (iris.csv, adult.csv)
results/                # Output CSV files and logs
pipeline_script.sh      # Complete automated pipeline
centralized_ml.py       # Centralized learning experiments
federated_ml.py         # Federated learning experiments
comparison.py           # Performance comparison analysis
```

## Requirements

- Python 3.8+
- scikit-learn >= 1.3.0
- flwr[simulation] >= 1.19.0
- pandas >= 2.0.0

## Data Format

Place your datasets in the `data/` directory:
- `data/iris.csv` - Iris classification dataset
- `data/adult.csv` - Adult income prediction dataset

Results are saved to `results/` as CSV files for analysis.