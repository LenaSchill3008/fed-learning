# run_pipeline.sh - Complete Federated vs Centralized ML Pipeline

set -e  # Exit on any error

echo "🚀 FEDERATED vs CENTRALIZED ML PIPELINE"
echo "========================================"

# Configuration
VENV_NAME=".venv"
PYTHON_VERSION="python3"
RESULTS_DIR="results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
print_step "Checking Python installation..."
if ! command -v $PYTHON_VERSION &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH"
    exit 1
fi
print_success "Python found: $(python3 --version)"

# Check if data directory exists
print_step "Checking data directory..."
if [ ! -d "data" ]; then
    print_warning "Data directory not found. Please ensure data/iris.csv and data/adult.csv exist."
    print_warning "Creating data directory..."
    mkdir -p data
    echo "Please add your datasets to the data/ directory and run this script again."
    exit 1
fi

if [ ! -f "data/iris.csv" ] || [ ! -f "data/adult.csv" ]; then
    print_warning "Required datasets not found:"
    print_warning "  - data/iris.csv"
    print_warning "  - data/adult.csv"
    print_warning "Please add these files and run this script again."
    exit 1
fi
print_success "Data files found"

# Create results directory
print_step "Setting up results directory..."
mkdir -p $RESULTS_DIR
print_success "Results directory ready"

# Virtual environment setup
print_step "Setting up virtual environment..."
if [ -d "$VENV_NAME" ]; then
    print_warning "Virtual environment already exists. Removing and recreating..."
    rm -rf $VENV_NAME
fi

$PYTHON_VERSION -m venv $VENV_NAME
print_success "Virtual environment created"

# Activate virtual environment
print_step "Activating virtual environment..."
source $VENV_NAME/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_step "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_step "Installing dependencies..."
if [ -f "pyproject.toml" ]; then
    print_step "Installing from pyproject.toml..."
    pip install -e .
else
    print_step "Installing required packages..."
    pip install "flwr[simulation]>=1.19.0" "scikit-learn>=1.3.0" "pandas>=2.0.0"
fi
print_success "Dependencies installed"

# Clean previous results
print_step "Cleaning previous results..."
rm -f $RESULTS_DIR/*.csv $RESULTS_DIR/*.log
print_success "Previous results cleaned"

# Run federated learning experiments
print_step "Running federated learning experiments..."
echo "This may take several minutes..."

if python federated_ml.py; then
    print_success "Federated experiments completed"
fi

# Check federated results
if [ ! -f "$RESULTS_DIR/results.csv" ]; then
    print_error "Federated results file not generated"
    exit 1
fi

# Run centralized learning experiments  
print_step "Running centralized learning experiments..."
if python centralized_ml.py; then
    print_success "Centralized experiments completed"
else
    print_error "Centralized experiments failed"
    exit 1
fi

# Check centralized results
if [ ! -f "$RESULTS_DIR/centralized_results.csv" ]; then
    print_error "Centralized results file not generated"
    exit 1
fi

# Run comparison analysis
print_step "Running comparison analysis..."
if python comparison.py; then
    print_success "Comparison analysis completed"
else
    print_error "Comparison analysis failed"
    exit 1
fi


echo ""
echo " Federated learning experiments: COMPLETED"
echo " Centralized learning experiments: COMPLETED"  
echo " Comparison analysis: COMPLETED"

# Deactivate virtual environment
deactivate