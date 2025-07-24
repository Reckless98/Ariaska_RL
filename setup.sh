#!/bin/bash

# ARIASKA_RL Setup Script
# Automated setup for development environment

set -e  # Exit on any error

echo "🚀 Setting up ARIASKA_RL development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
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

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_status "Found Python $PYTHON_VERSION"

# Check if version is >= 3.8
if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    print_success "Python version is compatible"
else
    print_error "Python 3.8+ is required, found $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_status "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "Main dependencies installed"
else
    print_warning "requirements.txt not found, installing core dependencies..."
    pip install torch numpy openai rich python-dotenv pytest
fi

# Install development dependencies
if [ -f "requirements-dev.txt" ]; then
    pip install -r requirements-dev.txt
    print_success "Development dependencies installed"
else
    print_status "Installing basic development tools..."
    pip install pytest pytest-cov flake8 black isort mypy
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        print_status "Creating .env file from template..."
        cp .env.example .env
        print_warning "Please edit .env file with your API keys and configuration"
    else
        print_status "Creating basic .env file..."
        cat > .env << EOF
# ARIASKA_RL Configuration
OPENAI_API_KEY=your_api_key_here
ENVIRONMENT_MODE=simulated
LOG_LEVEL=INFO
CUDA_VISIBLE_DEVICES=0
EOF
        print_warning "Please edit .env file with your actual API keys"
    fi
else
    print_status ".env file already exists"
fi

# Create logs directory
if [ ! -d "logs" ]; then
    print_status "Creating logs directory..."
    mkdir -p logs
    print_success "Logs directory created"
fi

# Create necessary directories
print_status "Creating project directories..."
mkdir -p data/checkpoints
mkdir -p data/experiments
mkdir -p data/results
mkdir -p runs
print_success "Project directories created"

# Check if CUDA is available
print_status "Checking CUDA availability..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" || print_warning "Could not check CUDA availability"

# Run basic tests if they exist
if [ -d "tests" ]; then
    print_status "Running basic tests..."
    if python3 -m pytest tests/ -v --tb=short -x; then
        print_success "Basic tests passed"
    else
        print_warning "Some tests failed - this is normal for initial setup"
    fi
else
    print_status "No tests directory found"
fi

# Generate a basic test to verify installation
print_status "Verifying installation..."
python3 -c "
import torch
import numpy as np
import openai
from rich.console import Console

console = Console()
console.print('[green]✓[/green] All core dependencies imported successfully')
console.print(f'[blue]PyTorch version:[/blue] {torch.__version__}')
console.print(f'[blue]NumPy version:[/blue] {np.__version__}')
console.print(f'[blue]CUDA available:[/blue] {torch.cuda.is_available()}')
" || {
    print_error "Installation verification failed"
    exit 1
}

print_success "🎉 ARIASKA_RL setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: source venv/bin/activate"
echo "3. Run: python main.py --help"
echo "4. Run tests: pytest tests/"
echo ""
print_status "Happy coding! 🧠⚔️"