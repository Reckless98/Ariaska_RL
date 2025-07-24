# Makefile for ARIASKA_RL Development

.PHONY: help install install-dev test test-unit test-integration lint format clean setup docs train

# Default target
help: ## Show this help message
	@echo "ARIASKA_RL Development Commands"
	@echo "==============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Environment setup
setup: ## Set up development environment
	@echo "🚀 Setting up ARIASKA_RL development environment..."
	chmod +x setup.sh
	./setup.sh

install: ## Install production dependencies
	@echo "📦 Installing production dependencies..."
	pip install -r requirements.txt

install-dev: install ## Install development dependencies
	@echo "🔧 Installing development dependencies..."
	pip install -r requirements-dev.txt
	pip install -e .

# Testing
test: ## Run all tests
	@echo "🧪 Running all tests..."
	pytest tests/ -v --tb=short --cov=core --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	@echo "🔬 Running unit tests..."
	pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests only
	@echo "🔗 Running integration tests..."
	pytest tests/integration/ -v --tb=short

test-fast: ## Run fast tests (exclude slow tests)
	@echo "⚡ Running fast tests..."
	pytest tests/ -v --tb=short -m "not slow"

test-coverage: ## Run tests with coverage report
	@echo "📊 Running tests with coverage..."
	pytest tests/ --cov=core --cov-report=html --cov-report=term --cov-fail-under=70

# Code quality
lint: ## Run linting checks
	@echo "🔍 Running linting checks..."
	flake8 core/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	mypy core/ --ignore-missing-imports
	bandit -r core/ -f json -o security-report.json || true

format: ## Format code
	@echo "✨ Formatting code..."
	black core/ tests/ --line-length=88
	isort core/ tests/ --profile=black

format-check: ## Check code formatting
	@echo "🔍 Checking code formatting..."
	black core/ tests/ --check --line-length=88
	isort core/ tests/ --check-only --profile=black

# Documentation
docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	sphinx-build -b html docs/ docs/_build/html

docs-serve: ## Serve documentation locally
	@echo "🌐 Serving documentation at http://localhost:8000"
	cd docs/_build/html && python -m http.server 8000

# Training and execution
train: ## Start training with default parameters
	@echo "🏋️ Starting training..."
	python main.py --train

train-fast: ## Quick training for testing (10 episodes)
	@echo "⚡ Quick training session..."
	python main.py --train --episodes 10 --steps 10

train-gpu: ## Train with GPU acceleration
	@echo "🚀 Training with GPU..."
	CUDA_VISIBLE_DEVICES=0 python main.py --train

test-env: ## Test environment setup
	@echo "🌍 Testing environment..."
	python main.py --test-env

ui: ## Launch visualization dashboard
	@echo "📊 Launching dashboard..."
	python main.py --ui

# Docker
docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t ariaska_rl .

docker-run: ## Run Docker container
	@echo "🐳 Running Docker container..."
	docker run -it --rm -v $(PWD):/workspace ariaska_rl

docker-compose-up: ## Start with docker-compose
	@echo "🐳 Starting with docker-compose..."
	docker-compose up -d

docker-compose-down: ## Stop docker-compose services
	@echo "🐳 Stopping docker-compose services..."
	docker-compose down

# Data and cleaning
clean: ## Clean temporary files and caches
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf .mypy_cache/

clean-data: ## Clean training data and checkpoints
	@echo "🗑️ Cleaning training data..."
	rm -rf data/checkpoints/*
	rm -rf data/experiments/*
	rm -rf logs/*
	rm -rf runs/*
	rm -rf chroma_data/*

clean-all: clean clean-data ## Clean everything

# Database operations
init-db: ## Initialize database
	@echo "🗄️ Initializing database..."
	python -c "from core.memory.memory_router import MemoryRouter; MemoryRouter().initialize_database()"

reset-db: ## Reset database
	@echo "🔄 Resetting database..."
	rm -f data/ariaska.db
	rm -rf chroma_data/*
	$(MAKE) init-db

# Profiling and performance
profile: ## Run performance profiling
	@echo "📈 Running performance profiling..."
	python -m cProfile -o profile.stats main.py --train --episodes 5
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

memory-profile: ## Run memory profiling
	@echo "🧠 Running memory profiling..."
	mprof run main.py --train --episodes 5
	mprof plot

# Security
security-scan: ## Run security scanning
	@echo "🔒 Running security scan..."
	bandit -r core/ -f json -o security-report.json
	safety check

# Environment validation
validate: ## Validate environment and configuration
	@echo "✅ Validating environment..."
	python -c "from core.utils.config_manager import ConfigManager; cm = ConfigManager(); valid, errors = cm.validate_config(); print('✅ Valid' if valid else '❌ Errors: ' + str(errors))"

check-deps: ## Check dependency versions
	@echo "📋 Checking dependencies..."
	pip list --outdated
	pip check

# Release preparation
pre-commit: format lint test ## Run pre-commit checks
	@echo "✅ Pre-commit checks completed"

release-check: pre-commit security-scan docs ## Full release preparation checks
	@echo "🚀 Release checks completed"

# Development utilities
jupyter: ## Start Jupyter notebook server
	@echo "📓 Starting Jupyter notebook..."
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

tensorboard: ## Start TensorBoard
	@echo "📊 Starting TensorBoard..."
	tensorboard --logdir=runs --host=0.0.0.0 --port=6006

monitor: ## Monitor system resources during training
	@echo "📊 Monitoring system resources..."
	watch -n 1 'nvidia-smi; echo ""; ps aux | grep python | head -10'

# Quick development workflows
dev-setup: setup install-dev init-db ## Complete development setup
	@echo "🎉 Development environment ready!"

quick-test: format-check lint test-fast ## Quick development testing
	@echo "⚡ Quick tests completed"

full-test: format-check lint test test-coverage security-scan ## Comprehensive testing
	@echo "🔍 Full testing completed"

# Environment info
info: ## Show environment information
	@echo "🔍 Environment Information"
	@echo "=========================="
	@echo "Python version: $$(python --version)"
	@echo "PyTorch version: $$(python -c 'import torch; print(torch.__version__)')"
	@echo "CUDA available: $$(python -c 'import torch; print(torch.cuda.is_available())')"
	@echo "GPU count: $$(python -c 'import torch; print(torch.cuda.device_count())')"
	@echo "Working directory: $$(pwd)"
	@echo "Virtual environment: $$VIRTUAL_ENV"

# Help with common tasks
.PHONY: help-dev help-training help-testing

help-dev: ## Show development help
	@echo "🔧 Development Workflow"
	@echo "======================"
	@echo "1. make dev-setup     # Complete environment setup"
	@echo "2. make quick-test    # Quick validation"
	@echo "3. make train-fast    # Test training"
	@echo "4. make full-test     # Before committing"

help-training: ## Show training help
	@echo "🏋️ Training Workflows"
	@echo "===================="
	@echo "make train            # Standard training"
	@echo "make train-fast       # Quick test (10 episodes)"
	@echo "make train-gpu        # GPU accelerated"
	@echo "make ui               # Launch dashboard"
	@echo "make tensorboard      # View training logs"

help-testing: ## Show testing help
	@echo "🧪 Testing Workflows"
	@echo "==================="
	@echo "make test             # All tests with coverage"
	@echo "make test-unit        # Unit tests only"
	@echo "make test-integration # Integration tests only"
	@echo "make test-fast        # Exclude slow tests"