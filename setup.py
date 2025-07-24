"""
Setup configuration for ARIASKA_RL package.
"""
from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

# Version information
VERSION = "2.1.0"

setup(
    name="ariaska-rl",
    version=VERSION,
    author="ARIASKA Team",
    author_email="team@ariaska.ai",
    description="Next-Generation GPT-Augmented Multi-Agent RL Cybersecurity Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Reckless98/Ariaska_RL",
    project_urls={
        "Bug Tracker": "https://github.com/Reckless98/Ariaska_RL/issues",
        "Documentation": "https://ariaska-rl.readthedocs.io/",
        "Source Code": "https://github.com/Reckless98/Ariaska_RL",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: System :: Networking :: Monitoring",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
        ],
        "monitoring": [
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
            "prometheus-client>=0.17.0",
        ],
        "security": [
            "cryptography>=41.0.3",
            "bandit>=1.7.5",
            "safety>=2.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ariaska=main:main",
            "ariaska-train=core.training.enhanced_trainer:main",
            "ariaska-cli=ariaska_cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ariaska_rl": [
            "config/*.yaml",
            "config/*.json",
            "data/examples/*",
            "docs/images/*",
        ],
    },
    zip_safe=False,
    keywords=[
        "reinforcement-learning",
        "cybersecurity",
        "multi-agent",
        "artificial-intelligence",
        "penetration-testing",
        "gpt",
        "llm",
        "red-team",
        "blue-team",
        "machine-learning",
    ],
    platforms=["any"],
    license="MIT",
    test_suite="tests",
    tests_require=[
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.11.1",
    ],
)