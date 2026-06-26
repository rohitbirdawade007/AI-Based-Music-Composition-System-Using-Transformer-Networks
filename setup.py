"""
AI-Based Music Composition System
Setup configuration for local package installation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="music-composition-ai",
    version="1.0.0",
    author="Rohit Birdawade",
    author_email="rohitbirdawade007@gmail.com",
    description="AI-Based Music Composition System Using Transformer Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rohitbirdawade007/AI-Based-Music-Composition-System-Using-Transformer-Networks",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pretty_midi>=0.2.10",
        "mido>=1.3.0",
        "music21>=9.1.0",
        "midiutil>=1.2.1",
        "faiss-cpu>=1.7.4",
        "streamlit>=1.28.0",
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
        "matplotlib>=3.7.0",
        "plotly>=5.17.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "rag": [
            "sentence-transformers>=2.2.0",
            "langchain>=0.1.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "flake8>=6.1.0",
            "black>=23.9.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio",
    ],
    keywords=[
        "music generation",
        "transformer",
        "deep learning",
        "midi",
        "generative ai",
        "music composition",
        "rag",
        "faiss",
    ],
    entry_points={
        "console_scripts": [
            "music-ai-train=training.train:main",
            "music-ai-generate=src.inference.generator:cli_generate",
        ],
    },
)
