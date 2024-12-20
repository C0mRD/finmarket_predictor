from setuptools import setup, find_packages

setup(
    name="finmarket_predictor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.1',
        'transformers>=4.31.0',
        'pandas>=1.5.3',
        'numpy>=1.24.3',
        'yfinance>=0.2.28',
        'alpha_vantage>=2.3.1',
        'newsapi-python>=0.2.7',
        'fastapi>=0.95.2',
        'uvicorn>=0.22.0',
        'streamlit>=1.22.0',
        'pytorch-lightning>=2.0.2',
        'scikit-learn>=1.2.2',
        'plotly>=5.15.0',
        'ta>=0.10.2',
        'seaborn>=0.12.2',
        'pyarrow>=9.0.0'
    ],
    python_requires='>=3.8',
)