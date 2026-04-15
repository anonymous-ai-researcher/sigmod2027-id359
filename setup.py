from setuptools import setup, find_packages

setup(
    name="phantom-neighbors",
    version="0.1.0",
    description="Information Leakage and Its Prevention in Access-Controlled Vector Databases",
    author="Anonymous",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
    ],
)
