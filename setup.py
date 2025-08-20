"""Setup script for hypergraph-spectral-decomposer."""

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="hypergraph-spectral-decomposer",
        version="0.1.0",
        description="Spectral hypergraph community detection via Zhou's normalized Laplacian",
        author="faiazparis",
        author_email="faiazparis@example.com",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        python_requires=">=3.8",
        install_requires=[
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "scikit-learn>=1.0.0",
            "pandas>=1.3.0",
            "typer[all]>=0.9.0",
            "matplotlib>=3.5.0",
        ],
        extras_require={
            "dev": [
                "pytest>=7.0.0",
                "pytest-cov>=4.0.0",
                "black>=23.0.0",
                "isort>=5.12.0",
                "flake8>=6.0.0",
                "mypy>=1.0.0",
                "pre-commit>=3.0.0",
            ]
        },
        entry_points={
            "console_scripts": [
                "hgsd=hypergraph_spectral_decomposer.cli:app",
            ],
        },
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Information Analysis",
        ],
    )
