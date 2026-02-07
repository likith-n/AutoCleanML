"""
AutoCleanML Setup Configuration
================================
"""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')
requirements = (this_directory / "requirements.txt").read_text(encoding='utf-8').splitlines()
requirements = [line.strip() for line in requirements if line.strip() and not line.startswith("#")]

setup(
    name="autocleanml",
    version="0.1.0",  
    author="Likith N",
    author_email="nlikith54@gmail.com",
    description="Automated ML data cleaning and preprocessing pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/likith-n/AutoCleanML",
    
    # IMPORTANT: src layout configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include package data
    include_package_data=True,
    
    # Dependencies
    python_requires=">=3.7",
    install_requires=requirements,
    
    # Optional dependencies
    extras_require={
        'dev': ['pytest>=7.0.0', 'jupyter>=1.0.0'],
        'examples': ['streamlit>=1.28.0', 'matplotlib>=3.5.0'],
    },
    
    # PyPI classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    # Keywords
    keywords=["machine-learning", "data-cleaning", "automated-ml"],
    
    # Zip safe
    zip_safe=False,
)
