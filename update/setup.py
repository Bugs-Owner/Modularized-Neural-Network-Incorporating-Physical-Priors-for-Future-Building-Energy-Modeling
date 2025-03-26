from setuptools import setup, find_packages

setup(
    name='ModNN',
    version='0.1.0',
    description='Physics-Informed Modularized Neural Network for Building Energy Modeling',
    author='Zixin Jiang',
    author_email='zjiang19@syr.edu',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'tqdm',
    ],
    python_requires='>=3.7',
)
