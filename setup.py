from setuptools import setup

setup(
    name='pypenalty',
    version='0.1.0',
    description='A Python package for smooth penalty functions for optimization.',
    author='Christian Hauschel',
    packages=['pypenalty'],
    python_requires='>=3.5',
    install_requires=[
        'numpy',
    ],
)