from setuptools import setup, find_packages

setup(
    name='kernels',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'atomicwrites',
        'matplotlib',
        'numdifftools',
        'numpy',
        'pandas',
        'scipy',
        'tqdm',
    ],
    extras_require={
        'tests': [
            'pytest',
            'pytest-cov',
        ],
        'docs': [
            'sphinx',
        ],
    },
)
