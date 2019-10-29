from setuptools import setup
import causal_selection

setup(
    name='causal_selection',
    version=causal_selection.__version__,
    description='Select and score features for causal inferences',
    long_description=causal_selection.__doc__,
    author='Reid Pryzant',
    author_email='rpryzant@stanford.edu',
    url='https://github.com/rpryzant/causal_selection',
    keywords=['feature selection'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['causal_selection'],
    install_requires=[
        'scipy',
        'sklearn',
        'nltk',
        'pandas',
        'torch>=1.0.0'
    ],
    python_requires='>=3.6'
)

    
