from setuptools import setup
import causal_attribution

setup(
    name='causal_attribution',
    version=causal_attribution.__version__,
    description='Select and score features for causal inferences.',
    long_description=causal_attribution.__doc__,
    author='Reid Pryzant',
    author_email='rpryzant@stanford.edu',
    url='https://github.com/rpryzant/causal_attribution',
    keywords=['feature selection'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['causal_attribution'],
    install_requires=[
        'scipy',
        'sklearn',
        'nltk',
        'pandas',
        'torch>=1.0.0'
    ],
    python_requires='>=3.6'
)

    
