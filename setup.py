from setuptools import setup

import daxmod

VERSION = daxmod.__version__
DESCRIPTION = 'Daxmod. A Python toolbox for text classification'
with open('README.md') as f:
        LONG_DESCRIPTION = f.read()
        
KEYWORDS = ['python','text classification','feature extraction'] 
REQUIRES = [
                "joblib >= 1.1.0",
                "numpy >= 1.21.4",
                "pandas >= 1.3.4",
                "scikit_learn >= 1.0.1",
                "tensorflow >= 2.7.0",
                "tensorflow_hub >= 0.12.0",
                "tensorflow_text >= 2.7.3",
        ]


setup(
        name='daxmod',
        version=VERSION,
        author='Ahmed Tchagnaou',
        author_email='ahmed.tchagnaou@etu.univ-tours.fr',
        packages=['daxmod'],
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license='BSD 2-clause',
        url='https://github.com/Authentic10/daxmod.git',
        install_requires = REQUIRES,
        keywords=KEYWORDS,
)
