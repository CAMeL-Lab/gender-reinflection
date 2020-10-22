import os
from setuptools import setup


CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering :: Deep Learning',
    'Topic :: Grammatical Gender Reinflection :: Linguistics',
]

DESCRIPTION = ('Code for the '
               'Gender-Aware Reinflection using Linguistically Enhanced Neural Models '
               'paper at GeBNLP2020.')

README_FILE = os.path.join(os.path.dirname(__file__), 'README.md')

with open(README_FILE, 'r') as fh:
    LONG_DESCRIPTION = fh.read().strip()

INSTALL_REQUIRES = [
    'torch==1.3.0',
    'scikit-learn==0.21.3',
    'sacrebleu',
    'camel_tools',
    'gensim==3.8.0',
    'matplotlib==3.1.1'
]

setup(
    name='gender_reinflection',
    version='0.1',
    author='Bashar Alhafni',
    author_email='ba63@nyu.edu',
    maintainer='Bashar Alhafni',
    maintainer_email='ba63@nyu.edu',
    license='MIT',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
)