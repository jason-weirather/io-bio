from setuptools import setup, find_packages
from codecs import open
from os import path

this_folder = path.abspath(path.dirname(__file__))
with open(path.join(this_folder,'README.md'),encoding='utf-8') as inf:
  long_description = inf.read()

setup(
  name='iobio',
  version='0.1.0',
  description='Simple tools for analysis',
  long_description=long_description,
  url='https://github.com/jason-weirather/io-bio',
  author='Jason L Weirather',
  author_email='JasonL_Weirather@dfci.harvard.edu',
  license='Apache License, Version 2.0',
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'License :: OSI Approved :: Apache Software License'
  ],
  keywords='bioinformatics',
  install_requires=['matplotlib','seaborn','pandas'],
  packages=['iobio',
            'iobio.qnorm',
            'iobio.entropy',
            'iobio.explicitsemanticanalysis',
            'iobio.stackedheatmap']
)
