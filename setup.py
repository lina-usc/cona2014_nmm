""" Cona et al. (2014) model of the thalamocortical loop.

Authors:
Christian O'Reilly <christian.oreilly@sc.edu>

This code is a translation and refactoring of the Matlab code originally 
from the authors of the paper:

Filippo Cona, M. Lacanna, Mauro Ursino: A thalamo-cortical neural mass model
for the simulation of brain rhythms during sleep. J. Comput. Neurosci. 
37(1): 125-148 (2014)

License: MIT
"""

from pathlib import Path
from setuptools import setup, find_packages

with Path('requirements.txt').open() as f:
    requirements = f.read().splitlines()

extras = {}

extras_require = {}
for extra, req_file in extras.items():
    with Path(req_file).open() as file:
        requirements_extra = file.read().splitlines()
    extras_require[extra] = requirements_extra

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='cona2014_nmm',
    version='0.1.0',
    description='Cona et al. (2014) model of the thalamocortical loop.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Christian O'Reilly",
    author_email='christian.oreilly@sc.edu',
    url='https://github.com/lina-usc/cona2024_nmm',
    packages=find_packages(),
    install_requires=requirements,
    extras_require=extras_require,
    include_package_data=True,
)
