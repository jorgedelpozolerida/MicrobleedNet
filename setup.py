#!/usr/bin/env python

import sys
import setuptools


# TODO: 
# - add python versions dependencies
# - add more complex setup (README, license, etc)


setuptools.setup(name='cmbnet',
                version='0.1.0',
                description='Segmentation of Cerebral Microbleeds',
                author='Jorge del Pozo Lérida',
                url='https://github.com/jorgedelpozolerida/MicrobleedNet',
                keywords=['segmentation', 'microbleeds', 'brain'],
                packages=setuptools.find_packages(),
                # python_requires='>=3.6',
                include_package_data=True)
