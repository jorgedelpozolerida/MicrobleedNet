#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Module with functions for Rodeja dataset

paper: https://arxiv.org/abs/2301.09322


@author: jorgedelpozolerida
@date: 13/02/2024
"""
import os
import argparse
import traceback

import logging                                                                      
import numpy as np                                                                  
import pandas as pd                                                                 
from tqdm import tqdm
import nibabel as nib
from scipy.io import loadmat
import glob
import sys
from typing import Tuple, Dict, List, Any

import cmbnet.preprocessing.process_masks as utils_process
import cmbnet.utils.utils_general as utils_general


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def load_RODEJA_data():
    raise NotImplementedError