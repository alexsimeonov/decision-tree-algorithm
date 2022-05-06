# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 23:38:36 2021

@author: alexs
"""

# import os
# import sys
# import uuid

# wpath = '/Users/alexandersimeonov/Library/Python/2.7/lib/python/site-packages'
# # path to dataset
# x_path = '/Users/alexandersimeonov/Documents/Development/University/decision-tree-algorithm/binning/aislab/data/x_samples_250000.csv'
# # path to output values
# y_path = '/Users/alexandersimeonov/Documents/Development/University/decision-tree-algorithm/binning/aislab/data/y_samples_250000.csv'
# c_path = '/Users/alexandersimeonov/Documents/Development/University/decision-tree-algorithm/binning/aislab/data/cnf_LC.csv'

# sys.path.append(wpath)
# os.chdir(wpath)

from enum import Enum
from aislab.dp_feng.binenc import *
from aislab.gnrl import *

class Status(Enum):
    ROOT = 0
    DECISION = 1
    TERMINAL = 2
    LEAF = 3

class Node:
    def __init__(self, params, x, y, config, parent=None):
        self.params = params
        self.parent = parent # If present
        self.children = None # Add children after split
        self.status = Status.TERMINAL
        self.x_path = x
        self.y_path = y
        self.config_path = config

    def split(self):
        print("Node starts splitting.")

        x = pd.read_csv(self.x_path)
        y = pd.read_csv(self.y_path)
        cnf = pd.read_csv(self.config_path)

        N = 1000 # x.shape[0] #

        x = x.iloc[:N, :]
        y = y.iloc[:N, :].values

        tic2() # Overall time

        cname = cnf['cnames'].tolist()
        xtp = cnf['xtp'].values
        vtp = cnf['xtp'].values
        order = cnf['order']
        x = x[cname]
        w = ones((N, 1))
        ytp = ['bin']
        dsp = 1
        order = order.values
        dlm = '$'
        # 1. All categorical vars to int
        tic()
        xe = enc_int(x, cname, xtp, vtp, order, dsp, dlm)
        toc('INT-ENCODING')

        # 2. BINNING
        tic()
        ub = ubng(xe, xtp, w, y=y, ytp=ytp, cnames=cname)     # unsupervised binning
        toc('UBNG')
        tic()
        sb = sbng(ub)
        # new Node(x = sb[0].x) pseudo          # supervised binning
        # toc('SBNG')


            # When fix the import use the ubng method in here
            # Always apply ubng and sbng
            # Recursively executed the actions function described below
            # Call the external function (split_method) from main.py
            # with self.data_set use the result in order to create a new Node instances
            # for left and right properties and call their split methods
            # This should continue until one of the predefined conditions is reached

    def prune():
        print("pruning")
        #for the pruning phase, not yet implemented
