# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 23:38:36 2021

@author: alexs
"""

from enum import Enum
from aislab.dp_feng.binenc import *
from aislab.gnrl import *
import pandas as pd

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
        self.x = x
        self.y = y
        self.config = config

    def split(self):
        print("Node starts splitting.")

        N = 1000 # x.shape[0]

        x = self.x.iloc[:N, :]
        y = self.y.iloc[:N, :].values

        tic2() # Overall time

        cname = self.config['cnames'].tolist()
        xtp = self.config['xtp'].values
        vtp = self.config['xtp'].values
        order = self.config['order']
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
        sb = sbng(ub)       # supervised binning
        toc('SBNG')
        print(sb[0])
        print('Finished successfully!')

    def prune():
        print("pruning")
        #for the pruning phase, not yet implemented
