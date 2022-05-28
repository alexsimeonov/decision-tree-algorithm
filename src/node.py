# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 23:38:36 2021

@author: alexs
"""

from enum import Enum
from aislab.dp_feng.binenc import *
from aislab.gnrl import *
import pandas as pd
from utils import filter_dictionary

class Status(Enum):
    ROOT = 0
    DECISION = 1
    TERMINAL = 2
    LEAF = 3

class Node:
    def __init__(self, params, x, y, config, parent=None):
        self.params = params
        self.parent = parent # If present
        self.children = [] # Add children after split
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
        self.xe = enc_int(x, cname, xtp, vtp, order, dsp, dlm)
        toc('INT-ENCODING')

        # 2. BINNING
        tic()
        ub = ubng(self.xe, xtp, w, y=y, ytp=ytp, cnames=cname)     # unsupervised binning
        toc('UBNG finished successfully.')
        tic()
        sb = sbng(ub)       # supervised binning
        toc('SBNG finished successfully.')
        best_split = self.get_best_split(sb)

        # # get only the 'Normal' bins and form the node's children from them
        normal_bins = filter_dictionary(best_split[0]['bns'], lambda bin: bin['type'] == 'Normal')
        self.define_node_chidren(normal_bins, best_split[0]['cname'])
        print(self.children)
        print('Finished successfully!')

    def get_best_split(self, binning_result):
        best_split_variable = filter(lambda variable: variable['st'][0]['Chi2'][0] == max(map(lambda var: var['st'][0]['Chi2'][0], binning_result)), binning_result)
        return list(best_split_variable)

    def define_node_chidren(self, bins, column_name):
        # currently using lists for the structures that I am building
        # discuss if this is optimal or should use dictionaries instead
        for (key, value) in bins.items():
            self.children.append(self.compose_child(value, column_name))

    def compose_child(self, bin, column_name):
        child_x = None
        child_y = None

        if len(bin['lb']) and not len(bin['rb']):
            child_x = self.xe[(self.xe[column_name].isin(bin['lb']))]
        elif len(bin['lb']) == 1 | len(bin['rb']) == 1:
            child_x = self.xe[(column_name >= bin['lb'][0]) and (column_name <= bin['rb'][0])]

        child_y = self.y[self.y.index.isin(list(child_x.index))]

        return Node(self.params, child_x, child_y, self.config)

    def prune():
        print("pruning")
        #for the pruning phase, not yet implemented
