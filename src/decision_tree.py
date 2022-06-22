import pandas as pd

from node import Node
from aislab.dp_feng.binenc import *
from aislab.gnrl import *
from treelib import Node as TreelibNode, Tree

class DecisionTree:
    def __init__(self, x, y, config, hyperparams):
        self.hyperparams = hyperparams
        self.x = x
        self.y = y
        self.config = config
        # Initialize the root node, using the data passed to the DT
        self.root_node = Node(self.hyperparams)
        self.dictionary_structure = self.root_node
        self.structure = Tree()
        self.statistics = { 'leaf_count': 0, 'nodes_count': 1, 'growing_time': None }

    def grow(self):
        print('Decision tree starts growing.')
        encoded_values = self.encode_values(self.x, self.y, self.config)
        self.structure.create_node('Root', 'root')
        self.root_node.split(encoded_values, self.config, tree=self.structure)
        print('Decision tree growing phase successful.')

    def represent_structure(self):
        print('Decision tree structure representation:\n')
        self.structure.show()

    def encode_values(self, x, y, config):
        tic2() # Overall time

        cname = config['cnames'].tolist()
        xtp = config['xtp'].values
        vtp = config['xtp'].values
        order = config['order']
        x = x[cname]

        # Setting the number of samples to perform for the algorithm on
        N = self.hyperparams['initial_dataset_samples_count']

        x = x.iloc[:N, :]
        y = y.iloc[:N, :]
        w = ones((len(x.index), 1))
        ytp = ['bin']
        dsp = 1
        order = order.values
        dlm = '$'
        # 1. All categorical vars to int
        tic()
        xe = enc_int(x, cname, xtp, vtp, order, dsp, dlm)
        toc('INT-ENCODING')
        return { 'x': xe, 'y': y, 'xtp': xtp, 'ytp': ytp, 'vtp': vtp, 'w': w, 'cname': cname }
