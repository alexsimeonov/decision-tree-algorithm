from node import Node
import pandas as pd
from aislab.dp_feng.binenc import *
from aislab.gnrl import *
from utils.tree_representation import represent_tree

class DecisionTree:
    def __init__(self, x, y, config, hyperparams):
        self.hyperparams = hyperparams
        self.structure = []
        self.x = x
        self.y = y
        self.config = config
        # Initialize the root node, using the data passed to the DT
        self.root_node = Node(self.hyperparams)
        self.structure.append(self.root_node)
        self.encoded_values = None
        self.depth = 1;

    def grow(self):
        print('Decision tree starts growing.')
        self.encoded_values = self.encode_values(self.x, self.y, self.config)
        self.root_node.split(self.encoded_values['xe'], self.y, self.config, self.encoded_values, self.increase_tree_depth, self.get_tree_depth)

        print('Growing finished successfully!')

    def represent_structure(self):
        represent_tree(self.structure)

    def encode_values(self, x, y, config):
        y_values = y.values

        tic2() # Overall time

        cname = config['cnames'].tolist()
        xtp = config['xtp'].values
        vtp = config['xtp'].values
        order = config['order']
        x = x[cname]
        w = ones((len(x.index), 1))
        ytp = ['bin']
        dsp = 1
        order = order.values
        dlm = '$'
        # 1. All categorical vars to int
        tic()
        xe = enc_int(x, cname, xtp, vtp, order, dsp, dlm)
        toc('INT-ENCODING')
        return { 'xe': xe, 'y': y_values, 'xtp': xtp, 'ytp': ytp, 'vtp': vtp, 'w': w, 'cname': cname }

    def increase_tree_depth(self):
        self.depth += 1

    def get_tree_depth(self):
        return self.depth
