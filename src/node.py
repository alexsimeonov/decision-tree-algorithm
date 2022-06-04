from enum import Enum
from aislab.dp_feng.binenc import *
from aislab.gnrl import *
import pandas as pd
from utils.dictionary_utils import filter_dictionary

class Status(Enum):
    ROOT = 0
    DECISION = 1
    TERMINAL = 2
    LEAF = 3

class Node:
    def __init__(self, params, bin=None):
        self.params = params
        self.children = [] # Add children after split
        self.bin = bin
        self.status = Status.TERMINAL if self.bin else Status.ROOT

    def split(self, x, y, config, encoded_values, increase_tree_depth, get_tree_depth, column_name=None):
        print('Node starts splitting.')
        self_data = self.compose_self_data(self.bin, x, y, column_name)

        encoded_values = { 'xe': self_data['x'], 'y': self_data['y'], 'xtp': encoded_values['xtp'], 'ytp': encoded_values['ytp'], 'vtp': encoded_values['vtp'], 'w': encoded_values['w'], 'cname': self_data['column_name']}
        binning_result = self.binning(encoded_values)
        best_split = self.get_best_split(binning_result['sb'])
        column_name = best_split[0]['cname']
        # get only the 'Normal' bins and form the node's children from them
        normal_bins = filter_dictionary(best_split[0]['bns'], lambda bin: bin['type'] == 'Normal')
        self.define_node_chidren(normal_bins)

        # if get_tree_depth() < self.params['max_depth'] and len(self.children):
        #     increase_tree_depth()
        #     self.split_children(self_data['x'], self_data['y'], config, encoded_values, increase_tree_depth, get_tree_depth, column_name)
        # print(self.children)

    def binning(self, encoded_values):
        print('ENTERS AT LEAST')
        # 2. BINNING
        tic()
        ub = ubng(encoded_values['xe'], encoded_values['xtp'], encoded_values['w'], y=encoded_values['y'], ytp=encoded_values['ytp'], cnames=encoded_values['cname'])     # unsupervised binning
        toc('UBNG finished successfully.')
        tic()
        sb = sbng(ub)       # supervised binning
        toc('SBNG finished successfully.')
        return { 'ub': ub, 'sb': sb }

    def get_best_split(self, binning_result):
        best_split_variable = filter(lambda variable: variable['st'][0]['Chi2'][0] == max(map(lambda var: var['st'][0]['Chi2'][0], binning_result)), binning_result)
        return list(best_split_variable)

    def split_children(self, x, y, config, encoded_values, increase_tree_depth, get_tree_depth, column_name):
        for child in self.children:
            child.split(x, y, config, encoded_values, increase_tree_depth, get_tree_depth, column_name)

    def define_node_chidren(self, bins):
        # currently using lists for the structures that I am building
        # discuss if this is optimal or should use dictionaries instead
        for (key, value) in bins.items():
            child = self.compose_child(value)
            self.children.append(child)

    def compose_child(self, bin):
        return Node(self.params, bin)

    def compose_self_data(self, bin, x, y, column_name):
        current_x = x
        current_y = y

        if self.status != Status.ROOT:
            if len(column_name) and not len(bin['rb']):
                current_x = x[(x[column_name].isin(bin['lb']))]
            elif len(bin['lb']) == 1 | len(bin['rb']) == 1:
                current_x = x[(column_name >= bin['lb'][0]) and (column_name <= bin['rb'][0])]

            current_y = y[y.index.isin(list(current_x.index))]

        return { 'x': current_x, 'y': current_y, 'column_name': column_name }
