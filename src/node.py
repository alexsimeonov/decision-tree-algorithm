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
        self.binning_results = None

    def split(self, encoded_values, config, column_name=None):
        print('Node starts splitting.')
        self_data = self.compose_self_data(self.bin, encoded_values['x'], encoded_values['y'], column_name)
        self.binning_results = self.binning(encoded_values)
        best_split = self.get_best_split(self.binning_results['sb'])

        bins = filter_dictionary(best_split[0]['bns'], lambda bin: bin['type'] == 'Normal' or bin['type'] == 'Missing')
        self.define_node_chidren(bins)
        encoded_values.update({ 'x': self_data['x'], 'y': self_data['y'] })

        print('CHILDREN: ', self.children)
        # for child in self.children:
            # child.split(encoded_values, config, best_split[0]['cname'])
            # print('BIN: ', child.bin)

        print('Finished successfully!')

    def binning(self, encoded_values):
        # 2. BINNING
        tic()
        y = encoded_values['y'].values
        ub = ubng(encoded_values['x'], encoded_values['xtp'], encoded_values['w'], y=y, ytp=encoded_values['ytp'], cnames=encoded_values['cname'])     # unsupervised binning
        toc('UBNG finished successfully.')
        tic()
        sb = sbng(ub)       # supervised binning
        toc('SBNG finished successfully.')
        return { 'ub': ub, 'sb': sb }

    def get_best_split(self, binning_result):
        best_split_variable = filter(lambda variable: variable['st'][0][self.params['criterion']][0] == max(map(lambda var: var['st'][0][self.params['criterion']][0], binning_result)), binning_result)
        return list(best_split_variable)

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
            if bin['type'] == 'Normal':
                if len(bin['lb']) and not len(bin['rb']):
                    current_x = x[(x[column_name].isin(bin['lb']))]
                elif len(bin['lb']) == 1 | len(bin['rb']) == 1:
                    current_x = x[(column_name >= bin['lb'][0]) and (column_name <= bin['rb'][0])]
            elif bin['type'] == 'Missing':
                current_x = x[(x[column_name].isnull().values.any())]
                print(current_x)

        current_y = y[y.index.isin(list(current_x.index))]
        return { 'x': current_x, 'y': current_y, 'column_name': column_name }
