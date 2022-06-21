from enum import Enum
from aislab.dp_feng.binenc import *
from aislab.gnrl import *
import pandas as pd
from utils.dictionary_utils import filter_dictionary
import copy

class Status(Enum):
    ROOT = 0
    DECISION = 1
    TERMINAL = 2
    LEAF = 3

class Node:
    def __init__(self, params, bin=None, parent_level=0):
        self.params = params
        self.children = [] # Add children after split
        self.bin = bin
        self.status = Status.TERMINAL if self.bin else Status.ROOT
        self.binning_results = None
        self.level = parent_level + 1 if self.bin else parent_level

    def split(self, encoded_values, config, column_name=None):
        print('Node starts splitting.')
        self_data = self.compose_self_data(self.bin, encoded_values['x'], encoded_values['y'], column_name)
        print('Per Node:', self.status, 'inc:', len(encoded_values['y']), 'self:', len(self_data['y']), column_name, self.bin)
        self.binning_results = self.binning(encoded_values)
        best_split = self.get_best_split(self.binning_results['sb'])

        bins = filter_dictionary(best_split[0]['bns'], lambda bin: bin['type'] == 'Normal')
        self.define_node_chidren(bins)
        old_records_length = len(encoded_values['x'])
        updated_encoded_values = copy.deepcopy(encoded_values)
        updated_encoded_values.update({ 'x': self_data['x'], 'y': self_data['y'], 'w': self_data['w'] })

        if len(updated_encoded_values['y']) == 0 or (self.status != Status.ROOT and len(updated_encoded_values['x']) != old_records_length):
            self.status = Status.LEAF

        if (self.status != Status.LEAF) and (self.level < self.params['max_depth']) and (len(updated_encoded_values['y'])) and (self.status == Status.ROOT or len(updated_encoded_values['x']) != old_records_length):
            for idx, child in enumerate(self.children):
                print('SPLITTING CHILD AT LEVEL', self.level, ':', idx + 1)
                print(child.bin)
                child.split(updated_encoded_values, config, best_split[0]['cname'])

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
        for (key, value) in bins.items():
            child = self.compose_child(value)
            self.children.append(child)

    def compose_child(self, bin):
        return Node(self.params, bin, self.level)

    def compose_self_data(self, bin, x, y, column_name):
        current_x = x
        current_y = y

        if self.status != Status.ROOT:
            if bin['type'] == 'Normal':
                if len(bin['lb']) and not len(bin['rb']):
                    current_x = x[(x[column_name].isin(bin['lb']))]
                elif len(bin['lb']) == 1 | len(bin['rb']) == 1:
                    current_x = x[(column_name >= bin['lb'][0]) and (column_name <= bin['rb'][0])]

        current_y = y[y.index.isin(list(current_x.index))]
        current_w = ones((len(current_x.index), 1))

        return { 'x': current_x, 'y': current_y, 'w': current_w, 'column_name': column_name }
