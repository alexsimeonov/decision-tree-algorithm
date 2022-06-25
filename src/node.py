import copy
import pandas as pd
import uuid

from enum import Enum
from aislab.dp_feng.binenc import *
from aislab.gnrl import *
from utils.dictionary_utils import filter_dictionary

class Status(Enum):
    ROOT = 0
    DECISION = 1
    TERMINAL = 2
    LEAF = 3

class Node:
    def __init__(self, params, bin=None, parent_level=0, parent_statistics=None):
        self.params = params
        self.children = []
        self.bin = bin
        self.status = Status.TERMINAL if self.bin else Status.ROOT
        self.binning_results = None
        self.level = parent_level + 1 if self.bin else parent_level
        self.parent_statistics = parent_statistics

    def split(
        self, encoded_values,
        tree_statistics, node_statistics,
        config, tree,
        index=None, column_name=None,
        parent_id=None, parent_label='Root'):
        self_data = self.compose_self_data(
            self.bin, encoded_values['x'],
            encoded_values['y'], column_name)
        tree_statistics['nodes_count'] += 1

        # new configuration----
        old_records_length = len(encoded_values['x'])
        updated_encoded_values = copy.deepcopy(encoded_values)
        updated_encoded_values.update({ 'x': self_data['x'], 'y': self_data['y'], 'w': self_data['w'] })
        # ----

        self.binning_results = self.binning(updated_encoded_values)
        best_split = self.get_best_split(self.binning_results['sb'], column_name)
        bins = filter_dictionary(
            best_split[0]['bns'],
            lambda bin: bin['type'] == 'Normal')

        # old configuration
        # old_records_length = len(encoded_values['x'])
        # updated_encoded_values = copy.deepcopy(encoded_values)
        # updated_encoded_values.update({ 'x': self_data['x'], 'y': self_data['y'], 'w': self_data['w'] })
        # -----

        if (self.status == Status.ROOT or self.level == 1) or column_name != best_split[0]['cname']:
            parent_statistics = { 'my1': best_split[0]['st'][0]['my1'][0][0], 'my0': best_split[0]['st'][0]['my0'][0][0] }
            self.define_node_chidren(bins, parent_statistics)
        else:
            self.status = Status.LEAF

        if (len(updated_encoded_values['x']) < (2 * self.params['min_samples_split'])) or len(self.children) <= 1:
            self.status = Status.LEAF

        treelib_node_id = uuid.uuid4()
        treelib_node_label = 'Root' if self.status == Status.ROOT else self.compose_node_label(column_name, index)
        self.update_tree_structure(tree, treelib_node_id, parent_id, treelib_node_label)

        # Adding statistics for current node into the tree table
        if self.status != Status.ROOT:
            node_statistics.append(
                {
                    'parent': parent_label,
                    'split_variable': treelib_node_label,
                    'children': len(self.children) if self.status != Status.LEAF else 0,
                    'records': self.bin['n'],
                    'my0': self.parent_statistics['my0'],
                    'my1': self.parent_statistics['my1'],
                    'Gini': best_split[0]['st'][0]['Gini'][0],
                    'Chi2': best_split[0]['st'][0]['Chi2'][0]
                })

        # Splitting children
        if (self.status != Status.LEAF) and (self.level < self.params['max_depth']) and len(self.children) > 1:
            self.status = Status.DECISION
            for idx, child in enumerate(self.children):
                child.split(
                    updated_encoded_values, tree_statistics,
                    node_statistics, config, tree,
                    idx, best_split[0]['cname'],
                    parent_id=treelib_node_id, parent_label=treelib_node_label)

        if self.status == Status.LEAF:
            tree_statistics['leaf_count'] += 1

    def compose_self_data(self, bin, x, y, column_name):
        current_x = x
        current_y = y

        if self.status != Status.ROOT and bin['type'] == 'Normal':
            if len(bin['lb']) and not len(bin['rb']):
                current_x = x[(x[column_name].isin(bin['lb']))]
            elif len(bin['lb']) == 1 | len(bin['rb']) == 1:
                current_x = x[x[column_name].between(float(bin['lb'][0]), float(bin['rb'][0]))]

        current_y = y[y.index.isin(list(current_x.index))]
        current_w = ones((len(current_x.index), 1))

        return { 'x': current_x, 'y': current_y, 'w': current_w }

    def binning(self, encoded_values):
        tic()
        y = encoded_values['y'].values
        ub = ubng(
            encoded_values['x'], encoded_values['xtp'],
            encoded_values['w'], y=y,
            ytp=encoded_values['ytp'], cnames=encoded_values['cname'],
            md=self.params['max_children_count'], nmin=self.params['min_samples_split'])
        toc('UBNG finished successfully.')
        tic()
        sb = sbng(ub, md=self.params['max_children_count'])
        toc('SBNG finished successfully.')
        return { 'ub': ub, 'sb': sb }

    def get_best_split(self, binning_result, column_name):
        splits_by_different_col = list(filter(lambda variable: variable['cname'] != column_name, binning_result))
        best_split_variable = filter(
            lambda variable: variable['st'][0][self.params['criterion']][0] == max(map(lambda var: var['st'][0][self.params['criterion']][0],
            splits_by_different_col)), splits_by_different_col)
        return list(best_split_variable)

    def update_tree_structure(self, tree, id, parent_id, label):
        if self.level == 1:
            tree.create_node(label, id, parent='root')
        elif self.level != 0 and self.level != 1:
            tree.create_node(label, id, parent=parent_id)

    def compose_node_label(self, column_name, index):
        return column_name + '_' + str(self.level) + '_' + str(index)

    def define_node_chidren(self, bins, parent_statistics):
        for (key, value) in bins.items():
            if value['n'] >= self.params['min_samples_split']:
                child = self.compose_child(value, parent_statistics)
                self.children.append(child)

    def compose_child(self, bin, parent_statistics):
        return Node(self.params, bin, self.level, parent_statistics=parent_statistics)
