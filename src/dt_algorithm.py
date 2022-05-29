import os
import sys
import pandas as pd

from node import Node
from decision_tree import DecisionTree
from enum import Enum

wpath = '/Users/alexandersimeonov/Library/Python/2.7/lib/python/site-packages'
# path to dataset
x_path = '/Users/alexandersimeonov/Documents/Development/University/decision-tree-algorithm/data/x_samples_250000.csv'
# path to output values
y_path = '/Users/alexandersimeonov/Documents/Development/University/decision-tree-algorithm/data/y_samples_250000.csv'
config_path = '/Users/alexandersimeonov/Documents/Development/University/decision-tree-algorithm/data/cnf_LC.csv'

sys.path.append(wpath)
os.chdir(wpath)

class Criterion(Enum):
    GINI = 'gini'
    ENTROPY = 'entropy'

print('The Criterion used by the binning algorithm is Chi2')
max_depth = input('Enter desired max depth of decision tree: ')
min_samples_split = input('Enter minimum samples in a node after split: ')
min_samples_leaf = input('Enter minimum samples in a leaf node: ')
max_children_count = input('Enter maximum number of children per node: ')

hyperparams = {}

if max_depth:
    hyperparams['max_depth'] = max_depth
if min_samples_split:
    hyperparams['min_samples_split'] = min_samples_split
if min_samples_leaf:
    hyperparams['min_samples_leaf'] = min_samples_leaf
if max_children_count:
    hyperparams['max_children_count'] = max_children_count

x = pd.read_csv(x_path)
y = pd.read_csv(y_path)
config = pd.read_csv(config_path)

decision_tree = DecisionTree(x, y, config, hyperparams)
decision_tree.grow()
decision_tree.represent_structure()
