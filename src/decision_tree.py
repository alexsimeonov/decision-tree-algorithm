from node import Node
import pandas as pd
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

    def grow(self):
        print('Decision tree starts growing.')

        self.root_node.split(self.x, self.y, self.config)

    def represent_structure(self):
        represent_tree(self.structure)
