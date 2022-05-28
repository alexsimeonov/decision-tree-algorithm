from node import Node
import pandas as pd

class DecisionTree:
    def __init__(self, x, y, config, hyperparams):
        self.hyperparams = hyperparams
        self.structure = []
        # Initialize the root node, using the data passed to the DT
        self.root_node = Node(self.hyperparams, x, y, config)

    def grow(self):
        print('Decision tree starts growing.')

        self.root_node.split()
