from node import Node
import pandas as pd

class DecisionTree:
    def __init__(self, x, y, config, hyperparams):
        self.x_path = x
        self.y_path = y
        self.config_path = config
        self.data_set = pd.read_csv(self.config_path)
        self.hyperparams = hyperparams
        self.structure = None
        # Initialize the root node, using the data passed to the DT
        self.root_node = Node(self.hyperparams, self.x_path, self.y_path, self.config_path)

    def grow(self):
        print('Decision tree starts growing.')

        self.root_node.split()
