from typing import Tuple
# Binarytree used just for visualization purposes
from binarytree import Node
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class DecisionTreeNode:
    def __init__(self, left, right, split_criteria=None, label=None):
        self.left = left
        self.right = right
        self.split_criteria = split_criteria
        if label is not None: self.label = int(label)
        else: self.label = label
    
    
    def get_next_node(self, data):
        if data[self.split_criteria[0]] >= self.split_criteria[1]:
            return self.left
        else:
            return self.right


    def is_leaf(self):
        if self.label is not None:
            return True
        else:
            return False

    def classify_data(self, data):
        if self.is_leaf():
            return self.label
        else:
            next_node = self.get_next_node(data)
            return next_node.classify_data(data)
    
    # Recursively copy a tree to binary tree
    @staticmethod
    def convert_to_bt(node):
        if node.is_leaf():
            text_desc = "Class " + str(node.label)
            return Node(text_desc)
        else:
            if node.split_criteria[0] == 0: dimension = 'x1'
            else: dimension = 'x2'
            text_desc = f'{dimension} >= {node.split_criteria[1]}'
            return Node(value=text_desc, left=DecisionTreeNode.convert_to_bt(node.left), right=DecisionTreeNode.convert_to_bt(node.right))
    
    def to_text(self):
        bt_rep = DecisionTreeNode.convert_to_bt(self)
        return str(bt_rep)

    def plot_dt(self, data, colors=["r", "b"]):
        xbounds = (0, np.max(data[:, 0]))
        ybounds = (0, np.max(data[:, 1]))
        plt.xlim(left=xbounds[0], right=xbounds[1])
        plt.ylim(bottom=ybounds[0], top=ybounds[1])
        DecisionTreeNode.plot_dt_helper(self, xbounds, ybounds)
        class_1_data = data[data[:, 2] == 1]
        class_0_data = data[data[:, 2] == 0]
        plt.scatter(class_1_data[:,0],class_1_data[:,1], color='r', s=1, label="Class_1")
        plt.scatter(class_0_data[:,0],class_0_data[:,1], color='b', s=1, label="Class_0")
        plt.legend()


    def plot_dt_helper(node, xlim:Tuple[float, float], ylim:Tuple[float, float]):
        if node.is_leaf():
            if node.label == 1: color = 'lightcoral'
            else: color = 'lightskyblue'
            patch = patches.Rectangle(xy=[xlim[0], ylim[0]], width=(xlim[1] - xlim[0]), height=(ylim[1] - ylim[0]), color=color)
            plt.gca().add_patch(patch)
            return
        else:
            if node.split_criteria[0] == 0:
                left_xlim = (node.split_criteria[1], xlim[1])
                right_xlim = (xlim[0], node.split_criteria[1])
                left_ylim = ylim
                right_ylim = ylim
            else:
                left_xlim = xlim
                right_xlim = xlim
                left_ylim = (node.split_criteria[1], ylim[1])
                right_ylim = (ylim[0], node.split_criteria[1])
            DecisionTreeNode.plot_dt_helper(node.left, left_xlim, left_ylim)
            DecisionTreeNode.plot_dt_helper(node.right, right_xlim, right_ylim)
    def count_nodes_rec(node):
        if node.is_leaf():
            return 1
        else:
            left_count = DecisionTreeNode.count_nodes_rec(node.left)
            right_count = DecisionTreeNode.count_nodes_rec(node.right)
            return (1 + left_count + right_count)

    def count_nodes(self):
        return DecisionTreeNode.count_nodes_rec(self)
    

        
