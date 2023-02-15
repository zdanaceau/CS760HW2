import numpy as np
from scipy.stats import entropy
from scipy.stats import mode
from DecisionTreeNode import DecisionTreeNode
class DecisionTree:
    def __init__(self, training_data:np.ndarray):
        self.training_data = training_data
    
    @staticmethod
    # Make candidate splits at each class boundary and drop any redundant splits
    def candidate_splits(data_set:np.ndarray):
        splits = list()
        split_data = data_set.copy()
        for dim in range(split_data.shape[1] - 1):
            split_data = split_data[split_data[:, dim].argsort()]
            # Identify split indices based class boundaries for sorted data
            split_indices_labels = np.abs(np.diff(split_data[:, 2])).nonzero()[0]
            split_indices_features = np.abs(np.diff(split_data[:, dim])).nonzero()[0]
           
            # Handle cases in which a split occurs between two identical features by classes
            intersection = np.intersect1d(split_indices_features, split_indices_labels)
            diff = np.setdiff1d(split_indices_labels, intersection)
            if intersection.shape[0] != split_indices_labels.shape[0]:
                replacement_splits = np.zeros(diff.shape[0], dtype=int)
                for ind, item in enumerate(diff):
                    new_ind = np.searchsorted(split_indices_features, item + 1, side='right')
                    replacement_splits[ind] = new_ind
                split_indices = np.concatenate((intersection, replacement_splits))
            else: split_indices = split_indices_labels
            dim_splits = [(dim, split_data[ind + 1, dim]) for ind in split_indices]
            splits.extend(dim_splits)
        return list(set(splits))

    # Standard entropy calculation for a set of frequencies    
    @staticmethod
    def entropy_calc(freqencies):
        total = 0
        for freq in freqencies:
            if freq != 0:
                total += (-1* freq * np.log2(freq))
        return total

    # Calculate the entropy of a dataset based on the labels
    @staticmethod
    def label_entropy(data_set, label_axis=2):
        values,counts = np.unique(data_set[:, label_axis], return_counts=True)
        frequencies = counts / np.sum(counts)
        set_entropy = DecisionTree.entropy_calc(frequencies)
        return set_entropy

    # Get split info from all splits in the dataset
    # Give (gain_ratio, split_entropy)
    @staticmethod
    def split_info(data_set, split, data_entropy):
        # Split dataset into split and its compliment
        split_dataset = data_set[np.asarray(data_set[:, split[0]] >= split[1]).nonzero()[0]]
        split_compliment = data_set[np.asarray(data_set[:, split[0]] < split[1]).nonzero()[0]]

        # Calculate probability of being in the split_pobs[0] and outstide split_pobs[1]
        split_probs = (np.array([split_dataset.shape[0], data_set.shape[0] - split_dataset.shape[0]])/ data_set.shape[0])
        split_dset_entropy = DecisionTree.label_entropy(split_dataset)
        split_comp_entropy = DecisionTree.label_entropy(split_compliment)
        split_cond_entropy = (split_probs[0] * split_dset_entropy) + (split_probs[1] * split_comp_entropy)

        # Calculate the entropy of the split
        split_entropy = DecisionTree.entropy_calc(split_probs)
        if split_entropy != 0:
            gain_ratio = (data_entropy - split_cond_entropy)/split_entropy
        
        # Allow mutual information to be recovered by multiplying gain ratio by 1e9
        else: gain_ratio = (data_entropy - split_cond_entropy) / 1e-9
        return gain_ratio, split_entropy

    @staticmethod
    # Identify best splits by gain ratio
    def best_splits(splits, data):
        data_entropy = DecisionTree.label_entropy(data)
        gain_ratios = [DecisionTree.split_info(data, split, data_entropy)[0] for split in splits]
        best_split = splits[np.argmax(gain_ratios)]
        return best_split
    
    @staticmethod
    # Recursive method to generate decision trees
    def make_subtree(data_set):
        # Case in which data at a node is empty
        if data_set.shape[0] == 0:
            return DecisionTreeNode(None, None, label=1)

        # Generate candidate splits and calculate entropy of the dataset
        splits = DecisionTree.candidate_splits(data_set)
        data_entropy = DecisionTree.label_entropy(data_set)

        # Create a leaf node if the entropy of the dataset is 0 at this point
        if data_entropy == 0:
            return DecisionTreeNode(None, None, label=mode(data_set[:, 2]).mode[0])

        #Get Gain ratio, split entropy in a tuple for each split
        split_info = [DecisionTree.split_info(data_set, split, data_entropy) for split in splits]

        # Check if any splits have an entropy of zero 
        for ind, split in enumerate(splits):
            split_set = data_set[np.asarray(data_set[:, split[0]] >= split[1]).nonzero()[0]]

            # Case in which a given split entropy is zero
            if split_info[ind][1] == 0:
                # create complementary set to the split and generate 
                split_compliment = data_set[np.asarray(data_set[:, split[0]] < split[1]).nonzero()[0]]
                leaf_label = mode(split_set[:, 2]).mode[0]
                leaf = DecisionTreeNode(None, None, label=leaf_label)
                split_node = DecisionTreeNode(leaf, DecisionTree.make_subtree(split_compliment), split_criteria=split)
                return split_node
            
        # If all splits have a gain ratio of 0 make leaf node  
        if max(split_info, key=lambda x: x[0]) == 0:
            end_node = DecisionTreeNode(None, None, label=mode(data_set[:, 2]).mode[0])
            return end_node
        
        # Generate the best split for the data set (this is redundant)

        best_split = DecisionTree.best_splits(splits, data_set)

        # Generate split and split complement and recursively make splits
        split_data = data_set[np.asarray(data_set[:, best_split[0]] >= best_split[1]).nonzero()[0]]
        split_comp = data_set[np.asarray(data_set[:, best_split[0]] < best_split[1]).nonzero()[0]]
        return DecisionTreeNode(DecisionTree.make_subtree(split_data), DecisionTree.make_subtree(split_comp), split_criteria=best_split)

    def train_dt(self):
        self.trained_dt = DecisionTree.make_subtree(self.training_data)

    def plot_dt(self):
        self.trained_dt.plot_dt(data=self.training_data)

    def classify_point(self, data_point):
        return self.trained_dt.classify_data(data_point)