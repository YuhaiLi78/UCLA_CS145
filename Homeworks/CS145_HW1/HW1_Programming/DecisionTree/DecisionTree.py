#!/usr/bin/env python3
import sys
import random
import numpy as np
from sklearn.model_selection import KFold
from collections import Counter

# Class for instances with operations
class Instances(object):
    def __init__(self):
        self.label = []
        self.attrs = []
        self.num_attrs = -1
        self.num_instances = 0
        self.attr_set = []
        

    def add_instance(self, _lbl, _attrs):
        self.label.append(_lbl)
        self.attrs.append(_attrs)
        if self.num_attrs == -1:
            self.num_attrs = len(_attrs)
        else:
            assert(self.num_attrs == len(_attrs))
        self.num_instances += 1
        assert(self.num_instances == len(self.label))

    
    def make_attr_set(self):
        self.attr_set = [set([self.attrs[i][j] for i in range(self.num_instances)]) for j in range(self.num_attrs)]


    def load_file(self, file_name):
        with open(file_name, 'r') as f:
            for line in f:
                data = line.strip().split(',')
                self.add_instance(data[0], data[1:])
        self.make_attr_set()
        return self
    

    def split(self, att_idx):
        assert(0 <= att_idx < self.num_attrs)
        split_data = {x: Instances() for x in self.attr_set[att_idx]}
        for i in range(self.num_instances):
            key = self.attrs[i][att_idx] 
            split_data[key].add_instance(self.label[i], self.attrs[i])
        for key in split_data:
            split_data[key].attr_set = self.attr_set
        return split_data
    

    def shuffle(self):
        indices = list(range(len(self.label)))
        random.shuffle(indices)
        res = Instances()
        for x in indices:
            res.add_instance(self.label[x], self.attrs[x])
        res.attr_set = self.attr_set
        return res


    def get_subset(self, keys):
        res = Instances()
        for x in keys:
            res.add_instance(self.label[x], self.attrs[x])
        res.attr_set = self.attr_set
        return res


def compute_entropy(data):
    total_entropy = 0.0
    ########## Please Fill Missing Lines Here ##########

    return total_entropy
    

def compute_info_gain(data, att_idx):
    info_gain = 0.0
    ########## Please Fill Missing Lines Here ##########

    return info_gain


def comput_gain_ratio(data, att_idx):
    gain_ratio = 0.0
    ########## Please Fill Missing Lines Here ##########

    return gain_ratio


# Class of the decision tree model based on the ID3 algorithm
class DecisionTree(object):
    def __init__(self, _instances, _sel_func):
        self.instances = _instances
        self.sel_func = _sel_func
        self.gain_function = compute_info_gain if _sel_func == 0 else comput_gain_ratio
        self.m_attr_idx = None # The decision attribute if the node is a branch
        self.m_class = None # The decision class if the node is a leaf
        self.make_tree()

    def make_tree(self):
        if self.instances.num_instances == 0:
            # No any instance for this node
            self.m_class = '**MISSING**'
        else:
            gains = [self.gain_function(self.instances, i) for i in range(self.instances.num_attrs)]
            self.m_attr_idx = np.argmax(gains)
            if np.abs(gains[self.m_attr_idx]) < 1e-9:
                # A leaf to decide the decided class
                self.m_attr_idx = None
                ########## Please Fill Missing Lines Here ##########

            else:
                # A branch
                split_data = self.instances.split(self.m_attr_idx)
                self.m_successors = {x: DecisionTree(split_data[x], self.sel_func) for x in split_data}
                for x in self.m_successors:
                    self.m_successors[x].make_tree()

    def classify(self, attrs):
        assert((self.m_attr_idx != None) or (self.m_class != None))
        if self.m_attr_idx == None:
            return self.m_class
        else:
            return self.m_successors[attrs[self.m_attr_idx]].classify(attrs)
            
 

if __name__ == '__main__':
    if len(sys.argv) < 1 + 1:
        print('--usage python3 %s data [0/1, 0-Information Gain, 1-Gain Ratio, default: 0]' % sys.argv[0], file=sys.stderr)
        sys.exit(0)
    random.seed(27145)
    np.random.seed(27145)
    
    sel_func = int(sys.argv[2]) if len(sys.argv) > 1 + 1 else 0
    assert(0 <= sel_func <= 1) 

    data = Instances().load_file(sys.argv[1])
    data = data.shuffle()
    
    # 5-Fold CV
    kf = KFold(n_splits=5)
    n_fold = 0
    accuracy = []
    for train_keys, test_keys in kf.split(range(data.num_instances)):
        train_data = data.get_subset(train_keys)
        test_data = data.get_subset(test_keys)
        n_fold += 1
        model = DecisionTree(train_data, sel_func)
        predictions = [model.classify(test_data.attrs[i]) for i in range(test_data.num_instances)]
        num_correct_predictions = sum([1 if predictions[i] == test_data.label[i] else 0 for i in range(test_data.num_instances)])
        nfold_acc = float(num_correct_predictions) / float(test_data.num_instances)
        accuracy.append(nfold_acc)
        print('Fold-{}: {}'.format(n_fold, nfold_acc))

    print('5-CV Accuracy = {}'.format(np.mean(accuracy)))
        
