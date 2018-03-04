# -*- coding: utf-8 -*-
"""
Different kd-trees implementations and utility functions.

Please not that the kd-tree is only built through a function as nested tuple
object with numpy arrays or None object in leafs. This is because a complete
implementation with a proper class and all possible queries was not needed 
for the project.
"""
from compute_diameter import compute_diameter, compute_diameter_approx
from gram_schmidt import fast_gram_schmidt
import numpy as np

def kd_tree_paper(data, jitter=True, approx_diameter=False, max_leaf_size=1):
    return kd_tree_axis(np.dot(data, fast_gram_schmidt(data.shape[1])),
                        jitter=jitter,
                        approx_diameter=approx_diameter,
                        max_leaf_size=max_leaf_size)

def kd_tree_axis(data, i=0, jitter=True, approx_diameter=False, max_leaf_size=1):
    if data.shape[0] == 0:
        return None
    if data.shape[0] <= max_leaf_size:
        return data
    d = data.shape[1]
    if approx_diameter is False:
        Delta = np.sqrt(compute_diameter(data)[0])
    else:
        Delta = np.sqrt(compute_diameter_approx(data, epsilon=approx_diameter)[0])
    med = np.median(data[:,i])
    if jitter:
        delta = np.random.uniform(-1,1)* Delta/np.sqrt(d)/2.
    else:
        delta = 0.
    return (kd_tree_axis(data[data[:,i] <= med + delta], (i + 1)%d, approx_diameter=approx_diameter, jitter=jitter, max_leaf_size=max_leaf_size), 
            kd_tree_axis(data[data[:,i] > med + delta], (i + 1)%d, approx_diameter=approx_diameter, jitter=jitter, max_leaf_size=max_leaf_size),
                          Delta)
                          
def rp_tree(data, jitter=True, approx_diameter=False, max_leaf_size=1):
    if data.shape[0] == 0:
        return None
    if data.shape[0] <= max_leaf_size:
        return data
    d = data.shape[1]
    if approx_diameter is False:
        Delta = np.sqrt(compute_diameter(data)[0])
    else:
        Delta = np.sqrt(compute_diameter_approx(data, epsilon=approx_diameter)[0])
    v_dir = np.random.normal(size=(d,1))
    data_proj = np.dot(data, v_dir)
    med = np.median(data_proj[:,0])
    if jitter:
        delta = np.random.uniform(-1,1)* Delta/np.sqrt(d)/2.
    else:
        delta = 0.
    return (rp_tree(data[data_proj[:,0] <= med + delta], approx_diameter=approx_diameter, jitter=jitter, max_leaf_size=max_leaf_size), 
            rp_tree(data[data_proj[:,0] > med + delta], approx_diameter=approx_diameter, jitter=jitter, max_leaf_size=max_leaf_size),
                          Delta)
                          
def get_tree_statistics(tree):
    distribution = []
    get_branch_statistics(tree, distribution)
    means = [np.mean(level) for level in distribution]
    stds = [np.std(level) for level in distribution]
    mins = [np.min(level) for level in distribution]
    maxs = [np.max(level) for level in distribution]
    stats = [means, stds, mins, maxs]
    return distribution, stats

def get_branch_statistics(tree, distribution, depth=1):
    if len(tree) == 3:
        if len(distribution) < depth:
            distribution.append([])
        distribution[depth - 1].append(tree[2])
        if tree[0] is not None:
            get_branch_statistics(tree[0], distribution, depth=depth+1)
        if tree[1] is not None:
            get_branch_statistics(tree[1], distribution, depth=depth+1)
        
def get_count_tree(tree):
    if len(tree) == 3:
        t_1 = get_count_tree(tree[0])
        t_2 = get_count_tree(tree[1])
        return t_1[0] + t_2[0], t_1, t_2
    elif len(tree) == 1:
        return [tree.shape[0]]
    elif len(tree) == 0:
        return [0]
    else:
        print(tree)
        return None
    