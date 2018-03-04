# -*- coding: utf-8 -*-
"""
Debug script to setup the performance test that were done in the interactive 
console.
"""

from compute_diameter import compute_diameter, compute_diameter_approx
from compute_diameter_old import compute_diameter as compute_diameter_old
from compute_diameter_old import compute_diameter_approx as compute_diameter_approx_old
from data_generation import smooth_1d_manifold
from compute_diameter import exhaustive_search, index_array_test
from compute_diameter_old import exhaustive_search as exhaustive_search_old
import numpy as np

data = smooth_1d_manifold(1000,8)
idxs = set(np.arange(data.shape[0]))