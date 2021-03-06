# -*- coding: utf-8 -*-
"""
Diameter computation with vanilla python only.
"""

import numpy as np

def furthest_from_idx(data, idxs, idx):
    """
    find point furthest from point of index idx in data, for all index idxs
    """   
    curr_max = -1
    for q in idxs:
        q_vec = data[q,:] - data[idx,:]
        dist = np.dot(q_vec,q_vec)
        if dist > curr_max:
            q_max = q
            curr_max = dist
    return curr_max, q_max
    
def furthest_from_point(data, idxs, p):
    """
    Find point furthest from point p in data, for all index idxs    
    """  
    curr_max = -1
    for q in idxs:
        q_vec = data[q,:] - p
        dist = np.dot(q_vec,q_vec)
        if dist > curr_max:
            q_max = q
            curr_max = dist
    return curr_max, q_max
    
def compute_Q(data, idxs, pq, dpq):
    """
    Compute Q, the set of idxs of points outside the ball defined by pq and dpq.
    
    The ball is defined as such: it is of center the center of the segment of
    points of index pq[0] and pq[1] and of diameter dpq, thus square radius of 
    dpq/4.
    """
    c = (data[pq[0],:] + data[pq[1],:])/2
    Q = set()
    for idx in idxs:
        d_vec = c - data[idx,:]
        if np.dot(d_vec,d_vec) > dpq/4.:
            Q.add(idx)
    return Q
   
def compute_double_normal(data, idxs, p):
    """
    Find a double normal in data, aming index idxs.
    
    Start with p. A double normal is defined as a pair of points (p,q), 
    such that p is furthest from q and q furthest from p.
    """
    converged = False
    dpq = - 1
    while not converged:
        idxs.remove(p)
        curr_max, q = furthest_from_idx(data, idxs, p)
        if curr_max > dpq:
            dpq = curr_max
            pq = (p, q)
            p = q
            converged = False
        else:
            converged = True
    return idxs, dpq, pq

def search_double_normals(data):
    """
    Iteratively search for bigger double normals in hope of finding diameter.
    
    The procedure is very fast (linear), may find the diameter immediatly, 
    always find an approximation of diameter D by finding d such that :
    d <= D <= sqrt(3), and often find a very close approximation.
    """
    finish = False
    idxs = set(np.arange(data.shape[0]))
    S = []
    
    idxs, dpq, pq = compute_double_normal(data, idxs, 0)
    d, pq_minus_1 = dpq, pq
    d = dpq
    Q = compute_Q(data, idxs, pq, d)
    if Q:
        c = (data[pq[0],:] + data[pq[1],:])/2
        _, m = furthest_from_point(data, Q, c)
    
    while Q and not finish:
        pq_minus_1 = pq
        idxs, dpq, pq = compute_double_normal(data, idxs, m)
        if dpq > d:
            S.append(pq_minus_1)
            d = dpq
            Q = compute_Q(data, idxs, pq, d)
            if Q:
                c = (data[pq[0],:] + data[pq[1],:])/2
                _, m = furthest_from_point(data, Q, c)
        else:
            S.append(pq)
            finish = True
    return idxs, pq, pq_minus_1, dpq, d, finish, S
    
def filter_set_1(data, Q, d, c):
    """
    Compute intersection of Q with ball centered at c, of square diameter d.
    """
    S1 = set()
    for q_idx in Q:
        b_vec = data[q_idx,:] - c
        b_dist = np.dot(b_vec,b_vec)
        if b_dist <= d/4.:
            S1.add(q_idx)
    return S1
    
def filter_set_2(data, idxs, Q, d, c):
    """
    Compute idxs excluding Q and ball of center c and square diameter d.
    """
    S2 = set()
    for p_idx in idxs:
        b_vec = data[p_idx,:] - c
        b_dist = np.dot(b_vec,b_vec)
        if b_dist > d/4.:
            if p_idx not in Q:
                S2.add(p_idx)
    return S2
    
def exhaustive_search(data, S1, S2):
    """
    Exhaustive search of maximum distance between points of sets S1 and S2
    """
    max_d = -1
    max_pq = -1, -1
    for q_idx in S1:
        for p_idx in S2:
            delta_vec = data[q_idx,:] - data[p_idx,:]
            new_d = np.dot(delta_vec, delta_vec)
            if new_d > max_d:
                max_d = new_d
                max_pq = p_idx, q_idx
    return max_d, max_pq
    
def exhaustive_search_approximate(data, S1, S2, epsilon, up_bound, verbose=False):
    """
    
    """
    max_d = -1
    max_pq = -1, -1
    for q_idx in S1:
        for p_idx in S2:
            delta_vec = data[q_idx,:] - data[p_idx,:]
            new_d = np.dot(delta_vec, delta_vec)
            if new_d > max_d:
                max_d = new_d
                max_pq = p_idx, q_idx
                c = (data[p_idx,:] + data[q_idx,:])/2.
                new_up_bound, _ = furthest_from_point(data, S2, c)
                if 4 * new_up_bound < up_bound:
                    up_bound = 4 * new_up_bound
                    if verbose:
                        print("Current Upper Bound")
                        print(up_bound)
            if up_bound <= max_d + epsilon:
                return max_d, max_pq
    return max_d, max_pq
    
def Q_without_B(data, Q, d, c):
    """
    Compute Q without ball B of center c and square diameter d.
    """
    Q_new = set()
    for q_idx in Q:
        b_vec = data[q_idx,:] - c
        b_dist = np.dot(b_vec,b_vec)
        if b_dist > d/4.:
            Q_new.add(q_idx)
    return Q_new

def compute_diameter(data, brute_force_treshold = 10):
    """
    Compute diameter of point set data using Malandain, Boissonat approach.
    """
    idxs = set(np.arange(data.shape[0]))
    if data.shape[0] <= 1:
        return None
    if data.shape[0] == 2:
        return np.dot(data[0,:] - data[1,:], data[0,:] - data[1,:]), (0, 1)
    if data.shape[0] <= brute_force_treshold:
        return exhaustive_search(data, idxs, idxs)
    idxs, pq, fin_pq, dpq, d, finish, S = search_double_normals(data)
    #print(S)
    if not finish:
        return dpq, pq
    else:
        Q = compute_Q(data, idxs, fin_pq, d)
        if not Q:
            return d, fin_pq
        while S:
            #print("Q_len")
            #print(len(Q))
            pi_qi = S.pop(-1)
            c = (data[pi_qi[0],:] + data[pi_qi[1],:])/2.
            #print("Sets size")
            S1 = filter_set_1(data, Q, d, c)
            #print(len(S1))
            S2 = filter_set_2(data, idxs, Q, d, c)
            #print(len(S2))
            max_d, max_pq = exhaustive_search(data, S1, S2)
            if max_d > d:
                d = max_d
                fin_pq = max_pq
                S.append(max_pq)
            Q = Q_without_B(data, Q, d, c)
            if not Q:
                return d, fin_pq
        max_d, max_pq = exhaustive_search(data, Q, idxs)
        if max_d > d:
            return max_d, max_pq
        else:
            return d, fin_pq

def compute_initial_upbound(data, idxs, d, c, epsilon):
    """
    Compute upbound, diameter of ball centered at c if it is higher than d
    """
    R = set()
    for p in idxs:
        d_vec = data[p,:] - c
        d_dist = np.dot(d_vec, d_vec)
        if d_dist <= (d + epsilon)/4. and d_dist > d/4.:
            R.add(p)
    curr_max, _ = furthest_from_point(data, R, c)
    print(4 * curr_max)
    return 4 * curr_max

def compute_diameter_approx(data, epsilon = 0.1, brute_force_treshold = 10, verbose=False):
    """
    Compute diameter of point set with relative precision epsilon.
    """
    idxs = set(np.arange(data.shape[0]))    
    if data.shape[0] <= 1:
        return None
    if data.shape[0] == 2:
        return np.dot(data[0,:] - data[1,:], data[0,:] - data[1,:]), (0, 1)
    if data.shape[0] <= brute_force_treshold:
        return exhaustive_search(data, idxs, idxs)
    idxs, pq, fin_pq, dpq, d, finish, S = search_double_normals(data)
    epsilon = np.sqrt(epsilon) * d
    d_up_bound = 3.*d
    if verbose: 
        print(d)
        print(S)
    if not finish:
        return dpq, pq
    else:
        Q = compute_Q(data, idxs, fin_pq, d + epsilon)
        if not Q:
            return d, fin_pq
        c = (data[fin_pq[0],:] + data[fin_pq[1],:])/2.
        #d_up_bound = compute_initial_upbound(data, idxs, d, c, epsilon)
        while S:
            
            if verbose: print("Q_len")
            if verbose: print(len(Q))
            
            pi_qi = S.pop(0)
            c = (data[pi_qi[0],:] + data[pi_qi[1],:])/2.
            new_up_bound, _ = furthest_from_point(data, idxs, c)
            if 4 * new_up_bound < d_up_bound:
                d_up_bound = 4 * new_up_bound
                
            if verbose: print("Current Upper Bound")
            if verbose: print(d_up_bound)
                
            if d_up_bound <= d + epsilon:
                return d, fin_pq
                
            if verbose: print("Sets size")
            S1 = filter_set_1(data, Q, d + epsilon, c)
            if verbose: print(len(S1))
            S2 = filter_set_2(data, idxs, Q, d + epsilon, c)
            if verbose: print(len(S2))
                
            max_d, max_pq = exhaustive_search(data, S1, S2)
            if max_d > d:
                d = max_d
                fin_pq = max_pq
                S.append(max_pq)
            Q = Q_without_B(data, Q, d + epsilon, c)
            if not Q:
                return d, fin_pq
        if verbose: print("Approximate Exhaustive Search")
        max_d, max_pq = exhaustive_search_approximate(data, Q, idxs, epsilon, d_up_bound, verbose=verbose)
        if max_d > d:
            return max_d, max_pq
        else:
            return d, fin_pq

def brute_force_diameter(data):
    """
    Use brute force to find diameter
    """
    d_max = -1
    for i in range(data.shape[0]):
        for j in range(i):
            d_vec = data[i,:] - data[j,:]
            d = np.dot(d_vec, d_vec)
            if d > d_max:
                max_p = i
                max_q = j
                d_max = d
    return d_max, (max_p, max_q)