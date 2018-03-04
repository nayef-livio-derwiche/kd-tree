# -*- coding: utf-8 -*-
"""
Code for the generation of synthetic data for experiments.
"""

import numpy as np
from scipy import ndimage
from scipy.misc import imresize
from compute_diameter import compute_diameter

def gaussian_distribution(n, ambiant_dim, intrinsic_dim):
    """
    Generate random points following a standard gaussian distribution.
    
    Generate n points, of ambiant dimension ambiant_dim on an affine subspace
    of dimension intrinsic_dim. The affine subspace is aligned with the first
    axes.
    """
    
    data = np.zeros((n, ambiant_dim))
    data[:,:intrinsic_dim] = np.random.normal(size=(n, intrinsic_dim))
    return data

def uniform_distribution(n, ambiant_dim, intrinsic_dim):
    """
    Generate random points following a uniform distribution.
    
    Generate n points, of ambiant dimension ambiant_dim on an affine subspace
    of dimension intrinsic_dim. The affine subspace is aligned with the first
    axes.
    """
    data = np.zeros((n, ambiant_dim))
    data[:,:intrinsic_dim] = np.random.uniform(size=(n, intrinsic_dim))
    return data
    
def hollow_hypersphere(n, d):
    """
    Generate point uniformly distributed on the surface of a sphere of dim d.
    
    The intrinsic dimension is therefore d-1.
    """
    data = np.random.normal(size=(n,d))
    data = data / np.expand_dims(np.sqrt(np.square(data).sum(axis=1)), axis=-1)
    return data

def mechanical_arm(n, space_dim, number_of_joints):
    """
    Generate points encoding the position of an arm with given number of joints.
    
    The root join is attached at the origin, and each joint can rotate through
    one given axis, so the intrinsic dimension and degrees of freedom is equal
    to the number of joint and the position is encoded in an ambiant space of
    dimension space_dim * number_of_joints.
    """
    data = np.zeros((n, space_dim * number_of_joints))
    for i in range(number_of_joints):
        random_vec_on_unit_circle = hollow_hypersphere(n, 2)
        random_rotation_on_one_axis = np.zeros((n, space_dim))
        random_rotation_on_one_axis[:,:2] = random_vec_on_unit_circle
        random_rotation_on_one_axis = np.dot(random_rotation_on_one_axis,
                                             np.random.rand(space_dim, space_dim))
        if i == 0:
            data[:,:space_dim] = random_rotation_on_one_axis
        else:
            data[:,i*space_dim:(i+1)*space_dim] = data[:,(i-1)*space_dim:(i)*space_dim] + random_rotation_on_one_axis
    return data
    
def swiss_roll(n, turns=3):
    """
    Generate points located on the famous swiss roll manifold.
    
    "Turns" encode how many time the roll wraps arount itself.
    """
    data = np.zeros((n,3))
    data[:,0:1] = np.random.rand(n,1)
    r = np.random.rand(n,1)
    data[:,1:2] = r * np.cos(r * 2 * np.pi * turns)
    data[:,2:3] = r * np.sin(r * 2 * np.pi * turns)
    return data

def rotation_of_picure_bw(n, img, size=(19,19)):
    """
    Generate datapoint by rotating a picture (encoded in grayscale).
    
    The intrinsic dimension should be 1, way lower than the ambiant dimension
    which is equal to x_size * y_size.
    """
    img = ndimage.imread(img, mode='L')
    img  = imresize(img, size)
    data = np.zeros((n, size[0]*size[1]))
    for i in range(data.shape[0]):
        data[i,:] = ndimage.rotate(img, 360 * np.random.uniform(), reshape=False).flatten()
    return data
    
def smooth_1d_manifold(n, d):
    """
    Generate points on a smooth 1D manifold described in Verma et al 2009.
    """
    if d%2 != 0:
        d += 1
    t = np.random.uniform(0, 2*np.pi, size=(n,1))
    data = np.zeros((n,d))
    for i in range(int(d/2)):
        data[:,2*i:2*i+1] = np.sqrt(2./d) * np.sin(t * (i + 1))
        data[:,2*i+1:2*i+2] = np.sqrt(2./d) * np.cos(t * (i + 1))
    return data
    
def local_covariance_dimension(data, epsilon=0.001, n_samples=100, n_scales=20, ball_treshold=10):
    """
    Estimate local covariance dimension by sampling.
    
    Data is sampled at various points and scale of the dataset. A minimum 
    number of sample is required for the sample to be meaningful. It should be
    around an order of magnitude bigger than the intrinsic dimension.
    """    
    
    delta = np.sqrt(compute_diameter(data)[0])
    samples = np.random.choice(data.shape[0], n_samples, replace=False)
    scales = np.square(np.exp(np.linspace(np.log(0 + 1./n_scales), np.log(1), num=n_scales)) * delta)
    dim_at_scale = []
    for scale in scales:
        dim_at_scale.append([])
        for sample in samples:
            ball = []
            for x in data:
                d_vec = data[sample] - x
                d_dist = np.dot(d_vec, d_vec)
                if d_dist < scale:
                    ball.append(x)
            if len(ball) > ball_treshold:
                ball = np.array(ball)
                c = np.cov(ball, rowvar=False)
                e = np.sort(np.linalg.eigvals(c))[::-1]
                e = np.cumsum(e)
                e = e/e[-1]
                loc_dim = np.zeros_like(e)
                loc_dim[e < 1 - epsilon] = 1
                loc_dim = int(np.sum(loc_dim)) + 1
                dim_at_scale[-1].append(loc_dim)
    return [np.mean(dim) for dim in dim_at_scale] 
            
            
   
