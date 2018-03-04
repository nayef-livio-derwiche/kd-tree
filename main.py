# -*- coding: utf-8 -*-
"""
Generate all the experiments, plot and save the results.
"""

from data_generation import gaussian_distribution, uniform_distribution, hollow_hypersphere
from data_generation import mechanical_arm, swiss_roll, smooth_1d_manifold, local_covariance_dimension
from kd_tree import kd_tree_paper, kd_tree_axis, rp_tree, get_tree_statistics
import matplotlib.pyplot as plt
import numpy as np

output_folder = "outputs/"

def main_experiment():
    """
    Experiment with smooth 1D manifold described in Verma et al 2009.    
    
    Generate smooth manifolds of dimension 1 with different ambiant dimensions
    and evaluate the diameter reduction rate of the kd-tree with increased
    depth.
    """
    print("Do main experiment")
    
    data_1 = smooth_1d_manifold(20000,2)
    data_2 = smooth_1d_manifold(20000,4)
    data_3 = smooth_1d_manifold(20000,8)
    data_4 = smooth_1d_manifold(20000,16)
    data_5 = smooth_1d_manifold(20000,32)
    
    print("generate kd_tree 1")
    t_1 = kd_tree_paper(data_1, approx_diameter=0.25)
    print("generate kd_tree 2")
    t_2 = kd_tree_paper(data_2, approx_diameter=0.25)
    print("generate kd_tree 3")
    t_3 = kd_tree_paper(data_3, approx_diameter=0.25)
    print("generate kd_tree 4")
    t_4 = kd_tree_paper(data_4, approx_diameter=0.25)
    print("generate kd_tree 5")
    t_5 = kd_tree_paper(data_5, approx_diameter=0.25)
    print("done")
    
    distrib1, stats_1 = get_tree_statistics(t_1)
    distrib2, stats_2 = get_tree_statistics(t_2)
    distrib3, stats_3 = get_tree_statistics(t_3)
    distrib4, stats_4 = get_tree_statistics(t_4)
    distrib5, stats_5 = get_tree_statistics(t_5)
    
    avg_1 = np.array(stats_1[0][:20])
    avg_2 = np.array(stats_2[0][:20])
    avg_3 = np.array(stats_3[0][:20])
    avg_4 = np.array(stats_4[0][:20])
    avg_5 = np.array(stats_5[0][:20])
    
    avg_1 /= avg_1[0]
    avg_2 /= avg_2[0]
    avg_3 /= avg_3[0]
    avg_4 /= avg_4[0]
    avg_5 /= avg_5[0]
    
    plt.figure()
    plt.title("Experiment on smooth 1D manifold")
    plt.plot(np.log(avg_1), linestyle="--", marker="o", label="D=2, d=1")
    plt.plot(np.log(avg_2), linestyle="--", marker="o", label="D=4, d=1")
    plt.plot(np.log(avg_3), linestyle="--", marker="o", label="D=8, d=1")
    plt.plot(np.log(avg_4), linestyle="--", marker="o", label="D=16, d=1")
    plt.plot(np.log(avg_5), linestyle="--", marker="o", label="D=32, d=1")
    plt.ylabel("Average log-Diameter (rescaled at depth 0 = 0)")
    plt.xlabel("Depth of tree node")
    plt.legend()
    plt.savefig(output_folder + 'fig1.png', bbox_inches='tight')

def gaussian_experiment():
    """
    Experiment with normally distributed points in affine subspace.    
    
    Generate normally distributed points in affine subspace with same intrinsic
    dimension but different ambiant dimensions and evaluate the diameter 
    reduction rate of the kd-tree with increased depth.
    """
    print("Gaussian distribution")
    
    data_1 = gaussian_distribution(20000, 4, 3)
    data_2 = gaussian_distribution(20000, 12, 3)
    data_3 = gaussian_distribution(20000, 32, 3)
    
    print("generate kd_tree 1")
    t_1 = kd_tree_paper(data_1, approx_diameter=0.25)
    print("generate kd_tree 2")
    t_2 = kd_tree_paper(data_2, approx_diameter=0.25)
    print("generate kd_tree 3")
    t_3 = kd_tree_paper(data_3, approx_diameter=0.25)
    
    distrib1, stats_1 = get_tree_statistics(t_1)
    distrib2, stats_2 = get_tree_statistics(t_2)
    distrib3, stats_3 = get_tree_statistics(t_3)
    
    avg_1 = np.array(stats_1[0][:20])
    avg_2 = np.array(stats_2[0][:20])
    avg_3 = np.array(stats_3[0][:20])
    
    avg_1 /= avg_1[0]
    avg_2 /= avg_2[0]
    avg_3 /= avg_3[0]
    
    plt.figure()
    plt.title("Experiment with Gaussian distribution in 3D affine subspace")
    plt.plot(np.log(avg_1), linestyle="--", marker="o", label="D=4, d=3")
    plt.plot(np.log(avg_2), linestyle="--", marker="o", label="D=12, d=3")
    plt.plot(np.log(avg_3), linestyle="--", marker="o", label="D=32 d=3")
    plt.ylabel("Average log-Diameter (rescaled at depth 0 = 0)")
    plt.xlabel("Depth of tree node")
    plt.legend()
    plt.savefig(output_folder + 'fig2.png', bbox_inches='tight')

def uniform_experiment():
    """
    Experiment with uniformly distributed points in affine subspace.    
    
    Generate uniformly distributed points in affine subspace with same intrinsic
    dimension but different ambiant dimensions and evaluate the diameter 
    reduction rate of the kd-tree with increased depth.
    """
    print("Uniform distribution")
    
    data_1 = uniform_distribution(20000, 4, 3)
    data_2 = uniform_distribution(20000, 12, 3)
    data_3 = uniform_distribution(20000, 32, 3)
    
    print("generate kd_tree 1")
    t_1 = kd_tree_paper(data_1, approx_diameter=0.25)
    print("generate kd_tree 2")
    t_2 = kd_tree_paper(data_2, approx_diameter=0.25)
    print("generate kd_tree 3")
    t_3 = kd_tree_paper(data_3, approx_diameter=0.25)
    
    distrib1, stats_1 = get_tree_statistics(t_1)
    distrib2, stats_2 = get_tree_statistics(t_2)
    distrib3, stats_3 = get_tree_statistics(t_3)
    
    avg_1 = np.array(stats_1[0][:20])
    avg_2 = np.array(stats_2[0][:20])
    avg_3 = np.array(stats_3[0][:20])
    
    avg_1 /= avg_1[0]
    avg_2 /= avg_2[0]
    avg_3 /= avg_3[0]
    
    plt.figure()
    plt.title("Experiment with Uniform distribution in 3D affine subspace")
    plt.plot(np.log(avg_1), linestyle="--", marker="o", label="D=4, d=3")
    plt.plot(np.log(avg_2), linestyle="--", marker="o", label="D=12, d=3")
    plt.plot(np.log(avg_3), linestyle="--", marker="o", label="D=32 d=3")
    plt.ylabel("Average log-Diameter (rescaled at depth 0 = 0)")
    plt.xlabel("Depth of tree node")
    plt.legend()
    plt.savefig(output_folder + 'fig3.png', bbox_inches='tight')

def hollow_gauss_uniform_experiment():
    """
    Experiment with uniformly distributed points on a hypersphere's surface.    
    
    Generate uniformly points on an hypersphare and compare it with other point
    sets of similar expected intrinsic dimension in term of diameter reduction 
    rate of the kd-tree with increased depth.
    """
    print("Hollow hypersphere vs Gaussian Distribution vs Uniform Distribution (4D manifold)")
    
    data_1 = hollow_hypersphere(20000, 5)
    data_2 = uniform_distribution(20000, 5, 4)
    data_3 = gaussian_distribution(20000, 10, 4)
    
    print("generate kd_tree 1")
    t_1 = kd_tree_paper(data_1, approx_diameter=0.25)
    print("generate kd_tree 2")
    t_2 = kd_tree_paper(data_2, approx_diameter=0.25)
    print("generate kd_tree 3")
    t_3 = kd_tree_paper(data_3, approx_diameter=0.25)
    
    distrib1, stats_1 = get_tree_statistics(t_1)
    distrib2, stats_2 = get_tree_statistics(t_2)
    distrib3, stats_3 = get_tree_statistics(t_3)
    
    avg_1 = np.array(stats_1[0][:20])
    avg_2 = np.array(stats_2[0][:20])
    avg_3 = np.array(stats_3[0][:20])
    
    avg_1 /= avg_1[0]
    avg_2 /= avg_2[0]
    avg_3 /= avg_3[0]
    
    plt.figure()
    plt.title("Comparison with Hollow Hypersphere, Gaussian and Uniform")
    plt.plot(np.log(avg_1), linestyle="--", marker="o", label="Hollow Sphere, D=5, d=4")
    plt.plot(np.log(avg_2), linestyle="--", marker="o", label="Uniform, D=5, d=4")
    plt.plot(np.log(avg_3), linestyle="--", marker="o", label="Gaussian, D=10 d=4")
    plt.ylabel("Average log-Diameter (rescaled at depth 0 = 0)")
    plt.xlabel("Depth of tree node")
    plt.legend()
    plt.savefig(output_folder + 'fig4.png', bbox_inches='tight')

def mecharm_smooth1D_uniform_experiment():
    """
    Experiment with uniformly distributed points on a hypersphere's surface.    
    
    Generate uniformly points on an hypersphare and compare it with other point
    sets of similar expected intrinsic dimension in term of diameter reduction 
    rate of the kd-tree with increased depth.
    """
    print("Mechanical Arm vs 1D smooth manifold vs Uniform Distribution (1D manifold)")
    
    data_1 = mechanical_arm(20000, 3, 1)
    data_2 = smooth_1d_manifold(20000, 8)
    data_3 = uniform_distribution(20000, 16, 1)
    
    print("generate kd_tree 1")
    t_1 = kd_tree_paper(data_1, approx_diameter=0.25)
    print("generate kd_tree 2")
    t_2 = kd_tree_paper(data_2, approx_diameter=0.25)
    print("generate kd_tree 3")
    t_3 = kd_tree_paper(data_3, approx_diameter=0.25)
    
    distrib1, stats_1 = get_tree_statistics(t_1)
    distrib2, stats_2 = get_tree_statistics(t_2)
    distrib3, stats_3 = get_tree_statistics(t_3)
    
    avg_1 = np.array(stats_1[0][:20])
    avg_2 = np.array(stats_2[0][:20])
    avg_3 = np.array(stats_3[0][:20])
    
    avg_1 /= avg_1[0]
    avg_2 /= avg_2[0]
    avg_3 /= avg_3[0]
    
    plt.figure()
    plt.title("Comparison with Mechanical Arm, 1D smooth manifold and Uniform")
    plt.plot(np.log(avg_1), linestyle="--", marker="o", label="Mechanical Arm, D=3, d=1")
    plt.plot(np.log(avg_2), linestyle="--", marker="o", label="1D Smooth, D=8, d=1")
    plt.plot(np.log(avg_3), linestyle="--", marker="o", label="Uniform, D=16 d=1")
    plt.ylabel("Average log-Diameter (rescaled at depth 0 = 0)")
    plt.xlabel("Depth of tree node")
    plt.legend()
    plt.savefig(output_folder + 'fig5.png', bbox_inches='tight')

def swissroll_mecharm_uniform():
    """
    Experiment with randomly distributed points on the swissroll.    
    
    Generate randomly points on a "swissroll" and compare it with other point
    sets of similar expected intrinsic dimension in term of diameter reduction 
    rate of the kd-tree with increased depth.
    """
    print("Swiss Roll vs Mechanical Arm vs Uniform")
    
    data_1 = swiss_roll(20000)
    data_2 = mechanical_arm(20000,3, 2)
    data_3 = mechanical_arm(20000,6, 3)
    data_4 = uniform_distribution(20000,8, 3)
    
    print("generate kd_tree 1")
    t_1 = kd_tree_paper(data_1, approx_diameter=0.25)
    print("generate kd_tree 2")
    t_2 = kd_tree_paper(data_2, approx_diameter=0.25)
    print("generate kd_tree 3")
    t_3 = kd_tree_paper(data_3, approx_diameter=0.25)
    print("generate kd_tree 4")
    t_4 = kd_tree_paper(data_4, approx_diameter=0.25)
    print("done")
    
    distrib1, stats_1 = get_tree_statistics(t_1)
    distrib2, stats_2 = get_tree_statistics(t_2)
    distrib3, stats_3 = get_tree_statistics(t_3)
    distrib4, stats_4 = get_tree_statistics(t_4)
    
    avg_1 = np.array(stats_1[0][:20])
    avg_2 = np.array(stats_2[0][:20])
    avg_3 = np.array(stats_3[0][:20])
    avg_4 = np.array(stats_4[0][:20])
    
    avg_1 /= avg_1[0]
    avg_2 /= avg_2[0]
    avg_3 /= avg_3[0]
    avg_4 /= avg_4[0]
    
    plt.figure()
    plt.title("Swiss Roll vs Mechanical Arm vs Uniform")
    plt.plot(np.log(avg_1), linestyle="--", marker="o", label="Swiss Roll, D=3, d=2")
    plt.plot(np.log(avg_2), linestyle="--", marker="o", label="Mechanical Arm, D=6, d=2")
    plt.plot(np.log(avg_3), linestyle="--", marker="o", label="Mechanical Arm, D=18, d=3")
    plt.plot(np.log(avg_4), linestyle="--", marker="o", label="Uniform, D=8, d=3")
    plt.ylabel("Average log-Diameter (rescaled at depth 0 = 0)")
    plt.xlabel("Depth of tree node")
    plt.legend()
    plt.savefig(output_folder + 'fig6.png', bbox_inches='tight')

def rp_tree_experiment():
    """
    Compare random kd-tree of the paper with random projection trees.
    """
    print("Compare with RP-tree")
    
    data_1 = uniform_distribution(20000,10,3)
    data_2 = uniform_distribution(20000,10,3)
    data_3 = uniform_distribution(20000,50,3)
    data_4 = uniform_distribution(20000,50,3)
    
    print("generate kd_tree 1")
    t_1 = kd_tree_paper(data_1, approx_diameter=0.25)
    print("generate kd_tree 2")
    t_2 = rp_tree(data_2, approx_diameter=0.25)
    print("generate kd_tree 3")
    t_3 = kd_tree_paper(data_3, approx_diameter=0.25)
    print("generate kd_tree 4")
    t_4 = rp_tree(data_4, approx_diameter=0.25)
    print("done")
    
    distrib1, stats_1 = get_tree_statistics(t_1)
    distrib2, stats_2 = get_tree_statistics(t_2)
    distrib3, stats_3 = get_tree_statistics(t_3)
    distrib4, stats_4 = get_tree_statistics(t_4)
    
    avg_1 = np.array(stats_1[0][:17])
    avg_2 = np.array(stats_2[0][:17])
    avg_3 = np.array(stats_3[0][:17])
    avg_4 = np.array(stats_4[0][:17])
    
    avg_1 /= avg_1[0]
    avg_2 /= avg_2[0]
    avg_3 /= avg_3[0]
    avg_4 /= avg_4[0]
    
    plt.figure()
    plt.title("Random kd-tree vs RP tree, uniform distribution")
    plt.plot(np.log(avg_1), linestyle="--", marker="o", label="Random, D=10, d=3")
    plt.plot(np.log(avg_2), linestyle="--", marker="o", label="RP, D=10, d=3")
    plt.plot(np.log(avg_3), linestyle="--", marker="o", label="Random, D=50, d=3")
    plt.plot(np.log(avg_4), linestyle="--", marker="o", label="RP, D=50, d=3")
    plt.ylabel("Average log-Diameter (rescaled at depth 0 = 0)")
    plt.xlabel("Depth of tree node")
    plt.legend()
    plt.savefig(output_folder + 'fig7.png', bbox_inches='tight')

#comparison with axis-aligned tree
def axis_aligned_experiment():
    """
    Compare  random kd-tree with axis-aligned kd-tree.
    """
    print("Compare with basic axis-aligned tree")
    
    data_1 = uniform_distribution(20000,10,3)
    data_2 = uniform_distribution(20000,10,3)
    data_3 = uniform_distribution(20000,20,3)
    data_4 = uniform_distribution(20000,20,3)
    
    print("generate kd_tree 1")
    t_1 = kd_tree_paper(data_1, approx_diameter=0.25)
    print("generate kd_tree 2")
    t_2 = kd_tree_axis(data_2, approx_diameter=0.25)
    print("generate kd_tree 3")
    t_3 = kd_tree_paper(data_3, approx_diameter=0.25)
    print("generate kd_tree 4")
    t_4 = kd_tree_axis(data_4, approx_diameter=0.25)
    print("done")
    
    distrib1, stats_1 = get_tree_statistics(t_1)
    distrib2, stats_2 = get_tree_statistics(t_2)
    distrib3, stats_3 = get_tree_statistics(t_3)
    distrib4, stats_4 = get_tree_statistics(t_4)
    
    avg_1 = np.array(stats_1[0][:20])
    avg_2 = np.array(stats_2[0][:20])
    avg_3 = np.array(stats_3[0][:20])
    avg_4 = np.array(stats_4[0][:20])
    
    avg_1 /= avg_1[0]
    avg_2 /= avg_2[0]
    avg_3 /= avg_3[0]
    avg_4 /= avg_4[0]
    
    plt.figure()
    plt.title("Random kd-tree vs axis-aligned tree, uniform distribution")
    plt.plot(np.log(avg_1), linestyle="--", marker="o", label="Random, D=10, d=3")
    plt.plot(np.log(avg_2), linestyle="--", marker="o", label="Axis, D=10, d=3")
    plt.plot(np.log(avg_3), linestyle="--", marker="o", label="Random, D=20, d=3")
    plt.plot(np.log(avg_4), linestyle="--", marker="o", label="Axis aligned, D=20, d=3")
    plt.ylabel("Average log-Diameter (rescaled at depth 0 = 0)")
    plt.xlabel("Depth of tree node")
    plt.legend(loc=3)
    plt.savefig(output_folder + 'fig8.png', bbox_inches='tight')

def local_covariance_dimension_estimation():
    """
    Estimate local covariance dimension on the smooth 1D manifold dataset.
    """
    
    data = smooth_1d_manifold(5000,20)
    data  += np.random.normal(0, 0.001, size=data.shape)
    loc_dim = local_covariance_dimension(data, epsilon=0.025, n_samples=200, n_scales=30)
    plt.figure()
    plt.title("Local Covariance Dimension estimation on 1D manifold in 10D space")
    plt.plot(loc_dim, linestyle="--", marker="o")
    plt.ylabel("Dimension")
    plt.xlabel("Log-scale of sample, maximum is set diameter")
    plt.savefig(output_folder + 'fig9.png', bbox_inches='tight')
    
def main():
    main_experiment()
    print("Do with other datasets")
    gaussian_experiment()
    uniform_experiment()
    hollow_gauss_uniform_experiment()
    mecharm_smooth1D_uniform_experiment()
    swissroll_mecharm_uniform()
    rp_tree_experiment()
    axis_aligned_experiment()
    local_covariance_dimension_estimation()
    
if __name__ == "__main__":
    main()