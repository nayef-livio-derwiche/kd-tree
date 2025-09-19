# KD-Tree Implementation for Dimensionality Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Project Overview

This project implements kd-trees (k-dimensional trees) for analyzing high-dimensional data. KD-trees are binary trees that partition data along coordinate axes, enabling efficient range searches, nearest neighbor searches, and dimensionality analysis.

**Key Use Cases:**
- Dimensionality reduction analysis
- Manifold learning
- Nearest neighbor search
- Range queries in high-dimensional spaces
- Geometric data analysis

## ✨ Features

- **Multiple KD-Tree Variants:**
  - Standard axis-aligned kd-tree
  - Random projection trees (RP-trees)
  - Paper-specific implementation with Gram-Schmidt orthogonalization

- **Advanced Techniques:**
  - Approximate diameter computation for efficiency
  - Jittering for robust partitioning
  - Configurable leaf size

- **Comprehensive Analysis Tools:**
  - Tree statistics collection
  - Branch statistics analysis
  - Node counting and distribution metrics

- **Performance Optimizations:**
  - Cython implementations for critical operations
  - Efficient diameter computation algorithms

## 🚀 Installation

### Prerequisites
- Python 3.5+
- NumPy
- Cython (for compiling extensions)
- Matplotlib (for visualization)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/kd-tree-analysis.git
   cd kd-tree-analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy matplotlib cython
   ```

3. **Compile Cython extensions:**
   ```bash
   python setup.py build_ext --inplace
   ```

This will compile the Cython files:
- `gram_schmidt.pyx` - Fast Gram-Schmidt orthogonalization
- `compute_diameter.pyx` - Efficient diameter computation algorithms

## 🎯 Usage

### Basic Example

```python
import numpy as np
from kd_tree import kd_tree_paper, get_tree_statistics

# Generate sample data
data = np.random.rand(1000, 10)  # 1000 points in 10D space

# Build kd-tree
tree = kd_tree_paper(data, approx_diameter=0.25)

# Get tree statistics
distribution, stats = get_tree_statistics(tree)
print("Mean diameters by depth:", stats[0])
print("Standard deviations:", stats[1])
```

### Advanced Usage

```python
from kd_tree import kd_tree_axis, rp_tree

# Axis-aligned kd-tree
axis_tree = kd_tree_axis(data, approx_diameter=0.1, jitter=True)

# Random projection tree
rp_tree = rp_tree(data, approx_diameter=0.1, max_leaf_size=5)

# Compare statistics
axis_stats = get_tree_statistics(axis_tree)
rp_stats = get_tree_statistics(rp_tree)
```

### Visualization

The project includes visualization tools for analyzing tree performance:

```python
import matplotlib.pyplot as plt

# Example from main.py
data = smooth_1d_manifold(20000, 8)
tree = kd_tree_paper(data, approx_diameter=0.25)
distrib, stats = get_tree_statistics(tree)

plt.figure()
plt.plot(np.log(stats[0][:20]), marker="o")
plt.title("KD-Tree Diameter Reduction")
plt.xlabel("Tree Depth")
plt.ylabel("Log Diameter (normalized)")
plt.savefig("kd_tree_analysis.png")
```

## 📊 Data Generation Utilities

The project includes comprehensive data generation utilities for creating synthetic datasets:

```python
from data_generation import (
    gaussian_distribution,
    uniform_distribution,
    hollow_hypersphere,
    mechanical_arm,
    swiss_roll,
    smooth_1d_manifold
)

# Gaussian distribution in affine subspace
gauss_data = gaussian_distribution(n=1000, ambiant_dim=10, intrinsic_dim=3)

# Uniform distribution in affine subspace
uniform_data = uniform_distribution(n=1000, ambiant_dim=10, intrinsic_dim=3)

# Points on hypersphere surface
hypersphere_data = hollow_hypersphere(n=1000, d=5)  # 4D manifold

# Mechanical arm data
arm_data = mechanical_arm(n=1000, space_dim=8, number_of_joints=3)

# Swiss roll manifold
swiss_data = swiss_roll(n=1000)

# Smooth 1D manifold
smooth_data = smooth_1d_manifold(n=1000, ambiant_dim=8)
```

## 🧪 Experiments

The project includes comprehensive experiments for analyzing kd-tree performance on various datasets:

### Main Experiments

1. **Smooth 1D Manifold:** Analyze diameter reduction on smooth manifolds with different ambient dimensions
2. **Gaussian Distribution:** Compare performance on normally distributed points in affine subspaces
3. **Uniform Distribution:** Test with uniformly distributed points in affine subspaces
4. **Hollow Hypersphere:** Compare with points on hypersphere surfaces
5. **Mechanical Arm:** Analyze performance on mechanical arm configurations
6. **Swiss Roll:** Test with Swiss roll manifold data

### Tree Variants Comparison

- **Random kd-tree vs RP-tree:** Compare random projection trees with standard kd-trees
- **Random kd-tree vs axis-aligned tree:** Compare axis-aligned partitioning with random partitioning

### Running Experiments

```bash
python main.py
```

This will generate all experiments and save the results as PNG files in the `outputs/` directory.

## 🚀 Performance

### Time Complexity
- **Tree Construction:** O(n log n) average case
- **Diameter Computation:** O(n²) for exact, O(n) for approximate
- **Statistics Collection:** O(n) tree traversal

### Space Complexity
- **Tree Storage:** O(n) for node storage
- **Statistics:** O(d) where d is maximum tree depth

### Benchmark Results

The project includes performance tests in `test_performance.py`. Key optimizations:
- Cython implementations for critical numerical operations
- Approximate diameter computation reduces complexity significantly
- Efficient memory usage through tuple-based tree structure

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit them (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest
```

## 📚 API Reference

### Main Functions

#### `kd_tree_paper(data, jitter=True, approx_diameter=False, max_leaf_size=1)`
- Builds a kd-tree using the paper's method with Gram-Schmidt orthogonalization
- **Parameters:**
  - `data`: numpy array of shape (n_samples, n_dimensions)
  - `jitter`: Add small random perturbations to splits
  - `approx_diameter`: Use approximate diameter computation
  - `max_leaf_size`: Maximum points per leaf node

#### `kd_tree_axis(data, i=0, jitter=True, approx_diameter=False, max_leaf_size=1)`
- Builds a standard axis-aligned kd-tree
- **Parameters:** Same as `kd_tree_paper` plus starting axis `i`

#### `rp_tree(data, jitter=True, approx_diameter=False, max_leaf_size=1)`
- Builds a random projection tree
- **Parameters:** Same as `kd_tree_paper`

### Utility Functions

#### `get_tree_statistics(tree)`
- Computes statistics about tree node diameters
- **Returns:** (distribution, [means, stds, mins, maxs])

#### `get_branch_statistics(tree, distribution, depth=1)`
- Recursively collects diameter information (internal use)

#### `get_count_tree(tree)`
- Counts nodes in the tree
- **Returns:** (total_count, left_subtree, right_subtree)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

## 🎓 References

- Verma, V., et al. (2009). "Fast dimension reduction for manifold data."
- Original kd-tree algorithm by Jon Louis Bentley (1975)

## 🔮 Future Work

- Implement k-nearest neighbor search
- Add parallel tree construction
- Extend to dynamic kd-trees
- Add more visualization tools

---

Made with ❤️ by the KD-Tree Analysis Team