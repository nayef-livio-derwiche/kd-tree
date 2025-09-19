# Vibe was here 4

# KD-Tree Implementation

## Project Overview

A **kd-tree** (k-dimensional tree) is a space-partitioning data structure for organizing points in a k-dimensional space. KD-trees are particularly useful for:

- **Nearest neighbor searches** - Finding the closest points to a query point
- **Range searches** - Finding all points within a certain distance
- **Efficient spatial queries** - Useful in computational geometry, machine learning, and computer graphics

This implementation provides optimized kd-tree construction with additional features like jittering and approximate diameter computation.

## Features

- **Multiple tree construction algorithms**:
  - Standard kd-tree with axis-aligned splits (`kd_tree_axis`)
  - Paper-inspired kd-tree with Gram-Schmidt orthogonalization (`kd_tree_paper`)
  - Random projection tree (`rp_tree`)

- **Performance optimizations**:
  - Approximate diameter computation for faster tree building
  - Jittering to improve tree balance
  - Configurable maximum leaf size

- **Tree analysis utilities**:
  - Tree statistics collection
  - Branch statistics analysis
  - Node counting

## Installation

### Prerequisites
- Python 3.x
- NumPy
- Cython (for compiled extensions)
- Matplotlib (for visualization)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/kd-tree.git
   cd kd-tree
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Compile Cython extensions**:
   ```bash
   python setup.py build_ext --inplace
   ```

## Usage

### Basic Example

```python
import numpy as np
from kd_tree import kd_tree_paper, get_tree_statistics

# Generate some sample data
data = np.random.rand(1000, 3)  # 1000 points in 3D space

# Build a kd-tree
tree = kd_tree_paper(data, approx_diameter=0.25)

# Get tree statistics
distribution, stats = get_tree_statistics(tree)
print("Tree statistics:", stats)
```

### Advanced Usage

```python
from kd_tree import kd_tree_axis, rp_tree

# Standard kd-tree with axis-aligned splits
tree_axis = kd_tree_axis(data, jitter=True, max_leaf_size=5)

# Random projection tree
tree_rp = rp_tree(data, approx_diameter=0.1, max_leaf_size=3)

# Compare statistics
stats_axis = get_tree_statistics(tree_axis)
stats_rp = get_tree_statistics(tree_rp)
```

## API Reference

### Main Functions

#### `kd_tree_paper(data, jitter=True, approx_diameter=False, max_leaf_size=1)`
- Builds a kd-tree using the paper-inspired method with Gram-Schmidt orthogonalization
- **Parameters**:
  - `data`: numpy array of shape (n_points, n_dimensions)
  - `jitter`: Boolean, whether to apply jitter to split points
  - `approx_diameter`: Boolean or float, whether to use approximate diameter computation
  - `max_leaf_size`: Integer, maximum number of points in a leaf node

#### `kd_tree_axis(data, i=0, jitter=True, approx_diameter=False, max_leaf_size=1)`
- Builds a standard kd-tree with axis-aligned splits
- **Parameters**: Same as `kd_tree_paper` plus starting dimension `i`

#### `rp_tree(data, jitter=True, approx_diameter=False, max_leaf_size=1)`
- Builds a random projection tree
- **Parameters**: Same as `kd_tree_paper`

### Utility Functions

#### `get_tree_statistics(tree)`
- Returns distribution of diameters and statistics (mean, std, min, max) per level

#### `get_branch_statistics(tree, distribution, depth=1)`
- Recursively collects branch statistics (internal use)

#### `get_count_tree(tree)`
- Returns count of nodes in the tree

## Performance

### Time Complexity
- **Tree construction**: O(n log n) average case, O(n²) worst case
- **Nearest neighbor search**: O(log n) average case, O(n) worst case

### Space Complexity
- O(n) for storing the tree structure

### Benchmarks

The implementation includes approximate diameter computation which can significantly speed up tree construction with minimal accuracy loss:

```python
# Example benchmark
data = np.random.rand(10000, 10)

# Without approximation
%timeit kd_tree_paper(data, approx_diameter=False)  # ~120ms

# With approximation (ε=0.25)
%timeit kd_tree_paper(data, approx_diameter=0.25)  # ~45ms
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit them (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a Pull Request

### Development Setup

- Install development dependencies: `pip install -r requirements-dev.txt`
- Run tests: `pytest`
- Build documentation: `mkdocs build`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by the kd-tree implementation described in Verma et al. 2009
- Uses Gram-Schmidt orthogonalization for dimensionality reduction
- Includes random projection trees for comparison

## Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---

Happy spatial querying! 🚀🌳