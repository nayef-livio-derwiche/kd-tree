# kd-tree Implementation for Dimensionality Analysis

This repository contains a specialized kd-tree implementation designed for analyzing dimensionality reduction in various datasets. The implementation focuses on studying how the diameter of data points changes as we traverse deeper into the kd-tree structure.

## Overview

This project implements different kd-tree variants and uses them to analyze how the diameter of data points decreases with tree depth across various datasets. The main focus is on understanding the intrinsic dimensionality of different data distributions.

## Key Features

- **Multiple kd-tree implementations**:
  - Standard axis-aligned kd-tree
  - Random projection kd-tree (RP-tree)
  - Paper-specific kd-tree implementation

- **Comprehensive experiments** with various data distributions:
  - Smooth 1D manifolds
  - Gaussian distributions in affine subspaces
  - Uniform distributions in affine subspaces
  - Hollow hyperspheres
  - Mechanical arm simulations
  - Swiss roll manifolds

- **Performance analysis** tools:
  - Diameter computation (exact and approximate)
  - Tree statistics collection
  - Local covariance dimension estimation

- **Visualization** of experimental results

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/nayef-livio-derwiche/kd-tree.git
   cd kd-tree
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Compile Cython extensions:
   ```bash
   python setup.py build_ext --inplace
   ```

## Usage

### Running Experiments

To run all experiments and generate plots:
```bash
python main.py
```

This will:
1. Generate various datasets
2. Build kd-trees for each dataset
3. Compute statistics about diameter reduction
4. Create and save plots in the `outputs/` directory

### Key Files

- `kd_tree.py`: Contains kd-tree implementations and utility functions
- `main.py`: Runs all experiments and generates plots
- `data_generation.py`: Functions for generating various datasets
- `compute_diameter.pyx`: Cython module for diameter computation
- `gram_schmidt.pyx`: Cython module for Gram-Schmidt orthogonalization
- `setup.py`: Compilation script for Cython extensions
- `test_performance.py`: Performance testing and debugging scripts

## Implementation Details

### kd-tree Variants

1. **Standard kd-tree** (`kd_tree_axis`): Traditional axis-aligned implementation
2. **Random projection tree** (`rp_tree`): Uses random projections for splitting
3. **Paper-specific implementation** (`kd_tree_paper`): Specialized version for the analysis

### Key Parameters

- `approx_diameter`: Boolean or float for approximate diameter calculation
- `jitter`: Adds small random noise to splits
- `max_leaf_size`: Controls when to stop splitting

## Experiments

The project includes several experiments comparing:
- Different data distributions with same intrinsic dimension
- Various ambient dimensions
- Different tree construction methods
- Diameter reduction rates

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit them (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License.

## References

This implementation is based on the analysis presented in the paper about dimensionality reduction and kd-tree behavior. The experiments demonstrate how different tree construction methods affect the diameter reduction rate for various data distributions.

For more details, see the included report: [report_gmda-kd-trees.pdf](report_gmda-kd-trees.pdf)