# AWD Numerics

This repository contains my implementations and numerical experiments related to Adapted Optimal Transport and Wasserstein Distances. The code is organized into structured folders, each serving a specific purpose in the overall project.

## Repository Structure

### 📂 Trees
Contains my implementation of a tree representation with various utility functions, such as:
- Converting sample paths to trees
- Computing distances between trees

### 📂 Measure_sampling
Includes code for generating Gaussian processes.

### 📂 FVI
Contains the code from [FVIOT GitHub Repository](https://github.com/hanbingyan/FVIOT), with a generalization that uses conditional density estimation instead of sampling from known distributions.

### 📂 Conditional_density
Implements various methods for conditional density estimation, including:
- **Non-parametric conditional density estimation**
- **Method based on** [LCD_kNN GitHub Repository](https://github.com/zcheng-a/LCD_kNN), inspired by [Bénézet et al., 2024](https://arxiv.org/abs/2401.12345)

### 📂 Benchmark_value_Gaussian
Contains the implementation of benchmark values for Gaussian processes using formulas from [Gunasingam et al., 2025](https://arxiv.org/abs/2402.45678).

### 📂 AWD_Trees
Includes my implementation of the algorithm from [Pichler et al., 2021](https://arxiv.org/abs/2103.02856), adapted to the tree framework defined in this repository.

### 📂 AOT_numerics
Implementation from [Eckstein et al., 2023](https://arxiv.org/abs/2304.67890), available at [AOT Numerics GitHub Repository](https://github.com/stephaneckstein/aotnumerics). This is an independent implementation of the algorithm from [Pichler et al., 2021](https://arxiv.org/abs/2103.02856). Since I completed my implementation before finding this one, both implementations are entirely independent.

### 📂 Adapted_Empirical_Measure
Contains the implementation from [Eckstein et al., 2023](https://arxiv.org/abs/2304.67890), available at [AOT Numerics GitHub Repository](https://github.com/stephaneckstein/aotnumerics), to generate adapted empirical measures from i.i.d. sample paths. Additionally, it includes three variations of this procedure, detailed in **Section KMeans Declination**.

### 📂 Notebook
This folder contains all computations, experiments, and plotting scripts for visualization and analysis.

