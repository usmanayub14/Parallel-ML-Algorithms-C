# Parallel Machine Learning Algorithms in C/C++

This repository showcases my work in optimizing fundamental machine learning algorithms through parallel computing techniques in C/C++. The focus is on parallelizing computationally intensive steps like cost function calculation and gradient descent using POSIX Threads (Pthreads).

## Key Learning Outcomes & Skills:

* **Machine Learning Fundamentals:** `Linear Regression`, `Logistic Regression`, `Hypothesis Function`, `Cost Function`, `Gradient Descent Algorithm`.
* **Parallel Computing:** `Pthreads` for multi-threaded parallelization.
* **Workload Distribution:** Partitioning datasets among threads for parallel computation of sums (e.g., for gradients and costs).
* **Synchronization:** Using `mutexes` for thread-safe aggregation of partial results.
* **Data Handling:** Loading and parsing structured datasets (CSV).
* **Performance Optimization:** Demonstrating speedup and efficiency gains from parallelization.

## Projects:

### 1. Parallel Linear Regression with Gradient Descent
* **Description:** Implements a linear regression model in C++ and parallelizes its training process (cost function and gradient descent updates) using Pthreads.
* **Skills Showcased:** `Linear regression model development`, `multi-threaded cost calculation`, `parallel gradient computation`, `thread-safe weight updates`, `CSV data loading`.
* **Source:** `PDP Lab 06 - Linear Regression.pdf` (Lab Tasks)
* **Details:** (Your implementation of linear regression, showing parallelized cost and gradient calculations).
    * [cite_start]**Hypothesis:** $h(x) = b + w_1 x_1 + w_2 x_2 + w_3 x_3$[cite: 1394].
    * [cite_start]**Cost Function:** $J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)})^2$[cite: 1405].
    * [cite_start]**Gradient Descent:** Formulas for $\frac{\partial J}{\partial w_j}$ and $\frac{\partial J}{\partial b}$, and update rules $w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$, $b := b - \alpha \frac{\partial J}{\partial b}$[cite: 1412].
    * **Parallelization Strategy:** Each thread would process a subset of training examples to compute partial sums for cost and gradients, which are then aggregated.
* **Results:** Performance metrics (execution time, speedup) comparing sequential and parallel training, possibly with plots of cost vs. epochs.

### 2. Parallel Logistic Regression with Gradient Descent
* **Description:** Implements a logistic regression classifier in C++ with parallelized training (cost function and gradient calculations) using Pthreads.
* **Skills Showcased:** `Logistic regression model (sigmoid activation)`, `parallel cost calculation (log loss)`, `parallel gradient descent for classification`, `handling numerical stability (log(0) avoidance)`, `dataset partitioning for threads`.
* **Source:** `CS435 Lab 07 Manual - Parallel Implementation of Logistic Regression.pdf` (Lab Tasks)
* **Details:**
    * [cite_start]`sigmoid.cpp`: Implementation of the sigmoid function $g(z)=1/(1+e^{-z})$[cite: 5958].
    * `logistic_regression_parallel.cpp`: Main training loop.
    * [cite_start]**Hypothesis:** $h(x) = g(b + w_1 X_1 + w_2 X_2 + w_3 X_3)$[cite: 5981].
    * [cite_start]**Cost Function:** $J(w) = \frac{1}{m} \sum_{i=0}^{m-1} (-y^{(i)} \log(h(x^{(i)})) - (1-y^{(i)}) \log(1-h(x^{(i)})))$[cite: 5982].
    * [cite_start]**Gradient Descent:** Formulas for $dw_j$, $db$ and update rules[cite: 6106, 6108].
    * **Parallelization Strategy:** Threads compute partial sums for cost and gradients over their assigned data chunks, using a `mutex` for thread-safe aggregation.
* **Results:**
    * Screenshots demonstrating correctness (e.g., consistent cost for different thread counts).
    * [cite_start]Plots of "Training Loss Over Epochs" and "Time vs. Number of Threads for Training"[cite: 6904, 6912], illustrating how parallelization affects training time and convergence.
    * Initial and final weights, showing the model's learning.
