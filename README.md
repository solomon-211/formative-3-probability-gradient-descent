# Formative 3 — Probability Distributions, Bayesian Probability & Gradient Descent

---

# Part 1: Expectation-Maximization (EM) Algorithm

## Overview

This part implements the Expectation-Maximization (EM) algorithm from scratch to fit a Gaussian Mixture Model (GMM) with two components to a dataset of children's and fathers' heights.

The objective is to identify two height distributions from an unlabelled collection of observations. The algorithm estimates the mean, variance, and mixing weight of each component and updates these parameters iteratively until the model converges.

## Dataset

**Dataset:** [Parents' Heights vs. Children's Heights — Galton Data](https://www.kaggle.com/datasets/jacopoferretti/parents-heights-vs-children-heights-galton-data)

The dataset contains family height measurements, including children's and fathers' heights.

For this implementation:
- Children's heights are extracted from the dataset.
- One father's height per family is selected to avoid counting the same father multiple times.
- Both sets of heights are combined into one unlabelled dataset.
- The resulting dataset contains 1,139 observations: 934 children's heights and 205 fathers' heights.

The dataset is not included in this repository. Download it from Kaggle before running the notebook.

## Downloading and Setting Up the Dataset

1. Open the [Kaggle dataset page](https://www.kaggle.com/datasets/jacopoferretti/parents-heights-vs-children-heights-galton-data).
2. Sign in to Kaggle if prompted.
3. Download the dataset as a ZIP file.
4. Extract the ZIP file and locate the CSV containing the Galton family data.
5. Ensure the CSV filename and file path match the path specified in the notebook. If necessary, rename the file or update the code.

If you are using Google Colab, upload the CSV using the Files panel and make sure the notebook reads from the correct location.

## Methodology

The EM algorithm estimates the parameters of the two Gaussian components through the following steps:

1. **Initialization:** Set initial estimates for the component means, variances, and mixing weights.
2. **Expectation step (E-step):** Calculate the probability that each height belongs to each component.
3. **Maximization step (M-step):** Update the means, variances, and mixing weights using the probabilities calculated in the E-step.
4. **Convergence check:** Calculate the log-likelihood and repeat the E-step and M-step until the change in log-likelihood is smaller than the chosen tolerance or the maximum number of iterations is reached.

The implementation uses a convergence tolerance of `1e-6` and a maximum of `1,000` iterations.

## Results

The following table records the initial parameter estimates and the results after the first two EM iterations.

| Iteration | Mean 1 | Mean 2 | Variance 1 | Variance 2 | Weight 1 | Weight 2 | Log-Likelihood |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (Initial) | 64.2299 | 70.1819 | 12.7241 | 12.7241 | 0.5000 | 0.5000 | -3136.4676 |
| 1 | 65.1207 | 69.2760 | 8.7032 | 8.1149 | 0.4976 | 0.5024 | -3056.6103 |
| 2 | 65.0591 | 69.3312 | 8.5547 | 7.7733 | 0.4969 | 0.5031 | -3055.5576 |

These results show how the component parameters change during the first iterations. The log-likelihood increases, indicating that the model fit improves as the algorithm updates its estimates.

### Example Prediction

A height of 68 inches was used to demonstrate the component-membership probabilities calculated by the model.

| Component | Posterior Probability |
|---|---:|
| Component 1 (shorter-height distribution) | 19.88% |
| Component 2 (taller-height distribution) | 80.12% |

For this observation, the model assigns a higher probability to the taller-height component.

**Note:** The two components represent statistical height distributions. They do not automatically identify whether a person is a child or a parent.

## Implementation

The notebook implements the following functionality:
- Gaussian probability density calculations.
- E-step responsibility calculations.
- M-step parameter updates.
- Log-likelihood calculation.
- Iterative convergence checking.
- Posterior probability calculations for a new height.

The algorithm is implemented using Python, NumPy, and Pandas.

## How to Run

1. Download and extract the dataset from Kaggle.
2. Place the CSV where the notebook expects it, or update the file path.
3. Open the Part 1 Jupyter notebook.
4. Run all cells in order.
5. Review the parameter-tracking table, convergence output, and example posterior probabilities.

## Conclusion

The implementation demonstrates how the EM algorithm can estimate a two-component Gaussian mixture from unlabelled height data. It shows the iterative process of updating model parameters, monitoring log-likelihood, and estimating the probability that an observation belongs to each component.

## Part 2: Bayesian Probability
### Overview

Part 2 applies Bayes' theorem to movie reviews from the IMDb Dataset of 50K Movie Reviews. The objective is to investigate the relationship between selected keywords and negative sentiment by calculating prior, likelihood, marginal, and posterior probabilities.

The analysis was completed in Python using the built-in `csv` module. Each review was checked for the presence of four selected keywords: `horrible`, `disappointing`, `pathetic`, and `boring`. Each keyword was counted at most once per review, regardless of how many times it appeared.

### Dataset
- **Dataset:** IMDb Dataset of 50K Movie Reviews
- **Source:** [Kaggle: IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **File required:** `IMDB Dataset.csv`
- **Total reviews:** 50,000
- **Positive reviews:** 25,000
- **Negative reviews:** 25,000
- **Columns:** `review` and `sentiment`

The dataset is not included in this repository. Download it separately before running the analysis.

### Download and Setup Instructions
1. Open the [IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
2. Sign in to your Kaggle account, creating a free account if necessary.
3. Click **Download** to download the dataset archive.
4. Extract the downloaded ZIP file. It contains `IMDB Dataset.csv`.
5. Make the CSV file available in the environment where you run the analysis:
   - **Google Colab:** Open the notebook, click the Files icon in the left sidebar, and upload `IMDB Dataset.csv`.
6. Run the notebook cells in order.

The CSV filename must match `IMDB Dataset.csv` because the code opens that exact filename.

### Results

The following table summarizes the probabilities calculated from the dataset.

| Keyword | Prior: P(Negative) | Likelihood: P(Keyword\|Negative) | Marginal: P(Keyword) | Posterior: P(Negative\|Keyword) |
|---|---:|---:|---:|---:|
| horrible | 0.5000 (50%)| 0.07116 (7.15%)| 0.04210 (4.21%)| 0.8451 (84.51%) |
| disappointing | 0.5000 (50%)| 0.02556 (2.56%)| 0.01646 (1.65%)| 0.7764 (77.64%) |
| pathetic | 0.5000 (50%)| 0.04424 (4.42%)| 0.03128 (3.13%)| 0.7072 (70.72%) |
| boring | 0.5000 (50%)| 0.09992 (9.99%)| 0.06232 (6.23%)| 0.8017 (80.17%) |

### Interpretation

- **Horrible:** Reviews containing this keyword had an estimated 84.51% probability of being labelled negative, the highest posterior probability among the four keywords.
- **Disappointing:** Reviews containing this keyword had a 77.64% probability of being labelled negative.
- **Pathetic:** Reviews containing this keyword had a 70.72% probability of being labelled negative, the lowest posterior probability among the four keywords.
- **Boring:** Reviews containing this keyword had an 80.17% probability of being labelled negative.

### Limitations

This analysis uses keyword presence as a simple indicator of sentiment. It does not understand the context in which a word is used. For example, a positive review might say that a movie "wasn't horrible," even though it contains the keyword `horrible`.

The method also does not account for negation, sarcasm, or the overall meaning of a review. Therefore, these probabilities describe the relationship between keyword presence and the dataset's sentiment labels; they do not guarantee that an individual review has been interpreted correctly.

### Conclusion

Bayes' theorem was used to estimate the probability of negative sentiment given the presence of four selected keywords. Among them, `horrible` had the highest posterior probability at 84.51%, while `pathetic` had the lowest at 70.72%.

---

## Part 4 — Gradient Descent in Code

### Overview

`gradient_descent.ipynb` converts the manual calculations from Part 3 into clean,
transparent Python code. Every intermediate value (ŷ, e, J, gradients, updated
parameters) is printed at each iteration so the computation mirrors the hand-worked
steps exactly. The code is modular and follows the DRY principle — each function does
exactly one job and is reused across every iteration.

### Problem Setup

Linear model: **ŷ = Xm + b** (matrix form)

| Parameter | Initial value |
|-----------|--------------|
| m | [-1, 2]ᵀ |
| b | [1, 1]ᵀ |
| X | [[1, 3], [4, 10]] |
| y | [5, 6]ᵀ |
| Learning rate α | 0.01 |
| Number of updates | 3 |

### Gradient Derivation (Chain Rule)

Cost function (MSE):

```
J(m, b) = (1/n) · eᵀe     where e = ŷ − y = Xm + b − y
```

Applying the chain rule J → e → ŷ → (m, b):

```
∂J/∂ŷ = (2/n) · e
∂ŷ/∂m = Xᵀ
∂ŷ/∂b = I

∴ ∂J/∂m = (2/n) · Xᵀe
∴ ∂J/∂b = (2/n) · e
```

Update rules:

```
m ← m − α · ∂J/∂m
b ← b − α · ∂J/∂b
```

### Code Structure (Modular / DRY)

Each function maps directly to one step of the derivation above and is reused in every iteration:

| Function | Responsibility |
|----------|---------------|
| `predict(X, m, b)` | Compute ŷ = Xm + b |
| `compute_error(y_hat, y)` | Compute e = ŷ − y |
| `compute_cost(e)` | Compute J = (1/n) eᵀe |
| `compute_gradients(X, e)` | Compute ∂J/∂m = (2/n)Xᵀe and ∂J/∂b = (2/n)e |
| `update_params(m, b, dJ_dm, dJ_db, alpha)` | Apply m ← m − α·∂J/∂m and b ← b − α·∂J/∂b |
| `compute_derivative(equation, point)` | Numerical gradient via SciPy (used to verify hand-derived gradients) |

### SciPy Integration

`compute_derivative` accepts any equation as a function and a point, and returns the
numerical gradient using `scipy.optimize.approx_fprime`. It is used to cross-check the
hand-derived gradients at every iteration — an assertion confirms they match to within 1e-4.

```python
from scipy.optimize import approx_fprime

def compute_derivative(equation, point):
    """Numerical derivative (gradient) of `equation` at `point`, using SciPy."""
    return approx_fprime(point, equation, epsilon=1e-7)
```

The cost function is wrapped so SciPy can differentiate it with respect to a single flat
parameter vector `[m1, m2, b1, b2]`, making the verification straightforward.

### Printed Output (3 iterations)

```
===== Update 1 =====
y_hat = [ 6. 17.]
e     = [ 1. 11.]
J     = 61.0
dJ/dm = [ 45. 113.]
dJ/db = [ 1. 11.]
SciPy gradient [dm1, dm2, db1, db2] = [ 45. 113.   1.  11.]
new m = [-1.45  0.87]
new b = [0.99 0.89]

===== Update 2 =====
y_hat = [2.15 3.79]
e     = [-2.85 -2.21]
J     = 6.5033
dJ/dm = [-11.69 -30.65]
dJ/db = [-2.85 -2.21]
SciPy gradient [dm1, dm2, db1, db2] = [-11.69 -30.65  -2.85  -2.21]
new m = [-1.3331  1.1765]
new b = [1.0185 0.9121]

===== Update 3 =====
y_hat = [3.2149 7.3447]
e     = [-1.7851  1.3447]
J     = 2.4974
dJ/dm = [3.5937 8.0917]
dJ/db = [-1.7851  1.3447]
SciPy gradient [dm1, dm2, db1, db2] = [ 3.5937  8.0917 -1.7851  1.3447]
new m = [-1.369037  1.095583]
new b = [1.036351 0.898653]
```

### Final Predictions

```
final m      = [-1.369037  1.095583]
final b      = [1.036351 0.898653]
predictions  = [2.954063 6.378335]
actual y     = [5. 6.]
final cost J = 2.1645
```

### Plots

Two Matplotlib figures are produced and saved:

**1. Parameter change over iterations** (`m_and_b_plot.png`) — shows how m₁, m₂, b₁,
and b₂ evolve across the three update steps.

**2. Error (cost) over iterations** (`error_plot.png`) — shows the MSE cost J decreasing
from 61 → 6.50 → 2.50 → 2.16, demonstrating convergence.

### Observed Trend — Are m and b Moving in the Right Direction?

Yes. The cost drops from **61.0 → 6.50 → 2.50 → 2.16** across the three updates,
confirming that the parameters are consistently moving in a direction that reduces the
error.

- **m₂** falls from 2 → ~1.10 and **m₁** moves from −1 → ~−1.37. The first update is
  the largest because the initial error (J = 61) was large, producing large gradients.
- **b** changes only slightly because its gradient (e) is much smaller in magnitude than
  the gradient for m (Xᵀe), so the step size is small.
- The first update overshoots slightly — predictions jump from too high to too low —
  causing the error signs to flip in Update 2. This is expected behaviour at a learning
  rate of 0.01 with a large initial error. Subsequent updates are smaller and the cost
  continues to decrease, showing the algorithm is converging.

---

## How to Run

1. Open `gradient_descent.ipynb` in Jupyter or Google Colab.
2. Run all cells in order — each cell corresponds to one labelled section.
3. The notebook prints step-by-step results and displays/saves both plots automatically.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Matrix operations |
| `scipy` | Numerical differentiation (`approx_fprime`) |
| `matplotlib` | Plotting parameter and error histories |

Install with:

```bash
pip install numpy scipy matplotlib
```
