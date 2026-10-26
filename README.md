# Formative 3 — Probability Distributions, Bayesian Probability & Gradient Descent

---

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
steps exactly.

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

### Key Design Decisions

- **Matrix operations throughout** — NumPy arrays are used for all calculations; no scalar shortcuts.
- **Modular building blocks** — each function does exactly one job and maps directly to one step of the hand calculation:

| Function | Responsibility |
|----------|---------------|
| `predict(X, m, b)` | Compute ŷ = Xm + b |
| `compute_error(y_hat, y)` | Compute e = ŷ − y |
| `compute_cost(e)` | Compute J = (1/n) eᵀe |
| `compute_gradients(X, e)` | Compute ∂J/∂m and ∂J/∂b |
| `update_params(m, b, dJ_dm, dJ_db, alpha)` | Apply gradient descent step |
| `compute_derivative(equation, point)` | Numerical gradient via SciPy |

- **SciPy verification** — `scipy.optimize.approx_fprime` numerically differentiates the cost function and cross-checks the hand-derived gradients at every iteration. An assertion confirms they match to within 1e-4.

### SciPy Integration

```python
from scipy.optimize import approx_fprime

def compute_derivative(equation, point):
    """Numerical derivative (gradient) of `equation` at `point`, using SciPy."""
    return approx_fprime(point, equation, epsilon=1e-7)
```

The cost function is wrapped so SciPy can differentiate it with respect to a single flat parameter vector `[m1, m2, b1, b2]`.

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

**1. Parameter change over iterations** (`m_and_b_plot.png`) — shows how m₁, m₂, b₁, and b₂ evolve across the three update steps.

**2. Error (cost) over iterations** (`error_plot.png`) — shows the MSE cost J decreasing from 61 → 6.50 → 2.50 → 2.16, demonstrating convergence.

### Observed Trend

- m₂ falls from 2 → ~1.10; m₁ moves from −1 → ~−1.37.
- b changes only slightly because its gradient (e) is much smaller than the gradient for m (Xᵀe).
- The first update overshoots (predictions go from too high to too low), so the error signs flip in Update 2. Later updates are smaller.
- The cost consistently decreases, confirming the parameters are moving in the correct direction.

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
