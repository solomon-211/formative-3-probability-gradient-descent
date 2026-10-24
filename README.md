# Formative 3 — Probability Distributions, Bayesian Probability & Gradient Descent

---

## Part 4 — Gradient Descent in Code

> **This is the part I was responsible for.**

### Overview

`gradient_descent.ipynb` converts the manual calculations from Part 3 into clean,
transparent Python code. Every intermediate value (ŷ, e, J, gradients, updated
parameters) is printed at each iteration so the computation mirrors the hand-worked
steps exactly.

### Key Design Decisions

- **Matrix operations throughout** — NumPy arrays are used for all calculations; no
  scalar shortcuts.
- **Modular building blocks** — each function does exactly one job and maps directly to
  one step of the hand calculation:

| Function | Responsibility |
|----------|---------------|
| `predict(X, m, b)` | Compute ŷ = Xm + b |
| `compute_error(y_hat, y)` | Compute e = ŷ − y |
| `compute_cost(e)` | Compute J = (1/n) eᵀe |
| `compute_gradients(X, e)` | Compute ∂J/∂m and ∂J/∂b |
| `update_params(m, b, dJ_dm, dJ_db, alpha)` | Apply gradient descent step |
| `compute_derivative(equation, point)` | Numerical gradient via SciPy |

- **SciPy verification** — `scipy.optimize.approx_fprime` is used to numerically
  differentiate the cost function and cross-check the hand-derived gradients at every
  iteration. An assertion confirms they match to within 1e-4.

### SciPy Integration

```python
from scipy.optimize import approx_fprime

def compute_derivative(equation, point):
    """Numerical derivative (gradient) of `equation` at `point`, using SciPy."""
    return approx_fprime(point, equation, epsilon=1e-7)
```

The cost function is wrapped so that SciPy can differentiate it with respect to a single
flat parameter vector `[m1, m2, b1, b2]`, making the verification straightforward.

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

**1. Parameter change over iterations** (`m_and_b_plot.png`)  
Shows how m₁, m₂, b₁, and b₂ evolve across the three update steps.

**2. Error (cost) over iterations** (`error_plot.png`)  
Shows the MSE cost J decreasing from 61 → 6.50 → 2.50 → 2.16, demonstrating
convergence.

### Observed Trend

- m₂ falls from 2 → ~1.10; m₁ moves from −1 → ~−1.37.
- b changes only slightly because its gradient (e) is much smaller than the gradient
  for m (Xᵀe).
- The first update overshoots (predictions go from too high to too low), so the error
  signs flip in Update 2. Later updates are smaller.
- The cost consistently decreases, confirming the parameters are moving in the correct
  direction.

---

## How to Run

1. Clone the repository and open `gradient_descent.ipynb` in Jupyter or Google Colab.
2. Run all cells in order — each cell corresponds to one labelled section.
3. The notebook will print step-by-step results and display/save both plots automatically.

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
