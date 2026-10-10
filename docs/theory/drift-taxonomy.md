# Drift Taxonomy in Data Streams

In streaming machine learning, non-stationarity alters the joint probability distribution $P(X, Y)$ over time $t$. Understanding the formal taxonomy of drift is essential for selecting appropriate detection and explanation mechanisms.

---

## Formal Decomposition

The joint probability of features $X \in \mathbb{R}^D$ and labels $Y \in \{0, 1\}$ decomposes as:

$$P_t(X, Y) = P_t(X) \cdot P_t(Y \mid X)$$

or symmetrically:

$$P_t(X, Y) = P_t(Y) \cdot P_t(X \mid Y)$$

---

## Types of Drift

### 1. Covariate Shift (Virtual Drift / Data Drift)
Occurs when the marginal input distribution $P(X)$ changes while the conditional posterior $P(Y \mid X)$ remains unchanged:

$$P_{t_1}(X) \neq P_{t_2}(X) \quad \text{and} \quad P_{t_1}(Y \mid X) = P_{t_2}(Y \mid X)$$

*Implication*: The optimal decision boundary remains stationary, though existing models may experience elevated error if test instances shift into sparse regions of the feature space.

### 2. Real Concept Drift
Occurs when the conditional posterior distribution $P(Y \mid X)$ changes, regardless of whether $P(X)$ changes:

$$P_{t_1}(Y \mid X) \neq P_{t_2}(Y \mid X)$$

*Implication*: The true decision boundary alters. A model trained on $t_1$ is guaranteed to suffer performance degradation unless updated.

### 3. Prior Probability Shift
Occurs when the class balance $P(Y)$ alters:

$$P_{t_1}(Y) \neq P_{t_2}(Y)$$

---

## Drift Speed and Transition Forms

- **Sudden (Abrupt) Drift**: Immediate replacement of concept $S_1$ by $S_2$ at sample $t_0$.
- **Gradual Drift**: A transition window $W$ where instances are sampled from both concepts via a time-dependent mixing probability:
  $$P(\text{sample from } S_2) = \frac{1}{1 + e^{-s(t - t_0)}}$$
- **Incremental Drift**: Continuous, small-step migration across intermediate concepts.
- **Recurring Drift**: A historical concept re-emerges after a dormant period.
