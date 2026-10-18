# Mathematical Formulations in STRIDE

This document provides formal mathematical definitions, objective functions, statistical formulations, and algorithmic mechanics for the Explainable AI (xAI) algorithms implemented in STRIDE.

---

## 1. Distribution Divergences & Hypothesis Tests

### 1.1 First Wasserstein Metric ($\mathcal{W}_1$)

Given two univariate probability measures $u$ and $v$ defined over $\mathbb{R}$ with cumulative distribution functions $U(x) = P(X \le x)$ and $V(x) = P(Y \le x)$, the **1D First Wasserstein Distance** (Earth Mover's Distance) is defined as:

$$\mathcal{W}_1(u, v) = \int_{-\infty}^{+\infty} \left| U(x) - V(x) \right| \, dx$$

By the **Kantorovich-Rubinstein duality**, $\mathcal{W}_1$ admits the equivalent dual representation:

$$\mathcal{W}_1(u, v) = \sup_{f \in \text{Lip}_1} \left( \mathbb{E}_{X \sim u}[f(X)] - \mathbb{E}_{Y \sim v}[f(Y)] \right)$$

where $\text{Lip}_1$ represents the set of all 1-Lipschitz continuous functions:

$$\text{Lip}_1 = \left\{ f : \mathbb{R} \to \mathbb{R} \;\Big|\; |f(x) - f(y)| \le |x - y|, \; \forall x, y \in \mathbb{R} \right\}$$

Unlike the Kullback-Leibler (KL) divergence or Jensen-Shannon divergence, $\mathcal{W}_1$ satisfies all metric properties (non-negativity, identity of indiscernibles, symmetry, triangle inequality) and remains well-defined even when the supports of $u$ and $v$ are disjoint ($\text{supp}(u) \cap \text{supp}(v) = \emptyset$).

---

### 1.2 Two-Sample Kolmogorov-Smirnov Test

Given two independent continuous empirical samples $X_{\text{ref}} = \{x_1, \dots, x_n\}$ and $X_{\text{det}} = \{x'_1, \dots, x'_m\}$, the two-sample **Kolmogorov-Smirnov statistic** $D_{n, m}$ evaluates the supremum distance between their empirical cumulative distribution functions (ECDFs):

$$D_{n, m} = \sup_{x \in \mathbb{R}} \left| F_n(x) - G_m(x) \right|$$

where:

$$F_n(x) = \frac{1}{n} \sum_{i=1}^n \mathbb{I}(x_i \le x), \qquad G_m(x) = \frac{1}{m} \sum_{j=1}^m \mathbb{I}(x'_j \le x)$$

Under the null hypothesis $H_0: F = G$, the scaled test statistic converges asymptotically to the Kolmogorov distribution:

$$\sqrt{\frac{n m}{n + m}} D_{n, m} \xrightarrow{d} \sup_{t \in [0, 1]} |B(t)|$$

where $B(t)$ is a standard Brownian bridge. Drift is declared when the asymptotic $p$-value satisfies $p < \alpha$.

---

### 1.3 Two-Sample Anderson-Darling Test

The **Anderson-Darling test** evaluates divergence across the entire support while assigning higher quadratic weights to deviations located in the distribution tails:

$$A_{n, m}^2 = \frac{n m}{N} \int_{-\infty}^{+\infty} \frac{\left( F_n(x) - G_m(x) \right)^2}{H_N(x) \left( 1 - H_N(x) \right)} \, dH_N(x)$$

where $N = n + m$ and $H_N(x) = \frac{n F_n(x) + m G_m(x)}{N}$ represents the pooled empirical distribution function.

The denominator weight $w(t) = \frac{1}{t(1 - t)}$ penalizes divergences near the tails ($t \to 0$ and $t \to 1$), enabling earlier detection of boundary shifts and outlier density drifts than the Kolmogorov-Smirnov test.

---

## 2. Self-Supervised Neighbor Projection (SSNP)

To construct Supervised Decision Boundary Maps (SDBM) in high dimensions ($D > 2$), STRIDE implements **SSNP** (*Espadoto et al., 2020*), training a neural autoencoder consisting of:

* An encoder network $p_\theta: \mathbb{R}^D \to \mathbb{R}^2$ parameterized by weights $\theta$.
* An inverse decoder network $q_\phi: \mathbb{R}^2 \to \mathbb{R}^D$ parameterized by weights $\phi$.

### Joint Objective Function

The network parameters $\theta, \phi$ are optimized jointly over training instances $\{x_1, \dots, x_N\}$:

$$\min_{\theta, \phi} \mathcal{L}(\theta, \phi) = \mathcal{L}_{\text{recon}}(\theta, \phi) + \lambda_{\text{nbr}} \mathcal{L}_{\text{nbr}}(\theta)$$

#### Reconstruction Loss (Cycle Consistency)
Measures the Mean Squared Error (MSE) between original feature vectors and their decoded reconstructions:

$$\mathcal{L}_{\text{recon}} = \frac{1}{N} \sum_{i=1}^N \left\| x_i - q_\phi\left( p_\theta(x_i) \right) \right\|_2^2$$

#### Neighbor Distance Preservation Loss
Preserves local topological relationships by penalizing discrepancies between high-dimensional and 2D latent distances among $k$-nearest neighbors $\mathcal{N}_k(i)$:

$$\mathcal{L}_{\text{nbr}} = \frac{1}{N \cdot k} \sum_{i=1}^N \sum_{j \in \mathcal{N}_k(i)} \left( \left\| p_\theta(x_i) - p_\theta(x_j) \right\|_2 - d(x_i, x_j) \right)^2$$

where $d(x_i, x_j) = \|x_i - x_j\|_2$ denotes the Euclidean distance in the original feature space $\mathbb{R}^D$.

---

## 3. Cluster Correspondence and Drift Decision

### 3.1 Reference-Anchored Cluster Tracking (default)

Let $C = \{c_1, \dots, c_K\}$ be the X-Means centroids of one class in the reference window. A K-means partition is the Voronoi partition of its centroids, so every detection-window sample $x$ of the same class is assigned to

$$j^*(x) = \arg\min_{j} \lVert x - c_j \rVert_2,$$

which splits both windows by the same cell boundaries. Let $r_j$ be the $q$-quantile (default $q = 0.95$) of the distances of reference samples in cluster $j$ to $c_j$. A detection sample is **novel** if $\lVert x - c_{j^*(x)} \rVert_2 > r_{j^*(x)}$. The detection window is also clustered on its own, and a cluster whose members are mostly novel is reported as **appeared**. A reference cluster with reference share $\pi_j$ that receives $m_j$ of $n$ detection samples is reported as **disappeared** when $m_j < 0.2\, n \pi_j$ and the one-sided binomial test $P(\text{Bin}(n, \pi_j) \le m_j) < \alpha$.

### 3.2 Optimal Bipartite Centroid Matching (Hungarian Algorithm)

With `matching="independent"`, both windows are clustered separately. Let $C_{\text{before}} = \{c_1, \dots, c_{K_1}\}$ and $C_{\text{after}} = \{c'_1, \dots, c'_{K_2}\}$ denote cluster centroids extracted from reference and detection windows, respectively.

We construct a cost matrix $C \in \mathbb{R}^{K_1 \times K_2}$ whose entries define the pairwise Euclidean distance:

$$C_{i, j} = \left\| c_i - c'_j \right\|_2$$

Assuming $K = \min(K_1, K_2)$, optimal matching solves:

$$\min_{\pi \in \Pi} \sum_{i=1}^K C_{i, \pi(i)}$$

where $\Pi$ denotes the set of all bijective assignments. This combinatorial optimization is solved in polynomial time $\mathcal{O}(K^3)$ using the **Kuhn-Munkres (Hungarian)** algorithm.

### 3.3 Permutation Test

The drift strength $S$ (weighted share of triggered checks) is used as a test statistic. Under the null hypothesis of no drift, the samples of both windows are exchangeable within each class. For $b = 1, \dots, B$ the pooled samples of every class are randomly re-split into windows of the original sizes and the strength $S_b$ is recomputed. The p-value

$$p = \frac{1 + \sum_{b=1}^{B} \mathbb{1}[S_b \geq S]}{1 + B}$$

is valid for any data distribution, so rejecting when $p < \alpha$ bounds the false-alarm rate by $\alpha$ without tuning a threshold on $S$.

---

## 4. Bayesian Information Criterion in X-Means

X-Means evaluates whether a parent cluster should be partitioned into two child clusters using the **Bayesian Information Criterion (BIC)**.

Under the assumption of identical spherical Gaussian distributions with pooled variance $\hat{\sigma}^2$:

$$\hat{\sigma}^2 = \frac{1}{R - K} \sum_{i=1}^R \left\| x_i - c_{(i)} \right\|_2^2$$

where $R$ is the number of points assigned to the cluster, $K$ is the number of clusters evaluated ($K=1$ for parent, $K=2$ for split), and $c_{(i)}$ is the centroid nearest to $x_i$.

The log-likelihood of data points $D$ under model $M$ is:

$$\hat{l}(D) = -\frac{R \cdot D}{2} \ln(2\pi) - \frac{R \cdot D}{2} \ln(\hat{\sigma}^2) - \frac{R - K}{2} + R \ln(R) - R \ln(N)$$

The BIC score is then computed as:

$$\text{BIC}(M) = \hat{l}(D) - \frac{p}{2} \ln(R)$$

where $p = K(D + 1)$ represents the number of estimated parameters. The parent cluster is split if and only if:

$$\text{BIC}(\text{split}) > \text{BIC}(\text{parent})$$

---

## 5. Conformal Selective Classification for Drift Attribution

Let $W_{\text{cal}} = \{(z_i, t_i)\}_{i=1}^{n_{\text{cal}}}$ denote a held-out calibration window where $t_i \in \{0, 1\}$ represents the time window label. Given an auxiliary discriminator $g(z)$ outputting class probability estimates $\hat{P}(t \mid z)$, the non-conformity score for instance $i$ under label $y$ is defined as:

$$\alpha_i = 1 - \hat{P}(y \mid z_i)$$

For an unlabelled test sample $z_{n+1}$ and candidate label $y \in \{0, 1\}$, the conformal $p$-value is:

$$p_y(z_{n+1}) = \frac{1}{|W_{\text{cal}}| + 1} \left( 1 + \sum_{i \in W_{\text{cal}}} \mathbb{I}\left( \alpha_i \ge \alpha_{n+1}^y \right) \right)$$

### Abstention Condition

At user-specified significance level $\epsilon \in (0, 1)$:

$$\text{Decision}(z_{n+1}) = \begin{cases}
0, & \text{if } p_0(z_{n+1}) > \epsilon \text{ and } p_1(z_{n+1}) \le \epsilon \\
1, & \text{if } p_1(z_{n+1}) > \epsilon \text{ and } p_0(z_{n+1}) \le \epsilon \\
\text{Abstain}, & \text{otherwise (ambiguous region)}
\end{cases}$$

Attribution algorithms (such as PFI or SHAP) are computed exclusively on non-abstained instances, isolating features that unambiguously distinguish the reference window from the detection window.

---

## 6. ADWIN (Adaptive Windowing) Drift Bound

ADWIN (*Bifet & Gavaldà, 2007*) maintains a dynamically sized sliding window $W$ of observed stream values (such as binary prediction errors $e_t \in \{0, 1\}$). For every contiguous partition $W = W_0 \cdot W_1$ with subwindow lengths $n_0 = |W_0|$ and $n_1 = |W_1|$, the harmonic mean sample size is:

$$m = \frac{1}{\frac{1}{n_0} + \frac{1}{n_1}}$$

ADWIN tests the null hypothesis that subwindows $W_0$ and $W_1$ share identical expectations:

$$H_0: \mathbb{E}[\mu_{W_0}] = \mathbb{E}[\mu_{W_1}]$$

Drift is flagged whenever the empirical mean difference exceeds the Hoeffding cut threshold:

$$|\hat{\mu}_{W_0} - \hat{\mu}_{W_1}| \ge \epsilon_{\text{cut}}$$

where:

$$\epsilon_{\text{cut}} = \sqrt{\frac{1}{2m} \ln \left(\frac{4 |W|}{\delta}\right)}$$

and $\delta \in (0, 1)$ bounds the false positive alarm rate under stationary conditions. When the bound is exceeded, older observations in $W_0$ are dropped from the window buffer.

In STRIDE, `DualADWIN` operates two coupled estimators with confidence bounds configured such that $\delta_{\text{warn}} > \delta_{\text{drift}}$:

$$\epsilon_{\text{cut}}(\delta_{\text{warn}}) < \epsilon_{\text{cut}}(\delta_{\text{drift}})$$

This formulation guarantees that the early warning threshold triggers prior to final drift confirmation, providing an adaptive lookback horizon bounded by the current ADWIN subwindow width ($W$).
