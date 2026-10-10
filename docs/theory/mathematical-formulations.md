# Mathematical Formulations in STRIDE

This document outlines the core mathematical definitions, divergences, and loss functions underpinning STRIDE's xAI algorithms.

---

## 1. Wasserstein Metric (Earth Mover's Distance)

For two univariate probability measures $u$ and $v$ with cumulative distribution functions $U(x)$ and $V(x)$, the first Wasserstein distance $\mathcal{W}_1$ is defined as:

$$\mathcal{W}_1(u, v) = \int_{-\infty}^{+\infty} |U(x) - V(x)| \, dx$$

Unlike the Kullback-Leibler divergence, $\mathcal{W}_1$ remains defined and non-zero even when the supports of $u$ and $v$ are disjoint.

---

## 2. SSNP Bidirectional Projection Loss

Self-Supervised Neighbor Projection trains a lightweight neural network consisting of an encoder $p_\theta: \mathbb{R}^D \to \mathbb{R}^2$ and an inverse decoder $q_\phi: \mathbb{R}^2 \to \mathbb{R}^D$.

The objective minimizes a compound loss:

$$\mathcal{L}(\theta, \phi) = \mathcal{L}_{\text{recon}} + \lambda_{\text{nbr}} \mathcal{L}_{\text{nbr}}$$

where the reconstruction loss measures cycle-consistency:

$$\mathcal{L}_{\text{recon}} = \frac{1}{N} \sum_{i=1}^N \| x_i - q_\phi(p_\theta(x_i)) \|_2^2$$

and the neighbor loss preserves $k$-nearest neighbor distances in the 2D plane:

$$\mathcal{L}_{\text{nbr}} = \sum_{i=1}^N \sum_{j \in \mathcal{N}_k(i)} \left( \| p_\theta(x_i) - p_\theta(x_j) \|_2 - d(x_i, x_j) \right)^2$$

---

## 3. Hungarian Optimal Bipartite Centroid Assignment

Given $K_1$ centroids $C_{\text{before}} = \{c_1, \dots, c_{K_1}\}$ and $K_2$ centroids $C_{\text{after}} = \{c'_1, \dots, c'_{K_2}\}$ with $K = \min(K_1, K_2)$, optimal matching minimizes total Euclidean displacement:

$$\min_{\pi \in \Pi} \sum_{i=1}^K \| c_i - c'_{\pi(i)} \|_2$$

where $\Pi$ represents the set of all bijective assignments.

---

## 4. X-Means Bayesian Information Criterion (BIC)

X-Means decides whether to split a parent cluster into two children clusters using the BIC score. Assuming spherical Gaussian distributions with identical variance $\hat{\sigma}^2$:

$$\text{BIC}(M) = \hat{l}(D) - \frac{p}{2} \ln(R)$$

where:
- $\hat{l}(D)$ is the log-likelihood of data points $D$,
- $p$ is the number of estimated parameters,
- $R$ is the number of points in the evaluated cluster.

A parent cluster is split if and only if $\text{BIC}(\text{split}) > \text{BIC}(\text{parent})$.

---

## 5. ADWIN (Adaptive Windowing) Drift Bound

ADWIN (*Bifet & Gavaldà, 2007*) adaptively maintains a sliding window $W$ of observed stream values (such as binary prediction errors $e_t \in \{0, 1\}$). For every contiguous split $W = W_0 \cdot W_1$ with subwindow lengths $n_0 = |W_0|$ and $n_1 = |W_1|$, the harmonic mean sample size is:

$$m = \frac{1}{\frac{1}{n_0} + \frac{1}{n_1}}$$

ADWIN tests the null hypothesis that subwindows $W_0$ and $W_1$ have equal expectations:

$$H_0: \mathbb{E}[\mu_{W_0}] = \mathbb{E}[\mu_{W_1}]$$

Drift is flagged whenever the empirical mean divergence exceeds the Hoeffding cut threshold:

$$|\hat{\mu}_{W_0} - \hat{\mu}_{W_1}| \ge \epsilon_{\text{cut}}$$

where:

$$\epsilon_{\text{cut}} = \sqrt{\frac{1}{2m} \ln \left(\frac{4 |W|}{\delta}\right)}$$

and $\delta \in (0, 1)$ bounds the probability of a false positive alarm under stationary conditions. When the threshold is exceeded, older observations in $W_0$ are discarded.

In STRIDE, `DualADWIN` runs dual estimators with $\delta_{\text{warn}} > \delta_{\text{drift}}$:

$$\epsilon_{\text{cut}}(\delta_{\text{warn}}) < \epsilon_{\text{cut}}(\delta_{\text{drift}})$$

ensuring that the warning signal provides conservative pre-drift caching before final drift confirmation.
