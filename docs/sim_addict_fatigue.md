# Monte Carlo Simulation for Recommendation with User Fatigue

This section presents a Monte Carlo design tailored to recommendation with user fatigue in which the immediate engagement outcome is a **binary click indicator** ($Y_{it} \in \{0,1\}$). The DGP is constructed to reproduce what the industry calls "追打问题": myopic, relevance-chasing recommendation increases near-term click probability but accelerates fatigue accumulation, raising the stop hazard and reducing long-run value.

## 1. Overview and objectives

The Monte Carlo study serves three purposes.

1. **Mechanism validation:** establish that repeated exposure to homogeneous content can be locally click-maximizing but globally suboptimal due to fatigue-driven attrition.

2. **Estimation validation:** evaluate bias, RMSE, and coverage of the proposed orthogonalized estimator of the average fatigue sensitivity ($\Lambda_0=\mathbb{E}[\lambda_0(S)]$) and the average marginal effect (AME) on click probability.

3. **Policy evaluation validation:** compare IPS and doubly robust (DR) off-policy estimators of ($V(\pi)$) under varying degrees of overlap and confounding.

---

## 2. Users, items, and attribute structure

### 2.1 Users

We simulate ($n$) users indexed by ($i$). Each user has:

* **Long-run interest vector** ($\theta_i\in\mathbb{R}^d$),
* **Fatigue sensitivity** ($\lambda_i>0$),
* **Fatigue decay** ($\delta_{Fi}\in(0,1)$),
* **Habit sensitivity** ($\eta_i\ge 0$),
* Optional **habit decay** ($\delta_G\in(0,1)$) (kept homogeneous for simplicity).

Draw:
$$
\theta_i \sim \mathcal{N}(0, I_d),\quad
\log \lambda_i \sim \mathcal{N}(\mu_\lambda,\sigma_\lambda^2),
\quad
\text{logit}(\delta_{Fi})\sim \mathcal{N}(\mu_\delta,\sigma_\delta^2),
\quad
\eta_i \sim \text{HalfNormal}(\sigma_\eta).
$$

This structure induces heterogeneity in both the *level* and *persistence* of fatigue.

### 2.2 Items

We generate ($J$) items with attribute embeddings ($a_j\in\mathbb{R}^d$), normalized ($|a_j|=1$). To create “homogeneous regions,” we draw from a mixture of ($M$) clusters:
$$
a_j \sim \sum_{m=1}^M \pi_m ,\mathcal{N}(\mu_m,\Sigma_m),
\qquad |a_j|=1.
$$
Cluster structure makes it possible for a myopic recommender to lock onto a cluster and repeatedly serve similar content, causing fatigue.

---

### 2.3 State evolution: fatigue and habit stocks

Initialize ($F_{i1}=0\in\mathbb{R}^d$) and ($G_{i1}=0\in\mathbb{R}$). Conditional on continuation, stocks evolve as:

**Fatigue stock**
$$
F_{i,t+1}=(1-\delta_{Fi})F_{it}+a_{A_{it}}\cdot \mathbf{1}\{C_{it}=1\}.
$$

**Habit stock**
$$
G_{i,t+1}=(1-\delta_G)G_{it}+W_{it}\cdot \mathbf{1}\{C_{it}=1\}.
$$

Thus, clicking reinforces future propensity via (G), while repeated exposure reinforces fatigue via (F).

Define the fatigue index:
$$
T_{it} := \langle F_{it}, a_{A_{it}}\rangle.
$$

---

### 2.4 Outcomes: click and continuation

#### 2.4.1 Click model (binary outcome)

Clicks follow a logistic index model:
$$
\Pr(W_{it}=1\mid \theta_i,F_{it},G_{it},A_{it})
=
\sigma\left(
\underbrace{\kappa_0 + \kappa_1\theta_i^\top a_{A_{it}}}_{\text{relevance}}
-
\underbrace{\lambda_i\langle F_{it},a_{A_{it}}\rangle}_{\text{fatigue penalty}}
+
\underbrace{\eta_i,G_{it}}_{\text{habit reinforcement}}
+
u_{it}
\right),
$$
where $u_{it}\sim\mathcal{N}(0,\sigma_u^2)$ is an idiosyncratic preference shock observed by the platform but not fully by the econometrician (this generates confounding).

#### 2.4.2 Continuation / stopping model

Continuation is Bernoulli with:
$$
\Pr(C_{it}=1\mid W_{it},F_{it},G_{it})
=
\sigma\left(
\alpha_0+\alpha_1 W_{it}+\eta_i G_{it}-\xi|F_{it}|
\right).
$$
The term $-\xi|F_{it}|$ captures fatigue-induced stopping; higher fatigue raises the hazard of exit.

Exit is absorbing: if $C_{it}=0$, then $C_{i,t'}=0$ and $W_{i,t'}=0$ for all $t'>t$.

---

### 2.5 Logged actions: behavior policy and candidate set

To mimic real recommenders, the logging policy is myopically optimized for click probability **under a misspecified state** (it underweights fatigue). This creates both confounding and the 追打 failure mode.

#### 2.5.1 Candidate generation

At each time $t$, the system considers a candidate set $\mathcal{J}_{it}\subset\mathcal{J}$ of size $K$, formed as:

* with probability $\rho$, sample $K$ items uniformly (exploration bucket),
* with probability $1-\rho$, sample $K$ items with high $\theta_i^\top a_j$ (relevance-biased recall).

This reflects the two-stage architecture (retrieval then ranking).

#### 2.5.2 Logging policy $\pi_0$

Given candidates $\mathcal{J}_{it}$, define the logging score:
$$
\widehat s_{it}(j)=\kappa_0+\kappa_1\theta_i^\top a_j + \eta_i G_{it} + \omega_{it}(j),
\quad \omega_{it}(j)\sim\mathcal{N}(0,\sigma_\omega^2),
$$
and choose action via softmax:
$$
\pi_0(j\mid S_{it}) \propto \exp(\tau \widehat s_{it}(j)),\qquad j\in\mathcal{J}_{it}.
$$
Crucially, $\widehat s_{it}$ does **not** include $-\lambda_i \langle F,a_j\rangle$, so $\pi_0$ tends to repeatedly pick within the same cluster even when fatigue is accumulating.

We vary $(\rho,\tau,\sigma_\omega)$ to control overlap and confounding strength.

---

### 2.6 Policies to be evaluated

We evaluate three policies, all restricted to the same candidate set $\mathcal{J}_{it}$ to ensure overlap.

1. **Myopic relevance policy**
   $$
   \pi_{\text{myopic}}(j\mid S_{it})=\mathbf{1}{j=\arg\max_{k\in\mathcal{J}_{it}} \widehat s_{it}(k)}.
   $$

2. **Fatigue-aware structural reranking policy**
   $$
   \text{Score}_{it}(j)=\widehat s_{it}(j)-\kappa\langle F_{it},a_j\rangle,
   \quad
   \pi_{\text{fatigue}}(j\mid S_{it})=\mathbf{1}{j=\arg\max_{k\in\mathcal{J}_{it}}\text{Score}_{it}(k)}.
   $$
   We consider two versions:

* **oracle** $\kappa=\mathbb{E}[\lambda_i]$,
* **estimated** $\kappa=\widehat\Lambda$ from Section 7.

3. **One-step diversity heuristic**
   Penalize similarity to the *last* item only:
   $$
   \text{Score}_{it}^{\text{1-step}}(j)=\widehat s_{it}(j)-\kappa_1 \langle a_{A_{i,t-1}},a_j\rangle.
   $$
   This distinguishes “true fatigue stock” from a common heuristic.

---

### 2.7 Performance metrics and estimands

We report:

* **Short-run click rate**
  $$
  \text{CTR}(\pi)=\mathbb{E}^\pi[W_{it}\mid t\le t_0]
  $$
  for early (t_0) (e.g., first 10 impressions).

* **Expected session length**
  $$
  L(\pi)=\mathbb{E}^\pi\left[\sum_{t\ge1} C_{it}\right].
  $$

* **Long-run discounted value**
  $$
  V(\pi)=\mathbb{E}^\pi\left[\sum_{t\ge1}\beta^{t-1} R_{it}\right],
  \qquad R_{it}=W_{it}\cdot C_{it}.
  $$

* **Fatigue accumulation**
  $$
  \mathbb{E}^\pi[|F_{it}|] \ \text{and}\ \mathbb{E}^\pi[T_{it}]
  $$
  as process diagnostics.

The “追打 gap” is
$$
\Delta V := V(\pi_{\text{fatigue}})-V(\pi_{\text{myopic}}).
$$

---

### 2.8 Estimators compared

#### 2.8.1 Estimation of fatigue effect

Using the binary-outcome estimation procedure in Section 7, we estimate:

* $\widehat\lambda(S)$ (heterogeneous),
* $\widehat\Lambda$ (average fatigue sensitivity),
* $\widehat{\text{AME}}$ (average marginal effect on click probability).

We compare to naïve alternatives:

* **Naïve logistic regression** of $W$ on $T$ and controls without orthogonalization,
* **Naïve ML** where a flexible model includes both $a_j$ and $F$ and learns $T$ implicitly (demonstrates “soaking up” and confounding).

We report bias/RMSE and confidence interval coverage for $\Lambda_0$ and AME.

#### 2.8.2 Off-policy evaluation (OPE)

For each target policy $\pi$, we estimate $V(\pi)$ using:

* **IPS**
  $$
  \widehat V_{\mathrm{IPS}}(\pi)=\frac{1}{n}\sum_i\sum_t \beta^{t-1}\frac{\pi(A_{it}\mid S_{it})}{\pi_0(A_{it}\mid S_{it})}R_{it}.
  $$

* **Doubly robust (DR)**
  $$
  \widehat V_{\mathrm{DR}}(\pi)=\frac{1}{n}\sum_i\sum_t \beta^{t-1}\left[
  \sum_a \pi(a\mid S_{it})\widehat Q(S_{it},a)
      -
  \frac{\pi(A_{it}\mid S_{it})}{\pi_0(A_{it}\mid S_{it})}\big(R_{it}-\widehat Q(S_{it},A_{it})\big)
  \right].
$$

We also report stability diagnostics:

* distribution of importance weights,
* effective sample size,
* sensitivity to overlap $\rho$ and softmax temperature $\tau$.

---

### 2.9 Experimental design: factors varied

We vary four key dimensions:

1. **Fatigue strength and heterogeneity:** $(\mu_\lambda,\sigma_\lambda)$
2. **Fatigue persistence:** $\sigma_\delta$ in $\delta_{Fi}$
3. **Confounding strength:** $\sigma_u,\tau$
4. **Overlap/exploration:** $\rho$ and candidate size $K$

For each configuration we simulate $B$ independent datasets (e.g., $B=200$) and compute Monte Carlo averages of estimators and their standard errors.

---

### 2.10 Expected qualitative results (what the design produces)

The design is calibrated so that:

1. **Myopic wins early CTR but loses long-run value.**
   Since $\pi_{\text{myopic}}$ chases relevance without internalizing fatigue, it tends to stay in one cluster, yielding high early $\theta^\top a$ and thus higher initial click probability. But this drives $F_t$ rapidly upward, increasing $T_t$ and $|F_t|$, reducing click probability via (9.6) and increasing exit via (9.7). Consequently $V(\pi_{\text{myopic}})$ falls below $V(\pi_{\text{fatigue}})$ for moderate-to-large $\mu_\lambda$ and $\xi$.

2. **Fatigue-aware reranking improves retention and $V(\pi)$.**
   By penalizing $\langle F,a\rangle$, $\pi_{\text{fatigue}}$ diversifies across clusters, slows fatigue accumulation, increases expected session length, and increases discounted click-based value.

3. **One-step diversity helps but is dominated.**
   The heuristic $\text{Score}_{it}^{\text{1-step}}$ reduces immediate repetition but fails to control multi-step fatigue accumulation; it typically improves on myopic but underperforms the full-stock penalty.

4. **Orthogonalized estimation recovers average fatigue sensitivity with good coverage under overlap.**
   Naïve logistic regressions overstate or understate fatigue effects depending on the confounding regime; the orthogonal pseudo-outcome method in Section 7 yields small bias and near-nominal coverage when $\rho$ is not too small and $\pi_0$ is sufficiently stochastic.

5. **DR-OPE dominates IPS in variance, especially when overlap is moderate.**
   IPS becomes unstable as $\rho\to 0$ or $\tau$ becomes large; DR remains substantially more stable when either $\widehat Q$ or $\widehat\pi_0$ is accurate, but still degrades when overlap fails.

---

### 2.11 Reporting template (what goes in the paper)

For each Monte Carlo configuration, report:

* Table: true $\Lambda_0$, mean $\widehat\Lambda$, bias, RMSE, 95\% coverage; similarly for AME.
* Table: true $V(\pi)$ for $\pi_{\text{myopic}},\pi_{\text{fatigue}},\pi_{\text{1-step}}$, and OPE estimates (IPS, DR) with bias/RMSE.
* Figure: trajectories of $\mathbb{E}[|F_t|]$, $\mathbb{E}[T_t]$, CTR over time, and survival curves (continuation) under each policy.
* Diagnostics: histograms of importance weights and effective sample size.

---

If you want, I can also add an **Appendix Monte Carlo** where (i) the econometrician’s attribute embedding $a_j$ is *noisy* (to mimic imperfect semantic measurement), and (ii) we show how using an “LLM-quality” embedding (less noise) increases the estimated fatigue effect’s power and improves policy gains—this maps directly to your Kuaishou/KuaiRand setting.
