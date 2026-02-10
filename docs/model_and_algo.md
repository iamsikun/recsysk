# Semiparametric Fatigue–Habit Recommender

This document specifies, in a reproducible and implementation-oriented way:

1. **A Monte-Carlo DGP** that generates the “追打 (chasing homogeneous content)” failure mode with **binary clicks**.
2. **A semiparametric structural model** (click + continuation) with **fatigue (satiation)** and **habit (reinforcement)** stocks, layered on top of a flexible deep baseline (DIEN/BERT4Rec-style).
3. **A detailed estimation algorithm** (cross-fitting + orthogonalization) suitable for KuaiRand-like logged data.
4. How to use the fitted model for:
   - **Prediction** (click probability under candidate items),
   - **Policy learning** (fatigue-aware reranking + safe off-policy evaluation).

---

## 0. Notation and core objects

- Users: $i = 1, \dots, n$
- Decision times (impressions): $t = 1, \dots, T_i$ (within and across sessions)
- Action: the shown video/item at time $t$:  
  $$
  A_{it} \in \mathcal{J}
  $$
- **Click** outcome (binary):  
  $$
  W_{it}\in\{0,1\}.
  $$
- Continuation indicator:  
  $$
  C_{it}\in\{0,1\},
  $$
  where $C_{it}=1$ means user remains active into $t+1$ (or session continues), and $C_{it}=0$ is stopping/exit (absorbing).
- Discount factor: $\beta\in(0,1)$.

### Content representations

You need **two kinds** of item representations:

1. **Baseline relevance features** $X_j$ (can include IDs, collaborative features, creator stats, etc.).
2. **Fatigue-attribute embedding** $a_j \in \mathbb{R}^d$, intended to capture the semantic axis along which homogeneity causes fatigue.

**Crucial design rule:** keep $a_j$ reserved for fatigue; do **not** let the baseline learner trivially reconstruct the fatigue index from the same inputs, or you will destroy identifiability (the nuisance can absorb the effect you want to estimate).

---

## 1. DGP for Monte Carlo (binary click + fatigue-driven stopping)

This DGP generates the 追打 phenomenon: myopic relevance-chasing increases early CTR but accelerates fatigue accumulation and increases stopping, reducing long-run value.

### 1.1 Users

Choose dimension $d$ (e.g., 16 or 32). For each user $i$, draw:

- Long-run interest vector:
  $$
  \theta_i \sim \mathcal{N}(0, I_d).
  $$
- Fatigue sensitivity (heterogeneous):
  $$
  \log \lambda_i \sim \mathcal{N}(\mu_\lambda, \sigma_\lambda^2),\quad \lambda_i>0.
  $$
- Fatigue decay (heterogeneous):
  $$
  \text{logit}(\delta_{Fi}) \sim \mathcal{N}(\mu_\delta, \sigma_\delta^2),\quad \delta_{Fi}\in(0,1).
  $$
- Habit sensitivity (heterogeneous):
  $$
  \eta_i \sim \text{HalfNormal}(\sigma_\eta),\quad \eta_i\ge 0.
  $$
- Habit decay (homogeneous for simplicity): $\delta_G\in(0,1)$.

Initialize stocks:
- $F_{i1}=0\in\mathbb{R}^d$,
- $G_{i1}=0\in\mathbb{R}$.

### 1.2 Items

Generate $J$ items with fatigue-attribute embeddings $a_j \in \mathbb{R}^d$, normalized to $|a_j|=1$. Use a mixture of clusters to create “homogeneous content regions”:

$$
a_j \sim \sum_{m=1}^M \pi_m ,\mathcal{N}(\mu_m,\Sigma_m),\quad |a_j|=1.
$$

Optionally add baseline features $X_j$ correlated with $a_j$, plus noise, to mimic real platforms.

### 1.3 State evolution (fatigue and habit)

$$
T_{it} := \langle F_{it}, a_{A_{it}}\rangle.
$$

If $C_{it}=1$ (continue), update:

**Fatigue stock**
$$
F_{i,t+1}=(1-\delta_{Fi})F_{it} + a_{A_{it}}.
$$

**Habit stock**
$$
G_{i,t+1}=(1-\delta_G)G_{it} + W_{it}.
$$

If $C_{it}=0$, the process enters an absorbing state (no further actions/outcomes).

### 1.4 Click outcome (binary)

Clicks follow a logistic index model:

$$
\Pr(W_{it}=1\mid \theta_i,F_{it},G_{it},A_{it})
=
\sigma\Big(
\kappa_0 + \kappa_1\theta_i^\top a_{A_{it}}
- \lambda_i\langle F_{it},a_{A_{it}}\rangle
+ \eta_i G_{it}
+ u_{it}
\Big),
$$
where $\sigma(x)=1/(1+e^{-x})$, and $u_{it} \sim \mathcal{N}(0, \sigma_u^2)$ is a preference shock **observed by the platform** but not fully by the econometrician (confounding).

### 1.5 Continuation (stop hazard)

$$
\Pr(C_{it}=1\mid W_{it},F_{it},G_{it})
=
\sigma\Big(
\alpha_0 + \alpha_1 W_{it} + \eta_i G_{it} - \xi|F_{it}|
\Big).
$$

Fatigue reduces continuation; clicks and habit increase continuation.

### 1.6 Logging policy (behavior policy) generating confounding

Mimic a two-stage recommender:

1. **Candidate generation**: produce candidate set $\mathcal{J}_{it}$ of size `K`:
   - with probability $\rho$ (exploration), sample $K$ uniformly,
   - with probability $1-\rho$, sample high-relevance items w.r.t. $\theta_i^\top a_j$.

2. **Ranking / action choice**: choose action via softmax over a score that **ignores fatigue**:
$$
\widehat s_{it}(j)=\kappa_0+\kappa_1\theta_i^\top a_j + \eta_i G_{it}+\omega_{it}(j),
\quad \omega_{it}(j)\sim \mathcal{N}(0,\sigma_\omega^2),
$$
$$
\pi_0(j\mid S_{it})\propto \exp(\tau\widehat s_{it}(j)),\quad j\in\mathcal{J}_{it}.
$$

This creates: high early CTR + rapid fatigue buildup → 追打.

---

## 2. Structural–Semiparametric Model (binary click)

Now we specify the model used for **estimation on logged data**.

### 2.1 Working state and fatigue index

Let the econometrician’s observed/constructed state be:
$$
S_{it}=(h_{it},F_{it},G_{it},X_{it}),
$$
where:
- $h_{it}$ is a learned representation (DIEN/BERT4Rec-style) summarizing history,
- $F_{it}$ and $G_{it}$ are computed stocks (§2.2),
- $X_{it}$ includes time/device/session position.

Define fatigue index:
$$
T_{it} = \langle F_{it}, a_{A_{it}}\rangle.
$$

### 2.2 Stock construction in real data

Given decay parameters $(δ_F, δ_G)$, construct:
$$
F_{i,t+1}=(1-\delta_F)F_{it} + a_{A_{it}}\,\mathbf{1}\{C_{it}=1\},\quad F_{i1}=0,
$$
$$
G_{i,t+1}=(1-\delta_G)G_{it} + g(W_{it})\,\mathbf{1}\{C_{it}=1\},\quad G_{i1}=0,
$$
where for clicks $g(w)=w$ is natural (habit counts “successful engagements”).

### 2.3 Click model (semiparametric logit-index)

We model:
$$
p_0(S_{it},A_{it})
:=\Pr(W_{it}=1\mid S_{it},A_{it})
=
\sigma\Big(
g_0(S_{it},X_{A_{it}})
- \lambda_0(S_{it})\,T_{it}
+ \eta_{w,0}(S_{it})\,G_{it}
\Big).
$$

- $g_0$ is a **high-dimensional nuisance baseline index** (learned by deep models).
- $\lambda_0(S)$ is the heterogeneous **fatigue sensitivity**.
- $\eta_{w,0}(S)$ is optional heterogeneous **habit reinforcement** in click propensity.

**Marginal effect on probability:**
$$
\frac{\partial p_0}{\partial T} = -\lambda_0(S)\,p_0(1-p_0).
$$

### 2.4 Continuation model (optional but recommended)

$$
\Pr(C_{it}=1\mid S_{it},A_{it})
=
\sigma\Big(
\alpha_0 + \alpha_1\,p_0(S_{it},A_{it})
+ \eta_{c,0}(S_{it})G_{it}
- \xi_0(S_{it})|F_{it}|
\Big).
$$

Continuation is where “fatigue causes churn” becomes explicit.

---

## 3. Identification sketch (why KuaiRand-style random exposure helps)

In KuaiRand-like data, a small fraction of recommendations are replaced by random items ($is_rand=1$). This provides variation in shown items that is closer to exogenous given state.

Pragmatically, you can:

- Use `is_rand=1` subset to **estimate** fatigue effects ($\lambda$) with minimal confounding,
- Use `is_rand=0` standard logs primarily to learn the rich representation $h_{it}$ and baseline nuisance $g_0$.

Even when not perfectly randomized, treating random exposure as a known logging policy with propensities improves overlap and stabilizes off-policy evaluation.

---

## 4. Estimation algorithm (click outcome): detailed steps

### 4.1 Inputs and outputs

**Inputs**
- Logged impressions `(user_id, time, item_id, W, ... )`
- Optional continuation/session boundaries (construct $C_{it}$ via time gaps)
- Item metadata (tags, captions, categories, creator info)
- Random exposure indicator (e.g., `is_rand`) and/or propensities if available

**Outputs**
- Baseline click model $\hat{g}(S,X_A)$ and probability $\hat{p}^0=σ(\hat{g})$
- Treatment regression $\hat{\mu}_T(S,X_A)$
- Heterogeneous fatigue model $\hat{\lambda}(S)$ and summaries (average sensitivity, AME)
- Optionally continuation model parameters
- Policy objects: reranking rule and OPE estimator for policy value

### 4.2 Step A — Build fatigue-attribute embedding $a_j$

You must define $a_j$ so that “homogeneity” is meaningful.

Recommended options (in increasing sophistication):
1. **Category-probability vector**: use provided category hierarchy probabilities (dimension = number of categories).
2. **Tag bag-of-words**: TF-IDF or hashing trick over tags.
3. **Text embedding**: encode captions with a language model; optionally reduce to $d$ dims via PCA.
4. **Hybrid**: concatenate (category probs, tag embedding, caption embedding) then project to $d$ dims.

Normalize $a_j \leftarrow a_j / ||a_j||$.

**Important:** keep this representation fixed (or updated slowly) to maintain interpretability.

### 4.3 Step B — Define sessions and continuation $C_{it}$ (if needed)

If $C_{it}$ is not directly available, define sessions using a time-gap rule:

- Sort impressions by `time_ms` per user.
- Start a new session when gap > Δ (e.g., 30 minutes).
- Set $C_{it}=1$ if the next impression is in the same session; $C_{it}=0$ for the last impression in each session.

This converts impression logs into a dynamic process with stopping.

### 4.4 Step C — Compute stocks $F_{it}, G_{it}$

Pick decay $(\delta_F, \delta_G)$.

- Start with a grid (e.g., $\delta_F \in {0.05, 0.1, 0.2}$, $\delta_G \in {0.05, 0.1}$).
- Choose via out-of-sample predictive likelihood on continuation (or by maximizing long-horizon policy fit).

Compute recursively:
- $F_{t+1}=(1-\delta_F)F_t + a_{A_t} * 1{C_t=1}$
- $G_{t+1}=(1-\delta_G)G_t + W_t * 1{C_t=1}$

Then compute fatigue index:
- $T_{it} = <F_{it}, a_{A_{it}}>$

### 4.5 Step D — Cross-fitting split (user-level)

Split users into $K$ folds (e.g., 5). For each fold $k$:

- Train nuisance models on folds $\neq k$,
- Compute residuals/pseudo-outcomes on fold $k$,
- Aggregate across folds.

This is essential for valid inference and to prevent overfitting bias.

### 4.6 Step E — Estimate baseline click index $\hat{g}(S,X_A)$ (nuisance)

Goal: approximate:
$$
\Pr(W=1\mid S,X_A)=\sigma(g_0(S,X_A)).
$$

Implementation:
- Any strong classifier (deep net, GBDT).
- Inputs: $h_{it}$, $X_{it}$, $X_{A_{it}}$, optionally $G_{it}$.
- **Do NOT include** $a_{A_{it}}$ in a way that makes $T=<F,a>$ redundant.  
  (You may include coarse categories in $X_A$, but then define $a$ in a more semantic space to preserve variation.)

Output on held-out fold: baseline probability:
$$
\hat{p}^{0}_{it}=\sigma(\hat{g}(S_{it},X_{A_{it}})).
$$

### 4.7 Step F — Estimate treatment regression $\hat{\mu}_T(S,X_A)$

Estimate:
$$
\mu_{T,0}(S,X_A)=\mathbb{E}[T\mid S,X_A].
$$

Use a flexible regressor (since $T$ is continuous):
- Inputs: $(S, X_A)$ **excluding** $a$ itself.
- Output: $\hat{\mu}_T(S,X_A)$ on held-out fold.

Then residualize:
$$
\tilde{T}_{it}=T_{it}-\hat{\mu}_T(S_{it},X_{A_{it}}).
$$

### 4.8 Step G — Form logistic pseudo-outcome

Compute:
$$
\tilde{Y}_{it}=\frac{W_{it}-\hat{p}^{0}_{it}}{\hat{p}^{0}_{it}(1-\hat{p}^{0}_{it})}.
$$

Stabilize:
- Clip $\hat{p}^0$ to $[ε, 1-ε]$ with $ε≈1e-3$.
- Weight $\omega_{it}=\hat{p}^0(1-\hat{p}^0)$.

### 4.9 Step H — Learn heterogeneous fatigue sensitivity $\hat{\lambda}(S)$ (HTE)

Pick a function class:
- Neural net: $\lambda_\theta(S)=softplus(f_\theta(S))$ to enforce nonnegativity.
- Or spline / random forest.

Estimate by weighted least squares on each held-out fold:
$$
\hat{\theta}
\in\arg\min_\theta
\sum_{(i,t)\in\text{fold}}
\omega_{it}\big(\tilde{Y}_{it}+\lambda_\theta(S_{it})\,\tilde{T}_{it}\big)^2.
$$

Aggregate across folds to obtain cross-fitted $\hat{\lambda}(S)$.

### 4.10 Step I — Summaries and inference

Compute average sensitivity:
$$
\hat{\Lambda}=\frac{\sum \hat{\lambda}(S_{it})\,\tilde{T}_{it}^2}{\sum \tilde{T}_{it}^2}.
\frac{\sum \hat{\lambda}(S_{it})\,\tilde{T}_{it}^2}{\sum \tilde{T}_{it}^2}.
$$

Compute average marginal effect on click probability:
$$
\widehat{AME}=-\frac{1}{N}\sum \hat{\lambda}(S_{it})\,\hat{p}_{it}(1-\hat{p}_{it}),
\frac{1}{N}\sum \hat{\lambda}(S_{it})\,\hat{p}_{it}(1-\hat{p}_{it}),
$$
where $\hat{p}$ is the full fitted click probability including the fatigue term.

Inference:
- Use **user-level block bootstrap** (recommended): resample users, rerun the full pipeline (or rerun Steps E–I with fixed embeddings).
- Alternatively compute user-clustered sandwich SEs on low-dimensional functionals.

---

## 5. Using the fitted model for prediction

You can predict click probability for a candidate item $j$ shown to user $i$ at time $t$ as follows.

### 5.1 Construct current state

Given user history up to time `t`:
1. Compute representation $h_{it}$ from your sequence model (DIEN/BERT4Rec encoder).
2. Compute stocks $(F_{it}, G_{it})$ by replaying history (or maintaining online).
3. Form state $S_{it}=(h_{it},F_{it},G_{it},X_{it})$.

### 5.2 Compute baseline probability

Compute baseline index from the nuisance model:
$$
\hat{p}^0_{it}(j)=\sigma\big(\hat{g}(S_{it},X_j)\big).
$$

### 5.3 Apply fatigue and habit adjustments

Compute fatigue index for item $j$:
$$
T_{it}(j)=\langle F_{it},a_j\rangle.
$$

Predict heterogeneous fatigue sensitivity:
$$
\hat{\lambda}_{it}=\hat{\lambda}(S_{it}).
$$

Then the predicted click probability under item $j$ is:
$$
\hat{p}_{it}(j)=
\sigma\Big(
\text{logit}(\hat{p}^0_{it}(j))
- \hat{\lambda}_{it}\,T_{it}(j)
+ \widehat{\eta_w}(S_{it})\,G_{it}
\Big),
$$
where $\eta_w$ is optional; if omitted, drop it.

### 5.4 Calibration

Because you combine models, you should calibrate:
- Fit a calibration layer (Platt scaling or isotonic) on a validation set mapping $\hat{p}$ to observed click frequencies.
- Report Brier score and calibration curves.

---

## 6. Policy learning: fatigue-aware reranking + safe OPE

### 6.1 Candidate generation (fixed)

In practice you keep candidate generation as in production (retrieval stage). Let the candidate set at time $t$ be $\mathcal{J}_{it}$.

### 6.2 Fatigue-aware reranking policy class

Define a scoring rule:
$$
Score_{it}(j)=
\text{logit}\big(\hat{p}^0_{it}(j)\big)
- \kappa\,\hat{\lambda}(S_{it})\,\langle F_{it},a_j\rangle
+ \text{ExploreTerm}_{it}(j),
$$
and recommend $\argmax_{j\in\mathcal{J}_{it}} Score_{it}(j)$.

- $\kappa$ is a policy knob (often start at 1).
- $\text{ExploreTerm}_{it}(j)$ can be a small novelty bonus to preserve overlap.

This creates a *structural reranker* that explicitly avoids over-serving already-fatiguing attributes.

### 6.3 Tuning κ offline (policy search)

Grid-search κ (and novelty parameters) to maximize estimated long-horizon value:

1. For each $\kappa$, define policy $\pi_\kappa$.
2. Estimate value $\widehat V_{DR}(\pi_\kappa)$ using DR OPE.
3. Choose $\kappa^*\in\arg\max \widehat V_{DR}(\pi_\kappa)$.

### 6.4 Off-policy evaluation (OPE) — practical recipe

Define reward (examples):
- $R=W$ (clicks),
- $R=W*C$ (clicks before stopping),
- $R=\beta^{t-1}W$ aggregated (discounted clicks).

Estimate:
- Propensity $\hat{\pi}_0(A|S)$:
  - If you have a randomized bucket (`is_rand=1`), use known/estimated propensities in that bucket.
  - Else estimate a multinomial/softmax propensity model over candidates.
- Outcome model $\hat{Q}(S,a)=E[R|S,a]$ using a flexible regressor (can reuse click model).

Then compute DR estimator:
$$
\widehat V_{DR}(\pi)=
\frac{1}{n}\sum_i \sum_t \beta^{t-1}
\left[
\sum_a \pi(a|S_{it})\,\hat{Q}(S_{it},a)
+
\frac{\pi(A_{it}|S_{it})}{\hat{\pi}_0(A_{it}|S_{it})}
\big(R_{it}-\hat{Q}(S_{it},A_{it})\big)
\right].
$$

**Variance control**
- Clip weights: $w=\pi/\pi_0$ clipped to $[0, w_max]$.
- Restrict policy to “supported” actions: only rerank within the logged candidate sets.

### 6.5 Long-horizon dynamics and stopping

To capture fatigue-induced churn, you need session-level continuation:

- Build sessions and $C_{it}$.
- Define $R=W*C$ and apply discount.
- Optionally fit a continuation model and incorporate it into $\hat{Q}$ for better DR performance.

### 6.6 Deployment-safe policy learning

Before A/B tests:
- Ensure overlap: evaluate how often your new policy chooses items rarely shown in logs.
- Start with small κ and conservative novelty.
- Use random exposure buckets to validate causal effects of fatigue penalty.

---

## 7. Implementation checklist specific to KuaiRand-like data

1. **Choose dataset version that preserves sequences** (for RL/OPE tasks).
2. **Construct semantic embeddings $a_j$** from category probabilities + captions/tags.
3. **Compute sessions** from timestamps; define $C_{it}$.
4. **Train representation $h_{it}$** using standard logs (`is_rand=0`), not only random exposures.
5. **Estimate fatigue effect** primarily using random exposures ($is_rand=1$) or use it as a strong overlap source.
6. **Run cross-fitting** at user level.
7. **Do DR OPE** for reranking policies; restrict to candidate sets; weight clipping.

---

## 8. Extensions you can add later (optional)

- **Heterogeneous decay:** learn $\delta_F(S)$ instead of fixed $\delta_F$ (harder but doable with state-space methods).
- **Attribute-specific fatigue:** maintain multiple fatigue stocks by category; $F^{(k)}$ for each cluster.
- **Slate/ranking actions:** define $T$ for lists as exposure-weighted sum over positions; use slate OPE estimators.
- **LLM-enhanced $a_j$:** use a language model to derive richer semantic attributes (topic, tone, emotion), improving fatigue measurement.

---

## 9. Minimal pseudocode (end-to-end)

**Training pipeline**
1. Build item embeddings $a_j$ and baseline features $X_j$.
2. Sessionize logs → define $C_{it}$.
3. Split users into K folds.
4. For each fold:
   - Train baseline click model $\hat{g}$ on other folds → predict $\hat{p}^0$ on fold.
   - Train treatment regression $\hat{\mu}_T$ on other folds → predict on fold.
   - Build pseudo outcome $\tilde{Y}$ and residual $\tilde{T}$.
   - Fit $\lambda_\theta(S)$ on fold via weighted LS.
5. Aggregate cross-fitted $\hat{\lambda}(S)$, compute $\hat{\Lambda}$, AME, bootstrap SEs.
6. Fit/validate continuation model (optional).
7. Policy learning: grid-search $\kappa$; OPE each policy with DR; pick $\kappa^*$.

**Online prediction**
- Maintain $F,G$ per user online.
- For each candidate item $j$:
  - compute baseline score $\text{logit}(\hat{p}^0(S,j))$,
  - compute fatigue penalty $\hat{\lambda}(S) * <F,a_j>$,
  - rank by adjusted score.

---

### End
