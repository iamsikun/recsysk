"""
Monte Carlo Simulation for Recommendation with User Fatigue.

This module implements the Data Generating Process (DGP) for a recommendation
environment where user engagement (clicks) and retention (continuation) are
influenced by latent fatigue and habit stocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Type aliases for clarity in array shapes and contents
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(slots=True, frozen=True)
class AddictFatigueConfig:
    """
    Configuration for the additive fatigue-habit simulation model.

    This configuration contains all hyper-parameters for the user population,
    item catalog, and the behavioral outcome models (click and continuation).
    """

    # Simulation scale
    n_users: int = 1_000
    n_items: int = 300
    embedding_dim: int = 8
    horizon: int = 40  # Maximum session length (T)

    # Item cluster parameters (Section 2.2)
    n_clusters: int = 6
    cluster_std: float = 0.35
    cluster_weights: tuple[float, ...] | None = None

    # User heterogeneity parameters (Section 2.1)
    mu_lambda: float = -0.2  # Mean of log fatigue sensitivity (log lambda_i)
    sigma_lambda: float = 0.5  # Std of log fatigue sensitivity
    mu_delta_f: float = -0.4  # Mean of logit fatigue decay (logit delta_Fi)
    sigma_delta_f: float = 0.8  # Std of logit fatigue decay
    sigma_eta: float = 0.4  # Scale for HalfNormal habit sensitivity (eta_i)

    # Habit decay (homogeneous for simplicity, Section 2.3)
    delta_g: float = 0.2

    # Click model parameters (Section 2.4.1)
    kappa0: float = -0.5  # Baseline click propensity (kappa_0)
    kappa1: float = 1.2  # Relevance sensitivity (kappa_1)
    sigma_u: float = 0.6  # Idiosyncratic preference shock std (sigma_u)

    # Continuation model parameters (Section 2.4.2)
    alpha0: float = 0.4  # Baseline continuation propensity (alpha_0)
    alpha1: float = 1.0  # Click reinforcement on retention (alpha_1)
    xi: float = 0.7  # Fatigue penalty on continuation (xi)
    seed: int | None = None  # Random seed for reproducibility

    def __post_init__(self) -> None:
        """Validate configuration parameters for consistency and physical bounds."""
        if self.n_users <= 0:
            raise ValueError("n_users must be positive")
        if self.n_items <= 0:
            raise ValueError("n_items must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if self.cluster_std <= 0:
            raise ValueError("cluster_std must be positive")
        if self.sigma_lambda < 0:
            raise ValueError("sigma_lambda must be non-negative")
        if self.sigma_delta_f < 0:
            raise ValueError("sigma_delta_f must be non-negative")
        if self.sigma_eta < 0:
            raise ValueError("sigma_eta must be non-negative")
        if not (0.0 < self.delta_g < 1.0):
            raise ValueError("delta_g must lie in (0, 1)")
        if self.sigma_u < 0:
            raise ValueError("sigma_u must be non-negative")
        if self.cluster_weights is not None:
            if len(self.cluster_weights) != self.n_clusters:
                raise ValueError("cluster_weights must have length equal to n_clusters")
            if any(weight <= 0 for weight in self.cluster_weights):
                raise ValueError("cluster_weights must be strictly positive")

    def normalized_cluster_weights(self) -> FloatArray:
        """Return normalized probabilities for item cluster sampling."""
        if self.cluster_weights is None:
            return np.full(self.n_clusters, 1.0 / self.n_clusters, dtype=np.float64)
        weights = np.asarray(self.cluster_weights, dtype=np.float64)
        return weights / weights.sum()


@dataclass(slots=True, frozen=True)
class UserPopulation:
    """
    User-level latent parameters drawn from distributions in Section 2.1.

    Attributes:
        theta: Long-run interest vectors (n_users, embedding_dim).
        fatigue_sensitivity: Sensitivity to accumulated fatigue stock (lambda_i).
        fatigue_decay: Decay rate for fatigue stock (delta_Fi).
        habit_sensitivity: Strength of habit reinforcement (eta_i).
    """

    theta: FloatArray
    fatigue_sensitivity: FloatArray
    fatigue_decay: FloatArray
    habit_sensitivity: FloatArray


@dataclass(slots=True, frozen=True)
class ItemCatalog:
    """
    Item embeddings and cluster metadata from Section 2.2.

    Attributes:
        embeddings: Normalized item attribute vectors (n_items, embedding_dim).
        cluster_ids: Mapping of items to their source cluster index (n_items,).
        cluster_centers: Centroids of the clusters (n_clusters, embedding_dim).
    """

    embeddings: FloatArray
    cluster_ids: IntArray
    cluster_centers: FloatArray


@dataclass(slots=True, frozen=True)
class SimulationRollout:
    """
    Results and intermediate states of a simulation run (Sections 2.3 & 2.4).

    This object stores the full trajectory of a simulation, which can be
    used for estimation or off-policy evaluation tasks.
    """

    users: UserPopulation
    items: ItemCatalog
    actions: IntArray  # item index served at each step (n_users, horizon)
    active: IntArray  # 1 if user is alive at step t, 0 otherwise
    clicks: IntArray  # click indicators W_it (n_users, horizon)
    continuation: IntArray  # continuation indicators C_it (n_users, horizon)
    click_prob: FloatArray  # P(W_it=1)
    continuation_prob: FloatArray  # P(C_it=1)
    fatigue_index: FloatArray  # T_it = <F_it, a_it>
    fatigue_norm: FloatArray  # |F_it|
    click_shock: FloatArray  # idiosyncratic shocks u_it
    fatigue_state: FloatArray  # history of F_it vectors (n_users, horizon+1, d)
    habit_state: FloatArray  # history of G_it levels (n_users, horizon+1)

    @property
    def rewards(self) -> IntArray:
        """Realized rewards R_it = W_it * C_it (engagement while active)."""
        return self.clicks * self.continuation


class AddictFatigueSimulator:
    """
    Monte Carlo simulator implementing Sections 2.1 to 2.4 of the design doc.

    This simulator handles the initialization of users/items and the execution
    of simulation steps based on a provided action policy.
    """

    def __init__(self, config: AddictFatigueConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def sample_users(self) -> UserPopulation:
        """
        Sample latent user parameters according to Section 2.1.

        Uses Normal, LogNormal, LogitNormal, and HalfNormal distributions.
        """
        cfg = self.config

        # Long-run interest: theta_i ~ N(0, I_d)
        theta = self._rng.normal(
            loc=0.0,
            scale=1.0,
            size=(cfg.n_users, cfg.embedding_dim),
        )

        # Fatigue sensitivity: log lambda_i ~ N(mu_lambda, sigma_lambda)
        log_lambda = self._rng.normal(
            loc=cfg.mu_lambda, scale=cfg.sigma_lambda, size=cfg.n_users
        )
        fatigue_sensitivity = np.exp(log_lambda)

        # Fatigue decay: logit(delta_Fi) ~ N(mu_delta, sigma_delta)
        logit_delta_f = self._rng.normal(
            loc=cfg.mu_delta_f, scale=cfg.sigma_delta_f, size=cfg.n_users
        )
        fatigue_decay = _sigmoid(logit_delta_f)

        # Habit sensitivity: eta_i ~ HalfNormal(sigma_eta)
        # Approximated by absolute value of Normal
        habit_sensitivity = np.abs(
            self._rng.normal(loc=0.0, scale=cfg.sigma_eta, size=cfg.n_users)
        )

        return UserPopulation(
            theta=theta,
            fatigue_sensitivity=fatigue_sensitivity,
            fatigue_decay=fatigue_decay,
            habit_sensitivity=habit_sensitivity,
        )

    def sample_items(self) -> ItemCatalog:
        """
        Sample item catalog from a mixture of clusters (Section 2.2).

        Items are drawn from cluster-specific Normal distributions and then
        projected onto the unit sphere to ensure |a_j|=1.
        """
        cfg = self.config

        # Generate M cluster centroids
        cluster_centers = self._rng.normal(
            loc=0.0,
            scale=1.0,
            size=(cfg.n_clusters, cfg.embedding_dim),
        )

        # Sample cluster assignment for each item
        cluster_ids = self._rng.choice(
            cfg.n_clusters,
            size=cfg.n_items,
            p=cfg.normalized_cluster_weights(),
        )

        # Sample item positions around centroids
        noise = self._rng.normal(
            loc=0.0,
            scale=cfg.cluster_std,
            size=(cfg.n_items, cfg.embedding_dim),
        )
        raw_embeddings = cluster_centers[cluster_ids] + noise

        # Final item attributes: a_j s.t. |a_j| = 1
        embeddings = _normalize_rows(raw_embeddings)

        return ItemCatalog(
            embeddings=embeddings,
            cluster_ids=cluster_ids.astype(np.int64),
            cluster_centers=_normalize_rows(cluster_centers),
        )

    def sample_uniform_actions(
        self,
        horizon: int | None = None,
        n_users: int | None = None,
    ) -> IntArray:
        """Generate a random action matrix (uniform sampling from catalog)."""
        n_steps = self.config.horizon if horizon is None else horizon
        n_agents = self.config.n_users if n_users is None else n_users

        if n_steps <= 0:
            raise ValueError("horizon must be positive")
        if n_agents <= 0:
            raise ValueError("n_users must be positive")

        return self._rng.integers(
            low=0,
            high=self.config.n_items,
            size=(n_agents, n_steps),
            dtype=np.int64,
        )

    def simulate(
        self,
        actions: IntArray,
        users: UserPopulation | None = None,
        items: ItemCatalog | None = None,
    ) -> SimulationRollout:
        """
        Run the simulation rollout given an action sequence.

        This method computes the time-evolution of fatigue and habit stocks,
        and determines click and continuation outcomes at each step.
        """
        users = users or self.sample_users()
        items = items or self.sample_items()
        self._validate_population(users=users, items=items)

        # Basic input validation
        action_array = np.asarray(actions, dtype=np.int64)
        if action_array.ndim != 2:
            raise ValueError("actions must be a 2D array (n_users, horizon)")

        n_users, horizon = action_array.shape
        if n_users != users.theta.shape[0]:
            raise ValueError(
                f"actions row count {n_users} must match user population {users.theta.shape[0]}"
            )

        # Pre-allocate output tensors
        embedding_dim = items.embeddings.shape[1]
        active = np.zeros((n_users, horizon), dtype=np.int64)
        clicks = np.zeros((n_users, horizon), dtype=np.int64)
        continuation = np.zeros((n_users, horizon), dtype=np.int64)
        click_prob = np.zeros((n_users, horizon), dtype=np.float64)
        continuation_prob = np.zeros((n_users, horizon), dtype=np.float64)
        fatigue_index = np.zeros((n_users, horizon), dtype=np.float64)
        fatigue_norm = np.zeros((n_users, horizon), dtype=np.float64)
        click_shock = np.zeros((n_users, horizon), dtype=np.float64)

        # State histories (initialized at 0 for t=0)
        fatigue_state = np.zeros(
            (n_users, horizon + 1, embedding_dim), dtype=np.float64
        )
        habit_state = np.zeros((n_users, horizon + 1), dtype=np.float64)

        # 'alive' tracks users who haven't stopped yet
        alive = np.ones(n_users, dtype=bool)
        cfg = self.config

        for t in range(horizon):
            # Carry over state to the next step
            fatigue_state[:, t + 1, :] = fatigue_state[:, t, :]
            habit_state[:, t + 1] = habit_state[:, t]

            # Record who is active at the start of time t
            active[:, t] = alive.astype(np.int64)

            if not np.any(alive):
                # Early exit if everyone has stopped
                continue

            active_idx = np.flatnonzero(alive)
            action_t = action_array[active_idx, t]
            item_vec_t = items.embeddings[action_t]

            # 1. Compute latent indices
            # -------------------------
            fatigue_t = fatigue_state[active_idx, t, :]
            habit_t = habit_state[active_idx, t]

            # Relevance: theta_i^T * a_it
            relevance_t = np.einsum("ij,ij->i", users.theta[active_idx], item_vec_t)

            # Fatigue index: T_it = <F_it, a_it> (Eq from Section 2.3)
            fatigue_index_t = np.einsum("ij,ij->i", fatigue_t, item_vec_t)

            # Fatigue norm: |F_it| (penalty for continuation hazard in Section 2.4.2)
            fatigue_norm_t = np.linalg.norm(fatigue_t, axis=1)

            # Idiosyncratic preference shock: u_it ~ N(0, sigma_u^2)
            shock_t = self._rng.normal(loc=0.0, scale=cfg.sigma_u, size=active_idx.size)

            # 2. Click model (Section 2.4.1)
            # ------------------------------
            # Probability determined by relevance, fatigue penalty, and habit reinforcement
            click_logit_t = (
                cfg.kappa0
                + cfg.kappa1 * relevance_t
                - users.fatigue_sensitivity[active_idx] * fatigue_index_t
                + users.habit_sensitivity[active_idx] * habit_t
                + shock_t
            )
            click_prob_t = _sigmoid(click_logit_t)
            click_draw_t = self._rng.random(active_idx.size) < click_prob_t
            click_int_t = click_draw_t.astype(np.int64)

            # 3. Continuation model (Section 2.4.2)
            # -------------------------------------
            # Exit is absorbing; probability depends on engagement and total fatigue burden
            continuation_logit_t = (
                cfg.alpha0
                + cfg.alpha1 * click_int_t
                + users.habit_sensitivity[active_idx] * habit_t
                - cfg.xi * fatigue_norm_t
            )
            continuation_prob_t = _sigmoid(continuation_logit_t)
            continuation_draw_t = (
                self._rng.random(active_idx.size) < continuation_prob_t
            )

            # Log step outcomes
            click_prob[active_idx, t] = click_prob_t
            continuation_prob[active_idx, t] = continuation_prob_t
            fatigue_index[active_idx, t] = fatigue_index_t
            fatigue_norm[active_idx, t] = fatigue_norm_t
            click_shock[active_idx, t] = shock_t
            clicks[active_idx, t] = click_int_t
            continuation[active_idx, t] = continuation_draw_t.astype(np.int64)

            # 4. State evolution (Section 2.3)
            # --------------------------------
            # Stocks evolve only for users who continue (C_it = 1)
            next_idx = active_idx[continuation_draw_t]
            if next_idx.size > 0:
                # Fatigue stock: F_{i,t+1} = (1-delta_Fi)F_it + a_it
                decay_f = 1.0 - users.fatigue_decay[next_idx]
                fatigue_state[next_idx, t + 1, :] = (
                    decay_f[:, None] * fatigue_state[next_idx, t, :]
                    + items.embeddings[action_array[next_idx, t]]
                )

                # Habit stock: G_{i,t+1} = (1-delta_G)G_it + W_it
                habit_state[next_idx, t + 1] = (1.0 - cfg.delta_g) * habit_state[
                    next_idx, t
                ] + clicks[next_idx, t]

            # Update survival status for the next loop iteration
            alive[:] = False
            alive[next_idx] = True

        return SimulationRollout(
            users=users,
            items=items,
            actions=action_array,
            active=active,
            clicks=clicks,
            continuation=continuation,
            click_prob=click_prob,
            continuation_prob=continuation_prob,
            fatigue_index=fatigue_index,
            fatigue_norm=fatigue_norm,
            click_shock=click_shock,
            fatigue_state=fatigue_state,
            habit_state=habit_state,
        )

    def simulate_uniform(
        self,
        horizon: int | None = None,
        users: UserPopulation | None = None,
        items: ItemCatalog | None = None,
    ) -> SimulationRollout:
        """Run a simulation using a uniform random selection policy."""
        if users is not None:
            n_users = users.theta.shape[0]
        else:
            n_users = self.config.n_users
        action_matrix = self.sample_uniform_actions(horizon=horizon, n_users=n_users)
        return self.simulate(actions=action_matrix, users=users, items=items)

    def _validate_population(self, users: UserPopulation, items: ItemCatalog) -> None:
        """Check consistency of user and item data structures."""
        if users.theta.ndim != 2:
            raise ValueError("users.theta must be 2D")
        if items.embeddings.ndim != 2:
            raise ValueError("items.embeddings must be 2D")
        if users.theta.shape[1] != items.embeddings.shape[1]:
            raise ValueError("User and item embeddings must share the same dimension")

        n_users = users.theta.shape[0]
        for name, value in (
            ("fatigue_sensitivity", users.fatigue_sensitivity),
            ("fatigue_decay", users.fatigue_decay),
            ("habit_sensitivity", users.habit_sensitivity),
        ):
            if value.shape != (n_users,):
                raise ValueError(f"users.{name} must have shape ({n_users},)")

        if users.fatigue_sensitivity.min() <= 0:
            raise ValueError("fatigue_sensitivity must be strictly positive")
        if np.any(users.fatigue_decay <= 0) or np.any(users.fatigue_decay >= 1):
            raise ValueError("fatigue_decay must lie in (0, 1)")
        if np.any(users.habit_sensitivity < 0):
            raise ValueError("habit_sensitivity must be non-negative")

        if items.cluster_ids.shape != (items.embeddings.shape[0],):
            raise ValueError("items.cluster_ids must have one entry per item")


def _normalize_rows(values: FloatArray) -> FloatArray:
    """Project rows onto the unit sphere (|row| = 1)."""
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-12)
    return values / safe_norms


def _sigmoid(values: FloatArray) -> FloatArray:
    """Compute the logistic sigmoid function with numerical stability."""
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    # Stability for large positive values
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    # Stability for large negative values
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result
