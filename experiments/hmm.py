"""Shared AR-HMM inference, temporal GMM initialization, and EM fitting."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import math
import secrets
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
from torch import nn


def infer_states(emission: torch.Tensor, log_start: torch.Tensor,
                 log_transition: torch.Tensor, *,
                 return_evidence: bool = False) -> dict[str, torch.Tensor]:
    """Exact log-space inference; shapes [B,C,T,K], [C,K], [C,K,K]."""
    length = emission.shape[2]
    forward = [emission[:, :, 0] + log_start]
    for t in range(1, length):
        forward.append(emission[:, :, t] + torch.logsumexp(
            forward[-1].unsqueeze(-1) + log_transition, dim=-2))
    backward = [torch.zeros_like(forward[0])]
    for t in range(length - 2, -1, -1):
        backward.append(torch.logsumexp(log_transition + (
            emission[:, :, t + 1] + backward[-1]).unsqueeze(-2), dim=-1))
    alpha = torch.stack(forward, dim=2)
    beta = torch.stack(backward[::-1], dim=2)
    likelihood = torch.logsumexp(alpha[:, :, -1], dim=-1)
    # Local normalization avoids cancellation from subtracting a large sequence LL.
    gamma = (alpha + beta).softmax(dim=-1)
    messages = log_transition[None, :, None] + (emission[:, :, 1:] + beta[:, :, 1:]).unsqueeze(-2)
    conditional = messages.softmax(dim=-1)
    xi = gamma[:, :, :-1, :, None] * conditional
    result = {"log_likelihood": likelihood, "gamma": gamma, "xi": xi,
              "conditional": conditional}
    if return_evidence:
        prefix = torch.logsumexp(alpha, dim=-1)
        result["log_evidence"] = torch.cat((prefix[..., :1],
                                             prefix[..., 1:] - prefix[..., :-1]), dim=-1)
    return result


def ordered_pairs(posterior: dict[str, torch.Tensor]) -> torch.Tensor:
    """P(z_t=a,z_s=b|x,c), [B,C,T,T,K,K], retaining direction and both positions."""
    gamma, conditional = posterior["gamma"], posterior["conditional"]
    length = gamma.shape[2]
    by_lag = [torch.diag_embed(gamma)]
    for lag in range(1, length):
        by_lag.append(by_lag[-1][:, :, :-1] @ conditional[:, :, lag - 1:])
    # Pack the upper triangle by lag, then gather all ordered time pairs in one
    # operation. Reverse pairs transpose state axes without merging positions.
    packed = torch.cat(by_lag, dim=2)
    t = torch.arange(length, device=gamma.device)
    lag = (t[:, None] - t[None, :]).abs()
    offset = lag * length - lag * (lag - 1) // 2
    index = offset + torch.minimum(t[:, None], t[None, :])
    pairs = packed.index_select(2, index.flatten()).reshape(
        *gamma.shape[:2], length, length, gamma.shape[-1], gamma.shape[-1])
    reverse = (t[:, None] > t[None, :])[None, None, :, :, None, None]
    return torch.where(reverse, pairs.transpose(-1, -2), pairs)


class ARHMM(nn.Module):
    """Independent pi/A/means and full/diagonal covariance, with AR(0/1/2) emissions.

    At t=0: x_0 ~ N(mu_z, Sigma_z).
    AR(1) at t>0: x_t ~ N(mu_z + B_z (x_{t-1} - mu_z), Sigma_z).
    AR(2) additionally uses B2_z (x_{t-2} - mu_z) from t=2 onward;
    its second position uses B1 only. Lags never cross sample boundaries.
    Conditional on the observed sequence these are ordinary HMM emission
    potentials, so forward/backward and pair posteriors remain exact.
    """

    def __init__(self, models: int, states: int = 6, channels: int = 3,
                 ar_order: int = 1, covariance_floor: float = 1e-3,
                 covariance_type: str = "full"):
        super().__init__()
        if ar_order not in (0, 1, 2) or states < 2 or not math.isfinite(covariance_floor) or covariance_floor <= 0:
            raise ValueError("Use ar_order 0/1/2, states >= 2 and a positive covariance floor")
        self.models, self.states, self.channels = models, states, channels
        self.ar_order, self.covariance_floor = ar_order, covariance_floor
        if covariance_type not in ("full", "diag"):
            raise ValueError("Use full or diag emission covariance")
        self.covariance_type = covariance_type
        self.start_logits = nn.Parameter(torch.zeros(models, states))
        self.transition_logits = nn.Parameter(torch.zeros(models, states, states))
        self.means = nn.Parameter(torch.randn(models, states, channels) * 0.1)
        self.covariance_factor = nn.Parameter(torch.eye(channels).repeat(models, states, 1, 1))
        if ar_order:
            self.ar = nn.Parameter(torch.zeros(models, states, channels, channels * ar_order))
        else:
            self.register_buffer("ar", torch.zeros(models, states, channels, channels))

    def covariance(self) -> torch.Tensor:
        if self.covariance_type == "diag":
            diagonal = self.covariance_factor.diagonal(dim1=-2, dim2=-1)
            return torch.diag_embed(diagonal.square() + self.covariance_floor)
        lower = self.covariance_factor.tril()
        identity = torch.eye(self.channels, dtype=lower.dtype, device=lower.device)
        return lower @ lower.transpose(-1, -2) + self.covariance_floor * identity

    @torch.no_grad()
    def _initialize_model(self, train_x: np.ndarray, index: int, seed: int) -> None:
        """Initialize one model using input sequences only.

        KMeans supplies state centers and covariance estimates, not a frozen
        state representation. Initial pi/A use within-sample hard assignments
        with one pseudocount per outcome, without a diagonal/sticky preference.
        """
        x = np.asarray(train_x, dtype=np.float64)
        if x.ndim != 3 or x.shape[2] != self.channels or not np.isfinite(x).all():
            raise ValueError("Expected finite training sequences [N,T,channels]")
        flat = x.reshape(-1, self.channels)
        with threadpool_limits(limits=1):
            clustering = KMeans(n_clusters=self.states, n_init=5, random_state=seed).fit(flat)
        assignment = clustering.labels_.reshape(x.shape[:2])
        start = np.bincount(assignment[:, 0], minlength=self.states).astype(float) + 1
        transition = np.ones((self.states, self.states))
        np.add.at(transition, (assignment[:, :-1].ravel(), assignment[:, 1:].ravel()), 1)
        covariance = []
        for k in range(self.states):
            residual = flat[clustering.labels_ == k] - clustering.cluster_centers_[k]
            cov = residual.T @ residual / max(1, len(residual))
            # Shrink only the initializer; off-diagonal covariance stays trainable.
            cov = 0.95 * cov + 0.05 * np.diag(np.diag(cov))
            if self.covariance_type == "diag":
                cov = np.diag(np.diag(cov))
            covariance.append(np.linalg.cholesky(cov + self.covariance_floor * np.eye(self.channels)))
        for target, value in (
            (self.start_logits[index], np.log(start / start.sum())),
            (self.transition_logits[index], np.log(transition / transition.sum(axis=1, keepdims=True))),
            (self.means[index], clustering.cluster_centers_),
            (self.covariance_factor[index], np.stack(covariance)),
        ):
            target.copy_(torch.as_tensor(value, dtype=target.dtype, device=target.device))
        self.ar[index].zero_()

    def log_emissions(self, x: torch.Tensor) -> torch.Tensor:
        # [B,C,T,K,D]; the AR predictor is reset at every sample boundary.
        centered = x[:, None, :, None, :] - self.means[None, :, None]
        if self.ar_order == 2:
            lag1 = torch.zeros_like(centered)
            lag2 = torch.zeros_like(centered)
            lag1[:, :, 1:] = centered[:, :, :-1]
            lag2[:, :, 2:] = centered[:, :, :-2]
            history = torch.cat([lag1, lag2], dim=-1)
            prediction = torch.einsum("bctki,ckji->bctkj", history, self.ar)
            residual = centered - prediction
        else:
            prediction = torch.einsum("bctki,ckji->bctkj", centered[:, :, :-1], self.ar)
            residual = torch.cat([centered[:, :, :1], centered[:, :, 1:] - prediction], dim=2)
        cholesky = torch.linalg.cholesky(self.covariance())
        whitened = torch.linalg.solve_triangular(
            cholesky[None, :, None], residual.unsqueeze(-1), upper=False).squeeze(-1)
        logdet = 2 * cholesky.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        return -0.5 * (self.channels * math.log(2 * math.pi) + logdet[None, :, None]
                       + whitened.square().sum(dim=-1))

    def forward(self, x: torch.Tensor, *, return_evidence: bool = False) -> dict[str, torch.Tensor]:
        if return_evidence:
            return infer_states(self.log_emissions(x), self.start_logits.log_softmax(-1),
                                self.transition_logits.log_softmax(-1), return_evidence=True)
        return infer_states(self.log_emissions(x), self.start_logits.log_softmax(-1),
                            self.transition_logits.log_softmax(-1))


class SharedARHMM(ARHMM):
    """One label-free AR-HMM; the singleton model axis is not a class axis."""

    def __init__(self, states: int = 6, channels: int = 3, ar_order: int = 1,
                 covariance_floor: float = 1e-3, covariance_type: str = "full"):
        super().__init__(1, states, channels, ar_order, covariance_floor, covariance_type)

    @torch.no_grad()
    def initialize(self, train_x: np.ndarray, seed: int) -> None:
        self._initialize_model(train_x, 0, seed)


def temporal_features(posterior: dict[str, torch.Tensor]) -> torch.Tensor:
    """[B,T,C*(K+K*K)], with every xi_t attached to its source time t."""
    gamma = posterior["gamma"].permute(0, 2, 1, 3).flatten(start_dim=2)
    xi = posterior["xi"].permute(0, 2, 1, 3, 4).flatten(start_dim=2)
    # Last time point has no outgoing transition; it is not connected to a new sample.
    xi = F.pad(xi, (0, 0, 0, 1))
    return torch.cat([gamma, xi], dim=-1)


class TemporalRelationBias(nn.Module):
    """Learn separate weights for patch offsets, ordered states, classes and lags."""

    def __init__(self, heads: int, classes: int, states: int, patch_size: int, length: int):
        super().__init__()
        self.patch_size, self.length = patch_size, length
        self.weights = nn.Parameter(torch.empty(heads, classes, patch_size, patch_size, states, states))
        nn.init.normal_(self.weights, std=0.02)
        self.lag_weights = nn.Parameter(torch.ones(heads, 2 * length - 1))
        t = torch.arange(length)
        self.register_buffer("lag_index", t[None, :] - t[:, None] + length - 1)
        # Independent uniform states: same-time pairs are diagonal, others iid.
        reference = torch.full((length, length, states, states), 1.0 / states ** 2)
        reference[t, t] = torch.eye(states) / states
        self.register_buffer("uniform_pairs", reference[None, None], persistent=False)

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        batch, classes, length, _, states, _ = pairs.shape
        patch, tokens = self.patch_size, length // self.patch_size
        # No averaging of the P*P ordered offset pairs, including same-patch pairs.
        relation = (pairs - self.uniform_pairs).reshape(
            batch, classes, tokens, patch, tokens, patch, states, states)
        lag = self.lag_weights[:, self.lag_index].reshape(-1, tokens, patch, tokens, patch)
        bias = torch.einsum("bcirjsuv,hcrsuv,hirjs->bhij", relation, self.weights, lag)
        # Subtracting the iid uniform reference makes its relation zero;
        # row centering additionally removes the mean relation per query.
        return bias - bias.mean(dim=-1, keepdim=True)


def fuse_attention_scores(content_scores: torch.Tensor, relation_scores: torch.Tensor,
                          mode: str) -> torch.Tensor:
    """Combine raw QK scores with the same full-position HMM relation features."""
    if mode == "attention":
        return content_scores + relation_scores
    if mode == "attention_gate":
        # A positive, bounded gate starts near identity and cannot reverse score signs.
        return content_scores * (1.0 + 0.5 * torch.tanh(relation_scores))
    raise ValueError(f"Unsupported attention fusion mode: {mode}")


def adjacent_observations(x: np.ndarray) -> np.ndarray:
    """One [previous,current] row per within-sequence transition, never boundaries."""
    return np.concatenate((x[:, :-1], x[:, 1:]), axis=-1).reshape(-1, 2 * x.shape[-1])


def marginal_responsibilities(x: np.ndarray, mixture: GaussianMixture,
                              dimensions: slice) -> np.ndarray:
    """Exact Gaussian-mixture marginal responsibilities for the first observation."""
    means = mixture.means_[:, dimensions]
    covariances = mixture.covariances_[:, dimensions, dimensions]
    log_scores = []
    for weight, mean, covariance in zip(mixture.weights_, means, covariances):
        lower = np.linalg.cholesky(covariance)
        residual = solve_triangular(lower, (x - mean).T, lower=True)
        log_scores.append(np.log(weight) - 0.5 * (
            x.shape[1] * np.log(2 * np.pi) + 2 * np.log(np.diag(lower)).sum()
            + np.square(residual).sum(axis=0)))
    log_scores = np.stack(log_scores, axis=1)
    return np.exp(log_scores - logsumexp(log_scores, axis=1, keepdims=True))


@torch.no_grad()
def initialize_shared_hmm(hmm: SharedARHMM, train_x: np.ndarray, seed: int,
                          method: str, gmm_max_iterations: int = 5000,
                          gmm_tolerance: float = 1e-8, accept_gmm_max_iterations: bool = False, *,
                          gmm_n_init: int = 1, gmm_init_params: str = "random_from_data",
                          gmm_reg_covar: float | None = None) -> tuple[dict, np.ndarray | None]:
    """Set full initial parameters; returned state weights are initializer-only.

    The caller fits the original tied-mu AR emission to GMM responsibilities by
    full conditional maximization, then starts exact HMM EM/ECM. No product of
    marginal probabilities substitutes for any HMM pair posterior.
    """
    if method not in ("gmm", "random", "temporal_gmm",
                      "temporal_gmm_soft_full", "temporal_gmm_hard_full"):
        raise ValueError("Use gmm, random or a temporal GMM initialization")
    if not isinstance(gmm_n_init, int) or gmm_n_init < 1:
        raise ValueError("gmm_n_init must be a positive integer")
    if gmm_init_params not in ("kmeans", "k-means++", "random", "random_from_data"):
        raise ValueError("Unsupported GMM initialization strategy")
    gmm_reg_covar = hmm.covariance_floor if gmm_reg_covar is None else float(gmm_reg_covar)
    if not math.isfinite(gmm_reg_covar) or gmm_reg_covar <= 0:
        raise ValueError("gmm_reg_covar must be finite and positive")
    x = np.asarray(train_x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] < 2 or x.shape[2] != hmm.channels or not np.isfinite(x).all():
        raise ValueError("Expected complete finite multichannel sequences")
    flat, states = x.reshape(-1, hmm.channels), hmm.states
    rng = np.random.default_rng(seed)
    # The existing random-prior methods share these draws at a given seed.
    start = rng.dirichlet(np.ones(states))
    transition = rng.dirichlet(np.ones(states), size=states)
    report = {"method": method, "seed": seed, "uses_labels": False,
              "samples": len(x), "time_points": len(flat), "states": states,
              "start_transition_initialization": "independent Dirichlet(1), no sticky preference",
              "gmm": None}
    weights = None
    if method == "random":
        means = flat[rng.choice(len(flat), size=states, replace=False)].copy()
        centered = flat - flat.mean(axis=0)
        covariance = centered.T @ centered / len(flat) + hmm.covariance_floor * np.eye(hmm.channels)
        covariances = np.repeat(covariance[None], states, axis=0)
    else:
        observations = flat if method == "gmm" else adjacent_observations(x)
        mixture = GaussianMixture(
            n_components=states, covariance_type="full", init_params=gmm_init_params,
            n_init=gmm_n_init, random_state=seed, reg_covar=gmm_reg_covar,
            max_iter=gmm_max_iterations, tol=gmm_tolerance)
        with threadpool_limits(limits=1):
            mixture.fit(observations)
            if not mixture.converged_ and not accept_gmm_max_iterations:
                raise RuntimeError(f"Full GMM initializer did not converge in {gmm_max_iterations} iterations")
            if method == "gmm":
                weights = mixture.predict_proba(flat).reshape(len(x), x.shape[1], states)
                means, covariances = mixture.means_.copy(), mixture.covariances_.copy()
            else:
                later = mixture.predict_proba(observations).reshape(len(x), x.shape[1] - 1, states)
                first = marginal_responsibilities(x[:, 0], mixture, slice(hmm.channels, None))
                weights = np.concatenate((first[:, None], later), axis=1)
                if method == "temporal_gmm_hard_full":
                    weights = np.eye(states, dtype=np.float64)[weights.argmax(axis=-1)]
                if method in ("temporal_gmm_soft_full", "temporal_gmm_hard_full"):
                    start_counts = weights[:, 0].sum(axis=0)
                    transition_counts = np.einsum(
                        "ntj,ntk->jk", weights[:, :-1], weights[:, 1:])
                    # Positive initializer support keeps every state and edge
                    # learnable by the unchanged full HMM EM updates.
                    pseudocount = 1.0
                    start = (start_counts + pseudocount) / (start_counts.sum() + states * pseudocount)
                    transition = (transition_counts + pseudocount)
                    transition /= transition.sum(axis=1, keepdims=True)
                    report["start_transition_initialization"] = (
                        "hard GMM state counts with one full-support pseudocount per cell"
                        if method == "temporal_gmm_hard_full" else
                        "adjacent soft GMM responsibility outer products with one full-support "
                        "pseudocount per cell; initializer only, not HMM pair posterior")
                    report["start_counts"] = start_counts.tolist()
                    report["transition_counts"] = transition_counts.tolist()
                    report["start_transition_pseudocount"] = pseudocount
                mass = weights.sum(axis=(0, 1))
                if np.any(mass <= 0):
                    raise RuntimeError("A temporal GMM component has no assigned training observations")
                means = np.einsum("ntk,ntd->kd", weights, x) / mass[:, None]
                covariances = []
                for k in range(states):
                    residual = x - means[k]
                    covariance = np.einsum("nt,nti,ntj->ij", weights[:, :, k], residual, residual) / mass[k]
                    covariances.append(covariance + hmm.covariance_floor * np.eye(hmm.channels))
                covariances = np.stack(covariances)
        report["gmm"] = {"converged": bool(mixture.converged_), "termination": ("converged" if mixture.converged_ else "max_iterations"),
                         "accepted_at_max_iterations": bool(not mixture.converged_ and accept_gmm_max_iterations),
                         "iterations": int(mixture.n_iter_),
                         "observations": len(observations), "dimensions": observations.shape[1],
                         "covariance_type": "full", "init_params": gmm_init_params, "n_init": gmm_n_init,
                         "reg_covar": gmm_reg_covar,
                         "max_iterations": gmm_max_iterations, "mean_lower_bound_tolerance": gmm_tolerance,
                         "mean_lower_bound": float(mixture.lower_bound_),
                         "first_position": ("current-observation marginal of the 6D mixture"
                                            if method.startswith("temporal_gmm") else "3D mixture responsibility")}
    factors = []
    for covariance in covariances:
        if hmm.covariance_type == "diag":
            covariance = np.diag(np.diag(covariance))
        # Preserve the full covariance/eigenvectors in Sigma = L L^T + floor I.
        values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
        residual_covariance = (vectors * np.maximum(values - hmm.covariance_floor, 1e-10)) @ vectors.T
        factors.append(np.linalg.cholesky(residual_covariance))
    for parameter, value in ((hmm.start_logits[0], np.log(start)),
                             (hmm.transition_logits[0], np.log(transition)),
                             (hmm.means[0], means),
                             (hmm.covariance_factor[0], np.stack(factors))):
        parameter.copy_(torch.as_tensor(value, dtype=parameter.dtype, device=parameter.device))
    hmm.ar.zero_()
    return report, weights


class HMMConvergenceError(RuntimeError):
    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


@torch.no_grad()
def expectation(model: ARHMM, samples: torch.Tensor,
                batch_size: int) -> tuple[float, torch.Tensor, torch.Tensor]:
    """All soft state weights and expected transitions; never join sample ends."""
    gamma, transitions = [], torch.zeros_like(model.transition_logits[0])
    likelihood = samples.new_zeros(())
    for batch in samples.split(batch_size):
        posterior = model(batch)
        likelihood += posterior["log_likelihood"].sum()
        gamma.append(posterior["gamma"][:, 0])
        transitions += posterior["xi"][:, 0].sum(dim=(0, 1))
    return float(likelihood), torch.cat(gamma), transitions


def emission_statistics(samples: torch.Tensor, gamma: torch.Tensor, ar_order: int = 1) -> dict:
    """Exact additive Gaussian sufficient statistics for a fixed soft E-step."""
    if ar_order == 2:
        from ar2_emissions import ar2_statistics
        return ar2_statistics(samples, gamma)
    first, previous, current = samples[:, 0], samples[:, :-1], samples[:, 1:]
    w0, w1 = gamma[:, 0], gamma[:, 1:]
    return {
        "n0": w0.sum(0), "n1": w1.sum((0, 1)),
        "s0": torch.einsum("nk,nd->kd", w0, first),
        "s00": torch.einsum("nk,ni,nj->kij", w0, first, first),
        "sx": torch.einsum("ntk,ntd->kd", w1, previous),
        "sy": torch.einsum("ntk,ntd->kd", w1, current),
        "sxx": torch.einsum("ntk,nti,ntj->kij", w1, previous, previous),
        "syy": torch.einsum("ntk,nti,ntj->kij", w1, current, current),
        "syx": torch.einsum("ntk,nti,ntj->kij", w1, current, previous),
    }


def residual_scatter(stats: dict, mean: torch.Tensor, ar: torch.Tensor) -> torch.Tensor:
    """Sum gamma_t * residual_t residual_t^T, including the distinct t=0 law."""
    outer = torch.outer
    first = (stats["s00"] - outer(stats["s0"], mean) - outer(mean, stats["s0"])
             + stats["n0"] * outer(mean, mean))
    intercept = mean - ar @ mean
    residual_sum = stats["sy"] - ar @ stats["sx"]
    later = (stats["syy"] - ar @ stats["syx"].T - stats["syx"] @ ar.T
             + ar @ stats["sxx"] @ ar.T - outer(residual_sum, intercept)
             - outer(intercept, residual_sum) + stats["n1"] * outer(intercept, intercept))
    scatter = first + later
    return (scatter + scatter.T) / 2


def emission_objective(stats: dict, mean: torch.Tensor, ar: torch.Tensor,
                       covariance: torch.Tensor) -> torch.Tensor:
    """Complete-data expected log density, excluding the constant log(2*pi)."""
    scatter = residual_scatter(stats, mean, ar)
    factor = torch.linalg.cholesky(covariance)
    logdet = 2 * factor.diagonal().log().sum()
    return -0.5 * ((stats["n0"] + stats["n1"]) * logdet
                   + torch.cholesky_solve(scatter, factor).trace())


def covariance_update(scatter: torch.Tensor, count: torch.Tensor,
                      floor: float, covariance_type: str = "full") -> torch.Tensor:
    if covariance_type == "diag":
        return torch.diag(scatter.diagonal().div(count).clamp_min(floor + 1e-10))
    if covariance_type != "full":
        raise ValueError("Use full or diag emission covariance")
    # The full eigenbasis is retained. This is the constrained covariance M-step,
    # not diagonalization of the emission model. The margin permits a Cholesky
    # factor of Sigma - floor*I in the model's existing parameterization.
    values, vectors = torch.linalg.eigh(scatter / count)
    values = values.clamp_min(floor + 1e-10)
    return (vectors * values) @ vectors.T


def later_regression_proposal(stats: dict, floor: float,
                              covariance_type: str) -> tuple | None:
    """Propose tied-mu parameters without dropping any first-position mass.

    The later-only regression is a feasible proposal, not a replacement
    objective. Its covariance and acceptance objective include all original
    first-position sufficient statistics. The caller accepts it only if the
    complete original objective does not decrease, then continues exact CM.
    """
    if stats["n1"] <= 0:
        return None
    try:
        xbar, ybar = stats["sx"] / stats["n1"], stats["sy"] / stats["n1"]
        xx = stats["sxx"] - torch.outer(stats["sx"], xbar)
        yx = stats["syx"] - torch.outer(stats["sy"], xbar)
        ar = torch.linalg.solve((xx + xx.T) / 2, yx.T).T
        identity = torch.eye(len(xbar), dtype=xbar.dtype, device=xbar.device)
        mean = torch.linalg.solve(identity - ar, ybar - ar @ xbar)
        covariance = covariance_update(residual_scatter(stats, mean, ar),
                                       stats["n0"] + stats["n1"], floor,
                                       covariance_type)
        objective = emission_objective(stats, mean, ar, covariance)
    except torch.linalg.LinAlgError:
        return None
    if not all(torch.isfinite(value).all() for value in (mean, ar, covariance, objective)):
        return None
    return mean, ar, covariance, objective


@torch.no_grad()
def maximize_emissions(model: ARHMM, stats: dict,
                       max_iterations: int, tolerance: float) -> dict:
    """Conditional maximization of mu, full B and full Sigma to convergence.

    The same mu governs x_0 and (I-B)mu at later times. An independent intercept
    regression would change that model and is deliberately not used here.
    """
    if model.ar_order == 2:
        from ar2_emissions import maximize_ar2_emissions
        def update_covariance(scatter, count, floor):
            return covariance_update(scatter, count, floor, model.covariance_type)
        return maximize_ar2_emissions(model, stats, max_iterations, tolerance, update_covariance)
    identity = torch.eye(model.channels, dtype=model.means.dtype, device=model.means.device)
    iterations, gains, zero_first_mass_states = [], [], []
    proposal_attempted, proposal_accepted = [], []
    for state in range(model.states):
        s = {name: value[state] for name, value in stats.items()}
        count = s["n0"] + s["n1"]
        if not torch.isfinite(count) or count <= 0:
            raise RuntimeError(f"State {state} has no finite posterior mass; no state is removed")
        mean, ar = model.means[0, state].clone(), model.ar[0, state].clone()
        covariance = model.covariance()[0, state].clone()
        before = emission_objective(s, mean, ar, covariance)
        if (model.covariance_type == "diag" and model.ar_order == 1
                and 0 < s["n0"] <= torch.finfo(mean.dtype).eps * count):
            # This threshold only selects an optional numerical proposal. Every
            # nonzero weight remains in the objective and all subsequent CM
            # updates, regardless of whether the proposal is accepted.
            proposal_attempted.append(state)
            proposal = later_regression_proposal(s, model.covariance_floor,
                                                  model.covariance_type)
            if proposal is not None and proposal[-1] >= before:
                mean, ar, covariance, before = proposal
                proposal_accepted.append(state)
        if model.ar_order and s["n0"] == 0:
            zero_first_mass_states.append(state)
        for iteration in range(1, max_iterations + 1):
            if model.ar_order and s["n0"] == 0:
                # With exactly zero first-position mass, the full weighted
                # regression optimum is feasible in the tied-mu model when
                # I-B is invertible. Recover mu exactly; no free intercept is
                # added to the model and no positive mass is discarded.
                xbar, ybar = s["sx"] / s["n1"], s["sy"] / s["n1"]
                centered_xx = s["sxx"] - torch.outer(s["sx"], xbar)
                centered_yx = s["syx"] - torch.outer(s["sy"], xbar)
                ar = torch.linalg.solve((centered_xx + centered_xx.T) / 2, centered_yx.T).T
                mean = torch.linalg.solve(identity - ar, ybar - ar @ xbar)
            elif model.ar_order:
                inverse = torch.cholesky_inverse(torch.linalg.cholesky(covariance))
                coefficient = identity - ar
                hessian = s["n0"] * inverse + s["n1"] * coefficient.T @ inverse @ coefficient
                rhs = inverse @ s["s0"] + coefficient.T @ inverse @ (s["sy"] - ar @ s["sx"])
                mean = torch.linalg.solve((hessian + hessian.T) / 2, rhs)
                centered_xx = (s["sxx"] - torch.outer(s["sx"], mean)
                               - torch.outer(mean, s["sx"]) + s["n1"] * torch.outer(mean, mean))
                centered_yx = (s["syx"] - torch.outer(s["sy"], mean)
                               - torch.outer(mean, s["sx"]) + s["n1"] * torch.outer(mean, mean))
                ar = torch.linalg.solve((centered_xx + centered_xx.T) / 2, centered_yx.T).T
            else:
                mean = (s["s0"] + s["sy"]) / count
            covariance = covariance_update(residual_scatter(s, mean, ar), count,
                                           model.covariance_floor, model.covariance_type)
            after = emission_objective(s, mean, ar, covariance)
            gain = float(after - before)
            if not torch.isfinite(after) or gain < -1e-7:
                raise RuntimeError(
                    f"Emission CM objective decreased or is non-finite: state={state}, gain={gain}, "
                    f"first_mass={float(s['n0']):.12g}, later_mass={float(s['n1']):.12g}, "
                    f"mean_norm={float(mean.norm()):.12g}")
            if not model.ar_order or abs(gain) <= tolerance:
                break
            before = after
        else:
            # Authorized amendment: a finite capped CM update may enter the next E-step.
            # The original per-iteration objective and outer observed-LL guards remain.
            if not all(torch.isfinite(v).all() for v in (mean, ar, covariance, after)):
                raise RuntimeError("Non-finite capped CM state")
            torch.linalg.cholesky(covariance)
            if float(torch.linalg.eigvalsh(covariance).min()) <= 0:
                raise RuntimeError("Invalid capped CM covariance")
        model.means[0, state].copy_(mean)
        model.ar[0, state].copy_(ar)
        model.covariance_factor[0, state].copy_(
            torch.linalg.cholesky(covariance - model.covariance_floor * identity))
        iterations.append(iteration)
        gains.append(gain)
    report = {"iterations_per_state": iterations, "last_objective_gain_per_state": gains,
              "zero_first_mass_states": zero_first_mass_states,
              "cm_cap_policy": "accept_finite_update_continue_outer_EM",
              "cap_accepted_states": [i for i, (n, g) in enumerate(zip(iterations, gains))
                                      if n == max_iterations and abs(g) > tolerance],
              "converged_per_state": [abs(g) <= tolerance or not model.ar_order for g in gains]}
    if model.covariance_type == "diag":
        report.update(joint_regression_attempted_states=proposal_attempted,
                      joint_regression_accepted_states=proposal_accepted)
    return report


@torch.no_grad()
def _fit_shared_hmm_once(hmm: SharedARHMM, train_x: np.ndarray, seed: int,
                   max_iterations: int = 5000, tolerance: float = 1e-3, batch_size: int = 128,
                   cm_max_iterations: int = 10000, cm_tolerance: float = 1e-8,
                   progress: Callable[[dict], None] | None = None, *,
                   initialization: str = "kmeans", gmm_max_iterations: int = 5000,
                   gmm_tolerance: float = 1e-8, accept_gmm_max_iterations: bool = False,
                   gmm_n_init: int = 1, gmm_init_params: str = "random_from_data",
                   gmm_reg_covar: float | None = None) -> dict:
    """Fit sum_i log p(X_i) on all training sequences, without label arguments.

    All observations and exact soft posteriors enter each EM/ECM update. Fitting
    and the convergence gate use only inputs; no class priors or pseudo-labels.
    Parameters are committed and frozen only after convergence.
    """
    if min(max_iterations, batch_size, cm_max_iterations) < 1 or min(tolerance, cm_tolerance) <= 0:
        raise ValueError("Positive EM/CM iteration budgets, batch size and tolerances required")
    device = hmm.means.device
    fitted = SharedARHMM(hmm.states, hmm.channels, hmm.ar_order, hmm.covariance_floor, hmm.covariance_type).to(
        device=device, dtype=torch.float64)
    samples = torch.as_tensor(train_x, dtype=torch.float64, device=device)
    report = {"method": "exact_EM_ECM", "hmm_kind": "shared", "uses_labels": False,
              "objective": "sum_i log p(X_i)", "seed": seed, "converged": False,
              "samples": len(samples), "sequence_length": samples.shape[1], "states": hmm.states,
              "ar_order": hmm.ar_order, "covariance_floor": hmm.covariance_floor,
              "covariance_type": hmm.covariance_type,
              "max_iterations": max_iterations, "absolute_log_likelihood_tolerance": tolerance,
              "required_stable_iterations": 3, "cm_max_iterations": cm_max_iterations,
              "absolute_cm_objective_tolerance": cm_tolerance, "fit_dtype": "float64",
              "inference_dtype": str(hmm.means.dtype), "initialization_method": initialization, "history": []}
    stable = 0
    try:
        if initialization == "kmeans":
            fitted.initialize(train_x, seed)
        else:
            initial, weights = initialize_shared_hmm(fitted, train_x, seed, initialization,
                                                      gmm_max_iterations, gmm_tolerance, accept_gmm_max_iterations,
                                                      gmm_n_init=gmm_n_init, gmm_init_params=gmm_init_params,
                                                      gmm_reg_covar=gmm_reg_covar)
            report["initialization"] = initial
            if weights is not None:
                # Fit the initializer's state weights to the actual tied-mu
                # AR model. This is not independent-intercept regression.
                initial["ar_cm"] = maximize_emissions(
                    fitted, emission_statistics(samples, torch.as_tensor(weights, device=device), fitted.ar_order),
                    cm_max_iterations, cm_tolerance)
                del weights
        ll, gamma, transitions = expectation(fitted, samples, batch_size)
        report["initial_log_likelihood"] = ll
        for iteration in range(1, max_iterations + 1):
            start = gamma[:, 0].sum(0)
            if (transitions.sum(-1) <= 0).any():
                raise RuntimeError("A state has no expected outgoing transitions; no fallback or state pruning")
            fitted.start_logits[0].copy_((start / start.sum()).log())
            fitted.transition_logits[0].copy_((transitions / transitions.sum(-1, keepdim=True)).log())
            cm = maximize_emissions(fitted, emission_statistics(samples, gamma, fitted.ar_order), cm_max_iterations, cm_tolerance)
            updated_ll, gamma, transitions = expectation(fitted, samples, batch_size)
            gain = updated_ll - ll
            if not np.isfinite(updated_ll) or gain < -1e-7:
                raise RuntimeError(f"Observed training likelihood decreased or is non-finite: gain={gain}")
            stable = stable + 1 if abs(gain) <= tolerance else 0
            row = {"iteration": iteration, "log_likelihood": updated_ll, "gain": gain, **cm}
            report["history"].append(row)
            ll = updated_ll
            if progress is not None:
                progress(row)
            if stable >= 3:
                report["converged"] = True
                break
        if not report["converged"]:
            raise RuntimeError(f"Shared HMM did not converge in {max_iterations} EM iterations")
    except RuntimeError as error:
        report["failure"] = str(error)
        raise HMMConvergenceError(str(error), report) from error
    report["iterations"] = iteration
    report["final_log_likelihood"] = ll
    hmm.load_state_dict(fitted.state_dict())
    hmm.requires_grad_(False)
    for parameter in hmm.parameters():
        parameter.grad = None
    hmm.eval()
    return report


def _random_restart_seeds(restarts: int) -> list[int]:
    """Draw distinct initialization seeds from the operating system RNG."""
    seeds, used = [], set()
    while len(seeds) < restarts:
        value = secrets.randbits(32)
        if value not in used:
            seeds.append(value)
            used.add(value)
    return seeds


@torch.no_grad()
def fit_shared_hmm(hmm: SharedARHMM, train_x: np.ndarray, seed: int,
                   max_iterations: int = 5000, tolerance: float = 1e-3, batch_size: int = 128,
                   cm_max_iterations: int = 10000, cm_tolerance: float = 1e-8,
                   progress: Callable[[dict], None] | None = None, *,
                   initialization: str = "kmeans", restarts: int = 1,
                   gmm_max_iterations: int = 5000, gmm_tolerance: float = 1e-8, accept_gmm_max_iterations: bool = False,
                   restart_callback: Callable[[int, SharedARHMM, dict], None] | None = None,
                   restart_seeds: Sequence[int] | None = None,
                   gmm_n_init: int = 1, gmm_init_params: str = "random_from_data",
                   gmm_reg_covar: float | None = None) -> dict:
    """Converge every requested restart; select only by training input likelihood."""
    if restarts < 1 or initialization not in (
            "kmeans", "gmm", "random", "temporal_gmm",
            "temporal_gmm_soft_full", "temporal_gmm_hard_full"):
        raise ValueError("Positive restart count and a supported initializer required")
    options = dict(max_iterations=max_iterations, tolerance=tolerance, batch_size=batch_size,
                   cm_max_iterations=cm_max_iterations, cm_tolerance=cm_tolerance,
                   initialization=initialization, gmm_max_iterations=gmm_max_iterations,
                   gmm_tolerance=gmm_tolerance, accept_gmm_max_iterations=accept_gmm_max_iterations,
                   gmm_n_init=gmm_n_init, gmm_init_params=gmm_init_params, gmm_reg_covar=gmm_reg_covar)
    if restarts == 1:
        if restart_seeds is not None:
            raise ValueError("Explicit restart seeds require multiple restarts")
        return _fit_shared_hmm_once(hmm, train_x, seed, progress=progress, **options)
    seed_source = "os_random_unique" if restart_seeds is None else "provided_unique"
    restart_seeds = (_random_restart_seeds(restarts) if restart_seeds is None
                     else list(restart_seeds))
    if (len(restart_seeds) != restarts or len(set(restart_seeds)) != restarts or
            any(type(value) is not int or not 0 <= value < 2**32 for value in restart_seeds)):
        raise ValueError("Restart seeds must be distinct 32-bit nonnegative integers")
    histories, best_state, best_index, best_likelihood = [], None, None, -float("inf")
    for index in range(restarts):
        restart_seed = restart_seeds[index]
        candidate = SharedARHMM(hmm.states, hmm.channels, hmm.ar_order, hmm.covariance_floor, hmm.covariance_type).to(
            device=hmm.means.device, dtype=hmm.means.dtype)

        def restart_progress(row):
            if progress is not None:
                progress({**row, "restart": index + 1, "restarts": restarts,
                          "initialization_seed": restart_seed})

        try:
            report = _fit_shared_hmm_once(candidate, train_x, restart_seed,
                                          progress=restart_progress, **options)
        except HMMConvergenceError as error:
            histories.append(error.report)
            failure = {"converged": False, "hmm_kind": "shared", "uses_labels": False,
                       "seed": seed, "initialization_method": initialization, "restarts_requested": restarts,
                       "restart_seed_source": seed_source, "initialization_seeds": restart_seeds,
                       "restart_reports": histories, "failure": f"Restart {index + 1}: {error}"}
            raise HMMConvergenceError(failure["failure"], failure) from error
        histories.append(report)
        if restart_callback is not None:
            restart_callback(index + 1, candidate, report)
        if report["final_log_likelihood"] > best_likelihood:
            best_index, best_likelihood = index, report["final_log_likelihood"]
            best_state = {name: value.detach().clone() for name, value in candidate.state_dict().items()}
    hmm.load_state_dict(best_state)
    hmm.requires_grad_(False)
    for parameter in hmm.parameters():
        parameter.grad = None
    hmm.eval()
    return {**histories[best_index], "seed": seed, "restarts_requested": restarts,
            "restart_seed_source": seed_source, "initialization_seeds": restart_seeds,
            "selected_restart": best_index + 1, "selected_initialization_seed": histories[best_index]["seed"],
            "selection": "maximum final float64 training-input log likelihood among all converged restarts",
            "restart_reports": histories}
