"""Electricity price samplers for MDP models."""

from abc import ABC, abstractmethod
from math import erf as _erf, sqrt as _sqrt
from typing import Callable, Literal

import numpy as np
import pandas as pd


Season = Literal["spring", "summer", "autumn", "winter"]

SEASONS: list[Season] = ["spring", "summer", "autumn", "winter"]

_SEASON_IDX: dict[str, int] = {s: i for i, s in enumerate(SEASONS)}


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _as_rng(rng: "np.random.Generator | None") -> np.random.Generator:
    """Return the given Generator, or a fresh seeded-from-entropy one if None.

    Passing an explicit Generator makes draws reproducible and lets callers share
    a stream across samplers (common random numbers); None preserves the old
    non-reproducible behaviour for incidental callers.
    """
    return rng if rng is not None else np.random.default_rng()


def _gaussian_cdf(x: float, mean: float, std: float) -> float:
    if std <= 0:
        return 1.0 if x > mean else 0.0
    return 0.5 * (1.0 + _erf((x - mean) / (std * _sqrt(2.0))))


def _gmm_bin_probs(
    weights: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    params,
) -> np.ndarray:
    """Bin probability vector for a Gaussian mixture with given parameters."""
    delta = params.lambda_max / params.K
    edges = [j * delta for j in range(params.K + 1)]
    probs = np.zeros(params.K)
    for w, mu, sigma in zip(weights, means, stds):
        component = np.empty(params.K)
        component[0] = _gaussian_cdf(edges[1], mu, sigma)
        for j in range(1, params.K - 1):
            component[j] = _gaussian_cdf(edges[j + 1], mu, sigma) - _gaussian_cdf(edges[j], mu, sigma)
        component[-1] = 1.0 - _gaussian_cdf(edges[-2], mu, sigma)
        probs += w * component
    return probs


def _encode_context(is_weekend: bool, hour: int, season: Season) -> np.ndarray:
    """7-dim feature vector: [is_weekend, sin_hour, cos_hour, season_onehot(4)]."""
    angle = 2 * np.pi * hour / 24
    onehot = np.zeros(4)
    onehot[_SEASON_IDX[season]] = 1.0
    return np.array([float(is_weekend), np.sin(angle), np.cos(angle), *onehot], dtype=np.float32)


# ── Abstract base ──────────────────────────────────────────────────────────────

class AbstractSampler(ABC):
    """Base class for electricity price samplers."""

    @abstractmethod
    def fit(self, df: pd.DataFrame, _progress=None) -> "AbstractSampler":
        """Fit the sampler to a preprocessed price DataFrame (from entsoe_loader).

        _progress: optional callable(fraction: float, message: str) called periodically.
        """

    @abstractmethod
    def sample(self, dow: int, hour: int, season: Season,
               rng: "np.random.Generator | None" = None) -> float:
        """Sample a price in EUR/kWh given day-of-week, hour, and season.

        rng: optional NumPy Generator for reproducible draws; None uses a fresh one.
        """

    @abstractmethod
    def bin_probs(self, dow: int, hour: int, season: Season, params) -> np.ndarray:
        """
        Return a (K,) probability vector over price bins for the given context.

        Bins are defined by params.K and params.lambda_max (identical to model_utils).
        """


# ── Per-bin models ─────────────────────────────────────────────────────────────

class GaussianBinnedSampler(AbstractSampler):
    """Fits a Gaussian N(μ, σ²) per (is_weekend, hour, season) bin."""

    def __init__(self) -> None:
        self._params: dict[tuple, tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame, _progress=None) -> "GaussianBinnedSampler":
        groups = list(df.groupby(["is_weekend", "hour", "season"]))
        n = len(groups)
        for i, (key, group) in enumerate(groups):
            prices = group["price_eur_kwh"].to_numpy(dtype=float)
            prices = prices[~np.isnan(prices)]
            mean = float(np.mean(prices))
            std  = float(np.std(prices, ddof=1)) if len(prices) > 1 else 0.0
            self._params[key] = (mean, std)
            if _progress is not None:
                _progress((i + 1) / n, f"Bin {i + 1}/{n}")
        return self

    def sample(self, dow: int, hour: int, season: Season,
               rng: "np.random.Generator | None" = None) -> float:
        mean, std = self._get_params(dow, hour, season)
        return float(max(0.0, _as_rng(rng).normal(mean, std)))

    def bin_probs(self, dow: int, hour: int, season: Season, params) -> np.ndarray:
        mean, std = self._get_params(dow, hour, season)
        return _gmm_bin_probs(np.array([1.0]), np.array([mean]), np.array([std]), params)

    def bin_stats(self, dow: int, hour: int, season: Season) -> tuple[float, float]:
        return self._get_params(dow, hour, season)

    def _get_params(self, dow: int, hour: int, season: Season) -> tuple[float, float]:
        key = (dow >= 5, hour, season)
        p = self._params.get(key)
        if p is None:
            raise KeyError(f"No data for bin {key}")
        return p


class GMMSampler(AbstractSampler):
    """
    Fits a Gaussian Mixture Model per (is_weekend, hour, season) bin.

    Uses scikit-learn GaussianMixture.  bin_probs is the weighted sum of
    Gaussian CDFs — analytically exact.
    """

    def __init__(self, n_components: int = 3) -> None:
        self.n_components = n_components
        self._gmms: dict[tuple, object] = {}

    def fit(self, df: pd.DataFrame, _progress=None) -> "GMMSampler":
        from sklearn.mixture import GaussianMixture
        groups = list(df.groupby(["is_weekend", "hour", "season"]))
        n_groups = len(groups)
        for i, (key, group) in enumerate(groups):
            prices = group["price_eur_kwh"].to_numpy(dtype=float).reshape(-1, 1)
            prices = prices[~np.isnan(prices[:, 0])]
            n = min(self.n_components, len(prices))
            gmm = GaussianMixture(n_components=n, covariance_type="full", random_state=0)
            gmm.fit(prices)
            self._gmms[key] = gmm
            if _progress is not None:
                _progress((i + 1) / n_groups, f"Bin {i + 1}/{n_groups}")
        return self

    def sample(self, dow: int, hour: int, season: Season,
               rng: "np.random.Generator | None" = None) -> float:
        # Draw from the mixture explicitly with the supplied Generator.
        # GaussianMixture.sample() is avoided: it reseeds from the fixed
        # random_state every call, so it returns the same value each time.
        gmm = self._get_gmm(dow, hour, season)
        r       = _as_rng(rng)
        weights = gmm.weights_
        means   = gmm.means_[:, 0]
        stds    = np.sqrt(gmm.covariances_[:, 0, 0])
        k = int(r.choice(len(weights), p=weights))
        return float(max(0.0, r.normal(means[k], stds[k])))

    def bin_probs(self, dow: int, hour: int, season: Season, params) -> np.ndarray:
        gmm = self._get_gmm(dow, hour, season)
        weights = gmm.weights_
        means   = gmm.means_[:, 0]
        stds    = np.sqrt(gmm.covariances_[:, 0, 0])
        return _gmm_bin_probs(weights, means, stds, params)

    def _get_gmm(self, dow: int, hour: int, season: Season):
        key = (dow >= 5, hour, season)
        gmm = self._gmms.get(key)
        if gmm is None:
            raise KeyError(f"No GMM for bin {key}")
        return gmm


# ── Global model ───────────────────────────────────────────────────────────────

class MDNSampler(AbstractSampler):
    """
    Mixture Density Network: one model trained on all data jointly.

    Input  : 7-dim context vector (is_weekend, sin_hour, cos_hour, season_onehot(4))
    Output : parameters of a K-component Gaussian mixture (π, μ, σ)

    Training minimises negative log-likelihood over all observations.
    """

    def __init__(
        self,
        n_components: int = 3,
        hidden_dims: list[int] | None = None,
        epochs: int = 200,
        batch_size: int = 1024,
        lr: float = 1e-3,
    ) -> None:
        self.n_components = n_components
        self.hidden_dims  = hidden_dims or [64, 64]
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self._net         = None

    def fit(self, df: pd.DataFrame, _progress=None, _history=None) -> "MDNSampler":
        """Fit the MDN. Pass a list as ``_history`` to record per-epoch training
        metrics (loss, original-space NLL, mean mixture weights) for plotting."""
        import torch
        import torch.nn as nn

        torch.set_num_threads(1)

        prices = df["price_eur_kwh"].to_numpy(dtype=np.float32)
        mask   = ~np.isnan(prices)
        prices = prices[mask]

        # Build feature matrix (N, 7) — vectorised, avoids slow itertuples loop
        rows = df[mask].reset_index(drop=True)
        if _progress is not None:
            _progress(0.0, "Building feature matrix…")
        angle      = (2 * np.pi * rows["hour"].to_numpy(dtype=np.float32)) / 24
        is_we      = rows["is_weekend"].to_numpy(dtype=np.float32)
        season_idx = rows["season"].map(_SEASON_IDX).to_numpy(dtype=np.int32)
        onehot     = np.zeros((len(rows), 4), dtype=np.float32)
        onehot[np.arange(len(rows)), season_idx] = 1.0
        X = np.column_stack([is_we, np.sin(angle), np.cos(angle), onehot])

        # Standardise target so network sigma heads initialise near the right scale
        self._price_mean = float(prices.mean())
        self._price_std  = float(prices.std()) or 1.0
        prices_norm = (prices - self._price_mean) / self._price_std

        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(prices_norm).unsqueeze(1)

        self._net = _MDNNet(
            in_dim=7,
            hidden_dims=self.hidden_dims,
            n_components=self.n_components,
        )
        optimiser = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        N = len(prices)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=10, factor=0.5, min_lr=1e-5
        )

        self._net.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, N, self.batch_size):
                idx = perm[start:start + self.batch_size]
                pi, mu, sigma = self._net(X_t[idx])
                loss = -_mdn_log_prob(pi, mu, sigma, y_t[idx]).mean()
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                optimiser.step()
                epoch_loss += loss.item()
                n_batches += 1
            epoch_loss /= n_batches
            scheduler.step(epoch_loss)
            if _history is not None:
                with torch.no_grad():
                    pi_all, _, _ = self._net(X_t[:2048])
                    mean_weights = pi_all.mean(dim=0)
                row = {
                    "step": epoch + 1,
                    "loss": epoch_loss,
                    "loss_original_space": epoch_loss + np.log(self._price_std),
                }
                for k, w in enumerate(mean_weights.tolist()):
                    row[f"pi_{k}"] = w
                _history.append(row)
            if _progress is not None and (epoch % 5 == 0 or epoch == self.epochs - 1):
                _progress((epoch + 1) / self.epochs,
                          f"Epoch {epoch + 1}/{self.epochs}  loss {epoch_loss:.4f}")

        self._net.eval()
        return self

    def sample(self, dow: int, hour: int, season: Season,
               rng: "np.random.Generator | None" = None) -> float:
        pi, mu, sigma = self._predict(dow, hour, season)
        r = _as_rng(rng)
        k = int(r.choice(len(pi), p=pi))
        return float(max(0.0, r.normal(mu[k], sigma[k])))

    def bin_probs(self, dow: int, hour: int, season: Season, params) -> np.ndarray:
        pi, mu, sigma = self._predict(dow, hour, season)
        return _gmm_bin_probs(pi, mu, sigma, params)

    def _predict(self, dow: int, hour: int, season: Season) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch
        if self._net is None:
            raise RuntimeError("MDNSampler not fitted — call fit() first")
        x = torch.from_numpy(_encode_context(dow >= 5, hour, season)).unsqueeze(0)
        with torch.no_grad():
            pi, mu, sigma = self._net(x)
        mu_np    = mu[0].numpy()    * self._price_std + self._price_mean
        sigma_np = sigma[0].numpy() * self._price_std
        return pi[0].numpy(), mu_np, sigma_np


class _MDNNet:
    """Thin wrapper around a PyTorch module to avoid exposing torch at the module level."""

    def __init__(self, in_dim: int, hidden_dims: list[int], n_components: int) -> None:
        import torch.nn as nn
        K = n_components
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        self._trunk = nn.Sequential(*layers)
        self._pi_head    = nn.Linear(prev, K)
        self._mu_head    = nn.Linear(prev, K)
        self._sigma_head = nn.Linear(prev, K)
        # collect all parameters for the optimiser
        self._modules = nn.ModuleList([self._trunk, self._pi_head, self._mu_head, self._sigma_head])

    def __call__(self, x):
        import torch
        import torch.nn.functional as F
        h     = self._trunk(x)
        pi    = F.softmax(self._pi_head(h), dim=-1)
        mu    = self._mu_head(h)
        sigma = F.softplus(self._sigma_head(h)) + 1e-4
        return pi, mu, sigma

    def parameters(self):
        return self._modules.parameters()

    def train(self):
        self._modules.train()

    def eval(self):
        self._modules.eval()


def _mdn_log_prob(pi, mu, sigma, y):
    """Log-likelihood of y under the Gaussian mixture (pi, mu, sigma). Shape: (batch,)."""
    import torch
    import torch.distributions as D
    # y: (B, 1), mu/sigma: (B, K)
    dist = D.Normal(mu, sigma)
    log_p = dist.log_prob(y)          # (B, K)
    log_pi = torch.log(pi + 1e-8)     # (B, K)
    return torch.logsumexp(log_p + log_pi, dim=-1)  # (B,)


# ── Factory ────────────────────────────────────────────────────────────────────

def make_price_bin_probs_fn(
    sampler: AbstractSampler,
    params,
    season: Season,
    is_weekend: bool,
) -> Callable[[int], np.ndarray]:
    """
    Wrap a fitted sampler as the price_bin_probs_fn expected by backward induction.

    Parameters
    ----------
    sampler    : fitted AbstractSampler
    params     : model params (needs K, lambda_max)
    season     : season for all bin lookups
    is_weekend : whether the horizon is a weekend (True) or weekday (False)
    """
    _dow = 5 if is_weekend else 0
    # Pre-compute all 24 hourly bin-prob vectors so repeated calls (e.g. from
    # the backward-induction precomputation loop) are O(1) dict lookups.
    _cache = {h: sampler.bin_probs(_dow, h, season, params) for h in range(24)}

    def fn(t: int) -> np.ndarray:
        return _cache[(t // 60) % 24]

    return fn
