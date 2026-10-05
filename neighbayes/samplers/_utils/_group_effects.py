r"""Group fixed effects integrated out of a Gaussian working likelihood.

The NB count samplers reduce each sweep, through the Pólya–Gamma
augmentation, to a Gaussian working model

.. math::

    z = U\beta + D c + e, \qquad e \sim N(0, \Omega^{-1}),

with ``Ω = diag(ω)`` and ``D`` the one-hot map from working rows to groups
(spatial units or origin–destination pairs).  Under a Normal prior
``c_g ~ N(m, s²)`` the group block is integrated out exactly.  Its precision
is diagonal, ``d_g = Σ_{r∈g} ω_r + 1/s²``, so the Schur complement is a
per-group, ω-weighted within transform with prior shrinkage:

.. math::

    \langle x, y \rangle_{\tilde\Omega}
    = \sum_r \omega_r x_r y_r - \sum_g \frac{S(x)_g\, S(y)_g}{d_g},
    \qquad S(x)_g = \sum_{r \in g} \omega_r x_r.

Every Gram matrix, cross product and quadratic form a sampler forms with
``Ω`` is replaced by its ``Ω̃`` version.  The one-hot ``D`` is never built:
group sums are ``np.bincount`` over a row→group index, so groups may have
any sizes.

Quantities passed to :meth:`GroupProjector.gram`, :meth:`~GroupProjector.cross`
and :meth:`~GroupProjector.quad` are *centered*: subtract ``spec.mu`` from the
working response first, so the group block has prior mean zero.
:meth:`~GroupProjector.draw` takes the raw residual ``z − Uβ`` and returns
``c`` on its own scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroupEffects:
    """Row→group index and Normal prior of a group fixed-effect block.

    Parameters
    ----------
    row_group : ndarray of int, shape (n_rows,)
        Group of each working row, in ``[0, n_groups)``.
    n_groups : int
        Number of groups.
    mu, sigma : float
        Prior ``c_g ~ N(mu, sigma²)``.
    sigma_scale : float, optional
        When given, ``sigma`` is a starting value and the sd is learned with a
        half-Student-t(3, ``sigma_scale``) prior — partially pooled (random)
        effects instead of effects under a fixed prior.
    """

    row_group: np.ndarray
    n_groups: int
    mu: float
    sigma: float
    sigma_scale: float | None = None

    def __post_init__(self):
        g = np.asarray(self.row_group)
        if g.ndim != 1 or not np.issubdtype(g.dtype, np.integer):
            raise ValueError("row_group must be a 1-D integer array.")
        if g.size and (g.min() < 0 or g.max() >= self.n_groups):
            raise ValueError("row_group entries must lie in [0, n_groups).")
        if not self.sigma > 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}.")


#: Degrees of freedom of the half-Student-t prior on a learned effect sd.
SD_PRIOR_NU = 3.0


def _slice_log_sd(log_density, sigma: float, scale: float, w: float, rng) -> float:
    """One slice draw of ``log σ`` on ``[log A − 10, log A + 5]``, returned as σ.

    The support is centred on the prior scale ``A``: a fixed ``[−10, 5]`` caps
    σ at e⁵ ≈ 148 whatever the outcome's units, which truncates a Gaussian
    effect sd whose prior scale is ``sd(y)``.  At ``A = 1`` nothing changes.
    """
    from ._slice import slice_sample_1d

    c = float(np.log(scale))
    lo, hi = c - 10.0, c + 5.0
    new, _ = slice_sample_1d(
        log_density,
        float(np.clip(np.log(sigma), lo, hi)),
        lower=lo,
        upper=hi,
        w=w,
        rng=rng,
    )
    return float(np.exp(new))


def sample_effect_sd(
    sigma: float, dev2: float, n_groups: int, scale: float, rng
) -> float:
    r"""Draw an effect sd ``σ | c`` under a half-t(3, ``scale``) prior.

    ``dev2 = Σ_g (c_g − μ)²``.  A slice on ``log σ`` of
    ``−G log σ − dev2 / (2σ²) + log p(σ) + log σ``.  Valid as a Gibbs step after
    the effects are drawn: the filter and coefficient steps condition on σ.
    """
    nu, s = SD_PRIOR_NU, float(scale)

    def log_density(ls):
        v = np.exp(2.0 * ls)
        return (
            -n_groups * ls
            - 0.5 * dev2 / v
            - 0.5 * (nu + 1.0) * np.log1p(v / (nu * s * s))
            + ls
        )

    return _slice_log_sd(log_density, sigma, s, 0.5, rng)


def noncentered_effect_sd(
    sigma: float, P: float, h: float, scale: float, rng, nu: float = SD_PRIOR_NU
) -> float:
    r"""Non-centred draw of an effect sd: ``σ | c̃``, the interweaving half.

    With the standardized effects ``c̃ = (c − μ)/σ`` held fixed, the group part
    of the working predictor is ``σ c̃``, so the Gaussian working likelihood is
    ``exp(−P σ²/2 + h σ)`` with ``P = Σ ω c̃_g²`` and ``h = Σ ω c̃_g r`` (``r`` the
    working response less the rest of the predictor and ``μ``).  Alternating
    this with the centred ``σ | c`` (ancillarity–sufficiency interweaving, Yu &
    Meng 2011) keeps σ mixing when each group's data say little, the usual
    large-N, small-T spatial panel.
    """
    nu, s = float(nu), float(scale)

    def log_density(ls):
        sd = np.exp(ls)
        return (
            -0.5 * P * sd * sd
            + h * sd
            - 0.5 * (nu + 1.0) * np.log1p(sd * sd / (nu * s * s))
            + ls
        )

    return _slice_log_sd(log_density, sigma, s, 0.5, rng)


def exact_noncentered_sd(sigma: float, scale: float, loglik_at, rng) -> float:
    r"""Non-centred ``σ | c̃`` on the **exact** likelihood.

    ``loglik_at(sd)`` is the data log-likelihood with the group effects at
    ``μ + sd·c̃`` and everything else held.  Unlike the working-likelihood
    step it conditions on no augmentation (Pólya–Gamma weights, missed
    zeros), whose missing information otherwise throttles σ when each group's
    data are thin; those are redrawn from their conditionals before their next
    use, so the step is valid.
    """
    nu, s = SD_PRIOR_NU, float(scale)

    def log_density(ls):
        sd = np.exp(ls)
        val = loglik_at(sd) - 0.5 * (nu + 1.0) * np.log1p(sd * sd / (nu * s * s)) + ls
        return val if np.isfinite(val) else -np.inf

    return _slice_log_sd(log_density, sigma, s, 0.3, rng)


def interweave_group_sd(groups, c, resid, omega, row_group, rng):
    """Both halves of the sd update for row-indexed group effects.

    ``resid`` is the working response less the non-group predictor, on the
    rows of ``row_group`` (with weights ``omega``).  Returns ``(groups, c)``:
    the centred draw ``σ | c``, then the non-centred ``σ | c̃`` with ``c``
    rescaled to match.
    """
    from dataclasses import replace

    if groups.sigma_scale is None:
        return groups, c
    groups = draw_group_sd(groups, c, rng)
    ct = (np.asarray(c) - groups.mu) / groups.sigma
    a = ct[row_group]
    r = np.asarray(resid) - groups.mu
    P = float(np.dot(omega, a * a))
    h = float(np.dot(omega, a * r))
    sigma = noncentered_effect_sd(groups.sigma, P, h, groups.sigma_scale, rng)
    return replace(groups, sigma=sigma), groups.mu + sigma * ct


def draw_group_sd(groups: GroupEffects, c: np.ndarray, rng) -> GroupEffects:
    """``groups`` with its sd redrawn given the effects ``c`` (unchanged if fixed)."""
    from dataclasses import replace

    if groups.sigma_scale is None:
        return groups
    dev2 = float(np.sum((np.asarray(c) - groups.mu) ** 2))
    sigma = sample_effect_sd(
        groups.sigma, dev2, groups.n_groups, groups.sigma_scale, rng
    )
    return replace(groups, sigma=sigma)


class GroupProjector:
    """The ω-weighted within transform for one draw of the working precision.

    Parameters
    ----------
    spec : GroupEffects
    omega : ndarray, shape (n_rows,)
        Working precisions of the current sweep.
    """

    def __init__(self, spec: GroupEffects, omega: np.ndarray):
        self.spec = spec
        self.omega = np.asarray(omega, dtype=np.float64)
        self._g = np.asarray(spec.row_group)
        if self.omega.shape != self._g.shape:
            raise ValueError(
                f"omega has shape {self.omega.shape}, row_group {self._g.shape}."
            )
        self.w = np.bincount(self._g, weights=self.omega, minlength=spec.n_groups)
        self.d = self.w + 1.0 / spec.sigma**2

    def sums(self, x: np.ndarray) -> np.ndarray:
        """``S(x)``: ω-weighted group sums, shape ``(n_groups,)`` or ``(n_groups, k)``."""
        x = np.asarray(x, dtype=np.float64)
        G = self.spec.n_groups
        if x.ndim == 1:
            return np.bincount(self._g, weights=self.omega * x, minlength=G)
        wx = self.omega[:, None] * x
        out = np.empty((G, x.shape[1]))
        for j in range(x.shape[1]):
            out[:, j] = np.bincount(self._g, weights=wx[:, j], minlength=G)
        return out

    def gram(self, U: np.ndarray, S_U: np.ndarray | None = None) -> np.ndarray:
        """``Uᵀ Ω̃ U``.  Pass ``S_U = sums(U)`` to reuse it."""
        S_U = self.sums(U) if S_U is None else S_U
        return U.T @ (self.omega[:, None] * U) - S_U.T @ (S_U / self.d[:, None])

    def cross(
        self, U: np.ndarray, r: np.ndarray, S_U: np.ndarray | None = None
    ) -> np.ndarray:
        """``Uᵀ Ω̃ r`` for a centered ``r``."""
        S_U = self.sums(U) if S_U is None else S_U
        return U.T @ (self.omega * r) - S_U.T @ (self.sums(r) / self.d)

    def quad(self, r: np.ndarray) -> float:
        """``rᵀ Ω̃ r`` for a centered ``r``."""
        s = self.sums(r)
        return float(np.dot(r, self.omega * r) - np.dot(s, s / self.d))

    def logdet(self) -> float:
        """``Σ_g log d_g``; constant in ρ and β given ω."""
        return float(np.sum(np.log(self.d)))

    def mean(self, resid: np.ndarray) -> np.ndarray:
        """Posterior mean of ``c`` given the raw residual ``resid = z − Uβ``."""
        m = self.spec.mu
        return m + (self.sums(resid) - m * self.w) / self.d

    def draw(self, resid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw ``c | β, ω`` given the raw residual ``resid = z − Uβ``."""
        return self.mean(resid) + rng.standard_normal(self.spec.n_groups) / np.sqrt(
            self.d
        )


# ---------------------------------------------------------------------------
# Collapsed Gaussian block shared by the count samplers
# ---------------------------------------------------------------------------


def _weighted_moments(U, r, omega, proj):
    """``(UᵀΩ̃U, UᵀΩ̃r, rᵀΩ̃r)``; plain ``Ω`` when ``proj`` is None."""
    if proj is None:
        Uw = U * omega[:, None]
        return U.T @ Uw, Uw.T @ r, float(np.dot(r, omega * r))
    S_U = proj.sums(U)
    return proj.gram(U, S_U), proj.cross(U, r, S_U), proj.quad(r)


def collapsed_log_density(
    U: np.ndarray,
    z: np.ndarray,
    omega: np.ndarray,
    mu0: np.ndarray,
    prec0: np.ndarray,
    proj: GroupProjector | None = None,
) -> float:
    r"""``log p(z | U, ω)`` with ``θ ~ N(μ₀, diag(prec₀)⁻¹)`` and ``c`` integrated out.

    Up to terms constant in ``U`` (the sampler's ρ enters only through
    ``U = A(ρ)⁻¹X``): ``−½ log|M| − ½ (rᵀΩ̃r − vᵀM⁻¹v)`` with
    ``r = z − m − Uμ₀``, ``M = diag(prec₀) + UᵀΩ̃U`` and ``v = UᵀΩ̃r``.
    Returns ``−inf`` when ``M`` is not positive definite.
    """
    m = proj.spec.mu if proj is not None else 0.0
    r = z - m - U @ mu0
    M, v, quad = _weighted_moments(U, r, omega, proj)
    M[np.diag_indices_from(M)] += prec0
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return -np.inf
    w = np.linalg.solve(L, v) if v.size else v
    val = -float(np.sum(np.log(np.diag(L)))) - 0.5 * (quad - float(w @ w))
    return val if np.isfinite(val) else -np.inf


def draw_collapsed(
    U: np.ndarray,
    z: np.ndarray,
    omega: np.ndarray,
    mu0: np.ndarray,
    prec0: np.ndarray,
    proj: GroupProjector | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Joint draw of ``(θ, c) | U, ω``: θ from the Schur Gram, then ``c | θ``.

    Returns ``(θ, c)``, with ``c = None`` when there are no group effects.
    """
    m = proj.spec.mu if proj is not None else 0.0
    G, h, _ = _weighted_moments(U, z - m, omega, proj)
    G[np.diag_indices_from(G)] += prec0
    h = h + prec0 * mu0
    L = np.linalg.cholesky(G)
    mean = np.linalg.solve(L.T, np.linalg.solve(L, h))
    theta = mean + np.linalg.solve(L.T, rng.standard_normal(mean.size))
    if proj is None:
        return theta, None
    return theta, proj.draw(z - U @ theta, rng)
