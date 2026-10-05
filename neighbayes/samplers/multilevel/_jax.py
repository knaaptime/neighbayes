"""JAX backend for the spatial multilevel Gibbs sampler.

The same sweep as :mod:`._core`, compiled: each chain runs fixed-length
``lax.scan`` chunks of sweeps (:func:`.._utils._jax_utils.run_chains_chunked`),
so no Python runs between log-density evaluations.  The linear block ``z``
uses sparsax's CHOLMOD — ``factor_solve`` (a solve and ``log|P|`` from one
factorization) inside every collapsed evaluation and ``sample_gaussian`` for
the draw — on the union pattern :class:`._core.MultilevelStructure` builds,
with ``P``'s values assembled from the same per-level polynomial terms.  The
non-centred moves solve with ``I − ρW_ℓ`` by sparsax's LU (KLU or UMFPACK,
routed by the pattern).  Slice widths adapt during warmup.

Two savings cut the factorizations per sweep (each collapsed slice evaluation
costs one):

* **The top level as a dense Schur complement.**  ``ρ_L`` and ``σ_L`` touch
  only the top block of ``P`` (``β_L, θ_L``: a few dozen rows for states).
  When that block is small, the rest of ``P`` is factored once per sweep, its
  Schur complement onto the top block formed densely, and every top-level
  evaluation becomes a small dense Cholesky; the draw of ``z`` takes the top
  block from the dense factor and the rest from the cached sparse one.
* **Cached current densities.**  Each slice starts from the density the
  previous update already computed (the ρ_0 step supplies the first), so no
  slice re-evaluates its starting point.

**Compiled once per structure.**  The sweep closes over nothing but the
model's structure — sizes, offsets, each level's process, the kind of its
log-determinant and its LU backend.  Everything that depends on the data or the
priors (``P``'s term values, ``y``, the designs, the prior scales, the ρ bounds,
each log-determinant's coefficients) reaches it as an argument.  One sweep is
kept per structure (:func:`.._utils._jax_utils.cached_sweep`), and
:func:`.._utils._jax_utils.run_chains_chunked` caches
its compiled chunk on it, so a refit — or a new dataset of the same shape, as
in a simulation study — reuses the compiled program instead of retracing.

Slice evaluations stay a hair inside each autoregressive parameter's support
(``1e-6`` of its width), because sparsax raises — it does not return ``nan``
— on a matrix that is not positive definite, and at ``ρ = ±1`` a level whose
groups have no children below can make ``P`` singular.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._core import MultilevelGibbsPriors, MultilevelStructure, initialize

_EDGE = 1e-6  # fraction of a ρ support kept clear of each end
SCHUR_MAX = 160  # largest top block handled as a dense Schur complement


@dataclass
class _Static:
    """The structure the compiled sweep closes over — nothing data-dependent."""

    L: int
    J: tuple
    k: tuple
    off_b: tuple  # (ℓ, offset) pairs
    off_t: tuple
    dim: int
    procs: tuple
    schur: bool
    nb0: int  # first index of the top block (β_L, θ_L) in z
    ld_kinds: tuple  # per level: "eig", "cheb", "aaa", "closure" or None
    ld_closures: tuple  # per level: the logdet function when its kind is "closure"
    lu_solves: tuple  # per level: sparsax's LU solve (a module function) or None

    def key(self, *flags) -> tuple:
        return (
            self.L, self.J, self.k, self.off_b, self.off_t, self.dim, self.procs,
            self.schur, self.nb0, self.ld_kinds, self.ld_closures, self.lu_solves,
            *flags,
        )  # fmt: skip


def _consts_and_static(st: MultilevelStructure, priors: MultilevelGibbsPriors):
    """Data arrays (passed to the compiled chunk) and the static structure."""
    import jax.numpy as jnp
    from jax.experimental import sparse as jsparse

    from ..negbin_reduced._flow_jax import build_sar_pattern

    P = st.pattern
    rows = P.indices
    cols = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))
    up = rows <= cols

    def upper(v):
        return jnp.zeros(int(up.sum())) if v is None else jnp.asarray(v[up])

    levels, L = st.levels, st.L
    procs = [lv.process for lv in levels]
    c = {
        "Ai": jnp.asarray(rows[up], dtype=jnp.int32),
        "Aj": jnp.asarray(cols[up], dtype=jnp.int32),
        "lam": jnp.asarray(st._lam_vals[up]),
        "u": tuple(upper(v) for v in st._unit_vals),
        "v": tuple(tuple(upper(v) for v in st._vals[ell]) for ell in range(1, L + 1)),
        "q_mu": jnp.asarray(st.q_mu),
        "Cty": jnp.asarray(st.Cty),
        "y": jnp.asarray(st.y),
        "X": tuple(jnp.asarray(lv.X) for lv in levels),
        "par": tuple(jnp.asarray(lv.parent, dtype=jnp.int32) for lv in levels[:-1]),
        "W": tuple(
            jsparse.BCOO.from_scipy_sparse(lv.W.tocsr()) if lv.W is not None else None
            for lv in levels
        ),
        "Wt": tuple(
            jsparse.BCOO.from_scipy_sparse(lv.W.T.tocsr())
            if (lv.W is not None and lv.process == "error")
            else None
            for lv in levels
        ),
    }
    y, Wy = st.y, st.Wy
    if procs[0] in ("lag", "error"):
        c["Wy"] = jnp.asarray(Wy)
        c["yy"] = jnp.float64(y @ y)
        c["yWy"] = jnp.float64(y @ Wy)
        c["WyWy"] = jnp.float64(Wy @ Wy)
    if procs[0] == "lag":
        c["Ctw"] = jnp.asarray(st.Ctw)
    if procs[0] == "error":
        W0 = levels[0].W
        c["c1"] = jnp.asarray(st.C.T @ (Wy + W0.T @ y))
        c["c2"] = jnp.asarray(st.C.T @ (W0.T @ Wy))
    lu = [None]
    for ell in range(1, L + 1):
        lv = levels[ell]
        if lv.process == "none":
            lu.append(None)
            continue
        pat = build_sar_pattern(lv.W, lv.W.shape[0])
        lu.append(
            (
                jnp.asarray(pat["Ai"], dtype=jnp.int32),
                jnp.asarray(pat["Aj"], dtype=jnp.int32),
                jnp.asarray(pat["eye_vals"]),
                jnp.asarray(pat["w_vals"]),
            )
        )
    c["lu"] = tuple(lu)
    c["prior"] = jnp.asarray(
        [priors.sigma2_alpha, priors.sigma2_beta, priors.sigma_nu, priors.sigma_scale],
        dtype=jnp.float64,
    )
    c["lo"] = jnp.asarray([lv.rho_lower for lv in levels], dtype=jnp.float64)
    c["hi"] = jnp.asarray([lv.rho_upper for lv in levels], dtype=jnp.float64)

    from ..._logdet._jax import logdet_jax_params
    from .._utils._sparsax_lu import sparsax_lu

    kinds, closures, params, solves = [], [], [], []
    for ell, lv in enumerate(levels):
        if not lv.spatial:
            kinds.append(None), closures.append(None), params.append(None)
            solves.append(None)
            continue
        kind, prm = lv.logdet_jax or logdet_jax_params(
            lv.W, lv.logdet_method, lv.rho_lower, lv.rho_upper
        )
        kinds.append(kind)
        closures.append(prm if kind == "closure" else None)
        params.append(None if kind == "closure" else prm)
        if ell >= 1:
            Ai, Aj, eye, w = c["lu"][ell]
            solves.append(sparsax_lu(Ai, Aj, st.J[ell]).solve)
        else:
            solves.append(None)
    c["ld"] = tuple(params)

    # The top block (β_L, θ_L) is z's tail.  When it is small, its rows of P
    # are handled as a dense Schur complement (see the module docstring).
    nb0 = st.off_b[L]
    db = st.dim - nb0
    schur = db <= SCHUR_MAX
    if schur:
        r_u, c_u = rows[up], cols[up]
        aa = c_u < nb0
        ab = (r_u < nb0) & (c_u >= nb0)
        bb = r_u >= nb0
        c["aa_idx"] = jnp.asarray(np.flatnonzero(aa))
        c["Ai_aa"] = jnp.asarray(r_u[aa], dtype=jnp.int32)
        c["Aj_aa"] = jnp.asarray(c_u[aa], dtype=jnp.int32)
        c["ab_idx"] = jnp.asarray(np.flatnonzero(ab))
        c["ab_r"] = jnp.asarray(r_u[ab], dtype=jnp.int32)
        c["ab_c"] = jnp.asarray(c_u[ab] - nb0, dtype=jnp.int32)
        c["bb_idx"] = jnp.asarray(np.flatnonzero(bb))
        c["bb_r"] = jnp.asarray(r_u[bb] - nb0, dtype=jnp.int32)
        c["bb_c"] = jnp.asarray(c_u[bb] - nb0, dtype=jnp.int32)
        c["Vb"] = tuple(
            jnp.zeros((db, db))
            if M is None
            else jnp.asarray(M.tocsr()[nb0:, nb0:].toarray())
            for M in st._level_terms[L]
        )
    static = _Static(
        L=L,
        J=tuple(st.J),
        k=tuple(st.k),
        off_b=tuple(sorted(st.off_b.items())),
        off_t=tuple(sorted(st.off_t.items())),
        dim=st.dim,
        procs=tuple(procs),
        schur=schur,
        nb0=nb0,
        ld_kinds=tuple(kinds),
        ld_closures=tuple(closures),
        lu_solves=tuple(solves),
    )
    return c, static


def _make_sweep(
    static: _Static,
    parametrization: str,
    store_theta: bool,
    store_log_lik: bool,
):
    """``sweep(state, key, tuning, consts) -> (state, trace)`` for one chain.

    Closes over ``static`` only; every data-dependent value comes from
    ``consts``.
    """
    import jax
    import jax.numpy as jnp
    import sparsax

    from ..._logdet._jax import eval_logdet_params
    from .._utils._jax_slice import adapt_slice_width, jax_slice_sample_1d

    S = static
    L, J, k, procs = S.L, S.J, S.k, S.procs
    off_b, off_t = dict(S.off_b), dict(S.off_t)
    MODE_A = sparsax.MODE_A
    collapsed = parametrization == "collapsed"
    centred = parametrization in ("interweave", "centred")
    noncentred = parametrization in ("interweave", "noncentred")
    spatial = [p in ("lag", "error") for p in procs]

    def beta(z, ell):
        return z[off_b[ell] : off_b[ell] + k[ell]]

    def theta(z, ell):
        return z[off_t[ell] : off_t[ell] + J[ell]]

    def bounds(c, ell):
        lo, hi = c["lo"][ell], c["hi"][ell]
        pad = _EDGE * (hi - lo)
        return lo + pad, hi - pad

    def logdet_at(c, ell, r):
        """``log|I − rW_ℓ|`` from the level's parameters."""
        kind = S.ld_kinds[ell]
        prm = S.ld_closures[ell] if kind == "closure" else c["ld"][ell]
        return eval_logdet_params(kind, prm, r)

    def slice1(logf, x0, lo, hi, w, key, f0=None):
        x, ld, left, right = jax_slice_sample_1d(
            logf, x0, lo, hi, key=key, w=w, return_steps=True, log_density_x0=f0
        )
        return x, ld, left, right

    def known(ld, value, fallback):
        """``value`` when the slice returned a finite density, else ``fallback()``."""
        return jax.lax.cond(
            jnp.isfinite(ld), lambda _: value, lambda _: fallback(), None
        )

    # -- the top block as a dense Schur complement ----------------------------

    nb0, db = S.nb0, S.dim - S.nb0

    def dense_sym(c, vals):
        M = jnp.zeros((db, db)).at[c["bb_r"], c["bb_c"]].add(vals)
        return M + M.T - jnp.diag(jnp.diag(M))

    def top_term(c, r, s2):
        V0, V1, V2 = c["Vb"]
        return (V0 + r * (V1 + r * V2)) / s2

    def schur_setup(c, rho, sig2, q):
        """Factor P_aa once; return the pieces of the top block's marginal."""
        vals = P_vals(c, rho, sig2)
        v_aa = vals[c["aa_idx"]]
        Pab = jnp.zeros((nb0, db)).at[c["ab_r"], c["ab_c"]].add(vals[c["ab_idx"]])
        qa, qb = q[:nb0], q[nb0:]
        (G, g), ld_aa = sparsax.factor_solve(
            c["Ai_aa"], c["Aj_aa"], v_aa, [(Pab, MODE_A), (qa, MODE_A)],
            want_logdet=True, n=nb0,
        )  # fmt: skip
        S0 = dense_sym(c, vals[c["bb_idx"]]) - top_term(c, rho[L], sig2[L]) - Pab.T @ G
        return {
            "S0": S0,
            "r": qb - Pab.T @ g,
            "base": -0.5 * ld_aa + 0.5 * (qa @ g),
            "v_aa": v_aa,
            "Pab": Pab,
            "qa": qa,
        }

    def top_marg(sc, c, r, s2):
        Sm = sc["S0"] + top_term(c, r, s2)
        Lc = jnp.linalg.cholesky(Sm)
        u = jax.scipy.linalg.solve_triangular(Lc, sc["r"], lower=True)
        val = sc["base"] - jnp.sum(jnp.log(jnp.diag(Lc))) + 0.5 * (u @ u)
        return jnp.where(jnp.isfinite(val), val, -jnp.inf)

    def schur_draw(sc, c, rho, sig2, key):
        k_b, k_a = jax.random.split(key)
        Lc = jnp.linalg.cholesky(sc["S0"] + top_term(c, rho[L], sig2[L]))
        mean_b = jax.scipy.linalg.cho_solve((Lc, True), sc["r"])
        z_b = mean_b + jax.scipy.linalg.solve_triangular(
            Lc.T, jax.random.normal(k_b, (db,)), lower=False
        )
        z_a, _ = sparsax.sample_gaussian(
            c["Ai_aa"], c["Aj_aa"], sc["v_aa"], sc["qa"] - sc["Pab"] @ z_b,
            jax.random.normal(k_a, (nb0,)),
        )  # fmt: skip
        return jnp.concatenate([z_a, z_b])

    # -- the precision and the linear term --------------------------------

    def P_vals(c, rho, sig2):
        u0, u1, u2 = c["u"]
        unit = u0 + rho[0] * (u1 + rho[0] * u2) if procs[0] == "error" else u0
        vals = c["lam"] + unit / sig2[0]
        for ell in range(1, L + 1):
            v0, v1, v2 = c["v"][ell - 1]
            r = rho[ell]
            term = v0 + r * (v1 + r * v2) if spatial[ell] else v0
            vals = vals + term / sig2[ell]
        return vals

    def q_unit(c, r0, s0):
        if procs[0] == "lag":
            Ct = c["Cty"] - r0 * c["Ctw"]
        elif procs[0] == "error":
            Ct = c["Cty"] - r0 * c["c1"] + r0 * r0 * c["c2"]
        else:
            Ct = c["Cty"]
        return c["q_mu"] + Ct / s0

    def solve_logdet(c, Ax, rhs):
        sols, ld = sparsax.factor_solve(
            c["Ai"], c["Aj"], Ax, [(b, MODE_A) for b in rhs], want_logdet=True, n=S.dim
        )
        return sols, ld

    # -- residuals ----------------------------------------------------------

    def unit_resid(c, z, r0):
        u = c["y"] - theta(z, 1)[c["par"][0]]
        if k[0]:
            u = u - c["X"][0] @ beta(z, 0)
        if procs[0] == "lag":
            return u - r0 * c["Wy"]
        if procs[0] == "error":
            return u - r0 * (c["W"][0] @ u)
        return u

    def level_parts(c, z, ell):
        p = theta(z, ell)
        if k[ell]:
            p = p - c["X"][ell] @ beta(z, ell)
        if ell < L:
            p = p - theta(z, ell + 1)[c["par"][ell]]
        if procs[ell] == "lag":
            return p, -(c["W"][ell] @ theta(z, ell))
        if procs[ell] == "error":
            return p, -(c["W"][ell] @ p)
        return p, None

    def resid(c, z, ell, rho):
        if ell == 0:
            return unit_resid(c, z, rho[0])
        p, s = level_parts(c, z, ell)
        return p if s is None else p + rho[ell] * s

    def filter_solve(c, ell, r, rhs):
        Ai, Aj, eye, w = c["lu"][ell]
        return S.lu_solves[ell](Ai, Aj, eye - r * w, rhs)

    # -- the sweep ----------------------------------------------------------

    def sweep(state, key, tuning, c):
        z, rho, sig2, aux = state["z"], state["rho"], state["sig2"], state["aux"]
        w_rho, w_nc = state["w_rho"], state["w_nc"]
        w_sig, w_snc = state["w_sig"], state["w_snc"]
        keys = iter(jax.random.split(key, 4 + 8 * (L + 1)))
        a0, b0, nu, A = c["prior"][0], c["prior"][1], c["prior"][2], c["prior"][3]
        lsd_lo, lsd_hi = jnp.log(A) - 10.0, jnp.log(A) + 5.0

        # 1. σ_0² | z, ρ_0
        e0 = unit_resid(c, z, rho[0])
        g = jax.random.gamma(next(keys), a0 + 0.5 * J[0])
        sig2 = sig2.at[0].set((b0 + 0.5 * (e0 @ e0)) / g)
        s0 = sig2[0]

        # 2. ρ_0 with z integrated out
        if procs[0] == "lag":
            a = c["q_mu"] + c["Cty"] / s0
            b = c["Ctw"] / s0
            (Pa, Pb), ld_P = solve_logdet(c, P_vals(c, rho, sig2), [a, b])
            c0 = -0.5 * c["yy"] / s0 + 0.5 * (a @ Pa)
            c1 = c["yWy"] / s0 - (a @ Pb)
            c2 = -0.5 * c["WyWy"] / s0 + 0.5 * (b @ Pb)

            def logf0(r):
                return logdet_at(c, 0, r) + c0 + r * (c1 + r * c2)

        elif procs[0] == "error":

            def logf0(lam):
                q = q_unit(c, lam, s0)
                (sol,), ld = solve_logdet(c, P_vals(c, rho.at[0].set(lam), sig2), [q])
                Ay2 = c["yy"] - 2.0 * lam * c["yWy"] + lam * lam * c["WyWy"]
                return (
                    logdet_at(c, 0, lam) - 0.5 * ld - 0.5 * Ay2 / s0 + 0.5 * (q @ sol)
                )

        marg_cur = None  # −½log|P| + ½qᵀP⁻¹q at the current state, when known
        if spatial[0]:
            lo, hi = bounds(c, 0)
            r0, ld0, sl, sr = slice1(logf0, rho[0], lo, hi, w_rho[0], next(keys))
            rho = rho.at[0].set(r0)
            w_rho = w_rho.at[0].set(adapt_slice_width(w_rho[0], sl, sr, tuning))
            if procs[0] == "lag":
                marg_cur = -0.5 * ld_P + 0.5 * (
                    (a @ Pa) - 2.0 * r0 * (a @ Pb) + r0 * r0 * (b @ Pb)
                )
            else:
                Ay2 = c["yy"] - 2.0 * r0 * c["yWy"] + r0 * r0 * c["WyWy"]

                def direct(r0=r0):
                    q0 = q_unit(c, r0, s0)
                    (sol,), ld = solve_logdet(c, P_vals(c, rho, sig2), [q0])
                    return -0.5 * ld + 0.5 * (q0 @ sol)

                marg_cur = known(
                    ld0, ld0 - logdet_at(c, 0, r0) + 0.5 * Ay2 / s0, direct
                )

        # 3. collapsed ρ_ℓ, σ_ℓ (z integrated out)
        q = q_unit(c, rho[0], sig2[0])
        sc = None
        if collapsed:

            def marg(rr, ss):
                (sol,), ld = solve_logdet(c, P_vals(c, rr, ss), [q])
                return -0.5 * ld + 0.5 * (q @ sol)

            def sd_prior(ls, ell):
                v = jnp.exp(2.0 * ls)
                return (
                    -J[ell] * ls - 0.5 * (nu + 1.0) * jnp.log1p(v / (nu * A * A)) + ls
                )

            # Levels below the top (and the top when it is large): full P.
            for ell in range(1, L if S.schur else L + 1):
                if marg_cur is None:
                    marg_cur = marg(rho, sig2)
                if spatial[ell]:

                    def logf_r(r, ell=ell, rho=rho, sig2=sig2):
                        return logdet_at(c, ell, r) + marg(rho.at[ell].set(r), sig2)

                    lo, hi = bounds(c, ell)
                    r_new, ld, sl, sr = slice1(
                        logf_r, rho[ell], lo, hi, w_rho[ell], next(keys),
                        f0=logdet_at(c, ell, rho[ell]) + marg_cur,
                    )  # fmt: skip
                    rho = rho.at[ell].set(r_new)
                    w_rho = w_rho.at[ell].set(
                        adapt_slice_width(w_rho[ell], sl, sr, tuning)
                    )
                    marg_cur = known(
                        ld,
                        ld - logdet_at(c, ell, r_new),
                        lambda rho=rho, sig2=sig2: marg(rho, sig2),
                    )

                def logf_s(ls, ell=ell, rho=rho, sig2=sig2):
                    return sd_prior(ls, ell) + marg(
                        rho, sig2.at[ell].set(jnp.exp(2.0 * ls))
                    )

                ls0 = jnp.clip(0.5 * jnp.log(sig2[ell]), lsd_lo, lsd_hi)
                ls_new, ld, sl, sr = slice1(
                    logf_s, ls0, lsd_lo, lsd_hi, w_sig[ell], next(keys),
                    f0=sd_prior(ls0, ell) + marg_cur,
                )  # fmt: skip
                sig2 = sig2.at[ell].set(jnp.exp(2.0 * ls_new))
                w_sig = w_sig.at[ell].set(adapt_slice_width(w_sig[ell], sl, sr, tuning))
                marg_cur = known(
                    ld,
                    ld - sd_prior(ls_new, ell),
                    lambda rho=rho, sig2=sig2: marg(rho, sig2),
                )

            # The top level through its dense Schur complement.
            if S.schur:
                sc = schur_setup(c, rho, sig2, q)
                if spatial[L]:

                    def logf_rt(r, s2=sig2[L]):
                        return logdet_at(c, L, r) + top_marg(sc, c, r, s2)

                    lo, hi = bounds(c, L)
                    r_new, _, sl, sr = slice1(
                        logf_rt, rho[L], lo, hi, w_rho[L], next(keys)
                    )
                    rho = rho.at[L].set(r_new)
                    w_rho = w_rho.at[L].set(adapt_slice_width(w_rho[L], sl, sr, tuning))

                def logf_st(ls, r=rho[L]):
                    return sd_prior(ls, L) + top_marg(sc, c, r, jnp.exp(2.0 * ls))

                ls0 = jnp.clip(0.5 * jnp.log(sig2[L]), lsd_lo, lsd_hi)
                ls_new, _, sl, sr = slice1(
                    logf_st, ls0, lsd_lo, lsd_hi, w_sig[L], next(keys)
                )
                sig2 = sig2.at[L].set(jnp.exp(2.0 * ls_new))
                w_sig = w_sig.at[L].set(adapt_slice_width(w_sig[L], sl, sr, tuning))

        # 4. z | everything
        if sc is not None:
            z = schur_draw(sc, c, rho, sig2, next(keys))
        else:
            draw = jax.random.normal(next(keys), (S.dim,))
            z, _ = sparsax.sample_gaussian(
                c["Ai"], c["Aj"], P_vals(c, rho, sig2), q, draw
            )

        # 5. centred ρ_ℓ, σ_ℓ given z
        if centred:
            for ell in range(1, L + 1):
                p, s = level_parts(c, z, ell)
                e = p if s is None else p + rho[ell] * s
                a_new = (nu / sig2[ell] + 1.0 / A**2) / jax.random.gamma(
                    next(keys), 0.5 * (nu + 1.0)
                )
                aux = aux.at[ell].set(a_new)
                sig2 = sig2.at[ell].set(
                    (0.5 * (e @ e) + nu / a_new)
                    / jax.random.gamma(next(keys), 0.5 * (J[ell] + nu))
                )
                if s is None:
                    continue
                pp, ps, ss, s2 = p @ p, p @ s, s @ s, sig2[ell]

                def logf_c(r, ell=ell, pp=pp, ps=ps, ss=ss, s2=s2):
                    return (
                        logdet_at(c, ell, r) - 0.5 * (pp + r * (2.0 * ps + r * ss)) / s2
                    )

                lo, hi = bounds(c, ell)
                r_new, _, sl, sr = slice1(
                    logf_c, rho[ell], lo, hi, w_rho[ell], next(keys)
                )
                rho = rho.at[ell].set(r_new)
                w_rho = w_rho.at[ell].set(adapt_slice_width(w_rho[ell], sl, sr, tuning))

        # 6. non-centred ρ_ℓ, σ_ℓ given ε̃_ℓ
        if noncentred:
            for ell in range(1, L + 1):
                m = ell - 1
                sd = jnp.sqrt(sig2[ell])
                eps = resid(c, z, ell, rho) / sd
                mean = jnp.zeros(J[ell])
                if k[ell]:
                    mean = mean + c["X"][ell] @ beta(z, ell)
                if ell < L:
                    mean = mean + theta(z, ell + 1)[c["par"][ell]]
                par, r_m = c["par"][m], rho[m]

                def GD(th, par=par, r_m=r_m, m=m):
                    g = th[par]
                    return g - r_m * (c["W"][m] @ g) if procs[m] == "error" else g

                def GDt(v, par=par, r_m=r_m, m=m, ell=ell):
                    if procs[m] == "error":
                        v = v - r_m * (c["Wt"][m] @ v)
                    return jax.ops.segment_sum(v, par, num_segments=J[ell])

                h = GDt(resid(c, z, m, rho) + GD(theta(z, ell)))
                s2 = sig2[m]

                def parts(r, ell=ell, mean=mean, eps=eps):
                    if procs[ell] == "lag":
                        return filter_solve(c, ell, r, mean), filter_solve(
                            c, ell, r, eps
                        )
                    if procs[ell] == "error":
                        return mean, filter_solve(c, ell, r, eps)
                    return mean, eps

                if spatial[ell]:

                    def logf_n(r, ell=ell, mean=mean, eps=eps, sd=sd, h=h, s2=s2):
                        if procs[ell] == "lag":
                            th = filter_solve(c, ell, r, mean + sd * eps)
                        else:
                            th = mean + sd * filter_solve(c, ell, r, eps)
                        gth = GD(th)
                        return -0.5 * ((gth @ gth) - 2.0 * (h @ th)) / s2

                    lo, hi = bounds(c, ell)
                    r_new, _, sl, sr = slice1(
                        logf_n, rho[ell], lo, hi, w_nc[ell], next(keys)
                    )
                    rho = rho.at[ell].set(r_new)
                    w_nc = w_nc.at[ell].set(
                        adapt_slice_width(w_nc[ell], sl, sr, tuning)
                    )

                a_, b_ = parts(rho[ell])
                Ga, Gb = GD(a_), GD(b_)
                P_s = (Gb @ Gb) / s2
                h_s = ((h @ b_) - (Ga @ Gb)) / s2

                def logf_sd(ls, P_s=P_s, h_s=h_s):
                    v = jnp.exp(ls)
                    return (
                        -0.5 * P_s * v * v
                        + h_s * v
                        - 0.5 * (nu + 1.0) * jnp.log1p(v * v / (nu * A * A))
                        + ls
                    )

                ls0 = jnp.clip(jnp.log(sd), lsd_lo, lsd_hi)
                ls_new, _, sl, sr = slice1(
                    logf_sd, ls0, lsd_lo, lsd_hi, w_snc[ell], next(keys)
                )
                w_snc = w_snc.at[ell].set(adapt_slice_width(w_snc[ell], sl, sr, tuning))
                sd_new = jnp.exp(ls_new)
                sig2 = sig2.at[ell].set(sd_new * sd_new)
                z = z.at[off_t[ell] : off_t[ell] + J[ell]].set(a_ + sd_new * b_)

        new_state = dict(
            z=z, rho=rho, sig2=sig2, aux=aux,
            w_rho=w_rho, w_nc=w_nc, w_sig=w_sig, w_snc=w_snc,
        )  # fmt: skip
        trace = {
            "rho": rho,
            "sigma": jnp.sqrt(sig2),
            "beta": tuple(beta(z, ell) for ell in range(L + 1)),
        }
        if store_theta:
            trace["theta"] = tuple(theta(z, ell) for ell in range(1, L + 1))
        if store_log_lik:
            e = unit_resid(c, z, rho[0])
            ll = -0.5 * e * e / sig2[0] - 0.5 * jnp.log(2.0 * jnp.pi * sig2[0])
            if spatial[0]:
                ll = ll + logdet_at(c, 0, rho[0]) / J[0]
            trace["log_lik"] = ll
        return new_state, trace

    return sweep


def run_multilevel_jax(
    st: MultilevelStructure,
    priors: MultilevelGibbsPriors,
    draws: int,
    tune: int,
    chains: int,
    seeds,
    *,
    thin: int = 1,
    parametrization: str = "collapsed",
    store_theta: bool = True,
    store_log_lik: bool = False,
    progressbar: bool = False,
) -> list[dict]:
    """Run ``chains`` chains; returns one trace dict per chain, as the NumPy runner."""
    import jax
    import jax.numpy as jnp

    from ..._jax_dispatch import ensure_x64
    from .._utils._jax_utils import cached_sweep, run_chains_chunked
    from .._utils._progress import GibbsProgressBarManager
    from .._utils._sparsax_lu import set_sparsax_lu_cache_size
    from ._core import PARAMETRIZATIONS

    if parametrization not in PARAMETRIZATIONS:
        raise ValueError(
            f"parametrization must be one of {PARAMETRIZATIONS}, got {parametrization!r}"
        )
    ensure_x64()
    set_sparsax_lu_cache_size(max(32, 6 * chains))
    consts, static = _consts_and_static(st, priors)
    sweep = cached_sweep(
        ("multilevel", static.key(parametrization, store_theta, store_log_lik)),
        lambda: _make_sweep(static, parametrization, store_theta, store_log_lik),
    )

    L = static.L
    states, warm_keys, draw_keys = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        init = initialize(st, priors, rng)
        w_rho = np.array(
            [0.1 * (lv.rho_upper - lv.rho_lower) for lv in st.levels], dtype=np.float64
        )
        states.append(
            {
                "z": jnp.asarray(init.z),
                "rho": jnp.asarray(init.rho),
                "sig2": jnp.asarray(init.sig2),
                "aux": jnp.asarray(init.aux),
                "w_rho": jnp.asarray(w_rho),
                "w_nc": jnp.asarray(w_rho),
                "w_sig": jnp.full(L + 1, 0.5),
                "w_snc": jnp.full(L + 1, 0.5),
            }
        )
        k_int = int(rng.integers(0, 2**31 - 1))
        warm_keys.append(jax.random.PRNGKey(k_int))
        draw_keys.append(jax.random.fold_in(jax.random.PRNGKey(k_int), 1))

    with GibbsProgressBarManager(
        chains=chains,
        draws=draws,
        tune=tune,
        progressbar=progressbar,
        model_type="multilevel",
    ) as pm:
        if pm is not None:
            for ch in range(chains):
                pm.start_chain(ch)

        def _progress(i, tuning):
            if pm is not None:
                for ch in range(chains):
                    pm.update(ch, i, tuning=tuning)

        _, traces = run_chains_chunked(
            sweep,
            states,
            warm_keys,
            draw_keys,
            tune=tune,
            draws=draws,
            on_chunk=_progress,
            consts=consts,
        )

    n_keep = draws // thin
    sl = slice(None, None, thin)
    out = []
    for ch in range(chains):
        res = {
            "rho": np.asarray(traces["rho"][ch])[sl][:n_keep],
            "sigma": np.asarray(traces["sigma"][ch])[sl][:n_keep],
        }
        for ell in range(L + 1):
            if static.k[ell]:
                res[f"beta_{ell}"] = np.asarray(traces["beta"][ell][ch])[sl][:n_keep]
            if ell >= 1 and store_theta:
                res[f"theta_{ell}"] = np.asarray(traces["theta"][ell - 1][ch])[sl][
                    :n_keep
                ]
        if store_log_lik:
            res["log_lik"] = np.asarray(traces["log_lik"][ch])[sl][:n_keep]
        out.append(res)
    return out
