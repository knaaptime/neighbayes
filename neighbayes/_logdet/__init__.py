"""Log-determinant computation for spatial econometric models.

A model names its log-determinant by ``logdet_method`` (the approximation over
ρ: ``exact``, ``cheb``, ``aaa``, ``spline``, ``stochastic``, or the unrestricted
flows' ``trace_series`` and ``resolvent``) and ``logdet_decomp`` (what supplies
the exact values: ``eig``, ``chol`` or ``lu``).  :mod:`._config` documents the
names, the auto-selection and the internal keys the factories dispatch on.
"""

from ._aaa import (
    AAAContext,
    AAAPrecompute,
    CholAAAContext,
    aaa_logdet_eval,
    aaa_logdet_eval_vec,
    aaa_logdet_precompute,
    chol_aaa_logdet_precompute,
)
from ._cheb_stochastic import (
    ChebStochasticPrecompute,
    cheb_stochastic_logdet_eval,
    cheb_stochastic_logdet_eval_vec,
    cheb_stochastic_logdet_precompute,
)
from ._chebyshev import chebyshev
from ._chol_cheb import (
    CholChebPrecompute,
    chol_cheb_logdet_eval,
    chol_cheb_logdet_eval_vec,
    chol_cheb_logdet_precompute,
)
from ._config import (
    FLOW_LOGDET_METHODS,
    LOGDET_DECOMPS,
    LOGDET_METHODS,
    VALID_LOGDET_METHODS,
    LogdetBounds,
    LogDetMethod,
    LogDetMethodName,
    logdet_key,
    logdet_label,
    resolve_logdet_bounds,
    resolve_logdet_method,
)
from ._factories import (
    clear_logdet_fn_cache,
    get_cached_logdet_fn,
    make_logdet_fn,
    make_logdet_grad_numpy_fn,
    make_logdet_grad_numpy_vec_fn,
    make_logdet_numpy_fn,
    make_logdet_numpy_vec_fn,
)
from ._flow import (
    make_flow_separable_logdet,
    make_flow_separable_logdet_numpy,
)

# Flow-resolvent exports are resolved lazily (see __getattr__ below):
# _flow_resolvent pulls in the quadrature/JAX machinery, which the
# cross-sectional import path never needs.
_FLOW_RESOLVENT_EXPORTS = frozenset(
    {
        "FlowKron",
        "FlowKronJax",
        "flow_logdet_grad",
        "flow_logdet_grad_exact",
        "flow_logdet_grad_jax",
        "flow_logdet_value",
        "flow_logdet_value_and_grad",
        "flow_logdet_value_and_grad_jax",
    }
)


def __getattr__(name):
    if name in _FLOW_RESOLVENT_EXPORTS:
        from . import _flow_resolvent

        return getattr(_flow_resolvent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from ._jax import jax_logdet_chebyshev, make_logdet_jax_fn
from ._pytensor import logdet_chebyshev, logdet_eigenvalue
from ._resolvent import (
    logdet_grad_aaa,
    logdet_grad_chebyshev,
    logdet_grad_eigenvalue,
)

__all__ = [
    # Config
    "LogDetMethod",
    "LogDetMethodName",
    "LogdetBounds",
    "VALID_LOGDET_METHODS",
    "LOGDET_METHODS",
    "FLOW_LOGDET_METHODS",
    "LOGDET_DECOMPS",
    "logdet_key",
    "logdet_label",
    "resolve_logdet_method",
    "resolve_logdet_bounds",
    # PyTensor primitives
    "logdet_eigenvalue",
    "logdet_chebyshev",
    # JAX
    "jax_logdet_chebyshev",
    "make_logdet_jax_fn",
    # Factories
    "make_logdet_fn",
    "make_logdet_numpy_fn",
    "make_logdet_numpy_vec_fn",
    "make_logdet_grad_numpy_fn",
    "make_logdet_grad_numpy_vec_fn",
    "get_cached_logdet_fn",
    "clear_logdet_fn_cache",
    # Resolvent trace / logdet gradient core
    "logdet_grad_eigenvalue",
    "logdet_grad_chebyshev",
    "logdet_grad_aaa",
    # Flow (separable factorization)
    "make_flow_separable_logdet",
    "make_flow_separable_logdet_numpy",
    # Flow resolvent-Kronecker gradient (scalable, eigenvalue-free)
    "FlowKron",
    "flow_logdet_grad",
    "flow_logdet_grad_exact",
    "flow_logdet_value",
    "flow_logdet_value_and_grad",
    # Chebyshev
    "chebyshev",
    # Stochastic Chebyshev
    "ChebStochasticPrecompute",
    "cheb_stochastic_logdet_precompute",
    "cheb_stochastic_logdet_eval",
    "cheb_stochastic_logdet_eval_vec",
]
