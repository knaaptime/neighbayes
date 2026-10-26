"""Octave harness for the LeSage-toolbox validation tests.

The flagged tests shell out to Octave rather than import it, so the package's
import surface stays pure Python.  Each test exports one dataset as ASCII
(``y.txt`` / ``X.txt`` / ``W.txt`` — the ``.m`` side reads the same matrix
file the Python side fit, per the plan's "one dataset, one W" standard),
writes a generated driver script, and reads the toolbox's posterior draws
back from the text files the driver dumps.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

_OCTAVE_ENV = "NEIGHBAYES_OCTAVE"
_TOOLBOX_ENV = "NEIGHBAYES_TOOLBOX_DIR"

_TIMEOUT_SECONDS = 1800.0


def octave_binary() -> str | None:
    """Locate an Octave executable, from ``NEIGHBAYES_OCTAVE`` or ``PATH``."""
    explicit = os.environ.get(_OCTAVE_ENV)
    if explicit:
        return explicit
    return shutil.which("octave")


def toolbox_dir() -> Path | None:
    """Locate the LeSage May-2021 toolbox, from ``NEIGHBAYES_TOOLBOX_DIR``.

    Falls back to the undisturbed local copy alongside the repo
    (``../reference/toolbox_may2021``) for interactive development; the
    bespoke Actions workflow fetches the toolbox at its pinned URL and sets
    the environment variable instead.
    """
    explicit = os.environ.get(_TOOLBOX_ENV)
    if explicit:
        path = Path(explicit)
        return path if path.is_dir() else None
    local = Path(__file__).resolve().parents[3] / "reference" / "toolbox_may2021"
    return local if local.is_dir() else None


def run_octave(script: Path, workdir: Path, timeout: float = _TIMEOUT_SECONDS):
    """Run ``script`` as ``octave --no-gui --quiet script`` inside ``workdir``.

    Toolboxes added on ``genpath`` shadow a few Octave core functions
    (``ols``, ``quantile`` and kin) — warnings, not errors; the harness
    accepts them and only fails on a nonzero exit.
    """
    exe = octave_binary()
    if exe is None:
        raise RuntimeError("Octave not found on PATH (or NEIGHBAYES_OCTAVE)")
    try:
        proc = subprocess.run(
            [exe, "--no-gui", "--quiet", script.name],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Octave timed out after {timeout}s") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"Octave exited {proc.returncode} on {script.name}\n"
            f"stdout tail:\n{proc.stdout[-2000:]}\n"
            f"stderr tail:\n{proc.stderr[-4000:]}"
        )
    return proc.stdout


# ---------------------------------------------------------------------------
# Driver scripts
# ---------------------------------------------------------------------------
# The toolbox paths shadow Octave core functions — warnings only, accepted
# above.  ``prior.novi = 1`` selects the homoscedastic model, which is the
# one neighbayes fits.  Draw dumps go through ``fprintf`` rather than
# ``dlmwrite`` so the files are plain whitespace-separated ASCII that
# ``np.loadtxt`` reads without format guessing.


def _dump_lines(varname: str, out: str) -> str:
    return (
        f"fid = fopen('{out}', 'w');"
        f"fprintf(fid, '%.17e\\n', {varname}(:));"
        "fclose(fid);\n"
    )


def _dump_matrix(varname: str, out: str) -> str:
    # One line per draw: repmat builds a format with one spec per column and
    # fprintf consumes the transposed matrix column-major, so the output is
    # row-major.  (A single-spec format would print every element on its own
    # line.)
    return (
        f"fid = fopen('{out}', 'w');"
        f"fmt = [repmat('%.17e ', 1, size({varname}, 2)) '\\n'];"
        f"fprintf(fid, fmt, {varname}.');"
        "fclose(fid);\n"
    )


TOOLBOX_PATHS = (
    "addpath(genpath('{toolbox}'));\n"
    "addpath('{shims}');\n"  # shims shadow Octave core on purpose (see below)
)

FMINBND_SHIM = """function [x, fval, exitflag, output] = fminbnd(fcn, x1, x2, options, varargin)
% MATLAB-style trailing-parameter passthrough for fminbnd.
% The toolbox's ML routines call fminbnd('f_m', rmin, rmax, options, P...),
% the idiom MATLAB's fminbnd accepts and Octave's (which takes no extra
% parameters) rejects with "function called with too many inputs".  This
% shim folds the trailing parameters into the objective and delegates to
% Octave's own fminbnd, dropping itself from the path for the call.  It is
% written by the harness into workdir/shims; the toolbox stays unmodified.
  here = fileparts(mfilename('fullpath'));
  rmpath(here);
  restore = onCleanup(@() addpath(here));

  if nargin == 1 && ischar(fcn) && strcmp(fcn, 'defaults')
    x = fminbnd('defaults');      % Octave core's answer, shim off the path
    return;
  end;

  if isempty(varargin)
    [x, fval, exitflag, output] = fminbnd(fcn, x1, x2, options);
    return;
  end;

  wrap = @(z) feval(fcn, z, varargin{:});
  [x, fval, exitflag, output] = fminbnd(wrap, x1, x2, options);
end;
"""


WAITBAR_SHIM = """function h = waitbar(frac, varargin)
% Headless harness shim for the toolbox's progress bar.  The toolbox's MCMC
% loops call waitbar(iter/ndraw) after incrementing iter, so the final pass
% hands it a fraction above 1, and the run is headless besides; Octave's
% waitbar refuses both.  Keep one hidden stub figure so the caller's
% close(handle) stays valid; the fraction itself is discarded.
  persistent stub;
  if isempty(stub) || ~ishandle(stub)
    stub = figure('visible', 'off');
  end;
  h = stub;
end;
"""


NORMPDF_SHIM = """function p = normpdf(x, mu, sigma)
% The normal density, for the toolbox's probit effects.  MATLAB ships
% normpdf in its Statistics Toolbox, Octave in its separately installed
% statistics package; the harness supplies the one function rather than the
% package.
  if nargin < 2, mu = 0; end;
  if nargin < 3, sigma = 1; end;
  p = exp(-0.5 * ((x - mu) ./ sigma) .^ 2) ./ (sqrt(2 * pi) .* sigma);
end;
"""


def write_shims(workdir: Path) -> Path:
    """Write the harness shims into ``workdir/shims`` and return that dir."""
    shims = workdir / "shims"
    shims.mkdir(parents=True, exist_ok=True)
    (shims / "fminbnd.m").write_text(FMINBND_SHIM)
    (shims / "waitbar.m").write_text(WAITBAR_SHIM)
    (shims / "normpdf.m").write_text(NORMPDF_SHIM)
    return shims


GAUSS_DRIVER = (
    TOOLBOX_PATHS
    + "load('y.txt');\n"
    + "load('X.txt');\n"
    + "load('W.txt');\n"
    + "x = X;\n"
    + "rand('state', {seed});\n"
    + "randn('state', {seed});\n"
    + "prior.novi = 1;\n"
    + "prior.eig = 0;\n"
    + "prior.lflag = 0;\n"
    + "res = {routine}(y, x, W, {ndraw}, {nomit}, prior);\n"
    + _dump_lines("res.pdraw", "{prefix}_rho.txt")
    + _dump_matrix("res.bdraw", "{prefix}_beta.txt")
    + _dump_lines("res.sdraw", "{prefix}_sige.txt")
    + "fprintf('METH %s\\n', res.meth);\n"
)


def gauss_driver(
    routine: str,
    toolbox: Path,
    workdir: Path,
    ndraw: int,
    nomit: int,
    seed: int,
    prefix: str,
) -> str:
    """Driver text for ``sar_g``/``sem_g``/``sdm_g`` on the exported dataset."""
    return GAUSS_DRIVER.format(
        toolbox=toolbox,
        shims=write_shims(workdir),
        routine=routine,
        ndraw=ndraw,
        nomit=nomit,
        seed=seed,
        prefix=prefix,
    )


PANEL_DRIVER = (
    TOOLBOX_PATHS
    + "load('y.txt');\n"
    + "load('X.txt');\n"
    + "load('W.txt');\n"
    + "x = X;\n"
    + "rand('state', {seed});\n"
    + "randn('state', {seed});\n"
    + "prior.model = 1;\n"  # unit fixed effects
    + "prior.novi_flag = 1;\n"
    + "prior.lflag = 0;\n"
    + "res = sar_panel_FE_g(y, x, W, {T}, {ndraw}, {nomit}, prior);\n"
    + _dump_lines("res.pdraw", "{prefix}_rho.txt")
    + _dump_matrix("res.bdraw", "{prefix}_beta.txt")
    + _dump_lines("res.sdraw", "{prefix}_sige.txt")
    + "fprintf('METH %s\\n', res.meth);\n"
)


def panel_driver(
    toolbox: Path,
    workdir: Path,
    T: int,
    ndraw: int,
    nomit: int,
    seed: int,
    prefix: str,
) -> str:
    """Driver text for ``sar_panel_FE_g`` (unit effects) on the exported panel.

    The data are stacked by period, units within each period, as both sides
    expect; ``W`` is the ``N x N`` matrix, which the toolbox expands to
    ``I_T (x) W`` itself.
    """
    return PANEL_DRIVER.format(
        toolbox=toolbox,
        shims=write_shims(workdir),
        T=T,
        ndraw=ndraw,
        nomit=nomit,
        seed=seed,
        prefix=prefix,
    )


PROBIT_DRIVER = (
    TOOLBOX_PATHS
    + "load('y.txt');\n"
    + "load('X.txt');\n"
    + "load('W.txt');\n"
    + "x = X;\n"
    + "rand('state', {seed});\n"
    + "randn('state', {seed});\n"
    + "prior.eig = 0;\n"
    + "prior.lflag = 0;\n"
    + "res = {routine}(y, x, W, {ndraw}, {nomit}, prior);\n"
    + _dump_lines("res.pdraw", "{prefix}_rho.txt")
    + _dump_matrix("res.bdraw", "{prefix}_beta.txt")
    + "fprintf('METH %s\\n', res.meth);\n"
)


def probit_driver(
    routine: str,
    toolbox: Path,
    workdir: Path,
    ndraw: int,
    nomit: int,
    seed: int,
    prefix: str,
) -> str:
    """Driver text for ``sarp_g`` (or its repair) on the exported binary dataset."""
    return PROBIT_DRIVER.format(
        toolbox=toolbox,
        shims=write_shims(workdir),
        routine=routine,
        ndraw=ndraw,
        nomit=nomit,
        seed=seed,
        prefix=prefix,
    )


# ``sarp_g`` as shipped does not sample the SAR probit posterior; three
# repairs make it do so, each measured against neighbayes and against a plain
# full-conditional Gibbs sampler (wiki: lesage-toolbox-validation).
#
# 1. Stale deviations.  The sweep keeps z = y* - mu between iterations while
#    mu = (I - rho W)^-1 X beta changes, so each single-site pass starts from
#    utilities shifted by mu_new - mu_old.  Recentring z on the stored
#    utilities first restores the Gibbs step; without it the beta posterior
#    is about 12% too wide.
# 2. Step order.  beta is drawn given the current rho, then rho with beta
#    integrated out, then the utilities given that beta and the new rho.  After
#    a draw that integrates beta out, beta must be drawn again before anything
#    conditions on it (van Dyk & Park 2008); the rho draw moves ahead.
# 3. The rho conditional is sar_g's, which integrates out an error variance
#    (|A| e(rho)^-(n-k)/2) the probit fixes at one (|A| exp(-e(rho)/2)).
#
# The repaired copy is written into the work directory from the served file,
# nothing else changed; each anchor must match exactly once.
_SARP_G = Path("spatial") / "sar_models" / "sarp_g.m"
SARP_G_REPAIRED = "sarp_g_repaired"


def write_sarp_g_repaired(toolbox: Path, workdir: Path) -> str:
    """Write ``sarp_g`` with its sweep repaired as ``workdir/sarp_g_repaired.m``.

    Returns the routine's name for :func:`probit_driver`.
    """
    lines = (toolbox / _SARP_G).read_text().replace("\r\n", "\n").split("\n")

    def only(match, what):
        hits = [i for i, line in enumerate(lines) if match(line.strip())]
        if len(hits) != 1:
            raise ValueError(f"sarp_g.m: {len(hits)} lines match the {what} anchor")
        return hits[0]

    # 1. recentre the deviations on the stored utilities before each pass
    sweep = only(lambda s: s == "for initer=1:nsample;", "latent-sweep")
    lines.insert(sweep, "          if iter > 1, z = y - mu; end;")
    # 2. draw rho, then beta given it
    beta_from = only(lambda s: s == "% update beta", "beta-draw start")
    beta_to = only(lambda s: s == "xb = x*bhat;", "beta-draw end")
    rho_from = only(
        lambda s: "numerical integration to perform rho-draw" in s, "rho-draw start"
    )
    rho_to = only(lambda s: s.startswith("rho = draw_rho("), "rho-draw end")
    if not beta_from < beta_to < rho_from < rho_to < sweep:
        raise ValueError("sarp_g.m: the sweep's steps are not in the expected order")
    lines[beta_from : rho_to + 1] = (
        lines[rho_from : rho_to + 1] + lines[beta_from : beta_to + 1]
    )
    # 3. the probit's rho conditional, variance one
    profile = only(lambda s: s == "z = -nmk*log(z);", "rho-conditional")
    lines[profile] = lines[profile].replace("-nmk*log(z)", "-0.5*z")
    header = only(lambda s: s.startswith("function results = sarp_g("), "header")
    lines[header] = lines[header].replace("sarp_g(", f"{SARP_G_REPAIRED}(", 1)
    (workdir / f"{SARP_G_REPAIRED}.m").write_text("\n".join(lines))
    return SARP_G_REPAIRED


QUALIFICATION_DRIVER = (
    TOOLBOX_PATHS
    +  # The toolbox's own documented Columbus example: data/anselin.txt gives
    # the call pattern, data/wmat.txt the row-standardized weights.
    "load(fullfile('{toolbox}', 'data', 'anselin.dat'));\n"
    + "load(fullfile('{toolbox}', 'data', 'wmat.dat'));\n"
    + "W = sparse(wmat(:,1), wmat(:,2), wmat(:,3));\n"
    + "n = length(anselin);\n"
    + "y = anselin(:,1);\n"
    + "x = [ones(n,1) anselin(:,2:3)];\n"
    + "fprintf('NOBS %d\\n', n);\n"
    + "rand('state', {seed});\n"
    + "randn('state', {seed});\n"
    + "prior.novi = 1;\n"
    + "prior.eig = 0;\n"
    + "prior.lflag = 0;\n"
    + "res = sar_g(y, x, W, {ndraw}, {nomit}, prior);\n"
    + _dump_lines("res.pdraw", "tbq_rho.txt")
    + _dump_matrix("res.bdraw", "tbq_beta.txt")
    + _dump_lines("res.sdraw", "tbq_sige.txt")
    + "info.lflag = 0;\n"
    + "mle = sar(y, x, W, info);\n"
    + "fprintf('MLE_RHO %.6f\\n', mle.rho);\n"
    + "fprintf('MLE_RHO_SE %.6f\\n', abs(mle.rho/mle.tstat(end)));\n"
    + "prt(res);\n"
    + "fprintf('QUAL_OK\\n');\n"
)


def qualification_driver(
    toolbox: Path, workdir: Path, ndraw: int, nomit: int, seed: int
) -> str:
    return QUALIFICATION_DRIVER.format(
        toolbox=toolbox,
        shims=write_shims(workdir),
        ndraw=ndraw,
        nomit=nomit,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Data exchange
# ---------------------------------------------------------------------------


def write_ascii(dataset: dict, workdir: Path) -> None:
    """Export y / X / W as the ASCII files the ``.m`` side loads verbatim."""
    workdir.mkdir(parents=True, exist_ok=True)
    for key, name in (("y", "y.txt"), ("X", "X.txt"), ("W", "W.txt")):
        arr = np.asarray(dataset[key], dtype=float)
        np.savetxt(workdir / name, arr, fmt="%.17e")


def load_draws(
    workdir: Path, prefix: str, keys: tuple[str, ...] = ("rho", "beta", "sige")
) -> dict:
    """Read back the toolbox's posterior draws.

    ``beta`` comes back as ``(draws, k)`` (one row per draw, also for a single
    regressor), the scalar parameters as ``(draws,)``.
    """
    return {
        key: np.loadtxt(
            workdir / f"{prefix}_{key}.txt", ndmin=2 if key == "beta" else 1
        )
        for key in keys
    }
