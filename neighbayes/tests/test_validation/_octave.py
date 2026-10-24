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


def write_shims(workdir: Path) -> Path:
    """Write the harness shims into ``workdir/shims`` and return that dir."""
    shims = workdir / "shims"
    shims.mkdir(parents=True, exist_ok=True)
    (shims / "fminbnd.m").write_text(FMINBND_SHIM)
    (shims / "waitbar.m").write_text(WAITBAR_SHIM)
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


def load_draws(workdir: Path, prefix: str) -> dict:
    """Read back the toolbox's posterior draws as flat arrays."""
    out = {}
    for key, name in (
        ("rho", f"{prefix}_rho.txt"),
        ("beta", f"{prefix}_beta.txt"),
        ("sige", f"{prefix}_sige.txt"),
    ):
        arr = np.atleast_2d(np.loadtxt(workdir / name))
        out[key] = arr[0] if arr.shape[0] == 1 and key != "beta" else arr
    return out
