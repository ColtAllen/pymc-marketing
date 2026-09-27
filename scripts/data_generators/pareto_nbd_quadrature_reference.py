#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Pre-compute high-precision Pareto/NBD log-likelihood and gradient reference values.

Two independent mpmath routes are evaluated at 40 significant digits:

- ``logp_ref``: the closed form with two Gaussian hypergeometric functions, as implemented
  in :meth:`pymc_marketing.clv.distributions.ParetoNBD.logp`. This is the primary reference:
  at 40 digits the closed form is an exact identity, whereas numerical integration of the
  dropout term becomes unreliable for heavy buyers (large ``x``), where the integrand decays
  over hundreds of orders of magnitude just after ``t_x``.
- ``logp_ref_quad``: the integral form, equation (22) of Fader & Hardie (2005), with the
  dropout integral evaluated by ``mpmath.quad`` on 40 geometrically spaced sub-intervals.
  It is a cross-check: the two routes must agree to a relative 1e-9 for ``x <= 80``; for
  larger ``x`` the discrepancy is only reported.

``grad_ref`` holds the derivatives of ``logp_ref`` with respect to ``(r, alpha, s, beta)``
from ``mpmath.diff``.

mpmath is not a declared dependency; the output ``.npz`` is committed and used by
``tests/clv/test_pareto_nbd_quadrature.py``.

Source:
- Fader, P. S. & Hardie, B. G. S. (2005), "A Note on Deriving the Pareto/NBD Model and Related Expressions"
  http://brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf
"""

import os
import time

import mpmath as mp
import numpy as np

mp.mp.dps = 40

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "tests",
    "clv",
    "data",
    "pareto_nbd_quadrature_reference.npz",
)

# (r, alpha, s, beta)
PARAMS = [
    (0.5534, 10.5802, 0.6061, 11.6562),  # CDNOW estimates
    (0.5534, 11.6562, 0.6061, 10.5802),  # alpha and beta swapped
    (0.5534, 10.58, 0.6061, 10.58),  # alpha == beta
    (2.0, 1.0, 0.5, 50.0),  # pt.hyp2f1 returns NaN here for heavy buyers
    (0.2, 0.5, 3.0, 0.1),  # steep integrand near tau = 0
    (0.5, 200.0, 0.5, 200.0),  # large scale parameters
]
T_VALUES = [12.0, 52.0, 365.0]
X_VALUES = [1.0, 5.0, 20.0, 80.0, 200.0, 500.0]
TX_FRACTIONS = [0.0, 0.1, 0.5, 0.9]


def build_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return ``params`` of shape (n, 4) and ``values`` of shape (n, 3) holding ``(t_x, x, T)``."""
    params, values = [], []
    for theta in PARAMS:
        for T in T_VALUES:
            # Non-repeat customers have t_x == 0 by construction
            params.append(theta)
            values.append((0.0, 0.0, T))
            for x in X_VALUES:
                for t_x in [*(f * T for f in TX_FRACTIONS), T - 1e-3, T]:
                    params.append(theta)
                    values.append((t_x, x, T))
    return np.array(params, dtype=float), np.array(values, dtype=float)


def log_a1(r, alpha, s, beta, x):
    """Common prefactor: log Gamma(r+x) - log Gamma(r) + r log alpha + s log beta."""
    return mp.loggamma(r + x) - mp.loggamma(r) + r * mp.log(alpha) + s * mp.log(beta)


def logp_integral(r, alpha, s, beta, t_x, x, T):
    """Log-likelihood from the integral form, equation (22) of Fader & Hardie (2005)."""
    r, alpha, s, beta, t_x, x, T = map(mp.mpf, (r, alpha, s, beta, t_x, x, T))
    alive = (alpha + T) ** (-(r + x)) * (beta + T) ** (-s)

    def integrand(tau):
        return (alpha + tau) ** (-(r + x)) * (beta + tau) ** (-(s + 1))

    if t_x < T:
        # Geometric breakpoints in (alpha + tau) so the steep region after t_x is resolved
        ratio = (alpha + T) / (alpha + t_x)
        breakpoints = [
            t_x,
            *((alpha + t_x) * ratio ** (k / 40) - alpha for k in range(1, 40)),
            T,
        ]
        integral = mp.quad(integrand, breakpoints)
    else:
        integral = mp.mpf(0)
    return log_a1(r, alpha, s, beta, x) + mp.log(alive + s * integral)


def logp_hyp2f1(r, alpha, s, beta, t_x, x, T):
    """Log-likelihood from the closed form used by ``ParetoNBD.logp``."""
    r, alpha, s, beta, t_x, x, T = map(mp.mpf, (r, alpha, s, beta, t_x, x, T))
    rsx = r + s + x
    rx = r + x
    if alpha >= beta:
        larger, diff = alpha, alpha - beta
        b1, b2 = s + 1, s
    else:
        larger, diff = beta, beta - alpha
        b1, b2 = rx, rx + 1
    term1 = mp.hyp2f1(rsx, b1, rsx + 1, diff / (larger + t_x)) / (larger + t_x) ** rsx
    term2 = mp.hyp2f1(rsx, b2, rsx + 1, diff / (larger + T)) / (larger + T) ** rsx
    bracket = (s / rsx) * term1 + (rx / rsx) * term2
    return log_a1(r, alpha, s, beta, x) + mp.log(bracket)


def gradient(r, alpha, s, beta, t_x, x, T):
    """Partial derivatives of the closed-form log-likelihood with respect to r, alpha, s, beta."""

    def wrt(index):
        def f(v):
            theta = [mp.mpf(r), mp.mpf(alpha), mp.mpf(s), mp.mpf(beta)]
            theta[index] = v
            return logp_hyp2f1(*theta, t_x, x, T)

        return mp.diff(f, [r, alpha, s, beta][index])

    return [wrt(i) for i in range(4)]


def main() -> None:
    """Compute and save the reference values."""
    params, values = build_grid()
    n = len(params)
    logp_ref = np.empty(n)
    logp_ref_quad = np.empty(n)
    grad_ref = np.empty((n, 4))
    worst_heavy = mp.mpf(0)

    start = time.perf_counter()
    for i, ((r, alpha, s, beta), (t_x, x, T)) in enumerate(
        zip(params, values, strict=True)
    ):
        lp_2f1 = logp_hyp2f1(r, alpha, s, beta, t_x, x, T)
        lp_int = logp_integral(r, alpha, s, beta, t_x, x, T)
        discrepancy = abs(lp_int - lp_2f1) / max(abs(lp_2f1), 1)
        if x <= 80:
            if discrepancy > mp.mpf("1e-9"):
                raise RuntimeError(
                    f"Integral and hyp2f1 references disagree at row {i}: "
                    f"{(r, alpha, s, beta, t_x, x, T)} -> {lp_int} vs {lp_2f1}"
                )
        else:
            worst_heavy = max(worst_heavy, discrepancy)
        logp_ref[i] = float(lp_2f1)
        logp_ref_quad[i] = float(lp_int)
        grad_ref[i] = [float(g) for g in gradient(r, alpha, s, beta, t_x, x, T)]
        if (i + 1) % 100 == 0:
            print(
                f"  {i + 1}/{n} rows, {time.perf_counter() - start:.0f}s elapsed",
                flush=True,
            )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        params=params,
        values=values,
        logp_ref=logp_ref,
        logp_ref_quad=logp_ref_quad,
        grad_ref=grad_ref,
    )
    print(
        "Largest relative integral-vs-hyp2f1 discrepancy for x > 80: "
        f"{mp.nstr(worst_heavy, 3)}"
    )
    print(f"Saved {n} rows to {os.path.normpath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
