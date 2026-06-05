#!/usr/bin/env python3
"""
epidemiology.py
===============
Analytical model for accelerated worm spread with multiple independent
initial infection seeds (patient zeros).

When an adversary can simultaneously seed the malware in *n₀* distinct
vessels rather than a single one, the expected time to global spread follows
a sublinear power-law relationship:

    T_n = T₁ / n₀^α

where:
  - T₁   is the baseline time-to-global-spread for a single seed (days)
  - n₀   is the number of independent initial seeds
  - α    is the sublinear scaling exponent (0 < α < 1)

The exponent α captures the diminishing returns from additional seeds: each
additional patient zero reduces spread time, but maritime chokepoints impose
hard geographic bottlenecks that no amount of parallel seeding can eliminate.

Usage
-----
    python epidemiology.py

Or import and call :func:`compute_infection_time` directly.
"""


def compute_infection_time(T1: float, alpha: float, n0: int) -> float:
    """
    Compute the expected global spread time for *n0* simultaneous seeds.

    Parameters
    ----------
    T1 : float
        Single-seed baseline spread time (days).
    alpha : float
        Sublinear scaling exponent (must be > 0).
    n0 : int
        Number of independent initial seeds (must be >= 1).

    Returns
    -------
    float  Spread time in the same units as *T1*.

    Raises
    ------
    ValueError  if *n0* < 1 or *alpha* <= 0.
    """
    if n0 < 1:
        raise ValueError("n0 must be a positive integer.")
    if alpha <= 0.0:
        raise ValueError("alpha must be a positive number.")
    return T1 / (n0 ** alpha)


if __name__ == "__main__":
    # Example: single-seed baseline of 51 days, α = 0.25
    # with n0 = 315 simultaneous seeds (one per vulnerable vessel
    # present in Jebel Ali at the start of the simulation window).
    T1    = 51       # days (single-seed global spread)
    alpha = 0.25     # sublinear exponent
    n0    = 315      # simultaneous seeds

    Tn = compute_infection_time(T1, alpha, n0)
    print(f"T1 = {T1} days, alpha = {alpha}, n0 = {n0}")
    print(f"Estimated spread time T_n ≈ {Tn:.2f} days")
