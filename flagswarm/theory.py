"""Offline neutral copying model: exact mean recursion, closed form, Monte Carlo, and schedule-level predictors.

Notation follows the implementation prompt.
  N agents; z_T frozen truth speakers, z_R frozen rival speakers, U = N - Z flexible agents (Z = z_T + z_R).
  One contact slot = one directed (speaker, listener) pair drawn uniformly from C ordered pairs:
    C = N(N-1) when self-contacts are excluded (the upstream engine), C = N^2 when allowed (Eq. 5 of the paper).
  A flexible listener copies the speaker's current label with probability q.
  x_t = mean truth fraction among flexible agents immediately before slot t.
  E Y_t = (z_T + U x_t) / N is the expected whole-group truth share.

Every number produced here is OFFLINE (no provider call). Records carry source="offline".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def n_pairs(N: int, self_contacts: bool) -> int:
    return N * N if self_contacts else N * (N - 1)


# ---------------------------------------------------------------------------------------------
# Exact mean dynamics for one frozen truth target muted with probability h_t
# ---------------------------------------------------------------------------------------------

def mean_recursion(N: int, z_T: int, z_R: int, q: float, h: np.ndarray, x0: float, self_contacts: bool) -> np.ndarray:
    """x_{t+1} = x_t + (q/C)[(z_T - h_t)(1 - x_t) - z_R x_t]; returns x_0..x_S for S = len(h).

    Flexible-to-flexible contacts cancel in the mean, so only frozen speakers move x.
    Works for any initial mean and any (non-contiguous) mute profile h_t in [0, 1].
    """
    C = n_pairs(N, self_contacts)
    x = np.empty(len(h) + 1)
    x[0] = x0
    for t, ht in enumerate(h):
        x[t + 1] = x[t] + (q / C) * ((z_T - ht) * (1.0 - x[t]) - z_R * x[t])
    return x


def expected_truth_share(N: int, z_T: int, z_R: int, x: np.ndarray | float) -> np.ndarray | float:
    return (z_T + (N - z_T - z_R) * np.asarray(x)) / N


def closed_form_loss(N: int, z_T: int, z_R: int, q: float, h: float, L: int, K: int, self_contacts: bool) -> float:
    """D(h,L,K) = (U/N) * h z_R / (Z(Z-h)) * (1 - r_h^L) * r_0^K, r_h = 1 - q(Z-h)/C.

    Starts from the unperturbed stationary mean x0 = z_T/Z; h held for L slots, then K slots restored.
    Requires Z > 1 (Z = h = 1 would leave no fixed point for the muted dynamics).
    """
    Z = z_T + z_R
    if Z <= 1:
        raise ValueError("closed form needs Z = z_T + z_R > 1")
    U = N - Z
    C = n_pairs(N, self_contacts)
    r_h = 1.0 - q * (Z - h) / C
    r_0 = 1.0 - q * Z / C
    return (U / N) * h * z_R / (Z * (Z - h)) * (1.0 - r_h**L) * r_0**K


def persistent_mute_limit(N: int, z_T: int, z_R: int, h: float = 1.0) -> float:
    Z = z_T + z_R
    return (N - Z) / N * h * z_R / (Z * (Z - h))


def late_horizon_rounds(N: int, q: float, Z: int) -> int:
    """T_late = ceil(3 (N-1) / (q (Z-1))) rounds: about three muted relaxation times (self-contacts excluded)."""
    if Z < 2:
        raise ValueError("T_late defined for Z >= 2")
    return math.ceil(3 * (N - 1) / (q * (Z - 1)))


# ---------------------------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------------------------

def sample_schedule(N: int, slots: int, rng: np.random.Generator, self_contacts: bool) -> np.ndarray:
    """(slots, 2) array of (speaker, listener). Self-contacts excluded: uniform distinct listener, as in the engine."""
    s = rng.integers(0, N, size=slots)
    if self_contacts:
        l = rng.integers(0, N, size=slots)
    else:
        l = rng.integers(0, N - 1, size=slots)
        l = l + (l >= s)
    return np.stack([s, l], axis=1)


# ---------------------------------------------------------------------------------------------
# Monte Carlo: planted binary copying game (frozen witnesses, neutral copying among flexible agents)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PlantedGame:
    N: int
    z_T: int
    z_R: int
    q: float
    self_contacts: bool

    @property
    def Z(self) -> int:
        return self.z_T + self.z_R

    @property
    def U(self) -> int:
        return self.N - self.Z


def simulate_planted(game: PlantedGame, h: np.ndarray, reps: int, seed: int, x0: float | None = None,
                     readout_slots: list[int] | None = None, coupled_h0: bool = False) -> dict[str, np.ndarray]:
    """Vectorized Monte Carlo over `reps` independent games. Agent 0 is the target truth witness.

    Agents 0..z_T-1 are frozen truth (label 1), z_T..Z-1 frozen rival (label 0), the rest flexible with
    initial labels iid Bernoulli(x0) (default: stationary mean z_T/Z). Each game draws its own schedule.
    The target's transmission at slot t is dropped with probability h[t] (independent per game and slot).

    Returns truth share Y at each readout slot for the muted branch and, sharing every random draw,
    for the unmuted baseline (paired). With coupled_h0, the baseline uses the same mask uniforms, so
    h = 0 reproduces it exactly.
    """
    rng = np.random.default_rng(seed)
    N, z_T, Z, q = game.N, game.z_T, game.Z, game.q
    S = len(h)
    x0 = z_T / Z if x0 is None else x0
    readout_slots = readout_slots or [S]
    labels0 = np.zeros((reps, N), dtype=np.int8)
    labels0[:, :z_T] = 1
    labels0[:, Z:] = (rng.random((reps, game.U)) < x0).astype(np.int8)
    a = labels0.copy()  # muted branch
    b = labels0.copy()  # baseline branch (shares schedule, copy coins)
    rows = np.arange(reps)
    out_a, out_b = [], []
    readout_set = {r: i for i, r in enumerate(readout_slots)}
    if 0 in readout_set:
        out_a.append(a.mean(1)); out_b.append(b.mean(1))
    for t in range(S):
        s = rng.integers(0, N, size=reps)
        if game.self_contacts:
            l = rng.integers(0, N, size=reps)
        else:
            l = rng.integers(0, N - 1, size=reps)
            l = l + (l >= s)
        copy = rng.random(reps) < q
        mask_u = rng.random(reps)
        dropped = (s == 0) & (mask_u < h[t])
        flexible = l >= Z
        do_a = flexible & copy & ~dropped
        do_b = flexible & copy
        a[rows[do_a], l[do_a]] = a[rows[do_a], s[do_a]]
        b[rows[do_b], l[do_b]] = b[rows[do_b], s[do_b]]
        if (t + 1) in readout_set:
            out_a.append(a.mean(1)); out_b.append(b.mean(1))
    return {"Y_muted": np.stack(out_a, 1), "Y_base": np.stack(out_b, 1), "readout_slots": np.array(readout_slots)}


# ---------------------------------------------------------------------------------------------
# Schedule-level neutral predictor (forward/backward) for a fixed contact sequence
# ---------------------------------------------------------------------------------------------

def forward(p0: np.ndarray, schedule: np.ndarray, gains: np.ndarray, delivered: np.ndarray | None = None) -> np.ndarray:
    """p_j <- (1 - q_j) p_j + q_j p_i at each delivered contact i -> j. Returns (M+1, N) trajectory.

    This is the exact expected label vector of the neutral copying model on the given schedule
    (expectation is linear in the coin flips), for any correlated initial state with means p0.
    """
    M = len(schedule)
    P = np.empty((M + 1, len(p0)))
    P[0] = p0
    for k, (i, j) in enumerate(schedule):
        P[k + 1] = P[k]
        if delivered is None or delivered[k]:
            P[k + 1, j] = (1 - gains[j]) * P[k, j] + gains[j] * P[k, i]
    return P


def backward_weights(N: int, schedule: np.ndarray, gains: np.ndarray, terminal: np.ndarray | None = None) -> np.ndarray:
    """v_M = 1/N (terminal vote weights), v_k = B_k^T v_{k+1}. Returns (M+1, N)."""
    M = len(schedule)
    V = np.empty((M + 1, N))
    V[M] = np.full(N, 1.0 / N) if terminal is None else terminal
    for k in range(M - 1, -1, -1):
        i, j = schedule[k]
        v = V[k + 1].copy()
        # B_k maps p -> p with row j replaced by (1-q_j) e_j + q_j e_i; B^T v moves weight q_j v_j from j to i.
        v[i] += gains[j] * V[k + 1, j]
        v[j] -= gains[j] * V[k + 1, j]
        V[k] = v
    return V


def single_drop_losses(p0: np.ndarray, schedule: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Predicted terminal truth-share loss from dropping only contact k: q_j (p_{k,i} - p_{k,j}) v_{k+1,j}."""
    P = forward(p0, schedule, gains)
    V = backward_weights(len(p0), schedule, gains)
    i, j = schedule[:, 0], schedule[:, 1]
    k = np.arange(len(schedule))
    return gains[j] * (P[k, i] - P[k, j]) * V[k + 1, j]


def susceptibility_by_sender(p0: np.ndarray, schedule: np.ndarray, gains: np.ndarray, window: tuple[int, int] | None = None) -> np.ndarray:
    """Sum of single-drop losses by sender (the weak-response ranking chi_i), optionally restricted to slots in window."""
    d = single_drop_losses(p0, schedule, gains)
    sel = np.ones(len(schedule), bool) if window is None else (np.arange(len(schedule)) >= window[0]) & (np.arange(len(schedule)) < window[1])
    out = np.zeros(len(p0))
    np.add.at(out, schedule[sel, 0], d[sel])
    return out


def full_mute_loss(p0: np.ndarray, schedule: np.ndarray, gains: np.ndarray, target: int, window: tuple[int, int] | None = None) -> float:
    """Finite intervention: replace every affected update with the identity; loss in terminal mean truth share."""
    k = np.arange(len(schedule))
    affected = schedule[:, 0] == target
    if window is not None:
        affected &= (k >= window[0]) & (k < window[1])
    base = forward(p0, schedule, gains)[-1].mean()
    muted = forward(p0, schedule, gains, delivered=~affected)[-1].mean()
    return float(base - muted)


def exact_enumeration_mean(p0: np.ndarray, schedule: np.ndarray, gains: np.ndarray, delivered: np.ndarray | None = None) -> float:
    """Brute-force expected terminal truth share over all initial labels and copy coins (tiny cases only)."""
    N, M = len(p0), len(schedule)
    total = 0.0
    for init in range(2 ** N):
        labels0 = np.array([(init >> a) & 1 for a in range(N)])
        w0 = float(np.prod(np.where(labels0 == 1, p0, 1 - p0)))
        if w0 == 0:
            continue
        for coins in range(2 ** M):
            labels = labels0.copy()
            w = w0
            for k, (i, j) in enumerate(schedule):
                c = (coins >> k) & 1
                w *= gains[j] if c else (1 - gains[j])
                if c and (delivered is None or delivered[k]):
                    labels[j] = labels[i]
            total += w * labels.mean()
    return total


# ---------------------------------------------------------------------------------------------
# Temporal closeness benchmark (paper Table 5 comparator)
# ---------------------------------------------------------------------------------------------

def earliest_arrival(N: int, schedule: np.ndarray) -> np.ndarray:
    """tau[i, j] = earliest slot count (1-based) by which information from i can reach j along time-respecting
    contacts starting at initialization; inf when unreachable; tau[i, i] = 0."""
    tau = np.full((N, N), np.inf)
    np.fill_diagonal(tau, 0.0)
    for k, (s, l) in enumerate(schedule, start=1):
        # Anyone whose information has reached s (tau[:, s] < k) now reaches l at time k.
        reach = tau[:, s] < k
        tau[reach, l] = np.minimum(tau[reach, l], k)
    return tau


def temporal_closeness(N: int, schedule: np.ndarray) -> np.ndarray:
    """E_i = (N-1)^{-1} sum_{j != i} N / tau_ij; unreachable peers contribute zero."""
    tau = earliest_arrival(N, schedule)
    with np.errstate(divide="ignore"):
        inv = np.where(np.isfinite(tau) & (tau > 0), N / tau, 0.0)
    np.fill_diagonal(inv, 0.0)
    return inv.sum(1) / (N - 1)
