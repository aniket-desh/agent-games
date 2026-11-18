"""Offline model checks: fixture, recursion = closed form, Monte Carlo = recursion, predictor = exact enumeration."""
import numpy as np
import pytest

from flagswarm import theory as th


@pytest.mark.parametrize("self_contacts,want", [(True, 0.0188556892), (False, 0.0190012580)])
def test_fixture(self_contacts, want):
    N, zT, zR, q, L = 128, 1, 1, 0.5, 1280
    assert th.closed_form_loss(N, zT, zR, q, 1.0, L, 0, self_contacts) == pytest.approx(want, abs=1e-10)
    xb = th.mean_recursion(N, zT, zR, q, np.zeros(L), 0.5, self_contacts)
    xm = th.mean_recursion(N, zT, zR, q, np.ones(L), 0.5, self_contacts)
    rec = th.expected_truth_share(N, zT, zR, xb[-1]) - th.expected_truth_share(N, zT, zR, xm[-1])
    assert rec == pytest.approx(want, abs=1e-10)
    assert th.persistent_mute_limit(N, zT, zR) == 0.4921875


@pytest.mark.parametrize("self_contacts", [True, False])
@pytest.mark.parametrize("N,zT,zR,q,h,L,K", [(16, 1, 1, 0.5, 1.0, 50, 30), (32, 2, 3, 0.3, 0.5, 200, 0), (10, 3, 1, 0.8, 0.25, 7, 11)])
def test_recursion_matches_closed_form(self_contacts, N, zT, zR, q, h, L, K):
    hs = np.r_[np.full(L, h), np.zeros(K)]
    x0 = zT / (zT + zR)
    xb = th.mean_recursion(N, zT, zR, q, np.zeros(L + K), x0, self_contacts)
    xm = th.mean_recursion(N, zT, zR, q, hs, x0, self_contacts)
    rec = th.expected_truth_share(N, zT, zR, xb[-1]) - th.expected_truth_share(N, zT, zR, xm[-1])
    assert rec == pytest.approx(th.closed_form_loss(N, zT, zR, q, h, L, K, self_contacts), rel=1e-9, abs=1e-14)


@pytest.mark.parametrize("self_contacts", [True, False])
def test_monte_carlo_recovers_mean_response_and_recovery(self_contacts):
    """Bernoulli copying games: mute the truth witness for L slots, restore for K; readouts during and after."""
    g = th.PlantedGame(N=12, z_T=1, z_R=1, q=0.5, self_contacts=self_contacts)
    L, K = 240, 240
    h = np.r_[np.ones(L), np.zeros(K)]
    readouts = [60, L, L + 60, L + K]
    sim = th.simulate_planted(g, h, reps=40000, seed=1, readout_slots=readouts)
    xb = th.mean_recursion(g.N, g.z_T, g.z_R, g.q, np.zeros(L + K), 0.5, self_contacts)
    xm = th.mean_recursion(g.N, g.z_T, g.z_R, g.q, h, 0.5, self_contacts)
    for c, r in enumerate(readouts):
        pred_base = th.expected_truth_share(g.N, 1, 1, xb[r])
        pred_loss = pred_base - th.expected_truth_share(g.N, 1, 1, xm[r])
        base, loss = sim["Y_base"][:, c], sim["Y_base"][:, c] - sim["Y_muted"][:, c]
        assert abs(base.mean() - pred_base) < 4 * base.std() / np.sqrt(len(base))
        assert abs(loss.mean() - pred_loss) < 4 * loss.std() / np.sqrt(len(loss)) + 1e-12
    # recovery: the loss after restoration is smaller than at the end of the mute
    assert sim["Y_base"][:, 3].mean() - sim["Y_muted"][:, 3].mean() < sim["Y_base"][:, 1].mean() - sim["Y_muted"][:, 1].mean()


def test_h0_has_no_effect_under_coupled_randomness():
    g = th.PlantedGame(N=10, z_T=1, z_R=1, q=0.5, self_contacts=False)
    sim = th.simulate_planted(g, np.zeros(300), reps=500, seed=3)
    assert np.array_equal(sim["Y_base"], sim["Y_muted"])


def _tiny():
    rng = np.random.default_rng(7)
    N, M = 3, 7
    sched = th.sample_schedule(N, M, rng, self_contacts=False)
    p0 = np.array([0.9, 0.2, 0.55])
    gains = np.array([0.0, 0.6, 0.35])   # agent 0 is an ideal frozen witness (zero receiver gain)
    return N, sched, p0, gains


def test_forward_matches_exact_enumeration():
    N, sched, p0, gains = _tiny()
    assert th.forward(p0, sched, gains)[-1].mean() == pytest.approx(th.exact_enumeration_mean(p0, sched, gains), abs=1e-12)


def test_weak_and_full_mute_predictions_match_enumeration():
    N, sched, p0, gains = _tiny()
    base = th.exact_enumeration_mean(p0, sched, gains)
    d = th.single_drop_losses(p0, sched, gains)
    for k in range(len(sched)):
        keep = np.ones(len(sched), bool); keep[k] = False
        assert d[k] == pytest.approx(base - th.exact_enumeration_mean(p0, sched, gains, keep), abs=1e-12)
    for target in range(N):
        keep = sched[:, 0] != target
        assert th.full_mute_loss(p0, sched, gains, target) == pytest.approx(base - th.exact_enumeration_mean(p0, sched, gains, keep), abs=1e-12)


def test_temporal_closeness_by_hand():
    # 0->1 at slot 1, 1->2 at slot 2, 2->0 at slot 3 (N=3)
    sched = np.array([[0, 1], [1, 2], [2, 0]])
    tau = th.earliest_arrival(3, sched)
    assert tau[0, 1] == 1 and tau[0, 2] == 2 and tau[1, 2] == 2 and tau[2, 0] == 3
    assert tau[1, 0] == 3  # 1->2 (slot 2), 2->0 (slot 3)
    assert np.isinf(tau[2, 1])
    E = th.temporal_closeness(3, sched)
    assert E[0] == pytest.approx((3 / 1 + 3 / 2) / 2)
    assert E[2] == pytest.approx((3 / 3 + 0) / 2)
