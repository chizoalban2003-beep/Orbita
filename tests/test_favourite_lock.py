"""Directional favourite-lock force — the sign-gated re-spec of scalar drag."""
import numpy as np

from orbita.forces import favourite_lock_force

FAV = np.array([5.0, 0.0])          # home well


def test_off_when_strength_zero():
    q = np.array([1.0, 0.5])
    v = np.array([-1.0, 0.3])
    assert np.allclose(favourite_lock_force(q, v, FAV, 0.0), [0.0, 0.0])


def test_no_force_when_advancing_toward_favourite():
    # velocity points toward the favourite well => nothing to resist
    q = np.array([0.0, 0.0])
    v = np.array([1.0, 0.0])                     # heading to +x (home)
    assert np.allclose(favourite_lock_force(q, v, FAV, 0.8), [0.0, 0.0])


def test_pushes_back_when_retreating():
    # velocity points away from the favourite well => force pushes back toward it
    q = np.array([0.0, 0.0])
    v = np.array([-1.0, 0.0])                    # heading to −x (away from home)
    f = favourite_lock_force(q, v, FAV, 0.8)
    assert f[0] > 0.0 and abs(f[1]) < 1e-9       # pushed back toward +x


def test_resists_drift_toward_the_draw():
    # from an off-axis lead, drifting toward the central draw well [0,5]
    # increases distance from the home well => the retreat is resisted.
    q = np.array([3.0, 1.0])
    v = np.array([0.0, 5.0]) - q                 # heading toward the draw well
    f = favourite_lock_force(q, v, FAV, 0.8)
    assert (FAV - q) @ f > 0.0                   # force points back toward home
    assert v @ f < 0.0                           # and opposes the retreat


def test_asymmetric_under_velocity_reversal():
    # the whole point: a linear tensor would give |F(+v)| == |F(−v)|; this does not
    q = np.array([0.0, 0.0])
    v = np.array([1.0, 0.2])
    f_fwd = favourite_lock_force(q, v, FAV, 0.8)
    f_rev = favourite_lock_force(q, -v, FAV, 0.8)
    assert np.linalg.norm(f_fwd) < np.linalg.norm(f_rev)


def test_batched_matches_single():
    q = np.array([[0.0, 0.0], [1.0, -0.5]])
    v = np.array([[-1.0, 0.0], [0.5, 0.5]])
    batch = favourite_lock_force(q, v, FAV, 0.8)
    for i in range(2):
        assert np.allclose(batch[i], favourite_lock_force(q[i], v[i], FAV, 0.8))
