"""Smoke tests for the twin: it steps, `perturb` defaults are a no-op, and the action-column
convention the whole repository rests on is read out of the XML."""
from __future__ import annotations
import mujoco
import numpy as np
import pytest

from growbot_cerebellum.sim import GrowBotSim, ServoModel, BODIES, perturb, collect, CTRL_HZ, OBS_DIM, ACT_DIM
from growbot_cerebellum.servo_id import check_side_convention, sim_side_columns, RIGHT_COL, LEFT_COL


@pytest.mark.parametrize("body", sorted(BODIES))
def test_sim_steps_nan_free(body):
    sim = GrowBotSim(seed=0, body=body)
    o = sim.reset()
    assert o.shape == (OBS_DIM,)
    rng = np.random.default_rng(0)
    for _ in range(2 * CTRL_HZ):
        o = sim.step(rng.uniform(-0.5, 0.5, ACT_DIM))
        assert np.isfinite(o).all()
    assert np.abs(o[:3]).max() <= np.pi + 1e-6


def test_perturb_defaults_are_a_noop():
    # At defaults perturb() leaves every field as loaded except the two leg inertias, which it
    # re-derives from the box formula at the XML's own size -- the same numbers to ~1e-19
    # relative. That is why contact_friction.py's NOOP corner passes {"mass_scale": 1.0}: a
    # twin built with dr=None and one built through perturb() differ by exactly that
    # rounding, and a corner must be compared against the path it was built through.
    m0 = mujoco.MjModel.from_xml_path(str(BODIES["walk"]))
    m1 = mujoco.MjModel.from_xml_path(str(BODIES["walk"]))
    m2 = mujoco.MjModel.from_xml_path(str(BODIES["walk"]))
    perturb(m1); perturb(m2)
    for field in ("body_mass", "body_ipos", "geom_size", "geom_pos",
                  "geom_friction", "geom_condim", "actuator_gainprm", "actuator_biasprm"):
        assert np.array_equal(getattr(m0, field), getattr(m1, field)), field
    assert np.allclose(m1.body_inertia, m0.body_inertia, rtol=1e-12, atol=0)
    legs = {mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("right_leg", "left_leg")}
    assert set(np.flatnonzero(np.any(m1.body_inertia != m0.body_inertia, axis=1))) <= legs
    # and the path itself is deterministic: two perturbed models are bit-identical
    for field in ("body_mass", "body_inertia", "body_ipos", "geom_size", "geom_pos", "geom_friction"):
        assert np.array_equal(getattr(m1, field), getattr(m2, field)), field


def test_perturb_edits_only_what_it_is_told():
    m0 = mujoco.MjModel.from_xml_path(str(BODIES["walk"]))
    m1 = mujoco.MjModel.from_xml_path(str(BODIES["walk"]))
    perturb(m1, mass_scale=1.25, dcom=(0.03, 0.0, 0.0))
    base = mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_BODY, "base_body")
    assert np.allclose(m1.body_mass, m0.body_mass * 1.25)
    assert np.isclose(m1.body_ipos[base, 0] - m0.body_ipos[base, 0], 0.03)
    assert np.array_equal(m1.geom_friction, m0.geom_friction)
    assert np.array_equal(m1.actuator_gainprm, m0.actuator_gainprm)


def test_collect_returns_coherent_arrays():
    O, A, O2, D, M = collect(300, seed=0, episode_s=2.0)
    assert O.shape == (300, OBS_DIM) and A.shape == (300, ACT_DIM) and O2.shape == O.shape
    assert D.dtype == bool and len(D) == 300 and len(M) == 300
    assert np.isfinite(O).all() and np.isfinite(O2).all()
    # next_obs[t] is obs[t+1] wherever the transition is not a cut
    keep = ~D[:-1]
    assert np.array_equal(O2[:-1][keep], O[1:][keep])


def test_servo_model_delays_by_calls():
    # delay_ticks counts CALLS (AGENTS.md): two calls of a 2-tick delay still hold the reset pose
    s = ServoModel(delay_ticks=2, slew_rad_s=None, deadband=0.0)
    s.reset()
    target = np.array([0.5, -0.5], np.float32)
    first, second, third = (s(target, 1 / CTRL_HZ).copy() for _ in range(3))
    assert np.allclose(first, 0.0) and np.allclose(second, 0.0)
    assert np.allclose(third, target)


def test_action_columns_match_the_xml():
    ok, why = check_side_convention("walk")
    assert ok, why
    cols = sim_side_columns("walk")
    assert cols == {"right": RIGHT_COL, "left": LEFT_COL}
