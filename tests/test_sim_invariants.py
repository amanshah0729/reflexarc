"""Regressions against simulator bugs that produce wrong numbers, not errors.

Each test here corresponds to a specific way this stack was observed to fail
silently. They are slow because they need a real LIBERO scene; that is the
point, since none of them can be reproduced against a mock.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def scene():
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.tasks[0]
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder,
                        task.bddl_file)
    init = torch.load(
        os.path.join(get_libero_path("init_states"), task.problem_folder,
                     task.init_states_file), weights_only=False)
    np.random.seed(0)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)
    env.reset()
    env.set_init_state(init[0])
    yield env
    env.close()


def test_mass_scaling_does_not_move_the_scene(scene):
    """M1: `mj_setConst` computes constants at qpos0 and uses mjData as scratch.

    Applying it naively teleports every object from the table to z = 0 and the
    episode still runs, reporting an ordinary success rate for a scrambled
    scene. Asserted immediately after the mutation -- re-applying the init
    state afterwards repairs the damage and hides the bug.
    """
    from reflexarc.disturb import MassScale, free_bodies
    from reflexarc.sense import _unwrap

    _, data = _unwrap(scene.sim)
    watch = free_bodies(scene.sim)
    before = np.array([data.xpos[b][2] for b in watch])
    assert (before > 0.5).all(), "objects should start on the table, not at the origin"

    MassScale(100.0).apply(scene.sim)

    after = np.array([data.xpos[b][2] for b in watch])
    assert np.abs(after - before).max() < 1e-6, (
        f"mass scaling moved objects by up to "
        f"{np.abs(after - before).max()*1000:.1f} mm"
    )


def test_mass_scaling_actually_changes_mass(scene):
    """The mirror of the test above: a fix that no-ops would also pass it."""
    from reflexarc.disturb import MassScale, free_bodies
    from reflexarc.sense import _unwrap

    model, _ = _unwrap(scene.sim)
    watch = free_bodies(scene.sim)
    before = np.array([model.body_mass[b] for b in watch])
    MassScale(10.0).apply(scene.sim)
    after = np.array([model.body_mass[b] for b in watch])
    assert np.allclose(after, before * 10.0)


def test_free_bodies_excludes_articulated_furniture(scene):
    """Cabinet drawers weigh 3 kg and are named like objects, but hinged parts
    are task geometry rather than things the robot picks up."""
    import mujoco

    from reflexarc.disturb import free_bodies
    from reflexarc.sense import _unwrap

    model, _ = _unwrap(scene.sim)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
             for b in free_bodies(scene.sim)]
    assert names, "no free bodies found"
    assert not any("cabinet" in n or "stove" in n for n in names), names


def test_finger_geoms_resolve_shafts_and_pads(scene):
    """A tactile sensor that silently reads nothing looks like a stable grasp."""
    from reflexarc.sense import FingerGeoms

    f = FingerGeoms.resolve(scene.sim)
    assert f.left and f.right and f.pads
    assert set(f.pads) <= set(f.all)


def test_zero_magnitude_impulse_is_an_exact_noop(scene):
    """Every sweep gets its control from the same code path as its treatment."""
    from reflexarc.disturb import Impulse
    from reflexarc.sense import FingerGeoms, _unwrap

    _, data = _unwrap(scene.sim)
    fingers = FingerGeoms.resolve(scene.sim)
    imp = Impulse(magnitude=0.0)
    imp.reset(scene.sim)
    for step in range(5):
        imp.update(scene.sim, fingers, step)
    assert np.abs(data.xfrc_applied).max() == 0.0
    assert imp.fired_at is None


def test_friction_perturbation_actually_lowers_contact_friction(scene):
    """MuJoCo combines a contacting pair's friction by elementwise maximum.

    Scaling only the gripper therefore cannot push the effective coefficient
    below the object's own value, and a friction ladder built that way reads
    as total invariance -- measured, a 50x reduction moved success 0 points.
    """
    from reflexarc.disturb import ContactFriction
    from reflexarc.sense import FingerGeoms, _unwrap

    model, _ = _unwrap(scene.sim)
    assert int(model.npair) == 0, "explicit pairs would change the mixing rule"

    fingers = FingerGeoms.resolve(scene.sim)
    before = ContactFriction(1.0).effective(scene.sim, fingers)
    after = ContactFriction(0.05).apply(scene.sim, fingers)
    assert after < before * 0.2, (
        f"effective friction {after:.3f} vs {before:.3f}: the perturbation is "
        "being clamped by the geom it does not touch"
    )
