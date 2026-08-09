"""Rule 1 is not a convention here -- it is enforced."""

from __future__ import annotations

import numpy as np
import pytest

from environments.sealed import EnvironmentAccessError, seal


def test_the_five_public_calls_work_through_the_seal(any_env):
    api = seal(any_env)
    obs = api.reset(seed=2)
    assert api.observation_space().contains(obs)
    action = api.action_space().sample(np.random.default_rng(0))
    assert len(api.step(action)) == 5
    assert api.observation_space().contains(api.observe())


@pytest.mark.parametrize(
    "attribute",
    [
        "_victim",          # the YOLO model
        "_renderer",        # the renderer
        "_body",            # physics state
        "_world",           # the object-based scene graph
        "_baseline_bbox",   # ground-truth box
        "_baseline_conf",   # victim confidence
        "_clean",           # the internal image
        "render_human",     # the full-resolution frame
        "pop_telemetry",    # privileged logs
        "victim_report",    # direct access to the victim
        "target",
        "world",
    ],
)
def test_everything_else_is_blocked(any_env, attribute):
    api = seal(any_env)
    with pytest.raises(EnvironmentAccessError):
        getattr(api, attribute)


def test_agents_cannot_write_environment_state(any_env):
    api = seal(any_env)
    with pytest.raises(EnvironmentAccessError):
        api._baseline_conf = 0.0


def test_sealing_is_idempotent(any_env):
    api = seal(any_env)
    assert seal(api) is api
