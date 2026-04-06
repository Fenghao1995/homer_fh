from failcorrector.config import FailCorrectorConfig
from failcorrector.core import FailCorrectorController
from failcorrector.example_usage import (
    DummyBasePolicy,
    DummyCorrector,
    DummyDreamZero,
    DummyFIPER,
)
from failcorrector.fiper_bridge import FiperBridgeController


class StableWorldModel(DummyDreamZero):
    def rollout(self, obs_history, corrected_actions, horizon):
        # deterministic future for test stability
        last = list(obs_history[-1]) if obs_history else [0.0, 0.0, 0.0]
        future_obs = [last[:] for _ in range(horizon)]
        return future_obs, corrected_actions[:horizon]


def build_controller(obs_threshold=0.5, act_threshold=0.5, use_trend_gate=False):
    return FailCorrectorController(
        base_policy=DummyBasePolicy(),
        evaluator=DummyFIPER(),
        corrector=DummyCorrector(),
        world_model=StableWorldModel(),
        config=FailCorrectorConfig(
            obs_threshold=obs_threshold,
            act_threshold=act_threshold,
            use_trend_gate=use_trend_gate,
        ),
    )


def test_no_trigger_when_below_threshold():
    c = build_controller(obs_threshold=10, act_threshold=10)
    out = c.step([[0.1, 0.1, 0.1]])
    assert out.trigger is False
    assert out.corrected_actions == out.base_actions


def test_trigger_and_returns_chunk_shape():
    c = build_controller(obs_threshold=0.001, act_threshold=0.001)
    out = c.step([[1.0, 1.0, 1.0]])
    assert out.trigger is True
    assert len(out.corrected_actions) == len(out.base_actions)
    assert len(out.corrected_actions[0]) == len(out.base_actions[0])


def test_bridge_controller_is_runnable():
    b = FiperBridgeController(
        fiper_policy=DummyBasePolicy(),
        fiper_evaluator=DummyFIPER(),
        fiper_corrector=DummyCorrector(),
        fiper_world_model=StableWorldModel(),
        config=FailCorrectorConfig(obs_threshold=0.001, act_threshold=0.001, use_trend_gate=False),
    )
    out = b.step([[0.5, 0.2, 0.1]])
    assert out is not None
    assert hasattr(out, "corrected_actions")
