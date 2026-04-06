from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .config import FailCorrectorConfig
from .core import CorrectionOutput, FailCorrectorController
from .interfaces import DeviationState


@dataclass
class FiperMethodMap:
    """### [FailCorrector-NEW] Flexible method mapping for real FIPER classes."""

    policy_act: str = "act"
    eval_obs: str = "evaluate_observation_deviation"
    eval_act: str = "evaluate_action_deviation"
    corr_residual: str = "residual"
    corr_gate: str = "gate"
    world_rollout: str = "rollout"


class _CallableWrapper:
    def __init__(self, obj: Any, fn_name: str):
        self._obj = obj
        self._fn = fn_name

    def __call__(self, *args, **kwargs):
        fn: Callable = getattr(self._obj, self._fn)
        return fn(*args, **kwargs)


class _PolicyAdapter:
    def __init__(self, policy: Any, method_map: FiperMethodMap):
        self._call = _CallableWrapper(policy, method_map.policy_act)

    def act(self, obs_history: Sequence[Sequence[float]]):
        return self._call(obs_history)


class _EvaluatorAdapter:
    def __init__(self, evaluator: Any, method_map: FiperMethodMap):
        self._obs = _CallableWrapper(evaluator, method_map.eval_obs)
        self._act = _CallableWrapper(evaluator, method_map.eval_act)

    def evaluate_observation_deviation(self, obs_history):
        return float(self._obs(obs_history))

    def evaluate_action_deviation(self, obs_history, candidate_actions):
        return float(self._act(obs_history, candidate_actions))


class _CorrectorAdapter:
    def __init__(self, corrector: Any, method_map: FiperMethodMap):
        self._residual = _CallableWrapper(corrector, method_map.corr_residual)
        self._gate = _CallableWrapper(corrector, method_map.corr_gate)

    def residual(self, latent_state: object, deviation: DeviationState):
        return self._residual(latent_state, deviation)

    def gate(self, deviation: DeviationState):
        return float(self._gate(deviation))


class _WorldAdapter:
    def __init__(self, world_model: Any, method_map: FiperMethodMap):
        self._rollout = _CallableWrapper(world_model, method_map.world_rollout)

    def rollout(self, obs_history, corrected_actions, horizon: int):
        return self._rollout(obs_history, corrected_actions, horizon)


class FiperBridgeController:
    """
    ### [FailCorrector-NEW] Bridge that wires FailCorrector into real FIPER objects
    without changing FIPER source method names.
    """

    def __init__(
        self,
        fiper_policy: Any,
        fiper_evaluator: Any,
        fiper_corrector: Any,
        fiper_world_model: Any,
        config: FailCorrectorConfig | None = None,
        method_map: FiperMethodMap | None = None,
    ):
        method_map = method_map or FiperMethodMap()
        self.controller = FailCorrectorController(
            base_policy=_PolicyAdapter(fiper_policy, method_map),
            evaluator=_EvaluatorAdapter(fiper_evaluator, method_map),
            corrector=_CorrectorAdapter(fiper_corrector, method_map),
            world_model=_WorldAdapter(fiper_world_model, method_map),
            config=config,
        )

    def step(self, obs_history) -> CorrectionOutput:
        return self.controller.step(obs_history)
