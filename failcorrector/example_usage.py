"""Integration example for FailCorrector and bridge mode."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .config import FailCorrectorConfig
from .fiper_bridge import FiperBridgeController, FiperMethodMap
from .interfaces import ActionChunk, DeviationState, Observation


@dataclass
class DummyBasePolicy:
    chunk_len: int = 4
    act_dim: int = 3

    def act(self, obs_history: Sequence[Observation]):
        chunk = [[0.1 for _ in range(self.act_dim)] for _ in range(self.chunk_len)]
        latent = {"h": sum(sum(o) for o in obs_history)}
        return chunk, latent


@dataclass
class DummyFIPER:
    def evaluate_observation_deviation(self, obs_history: Sequence[Observation]) -> float:
        if not obs_history:
            return 0.0
        return abs(sum(obs_history[-1])) * 0.05

    def evaluate_action_deviation(self, obs_history, candidate_actions):
        if not candidate_actions:
            return 0.0
        seq = candidate_actions[0]
        return sum(abs(v) for a in seq for v in a) * 0.02


@dataclass
class DummyCorrector:
    def residual(self, latent_state: object, deviation: DeviationState) -> ActionChunk:
        scale = min(1.0, 0.2 * (deviation.d_obs + deviation.d_act))
        return [[scale, -scale, scale] for _ in range(4)]

    def gate(self, deviation: DeviationState) -> float:
        return max(0.0, min(1.0, deviation.d_obs + deviation.d_act))


@dataclass
class DummyDreamZero:
    def rollout(self, obs_history, corrected_actions, horizon):
        future_obs = []
        last = list(obs_history[-1]) if obs_history else [0.0, 0.0, 0.0]
        for _ in range(horizon):
            noise = (random.random() - 0.5) * 0.02
            last = [v + noise for v in last]
            future_obs.append(last[:])
        return future_obs, corrected_actions[:horizon]


def run_bridge_example() -> None:
    # ### [FailCorrector-NEW] Demonstrates no-source-change FIPER integration via bridge.
    controller = FiperBridgeController(
        fiper_policy=DummyBasePolicy(),
        fiper_evaluator=DummyFIPER(),
        fiper_corrector=DummyCorrector(),
        fiper_world_model=DummyDreamZero(),
        config=FailCorrectorConfig(obs_threshold=0.01, act_threshold=0.01, use_trend_gate=False),
        method_map=FiperMethodMap(),
    )

    obs_hist = [[0.2, 0.1, 0.3], [0.3, -0.2, 0.4]]
    out = controller.step(obs_hist)
    print("trigger:", out.trigger)
    print("gate:", round(out.gate, 4))
    print("objective:", round(out.objective, 4))


if __name__ == "__main__":
    run_bridge_example()
