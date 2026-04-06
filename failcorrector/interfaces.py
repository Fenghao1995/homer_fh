from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, Tuple


Action = Sequence[float]
Observation = Sequence[float]
ActionChunk = Sequence[Action]


@dataclass
class DeviationState:
    d_obs: float
    d_act: float
    delta_d_obs: float
    delta_d_act: float


class BasePolicy(Protocol):
    def act(self, obs_history: Sequence[Observation]) -> Tuple[ActionChunk, object]:
        """Return (action_chunk, latent_state)."""


class FIPEREvaluator(Protocol):
    def evaluate_observation_deviation(self, obs_history: Sequence[Observation]) -> float:
        ...

    def evaluate_action_deviation(
        self,
        obs_history: Sequence[Observation],
        candidate_actions: Sequence[ActionChunk],
    ) -> float:
        ...


class ResidualCorrector(Protocol):
    def residual(self, latent_state: object, deviation: DeviationState) -> ActionChunk:
        ...

    def gate(self, deviation: DeviationState) -> float:
        ...


class DreamZeroWorldModel(Protocol):
    def rollout(
        self,
        obs_history: Sequence[Observation],
        corrected_actions: ActionChunk,
        horizon: int,
    ) -> Tuple[Sequence[Observation], Sequence[Action]]:
        ...
