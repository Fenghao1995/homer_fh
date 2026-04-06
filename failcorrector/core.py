from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Sequence

from .config import FailCorrectorConfig
from .interfaces import (
    Action,
    ActionChunk,
    BasePolicy,
    DeviationState,
    DreamZeroWorldModel,
    FIPEREvaluator,
    Observation,
    ResidualCorrector,
)


@dataclass
class CorrectionOutput:
    base_actions: ActionChunk
    corrected_actions: ActionChunk
    deviation: DeviationState
    gate: float
    trigger: bool
    objective: float


class FailCorrectorController:
    """
    ### [FailCorrector-NEW] Deviation-guided closed-loop correction.

    Trigger: hard AND on observation/action deviation + optional trend gate.
    Correction: gated residual action.
    Closed loop: DreamZero rollout + FIPER re-evaluation + fallback.
    """

    def __init__(
        self,
        base_policy: BasePolicy,
        evaluator: FIPEREvaluator,
        corrector: ResidualCorrector,
        world_model: DreamZeroWorldModel,
        config: FailCorrectorConfig | None = None,
    ):
        self.base_policy = base_policy
        self.evaluator = evaluator
        self.corrector = corrector
        self.world_model = world_model
        self.cfg = config or FailCorrectorConfig()
        self.obs_hist: Deque[float] = deque(maxlen=self.cfg.sliding_window)
        self.act_hist: Deque[float] = deque(maxlen=self.cfg.sliding_window)
        self.prev_obs_sum = 0.0
        self.prev_act_sum = 0.0
        self.prev_corrected_actions: List[Action] | None = None

    def step(self, obs_history: Sequence[Observation]) -> CorrectionOutput:
        base_actions, latent = self.base_policy.act(obs_history)

        d_obs_inst = self.evaluator.evaluate_observation_deviation(obs_history)
        d_act_inst = self.evaluator.evaluate_action_deviation(obs_history, [base_actions])

        self.obs_hist.append(d_obs_inst)
        self.act_hist.append(d_act_inst)

        s_obs = sum(self.obs_hist)
        s_act = sum(self.act_hist)
        d_obs_delta = s_obs - self.prev_obs_sum
        d_act_delta = s_act - self.prev_act_sum
        deviation = DeviationState(s_obs, s_act, d_obs_delta, d_act_delta)

        trigger = self._trigger(deviation)
        if not trigger:
            self.prev_obs_sum, self.prev_act_sum = s_obs, s_act
            return CorrectionOutput(
                base_actions=base_actions,
                corrected_actions=base_actions,
                deviation=deviation,
                gate=0.0,
                trigger=False,
                objective=0.0,
            )

        gate = self.corrector.gate(deviation)
        residual = self.corrector.residual(latent, deviation)
        corrected = self._blend(base_actions, residual, gate)

        corr_future_obs, corr_future_act = self.world_model.rollout(
            obs_history=obs_history,
            corrected_actions=corrected,
            horizon=self.cfg.horizon,
        )
        corr_obj = self._future_objective(base_actions, corrected, corr_future_obs, corr_future_act)

        # ### [FailCorrector-NEW] Baseline future for fair comparison.
        base_future_obs, base_future_act = self.world_model.rollout(
            obs_history=obs_history,
            corrected_actions=base_actions,
            horizon=self.cfg.horizon,
        )
        base_obj = self._future_objective(base_actions, base_actions, base_future_obs, base_future_act)

        if corr_obj > base_obj:
            corrected = base_actions
            gate = 0.0
            objective = base_obj
        else:
            objective = corr_obj

        self.prev_obs_sum, self.prev_act_sum = s_obs, s_act
        self.prev_corrected_actions = [list(a) for a in corrected]
        return CorrectionOutput(
            base_actions=base_actions,
            corrected_actions=corrected,
            deviation=deviation,
            gate=gate,
            trigger=True,
            objective=objective,
        )

    def _trigger(self, deviation: DeviationState) -> bool:
        hard = (deviation.d_obs > self.cfg.obs_threshold) and (
            deviation.d_act > self.cfg.act_threshold
        )
        if not self.cfg.use_trend_gate:
            return hard
        return hard and (
            deviation.delta_d_obs > self.cfg.obs_delta_threshold
            or deviation.delta_d_act > self.cfg.act_delta_threshold
        )

    def _blend(self, base: ActionChunk, residual: ActionChunk, gate: float) -> List[List[float]]:
        out: List[List[float]] = []
        for b, r in zip(base, residual):
            out.append([(bi + gate * ri) for bi, ri in zip(b, r)])
        return out

    def _future_objective(
        self,
        base_actions: ActionChunk,
        corrected_actions: ActionChunk,
        future_obs: Sequence[Observation],
        future_actions: Sequence[Action],
    ) -> float:
        obs_loss = 0.0
        act_loss = 0.0

        # ### [FailCorrector-NEW] Re-evaluate future with FIPER metrics.
        for i in range(1, min(self.cfg.horizon, len(future_obs)) + 1):
            d_o = self.evaluator.evaluate_observation_deviation(future_obs[:i])
            obs_loss += max(0.0, d_o)

        for i in range(1, min(self.cfg.horizon, len(future_actions)) + 1):
            d_a = self.evaluator.evaluate_action_deviation(future_obs[:i], [future_actions[:i]])
            act_loss += max(0.0, d_a)

        anchor = 0.0
        for b, c in zip(base_actions, corrected_actions):
            anchor += sum((ci - bi) ** 2 for bi, ci in zip(b, c))

        smooth = 0.0
        if self.prev_corrected_actions is not None:
            for prev, cur in zip(self.prev_corrected_actions, corrected_actions):
                smooth += sum((ci - pi) ** 2 for pi, ci in zip(prev, cur))

        return (
            self.cfg.lambda_obs * obs_loss
            + self.cfg.lambda_act * act_loss
            + self.cfg.lambda_anchor * anchor
            + self.cfg.lambda_smooth * smooth
        )
