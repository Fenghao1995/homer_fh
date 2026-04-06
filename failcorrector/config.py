from dataclasses import dataclass


@dataclass
class FailCorrectorConfig:
    """Configuration for deviation-guided closed-loop correction."""

    obs_threshold: float = 0.15
    act_threshold: float = 0.15
    obs_delta_threshold: float = 0.0
    act_delta_threshold: float = 0.0
    use_trend_gate: bool = True

    sliding_window: int = 4
    horizon: int = 5

    lambda_obs: float = 1.0
    lambda_act: float = 1.0
    lambda_anchor: float = 0.2
    lambda_smooth: float = 0.05
