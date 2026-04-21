"""Central configuration for the Part6-Core simulation and analysis pipeline.

This module concentrates parameters referenced by the paper so that the codebase
has one reproducible source of truth.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).resolve().parent
SIMULATION_RESULTS_DIR = BASE_DIR / "simulation_results"
ANALYSIS_RESULTS_DIR = BASE_DIR / "analysis_results"
PAPER_FIGURES_DIR = BASE_DIR / "paper_figures"


@dataclass(frozen=True)
class SimulationConfig:
    """Simulation defaults aligned with paper Section 7.4.2."""

    rider_num: int = 100
    run_len: int = 3600
    num_runs: int = 10
    one_day: int = 120
    one_hour: int = 5
    order_weight: float = 0.3
    order_bf: int = 3
    max_orders: int = 5
    step_move_distance: int = 30
    choose_order_step_interval: int = 15
    business_district_num: int = 10
    seed_base: int = 42


@dataclass(frozen=True)
class ThresholdConfig:
    """Thresholds used by the three-layer analysis framework."""

    low_involution: float = 30.0
    high_involution: float = 60.0


@dataclass(frozen=True)
class MetricConfig:
    """Metric parameters used by individual and platform calculators."""

    crra_eta: float = 0.5
    individual_Tw: int = 10
    individual_Tin: int = 10
    individual_Ts: int = 10
    platform_Tf: int = 10
    platform_Th: int = 10
    platform_Rh: int = 20
    platform_Tei: int = 10
    platform_Rd: int = 20


@dataclass(frozen=True)
class OrderDistributionConfig:
    """Gaussian mixture parameters used in Section 7.4.2 order generation."""

    amplitudes: List[float] = field(
        default_factory=lambda: [314.2, 188.3, 95.56, 22.9, 48.67]
    )
    centers: List[float] = field(
        default_factory=lambda: [172.5, 281.5, 315.5, 228.9, 267.1]
    )
    widths: List[float] = field(
        default_factory=lambda: [4.645, 1.559, 10.69, 167.7, 13.1]
    )


@dataclass(frozen=True)
class LLMConfig:
    """LLM defaults aligned with the reproducibility requirement."""

    model: str = "DeepSeek-R1-Distill-Qwen-32B"
    temperature: float = 0.0
    base_url_env: str = "DASHSCOPE_BASE_URL"
    api_key_env: str = "DASHSCOPE_API_KEY"
    model_env: str = "DASHSCOPE_MODEL"
    default_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"


SIMULATION_CONFIG = SimulationConfig()
THRESHOLD_CONFIG = ThresholdConfig()
METRIC_CONFIG = MetricConfig()
ORDER_DISTRIBUTION_CONFIG = OrderDistributionConfig()
LLM_CONFIG = LLMConfig()
