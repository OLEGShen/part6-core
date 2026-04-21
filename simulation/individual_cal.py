"""Individual rider metrics used by the simulation.

Model element mapping (Table 7-4): Agent model.
The class is called by `simulation.rider.Rider` after each step and returns
time-varying individual metrics consumed by `simulation.sys_cal.SysCal`.
"""

import math
from typing import Iterable, List

from config import METRIC_CONFIG


class IndividualCal:
    """Track individual performance and risk-sensitive utility.

    Inputs:
    - income/cost/profit time series updated by `Rider.target_update()`.
    Outputs:
    - `stability`, `robustness`, `inv`, and `utility` together with histories.
    """

    def __init__(self, Tw=None, Tin=None, Ts=None, eta=None):
        self.stability = 0.0
        self.stability_list: List[float] = []
        self.robustness = 0.0
        self.robustness_list: List[float] = []
        self.inv = 0.0
        self.inv_list: List[float] = []
        self.utility = 0.0
        self.utility_list: List[float] = []
        self.cost_present_time = 0.0
        self.income_present_time = 0.0
        self.profit_present_time = 0.0
        self.cost_list_present_time: List[float] = []
        self.income_list_present_time: List[float] = []
        self.profit_list_present_time: List[float] = []
        self.Ts = Ts if Ts is not None else METRIC_CONFIG.individual_Ts
        self.Tw = Tw if Tw is not None else METRIC_CONFIG.individual_Tw
        self.Tin = Tin if Tin is not None else METRIC_CONFIG.individual_Tin
        self.eta = eta if eta is not None else METRIC_CONFIG.crra_eta
        if self.eta <= 0:
            raise ValueError("CRRA 参数 eta 必须满足 eta > 0。")

    @staticmethod
    def _window(values: Iterable[float], window_size: int) -> List[float]:
        values = list(values)
        if len(values) < window_size:
            return []
        return values[-window_size:]

    @staticmethod
    def _mean(values: Iterable[float]) -> float:
        values = list(values)
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _std(values: Iterable[float]) -> float:
        values = list(values)
        if not values:
            return 0.0
        mean_value = IndividualCal._mean(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        return math.sqrt(variance)

    def update_stability(self):
        """Update rider income stability over the last `Ts` steps."""

        profits = self._window(self.profit_list_present_time, self.Ts)
        if not profits:
            self.stability = 0.0
            self.stability_list.append(self.stability)
            return

        sigma = self._std(profits)
        self.stability = 1.0 if sigma == 0 else 1.0 / sigma
        self.stability_list.append(self.stability)

    def update_robustness(self):
        """Update robustness using the inverse dispersion of profit, cost, and income."""

        profits = self._window(self.profit_list_present_time, self.Tw)
        costs = self._window(self.cost_list_present_time, self.Tw)
        incomes = self._window(self.income_list_present_time, self.Tw)
        if not profits or not costs or not incomes:
            self.robustness = 0.0
            self.robustness_list.append(self.robustness)
            return

        dispersion = self._std(profits) + self._std(costs) + self._std(incomes)
        self.robustness = 1.0 / (1.0 + dispersion)
        self.robustness_list.append(self.robustness)

    def update_inv(self):
        """Update rider-level revenue-cost ratio over the last `Tin` steps."""

        deltaR = sum(self._window(self.income_list_present_time, self.Tin))
        deltaC = sum(self._window(self.cost_list_present_time, self.Tin))
        if deltaR == 0 and deltaC == 0:
            self.inv = 0.0
        else:
            self.inv = deltaR / max(deltaC, 1e-9)
        self.inv_list.append(self.inv)

    def crra(self, z: float) -> float:
        r"""Compute the CRRA utility transform.

        对应论文公式 (CRRA)
        LaTeX: \mathrm{crra}(z)=\frac{z^{1-\eta}-1}{1-\eta},\ \eta>0
        """

        z = max(z, 1e-9)
        if math.isclose(self.eta, 1.0):
            return math.log(z)
        return (z ** (1.0 - self.eta) - 1.0) / (1.0 - self.eta)

    def compute_utility(self) -> float:
        r"""Compute rider utility from cumulative reward and cost.

        对应论文公式 (Utility)
        LaTeX: \mathrm{Utility}(T,i)=\mathrm{crra}\left(\sum \mathrm{Reward}-\sum \mathrm{Cost}\right)
        """

        total_reward = sum(self.income_list_present_time)
        total_cost = sum(self.cost_list_present_time)
        net_reward = max(total_reward - total_cost, 1e-9)
        return self.crra(net_reward)

    def update_utility(self):
        """Update and store the rider utility time series."""

        self.utility = self.compute_utility()
        self.utility_list.append(self.utility)
