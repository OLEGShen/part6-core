"""Platform-level metrics used by the simulation.

Model element mapping (Table 7-4): Rule model + environment model.
The class is updated by `simulation.platform.Platform` and consumes rider
metrics from `simulation.individual_cal.IndividualCal`.
"""

import math
from typing import Iterable, List

from config import METRIC_CONFIG


class SysCal:
    """Track platform welfare, diversity, entropy, and involution."""

    def __init__(self, agent_num=0, profit=0, Tf=None, Th=None, Rh=None, Tei=None, Rd=None):
        self.num_gov = agent_num
        self.profit = profit if profit is not None else 0.0
        self.profit_list: List[float] = []
        self.utility = 0.0
        self.utility_list: List[float] = []
        self.fairness = 0.0
        self.fairness_list: List[float] = []
        self.swf = 0.0
        self.swf_list: List[float] = []
        self.involution = 0.0
        self.involution_list: List[float] = []
        self.Tf = Tf if Tf is not None else METRIC_CONFIG.platform_Tf
        self.variety = 0.0
        self.variety_list: List[float] = []
        self.Th = Th if Th is not None else METRIC_CONFIG.platform_Th
        self.Rh = Rh if Rh is not None else METRIC_CONFIG.platform_Rh
        self.entropy_increase = 0.0
        self.entropy_increase_list: List[float] = []
        self.Tei = Tei if Tei is not None else METRIC_CONFIG.platform_Tei
        self.Rd = Rd if Rd is not None else METRIC_CONFIG.platform_Rd

    def update_profit(self, money):
        self.profit += money
        self.profit_list.append(self.profit)

    def update_fairness(self, agents):
        """Update platform fairness using a bounded Gini-based equality term."""

        self.fairness = 0
        x_gs = []
        for gov in agents:
            if len(gov.target.profit_list_present_time) <= self.Tf:
                self.fairness_list.append(self.fairness)
                return
            len_present_time = len(gov.target.profit_list_present_time)
            x_gk = sum(gov.target.profit_list_present_time[len_present_time - self.Tf:len_present_time]) / self.Tf
            x_gs.append(x_gk)
        self.fairness = self.compute_eq(x_gs)
        self.fairness_list.append(self.fairness)

    def update_variety(self, agents):
        """Update strategy diversity with a normalized entropy score."""

        self.variety = 0
        avg_profit_list = []
        for gov in agents:
            if len(gov.target.profit_list_present_time) <= self.Th:
                self.variety_list.append(self.variety)
                return
            len_present_time = len(gov.target.profit_list_present_time)
            avg_profit = sum(gov.target.profit_list_present_time[len_present_time - self.Th:]) / self.Th
            avg_profit_list.append(avg_profit)
        rh_dict = {}
        for avg_profit in avg_profit_list:
            key = str(int(avg_profit / self.Rh))
            rh_dict[key] = rh_dict.get(key, 0) + 1
        n = len(rh_dict)
        fz = sum((v / self.num_gov) * math.log(v / self.num_gov) for v in rh_dict.values() if v > 0)
        self.variety = 1 if n == 1 else -fz / math.log(n)
        self.variety_list.append(self.variety)

    def update_entropy_increase(self, agents):
        """Update entropy increase from the change in profit diversity."""

        self.entropy_increase = 0
        profit_list1 = []
        profit_list2 = []
        for gov in agents:
            if len(gov.target.profit_list_present_time) <= self.Tei:
                self.entropy_increase_list.append(self.entropy_increase)
                return
            len_present_time = len(gov.target.profit_list_present_time)
            profit_list1.append(gov.target.profit_list_present_time[len_present_time - 1])
            profit_list2.append(gov.target.profit_list_present_time[len_present_time - self.Tei])
        rd_dict = {}
        for profit in profit_list1:
            key = str(int(profit / self.Rd))
            rd_dict[key] = rd_dict.get(key, 0) + 1
        d1 = 1 - sum((v / self.num_gov) ** 2 for v in rd_dict.values())
        rd_dict = {}
        for profit in profit_list2:
            key = str(int(profit / self.Rd))
            rd_dict[key] = rd_dict.get(key, 0) + 1
        d2 = 1 - sum((v / self.num_gov) ** 2 for v in rd_dict.values())
        self.entropy_increase = 1 - math.fabs((d1 - d2) / self.Tei)
        self.entropy_increase_list.append(self.entropy_increase)

    @staticmethod
    def _gini(values: Iterable[float]) -> float:
        values = sorted(float(value) for value in values if value is not None)
        n = len(values)
        if n == 0:
            return 0.0
        total = sum(values)
        if total == 0:
            return 0.0
        diff_sum = sum(abs(a - b) for a in values for b in values)
        return diff_sum / (2 * n * total)

    def compute_eq(self, values: Iterable[float]) -> float:
        r"""Compute the equality term in the social welfare function.

        对应论文公式 (Swf-eq)
        LaTeX: eq(t)=1-\mathrm{gini}(a)\cdot\frac{N}{N-1}
        """

        values = list(values)
        n = len(values)
        if n <= 1:
            return 1.0
        equality = 1.0 - self._gini(values) * n / (n - 1)
        return max(0.0, min(1.0, equality))

    def compute_productivity(self, agents) -> float:
        r"""Compute productivity as the total rider income.

        对应论文公式 (Swf-prod)
        LaTeX: prod(t)=\sum_i a_i
        """

        total_income = 0.0
        for agent in agents:
            income_window = [max(value, 0.0) for value in agent.target.income_list_present_time[-self.Tf:]]
            if income_window:
                total_income += sum(income_window) / len(income_window)
        return total_income

    def compute_average_utility(self, agents) -> float:
        """Compute the mean rider utility over the same time window `T`."""

        utilities = []
        for agent in agents:
            incomes = agent.target.income_list_present_time[-self.Tf:]
            costs = agent.target.cost_list_present_time[-self.Tf:]
            if not incomes and not costs:
                continue
            net_reward = max(sum(incomes) - sum(costs), 1e-9)
            utilities.append(agent.target.crra(net_reward))
        return sum(utilities) / len(utilities) if utilities else 0.0

    def compute_swf(self, agents) -> float:
        r"""Compute platform social welfare.

        对应论文公式 (Swf)
        LaTeX: \mathrm{Swf}(T)=eq(t)\cdot prod(t)
        """

        avg_profit_window = []
        for agent in agents:
            profits = agent.target.profit_list_present_time[-self.Tf:]
            if profits:
                avg_profit_window.append(sum(profits) / len(profits))
        eq_t = self.compute_eq(avg_profit_window)
        prod_t = self.compute_productivity(agents)
        return eq_t * prod_t

    def compute_involution(self, agents) -> float:
        r"""Compute the involution index.

        对应论文公式 (Involution)
        LaTeX: \mathrm{Involution}(t)=\frac{\mathrm{Swf}(T)}{\mathrm{avg}(\mathrm{Utility}(T,i))}
        """

        avg_utility = self.compute_average_utility(agents)
        if avg_utility <= 0:
            return 0.0
        return self.swf / avg_utility

    def update_utility(self, agents=None):
        """Update platform welfare metrics and expose social welfare as utility."""

        agents = agents or []
        self.swf = self.compute_swf(agents)
        self.swf_list.append(self.swf)
        self.involution = self.compute_involution(agents)
        self.involution_list.append(self.involution)
        self.utility = self.swf
        self.utility_list.append(self.utility)
