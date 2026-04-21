import math


class IndividualCal:
    def __init__(self, Tw=10, Tin=10, Ts=10):
        self.stability = 0
        self.stability_list = []
        self.robustness = 0
        self.robustness_list = []
        self.inv = 0
        self.inv_list = []
        self.utility = 0
        self.utility_list = []
        self.cost_present_time = 0
        self.income_present_time = 0
        self.profit_present_time = 0
        self.cost_list_present_time = []
        self.income_list_present_time = []
        self.profit_list_present_time = []
        self.Ts = Ts if Ts is not None else 10
        self.Tw = Tw if Tw is not None else 10
        self.Tin = Tin if Tin is not None else 10
        self.w1 = 20
        self.w2 = 20
        self.w3 = 20
        self.w4 = 20

    def update_stability(self):
        self.stability = 0
        if len(self.profit_list_present_time) <= self.Ts:
            self.stability_list.append(self.stability)
            return
        sum_profit = 0
        list_len = len(self.profit_list_present_time)
        for i in range(list_len - self.Ts, list_len):
            sum_profit += self.profit_list_present_time[i]
        mu = sum_profit / self.Ts
        sigma = 0
        for i in range(list_len - self.Ts, list_len):
            sigma += (self.profit_list_present_time[i] - mu) ** 2
        sigma = math.sqrt(sigma / self.Ts)
        self.stability = 1 if sigma == 0 else 1.0 / sigma
        self.stability_list.append(self.stability)

    def update_robustness(self):
        self.robustness = 0
        if len(self.profit_list_present_time) <= self.Tw:
            self.robustness_list.append(self.robustness)
            return
        profits = self.profit_list_present_time[-self.Tw:]
        costs = self.cost_list_present_time[-self.Tw:]
        incomes = self.income_list_present_time[-self.Tw:]
        mu_profit = sum(profits) / self.Tw
        mu_cost = sum(costs) / self.Tw
        mu_income = sum(incomes) / self.Tw
        sigma_profit = math.sqrt(sum((p - mu_profit) ** 2 for p in profits) / self.Tw)
        sigma_cost = math.sqrt(sum((c - mu_cost) ** 2 for c in costs) / self.Tw)
        sigma_income = math.sqrt(sum((i - mu_income) ** 2 for i in incomes) / self.Tw)
        z_profit = (mu_profit - mu_profit) / sigma_profit if sigma_profit != 0 else 0
        z_cost = (mu_cost - mu_cost) / sigma_cost if sigma_cost != 0 else 0
        z_income = (mu_income - mu_income) / sigma_income if sigma_income != 0 else 0
        self.robustness = math.sqrt(z_profit ** 2 + z_cost ** 2 + z_income ** 2)
        self.robustness_list.append(self.robustness)

    def update_inv(self):
        self.inv = 0
        if len(self.cost_list_present_time) <= self.Tin or len(self.income_list_present_time) <= self.Tin:
            self.inv_list.append(self.inv)
            return
        deltaR = sum(self.income_list_present_time)
        deltaC = sum(self.cost_list_present_time)
        self.inv = 1 if deltaC == 0 else deltaR / deltaC
        self.inv_list.append(self.inv)

    def update_utility(self):
        self.utility = 0
        R_max = max(self.income_list_present_time) if self.income_list_present_time else 0
        C_min = min(self.cost_list_present_time) if self.cost_list_present_time else 1e9
        f1 = 0 if self.profit_present_time == 0 else self.profit_present_time / (R_max - C_min) if R_max != C_min else 0
        f2 = 1 if self.stability >= 1 else self.stability
        f3 = 1 / (1 + math.exp(-self.inv))
        self.utility = self.w1 * f1 + self.w2 * f2 + self.w3 * self.robustness + self.w4 * f3
        self.utility_list.append(self.utility)
