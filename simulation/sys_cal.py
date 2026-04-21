import math


class SysCal:
    def __init__(self, agent_num=0, profit=0, Tf=10, Th=10, Rh=20, Tei=10, Rd=20):
        self.num_gov = agent_num
        self.profit = profit if profit is not None else 0
        self.profit_list = []
        self.utility = 0
        self.utility_list = []
        self.fairness = 0
        self.fairness_list = []
        self.Tf = Tf if Tf is not None else 10
        self.variety = 0
        self.variety_list = []
        self.Th = Th if Th is not None else 10
        self.Rh = Rh if Rh is not None else 20
        self.entropy_increase = 0
        self.entropy_increase_list = []
        self.Tei = Tei if Tei is not None else 10
        self.Rd = Rd if Rd is not None else 20
        self.w1 = 1
        self.w2 = 1
        self.w3 = 1
        self.w4 = 1

    def update_profit(self, money):
        self.profit += money

    def update_fairness(self, agents):
        self.fairness = 0
        x_gs = []
        for gov in agents:
            if len(gov.target.profit_list_present_time) <= self.Tf:
                self.fairness_list.append(self.fairness)
                return
            len_present_time = len(gov.target.profit_list_present_time)
            x_gk = sum(gov.target.profit_list_present_time[len_present_time - self.Tf:len_present_time]) / self.Tf
            x_gs.append(x_gk)
        fz = sum(math.fabs(xg1 - xg2) for xg1 in x_gs for xg2 in x_gs)
        sum_xg = sum(x_gs)
        mu = 2 * self.num_gov * self.num_gov * sum_xg / len(x_gs) if len(x_gs) > 0 else 0
        self.fairness = 0 if mu == 0 else 1 - fz / mu
        self.fairness_list.append(self.fairness)

    def update_variety(self, agents):
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

    def update_utility(self):
        self.utility = self.w1 * self.profit + self.w2 * self.fairness + self.w3 * self.variety + self.w4 * self.entropy_increase
        self.utility_list.append(self.utility)
