"""Platform state manager for the simulation.

Model element mapping (Table 7-4): Rule model.
The platform manages order pools, updates rider rankings, and aggregates
system-level metrics through `simulation.sys_cal.SysCal`.
"""

from simulation.sys_cal import SysCal


class Platform:
    """Maintain platform state, orders, and system metrics."""

    def __init__(self, order_bf, riders, start_time):
        self.riders = riders
        self.order_BF = order_bf
        self.order_id = [0]
        self.all_normal_orders_info = []
        self.normal_orders_dict = {}
        self.now_orders_info = {}

        self.dis_day_rank = []
        self.money_day_rank = []
        self.order_day_rank = []

        self.target = SysCal(len(self.riders))
        self.start_time = start_time

    def update_target(self):
        """Update fairness, diversity, entropy, welfare, and involution."""

        self.target.update_fairness(list(self.riders.values()))
        self.target.update_variety(list(self.riders.values()))
        self.target.update_entropy_increase(list(self.riders.values()))
        self.target.update_utility(list(self.riders.values()))

    def generate_order_now_step(self, runner_step):
        if runner_step >= len(self.all_normal_orders_info):
            return
        for order_info in self.all_normal_orders_info[runner_step]:
            from simulation.order import Order
            new_order = Order(
                self.order_id[0],
                order_info[2],
                order_info[3],
                (runner_step, runner_step + 15),
                (runner_step, runner_step + 15),
                order_info[1],
                runner_step
            )
            self.order_id[0] += 1
            self.normal_orders_dict[new_order.id_num] = new_order
            self.now_orders_info[new_order.id_num] = {
                "order_id": new_order.id_num,
                "pickup_location": new_order.pickup_location,
                "delivery_location": new_order.delivery_location,
                "money": round(new_order.money, 2)
            }

    def choose_order_update(self, choose_list):
        for i in choose_list:
            try:
                self.target.update_profit(self.normal_orders_dict[i].platform_money)
                if i in self.now_orders_info:
                    del self.now_orders_info[i]
            except Exception:
                pass

    def check_orders_rider(self):
        return self.now_orders_info

    def return_rank(self):
        return self.money_day_rank, self.dis_day_rank, self.order_day_rank

    def update_rank(self):
        self.dis_day_rank.clear()
        self.order_day_rank.clear()
        self.money_day_rank.clear()
        for rider in self.riders.values():
            self.dis_day_rank.append((rider.dis_day, rider))
            self.money_day_rank.append((rider.money_day, rider))
            self.order_day_rank.append((rider.order_day, rider))
        self.dis_day_rank.sort(key=lambda i: i[0], reverse=True)
        self.money_day_rank.sort(key=lambda i: i[0], reverse=True)
        self.order_day_rank.sort(key=lambda i: i[0], reverse=True)
        for i in range(len(self.dis_day_rank)):
            self.dis_day_rank[i][1].rank_day['dis_rank'] = i + 1
            self.money_day_rank[i][1].rank_day['money_rank'] = i + 1
            self.order_day_rank[i][1].rank_day['order_rank'] = i + 1
        for rider in self.riders.values():
            rider.dis_day = 0
            rider.money_day = 0
            rider.order_day = 0

    def get_order(self, order_id):
        return self.normal_orders_dict.get(order_id)

    def del_orders(self, orders, rider_id, runner_step):
        for order_id, order in orders:
            if order_id in self.normal_orders_dict:
                del self.normal_orders_dict[order_id]
