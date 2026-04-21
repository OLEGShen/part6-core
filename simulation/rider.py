import json
import os
import random
from simulation.individual_cal import IndividualCal
from simulation.llm_agent import LLMDecisionError, RiderLLMAgent


class Rider:
    """Rider agent with heuristic or LLM-backed decisions.

    Model element mapping (Table 7-4): Agent model.
    The rider consumes platform order pools and emits time-varying behavior,
    thought logs, and individual metrics for the platform aggregator.
    """

    TYPE = 0

    def __init__(self, rider_id, location, max_orders=5, one_day=120,
                 go_work_time=8, get_off_work_time=18, step_move_distance=30,
                 rider_num=100, role_profile=None, output_dir=None,
                 decision_mode="auto", llm_model=None, llm_base_url=None,
                 Tw=None, Tin=None, Ts=None, eta=None, prompt_complexity="medium"):
        self.id = rider_id
        self.pt = location
        self.location = (location[0], location[1]) if hasattr(location, '__getitem__') else (location.x, location.y)
        self.max_orders = max_orders
        self.one_day = one_day
        self.go_work_time = go_work_time
        self.get_off_work_time = get_off_work_time
        self.default_go_work_time = go_work_time
        self.default_get_off_work_time = get_off_work_time

        self.route = []
        self.will_move_positions = []
        self.finish_orders_time = {}
        self.order_count = 0
        self.total_order = 0
        self.move_step = step_move_distance
        self.labor = 0
        self.dis = 0
        self.no_choose_order_step = 0
        self.money = 0

        self.target = IndividualCal(Tw=Tw, Tin=Tin, Ts=Ts, eta=eta)

        self.rank_day = {'order_rank': 0, 'dis_rank': 0, 'money_rank': 0}
        self.order_day = 0
        self.dis_day = 0
        self.money_day = 0

        self.rider_num = rider_num
        self.choose_order_step_interval = 15
        self.output_dir = output_dir
        self.role_profile = role_profile or {
            "role_description": "你是一名希望稳定挣钱的美团骑手。",
            "personality": "稳健"
        }
        self.decision_mode = decision_mode
        self.decision_backend = "heuristic"
        self.llm_agent = None
        self.prompt_complexity = prompt_complexity
        self._init_llm_agent(llm_model=llm_model, llm_base_url=llm_base_url)

    def step(self, meituan, runner_step):
        """Advance the rider by one simulation step."""

        self.decide_work_time(runner_step)
        self.orders_status_update(runner_step, meituan)
        self.route_to_walk(meituan, runner_step)
        self.walk_to_move()
        self.target_update()
        day_step = runner_step % self.one_day
        work_start = self.go_work_time * (self.one_day / 24)
        work_end = self.get_off_work_time * (self.one_day / 24)
        if day_step < work_start or day_step > work_end:
            return
        self.choose_order(meituan, runner_step)

    def target_update(self):
        """Refresh individual metrics after movement and order completion."""

        self.target.profit_present_time = max(self.target.income_present_time - self.target.cost_present_time, 0)
        self.target.cost_list_present_time.append(self.target.cost_present_time)
        self.target.income_list_present_time.append(self.target.income_present_time)
        self.target.profit_list_present_time.append(self.target.profit_present_time)
        self.target.update_stability()
        self.target.update_robustness()
        self.target.update_inv()
        self.target.update_utility()

    def _init_llm_agent(self, llm_model=None, llm_base_url=None):
        should_use_llm = self.decision_mode == "llm"
        if self.decision_mode == "auto":
            should_use_llm = bool(os.getenv("DASHSCOPE_API_KEY"))

        if not should_use_llm:
            return

        try:
            self.llm_agent = RiderLLMAgent(
                model=llm_model,
                base_url=llm_base_url,
                prompt_complexity=self.prompt_complexity,
            )
            self.decision_backend = "llm"
        except LLMDecisionError:
            if self.decision_mode == "llm":
                raise
            self.llm_agent = None
            self.decision_backend = "heuristic"

    def decide_work_time(self, runner_step):
        if runner_step % self.one_day == 0:
            info = {
                "before_go_work_time": self.go_work_time,
                "before_get_off_work_time": self.get_off_work_time,
                "rider_num": self.rider_num,
                "money_rank": self.rank_day["money_rank"],
                "dis_rank": self.rank_day["dis_rank"],
                "order_rank": self.rank_day["order_rank"],
                "money_day": self.money_day,
                "dis_day": self.dis_day,
                "order_day": self.order_day,
            }
            if self.llm_agent is not None:
                self._decide_work_time_by_llm(runner_step, info)
            else:
                self._decide_work_time_heuristically(runner_step, info)

    def _decide_work_time_heuristically(self, runner_step, info):
        go_work_time = info["before_go_work_time"]
        get_off_work_time = info["before_get_off_work_time"]
        if self.prompt_complexity == "low":
            go_work_time = min(10, max(6, go_work_time + random.choice([-1, 0, 1])))
            get_off_work_time = max(go_work_time + 6, min(23, get_off_work_time + random.choice([-1, 0, 1])))
        elif info["money_rank"] > max(1, self.rider_num // 2):
            go_work_time = max(6, go_work_time - 1)
            get_off_work_time = min(23, get_off_work_time + 1)
        elif 0 < info["money_rank"] <= max(1, self.rider_num // 5):
            get_off_work_time = max(go_work_time + 6, get_off_work_time - 1)
        elif self.prompt_complexity == "high":
            get_off_work_time = max(go_work_time + 8, min(23, get_off_work_time))

        self.go_work_time = go_work_time
        self.get_off_work_time = get_off_work_time
        self._log_thought(
            runner_step,
            event_type="work_time",
            param_dict=info,
            think="基于昨日收益排名进行启发式调整工作时长。",
            result={
                "go_work_time": self.go_work_time,
                "get_off_work_time": self.get_off_work_time,
                "decision_backend": self.decision_backend,
            },
        )

    def _decide_work_time_by_llm(self, runner_step, info):
        try:
            response = self.llm_agent.decide_work_time(self.role_profile, info)
            go_time = self._safe_hour(response.get("go_to_work_time"), self.default_go_work_time)
            off_time = self._safe_hour(response.get("get_off_work_time"), self.default_get_off_work_time)
            if off_time <= go_time:
                off_time = min(23, go_time + 8)
            self.go_work_time = go_time
            self.get_off_work_time = off_time
            self._log_thought(
                runner_step,
                event_type="work_time",
                param_dict=info,
                think=str(response.get("thought", "")),
                result={
                    "go_work_time": self.go_work_time,
                    "get_off_work_time": self.get_off_work_time,
                    "decision_backend": self.decision_backend,
                },
            )
        except Exception as exc:
            self._decide_work_time_heuristically(runner_step, info)
            self._log_thought(
                runner_step,
                event_type="work_time",
                param_dict=info,
                mixed_thought=f"LLM 决策失败，回退到启发式规则: {exc}",
                result={
                    "go_work_time": self.go_work_time,
                    "get_off_work_time": self.get_off_work_time,
                    "decision_backend": "heuristic_fallback",
                },
            )

    def choose_order(self, meituan, runner_step):
        chosen_order_list = []
        chosen_order_id_list = []
        order_list = meituan.now_orders_info
        if len(order_list) > 0 and self.max_orders - self.order_count > 0 and self.no_choose_order_step >= self.choose_order_step_interval:
            if len(order_list) > 10:
                short_order_list = list(order_list.items())[:10]
            else:
                short_order_list = list(order_list.items())
            info = {
                'order_list': short_order_list,
                'now_location': self.location,
                'accept_count': self.max_orders - self.order_count,
                'money_today': self.money_day,
                'orders_today': self.order_day,
                'work_window': [self.go_work_time, self.get_off_work_time],
            }
            chosen_order_id_list = self.take_order(runner_step, info)
            for uid in chosen_order_id_list:
                if type(uid) is int and uid in meituan.normal_orders_dict.keys():
                    chosen_order_list.append(meituan.normal_orders_dict[uid])
            self.update_route(chosen_order_list)
            self.order_count += len(chosen_order_list)
            self.total_order += len(chosen_order_list)
            self.order_day += len(chosen_order_list)
            meituan.choose_order_update(chosen_order_id_list)
            self.no_choose_order_step = 0
        else:
            self.no_choose_order_step += 1
        return chosen_order_list

    def take_order(self, runner_step, info):
        if self.llm_agent is not None:
            return self._take_order_by_llm(runner_step, info)
        return self._take_order_heuristically(runner_step, info)

    def _take_order_heuristically(self, runner_step, info):
        candidate_orders = [item[1] for item in info['order_list']]
        ranked = sorted(
            candidate_orders,
            key=lambda order: self._score_order(order, info["now_location"]),
            reverse=True,
        )
        if self.prompt_complexity == "low":
            random.shuffle(ranked)
        elif self.prompt_complexity == "medium" and len(ranked) > 1:
            ranked = ranked[: max(info["accept_count"] + 1, min(3, len(ranked)))] + ranked[max(info["accept_count"] + 1, min(3, len(ranked))):]
        selected = [order["order_id"] for order in ranked[:info["accept_count"]]]
        self._log_thought(
            runner_step,
            event_type="take_order",
            param_dict=info,
            think="基于收益与距离比的启发式策略选择订单。",
            result={
                "selected_order_ids": selected,
                "decision_backend": self.decision_backend,
            },
        )
        return selected

    def _take_order_by_llm(self, runner_step, info):
        candidate_orders = [item[1] for item in info["order_list"]]
        try:
            response = self.llm_agent.choose_orders(
                self.role_profile,
                {
                    "now_location": info["now_location"],
                    "accept_count": info["accept_count"],
                    "money_today": info["money_today"],
                    "orders_today": info["orders_today"],
                    "work_window": info["work_window"],
                },
                candidate_orders,
            )
            valid_ids = {order["order_id"] for order in candidate_orders}
            selected = []
            for order_id in response.get("selected_order_ids", []):
                if order_id in valid_ids and order_id not in selected:
                    selected.append(order_id)
                if len(selected) >= info["accept_count"]:
                    break
            self._log_thought(
                runner_step,
                event_type="take_order",
                param_dict=info,
                think=str(response.get("thought", "")),
                result={
                    "selected_order_ids": selected,
                    "decision_backend": self.decision_backend,
                },
            )
            return selected
        except Exception as exc:
            fallback = self._take_order_heuristically(runner_step, info)
            self._log_thought(
                runner_step,
                event_type="take_order",
                param_dict=info,
                mixed_thought=f"LLM 接单失败，回退到启发式规则: {exc}",
                result={
                    "selected_order_ids": fallback,
                    "decision_backend": "heuristic_fallback",
                },
            )
            return fallback

    def _score_order(self, order, now_location):
        pickup = order["pickup_location"]
        delivery = order["delivery_location"]
        pickup_dist = abs(pickup[0] - now_location[0]) + abs(pickup[1] - now_location[1])
        delivery_dist = abs(delivery[0] - pickup[0]) + abs(delivery[1] - pickup[1])
        total_dist = max(1, pickup_dist + delivery_dist)
        return float(order["money"]) / total_dist

    def update_route(self, chosen_order_list):
        for order in chosen_order_list:
            self.route.append(('pickup', order.id))
            self.route.append(('delivery', order.id))

    def route_to_walk(self, meituan, runner_step):
        move_step = 0
        for pos in self.route:
            order = meituan.get_order(pos[1])
            if order is None:
                continue
            now_loc = self.now_plan_pos()
            if pos[0] == 'pickup':
                path = self.plan_path(now_loc, order.pickup_location)
                move_step += len(path)
                self.will_move_positions += path
            else:
                path = self.plan_path(now_loc, order.delivery_location)
                move_step += len(path)
                self.will_move_positions += path
                order.finish_time = runner_step + int(move_step / self.move_step)
                if order.finish_time in self.finish_orders_time:
                    self.finish_orders_time[order.finish_time].append((pos[1], order))
                else:
                    self.finish_orders_time[order.finish_time] = [(pos[1], order)]
        self.route = []

    def plan_path(self, start, goal):
        path = []
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return [(start[0], start[1])]
        for i in range(steps):
            x = start[0] + int(dx * i / steps)
            y = start[1] + int(dy * i / steps)
            path.append((x, y))
        return path

    def now_plan_pos(self):
        if len(self.will_move_positions):
            return self.will_move_positions[-1]
        return self.location

    def walk_to_move(self):
        move_step = self.move_step
        i = 0
        self.target.cost_present_time = 0
        while i < move_step:
            if len(self.will_move_positions):
                pos = self.will_move_positions.pop(0)
                self.location = pos
                self.labor += 1
                self.dis_day += 1
                self.target.cost_present_time += 0.1
            else:
                break
            i += 1

    def orders_status_update(self, runner_step, meituan):
        if runner_step in self.finish_orders_time:
            self.order_count -= len(self.finish_orders_time[runner_step])
            now_money = sum(order[1].money for order in self.finish_orders_time[runner_step])
            self.money += now_money
            self.target.income_present_time = self.money
            self.money_day += now_money
            meituan.del_orders(self.finish_orders_time[runner_step], self.id, runner_step)
            del self.finish_orders_time[runner_step]

    def _safe_hour(self, value, default):
        try:
            hour = int(value)
        except (TypeError, ValueError):
            return default
        return min(23, max(0, hour))

    def _log_thought(self, runner_step, event_type, param_dict, think="", mixed_thought="", result=None):
        if not self.output_dir:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        file_path = os.path.join(self.output_dir, f"{self.id}_thought.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        data.append(
            {
                "runner_step": runner_step,
                "event_type": event_type,
                "param_dict": param_dict,
                "mixed_thought": mixed_thought,
                "result": result,
                "think": think,
                "decision_backend": self.decision_backend,
            }
        )

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
