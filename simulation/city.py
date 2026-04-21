import random
import datetime
from simulation.rider import Rider
from simulation.platform import Platform
from simulation.order_generator import all_orders_list


class City:
    def __init__(self, rider_num, run_len, one_day=120, one_hour=5,
                 order_bf=3, order_weight=0.3, seed=None, output_dir=None,
                 decision_mode="auto", llm_model=None, llm_base_url=None):
        if seed is not None:
            random.seed(seed)

        self.run_len = run_len
        self.one_day = one_day
        self.one_hour = one_hour
        self.runner_step = 0
        self.rider_num = rider_num
        self.output_dir = output_dir
        self.decision_mode = decision_mode
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url

        self.riders = {}
        self._init_riders()

        self.meituan = Platform(order_bf, self.riders, datetime.datetime.now())
        self._init_orders(order_bf, order_weight)

    def _init_riders(self):
        home_positions = self._get_home_positions()
        role_profiles = self._get_role_profiles()
        for i in range(self.rider_num):
            init_pos = random.choice(home_positions)
            rider = Rider(
                i,
                init_pos,
                max_orders=5,
                one_day=self.one_day,
                rider_num=self.rider_num,
                role_profile=role_profiles[i % len(role_profiles)],
                output_dir=self.output_dir,
                decision_mode=self.decision_mode,
                llm_model=self.llm_model,
                llm_base_url=self.llm_base_url,
            )
            self.riders[rider.id] = rider

    def _get_home_positions(self):
        return [(x, y) for x in range(10, 100, 10) for y in range(10, 100, 10)]

    def _get_role_profiles(self):
        return [
            {
                "role_description": "你追求更高收入，愿意延长工作时长并积极抢单。",
                "personality": "激进竞争型",
            },
            {
                "role_description": "你更注重稳定收益，希望保持可持续的工作节奏。",
                "personality": "稳健平衡型",
            },
            {
                "role_description": "你更在意体力和风险，接单时会更谨慎。",
                "personality": "保守谨慎型",
            },
        ]

    def _init_orders(self, order_bf, order_weight):
        mer_list = [(x, y) for x in range(20, 80, 10) for y in range(20, 80, 10)]
        user_list = [(x, y) for x in range(10, 90, 10) for y in range(10, 90, 10)]
        self.meituan.all_normal_orders_info = all_orders_list(
            self.run_len, int(self.one_day / 2), mer_list, user_list, order_weight
        )

    def step(self):
        tmp_order_num = len(self.meituan.now_orders_info)
        if self.runner_step % self.one_day == 0:
            self.meituan.update_rank()

        self.meituan.generate_order_now_step(self.runner_step)

        agent_list = list(self.riders.values())
        random.shuffle(agent_list)

        for agent in agent_list:
            agent.step(self.meituan, self.runner_step)

        self.meituan.update_target()
        self.runner_step += 1

    def run(self):
        for i in range(self.run_len):
            self.step()
        return self.collect_results()

    def collect_results(self):
        results = {
            'rider_records': {},
            'platform_record': {}
        }

        for rider_id, rider in self.riders.items():
            results['rider_records'][rider_id] = {
                'money': rider.money,
                'labor': rider.labor,
                'total_order': rider.total_order,
                'go_work_time': rider.go_work_time,
                'get_off_work_time': rider.get_off_work_time,
                'stability': rider.target.stability,
                'robustness': rider.target.robustness,
                'inv': rider.target.inv,
                'utility': rider.target.utility
            }

        results['platform_record'] = {
            'profit': self.meituan.target.profit,
            'fairness': self.meituan.target.fairness,
            'variety': self.meituan.target.variety,
            'entropy_increase': self.meituan.target.entropy_increase,
            'utility': self.meituan.target.utility
        }

        return results
