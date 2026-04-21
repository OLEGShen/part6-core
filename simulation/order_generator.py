"""Order generation functions for the environment model.

Model element mapping (Table 7-4): Environment model.
`simulation.city.City` calls this module to generate temporally varying order
arrival intensity and spatially sampled orders.
"""

import random

import numpy as np

from config import ORDER_DISTRIBUTION_CONFIG


def all_orders_list(step_len, repeat_len, mer_list, rest_list, n, rand=True):
    """Generate all orders across the full simulation horizon."""

    if not rand:
        random.seed(0)
    all_list = [[]]
    for step in range(1, step_len):
        local_step = step % repeat_len
        order_num = max(0, int(n * gaussian_mixture_distribution(local_step)))
        orders = orders_list(order_num, mer_list, rest_list)
        for order in orders:
            order.append(local_step)
        all_list.append(orders)
    return all_list


def gaussian_mixture_distribution(x, amplitudes=None, centers=None, widths=None):
    r"""Compute the temporal order intensity from a Gaussian mixture.

    对应论文公式 (Order Distribution)
    LaTeX: f(x)=\sum_{i=1}^{5} a_i \exp\left(-\frac{(x-b_i)^2}{c_i^2}\right)
    """

    amplitudes = amplitudes or ORDER_DISTRIBUTION_CONFIG.amplitudes
    centers = centers or ORDER_DISTRIBUTION_CONFIG.centers
    widths = widths or ORDER_DISTRIBUTION_CONFIG.widths
    fitting_model = 0.0
    for amplitude, center, width in zip(amplitudes, centers, widths):
        fitting_model += amplitude * np.exp(-((x - center) ** 2) / (width ** 2))
    return float(fitting_model)


def fitting_dist(x):
    """Backward-compatible alias for the Gaussian mixture order generator."""

    return gaussian_mixture_distribution(x)


def get_order_type():
    weight = {"A": 0.2, "B": 0.4, "C": 0.4}
    return random.choices(list(weight.keys()), weights=list(weight.values()), k=1)[0]


def order_difficulty():
    return random.randint(1, 3)


def order_cooperation():
    return random.randint(0, 1)


def order_money(order_type):
    if order_type == 'A':
        return random.uniform(3, 4)
    elif order_type == 'B':
        return random.uniform(4, 8)
    elif order_type == 'C':
        return random.uniform(10, 12)


def merchant_position(merchant_list):
    return random.choice(merchant_list)


def rest_position(rest_list):
    return random.choice(rest_list)


def orders_list(order_num, mer_pos, rest_pos):
    daily_order = []
    for _ in range(order_num):
        order_type = get_order_type()
        daily_order.append([
            order_type,
            order_money(order_type),
            merchant_position(mer_pos),
            rest_position(rest_pos),
            order_difficulty()
        ])
    return daily_order
