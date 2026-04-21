import numpy as np
import random


def all_orders_list(step_len, repeat_len, mer_list, rest_list, n, rand=True):
    if not rand:
        random.seed(0)
    all_list = []
    for i in range(1, step_len):
        i = i % repeat_len
        order_num = int(n * int(fitting_dist(i)))
        orders = orders_list(order_num, mer_list, rest_list)
        for order in orders:
            order.append(i)
        all_list.append(orders)
    return all_list


def fitting_dist(x):
    a = [314.2, 188.3, 95.56, 22.9, 48.67]
    b = [172.5, 281.5, 315.5, 228.9, 267.1]
    c = [4.645, 1.559, 10.69, 167.7, 13.1]
    fitting_model = 0
    for i in range(5):
        fitting_model += a[i] * np.exp(-((x - b[i]) / c[i]) ** 2)
    return fitting_model


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
