import math


def distance(location, other):
    x1, y1 = location
    x2, y2 = other
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def objective(route, orders):
    total_delay = 0
    total_distance = 0
    current_time = 0
    current_location = (0, 0)
    order_dict = {order.id: order for order in orders}
    for event_type, order_id in route:
        order = order_dict[order_id]
        travel_time = distance(current_location, order.pickup_location if event_type == 'pickup' else order.delivery_location)
        current_time += travel_time
        total_distance += travel_time
        if event_type == 'pickup':
            current_location = order.pickup_location
            if current_time < order.pickup_time[0]:
                current_time = order.pickup_time[0]
        else:
            current_location = order.delivery_location
            if current_time > order.delivery_time[1]:
                total_delay += current_time - order.delivery_time[1]
    return total_delay + total_distance


def is_valid_route(route, orders):
    picked_up_orders = set()
    for event_type, order_id in route:
        if event_type == 'pickup':
            picked_up_orders.add(order_id)
        else:
            if order_id not in picked_up_orders:
                return False
            picked_up_orders.remove(order_id)
    return True


def local_search(route, orders):
    best_route = route
    best_value = objective(route, orders)
    while True:
        improved = False
        for i in range(len(route)):
            for j in range(i + 1, len(route)):
                new_route = best_route[:i] + best_route[j:] + best_route[i:j]
                if is_valid_route(new_route, orders):
                    value = objective(new_route, orders)
                    if value < best_value:
                        best_route = new_route
                        best_value = value
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    return best_route


def greedy_insertion_with_clustering(orders, rider, D=50):
    events = []
    for order in orders:
        events.append(('pickup', order.id))
        events.append(('delivery', order.id))
    locations = [order.pickup_location for order in orders] + [order.delivery_location for order in orders]
    clusters = hierarchical_clustering(locations, D)
    sorted_events = sorted(events, key=lambda x: orders[x[1]].pickup_time[0] if x[0] == 'pickup' else float('inf'))
    order_status = {order.id: {'pickup': False, 'delivery': False} for order in orders}

    for event_type, order_id in sorted_events:
        order = next(o for o in orders if o.id == order_id)
        if order.status == 'processed' or rider.order_count >= rider.max_orders:
            continue
        valid_positions = []
        for i in range(len(rider.route) + 1):
            new_route = rider.route[:i] + [(event_type, order_id)] + rider.route[i:]
            if is_valid_route(new_route, orders):
                if event_type == 'delivery' and not any(et == 'pickup' and oid == order_id for et, oid in new_route):
                    continue
                valid_positions.append((i, objective(new_route, orders)))
        if not valid_positions:
            continue
        best_position, best_value = min(valid_positions, key=lambda x: x[1])
        rider.route = rider.route[:best_position] + [(event_type, order_id)] + rider.route[best_position:]
        order_status[order_id][event_type] = True
        if all(order_status[order_id].values()):
            order.status = 'processed'
            rider.order_count += 1


def hierarchical_clustering(locations, D):
    clusters = []
    for location in locations:
        assigned = False
        for cluster in clusters:
            if min(distance(location, other) for other in cluster) < D:
                cluster.append(location)
                assigned = True
                break
        if not assigned:
            clusters.append([location])
    return clusters


def two_stage_fast_heuristic(orders, riders, D=50):
    for rider in riders:
        greedy_insertion_with_clustering(orders, rider, D)
        rider.route = local_search(rider.route, orders)
    routes = {}
    for rider in riders:
        path = [rider.location]
        for action, order_id in rider.route:
            order = next((o for o in orders if o.id == order_id), None)
            if order:
                if action == 'pickup':
                    path.append(order.pickup_location)
                elif action == 'delivery':
                    path.append(order.delivery_location)
        routes[rider.id] = path
    return routes
