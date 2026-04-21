class Order:
    def __init__(self, order_id, pickup_location, delivery_location, pickup_time, delivery_time, money, create_step):
        self.id = order_id
        self.id_num = order_id
        self.pickup_location = pickup_location
        self.delivery_location = delivery_location
        self.pickup_time = pickup_time
        self.delivery_time = delivery_time
        self.status = 'unprocessed'
        self.money = money
        self.delete_time = create_step + 15
        self.finish_time = 0
        self.platform_money = money * 0.1
