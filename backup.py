from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import numpy as np

class Trader:
    def __init__(self):
        self.position_limits = 20
        self.max_volume = 3
        self.price_history = {}
        self.mean_reversion_window = 10  # Window size for mean reversion calculation
        self.deviation_threshold = 2  # Threshold for deviation from mean

    def run(self, state: TradingState):
        result = {}

        for product, order_depth in state.order_depths.items():
            orders = []
            position = state.position.get(product, 0)

            if product not in self.price_history:
                self.price_history[product] = []

            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())

                if best_ask <= best_bid:
                    continue

                mid_price = (best_bid + best_ask) / 2
                self.price_history[product].append(mid_price)
                self.price_history[product] = self.price_history[product][-self.mean_reversion_window:]

                mean_price = np.mean(self.price_history[product])
                deviation = mid_price - mean_price

                if deviation > self.deviation_threshold:
                    # Sell when price is above mean
                    sell_price = int(mid_price)
                    sell_volume = min(self.position_limits + position, self.max_volume)
                    if sell_price > best_bid and sell_volume > 0:
                        orders.append(Order(product, sell_price, -sell_volume))
                elif deviation < -self.deviation_threshold:
                    # Buy when price is below mean
                    buy_price = int(mid_price)
                    buy_volume = min(self.position_limits - position, self.max_volume)
                    if buy_price < best_ask and buy_volume > 0:
                        orders.append(Order(product, buy_price, buy_volume))

            result[product] = orders

        return result, 0, ""