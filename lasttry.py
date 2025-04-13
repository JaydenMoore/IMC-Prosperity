from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import numpy as np

class Trader:
    def __init__(self):
        self.position_limits = 20
        self.max_volume = 3
        self.price_history = {}

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
                self.price_history[product] = self.price_history[product][-20:]

                # Compute simple moving average to identify short-term momentum
                if len(self.price_history[product]) >= 5:
                    sma_short = np.mean(self.price_history[product][-3:])
                    sma_long = np.mean(self.price_history[product])
                    trend = sma_short - sma_long
                else:
                    trend = 0

                # Base prices with safe profit spread
                buy_price = int(mid_price - 2)
                sell_price = int(mid_price + 2)

                # Adjust prices slightly in trend direction (safe momentum)
                if trend > 0:
                    sell_price += 1  # Uptrend, more aggressive selling
                elif trend < 0:
                    buy_price -= 1  # Downtrend, more aggressive buying

                # Conservative trade size
                volume = self.max_volume

                if position < self.position_limits:
                    buy_volume = min(self.position_limits - position, volume)
                    if buy_price < best_ask:
                        orders.append(Order(product, buy_price, buy_volume))

                if position > -self.position_limits:
                    sell_volume = min(self.position_limits + position, volume)
                    if sell_price > best_bid:
                        orders.append(Order(product, sell_price, -sell_volume))

            result[product] = orders

        return result, 0, ""
