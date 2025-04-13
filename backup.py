from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import numpy as np

class Trader:
    def __init__(self):
        # Define position limits for each product
        self.position_limits = {
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBE": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            # Add default limit for other products
            "DEFAULT": 20
        }
        
        # Define basket compositions
        self.basket_compositions = {
            "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBE": 1},
            "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2}
        }
        
        self.max_volume = 3
        self.price_history = {}
        self.mean_reversion_window = 10  # Window size for mean reversion calculation
        self.deviation_threshold = 2  # Threshold for deviation from mean

    def run(self, state: TradingState):
        result = {}

        # First, apply mean reversion strategy on all available products
        for product, order_depth in state.order_depths.items():
            orders = []
            position = state.position.get(product, 0)
            position_limit = self.position_limits.get(product, self.position_limits["DEFAULT"])

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
                    sell_volume = min(position_limit + position, self.max_volume)
                    if sell_price > best_bid and sell_volume > 0:
                        orders.append(Order(product, sell_price, -sell_volume))
                elif deviation < -self.deviation_threshold:
                    # Buy when price is below mean
                    buy_price = int(mid_price)
                    buy_volume = min(position_limit - position, self.max_volume)
                    if buy_price < best_ask and buy_volume > 0:
                        orders.append(Order(product, buy_price, buy_volume))

            result[product] = orders

        # Check for arbitrage opportunities between baskets and individual items
        self.add_basket_arbitrage_orders(state, result)
        
        return result, 0, ""
    
    def add_basket_arbitrage_orders(self, state: TradingState, result: Dict):
        """Add orders for basket arbitrage opportunities"""
        for basket, composition in self.basket_compositions.items():
            # Skip if basket or any component isn't in the order depths
            if basket not in state.order_depths:
                continue
            
            if not all(component in state.order_depths for component in composition):
                continue
                
            # Calculate the value of the basket from its components
            basket_component_buy_value = 0
            basket_component_sell_value = 0
            can_arbitrage = True
            
            for component, quantity in composition.items():
                if not (state.order_depths[component].buy_orders and state.order_depths[component].sell_orders):
                    can_arbitrage = False
                    break
                
                # For buying components and selling basket
                basket_component_buy_value += min(state.order_depths[component].sell_orders.keys()) * quantity
                
                # For selling components and buying basket
                basket_component_sell_value += max(state.order_depths[component].buy_orders.keys()) * quantity
            
            if not can_arbitrage:
                continue
                
            # Check if arbitrage is possible
            basket_orders = state.order_depths[basket]
            if basket_orders.buy_orders and basket_orders.sell_orders:
                basket_buy_price = max(basket_orders.buy_orders.keys())
                basket_sell_price = min(basket_orders.sell_orders.keys())
                
                # Arbitrage 1: Buy basket, sell components
                if basket_buy_price > basket_component_sell_value:
                    # Profitable to buy basket and sell components
                    basket_position = state.position.get(basket, 0)
                    volume = min(1, self.position_limits[basket] - basket_position)
                    
                    if volume > 0:
                        # Add buy order for basket
                        basket_orders = result.get(basket, [])
                        basket_orders.append(Order(basket, basket_buy_price, volume))
                        result[basket] = basket_orders
                        
                        # Add sell orders for components
                        for component, quantity in composition.items():
                            component_orders = result.get(component, [])
                            component_sell_price = max(state.order_depths[component].buy_orders.keys())
                            component_position = state.position.get(component, 0)
                            
                            if component_position - (quantity * volume) >= -self.position_limits[component]:
                                component_orders.append(Order(component, component_sell_price, -quantity * volume))
                                result[component] = component_orders
                
                # Arbitrage 2: Buy components, sell basket
                if basket_sell_price < basket_component_buy_value:
                    # Profitable to buy components and sell basket
                    basket_position = state.position.get(basket, 0)
                    volume = min(1, basket_position + self.position_limits[basket])
                    
                    if volume > 0:
                        # Add sell order for basket
                        basket_orders = result.get(basket, [])
                        # Penny the market by selling at 1 tick lower to gain execution priority
                        penny_sell_price = basket_sell_price - 1
                        basket_orders.append(Order(basket, penny_sell_price, -volume))
                        result[basket] = basket_orders
                        
                        # Add buy orders for components with fair value estimation
                        remaining_volume = volume
                        for component, quantity in composition.items():
                            component_orders = result.get(component, [])
                            # Calculate VWAP for better price estimation
                            sell_prices = list(state.order_depths[component].sell_orders.keys())
                            sell_volumes = [abs(state.order_depths[component].sell_orders[price]) for price in sell_prices]
                            
                            # Get the minimum asking price
                            component_buy_price = min(sell_prices)
                            component_position = state.position.get(component, 0)
                            
                            # Apply soft position limits for better risk management
                            soft_limit = self.position_limits[component] * 0.9
                            allowed_volume = min(
                                quantity * remaining_volume,
                                max(0, soft_limit - component_position)
                            )
                            
                            if allowed_volume > 0:
                                # Penny up by 1 to gain execution priority
                                component_orders.append(Order(component, component_buy_price + 1, allowed_volume))
                                result[component] = component_orders
                                remaining_volume = remaining_volume - (allowed_volume / quantity)
                                if remaining_volume <= 0:
                                    break