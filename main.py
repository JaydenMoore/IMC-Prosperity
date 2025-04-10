from datamodel import Order, TradingState, OrderDepth
import json
import statistics
import numpy as np
from typing import Dict, List, Tuple
import traceback

class Trader:
    def __init__(self):
        self.position = {'SQUID_INK': 0, 'KELP': 0, 'RAINFOREST_RESIN': 0}
        self.price_history = {product: [] for product in self.position.keys()}
        self.max_drawdown = 0.10
        self.max_position_duration = 100
        self.position_entries = {product: [] for product in self.position.keys()}
        self.position_limit = 50
        self.volatility_lookback = 20
        self.max_daily_loss = 5000  # More reasonable limit
        self.daily_pnl = 0
        self.today_pnl = 0  # Track today's PnL separately
        
        # Regime thresholds
        self.regimes = {
            'bull': {'volatility': 0.08, 'trend': 0.01},  # More sensitive
            'bear': {'volatility': 0.10, 'trend': -0.02},
            'volatile': {'volatility': 0.15}
        }
        
        # Strategy configurations
        self.strategies = {
            'volatile': {
                'window': 30,
                'bull_threshold': 0.03,
                'bear_threshold': 0.05,
                'bull_multiplier': 1.2,
                'bear_multiplier': 0.6,
                'max_risk': 0.10
            },
            'stable': {
                'window': 5,
                'bull_threshold': 0.015,
                'bear_threshold': 0.025,
                'base_size': 4,
                'volatility_scaling': True
            }
        }
        
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        orders = {}
        conversions = 0
        trader_data = ""
        
        try:
            # Initialize orders for all products
            products = ['SQUID_INK', 'KELP', 'RAINFOREST_RESIN']
            for product in products:
                orders[product] = []
                
                # Product-specific parameters
                config = {
                    'SQUID_INK': {'position_limit': 40, 'spread_pct': 0.005},
                    'KELP': {'position_limit': 30, 'spread_pct': 0.01},
                    'RAINFOREST_RESIN': {'position_limit': 20, 'spread_pct': 0.02}
                }[product]
                
                position = state.position.get(product, 0)
                mid_price = None
                
                # Try to get price from order depth first
                if product in state.order_depths:
                    order_depth = state.order_depths[product]
                    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                    
                    if best_ask and best_bid:
                        mid_price = (best_ask + best_bid) / 2
                
                # If no order depth, use last trade price for this product
                if mid_price is None and state.market_trades:
                    product_trades = [t for t in state.market_trades if t['symbol'] == product]
                    if product_trades:
                        last_trade = product_trades[-1]
                        mid_price = last_trade['price']
                        print(f"Using last trade price for {product}: {mid_price}")
                
                if mid_price is not None:
                    # Dynamic spread adjustments
                    if product == 'SQUID_INK':
                        spread_pct = 0.03 + (0.01 * len(self.price_history[product])/100)  # 3-4% range
                    elif product == 'RAINFOREST_RESIN':
                        spread_pct = 0.02 - (0.002 * self.position.get(product, 0))  # Inventory skew
                    elif product == 'KELP':
                        spread_pct = 0.04 if state.market_trades else 0.02  # Tighten during activity
                    
                    spread = mid_price * spread_pct
                    
                    # Buy order
                    if position < config['position_limit']:
                        buy_price = int(mid_price - spread)
                        buy_qty = min(5, config['position_limit'] - position)
                        orders[product].append(Order(buy_price, buy_qty))
                        print(f"BUY {product}: {buy_qty} @ {buy_price}")
                        
                    # Sell order
                    if position > -config['position_limit']:
                        sell_price = int(mid_price + spread)
                        sell_qty = min(5, config['position_limit'] + position)
                        orders[product].append(Order(sell_price, -sell_qty))
                        print(f"SELL {product}: {sell_qty} @ {sell_price}")
                        
                    # Enhanced SQUID_INK strategy
                    if product == 'SQUID_INK':
                        position_limit = 15
                        
                        # Dynamic volatility scaling (0.03-0.08 range)
                        if len(self.price_history[product]) > 10:
                            vol = np.std(self.price_history[product][-10:])/mid_price
                            spread_pct = min(0.08, max(0.03, 0.03 + vol*2))  # More aggressive scaling
                        else:
                            spread_pct = 0.05
                        
                        # Incorporate trade signals
                        if state.market_trades:
                            recent_trades = [t for t in state.market_trades if t['symbol'] == product]
                            if recent_trades:
                                avg_trade_price = np.mean([t['price'] for t in recent_trades])
                                if avg_trade_price > mid_price * 1.01:
                                    acceptable_price = min(mid_price * 1.005, avg_trade_price)
                                elif avg_trade_price < mid_price * 0.99:
                                    acceptable_price = max(mid_price * 0.995, avg_trade_price)
                                else:
                                    acceptable_price = mid_price
                            else:
                                acceptable_price = mid_price
                        else:
                            acceptable_price = mid_price
                        
                        # Dynamic position adjustment
                        position_weight = 1 - (abs(self.position[product]) / position_limit)
                        bid_quantity = int(position_limit * position_weight)
                        ask_quantity = int(position_limit * position_weight)
                        
                        orders[product].append(Order(round(acceptable_price*(1-spread_pct)), bid_quantity))
                        orders[product].append(Order(round(acceptable_price*(1+spread_pct)), -ask_quantity))
                        
                    # Enhanced RAINFOREST_RESIN strategy
                    elif product == 'RAINFOREST_RESIN':
                        position_limit = 20
                        spread_pct = 0.02 + (0.01 * (abs(self.position[product])/position_limit))  
                        acceptable_price = mid_price * (1 + (0.005 * np.sign(self.position[product])))  
                        
                        if len(order_depth.sell_orders) > 0:
                            best_ask = min(order_depth.sell_orders.keys())
                            if best_ask < acceptable_price:
                                quantity = min(position_limit - self.position[product], -order_depth.sell_orders[best_ask])
                                orders[product].append(Order(product, best_ask, quantity))
                        
                        if len(order_depth.buy_orders) > 0:
                            best_bid = max(order_depth.buy_orders.keys())
                            if best_bid > acceptable_price:
                                quantity = min(position_limit + self.position[product], order_depth.buy_orders[best_bid])
                                orders[product].append(Order(product, best_bid, -quantity))
                        
                        # Market making orders
                        orders[product].append(Order(round(acceptable_price*(1-spread_pct/2)), position_limit - self.position[product]))
                        orders[product].append(Order(round(acceptable_price*(1+spread_pct/2)), -position_limit - self.position[product]))
                        
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            traceback.print_exc()
            
        return orders, conversions, trader_data
    
    def _volatile_strategy(self, state, product, volatility):
        print(f"[STRATEGY] Running volatile strategy for {product}")
        
        orders = []
        position = state.position.get(product, 0) if hasattr(state, 'position') else 0
        
        # Get current market prices
        if product not in state.order_depths:
            print("  No order depth available")
            return orders
            
        order_depth = state.order_depths[product]
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        
        if best_ask and best_bid:
            print(f"  Current prices - Ask: {best_ask}, Bid: {best_bid}")
            print(f"  Position: {position}")
            
            # More aggressive strategy during volatility
            fair_value = (best_ask + best_bid) / 2
            spread = max(1, int(volatility * fair_value))
            print(f"  Fair value: {fair_value:.2f}")
            print(f"  Spread: {spread}")
            
            # Generate orders
            if position < self.position_limit:
                buy_price = int(fair_value - spread)
                orders.append(Order(buy_price, min(30, self.position_limit - position)))
                print(f"  BUY order: {min(30, self.position_limit - position)} @ {buy_price}")
                
            if position > -self.position_limit:
                sell_price = int(fair_value + spread)
                orders.append(Order(sell_price, -min(30, self.position_limit + position)))
                print(f"  SELL order: {min(30, self.position_limit + position)} @ {sell_price}")
            
        return orders

    def _stable_strategy(self, state, product, volatility):
        print(f"[STRATEGY] Running stable strategy for {product}")
        
        orders = []
        position = state.position.get(product, 0) if hasattr(state, 'position') else 0
        
        # Get current market prices
        if product not in state.order_depths:
            print("  No order depth available")
            return orders
            
        order_depth = state.order_depths[product]
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        
        print(f"  Current prices - Ask: {best_ask}, Bid: {best_bid}")
        print(f"  Position: {position}")
        
        # Always trade at mid price if no history yet
        if len(self.price_history.get(product, [])) < 5 and best_ask and best_bid:
            mid_price = (best_ask + best_bid) / 2
            buy_price = int(mid_price * 0.99)
            sell_price = int(mid_price * 1.01)
            
            if position < self.position_limit:
                orders.append(Order(buy_price, min(10, self.position_limit - position)))
                print(f"  INITIAL BUY order: {min(10, self.position_limit - position)} @ {buy_price}")
            
            if position > -self.position_limit:
                orders.append(Order(sell_price, -min(10, self.position_limit + position)))
                print(f"  INITIAL SELL order: {min(10, self.position_limit + position)} @ {sell_price}")
            
            return orders
            
        # Normal strategy with enough history
        if len(self.price_history.get(product, [])) >= 5 and best_ask and best_bid:
            fair_value = sum(self.price_history[product][-5:]) / 5
            spread = max(1, int(volatility * fair_value * 0.75))
            print(f"  Fair value: {fair_value:.2f}")
            print(f"  Spread: {spread}")
            
            # Generate orders
            if position < self.position_limit and best_ask:
                buy_price = int(fair_value - spread)
                if buy_price >= best_ask:
                    orders.append(Order(buy_price, min(20, self.position_limit - position)))
                    print(f"  BUY order: {min(20, self.position_limit - position)} @ {buy_price}")
                
            if position > -self.position_limit and best_bid:
                sell_price = int(fair_value + spread)
                if sell_price <= best_bid:
                    orders.append(Order(sell_price, -min(20, self.position_limit + position)))
                    print(f"  SELL order: {min(20, self.position_limit + position)} @ {sell_price}")
            
        return orders

    def _calculate_position_size(self, volatility):
        base_size = 4
        risk_multiplier = 1 + (volatility * 5)  # Scale with volatility
        return min(int(base_size * risk_multiplier), self.position_limit)
    
    def _update_market_data(self, state, product):
        """Update market data and return True if successful"""
        try:
            if not state.order_depths or product not in state.order_depths:
                print(f"[MARKET] No order depth for {product}")
                return False
                
            order_depth = state.order_depths[product]
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            
            if best_ask is None or best_bid is None:
                print(f"[MARKET] No valid prices for {product}")
                return False
                
            mid_price = (best_ask + best_bid) / 2
            if product not in self.price_history:
                self.price_history[product] = []
            self.price_history[product].append(mid_price)
            
            print(f"[MARKET] Updated {product} price: {mid_price:.2f}")
            print(f"  Best ask: {best_ask}, Best bid: {best_bid}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] in _update_market_data(): {str(e)}")
            return False
    
    def _calculate_volatility(self, product):
        if len(self.price_history.get(product, [])) < 5:
            return 0
        returns = np.diff(self.price_history[product][-self.volatility_lookback:])/self.price_history[product][-self.volatility_lookback:-1]
        vol = np.nan_to_num(statistics.stdev(returns), nan=0)
        print(f"[VOL CALC] Returns: {returns[-5:]}, Vol: {vol:.4f}")
        return vol
    
    def _calculate_trend(self, product):
        if len(self.price_history.get(product, [])) < 10:
            return 0
        window = min(30, len(self.price_history[product]))
        recent = self.price_history[product][-window:]
        trend = (recent[-1] - recent[0])/recent[0]
        print(f"[TREND CALC] Trend: {trend:.4f}")
        return trend
    
    def _detect_regime(self, volatility, trend=None):
        """More sophisticated regime detection considering both volatility and trend"""
        if volatility > self.regimes['volatile']['volatility']:
            return 'volatile'
        if trend is not None and trend > self.regimes['bull']['trend'] and volatility < self.regimes['bull']['volatility']:
            return 'bull'
        if trend is not None and trend < self.regimes['bear']['trend'] and volatility < self.regimes['bear']['volatility']:
            return 'bear'
        return 'neutral'
    
    def _check_daily_loss_limit(self, state):
        """Simplified version for testing"""
        if not hasattr(state, 'position'):
            return False
            
        # Convert position to dict if needed
        positions = state.position if isinstance(state.position, dict) else {'SQUID_INK': state.position}
        
        for product, position in positions.items():
            if product in self.position:
                if len(self.price_history.get(product, [])) > 1:
                    price_change = self.price_history[product][-1] - self.price_history[product][-2]
                    self.today_pnl += (position - self.position[product]) * price_change
            self.position[product] = position
            
        return False  # Temporarily disabled
        
    def _scale_out_positions(self, position, product):
        """Gradually reduce large positions"""
        if abs(position) > self.position_limit * 0.8:
            return int(position * 0.5)  # Close half the position
        return position