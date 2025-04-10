# submission.py - IMC Prosperity Competition Submission
from datamodel import Order, OrderDepth, TradingState
import numpy as np
from typing import Dict, List
import json

def calculate_rolling_average(prices):
    """Calculate simple moving average of price history"""
    return sum(prices)/len(prices) if prices else 0

def initialize_trader_data(trader_data_str):
    """Initialize or parse trader data string"""
    if not trader_data_str:
        return {"price_history": {}}
    return json.loads(trader_data_str)

class Trader:
    def __init__(self):
        self.position_limits = {
            'SQUID_INK': 40,
            'KELP': 30,
            'RAINFOREST_RESIN': 20
        }
        self.position = {product: 0 for product in self.position_limits.keys()}
        self.price_history = {product: [] for product in self.position_limits.keys()}
        self.pnl = {product: 0.0 for product in self.position_limits.keys()}  # Track P&L per product
        
        # Advanced features initialization
        self.volatility_window = 10  # For regime detection
        self.base_trade_size = 5
        self.product_correlations = {
            ('SQUID_INK', 'KELP'): 0.82  # Historical correlation
        }
        
    def log_state(self, state: TradingState):
        print(f"=== TRADING STATE @ {state.timestamp} ===")
        for product in self.position_limits:
            if product in state.order_depths:
                bids = sorted(state.order_depths[product].buy_orders.items(), reverse=True)
                asks = sorted(state.order_depths[product].sell_orders.items())
                print(f"{product}:")
                print(f"  Bids: {bids[:3]}..." if len(bids) > 3 else f"  Bids: {bids}")
                print(f"  Asks: {asks[:3]}..." if len(asks) > 3 else f"  Asks: {asks}")
                print(f"  Position: {state.position.get(product, 0)}/{self.position_limits[product]}")
                print(f"  P&L: ${self.pnl[product]:.2f}")  # Show current P&L
                
    def update_pnl(self, product: str, price: float, quantity: int, is_buy: bool):
        """Update P&L when orders are executed"""
        if is_buy:
            self.pnl[product] -= abs(quantity) * price
        else:
            self.pnl[product] += abs(quantity) * price

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        self.log_state(state)
        orders: Dict[str, List[Order]] = {}
        conversions = 0
        trader_data = ""

        # Update price history
        for product in self.position_limits:
            if product in state.order_depths:
                mid_price = (min(state.order_depths[product].sell_orders.keys()) + 
                            max(state.order_depths[product].buy_orders.keys())) / 2
                self.price_history[product].append(mid_price)

        try:
            for product in self.position_limits:
                if product not in state.order_depths:
                    continue
                    
                order_depth = state.order_depths[product]
                current_pos = state.position.get(product, 0)
                
                # Market regime detection
                volatility = 0
                if len(self.price_history[product]) >= self.volatility_window:
                    volatility = np.std(self.price_history[product][-self.volatility_window:])
                
                # Dynamic position sizing
                trade_multiplier = 1 + min(volatility / 10, 2)  # 1-3x base size
                max_trade = min(
                    int(self.base_trade_size * trade_multiplier),
                    self.position_limits[product] - abs(current_pos)
                )
                
                # Core trading logic
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                
                if max_trade > 0:
                    # Buy order
                    orders[product] = [Order(
                        symbol=product,
                        price=best_bid,
                        quantity=max_trade
                    )]
                    self.update_pnl(product, best_bid, max_trade, is_buy=True)
                    
                    # Sell order
                    orders[product].append(Order(
                        symbol=product,
                        price=best_ask,
                        quantity=-max_trade
                    ))
                    self.update_pnl(product, best_ask, max_trade, is_buy=False)
                
                # Statistical arbitrage
                for (prod1, prod2), corr in self.product_correlations.items():
                    if product == prod1 and prod2 in state.order_depths:
                        spread = best_bid - max(state.order_depths[prod2].sell_orders.keys())
                        if spread > 2 * volatility:  # Significant divergence
                            arb_size = min(3, self.position_limits[prod1] - abs(current_pos))
                            if arb_size > 0:
                                orders[product].append(Order(
                                    symbol=product,
                                    price=best_bid,
                                    quantity=arb_size
                                ))

        except Exception as e:
            print(f"[ERROR] {str(e)}")
        
        return orders, conversions, trader_data