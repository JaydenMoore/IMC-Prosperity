from typing import Dict, List, Tuple
import json
import math
import pandas as pd
import numpy as np
from modules import Order, ProsperityEncoder, Symbol, Trade, TradingState, Trader
from modules import calculate_rolling_average, calculate_deviation, get_position, initialize_trader_data

class Trader:
    def __init__(self):
        # Moving averages
        self.sma_short = 10
        self.sma_long = 20
        self.ema_short = 5
        self.ema_long = 15
        
        # RSI parameters
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
        # Position sizing
        self.base_position_limit = 50
        self.volatility_factor = 1.5
        
        # Products to trade
        self.products = ["SQUID_INK", "KELP", "RAINFOREST_RESIN"]
        
        # Historical data
        self.historical_data = self.load_historical_data()
        
    def load_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Load historical price data from CSV files"""
        data_dir = "/Users/jaydenmoore/Downloads/round-1-island-data-bottle"
        files = ["prices_round_1_day_-1.csv", "prices_round_1_day_-2.csv", "prices_round_1_day_0.csv"]
        
        historical_data = {}
        
        for file in files:
            try:
                df = pd.read_csv(f"{data_dir}/{file}", delimiter=";")
                for product in self.products:
                    product_df = df[df["product"] == product].copy()
                    if not product_df.empty:
                        # Convert timestamp to datetime
                        product_df["timestamp"] = pd.to_datetime(product_df["timestamp"], unit="ms")
                        product_df.set_index("timestamp", inplace=True)
                        
                        # Add mid_price if not present
                        if "mid_price" not in product_df.columns:
                            product_df["mid_price"] = (product_df["bid_price_1"] + product_df["ask_price_1"]) / 2
                        
                        # Sort by timestamp
                        product_df = product_df.sort_index()
                        
                        if product not in historical_data:
                            historical_data[product] = product_df
                        else:
                            historical_data[product] = pd.concat([historical_data[product], product_df])
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        # Ensure we have data for all products
        for product in self.products:
            if product not in historical_data:
                historical_data[product] = pd.DataFrame(columns=["mid_price"])
        
        return historical_data
    
    def calculate_indicators(self, product: str, price_history: List[float]) -> Dict[str, float]:
        """Calculate technical indicators for a product"""
        if not price_history or len(price_history) < self.rsi_period:
            return {
                "sma_short": np.nan,
                "sma_long": np.nan,
                "ema_short": np.nan,
                "ema_long": np.nan,
                "rsi": np.nan,
                "volatility": np.nan
            }
            
        prices = pd.Series(price_history)
        
        # Moving Averages
        sma_short = prices.rolling(window=self.sma_short).mean().iloc[-1]
        sma_long = prices.rolling(window=self.sma_long).mean().iloc[-1]
        ema_short = prices.ewm(span=self.ema_short).mean().iloc[-1]
        ema_long = prices.ewm(span=self.ema_long).mean().iloc[-1]
        
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Volatility
        volatility = prices.pct_change().rolling(window=self.sma_long).std().iloc[-1]
        
        # Print indicator values for debugging
        print(f"\nIndicators for {product}:")
        print(f"SMA short: {sma_short:.2f}")
        print(f"SMA long: {sma_long:.2f}")
        print(f"EMA short: {ema_short:.2f}")
        print(f"EMA long: {ema_long:.2f}")
        print(f"RSI: {rsi:.2f}")
        print(f"Volatility: {volatility:.4f}")
        
        return {
            "sma_short": float(sma_short),
            "sma_long": float(sma_long),
            "ema_short": float(ema_short),
            "ema_long": float(ema_long),
            "rsi": float(rsi),
            "volatility": float(volatility)
        }
    
    def calculate_position_size(self, volatility: float) -> int:
        """Calculate position size based on volatility"""
        if np.isnan(volatility):
            return self.base_position_limit
            
        position_size = int(self.base_position_limit * (1 / (1 + volatility * self.volatility_factor)))
        return max(1, min(position_size, self.base_position_limit))
    
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], List[str], str]:
        """
        Execute trading logic based on market conditions
        """
        orders = {}
        conversions = []
        
        # Get existing trader data or initialize if not present
        trader_data = json.loads(state.traderData) if state.traderData else {}
        
        # Initialize price history if not present
        if "price_history" not in trader_data:
            trader_data["price_history"] = {}
        
        price_history = trader_data["price_history"]
        
        # Update price history with current trades for each product
        for product in state.market_trades:
            if product not in price_history:
                price_history[product] = []
            
            # Add current trades to price history
            for trade in state.market_trades[product]:
                price_history[product].append(trade.price)
                
                # Keep only the last N prices
                if len(price_history[product]) > self.sma_long:
                    price_history[product].pop(0)
        
        # Calculate indicators and make trading decisions for each product
        for product in state.market_trades:
            if product not in price_history or len(price_history[product]) < self.rsi_period:
                continue
                
            # Calculate indicators
            indicators = self.calculate_indicators(product, price_history[product])
            
            # Get current market conditions
            mid_price = None
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    mid_price = (best_bid + best_ask) / 2
            
            position = state.position.get(product, 0)
            position_size_limit = 49  # Maximum position size
            
            # Log market conditions
            print(f"\nMarket Conditions for {product}:")
            print(f"Current mid_price: {mid_price:.2f}") if mid_price else print("Current mid_price: None")
            print(f"Current position: {position}")
            print(f"Position size limit: {position_size_limit}")
            print(f"SMA short: {indicators['sma_short']:.2f}")
            print(f"SMA long: {indicators['sma_long']:.2f}")
            print(f"RSI: {indicators['rsi']:.2f}")
            print(f"Volatility: {indicators['volatility']:.4f}")
            
            # Check trading conditions
            trend_following_condition = indicators['sma_short'] > indicators['sma_long']
            mean_reversion_condition = indicators['rsi'] < self.rsi_oversold
            price_action_condition = mid_price > indicators['sma_short'] if mid_price else False
            
            # Log trading conditions
            print("\nTrading Conditions:")
            print(f"Trend following condition: {trend_following_condition}")
            print(f"Mean reversion condition: {mean_reversion_condition}")
            print(f"Price action condition: {price_action_condition}")
            
            # Determine order size based on position room
            position_room = position_size_limit - abs(position)
            order_size = min(position_room, 10)  # Cap order size at 10
            
            # Generate orders if any condition is met
            if trend_following_condition or mean_reversion_condition or price_action_condition:
                if mid_price and position_room > 0:
                    if trend_following_condition or price_action_condition:
                        # Buy signal
                        buy_price = min(state.order_depths[product].sell_orders.keys()) - 1
                        if buy_price in state.order_depths[product].sell_orders:
                            orders[product] = [Order(product, buy_price, order_size)]
                            print(f"\nPlacing buy order: {buy_price} x {order_size}")
                    elif mean_reversion_condition:
                        # Sell signal
                        sell_price = max(state.order_depths[product].buy_orders.keys()) + 1
                        if sell_price in state.order_depths[product].buy_orders:
                            orders[product] = [Order(product, sell_price, -order_size)]
                            print(f"\nPlacing sell order: {sell_price} x {-order_size}")
                else:
                    print("\nNo trade conditions met:")
                    if not mid_price:
                        print("  - Mid price not available")
                    if position_room <= 0:
                        print("  - No position room available")
            else:
                print("\nNo trade conditions met:")
                if not trend_following_condition:
                    print("  - Trend following condition not met")
                if not mean_reversion_condition:
                    print("  - Mean reversion condition not met")
                if not price_action_condition:
                    print("  - Price action condition not met")
            
            # Update indicators in trader data
            if "indicators" not in trader_data:
                trader_data["indicators"] = {}
            trader_data["indicators"][product] = indicators
        
        # Update trader data with new price history
        trader_data["price_history"] = price_history
        
        return orders, conversions, json.dumps(trader_data)