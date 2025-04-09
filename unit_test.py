from main import Trader
from modules import TradingState, OrderDepth, Trade
import pandas as pd
import json
from datetime import datetime

def create_order_depth(bid_price, bid_volume, ask_price, ask_volume):
    return OrderDepth({bid_price: bid_volume}, {ask_price: ask_volume})

def create_market_trade(product, price, volume, timestamp):
    return [Trade(product, price, volume, "MARKET", "TRADER1", timestamp)]

def create_position(product, volume):
    return {product: volume}

def create_listing(product, quantity):
    return {product: quantity}

def create_trader_data(price_history=None, indicators=None):
    return json.dumps({
        "price_history": price_history if price_history else {},
        "indicators": indicators if indicators else {}
    })

def load_historical_data():
    """Load historical data from CSV files"""
    data_dir = "/Users/jaydenmoore/Downloads/round-1-island-data-bottle"
    files = ["prices_round_1_day_-1.csv", "prices_round_1_day_-2.csv", "prices_round_1_day_0.csv"]
    
    historical_data = {}
    
    for file in files:
        try:
            df = pd.read_csv(f"{data_dir}/{file}", delimiter=";")
            for product in ["SQUID_INK", "KELP", "RAINFOREST_RESIN"]:
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
    for product in ["SQUID_INK", "KELP", "RAINFOREST_RESIN"]:
        if product not in historical_data:
            historical_data[product] = pd.DataFrame(columns=["mid_price"])
    
    return historical_data

def create_trading_state(historical_data, timestamp):
    """Create a TradingState with historical data"""
    order_depths = {}
    market_trades = {}
    position = {}
    listings = {}
    
    for product in ["SQUID_INK", "KELP", "RAINFOREST_RESIN"]:
        if product in historical_data:
            product_data = historical_data[product]
            if not product_data.empty:
                # Get latest data point
                latest = product_data.iloc[-1]
                
                # Create order depth
                bid_price = latest["bid_price_1"]
                bid_volume = latest["bid_volume_1"]
                ask_price = latest["ask_price_1"]
                ask_volume = latest["ask_volume_1"]
                
                order_depths[product] = create_order_depth(bid_price, bid_volume, ask_price, ask_volume)
                
                # Create market trades
                market_trades[product] = create_market_trade(product, latest["mid_price"], latest["bid_volume_1"], timestamp)
                
                # Initialize position and listings
                position[product] = 0
                listings[product] = 100000
    
    return TradingState(
        timestamp=timestamp,
        listings=listings,
        order_depths=order_depths,
        own_trades={},
        market_trades=market_trades,
        position=position,
        observations={},
        traderData=""
    )

def test_trading_algorithm():
    # Load historical data
    historical_data = load_historical_data()
    
    # Initialize trader
    trader = Trader()
    
    # Test different timestamps
    test_timestamps = [
        pd.to_datetime("2025-04-08 00:00:00"),  # Start of day
        pd.to_datetime("2025-04-08 00:05:00"),  # Mid-day
        pd.to_datetime("2025-04-08 00:10:00")   # End of day
    ]
    
    for timestamp in test_timestamps:
        # Create trading state
        state = create_trading_state(historical_data, timestamp)
        
        # Run trader
        orders, conversions, trader_data = trader.run(state)
        
        # Print results
        print(f"\nResults for timestamp: {timestamp}")
        print(f"Orders: {orders}")
        print(f"Conversions: {conversions}")
        print(f"Trader Data: {trader_data}")

if __name__ == "__main__":
    test_trading_algorithm()