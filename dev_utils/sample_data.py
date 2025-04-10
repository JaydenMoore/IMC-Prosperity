import json
import random
from modules import OrderDepth, Trade, TradingState

class SampleDataGenerator:
    """
    Generates sample data for testing the SQUID_INK trading algorithm.
    This class provides various market scenarios to test algorithm behavior.
    """
    
    def __init__(self):
        self.product = "SQUID_INK"
        
    def generate_basic_dataset(self):
        """
        Generate a basic dataset with a simple order book and some market trades
        """
        order_depth = OrderDepth(
            buy_orders={95: 10, 94: 20, 93: 15},  # Bids
            sell_orders={97: -10, 98: -15, 99: -20}  # Asks
        )
        
        market_trades = [
            Trade(self.product, 96, 5, "MARKET", "TRADER1", 1000),
            Trade(self.product, 96, 3, "TRADER2", "MARKET", 1001),
            Trade(self.product, 97, 2, "MARKET", "TRADER3", 1002)
        ]
        
        return {
            "timestamp": 1003,
            "order_depths": {self.product: order_depth},
            "market_trades": {self.product: market_trades},
            "position": {self.product: 0},
            "trader_data": ""
        }
        
    def generate_trending_up_dataset(self):
        """
        Generate a dataset that simulates an upward trending market
        """
        # Create historical data showing prices trending up
        historical_data = {
            "price_history": {
                self.product: [90, 91, 93, 94, 95, 97, 98, 99, 100, 101]
            },
            "metrics": {}
        }
        
        order_depth = OrderDepth(
            buy_orders={102: 15, 101: 25, 100: 20},  # Bids higher than historical average
            sell_orders={103: -10, 104: -15, 105: -20}
        )
        
        market_trades = [
            Trade(self.product, 102, 4, "TRADER1", "MARKET", 1100),
            Trade(self.product, 103, 2, "MARKET", "TRADER2", 1101),
            Trade(self.product, 102, 3, "TRADER3", "MARKET", 1102)
        ]
        
        return {
            "timestamp": 1103,
            "order_depths": {self.product: order_depth},
            "market_trades": {self.product: market_trades},
            "position": {self.product: 0},
            "trader_data": json.dumps(historical_data)
        }
    
    def generate_trending_down_dataset(self):
        """
        Generate a dataset that simulates a downward trending market
        """
        # Create historical data showing prices trending down
        historical_data = {
            "price_history": {
                self.product: [110, 108, 107, 105, 104, 103, 102, 101, 100, 99]
            },
            "metrics": {}
        }
        
        order_depth = OrderDepth(
            buy_orders={92: 10, 91: 15, 90: 20},
            sell_orders={93: -15, 94: -20, 95: -10}  # Asks lower than historical average
        )
        
        market_trades = [
            Trade(self.product, 93, 3, "MARKET", "TRADER1", 1200),
            Trade(self.product, 92, 4, "TRADER2", "MARKET", 1201),
            Trade(self.product, 92, 2, "TRADER3", "MARKET", 1202)
        ]
        
        return {
            "timestamp": 1203,
            "order_depths": {self.product: order_depth},
            "market_trades": {self.product: market_trades},
            "position": {self.product: 0},
            "trader_data": json.dumps(historical_data)
        }
    
    def generate_volatile_dataset(self):
        """
        Generate a dataset that simulates a volatile market
        """
        # Create historical data showing volatile prices
        historical_data = {
            "price_history": {
                self.product: [100, 107, 95, 103, 92, 108, 94, 105, 98, 102]
            },
            "metrics": {}
        }
        
        order_depth = OrderDepth(
            buy_orders={95: 25, 94: 30, 93: 20},  # Wide spread
            sell_orders={105: -25, 106: -15, 107: -20}
        )
        
        market_trades = [
            Trade(self.product, 104, 5, "TRADER1", "MARKET", 1300),
            Trade(self.product, 96, 6, "MARKET", "TRADER2", 1301),
            Trade(self.product, 105, 3, "TRADER3", "MARKET", 1302)
        ]
        
        return {
            "timestamp": 1303,
            "order_depths": {self.product: order_depth},
            "market_trades": {self.product: market_trades},
            "position": {self.product: 0},
            "trader_data": json.dumps(historical_data)
        }
    
    def generate_position_limit_test_dataset(self):
        """
        Generate a dataset to test position limits
        """
        # Create historical data with price below current market
        historical_data = {
            "price_history": {
                self.product: [120, 118, 119, 120, 121, 119, 120, 118, 119, 120]  # Avg: ~119.4
            },
            "metrics": {}
        }
        
        # Current price is much lower - should trigger buys but we're close to position limit
        order_depth = OrderDepth(
            buy_orders={109: 20, 108: 15, 107: 25},
            sell_orders={110: -30, 111: -20, 112: -15}
        )
        
        market_trades = [
            Trade(self.product, 110, 5, "MARKET", "TRADER1", 1400),
            Trade(self.product, 109, 4, "TRADER2", "MARKET", 1401),
        ]
        
        return {
            "timestamp": 1403,
            "order_depths": {self.product: order_depth},
            "market_trades": {self.product: market_trades},
            "position": {self.product: 45},  # Close to position limit
            "trader_data": json.dumps(historical_data)
        }
        
    def generate_missing_data_test(self):
        """
        Generate a dataset with missing or incomplete market data
        """
        # Create empty order book for SQUID_INK
        order_depth = OrderDepth({}, {})
        
        # No market trades
        market_trades = []
        
        return {
            "timestamp": 1500,
            "order_depths": {self.product: order_depth},
            "market_trades": {self.product: market_trades},
            "position": {self.product: 10},
            "trader_data": ""
        }
        
    def generate_rapid_reversal_dataset(self):
        """
        Generate a dataset that simulates a rapid price reversal
        """
        # Historical data showing uptrend
        historical_data = {
            "price_history": {
                self.product: [90, 92, 95, 98, 100, 103, 105, 107, 108, 110]
            },
            "metrics": {}
        }
        
        # But current book shows severe drop
        order_depth = OrderDepth(
            buy_orders={90: 30, 89: 40, 88: 35},
            sell_orders={91: -25, 92: -30, 93: -20}
        )
        
        # Recent trades show the crash
        market_trades = [
            Trade(self.product, 100, 10, "MARKET", "TRADER1", 1600),
            Trade(self.product, 95, 15, "MARKET", "TRADER2", 1601),
            Trade(self.product, 92, 20, "MARKET", "TRADER3", 1602),
            Trade(self.product, 91, 10, "MARKET", "TRADER4", 1603)
        ]
        
        return {
            "timestamp": 1604,
            "order_depths": {self.product: order_depth},
            "market_trades": {self.product: market_trades},
            "position": {self.product: -20},  # We're short from before
            "trader_data": json.dumps(historical_data)
        }
        
    def create_trading_state(self, dataset):
        """
        Convert a dataset dictionary into a TradingState object
        """
        return TradingState(
            timestamp=dataset["timestamp"],
            listings={self.product: 10000},  # Default listing
            order_depths=dataset["order_depths"],
            own_trades={},
            market_trades=dataset["market_trades"] if "market_trades" in dataset else {},
            position=dataset["position"],
            observations={},
            traderData=dataset["trader_data"]
        )
        
    def generate_all_test_cases(self):
        """
        Generate all test cases and return as TradingState objects
        """
        test_cases = {
            "basic": self.generate_basic_dataset(),
            "trending_up": self.generate_trending_up_dataset(),
            "trending_down": self.generate_trending_down_dataset(),
            "volatile": self.generate_volatile_dataset(),
            "position_limit": self.generate_position_limit_test_dataset(),
            "missing_data": self.generate_missing_data_test(),
            "rapid_reversal": self.generate_rapid_reversal_dataset()
        }
        
        return {name: self.create_trading_state(data) for name, data in test_cases.items()}

def run_test_suite():
    """
    Run all test cases against the trader algorithm
    """
    from main import Trader
    
    generator = SampleDataGenerator()
    test_cases = generator.generate_all_test_cases()
    trader = Trader()
    
    results = {}
    
    print("Running test suite on SQUID_INK trading algorithm...\n")
    print(f"{'Test Case':<20} {'Orders':<30} {'Position':<10}")
    print(f"{'-'*20} {'-'*30} {'-'*10}")
    
    for name, state in test_cases.items():
        # Get current position
        position = state.position.get("SQUID_INK", 0)
        
        # Run the trader algorithm
        orders, conversions, trader_data = trader.run(state)
        
        # Format order summary
        order_summary = "None"
        if "SQUID_INK" in orders and orders["SQUID_INK"]:
            orders_list = orders["SQUID_INK"]
            order_summary = ", ".join([f"{o.quantity}@{o.price}" for o in orders_list])
            
        # Store and print results
        results[name] = {
            "state": state,
            "orders": orders,
            "trader_data": trader_data
        }
        
        print(f"{name:<20} {order_summary:<30} {position:<10}")
    
    print("\nTest suite completed!")
    return results

if __name__ == "__main__":
    results = run_test_suite()