import json
import random
from modules import TradingState, OrderDepth, Trade, Order
from main import Trader

class MarketSimulator:
    """
    A simple market simulator to test the SQUID_INK trading algorithm
    across various market scenarios.
    """
    
    def __init__(self, initial_price=100, price_volatility=0.05, volume_volatility=0.2, 
                 trend_strength=0.0, mean_reversion_strength=0.5):
        """
        Initialize the market simulator with parameters to control price behavior
        
        Args:
            initial_price: Starting price for SQUID_INK
            price_volatility: Controls random price movement (higher = more volatile)
            volume_volatility: Controls variation in trade volumes
            trend_strength: Adds directional bias to price movements (-1.0 to 1.0)
            mean_reversion_strength: Tendency for price to revert to initial price (0.0 to 1.0)
        """
        self.product = "SQUID_INK"
        self.current_price = initial_price
        self.initial_price = initial_price
        self.price_volatility = price_volatility
        self.volume_volatility = volume_volatility
        self.trend_strength = trend_strength
        self.mean_reversion_strength = mean_reversion_strength
        self.timestamp = 1000
        self.trader = Trader()
        self.trader_data = ""
        self.position = 0
        self.cash = 0
        self.trades = []
        
    def generate_order_book(self):
        """
        Generate a realistic order book around the current price
        """
        # Spread is wider when volatility is higher
        spread = max(1, int(self.current_price * 0.01 * (1 + self.price_volatility * 10)))
        
        # Generate bid and ask prices
        best_bid = self.current_price - spread // 2
        best_ask = self.current_price + spread // 2
        
        # Create order book with multiple levels
        buy_orders = {}
        sell_orders = {}
        
        # Generate buy orders (bids)
        for i in range(3):
            price = best_bid - i
            volume = random.randint(5, 20)
            buy_orders[price] = volume
            
        # Generate sell orders (asks)
        for i in range(3):
            price = best_ask + i
            volume = random.randint(5, 20) * -1  # Negative for sell orders
            sell_orders[price] = volume
        
        return OrderDepth(buy_orders, sell_orders)
    
    def generate_market_trades(self):
        """
        Generate recent market trades around the current price
        """
        trades = []
        num_trades = random.randint(1, 3)
        
        for i in range(num_trades):
            # Price varies around current price
            price_variance = random.uniform(-0.02, 0.02) * self.current_price
            trade_price = max(1, int(self.current_price + price_variance))
            
            # Random volume
            volume = random.randint(1, 5)
            
            # Create trade
            trades.append(Trade(
                symbol=self.product,
                price=trade_price,
                quantity=volume,
                buyer="MARKET",
                seller="MARKET",
                timestamp=self.timestamp - random.randint(1, 30)
            ))
        
        return trades
    
    def execute_trades(self, orders):
        """
        Execute the orders from the trader against the simulated market
        """
        if self.product not in orders:
            return []
            
        executed_trades = []
        
        for order in orders[self.product]:
            # Simulate partial fills and slippage
            fill_rate = random.uniform(0.8, 1.0)
            filled_quantity = int(order.quantity * fill_rate)
            
            if filled_quantity == 0:
                continue
                
            # Add some price slippage
            slippage = random.uniform(-0.005, 0.005) * order.price
            executed_price = max(1, int(order.price + slippage))
            
            # Record the trade
            trade = Trade(
                symbol=self.product,
                price=executed_price,
                quantity=filled_quantity,
                buyer="TRADER" if filled_quantity > 0 else "MARKET",
                seller="MARKET" if filled_quantity > 0 else "TRADER",
                timestamp=self.timestamp
            )
            
            executed_trades.append(trade)
            
            # Update position and cash
            self.position += filled_quantity
            self.cash -= filled_quantity * executed_price
            
        return executed_trades
    
    def update_market_price(self):
        """
        Update the market price based on configured parameters
        """
        # Random component (volatility)
        random_change = random.uniform(-1.0, 1.0) * self.price_volatility * self.current_price
        
        # Trending component
        trend_change = self.trend_strength * self.current_price * 0.01
        
        # Mean reversion component
        distance_from_mean = self.current_price - self.initial_price
        mean_reversion_change = -1 * self.mean_reversion_strength * distance_from_mean * 0.05
        
        # Combine all factors
        total_change = random_change + trend_change + mean_reversion_change
        
        # Update price, ensure it's positive
        self.current_price = max(1, int(self.current_price + total_change))
    
    def run_simulation(self, iterations=100):
        """
        Run a multi-period simulation
        
        Args:
            iterations: Number of market periods to simulate
            
        Returns:
            Dictionary with simulation results
        """
        price_history = []
        position_history = []
        pnl_history = []
        
        print(f"Starting simulation with {iterations} iterations...")
        print(f"Initial price: {self.current_price}")
        
        for i in range(iterations):
            # Update timestamp
            self.timestamp += 100
            
            # Generate market data
            order_depth = self.generate_order_book()
            market_trades = self.generate_market_trades()
            
            # Create trading state
            order_depths = {self.product: order_depth}
            market_trades_dict = {self.product: market_trades}
            position_dict = {self.product: self.position}
            listings = {self.product: 10000}  # Arbitrary listing value
            
            state = TradingState(
                timestamp=self.timestamp,
                listings=listings,
                order_depths=order_depths,
                own_trades={},
                market_trades=market_trades_dict,
                position=position_dict,
                observations={},
                traderData=self.trader_data
            )
            
            # Run the trader algorithm
            orders, conversions, self.trader_data = self.trader.run(state)
            
            # Execute trades
            executed_trades = self.execute_trades(orders)
            
            # Calculate PnL (mark to market)
            mid_price = self.current_price
            mark_to_market = self.position * mid_price + self.cash
            
            # Record history
            price_history.append(self.current_price)
            position_history.append(self.position)
            pnl_history.append(mark_to_market)
            
            # Update market price for next iteration
            self.update_market_price()
            
            # Print update every 10 iterations
            if i % 10 == 0 or i == iterations - 1:
                print(f"Iteration {i+1:3d}: Price = {self.current_price:4d}, " 
                      f"Position = {self.position:3d}, PnL = {mark_to_market:.2f}")
        
        print("\nSimulation completed!")
        print(f"Final position: {self.position}")
        print(f"Final PnL: {pnl_history[-1]:.2f}")
        
        # Return results
        return {
            "price_history": price_history,
            "position_history": position_history,
            "pnl_history": pnl_history,
            "final_position": self.position,
            "final_pnl": pnl_history[-1]
        }

def run_market_scenarios():
    """Run the algorithm through different market scenarios"""
    scenarios = [
        {
            "name": "Normal Market",
            "params": {
                "initial_price": 100,
                "price_volatility": 0.03,
                "trend_strength": 0.0,
                "mean_reversion_strength": 0.3
            }
        },
        {
            "name": "Trending Up Market",
            "params": {
                "initial_price": 100,
                "price_volatility": 0.03,
                "trend_strength": 0.2,
                "mean_reversion_strength": 0.1
            }
        },
        {
            "name": "Trending Down Market",
            "params": {
                "initial_price": 100,
                "price_volatility": 0.03,
                "trend_strength": -0.2,
                "mean_reversion_strength": 0.1
            }
        },
        {
            "name": "Highly Volatile Market",
            "params": {
                "initial_price": 100,
                "price_volatility": 0.10,
                "trend_strength": 0.0,
                "mean_reversion_strength": 0.3
            }
        },
        {
            "name": "Mean-Reverting Market",
            "params": {
                "initial_price": 100,
                "price_volatility": 0.05,
                "trend_strength": 0.0,
                "mean_reversion_strength": 0.8
            }
        }
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n{'=' * 60}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"{'=' * 60}")
        
        simulator = MarketSimulator(**scenario["params"])
        result = simulator.run_simulation(iterations=50)
        results[scenario["name"]] = result
    
    return results

if __name__ == "__main__":
    # Run simulations across different market scenarios
    scenario_results = run_market_scenarios()
    
    # Summary of results
    print("\n\nSCENARIO RESULTS SUMMARY:")
    print(f"{'Scenario':<25} {'Final PnL':>10} {'Final Position':>15}")
    print(f"{'-' * 25} {'-' * 10} {'-' * 15}")
    
    for name, result in scenario_results.items():
        print(f"{name:<25} {result['final_pnl']:>10.2f} {result['final_position']:>15}")