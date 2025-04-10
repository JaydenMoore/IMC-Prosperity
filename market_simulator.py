import unittest
import random
import numpy as np
from datamodel import TradingState, OrderDepth, Trade
from main import Trader
from backtester import Backtester
import statistics

class MarketSimulator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trader = Trader()
        self.backtester = Backtester()

    def test_market_scenarios(self):
        """Test across different statistical market regimes"""
        scenarios = [
            {'name': 'Low Volatility', 'volatility': 0.02, 'trend': 0.0},
            {'name': 'High Volatility', 'volatility': 0.1, 'trend': 0.0},
            {'name': 'Bull Market', 'volatility': 0.05, 'trend': 0.2},
            {'name': 'Bear Market', 'volatility': 0.07, 'trend': -0.15}
        ]
        
        for scenario in scenarios:
            print(f"\nTesting scenario: {scenario['name']}")
            result = self._run_simulation(
                iterations=100,
                volatility=scenario['volatility'],
                trend=scenario['trend']
            )
            print(f"Final PnL: {result['final_pnl']:.2f}")
            print(f"Max Drawdown: {result['max_drawdown']:.2%}")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")

    def _run_simulation(self, iterations, volatility, trend):
        """Run statistical market simulation"""
        price = 10000  # SQUID_INK typical price
        position = 0
        cash = 0
        pnl_history = []
        trader_data = ""
        
        for i in range(iterations):
            # Generate market data with mean-reverting tendency
            price_change = price * (trend + random.gauss(0, volatility))
            price = max(5000, min(15000, price + price_change))
            
            # Create synthetic market state
            state = TradingState(
                timestamp=i,
                listings={"SQUID_INK": {"symbol": "SQUID_INK", "product": "SQUID_INK"}},
                order_depths={"SQUID_INK": self._generate_order_book(price)},
                own_trades={},
                market_trades={"SQUID_INK": [
                    Trade("SQUID_INK", price, 10, "", "", i)
                ]},
                position={},
                observations={},
                traderData=trader_data
            )
            
            # Run trader
            result, conversions, trader_data = self.trader.run(state)
            
            # Update position
            if result.get("SQUID_INK"):
                for order in result["SQUID_INK"]:
                    if order.quantity > 0:  # Buy
                        cash -= order.quantity * price
                        position += order.quantity
                    else:  # Sell
                        cash += abs(order.quantity) * price
                        position -= abs(order.quantity)
            
            # Track PnL
            pnl = cash + position * price
            pnl_history.append(pnl)
            
        # Calculate metrics with safe division
        if len(pnl_history) < 2:
            return {
                'final_pnl': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
            
        try:
            returns = np.diff(pnl_history)/np.abs(pnl_history[:-1])
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Calculate max drawdown safely
            peak = pnl_history[0]
            max_dd = 0.0
            for pnl in pnl_history:
                if pnl > peak:
                    peak = pnl
                if peak != 0:
                    dd = (peak - pnl)/peak
                    if dd > max_dd:
                        max_dd = dd
            
            # Calculate Sharpe ratio safely
            if len(returns) > 1 and statistics.stdev(returns) != 0:
                sharpe = statistics.mean(returns)/statistics.stdev(returns)
            else:
                sharpe = 0.0
            
            return {
                'final_pnl': pnl_history[-1],
                'max_drawdown': max_dd,
                'sharpe_ratio': sharpe
            }
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return {
                'final_pnl': pnl_history[-1] if pnl_history else 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }

    def _generate_order_book(self, price: float) -> OrderDepth:
        """Generate realistic order book around current price"""
        spread = price * 0.01  # 1% spread
        
        # Initialize with empty orders
        buy_orders = {}
        sell_orders = {}
        
        # Generate buy orders (bids)
        for i in range(5):
            bid_price = price * (1 - (i+1)*0.005)  # 0.5% decrements
            bid_volume = random.randint(1, 10)
            buy_orders[bid_price] = bid_volume
            
        # Generate sell orders (asks)
        for i in range(5):
            ask_price = price * (1 + (i+1)*0.005)  # 0.5% increments
            ask_volume = random.randint(1, 10)
            sell_orders[ask_price] = ask_volume
            
        return OrderDepth(buy_orders=buy_orders, sell_orders=sell_orders)

if __name__ == "__main__":
    unittest.main()