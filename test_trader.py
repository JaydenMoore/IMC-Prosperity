import unittest
import json
from main import Trader
from modules import TradingState, OrderDepth, Trade, Order
import pandas as pd

class TestSQUIDINKStrategy(unittest.TestCase):
    def setUp(self):
        self.trader = Trader()
        self.product = "SQUID_INK"
        
        # Load historical data
        self.trader.historical_data = self.trader.load_historical_data()
        
        # Ensure we have data for our product
        if self.product not in self.trader.historical_data:
            self.trader.historical_data[self.product] = pd.DataFrame(columns=["mid_price"])
            
        # Print basic information about the data
        print(f"\nHistorical Data for {self.product}:")
        print(f"Number of data points: {len(self.trader.historical_data[self.product])}")
        if not self.trader.historical_data[self.product].empty:
            min_price = self.trader.historical_data[self.product]["mid_price"].min()
            max_price = self.trader.historical_data[self.product]["mid_price"].max()
            print(f"Price range: {min_price:.2f} to {max_price:.2f}")

    def create_state(self, timestamp: int, buy_orders: dict, sell_orders: dict,
                    market_trades: list, position: int, trader_data: str = None) -> TradingState:
        """Create a TradingState object for testing"""
        order_depths = {self.product: OrderDepth(buy_orders, sell_orders)}
        market_trades_dict = {self.product: market_trades} if market_trades else {}
        position_dict = {self.product: position} if position is not None else {}
        
        return TradingState(
            timestamp=timestamp,
            listings={},
            order_depths=order_depths,
            own_trades={},
            market_trades=market_trades_dict,
            position=position_dict,
            observations={},
            traderData=trader_data
        )

    def create_market_trade(self, price, quantity, timestamp, side="BUY"):
        """Helper method to create a market trade"""
        return Trade(
            symbol=self.product,
            price=price,
            quantity=quantity,
            buyer="TEST" if side == "BUY" else None,
            seller="TEST" if side == "SELL" else None,
            timestamp=timestamp
        )

    def test_initial_data_collection(self):
        """Test that the algorithm collects data before trading"""
        state = self.create_state(
            timestamp=1000,
            buy_orders={1800: 5, 1790: 10},
            sell_orders={1820: -5, 1830: -10},
            market_trades=[],
            position=0
        )
        
        orders, conversions, trader_data = self.trader.run(state)
        
        # Should have no orders initially
        self.assertEqual(len(orders), 0)
        
        # Should have collected the price history
        data = json.loads(trader_data)
        self.assertIn(self.product, data.get("price_history", {}))
        self.assertEqual(len(data.get("price_history", {}).get(self.product, [])), 0)

    def test_insufficient_data(self):
        """Test behavior when there's insufficient historical data"""
        state = self.create_state(
            timestamp=1000,
            buy_orders={1800: 5, 1790: 10},
            sell_orders={1820: -5, 1830: -10},
            market_trades=[],
            position=0
        )
        
        orders, conversions, trader_data = self.trader.run(state)
        
        # Should have no orders due to insufficient data
        self.assertEqual(len(orders), 0)
        
        # Should have collected the price history
        data = json.loads(trader_data)
        self.assertIn(self.product, data.get("price_history", {}))
        self.assertEqual(len(data.get("price_history", {}).get(self.product, [])), 0)

    def test_mean_reversion_buy_signal(self):
        """Test that the algorithm buys when price is below average"""
        # Create historical data showing oversold condition
        historical_data = {
            "price_history": {
                self.product: [1850] * 10 + [1800] * 5 + [1850] * 5 + [1800] * 5
            },
            "indicators": {}
        }
        
        state = self.create_state(
            timestamp=1000,
            buy_orders={1810: 5, 1800: 10},
            sell_orders={1820: -5, 1830: -10},
            market_trades=[],
            position=0,
            trader_data=json.dumps(historical_data)
        )
        
        orders, conversions, trader_data = self.trader.run(state)
        
        # Should have placed a buy order
        self.assertIn(self.product, orders)
        self.assertEqual(len(orders[self.product]), 1)
        self.assertGreater(orders[self.product][0].price, 1800)
        self.assertLess(orders[self.product][0].price, 1820)

    def test_mean_reversion_sell_signal(self):
        """Test that the algorithm sells when price is above average"""
        # Create historical data showing overbought condition
        historical_data = {
            "price_history": {
                self.product: [1800] * 10 + [1850] * 5 + [1800] * 5 + [1850] * 5
            },
            "indicators": {}
        }
        
        state = self.create_state(
            timestamp=1000,
            buy_orders={1840: 5, 1830: 10},
            sell_orders={1860: -5, 1870: -10},
            market_trades=[],
            position=0,
            trader_data=json.dumps(historical_data)
        )
        
        orders, conversions, trader_data = self.trader.run(state)
        
        # Should have placed a sell order
        self.assertIn(self.product, orders)
        self.assertEqual(len(orders[self.product]), 1)
        self.assertGreater(orders[self.product][0].price, 1840)
        self.assertLess(orders[self.product][0].price, 1860)

    def test_position_limit_long(self):
        """Test that the algorithm respects position limits when going long"""
        # Create historical data showing strong buy signal
        historical_data = {
            "price_history": {
                self.product: [1800] * 10 + [1900] * 5 + [2000] * 5
            },
            "indicators": {}
        }
        
        state = self.create_state(
            timestamp=1000,
            buy_orders={1950: 5, 1940: 10},
            sell_orders={1960: -5, 1970: -10},
            market_trades=[],
            position=45,  # Close to position limit
            trader_data=json.dumps(historical_data)
        )
        
        orders, conversions, trader_data = self.trader.run(state)
        
        # Should have placed a buy order but with reduced size
        self.assertIn(self.product, orders)
        self.assertEqual(len(orders[self.product]), 1)
        self.assertLessEqual(orders[self.product][0].quantity, 4)  # 49 - 45 = 4

    def test_position_limit_short(self):
        """Test that the algorithm respects position limits when going short"""
        # Create historical data showing strong sell signal
        historical_data = {
            "price_history": {
                self.product: [2000] * 10 + [1900] * 5 + [1800] * 5
            },
            "indicators": {}
        }
        
        state = self.create_state(
            timestamp=1000,
            buy_orders={1850: 5, 1840: 10},
            sell_orders={1870: -5, 1880: -10},
            market_trades=[],
            position=-45,  # Close to position limit
            trader_data=json.dumps(historical_data)
        )
        
        orders, conversions, trader_data = self.trader.run(state)
        
        # Should have placed a sell order but with reduced size
        self.assertIn(self.product, orders)
        self.assertEqual(len(orders[self.product]), 1)
        self.assertLessEqual(abs(orders[self.product][0].quantity), 4)  # 49 - 45 = 4

    def test_position_sizing(self):
        """Test that the algorithm sizes orders correctly based on position room"""
        # Create historical data showing strong buy signal
        historical_data = {
            "price_history": {
                self.product: [1800] * 10 + [1900] * 5 + [2000] * 5
            },
            "indicators": {}
        }
        
        state = self.create_state(
            timestamp=1000,
            buy_orders={1950: 5, 1940: 10},
            sell_orders={1960: -5, 1970: -10},
            market_trades=[],
            position=20,  # Some position already held
            trader_data=json.dumps(historical_data)
        )
        
        orders, conversions, trader_data = self.trader.run(state)
        
        # Should have placed a buy order with reduced size
        self.assertIn(self.product, orders)
        self.assertEqual(len(orders[self.product]), 1)
        self.assertLessEqual(orders[self.product][0].quantity, 29)  # 49 - 20 = 29

    def test_price_action_signal(self):
        """Test that price action signals are detected correctly"""
        # Create historical data showing price above SMA
        historical_data = {
            "price_history": {
                self.product: [1800] * 10 + [1850] * 5 + [1900] * 5 + [1950] * 5
            },
            "indicators": {}
        }
        
        state = self.create_state(
            timestamp=1000,
            buy_orders={1940: 5, 1930: 10},
            sell_orders={1960: -5, 1970: -10},
            market_trades=[],
            position=0,
            trader_data=json.dumps(historical_data)
        )
        
        orders, conversions, trader_data = self.trader.run(state)
        
        # Should have placed a buy order
        self.assertIn(self.product, orders)
        self.assertEqual(len(orders[self.product]), 1)
        self.assertGreater(orders[self.product][0].price, 1930)
        self.assertLess(orders[self.product][0].price, 1960)

    def test_trend_following_signal(self):
        """Test that trend following signals are detected correctly"""
        # Create historical data showing an uptrend
        historical_data = {
            "price_history": {
                self.product: [1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940,
                             1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090]
            },
            "indicators": {}
        }
        
        state = self.create_state(
            timestamp=1000,
            buy_orders={2080: 5, 2070: 10},
            sell_orders={2100: -5, 2110: -10},
            market_trades=[],
            position=0,
            trader_data=json.dumps(historical_data)
        )
        
        orders, conversions, trader_data = self.trader.run(state)
        
        # Check that trend following condition is met
        data = json.loads(trader_data)
        indicators = data.get("indicators", {}).get(self.product, {})
        self.assertTrue(indicators.get("sma_short", 0) > indicators.get("sma_long", 0))
        self.assertTrue(indicators.get("ema_short", 0) > indicators.get("ema_long", 0))

    def test_realistic_market_scenario(self):
        """Test a more realistic market scenario with price movement"""
        # Initial state with downward trend
        state1 = self.create_state(
            timestamp=1000,
            buy_orders={1800: 5, 1790: 10},
            sell_orders={1820: -5, 1830: -10},
            market_trades=[],
            position=0
        )
        
        # Run first time
        orders1, conversions1, trader_data1 = self.trader.run(state1)
        
        # Second state with price reversal
        state2 = self.create_state(
            timestamp=1001,
            buy_orders={1850: 5, 1840: 10},
            sell_orders={1870: -5, 1880: -10},
            market_trades=[],
            position=0,
            trader_data=trader_data1
        )
        
        # Run second time
        orders2, conversions2, trader_data2 = self.trader.run(state2)
        
        # Third state with upward trend
        state3 = self.create_state(
            timestamp=1002,
            buy_orders={1900: 5, 1890: 10},
            sell_orders={1920: -5, 1930: -10},
            market_trades=[],
            position=0,
            trader_data=trader_data2
        )
        
        # Run third time
        orders3, conversions3, trader_data3 = self.trader.run(state3)
        
        # Check that orders were placed appropriately
        self.assertIn(self.product, orders1)
        self.assertIn(self.product, orders2)
        self.assertIn(self.product, orders3)

    def test_price_history_update(self):
        """Test that price history is properly updated with new trades"""
        # Create initial state
        state = self.create_state(
            timestamp=1000,
            buy_orders={1800: 5, 1790: 10},
            sell_orders={1820: -5, 1830: -10},
            market_trades=[],
            position=0
        )
        
        # Run once to initialize
        orders1, conversions1, trader_data1 = self.trader.run(state)
        
        # Create state with new trades
        trades = [self.create_market_trade(1810, 5, 1001)]
        state2 = self.create_state(
            timestamp=1001,
            buy_orders={1800: 5, 1790: 10},
            sell_orders={1820: -5, 1830: -10},
            market_trades=trades,
            position=0,
            trader_data=trader_data1
        )
        
        # Run again with new trades
        orders2, conversions2, trader_data2 = self.trader.run(state2)
        
        # Check that price history was updated
        data = json.loads(trader_data2)
        self.assertIn(self.product, data.get("price_history", {}))
        self.assertEqual(len(data.get("price_history", {}).get(self.product, [])), 1)
        self.assertEqual(data.get("price_history", {}).get(self.product, [])[0], 1810)

if __name__ == "__main__":
    unittest.main()