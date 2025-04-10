import unittest
from main import Trader
from datamodel import Order, TradingState, Trade, OrderDepth
from market_simulator import MarketSimulator

class TestTrader(MarketSimulator):
    def setUp(self):
        self.trader = Trader()
        
    def test_buy_signal(self):
        """Test buy signal conditions"""
        state = TradingState(
            traderData='{\"prices\": [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,97]}',
            timestamp=0,
            listings={},
            order_depths={"SQUID_INK": OrderDepth({}, {})},
            own_trades={},
            market_trades={"SQUID_INK": [Trade("SQUID_INK", 97, 1, "", "", 0)]},
            position={},
            observations={}
        )
        result, _, _ = self.trader.run(state)
        self.assertTrue(result and any(o.quantity > 0 for o in result["SQUID_INK"]))
        
    def test_sell_signal(self):
        """Test sell signal conditions"""
        state = TradingState(
            traderData='{\"prices\": [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,103]}',
            timestamp=0,
            listings={},
            order_depths={"SQUID_INK": OrderDepth({}, {})},
            own_trades={},
            market_trades={"SQUID_INK": [Trade("SQUID_INK", 103, 1, "", "", 0)]},
            position={},
            observations={}
        )
        result, _, _ = self.trader.run(state)
        self.assertTrue(result and any(o.quantity < 0 for o in result["SQUID_INK"]))

if __name__ == "__main__":
    unittest.main()
