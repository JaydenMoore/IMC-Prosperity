from typing import Dict, List
from modules import Order, OrderDepth, Trade, Symbol, TradingState, ProsperityEncoder
from dataclasses import dataclass

# This is a stub file with class definitions for the IMC Prosperity challenge
# In the actual environment, this would be provided by the platform

@dataclass
class Order:
    price: int
    quantity: int

@dataclass
class OrderDepth:
    buy_orders: Dict[int, int]
    sell_orders: Dict[int, int]

@dataclass
class Trade:
    symbol: str
    price: int
    quantity: int
    buyer: str
    seller: str
    timestamp: int

class Symbol:
    def __init__(self, name: str):
        self.name = name

@dataclass
class TradingState:
    timestamp: int
    listings: Dict[str, int]
    order_depths: Dict[str, OrderDepth]
    own_trades: Dict[str, List[Trade]]
    market_trades: Dict[str, List[Trade]]
    position: Dict[str, int]
    observations: Dict[str, int]
    traderData: str

class ProsperityEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Order):
            return {"price": obj.price, "quantity": obj.quantity}
        return super().default(obj)