from json import JSONEncoder
from typing import Dict, List, Any, Optional
from modules import Symbol
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
    traderData: str
    timestamp: int
    listings: Dict[str, Any]
    order_depths: Dict[str, OrderDepth]
    own_trades: Dict[str, List[Trade]]
    market_trades: Dict[str, List[Trade]]
    position: Dict[str, int]
    observations: Dict[str, Any]

class ProsperityEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, Order) or isinstance(o, OrderDepth) or isinstance(o, Trade) or isinstance(o, TradingState):
            return o.__dict__
        return super().default(o)