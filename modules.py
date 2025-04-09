from typing import Dict, List, Tuple
from dataclasses import dataclass
from json import JSONEncoder, dumps, loads

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

# Utility functions
def calculate_rolling_average(prices: List[float]) -> float:
    """
    Calculate the rolling average of a list of prices.
    
    Args:
        prices: List of historical prices
        
    Returns:
        The rolling average price
    """
    if not prices:
        return 0
    
    return sum(prices) / len(prices)

def calculate_deviation(current_price: float, average_price: float) -> float:
    """
    Calculate the deviation of the current price from the average price.
    
    Args:
        current_price: Current market price
        average_price: Rolling average price
        
    Returns:
        Deviation as a decimal (e.g., 0.05 means 5% above average)
    """
    if average_price == 0:
        return 0
    
    return (current_price - average_price) / average_price

def get_position(state: TradingState, product: Symbol) -> int:
    """
    Get the current position for a product.
    
    Args:
        state: TradingState object
        product: Product symbol
        
    Returns:
        Current position (positive for long, negative for short)
    """
    if state.position and product in state.position:
        return state.position[product]
    return 0

def initialize_trader_data(trader_data_str: str) -> Dict:
    """
    Initialize or load trader data from JSON string.
    
    Args:
        trader_data_str: JSON string from previous iteration
        
    Returns:
        Dictionary with trader data
    """
    if trader_data_str:
        try:
            return loads(trader_data_str)
        except:
            print("Error decoding trader data, initializing new data")
    
    return {
        "price_history": {},
        "metrics": {}
    }

class Trader:
    def __init__(self):
        self.window_size = 10  # Size of rolling window for price history
        self.base_deviation_threshold = 0.03  # Base threshold for trade decisions
        self.position_limit = 50  # Maximum position (long or short)
        self.product = "SQUID_INK"  # Product we're trading
        self.min_order_size = 5  # Minimum order size
        self.max_order_size = 25  # Maximum order size
        self.stop_loss_threshold = 0.05  # 5% stop loss
        self.profit_take_threshold = 0.10  # 10% profit take
        
    def calculate_order_size(self, price_history, current_position, available_volume):
        """
        Calculate optimal order size based on:
        - Current position
        - Available liquidity
        - Market volatility
        - Risk management constraints
        """
        # Calculate market volatility
        if len(price_history) > 1:
            price_changes = [abs(price_history[i] - price_history[i-1]) 
                           for i in range(1, len(price_history))]
            volatility = sum(price_changes) / len(price_changes)
        else:
            volatility = 1.0
            
        # Adjust deviation threshold based on volatility
        deviation_threshold = self.base_deviation_threshold * (1 + volatility / 100)
        
        # Calculate maximum position change based on volatility
        max_position_change = int(self.position_limit * (1 - volatility / 100))
        
        # Calculate available position room
        if current_position >= 0:  # Long position
            position_room = self.position_limit - current_position
        else:  # Short position
            position_room = self.position_limit + current_position
            
        # Calculate order size
        order_size = min(
            max(self.min_order_size, int(position_room * 0.3)),  # 30% of position room
            self.max_order_size,
            available_volume,
            max_position_change
        )
        
        return order_size, deviation_threshold
        
    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], Dict[Symbol, List[Order]], str]:
        """
        Only method required by the IMC Prosperity trading engine.
        
        Args:
            state: TradingState object with market data
            
        Returns:
            orders: Orders to be placed on exchange
            conversions: Conversions to be performed
            trader_data: JSON string representing state to be carried over to next iteration
        """
        # Initialize the method output
        orders: Dict[Symbol, List[Order]] = {}
        conversions = {}
        
        # Initialize or load trader data
        trader_data = initialize_trader_data(state.traderData)
        if self.product not in trader_data["price_history"]:
            trader_data["price_history"][self.product] = []
        
        # Shorthand for price history
        price_history = trader_data["price_history"][self.product]
        
        # Update price history with new market trades
        if self.product in state.market_trades:
            for trade in state.market_trades[self.product]:
                price_history.append(trade.price)
        
        # Trim price history to window size
        while len(price_history) > self.window_size:
            price_history.pop(0)
        
        # If we don't have enough data yet, don't trade
        if len(price_history) < self.window_size / 2:
            print(f"Not enough price history for {self.product}. Waiting for more data.")
            return {}, {}, dumps(trader_data)
        
        # Calculate rolling average
        rolling_avg = calculate_rolling_average(price_history)
        
        # Get current position
        current_position = get_position(state, self.product)
        
        # Get order book for SQUID_INK
        product_orders = []
        if self.product in state.order_depths:
            order_depth = state.order_depths[self.product]
            
            # Get best bid and ask
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            
            if best_bid is not None and best_ask is not None:
                mid_price = (best_bid + best_ask) / 2
                
                # Calculate deviation
                deviation = calculate_deviation(mid_price, rolling_avg)
                
                # Calculate order size and adjusted threshold
                if deviation < 0:  # Buying
                    available_volume = abs(order_depth.sell_orders[best_ask])
                else:  # Selling
                    available_volume = abs(order_depth.buy_orders[best_bid])
                    
                order_size, deviation_threshold = self.calculate_order_size(
                    price_history, current_position, available_volume
                )
                
                print(f"SQUID_INK - Rolling Avg: {rolling_avg}, Current Mid: {mid_price}, "
                      f"Deviation: {deviation:.4f}, Available Vol: {available_volume}")
                
                # Check stop-loss
                if current_position > 0 and mid_price < rolling_avg * (1 - self.stop_loss_threshold):
                    print(f"Stop-loss triggered: SELL {current_position} SQUID_INK at {best_bid}")
                    product_orders.append(Order(best_bid, -current_position))
                elif current_position < 0 and mid_price > rolling_avg * (1 + self.stop_loss_threshold):
                    print(f"Stop-loss triggered: BUY {-current_position} SQUID_INK at {best_ask}")
                    product_orders.append(Order(best_ask, -current_position))
                
                # Check profit-taking
                elif current_position > 0 and mid_price > rolling_avg * (1 + self.profit_take_threshold):
                    print(f"Profit take triggered: SELL {current_position} SQUID_INK at {best_bid}")
                    product_orders.append(Order(best_bid, -current_position))
                elif current_position < 0 and mid_price < rolling_avg * (1 - self.profit_take_threshold):
                    print(f"Profit take triggered: BUY {-current_position} SQUID_INK at {best_ask}")
                    product_orders.append(Order(best_ask, -current_position))
                
                # Make trading decisions based on deviation
                elif deviation < -deviation_threshold and current_position < self.position_limit:
                    # Price is below average - BUY
                    if order_size > 0:
                        print(f"BUY {order_size} SQUID_INK at {best_ask} (below average)")
                        product_orders.append(Order(best_ask, order_size))
                
                elif deviation > deviation_threshold and current_position > -self.position_limit:
                    # Price is above average - SELL
                    if order_size > 0:
                        print(f"SELL {order_size} SQUID_INK at {best_bid} (above average)")
                        product_orders.append(Order(best_bid, -order_size))
        
        # Add orders for the product
        if product_orders:
            orders[self.product] = product_orders
        
        # Update trader data for next iteration
        trader_data["price_history"][self.product] = price_history
            
        # Return result
        return orders, conversions, dumps(trader_data)
