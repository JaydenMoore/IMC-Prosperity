from typing import Dict, List
import json
import jsonpickle
import math
import statistics
from modules import Symbol, Trade, TradingState

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
    
    return statistics.mean(prices)

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
            return jsonpickle.decode(trader_data_str)
        except:
            print("Error decoding trader data, initializing new data")
    
    # Default structure if no data or error
    return {
        "price_history": {},
        "metrics": {}
    }