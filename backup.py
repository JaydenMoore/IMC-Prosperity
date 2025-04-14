import numpy as np
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List

class Trader:
    def __init__(self,
                 fair_value_method: str = "MA",   # Method to estimate fair value: "MA" (Moving Average) or "VWAP" (Volume Weighted Average Price).
                 ma_length: int = 50,             # Window size (number of ticks) for calculating the Moving Average fair value. Longer windows smooth more but react slower.
                 vwap_depth: int = 30,            # Number of price levels on each side (bid/ask) to consider when calculating VWAP fair value.
                 penny_offset: int = 10,           # How many price ticks away from the best bid/ask to place passive limit orders (e.g., 1 means place buy at best_ask - 1).
                 soft_limit_factor: float = 0.8,  # A factor applied to scale down order volume, potentially used as position approaches limit (currently applied multiplicatively with risk scaling).
                 risk_vol_threshold: float = 4.0, # The volatility (standard deviation of recent mid-prices) threshold. If current volatility exceeds this, order volume is scaled down.
                 position_limit: int = 20,        # The maximum absolute number of units (long or short) the trader is allowed to hold for any single product.
                 max_volume: int = 2,             # The maximum number of units to trade in a single order.
                 arb_margin: float = 0.9,         # Placeholder: A potential threshold for triggering arbitrage trades between correlated assets (not currently implemented).
                 dead_zone: float = 2.5           # The minimum absolute difference required between the current mid-price and the estimated fair value to consider placing a trade. Helps filter out noise.
                 ):
        self.fair_value_method = fair_value_method
        self.ma_length = ma_length
        self.vwap_depth = vwap_depth
        self.penny_offset = penny_offset
        self.soft_limit_factor = soft_limit_factor
        self.risk_vol_threshold = risk_vol_threshold
        self.position_limit = position_limit
        self.max_volume = max_volume
        self.arb_margin = arb_margin
        self.dead_zone = dead_zone

        self.price_history: Dict[str, List[float]] = {}
        self.volatility: Dict[str, float] = {}
        # Cache precomputed best bid/ask to lower latency
        self.quote_cache: Dict[str, Dict[str, float]] = {}  # {product: {'best_bid':..., 'best_ask':...}}

    def estimate_fair_value(self, product: str, order_depth: OrderDepth) -> float:
        # For VWAP, we'll use the top vwap_depth of both sides.
        if self.fair_value_method.upper() == "VWAP":
            # Use combined orders: weighted average of bid and ask prices
            bid_prices = list(order_depth.buy_orders.keys())[:self.vwap_depth]
            ask_prices = list(order_depth.sell_orders.keys())[:self.vwap_depth]
            if len(bid_prices)==0 or len(ask_prices)==0:
                return 0
            bid_vols = np.array([order_depth.buy_orders[p] for p in bid_prices])
            ask_vols = np.array([order_depth.sell_orders[p] for p in ask_prices])
            # VWAP based on mid-price of each side:
            bid_vwap = np.sum(np.array(bid_prices)*bid_vols) / np.sum(bid_vols)
            ask_vwap = np.sum(np.array(ask_prices)*ask_vols) / np.sum(ask_vols)
            return (bid_vwap + ask_vwap) / 2
        else:
            # Use moving average from price history.
            if product in self.price_history and len(self.price_history[product]) >= self.ma_length:
                return np.mean(self.price_history[product][-self.ma_length:])
            else:
                return 0

    def update_volatility(self, product: str, window: int) -> float:
        if product in self.price_history and len(self.price_history[product]) >= window:
            vol = np.std(self.price_history[product][-window:])
            self.volatility[product] = vol
            return vol
        return 0

    def compute_order_volume(self, base_volume: int, available: int, current_vol: float, risk_vol: float) -> int:
        # Scale volume down if current volatility is high relative to risk_vol threshold.
        vol_scale = 1.0
        if risk_vol > 0 and current_vol > risk_vol:
            vol_scale = risk_vol / current_vol
        return min(base_volume, available, max(1, int(base_volume * vol_scale * self.soft_limit_factor)))

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            orders = []
            position = state.position.get(product, 0)
            # Set default limits if no product‐specific ones are provided.
            pos_limit = self.position_limit 
            max_vol = self.max_volume

            # Precompute best bid/ask and cache them.
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            if best_bid is None or best_ask is None or best_ask <= best_bid:
                result[product] = orders
                continue
            self.quote_cache[product] = {'best_bid': best_bid, 'best_ask': best_ask}

            mid_price = (best_bid + best_ask) / 2

            # Update price history for this product.
            if product not in self.price_history:
                self.price_history[product] = []
            self.price_history[product].append(mid_price)
            self.price_history[product] = self.price_history[product][-max(self.ma_length, self.vwap_depth):]

            # Estimate fair value.
            fair_value = self.estimate_fair_value(product, order_depth)
            if fair_value == 0:
                fair_value = mid_price  # fallback

            # Get volatility.
            current_vol = self.update_volatility(product, max(self.ma_length, self.vwap_depth))
            # NEW: Skip trading if volatility is too high (more than 20% above risk_vol_threshold)
            if current_vol > self.risk_vol_threshold * 1.2:
                result[product] = orders
                continue

            # Calculate risk-adjusted order scaling factor (if volatility too high, reduce order size)
            risk_scale = 1.0
            if self.risk_vol_threshold > 0 and current_vol > self.risk_vol_threshold:
                risk_scale = self.risk_vol_threshold / current_vol

            # Calculate difference between mid_price and fair value.
            diff = mid_price - fair_value

            # NEW: Use product-specific dead_zone if defined (values slightly increased)
            custom_dead_zone = {
                "PICNIC_BASKET1": 0.75, # Increased from 0.5
                "PICNIC_BASKET2": 0.75, # Increased from 0.5
                "KELP": 0.75,             # Increased from 0.5
                "VOLCANIC_ROCK": 0.75,    # Increased from 0.5
                "VOLCANIC_ROCK_VOUCHER_10000": 0.75, # Increased from 0.5
                "VOLCANIC_ROCK_VOUCHER_10250": 0.75, # Increased from 0.5
                "VOLCANIC_ROCK_VOUCHER_10500": 0.75, # Increased from 0.5
                "VOLCANIC_ROCK_VOUCHER_9500": 0.75,  # Increased from 0.5
                "VOLCANIC_ROCK_VOUCHER_9750": 0.75   # Increased from 0.5
            }
            current_dead_zone = custom_dead_zone.get(product, self.dead_zone)
            
            # NEW: Only trade if the difference magnitude exceeds the current_dead_zone.
            if abs(diff) < current_dead_zone:
                result[product] = orders
                continue

            # Determine base order volume.
            base_volume = max_vol

            # Apply pennying: if diff is positive, we want to sell at a price just better than bid; if negative, buy at just better than ask.
            if diff > 0:
                # Fair value is below mid_price -> signal to sell.
                volume = self.compute_order_volume(base_volume, pos_limit + position, current_vol, self.risk_vol_threshold)
                volume = max(1, int(volume * risk_scale))
                order_price = best_bid + self.penny_offset
                orders.append(Order(product, order_price, -volume))
            elif diff < 0:
                # Fair value above mid_price -> signal to buy.
                volume = self.compute_order_volume(base_volume, pos_limit - position, current_vol, self.risk_vol_threshold)
                volume = max(1, int(volume * risk_scale))
                order_price = best_ask - self.penny_offset
                orders.append(Order(product, order_price, volume))
            # --- Arbitrage (placeholder) ---
            # Here you would check for arbitrage opportunities between correlated assets.
            # For example, if product is an ETF and its constituents show mispricing,
            # you could place matching orders. This is left as a stub.
            # if product in correlated_assets and mispricing_detected:
            #     orders.append(...)
            result[product] = orders

        return result, 0, ""